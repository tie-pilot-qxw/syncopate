
import os
from typing import List, Dict, Optional

import torch
from syncopate.communication.comm_runtime.non_tma_persistent_copy import persistent_naive_copy_last_contig
from syncopate.communication.comm_runtime.operations import CollectiveComm, CommInfo, CopyEngineOp, IntraNodeSignal, SMCopyOp, WaitIntraNodeSignal
from syncopate.communication.descriptor import CollectiveOp, SignalType
from syncopate.communication.comm_runtime.all_reduce_collective import NVSHMEMAllReduceCollective
from triton_dist.utils import nvshmem_create_tensors, NVSHMEM_SIGNAL_DTYPE, CUDA_CHECK, nvshmem_free_tensor_sync, dist_print
from triton_dist.kernels.nvidia.common_ops import _set_signal_cuda, _wait_eq_cuda
from syncopate.communication.comm_runtime.tma_persistent_copy import persistent_tma_copy_last_contig
from cuda import cuda, cudart

class CommContext:
    """Context for communication code generation."""
    rank: int
    local_rank: int
    node_id: int
    info: CommInfo


    recv_signal_bufs: List[torch.Tensor] # [local_world_size]
    transit_signal_bufs: List[torch.Tensor] # [local_world_size]
    compute_signal_bufs: List[torch.Tensor] # [local_world_size]
    comm_buffers: Dict[str, List[torch.Tensor]] # buffer name -> [local_world_size] tensors

    streams: List[torch.cuda.Stream]
    compute_stream: Optional[torch.cuda.Stream]

    def __init__(self, rank: int, info: CommInfo, multi_info: bool = False):
        self.rank = rank
        self.local_rank = rank % info.local_world_size
        self.node_id = rank // info.local_world_size
        self.info = info

        self.ar_collective: Optional[NVSHMEMAllReduceCollective] = None
        if info.recv_signal_num > 0:
            num_recv_signals = info.recv_signal_num
            if multi_info:
                num_recv_signals = num_recv_signals + 100 # to leave enough space for new infos, this is because we don't know why creating two CommContext will lead to internal error in nvshmem
            self.recv_signal_bufs = nvshmem_create_tensors((num_recv_signals,), NVSHMEM_SIGNAL_DTYPE, rank, info.local_world_size)
        else:
            self.recv_signal_bufs = []
        if info.transit_signal_num > 0:
            num_transit_signals = info.transit_signal_num
            if multi_info:
                num_transit_signals = num_transit_signals + 100 # to leave enough space for new infos
            self.transit_signal_bufs = nvshmem_create_tensors((num_transit_signals,), NVSHMEM_SIGNAL_DTYPE, rank, info.local_world_size)
        else:
            self.transit_signal_bufs = []
        if info.compute_signal_num > 0:
            num_compute_signals = info.compute_signal_num
            if multi_info:
                num_compute_signals = num_compute_signals + 100 # to leave enough space for new infos
            self.compute_signal_bufs = nvshmem_create_tensors((num_compute_signals,), NVSHMEM_SIGNAL_DTYPE, rank, info.local_world_size)
        else:
            self.compute_signal_bufs = []

        self.comm_buffers = {}
        for buf_name, (buf_size, buf_dtype) in info.comm_buffer_sizes.items():
            self.comm_buffers[buf_name] = nvshmem_create_tensors(buf_size, buf_dtype, rank, info.local_world_size)

        self.streams = []

        if info.need_green_ctx:
            assert len(info.operations[self.rank]) == 1, "currently only support one communication stream when using green context"
            from syncopate.communication.comm_runtime.split_green_ctx import split_device_green_ctx
            dev = torch.device(f"cuda:{torch.cuda.current_device()}")
            green_streams, _ = split_device_green_ctx(dev, 1, info.num_copy_sms)
            self.streams.append(green_streams[0])
            self.compute_stream = green_streams[1]
        else:
            for _ in range(len(info.operations[self.rank])):
                self.streams.append(torch.cuda.Stream())
            self.compute_stream = None

    def __del__(self):
        if self.info.recv_signal_num > 0:
            nvshmem_free_tensor_sync(self.recv_signal_bufs[self.local_rank])
        if self.info.transit_signal_num > 0:
            nvshmem_free_tensor_sync(self.transit_signal_bufs[self.local_rank])
        if self.info.compute_signal_num > 0:
            nvshmem_free_tensor_sync(self.compute_signal_bufs[self.local_rank])
        for buf_list in self.comm_buffers.values():
            nvshmem_free_tensor_sync(buf_list[self.local_rank])
        if self.ar_collective is not None:
            self.ar_collective.finalize()

    def reset_signals(self):
        if len(self.recv_signal_bufs) > 0:
            self.recv_signal_bufs[self.local_rank].fill_(0)
        if len(self.transit_signal_bufs) > 0:
            self.transit_signal_bufs[self.local_rank].fill_(0)
        if len(self.compute_signal_bufs) > 0:
            self.compute_signal_bufs[self.local_rank].fill_(0)

    def start_after(self, stream: torch.cuda.Stream = torch.cuda.current_stream()):
        """Make all communication streams wait for the given stream."""
        for comm_stream in self.streams:
            comm_stream.wait_stream(stream)

    def end_before(self, stream: torch.cuda.Stream = torch.cuda.current_stream()):
        """Make all communication streams wait on the given stream."""
        for comm_stream in self.streams:
            stream.wait_stream(comm_stream)

    def execute(self, neglect_local: bool = False):
        """Execute the communication operations for this rank."""
        for stream_idx, ops in enumerate(self.info.operations[self.rank]):
            stream = self.streams[stream_idx]
            with torch.cuda.stream(stream):
                for op in ops:
                    if isinstance(op, CopyEngineOp):
                        assert op.src_rank == self.rank or op.dst_rank == self.rank
                        assert op.src_rank // self.info.local_world_size == op.dst_rank // self.info.local_world_size
                        
                        src_local_rank = op.src_rank % self.info.local_world_size
                        dst_local_rank = op.dst_rank % self.info.local_world_size
                        src_buf = self.comm_buffers[op.src][src_local_rank]
                        dst_buf = self.comm_buffers[op.dst][dst_local_rank]
                        assert src_buf.is_contiguous()
                        assert dst_buf.is_contiguous()

                        if op.src_rank == op.dst_rank:
                            if neglect_local:
                                continue
                            # local copy
                            (err, ) = cudart.cudaMemcpyAsync(
                                dst_buf.data_ptr() + op.dst_offset,
                                src_buf.data_ptr() + op.src_offset,
                                op.size,
                                cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                                stream.cuda_stream,
                            )
                        else:
                            (err, ) = cudart.cudaMemcpyAsync(
                                dst_buf.data_ptr() + op.dst_offset,
                                src_buf.data_ptr() + op.src_offset,
                                op.size,
                                cudart.cudaMemcpyKind.cudaMemcpyDefault,
                                stream.cuda_stream,
                            )
                        CUDA_CHECK(err)
                    elif isinstance(op, IntraNodeSignal):
                        assert op.signal_type == SignalType.RECEIVE_DATA
                        assert op.rank // self.info.local_world_size == self.node_id
                        local_target_rank = op.rank % self.info.local_world_size
                        signal_buf = self.recv_signal_bufs[local_target_rank]
                        _set_signal_cuda(signal_buf[op.offset], 1, stream)
                    elif isinstance(op, WaitIntraNodeSignal):
                        assert op.rank // self.info.local_world_size == self.node_id
                        local_target_rank = op.rank % self.info.local_world_size
                        if op.signal_type == SignalType.COMPUTE_DONE:
                            signal_buf = self.compute_signal_bufs[local_target_rank]
                        elif op.signal_type == SignalType.RECEIVE_DATA:
                            signal_buf = self.recv_signal_bufs[local_target_rank]
                        else:
                            raise NotImplementedError(f"Signal type {op.signal_type} not implemented yet.")
                        _wait_eq_cuda(signal_buf[op.offset], 1, stream)
                    elif isinstance(op, SMCopyOp):
                        assert op.src_rank == self.rank or op.dst_rank == self.rank
                        assert op.src_rank // self.info.local_world_size == op.dst_rank // self.info.local_world_size
                        assert self.info.num_copy_sms > 0

                        if neglect_local and op.src_rank == op.dst_rank:
                            continue

                        src_local_rank = op.src_rank % self.info.local_world_size
                        dst_local_rank = op.dst_rank % self.info.local_world_size
                        src_buf = self.comm_buffers[op.src][src_local_rank]
                        dst_buf = self.comm_buffers[op.dst][dst_local_rank]
                        src_slice = src_buf[tuple(slice(axis.start, axis.end) for axis in op.src_slice)]
                        dst_slice = dst_buf[tuple(slice(axis.start, axis.end) for axis in op.dst_slice)]

                        if op.use_tma:
                            persistent_tma_copy_last_contig(
                                src=src_slice,
                                dst=dst_slice,
                                num_sms = self.info.num_copy_sms,
                                reduce_op = op.reduce_op
                            )
                        else:
                            persistent_naive_copy_last_contig(
                                src=src_slice,
                                dst=dst_slice,
                                num_sms = self.info.num_copy_sms,
                                reduce_op = op.reduce_op
                            )
                    elif isinstance(op, CollectiveComm):
                        src_buf = self.comm_buffers[op.src][self.local_rank]
                        dst_buf = self.comm_buffers[op.dst][self.local_rank]

                        src_idx = tuple(slice(axis.start, axis.end) for axis in op.src_slice)
                        dst_idx = tuple(slice(axis.start, axis.end) for axis in op.dst_slice)

                        src_slice = src_buf[src_idx]
                        dst_slice = dst_buf[dst_idx]

                        if op.op == CollectiveOp.REDUCE_SCATTER:
                            if not src_slice.is_contiguous():
                                src_buf = torch.empty_like(src_slice)
                                src_buf.copy_(src_slice)
                            else:
                                src_buf = src_slice

                            if not dst_slice.is_contiguous():
                                dst_buf = torch.empty_like(dst_slice)
                                torch.distributed.reduce_scatter_tensor(dst_buf, src_buf)
                                dst_slice.copy_(dst_buf)
                            else:
                                torch.distributed.reduce_scatter_tensor(dst_slice, src_buf)
                        elif op.op == CollectiveOp.ALL_REDUCE:
                            assert src_idx == dst_idx

                            if op.src != op.dst:
                                raise ValueError("NCCL all-reduce currently requires src and dst to be the same buffer.")
                            if not src_slice.is_contiguous():
                                src_buf = torch.empty_like(src_slice)
                                src_buf.copy_(src_slice)
                            else:
                                src_buf = src_slice

                            torch.distributed.all_reduce(src_buf)
                            
                            if not src_slice.is_contiguous():
                                dst_slice.copy_(src_buf)
                        elif op.op == CollectiveOp.ALL_REDUCE_NVSHARP:
                            if op.src == op.dst:
                                raise ValueError("NVSHMEM all-reduce requires different src and dst buffers.")
                            used_nvshmem = self._try_nvshmem_all_reduce(src_slice, dst_slice, stream)
                            if not used_nvshmem:
                                raise ValueError("NVSHMEM all-reduce failed, falling back to NCCL is not supported for now.")
                        else:
                            raise NotImplementedError(f"Collective operation {op.op} not implemented yet.")

                        
    def get_comm_sms(self) -> int:
        """Get the number of SMS used for communication."""
        return self.info.num_copy_sms
    
    def get_graph(self) -> torch.cuda.CUDAGraph:
        """Capture the communication operations into a CUDA graph."""
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.start_after(torch.cuda.current_stream())
            self.execute()
            self.end_before(torch.cuda.current_stream())
        return graph

    def update_comm_info(self, info: CommInfo):
        """Update the communication info and reallocate buffers if needed."""
        self.info = info
        # assume all buffers are the same size for simplicity
        self.streams = []

        if info.need_green_ctx:
            assert len(info.operations[self.rank]) == 1, "currently only support one communication stream when using green context"
            from syncopate.communication.comm_runtime.split_green_ctx import split_device_green_ctx
            dev = torch.device(f"cuda:{torch.cuda.current_device()}")
            green_streams, _ = split_device_green_ctx(dev, 1, info.num_copy_sms)
            self.streams.append(green_streams[0])
            self.compute_stream = green_streams[1]
        else:
            for _ in range(len(info.operations[self.rank])):
                self.streams.append(torch.cuda.Stream())
            self.compute_stream = None

        if self.ar_collective is not None and self.ar_collective.num_comm_sms != info.num_copy_sms and info.num_copy_sms > 0:
            self.ar_collective.finalize()
            self.ar_collective = None

    def _init_all_reduce_collective(self) -> Optional[NVSHMEMAllReduceCollective]:
        if self.info.world_size != self.info.local_world_size:
            dist_print("NVSHMEM all-reduce currently requires world_size == local_world_size, falling back to NCCL.", allowed_ranks="all", need_sync=False)
            self.use_nvshmem_all_reduce = False
            return None
        if self.ar_collective is None:
            num_sms = self.info.num_copy_sms if self.info.num_copy_sms > 0 else 8
            block_m = 128
            block_n = 256
            use_multimem_st = True
            self.ar_collective = NVSHMEMAllReduceCollective(
                self.info.world_size,
                self.info.local_world_size,
                num_sms,
                (block_m, block_n),
                use_multimem_st=use_multimem_st,
            )
        return self.ar_collective

    def _try_nvshmem_all_reduce(self, src_tensor: torch.Tensor, dst_tensor: torch.Tensor, stream: torch.cuda.Stream) -> bool:
        collective = self._init_all_reduce_collective()
        if collective is None:
            return False
        if not src_tensor.is_contiguous() or not dst_tensor.is_contiguous():
            return False
        if src_tensor.dtype not in [torch.float16, torch.bfloat16, torch.float32]:
            return False
        try:
            collective.all_reduce_(src_tensor, dst_tensor, stream)
            return True
        except ValueError:
            return False

        
