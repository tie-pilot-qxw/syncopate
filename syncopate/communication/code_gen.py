from dataclasses import dataclass
from typing import List, Tuple

from syncopate.communication.comm_runtime.operations import CollectiveComm, CommInfo, CopyEngineOp, IntraNodeSignal, SMCopyOp, WaitIntraNodeSignal
from .descriptor import AxisSlice, Collective, CollectiveOp, CommPlan, ComputeDependency, SignalType, Transfer, TransferOp, XferDependency

@dataclass
class CodeGenOptions:
    copy_engine: bool = True
    low_latency: bool = False
    use_tma: bool = True
    fuse_with_compute_intra: bool = False
    fuse_with_compute_inter: bool = False

    def __init__(self, copy_engine: bool = True, low_latency: bool = False, use_tma: bool = True, fuse_with_compute_intra: bool = False, fuse_with_compute_inter: bool = False):
        self.copy_engine = copy_engine
        self.low_latency = low_latency
        self.use_tma = use_tma
        self.fuse_with_compute_intra = fuse_with_compute_intra
        self.fuse_with_compute_inter = fuse_with_compute_inter

    def validate(self):
        if self.low_latency:
            raise NotImplementedError("low_latency codegen is not implemented yet")
        
        if self.copy_engine and (self.fuse_with_compute_intra):
            raise ValueError("copy_engine and fuse_with_compute_intra cannot be both enabled")
        
        if self.fuse_with_compute_inter and not self.fuse_with_compute_intra:
            raise ValueError("fuse_with_compute_inter requires fuse_with_compute_intra to be enabled")

class CommGenerator(CommPlan):
    """Generate code for NVIDIA GPUs using Triton-Distributed."""
    options: CodeGenOptions

    def generate_code_for_plan(self, options: CodeGenOptions = CodeGenOptions()) -> CommInfo:
        """Generate triton-distributed code for the given communication plan."""
        options.validate()

        result = CommInfo()

        result.world_size = len(self.device_plans)
        receive_signals, transit_signals, compute_signals, comm_buffer_sizes = self._get_buffer_info()
        result.recv_signal_num = receive_signals
        result.transit_signal_num = transit_signals
        result.compute_signal_num = compute_signals
        result.comm_buffer_sizes = comm_buffer_sizes

        def _with_base_offset(region):
            if region.base_offset == 0:
                shifts = [0] * len(region.strides)
            else:
                if region.base_offset % region.elem_size != 0:
                    raise ValueError("base_offset must align with elem_size for TMA copy")
                remaining = region.base_offset // region.elem_size
                shifts = []
                for stride in region.strides:
                    step = remaining // stride
                    shifts.append(step)
                    remaining -= step * stride
                if remaining != 0:
                    raise ValueError("base_offset not representable by strides for TMA copy")
            adjusted = []
            for axes in region.regions:
                adjusted.append(tuple(AxisSlice(axis.start + shifts[i], axis.end + shifts[i]) for i, axis in enumerate(axes)))
            return adjusted


        # first we only generate intra-node communication code
        for dev, dev_plan in self.device_plans.items():
            result.operations.append([])  # one entry per device
            for transfers_stream in dev_plan.xfer:
                result.operations[dev].append([])
                for transfer in transfers_stream:
                    # resolve dependency first
                    for dep in transfer.dependency:
                        if isinstance(dep, ComputeDependency):
                            wait_op = WaitIntraNodeSignal()
                            wait_op.signal_type = SignalType.COMPUTE_DONE
                            assert dep.signal is not None
                            wait_op.offset = dep.signal.dst_offset
                            # assert dep.signal.dst_rank == dev_plan.dev # wait local
                            wait_op.rank = dep.signal.dst_rank
                            result.operations[dev][-1].append(wait_op)
                        elif isinstance(dep, XferDependency):
                            if dep.signal is None:
                                raise ValueError("XferDependency requires planned signals before codegen")
                            wait_op = WaitIntraNodeSignal()
                            wait_op.signal_type = dep.signal.type
                            wait_op.offset = dep.signal.dst_offset
                            wait_op.rank = dep.signal.dst_rank
                            result.operations[dev][-1].append(wait_op)
                        else:
                            raise NotImplementedError("dependency handling is not implemented yet")
                    
                    if isinstance(transfer, Collective):
                        if transfer.src_region is None or transfer.dst_region is None:
                            raise ValueError("Collective requires materialized regions.")
                        if transfer.op == CollectiveOp.ALL_REDUCE and transfer.src_buf != transfer.dst_buf:
                            raise ValueError("ALL_REDUCE must use the same buffer for src and dst.")

                        src_slices = _with_base_offset(transfer.src_region)
                        dst_slices = _with_base_offset(transfer.dst_region)
                        if len(src_slices) != len(dst_slices):
                            raise ValueError("source and destination region counts differ for collective")

                        for src_slice, dst_slice in zip(src_slices, dst_slices):
                            collective_op = CollectiveComm()
                            collective_op.op = transfer.op
                            collective_op.src = transfer.src_buf
                            collective_op.dst = transfer.dst_buf
                            collective_op.src_slice = src_slice
                            collective_op.dst_slice = dst_slice
                            result.operations[dev][-1].append(collective_op)
                    elif isinstance(transfer, Transfer):
                        if options.copy_engine:
                            if transfer.reduce_op is not None:
                                raise ValueError("Copy engine transfer cannot do reduction.")
                            
                            copys = transfer.to_memcpys()
                            for copy in copys:
                                copy_op = CopyEngineOp()
                                if transfer.op == TransferOp.LOCAL_COPY:
                                    copy_op.src_rank = dev_plan.dev
                                    copy_op.dst_rank = dev_plan.dev
                                elif transfer.op == TransferOp.PUSH:
                                    copy_op.src_rank = dev_plan.dev
                                    assert transfer.peer is not None
                                    copy_op.dst_rank = transfer.peer
                                else:  # PULL
                                    assert transfer.peer is not None
                                    copy_op.src_rank = transfer.peer
                                    copy_op.dst_rank = dev_plan.dev

                                copy_op.src = transfer.src_buf
                                copy_op.dst = transfer.dst_buf
                                copy_op.src_offset = copy[0]
                                copy_op.dst_offset = copy[1]
                                copy_op.size = copy[2]
                                result.operations[dev][-1].append(copy_op)
                        else:
                            assert transfer.src_region is not None and transfer.dst_region is not None

                            src_slices = _with_base_offset(transfer.src_region)
                            dst_slices = _with_base_offset(transfer.dst_region)
                            if len(src_slices) != len(dst_slices):
                                raise ValueError("source and destination region counts differ for TMA copy")

                            for src_slice, dst_slice in zip(src_slices, dst_slices):
                                copy_op = SMCopyOp()
                                copy_op.use_tma = options.use_tma
                                if transfer.op == TransferOp.LOCAL_COPY:
                                    copy_op.src_rank = dev_plan.dev
                                    copy_op.dst_rank = dev_plan.dev
                                elif transfer.op == TransferOp.PUSH:
                                    copy_op.src_rank = dev_plan.dev
                                    assert transfer.peer is not None
                                    copy_op.dst_rank = transfer.peer
                                else:
                                    assert transfer.peer is not None
                                    copy_op.src_rank = transfer.peer
                                    copy_op.dst_rank = dev_plan.dev

                                copy_op.src = transfer.src_buf
                                copy_op.dst = transfer.dst_buf
                                copy_op.src_slice = src_slice
                                copy_op.dst_slice = dst_slice
                                copy_op.reduce_op = transfer.reduce_op
                                result.operations[dev][-1].append(copy_op)

                    for signal in transfer.set_signals:
                        # currently only handle intra node signals
                        # so we can directly use `cuStreamWriteValue`
                        assert signal.type == SignalType.RECEIVE_DATA
                        signal_op = IntraNodeSignal()
                        signal_op.signal_type = signal.type
                        signal_op.offset = signal.dst_offset
                        signal_op.rank = signal.dst_rank
                        result.operations[dev][-1].append(signal_op)

        return result

    
    def _get_buffer_info(self) -> Tuple[int, int, int, dict]:
        """Generate initialization code for the communication plan."""
        receive_signals = max(self.receive_signal_counters.values()) if self.receive_signal_counters else 0
        transit_signals = max(self.transit_signal_counters.values()) if self.transit_signal_counters else 0
        compute_signals = max(self.compute_signal_counters.values()) if self.compute_signal_counters else 0
        comm_buffer_sizes = {}
        for dev_plan in self.device_plans.values():
            for buf_name, (tensor_size, tensor_dtype) in dev_plan.tensors_involved.items():
                if buf_name not in comm_buffer_sizes:
                    comm_buffer_sizes[buf_name] = (tensor_size, tensor_dtype)
                else:
                    # ensure the sizes and dtypes are consistent
                    existing_size, existing_dtype = comm_buffer_sizes[buf_name]
                    if existing_size != tensor_size or existing_dtype != tensor_dtype:
                        raise ValueError(f"Buffer {buf_name} has inconsistent size or dtype across devices.")
        return (receive_signals, transit_signals, compute_signals, comm_buffer_sizes)                    

    def _generate_nic_code(self):
        """Generate code for NIC-based communication."""
        pass
    
