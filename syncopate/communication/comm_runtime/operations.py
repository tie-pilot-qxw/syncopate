from typing import Dict, List, Optional, Tuple
import torch
from syncopate.communication.descriptor import AxisSlice, CollectiveOp, ReduceOp, SignalType


class CopyEngineOp:
    """An operation to be executed by the copy engine."""
    src: str
    dst: str
    src_rank: int
    dst_rank: int
    src_offset: int
    dst_offset: int
    size: int

    def __str__(self) -> str:
        return (f"CopyEngineOp(src_rank={self.src_rank}, dst_rank={self.dst_rank}, "
                f"src_offset={self.src_offset}, dst_offset={self.dst_offset}, size={self.size})")

class SMCopyOp:
    """use TMA to perform the device to device copy."""
    src: str
    dst: str
    src_rank: int
    dst_rank: int
    src_slice: Tuple[AxisSlice, ...]
    dst_slice: Tuple[AxisSlice, ...]
    reduce_op: Optional[ReduceOp] = None # TMA can also do reduction
    use_tma: bool = True

    def __str__(self) -> str:
        return (f"SMCopyOp(src_rank={self.src_rank}, dst_rank={self.dst_rank}, "
                f"src_slice={self.src_slice}, dst_slice={self.dst_slice}, reduce_op={self.reduce_op})")
    
class CollectiveComm:
    op: CollectiveOp
    src: str
    dst: str
    src_slice: Tuple[AxisSlice, ...]
    dst_slice: Tuple[AxisSlice, ...]

    def __str__(self) -> str:
        return (f"CollectiveComm(op={self.op}, src={self.src}, dst={self.dst}, "
                f"src_slice={self.src_slice}, dst_slice={self.dst_slice})")

class IntraNodeSignal:
    """An intra-node signal operation."""
    signal_type: SignalType
    offset: int
    rank: int

    def __str__(self) -> str:
        return (f"IntraNodeSignal(type={self.signal_type}, offset={self.offset}, rank={self.rank})")

class WaitIntraNodeSignal:
    """A wait local signal operation."""
    signal_type: SignalType
    offset: int
    rank: int

    def __str__(self) -> str:
        return (f"WaitIntraNodeSignal(type={self.signal_type}, offset={self.offset}, rank={self.rank})")
class CommInfo:
    """Information to execute communication on a device."""
    world_size: int
    local_world_size: int
    recv_signal_num: int
    transit_signal_num: int
    compute_signal_num: int
    comm_buffer_sizes: Dict[str, Tuple[torch.Size, torch.dtype]] # buffer name -> (size, dtype)
    operations: List[List[List]] # [rank, stream, ops]
    num_copy_sms: int
    need_green_ctx: bool

    def __init__(self) -> None:
        self.world_size = 0
        self.local_world_size = 0
        self.recv_signal_num = 0
        self.transit_signal_num = 0
        self.comm_buffer_sizes = {}
        self.operations = []
        self.num_copy_sms = 0
        self.need_green_ctx = False

    def __str__(self) -> str:
        lines = [
            f"world_size={self.world_size}",
            f"local_world_size={self.local_world_size}",
            f"recv_signal_num={self.recv_signal_num}",
            f"transit_signal_num={self.transit_signal_num}",
            f"compute_signal_num={self.compute_signal_num}"
        ]
        if self.comm_buffer_sizes:
            lines.append("comm_buffers:")
            for name, (shape, dtype) in self.comm_buffer_sizes.items():
                lines.append(f"  {name}: shape={tuple(shape)} dtype={dtype}")
        else:
            lines.append("comm_buffers: none")

        for rank, streams in enumerate(self.operations):
            lines.append(f"rank {rank}:")
            for stream_idx, ops in enumerate(streams):
                lines.append(f"  stream {stream_idx}:")
                for op in ops:
                    lines.append(f"    {op}")
        return "\n".join(lines)
