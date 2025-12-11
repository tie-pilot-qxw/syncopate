"""Minimal IR definitions for communication plans."""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import triton
import torch

@dataclass
class Period:
    """A period representing a time interval"""
    start: Optional[float] = None # inferred in lowering
    duration: float = 1.0 # default 1.0 unit

    def split(self, num_splits: int) -> "Period":
        assert self.start is None, "Cannot split a Period with known start time."
        if num_splits <= 1:
            return self
        split_duration = self.duration / num_splits
        return Period(start=None, duration=split_duration)

@dataclass(frozen=True)
class MemorySegment:
    """Contiguous byte span within a logical buffer."""

    offset: int
    nbytes: int

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ValueError("segment offset must be non-negative")
        if self.nbytes < 0:
            raise ValueError("segment nbytes must be non-negative")

    @property
    def end(self) -> int:
        return self.offset + self.nbytes


@dataclass(frozen=True)
class AxisSlice:
    """Half-open range along a logical axis (start inclusive, end exclusive)."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("axis slice start must be non-negative")
        if self.end < self.start:
            raise ValueError("axis slice end must be >= start")

    def length(self) -> int:
        return self.end - self.start

# better to be replaced to list of faketensors to utilize pytorch's indexing logic
@dataclass
class BufferRegion:
    """Union of rectangular axis-aligned slices within a buffer."""

    regions: Tuple[Tuple[AxisSlice, ...], ...]
    elem_size: int
    strides: Tuple[int, ...]
    base_offset: int = 0
    _segments: Tuple[MemorySegment, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.regions:
            raise ValueError("BufferRegion requires at least one region")
        if self.elem_size <= 0:
            raise ValueError("elem_size must be positive")
        if self.base_offset < 0:
            raise ValueError("base_offset must be non-negative")

        axis_count = len(self.regions[0])
        if len(self.strides) != axis_count:
            raise ValueError("strides length must match number of axes")

        for region in self.regions:
            if len(region) != axis_count:
                raise ValueError("all regions must have the same number of axes")

        segments: List[MemorySegment] = []
        for region in self.regions:
            segments.extend(self._segments_for_region(region))

        segments.sort(key=lambda seg: seg.offset)
        self._segments = _merge_adjacent_segments(segments)

    @classmethod
    def contiguous(cls, offset: int, nbytes: int) -> "BufferRegion":
        """Build a region representing a single contiguous byte span."""

        return cls.from_segments([(offset, nbytes)])

    @classmethod
    def from_segments(
        cls, segments: Iterable[Union[MemorySegment, Tuple[int, int]]]
    ) -> "BufferRegion":
        normalized: List[MemorySegment] = []
        for seg in segments:
            if isinstance(seg, MemorySegment):
                normalized.append(seg)
            else:
                offset, nbytes = seg
                normalized.append(MemorySegment(offset, nbytes))
        if not normalized:
            raise ValueError("from_segments expects at least one segment")
        regions = tuple(
            (AxisSlice(seg.offset, seg.offset + seg.nbytes),)
            for seg in normalized
        )
        return cls(regions=regions, elem_size=1, strides=(1,), base_offset=0)

    @classmethod
    def from_slices(
        cls,
        *,
        elem_size: int,
        strides: Sequence[int],
        slices: Sequence[Union[AxisSlice, slice, Tuple[int, int]]],
        base_offset: int = 0,
    ) -> "BufferRegion":
        region = cls._normalize_region(slices)
        return cls(
            regions=(region,),
            elem_size=elem_size,
            strides=tuple(strides),
            base_offset=base_offset,
        )

    @classmethod
    def from_multi_slices(
        cls,
        *,
        elem_size: int,
        strides: Sequence[int],
        regions: Sequence[Sequence[Union[AxisSlice, slice, Tuple[int, int]]]],
        base_offset: int = 0,
    ) -> "BufferRegion":
        normalized = tuple(cls._normalize_region(region) for region in regions)
        if not normalized:
            raise ValueError("from_multi_slices expects at least one region")
        return cls(
            regions=normalized,
            elem_size=elem_size,
            strides=tuple(strides),
            base_offset=base_offset,
        )

    @classmethod
    def strided(
        cls, base_offset: int, segment_bytes: int, stride: int, count: int
    ) -> "BufferRegion":
        if count <= 0:
            raise ValueError("count must be positive for strided region")
        if segment_bytes <= 0:
            raise ValueError("segment_bytes must be positive")
        if stride < 0:
            raise ValueError("stride must be non-negative")
        segments = [
            MemorySegment(base_offset + i * stride, segment_bytes)
            for i in range(count)
        ]
        return cls.from_segments(segments)

    @staticmethod
    def _normalize_region(
        slices: Sequence[Union[AxisSlice, slice, Tuple[int, int]]]
    ) -> Tuple[AxisSlice, ...]:
        normalized: List[AxisSlice] = []
        for spec in slices:
            if isinstance(spec, AxisSlice):
                normalized.append(spec)
                continue
            if isinstance(spec, slice):
                if spec.step not in (None, 1):
                    raise ValueError("slice step other than 1 is not supported")
                start = 0 if spec.start is None else spec.start
                if spec.stop is None:
                    raise ValueError("slice stop must be provided")
                normalized.append(AxisSlice(start, spec.stop))
                continue
            start, end = spec
            normalized.append(AxisSlice(start, end))
        return tuple(normalized)

    def _segments_for_region(self, region: Tuple[AxisSlice, ...]) -> List[MemorySegment]:
        if any(axis.length() == 0 for axis in region):
            return []

        contiguous_bytes = self.elem_size
        split_axis = -1
        for axis_idx in range(len(region) - 1, -1, -1):
            stride_bytes = self.strides[axis_idx] * self.elem_size
            axis_len = region[axis_idx].length()
            if stride_bytes == contiguous_bytes:
                contiguous_bytes *= axis_len
            else:
                split_axis = axis_idx
                break

        if split_axis == -1:
            offset_elems = sum(
                region[idx].start * self.strides[idx]
                for idx in range(len(region))
            )
            offset_bytes = self.base_offset + offset_elems * self.elem_size
            return [MemorySegment(offset_bytes, contiguous_bytes)]

        trailing_offset = sum(
            region[idx].start * self.strides[idx] * self.elem_size
            for idx in range(split_axis + 1, len(region))
        )

        segments: List[MemorySegment] = []

        def recurse(axis_idx: int, accum_offset: int) -> None:
            axis = region[axis_idx]
            stride_bytes = self.strides[axis_idx] * self.elem_size
            for idx in range(axis.start, axis.end):
                next_offset = accum_offset + idx * stride_bytes
                if axis_idx == split_axis:
                    seg_offset = self.base_offset + next_offset + trailing_offset
                    segments.append(MemorySegment(seg_offset, contiguous_bytes))
                else:
                    recurse(axis_idx + 1, next_offset)

        recurse(0, 0)
        return segments

    def total_nbytes(self) -> int:
        return sum(seg.nbytes for seg in self._segments)

    def is_contiguous(self) -> bool:
        return len(self._segments) == 1

    def as_slice(self) -> Optional[slice]:
        if (
            len(self.regions) == 1
            and len(self.regions[0]) == 1
            and self.strides == (1,)
            and self.elem_size == 1
            and self.base_offset == 0
        ):
            axis = self.regions[0][0]
            return slice(axis.start, axis.end)
        return None

    def split(self, num_splits: int, axis: Optional[int] = None) -> List["BufferRegion"]:
        if num_splits <= 1:
            return [self]

        if axis is not None:
            return self._split_along_axis(num_splits, axis)

        return self._split_by_bytes(num_splits)

    def _split_by_bytes(self, num_splits: int) -> List["BufferRegion"]:
        total = self.total_nbytes()
        if total == 0:
            return [self]

        split_size = triton.cdiv(total, num_splits)
        remaining_total = total
        seg_idx = 0
        seg_consumed = 0
        res: List[BufferRegion] = []

        segments = list(self._segments)
        for _ in range(num_splits):
            target = min(split_size, remaining_total)
            if target <= 0:
                break

            new_segments: List[MemorySegment] = []
            while target > 0:
                if seg_idx >= len(segments):
                    raise RuntimeError("not enough data to satisfy split request")
                seg = segments[seg_idx]
                available = seg.nbytes - seg_consumed
                take = min(target, available)
                new_segments.append(
                    MemorySegment(seg.offset + seg_consumed, take)
                )

                target -= take
                remaining_total -= take
                seg_consumed += take

                if seg_consumed == seg.nbytes:
                    seg_idx += 1
                    seg_consumed = 0

            res.append(BufferRegion.from_segments(new_segments))

        return res

    def _split_along_axis(self, num_splits: int, axis: int) -> List["BufferRegion"]:
        if not self.regions:
            return [self]

        axis_count = len(self.strides)
        if axis < 0:
            axis += axis_count
        if not 0 <= axis < axis_count:
            raise IndexError(f"axis {axis} out of range for BufferRegion with {axis_count} dims")

        if len(self.regions) != 1:
            raise NotImplementedError(
                "Axis-aligned splitting currently supports regions with a single contiguous block"
            )

        region = list(self.regions[0])
        target_axis = region[axis]
        axis_len = target_axis.length()
        if axis_len == 0:
            return [self]

        chunk = triton.cdiv(axis_len, num_splits)
        if chunk <= 0:
            return [self]

        splits: List[BufferRegion] = []
        current = target_axis.start
        while current < target_axis.end and len(splits) < num_splits:
            end = min(target_axis.end, current + chunk)
            new_region = list(region)
            new_region[axis] = AxisSlice(current, end)
            splits.append(
                BufferRegion(
                    regions=(tuple(new_region),),
                    elem_size=self.elem_size,
                    strides=self.strides,
                    base_offset=self.base_offset,
                )
            )
            current = end

        return splits if splits else [self]

    def describe(self) -> str:
        def fmt_region(region: Tuple[AxisSlice, ...]) -> str:
            if not region:
                return "[]"
            axes = ", ".join(f"{axis.start}:{axis.end}" for axis in region)
            return f"[{axes}]"

        if len(self.regions) == 1:
            return fmt_region(self.regions[0])
        return "{" + ", ".join(fmt_region(region) for region in self.regions) + "}"

    def to_segments(self) -> Tuple[MemorySegment, ...]:
        return self._segments


def _merge_adjacent_segments(
    segments: List[MemorySegment],
) -> Tuple[MemorySegment, ...]:
    if not segments:
        return tuple()
    merged: List[MemorySegment] = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        if prev.end == seg.offset:
            merged[-1] = MemorySegment(prev.offset, prev.nbytes + seg.nbytes)
        else:
            merged.append(seg)
    return tuple(merged)

###### High-Level Communication Descriptor #######
@dataclass
class DataPlacement:
    """data placement info for data distribution"""
    rank: int
    region: Optional[BufferRegion] = None

@dataclass
class CommDescriptor:
    """A communication descriptor describes how to perform a collective communication operation.
    It only contains data info and does not contain any scheduling info.
    """
    data_info: dict[int, Optional[int]] # data_idx -> data_size (None for dynamic size)
    before_placement: dict[int, DataPlacement] # data_idx -> placement before communication
    after_placement: dict[int, DataPlacement] # data_idx -> placement after communication


###### Mid-Level Communication Plan #######
@dataclass
class XferDependency:
    """
    Represents a data transfer dependency between two transfers.
    e.g.: xfer1: A->B xfer2: B->C then xfer2 depends on xfer1
    """
    depend_rank: int
    depend_xfer_stream: int
    depend_xfer_idx: int
    signal: Optional[Signal] = None # After planning, this field should be filled with the actual signal

@dataclass
class ComputeDependency:
    compute_rank: int
    src_region: Optional[BufferRegion]
    signal: Optional[Signal] = None # After planning, this field should be filled with the actual signal

    def __post_init__(self):
        if self.signal is not None:
            assert self.signal.type == SignalType.COMPUTE_DONE
class TransferOp(Enum):
    PUSH = "push"
    PULL = "pull"
    LOCAL_COPY = "local_copy"

class SignalType(Enum):
    RECEIVE_DATA = "receive_data" # standard signal for data receivers
    COMPUTE_DONE = "compute_done" # signal to indicate compute is done
    REMOTE_DATA_READY = "remote_data_ready" # signal to indicate transfer is done (to non-receivers)

@dataclass
class Signal:
    """A signal represents a message after a transfer is done."""
    dst_rank: int # which rank will receive the signal
    dst_offset: int # offset in the dst_rank's signal buffer
    type: SignalType
    period: Period = field(default_factory=Period)

class ReduceOp(Enum):
    SUM = 1
    MAX = 2

@dataclass
class Transfer:
    """Class for pull/push/local copy operations."""
    op: TransferOp
    src_buf: str
    src_region: Optional[BufferRegion]
    dst_buf: str
    dst_region: Optional[BufferRegion]
    chunk_idx: int
    dependency: List[XferDependency | ComputeDependency] = field(default_factory=list) # used to represent complex dependency such as ring
    set_signals: List[Signal] = field(default_factory=list) # should be infered from dependency
    peer: Optional[int] = None  # only for push/pull
    period: Period = field(default_factory=Period)
    reduce_op: Optional[ReduceOp] = None # only for intra-node sm-based communication

    def __post_init__(self):
        if self.op == TransferOp.PULL or self.op == TransferOp.PUSH:
            assert self.peer is not None
        else:
            assert self.peer is None


    @property
    def nbytes(self) -> Optional[int]:
        if self.src_region is None or self.dst_region is None:
            return None
        if not self.src_region.is_contiguous() or not self.dst_region.is_contiguous():
            return None
        src_total = self.src_region.total_nbytes()
        dst_total = self.dst_region.total_nbytes()
        if src_total != dst_total:
            raise ValueError("source and destination regions disagree on total bytes")
        return src_total
    
    @property
    def src_off(self) -> Optional[int]:
        if self.src_region and self.src_region.is_contiguous():
            return self.src_region.to_segments()[0].offset
        return None

    @property
    def dst_off(self) -> Optional[int]:
        if self.dst_region and self.dst_region.is_contiguous():
            return self.dst_region.to_segments()[0].offset
        return None

    def split(self, num_splits: int) -> List[Transfer]:
        if num_splits <= 1:
            return [self]
        if self.src_region is None or self.dst_region is None:
            raise ValueError("Cannot split a Transfer with dynamic regions.")

        src_splits = self.src_region.split(num_splits)
        dst_splits = self.dst_region.split(num_splits)
        if len(src_splits) != len(dst_splits):
            raise ValueError("Source and destination splits differ in count")

        split_period = self.period.split(num_splits)
        splits: List[Transfer] = []
        for i, (src_part, dst_part) in enumerate(zip(src_splits, dst_splits)):
            # can't split after planning dependency and signals
            assert len(self.dependency) == 0
            assert len(self.set_signals) == 0
            splits.append(
                Transfer(
                    op=self.op,
                    src_buf=self.src_buf,
                    src_region=src_part,
                    dst_buf=self.dst_buf,
                    dst_region=dst_part,
                    chunk_idx=self.chunk_idx * num_splits + i,
                    dependency=copy.deepcopy(self.dependency),
                    peer=self.peer,
                    set_signals=copy.deepcopy(self.set_signals),
                )
            )
        return splits
    
    def to_memcpys(self) -> List[Tuple[int, int, int]]:
        """Convert to a list of (src_offset, dst_offset, nbytes) for each contiguous segment."""
        if self.src_region is None or self.dst_region is None:
            raise ValueError("Cannot materialize memcpy plan for dynamic regions.")

        src_segments = self.src_region.to_segments()
        dst_segments = self.dst_region.to_segments()

        if not src_segments or not dst_segments:
            return []

        if len(src_segments) == 1 and len(dst_segments) == 1:
            src_seg = src_segments[0]
            dst_seg = dst_segments[0]
            if src_seg.nbytes != dst_seg.nbytes:
                raise ValueError("Source and destination byte counts differ.")
            if src_seg.nbytes == 0:
                return []
            return [(src_seg.offset, dst_seg.offset, src_seg.nbytes)]

        src_total = sum(seg.nbytes for seg in src_segments)
        dst_total = sum(seg.nbytes for seg in dst_segments)
        if src_total != dst_total:
            raise ValueError("Source and destination regions disagree on total bytes.")
        if src_total == 0:
            return []

        result: List[Tuple[int, int, int]] = []
        src_idx = dst_idx = 0
        src_seg = src_segments[src_idx]
        dst_seg = dst_segments[dst_idx]
        src_offset = src_seg.offset
        dst_offset = dst_seg.offset
        src_remaining = src_seg.nbytes
        dst_remaining = dst_seg.nbytes

        while True:
            # Consume bytes from the current source and destination segments
            # in lockstep so the resulting memcpy spans are contiguous.
            chunk = src_remaining if src_remaining < dst_remaining else dst_remaining
            if chunk == 0:
                raise RuntimeError("Encountered zero-length segment while constructing memcpy plan.")

            # Merge with the previous memcpy when the new span is immediately adjacent
            # on both source and destination to minimize the number of descriptors.
            if result and result[-1][0] + result[-1][2] == src_offset and result[-1][1] + result[-1][2] == dst_offset:
                prev_src, prev_dst, prev_bytes = result[-1]
                result[-1] = (prev_src, prev_dst, prev_bytes + chunk)
            else:
                result.append((src_offset, dst_offset, chunk))

            src_offset += chunk
            dst_offset += chunk
            src_remaining -= chunk
            dst_remaining -= chunk

            if src_remaining == 0:
                # Advance to the next source segment and reset its counters.
                src_idx += 1
                if src_idx == len(src_segments):
                    if dst_remaining != 0 or dst_idx != len(dst_segments) - 1:
                        raise ValueError("Destination segments have remaining bytes after source exhausted.")
                    break
                src_seg = src_segments[src_idx]
                src_remaining = src_seg.nbytes
                src_offset = src_seg.offset
            if dst_remaining == 0:
                # Advance to the next destination segment and reset its counters.
                dst_idx += 1
                if dst_idx == len(dst_segments):
                    if src_remaining != 0 or src_idx != len(src_segments) - 1:
                        raise ValueError("Source segments have remaining bytes after destination exhausted.")
                    break
                dst_seg = dst_segments[dst_idx]
                dst_remaining = dst_seg.nbytes
                dst_offset = dst_seg.offset

        return result

class CollectiveOp(Enum):
    REDUCE_SCATTER = "reduce_scatter"
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    ALL_REDUCE_NVSHARP = "all_reduce_nvsharp"

@dataclass
class Collective:
    """Class for collective communication operations."""
    op: CollectiveOp
    src_buf: str
    src_region: Optional[BufferRegion]
    dst_buf: str
    dst_region: Optional[BufferRegion]
    dependency: List[XferDependency | ComputeDependency] = field(default_factory=list) # used to represent complex dependency such as ring
    set_signals: List[Signal] = field(default_factory=list) # should be infered from dependency
    period: Period = field(default_factory=Period)

    def split(self, num_splits: int) -> List["Collective"]:
        if num_splits <= 1:
            return [self]
        if self.src_region is None or self.dst_region is None:
            raise ValueError("Cannot split a Collective with dynamic regions.")

        src_splits = self.src_region.split(num_splits)
        dst_splits = self.dst_region.split(num_splits)
        if len(src_splits) != len(dst_splits):
            raise ValueError("Source and destination splits differ in count")

        split_period = self.period.split(num_splits)
        splits: List[Collective] = []
        for src_part, dst_part in zip(src_splits, dst_splits):
            # can't split after planning dependency and signals
            assert len(self.dependency) == 0
            assert len(self.set_signals) == 0
            splits.append(
                Collective(
                    op=self.op,
                    src_buf=self.src_buf,
                    src_region=src_part,
                    dst_buf=self.dst_buf,
                    dst_region=dst_part,
                    dependency=copy.deepcopy(self.dependency),
                    set_signals=copy.deepcopy(self.set_signals),
                    period=split_period,
                )
            )
        return splits

CommOp = Transfer | Collective

@dataclass
class DevicePlan:
    dev: int
    xfer: List[List[CommOp]] = field(default_factory=list)
    # Data that resides on the device before any transfer executes.
    local_regions: Dict[str, List[BufferRegion]] = field(default_factory=dict)
    # A push to B, but A wants to notify C after the transfer is done
    # We need to ask B to send a signal to C after receiving signal from A
    # This is because on systems with both NVLink and InfiniBand,
    # `quiet` visibility is only guaranteed at the destination PE.
    transit_signals: List[Tuple[Signal, Signal]] = field(default_factory=list) # (src_signal, dst_signal)
    tensors_involved: Dict[str, Tuple[torch.Size, torch.dtype]] = field(default_factory=dict) # buffer name -> (tensor, dtype)

    def __post_init__(self) -> None:
        if not self.xfer:
            self.xfer.append([])

    def _ensure_stream(self, idx: int) -> List[CommOp]:
        if idx < 0:
            raise ValueError("stream index must be non-negative")
        while len(self.xfer) <= idx:
            self.xfer.append([])
        return self.xfer[idx]

    def add_op(self, op: CommOp, *, stream_idx: int = 0) -> None:
        stream = self._ensure_stream(stream_idx)
        stream.append(op)

    def add_stream(self) -> int:
        self.xfer.append([])
        return len(self.xfer) - 1

    @property
    def xfers(self) -> List[Transfer]:
        return [
            op
            for stream in self.xfer
            for op in stream
            if isinstance(op, Transfer) and op.op in (TransferOp.PUSH, TransferOp.PULL)
        ]

    @property
    def local_copies(self) -> List[Transfer]:
        return [
            op
            for stream in self.xfer
            for op in stream
            if isinstance(op, Transfer) and op.op == TransferOp.LOCAL_COPY
        ]

    def pretty(self) -> str:
        lines = [f"DevicePlan(dev={self.dev})"]
        for stream_idx, stream in enumerate(self.xfer):
            lines.append(f"  Stream {stream_idx}:")
            if not stream:
                lines.append("    (none)")
                continue
            for op_idx, op in enumerate(stream):
                if isinstance(op, Transfer):
                    if op.op == TransferOp.PUSH:
                        detail = _format_push(op)
                    elif op.op == TransferOp.PULL:
                        detail = _format_pull(op)
                    else:
                        detail = _format_local_copy(op)
                elif isinstance(op, Collective):
                    detail = _format_collective(op)
                else:  # pragma: no cover - defensive
                    detail = f"unknown op {op}"
                lines.append(f"    [{op_idx}] {detail}")
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - convenience display
        return self.pretty()

    def split_xfers(self, num_splits: int) -> DevicePlan:
        new_streams: List[List[CommOp]] = []
        for stream in self.xfer:
            new_stream: List[CommOp] = []
            for op in stream:
                if isinstance(op, Transfer) and op.op in (TransferOp.PUSH, TransferOp.PULL):
                    new_stream.extend(op.split(num_splits))
                elif isinstance(op, Collective):
                    new_stream.extend(op.split(num_splits))
                else:
                    new_stream.append(copy.deepcopy(op))
            new_streams.append(new_stream)
        return DevicePlan(
            dev=self.dev,
            xfer=new_streams,
            local_regions=copy.deepcopy(self.local_regions),
            transit_signals=copy.deepcopy(self.transit_signals),
            tensors_involved=copy.deepcopy(self.tensors_involved),
        )

    def check_no_dependency(self) -> bool:
        for stream in self.xfer:
            for op in stream:
                if isinstance(op, Transfer) and op.op in (TransferOp.PUSH, TransferOp.PULL) and op.dependency is not None:
                    return False
                elif isinstance(op, Collective) and op.dependency is not None:
                    return False
        return True
        
@dataclass
class CommPlan:
    """Collection of per-device transfer plans plus signal dependency logic."""

    device_plans: dict[int, DevicePlan] = field(default_factory=dict)
    receive_signal_counters: dict[int, int] = field(default_factory=dict) # dev -> next available receive signal offset
    transit_signal_counters: dict[int, int] = field(default_factory=dict) # dev -> next available transit signal offset
    compute_signal_counters: dict[int, int] = field(default_factory=dict) # dev -> next available compute signal offset

    def __init__(self, device_plans: Optional[dict[int, DevicePlan]] = None) -> None:
        self.device_plans: dict[int, DevicePlan] = device_plans or {}
        self.receive_signal_counters: dict[int, int] = {dev: 0 for dev in self.device_plans}
        self.transit_signal_counters: dict[int, int] = {dev: 0 for dev in self.device_plans}
        self.compute_signal_counters: dict[int, int] = {dev: 0 for dev in self.device_plans}

    def add_device_plan(self, plan: DevicePlan) -> None:
        """Register a device plan under its device identifier."""
        if plan.dev in self.device_plans:
            raise ValueError(f"device plan for dev {plan.dev} already exists")
        self.device_plans[plan.dev] = plan

    @property
    def xfers(self) -> List[Transfer]:
        """Flatten all transfers across devices."""
        transfers: List[Transfer] = []
        for dev_plan in self.device_plans.values():
            transfers.extend(dev_plan.xfers)
        return transfers

    def pretty(self, include_signals: bool = False) -> str:
        """Render the communication plan. Optionally include signal details."""
        lines: List[str] = [f"CommPlan(devices={len(self.device_plans)})"]
        for dev in sorted(self.device_plans):
            dev_plan = self.device_plans[dev]
            lines.append(_indent(dev_plan.pretty(), "  "))
            if include_signals:
                lines.extend(self._format_signal_details(dev_plan))
        return "\n".join(lines)

    def visualize_signals(self) -> str:
        """Render a concise view of the signal dependencies."""
        lines: List[str] = ["CommPlan Signals:"]
        for dev in sorted(self.device_plans):
            dev_plan = self.device_plans[dev]
            lines.append(f"  Device {dev}:")
            details = self._format_signal_details(dev_plan)
            if details:
                lines.extend(details)
            else:
                lines.append("    (no signal activity)")
        return "\n".join(lines)

    def _format_signal_details(self, dev_plan: DevicePlan) -> List[str]:
        lines: List[str] = []
        for stream_idx, stream in enumerate(dev_plan.xfer):
            header_emitted = False
            for op_idx, op in enumerate(stream):
                if not header_emitted:
                    lines.append(f"    Stream {stream_idx}:")
                    header_emitted = True
                lines.append(
                    f"      op[{op_idx}] {op.op.value} -> signals: "
                    f"{', '.join(_format_signal(sig) for sig in op.set_signals) or '(none)'}"
                )
                if op.dependency:
                    for dep_idx, dep in enumerate(op.dependency):
                        if isinstance(dep, XferDependency):
                            target = (
                                f"xfer({dep.depend_rank},"
                                f"{dep.depend_xfer_stream},"
                                f"{dep.depend_xfer_idx})"
                            )
                            signal_str = (
                                _format_signal(dep.signal)
                                if dep.signal is not None
                                else "pending"
                            )
                            lines.append(
                                f"        dep[{dep_idx}] <- {target} -> {signal_str}"
                            )
                        elif isinstance(dep, ComputeDependency):
                            signal_str = (
                                _format_signal(dep.signal)
                                if dep.signal is not None
                                else "pending"
                            )
                            lines.append(
                                f"        dep[{dep_idx}] <- compute({dep.compute_rank}) -> {signal_str}"
                            )
            if not header_emitted:
                lines.append(f"    Stream {stream_idx}: (none)")
        if dev_plan.transit_signals:
            lines.append("    Transit signals:")
            for src_sig, dst_sig in dev_plan.transit_signals:
                lines.append(
                    f"      {_format_signal(src_sig)} -> {_format_signal(dst_sig)}"
                )
        return lines

    def _check_compute_dependency_signals_set(self, plan_compute_signals: bool) -> bool:
        for dev_plan in self.device_plans.values():
            for stream in dev_plan.xfer:
                for op in stream:
                    for dep in op.dependency:
                        if isinstance(dep, ComputeDependency):
                            if plan_compute_signals:
                                if dep.signal is not None:
                                    return False
                            else:
                                if dep.signal is None:
                                    return False
        return True

    def plan_signals(self, plan_compute_signals=True):
        """
        plan signals for all transfers based on their dependencies
        if plan_compute_signals is True, also plan signals for compute dependencies
        otherwise, compute signals should be handled by compute kernel writers
        """

        def get_signal_offset(dev: int, signal_counter: dict[int, int]) -> int:
            if dev not in signal_counter:
                raise ValueError(f"device {dev} not in signal counter")
            offset = signal_counter[dev]
            signal_counter[dev] += 1
            return offset
              
        assert all(dev_plan.transit_signals == [] for dev_plan in self.device_plans.values()), "transit_signals should be empty before planning signals"
        assert all(op.set_signals == [] for dev_plan in self.device_plans.values() for stream in dev_plan.xfer for op in stream), "set_signals should be empty before planning signals"

        assert self._check_compute_dependency_signals_set(plan_compute_signals), "if plan_compute_signals is True, compute dependency signals should be None before planning; if False, they should be already set"

        # tackle the receive signals first
        for dev, dev_plan in self.device_plans.items():
            for stream in dev_plan.xfer:
                for op in stream:
                    if isinstance(op, Transfer):
                        if op.op == TransferOp.PUSH:
                            # push: need to notify dst when done
                            assert op.peer is not None
                            offset = get_signal_offset(op.peer, self.receive_signal_counters)
                            dst_rank = op.peer
                        else:
                            # pull and local copy: just notify self when done
                            offset = get_signal_offset(dev, self.receive_signal_counters)
                            dst_rank = dev
                    elif isinstance(op, Collective):
                        # collective: just notify self when done
                        offset = get_signal_offset(dev, self.receive_signal_counters)
                        dst_rank = dev
                        
                    op.set_signals.append(
                        Signal(
                            dst_rank=dst_rank,
                            dst_offset=offset,
                            type=SignalType.RECEIVE_DATA
                        )
                    )

        # now turn the depenedencies into signals
        for dev, dev_plan in self.device_plans.items():
            for stream in dev_plan.xfer:
                for op in stream:
                    for dep in op.dependency:
                        if isinstance(dep, XferDependency):
                            if isinstance(op, Collective):
                                raise ValueError("Collective operations cannot have XferDependency for now")
                            depend_dev_plans = self.device_plans.get(dep.depend_rank, None)
                            assert depend_dev_plans is not None, f"depend_rank {dep.depend_rank} not found in device_plans"
                            depend_dev_transfer = depend_dev_plans.xfer[dep.depend_xfer_stream][dep.depend_xfer_idx]
                            
                            if not isinstance(depend_dev_transfer, Transfer):
                                raise ValueError("XferDependency can only depend on Collective operations for now")
                            
                            if depend_dev_transfer.op == TransferOp.PUSH:
                                # depend transfer is a push: need a transit siganl if current dev != push target
                                if dev != depend_dev_transfer.peer:
                                    # need a transit signal
                                    assert depend_dev_transfer.peer is not None
                                    offset = get_signal_offset(dev, self.transit_signal_counters)
                                    transit_signal = Signal(
                                        dst_rank = dev,
                                        dst_offset=offset,
                                        type=SignalType.REMOTE_DATA_READY
                                    )
                                    self.device_plans[depend_dev_transfer.peer].transit_signals.append(
                                        (depend_dev_transfer.set_signals[0], transit_signal)
                                    )
                                    dep.signal = transit_signal
                                else:
                                    # just use the receive signal
                                    assert depend_dev_transfer.set_signals[0].type == SignalType.RECEIVE_DATA
                                    dep.signal = depend_dev_transfer.set_signals[0] # receive signal is always the first one
                            else:
                                # depend transfer is a push or local copy: can know when the transfer is done
                                if dev != dep.depend_rank:
                                    # need additional signal
                                    offset = get_signal_offset(dev, self.receive_signal_counters)
                                    dep_signal = Signal(
                                        dst_rank=dev,
                                        dst_offset=offset,
                                        type=SignalType.REMOTE_DATA_READY
                                    )
                                    depend_dev_transfer.set_signals.append(dep_signal)
                                    dep.signal = dep_signal
                                else:
                                    # use the receive siganl
                                    assert depend_dev_transfer.set_signals[0].type == SignalType.RECEIVE_DATA
                                    dep.signal = depend_dev_transfer.set_signals[0] # receive signal is always the first one
                        elif isinstance(dep, ComputeDependency):
                            # assme each compute dependency corresponds to one compute task on compute_rank
                            # and one compute task only produces one compute done signal
                            if plan_compute_signals:
                                offset = get_signal_offset(dev, self.compute_signal_counters) # use the compute kernel to send the signal
                                compute_signal = Signal(
                                    dst_rank=dep.compute_rank,
                                    dst_offset=offset,
                                    type=SignalType.COMPUTE_DONE
                                )
                                dep.signal = compute_signal

                            # originally we want the user to provide the compute done signals on local rank, so we need to transit signals for remote ranks
                            # but this complicates the logic a lot, so we now let the signal be directly
                            
                            # if dep.compute_rank != dev:
                            #     # need to replace the signal with a transit signal
                            #     # because remote signals are not directly visible to local
                            #     offset = get_signal_offset(dev, self.transit_signal_counters)
                            #     transit_signal = Signal(
                            #         dst_rank=dev,
                            #         dst_offset=offset,
                            #         type=SignalType.COMPUTE_DONE
                            #     )
                            #     self.device_plans[dep.compute_rank].transit_signals.append(
                            #         (dep.signal, transit_signal)
                            #     )
                            #     dep.signal = transit_signal
                            # else:
                            #     # local compute signal, nothing to do
                            #     pass
                        else:
                            raise ValueError(f"unknown dependency type: {type(dep)}")

    def estimate_period(self):
        """
        estimate the start for each transfer based on their dependencies
        this is a lower-bound estimation that ignores the compute dependencies
        """
        node_lookup: Dict[Tuple[int, int, int], int] = {}
        transfers: List[CommOp] = []
        dependencies: List[set[int]] = []
        dependents: List[List[int]] = []

        # First, index every transfer and add in-stream ordering constraints.
        for dev, dev_plan in self.device_plans.items():
            for stream_idx, stream in enumerate(dev_plan.xfer):
                prev_node: Optional[int] = None
                for op_idx, op in enumerate(stream):
                    node_id = len(transfers)
                    transfers.append(op)
                    node_lookup[(dev, stream_idx, op_idx)] = node_id
                    dependencies.append(set())
                    dependents.append([])
                    if prev_node is not None:
                        dependencies[node_id].add(prev_node)
                        dependents[prev_node].append(node_id)
                    prev_node = node_id

        num_nodes = len(transfers)
        if num_nodes == 0:
            return

        # Next, add explicit transfer-to-transfer dependencies.
        for dev, dev_plan in self.device_plans.items():
            for stream_idx, stream in enumerate(dev_plan.xfer):
                for op_idx, op in enumerate(stream):
                    node_id = node_lookup[(dev, stream_idx, op_idx)]
                    for dep in op.dependency:
                        if not isinstance(dep, XferDependency):
                            continue
                        key = (dep.depend_rank, dep.depend_xfer_stream, dep.depend_xfer_idx)
                        if key not in node_lookup:
                            raise ValueError(
                                f"dependency target {key} not found in communication plan"
                            )
                        parent_id = node_lookup[key]
                        if parent_id not in dependencies[node_id]:
                            dependencies[node_id].add(parent_id)
                            dependents[parent_id].append(node_id)

        indegree = [len(dep_set) for dep_set in dependencies]
        ready_time = [0.0] * num_nodes
        queue = deque(idx for idx, deg in enumerate(indegree) if deg == 0)
        visited = 0

        while queue:
            node_id = queue.popleft()
            visited += 1
            transfer = transfers[node_id]
            start_time = ready_time[node_id]
            transfer.period.start = start_time
            completion = start_time + transfer.period.duration

            for signal in transfer.set_signals:
                signal.period.start = completion

            for child in dependents[node_id]:
                indegree[child] -= 1
                if completion > ready_time[child]:
                    ready_time[child] = completion
                if indegree[child] == 0:
                    queue.append(child)

        if visited != num_nodes:
            raise RuntimeError("cycle detected in transfer dependencies")

def _format_push(push: Transfer) -> str:
    peer = "?" if push.peer is None else push.peer
    return (
        "push peer="
        f"{peer} {_format_region(push.src_buf, push.src_region)} -> "
        f"{_format_region(push.dst_buf, push.dst_region)}"
    )


def _format_pull(pull: Transfer) -> str:
    peer = "?" if pull.peer is None else pull.peer
    return (
        "pull peer="
        f"{peer} {_format_region(pull.src_buf, pull.src_region)} -> "
        f"{_format_region(pull.dst_buf, pull.dst_region)}"
    )


def _format_local_copy(copy: Transfer) -> str:
    return (
        "copy "
        f"{_format_region(copy.src_buf, copy.src_region)} -> "
        f"{_format_region(copy.dst_buf, copy.dst_region)}"
    )

def _format_collective(coll: Collective) -> str:
    return (
        f"collective {coll.op.value} "
        f"{_format_region(coll.src_buf, coll.src_region)} -> "
        f"{_format_region(coll.dst_buf, coll.dst_region)}"
    )


def _format_region(buf_name: str, region: Optional[BufferRegion]) -> str:
    if region is None:
        return f"{buf_name}[?]"
    return f"{buf_name}{region.describe()}"


def _format_signal(signal: Signal) -> str:
    return f"{signal.type.value}@{signal.dst_rank}[{signal.dst_offset}]"


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


__all__ = [
    "CommPlan",
    "DevicePlan",
    "Transfer",
    "TransferOp",
    "Collective",
    "CollectiveOp",
    "CommOp",
    "Signal",
    "CommDescriptor",
    "DataPlacement",
    "XferDependency",
    "MemorySegment",
    "BufferRegion",
    "AxisSlice",
]
