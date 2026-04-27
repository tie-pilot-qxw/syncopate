"""Utilities for converting communication plans into tile schedules."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from syncopate.communication.descriptor import (
    BufferRegion,
    Collective,
    CommPlan,
    CommOp,
    DevicePlan,
    ReduceOp,
    Signal,
    Transfer,
    TransferOp,
)

from .tile_schedule import RawSchedule, TileCoord, TileBlock


class _ScheduleError(ValueError):
    """Internal error used to surface issues encountered during lowering."""


@dataclass
class _BufferScheduleBuilder:
    """Accumulates tile coordinates for a single logical buffer."""

    expected_shape: Optional[Tuple[int, ...]] = None

    def __post_init__(self) -> None:
        if self.expected_shape is not None and any(dim <= 0 for dim in self.expected_shape):
            raise _ScheduleError("expected_shape must contain positive dimensions")
        self._waves: List[List[TileBlock]] = []

    def append_wave(self, blocks: Iterable[TileBlock]) -> None:
        """Append a new wave composed of the provided tile coordinates."""
        blocks = list(blocks)
        if not blocks:
            return

        self._waves.append(blocks)

    def build(self) -> RawSchedule:
        """Finalize the accumulated waves into a RawSchedule."""
        if not self._waves:
            raise _ScheduleError("cannot build schedule with no waves")

        extent: Optional[List[int]] = None
        for wave in self._waves:
            for block in wave:
                tile = block.tile
                if extent is None:
                    extent = [0] * tile.ndim
                elif len(extent) != tile.ndim:
                    raise _ScheduleError("inconsistent dimensionality encountered while lowering")
                for axis in range(tile.ndim):
                    bound = tile.offsets[axis] + tile.shape[axis]
                    if bound > extent[axis]:
                        extent[axis] = bound

        if extent is None:
            raise _ScheduleError("buffer extent is unknown after lowering")

        if self.expected_shape is not None:
            if len(self.expected_shape) != len(extent):
                raise _ScheduleError("mismatched dimensionality between tiles and expected shape")
            for axis, (observed, expected) in enumerate(zip(extent, self.expected_shape)):
                if observed > expected:
                    raise _ScheduleError(
                        f"observed extent {observed} exceeds expected shape {expected} on axis {axis}"
                    )
            global_shape = tuple(self.expected_shape)
        else:
            global_shape = tuple(extent)

        return RawSchedule(
            num_waves=len(self._waves),
            block_infos=[list(wave) for wave in self._waves],
            global_shape=global_shape,
        )

    def has_waves(self) -> bool:
        return bool(self._waves)


def lower_comm_plan_to_raw_schedules(
    comm_plan: CommPlan,
) -> Dict[int, Dict[str, RawSchedule]]:
    """Lower a communication plan into per-device tile schedules."""

    comm_plan.estimate_period()

    shape_hints: Dict[int, Dict[str, Tuple[int, ...]]] = {}
    for dev, device_plan in comm_plan.device_plans.items():
        shape_hints[dev] = {
            name: tuple(size) for name, (size, _dtype) in device_plan.tensors_involved.items()
        }

    builders: Dict[int, Dict[str, _BufferScheduleBuilder]] = {}

    def get_builder(device: int, buffer_name: str) -> _BufferScheduleBuilder:
        per_device = builders.setdefault(device, {})
        builder = per_device.get(buffer_name)
        if builder is None:
            expected_shape = shape_hints.get(device, {}).get(buffer_name)
            builder = _BufferScheduleBuilder(expected_shape=expected_shape)
            per_device[buffer_name] = builder
        return builder

    # Capture any data that is already resident on each device.
    local_region_blocks: Dict[Tuple[int, str], List[TileBlock]] = defaultdict(list)
    for dev, device_plan in comm_plan.device_plans.items():
        for buffer_name, regions in device_plan.local_regions.items():
            blocks: List[TileBlock] = []
            for region in regions:
                for tile in _region_to_tiles(region):
                    blocks.append(TileBlock(tile=tile))
            if blocks:
                local_region_blocks[(dev, buffer_name)].extend(blocks)

    # Timeline of transfers across all devices.
    ops_with_time: List[Tuple[float, int, int, int, CommOp]] = []
    for dev, device_plan in comm_plan.device_plans.items():
        for stream_idx, stream in enumerate(device_plan.xfer):
            for op_idx, op in enumerate(stream):
                if op.period.start is None:
                    raise _ScheduleError("transfer period start is unknown; run period estimation first")
                ops_with_time.append((op.period.start, dev, stream_idx, op_idx, op))

    ops_with_time.sort(key=lambda item: (item[0], item[1], item[2], item[3]))

    device_events: Dict[int, List[Tuple[float, str, Tuple[int, ...], TileBlock]]] = defaultdict(list)
    consumer_first_touch: Dict[
        Tuple[int, str], Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Tuple[float, Tuple[int, ...], TileBlock]]
    ] = defaultdict(dict)
    role_flags: Dict[Tuple[int, str], Dict[str, bool]] = defaultdict(lambda: {"consumer": False, "producer": False})

    def record_region_event(
        device: int,
        buffer_name: str,
        region: BufferRegion,
        *,
        is_consumer: bool,
        signal: Optional[Signal],
        start_time: float,
        order_prefix: Tuple[int, ...],
        peer_rank: Optional[int],
        reduce_op: Optional[ReduceOp],
        peer_region: Optional[BufferRegion],
    ) -> None:
        tiles = _region_to_tiles(region)
        if not tiles:
            return

        peer_tiles = _region_to_tiles(peer_region) if peer_region is not None else []

        flags = role_flags[(device, buffer_name)]
        if is_consumer:
            flags["consumer"] = True
        else:
            flags["producer"] = True

        # is_consumer reflects the comm's perspective on the buffer. The
        # buffer holder's compute role is the inverse: if comm consumes the
        # buffer (reads it out), compute produced it.
        compute_role = "producer" if is_consumer else "consumer"

        for tile_idx, tile in enumerate(tiles):
            # Match each local tile to its peer-side counterpart by index. If
            # the peer side enumerates a different number of tiles we fall
            # back to None — the rewrite uses tile.offsets when dst_offsets is
            # absent which keeps canonical accesses in place.
            dst_offsets: Optional[Tuple[int, ...]]
            if tile_idx < len(peer_tiles):
                dst_offsets = peer_tiles[tile_idx].offsets
            else:
                dst_offsets = None

            block = TileBlock(
                tile=tile,
                signal=signal,
                peer_rank=peer_rank,
                reduce_op=reduce_op,
                role=compute_role,
                dst_offsets=dst_offsets,
            )
            order_key = order_prefix + (tile_idx,)
            if is_consumer:
                key = (tile.offsets, tile.shape)
                per_buffer = consumer_first_touch[(device, buffer_name)]
                stored = per_buffer.get(key)
                if stored is None:
                    per_buffer[key] = (start_time, order_key, block)
                else:
                    prev_time, prev_order, _ = stored
                    if start_time < prev_time - 1e-9:
                        per_buffer[key] = (start_time, order_key, block)
                    elif math.isclose(start_time, prev_time, rel_tol=1e-9, abs_tol=1e-9):
                        if order_key < prev_order:
                            per_buffer[key] = (start_time, order_key, block)
            else:
                device_events[device].append((start_time, buffer_name, order_key, block))

    for start_time, origin_dev, stream_idx, op_idx, op in ops_with_time:
        op_reduce_op = _reduce_op_for(op)
        local_order_prefix = (origin_dev, 0, stream_idx, op_idx)
        for buffer_name, region, is_consumer, peer_rank, peer_region in _local_buffer_regions(op):
            if is_consumer:
                signal = _dependency_signal_for_device(op, origin_dev)
            else:
                signal = _transfer_signal_for_device(op, origin_dev)
            record_region_event(
                origin_dev,
                buffer_name,
                region,
                is_consumer=is_consumer,
                signal=signal,
                start_time=start_time,
                order_prefix=local_order_prefix,
                peer_rank=peer_rank,
                reduce_op=op_reduce_op,
                peer_region=peer_region,
            )

        remote_order_prefix = (origin_dev, 1, stream_idx, op_idx)
        for target_dev, buffer_name, region, is_consumer, peer_rank, peer_region in _remote_buffer_regions(
            origin_dev, op
        ):
            if target_dev not in comm_plan.device_plans:
                continue
            if is_consumer:
                signal = _dependency_signal_for_device(op, target_dev)
            else:
                signal = _transfer_signal_for_device(op, target_dev)
            record_region_event(
                target_dev,
                buffer_name,
                region,
                is_consumer=is_consumer,
                signal=signal,
                start_time=start_time,
                order_prefix=remote_order_prefix,
                peer_rank=peer_rank,
                reduce_op=op_reduce_op,
                peer_region=peer_region,
            )

    for (device, buffer_name), tiles in consumer_first_touch.items():
        flags = role_flags.get((device, buffer_name))
        if flags and flags["producer"]:
            continue
        for start_time, order_key, block in tiles.values():
            device_events[device].append((start_time, buffer_name, order_key, block))

    # Inject local data only if the buffer is not exclusively consumed by communication.
    for (device, buffer_name), blocks in local_region_blocks.items():
        flags = role_flags.get((device, buffer_name))
        if flags and flags["consumer"] and not flags["producer"]:
            continue
        if not blocks:
            continue
        # Local-residency blocks represent the device's own slice of the buffer
        # (peer == self). Infer compute role from observed comm role: comm
        # producing the buffer means compute consumes it, and vice versa.
        if flags is None:
            compute_role: Optional[str] = None
        elif flags["producer"] and not flags["consumer"]:
            compute_role = "consumer"
        elif flags["consumer"] and not flags["producer"]:
            compute_role = "producer"
        else:
            compute_role = None
        annotated_blocks = [
            TileBlock(
                tile=block.tile,
                signal=block.signal,
                peer_rank=device,
                reduce_op=block.reduce_op,
                role=compute_role,
            )
            for block in blocks
        ]
        get_builder(device, buffer_name).append_wave(annotated_blocks)

    for device, events in device_events.items():
        if not events:
            continue
        events.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
                item[3].tile.offsets,
                item[3].tile.shape,
            )
        )

        current_start: Optional[float] = None
        wave_blocks: Dict[str, List[TileBlock]] = defaultdict(list)

        def flush_wave() -> None:
            if not wave_blocks:
                return
            for buffer_name, blocks in wave_blocks.items():
                get_builder(device, buffer_name).append_wave(blocks)
            wave_blocks.clear()

        for start_time, buffer_name, _order_key, block in events:
            if current_start is None or not math.isclose(start_time, current_start, rel_tol=1e-9, abs_tol=1e-9):
                flush_wave()
                current_start = start_time
            wave_blocks[buffer_name].append(block)

        flush_wave()

    lowered: Dict[int, Dict[str, RawSchedule]] = {}
    for device, per_device_builders in builders.items():
        schedules: Dict[str, RawSchedule] = {}
        for buffer_name, builder in per_device_builders.items():
            if not builder.has_waves():
                continue
            schedules[buffer_name] = builder.build()
        if schedules:
            lowered[device] = schedules

    return lowered


def _local_buffer_regions(
    op: CommOp,
) -> List[Tuple[str, BufferRegion, bool, Optional[int], Optional[BufferRegion]]]:
    """Return buffer-region-role tuples that reside on the local device.

    The boolean flag is ``True`` when the transfer *consumes* data from the
    buffer (communication acts as the consumer) and ``False`` when the transfer
    produces data into the buffer. The fourth element is the cross-device peer
    associated with this regional touch (``None`` for purely local copies or
    collectives). The fifth element is the *peer-side* counterpart region (the
    region of the buffer on the peer device) for PUSH/PULL transfers; ``None``
    when there is no peer-side region.
    """

    def ensure_region(name: str, region: Optional[BufferRegion]) -> BufferRegion:
        if region is None:
            raise _ScheduleError(f"transfer for buffer '{name}' lacks a concrete region")
        return region

    if isinstance(op, Transfer):
        if op.op == TransferOp.PUSH:
            local = ensure_region(op.src_buf, op.src_region)
            peer_side = ensure_region(op.dst_buf, op.dst_region)
            return [(op.src_buf, local, True, op.peer, peer_side)]
        if op.op == TransferOp.PULL:
            local = ensure_region(op.dst_buf, op.dst_region)
            peer_side = ensure_region(op.src_buf, op.src_region)
            return [(op.dst_buf, local, False, op.peer, peer_side)]
        if op.op == TransferOp.LOCAL_COPY:
            pairs: List[Tuple[str, BufferRegion, bool, Optional[int], Optional[BufferRegion]]] = []
            src_region = ensure_region(op.src_buf, op.src_region)
            dst_region = ensure_region(op.dst_buf, op.dst_region)
            # No cross-device peer for a local copy, but the data still has a
            # "destination side" — the dst_region within dst_buf on the same
            # device. Capturing it lets fused producers see e.g. dst_offset=0
            # for the local wave of a reduce-scatter, instead of falling back
            # to the source-side row offset which would land out of bounds.
            pairs.append((op.src_buf, src_region, True, None, dst_region))
            if op.dst_buf != op.src_buf or op.dst_region != op.src_region:
                pairs.append((op.dst_buf, dst_region, False, None, src_region))
            return pairs
        raise _ScheduleError(f"unsupported transfer operation: {op.op}")

    if isinstance(op, Collective):
        pairs: List[Tuple[str, BufferRegion, bool, Optional[int], Optional[BufferRegion]]] = []
        pairs.append((op.src_buf, ensure_region(op.src_buf, op.src_region), True, None, None))
        pairs.append((op.dst_buf, ensure_region(op.dst_buf, op.dst_region), False, None, None))
        return pairs

    raise _ScheduleError(f"unsupported op type: {type(op)}")


def _remote_buffer_regions(
    origin_dev: int, op: CommOp
) -> List[Tuple[int, str, BufferRegion, bool, int, Optional[BufferRegion]]]:
    """Return remote buffer interactions for the transfer.

    Like ``_local_buffer_regions`` the last element is the peer-side
    counterpart region — but here "peer" is the *origin* device from the remote
    side's perspective.
    """

    def ensure_region(name: str, region: Optional[BufferRegion]) -> BufferRegion:
        if region is None:
            raise _ScheduleError(f"transfer for buffer '{name}' lacks a concrete region")
        return region

    if isinstance(op, Transfer):
        if op.op == TransferOp.PUSH:
            if op.peer is None:
                raise _ScheduleError("push transfer lacks peer device")
            local = ensure_region(op.dst_buf, op.dst_region)
            peer_side = ensure_region(op.src_buf, op.src_region)
            return [(op.peer, op.dst_buf, local, False, origin_dev, peer_side)]
        if op.op == TransferOp.PULL:
            if op.peer is None:
                raise _ScheduleError("pull transfer lacks peer device")
            local = ensure_region(op.src_buf, op.src_region)
            peer_side = ensure_region(op.dst_buf, op.dst_region)
            return [(op.peer, op.src_buf, local, True, origin_dev, peer_side)]
        return []

    if isinstance(op, Collective):
        return []

    raise _ScheduleError(f"unsupported op type: {type(op)}")


def _reduce_op_for(op: CommOp) -> Optional[ReduceOp]:
    if isinstance(op, Transfer):
        return op.reduce_op
    return None


def _transfer_signal_for_device(op: CommOp, device: int) -> Optional[Signal]:
    for signal in op.set_signals:
        if signal.dst_rank == device:
            return signal
    return None


def _dependency_signal_for_device(op: CommOp, device: int) -> Optional[Signal]:
    for dep in op.dependency:
        signal = getattr(dep, "signal", None)
        if signal is not None and signal.dst_rank == device:
            return signal
    return None


def _region_to_tiles(region: BufferRegion) -> List[TileCoord]:
    tiles: List[TileCoord] = []
    for region_axes in region.regions:
        offsets = tuple(axis.start for axis in region_axes)
        shape = tuple(axis.length() for axis in region_axes)
        if not shape or any(dim <= 0 for dim in shape):
            continue
        tiles.append(TileCoord(offsets=offsets, shape=shape))
    return tiles


__all__ = ["lower_comm_plan_to_raw_schedules"]
