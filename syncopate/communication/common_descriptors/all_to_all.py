"""All-to-all plan builder utilities."""

from __future__ import annotations

from typing import Literal, Sequence

from ..descriptor import BufferRegion, ComputeDependency, DevicePlan, Signal, SignalType, Transfer, TransferOp, XferDependency
from .utils import block_nbytes, ensure_divisible, row_major_strides
import torch

def build_all_to_all_plan(
    shape: Sequence[int],
    *,
    dtype: torch.dtype,
    mesh_size: int,
    rank: int,
    src_buffer: str,
    dst_buffer: str,
    transfer_kind: Literal["push", "pull"] = "push",
    compute_producer: bool = False,
) -> DevicePlan:
    """Build an all-to-all plan where dim0 is sharded across the mesh.

    Args:
        shape: Local source tensor shape on the current rank.
        elem_size: Size of a single element in bytes.
        mesh_size: Number of participating devices.
        rank: Rank of the current device in the mesh.
        src_buffer: Logical name for the source buffer.
        dst_buffer: Logical name for the destination buffer.
        transfer_kind: Communication strategy (push or pull).

    Returns:
        DevicePlan describing the transfers required for the local rank.

    Notes:
        The destination logical layout is inferred as ``(mesh_size, shape[0] / mesh_size, *shape[1:])``.
    """

    elem_size = dtype.itemsize
    if mesh_size <= 0:
        raise ValueError("mesh_size must be positive")
    if elem_size <= 0:
        raise ValueError("elem_size must be positive")

    shape = tuple(shape)
    if not shape:
        raise ValueError("all_to_all expects at least one dimension in the source shape")

    ensure_divisible(shape[0], mesh_size, axis=0)
    chunk = shape[0] // mesh_size
    if chunk == 0:
        raise ValueError("all-to-all requires non-empty slices along dim0")

    transfer_bytes = block_nbytes(shape, elem_size, {0: chunk})
    if transfer_bytes == 0:
        raise ValueError("all-to-all transfer would be empty")

    dst_shape = (mesh_size, chunk, *shape[1:])

    src_strides = row_major_strides(shape)
    dst_strides = row_major_strides(dst_shape)

    def src_region_for(dest_peer: int) -> BufferRegion:
        axis_slices = [(dest_peer * chunk, (dest_peer + 1) * chunk)]
        axis_slices.extend((0, dim_len) for dim_len in shape[1:])
        return BufferRegion.from_slices(
            elem_size=elem_size,
            strides=src_strides,
            slices=axis_slices,
        )

    def dst_region_for(source_rank: int) -> BufferRegion:
        axis_slices = [(source_rank, source_rank + 1), (0, chunk)]
        axis_slices.extend((0, dim_len) for dim_len in shape[1:])
        return BufferRegion.from_slices(
            elem_size=elem_size,
            strides=dst_strides,
            slices=axis_slices,
        )

    plan = DevicePlan(dev=rank)
    plan.src_shape = shape  # type: ignore[attr-defined]
    plan.dst_shape = dst_shape  # type: ignore[attr-defined]
    plan.tensors_involved[src_buffer] = (torch.Size(shape), dtype)
    plan.tensors_involved[dst_buffer] = (torch.Size(dst_shape), dtype)

    local_src_region = src_region_for(rank)
    local_dst_region = dst_region_for(rank)
    plan.local_regions.setdefault(src_buffer, []).append(local_src_region)

    if transfer_kind not in {"push", "pull"}:
        raise ValueError("transfer_kind must be 'push' or 'pull'")

    if compute_producer:
        compute_dependency = ComputeDependency(
            compute_rank=rank, 
            src_region=local_src_region, 
            signal=None,
        )
        dependency: list[XferDependency | ComputeDependency] = [compute_dependency]
    else:
        dependency = []

    local_copy = Transfer(
        op=TransferOp.LOCAL_COPY,
        src_buf=src_buffer,
        dst_buf=dst_buffer,
        src_region=local_src_region,
        dst_region=local_dst_region,
        chunk_idx=-1,
        dependency=dependency
    )

    if not compute_producer:
        # Comm is the producer, so place local copy first.
        plan.add_op(local_copy, stream_idx=0)

    if transfer_kind == "push":
        for i in range(1, mesh_size):
            peer = (i + rank) % mesh_size
            if compute_producer:
                compute_dependency = ComputeDependency(
                    compute_rank=rank, 
                    src_region=src_region_for(peer), 
                    signal=None,
                )
                dependency: list[XferDependency | ComputeDependency] = [compute_dependency]
            else:
                dependency: list[XferDependency | ComputeDependency] = []
            plan.add_op(
                Transfer(
                    op=TransferOp.PUSH,
                    src_buf=src_buffer,
                    src_region=src_region_for(peer),
                    peer=peer,
                    chunk_idx=rank * mesh_size + peer,
                    dst_buf=dst_buffer,
                    dst_region=local_dst_region,
                    dependency=dependency
                )
            )
    else:  # pull
        for i in range(1, mesh_size):
            peer = (i + rank) % mesh_size
            if compute_producer:
                compute_dependency = ComputeDependency(
                    compute_rank=peer,
                    src_region=local_src_region,
                )
                dependency: list[XferDependency | ComputeDependency] = [compute_dependency]
            else:
                dependency: list[XferDependency | ComputeDependency] = []
            plan.add_op(
                Transfer(
                    op=TransferOp.PULL,
                    dst_buf=dst_buffer,
                    dst_region=dst_region_for(peer),
                    peer=peer,
                    chunk_idx=peer * mesh_size + rank,
                    src_buf=src_buffer,
                    src_region=local_src_region,
                    dependency=dependency
                )
            )

    if compute_producer:
        plan.add_op(local_copy)
    return plan


def _normalize_dim(axis: int, ndim: int) -> int:
    if axis < 0:
        axis += ndim
    if not 0 <= axis < ndim:
        raise IndexError(f"axis {axis} out of range for tensor with {ndim} dims")
    return axis

def build_all_to_all_plan_dim(
    shape: Sequence[int],
    *,
    dtype: torch.dtype,
    mesh_size: int,
    rank: int,
    src_buffer: str,
    dst_buffer: str,
    src_split_axis: int,
    dst_split_axis: int,
    transfer_kind: Literal["push", "pull"] = "push",
    compute_producer: bool = False,
    peer_order: Literal["clockwise", "counter-clockwise"] = "clockwise",
) -> DevicePlan:
    """All-to-all that moves sharding from ``src_split_axis`` to ``dst_split_axis``.

    ``shape`` is the local source shape, which is expected to be sharded along
    ``src_split_axis`` (global length = ``shape[src_split_axis] * mesh_size``).
    The destination shape shreds ``dst_split_axis`` instead, producing a layout
    like ``[B, S/w, H, D] -> [B, S, H/w, D]``. ``peer_order`` controls whether
    peers are traversed clockwise or counter-clockwise around the mesh ring.
    """
    elem_size = dtype.itemsize
    if mesh_size <= 0:
        raise ValueError("mesh_size must be positive")
    if elem_size <= 0:
        raise ValueError("elem_size must be positive")

    shape = tuple(shape)
    if not shape:
        raise ValueError("all_to_all expects at least one dimension in the source shape")

    ndim = len(shape)
    src_split_axis = _normalize_dim(src_split_axis, ndim)
    dst_split_axis = _normalize_dim(dst_split_axis, ndim)
    if src_split_axis == dst_split_axis:
        raise ValueError("src_split_axis and dst_split_axis must differ")

    src_chunk = shape[src_split_axis]
    if src_chunk <= 0:
        raise ValueError("all-to-all requires non-empty slices along the split axis")
    ensure_divisible(shape[dst_split_axis], mesh_size, axis=dst_split_axis)
    dst_chunk = shape[dst_split_axis] // mesh_size
    if dst_chunk == 0:
        raise ValueError("all-to-all requires non-empty slices along the destination axis")

    transfer_bytes = block_nbytes(
        shape,
        elem_size,
        {
            src_split_axis: src_chunk,
            dst_split_axis: dst_chunk,
        },
    )
    if transfer_bytes == 0:
        raise ValueError("all-to-all transfer would be empty")

    global_shape = list(shape)
    global_shape[src_split_axis] = src_chunk * mesh_size
    dst_shape = list(global_shape)
    dst_shape[dst_split_axis] = dst_chunk

    src_strides = row_major_strides(shape)
    dst_strides = row_major_strides(dst_shape)

    def src_region_for(dest_peer: int) -> BufferRegion:
        axis_slices = []
        for dim, dim_len in enumerate(shape):
            if dim == src_split_axis:
                axis_slices.append((0, src_chunk))
            elif dim == dst_split_axis:
                start = dest_peer * dst_chunk
                axis_slices.append((start, start + dst_chunk))
            else:
                axis_slices.append((0, dim_len))
        return BufferRegion.from_slices(
            elem_size=elem_size,
            strides=src_strides,
            slices=axis_slices,
        )

    def dst_region_for(source_rank: int) -> BufferRegion:
        axis_slices = []
        for dim, dim_len in enumerate(dst_shape):
            if dim == src_split_axis:
                start = source_rank * src_chunk
                axis_slices.append((start, start + src_chunk))
            elif dim == dst_split_axis:
                axis_slices.append((0, dst_chunk))
            else:
                axis_slices.append((0, dim_len))
        return BufferRegion.from_slices(
            elem_size=elem_size,
            strides=dst_strides,
            slices=axis_slices,
        )

    plan = DevicePlan(dev=rank)
    plan.src_shape = shape  # type: ignore[attr-defined]
    plan.dst_shape = tuple(dst_shape)  # type: ignore[attr-defined]
    plan.tensors_involved[src_buffer] = (torch.Size(shape), dtype)
    plan.tensors_involved[dst_buffer] = (torch.Size(dst_shape), dtype)

    local_src_region = src_region_for(rank)
    local_dst_region = dst_region_for(rank)
    plan.local_regions.setdefault(src_buffer, []).append(local_src_region)

    if transfer_kind not in {"push", "pull"}:
        raise ValueError("transfer_kind must be 'push' or 'pull'")
    if peer_order not in {"clockwise", "counter-clockwise"}:
        raise ValueError("peer_order must be 'clockwise' or 'counter-clockwise'")

    if compute_producer:
        compute_dependency = ComputeDependency(
            compute_rank=rank,
            src_region=local_src_region,
            signal=None,
        )
        dependency: list[XferDependency | ComputeDependency] = [compute_dependency]
    else:
        dependency = []

    local_copy = Transfer(
        op=TransferOp.LOCAL_COPY,
        src_buf=src_buffer,
        dst_buf=dst_buffer,
        src_region=local_src_region,
        dst_region=local_dst_region,
        chunk_idx=-1,
        dependency=dependency,
    )

    if not compute_producer:
        # Communication is the producer, so move local copy to the front.
        plan.add_op(local_copy, stream_idx=0)

    if peer_order == "clockwise":
        peers = [(rank + i) % mesh_size for i in range(1, mesh_size)]
    else:
        peers = [(rank - i) % mesh_size for i in range(1, mesh_size)]

    if transfer_kind == "push":
        for peer in peers:
            if compute_producer:
                compute_dependency = ComputeDependency(
                    compute_rank=rank,
                    src_region=src_region_for(peer),
                    signal=None,
                )
                dependency: list[XferDependency | ComputeDependency] = [compute_dependency]
            else:
                dependency = []

            plan.add_op(
                Transfer(
                    op=TransferOp.PUSH,
                    src_buf=src_buffer,
                    src_region=src_region_for(peer),
                    peer=peer,
                    chunk_idx=rank * mesh_size + peer,
                    dst_buf=dst_buffer,
                    dst_region=local_dst_region,
                    dependency=dependency,
                )
            )
    else:  # pull
        for peer in peers:
            if compute_producer:
                compute_dependency = ComputeDependency(
                    compute_rank=peer,
                    src_region=local_src_region,
                )
                dependency = [compute_dependency]
            else:
                dependency = []

            plan.add_op(
                Transfer(
                    op=TransferOp.PULL,
                    dst_buf=dst_buffer,
                    dst_region=dst_region_for(peer),
                    peer=peer,
                    chunk_idx=peer * mesh_size + rank,
                    src_buf=src_buffer,
                    src_region=local_src_region,
                    dependency=dependency,
                )
            )

    if compute_producer:
        plan.add_op(local_copy)
    return plan


def build_all_to_all_plan_dim_waves(
    shape: Sequence[int],
    *,
    dtype: torch.dtype,
    mesh_size: int,
    rank: int,
    src_buffer: str,
    dst_buffer: str,
    src_split_axis: int,
    dst_split_axis: int,
    num_waves: int,
    transfer_kind: Literal["push", "pull"] = "push",
    compute_producer: bool = False,
    peer_order: Literal["clockwise", "counter-clockwise"] = "clockwise",
) -> DevicePlan:
    """All-to-all dim variant that splits the destination shard into waves.

    Waves are carved along ``dst_split_axis`` so receivers start consuming earlier
    slices (e.g., for ``[x, y/w] -> [x/w, y]`` the first wave can deliver
    ``[x/w/k, y]``). Local and remote peers are treated uniformly instead of
    handling the local shard separately.
    """

    elem_size = dtype.itemsize
    if mesh_size <= 0:
        raise ValueError("mesh_size must be positive")
    if elem_size <= 0:
        raise ValueError("elem_size must be positive")
    if num_waves <= 0:
        raise ValueError("num_waves must be positive")

    shape = tuple(shape)
    if not shape:
        raise ValueError("all_to_all expects at least one dimension in the source shape")

    ndim = len(shape)
    src_split_axis = _normalize_dim(src_split_axis, ndim)
    dst_split_axis = _normalize_dim(dst_split_axis, ndim)
    if src_split_axis == dst_split_axis:
        raise ValueError("src_split_axis and dst_split_axis must differ")

    src_chunk = shape[src_split_axis]
    if src_chunk <= 0:
        raise ValueError("all-to-all requires non-empty slices along the split axis")
    ensure_divisible(shape[dst_split_axis], mesh_size, axis=dst_split_axis)
    dst_chunk = shape[dst_split_axis] // mesh_size
    if dst_chunk == 0:
        raise ValueError("all-to-all requires non-empty slices along the destination axis")
    ensure_divisible(dst_chunk, num_waves, "num_waves")
    wave_chunk = dst_chunk // num_waves
    if wave_chunk == 0:
        raise ValueError("wave splitting produced an empty chunk along dst_split_axis")

    transfer_bytes = block_nbytes(
        shape,
        elem_size,
        {
            src_split_axis: src_chunk,
            dst_split_axis: wave_chunk,
        },
    )
    if transfer_bytes == 0:
        raise ValueError("all-to-all transfer would be empty")

    global_shape = list(shape)
    global_shape[src_split_axis] = src_chunk * mesh_size
    dst_shape = list(global_shape)
    dst_shape[dst_split_axis] = dst_chunk

    src_strides = row_major_strides(shape)
    dst_strides = row_major_strides(dst_shape)

    def src_region_for(dest_peer: int, wave_idx: int | None = None) -> BufferRegion:
        axis_slices = []
        for dim, dim_len in enumerate(shape):
            if dim == src_split_axis:
                axis_slices.append((0, src_chunk))
            elif dim == dst_split_axis:
                start = dest_peer * dst_chunk
                if wave_idx is not None:
                    start += wave_idx * wave_chunk
                    axis_slices.append((start, start + wave_chunk))
                else:
                    axis_slices.append((start, start + dst_chunk))
            else:
                axis_slices.append((0, dim_len))
        return BufferRegion.from_slices(
            elem_size=elem_size,
            strides=src_strides,
            slices=axis_slices,
        )

    def dst_region_for(source_rank: int, wave_idx: int | None = None) -> BufferRegion:
        axis_slices = []
        for dim, dim_len in enumerate(dst_shape):
            if dim == src_split_axis:
                start = source_rank * src_chunk
                axis_slices.append((start, start + src_chunk))
            elif dim == dst_split_axis:
                if wave_idx is None:
                    axis_slices.append((0, dst_chunk))
                else:
                    start = wave_idx * wave_chunk
                    axis_slices.append((start, start + wave_chunk))
            else:
                axis_slices.append((0, dim_len))
        return BufferRegion.from_slices(
            elem_size=elem_size,
            strides=dst_strides,
            slices=axis_slices,
        )

    plan = DevicePlan(dev=rank)
    plan.src_shape = shape  # type: ignore[attr-defined]
    plan.dst_shape = tuple(dst_shape)  # type: ignore[attr-defined]
    plan.tensors_involved[src_buffer] = (torch.Size(shape), dtype)
    plan.tensors_involved[dst_buffer] = (torch.Size(dst_shape), dtype)

    plan.local_regions.setdefault(src_buffer, []).append(src_region_for(rank))

    if transfer_kind not in {"push", "pull"}:
        raise ValueError("transfer_kind must be 'push' or 'pull'")
    if peer_order not in {"clockwise", "counter-clockwise"}:
        raise ValueError("peer_order must be 'clockwise' or 'counter-clockwise'")

    if peer_order == "clockwise":
        peers = [(rank + i) % mesh_size for i in range(mesh_size)]
    else:
        peers = [(rank - i) % mesh_size for i in range(mesh_size)]

    for wave_idx in range(num_waves):
        for peer in peers:
            if peer == rank:
                dependency: list[XferDependency | ComputeDependency] = []
                if compute_producer:
                    dependency.append(
                        ComputeDependency(
                            compute_rank=rank,
                            src_region=src_region_for(rank, wave_idx),
                            signal=None,
                        )
                    )
                plan.add_op(
                    Transfer(
                        op=TransferOp.LOCAL_COPY,
                        src_buf=src_buffer,
                        dst_buf=dst_buffer,
                        src_region=src_region_for(rank, wave_idx),
                        dst_region=dst_region_for(rank, wave_idx),
                        chunk_idx=(rank * mesh_size + rank) * num_waves + wave_idx,
                        dependency=dependency,
                    )
                )
                continue

            if transfer_kind == "push":
                dependency = []
                if compute_producer:
                    dependency.append(
                        ComputeDependency(
                            compute_rank=rank,
                            src_region=src_region_for(peer, wave_idx),
                            signal=None,
                        )
                    )
                plan.add_op(
                    Transfer(
                        op=TransferOp.PUSH,
                        src_buf=src_buffer,
                        src_region=src_region_for(peer, wave_idx),
                        peer=peer,
                        chunk_idx=(rank * mesh_size + peer) * num_waves + wave_idx,
                        dst_buf=dst_buffer,
                        dst_region=dst_region_for(rank, wave_idx),
                        dependency=dependency,
                    )
                )
            else:  # pull
                dependency = []
                if compute_producer:
                    dependency.append(
                        ComputeDependency(
                            compute_rank=peer,
                            src_region=src_region_for(rank, wave_idx),
                        )
                    )
                plan.add_op(
                    Transfer(
                        op=TransferOp.PULL,
                        dst_buf=dst_buffer,
                        dst_region=dst_region_for(peer, wave_idx),
                        peer=peer,
                        chunk_idx=(peer * mesh_size + rank) * num_waves + wave_idx,
                        src_buf=src_buffer,
                        src_region=src_region_for(rank, wave_idx),
                        dependency=dependency,
                    )
                )

    return plan


def build_all_to_all_plan_axis(
    shape: Sequence[int],
    *,
    dtype: torch.dtype,
    mesh_size: int,
    rank: int,
    src_buffer: str,
    dst_buffer: str,
    split_axis: int,
    transfer_kind: Literal["push", "pull"] = "push",
    compute_producer: bool = False,
) -> DevicePlan:
    """Standard all-to-all along ``split_axis`` (same axis before/after).

    ``shape`` is the local source shape on the current rank, with
    ``shape[split_axis]`` divisible by ``mesh_size``.
    """

    elem_size = dtype.itemsize
    if mesh_size <= 0:
        raise ValueError("mesh_size must be positive")
    if elem_size <= 0:
        raise ValueError("elem_size must be positive")

    shape = tuple(shape)
    if not shape:
        raise ValueError("all_to_all expects at least one dimension in the source shape")

    ndim = len(shape)
    split_axis = _normalize_dim(split_axis, ndim)

    axis_len = shape[split_axis]
    ensure_divisible(axis_len, mesh_size, axis=split_axis)
    chunk = axis_len // mesh_size
    if chunk == 0:
        raise ValueError("all-to-all requires non-empty slices along the split axis")

    transfer_bytes = block_nbytes(shape, elem_size, {split_axis: chunk})
    if transfer_bytes == 0:
        raise ValueError("all-to-all transfer would be empty")

    dst_shape = shape
    strides = row_major_strides(shape)

    def src_region_for(dest_peer: int) -> BufferRegion:
        axis_slices = []
        for dim, dim_len in enumerate(shape):
            if dim == split_axis:
                start = dest_peer * chunk
                axis_slices.append((start, start + chunk))
            else:
                axis_slices.append((0, dim_len))
        return BufferRegion.from_slices(
            elem_size=elem_size,
            strides=strides,
            slices=axis_slices,
        )

    def dst_region_for(source_rank: int) -> BufferRegion:
        axis_slices = []
        for dim, dim_len in enumerate(dst_shape):
            if dim == split_axis:
                start = source_rank * chunk
                axis_slices.append((start, start + chunk))
            else:
                axis_slices.append((0, dim_len))
        return BufferRegion.from_slices(
            elem_size=elem_size,
            strides=strides,
            slices=axis_slices,
        )

    plan = DevicePlan(dev=rank)
    plan.src_shape = shape  # type: ignore[attr-defined]
    plan.dst_shape = dst_shape  # type: ignore[attr-defined]
    plan.tensors_involved[src_buffer] = (torch.Size(shape), dtype)
    plan.tensors_involved[dst_buffer] = (torch.Size(dst_shape), dtype)

    local_src_region = src_region_for(rank)
    local_dst_region = dst_region_for(rank)
    plan.local_regions.setdefault(src_buffer, []).append(local_src_region)

    if transfer_kind not in {"push", "pull"}:
        raise ValueError("transfer_kind must be 'push' or 'pull'")

    if compute_producer:
        compute_dependency = ComputeDependency(
            compute_rank=rank,
            src_region=local_src_region,
            signal=None,
        )
        dependency: list[XferDependency | ComputeDependency] = [compute_dependency]
    else:
        dependency = []

    local_copy = Transfer(
        op=TransferOp.LOCAL_COPY,
        src_buf=src_buffer,
        dst_buf=dst_buffer,
        src_region=local_src_region,
        dst_region=local_dst_region,
        chunk_idx=-1,
        dependency=dependency,
    )

    if not compute_producer:
        plan.add_op(local_copy, stream_idx=0)

    if transfer_kind == "push":
        for i in range(1, mesh_size):
            peer = (i + rank) % mesh_size
            if compute_producer:
                compute_dependency = ComputeDependency(
                    compute_rank=rank,
                    src_region=src_region_for(peer),
                    signal=None,
                )
                dependency = [compute_dependency]
            else:
                dependency = []

            plan.add_op(
                Transfer(
                    op=TransferOp.PUSH,
                    src_buf=src_buffer,
                    src_region=src_region_for(peer),
                    peer=peer,
                    chunk_idx=rank * mesh_size + peer,
                    dst_buf=dst_buffer,
                    dst_region=local_dst_region,
                    dependency=dependency,
                )
            )
    else:  # pull
        for i in range(1, mesh_size):
            peer = (i + rank) % mesh_size
            if compute_producer:
                compute_dependency = ComputeDependency(
                    compute_rank=peer,
                    src_region=local_src_region,
                )
                dependency = [compute_dependency]
            else:
                dependency = []

            plan.add_op(
                Transfer(
                    op=TransferOp.PULL,
                    dst_buf=dst_buffer,
                    dst_region=dst_region_for(peer),
                    peer=peer,
                    chunk_idx=peer * mesh_size + rank,
                    src_buf=src_buffer,
                    src_region=local_src_region,
                    dependency=dependency,
                )
            )

    if compute_producer:
        plan.add_op(local_copy)
    return plan


__all__ = [
    "build_all_to_all_plan",
    "build_all_to_all_plan_dim",
    "build_all_to_all_plan_dim_waves",
    "build_all_to_all_plan_axis",
]
