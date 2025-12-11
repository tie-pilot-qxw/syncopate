"""Reduce-scatter plan builder."""

from __future__ import annotations

from typing import Literal, Sequence

import torch

from ..descriptor import (
    BufferRegion,
    Collective,
    CollectiveOp,
    ComputeDependency,
    DevicePlan,
    ReduceOp,
    Transfer,
    TransferOp,
)
from .utils import ensure_divisible, row_major_strides


def build_reduce_scatter_collective_plan(
    shape: Sequence[int],
    *,
    dtype: torch.dtype,
    axis: int,
    split_axis: int | None = None,
    mesh_size: int,
    rank: int,
    src_buffer: str,
    dst_buffer: str,
    num_reduce_scatters: int = 1,
    compute_producer: bool = False,
) -> DevicePlan:
    """Build a reduce-scatter plan along ``axis`` using collective ops.

    Args:
        shape: Logical input shape on the current rank (full pre-reduction tensor).
        dtype: Element dtype.
        axis: Dimension along which to reduce and scatter.
        split_axis: Dimension to shard into multiple collective calls. Must differ
            from ``axis``. If ``None``, ``num_reduce_scatters`` must be 1.
        mesh_size: Number of participating devices.
        rank: Rank of the current device.
        src_buffer: Name of the input buffer containing local contributions.
        dst_buffer: Name of the output buffer for the scattered slice.
        num_reduce_scatters: Number of sub-collectives along ``split_axis``.
        compute_producer: If ``True``, add a compute dependency so comm waits for
            the local producer to finish before launching the collective.
    """

    elem_size = dtype.itemsize
    if mesh_size <= 0:
        raise ValueError("mesh_size must be positive")
    if elem_size <= 0:
        raise ValueError("elem_size must be positive")
    if num_reduce_scatters <= 0:
        raise ValueError("num_reduce_scatters must be positive")

    shape = tuple(shape)
    ndim = len(shape)
    if ndim == 0:
        raise ValueError("reduce_scatter expects at least one dimension")

    if axis < 0:
        axis += ndim
    if not 0 <= axis < ndim:
        raise IndexError(f"axis {axis} out of range for tensor with {ndim} dims")

    axis_len = shape[axis]
    ensure_divisible(axis_len, mesh_size, axis)
    per_rank = axis_len // mesh_size

    if split_axis is None:
        if num_reduce_scatters != 1:
            raise ValueError("split_axis is required when num_reduce_scatters > 1")
        split_axis = axis + 1 if axis + 1 < ndim else 0
        if split_axis == axis:
            split_axis = (axis + 1) % ndim

    if split_axis < 0:
        split_axis += ndim
    if not 0 <= split_axis < ndim:
        raise IndexError(f"split_axis {split_axis} out of range for tensor with {ndim} dims")
    if split_axis == axis:
        raise ValueError("split_axis must differ from reduce axis")

    split_len = shape[split_axis]
    ensure_divisible(split_len, num_reduce_scatters, "num_reduce_scatters")
    split_chunk = split_len // num_reduce_scatters
    if split_chunk == 0:
        raise ValueError("split produces empty chunks")

    src_strides = row_major_strides(shape)
    dst_shape = list(shape)
    dst_shape[axis] = per_rank
    dst_shape = tuple(dst_shape)
    dst_strides = row_major_strides(dst_shape)

    def make_region(
        split_start: int,
        split_end: int,
        *,
        strides: Sequence[int],
        region_shape: Sequence[int],
    ) -> BufferRegion:
        slices = []
        for dim, dim_len in enumerate(region_shape):
            if dim == split_axis:
                slices.append((split_start, split_end))
            else:
                slices.append((0, dim_len))
        return BufferRegion.from_slices(
            elem_size=elem_size,
            strides=strides,
            slices=slices,
        )

    plan = DevicePlan(dev=rank)
    plan.tensors_involved[src_buffer] = (torch.Size(shape), dtype)
    plan.tensors_involved[dst_buffer] = (torch.Size(dst_shape), dtype)

    src_region_full = make_region(0, split_len, strides=src_strides, region_shape=shape)
    plan.local_regions.setdefault(src_buffer, []).append(src_region_full)

    for split_idx in range(num_reduce_scatters):
        split_start = split_idx * split_chunk
        split_end = split_start + split_chunk

        src_region = make_region(
            split_start,
            split_end,
            strides=src_strides,
            region_shape=shape,
        )
        dst_region = make_region(
            split_start,
            split_end,
            strides=dst_strides,
            region_shape=dst_shape,
        )

        dependency = []
        if compute_producer:
            dependency.append(ComputeDependency(compute_rank=rank, src_region=src_region))

        plan.add_op(
            Collective(
                op=CollectiveOp.REDUCE_SCATTER,
                src_buf=src_buffer,
                src_region=src_region,
                dst_buf=dst_buffer,
                dst_region=dst_region,
                dependency=dependency,
            )
        )

    return plan


def build_reduce_scatter_direct_reduce_plan(
    shape: Sequence[int],
    *,
    dtype: torch.dtype,
    axis: int,
    mesh_size: int,
    rank: int,
    src_buffer: str,
    dst_buffer: str,
    reduce_op: ReduceOp = ReduceOp.SUM,
    transfer_kind: Literal["push", "pull"] = "push",
) -> DevicePlan:
    elem_size = dtype.itemsize
    if mesh_size <= 0:
        raise ValueError("mesh_size must be positive")
    if elem_size <= 0:
        raise ValueError("elem_size must be positive")
    if rank < 0 or rank >= mesh_size:
        raise ValueError("rank must be in [0, mesh_size)")

    shape = tuple(shape)
    ndim = len(shape)
    if ndim == 0:
        raise ValueError("reduce_scatter expects at least one dimension")

    if axis < 0:
        axis += ndim
    if not 0 <= axis < ndim:
        raise IndexError(f"axis {axis} out of range for tensor with {ndim} dims")

    axis_len = shape[axis]
    ensure_divisible(axis_len, mesh_size, axis)
    per_rank = axis_len // mesh_size
    if per_rank == 0:
        raise ValueError("reduce_scatter requires non-empty slices along the reduce axis")
    if transfer_kind not in {"push", "pull"}:
        raise ValueError("transfer_kind must be 'push' or 'pull'")

    src_strides = row_major_strides(shape)
    dst_shape = list(shape)
    dst_shape[axis] = per_rank
    dst_shape = tuple(dst_shape)
    dst_strides = row_major_strides(dst_shape)

    plan = DevicePlan(dev=rank)
    plan.tensors_involved[src_buffer] = (torch.Size(shape), dtype)
    plan.tensors_involved[dst_buffer] = (torch.Size(dst_shape), dtype)

    full_src_region = BufferRegion.from_slices(
        elem_size=elem_size,
        strides=src_strides,
        slices=[(0, dim_len) for dim_len in shape],
    )
    plan.local_regions.setdefault(src_buffer, []).append(full_src_region)

    def src_chunk_region(chunk_idx: int) -> BufferRegion:
        slices = []
        for dim, dim_len in enumerate(shape):
            if dim == axis:
                start = chunk_idx * per_rank
                slices.append((start, start + per_rank))
            else:
                slices.append((0, dim_len))
        return BufferRegion.from_slices(
            elem_size=elem_size,
            strides=src_strides,
            slices=slices,
        )

    dst_region = BufferRegion.from_slices(
        elem_size=elem_size,
        strides=dst_strides,
        slices=[(0, dim_len) for dim_len in dst_shape],
    )

    local_chunk_region = src_chunk_region(rank)

    if transfer_kind == "push":
        for i in range(mesh_size - 1):
            peer = (rank + i + 1) % mesh_size
            src_region = src_chunk_region(peer)
            dependency = [
                ComputeDependency(
                    compute_rank=rank,
                    src_region=src_region,
                )
            ]

            plan.add_op(
                Transfer(
                    op=TransferOp.PUSH,
                    src_buf=src_buffer,
                    src_region=src_region,
                    dst_buf=dst_buffer,
                    dst_region=dst_region,
                    chunk_idx=i,
                    peer=peer,
                    dependency=dependency,
                    reduce_op=reduce_op,
                )
            )
    else:
        for i in range(mesh_size - 1):
            peer = (rank + i + 1) % mesh_size
            dependency = [
                ComputeDependency(
                    compute_rank=peer,
                    src_region=local_chunk_region,
                )
            ]
            plan.add_op(
                Transfer(
                    op=TransferOp.PULL,
                    src_buf=src_buffer,
                    src_region=local_chunk_region,
                    dst_buf=dst_buffer,
                    dst_region=dst_region,
                    chunk_idx=i,
                    peer=peer,
                    dependency=dependency,
                    reduce_op=reduce_op,
                )
            )

    dependency = [
        ComputeDependency(
            compute_rank=rank,
            src_region=local_chunk_region,
        )
    ]
    plan.add_op(
        Transfer(
            op=TransferOp.LOCAL_COPY,
            src_buf=src_buffer,
            src_region=local_chunk_region,
            dst_buf=dst_buffer,
            dst_region=dst_region,
            chunk_idx=mesh_size - 1,
            dependency=dependency,
            reduce_op=reduce_op,
        )
    )

    return plan


__all__ = ["build_reduce_scatter_collective_plan"]
