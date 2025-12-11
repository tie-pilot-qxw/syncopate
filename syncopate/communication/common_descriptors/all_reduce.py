"""All-reduce plan builder."""

from __future__ import annotations

from typing import Sequence

import torch

from ..descriptor import BufferRegion, Collective, CollectiveOp, ComputeDependency, DevicePlan
from .utils import ensure_divisible, row_major_strides


def build_all_reduce_plan(
    shape: Sequence[int],
    *,
    dtype: torch.dtype,
    mesh_size: int,
    rank: int,
    buffer_name: str,
    num_all_reduces: int = 1,
    split_axis: int | None = None,
    compute_producer: bool = False,
) -> DevicePlan:
    """Build an all-reduce plan using collective ops (in-place).

    Args:
        shape: Logical tensor shape visible on the current rank.
        dtype: Element dtype.
        mesh_size: Number of participating devices.
        rank: Current device rank.
        buffer_name: Name of the buffer containing local contributions and
            receiving the reduced result (in-place).
        num_all_reduces: Number of sub-collectives along ``split_axis``.
        split_axis: Dimension to shard into multiple collective calls. If
            ``None``, ``num_all_reduces`` must be 1.
        compute_producer: If ``True``, add a compute dependency so comm waits for
            the local producer to finish before launching the collective.
    """

    elem_size = dtype.itemsize
    if mesh_size <= 0:
        raise ValueError("mesh_size must be positive")
    if elem_size <= 0:
        raise ValueError("elem_size must be positive")
    if num_all_reduces <= 0:
        raise ValueError("num_all_reduces must be positive")

    shape = tuple(shape)
    ndim = len(shape)
    if ndim == 0:
        raise ValueError("all_reduce expects at least one dimension")

    if split_axis is None:
        if num_all_reduces != 1:
            raise ValueError("split_axis is required when num_all_reduces > 1")
        split_axis = 0

    if split_axis < 0:
        split_axis += ndim
    if not 0 <= split_axis < ndim:
        raise IndexError(f"split_axis {split_axis} out of range for tensor with {ndim} dims")

    split_len = shape[split_axis]
    ensure_divisible(split_len, num_all_reduces, "num_all_reduces")
    split_chunk = split_len // num_all_reduces
    if split_chunk == 0:
        raise ValueError("split produces empty chunks")

    strides = row_major_strides(shape)
    plan = DevicePlan(dev=rank)
    plan.tensors_involved[buffer_name] = (torch.Size(shape), dtype)

    full_region = BufferRegion.from_slices(
        elem_size=elem_size,
        strides=strides,
        slices=[(0, dim) for dim in shape],
    )
    plan.local_regions.setdefault(buffer_name, []).append(full_region)

    def make_region(split_start: int, split_end: int) -> BufferRegion:
        slices = []
        for dim, dim_len in enumerate(shape):
            if dim == split_axis:
                slices.append((split_start, split_end))
            else:
                slices.append((0, dim_len))
        return BufferRegion.from_slices(
            elem_size=elem_size,
            strides=strides,
            slices=slices,
        )

    for split_idx in range(num_all_reduces):
        split_start = split_idx * split_chunk
        split_end = split_start + split_chunk

        region = make_region(split_start, split_end)

        dependency = []
        if compute_producer:
            dependency.append(ComputeDependency(compute_rank=rank, src_region=region))

        plan.add_op(
            Collective(
                op=CollectiveOp.ALL_REDUCE,
                src_buf=buffer_name,
                src_region=region,
                dst_buf=buffer_name,
                dst_region=region,
                dependency=dependency,
            )
        )

    return plan

def build_all_reduce_nvsharp_plan(
    shape: Sequence[int],
    *,
    dtype: torch.dtype,
    mesh_size: int,
    rank: int,
    src_name: str,
    dst_name: str,
    num_all_reduces: int = 1,
    split_axis: int | None = None,
    compute_producer: bool = False,
) -> DevicePlan:
    """Build an all-reduce plan using collective ops (in-place).

    Args:
        shape: Logical tensor shape visible on the current rank.
        dtype: Element dtype.
        mesh_size: Number of participating devices.
        rank: Current device rank.
        buffer_name: Name of the buffer containing local contributions and
            receiving the reduced result (in-place).
        num_all_reduces: Number of sub-collectives along ``split_axis``.
        split_axis: Dimension to shard into multiple collective calls. If
            ``None``, ``num_all_reduces`` must be 1.
        compute_producer: If ``True``, add a compute dependency so comm waits for
            the local producer to finish before launching the collective.
    """

    elem_size = dtype.itemsize
    if mesh_size <= 0:
        raise ValueError("mesh_size must be positive")
    if elem_size <= 0:
        raise ValueError("elem_size must be positive")
    if num_all_reduces <= 0:
        raise ValueError("num_all_reduces must be positive")

    shape = tuple(shape)
    ndim = len(shape)
    if ndim == 0:
        raise ValueError("all_reduce expects at least one dimension")

    if split_axis is None:
        if num_all_reduces != 1:
            raise ValueError("split_axis is required when num_all_reduces > 1")
        split_axis = 0

    if split_axis < 0:
        split_axis += ndim
    if not 0 <= split_axis < ndim:
        raise IndexError(f"split_axis {split_axis} out of range for tensor with {ndim} dims")

    split_len = shape[split_axis]
    ensure_divisible(split_len, num_all_reduces, "num_all_reduces")
    split_chunk = split_len // num_all_reduces
    if split_chunk == 0:
        raise ValueError("split produces empty chunks")

    strides = row_major_strides(shape)
    plan = DevicePlan(dev=rank)
    plan.tensors_involved[src_name] = (torch.Size(shape), dtype)
    plan.tensors_involved[dst_name] = (torch.Size(shape), dtype)

    def make_region(split_start: int, split_end: int) -> BufferRegion:
        slices = []
        for dim, dim_len in enumerate(shape):
            if dim == split_axis:
                slices.append((split_start, split_end))
            else:
                slices.append((0, dim_len))
        return BufferRegion.from_slices(
            elem_size=elem_size,
            strides=strides,
            slices=slices,
        )

    for split_idx in range(num_all_reduces):
        split_start = split_idx * split_chunk
        split_end = split_start + split_chunk

        region = make_region(split_start, split_end)

        dependency = []
        if compute_producer:
            dependency.append(ComputeDependency(compute_rank=rank, src_region=region))

        plan.add_op(
            Collective(
                op=CollectiveOp.ALL_REDUCE_NVSHARP,
                src_buf=src_name,
                src_region=region,
                dst_buf=dst_name,
                dst_region=region,
                dependency=dependency,
            )
        )

    return plan

__all__ = ["build_all_reduce_plan", "build_all_reduce_nvsharp_plan"]
