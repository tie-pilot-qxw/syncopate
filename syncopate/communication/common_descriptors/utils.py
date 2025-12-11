"""Shared helpers for collective plan builders."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence


def ensure_divisible(length: int, mesh_size: int, axis: str | int) -> None:
    if length % mesh_size:
        raise ValueError(
            f"axis '{axis}' length {length} must be divisible by mesh size {mesh_size}"
        )


def lookup_mesh_size(mesh: Dict[str, int], mesh_axis: str) -> int:
    try:
        mesh_size = mesh[mesh_axis]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"mesh axis '{mesh_axis}' not found in mesh") from exc
    if mesh_size <= 0:
        raise ValueError(f"mesh axis '{mesh_axis}' must be positive")
    return mesh_size


def row_major_strides(shape: Sequence[int]) -> Sequence[int]:
    stride = 1
    strides = [0] * len(shape)
    for idx in reversed(range(len(shape))):
        strides[idx] = stride
        stride *= shape[idx]
    return strides


def compute_offset(strides: Sequence[int], starts: Mapping[int, int]) -> int:
    return sum(starts.get(axis, 0) * strides[axis] for axis in range(len(strides)))


def block_nbytes(
    shape: Sequence[int],
    elem_size: int,
    lengths: Mapping[int, int],
) -> int:
    elems = 1
    for axis, size in enumerate(shape):
        elems *= lengths.get(axis, size)
    return elems * elem_size


__all__ = [
    "block_nbytes",
    "compute_offset",
    "ensure_divisible",
    "lookup_mesh_size",
    "row_major_strides",
]
