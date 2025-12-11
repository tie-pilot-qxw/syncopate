import math
from typing import Iterable, Sequence, Tuple

import torch
from syncopate.communication.descriptor import ReduceOp
import triton
import triton.language as tl

MAX_RANK = 5


def _product(vals: Iterable[int]) -> int:
    p = 1
    for v in vals:
        p *= int(v)
    return p


def _pad_shape_strides(
    shape: Sequence[int],
    strides: Sequence[int],
    *,
    elem_bytes: int,
    target_rank: int = MAX_RANK,
) -> tuple[list[int], list[int]]:
    if len(shape) != len(strides):
        raise ValueError("shape/stride rank mismatch")
    if len(shape) > target_rank:
        raise ValueError(f"rank > {target_rank} not supported")

    pad = target_rank - len(shape)
    align_elems = max(1, math.ceil(16 / elem_bytes))
    pad_stride = max(align_elems, 2)

    shape_p = [1] * pad + [int(x) for x in shape]
    stride_p = [pad_stride] * pad + [int(x) for x in strides]
    return shape_p, stride_p


def squeeze_common_dense_suffix(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Squeeze (merge) the largest *common* dense (contiguous) suffix dims of a and b into 1 dim.
    Requirements for a suffix dim to be merged:
      - sizes match between a and b for each merged dim
      - both tensors are dense-contiguous over that suffix (stride pattern == contiguous), ignoring size==1 dims
      - rejects expand/broadcast (stride==0 with size>1)
    """
    da, db = a.dim(), b.dim()
    if da <= 1 or db <= 1:
        return a, b

    n = min(da, db)
    sa, sb = list(a.shape), list(b.shape)
    stra, strb = list(a.stride()), list(b.stride())

    r = None
    for k in range(n):
        ia, ib = da - 1 - k, db - 1 - k
        if sa[ia] != sb[ib]:
            break
        if sa[ia] > 1:
            if stra[ia] == 1 and strb[ib] == 1:
                r = k
            break

    if r is None:
        return a, b

    expected = 1
    start_k = 0
    for k in range(n):
        ia, ib = da - 1 - k, db - 1 - k
        if sa[ia] != sb[ib]:
            break

        size = sa[ia]
        if size == 1:
            start_k = k
            continue

        if stra[ia] == 0 or strb[ib] == 0:
            break

        if stra[ia] != expected or strb[ib] != expected:
            break

        start_k = k
        expected *= size

    start_a = da - 1 - start_k
    start_b = db - 1 - start_k

    if start_a == da - 1:
        return a, b

    merged = math.prod(sa[start_a:])
    new_shape_a = sa[:start_a] + [merged]
    new_shape_b = sb[:start_b] + [merged]

    new_stride_a = list(stra[:start_a]) + [1]
    new_stride_b = list(strb[:start_b]) + [1]

    a2 = a.as_strided(new_shape_a, new_stride_a, a.storage_offset())
    b2 = b.as_strided(new_shape_b, new_stride_b, b.storage_offset())
    return a2, b2


@triton.jit
def _naive_multi_dim_persistent_copy_last_contig_kernel(
    src_ptr,
    dst_ptr,
    dim0: tl.constexpr,
    dim1: tl.constexpr,
    dim2: tl.constexpr,
    dim3: tl.constexpr,
    dim4: tl.constexpr,
    stride0_src,
    stride1_src,
    stride2_src,
    stride3_src,
    stride4_src,
    stride0_dst,
    stride1_dst,
    stride2_dst,
    stride3_dst,
    stride4_dst,
    BLOCK_0: tl.constexpr,
    BLOCK_1: tl.constexpr,
    BLOCK_2: tl.constexpr,
    BLOCK_3: tl.constexpr,
    BLOCK_4: tl.constexpr,
    NUM_SMS: tl.constexpr,
    reduce_op_id: tl.constexpr,
):
    pid = tl.program_id(0)

    num_pids_0 = tl.cdiv(dim0, BLOCK_0)
    num_pids_1 = tl.cdiv(dim1, BLOCK_1)
    num_pids_2 = tl.cdiv(dim2, BLOCK_2)
    num_pids_3 = tl.cdiv(dim3, BLOCK_3)
    num_pids_4 = tl.cdiv(dim4, BLOCK_4)

    num_pid_34 = num_pids_3 * num_pids_4
    num_pid_234 = num_pids_2 * num_pid_34
    num_pid_1234 = num_pids_1 * num_pid_234

    total_tiles = num_pids_0 * num_pid_1234
    tile_id = pid

    while tile_id < total_tiles:
        tile_id_0 = tile_id // num_pid_1234
        rem0 = tile_id - tile_id_0 * num_pid_1234
        tile_id_1 = rem0 // num_pid_234
        rem1 = rem0 - tile_id_1 * num_pid_234
        tile_id_2 = rem1 // num_pid_34
        rem2 = rem1 - tile_id_2 * num_pid_34
        tile_id_3 = rem2 // num_pids_4
        tile_id_4 = rem2 - tile_id_3 * num_pids_4

        offs3 = tile_id_3 * BLOCK_3 + tl.arange(0, BLOCK_3)
        offs4 = tile_id_4 * BLOCK_4 + tl.arange(0, BLOCK_4)
        mask3 = offs3 < dim3
        mask4 = offs4 < dim4
        mask = mask3[:, None] & mask4[None, :]

        base_src = (
            tile_id_0 * BLOCK_0 * stride0_src
            + tile_id_1 * BLOCK_1 * stride1_src
            + tile_id_2 * BLOCK_2 * stride2_src
        )
        base_dst = (
            tile_id_0 * BLOCK_0 * stride0_dst
            + tile_id_1 * BLOCK_1 * stride1_dst
            + tile_id_2 * BLOCK_2 * stride2_dst
        )

        src_idx = base_src + offs3[:, None] * stride3_src + offs4[None, :] * stride4_src
        dst_idx = base_dst + offs3[:, None] * stride3_dst + offs4[None, :] * stride4_dst

        tile = tl.load(src_ptr + src_idx, mask=mask, other=0)
        if reduce_op_id == 0:
            tl.store(dst_ptr + dst_idx, tile, mask=mask)
        elif reduce_op_id == 1:
            tl.atomic_add(dst_ptr + dst_idx, tile, mask=mask)
        else:
            tl.static_assert(False, "unsupported reduce_op_id")

        tile_id += NUM_SMS


def persistent_naive_copy_last_contig(
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    num_sms: int | None = None,
    fallback_to_torch_copy: bool = False,
    stages: int = 1,
    reduce_op: ReduceOp | None = None,
) -> torch.Tensor:
    """
    Persistent copy using standard load/store instead of TMA.
    Matches the interface and semantics of `persistent_tma_copy_last_contig`.
    """
    # remove the leading dimensions with size 1 (no effect on contiguity or TMA)
    def _leading_one_dims(t: torch.Tensor) -> torch.Tensor:
        squeeze_dims = []
        for i in range(t.dim() - 1): # skip last dim
            if t.shape[i] != 1:
                break
            squeeze_dims.append(i)
        return t.squeeze(dim=squeeze_dims)
    src = _leading_one_dims(src)
    dst = _leading_one_dims(dst)
    if not (src.is_cuda and dst.is_cuda):
        raise ValueError("CUDA tensors only")
    if src.device != dst.device:
        raise ValueError("src/dst must be on same device")
    if src.shape != dst.shape:
        raise ValueError("src/dst must have same shape")
    if src.dtype != dst.dtype:
        raise ValueError("src/dst must have same dtype")
    if src.dim() > MAX_RANK:
        raise ValueError(f"rank > {MAX_RANK} not supported")

    src, dst = squeeze_common_dense_suffix(src, dst)

    if src.stride()[-1] != 1 or dst.stride()[-1] != 1:
        if fallback_to_torch_copy:
            dst.copy_(src)
            return dst
        raise ValueError(f"require stride[-1]==1. got src {src.stride()} dst {dst.stride()}")

    if reduce_op is not None and reduce_op != ReduceOp.SUM:
        raise ValueError(f"unsupported reduce_op {reduce_op}")

    elem_bytes = src.element_size()
    shared_memory_limit = torch.cuda.get_device_properties(src.device).shared_memory_per_multiprocessor

    block_n = 1
    while block_n * 2 * elem_bytes * stages < shared_memory_limit and block_n < src.shape[-1]:
        block_n *= 2
    block_m = 1
    if src.dim() >= 2:
        while block_m * block_n * 2 * elem_bytes * stages < shared_memory_limit and block_m < src.shape[-2]:
            block_m *= 2

    align_elems = max(1, math.ceil(16 / elem_bytes))
    block_n = max(int(block_n), align_elems)

    if num_sms is None:
        num_sms = torch.cuda.get_device_properties(src.device).multi_processor_count
    num_sms = int(num_sms)
    if num_sms <= 0:
        raise ValueError("num_sms must be positive")

    shape_p, src_stride_p = _pad_shape_strides(src.shape, src.stride(), elem_bytes=elem_bytes, target_rank=MAX_RANK)
    _shape_p2, dst_stride_p = _pad_shape_strides(dst.shape, dst.stride(), elem_bytes=elem_bytes, target_rank=MAX_RANK)
    assert shape_p == _shape_p2

    outer_shape = shape_p[:4]
    inner_dim = shape_p[4]

    outer_tiles = _product(outer_shape)
    inner_tiles = math.ceil(inner_dim / block_n)
    total_tiles = outer_tiles * inner_tiles
    if total_tiles == 0:
        return dst

    grid = (min(num_sms, total_tiles),)

    if reduce_op is None:
        reduce_op_id = 0
    else:
        reduce_op_id = reduce_op.value

    _naive_multi_dim_persistent_copy_last_contig_kernel[grid](
        src,
        dst,
        dim0=outer_shape[0],
        dim1=outer_shape[1],
        dim2=outer_shape[2],
        dim3=outer_shape[3],
        dim4=inner_dim,
        stride0_src=src_stride_p[0],
        stride1_src=src_stride_p[1],
        stride2_src=src_stride_p[2],
        stride3_src=src_stride_p[3],
        stride4_src=src_stride_p[4],
        stride0_dst=dst_stride_p[0],
        stride1_dst=dst_stride_p[1],
        stride2_dst=dst_stride_p[2],
        stride3_dst=dst_stride_p[3],
        stride4_dst=dst_stride_p[4],
        BLOCK_0=1,
        BLOCK_1=1,
        BLOCK_2=1,
        BLOCK_3=block_m,
        BLOCK_4=block_n,
        NUM_SMS=num_sms,
        reduce_op_id=reduce_op_id,
        num_warps=32,
        num_stages=stages,
    )
    return dst
