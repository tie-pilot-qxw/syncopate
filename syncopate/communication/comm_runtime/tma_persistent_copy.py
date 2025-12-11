import math
from typing import Iterable, Sequence, Tuple

import torch
from syncopate.communication.descriptor import ReduceOp
import triton
import triton.language as tl

MAX_RANK = 5

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except Exception:
    TensorDescriptor = None


def _product(vals: Iterable[int]) -> int:
    p = 1
    for v in vals:
        p *= int(v)
    return p


def _outer_strides_from_shape(shape: Sequence[int]) -> list[int]:
    # row-major flattening strides for decoding outer_idx
    strides: list[int] = []
    prod = 1
    for size in reversed(shape):
        strides.insert(0, prod)
        prod *= int(size)
    return strides


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
    # pad dims are size-1, stride value doesn't matter for addressing, but must not be 0
    # also keep it 16B aligned (in bytes) to avoid descriptor rules tripping in some builds.
    align_elems = max(1, math.ceil(16 / elem_bytes))
    pad_stride = max(align_elems, 2)

    shape_p = [1] * pad + [int(x) for x in shape]
    stride_p = [pad_stride] * pad + [int(x) for x in strides]
    return shape_p, stride_p


def _tma_stride_ok(shape: Sequence[int], strides: Sequence[int], elem_bytes: int) -> tuple[bool, str]:
    # common TMA constraints: last dim stride must be 1, and other strides (bytes) multiple of 16 when size>1
    if strides[-1] != 1:
        return False, f"last dim stride must be 1, got {strides[-1]}"
    for i in range(len(shape) - 1):
        if int(shape[i]) <= 1:
            continue
        if (int(strides[i]) * elem_bytes) % 16 != 0:
            return False, f"dim {i} stride {strides[i]} elems => {strides[i]*elem_bytes}B not multiple of 16"
    return True, ""


@triton.jit
def _tma_persistent_copy_last_contig_kernel(
    src_desc,
    dst_desc,
    outer_dim0: tl.constexpr,
    outer_dim1: tl.constexpr,
    outer_dim2: tl.constexpr,
    outer_dim3: tl.constexpr,
    outer_stride0: tl.constexpr,
    outer_stride1: tl.constexpr,
    outer_stride2: tl.constexpr,
    outer_stride3: tl.constexpr,
    inner_dim: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)

    outer_tiles = outer_dim0 * outer_dim1 * outer_dim2 * outer_dim3
    inner_tiles = tl.cdiv(inner_dim, BLOCK_N)
    total_tiles = outer_tiles * inner_tiles

    while pid < total_tiles:
        outer_idx = pid // inner_tiles
        tile_col = pid - outer_idx * inner_tiles
        col_start = tile_col * BLOCK_N

        # decode outer_idx -> (i0,i1,i2,i3) using explicit arithmetic (no list indexing)
        i0 = outer_idx // outer_stride0
        rem0 = outer_idx - i0 * outer_stride0

        i1 = rem0 // outer_stride1
        rem1 = rem0 - i1 * outer_stride1

        i2 = rem1 // outer_stride2
        rem2 = rem1 - i2 * outer_stride2

        i3 = rem2  # since outer_stride3 == 1

        idx = [i0, i1, i2, i3, col_start]
        tile = src_desc.load(idx)
        dst_desc.store(idx, tile)

        pid += NUM_SMS

@triton.jit
def _tma_multi_dim_persistent_copy_last_contig_kernel(
    src_desc,
    dst_desc,
    dim0: tl.constexpr,
    dim1: tl.constexpr,
    dim2: tl.constexpr,
    dim3: tl.constexpr,
    dim4: tl.constexpr,
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

        idx = [tile_id_0 * BLOCK_0,
               tile_id_1 * BLOCK_1,
               tile_id_2 * BLOCK_2,
               tile_id_3 * BLOCK_3,
               tile_id_4 * BLOCK_4]
        
        tile = src_desc.load(idx)
        if reduce_op_id == 0:
            # simple copy
            dst_desc.store(idx, tile)
        elif reduce_op_id == 1:
            dst_desc.atomic_add(idx, tile)
        else:
            tl.static_assert(False, "unsupported reduce_op_id")
            
        tile_id += NUM_SMS

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

    # we only consider aligned suffix (right-aligned) dims
    n = min(da, db)
    sa, sb = list(a.shape), list(b.shape)
    stra, strb = list(a.stride()), list(b.stride())

    # work on suffix indices in each tensor
    # ia = da-1-k, ib = db-1-k for k in [0..n-1]
    # find rightmost size>1 dim that is common and dense base (stride==1 for both)
    r = None
    for k in range(n):
        ia, ib = da - 1 - k, db - 1 - k
        if sa[ia] != sb[ib]:
            break
        if sa[ia] > 1:
            if stra[ia] == 1 and strb[ib] == 1:  # dense base
                r = k
            break

    if r is None:
        # either mismatch at very end, or only trailing size==1 dims, or not dense
        return a, b

    expected = 1
    start_k = 0  # number of suffix dims included so far (in k-space)
    for k in range(n):  # from last dim upward
        ia, ib = da - 1 - k, db - 1 - k
        if sa[ia] != sb[ib]:
            break

        size = sa[ia]
        if size == 1:
            start_k = k  # can always include size-1 dims
            continue

        if stra[ia] == 0 or strb[ib] == 0:  # expand/broadcast => not dense
            break

        if stra[ia] != expected or strb[ib] != expected:
            break

        start_k = k
        expected *= size

    # start index in each tensor
    start_a = da - 1 - start_k
    start_b = db - 1 - start_k

    # if suffix length == 1, nothing to merge
    if start_a == da - 1:
        return a, b

    merged = math.prod(sa[start_a:])  # same as math.prod(sb[start_b:])
    new_shape_a = sa[:start_a] + [merged]
    new_shape_b = sb[:start_b] + [merged]

    new_stride_a = list(stra[:start_a]) + [1]
    new_stride_b = list(strb[:start_b]) + [1]

    a2 = a.as_strided(new_shape_a, new_stride_a, a.storage_offset())
    b2 = b.as_strided(new_shape_b, new_stride_b, b.storage_offset())
    return a2, b2

def persistent_tma_copy_last_contig(
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    num_sms: int | None = None,
    fallback_to_torch_copy: bool = False,
    stages: int = 1,
    reduce_op: ReduceOp | None = None,
) -> torch.Tensor:
    """
    Simplified TMA copy:
      - REQUIRE: last dimension is contiguous for BOTH src and dst (stride[-1] == 1).
      - Higher dims can be strided (non-contig), but must satisfy common TMA alignment constraints.

    This uses host-side TensorDescriptor (stable) and a persistent loop (pid += NUM_SMS).
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
        raise ValueError(f"src/dst must have same shape, got src {src.shape} dst {dst.shape}")
    if src.dtype != dst.dtype:
        raise ValueError("src/dst must have same dtype")
    if src.dim() > MAX_RANK:
        raise ValueError(f"rank > {MAX_RANK} not supported")
    if TensorDescriptor is None:
        if fallback_to_torch_copy:
            dst.copy_(src)
            return dst
        raise RuntimeError("TensorDescriptor not available in this Triton build")

    src, dst = squeeze_common_dense_suffix(src, dst)
    shared_memory_limit = torch.cuda.get_device_properties(src.device).shared_memory_per_multiprocessor

    block_n = 1
    while block_n * 2 * src.element_size() * stages < shared_memory_limit and block_n  < src.shape[-1]:
        block_n *= 2
    block_m = 1
    if src.dim() >= 2:
        while block_m * block_n * 2 * src.element_size() * stages < shared_memory_limit and block_m < src.shape[-2]:
            block_m *= 2
    elem_bytes = src.element_size()
    align_elems = max(1, math.ceil(16 / elem_bytes))
    block_n = max(int(block_n), align_elems)
    if block_n * elem_bytes < 16:
        raise ValueError("block_n must cover >= 16 bytes")

    # last dim must be contiguous
    if src.stride()[-1] != 1 or dst.stride()[-1] != 1:
        if fallback_to_torch_copy:
            dst.copy_(src)
            return dst
        raise ValueError(f"require stride[-1]==1. got src {src.stride()} dst {dst.stride()}")

    # base pointer alignment (common TMA requirement)
    if (src.data_ptr() % 16) != 0 or (dst.data_ptr() % 16) != 0:
        if fallback_to_torch_copy:
            dst.copy_(src)
            return dst
        raise ValueError("require 16B-aligned data_ptr for TMA")

    if num_sms is None:
        num_sms = torch.cuda.get_device_properties(src.device).multi_processor_count
    num_sms = int(num_sms)
    if num_sms <= 0:
        raise ValueError("num_sms must be positive")

    # pad to 5D descriptor (leading size-1 dims)
    shape_p, src_stride_p = _pad_shape_strides(src.shape, src.stride(), elem_bytes=elem_bytes, target_rank=MAX_RANK)
    _shape_p2, dst_stride_p = _pad_shape_strides(dst.shape, dst.stride(), elem_bytes=elem_bytes, target_rank=MAX_RANK)
    assert shape_p == _shape_p2

    ok_s, msg_s = _tma_stride_ok(shape_p, src_stride_p, elem_bytes)
    ok_d, msg_d = _tma_stride_ok(shape_p, dst_stride_p, elem_bytes)
    if not (ok_s and ok_d):
        if fallback_to_torch_copy:
            dst.copy_(src)
            return dst
        raise ValueError(f"TMA stride constraint fail: src({msg_s}) dst({msg_d})")

    # allocator for descriptor backing storage
    def alloc_fn(size: int, alignment: int, stream: int | None):
        return torch.empty((size,), device=src.device, dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    # build descriptor views (5D)
    src_view = src.as_strided(size=shape_p, stride=src_stride_p)
    dst_view = dst.as_strided(size=shape_p, stride=dst_stride_p)

    blk = [1, 1, 1, block_m, block_n]
    src_desc = TensorDescriptor.from_tensor(src_view, blk)
    dst_desc = TensorDescriptor.from_tensor(dst_view, blk)

    outer_shape = shape_p[:4]
    inner_dim = shape_p[4]

    outer_strides = _outer_strides_from_shape(outer_shape)
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

    _tma_multi_dim_persistent_copy_last_contig_kernel[grid](
        src_desc,
        dst_desc,
        dim0=outer_shape[0],
        dim1=outer_shape[1],
        dim2=outer_shape[2],
        dim3=outer_shape[3],
        dim4=inner_dim,
        BLOCK_0=1,
        BLOCK_1=1,
        BLOCK_2=1,
        BLOCK_3=block_m,
        BLOCK_4=block_n,
        NUM_SMS=num_sms,
        reduce_op_id=reduce_op_id,
    )
    # _tma_persistent_copy_last_contig_kernel[grid](
    #     src_desc,
    #     dst_desc,
    #     outer_dim0=outer_shape[0],
    #     outer_dim1=outer_shape[1],
    #     outer_dim2=outer_shape[2],
    #     outer_dim3=outer_shape[3],
    #     outer_stride0=outer_strides[0],
    #     outer_stride1=outer_strides[1],
    #     outer_stride2=outer_strides[2],
    #     outer_stride3=outer_strides[3],
    #     inner_dim=inner_dim,
    #     BLOCK_N=block_n,
    #     NUM_SMS=num_sms,
    #     num_warps=4,
    #     num_stages=stages,
    # )
    return dst


if __name__ == "__main__":
    torch.manual_seed(0)

    # A case where last dim is contiguous but higher dims are strided:
    # take every other row => stride[0] changes, stride[-1] stays 1.
    a = torch.arange(8 * 6, device="cuda", dtype=torch.float64).reshape(8, 6)
    src = a[::2, :]  # shape (4,6), stride (12,1)
    dst_base = torch.empty_like(a)
    dst = dst_base[::2, :]  # same shape/stride (12,1)

    persistent_tma_copy_last_contig(src, dst, num_sms=4,fallback_to_torch_copy=False)
    torch.testing.assert_close(dst, src)

    # 3D: stride in dim0 but last dim still contiguous
    b = torch.arange(10 * 3 * 4, device="cuda", dtype=torch.float64).reshape(10, 3, 4)
    src3 = b[::2, :, :]  # shape (5,3,4), stride (24,4,1)
    dstb = torch.empty_like(b)
    dst3 = dstb[::2, :, :]

    persistent_tma_copy_last_contig(src3, dst3, num_sms=4, fallback_to_torch_copy=False)
    torch.testing.assert_close(dst3, src3)

    print("OK")
