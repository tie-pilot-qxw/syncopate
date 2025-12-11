from typing import Optional
import triton
import triton.language as tl
import torch

@triton.jit
def get_pid_mnk(tile_id, num_pid_m, num_pid_n, num_pid_k, GROUP_SIZE_M: tl.constexpr):
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n, 0, num_pid_k

################### triton kernel ###################
@triton.jit
def kernel_gemm(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M) # @sy.axis_count M block=BLOCK_SIZE_M
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N) # @sy.axis_count N block=BLOCK_SIZE_N
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K) # @sy.axis_count K=range block=BLOCK_SIZE_K
    num_tiles = num_pid_m * num_pid_n # @sy.num_tiles

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[
            BLOCK_SIZE_M,
            BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2,
        ],
    )

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS # @sy.tile_id persistent
    ki = -1

    pid_m = 0
    pid_n = 0
    pid_k_lo = 0
    pid_k_hi = 0
    offs_am = 0
    offs_bn = 0

    # @sy.persistent_init begin
    # @sy.persistent_init end

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, tiles_per_SM):
        tile_id += NUM_SMS

        # @sy.dispatch begin

        # @sy.pid_map M=pid_m N=pid_n K=pid_k_lo..pid_k_hi
        pid_m, pid_n, pid_k_lo, pid_k_hi = get_pid_mnk(tile_id, num_pid_m, num_pid_n, num_pid_k, GROUP_SIZE_M)

        # @sy.dispatch end

        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        for ki in range(pid_k_lo, pid_k_hi):
            offs_k = ki * BLOCK_SIZE_K

            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator,
                                (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc.atomic_add([offs_am, offs_bn], c0)
            c1 = acc1.to(dtype)
            c_desc.atomic_add([offs_am, offs_bn + BLOCK_SIZE_N // 2], c1)
        else:
            c = accumulator.to(dtype)
            c_desc.atomic_add([offs_am, offs_bn], c)
        # @sy.producer_epilogue
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N),
                                dtype=tl.float32)


def gemm(a, # @sy.host_entry
                    b,
                    c,
                    world_size,
                    num_gemm_sms,
                    BLOCK_SIZE_M=128,
                    BLOCK_SIZE_N=256,
                    BLOCK_SIZE_K=64,
                    GROUP_SIZE_M=8,
                    STAGES=3):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, local_K = a.shape
    N, local_K = b.shape

    M_per_rank = M // world_size

    assert M_per_rank % BLOCK_SIZE_M == 0

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = lambda META: (min(
        num_gemm_sms,
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(
            N, META["BLOCK_SIZE_N"]),
    ), )

    # Launch the Triton GEMM kernel. Once the kernel has completed the computation of the output tiles
    # that send to a specific rank, will set the corresponding barrier to 1.
    # @sy.kernel_launch
    compiled = kernel_gemm[grid](
        a,
        b,
        c,
        M,
        N,
        local_K,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M,
        False,
        NUM_SMS=num_gemm_sms,  #
        num_stages=STAGES,
        num_warps=8,
    )

if __name__ == "__main__":
    M = 8192
    N = 8192
    K = 8192
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((N, K), device='cuda', dtype=torch.float16)
    c = torch.zeros((M, N), device='cuda', dtype=torch.float16)


    gemm(
        a,
        b,
        c,
        1,  # world_size
        132,  # num_gemm_sms
    )

    golden = torch.matmul(a, b.T)
    torch.testing.assert_close(c, golden, rtol=1e-2, atol=1e-2)

    ms = triton.testing.do_bench(
        lambda: torch.matmul(a, b.T),
        rep=2000,
        warmup=200,
    )

    ms_triton = triton.testing.do_bench(
        lambda: gemm(
            a,
            b,
            c,
            1,  # world_size
            132,  # num_gemm_sms
        ),
        rep=2000,
        warmup=200,
    )

    cal_tflops = lambda ms: 2 * M * N * K / (ms / 1000) / 1e12
    print(
        f"triton gemm: {ms_triton:.2f}ms, {cal_tflops(ms_triton):.2f} TFlops, speedup {ms/ms_triton:.2f}x over torch.matmul {ms:.2f}ms, {cal_tflops(ms):.2f} TFlops"
    )
