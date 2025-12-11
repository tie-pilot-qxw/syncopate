from typing import Optional
import triton
import triton.language as tl
import torch

@triton.jit
def get_pid_mn(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M: tl.constexpr):
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n

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
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
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
        shape=[N, M],
        strides=[M, 1],
        block_shape=[
            BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2,
            BLOCK_SIZE_M,
        ],
    )

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS # @sy.tile_id persistent
    ki = -1

    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0

    # @sy.persistent_init begin
    # @sy.persistent_init end

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS

            # @sy.dispatch begin

            # @sy.pid_map M=pid_m N=pid_n
            pid_m, pid_n = get_pid_mn(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M)

            # @sy.dispatch end

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N


        offs_k = ki * BLOCK_SIZE_K

        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            if EPILOGUE_SUBTILE:
                acc = tl.reshape(accumulator,
                                 (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
                acc = tl.permute(acc, (2, 0, 1))
                acc0, acc1 = tl.split(acc)
                c0 = acc0.to(dtype)
                c_desc.store([offs_bn, offs_am], c0)
                c1 = acc1.to(dtype)
                c_desc.store([offs_bn + BLOCK_SIZE_N // 2, offs_am], c1)
            else:
                acc = tl.permute(accumulator, (1, 0))
                c = acc.to(dtype)
                c_desc.store([offs_bn, offs_am], c)
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
    N = 4096
    K = 8192
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((N, K), device='cuda', dtype=torch.float16)
    c = torch.zeros((N, M), device='cuda', dtype=torch.float16)


    gemm(
        a,
        b,
        c,
        1,  # world_size
        132,  # num_gemm_sms
    )

    golden = torch.matmul(a, b.T)
    torch.testing.assert_close(c.T, golden, rtol=1e-2, atol=1e-2)

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
