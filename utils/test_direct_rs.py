import dataclasses
import os
from functools import partial
from typing import List, Optional

import torch

import triton
import triton.language as tl
import triton_dist.language as dl
from triton_dist.utils import (dist_print, generate_data,
                               nvshmem_barrier_all_on_stream,
                               nvshmem_create_tensor,
                               nvshmem_free_tensor_sync, perf_func,
                               finalize_distributed, initialize_distributed)


################### triton kernel ###################
@triton.jit
def kernel_gemm_rs_producer_persistent(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    local_world_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """
    The 'kernel_gemm_rs_producer_persistent' kernel is almost identical to a regular Triton GEMM kernel, with only two minor differences:
    1. The computation order of tiles is swizzled according to the rank.
    2. There is an additional operation to set the barrier in the epilogue.
    """
    rank = dl.rank()
    num_ranks = dl.num_ranks()
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    node_id = rank // local_world_size
    nnodes = num_ranks // local_world_size
    M_per_rank = M // num_ranks

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

    start_rank = (rank + 1) % local_world_size
    last_rank = start_rank
    remote_c_ptr = dl.symm_at(c_ptr, start_rank)
    c_desc = tl.make_tensor_descriptor(
        remote_c_ptr,
        shape=[M_per_rank, N],
        strides=[N, 1],
        block_shape=[
            BLOCK_SIZE_M,
            BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2,
        ],
    )

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0
    offs_cm = 0

    # M_per_rank % BLOCK_SIZE_M == 0 is guaranteed by the caller
    num_pid_m_per_rank = M_per_rank // BLOCK_SIZE_M

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            m_rank = pid_m // num_pid_m_per_rank
            pid_m_intra_rank = pid_m - m_rank * num_pid_m_per_rank
            """
            Difference 1: Based on the m dimension, calculate the target rank where the output data will be scattered to.
            Then, perform a swizzle operation according to the local rank and the node_id of the current GPU.
            This ensures that during communication, the data sent and received by each rank is balanced, maximizing the utilization of all communication bandwidth.
            """
            # original rank and node_id
            swizzle_m_local_rank = (m_rank + rank + 1) % local_world_size
            swizzle_m_rank = swizzle_m_local_rank

            if swizzle_m_rank != last_rank:
                # a new remote c_desc
                last_rank = swizzle_m_rank
                remote_c_ptr = dl.symm_at(c_ptr, swizzle_m_rank)
                c_desc = tl.make_tensor_descriptor(
                    remote_c_ptr,
                    shape=[M_per_rank, N],
                    strides=[N, 1],
                    block_shape=[
                        BLOCK_SIZE_M,
                        BLOCK_SIZE_N
                        if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2,
                    ],
                )

            # perform swizzle
            pid_m = swizzle_m_rank * num_pid_m_per_rank + pid_m_intra_rank

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N
            offs_cm = pid_m_intra_rank * BLOCK_SIZE_M

        offs_k = ki * BLOCK_SIZE_K

        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            if EPILOGUE_SUBTILE:
                acc = tl.reshape(accumulator,
                                 (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
                acc = tl.permute(acc, (0, 2, 1))
                acc0, acc1 = tl.split(acc)
                c0 = acc0.to(dtype)
                c_desc.atomic_add([offs_cm, offs_bn], c0)
                c1 = acc1.to(dtype)
                c_desc.atomic_add([offs_cm, offs_bn + BLOCK_SIZE_N // 2], c1)
            else:
                c = accumulator.to(dtype)
                c_desc.atomic_add([offs_cm, offs_bn], c)

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N),
                                   dtype=tl.float32)


def gemm_rs_producer_persistent(a,
                                b,
                                c,
                                world_size,
                                local_world_size,
                                num_gemm_sms,
                                BLOCK_SIZE_M=128,
                                BLOCK_SIZE_N=256,
                                BLOCK_SIZE_K=64,
                                GROUP_SIZE_M=8,
                                STAGES=3):
    # Check constraints.
    assert a.shape[1] == b.shape[
        1], "Incompatible dimensions"  # b is transposed
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

    c.fill_(0)
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    # Launch the Triton GEMM kernel. Once the kernel has completed the computation of the output tiles
    # that send to a specific rank, will set the corresponding barrier to 1.
    compiled = kernel_gemm_rs_producer_persistent[grid](
        a,
        b,
        c,
        M,
        N,
        local_K,
        local_world_size,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M,
        False,
        NUM_SMS=num_gemm_sms,  #
        num_stages=STAGES,
        num_warps=8,
    )

    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    return compiled



def torch_gemm_rs(
    input: torch.Tensor,  # [M, local_k]
    weight: torch.Tensor,  # [N, local_K]
    TP_GROUP,
):
    M, local_K = input.shape
    N = weight.shape[0]
    output = torch.matmul(input, weight.T)
    rs_output = torch.empty((M // WORLD_SIZE, N),
                            dtype=output.dtype,
                            device=input.device)
    torch.distributed.reduce_scatter_tensor(rs_output, output, group=TP_GROUP)
    return rs_output


if __name__ == "__main__":
    if torch.cuda.get_device_capability()[0] < 9:
        print("Skip the test because the device is not sm90 or higher")
        import sys
        sys.exit()

    # init
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    TP_GROUP = initialize_distributed()
    torch.cuda.synchronize()
    configs = {
        "LLaMA-7B": {"M": 8192, "K": 11008, "N": 4096, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
        "LLaMA-3.1-8B": {"M": 8192, "K": 14336, "N": 4096, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
        "LLaMA-3.1-70B": {"M": 8192, "K": 28672, "N": 8192, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
        "LLaMA-3.1-405B": {"M": 8192, "K": 53248, "N": 16384, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
        "Qwen2-72B": {"M": 8192, "K": 29568, "N": 8192, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
    }
    config = configs["Qwen2-72B"]
    M, N, K = config["M"], config["N"], config["K"]
    local_K = K // TP_GROUP.size()

    # gen input
    input_dtype = torch.bfloat16
    output_dtype = input_dtype
    scale = TP_GROUP.rank() + 1

    def _make_data(M):
        data_config = [
            ((M, local_K), input_dtype, (0.01 * scale, 0)),  # A
            ((N, local_K), input_dtype, (0.01 * scale, 0)),  # B
        ]
        generator = generate_data(data_config)
        input, weight = next(generator)
        return input, weight

    input, weight = _make_data(M)

    golden = torch_gemm_rs(input, weight, TP_GROUP)

    triton_out = nvshmem_create_tensor((M // WORLD_SIZE, N),
                                      dtype=output_dtype,)

    triton_out.fill_(0)


    # triton impl
    gemm_rs_producer_persistent(
        input,
        weight,
        triton_out,
        WORLD_SIZE,
        WORLD_SIZE,
        132)

    torch.testing.assert_close(triton_out, golden, atol=5e-2, rtol=5e-2)

    _, dur_ms = perf_func(
        partial(
            gemm_rs_producer_persistent,
            a=input,
            b=weight,
            c=triton_out,
            world_size=WORLD_SIZE,
            local_world_size=WORLD_SIZE,
            num_gemm_sms=132,
        ),
        iters=10,
        warmup_iters=5,
    )

    torch.testing.assert_close(triton_out, golden, atol=1e-1, rtol=1e-1)

    dist_print(f"tflops: {2*M*N*local_K/1e12/(dur_ms/1e3)}, ms: {dur_ms}", allowed_ranks="all", need_sync=True)

    nvshmem_free_tensor_sync(triton_out)
    finalize_distributed()
