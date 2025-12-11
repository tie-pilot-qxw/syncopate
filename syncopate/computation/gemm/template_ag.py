"""
Template GEMM that directly reads remote shards to avoid an explicit all-gather.
The schedule (cur_wave_size, wave_offset, target_rank) describes which remote
rank owns each wave of tiles.
"""

import os
from functools import partial
from typing import Optional

import torch
import triton
import triton.language as tl
import triton_dist.language as dl
from syncopate.communication.code_gen import CommGenerator
from syncopate.communication.common_descriptors.all_gather import (
    build_all_gather_plan_1d_swizzle,
)
from syncopate.interface.lowering import lower_comm_plan_to_raw_schedules
from triton_dist.utils import (
    dist_print,
    generate_data,
    nvshmem_barrier_all_on_stream,
    nvshmem_create_tensor,
    nvshmem_free_tensor_sync,
    perf_func,
    finalize_distributed,
    initialize_distributed,
)


@triton.jit
def get_pid_mn(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M: tl.constexpr):
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def kernel_gemm_ag_read_remote(
    a_ptr,
    b_ptr,
    c_ptr,
    cur_wave_size,
    wave_offset,
    target_rank,
    M,
    N,
    K,
    local_world_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    USE_A_TMA: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    rank = dl.rank()
    num_ranks = dl.num_ranks()
    M_per_rank = M // num_ranks

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

    cur_wave = 0
    cur_offset_m = tl.load(wave_offset) // BLOCK_SIZE_M
    cur_offset_n = tl.load(wave_offset + 1) // BLOCK_SIZE_N

    past_tiles = 0
    cur_tile_x = tl.load(cur_wave_size + cur_wave * 2) // BLOCK_SIZE_M
    cur_tile_y = tl.load(cur_wave_size + cur_wave * 2 + 1) // BLOCK_SIZE_N
    cur_target_rank = tl.load(target_rank + cur_wave)

    cur_tiles = cur_tile_x * cur_tile_y

    remote_a_ptr = dl.symm_at(a_ptr, cur_target_rank)
    if USE_A_TMA:
        a_desc = tl.make_tensor_descriptor(
            remote_a_ptr,
            shape=[M_per_rank, K],
            strides=[K, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        )

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    offs_am = 0
    offs_bn = 0
    offs_cm = 0

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS

            new_wave = False
            while tile_id >= past_tiles + cur_tiles:
                past_tiles += cur_tiles
                cur_wave += 1
                cur_tile_x = tl.load(cur_wave_size + cur_wave * 2) // BLOCK_SIZE_M
                cur_tile_y = tl.load(cur_wave_size + cur_wave * 2 + 1) // BLOCK_SIZE_N
                cur_tiles = cur_tile_x * cur_tile_y
                new_wave = True

            if new_wave:
                cur_offset_m = tl.load(wave_offset + cur_wave * 2) // BLOCK_SIZE_M
                cur_offset_n = tl.load(wave_offset + cur_wave * 2 + 1) // BLOCK_SIZE_N
                cur_target_rank = tl.load(target_rank + cur_wave)

                remote_a_ptr = dl.symm_at(a_ptr, cur_target_rank)
                if USE_A_TMA:
                    a_desc = tl.make_tensor_descriptor(
                        remote_a_ptr,
                        shape=[M_per_rank, K],
                        strides=[K, 1],
                        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
                    )

            pid_m, pid_n = get_pid_mn(
                tile_id - past_tiles, cur_tile_x, cur_tile_y, GROUP_SIZE_M
            )

            pid_m_global = pid_m + cur_offset_m
            pid_n_global = pid_n + cur_offset_n

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n_global * BLOCK_SIZE_N
            offs_cm = pid_m_global * BLOCK_SIZE_M

        offs_k = ki * BLOCK_SIZE_K

        if USE_A_TMA:
            a = a_desc.load([offs_am, offs_k])
        else:
            offs_m = offs_am + tl.arange(0, BLOCK_SIZE_M)
            offs_k_vec = offs_k + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = remote_a_ptr + offs_m[:, None] * K + offs_k_vec[None, :]
            a = tl.load(a_ptrs)
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            if EPILOGUE_SUBTILE:
                acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
                acc = tl.permute(acc, (0, 2, 1))
                acc0, acc1 = tl.split(acc)
                c0 = acc0.to(dtype)
                c_desc.store([offs_cm, offs_bn], c0)
                c1 = acc1.to(dtype)
                c_desc.store([offs_cm, offs_bn + BLOCK_SIZE_N // 2], c1)
            else:
                c = accumulator.to(dtype)
                c_desc.store([offs_cm, offs_bn], c)

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


def gemm_ag_direct_read(
    a,
    b,
    c,
    cur_wave_size,
    wave_offset,
    target_rank,
    world_size,
    local_world_size,
    num_gemm_sms,
    use_a_tma: bool = True,
    BLOCK_SIZE_M=128,
    BLOCK_SIZE_N=256,
    BLOCK_SIZE_K=64,
    GROUP_SIZE_M=8,
    STAGES=3,
):
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    local_M, K = a.shape
    N, _ = b.shape
    world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else world_size
    assert world is not None and world > 0, "world size must be provided"
    assert local_M * world == c.shape[0], f"c must hold gathered rows (expected {local_M * world}, got {c.shape[0]})"
    assert c.shape[1] == N

    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = lambda META: (
        min(
            num_gemm_sms,
            triton.cdiv(local_M * world, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ),
    )

    compiled = kernel_gemm_ag_read_remote[grid](
        a,
        b,
        c,
        cur_wave_size,
        wave_offset,
        target_rank,
        local_M * world,
        N,
        K,
        local_world_size,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M,
        False,
        use_a_tma,
        NUM_SMS=num_gemm_sms,
        num_stages=STAGES,
        num_warps=8,
    )

    return compiled


if __name__ == "__main__":
    if torch.cuda.get_device_capability()[0] < 9:
        print("Skip the test because the device is not sm90 or higher")
        import sys
        sys.exit()

    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    TP_GROUP = initialize_distributed()
    torch.cuda.synchronize()

    configs = {
        "LLaMA-7B": {"M": 8192, "N": 11008, "K": 4096, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
        "LLaMA-3.1-8B": {"M": 8192, "N": 14336, "K": 4096, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
        "LLaMA-3.1-70B": {"M": 8192, "N": 28672, "K": 8192, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
        "LLaMA-3.1-405B": {"M": 8192, "N": 53248, "K": 16384, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
        "Qwen2-72B": {"M": 8192, "N": 29568, "K": 8192, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
    }
    config = configs["LLaMA-3.1-405B"]
    M, N, K = config["M"], config["N"], config["K"]
    local_M = M // TP_GROUP.size()
    use_tma_a = False

    input_dtype = torch.bfloat16
    output_dtype = input_dtype
    scale = TP_GROUP.rank() + 1

    def _make_data(local_rows):
        data_config = [
            ((local_rows, K), input_dtype, (0.01 * scale, 0)),  # A shard
            ((N, K), input_dtype, (0.01 * scale, 0)),  # B
        ]
        generator = generate_data(data_config)
        a_local, b = next(generator)
        return a_local, b

    a_local, b = _make_data(local_M)

    golden_a = torch.empty((M, K), device="cuda", dtype=input_dtype)
    torch.distributed.all_gather_into_tensor(golden_a, a_local, group=TP_GROUP)
    golden_c = torch.matmul(golden_a, b.T)

    device_plans = {
        r: build_all_gather_plan_1d_swizzle(
            shape=(M, K),
            dtype=input_dtype,
            axis=0,
            mesh_size=WORLD_SIZE,
            rank=r,
            buffer_name="a",
            transfer_kind="pull",
        )
        for r in range(WORLD_SIZE)
    }

    generator = CommGenerator(device_plans)
    generator.plan_signals()
    schedule = lower_comm_plan_to_raw_schedules(generator)[RANK]["a"]

    block_offsets_list = schedule.gen_block_offset_lists()
    block_shapes_list = schedule.gen_block_shape_lists()
    for i in range(len(block_offsets_list)):
        block_offsets_list[i][1] = 0
        block_shapes_list[i][1] = N

    target_ranks_list = [block_offsets_list[i][0] // local_M for i in range(len(block_shapes_list))]

    block_shapes = torch.tensor(block_shapes_list, dtype=torch.int32, device="cuda")
    block_offsets = torch.tensor(block_offsets_list, dtype=torch.int32, device="cuda")
    target_ranks = torch.tensor(target_ranks_list, dtype=torch.int32, device="cuda")

    dist_print(f"Rank {RANK} block_shapes: {block_shapes}", allowed_ranks="all", need_sync=True)
    dist_print(f"Rank {RANK} block_offsets: {block_offsets}", allowed_ranks="all", need_sync=True)
    dist_print(f"Rank {RANK} target_ranks: {target_ranks}", allowed_ranks="all", need_sync=True)

    a_sym = nvshmem_create_tensor((local_M, K), dtype=input_dtype)
    a_sym.zero_()
    a_sym.copy_(a_local)

    c_out = torch.empty((M, N), device="cuda", dtype=output_dtype)
    c_out.zero_()
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    gemm_ag_direct_read(
        a_sym,
        b,
        c_out,
        block_shapes,
        block_offsets,
        target_ranks,
        WORLD_SIZE,
        LOCAL_WORLD_SIZE,
        132,
        use_a_tma=use_tma_a,
        BLOCK_SIZE_M=config["BM"],
        BLOCK_SIZE_N=config["BN"],
        BLOCK_SIZE_K=config["BK"],
        GROUP_SIZE_M=8,
        STAGES=config["Stage"],
    )

    torch.testing.assert_close(c_out, golden_c, atol=5e-2, rtol=5e-2)

    _, dur_ms = perf_func(
        partial(
            gemm_ag_direct_read,
            a=a_sym,
            b=b,
            c=c_out,
            cur_wave_size=block_shapes,
            wave_offset=block_offsets,
            target_rank=target_ranks,
            world_size=WORLD_SIZE,
            local_world_size=LOCAL_WORLD_SIZE,
            num_gemm_sms=132,
            use_a_tma=use_tma_a,
            BLOCK_SIZE_M=config["BM"],
            BLOCK_SIZE_N=config["BN"],
            BLOCK_SIZE_K=config["BK"],
            GROUP_SIZE_M=8,
            STAGES=config["Stage"],
        ),
        iters=10,
        warmup_iters=5,
    )

    torch.testing.assert_close(c_out, golden_c, atol=1e-1, rtol=1e-1)
    dist_print(
        f"tflops: {2*M*N*K/1e12/(dur_ms/1e3)}, ms: {dur_ms}",
        allowed_ranks="all",
        need_sync=True,
    )

    nvshmem_free_tensor_sync(a_sym)
    finalize_distributed()
