from syncopate.communication.code_gen import CommGenerator
from syncopate.communication.comm_runtime.communication_context import CommContext
from syncopate.interface.lowering import lower_comm_plan_to_raw_schedules
from syncopate.computation.gemm.template import gemm_producer
from triton_dist.utils import initialize_distributed, nvshmem_barrier_all_on_stream, finalize_distributed, dist_print, group_profile, perf_func, sleep_async
from triton_dist.kernels.nvidia.common_ops import _wait_eq_cuda
import torch
from syncopate.communication.common_descriptors import build_all_to_all_plan
import os
import triton
import triton.language as tl

# uses the same config as triton distributed
configs = {
    "LLaMA-7B": {"M": 8192, "K": 11008, "N": 4096, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
    "LLaMA-3.1-8B": {"M": 8192, "K": 14336, "N": 4096, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
    "LLaMA-3.1-70B": {"M": 8192, "K": 28672, "N": 8192, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
    "LLaMA-3.1-405B": {"M": 8192, "K": 53248, "N": 16384, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
    "Qwen2-72B": {"M": 8192, "K": 29568, "N": 8192, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
}

@triton.jit
def kernel_ring_reduce_tma(
    c_ptr,  # [M, N]
    out_ptr,  # [M_per_split, N]
    local_out_ptr, # [M, N]
    # shape of matrix
    M_per_rank,
    N,
    rank,
    num_splits: tl.constexpr,
    # reduce tile shape
    BLOCK_SIZE_M: tl.constexpr = 256,
    BLOCK_SIZE_N: tl.constexpr = 64,
):
    local_out_desc = tl.make_tensor_descriptor(
        local_out_ptr,
        shape=[M_per_rank * num_splits, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M_per_rank * num_splits, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )
    output_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[M_per_rank, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    pid = tl.program_id(axis=0)
    num_pid = tl.num_programs(axis=0)
    num_tiles_m = tl.cdiv(M_per_rank, BLOCK_SIZE_M)
    num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_tiles_m * num_tiles_n
    for tile_id in range(pid, total_tiles, num_pid):
        tile_id_m = tile_id // num_tiles_n
        tile_id_n = tile_id % num_tiles_n
        # accum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=out_ptr.dtype.element_ty)
        accum = local_out_desc.load([tile_id_m * BLOCK_SIZE_M + rank * M_per_rank, tile_id_n * BLOCK_SIZE_N])
        # cur_rank = (begin_idx + 1) % num_splits
        # accum = c_desc.load([tile_id_m * BLOCK_SIZE_M + cur_rank * M_per_rank, tile_id_n * BLOCK_SIZE_N])
        for i in range(1, num_splits):
            cur_rank = (rank + i) % num_splits
            data = c_desc.load([tile_id_m * BLOCK_SIZE_M + cur_rank * M_per_rank, tile_id_n * BLOCK_SIZE_N])
            accum += data

        output_desc.store([tile_id_m * BLOCK_SIZE_M, tile_id_n * BLOCK_SIZE_N], accum)


def ring_reduce_tma(
    input: torch.Tensor,  # [M_per_node, N]
    output: torch.Tensor,  # [M_per_rank, N]
    local_output: torch.Tensor,  # [M, N]
    begin_idx,
    num_sms=-1,
):
    num_splits, M_per_split, N = input.shape

    assert output.shape[0] == M_per_split and output.shape[1] == N

    def alloc_fn(size, alignment, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    if num_sms == -1:
        grid = lambda META: (triton.cdiv(M_per_split, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
        kernel_ring_reduce_tma[grid](
            input,
            output,
            local_output,
            M_per_split,
            N,
            begin_idx,
            num_splits,
            BLOCK_SIZE_M=256,
            BLOCK_SIZE_N=64,
            num_warps=4,
        )
    else:
        grid = lambda META: (min(
            triton.cdiv(M_per_split, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), num_sms), )
        kernel_ring_reduce_tma[grid](
            input,
            output,
            local_output,
            M_per_split,
            N,
            begin_idx,
            num_splits,
            BLOCK_SIZE_M=256,
            BLOCK_SIZE_N=128,
            num_warps=8,
        )

    return output

if __name__ == "__main__":
    WORLD_SIZE = int(os.getenv("WORLD_SIZE", "-1"))
    LOCAL_WORLD_SIZE = int(os.getenv("LOCAL_WORLD_SIZE", "-1"))

    TP_GROUP = initialize_distributed()
    rank = TP_GROUP.rank()

    config = configs["LLaMA-3.1-70B"]
    BM = config["BM"]
    BN = config["BN"]
    BK = config["BK"]
    stage = config["Stage"]

    M = config["M"]
    N = config["N"]
    K = config["K"]
    dtype = torch.float16
    assert K % WORLD_SIZE == 0

    K_per_rank = K // WORLD_SIZE
    K_per_rank_pad = (K_per_rank + BK - 1) // BK * BK # pad to BK
    K_pad = K_per_rank_pad * WORLD_SIZE

    # generate data
    a = torch.randn((M, K_per_rank_pad), device="cuda", dtype=dtype) / 10.0 # scale down to reduce numical error
    b = torch.randn((N, K_per_rank_pad), device="cuda", dtype=dtype) / 10.0
    c = torch.empty((M // WORLD_SIZE, N), dtype=dtype, device="cuda")

    golden_c_partial = torch.matmul(a, b.T)
    golden_res = torch.empty((M // WORLD_SIZE, N),
                            dtype=a.dtype,
                            device=a.device)
    torch.distributed.reduce_scatter_tensor(golden_res, golden_c_partial, group=TP_GROUP)


    device_plans = {
        rank: build_all_to_all_plan(
            shape=(M, N),
            dtype=dtype,
            mesh_size=WORLD_SIZE,
            rank=rank,
            src_buffer="src",
            dst_buffer="dst",
            transfer_kind="push",
            compute_producer=True,
        )
        for rank in range(WORLD_SIZE)
    }

    generator = CommGenerator(device_plans)
    generator.plan_signals()
    schedule = lower_comm_plan_to_raw_schedules(generator)[rank]["src"]
    comm_info = generator.generate_code_for_plan()
    comm_info.local_world_size = comm_info.world_size  # for testing purpose, assume intra-node only
    dist_print(f"Rank {rank} CommInfo: {comm_info}", allowed_ranks="all", need_sync=True)


    block_offsets_list = schedule.gen_block_offset_lists()
    block_shapes_list = schedule.gen_block_shape_lists()
    signal_offsets_list = schedule.gen_signal_lists()
    block_offsets = torch.tensor(block_offsets_list, device="cuda", dtype=torch.int32)
    block_shapes = torch.tensor(block_shapes_list, device="cuda", dtype=torch.int32)
    signal_offsets = torch.tensor(signal_offsets_list, device="cuda", dtype=torch.int32)

    dist_print(f"Rank {rank} block_offsets: {block_offsets_list}", allowed_ranks="all", need_sync=True)
    dist_print(f"Rank {rank} block_shapes: {block_shapes_list}", allowed_ranks="all", need_sync=True)
    dist_print(f"Rank {rank} signal_offsets: {signal_offsets_list}", allowed_ranks="all", need_sync=True)


    comm_runtime = CommContext(rank, comm_info)

    comm_stream = torch.cuda.Stream()

    def fused_gemm_rs():
        comm_runtime.reset_signals()

        # after reset signals, comm and compute can start
        barrier_reset = torch.cuda.Event()
        barrier_reset.record(torch.cuda.current_stream())

        src_buf = comm_runtime.comm_buffers["src"][comm_runtime.local_rank]
        signal_buffer = comm_runtime.compute_signal_bufs[comm_runtime.local_rank]
        
        # gemm producer
        gemm_producer(a, b, src_buf, signal_buffer, block_shapes, block_offsets, signal_offsets, num_gemm_sms=132)

        comm_stream.wait_event(barrier_reset)
        nvshmem_barrier_all_on_stream(comm_stream)
        comm_runtime.start_after(comm_stream)
        comm_runtime.execute(neglect_local=True)
        comm_runtime.end_before(comm_stream)

        recv_done_signals = comm_runtime.recv_signal_bufs[comm_runtime.local_rank]

        # wait for all recv done
        for offset in range(recv_done_signals.shape[0]):
            _wait_eq_cuda(recv_done_signals[offset], 1, torch.cuda.current_stream())


        # # nvshmem_barrier_all_on_stream(comm_stream)

        # scatter_done = torch.cuda.Event()
        # scatter_done.record(comm_stream)

        # # after comm done, we can do the reduction
        # torch.cuda.current_stream().wait_event(scatter_done)

        # reduce at dim 0
        dst_buf = comm_runtime.comm_buffers["dst"][comm_runtime.local_rank]

        ring_reduce_tma(dst_buf, c, src_buf, rank)

    fused_gemm_rs()
    torch.testing.assert_close(c, golden_res, atol=1e-2, rtol=1e-2)
    dist_print(f"Rank {rank} rs-gemm results correct", allowed_ranks="all", need_sync=True)

    # performace test
    sleep_async(1000) # sleep 5s to ensure cpu issue is not counted in time measurement
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    _, dur_ms = perf_func(fused_gemm_rs, iters=10, warmup_iters=5)

    dist_print(f"tflops: {2*M*N*K_per_rank/1e12/(dur_ms/1e3)}, ms: {dur_ms}", allowed_ranks="all", need_sync=True)

    with group_profile("gemm_rs", False, group=TP_GROUP):
        for _ in range(10):
            fused_gemm_rs()

    del comm_runtime
    torch.cuda.synchronize()
    torch.distributed.barrier()
    finalize_distributed()



