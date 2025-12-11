from syncopate.communication.code_gen import CommGenerator
from syncopate.communication.comm_runtime.communication_context import CommContext
from syncopate.interface.lowering import lower_comm_plan_to_raw_schedules
from syncopate.computation.attn.template import attention
from triton_dist.utils import initialize_distributed, nvshmem_barrier_all_on_stream, finalize_distributed, dist_print, group_profile, perf_func, sleep_async
import torch
from syncopate.communication.common_descriptors import build_all_gather_plan_1d_swizzle
from flash_attn import flash_attn_func
import os

# def fused_all_gather_gemm(b, c, comm_runtime, block_shapes, block_offsets, signal_offsets,
#                            num_gemm_sms,
#                            BM,
#                            BN,
#                            BK,
#                            stages):
#     comm_runtime.reset_signals()
#     comm_buf = comm_runtime.comm_buffers["a"][comm_runtime.local_rank]
#     nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
#     comm_runtime.start_after(torch.cuda.current_stream())
#     comm_runtime.execute()
#     gemm_consumer(comm_buf, b, c, comm_runtime.recv_signal_bufs[comm_runtime.local_rank], block_shapes, block_offsets, signal_offsets, comm_info.world_size, num_gemm_sms,
#                   BM, BN, BK, 8, stages)
#     comm_runtime.end_before(torch.cuda.current_stream())


def all_gather_attn(comm_runtime, wave_dict, compute_stream):
    comm_buf = comm_runtime.comm_buffers["kv"][comm_runtime.local_rank]
    comm_runtime.reset_signals()
    nvshmem_barrier_all_on_stream(compute_stream)
    comm_runtime.start_after(compute_stream)
    comm_runtime.execute()
    k_full = comm_buf[0]
    v_full = comm_buf[1]
    out = attention(q, k_full, v_full, 1, wave_dict)
    comm_runtime.end_before(compute_stream)
    return out

if __name__ == "__main__":
    WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
    LOCAL_WORLD_SIZE = int(os.getenv("LOCAL_WORLD_SIZE", "1"))

    TP_GROUP = initialize_distributed()
    rank = TP_GROUP.rank()

    torch.cuda.set_device(rank % LOCAL_WORLD_SIZE)
    B = 1
    H = 32
    SEQ = 4096 * WORLD_SIZE
    DIM = 128
    dtype = torch.float16

    assert SEQ % WORLD_SIZE == 0

    SEQ_per_rank = SEQ // WORLD_SIZE

    BLOCK_Q = 128
    BLOCK_KV = 128

    # generate data
    q = torch.randn((B, H, SEQ_per_rank, DIM), device="cuda", dtype=dtype)
    kv = torch.randn((2, B, H, SEQ_per_rank, DIM), device="cuda", dtype=dtype)
    k = kv[0]
    v = kv[1]

    kv_perm = kv.permute(3, 0, 1, 2, 4).contiguous()
    comm_res = torch.empty(SEQ, 2, B, H, DIM, device="cuda", dtype=dtype)
    torch.distributed.all_gather_into_tensor(comm_res, kv_perm)
    kv_full_golden = comm_res.permute(1, 2, 3, 0, 4)
    attn_out_golden = flash_attn_func(q.permute(0, 2, 1, 3), kv_full_golden[0].permute(0, 2, 1, 3), kv_full_golden[1].permute(0, 2, 1, 3), softmax_scale=1).permute(0, 2, 1, 3)

    device_plans = {
        rank: build_all_gather_plan_1d_swizzle(
            shape=(2, B, H, SEQ, DIM),
            dtype=dtype,
            axis=3,
            mesh_size=WORLD_SIZE,
            rank=rank,
            buffer_name="kv",
            transfer_kind="pull",
        )
        for rank in range(WORLD_SIZE)
    }

    generator = CommGenerator(device_plans)
    generator.plan_signals()
    schedule = lower_comm_plan_to_raw_schedules(generator)[rank]["kv"]

    # this is for kv, so its shape is (2, B, H, SEQ, DIM)
    block_offsets_list = schedule.gen_block_offset_lists()
    block_shapes_list = schedule.gen_block_shape_lists()
    signal_offsets_list = schedule.gen_signal_lists()
    cum_block_counts = []

    # what we need is (B, H, SEQ_Q, SEQ_KV)
    for i in range(len(block_offsets_list)):
        block_offsets_list[i] = [block_offsets_list[i][1], block_offsets_list[i][2], 0, block_offsets_list[i][3] // BLOCK_KV]
        block_shapes_list[i] = [block_shapes_list[i][1], block_shapes_list[i][2], SEQ_per_rank // BLOCK_Q, block_shapes_list[i][3] // BLOCK_KV]
        cum_block_counts.append(block_shapes_list[i][0] * block_shapes_list[i][1] * block_shapes_list[i][2]) # number of blocks in this step, ignore the reduce dim(kv)
        if i > 0:
            cum_block_counts[i] += cum_block_counts[i - 1]

    dist_print(f"block_offsets: {block_offsets_list}, block_shapes: {block_shapes_list}, signal_offsets: {signal_offsets_list}, cum_block_counts: {cum_block_counts}", allowed_ranks="all", need_sync=True)


    block_offsets = torch.tensor(block_offsets_list, device="cuda", dtype=torch.int32)
    block_shapes = torch.tensor(block_shapes_list, device="cuda", dtype=torch.int32)
    signal_offsets = torch.tensor(schedule.gen_signal_lists(), device="cuda", dtype=torch.int32)
    cum_counts = torch.tensor(cum_block_counts, device="cuda", dtype=torch.int32)


    comm_info = generator.generate_code_for_plan()
    comm_info.local_world_size = comm_info.world_size  # for testing purpose, assume intra-node only
    
    # print(f"CommInfo: {comm_info}")

    dist_print(f"Rank {rank} CommInfo: {comm_info}")
    comm_runtime = CommContext(rank, comm_info)
    comm_buf = comm_runtime.comm_buffers["kv"][comm_runtime.local_rank]
    comm_buf.zero_()
    comm_buf[:, :, :, SEQ_per_rank * rank : SEQ_per_rank * (rank + 1), :].copy_(kv)

    wave_dict = {
        "w": WORLD_SIZE,
        "cum_wave_sizes": cum_counts,
        "wave_offsets": block_offsets,
        "wave_sizes": block_shapes,
        "signal_ptr": comm_runtime.recv_signal_bufs[comm_runtime.local_rank],
        "signal_offsets": signal_offsets,
        "CUM_CHUNK_BLOCK": WORLD_SIZE,
    }

    def func(compute_stream=torch.cuda.current_stream()):
        return all_gather_attn(comm_runtime, wave_dict, compute_stream)

    

    out_triton = func()

    torch.testing.assert_close(comm_buf, kv_full_golden, atol=1e-2, rtol=1e-2)
    dist_print(f"Rank {rank} all_gather results correct", allowed_ranks="all", need_sync=True)

    torch.testing.assert_close(out_triton, attn_out_golden, atol=1e-2, rtol=1e-2)
    dist_print(f"Rank {rank} attention results correct", allowed_ranks="all", need_sync=True)

    new_stream = torch.cuda.Stream()
    with torch.cuda.stream(new_stream):
        sleep_async(1000) # sleep 5s to ensure cpu issue is not counted in time measurement
        _, dur_ms = perf_func(func, iters=10, warmup_iters=5)

    flops_per_matmul = 2.0 * B * H * SEQ * SEQ * DIM
    total_flops = 2 * flops_per_matmul
    per_gpu_flops = total_flops / WORLD_SIZE
    tflops = per_gpu_flops / 1e12 / (dur_ms / 1e3)
    dist_print(f"tflops: {tflops} time: {dur_ms}", allowed_ranks="all", need_sync=True)

    # with group_profile("all_gather_attn", True, group=TP_GROUP):
    #     with torch.cuda.stream(new_stream):
    #         sleep_async(1000) # sleep 5s to ensure cpu issue is not counted in time measurement
    #         _, dur_ms = perf_func(func, iters=10, warmup_iters=5)

    del comm_runtime

    finalize_distributed()



