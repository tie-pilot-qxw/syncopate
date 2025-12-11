## this test tests how to gen a consumer gemm from a local gemm kernel for all-gather + gemm fused kernel


from pathlib import Path

from syncopate.communication.code_gen import CommGenerator
from syncopate.communication.comm_runtime.communication_context import CommContext
from syncopate.interface.lowering import lower_comm_plan_to_raw_schedules
from syncopate.computation.transform import AnnotationTransformer
from triton_dist.utils import initialize_distributed, nvshmem_barrier_all_on_stream, finalize_distributed, dist_print, group_profile, perf_func
import torch
from syncopate.communication.common_descriptors import build_all_gather_plan_1d_swizzle
import os
import importlib
import sys

def _load_transformed_gemm_consumer():
    transformer = AnnotationTransformer(enable_consumer=True, consumer_descriptors=("a_desc",))
    # the annotated local gemm kernel path
    example_path = Path("tests/computation/transform/examples/example_persistent_gemm.py")
    transformed_source = transformer.transform(example_path.read_text())
    store_path = Path("/tmp/" + example_path.name.replace(".py", "_consumer_transformed.py"))
    with open(store_path, "w") as f:
        f.write(transformed_source)
        
    spec = importlib.util.spec_from_file_location("generated_tmp_kernel", store_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["generated_tmp_kernel"] = module
    spec.loader.exec_module(module)
    return module.gemm


generated_gemm_consumer = _load_transformed_gemm_consumer()

configs = {
    "LLaMA-7B": {"M": 8192, "N": 11008, "K": 4096, "BM": 128, "BN": 128, "BK": 64, "Stage": 5},
    "LLaMA-3.1-8B": {"M": 8192, "N": 14336, "K": 4096, "BM": 128, "BN": 128, "BK": 64, "Stage": 5},
    "LLaMA-3.1-70B": {"M": 8192, "N": 28672, "K": 8192, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
    "LLaMA-3.1-405B": {"M": 8192, "N": 53248, "K": 16384, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
    "Mistral-7B": {"M": 8192, "N": 14336, "K": 4096, "BM": 128, "BN": 128, "BK": 64, "Stage": 5},
    "Qwen2-72B": {"M": 8192, "N": 29568, "K": 8192, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
}

def fused_all_gather_gemm(b, c, comm_runtime, block_shapes, block_offsets, signal_offsets,
                           cum_tiles,
                           num_gemm_sms,
                           BM,
                           BN,
                           BK,
                           stages):
    comm_runtime.reset_signals()
    comm_buf = comm_runtime.comm_buffers["a"][comm_runtime.local_rank]
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    comm_runtime.start_after(torch.cuda.current_stream())
    comm_runtime.execute()
    signal_buffer = comm_runtime.recv_signal_bufs[comm_runtime.local_rank]
    gemm_kwargs = {
        "cur_wave_sizes": block_shapes,
        "wave_offsets": block_offsets,
        "signal_offsets": signal_offsets,
        "signal_ptr": signal_buffer,
        "cum_tiles": cum_tiles,
    }
    generated_gemm_consumer(
        comm_buf,
        b,
        c,
        comm_info.world_size,
        num_gemm_sms,
        BLOCK_SIZE_M=BM,
        BLOCK_SIZE_N=BN,
        BLOCK_SIZE_K=BK,
        GROUP_SIZE_M=8,
        STAGES=stages,
        **gemm_kwargs,
    )
    comm_runtime.end_before(torch.cuda.current_stream())

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
    n_warmups = 5
    n_iters = 10

    dtype = torch.float16
    assert M % WORLD_SIZE == 0
    assert N % WORLD_SIZE == 0

    M_per_rank = M // WORLD_SIZE
    M_per_rank_pad = (M_per_rank + BM - 1) // BM * BM # pad to BM
    M_pad = M_per_rank_pad * WORLD_SIZE

    N_per_rank = N // WORLD_SIZE
    N_per_rank_pad = (N_per_rank + BN - 1) // BN * BN # pad to BN
    N_pad = N_per_rank_pad * WORLD_SIZE

    # generate data
    a = torch.randn((M_per_rank_pad, K), device="cuda", dtype=dtype)
    b = torch.randn((N_per_rank_pad, K), device="cuda", dtype=dtype)
    c = torch.zeros((M_pad, N_per_rank_pad), device="cuda", dtype=dtype)
    dist_print(f"shape a: {a.shape}, b: {b.shape}, c: {c.shape}")

    golden_a = torch.empty((M_pad, K), device="cuda", dtype=dtype)
    torch.distributed.all_gather_into_tensor(golden_a, a)
    golden_c = torch.matmul(golden_a, b.T)

    device_plans = {
        rank: build_all_gather_plan_1d_swizzle(
            shape=(M_pad, K),
            dtype=dtype,
            axis=0,
            mesh_size=WORLD_SIZE,
            rank=rank,
            buffer_name="a",
            transfer_kind="pull",
        )
        for rank in range(WORLD_SIZE)
    }

    generator = CommGenerator(device_plans)
    generator.plan_signals()
    schedule = lower_comm_plan_to_raw_schedules(generator)[rank]["a"]

    block_offsets_list = schedule.gen_block_offset_lists()
    block_shapes_list = schedule.gen_block_shape_lists()
    for i in range(len(block_offsets_list)):
        block_offsets_list[i][0] = block_offsets_list[i][0]
        block_offsets_list[i][1] = 0
        block_shapes_list[i][0] = block_shapes_list[i][0]
        block_shapes_list[i][1] = N_per_rank_pad

    cum_tiles = sum(int(shape[0]) * int(shape[1]) for shape in block_shapes_list)
    block_offsets = torch.tensor(block_offsets_list, device="cuda", dtype=torch.int32)
    dist_print(f"Rank {rank} block offsets: {block_offsets}", allowed_ranks="all", need_sync=True)
    block_shapes = torch.tensor(block_shapes_list, device="cuda", dtype=torch.int32)
    dist_print(f"Rank {rank} block shapes: {block_shapes}", allowed_ranks="all", need_sync=True)
    signal_offsets = torch.tensor(schedule.gen_signal_lists(), device="cuda", dtype=torch.int32)
    dist_print(f"Rank {rank} signal offsets: {signal_offsets}", allowed_ranks="all", need_sync=True)

    # full_signal_buf = torch.ones((3,), device="cuda", dtype=torch.int64)
    # gemm_consumer(golden_a, b, c, full_signal_buf, block_shapes, block_offsets, signal_offsets, WORLD_SIZE, 132)

    # torch.testing.assert_close(c, golden_c, atol=1e-2, rtol=1e-2)
    # dist_print(f"Rank {rank} single-node gemm results correct", allowed_ranks="all", need_sync=True)

    comm_info = generator.generate_code_for_plan()
    comm_info.local_world_size = comm_info.world_size  # for testing purpose, assume intra-node only
    
    dist_print(f"Rank {rank} CommInfo: {comm_info}")
    comm_runtime = CommContext(rank, comm_info)

    dist_print(f"Rank {rank} signal_buf: recv{comm_runtime.recv_signal_bufs[comm_runtime.local_rank]}", allowed_ranks="all", need_sync=True)

    # copy data to comm buffer
    comm_buf = comm_runtime.comm_buffers["a"][comm_runtime.local_rank]
    comm_buf.zero_()
    comm_buf[M_per_rank * rank : M_per_rank * (rank + 1), :].copy_(a)





    for tile_m in [64, 128, 256]:
        for tile_n in [64, 128, 256]:
            for tile_k in [64, 128]:
                for stages in [2, 3, 4, 5]:
                    BM = tile_m
                    BN = tile_n
                    BK = tile_k
                    stage = stages
                    def func():
                        fused_all_gather_gemm(b, c, comm_runtime, block_shapes, block_offsets, signal_offsets, cum_tiles, 132
                                , BM, BN, BK, stage)

                    try:
                        _, dur_ms = perf_func(func, iters=n_iters, warmup_iters=n_warmups)
                    except Exception as e:
                        continue
                    total_flop = 2 * M * N_per_rank * K
                    total_tflop = total_flop / 1e12
                    tflops = total_tflop/(dur_ms/1e3)
                    tflops_tensor = torch.tensor(tflops, device="cuda")
                    torch.distributed.all_reduce(tflops_tensor, op=torch.distributed.ReduceOp.MIN)
                    total_tflop = tflops_tensor.item()
                    dist_print(f"tile_m: {tile_m}, tile_n: {tile_n}, tile_k: {tile_k}, stages: {stages}, tflops: {total_tflop}")
                    # dist_print(f"tflops: {total_tflop/(dur_ms/1e3)}, time: {dur_ms}ms tflop: {total_tflop}", allowed_ranks="all", need_sync=True)


    # with group_profile("all_gather_gemm", True, group=TP_GROUP):
    #     for _ in range(10):
    #         fused_all_gather_gemm(b, c, comm_runtime, block_shapes, block_offsets, signal_offsets, cum_tiles, 132,
    #                               BM, BN, BK, stage)

    torch.testing.assert_close(comm_buf, golden_a, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(c, golden_c, atol=1e-2, rtol=1e-2)
    dist_print(f"Rank {rank} results correct, signal_buf: recv{comm_runtime.recv_signal_bufs[comm_runtime.local_rank]}", allowed_ranks="all", need_sync=True)
    del comm_runtime

    finalize_distributed()

