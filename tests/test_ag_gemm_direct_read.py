"""Standalone perf test for the AG direct-read GEMM.

Drives the AnnotationTransformer + RemoteBinding path against the annotated
``example_persistent_gemm_remote_ag.py`` source. The generated kernel reads A
through ``dl.symm_at(a_ptr, peer)`` per wave, fusing the all-gather into the
GEMM. There is no separate comm runtime — each program reads remote shards on
its own as it advances through the wave schedule.

Launch (e.g. 8-rank intra-node):

    torchrun --nproc-per-node 8 tests/test_ag_gemm_direct_read.py

"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch

from syncopate.communication.common_descriptors.all_gather import (
    build_all_gather_plan_1d_swizzle,
)
from syncopate.communication.descriptor import CommPlan
from syncopate.computation.transform import AnnotationTransformer, RemoteBinding
from syncopate.interface.lowering import lower_comm_plan_to_raw_schedules
from triton_dist.utils import (
    dist_print,
    finalize_distributed,
    initialize_distributed,
    nvshmem_barrier_all_on_stream,
    nvshmem_create_tensor,
    nvshmem_free_tensor_sync,
    perf_func,
)


configs = {
    "LLaMA-7B": {"M": 8192, "N": 11008, "K": 4096, "BM": 128, "BN": 128, "BK": 64, "Stage": 5},
    "LLaMA-3.1-8B": {"M": 8192, "N": 14336, "K": 4096, "BM": 128, "BN": 128, "BK": 64, "Stage": 5},
    "LLaMA-3.1-70B": {"M": 8192, "N": 28672, "K": 8192, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
    "LLaMA-3.1-405B": {"M": 8192, "N": 53248, "K": 16384, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
    "Mistral-7B": {"M": 8192, "N": 14336, "K": 4096, "BM": 128, "BN": 128, "BK": 64, "Stage": 5},
    "Qwen2-72B": {"M": 8192, "N": 29568, "K": 8192, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
}


def _load_transformed_gemm_ag_direct(schedule):
    transformer = AnnotationTransformer(
        remote_descriptors={
            "a_desc": RemoteBinding(base_ptr_arg="a_ptr", schedule=schedule),
        }
    )
    source_path = Path(
        "tests/computation/transform/examples/example_persistent_gemm.py"
    )
    transformed = transformer.transform(source_path.read_text())
    # Per-process path: when multiple ranks share /tmp, a single shared path
    # races (truncated reads -> module missing trailing defs).
    out_path = Path("/tmp") / f"gemm_ag_direct_generated_pid{os.getpid()}.py"
    out_path.write_text(transformed)

    spec = importlib.util.spec_from_file_location("generated_gemm_ag_direct", out_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["generated_gemm_ag_direct"] = module
    spec.loader.exec_module(module)
    return module.gemm


def main():
    WORLD_SIZE = int(os.getenv("WORLD_SIZE", "-1"))
    LOCAL_WORLD_SIZE = int(os.getenv("LOCAL_WORLD_SIZE", "-1"))

    TP_GROUP = initialize_distributed()
    rank = TP_GROUP.rank()
    if WORLD_SIZE <= 0:
        WORLD_SIZE = TP_GROUP.size()
    if LOCAL_WORLD_SIZE <= 0:
        LOCAL_WORLD_SIZE = WORLD_SIZE
    torch.cuda.set_device(rank % LOCAL_WORLD_SIZE)

    config = configs["LLaMA-3.1-70B"]
    BM = config["BM"]
    BN = config["BN"]
    BK = config["BK"]
    stage = config["Stage"]
    M = config["M"]
    N = config["N"]
    K = config["K"]
    dtype = torch.float16

    assert M % WORLD_SIZE == 0
    M_per_rank = M // WORLD_SIZE
    M_per_rank_pad = (M_per_rank + BM - 1) // BM * BM
    M_pad = M_per_rank_pad * WORLD_SIZE

    # Inputs: each rank owns one M-shard of A and full B.
    a_local = torch.randn((M_per_rank_pad, K), device="cuda", dtype=dtype)
    b = torch.randn((N, K), device="cuda", dtype=dtype)

    # Golden: gather A then matmul.
    golden_a = torch.empty((M_pad, K), device="cuda", dtype=dtype)
    torch.distributed.all_gather_into_tensor(golden_a, a_local, group=TP_GROUP)
    golden_c = torch.matmul(golden_a, b.T)

    device_plans = {
        r: build_all_gather_plan_1d_swizzle(
            shape=(M_pad, K),
            dtype=dtype,
            axis=0,
            mesh_size=WORLD_SIZE,
            rank=r,
            buffer_name="a",
            transfer_kind="pull",
        )
        for r in range(WORLD_SIZE)
    }
    comm_plan = CommPlan(device_plans)
    comm_plan.plan_signals()
    schedule = lower_comm_plan_to_raw_schedules(comm_plan)[rank]["a"]

    # AG schedule only varies along M; broaden each tile to span the full N.
    block_offsets_list = schedule.gen_block_offset_lists()
    block_shapes_list = schedule.gen_block_shape_lists()
    for i in range(len(block_offsets_list)):
        block_offsets_list[i][1] = 0
        block_shapes_list[i][1] = N
    target_ranks_list = [
        r if r >= 0 else rank for r in schedule.gen_target_rank_list()
    ]
    # The all-gather plan models `a` as a full-M tensor on every rank, so
    # `schedule.gen_dst_offset_lists()` reports global row offsets. The kernel
    # actually reads from a per-rank symmetric tensor (M_per_rank rows), so
    # we override dst_offsets with all zeros — every wave reads from row 0 of
    # the peer's slice.
    dst_offsets_list = [[0, 0] for _ in range(len(block_offsets_list))]
    cum_tiles = sum(int(s[0]) * int(s[1]) for s in block_shapes_list)

    block_shapes = torch.tensor(block_shapes_list, device="cuda", dtype=torch.int32)
    block_offsets = torch.tensor(block_offsets_list, device="cuda", dtype=torch.int32)
    target_ranks = torch.tensor(target_ranks_list, device="cuda", dtype=torch.int32)
    dst_offsets = torch.tensor(dst_offsets_list, device="cuda", dtype=torch.int32)

    dist_print(
        f"Rank {rank} block_offsets: {block_offsets_list}",
        allowed_ranks="all",
        need_sync=True,
    )
    dist_print(
        f"Rank {rank} block_shapes: {block_shapes_list}",
        allowed_ranks="all",
        need_sync=True,
    )
    dist_print(
        f"Rank {rank} target_ranks: {target_ranks_list}",
        allowed_ranks="all",
        need_sync=True,
    )
    dist_print(
        f"Rank {rank} dst_offsets: {dst_offsets_list}",
        allowed_ranks="all",
        need_sync=True,
    )

    gemm_ag_direct = _load_transformed_gemm_ag_direct(schedule)

    # A must be symmetric so peers can read each other via dl.symm_at.
    a_sym = nvshmem_create_tensor((M_per_rank_pad, K), dtype=dtype)
    a_sym.zero_()
    a_sym.copy_(a_local)
    c_out = torch.empty((M_pad, N), device="cuda", dtype=dtype)

    def fused_ag_gemm_direct():
        c_out.zero_()
        # Make sure every rank's local A shard is visible before reads start.
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

        gemm_ag_direct(
            a_sym,
            b,
            c_out,
            WORLD_SIZE,
            132,
            BLOCK_SIZE_M=BM,
            BLOCK_SIZE_N=BN,
            BLOCK_SIZE_K=BK,
            GROUP_SIZE_M=8,
            STAGES=stage,
            cur_wave_sizes=block_shapes,
            wave_offsets=block_offsets,
            target_rank=target_ranks,
            dst_offsets=dst_offsets,
            cum_tiles=cum_tiles,
        )

        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    fused_ag_gemm_direct()
    dist_print(f"Rank {rank} c_out stats: min={c_out.min().item()} max={c_out.max().item()} mean={c_out.mean().item()}  zeros={int((c_out==0).sum().item())}/{c_out.numel()}", allowed_ranks="all", need_sync=True)                               
    dist_print(f"Rank {rank} golden_c stats: min={golden_c.min().item()} max={golden_c.max().item()} mean={golden_c.mean().item()}", allowed_ranks="all", need_sync=True)                                                    
    dist_print(f"Rank {rank} c_out[0,0:8]={c_out[0,0:8]} golden[0,0:8]={golden_c[0,0:8]}", allowed_ranks="all",need_sync=True)
    torch.testing.assert_close(c_out, golden_c, atol=1e-2, rtol=1e-2)
    dist_print(
        f"Rank {rank} ag-direct-read results correct",
        allowed_ranks="all",
        need_sync=True,
    )

    _, dur_ms = perf_func(fused_ag_gemm_direct, iters=10, warmup_iters=5)
    total_flop = 2 * M * N * K
    dist_print(
        f"tflops: {total_flop / 1e12 / (dur_ms / 1e3)}, ms: {dur_ms}",
        allowed_ranks="all",
        need_sync=True,
    )

    nvshmem_free_tensor_sync(a_sym)
    torch.cuda.synchronize()
    torch.distributed.barrier()
    finalize_distributed()


if __name__ == "__main__":
    main()
