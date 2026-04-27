"""Standalone perf test for the RS direct-reduce GEMM.

Drives the AnnotationTransformer + RemoteBinding path against the annotated
``example_persistent_gemm_remote_rs.py`` source. The generated kernel issues
``c_desc.atomic_add`` against ``dl.symm_at(c_ptr, peer)``, which fuses the
reduce-scatter into the GEMM epilogue. There is no separate comm runtime: the
kernel itself moves data.

Launch (e.g. 8-rank intra-node):

    torchrun --nproc-per-node 8 tests/test_gemm_rs_direct_reduce.py

"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch

from syncopate.communication.common_descriptors.reduce_scatter import (
    build_reduce_scatter_direct_reduce_plan,
)
from syncopate.communication.descriptor import CommPlan
from syncopate.computation.transform import AnnotationTransformer, RemoteBinding
from syncopate.interface.lowering import lower_comm_plan_to_raw_schedules
from test_gemm_rs import configs
from triton_dist.utils import (
    dist_print,
    finalize_distributed,
    initialize_distributed,
    nvshmem_barrier_all_on_stream,
    nvshmem_create_tensor,
    nvshmem_free_tensor_sync,
    perf_func,
    sleep_async,
)


def _load_transformed_gemm_rs_direct(schedule):
    transformer = AnnotationTransformer(
        remote_descriptors={
            "c_desc": RemoteBinding(base_ptr_arg="c_ptr", schedule=schedule),
        }
    )
    source_path = Path(
        "tests/computation/transform/examples/example_persistent_gemm.py"
    )
    transformed = transformer.transform(source_path.read_text())
    # Per-process path: when multiple ranks share /tmp, a single shared path
    # races (truncated reads -> module missing trailing defs).
    out_path = Path("/tmp") / f"gemm_rs_direct_generated_pid{os.getpid()}.py"
    out_path.write_text(transformed)

    spec = importlib.util.spec_from_file_location("generated_gemm_rs_direct", out_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["generated_gemm_rs_direct"] = module
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

    config = configs["LLaMA-3.1-405B"]
    BM = config["BM"]
    BN = config["BN"]
    BK = config["BK"]
    stage = config["Stage"]
    M = config["M"]
    N = config["N"]
    K = config["K"]
    dtype = torch.float16

    assert K % WORLD_SIZE == 0
    assert M % WORLD_SIZE == 0
    K_per_rank = K // WORLD_SIZE
    K_per_rank_pad = (K_per_rank + BK - 1) // BK * BK
    M_per_rank = M // WORLD_SIZE

    # Inputs: each rank owns full M but only a K-shard of A and B.
    a = torch.randn((M, K_per_rank_pad), device="cuda", dtype=dtype) / 10.0
    b = torch.randn((N, K_per_rank_pad), device="cuda", dtype=dtype) / 10.0

    # Golden: do the full GEMM locally (partial sum) then reduce-scatter the rows.
    golden_partial = torch.matmul(a, b.T)
    golden_res = torch.empty((M_per_rank, N), dtype=dtype, device="cuda")
    torch.distributed.reduce_scatter_tensor(golden_res, golden_partial, group=TP_GROUP)

    # Build a reduce-scatter direct-reduce plan; reduce_op=SUM propagates onto
    # every Transfer, so schedule.is_reduce() is True and the transformer will
    # rewrite c_desc.store -> c_desc.atomic_add automatically.
    device_plans = {
        r: build_reduce_scatter_direct_reduce_plan(
            shape=(M, N),
            dtype=dtype,
            axis=0,
            mesh_size=WORLD_SIZE,
            rank=r,
            src_buffer="src",
            dst_buffer="dst",
            transfer_kind="push",
        )
        for r in range(WORLD_SIZE)
    }
    comm_plan = CommPlan(device_plans)
    comm_plan.plan_signals()
    schedule = lower_comm_plan_to_raw_schedules(comm_plan)[rank]["src"]

    block_offsets_list = schedule.gen_block_offset_lists()
    block_shapes_list = schedule.gen_block_shape_lists()
    # Local-residency block carries peer_rank=None (-> -1); fold to self so
    # `dl.symm_at(c_ptr, RANK)` returns the local pointer for that wave.
    target_ranks_list = [
        r if r >= 0 else rank for r in schedule.gen_target_rank_list()
    ]
    # Per-wave dst offsets: for the reduce-scatter direct-reduce plan every
    # remote tile lands at offset 0 within the peer's per-rank c allocation,
    # and the local block's offsets coincide with the wave's source offsets.
    dst_offsets_list = schedule.gen_dst_offset_lists()
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
        f"Rank {rank} dst_offsets: {dst_offsets}",
        allowed_ranks="all",
        need_sync=True,
    )

    gemm_rs_direct = _load_transformed_gemm_rs_direct(schedule)

    # C must be symmetric so peers can issue dl.symm_at + atomic_add against it.
    c_sym = nvshmem_create_tensor((M_per_rank, N), dtype=dtype)

    def fused_gemm_rs_direct():
        c_sym.fill_(0)
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

        gemm_rs_direct(
            a,
            b,
            c_sym,
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

        # Wait for all peers' atomic_adds to land before reading the result.
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    fused_gemm_rs_direct()
    torch.testing.assert_close(c_sym, golden_res, atol=1e-2, rtol=1e-2)
    dist_print(
        f"Rank {rank} rs-direct-reduce results correct",
        allowed_ranks="all",
        need_sync=True,
    )

    sleep_async(1000)
    _, dur_ms = perf_func(fused_gemm_rs_direct, iters=10, warmup_iters=5)
    dist_print(
        f"tflops: {2 * M * N * K_per_rank / 1e12 / (dur_ms / 1e3)}, ms: {dur_ms}",
        allowed_ranks="all",
        need_sync=True,
    )

    nvshmem_free_tensor_sync(c_sym)
    torch.cuda.synchronize()
    torch.distributed.barrier()
    finalize_distributed()


if __name__ == "__main__":
    main()
