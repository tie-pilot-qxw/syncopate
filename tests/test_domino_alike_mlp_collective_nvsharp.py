"""
Domino-alike single-microbatch MLP using NVSHARP all-reduce collectives for the second GEMM.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from syncopate.communication.code_gen import CommGenerator
from syncopate.communication.comm_runtime.communication_context import CommContext
from syncopate.communication.common_descriptors import build_all_reduce_nvsharp_plan
from syncopate.interface.lowering import lower_comm_plan_to_raw_schedules
from tests.test_gemm_rs import configs
from triton_dist.utils import (
    dist_print,
    finalize_distributed,
    group_profile,
    initialize_distributed,
    nvshmem_barrier_all_on_stream,
    perf_func,
    sleep_async,
)


def _load_transformed_gemm_producer():
    """Load the transformed persistent GEMM producer used in other collective tests."""
    from syncopate.computation.transform import AnnotationTransformer

    transformer = AnnotationTransformer(enable_producer=True)
    example_path = Path("tests/computation/transform/examples/example_persistent_gemm.py")
    transformed_source = transformer.transform(example_path.read_text())
    generated_path = Path("/tmp/" + example_path.name.replace(".py", "_producer_transformed.py"))
    try:
        with open(generated_path, "r") as f:
            existing = f.read()
        if existing != transformed_source:
            with open(generated_path, "w") as f:
                f.write(transformed_source)
    except FileNotFoundError:
        with open(generated_path, "w") as f:
            f.write(transformed_source)

    spec = importlib.util.spec_from_file_location("generated_gemm_producer", generated_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["generated_gemm_producer"] = module
    spec.loader.exec_module(module)
    return module.gemm


def _gather_weight_row_parallel(param: torch.Tensor, dim: int, group: dist.ProcessGroup) -> torch.Tensor:
    """All-gather shards along the given dimension to rebuild the full weight."""
    if not dist.is_initialized() or dist.get_world_size(group) == 1:
        return param
    parts = [torch.empty_like(param) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, param, group=group)
    return torch.cat(parts, dim=dim)


def _build_mlp_tensors(config_name: str, dtype: torch.dtype, tp_group):
    config = configs[config_name]
    batch_size = 8
    M = config["M"]  # tokens * batch_size
    FFN_hidden = config["K"]
    hidden_size = config["N"]

    world = dist.get_world_size(tp_group) if dist.is_initialized() else 1

    FFN_per_rank = FFN_hidden // world

    # Inputs and weights (column-parallel fc1, row-parallel fc2).
    hidden = torch.randn((M, hidden_size), device="cuda", dtype=dtype) / 10.0
    w1_shard = torch.randn((FFN_per_rank, hidden_size), device="cuda", dtype=dtype) / 10.0
    w2_shard = torch.randn((hidden_size, FFN_per_rank), device="cuda", dtype=dtype) / 10.0

    return hidden, w1_shard, w2_shard


def _build_comm(
    plan_shape: Tuple[int, int], dtype: torch.dtype, num_splits: int, num_comm_sms: int, world_size: int, rank: int
):
    plans = {
        r: build_all_reduce_nvsharp_plan(
            shape=plan_shape,
            dtype=dtype,
            mesh_size=world_size,
            rank=r,
            src_name="src",
            dst_name="dst",
            num_all_reduces=num_splits,
            split_axis=0,
            compute_producer=True,
        )
        for r in range(world_size)
    }

    generator = CommGenerator(plans)
    generator.plan_signals()
    lowered = lower_comm_plan_to_raw_schedules(generator)
    schedule = lowered[rank]["src"]

    block_offsets_list = schedule.gen_block_offset_lists()
    block_shapes_list = schedule.gen_block_shape_lists()
    signal_offsets_list = schedule.gen_signal_lists()

    comm_info = generator.generate_code_for_plan()
    comm_info.local_world_size = comm_info.world_size
    comm_info.num_copy_sms = num_comm_sms

    block_offsets = torch.tensor(block_offsets_list, device="cuda", dtype=torch.int32)
    block_shapes = torch.tensor(block_shapes_list, device="cuda", dtype=torch.int32)
    signal_offsets = torch.tensor(signal_offsets_list, device="cuda", dtype=torch.int32)
    cum_tiles = sum(int(shape[0]) * int(shape[1]) for shape in block_shapes_list)

    return comm_info, block_offsets, block_shapes, signal_offsets, cum_tiles


def main():
    gemm_producer = _load_transformed_gemm_producer()

    WORLD_SIZE = int(os.getenv("WORLD_SIZE", "-1"))
    LOCAL_WORLD_SIZE = int(os.getenv("LOCAL_WORLD_SIZE", "-1"))

    tp_group = initialize_distributed()
    rank = tp_group.rank()
    if WORLD_SIZE <= 0:
        WORLD_SIZE = tp_group.size()
    if LOCAL_WORLD_SIZE <= 0:
        LOCAL_WORLD_SIZE = WORLD_SIZE
    torch.cuda.set_device(rank % LOCAL_WORLD_SIZE)

    dtype = torch.float16
    hidden, w1_shard, w2_shard = _build_mlp_tensors("LLaMA-3.1-70B", dtype, tp_group)

    dist_print(
        f"hidden shape: {hidden.shape}, w1_shard shape: {w1_shard.shape}, w2_shard shape: {w2_shard.shape}"
    )

    M, N = hidden.shape
    warmup = 5
    iters = 10
    profile = False

    # Broadcast input so all ranks use identical hidden states for validation.
    dist.broadcast(hidden, src=0, group=tp_group)

    # Golden: gather weights and compute full MLP locally, then compare.
    w1_full = _gather_weight_row_parallel(w1_shard, dim=0, group=tp_group)
    w2_full = _gather_weight_row_parallel(w2_shard, dim=1, group=tp_group)
    golden = F.linear(F.gelu(F.linear(hidden, w1_full)), w2_full)

    num_splits_candidates = [2, 4, 8, 16]
    num_comm_sms_candidates = [2, 4, 8, 16, 24]

    comm_runtime = None
    best_ms = float("inf")
    best_cfg = (None, None)

    for num_splits in num_splits_candidates:
        for num_comm_sms in num_comm_sms_candidates:
            comm_info, block_offsets, block_shapes, signal_offsets, cum_tiles = _build_comm(
                plan_shape=(M, N),
                dtype=dtype,
                num_splits=num_splits,
                num_comm_sms=num_comm_sms,
                world_size=WORLD_SIZE,
                rank=rank,
            )

            if comm_runtime is None:
                comm_runtime = CommContext(rank, comm_info, multi_info=True)
            else:
                comm_runtime.update_comm_info(comm_info)

            signal_ptr = comm_runtime.compute_signal_bufs[comm_runtime.local_rank]
            counter_ptr = torch.zeros((signal_ptr.numel(),), device="cuda", dtype=torch.int32)

            def fused_domino_alike_mlp():
                comm_runtime.reset_signals()
                counter_ptr.zero_()

                comm_runtime.start_after(torch.cuda.current_stream())

                src_buf = comm_runtime.comm_buffers["src"][comm_runtime.local_rank]
                compute_stream = torch.cuda.current_stream()

                act = F.gelu(torch.matmul(hidden, w1_shard.t()))
                with torch.cuda.stream(compute_stream):
                    gemm_producer(
                        act,
                        w2_shard,
                        src_buf,
                        WORLD_SIZE,
                        132 - num_comm_sms,
                        cur_wave_sizes=block_shapes,
                        wave_offsets=block_offsets,
                        signal_offsets=signal_offsets,
                        signal_ptr=signal_ptr,
                        counter_ptr=counter_ptr,
                        cum_tiles=cum_tiles,
                    )

                comm_runtime.execute()
                comm_runtime.end_before(torch.cuda.current_stream())

                return comm_runtime.comm_buffers["dst"][comm_runtime.local_rank]

            # Run fused MLP and validate.
            out = fused_domino_alike_mlp()
            torch.testing.assert_close(out, golden, atol=1e-1, rtol=1e-1)
            dist_print(
                f"Rank {rank} correct for num_comm_sms={num_comm_sms}, num_splits={num_splits}",
                allowed_ranks="all",
                need_sync=True,
            )

            with group_profile("domino_alike_mlp_nvsharp", profile):
                sleep_async(1000)
                nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
                _, dur_ms = perf_func(fused_domino_alike_mlp, iters=iters, warmup_iters=warmup)

            dist_print(
                f"Rank {rank} latency: {dur_ms:.2f} ms (num_comm_sms={num_comm_sms}, num_splits={num_splits})",
                allowed_ranks="all",
                need_sync=True,
            )

            if dur_ms < best_ms:
                best_ms = dur_ms
                best_cfg = (num_comm_sms, num_splits)

    # "LLaMA-3.1-70B": {"M": 8192, "K": 28672, "N": 8192, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
    total_flops = 4 * 8192 * 28672 * 8192
    total_tflops = total_flops / 1e12
    per_rank_tflops = total_tflops / WORLD_SIZE
    tflops = per_rank_tflops / (best_ms / 1e3)
    dist_print(
        f"Best latency: {best_ms:.2f} ms with num_comm_sms={best_cfg[0]}, num_splits={best_cfg[1]}, TFLOPS={tflops:.2f}",
        allowed_ranks="all",
        need_sync=True,
    )

    del comm_runtime
    torch.cuda.synchronize()
    dist.barrier()
    finalize_distributed()


if __name__ == "__main__":
    main()
