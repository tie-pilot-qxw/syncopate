"""
Domino-alike single-microbatch MLP using our comm stack: fc1 -> GELU -> fc2 + all-reduce.
Uses the GEMM+all-reduce producer pattern from test_gemm_ar_collective for the second GEMM.
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
from syncopate.communication.common_descriptors import build_all_reduce_plan
from syncopate.interface.lowering import lower_comm_plan_to_raw_schedules
from tests.test_gemm_rs import configs
from triton_dist.utils import initialize_distributed, finalize_distributed, nvshmem_barrier_all_on_stream, perf_func, sleep_async, dist_print, group_profile


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
    M = config["M"] # tokens * batch_size
    FFN_hidden = config["K"]
    hidden_size = config["N"]

    world = dist.get_world_size(tp_group) if dist.is_initialized() else 1

    FFN_per_rank = FFN_hidden // world

    # Inputs and weights (column-parallel fc1, row-parallel fc2).
    hidden = torch.randn((M, hidden_size), device="cuda", dtype=dtype) / 10.0
    w1_shard = torch.randn((FFN_per_rank, hidden_size), device="cuda", dtype=dtype) / 10.0
    w2_shard = torch.randn((hidden_size, FFN_per_rank), device="cuda", dtype=dtype) / 10.0

    return hidden, w1_shard, w2_shard

def _build_comm(plan_shape: Tuple[int, int], dtype: torch.dtype, num_splits: int, num_comm_sms: int, tp_group):
    rank = tp_group.rank()
    world = tp_group.size()
    plans = {
        r: build_all_reduce_plan(
            shape=plan_shape,
            dtype=dtype,
            mesh_size=world,
            rank=r,
            buffer_name="src",
            num_all_reduces=num_splits,
            split_axis=0,
            compute_producer=True,
        )
        for r in range(world)
    }

    generator = CommGenerator(plans)
    generator.plan_signals()
    lowered = lower_comm_plan_to_raw_schedules(generator)
    schedule = lowered[rank]["src"]

    # Skip the dummy head entry (src=dst bug workaround) like test_gemm_ar_collective.
    block_offsets_list = schedule.gen_block_offset_lists()[1:]
    block_shapes_list = schedule.gen_block_shape_lists()[1:]
    signal_offsets_list = schedule.gen_signal_lists()[1:]

    comm_info = generator.generate_code_for_plan()
    comm_info.local_world_size = comm_info.world_size
    comm_info.num_copy_sms = num_comm_sms
    comm_info.need_green_ctx = True

    comm_runtime = CommContext(rank, comm_info)
    block_offsets = torch.tensor(block_offsets_list, device="cuda", dtype=torch.int32)
    block_shapes = torch.tensor(block_shapes_list, device="cuda", dtype=torch.int32)
    signal_offsets = torch.tensor(signal_offsets_list, device="cuda", dtype=torch.int32)
    signal_ptr = comm_runtime.compute_signal_bufs[comm_runtime.local_rank]
    counter_ptr = torch.zeros((signal_ptr.numel(),), device="cuda", dtype=torch.int32)
    cum_tiles = sum(int(shape[0]) * int(shape[1]) for shape in block_shapes_list)

    return comm_runtime, block_offsets, block_shapes, signal_offsets, signal_ptr, counter_ptr, cum_tiles


def main():
    gemm_producer = _load_transformed_gemm_producer()

    tp_group = initialize_distributed()
    rank = tp_group.rank()
    world = tp_group.size()
    torch.cuda.set_device(rank % torch.cuda.device_count())

    dtype = torch.float16
    hidden, w1_shard, w2_shard = _build_mlp_tensors("Qwen2-72B", dtype, tp_group)

    dist_print(f"hidden shape: {hidden.shape}, w1_shard shape: {w1_shard.shape}, w2_shard shape: {w2_shard.shape}")

    M, N = hidden.shape
    K_per_rank = w1_shard.shape[0]
    num_comm_sms = 24
    num_splits = 8
    warmup = 5
    iters = 10
    profile = False

    comm_runtime, block_offsets, block_shapes, signal_offsets, signal_ptr, counter_ptr, cum_tiles = _build_comm(
        plan_shape=(M, N), dtype=dtype, num_splits=num_splits, num_comm_sms=num_comm_sms, tp_group=tp_group
    )

    # Broadcast input so all ranks use identical hidden states for validation.
    dist.broadcast(hidden, src=0, group=tp_group)

    def fused_domino_alike_mlp():
        comm_runtime.reset_signals()
        counter_ptr.zero_()

        comm_runtime.start_after(torch.cuda.current_stream())

        src_buf = comm_runtime.comm_buffers["src"][comm_runtime.local_rank]

        
        act = F.gelu(torch.matmul(hidden, w1_shard.t()))
        compute_stream = comm_runtime.compute_stream
        compute_stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(compute_stream):
            gemm_producer(
                act,
                w2_shard,
                src_buf,
                world,
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

        return comm_runtime.comm_buffers["src"][comm_runtime.local_rank]

    # Run fused MLP.
    out = fused_domino_alike_mlp()

    # Golden: gather weights and compute full MLP locally, then compare.
    w1_full = _gather_weight_row_parallel(w1_shard, dim=0, group=tp_group)
    w2_full = _gather_weight_row_parallel(w2_shard, dim=1, group=tp_group)
    golden = F.linear(F.gelu(F.linear(hidden, w1_full)), w2_full)

    torch.testing.assert_close(out, golden, atol=1e-1, rtol=1e-1)
    dist_print(f"Rank {rank} domino-alike MLP collective correct", allowed_ranks="all", need_sync=True)


    with group_profile("domino_alike_mlp", profile):
        sleep_async(1000)
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        _, dur_ms = perf_func(fused_domino_alike_mlp, iters=iters, warmup_iters=warmup)
    dist_print(f"Rank {rank} latency: {dur_ms:.2f} ms", allowed_ranks="all", need_sync=True)

    del comm_runtime
    torch.cuda.synchronize()
    dist.barrier()
    finalize_distributed()


if __name__ == "__main__":
    main()
