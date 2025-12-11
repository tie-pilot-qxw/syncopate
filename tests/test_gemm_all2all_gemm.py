"""
GEMM -> all-to-all front half using transformed persistent GEMM producer.
We launch the producer GEMM (transformed via AnnotationTransformer) into the
communication source buffer, run all-to-all, and verify the received shard.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.distributed as dist

from syncopate.communication.code_gen import CodeGenOptions, CommGenerator
from syncopate.communication.comm_runtime.communication_context import CommContext
from syncopate.communication.common_descriptors import build_all_to_all_plan
from syncopate.communication.common_descriptors.all_to_all import build_all_to_all_plan_dim
from syncopate.interface.lowering import lower_comm_plan_to_raw_schedules
from syncopate.computation.transform import AnnotationTransformer
from triton_dist.kernels.nvidia.common_ops import _wait_eq_cuda
from triton_dist.utils import (
    initialize_distributed,
    finalize_distributed,
    nvshmem_barrier_all_on_stream,
    perf_func,
    dist_print,
    group_profile,
)


def _load_transformed_gemm_producer():
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

def _load_transformed_gemm_consumer():
    transformer = AnnotationTransformer(enable_consumer=True, consumer_descriptors=("a_desc",))
    example_path = Path("tests/computation/transform/examples/example_splitk_gemm.py")
    transformed_source = transformer.transform(example_path.read_text())
    generated_path = Path("/tmp/" + example_path.name.replace(".py", "_consumer_transformed.py"))
    try:
        with open(generated_path, "r") as f:
            existing = f.read()
        if existing != transformed_source:
            with open(generated_path, "w") as f:
                f.write(transformed_source)
    except FileNotFoundError:
        with open(generated_path, "w") as f:
            f.write(transformed_source)

    spec = importlib.util.spec_from_file_location("generated_gemm_consumer", generated_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["generated_gemm_consumer"] = module
    spec.loader.exec_module(module)
    return module.gemm

configs = {
    "LLaMA-7B": {"M": 8192, "K": 11008, "N": 4096, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
    "LLaMA-3.1-8B": {"M": 8192, "K": 14336, "N": 4096, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
    "LLaMA-3.1-70B": {"M": 8192, "K": 28672, "N": 8192, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
    "LLaMA-3.1-405B": {"M": 8192, "K": 53248, "N": 16384, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
    "Qwen2-72B": {"M": 8192, "K": 29568, "N": 8192, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
}


def _pad_dim(x: int, block: int) -> int:
    return (x + block - 1) // block * block


def _build_schedules(comm_generator: CommGenerator, rank: int, hidden: int):
    lowered = lower_comm_plan_to_raw_schedules(comm_generator)
    schedules = lowered[rank]
    src_sched = schedules["src"]
    dst_sched = schedules["dst"]

    # the schedule for first gemm is trival
    prod_offsets = torch.tensor(src_sched.gen_block_offset_lists(), device="cuda", dtype=torch.int32)
    prod_shapes = torch.tensor(src_sched.gen_block_shape_lists(), device="cuda", dtype=torch.int32)
    prod_signals = torch.tensor(src_sched.gen_signal_lists(), device="cuda", dtype=torch.int32)


    dst_block_offset_lists = dst_sched.gen_block_offset_lists()
    dst_block_shape_lists = dst_sched.gen_block_shape_lists()
    dst_signal_lists = dst_sched.gen_signal_lists()

    dist_print(f"rank {rank} dst block offsets: {dst_block_offset_lists}, shapes: {dst_block_shape_lists}, signals: {dst_signal_lists}", allowed_ranks="all")

    transformed_dst_offsets = []
    transformed_dst_shapes = []

    for i in range(len(dst_block_shape_lists)):
        block_shape = [dst_block_shape_lists[i][0], hidden, dst_block_shape_lists[i][1]]
        block_offset = [dst_block_offset_lists[i][0], 0, dst_block_offset_lists[i][1]]
        transformed_dst_offsets.append(block_offset)
        transformed_dst_shapes.append(block_shape)
        
    dst_offsets = torch.tensor(transformed_dst_offsets, device="cuda", dtype=torch.int32)
    dst_shapes = torch.tensor(transformed_dst_shapes, device="cuda", dtype=torch.int32)
    dst_signals = torch.tensor(dst_signal_lists, device="cuda", dtype=torch.int32)
    
    # dist_print(f"rank {rank} transformed dst offsets: {dst_offsets}, shapes: {dst_shapes}, signals: {dst_signals}", allowed_ranks="all", need_sync=True)

    return (prod_offsets, prod_shapes, prod_signals), (dst_offsets, dst_shapes, dst_signals)


def fused_gemm_all2all(
    x: torch.Tensor,
    w1_shard: torch.Tensor,
    w2: torch.Tensor,
    comm_runtime: CommContext,
    prod_sched,
    dst_sched,
    bm: int,
    bn: int,
    bk: int,
    stages: int,
    cum_tiles_prod: int,
    cum_tiles_dst: int,
    OVERLAP_SECOND_GEMM: bool = True
):
    """
    Producer GEMM -> all-to-all.
    """
    comm_runtime.reset_signals()

    prod_offsets, prod_shapes, prod_signals = prod_sched
    dst_offsets, dst_shapes, dst_signals = dst_sched

    src_buf = comm_runtime.comm_buffers["src"][comm_runtime.local_rank]
    dst_buf = comm_runtime.comm_buffers["dst"][comm_runtime.local_rank]
    compute_signal_ptr = comm_runtime.compute_signal_bufs[comm_runtime.local_rank]
    recv_signals = comm_runtime.recv_signal_bufs[comm_runtime.local_rank]
    counter_ptr = torch.zeros((compute_signal_ptr.numel(),), device="cuda", dtype=torch.int32)

    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    comm_runtime.start_after(torch.cuda.current_stream())


    # Launch transformed producer GEMM.
    gemm_producer(
        x,
        w1_shard,
        src_buf,
        comm_runtime.info.world_size,
        132 - comm_runtime.info.num_copy_sms,  # num_gemm_sms
        cur_wave_sizes=prod_shapes,
        wave_offsets=prod_offsets,
        signal_offsets=prod_signals,
        signal_ptr=compute_signal_ptr,
        counter_ptr=counter_ptr,
        cum_tiles=cum_tiles_prod,
        BLOCK_SIZE_M=bm,
        BLOCK_SIZE_N=bn,
        BLOCK_SIZE_K=bk,
        GROUP_SIZE_M=8,
        STAGES=stages,
    )

    # dist_print(f"Rank {comm_runtime.local_rank} compute signal ptr: {compute_signal_ptr}", allowed_ranks="all", need_sync=True)
    # Kick off communication after reset to allow overlap.
    comm_runtime.execute()

    if OVERLAP_SECOND_GEMM:
        output = torch.zeros((dst_buf.shape[0], w2.shape[0]), device="cuda", dtype=dst_buf.dtype)
        gemm_consumer(
            dst_buf,
            w2,
            output,
            comm_runtime.info.world_size,
            132 ,  # num_gemm_sms
            cur_wave_sizes=dst_shapes,
            wave_offsets=dst_offsets,
            signal_offsets=dst_signals,
            signal_ptr=recv_signals,
            cum_tiles=cum_tiles_dst,
        )
        return output
    
    comm_runtime.end_before(torch.cuda.current_stream())


    # Wait for all recv signals to ensure dst buffer is ready.
    for offset in range(recv_signals.shape[0]):
        _wait_eq_cuda(recv_signals[offset], 1, torch.cuda.current_stream())
    
 
    return torch.matmul(dst_buf, w2.T)

    # return dst_buf


if __name__ == "__main__":
    gemm_producer = _load_transformed_gemm_producer()
    gemm_consumer = _load_transformed_gemm_consumer()
    copy_sms = 12

    TP_GROUP = initialize_distributed()
    rank = TP_GROUP.rank()
    world = TP_GROUP.size()

    config = configs["LLaMA-3.1-70B"]
    M = config["M"]
    ffn_hidden = config["K"]
    hidden = config["N"]
    bm, bn, bk, stages = config["BM"], config["BN"], config["BK"], config["Stage"]

    assert M % world == 0 and ffn_hidden % world == 0, "Sequence and FFN must be divisible by world size."

    seq_local = M // world
    seq_local_pad = _pad_dim(seq_local, bm)
    seq_pad = seq_local_pad * world

    ffn_local = ffn_hidden // world
    ffn_local_pad = _pad_dim(ffn_local, bn)
    ffn_hidden_pad = ffn_local_pad * world

    hidden_pad = _pad_dim(hidden, bk)

    torch.manual_seed(0)
    x = torch.randn((seq_pad, hidden_pad), device="cuda", dtype=torch.float16) / 10.0
    w1_shard = torch.randn((ffn_local_pad, hidden_pad), device="cuda", dtype=torch.float16) / 10.0
    w2 = torch.randn((hidden_pad, ffn_hidden_pad), device="cuda", dtype=torch.float16) / 10.0

    # Zero out padded regions so validation ignores padding artifacts.
    if seq_pad > M:
        x[M:, :] = 0
    if ffn_local_pad > ffn_local:
        w1_shard[ffn_local:, :] = 0
    if hidden_pad > hidden:
        w1_shard[:, hidden:] = 0

    # Make inputs consistent across ranks for validation.
    dist.broadcast(x, src=0)
    dist.broadcast(w2, src=0)

    plans = {
        r: build_all_to_all_plan_dim(
            shape=(seq_pad, ffn_local_pad),
            dtype=x.dtype,
            mesh_size=world,
            rank=r,
            src_buffer="src",
            dst_buffer="dst",
            src_split_axis=1,
            dst_split_axis=0,
            transfer_kind="push",
            compute_producer=True,
        )
        for r in range(world)
    }

    generator = CommGenerator(plans)
    generator.plan_signals()

    options = CodeGenOptions(copy_engine=False)
    comm_info = generator.generate_code_for_plan(options)
    dist_print(comm_info)
    comm_info.local_world_size = comm_info.world_size  # single-node assumption
    comm_info.num_copy_sms = copy_sms
    comm_runtime = CommContext(rank, comm_info)


    prod_sched, dst_sched = _build_schedules(generator, rank, hidden)
    prod_shapes_tensor = prod_sched[1]
    cum_tiles = int((prod_shapes_tensor[:, 0] * prod_shapes_tensor[:, 1]).sum().item())
    dst_shapes_tensor = dst_sched[1]
    cum_tiles_dst = int((dst_shapes_tensor[:, 0] * dst_shapes_tensor[:, 1]).sum().item())

    # dist_print(f"rank {rank} prod_sched {prod_sched}, cum_tiles {cum_tiles}", allowed_ranks="all")


    def run_once():
        return fused_gemm_all2all(
            x,
            w1_shard,
            w2,
            comm_runtime,
            prod_sched,
            dst_sched,
            bm=bm,
            bn=bn,
            bk=bk,
            stages=stages,
            cum_tiles_prod=cum_tiles,
            cum_tiles_dst=cum_tiles_dst,
        )

    # Correctness: compare to reference GEMM then torch all_to_all_single.
    out_local = run_once()

    dist_print(f"out_local shape: {out_local.shape}", allowed_ranks="all", need_sync=True)

    w1_full = torch.empty((ffn_hidden_pad, hidden_pad), device="cuda", dtype=x.dtype)
    dist.all_gather_into_tensor(w1_full, w1_shard)
    x2_full = torch.matmul(x, w1_full.t())  # [seq_pad, ffn_hidden_pad]
    x3_full = torch.matmul(x2_full, w2.T)  # [seq_pad, hidden_pad]
    # dist_print(f"x2 shape: {x2_full.shape}", allowed_ranks="all", need_sync=True)
    # Reference all-to-all to match comm buffer layout.
    # x2_split = x2_full.reshape(world, seq_local_pad, ffn_hidden_pad)[rank, :, :]
    x3_partial = x3_full.reshape(world, seq_local_pad, hidden_pad)[rank, :, :]

    torch.testing.assert_close(
        out_local,
        x3_partial,
        atol=5e-1,
        rtol=5e-1,
    )
    dist_print(f"Rank {rank} gemm+all2all front half correct", allowed_ranks="all", need_sync=True)

    # Simple perf sample.

    # for num_comm_sms in range(4, 16):
    with group_profile(f"GEMM+all2all+GEMM", False):
        # comm_runtime.info.num_copy_sms = num_comm_sms
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        _, dur_ms = perf_func(run_once, iters=10, warmup_iters=5)

    total_flops = M * ffn_hidden * hidden * 2 + M * hidden * ffn_hidden * 2
    per_gpu_flops = total_flops / world
    tflops = per_gpu_flops / 1e12 / (dur_ms / 1e3)

    dist_print(
        f"Rank {rank} latency: {dur_ms:.2f} ms tflops: {tflops:.2f} , num_comm_sms: {comm_runtime.info.num_copy_sms}",
        allowed_ranks="all",
        need_sync=True,
    )

    del comm_runtime
    torch.cuda.synchronize()
    dist.barrier()
    finalize_distributed()
