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
    sleep_async,
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


def _pad_dim(x: int, block: int) -> int:
    return (x + block - 1) // block * block


def _build_schedules(comm_generator: CommGenerator, rank: int, hidden: int, headdim: int, seq_len_local: int):
    lowered = lower_comm_plan_to_raw_schedules(comm_generator)
    schedules = lowered[rank]
    dst_sched = schedules["dst"]

    dst_block_offset_lists = dst_sched.gen_block_offset_lists()
    dst_block_shape_lists = dst_sched.gen_block_shape_lists()
    dst_signal_lists = dst_sched.gen_signal_lists()

    dist_print(f"rank {rank} dst block offsets: {dst_block_offset_lists}, shapes: {dst_block_shape_lists}, signals: {dst_signal_lists}", allowed_ranks="all")

    transformed_dst_offsets = []
    transformed_dst_shapes = []

    for i in range(len(dst_block_shape_lists)):
        block_shape = [dst_block_shape_lists[i][0] * dst_block_shape_lists[i][1], hidden, dst_block_shape_lists[i][2] * dst_block_shape_lists[i][3]]
        block_offset = [0, 0, dst_block_offset_lists[i][2] * headdim + dst_block_offset_lists[i][3]]
        transformed_dst_offsets.append(block_offset)
        transformed_dst_shapes.append(block_shape)
        
    dst_offsets = torch.tensor(transformed_dst_offsets, device="cuda", dtype=torch.int32)
    dst_shapes = torch.tensor(transformed_dst_shapes, device="cuda", dtype=torch.int32)
    dst_signals = torch.tensor(dst_signal_lists, device="cuda", dtype=torch.int32)
    
    dist_print(f"rank {rank} transformed dst offsets: {dst_offsets}, shapes: {dst_shapes}, signals: {dst_signals}", allowed_ranks="all", need_sync=True)

    return (dst_offsets, dst_shapes, dst_signals)


def fused_all2all_gemm(
    w: torch.Tensor,
    comm_runtime: CommContext,
    dst_sched,
    bm: int,
    bn: int,
    bk: int,
    stages: int,
    cum_tiles_dst: int,
    OVERLAP_SECOND_GEMM: bool = True
):
    """
    Producer GEMM -> all-to-all.
    """
    comm_runtime.reset_signals()

    dst_offsets, dst_shapes, dst_signals = dst_sched

    dst_buf = comm_runtime.comm_buffers["dst"][comm_runtime.local_rank]
    gemm_input = dst_buf.reshape(-1, w.shape[1])  # [M_local, K]

    recv_signals = comm_runtime.recv_signal_bufs[comm_runtime.local_rank]

    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    comm_runtime.start_after(torch.cuda.current_stream())


    # dist_print(f"Rank {comm_runtime.local_rank} compute signal ptr: {compute_signal_ptr}", allowed_ranks="all", need_sync=True)
    # Kick off communication after reset to allow overlap.
    comm_runtime.execute()

    output = torch.zeros((gemm_input.shape[0], w.shape[0]), device="cuda", dtype=dst_buf.dtype)
    gemm_consumer(
        gemm_input,
        w,
        output,
        comm_runtime.info.world_size,
        132 - comm_runtime.info.num_copy_sms,
        cur_wave_sizes=dst_shapes,
        wave_offsets=dst_offsets,
        signal_offsets=dst_signals,
        signal_ptr=recv_signals,
        cum_tiles=cum_tiles_dst,
        BLOCK_SIZE_M=bm,
        BLOCK_SIZE_N=bn,
        BLOCK_SIZE_K=bk,
        GROUP_SIZE_M=8,
        STAGES=stages,
    )
    return output

def nccl_all_to_all(o, world):
    B, S, H_local, D = o.shape
    S_local = S // world

    # input shape [B, S, H_local, D]
    o = o.reshape(B, world, S_local, H_local, D) # world * S_local
    send_buf = o.permute(1, 0, 2, 3, 4).contiguous()
    # shape = [world, B, S_local, H_local, D]
    recv_buf = torch.empty_like(send_buf)
    dist.all_to_all_single(recv_buf, send_buf)
    # recv_buf: [world, B, S_local, H_local, D]
    x = recv_buf.permute(1, 2, 0, 3, 4).contiguous()
    # now shape = [B, S_local, world, H_local, D]
    x = x.reshape(B, S_local, H_local * world, D)
    # shape = [B, S_local, H, D]
    return x

if __name__ == "__main__":
    configs = {
        "LLaMA-7B": {"M": 32*1024, "N": 11008, "K": 4096, "BM": 128, "BN": 128, "BK": 64, "Stage": 5},
        "LLaMA-3.1-8B": {"M": 32*1024, "N": 14336, "K": 4096, "BM": 128, "BN": 128, "BK": 64, "Stage": 5},
        "LLaMA-3.1-70B": {"M": 32*1024, "N": 28672, "K": 8192, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
        "LLaMA-3.1-405B": {"M": 32*1024, "N": 53248, "K": 16384, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
        "Qwen2-72B": {"M": 32*1024, "N": 29568, "K": 8192, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
    }
    
    gemm_consumer = _load_transformed_gemm_consumer()
    copy_sms = 12

    TP_GROUP = initialize_distributed()
    rank = TP_GROUP.rank()
    world = TP_GROUP.size()

    config = configs["LLaMA-3.1-70B"]

    M = world * 4096 # per rank 4096
    HIDDEN = config["K"]
    head_dim = 128
    batch_size = 2
    assert M % batch_size == 0
    seq_len = M // batch_size
    assert seq_len % world == 0
    seq_len_local = seq_len // world

    assert HIDDEN % head_dim == 0
    num_heads = HIDDEN // head_dim

    assert num_heads % world == 0
    num_heads_local = num_heads // world



    bm, bn, bk, stages = config["BM"], config["BN"], config["BK"], config["Stage"]


    torch.manual_seed(0)
    attn_out = torch.randn((batch_size, seq_len, num_heads_local, head_dim), device="cuda", dtype=torch.float16) / 10.0
    w = torch.randn((HIDDEN, HIDDEN), device="cuda", dtype=torch.float16) / 10.0


    dist.broadcast(w, src=0) # output projection weight shared across ranks

    plans = {
        r: build_all_to_all_plan_dim(
            shape=(batch_size, seq_len, num_heads_local, head_dim),
            dtype=attn_out.dtype,
            mesh_size=world,
            rank=r,
            src_buffer="src",
            dst_buffer="dst",
            src_split_axis=2,
            dst_split_axis=1,
            transfer_kind="push",
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

    comm_runtime.comm_buffers["src"][rank].copy_(attn_out)


    dst_sched = _build_schedules(generator, rank, HIDDEN, head_dim, seq_len_local)

    dst_shapes_tensor = dst_sched[1]
    cum_tiles_dst = int((dst_shapes_tensor[:, 0] * dst_shapes_tensor[:, 1]).sum().item())

    # dist_print(f"rank {rank} prod_sched {prod_sched}, cum_tiles {cum_tiles}", allowed_ranks="all")


    def run_once():
        return fused_all2all_gemm(
            w,
            comm_runtime,
            dst_sched,
            bm=bm,
            bn=bn,
            bk=bk,
            stages=stages,
            cum_tiles_dst=cum_tiles_dst,
        )

    # Correctness: compare to reference GEMM then torch all_to_all_single.
    out_local = run_once()

    dist_print(f"out_local shape: {out_local.shape}", allowed_ranks="all", need_sync=True)

    truth_a2a = nccl_all_to_all(attn_out, world)
    golden_out = torch.matmul(truth_a2a.reshape(-1, HIDDEN), w.T)
    dist_print(f"golden_out shape: {golden_out.shape}", allowed_ranks="all", need_sync=True)

    torch.testing.assert_close(
        out_local,
        golden_out,
        atol=5e-1,
        rtol=5e-1,
    )
    dist_print(f"Rank {rank} a2a+gemm correct", allowed_ranks="all", need_sync=True)

    # Simple perf sample.

    # for num_comm_sms in range(4, 16):
    with group_profile(f"all2all+GEMM", False):
        # comm_runtime.info.num_copy_sms = num_comm_sms
        sleep_async(1000)
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        _, dur_ms = perf_func(run_once, iters=10, warmup_iters=5)

        total_flops = M * HIDDEN * HIDDEN * 2
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
