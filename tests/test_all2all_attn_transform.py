"""
All-to-all + attention (Ulysses HP style) using AnnotationTransformer.
Uses B×H×S×D layout so sequence shards are exchanged into head shards.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Dict

import torch

from syncopate.communication.code_gen import CommGenerator
from syncopate.communication.comm_runtime.communication_context import CommContext
from syncopate.communication.common_descriptors import build_all_to_all_plan_dim
from syncopate.communication.descriptor import DevicePlan
from syncopate.computation.transform import AnnotationTransformer
from syncopate.interface.lowering import lower_comm_plan_to_raw_schedules
from flash_attn import flash_attn_func
from triton_dist.utils import (
    dist_print,
    finalize_distributed,
    initialize_distributed,
    nvshmem_barrier_all_on_stream,
    perf_func,
    sleep_async,
    group_profile,
)
import torch.distributed as dist
from triton_dist.kernels.nvidia.common_ops import _wait_eq_cuda


def _load_transformed_attention(example_path: Path):
    transformer = AnnotationTransformer(enable_consumer=True)
    transformed = transformer.transform(example_path.read_text())

    generated_path = Path("/tmp" + example_path.name.replace(".py", "_transformed.py"))
    with open(generated_path, "r") as f:
        existing = f.read()
    if existing != transformed:
        with open(generated_path, "w") as f:
            f.write(transformed)

    spec = importlib.util.spec_from_file_location("generated_tmp_kernel", generated_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["generated_tmp_kernel"] = module
    spec.loader.exec_module(module)
    return module.attention_forward


def _derive_wave_metadata(schedule, axes):
    """
    We need a quite special schedule for all-to-all attention:
    arrival order:
        KV1 KV2 KV3 ... KVn
     Q1  0   1   2  ... n-1
     Q2  1   1   2  ... n-1
     Q3  2   2   2  ... n-1
     ...
     Qn  n-1 n-1 n-1 ... n-1
    
    So each wave need multiple blocks
    """
    block_offsets_src = schedule.gen_block_offset_lists()
    block_shapes_src = schedule.gen_block_shape_lists()
    signal_offsets_src = schedule.gen_signal_lists()

    offsets = []
    shapes = []
    signal_offsets = []
    cum_counts = []
    total = 0

    b_idx = axes["batch"]
    h_idx = axes["head"]
    s_idx = axes["seq"]

    for wave_id in range(len(block_shapes_src)):
        offs_wave = block_offsets_src[wave_id]
        shape_wave = block_shapes_src[wave_id]
        offsets.append([offs_wave[b_idx], offs_wave[h_idx], offs_wave[s_idx], offs_wave[s_idx]])
        shapes.append([shape_wave[b_idx], shape_wave[h_idx], shape_wave[s_idx], shape_wave[s_idx]])
        signal_offsets.append(signal_offsets_src[wave_id])
        total += shape_wave[b_idx] * shape_wave[h_idx] * shape_wave[s_idx]
        cum_counts.append(total)

        for past_wave_id in range(wave_id):
            offs_past = block_offsets_src[past_wave_id]
            shape_past = block_shapes_src[past_wave_id]

            # KV from current wave, Q from past wave
            offsets.append([offs_past[b_idx], offs_past[h_idx], offs_past[s_idx], offs_wave[s_idx]])
            shapes.append([shape_past[b_idx], shape_past[h_idx], shape_past[s_idx], shape_wave[s_idx]])
            signal_offsets.append(signal_offsets_src[wave_id])
            total += shape_past[b_idx] * shape_past[h_idx] * shape_wave[s_idx]
            cum_counts.append(total)

            # Q from current wave, K/V from past wave
            offsets.append([offs_past[b_idx], offs_past[h_idx], offs_wave[s_idx], offs_past[s_idx]])
            shapes.append([shape_past[b_idx], shape_past[h_idx], shape_wave[s_idx], shape_past[s_idx]])
            signal_offsets.append(signal_offsets_src[wave_id])
            total += shape_past[b_idx] * shape_past[h_idx] * shape_past[s_idx]
            cum_counts.append(total)

    device = torch.device("cuda")
    offsets_t = torch.tensor(offsets, device=device, dtype=torch.int32)
    shapes_t = torch.tensor(shapes, device=device, dtype=torch.int32)
    signal_offsets_t = torch.tensor(signal_offsets, device=device, dtype=torch.int32)
    cum_counts_t = torch.tensor(cum_counts, device=device, dtype=torch.int32)
    return offsets_t, shapes_t, signal_offsets_t, cum_counts_t


def _fused_all_to_all_attention(attn_fn, comm_runtime, wave_kwargs, sm_scale, graph):
    compute_stream = torch.cuda.current_stream()
    communication_stream = torch.cuda.Stream()
    
    comm_runtime.reset_signals()

    communication_stream.wait_stream(compute_stream)
    with torch.cuda.stream(communication_stream):
        nvshmem_barrier_all_on_stream(communication_stream)
        graph.replay()
    
    # nvshmem_barrier_all_on_stream(compute_stream)
    # comm_runtime.start_after(compute_stream)
    dst_buf = comm_runtime.comm_buffers["dst"][comm_runtime.local_rank]
    q_dst, k_dst, v_dst = dst_buf[0], dst_buf[1], dst_buf[2]
    
    # comm_runtime.execute()

    # this is because a strange thing: device2device copy are not using copy engine,
    # so we need to make sure all copies are done before compute
    # other wise, there will be deadlock unless setting CUDA_DEVICE_MAX_CONNECTIONS=1 to prevent concurrent kernel launch
    _wait_eq_cuda(wave_kwargs["signal_ptr"][0], 1, compute_stream)
    out = attn_fn(q_dst, k_dst, v_dst, sm_scale, **wave_kwargs)

    compute_stream.wait_stream(communication_stream)
    # comm_runtime.end_before(compute_stream)
    return out


def _to_flash_order(tensor):
    # flash-attn expects B×S×H×D
    return tensor.permute(0, 2, 1, 3).contiguous()


def _from_flash_order(tensor):
    return tensor.permute(0, 2, 1, 3).contiguous()

def nccl_all_to_all(qkv_src, world):
    _, B, H, S_local, D = qkv_src.shape
    H_local = H // world

    # ---------------------------------------------------------------
    # 1) Move head and seq dimensions to the end to make chunking easier
    #    [3, B, H, S_local, D] → [3, B, S_local, H, D]
    # ---------------------------------------------------------------
    x = qkv_src.permute(0, 1, 3, 2, 4)   # [3, B, S_local, H, D]

    # ---------------------------------------------------------------
    # 2) Split along the head dimension into world slices (units for all-to-all)
    #    Shape after split: world chunks of [3, B, S_local, H_local, D]
    # ---------------------------------------------------------------
    x = x.reshape(3, B, S_local, world, H_local, D)

    # Send buffer: flatten world dimension to the 0th (rank) axis
    send_buf = x.permute(3, 0, 1, 2, 4, 5).contiguous()  
    # shape = [world, 3, B, S_local, H_local, D]

    # ---------------------------------------------------------------
    # 3) Perform all-to-all: each rank receives the same-shaped recv_buf
    # ---------------------------------------------------------------
    recv_buf = torch.empty_like(send_buf)

    dist.all_to_all_single(recv_buf, send_buf)  
    # recv_buf: [world, 3, B, S_local, H_local, D]
    # recv_buf[i] is the chunk sent from rank i

    # ---------------------------------------------------------------
    # 4) Stitch chunks back together (concatenate world S_local blocks into S)
    #    Target shape: [3, B, H_local, S, D]
    # ---------------------------------------------------------------
    # First merge the world dimension into the S dimension
    x = recv_buf.permute(1, 2, 0, 3, 4, 5).contiguous()
    # now shape = [3, B, world, S_local, H_local, D]

    x = x.reshape(3, B, world * S_local, H_local, D)  
    # shape = [3, B, S, H_local, D]

    # Finally permute back to the target layout
    qkv_dst = x.permute(0, 1, 3, 2, 4).contiguous()
    # shape = [3, B, H_local, S, D]
    return qkv_dst

def main():
    attention_fn = _load_transformed_attention(Path("tests/computation/transform/examples/example_attn.py"))

    WORLD_SIZE = int(os.getenv("WORLD_SIZE", "4"))
    LOCAL_WORLD_SIZE = int(os.getenv("LOCAL_WORLD_SIZE", "4"))

    TP_GROUP = initialize_distributed()
    rank = TP_GROUP.rank()
    torch.cuda.set_device(rank % LOCAL_WORLD_SIZE)
    dist_print("Running bhsd layout all-to-all attention overlap")

    B = 2
    H = 16
    SEQ = WORLD_SIZE * 8192
    DIM = 128
    dtype = torch.float16

    assert SEQ % WORLD_SIZE == 0
    assert H % WORLD_SIZE == 0

    SEQ_per_rank = SEQ // WORLD_SIZE
    H_per_rank = H // WORLD_SIZE

    q_local = torch.randn((B, H, SEQ_per_rank, DIM), device="cuda", dtype=dtype)
    k_local = torch.randn((B, H, SEQ_per_rank, DIM), device="cuda", dtype=dtype)
    v_local = torch.randn((B, H, SEQ_per_rank, DIM), device="cuda", dtype=dtype)
    qkv_src = torch.stack((q_local, k_local, v_local), dim=0)  # [3, B, H, S/w, D]
    golden_qkv = nccl_all_to_all(qkv_src, WORLD_SIZE)  # [3, B, H/w, S, D]

    q_flash = _to_flash_order(golden_qkv[0])
    k_flash = _to_flash_order(golden_qkv[1])
    v_flash = _to_flash_order(golden_qkv[2])
    out_flash = flash_attn_func(q_flash, k_flash, v_flash, softmax_scale=1.0)
    out_flash = _from_flash_order(out_flash) # [B, S, H/w, D]
    dist_print(f"out_flash shape: {out_flash.shape}")


    plan_shape = qkv_src.shape
    device_plans: Dict[int, DevicePlan] = {
        r: build_all_to_all_plan_dim(
            shape=plan_shape,
            dtype=dtype,
            mesh_size=WORLD_SIZE,
            rank=r,
            src_buffer="src",
            dst_buffer="dst",
            src_split_axis=3,  # sequence
            dst_split_axis=2,  # head
            transfer_kind="pull",
            compute_producer=False,
        )
        for r in range(WORLD_SIZE)
    }

    generator = CommGenerator(device_plans)
    generator.plan_signals()
    schedule = lower_comm_plan_to_raw_schedules(generator)[rank]["dst"]

    axes = {"batch": 1, "head": 2, "seq": 3}
    block_offsets, block_shapes, signal_offsets, cum_counts = _derive_wave_metadata(
        schedule, axes=axes
    )

    dist_print(f"block_offsets: {block_offsets}")
    dist_print(f"block_shapes: {block_shapes}")
    dist_print(f"signal_offsets: {signal_offsets}")
    dist_print(f"cum_counts: {cum_counts}")

    comm_info = generator.generate_code_for_plan()
    comm_info.local_world_size = comm_info.world_size  # intra-node testing
    comm_runtime = CommContext(rank, comm_info)
    comm_runtime.comm_buffers["dst"][comm_runtime.local_rank].zero_()

    src_buf = comm_runtime.comm_buffers["src"][comm_runtime.local_rank]
    src_buf.copy_(qkv_src)

    graph = comm_runtime.get_graph()

    comm_runtime.reset_signals()
    # nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    # graph.replay()
    # dist_print(f"signals after graph replay: {comm_runtime.recv_signal_bufs[comm_runtime.local_rank]}", allowed_ranks="all", need_sync=True)
    signal_ptr = comm_runtime.recv_signal_bufs[comm_runtime.local_rank]

    wave_kwargs = {
        "wave_offsets": block_offsets,
        "wave_sizes": block_shapes,
        "cum_wave_sizes": cum_counts,
        "signal_offsets": signal_offsets,
        "signal_ptr": signal_ptr,
        "NUM_WAVES": block_offsets.shape[0],
        "cum_tiles": cum_counts[-1].item(),
    }

    def func():
        return _fused_all_to_all_attention(attention_fn, comm_runtime, wave_kwargs, 1.0, graph)

    # comm_runtime.reset_signals()
    # nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    # comm_runtime.start_after(torch.cuda.current_stream())
    # comm_runtime.execute()
    # out_trion = attention_fn(
    #     comm_runtime.comm_buffers["dst"][comm_runtime.local_rank][0],
    #     comm_runtime.comm_buffers["dst"][comm_runtime.local_rank][1],
    #     comm_runtime.comm_buffers["dst"][comm_runtime.local_rank][2],
    #     1.0,
    #     **wave_kwargs,
    # )
    # comm_runtime.end_before(torch.cuda.current_stream())

    out_triton = func()


    dst_buf = comm_runtime.comm_buffers["dst"][comm_runtime.local_rank]
    dist_print(f"dst_buf shape: {dst_buf.shape}")

    torch.testing.assert_close(dst_buf, golden_qkv, atol=1e-2, rtol=1e-2)
    dist_print(f"Rank {rank} all_to_all results correct for bhsd layout")

    torch.testing.assert_close(out_triton, out_flash, atol=1e-2, rtol=1e-2)
    dist_print(f"Rank {rank} attention results correct for bhsd layout")

    with group_profile("all2all_attn", False, group=TP_GROUP):
        sleep_async(1000)
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        _, dur_ms = perf_func(func, iters=10, warmup_iters=5)
    flops_per_matmul = 2.0 * B * H * SEQ * SEQ * DIM
    total_flops = 2 * flops_per_matmul
    per_gpu_flops = total_flops / WORLD_SIZE
    tflops = per_gpu_flops / 1e12 / (dur_ms / 1e3)
    dist_print(f"layout=bhsd tflops: {tflops} time: {dur_ms}", allowed_ranks="all", need_sync=True)

    del comm_runtime
    finalize_distributed()


if __name__ == "__main__":
    main()
