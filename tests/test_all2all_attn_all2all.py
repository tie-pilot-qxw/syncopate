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

from syncopate.communication.code_gen import CodeGenOptions, CommGenerator
from syncopate.communication.comm_runtime.communication_context import CommContext
from syncopate.communication.common_descriptors import build_all_to_all_plan_axis
from syncopate.communication.common_descriptors.all_to_all import build_all_to_all_plan_dim
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


def _load_transformed_attention(example_path: Path):
    transformer = AnnotationTransformer(enable_producer=True, enable_consumer=True, consumer_descriptors=("desc_q",))
    transformed = transformer.transform(example_path.read_text())

    generated_path = Path("/tmp/" + example_path.name.replace(".py", "_consumer_producer_transformed.py"))
    try:
        with open(generated_path, "r") as f:
            existing = f.read()
        if existing != transformed:
            with open(generated_path, "w") as f:
                f.write(transformed)
    except FileNotFoundError:
        with open(generated_path, "w") as f:
            f.write(transformed)

    spec = importlib.util.spec_from_file_location("generated_tmp_kernel", generated_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["generated_tmp_kernel"] = module
    spec.loader.exec_module(module)
    return module.attention_forward


def _derive_wave_metadata(schedule, axes, kv_dim):
    block_offsets_src = schedule.gen_block_offset_lists()
    block_shapes_src = schedule.gen_block_shape_lists()
    signal_offsets_src = schedule.gen_signal_lists()
    dist_print(f"block_offsets_src: {block_offsets_src}")
    dist_print(f"block_shapes_src: {block_shapes_src}")
    dist_print(f"signal_offsets_src: {signal_offsets_src}")

    offsets = []
    shapes = []
    cum_counts = []
    total = 0

    b_idx = axes["batch"]
    h_idx = axes["head"]
    s_idx = axes["seq"]

    for offset, shape in zip(block_offsets_src, block_shapes_src):
        offsets.append([offset[b_idx], offset[h_idx], offset[s_idx], 0])
        shapes.append([shape[b_idx], shape[h_idx], shape[s_idx], kv_dim])
        total += (shape[b_idx] * shape[h_idx] * shape[s_idx])
        cum_counts.append(total)

    device = torch.device("cuda")
    offsets_t = torch.tensor(offsets, device=device, dtype=torch.int32)
    shapes_t = torch.tensor(shapes, device=device, dtype=torch.int32)
    signal_offsets_t = torch.tensor(signal_offsets_src, device=device, dtype=torch.int32)
    cum_counts_t = torch.tensor(cum_counts, device=device, dtype=torch.int32)
    return offsets_t, shapes_t, signal_offsets_t, cum_counts_t


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
    
def main():
    attention_fn = _load_transformed_attention(Path("tests/computation/transform/examples/example_attn_bshd_no_split.py"))

    WORLD_SIZE = int(os.getenv("WORLD_SIZE", "4"))
    LOCAL_WORLD_SIZE = int(os.getenv("LOCAL_WORLD_SIZE", "4"))

    TP_GROUP = initialize_distributed()
    rank = TP_GROUP.rank()
    # rank = 0
    torch.cuda.set_device(rank % LOCAL_WORLD_SIZE)

    comm_sms = 8

    B = 2
    H = 32
    SEQ = WORLD_SIZE * 2048
    DIM = 128
    dtype = torch.float16

    assert SEQ % WORLD_SIZE == 0
    assert H % WORLD_SIZE == 0

    SEQ_per_rank = SEQ // WORLD_SIZE
    H_per_rank = H // WORLD_SIZE

    q_local = torch.randn((B, SEQ, H_per_rank, DIM), device="cuda", dtype=dtype)
    k_local = torch.randn((B, SEQ, H_per_rank, DIM), device="cuda", dtype=dtype)
    v_local = torch.randn((B, SEQ, H_per_rank, DIM), device="cuda", dtype=dtype)
    out_flash = flash_attn_func(q_local, k_local, v_local, softmax_scale=1.0)
    golden_out = nccl_all_to_all(out_flash, WORLD_SIZE)  # [ B, S/w, H, D]

    dist_print(f"out_flash shape: {out_flash.shape}")


    # what we want: [B, S, H/w, D] -> all2all -> [B, S/w, H, D]
    device_plans: Dict[int, DevicePlan] = {
        r: build_all_to_all_plan_dim(
            shape=[B, SEQ, H_per_rank, DIM],
            dtype=dtype,
            mesh_size=WORLD_SIZE,
            rank=r,
            src_buffer="src",
            dst_buffer="dst",
            src_split_axis=2,  # head
            dst_split_axis=1,  # sequence
            transfer_kind="push",
            compute_producer=True,
        )
        for r in range(WORLD_SIZE)
    }

    # print device plans
    for r, plan in device_plans.items():
        dist_print(f"Device plan for rank {r}:\n{plan.pretty()}")

    generator = CommGenerator(device_plans)
    generator.plan_signals()
    schedule = lower_comm_plan_to_raw_schedules(generator)[rank]["src"]

    axes = {"batch": 0, "head": 2, "seq": 1}
    block_offsets, block_shapes, signal_offsets, cum_counts = _derive_wave_metadata(
        schedule, axes=axes, kv_dim=SEQ
    )

    dist_print(f"block_offsets: {block_offsets}")
    dist_print(f"block_shapes: {block_shapes}")
    dist_print(f"signal_offsets: {signal_offsets}")
    dist_print(f"cum_counts: {cum_counts}")

    option = CodeGenOptions(copy_engine=False)
    comm_info = generator.generate_code_for_plan(options=option)
    dist_print(f"Comm Info:\n{comm_info}")
    comm_info.local_world_size = comm_info.world_size  # intra-node testing

    comm_info.need_green_ctx = True
    comm_info.num_copy_sms = comm_sms

    comm_runtime = CommContext(rank, comm_info)

    src_buf = comm_runtime.comm_buffers["src"][comm_runtime.local_rank]
    dst_buf = comm_runtime.comm_buffers["dst"][comm_runtime.local_rank] # [B, w * Seq/w, H/w, D]
    dist_print(f"src_buf shape: {src_buf.shape}")
    dist_print(f"dst_buf shape: {dst_buf.shape}")
    # graph = comm_runtime.get_graph()


    signal_ptr = comm_runtime.compute_signal_bufs[comm_runtime.local_rank]
    counter_ptr = torch.zeros((signal_ptr.numel(),), device="cuda", dtype=torch.int32)
    
    consumer_signal_offset = torch.empty_like(signal_offsets)
    consumer_signal_offset.fill_(-1)
    consumer_signal_ptr = torch.empty_like(signal_ptr)

    wave_kwargs = {
        "wave_offsets": block_offsets,
        "wave_sizes": block_shapes,
        "cum_wave_sizes": cum_counts,
        "producer_signal_offsets": signal_offsets,
        "producer_signal_ptr": signal_ptr,
        "NUM_WAVES": block_offsets.shape[0],
        "consumer_signal_offsets": consumer_signal_offset,
        "consumer_signal_ptr": consumer_signal_ptr,
        "cum_tiles": cum_counts[-1].item(),
        "producer_counter_ptr": counter_ptr,
    }

    # comm_runtime.reset_signals()
    # nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    # comm_runtime.start_after(torch.cuda.current_stream())
    # out_triton = attention_fn(q_local, k_local, v_local, 1.0, output_buffer=src_buf, **wave_kwargs)
    # comm_runtime.execute()
    # comm_runtime.end_before(torch.cuda.current_stream())
    comm_stream = torch.cuda.Stream()

    def func():
        signal_set_event = torch.cuda.Event()
        comm_runtime.reset_signals()
        wave_kwargs["producer_counter_ptr"].zero_()

        signal_set_event.record(torch.cuda.current_stream())

        compute_stream = comm_runtime.compute_stream
        

        compute_stream.wait_event(signal_set_event)

        comm_runtime.start_after(torch.cuda.current_stream())

        with torch.cuda.stream(compute_stream):
            attention_fn(q_local, k_local, v_local, 1.0, output_buffer=src_buf, **wave_kwargs)

        comm_runtime.execute()
        comm_runtime.end_before(torch.cuda.current_stream())

        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

        return src_buf, dst_buf


    out_local, out_dist = func()


    torch.testing.assert_close(out_local, out_flash, atol=1e-2, rtol=1e-2)
    dist_print(f"Rank {rank} attention results correct for bhsd layout", allowed_ranks="all", need_sync=True)

    torch.testing.assert_close(out_dist, golden_out, atol=1e-2, rtol=1e-2)
    dist_print(f"Rank {rank} attn_all_to_all results correct for bhsd layout", allowed_ranks="all", need_sync=True)

    with group_profile("all2all_attn_all2all", False, group=TP_GROUP):
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
