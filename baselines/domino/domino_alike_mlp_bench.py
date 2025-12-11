#!/usr/bin/env python3
import argparse
import os
import time
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, record_function

from baselines.domino.domino_alike_mlp import DominoAlikeMLP, DominoAlikeMLPConfig


def parse_dtype(name: str) -> torch.dtype:
    lowered = name.lower()
    if lowered in ("fp32", "float32"):
        return torch.float32
    if lowered in ("fp16", "float16", "half"):
        return torch.float16
    if lowered in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def ensure_distributed_initialized(backend: str):
    defaults = {
        "RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_RANK": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend,
                                init_method="env://",
                                rank=int(os.environ["RANK"]),
                                world_size=int(os.environ["WORLD_SIZE"]))


def build_inputs(args, dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    micro_batch = args.batch_size // 2
    shape = (args.seq_len, micro_batch, args.hidden_size)
    hidden_a = torch.randn(shape, device=device, dtype=dtype)
    hidden_b = torch.randn(shape, device=device, dtype=dtype)
    return hidden_a, hidden_b


def build_mlp(args, dtype: torch.dtype, device: torch.device, group: Optional[dist.ProcessGroup]) -> DominoAlikeMLP:
    def _init_fn(weight: torch.Tensor):
        return torch.nn.init.normal_(weight, mean=0.0, std=args.init_std)

    config = DominoAlikeMLPConfig(hidden_size=args.hidden_size,
                                  ffn_hidden_size=args.ffn_hidden_size,
                                  params_dtype=dtype,
                                  init_method=_init_fn,
                                  output_layer_init_method=_init_fn,
                                  add_bias_linear=args.add_bias,
                                  perform_initialization=True)
    mlp = DominoAlikeMLP(config=config, tp_group=group)
    return mlp.to(device=device, dtype=dtype)


def benchmark(
    inputs: Tuple[torch.Tensor, torch.Tensor],
    mlp: DominoAlikeMLP,
    warmup_iters: int,
    benchmark_iters: int,
) -> float:
    hidden_a, hidden_b = inputs
    use_cuda = hidden_a.is_cuda
    if dist.is_initialized():
        dist.barrier()
    with torch.no_grad():
        for _ in range(warmup_iters):
            mlp((hidden_a, hidden_b))
        if use_cuda:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(benchmark_iters):
                mlp((hidden_a, hidden_b))
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end) / 1e3
        else:
            start_t = time.perf_counter()
            for _ in range(benchmark_iters):
                mlp((hidden_a, hidden_b))
            elapsed = time.perf_counter() - start_t
    return elapsed / max(benchmark_iters, 1)


def capture_timeline(
    inputs: Tuple[torch.Tensor, torch.Tensor],
    mlp: DominoAlikeMLP,
    warmup_iters: int,
    record_iters: int,
    output_path: str,
    rank: int,
):
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    hidden_a, hidden_b = inputs

    if dist.is_initialized():
        dist.barrier()

    with profile(activities=activities, record_shapes=True, with_stack=True) as prof:
        with torch.no_grad():
            for _ in range(warmup_iters):
                mlp((hidden_a, hidden_b))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            for _ in range(record_iters):
                with record_function("domino_alike_mlp"):
                    mlp((hidden_a, hidden_b))
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    if rank == 0:
        prof.export_chrome_trace(output_path)
        print(f"Timeline trace written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Domino-alike MLP benchmark.")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length per micro batch.")
    parser.add_argument("--batch-size", type=int, default=4, help="Total batch size (must be divisible by 2).")
    parser.add_argument("--hidden-size", type=int, default=8192, help="Hidden size.")
    parser.add_argument("--ffn-hidden-size", type=int, default=28672, help="FFN hidden size (defaults to 4x hidden).")
    parser.add_argument("--dtype", type=str, default="bf16", help="Computation dtype (fp32, fp16, bf16).")
    parser.add_argument("--warmup-iters", type=int, default=5, help="Number of warmup steps.")
    parser.add_argument("--iters", type=int, default=10, help="Number of timed forward passes.")
    parser.add_argument("--init-std", type=float, default=0.02, help="Stddev for weight initialization.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--add-bias", action="store_true", help="Include biases in the MLP projections.")
    parser.add_argument("--timeline-output",
                        type=str,
                        default=None,
                        help="Path to save torch.profiler Chrome trace (disabled if omitted).")
    parser.add_argument("--timeline-warmup", type=int, default=5, help="Warmup iterations before profiling.")
    parser.add_argument("--timeline-iters", type=int, default=10, help="Number of profiled iterations.")
    args = parser.parse_args()

    if args.batch_size % 2 != 0:
        raise ValueError("batch-size must be divisible by 2 to create two micro-batches.")

    dtype = parse_dtype(args.dtype)
    if args.ffn_hidden_size is None:
        args.ffn_hidden_size = args.hidden_size * 4

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    ensure_distributed_initialized(backend)
    group = dist.group.WORLD if dist.is_initialized() else None
    rank = dist.get_rank() if dist.is_initialized() else 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    hidden_a, hidden_b = build_inputs(args, dtype, device)
    mlp = build_mlp(args, dtype, device, group)

    if args.timeline_output:
        capture_timeline((hidden_a, hidden_b), mlp, args.timeline_warmup, args.timeline_iters, args.timeline_output,
                         rank)

    avg_latency = benchmark((hidden_a, hidden_b), mlp, args.warmup_iters, args.iters)
    tokens = args.seq_len * args.batch_size
    if rank == 0:
        print("Domino-alike MLP benchmark")
        print(f"seq_len={args.seq_len}, batch_size={args.batch_size}, hidden={args.hidden_size}, "
              f"ffn_hidden={args.ffn_hidden_size}, dtype={dtype}")
        print(f"Average latency: {avg_latency * 1e3:.2f} ms")
        print(f"Throughput: {tokens / avg_latency / 1e6:.2f} M tokens/s")


if __name__ == "__main__":
    main()
