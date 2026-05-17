import argparse
import os
import re
from typing import Dict, Tuple

import torch
import torch.distributed as dist

CONFIGS: Dict[str, Dict[str, int]] = {
    "LLaMA-7B": {"M": 8192, "K": 11008, "N": 4096},
    "LLaMA-3.1-8B": {"M": 8192, "K": 14336, "N": 4096},
    "LLaMA-3.1-70B": {"M": 8192, "K": 28672, "N": 8192},
    "LLaMA-3.1-405B": {"M": 8192, "K": 53248, "N": 16384},
    "Mistral-7B": {"M": 8192, "K": 14336, "N": 4096},
    "Qwen2-72B": {"M": 8192, "K": 29568, "N": 8192},
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark cuBLAS (GEMM) + NCCL (all_reduce) baseline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="LLaMA-7B",
        help=(
            "Shape preset or MxNxK tuple. "
            "Global A is MxK, Global B is NxK. "
            "Each rank owns Mx(K/world) and Nx(K/world). "
            "We compute C_partial = A @ B (MxN), then all_reduce to get full C."
        ),
    )
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _resolve_dtype(name: str) -> torch.dtype:
    aliases = {
        "float": torch.float32,
        "float32": torch.float32,
        "f32": torch.float32,
        "half": torch.float16,
        "float16": torch.float16,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if name not in aliases:
        raise ValueError(f"Unsupported dtype: {name}")
    return aliases[name]


def _parse_config(config: str) -> Tuple[int, int, int]:
    if config in CONFIGS:
        cfg = CONFIGS[config]
        return cfg["M"], cfg["N"], cfg["K"]
    tokens = [t for t in re.split(r"[x,_]", config) if t]
    if len(tokens) != 3:
        raise ValueError(
            f"Config must be one of {list(CONFIGS)} or formatted as MxNxK, got {config}"
        )
    try:
        return tuple(int(t) for t in tokens)  # type: ignore[return-value]
    except ValueError as exc:
        raise ValueError(f"Non-integer value in config {config}") from exc


def _make_inputs(
    global_dims: Tuple[int, int, int],
    world_size: int,
    rank: int,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor]:
    M_global, N_global, K_global = global_dims

    if K_global % world_size != 0:
        raise ValueError(
            f"K ({K_global}) must be divisible by world_size ({world_size})."
        )

    K_local = K_global // world_size
    dtype = _resolve_dtype(args.dtype)
    torch.manual_seed(args.seed + rank)

    # A_shard: [M, K_local]
    A_shard = torch.randn(M_global, K_local, device=device, dtype=dtype)

    # B_local: [N, K_local] -> transpose to [K_local, N] for matmul
    B_local = torch.randn(N_global, K_local, device=device, dtype=dtype)
    B_input = B_local.t().contiguous()  # [K_local, N]

    return A_shard, B_input


def _compute_tflops(
    global_dims: Tuple[int, int, int],
    world_size: int,
) -> float:
    M_global, N_global, K_global = global_dims
    K_local = K_global // world_size
    flops_per_rank = 2 * M_global * N_global * K_local
    return flops_per_rank / 1.0e12


def _benchmark_nccl_cublas(
    A_shard: torch.Tensor,
    B: torch.Tensor,
    warmup: int,
    iters: int,
) -> Tuple[float, float]:
    """
    Benchmark: C_partial = A_shard @ B (cuBLAS GEMM),
               then all_reduce(C_partial) (NCCL).
    """
    device = A_shard.device

    dist.barrier()

    # Warmup
    for _ in range(warmup):
        C_partial = torch.matmul(A_shard, B)
        dist.all_reduce(C_partial, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    dist.barrier()
    start.record()
    for _ in range(iters):
        C_partial = torch.matmul(A_shard, B)
        dist.all_reduce(C_partial, op=dist.ReduceOp.SUM)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    elapsed_s = elapsed_ms / 1e3
    per_iter = elapsed_s / max(iters, 1)
    return elapsed_s, per_iter


def main() -> None:
    args = _parse_args()
    global_dims = _parse_config(args.config)

    backend = "nccl"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    dist.init_process_group(backend=backend)
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    A_shard, B = _make_inputs(
        global_dims, world_size, rank, device, args
    )

    elapsed, per_iter = _benchmark_nccl_cublas(
        A_shard,
        B,
        warmup=args.warmup,
        iters=args.iters,
    )

    per_rank_time = torch.tensor(per_iter, device=device)
    dist.all_reduce(per_rank_time, op=dist.ReduceOp.MAX)
    max_time = per_rank_time.item()

    tflops_per_rank_theoretical = _compute_tflops(global_dims, world_size)
    tflops_per_rank = (
        tflops_per_rank_theoretical / per_iter if per_iter > 0 else float("inf")
    )
    tflops_global = (
        tflops_per_rank_theoretical * world_size / max_time
        if max_time > 0
        else float("inf")
    )

    if rank == 0:
        M, N, K = global_dims
        print(
            f"[Baseline] GEMM + all_reduce | config={args.config} "
            f"(global M,N,K={global_dims}) | dtype={args.dtype}"
        )
        print(
            f"Per-rank: {per_iter * 1e3:.3f} ms [{tflops_per_rank:.2f} TFLOP/s] | "
            f"Global (max-time): {max_time * 1e3:.3f} ms [{tflops_global:.2f} TFLOP/s]"
        )
        print(
            f"Shapes: "
            f"Input A=({M}, {K // world_size}), "
            f"Input B=({K // world_size}, {N}), "
            f"Output after all-reduce=({M}, {N})"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
