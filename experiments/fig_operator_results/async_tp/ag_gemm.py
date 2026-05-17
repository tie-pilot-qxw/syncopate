import argparse
import os
import re
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from torch.distributed._symmetric_memory import (
    _fused_all_gather_matmul_fallback,
)


CONFIGS: Dict[str, Dict[str, int]] = {
    "LLaMA-7B": {"M": 8192, "N": 11008, "K": 4096},
    "LLaMA-3.1-8B": {"M": 8192, "N": 14336, "K": 4096},
    "LLaMA-3.1-70B": {"M": 8192, "N": 28672, "K": 8192},
    "LLaMA-3.1-405B": {"M": 8192, "N": 53248, "K": 16384},
    "Mistral-7B": {"M": 8192, "N": 14336, "K": 4096},
    "Qwen2-72B": {"M": 8192, "N": 29568, "K": 8192},
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark torch.ops.symm_mem.fused_all_gather_matmul."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="LLaMA-7B",
        help=(
            "Shape preset or MxNxK tuple for the global problem. "
            "Global A is MxK, global B is NxK; each rank owns (M/world)xK and (N/world)xK."
        ),
    )
    parser.add_argument("--mm-count", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare-fallback", action="store_true")
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
    except ValueError as exc:  # pragma: no cover
        raise ValueError(f"Non-integer value in config {config}") from exc


def _make_inputs(
    global_dims: Tuple[int, int, int],
    world_size: int,
    rank: int,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    M_global, N_global, K = global_dims
    if M_global % world_size != 0:
        raise ValueError(
            f"M ({M_global}) must be divisible by world_size ({world_size})"
        )
    if N_global % world_size != 0:
        raise ValueError(
            f"N ({N_global}) must be divisible by world_size ({world_size})"
        )

    M_local = M_global // world_size
    N_local = N_global // world_size

    dtype = _resolve_dtype(args.dtype)
    torch.manual_seed(args.seed + rank)

    A_shard = torch.randn(M_local, K, device=device, dtype=dtype)

    Bs: List[torch.Tensor] = []
    for _ in range(args.mm_count):
        # Generate N_local x K to mimic transposed B inputs (N x K).
        B_local = torch.randn(N_local, K, device=device, dtype=dtype)
        Bs.append(B_local.t().contiguous())
    return A_shard, Bs


def _benchmark(
    A_shard: torch.Tensor,
    Bs: List[torch.Tensor],
    group_name: str,
    warmup: int,
    iters: int,
) -> Tuple[float, float]:
    dist.barrier()
    for _ in range(warmup):
        torch.ops.symm_mem.fused_all_gather_matmul(
            A_shard, Bs, gather_dim=0, group_name=group_name
        )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        torch.ops.symm_mem.fused_all_gather_matmul(
            A_shard, Bs, gather_dim=0, group_name=group_name
        )
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    elapsed_s = elapsed_ms / 1e3
    return elapsed_s, elapsed_s / max(iters, 1)


def _compute_tflops(
    global_dims: Tuple[int, int, int],
    world_size: int,
    mm_count: int,
) -> float:
    M_global, N_global, K = global_dims
    N_local = N_global // world_size
    flops_per_rank = 2 * M_global * K * N_local * mm_count
    return flops_per_rank / 1.0e12


def _validate(
    A_shard: torch.Tensor,
    Bs: List[torch.Tensor],
    group_name: str,
) -> None:
    with torch.cuda.amp.autocast(enabled=False):
        ref_ag, ref_mm = _fused_all_gather_matmul_fallback(
            A_shard, Bs, gather_dim=0, group_name=group_name
        )
        agg, outs = torch.ops.symm_mem.fused_all_gather_matmul(
            A_shard, Bs, gather_dim=0, group_name=group_name
        )
        torch.testing.assert_close(agg, ref_ag)
        for lhs, rhs in zip(outs, ref_mm):
            torch.testing.assert_close(lhs, rhs)


def main() -> None:
    args = _parse_args()
    global_dims = _parse_config(args.config)
    backend = "nccl"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    dist.init_process_group(backend=backend)
    world_size = dist.get_world_size()

    group_name = dist.group.WORLD.group_name

    A_shard, Bs = _make_inputs(
        global_dims, world_size, dist.get_rank(), device, args
    )

    if args.compare_fallback:
        _validate(A_shard, Bs, group_name)

    elapsed, per_iter = _benchmark(
        A_shard,
        Bs,
        group_name=group_name,
        warmup=args.warmup,
        iters=args.iters,
    )

    avg_tensor = torch.tensor(per_iter, device=device)
    dist.all_reduce(avg_tensor)
    avg_per_rank = avg_tensor.item() / world_size

    tflops = _compute_tflops(global_dims, world_size, args.mm_count)
    tflops_per_rank = tflops / per_iter if per_iter > 0 else float("inf")
    tflops_global = (tflops * world_size) / avg_per_rank if avg_per_rank > 0 else float(
        "inf"
    )

    if dist.get_rank() == 0:
        print(
            f"fused_all_gather_matmul | config={args.config} (global M,N,K={global_dims}) | "
            f"mm_count={args.mm_count} | dtype={args.dtype}"
        )
        print(
            f"Per-rank: {per_iter * 1e3:.3f} ms [{tflops_per_rank:.2f} TFLOP/s] | "
            f"Across ranks (avg): {avg_per_rank * 1e3:.3f} ms [{tflops_global:.2f} TFLOP/s]"
        )
        print(
            f"Local shapes: A=({global_dims[0] // world_size}, {global_dims[2]}) "
            f"B=({global_dims[1] // world_size}, {global_dims[2]})"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
