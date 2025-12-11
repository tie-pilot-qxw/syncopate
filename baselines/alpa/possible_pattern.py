"""
possible pattern searched by alpa

Pattern:
gemm + all2all + gemm
x: [seq, hidden] w1: [hidden, ffn_hid / world_size]
-> x2: [seq, ffn_hid / world_size]
all2all
-> x3: [seq / world_size, ffn_hid] w2: [ffn_hid, hidden]
-> out: [seq / world_size, hidden]
"""
import argparse
import os
import re
from typing import Dict, Tuple

import torch
import torch.distributed as dist

# Uses the same shapes as other baselines in this repo.
CONFIGS: Dict[str, Dict[str, int]] = {
    "LLaMA-7B": {"seq": 8192, "hidden": 4096, "ffn": 11008},
    "LLaMA-3.1-8B": {"seq": 8192, "hidden": 4096, "ffn": 14336},
    "LLaMA-3.1-70B": {"seq": 8192, "hidden": 8192, "ffn": 28672},
    "LLaMA-3.1-405B": {"seq": 8192, "hidden": 16384, "ffn": 53248},
    "Qwen2-72B": {"seq": 8192, "hidden": 8192, "ffn": 29568},
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GEMM + all-to-all + GEMM baseline (Alpa-style pattern)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="LLaMA-3.1-70B",
        help=(
            "Shape preset or seqXhiddenXffn tuple. "
            "Input X: [seq, hidden]; W1: [hidden, ffn/world]; "
            "after all-to-all -> [seq/world, ffn]; W2: [ffn, hidden]."
        ),
    )
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Check results against a single-rank reference.",
    )
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
        return cfg["seq"], cfg["hidden"], cfg["ffn"]

    tokens = [t for t in re.split(r"[x,_]", config) if t]
    if len(tokens) != 3:
        raise ValueError(
            f"Config must be one of {list(CONFIGS)} or formatted as seqXhiddenXffn, got {config}"
        )
    try:
        return tuple(int(t) for t in tokens)  # type: ignore[return-value]
    except ValueError as exc:
        raise ValueError(f"Non-integer value in config {config}") from exc


def _make_inputs(
    seq: int,
    hidden: int,
    ffn_hidden: int,
    world_size: int,
    rank: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if ffn_hidden % world_size != 0:
        raise ValueError(
            f"ffn_hidden ({ffn_hidden}) must be divisible by world_size ({world_size})."
        )
    if seq % world_size != 0:
        raise ValueError(
            f"seq ({seq}) must be divisible by world_size ({world_size})."
        )

    ffn_local = ffn_hidden // world_size

    if rank == 0:
        shared_gen = torch.Generator(device=device)
        shared_gen.manual_seed(seed)
        x = torch.randn(seq, hidden, device=device, dtype=dtype, generator=shared_gen)
        w2 = torch.randn(ffn_hidden, hidden, device=device, dtype=dtype, generator=shared_gen)
    else:
        x = torch.empty(seq, hidden, device=device, dtype=dtype)
        w2 = torch.empty(ffn_hidden, hidden, device=device, dtype=dtype)

    dist.broadcast(x, src=0)
    dist.broadcast(w2, src=0)

    shard_gen = torch.Generator(device=device)
    shard_gen.manual_seed(seed + 1 + rank)
    w1 = torch.randn(hidden, ffn_local, device=device, dtype=dtype, generator=shard_gen)
    return x, w1, w2


def _run_pattern(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    seq = x.size(0)
    ffn_local = w1.size(1)
    seq_local = seq // world_size

    # Local GEMM 1: [S, H] @ [H, F_local] -> [S, F_local]
    x2 = torch.matmul(x, w1)

    # All-to-all: reshape to [world, S_local, F_local] and exchange token blocks.
    send_buf = x2.reshape(world_size, seq_local, ffn_local).contiguous()
    recv_buf = torch.empty_like(send_buf)
    dist.all_to_all_single(recv_buf, send_buf)

    # Reassemble to [S_local, F]
    x3 = recv_buf.permute(1, 0, 2).reshape(seq_local, ffn_local * world_size)

    # Local GEMM 2: [S_local, F] @ [F, H] -> [S_local, H]
    out = torch.matmul(x3, w2)
    return out


def _benchmark(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    world_size: int,
    warmup: int,
    iters: int,
) -> Tuple[float, float]:
    dist.barrier()
    with torch.no_grad():
        for _ in range(warmup):
            _run_pattern(x, w1, w2, world_size)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            for _ in range(iters):
                _run_pattern(x, w1, w2, world_size)
        end.record()
        torch.cuda.synchronize()
        elapsed_s = start.elapsed_time(end) / 1e3
    else:
        import time

        start_t = time.perf_counter()
        with torch.no_grad():
            for _ in range(iters):
                _run_pattern(x, w1, w2, world_size)
        elapsed_s = time.perf_counter() - start_t

    per_iter = elapsed_s / max(iters, 1)
    return elapsed_s, per_iter


def _compute_tflops(
    seq: int,
    hidden: int,
    ffn_hidden: int,
    world_size: int,
) -> float:
    ffn_local = ffn_hidden // world_size
    seq_local = seq // world_size

    gemm1 = 2 * seq * hidden * ffn_local
    gemm2 = 2 * seq_local * ffn_hidden * hidden
    flops_per_rank = gemm1 + gemm2
    return flops_per_rank / 1.0e12


def _validate(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    world_size: int,
) -> None:
    """
    Validate by comparing distributed result to a single-rank computation.
    """
    rank = dist.get_rank()
    seq = x.size(0)
    hidden = x.size(1)
    ffn_hidden = w2.size(0)
    seq_local = seq // world_size

    # Distributed path.
    dist_out = _run_pattern(x, w1, w2, world_size)

    # Gather inputs on rank 0 and compute reference.
    with torch.no_grad():
        w1_list = [torch.empty_like(w1) for _ in range(world_size)]
        dist.all_gather(w1_list, w1)

        if rank == 0:
            w1_full = torch.cat(w1_list, dim=1)
            ref_x2 = torch.matmul(x, w1_full)
            ref_chunks = list(ref_x2.chunk(world_size, dim=0))
            ref_out_chunks = [torch.matmul(chunk, w2) for chunk in ref_chunks]
        else:
            ref_out_chunks = []

        ref_local = torch.empty(seq_local, hidden, device=x.device, dtype=x.dtype)
        dist.scatter(ref_local, scatter_list=ref_out_chunks if rank == 0 else [], src=0)

    torch.testing.assert_close(dist_out, ref_local, atol=5e-2, rtol=5e-2)


def main() -> None:
    args = _parse_args()
    seq, hidden, ffn_hidden = _parse_config(args.config)
    dtype = _resolve_dtype(args.dtype)

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    dist.init_process_group(backend=backend)
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    x, w1, w2 = _make_inputs(
        seq,
        hidden,
        ffn_hidden,
        world_size,
        rank,
        device,
        dtype,
        args.seed,
    )

    if args.validate:
        if rank == 0:
            print("Running validation against full-reference computation...")
        _validate(x, w1, w2, world_size)
        if rank == 0:
            print("Validation passed.")

    elapsed, per_iter = _benchmark(
        x,
        w1,
        w2,
        world_size=world_size,
        warmup=args.warmup,
        iters=args.iters,
    )

    per_rank_time = torch.tensor(per_iter, device=device)
    dist.all_reduce(per_rank_time, op=dist.ReduceOp.MAX)
    global_time = per_rank_time.item()

    per_rank_tflops_theoretical = _compute_tflops(seq, hidden, ffn_hidden, world_size)
    per_rank_tflops = per_rank_tflops_theoretical / per_iter if per_iter > 0 else float("inf")
    global_tflops = (
        per_rank_tflops_theoretical * world_size / global_time
        if global_time > 0
        else float("inf")
    )

    if rank == 0:
        print(
            f"[Baseline] gemm -> all_to_all -> gemm | config={args.config} "
            f"(seq={seq}, hidden={hidden}, ffn={ffn_hidden}) | dtype={args.dtype}"
        )
        print(
            f"Per-rank: {per_iter * 1e3:.3f} ms [{per_rank_tflops:.2f} TFLOP/s] | "
            f"Global (max-time): {global_time * 1e3:.3f} ms [{global_tflops:.2f} TFLOP/s]"
        )
        print(
            f"Shapes: X=({seq}, {hidden}); W1=({hidden}, {ffn_hidden // world_size}); "
            f"after all-to-all -> ({seq // world_size}, {ffn_hidden}); "
            f"W2=({ffn_hidden}, {hidden}); output=({seq // world_size}, {hidden})"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
