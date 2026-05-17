import argparse
import os
import re
from typing import Dict, Tuple, Optional

import torch
import torch.distributed as dist
from pathlib import Path


class group_profile:

    def __init__(
        self,
        name: str = None,
        do_prof: bool = True,
        merge_group: bool = False,
        keep_merged_only: bool = True,
        compress: bool = False,
        group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        self.name = name
        self.do_prof = do_prof
        self.profile = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
        )
        self.group = group or torch.distributed.group.WORLD
        self.merge_group = merge_group
        self.keep_merged_only = keep_merged_only
        self.compress = compress
        self.trace_file = (Path("prof") / f"{self.name}" / f"rank{self.group.rank()}.json")

    def __enter__(self):
        if self.do_prof:
            self.profile.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.do_prof:
            self.profile.__exit__(exc_type, exc_val, exc_tb)
            # export chrome trace
            self.trace_file.parent.mkdir(parents=True, exist_ok=True)
            print(f"export chrome trace to {self.trace_file}")
            self.profile.export_chrome_trace(str(self.trace_file))
            if self.merge_group:
                self.merge_all()

    def _collect_all_to_rank0(self):
        # merge all
        if self.merge_group:
            torch.cuda.synchronize()  # wait for all ranks export
            with open(self.trace_file, "rb") as f:
                trace_content = f.read()
            trace_content_list = [None for _ in range(self.group.size())]
            torch.distributed.gather_object(
                trace_content,
                trace_content_list if self.group.rank() == 0 else None,
                dst=0,
                group=self.group,
            )
            torch.cuda.synchronize()  # wait for all ranks export
            return trace_content_list if self.group.rank() == 0 else None

    def _merge_all_trace(self, trace_content_list):
        print("merge profiles...")
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir).mkdir(exist_ok=True)

            for n in range(self.group.size()):
                with open(Path(tmpdir) / f"trace_{n}.json", "wb") as f:
                    f.write(trace_content_list[n])

            # merge all json
            to_merge_files = [Path(tmpdir) / f"trace_{n}.json" for n in range(self.group.size())]
            merged_json = Path("prof") / f"{self.name}_merged.json"
            _merge_json(to_merge_files, merged_json, self.compress)

    def merge_all(self):
        trace_content_list = self._collect_all_to_rank0()
        if self.group.rank() == 0:
            self._merge_all_trace(trace_content_list)
        self.group.barrier()
        torch.cuda.synchronize()
        outdir = Path("prof") / f"{self.name}"
        if self.keep_merged_only:
            print(f"remove profile directory: {outdir}")
            self.trace_file.unlink(missing_ok=True)
            if torch.cuda.current_device() == 0:  # run once for a device
                shutil.rmtree(self.trace_file.parent, ignore_errors=True)

# Try to import the fallback op; fall back to the manual implementation during validate if it is missing
try:
    from torch.distributed._symmetric_memory import (
        _fused_matmul_reduce_scatter_fallback,
    )
    HAS_FALLBACK_OP = True
except ImportError:
    HAS_FALLBACK_OP = False

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
        description="Benchmark torch.ops.symm_mem.fused_matmul_reduce_scatter."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="LLaMA-7B",
        help=(
            "Shape preset or MxNxK tuple. "
            "Simulates Row Parallel Linear (TP): "
            "Global A is MxK, Global B is NxK. "
            "Each rank owns Mx(K/world) and Nx(K/world). "
            "Output is scattered on M, resulting in (M/world)xN."
        ),
    )
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
    if M_global % world_size != 0:
        raise ValueError(
            f"M ({M_global}) must be divisible by world_size ({world_size})."
        )

    K_local = K_global // world_size
    dtype = _resolve_dtype(args.dtype)
    torch.manual_seed(args.seed + rank)

    # A_shard: [M, K_local]
    A_shard = torch.randn(M_global, K_local, device=device, dtype=dtype)

    # B_local: [N, K_local] -> Transpose to [K_local, N] for MatMul (A @ B)
    # We create it as (N, K_local) to mimic weight shape, then transpose.
    B_local = torch.randn(N_global, K_local, device=device, dtype=dtype)
    B_input = B_local.t().contiguous()
        
    return A_shard, B_input


def _benchmark(
    A_shard: torch.Tensor,
    B: torch.Tensor,
    group_name: str,
    warmup: int,
    iters: int,
) -> Tuple[float, float]:
    dist.barrier()
    # scatter_dim=0 implies we reduce global result and scatter along M dimension.
    for _ in range(warmup):
        torch.ops.symm_mem.fused_matmul_reduce_scatter(
            A_shard, B, reduce_op="sum", scatter_dim=0, group_name=group_name
        )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        torch.ops.symm_mem.fused_matmul_reduce_scatter(
            A_shard, B, reduce_op="sum", scatter_dim=0, group_name=group_name
        )
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    elapsed_s = elapsed_ms / 1e3
    return elapsed_s, elapsed_s / max(iters, 1)


def _compute_tflops(
    global_dims: Tuple[int, int, int],
    world_size: int,
) -> float:
    M_global, N_global, K_global = global_dims
    K_local = K_global // world_size
    # Per rank FLOPs = 2 * M * N * K_local
    flops_per_rank = 2 * M_global * N_global * K_local
    return flops_per_rank / 1.0e12


def _validate(
    A_shard: torch.Tensor,
    B: torch.Tensor,
    group_name: str,
) -> None:
    # 1. Run Fused Op
    results_fused = torch.ops.symm_mem.fused_matmul_reduce_scatter(
        A_shard, B, reduce_op="sum", scatter_dim=0, group_name=group_name
    )
    
    # 2. Run Reference
    if HAS_FALLBACK_OP:
        results_ref = _fused_matmul_reduce_scatter_fallback(
            A_shard, B, reduce_op="sum", scatter_dim=0, group_name=group_name
        )
    else:
        world_size = dist.get_world_size()
        M = A_shard.size(0)
        M_local_out = M // world_size
        
        # Local Matmul: [M, K_local] @ [K_local, N] -> [M, N]
        matmul_out = torch.matmul(A_shard, B)
        
        # Reduce Scatter manually
        output_shard = torch.empty(
            M_local_out, matmul_out.size(1), 
            device=A_shard.device, dtype=A_shard.dtype
        )
        dist.reduce_scatter_tensor(output_shard, matmul_out, op=dist.ReduceOp.SUM)
        results_ref = output_shard

    # 3. Compare
    torch.testing.assert_close(results_fused, results_ref, atol=5e-1, rtol=5e-1)


def main() -> None:
    args = _parse_args()
    global_dims = _parse_config(args.config)
    backend = "nccl"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    dist.init_process_group(backend=backend)
    world_size = dist.get_world_size()

    try:
        group_name = dist.group.WORLD.group_name
    except AttributeError:
        group_name = "default_pg"

    A_shard, B = _make_inputs(
        global_dims, world_size, dist.get_rank(), device, args
    )

    if args.compare_fallback:
        if dist.get_rank() == 0:
            print("Running validation...")
        _validate(A_shard, B, group_name)
        if dist.get_rank() == 0:
            print("Validation passed.")


    with group_profile(name="async_tp_gemm_rs", group=dist.group.WORLD):
        elapsed, per_iter = _benchmark(
            A_shard,
            B,
            group_name=group_name,
            warmup=args.warmup,
            iters=args.iters,
        )

    avg_tensor = torch.tensor(per_iter, device=device)
    dist.all_reduce(avg_tensor)
    avg_per_rank = avg_tensor.item() / world_size

    tflops = _compute_tflops(global_dims, world_size)
    tflops_per_rank = tflops / per_iter if per_iter > 0 else float("inf")
    tflops_global = (tflops * world_size) / avg_per_rank if avg_per_rank > 0 else float(
        "inf"
    )

    if dist.get_rank() == 0:
        M, N, K = global_dims
        print(
            f"fused_matmul_reduce_scatter | config={args.config} (global M,N,K={global_dims}) | "
            f"dtype={args.dtype}"
        )
        print(
            f"Per-rank: {per_iter * 1e3:.3f} ms [{tflops_per_rank:.2f} TFLOP/s] | "
            f"Across ranks (avg): {avg_per_rank * 1e3:.3f} ms [{tflops_global:.2f} TFLOP/s]"
        )
        print(
            f"Shapes (Row Parallel): "
            f"Input A=({M}, {K // world_size}), "
            f"Input B=({K // world_size}, {N}), "
            f"Output=({M // world_size}, {N})"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
