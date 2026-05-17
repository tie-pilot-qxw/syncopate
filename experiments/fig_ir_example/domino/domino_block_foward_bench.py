#!/usr/bin/env python3
# Lightweight benchmark for a single DominoTransformerLayer forward pass.

import argparse
import os
import time
from dataclasses import dataclass
from typing import Callable, Tuple

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
import torch.distributed as torch_dist
from torch.profiler import ProfilerActivity, profile, record_function

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.domino.transformer import (
    DominoTransformerLayer,
    DominoUtil,
    LayerType,
    AttnMaskType,
)


@dataclass
class DominoLayerBenchConfig:
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    kv_channels: int
    attention_dropout: float
    hidden_dropout: float
    layernorm_epsilon: float
    add_bias_linear: bool
    gated_linear_unit: bool
    params_dtype: torch.dtype
    init_method: Callable[[torch.Tensor], torch.Tensor]
    output_layer_init_method: Callable[[torch.Tensor], torch.Tensor]
    apply_residual_connection_post_layernorm: bool = True
    domino_intra_layer_overlap: bool = False
    num_layers: int = 1
    perform_initialization: bool = True


class SimpleMPU:

    def __init__(self, tp_size: int):
        world_size = dist.get_world_size()
        assert tp_size == world_size, "This helper expects tensor parallel size to match world size."
        self._tp_size = tp_size
        self._tp_group = torch_dist.group.WORLD

    def get_tensor_model_parallel_world_size(self) -> int:
        return self._tp_size

    def get_tensor_model_parallel_group(self):
        return self._tp_group


def parse_dtype(name: str) -> torch.dtype:
    lowered = name.lower()
    if lowered in ("fp32", "float32"):
        return torch.float32
    if lowered in ("fp16", "float16", "half"):
        return torch.float16
    if lowered in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def ensure_distributed_initialized(dist_backend: str):
    defaults = {
        "RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_RANK": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    if not torch_dist.is_initialized():
        dist.init_distributed(dist_backend=dist_backend,
                              init_method="env://",
                              rank=int(os.environ["RANK"]),
                              world_size=int(os.environ["WORLD_SIZE"]))


def identity_rotary(x: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
    return x


def build_layer(args, dtype: torch.dtype) -> DominoTransformerLayer:
    init_std = args.init_std

    def _init_fn(weight: torch.Tensor):
        return torch.nn.init.normal_(weight, mean=0.0, std=init_std)

    config = DominoLayerBenchConfig(hidden_size=args.hidden_size,
                                    ffn_hidden_size=args.ffn_hidden_size,
                                    num_attention_heads=args.num_heads,
                                    kv_channels=args.kv_channels,
                                    attention_dropout=args.attention_dropout,
                                    hidden_dropout=args.hidden_dropout,
                                    layernorm_epsilon=1e-5,
                                    add_bias_linear=True,
                                    gated_linear_unit=False,
                                    params_dtype=dtype,
                                    init_method=_init_fn,
                                    output_layer_init_method=_init_fn)

    mpu = SimpleMPU(tp_size=dist.get_world_size())

    layer = DominoTransformerLayer(config=config,
                                   mpu=mpu,
                                   apply_rotary_pos_emb=identity_rotary,
                                   layer_number=1,
                                   layer_type=LayerType.encoder,
                                   self_attn_mask_type=AttnMaskType.causal,
                                   drop_path_rate=0.0)
    return layer.to(dtype=dtype, device=torch.device(get_accelerator().device_name()))


def build_inputs(args, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(get_accelerator().device_name())
    micro_batch = args.batch_size // 2
    shape = (args.seq_len, micro_batch, args.hidden_size)
    hidden_a = torch.randn(shape, device=device, dtype=dtype)
    hidden_b = torch.randn(shape, device=device, dtype=dtype)
    return hidden_a, hidden_b


def benchmark(layer: DominoTransformerLayer,
              inputs: Tuple[torch.Tensor, torch.Tensor],
              warmup_iters: int,
              benchmark_iters: int) -> float:
    acc = get_accelerator()
    layer.eval()
    torch_dist.barrier()
    with torch.no_grad():
        for _ in range(warmup_iters):
            layer(inputs, attention_mask=None)
        acc.synchronize()
        start = time.perf_counter()
        for _ in range(benchmark_iters):
            layer(inputs, attention_mask=None)
        acc.synchronize()
        elapsed = time.perf_counter() - start
    return elapsed / benchmark_iters


def capture_timeline(layer: DominoTransformerLayer,
                     inputs: Tuple[torch.Tensor, torch.Tensor],
                     warmup_iters: int,
                     record_iters: int,
                     output_path: str,
                     rank: int):
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    acc = get_accelerator()
    layer.eval()
    torch_dist.barrier()

    with profile(activities=activities, record_shapes=True, with_stack=True) as prof:
        with torch.no_grad():
            for _ in range(warmup_iters):
                layer(inputs, attention_mask=None)
            acc.synchronize()
            for _ in range(record_iters):
                with record_function("domino_forward"):
                    layer(inputs, attention_mask=None)
            acc.synchronize()

    if rank == 0:
        prof.export_chrome_trace(output_path)
        print(f"Timeline trace written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Domino transformer block forward benchmark.")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length per micro batch.")
    parser.add_argument("--batch-size", type=int, default=4, help="Total batch size (must be divisible by 2).")
    parser.add_argument("--hidden-size", type=int, default=8192, help="Transformer hidden size.")
    parser.add_argument("--ffn-hidden-size", type=int, default=28672, help="FFN hidden size (defaults to 4x hidden).")
    parser.add_argument("--num-heads", type=int, default=32, help="Number of attention heads.")
    parser.add_argument("--kv-channels",
                        type=int,
                        default=None,
                        help="Size per key/value head (defaults to hidden/num_heads).")
    parser.add_argument("--dtype", type=str, default="bf16", help="Computation dtype (fp32, fp16, bf16).")
    parser.add_argument("--warmup-iters", type=int, default=10, help="Number of warmup steps.")
    parser.add_argument("--iters", type=int, default=50, help="Number of timed forward passes.")
    parser.add_argument("--init-std", type=float, default=0.02, help="Stddev for weight initialization.")
    parser.add_argument("--attention-dropout", type=float, default=0.0, help="Attention dropout.")
    parser.add_argument("--hidden-dropout", type=float, default=0.0, help="Activation dropout.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--timeline-output",
                        type=str,
                        default=None,
                        help="Path to save torch.profiler Chrome trace (disabled if omitted).")
    parser.add_argument("--timeline-warmup", type=int, default=5, help="Warmup iterations before profiling.")
    parser.add_argument("--timeline-iters", type=int, default=10, help="Number of profiled iterations.")
    args = parser.parse_args()

    if args.batch_size % 2 != 0:
        raise ValueError("batch-size must be divisible by 2 to create two Domino micro-batches.")

    dtype = parse_dtype(args.dtype)
    if args.ffn_hidden_size is None:
        args.ffn_hidden_size = args.hidden_size * 4
    if args.kv_channels is None:
        args.kv_channels = args.hidden_size // args.num_heads

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    accelerator = get_accelerator()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if hasattr(accelerator, "device_count") and accelerator.device_count() > 0:
        accelerator.set_device(local_rank)

    ensure_distributed_initialized(accelerator.communication_backend_name())
    rank = dist.get_rank() if dist.is_initialized() else 0

    DominoUtil.HANDLE_DIC = {"BATCH0": None, "BATCH1": None}

    layer = build_layer(args, dtype)
    inputs = build_inputs(args, dtype)

    avg_latency = benchmark(layer, inputs, args.warmup_iters, args.iters)
    tokens = args.seq_len * args.batch_size
    samples = args.batch_size
    if rank == 0:
        print("DominoTransformerLayer forward benchmark")
        print(f"seq_len={args.seq_len}, batch_size={args.batch_size}, hidden={args.hidden_size}, "
              f"ffn_hidden={args.ffn_hidden_size}, heads={args.num_heads}, dtype={dtype}")
        print(f"Average latency: {avg_latency * 1e3:.2f} ms")
        print(f"Samples/sec: {samples / avg_latency:.2f}")
        print(f"Tokens/sec: {tokens / avg_latency:.2f}")

    if args.timeline_output:
        capture_timeline(layer,
                         inputs,
                         args.timeline_warmup,
                         args.timeline_iters,
                         args.timeline_output,
                         rank)


if __name__ == "__main__":
    main()
