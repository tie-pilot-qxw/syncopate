from flash_attn_interface import flash_attn_func
import os
import torch
import torch.distributed as dist

def head_to_seq(x, B, world_size, S, H, D, group=None):
    """
    Convert (B, S, H/P, D) back to (B, S/P, H, D).
    Args:
        x: Input tensor, shape (B, S_global, H_local, D)
        S: Global sequence length
        H: Global number of heads
    """
    H_local = H // world_size
    S_local = S // world_size
    
    # x shape: (B, S_global, H_local, D)
    
    # 1. Split sequence dimension to expose the source rank (split seq across ranks)
    # (B, P * S_local, H_local, D) -> (B, P, S_local, H_local, D)
    x = x.view(B, world_size, S_local, H_local, D)
    
    # 2. Permute: move P to dim 0 to prepare for send
    # (B, P, S_local, H_local, D) -> (P, B, S_local, H_local, D)
    x = x.permute(1, 0, 2, 3, 4).contiguous()
    
    # 3. All-to-all communication
    # input: (P, B, S_local, H_local, D) - dim0 P is the target rank
    # output: (P, B, S_local, H_local, D) - dim0 P is the source rank
    output = torch.empty_like(x)
    dist.all_to_all_single(output, x, group=group)
    
    # 4. Permute back: move source rank (P) next to the head dimension
    # (P, B, S_local, H_local, D) -> (B, S_local, P, H_local, D)
    output = output.permute(1, 2, 0, 3, 4).contiguous()
    
    # 5. Merge the head dimension (combine partial heads from different ranks)
    # (B, S_local, P, H_local, D) -> (B, S_local, P * H_local, D) -> (B, S_local, H, D)
    output = output.view(B, S_local, H, D)
    
    return output

def attention_then_all2all(q_head, k_head, v_head, B, world_size, S_global, H_global, D, group=None):
    """
    Flow:
    1. Input is already head-parallel: (B, S_global, H/P, D)
    2. Compute attention
    3. All-to-all to convert to sequence-parallel: (B, S_global/P, H, D)
    """
    
    # 1. Attention
    # Input: (B, S, H_local, D)
    # FlashAttn expects (Batch, Seq, Head, Dim) without an extra transpose
    attn_out_head = flash_attn_func(q_head, k_head, v_head, causal=False)
    
    # 2. Transform: head-parallel -> seq-parallel
    # Input: (B, S_global, H_local, D)
    # Output: (B, S_local, H_global, D)
    output_seq = head_to_seq(attn_out_head, B, world_size, S_global, H_global, D, group)
    
    return output_seq

if __name__ == "__main__":
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    n_warmup = 5
    n_iters = 10

    B = 2
    H = 32 # Global Heads
    SEQ = 20 * 1024 * 1 # Global Sequence Length
    DIM = 128
    dtype = torch.float16

    assert H % WORLD_SIZE == 0, "Heads must be divisible by World Size"
    H_local = H // WORLD_SIZE
    S_local = SEQ // WORLD_SIZE

    # -----------------------------------------------------------------
    # Initialize data in head-parallel mode
    # Each rank owns the full sequence but only H_local heads
    # Shape: [B, SEQ, H_local, DIM]
    # -----------------------------------------------------------------
    q_head = torch.randn((B, SEQ, H_local, DIM), device="cuda", dtype=dtype)
    k_head = torch.randn((B, SEQ, H_local, DIM), device="cuda", dtype=dtype)
    v_head = torch.randn((B, SEQ, H_local, DIM), device="cuda", dtype=dtype)

    # =================================================================
    # 1. Correctness check
    # =================================================================
    
    # Run function under test
    output_seq_parallel = attention_then_all2all(q_head, k_head, v_head, B, WORLD_SIZE, SEQ, H, DIM)

    # Build reference on a single GPU by gathering all heads into (B, SEQ, H, DIM)
    
    # 1. Move H_local to dim 0 for easier gather: (H_local, B, SEQ, D)
    q_head_t = q_head.permute(2, 0, 1, 3).contiguous()
    k_head_t = k_head.permute(2, 0, 1, 3).contiguous()
    v_head_t = v_head.permute(2, 0, 1, 3).contiguous()
    
    q_global_t = torch.empty((H, B, SEQ, DIM), device="cuda", dtype=dtype)
    k_global_t = torch.empty((H, B, SEQ, DIM), device="cuda", dtype=dtype)
    v_global_t = torch.empty((H, B, SEQ, DIM), device="cuda", dtype=dtype)
    
    # Gather all heads
    dist.all_gather_into_tensor(q_global_t, q_head_t)
    dist.all_gather_into_tensor(k_global_t, k_head_t)
    dist.all_gather_into_tensor(v_global_t, v_head_t)
    
    # Restore shape: (B, SEQ, H, D)
    q_global = q_global_t.permute(1, 2, 0, 3).contiguous()
    k_global = k_global_t.permute(1, 2, 0, 3).contiguous()
    v_global = v_global_t.permute(1, 2, 0, 3).contiguous()
    
    # Run global attention
    attn_out_global = flash_attn_func(q_global, k_global, v_global, causal=False)
    
    # Validate: output_seq_parallel is sequence-parallel with shape (B, S_local, H, D)
    # It is a slice of the global SEQ dimension, so slice the reference by rank
    start_idx = rank * S_local
    end_idx = (rank + 1) * S_local
    
    attn_out_ref_slice = attn_out_global[:, start_idx:end_idx, :, :]
    
    if torch.allclose(output_seq_parallel, attn_out_ref_slice, atol=1e-2, rtol=1e-2):
        print(f"Rank {rank}: Correctness check PASSED.")
    else:
        diff = torch.abs(output_seq_parallel - attn_out_ref_slice).max()
        print(f"Rank {rank}: Correctness check FAILED. Max diff: {diff}")

    torch.cuda.synchronize()
    dist.barrier()

    # =================================================================
    # 2. Benchmark
    # =================================================================
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(n_warmup):
        _ = attention_then_all2all(q_head, k_head, v_head, B, WORLD_SIZE, SEQ, H, DIM)

    # Run
    start.record()
    for _ in range(n_iters):
        _ = attention_then_all2all(q_head, k_head, v_head, B, WORLD_SIZE, SEQ, H, DIM)
    end.record()
    
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end) / n_iters  # milliseconds

    # FLOPS Calculation (Standard Attn FLOPS)
    flops_per_matmul = 2.0 * B * H * SEQ * SEQ * DIM
    total_flops = 2 * flops_per_matmul
    
    # FLOPS per GPU (total computation / world size)
    per_gpu_flops = total_flops / WORLD_SIZE
    tflops = per_gpu_flops / 1e12 / (elapsed / 1e3)

    # Avoid jumbled prints across ranks; print in rank order
    for rank_id in range(WORLD_SIZE):
        if rank == rank_id:
            print(f"Rank {rank} | Time: {elapsed:.2f} ms | Performance: {tflops:.2f} TFLOPs")
        dist.barrier()

    dist.destroy_process_group()
