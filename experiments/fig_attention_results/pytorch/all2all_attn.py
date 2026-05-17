
from flash_attn_interface import flash_attn_func
import os
import torch
import torch.distributed as dist

def head_to_seq(x, B, world_size, S, H, D, group=None):
    """
    Inverse transform: convert (B, S, H/P, D) back to (B, S/P, H, D)
    """

    H_local = H // world_size
    S_local = S // world_size
    # x shape: (B, S_global, H_local, D)
    
    # 1. Split sequence dimension to expose the source rank
    # (B, P * S_local, H_local, D) -> (B, P, S_local, H_local, D)
    x = x.view(B, world_size, S_local, H_local, D)
    
    # 2. Permute: move P to dim 0 to prepare for send
    # (B, P, S_local, H_local, D) -> (P, B, S_local, H_local, D)
    x = x.permute(1, 0, 2, 3, 4).contiguous()
    
    # 3. All-to-all communication
    output = torch.empty_like(x)
    dist.all_to_all_single(output, x, group=group)
    
    # output shape: (P, B, S_local, H_local, D)
    
    # 4. Permute back to restore the head dimension position
    # (P, B, S_local, H_local, D) -> (B, S_local, P, H_local, D)
    output = output.permute(1, 2, 0, 3, 4).contiguous()
    
    # 5. Merge the head dimension
    # (B, S_local, P, H_local, D) -> (B, S_local, P * H_local, D) -> (B, S_local, H, D)
    output = output.view(B, S_local, H, D)
    
    return output


def head_parallel_attention(q_local, k_local, v_local, world_size, group=None):
    """
    Args:
        q_local, k_local, v_local: tensors shaped (B, SEQ_LEN // WORLD_SIZE, H, D)
        num_heads (H): total number of heads
        world_size (P): parallel degree
        group: communication group, defaults to global
    Returns:
        output: attention output shaped (B, SEQ_LEN // WORLD_SIZE, H, D)
    """
    
    # -------------------------------------------------------
    # 1. Prep: get dimension info
    # -------------------------------------------------------
    B, S_local, H, D = q_local.shape
    # Ensure head count is divisible by world_size
    assert H % world_size == 0, "Heads must be divisible by world size"
    H_local = H // world_size
    
    # Helper for transpose + communication
    def seq_to_head(x):
        """
        Convert (B, S/P, H, D) to (B, S, H/P, D) via all-to-all
        """
        # 1. Reshape to expose split dimension
        # (B, S/P, H, D) -> (B, S/P, P, H/P, D)
        x = x.view(B, S_local, world_size, H_local, D)
        
        # 2. Permute to put target rank (P) at dim0 for scatter
        # (B, S/P, P, H/P, D) -> (P, B, S/P, H_local, D)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        
        # 3. All-to-all communication
        # input: (P, B, S_local, H_local, D) -> dim0 P is destination rank
        # output: (P, B, S_local, H_local, D) -> dim0 P is source rank
        # Note: all_to_all_single expects flattened or matching shapes;
        # here inputs/outputs have identical shapes except for dim0 split
        output = torch.empty_like(x)
        dist.all_to_all_single(output, x, group=group)
        
        # 4. Restore shape
        # (P, B, S_local, H_local, D) -> (B, P, S_local, H_local, D)
        output = output.permute(1, 0, 2, 3, 4).contiguous()
        
        # 5. Merge sequence dimension
        # (B, P, S_local, H_local, D) -> (B, P * S_local, H_local, D) -> (B, S_global, H_local, D)
        output = output.view(B, -1, H_local, D)
        
        return output


    # -------------------------------------------------------
    # 2. Transform: seq-parallel -> head-parallel
    # -------------------------------------------------------
    # q, k, v now hold the full sequence length but only 1/P of the heads
    # Shape: (B, S_total, H/P, D)
    q_global = seq_to_head(q_local)
    k_global = seq_to_head(k_local)
    v_global = seq_to_head(v_local)

    # -------------------------------------------------------
    # 3. Compute attention
    # -------------------------------------------------------
    # Shape is (B, Seq, H_local, Dim); transpose to (B, H_local, Seq, Dim) for SDPA
    attn_out = flash_attn_func(q_global, k_global, v_global, causal=False)

    # -------------------------------------------------------
    # 4. Transform back: head-parallel -> seq-parallel
    # -------------------------------------------------------
    # Back to (B, S_local, H, D)
    
    return attn_out

if __name__ == "__main__":
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    n_warmup = 5
    n_iters = 10

    B = 2
    H = 16
    SEQ = WORLD_SIZE * 8192
    DIM = 128
    dtype = torch.float16

    q_local = torch.randn((B, SEQ // WORLD_SIZE, H, DIM), device="cuda", dtype=dtype)
    k_local = torch.randn((B, SEQ // WORLD_SIZE, H, DIM), device="cuda", dtype=dtype)
    v_local = torch.randn((B, SEQ // WORLD_SIZE, H, DIM), device="cuda", dtype=dtype)

    attn_out = head_parallel_attention(q_local, k_local, v_local, WORLD_SIZE)
    output_local = head_to_seq(attn_out, B, WORLD_SIZE, SEQ, H, DIM)

    # check correctness with non-parallel attention
    # [B, S_local, H, D] -> [S_local, B, H, D]
    q_local_t = q_local.transpose(0, 1).contiguous()
    k_local_t = k_local.transpose(0, 1).contiguous()
    v_local_t = v_local.transpose(0, 1).contiguous()

    # all_gather along dim0 -> [WORLD_SIZE * S_local, B, H, D]
    q_global_t = torch.empty((SEQ, B, H, DIM), device="cuda", dtype=dtype)
    k_global_t = torch.empty((SEQ, B, H, DIM), device="cuda", dtype=dtype)
    v_global_t = torch.empty((SEQ, B, H, DIM), device="cuda", dtype=dtype)

    dist.all_gather_into_tensor(q_global_t, q_local_t)
    dist.all_gather_into_tensor(k_global_t, k_local_t)
    dist.all_gather_into_tensor(v_global_t, v_local_t)

    # [SEQ, B, H, D] -> [B, SEQ, H, D]
    q_global = q_global_t.transpose(0, 1).contiguous()
    k_global = k_global_t.transpose(0, 1).contiguous()
    v_global = v_global_t.transpose(0, 1).contiguous()

    attn_out_global = flash_attn_func(q_global, k_global, v_global, causal=False)

    # each rank gets its own slice
    attn_out_local_ref = attn_out_global[:, rank * (SEQ // WORLD_SIZE):(rank + 1) * (SEQ // WORLD_SIZE), :, :]
    assert torch.allclose(output_local, attn_out_local_ref, atol=1e-2, rtol=1e-2)
    print("Rank", rank, "passed the correctness check.")

    torch.cuda.synchronize()
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # use torch profiler to profile the all-to-all attention
    # with torch.profiler.profile(
    #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True
    # ) as prof:
    for _ in range(n_warmup):
        output_local = head_parallel_attention(q_local, k_local, v_local, WORLD_SIZE)

    # prof.export_chrome_trace(f"head_parallel_attention_rank{rank}.json")

    start.record()
    for _ in range(n_iters):
        output_local = head_parallel_attention(q_local, k_local, v_local, WORLD_SIZE)
    end.record()
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end) / n_iters  # milliseconds

    flops_per_matmul = 2.0 * B * H * SEQ * SEQ * DIM
    total_flops = 2 * flops_per_matmul
    per_gpu_flops = total_flops / WORLD_SIZE
    tflops = per_gpu_flops / 1e12 / (elapsed / 1e3)

    for rank_id in range(WORLD_SIZE):
        if rank == rank_id:
            print(f"Rank {rank}: tflops: {tflops:.2f} TFLOPs, time: {elapsed:.2f} ms")
        dist.barrier()

    dist.destroy_process_group()
