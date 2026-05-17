from flash_attn import flash_attn_func
import os
import torch
import torch.distributed as dist

import pytest
import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton._internal_testing import is_cuda
from triton.tools.tensor_descriptor import TensorDescriptor
from triton.language.extra.cuda.language_extra import tid, atomic_cas, __fence
from triton_dist.utils import sleep_async
DEVICE = triton.runtime.driver.active.get_active_torch_device()


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    NUM_MMA_GROUPS = nargs["NUM_MMA_GROUPS"]
    BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS

    if nargs["FP8_OUTPUT"]:
        raise RuntimeError("BSHD attention example does not support FP8 inputs yet")

    nargs["desc_q"].block_shape = [BLOCK_M_SPLIT, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M_SPLIT, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_m"].block_shape = [BLOCK_M_SPLIT]


configs = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'NUM_BUFFERS': 2, 'NUM_MMA_WARPS': 8, 'NUM_MMA_GROUPS': 2},
                  num_stages=0, num_warps=4, pre_hook=_host_descriptor_pre_hook),
]

@triton.jit
def get_pid_range(tile_id, Z, H, num_pid_q, num_pid_kv):
    tile_id_q = tile_id % num_pid_q
    tile_id_h = (tile_id // num_pid_q) % H
    tile_id_z = tile_id // (num_pid_q * H)
    return tile_id_z, tile_id_h, tile_id_q, 0, num_pid_kv

@triton.autotune(configs=configs, key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"])
@triton.jit
def _attn_fwd_ws_pipelined_pingpong(sm_scale, desc_m,  #
                                    Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX_Q, N_CTX_KV,  #
                                    HEAD_DIM: tl.constexpr,  #
                                    BLOCK_M: tl.constexpr,  #
                                    BLOCK_N: tl.constexpr,  #
                                    FP8_OUTPUT: tl.constexpr,  #
                                    NUM_BUFFERS: tl.constexpr,  #
                                    NUM_MMA_WARPS: tl.constexpr,  #
                                    NUM_MMA_GROUPS: tl.constexpr,  #
                                    ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // NUM_MMA_GROUPS

    # allocate buffers
    q_tiles = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), tlx.dtype_of(desc_q), NUM_MMA_GROUPS)
    k_tiles = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(desc_k), NUM_BUFFERS)
    v_tiles = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(desc_v), NUM_BUFFERS)

    # allocate barriers
    q_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS, arrive_count=1)
    k_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=NUM_MMA_GROUPS)
    k_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=1)
    v_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=NUM_MMA_GROUPS)
    v_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=1)

    # rewrite to tile_id pattern:
    tile_id = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0) + tl.program_id(2) * tl.num_programs(0) * tl.num_programs(1) # @sy.tile_id

    z_blocks = Z # @sy.axis_count Z
    h_blocks = H # @sy.axis_count H
    q_blocks = tl.cdiv(N_CTX_Q, BLOCK_M) # @sy.axis_count Q block=BLOCK_M
    kv_blocks = tl.cdiv(N_CTX_KV, BLOCK_N) # @sy.axis_count KV=range block=BLOCK_N

    # @sy.dispatch begin

    # @sy.pid_map Z=pid_z H=pid_h Q=pid_q KV=kv_lo..kv_hi
    pid_z, pid_h, pid_q, kv_lo, kv_hi = get_pid_range(tile_id, z_blocks, h_blocks, q_blocks, kv_blocks)
    # @sy.dispatch end

    start_m = pid_q
    off_hz = pid_z * H + pid_h
    off_z = pid_z
    off_h = pid_h
    
    with tlx.async_tasks():
        # producer group
        with tlx.async_task("default"):
            # initialize offsets

            q_row_offset = off_z * N_CTX_Q
            q_col_offset = off_h * HEAD_DIM
            kv_row_offset = off_z * N_CTX_KV
            kv_col_offset = q_col_offset
            q_seq_offset = start_m * BLOCK_M
            kv_seq_offset = kv_lo * BLOCK_N

            # load q: it will stay in SRAM throughout
            for cid in tl.range(0, NUM_MMA_GROUPS, loop_unroll_factor=NUM_MMA_GROUPS):
                q_full = tlx.local_view(q_fulls, cid)
                tlx.barrier_expect_bytes(q_full, 2 * BLOCK_M_SPLIT * HEAD_DIM)  # float16
                q_tile = tlx.local_view(q_tiles, cid)
                seq_offset = q_seq_offset + cid * BLOCK_M_SPLIT
                row = q_row_offset + seq_offset
                tlx.async_descriptor_load(desc_q, q_tile, [row, q_col_offset], q_full)

            # loop over loading k, v
            kv_phase = 0
            acc_cnt = 0
            for _ in tl.range(kv_lo, kv_hi):
                buf_id = acc_cnt % NUM_BUFFERS
                # buffers in a row share the same phase
                kv_phase = kv_phase ^ (buf_id == 0)

                # wait for the K buffer to be released by the consumer
                k_empty = tlx.local_view(k_empties, buf_id)
                tlx.barrier_wait(k_empty, kv_phase)
                # load K
                k_full = tlx.local_view(k_fulls, buf_id)
                k_tile = tlx.local_view(k_tiles, buf_id)
                tlx.barrier_expect_bytes(k_full, 2 * BLOCK_N * HEAD_DIM)  # float16
                row_k = kv_row_offset + kv_seq_offset
                tlx.async_descriptor_load(desc_k, k_tile, [row_k, kv_col_offset], k_full)

                # wait for the V buffer to be released by the consumer
                v_empty = tlx.local_view(v_empties, buf_id)
                tlx.barrier_wait(v_empty, kv_phase)
                # load V
                v_full = tlx.local_view(v_fulls, buf_id)
                v_tile = tlx.local_view(v_tiles, buf_id)
                tlx.barrier_expect_bytes(v_full, 2 * BLOCK_N * HEAD_DIM)  # float16
                row_v = kv_row_offset + kv_seq_offset
                tlx.async_descriptor_load(desc_v, v_tile, [row_v, kv_col_offset], v_full)

                kv_seq_offset += BLOCK_N
                acc_cnt += 1

        # consumer group
        with tlx.async_task(num_warps=NUM_MMA_WARPS // NUM_MMA_GROUPS, registers=232, replicate=NUM_MMA_GROUPS):
            # initialize pointer to m and l
            m_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) - float("inf")
            l_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) + 1.0
            acc = tl.zeros([BLOCK_M_SPLIT, HEAD_DIM], dtype=tl.float32)

            # load scales
            qk_scale = sm_scale
            qk_scale *= 1.44269504  # 1/log(2)

            # wait for the Q buffer to be populated by the producer
            cid: tl.constexpr = tlx.async_task_replica_id()
            q_full = tlx.local_view(q_fulls, cid)
            tlx.barrier_wait(q_full, 0)
            q_tile = tlx.local_view(q_tiles, cid)

            k_phase = 0
            v_phase = 1
            k_buf_id = 0
            v_buf_id = 0

            # wait for the K[0] buffer to be populated by the producer
            k_full = tlx.local_view(k_fulls, k_buf_id)
            tlx.barrier_wait(k_full, k_phase)
            k_tile = tlx.local_view(k_tiles, k_buf_id)

            # -- compute qk[0] ----
            k_tile = tlx.local_trans(k_tile)

            if cid == 0:
                # Consumer 0 waits for Consumer 1 to reach synchronization point at barrier 9.
                tlx.named_barrier_wait(9, 256)
            else:
                # Consumer 1 signals its arrival at barrier 9.
                tlx.named_barrier_arrive(9, 256)
                # Then waits at barrier 10 until Consumer 0 finishes issuing its async_dot.
                tlx.named_barrier_wait(10, 256)

            qk = tlx.async_dot(q_tile, k_tile)

            if cid == 0:
                # After issuing async_dot, Consumer 0 signals barrier 10 to unblock Consumer 1.
                tlx.named_barrier_arrive(10, 256)

            # wait for the MMA using to complete
            qk = tlx.async_dot_wait(0, qk)
            # release the K buffer
            k_empty = tlx.local_view(k_empties, k_buf_id)
            tlx.barrier_arrive(k_empty, 1)

            # -- compute m_i and l_i ----
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = tl.math.exp2(qk)
            # -- compute correction factor
            alpha = tl.math.exp2(m_i - m_ij)
            # -- update output accumulator[0] --
            acc = acc * alpha[:, None]
            l_ij = tl.sum(p, 1)
            l_i = l_i * alpha + l_ij
            m_i = m_ij
            acc_cnt = 1

            # loop over k, v and update accumulator
            for _ in tl.range(kv_lo + 1, kv_hi):
                k_buf_id = acc_cnt % NUM_BUFFERS
                # buffers in a row share the same phase
                k_phase = k_phase ^ (k_buf_id == 0)

                # wait for the K buffer to be populated by the producer
                k_full = tlx.local_view(k_fulls, k_buf_id)
                tlx.barrier_wait(k_full, k_phase)
                k_tile = tlx.local_view(k_tiles, k_buf_id)

                # compute qk for the current iteration
                k_tile = tlx.local_trans(k_tile)
                qk = tlx.async_dot(q_tile, k_tile)

                # compute pv from the previous iteration
                # wait for the previous V buffer to be populated by the producer
                v_buf_id = (acc_cnt - 1) % NUM_BUFFERS
                v_phase = v_phase ^ (v_buf_id == 0)
                v_full = tlx.local_view(v_fulls, v_buf_id)
                tlx.barrier_wait(v_full, v_phase)
                v_tile = tlx.local_view(v_tiles, v_buf_id)
                # prepare p and v for the dot
                p = p.to(tlx.dtype_of(desc_k))
                acc = tlx.async_dot(p, v_tile, acc)

                # wait for the current qk MMA to complete
                qk = tlx.async_dot_wait(1, qk)
                # release the K buffer
                k_empty = tlx.local_view(k_empties, k_buf_id)
                tlx.barrier_arrive(k_empty, 1)

                # -- compute m_i and l_i ----
                m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
                qk = qk * qk_scale - m_ij[:, None]
                p = tl.math.exp2(qk)
                # -- compute correction factor
                alpha = tl.math.exp2(m_i - m_ij)
                l_ij = tl.sum(p, 1)
                # update m_i and l_i
                l_i = l_i * alpha + l_ij
                m_i = m_ij

                # -- update output accumulator --
                # wait for the previous pv MMA to complete
                acc = tlx.async_dot_wait(0, acc)
                # release the V buffer
                v_empty = tlx.local_view(v_empties, v_buf_id)
                tlx.barrier_arrive(v_empty, 1)
                acc = acc * alpha[:, None]
                acc_cnt += 1

            # compute pv from the last iteration
            # wait for the V buffer to be populated by the producer
            v_buf_id = (acc_cnt - 1) % NUM_BUFFERS
            v_phase = v_phase ^ (v_buf_id == 0)
            v_full = tlx.local_view(v_fulls, v_buf_id)
            tlx.barrier_wait(v_full, v_phase)
            v_tile = tlx.local_view(v_tiles, v_buf_id)
            # prepare p and v for the dot
            p = p.to(tlx.dtype_of(desc_k))
            acc = tlx.async_dot(p, v_tile, acc)
            # wait for the MMA using to complete
            acc = tlx.async_dot_wait(0, acc)
            # release the V buffer
            v_empty = tlx.local_view(v_empties, v_buf_id)
            tlx.barrier_arrive(v_empty, 1)

            # epilogue
            qo_row_offset = off_z * N_CTX_Q + start_m * BLOCK_M + cid * BLOCK_M_SPLIT
            qo_col_offset = off_h * HEAD_DIM

            # cal log2 lse
            m_i += tl.math.log2(l_i)
            acc = acc / l_i[:, None]
            # offs_m = start_m * BLOCK_M + cid * BLOCK_M_SPLIT + tl.arange(0, BLOCK_M_SPLIT)
            # m_ptrs = M + off_hz * N_CTX + offs_m

            offs_m = off_hz * N_CTX_Q + start_m * BLOCK_M + cid * BLOCK_M_SPLIT

            desc_m.store([offs_m], m_i)
            desc_o.store([qo_row_offset, qo_col_offset], acc.to(tlx.dtype_of(desc_o)))

    # @sy.producer_epilogue


def attention_forward(q, k, v, sm_scale, output_buffer = None): # @sy.host_entry
    # shape constraints
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "expected B×S×H×D layout"
    B, N_CTX_Q, H, HEAD_DIM_Q = q.shape
    Bk, N_CTX_KV, Hk, HEAD_DIM_K = k.shape
    Bv, Nv, Hv, HEAD_DIM_V = v.shape
    assert (B, H, HEAD_DIM_Q) == (Bk, Hk, HEAD_DIM_K) == (Bv, Hv, HEAD_DIM_V)
    assert HEAD_DIM_Q in {16, 32, 64, 128, 256}
    if q.dtype == torch.float8_e5m2:
        raise NotImplementedError("BSHD attention example does not support FP8 inputs yet")
    if output_buffer is not None:
        o = output_buffer
    else:
        o = torch.empty_like(q)

    # metadata tensors follow the B×H×S order expected by the kernel epilogue.
    M = torch.empty((B, H, N_CTX_Q), device=q.device, dtype=torch.float32)

    dummy_block = [1, 1]
    # 2D descriptor: rows index (B, N_CTX), cols index (H, HEAD_DIM)
    row_stride = H * HEAD_DIM_Q
    desc_q = TensorDescriptor(q, shape=[B * N_CTX_Q, H * HEAD_DIM_Q], strides=[row_stride, 1], block_shape=dummy_block)
    desc_o = TensorDescriptor(o, shape=[B * N_CTX_Q, H * HEAD_DIM_Q], strides=[row_stride, 1], block_shape=dummy_block)
    desc_k = TensorDescriptor(k, shape=[B * N_CTX_KV, H * HEAD_DIM_Q], strides=[row_stride, 1], block_shape=dummy_block)
    desc_v = TensorDescriptor(v, shape=[B * N_CTX_KV, H * HEAD_DIM_Q], strides=[row_stride, 1], block_shape=dummy_block)
    desc_m = TensorDescriptor(M, shape=[B * H * N_CTX_Q], strides=[1], block_shape=[1])

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)


    def grid(META):
        # assert q.shape[2] % (META["KV_SPLIT"] * META["BLOCK_N"]) == 0 # ensure even split
        num_tiles = triton.cdiv(N_CTX_Q, META["BLOCK_M"]) * B * H # @sy.num_tiles
        return (num_tiles,)

    # @sy.kernel_launch
    _attn_fwd_ws_pipelined_pingpong[grid](
        sm_scale, desc_m,  #
        B, H,  #
        desc_q, desc_k, desc_v, desc_o,  #
        N_CTX_Q=N_CTX_Q,  #
        N_CTX_KV=N_CTX_KV,  #
        HEAD_DIM=HEAD_DIM_Q,  #
        FP8_OUTPUT=False,  #
    )

    return o


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
    attn_out_head = attention_forward(q_head, k_head, v_head, sm_scale=1.0)
    
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
    SEQ = 1024 * 4 # Global Sequence Length
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
    attn_out_global = flash_attn_func(q_global, k_global, v_global, softmax_scale=1.0)
    
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
    sleep_async(1000)
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
