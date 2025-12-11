import pytest
import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton._internal_testing import is_cuda
from triton.tools.tensor_descriptor import TensorDescriptor
from triton.language.extra.cuda.language_extra import tid, atomic_cas, __fence

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    HEAD_DIM = nargs["HEAD_DIM"]
    NUM_MMA_GROUPS = nargs["NUM_MMA_GROUPS"]
    BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
    nargs["desc_q"].block_shape = [BLOCK_M_SPLIT, HEAD_DIM]
    if nargs["FP8_OUTPUT"]:
        nargs["desc_v"].block_shape = [HEAD_DIM, BLOCK_N]
    else:
        nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M_SPLIT, HEAD_DIM]
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
def _attn_fwd_ws_pipelined_pingpong(sm_scale, first, guard, desc_m,  #
                                    Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX_Q, N_CTX_KV,  #
                                    KV_SPLIT: tl.constexpr,
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
    tile_id = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0) + tl.program_id(2) * tl.num_programs(0) * tl.num_programs(1)

    local_tile_id = tile_id - tl.program_id(2) * tl.num_programs(0) * tl.num_programs(1)

    q_blocks = tl.cdiv(N_CTX_Q, BLOCK_M)
    kv_blocks = tl.cdiv(N_CTX_KV // KV_SPLIT, BLOCK_N)


    pid_z, pid_h, pid_q, kv_lo, kv_hi = get_pid_range(local_tile_id, Z, H, q_blocks, kv_blocks)

    offset_kv = tl.program_id(2) * kv_blocks
    kv_lo += offset_kv
    kv_hi += offset_kv

    start_m = pid_q
    off_hz = pid_z * H + pid_h
    off_z = pid_z
    off_h = pid_h
    
    with tlx.async_tasks():
        # producer group
        with tlx.async_task("default"):
            # initialize offsets

            offset_y_q = off_z * (N_CTX_Q * H) + off_h * N_CTX_Q
            offset_y_kv = off_z * (N_CTX_KV * H) + off_h * N_CTX_KV
            qo_offset_y = offset_y_q + start_m * BLOCK_M
            kv_offset_y = offset_y_kv + kv_lo * BLOCK_N

            # load q: it will stay in SRAM throughout
            for cid in tl.range(0, NUM_MMA_GROUPS, loop_unroll_factor=NUM_MMA_GROUPS):
                q_full = tlx.local_view(q_fulls, cid)
                tlx.barrier_expect_bytes(q_full, 2 * BLOCK_M_SPLIT * HEAD_DIM)  # float16
                q_tile = tlx.local_view(q_tiles, cid)
                qo_offset_y_split = qo_offset_y + cid * BLOCK_M_SPLIT
                tlx.async_descriptor_load(desc_q, q_tile, [qo_offset_y_split, 0], q_full)

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
                tlx.async_descriptor_load(desc_k, k_tile, [kv_offset_y, 0], k_full)

                # wait for the V buffer to be released by the consumer
                v_empty = tlx.local_view(v_empties, buf_id)
                tlx.barrier_wait(v_empty, kv_phase)
                # load V
                v_full = tlx.local_view(v_fulls, buf_id)
                v_tile = tlx.local_view(v_tiles, buf_id)
                tlx.barrier_expect_bytes(v_full, 2 * BLOCK_N * HEAD_DIM)  # float16
                tlx.async_descriptor_load(desc_v, v_tile, [kv_offset_y, 0], v_full)

                kv_offset_y += BLOCK_N
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
            offset_y = off_z * (N_CTX_Q * H) + off_h * N_CTX_Q
            qo_offset_y = offset_y + start_m * BLOCK_M
            qo_offset_y_split = qo_offset_y + cid * BLOCK_M_SPLIT

            # cal log2 lse
            m_i += tl.math.log2(l_i)
            acc = acc / l_i[:, None]
            # offs_m = start_m * BLOCK_M + cid * BLOCK_M_SPLIT + tl.arange(0, BLOCK_M_SPLIT)
            # m_ptrs = M + off_hz * N_CTX + offs_m

            offs_m = off_hz * N_CTX_Q + start_m * BLOCK_M + cid * BLOCK_M_SPLIT

            offs_lock = off_hz * N_CTX_Q + start_m * NUM_MMA_GROUPS + cid # unique lock per warpgroup

            # chose one thread per warpgroup to acquire the lock
            if tid(axis=0) % 128 == 0:
                while atomic_cas(guard + offs_lock, 0, 1, semantic="acquire", scope="gpu") != 0:
                    pass

            # sync warpgroup
            if cid == 0:
                tlx.named_barrier_wait(11, 128)
            else:
                tlx.named_barrier_wait(12, 128)
                
            first_load = tl.load(first + offs_lock)
            if first_load == 0: # direct store
                desc_m.store([offs_m], m_i)
                # tl.store(m_ptrs, m_i)
                desc_o.store([qo_offset_y_split, 0], acc.to(tlx.dtype_of(desc_o)))
                tl.store(first + offs_lock, 1)

            else: # load the old lse and acc
                m_old = desc_m.load([offs_m])
                acc_old = desc_o.load([qo_offset_y_split, 0]).to(tl.float32)

                delta = m_old - m_i
                mmax = tl.maximum(m_old, m_i)
                m_new = mmax + tl.math.log2(1.0 + tl.math.exp2(-tl.abs(delta)))

                w_old = tl.math.exp2(m_old - m_new)
                w_part = tl.math.exp2(m_i - m_new)

                acc_new = acc_old * w_old[:, None] + acc * w_part[:, None]

                desc_m.store([offs_m], m_new)
                desc_o.store([qo_offset_y_split, 0], acc_new.to(tlx.dtype_of(desc_o)))

            # sync warpgroup
            if cid == 0:
                tlx.named_barrier_wait(11, 128)
            else:
                tlx.named_barrier_wait(12, 128)

            # chose one thread per warpgroup to release the lock
            if tid(axis=0) % 128 == 0:
                # this is for test purpose, normally the release should not fail
                while atomic_cas(guard + offs_lock, 1, 0, semantic="release", scope="gpu") != 1:
                    pass




class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        extra_kern_args = {}

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        guard = torch.zeros((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.int32) # lock to prevent race condition
        first = torch.zeros((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.int8) # flag to indicate first write
        # Note that on Hopper we cannot perform a FP8 dot with a non-transposed second tensor
        y_dim_q  = q.shape[0] * q.shape[1] * q.shape[2]   # Z*H*N_CTX_Q
        y_dim_kv = k.shape[0] * k.shape[1] * k.shape[2]   # Z*H*N_CTX_KV
        dummy_block = [1, 1]
        # Q / O / M flattened along N_CTX_Q
        desc_q = TensorDescriptor(q, shape=[y_dim_q,  HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        desc_o = TensorDescriptor(o, shape=[y_dim_q,  HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        desc_m = TensorDescriptor(M, shape=[y_dim_q],               strides=[1],            block_shape=[1])
        # K / V flattened along N_CTX_KV
        desc_k = TensorDescriptor(k, shape=[y_dim_kv, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        if q.dtype == torch.float8_e5m2:
            # In the FP8 path v is transposed: v ∈ [Z, H, HEAD_DIM, Nk]
            # The first stride entry is the row stride (elements to move along y), so use Nk instead of Nq
            desc_v = TensorDescriptor(v, shape=[HEAD_DIM_K, y_dim_kv], strides=[k.shape[2], 1], block_shape=dummy_block)
        else:
            desc_v = TensorDescriptor(v, shape=[y_dim_kv, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        kv_split = 1


        def grid(META):
            assert q.shape[2] % (META["KV_SPLIT"] * META["BLOCK_N"]) == 0 # ensure even split
            return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], META["KV_SPLIT"])

        ctx.grid = grid
        _attn_fwd_ws_pipelined_pingpong[grid](
            sm_scale, first, guard, desc_m,  #
            q.shape[0], q.shape[1],  #
            desc_q, desc_k, desc_v, desc_o,  #
            N_CTX_Q=q.shape[2],  #
            N_CTX_KV=k.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
            KV_SPLIT=kv_split,  #
            **extra_kern_args)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        return o


attention = _attention.apply


@pytest.mark.parametrize("Z", [8])
@pytest.mark.parametrize("H", [16])
@pytest.mark.parametrize("N_CTX", [1024])
@pytest.mark.parametrize("HEAD_DIM", [128])
@pytest.mark.parametrize("mode", ["fwd"])
@pytest.mark.parametrize("provider", ["triton-fp16"])
@pytest.mark.skipif(
    not is_cuda() or torch.cuda.get_device_capability()[0] != 9,
    reason="Requires Hopper GPU",
)
def test_op(Z, H, N_CTX, HEAD_DIM, mode, provider, dtype=torch.float16):
    if mode == "bwd":
        pytest.skip("Backward pass not supported.")
    # torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX // 2, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()) # test kv with different sequence length
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 0.5
    from flash_attn.flash_attn_interface import flash_attn_func
    fa_out = flash_attn_func(q.permute(0, 2, 1, 3).contiguous(),
                             k.permute(0, 2, 1, 3).contiguous(),
                             v.permute(0, 2, 1, 3).contiguous(),
                             causal=False,
                             softmax_scale=sm_scale).permute(0, 2, 1, 3).contiguous()
    
    # reference implementation
    ref_dtype = dtype
    if mode == "fwd" and "fp8" in provider:
        ref_dtype = torch.float32
    q = q.to(ref_dtype)
    k = k.to(ref_dtype)
    v = v.to(ref_dtype)
    M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p = torch.softmax(p.float(), dim=-1)
    p = p.to(ref_dtype)
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v).half()
    # triton implementation
    if mode == "fwd" and "fp8" in provider:
        q = q.to(torch.float8_e5m2)
        k = k.to(torch.float8_e5m2)
        v = v.permute(0, 1, 3, 2).contiguous()
        v = v.permute(0, 1, 3, 2)
        v = v.to(torch.float8_e5m2)
    tri_out = attention(q, k, v, sm_scale).half()
    if mode == "fwd":
        atol = 3 if "fp8" in provider else 5e-2
        torch.testing.assert_close(tri_out, ref_out, atol=atol, rtol=0)
        return
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    # compare
    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)
    torch.testing.assert_close(fa_out, tri_out, atol=1e-2, rtol=0)
    rtol = 0.0
    torch.testing.assert_close(tri_dv, ref_dv, atol=1e-2, rtol=rtol)
    torch.testing.assert_close(tri_dk, ref_dk, atol=1e-2, rtol=rtol)
    torch.testing.assert_close(tri_dq, ref_dq, atol=1e-2, rtol=rtol)


try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

TORCH_HAS_FP8 = False
BATCH, N_HEADS, HEAD_DIM = 4, 32, 128
# vary seq length for fixed head and batch=4
configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["N_CTX"],
        x_vals=[2**i for i in range(10, 15)],
        line_arg="provider",
        line_vals=["triton-fp16"] + (["flash"] if HAS_FLASH else []),
        line_names=["Triton [FP16]"] + (["Flash-2"] if HAS_FLASH else []),
        styles=[("red", "-"), ("blue", "-"), ("green", "-")],
        ylabel="TFLOPS",
        plot_name=f"fused-attention-ws-pipelined-pingpong-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}",
        args={
            "H": N_HEADS,
            "BATCH": BATCH,
            "HEAD_DIM": HEAD_DIM,
            "mode": "fwd",
        },
    ))


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, mode, provider, device=DEVICE):
    assert mode in ["fwd", "bwd"]
    dtype = torch.float16
    if "triton" in provider:
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        if mode == "fwd" and "fp8" in provider:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
            v = v.permute(0, 1, 3, 2).contiguous()
            v = v.permute(0, 1, 3, 2)
            v = v.to(torch.float8_e5m2)
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, sm_scale)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)

    if provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(qkv)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    test_op(
        Z=1,
        H=1,
        N_CTX=8192 * 2,
        HEAD_DIM=128,
        mode="fwd",
        provider="triton-fp16",
        dtype=torch.float16,
    )
    if is_cuda() and torch.cuda.get_device_capability()[0] == 9:
        print("Running benchmarks...")
        bench_flash_attention.run(save_path=None, print_data=True, show_plots=False)
    else:
        print("Skipping benchmarks, no Hopper GPU found.")
