from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
import triton_dist.language as dl
from triton.language.extra.cuda.language_extra import (
    __syncthreads,
    ld,
    st,
    tid,
    multimem_ld_reduce_v4,
    multimem_st_v4,
    st_v4_b32,
)
from triton.language.extra.cuda.utils import num_warps
from triton_dist.language.extra import libshmem_device
from triton_dist.utils import nvshmem_create_tensor, nvshmem_free_tensor_sync, nvshmem_barrier_all_on_stream

# this kernel is adopted from Triton-distributed

@triton.jit(do_not_specialize=[])
def _standalone_all_reduce_kernel(
    symm_input_ptr,
    symm_out_ptr,
    out_ptr,  #
    multi_st_barrier_ptr,  #
    M,
    N,  #
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    NUM_COMM_SMS: tl.constexpr,
    USE_MULTIMEM_ST: tl.constexpr,
):
    rank = dl.rank()
    world_size = dl.num_ranks()
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_pid_m * num_pid_n
    thread_idx = tid(0)
    block_dim = num_warps() * 32
    VEC_SIZE: tl.constexpr = 128 // tl.constexpr(symm_input_ptr.dtype.element_ty.primitive_bitwidth)
    tl.static_assert(BLOCK_SIZE_N % VEC_SIZE == 0)
    VEC_PER_ROW = BLOCK_SIZE_N // VEC_SIZE
    src_data_mc_ptr = libshmem_device.remote_mc_ptr(libshmem_device.NVSHMEMX_TEAM_NODE, symm_input_ptr)
    if not USE_MULTIMEM_ST:
        for tile_id in range(pid, num_tiles, NUM_COMM_SMS):
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
            tile_m = min(M - pid_m * BLOCK_SIZE_M, BLOCK_SIZE_M)
            cur_tile_nelem = tile_m * BLOCK_SIZE_N
            for idx in range(thread_idx, cur_tile_nelem // VEC_SIZE, block_dim):
                row_id = idx // VEC_PER_ROW
                col_id = idx % VEC_PER_ROW
                offset = (row_id + pid_m * BLOCK_SIZE_M) * N + col_id * VEC_SIZE + pid_n * BLOCK_SIZE_N
                val0, val1, val2, val3 = multimem_ld_reduce_v4(src_data_mc_ptr + offset)
                st_v4_b32(out_ptr + offset, val0, val1, val2, val3)
    else:
        symm_out_mc_ptr = libshmem_device.remote_mc_ptr(libshmem_device.NVSHMEMX_TEAM_NODE, symm_out_ptr)
        for tile_id in range(pid + rank * NUM_COMM_SMS, num_tiles, NUM_COMM_SMS * world_size):
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n

            tile_m = min(M - pid_m * BLOCK_SIZE_M, BLOCK_SIZE_M)
            cur_tile_nelem = tile_m * BLOCK_SIZE_N
            for idx in range(thread_idx, cur_tile_nelem // VEC_SIZE, block_dim):
                row_id = idx // VEC_PER_ROW
                col_id = idx % VEC_PER_ROW
                offset = (row_id + pid_m * BLOCK_SIZE_M) * N + col_id * VEC_SIZE + pid_n * BLOCK_SIZE_N
                val0, val1, val2, val3 = multimem_ld_reduce_v4(src_data_mc_ptr + offset)
                multimem_st_v4(symm_out_mc_ptr + offset, val0, val1, val2, val3)
        __syncthreads()

        if thread_idx < world_size:
            peer_ptr = dl.symm_at(multi_st_barrier_ptr, thread_idx)
            st(peer_ptr + rank * NUM_COMM_SMS + pid, 1, scope="sys", semantic="release")

        if thread_idx < world_size:
            multi_st_barrier_idx = thread_idx * NUM_COMM_SMS + pid
            while ld(multi_st_barrier_ptr + multi_st_barrier_idx, scope="sys", semantic="acquire") != 1:
                pass
            st(multi_st_barrier_ptr + multi_st_barrier_idx, 0)



class NVSHMEMAllReduceCollective:
    """NVSHMEM-based all-reduce built from the Triton GEMM+AR consumer kernel."""

    def __init__(
        self,
        world_size: int,
        local_world_size: int,
        num_comm_sms: int,
        block_size: Tuple[int, int] = (128, 256),
        use_multimem_st: bool = True,
    ) -> None:
        if world_size != local_world_size:
            raise ValueError("NVSHMEM all-reduce currently assumes a single-node group (world == local world).")
        if num_comm_sms <= 0:
            raise ValueError("num_comm_sms must be positive for NVSHMEM all-reduce.")

        self.world_size = world_size
        self.local_world_size = local_world_size
        self.num_comm_sms = num_comm_sms
        self.block_size_m, self.block_size_n = block_size
        self.use_multimem_st = use_multimem_st

        self.gemm_barrier_buf: Optional[torch.Tensor] = None
        self.multi_st_barrier_buf: Optional[torch.Tensor] = None

    def _ensure_buffers(self, M: int, N: int) -> None:
        """Allocate or grow NVSHMEM buffers to cover the given tile space."""
        if self.multi_st_barrier_buf is None:
            self.multi_st_barrier_buf = nvshmem_create_tensor((self.world_size * self.num_comm_sms,), torch.int32)

    def all_reduce_(self, input: torch.Tensor, output: torch.Tensor, stream: Optional[torch.cuda.Stream] = None) -> torch.Tensor:
        """In-place NVSHMEM all-reduce on a contiguous symmetric buffer."""
        if stream is None:
            stream = torch.cuda.current_stream()

        if not input.is_contiguous() or not output.is_contiguous():
            raise ValueError("NVSHMEM all-reduce requires a contiguous tensor.")
        if input.numel() == 0 or output.numel() == 0:
            return input
        if input.shape != output.shape:
            raise ValueError("Input and output tensors must have the same shape for all-reduce.")
        
        flat_input = input
        flat_output = output
        if input.dim() == 1:
            flat_input = input.view(1, input.shape[0])
            flat_output = output.view(1, output.shape[0])
        elif input.dim() > 2:
            flat_input = input.view(-1, input.shape[-1])
            flat_output = output.view(-1, output.shape[-1])

        M, N = flat_input.shape
        if N % self.block_size_n != 0:
            raise ValueError(f"Last dimension ({N}) must be a multiple of BLOCK_SIZE_N={self.block_size_n}.")

        self._ensure_buffers(M, N)

        with torch.cuda.stream(stream):
            # mark every tile ready for consumption
            if self.multi_st_barrier_buf is not None:
                self.multi_st_barrier_buf.zero_()

            grid = (self.num_comm_sms,)
            nvshmem_barrier_all_on_stream(stream)
            _standalone_all_reduce_kernel[grid](
                flat_input,
                flat_output,
                flat_output,
                self.multi_st_barrier_buf,
                M,
                N,
                BLOCK_SIZE_M=self.block_size_m,
                BLOCK_SIZE_N=self.block_size_n,
                NUM_COMM_SMS=self.num_comm_sms,
                USE_MULTIMEM_ST=self.use_multimem_st,
                num_warps=32,
            )

        return output

    def finalize(self) -> None:
        if self.multi_st_barrier_buf is not None:
            nvshmem_free_tensor_sync(self.multi_st_barrier_buf)
            self.multi_st_barrier_buf = None

    def __del__(self) -> None:
        try:
            self.finalize()
        except Exception:
            # avoid noisy destructor failures during interpreter shutdown
            pass
