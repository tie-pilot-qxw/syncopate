import triton
import triton.language as tl


@triton.jit
def example_local_kernel(out_ptr, M, BLOCK_M: tl.constexpr):
    tile_id = tl.program_id(0)  # @sy.tile_id
    num_pid_m = tl.cdiv(M, BLOCK_M)  # @sy.axis_count M block=BLOCK_M
    # @sy.dispatch begin
    # @sy.pid_map M=pid_m
    pid_m = tile_id
    offs = pid_m * BLOCK_M
    # @sy.dispatch end
    tl.store(out_ptr + offs + tl.arange(0, BLOCK_M), pid_m)
