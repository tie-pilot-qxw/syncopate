import triton
import triton.language as tl


@triton.jit
def example_local_kernel(out_ptr, M, BLOCK_M: tl.constexpr, wave_offsets=None, wave_sizes=None, cum_wave_sizes=None, cum_tiles=0, NUM_WAVES: tl.constexpr=0):
    tile_id = tl.program_id(0)  # @sy.tile_id
    num_pid_m = tl.cdiv(M, BLOCK_M)  # @sy.axis_count M block=BLOCK_M
    # auto-generated dispatch (wave-based)
    wave_dim: tl.constexpr = 1
    cum_wave_range = tl.arange(0, NUM_WAVES)
    cum_wave_sizes_vec = tl.load(cum_wave_sizes + cum_wave_range)
    cum_wave_sizes_vec = (cum_wave_sizes_vec) // ((BLOCK_M))
    wave_candidates = tl.where(cum_wave_sizes_vec > tile_id, cum_wave_range, NUM_WAVES)
    wave_idx = tl.min(wave_candidates)
    previous_cum = tl.where(wave_idx == 0, 0, (tl.load(cum_wave_sizes + wave_idx - 1)) // ((BLOCK_M)))
    local_tile_id = tile_id - previous_cum
    wave_m_offset = (tl.load(wave_offsets + wave_idx * wave_dim + 0)) // (BLOCK_M)
    wave_m_size = (tl.load(wave_sizes + wave_idx * wave_dim + 0)) // (BLOCK_M)
    local_num_pid_m = wave_m_size
    pid_m = local_tile_id
    pid_m += wave_m_offset
    offs = pid_m * BLOCK_M
    tl.store(out_ptr + offs + tl.arange(0, BLOCK_M), pid_m)
