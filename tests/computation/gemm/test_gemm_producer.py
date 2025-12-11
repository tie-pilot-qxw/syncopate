import torch
from syncopate.computation.gemm.template import gemm_producer
from triton_dist.utils import perf_func

if __name__ == '__main__':
    M = 8192
    N = 8192
    K = 28672 // 4
    world_size = 1
    num_gemm_sms = 132

    waves = 4

    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((N, K), device='cuda', dtype=torch.float16)
    c = torch.zeros((M, N), device='cuda', dtype=torch.float16)
    c_golden = torch.matmul(a, b.T)

    signal_buffer = torch.zeros((waves,), device='cuda', dtype=torch.uint64)
    signal_offsets = torch.arange(0, waves, device='cuda', dtype=torch.int32)
    cur_wave_size = torch.tensor([(M // waves, N) for _ in range(waves)], device='cuda', dtype=torch.int32)
    cur_wave_offset = torch.tensor([(i * (M // waves), 0) for i in range(waves)], device='cuda', dtype=torch.int32)

    print(signal_buffer)
    print(signal_offsets)
    print(cur_wave_size)
    print(cur_wave_offset)

    def func():
        gemm_producer(
            a, b, c, signal_buffer=signal_buffer, cur_wave_size=cur_wave_size, wave_offset=cur_wave_offset, signal_offsets=signal_offsets, num_gemm_sms=num_gemm_sms,
        )

    func()
    torch.testing.assert_close(c, c_golden, rtol=1e-2, atol=1e-2)
    print(signal_buffer)

    _, dur_cublas = perf_func(lambda: torch.matmul(a, b.T), warmup_iters=5, iters=10)
    _, dur_ms = perf_func(func, warmup_iters=5, iters=10)
    tflops = 2 * M * N * K / 1e12 / (dur_ms / 1e3)
    print(f"tflops: {tflops} time: {dur_ms}")
    tflops_cublas = 2 * M * N * K / 1e12 / (dur_cublas / 1e3)
    print(f"cublas tflops: {tflops_cublas} time: {dur_cublas}")
