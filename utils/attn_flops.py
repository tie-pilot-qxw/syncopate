

def calculate_attention_tflops(B, H, SEQ, DIM, dur_ms, WORLD_SIZE):
    """Calculate TFLOPS for attention operation."""
    flops_per_matmul = 2.0 * B * H * SEQ * SEQ * DIM
    total_flops = 2 * flops_per_matmul
    per_gpu_flops = total_flops / WORLD_SIZE
    tflops = per_gpu_flops / 1e12 / (dur_ms / 1e3)
    return tflops


def calculate_time_from_tflops(B, H, SEQ, DIM, tflops, WORLD_SIZE):
    """Calculate time in milliseconds from TFLOPS for attention operation."""
    flops_per_matmul = 2.0 * B * H * SEQ * SEQ * DIM
    total_flops = 2 * flops_per_matmul
    per_gpu_flops = total_flops / WORLD_SIZE
    dur_ms = per_gpu_flops / 1e12 / tflops * 1e3
    return dur_ms


if __name__ == "__main__":
    B = 1
    H = 32

    DIM = 128

    tflops_4 = [628.8765431, 245.8397174, 631.3876933, 252.11]
    tflops_8 = [592.1637506, 184.0389392, 598.8886533, 112.62]

    world = 4
    SEQ = world * 4096
    for tflops in tflops_4:
        dur_ms = calculate_time_from_tflops(B, H, SEQ, DIM, tflops, world)
        print(f"tflops: {tflops:.2f}, time: {dur_ms:.2f} ms")

    world = 8
    SEQ = world * 4096
    for tflops in tflops_8:
        dur_ms = calculate_time_from_tflops(B, H, SEQ, DIM, tflops, world)
        print(f"tflops: {tflops:.2f}, time: {dur_ms:.2f} ms")
