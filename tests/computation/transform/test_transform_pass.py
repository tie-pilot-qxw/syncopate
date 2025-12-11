import linecache
import re
from pathlib import Path
import textwrap
from typing import Iterable, Sequence, Tuple

import pytest
import torch

from syncopate.computation.transform import AnnotationTransformer


def __remove_autotune_decorator(source: str) -> str:
    pattern = re.compile(r"@triton\.autotune\([\s\S]*?\)\n", re.MULTILINE)
    return re.sub(pattern, "", source, count=1)


def test_example_local_kernel_transformation_matches_expected():
    transformer = AnnotationTransformer()
    base = Path("tests/computation/transform/examples")
    source = (base / "simple.py").read_text()
    expected = (base / "simple_expected.py").read_text()

    result = transformer.transform(source)
    print(result)

    assert result.strip() == expected.strip()


def test_example_matmul_injects_wave_dispatch_block():
    transformer = AnnotationTransformer()
    base = Path("tests/computation/transform/examples")
    source = (base / "example_matmul.py").read_text()
    if "# Unit Test" in source:
        source = source.split("# Unit Test", 1)[0]
    source = __remove_autotune_decorator(source)

    result = transformer.transform(source)
    lines = result.splitlines()
    print(result)

    start = next(
        idx for idx, line in enumerate(lines) if "# auto-generated dispatch" in line
    )
    end = next(
        idx for idx in range(start, len(lines))
        if "pid_n += wave_n_offset" in lines[idx]
    )
    snippet = "\n".join(lines[start : end + 1])

    expected_snippet = textwrap.dedent(
        """
        # auto-generated dispatch (wave-based)
        wave_dim: tl.constexpr = 2
        cum_wave_range = tl.arange(0, NUM_WAVES)
        cum_wave_sizes_vec = tl.load(cum_wave_sizes + cum_wave_range)
        cum_wave_sizes_vec = (cum_wave_sizes_vec) // ((BLOCK_SIZE_M) * (BLOCK_SIZE_N))
        wave_candidates = tl.where(cum_wave_sizes_vec > pid, cum_wave_range, NUM_WAVES)
        wave_idx = tl.min(wave_candidates)
        previous_cum = tl.where(wave_idx == 0, 0, (tl.load(cum_wave_sizes + wave_idx - 1)) // ((BLOCK_SIZE_M) * (BLOCK_SIZE_N)))
        local_pid = pid - previous_cum
        wave_m_offset = (tl.load(wave_offsets + wave_idx * wave_dim + 0)) // (BLOCK_SIZE_M)
        wave_m_size = (tl.load(wave_sizes + wave_idx * wave_dim + 0)) // (BLOCK_SIZE_M)
        local_num_pid_m = wave_m_size
        wave_n_offset = (tl.load(wave_offsets + wave_idx * wave_dim + 1)) // (BLOCK_SIZE_N)
        wave_n_size = (tl.load(wave_sizes + wave_idx * wave_dim + 1)) // (BLOCK_SIZE_N)
        local_num_pid_n = wave_n_size

        pid_m, pid_n = _get_pid_mn(local_pid, local_num_pid_m, local_num_pid_n, GROUP_SIZE_M)
        pid_m += wave_m_offset
        pid_n += wave_n_offset
        """
    ).strip()

    assert textwrap.dedent(snippet).strip() == expected_snippet


def test_example_attn_injects_wave_dispatch_with_range_axis():
    transformer = AnnotationTransformer()
    base = Path("tests/computation/transform/examples")
    source = (base / "example_attn.py").read_text()
    source = __remove_autotune_decorator(source)

    result = transformer.transform(source)
    print(result)
    lines = result.splitlines()

    start = next(
        idx for idx, line in enumerate(lines) if "# auto-generated dispatch" in line
    )
    end = next(
        idx
        for idx in range(start, len(lines))
        if "kv_hi += wave_kv_offset" in lines[idx]
    )
    snippet = "\n".join(lines[start : end + 1])

    assert "wave_dim: tl.constexpr = 4" in snippet
    assert (
        "pid_z, pid_h, pid_q, kv_lo, kv_hi = get_pid_range("
        "local_tile_id, local_z_blocks, local_h_blocks, local_q_blocks, local_kv_blocks"
        ")" in snippet
    )
    assert "kv_lo += wave_kv_offset" in snippet
    assert "kv_hi += wave_kv_offset" in snippet


def test_example_attn_consumer_mode_injects_signal_wait():
    transformer = AnnotationTransformer(enable_consumer=True)
    base = Path("tests/computation/transform/examples")
    source = (base / "example_attn.py").read_text()
    source = __remove_autotune_decorator(source)

    result = transformer.transform(source)
    assert "import triton_dist.language as dl" in result

    lines = result.splitlines()
    for line in lines:
        print(line)
    start = next(
        idx for idx, line in enumerate(lines) if "# auto-generated dispatch" in line
    )
    end = next(
        idx
        for idx in range(start, len(lines))
        if "kv_hi += wave_kv_offset" in lines[idx]
    )
    snippet = "\n".join(lines[start : end + 1])

    assert "signal_ptr=None" in result
    assert "wave_signal_offset = tl.load(signal_offsets + wave_idx)" in snippet
    assert 'token = dl.wait(' in snippet
    assert "dl.consume_token(desc_k, token)" in snippet
    assert "dl.consume_token(desc_v, token)" in snippet


def test_persistent_gemm_injects_persistent_blocks():
    transformer = AnnotationTransformer(enable_consumer=True, consumer_descriptors=("a_desc",))
    base = Path("tests/computation/transform/examples")
    source = (base / "example_persistent_gemm.py").read_text()

    result = transformer.transform(source)
    print(result)

    assert "cur_wave_sizes=None" in result
    assert "wave_offsets=None" in result
    assert "signal_offsets=None" in result
    assert "signal_ptr=None" in result

    assert "# auto-generated persistent init (wave-based)" in result
    assert "cur_wave = 0" in result
    assert "cur_signal_offset = tl.load(signal_offsets + cur_wave)" in result
    assert "cur_tiles = cur_m_tiles * cur_n_tiles" in result

    assert "# auto-generated persistent dispatch (wave-based)" in result
    assert "while tile_id >= past_tiles + cur_tiles" in result
    assert "if new_wave and cur_signal_offset >= 0" in result
    assert "tile_id - past_tiles" in result
    assert "pid_m += cur_m_offset" in result
    assert "pid_n += cur_n_offset" in result


def test_persistent_gemm_producer_injects_epilogue():
    transformer = AnnotationTransformer(enable_producer=True)
    base = Path("tests/computation/transform/examples")
    source = (base / "example_persistent_gemm.py").read_text()

    result = transformer.transform(source)
    print(result)

    assert "signal_ptr=None" in result
    assert "signal_offsets=None" in result
    assert "counter_ptr=None" in result
    assert "# auto-generated producer epilogue" in result
    assert "val = tl.atomic_add(counter_ptr + cur_wave" in result
    assert "dl.notify(signal_ptr + cur_signal_offset" in result


def test_persistent_gemm_dual_role_uses_distinct_signals():
    transformer = AnnotationTransformer(
        enable_consumer=True,
        consumer_descriptors=("a_desc",),
        enable_producer=True,
    )
    base = Path("tests/computation/transform/examples")
    source = (base / "example_persistent_gemm.py").read_text()

    result = transformer.transform(source)
    print(result)

    assert "consumer_signal_ptr=None" in result
    assert "producer_signal_ptr=None" in result
    assert "producer_counter_ptr=None" in result
    assert "cur_consumer_signal_offset = tl.load(consumer_signal_offsets + cur_wave)" in result
    assert "cur_producer_signal_offset = tl.load(producer_signal_offsets + cur_wave)" in result
    assert "dl.wait(consumer_signal_ptr + cur_consumer_signal_offset" in result
    assert "tl.atomic_add(producer_counter_ptr + cur_wave" in result
    assert "dl.notify(producer_signal_ptr + cur_producer_signal_offset" in result


def _reference_get_pid_mn(tile_id: int, num_pid_m: int, num_pid_n: int, group_size_m: int):
    num_pid_in_group = group_size_m * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * group_size_m
    group_size_m_eff = min(num_pid_m - first_pid_m, group_size_m)
    pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m_eff)
    pid_n = (tile_id % num_pid_in_group) // group_size_m_eff
    return pid_m, pid_n


def _simulate_wave_dispatch(
    tile_id: int,
    wave_offsets: Sequence[Tuple[int, int]],
    wave_sizes: Sequence[Tuple[int, int]],
    cum_wave_sizes: Sequence[int],
    group_size_m: int,
) -> Tuple[int, int]:
    for idx, cumulative in enumerate(cum_wave_sizes):
        if tile_id < cumulative:
            wave_idx = idx
            break
    else:
        raise AssertionError("tile_id exceeds total number of tiles")

    previous_cum = 0 if wave_idx == 0 else cum_wave_sizes[wave_idx - 1]
    local_tile_id = tile_id - previous_cum
    local_m, local_n = wave_sizes[wave_idx]
    pid_m, pid_n = _reference_get_pid_mn(local_tile_id, local_m, local_n, group_size_m)
    offset_m, offset_n = wave_offsets[wave_idx]
    return pid_m + offset_m, pid_n + offset_n


@pytest.mark.parametrize(
    (
        "num_pid_m",
        "num_pid_n",
        "group_size_m",
        "wave_offsets",
        "wave_sizes",
        "cum_wave_sizes",
        "expected_pairs",
    ),
    [
        (5, 7, 4, [(0, 0)], [(5, 7)], [35], None),
        (3, 2, 2, [(0, 0)], [(3, 2)], [6], None),
        (
            4,
            2,
            8,
            [(0, 0)],
            [(4, 2)],
            [8],
            [
                (0, 0),
                (1, 0),
                (2, 0),
                (3, 0),
                (0, 1),
                (1, 1),
                (2, 1),
                (3, 1),
            ],
        ),  # mirrors tmp.py case 1
        (
            4,
            2,
            8,
            [(0, 0), (2, 0)],  # mirrors tmp.py case 2
            [(2, 2), (2, 2)],
            [4, 8],
            [
                (0, 0),
                (1, 0),
                (0, 1),
                (1, 1),
                (2, 0),
                (3, 0),
                (2, 1),
                (3, 1),
            ],
        ),
    ],
)
def test_wave_dispatch_matches_reference_for_cases(
    num_pid_m: int,
    num_pid_n: int,
    group_size_m: int,
    wave_offsets: Iterable[Tuple[int, int]],
    wave_sizes: Iterable[Tuple[int, int]],
    cum_wave_sizes: Iterable[int],
    expected_pairs: Iterable[Tuple[int, int]] | None,
):
    wave_offsets = list(wave_offsets)
    wave_sizes = list(wave_sizes)
    cum_wave_sizes = list(cum_wave_sizes)
    expected_pairs = None if expected_pairs is None else list(expected_pairs)

    assert len(wave_offsets) == len(wave_sizes) == len(cum_wave_sizes)
    total_tiles = num_pid_m * num_pid_n
    assert cum_wave_sizes[-1] == total_tiles

    for tile_id in range(total_tiles):
        if expected_pairs is None:
            expected = _reference_get_pid_mn(tile_id, num_pid_m, num_pid_n, group_size_m)
        else:
            expected = expected_pairs[tile_id]
        simulated = _simulate_wave_dispatch(
            tile_id, wave_offsets, wave_sizes, cum_wave_sizes, group_size_m
        )
        assert simulated == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")
def test_transformed_matmul_kernel_executes_on_gpu():
    pytest.importorskip("triton")
    transformer = AnnotationTransformer()
    base = Path("tests/computation/transform/examples")
    source = (base / "example_matmul.py").read_text()
    transformed = transformer.transform(source)
    if "torch.manual_seed(0)" in transformed:
        transformed = transformed.split("torch.manual_seed(0)", 1)[0]
    assert "triton_output" not in transformed

    filename = f"<generated_matmul_{hash(transformed) & 0xFFFFFFFF:x}>"
    linecache.cache[filename] = (
        len(transformed),
        None,
        [line + "\n" for line in transformed.splitlines()],
        filename,
    )
    module_globals: dict = {"__name__": "__generated_matmul__", "__file__": filename}
    exec(compile(transformed, filename, "exec"), module_globals)

    torch_mod = module_globals["torch"]
    triton_mod = module_globals["triton"]
    matmul = module_globals["matmul"]
    device = module_globals["DEVICE"]
    is_hip_cdna2 = module_globals["is_hip_cdna2"]

    if not torch_mod.cuda.is_available():
        pytest.skip("CUDA device is required to run the transformed kernel")

    M = N = K = 512
    torch_mod.manual_seed(0)
    a = torch_mod.randn((M, K), device=device, dtype=torch_mod.float16)
    b = torch_mod.randn((K, N), device=device, dtype=torch_mod.float16)

    cases = [
        {
            "offsets": [[0, 0]],
            "sizes": [[M, N]],
        },
        {
            "offsets": [[M//2, 0], [0, 0]],
            "sizes": [[M//2, N], [M//2, N]],
        },
    ]

    for case in cases:
        offsets = torch_mod.tensor(case["offsets"], device=device, dtype=torch_mod.int32)
        sizes = torch_mod.tensor(case["sizes"], device=device, dtype=torch_mod.int32)
        cum = torch_mod.cumsum(
            sizes[:, 0] * sizes[:, 1], dim=0, dtype=torch_mod.int32
        )
        num_waves = offsets.shape[0]

        out = matmul(
            a,
            b,
            wave_offsets=offsets,
            wave_sizes=sizes,
            cum_wave_sizes=cum,
            NUM_WAVES=num_waves,
        )
        ref = torch_mod.matmul(a, b)
        rtol = 1e-2 if is_hip_cdna2() else 0
        assert torch_mod.allclose(out, ref, atol=1e-2, rtol=rtol)

if __name__ == "__main__":
    # test_example_matmul_injects_wave_dispatch_block()
    # test_example_attn_injects_wave_dispatch_with_range_axis()
    # test_example_attn_consumer_mode_injects_signal_wait()
    # test_example_local_kernel_transformation_matches_expected()
    # test_persistent_gemm_injects_persistent_blocks()
    test_persistent_gemm_producer_injects_epilogue()
    test_persistent_gemm_dual_role_uses_distinct_signals()
