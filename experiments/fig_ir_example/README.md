# Figure: IR Example

This directory organizes the scripts corresponding to the IR-lowering / compiler-integration figure.

## Current baselines in this directory

- `domino/domino_alike_mlp_bench.py`
- `alpa/possible_pattern.py`

## Ours

The Syncopate scripts are intentionally kept under `tests/` for now, because moving them may break relative imports, generated-file paths, or other assumptions in the current experiment code.

Current Syncopate entries for this figure:

- `tests/test_domino_alike_mlp_collective_nvsharp.py`
- `tests/test_gemm_all2all_gemm.py`
- `tests/test_ring_allgather_attn_transform.py`
- `tests/test_double_ring_ag_attn.py`

Mercury was used in the paper figure but is not currently vendored into this repository in a clean, self-contained form. See `experiments/README.md` for the upstream-baseline repositories and links.
