# Figure: Operator Results

This directory organizes the scripts corresponding to the GEMM operator-results figure by legend/system name.

## Current baselines in this directory

- `pytorch/ag_gemm.py`
- `pytorch/gemm_rs.py`
- `pytorch/gemm_ar.py`
- `async_tp/ag_gemm.py`
- `async_tp/gemm_rs.py`

## Ours

The Syncopate scripts are kept under `tests/` for now, because moving them may break relative imports, generated-file paths, or other assumptions in the current experiment code.

Current Syncopate entries for this figure:

- `tests/test_ag_gemm_gen.py`
- `syncopate/computation/gemm/template_rs.py`
- `tests/test_gemm_ar_collective.py`

## GEMM-RS note

For GEMM-RS, the result used in the paper figure originally corresponds to `syncopate/computation/gemm/template_rs.py`. At the time of the paper experiments, the full compiler path for this intra-kernel variant was not yet fully integrated, so we used this hand-written transformed implementation to approximate the intended generated execution pattern.

The repository now also contains a more complete compiler-oriented path in `tests/test_gemm_rs_direct_reduce.py`. In addition, `tests/test_gemm_rs_transform.py` was already supported at the time and uses the copy-engine-based path; in practice its measured performance is also very close to the paper result.

These implementations follow the same high-level overlap strategy and typically achieve very similar performance, but small differences may still appear due to implementation-level differences in the current codebase.

Therefore, for strict correspondence to the paper figure, `syncopate/computation/gemm/template_rs.py` should be treated as the original source of the reported GEMM-RS result, while `tests/test_gemm_rs_transform.py` and `tests/test_gemm_rs_direct_reduce.py` serve as the current runnable references in this repository.
