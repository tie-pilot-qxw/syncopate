# Figure: Ablation and Sensitivity

## (a) Communication Backend Selection

This panel corresponds to multiple implementation paths rather than a single script.

### GEMM-RS

- `copy engine`: `tests/test_gemm_rs_transform.py`
- `intra TMA`: `syncopate/computation/gemm/template_rs.py`
- `intra CUDA`: `syncopate/computation/gemm/template_rs.py`
- `inter TMA`: `tests/test_gemm_rs_sm.py`
- `inter CUDA`: `tests/test_gemm_rs_sm.py`

Notes:

- `template_rs.py` switches the intra-kernel backend through `use_tma`.
- `tests/test_gemm_rs_sm.py` switches the inter-SM backend through `use_tma`.
- The current compiler-oriented counterpart of `template_rs.py` is `tests/test_gemm_rs_direct_reduce.py`.

### AG-GEMM

- `copy engine`: `tests/test_ag_gemm_gen.py`
- `intra TMA`: `syncopate/computation/gemm/template_ag.py`
- `intra CUDA`: `syncopate/computation/gemm/template_ag.py`
- `inter TMA`: `tests/test_ag_gemm_with_sms.py`
- `inter CUDA`: `tests/test_ag_gemm_with_sms.py`

Notes:

- `template_ag.py` switches the intra-kernel backend through `use_tma_a`.
- `tests/test_ag_gemm_with_sms.py` switches the inter-SM backend through `use_tma`.
- The current compiler-oriented counterpart of `template_ag.py` is `tests/test_ag_gemm_direct_read.py`.

## (b) Chunk Split Tuning

Current in-repo runnable counterparts:

- `GEMM-AR`: `tests/test_gemm_ar_collective.py`
- `A2A-GEMM`: `tests/test_all2all_gemm_split_collective.py`

## (c) SM Number Tuning

- `tests/test_ag_gemm_with_sms.py`


## (d) Intra-Tile Scheduling

- `tests/test_ag_gemm_tile_size_sensitivity.py`
