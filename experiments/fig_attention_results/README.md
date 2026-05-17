# Figure: Attention Results

This directory organizes the scripts corresponding to the attention operator-results figure by legend/system name.

## Current baselines in this directory

- `pytorch/all2all_attn.py`
- `pytorch/attn_all2all.py`
- `triton_nccl/all2all_attn_tlx.py`
- `triton_nccl/attn_all2all_tlx.py`

The `_tlx` variants correspond to the Triton+NCCL baseline rather than the plain PyTorch baseline, so they are grouped separately under `triton_nccl/`.

## Ours

The Syncopate scripts are intentionally kept under `tests/` for now, because moving them may break relative imports, generated-file paths, or other assumptions in the current experiment code.

Current Syncopate entries for this figure:

- `HP-Attn`: `tests/test_all2all_attn_transform.py`
- `Attn-A2A`: `tests/test_attn_all2all_transform.py`
- `Ring-Attn`: `tests/test_allgather_attn_transform.py`

## External baseline note

The PyTorch Ring-Attention baseline used for the paper figure comes from the YunChang repository:

- [YunChang / long-context-attention](https://github.com/feifeibear/long-context-attention)

That repository provides the `yunchang` implementation of unified / hybrid sequence parallel attention and includes the Ring-Attention-related code path used as the external baseline reference for this figure.

Other external baseline repositories referenced by the paper are listed in `experiments/README.md`.
