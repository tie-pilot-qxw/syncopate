# Experiments

This directory reorganizes experiment entrypoints according to the figures in the paper, so reviewers can more quickly find the corresponding scripts.

Current figure directories:

- `fig_operator_results/`
- `fig_attention_results/`
- `fig_ir_example/`
- `fig_ablation_sensitivity/`

Some baselines used in the paper are not currently vendored into this repository in a clean, self-contained form. For those baselines, please retrieve the corresponding upstream repositories and run the matching experiment scripts there.

This applies to baselines such as:

- [ThunderKittens](https://github.com/HazyResearch/ThunderKittens)
- [TritonDistributed](https://github.com/ByteDance-Seed/Triton-distributed)
- [Flux](https://github.com/bytedance/flux)
- [Mercury](https://github.com/ChandlerGuan/mercury_artifact)
- [YunChang / long-context-attention](https://github.com/feifeibear/long-context-attention)

The original experiments were run with upstream versions from roughly late November 2025. We cannot currently provide an exact commit hash for every missing baseline, because the original Docker environment used for those runs is no longer available.

If exact baseline reproduction is required, the practical approach is:

1. Start from the upstream repository version closest to late November 2025.
2. Identify the script matching the figure and operator in the paper.
3. Use the same hardware/software stack described in the paper when possible.
