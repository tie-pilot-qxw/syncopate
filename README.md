# Syncopate

Experimental compiler for fine-grained communication/computation overlap.

The instructions below are intended to
let an evaluator verify that the code is retrievable and that the core compiler
passes are runnable. Reproducing the full set of paper numbers requires
specialized multi-GPU hardware (see [Detailed Instructions](#detailed-instructions)).

## Paper Reference

**[Syncopate: Efficient Multi-GPU AI Kernels via Automatic Chunk-Centric Compute-Communication Overlap](https://www.usenix.org/conference/osdi26/presentation/qiang)**  
Xinwei Qiang, Yue Guan, Zhengding Hu, Keren Zhou, Yufei Ding, Adnan Aziz.  
In *Proceedings of the 20th USENIX Symposium on Operating Systems Design and Implementation (OSDI '26)*, 2026.
```bibtex
@inproceedings {syncopate,
author = {Xinwei Qiang and Yue Guan and Zhengding Hu and Keren Zhou and Yufei Ding and Adnan Aziz},
title = {Syncopate: Efficient {Multi-GPU} {AI} Kernels via Automatic {Chunk-Centric} {Compute-Communication} Overlap},
booktitle = {20th USENIX Symposium on Operating Systems Design and Implementation (OSDI 26)},
year = {2026},
isbn = {978-1-939133-55-7},
address = {Seattle, WA},
pages = {331--347},
url = {https://www.usenix.org/conference/osdi26/presentation/qiang},
publisher = {USENIX Association},
month = jul
}
```

## Artifact Evaluation Version

The fixed version submitted for OSDI '26 Artifact Evaluation is:

- Git commit: `ce2b13e496eb6629ab42aac2aec1d9e31e084ba6`

AEC reviewers should use this commit as the evaluated version of the artifact.

---

## Getting Started Instructions

These steps verify that the package installs and that the CPU-only compiler
passes (planning, lowering, tile scheduling) work end-to-end. They do **not**
require a GPU and should finish in under five minutes.

### 1. Install

```bash
git clone https://github.com/tie-pilot-qxw/syncopate.git syncopate
cd syncopate
pip install -e .
```

The base install needs only Python ≥ 3.10, PyTorch, and pytest. The full set
of GPU and distributed dependencies (Triton-distributed, NVSHMEM, CUDA toolkit)
is only required for the experiments described in
[Detailed Instructions](#detailed-instructions).

### 2. Run the CPU-only test suite

```bash
pytest tests/test_comm_plan.py \
       tests/test_descriptor.py \
       tests/test_tile_schedule.py \
       tests/test_plan_compute_signal.py \
       tests/test_allgather_schedule.py \
       tests/test_lowering.py
```

Expected output: 45 tests passing in roughly one second. These tests cover
the communication-plan descriptors, signal computation, tile schedule
simplification, and the lowering from communication plans to per-rank
schedules — i.e. the parts of the compiler that do not depend on a GPU
backend.

### 3. Hello-world: build and lower an all-gather plan

The following snippet builds a 1D all-gather plan, plans signals, and lowers
it to per-rank raw schedules. It runs on CPU.

```python
import torch
from syncopate.communication.common_descriptors import build_all_gather_plan_1d_swizzle
from syncopate.communication.code_gen import CommGenerator
from syncopate.interface.lowering import lower_comm_plan_to_raw_schedules

mesh = 4
shape = (1024, 512)  # per-rank logical shape along the gathered axis
dtype = torch.float16
device_plans = {
    rank: build_all_gather_plan_1d_swizzle(
        shape=shape, dtype=dtype, axis=0, mesh_size=mesh,
        rank=rank, buffer_name="a", transfer_kind="pull",
    )
    for rank in range(mesh)
}

plan = CommGenerator(device_plans)
plan.plan_signals()
print(plan.visualize_signals())  # human-readable signal graph

comm_info = plan.generate_code_for_plan()
schedules = lower_comm_plan_to_raw_schedules(plan)
# schedules[rank]["a"] carries block offsets/sizes for kernel args
```

If this runs without error and prints a signal graph, the artifact is set up
correctly.

---

## Detailed Instructions

The remaining tests and benchmarks generate and run real GPU kernels and
require a multi-GPU machine with the full Triton-distributed / NVSHMEM stack
installed. We provide a `Dockerfile` that pins all of the dependencies.

Experiment entrypoints are additionally reorganized under `experiments/`
according to the paper figures. See `experiments/README.md` for the
figure-to-script mapping and notes about which external baselines are not
currently vendored in this repository.

### Hardware

The experiments in the paper were run on multi-GPU NVIDIA Hopper systems
(H100/H800-class) connected by NVLink/NVSwitch. Most tests assume a world
size of at least 4 GPUs on a single node; a few tests use 8.

The compiler frontend itself is hardware-independent and runs on CPU
(see Getting Started above).

### Build the Docker image

```bash
docker build -t syncopate:ae .
```

The image is built on top of `nvcr.io/nvidia/pytorch:25.06-py3` and clones
a pinned branch of Triton-distributed (with the [TLX](https://github.com/facebookexperimental/triton)
extension) into `/workspace/Triton-distributed`. Without TLX, the
attention-related tests will not work; the GEMM-related tests still do.

Run the container with GPUs exposed:

```bash
docker run --gpus all --shm-size=16g --network=host -it \
    -v "$(pwd)":/workspace/syncopate \
    syncopate:ae bash
# inside the container:
cd /workspace/syncopate && pip install -e .
```

### Launching multi-GPU tests

`utils/launch.sh` is a thin wrapper around `torchrun` copied from the Triton-distributed repository that sets the NVSHMEM bootstrap socket and a few other distributed env vars. A typical invocation:

```bash
bash utils/launch.sh --nproc-per-node=4 tests/test_ag_gemm_gen.py
```

Override `NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME` to the network interface on
your machine if the default does not exist.

### What is in the repository

| Path | Contents |
| --- | --- |
| `syncopate/communication/` | Communication-plan descriptors, code generator, runtime |
| `syncopate/computation/` | GEMM and attention kernel templates and the transform pass |
| `syncopate/interface/` | Lowering from plans to per-rank schedules and tile schedules |
| `tests/` | End-to-end tests, including fused all-gather + GEMM, all-to-all + attention, GEMM + reduce-scatter, and the CPU-only compiler-pass tests used by Getting Started |
| `experiments/` | Figure-oriented experiment entrypoints and per-figure README files that map paper results to in-repo scripts and external baselines |
| `utils/` | `launch.sh`, microbenchmarks (`benchmark_gemm_lookup.py`, `copy_engine_bandwidth.py`, `sm_bandwidth.py`, `sm_throughput.py`, `attn_flops.py`), and small helpers |

### Notable end-to-end tests

These exercise the full generate-and-run path on GPU and roughly correspond
to the workloads discussed in the paper:

- `tests/test_ag_gemm_gen.py` — fused all-gather + GEMM, generated path
- `tests/test_ag_gemm_hand.py` — fused all-gather + GEMM, hand-written reference
- `tests/test_gemm_rs_transform.py` — fused GEMM + reduce-scatter through the transform pass
- `tests/test_allgather_attn_transform.py` — fused all-gather + attention
- `tests/test_all2all_attn_transform.py` — all-to-all + attention (Ulysses-style)
- `tests/test_domino_alike_mlp_collective.py` — Domino-style MLP block

Each test prints kernel timing and a correctness check against an unfused
reference.

For figure-oriented reproduction, use `experiments/README.md` as the top-level
index:

- `experiments/fig_operator_results/` — GEMM operator-results figure
- `experiments/fig_attention_results/` — attention operator-results figure
- `experiments/fig_ir_example/` — IR-lowering / compiler-integration figure
- `experiments/fig_ablation_sensitivity/` — ablation and sensitivity figure

---

## License and dependencies

This repository depends on:

- [PyTorch](https://pytorch.org/)
- [Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed) — required for code generation; the attention path additionally needs the TLX-extended fork
- [NVSHMEM](https://developer.nvidia.com/nvshmem) — required at runtime for cross-rank transfers

All third-party code retains its original license; see the respective
upstream repositories.

This project is licensed under the MIT License. Please see the LICENSE file for details. We welcome the community to use, compare, and extend this artifact for research purposes.
