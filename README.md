# Syncopate

Experimental compiler for fine-grained communication/computation overlap.

## Dependencies
- Python 3.10+
- PyTorch
- Triton Distributed: install from https://github.com/ByteDance-Seed/Triton-distributed (this repository expects the custom merged variant). Without it, attention-related code/tests will fail.

## Install
- `pip install -e .`

## Quick communication plan example
Build a 1D all-gather plan (pull-based), lower it, and generate runtime code:

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
        shape=shape, dtype=dtype, axis=0, mesh_size=mesh, rank=rank, buffer_name="a", transfer_kind="pull"
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

See `tests/test_comm_plan.py` for more patterns (ring, transit signals) and `syncopate/communication/common_descriptors/*` for other collectives.

## Fused all-gather + GEMM (outline)
The test `tests/test_ag_gemm_gen.py` demonstrates fusing an all-gather with a GEMM:
1) Transform the annotated local GEMM kernel into a consumer (`AnnotationTransformer(enable_consumer=True, consumer_descriptors=("a_desc",))`).
2) Build per-rank plans with `build_all_gather_plan_1d_swizzle(...)`, wrap them in `CommGenerator`, and call `plan_signals()` + `generate_code_for_plan()`.
3) Lower to schedules (`lower_comm_plan_to_raw_schedules`) to get `block_offsets/block_shapes/signal_offsets` for the fused consumer kernel args.
4) Create a runtime (`CommContext(rank, comm_info)`), start/execute/stop it around the consumer GEMM launch:

```python
comm_runtime = CommContext(rank, comm_info)
comm_runtime.start_after(torch.cuda.current_stream())
comm_runtime.execute()
generated_gemm_consumer(
    comm_runtime.comm_buffers["a"][rank],  # gathered A buffer
    b, c, comm_info.world_size,
    num_gemm_sms=132,
    BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
    cur_wave_sizes=block_shapes,
    wave_offsets=block_offsets,
    signal_offsets=signal_offsets,
    signal_ptr=comm_runtime.recv_signal_bufs[rank],
    cum_tiles=cum_tiles,
)
comm_runtime.end_before(torch.cuda.current_stream())
```

