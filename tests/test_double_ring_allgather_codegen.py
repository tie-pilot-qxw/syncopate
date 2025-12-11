from __future__ import annotations

import torch

from syncopate.communication.code_gen import CommGenerator
from syncopate.communication.comm_runtime.operations import CopyEngineOp, IntraNodeSignal, WaitIntraNodeSignal
from syncopate.communication.common_descriptors import build_double_ring_all_gather_plan
from syncopate.communication.descriptor import TransferOp


def test_double_ring_all_gather_codegen_print():
    mesh_size = 9
    inner_size = 3
    shape = (18, 2)
    dtype = torch.float16

    device_plans = {
        rank: build_double_ring_all_gather_plan(
            shape=shape,
            dtype=dtype,
            axis=0,
            mesh_size=mesh_size,
            inner_mesh_size=inner_size,
            rank=rank,
            buffer_name="buf",
        )
        for rank in range(mesh_size)
    }

    for rank, plan in device_plans.items():
        print(f"Device Plan for rank {rank}:\n{plan.pretty()}")

    generator = CommGenerator(device_plans)
    generator.plan_signals()
    comm_info = generator.generate_code_for_plan()
    comm_info.local_world_size = comm_info.world_size

    print("\nGenerated Communication Info:")
    print(comm_info)

    num_groups = mesh_size // inner_size
    total_pushes_per_rank = num_groups * inner_size - 1  # last outer step skips the outer send

    for rank in range(mesh_size):
        pushes = device_plans[rank].xfers
        assert len(pushes) == total_pushes_per_rank
        assert all(xf.op == TransferOp.PUSH for xf in pushes)

        ops = comm_info.operations[rank][0]
        waits = [op for op in ops if isinstance(op, WaitIntraNodeSignal)]
        copies = [op for op in ops if isinstance(op, CopyEngineOp)]
        signals = [op for op in ops if isinstance(op, IntraNodeSignal)]

        # Every push except the very first has a dependency.
        expected_waits = total_pushes_per_rank - 1
        assert len(waits) == expected_waits
        assert len(copies) == total_pushes_per_rank
        assert len(signals) == total_pushes_per_rank

        # Every wait should be immediately followed by a copy.
        for idx, op in enumerate(ops):
            if isinstance(op, WaitIntraNodeSignal):
                assert isinstance(ops[idx + 1], CopyEngineOp)


if __name__ == "__main__":
    test_double_ring_all_gather_codegen_print()
