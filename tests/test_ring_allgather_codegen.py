from __future__ import annotations

import torch

from syncopate.communication.code_gen import CommGenerator
from syncopate.communication.comm_runtime.operations import CopyEngineOp, IntraNodeSignal, WaitIntraNodeSignal
from syncopate.communication.common_descriptors import build_ring_all_gather_plan
from syncopate.communication.descriptor import SignalType, TransferOp


def test_ring_all_gather_codegen_waits_and_signals():
    mesh_size = 4
    shape = (8, 2)
    dtype = torch.float16
    axis = 0
    buffer_name = "buf"

    device_plans = {
        rank: build_ring_all_gather_plan(
            shape=shape,
            dtype=dtype,
            axis=axis,
            mesh_size=mesh_size,
            rank=rank,
            buffer_name=buffer_name,
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
    # Each rank has mesh_size-1 pushes on stream 0.
    for rank in range(mesh_size):
        pushes = device_plans[rank].xfers
        assert len(pushes) == mesh_size - 1
        assert all(xf.op == TransferOp.PUSH for xf in pushes)

        ops = comm_info.operations[rank][0]
        waits = [op for op in ops if isinstance(op, WaitIntraNodeSignal)]
        copies = [op for op in ops if isinstance(op, CopyEngineOp)]
        signals = [op for op in ops if isinstance(op, IntraNodeSignal)]

        # First push has no dependency; the rest each wait on upstream data.
        assert len(waits) == mesh_size - 2
        assert len(copies) == mesh_size - 1
        assert len(signals) == mesh_size - 1
        assert all(wait.signal_type == SignalType.RECEIVE_DATA for wait in waits)

        # Operations should begin with a copy (no wait) then repeat wait->copy->signal.
        assert isinstance(ops[0], CopyEngineOp)
        assert isinstance(ops[1], IntraNodeSignal)
        for idx in range(2, len(ops), 3):
            assert isinstance(ops[idx], WaitIntraNodeSignal)
            assert isinstance(ops[idx + 1], CopyEngineOp)
            assert isinstance(ops[idx + 2], IntraNodeSignal)

if __name__ == "__main__":
    test_ring_all_gather_codegen_waits_and_signals()