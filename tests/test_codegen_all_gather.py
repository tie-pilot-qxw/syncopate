from __future__ import annotations

from syncopate.communication.code_gen import CodeGenOptions, CommGenerator
from syncopate.communication.comm_runtime.operations import (
    CopyEngineOp,
    IntraNodeSignal,
)
from syncopate.communication.common_descriptors import build_all_gather_plan_1d_swizzle
from syncopate.communication.descriptor import SignalType
from syncopate.interface.lowering import lower_comm_plan_to_raw_schedules
from syncopate.interface.tile_schedule import SimplifiedSchedule
import torch

def test_all_gather_codegen_generates_comm_info():
    mesh_size = 4
    shape = (100,20)
    dtype = torch.float16
    axis = 0
    buffer_name = "buf"

    device_plans = {
        rank: build_all_gather_plan_1d_swizzle(
            shape=shape,
            dtype=dtype,
            axis=axis,
            mesh_size=mesh_size,
            rank=rank,
            buffer_name=buffer_name,
            transfer_kind="pull",
        )
        for rank in range(mesh_size)
    }

    generator = CommGenerator(device_plans)
    generator.plan_signals()
    raw_schedules = lower_comm_plan_to_raw_schedules(generator)

    comm_info = generator.generate_code_for_plan()
    comm_info.local_world_size = comm_info.world_size  # for testing purpose, assume intra-node only

    assert comm_info.recv_signal_num == mesh_size - 1
    assert comm_info.transit_signal_num == 0
    assert len(comm_info.operations) == mesh_size

    # print(f"Generated CommInfo: {comm_info}")
    # print("Generated Raw Schedules:")
    # for rank, schedules in raw_schedules.items():
    #     print(f"Rank {rank}:")
    #     for buf_name, schedule in schedules.items():
    #         try_simplify = SimplifiedSchedule(schedule)
    #         try_simplify.simplify()
    #         assert try_simplify.simplified_schedule is not None
    #         print(f"  Buffer {buf_name}, Schedule: {schedule}, Simplified: {try_simplify.simplified_schedule}")
    #         print(f"  Shape: {schedule.gen_block_shape_lists()}, Offsets: {schedule.gen_block_offset_lists()}, Signals: {schedule.gen_signal_lists()}")


def test_all_gather_codegen_sm():
    mesh_size = 4
    shape = (100,20)
    dtype = torch.float16
    axis = 0
    buffer_name = "buf"

    device_plans = {
        rank: build_all_gather_plan_1d_swizzle(
            shape=shape,
            dtype=dtype,
            axis=axis,
            mesh_size=mesh_size,
            rank=rank,
            buffer_name=buffer_name,
            transfer_kind="pull",
        )
        for rank in range(mesh_size)
    }

    options = CodeGenOptions(copy_engine=False)
    generator = CommGenerator(device_plans)
    generator.plan_signals()

    comm_info = generator.generate_code_for_plan(options=options)
    comm_info.local_world_size = comm_info.world_size
    print(f"Generated CommInfo with SM codegen: {comm_info}")


if __name__ == "__main__":
    test_all_gather_codegen_sm()