from syncopate.communication.common_descriptors.all_gather import build_all_gather_plan_1d_swizzle
from syncopate.communication.common_descriptors.all_to_all import build_all_to_all_plan
from syncopate.communication.common_descriptors.utils import row_major_strides
from syncopate.communication.descriptor import (
    BufferRegion,
    CommPlan,
    DevicePlan,
    SignalType,
    Transfer,
    TransferOp,
)
from syncopate.interface import lower_comm_plan_to_raw_schedules
import torch

def test_lowering_all_gather_pull_generates_expected_tiles():
    plan = build_all_gather_plan_1d_swizzle(
        shape=(16, 2),
        dtype=torch.float16,
        axis=0,
        mesh_size=4,
        rank=2,
        buffer_name="buf",
        transfer_kind="pull",
    )
    plan.tensors_involved["buf"] = ((16, 2), None)

    schedules = lower_comm_plan_to_raw_schedules(CommPlan({2: plan}))
    assert 2 in schedules and "buf" in schedules[2]

    schedule = schedules[2]["buf"]
    assert schedule.num_waves == 4
    assert schedule.global_shape == (16, 2)
    assert all(len(wave) == 1 for wave in schedule.block_infos)

    offsets = [wave[0].tile.offsets for wave in schedule.block_infos]
    assert offsets == [(8, 0), (12, 0), (0, 0), (4, 0)]
    shapes = [wave[0].tile.shape for wave in schedule.block_infos]
    assert all(shape == (4, 2) for shape in shapes)


def test_lowering_all_gather_push_respects_shape_hint():
    plan = build_all_gather_plan_1d_swizzle(
        shape=(16, 2),
        dtype=torch.float16,
        axis=0,
        mesh_size=4,
        rank=1,
        buffer_name="buf",
        transfer_kind="push",
    )
    plan.tensors_involved["buf"] = ((16, 2), None)

    schedules = lower_comm_plan_to_raw_schedules(CommPlan({1: plan}))
    schedule = schedules[1]["buf"]
    assert schedule.num_waves == 1
    assert schedule.global_shape == (16, 2)

    offsets = [wave[0].tile.offsets for wave in schedule.block_infos]
    assert offsets == [(4, 0)]


def test_lowering_push_consumer_uses_earliest_transfer_time():
    strides = row_major_strides((4,))
    region = BufferRegion.from_slices(elem_size=1, strides=strides, slices=[(0, 4)])

    plan = DevicePlan(dev=0)
    plan.add_op(
        Transfer(
            op=TransferOp.PUSH,
            src_buf="buf",
            src_region=region,
            dst_buf="remote",
            dst_region=region,
            peer=1,
            chunk_idx=0,
        ),
        stream_idx=0,
    )
    plan.add_op(
        Transfer(
            op=TransferOp.PUSH,
            src_buf="buf",
            src_region=region,
            dst_buf="remote",
            dst_region=region,
            peer=2,
            chunk_idx=1,
        ),
        stream_idx=0,
    )
    plan.tensors_involved["buf"] = ((4,), torch.int8)

    schedules = lower_comm_plan_to_raw_schedules(CommPlan({0: plan}))
    schedule = schedules[0]["buf"]

    assert schedule.num_waves == 1
    # Only the earliest consumer touch should appear despite two pushes.
    assert len(schedule.block_infos[0]) == 1
    assert schedule.block_infos[0][0].tile.offsets == (0,)


def test_lowering_all_to_all_compute_producer_uses_compute_signals():
    mesh_size = 4
    shape = (16, 8)
    device_plans = {
        rank: build_all_to_all_plan(
            shape=shape,
            dtype=torch.float16,
            mesh_size=mesh_size,
            rank=rank,
            src_buffer="src",
            dst_buffer="dst",
            transfer_kind="push",
            compute_producer=True,
        )
        for rank in range(mesh_size)
    }

    comm_plan = CommPlan(device_plans)
    comm_plan.plan_signals()
    schedules = lower_comm_plan_to_raw_schedules(comm_plan)

    src_schedule = schedules[0]["src"]
    signals = [block.signal for wave in src_schedule.block_infos for block in wave]
    assert all(signal is not None for signal in signals)
    assert {signal.type for signal in signals} == {SignalType.COMPUTE_DONE}


def test_lowering_combines_parallel_streams_into_single_wave():
    strides = row_major_strides((4, 4))
    region_a = BufferRegion.from_slices(elem_size=1, strides=strides, slices=[(0, 2), (0, 4)])
    region_b = BufferRegion.from_slices(elem_size=1, strides=strides, slices=[(2, 4), (0, 4)])

    plan = DevicePlan(dev=0)
    plan.add_op(
        Transfer(
            op=TransferOp.PULL,
            src_buf="remote_a",
            src_region=region_a,
            dst_buf="buf",
            dst_region=region_a,
            peer=1,
            chunk_idx=0,
        ),
        stream_idx=0,
    )
    plan.add_op(
        Transfer(
            op=TransferOp.PULL,
            src_buf="remote_b",
            src_region=region_b,
            dst_buf="buf",
            dst_region=region_b,
            peer=2,
            chunk_idx=1,
        ),
        stream_idx=1,
    )
    plan.tensors_involved["buf"] = ((4, 4), None)

    schedules = lower_comm_plan_to_raw_schedules(CommPlan({0: plan}))
    schedule = schedules[0]["buf"]

    assert schedule.num_waves == 1
    assert len(schedule.block_infos[0]) == 2
    assert schedule.global_shape == (4, 4)


def test_lowering_attaches_signals():
    plan = build_all_gather_plan_1d_swizzle(
        shape=(16, 2),
        dtype=torch.float16,
        axis=0,
        mesh_size=4,
        rank=2,
        buffer_name="buf",
        transfer_kind="pull",
    )
    plan.tensors_involved["buf"] = ((16, 2), None)

    comm_plan = CommPlan({2: plan})
    comm_plan.plan_signals()

    schedules = lower_comm_plan_to_raw_schedules(comm_plan)
    schedule = schedules[2]["buf"]

    # Initial local shard should not have a signal.
    assert schedule.block_infos[0][0].signal == None
    # Subsequent waves correspond to remote transfers and should carry receive signals.
    for wave in schedule.block_infos[1:]:
        signals = wave[0].signal
        assert signals.type == SignalType.RECEIVE_DATA
        assert signals.dst_rank == 2

def test_all_to_all_waves():
    M = 16
    N = 8
    dtype = torch.float16
    WORLD_SIZE = 4
    device_plans = {
        rank: build_all_to_all_plan(
            shape=(M, N),
            dtype=dtype,
            mesh_size=WORLD_SIZE,
            rank=rank,
            src_buffer="src",
            dst_buffer="dst",
            transfer_kind="push",
            compute_producer=True,
        )
        for rank in range(WORLD_SIZE)
    }
    comm_plan = CommPlan(device_plans)
    comm_plan.plan_signals()
    schedules = lower_comm_plan_to_raw_schedules(comm_plan)

    for rank in range(WORLD_SIZE):
        print(f"Rank {rank} schedule:")
        src_schedule = schedules[rank]["src"]
        dst_schedule = schedules[rank]["dst"]
        print("  Src schedule:")
        for wave_idx, wave in enumerate(src_schedule.block_infos):
            print(f"    Wave {wave_idx}:")
            for block_info in wave:
                print(f"      Offsets: {block_info.tile.offsets}, Shape: {block_info.tile.shape}, Signal: {block_info.signal}")
        print("  Dst schedule:")
        for wave_idx, wave in enumerate(dst_schedule.block_infos):
            print(f"    Wave {wave_idx}:")
            for block_info in wave:
                print(f"      Offsets: {block_info.tile.offsets}, Shape: {block_info.tile.shape}, Signal: {block_info.signal}")

if __name__ == "__main__":
    test_all_to_all_waves()
