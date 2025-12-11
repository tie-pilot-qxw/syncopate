import torch

from syncopate.communication.code_gen import CodeGenOptions, CommGenerator
from syncopate.communication.comm_runtime.operations import CollectiveComm, IntraNodeSignal, WaitIntraNodeSignal
from syncopate.communication.common_descriptors import build_reduce_scatter_collective_plan
from syncopate.communication.common_descriptors.reduce_scatter import build_reduce_scatter_direct_reduce_plan
from syncopate.communication.descriptor import AxisSlice, CollectiveOp, SignalType
from syncopate.interface.lowering import lower_comm_plan_to_raw_schedules


def test_reduce_scatter_plan_to_comm_info():
    mesh_size = 2
    num_rs = 2
    shape = (4, 4)
    axis = 0
    split_axis = 1

    device_plans = {
        rank: build_reduce_scatter_collective_plan(
            shape=shape,
            dtype=torch.float32,
            axis=axis,
            split_axis=split_axis,
            mesh_size=mesh_size,
            rank=rank,
            src_buffer="src",
            dst_buffer="dst",
            num_reduce_scatters=num_rs,
        )
        for rank in range(mesh_size)
    }

    for rank, plan in device_plans.items():
        print(f"Device Plan for rank {rank}:\n{plan.pretty()}\n")

    generator = CommGenerator(device_plans)
    generator.plan_signals()
    comm_info = generator.generate_code_for_plan()

    assert comm_info.world_size == mesh_size
    # Each rank has num_rs receive signals; counter tracks max offset.
    assert comm_info.recv_signal_num == num_rs

    split_chunk = shape[split_axis] // num_rs
    reduce_len = shape[axis] // mesh_size

    for rank in range(mesh_size):
        streams = comm_info.operations[rank]
        assert len(streams) == 1
        ops = streams[0]
        # Expect alternating collective and signal ops per split.
        assert len(ops) == num_rs * 2
        for idx in range(num_rs):
            coll = ops[idx * 2]
            sig = ops[idx * 2 + 1]
            assert isinstance(coll, CollectiveComm)
            assert coll.op == CollectiveOp.REDUCE_SCATTER
            assert coll.src == "src"
            assert coll.dst == "dst"
            assert isinstance(coll.src_slice, tuple) and isinstance(coll.dst_slice, tuple)
            assert coll.src_slice[axis] == AxisSlice(0, shape[axis])
            assert coll.src_slice[split_axis] == AxisSlice(idx * split_chunk, (idx + 1) * split_chunk)
            assert coll.dst_slice[axis] == AxisSlice(0, reduce_len)
            assert coll.dst_slice[split_axis] == AxisSlice(idx * split_chunk, (idx + 1) * split_chunk)

            assert isinstance(sig, IntraNodeSignal)
            assert sig.offset == idx
            assert sig.rank == rank

    print("CommInfo:")
    print(comm_info)


def test_reduce_scatter_lowering_produces_dst_schedule():
    mesh_size = 2
    num_rs = 2
    shape = (4, 4)
    axis = 0
    split_axis = 1

    device_plans = {
        rank: build_reduce_scatter_collective_plan(
            shape=shape,
            dtype=torch.float32,
            axis=axis,
            split_axis=split_axis,
            mesh_size=mesh_size,
            rank=rank,
            src_buffer="src",
            dst_buffer="dst",
            num_reduce_scatters=num_rs,
        )
        for rank in range(mesh_size)
    }

    generator = CommGenerator(device_plans)
    generator.plan_signals()
    lowered = lower_comm_plan_to_raw_schedules(generator)

    dst_shape = (shape[axis] // mesh_size, shape[split_axis])
    for rank in range(mesh_size):
        assert rank in lowered
        assert "dst" in lowered[rank]
        schedule = lowered[rank]["dst"]
        assert schedule.global_shape == dst_shape
        # Two waves (one per split), each with a single tile covering the chunk.
        assert schedule.num_waves == num_rs
        assert all(len(wave) == 1 for wave in schedule.block_infos)


def test_reduce_scatter_plan_with_compute_dependency():
    mesh_size = 2
    shape = (4, 4)
    axis = 0
    split_axis = 1

    device_plans = {
        rank: build_reduce_scatter_collective_plan(
            shape=shape,
            dtype=torch.float32,
            axis=axis,
            split_axis=split_axis,
            mesh_size=mesh_size,
            rank=rank,
            src_buffer="src",
            dst_buffer="dst",
            compute_producer=True,
        )
        for rank in range(mesh_size)
    }

    generator = CommGenerator(device_plans)
    generator.plan_signals()
    comm_info = generator.generate_code_for_plan()

    assert comm_info.compute_signal_num == 1
    for rank in range(mesh_size):
        ops = comm_info.operations[rank][0]
        assert isinstance(ops[0], WaitIntraNodeSignal)
        assert ops[0].signal_type == SignalType.COMPUTE_DONE
        assert isinstance(ops[1], CollectiveComm)
        assert isinstance(ops[2], IntraNodeSignal)
    print(comm_info)

def test_reduce_scatter_direct():
    mesh_size = 4
    shape = (8, 8)
    axis = 0
    
    device_plans = {
        rank: build_reduce_scatter_direct_reduce_plan(
            shape=shape,
            dtype=torch.float32,
            axis=axis,
            mesh_size=mesh_size,
            rank=rank,
            src_buffer="src",
            dst_buffer="dst",
        )
        for rank in range(mesh_size)
    }

    for rank, plan in device_plans.items():
        print(f"Device Plan for rank {rank}:\n{plan.pretty()}\n")

    generator = CommGenerator(device_plans)
    generator.plan_signals()
    options = CodeGenOptions(copy_engine=False)
    comm_info = generator.generate_code_for_plan(options=options)

    print("CommInfo:")
    print(comm_info)

if __name__ == "__main__":
    test_reduce_scatter_plan_with_compute_dependency()
    test_reduce_scatter_direct()
