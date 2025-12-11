import torch

from syncopate.communication.code_gen import CommGenerator
from syncopate.communication.comm_runtime.operations import CollectiveComm, IntraNodeSignal, WaitIntraNodeSignal
from syncopate.communication.common_descriptors import build_all_reduce_plan
from syncopate.communication.descriptor import AxisSlice, CollectiveOp, SignalType


def test_all_reduce_plan_to_comm_info():
    mesh_size = 2
    num_ar = 2
    shape = (4, 4)
    split_axis = 1

    device_plans = {
        rank: build_all_reduce_plan(
            shape=shape,
            dtype=torch.float32,
            mesh_size=mesh_size,
            rank=rank,
            buffer_name="buf",
            num_all_reduces=num_ar,
            split_axis=split_axis,
        )
        for rank in range(mesh_size)
    }

    for rank in range(mesh_size):
        print(f"Device Plan for rank {rank}:\n{device_plans[rank].pretty()}\n")

    generator = CommGenerator(device_plans)
    generator.plan_signals()
    comm_info = generator.generate_code_for_plan()
    print(comm_info)
    assert comm_info.world_size == mesh_size
    assert comm_info.recv_signal_num == num_ar

    split_chunk = shape[split_axis] // num_ar

    for rank in range(mesh_size):
        streams = comm_info.operations[rank]
        assert len(streams) == 1
        ops = streams[0]
        assert len(ops) == num_ar * 2
        for idx in range(num_ar):
            coll = ops[idx * 2]
            sig = ops[idx * 2 + 1]
            assert isinstance(coll, CollectiveComm)
            assert coll.op == CollectiveOp.ALL_REDUCE
            assert coll.src == "buf"
            assert coll.dst == "buf"
            assert coll.src_slice[split_axis] == AxisSlice(idx * split_chunk, (idx + 1) * split_chunk)
            assert coll.dst_slice[split_axis] == AxisSlice(idx * split_chunk, (idx + 1) * split_chunk)
            for dim, dim_len in enumerate(shape):
                if dim == split_axis:
                    continue
                assert coll.src_slice[dim] == AxisSlice(0, dim_len)
                assert coll.dst_slice[dim] == AxisSlice(0, dim_len)

            assert isinstance(sig, IntraNodeSignal)
            assert sig.offset == idx
            assert sig.rank == rank


def test_all_reduce_plan_with_compute_dependency():
    mesh_size = 2
    shape = (4, 4)

    device_plans = {
        rank: build_all_reduce_plan(
            shape=shape,
            dtype=torch.float32,
            mesh_size=mesh_size,
            rank=rank,
            buffer_name="buf",
            compute_producer=True,
        )
        for rank in range(mesh_size)
    }
    for rank in range(mesh_size):
        print(f"Device Plan for rank {rank}:\n{device_plans[rank].pretty()}\n")

    generator = CommGenerator(device_plans)
    generator.plan_signals()
    comm_info = generator.generate_code_for_plan()
    print(comm_info)
    
    assert comm_info.compute_signal_num == 1
    for rank in range(mesh_size):
        ops = comm_info.operations[rank][0]
        assert isinstance(ops[0], WaitIntraNodeSignal)
        assert ops[0].signal_type == SignalType.COMPUTE_DONE
        assert isinstance(ops[1], CollectiveComm)
        assert isinstance(ops[2], IntraNodeSignal)

if __name__ == "__main__":
    test_all_reduce_plan_to_comm_info()
    test_all_reduce_plan_with_compute_dependency()