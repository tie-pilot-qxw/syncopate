import torch

from syncopate.communication.common_descriptors import build_all_to_all_plan
from syncopate.communication.descriptor import CommPlan, ComputeDependency, SignalType


def _build_comm_plan(transfer_kind: str, mesh_size: int = 4) -> tuple[int, CommPlan]:
    device_plans = {
        rank: build_all_to_all_plan(
            shape=(4, 4),
            dtype=torch.float32,
            mesh_size=mesh_size,
            rank=rank,
            src_buffer="src",
            dst_buffer="dst",
            transfer_kind=transfer_kind,
            compute_producer=True,
        )
        for rank in range(mesh_size)
    }
    return mesh_size, CommPlan(device_plans)


def _collect_compute_dependencies(comm_plan: CommPlan):
    for dev, plan in comm_plan.device_plans.items():
        for stream in plan.xfer:
            for op in stream:
                for dep in op.dependency:
                    if isinstance(dep, ComputeDependency):
                        yield dev, dep

def _compute_deps_by_device(comm_plan: CommPlan) -> dict[int, list[ComputeDependency]]:
    deps: dict[int, list[ComputeDependency]] = {}
    for dev, dep in _collect_compute_dependencies(comm_plan):
        deps.setdefault(dev, []).append(dep)
    return deps


def test_all_to_all_push_plan():
    mesh_size, comm_plan = _build_comm_plan("push")

    # All compute dependencies should start without signals so planning can populate them.
    for _, dep in _collect_compute_dependencies(comm_plan):
        assert dep.signal is None

    comm_plan.plan_signals(plan_compute_signals=True)

    # Every push depends on a locally produced compute signal with unique offsets.
    deps_by_dev = _compute_deps_by_device(comm_plan)
    for dev in comm_plan.device_plans:
        offsets = set()
        for dep in deps_by_dev.get(dev, []):
            assert dep.signal is not None
            assert dep.signal.type == SignalType.COMPUTE_DONE
            assert dep.signal.dst_rank == dev
            offsets.add(dep.signal.dst_offset)
        assert len(offsets) == mesh_size
        assert offsets == set(range(mesh_size))

    print(comm_plan.visualize_signals())


def test_all_to_all_pull_plan():
    mesh_size, comm_plan = _build_comm_plan("pull")

    comm_plan.plan_signals(plan_compute_signals=True)

    # Pull transfers depend on remote compute, but the signals should still target the local rank.
    deps_by_dev = _compute_deps_by_device(comm_plan)
    for dev, deps in deps_by_dev.items():
        for dep in deps:
            assert dep.signal is not None
            assert dep.signal.type == SignalType.COMPUTE_DONE

    # Each rank should get a unique offset per dependency; no transit signals are introduced.
    for dev, plan in comm_plan.device_plans.items():
        offsets = {dep.signal.dst_offset for dep in deps_by_dev.get(dev, []) if dep.signal is not None}
        assert len(offsets) == mesh_size
        assert plan.transit_signals == []

    print(comm_plan.visualize_signals())
    
if __name__ == "__main__":
    test_all_to_all_push_plan()
    test_all_to_all_pull_plan()