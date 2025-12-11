import pytest
from syncopate.communication.common_descriptors import (
    build_all_gather_plan_1d_swizzle,
    build_ring_all_gather_plan,
    build_all_to_all_plan,
    build_all_to_all_plan_dim,
    build_all_to_all_plan_axis,
    build_all_to_all_plan_dim_waves,
)
from syncopate.communication.descriptor import (
    BufferRegion,
    CommPlan,
    DevicePlan,
    Transfer,
    TransferOp,
)
from syncopate.communication.common_descriptors.utils import row_major_strides


@pytest.fixture
def mesh():
    return {"M": 4}

import torch

@pytest.fixture
def gather_meta():
    return {"shape": (16, 2), "dtype": torch.float16}


def test_all_gather_push_plan(mesh, gather_meta):
    plan = build_all_gather_plan_1d_swizzle(
        shape=gather_meta["shape"],
        dtype=gather_meta["dtype"],
        axis=0,
        mesh_size=mesh["M"],
        rank=2,
        buffer_name="buf",
        transfer_kind="push",
    )
    assert len(plan.xfers) == 3
    assert len(plan.local_copies) == 0

    peers = {xf.peer for xf in plan.xfers}
    assert peers == {0, 1, 3}

    push_peer0 = {xf.peer: xf for xf in plan.xfers}[0]
    assert push_peer0.src_off == 32
    assert push_peer0.dst_off == 32
    assert push_peer0.nbytes == 16


def test_all_gather_pull_plan(mesh, gather_meta):
    plan = build_all_gather_plan_1d_swizzle(
        shape=gather_meta["shape"],
        dtype=gather_meta["dtype"],
        axis=0,
        mesh_size=mesh["M"],
        rank=2,
        transfer_kind="pull",
        buffer_name="buf",
    )
    assert len(plan.xfers) == 3
    assert all(xfer.op == TransferOp.PULL for xfer in plan.xfers)


def test_all_to_all_push_plan(mesh):
    plan = build_all_to_all_plan(
        shape=(4, 4),
        dtype=torch.float32,
        mesh_size=mesh["M"],
        rank=1,
        src_buffer="src",
        dst_buffer="dst",
        transfer_kind="push",
    )
    assert getattr(plan, "dst_shape") == (4, 1, 4)
    assert len(plan.xfers) == 3
    assert len(plan.local_copies) == 1

    offsets = {push.peer: (push.src_off, push.dst_off, push.nbytes) for push in plan.xfers}
    assert offsets == {0: (0, 16, 16), 2: (32, 16, 16), 3: (48, 16, 16)}
    local_copy = plan.local_copies[0]
    assert (
        local_copy.src_buf,
        local_copy.dst_buf,
        local_copy.src_off,
        local_copy.dst_off,
        local_copy.nbytes,
    ) == ("src", "dst", 16, 16, 16)


def test_all_to_all_dim_plan_push_reshards_between_axes():
    mesh_size = 4
    plan = build_all_to_all_plan_dim(
        shape=(2, 2, 12, 3),  # [B, S/w, H, D]
        dtype=torch.float32,
        mesh_size=mesh_size,
        rank=1,
        src_buffer="src",
        dst_buffer="dst",
        src_split_axis=1,
        dst_split_axis=2,
        transfer_kind="push",
    )

    print(plan.pretty())

    assert getattr(plan, "dst_shape") == (2, 8, 3, 3)  # [B, S, H/w, D]
    assert len(plan.xfers) == mesh_size - 1
    assert len(plan.local_copies) == 1

    regions = {xf.peer: xf.src_region.describe() for xf in plan.xfers}
    assert regions == {
        0: "[0:2, 0:2, 0:3, 0:3]",
        2: "[0:2, 0:2, 6:9, 0:3]",
        3: "[0:2, 0:2, 9:12, 0:3]",
    }

    local_copy = plan.local_copies[0]
    assert local_copy.dst_region.describe() == "[0:2, 2:4, 0:3, 0:3]"


def test_ring_all_gather_dependencies(mesh, gather_meta):
    # Build ring plans for all ranks to plan signals and dependencies.
    device_plans = {
        rank: build_ring_all_gather_plan(
            shape=gather_meta["shape"],
            dtype=gather_meta["dtype"],
            axis=0,
            mesh_size=mesh["M"],
            rank=rank,
            buffer_name="buf",
        )
        for rank in range(mesh["M"])
    }

    comm_plan = CommPlan(device_plans)
    comm_plan.plan_signals()

    # Rank 2 should forward chunks 2 -> 1 -> 0 to its next neighbor (rank 3).
    rank2_pushes = device_plans[2].xfers
    assert len(rank2_pushes) == mesh["M"] - 1
    assert all(push.op == TransferOp.PUSH and push.peer == 3 for push in rank2_pushes)

    # First hop is local chunk, no dependency.
    assert not rank2_pushes[0].dependency
    assert rank2_pushes[0].chunk_idx == 2

    # Later hops wait on upstream pushes from rank 1.
    dep_1 = rank2_pushes[1].dependency[0]
    upstream_1 = device_plans[1].xfers[0]
    assert dep_1.depend_rank == 1 and dep_1.depend_xfer_idx == 0
    assert dep_1.signal is upstream_1.set_signals[0]
    assert rank2_pushes[1].chunk_idx == 1

    dep_2 = rank2_pushes[2].dependency[0]
    upstream_2 = device_plans[1].xfers[1]
    assert dep_2.depend_rank == 1 and dep_2.depend_xfer_idx == 1
    assert dep_2.signal is upstream_2.set_signals[0]
    assert rank2_pushes[2].chunk_idx == 0

    assert all(not plan.transit_signals for plan in device_plans.values())


def test_all_to_all_dim_plan_pull_reverse_reshard():
    mesh_size = 4
    plan = build_all_to_all_plan_dim(
        shape=(2, 8, 3, 3),  # [B, S, H/w, D]
        dtype=torch.float16,
        mesh_size=mesh_size,
        rank=2,
        src_buffer="src",
        dst_buffer="dst",
        src_split_axis=2,
        dst_split_axis=1,
        transfer_kind="pull",
    )

    assert getattr(plan, "dst_shape") == (2, 2, 12, 3)  # [B, S/w, H, D]
    assert len(plan.xfers) == mesh_size - 1
    assert all(xf.op == TransferOp.PULL for xf in plan.xfers)

    dst_regions = {xf.peer: xf.dst_region.describe() for xf in plan.xfers}
    assert dst_regions == {
        0: "[0:2, 0:2, 0:3, 0:3]",
        1: "[0:2, 0:2, 3:6, 0:3]",
        3: "[0:2, 0:2, 9:12, 0:3]",
    }


def test_all_to_all_dim_wave_plan_splits_dst_axis():
    mesh_size = 2
    num_waves = 3
    plan = build_all_to_all_plan_dim_waves(
        shape=(6, 2),  # [x, y/w]
        dtype=torch.float32,
        mesh_size=mesh_size,
        rank=0,
        src_buffer="src",
        dst_buffer="dst",
        src_split_axis=1,
        dst_split_axis=0,
        num_waves=num_waves,
        transfer_kind="push",
    )

    assert getattr(plan, "dst_shape") == (3, 4)  # [x/w, y]
    assert len(plan.xfers) == (mesh_size - 1) * num_waves
    assert len(plan.local_copies) == num_waves

    src_regions = [xf.src_region.describe() for xf in plan.xfers]
    dst_regions = [xf.dst_region.describe() for xf in plan.xfers]
    assert src_regions == ["[3:4, 0:2]", "[4:5, 0:2]", "[5:6, 0:2]"]
    assert dst_regions == ["[0:1, 0:2]", "[1:2, 0:2]", "[2:3, 0:2]"]

    local_copy_regions = [copy.dst_region.describe() for copy in plan.local_copies]
    assert local_copy_regions == ["[0:1, 0:2]", "[1:2, 0:2]", "[2:3, 0:2]"]

    chunk_ids = [xf.chunk_idx for xf in plan.xfers] + [copy.chunk_idx for copy in plan.local_copies]
    assert sorted(chunk_ids) == [0, 1, 2, 3, 4, 5]


def test_all_to_all_axis_same_dim_push():
    plan = build_all_to_all_plan_axis(
        shape=(2, 4, 3),
        dtype=torch.float32,
        mesh_size=4,
        rank=1,
        src_buffer="src",
        dst_buffer="dst",
        split_axis=1,
        transfer_kind="push",
    )

    assert getattr(plan, "dst_shape") == (2, 4, 3)
    assert len(plan.xfers) == 3
    assert len(plan.local_copies) == 1

    for push in plan.xfers:
        assert push.src_region.total_nbytes() == push.dst_region.total_nbytes() == 24
    local_copy = plan.local_copies[0]
    assert local_copy.src_region.total_nbytes() == 24
    src_offsets = {push.peer: [seg.offset for seg in push.src_region.to_segments()] for push in plan.xfers}
    dst_offsets = {push.peer: [seg.offset for seg in push.dst_region.to_segments()] for push in plan.xfers}
    assert src_offsets == {0: [0, 48], 2: [24, 72], 3: [36, 84]}
    assert dst_offsets == {0: [12, 60], 2: [12, 60], 3: [12, 60]}
    assert [seg.offset for seg in local_copy.src_region.to_segments()] == [12, 60]

def test_device_plan_pretty_contains_transfers(mesh, gather_meta):
    plan = build_all_gather_plan_1d_swizzle(
        shape=gather_meta["shape"],
        dtype=gather_meta["dtype"],
        axis=0,
        mesh_size=mesh["M"],
        rank=1,
        buffer_name="buf",
        transfer_kind="push",
    )
    rendered = plan.pretty()
    assert "DevicePlan(dev=1)" in rendered
    assert "Stream 0:" in rendered
    assert "push peer=" in rendered


def test_push_with_non_contiguous_regions():
    src = BufferRegion.from_segments([(0, 8), (16, 8)])
    dst = BufferRegion.strided(base_offset=32, segment_bytes=8, stride=16, count=2)
    push = Transfer(
        op=TransferOp.PUSH,
        src_buf="src",
        src_region=src,
        peer=1,
        dst_buf="dst",
        dst_region=dst,
        chunk_idx=42,
    )

    assert push.nbytes is None
    assert push.src_off is None
    assert push.dst_off is None
    assert push.src_region.total_nbytes() == 16
    assert push.dst_region.total_nbytes() == 16

    splits = push.split(2)
    assert len(splits) == 2
    for idx, part in enumerate(splits):
        assert part.nbytes == 8
        assert part.src_off in {0, 16}
        assert part.dst_off in {32, 48}
        assert part.chunk_idx == push.chunk_idx * 2 + idx


def test_buffer_region_from_slices_to_segments():
    strides = row_major_strides((4, 2))
    region = BufferRegion.from_slices(
        elem_size=2,
        strides=strides,
        slices=[(2, 4), (0, 2)],
    )

    segments = region.to_segments()
    assert len(segments) == 1
    assert segments[0].offset == 8
    assert segments[0].nbytes == 8
    assert region.describe() == "[2:4, 0:2]"


def test_buffer_region_split_along_axis():
    strides = row_major_strides((8, 4))
    region = BufferRegion.from_slices(
        elem_size=4,
        strides=strides,
        slices=[(0, 8), (0, 4)],
    )

    parts = region.split(4, axis=0)
    assert len(parts) == 4
    assert [part.describe() for part in parts] == [
        "[0:2, 0:4]",
        "[2:4, 0:4]",
        "[4:6, 0:4]",
        "[6:8, 0:4]",
    ]
    assert sum(part.total_nbytes() for part in parts) == region.total_nbytes()

    parts1 = region.split(2, axis=1)
    assert len(parts1) == 2
    assert [part.describe() for part in parts1] == [
        "[0:8, 0:2]",
        "[0:8, 2:4]",
    ]
    assert sum(part.total_nbytes() for part in parts1) == region.total_nbytes()


def test_device_plan_supports_multiple_streams():
    plan = DevicePlan(dev=0)
    second_stream = plan.add_stream()
    assert second_stream == 1

    copy_region = BufferRegion.contiguous(0, 4)
    plan.add_op(
        Transfer(
            op=TransferOp.LOCAL_COPY,
            src_buf="a",
            src_region=copy_region,
            dst_buf="b",
            dst_region=copy_region,
            chunk_idx=-1,
        ),
        stream_idx=second_stream,
    )

    assert len(plan.xfer) == 2
    assert plan.xfer[0] == []
    assert len(plan.xfer[1]) == 1
    assert plan.xfer[1][0].op == TransferOp.LOCAL_COPY


def test_estimate_period_all_gather(mesh, gather_meta):
    device_plans = {
        rank: build_all_gather_plan_1d_swizzle(
            shape=gather_meta["shape"],
            dtype=gather_meta["dtype"],
            axis=0,
            mesh_size=mesh["M"],
            rank=rank,
            buffer_name="buf",
            transfer_kind="push",
        )
        for rank in range(mesh["M"])
    }

    comm_plan = CommPlan(device_plans=device_plans)
    comm_plan.estimate_period()

    target_plan = device_plans[1]
    starts = [xfer.period.start for xfer in target_plan.xfers]

    assert starts == pytest.approx([0.0, 1.0, 2.0])

if __name__ == "__main__":
    shape = (4, 4)
    elem_size = 2
    axis = 0
    mesh_size = 4
    buffer_name = "buf"
    rank = 1

    print("Example all-gather pull plan:")
    plan = build_all_gather_plan_1d_swizzle(
        shape=shape,
        dtype=torch.float16,
        axis=axis,
        mesh_size=mesh_size,
        rank=rank,
        buffer_name=buffer_name,
        transfer_kind="pull",
    )
    print(plan.pretty())
    print("Example all-gather push plan:")
    plan = build_all_gather_plan_1d_swizzle(
        shape=shape,
        dtype=torch.float16,
        axis=axis,
        mesh_size=mesh_size,
        rank=2,
        buffer_name="buf",
        transfer_kind="push",
    )
    print(plan.pretty())
    
    print("Example all-to-all push plan:")
    plan = build_all_to_all_plan(
        shape=(8, 8),
        dtype=torch.float32,
        mesh_size=mesh_size,
        rank=1,
        src_buffer="src",
        dst_buffer="dst",
        transfer_kind="push",
    )
    print(plan.pretty())

    print("Example all-to-all dim plan (pull):")
    plan = build_all_to_all_plan_dim(
        shape =(2, 8, 4, 4), # [B, H, S/w, D]
        dtype=torch.float16,
        mesh_size=mesh_size,
        rank=2,
        src_buffer="src",
        dst_buffer="dst",
        src_split_axis=2,
        dst_split_axis=1,
        transfer_kind="pull",
    )
    print(plan.pretty())

    print("Example all-to-all axis plan (pull):")
    plan = build_all_to_all_plan_axis(
        shape=(4, mesh_size, 4),
        dtype=torch.float16,
        mesh_size=mesh_size,
        rank=2,
        src_buffer="src",
        dst_buffer="dst",
        split_axis=1,
        transfer_kind="pull",
    )
    print(plan.pretty())

    mesh_size = 4
    num_waves = 2
    plan = build_all_to_all_plan_dim_waves(
        shape=(2, 8, 2),  # [x, y/w]
        dtype=torch.float32,
        mesh_size=mesh_size,
        rank=1,
        src_buffer="src",
        dst_buffer="dst",
        src_split_axis=2,
        dst_split_axis=1,
        num_waves=num_waves,
        transfer_kind="push",
    )
    print(plan.pretty())