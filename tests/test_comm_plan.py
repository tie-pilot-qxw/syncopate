from syncopate.communication.descriptor import (
    BufferRegion,
    CommPlan,
    DevicePlan,
    SignalType,
    Transfer,
    TransferOp,
    XferDependency,
)


def _contiguous(offset: int) -> BufferRegion:
    return BufferRegion.contiguous(offset=offset, nbytes=4)


def _build_push(
    src_dev: int,
    dst_dev: int,
    *,
    chunk_idx: int,
) -> Transfer:
    return Transfer(
        op=TransferOp.PUSH,
        src_buf=f"buf{src_dev}",
        src_region=_contiguous(offset=0),
        dst_buf=f"buf{dst_dev}",
        dst_region=_contiguous(offset=0),
        chunk_idx=chunk_idx,
        peer=dst_dev,
    )


def test_comm_plan_ring_signal_planning():
    # Ring with three devices, each push depends on the upstream peer.
    push_0_to_1 = _build_push(src_dev=0, dst_dev=1, chunk_idx=0)
    push_1_to_2 = _build_push(src_dev=1, dst_dev=2, chunk_idx=0)
    push_2_to_0 = _build_push(src_dev=2, dst_dev=0, chunk_idx=0)

    push_1_to_2.dependency.append(XferDependency(depend_rank=0, depend_xfer_stream=0, depend_xfer_idx=0))
    push_2_to_0.dependency.append(XferDependency(depend_rank=1, depend_xfer_stream=0, depend_xfer_idx=0))

    comm_plan = CommPlan(
        {
            0: DevicePlan(dev=0, xfer=[[push_0_to_1]]),
            1: DevicePlan(dev=1, xfer=[[push_1_to_2]]),
            2: DevicePlan(dev=2, xfer=[[push_2_to_0]]),
        }
    )

    comm_plan.plan_signals()

    # Every push should emit a receive_data signal for its peer.
    sig_0 = comm_plan.device_plans[0].xfer[0][0].set_signals[0]
    sig_1 = comm_plan.device_plans[1].xfer[0][0].set_signals[0]
    sig_2 = comm_plan.device_plans[2].xfer[0][0].set_signals[0]

    assert sig_0.dst_rank == 1 and sig_0.type == SignalType.RECEIVE_DATA
    assert sig_1.dst_rank == 2 and sig_1.type == SignalType.RECEIVE_DATA
    assert sig_2.dst_rank == 0 and sig_2.type == SignalType.RECEIVE_DATA

    # Dependencies should reuse the peer receive signals without extra transit signals.
    dep_1_on_0 = comm_plan.device_plans[1].xfer[0][0].dependency[0]
    dep_2_on_1 = comm_plan.device_plans[2].xfer[0][0].dependency[0]
    assert dep_1_on_0.signal is sig_0
    assert dep_2_on_1.signal is sig_1

    assert all(not plan.transit_signals for plan in comm_plan.device_plans.values())

    viz = comm_plan.visualize_signals()
    assert "receive_data@1[0]" in viz
    assert "receive_data@2[0]" in viz
    assert "receive_data@0[0]" in viz
    print(viz)


def test_comm_plan_transit_signal_for_remote_dependency():
    push_0_to_1 = _build_push(src_dev=0, dst_dev=1, chunk_idx=0)
    local_copy_dev2 = Transfer(
        op=TransferOp.LOCAL_COPY,
        src_buf="buf2",
        src_region=_contiguous(offset=0),
        dst_buf="buf2_out",
        dst_region=_contiguous(offset=0),
        chunk_idx=0,
    )
    local_copy_dev2.dependency.append(XferDependency(depend_rank=0, depend_xfer_stream=0, depend_xfer_idx=0))

    comm_plan = CommPlan(
        {
            0: DevicePlan(dev=0, xfer=[[push_0_to_1]]),
            1: DevicePlan(dev=1, xfer=[[]]),
            2: DevicePlan(dev=2, xfer=[[local_copy_dev2]]),
        }
    )

    comm_plan.plan_signals()

    push_signal = comm_plan.device_plans[0].xfer[0][0].set_signals[0]
    copy_signal = comm_plan.device_plans[2].xfer[0][0].set_signals[0]

    assert push_signal.dst_rank == 1 and push_signal.type == SignalType.RECEIVE_DATA
    assert copy_signal.dst_rank == 2 and copy_signal.type == SignalType.RECEIVE_DATA

    # Remote dependency should introduce a transit signal on the intermediate peer (device 1).
    transit = comm_plan.device_plans[1].transit_signals
    assert len(transit) == 1

    src_signal, dst_signal = transit[0]
    assert src_signal is push_signal
    assert dst_signal.type == SignalType.REMOTE_DATA_READY
    assert dst_signal.dst_rank == 2

    dependency = comm_plan.device_plans[2].xfer[0][0].dependency[0]
    assert dependency.signal is dst_signal

    viz = comm_plan.visualize_signals()
    assert "Transit signals" in viz
    assert "remote_data_ready@2[0]" in viz
