import torch

from syncopate.communication.code_gen import CommGenerator
from syncopate.communication.common_descriptors.all_gather import build_all_gather_plan_1d_swizzle
from syncopate.interface.lowering import lower_comm_plan_to_raw_schedules

def test_allgather_attn():
    WORLD_SIZE = 4
    B=1
    H=1 
    SEQ = WORLD_SIZE
    DIM=1
    dtype = torch.float16
    device_plans = {
        rank: build_all_gather_plan_1d_swizzle(
            shape=(2, B, H, SEQ, DIM),
            dtype=dtype,
            axis=3,
            mesh_size=WORLD_SIZE,
            rank=rank,
            buffer_name="kv",
            transfer_kind="pull",
        )
        for rank in range(WORLD_SIZE)
    }

    generator = CommGenerator(device_plans)
    generator.plan_signals()
    schedule0 = lower_comm_plan_to_raw_schedules(generator)[0]["kv"]
    offset_lists = schedule0.gen_block_offset_lists()
    print(f"Rank 0 block offsets: {offset_lists}")
    assert offset_lists == [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 2, 0],
        [0, 0, 0, 3, 0],
    ]


if __name__ == "__main__":
    test_allgather_attn()