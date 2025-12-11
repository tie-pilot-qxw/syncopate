import os

import pytest
import torch

pytest.importorskip("nvshmem")

from syncopate.communication.comm_runtime.all_reduce_collective import NVSHMEMAllReduceCollective
from triton_dist.utils import (
    finalize_distributed,
    initialize_distributed,
    nvshmem_barrier_all_on_stream,
    nvshmem_create_tensor,
    nvshmem_free_tensor_sync,
    dist_print,
)


def _test_nvshmem_all_reduce_separate_buffers():
    collective = None
    tp_group = initialize_distributed()
    rank = tp_group.rank()
    world_size = tp_group.size()
    local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", str(world_size)))

    try:
        # currently NVSHMEM path assumes single-node group
        if world_size != local_world_size:
            pytest.skip("NVSHMEM all-reduce requires world_size == local_world_size")

        torch.cuda.set_device(rank % local_world_size)

        M, N = 512, 256  # N must be divisible by block_n

        src_data = torch.randn((M, N), device="cuda", dtype=torch.float16)
        src = nvshmem_create_tensor((M, N), torch.float16)
        dst = nvshmem_create_tensor((M, N), torch.float16)

        src.copy_(src_data)
        dst.zero_()

        collective = NVSHMEMAllReduceCollective(
            world_size=world_size,
            local_world_size=local_world_size,
            num_comm_sms=4,
            block_size=(128, 256),
            use_multimem_st=True,
        )

        collective.all_reduce_(src, dst, torch.cuda.current_stream())
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()

        torch.distributed.all_reduce(src_data)

        torch.testing.assert_close(dst, src_data, atol=5e-2, rtol=0.0)
        dist_print("NVSHMEM all-reduce separate buffers test passed.", allowed_ranks=[0], need_sync=True)
    finally:
        if collective is not None:
            collective.finalize()
        nvshmem_free_tensor_sync(src)
        nvshmem_free_tensor_sync(dst)
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()
        finalize_distributed()

if __name__ == "__main__":
    _test_nvshmem_all_reduce_separate_buffers()