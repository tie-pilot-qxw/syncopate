
import dataclasses
import os
from functools import partial
from typing import List, Optional

import torch

from syncopate.communication.code_gen import CommGenerator
from syncopate.communication.common_descriptors.all_to_all import build_all_to_all_plan
from syncopate.interface.lowering import lower_comm_plan_to_raw_schedules
import triton
import triton.language as tl
import triton_dist.language as dl
from triton_dist.utils import (dist_print, generate_data,
                               nvshmem_barrier_all_on_stream,
                               nvshmem_create_tensors,
                               nvshmem_free_tensor_sync, perf_func,
                               finalize_distributed, initialize_distributed)


if __name__ == "__main__":
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    TP_GROUP = initialize_distributed()
    rank = TP_GROUP.rank()
    torch.cuda.set_device("cuda:" + str(LOCAL_RANK))

    max_degree = 30 # 1GB
    buffer = nvshmem_create_tensors((1 << max_degree,), dtype=torch.int8, rank=rank, local_world_size=LOCAL_WORLD_SIZE)
    buffer[rank].fill_(rank)
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    for degree in range(14, max_degree + 1):
        size = 1 << degree
        src = buffer[rank][:size].reshape(1024, -1)
        dst = buffer[1 - rank][:size].reshape(1024, -1)

        def one_side_transfer():
            # if rank == 0:
            dst.copy_(src)

        

        _, dur_ms = perf_func(one_side_transfer, warmup_iters=10, iters=100)
        nvshmem_barrier_all_on_stream()

        dist_print(
            f"Size {size} bytes, One side transfer Bandwidth: {size / dur_ms / 1e6:.2f} GB/s"
        )

        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    nvshmem_free_tensor_sync(buffer[rank])
    finalize_distributed()