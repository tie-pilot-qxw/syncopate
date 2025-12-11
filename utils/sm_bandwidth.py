
import dataclasses
import os
from functools import partial
from typing import List, Optional

import torch

from syncopate.communication.code_gen import CommGenerator
from syncopate.communication.common_descriptors.all_to_all import build_all_to_all_plan
from syncopate.communication.descriptor import ReduceOp
from syncopate.interface.lowering import lower_comm_plan_to_raw_schedules
import triton
import triton.language as tl
import triton_dist.language as dl
from triton_dist.utils import (dist_print, generate_data,
                               nvshmem_barrier_all_on_stream,
                               nvshmem_create_tensors,
                               nvshmem_free_tensor_sync, perf_func,
                               finalize_distributed, initialize_distributed, group_profile, sleep_async)

# @triton.jit
# def _copy_naive_kernel(src_ptr, dst_ptr, n_elements, BLOCK: tl.constexpr,
#                        NUM_SMS: tl.constexpr):
#     pid = tl.program_id(0)
#     start = pid * BLOCK
#     while start < n_elements:
#         offsets = start + tl.arange(0, BLOCK)
#         mask = offsets < n_elements
#         values = tl.load(src_ptr + offsets, mask=mask, other=0)
#         tl.atomic_add(dst_ptr + offsets, values, mask=mask)
#         start += BLOCK * NUM_SMS


# @triton.jit
# def _copy_tma_kernel(src_ptr, dst_ptr, n_elements, BLOCK: tl.constexpr,
#                      NUM_SMS: tl.constexpr):
#     pid = tl.program_id(0)

#     src_desc = tl.make_tensor_descriptor(src_ptr,
#                                          shape=[n_elements],
#                                          strides=[1],
#                                          block_shape=[BLOCK])
#     dst_desc = tl.make_tensor_descriptor(dst_ptr,
#                                          shape=[n_elements],
#                                          strides=[1],
#                                          block_shape=[BLOCK])

#     total_tiles = tl.cdiv(n_elements, BLOCK)
#     tile_id = pid
#     while tile_id < total_tiles:
#         col = tile_id * BLOCK
#         tile = src_desc.load([col])
#         # tile = tl.full(BLOCK, 0, dtype=tl.float16)
#         dst_desc.atomic_add([col], tile)
#         tile_id += NUM_SMS


# def copy_naive(src, dst, num_sms=16):
#     n_elements = src.numel()
#     if n_elements == 0:
#         return
#     per_sm = n_elements // num_sms
#     next_power_of_2 = 1 << (per_sm - 1).bit_length()
#     if next_power_of_2 < 65536:
#         BLOCK = next_power_of_2
#     else:
#         BLOCK = 65536
#     total_tiles = triton.cdiv(n_elements, BLOCK)
#     active_sms = min(num_sms, total_tiles)
#     grid = (active_sms, )
#     _copy_naive_kernel[grid](src,
#                              dst,
#                              n_elements,
#                              BLOCK=BLOCK,
#                              NUM_SMS=active_sms,
#                              num_warps=32)


# def copy_tma(src, dst, num_sms=16):
#     n_elements = src.numel()
#     if n_elements == 0:
#         return
#     per_sm = n_elements // num_sms
#     next_power_of_2 = 1 << (per_sm - 1).bit_length()
#     if next_power_of_2 < 65536:
#         BLOCK = next_power_of_2
#     else:
#         BLOCK = 65536

#     def alloc_fn(size: int, alignment: int, stream: int | None):
#         return torch.empty((size,), device=src.device, dtype=torch.int8)

#     triton.set_allocator(alloc_fn)

#     BLOCK = max(BLOCK, 16 // src.element_size()) # TMA requrie 16 byte aligned access
#     total_tiles = triton.cdiv(n_elements, BLOCK)
#     active_sms = min(num_sms, total_tiles)
#     grid = (active_sms, )
#     _copy_tma_kernel[grid](src,
#                            dst,
#                            n_elements,
#                            BLOCK=BLOCK,
#                            NUM_SMS=active_sms,
#                            num_warps=4,
#                            num_stages=1)


if __name__ == "__main__":
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    TP_GROUP = initialize_distributed()
    rank = TP_GROUP.rank()
    torch.cuda.set_device("cuda:" + str(LOCAL_RANK))

    max_degree = 30 # 1GB
    buffer = nvshmem_create_tensors((1 << (max_degree + 1),), dtype=torch.float16, rank=rank, local_world_size=LOCAL_WORLD_SIZE)
    buffer[rank].fill_(rank)
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    # for degree in range(3, max_degree + 1):
        # size = 1 << degree

    with group_profile("sm_bandwidth", False):
        for use_tma in [False, True]:
            for num_sms in [8, 16, 32, 64, 128]:
                for degree in range(3, max_degree + 1):
                    len = 1 << degree
                    src = buffer[rank][:len]
                    dst = buffer[1 - rank][:len]
                    size = src.numel() * src.element_size()


                    def one_side_transfer():
                        if rank == 0:
                            if use_tma:
                                from syncopate.communication.comm_runtime.tma_persistent_copy import persistent_tma_copy_last_contig
                                persistent_tma_copy_last_contig(src, dst, num_sms=num_sms, reduce_op=ReduceOp.SUM)
                            else:
                                from syncopate.communication.comm_runtime.non_tma_persistent_copy import persistent_naive_copy_last_contig
                                persistent_naive_copy_last_contig(src, dst, num_sms=num_sms, reduce_op=ReduceOp.SUM)
            
                    sleep_async(5)
                    _, dur_ms = perf_func(one_side_transfer, warmup_iters=5, iters=10)
                    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

                    dist_print(
                        f"Size {size} bytes, tma={use_tma}, num_sms={num_sms}, One side transfer Bandwidth: {size / dur_ms / 1e6:.2f} GB/s"
                    )

                    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    nvshmem_free_tensor_sync(buffer[rank])
    finalize_distributed()
