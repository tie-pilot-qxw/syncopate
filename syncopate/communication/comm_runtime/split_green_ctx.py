from typing import List, Tuple

import cuda.bindings.driver as driver
import cuda.bindings.runtime as runtime
import cuda.cudart as cudart
import cuda.nvrtc as nvrtc
import torch
from cuda.bindings.driver import CUdevice, CUdevResource


def _cudaGetErrorEnum(error):
    """Return the name string for a CUDA error enum.

    Args:
        error: CUDA error object (driver, runtime, or nvrtc)

    Returns:
        Readable error name
    """
    if isinstance(error, driver.CUresult):
        # CUDA Driver API error
        err, name = driver.cuGetErrorName(error)
        return name if err == driver.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, runtime.cudaError_t):
        # CUDA Runtime API error
        return cudart.cudaGetErrorName(error)[1]
    elif isinstance(error, nvrtc.nvrtcResult):
        # NVRTC compile error
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError(f"Unknown error type: {error}")


def checkCudaErrors(result):
    """Validate a CUDA API call result and raise on error.

    Args:
        result: CUDA API return tuple

    Returns:
        The returned data portion if no error was raised

    Raises:
        RuntimeError: if a CUDA call fails
    """
    if result[0].value:
        # Non-zero error code means failure
        raise RuntimeError(
            f"CUDA error code={result[0].value}({_cudaGetErrorEnum(result[0])})"
        )
    # Decide what to return based on tuple length
    if len(result) == 1:
        return None  # Only the error code, no data
    elif len(result) == 2:
        return result[1]  # Single return value
    else:
        return result[1:]  # Multiple return values


def get_cudevice(dev: torch.device) -> CUdevice:
    """Get the CUDA device handle for the given PyTorch device.

    Args:
        dev: PyTorch device

    Returns:
        CUDA device handle
    """
    try:
        # Try to obtain the CUDA device directly
        cu_dev = checkCudaErrors(driver.cuDeviceGet(dev.index))
    except RuntimeError as e:
        # If it fails, initialize the device and retry
        runtime.cudaInitDevice(dev.index, 0, 0)
        cu_dev = checkCudaErrors(driver.cuDeviceGet(dev.index))
    return cu_dev


def get_device_resource(cu_dev: CUdevice) -> CUdevResource:
    """Get the SM (streaming multiprocessor) resource for a CUDA device.

    Args:
        cu_dev: CUDA device handle

    Returns:
        SM resource object
    """
    return checkCudaErrors(
        driver.cuDeviceGetDevResource(
            cu_dev, driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
        )
    )


def split_resource(
    resource: CUdevResource,
    num_groups: int,
    min_count: int,
) -> Tuple[CUdevResource, CUdevResource]:
    """Split an SM resource into multiple groups.

    Args:
        resource: SM resource to split
        num_groups: number of groups to produce
        min_count: minimum SMs per group

    Returns:
        Tuple of (split resources, remaining resource)
    """
    results, _, remaining = checkCudaErrors(
        driver.cuDevSmResourceSplitByCount(
            num_groups,      # number of groups
            resource,        # original resource
            0,              # useFlags - 0 means default
            min_count,      # minimum SMs per group
        )
    )
    return results, remaining


def create_green_ctx_streams(
    cu_dev: CUdevice, resources: List[CUdevResource]
) -> List[torch.Stream]:
    """Create a Green Context and stream for each SM resource group.

    Args:
        cu_dev: CUDA device handle
        resources: list of SM resource groups

    Returns:
        PyTorch streams corresponding to each resource group
    """
    streams = []
    for split in resources:
        # Build a descriptor for each split resource
        desc = checkCudaErrors(driver.cuDevResourceGenerateDesc([split], 1))
        
        # Create Green Context (CUDA 12.0+), allowing concurrent kernels on SM partitions
        green_ctx = checkCudaErrors(
            driver.cuGreenCtxCreate(
                desc,    # resource descriptor
                cu_dev,  # device handle
                driver.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM  # creation flag
            )
        )
        
        # Create a stream within the Green Context
        stream = checkCudaErrors(
            driver.cuGreenCtxStreamCreate(
                green_ctx,  # Green Context
                driver.CUstream_flags.CU_STREAM_NON_BLOCKING,  # non-blocking stream
                0,          # priority; 0 is default
            )
        )
        
        # Convert the CUDA Driver API stream into a PyTorch stream
        streams.append(torch.cuda.get_stream_from_external(stream))

    return streams


def split_device_green_ctx(
    dev: torch.device, num_groups: int, min_count: int
) -> Tuple[List[torch.Stream], List[CUdevResource]]:
    r"""
    Partition a device into multiple Green Contexts and return streams/resources for each group plus the remainder.
    Green Context allows concurrent kernels on different SM partitions.

    Args:
        dev: device to partition
        num_groups: number of groups to create
        min_count: minimum SMs per group (aligned/granularity-adjusted)

    Returns:
        streams: torch.Stream objects for each Green Context
        resources: CUdevResource objects for each Green Context

    Example:
        >>> from flashinfer.green_ctx import split_device_green_ctx
        >>> import torch
        >>> dev = torch.device("cuda:0")
        >>> streams, resources = split_device_green_ctx(dev, 2, 16)
        >>> print([r.sm.smCount for r in resources])
        [16, 16, 100]
        >>> with torch.cuda.stream(streams[0]):
        ...     x = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
        ...     y = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
        ...     z = x @ y
        ...     print(z.shape)
        ...
        torch.Size([8192, 8192])

    Note:
        The returned streams/resources have length ``num_groups + 1``, with the final entry holding the remaining SMs.

    Raises:
        RuntimeError: when the requested SM allocation exceeds device capacity:
        ``num_groups * round_up(min_count, 8) > num_sm``
    """
    # 1. Get the CUDA device handle
    cu_dev = get_cudevice(dev)
    
    # 2. Fetch the device SM resources
    resource = get_device_resource(cu_dev)
    
    # 3. Split SM resources into the requested groups
    results, remaining = split_resource(resource, num_groups, min_count)
    
    # 4. Merge group resources with the remainder
    resources = results + [remaining]
    
    # 5. Create Green Contexts and streams for each resource group
    streams = create_green_ctx_streams(cu_dev, resources)
    
    return streams, resources

if __name__ == "__main__":
    dev = torch.device("cuda:0")
    streams, resources = split_device_green_ctx(dev, 1, 8)
    print([r.sm.smCount for r in resources])
