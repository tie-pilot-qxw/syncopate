from ..descriptor import ComputeDescriptor, Buffer, Axis

# A @ B.T
def create_gemm_descriptor() -> ComputeDescriptor:
    # Define axes
    M = Axis(name="M")
    N = Axis(name="N")
    K = Axis(name="K", reduction=True)

    # Define input buffers
    A = Buffer(name="A", axes=[M, K])
    B = Buffer(name="B", axes=[N, K])

    # Define output buffer
    C = Buffer(name="C", axes=[M, N])

    # Create and return the compute descriptor
    return ComputeDescriptor(
        axes=[M, N, K],
        input_buffers=[A, B],
        output_buffers=[C]
    )