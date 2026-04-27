FROM nvcr.io/nvidia/pytorch:25.06-py3

SHELL ["/bin/bash", "-lc"]

WORKDIR /workspace

ENV USE_TRITON_DISTRIBUTED_AOT=0
ENV CPPFLAGS="-I/usr/local/cuda/include"

# Basic hygiene
RUN python3 -m pip install --upgrade pip setuptools wheel

# Clone Triton-distributed
RUN git clone https://github.com/tie-pilot-qxw/Triton-distributed.git -b merge-tlx /workspace/Triton-distributed

WORKDIR /workspace/Triton-distributed

# Reset submodules cleanly
RUN git submodule deinit --all -f || true && \
    rm -rf 3rdparty/triton && \
    git submodule update --init --recursive

# Install NVSHMEM / CUDA Python dependencies
RUN pip3 install \
    nvidia-nvshmem-cu12==3.3.9 \
    cuda.core==0.2.0 \
    "Cython>=0.29.24"

RUN CPPFLAGS="-I/usr/local/cuda/include" pip3 install \
    https://developer.download.nvidia.com/compute/nvshmem/redist/nvshmem_python/source/nvshmem_python-source-0.1.0.36132199_cuda12-archive.tar.xz

# Remove Triton installed with torch / previous triton-dist if present
RUN pip3 uninstall -y triton triton_dist || true && \
    python3 - <<'PY'
import site, shutil, pathlib
for p in site.getsitepackages():
    triton_dir = pathlib.Path(p) / "triton"
    if triton_dir.exists():
        print(f"Removing {triton_dir}")
        shutil.rmtree(triton_dir)
PY

# Install Triton-distributed
RUN cd /workspace/Triton-distributed && \
    USE_TRITON_DISTRIBUTED_AOT=0 \
    pip3 install -e python --verbose --no-build-isolation --use-pep517

WORKDIR /workspace

CMD ["/bin/bash"]