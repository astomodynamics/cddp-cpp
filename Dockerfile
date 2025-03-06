# Description: A Dockerfile for building a C++ project with CUDA support.
FROM nvidia/cuda:12.4.0-base-ubuntu22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        wget \
        libjpeg-dev \
        libpng-dev \
        libeigen3-dev \
    rm -rf /var/lib/apt/lists/*

# Install Cuda Toolkit
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin && \
    mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb && \
    cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get -y install cuda-toolkit-12-4

# Install LibTorch (adjust URL for your desired version)
RUN apt-get update && \
    apt-get install -y --no-install-recommends unzip && \
    wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-2.5.1+cu124.zip -d libtorch && \
    rm libtorch-cxx11-abi-shared-with-deps-2.5.1+cu124.zip && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables for LibTorch
ENV LIBTORCH_DIR=/libtorch/libtorch
ENV LD_LIBRARY_PATH=$LIBTORCH_DIR/lib:$LD_LIBRARY_PATH

# Create a directory for your project
WORKDIR /app

# Copy your project source code
COPY . /app


# # Configure and build your project
RUN rm -rf build && \ 
    mkdir build && \
    cd build && \
    cmake \
            -DCMAKE_BUILD_TYPE=Release \
            -DCDDP_CPP_BUILD_TESTS=ON \
            -DCDDP_CPP_TORCH=ON \
            -DCDDP_CPP_TORCH_GPU=ON \
            -DCDDP_CPP_CASADI=OFF \
            -DPython_EXECUTABLE=/usr/bin/python3 \
            -DLIBTORCH_DIR=/libtorch/libtorch \
            .. && \
    make -j$(nproc) && \
    make test
