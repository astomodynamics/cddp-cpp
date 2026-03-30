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
        libeigen3-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Cuda Toolkit
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin && \
    mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb && \
    cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get -y install cuda-toolkit-12-4

# Create a directory for your project
WORKDIR /app

# Copy your project source code
COPY . /app


# Configure and build the project
RUN rm -rf build && \
    cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCDDP_CPP_BUILD_TESTS=ON \
        -DCDDP_CPP_BUILD_EXAMPLES=ON \
        -DCDDP_CPP_CASADI=OFF && \
    cmake --build build -j$(nproc) && \
    ctest --test-dir build --output-on-failure
