#!/bin/bash

# Check if Python version is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <PYTHON_VERSION>"
    exit 1
fi

PYTHON_VERSION=$1

cd Genesis/genesis/ext/LuisaRender && \
git submodule update --init --recursive && \
mkdir -p build && \
cmake -S . -B build \
    -D CMAKE_BUILD_TYPE=Release \
    -D PYTHON_VERSIONS=$PYTHON_VERSION \
    -D LUISA_COMPUTE_DOWNLOAD_NVCOMP=ON \
    -D LUISA_COMPUTE_DOWNLOAD_OIDN=ON \
    -D LUISA_COMPUTE_ENABLE_GUI=OFF \
    -D LUISA_COMPUTE_ENABLE_CUDA=ON \
    -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())") && \
cmake --build build -j $(nproc)