ARG CUDA_VERSION=12.8

# ===============================================================
# Stage 1: Build LuisaRender
# ===============================================================
FROM pytorch/pytorch:2.11.0-cuda${CUDA_VERSION}-cudnn9-devel AS builder

ENV DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.12

# Install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    manpages-dev \
    libvulkan-dev \
    zlib1g-dev \
    xorg-dev libglu1-mesa-dev \
    libsnappy-dev \
    software-properties-common \
    git \
    curl \
    wget
RUN add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt update && \
    apt install -y --no-install-recommends \
    gcc-11 \
    g++-11 \
    gcc-11 g++-11 patchelf && \
    rm -rf /var/lib/apt/lists/*

# Set GCC-11 and G++-11 as the default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 110

# Install Rust for build requirements
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

RUN python3 -m pip install --break-system-packages "pybind11[global]"

# Install CMake
RUN python3 -m pip install --break-system-packages --no-cache-dir cmake==3.31.6

# Build LuisaRender
WORKDIR /workspace
RUN git clone https://github.com/Genesis-Embodied-AI/Genesis.git && \
    cd Genesis && \
    git submodule update --init --recursive
COPY build_luisa.sh /workspace/build_luisa.sh
RUN chmod +x ./build_luisa.sh && ./build_luisa.sh ${PYTHON_VERSION}

# ===============================================================
# Stage 2: Runtime Environment
# ===============================================================
FROM pytorch/pytorch:2.11.0-cuda${CUDA_VERSION}-cudnn9-devel

ARG PYTHON_VERSION=3.12
ENV PYTHON_VERSION=${PYTHON_VERSION}
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tmux \
    git \
    curl \
    wget \
    gosu \
    bash-completion \
    libgl1 \
    libgl1-mesa-dri \
    libglu1-mesa \
    libegl-dev \
    libegl1 \
    libxrender1 \
    libglib2.0-0 \
    ffmpeg \
    libgtk2.0-dev \
    pkg-config \
    libvulkan-dev \
    libgles2 \
    libglvnd0 \
    libglx0 \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# --------------------------- Genesis ----------------------------
RUN python3 -m pip install --break-system-packages --no-cache-dir open3d
RUN git clone https://github.com/Genesis-Embodied-AI/Genesis.git && \
    cd Genesis && \
    python3 -m pip install --break-system-packages . && \
    python3 -m pip install --break-system-packages --no-cache-dir PyOpenGL==3.1.5

# -------------------- Surface Reconstruction --------------------
# Set the LD_LIBRARY_PATH directly in the environment
COPY --from=builder /workspace/Genesis/genesis/ext/ParticleMesher/ParticleMesherPy /opt/conda/lib/python${PYTHON_VERSION}/site-packages/genesis/ext/ParticleMesher/ParticleMesherPy
ENV LD_LIBRARY_PATH=/opt/conda/lib/python${PYTHON_VERSION}/site-packages/genesis/ext/ParticleMesher/ParticleMesherPy:$LD_LIBRARY_PATH

# --------------------- Ray Tracing Renderer ---------------------
# Copy LuisaRender build artifacts from the builder stage
COPY --from=builder /workspace/Genesis/genesis/ext/LuisaRender/build/bin /opt/conda/lib/python${PYTHON_VERSION}/site-packages/genesis/ext/LuisaRender/build/bin
# fix GLIBCXX_3.4.30 not found
# Remove this — it's dangerous and broke cmake
# RUN cd /opt/conda/lib && \
#     mv libstdc++.so.6 libstdc++.so.6.old && \
#     ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 libstdc++.so.6
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

COPY 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
COPY nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json

# ---------------------- Custom Entrypoint -----------------------
# Resolve a user matching the caller's host UID at startup. If no user owns that UID yet, create one; if one does
# (including root), reuse it as-is. This avoids mutating the base image's accounts and works for any LOCAL_USER_ID.
RUN cat <<'EOF' > /entrypoint.sh
#!/bin/bash

if [ -n "${LOCAL_USER_ID}" ]; then
    USER_NAME=$(getent passwd "${LOCAL_USER_ID}" | cut -d: -f1)
    if [ -z "${USER_NAME}" ]; then
        USER_NAME=user
        useradd --shell /bin/bash --uid "${LOCAL_USER_ID}" -m "${USER_NAME}"
    fi
    chown -R "${USER_NAME}" "/opt/conda/lib/python${PYTHON_VERSION}/site-packages/genesis/ext/LuisaRender/" || true
    exec gosu "${USER_NAME}" "$@"
else
    exec "$@"
fi
EOF

RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
