# Start with a base image that includes CUDA
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Label the image
LABEL description="Python 3.11 with PyTorch, CUDA support, and a user named ci with sudo access"

# Install necessary packages for Python 3.11 and sudo
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common sudo && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user 'ci' with a password and give sudo privileges
RUN useradd -ms /bin/bash ci && \
    echo 'ci:password' | chpasswd && \
    echo 'ci ALL=(ALL) NOPASSWD: ALL' > /etc/sudoers.d/ci

# Switch to user 'ci'
USER ci
WORKDIR /home/ci

# Set up the virtual environment in user's home directory
RUN python3.11 -m venv venv
ENV PATH="/home/ci/venv/bin:$PATH"

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113