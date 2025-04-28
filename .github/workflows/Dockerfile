# Start with a base image that includes CUDA
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHONDONTWRITEBYTECODE=1

# Install necessary packages for Python 3.12 and sudo
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common sudo && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y curl wget git cmake gcc tk-dev libssl-dev libffi-dev libbz2-dev && \
    rm -rf /var/lib/apt/lists/*

# Set up python environment
RUN curl https://pyenv.run | bash && \
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> "$HOME/.bash_profile" && \
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> "$HOME/.bash_profile" && \
    echo 'eval "$(pyenv init -)"' >> "$HOME/.bash_profile" && \
    . "$HOME/.bash_profile" && \
    pyenv install 3.10 3.11 3.12

# Install PyTorch with CUDA support
RUN . "$HOME/.bash_profile" && \
    for pyver in "3.10" "3.11" "3.12"; do \
      pyenv shell "$pyver" && \
      pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cu113; \
    done
