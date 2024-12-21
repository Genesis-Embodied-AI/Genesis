FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel
ARG DEBIAN_FRONTEND=noninteractive

# RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

RUN apt update && apt install -y \
    git \
    vim \
    wget \
    curl \
    libgl1-mesa-glx \
    libegl-dev \
    libxrender1 \
    libglib2.0-0 \
    ffmpeg \
    libgtk2.0-dev \
    pkg-config \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace
WORKDIR /workspace

RUN pip install --upgrade --no-cache-dir pip && \
    pip install --no-cache-dir PyOpenGL==3.1.6 && \
    pip install -e . --no-cache-dir