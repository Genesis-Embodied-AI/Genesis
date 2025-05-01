#!/bin/sh

echo "Installing all dependencies..."
apt-get update
apt install -y \
    libegl1 \
    libgl1 \
    libglvnd-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libegl-dev \
    libx11-6  \
    libxrender1 \
    libglu1-mesa \
    libglib2.0-0 \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libosmesa6-dev \
    mesa-utils
echo "Done!"