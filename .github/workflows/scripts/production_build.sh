#!/bin/bash

set -ex

# sudo apt update
# sudo apt install -y tmate
# tmate -S /tmp/tmate.sock new-session -d
# tmate -S /tmp/tmate.sock wait tmate-ready
# tmate -S /tmp/tmate.sock display -p '#{tmate_ssh}'

pwd
ls
whoami
echo $HOME
ls ~

VENV_DIR="/venv"

if [[ -d "${VENV_DIR}" ]]; then
    # Remove existing venv (use unique temp name to avoid collisions)
    temp_dir="/dev/shm/_venv_$(date +%s)_$$"
    mv "${VENV_DIR}" "$temp_dir" 2>/dev/null || rm -rf "${VENV_DIR}"
    rm -rf "$temp_dir" &
fi

# # Try to create a tmpfs mount inside the container
# if ! mountpoint -q "${VENV_DIR}"; then
#     mkdir -m 000 -p "${VENV_DIR}"
#     if mount -t tmpfs -o size=10G tmpfs "${VENV_DIR}" 2>/dev/null; then
#         echo "Created tmpfs at ${VENV_DIR}"
#     else
#         echo "Warning: Could not create tmpfs, using regular filesystem"
#     fi
# fi

df -hT ${VENV_DIR}

python3 -m pip install virtualenv
virtualenv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
# Pre-install problematic packages that conflict with distutils-installed system packages
pip install --ignore-installed blinker pyparsing setuptools
pip install ".[dev,render]"
pip install torch

# tmate -S /tmp/tmate.sock wait tmate-exit
