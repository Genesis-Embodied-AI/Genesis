#!/bin/bash

set -ex

# sudo apt update
# sudo apt install -y tmate
# tmate -S /tmp/tmate.sock new-session -d
# tmate -S /tmp/tmate.sock wait tmate-ready
# tmate -S /tmp/tmate.sock display -p '#{tmate_ssh}'

pwd
ls

curl -LsSf https://astral.sh/uv/install.sh | sh
which uv
uv --version

VENV_DIR="/venv"

if [[ -d "${VENV_DIR}" ]]; then
    # Remove existing venv (use unique temp name to avoid collisions)
    temp_dir="/_venv_$(date +%s)_$$"
    mv "${VENV_DIR}" "$temp_dir" 2>/dev/null || rm -rf "${VENV_DIR}"
    rm -rf "$temp_dir" &
fi


uv venv --python '3.10' --allow-existing ${VENV_DIR}
source "${VENV_DIR}/bin/activate"
uv pip install ".[dev,render]"
uv pip install torch

# tmate -S /tmp/tmate.sock wait tmate-exit
