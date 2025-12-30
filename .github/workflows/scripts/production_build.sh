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

if [[ -d /tmp/venv ]]; then {
    mv /tmp/venv /tmp/_venv
    rm -Rf /tmp/_venv &
} fi

# export UV_CACHE_DIR=${PWD}/.uv_cache

uv venv --python '3.10' --allow-existing /tmp/venv
source /tmp/venv/bin/activate
uv pip install ".[dev,render]"
uv pip install torch

# pytest --print -x -m "benchmarks" ./tests
# cat speed_test*.txt > "/mnt/data/artifacts/speed_test_${SLURM_JOB_NAME}.txt"

# tmate -S /tmp/tmate.sock wait tmate-exit
