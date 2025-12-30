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

uv venv --python '3.10'
source .venv/bin/activate
uv pip install --no-input ".[dev,render]"

pytest --print -x -m "benchmarks" ./tests
cat speed_test*.txt > "/mnt/data/artifacts/speed_test_${SLURM_JOB_NAME}.txt"

# tmate -S /tmp/tmate.sock wait tmate-exit
