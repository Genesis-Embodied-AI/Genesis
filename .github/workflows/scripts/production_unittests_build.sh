#!/bin/bash

set -ex

curl -LsSf https://astral.sh/uv/install.sh | sh
which uv
uv --version

uv venv --python '3.10' /venv
source /venv/bin/activate
# TODO: USD baking does not currently support Python 3.11 since
# NVIDIA does not currently release `omniverse-kit==107.3` on PyPI.
# See: https://github.com/Genesis-Embodied-AI/Genesis/pull/1300
uv pip install omniverse-kit --index-url https://pypi.nvidia.com/
uv pip install torch --index-url https://download.pytorch.org/whl/cu129
uv pip install ".[dev,render,usd]"
