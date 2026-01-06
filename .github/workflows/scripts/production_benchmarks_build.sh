#!/bin/bash

set -ex

curl -LsSf https://astral.sh/uv/install.sh | sh
which uv
uv --version

uv venv --python '3.10' /venv
source /venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu129
uv pip install ".[dev,render]"
