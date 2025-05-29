import os
import subprocess
import tempfile
from traceback import TracebackException

import pytest

START_REF = "v0.2.1"
END_REF = "upstream/main"

BENCHMARK_SCRIPT = r"""
#!/bin/bash
set -e

# Make sure that WanDB is configured, otherwise running the benchmarks is useless
if [[ -z "${WANDB_API_KEY}" ]] ; then
    exit 1;
fi

# Make sure that Genesis is properly installed, with including all its requirements
pip uninstall --no-input -y genesis-world
pip install --no-input --no-user --no-cache --quiet -e ".[dev]" "libigl==2.5.1"

# Run the benchmarks
pytest --print -x -m benchmarks --backend gpu "./tests_2/test_rigid_benchmarks.py"
"""

# Get all commit hashes after a given date (oldest to newest)
commits = subprocess.check_output(
    ["git", "rev-list", "--reverse", f"{START_REF}^..{END_REF}"],
    cwd=os.path.dirname(__file__),
    stderr=subprocess.DEVNULL,
    encoding="utf-8",
).splitlines()
print(f"Found {len(commits)} commits since {START_REF}")

with tempfile.NamedTemporaryFile("w", suffix=".sh") as fd:
    script_fullpath = fd.name
    fd.write(BENCHMARK_SCRIPT)
    fd.flush()
    os.chmod(fd.name, 0o755)  # Make the script executable

    for i, commit in enumerate(commits):
        print(f"\n[{i+1}/{len(commits)}] Checking out {commit}")
        subprocess.run(["git", "checkout", "-f", commit], check=True)

        print("================= ...Running benchmarks... ==================")
        process = subprocess.Popen(
            ["bash", script_fullpath],
            cwd=os.path.dirname(__file__),
        )
        process.wait()
