import os
import sys
import subprocess
from pathlib import Path

import pytest


EXAMPLES_DIR = Path(__file__).parents[1] / "examples"

ALLOW_PATTERNS = {
    "*.py",
    "collision/**/*.py",
    "coupling/**/*.py",
    "drone/interactive_drone.py",
    "drone/fly_route.py",
    # "IPC_Solver/**/*.py",  # TODO: wait for IPC's wheel to be released
    "rigid/**/*.py",
    "render_async/**/*.py",
    "sap_coupling/**/*.py",
    "sensors/**/*.py",
    "tutorials/**/*.py",
}
IGNORE_SCRIPT_NAMES = {
    "ddp_multi_gpu.py",
    "multi_gpu.py",
    "single_franka_batch_render.py",  # FIXME: segfault on exit
    "fem_cube_linked_with_arm.py",  # FIXME: segfault on exit (corrupted double-linked list)
}
if sys.platform != "linux":
    IGNORE_SCRIPT_NAMES |= {
        "cut_dragon.py",
    }

TIMEOUT = 450.0


pytestmark = [
    pytest.mark.examples,
]


def _discover_examples():
    if not EXAMPLES_DIR.exists():
        raise ValueError(f"Example directory '{EXAMPLES_DIR}' does not exist.")

    files = []
    for pattern in ALLOW_PATTERNS:
        for path in EXAMPLES_DIR.glob(pattern):
            if path.name not in IGNORE_SCRIPT_NAMES:
                files.append(path)

    return sorted(files)


@pytest.mark.examples
@pytest.mark.parametrize("backend", [None])  # Disable genesis initialization at worker level
@pytest.mark.parametrize("file", _discover_examples(), ids=lambda p: p.relative_to(EXAMPLES_DIR).as_posix())
def test_example(file: Path):
    # Disable keyboard control and monitoring when running the unit tests
    env = os.environ.copy()
    env["PYNPUT_BACKEND"] = "dummy"

    path_rel = file.relative_to(EXAMPLES_DIR).as_posix()
    try:
        result = subprocess.run(
            [sys.executable, str(file)], env=env, capture_output=True, text=True, check=False, timeout=TIMEOUT
        )
    except subprocess.TimeoutExpired as e:
        err_msg = f"Timeout running example {path_rel}."
        if e.stdout is not None:
            err_msg += f"\n\n--- STDOUT ---\n{e.stdout.decode()}"
        if e.stderr is not None:
            err_msg += f"\n\n--- STDERR ---\n{e.stderr.decode()}"
        pytest.fail(err_msg)

    if result.returncode != 0:
        pytest.fail(
            f"Failed to run example {path_rel} (Exit Code {result.returncode}).\n\n"
            f"--- STDOUT ---\n{result.stdout}\n\n--- STDERR ---\n{result.stderr}"
        )
