import os
import sys
import subprocess
import shutil
import shlex
from pathlib import Path

import pytest


EXAMPLES_DIR = Path(__file__).parents[1] / "examples"

ALLOW_PATTERNS = {
    "/rigid/",
    "/coupling/",
    "/collision/",
}
ALLOW_TOPLEVEL_SCRIPT_NAMES = {
    "smoke.py",
    "elastic_dragon.py",
    "fem_hard_and_soft_constraint.py",
}
SKIP_DIR_NAMES = {
    "rendering",
    "speed_benchmark",
    "tutorials",
    "drone",
    "locomotion",
    "manipulation",
}
SKIP_BASENAMES = {
    "keyboard_teleop.py",
    "render_async.py",
    "ddp_multi_gpu.py",
    "multi_gpu.py",
    "cut_dragon.py",
    "differentiable_push.py",
    "single_franka_batch_render.py",  # FIXME: it will have segfault on exit
    "fem_cube_linked_with_arm.py",  # FIXME: memory bug
}

TIMEOUT = 400.0


# Only enable this suite when explicitly requested (e.g., in the dedicated CI workflow)
pytestmark = [
    pytest.mark.examples,
    pytest.mark.taichi_offline_cache(False),
]


def _is_skipped_example(path: Path) -> bool:
    parts = set(p.name for p in path.parents if p.name)
    if parts & SKIP_DIR_NAMES:
        return True
    if path.name in SKIP_BASENAMES:
        return True
    return False


def _discover_examples():
    if not EXAMPLES_DIR.exists():
        raise ValueError(f"Example directory '{EXAMPLES_DIR}' does not exist.")

    # Keep a conservative subset known to be headless-friendly
    files = []
    for path in EXAMPLES_DIR.rglob("*.py"):
        # Prefer common, quick, non-interactive categories
        if any(pattern in str(path) for pattern in ALLOW_PATTERNS) and not _is_skipped_example(path):
            files.append(path)
        # Allow a few top-level simple demos
        elif path.parent == EXAMPLES_DIR and path.name in ALLOW_TOPLEVEL_SCRIPT_NAMES:
            files.append(path)

    return files


@pytest.mark.examples
@pytest.mark.parametrize("file", _discover_examples(), ids=lambda p: p.relative_to(EXAMPLES_DIR).as_posix())
def test_example(file: Path):
    run_cmd = [sys.executable, str(file)]

    # Use a pseudo-TTY on Linux to avoid os.get_terminal_size() OSError in non-TTY envs
    if sys.platform.startswith("linux") and shutil.which("script") is not None:
        run_cmd = ["script", "-qec", " ".join(map(shlex.quote, run_cmd)), "/dev/null"]

    path_rel = file.relative_to(EXAMPLES_DIR).as_posix()
    try:
        result = subprocess.run(run_cmd, env=os.environ, capture_output=True, text=True, check=False, timeout=TIMEOUT)
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
