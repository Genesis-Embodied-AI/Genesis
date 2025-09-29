import os
import sys
import subprocess
import shutil
import shlex
from pathlib import Path

import pytest


# Only enable this suite when explicitly requested (e.g., in the dedicated CI workflow)
if os.environ.get("CI_EXAMPLES") != "1":
    pytest.skip("Example suite disabled (set CI_EXAMPLES=1 to enable).", allow_module_level=True)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"


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
    "flush_cubes.py",  # FIXME: #1721 might be related
    "fem_cube_linked_with_arm.py",  # FIXME: memory bug
}


def _is_skipped_example(path: Path) -> bool:
    parts = set(p.name for p in path.parents if p.name)
    if parts & SKIP_DIR_NAMES:
        return True
    if path.name in SKIP_BASENAMES:
        return True
    return False


def _discover_examples():
    if not EXAMPLES_DIR.exists():
        return []
    candidates = list(EXAMPLES_DIR.rglob("*.py"))
    # Keep a conservative subset known to be headless-friendly
    kept = []
    for p in candidates:
        # Prefer common, quick, non-interactive categories
        if ("/rigid/" in str(p) or "/coupling/" in str(p) or "/collision/" in str(p)) and not _is_skipped_example(p):
            kept.append(p)
        # Allow a few top-level simple demos
        elif p.parent == EXAMPLES_DIR and p.name in {
            "smoke.py",
            "elastic_dragon.py",
            "fem_hard_and_soft_constraint.py",
        }:
            kept.append(p)
    # Stable order for reproducibility
    print("examples:", kept)
    return sorted(set(kept))


EXAMPLE_FILES = _discover_examples()


@pytest.mark.examples_ci
@pytest.mark.required
@pytest.mark.parametrize("example_file", EXAMPLE_FILES, ids=lambda p: p.relative_to(EXAMPLES_DIR).as_posix())
def test_examples_run_headless(example_file: Path):
    env = os.environ.copy()
    cmd = [sys.executable, str(example_file)]
    # Use a pseudo-TTY on Linux to avoid os.get_terminal_size() OSError in non-TTY envs
    if sys.platform.startswith("linux") and shutil.which("script") is not None:
        quoted = " ".join(shlex.quote(c) for c in cmd)
        run_cmd = ["script", "-qec", quoted, "/dev/null"]
    else:
        run_cmd = cmd

    try:
        result = subprocess.run(
            run_cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        pytest.fail(
            f"Timeout running example {example_file}.\n--- STDOUT ---\n{e.stdout or ''}\n\n--- STDERR ---\n{e.stderr or ''}"
        )

    if result.returncode != 0:
        rel = (
            Path(example_file).relative_to(EXAMPLES_DIR)
            if Path(example_file).is_absolute() or Path(example_file).anchor
            else example_file
        )
        pytest.fail(
            f"Failed to run example {rel}.\n"
            f"Exit Code: {result.returncode}\n\n"
            f"--- STDOUT ---\n{result.stdout}\n\n"
            f"--- STDERR ---\n{result.stderr}"
        )
