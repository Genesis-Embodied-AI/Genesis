import glob
import subprocess
import sys
from pathlib import Path

import pytest

rigid_files = glob.glob(Path(__file__).parent.parent / "examples" / "rigid" / "*.py")
coupling_files = glob.glob(Path(__file__).parent.parent / "examples" / "coupling" / "*.py")

example_files = coupling_files + rigid_files
# a regular expression to match the files to exclude
# we want to exclude files that require a display or are interactive
exclude_files = [
    "multi_gpu.py",
]
example_files = [f for f in example_files if not f in exclude_files]


@pytest.mark.example
@pytest.mark.parametrize("example_file", example_files)
def test_example_files(example_file):
    """
    Runs each example script in the examples/rigid directory to ensure they execute without errors.
    """
    # Run each example in a separate process to ensure a clean environment.
    result = subprocess.run([sys.executable, example_file], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        pytest.fail(
            f"Failed to run example {example_file}.\n"
            f"Exit Code: {result.returncode}\n\n"
            f"--- STDOUT ---\n{result.stdout}\n\n"
            f"--- STDERR ---\n{result.stderr}"
        )
