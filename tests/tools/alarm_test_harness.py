"""
Test harness for the alarm.yml GitHub Actions workflow.

This script extracts and runs the Python code from alarm.yml with mocked dependencies,
allowing local testing without running the full GitHub Actions workflow.
"""

import os
import sys
import tempfile
import shutil
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import Mock, patch
from dataclasses import dataclass
import argparse
from ruamel.yaml import YAML
import subprocess


yaml = YAML()


# Mock GitHub API calls capture
github_api_calls = []


def create_sample_artifacts(artifacts_dir: Path, scenario: str = "ok"):
    """
    Create sample artifact files for testing.

    Args:
        artifacts_dir: Directory to create artifacts in
        scenario: Test scenario - "ok", "regression", "alert", or "new_benchmark"
    """
    # Create speed test output file
    speed_test_dir = artifacts_dir / "speed-test-results"
    speed_test_dir.mkdir(parents=True, exist_ok=True)

    speed_test_file = speed_test_dir / "speed_test.txt"

    if scenario == "ok":
        # Normal results within tolerance
        content = """solver=PBD | backend=cpu | n_envs=128 | compile_time=2.52 | runtime_fps=990.0 | realtime_factor=49.5
solver=PBD | backend=gpu | n_envs=1024 | compile_time=2.54 | runtime_fps=985.0 | realtime_factor=49.3
solver=MPM | backend=cpu | n_envs=64 | compile_time=2.53 | runtime_fps=988.0 | realtime_factor=49.4
"""
    elif scenario == "regression":
        # Runtime FPS regression (>8% drop)
        content = """solver=PBD | backend=cpu | n_envs=128 | compile_time=2.52 | runtime_fps=900.0 | realtime_factor=45.0
solver=PBD | backend=gpu | n_envs=1024 | compile_time=2.54 | runtime_fps=880.0 | realtime_factor=44.0
solver=MPM | backend=cpu | n_envs=64 | compile_time=2.53 | runtime_fps=890.0 | realtime_factor=44.5
"""
    elif scenario == "alert":
        # Unusually good performance (>8% improvement - alert)
        content = """solver=PBD | backend=cpu | n_envs=128 | compile_time=2.52 | runtime_fps=1100.0 | realtime_factor=55.0
solver=PBD | backend=gpu | n_envs=1024 | compile_time=2.54 | runtime_fps=1120.0 | realtime_factor=56.0
solver=MPM | backend=cpu | n_envs=64 | compile_time=2.53 | runtime_fps=1110.0 | realtime_factor=55.5
"""
    elif scenario == "compile_regression":
        # Compile time regression (>16% increase)
        content = """solver=PBD | backend=cpu | n_envs=128 | compile_time=3.0 | runtime_fps=990.0 | realtime_factor=49.5
solver=PBD | backend=gpu | n_envs=1024 | compile_time=3.1 | runtime_fps=985.0 | realtime_factor=49.3
solver=MPM | backend=cpu | n_envs=64 | compile_time=3.05 | runtime_fps=988.0 | realtime_factor=49.4
"""
    elif scenario == "new_benchmark":
        # Include a new benchmark not in historical data
        content = """solver=PBD | backend=cpu | n_envs=128 | compile_time=2.52 | runtime_fps=990.0 | realtime_factor=49.5
solver=PBD | backend=gpu | n_envs=1024 | compile_time=2.54 | runtime_fps=985.0 | realtime_factor=49.3
solver=MPM | backend=cpu | n_envs=64 | compile_time=2.53 | runtime_fps=988.0 | realtime_factor=49.4
solver=SAP | backend=gpu | n_envs=512 | compile_time=1.8 | runtime_fps=1500.0 | realtime_factor=75.0
"""
    else:
        content = ""

    speed_test_file.write_text(content)

    # Create memory test CSV file
    mem_test_dir = artifacts_dir / "mem-test-results"
    mem_test_dir.mkdir(parents=True, exist_ok=True)
    mem_test_file = mem_test_dir / "mem.csv"

    # Create sample memory data
    mem_content = """env=franka 	| constraint_solver=None 	| gjk_collision=True 	| batch_size=30000 	| backend=cuda 	| dtype=field 	| max_mem_mb=123
env=anymal 	| constraint_solver=Newton 	| gjk_collision=False 	| batch_size=0 	| backend=cpu 	| dtype=ndarray 	| max_mem_mb=234
"""
    mem_test_file.write_text(mem_content)

    print(f"Created sample artifacts in {artifacts_dir} with scenario: {scenario}")
    return artifacts_dir


def parse_alarm_yml(alarm_yml_path: Path) -> Tuple[str, Dict[str, str]]:
    """
    Parse alarm.yml and extract the Python script and environment variables.

    Returns:
        Tuple of (python_script_code, env_vars_dict)
    """
    content = alarm_yml_path.read_text()
    # with open(alarm_yml_path) as f:
    yaml_data = yaml.load(content)
    import json
    # print(list(content_dict))
    print(json.dumps(yaml_data, indent=2))
    python_step = [
        step for step in yaml_data["jobs"]["comment-if-regressed"]["steps"]
        if step["name"] == "Check regressions + build outputs"
    ][0]
    print('python_step', python_step)
    python_code = python_step["run"]
    print("python_code", python_code)
    python_code = python_code.rstrip() + " \\\n --debug-body-output-path ${CHECK_BODY_PATH}"

    # # Extract Python code between heredoc delimiters
    # # Looking for: python - << 'PY' ... PY
    # match = re.search(r"python - << 'PY';\s*EXIT_CODE=\$\?;\s*} \|\| true\s*\n\s*\n(.*?)\n\s*PY", content, re.DOTALL)
    # assert match
    # python_code = match.group(1)
    # python_code = textwrap.dedent(python_code)

    # yaml_data = yaml.load(content)
    env_vars = {}
    steps = yaml_data["jobs"]["comment-if-regressed"]["steps"]
    for step in steps:
        if step.get("name") == "Check regressions + build outputs":
            env_vars = step.get("env", {})
            break

    clean_env = {}
    for key, value in env_vars.items():
        # Skip keys with GitHub Actions expressions like ${{ }}
        if isinstance(value, str) and "${{" not in value:
            clean_env[key] = value
        elif isinstance(value, (int, float)):
            clean_env[key] = str(value)

    return python_code, clean_env


def run_alarm_script(artifacts_dir: Path, output_dir: Path, alarm_yml_path: Path):
    python_code, yaml_env = parse_alarm_yml(alarm_yml_path)
    env = {
        **yaml_env,
        "SPEED_ARTIFACTS_DIR": str(artifacts_dir),
        "MEM_ARTIFACTS_DIR": str(artifacts_dir),
        "CHECK_BODY_PATH": str(output_dir / "check_output.md"),
        "CSV_RUNTIME_PATH": str(output_dir / "runtime_fps.csv"),
        "CSV_COMPILE_PATH": str(output_dir / "compile_time.csv"),
        "CSV_MEM_PATH": str(output_dir / "mem.csv"),
    }
    for k, v in env.items():
        python_code = python_code.replace(f"${{{k}}}", v)
    print(python_code)
    old_env = os.environ.copy()
    os.environ.update(env)
    debug_py_file = output_dir / "extracted_script.py"
    debug_py_file.write_text(python_code)
    print("dumped alarm script to", debug_py_file)
    # exit_code = 0

    # python_cmd_line = python_code.split(" ")
    exit_code = os.system(python_code)
    # print(python_cmd_line)
    # subprocess.check_output(python_cmd_line)

    # try:
    #     # Execute the Python script extracted from alarm.yml
    #     # Use a proper globals dict to ensure variables are properly scoped
    #     exec_globals = {"__name__": "__main__", "__file__": "<alarm.yml>"}
    #     exec(compile(python_code, "<alarm.yml>", "exec"), exec_globals)
    # except SystemExit as e:
    #     exit_code = e.code
    # finally:
    #     os.environ.clear()
    #     os.environ.update(old_env)
    return exit_code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=["ok", "regression", "alert", "compile_regression", "new_benchmark"],
        default="ok",
    )
    parser.add_argument("--temp-dir", type=Path)

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    alarm_yml_path = repo_root / ".github" / "workflows" / "alarm.yml"

    if args.temp_dir:
        temp_base = args.temp_dir
        temp_base.mkdir(parents=True, exist_ok=True)
    else:
        temp_base = Path(tempfile.mkdtemp(prefix="alarm_test_"))

    artifacts_dir = temp_base / "artifacts"
    output_dir = temp_base / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    create_sample_artifacts(artifacts_dir, args.scenario)
    exit_code = run_alarm_script(artifacts_dir, output_dir, alarm_yml_path)
    if exit_code == 42:
        print("   Status: ❌ REGRESSION DETECTED")
    elif exit_code == 43:
        print("   Status: ⚠️  ALERT (unusual deviation)")
    elif exit_code == 0:
        print("   Status: ✅ OK")
    else:
        print(f"   Status: ❓ UNKNOWN (exit code {exit_code})")
    os.system(f"cat {output_dir / 'check_output.md'}")


if __name__ == "__main__":
    main()
