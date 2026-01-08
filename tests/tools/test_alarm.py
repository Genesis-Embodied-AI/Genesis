#!/usr/bin/env python3
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
import yaml
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import Mock, patch
from dataclasses import dataclass
import argparse


# Mock GitHub API calls capture
github_api_calls = []


@dataclass
class MockWandbRun:
    """Mock for W&B run object"""

    config: Dict[str, Any]
    summary: Dict[str, Any]
    state: str

    def __init__(self, config, summary, state="finished"):
        self.config = config
        self.summary = MockSummary(summary)
        self.state = state


class MockSummary:
    """Mock for W&B summary object"""

    def __init__(self, data):
        self._json_dict = data
        for k, v in data.items():
            setattr(self, k, v)

    def get(self, key, default=None):
        """Add get method for dict-like access"""
        return self._json_dict.get(key, default)

    def __getitem__(self, key):
        """Support subscript access like summary['key']"""
        return self._json_dict[key]


class MockWandbApi:
    """Mock for W&B API with sample historical data"""

    def __init__(self, historical_runs: List[MockWandbRun]):
        self.historical_runs = historical_runs

    def runs(self, path: str, order: str = None):
        """Return mock runs iterator"""
        return iter(self.historical_runs)


def create_sample_historical_data() -> List[MockWandbRun]:
    """
    Create sample historical benchmark data for testing.
    Returns 5 commits worth of data with slight variations.
    """
    base_commits = [
        "abc123def456",
        "def456ghi789",
        "ghi789jkl012",
        "jkl012mno345",
        "mno345pqr678",
    ]

    # Base benchmark configurations for speed tests
    benchmark_configs = [
        "solver=PBD-backend=cpu-n_envs=128",
        "solver=PBD-backend=gpu-n_envs=1024",
        "solver=MPM-backend=cpu-n_envs=64",
    ]

    # Base memory configurations
    memory_configs = [
        "scenario=franka-n_envs=30000-backend=gpu-flag=True",
        "scenario=go2-n_envs=4096-backend=gpu-flag=True",
        "scenario=box_pyramid_5-n_envs=4096-backend=gpu",
    ]

    runs = []
    for i, commit in enumerate(base_commits):
        # Speed benchmarks
        for bench_config in benchmark_configs:
            # Add some variation to the baseline data
            variation = 1.0 + (i * 0.01)  # Slight increase over time

            config = {
                "revision": f"{commit}@Genesis-Embodied-AI/genesis",
                "benchmark_id": f"rigid_body-{bench_config}",
            }

            summary = {
                "compile_time": 2.5 * variation,
                "runtime_fps": 1000.0 / variation,
                "realtime_factor": 50.0 / variation,
            }

            runs.append(MockWandbRun(config, summary, state="finished"))

        # Memory benchmarks
        for mem_config in memory_configs:
            # Add some variation to memory data
            variation = 1.0 + (i * 0.02)  # Slight increase over time

            # Base memory values
            base_mem = {
                "scenario=franka-n_envs=30000-backend=gpu-flag=True": 13000,
                "scenario=go2-n_envs=4096-backend=gpu-flag=True": 3100,
                "scenario=box_pyramid_5-n_envs=4096-backend=gpu": 5800,
            }

            config = {
                "revision": f"{commit}@Genesis-Embodied-AI/genesis",
                "benchmark_id": f"memory-{mem_config}",
            }

            summary = {
                "max_mem_mb": base_mem[mem_config] * variation,
            }

            runs.append(MockWandbRun(config, summary, state="finished"))

    return runs


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
    mem_content = """test,max_mem_mb
test_speed[franka-None-True-30000-gpu],13213
test_speed[go2-None-True-4096-gpu],3129
test_speed[box_pyramid_5-None-None-4096-gpu],5859
"""
    mem_test_file.write_text(mem_content)

    print(f"Created sample artifacts in {artifacts_dir} with scenario: {scenario}")
    return artifacts_dir


def mock_github_checks_create(*args, **kwargs):
    """Mock for github.rest.checks.create"""
    call_info = {
        "api": "checks.create",
        "args": args,
        "kwargs": kwargs,
    }
    github_api_calls.append(call_info)
    print("\n" + "=" * 80)
    print("GITHUB API CALL: checks.create")
    print("=" * 80)
    print(f"Repository: {kwargs.get('owner')}/{kwargs.get('repo')}")
    print(f"HEAD SHA: {kwargs.get('head_sha')}")
    print(f"Check Name: {kwargs.get('name')}")
    print(f"Status: {kwargs.get('status')}")
    print(f"Conclusion: {kwargs.get('conclusion')}")
    print(f"\nOutput:")
    output = kwargs.get("output", {})
    print(f"  Title: {output.get('title')}")
    print(f"  Summary: {output.get('summary')}")
    if output.get("text"):
        print(f"\n  Text (Report Body):")
        print("  " + "-" * 76)
        for line in output.get("text", "").split("\n"):
            print(f"  {line}")
        print("  " + "-" * 76)
    print("=" * 80 + "\n")

    return Mock(data={"html_url": "https://github.com/test/repo/runs/12345"})


def mock_github_issues_create_comment(*args, **kwargs):
    """Mock for github.rest.issues.createComment"""
    call_info = {
        "api": "issues.createComment",
        "args": args,
        "kwargs": kwargs,
    }
    github_api_calls.append(call_info)
    print("\n" + "=" * 80)
    print("GITHUB API CALL: issues.createComment")
    print("=" * 80)
    print(f"Repository: {kwargs.get('owner')}/{kwargs.get('repo')}")
    print(f"Issue/PR Number: {kwargs.get('issue_number')}")
    print(f"\nComment Body:")
    print("-" * 80)
    print(kwargs.get("body", ""))
    print("-" * 80)
    print("=" * 80 + "\n")

    return Mock(data={"id": 98765, "html_url": "https://github.com/test/repo/issues/1#issuecomment-98765"})


def mock_github_repos_list_pull_requests(*args, **kwargs):
    """Mock for github.rest.repos.listPullRequestsAssociatedWithCommit"""
    print(f"\nGitHub API: Listing PRs for commit {kwargs.get('commit_sha')}")
    return Mock(data=[{"number": 123, "title": "Test PR"}])


def parse_alarm_yml(alarm_yml_path: Path) -> Tuple[str, Dict[str, str]]:
    """
    Parse alarm.yml and extract the Python script and environment variables.

    Returns:
        Tuple of (python_script_code, env_vars_dict)
    """
    content = alarm_yml_path.read_text()

    # Extract Python code between heredoc delimiters
    # Looking for: python - << 'PY' ... PY
    match = re.search(r"python - << 'PY';\s*EXIT_CODE=\$\?;\s*} \|\| true\s*\n\s*\n(.*?)\n\s*PY", content, re.DOTALL)
    if not match:
        raise ValueError("Could not find Python code block in alarm.yml")

    python_code = match.group(1)

    # Dedent the Python code to remove YAML indentation
    python_code = textwrap.dedent(python_code)

    # Parse the YAML to extract environment variables
    yaml_data = yaml.safe_load(content)

    # Navigate to the env section: jobs.comment-if-regressed.steps[N].env
    env_vars = {}
    try:
        steps = yaml_data["jobs"]["comment-if-regressed"]["steps"]
        for step in steps:
            if step.get("name") == "Check regressions + build outputs":
                env_vars = step.get("env", {})
                break
    except (KeyError, TypeError):
        pass

    # Convert env values to strings and filter out GitHub expressions
    clean_env = {}
    for key, value in env_vars.items():
        # Skip keys with GitHub Actions expressions like ${{ }}
        if isinstance(value, str) and "${{" not in value:
            clean_env[key] = value
        elif isinstance(value, (int, float)):
            clean_env[key] = str(value)

    return python_code, clean_env


def run_alarm_script(artifacts_dir: Path, output_dir: Path, wandb_api_mock: MockWandbApi, alarm_yml_path: Path):
    """
    Run the alarm.yml Python script with mocked dependencies.

    Args:
        artifacts_dir: Directory containing mock artifacts
        output_dir: Directory for output files
        wandb_api_mock: Mocked W&B API instance
        alarm_yml_path: Path to alarm.yml file
    """
    # Extract Python code and environment variables from alarm.yml
    print("📖 Parsing alarm.yml...")
    python_code, yaml_env = parse_alarm_yml(alarm_yml_path)
    print(f"   Extracted {len(python_code)} chars of Python code")
    print(f"   Extracted {len(yaml_env)} environment variables from YAML")

    # Set up environment variables, overriding paths for testing
    env = {
        **yaml_env,
        # Override paths for local testing
        "SPEED_ARTIFACTS_DIR": str(artifacts_dir),
        "MEM_ARTIFACTS_DIR": str(artifacts_dir),
        "CHECK_BODY_PATH": str(output_dir / "check_output.md"),
        "CSV_RUNTIME_PATH": str(output_dir / "runtime_fps.csv"),
        "CSV_COMPILE_PATH": str(output_dir / "compile_time.csv"),
        "CSV_MEM_PATH": str(output_dir / "mem.csv"),
        "WANDB_API_KEY": "mock_key",
    }

    # Update os.environ
    old_env = os.environ.copy()
    os.environ.update(env)

    print(f"   Set {len(env)} environment variables\n")

    # Debug: Save extracted Python code to file for inspection
    debug_py_file = output_dir / "extracted_script.py"
    debug_py_file.write_text(python_code)
    print(f"   💾 Saved extracted Python code to {debug_py_file}\n")

    exit_code = 0
    try:
        # Mock wandb.Api to return our mock
        with patch("wandb.Api", return_value=wandb_api_mock):
            # Execute the Python script extracted from alarm.yml
            # Use a proper globals dict to ensure variables are properly scoped
            exec_globals = {"__name__": "__main__", "__file__": "<alarm.yml>"}
            exec(compile(python_code, "<alarm.yml>", "exec"), exec_globals)
    except SystemExit as e:
        exit_code = e.code
    finally:
        # Restore environment
        os.environ.clear()
        os.environ.update(old_env)

    return exit_code


def simulate_github_actions_steps(output_dir: Path, exit_code: int):
    """
    Simulate the GitHub Actions steps that would run after the Python script.
    This includes the check creation and PR comment steps.
    """
    check_body_path = output_dir / "check_output.md"

    # Read the check output
    check_output = ""
    if check_body_path.exists():
        check_output = check_body_path.read_text()

    # Determine status based on exit code
    has_regressions = exit_code == 42
    has_alerts = exit_code == 43

    print("\n" + "=" * 80)
    print("SIMULATING GITHUB ACTIONS ENVIRONMENT VARIABLES")
    print("=" * 80)
    print(f"EXIT_CODE: {exit_code}")
    print(f"HAS_REGRESSIONS: {1 if has_regressions else 0}")
    print(f"HAS_ALERTS: {1 if has_alerts else 0}")
    print("=" * 80 + "\n")

    # Simulate "Publish PR check" step
    if has_regressions:
        summary = "🔴 Regressions detected. See tables below."
        conclusion = "failure"
    elif has_alerts:
        summary = "⚠️ Large deviation detected. See tables below."
        conclusion = "neutral"
    else:
        summary = "✅ No regressions detected. See tables below."
        conclusion = "success"

    artifact_url = "https://github.com/test/repo/actions/artifacts/123456"
    body = check_output
    if body and artifact_url:
        body += f"\n\n**Artifact:** [Download raw data]({artifact_url})"

    # Mock the checks.create call
    check_response = mock_github_checks_create(
        owner="Genesis-Embodied-AI",
        repo="genesis",
        head_sha="test_commit_sha_123456",
        name="Benchmark Comparison",
        status="completed",
        conclusion=conclusion,
        output={
            "title": "Benchmark Comparison",
            "summary": summary,
            "text": body or None,
        },
    )

    check_url = check_response.data["html_url"]

    # Simulate "Add PR comment" step (only if regressions or alerts)
    if has_regressions or has_alerts:
        # Mock getting PR number
        pr_response = mock_github_repos_list_pull_requests(
            owner="Genesis-Embodied-AI",
            repo="genesis",
            commit_sha="test_commit_sha_123456",
        )

        if pr_response.data:
            pr_number = pr_response.data[0]["number"]

            title = "🔴 Benchmark Regression Detected" if has_regressions else "⚠️ Abnormal Benchmark Result Detected"
            comment = f"**{title} ➡️ [Report]({check_url})**"

            mock_github_issues_create_comment(
                owner="Genesis-Embodied-AI",
                repo="genesis",
                issue_number=pr_number,
                body=comment,
            )


def print_output_files(output_dir: Path):
    """Print the contents of generated output files"""
    print("\n" + "=" * 80)
    print("OUTPUT FILES GENERATED")
    print("=" * 80)

    for file_path in sorted(output_dir.iterdir()):
        if file_path.is_file():
            print(f"\n📄 {file_path.name}")
            print("-" * 80)
            content = file_path.read_text()
            # Truncate very long files
            if len(content) > 2000:
                print(content[:2000])
                print(f"\n... (truncated, total length: {len(content)} chars)")
            else:
                print(content)
            print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Test harness for alarm.yml workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Scenarios:
  ok                - Normal results within tolerance (no regression)
  regression        - Runtime FPS regression detected (>8% drop)
  alert             - Unusually good performance (>8% improvement)
  compile_regression - Compile time regression (>16% increase)
  new_benchmark     - New benchmark not in historical data
        """,
    )
    parser.add_argument(
        "--scenario",
        choices=["ok", "regression", "alert", "compile_regression", "new_benchmark"],
        default="ok",
        help="Test scenario to run (default: ok)",
    )
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary files after test")
    parser.add_argument(
        "--temp-dir", type=Path, help="Use specific directory for temporary files (implies --keep-temp)"
    )
    parser.add_argument(
        "--alarm-yml", type=Path, help="Path to alarm.yml file (default: auto-detect from script location)"
    )

    args = parser.parse_args()

    # Find alarm.yml
    if args.alarm_yml:
        alarm_yml_path = args.alarm_yml
    else:
        # Auto-detect: script is in tests/tools/, alarm.yml is in .github/workflows/
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent.parent
        alarm_yml_path = repo_root / ".github" / "workflows" / "alarm.yml"

    if not alarm_yml_path.exists():
        print(f"❌ Error: alarm.yml not found at {alarm_yml_path}")
        print("   Use --alarm-yml to specify the path")
        return 1

    print(f"Using alarm.yml from: {alarm_yml_path}\n")

    # Create temporary directories
    if args.temp_dir:
        temp_base = args.temp_dir
        temp_base.mkdir(parents=True, exist_ok=True)
        keep_temp = True
    else:
        temp_base = Path(tempfile.mkdtemp(prefix="alarm_test_"))
        keep_temp = args.keep_temp

    artifacts_dir = temp_base / "artifacts"
    output_dir = temp_base / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ALARM.YML TEST HARNESS")
    print("=" * 80)
    print(f"Scenario: {args.scenario}")
    print(f"Temp directory: {temp_base}")
    print(f"Keep temp files: {keep_temp}")
    print("=" * 80 + "\n")

    try:
        # Step 1: Create mock W&B historical data
        print("📊 Creating mock W&B historical data...")
        historical_runs = create_sample_historical_data()
        wandb_api_mock = MockWandbApi(historical_runs)
        print(f"   Created {len(historical_runs)} historical run records\n")

        # Step 2: Create sample artifacts
        print("📦 Creating sample artifacts...")
        create_sample_artifacts(artifacts_dir, args.scenario)
        print()

        # Step 3: Run the alarm script
        print("🔬 Running alarm script...")
        exit_code = run_alarm_script(artifacts_dir, output_dir, wandb_api_mock, alarm_yml_path)
        print(f"   Script exit code: {exit_code}")

        # Interpret exit code
        if exit_code == 42:
            print("   Status: ❌ REGRESSION DETECTED")
        elif exit_code == 43:
            print("   Status: ⚠️  ALERT (unusual deviation)")
        elif exit_code == 0:
            print("   Status: ✅ OK")
        else:
            print(f"   Status: ❓ UNKNOWN (exit code {exit_code})")
        print()

        # Step 4: Simulate GitHub Actions steps
        print("🤖 Simulating GitHub Actions API calls...")
        simulate_github_actions_steps(output_dir, exit_code)

        # Step 5: Display output files
        print_output_files(output_dir)

        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Scenario: {args.scenario}")
        print(f"Exit Code: {exit_code}")
        print(f"GitHub API Calls Made: {len(github_api_calls)}")
        print(f"Output Files: {list(output_dir.glob('*'))}")
        if keep_temp:
            print(f"\n📁 Files saved in: {temp_base}")
        print("=" * 80 + "\n")

        return exit_code

    except Exception as e:
        print(f"\n❌ Error running test: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        # Cleanup
        if not keep_temp:
            print(f"\n🧹 Cleaning up temporary files in {temp_base}")
            shutil.rmtree(temp_base, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
