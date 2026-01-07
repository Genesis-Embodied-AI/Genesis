"""
Unit tests for the alarm.yml GitHub Actions workflow.

These tests use the test harness in tests/tools/test_alarm.py to validate
the benchmark regression detection logic without running the full CI/CD pipeline.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import csv
import sys

# Import the test harness components
sys.path.insert(0, str(Path(__file__).parent / "tools"))
from test_alarm import (
    create_sample_historical_data,
    create_sample_artifacts,
    run_alarm_script,
    MockWandbApi,
    github_api_calls,
)


@pytest.fixture
def alarm_yml_path():
    """Get the path to alarm.yml"""
    repo_root = Path(__file__).parent.parent
    return repo_root / ".github" / "workflows" / "alarm.yml"


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for test artifacts and outputs"""
    temp_dir = Path(tempfile.mkdtemp(prefix="alarm_test_"))
    artifacts_dir = temp_dir / "artifacts"
    output_dir = temp_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    yield {
        "base": temp_dir,
        "artifacts": artifacts_dir,
        "output": output_dir,
    }
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_wandb_api():
    """Create mock W&B API with historical data"""
    historical_runs = create_sample_historical_data()
    return MockWandbApi(historical_runs)


@pytest.fixture(autouse=True)
def reset_github_api_calls():
    """Reset GitHub API calls tracker before each test"""
    github_api_calls.clear()


class TestAlarmWorkflowOkScenario:
    """Tests for the OK scenario (no regressions)"""
    
    def test_ok_scenario_exit_code(self, temp_workspace, mock_wandb_api, alarm_yml_path):
        """Test that OK scenario returns exit code 0"""
        create_sample_artifacts(temp_workspace["artifacts"], "ok")
        exit_code = run_alarm_script(
            temp_workspace["artifacts"],
            temp_workspace["output"],
            mock_wandb_api,
            alarm_yml_path
        )
        assert exit_code == 0, "OK scenario should return exit code 0"
    
    def test_ok_scenario_creates_output_files(self, temp_workspace, mock_wandb_api, alarm_yml_path):
        """Test that all expected output files are created"""
        create_sample_artifacts(temp_workspace["artifacts"], "ok")
        run_alarm_script(
            temp_workspace["artifacts"],
            temp_workspace["output"],
            mock_wandb_api,
            alarm_yml_path
        )
        
        output_dir = temp_workspace["output"]
        assert (output_dir / "check_output.md").exists(), "check_output.md should be created"
        assert (output_dir / "runtime_fps.csv").exists(), "runtime_fps.csv should be created"
        assert (output_dir / "compile_time.csv").exists(), "compile_time.csv should be created"
    
    def test_ok_scenario_markdown_contains_ok_status(self, temp_workspace, mock_wandb_api, alarm_yml_path):
        """Test that markdown output contains OK status indicators"""
        create_sample_artifacts(temp_workspace["artifacts"], "ok")
        run_alarm_script(
            temp_workspace["artifacts"],
            temp_workspace["output"],
            mock_wandb_api,
            alarm_yml_path
        )
        
        markdown = (temp_workspace["output"] / "check_output.md").read_text()
        assert "✅" in markdown, "Markdown should contain OK emoji"
        assert "Thresholds: runtime ± 8%, compile ± 16%" in markdown
        assert "**5** commits" in markdown, "Should reference 5 baseline commits"
    
    def test_ok_scenario_csv_has_ok_status(self, temp_workspace, mock_wandb_api, alarm_yml_path):
        """Test that CSV files contain 'ok' status"""
        create_sample_artifacts(temp_workspace["artifacts"], "ok")
        run_alarm_script(
            temp_workspace["artifacts"],
            temp_workspace["output"],
            mock_wandb_api,
            alarm_yml_path
        )
        
        # Check runtime_fps.csv
        with open(temp_workspace["output"] / "runtime_fps.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 3, "Should have 3 benchmark rows"
            for row in rows:
                assert row["status"] == "ok", f"Status should be 'ok', got {row['status']}"


class TestAlarmWorkflowRegressionScenario:
    """Tests for the regression scenario"""
    
    def test_regression_scenario_exit_code(self, temp_workspace, mock_wandb_api, alarm_yml_path):
        """Test that regression scenario returns exit code 42"""
        create_sample_artifacts(temp_workspace["artifacts"], "regression")
        exit_code = run_alarm_script(
            temp_workspace["artifacts"],
            temp_workspace["output"],
            mock_wandb_api,
            alarm_yml_path
        )
        assert exit_code == 42, "Regression scenario should return exit code 42"
    
    def test_regression_scenario_markdown_shows_regression(self, temp_workspace, mock_wandb_api, alarm_yml_path):
        """Test that markdown output shows regression indicators"""
        create_sample_artifacts(temp_workspace["artifacts"], "regression")
        run_alarm_script(
            temp_workspace["artifacts"],
            temp_workspace["output"],
            mock_wandb_api,
            alarm_yml_path
        )
        
        markdown = (temp_workspace["output"] / "check_output.md").read_text()
        assert "🔴" in markdown, "Markdown should contain regression emoji"
        assert "**-1" in markdown, "Should show negative percentage changes"
    
    def test_regression_scenario_csv_has_regression_status(self, temp_workspace, mock_wandb_api, alarm_yml_path):
        """Test that CSV files contain 'regression' status"""
        create_sample_artifacts(temp_workspace["artifacts"], "regression")
        run_alarm_script(
            temp_workspace["artifacts"],
            temp_workspace["output"],
            mock_wandb_api,
            alarm_yml_path
        )
        
        # Check runtime_fps.csv
        with open(temp_workspace["output"] / "runtime_fps.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            for row in rows:
                assert row["status"] == "regression", f"Status should be 'regression', got {row['status']}"
    
    def test_regression_scenario_validates_thresholds(self, temp_workspace, mock_wandb_api, alarm_yml_path):
        """Test that regressions are properly detected based on thresholds"""
        create_sample_artifacts(temp_workspace["artifacts"], "regression")
        run_alarm_script(
            temp_workspace["artifacts"],
            temp_workspace["output"],
            mock_wandb_api,
            alarm_yml_path
        )
        
        # Check that FPS values are significantly lower (>8% regression)
        with open(temp_workspace["output"] / "runtime_fps.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                current = float(row["current"])
                baseline = float(row["baseline_last"])
                delta_pct = (current - baseline) / baseline * 100
                # Should be below -8% threshold
                assert delta_pct < -8, f"Regression should exceed -8% threshold, got {delta_pct:.1f}%"


class TestAlarmWorkflowAlertScenario:
    """Tests for the alert scenario (unusual improvement)"""
    
    def test_alert_scenario_exit_code(self, temp_workspace, mock_wandb_api, alarm_yml_path):
        """Test that alert scenario returns exit code 43"""
        create_sample_artifacts(temp_workspace["artifacts"], "alert")
        exit_code = run_alarm_script(
            temp_workspace["artifacts"],
            temp_workspace["output"],
            mock_wandb_api,
            alarm_yml_path
        )
        assert exit_code == 43, "Alert scenario should return exit code 43"
    
    def test_alert_scenario_markdown_shows_alert(self, temp_workspace, mock_wandb_api, alarm_yml_path):
        """Test that markdown output shows alert indicators"""
        create_sample_artifacts(temp_workspace["artifacts"], "alert")
        run_alarm_script(
            temp_workspace["artifacts"],
            temp_workspace["output"],
            mock_wandb_api,
            alarm_yml_path
        )
        
        markdown = (temp_workspace["output"] / "check_output.md").read_text()
        assert "⚠️" in markdown, "Markdown should contain alert emoji"
        assert "**+1" in markdown, "Should show positive percentage changes"
    
    def test_alert_scenario_csv_has_alert_status(self, temp_workspace, mock_wandb_api, alarm_yml_path):
        """Test that CSV files contain 'alert' status"""
        create_sample_artifacts(temp_workspace["artifacts"], "alert")
        run_alarm_script(
            temp_workspace["artifacts"],
            temp_workspace["output"],
            mock_wandb_api,
            alarm_yml_path
        )
        
        # Check runtime_fps.csv
        with open(temp_workspace["output"] / "runtime_fps.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            for row in rows:
                assert row["status"] == "alert", f"Status should be 'alert', got {row['status']}"


class TestAlarmWorkflowCompileRegression:
    """Tests for compile time regression scenario"""
    
    def test_compile_regression_exit_code(self, temp_workspace, mock_wandb_api, alarm_yml_path):
        """Test that compile regression returns exit code 42"""
        create_sample_artifacts(temp_workspace["artifacts"], "compile_regression")
        exit_code = run_alarm_script(
            temp_workspace["artifacts"],
            temp_workspace["output"],
            mock_wandb_api,
            alarm_yml_path
        )
        assert exit_code == 42, "Compile regression should return exit code 42"
    
    def test_compile_regression_csv_shows_regression(self, temp_workspace, mock_wandb_api, alarm_yml_path):
        """Test that compile_time.csv shows regression status"""
        create_sample_artifacts(temp_workspace["artifacts"], "compile_regression")
        run_alarm_script(
            temp_workspace["artifacts"],
            temp_workspace["output"],
            mock_wandb_api,
            alarm_yml_path
        )
        
        # Check compile_time.csv
        with open(temp_workspace["output"] / "compile_time.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            for row in rows:
                assert row["status"] == "regression", f"Compile time status should be 'regression'"
    
    def test_compile_regression_threshold(self, temp_workspace, mock_wandb_api, alarm_yml_path):
        """Test that compile regression uses 16% threshold"""
        create_sample_artifacts(temp_workspace["artifacts"], "compile_regression")
        run_alarm_script(
            temp_workspace["artifacts"],
            temp_workspace["output"],
            mock_wandb_api,
            alarm_yml_path
        )
        
        # Check that compile times exceed 16% threshold
        with open(temp_workspace["output"] / "compile_time.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                current = float(row["current"])
                baseline = float(row["baseline_last"])
                delta_pct = (current - baseline) / baseline * 100
                # Should exceed +16% threshold (note: negative sign because compile time going up is bad)
                assert delta_pct > 16, f"Compile regression should exceed +16% threshold, got {delta_pct:.1f}%"


class TestAlarmWorkflowNewBenchmark:
    """Tests for new benchmark scenario"""
    
    def test_new_benchmark_exit_code(self, temp_workspace, mock_wandb_api, alarm_yml_path):
        """Test that new benchmark returns appropriate exit code"""
        create_sample_artifacts(temp_workspace["artifacts"], "new_benchmark")
        exit_code = run_alarm_script(
            temp_workspace["artifacts"],
            temp_workspace["output"],
            mock_wandb_api,
            alarm_yml_path
        )
        # Exit code could be 0 if existing benchmarks are ok
        assert exit_code in [0, 42, 43], f"Exit code should be 0, 42, or 43, got {exit_code}"
    
    def test_new_benchmark_shows_info_status(self, temp_workspace, mock_wandb_api, alarm_yml_path):
        """Test that new benchmarks show info indicator"""
        create_sample_artifacts(temp_workspace["artifacts"], "new_benchmark")
        run_alarm_script(
            temp_workspace["artifacts"],
            temp_workspace["output"],
            mock_wandb_api,
            alarm_yml_path
        )
        
        markdown = (temp_workspace["output"] / "check_output.md").read_text()
        assert "ℹ️" in markdown, "Markdown should contain info emoji for new benchmark"


class TestAlarmWorkflowOutputFormat:
    """Tests for output format validation"""
    
    def test_csv_has_required_columns(self, temp_workspace, mock_wandb_api, alarm_yml_path):
        """Test that CSV files have all required columns"""
        create_sample_artifacts(temp_workspace["artifacts"], "ok")
        run_alarm_script(
            temp_workspace["artifacts"],
            temp_workspace["output"],
            mock_wandb_api,
            alarm_yml_path
        )
        
        # Check runtime_fps.csv
        with open(temp_workspace["output"] / "runtime_fps.csv") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            required_columns = ["solver", "backend", "n_envs", "current", "baseline_last", "status"]
            for col in required_columns:
                assert col in headers, f"CSV should have column '{col}'"
    
    def test_markdown_has_table_structure(self, temp_workspace, mock_wandb_api, alarm_yml_path):
        """Test that markdown output has proper table structure"""
        create_sample_artifacts(temp_workspace["artifacts"], "ok")
        run_alarm_script(
            temp_workspace["artifacts"],
            temp_workspace["output"],
            mock_wandb_api,
            alarm_yml_path
        )
        
        markdown = (temp_workspace["output"] / "check_output.md").read_text()
        
        # Check for table headers
        assert "### Runtime FPS" in markdown
        assert "### Compile Time" in markdown
        assert "| status |" in markdown
        assert "|:------:|" in markdown  # Table alignment row
        
        # Check for footnotes
        assert "(*1)" in markdown
        assert "(*2)" in markdown
    
    def test_baseline_statistics_present(self, temp_workspace, mock_wandb_api, alarm_yml_path):
        """Test that baseline statistics are calculated and present"""
        create_sample_artifacts(temp_workspace["artifacts"], "ok")
        run_alarm_script(
            temp_workspace["artifacts"],
            temp_workspace["output"],
            mock_wandb_api,
            alarm_yml_path
        )
        
        with open(temp_workspace["output"] / "runtime_fps.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Should have baseline statistics
                assert row["baseline_last"] is not None and row["baseline_last"] != ""
                assert row["baseline_mean"] is not None and row["baseline_mean"] != ""
                assert row["baseline_min"] is not None and row["baseline_min"] != ""
                assert row["baseline_max"] is not None and row["baseline_max"] != ""


class TestAlarmWorkflowBenchmarkParsing:
    """Tests for benchmark ID parsing"""
    
    def test_benchmark_params_extracted(self, temp_workspace, mock_wandb_api, alarm_yml_path):
        """Test that benchmark parameters are correctly extracted"""
        create_sample_artifacts(temp_workspace["artifacts"], "ok")
        run_alarm_script(
            temp_workspace["artifacts"],
            temp_workspace["output"],
            mock_wandb_api,
            alarm_yml_path
        )
        
        with open(temp_workspace["output"] / "runtime_fps.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            # Check that we have the expected benchmark configurations
            solvers = {row["solver"] for row in rows}
            backends = {row["backend"] for row in rows}
            
            assert "PBD" in solvers, "Should have PBD solver"
            assert "MPM" in solvers, "Should have MPM solver"
            assert "cpu" in backends, "Should have cpu backend"
            assert "gpu" in backends, "Should have gpu backend"


class TestAlarmWorkflowEdgeCases:
    """Tests for edge cases and error handling"""
    
    def test_no_artifacts_exits_gracefully(self, temp_workspace, mock_wandb_api, alarm_yml_path):
        """Test that script exits gracefully when no artifacts present"""
        # Don't create artifacts
        exit_code = run_alarm_script(
            temp_workspace["artifacts"],
            temp_workspace["output"],
            mock_wandb_api,
            alarm_yml_path
        )
        # Should exit with 0 (no artifacts to check)
        assert exit_code == 0, "Should exit gracefully when no artifacts present"
        
        # Should create empty check file
        assert (temp_workspace["output"] / "check_output.md").exists()


class TestAlarmWorkflowMemoryTracking:
    """Tests for memory usage tracking"""
    
    def test_memory_table_in_output(self, temp_workspace, mock_wandb_api, alarm_yml_path):
        """Test that memory table appears in markdown output"""
        create_sample_artifacts(temp_workspace["artifacts"], "ok")
        run_alarm_script(
            temp_workspace["artifacts"],
            temp_workspace["output"],
            mock_wandb_api,
            alarm_yml_path
        )
        
        markdown = (temp_workspace["output"] / "check_output.md").read_text()
        assert "### Memory Usage" in markdown, "Markdown should contain Memory Usage section"
        assert "Memory (MB)" in markdown, "Should have memory column header"
    
    def test_memory_csv_created(self, temp_workspace, mock_wandb_api, alarm_yml_path):
        """Test that mem.csv file is created"""
        create_sample_artifacts(temp_workspace["artifacts"], "ok")
        run_alarm_script(
            temp_workspace["artifacts"],
            temp_workspace["output"],
            mock_wandb_api,
            alarm_yml_path
        )
        
        assert (temp_workspace["output"] / "mem.csv").exists(), "mem.csv should be created"
    
    def test_memory_csv_has_correct_columns(self, temp_workspace, mock_wandb_api, alarm_yml_path):
        """Test that mem.csv has expected columns"""
        create_sample_artifacts(temp_workspace["artifacts"], "ok")
        run_alarm_script(
            temp_workspace["artifacts"],
            temp_workspace["output"],
            mock_wandb_api,
            alarm_yml_path
        )
        
        with open(temp_workspace["output"] / "mem.csv") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            assert "current" in headers, "mem.csv should have 'current' column"
            assert "status" in headers, "mem.csv should have 'status' column"
            rows = list(reader)
            assert len(rows) > 0, "mem.csv should have data rows"
    
    def test_memory_thresholds_in_report(self, temp_workspace, mock_wandb_api, alarm_yml_path):
        """Test that memory threshold is mentioned in report"""
        create_sample_artifacts(temp_workspace["artifacts"], "ok")
        run_alarm_script(
            temp_workspace["artifacts"],
            temp_workspace["output"],
            mock_wandb_api,
            alarm_yml_path
        )
        
        markdown = (temp_workspace["output"] / "check_output.md").read_text()
        assert "memory ± 10%" in markdown, "Should mention memory threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

