"""
Unit tests for upload_mem_to_wandb.py
"""

import pytest
import tempfile
from pathlib import Path
import sys
import os

# Add tests directory to path
sys.path.insert(0, str(Path(__file__).parent))

from upload_mem_to_wandb import parse_test_name, get_revision


class TestParseTestName:
    """Tests for parse_test_name function"""

    def test_parse_with_all_params(self):
        """Test parsing with all parameters present"""
        test_name = "test_speed[franka-None-True-30000-gpu]"
        result = parse_test_name(test_name)

        assert result == {"scenario": "franka", "flag": "True", "n_envs": "30000", "backend": "gpu"}
        # 'solver': 'None' should be filtered out
        assert "solver" not in result

    def test_parse_with_solver(self):
        """Test parsing with solver parameter"""
        test_name = "test_speed[go2-CG-False-4096-gpu]"
        result = parse_test_name(test_name)

        assert result == {"scenario": "go2", "solver": "CG", "flag": "False", "n_envs": "4096", "backend": "gpu"}

    def test_parse_without_flag(self):
        """Test parsing without flag parameter"""
        test_name = "test_speed[box_pyramid_5-None-None-4096-gpu]"
        result = parse_test_name(test_name)

        assert result == {"scenario": "box_pyramid_5", "n_envs": "4096", "backend": "gpu"}
        # Both solver and flag should be filtered out
        assert "solver" not in result
        assert "flag" not in result

    def test_parse_invalid_format(self):
        """Test parsing with invalid format returns empty dict"""
        test_name = "test_speed_invalid"
        result = parse_test_name(test_name)
        assert result == {}

    def test_parse_insufficient_params(self):
        """Test parsing with too few parameters"""
        test_name = "test_speed[franka-gpu]"
        result = parse_test_name(test_name)
        assert result == {}


class TestGetRevision:
    """Tests for get_revision function"""

    def test_get_revision_format(self):
        """Test that revision has correct format"""
        revision = get_revision()

        # Should be in format: commit_hash@org/repo
        assert "@" in revision
        parts = revision.split("@")
        assert len(parts) == 2

        # Commit hash should be hex string
        commit_hash = parts[0]
        assert len(commit_hash) >= 7  # Short hash at minimum
        assert all(c in "0123456789abcdef" for c in commit_hash.lower())


class TestUploadMemToWandb:
    """Integration tests for memory upload"""

    def test_script_runs_without_api_key(self):
        """Test script exits gracefully without WANDB_API_KEY"""
        # Create temp CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("test,max_mem_mb\n")
            f.write("test_speed[franka-None-True-30000-gpu],13213\n")
            temp_path = f.name

        try:
            # Remove WANDB_API_KEY if set
            old_key = os.environ.pop("WANDB_API_KEY", None)

            from upload_mem_to_wandb import upload_memory_to_wandb

            result = upload_memory_to_wandb(temp_path)

            assert result == 0, "Should return 0 when no API key"

            # Restore key if it existed
            if old_key:
                os.environ["WANDB_API_KEY"] = old_key
        finally:
            os.unlink(temp_path)

    def test_script_handles_missing_file(self):
        """Test script handles missing CSV file gracefully"""
        from upload_mem_to_wandb import upload_memory_to_wandb

        # Set a mock API key so it doesn't skip due to no key
        old_key = os.environ.get("WANDB_API_KEY")
        os.environ["WANDB_API_KEY"] = "mock_key"

        try:
            result = upload_memory_to_wandb("/nonexistent/path.csv")
            assert result == 1, "Should return 1 when file not found"
        finally:
            # Restore original state
            if old_key:
                os.environ["WANDB_API_KEY"] = old_key
            else:
                os.environ.pop("WANDB_API_KEY", None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
