"""Pytest plugin to register custom command-line options.

This plugin ensures that custom pytest options are registered early enough
to be available to pytest-xdist worker processes, which may restart before
conftest.py is fully loaded.
"""

from argparse import SUPPRESS
import subprocess
import sys

import pytest


def is_mem_monitoring_supported():
    try:
        assert sys.platform.startswith("linux")
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, timeout=2)
        return True, None
    except Exception as exc:  # platform or nvidia-smi unavailable
        return False, exc


def pytest_addoption(parser):
    """Register custom pytest options that need to be available to pytest-xdist workers."""
    supported, _reason = is_mem_monitoring_supported()
    help_text = (
        "Run memory monitoring, and store results to mem_monitoring_filepath. CUDA on linux ONLY."
        if supported
        else SUPPRESS
    )
    parser.addoption("--mem-monitoring-filepath", action="store", default=None, help=help_text)
    parser.addoption("--backend", action="store", default=None, help="Default simulation backend.")
    parser.addoption(
        "--logical", action="store_true", default=False, help="Consider logical cores in default number of workers."
    )
    parser.addoption("--vis", action="store_true", default=False, help="Enable interactive viewer.")
    parser.addoption("--dev", action="store_true", default=False, help="Enable genesis debug mode.")
