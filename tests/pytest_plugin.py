"""Pytest plugin to register custom command-line options.

This plugin ensures that custom pytest options are registered early enough
to be available to pytest-xdist worker processes, which may restart before
conftest.py is fully loaded.
"""
import pytest


def pytest_addoption(parser):
    """Register custom pytest options that need to be available to pytest-xdist workers."""
    parser.addoption(
        "--mem-monitoring-filepath",
        action="store",
        default=None,
        help="Run memory monitoring, and store results to mem_monitoring_filepath. CUDA on linux ONLY.",
    )
    parser.addoption("--backend", action="store", default=None, help="Default simulation backend.")
    parser.addoption(
        "--logical", action="store_true", default=False, help="Consider logical cores in default number of workers."
    )
    parser.addoption("--vis", action="store_true", default=False, help="Enable interactive viewer.")
    parser.addoption("--dev", action="store_true", default=False, help="Enable genesis debug mode.")

