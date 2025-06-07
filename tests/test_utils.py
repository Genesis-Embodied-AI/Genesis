import pytest
import torch
import numpy as np
from unittest.mock import patch
import pytest

import genesis as gs
from genesis.utils.geom import *
from genesis.utils import warnings as warnings_mod
from genesis.utils.warnings import warn_once
from scipy.spatial.transform import Rotation as R


@pytest.mark.parametrize("batch_size", [1, 10, 100])
def test_torch_round_trip(batch_size):
    print(f"Testing torch implementation with batch size {batch_size}...")
    rot = R.random(batch_size)
    R_np = rot.as_matrix()
    if batch_size == 1:
        R_np = R_np[0]
    R_torch = torch.tensor(R_np, dtype=torch.float32)

    d6 = R_to_rot6d(R_torch)
    R_recon = rot6d_to_R(d6)

    np.testing.assert_allclose(
        R_torch.cpu().numpy(),
        R_recon.cpu().numpy(),
        rtol=1e-5,
        atol=1e-6,
        err_msg=f"Torch round-trip failed for batch size {batch_size}",
    )


@pytest.mark.parametrize("batch_size", [1, 10, 100])
def test_numpy_round_trip(batch_size):
    print(f"Testing numpy implementation with batch size {batch_size}...")
    rot = R.random(batch_size)
    R_np = rot.as_matrix()
    if batch_size == 1:
        R_np = R_np[0]

    d6 = R_to_rot6d(R_np)
    R_recon = rot6d_to_R(d6)

    np.testing.assert_allclose(
        R_np, R_recon, rtol=1e-5, atol=1e-6, err_msg=f"NumPy round-trip failed for batch size {batch_size}"
    )


@pytest.fixture
def clear_seen_fixture():
    warnings_mod._seen.clear()
    yield
    warnings_mod._seen.clear()


def test_warn_once_logs_once(clear_seen_fixture):
    msg = "This is a warning"
    with patch.object(gs, "logger", create=True) as mock_logger:
        with patch.object(mock_logger, "warning") as mock_warning:
            warn_once(msg)
            warn_once(msg)
            mock_warning.assert_called_once_with(msg)


def test_warn_once_logs_different_messages(clear_seen_fixture):
    msg1 = "Warning 1"
    msg2 = "Warning 2"
    with patch.object(gs, "logger", create=True) as mock_logger:
        with patch.object(mock_logger, "warning") as mock_warning:
            warn_once(msg1)
            warn_once(msg2)
            assert mock_warning.call_count == 2
            mock_warning.assert_any_call(msg1)
            mock_warning.assert_any_call(msg2)


def test_warn_once_with_empty_message(clear_seen_fixture):
    with patch.object(gs, "logger", create=True) as mock_logger:
        with patch.object(mock_logger, "warning") as mock_warning:
            warn_once("")
            warn_once("")
            mock_warning.assert_called_once_with("")
