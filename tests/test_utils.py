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
from .utils import (
    assert_allclose,
)

TOL = 1e-7

pytestmark = [pytest.mark.required]


@pytest.fixture
def clear_seen_fixture():
    warnings_mod._seen.clear()
    yield
    warnings_mod._seen.clear()


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

    assert_allclose(R_torch, R_recon, tol=TOL)


@pytest.mark.parametrize("batch_size", [1, 10, 100])
def test_numpy_round_trip(batch_size):
    print(f"Testing numpy implementation with batch size {batch_size}...")
    rot = R.random(batch_size)
    R_np = rot.as_matrix()
    if batch_size == 1:
        R_np = R_np[0]

    d6 = R_to_rot6d(R_np)
    R_recon = rot6d_to_R(d6)

    assert_allclose(R_np, R_recon, tol=TOL)


@pytest.mark.parametrize("batch_size", [0, 1, 100])
@pytest.mark.parametrize("func", ["T", "R"])
def test_torch_identity_transform(batch_size, func):
    if batch_size == 0:
        pos = torch.randn(3)
        b_pos = torch.randn(10, 3)
    else:
        pos = torch.randn(batch_size, 3)
        b_pos = torch.randn(batch_size, 10, 3)

    if func == "R":
        identity = torch.eye(3)
        batched_identity = torch.eye(3).reshape(1, 3, 3).repeat(batch_size, 1, 1)
        transform = transform_by_R
    elif func == "T":
        identity = torch.eye(4)
        batched_identity = torch.eye(4).reshape(1, 4, 4).repeat(batch_size, 1, 1)
        transform = transform_by_T

    transformed = transform(pos, identity)
    assert_allclose(pos, transformed, tol=TOL)
    transformed = transform(b_pos, identity)
    assert_allclose(b_pos, transformed, tol=TOL)

    if batch_size > 0:
        transformed = transform(pos, batched_identity)
        assert_allclose(pos, transformed, tol=TOL)
        transformed = transform(b_pos, batched_identity)
        assert_allclose(b_pos, transformed, tol=TOL)


@pytest.mark.parametrize("batch_size", [0, 1, 100])
@pytest.mark.parametrize("func", ["T", "R"])
def test_numpy_identity_transform(batch_size, func):
    if batch_size == 0:
        pos = np.random.randn(3)
        b_pos = np.random.randn(10, 3)
    else:
        pos = np.random.randn(batch_size, 3)
        b_pos = np.random.randn(batch_size, 10, 3)

    if func == "R":
        identity = np.eye(3)
        batched_identity = np.eye(3).reshape(1, 3, 3).repeat(batch_size, 0)
        transform = transform_by_R
    if func == "T":
        identity = np.eye(4)
        batched_identity = np.eye(4).reshape(1, 4, 4).repeat(batch_size, 0)
        transform = transform_by_T

    transformed = transform(pos, identity)
    assert_allclose(pos, transformed, tol=TOL)
    transformed = transform(b_pos, identity)
    assert_allclose(b_pos, transformed, tol=TOL)

    if batch_size > 0:
        transformed = transform(pos, batched_identity)
        assert_allclose(pos, transformed, tol=TOL)
        transformed = transform(b_pos, batched_identity)
        assert_allclose(b_pos, transformed, tol=TOL)


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
