import pytest
import torch
import numpy as np

import genesis as gs
from genesis.utils.geom import *
from scipy.spatial.transform import Rotation as R
from .utils import (
    assert_allclose,
)

TOL = 1e-7

pytestmark = [pytest.mark.required]


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

    assert_allclose(R_torch.cpu(), R_recon.cpu(), tol=TOL)


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
def test_torch_identity_transform(batch_size):
    if batch_size == 0:
        pos = torch.randn(3)
        b_pos = torch.randn(10, 3)
    else:
        pos = torch.randn(batch_size, 3)
        b_pos = torch.randn(batch_size, 10, 3)

    T_identity = torch.eye(4)

    transformed = transform_by_T(pos, T_identity).cpu()
    assert_allclose(pos.cpu(), transformed, tol=TOL)
    transformed = transform_by_T(b_pos, T_identity).cpu()
    assert_allclose(b_pos.cpu(), transformed, tol=TOL)

    if batch_size > 0:
        T_batched_identity = torch.eye(4).reshape(1, 4, 4).repeat(batch_size, 1, 1)
        transformed = transform_by_T(pos, T_batched_identity).cpu()
        assert_allclose(pos.cpu(), transformed, tol=TOL)
        transformed = transform_by_T(b_pos, T_batched_identity).cpu()
        assert_allclose(b_pos.cpu(), transformed, tol=TOL)


@pytest.mark.parametrize("batch_size", [0, 1, 100])
def test_numpy_identity_transform(batch_size):
    if batch_size == 0:
        pos = np.random.randn(3)
        b_pos = np.random.randn(10, 3)
    else:
        pos = np.random.randn(batch_size, 3)
        b_pos = np.random.randn(batch_size, 10, 3)

    T_identity = np.eye(4)

    transformed = transform_by_T(pos, T_identity)
    assert_allclose(pos, transformed, tol=TOL)
    transformed = transform_by_T(b_pos, T_identity)
    assert_allclose(b_pos, transformed, tol=TOL)

    if batch_size > 0:
        T_batched_identity = np.eye(4).reshape(1, 4, 4).repeat(batch_size, 0)
        transformed = transform_by_T(pos, T_batched_identity)
        assert_allclose(pos, transformed, tol=TOL)
        transformed = transform_by_T(b_pos, T_batched_identity)
        assert_allclose(b_pos, transformed, tol=TOL)
