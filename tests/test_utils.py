import pytest
import torch
import numpy as np

import genesis as gs
from genesis.utils.geom import *
from scipy.spatial.transform import Rotation as R


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

    transformed = transform(pos, identity).cpu().numpy()
    np.testing.assert_allclose(
        pos.cpu().numpy(),
        transformed,
        rtol=1e-5,
        atol=1e-6,
        err_msg=f"NumPy identity_transform failed for batch size {batch_size}",
    )
    transformed = transform(b_pos, identity).cpu().numpy()
    np.testing.assert_allclose(
        b_pos.cpu().numpy(),
        transformed,
        rtol=1e-5,
        atol=1e-6,
        err_msg=f"NumPy identity_transform failed for batch size {batch_size}",
    )

    if batch_size > 0:
        
        transformed = transform(pos, batched_identity).cpu().numpy()
        np.testing.assert_allclose(
            pos.cpu().numpy(),
            transformed,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"NumPy identity_transform failed for batch size {batch_size}",
        )
        transformed = transform(b_pos, batched_identity).cpu().numpy()
        np.testing.assert_allclose(
            b_pos.cpu().numpy(),
            transformed,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"NumPy identity_transform failed for batch size {batch_size}",
        )


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
    np.testing.assert_allclose(
        pos, transformed, rtol=1e-5, atol=1e-6, err_msg=f"NumPy identity_transform failed for batch size {batch_size}"
    )
    transformed = transform(b_pos, identity)
    np.testing.assert_allclose(
        b_pos, transformed, rtol=1e-5, atol=1e-6, err_msg=f"NumPy identity_transform failed for batch size {batch_size}"
    )

    if batch_size > 0:
        transformed = transform(pos, batched_identity)
        np.testing.assert_allclose(
            pos,
            transformed,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"NumPy identity_transform failed for batch size {batch_size}",
        )
        transformed = transform(b_pos, batched_identity)
        np.testing.assert_allclose(
            b_pos,
            transformed,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"NumPy identity_transform failed for batch size {batch_size}",
        )
