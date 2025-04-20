import pytest
import torch
import numpy as np

import genesis as gs
from genesis.utils.geom import *
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
