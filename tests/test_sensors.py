import numpy as np
import pytest
import torch

import genesis as gs
from genesis.sensors.imu import IMUOptions

from .utils import assert_allclose, assert_array_equal


@pytest.mark.required
def test_imu_sensor(show_viewer):
    """Test if the IMU sensor returns the correct data."""
    GRAVITY = -10.0
    DT = 1e-2
    BIAS = (0.1, 0.2, 0.3)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=1,
            gravity=(0.0, 0.0, GRAVITY),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    scene.add_entity(gs.morphs.Plane())

    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.2),
        ),
    )

    imu_biased = scene.add_sensor(IMUOptions(entity_idx=box.idx, accelerometer_bias=BIAS, gyroscope_bias=BIAS))
    imu_delayed = scene.add_sensor(IMUOptions(entity_idx=box.idx, read_delay=DT * 2))

    scene.build()

    # box is in freefall
    for _ in range(10):
        scene.step()

    # IMU should calculate "classical linear acceleration" using the local frame without accounting for gravity
    # acc_classical_lin_z = - theta_dot ** 2 - cos(theta) * g
    assert_allclose(imu_biased.read()["lin_acc"], BIAS, tol=1e-7)
    assert_allclose(imu_biased.read()["ang_vel"], BIAS, tol=1e-7)
    assert_allclose(imu_delayed.read()["lin_acc"], 0.0, tol=1e-7)
    assert_allclose(imu_delayed.read()["ang_vel"], 0.0, tol=1e-7)

    # shift COM to induce angular velocity
    box.set_COM_shift(torch.tensor([[0.1, 0.1, 0.1]]))

    # box collides with ground
    for _ in range(30):
        scene.step()

    assert_array_equal(imu_biased.read_ground_truth()["lin_acc"], imu_delayed.read_ground_truth()["lin_acc"])
    assert_array_equal(imu_biased.read_ground_truth()["ang_vel"], imu_delayed.read_ground_truth()["ang_vel"])

    with np.testing.assert_raises(AssertionError, msg="Angular velocity should not be zero due to COM shift"):
        assert_allclose(imu_biased.read_ground_truth()["ang_vel"], 0.0, tol=1e-7)

    with np.testing.assert_raises(AssertionError, msg="Delayed data should not be equal to the ground truth data"):
        assert_array_equal(imu_delayed.read()["lin_acc"] - imu_delayed.read_ground_truth()["lin_acc"], 0.0)

    box.set_COM_shift(torch.tensor([[0.0, 0.0, 0.0]]))

    # box is stationary on ground
    for _ in range(80):
        scene.step()

    assert_allclose(imu_biased.read()["lin_acc"], torch.tensor([BIAS[0], BIAS[1], BIAS[2] - GRAVITY]), tol=1e-7)
    assert_allclose(imu_biased.read()["ang_vel"], BIAS, tol=1e-5)
