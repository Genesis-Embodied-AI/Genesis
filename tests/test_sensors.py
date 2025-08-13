import numpy as np
import torch

import genesis as gs
from genesis.sensors.imu import IMUOptions

from .utils import assert_allclose


def test_imu_sensor(show_viewer):
    """Test if the IMU sensor returns the correct data."""
    GRAVITY = -10.0
    DT = 1e-2

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=1,
            gravity=(0.0, 0.0, GRAVITY),
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(gs.morphs.Plane())

    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.2),
        ),
    )

    imu = scene.add_sensor(IMUOptions(link_idx=box.base_link_idx))
    imu_delayed = scene.add_sensor(IMUOptions(link_idx=box.base_link_idx, read_delay=DT))

    scene.build()

    for _ in range(10):
        scene.step()

    # freefall
    imu_data = imu.read()
    assert_allclose(imu_data["lin_acc"], torch.tensor([0.0, 0.0, 0.0]), tol=1e-7)
    assert_allclose(imu_data["ang_vel"], torch.tensor([0.0, 0.0, 0.0]), tol=1e-7)

    # shift COM to induce angular velocity
    box.set_COM_shift(torch.tensor([[0.1, 0.1, 0.1]]))

    for _ in range(30):
        scene.step()
    # collision with ground

    imu_data = imu.read()
    imu_delayed_data = imu_delayed.read()

    with np.testing.assert_raises(AssertionError, msg="Angular velocity should not be zero due to COM shift"):
        assert_allclose(imu_data["ang_vel"], torch.tensor([0.0, 0.0, 0.0]), tol=1e-3)

    with np.testing.assert_raises(AssertionError, msg="Delayed data should not be equal to the ground truth data"):
        assert_allclose(imu_data["lin_acc"] - imu_delayed_data["lin_acc"], [0.0, 0.0, 0.0], tol=1e-3)

    box.set_COM_shift(torch.tensor([[0.0, 0.0, 0.0]]))

    for _ in range(80):
        scene.step()

    # on ground
    imu_data = imu.read()
    assert_allclose(imu_data["lin_acc"], torch.tensor([0.0, 0.0, -GRAVITY]), tol=1e-7)
    assert_allclose(imu_data["ang_vel"], torch.tensor([0.0, 0.0, 0.0]), tol=1e-5)
