import numpy as np
import pytest
import torch

import genesis as gs
from genesis.sensors.imu import IMUOptions
from genesis.sensors.tactile import ContactSensorOptions, ForceSensorOptions

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

    imu_biased = scene.add_sensor(IMUOptions(entity_idx=box.idx, acc_bias=BIAS, gyro_bias=BIAS))
    imu_delayed = scene.add_sensor(IMUOptions(entity_idx=box.idx, delay=DT * 2))
    imu_noisy = scene.add_sensor(
        IMUOptions(
            entity_idx=box.idx,
            acc_axes_skew=0.01,
            gyro_axes_skew=(0.02, 0.03, 0.04),
            acc_noise_std=(0.01, 0.01, 0.01),
            gyro_noise_std=(0.01, 0.01, 0.01),
            acc_bias_drift_std=(0.001, 0.001, 0.001),
            gyro_bias_drift_std=(0.001, 0.001, 0.001),
            delay=DT,
            jitter=DT * 0.1,
            interpolate_for_delay=True,
        )
    )
    imu_skewed = scene.add_sensor(
        IMUOptions(
            entity_idx=box.idx,
            acc_axes_skew=(0.0, 0.0, 1.0),
        )
    )

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
    assert_allclose(imu_noisy.read()["lin_acc"], 0.0, tol=1e-1)
    assert_allclose(imu_noisy.read()["ang_vel"], 0.0, tol=1e-1)

    # shift COM to induce angular velocity
    box.set_COM_shift(torch.tensor([[0.1, 0.1, 0.1]]))

    # try updating noise and bias for accelerometer and gyroscope
    imu_noisy.set_acc_noise_std([0.01, 0.01, 0.01])
    imu_noisy.set_gyro_noise_std([0.02, 0.02, 0.02])
    imu_noisy.set_bias([0.01, 0.01, 0.01, 0.02, 0.02, 0.02])
    imu_noisy.set_jitter(0.001)

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

    assert_allclose(imu_skewed.read()["lin_acc"], -GRAVITY, tol=1e-7)
    assert_allclose(imu_biased.read()["lin_acc"], torch.tensor([BIAS[0], BIAS[1], BIAS[2] - GRAVITY]), tol=1e-7)
    assert_allclose(imu_biased.read()["ang_vel"], BIAS, tol=1e-5)


def test_rigid_tactile_sensors_gravity_force(show_viewer):
    """Test if the sensor will detect the correct forces being applied on a falling box."""
    GRAVITY = -10.0
    N_ENVS = 0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=1,
            gravity=(0.0, 0.0, GRAVITY),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    scene.add_entity(morph=gs.morphs.Plane())

    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(1.0, 1.0, 1.0),  # volume = 1 m^3
            pos=(0.0, 0.0, 1.1),
        ),
        material=gs.materials.Rigid(rho=1.0),  # mass = 1 kg
    )

    bool_sensor = scene.add_sensor(ContactSensorOptions(entity_idx=box.idx))
    force_sensor = scene.add_sensor(ForceSensorOptions(entity_idx=box.idx))
    normtan_force_sensor = scene.add_sensor(ForceSensorOptions(entity_idx=box.idx, return_normtan=True))

    scene.build(n_envs=N_ENVS)

    assert not bool_sensor.read().any(), "RigidContactSensor should not be in contact with the ground yet."
    assert_array_equal(
        force_sensor.read()["force"], 0.0, err_msg="RigidContactForceSensor should be zero before contact."
    )
    assert_array_equal(
        normtan_force_sensor.read()["normal"],
        0.0,
        err_msg="RigidContactForceSensor normal should be zero before contact.",
    )

    for _ in range(100):
        scene.step()

    assert bool_sensor.read().all(), "Sensor should detect contact with the ground"
    assert_allclose(
        force_sensor.read()["force"],
        torch.tensor([0.0, 0.0, -GRAVITY]),
        tol=1e-6,
        err_msg="Force should be equal to -gravity (normal) force.",
    )
    assert_allclose(
        force_sensor.read()["magnitude"],
        -GRAVITY,
        tol=1e-6,
        err_msg="Force magnitude should be equal to -gravity (normal) force.",
    )
    assert_allclose(
        normtan_force_sensor.read()["normal"],
        -GRAVITY,
        tol=1e-6,
        err_msg="Normal force should be equal to -gravity (normal) force.",
    )
