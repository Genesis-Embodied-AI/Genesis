import numpy as np
import pytest
import torch

import genesis as gs

from .utils import assert_allclose, assert_array_equal


@pytest.mark.required
def test_imu_sensor(show_viewer, tol):
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
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=show_viewer,
    )

    scene.add_entity(gs.morphs.Plane())

    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.2),
        ),
    )

    imu_biased = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=box.idx,
            acc_bias=BIAS,
            gyro_bias=BIAS,
        )
    )
    imu_delayed = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=box.idx,
            delay=DT * 2,
        )
    )
    imu_noisy = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=box.idx,
            acc_axes_skew=0.01,
            gyro_axes_skew=(0.02, 0.03, 0.04),
            acc_noise=(0.01, 0.01, 0.01),
            gyro_noise=(0.01, 0.01, 0.01),
            acc_random_walk=(0.001, 0.001, 0.001),
            gyro_random_walk=(0.001, 0.001, 0.001),
            delay=DT,
            jitter=DT * 0.1,
            interpolate=True,
        )
    )
    imu_skewed = scene.add_sensor(
        gs.sensors.IMU(
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
    assert_allclose(imu_biased.read()["lin_acc"], BIAS, tol=tol)
    assert_allclose(imu_biased.read()["ang_vel"], BIAS, tol=tol)
    assert_allclose(imu_delayed.read()["lin_acc"], 0.0, tol=tol)
    assert_allclose(imu_delayed.read()["ang_vel"], 0.0, tol=tol)
    assert_allclose(imu_noisy.read()["lin_acc"], 0.0, tol=1e-1)
    assert_allclose(imu_noisy.read()["ang_vel"], 0.0, tol=1e-1)

    # shift COM to induce angular velocity
    box.set_COM_shift(torch.tensor([[0.1, 0.1, 0.1]]))

    # update noise and bias for accelerometer and gyroscope
    imu_noisy.set_acc_noise([0.01, 0.01, 0.01])
    imu_noisy.set_gyro_noise([0.02, 0.02, 0.02])
    imu_noisy.set_bias([0.01, 0.01, 0.01, 0.02, 0.02, 0.02])
    imu_noisy.set_jitter(0.001)

    # box collides with ground
    for _ in range(30):
        scene.step()

    assert_array_equal(imu_biased.read_ground_truth()["lin_acc"], imu_delayed.read_ground_truth()["lin_acc"])
    assert_array_equal(imu_biased.read_ground_truth()["ang_vel"], imu_delayed.read_ground_truth()["ang_vel"])

    with np.testing.assert_raises(AssertionError, msg="Angular velocity should not be zero due to COM shift"):
        assert_allclose(imu_biased.read_ground_truth()["ang_vel"], 0.0, tol=tol)

    with np.testing.assert_raises(AssertionError, msg="Delayed data should not be equal to the ground truth data"):
        assert_array_equal(imu_delayed.read()["lin_acc"] - imu_delayed.read_ground_truth()["lin_acc"], 0.0)

    box.set_COM_shift(torch.tensor([[0.0, 0.0, 0.0]]))

    # box is stationary on ground
    for _ in range(80):
        scene.step()

    assert_allclose(imu_skewed.read()["lin_acc"], -GRAVITY, tol=tol)
    assert_allclose(imu_biased.read()["lin_acc"], (BIAS[0], BIAS[1], BIAS[2] - GRAVITY), tol=tol)
    assert_allclose(imu_biased.read()["ang_vel"], BIAS, tol=1e-5)

    scene.reset()

    assert_allclose(imu_biased.read()["lin_acc"], 0.0, tol=gs.EPS)  # biased, but cache hasn't been updated yet
    assert_allclose(imu_delayed.read()["lin_acc"], 0.0, tol=gs.EPS)
    assert_allclose(imu_noisy.read()["ang_vel"], 0.0, tol=gs.EPS)

    scene.step()
    assert_allclose(imu_biased.read()["lin_acc"], BIAS, tol=tol)


@pytest.mark.required
def test_rigid_tactile_sensors_gravity_force(show_viewer, tol):
    """Test if the sensor will detect the correct forces being applied on a falling box."""
    GRAVITY = -10.0
    BIAS = (0.1, 0.2, 0.3)
    NOISE = 0.01

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=1,
            gravity=(0.0, 0.0, GRAVITY),
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=show_viewer,
    )

    scene.add_entity(morph=gs.morphs.Plane())

    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(1.0, 1.0, 1.0),  # volume = 1 m^3
            pos=(0.0, 0.0, 1.1),
        ),
        material=gs.materials.Rigid(rho=1.0),  # mass = 1 kg
    )

    bool_sensor = scene.add_sensor(
        gs.sensors.Contact(
            entity_idx=box.idx,
        )
    )
    force_sensor = scene.add_sensor(
        gs.sensors.ContactForce(
            entity_idx=box.idx,
            min_force=0.01,
            max_force=(10.0, 20.0, -GRAVITY / 2),
            noise=NOISE,
            bias=BIAS,
            random_walk=(NOISE * 0.01, NOISE * 0.02, NOISE * 0.03),
            delay=0.05,
            jitter=0.01,
            interpolate=True,
        )
    )

    scene.build()

    scene.step()

    assert not bool_sensor.read().any(), "RigidContactSensor should not be in contact with the ground yet."
    assert_allclose(
        force_sensor.read_ground_truth(),
        0.0,
        tol=gs.EPS,
        err_msg="RigidContactForceSensor ground truth should be zero before contact.",
    )
    assert_allclose(
        force_sensor.read(),
        BIAS,
        tol=NOISE * 2,
        err_msg="RigidContactForceSensor should only read bias and small amount of noise before contact.",
    )

    for _ in range(120):
        scene.step()

    assert bool_sensor.read().all(), "Sensor should detect contact with the ground"
    assert_allclose(
        force_sensor.read_ground_truth(),
        (0.0, 0.0, -GRAVITY),
        tol=tol,
        err_msg="RigidContactForceSensor ground truth should be equal to -gravity (normal) force.",
    )
    assert_allclose(
        force_sensor.read(),
        (BIAS[0], BIAS[1], -GRAVITY / 2),
        tol=NOISE * 10,
        err_msg="RigidContactForceSensor should read bias and noise and -gravity (normal) force clipped by max_force.",
    )
