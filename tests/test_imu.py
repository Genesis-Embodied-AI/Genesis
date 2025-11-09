import numpy as np
import torch
from utils import assert_allclose, assert_array_equal

import genesis as gs
import genesis.utils.geom as gu


def expand_batch_dim(values: tuple[float, ...], n_envs: int) -> tuple[float, ...] | np.ndarray:
    """Helper function to expand expected values for n_envs dimension."""
    if n_envs == 0:
        return values
    return np.tile(np.array(values), (n_envs,) + (1,) * len(values))


def test_imu_sensor(show_viewer, tol, n_envs):
    """Test if the IMU sensor returns the correct data."""
    GRAVITY = -10.0
    DT = 1e-2
    BIAS = (0.1, 0.2, 0.3)
    DELAY_STEPS = 2

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

    imu = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=box.idx,
        )
    )
    imu_delayed = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=box.idx,
            delay=DT * DELAY_STEPS,
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

    scene.build(n_envs=n_envs)

    # box is in freefall
    for _ in range(10):
        scene.step()

    # IMU should calculate "classical linear acceleration" using the local frame without accounting for gravity
    # acc_classical_lin_z = - theta_dot ** 2 - cos(theta) * g
    assert_allclose(imu.read().lin_acc, 0.0, tol=tol)
    assert_allclose(imu.read().ang_vel, 0.0, tol=tol)
    assert_allclose(imu_noisy.read().lin_acc, 0.0, tol=1e-1)
    assert_allclose(imu_noisy.read().ang_vel, 0.0, tol=1e-1)

    # shift COM to induce angular velocity
    com_shift = torch.tensor([[0.05, 0.05, 0.05]])
    box.set_COM_shift(com_shift.expand((n_envs, 1, 3)) if n_envs > 0 else com_shift)

    # update noise and bias for accelerometer and gyroscope
    imu_noisy.set_noise((0.01, 0.01, 0.01, 0.02, 0.02, 0.02))
    imu_noisy.set_bias((0.01, 0.01, 0.01, 0.02, 0.02, 0.02))
    imu_noisy.set_jitter(0.001)

    for _ in range(10 - DELAY_STEPS):
        scene.step()

    true_imu_delayed_reading = imu_delayed.read_ground_truth()

    for _ in range(DELAY_STEPS):
        scene.step()

    assert_array_equal(imu_delayed.read().lin_acc, true_imu_delayed_reading.lin_acc)
    assert_array_equal(imu_delayed.read().ang_vel, true_imu_delayed_reading.ang_vel)

    # check that position offset affects linear acceleration
    imu.set_pos_offset((0.5, 0.0, 0.0))
    lin_acc_no_offset = imu.read().lin_acc
    scene.step()
    lin_acc_with_offset = imu.read().lin_acc
    assert not np.allclose(lin_acc_no_offset, lin_acc_with_offset, atol=0.2)
    imu.set_pos_offset((0.0, 0.0, 0.0))

    # let box collide with ground
    for _ in range(20):
        scene.step()

    assert_array_equal(imu.read_ground_truth().lin_acc, imu_delayed.read_ground_truth().lin_acc)
    assert_array_equal(imu.read_ground_truth().ang_vel, imu_delayed.read_ground_truth().ang_vel)

    with np.testing.assert_raises(AssertionError, msg="Angular velocity should not be zero due to COM shift"):
        assert_allclose(imu.read_ground_truth().ang_vel, 0.0, tol=tol)

    with np.testing.assert_raises(AssertionError, msg="Delayed data should not be equal to the ground truth data"):
        assert_array_equal(imu_delayed.read().lin_acc - imu_delayed.read_ground_truth().lin_acc, 0.0)

    zero_com_shift = torch.tensor([[0.0, 0.0, 0.0]])
    box.set_COM_shift(zero_com_shift.expand((n_envs, 1, 3)) if n_envs > 0 else zero_com_shift)
    quat_tensor = torch.tensor([0.0, 0.0, 0.0, 1.0])
    box.set_quat(quat_tensor.expand((n_envs, 4)) if n_envs > 0 else quat_tensor)

    # box is stationary on ground
    for _ in range(80):
        scene.step()

    assert_allclose(
        imu.read().lin_acc,
        expand_batch_dim((0.0, 0.0, -GRAVITY), n_envs),
        tol=5e-6,
    )
    assert_allclose(imu.read().ang_vel, expand_batch_dim((0.0, 0.0, 0.0), n_envs), tol=1e-5)

    # rotate IMU 90 deg around x axis means gravity should be along -y axis
    imu.set_quat_offset(gu.euler_to_quat((90.0, 0.0, 0.0)))
    imu.set_acc_axes_skew((0.0, 1.0, 0.0))
    scene.step()
    assert_allclose(imu.read().lin_acc, GRAVITY, tol=5e-6)
    imu.set_quat_offset((0.0, 0.0, 0.0, 1.0))
    imu.set_acc_axes_skew((0.0, 0.0, 0.0))

    scene.reset()

    assert_allclose(imu.read().lin_acc, 0.0, tol=gs.EPS)  # biased, but cache hasn't been updated yet
    assert_allclose(imu_delayed.read().lin_acc, 0.0, tol=gs.EPS)
    assert_allclose(imu_noisy.read().ang_vel, 0.0, tol=gs.EPS)

    imu.set_bias(BIAS + (0.0, 0.0, 0.0))
    scene.step()
    assert_allclose(imu.read().lin_acc, expand_batch_dim(BIAS, n_envs), tol=tol)


if __name__ == "__main__":
    gs.init(backend=gs.cpu)
    # test_imu_sensor(show_viewer=False, tol=1e-4, n_envs=0)
    test_imu_sensor(show_viewer=False, tol=1e-4, n_envs=2)
