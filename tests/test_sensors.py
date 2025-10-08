import numpy as np
import pytest
import torch

import genesis as gs

from .utils import assert_allclose, assert_array_equal, rgb_array_to_png_bytes


def expand_batch_dim(values: tuple[float, ...], n_envs: int) -> tuple[float, ...] | np.ndarray:
    """Helper function to expand expected values for n_envs dimension."""
    if n_envs == 0:
        return values
    return np.tile(np.array(values), (n_envs,) + (1,) * len(values))


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
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
    imu_skewed = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=box.idx,
            acc_axes_skew=(0.0, 0.0, 1.0),
        )
    )

    scene.build(n_envs=n_envs)

    # box is in freefall
    for _ in range(10):
        scene.step()

    # IMU should calculate "classical linear acceleration" using the local frame without accounting for gravity
    # acc_classical_lin_z = - theta_dot ** 2 - cos(theta) * g
    assert_allclose(imu_biased.read().lin_acc, expand_batch_dim(BIAS, n_envs), tol=tol)
    assert_allclose(imu_biased.read().ang_vel, expand_batch_dim(BIAS, n_envs), tol=tol)
    assert_allclose(imu_delayed.read().lin_acc, 0.0, tol=tol)
    assert_allclose(imu_delayed.read().ang_vel, 0.0, tol=tol)
    assert_allclose(imu_noisy.read().lin_acc, 0.0, tol=1e-1)
    assert_allclose(imu_noisy.read().ang_vel, 0.0, tol=1e-1)

    # shift COM to induce angular velocity
    com_shift = torch.tensor([[0.1, 0.1, 0.1]])
    box.set_COM_shift(com_shift.expand((n_envs, 1, 3)) if n_envs > 0 else com_shift)

    # update noise and bias for accelerometer and gyroscope
    imu_noisy.set_acc_noise([0.01, 0.01, 0.01])
    imu_noisy.set_gyro_noise([0.02, 0.02, 0.02])
    imu_noisy.set_bias([0.01, 0.01, 0.01, 0.02, 0.02, 0.02])
    imu_noisy.set_jitter(0.001)

    for _ in range(10 - DELAY_STEPS):
        scene.step()

    true_imu_delayed_reading = imu_delayed.read_ground_truth()

    for _ in range(DELAY_STEPS):
        scene.step()

    assert_array_equal(imu_delayed.read().lin_acc, true_imu_delayed_reading.lin_acc)
    assert_array_equal(imu_delayed.read().ang_vel, true_imu_delayed_reading.ang_vel)

    # let box collide with ground
    for _ in range(20):
        scene.step()

    assert_array_equal(imu_biased.read_ground_truth().lin_acc, imu_delayed.read_ground_truth().lin_acc)
    assert_array_equal(imu_biased.read_ground_truth().ang_vel, imu_delayed.read_ground_truth().ang_vel)

    with np.testing.assert_raises(AssertionError, msg="Angular velocity should not be zero due to COM shift"):
        assert_allclose(imu_biased.read_ground_truth().ang_vel, 0.0, tol=tol)

    with np.testing.assert_raises(AssertionError, msg="Delayed data should not be equal to the ground truth data"):
        assert_array_equal(imu_delayed.read().lin_acc - imu_delayed.read_ground_truth().lin_acc, 0.0)

    zero_com_shift = torch.tensor([[0.0, 0.0, 0.0]])
    box.set_COM_shift(zero_com_shift.expand((n_envs, 1, 3)) if n_envs > 0 else zero_com_shift)

    # box is stationary on ground
    for _ in range(80):
        scene.step()

    assert_allclose(imu_skewed.read().lin_acc, -GRAVITY, tol=5e-6)
    assert_allclose(
        imu_biased.read().lin_acc,
        expand_batch_dim((BIAS[0], BIAS[1], BIAS[2] - GRAVITY), n_envs),
        tol=5e-6,
    )
    assert_allclose(imu_biased.read().ang_vel, expand_batch_dim(BIAS, n_envs), tol=1e-5)

    scene.reset()

    assert_allclose(imu_biased.read().lin_acc, 0.0, tol=gs.EPS)  # biased, but cache hasn't been updated yet
    assert_allclose(imu_delayed.read().lin_acc, 0.0, tol=gs.EPS)
    assert_allclose(imu_noisy.read().ang_vel, 0.0, tol=gs.EPS)

    scene.step()
    assert_allclose(imu_biased.read().lin_acc, expand_batch_dim(BIAS, n_envs), tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_rigid_tactile_sensors_gravity_force(show_viewer, tol, n_envs):
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

    floor = scene.add_entity(morph=gs.morphs.Plane())

    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(1.0, 1.0, 1.0),  # volume = 1 m^3
            pos=(0.0, 0.0, 1.1),
        ),
        material=gs.materials.Rigid(rho=1.0),  # mass = 1 kg
    )

    bool_sensor_floor = scene.add_sensor(
        gs.sensors.Contact(
            entity_idx=floor.idx,
        )
    )
    bool_sensor_box = scene.add_sensor(
        gs.sensors.Contact(
            entity_idx=box.idx,
        )
    )
    force_sensor = scene.add_sensor(
        gs.sensors.ContactForce(
            entity_idx=box.idx,
        )
    )
    force_sensor_noisy = scene.add_sensor(
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

    scene.build(n_envs=n_envs)

    scene.step()

    assert not bool_sensor_floor.read().any(), "ContactSensor for floor should not detect any contact yet."
    assert not bool_sensor_box.read().any(), "ContactSensor for box should not detect any contact yet."
    assert_allclose(
        force_sensor_noisy.read_ground_truth(),
        0.0,
        tol=gs.EPS,
        err_msg="noisy ContactForceSensor ground truth reading should be zero before contact.",
    )
    assert_allclose(
        force_sensor.read(),
        force_sensor_noisy.read_ground_truth(),
        tol=gs.EPS,
        err_msg="noisy ContactForceSensor ground truth reading should equal noise ContactForceSensor reading.",
    )
    assert_allclose(
        force_sensor_noisy.read(),
        expand_batch_dim(BIAS, n_envs),
        tol=NOISE * 3,
        err_msg="noisy ContactForceSensor should only read bias and small amount of noise before contact.",
    )

    for _ in range(120):
        scene.step()

    assert bool_sensor_box.read().all(), "Sensor should detect contact with the ground"
    assert_allclose(
        force_sensor_noisy.read_ground_truth(),
        expand_batch_dim((0.0, 0.0, -GRAVITY), n_envs),
        tol=tol,
        err_msg="ContactForceSensor ground truth should be equal to -gravity (normal) force.",
    )
    assert_allclose(
        force_sensor_noisy.read(),
        expand_batch_dim((BIAS[0], BIAS[1], -GRAVITY / 2), n_envs),
        tol=NOISE * 10,
        err_msg="ContactForceSensor should read bias and noise and -gravity (normal) force clipped by max_force.",
    )


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_raycaster_hits(show_viewer, tol, n_envs):
    """Test if the Raycaster sensor with GridPattern rays pointing to ground returns the correct distance."""
    EXPECTED_DISTANCE = 1.2
    NUM_RAYS_XY = 3
    BOX_HEIGHT = 0.2
    SPHERE_POS = (4.0, 0.0, 1.0)
    RAYCAST_GRID_SIZE = 0.5

    scene = gs.Scene(
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=show_viewer,
    )

    scene.add_entity(gs.morphs.Plane())

    box_obstacle = scene.add_entity(
        gs.morphs.Box(
            size=(RAYCAST_GRID_SIZE / 2.0, RAYCAST_GRID_SIZE / 2.0, BOX_HEIGHT),
            # pos=(0.0, 0.0, -BOX_HEIGHT),  # init below ground to not interfere with first raycast
            pos=(RAYCAST_GRID_SIZE, RAYCAST_GRID_SIZE, EXPECTED_DISTANCE / 2.0 + BOX_HEIGHT / 2.0),
        ),
    )
    grid_sensor_box = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, EXPECTED_DISTANCE + BOX_HEIGHT),
            fixed=True,
        ),
    )
    grid_raycaster = scene.add_sensor(
        gs.sensors.Raycaster(
            pattern=gs.sensors.raycaster.GridPattern(
                resolution=1.0 / (NUM_RAYS_XY - 1.0),
                size=(1.0, 1.0),
                direction=(0.0, 0.0, -1.0),  # pointing downwards to ground
            ),
            entity_idx=grid_sensor_box.idx,
            pos_offset=(0.0, 0.0, -BOX_HEIGHT),
            return_world_frame=True,
            draw_debug=True,
        )
    )

    spherical_sensor = scene.add_entity(
        gs.morphs.Sphere(
            radius=EXPECTED_DISTANCE,
            pos=SPHERE_POS,
            fixed=True,
        ),
    )
    spherical_raycaster = scene.add_sensor(
        gs.sensors.Raycaster(
            pattern=gs.sensors.raycaster.SphericalPattern(
                n_points=(NUM_RAYS_XY, NUM_RAYS_XY),
            ),
            entity_idx=spherical_sensor.idx,
            return_world_frame=False,
        )
    )

    scene.build(n_envs=n_envs)

    scene.step()

    grid_hits = grid_raycaster.read().points
    grid_distances = grid_raycaster.read().distances
    spherical_distances = spherical_raycaster.read().distances

    expected_shape = (NUM_RAYS_XY, NUM_RAYS_XY) if n_envs == 0 else (n_envs, NUM_RAYS_XY, NUM_RAYS_XY)
    assert grid_distances.shape == spherical_distances.shape == expected_shape

    grid_distance_min = grid_distances.min()
    assert grid_distances.min() < EXPECTED_DISTANCE - tol, "Raycaster grid pattern should have hit obstacle"
    ground_hit_mask = grid_distances > grid_distance_min + tol
    grid_hits = grid_hits[ground_hit_mask]
    grid_distances = grid_distances[ground_hit_mask]

    assert_allclose(
        grid_hits[..., 2],
        0.0,
        tol=tol,
        err_msg="Raycaster grid pattern should hit ground (z≈0)",
    )
    assert_allclose(
        grid_distances,
        EXPECTED_DISTANCE,
        tol=tol,
        err_msg=f"Raycaster grid pattern should measure {EXPECTED_DISTANCE}m to ground plane",
    )
    assert_allclose(
        spherical_distances,
        EXPECTED_DISTANCE,
        tol=1e-2,  # since sphere mesh is discretized, we need a larger tolerance here
        err_msg=f"Raycaster spherical pattern should measure {EXPECTED_DISTANCE}m to the sphere around it",
    )
