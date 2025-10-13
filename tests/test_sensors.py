import numpy as np
import pytest
import torch

import genesis as gs

from .utils import assert_allclose, assert_array_equal


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
def test_raycaster_hits(show_viewer, n_envs):
    """Test if the Raycaster sensor with GridPattern rays pointing to ground returns the correct distance."""
    NUM_RAYS_XY = (3, 5)
    SPHERE_POS = (2.5, 0.5, 1.0)
    BOX_SIZE = 0.05
    RAYCAST_BOX_SIZE = 0.1
    RAYCAST_GRID_SIZE_X = 1.0
    RAYCAST_HEIGHT = 1.0

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(-3.0, RAYCAST_GRID_SIZE_X * (NUM_RAYS_XY[1] / NUM_RAYS_XY[0]), 2 * RAYCAST_HEIGHT),
            camera_lookat=(1.5, RAYCAST_GRID_SIZE_X * (NUM_RAYS_XY[1] / NUM_RAYS_XY[0]), RAYCAST_HEIGHT),
        ),
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=(0,),
            env_separate_rigid=False,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())

    spherical_sensor = scene.add_entity(
        gs.morphs.Sphere(
            radius=RAYCAST_HEIGHT,
            pos=SPHERE_POS,
            fixed=True,
        ),
    )
    spherical_raycaster = scene.add_sensor(
        gs.sensors.Raycaster(
            pattern=gs.sensors.raycaster.SphericalPattern(
                n_points=NUM_RAYS_XY,
            ),
            entity_idx=spherical_sensor.idx,
            return_world_frame=False,
            draw_debug=show_viewer,
            debug_ray_start_color=(0.0, 0.0, 0.0, 0.0),
            debug_ray_hit_color=(1.0, 0.0, 0.0, 1.0),
        )
    )

    grid_sensor = scene.add_entity(
        gs.morphs.Box(
            size=(RAYCAST_BOX_SIZE, RAYCAST_BOX_SIZE, RAYCAST_BOX_SIZE),
            pos=(0.0, 0.0, RAYCAST_HEIGHT + 0.5 * RAYCAST_BOX_SIZE),
            collision=False,
            fixed=False,
        ),
    )
    grid_res = RAYCAST_GRID_SIZE_X / (NUM_RAYS_XY[0] - 1)
    grid_size_y = grid_res * (NUM_RAYS_XY[1] - 1)
    grid_raycaster = scene.add_sensor(
        gs.sensors.Raycaster(
            pattern=gs.sensors.raycaster.GridPattern(
                resolution=grid_res,
                size=(RAYCAST_GRID_SIZE_X, grid_size_y),
                direction=(0.0, 0.0, -1.0),  # pointing downwards to ground
            ),
            entity_idx=grid_sensor.idx,
            pos_offset=(0.0, 0.0, -0.5 * RAYCAST_BOX_SIZE),
            return_world_frame=True,
            draw_debug=show_viewer,
            debug_ray_start_color=(0.0, 0.0, 0.0, 0.0),
            debug_ray_hit_color=(0.0, 1.0, 0.0, 1.0),
        )
    )
    depth_camera = scene.add_sensor(
        gs.sensors.DepthCamera(
            pattern=gs.sensors.raycaster.DepthCameraPattern(
                res=NUM_RAYS_XY[::-1],
            ),
            entity_idx=spherical_sensor.idx,
            draw_debug=show_viewer,
            debug_ray_start_color=(0.0, 0.0, 0.0, 0.0),
            debug_ray_hit_color=(0.0, 0.0, 1.0, 1.0),
        ),
    )

    obstacle_1 = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(grid_res, grid_res, 0.5 * BOX_SIZE),
        ),
    )
    obstacle_2 = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(RAYCAST_GRID_SIZE_X, grid_size_y, RAYCAST_HEIGHT + RAYCAST_BOX_SIZE + BOX_SIZE),
        ),
    )

    # Build the simulation and do one step
    scene.build(n_envs=n_envs)
    batch_shape = (n_envs,) if n_envs > 0 else ()

    # Validate grid raycast
    for obstacle_pos, sensor_pos, hit_ij in (
        (None, None, (-1, -2)),
        ((grid_res, grid_res, BOX_SIZE), None, (-1, -2)),
        (None, (*(grid_res * (e - 2) for e in NUM_RAYS_XY), RAYCAST_HEIGHT + 0.5 * RAYCAST_BOX_SIZE), (1, 0)),
    ):
        # Update obstacle and/or sensor position if necessary
        if obstacle_pos is not None:
            obstacle_1.set_pos(np.tile(obstacle_pos, (*batch_shape, 1)))
        obstacle_pos = obstacle_1.get_pos()
        if sensor_pos is not None:
            grid_sensor.set_pos(np.tile(sensor_pos, (*batch_shape, 1)))
        scene.sim._sensor_manager.step()
        if show_viewer:
            scene.visualizer.update(force=True)

        # Fetch updated sensor data
        grid_hits = grid_raycaster.read().points
        grid_distances = grid_raycaster.read().distances
        assert grid_distances.shape == (*batch_shape, *NUM_RAYS_XY)

        # Check hits
        grid_sensor_origin = grid_sensor.get_pos()
        x = torch.linspace(-0.5, 0.5, NUM_RAYS_XY[0]) * RAYCAST_GRID_SIZE_X + grid_sensor_origin[..., [0]]
        y = torch.linspace(-0.5, 0.5, NUM_RAYS_XY[1]) * grid_size_y + grid_sensor_origin[..., [1]]
        # xg, yg = torch.meshgrid(x, y, indexing="ij")
        xg = x.unsqueeze(-1).expand((*batch_shape, -1, NUM_RAYS_XY[1]))
        yg = y.unsqueeze(-2).expand((*batch_shape, NUM_RAYS_XY[0], -1))
        zg = torch.zeros((*batch_shape, *NUM_RAYS_XY))
        zg[(..., *hit_ij)] = obstacle_pos[..., 2] + 0.5 * BOX_SIZE
        grid_hits_ref = torch.stack([xg, yg, zg], dim=-1)
        assert_allclose(grid_hits, grid_hits_ref, tol=gs.EPS)

        # Check distances
        grid_distances_ref = torch.full((*batch_shape, *NUM_RAYS_XY), RAYCAST_HEIGHT)
        grid_distances_ref[(..., *hit_ij)] = RAYCAST_HEIGHT - obstacle_pos[..., 2] - 0.5 * BOX_SIZE
        assert_allclose(grid_distances, grid_distances_ref, tol=gs.EPS)

    # Validate spherical raycast
    spherical_distances = spherical_raycaster.read().distances
    assert spherical_distances.shape == (*batch_shape, *NUM_RAYS_XY)
    # Note that the tolerance must be large bevcause the sphere geometry is discretized
    assert_allclose(spherical_distances, RAYCAST_HEIGHT, tol=5e-3)

    # Check that we can read image from depth camera
    assert_array_equal(depth_camera.read_image().shape, batch_shape + NUM_RAYS_XY)
    # Note that the tolerance must be large bevcause the sphere geometry is discretized
    assert_allclose(depth_camera.read_image(), RAYCAST_HEIGHT, tol=5e-3)

    # Simulate for a while and check again that the ray is casted properly
    offset = torch.from_numpy(np.random.rand(*batch_shape, 3)).to(dtype=gs.tc_float, device=gs.device)
    for entity in (grid_sensor, obstacle_1, obstacle_2):
        entity.set_pos(entity.get_pos() + offset)
    grid_sensor_pos = grid_sensor.get_pos()
    for _ in range(100):
        scene.step()
    grid_sensor.set_pos(grid_sensor_pos)
    scene.sim._sensor_manager.step()
    if show_viewer:
        scene.visualizer.update(force=True)

    grid_distances = grid_raycaster.read().distances
    grid_distances_ref = torch.full((*batch_shape, *NUM_RAYS_XY), RAYCAST_HEIGHT)
    grid_distances_ref[(..., -1, -2)] = RAYCAST_HEIGHT - BOX_SIZE
    grid_distances_ref[(..., *hit_ij)] = RAYCAST_HEIGHT - BOX_SIZE
    grid_distances_ref += offset[..., 2].reshape((*(-1 for e in batch_shape), 1, 1))
    assert_allclose(grid_distances, grid_distances_ref, tol=1e-3)
