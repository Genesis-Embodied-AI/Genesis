import numpy as np
import pytest
import torch

import genesis as gs
import genesis.utils.geom as gu

from .utils import assert_allclose, assert_array_equal


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_imu_sensor(show_viewer, tol, n_envs):
    """Test if the IMU sensor returns the correct data."""
    GRAVITY = -10.0
    DT = 1e-2
    BIAS = (0.1, 0.2, 0.3)
    DELAY_STEPS = 2
    MAG_FIELD = (0.3, 0.1, 0.5)  # arbitrary world magnetic field

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
            magnetic_field=MAG_FIELD,
        )
    )
    imu_delayed = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=box.idx,
            delay=DT * DELAY_STEPS,
            magnetic_field=MAG_FIELD,
        )
    )
    imu_noisy = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=box.idx,
            acc_cross_axis_coupling=0.01,
            gyro_cross_axis_coupling=(0.02, 0.03, 0.04),
            mag_cross_axis_coupling=0.01,
            acc_noise=(0.01, 0.01, 0.01),
            gyro_noise=(0.01, 0.01, 0.01),
            mag_noise=(0.01, 0.01, 0.01),
            acc_random_walk=(0.001, 0.001, 0.001),
            gyro_random_walk=(0.001, 0.001, 0.001),
            mag_random_walk=(0.001, 0.001, 0.001),
            delay=DT,
            magnetic_field=MAG_FIELD,
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
    assert_allclose(imu.read().mag, MAG_FIELD, tol=tol)
    assert_allclose(imu_noisy.read().lin_acc, 0.0, tol=1e-1)
    assert_allclose(imu_noisy.read().ang_vel, 0.0, tol=1e-1)
    assert_allclose(imu_noisy.read().mag, MAG_FIELD, tol=1e-1)

    # shift COM to induce angular velocity
    box.set_COM_shift([0.05, 0.05, 0.05])

    # update noise and bias for accelerometer, gyroscope and magnetometer
    imu_noisy.set_noise((0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05))
    imu_noisy.set_bias((0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05))
    imu_noisy.set_jitter(0.001)

    for _ in range(10 - DELAY_STEPS):
        scene.step()

    true_imu_delayed_reading = imu_delayed.read_ground_truth()

    for _ in range(DELAY_STEPS):
        scene.step()

    assert_array_equal(imu_delayed.read().lin_acc, true_imu_delayed_reading.lin_acc)
    assert_array_equal(imu_delayed.read().ang_vel, true_imu_delayed_reading.ang_vel)
    assert_array_equal(imu_delayed.read().mag, true_imu_delayed_reading.mag)

    # check that position offset affects linear acceleration
    imu.set_pos_offset((0.5, 0.0, 0.0))
    lin_acc_no_offset = imu.read().lin_acc
    scene.step()
    lin_acc_with_offset = imu.read().lin_acc
    with np.testing.assert_raises(AssertionError):
        assert_allclose(lin_acc_no_offset, lin_acc_with_offset, atol=0.2)
    imu.set_pos_offset((0.0, 0.0, 0.0))

    # let box collide with ground
    for _ in range(20):
        scene.step()

    assert_array_equal(imu.read_ground_truth().lin_acc, imu_delayed.read_ground_truth().lin_acc)
    assert_array_equal(imu.read_ground_truth().ang_vel, imu_delayed.read_ground_truth().ang_vel)
    assert_array_equal(imu.read_ground_truth().mag, imu_delayed.read_ground_truth().mag)

    with np.testing.assert_raises(AssertionError, msg="Angular velocity should not be zero due to COM shift"):
        assert_allclose(imu.read_ground_truth().ang_vel, 0.0, tol=tol)

    with np.testing.assert_raises(AssertionError, msg="Delayed accl data should not be equal to the ground truth data"):
        assert_array_equal(imu_delayed.read().lin_acc - imu_delayed.read_ground_truth().lin_acc, 0.0)

    with np.testing.assert_raises(AssertionError, msg="Delayed mag data should not be equal to the ground truth data"):
        assert_array_equal(imu_delayed.read().mag - imu_delayed.read_ground_truth().mag, 0.0)

    box.set_COM_shift((0.0, 0.0, 0.0))
    box.set_quat((0.0, 0.0, 0.0, 1.0))  # pi rotation around z-axis

    # wait for the box to be stationary on ground
    for _ in range(50):
        scene.step()

    assert_allclose(imu.read().lin_acc, (0.0, 0.0, -GRAVITY), tol=5e-6)
    assert_allclose(imu.read().ang_vel, (0.0, 0.0, 0.0), tol=1e-5)
    assert_allclose(imu.read().mag, (-MAG_FIELD[0], -MAG_FIELD[1], MAG_FIELD[2]), tol=tol)

    # rotate IMU 90 deg around x axis means gravity should be along -y axis
    imu.set_quat_offset(gu.euler_to_quat((90.0, 0.0, 0.0)))
    scene.step()
    assert_allclose(imu.read().lin_acc, (0.0, GRAVITY, 0.0), tol=5e-6)
    assert_allclose(imu.read().mag, (-MAG_FIELD[0], -MAG_FIELD[2], -MAG_FIELD[1]), tol=tol)

    imu.set_acc_cross_axis_coupling((0.0, 1.0, 0.0))
    scene.step()
    assert_allclose(imu.read().lin_acc, GRAVITY, tol=5e-6)

    scene.reset()
    box.set_dofs_velocity((1.0, 2.0, 3.0), dofs_idx_local=slice(3, None))
    scene.step()
    assert_allclose(imu.read_ground_truth().ang_vel, (1.0, 3.0, -2.0), tol=0.1)

    imu.set_quat_offset((1.0, 0.0, 0.0, 0.0))
    imu.set_acc_cross_axis_coupling((0.0, 0.0, 0.0))
    scene.reset()

    assert_allclose(imu.read().lin_acc, 0.0, tol=gs.EPS)  # biased, but cache hasn't been updated yet
    assert_allclose(imu_delayed.read().lin_acc, 0.0, tol=gs.EPS)
    assert_allclose(imu_noisy.read().ang_vel, 0.0, tol=gs.EPS)
    assert_allclose(imu_noisy.read().mag, 0.0, tol=gs.EPS)  # biased

    imu.set_bias(BIAS + 2 * (0.0, 0.0, 0.0))
    scene.step()
    assert_allclose(imu.read().lin_acc, BIAS, tol=tol)
    assert_allclose(imu.read().mag, MAG_FIELD, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_rigid_tactile_sensors_gravity_force(n_envs, show_viewer, tol):
    """Test if the sensor will detect the correct forces being applied on a falling box."""
    GRAVITY = -10.0
    BIAS = (0.1, 0.2, 0.3)
    NOISE = 0.01

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, GRAVITY),
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=show_viewer,
    )

    floor = scene.add_entity(morph=gs.morphs.Plane())

    # Add duck (with convex decomposition enabled) to offset geom index vs link index
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/duck.obj",
            scale=0.04,
            pos=(0.0, 1.0, 0.2),
            euler=(90, 0, 90),
        ),
    )

    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(1.0, 1.0, 1.0),  # volume = 1 m^3
            pos=(0.0, 0.0, 0.51),
        ),
        material=gs.materials.Rigid(
            rho=1.0,  # mass = 1.0 kg
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.0, 0.0, 1.0),
        ),
    )
    box_2 = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.2, 0.2, 0.2),  # volume = 0.008 m^3
            pos=(1.0, 0.0, 0.4),
        ),
        material=gs.materials.Rigid(
            rho=100.0,  # mass = 0.8 kg
        ),
        surface=gs.surfaces.Default(
            color=(0.0, 1.0, 0.0, 1.0),
        ),
    )
    box_3 = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.2, 0.2, 0.2),  # volume = 0.008 m^3
            pos=(1.0, 0.0, 0.61),
        ),
        material=gs.materials.Rigid(
            rho=25.0,  # mass = 0.2 kg
        ),
        surface=gs.surfaces.Default(
            color=(0.0, 0.0, 1.0, 1.0),
        ),
    )

    bool_sensor_floor = scene.add_sensor(
        gs.sensors.Contact(
            entity_idx=floor.idx,
        )
    )
    bool_sensor_box_2 = scene.add_sensor(
        gs.sensors.Contact(
            entity_idx=box_2.idx,
        )
    )
    force_sensor = scene.add_sensor(
        gs.sensors.ContactForce(
            entity_idx=box.idx,
        )
    )
    force_sensor_box_2 = scene.add_sensor(
        gs.sensors.ContactForce(
            entity_idx=box_2.idx,
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
    # Adding extra sensor sharing same dtype to force discontinuous memory layout for ground truth when batched
    scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=box.idx,
        )
    )

    scene.build(n_envs=n_envs)

    # Move CoM to get unbalanced forces on each contact points
    box_com_offset = (0.3, 0.1, 0.0)
    box.set_COM_shift(box_com_offset)

    # Rotate the box make sure the force is correctly computed in local frame
    box_2.set_dofs_position((np.pi / 2, np.pi / 4, np.pi / 2), dofs_idx_local=slice(3, None))

    # Add another cube on top of it make sure the forces are correctly aggregated
    box_3.set_dofs_position((-np.pi / 2, -np.pi / 4, -np.pi / 2), dofs_idx_local=slice(3, None))

    # Note that it is necessary to do a first step, because the initial state right after reset is not valid
    scene.step()

    # Make sure that box CoM is valid
    assert_allclose(box.get_links_pos(ref="root_com")[..., :2], box_com_offset[:2], tol=tol)

    assert not bool_sensor_floor.read().any(), "ContactSensor for floor should not detect any contact yet."
    assert not bool_sensor_box_2.read().any(), "ContactSensor for box_2 should not detect any contact yet."
    assert_allclose(force_sensor_noisy.read_ground_truth(), 0.0, tol=gs.EPS)
    assert_allclose(force_sensor.read(), force_sensor_noisy.read_ground_truth(), tol=gs.EPS)
    assert_allclose(force_sensor_noisy.read(), BIAS, tol=NOISE * 3)

    for _ in range(10):
        scene.step()

    assert bool_sensor_floor.read().all(), "ContactSensor for floor should detect contact with the ground"
    assert not bool_sensor_box_2.read().any(), "ContactSensor for box_2 should not detect any contact yet."
    assert_allclose(force_sensor_noisy.read(), force_sensor_noisy.read(), tol=gs.EPS)

    for _ in range(90):
        scene.step()

    assert bool_sensor_box_2.read().all(), "ContactSensor for box_2 should detect contact with the ground"

    # Moving force back in world frame because box is not perfectly flat on the ground due to CoM offset
    with np.testing.assert_raises(AssertionError):
        assert_allclose(box.get_quat(), 0.0, atol=tol)
    assert_allclose(
        gu.transform_by_quat(force_sensor_noisy.read_ground_truth(), box.get_quat()), (0.0, 0.0, -GRAVITY), tol=tol
    )

    # FIXME: Adding CoM offset on box is disturbing contact force computations on box_2 for some reason...
    assert_allclose(force_sensor_box_2.read_ground_truth(), (-0.8 * GRAVITY, 0.0, 0.0), tol=1e-2)

    assert_allclose(force_sensor_noisy.read()[..., :2], BIAS[:2], tol=NOISE * 3)
    assert_allclose(force_sensor_noisy.read()[..., 2], -GRAVITY / 2, tol=gs.EPS)


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
            fixed=True,
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
    # Note that the tolerance must be large because the sphere geometry is discretized
    assert_allclose(spherical_distances, RAYCAST_HEIGHT, tol=5e-3)

    # Check that we can read image from depth camera
    assert_array_equal(depth_camera.read_image().shape, batch_shape + NUM_RAYS_XY)
    # Note that the tolerance must be large because the sphere geometry is discretized
    assert_allclose(depth_camera.read_image(), RAYCAST_HEIGHT, tol=5e-3)

    # Simulate for a while and check again that the ray is casted properly
    offset = torch.from_numpy(np.random.rand(*batch_shape, 3)).to(dtype=gs.tc_float, device=gs.device)
    for entity in (grid_sensor, obstacle_1, obstacle_2):
        pos = entity.get_pos() + offset
        if entity is obstacle_2:
            pos[..., 2] = BOX_SIZE / 2
        entity.set_pos(pos)
    if show_viewer:
        scene.visualizer.update(force=True)
    grid_sensor_pos = grid_sensor.get_pos().clone()
    for _ in range(60):
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


@pytest.mark.required
def test_lidar_bvh_parallel_env(show_viewer, tol):
    """Verify each environment receives a different lidar distance when geometries differ."""
    scene = gs.Scene(
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=(1,),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1, -5, 3),
            camera_lookat=(1, 0.5, 0),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())

    sensor_mount = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.5),
            fixed=True,
            collision=False,
        )
    )
    obstacle_1 = scene.add_entity(
        gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(1.0, 0.0, 0.5),
            fixed=True,
        ),
    )
    obstacle_2 = scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.4, 0.4),
            pos=(1.0, 0.0, 0.5),
            fixed=True,
        ),
    )

    lidar = scene.add_sensor(
        gs.sensors.Lidar(
            entity_idx=sensor_mount.idx,
            pattern=gs.options.sensors.SphericalPattern(
                n_points=(1, 1),
                fov=(0.0, 0.0),
            ),
            max_range=5.0,
            draw_debug=show_viewer,
            debug_ray_start_color=(0.0, 0.0, 0.0, 0.0),
            debug_ray_hit_color=(1.0, 0.0, 0.0, 1.0),
        )
    )

    scene.build(n_envs=2)

    sensor_positions = np.array([[0.0, 0.0, 0.5], [0.0, 1.0, 0.5]], dtype=gs.np_float)
    obstacle_1_positions = np.array([[1.1, 0.0, 0.5], [2.5, 1.0, 0.5]], dtype=gs.np_float)
    obstacle_2_positions = np.array([[1.4, 0.0, 0.5], [2.2, 1.0, 0.5]], dtype=gs.np_float)
    sensor_mount.set_pos(sensor_positions)
    obstacle_1.set_pos(obstacle_1_positions)
    obstacle_2.set_pos(obstacle_2_positions)

    scene.step()

    distances = lidar.read().distances
    assert distances.shape == (2, 1, 1)
    lidar_distances = distances[:, 0, 0]

    front_positions = np.minimum(obstacle_1_positions[:, 0] - 0.1, obstacle_2_positions[:, 0] - 0.025)
    expected_distances = front_positions - sensor_positions[:, 0]
    assert_allclose(lidar_distances, expected_distances, tol=tol)


@pytest.mark.required
def test_lidar_cache_offset_parallel_env(show_viewer, tol):
    scene = gs.Scene(
        show_viewer=show_viewer,
    )

    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    cube = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 1.0),
            pos=(0.0, 0.0, 0.5),
        ),
    )

    sensors = [
        scene.add_sensor(
            gs.sensors.Raycaster(
                pattern=gs.sensors.raycaster.SphericalPattern(
                    n_points=(2, 2),
                ),
                entity_idx=cube.idx,
                return_world_frame=False,
            )
        ),
        scene.add_sensor(
            gs.sensors.Raycaster(
                pattern=gs.sensors.raycaster.SphericalPattern(
                    n_points=(2, 2),
                ),
                entity_idx=cube.idx,
                return_world_frame=False,
            )
        ),
        scene.add_sensor(
            gs.sensors.Raycaster(
                pattern=gs.sensors.raycaster.SphericalPattern(
                    n_points=(2, 2),
                ),
                entity_idx=cube.idx,
                return_world_frame=False,
            )
        ),
    ]

    scene.build()

    scene.step()
    for sensor in sensors:
        sensor_data = sensor.read()
        assert (sensor_data.distances > gs.EPS).any()
        assert (sensor_data.points.abs() > gs.EPS).any()
