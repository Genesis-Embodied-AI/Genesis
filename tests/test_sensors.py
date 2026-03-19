import importlib
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

import genesis as gs
import genesis.utils.geom as gu

from .utils import assert_allclose, assert_equal

# ------------------------------------------------------------------------------------------
# -------------------------------- Lazy Sensor Discovery -----------------------------------
# ------------------------------------------------------------------------------------------


@pytest.mark.required
def test_lazy_sensor_discovery(show_viewer, tmp_path):
    """Test that add_sensor auto-discovers sensor classes from the options class's sibling modules."""
    from genesis.engine.sensors.camera import RasterizerCameraSensor
    from genesis.engine.sensors.contact_force import ContactSensor
    from genesis.engine.sensors.depth_camera import DepthCameraSensor
    from genesis.engine.sensors.imu import IMUSensor
    from genesis.engine.sensors.sensor_manager import SensorManager

    # Verify built-in registrations resolve to the exact sensor classes
    assert SensorManager.SENSOR_TYPES_MAP[gs.sensors.Contact] is ContactSensor
    assert SensorManager.SENSOR_TYPES_MAP[gs.sensors.IMU] is IMUSensor
    assert SensorManager.SENSOR_TYPES_MAP[gs.sensors.RasterizerCameraOptions] is RasterizerCameraSensor
    # DepthCamera inherits from Raycaster without re-parameterizing, registered only by sensor side
    assert SensorManager.SENSOR_TYPES_MAP[gs.sensors.DepthCamera] is DepthCameraSensor

    # Create a fake plugin package in a temp directory
    pkg_dir = tmp_path / "fake_sensor_plugin"
    pkg_dir.mkdir()

    (pkg_dir / "__init__.py").write_text("")

    (pkg_dir / "options.py").write_text(
        textwrap.dedent(
            """\
        from genesis.options.sensors.options import SensorOptions

        class FakeSensorOptions(SensorOptions["FakeSensor"]):
            pass
        """
        )
    )

    (pkg_dir / "sensor.py").write_text(
        textwrap.dedent(
            """\
        from dataclasses import dataclass

        import genesis as gs
        from genesis.engine.sensors.base_sensor import Sensor, SharedSensorMetadata

        from .options import FakeSensorOptions


        @dataclass
        class FakeSensorMetadata(SharedSensorMetadata):
            pass


        class FakeSensor(Sensor[FakeSensorOptions, FakeSensorMetadata]):
            def _get_return_format(self):
                return (1,)

            @classmethod
            def _get_cache_dtype(cls):
                return gs.tc_float

            @classmethod
            def _update_shared_ground_truth_cache(cls, metadata, cache):
                pass

            @classmethod
            def _update_shared_cache(cls, metadata, gt_cache, cache, buffer):
                pass

            @classmethod
            def reset(cls, metadata, shared_ground_truth_cache, envs_idx):
                pass

            def build(self):
                pass
        """
        )
    )

    sys.path.insert(0, str(tmp_path))
    try:
        # Import ONLY the options module (not the sensor module)
        options_mod = importlib.import_module("fake_sensor_plugin.options")
        FakeSensorOptions = options_mod.FakeSensorOptions

        # Verify it's not yet registered
        assert FakeSensorOptions not in SensorManager.SENSOR_TYPES_MAP

        # Trigger lazy discovery via resolve
        sensor_cls = SensorManager._resolve_sensor_cls(FakeSensorOptions)
        assert sensor_cls.__name__ == "FakeSensor"

        # Now it should be registered
        assert SensorManager.SENSOR_TYPES_MAP[FakeSensorOptions] is sensor_cls

        # Verify it works end-to-end with a scene
        scene = gs.Scene(show_viewer=show_viewer)
        scene.add_entity(gs.morphs.Plane())
        sensor = scene.add_sensor(FakeSensorOptions())
        scene.build()
        scene.step()
        data = sensor.read()
        assert data.shape[-1] == 1
    finally:
        sys.path.remove(str(tmp_path))
        for mod_name in list(sys.modules):
            if mod_name.startswith("fake_sensor_plugin"):
                del sys.modules[mod_name]
        SensorManager.SENSOR_TYPES_MAP.pop(FakeSensorOptions, None)


# ------------------------------------------------------------------------------------------
# -------------------------------------- IMU Sensors ---------------------------------------
# ------------------------------------------------------------------------------------------


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
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
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

    assert_equal(imu_delayed.read().lin_acc, true_imu_delayed_reading.lin_acc)
    assert_equal(imu_delayed.read().ang_vel, true_imu_delayed_reading.ang_vel)
    assert_equal(imu_delayed.read().mag, true_imu_delayed_reading.mag)

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

    assert_equal(imu.read_ground_truth().lin_acc, imu_delayed.read_ground_truth().lin_acc)
    assert_equal(imu.read_ground_truth().ang_vel, imu_delayed.read_ground_truth().ang_vel)
    assert_equal(imu.read_ground_truth().mag, imu_delayed.read_ground_truth().mag)

    with np.testing.assert_raises(AssertionError, msg="Angular velocity should not be zero due to COM shift"):
        assert_allclose(imu.read_ground_truth().ang_vel, 0.0, tol=tol)

    with np.testing.assert_raises(AssertionError, msg="Delayed accl data should not be equal to the ground truth data"):
        assert_equal(imu_delayed.read().lin_acc - imu_delayed.read_ground_truth().lin_acc, 0.0)

    with np.testing.assert_raises(AssertionError, msg="Delayed mag data should not be equal to the ground truth data"):
        assert_equal(imu_delayed.read().mag - imu_delayed.read_ground_truth().mag, 0.0)

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


# ------------------------------------------------------------------------------------------
# ------------------------------------ Contact Sensors -------------------------------------
# ------------------------------------------------------------------------------------------


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_contact_sensors_gravity_force(n_envs, show_viewer, tol):
    """Test if the sensor will detect the correct forces being applied on a falling box."""
    GRAVITY = -10.0
    BIAS = (0.1, 0.2, 0.3)
    NOISE = 0.01

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, GRAVITY),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
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


# ------------------------------------------------------------------------------------------
# ------------------------------------ Raycast Sensors -------------------------------------
# ------------------------------------------------------------------------------------------


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
    assert_equal(depth_camera.read_image().shape, batch_shape + NUM_RAYS_XY)
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
    grid_sensor_pos = grid_sensor.get_pos()
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


# ------------------------------------------------------------------------------------------
# -------------------------------------- Kinematic Tactile Sensors ---------------------------------------
# ------------------------------------------------------------------------------------------


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_temperature_grid_sensor_contact_and_reset(show_viewer, tol, n_envs):
    """After build, grid is at base temp. Hot box on center heats center above corner; cold box cools it. Move away -> near base; reset -> exactly base."""
    BOX_SIZE = 0.06
    PLATFORM_SIZE = 0.2
    FAR_POS = (PLATFORM_SIZE * 1.5, PLATFORM_SIZE * 1.5, PLATFORM_SIZE * 1.5)
    GRID_SIZE = (3, 3, 1)
    GRID_CENTER = (GRID_SIZE[0] // 2, GRID_SIZE[1] // 2, GRID_SIZE[2] // 2)
    BASE_TEMP = 22.0
    DIFF_TEMP = 0.5

    scene = gs.Scene(show_viewer=show_viewer)
    scene.add_entity(gs.morphs.Plane())
    platform = scene.add_entity(
        gs.morphs.Box(
            size=(PLATFORM_SIZE, PLATFORM_SIZE, PLATFORM_SIZE),
            pos=(0.0, 0.0, PLATFORM_SIZE / 2),
            fixed=True,
        )
    )
    hot_box = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(0.0, 0.0, PLATFORM_SIZE + BOX_SIZE / 2),
        )
    )
    cold_box = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=FAR_POS,
        ),
    )
    TemperatureProperties = gs.sensors.TemperatureProperties
    sensor = scene.add_sensor(
        gs.sensors.TemperatureGrid(
            ambient_temperature=BASE_TEMP,
            convection_coefficient=0.0,
            simulate_all_link_temperatures=False,
            entity_idx=platform.idx,
            grid_size=GRID_SIZE,
            properties_dict={
                platform.base_link_idx: TemperatureProperties(
                    base_temperature=BASE_TEMP,
                    conductivity=400.0,
                    density=2000.0,
                    specific_heat=1.0,
                    emissivity=0.95,
                ),
                hot_box.base_link_idx: TemperatureProperties(
                    base_temperature=BASE_TEMP + 100.0,
                    conductivity=200.0,
                    density=3000.0,
                    specific_heat=1.0,
                    emissivity=0.1,
                ),
                # default properties; should apply to the cold box
                -1: TemperatureProperties(
                    base_temperature=BASE_TEMP - 100.0,
                    conductivity=150.0,
                    density=8000.0,
                    specific_heat=1.0,
                    emissivity=0.2,
                ),
            },
        )
    )
    scene.build(n_envs=n_envs)

    # After build, all cells at base temperature
    assert_allclose(sensor.read_ground_truth(), BASE_TEMP, tol=tol)

    # Hot box on center
    hot_box.set_pos((0.0, 0.0, PLATFORM_SIZE + BOX_SIZE / 2))
    for _ in range(50):
        scene.step()
    data = sensor.read()
    assert (data > BASE_TEMP + DIFF_TEMP).all(), f"Hot box should have heated the grid by at least {DIFF_TEMP}°C"
    assert (data[..., GRID_CENTER[0], GRID_CENTER[1], GRID_CENTER[2]] > data[0, 0, 0]).all(), (
        "Center cell should be hotter than corner"
    )

    # Reset: exactly base temperature everywhere
    scene.reset()
    assert_allclose(sensor.read_ground_truth(), BASE_TEMP, tol=tol)

    # Cold box on center
    hot_box.set_pos(FAR_POS)
    cold_box.set_pos((0.0, 0.0, PLATFORM_SIZE + BOX_SIZE / 2))
    for _ in range(50):
        scene.step()
    data = sensor.read()
    assert (data < BASE_TEMP - DIFF_TEMP).all(), f"Cold box should have cooled the grid by at least {DIFF_TEMP}°C"
    assert (data[..., GRID_CENTER[0], GRID_CENTER[1], GRID_CENTER[2]] < data[0, 0, 0]).all(), (
        "Center cell should be colder than corner"
    )

    # Move both away; step until grid returns near base
    hot_box.set_pos(FAR_POS)
    cold_box.set_pos((-FAR_POS[0], -FAR_POS[1], FAR_POS[2]))
    for _ in range(150):
        scene.step()
    data = sensor.read()
    assert_allclose(data, BASE_TEMP, tol=5e-2)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_temperature_grid_simulate_all_link_temps(show_viewer, tol, n_envs):
    """With simulate_all_link_temperatures=True, two boxes in contact exchange heat."""
    BOX_SIZE = 0.06
    BASE_TEMP = 22.0
    HOT_BASE = BASE_TEMP + 80.0
    COLD_BASE = BASE_TEMP - 80.0

    scene = gs.Scene(show_viewer=show_viewer)
    scene.add_entity(gs.morphs.Plane())
    hot_box = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(0.0, 0.0, BOX_SIZE),
        )
    )
    cold_box = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(0.0, 0.0, BOX_SIZE * 2 + 0.001),
        )
    )
    hot_link_idx = hot_box.base_link_idx
    cold_link_idx = cold_box.base_link_idx
    sensor1 = scene.add_sensor(
        gs.sensors.TemperatureGrid(
            entity_idx=hot_box.idx,
            grid_size=(1, 1, 1),
            ambient_temperature=BASE_TEMP,
            properties_dict={
                hot_link_idx: gs.sensors.TemperatureProperties(
                    base_temperature=HOT_BASE,
                    conductivity=200.0,
                    density=2000.0,
                    specific_heat=1.0,
                    emissivity=0.1,
                ),
                cold_link_idx: gs.sensors.TemperatureProperties(
                    base_temperature=COLD_BASE,
                    conductivity=200.0,
                    density=2000.0,
                    specific_heat=1.0,
                    emissivity=0.1,
                ),
            },
            simulate_all_link_temperatures=True,
        )
    )
    sensor2 = scene.add_sensor(
        gs.sensors.TemperatureGrid(
            entity_idx=cold_box.idx,
            grid_size=(1, 1, 1),
        )
    )
    scene.build(n_envs=n_envs)

    link_temps = sensor1.link_temperatures  # (n_envs, n_links)

    assert_equal(link_temps[..., hot_link_idx], HOT_BASE)
    assert_equal(link_temps[..., cold_link_idx], COLD_BASE)

    cold_box.set_pos((0.0, 0.0, BOX_SIZE / 2))
    for _ in range(100):
        scene.step()

    assert_equal(sensor1.link_temperatures, sensor2.link_temperatures)

    assert (link_temps[..., hot_link_idx] < HOT_BASE - 1.0).all(), "Hot box link should have cooled"
    assert (link_temps[..., cold_link_idx] > COLD_BASE + 1.0).all(), "Cold box link should have heated up"

    assert_allclose(torch.mean(sensor1.read()), link_temps[..., hot_link_idx], tol=2e-2)
    assert_allclose(torch.mean(sensor2.read()), link_temps[..., cold_link_idx], tol=2e-2)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_kinematic_contact_probe_box_support(show_viewer, tol, n_envs):
    """Test KinematicContactProbe for a box resting on the ground and a fixed sphere on top of it."""
    BOX_SIZE = 0.5
    PROBE_RADIUS = 0.05
    PENETRATION = 0.02
    STIFFNESS = 100.0
    SPHERE_RADIUS = 0.1
    NOISE = 0.001
    GRAVITY = -10.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, GRAVITY),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(gs.morphs.Plane())

    box = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(0.0, 0.0, BOX_SIZE / 2 - PENETRATION),  # box is penetrating ground plane
            fixed=False,  # probe will not detect fixed-fixed contact
        ),
    )

    sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=SPHERE_RADIUS,
            pos=(0.0, 0.0, BOX_SIZE + SPHERE_RADIUS + 0.2),  # start with sphere above the box
            fixed=True,
        ),
    )

    probe_normals = (
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    )
    probe = scene.add_sensor(
        gs.sensors.KinematicContactProbe(
            entity_idx=box.idx,
            probe_local_pos=(
                (0.0, 0.0, BOX_SIZE / 2),  # top of box, center
                (BOX_SIZE / 4, BOX_SIZE / 4, BOX_SIZE / 2),  # top of box
                (-BOX_SIZE / 4, -BOX_SIZE / 4, BOX_SIZE / 2),  # top of box
                (0.0, 0.0, -BOX_SIZE / 2),  # bottom of box, center
            ),
            probe_local_normal=probe_normals,
            probe_radius=(
                PROBE_RADIUS,
                PROBE_RADIUS / 10,  # small radius which cannot detect sphere unless it's perfectly on top
                BOX_SIZE / 3,  # large radius that can detect sphere when not aligned
                PROBE_RADIUS,
            ),
            stiffness=STIFFNESS,
            noise=NOISE,
            random_walk=NOISE * 0.1,
            draw_debug=show_viewer,
        )
    )

    sphere_probe = scene.add_sensor(
        gs.sensors.KinematicContactProbe(
            entity_idx=sphere.idx,
            probe_local_pos=[(0.0, 0.0, -SPHERE_RADIUS)],
            probe_local_normal=[(0.0, 0.0, -1.0)],
            probe_radius=PROBE_RADIUS,
            stiffness=STIFFNESS,
            debug_sphere_color=(0.0, 0.0, 1.0, 0.5),
            draw_debug=show_viewer,
        )
    )

    scene.build(n_envs=n_envs)

    scene.step()

    noisy_data = probe.read()
    box_data = probe.read_ground_truth()

    with np.testing.assert_raises(AssertionError):
        assert_allclose(noisy_data.penetration, box_data.penetration, tol=gs.EPS)
    with np.testing.assert_raises(AssertionError):
        assert_allclose(noisy_data.force, box_data.force, tol=gs.EPS)

    noise_tol = NOISE * 10.0
    assert_allclose(noisy_data.penetration, box_data.penetration, atol=noise_tol)
    assert_allclose(noisy_data.force, box_data.force, atol=noise_tol)

    # Check that the box's bottom probe (idx 3) detects the ground
    assert (box_data.penetration[..., 3] > tol).all(), "Bottom probe should detect ground contact"
    assert (box_data.force[..., 3, 2] > tol).all(), "Bottom probe should have upward force from ground"

    # Forces should be equivalent to the penetration * stiffness along normal vector
    normals = torch.stack([-torch.tensor(n) for n in probe_normals])
    expected_force = (box_data.penetration * STIFFNESS).unsqueeze(-1) * normals
    assert_allclose(box_data.force, expected_force, tol=tol)

    # Top probes should not detect anything yet
    assert_allclose(box_data.penetration[..., :3], 0.0, tol=gs.EPS)
    assert_allclose(box_data.force[..., :3, :], 0.0, tol=gs.EPS)

    # Now position the sphere to penetrate the top of the box
    sphere.set_pos((0.0, 0.0, BOX_SIZE + SPHERE_RADIUS - PENETRATION))
    scene.step()

    box_data = probe.read_ground_truth()
    sphere_data = sphere_probe.read()

    assert (box_data.penetration[..., 0] > tol).all(), "Top probe should detect sphere contact"
    assert (box_data.force[..., 0, 2] < -tol).all(), "Top probe should have downward force from sphere"
    assert (sphere_data.penetration[..., 0] > tol).all(), "Sphere probe should detect box contact"
    assert_allclose(
        sphere_data.penetration[..., 0],
        box_data.penetration[..., 0],
        tol=2e-3,
        err_msg="Sphere probe penetration should match top box probe penetration",
    )
    assert_equal(
        box_data.penetration[..., 1], 0.0, err_msg="Noncenter probe with small radius should not detect contact"
    )
    assert (box_data.penetration[..., 2] > tol).all(), "Noncenter probe with large radius should detect contact"

    # Move sphere away and check no contact
    sphere.set_pos((0.0, 0.0, BOX_SIZE / 2 + SPHERE_RADIUS + PROBE_RADIUS + 0.2))
    scene.step()

    sphere_data = sphere_probe.read()
    sphere_ground_truth = sphere_probe.read_ground_truth()
    assert_allclose(sphere_data.penetration, sphere_ground_truth.penetration, tol=gs.EPS)
    assert_allclose(sphere_data.force, sphere_ground_truth.force, tol=gs.EPS)
    assert_allclose(sphere_data.penetration, 0.0, tol=gs.EPS)
    assert_allclose(sphere_data.force, 0.0, tol=gs.EPS)


def _build_hemisphere_probes(radius: float, n_theta: int, n_phi: int):
    """Probe positions and outward normals on the bottom hemisphere (z <= 0 in link frame)."""
    theta = (np.pi / 2) * (1 + torch.arange(n_theta, dtype=gs.tc_float, device=gs.device) / n_theta)
    phi = torch.arange(n_phi, dtype=gs.tc_float, device=gs.device) * (2 * np.pi) / n_phi
    theta, phi = torch.meshgrid(theta, phi, indexing="ij")
    theta = theta.ravel()
    phi = phi.ravel()
    x = radius * theta.sin() * phi.cos()
    y = radius * theta.sin() * phi.sin()
    z = radius * theta.cos()
    positions = torch.stack([x, y, z], dim=-1)
    normals = positions / radius
    return positions, normals


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_elastomer_displacement_sensor_sphere_ground(show_viewer, tol, n_envs):
    """Test ElastomerDisplacementSensor with bottom-hemisphere probes on a sphere penetrating the ground."""

    SPHERE_RADIUS = 0.2
    PROBE_RADIUS = 0.02
    PENETRATION = 0.01
    RING_ANGLE_DEG = 6.0
    N_RING = 6
    MAX_DELTAS = (1.0, 1.0, 60.0)

    scene = gs.Scene(
        show_viewer=show_viewer,
    )

    scene.add_entity(gs.morphs.Plane())

    # Sphere penetrating the ground (center below z=0 by PENETRATION)
    sphere_init_pos = (0.0, 0.0, SPHERE_RADIUS - PENETRATION)
    sphere_init_quat = (1.0, 0.0, 0.0, 0.0)
    sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=SPHERE_RADIUS,
            pos=sphere_init_pos,
        ),
    )

    # One probe at bottom of sphere plus a ring at RING_ANGLE_DEG from bottom (angle from center)
    angle_rad = torch.tensor(RING_ANGLE_DEG * torch.pi / 180, dtype=gs.tc_float, device=gs.device)
    theta_ring = torch.pi - angle_rad
    z_ring = SPHERE_RADIUS * theta_ring.cos()
    r_xy = SPHERE_RADIUS * theta_ring.sin()
    phi = torch.arange(N_RING, dtype=gs.tc_float, device=gs.device) * (2 * torch.pi) / N_RING
    ring_positions = torch.stack([r_xy * phi.cos(), r_xy * phi.sin(), torch.full_like(phi, z_ring)], dim=-1)
    ring_normals = ring_positions / SPHERE_RADIUS
    bottom_pos = torch.tensor([[0.0, 0.0, -SPHERE_RADIUS]], dtype=gs.tc_float, device=gs.device)
    bottom_normal = torch.tensor([[0.0, 0.0, -1.0]], dtype=gs.tc_float, device=gs.device)
    probe_positions = torch.cat([bottom_pos, ring_positions], dim=0)
    probe_normals = torch.cat([bottom_normal, ring_normals], dim=0)

    sensor_kwargs = dict(
        entity_idx=sphere.idx,
        probe_local_pos=probe_positions,
        probe_local_normal=probe_normals,
        probe_radius=PROBE_RADIUS,
        draw_debug=show_viewer,
        dilate_coefficient=1e-2,
        shear_coefficient=1e-2,
        twist_coefficient=1e-2,
    )
    dilate_sensor = scene.add_sensor(
        gs.sensors.ElastomerDisplacement(
            dilate_max_delta=MAX_DELTAS[0],
            shear_max_delta=0.0,
            twist_max_delta=0.0,
            **sensor_kwargs,
        )
    )
    shear_sensor = scene.add_sensor(
        gs.sensors.ElastomerDisplacement(
            dilate_max_delta=0.0,
            shear_max_delta=MAX_DELTAS[1],
            twist_max_delta=0.0,
            **sensor_kwargs,
        )
    )
    twist_sensor = scene.add_sensor(
        gs.sensors.ElastomerDisplacement(
            dilate_max_delta=0.0,
            shear_max_delta=0.0,
            twist_max_delta=MAX_DELTAS[2],
            **sensor_kwargs,
        )
    )
    sensor = scene.add_sensor(
        gs.sensors.ElastomerDisplacement(
            dilate_max_delta=MAX_DELTAS[0],
            shear_max_delta=MAX_DELTAS[1],
            twist_max_delta=MAX_DELTAS[2],
            **sensor_kwargs,
        )
    )

    if show_viewer:
        rec_kwargs = dict(
            normal=(0.0, 0.0, -1.0),
            scale_factor=10.0,
            max_magnitude=1.0e-2,
            positions=probe_positions,
        )
        dilate_sensor.start_recording(
            rec_options=gs.recorders.MPLVectorFieldPlot(
                title="Dilate Sensor",
                **rec_kwargs,
            ),
        )
        shear_sensor.start_recording(
            rec_options=gs.recorders.MPLVectorFieldPlot(
                title="Shear Sensor",
                **rec_kwargs,
            ),
        )
        twist_sensor.start_recording(
            rec_options=gs.recorders.MPLVectorFieldPlot(
                title="Twist Sensor",
                **rec_kwargs,
            ),
        )

    scene.build(n_envs=n_envs)

    dt = scene.dt

    scene.step()

    # test dilate displacement
    dilate_data = dilate_sensor.read()
    # Contact point in sphere link frame (south pole); direction away from contact for each probe
    contact_pos = torch.tensor([0.0, 0.0, -PENETRATION], dtype=gs.tc_float, device=gs.device)
    direction_away = probe_positions - contact_pos
    direction_away = direction_away / (direction_away.norm(dim=-1, keepdim=True).clamp(min=1e-12))
    dots = (dilate_data * direction_away).sum(dim=-1)
    assert (dots < tol).all(), "All dilate displacements should point away from the contact"

    # test shear displacement
    sphere.set_pos(sphere_init_pos)
    sphere.set_quat(sphere_init_quat)
    sphere.set_dofs_velocity((-0.2, 0.0, 0.0, 0.0, 0.0, 0.0))
    scene.step()
    # shear sensor should detect 0.5 m/s of shear displacement
    assert_allclose(shear_sensor.read()[..., 0], 0.2 * dt, rtol=1.5)
    assert_allclose(twist_sensor.read(), 0.0, tol=tol)

    # test twist displacement
    sphere.set_pos(sphere_init_pos)
    sphere.set_quat(sphere_init_quat)
    sphere.set_dofs_velocity((0.0, 0.0, 0.0, 0.0, 0.0, 30.0))
    scene.step()
    # twist sensor should detect 0.05 m of twist displacement
    assert_allclose(twist_sensor.read()[..., 1:, :2].norm(dim=-1), 0.2 * dt, rtol=1.5)
    assert_allclose(twist_sensor.read()[..., 2], 0.0, tol=dt)
    assert_allclose(shear_sensor.read(), 0.0, tol=tol)

    # test combined displacement
    sphere.set_pos(sphere_init_pos)
    sphere.set_quat(sphere_init_quat)
    sphere.set_dofs_velocity((0.2, 0.0, 0.0, 0.0, 0.0, 0.2))
    scene.step()
    dilate_data = dilate_sensor.read()
    shear_data = shear_sensor.read()
    twist_data = twist_sensor.read()
    combined_data = sensor.read()
    assert_allclose(combined_data, dilate_data + shear_data + twist_data, tol=tol)

    # test no contact
    sphere.set_pos((0.0, 0.0, SPHERE_RADIUS + 0.05))
    scene.step()
    data = sensor.read()
    assert_equal(data, 0.0, err_msg="Displacement should be zero with no contact")


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_elastomer_displacement_sensor_box_sphere(show_viewer, tol, n_envs):
    """Test ElastomerDisplacementSensor with probes on a box resting on a sphere."""
    SPHERE_RADIUS = 0.1
    PROBE_RADIUS = 0.02
    PENETRATION = 0.01
    BOX_SIZE = 0.1
    GRID_SIZE = (8, 8)

    scene = gs.Scene(
        show_viewer=show_viewer,
    )

    scene.add_entity(gs.morphs.Plane())

    # Sphere penetrating the ground (center below z=0 by PENETRATION)
    sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=SPHERE_RADIUS,
            pos=(0.0, 0.0, SPHERE_RADIUS),
            fixed=True,
        ),
    )
    box = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(0.0, 0.0, SPHERE_RADIUS * 2 + BOX_SIZE / 2 - PENETRATION),
        ),
    )
    sensor_kwargs = dict(
        entity_idx=box.idx,
        link_idx_local=0,
        probe_local_normal=(0.0, 0.0, -1.0),
        probe_radius=PROBE_RADIUS,
        dilate_coefficient=1e-2,
        shear_coefficient=1e-2,
        twist_coefficient=1e-2,
        draw_debug=show_viewer,
    )
    probe_local_pos = gu.generate_grid_points_on_plane(
        lo=(-BOX_SIZE / 2, -BOX_SIZE / 2, -BOX_SIZE / 2),
        hi=(BOX_SIZE / 2, BOX_SIZE / 2, -BOX_SIZE / 2),
        normal=(0.0, 0.0, -1.0),
        nx=GRID_SIZE[0],
        ny=GRID_SIZE[1],
    )
    elastomer_grid_sensor = scene.add_sensor(
        gs.sensors.ElastomerDisplacement(
            probe_local_pos=probe_local_pos,
            **sensor_kwargs,
        )
    )
    elastomer_sensor = scene.add_sensor(
        gs.sensors.ElastomerDisplacement(
            probe_local_pos=probe_local_pos.reshape(-1, 3),
            **sensor_kwargs,
        )
    )
    assert elastomer_grid_sensor._is_grid and not elastomer_sensor._is_grid
    assert_allclose(elastomer_sensor.probe_local_pos, elastomer_grid_sensor.probe_local_pos, tol=gs.EPS)

    scene.build(n_envs=n_envs)

    scene.step()

    # grid sensor should match
    grid_data = elastomer_grid_sensor.read()
    data = elastomer_sensor.read()

    assert_allclose(data, grid_data, tol=tol)

    # test no contact
    box.set_pos((0.0, 0.0, BOX_SIZE + SPHERE_RADIUS * 2 + PENETRATION))
    scene.step()

    data = elastomer_grid_sensor.read()
    assert_equal(data, 0.0, err_msg="Displacement should be zero with no contact")


# ------------------------------------------------------------------------------------------
# ----------------------------------- Proximity Sensor -------------------------------------
# ------------------------------------------------------------------------------------------


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_proximity_sensor_box_sphere(n_envs, show_viewer, tol):
    """Test proximity sensor returns distance and nearest points with correct shapes and plausible values."""
    SPHERE_RADIUS = 0.05
    DISTANCE = 0.15
    MAX_RANGE = 10.0
    BOX_PROBE_POS = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.05)]
    SPHERE_PROBE_POS = [(0.0, 0.0, SPHERE_RADIUS)]

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )
    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.0),
        ),
    )
    # Tracked objects
    sphere1 = scene.add_entity(
        gs.morphs.Sphere(
            radius=SPHERE_RADIUS,
            pos=(0.0, 0.0, DISTANCE),
        ),
    )
    sphere2 = scene.add_entity(
        gs.morphs.Sphere(
            radius=SPHERE_RADIUS,
            pos=(0.0, 0.0, DISTANCE * 2.0),
        ),
    )
    # Not tracked objects
    sphere3 = scene.add_entity(
        gs.morphs.Sphere(
            radius=SPHERE_RADIUS,
            pos=(0.0, DISTANCE / 2.0, 0.0),
        ),
    )

    box_prox_sensor = scene.add_sensor(
        gs.sensors.Proximity(
            entity_idx=box.idx,
            probe_local_pos=BOX_PROBE_POS,
            track_link_idx=(sphere1.base_link_idx, sphere2.base_link_idx),
            max_range=MAX_RANGE,
        )
    )
    sphere_prox_sensor = scene.add_sensor(
        gs.sensors.Proximity(
            entity_idx=sphere1.idx,
            probe_local_pos=SPHERE_PROBE_POS,
            track_link_idx=(box.base_link_idx,),
            max_range=MAX_RANGE,
            resolution=0.001,
            bias=0.1,
            noise=0.01,
            random_walk=0.01,
        )
    )
    scene.build(n_envs=n_envs)

    scene.step()

    box_prox_data = box_prox_sensor.read()
    sphere_prox_noisy_data = sphere_prox_sensor.read()
    sphere_prox_data = sphere_prox_sensor.read_ground_truth()

    for i in range(len(BOX_PROBE_POS)):
        assert_allclose(box_prox_data.distance[..., i], DISTANCE - SPHERE_RADIUS - BOX_PROBE_POS[i][2], tol=tol)
    assert_allclose(box_prox_data.points, (0.0, 0.0, DISTANCE - SPHERE_RADIUS), tol=tol)
    assert_allclose(sphere_prox_data.distance, DISTANCE, tol=tol)

    with np.testing.assert_raises(AssertionError):
        assert_allclose(sphere_prox_noisy_data.distance, sphere_prox_data.distance, tol=tol)

    sphere1_pos = np.array((0.0, 0.0, DISTANCE * 3.0))
    sphere1.set_pos(sphere1_pos)

    scene.step()

    box_prox_data = box_prox_sensor.read()
    sphere_prox_data = sphere_prox_sensor.read_ground_truth()

    assert_allclose(box_prox_data.distance[..., 0], DISTANCE * 2.0 - SPHERE_RADIUS, tol=tol)
    assert_allclose(box_prox_data.distance[..., 1], DISTANCE * 2.0 - SPHERE_RADIUS - 0.05, tol=tol)
    assert_allclose(sphere_prox_data.distance, DISTANCE * 3.0, tol=tol)

    box_pos = np.array((0.0, 0.0, -MAX_RANGE))
    box.set_pos(box_pos)
    scene.step()

    box_prox_data = box_prox_sensor.read()
    sphere_prox_data = sphere_prox_sensor.read_ground_truth()

    assert_allclose(box_prox_data.distance, MAX_RANGE, tol=tol)
    assert_allclose(sphere_prox_data.distance, MAX_RANGE, tol=tol)
    for i in range(len(BOX_PROBE_POS)):
        assert_allclose(
            box_prox_data.points[..., i, :],
            np.array(BOX_PROBE_POS[i]) + box_pos,
            tol=tol,
            err_msg="When out of range, points should be the probe position in world frame",
        )
    assert_allclose(
        sphere_prox_data.points,
        np.array(SPHERE_PROBE_POS) + sphere1_pos,
        tol=tol,
        err_msg="When out of range, points should be the probe position in world frame",
    )
