import numpy as np
import pytest

import genesis as gs

from .utils import assert_allclose, assert_array_equal


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_rasterizer_camera_sensor(show_viewer, tol, n_envs):
    """Test if the RasterizerCameraSensor returns images with correct shapes and valid pixel values."""

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            gravity=(0, 0, -9.8),
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=show_viewer,
    )

    # Add entities
    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
        surface=gs.surfaces.Rough(color=(0.4, 0.4, 0.4)),
    )

    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(pos=(0.0, 0.0, 2.0), radius=0.5),
        surface=gs.surfaces.Smooth(color=(1.0, 0.5, 0.5)),
    )

    box = scene.add_entity(
        morph=gs.morphs.Box(pos=(1.0, 1.0, 1.0), size=(0.3, 0.3, 0.3)),
        surface=gs.surfaces.Rough(color=(0.5, 1.0, 0.5)),
    )

    # Add rasterizer cameras with different resolutions
    raster_cam0 = scene.add_sensor(
        gs.sensors.RasterizerCameraOptions(
            res=(512, 512),
            pos=(3.0, 0.0, 2.0),
            lookat=(0.0, 0.0, 1.0),
            up=(0.0, 0.0, 1.0),
            fov=60.0,
            near=0.1,
            far=100.0,
        )
    )

    raster_cam1 = scene.add_sensor(
        gs.sensors.RasterizerCameraOptions(
            res=(256, 256),
            pos=(0.0, 3.0, 2.0),
            lookat=(0.0, 0.0, 1.0),
            up=(0.0, 0.0, 1.0),
            fov=45.0,
        )
    )

    # Camera for attachment testing
    raster_cam_attached = scene.add_sensor(
        gs.sensors.RasterizerCameraOptions(
            res=(320, 240),
            pos=(0.0, 0.0, 3.0),
            lookat=(0.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
            fov=70.0,
        )
    )

    # Add lights
    raster_cam0.add_light(
        pos=(2.0, 2.0, 5.0),
        color=(1.0, 1.0, 1.0),
        intensity=5.0,
    )

    scene.build(n_envs=n_envs)

    # Attach camera to sphere
    offset_T = np.eye(4, dtype=np.float32)
    offset_T[2, 3] = 1.0  # 1 meter above the sphere center
    sphere_link = sphere.links[0]
    raster_cam_attached.attach(sphere_link, offset_T)

    # Run simulation
    for i in range(10):
        scene.step()

    # Test render and read
    raster_cam0.render()
    data_cam0 = raster_cam0.read()

    raster_cam1.render()
    data_cam1 = raster_cam1.read()

    raster_cam_attached.render()
    data_attached = raster_cam_attached.read()

    # Check shapes
    expected_batch_shape = (n_envs,) if n_envs > 0 else ()
    assert data_cam0.rgb.shape == expected_batch_shape + (
        512,
        512,
        3,
    ), f"cam0 shape mismatch: got {data_cam0.rgb.shape}"
    assert data_cam1.rgb.shape == expected_batch_shape + (
        256,
        256,
        3,
    ), f"cam1 shape mismatch: got {data_cam1.rgb.shape}"
    assert data_attached.rgb.shape == expected_batch_shape + (
        240,
        320,
        3,
    ), f"attached cam shape mismatch: got {data_attached.rgb.shape}"

    # Check that images are not pure black (all zeros) or pure white (all 255s)
    for cam_name, data in [("cam0", data_cam0), ("cam1", data_cam1), ("attached", data_attached)]:
        rgb = data.rgb

        # Check not pure black
        mean_value = np.mean(rgb)
        assert mean_value > 1.0, f"{cam_name}: Image is too dark (mean={mean_value:.2f}), likely pure black"

        # Check not pure white
        assert mean_value < 254.0, f"{cam_name}: Image is too bright (mean={mean_value:.2f}), likely pure white"

        # Check variance (should have some texture/variation)
        variance = np.var(rgb)
        assert variance > 1.0, f"{cam_name}: Image has no variation (var={variance:.2f}), likely uniform color"

    # Test read with envs_idx
    if n_envs > 0:
        data_env0 = raster_cam0.read(envs_idx=0)
        assert data_env0.rgb.shape == (512, 512, 3), f"Single env read shape mismatch: got {data_env0.rgb.shape}"

        data_env1 = raster_cam0.read(envs_idx=1)
        assert data_env1.rgb.shape == (512, 512, 3), f"Single env read shape mismatch: got {data_env1.rgb.shape}"

    # Helper to query the world position of a rasterizer camera sensor
    def _get_camera_world_pos(sensor):
        renderer = sensor._shared_metadata.renderer
        context = sensor._shared_metadata.context
        node = renderer._camera_nodes[sensor._idx]
        pose = context._scene.get_pose(node)
        # Handle possible batched pose (n_envs, 4, 4)
        if pose.ndim == 3:
            pose = pose[0]
        return pose[:3, 3].copy()

    # Store attached camera position before detachment
    cam_pos_initial = _get_camera_world_pos(raster_cam_attached)

    # Continue simulation to let sphere fall
    for i in range(20):
        scene.step()

    # Render attached camera (should follow sphere)
    raster_cam_attached.render()
    data_attached_moved = raster_cam_attached.read()

    # Camera position should be different after sphere moved
    cam_pos_moved = _get_camera_world_pos(raster_cam_attached)
    cam_move_dist = np.linalg.norm(cam_pos_moved - cam_pos_initial)
    assert cam_move_dist > 1e-3, f"Attached camera position didn't change after sphere moved (dist={cam_move_dist:.3e})"

    # Store camera position at the moment of detachment
    cam_pos_at_detach = cam_pos_moved.copy()

    # Detach camera
    raster_cam_attached.detach()

    # Continue simulation
    for i in range(20):
        scene.step()

    # Render detached camera
    raster_cam_attached.render()
    data_detached = raster_cam_attached.read()

    # After detachment, camera should stay at same position while sphere continues falling
    # So the camera position should be (almost) unchanged
    cam_pos_after_detach = _get_camera_world_pos(raster_cam_attached)
    cam_move_after_detach = np.linalg.norm(cam_pos_after_detach - cam_pos_at_detach)
    assert (
        cam_move_after_detach < 1e-6
    ), f"Detached camera position changed too much (dist={cam_move_after_detach:.3e}), should stay static"
