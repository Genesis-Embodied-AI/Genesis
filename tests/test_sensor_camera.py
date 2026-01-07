import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array
from .utils import assert_allclose


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 1])
def test_rasterizer_camera_sensor(show_viewer, tol, n_envs):
    scene = gs.Scene(
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=show_viewer,
    )

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

    raster_cam0 = scene.add_sensor(
        gs.sensors.RasterizerCameraOptions(
            res=(512, 512),
            pos=(3.0, 0.0, 2.0),
            lookat=(0.0, 0.0, 1.0),
            up=(0.0, 0.0, 1.0),
            fov=60.0,
            near=0.1,
            far=100.0,
            lights=[
                {
                    "pos": (2.0, 2.0, 5.0),
                    "color": (1.0, 1.0, 1.0),
                    "intensity": 5.0,
                }
            ],
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
    raster_cam_attached = scene.add_sensor(
        gs.sensors.RasterizerCameraOptions(
            res=(320, 240),
            pos=(0.0, 0.0, 1.0),  # Relative to link when attached
            lookat=(0.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
            fov=70.0,
            entity_idx=sphere.idx,  # Attach to sphere
            link_idx_local=0,
        )
    )
    offset_T = np.eye(4)
    offset_T[2, 3] = 1.0
    raster_cam_offset_T = scene.add_sensor(
        gs.sensors.RasterizerCameraOptions(
            res=(320, 240),
            pos=(0.0, 0.0, 1.0),
            lookat=(0.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
            fov=70.0,
            entity_idx=sphere.idx,
            link_idx_local=0,
            offset_T=offset_T,
        )
    )

    scene.build(n_envs=n_envs)
    for i in range(10):
        scene.step()
    data_cam0 = raster_cam0.read()
    data_cam1 = raster_cam1.read()
    data_attached = raster_cam_attached.read()
    data_offset_T = raster_cam_offset_T.read()

    for cam_name, data in [
        ("cam0", data_cam0),
        ("cam1", data_cam1),
        ("attached", data_attached),
        ("offset_T", data_offset_T),
    ]:
        rgb = data.rgb
        rgb_np = tensor_to_array(rgb)
        mean_value = np.mean(rgb_np)
        assert mean_value > 1.0
        assert mean_value < 254.0
        variance = np.var(rgb_np)
        assert variance > 1.0
    data_env0 = raster_cam0.read(envs_idx=0)
    assert data_env0.rgb.shape == (512, 512, 3)

    def _get_camera_world_pos(sensor):
        renderer = sensor._shared_metadata.renderer
        context = sensor._shared_metadata.context
        node = renderer._camera_nodes[sensor._idx]
        pose = context._scene.get_pose(node)
        if pose.ndim == 3:
            pose = pose[0]
        return pose[:3, 3].copy()

    cam_pos_initial = _get_camera_world_pos(raster_cam_attached)
    cam_pos_initial_offset_T = _get_camera_world_pos(raster_cam_offset_T)

    for i in range(10):  # Test over multiple steps
        scene.step()

    data_attached_moved = raster_cam_attached.read()
    cam_pos_final = _get_camera_world_pos(raster_cam_attached)
    cam_move_dist = np.linalg.norm(cam_pos_final - cam_pos_initial)
    assert cam_move_dist > 1e-2
    data_offset_T_moved = raster_cam_offset_T.read()
    cam_pos_final_offset_T = _get_camera_world_pos(raster_cam_offset_T)
    cam_move_dist_offset_T = np.linalg.norm(cam_pos_final_offset_T - cam_pos_initial_offset_T)
    assert cam_move_dist_offset_T > 1e-2
    assert_allclose(cam_move_dist_offset_T, cam_move_dist, atol=1e-2)


# ========================== Multi-environment tests ==========================


@pytest.fixture
def rasterizer_camera_scene():
    """Create a simple scene with rasterizer camera sensor for multi-env tests."""
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            gravity=(0, 0, -9.8),
        ),
        renderer=gs.renderers.Rasterizer(),
        show_viewer=False,
    )

    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
        surface=gs.surfaces.Rough(color=(0.4, 0.4, 0.4)),
    )

    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(pos=(0.0, 0.0, 1.0), radius=0.3),
        surface=gs.surfaces.Smooth(color=(1.0, 0.5, 0.5)),
    )

    options = gs.sensors.RasterizerCameraOptions(
        res=(64, 64),
        pos=(3.0, 0.0, 1.5),
        lookat=(0.0, 0.0, 0.5),
        fov=60.0,
    )
    camera = scene.add_sensor(options)

    return scene, sphere, camera


@pytest.mark.parametrize("n_envs", [2, 4])
def test_rasterizer_camera_sensor_n_envs(rasterizer_camera_scene, n_envs):
    """Test that RasterizerCameraSensor works with n_envs > 1."""
    scene, sphere, camera = rasterizer_camera_scene

    scene.build(n_envs=n_envs, env_spacing=(0.0, 0.0))
    scene.step()

    data = camera.read()

    assert data.rgb.shape == (n_envs, 64, 64, 3), f"Expected shape ({n_envs}, 64, 64, 3), got {data.rgb.shape}"
    assert data.rgb.dtype == torch.uint8, f"Expected dtype torch.uint8, got {data.rgb.dtype}"


@pytest.mark.parametrize("n_envs", [2])
def test_rasterizer_camera_sensor_different_poses(rasterizer_camera_scene, n_envs):
    """Test that different environments can have different object poses."""
    scene, sphere, camera = rasterizer_camera_scene

    scene.build(n_envs=n_envs, env_spacing=(0.0, 0.0))

    # Set different sphere positions for each environment
    sphere.set_pos(torch.tensor([0.0, 0.0, 1.0], device=gs.device), envs_idx=[0])
    sphere.set_pos(torch.tensor([1.0, 0.0, 1.0], device=gs.device), envs_idx=[1])

    scene.step()

    data = camera.read()

    assert data.rgb.shape == (n_envs, 64, 64, 3)

    # Verify that the two environment images are different
    img0 = data.rgb[0].float()
    img1 = data.rgb[1].float()
    diff = (img0 - img1).abs().mean()

    assert diff > 1.0, f"Images should be different, but mean absolute diff is only {diff}"


@pytest.mark.parametrize("n_envs", [2])
def test_rasterizer_camera_sensor_attached_n_envs(n_envs):
    """Test that camera attachment works with n_envs > 1."""
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            gravity=(0, 0, -9.8),
        ),
        renderer=gs.renderers.Rasterizer(),
        show_viewer=False,
    )

    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
        surface=gs.surfaces.Rough(color=(0.4, 0.4, 0.4)),
    )

    box = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, 1.0), size=(0.3, 0.3, 0.3)),
        surface=gs.surfaces.Smooth(color=(0.2, 0.5, 0.8)),
    )

    target = scene.add_entity(
        morph=gs.morphs.Sphere(pos=(2.0, 0.0, 1.0), radius=0.2),
        surface=gs.surfaces.Smooth(color=(1.0, 0.3, 0.3)),
    )

    # Camera attached to the box
    options = gs.sensors.RasterizerCameraOptions(
        res=(64, 64),
        pos=(0.5, 0.0, 0.0),
        lookat=(2.0, 0.0, 1.0),
        fov=60.0,
        entity_idx=box.idx,
        link_idx_local=0,
    )
    camera = scene.add_sensor(options)

    scene.build(n_envs=n_envs, env_spacing=(0.0, 0.0))
    scene.step()

    data = camera.read()

    assert data.rgb.shape == (n_envs, 64, 64, 3), f"Expected shape ({n_envs}, 64, 64, 3), got {data.rgb.shape}"
    assert data.rgb.dtype == torch.uint8, f"Expected dtype torch.uint8, got {data.rgb.dtype}"
