import sys

import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array
from .utils import assert_allclose, rgb_array_to_png_bytes


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 1])
def test_rasterizer_camera_sensor(n_envs, show_viewer):
    scene = gs.Scene(
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(
        morph=gs.morphs.Plane(),
        surface=gs.surfaces.Rough(
            color=(0.4, 0.4, 0.4),
        ),
    )

    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=0.5,
            pos=(0.0, 0.0, 2.0),
        ),
        surface=gs.surfaces.Smooth(
            color=(1.0, 0.5, 0.5),
        ),
    )

    scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.3, 0.3, 0.3),
            pos=(1.0, 1.0, 1.0),
        ),
        surface=gs.surfaces.Rough(
            color=(0.5, 1.0, 0.5),
        ),
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
    for _ in range(10):
        scene.step()
    data_cam0 = raster_cam0.read()
    data_cam1 = raster_cam1.read()
    data_attached = raster_cam_attached.read()
    data_offset_T = raster_cam_offset_T.read()

    for _cam_name, data in [
        ("cam0", data_cam0),
        ("cam1", data_cam1),
        ("attached", data_attached),
        ("offset_T", data_offset_T),
    ]:
        rgb_np = tensor_to_array(data.rgb)
        mean = np.mean(rgb_np)
        assert 1.0 < mean < 254.0
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

    for _ in range(10):  # Test over multiple steps
        scene.step()

    raster_cam_attached.read()
    cam_pos_final = _get_camera_world_pos(raster_cam_attached)
    cam_move_dist = np.linalg.norm(cam_pos_final - cam_pos_initial)
    assert cam_move_dist > 1e-2
    raster_cam_offset_T.read()
    cam_pos_final_offset_T = _get_camera_world_pos(raster_cam_offset_T)
    cam_move_dist_offset_T = np.linalg.norm(cam_pos_final_offset_T - cam_pos_initial_offset_T)
    assert cam_move_dist_offset_T > 1e-2
    assert_allclose(cam_move_dist_offset_T, cam_move_dist, atol=1e-2)


# ========================== Multi-environment tests ==========================


@pytest.mark.required
@pytest.mark.skipif(sys.platform == "darwin", reason="Not supported on this machine because it requires OpenGL 4.2.")
def test_rasterizer_camera_sensor_n_envs(show_viewer, png_snapshot):
    scene = gs.Scene(
        show_viewer=show_viewer,
    )

    # Add a plane
    scene.add_entity(
        morph=gs.morphs.Plane(),
        surface=gs.surfaces.Rough(color=(0.4, 0.4, 0.4)),
    )

    # Add a sphere
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(pos=(0.0, 0.0, 1.0), radius=0.3),
        surface=gs.surfaces.Smooth(color=(1.0, 0.5, 0.5)),
    )
    camera = scene.add_sensor(
        gs.sensors.RasterizerCameraOptions(
            res=(64, 64), pos=(3.0, 0.0, 1.5), lookat=(0.0, 0.0, 0.5), fov=60.0, draw_debug=show_viewer
        )
    )
    scene.build(n_envs=2)

    # Disable shadows systematically for Rasterizer because they are forcibly disabled on CPU backend anyway
    camera._shared_metadata.context.shadow = False

    sphere.set_pos([[0.0, 0.0, 1.0], [0.2, 0.0, 0.5]])
    scene.step()

    data = camera.read()

    assert data.rgb.shape == (2, 64, 64, 3)
    assert data.rgb.dtype == torch.uint8
    assert (data.rgb[0] != data.rgb[1]).any(), "We should have different frames"

    for i in range(scene.n_envs):
        assert rgb_array_to_png_bytes(data.rgb[i]) == png_snapshot


@pytest.mark.required
@pytest.mark.skipif(sys.platform == "darwin", reason="Not supported on this machine because it requires OpenGL 4.2.")
def test_rasterizer_camera_sensor_n_envs_attached_camera():
    scene = gs.Scene()

    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=0.3,
            pos=(0.0, 0.0, 1.0),
        ),
        surface=gs.surfaces.Smooth(
            color=(1.0, 0.5, 0.5),
        ),
    )

    scene.add_sensor(
        gs.sensors.RasterizerCameraOptions(
            entity_idx=sphere.idx,
        )
    )

    with pytest.raises(gs.GenesisException, match="does not work with attached cameras yet."):
        scene.build(n_envs=2)
