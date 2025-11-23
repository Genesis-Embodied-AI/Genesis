import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import tensor_to_array


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 1])
def test_rasterizer_camera_sensor(show_viewer, tol, n_envs):
    scene = gs.Scene(
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
    raster_cam_attached = scene.add_sensor(
        gs.sensors.RasterizerCameraOptions(
            res=(320, 240),
            pos=(0.0, 0.0, 3.0),
            lookat=(0.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
            fov=70.0,
        )
    )
    raster_cam0.add_light(
        pos=(2.0, 2.0, 5.0),
        color=(1.0, 1.0, 1.0),
        intensity=5.0,
    )

    scene.build(n_envs=n_envs)
    offset_T = np.eye(4, dtype=np.float32)
    offset_T[2, 3] = 1.0  # 1 meter above the sphere center
    sphere_link = sphere.links[0]
    raster_cam_attached.attach(sphere_link, offset_T)
    for i in range(10):
        scene.step()
    data_cam0 = raster_cam0.read()
    data_cam1 = raster_cam1.read()
    data_attached = raster_cam_attached.read()

    for cam_name, data in [("cam0", data_cam0), ("cam1", data_cam1), ("attached", data_attached)]:
        rgb = data.rgb
        rgb_np = tensor_to_array(rgb)
        mean_value = np.mean(rgb_np)
        assert mean_value > 1.0
        assert mean_value < 254.0
        variance = np.var(rgb_np)
        assert variance > 1.0
    data_env0 = raster_cam0.read(envs_idx=0)
    assert data_env0.rgb.shape == (512, 512, 3), f"Single env read shape mismatch: got {data_env0.rgb.shape}"

    def _get_camera_world_pos(sensor):
        renderer = sensor._shared_metadata.renderer
        context = sensor._shared_metadata.context
        node = renderer._camera_nodes[sensor._idx]
        pose = context._scene.get_pose(node)
        if pose.ndim == 3:
            pose = pose[0]
        return pose[:3, 3].copy()

    cam_pos_initial = _get_camera_world_pos(raster_cam_attached)

    for i in range(1):
        scene.step()

    data_attached_moved = raster_cam_attached.read()
    cam_pos_moved = _get_camera_world_pos(raster_cam_attached)
    cam_move_dist = np.linalg.norm(cam_pos_moved - cam_pos_initial)
    assert cam_move_dist > 1e-3, f"Attached camera position didn't change after sphere moved (dist={cam_move_dist:.3e})"
    cam_pos_at_detach = cam_pos_moved.copy()
    raster_cam_attached.detach()
    for i in range(1):
        scene.step()

    # After detachment, camera should stay at same position while sphere continues falling
    # So the camera position should be (almost) unchanged
    cam_pos_after_detach = _get_camera_world_pos(raster_cam_attached)
    cam_move_after_detach = np.linalg.norm(cam_pos_after_detach - cam_pos_at_detach)
    assert cam_move_after_detach < 1e-6
