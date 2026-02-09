import sys
import weakref

import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array
from genesis.utils.geom import trans_to_T

from .utils import assert_allclose, assert_array_equal, rgb_array_to_png_bytes


try:
    import LuisaRenderPy

    ENABLE_RAYTRACER = True
except ImportError:
    ENABLE_RAYTRACER = False
try:
    import gs_madrona

    ENABLE_MADRONA = True
except ImportError:
    ENABLE_MADRONA = False


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 1])
def test_rasterizer_non_batched(n_envs, show_viewer):
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


@pytest.mark.required
@pytest.mark.skipif(sys.platform == "darwin", reason="Not supported on this machine because it requires OpenGL 4.2.")
def test_rasterizer_batched(show_viewer, png_snapshot):
    scene = gs.Scene(
        show_viewer=show_viewer,
    )

    # Add a plane
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )

    # Add a sphere
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(pos=(0.0, 0.0, 1.0), radius=0.3),
        surface=gs.surfaces.Smooth(color=(1.0, 0.5, 0.5)),
    )
    camera = scene.add_sensor(
        gs.sensors.RasterizerCameraOptions(
            res=(64, 64),
            pos=(3.0, 0.0, 1.5),
            lookat=(0.0, 0.0, 0.5),
            fov=60.0,
            draw_debug=show_viewer,
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
def test_rasterizer_attached_batched(show_viewer, png_snapshot):
    scene = gs.Scene(show_viewer=show_viewer)

    # Add a plane
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )

    # Add a sphere
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=0.3,
            pos=(0.0, 0.0, 1.0),
        ),
        surface=gs.surfaces.Smooth(
            color=(1.0, 0.5, 0.5),
        ),
    )

    options = gs.sensors.RasterizerCameraOptions(
        res=(64, 64),
        pos=(-0.4, 0.1, 2.0),
        lookat=(-0.6, 0.4, 1.0),
        fov=60.0,
        entity_idx=sphere.idx,
        draw_debug=show_viewer,
    )
    camera = scene.add_sensor(options)

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
@pytest.mark.parametrize("backend", [gs.cuda])
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.skipif(not ENABLE_MADRONA, reason="BatchRenderer is not supported because 'gs_madrona' is not available.")
def test_batch_renderer(n_envs, png_snapshot):
    CAM_RES = (128, 256)

    scene = gs.Scene(
        show_viewer=False,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=0.5,
            pos=(0.0, 0.0, 1.0),
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.5, 0.5),
        ),
    )

    camera_common_options = dict(
        res=CAM_RES,
        pos=(-2.0, 0.0, 1.5),
        lookat=(0.0, 0.0, 1.0),
        up=(0.0, 0.0, 1.5),
        fov=70.0,
        lights=[
            dict(
                pos=(2.0, 2.0, 5.0),
                color=(1.0, 0.5, 0.25),
                intensity=1.0,
                directional=False,
            )
        ],
        use_rasterizer=True,
    )
    camera_1 = scene.add_sensor(gs.sensors.BatchRendererCameraOptions(**camera_common_options))
    camera_2 = scene.add_sensor(
        gs.sensors.BatchRendererCameraOptions(
            **camera_common_options,
            entity_idx=sphere.idx,
            link_idx_local=0,
            offset_T=trans_to_T(np.array([0.0, 0.0, 3.0])),
        )
    )

    scene.build(n_envs=n_envs)

    scene.step()
    for camera in (camera_1, camera_2):
        data = camera.read()
        if n_envs > 0:
            for i in range(n_envs):
                assert rgb_array_to_png_bytes(data.rgb[i]) == png_snapshot
        else:
            assert rgb_array_to_png_bytes(data.rgb) == png_snapshot


@pytest.mark.required
def test_destroy_unbuilt_scene_with_camera():
    """Test that destroy on an unbuilt scene with cameras doesn't crash."""
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(morph=gs.morphs.Plane())
    scene.add_sensor(gs.sensors.RasterizerCameraOptions(res=(64, 64)))

    # Scene.__del__ calls destroy(), and a crash in destroy() would result in some
    # logspam.
    scene.destroy()


@pytest.mark.required
def test_destroy_idempotent_with_camera():
    """Test that calling destroy twice on a scene with cameras doesn't crash."""
    scene = gs.Scene(show_viewer=False)
    camera = scene.add_sensor(gs.sensors.RasterizerCameraOptions(res=(64, 64)))

    scene.build()
    camera.read()

    scene.destroy()
    # Scene.__del__ calls destroy(), which means it's expected that destroy() will
    # be called twice. A crash in destroy() would result in some logspam.
    scene.destroy()


@pytest.mark.required
def test_rasterizer_destroy():
    scene = gs.Scene(show_viewer=False)
    cam1 = scene.add_sensor(gs.sensors.RasterizerCameraOptions(res=(64, 64)))
    cam2 = scene.add_sensor(gs.sensors.RasterizerCameraOptions(res=(32, 32)))

    scene.build()
    cam1.read()
    cam2.read()

    offscreen_renderer_ref = weakref.ref(cam1._shared_metadata.renderer._renderer)

    scene.destroy()
    assert offscreen_renderer_ref() is None


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cuda])
@pytest.mark.skipif(not ENABLE_MADRONA, reason="BatchRenderer is not supported because 'gs_madrona' is not available.")
def test_batch_renderer_destroy():
    scene = gs.Scene(show_viewer=False)
    # FIXME: This test fails without any entities in the scene.
    scene.add_entity(morph=gs.morphs.Plane())
    cam1 = scene.add_sensor(gs.sensors.BatchRendererCameraOptions(res=(64, 64), use_rasterizer=True))
    cam2 = scene.add_sensor(gs.sensors.BatchRendererCameraOptions(res=(64, 64), use_rasterizer=True))

    scene.build()
    cam1.read()
    cam2.read()

    shared_metadata = cam1._shared_metadata
    assert cam1._shared_metadata is cam2._shared_metadata
    assert len(shared_metadata.sensors) == 2
    assert shared_metadata.renderer is not None

    scene.destroy()

    assert shared_metadata.sensors is None
    assert shared_metadata.renderer is None


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cuda])
@pytest.mark.skipif(not ENABLE_RAYTRACER, reason="RayTracer is not supported because 'LuisaRenderPy' is not available.")
def test_raytracer_destroy():
    scene = gs.Scene(
        renderer=gs.renderers.RayTracer(
            env_surface=gs.surfaces.Emission(
                emissive_texture=gs.textures.ColorTexture(color=(0.2, 0.3, 0.5)),
            ),
            env_radius=20.0,
        ),
        show_viewer=False,
    )

    cam1 = scene.add_sensor(gs.sensors.RaytracerCameraOptions(res=(64, 64)))
    cam2 = scene.add_sensor(gs.sensors.RaytracerCameraOptions(res=(64, 64)))

    scene.build()
    cam1.read()
    cam2.read()

    shared_metadata = cam1._shared_metadata
    assert cam1._shared_metadata is cam2._shared_metadata
    assert len(shared_metadata.sensors) == 2
    assert shared_metadata.renderer is not None

    scene.destroy()

    assert shared_metadata.sensors is None
    assert shared_metadata.renderer is None


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cuda])
@pytest.mark.skipif(not ENABLE_RAYTRACER, reason="RayTracer is not supported because 'LuisaRenderPy' is not available.")
def test_raytracer_attached_without_offset_T():
    """Test that RaytracerCameraSensor works when attached without explicit offset_T.

    Also checks consistency with a scene-level camera (scene.add_camera) using the same
    pose and attachment, to make sure both camera APIs produce matching output.
    """
    CAM_RES = (128, 64)
    CAM_POS = (0.0, 0.0, 2.0)

    scene = gs.Scene(renderer=gs.renderers.RayTracer())
    scene.add_entity(morph=gs.morphs.Plane())
    sphere = scene.add_entity(morph=gs.morphs.Sphere())

    # Sensor camera attached WITHOUT offset_T - should use pos as offset
    camera_common_options = dict(
        res=CAM_RES,
        lookat=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        fov=30.0,
        spp=64,
        denoise=False,
    )
    sensor_camera = scene.add_sensor(
        gs.sensors.RaytracerCameraOptions(
            **camera_common_options,
            pos=CAM_POS,
            entity_idx=sphere.idx,
        )
    )

    # Scene-level camera with the same pose, attached with explicit offset_T
    scene_camera = scene.add_camera(
        **camera_common_options,
    )

    scene.build()

    # Attach scene-level camera with equivalent offset_T
    scene_camera.attach(sphere.base_link, offset_T=trans_to_T(np.array(CAM_POS)))

    scene.step()

    sensor_data = sensor_camera.read()
    assert sensor_data.rgb.shape == (CAM_RES[1], CAM_RES[0], 3)
    assert sensor_data.rgb.float().std() > 1.0, "Sensor camera RGB std too low, image may be blank"

    scene_camera.move_to_attach()
    scene_rgb, *_ = scene_camera.render(rgb=True, force_render=True)
    scene_rgb = tensor_to_array(scene_rgb, dtype=np.int32)
    sensor_rgb = tensor_to_array(sensor_data.rgb, dtype=np.int32)

    # Both cameras should produce the same image
    assert_array_equal(sensor_rgb, scene_rgb)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cuda])
@pytest.mark.parametrize("n_envs", [0, 1])
@pytest.mark.skipif(not ENABLE_RAYTRACER, reason="RayTracer is not supported because 'LuisaRenderPy' is not available.")
def test_raytracer(n_envs, png_snapshot):
    # Relax pixel matching because RayTracer is not deterministic between different hardware (eg RTX6000 vs H100), even
    # without denoiser.
    png_snapshot.extension._blurred_kernel_size = 3

    scene = gs.Scene(
        renderer=gs.renderers.RayTracer(
            env_surface=gs.surfaces.Emission(
                emissive_texture=gs.textures.ColorTexture(
                    color=(0.2, 0.3, 0.5),
                ),
            ),
            env_radius=20.0,
        ),
        show_viewer=False,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=0.5,
            pos=(0.0, 0.0, 1.0),
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.5, 0.5),
        ),
    )

    camera_common_options = dict(
        res=(128, 256),
        pos=(-2.0, 0.0, 1.5),
        lookat=(0.0, 0.0, 1.0),
        up=(0.0, 0.0, 1.5),
        fov=70.0,
        model="pinhole",
        spp=64,
        denoise=False,
        lights=[
            dict(
                pos=(2.0, 2.0, 5.0),
                color=(10.0, 10.0, 10.0),
                intensity=1.0,
            )
        ],
    )
    camera_1 = scene.add_sensor(
        gs.sensors.RaytracerCameraOptions(
            **camera_common_options,
            env_surface=gs.surfaces.Emission(
                emissive_texture=gs.textures.ColorTexture(
                    color=(0.2, 0.3, 0.5),
                ),
            ),
            env_radius=20.0,
        )
    )
    camera_2 = scene.add_sensor(
        gs.sensors.RaytracerCameraOptions(
            **camera_common_options,
            entity_idx=sphere.idx,
            link_idx_local=0,
            offset_T=trans_to_T(np.array([0.0, 0.0, 3.0])),
        )
    )

    scene.build(n_envs=n_envs)

    scene.step()
    for camera in (camera_1, camera_2):
        data = camera.read()
        if n_envs > 0:
            for i in range(n_envs):
                assert rgb_array_to_png_bytes(data.rgb[i]) == png_snapshot
        else:
            assert rgb_array_to_png_bytes(data.rgb) == png_snapshot
