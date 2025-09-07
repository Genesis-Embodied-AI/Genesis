import itertools
import os
import re
import sys
import time
import enum

import numpy as np
import pyglet
import pytest
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils import set_random_seed
from genesis.utils.image_exporter import FrameImageExporter, as_grayscale_image
from genesis.utils.misc import tensor_to_array

from .conftest import IS_INTERACTIVE_VIEWER_AVAILABLE
from .utils import assert_allclose, assert_array_equal, get_hf_dataset


IMG_STD_ERR_THR = 1.0


class RENDERER_TYPE(enum.IntEnum):
    RASTERIZER = 0
    RAYTRACER = 1
    BATCHRENDER_RASTERIZER = 2
    BATCHRENDER_RAYTRACER = 3


@pytest.fixture(scope="function")
def renderer(renderer_type):
    if renderer_type == RENDERER_TYPE.RASTERIZER:
        return gs.renderers.Rasterizer()
    elif renderer_type == RENDERER_TYPE.RAYTRACER:
        return gs.renderers.RayTracer()
    else:
        return gs.renderers.BatchRenderer(
            use_rasterizer=renderer_type == RENDERER_TYPE.BATCHRENDER_RASTERIZER,
        )


@pytest.fixture(scope="function")
def backend(pytestconfig, renderer_type):
    if renderer_type in (RENDERER_TYPE.BATCHRENDER_RASTERIZER, RENDERER_TYPE.BATCHRENDER_RAYTRACER):
        return gs.cuda

    backend = pytestconfig.getoption("--backend") or gs.cpu
    if isinstance(backend, str):
        return getattr(gs.constants.backend, backend)
    return backend


@pytest.fixture(scope="function", autouse=True)
def skip_if_not_installed(renderer_type):
    if renderer_type in (RENDERER_TYPE.BATCHRENDER_RASTERIZER, RENDERER_TYPE.BATCHRENDER_RAYTRACER):
        pytest.importorskip("gs_madrona", reason="Python module 'gs-madrona' not installed.")


@pytest.mark.required
@pytest.mark.parametrize(
    "renderer_type",
    [RENDERER_TYPE.RASTERIZER, RENDERER_TYPE.BATCHRENDER_RASTERIZER, RENDERER_TYPE.BATCHRENDER_RAYTRACER],
)
@pytest.mark.xfail(sys.platform == "darwin", raises=AssertionError, reason="Flaky on MacOS with CPU-based OpenGL")
def test_render_api(show_viewer, renderer_type, renderer):
    scene = gs.Scene(
        renderer=renderer,
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, 0.0),
            radius=1.0,
            fixed=True,
        ),
    )
    camera = scene.add_camera(
        pos=(0.0, 0.0, 10.0),
        lookat=(0.0, 0.0, 0.0),
        GUI=show_viewer,
    )
    scene.build()

    rgb_arrs, depth_arrs, seg_arrs, normal_arrs = [], [], [], []
    for rgb, depth, seg, normal in itertools.product((True, False), repeat=4):
        rgb_arr, depth_arr, seg_arr, normal_arr = camera.render(rgb=rgb, depth=depth, segmentation=seg, normal=normal)
        if rgb:
            rgb_arrs.append(tensor_to_array(rgb_arr).astype(np.float32))
        if depth:
            depth_arrs.append(tensor_to_array(depth_arr).astype(np.float32))
        if seg:
            seg_arrs.append(tensor_to_array(seg_arr).astype(np.float32))
        if normal:
            normal_arrs.append(tensor_to_array(normal_arr).astype(np.float32))

    if renderer_type == RENDERER_TYPE.BATCHRENDER_RAYTRACER:
        pytest.xfail(reason="'BATCHRENDER_RAYTRACER' is not working for some reason... it always returns empty data.")

    assert_allclose(np.diff(rgb_arrs, axis=0), 0.0, tol=gs.EPS)
    assert_allclose(np.diff(seg_arrs, axis=0), 0.0, tol=gs.EPS)
    assert_allclose(np.diff(normal_arrs, axis=0), 0.0, tol=gs.EPS)

    # Depth is not matching at machine-precision because of MSAA being disabled for depth-only
    msaa_mask = [0, 1, 2, 4, 5, 6] if renderer_type == RENDERER_TYPE.RASTERIZER else slice(None)
    assert_allclose(np.diff(depth_arrs, axis=0)[msaa_mask], 0.0, tol=gs.EPS)


@pytest.mark.required
@pytest.mark.xfail(sys.platform == "darwin", reason="Flaky on MacOS with CPU-based OpenGL")
@pytest.mark.parametrize(
    "renderer_type",
    [RENDERER_TYPE.RASTERIZER, RENDERER_TYPE.BATCHRENDER_RASTERIZER, RENDERER_TYPE.BATCHRENDER_RAYTRACER],
)
def test_deterministic(tmp_path, show_viewer, tol):
    scene = gs.Scene(
        vis_options=gs.options.VisOptions(
            # rendered_envs_idx=(0, 1, 2),
            env_separate_rigid=False,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
        surface=gs.surfaces.Aluminium(
            ior=10.0,
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.1,
            pos=(-0.2, -0.8, 0.2),
            fixed=True,
        ),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ColorTexture(
                color=(1.0, 0.5, 0.5),
            ),
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.1,
            pos=(-0.2, -0.5, 0.2),
            fixed=True,
        ),
        surface=gs.surfaces.Rough(
            color=(1.0, 1.0, 1.0),
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.1,
            pos=(-0.2, -0.2, 0.2),
            fixed=True,
        ),
        surface=gs.surfaces.Smooth(
            color=(0.6, 0.8, 1.0),
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.1,
            pos=(-0.2, 0.2, 0.2),
            fixed=True,
        ),
        surface=gs.surfaces.Iron(
            color=(1.0, 1.0, 1.0),
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.1,
            pos=(-0.2, 0.5, 0.2),
            fixed=True,
        ),
        surface=gs.surfaces.Gold(
            color=(1.0, 1.0, 1.0),
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.1,
            pos=(-0.2, 0.8, 0.2),
            fixed=True,
        ),
        surface=gs.surfaces.Glass(
            color=(1.0, 1.0, 1.0),
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.1,
            pos=(0.2, -0.8, 0.2),
            fixed=True,
        ),
        surface=gs.surfaces.Smooth(
            color=(1.0, 1.0, 1.0, 0.5),
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/wooden_sphere_OBJ/wooden_sphere.obj",
            scale=0.025,
            pos=(0.2, -0.5, 0.2),
            fixed=True,
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/wooden_sphere_OBJ/wooden_sphere.obj",
            scale=0.025,
            pos=(0.2, -0.2, 0.2),
            fixed=True,
        ),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ImageTexture(
                image_path="textures/checker.png",
            )
        ),
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    cam = scene.add_camera(
        pos=(0.9, 0.0, 0.4),
        lookat=(0.0, 0.0, 0.4),
        res=(500, 500),
        fov=60,
        spp=512,
        GUI=False,
    )
    scene.build(n_envs=3, env_spacing=(2.0, 2.0))

    cam.start_recording()
    for _ in range(7):
        dofs_lower_bound, dofs_upper_bound = robot.get_dofs_limit()
        qpos = dofs_lower_bound + (dofs_upper_bound - dofs_lower_bound) * torch.as_tensor(
            np.random.rand(robot.n_qs), dtype=gs.tc_float, device=gs.device
        )

        steps_rgb_arrays = []
        for _ in range(2):
            scene.step()

            robots_rgb_arrays = []
            robot.set_qpos(torch.tile(qpos, (3, 1)))
            if show_viewer:
                scene.visualizer.update()
            for i in range(3):
                pos_i = scene.envs_offset[i] + np.array([0.9, 0.0, 0.4])
                lookat_i = scene.envs_offset[i] + np.array([0.0, 0.0, 0.4])
                cam.set_pose(pos=pos_i, lookat=lookat_i)
                rgb_array, *_ = cam.render(
                    rgb=True, depth=False, segmentation=False, colorize_seg=False, normal=False, force_render=True
                )
                assert np.max(np.std(rgb_array.reshape((-1, 3)), axis=0)) > 10.0
                robots_rgb_arrays.append(rgb_array)
            steps_rgb_arrays.append(robots_rgb_arrays)

        for i in range(3):
            assert_allclose(steps_rgb_arrays[0][i], steps_rgb_arrays[1][i], tol=tol)
    cam.stop_recording(save_to_filename=(tmp_path / "video.mp4"))


@pytest.mark.required
@pytest.mark.parametrize(
    "renderer_type",
    [RENDERER_TYPE.RASTERIZER, RENDERER_TYPE.BATCHRENDER_RASTERIZER, RENDERER_TYPE.BATCHRENDER_RAYTRACER],
)
@pytest.mark.parametrize("n_envs", [0, 4])
@pytest.mark.xfail(sys.platform == "darwin", raises=AssertionError, reason="Flaky on MacOS with CPU-based OpenGL")
def test_render_api_advanced(tmp_path, n_envs, show_viewer, png_snapshot, renderer_type, renderer):
    CAM_RES = (256, 256)
    DIFF_TOL = 0.01
    NUM_STEPS = 5

    IS_BATCHRENDER = renderer_type in (RENDERER_TYPE.BATCHRENDER_RASTERIZER, RENDERER_TYPE.BATCHRENDER_RAYTRACER)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.04,
        ),
        vis_options=gs.options.VisOptions(
            # Disable shadows systematically for Rasterizer because they are forcibly disabled on CPU backend anyway
            shadow=(renderer_type != RENDERER_TYPE.RASTERIZER),
        ),
        renderer=renderer,
    )
    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
        surface=gs.surfaces.Aluminium(
            ior=10.0,
        ),
    )
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            merge_fixed_links=False,
        ),
    )
    cam_debug = scene.add_camera(
        res=(640, 480),
        pos=(1.5, 0.5, 1.5),
        lookat=(0.0, 0.0, 0.5),
        fov=45,
        debug=True,
        GUI=show_viewer,
    )
    cameras = []
    for i in range(max(1 if IS_BATCHRENDER else n_envs, 1)):
        env_idx = None if i < 1 else i
        cam_0 = scene.add_camera(
            res=CAM_RES,
            pos=(1.5, 0.5, 1.5),
            lookat=(0.0, 0.0, 0.5),
            fov=45,
            near=0.05,
            far=100.0,
            env_idx=env_idx,
            GUI=show_viewer,
        )
        cam_1 = scene.add_camera(
            res=CAM_RES,
            pos=(1.5, -0.5, 1.5),
            lookat=(0.0, 0.0, 0.5),
            fov=45,
            near=0.05,
            far=100.0,
            env_idx=env_idx,
            GUI=show_viewer,
        )
        cam_2 = scene.add_camera(
            res=CAM_RES,
            fov=45,
            env_idx=env_idx,
            near=0.05,
            far=100.0,
            GUI=show_viewer,
        )
        cameras += (cam_0, cam_1, cam_2)
    if IS_BATCHRENDER:
        scene.add_light(
            pos=(0.0, 0.0, 1.5),
            dir=(1.0, 1.0, -2.0),
            directional=True,
            castshadow=True,
            cutoff=45.0,
            intensity=0.5,
        )
        scene.add_light(
            pos=(4.0, -4.0, 4.0),
            dir=(-1.0, 1.0, -1.0),
            directional=False,
            castshadow=True,
            cutoff=45.0,
            intensity=0.5,
        )
    scene.build(n_envs=n_envs, env_spacing=(4.0, 4.0))

    # Attach cameras
    for i in range(0, len(cameras), 3):
        cam_0, cam_1, cam_2 = cameras[i : (i + 3)]
        R = np.eye(3)
        trans = np.array([0.1, 0.0, 0.2])
        cam_2.attach(robot.get_link("Head_upper"), gu.trans_R_to_T(trans, R))
        cam_1.follow_entity(robot)

    # Create image exporter
    exporter = FrameImageExporter(tmp_path)

    # Initialize the simulation
    set_random_seed(0)
    for i in range(max(n_envs, 1)):
        qpos = torch.zeros(robot.n_dofs, device=gs.device)
        qpos[:2] = torch.as_tensor(np.random.rand(2), dtype=gs.tc_float, device=gs.device) - 0.5
        qpos[2] = 1.0
        qpos[3:6] = 0.5 * (torch.as_tensor(np.random.rand(3), dtype=gs.tc_float, device=gs.device) - 0.5)
        qpos[6:] = torch.as_tensor(np.random.rand(robot.n_dofs - 6), dtype=gs.tc_float, device=gs.device) - 0.5
        robot.set_dofs_position(qpos, envs_idx=([i] if n_envs else None))

        qvel = torch.zeros(robot.n_dofs, device=gs.device)
        qvel[:6] = torch.as_tensor(np.random.rand(6), dtype=gs.tc_float, device=gs.device) - 0.5
        robot.set_dofs_velocity(qvel, envs_idx=([i] if n_envs else None))

    # Run a few simulation steps while monitoring the result
    cam_debug.start_recording()

    frames_prev = None
    for i in range(NUM_STEPS):
        # Move forward step forward in time
        scene.step()

        # Render cameras
        if IS_BATCHRENDER:
            # Note that the individual cameras is rendered alone first on purpose to make sure it works
            rgba_1, depth_1, seg_1, normal_1 = cam_1.render(
                rgb=True, depth=True, segmentation=True, colorize_seg=True, normal=True
            )
            rgba_all, depth_all, seg_all, normal_all = scene.render_all_cameras(
                rgb=True, depth=True, segmentation=True, colorize_seg=True, normal=True
            )
            assert all(isinstance(img_data, torch.Tensor) for img_data in (rgba_1, depth_1, seg_1, normal_1))
            assert all(
                isinstance(img_data, torch.Tensor) for img_data in (*rgba_all, *depth_all, *seg_all, *normal_all)
            )
        else:
            # Emulate batch rendering which is not supported natively
            colorize_seg = False
            rgba_all, depth_all, seg_all, normal_all = zip(
                *(
                    camera.render(rgb=True, depth=True, segmentation=True, colorize_seg=True, normal=True)
                    for camera in scene._visualizer._cameras
                    if not camera.debug
                )
            )
            if n_envs > 0:
                rgba_all, depth_all, seg_all, normal_all = (
                    tuple(np.swapaxes(np.stack(img_data, axis=0).reshape((n_envs, 3, *img_data[0].shape)), 0, 1))
                    for img_data in (rgba_all, depth_all, seg_all, normal_all)
                )
            rgba_1, depth_1, seg_1, normal_1 = rgba_all[1], depth_all[1], seg_all[1], normal_all[1]

        # Check that the dimensions are valid
        batch_shape = (*((n_envs,) if n_envs else ()), *CAM_RES)
        assert len(rgba_all) == len(depth_all) == 3
        assert all(e.shape == (*batch_shape, 3) for e in (*rgba_all, *seg_all, *normal_all, rgba_1, seg_1, normal_1))
        assert all(e.shape == batch_shape for e in (*depth_all, depth_1))

        # Check that the camera whose output was rendered individually is matching batched output
        for img_data_1, img_data_2 in (
            (rgba_all[1], rgba_1),
            (depth_all[1], depth_1),
            (seg_all[1], seg_1),
            (normal_all[1], normal_1),
        ):
            assert_allclose(img_data_1, img_data_2, tol=gs.EPS)

        # Check that there is something to see here
        depth_normalized_all = tuple(as_grayscale_image(tensor_to_array(img_data)) for img_data in depth_all)
        frame_data = tuple(
            tensor_to_array(img_data).astype(np.float32)
            for img_data in (*rgba_all, *depth_normalized_all, *seg_all, *normal_all)
        )
        for img_data in frame_data:
            for img_data_i in img_data if n_envs else (img_data,):
                assert np.max(np.std(img_data_i.reshape((-1, img_data_i.shape[-1])), axis=0)) > 10.0

        # Export a few frames for later pixel-matching validation
        if i < 2:
            exporter.export_frame_all_cameras(i, rgb=rgba_all, depth=depth_all, segmentation=seg_all, normal=normal_all)
            exporter.export_frame_single_camera(
                i, cam_1.idx, rgb=rgba_1, depth=depth_1, segmentation=seg_1, normal=normal_1
            )

        # Check that cameras are recording different part of the scene
        for rgb_diff in np.diff(frame_data[:3], axis=0):
            for rgb_diff_i in rgb_diff if n_envs else (rgb_diff,):
                assert np.max(np.std(rgb_diff.reshape((-1, rgb_diff_i.shape[-1])), axis=0)) > 10.0

        # Check that images are changing over time.
        # We expect sufficient difference between two consecutive frames.
        if frames_prev is not None:
            for img_data_prev, img_data in zip(frames_prev, frame_data):
                assert np.sum(np.abs(img_data_prev - img_data) > np.finfo(np.float32).eps) > DIFF_TOL * img_data.size
        frames_prev = frame_data

        # Add current frame to monitor video
        rgb_debug, *_ = cam_debug.render(rgb=True, depth=False, segmentation=False, colorize_seg=False, normal=False)
        assert isinstance(rgb_debug, np.ndarray)
        assert rgb_debug.shape == (480, 640, 3)

    assert len(cam_debug._recorded_imgs) == NUM_STEPS
    cam_debug.stop_recording(save_to_filename=(tmp_path / "video.mp4"))

    # Verify that the output is correct pixel-wise over multiple simulation steps
    for image_file in sorted(tmp_path.rglob("*.png")):
        with open(image_file, "rb") as f:
            assert f.read() == png_snapshot


@pytest.mark.field_only
@pytest.mark.parametrize(
    "renderer_type",
    [RENDERER_TYPE.RASTERIZER, RENDERER_TYPE.BATCHRENDER_RASTERIZER, RENDERER_TYPE.BATCHRENDER_RAYTRACER],
)
@pytest.mark.parametrize("segmentation_level", ["entity", "link", "geom"])
@pytest.mark.parametrize("particle_mode", ["visual", "particle"])
def test_segmentation_map(segmentation_level, particle_mode, renderer_type, renderer, show_viewer):
    """Test segmentation rendering."""
    scene = gs.Scene(
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
        ),
        vis_options=gs.options.VisOptions(
            segmentation_level=segmentation_level,
        ),
        show_viewer=False,
        renderer=renderer,
    )

    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/simple/two_link_arm.urdf",
            pos=(-1.0, -1.0, 0.5),
            euler=(0, 0, 90),
        ),
    )

    # We don't test "recon" for vis_mode because it is hard to install.
    materials = ((gs.materials.Rigid(), "visual"),)
    if renderer_type == RENDERER_TYPE.RASTERIZER:
        materials = (
            *materials,
            (gs.materials.Tool(), "visual"),
            (gs.materials.FEM.Elastic(), "visual"),
            (gs.materials.MPM.Elastic(), particle_mode),
            (gs.materials.PBD.Cloth(), particle_mode),
            (gs.materials.SPH.Liquid(), "particle" if particle_mode == "visual" else particle_mode),
            # TODO: Add avatar. Currently avatar solver is buggy.
        )

    ducks = []
    spacing = 0.5
    for i, pack in enumerate(materials):
        col_idx, row_idx = i // 3 - 1, i % 3 - 1
        material, vis_mode = pack
        ducks.append(
            scene.add_entity(
                material=material,
                morph=gs.morphs.Mesh(
                    file="meshes/duck.obj",
                    scale=0.1,
                    pos=(col_idx * spacing, row_idx * spacing, 0.5),
                ),
                surface=gs.surfaces.Default(
                    color=np.random.rand(3),
                    vis_mode=vis_mode,
                ),
            )
        )

    camera = scene.add_camera(
        res=(512, 512),
        pos=(2.0, 0.0, 2.0),
        lookat=(0, 0, 0.5),
        fov=40,
        GUI=show_viewer,
    )
    scene.build()

    seg_num = len(materials) + (2 if segmentation_level == "entity" else 3)
    idx_dict = scene.segmentation_idx_dict
    assert len(idx_dict) == seg_num
    comp_key = 0
    for seg_key in idx_dict.values():
        if isinstance(seg_key, tuple):
            comp_key += 1
    assert comp_key == (0 if segmentation_level == "entity" else 3)

    for i in range(2):
        scene.step()
        _, _, seg, _ = camera.render(rgb=False, depth=False, segmentation=True, colorize_seg=False, normal=False)
        seg = tensor_to_array(seg)
        assert_array_equal(np.sort(np.unique(seg.flat)), np.arange(0, seg_num))


@pytest.mark.required
@pytest.mark.parametrize(
    "renderer_type",
    [RENDERER_TYPE.RASTERIZER, RENDERER_TYPE.BATCHRENDER_RASTERIZER, RENDERER_TYPE.BATCHRENDER_RAYTRACER],
)
def test_point_cloud(renderer_type, renderer, show_viewer):
    N_ENVS = 2
    CAM_RES = (256, 256)
    CAMERA_DIST = 8.0
    OBJ_OFFSET = 10.0
    BOX_HALFSIZE = 1.0
    SPHERE_RADIUS = 1.0

    IS_BATCHRENDER = renderer_type in (RENDERER_TYPE.BATCHRENDER_RASTERIZER, RENDERER_TYPE.BATCHRENDER_RAYTRACER)
    BATCH_SHAPE = (N_ENVS,) if N_ENVS > 0 and IS_BATCHRENDER else ()

    scene = gs.Scene(
        renderer=renderer,
        show_viewer=show_viewer,
        show_FPS=False,
    )
    if renderer_type in (RENDERER_TYPE.BATCHRENDER_RASTERIZER, RENDERER_TYPE.BATCHRENDER_RAYTRACER):
        scene.add_light(
            pos=(0.0, 0.0, 1.5),
            dir=(1.0, 1.0, -2.0),
            directional=True,
            castshadow=True,
            cutoff=45.0,
            intensity=0.5,
        )
        scene.add_light(
            pos=(4.0, -4.0, 4.0),
            dir=(-1.0, 1.0, -1.0),
            directional=False,
            castshadow=True,
            cutoff=45.0,
            intensity=0.5,
        )
    scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.0, OBJ_OFFSET, 0.0),
            radius=SPHERE_RADIUS,
            fixed=True,
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, -OBJ_OFFSET, 0.0),
            size=(2.0 * BOX_HALFSIZE, 2.0 * BOX_HALFSIZE, 2.0 * BOX_HALFSIZE),
            fixed=True,
        )
    )
    camera_sphere = scene.add_camera(
        res=CAM_RES,
        pos=(0.0, OBJ_OFFSET, CAMERA_DIST),
        lookat=(0.0, OBJ_OFFSET, 0.0),
        near=2.0,
        far=15.0,
        GUI=show_viewer,
    )
    camera_box_1 = scene.add_camera(
        res=CAM_RES,
        pos=(0.0, -OBJ_OFFSET, CAMERA_DIST),
        lookat=(0.0, -OBJ_OFFSET, 0.0),
        near=2.0,
        far=15.0,
        GUI=show_viewer,
    )
    camera_box_2 = scene.add_camera(
        res=CAM_RES,
        pos=np.array((CAMERA_DIST, CAMERA_DIST - OBJ_OFFSET, CAMERA_DIST)),
        lookat=(0.0, -OBJ_OFFSET, 0.0),
        near=2.0,
        far=15.0,
        GUI=show_viewer,
    )
    scene.build(n_envs=N_ENVS)

    if show_viewer:
        for camera in scene.visualizer.cameras:
            camera.render(rgb=True, depth=True)

    point_cloud, mask = camera_box_1.render_pointcloud(world_frame=False)
    assert point_cloud.shape == (*BATCH_SHAPE, *CAM_RES, 3)
    point_cloud = point_cloud[mask]
    assert_allclose(CAMERA_DIST - point_cloud[:, 2], BOX_HALFSIZE, atol=1e-4)
    assert np.all(-BOX_HALFSIZE <= point_cloud[:, :2].min(axis=0))
    assert np.all(point_cloud[:, :2].max(axis=0) <= BOX_HALFSIZE)

    point_cloud, mask = camera_box_2.render_pointcloud(world_frame=False)
    assert point_cloud.shape == (*BATCH_SHAPE, *CAM_RES, 3)
    point_cloud = point_cloud[mask]
    point_cloud = point_cloud @ gu.z_up_to_R(np.array((1.0, 1.0, 1.0)), np.array((0.0, 0.0, 1.0))).T
    point_cloud -= np.array((CAMERA_DIST, CAMERA_DIST, CAMERA_DIST))
    # FIXME: Tolerance must be increased whe using Apple's Software Rendering, probably due to an OpenGL bug...
    tol = 2e-4 if sys.platform == "darwin" else 1e-4
    assert_allclose(np.linalg.norm(point_cloud, ord=float("inf"), axis=-1), BOX_HALFSIZE, atol=tol)

    point_cloud, mask = camera_box_2.render_pointcloud(world_frame=True)
    assert point_cloud.shape == (*BATCH_SHAPE, *CAM_RES, 3)
    point_cloud = point_cloud[mask]
    point_cloud += np.array((0.0, OBJ_OFFSET, 0.0))
    assert_allclose(np.linalg.norm(point_cloud, ord=float("inf"), axis=-1), BOX_HALFSIZE, atol=tol)

    # It is not possible to get higher accuracy because of tesselation
    point_cloud, mask = camera_sphere.render_pointcloud(world_frame=False)
    assert point_cloud.shape == (*BATCH_SHAPE, *CAM_RES, 3)
    point_cloud = point_cloud[mask]
    assert_allclose(np.linalg.norm((0.0, 0.0, CAMERA_DIST) - point_cloud, axis=-1), SPHERE_RADIUS, atol=1e-2)


@pytest.mark.required
@pytest.mark.parametrize("renderer_type", [RENDERER_TYPE.RASTERIZER])
def test_debug_draw(show_viewer):
    scene = gs.Scene(
        show_viewer=show_viewer,
    )
    cam = scene.add_camera(
        pos=(3.5, 0.5, 2.5),
        lookat=(0.0, 0.0, 0.5),
        up=(0.0, 0.0, 1.0),
        fov=40,
        res=(640, 640),
        GUI=show_viewer,
    )
    scene.build(n_envs=2)

    rgb_array, *_ = cam.render(rgb=True, depth=False, segmentation=False, colorize_seg=False, normal=False)
    assert_allclose(np.std(rgb_array.reshape((-1, 3)), axis=0), 0.0, tol=gs.EPS)
    scene.draw_debug_arrow(
        pos=(0, 0.4, 0.1),
        vec=(0, 0.3, 0.8),
        color=(1, 0, 0),
    )
    scene.draw_debug_line(
        start=(0.7, -0.3, 0.7),
        end=(0.6, 0.2, 0.7),
        radius=0.01,
        color=(1, 0, 0, 1),
    )
    scene.draw_debug_sphere(
        pos=(-0.3, 0.3, 0.0),
        radius=0.15,
        color=(0, 1, 0),
    )
    scene.draw_debug_frame(
        T=np.array(
            [
                [1.0, 0.0, 0.0, -0.3],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, -0.2],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        axis_length=0.5,
        origin_size=0.03,
        axis_radius=0.02,
    )
    scene.step()
    rgb_array, *_ = cam.render(rgb=True, depth=False, segmentation=False, colorize_seg=False, normal=False)
    if "GS_DISABLE_OFFSCREEN_MARKERS" in os.environ:
        assert_allclose(np.std(rgb_array.reshape((-1, 3)), axis=0), 0.0, tol=gs.EPS)
    else:
        assert np.max(np.std(rgb_array.reshape((-1, 3)), axis=0)) > 10.0
    scene.clear_debug_objects()
    scene.step()
    rgb_array, *_ = cam.render(rgb=True, depth=False, segmentation=False, colorize_seg=False, normal=False)
    assert_allclose(np.std(rgb_array.reshape((-1, 3)), axis=0), 0.0, tol=gs.EPS)


@pytest.mark.required
@pytest.mark.parametrize("renderer_type", [RENDERER_TYPE.RASTERIZER])
@pytest.mark.skipif(not IS_INTERACTIVE_VIEWER_AVAILABLE, reason="Interactive viewer not supported on this platform.")
def test_interactive_viewer_key_press(tmp_path, monkeypatch, png_snapshot, show_viewer):
    IMAGE_FILENAME = tmp_path / "screenshot.png"

    # Mock 'get_save_filename' to avoid poping up an interactive dialog
    def get_save_filename(self, file_exts):
        return IMAGE_FILENAME

    monkeypatch.setattr("genesis.ext.pyrender.viewer.Viewer._get_save_filename", get_save_filename)

    # Mock 'on_key_press' to determine whether requests have been processed
    is_done = False
    on_key_press_orig = gs.ext.pyrender.viewer.Viewer.on_key_press

    def on_key_press(self, symbol: int, modifiers: int):
        nonlocal is_done
        assert not is_done
        ret = on_key_press_orig(self, symbol, modifiers)
        is_done = True
        return ret

    monkeypatch.setattr("genesis.ext.pyrender.viewer.Viewer.on_key_press", on_key_press)

    # Create a scene
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            # Force screen-independent low-quality resolution when running unit tests for consistency
            res=(640, 480),
            # Enable running in background thread if supported by the platform
            run_in_thread=sys.platform == "linux",
        ),
        show_viewer=True,
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.5, 0.5, 0.5),
            pos=(0.0, 0.0, 0.0),
        ),
    )
    scene.build()
    pyrender_viewer = scene.visualizer.viewer._pyrender_viewer
    assert pyrender_viewer.is_active

    # Try saving the current frame
    pyrender_viewer.dispatch_event("on_key_press", pyglet.window.key.S, 0)

    # Waiting for request completion
    if pyrender_viewer.run_in_thread:
        for i in range(100):
            if is_done:
                is_done = False
                break
            time.sleep(0.1)
        else:
            raise AssertionError("Keyboard event not processed before timeout")
    else:
        pyrender_viewer.dispatch_pending_events()
        pyrender_viewer.dispatch_events()

    # Skip the rest of the test if necessary
    if sys.platform == "linux":
        glinfo = pyrender_viewer.context.get_info()
        renderer = glinfo.get_renderer()
        if "llvmpipe" in renderer:
            llvm_version = re.search(r"LLVM\s+([\d.]+)", renderer).group(1)
            if llvm_version < "20":
                pytest.xfail("Text is blurry on Linux using old CPU-based Mesa rendering driver.")

    # Make sure that the result is valid
    with open(IMAGE_FILENAME, "rb") as f:
        assert f.read() == png_snapshot


@pytest.mark.parametrize(
    "renderer_type",
    [RENDERER_TYPE.RASTERIZER],
)
def test_render_planes(tmp_path, png_snapshot, renderer):
    CAM_RES = (256, 256)

    for test_idx, (plane_size, tile_size) in enumerate(
        (
            ((3, 4.5), (0.5, 0.75)),
            ((3.0, 5.0), (5.0, 3.0)),
            ((4.0, 4.0), (1.0, 1.0)),
        )
    ):
        scene = gs.Scene(
            renderer=renderer,
        )
        plane = scene.add_entity(
            gs.morphs.Plane(plane_size=plane_size, tile_size=tile_size),
        )
        camera = scene.add_camera(
            res=CAM_RES,
            pos=(0.0, 0.0, 8),
            lookat=(0.0, 0.0, 0.0),
            fov=45,
            GUI=False,
        )
        scene.build()

        exporter = FrameImageExporter(tmp_path)
        rgba, depth, _, _ = camera.render(rgb=True, depth=False)
        exporter.export_frame_single_camera(test_idx, camera.idx, rgb=rgba, depth=depth)

    for image_file in sorted(tmp_path.rglob("*.png")):
        with open(image_file, "rb") as f:
            assert f.read() == png_snapshot
