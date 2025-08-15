import itertools
import queue
import os
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from syrupy.extensions.image import PNGImageSnapshotExtension
from syrupy.location import PyTestLocation

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils import set_random_seed
from genesis.utils.image_exporter import FrameImageExporter

from .utils import assert_allclose, assert_array_equal, get_hf_dataset


IMG_STD_ERR_THR = 0.8


class PixelMatchSnapshotExtension(PNGImageSnapshotExtension):
    def matches(self, *, serialized_data, snapshot_data) -> bool:
        img_arrays = []
        for data in (serialized_data, snapshot_data):
            buffer = BytesIO()
            buffer.write(data)
            buffer.seek(0)
            img_arrays.append(np.asarray(Image.open(buffer)))
        img_delta = img_arrays[1].astype(np.int32) - img_arrays[0].astype(np.int32)
        return np.std(img_delta) < IMG_STD_ERR_THR

    # def diff_snapshots(self, serialized_data, snapshot_data) -> "SerializableData":
    #     # re-run pixelmatch and return a diff image (can cache on the class instance)
    #     pass


@pytest.fixture
def png_snapshot(snapshot):
    return snapshot.use_extension(PixelMatchSnapshotExtension)


@pytest.mark.required
@pytest.mark.parametrize("segmentation_level", ["entity", "link"])
@pytest.mark.parametrize("particle_mode", ["visual", "particle"])
def test_segmentation(segmentation_level, particle_mode):
    """Test segmentation rendering."""
    scene = gs.Scene(
        fem_options=gs.options.FEMOptions(use_implicit_solver=True),
        vis_options=gs.options.VisOptions(segmentation_level=segmentation_level),
        show_viewer=False,
    )

    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/simple/two_link_arm.urdf",
            pos=(-1.0, -1.0, 0.5),
            euler=(0, 0, 90),
        ),
    )

    # We don't test "recon" for vis_mode because it is hard to install.
    sph_mode = "particle" if particle_mode == "visual" else particle_mode
    materials = [
        (gs.materials.Rigid(), "visual"),
        (gs.materials.Tool(), "visual"),
        (gs.materials.FEM.Elastic(), "visual"),
        (gs.materials.MPM.Elastic(), particle_mode),
        (gs.materials.PBD.Cloth(), particle_mode),
        (gs.materials.SPH.Liquid(), sph_mode),
        # TODO: Add avatar. Currently avatar solver is buggy.
    ]
    ducks = []
    spacing = 0.5
    for i, pack in enumerate(materials):
        col_idx = i // 3 - 1
        row_idx = i % 3 - 1
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
    )
    scene.build()

    seg_num = len(materials) + (3 if segmentation_level == "link" else 2)
    idx_dict = camera.get_segmentation_idx_dict()
    assert len(idx_dict) == seg_num
    comp_key = 0
    for seg_key in idx_dict.values():
        if isinstance(seg_key, tuple):
            comp_key += 1
    assert comp_key == (3 if segmentation_level == "link" else 0)

    for i in range(2):
        scene.step()
        _, _, seg, _ = camera.render(rgb=False, depth=False, segmentation=True, colorize_seg=False, normal=False)
        assert_array_equal(np.sort(np.unique(seg.flat)), np.arange(0, seg_num))


@pytest.mark.required
@pytest.mark.xfail(sys.platform == "darwin", reason="Flaky on MacOS with CPU-based OpenGL")
def test_batched_offscreen_rendering(tmp_path, show_viewer, tol):
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
        qpos = dofs_lower_bound + (dofs_upper_bound - dofs_lower_bound) * torch.rand(robot.n_qs)

        steps_rgb_arrays = []
        for _ in range(2):
            scene.step()

            robots_rgb_arrays = []
            robot.set_qpos(torch.tile(qpos, (3, 1)))
            scene.visualizer.update()
            for i in range(3):
                pos_i = scene.envs_offset[i] + np.array([0.9, 0.0, 0.4])
                lookat_i = scene.envs_offset[i] + np.array([0.0, 0.0, 0.4])
                cam.set_pose(pos=pos_i, lookat=lookat_i)
                rgb_array, *_ = cam.render(rgb=True, depth=False, segmentation=False, colorize_seg=False, normal=False)
                assert np.max(np.std(rgb_array.reshape((-1, 3)), axis=0)) > 10.0
                robots_rgb_arrays.append(rgb_array)

            steps_rgb_arrays.append(robots_rgb_arrays)

        for i in range(3):
            assert_allclose(steps_rgb_arrays[0][i], steps_rgb_arrays[1][i], tol=tol)
    cam.stop_recording(save_to_filename=(tmp_path / "video.mp4"))


@pytest.mark.required
@pytest.mark.xfail(sys.platform == "darwin", reason="Flaky on MacOS with CPU-based OpenGL")
def test_render_api(show_viewer):
    scene = gs.Scene(
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
            rgb_arrs.append(rgb_arr.astype(np.float32))
        if depth:
            depth_arrs.append(depth_arr.astype(np.float32))
        if seg:
            seg_arrs.append(seg_arr.astype(np.float32))
        if normal:
            normal_arrs.append(normal_arr.astype(np.float32))

    assert_allclose(np.diff(rgb_arrs, axis=0), 0.0, tol=gs.EPS)
    assert_allclose(np.diff(seg_arrs, axis=0), 0.0, tol=gs.EPS)
    assert_allclose(np.diff(normal_arrs, axis=0), 0.0, tol=gs.EPS)
    # Depth is not matching at machine-precision because of MSAA being disabled for depth-only
    # FIXME: There is one pixel off on MacOS with Apple's Software Rendering, probably due to a bug...
    tol = 2e-4 if sys.platform == "darwin" else gs.EPS
    assert_allclose(np.diff(depth_arrs, axis=0)[[0, 1, 2, 4, 5, 6]], 0.0, tol=tol)


@pytest.mark.required
def test_point_cloud(show_viewer):
    CAMERA_DIST = 8.0
    OBJ_OFFSET = 10.0
    BOX_HALFSIZE = 1.0
    SPHERE_RADIUS = 1.0

    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.0, OBJ_OFFSET, 0.0),
            radius=SPHERE_RADIUS,
            fixed=True,
        ),
    )
    camera_sphere = scene.add_camera(
        pos=(0.0, OBJ_OFFSET, CAMERA_DIST),
        lookat=(0.0, OBJ_OFFSET, 0.0),
        GUI=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, -OBJ_OFFSET, 0.0),
            size=(2.0 * BOX_HALFSIZE, 2.0 * BOX_HALFSIZE, 2.0 * BOX_HALFSIZE),
            fixed=True,
        )
    )
    camera_box_1 = scene.add_camera(
        pos=(0.0, -OBJ_OFFSET, CAMERA_DIST),
        lookat=(0.0, -OBJ_OFFSET, 0.0),
        GUI=show_viewer,
    )
    camera_box_2 = scene.add_camera(
        pos=np.array((CAMERA_DIST, CAMERA_DIST - OBJ_OFFSET, CAMERA_DIST)),
        lookat=(0.0, -OBJ_OFFSET, 0.0),
        GUI=show_viewer,
    )
    for camera in scene.visualizer.cameras:
        camera._near = 2.0
        camera._far = 15.0
    scene.build()

    if show_viewer:
        for camera in scene.visualizer.cameras:
            camera.render(rgb=True, depth=True)

    point_cloud, mask = camera_box_1.render_pointcloud(world_frame=False)
    point_cloud = point_cloud[mask]
    assert_allclose(CAMERA_DIST - point_cloud[:, 2], BOX_HALFSIZE, atol=1e-4)
    assert np.all(-BOX_HALFSIZE <= point_cloud[:, :2].min(axis=0))
    assert np.all(point_cloud[:, :2].max(axis=0) <= BOX_HALFSIZE)

    point_cloud, mask = camera_box_2.render_pointcloud(world_frame=False)
    point_cloud = point_cloud[mask]
    point_cloud = point_cloud @ gu.z_up_to_R(np.array((1.0, 1.0, 1.0)), np.array((0.0, 0.0, 1.0))).T
    point_cloud -= np.array((CAMERA_DIST, CAMERA_DIST, CAMERA_DIST))
    # FIXME: Tolerance must be increased whe using Apple's Software Rendering, probably due to a bug...
    tol = 2e-4 if sys.platform == "darwin" else 1e-4
    assert_allclose(np.linalg.norm(point_cloud, ord=float("inf"), axis=-1), BOX_HALFSIZE, atol=tol)

    point_cloud, mask = camera_box_2.render_pointcloud(world_frame=True)
    point_cloud = point_cloud[mask]
    point_cloud += np.array((0.0, OBJ_OFFSET, 0.0))
    assert_allclose(np.linalg.norm(point_cloud, ord=float("inf"), axis=-1), BOX_HALFSIZE, atol=tol)

    # It is not possible to get higher accuracy because of tesselation
    point_cloud, mask = camera_sphere.render_pointcloud(world_frame=False)
    point_cloud = point_cloud[mask]
    assert_allclose(np.linalg.norm((0.0, 0.0, CAMERA_DIST) - point_cloud, axis=-1), SPHERE_RADIUS, atol=1e-2)


@pytest.mark.required
def test_splashsurf_surface_reconstruction(show_viewer):
    scene = gs.Scene(
        show_viewer=show_viewer,
    )
    water = scene.add_entity(
        material=gs.materials.SPH.Liquid(),
        morph=gs.morphs.Box(
            pos=(0.15, 0.15, 0.22),
            size=(0.25, 0.25, 0.4),
        ),
        surface=gs.surfaces.Default(
            color=(0.2, 0.6, 1.0, 1.0),
            vis_mode="recon",
        ),
    )
    cam = scene.add_camera(
        pos=(1.3, 1.3, 0.8),
        lookat=(0.0, 0.0, 0.2),
        GUI=show_viewer,
    )
    scene.build()
    cam.render(rgb=True, depth=False, segmentation=False, colorize_seg=False, normal=False)


@pytest.mark.required
def test_batched_mounted_camera_rendering(show_viewer, tol):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.2,
            substeps=10,
        ),
        vis_options=gs.options.VisOptions(
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
            pos=(-0.2, -0.5, 0.2),
            fixed=True,
        ),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ColorTexture(
                color=(1.0, 0.5, 0.5),
            ),
        ),
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    n_envs = 3
    env_spacing = (2.0, 2.0)
    cams = [scene.add_camera(GUI=show_viewer, fov=70, env_idx=i_b) for i_b in range(n_envs)]
    scene.build(n_envs=n_envs, env_spacing=env_spacing)

    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    trans = np.array([0.1, 0.0, 0.1])
    for cam in cams:
        cam.attach(robot.get_link("hand"), gu.trans_R_to_T(trans, R))

    target_quat = np.tile(np.array([0, 1, 0, 0]), (n_envs, 1))  # pointing downwards
    center = np.tile(np.array([-0.25, -0.25, 0.5]), (n_envs, 1))
    rng = np.random.default_rng(42)
    angular_speed = rng.uniform(-10, 10, n_envs)
    r = 0.25

    ee_link = robot.get_link("hand")

    steps_rgb_queue: queue.Queue[list[np.ndarray]] = queue.Queue(maxsize=2)
    for i in range(20):
        target_pos = center.copy()
        target_pos[:, 0] += np.cos(i / 360 * np.pi * angular_speed) * r
        target_pos[:, 1] += np.sin(i / 360 * np.pi * angular_speed) * r

        q = robot.inverse_kinematics(
            link=ee_link,
            pos=target_pos,
            quat=target_quat,
            rot_mask=[False, False, True],  # for demo purpose: only restrict direction of z-axis
        )
        robot.set_qpos(q)
        scene.step()

        robots_rgb_arrays = []
        for cam in cams:
            rgb_array, *_ = cam.render(rgb=True, depth=False, segmentation=False, colorize_seg=False, normal=False)
            assert np.max(np.std(rgb_array.reshape((-1, 3)), axis=0)) > 10.0
            robots_rgb_arrays.append(rgb_array.astype(np.int32))
        steps_rgb_queue.put(robots_rgb_arrays)

        # Make sure that cameras are recording different part of the scene
        for rgb_diff in np.diff(robots_rgb_arrays, axis=0):
            assert np.std(rgb_diff) > 10.0

        # Make sure that the cameras are moving alongside the robot
        # We expect atlest 2% difference between two consecutive frames
        if steps_rgb_queue.full():
            diff_tol = 0.02
            frames_t_minus_1 = steps_rgb_queue.get()
            frames_t = steps_rgb_queue.get()
            for i in range(n_envs):
                diff = frames_t[i] - frames_t_minus_1[i]
                assert np.count_nonzero(diff) > diff_tol * np.prod(diff.shape)


@pytest.mark.required
def test_debug_draw(show_viewer):
    scene = gs.Scene(
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
        ),
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
@pytest.mark.parametrize("backend", [gs.cuda])
@pytest.mark.parametrize("use_rasterizer", [True, False])
@pytest.mark.parametrize("render_all_cameras", [True, False])
@pytest.mark.parametrize("n_envs", [0, 4])
def test_madrona_batch_rendering(tmp_path, use_rasterizer, render_all_cameras, n_envs, show_viewer, png_snapshot):
    CAM_RES = (256, 256)

    pytest.importorskip("gs_madrona", reason="Python module 'gs-madrona' not installed.")

    snapshot_dir = Path(PixelMatchSnapshotExtension.dirname(test_location=png_snapshot.test_location))
    get_hf_dataset(pattern=f"{snapshot_dir.name}/*", repo_name="snapshots", local_dir=snapshot_dir.parent)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.02,
            substeps=4,
        ),
        renderer=gs.options.renderers.BatchRenderer(
            use_rasterizer=use_rasterizer,
        ),
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
    cam_0 = scene.add_camera(
        res=CAM_RES,
        pos=(1.5, 0.5, 1.5),
        lookat=(0.0, 0.0, 0.5),
        fov=45,
        GUI=show_viewer,
    )
    cam_1 = scene.add_camera(
        res=CAM_RES,
        pos=(1.5, -0.5, 1.5),
        lookat=(0.0, 0.0, 0.5),
        fov=45,
        GUI=show_viewer,
    )
    cam_2 = scene.add_camera(
        res=CAM_RES,
        fov=45,
        GUI=show_viewer,
    )
    scene.add_light(
        pos=[0.0, 0.0, 1.5],
        dir=[1.0, 1.0, -2.0],
        directional=True,
        castshadow=True,
        cutoff=45.0,
        intensity=0.5,
    )
    scene.add_light(
        pos=[4.0, -4.0, 4.0],
        dir=[-1.0, 1.0, -1.0],
        directional=False,
        castshadow=True,
        cutoff=45.0,
        intensity=0.5,
    )
    scene.build(n_envs=n_envs, env_spacing=(4.0, 4.0))

    # Attach cameras
    R = np.eye(3)
    trans = np.array([0.1, 0.0, 0.1])
    cam_2.attach(robot.get_link("Head_upper"), gu.trans_R_to_T(trans, R))
    cam_1.follow_entity(robot)

    # Create an image exporter
    exporter = FrameImageExporter(tmp_path)

    # Initialize the simulation
    set_random_seed(0)
    dof_bounds = scene.rigid_solver.dofs_info.limit.to_torch(gs.device)
    for i in range(max(n_envs, 1)):
        qpos = torch.zeros(robot.n_dofs)
        qpos[:2] = torch.rand(2) - 0.5
        qpos[2] = 1.0
        qpos[3:6] = 0.5 * (torch.rand(3) - 0.5)
        qpos[6:] = torch.rand(robot.n_dofs - 6) - 0.5
        robot.set_dofs_position(qpos, envs_idx=([i] if n_envs else None))

        qvel = torch.zeros(robot.n_dofs)
        qvel[:6] = torch.rand(6) - 0.5
        robot.set_dofs_velocity(qvel, envs_idx=([i] if n_envs else None))

    # Verify that the output is correct pixel-wise over multiple simulation steps
    for i in range(2):
        batch_shape = (*((n_envs,) if n_envs else ()), *CAM_RES)
        if render_all_cameras:
            rgba, depth, _, _ = scene.render_all_cameras(rgb=True, depth=True)

            assert len(rgba) == len(depth) == len(scene.visualizer.cameras)
            assert all(e.shape == (*batch_shape, 3) for e in rgba)
            assert all(e.shape == (*batch_shape, 1) for e in depth)

            exporter.export_frame_all_cameras(i, rgb=rgba, depth=depth)
        else:
            rgba, depth, _, _ = cam_1.render(rgb=True, depth=True)

            assert rgba.shape == (*batch_shape, 3)
            assert depth.shape == (*batch_shape, 1)

            exporter.export_frame_single_camera(i, cam_1.idx, rgb=rgba, depth=depth)

        scene.step()

    for image_file in sorted(tmp_path.rglob("*.png")):
        with open(image_file, "rb") as f:
            assert f.read() == png_snapshot


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu])
def test_2_channels_luminance_alpha_textures(show_viewer):
    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )
    asset_path = get_hf_dataset(pattern="fridge/*")
    fridge = scene.add_entity(
        gs.morphs.URDF(
            file=f"{asset_path}/fridge/fridge.urdf",
            fixed=True,
        )
    )
    scene.build()
