import queue
import sys

import numpy as np
import pytest
import torch

import genesis as gs

from .utils import assert_allclose, assert_array_equal


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
@pytest.mark.flaky(reruns=3, condition=(sys.platform == "darwin"))
def test_batched_offscreen_rendering(show_viewer, tol):
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

    for _ in range(10):
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
                assert np.std(rgb_array) > 10.0
                robots_rgb_arrays.append(rgb_array)

            steps_rgb_arrays.append(robots_rgb_arrays)

        for i in range(3):
            assert_allclose(steps_rgb_arrays[0][i], steps_rgb_arrays[1][i], tol=tol)


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
    cams = [scene.add_camera(GUI=show_viewer, fov=70) for _ in range(n_envs)]
    scene.build(n_envs=n_envs, env_spacing=env_spacing)

    T = np.eye(4)
    T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    T[:3, 3] = np.array([0.1, 0.0, 0.1])
    for nenv, cam in enumerate(cams):
        cam.attach(robot.get_link("hand"), T, nenv)

    target_quat = np.tile(np.array([0, 1, 0, 0]), [n_envs, 1])  # pointing downwards
    center = np.tile(np.array([-0.25, -0.25, 0.5]), [n_envs, 1])
    rng = np.random.default_rng(42)
    angular_speed = rng.uniform(-10, 10, n_envs)
    r = 0.25

    ee_link = robot.get_link("hand")

    steps_rgb_queue: queue.Queue[list[np.ndarray]] = queue.Queue(maxsize=2)

    for i in range(50):
        target_pos = np.zeros([n_envs, 3])
        target_pos[:, 0] = center[:, 0] + np.cos(i / 360 * np.pi * angular_speed) * r
        target_pos[:, 1] = center[:, 1] + np.sin(i / 360 * np.pi * angular_speed) * r
        target_pos[:, 2] = center[:, 2]

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
            assert np.std(rgb_array) > 10.0
            robots_rgb_arrays.append(rgb_array)
        steps_rgb_queue.put(robots_rgb_arrays)

        if steps_rgb_queue.full():  # we have a set of 2 consecutive frames
            diff_tol = 0.02  # expect atlest 2% difference between each frame
            frames_t_minus_1 = steps_rgb_queue.get()
            frames_t = steps_rgb_queue.get()
            for i in range(n_envs):
                diff = frames_t[i] - frames_t_minus_1[i]
                assert np.count_nonzero(diff) > diff_tol * np.prod(diff.shape)
