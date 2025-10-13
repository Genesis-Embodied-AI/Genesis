import numpy as np
import pytest
import torch

import genesis as gs

from .utils import assert_allclose, get_hf_dataset


pytestmark = [
    pytest.mark.field_only,
]


@pytest.mark.required
def test_rigid_mpm_muscle(show_viewer):
    BALL_POS_INIT = (0.8, 0.6, 0.12)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=3e-3,
            substeps=10,
        ),
        rigid_options=gs.options.RigidOptions(
            gravity=(0, 0, -9.8),
            constraint_timeconst=0.02,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(0.0, 0.0, -0.2),
            upper_bound=(1.0, 1.0, 1.0),
            gravity=(0.0, 0.0, 0.0),  # mimic gravity compensation
            enable_CPIC=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 1.3, 0.5),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    scene.add_entity(morph=gs.morphs.Plane())
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/simple/two_link_arm.urdf",
            pos=(0.45, 0.45, 0.2),
            euler=(0.0, 0.0, 0.0),
            scale=0.2,
            fixed=True,
        ),
        material=gs.materials.Hybrid(
            material_rigid=gs.materials.Rigid(
                gravity_compensation=1.0,
            ),
            material_soft=gs.materials.MPM.Muscle(
                E=1e4,
                nu=0.45,
                rho=1000.0,
                model="neohooken",
            ),
            thickness=0.05,
            damping=1000.0,
        ),
    )
    ball = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=BALL_POS_INIT,
            radius=0.12,
        ),
        material=gs.materials.Rigid(rho=1000, friction=0.5),
    )
    scene.build()

    scene.reset()
    for i in range(150):
        robot.control_dofs_velocity(np.array([2.0 * np.sin(2 * np.pi * i * 0.006)] * robot.n_dofs))
        scene.step()

    ball_pos_delta = ball.get_pos() - torch.tensor(BALL_POS_INIT, dtype=gs.tc_float, device=gs.device)
    assert_allclose(ball_pos_delta[..., 0], 0.0, tol=1e-2)
    assert ((0.02 < ball_pos_delta[1]) & (ball_pos_delta[1] < 0.05)).all()
    assert_allclose(ball_pos_delta[..., 2], 0.0, tol=1e-3)


@pytest.mark.required
@pytest.mark.parametrize(
    "material_type",
    [
        gs.materials.PBD.Liquid,
        gs.materials.SPH.Liquid,
        gs.materials.MPM.Liquid,
        gs.materials.MPM.Sand,
        gs.materials.MPM.Snow,
        gs.materials.MPM.Elastic,  # This makes little sense but nothing prevents doing this
    ],
)
def test_fluid_emitter(material_type, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=4e-3,
            substeps=10,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(0.0, -1.5, 0.0),
            upper_bound=(1.0, 1.5, 4.0),
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=0.02,
        ),
        pbd_options=gs.options.PBDOptions(
            particle_size=0.02,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(5.5, 6.5, 3.2),
            camera_lookat=(0.5, 1.5, 1.5),
            camera_fov=35,
            max_FPS=120,
        ),
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=[0],
        ),
        show_viewer=show_viewer,
    )
    plane = scene.add_entity(gs.morphs.Plane())
    wheel = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/wheel/fancy_wheel.urdf",
            pos=(0.5, 0.25, 1.6),
            euler=(0, 0, 0),
            fixed=True,
            convexify=False,
        ),
    )
    emitter = scene.add_emitter(
        material=material_type(),
        max_particles=1400,
        surface=gs.surfaces.Glass(
            color=(0.7, 0.85, 1.0, 0.7),
        ),
    )
    scene.build(n_envs=2)

    emitter.emit_omni()
    for i in range(5):
        emitter.emit(droplet_shape="circle", droplet_size=0.25)
        scene.step()
    scene.step()


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_sap_rigid_rigid_hydroelastic_contact(show_viewer):
    BOX_POS = (0.0, 0.0, 0.1)
    BOX_HALFHEIGHT = 0.1

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1 / 60,
            substeps=2,
        ),
        coupler_options=gs.options.SAPCouplerOptions(
            pcg_threshold=1e-10,
            sap_convergence_atol=1e-10,
            sap_convergence_rtol=1e-10,
            linesearch_ftol=1e-10,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    plane = scene.add_entity(
        gs.morphs.Plane(
            collision=False,
        ),
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.5, 0.5, 2 * BOX_HALFHEIGHT),
            pos=(0.0, 0.0, BOX_HALFHEIGHT),
        ),
        material=gs.materials.Rigid(),
    )
    asset_path = get_hf_dataset(pattern="heavy_three_joint_link.xml")
    robot_1 = scene.add_entity(
        gs.morphs.MJCF(
            file=f"{asset_path}/heavy_three_joint_link.xml",
            pos=(-0.2, -0.26, 0.0),
            scale=0.3,
        ),
    )
    robot_2 = scene.add_entity(
        gs.morphs.MJCF(
            file=f"{asset_path}/heavy_three_joint_link.xml",
            pos=(0.17, -0.26, 0.1),
            euler=(0.0, 0.0, 90.0),
            scale=0.3,
        ),
    )
    scene.build()

    # Run simulation
    for _ in range(80):
        scene.step()

    # All the entities must be still
    for entity in scene.entities:
        assert_allclose(entity.get_links_vel(), 0.0, atol=2e-2)

    # The box should stay at its initial position
    assert_allclose(box.get_pos(), (0.0, 0.0, BOX_HALFHEIGHT), atol=2e-3)

    # The box, and both robots should be laying on top of each other
    robot_1_min_corner, robot_1_max_corner = robot_1.get_AABB()
    robot_2_min_corner, robot_2_max_corner = robot_2.get_AABB()
    assert (robot_1_min_corner[:2] > -0.4).all() and (robot_2_min_corner[:2] > -0.4).all()
    assert (robot_1_min_corner[:2] < 0.4).all() and (robot_2_min_corner[:2] < 0.4).all()
    assert robot_1_max_corner[2] > 2 * BOX_HALFHEIGHT
    assert robot_2_max_corner[2] > robot_1_max_corner[2] + 0.05


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_sap_fem_vs_robot(show_viewer):
    SPHERE_RADIUS = 0.2

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1 / 60,
            substeps=2,
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
        ),
        coupler_options=gs.options.SAPCouplerOptions(),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, 1.5, 1.2),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    plane = scene.add_entity(
        gs.morphs.Plane(
            collision=False,
        ),
    )
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, SPHERE_RADIUS),
            radius=SPHERE_RADIUS,
        ),
        material=gs.materials.FEM.Elastic(
            E=1e5,
            nu=0.4,
            model="linear_corotated",
        ),
    )
    asset_path = get_hf_dataset(pattern="cross.xml")
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file=f"{asset_path}/cross.xml",
            pos=(0.0, 0.0, 2 * SPHERE_RADIUS + 0.04),
            scale=0.5,
        ),
    )
    scene.build()

    # Run the simulation
    for _ in range(50):
        scene.step()

    # Check that the sphere did not move, and the slightly squished
    state = sphere.get_state()
    center = state.pos.mean(axis=(0, 1))
    assert_allclose(center[:2], 0.0, tol=0.01)
    assert center[2] < SPHERE_RADIUS - 0.02

    # Check that the ant is laying on top of the sphere
    robot_pos = robot.get_pos()
    assert_allclose(robot_pos[:2], 0.0, tol=0.03)
    assert robot_pos[2] > (2 * SPHERE_RADIUS + 0.04) - 0.05

    # Check that the legs of the ants are resting on the sphere
    assert_allclose(robot.get_qpos()[-4:].abs(), 1.0, tol=0.1)
