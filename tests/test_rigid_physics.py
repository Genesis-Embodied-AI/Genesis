import pytest
import xml.etree.ElementTree as ET

import numpy as np

import mujoco
import genesis as gs
from genesis.ext import trimesh

from .utils import (
    init_simulators,
    check_mujoco_model_consistency,
    check_mujoco_data_consistency,
    simulate_and_check_mujoco_consistency,
)


@pytest.fixture
def xml_path(request, tmp_path, model_name):
    mjcf = request.getfixturevalue(model_name)
    xml_tree = ET.ElementTree(mjcf)
    file_path = tmp_path / f"{model_name}.xml"
    xml_tree.write(file_path, encoding="utf-8", xml_declaration=True)
    return str(file_path)


@pytest.fixture(scope="session")
def box_plan():
    """Generate an XML model for a box on a plane."""
    mjcf = ET.Element("mujoco", model="one_box")
    ET.SubElement(mjcf, "option", timestep="0.01")
    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "geom", contype="1", conaffinity="1", condim="3", friction="1. 0.5 0.5")
    worldbody = ET.SubElement(mjcf, "worldbody")
    ET.SubElement(worldbody, "geom", type="plane", name="floor", pos="0. 0. 0.", size="40. 40. 40.")
    box_body = ET.SubElement(worldbody, "body", name="box", pos="0. 0. 0.3")
    ET.SubElement(box_body, "geom", type="box", size="0.2 0.2 0.2", pos="0. 0. 0.")
    ET.SubElement(box_body, "joint", name="root", type="free")
    return mjcf


@pytest.fixture(scope="session")
def box_box():
    """Generate an XML model for two boxes."""
    mjcf = ET.Element("mujoco", model="one_box")
    ET.SubElement(mjcf, "option", timestep="0.01")  # FIXME: It only works for 5ms
    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "geom", contype="1", conaffinity="1", condim="3", friction="1. 0.5 0.5")
    worldbody = ET.SubElement(mjcf, "worldbody")
    ET.SubElement(worldbody, "geom", type="plane", name="floor", pos="0. 0. 0.", size="40. 40. 40.")
    box1_body = ET.SubElement(worldbody, "body", name="box1", pos="0. 0. 0.2")
    ET.SubElement(box1_body, "geom", type="box", size="0.2 0.2 0.2", pos="0. 0. 0.", rgba="0 1 0 0.4")
    ET.SubElement(box1_body, "joint", name="root1", type="free")
    box2_body = ET.SubElement(worldbody, "body", name="box2", pos="0. 0. 0.8")
    ET.SubElement(box2_body, "geom", type="box", size="0.2 0.2 0.2", pos="0. 0. 0.", rgba="0 0 1 0.4")
    ET.SubElement(box2_body, "joint", name="root2", type="free")
    return mjcf


def _build_chain_capsule_hinge(asset_tmp_path, enable_mesh):
    if enable_mesh:
        mesh_path = str(asset_tmp_path / "capsule.obj")
        tmesh = trimesh.creation.icosphere(radius=1.0, subdivisions=1)
        tmesh.apply_transform(np.diag([0.05, 0.05, 0.25, 1]))
        tmesh.export(mesh_path, file_type="obj")

    mjcf = ET.Element("mujoco", model="two_stick_robot")
    ET.SubElement(mjcf, "option", timestep="0.05")
    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "geom", contype="1", conaffinity="1", condim="3")
    if enable_mesh:
        asset = ET.SubElement(mjcf, "asset")
        ET.SubElement(asset, "mesh", name="capsule", refpos="0 0 -0.25", refquat="0.707 0 -0.707 0", file=mesh_path)
    worldbody = ET.SubElement(mjcf, "worldbody")
    link0 = ET.SubElement(worldbody, "body", name="body1", pos="0.1 0.2 0.0", quat="0.707 0 0.707 0")
    if enable_mesh:
        ET.SubElement(link0, "geom", type="mesh", mesh="capsule", rgba="0 0 1 0.3")
    else:
        ET.SubElement(link0, "geom", type="capsule", fromto="0 0 0 0.5 0 0", size="0.05", rgba="0 0 1 0.3")
    link1 = ET.SubElement(link0, "body", name="body2", pos="0.5 0.2 0.0", quat="0.92388 0 0 0.38268")
    if enable_mesh:
        ET.SubElement(link1, "geom", type="mesh", mesh="capsule")
    else:
        ET.SubElement(link1, "geom", type="capsule", fromto="0 0 0 0.5 0 0", size="0.05")
    ET.SubElement(link1, "joint", type="hinge", name="joint1", axis="0 0 1", pos="0.0 0.0 0.0")
    link2 = ET.SubElement(link1, "body", name="body3", pos="0.5 0.2 0.0", quat="0.92388 0 0.38268 0.0")
    if enable_mesh:
        ET.SubElement(link2, "geom", type="mesh", mesh="capsule")
    else:
        ET.SubElement(link2, "geom", type="capsule", fromto="0 0 0 0.5 0 0", size="0.05")
    ET.SubElement(link2, "joint", type="hinge", name="joint2", axis="0 1 0")
    return mjcf


@pytest.fixture(scope="session")
def chain_capsule_hinge_mesh(asset_tmp_path):
    return _build_chain_capsule_hinge(asset_tmp_path, enable_mesh=True)


@pytest.fixture(scope="session")
def chain_capsule_hinge_capsule(asset_tmp_path):
    return _build_chain_capsule_hinge(asset_tmp_path, enable_mesh=False)


@pytest.mark.parametrize("model_name", ["box_plan"])
@pytest.mark.parametrize(
    "gs_solver",
    [gs.constraint_solver.CG],  # FIXME: , gs.constraint_solver.Newton],
)
@pytest.mark.parametrize(
    "gs_integrator",
    [gs.integrator.implicitfast, gs.integrator.Euler],
)
@pytest.mark.parametrize("backend", [gs.cpu], indirect=True)
def test_box_plan_dynamics(gs_sim, mj_sim):
    (gs_robot,) = gs_sim.entities
    cube_pos = np.array([0.0, 0.0, 0.6])
    cube_quat = np.random.rand(4)
    cube_quat /= np.linalg.norm(cube_quat)
    qpos = np.concatenate((cube_pos, cube_quat))
    qvel = np.random.rand(gs_robot.n_dofs) * 0.2
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, qpos, qvel, num_steps=150)


@pytest.mark.parametrize("model_name", ["box_box"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu], indirect=True)
def test_box_box_dynamics(gs_sim):
    for _ in range(50):
        cube1_pos = np.array([0.0, 0.0, 0.2])
        cube1_quat = np.array([1.0, 0.0, 0.0, 0.0])
        cube2_pos = np.array([0.0, 0.0, 0.6 + 0.4 * np.random.rand()])
        cube2_quat = gs.utils.geom.xyz_to_quat(np.array([0.0, 0.0, 2 * np.pi * np.random.rand()]), degrees=False)
        init_simulators(gs_sim, qpos=np.concatenate((cube1_pos, cube1_quat, cube2_pos, cube2_quat)))
        for i in range(50):
            gs_sim.scene.step()

            gs_n_contacts = gs_sim.rigid_solver.collider.n_contacts.to_numpy()[0]
            if i > 20:
                gs_n_constraints = gs_sim.rigid_solver.constraint_solver.n_constraints.to_numpy()[0]

                gs_contact_pos = gs_sim.rigid_solver.collider.contact_data.pos.to_numpy()[:gs_n_contacts, 0]
                # for pos in gs_contact_pos:
                #     gs_sim.scene.draw_debug_sphere(pos, radius=0.01, color=(1, 0, 0, 1))

                gs_contact_normal = gs_sim.rigid_solver.collider.contact_data.normal.to_numpy()[:gs_n_contacts, 0]
                gs_penetration = gs_sim.rigid_solver.collider.contact_data.penetration.to_numpy()[:gs_n_contacts, 0]

                # gs_sim.scene.viewer.run()
                breakpoint()

            if gs_sim.scene.visualizer:
                gs_sim.scene.visualizer.update()
        breakpoint()


@pytest.mark.parametrize("model_name", ["chain_capsule_hinge_mesh"])  # FIXME: , "chain_capsule_hinge_capsule"])
@pytest.mark.parametrize(
    "gs_solver",
    [gs.constraint_solver.CG],  # FIXME: , gs.constraint_solver.Newton],
)
@pytest.mark.parametrize(
    "gs_integrator",
    [gs.integrator.implicitfast, gs.integrator.Euler],
)
@pytest.mark.parametrize("backend", [gs.cpu], indirect=True)
def test_simple_kinematic_chain(gs_sim, mj_sim):
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, num_steps=200)


@pytest.mark.parametrize("xml_path", ["xml/walker.xml"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu], indirect=True)
def test_walker(gs_sim, mj_sim):
    (gs_robot,) = gs_sim.entities
    qpos = np.zeros((gs_robot.n_qs,))
    qpos[2] += 0.5
    qvel = np.random.rand(gs_robot.n_dofs) * 0.2
    # Cannot simulate any longer because collision detection is very sensitive
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, qpos, qvel, num_steps=90)


@pytest.mark.parametrize("xml_path", ["xml/franka_emika_panda/panda.xml"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu], indirect=True)
def test_robot_kinematics(gs_sim, mj_sim):
    # Disable all constraints and actuation
    mj_sim.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    mj_sim.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
    gs_sim.rigid_solver.dofs_state.ctrl_mode.fill(gs.CTRL_MODE.FORCE)
    gs_sim.rigid_solver._enable_collision = False
    gs_sim.rigid_solver._enable_joint_limit = False

    check_mujoco_model_consistency(gs_sim, mj_sim)

    (gs_robot,) = gs_sim.entities
    dof_bounds = gs_sim.rigid_solver.dofs_info.limit.to_numpy()
    for _ in range(100):
        qpos = dof_bounds[:, 0] + dof_bounds[:, 1] * np.random.rand(gs_robot.n_qs)
        init_simulators(gs_sim, mj_sim, qpos)


def test_nonconvex_collision(show_viewer):
    scene = gs.Scene(show_viewer=show_viewer, show_FPS=False)
    tank = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/tank.obj",
            scale=5.0,
            fixed=True,
            euler=(90, 0, 0),
        ),
    )
    ball = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.05,
            pos=(0.0, 0.0, 0.8),
        ),
        surface=gs.surfaces.Default(
            color=(0.5, 0.7, 0.9, 1.0),
        ),
    )
    scene.build()
    ball.set_dofs_velocity(np.random.rand(ball.n_dofs) * 0.8)
    for i in range(1500):
        scene.step()
    qvel = scene.sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]
    np.testing.assert_allclose(qvel, 0, atol=0.1)


def test_stacked_boxes(show_viewer):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            max_collision_pairs=250,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(10, 10, 10),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    for n in range(5**3):
        i, j, k = int(n / 25), int(n / 5) % 5, n % 5
        scene.add_entity(
            gs.morphs.Box(
                pos=(i * 1.01, j * 1.01, k * 1.01 + 0.5),
                size=(1.0, 1.0, 1.0),
            ),
            surface=gs.surfaces.Default(
                color=(*np.random.rand(3), 0.7),
            ),
        )

    scene.build()
    for i in range(150):
        scene.step()

    for n, entity in enumerate(scene.entities[1:]):
        i, j, k = int(n / 25), int(n / 5) % 5, n % 5
        qvel = entity.get_dofs_velocity()
        np.testing.assert_allclose(qvel, 0, atol=0.01)
    for n, entity in enumerate(scene.entities[1:]):
        i, j, k = int(n / 25), int(n / 5) % 5, n % 5
        qpos = entity.get_dofs_position()
        qpos0 = np.array((i * 1.01, j * 1.01, k * 1.01 + 0.5))
        np.testing.assert_allclose(qpos[:3], qpos0, atol=0.05)
        np.testing.assert_allclose(qpos[3:], 0, atol=0.02)
