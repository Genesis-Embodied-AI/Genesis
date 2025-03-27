import pytest
import xml.etree.ElementTree as ET

import trimesh
import torch
import numpy as np

import mujoco
import genesis as gs

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
def mimic_hinges():
    mjcf = ET.Element("mujoco", model="mimic_hinges")
    ET.SubElement(mjcf, "compiler", angle="degree")
    ET.SubElement(mjcf, "option", timestep="0.01")
    worldbody = ET.SubElement(mjcf, "worldbody")
    parent = ET.SubElement(worldbody, "body", name="parent", pos="0 0 1.0")
    child1 = ET.SubElement(parent, "body", name="child1", pos="0.5 0 0")
    ET.SubElement(child1, "geom", type="capsule", size="0.05 0.2", rgba="0.9 0.1 0.1 1")
    ET.SubElement(child1, "joint", type="hinge", name="joint1", axis="0 1 0", range="-45 45")
    child2 = ET.SubElement(parent, "body", name="child2", pos="0 0.5 0")
    ET.SubElement(child2, "geom", type="capsule", size="0.05 0.2", rgba="0.1 0.1 0.9 1")
    ET.SubElement(child2, "joint", type="hinge", name="joint2", axis="0 1 0", range="-45 45")
    equality = ET.SubElement(mjcf, "equality")
    ET.SubElement(equality, "joint", name="joint_equality", joint1="joint1", joint2="joint2")
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
def two_aligned_hinges():
    mjcf = ET.Element("mujoco", model="two_aligned_hinges")
    ET.SubElement(mjcf, "option", timestep="0.05")
    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "geom", contype="1", conaffinity="1", condim="3")
    worldbody = ET.SubElement(mjcf, "worldbody")
    link0 = ET.SubElement(worldbody, "body", name="body0")
    ET.SubElement(link0, "geom", type="capsule", fromto="0 0 0 0.5 0 0", size="0.05")
    ET.SubElement(link0, "joint", type="hinge", name="joint0", axis="0 0 1")
    link1 = ET.SubElement(link0, "body", name="body1", pos="0.5 0 0")
    ET.SubElement(link1, "geom", type="capsule", fromto="0 0 0 0.5 0 0", size="0.05")
    ET.SubElement(link1, "joint", type="hinge", name="joint1", axis="0 0 1")
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
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu], indirect=True)
def test_box_plan_dynamics(gs_sim, mj_sim):
    cube_pos = np.array([0.0, 0.0, 0.6])
    cube_quat = np.random.rand(4)
    cube_quat /= np.linalg.norm(cube_quat)
    qpos = np.concatenate((cube_pos, cube_quat))
    qvel = np.random.rand(6) * 0.2
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, qpos, qvel, num_steps=150)


@pytest.mark.parametrize("model_name", ["two_aligned_hinges"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
def test_link_velocity(gs_sim):
    # Check the velocity for a few "easy" special cases
    init_simulators(gs_sim, qvel=np.array([0.0, 1.0]))
    np.testing.assert_allclose(gs_sim.rigid_solver.links_state.vel.to_numpy(), 0, atol=1e-9)

    init_simulators(gs_sim, qvel=np.array([1.0, 0.0]))
    cvel_0, cvel_1 = gs_sim.rigid_solver.links_state.vel.to_numpy()[:, 0]
    np.testing.assert_allclose(cvel_0, np.array([0.0, 0.5, 0.0]), atol=1e-9)
    np.testing.assert_allclose(cvel_1, np.array([0.0, 0.5, 0.0]), atol=1e-9)

    init_simulators(gs_sim, qpos=np.array([0.0, np.pi / 2.0]), qvel=np.array([0.0, 1.2]))
    COM = gs_sim.rigid_solver.links_state[0, 0].COM
    np.testing.assert_allclose(COM, np.array([0.375, 0.125, 0.0]), atol=1e-9)
    xanchor = gs_sim.rigid_solver.joints_state[1, 0].xanchor
    np.testing.assert_allclose(xanchor, np.array([0.5, 0.0, 0.0]), atol=1e-9)
    cvel_0, cvel_1 = gs_sim.rigid_solver.links_state.vel.to_numpy()[:, 0]
    np.testing.assert_allclose(cvel_0, 0, atol=1e-9)
    np.testing.assert_allclose(cvel_1, np.array([-1.2 * (0.125 - 0.0), 1.2 * (0.375 - 0.5), 0.0]), atol=1e-9)

    # Check that the velocity is valid for a random configuration
    init_simulators(gs_sim, qpos=np.array([-0.7, 0.2]), qvel=np.array([3.0, 13.0]))
    xanchor = gs_sim.rigid_solver.joints_state[1, 0].xanchor
    theta_0, theta_1 = gs_sim.rigid_solver.qpos.to_numpy()[:, 0]
    np.testing.assert_allclose(xanchor[0], 0.5 * np.cos(theta_0), atol=1e-9)
    np.testing.assert_allclose(xanchor[1], 0.5 * np.sin(theta_0), atol=1e-9)
    COM = gs_sim.rigid_solver.links_state[0, 0].COM
    COM_0 = np.array([0.25 * np.cos(theta_0), 0.25 * np.sin(theta_0), 0.0])
    COM_1 = np.array(
        [
            0.5 * np.cos(theta_0) + 0.25 * np.cos(theta_0 + theta_1),
            0.5 * np.sin(theta_0) + 0.25 * np.sin(theta_0 + theta_1),
            0.0,
        ]
    )
    np.testing.assert_allclose(COM, 0.5 * (COM_0 + COM_1), atol=1e-9)

    cvel_0, cvel_1 = gs_sim.rigid_solver.links_state.vel.to_numpy()[:, 0]
    omega_0, omega_1 = gs_sim.rigid_solver.links_state.ang.to_numpy()[:, 0, 2]
    np.testing.assert_allclose(omega_0, 3.0, atol=1e-9)
    np.testing.assert_allclose(omega_1 - omega_0, 13.0, atol=1e-9)
    cvel_0_ = omega_0 * np.array([-COM[1], COM[0], 0.0])
    np.testing.assert_allclose(cvel_0, cvel_0_, atol=1e-9)
    cvel_1_ = cvel_0 + (omega_1 - omega_0) * np.array([xanchor[1] - COM[1], COM[0] - xanchor[0], 0.0])
    np.testing.assert_allclose(cvel_1, cvel_1_, atol=1e-9)

    xpos_0, xpos_1 = gs_sim.rigid_solver.links_state.pos.to_numpy()[:, 0]
    np.testing.assert_allclose(xpos_0, 0.0, atol=1e-9)
    np.testing.assert_allclose(xpos_1, xanchor, atol=1e-9)
    xvel_0, xvel_1 = gs_sim.rigid_solver.get_links_vel()
    np.testing.assert_allclose(xvel_0, 0.0, atol=1e-9)
    xvel_1_ = omega_0 * np.array([-xpos_1[1], xpos_1[0], 0.0])
    np.testing.assert_allclose(xvel_1, xvel_1_, atol=1e-9)
    civel_0, civel_1 = gs_sim.rigid_solver.get_links_vel(ref="link_com")
    civel_0_ = omega_0 * np.array([-COM_0[1], COM_0[0], 0.0])
    np.testing.assert_allclose(civel_0, civel_0_, atol=1e-9)
    civel_1_ = omega_0 * np.array([-COM_1[1], COM_1[0], 0.0]) + (omega_1 - omega_0) * np.array(
        [xanchor[1] - COM_1[1], COM_1[0] - xanchor[0], 0.0]
    )
    np.testing.assert_allclose(civel_1, civel_1_, atol=1e-9)


@pytest.mark.parametrize("model_name", ["box_box"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu], indirect=True)
def test_box_box_dynamics(gs_sim):
    (gs_robot,) = gs_sim.entities
    for _ in range(20):
        cube1_pos = np.array([0.0, 0.0, 0.2])
        cube1_quat = np.array([1.0, 0.0, 0.0, 0.0])
        cube2_pos = np.array([0.0, 0.0, 0.65 + 0.1 * np.random.rand()])
        cube2_quat = gs.utils.geom.xyz_to_quat(
            np.array([*(0.15 * np.random.rand(2)), np.pi * np.random.rand()]), degrees=False
        )
        init_simulators(gs_sim, qpos=np.concatenate((cube1_pos, cube1_quat, cube2_pos, cube2_quat)))
        for i in range(100):
            gs_sim.scene.step()

        qvel = gs_robot.get_dofs_velocity().cpu()
        np.testing.assert_allclose(qvel, 0, atol=1e-2)
        qpos = gs_robot.get_dofs_position().cpu()
        np.testing.assert_allclose(qpos[8], 0.6, atol=1e-3)


@pytest.mark.parametrize("box_box_detection", [False, True])
@pytest.mark.parametrize("dynamics", [False, True])
@pytest.mark.parametrize("backend", [gs.cpu], indirect=True)  # TODO: Add GPU once tests run in parrallel
def test_many_boxes_dynamics(box_box_detection, dynamics, show_viewer):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            box_box_detection=box_box_detection,
            max_collision_pairs=1000,
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

    if dynamics:
        for entity in scene.entities[1:]:
            entity.set_dofs_velocity(4.0 * np.random.rand(6))
    for i in range(800 if dynamics else 300):
        scene.step()

    for n, entity in enumerate(scene.entities[1:]):
        i, j, k = int(n / 25), int(n / 5) % 5, n % 5
        qvel = entity.get_dofs_velocity().cpu()
        np.testing.assert_allclose(qvel, 0, atol=0.1 if dynamics else 0.05)
    for n, entity in enumerate(scene.entities[1:]):
        i, j, k = int(n / 25), int(n / 5) % 5, n % 5
        qpos = entity.get_dofs_position().cpu()
        if dynamics:
            assert qpos[:2].norm() < 20.0
            assert qpos[2] < 5.0
        else:
            qpos0 = np.array((i * 1.01, j * 1.01, k * 1.01 + 0.5))
            np.testing.assert_allclose(qpos[:3], qpos0, atol=0.05)
            np.testing.assert_allclose(qpos[3:], 0, atol=0.03)

    if show_viewer:
        scene.viewer.stop()


@pytest.mark.adjacent_collision(True)
@pytest.mark.parametrize("model_name", ["chain_capsule_hinge_mesh"])  # FIXME: , "chain_capsule_hinge_capsule"])
@pytest.mark.parametrize(
    "gs_solver",
    [gs.constraint_solver.CG],  # FIXME: , gs.constraint_solver.Newton],
)
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
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
    gs_sim.rigid_solver._disable_constraint = True

    check_mujoco_model_consistency(gs_sim, mj_sim)

    (gs_robot,) = gs_sim.entities
    dof_bounds = gs_sim.rigid_solver.dofs_info.limit.to_numpy()
    for _ in range(100):
        qpos = dof_bounds[:, 0] + dof_bounds[:, 1] * np.random.rand(gs_robot.n_qs)
        init_simulators(gs_sim, mj_sim, qpos)
        check_mujoco_data_consistency(gs_sim, mj_sim, atol=(1e-9 if gs.np_float == np.float64 else 5e-5))


@pytest.mark.dof_damping(True)
@pytest.mark.parametrize("xml_path", ["xml/humanoid.xml"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu], indirect=True)
def test_stickman(gs_sim, mj_sim):
    # Make sure that "static" model information are matching
    check_mujoco_model_consistency(gs_sim, mj_sim)

    # Initialize the simulation
    init_simulators(gs_sim)

    # Run the simulation for a few steps
    for i in range(6000):
        gs_sim.scene.step()

    (gs_robot,) = gs_sim.entities
    qvel = gs_robot.get_dofs_velocity().cpu()
    np.testing.assert_allclose(qvel, 0, atol=0.1)
    qpos = gs_robot.get_dofs_position().cpu()
    assert np.linalg.norm(qpos[:2]) < 1.3
    body_z = gs_sim.rigid_solver.links_state.pos.to_numpy()[:-1, 0, 2]
    np.testing.assert_array_less(0, body_z)


@pytest.mark.parametrize("backend", [gs.gpu, gs.cpu], indirect=True)
def test_inverse_kinematics(show_viewer):
    # create and build the scene
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            box_box_detection=True,
        ),
        show_viewer=show_viewer,
    )
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.65, 0.0, 0.02),
        ),
        surface=gs.surfaces.Plastic(color=(1, 0, 0)),
    )
    cube_2 = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.4, 0.2, 0.02),
        ),
        surface=gs.surfaces.Plastic(color=(0, 1, 0)),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        vis_mode="collision",
    )
    scene.build()

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    end_effector = franka.get_link("hand")

    # set control gains
    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )

    # move to pre-grasp pose
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.25]),
        quat=np.array([0, 1, 0, 0]),
    )
    # gripper open pos
    qpos[-2:] = 0.04
    path = franka.plan_path(
        qpos_goal=qpos,
        num_waypoints=100,  # 1s duration
    )
    # execute the planned path
    for waypoint in path:
        franka.control_dofs_position(waypoint)
        franka.control_dofs_force(np.array([0.5, 0.5]), fingers_dof)
        scene.step()

    # allow robot to reach the last waypoint
    for i in range(100):
        scene.step()

    # reach
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.130]),
        quat=np.array([0, 1, 0, 0]),
    )
    franka.control_dofs_position(qpos[:-2], motors_dof)
    for i in range(50):
        scene.step()

    # grasp
    franka.control_dofs_position(qpos[:-2], motors_dof)
    franka.control_dofs_force(np.array([-0.5, -0.5]), fingers_dof)
    for i in range(50):
        scene.step()

    # lift
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.28]),
        quat=np.array([0, 1, 0, 0]),
    )
    franka.control_dofs_position(qpos[:-2], motors_dof)
    for i in range(50):
        scene.step()

    # reach
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.4, 0.2, 0.18]),
        quat=np.array([0, 1, 0, 0]),
    )
    franka.control_dofs_position(qpos[:-2], motors_dof)
    for i in range(100):
        scene.step()

    # release
    franka.control_dofs_position(np.array([0.4, 0.4]), fingers_dof)
    for i in range(400):
        scene.step()

    qvel = cube.get_dofs_velocity().cpu()
    np.testing.assert_allclose(qvel, 0, atol=0.05)
    qpos = cube.get_dofs_position().cpu()
    np.testing.assert_allclose(qpos[2], 0.06, atol=1e-3)

    if show_viewer:
        scene.viewer.stop()


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

    if show_viewer:
        scene.viewer.stop()


@pytest.mark.parametrize("backend", [gs.gpu, gs.cpu], indirect=True)
def test_terrain_generation(show_viewer):
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(-5.0, -5.0, 10.0),
            camera_lookat=(5.0, 5.0, 0.0),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
        ),
    )
    terrain = scene.add_entity(
        morph=gs.morphs.Terrain(
            n_subterrains=(2, 2),
            subterrain_size=(6.0, 6.0),
            horizontal_scale=0.25,
            vertical_scale=0.005,
            subterrain_types=[
                ["flat_terrain", "random_uniform_terrain"],
                ["pyramid_sloped_terrain", "discrete_obstacles_terrain"],
            ],
        ),
    )
    ball = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(1.0, 1.0, 1.0),
            radius=0.1,
        ),
    )
    scene.build(n_envs=225)

    ball.set_pos(torch.cartesian_prod(*(torch.linspace(1.0, 10.0, 15),) * 2, torch.tensor((0.6,))))
    for _ in range(400):
        scene.step()

    # Make sure that at least one ball is as minimum height, and some are signficantly higher
    height_field = terrain.geoms[0].metadata["height_field"]
    height_field_min = terrain.terrain_scale[1] * height_field.min()
    height_field_max = terrain.terrain_scale[1] * height_field.max()
    height_balls = ball.get_pos().cpu()[:, 2]
    height_balls_min = height_balls.min() - 0.1
    height_balls_max = height_balls.max() - 0.1
    np.testing.assert_allclose(height_balls_min, height_field_min, atol=1e-3)
    assert height_balls_max - height_balls_min > 0.5 * (height_field_max - height_field_min)


@pytest.mark.parametrize("model_name", ["mimic_hinges"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu], indirect=True)
def test_equality_joint(gs_sim, mj_sim):
    # there is an equality constraint
    assert gs_sim.rigid_solver.n_equalities == 1

    qpos = np.array((0.0, -1.0))
    qvel = np.array((1, -0.3))
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, qpos, qvel, num_steps=300, atol=1e-8)

    # check if the two joints are equal
    gs_qpos = gs_sim.rigid_solver.qpos.to_numpy()[:, 0]
    np.testing.assert_allclose(gs_qpos[0], gs_qpos[1], atol=1e-9)


@pytest.mark.parametrize("backend", [gs.cpu], indirect=True)
def test_urdf_mimic_panda(show_viewer):
    # create and build the scene
    scene = gs.Scene(
        show_viewer=show_viewer,
    )

    hand = scene.add_entity(
        gs.morphs.URDF(file="urdf/panda_bullet/hand.urdf"),
    )
    scene.build()

    rigid = scene.sim.rigid_solver
    assert rigid.n_equalities == 1

    qvel = rigid.dofs_state.vel.to_numpy()
    qvel[-1] = 1
    rigid.dofs_state.vel.from_numpy(qvel)

    for i in range(200):
        scene.step()

    gs_qpos = rigid.qpos.to_numpy()[:, 0]
    np.testing.assert_allclose(gs_qpos[-1], gs_qpos[-2], atol=1e-9)

    if show_viewer:
        scene.viewer.stop()


@pytest.mark.parametrize("n_envs", [0, 3])
def test_data_accessor(n_envs):
    # TODO: Check that the setters are doing something and not just no-ops
    # TODO: Compare the getter output with their corresponding field value if applicable

    # create and build the scene
    scene = gs.Scene(
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    gs_robot = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    scene.build(n_envs=n_envs)
    gs_sim = scene.sim
    gs_solver = gs_sim.rigid_solver

    # Initialize the simulation
    dof_bounds = gs_sim.rigid_solver.dofs_info.limit.to_numpy()
    qpos_all = []
    for i in range(max(n_envs, 1)):
        qpos = dof_bounds[:, 0] + dof_bounds[:, 1] * np.random.rand(gs_robot.n_qs)
        if n_envs:
            gs_robot.set_qpos(qpos[None], envs_idx=[i])
            _qpos = gs_robot.get_qpos(envs_idx=[i]).squeeze(0).cpu()
        else:
            gs_robot.set_qpos(qpos)
            _qpos = gs_robot.get_qpos().squeeze(0).cpu()
        np.testing.assert_allclose(qpos, _qpos, atol=1e-9)
        if n_envs:
            _qpos = gs_robot.get_qpos()[i].cpu()
            np.testing.assert_allclose(qpos, _qpos, atol=1e-9)

    # Simulate for a while, until they collide with something
    for _ in range(400):
        gs_sim.step()
        gs_n_contacts = gs_sim.rigid_solver.collider.n_contacts.to_numpy()
        if (gs_n_contacts > 0).all():
            break
    else:
        assert False
    gs_sim.rigid_solver._kernel_forward_dynamics()
    gs_sim.rigid_solver._func_constraint_force()

    # Make sure that all the robots ends up in the different state
    qposs = gs_robot.get_qpos().cpu()
    for i in range(n_envs - 1):
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_allclose(qposs[i], qposs[i + 1], atol=1e-9)

    # Check attribute getters / setters.
    # First, without any any row or column masking:
    # * Call 'Get' -> Call 'Set' with 'Get' output -> Call 'Get'
    # Then, for any possible combinations of row and column masking:
    # * Call 'Get' -> Call 'Set' with 'Get' output -> Call 'Get'
    # * Compare first 'Get' output with last 'Get' output
    # * Compare last 'Get' output with corresponding slice of non-masking 'Get' output
    def get_all_supported_masks(i):
        return (
            i,
            [i],
            slice(i, i + 1),
            range(i, i + 1),
            np.array([i], dtype=np.int32),
            torch.tensor([i], dtype=torch.int64),
            torch.tensor([i], dtype=gs.tc_int, device=gs.device),
        )

    def must_cast(value):
        return not (isinstance(value, torch.Tensor) and value.dtype == gs.tc_int and value.device == gs.device)

    for arg1_max, arg2_max, getter, setter in (
        (gs_solver.n_links, n_envs, gs_solver.get_links_pos, None),
        (gs_solver.n_links, n_envs, gs_solver.get_links_quat, None),
        (gs_solver.n_links, n_envs, gs_solver.get_links_vel, None),
        (gs_solver.n_links, n_envs, gs_solver.get_links_ang, None),
        (gs_solver.n_links, n_envs, gs_solver.get_links_acc, None),
        (gs_solver.n_links, n_envs, gs_solver.get_links_COM, None),
        (gs_solver.n_links, n_envs, gs_solver.get_links_mass_shift, gs_solver.set_links_mass_shift),
        (gs_solver.n_links, n_envs, gs_solver.get_links_COM_shift, gs_solver.set_links_COM_shift),
        (gs_solver.n_links, -1, gs_solver.get_links_inertial_mass, gs_solver.set_links_inertial_mass),
        (gs_solver.n_links, -1, gs_solver.get_links_invweight, gs_solver.set_links_invweight),
        (gs_solver.n_dofs, n_envs, gs_solver.get_dofs_control_force, gs_solver.control_dofs_force),
        (gs_solver.n_dofs, n_envs, gs_solver.get_dofs_force, None),
        (gs_solver.n_dofs, n_envs, gs_solver.get_dofs_velocity, gs_solver.set_dofs_velocity),
        (gs_solver.n_dofs, n_envs, gs_solver.get_dofs_position, gs_solver.set_dofs_position),
        (gs_solver.n_dofs, -1, gs_solver.get_dofs_force_range, None),
        (gs_solver.n_dofs, -1, gs_solver.get_dofs_limit, None),
        (gs_solver.n_dofs, -1, gs_solver.get_dofs_stiffness, None),
        (gs_solver.n_dofs, -1, gs_solver.get_dofs_invweight, None),
        (gs_solver.n_dofs, -1, gs_solver.get_dofs_armature, None),
        (gs_solver.n_dofs, -1, gs_solver.get_dofs_damping, None),
        (gs_solver.n_dofs, -1, gs_solver.get_dofs_kp, gs_solver.set_dofs_kp),
        (gs_solver.n_dofs, -1, gs_solver.get_dofs_kv, gs_solver.set_dofs_kv),
        (gs_solver.n_geoms, n_envs, gs_solver.get_geoms_pos, None),
        (gs_solver.n_geoms, -1, gs_solver.get_geoms_friction, gs_solver.set_geoms_friction),
        (gs_solver.n_qs, n_envs, gs_solver.get_qpos, gs_solver.set_qpos),
        (gs_robot.n_links, n_envs, gs_robot.get_links_pos, None),
        (gs_robot.n_links, n_envs, gs_robot.get_links_quat, None),
        (gs_robot.n_links, n_envs, gs_robot.get_links_vel, None),
        (gs_robot.n_links, n_envs, gs_robot.get_links_ang, None),
        (gs_robot.n_links, n_envs, gs_robot.get_links_acc, None),
        (gs_robot.n_links, -1, gs_robot.get_links_inertial_mass, gs_robot.set_links_inertial_mass),
        (gs_robot.n_links, -1, gs_robot.get_links_invweight, gs_robot.set_links_invweight),
        (gs_robot.n_dofs, n_envs, gs_robot.get_dofs_control_force, None),
        (gs_robot.n_dofs, n_envs, gs_robot.get_dofs_force, None),
        (gs_robot.n_dofs, n_envs, gs_robot.get_dofs_velocity, gs_robot.set_dofs_velocity),
        (gs_robot.n_dofs, n_envs, gs_robot.get_dofs_position, gs_robot.set_dofs_position),
        (gs_robot.n_dofs, -1, gs_robot.get_dofs_force_range, None),
        (gs_robot.n_dofs, -1, gs_robot.get_dofs_limit, None),
        (gs_robot.n_dofs, -1, gs_robot.get_dofs_stiffness, None),
        (gs_robot.n_dofs, -1, gs_robot.get_dofs_invweight, None),
        (gs_robot.n_dofs, -1, gs_robot.get_dofs_armature, None),
        (gs_robot.n_dofs, -1, gs_robot.get_dofs_damping, None),
        (gs_robot.n_dofs, -1, gs_robot.get_dofs_kp, gs_robot.set_dofs_kp),
        (gs_robot.n_dofs, -1, gs_robot.get_dofs_kv, gs_robot.set_dofs_kv),
        (gs_robot.n_qs, n_envs, gs_robot.get_qpos, gs_robot.set_qpos),
        (-1, n_envs, gs_robot.get_links_net_contact_force, None),
        (-1, n_envs, gs_robot.get_pos, gs_robot.set_pos),
        (-1, n_envs, gs_robot.get_quat, gs_robot.set_quat),
    ):
        # Check getter and setter without row or column masking
        datas = getter()
        if setter is not None:
            setter(datas)
        datas = datas.cpu() if isinstance(datas, torch.Tensor) else [val.cpu() for val in datas]
        if arg1_max > 0:
            datas_ = getter(range(arg1_max))
            datas_ = datas_.cpu() if isinstance(datas_, torch.Tensor) else [val.cpu() for val in datas_]
            np.testing.assert_allclose(datas_, datas, atol=1e-9)

        # Check getter and setter for all possible combinations of row and column masking
        for i in range(arg1_max) if arg1_max > 0 else (None,):
            for arg1 in get_all_supported_masks(i) if arg1_max > 0 else (None,):
                for j in range(max(arg2_max, 1)) if arg2_max >= 0 else (None,):
                    for arg2 in get_all_supported_masks(j) if arg2_max > 0 else (None,):
                        if arg1 is None:
                            unsafe = not must_cast(arg2)
                            data = getter(arg2, unsafe=unsafe)
                            if setter is not None:
                                setter(data, arg2, unsafe=unsafe)
                            if n_envs:
                                if isinstance(datas, torch.Tensor):
                                    data_ = datas[[j]]
                                else:
                                    data_ = [val[[j]] for val in datas]
                            else:
                                data_ = datas
                        elif arg2 is None:
                            unsafe = not must_cast(arg1)
                            data = getter(arg1, unsafe=unsafe)
                            if setter is not None:
                                setter(data, arg1, unsafe=unsafe)
                            if isinstance(datas, torch.Tensor):
                                data_ = datas[[i]]
                            else:
                                data_ = [val[[i]] for val in datas]
                        else:
                            unsafe = not any(map(must_cast, (arg1, arg2)))
                            data = getter(arg1, arg2, unsafe=unsafe)
                            if setter is not None:
                                setter(data, arg1, arg2, unsafe=unsafe)
                            if isinstance(datas, torch.Tensor):
                                data_ = datas[[j], :][:, [i]]
                            else:
                                data_ = [val[[j], :][:, [i]] for val in datas]
                        data = data.cpu() if isinstance(data, torch.Tensor) else [val.cpu() for val in data]
                        np.testing.assert_allclose(data_, data, atol=1e-9)

    for dofs_idx in (*get_all_supported_masks(0), None):
        for envs_idx in (*(get_all_supported_masks(0) if n_envs > 0 else ()), None):
            unsafe = not any(map(must_cast, (dofs_idx, envs_idx)))
            dofs_pos = gs_solver.get_dofs_position(dofs_idx, envs_idx)
            dofs_vel = gs_solver.get_dofs_velocity(dofs_idx, envs_idx)
            gs_sim.rigid_solver.control_dofs_position(dofs_pos, dofs_idx, envs_idx)
            gs_sim.rigid_solver.control_dofs_velocity(dofs_vel, dofs_idx, envs_idx)
