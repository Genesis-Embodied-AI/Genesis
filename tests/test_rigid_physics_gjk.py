import math
import os
import sys
import xml.etree.ElementTree as ET

import pytest
import trimesh
import torch
import numpy as np
from huggingface_hub import snapshot_download

import mujoco
import genesis as gs

from .utils import (
    assert_allclose,
    build_mujoco_sim,
    build_genesis_sim,
    init_simulators,
    check_mujoco_model_consistency,
    check_mujoco_data_consistency,
    simulate_and_check_mujoco_consistency,
)

USE_GJK = True

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
    ET.SubElement(mjcf, "option", timestep="0.01")
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


@pytest.fixture
def collision_edge_cases(asset_tmp_path, mode):
    assets = {}
    for i, box_size in enumerate(((0.8, 0.8, 0.04), (0.04, 0.04, 0.005))):
        tmesh = trimesh.creation.box(extents=np.array(box_size) * 2)
        mesh_path = str(asset_tmp_path / f"box{i}.obj")
        tmesh.export(mesh_path, file_type="obj")
        assets[f"box{i}"] = mesh_path

    mjcf = ET.Element("mujoco", model="one_box")
    ET.SubElement(mjcf, "option", timestep="0.005")
    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "geom", contype="1", conaffinity="1", condim="3", friction="1. 0.5 0.5")

    asset = ET.SubElement(mjcf, "asset")
    for name, mesh_path in assets.items():
        ET.SubElement(asset, "mesh", name=name, refpos="0 0 0", refquat="1 0 0 0", file=mesh_path)

    worldbody = ET.SubElement(mjcf, "worldbody")

    if mode == 0:
        ET.SubElement(worldbody, "geom", type="box", size="0.8 0.8 0.04", pos="0. 0. 0.", rgba="0 1 0 0.4")
        box1_body = ET.SubElement(worldbody, "body", name="box1", pos="0.0 0.0 0.7")
        ET.SubElement(box1_body, "geom", type="box", size="0.04 0.04 0.005", pos="-0.758 -0.758 0.", rgba="0 0 1 0.4")
    elif mode == 1:
        ET.SubElement(worldbody, "geom", type="box", size="0.8 0.8 0.04", pos="0. 0. 0.", rgba="0 1 0 0.4")
        box1_body = ET.SubElement(worldbody, "body", name="box1", pos="-0.758 -0.758 0.7")
        ET.SubElement(box1_body, "geom", type="box", size="0.04 0.04 0.005", pos="0. 0. 0.", rgba="0 0 1 0.4")
    elif mode == 2:
        ET.SubElement(worldbody, "geom", type="box", size="0.8 0.8 0.04", pos="0. 0. 0.", rgba="0 1 0 0.4")
        box1_body = ET.SubElement(worldbody, "body", name="box1", pos="-0.758 -0.758 1.1")
        ET.SubElement(box1_body, "geom", type="box", size="0.04 0.04 0.005", pos="0. 0. 0.", rgba="0 0 1 0.4")
    elif mode == 3:
        ET.SubElement(worldbody, "geom", type="box", size="0.8 0.8 0.04", pos="0. 0. 0.", rgba="0 1 0 0.4")
        box1_body = ET.SubElement(worldbody, "body", name="box1", pos="0.0 0.0 0.7")
        ET.SubElement(box1_body, "geom", type="mesh", mesh="box1", pos="-0.758 -0.758 0.", rgba="0 0 1 0.4")
    elif mode == 4:
        ET.SubElement(worldbody, "geom", type="mesh", mesh="box0", pos="0. 0. 0.", rgba="0 1 0 0.4")
        box1_body = ET.SubElement(worldbody, "body", name="box1", pos="0.0 0.0 0.7")
        ET.SubElement(box1_body, "geom", type="mesh", mesh="box1", pos="-0.758 -0.758 0.", rgba="0 0 1 0.4")
    elif mode == 5:
        ET.SubElement(worldbody, "geom", type="mesh", mesh="box0", pos="0. 0. 0.", rgba="0 1 0 0.4")
        box1_body = ET.SubElement(worldbody, "body", name="box1", pos="-0.758 -0.758 0.7")
        ET.SubElement(box1_body, "geom", type="mesh", mesh="box1", pos="0. 0. 0.", rgba="0 0 1 0.4")
    elif mode == 6:
        ET.SubElement(worldbody, "geom", type="mesh", mesh="box0", pos="0. 0. 0.", rgba="0 1 0 0.4")
        box1_body = ET.SubElement(worldbody, "body", name="box1", pos="-0.758 -0.758 1.1")
        ET.SubElement(box1_body, "geom", type="mesh", mesh="box1", pos="0. 0. 0.", rgba="0 0 1 0.4")
    elif mode == 7:
        ET.SubElement(worldbody, "geom", type="mesh", mesh="box1", pos=" 0.758  0.758 0.", rgba="0 1 0 0.4")
        ET.SubElement(worldbody, "geom", type="mesh", mesh="box1", pos="-0.758 -0.758 0.", rgba="0 1 0 0.4")
        ET.SubElement(worldbody, "geom", type="mesh", mesh="box1", pos=" 0.758 -0.758 0.", rgba="0 1 0 0.4")
        ET.SubElement(worldbody, "geom", type="mesh", mesh="box1", pos="-0.758  0.758 0.", rgba="0 1 0 0.4")
        box1_body = ET.SubElement(worldbody, "body", name="box1", pos="0. 0. 0.7")
        ET.SubElement(box1_body, "geom", type="mesh", mesh="box0", pos="0. 0. 0.", rgba="0 0 1 0.4")
    elif mode == 8:
        ET.SubElement(worldbody, "geom", type="mesh", mesh="box1", pos=" 0.762  0.762 0.", rgba="0 1 0 0.4")
        ET.SubElement(worldbody, "geom", type="mesh", mesh="box1", pos="-0.762 -0.762 0.", rgba="0 1 0 0.4")
        ET.SubElement(worldbody, "geom", type="mesh", mesh="box1", pos=" 0.762 -0.762 0.", rgba="0 1 0 0.4")
        ET.SubElement(worldbody, "geom", type="mesh", mesh="box1", pos="-0.762  0.762 0.", rgba="0 1 0 0.4")
        box1_body = ET.SubElement(worldbody, "body", name="box1", pos="0. 0. 0.7")
        ET.SubElement(box1_body, "geom", type="mesh", mesh="box0", pos="0. 0. 0.", rgba="0 0 1 0.4")
    else:
        raise ValueError("Invalid mode")

    ET.SubElement(box1_body, "joint", name="root", type="free")

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


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["box_plan"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_box_plane_dynamics(gs_sim, mj_sim, tol):
    cube_pos = np.array([0.0, 0.0, 0.6])
    cube_quat = np.random.rand(4)
    cube_quat /= np.linalg.norm(cube_quat)
    qpos = np.concatenate((cube_pos, cube_quat))
    qvel = np.random.rand(6) * 0.2
    
    if gs_sim.rigid_solver.is_active():
        gs_sim.rigid_solver.collider.use_gjk = USE_GJK
    if gs_sim.avatar_solver.is_active():
        gs_sim.avatar_solver.collider.use_gjk = USE_GJK
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, qpos, qvel, num_steps=150, tol=tol)


@pytest.mark.required
@pytest.mark.adjacent_collision(True)
@pytest.mark.parametrize("model_name", ["chain_capsule_hinge_mesh"])  # FIXME: , "chain_capsule_hinge_capsule"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_simple_kinematic_chain(gs_sim, mj_sim, tol):
    
    if gs_sim.rigid_solver.is_active():
        gs_sim.rigid_solver.collider.use_gjk = USE_GJK
    if gs_sim.avatar_solver.is_active():
        gs_sim.avatar_solver.collider.use_gjk = USE_GJK
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, num_steps=200, tol=tol)


# Disable Genesis multi-contact because it relies on discretized geometry unlike Mujoco
@pytest.mark.required
@pytest.mark.multi_contact(False)
@pytest.mark.parametrize("xml_path", ["xml/walker.xml"])
@pytest.mark.parametrize(
    "gs_solver",
    [
        gs.constraint_solver.CG,
        # gs.constraint_solver.Newton,  # FIXME: This test is not passing because collision detection is too sensitive
    ],
)
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_walker(gs_sim, mj_sim, tol):
    # Force numpy seed because this test is very sensitive to the initial condition
    np.random.seed(0)
    (gs_robot,) = gs_sim.entities
    qpos = np.zeros((gs_robot.n_qs,))
    qpos[2] += 0.5
    qvel = np.random.rand(gs_robot.n_dofs) * 0.2

    # Cannot simulate any longer because collision detection is very sensitive
    if gs_sim.rigid_solver.is_active():
        gs_sim.rigid_solver.collider.use_gjk = USE_GJK
    if gs_sim.avatar_solver.is_active():
        gs_sim.avatar_solver.collider.use_gjk = USE_GJK
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, qpos, qvel, num_steps=90, tol=tol)


@pytest.mark.parametrize("model_name", ["mimic_hinges"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_equality_joint(gs_sim, mj_sim, gs_solver):
    # there is an equality constraint
    assert gs_sim.rigid_solver.n_equalities == 1

    qpos = np.array((0.0, -1.0))
    qvel = np.array((1.0, -0.3))
    # Note that it is impossible to be more accurate than this because of the inherent stiffness of the problem.
    tol = 2e-8 if gs_solver == gs.constraint_solver.Newton else 1e-8
    if gs_sim.rigid_solver.is_active():
        gs_sim.rigid_solver.collider.use_gjk = USE_GJK
    if gs_sim.avatar_solver.is_active():
        gs_sim.avatar_solver.collider.use_gjk = USE_GJK
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, qpos, qvel, num_steps=300, tol=tol)

    # check if the two joints are equal
    gs_qpos = gs_sim.rigid_solver.qpos.to_numpy()[:, 0]
    assert_allclose(gs_qpos[0], gs_qpos[1], tol=tol)


@pytest.mark.parametrize("xml_path", ["xml/four_bar_linkage_weld.xml"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_equality_weld(gs_sim, mj_sim, gs_solver):
    # Must disable self-collision caused by closing the kinematic chain (adjacent link filtering is not enough)
    gs_sim.rigid_solver._enable_collision = False
    mj_sim.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

    assert gs_sim.rigid_solver.n_equalities == 1
    np.random.seed(0)
    qpos = np.random.rand(gs_sim.rigid_solver.n_qs) * 0.1

    # Note that it is impossible to be more accurate than this because of the inherent stiffness of the problem.
    # The pose difference between Mujoco and Genesis (resulting from using quaternion instead of rotation matrices to
    # apply transform internally) is about 1e-15. This is fine and not surprising as it is consistent with machine
    # precision. These rounding errors are then amplified by 1e8 when computing the forces resulting from the kinematic
    # constraints. The constraints could be made softer by changing its impede parameters.
    tol = 1e-7 if gs_solver == gs.constraint_solver.Newton else 2e-5
    if gs_sim.rigid_solver.is_active():
        gs_sim.rigid_solver.collider.use_gjk = USE_GJK
    if gs_sim.avatar_solver.is_active():
        gs_sim.avatar_solver.collider.use_gjk = USE_GJK
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, qpos, num_steps=300, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("xml_path", ["xml/one_ball_joint.xml"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_one_ball_joint(gs_sim, mj_sim, tol):
    # FIXME: Mujoco is detecting collision for some reason...
    mj_sim.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

    if gs_sim.rigid_solver.is_active():
        gs_sim.rigid_solver.collider.use_gjk = USE_GJK
    if gs_sim.avatar_solver.is_active():
        gs_sim.avatar_solver.collider.use_gjk = USE_GJK
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, num_steps=600, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("xml_path", ["xml/rope_ball.xml", "xml/rope_hinge.xml"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_rope_ball(gs_sim, mj_sim, gs_solver, tol):
    # Make sure it is possible to set the configuration vector without failure
    gs_sim.rigid_solver.set_dofs_position(gs_sim.rigid_solver.get_dofs_position())

    if gs_sim.rigid_solver.is_active():
        gs_sim.rigid_solver.collider.use_gjk = USE_GJK
    if gs_sim.avatar_solver.is_active():
        gs_sim.avatar_solver.collider.use_gjk = USE_GJK
    check_mujoco_model_consistency(gs_sim, mj_sim, tol=tol)
    tol = 2e-9 if gs_solver == gs.constraint_solver.Newton else tol
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, num_steps=300, tol=tol)


@pytest.mark.required
@pytest.mark.xdist_group(name="huggingface_hub")
@pytest.mark.multi_contact(False)
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_urdf_rope(
    gs_solver, gs_integrator, multi_contact, mujoco_compatibility, adjacent_collision, dof_damping, show_viewer
):
    asset_path = snapshot_download(
        repo_type="dataset",
        repo_id="Genesis-Intelligence/assets",
        allow_patterns="linear_deformable.urdf",
        max_workers=1,
    )
    xml_path = os.path.join(asset_path, "linear_deformable.urdf")

    mj_sim = build_mujoco_sim(xml_path, gs_solver, gs_integrator, multi_contact, adjacent_collision, dof_damping)
    gs_sim = build_genesis_sim(
        xml_path, gs_solver, gs_integrator, multi_contact, mujoco_compatibility, adjacent_collision, show_viewer, mj_sim
    )

    # FIXME: Tolerance must be very large due to small masses and compounding of errors over long kinematic chains
    if gs_sim.rigid_solver.is_active():
        gs_sim.rigid_solver.collider.use_gjk = USE_GJK
    if gs_sim.avatar_solver.is_active():
        gs_sim.avatar_solver.collider.use_gjk = USE_GJK
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, num_steps=300, tol=5e-5)


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["two_aligned_hinges"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
def test_link_velocity(gs_sim, tol):
    # Check the velocity for a few "easy" special cases
    init_simulators(gs_sim, qvel=np.array([0.0, 1.0]))
    assert_allclose(gs_sim.rigid_solver.links_state.vel.to_numpy(), 0, tol=tol)

    init_simulators(gs_sim, qvel=np.array([1.0, 0.0]))
    cvel_0, cvel_1 = gs_sim.rigid_solver.links_state.vel.to_numpy()[:, 0]
    assert_allclose(cvel_0, np.array([0.0, 0.5, 0.0]), tol=tol)
    assert_allclose(cvel_1, np.array([0.0, 0.5, 0.0]), tol=tol)

    init_simulators(gs_sim, qpos=np.array([0.0, np.pi / 2.0]), qvel=np.array([0.0, 1.2]))
    COM = gs_sim.rigid_solver.links_state[0, 0].COM
    assert_allclose(COM, np.array([0.375, 0.125, 0.0]), tol=tol)
    xanchor = gs_sim.rigid_solver.joints_state[1, 0].xanchor
    assert_allclose(xanchor, np.array([0.5, 0.0, 0.0]), tol=tol)
    cvel_0, cvel_1 = gs_sim.rigid_solver.links_state.vel.to_numpy()[:, 0]
    assert_allclose(cvel_0, 0, tol=tol)
    assert_allclose(cvel_1, np.array([-1.2 * (0.125 - 0.0), 1.2 * (0.375 - 0.5), 0.0]), tol=tol)

    # Check that the velocity is valid for a random configuration
    init_simulators(gs_sim, qpos=np.array([-0.7, 0.2]), qvel=np.array([3.0, 13.0]))
    xanchor = gs_sim.rigid_solver.joints_state[1, 0].xanchor
    theta_0, theta_1 = gs_sim.rigid_solver.qpos.to_numpy()[:, 0]
    assert_allclose(xanchor[0], 0.5 * np.cos(theta_0), tol=tol)
    assert_allclose(xanchor[1], 0.5 * np.sin(theta_0), tol=tol)
    COM = gs_sim.rigid_solver.links_state[0, 0].COM
    COM_0 = np.array([0.25 * np.cos(theta_0), 0.25 * np.sin(theta_0), 0.0])
    COM_1 = np.array(
        [
            0.5 * np.cos(theta_0) + 0.25 * np.cos(theta_0 + theta_1),
            0.5 * np.sin(theta_0) + 0.25 * np.sin(theta_0 + theta_1),
            0.0,
        ]
    )
    assert_allclose(COM, 0.5 * (COM_0 + COM_1), tol=tol)

    cvel_0, cvel_1 = gs_sim.rigid_solver.links_state.vel.to_numpy()[:, 0]
    omega_0, omega_1 = gs_sim.rigid_solver.links_state.ang.to_numpy()[:, 0, 2]
    assert_allclose(omega_0, 3.0, tol=tol)
    assert_allclose(omega_1 - omega_0, 13.0, tol=tol)
    cvel_0_ = omega_0 * np.array([-COM[1], COM[0], 0.0])
    assert_allclose(cvel_0, cvel_0_, tol=tol)
    cvel_1_ = cvel_0 + (omega_1 - omega_0) * np.array([xanchor[1] - COM[1], COM[0] - xanchor[0], 0.0])
    assert_allclose(cvel_1, cvel_1_, tol=tol)

    xpos_0, xpos_1 = gs_sim.rigid_solver.links_state.pos.to_numpy()[:, 0]
    assert_allclose(xpos_0, 0.0, tol=tol)
    assert_allclose(xpos_1, xanchor, tol=tol)
    xvel_0, xvel_1 = gs_sim.rigid_solver.get_links_vel()
    assert_allclose(xvel_0, 0.0, tol=tol)
    xvel_1_ = omega_0 * np.array([-xpos_1[1], xpos_1[0], 0.0])
    assert_allclose(xvel_1, xvel_1_, tol=tol)
    civel_0, civel_1 = gs_sim.rigid_solver.get_links_vel(ref="link_com")
    civel_0_ = omega_0 * np.array([-COM_0[1], COM_0[0], 0.0])
    assert_allclose(civel_0, civel_0_, tol=tol)
    civel_1_ = omega_0 * np.array([-COM_1[1], COM_1[0], 0.0]) + (omega_1 - omega_0) * np.array(
        [xanchor[1] - COM_1[1], COM_1[0] - xanchor[0], 0.0]
    )
    assert_allclose(civel_1, civel_1_, tol=tol)


@pytest.mark.parametrize("model_name", ["box_box"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_box_box_dynamics(gs_sim):
    (gs_robot,) = gs_sim.entities
    for _ in range(20):
        cube1_pos = np.array([0.0, 0.0, 0.2])
        cube1_quat = np.array([1.0, 0.0, 0.0, 0.0])
        cube2_pos = np.array([0.0, 0.0, 0.65 + 0.1 * np.random.rand()])
        cube2_quat = gs.utils.geom.xyz_to_quat(
            np.array([*(0.15 * np.random.rand(2)), np.pi * np.random.rand()]),
        )
        init_simulators(gs_sim, qpos=np.concatenate((cube1_pos, cube1_quat, cube2_pos, cube2_quat)))
        if gs_sim.rigid_solver.is_active():
            gs_sim.rigid_solver.collider.use_gjk = USE_GJK
        if gs_sim.avatar_solver.is_active():
            gs_sim.avatar_solver.collider.use_gjk = USE_GJK
            
        for i in range(110):
            gs_sim.scene.step()
            if i > 100:
                qvel = gs_robot.get_dofs_velocity().cpu()
                assert_allclose(qvel, 0, atol=1e-2)

        qpos = gs_robot.get_dofs_position().cpu()
        assert_allclose(qpos[8], 0.6, atol=2e-3)


@pytest.mark.parametrize("box_box_detection, dynamics", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize("backend", [gs.cpu])  # TODO: Cannot afford GPU test for this one
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
    
    if scene.rigid_solver.is_active():
        scene.rigid_solver.collider.use_gjk = USE_GJK
    if scene.avatar_solver.is_active():
        scene.avatar_solver.collider.use_gjk = USE_GJK

    if dynamics:
        for entity in scene.entities[1:]:
            entity.set_dofs_velocity(4.0 * np.random.rand(6))
    num_steps = 650 if dynamics else 150
    for i in range(num_steps):
        scene.step()
        if i > num_steps - 50:
            qvel = scene.rigid_solver.get_dofs_velocity().cpu().reshape((6, -1))
            # Checking the average velocity because is always one cube moving depending on the machine.
            assert_allclose(torch.linalg.norm(qvel, dim=0).mean(), 0, atol=0.05)

    for n, entity in enumerate(scene.entities[1:]):
        i, j, k = int(n / 25), int(n / 5) % 5, n % 5
        qpos = entity.get_dofs_position().cpu()
        if dynamics:
            assert qpos[:2].norm() < 20.0
            assert qpos[2] < 5.0
        else:
            qpos0 = np.array((i * 1.01, j * 1.01, k * 1.01 + 0.5))
            assert_allclose(qpos[:3], qpos0, atol=0.05)
            assert_allclose(qpos[3:], 0, atol=0.03)


@pytest.mark.required
@pytest.mark.parametrize("xml_path", ["xml/franka_emika_panda/panda.xml"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_robot_kinematics(gs_sim, mj_sim, tol):
    # Disable all constraints and actuation
    mj_sim.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    mj_sim.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
    gs_sim.rigid_solver.dofs_state.ctrl_mode.fill(gs.CTRL_MODE.FORCE)
    gs_sim.rigid_solver._enable_collision = False
    gs_sim.rigid_solver._enable_joint_limit = False
    gs_sim.rigid_solver._disable_constraint = True
    
    if gs_sim.rigid_solver.is_active():
        gs_sim.rigid_solver.collider.use_gjk = USE_GJK
    if gs_sim.avatar_solver.is_active():
        gs_sim.avatar_solver.collider.use_gjk = USE_GJK

    check_mujoco_model_consistency(gs_sim, mj_sim, tol=tol)

    (gs_robot,) = gs_sim.entities
    dof_bounds = gs_sim.rigid_solver.dofs_info.limit.to_numpy()
    for _ in range(100):
        qpos = dof_bounds[:, 0] + (dof_bounds[:, 1] - dof_bounds[:, 0]) * np.random.rand(gs_robot.n_qs)
        init_simulators(gs_sim, mj_sim, qpos)
        check_mujoco_data_consistency(gs_sim, mj_sim, tol=tol)


@pytest.mark.required
def test_robot_scaling(show_viewer, tol):
    mass = None
    links_pos = None
    for scale in (0.5, 1.0, 2.0):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                gravity=(0, 0, -10.0),
            ),
            show_viewer=show_viewer,
            show_FPS=False,
        )
        robot = scene.add_entity(
            gs.morphs.MJCF(
                file="xml/franka_emika_panda/panda.xml",
                scale=scale,
            ),
        )
        scene.build()
        
        if scene.rigid_solver.is_active():
            scene.rigid_solver.collider.use_gjk = USE_GJK
        if scene.avatar_solver.is_active():
            scene.avatar_solver.collider.use_gjk = USE_GJK

        mass_ = robot.get_mass() / scale**3
        if mass is None:
            mass = mass_
        assert_allclose(mass, mass_, tol=tol)

        dofs_lower_bound, dofs_upper_bound = robot.get_dofs_limit()
        qpos = dofs_lower_bound
        robot.set_dofs_position(qpos)

        links_pos_ = robot.get_links_pos() / scale
        if links_pos is None:
            links_pos = links_pos_
        assert_allclose(links_pos, links_pos_, tol=tol)

        scene.step()
        qf_passive = scene.rigid_solver.dofs_state.qf_passive.to_numpy()
        assert_allclose(qf_passive, 0, tol=tol)


@pytest.mark.required
def test_info_batching(tol):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            batch_dofs_info=True,
            batch_joints_info=True,
            batch_links_info=True,
        ),
        show_viewer=False,
        show_FPS=False,
    )
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    scene.build(n_envs=2)
    
    if scene.rigid_solver.is_active():
        scene.rigid_solver.collider.use_gjk = USE_GJK
    if scene.avatar_solver.is_active():
        scene.avatar_solver.collider.use_gjk = USE_GJK

    scene.step()
    qposs = robot.get_qpos()
    assert_allclose(qposs[0], qposs[1], tol=tol)


@pytest.mark.required
@pytest.mark.xfail(reason="This test is not passing on all platforms for now.")
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
    
    if scene.rigid_solver.is_active():
        scene.rigid_solver.collider.use_gjk = USE_GJK
    if scene.avatar_solver.is_active():
        scene.avatar_solver.collider.use_gjk = USE_GJK

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
                rgb_array, *_ = cam.render()
                assert np.std(rgb_array) > 10.0
                robots_rgb_arrays.append(rgb_array)

            steps_rgb_arrays.append(robots_rgb_arrays)

        for i in range(3):
            assert_allclose(steps_rgb_arrays[0][i], steps_rgb_arrays[1][i], tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu])
def test_pd_control(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            substeps=1,  # This is essential to be able to emulate native PD control
        ),
        rigid_options=gs.options.RigidOptions(
            batch_dofs_info=True,
            enable_self_collision=False,
            integrator=gs.integrator.approximate_implicitfast,
        ),
        # vis_options=gs.options.VisOptions(
        #     rendered_envs_idx=(1,),
        # ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
    )
    scene.build(n_envs=2)
    
    if scene.rigid_solver.is_active():
        scene.rigid_solver.collider.use_gjk = USE_GJK
    if scene.avatar_solver.is_active():
        scene.avatar_solver.collider.use_gjk = USE_GJK

    MOTORS_POS_TARGET = torch.tensor(
        [0.6900, -0.1100, -0.7200, -2.7300, -0.1500, 2.6400, 0.8900, 0.0400, 0.0400],
        dtype=gs.tc_float,
        device=gs.device,
    )
    MOTORS_KP = torch.tensor(
        [4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0, 100.0, 100.0],
        dtype=gs.tc_float,
        device=gs.device,
    )
    MOTORS_KD = torch.tensor(
        [450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0, 10.0, 10.0],
        dtype=gs.tc_float,
        device=gs.device,
    )

    robot.set_dofs_kp(MOTORS_KP, envs_idx=0)
    robot.set_dofs_kv(MOTORS_KD, envs_idx=0)
    robot.control_dofs_position(MOTORS_POS_TARGET, envs_idx=0)

    # Must update DoF armature to emulate implicit damping for force control.
    # This is equivalent to the first-order correction term involved in implicit integration scheme,
    # in the particular case where `approximate_implicitfast` integrator is used.
    robot.set_dofs_armature(robot.get_dofs_armature(envs_idx=1) + MOTORS_KD * scene.sim._substep_dt, envs_idx=1)

    for i in range(1000):
        dofs_pos = robot.get_dofs_position(envs_idx=1)
        dofs_vel = robot.get_dofs_velocity(envs_idx=1)
        dofs_torque = MOTORS_KP * (MOTORS_POS_TARGET - dofs_pos) - MOTORS_KD * dofs_vel
        robot.control_dofs_force(dofs_torque, envs_idx=1)
        scene.step()
        qf_applied = scene.rigid_solver.dofs_state.qf_applied.to_torch(device="cpu").T
        # dofs_torque = robot.get_dofs_control_force().cpu()
        assert_allclose(qf_applied[0], qf_applied[1], tol=1e-6)


@pytest.mark.required
def test_set_root_pose(show_viewer, tol):
    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=(0.0, 0.4, 0.1),
            euler=(0, 0, 90),
        ),
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.65, 0.0, 0.02),
        ),
    )
    scene.build()
    
    if scene.rigid_solver.is_active():
        scene.rigid_solver.collider.use_gjk = USE_GJK
    if scene.avatar_solver.is_active():
        scene.avatar_solver.collider.use_gjk = USE_GJK

    for _ in range(2):
        scene.reset()

        assert_allclose(robot.get_pos(), (0.0, 0.4, 0.1), tol=tol)
        assert_allclose(gs.utils.geom.quat_to_xyz(robot.get_quat(), rpy=True, degrees=True), (0, 0, 90), tol=tol)
        robot.set_pos(torch.tensor((-0.1, -0.2, 0.2), dtype=gs.tc_float))
        assert_allclose(robot.get_pos(), (-0.1, -0.2, 0.2), tol=tol)

        assert_allclose(cube.get_pos(), (0.65, 0.0, 0.02), tol=tol)
        cube.set_pos(torch.tensor((0.0, 0.5, 0.2), dtype=gs.tc_float))
        assert_allclose(cube.get_pos(), (0.0, 0.5, 0.2), tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("n_envs, batched", [(0, False), (3, True)])
def test_set_sol_params(n_envs, batched, tol):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=1,
        ),
        rigid_options=gs.options.RigidOptions(
            batch_joints_info=batched,
        ),
        show_viewer=False,
        show_FPS=False,
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=(0.0, 0.4, 0.1),
            euler=(0, 0, 90),
        ),
    )
    scene.build(n_envs=2)
    
    if scene.rigid_solver.is_active():
        scene.rigid_solver.collider.use_gjk = USE_GJK
    if scene.avatar_solver.is_active():
        scene.avatar_solver.collider.use_gjk = USE_GJK
        
    assert scene.sim._substep_dt == 0.01
    

    for objs, batched in ((robot.joints, batched), (robot.geoms, False), (robot.equalities, True)):
        for obj in objs:
            sol_params = obj.sol_params + 1.0
            obj.set_sol_params(sol_params)
            with pytest.raises(AssertionError):
                assert_allclose(obj.sol_params, sol_params, tol=tol)
            sol_params = np.zeros(((scene.n_envs,) if scene.n_envs > 0 and batched else ()) + (7,))
            obj.set_sol_params(sol_params)
            sol_params = np.tile(
                [2.0e-02, 0.0, 1e-4, 1e-4, 0.0, 1e-4, 1.0],
                ((scene.n_envs,) if scene.n_envs > 0 and batched else ()) + (1,),
            )
            assert_allclose(obj.sol_params, sol_params, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("xml_path", ["xml/humanoid.xml"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_stickman(gs_sim, mj_sim, tol):
    # Make sure that "static" model information are matching
    check_mujoco_model_consistency(gs_sim, mj_sim, tol=tol)

    # Initialize the simulation
    init_simulators(gs_sim)
    
    if gs_sim.scene.rigid_solver.is_active():
        gs_sim.scene.rigid_solver.collider.use_gjk = USE_GJK
    if gs_sim.scene.avatar_solver.is_active():
        gs_sim.scene.avatar_solver.collider.use_gjk = USE_GJK

    # Run the simulation for a few steps
    for i in range(5100):
        gs_sim.scene.step()
        if i > 5000:
            (gs_robot,) = gs_sim.entities
            qvel = gs_robot.get_dofs_velocity().cpu()
            assert_allclose(qvel, 0, atol=0.4)

    qpos = gs_robot.get_dofs_position().cpu()
    assert np.linalg.norm(qpos[:2]) < 1.3
    body_z = gs_sim.rigid_solver.links_state.pos.to_numpy()[:-1, 0, 2]
    np.testing.assert_array_less(0, body_z + gs.EPS)


def move_cube(use_suction, mode, show_viewer):
    # Add DoF armature to improve numerical stability if not using 'approximate_implicitfast' integrator.
    #
    # This is necessary because the first-order correction term involved in the implicit integration schemes
    # 'implicitfast' and 'Euler' are only able to stabilize each entity independently, from the forces that were
    # obtained from the instable accelerations. As a result, eveything is fine as long as the entities are not
    # interacting with each other, but it induces unrealistic motion otherwise. In this case, the acceleration of the
    # cube being lifted is based on the acceleration that the gripper would have without implicit damping.
    #
    # The only way to correct this would be to take into account the derivative of the Jacobian of the constraints in
    # the first-order correction term. Doing this is challenging and would significantly increase the computation cost.
    #
    # In practice, it is more common to just go for a higher order integrator such as RK4.
    if mode == 0:
        integrator = gs.integrator.approximate_implicitfast
        substeps = 1
        armature = 0.0
    elif mode == 1:
        integrator = gs.integrator.implicitfast
        substeps = 4
        armature = 0.0
    elif mode == 2:
        integrator = gs.integrator.Euler
        substeps = 1
        armature = 2.0

    # Create and build the scene
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=substeps,
        ),
        rigid_options=gs.options.RigidOptions(
            box_box_detection=True,
            integrator=integrator,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(0.65, 0.0, 0.025),
        ),
        surface=gs.surfaces.Plastic(color=(1, 0, 0)),
    )
    cube_2 = scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(0.4, 0.2, 0.025),
        ),
        surface=gs.surfaces.Plastic(color=(0, 1, 0)),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        vis_mode="collision",
        visualize_contact=True,
    )
    scene.build()
    if scene.rigid_solver.is_active():
        scene.rigid_solver.collider.use_gjk = USE_GJK
    if scene.avatar_solver.is_active():
        scene.avatar_solver.collider.use_gjk = USE_GJK

    franka.set_dofs_armature(franka.get_dofs_armature() + armature)

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
        resolution=0.05,
        timeout=30.0,
    )
    assert path
    # execute the planned path
    franka.control_dofs_position(np.array([0.15, 0.15]), fingers_dof)
    for waypoint in path:
        franka.control_dofs_position(waypoint)
        scene.step()

    # Get more time to the robot to reach the last waypoint
    for i in range(100):
        scene.step()

    # reach
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.13]),
        quat=np.array([0, 1, 0, 0]),
    )
    franka.control_dofs_position(qpos[:-2], motors_dof)
    for i in range(50):
        scene.step()

    rigid = scene.sim.rigid_solver

    # grasp
    if use_suction:
        link_cube = np.array([cube.get_link("box_baselink").idx], dtype=gs.np_int)
        link_franka = np.array([franka.get_link("hand").idx], dtype=gs.np_int)
        rigid.add_weld_constraint(link_cube, link_franka)
    else:
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(np.array([-1.0, -1.0]), fingers_dof)
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
        pos=np.array([0.4, 0.2, 0.2]),
        quat=np.array([0, 1, 0, 0]),
    )
    path = franka.plan_path(
        qpos_goal=qpos,
        num_waypoints=150,
        resolution=0.05,
        timeout=30.0,
    )
    assert path
    for waypoint in path:
        franka.control_dofs_position(waypoint[:-2], motors_dof)
        scene.step()

    # Get more time to the robot to reach the last waypoint
    for i in range(50):
        scene.step()

    # release
    if use_suction:
        rigid.delete_weld_constraint(link_cube, link_franka)
    else:
        franka.control_dofs_position(np.array([0.15, 0.15]), fingers_dof)

    for i in range(550):
        scene.step()
        if i > 550:
            qvel = cube.get_dofs_velocity().cpu()
            assert_allclose(qvel, 0, atol=0.06)

    qpos = cube.get_dofs_position().cpu()
    assert_allclose(qpos[2], 0.075, atol=2e-3)


@pytest.mark.skipif(sys.platform == "win32", reason="OMPL is not supported on Windows OS.")
@pytest.mark.parametrize(
    "mode, backend",
    [
        pytest.param(0, gs.cpu, marks=pytest.mark.required),
        pytest.param(1, gs.cpu),
        pytest.param(2, gs.cpu),
        pytest.param(0, gs.gpu),
        pytest.param(1, gs.gpu),
        pytest.param(2, gs.gpu),
    ],
)
def test_inverse_kinematics(mode, show_viewer):
    move_cube(use_suction=False, mode=mode, show_viewer=show_viewer)


@pytest.mark.skipif(sys.platform == "win32", reason="OMPL is not supported on Windows OS.")
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_suction_cup(mode, show_viewer):
    move_cube(use_suction=True, mode=mode, show_viewer=show_viewer)


@pytest.mark.required
@pytest.mark.skipif(sys.platform == "win32", reason="OMPL is not supported on Windows OS.")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_path_planning_avoidance(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, 1, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            max_FPS=60,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    cubes = []
    for pos in (
        (-0.1, 0.2, 0.7),
        (0.0, 0.3, 0.8),
        (-0.1, -0.2, 0.7),
        (0.0, -0.3, 0.8),
        (0.3, 0.2, 0.6),
        (0.3, -0.2, 0.6),
        (0.3, 0.3, 0.7),
        (0.3, -0.3, 0.7),
    ):
        cube = scene.add_entity(
            gs.morphs.Box(
                size=(0.1, 0.1, 0.1),
                pos=pos,
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                color=(*np.random.rand(3), 0.7),
            ),
        )
        cubes.append(cube)
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        vis_mode="collision",
    )
    scene.build()
    
    if scene.rigid_solver.is_active():
        scene.rigid_solver.collider.use_gjk = USE_GJK
    if scene.avatar_solver.is_active():
        scene.avatar_solver.collider.use_gjk = USE_GJK

    hand = franka.get_link("hand")
    hand_pos_ref = torch.tensor([0.3, 0.25, 0.25], device=gs.device)
    hand_quat_ref = torch.tensor([0.3073, 0.5303, 0.7245, -0.2819], device=gs.device)
    qpos = franka.inverse_kinematics(hand, pos=hand_pos_ref, quat=hand_quat_ref)
    qpos[-2:] = 0.04

    avoidance_path = franka.plan_path(
        qpos_goal=qpos,
        num_waypoints=200,
        ignore_collision=False,
        resolution=0.002,
        timeout=180.0,
        max_retry=5,
    )
    assert avoidance_path
    assert_allclose(avoidance_path[0].cpu(), 0, tol=gs.EPS)
    assert_allclose(avoidance_path[-1].cpu(), qpos.cpu(), tol=gs.EPS)
    free_path = franka.plan_path(
        qpos_goal=qpos,
        num_waypoints=200,
        ignore_collision=True,
        resolution=0.002,
    )
    assert free_path
    assert_allclose(free_path[0].cpu(), 0, tol=gs.EPS)
    assert_allclose(free_path[-1].cpu(), qpos.cpu(), tol=gs.EPS)

    for path, ignore_collision in ((free_path, False), (avoidance_path, True)):
        max_penetration = float("-inf")
        for waypoint in path:
            franka.set_qpos(waypoint)
            scene.visualizer.update()

            # Check if the cube is colliding with the robot
            scene.rigid_solver._kernel_forward_dynamics()
            scene.rigid_solver._func_constraint_force()
            for i in range(scene.rigid_solver.collider.n_contacts.to_numpy()[0]):
                contact_data = scene.rigid_solver.collider.contact_data[i, 0]
                if any(i_g in tuple(range(len(cubes))) for i_g in (contact_data.link_a, contact_data.link_b)):
                    max_penetration = max(max_penetration, contact_data.penetration)

        args = (max_penetration, 5e-3)
        np.testing.assert_array_less(*(args if ignore_collision else args[::-1]))

        assert_allclose(hand_pos_ref.cpu(), hand.get_pos().cpu(), tol=5e-4)
        hand_quat_diff = gs.utils.geom.transform_quat_by_quat(
            gs.utils.geom.inv_quat(hand_quat_ref.cpu()), hand.get_quat().cpu()
        )
        theta = 2 * np.arctan2(torch.linalg.norm(hand_quat_diff[1:]), torch.abs(hand_quat_diff[0]))
        assert_allclose(theta, 0.0, tol=5e-3)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu])
def test_all_fixed(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, 1, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            max_FPS=60,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.0),
            fixed=True,
        ),
    )
    scene.build()
    
    if scene.rigid_solver.is_active():
        scene.rigid_solver.collider.use_gjk = USE_GJK
    if scene.avatar_solver.is_active():
        scene.avatar_solver.collider.use_gjk = USE_GJK
        
    scene.step()

    assert_allclose(cube.get_pos(), 0, tol=gs.EPS)
    assert_allclose(cube.get_quat(), (1.0, 0.0, 0.0, 0.0), tol=gs.EPS)
    assert_allclose(cube.get_vel(), 0, tol=gs.EPS)
    assert_allclose(cube.get_ang(), 0, tol=gs.EPS)
    assert_allclose(scene.rigid_solver.get_links_acc(), 0, tol=gs.EPS)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu])
def test_mass_mat(show_viewer, tol):
    # Create and build the scene
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=1,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    franka1 = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=(0, 0, 0)),
        vis_mode="collision",
        visualize_contact=True,
    )
    franka2 = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=(0, 2, 0)),
        vis_mode="collision",
        visualize_contact=True,
    )
    scene.build()
    
    if scene.rigid_solver.is_active():
        scene.rigid_solver.collider.use_gjk = USE_GJK
    if scene.avatar_solver.is_active():
        scene.avatar_solver.collider.use_gjk = USE_GJK

    mass_mat_1 = franka1.get_mass_mat(decompose=False)
    mass_mat_2 = franka2.get_mass_mat(decompose=False)
    assert mass_mat_1.shape == (franka1.n_dofs, franka1.n_dofs)
    assert_allclose(mass_mat_1, mass_mat_2, tol=tol)

    mass_mat_L, mass_mat_D_inv = franka1.get_mass_mat(decompose=True)
    mass_mat = mass_mat_L.T @ torch.diag(1.0 / mass_mat_D_inv) @ mass_mat_L
    assert_allclose(mass_mat, mass_mat_1, tol=tol)


@pytest.mark.parametrize("backend", [gs.cpu])
def test_nonconvex_collision(show_viewer):
    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )
    tank = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/tank.obj",
            scale=5.0,
            fixed=True,
            euler=(90, 0, 0),
            convexify=False,
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
        visualize_contact=True,
    )
    scene.build()
    
    if scene.rigid_solver.is_active():
        scene.rigid_solver.collider.use_gjk = USE_GJK
    if scene.avatar_solver.is_active():
        scene.avatar_solver.collider.use_gjk = USE_GJK

    # Force numpy seed because this test is very sensitive to the initial condition
    np.random.seed(0)
    ball.set_dofs_velocity(np.random.rand(ball.n_dofs) * 0.8)
    for i in range(1800):
        scene.step()
        if i > 1700:
            qvel = scene.sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]
            assert_allclose(qvel, 0, atol=0.65)


# FIXME: Force executing all 'huggingface_hub' tests on the same worker to prevent hitting HF rate limit
@pytest.mark.xdist_group(name="huggingface_hub")
@pytest.mark.parametrize("convexify", [True, False])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_mesh_repair(convexify, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.004,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    asset_path = snapshot_download(
        repo_type="dataset",
        repo_id="Genesis-Intelligence/assets",
        allow_patterns="work_table.glb",
        max_workers=1,
    )
    table = scene.add_entity(
        gs.morphs.Mesh(
            file=f"{asset_path}/work_table.glb",
            pos=(0.4, 0.0, -0.54),
            fixed=True,
        ),
        vis_mode="collision",
    )
    asset_path = snapshot_download(
        repo_type="dataset",
        repo_id="Genesis-Intelligence/assets",
        allow_patterns="spoon.glb",
        max_workers=1,
    )
    obj = scene.add_entity(
        gs.morphs.Mesh(
            file=f"{asset_path}/spoon.glb",
            pos=(0.3, 0, 0.015),
            quat=(0.707, 0.707, 0, 0),
            convexify=convexify,
            scale=1.0,
        ),
        vis_mode="collision",
        visualize_contact=True,
    )
    scene.build()
    
    if scene.rigid_solver.is_active():
        scene.rigid_solver.collider.use_gjk = USE_GJK
    if scene.avatar_solver.is_active():
        scene.avatar_solver.collider.use_gjk = USE_GJK

    if convexify:
        assert all(geom.metadata["decomposed"] for geom in obj.geoms)

    # MPR collision detection is significantly less reliable than SDF in terms of penetration depth estimation.
    tol_pos = 0.05 if convexify else 1e-6
    tol_rot = 1.3 if convexify else 1e-4
    for i in range(400):
        scene.step()
        if i > 300:
            qvel = obj.get_dofs_velocity().cpu()
            assert_allclose(qvel[:3], 0, atol=tol_pos)
            assert_allclose(qvel[3:], 0, atol=tol_rot)
    qpos = obj.get_dofs_position().cpu()
    assert_allclose(qpos[:2], (0.3, 0.0), atol=1e-3)


@pytest.mark.required
@pytest.mark.xdist_group(name="huggingface_hub")
@pytest.mark.parametrize("euler", [(90, 0, 90), (74, 15, 90)])
@pytest.mark.parametrize("gjk_collision", [True])
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_convexify(euler, gjk_collision, backend, show_viewer):
    OBJ_OFFSET_X = 0.0  # 0.02
    OBJ_OFFSET_Y = 0.15

    # The test check that the volume difference is under a given threshold and
    # that convex decomposition is only used whenever it is necessary.
    # Then run a simulation to see if it explodes, i.e. objects are at reset inside tank.
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            dt=0.004,
            use_gjk_collision=gjk_collision,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    box = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/blue_box/model.urdf",
            fixed=True,
            pos=(0.0, 1.0, 0.0),
        ),
        vis_mode="collision",
    )
    tank = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/tank.obj",
            scale=5.0,
            fixed=True,
            pos=(0.05, -0.1, 0.0),
            euler=euler,
            # coacd_options=gs.options.CoacdOptions(
            #     threshold=0.08,
            # ),
        ),
        vis_mode="collision",
    )
    objs = []
    for i, asset_name in enumerate(("mug_1", "donut_0", "cup_2", "apple_15")):
        asset_path = snapshot_download(
            repo_type="dataset",
            repo_id="Genesis-Intelligence/assets",
            allow_patterns=f"{asset_name}/*",
            max_workers=1,
        )
        obj = scene.add_entity(
            gs.morphs.MJCF(
                file=f"{asset_path}/{asset_name}/output.xml",
                pos=(OBJ_OFFSET_X * (1.5 - i), OBJ_OFFSET_Y * (i - 1.5), 0.4),
            ),
            vis_mode="collision",
            visualize_contact=True,
        )
        objs.append(obj)
    # cam = scene.add_camera(
    #     pos=(0.5, 0.0, 1.0),
    #     lookat=(0.0, 0.0, 0.0),
    #     res=(500, 500),
    #     fov=60,
    #     spp=512,
    #     GUI=False,
    # )
    scene.build()
    
    if scene.rigid_solver.is_active():
        scene.rigid_solver.collider.use_gjk = USE_GJK
    if scene.avatar_solver.is_active():
        scene.avatar_solver.collider.use_gjk = USE_GJK
        
    gs_sim = scene.sim

    # Make sure that all the geometries in the scene are convex
    assert gs_sim.rigid_solver.geoms_info.is_convex.to_numpy().all()
    assert not gs_sim.rigid_solver.collider._has_nonconvex_nonterrain

    # There should be only one geometry for the apple as it can be convexify without decomposition,
    # but for the others it is hard to tell... Let's use some reasonable guess.
    mug, donut, cup, apple = objs
    assert len(apple.geoms) == 1
    assert all(geom.metadata["decomposed"] for geom in donut.geoms) and 5 <= len(donut.geoms) <= 10
    assert all(geom.metadata["decomposed"] for geom in cup.geoms) and 5 <= len(cup.geoms) <= 20
    assert all(geom.metadata["decomposed"] for geom in mug.geoms) and 5 <= len(mug.geoms) <= 40
    assert all(geom.metadata["decomposed"] for geom in box.geoms) and 5 <= len(box.geoms) <= 20

    # Check resting conditions repeateadly rather not just once, for numerical robustness
    # cam.start_recording()
    num_steps = 1300 if euler == (90, 0, 90) else 1100
    atol = 0.65 if euler == (90, 0, 90) and backend == gs.gpu else 1.5
    for i in range(num_steps):
        scene.step()
        # cam.render()
        if i > num_steps - 100:
            qvel = gs_sim.rigid_solver.get_dofs_velocity().cpu()
            assert_allclose(qvel, 0, atol=atol)
    # cam.stop_recording(save_to_filename="video.mp4", fps=60)

    for obj in objs:
        qpos = obj.get_dofs_position().cpu()
        np.testing.assert_array_less(-0.1, qpos[2])
        np.testing.assert_array_less(qpos[2], 0.15)
        np.testing.assert_array_less(torch.linalg.norm(qpos[:2]), 0.5)

    # Check that the mug and donut are landing straight if the tank is horizontal.
    # The cup is tipping because it does not land flat due to convex decomposition error.
    if euler == (90, 0, 90):
        for i, obj in enumerate((mug, donut)):
            qpos = obj.get_dofs_position().cpu()
            assert_allclose(qpos[0], OBJ_OFFSET_X * (1.5 - i), atol=7e-3)
            assert_allclose(qpos[1], OBJ_OFFSET_Y * (i - 1.5), atol=5e-3)


@pytest.mark.mujoco_compatibility(False)
@pytest.mark.parametrize("mode", range(9))
@pytest.mark.parametrize("model_name", ["collision_edge_cases"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_collision_edge_cases(gs_sim, mode):
    
    if gs_sim.scene.rigid_solver.is_active():
        gs_sim.scene.rigid_solver.collider.use_gjk = USE_GJK
    if gs_sim.scene.avatar_solver.is_active():
        gs_sim.scene.avatar_solver.collider.use_gjk = USE_GJK
        
    qpos_0 = gs_sim.rigid_solver.get_dofs_position().cpu()
    for _ in range(200):
        gs_sim.scene.step()

    qvel = gs_sim.rigid_solver.get_dofs_velocity().cpu()
    assert_allclose(qvel, 0, atol=1e-2)
    qpos = gs_sim.rigid_solver.get_dofs_position().cpu()
    assert_allclose(qpos[[0, 1, 3, 4, 5]], qpos_0[[0, 1, 3, 4, 5]], atol=1e-4)


@pytest.mark.required
@pytest.mark.xdist_group(name="huggingface_hub")
@pytest.mark.parametrize("backend", [gs.cpu])
def test_collision_plane_convex(show_viewer, tol):
    for morph in (
        gs.morphs.Plane(),
        gs.morphs.Box(
            pos=(0.5, 0.0, -0.5),
            size=(1.0, 1.0, 1.0),
            fixed=True,
        ),
    ):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=0.001,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.0, -0.5, 0.5),
                camera_lookat=(0.5, 0.0, 0.0),
                camera_fov=30,
                max_FPS=60,
            ),
            show_viewer=show_viewer,
            show_FPS=False,
        )

        scene.add_entity(morph)

        asset_path = snapshot_download(
            repo_type="dataset",
            repo_id="Genesis-Intelligence/assets",
            allow_patterns="image_0000_segmented.glb",
            max_workers=1,
        )
        asset = scene.add_entity(
            gs.morphs.Mesh(
                file=f"{asset_path}/image_0000_segmented.glb",
                scale=0.03196910891804585,
                pos=(0.45184245, 0.05020455, 0.02),
                quat=(0.51982231, 0.44427745, 0.49720965, 0.53402704),
            ),
            vis_mode="collision",
            visualize_contact=True,
        )

        scene.build()
        
        if scene.rigid_solver.is_active():
            scene.rigid_solver.collider.use_gjk = USE_GJK
        if scene.avatar_solver.is_active():
            scene.avatar_solver.collider.use_gjk = USE_GJK

        for i in range(500):
            scene.step()
            if i > 400:
                qvel = asset.get_dofs_velocity()
                assert_allclose(qvel, 0, atol=0.14)


# @pytest.mark.xfail(reason="No reliable way to generate nan on all platforms.")
@pytest.mark.parametrize("mode", [3])
@pytest.mark.parametrize("model_name", ["collision_edge_cases"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_nan_reset(gs_sim, mode):
    
    if gs_sim.scene.rigid_solver.is_active():
        gs_sim.scene.rigid_solver.collider.use_gjk = USE_GJK
    if gs_sim.scene.avatar_solver.is_active():
        gs_sim.scene.avatar_solver.collider.use_gjk = USE_GJK
        
    for _ in range(200):
        gs_sim.scene.step()
        qvel = gs_sim.rigid_solver.get_dofs_velocity().cpu()
        if np.isnan(qvel).any():
            break
    else:
        raise AssertionError

    gs_sim.scene.reset()
    for _ in range(5):
        gs_sim.scene.step()
    qvel = gs_sim.rigid_solver.get_dofs_velocity().cpu()
    assert not np.isnan(qvel).any()


@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_terrain_generation(show_viewer):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(-5.0, -5.0, 10.0),
            camera_lookat=(5.0, 5.0, 0.0),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
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
    
    if scene.rigid_solver.is_active():
        scene.rigid_solver.collider.use_gjk = USE_GJK
    if scene.avatar_solver.is_active():
        scene.avatar_solver.collider.use_gjk = USE_GJK

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
    assert_allclose(height_balls_min, height_field_min, atol=2e-3)
    assert height_balls_max - height_balls_min > 0.5 * (height_field_max - height_field_min)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu])  # TODO: Cannot afford GPU test for this one
def test_urdf_mimic_panda(show_viewer, tol):
    # create and build the scene
    scene = gs.Scene(
        show_viewer=show_viewer,
    )

    hand = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/panda_bullet/hand.urdf",
            fixed=True,
        ),
    )
    scene.build()
    
    if scene.rigid_solver.is_active():
        scene.rigid_solver.collider.use_gjk = USE_GJK
    if scene.avatar_solver.is_active():
        scene.avatar_solver.collider.use_gjk = USE_GJK

    rigid = scene.sim.rigid_solver
    assert rigid.n_equalities == 1

    qvel = rigid.dofs_state.vel.to_numpy()
    qvel[-1] = 1
    rigid.dofs_state.vel.from_numpy(qvel)

    for i in range(200):
        scene.step()

    gs_qpos = rigid.qpos.to_numpy()[:, 0]
    assert_allclose(gs_qpos[-1], gs_qpos[-2], tol=tol)


@pytest.mark.required
@pytest.mark.xdist_group(name="huggingface_hub")
@pytest.mark.parametrize("backend", [gs.cpu])
def test_drone_advanced(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, 0.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            max_FPS=60,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    plane = scene.add_entity(gs.morphs.Plane())
    asset_path = snapshot_download(
        repo_type="dataset",
        repo_id="Genesis-Intelligence/assets",
        allow_patterns="drone_sus/*",
        max_workers=1,
    )
    drones = []
    for offset, merge_fixed_links in ((-0.3, False), (0.3, True)):
        drone = scene.add_entity(
            morph=gs.morphs.Drone(
                file=f"{asset_path}/drone_sus/drone_sus.urdf",
                merge_fixed_links=merge_fixed_links,
                pos=(0.0, offset, 1.5),
            ),
            vis_mode="collision",
            visualize_contact=True,
        )
        drones.append(drone)
    scene.build()
    
    if scene.rigid_solver.is_active():
        scene.rigid_solver.collider.use_gjk = USE_GJK
    if scene.avatar_solver.is_active():
        scene.avatar_solver.collider.use_gjk = USE_GJK

    for drone in drones:
        chain_dofs = range(6, drone.n_dofs)
        drone.set_dofs_armature(drone.get_dofs_armature(chain_dofs) + 1e-3, chain_dofs)

    for i in range(500):
        for drone in drones:
            drone.set_propellels_rpm(torch.full((4,), 50000.0))
        scene.step()
        scene.sim.rigid_solver._kernel_forward_dynamics()
        scene.sim.rigid_solver._func_constraint_force()
        if scene.rigid_solver.collider.n_contacts.to_numpy()[0] > 2:
            # FIXME: It starts to return assymetrical contact forces as soon as a third contact point in the middle
            # of the segment pops up. This discrepancy is very large on Linux and causes the simulation to diverge,
            # which is not the case on Windows OS or Mac OS.
            break
    assert 250 < i < 500

    pos_1 = drones[0].get_pos()
    pos_2 = drones[1].get_pos()
    assert abs(pos_1[0] - pos_2[0]) < gs.EPS
    assert abs(pos_1[1] + pos_2[1]) < gs.EPS
    assert abs(pos_1[2] - pos_2[2]) < gs.EPS
    quat_1 = drones[0].get_quat()
    quat_2 = drones[1].get_quat()
    assert abs(quat_1[1] + quat_2[1]) < gs.EPS
    assert abs(quat_1[2] - quat_2[2]) < gs.EPS
    assert abs(quat_1[2] - quat_2[2]) < gs.EPS


@pytest.mark.parametrize(
    "n_envs, batched, backend",
    [
        (0, False, gs.cpu),
        (0, False, gs.gpu),
        (3, False, gs.cpu),
        # (3, True, gs.cpu),  # FIXME: Must refactor the unit test to support batching
    ],
)
def test_data_accessor(n_envs, batched, tol):
    # create and build the scene
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            batch_dofs_info=batched,
            batch_joints_info=batched,
            batch_links_info=batched,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    gs_robot = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=(0.0, 0.0, 0.5),
        ),
    )
    scene.build(n_envs=n_envs)
    
    if scene.rigid_solver.is_active():
        scene.rigid_solver.collider.use_gjk = USE_GJK
    if scene.avatar_solver.is_active():
        scene.avatar_solver.collider.use_gjk = USE_GJK
        
    gs_sim = scene.sim
    gs_s = gs_sim.rigid_solver

    # Initialize the simulation
    np.random.seed(0)
    dof_bounds = gs_sim.rigid_solver.dofs_info.limit.to_torch(device="cpu")
    dof_bounds[..., :2, :] = torch.tensor((-1.0, 1.0))
    dof_bounds[..., 2, :] = torch.tensor((0.7, 1.0))
    dof_bounds[..., 3:6, :] = torch.tensor((-np.pi / 2, np.pi / 2))
    for i in range(max(n_envs, 1)):
        qpos = dof_bounds[:, 0] + (dof_bounds[:, 1] - dof_bounds[:, 0]) * np.random.rand(gs_robot.n_dofs)
        gs_robot.set_dofs_position(qpos, envs_idx=([i] if n_envs else None))

    # Simulate for a while, until they collide with something
    for _ in range(400):
        gs_sim.step()
        gs_n_contacts = gs_sim.rigid_solver.collider.n_contacts.to_torch(device="cpu")
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
            assert_allclose(qposs[i], qposs[i + 1], tol=tol)

    # Check attribute getters / setters.
    # First, without any any row or column masking:
    # * Call 'Get' -> Call 'Set' with random value -> Call 'Get'
    # * Compare first 'Get' ouput with field value
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

    for arg1_max, arg2_max, getter, setter, field in (
        (gs_s.n_links, n_envs, gs_s.get_links_pos, None, gs_s.links_state.pos),
        (gs_s.n_links, n_envs, gs_s.get_links_quat, None, gs_s.links_state.quat),
        (gs_s.n_links, n_envs, gs_s.get_links_vel, None, None),
        (gs_s.n_links, n_envs, gs_s.get_links_ang, None, gs_s.links_state.ang),
        (gs_s.n_links, n_envs, gs_s.get_links_acc, None, None),
        (gs_s.n_links, n_envs, gs_s.get_links_COM, None, gs_s.links_state.COM),
        (gs_s.n_links, n_envs, gs_s.get_links_mass_shift, gs_s.set_links_mass_shift, gs_s.links_state.mass_shift),
        (gs_s.n_links, n_envs, gs_s.get_links_COM_shift, gs_s.set_links_COM_shift, gs_s.links_state.i_pos_shift),
        (gs_s.n_links, -1, gs_s.get_links_inertial_mass, gs_s.set_links_inertial_mass, gs_s.links_info.inertial_mass),
        (gs_s.n_links, -1, gs_s.get_links_invweight, gs_s.set_links_invweight, gs_s.links_info.invweight),
        (gs_s.n_dofs, n_envs, gs_s.get_dofs_control_force, gs_s.control_dofs_force, None),
        (gs_s.n_dofs, n_envs, gs_s.get_dofs_force, None, gs_s.dofs_state.force),
        (gs_s.n_dofs, n_envs, gs_s.get_dofs_velocity, gs_s.set_dofs_velocity, gs_s.dofs_state.vel),
        (gs_s.n_dofs, n_envs, gs_s.get_dofs_position, gs_s.set_dofs_position, gs_s.dofs_state.pos),
        (gs_s.n_dofs, -1, gs_s.get_dofs_force_range, gs_s.set_dofs_force_range, gs_s.dofs_info.force_range),
        (gs_s.n_dofs, -1, gs_s.get_dofs_limit, None, gs_s.dofs_info.limit),
        (gs_s.n_dofs, -1, gs_s.get_dofs_stiffness, None, gs_s.dofs_info.stiffness),
        (gs_s.n_dofs, -1, gs_s.get_dofs_invweight, None, gs_s.dofs_info.invweight),
        (gs_s.n_dofs, -1, gs_s.get_dofs_armature, None, gs_s.dofs_info.armature),
        (gs_s.n_dofs, -1, gs_s.get_dofs_damping, None, gs_s.dofs_info.damping),
        (gs_s.n_dofs, -1, gs_s.get_dofs_kp, gs_s.set_dofs_kp, gs_s.dofs_info.kp),
        (gs_s.n_dofs, -1, gs_s.get_dofs_kv, gs_s.set_dofs_kv, gs_s.dofs_info.kv),
        (gs_s.n_geoms, n_envs, gs_s.get_geoms_pos, None, gs_s.geoms_state.pos),
        (gs_s.n_geoms, -1, gs_s.get_geoms_friction, gs_s.set_geoms_friction, gs_s.geoms_info.friction),
        (gs_s.n_qs, n_envs, gs_s.get_qpos, gs_s.set_qpos, gs_s.qpos),
        (gs_robot.n_links, n_envs, gs_robot.get_links_pos, None, None),
        (gs_robot.n_links, n_envs, gs_robot.get_links_quat, None, None),
        (gs_robot.n_links, n_envs, gs_robot.get_links_vel, None, None),
        (gs_robot.n_links, n_envs, gs_robot.get_links_ang, None, None),
        (gs_robot.n_links, n_envs, gs_robot.get_links_acc, None, None),
        (gs_robot.n_links, -1, gs_robot.get_links_inertial_mass, gs_robot.set_links_inertial_mass, None),
        (gs_robot.n_links, -1, gs_robot.get_links_invweight, gs_robot.set_links_invweight, None),
        (gs_robot.n_dofs, n_envs, gs_robot.get_dofs_control_force, None, None),
        (gs_robot.n_dofs, n_envs, gs_robot.get_dofs_force, None, None),
        (gs_robot.n_dofs, n_envs, gs_robot.get_dofs_velocity, gs_robot.set_dofs_velocity, None),
        (gs_robot.n_dofs, n_envs, gs_robot.get_dofs_position, gs_robot.set_dofs_position, None),
        (gs_robot.n_dofs, -1, gs_robot.get_dofs_force_range, gs_robot.set_dofs_force_range, None),
        (gs_robot.n_dofs, -1, gs_robot.get_dofs_limit, None, None),
        (gs_robot.n_dofs, -1, gs_robot.get_dofs_stiffness, None, None),
        (gs_robot.n_dofs, -1, gs_robot.get_dofs_invweight, None, None),
        (gs_robot.n_dofs, -1, gs_robot.get_dofs_armature, None, None),
        (gs_robot.n_dofs, -1, gs_robot.get_dofs_damping, None, None),
        (gs_robot.n_dofs, -1, gs_robot.get_dofs_kp, gs_robot.set_dofs_kp, None),
        (gs_robot.n_dofs, -1, gs_robot.get_dofs_kv, gs_robot.set_dofs_kv, None),
        (gs_robot.n_qs, n_envs, gs_robot.get_qpos, gs_robot.set_qpos, None),
        (-1, n_envs, gs_robot.get_mass_mat, None, None),
        (-1, n_envs, gs_robot.get_links_net_contact_force, None, None),
        (-1, n_envs, gs_robot.get_pos, gs_robot.set_pos, None),
        (-1, n_envs, gs_robot.get_quat, gs_robot.set_quat, None),
    ):
        # Check getter and setter without row or column masking
        datas = getter()
        datas = datas.cpu() if isinstance(datas, torch.Tensor) else [val.cpu() for val in datas]
        if field is not None:
            true = field.to_torch(device="cpu")
            true = true.movedim(true.ndim - getattr(field, "ndim", 0) - 1, 0)
            if isinstance(datas, torch.Tensor):
                true = true.reshape(datas.shape)
            else:
                true = torch.unbind(true, dim=-1)
                true = [val.reshape(data.shape) for data, val in zip(datas, true)]
            assert_allclose(datas, true, tol=tol)
        if setter is not None:
            if isinstance(datas, torch.Tensor):
                # Make sure that the vector is normalized and positive just in case it is a quaternion
                datas = torch.abs(torch.randn(datas.shape, dtype=gs.tc_float, device="cpu"))
                datas /= torch.linalg.norm(datas, dim=-1, keepdims=True)
                setter(datas)
            else:
                for val in datas:
                    val[:] = torch.abs(torch.randn(val.shape, dtype=gs.tc_float, device="cpu"))
                    val /= torch.linalg.norm(val, dim=-1, keepdims=True)
                setter(*datas)
        if arg1_max > 0:
            datas_ = getter(range(arg1_max))
            datas_ = datas_.cpu() if isinstance(datas_, torch.Tensor) else [val.cpu() for val in datas_]
            assert_allclose(datas_, datas, tol=tol)

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
                                if isinstance(data, torch.Tensor):
                                    setter(data, arg1, unsafe=unsafe)
                                else:
                                    setter(*data, arg1, unsafe=unsafe)
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
                        # FIXME: Not sure why tolerance must be increased for test to pass
                        assert_allclose(data_, data, tol=(5 * tol))

    for dofs_idx in (*get_all_supported_masks(0), None):
        for envs_idx in (*(get_all_supported_masks(0) if n_envs > 0 else ()), None):
            unsafe = not any(map(must_cast, (dofs_idx, envs_idx)))
            dofs_pos = gs_s.get_dofs_position(dofs_idx, envs_idx)
            dofs_vel = gs_s.get_dofs_velocity(dofs_idx, envs_idx)
            gs_sim.rigid_solver.control_dofs_position(dofs_pos, dofs_idx, envs_idx)
            gs_sim.rigid_solver.control_dofs_velocity(dofs_vel, dofs_idx, envs_idx)


@pytest.mark.parametrize("backend", [gs.cpu])
def test_mesh_to_heightfield(show_viewer):
    ########################## create a scene ##########################
    scene = gs.Scene(
        show_viewer=show_viewer,
        sim_options=gs.options.SimOptions(
            gravity=(2, 0, -2),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -50, 0),
            camera_lookat=(0, 0, 0),
        ),
    )

    horizontal_scale = 2.0
    gs_root = os.path.dirname(os.path.abspath(gs.__file__))
    path_terrain = os.path.join(gs_root, "assets", "meshes", "terrain_45.obj")
    hf_terrain, xs, ys = gs.utils.terrain.mesh_to_heightfield(path_terrain, spacing=horizontal_scale, oversample=1)

    # default heightfield starts at 0, 0, 0
    # translate to the center of the mesh
    translation = np.array([np.nanmin(xs), np.nanmin(ys), 0])

    terrain_heightfield = scene.add_entity(
        morph=gs.morphs.Terrain(
            horizontal_scale=horizontal_scale,
            vertical_scale=1.0,
            height_field=hf_terrain,
            pos=translation,
        ),
        vis_mode="collision",
    )

    ball = scene.add_entity(
        gs.morphs.Sphere(
            pos=(10, 15, 10),
            radius=1,
        ),
        vis_mode="collision",
    )

    scene.build()
    
    if scene.rigid_solver.is_active():
        scene.rigid_solver.collider.use_gjk = USE_GJK
    if scene.avatar_solver.is_active():
        scene.avatar_solver.collider.use_gjk = USE_GJK

    for i in range(1000):
        scene.step()

    # speed is around 0
    qvel = ball.get_dofs_velocity().cpu()
    assert_allclose(qvel, 0, atol=1e-2)

@pytest.mark.parametrize("xml_path", ["xml/tet_tet.xml"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("gjk_collision", [True])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_tet_tet(gs_sim, mj_sim, gs_solver, tol):
    # Make sure it is possible to set the configuration vector without failure
    gs_sim.rigid_solver.set_dofs_position(gs_sim.rigid_solver.get_dofs_position())
    
    check_mujoco_model_consistency(gs_sim, mj_sim, tol=tol)
    tol = 2e-7
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, num_steps=1000, tol=tol)

@pytest.mark.parametrize("xml_path", ["xml/tet_ball.xml"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("gjk_collision", [True])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_tet_ball(gs_sim, mj_sim, gs_solver, tol):
    # Make sure it is possible to set the configuration vector without failure
    gs_sim.rigid_solver.set_dofs_position(gs_sim.rigid_solver.get_dofs_position())
    
    check_mujoco_model_consistency(gs_sim, mj_sim, tol=tol)
    tol = 2e-7
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, num_steps=1000, tol=tol)

@pytest.mark.parametrize("xml_path", ["xml/tet_capsule.xml"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("gjk_collision", [True])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_tet_capsule(gs_sim, mj_sim, gs_solver, tol):
    # Make sure it is possible to set the configuration vector without failure
    gs_sim.rigid_solver.set_dofs_position(gs_sim.rigid_solver.get_dofs_position())
    
    check_mujoco_model_consistency(gs_sim, mj_sim, tol=tol)
    tol = 2e-5
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, num_steps=1000, tol=tol)

@pytest.mark.parametrize("xml_path", ["xml/tet_box.xml"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("gjk_collision", [True])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_tet_box(gs_sim, mj_sim, gs_solver, tol):
    # Make sure it is possible to set the configuration vector without failure
    gs_sim.rigid_solver.set_dofs_position(gs_sim.rigid_solver.get_dofs_position())
    
    check_mujoco_model_consistency(gs_sim, mj_sim, tol=tol)
    tol = 2e-7
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, num_steps=1000, tol=tol)
