import math
import os
import sys
import tempfile
from typing import cast
import xml.etree.ElementTree as ET
from pathlib import Path

import igl
import mujoco
import numpy as np
import pytest
import torch
import trimesh

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import get_assets_dir, tensor_to_array, ti_to_torch
from genesis.engine.entities.rigid_entity import RigidEntity

from .utils import (
    assert_allclose,
    build_genesis_sim,
    build_mujoco_sim,
    check_mujoco_data_consistency,
    check_mujoco_model_consistency,
    get_hf_dataset,
    init_simulators,
    simulate_and_check_mujoco_consistency,
)


@pytest.fixture
def xml_path(request, tmp_path, model_name):
    mjcf = request.getfixturevalue(model_name)
    xml_tree = ET.ElementTree(mjcf)
    file_name = f"{model_name}.urdf" if mjcf.tag == "robot" else f"{model_name}.xml"
    file_path = str(tmp_path / file_name)
    xml_tree.write(file_path, encoding="utf-8", xml_declaration=True)
    return file_path


@pytest.fixture(scope="session")
def box_plan():
    """Generate an MJCF model for a box on a plane."""
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
    """Generate an MJCF model for two boxes."""
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


def _build_multi_pendulum(n):
    """Generate an URDF model of a multi-link pendulum with n segments."""
    urdf = ET.Element("robot", name="multi_pendulum")

    # Base link
    ET.SubElement(urdf, "link", name="base")

    parent_link = "base"
    for i in range(n):
        # Continuous joint between parent and this arm
        joint = ET.SubElement(urdf, "joint", name=f"PendulumJoint_{i}", type="continuous")
        ET.SubElement(joint, "origin", xyz="0.0 0.0 0.0", rpy="0.0 0.0 0.0")
        ET.SubElement(joint, "axis", xyz="1 0 0")
        ET.SubElement(joint, "parent", link=parent_link)
        ET.SubElement(joint, "child", link=f"PendulumArm_{i}")
        ET.SubElement(joint, "limit", effort=str(100.0 * (n - i)), velocity="30.0")

        # Arm link
        arm = ET.SubElement(urdf, "link", name=f"PendulumArm_{i}")
        visual = ET.SubElement(arm, "visual")
        ET.SubElement(visual, "origin", xyz="0.0 0.0 0.5", rpy="0.0 0.0 0.0")
        geometry = ET.SubElement(visual, "geometry")
        ET.SubElement(geometry, "box", size="0.01 0.01 1.0")
        material = ET.SubElement(visual, "material", name="")
        ET.SubElement(material, "color", rgba="0.0 0.0 1.0 1.0")
        inertial = ET.SubElement(arm, "inertial")
        ET.SubElement(inertial, "origin", xyz="0.0 0.0 0.0", rpy="0.0 0.0 0.0")
        ET.SubElement(inertial, "mass", value="0.0")
        ET.SubElement(inertial, "inertia", ixx="0.0", ixy="0.0", ixz="0.0", iyy="0.0", iyz="0.0", izz="0.0")

        # Fixed joint to the mass
        joint2 = ET.SubElement(urdf, "joint", name=f"PendulumMassJoint_{i}", type="fixed")
        ET.SubElement(joint2, "origin", xyz="0.0 0.0 1.0", rpy="0.0 0.0 0.0")
        ET.SubElement(joint2, "parent", link=f"PendulumArm_{i}")
        ET.SubElement(joint2, "child", link=f"PendulumMass_{i}")

        # Mass link
        mass = ET.SubElement(urdf, "link", name=f"PendulumMass_{i}")
        visual = ET.SubElement(mass, "visual")
        ET.SubElement(visual, "origin", xyz="0.0 0.0 0.0", rpy="0.0 0.0 0.0")
        geometry = ET.SubElement(visual, "geometry")
        ET.SubElement(geometry, "sphere", radius="0.06")
        material = ET.SubElement(visual, "material", name="")
        ET.SubElement(material, "color", rgba="0.0 0.0 1.0 1.0")
        inertial = ET.SubElement(mass, "inertial")
        ET.SubElement(inertial, "origin", xyz="0.0 0.0 0.0", rpy="0.0 0.0 0.0")
        ET.SubElement(inertial, "mass", value="1.0")
        ET.SubElement(inertial, "inertia", ixx="1e-12", ixy="0.0", ixz="0.0", iyy="1e-12", iyz="0.0", izz="1e-12")

        parent_link = f"PendulumMass_{i}"

    return urdf


@pytest.fixture(scope="session")
def pendulum(asset_tmp_path):
    return _build_multi_pendulum(n=1)


@pytest.fixture(scope="session")
def double_pendulum(asset_tmp_path):
    return _build_multi_pendulum(n=2)


@pytest.fixture(scope="session")
def double_ball_pendulum():
    mjcf = ET.Element("mujoco", model="double_ball_pendulum")

    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "joint", armature="0.1", damping="0.5")

    worldbody = ET.SubElement(mjcf, "worldbody")
    base = ET.SubElement(worldbody, "body", name="base", pos="-0.02 0.0 0.0")
    ET.SubElement(base, "joint", name="joint1", type="ball")
    ET.SubElement(
        base, "geom", name="link1_geom", type="capsule", size="0.02", fromto="0 0 0 0 0 0.5", rgba="0.8 0.2 0.2 1.0"
    )
    link2 = ET.SubElement(base, "body", name="link2", pos="0 0 0.5")
    ET.SubElement(link2, "joint", name="joint2", type="ball")
    ET.SubElement(
        link2, "geom", name="link2_geom", type="capsule", size="0.02", fromto="0 0 0 0 0 0.3", rgba="0.2 0.8 0.2 1.0"
    )
    ee = ET.SubElement(link2, "body", name="end_effector", pos="0 0 0.3")
    ET.SubElement(ee, "geom", name="ee_geom", type="sphere", size="0.02", density="200", rgba="1.0 0.8 0.2 1.0")

    return mjcf


@pytest.fixture(scope="session")
def hinge_slide():
    mjcf = ET.Element("mujoco", model="hinge_slide")

    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "joint", damping="0.01")

    worldbody = ET.SubElement(mjcf, "worldbody")
    base = ET.SubElement(worldbody, "body", name="pendulum", pos="0.15 0.0 0.0")
    ET.SubElement(base, "joint", name="hinge", type="hinge", axis="0 1 0", frictionloss="0.08")
    ET.SubElement(base, "geom", name="geom1", type="capsule", size="0.02", fromto="0.0 0.0 0.0 0.1 0.0 0.0")
    link1 = ET.SubElement(base, "body", name="link1", pos="0.1 0.0 0.0")
    ET.SubElement(link1, "joint", name="slide", type="slide", axis="1 0 0", frictionloss="0.3", stiffness="200.0")
    ET.SubElement(link1, "geom", name="geom2", type="capsule", size="0.015", fromto="-0.1 0.0 0.0 0.1 0.0 0.0")

    return mjcf


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
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, qpos, qvel, num_steps=150, tol=tol)


@pytest.mark.required
@pytest.mark.adjacent_collision(True)
@pytest.mark.parametrize("model_name", ["chain_capsule_hinge_mesh"])  # FIXME: , "chain_capsule_hinge_capsule"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("gjk_collision", [True, False])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_simple_kinematic_chain(gs_sim, mj_sim, tol):
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, num_steps=200, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["hinge_slide"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_frictionloss(gs_sim, mj_sim, tol):
    qvel = np.array([0.7, -0.9])
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, qvel=qvel, num_steps=2000, tol=tol)

    # Check that final velocity is almost zero
    gs_qvel = gs_sim.rigid_solver.dofs_state.vel.to_numpy()
    assert_allclose(gs_qvel, 0.0, tol=1e-2)


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["hinge_slide"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_set_dofs_frictionloss_physics(gs_sim, mj_sim, tol):
    (robot,) = gs_sim.entities

    initial_velocity = np.array([1.0, 0.0])
    robot.set_dofs_velocity(initial_velocity)

    robot.set_dofs_frictionloss(np.array([0.0, 0.0]))
    for _ in range(10):
        gs_sim.step()
        mujoco.mj_step(mj_sim.model, mj_sim.data)
    velocity_zero = gs_sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]

    robot.set_dofs_velocity(initial_velocity)
    robot.set_dofs_frictionloss(np.array([1.0, 0.0]))
    for _ in range(10):
        gs_sim.step()
        mujoco.mj_step(mj_sim.model, mj_sim.data)
    velocity_high = gs_sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]

    np.testing.assert_array_less(velocity_high[0], velocity_zero[0])
    np.testing.assert_array_less(velocity_high[1], velocity_zero[1])

    robot.set_dofs_velocity(initial_velocity)
    robot.set_dofs_frictionloss(np.array([0.5]), dofs_idx_local=[0])
    for _ in range(10):
        gs_sim.step()
        mujoco.mj_step(mj_sim.model, mj_sim.data)
    velocity_medium = gs_sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]

    np.testing.assert_array_less(velocity_high[0], velocity_medium[0])
    np.testing.assert_array_less(velocity_medium[0], velocity_zero[0])

    friction_effect = velocity_zero[0] - velocity_high[0]
    np.testing.assert_array_less(tol, friction_effect)

    slide_friction_effect = velocity_zero[1] - velocity_high[1]
    np.testing.assert_array_less(tol, slide_friction_effect)


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
@pytest.mark.parametrize("gjk_collision", [True, False])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_walker(gs_sim, mj_sim, gjk_collision, tol):
    # Force numpy seed because this test is very sensitive to the initial condition
    np.random.seed(0)
    (gs_robot,) = gs_sim.entities
    qpos = np.zeros((gs_robot.n_qs,))
    qpos[2] += 0.5
    qvel = np.random.rand(gs_robot.n_dofs) * 0.2

    # Make sure it is possible to set the configuration vector without failure
    qpos = gs_robot.get_dofs_position()
    gs_robot.set_dofs_position(qpos)
    assert_allclose(gs_robot.get_dofs_position(), qpos, tol=gs.EPS)
    qpos = torch.rand(gs_robot.n_dofs).clip(*gs_robot.get_dofs_limit())
    gs_robot.set_dofs_position(qpos)
    assert_allclose(gs_robot.get_dofs_position(), qpos, tol=gs.EPS)

    # Cannot simulate any longer because collision detection is very sensitive
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, qpos, qvel, num_steps=90, tol=tol)


@pytest.mark.parametrize("model_name", ["mimic_hinges"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_equality_joint(gs_sim, mj_sim, gs_solver, tol):
    # there is an equality constraint
    assert gs_sim.rigid_solver.n_equalities == 1

    qpos = np.array((0.0, -1.0))
    qvel = np.array((1.0, -0.3))
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, qpos, qvel, num_steps=300, tol=tol)

    # check if the two joints are equal
    gs_qpos = gs_sim.rigid_solver.qpos.to_numpy()[:, 0]
    assert_allclose(gs_qpos[0], gs_qpos[1], tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("xml_path", ["xml/four_bar_linkage_weld.xml"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_equality_weld(gs_sim, mj_sim, gs_solver):
    # Must disable self-collision caused by closing the kinematic chain (adjacent link filtering is not enough)
    gs_sim.rigid_solver._enable_collision = False
    mj_sim.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

    # Must increase sol params to improve numerical stability
    sol_params = gu.default_solver_params()
    sol_params[0] = 0.02
    for entity in gs_sim.entities:
        for equality in entity.equalities:
            equality.set_sol_params(sol_params)
    mj_sim.model.eq_solref[:, 0] = sol_params[0]

    assert gs_sim.rigid_solver.n_equalities == 1
    np.random.seed(0)
    qpos = np.random.rand(gs_sim.rigid_solver.n_qs) * 0.1

    # Note that it is impossible to be more accurate than this because of the inherent stiffness of the problem.
    # The pose difference between Mujoco and Genesis (resulting from using quaternion instead of rotation matrices to
    # apply transform internally) is about 1e-15. This is fine and not surprising as it is consistent with machine
    # precision. These rounding errors are then amplified by 1e8 when computing the forces resulting from the kinematic
    # constraints. The constraints could be made softer by changing its impede parameters.
    tol = 1e-7 if gs_solver == gs.constraint_solver.Newton else 2e-5
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, qpos, num_steps=300, tol=tol)


@pytest.mark.required
def test_dynamic_weld(show_viewer, tol):
    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
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
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/universal_robots_ur5e/ur5e.xml",
        ),
    )
    scene.build(n_envs=4, env_spacing=(3.0, 3.0))

    end_effector = robot.get_link("ee_virtual_link")

    # Compute up and down robot configurations
    ee_pos_up = np.array((0.65, 0.0, 0.5), dtype=gs.np_float)
    ee_pos_down = np.array((0.65, 0.0, 0.15), dtype=gs.np_float)
    qpos_up = robot.inverse_kinematics(
        link=end_effector,
        pos=np.tile(ee_pos_up, (4, 1)),
        quat=np.tile(np.array((0.0, 1.0, 0.0, 0.0), dtype=gs.np_float), (4, 1)),
    )
    qpos_down = robot.inverse_kinematics(
        link=end_effector,
        pos=np.tile(ee_pos_down, (4, 1)),
        quat=np.tile(np.array((0.0, 1.0, 0.0, 0.0), dtype=gs.np_float), (4, 1)),
    )

    # move to pre-grasp pose
    robot.control_dofs_position(qpos_up)
    for i in range(120):
        scene.step()

    # reach
    robot.control_dofs_position(qpos_down)
    for i in range(70):
        scene.step()

    # add weld constraint and move back up
    scene.sim.rigid_solver.add_weld_constraint(cube.base_link.idx, end_effector.idx, envs_idx=(0, 1, 2))
    robot.control_dofs_position(qpos_up)
    for i in range(60):
        scene.step()
    cubes_pos, cubes_quat = cube.get_pos(), cube.get_quat()
    assert_allclose(torch.diff(cubes_quat, dim=0), 0.0, tol=1e-3)
    assert_allclose(torch.diff(cubes_pos[[0, 1, 2]], dim=0), 0.0, tol=tol)
    assert_allclose(cubes_pos[-1] - cubes_pos[0], ee_pos_down - ee_pos_up, tol=1e-2)

    # drop
    scene.sim.rigid_solver.delete_weld_constraint(cube.base_link.idx, end_effector.idx, envs_idx=(0, 1))
    for i in range(110):
        scene.step()
    cubes_pos, cubes_quat = cube.get_pos(), cube.get_quat()
    assert_allclose(torch.diff(cubes_quat, dim=0), 0.0, tol=1e-3)
    assert_allclose(torch.diff(cubes_pos[[0, 1, 3]], dim=0), 0.0, tol=1e-2)
    assert_allclose(cubes_pos[2] - cubes_pos[0], ee_pos_up - ee_pos_down, tol=1e-3)


@pytest.mark.required
@pytest.mark.parametrize("xml_path", ["xml/one_ball_joint.xml"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_one_ball_joint(gs_sim, mj_sim, tol):
    # FIXME: Mujoco is detecting collision for some reason...
    mj_sim.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, num_steps=600, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("xml_path", ["xml/rope_ball.xml", "xml/rope_hinge.xml"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("gjk_collision", [True, False])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_rope_ball(gs_sim, mj_sim, gs_solver, tol):
    # Make sure it is possible to set the configuration vector without failure
    qpos = gs_sim.rigid_solver.get_dofs_position()
    gs_sim.rigid_solver.set_dofs_position(qpos)
    assert_allclose(gs_sim.rigid_solver.get_dofs_position(), qpos, tol=gs.EPS)
    qpos = torch.rand(gs_sim.rigid_solver.n_dofs).clip(*gs_sim.rigid_solver.get_dofs_limit())
    gs_sim.rigid_solver.set_dofs_position(qpos)
    assert_allclose(gs_sim.rigid_solver.get_dofs_position(), qpos, tol=gs.EPS)

    check_mujoco_model_consistency(gs_sim, mj_sim, tol=tol)
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, num_steps=300, tol=1e-8)


@pytest.mark.required
@pytest.mark.multi_contact(False)
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast])
@pytest.mark.parametrize("gjk_collision", [True, False])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_urdf_rope(
    gs_solver,
    gs_integrator,
    merge_fixed_links,
    multi_contact,
    mujoco_compatibility,
    adjacent_collision,
    gjk_collision,
    dof_damping,
    show_viewer,
):
    asset_path = get_hf_dataset(pattern="linear_deformable.urdf")
    xml_path = os.path.join(asset_path, "linear_deformable.urdf")

    mj_sim = build_mujoco_sim(
        xml_path,
        gs_solver,
        gs_integrator,
        merge_fixed_links,
        multi_contact,
        adjacent_collision,
        dof_damping,
        gjk_collision,
    )
    gs_sim = build_genesis_sim(
        xml_path,
        gs_solver,
        gs_integrator,
        merge_fixed_links,
        multi_contact,
        mujoco_compatibility,
        adjacent_collision,
        gjk_collision,
        show_viewer,
        mj_sim,
    )

    # Must increase sol params to improve numerical stability
    sol_params = gu.default_solver_params()
    sol_params[0] = 0.02
    gs_sim.rigid_solver.set_global_sol_params(sol_params)
    mj_sim.model.jnt_solref[:, 0] = sol_params[0]
    mj_sim.model.geom_solref[:, 0] = sol_params[0]
    mj_sim.model.eq_solref[:, 0] = sol_params[0]

    # FIXME: Tolerance must be very large due to small masses and compounding of errors over long kinematic chains
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, num_steps=300, tol=5e-5)


@pytest.mark.required
@pytest.mark.mujoco_compatibility(True)
@pytest.mark.multi_contact(False)  # FIXME: Mujoco has errors with multi-contact, so this test is disabled
@pytest.mark.parametrize("xml_path", ["xml/tet_tet.xml", "xml/tet_ball.xml", "xml/tet_capsule.xml"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("gjk_collision", [True])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_tet_primitive_shapes(gs_sim, mj_sim, gs_integrator, gs_solver, xml_path, tol):
    # FIXME: Fix GsTaichi bug for ndarrays
    if os.environ.get("GS_USE_NDARRAY") == "1" and gs_solver == gs.constraint_solver.CG:
        pytest.xfail("This test is broken for ndarrays, probably due to a bug in gstaichi...")

    # Make sure it is possible to set the configuration vector without failure
    gs_sim.rigid_solver.set_dofs_position(gs_sim.rigid_solver.get_dofs_position())

    check_mujoco_model_consistency(gs_sim, mj_sim, tol=tol)
    # FIXME: Because of very small numerical error, error could be this large even if there is no logical error
    tol = 1e-6 if xml_path == "xml/tet_tet.xml" else 2e-8
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, num_steps=700, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["two_aligned_hinges"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
def test_link_velocity(gs_sim, tol):
    # Check the velocity for a few "easy" special cases
    init_simulators(gs_sim, qvel=np.array([0.0, 1.0]))
    assert_allclose(gs_sim.rigid_solver.links_state.cd_vel.to_numpy(), 0, tol=tol)

    init_simulators(gs_sim, qvel=np.array([1.0, 0.0]))
    cvel_0, cvel_1 = gs_sim.rigid_solver.links_state.cd_vel.to_numpy()[:, 0]
    assert_allclose(cvel_0, np.array([0.0, 0.5, 0.0]), tol=tol)
    assert_allclose(cvel_1, np.array([0.0, 0.5, 0.0]), tol=tol)

    init_simulators(gs_sim, qpos=np.array([0.0, np.pi / 2.0]), qvel=np.array([0.0, 1.2]))
    COM = gs_sim.rigid_solver.links_state.root_COM[0, 0]
    assert_allclose(COM, np.array([0.375, 0.125, 0.0]), tol=tol)
    xanchor = gs_sim.rigid_solver.joints_state.xanchor[1, 0]
    assert_allclose(xanchor, np.array([0.5, 0.0, 0.0]), tol=tol)
    cvel_0, cvel_1 = gs_sim.rigid_solver.links_state.cd_vel.to_numpy()[:, 0]
    assert_allclose(cvel_0, 0, tol=tol)
    assert_allclose(cvel_1, np.array([-1.2 * (0.125 - 0.0), 1.2 * (0.375 - 0.5), 0.0]), tol=tol)

    # Check that the velocity is valid for a random configuration
    init_simulators(gs_sim, qpos=np.array([-0.7, 0.2]), qvel=np.array([3.0, 13.0]))
    xanchor = gs_sim.rigid_solver.joints_state.xanchor[1, 0]
    theta_0, theta_1 = gs_sim.rigid_solver.qpos.to_numpy()[:, 0]
    assert_allclose(xanchor[0], 0.5 * np.cos(theta_0), tol=tol)
    assert_allclose(xanchor[1], 0.5 * np.sin(theta_0), tol=tol)
    COM = gs_sim.rigid_solver.links_state.root_COM[0, 0]
    COM_0 = np.array([0.25 * np.cos(theta_0), 0.25 * np.sin(theta_0), 0.0])
    COM_1 = np.array(
        [
            0.5 * np.cos(theta_0) + 0.25 * np.cos(theta_0 + theta_1),
            0.5 * np.sin(theta_0) + 0.25 * np.sin(theta_0 + theta_1),
            0.0,
        ]
    )
    link_COM0 = gs_sim.rigid_solver.get_links_pos(ref="link_com")[0]
    link_COM1 = gs_sim.rigid_solver.get_links_pos(ref="link_com")[1]

    assert_allclose(link_COM0, COM_0, tol=tol)
    assert_allclose(link_COM1, COM_1, tol=tol)
    assert_allclose(COM, 0.5 * (COM_0 + COM_1), tol=tol)

    cvel_0, cvel_1 = gs_sim.rigid_solver.links_state.cd_vel.to_numpy()[:, 0]
    omega_0, omega_1 = gs_sim.rigid_solver.links_state.cd_ang.to_numpy()[:, 0, 2]
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


@pytest.mark.required
@pytest.mark.merge_fixed_links(False)
@pytest.mark.parametrize("model_name", ["pendulum"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
def test_pendulum_links_acc(gs_sim, tol):
    pendulum = gs_sim.entities[0]
    g = gs_sim.rigid_solver._gravity[0][2]

    # Make sure that the linear and angular acceleration matches expectation
    theta = np.random.rand()
    theta_dot = np.random.rand()
    pendulum.set_qpos([theta])
    pendulum.set_dofs_velocity([theta_dot])
    for _ in range(100):
        # Backup state before integration
        theta = float(gs_sim.rigid_solver.qpos.to_numpy())
        theta_dot = float(gs_sim.rigid_solver.dofs_state.vel.to_numpy())

        # Run one simulation step
        gs_sim.scene.step()

        # Angular acceleration:
        # * acc_ang_x = - sin(theta) * g
        acc_ang = gs_sim.rigid_solver.get_links_acc_ang()
        assert_allclose(acc_ang[0], 0, tol=tol)
        assert_allclose(acc_ang[2], np.array([-np.sin(theta) * g, 0.0, 0.0]), tol=tol)
        # Linear spatial acceleration:
        # * acc_spatial_lin_y = sin(theta) * g
        acc_spatial_lin_world = gs_sim.rigid_solver.links_state.cacc_lin.to_numpy()
        assert_allclose(acc_spatial_lin_world[0], 0, tol=tol)
        R = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(theta), np.sin(theta)],
                [0.0, -np.sin(theta), np.cos(theta)],
            ]
        )
        acc_spatial_lin_local = R @ acc_spatial_lin_world[2, 0]
        assert_allclose(acc_spatial_lin_local, np.array([0.0, np.sin(theta) * g, 0.0]), tol=tol)
        # Linear true acceleration:
        # * acc_classical_lin_y = sin(theta) * g (tangential angular acceleration effect)
        # * acc_classical_lin_z = - theta_dot ** 2  (radial centripedal effect)
        acc_classical_lin_world = tensor_to_array(gs_sim.rigid_solver.get_links_acc())
        assert_allclose(acc_classical_lin_world[0], 0, tol=tol)
        acc_classical_lin_local = R @ acc_classical_lin_world[2]
        assert_allclose(acc_classical_lin_local, np.array([0.0, np.sin(theta) * g, -(theta_dot**2)]), tol=tol)

    # Hold the pendulum straight using PD controller and check again
    pendulum.set_dofs_kp([4000.0])
    pendulum.set_dofs_kv([100.0])
    pendulum.control_dofs_position([0.5 * np.pi])
    for _ in range(400):
        gs_sim.scene.step()
    acc_classical_lin_world = gs_sim.rigid_solver.get_links_acc()
    assert_allclose(acc_classical_lin_world, 0, tol=tol)


@pytest.mark.required
@pytest.mark.merge_fixed_links(False)
@pytest.mark.parametrize("model_name", ["double_pendulum"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
def test_double_pendulum_links_acc(gs_sim, tol):
    robot = gs_sim.entities[0]

    # Make sure that the linear and angular acceleration matches expectation
    qpos = np.random.rand(2)
    qvel = np.random.rand(2)
    robot.set_qpos(qpos)
    robot.set_dofs_velocity(qvel)
    for _ in range(100):
        # Backup state before integration
        theta = gs_sim.rigid_solver.qpos.to_numpy()[:, 0]
        theta_dot = gs_sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]

        # Run one simulation step
        gs_sim.scene.step()

        # Backup acceleration before integration
        theta_ddot = gs_sim.rigid_solver.dofs_state.acc.to_numpy()[:, 0]

        # Angular acceleration
        acc_ang = tensor_to_array(gs_sim.rigid_solver.get_links_acc_ang())
        assert_allclose(acc_ang[0], 0, tol=tol)
        assert_allclose(acc_ang[1], [theta_ddot[0], 0.0, 0.0], tol=tol)
        assert_allclose(acc_ang[-1], [theta_ddot[0] + theta_ddot[1], 0.0, 0.0], tol=tol)

        # Linear spatial acceleration
        cacc_spatial_lin_world = gs_sim.rigid_solver.links_state.cacc_lin.to_numpy()[[0, 2, 4], 0]
        com = gs_sim.rigid_solver.links_state.root_COM.to_numpy()[-1, 0]
        pos = gs_sim.rigid_solver.links_state.pos.to_numpy()[[0, 2, 4], 0]
        assert_allclose(cacc_spatial_lin_world[1], np.cross(acc_ang[2], com), tol=tol)
        acc_spatial_lin_world = cacc_spatial_lin_world + np.cross(acc_ang[[0, 2, 4]], pos - com)
        assert_allclose(acc_spatial_lin_world[0], 0, tol=tol)
        theta_world = theta.cumsum()
        R = np.array(
            [
                [np.ones_like(theta), np.zeros_like(theta), np.zeros_like(theta)],
                [np.zeros_like(theta), np.cos(theta_world), np.sin(theta_world)],
                [np.zeros_like(theta), -np.sin(theta_world), np.cos(theta_world)],
            ]
        )
        acc_spatial_lin_local = np.matmul(np.moveaxis(R, 2, 0), acc_spatial_lin_world[1:, :, None])[..., 0]
        assert_allclose(acc_spatial_lin_local[0], np.array([0.0, -theta_ddot[0], 0.0]), tol=tol)
        assert_allclose(
            acc_spatial_lin_local[1],
            R[..., 1] @ (R[..., 0].T @ np.array([0.0, -theta_ddot[0], theta_dot[0] * theta_dot[1]]))
            + np.array([0.0, -theta_ddot.sum(), 0.0]),
            tol=tol,
        )

        # Linear true acceleration
        acc_classical_lin_world = tensor_to_array(gs_sim.rigid_solver.get_links_acc()[[0, 2, 4]])
        assert_allclose(acc_classical_lin_world[0], 0, tol=tol)
        acc_classical_lin_local = np.matmul(np.moveaxis(R, 2, 0), acc_classical_lin_world[1:, :, None])[..., 0]
        assert_allclose(acc_classical_lin_local[0], np.array([0.0, -theta_ddot[0], -theta_dot[0] ** 2]), tol=tol)
        assert_allclose(
            acc_classical_lin_local[1],
            R[..., 1] @ acc_classical_lin_world[1] + np.array([0.0, -theta_ddot.sum(), -theta_dot.sum() ** 2]),
            tol=tol,
        )

    # Hold the double pendulum straight using PD controller and check again
    robot.set_dofs_kp([6000.0, 4000.0])
    robot.set_dofs_kv([200.0, 150.0])
    robot.control_dofs_position([0.5 * np.pi, 0.0])
    for _ in range(900):
        gs_sim.scene.step()
    acc_classical_lin_world = gs_sim.rigid_solver.get_links_acc()
    assert_allclose(acc_classical_lin_world, 0, tol=tol)


@pytest.mark.required
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
        cube2_quat = gu.xyz_to_quat(
            np.array([*(0.15 * np.random.rand(2)), np.pi * np.random.rand()]),
        )
        init_simulators(gs_sim, qpos=np.concatenate((cube1_pos, cube1_quat, cube2_pos, cube2_quat)))
        for i in range(110):
            gs_sim.scene.step()
            if i > 100:
                qvel = gs_robot.get_dofs_velocity()
                assert_allclose(qvel, 0, atol=1e-2)

        qpos = gs_robot.get_dofs_position()
        assert_allclose(qpos[8], 0.6, atol=2e-3)


@pytest.mark.parametrize(
    "box_box_detection, gjk_collision, dynamics",
    [
        (True, False, False),
        (False, False, False),
        (False, False, True),
        (False, True, False),
    ],
)
@pytest.mark.parametrize("backend", [gs.cpu])  # TODO: Cannot afford GPU test for this one
def test_many_boxes_dynamics(box_box_detection, gjk_collision, dynamics, show_viewer):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            box_box_detection=box_box_detection,
            max_collision_pairs=1000,
            use_gjk_collision=gjk_collision,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(10, 10, 10),
            camera_lookat=(0.0, 0.0, 0.0),
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
    num_steps = 650 if dynamics else 150
    for i in range(num_steps):
        scene.step()
        if i > num_steps - 50:
            qvel = scene.rigid_solver.get_dofs_velocity().reshape((6, -1))
            # Checking the average velocity because is always one cube moving depending on the machine.
            assert_allclose(torch.linalg.norm(qvel, dim=0).mean(), 0, atol=0.05)

    for n, entity in enumerate(scene.entities[1:]):
        i, j, k = int(n / 25), int(n / 5) % 5, n % 5
        qpos = entity.get_dofs_position()
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

    check_mujoco_model_consistency(gs_sim, mj_sim, tol=tol)

    (gs_robot,) = gs_sim.entities
    dof_bounds = gs_sim.rigid_solver.dofs_info.limit.to_numpy()
    for _ in range(100):
        qpos = dof_bounds[:, 0] + (dof_bounds[:, 1] - dof_bounds[:, 0]) * np.random.rand(gs_robot.n_qs)
        init_simulators(gs_sim, mj_sim, qpos)
        check_mujoco_data_consistency(gs_sim, mj_sim, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("xml_path", ["xml/franka_emika_panda/panda.xml", "urdf/go2/urdf/go2.urdf"])
def test_robot_scale_and_dofs_armature(xml_path, tol):
    ROBOT_SCALES = (1.0, 0.2, 5.0)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0, 0, -10.0),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
        ),
        show_viewer=False,
        show_FPS=False,
    )
    for i, scale in enumerate(ROBOT_SCALES):
        morph_kwargs = dict(file=xml_path, scale=scale)
        if xml_path.endswith(".xml"):
            morph = gs.morphs.MJCF(**morph_kwargs)
        else:
            morph = gs.morphs.URDF(**morph_kwargs)
        scene.add_entity(morph)
    scene.build()

    # Disable armature because it messes up with the mass matrix.
    # It is also a good opportunity to check that it updates 'invweight' and meaninertia accordingly.
    attr_orig = {}
    for scale, robot in zip(ROBOT_SCALES, scene.entities):
        links_invweight = robot.get_links_invweight()
        dofs_invweight = robot.get_dofs_invweight()
        robot.set_dofs_armature(torch.ones((robot.n_dofs,), dtype=gs.tc_float, device=gs.device))
        assert torch.all(robot.get_dofs_invweight() < 1.0)
        with pytest.raises(AssertionError):
            assert_allclose(robot.get_dofs_invweight(), dofs_invweight, tol=tol)
        with pytest.raises(AssertionError):
            assert_allclose(robot.get_links_invweight(), links_invweight, tol=tol)
        robot.set_dofs_armature(torch.zeros((robot.n_dofs,), dtype=gs.tc_float, device=gs.device))
        links_invweight = robot.get_links_invweight()
        dofs_invweight = robot.get_dofs_invweight()
        qpos = np.random.rand(robot.n_dofs)
        robot.set_dofs_position(qpos)
        robot.set_dofs_armature(torch.zeros((robot.n_dofs,), dtype=gs.tc_float, device=gs.device))
        assert_allclose(robot.get_dofs_invweight(), dofs_invweight, tol=gs.EPS)
        assert_allclose(robot.get_links_invweight(), links_invweight, tol=gs.EPS)
        scene.reset()
        assert_allclose(robot.get_dofs_invweight(), dofs_invweight, tol=gs.EPS)
        assert_allclose(robot.get_links_invweight(), links_invweight, tol=gs.EPS)

        mass = robot.get_mass() / scale**3
        attr_orig.setdefault("mass", mass)
        assert_allclose(mass, attr_orig["mass"], tol=tol)

        inertia = np.stack([link.inertial_i for link in robot.links], axis=0) / scale**5
        attr_orig.setdefault("inertia", inertia)
        assert_allclose(inertia, attr_orig["inertia"], tol=tol)

        joint_pos = np.stack([joint.pos for joint in robot.joints], axis=0) / scale
        attr_orig.setdefault("joint_pos", joint_pos)
        assert_allclose(joint_pos, attr_orig["joint_pos"], tol=tol)

        links_pos = robot.get_links_pos() / scale
        attr_orig.setdefault("links_pos", links_pos)
        assert_allclose(links_pos, attr_orig["links_pos"], tol=tol)

        # Check that links and dofs invweight are approximately valid.
        # Note that assessing whether the value is truly correct would be quite tricky.
        # FIXME: The tolerance must be very high when using 32bits precision. This means that our computation of the
        # inverse mass matrix has poor numerical robustness due to ill conditioning of the mass matrix. This is
        # concerning as it would impact the numerical stability of constraint solving, and by extension of the entire
        # rigid body dynamics.
        tol_ = tol if gs.backend == gs.cpu else 2e-3
        attr_orig.setdefault("links_invweight", links_invweight)
        attr_orig.setdefault("dofs_invweight", dofs_invweight)
        if scale > 1.0:
            scale_ratio_min, scale_ratio_max = scale**3, scale**5
        else:
            scale_ratio_min, scale_ratio_max = scale**5, scale**3
        assert torch.all(scale_ratio_min * links_invweight - tol_ < attr_orig["links_invweight"])
        assert torch.all(attr_orig["links_invweight"] < scale_ratio_max * links_invweight + tol_)
        dofs_invweight = robot.get_dofs_invweight()
        assert torch.all(scale_ratio_min * dofs_invweight - tol_ < attr_orig["dofs_invweight"])
        assert torch.all(attr_orig["dofs_invweight"] < scale_ratio_max * dofs_invweight + tol_)

    # Make sure that we are scaling bounds properly for linear joints
    # TODO: None of the robots being tested for now have linear joints...
    # TODO: Scaling of bounds depending on the type of joint should be explicitly checked.
    for robot in scene.entities:
        dofs_lower_bound, dofs_upper_bound = robot.get_dofs_limit()
        robot.set_dofs_position(dofs_lower_bound)
    scene.step()
    qf_passive = scene.rigid_solver.dofs_state.qf_passive.to_numpy()
    assert_allclose(qf_passive, 0.0, tol=tol)


@pytest.mark.required
def test_robot_scaling_primitive_collision(show_viewer):
    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    asset_path = get_hf_dataset(pattern="cross.xml")
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file=f"{asset_path}/cross.xml",
            scale=0.5,
        ),
        vis_mode="collision",
    )
    scene.build()

    robot.set_qpos([0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 1.0])
    for _ in range(50):
        scene.step()

    # Robot not moving anymore
    assert_allclose(robot.get_links_vel(), 0.0, atol=5e-3)

    # Robot in contact with the ground
    robot_min_corner, _ = robot.get_AABB()
    assert_allclose(robot_min_corner[2], 0.0, tol=1e-3)


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

    scene.step()
    qposs = robot.get_qpos()
    assert_allclose(qposs[0], qposs[1], tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu])
def test_pd_control(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            substeps=1,  # This is essential to be able to emulate native PD control
        ),
        rigid_options=gs.options.RigidOptions(
            batch_links_info=True,
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

    robot.set_dofs_kp(torch.zeros_like(MOTORS_KP), envs_idx=0)
    robot.set_dofs_kv(torch.zeros_like(MOTORS_KD), envs_idx=0)
    with pytest.raises(gs.GenesisException):
        robot.control_dofs_position(MOTORS_POS_TARGET, envs_idx=0)
    with pytest.raises(gs.GenesisException):
        robot.control_dofs_velocity(torch.zeros_like(MOTORS_POS_TARGET), envs_idx=0)
    robot.set_dofs_kv(MOTORS_KD, envs_idx=0)
    robot.control_dofs_position(MOTORS_POS_TARGET, envs_idx=0)
    robot.set_dofs_kp(MOTORS_KP, envs_idx=0)

    # Must update DoF armature to emulate implicit damping for force control.
    # This is equivalent to the first-order correction term involved in implicit integration scheme,
    # in the particular case where `approximate_implicitfast` integrator is used.
    # Note that the low-level internal API is used because invweights must NOT be updated, otherwise
    # the test cannot pass. This is unecessary and not recommended for practical applications.
    # robot.set_dofs_armature(robot.get_dofs_armature(envs_idx=1) + MOTORS_KD * scene.sim._substep_dt, envs_idx=1)
    dofs_armature = scene.rigid_solver.dofs_info.armature.to_numpy()
    dofs_armature[:, 1] += tensor_to_array(MOTORS_KD * scene.sim._substep_dt)
    scene.rigid_solver.dofs_info.armature.from_numpy(dofs_armature)

    for i in range(1000):
        dofs_pos = robot.get_qpos(envs_idx=1)
        dofs_vel = robot.get_dofs_velocity(envs_idx=1)
        dofs_torque = MOTORS_KP * (MOTORS_POS_TARGET - dofs_pos) - MOTORS_KD * dofs_vel
        robot.control_dofs_force(dofs_torque, envs_idx=1)
        scene.step()
        qf_applied = scene.rigid_solver.dofs_state.qf_applied.to_numpy().T
        # dofs_torque = robot.get_dofs_control_force()
        assert_allclose(qf_applied[0], qf_applied[1], tol=1e-6)


@pytest.mark.required
@pytest.mark.parametrize("relative", [False, True])
def test_set_root_pose(relative, show_viewer, tol):
    ROBOT_POS_ZERO = (0.0, 0.4, 0.1)
    ROBOT_EULER_ZERO = (0.0, 0.0, 90.0)
    CUBE_POS_ZERO = (0.65, 0.0, 0.02)
    CUBE_EULER_ZERO = (0.0, 90.0, 0.0)

    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=ROBOT_POS_ZERO,
            euler=ROBOT_EULER_ZERO,
        ),
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=CUBE_POS_ZERO,
            euler=CUBE_EULER_ZERO,
        ),
    )
    scene.build()

    robot_aabb_init, robot_base_aabb_init = robot.get_AABB(), robot.geoms[0].get_AABB()
    cube_aabb_init, cube_base_aabb_init = cube.get_AABB(), cube.geoms[0].get_AABB()

    for _ in range(2):
        scene.reset()

        for entity, pos_zero, euler_zero, entity_aabb_init, base_aabb_init in (
            (robot, ROBOT_POS_ZERO, ROBOT_EULER_ZERO, robot_aabb_init, robot_base_aabb_init),
            (cube, CUBE_POS_ZERO, CUBE_EULER_ZERO, cube_aabb_init, cube_base_aabb_init),
        ):
            pos_zero = torch.tensor(pos_zero, device=gs.device, dtype=gs.tc_float)
            euler_zero = torch.deg2rad(torch.tensor(euler_zero, dtype=gs.tc_float))
            assert_allclose(entity.get_pos(), pos_zero, tol=tol)
            euler = gu.quat_to_xyz(entity.get_quat(), rpy=True)
            assert_allclose(euler, euler_zero, tol=5e-4)
            assert_allclose(entity.geoms[0].get_AABB(), base_aabb_init, tol=tol)
            assert_allclose(entity.get_AABB(), entity_aabb_init, tol=tol)

            pos_delta = torch.as_tensor(np.random.rand(3), dtype=gs.tc_float, device=gs.device)
            entity.set_pos(pos_delta, relative=relative)
            pos_ref = pos_delta + pos_zero if relative else pos_delta
            assert_allclose(entity.get_pos(), pos_ref, tol=tol)
            assert_allclose(entity.geoms[0].get_AABB(), base_aabb_init + (pos_ref - pos_zero), tol=tol)
            assert_allclose(entity.get_AABB(), entity_aabb_init + (pos_ref - pos_zero), tol=tol)

            quat_delta = torch.as_tensor(np.random.rand(4), dtype=gs.tc_float, device=gs.device)
            quat_delta /= torch.linalg.norm(quat_delta)
            entity.set_quat(quat_delta, relative=relative)
            euler = gu.quat_to_xyz(entity.get_quat(), rpy=True)
            quat_zero = gu.xyz_to_quat(euler_zero, rpy=True)
            if relative:
                quat_ref = gu.transform_quat_by_quat(quat_zero, quat_delta)
            else:
                quat_ref = quat_delta
            euler_ref = gu.quat_to_xyz(quat_ref, rpy=True)
            assert_allclose(euler, euler_ref, tol=tol)


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
@pytest.mark.mujoco_compatibility(False)
@pytest.mark.parametrize("xml_path", ["xml/humanoid.xml"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
@pytest.mark.parametrize("gjk_collision", [True])
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_stickman(gs_sim, mj_sim, tol):
    # Make sure that "static" model information are matching
    check_mujoco_model_consistency(gs_sim, mj_sim, tol=tol)

    # Initialize the simulation
    init_simulators(gs_sim)

    # Run the simulation for a few steps
    qvel_norminf_all = []
    for i in range(6000):
        gs_sim.scene.step()
        if i > 4000:
            (gs_robot,) = gs_sim.entities
            qvel = gs_robot.get_dofs_velocity()
            qvel_norminf = torch.linalg.norm(qvel, ord=math.inf)
            qvel_norminf_all.append(qvel_norminf)
    np.testing.assert_array_less(torch.median(torch.stack(qvel_norminf_all, dim=0)).cpu(), 0.1)

    qpos = gs_robot.get_dofs_position()
    assert torch.linalg.norm(qpos[:2]) < 1.3
    body_z = gs_sim.rigid_solver.links_state.pos.to_numpy()[:-1, 0, 2]
    np.testing.assert_array_less(0, body_z + gs.EPS)


@pytest.mark.required
@pytest.mark.field_only
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_multilink_inverse_kinematics(show_viewer):
    TOL = 1e-5

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, 0.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=(1,),
        ),
        show_viewer=show_viewer,
    )
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/shadow_hand/shadow_hand.urdf",
        ),
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(0.0, 0.2, 0.05),
        ),
    )
    scene.build(n_envs=2)
    scene.reset()

    index_finger_distal = robot.get_link("index_finger_distal")
    middle_finger_distal = robot.get_link("middle_finger_distal")
    wrist = robot.get_link("wrist")
    index_finger_pos = np.array([[0.6, 0.5, 0.2]])
    middle_finger_pos = np.array([[0.63, 0.5, 0.2]])
    wrist_pos = index_finger_pos - np.array([[0.0, 0.0, 0.2]])

    qpos, err = robot.inverse_kinematics_multilink(
        links=(index_finger_distal, middle_finger_distal, wrist),
        poss=(index_finger_pos, middle_finger_pos, wrist_pos),
        envs_idx=(1,),
        pos_tol=TOL,
        rot_tol=TOL,
        return_error=True,
    )
    assert qpos.shape == (1, robot.n_qs)
    assert err.shape == (1, 3, 6)
    assert err.abs().max() < TOL
    if show_viewer:
        robot.set_qpos(qpos, envs_idx=(1,))
        scene.visualizer.update()

    links_pos, links_quat = robot.forward_kinematics(qpos, envs_idx=(1,))
    assert_allclose(links_pos[:, index_finger_distal.idx], index_finger_pos, tol=TOL)
    assert_allclose(links_pos[:, middle_finger_distal.idx], middle_finger_pos, tol=TOL)
    assert_allclose(links_pos[:, wrist.idx], wrist_pos, tol=TOL)

    robot.set_qpos(qpos, envs_idx=(1,))
    scene.rigid_solver._func_forward_kinematics_entity(
        i_e=robot.idx, envs_idx=torch.tensor((1,), dtype=gs.tc_int, device=gs.device)
    )
    assert_allclose(index_finger_distal.get_pos(envs_idx=(1,)), index_finger_pos, tol=TOL)
    assert_allclose(middle_finger_distal.get_pos(envs_idx=(1,)), middle_finger_pos, tol=TOL)
    assert_allclose(wrist.get_pos(envs_idx=(1,)), wrist_pos, tol=TOL)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_path_planning_avoidance(backend, n_envs, show_viewer, tol):
    CUBE_SIZE = 0.07

    # FIXME: Implement a more robust plan planning algorithm
    if sys.platform == "darwin" and backend == gs.gpu:
        pytest.skip(reason="This algorithm is very fragile and fail to converge on MacOS.")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, 1, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    cubes = []
    for pos_x in (-0.15, 0.15):
        for y_i in range(-3, 3):
            cube = scene.add_entity(
                gs.morphs.Box(
                    size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
                    pos=(pos_x, CUBE_SIZE * y_i, 0.75),
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
    scene.build(n_envs=n_envs)
    collider_state = scene.rigid_solver.collider._collider_state

    hand = franka.get_link("hand")
    hand_pos_ref = torch.tensor([0.3, 0.1, 0.1], dtype=gs.tc_float, device=gs.device)
    hand_quat_ref = torch.tensor([0.3073, 0.5303, 0.7245, -0.2819], dtype=gs.tc_float, device=gs.device)
    if n_envs > 0:
        hand_pos_ref = hand_pos_ref.repeat((n_envs, 1))
        hand_quat_ref = hand_quat_ref.repeat((n_envs, 1))
    qpos_goal = franka.inverse_kinematics(hand, pos=hand_pos_ref, quat=hand_quat_ref)
    qpos_goal[..., -2:] = 0.04
    franka.set_qpos(qpos_goal)
    scene.visualizer.update()
    scene.rigid_solver.collider.detection()
    assert not collider_state.n_contacts.to_numpy().any()
    franka.set_qpos(torch.zeros_like(qpos_goal))

    free_path, return_valid_mask = franka.plan_path(
        qpos_goal=qpos_goal,
        num_waypoints=300,
        resolution=0.05,
        ignore_collision=True,
        return_valid_mask=True,
    )
    assert return_valid_mask.all()
    assert_allclose(free_path[0], 0.0, tol=tol)
    assert_allclose(free_path[-1], qpos_goal, tol=tol)

    avoidance_path, return_valid_mask = franka.plan_path(
        qpos_goal=qpos_goal,
        num_waypoints=300,
        ignore_collision=False,
        return_valid_mask=True,
        resolution=0.05,
        max_nodes=4000,
        max_retry=40,
    )
    assert return_valid_mask.all()
    assert_allclose(avoidance_path[0], 0.0, tol=tol)
    assert_allclose(avoidance_path[-1], qpos_goal, tol=tol)

    for path, avoid_collision in ((free_path, False), (avoidance_path, True)):
        max_penetration = float("-inf")
        for waypoint in path:
            franka.set_qpos(waypoint)
            scene.visualizer.update()

            # Check if the cube is colliding with the robot
            scene.rigid_solver.collider.detection()
            n_contacts = collider_state.n_contacts.to_numpy()
            for i_b in range(max(scene.n_envs, 1)):
                for i_c in range(n_contacts[i_b]):
                    contact_link_a = collider_state.contact_data.link_a[i_c, i_b]
                    contact_link_b = collider_state.contact_data.link_b[i_c, i_b]
                    penetration = collider_state.contact_data.penetration[i_c, i_b]
                    if any(i_g < len(cubes) for i_g in (contact_link_a, contact_link_b)):
                        max_penetration = max(max_penetration, penetration)

        args = (max_penetration, 5e-3)
        np.testing.assert_array_less(*(args if avoid_collision else args[::-1]))

        assert_allclose(hand_pos_ref, hand.get_pos(), tol=5e-4)
        hand_quat_diff = gu.transform_quat_by_quat(gu.inv_quat(hand_quat_ref), hand.get_quat())
        theta = 2 * torch.arctan2(torch.linalg.norm(hand_quat_diff[..., 1:]), torch.abs(hand_quat_diff[..., 0]))
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
    scene.step()

    assert_allclose(cube.get_pos(), 0, tol=gs.EPS)
    assert_allclose(cube.get_quat(), (1.0, 0.0, 0.0, 0.0), tol=gs.EPS)
    assert_allclose(cube.get_vel(), 0, tol=gs.EPS)
    assert_allclose(cube.get_ang(), 0, tol=gs.EPS)
    assert_allclose(scene.rigid_solver.get_links_acc(), 0, tol=gs.EPS)


@pytest.mark.required
def test_contact_forces(show_viewer, tol):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            box_box_detection=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.65, 0.0, 0.02),
        ),
        visualize_contact=True,
    )
    scene.build()

    cube_weight = scene.rigid_solver._gravity[0][2] * cube.get_mass()
    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    qpos = np.array([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04])
    franka.set_qpos(qpos)
    scene.step()

    end_effector = franka.get_link("hand")
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.135]),
        quat=np.array([0, 1, 0, 0]),
    )
    franka.control_dofs_position(qpos[:-2], motors_dof)

    # hold
    for i in range(50):
        scene.step()
    contact_forces = cube.get_links_net_contact_force()
    assert_allclose(contact_forces[0], [0.0, 0.0, -cube_weight], atol=1e-5)

    # grasp
    for i in range(20):
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(np.array([0.0, 0.0]), fingers_dof)
        scene.step()

    # lift
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.3]),
        quat=np.array([0, 1, 0, 0]),
    )
    for i in range(200):
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(np.array([0.0, 0.0]), fingers_dof)
        scene.step()
    contact_forces = cube.get_links_net_contact_force()
    assert_allclose(contact_forces[0], [0.0, 0.0, -cube_weight], atol=5e-5)


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["double_ball_pendulum"])
def test_apply_external_forces(xml_path, show_viewer):
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -3.5, 2.5),
            camera_lookat=(0.0, 0.0, 1.0),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file=xml_path,
            quat=(1.0, 0, 1.0, 0),
        ),
    )
    scene.build()

    tol = 5e-3
    end_effector_link_idx = robot.links[-1].idx
    for step in range(801):
        ee_pos = scene.rigid_solver.get_links_pos([end_effector_link_idx])[0]
        if step == 0:
            assert_allclose(ee_pos, [0.8, 0.0, 0.02], tol=tol)
        elif step == 600:
            assert_allclose(ee_pos, [0.0, 0.0, 0.82], tol=tol)
        elif step == 800:
            assert_allclose(ee_pos, [-0.8 / math.sqrt(2), 0.8 / math.sqrt(2), 0.02], tol=tol)

        if step >= 600:
            force = np.array([[-5.0, 5.0, 0.0]])
        elif step >= 100:
            force = np.array([[0.0, 0.0, 10.0]])
        else:
            force = np.array([[0.0, 0.0, 0.0]])

        scene.rigid_solver.apply_links_external_force(force=force, links_idx=[end_effector_link_idx])
        scene.step()


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

    mass_mat_1 = franka1.get_mass_mat(decompose=False)
    mass_mat_2 = franka2.get_mass_mat(decompose=False)
    assert mass_mat_1.shape == (franka1.n_dofs, franka1.n_dofs)
    assert_allclose(mass_mat_1, mass_mat_2, tol=tol)

    mass_mat_L, mass_mat_D_inv = franka1.get_mass_mat(decompose=True)
    mass_mat = mass_mat_L.T @ torch.diag(1.0 / mass_mat_D_inv) @ mass_mat_L
    assert_allclose(mass_mat, mass_mat_1, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu])
def test_frictionloss_advanced(show_viewer, tol):
    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(gs.morphs.Plane())
    asset_path = get_hf_dataset(pattern="SO101/*")
    robot = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=f"{asset_path}/SO101/so101_new_calib.xml",
        ),
    )
    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.025, 0.025, 0.025),
        ),
    )
    scene.build()

    scene.reset()
    box.set_pos(torch.tensor((0.1, 0.0, 1.0), dtype=gs.tc_float, device=gs.device))
    for _ in range(200):
        scene.step()

    assert_allclose(robot.get_contacts()["position"][:, 2].min(), 0.0, tol=1e-4)
    # assert_allclose(robot.get_AABB()[2], 0.0, tol=1e-3)
    assert_allclose(box.get_dofs_velocity(), 0.0, tol=tol)


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

    # Force numpy seed because this test is very sensitive to the initial condition
    np.random.seed(0)
    ball.set_dofs_velocity(np.random.rand(ball.n_dofs) * 0.8)
    for i in range(1800):
        scene.step()
        if i > 1700:
            qvel = scene.sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]
            assert_allclose(qvel, 0, atol=0.65)


@pytest.mark.required
@pytest.mark.parametrize("convexify", [True, False])
@pytest.mark.parametrize("gjk_collision", [True, False])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_mesh_repair(convexify, show_viewer, gjk_collision):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.004,
        ),
        rigid_options=gs.options.RigidOptions(
            use_gjk_collision=gjk_collision,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    asset_path = get_hf_dataset(pattern="work_table.glb")
    table = scene.add_entity(
        gs.morphs.Mesh(
            file=f"{asset_path}/work_table.glb",
            pos=(0.4, 0.0, -0.54),
            fixed=True,
        ),
        vis_mode="collision",
    )
    asset_path = get_hf_dataset(pattern="spoon.glb")
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

    for geom in obj.geoms:
        assert ("decomposed" in geom.metadata) ^ (not convexify)
        max_faces = obj._morph.decimate_face_num if convexify else 5000
        num_faces = geom.face_end - geom.face_start
        assert num_faces <= max_faces
        assert ("convexified" in geom.metadata) ^ (not convexify)

    # MPR collision detection is less reliable than SDF and GJK in terms of penetration depth estimation
    is_mpr = convexify and not gjk_collision
    tol_pos = 0.05 if is_mpr else 0.01
    tol_rot = 1.1 if is_mpr else 0.4
    for i in range(450):
        scene.step()
        if i > 350:
            qvel = obj.get_dofs_velocity()
            assert_allclose(qvel[:3], 0, atol=tol_pos)
            assert_allclose(qvel[3:], 0, atol=tol_rot)
    qpos = obj.get_dofs_position()
    assert_allclose(qpos[:2], (0.3, 0.0), atol=2e-3)


@pytest.mark.required
@pytest.mark.parametrize("euler", [(90, 0, 90), (74, 15, 90)])
@pytest.mark.parametrize("gjk_collision", [True, False])
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_convexify(euler, backend, show_viewer, gjk_collision):
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
        asset_path = get_hf_dataset(pattern=f"{asset_name}/*")
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
    gs_sim = scene.sim

    # Make sure that all the geometries in the scene are convex
    assert gs_sim.rigid_solver.geoms_info.is_convex.to_numpy().all()
    assert not gs_sim.rigid_solver.collider._collider_static_config.has_nonconvex_nonterrain

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
    qvel_norminf_all = []
    for i in range(1700):
        scene.step()
        # cam.render()
        if i > 1600:
            qvel = gs_sim.rigid_solver.get_dofs_velocity()
            qvel_norminf = torch.linalg.norm(qvel, ord=math.inf)
            qvel_norminf_all.append(qvel_norminf)
    np.testing.assert_array_less(torch.median(torch.stack(qvel_norminf_all, dim=0)).cpu(), 4.0)
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
            qpos = obj.get_dofs_position()
            assert_allclose(qpos[0], OBJ_OFFSET_X * (1.5 - i), atol=7e-3)
            assert_allclose(qpos[1], OBJ_OFFSET_Y * (i - 1.5), atol=5e-3)


@pytest.mark.required
@pytest.mark.mujoco_compatibility(False)
@pytest.mark.parametrize("mode", range(9))
@pytest.mark.parametrize("model_name", ["collision_edge_cases"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
@pytest.mark.parametrize("gjk_collision", [True, False])
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_collision_edge_cases(gs_sim, mode, gjk_collision):
    qpos_0 = gs_sim.rigid_solver.get_dofs_position()
    for _ in range(200):
        gs_sim.scene.step()

    qvel = gs_sim.rigid_solver.get_dofs_velocity()
    assert_allclose(qvel, 0, atol=1e-2)
    qpos = gs_sim.rigid_solver.get_dofs_position()
    atol = 1e-3 if gjk_collision and mode in (4, 6) else 1e-4
    assert_allclose(qpos[[0, 1, 3, 4, 5]], qpos_0[[0, 1, 3, 4, 5]], atol=atol)


@pytest.mark.required
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
            ),
            show_viewer=show_viewer,
            show_FPS=False,
        )

        scene.add_entity(morph)

        asset_path = get_hf_dataset(pattern="image_0000_segmented.glb")
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

        for i in range(500):
            scene.step()
            if i > 400:
                qvel = asset.get_dofs_velocity()
                assert_allclose(qvel, 0, atol=0.14)


@pytest.mark.required
@pytest.mark.xfail(reason="No reliable way to generate nan...")
@pytest.mark.parametrize("mode", [3])
@pytest.mark.parametrize("model_name", ["collision_edge_cases"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_nan_reset(gs_sim, mode):
    for _ in range(200):
        gs_sim.scene.step()
        qvel = gs_sim.rigid_solver.get_dofs_velocity()
        if torch.isnan(qvel).any():
            break
    else:
        raise AssertionError

    gs_sim.scene.reset()
    for _ in range(5):
        gs_sim.scene.step()
    qvel = gs_sim.rigid_solver.get_dofs_velocity()
    assert not torch.isnan(qvel).any()


@pytest.mark.required
@pytest.mark.parametrize("offset", [(0.0, 0.0, 0.0), (10.0, -10.0, -1.0)])
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_terrain_generation(offset, show_viewer):
    TERRAIN_PATTERN = [
        ["flat_terrain", "flat_terrain", "flat_terrain", "flat_terrain", "flat_terrain"],
        ["flat_terrain", "fractal_terrain", "random_uniform_terrain", "sloped_terrain", "flat_terrain"],
        ["flat_terrain", "pyramid_sloped_terrain", "discrete_obstacles_terrain", "wave_terrain", "flat_terrain"],
        ["flat_terrain", "stairs_terrain", "pyramid_stairs_terrain", "stepping_stones_terrain", "flat_terrain"],
        ["flat_terrain", "flat_terrain", "flat_terrain", "flat_terrain", "flat_terrain"],
    ]
    TERRAIN_SIZE = 10.0
    SUBTERRAIN_GRID_SIZE = 15
    OBJ_SIZE = 0.1
    OBJ_HEIGHT_INIT = 0.3
    NUM_OBJ_SQRT = 15

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            dt=0.006,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(-5.0 + offset[0], -5.0 + offset[1], 10.0 + offset[2]),
            camera_lookat=(5.0 + offset[0], 5.0 + offset[1], 0.0 + offset[2]),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    terrain = scene.add_entity(
        morph=gs.morphs.Terrain(
            pos=offset,
            n_subterrains=(len(TERRAIN_PATTERN),) * 2,
            subterrain_size=(TERRAIN_SIZE / len(TERRAIN_PATTERN),) * 2,
            horizontal_scale=TERRAIN_SIZE / len(TERRAIN_PATTERN) / SUBTERRAIN_GRID_SIZE,
            vertical_scale=0.05,
            subterrain_types=TERRAIN_PATTERN,
        ),
    )
    obj = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(1.0, 1.0, 1.0),
            size=(0.1, 0.1, 0.1),
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.0, 0.0, 1.0),
        ),
    )
    scene.build(n_envs=NUM_OBJ_SQRT**2)

    # Spread objects across the entire field
    obj_pos_1d = torch.linspace(OBJ_SIZE / 2, TERRAIN_SIZE - OBJ_SIZE / 2, NUM_OBJ_SQRT)
    obj_pos_init_rel = torch.cartesian_prod(*(obj_pos_1d,) * 2, torch.tensor((OBJ_HEIGHT_INIT,)))
    obj.set_pos(obj_pos_init_rel + torch.tensor(offset))

    # Drop the objects and simulate for a while
    for _ in range(500):
        scene.step()

    # Check that objects are not moving anymore
    assert_allclose(obj.get_vel(), 0.0, tol=0.05)

    # Check that all objects are in contact with the terrain
    obj_pos = tensor_to_array(obj.get_pos()) - offset
    terrain_mesh = terrain.geoms[0].mesh
    signed_distance, *_ = igl.signed_distance(obj_pos, terrain_mesh.verts, terrain_mesh.faces)
    assert (signed_distance > 0.0).all()
    assert (signed_distance < 2 * OBJ_SIZE).all()


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu])
def test_terrain_size(show_viewer, tol):
    scene_ref = gs.Scene(show_viewer=show_viewer)
    terrain_ref = scene_ref.add_entity(
        morph=gs.morphs.Terrain(
            n_subterrains=(2, 2),
            subterrain_size=(12.0, 12.0),
            horizontal_scale=0.25,
            subterrain_types="wave_terrain",
        )
    )

    height_ref = terrain_ref.geoms[0].metadata["height_field"]

    scene_test = gs.Scene(show_viewer=show_viewer)
    terrain_test = scene_test.add_entity(
        morph=gs.morphs.Terrain(
            n_subterrains=(2, 2),
            subterrain_size=(12.0, 12.0),
            horizontal_scale=0.25,
            subterrain_types="wave_terrain",
            subterrain_parameters={"wave_terrain": {"amplitude": 0.2}},
        )
    )

    height_test = terrain_test.geoms[0].metadata["height_field"]

    assert_allclose((height_ref * 2.0), height_test, tol=tol)


@pytest.mark.required
@pytest.mark.merge_fixed_links(False)
@pytest.mark.parametrize("model_name", ["pendulum"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
def test_jacobian(gs_sim, tol):
    pendulum = cast(RigidEntity, gs_sim.entities[0])
    angle = 0.7
    pendulum.set_qpos(np.array([angle], dtype=gs.np_float))
    gs_sim.scene.step()

    link = pendulum.get_link("PendulumArm_0")

    p_local = np.array([0.05, -0.02, 0.12], dtype=gs.np_float)
    J_o = tensor_to_array(pendulum.get_jacobian(link))
    J_p = tensor_to_array(pendulum.get_jacobian(link, p_local))

    c, s = np.cos(angle), np.sin(angle)
    Rx = np.array(
        [
            [1, 0, 0],
            [0, c, -s],
            [0, s, c],
        ],
        dtype=gs.np_float,
    )
    r_world = Rx @ p_local
    r_cross = np.array(
        [
            [0, -r_world[2], r_world[1]],
            [r_world[2], 0, -r_world[0]],
            [-r_world[1], r_world[0], 0],
        ],
        dtype=gs.np_float,
    )

    lin_o, ang_o = J_o[:3, 0], J_o[3:, 0]
    lin_expected = lin_o - r_cross @ ang_o

    assert_allclose(J_p[3:, 0], ang_o, tol=tol)
    assert_allclose(J_p[:3, 0], lin_expected, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu])
def test_urdf_parsing(show_viewer, tol):
    POS_OFFSET = 0.8
    WOLRD_QUAT = np.array([1.0, 1.0, -0.3, +0.3])
    DOOR_JOINT_DAMPING = 1.5

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            # Must use GJK to make collision detection independent from the center of each geometry.
            # Note that it is also the case for MPR+SDF most of the time due to warm-start.
            use_gjk_collision=True,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    asset_path = get_hf_dataset(pattern="microwave/*")
    entities = {}
    for i, (fixed, merge_fixed_links) in enumerate(
        ((False, False), (False, True), (True, False), (True, True)),
    ):
        entity = scene.add_entity(
            morph=gs.morphs.URDF(
                file=f"{asset_path}/microwave/microwave.urdf",
                fixed=fixed,
                merge_fixed_links=merge_fixed_links,
                pos=(0.0, (i - 1.5) * POS_OFFSET, 0.0),
                quat=tuple(WOLRD_QUAT / np.linalg.norm(WOLRD_QUAT)),
            ),
            vis_mode="collision",
        )
        entities[(fixed, merge_fixed_links)] = entity
    scene.build()

    # four microwaves have four different root_idx
    root_idx_all = [link.root_idx for link in scene.rigid_solver.links]
    assert len(set(root_idx_all)) == 4

    def _check_entity_positions(relative, tol):
        nonlocal entities
        AABB_all = []
        for key in ((False, False), (False, True), (True, False), (True, True)):
            AABB = np.array(
                [
                    [np.inf, np.inf, np.inf],
                    [-np.inf, -np.inf, -np.inf],
                ]
            )
            for geom in entities[key].geoms:
                AABB_i = geom.get_AABB()
                AABB[0] = np.minimum(AABB[0], AABB_i[0])
                AABB[1] = np.maximum(AABB[1], AABB_i[1])
            AABB_all.append(AABB)
        AABB_diff = np.diff(AABB_all, axis=0)
        if relative:
            AABB_diff[..., 1] -= POS_OFFSET
        assert_allclose(AABB_diff, 0.0, tol=tol)

    # Check that `set_pos` / `set_quat` applies the same transform in all cases
    for relative in (False, True):
        for key in ((False, False), (False, True), (True, False), (True, True)):
            entities[key].set_pos(np.array([0.5, 0.0, 0.0]), relative=relative)
            entities[key].set_quat(np.array([0.0, 0.0, 0.0, 1.0]), relative=relative)
        if show_viewer:
            scene.visualizer.update()
        _check_entity_positions(relative, tol=gs.EPS)

    # Check that `set_qpos` applies the same absolute transform in all cases
    door_angle = np.array([1.1])
    for i, key in enumerate(((False, False), (False, True))):
        qpos = np.concatenate(
            ((0.0, (i - 1.5) * POS_OFFSET, 0.0), tuple(WOLRD_QUAT / np.linalg.norm(WOLRD_QUAT)), door_angle)
        )
        entities[key].set_qpos(qpos)
    for i, key in enumerate(((True, False), (True, True))):
        entities[key].set_pos(np.array([0.0, 0.0, 0.0]), relative=True)
        entities[key].set_quat(np.array([1.0, 0.0, 0.0, 0.0]), relative=True)
        entities[key].set_qpos(door_angle)
    if show_viewer:
        scene.visualizer.update()
    _check_entity_positions(relative=True, tol=gs.EPS)

    # Add dof damping to stabilitze the physics
    for key in ((False, False), (False, True), (True, False), (True, True)):
        entities[key].set_dofs_damping(entities[key].get_dofs_damping() + DOOR_JOINT_DAMPING)

    # Make sure that the dynamics of the door is the same in all cases
    door_vel = np.array([-0.2])
    entities[(False, False)].set_dofs_velocity(door_vel, 6)
    entities[(False, True)].set_dofs_velocity(door_vel, 6)
    entities[(True, False)].set_dofs_velocity(door_vel)
    entities[(True, True)].set_dofs_velocity(door_vel)
    link_1 = entities[(True, True)].link_start
    for key in ((False, False), (False, True)):
        link_2 = entities[key].link_start
        scene.rigid_solver.add_weld_constraint(link_1, link_2)

    for i in range(2000):
        scene.step()
        door_pos_all = (
            entities[(False, False)].get_dofs_position(6),
            entities[(False, True)].get_dofs_position(6),
            entities[(True, False)].get_dofs_position(0),
            entities[(True, True)].get_dofs_position(0),
        )
        door_pos_diff = np.diff(torch.concatenate(door_pos_all))
        assert_allclose(door_pos_diff, 0, tol=5e-3)
    assert_allclose(scene.rigid_solver.dofs_state.vel.to_numpy(), 0.0, tol=1e-3)
    _check_entity_positions(relative=True, tol=2e-3)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu])
def test_urdf_mimic(show_viewer, tol):
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
    assert scene.rigid_solver.n_equalities == 1

    qvel = scene.rigid_solver.dofs_state.vel.to_numpy()
    qvel[-1] = 1
    scene.rigid_solver.dofs_state.vel.from_numpy(qvel)
    for i in range(200):
        scene.step()

    gs_qpos = scene.rigid_solver.qpos.to_numpy()[:, 0]
    assert_allclose(gs_qpos[-1], gs_qpos[-2], tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu])
def test_gravity(show_viewer, tol):
    scene = gs.Scene(
        show_viewer=show_viewer,
    )

    sphere = scene.add_entity(gs.morphs.Sphere())
    scene.build(n_envs=3)

    scene.sim.set_gravity(torch.tensor([0.0, 0.0, 0.0]))
    scene.sim.set_gravity(torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]), envs_idx=[0, 1])
    scene.sim.set_gravity(torch.tensor([0.0, 0.0, 3.0]), envs_idx=2)

    with np.testing.assert_raises(AssertionError):
        scene.sim.set_gravity(torch.tensor([0.0, -10.0]))

    with np.testing.assert_raises(AssertionError):
        scene.sim.set_gravity(torch.tensor([[0.0, 0.0, -10.0], [0.0, 0.0, -10.0]]), envs_idx=1)

    scene.step()

    assert_allclose(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        sphere.get_links_acc().squeeze(),
        tol=tol,
    )


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_scene_saver_franka(show_viewer, tol):
    scene1 = gs.Scene(
        show_viewer=show_viewer,
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
    )
    franka1 = scene1.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    scene1.build()

    dof_idx = [j.dofs_idx_local[0] for j in franka1.joints]

    franka1.set_dofs_kp(np.full(len(dof_idx), 3000), dof_idx)
    franka1.set_dofs_kv(np.full(len(dof_idx), 300), dof_idx)

    target_pose = np.array([0.3, -0.8, 0.4, -1.6, 0.5, 1.0, -0.6, 0.03, 0.03], dtype=float)
    franka1.control_dofs_position(target_pose, dof_idx)

    for _ in range(100):
        scene1.step()

    pose_ref = franka1.get_dofs_position(dof_idx)

    ckpt_path = Path(tempfile.gettempdir()) / "franka_unit.pkl"
    scene1.save_checkpoint(ckpt_path)

    scene2 = gs.Scene(show_viewer=show_viewer)
    franka2 = scene2.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    scene2.build()
    scene2.load_checkpoint(ckpt_path)

    pose_loaded = franka2.get_dofs_position(dof_idx)

    # FIXME: It should be possible to achieve better accuracy with 64bits precision
    assert_allclose(pose_ref, pose_loaded, tol=2e-6)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu])
def test_drone_hover_same_with_and_without_substeps(show_viewer, tol):
    base_rpm = 15000
    scene_ref = gs.Scene(
        show_viewer=show_viewer,
        sim_options=gs.options.SimOptions(
            dt=0.002,
            substeps=1,
        ),
    )
    drone_ref = scene_ref.add_entity(
        morph=gs.morphs.Drone(
            file="urdf/drones/cf2x.urdf",
            pos=(0, 0, 1.0),
        ),
    )
    scene_ref.build()

    for _ in range(2500):
        drone_ref.set_propellels_rpm([base_rpm, base_rpm, base_rpm, base_rpm])
        scene_ref.step()

    pos_ref = drone_ref.get_dofs_position()

    scene_test = gs.Scene(
        show_viewer=show_viewer,
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=5,
        ),
    )
    drone_test = scene_test.add_entity(
        morph=gs.morphs.Drone(
            file="urdf/drones/cf2x.urdf",
            pos=(0, 0, 1.0),
        ),
    )
    scene_test.build()

    for _ in range(500):
        drone_test.set_propellels_rpm([base_rpm, base_rpm, base_rpm, base_rpm])
        scene_test.step()

    pos_test = drone_test.get_dofs_position()

    assert_allclose(pos_ref, pos_test, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu])
def test_drone_advanced(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, 0.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    plane = scene.add_entity(gs.morphs.Plane())
    asset_path = get_hf_dataset(pattern="drone_sus/*")
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

    for drone in drones:
        chain_dofs = range(6, drone.n_dofs)
        drone.set_dofs_armature(drone.get_dofs_armature(chain_dofs) + 1e-3, chain_dofs)

    # Wait for the drones to land on the ground and hold straight
    for i in range(400):
        for drone in drones:
            drone.set_propellels_rpm(torch.full((4,), 50000.0))
        scene.step()
        if i > 350:
            assert scene.rigid_solver.collider._collider_state.n_contacts.to_numpy()[0] == 2
            assert_allclose(scene.rigid_solver.get_dofs_velocity(), 0, tol=2e-3)

    # Push the drones symmetrically and wait for them to collide
    drones[0].set_dofs_velocity([0.2], [1])
    drones[1].set_dofs_velocity([-0.2], [1])
    for i in range(150):
        for drone in drones:
            drone.set_propellels_rpm(torch.full((4,), 50000.0))
        scene.step()
        if scene.rigid_solver.collider._collider_state.n_contacts.to_numpy()[0] > 2:
            break
    else:
        raise AssertionError

    tol = 1e-2
    pos_1 = drones[0].get_pos()
    pos_2 = drones[1].get_pos()
    assert abs(pos_1[0] - pos_2[0]) < tol
    assert abs(pos_1[1] + pos_2[1]) < tol
    assert abs(pos_1[2] - pos_2[2]) < tol
    quat_1 = drones[0].get_quat()
    quat_2 = drones[1].get_quat()
    assert abs(quat_1[1] + quat_2[1]) < tol
    assert abs(quat_1[2] - quat_2[2]) < tol
    assert abs(quat_1[2] - quat_2[2]) < tol


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu])
def test_get_constraints_api(show_viewer, tol):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(gravity=(0.0, 0.0, 0.0)),
        show_viewer=show_viewer,
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
    )
    cube = scene.add_entity(gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.2, 0.0, 0.05)))
    scene.build(n_envs=2)

    link_a, link_b = robot.base_link.idx, cube.base_link.idx
    scene.sim.rigid_solver.add_weld_constraint(link_a, link_b, envs_idx=[1])
    with np.testing.assert_raises(AssertionError):
        scene.sim.rigid_solver.add_weld_constraint(link_a, link_b, envs_idx=[1])

    for as_tensor, to_torch in ((True, True), (True, False), (False, True), (False, False)):
        weld_const_info = scene.sim.rigid_solver.get_weld_constraints(as_tensor, to_torch)
        link_a_, link_b_ = weld_const_info["link_a"], weld_const_info["link_b"]
        if as_tensor:
            assert_allclose((link_a_[0], link_b_[0]), ((-1,), (-1,)), tol=0)
        else:
            assert_allclose((link_a_[0], link_b_[0]), ((), ()), tol=0)
        assert_allclose((link_a_[1], link_b_[1]), ((link_a,), (link_b,)), tol=0)


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
    # Create and build the scene
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            batch_dofs_info=batched,
            batch_joints_info=batched,
            batch_links_info=batched,
        ),
        show_viewer=False,
        show_FPS=False,
    )
    scene.add_entity(gs.morphs.Plane())
    gs_robot = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
        ),
    )
    gs_link = gs_robot.get_link("RR_thigh")
    scene.build(n_envs=n_envs)
    gs_s = scene.sim.rigid_solver

    # Initialize the simulation
    np.random.seed(0)
    dof_bounds = gs_s.dofs_info.limit.to_numpy()
    dof_bounds[..., :2, :] = np.array((-1.0, 1.0))
    dof_bounds[..., 2, :] = np.array((0.7, 1.0))
    dof_bounds[..., 3:6, :] = np.array((-np.pi / 2, np.pi / 2))
    for i in range(max(n_envs, 1)):
        qpos = dof_bounds[:, 0] + (dof_bounds[:, 1] - dof_bounds[:, 0]) * np.random.rand(gs_robot.n_dofs)
        gs_robot.set_dofs_position(qpos, envs_idx=([i] if n_envs else None))

    # Simulate for a while, until they collide with something
    for _ in range(400):
        scene.step()

        gs_n_contacts = gs_s.collider._collider_state.n_contacts.to_numpy()
        assert len(gs_n_contacts) == max(n_envs, 1)
        for as_tensor in (False, True):
            for to_torch in (False, True):
                contacts_info = gs_s.collider.get_contacts(as_tensor, to_torch)
                for value in contacts_info.values():
                    if n_envs > 0:
                        assert n_envs == len(value)
                    else:
                        assert gs_n_contacts[0] == len(value)
                        value = value[None] if as_tensor else (value,)

                    for i_b in range(n_envs):
                        n_contacts = gs_n_contacts[i_b]
                        if as_tensor:
                            assert isinstance(value, torch.Tensor if to_torch else np.ndarray)
                            if value.dtype in (gs.tc_int, gs.np_int):
                                assert (value[i_b, :n_contacts] != -1).all()
                                assert (value[i_b, n_contacts:] == -1).all()
                            else:
                                assert_allclose(value[i_b, n_contacts:], 0.0, tol=0)
                        else:
                            assert isinstance(value, (list, tuple))
                            assert value[i_b].shape[0] == n_contacts
                            if value[i_b].dtype in (gs.tc_int, gs.np_int):
                                assert (value[i_b] != -1).all()

        if (gs_n_contacts > 0).all():
            break
    else:
        assert False
    gs_s._func_forward_dynamics()
    gs_s._func_constraint_force()

    # Make sure that all the robots ends up in the different state
    qposs = gs_robot.get_qpos()
    for i in range(n_envs - 1):
        with np.testing.assert_raises(AssertionError):
            assert_allclose(qposs[i], qposs[i + 1], tol=tol)

    # Check attribute getters / setters.
    # First, without any any row or column masking:
    # * Call 'Get' -> Call 'Set' with random value -> Call 'Get'
    # * Compare first 'Get' ouput with taichi value
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

    for arg1_max, arg2_max, getter_or_spec, setter, ti_data in (
        # SOLVER
        (gs_s.n_links, n_envs, gs_s.get_links_pos, None, gs_s.links_state.pos),
        (gs_s.n_links, n_envs, gs_s.get_links_quat, None, gs_s.links_state.quat),
        (gs_s.n_links, n_envs, gs_s.get_links_vel, None, None),
        (gs_s.n_links, n_envs, gs_s.get_links_ang, None, gs_s.links_state.cd_ang),
        (gs_s.n_links, n_envs, gs_s.get_links_acc, None, None),
        (gs_s.n_links, n_envs, gs_s.get_links_root_COM, None, gs_s.links_state.root_COM),
        (gs_s.n_links, n_envs, gs_s.get_links_mass_shift, gs_s.set_links_mass_shift, gs_s.links_state.mass_shift),
        (gs_s.n_links, n_envs, gs_s.get_links_COM_shift, gs_s.set_links_COM_shift, gs_s.links_state.i_pos_shift),
        (gs_s.n_links, -1, gs_s.get_links_inertial_mass, gs_s.set_links_inertial_mass, gs_s.links_info.inertial_mass),
        (gs_s.n_links, -1, gs_s.get_links_invweight, None, gs_s.links_info.invweight),
        (gs_s.n_dofs, n_envs, gs_s.get_dofs_control_force, gs_s.control_dofs_force, None),
        (gs_s.n_dofs, n_envs, gs_s.get_dofs_force, None, gs_s.dofs_state.force),
        (gs_s.n_dofs, n_envs, gs_s.get_dofs_velocity, gs_s.set_dofs_velocity, gs_s.dofs_state.vel),
        (gs_s.n_dofs, n_envs, gs_s.get_dofs_position, gs_s.set_dofs_position, gs_s.dofs_state.pos),
        (gs_s.n_dofs, -1, gs_s.get_dofs_force_range, gs_s.set_dofs_force_range, gs_s.dofs_info.force_range),
        (gs_s.n_dofs, -1, gs_s.get_dofs_limit, None, gs_s.dofs_info.limit),
        (gs_s.n_dofs, -1, gs_s.get_dofs_stiffness, None, gs_s.dofs_info.stiffness),
        (gs_s.n_dofs, -1, gs_s.get_dofs_invweight, None, gs_s.dofs_info.invweight),
        (gs_s.n_dofs, -1, gs_s.get_dofs_armature, gs_s.set_dofs_armature, gs_s.dofs_info.armature),
        (gs_s.n_dofs, -1, gs_s.get_dofs_damping, gs_s.set_dofs_damping, gs_s.dofs_info.damping),
        (gs_s.n_dofs, -1, gs_s.get_dofs_kp, gs_s.set_dofs_kp, gs_s.dofs_info.kp),
        (gs_s.n_dofs, -1, gs_s.get_dofs_kv, gs_s.set_dofs_kv, gs_s.dofs_info.kv),
        (gs_s.n_geoms, n_envs, gs_s.get_geoms_pos, None, gs_s.geoms_state.pos),
        (
            gs_s.n_geoms,
            n_envs,
            gs_s.get_geoms_friction_ratio,
            gs_s.set_geoms_friction_ratio,
            gs_s.geoms_state.friction_ratio,
        ),
        (gs_s.n_geoms, -1, gs_s.get_geoms_friction, gs_s.set_geoms_friction, gs_s.geoms_info.friction),
        (gs_s.n_qs, n_envs, gs_s.get_qpos, gs_s.set_qpos, gs_s.qpos),
        # ROBOT
        (gs_robot.n_links, n_envs, gs_robot.get_links_pos, None, None),
        (gs_robot.n_links, n_envs, gs_robot.get_links_quat, None, None),
        (gs_robot.n_links, n_envs, gs_robot.get_links_vel, None, None),
        (gs_robot.n_links, n_envs, gs_robot.get_links_ang, None, None),
        (gs_robot.n_links, n_envs, gs_robot.get_links_acc, None, None),
        (gs_robot.n_links, n_envs, (), gs_robot.set_mass_shift, None),
        (gs_robot.n_links, n_envs, (3,), gs_robot.set_COM_shift, None),
        (gs_robot.n_links, n_envs, (), gs_robot.set_friction_ratio, None),
        (gs_robot.n_links, -1, gs_robot.get_links_inertial_mass, gs_robot.set_links_inertial_mass, None),
        (gs_robot.n_links, -1, gs_robot.get_links_invweight, None, None),
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
        (-1, -1, gs_robot.get_mass, gs_robot.set_mass, None),
        (-1, -1, gs_robot.get_AABB, None, None),
        # LINK
        (-1, -1, gs_link.get_mass, gs_link.set_mass, None),
        (-1, -1, gs_link.get_AABB, None, None),
    ):
        getter, spec = (getter_or_spec, None) if callable(getter_or_spec) else (None, getter_or_spec)

        # Check getter and setter without row or column masking
        if getter is not None:
            datas = getter()
            is_tuple = isinstance(datas, (tuple, list))
            if arg1_max > 0:
                assert_allclose(getter(range(arg1_max)), datas, tol=tol)
        else:
            batch_shape = []
            if arg2_max > 0:
                batch_shape.append(arg2_max)
            if arg1_max > 0:
                batch_shape.append(arg1_max)
            is_tuple = spec and isinstance(spec[0], (tuple, list))
            if is_tuple:
                datas = [torch.ones((*batch_shape, *shape)) for shape in spec]
            else:
                datas = torch.ones((*batch_shape, *spec))
        if ti_data is not None:
            true = ti_to_torch(ti_data)
            ti_ndim = getattr(ti_data, "ndim", len(getattr(ti_data, "element_shape", ())))
            true = true.movedim(true.ndim - ti_ndim - 1, 0)
            if is_tuple:
                true = torch.unbind(true, dim=-1)
                true = [val.reshape(data.shape) for data, val in zip(datas, true)]
            else:
                true = true.reshape(datas.shape)
            assert_allclose(datas, true, tol=tol)
        if setter is not None:
            if is_tuple:
                datas = [torch.as_tensor(val) for val in datas]
            else:
                datas = torch.as_tensor(datas)
            datas_tp = datas if is_tuple else (datas,)
            if getter is not None:
                # Randomly sample new data that are strictly positive and normalized,
                # as this may be required for some setters (mass, quaternion, ...).
                for val in datas_tp:
                    val[()] = torch.abs(torch.randn(val.shape, dtype=gs.tc_float, device=gs.device)) + gs.EPS
                    val /= torch.linalg.norm(val, dim=-1, keepdims=True)
            setter(*datas_tp)
            if getter is not None:
                assert_allclose(getter(), datas, tol=tol)

        # Early return if neither rows or columns can be masked
        if not (arg1_max > 0 or arg2_max > 0):
            continue

        # Check getter and setter for all possible combinations of row and column masking
        for i in range(arg1_max) if arg1_max > 0 else (None,):
            for arg1 in get_all_supported_masks(i) if arg1_max > 0 else (None,):
                for j in range(max(arg2_max, 1)) if arg2_max >= 0 else (None,):
                    for arg2 in get_all_supported_masks(j) if arg2_max > 0 else (None,):
                        if arg1 is None and arg2 is not None:
                            unsafe = not must_cast(arg2)
                            if getter is not None:
                                data = getter(arg2, unsafe=unsafe)
                            else:
                                if is_tuple:
                                    data = [torch.ones((1, *shape)) for shape in spec]
                                else:
                                    data = torch.ones((1, *spec))
                            if setter is not None:
                                setter(data, arg2, unsafe=unsafe)
                            if n_envs:
                                if is_tuple:
                                    data_ = [val[[j]] for val in datas]
                                else:
                                    data_ = datas[[j]]
                            else:
                                data_ = datas
                        elif arg1 is not None and arg2 is None:
                            unsafe = not must_cast(arg1)
                            if getter is not None:
                                data = getter(arg1, unsafe=unsafe)
                            else:
                                if is_tuple:
                                    data = [torch.ones((1, *shape)) for shape in spec]
                                else:
                                    data = torch.ones((1, *spec))
                            if setter is not None:
                                if is_tuple:
                                    setter(*data, arg1, unsafe=unsafe)
                                else:
                                    setter(data, arg1, unsafe=unsafe)
                            if is_tuple:
                                data_ = [val[[i]] for val in datas]
                            else:
                                data_ = datas[[i]]
                        else:
                            unsafe = not any(map(must_cast, (arg1, arg2)))
                            if getter is not None:
                                data = getter(arg1, arg2, unsafe=unsafe)
                            else:
                                if is_tuple:
                                    data = [torch.ones((1, 1, *shape)) for shape in spec]
                                else:
                                    data = torch.ones((1, 1, *spec))
                            if setter is not None:
                                setter(data, arg1, arg2, unsafe=unsafe)
                            if is_tuple:
                                data_ = [val[[j], :][:, [i]] for val in datas]
                            else:
                                data_ = datas[[j], :][:, [i]]
                        # FIXME: Not sure why tolerance must be increased for tests to pass
                        assert_allclose(data_, data, tol=(5.0 * tol))

    for dofs_idx in (*get_all_supported_masks(0), None):
        for envs_idx in (*(get_all_supported_masks(0) if n_envs > 0 else ()), None):
            unsafe = not any(map(must_cast, (dofs_idx, envs_idx)))
            dofs_pos = gs_s.get_dofs_position(dofs_idx, envs_idx)
            dofs_vel = gs_s.get_dofs_velocity(dofs_idx, envs_idx)
            gs_s.control_dofs_position(dofs_pos, dofs_idx, envs_idx)
            gs_s.control_dofs_velocity(dofs_vel, dofs_idx, envs_idx)

    # Must be tested independently because of non-trival return type
    gs_robot.get_contacts()


@pytest.mark.parametrize("backend", [gs.cpu])
def test_mesh_to_heightfield(tmp_path, show_viewer):
    horizontal_scale = 2.0
    path_terrain = os.path.join(get_assets_dir(), "meshes", "terrain_45.obj")

    hf_terrain, xs, ys = gs.utils.terrain.mesh_to_heightfield(path_terrain, spacing=horizontal_scale, oversample=1)

    # default heightfield starts at 0, 0, 0
    # translate to the center of the mesh
    translation = np.array([np.nanmin(xs), np.nanmin(ys), 0])

    ########################## create a scene ##########################
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(2, 0, -2),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -50, 0),
            camera_lookat=(0, 0, 0),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

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

    for i in range(1000):
        scene.step()

    # speed is around 0
    qvel = ball.get_dofs_velocity()
    assert_allclose(qvel, 0, atol=1e-2)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu])
def test_get_cartesian_space_variables(show_viewer, tol):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(
            # by default, enable_mujoco_compatibility=False
            # the test will fail if enable_mujoco_compatibility=True
            enable_mujoco_compatibility=False,
        ),
        show_viewer=show_viewer,
    )

    box = scene.add_entity(
        gs.morphs.Box(
            size=(1.0, 1.0, 1.0),
            pos=(0.0, 0.0, 0.0),
        )
    )
    scene.build()

    for _ in range(2):
        for link in box.links:
            force = torch.tensor(np.array([0, 0, 0])).unsqueeze(0)
            acc = 50.0
            force[0, 0] = acc * link.inertial_mass
            pos = link.get_pos()
            vel = link.get_vel()

            dof_vel = link.solver.get_dofs_velocity()
            dof_pos = link.solver.get_qpos()

            assert_allclose(dof_vel[:3], vel, atol=tol)
            assert_allclose(dof_pos[:3], pos, atol=tol)

            link.solver.apply_links_external_force(force, (link.idx,), ref="link_com", local=False)

        scene.step()


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu])
def test_geom_pos_quat(show_viewer, tol):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, -10.0),
        ),
        show_viewer=show_viewer,
    )

    box = scene.add_entity(
        gs.morphs.Box(
            size=(1.0, 1.0, 1.0),
            pos=(0.0, 0.0, 2.0),
        )
    )
    scene.build()

    for link in box.links:
        for vgeom, geom in zip(link.vgeoms, link.geoms):
            assert_allclose(geom.get_pos(), vgeom.get_pos(), atol=tol)
            assert_allclose(geom.get_quat(), vgeom.get_quat(), atol=tol)


@pytest.mark.required
def test_contype_conaffinity(show_viewer, tol):
    GRAVITY = (0.0, 0.0, -10.0)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=GRAVITY,
        ),
        show_viewer=show_viewer,
    )

    plane = scene.add_entity(
        gs.morphs.Plane(
            pos=(0.0, 0.0, 0.0),
        )
    )
    box1 = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.5, 0.5, 0.5),
            pos=(0.0, 0.0, 0.5),
            contype=3,
            conaffinity=3,
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.0, 0.0, 1.0),
        ),
    )
    box2 = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.5, 0.5, 0.5),
            pos=(0.0, 0.0, 1.0),
            contype=2,
            conaffinity=2,
        ),
        surface=gs.surfaces.Default(
            color=(0.0, 1.0, 0.0, 1.0),
        ),
    )
    box3 = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.5, 0.5, 0.5),
            pos=(0.0, 0.0, 1.5),
            contype=1,
            conaffinity=1,
        ),
        surface=gs.surfaces.Default(
            color=(0.0, 0.0, 1.0, 1.0),
        ),
    )
    box4 = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.5, 0.5, 0.5),
            pos=(0.0, 0.0, 2.0),
            contype=0,
            conaffinity=0,
        ),
        surface=gs.surfaces.Default(
            color=(0.8, 0.8, 0.8, 1.0),
        ),
    )
    scene.build()

    for _ in range(80):
        scene.step()

    assert_allclose(box1.get_pos(), (0.0, 0.0, 0.25), atol=5e-4)
    assert_allclose(box2.get_pos(), (0.0, 0.0, 0.75), atol=1e-3)
    assert_allclose(box2.get_pos(), box3.get_pos(), atol=1e-4)
    assert_allclose(scene.rigid_solver.get_links_acc(slice(box4.link_start, box4.link_end)), GRAVITY, atol=tol)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu])
def test_mesh_primitive_COM(show_viewer, tol):
    GRAVITY = (0.0, 0.0, -10.0)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=GRAVITY,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )

    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    bunny = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/bunny.obj",
            pos=(-1.0, -1.0, 1.0),
        ),
        vis_mode="collision",
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.5, 0.5, 0.5),
            pos=(1.0, 1.0, 1.0),
        ),
        vis_mode="collision",
    )
    ############# build ##########################
    scene.build()
    rigid = scene.sim.rigid_solver
    for i in range(500):
        scene.step()

    link_COM = rigid.get_links_pos(ref="link_com")
    root_COM = rigid.get_links_pos(ref="root_com")
    bunny_z = link_COM[1, 2]
    cube_z = link_COM[2, 2]
    root_bunny_z = root_COM[1, 2]
    root_cube_z = root_COM[2, 2]

    assert_allclose(bunny_z, bunny.get_links_pos(links_idx_local=[0], ref="link_com")[0, 2], atol=gs.EPS)
    assert_allclose(cube_z, cube.get_links_pos(links_idx_local=[0], ref="link_com")[0, 2], atol=gs.EPS)
    assert_allclose(root_bunny_z, bunny.get_links_pos(links_idx_local=[0], ref="root_com")[0, 2], atol=gs.EPS)
    assert_allclose(root_cube_z, cube.get_links_pos(links_idx_local=[0], ref="root_com")[0, 2], atol=gs.EPS)

    # in the old (wrong) code, their initial COM are (0,0,0)
    # but now their z are above the plane
    assert_allclose(bunny_z, 0.3424, atol=1e-3)
    assert_allclose(cube_z, 0.2499, atol=1e-3)

    # root and link COM should be the same for single link
    assert_allclose(root_bunny_z, bunny_z, atol=tol)
    assert_allclose(root_cube_z, cube_z, atol=tol)


@pytest.mark.required
@pytest.mark.parametrize("scale", [0.1, 10.0])
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_noslip_iterations(scale, show_viewer, tol):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            noslip_iterations=5,
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=show_viewer,
    )

    boxes = []
    for i in range(3):
        boxes.append(
            scene.add_entity(
                gs.morphs.Box(
                    size=(scale, scale, scale),
                    pos=(i * scale, 0, 0),
                    fixed=(i == 0),
                ),
            )
        )
    scene.build()

    rho = 200
    coeff_f = 1.0
    n_box = 2
    g = 9.81
    # FIXME: we need apply a larger force than expected to keep the boxes static
    safety = 2.5

    # simulate for 20 seconds
    for i in range(2000):
        boxes[2].control_dofs_force(np.array([-safety / coeff_f * n_box * rho * scale**3 * g]), np.array([0]))
        scene.step()

    box_1_z = boxes[1].get_qpos().cpu().numpy()[2]
    # allow some small sliding due to first few frames
    # scale = 0.1 is less stable than bigger scale
    assert_allclose(box_1_z, 0.0, atol=4e-2 * scale)

    # reduce the multiplier and it will slide
    safety = 0.9
    for i in range(2000):
        # push to -x direction
        boxes[2].control_dofs_force(np.array([-safety / coeff_f * n_box * rho * scale**3 * g]), np.array([0]))
        scene.step()

    box_1_z = boxes[1].get_qpos().cpu().numpy()[2]
    # it will slip away
    assert box_1_z < -scale


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 1])
def test_axis_aligned_bounding_boxes(n_envs):
    scene = gs.Scene()
    scene.add_entity(
        gs.morphs.Plane(
            normal=(0, 0, 1),
            pos=(0, 0, 0),
        ),
    )
    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.5, 0, 0.05),
        ),
    )
    scene.add_entity(
        gs.morphs.Cylinder(
            height=0.8,
            radius=0.06,
            pos=(1.0, 0, 0.5),
        ),
    )
    scene.add_entity(
        gs.morphs.Sphere(
            radius=0.05,
            pos=(-0.5, 0, 0.05),
        ),
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
    )
    scene.build(n_envs=n_envs)

    aabb_shape = (*((n_envs,) if n_envs > 0 else ()), 2, 3)
    robot_aabb = robot.get_AABB()
    robot_geoms_aabb = torch.stack([geom.get_AABB().expand(aabb_shape) for geom in robot.geoms], dim=0)
    assert_allclose(torch.min(robot_geoms_aabb[..., 0, :], dim=0).values, robot_aabb[..., 0, :], tol=gs.EPS)
    assert_allclose(torch.max(robot_geoms_aabb[..., 1, :], dim=0).values, robot_aabb[..., 1, :], tol=gs.EPS)

    all_aabbs = scene.sim.rigid_solver.get_AABB()
    aabbs = [geom.get_AABB().expand(aabb_shape) for entity in scene.entities for geom in entity.geoms]
    if n_envs > 0:
        assert all_aabbs.ndim == 4 and len(all_aabbs) == n_envs
    else:
        assert all_aabbs.ndim == 3
    assert all_aabbs.shape[-3:] == (len(aabbs), 2, 3)
    assert_allclose(aabbs[:4], all_aabbs.swapaxes(-3, 0)[:4], atol=gs.EPS)
    with pytest.raises(AssertionError):
        assert_allclose(aabbs[4:], all_aabbs.swapaxes(-3, 0)[4:], atol=gs.EPS)

    box_aabb_min, box_aabb_max = aabbs[1].split(1, dim=-2)
    assert_allclose(box_aabb_min, (0.45, -0.05, 0.0), atol=gs.EPS)
    assert_allclose(box_aabb_max, (0.55, 0.05, 0.1), atol=gs.EPS)
    sphere_aabb_min, sphere_aabb_max = aabbs[3].split(1, dim=-2)
    assert_allclose(sphere_aabb_min, (-0.55, -0.05, 0.0), atol=gs.EPS)
    assert_allclose(sphere_aabb_max, (-0.45, 0.05, 0.1), atol=gs.EPS)


@pytest.mark.required
@pytest.mark.parametrize("batch_links_info", [False, True])
@pytest.mark.parametrize("batch_joints_info", [False, True])
@pytest.mark.parametrize("batch_dofs_info", [False, True])
def test_batched_info(batch_links_info, batch_joints_info, batch_dofs_info):
    """
    Test if batching options (batch_links_info, batch_joints_info, batch_dofs_info) work correctly.
    """
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            batch_links_info=batch_links_info,
            batch_joints_info=batch_joints_info,
            batch_dofs_info=batch_dofs_info,
        ),
    )
    terrain = scene.add_entity(gs.morphs.Terrain())
    scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build(n_envs=2)

    links_info = terrain.solver.data_manager.links_info
    entity_idx = links_info.entity_idx.to_numpy()
    assert entity_idx.shape == (12, 2) if batch_links_info else (12,)

    joints_info = terrain.solver.data_manager.joints_info
    pos = joints_info.pos.to_numpy()
    assert pos.shape == (10, 2, 3) if batch_joints_info else (10, 3)

    dofs_info = terrain.solver.data_manager.dofs_info
    kp = dofs_info.kp.to_numpy()
    assert kp.shape == (9, 2) if batch_dofs_info else (9,)
