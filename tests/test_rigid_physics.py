import math
import os
import sys
import xml.etree.ElementTree as ET
from contextlib import nullcontext
from copy import deepcopy
from itertools import product
from typing import TYPE_CHECKING

import igl
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pytest
import torch
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.terrain as tu
from genesis.ext import urdfpy
from genesis.utils import urdf as uu
from genesis.engine.states.solvers import RigidSolverState
from genesis.utils.misc import get_assets_dir, qd_to_numpy, qd_to_torch, tensor_to_array

from .utils import (
    assert_allclose,
    assert_equal,
    check_mujoco_data_consistency,
    check_mujoco_model_consistency,
    display_collision_pairs,
    get_genuine_interpenetration,
    get_hf_dataset,
    init_simulators,
    simulate_and_check_mujoco_consistency,
)

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_entity import RigidEntity


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
def decompose_fusion_groups(asset_tmp_path):
    """Generate an MJCF model of a single static link mixing several contact-parameter sub-groups: a plane, a
    nonconvex L-shaped mesh (hull error ~0.3, above the 0.15 threshold) with a small primitive box touching its inner
    corner, two disjoint convex mesh boxes (0.01 gap along x) with a different friction, two disjoint primitive boxes
    with yet another friction, and two mesh boxes with adjacent large collision masks that must never be grouped
    together."""
    lshape = trimesh.util.concatenate(
        [
            trimesh.creation.box(extents=(0.2, 0.1, 0.1)),
            trimesh.creation.box(
                extents=(0.1, 0.1, 0.3), transform=trimesh.transformations.translation_matrix((0.05, 0.0, 0.2))
            ),
        ]
    )
    lshape.export(asset_tmp_path / "lshape.obj")
    trimesh.creation.box(extents=(0.1, 0.1, 0.1)).export(asset_tmp_path / "small_box.obj")

    mjcf = ET.Element("mujoco", model="decompose_fusion_groups")
    asset = ET.SubElement(mjcf, "asset")
    ET.SubElement(asset, "mesh", name="lshape", file=str(asset_tmp_path / "lshape.obj"))
    ET.SubElement(asset, "mesh", name="small_box", file=str(asset_tmp_path / "small_box.obj"))
    worldbody = ET.SubElement(mjcf, "worldbody")
    ET.SubElement(worldbody, "geom", type="plane", size="5 5 0.1")
    ET.SubElement(worldbody, "geom", type="mesh", mesh="lshape", pos="0 0.5 1")
    ET.SubElement(worldbody, "geom", type="box", size="0.01 0.01 0.01", pos="-0.01 0.5 1.06")
    ET.SubElement(worldbody, "geom", type="mesh", mesh="small_box", pos="-0.055 0 1", friction="0.5")
    ET.SubElement(worldbody, "geom", type="mesh", mesh="small_box", pos="0.055 0 1", friction="0.5")
    ET.SubElement(worldbody, "geom", type="box", size="0.02 0.02 0.02", pos="0.3 0 1", friction="0.8")
    ET.SubElement(worldbody, "geom", type="box", size="0.02 0.02 0.02", pos="0.37 0 1", friction="0.8")
    ET.SubElement(worldbody, "geom", type="mesh", mesh="small_box", pos="0 -0.5 1", contype="16777216")
    ET.SubElement(worldbody, "geom", type="mesh", mesh="small_box", pos="0.15 -0.5 1", contype="16777217")
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


def _build_multi_pendulum(n, joint_damping, joint_friction):
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
        ET.SubElement(joint, "dynamics", damping=str(joint_damping), friction=str(joint_friction))

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


@pytest.fixture
def pendulum_with_joint_dynamics(joint_damping, joint_friction):
    return _build_multi_pendulum(n=1, joint_damping=joint_damping, joint_friction=joint_friction)


@pytest.fixture(scope="session")
def pendulum():
    return _build_multi_pendulum(n=1, joint_damping=0.0, joint_friction=0.0)


@pytest.fixture(scope="session")
def double_pendulum():
    return _build_multi_pendulum(n=2, joint_damping=0.0, joint_friction=0.0)


@pytest.fixture(scope="session")
def undefined_inertia():
    """Generate a URDF with a single link that has no inertial element."""
    urdf = ET.Element("robot", name="undefined_inertia")
    link = ET.SubElement(urdf, "link", name="base_link")
    visual = ET.SubElement(link, "visual")
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry, "sphere", radius="0.03")
    collision = ET.SubElement(link, "collision")
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "sphere", radius="0.03")
    return urdf


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
    ET.SubElement(
        ee,
        "geom",
        name="marker",
        type="sphere",
        contype="0",
        conaffinity="0",
        size="0.01",
        density="0",
        pos="0 -0.02 0",
        rgba="0.0 0.0 0.0 1.0",
    )

    return mjcf


@pytest.fixture(scope="session")
def long_chain():
    # Single kinematic tree with enough DOFs that its mass submatrix exceeds GPU shared memory, so the cooperative
    # >shared-cap mass assemble runs - the path whose lower-triangular linear-index inversion must stay exact on GPUs
    # with an imprecise sqrt.
    mjcf = ET.Element("mujoco", model="long_chain")
    ET.SubElement(mjcf, "compiler", angle="radian")
    worldbody = ET.SubElement(mjcf, "worldbody")
    body = ET.SubElement(worldbody, "body", name="root", pos="0 0 2")
    ET.SubElement(body, "geom", type="sphere", size="0.03", density="500")
    for i in range(128):
        body = ET.SubElement(body, "body", name=f"l{i}", pos="0 0 0.1")
        ET.SubElement(body, "joint", name=f"j{i}", type="hinge", axis=("1 0 0", "0 1 0", "0 0 1")[i % 3], damping="0.1")
        ET.SubElement(body, "geom", type="capsule", fromto="0 0 0 0 0 0.1", size="0.02", density="500")
    return mjcf


@pytest.fixture(scope="session")
def two_fixed_branches():
    # One entity whose worldbody holds two independent chains, each rigidly attached to the (fixed) world. Their DOFs
    # are kinematically decoupled, so the mass matrix is block-diagonal and must partition into one block per branch.
    mjcf = ET.Element("mujoco", model="two_fixed_branches")
    ET.SubElement(mjcf, "compiler", angle="radian")
    worldbody = ET.SubElement(mjcf, "worldbody")
    for name, x in (("a", 0.0), ("b", 1.0)):
        body = ET.SubElement(worldbody, "body", name=f"{name}root", pos=f"{x} 0 1")
        ET.SubElement(body, "geom", type="capsule", fromto="0 0 0 0 0 0.06", size="0.02", density="500")
        for i in range(4):
            body = ET.SubElement(body, "body", name=f"{name}{i}", pos="0 0 0.06")
            ET.SubElement(body, "joint", name=f"j{name}{i}", type="hinge", axis="0 1 0", damping="0.1")
            ET.SubElement(body, "geom", type="capsule", fromto="0 0 0 0 0 0.06", size="0.02", density="500")
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


def _ellipsoid_mjcf(semi_axes, body_name="obj", joint_name="root"):
    a, b, c = semi_axes
    mjcf = ET.Element("mujoco", model="ellipsoid")
    worldbody = ET.SubElement(mjcf, "worldbody")
    body = ET.SubElement(worldbody, "body", name=body_name, pos="0 0 0.0")
    ET.SubElement(body, "joint", name=joint_name, type="free")
    ET.SubElement(body, "geom", type="ellipsoid", size=f"{a} {b} {c}")
    return mjcf


@pytest.fixture(scope="session")
def ellipsoid():
    return _ellipsoid_mjcf((0.05, 0.05, 0.02))


@pytest.fixture(scope="session")
def general_actuator():
    """Generate an MJCF model with mixed actuator types: PD, general, and non-actuated."""
    mjcf = ET.Element("mujoco", model="general_actuator")
    ET.SubElement(mjcf, "option", timestep="0.01")
    worldbody = ET.SubElement(mjcf, "worldbody")
    body1 = ET.SubElement(worldbody, "body", name="link1", pos="0 0 1")
    ET.SubElement(body1, "joint", name="hinge_pd", type="hinge", axis="0 1 0", damping="0.5")
    ET.SubElement(body1, "geom", type="capsule", size="0.05 0.3", mass="1.0")
    body2 = ET.SubElement(body1, "body", name="link2", pos="0 0 -0.6")
    ET.SubElement(body2, "joint", name="hinge_general", type="hinge", axis="0 1 0", damping="0.3")
    ET.SubElement(body2, "geom", type="capsule", size="0.04 0.2", mass="0.5")
    body3 = ET.SubElement(body2, "body", name="link3", pos="0 0 -0.4")
    ET.SubElement(body3, "joint", name="hinge_motor", type="hinge", axis="0 1 0", damping="0.2")
    ET.SubElement(body3, "geom", type="capsule", size="0.03 0.15", mass="0.3")
    actuator = ET.SubElement(mjcf, "actuator")
    ET.SubElement(actuator, "position", name="act_pd", joint="hinge_pd", kp="100")
    ET.SubElement(
        actuator,
        "general",
        name="act_general",
        joint="hinge_general",
        gainprm="20 0 0",
        biastype="affine",
        biasprm="0.5 -10 -1",
    )
    ET.SubElement(actuator, "motor", name="act_motor", joint="hinge_motor", gear="5")
    return mjcf


@pytest.fixture(scope="session")
def compound_joint():
    mjcf = ET.Element("mujoco", model="compound_joint")
    ET.SubElement(mjcf, "compiler", angle="radian")
    ET.SubElement(mjcf, "option", gravity="0 0 0")
    worldbody = ET.SubElement(mjcf, "worldbody")
    seg1 = ET.SubElement(worldbody, "body", name="seg1", pos="0 0 0")
    ET.SubElement(seg1, "joint", name="j_x", type="hinge", axis="1 0 0")
    ET.SubElement(seg1, "joint", name="j_y", type="hinge", axis="0 1 0")
    ET.SubElement(seg1, "geom", type="capsule", size="0.02", fromto="0 0 0 0 0 0.4")
    seg2 = ET.SubElement(seg1, "body", name="seg2", pos="0 0 0.4")
    ET.SubElement(seg2, "joint", name="j_z", type="hinge", axis="0 0 1")
    ET.SubElement(seg2, "geom", type="capsule", size="0.02", fromto="0 0 0 0 0 0.4")
    return mjcf


@pytest.fixture(scope="session")
def depth_first_tree_mjcf():
    # A kinematic tree where breadth-first and depth-first orderings differ: root A has a child A1, and a sibling root
    # B has none, so depth-first visits A, A1, B (A's subtree contiguous) while breadth-first would give A, B, A1.
    mjcf = ET.Element("mujoco", model="depth_first_tree")
    worldbody = ET.SubElement(mjcf, "worldbody")
    a = ET.SubElement(worldbody, "body", name="A", pos="0 0 1")
    ET.SubElement(a, "freejoint")
    ET.SubElement(a, "geom", type="box", size="0.05 0.05 0.05")
    a1 = ET.SubElement(a, "body", name="A1", pos="0.15 0 0")
    ET.SubElement(a1, "joint", type="hinge", axis="0 0 1")
    ET.SubElement(a1, "geom", type="box", size="0.05 0.05 0.05")
    b = ET.SubElement(worldbody, "body", name="B", pos="1 0 1")
    ET.SubElement(b, "freejoint")
    ET.SubElement(b, "geom", type="box", size="0.05 0.05 0.05")
    return mjcf


@pytest.fixture(scope="session")
def depth_first_tree_urdf():
    # Same shape as depth_first_tree_mjcf but single-rooted (URDF): base -> {A, B}, A -> A1.
    robot = ET.Element("robot", name="depth_first_tree")
    for name in ("base", "A", "A1", "B"):
        link = ET.SubElement(robot, "link", name=name)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "mass", value="1.0")
        ET.SubElement(inertial, "inertia", ixx="0.01", iyy="0.01", izz="0.01", ixy="0", ixz="0", iyz="0")
        collision = ET.SubElement(link, "collision")
        ET.SubElement(ET.SubElement(collision, "geometry"), "box", size="0.1 0.1 0.1")
    for joint_name, parent, child in (("j_A", "base", "A"), ("j_A1", "A", "A1"), ("j_B", "base", "B")):
        joint = ET.SubElement(robot, "joint", name=joint_name, type="revolute")
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=child)
        ET.SubElement(joint, "origin", xyz="0 0 0.2")
        ET.SubElement(joint, "axis", xyz="0 0 1")
        ET.SubElement(joint, "limit", lower="-1", upper="1", effort="10", velocity="10")
    return robot


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["depth_first_tree_mjcf", "depth_first_tree_urdf"])
def test_depth_first_link_ordering(xml_path, model_name, show_viewer):
    # Links must be parsed depth-first so every subtree - hence every free body's DOFs - occupies a contiguous index
    # range. The per-tree mass-matrix factorization relies on this so a multi-body file costs the same as the
    # equivalent separate entities.
    scene = gs.Scene(show_viewer=show_viewer)
    morph = gs.morphs.MJCF(file=xml_path) if model_name.endswith("mjcf") else gs.morphs.URDF(file=xml_path, fixed=True)
    entity = scene.add_entity(morph)
    scene.build(n_envs=0)

    parents = [link.parent_idx for link in entity.links]
    n_links = len(parents)
    children: dict[int, list[int]] = {i: [] for i in range(n_links)}
    for i, parent in enumerate(parents):
        if parent != -1:
            children[parent].append(i)
    for i in range(n_links):
        subtree = []
        stack = [i]
        while stack:
            link = stack.pop()
            subtree.append(link)
            stack.extend(children[link])
        assert sorted(subtree) == list(range(i, i + len(subtree))), f"subtree at link {i} is not contiguous"


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["box_plan"])
@pytest.mark.parametrize(
    "gs_solver, gs_integrator",
    [
        (gs.constraint_solver.CG, gs.integrator.implicitfast),
        (gs.constraint_solver.CG, gs.integrator.Euler),
        (gs.constraint_solver.Newton, gs.integrator.implicitfast),
        (gs.constraint_solver.Newton, gs.integrator.Euler),
        # Elliptic (second-order) friction cone must match MuJoCo's elliptic cone. The box lands and slides with an
        # initial tangential + angular velocity so the tangential cone rows are exercised in both the sliding (cone
        # boundary) and sticking (bottom) regimes.
        pytest.param(
            gs.constraint_solver.CG,
            gs.integrator.implicitfast,
            marks=pytest.mark.friction_cone(gs.friction_cone.elliptic),
            id="CG-implicitfast-elliptic",
        ),
        pytest.param(
            gs.constraint_solver.Newton,
            gs.integrator.implicitfast,
            marks=pytest.mark.friction_cone(gs.friction_cone.elliptic),
            id="Newton-implicitfast-elliptic",
        ),
    ],
)
@pytest.mark.parametrize("backend", [gs.cpu])
def test_box_plane_dynamics(gs_sim, mj_sim, tol):
    cube_pos = np.array([0.0, 0.0, 0.6])
    cube_quat = np.random.rand(4)
    cube_quat /= np.linalg.norm(cube_quat)
    qpos = np.concatenate((cube_pos, cube_quat))
    qvel = np.random.rand(6) * 0.2
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, qpos, qvel, num_steps=150, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["general_actuator"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_general_actuator(gs_sim, mj_sim, tol):
    (entity,) = gs_sim.entities

    # get_dofs_kp raises for all DOFs (joint 1 is non-PD-reducible from parser)
    with pytest.raises(gs.GenesisException):
        entity.get_dofs_kp()

    # but succeeds for the PD joint (joint 0)
    entity.get_dofs_kp(dofs_idx_local=[0])

    # Set different control modes per DOF via public API
    entity.control_dofs_force(0.0, dofs_idx_local=[0])
    entity.control_dofs_velocity(0.0, dofs_idx_local=[1])
    entity.control_dofs_position(0.0, dofs_idx_local=[2])
    ctrl_mode = gs_sim.rigid_solver.dofs_state.ctrl_mode.to_numpy()[:, 0]
    assert ctrl_mode[entity.dof_start + 0] == gs.CTRL_MODE.FORCE
    assert ctrl_mode[entity.dof_start + 1] == gs.CTRL_MODE.VELOCITY
    assert ctrl_mode[entity.dof_start + 2] == gs.CTRL_MODE.POSITION

    # control_dofs_position overrides all to POSITION
    entity.control_dofs_position([0.0, 0.0, 0.0])
    ctrl_mode = gs_sim.rigid_solver.dofs_state.ctrl_mode.to_numpy()[:, 0]
    assert (ctrl_mode[entity.dof_start : entity.dof_start + 3] == gs.CTRL_MODE.POSITION).all()

    # Disable constraints, keep actuation enabled
    mj_sim.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    gs_sim.rigid_solver._enable_collision = False
    gs_sim.rigid_solver._enable_joint_limit = False
    gs_sim.rigid_solver._disable_constraint = True
    gs_sim.rigid_solver.collider.clear()
    gs_sim.rigid_solver.constraint_solver.clear()

    # Compare all dynamic quantities against MuJoCo with both PD and general actuators active.
    check_mujoco_model_consistency(gs_sim, mj_sim, tol=tol)
    init_simulators(gs_sim, mj_sim, qpos=np.array([0.2, 0.1, 0.0]), qvel=np.array([0.1, -0.1, 0.0]))

    mj_sim.data.ctrl[:] = [0.5, 0.3, 1.0]
    entity.control_dofs_position([0.5, 0.3, 0.0])
    entity.control_dofs_force(5.0, dofs_idx_local=[2])  # motor: gear(5) * gainprm(1) * ctrl(1) = 5

    # Pre-step so that Genesis computes qf_applied (needed for data consistency checks)
    mj_sim.data.qpos[:] = gs_sim.rigid_solver.qpos.to_numpy()[:, 0]
    mj_sim.data.qvel[:] = gs_sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]
    mujoco.mj_step(mj_sim.model, mj_sim.data)
    gs_sim.scene.step()

    for _ in range(99):
        check_mujoco_data_consistency(gs_sim, mj_sim, tol=tol, ignore_constraints=True)

        mj_sim.data.qpos[:] = gs_sim.rigid_solver.qpos.to_numpy()[:, 0]
        mj_sim.data.qvel[:] = gs_sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]
        mujoco.mj_step(mj_sim.model, mj_sim.data)
        gs_sim.scene.step()

    # Validate setter/getter round-trips for actuator parameters
    entity.set_dofs_act_gain([200.0], dofs_idx_local=[1])
    assert_allclose(entity.get_dofs_act_gain()[1], 200.0, tol=1e-6)
    entity.set_dofs_act_bias([0.5], [-100.0], [-5.0], dofs_idx_local=[1])
    b0, b1, b2 = entity.get_dofs_act_bias()
    assert_allclose(b0[1], 0.5, tol=1e-6)
    assert_allclose(b1[1], -100.0, tol=1e-6)
    assert_allclose(b2[1], -5.0, tol=1e-6)

    # set_dofs_kp restores PD on joint 1: act_gain=kp, act_bias[0]=0, act_bias[1]=-kp
    entity.set_dofs_kp([50.0], dofs_idx_local=[1])
    assert_allclose(entity.get_dofs_kp(dofs_idx_local=[0, 1]), [100.0, 50.0], tol=1e-6)
    b0, b1, _ = entity.get_dofs_act_bias()
    assert_allclose(b0[1], 0.0, tol=1e-6)
    assert_allclose(b1[1], -50.0, tol=1e-6)


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
@pytest.mark.parametrize("xml_path", ["xml/four_bar_linkage_weld.xml", "weld.xml", "connect.xml"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_equality_link(gs_sim, mj_sim, gs_solver, xml_path):
    # Must disable self-collision caused by closing the kinematic chain (adjacent link filtering is not enough)
    gs_sim.rigid_solver._enable_collision = False
    mj_sim.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

    # Must the time constant of the constraints to improve numerical stability
    TIME_CONSTANT = 0.02
    for entity in gs_sim.entities:
        for equality in entity.equalities:
            equality.set_sol_params((TIME_CONSTANT, *tensor_to_array(equality.sol_params)[1:]))
    mj_sim.model.eq_solref[:, 0] = TIME_CONSTANT

    # Randomize the initial condition for force convergence of the constraints
    np.random.seed(0)
    qpos = np.random.rand(gs_sim.rigid_solver.n_qs) * 0.1

    # Note that the world frame in which weld constraint is computed is different between Mujoco and Genesis for sites.
    # Mujoco is using site 1, whereas Genesis is using parent link frame of site 1 since it has no notion of site.
    ignore_constraints = np.any(
        (mj_sim.model.eq_objtype == mujoco.mjtObj.mjOBJ_SITE) & (mj_sim.model.eq_type == mujoco.mjtEq.mjEQ_WELD)
    )
    simulate_and_check_mujoco_consistency(
        gs_sim, mj_sim, qpos, num_steps=300, tol=1e-7, ignore_constraints=ignore_constraints
    )


@pytest.mark.slow  # ~250s
@pytest.mark.required
def test_dynamic_weld(show_viewer, tol):
    CUBE_POS = (0.65, 0.0, 0.02)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(5.5, 0.0, 2.5),
            camera_lookat=(1.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(
        gs.morphs.Plane(),
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=CUBE_POS,
        ),
        surface=gs.surfaces.Default(
            color=(1, 0, 0),
        ),
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
    for _ in range(60):
        scene.step()
    cubes_pos, cubes_quat = cube.get_pos(), tensor_to_array(cube.get_quat())
    assert_allclose(gu.quat_to_rotvec(cubes_quat), 0.0, tol=1e-3)
    assert_allclose(torch.diff(cubes_pos[[0, 1, 2]], dim=0), 0.0, tol=tol)
    assert_allclose(cubes_pos[3], CUBE_POS, tol=1e-3)
    assert_allclose(cubes_pos[-1] - cubes_pos[0], ee_pos_down - ee_pos_up, tol=1e-2)

    # drop
    scene.sim.rigid_solver.delete_weld_constraint(cube.base_link.idx, end_effector.idx, envs_idx=(0, 1))
    for _ in range(110):
        scene.step()
    cubes_pos, cubes_quat = cube.get_pos(), tensor_to_array(cube.get_quat())
    assert_allclose(gu.quat_to_rotvec(cubes_quat), 0.0, tol=1e-3)
    assert_allclose(torch.diff(cubes_pos[[0, 1, 3]], dim=0), 0.0, tol=1e-2)
    assert_allclose(cubes_pos[2] - cubes_pos[0], ee_pos_up - ee_pos_down, tol=1e-3)


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_dynamic_weld_scene_reset():
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            max_dynamic_constraints=10,
        ),
        show_viewer=False,
    )
    box1 = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0, 0, 0.5),
        )
    )
    box2 = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.2, 0, 0.5),
        )
    )
    scene.build(n_envs=2)

    solver = scene.rigid_solver
    n_eq_base = solver._rigid_global_info.n_equalities[None]

    solver.add_weld_constraint(box1.base_link_idx, box2.base_link_idx)
    assert solver.constraint_solver.constraint_state.qd_n_equalities[0] == n_eq_base + 1
    assert solver.constraint_solver.constraint_state.qd_n_equalities[1] == n_eq_base + 1

    scene.reset(state=scene.get_state(), envs_idx=[0])
    assert solver.constraint_solver.constraint_state.qd_n_equalities[0] == n_eq_base
    assert solver.constraint_solver.constraint_state.qd_n_equalities[1] == n_eq_base + 1


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_reset(show_viewer):
    BOOL_MASK = torch.tensor([True, False, True, False], dtype=torch.bool, device=gs.device)

    scene = gs.Scene(
        show_viewer=show_viewer,
    )
    scene.add_entity(
        gs.morphs.URDF(
            file="urdf/plane/plane.urdf",
            fixed=True,
        )
    )
    scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0, 0, 0.5),
        )
    )
    scene.build(n_envs=4)

    init_state = scene.get_state()
    init_rigid_state = next(s for s in init_state.solvers_state if isinstance(s, RigidSolverState))
    for _ in range(50):
        scene.step()
    fallen_state = scene.get_state()
    fallen_rigid_state = next(s for s in fallen_state.solvers_state if isinstance(s, RigidSolverState))

    for envs_idx in (BOOL_MASK, torch.where(BOOL_MASK)[0]):
        scene.reset(state=fallen_state)
        scene.reset(state=init_state, envs_idx=envs_idx)
        for actual, init_ref, fallen_ref in (
            (
                qd_to_torch(scene.rigid_solver._rigid_global_info.qpos, transpose=True, copy=True),
                init_rigid_state.qpos,
                fallen_rigid_state.qpos,
            ),
            (
                qd_to_torch(scene.rigid_solver.dofs_state.vel, transpose=True, copy=True),
                init_rigid_state.dofs_vel,
                fallen_rigid_state.dofs_vel,
            ),
            (
                qd_to_torch(scene.rigid_solver.links_state.pos, transpose=True, copy=True),
                init_rigid_state.links_pos,
                fallen_rigid_state.links_pos,
            ),
        ):
            assert_allclose(actual[BOOL_MASK], init_ref[BOOL_MASK], tol=gs.EPS)
            assert_allclose(actual[~BOOL_MASK], fallen_ref[~BOOL_MASK], tol=gs.EPS)

    # After reset, simulation from init_state should reproduce the original fallen_state trajectory
    for _ in range(50):
        scene.step()
    for actual, fallen_ref in (
        (qd_to_torch(scene.rigid_solver._rigid_global_info.qpos, transpose=True, copy=True), fallen_rigid_state.qpos),
        (qd_to_torch(scene.rigid_solver.dofs_state.vel, transpose=True, copy=True), fallen_rigid_state.dofs_vel),
        (qd_to_torch(scene.rigid_solver.links_state.pos, transpose=True, copy=True), fallen_rigid_state.links_pos),
    ):
        assert_allclose(actual[BOOL_MASK], fallen_ref[BOOL_MASK], tol=gs.EPS)


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
@pytest.mark.parametrize("xml_path", ["linear_deformable.urdf"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast])
@pytest.mark.parametrize("gjk_collision", [True, False])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_urdf_rope(gs_sim, mj_sim, gs_solver, xml_path):
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
@pytest.mark.parametrize("xml_path", ["xml/tet_tet.xml", "xml/tet_ball.xml", "xml/tet_capsule.xml"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("gjk_collision", [True])
@pytest.mark.parametrize("multi_contact", [True, False])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_tet_primitive_shapes(gs_sim, mj_sim, gs_integrator, gs_solver, xml_path, multi_contact, tol):
    # Make sure it is possible to set the configuration vector without failure
    gs_sim.rigid_solver.set_dofs_position(gs_sim.rigid_solver.get_dofs_position())

    check_mujoco_model_consistency(gs_sim, mj_sim, tol=tol)
    # FIXME: Because of very small numerical error, error could be this large even if there is no logical error.
    # Multi-contact perturbation introduces slightly larger errors due to GJK implementation differences.
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, num_steps=700, tol=2e-6)


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
        theta = gs_sim.rigid_solver.qpos[0, 0]
        theta_dot = gs_sim.rigid_solver.dofs_state.vel[0, 0]

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
        assert_allclose(acc_classical_lin_local[0], np.array([0.0, -theta_ddot[0], -(theta_dot[0] ** 2)]), tol=tol)
        assert_allclose(
            acc_classical_lin_local[1],
            R[..., 1] @ acc_classical_lin_world[1] + np.array([0.0, -theta_ddot.sum(), -(theta_dot.sum() ** 2)]),
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


def _capsule_mjcf_path(tmp_path, radius, length, name="capsule"):
    mjcf = ET.Element("mujoco", model=name)
    body = ET.SubElement(ET.SubElement(mjcf, "worldbody"), "body")
    ET.SubElement(body, "geom", type="capsule", size=f"{radius} {0.5 * length}")
    ET.SubElement(body, "joint", type="free")
    path = tmp_path / f"{name}.xml"
    ET.ElementTree(mjcf).write(path)
    return str(path)


def _build_primitive_pair_mjcf(prim_type, radius, length, offset, name=None):
    """Generate an MJCF model of two primitives attached to a single free body."""
    mjcf = ET.Element("mujoco", model=name or f"{prim_type}_pair")
    body = ET.SubElement(ET.SubElement(mjcf, "worldbody"), "body")
    for sign in (-0.5, 0.5):
        cx, cy, cz = sign * offset[0], sign * offset[1], sign * offset[2]
        if prim_type == "sphere":
            ET.SubElement(
                body,
                "geom",
                type="sphere",
                pos=f"{cx} {cy} {cz}",
                size=str(radius),
            )
        else:
            ET.SubElement(
                body,
                "geom",
                type=prim_type,
                fromto=f"{cx - 0.5 * length} {cy} {cz} {cx + 0.5 * length} {cy} {cz}",
                size=str(radius),
            )
    ET.SubElement(body, "joint", type="free")
    return mjcf


@pytest.fixture(scope="session")
def side_by_side_capsules():
    return _build_primitive_pair_mjcf("capsule", radius=0.0025, length=0.02, offset=(0.0, 0.0025, 0.0))


@pytest.fixture(scope="session")
def collinear_capsules():
    return _build_primitive_pair_mjcf("capsule", radius=0.0025, length=0.02, offset=(0.02, 0.0, 0.0))


@pytest.fixture(scope="session")
def side_by_side_cylinders():
    return _build_primitive_pair_mjcf("cylinder", radius=0.0025, length=0.02, offset=(0.0, 0.0025, 0.0))


@pytest.fixture(scope="session")
def collinear_cylinders():
    return _build_primitive_pair_mjcf("cylinder", radius=0.0025, length=0.02, offset=(0.02, 0.0, 0.0))


@pytest.fixture(scope="session")
def collinear_spheres():
    return _build_primitive_pair_mjcf("sphere", radius=0.0025, length=0.0, offset=(0.005, 0.0, 0.0))


def _ellipsoid_mjcf_path(tmp_path, semi_axes):
    path = tmp_path / "ellipsoid.xml"
    ET.ElementTree(_ellipsoid_mjcf(semi_axes)).write(path)
    return str(path)


@pytest.mark.parametrize(
    "entity_kind, entity_type, ground_type",
    [
        pytest.param("sphere", "prim", "prim", marks=pytest.mark.required),
        pytest.param("sphere", "prim", "mesh", marks=pytest.mark.required),
        pytest.param("capsule", "prim", "prim", marks=pytest.mark.required),
        pytest.param("capsule", "prim", "mesh", marks=pytest.mark.required),
        pytest.param("cylinder", "prim", "prim", marks=pytest.mark.required),
        pytest.param("cylinder", "prim", "mesh", marks=pytest.mark.required),
        pytest.param("ellipsoid", "prim", "prim", marks=pytest.mark.required),
        pytest.param("ellipsoid", "prim", "mesh", marks=pytest.mark.required),
        ("sphere", "prim", "terrain"),
        ("sphere", "prim", "nonconvex"),
        ("sphere", "mesh", "mesh"),
        ("sphere", "nonconvex", "prim"),
        ("sphere", "nonconvex", "nonconvex"),
        ("sphere", "nonconvex", "plane"),
    ],
)
@pytest.mark.parametrize("gjk_collision", [False, True])
def test_no_drift(gjk_collision, entity_kind, entity_type, ground_type, show_viewer, tmp_path):
    WORLD_TILT_ANGLE = 50.0
    HEIGHT = 0.02
    # The smooth-primitive characteristic length must be small enough to amplify the bias and make drift evident
    SMOOTH_RADIUS = 0.0025
    CYLINDER_HEIGHT = 0.005
    # Smallest semi-axis along body z so the ellipsoid rests on its narrowest cross-section
    ELLIPSOID_SEMI_AXES = (0.0035, 0.0030, SMOOTH_RADIUS)
    BOX_HALF_EXTENT = 0.1
    N_ENVS = 16
    SPHERE_TESSELLATION_SUBDIVISIONS = 3

    # The box and the gravity vector are rotated by the same tilt, which is physically equivalent to the untilted setup.
    tilt_axis = np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0)
    tilt_quat = gu.rotvec_to_quat(math.radians(WORLD_TILT_ANGLE) * tilt_axis)
    R = gu.quat_to_R(tilt_quat)
    box_pos_world = R @ np.array([0.0, 0.0, 0.5 * HEIGHT])
    gravity_world = R @ np.array([0.0, 0.0, -9.81])

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.003,
            gravity=gravity_world,
        ),
        rigid_options=gs.options.RigidOptions(
            use_gjk_collision=gjk_collision,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.25, 0.25, 0.2),
            camera_lookat=(0.0, 0.0, 0.5 * HEIGHT),
            camera_fov=30.0,
        ),
        show_viewer=show_viewer,
    )
    if ground_type in ("mesh", "nonconvex"):
        box_mesh = trimesh.creation.box(extents=(2.0 * BOX_HALF_EXTENT, 2.0 * BOX_HALF_EXTENT, HEIGHT))
        is_ground_convex = ground_type == "mesh"
        box = scene.add_entity(
            morph=gs.morphs.MeshSet(
                files=(box_mesh,),
                pos=box_pos_world,
                quat=tilt_quat,
                convexify=is_ground_convex,
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                smooth=False,
            ),
            visualize_contact=True,
        )
        # Manually overwrite convex flag to forcibly exercise non-convex collision path
        box.geoms[0]._is_convex = is_ground_convex
    elif ground_type == "terrain":
        flat_hf = np.zeros((2, 2), dtype=np.float32)
        terrain_pos_world = R @ np.array([-BOX_HALF_EXTENT, -BOX_HALF_EXTENT, HEIGHT])
        scene.add_entity(
            morph=gs.morphs.Terrain(
                horizontal_scale=2.0 * BOX_HALF_EXTENT,
                vertical_scale=2.0 * BOX_HALF_EXTENT,
                height_field=flat_hf,
                pos=terrain_pos_world,
                quat=tilt_quat,
            ),
            visualize_contact=True,
        )
    elif ground_type == "plane":
        plane_pos_world = R @ np.array([0.0, 0.0, HEIGHT])
        scene.add_entity(
            morph=gs.morphs.Plane(
                pos=plane_pos_world,
                plane_size=(2.0 * BOX_HALF_EXTENT, 2.0 * BOX_HALF_EXTENT),
                quat=tilt_quat,
                fixed=True,
            ),
            visualize_contact=True,
        )
    else:  # if ground_type == "prim":
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=box_pos_world,
                quat=tilt_quat,
                size=(2.0 * BOX_HALF_EXTENT, 2.0 * BOX_HALF_EXTENT, HEIGHT),
                fixed=True,
            ),
            visualize_contact=True,
        )

    if entity_kind == "sphere":
        if entity_kind == "sphere" and entity_type in ("mesh", "nonconvex"):
            sphere_mesh = trimesh.creation.icosphere(
                radius=SMOOTH_RADIUS, subdivisions=SPHERE_TESSELLATION_SUBDIVISIONS
            )
            # Rotate the icosphere so that one face plane is perpendicular to the body -z axis. With the sphere oriented
            # to match the box tilt, this puts that face squarely against the box's top, eliminating the discretization
            # xy shift the sphere would otherwise pick up while rocking onto its nearest supporting feature. We align
            # the face's OUTWARD NORMAL with -z; aligning the centroid direction instead leaves the face plane slightly
            # tilted because for subdivided icosphere faces the centroid is not exactly along the face normal.
            bottom_dir = sphere_mesh.face_normals[int(np.argmin(sphere_mesh.face_normals[:, 2]))]
            cross_axis = np.cross(bottom_dir, np.array([0.0, 0.0, -1.0]))
            sin_t = float(np.linalg.norm(cross_axis))
            if sin_t > 1e-12:
                cross_axis = cross_axis / sin_t
                angle = np.arctan2(sin_t, float(np.dot(bottom_dir, np.array([0.0, 0.0, -1.0]))))
                sphere_mesh.apply_transform(trimesh.transformations.rotation_matrix(angle, cross_axis))
            is_entity_convex = entity_type == "mesh"
            entity = scene.add_entity(
                morph=gs.morphs.MeshSet(
                    files=(sphere_mesh,),
                    convexify=is_entity_convex,
                    decimate=False,
                ),
                vis_mode="collision",
                # visualize_contact=True,
            )
            # Manually overwrite convex flag to forcibly exercise non-convex collision path
            entity.geoms[0]._is_convex = is_entity_convex
        else:
            entity = scene.add_entity(
                morph=gs.morphs.Sphere(
                    radius=SMOOTH_RADIUS,
                ),
            )
    elif entity_kind == "cylinder":
        entity = scene.add_entity(
            morph=gs.morphs.Cylinder(
                radius=SMOOTH_RADIUS,
                height=CYLINDER_HEIGHT,
            ),
        )
    elif entity_kind == "capsule":
        # Two capsule lengths exist as separate entities: the zero-length capsule (sphere-like, used by "vertical-axis"
        # envs because a full-length capsule standing on its cap is a tippy-pencil configuration that is numerically
        # unstable regardless of the bias fix) and the full-length capsule (used by "horizontal-axis" envs, barrel
        # contact). MuJoCo rejects an exact zero length so we use a tiny positive value.
        entity = scene.add_entity(
            morph=(
                gs.morphs.MJCF(
                    file=_capsule_mjcf_path(tmp_path, SMOOTH_RADIUS, gs.EPS, name="capsule_v"),
                ),
                gs.morphs.MJCF(
                    file=_capsule_mjcf_path(tmp_path, SMOOTH_RADIUS, CYLINDER_HEIGHT, name="capsule_h"),
                ),
            )
        )
    else:  # if entity_kind == "ellipsoid":
        entity = scene.add_entity(
            morph=gs.morphs.MJCF(
                file=_ellipsoid_mjcf_path(tmp_path, ELLIPSOID_SEMI_AXES),
            ),
        )
    scene.build(n_envs=N_ENVS)

    # Randomly sample position in local frame.
    # Add small vertical offset to ensure contact at init; otherwise the primitive will sink before bouncing up.
    smooth_xy_local = np.random.uniform(
        low=-(BOX_HALF_EXTENT - 2.0 * SMOOTH_RADIUS),
        high=BOX_HALF_EXTENT - 2.0 * SMOOTH_RADIUS,
        size=(N_ENVS, 2),
    )
    smooth_pos_local = np.concatenate([smooth_xy_local, np.full((N_ENVS, 1), HEIGHT + SMOOTH_RADIUS - 1e-4)], axis=-1)

    # Randomly sample orientation in local frame.
    # Special handling for capsule to ensure stable barrel contact if needed.
    smooth_quat_local = np.random.uniform(low=-1.0, high=1.0, size=(N_ENVS, 4))
    if entity_kind in "cylinder":
        singular_mask = np.ones((N_ENVS,), dtype=np.bool_)
        angle_pitch = 0.5 * np.pi
    elif entity_kind in "ellipsoid":
        singular_mask = np.ones((N_ENVS,), dtype=np.bool_)
        angle_pitch = 0.0
    elif entity_kind == "capsule":
        singular_mask = np.arange(N_ENVS) >= N_ENVS // 2
        angle_pitch = 0.5 * np.pi
    else:
        singular_mask = np.zeros((N_ENVS,), dtype=np.bool_)
        angle_pitch = 0.0
    n_singulars = np.sum(singular_mask)
    angle_yaw = np.random.uniform(low=-np.pi, high=np.pi, size=(n_singulars, 1))
    smooth_quat_local[singular_mask] = gu.xyz_to_quat(
        np.concatenate([np.zeros((n_singulars, 1)), np.full((n_singulars, 1), angle_pitch), angle_yaw], axis=-1),
        rpy=True,
    )

    # Convert pose from local to world frame
    smooth_pos_world = smooth_pos_local @ R.T
    smooth_quat_world = gu.transform_quat_by_quat(smooth_quat_local, np.tile(tilt_quat, (N_ENVS, 1)))

    entity.set_pos(smooth_pos_world)
    entity.set_quat(smooth_quat_world)
    if show_viewer:
        scene.visualizer.update()

    for _ in range(400):
        scene.step()

    pos_local = tensor_to_array(entity.get_pos()) @ R
    # The tolerance must be large enough to accomate small numerical error for mesh-mesh.
    assert_allclose(pos_local[..., :2], smooth_xy_local, atol=1e-3)


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.xfail(reason="De-duplication of repeated contact points is currently too naive for this test to pass...")
@pytest.mark.parametrize("surface_kind", ["primitive_box", "primitive_plane", "vertex_box", "flat_terrain"])
def test_contact_dedup(surface_kind, show_viewer):
    SPHERE_RADIUS = 0.05
    GROUND_SIZE = 1.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
        ),
        show_viewer=show_viewer,
    )
    if surface_kind == "primitive_box":
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=(0.0, 0.0, -0.05),
                size=(GROUND_SIZE, GROUND_SIZE, 0.1),
                fixed=True,
            ),
        )
    elif surface_kind == "primitive_plane":
        scene.add_entity(
            morph=gs.morphs.Plane(
                pos=(0.0, 0.0, 0.0),
            ),
        )
    elif surface_kind == "vertex_box":
        box_mesh = trimesh.creation.box(extents=(GROUND_SIZE, GROUND_SIZE, 0.1))
        scene.add_entity(
            morph=gs.morphs.MeshSet(
                files=(box_mesh,),
                pos=(0.0, 0.0, -0.05),
                fixed=True,
            ),
        )
    elif surface_kind == "flat_terrain":
        flat_hf = np.zeros((16, 16), dtype=np.float32)
        scene.add_entity(
            morph=gs.morphs.Terrain(
                horizontal_scale=0.1,
                vertical_scale=1.0,
                height_field=flat_hf,
                pos=(-0.8, -0.8, 0.0),
            ),
        )
    sphere = scene.add_entity(
        morph=gs.morphs.MeshSet(
            files=(trimesh.creation.icosphere(radius=SPHERE_RADIUS, subdivisions=3),),
            pos=(0.0, 0.0, SPHERE_RADIUS - 1e-4),
            decimate=False,
        ),
        vis_mode="collision",
        visualize_contact=True,
    )
    scene.build()

    for i in range(80):
        scene.step()
        if i == 20:
            sphere.set_dofs_velocity(0.2, dofs_idx_local=sphere.dof_start)
        n_contacts = scene.rigid_solver.collider._collider_state.n_contacts.to_numpy()
        assert np.all(n_contacts == 1), f"Expected 1 contact after dedup, got {n_contacts}"


@pytest.mark.required
@pytest.mark.parametrize("gjk_collision", [True, False])
def test_contact_pruning(gjk_collision, show_viewer):
    GEOM_HALF_SIZE = 0.1
    MARGIN = 1e-4

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
            gravity=(-1.0, -1.0, -1.0),
        ),
        rigid_options=gs.options.RigidOptions(
            # box_box_detection=True,
            use_gjk_collision=gjk_collision,
            contact_pruning_tolerance=0.02,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.4, 0.3, 0.3),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=(0,),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(GEOM_HALF_SIZE, 1.0, 1.0),
            pos=(MARGIN - 1.5 * GEOM_HALF_SIZE, 0.0, 0.0),
            fixed=True,
        ),
        surface=gs.surfaces.Default(
            color=(1, 0, 0, 0.8),
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(1.0, GEOM_HALF_SIZE, 1.0),
            pos=(0.0, MARGIN - 1.5 * GEOM_HALF_SIZE, 0.0),
            fixed=True,
        ),
        surface=gs.surfaces.Default(
            color=(0, 1, 0, 0.8),
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(1.0, 1.0, GEOM_HALF_SIZE),
            pos=(0.0, 0.0, MARGIN - 1.5 * GEOM_HALF_SIZE),
            fixed=True,
        ),
        surface=gs.surfaces.Default(
            color=(0, 0, 1, 0.8),
        ),
    )

    sub_meshes = []
    for sx, sy, sz in product((-1, 0, +1), repeat=3):
        mesh = trimesh.creation.box(extents=(2 / 3 * GEOM_HALF_SIZE,) * 3)
        mesh.apply_translation((2 / 3 * sx * GEOM_HALF_SIZE, 2 / 3 * sy * GEOM_HALF_SIZE, 2 / 3 * sz * GEOM_HALF_SIZE))
        sub_meshes.append(mesh)
    box = scene.add_entity(
        morph=gs.morphs.MeshSet(files=sub_meshes),
        surface=gs.surfaces.Default(
            smooth=False,
        ),
        vis_mode="collision",
        visualize_contact=True,
    )
    scene.build(n_envs=2)

    for step_idx in range(200):
        scene.step()
        # Within each contact-normal bucket, every surviving contact must be a vertex of the 2D convex hull of
        # contacts' positions projected onto the plane perpendicular to that shared normal. The bucket key is the
        # contact's dominant axial direction (this scene's normals are nearly axial, so axis + sign is enough; we
        # don't need to be fully generic). Redundant (interior or hull-edge-midpoint) contacts and >2-collinear
        # contacts both indicate the pruning kernel left work undone.
        contacts = scene.rigid_solver.collider.get_contacts(to_torch=False)
        for i_b in range(scene.n_envs):
            positions = contacts["position"][i_b]
            normals = contacts["normal"][i_b]
            buckets: dict[tuple[int, int], list[int]] = {}
            for i in range(len(positions)):
                axis = int(np.argmax(np.abs(normals[i])))
                sign = 1 if normals[i][axis] > 0 else -1
                buckets.setdefault((axis, sign), []).append(i)
            for key, idxs in buckets.items():
                if len(idxs) < 3:
                    continue
                other_axes = [a for a in range(3) if a != key[0]]
                proj = positions[idxs][:, other_axes].astype(np.float64)
                diam = float(np.linalg.norm(proj.max(axis=0) - proj.min(axis=0)))
                if diam < 1e-6:
                    continue
                try:
                    # Qhull's E tolerance merges nearly-collinear points into hull edges; without it, float noise on
                    # the order of 1e-6 hides the collinearity that the pruning kernel is supposed to detect.
                    hull = ConvexHull(proj, qhull_options=f"Qt E{diam * 1e-3}")
                    n_hull_vertices = len(hull.vertices)
                except QhullError:
                    raise AssertionError(
                        f"step {step_idx}, bucket axis={key[0]} sign={key[1]}: {len(idxs)} contacts are collinear in "
                        f"the contact plane. The pruning kernel should have kept at most 2 of them."
                    ) from None
                if n_hull_vertices == len(idxs):
                    continue
                non_hull = sorted(set(range(len(idxs))) - set(hull.vertices.tolist()))
                details = "\n".join(
                    f"    [{i}] contact={idxs[i]} pos={positions[idxs[i]]} proj={proj[i]}"
                    f"{'  <-- REDUNDANT' if i in non_hull else ''}"
                    for i in range(len(idxs))
                )
                raise AssertionError(
                    f"step {step_idx}, bucket axis={key[0]} sign={key[1]}: {len(idxs)} surviving contacts but only "
                    f"{n_hull_vertices} are vertices of the bucket's 2D convex hull. The pruning kernel should have "
                    f"dropped these {len(idxs) - n_hull_vertices} redundant contact(s):\n{details}"
                )
    assert_allclose(box.get_pos(), 0.0, atol=2e-3)


@pytest.mark.required
@pytest.mark.precision("32")
@pytest.mark.parametrize("gjk_collision", [False, True])
def test_contact_pruning_authored_decomp(gjk_collision, show_viewer):
    # A central pole carries six concentric rings, capped by a ball seated in the top ring's hole. Each ring collision
    # mesh is pre-decomposed into N_WEDGES convex slices, so stacked pieces touch face-to-face along the vertical axis.
    # Physically only vertical contacts are valid between stacked rings; any lateral contact is a spurious cross-sector
    # overlap of the convex decomposition. The ball rests on the curved hole surface, so it legitimately produces angled
    # normals and is exempt from the vertical-normal and one-per-slice checks.
    N_WEDGES = 16
    BASE_HEIGHT = 0.020
    RING_HEIGHT = 0.020
    BALL_HEIGHT = 0.019
    RINGS_ORDER = (0, 1, 2, 3, 5, 4)

    NUM_CHECKS = 10
    POS_TOL = 2e-3
    # FIXME: The top ball is slightly rotating around z-axis (~0.5degree)
    ROT_TOL = 1e-2

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            use_gjk_collision=gjk_collision,
            max_collision_pairs=1200,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.4, 0.0, 0.3),
            camera_lookat=(0.0, 0.0, 0.1),
        ),
        show_viewer=show_viewer,
    )
    plane = scene.add_entity(gs.morphs.Plane())
    pole_pos = (0.0, 0.0, BASE_HEIGHT / 2)
    pole = scene.add_entity(
        morph=gs.morphs.URDF(
            file="tower/base_pole.urdf",
            pos=pole_pos,
            file_meshes_are_zup=True,
        ),
        material=gs.materials.Rigid(
            rho=600.0,
        ),
        vis_mode="collision",
    )
    poss_init = [pole_pos]
    rpys_init = [(0.0, 0.0, 0.0)]
    rings = []
    height = BASE_HEIGHT
    for i, ring_idx in enumerate(RINGS_ORDER):
        ring_pos = (0.0, 0.0, height + (RING_HEIGHT - 1e-4) / 2)
        # Alternate rotational offset along z-axis to avoid lateral contacts
        ring_yaw = 180 / N_WEDGES * (i % 2)
        ring = scene.add_entity(
            morph=gs.morphs.URDF(
                file=f"tower/ring_{ring_idx + 1:02d}.urdf",
                pos=ring_pos,
                euler=(0.0, 0.0, ring_yaw),
                file_meshes_are_zup=True,
            ),
            material=gs.materials.Rigid(
                rho=600.0,
            ),
            vis_mode="collision",
            visualize_contact=True,
        )
        rings.append(ring)
        poss_init.append(ring_pos)
        rpys_init.append((0.0, 0.0, np.deg2rad(ring_yaw)))
        height += RING_HEIGHT - 1e-4
    ball_pos = (0.0, 0.0, height + BALL_HEIGHT)
    ball = scene.add_entity(
        morph=gs.morphs.URDF(
            file="tower/ball.urdf",
            pos=ball_pos,
            file_meshes_are_zup=True,
        ),
        material=gs.materials.Rigid(
            rho=600.0,
        ),
        vis_mode="collision",
    )
    poss_init.append(ball_pos)
    rpys_init.append((0.0, 0.0, 0.0))
    scene.build()

    geom_owner = {geom.idx: entity for entity in (plane, pole, *rings, ball) for geom in entity.geoms}
    ring_geoms = {geom.idx for ring in rings for geom in ring.geoms}
    ball_geoms = {geom.idx for geom in ball.geoms}

    # Tiny warm-up to deal with initial penetration (~5e-4)
    for _ in range(2):
        scene.step()

    # Check that the tower stay in place
    for _ in range(20):
        scene.step()
        for entity, pos_init, rpy_init in zip((pole, *rings, ball), poss_init, rpys_init):
            assert_allclose(entity.get_pos(), pos_init, atol=POS_TOL)
            assert_allclose(gu.quat_to_xyz(entity.get_quat(), rpy=True), rpy_init, atol=ROT_TOL)
        # Only check linear velocity at CoM and angular velocity around z-axis.
        # It is robust to loosing a few contact points while still asserting the failure modes that matter.
        assert_allclose(scene.rigid_solver.get_dofs_velocity(dofs_idx=(0, 1, 2, 5)), 0, tol=0.06)

    # A contact step is "ideal" when both invariants hold across all stacked interfaces (the ball seats on a curved
    # hole and is exempt from both):
    #   - normals are vertical: only axial contacts are physical between stacked rings; a lateral normal is a spurious
    #     cross-sector overlap of the convex decomposition,
    #   - pruning collapses each wedge-pair manifold to one contact per slice, so every pole-ring / ring-ring interface
    #     carries at most N_WEDGES contacts (without pruning each manifold would emit many more).
    # Both invariants fail together on a bad step (a spurious lateral overlap also inflates the slice count). MPR keeps
    # the sub-resolution overlaps below the rejection floor on every step; GJK's tighter penetration estimates let one
    # spike above it occasionally in fp32, so it only has to be ideal at least once.
    for _ in range(NUM_CHECKS):
        scene.step()
        contacts = scene.rigid_solver.collider.get_contacts(to_torch=False)
        geom_a, geom_b = contacts["geom_a"], contacts["geom_b"]
        penetration = contacts["penetration"]
        normal_z = contacts["normal"][:, 2]
        interface_counts = {}
        is_vertical = True
        for i in range(len(geom_a)):
            if penetration[i] <= 0.0:
                continue
            a, b = int(geom_a[i]), int(geom_b[i])
            if a in ball_geoms or b in ball_geoms:
                continue
            if abs(normal_z[i]) < 0.5:
                is_vertical = False
            if a in ring_geoms or b in ring_geoms:
                key = frozenset((geom_owner[a], geom_owner[b]))
                interface_counts[key] = interface_counts.get(key, 0) + 1
        # pole-ring0 plus each ring-ring interface up the stack
        is_pruned = len(interface_counts) == len(rings) and all(
            count <= N_WEDGES for count in interface_counts.values()
        )
        assert is_vertical and is_pruned


@pytest.mark.required
@pytest.mark.precision("32")
@pytest.mark.parametrize("backend", [gs.gpu])
@pytest.mark.parametrize("contact_pruning_tolerance", [0.02, None], ids=["prune", "noprune"])
@pytest.mark.parametrize("prefer_decomposed_solver", [0, 1], ids=["monolith", "decomposed"])
def test_gpu_simulation_determinism(prefer_decomposed_solver, contact_pruning_tolerance, monkeypatch, show_viewer):
    # Run-to-run reproducibility on GPU: from an identical initial state, every trial must reproduce a bit-identical
    # trajectory. CPU is serialized and deterministic by construction, so this targets GPU parallel races only
    # (atomic_add slot reservation, parallel reductions, scheduling). The two registered solve implementations are
    # numerically distinct, so each is pinned via prefer_decomposed_solver (0 -> monolith, 1 -> decomposed) to bypass
    # the perf-dispatch autotuner, whose timing-based choice between them is a separate nondeterminism source; this
    # isolates physics-kernel determinism per variant.
    #
    # The authored-decomposition tower is the stress case: stacked rings pre-split into convex wedges produce many
    # multi-contact manifolds per geom pair, exercising the narrowphase, contact pruning, the contact sort, and the
    # contact-coupled solve. The per-step fingerprints are compared in pipeline order so the assertion names the
    # earliest diverging stage, pinpointing the root:
    #   - contact set    -> narrowphase / pruning
    #   - contact order  -> contact sort
    #   - dofs velocity  -> constraint solve
    from genesis.utils.array_class import RigidSimStaticConfig

    init_orig = RigidSimStaticConfig.__init__

    def init_forced(self, *args, **kwargs):
        kwargs["prefer_decomposed_solver"] = prefer_decomposed_solver
        init_orig(self, *args, **kwargs)

    monkeypatch.setattr(RigidSimStaticConfig, "__init__", init_forced)

    N_TRIALS = 8
    N_STEPS = 25
    N_WEDGES = 16
    BASE_HEIGHT = 0.020
    RING_HEIGHT = 0.020
    BALL_HEIGHT = 0.019
    RINGS_ORDER = (0, 1, 2, 3, 5, 4)

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            use_gjk_collision=True,
            contact_pruning_tolerance=contact_pruning_tolerance,
            max_collision_pairs=1200,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        morph=gs.morphs.URDF(
            file="tower/base_pole.urdf",
            pos=(0.0, 0.0, BASE_HEIGHT / 2),
            file_meshes_are_zup=True,
        ),
        material=gs.materials.Rigid(rho=600.0),
    )
    height = BASE_HEIGHT
    for i, ring_idx in enumerate(RINGS_ORDER):
        scene.add_entity(
            morph=gs.morphs.URDF(
                file=f"tower/ring_{ring_idx + 1:02d}.urdf",
                pos=(0.0, 0.0, height + (RING_HEIGHT - 1e-4) / 2),
                # Alternate rotational offset along z-axis to avoid lateral contacts
                euler=(0.0, 0.0, 180 / N_WEDGES * (i % 2)),
                file_meshes_are_zup=True,
            ),
            material=gs.materials.Rigid(rho=600.0),
        )
        height += RING_HEIGHT - 1e-4
    ball = scene.add_entity(
        morph=gs.morphs.URDF(
            file="tower/ball.urdf",
            pos=(0.0, 0.0, height + BALL_HEIGHT),
            file_meshes_are_zup=True,
        ),
        material=gs.materials.Rigid(rho=600.0),
    )
    scene.build()
    solver = scene.rigid_solver

    # The ball is a sphere seated in the top ring's hole, so every ball contact normal must point radially
    ball_geoms_idx = {geom.idx for geom in ball.geoms}
    ball_center = np.atleast_2d(tensor_to_array(ball.get_pos()))[0]
    solver.collider.detection()
    contacts = solver.collider.get_contacts(to_torch=False)
    geom_a, geom_b = contacts["geom_a"], contacts["geom_b"]
    position, normal, penetration = contacts["position"], contacts["normal"], contacts["penetration"]
    for i in range(len(geom_a)):
        if penetration[i] <= 0.0 or (geom_a[i] not in ball_geoms_idx and geom_b[i] not in ball_geoms_idx):
            continue
        radial = ball_center - position[i]
        radial /= np.linalg.norm(radial)
        cos_angle = min(1.0, abs(np.dot(normal[i], radial)))
        assert np.degrees(np.arccos(cos_angle)) < 15.0

    # trials[trial][step] = (contact_set, contact_order, dofs_velocity, dofs_position)
    trials = []
    for _ in range(N_TRIALS):
        scene.reset()
        steps = []
        for _ in range(N_STEPS):
            scene.step()
            contacts = solver.collider.get_contacts(to_torch=False)
            geom_a, geom_b = contacts["geom_a"], contacts["geom_b"]
            position, normal, penetration = contacts["position"], contacts["normal"], contacts["penetration"]
            contact_order = tuple(
                (geom_a[i], geom_b[i], *position[i], *normal[i], penetration[i]) for i in range(len(geom_a))
            )
            dofs_velocity = tensor_to_array(solver.get_dofs_velocity()).copy()
            dofs_position = tensor_to_array(solver.get_qpos()).copy()
            steps.append((frozenset(contact_order), contact_order, dofs_velocity, dofs_position))
        trials.append(steps)

    ref = trials[0]
    for trial in range(1, N_TRIALS):
        for step in range(N_STEPS):
            ref_set, ref_order, ref_vel, ref_pos = ref[step]
            cur_set, cur_order, cur_vel, cur_pos = trials[trial][step]
            assert cur_set == ref_set
            assert cur_order == ref_order
            assert_equal(cur_vel, ref_vel)
            assert_equal(cur_pos, ref_pos)


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize(
    "model_name",
    [
        "side_by_side_capsules",
        "collinear_capsules",
        "side_by_side_cylinders",
        "collinear_cylinders",
        "collinear_spheres",
    ],
)
def test_contact_pruning_degenerated_hull(model_name, xml_path, show_viewer):
    HEIGHT = 0.02
    BOX_HALFSIZE = 0.15
    PRIM_RADIUS = 0.0025
    PRIM_LENGTH = 0.02
    N_ENVS = 16

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.004,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.25, 0.25, 0.2),
            camera_lookat=(0.0, 0.0, 0.5 * HEIGHT),
            camera_fov=30.0,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(2 * BOX_HALFSIZE, 2 * BOX_HALFSIZE, HEIGHT),
            pos=(0.0, 0.0, 0.5 * HEIGHT),
            fixed=True,
        ),
        visualize_contact=True,
    )
    entity = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=xml_path,
        ),
        surface=gs.surfaces.Default(
            smooth=False,
        ),
    )
    scene.build(n_envs=N_ENVS)

    # Randomly sample position in local frame.
    # Add small vertical offset to ensure contact at init; otherwise the primitive will sink before bouncing up.
    smooth_xy = np.random.uniform(
        low=-(BOX_HALFSIZE - 2.0 * PRIM_LENGTH), high=BOX_HALFSIZE - 2.0 * PRIM_LENGTH, size=(N_ENVS, 2)
    )
    smooth_pos = np.concatenate([smooth_xy, np.full((N_ENVS, 1), HEIGHT + PRIM_RADIUS - 1e-4)], axis=-1)
    entity.set_pos(smooth_pos)

    # Random yaw about world z; capsules/cylinders stay horizontal since their fromto axis lies in the body xy plane.
    angle_yaw = np.random.uniform(low=-np.pi, high=np.pi, size=(N_ENVS, 1))
    smooth_quat = gu.xyz_to_quat(np.concatenate([np.zeros((N_ENVS, 2)), angle_yaw], axis=-1), rpy=True)
    entity.set_quat(smooth_quat)

    if show_viewer:
        scene.visualizer.update()

    for _ in range(20):
        scene.step()
    for _ in range(300):
        scene.step()
        n_contacts = scene.rigid_solver.collider._collider_state.n_contacts.to_numpy()
        assert n_contacts.all()
        if model_name.startswith("side_by_side"):
            assert (n_contacts == 4).all()
        elif model_name == "collinear_spheres":
            assert (n_contacts == 2).all()

    assert_allclose(entity.get_pos()[..., :2], smooth_xy, atol=1e-3)


@pytest.mark.debug(False)  # Disable debug for speedup
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
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            max_collision_pairs=1000,
            box_box_detection=box_box_detection,
            use_gjk_collision=gjk_collision,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(10, 10, 10),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        gs.morphs.Plane(),
    )
    for n in range(5**3):
        i, j, k = int(n / 25), int(n / 5) % 5, n % 5
        scene.add_entity(
            gs.morphs.Box(
                pos=(i * (1.0 - 1e-3), j * (1.0 - 1e-3), k * (1.0 - 1e-3) + 0.5),
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
    num_steps = 850 if dynamics else 150
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
            qpos0 = np.array((i * (1.0 - 1e-3), j * (1.0 - 1e-3), k * (1.0 - 1e-3) + 0.5))
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
    gs_sim.rigid_solver.dofs_state.ctrl_mode.fill(int(gs.CTRL_MODE.FORCE))
    gs_sim.rigid_solver._enable_collision = False
    gs_sim.rigid_solver._enable_joint_limit = False
    gs_sim.rigid_solver._disable_constraint = True
    gs_sim.rigid_solver.collider.clear()
    gs_sim.rigid_solver.constraint_solver.clear()

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


@pytest.mark.slow  # ~200s
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


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_filter_neutral_self_collisions(show_viewer):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=True,
            enable_neutral_collision=False,
            enable_adjacent_collision=False,
        ),
        show_viewer=show_viewer,
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.08,
        ),
        surface=gs.surfaces.Default(
            color=(0.0, 2.0, 0.0, 1.0),
        ),
    )
    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.0, 0.0, 1.0),
        ),
    )
    sphere.attach(robot, "hand")
    scene.build()
    eq_type = scene.rigid_solver.equalities_info.eq_type.to_numpy()[: scene.rigid_solver.n_equalities, 0]
    eq_obj1id = scene.rigid_solver.equalities_info.eq_obj1id.to_numpy()[: scene.rigid_solver.n_equalities, 0]
    eq_obj2id = scene.rigid_solver.equalities_info.eq_obj2id.to_numpy()[: scene.rigid_solver.n_equalities, 0]

    scene.rigid_solver.collider.detection()
    contacts_data = scene.rigid_solver.collider.get_contacts()
    assert ((contacts_data["link_a"] == 12) & (contacts_data["link_b"] == 0)).any()

    for i in range(2):
        for i_ga in range(robot.geom_start, box.geom_start):
            for i_gb in range(i_ga + 1, box.geom_start):
                geom_a = scene.rigid_solver.geoms[i_ga]
                geom_b = scene.rigid_solver.geoms[i_gb]
                link_a = geom_a.link
                link_b = geom_b.link

                if link_a.idx == link_b.idx:
                    continue

                if link_a.is_fixed and link_b.is_fixed:
                    continue

                if (
                    (eq_type == gs.EQUALITY_TYPE.WELD)
                    & (
                        (eq_obj1id == link_a.idx & eq_obj2id == link_b.idx)
                        | (eq_obj1id == link_b.idx & eq_obj2id == link_a.idx)
                    )
                ).any():
                    continue

                is_adjacent = False
                link = link_b
                while link.parent_idx > 0:
                    if link.parent_idx == link_a.idx:
                        is_adjacent = True
                        break
                    if not all(joint.type is gs.JOINT_TYPE.FIXED for joint in link.joints):
                        break
                    link = scene.rigid_solver.links[link.parent_idx]
                if is_adjacent:
                    continue

                verts_a = tensor_to_array(geom_a.get_verts())
                verts_a = (1.0 - 1e-3) * verts_a + 1e-3 * verts_a.mean(axis=0, keepdims=True)
                mesh_a = trimesh.Trimesh(vertices=verts_a, faces=geom_a.init_faces, process=False)
                geom_b = scene.rigid_solver.geoms[i_gb]
                verts_b = tensor_to_array(geom_b.get_verts())
                verts_b = (1.0 - 1e-3) * verts_b + 1e-3 * verts_b.mean(axis=0, keepdims=True)
                mesh_b = trimesh.Trimesh(vertices=verts_b, faces=geom_b.init_faces, process=False)
                is_colliding = mesh_a.contains(mesh_b.vertices).any() or mesh_b.contains(mesh_a.vertices).any()
                assert is_colliding == ({(i_ga, i_gb)} in ({(5, 10)}, {(6, 10)}, {(11, 23)}, {(17, 23)}))
        scene.step()


@pytest.mark.slow  # ~200s
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
def test_position_control(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            substeps=1,  # This is essential to be able to emulate native PD control
        ),
        rigid_options=gs.options.RigidOptions(
            batch_links_info=True,
            batch_dofs_info=True,
            disable_constraint=True,
            integrator=gs.integrator.approximate_implicitfast,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
    )
    scene.build(n_envs=2, env_spacing=(1.0, 1.0))

    MOTORS_POS_TARGET = torch.tensor(
        [0.6900, -0.1100, -0.7200, -2.7300, -0.1500, 2.6400, 0.8900, 0.0400, 0.0400],
        dtype=gs.tc_float,
        device=gs.device,
    )
    MOTORS_VEL_TARGET = torch.rand_like(MOTORS_POS_TARGET)
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

    # FIXME: We do NOT raise exception anymore when setting control targets that would have no effect
    # robot.set_dofs_kp(torch.zeros_like(MOTORS_KP), envs_idx=0)
    # robot.set_dofs_kv(torch.zeros_like(MOTORS_KD), envs_idx=0)
    # with pytest.raises(gs.GenesisException):
    #     robot.control_dofs_position(MOTORS_POS_TARGET, envs_idx=0)
    # with pytest.raises(gs.GenesisException):
    #     robot.control_dofs_position_velocity(MOTORS_POS_TARGET, MOTORS_VEL_TARGET, envs_idx=0)
    # with pytest.raises(gs.GenesisException):
    #     robot.control_dofs_velocity(MOTORS_VEL_TARGET, envs_idx=0)
    # robot.set_dofs_kv(MOTORS_KD, envs_idx=0)
    # robot.control_dofs_velocity(MOTORS_VEL_TARGET, envs_idx=0)
    # with pytest.raises(gs.GenesisException):
    #     robot.control_dofs_position(MOTORS_POS_TARGET, envs_idx=0)
    # robot.control_dofs_position_velocity(MOTORS_POS_TARGET, MOTORS_VEL_TARGET, envs_idx=0)

    robot.set_dofs_kp(MOTORS_KP, envs_idx=0)
    robot.set_dofs_kv(MOTORS_KD, envs_idx=0)
    robot.control_dofs_position(MOTORS_POS_TARGET, envs_idx=0)
    robot.control_dofs_position_velocity(MOTORS_POS_TARGET, MOTORS_VEL_TARGET, envs_idx=0)

    # Must update DoF armature to emulate implicit damping for force control.
    # This is equivalent to the first-order correction term involved in implicit integration scheme,
    # in the particular case where `approximate_implicitfast` integrator is used.
    # Note that the low-level internal API is used because invweights must NOT be updated, otherwise
    # the test cannot pass. This is unecessary and not recommended for practical applications.
    # robot.set_dofs_armature(robot.get_dofs_armature(envs_idx=1) + MOTORS_KD * scene.sim._substep_dt, envs_idx=1)
    dofs_armature = scene.rigid_solver.dofs_info.armature.to_numpy()
    dofs_armature[:, 1] += tensor_to_array(MOTORS_KD * scene.sim._substep_dt)
    scene.rigid_solver.dofs_info.armature.from_numpy(dofs_armature)

    force_range = qd_to_torch(scene.rigid_solver.dofs_info.force_range)
    for i in range(200):
        dofs_pos = robot.get_qpos(envs_idx=1)
        dofs_vel = robot.get_dofs_velocity(envs_idx=1)
        dofs_torque = MOTORS_KP * (MOTORS_POS_TARGET - dofs_pos) + MOTORS_KD * (MOTORS_VEL_TARGET - dofs_vel)
        dofs_torque.clamp_(force_range[:, 1, 0], force_range[:, 1, 1])
        robot.control_dofs_force(dofs_torque, envs_idx=1)
        scene.step()
        qf_applied = scene.rigid_solver.dofs_state.qf_applied.to_numpy().T
        # dofs_torque = robot.get_dofs_control_force()
        assert_allclose(qf_applied[1], dofs_torque, tol=1e-6)
        assert_allclose(qf_applied[0], qf_applied[1], tol=1e-6)

    A = 0.1
    f = 1.0
    scene.reset()
    robot.set_dofs_kp(MOTORS_KP, envs_idx=1)
    robot.set_dofs_kv(MOTORS_KD, envs_idx=1)
    force_range[:, 1, 0] = float("-inf")
    force_range[:, 1, 1] = float("+inf")
    scene.rigid_solver.dofs_info.force_range.from_numpy(tensor_to_array(force_range))
    for i in range(1000):
        t = scene.t * scene.dt
        pos_target = A * np.sin(2 * np.pi * f * t)
        vel_target = A * 2 * np.pi * f * np.cos(2 * np.pi * f * t)
        robot.control_dofs_position_velocity(torch.full((9,), pos_target), torch.full((9,), vel_target), envs_idx=1)
        scene.step()
        assert_allclose(pos_target, robot.get_dofs_position(envs_idx=1), tol=1e-2)


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize("batch_fixed_verts", [False, True])
@pytest.mark.parametrize("relative", [False, True])
def test_set_root_pose(batch_fixed_verts, relative, show_viewer, tol):
    ROBOT_POS_ZERO = (0.0, 0.4, 0.1)
    ROBOT_EULER_ZERO = (0.0, 0.0, 90.0)
    CUBE_POS_ZERO = (0.65, 0.0, 0.02)
    CUBE_EULER_ZERO = (0.0, 90.0, 0.0)

    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            offset_pos=ROBOT_POS_ZERO,
            offset_euler=ROBOT_EULER_ZERO,
            batch_fixed_verts=batch_fixed_verts,
        ),
    )
    sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.04,
            batch_fixed_verts=False,
            fixed=True,
        ),
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            offset_pos=CUBE_POS_ZERO,
            offset_euler=CUBE_EULER_ZERO,
        ),
    )
    plain_box = scene.add_entity(
        gs.morphs.Box(
            pos=(2.0, 0.0, 0.2),
            size=(0.04, 0.04, 0.04),
        ),
    )
    POSED_BOX_POS = (2.0, 0.5, 0.3)
    POSED_BOX_OFFSET_EULER = (0.0, 0.0, 45.0)
    posed_box = scene.add_entity(
        gs.morphs.Box(
            pos=POSED_BOX_POS,
            size=(0.04, 0.04, 0.04),
            offset_pos=(0.0, 0.0, 0.5),
            offset_euler=POSED_BOX_OFFSET_EULER,
        ),
    )
    scene.build(n_envs=2)

    # A no-offset entity reports the same pose in the user and world frames.
    assert_allclose(plain_box.get_pos(relative=True), plain_box.get_pos(relative=False), tol=tol)
    assert_allclose(plain_box.get_pos(), (2.0, 0.0, 0.2), tol=tol)

    # With both a morph pose and an offset, the relative getter returns the morph pose while the world getter carries
    # the offset composed onto it (the offset position adds in z since the user orientation is identity).
    assert_allclose(posed_box.get_pos(relative=True), POSED_BOX_POS, tol=tol)
    assert_allclose(posed_box.get_quat(relative=True), gu.identity_quat(), tol=tol)
    assert_allclose(posed_box.get_pos(relative=False), (2.0, 0.5, 0.8), tol=tol)
    assert_allclose(
        posed_box.get_quat(relative=False),
        gu.xyz_to_quat(np.array(POSED_BOX_OFFSET_EULER), rpy=True, degrees=True),
        tol=tol,
    )

    # Setting the orientation in the user frame keeps the user-frame position fixed: the offset position rotates with
    # the orientation, so the world position is rewritten to preserve the reported relative position. Rotating about x
    # while the offset position is along z makes that offset contribution change, exercising the rewrite.
    new_quat = gu.xyz_to_quat(np.array((90.0, 0.0, 0.0)), rpy=True, degrees=True)
    posed_box.set_quat(new_quat, relative=True)
    assert_allclose(posed_box.get_pos(relative=True), POSED_BOX_POS, tol=tol)
    assert_allclose(posed_box.get_quat(relative=True), new_quat, tol=tol)

    robot_aabb_init, robot_base_aabb_init = robot.get_AABB(), robot.geoms[0].get_AABB()
    cube_aabb_init, cube_base_aabb_init = cube.get_AABB(), cube.geoms[0].get_AABB()

    # Make sure that it is not possible to end up in an inconsistent state for fixed geometries. These place entities
    # at absolute world positions, so they bypass the pose offset (relative=False).
    pos_delta = np.random.rand(2, 3)
    with nullcontext() if batch_fixed_verts else pytest.raises(gs.GenesisException):
        robot.set_pos(pos_delta, relative=False)
        if show_viewer:
            scene.visualizer.update()
    with nullcontext() if batch_fixed_verts else pytest.raises(gs.GenesisException):
        robot.set_pos(pos_delta[[0]], envs_idx=[0], relative=False)
        if show_viewer:
            scene.visualizer.update()
    cube.set_pos(pos_delta[[0]] + (0.0, 0.0, 0.16), envs_idx=[0], relative=False)
    cube.set_pos(pos_delta[[1]] + (0.0, 0.0, 0.11), envs_idx=[1], relative=False)
    sphere.set_pos(np.tile(pos_delta[[0]], (2, 1)) + 1.0, relative=False)
    quat_delta = np.random.rand(2, 4)
    with nullcontext() if batch_fixed_verts else pytest.raises(gs.GenesisException):
        robot.set_quat(quat_delta, relative=False)
        if show_viewer:
            scene.visualizer.update()
    with nullcontext() if batch_fixed_verts else pytest.raises(gs.GenesisException):
        robot.set_quat(quat_delta[[0]], envs_idx=[0], relative=False)
        if show_viewer:
            scene.visualizer.update()
    cube.set_quat(quat_delta, relative=False)
    if show_viewer:
        scene.visualizer.update()

    sphere_aabb, sphere_base_aabb = sphere.get_AABB(), sphere.geoms[0].get_AABB()
    assert_allclose(sphere_aabb.mean(dim=-2), pos_delta[0] + 1.0, tol=tol)
    assert_allclose(sphere_aabb, sphere_base_aabb, tol=tol)

    # Simulate for a while to check if the dynamic object is colliding with the static one
    if batch_fixed_verts:
        has_collided = torch.tensor([False, False], dtype=torch.bool, device=gs.device)
        for _ in range(20):
            scene.step()
            contacts_state = cube.get_contacts(with_entity=robot, exclude_self_contact=True)
            has_collided |= contacts_state["valid_mask"].any(dim=-1)
            if has_collided.all():
                break
        else:
            raise AssertionError("Cube never collided with robot for at least one of the environments.")

    for _ in range(2):
        scene.reset()

        for entity, pos_zero, euler_zero, entity_aabb_init, base_aabb_init in (
            (robot, ROBOT_POS_ZERO, ROBOT_EULER_ZERO, robot_aabb_init, robot_base_aabb_init),
            (cube, CUBE_POS_ZERO, CUBE_EULER_ZERO, cube_aabb_init, cube_base_aabb_init),
        ):
            pos_zero = torch.tensor(pos_zero, device=gs.device, dtype=gs.tc_float)
            euler_zero = torch.deg2rad(torch.tensor(euler_zero, dtype=gs.tc_float))
            quat_zero = gu.xyz_to_quat(euler_zero, rpy=True)
            # The pose lives in the offset, so the world frame (relative=False) carries it; the user frame is identity.
            assert_allclose(entity.get_pos(relative=False), pos_zero, tol=tol)
            assert_allclose(entity.get_pos(relative=True), 0.0, tol=tol)
            # Use quaternion for comparison to avoid gymbal lock issue in euler angles
            quat = entity.get_quat(relative=False)
            assert_allclose(quat, quat_zero, tol=tol)
            base_aabb = entity.geoms[0].get_AABB()
            assert base_aabb.shape == ((2, 2, 3) if not entity.geoms[0].is_fixed or batch_fixed_verts else (2, 3))
            assert_allclose(base_aabb, base_aabb_init, tol=tol)
            assert_allclose(entity.get_AABB(), entity_aabb_init, tol=tol)

            pos_delta = torch.as_tensor(np.random.rand(3), dtype=gs.tc_float, device=gs.device).expand((2, 3))
            entity.set_pos(pos_delta, relative=relative)

            pos_ref = pos_delta + pos_zero if relative else pos_delta
            # Round-trip in the frame it was set in: the getter must report back exactly what set_pos received.
            assert_allclose(entity.get_pos(relative=relative), pos_delta, tol=tol)
            assert_allclose(entity.geoms[0].get_AABB(), base_aabb_init + (pos_ref - pos_zero), tol=tol)
            assert_allclose(entity.get_AABB(), entity_aabb_init + (pos_ref - pos_zero), tol=tol)

            quat_delta = torch.tile(torch.as_tensor(np.random.rand(4), dtype=gs.tc_float, device=gs.device), (2, 1))
            quat_delta /= torch.linalg.norm(quat_delta, axis=1, keepdim=True)
            entity.set_quat(quat_delta, relative=relative)
            assert_allclose(entity.get_quat(relative=relative), quat_delta, tol=tol)


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_normalized_quat(show_viewer, tol):
    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
        ),
    )
    scene.build()

    # Make sure that the simulation state is not sensitive to qpos normalization
    quat = torch.randn((4,), dtype=gs.tc_float, device=gs.device)

    qpos = robot.get_qpos()
    qpos[3:7] = quat / torch.linalg.norm(quat)
    robot.set_qpos(qpos)
    scene.step()
    qpos_post = robot.get_qpos()
    assert_allclose(torch.linalg.norm(qpos_post[3:7]), 1.0, tol=tol)

    qpos[3:7] = quat
    scene.reset()
    robot.set_qpos(qpos)
    # assert_allclose(qpos, robot.get_qpos(), tol=tol)  # True, but not specification requirement
    scene.step()
    assert_allclose(qpos_post, robot.get_qpos(), tol=tol)

    scene.reset()
    robot.set_quat(quat)
    # assert_allclose(quat, qpos[3:7], tol=tol)  # True, but not specification requirement
    scene.step()
    assert_allclose(qpos_post, robot.get_qpos(), tol=tol)

    # Make sure that entity, link and geom quaternions are normalized.
    # "RigidEntity.set_quat" is calling 'kernel_forward_kinematics_links_geoms', which is relying on
    # 'func_update_cartesian_space' under the hood.
    # Let's check that everything is properly normalized at this stage already. If so, it means that all quaternions of
    # interest are guaranteed to be always normalized, since 'func_update_cartesian_space' is called internally during
    # forward dynamics 'step_1' at the very beginning of 'RigidSolver.step'.
    scene.reset()
    robot.set_quat(quat)
    assert_allclose(torch.linalg.norm(robot.get_quat()), 1.0, tol=tol)
    for link in robot.links:
        assert_allclose(torch.linalg.norm(link.get_quat()), 1.0, tol=tol)
    for geom in robot.geoms:
        assert_allclose(torch.linalg.norm(geom.get_quat()), 1.0, tol=tol)
    assert_allclose(torch.linalg.norm(scene.rigid_solver.get_links_quat(), dim=-1), 1.0, tol=tol)
    assert_allclose(torch.linalg.norm(scene.rigid_solver.get_geoms_quat(), dim=-1), 1.0, tol=tol)


@pytest.mark.slow  # ~200s
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
            obj.set_sol_params([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
            assert_allclose(obj.sol_params, [2.0e-02, 0.5, 1e-4, 1e-4, 0.0, 1e-4, 1.0], tol=tol)


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

    # Make sure that the simulation is deterministic
    (gs_robot,) = gs_sim.entities
    gs_sim.scene.reset()
    gs_sim.scene.step()
    dofs_vel = gs_robot.get_dofs_velocity()
    for _ in range(50):
        gs_sim.scene.reset()
        gs_sim.scene.step()
        assert_equal(gs_robot.get_dofs_velocity(), dofs_vel)

    # Run the simulation for a while
    qvel_norminf_all = []
    for i in range(750):
        gs_sim.scene.step()
        if i > 700:
            (gs_robot,) = gs_sim.entities
            qvel = gs_robot.get_dofs_velocity()
            qvel_norminf = torch.linalg.norm(qvel, ord=math.inf)
            qvel_norminf_all.append(qvel_norminf)
    assert_allclose(torch.quantile(torch.stack(qvel_norminf_all, dim=0), 0.5), 0.0, tol=0.1)

    qpos = gs_robot.get_dofs_position()
    assert torch.linalg.norm(qpos[:2]) < 1.3
    body_z = gs_sim.rigid_solver.links_state.pos.to_numpy()[:-1, 0, 2]
    np.testing.assert_array_less(0, body_z + gs.EPS)


@pytest.mark.slow("gpu")  # gpu ~250s
@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_inverse_kinematics_multilink(show_viewer, tol):
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
    # Add one extra entity, just to make sure there is no idx offset issues
    scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(0.0, 0.2, 0.05),
        ),
    )
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/shadow_hand/shadow_hand.urdf",
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
        pos_tol=tol,
        rot_tol=tol,
        max_solver_iters=100,
        return_error=True,
    )
    assert qpos.shape == (1, robot.n_qs)
    assert err.shape == (1, 3, 6)
    assert_allclose(err, 0.0, atol=tol)

    robot.set_qpos(qpos, envs_idx=(1,))
    if show_viewer:
        scene.visualizer.update(force=True)
    assert_allclose(index_finger_distal.get_pos(envs_idx=(1,)), index_finger_pos, tol=tol)
    assert_allclose(middle_finger_distal.get_pos(envs_idx=(1,)), middle_finger_pos, tol=tol)
    assert_allclose(wrist.get_pos(envs_idx=(1,)), wrist_pos, tol=tol)


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_inverse_kinematics_local_point(n_envs, show_viewer, tol):
    """Test IK with local_point parameter - positions an offset point at the target instead of link origin."""

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, 0.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
    )
    robot = scene.add_entity(
        morph=gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    scene.build(n_envs=n_envs)

    end_effector = robot.get_link("hand")

    # Define a local offset point in the end-effector frame (e.g., 10cm along Z-axis)
    local_offset = torch.tensor([0.0, 0.0, 0.1], dtype=gs.tc_float, device=gs.device)

    # Create different target positions and quaternions for each environment
    num_envs = max(n_envs, 1)
    target_pos_base = torch.tensor(
        [[0.5, 0.2, 0.4], [0.45, 0.15, 0.35], [0.55, 0.25, 0.45]], dtype=gs.tc_float, device=gs.device
    )[:num_envs]
    target_quat_base = torch.tensor(
        [[0.0, 1.0, 0.0, 0.0], [0.0, 0.9239, 0.3827, 0.0], [0.0, 0.9239, -0.3827, 0.0]],
        dtype=gs.tc_float,
        device=gs.device,
    )[:num_envs]

    # Handle different shapes based on n_envs
    if n_envs > 0:
        target_pos = target_pos_base
        target_quat = target_quat_base
    else:
        target_pos = target_pos_base[0]
        target_quat = target_quat_base[0]

    # Solve IK with local_point (local_offset stays 1D - it gets broadcast internally)
    qpos, err = robot.inverse_kinematics(
        link=end_effector,
        pos=target_pos,
        quat=target_quat,
        local_point=local_offset,
        pos_tol=tol,
        rot_tol=tol,
        max_solver_iters=100,
        return_error=True,
    )
    assert_allclose(err, 0.0, atol=tol)

    # Apply the solution
    robot.set_qpos(qpos)

    # Verify the offset point is at the target position
    link_pos = end_effector.get_pos()
    link_quat = end_effector.get_quat()

    # Transform local offset to world frame
    world_offset = gu.transform_by_quat(local_offset, link_quat)
    actual_point_pos = link_pos + world_offset

    # Check that the offset point reached the target
    assert_allclose(actual_point_pos, target_pos, tol=tol)

    # Also verify via forward kinematics
    links_pos, links_quat = robot.forward_kinematics(qpos)

    # Handle indexing based on n_envs
    if n_envs > 0:
        fk_link_pos = links_pos[:, end_effector.idx_local]
        fk_link_quat = links_quat[:, end_effector.idx_local]
    else:
        fk_link_pos = links_pos[end_effector.idx_local]
        fk_link_quat = links_quat[end_effector.idx_local]

    fk_world_offset = gu.transform_by_quat(local_offset, fk_link_quat)
    fk_actual_point_pos = fk_link_pos + fk_world_offset
    assert_allclose(fk_actual_point_pos, target_pos, tol=tol)

    if show_viewer:
        scene.visualizer.update()


@pytest.mark.slow("gpu")  # gpu ~250s
@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_inverse_kinematics_multilink_local_points(show_viewer, tol):
    """Test multi-link IK with local_points parameter."""

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, 0.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
    )
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/shadow_hand/shadow_hand.urdf",
        ),
    )
    scene.build()

    index_finger = robot.get_link("index_finger_distal")
    middle_finger = robot.get_link("middle_finger_distal")

    # Different local offsets for each finger (e.g., fingertip positions)
    index_local_offset = torch.tensor([0.0, 0.0, 0.02], dtype=gs.tc_float, device=gs.device)
    middle_local_offset = torch.tensor([0.0, 0.0, 0.02], dtype=gs.tc_float, device=gs.device)

    # Target positions for the fingertips
    index_target = torch.tensor([0.6, 0.5, 0.2], dtype=gs.tc_float, device=gs.device)
    middle_target = torch.tensor([0.63, 0.5, 0.2], dtype=gs.tc_float, device=gs.device)

    # Solve multi-link IK with local_points
    qpos, err = robot.inverse_kinematics_multilink(
        links=[index_finger, middle_finger],
        poss=[index_target, middle_target],
        local_points=[index_local_offset, middle_local_offset],
        pos_tol=tol,
        rot_tol=tol,
        max_solver_iters=100,
        return_error=True,
    )
    assert_allclose(err, 0.0, atol=tol)

    # Apply solution
    robot.set_qpos(qpos)
    if show_viewer:
        scene.visualizer.update(force=True)

    # Verify each offset point is at its target
    for link, local_offset, target in [
        (index_finger, index_local_offset, index_target),
        (middle_finger, middle_local_offset, middle_target),
    ]:
        link_pos = link.get_pos()
        link_quat = link.get_quat()
        world_offset = gu.transform_by_quat(local_offset, link_quat)
        actual_point_pos = link_pos + world_offset
        assert_allclose(actual_point_pos, target, tol=tol)


@pytest.mark.slow  # ~250s
@pytest.mark.required
def test_multi_robot_inverse_kinematics(show_viewer, tol):
    scene = gs.Scene(show_viewer=show_viewer)
    scene.add_entity(gs.morphs.Plane())

    robot_positions = [
        (0.0, -0.5, 0.005),
        (0.0, 0.0, 0.005),
        (0.0, 0.5, 0.005),
    ]
    robots: list[RigidEntity] = []
    for pos in robot_positions:
        robot = scene.add_entity(
            gs.morphs.MJCF(
                file="xml/franka_emika_panda/panda_non_overlap.xml",
                pos=pos,
                convexify=True,
            ),
        )
        robots.append(robot)

    scene.build()

    for robot, pos in zip(robots, robot_positions):
        target_pos = np.array(pos) + [0.4, 0.0, 0.4]
        qpos, err = robot.inverse_kinematics(
            link=robot.get_link("hand"),
            pos=target_pos,
            quat=[0, 1, 0, 0],
            pos_tol=tol,
            rot_tol=tol,
            max_solver_iters=100,
            return_error=True,
        )
        assert_allclose(err, 0.0, atol=tol)
        robot.set_qpos(qpos)
        ee_pos = robot.get_link("hand").get_pos()
        assert_allclose(target_pos, ee_pos, atol=tol)


@pytest.mark.slow("gpu")  # gpu ~300s
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

    num_waypoints = 300
    if n_envs == 0:
        free_path, return_valid_mask = franka.plan_path(
            qpos_goal=qpos_goal,
            num_waypoints=num_waypoints,
            resolution=0.05,
            ignore_collision=True,
            return_valid_mask=True,
        )
    else:
        return_valid_mask = torch.zeros((n_envs,), dtype=torch.bool, device=gs.device)
        free_path = torch.empty((num_waypoints, n_envs, franka.n_dofs), dtype=gs.tc_float, device=gs.device)
        for i in range(n_envs):
            free_path[:, i : i + 1], return_valid_mask[i : i + 1] = franka.plan_path(
                qpos_goal=qpos_goal[i : i + 1],
                envs_idx=[i],
                num_waypoints=num_waypoints,
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


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize("precision", ["32"])
@pytest.mark.parametrize("backend", [gs.gpu])
def test_contact_forces(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            # Enabling box-box algorithm to improve code coverage
            box_box_detection=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    scene.add_entity(
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
        # visualize_contact=True,
    )
    scene.build(n_envs=5)

    cube_weight = scene.rigid_solver._gravity[0] * cube.get_mass()
    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    qpos = np.array([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04])
    franka.set_qpos(qpos)
    scene.step()

    end_effector = franka.get_link("hand")
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.tile([0.65, 0.0, 0.13], (scene.n_envs, 1)),
        quat=np.tile([0, 1, 0, 0], (scene.n_envs, 1)),
    )
    franka.control_dofs_position(qpos[:, :-2], motors_dof)

    # hold
    for i in range(50):
        scene.step()
    contact_forces = cube.get_links_net_contact_force()
    assert_allclose(contact_forces[:, 0], -cube_weight, atol=1e-5)

    # grasp
    franka.control_dofs_position(qpos[:, :-2], motors_dof)
    franka.control_dofs_position(0.0, fingers_dof)
    for i in range(20):
        scene.step()

    # lift
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.tile([0.65, 0.0, 0.2], (scene.n_envs, 1)),
        quat=np.tile([0.0, 1, 0, 0], (scene.n_envs, 1)),
    )
    franka.control_dofs_position(qpos[:, :-2], motors_dof)
    for i in range(100):
        scene.step()

    # Check contact forces while randomizing gripper orientations across parallel envs.
    # Note that it is necessary to reset the scene state because the box is slowly falling without noslip solver.
    state = scene.get_state()
    rng = np.random.RandomState(0)
    all_errors = []
    for i_trial in range(10):
        scene.reset(state)

        angles = rng.uniform(-np.deg2rad(45), np.deg2rad(45), size=scene.n_envs).astype(gs.np_float)
        axes = rng.randn(scene.n_envs, 3).astype(gs.np_float)
        perturbs = gu.axis_angle_to_quat(angles, axes)
        lift_quats = gu.transform_quat_by_quat(perturbs, np.tile([0, 1, 0, 0], (scene.n_envs, 1)).astype(gs.np_float))
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=np.tile([0.65, 0.0, 0.2], (scene.n_envs, 1)).astype(gs.np_float),
            quat=lift_quats,
        )
        franka.control_dofs_position(qpos[:, :-2], motors_dof)
        franka.control_dofs_position(0.0, fingers_dof)
        for _ in range(160):
            scene.step()

        contact_forces = tensor_to_array(cube.get_links_net_contact_force())
        errors = np.linalg.norm(contact_forces[:, 0, :] + cube_weight, ord=np.inf, axis=-1)
        all_errors.append(errors)
    assert np.percentile(all_errors, 95) < 2e-4


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize("model_name", ["double_ball_pendulum"])
def test_apply_external_forces(xml_path, show_viewer):
    GRAVITY = 2.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            substeps=2,
            gravity=(0, 0, -GRAVITY),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -3.5, 2.5),
            camera_lookat=(0.0, 0.0, 1.0),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    scene.add_entity(
        gs.morphs.Plane(),
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file=xml_path,
            quat=(1.0, 0, 1.0, 0),
        ),
    )
    duck = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/duck.obj",
            scale=0.04,
            pos=(1.0, 0.0, 1.0),
            euler=(90, 0, 0),
            collision=False,
        ),
    )
    scene.build()
    rigid_solver = scene.rigid_solver

    end_effector_link_idx = robot.links[-1].idx
    duck_link_idx = duck.links[0].idx
    duck_mass = duck.get_mass()
    duck_init_link_pos, duck_init_link_R = duck.base_link.pos, gu.quat_to_R(duck.base_link.quat)
    for step in range(801):
        ee_pos = rigid_solver.get_links_pos([end_effector_link_idx])[0]
        duck_pos = rigid_solver.get_links_pos([duck_link_idx])[0]
        if step == 0:
            assert_allclose(ee_pos, (0.8, 0.0, 0.02), tol=1e-4)
        elif step in (500, 600):
            assert_allclose(ee_pos, (0.0, 0.0, 0.82), tol=0.01)
        elif step == 800:
            assert_allclose(ee_pos, (-0.8 / math.sqrt(2), 0.8 / math.sqrt(2), 0.02), tol=0.02)
        assert_allclose(duck_pos, duck_init_link_pos, tol=1e-3)

        if step >= 600:
            force = [-4.0, 4.0, 0.0]
            torque = [0.0, 0.0, 0.0]
        elif step >= 500:
            force = [0.0, 0.0, 0.0]
            torque = [0.0, 0.0, 2.0]
        elif step >= 50:
            force = [0.0, 0.0, 10.0]
            torque = [0.0, 0.0, 0.0]
        else:
            force = [0.0, 0.0, 0.0]
            torque = [0.0, 0.0, 0.0]

        rigid_solver.apply_links_external_force(
            force=duck_mass * GRAVITY * duck_init_link_R[2], links_idx=[duck_link_idx], ref="link_com", local=True
        )
        rigid_solver.apply_links_external_force(
            force=force, links_idx=[end_effector_link_idx], ref="link_origin", local=False
        )
        rigid_solver.apply_links_external_torque(
            torque=torque, links_idx=[end_effector_link_idx], ref="link_origin", local=False
        )
        scene.step()

    rigid_solver.apply_links_external_torque(torque=(0, 1, 0), links_idx=[duck_link_idx], ref="link_com", local=True)
    assert_allclose(rigid_solver.links_state.cfrc_applied_vel[duck_link_idx, 0], 0, tol=gs.EPS)
    assert_allclose(rigid_solver.links_state.cfrc_applied_ang[duck_link_idx, 0], -duck_init_link_R[:, 1], tol=gs.EPS)

    with np.testing.assert_raises(ValueError):
        rigid_solver.apply_links_external_force(force=(0, 0, 0), links_idx=[duck_link_idx], ref="root_com", local=True)
    with np.testing.assert_raises(ValueError):
        rigid_solver.apply_links_external_torque(
            torque=(0, 0, 0), links_idx=[duck_link_idx], ref="root_com", local=True
        )


@pytest.mark.slow  # ~250s
@pytest.mark.required
@pytest.mark.parametrize("model_name", ["long_chain"])
def test_mass_mat(xml_path, show_viewer, tol):
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
    # High-DOF single tree: its mass submatrix exceeds GPU shared memory, exercising the cooperative >shared-cap
    # assemble (the low-DOF frankas exercise the under-cap shared-memory factor instead).
    long_chain = scene.add_entity(
        gs.morphs.MJCF(
            file=xml_path,
            pos=(5, 0, 2),
        ),
    )
    scene.build()

    # Two identical entities must yield identical mass matrices, and the LTDL factor must reconstruct it.
    mass_mat_1 = franka1.get_mass_mat(decompose=False)
    mass_mat_2 = franka2.get_mass_mat(decompose=False)
    assert mass_mat_1.shape == (franka1.n_dofs, franka1.n_dofs)
    assert_allclose(mass_mat_1, mass_mat_2, tol=tol)

    mass_mat_L, mass_mat_D_inv = franka1.get_mass_mat(decompose=True)
    mass_mat = mass_mat_L.T @ torch.diag(1.0 / mass_mat_D_inv) @ mass_mat_L
    assert_allclose(mass_mat, mass_mat_1, tol=tol)

    # The cooperative >shared-cap assemble maps a flat lane index to a lower-triangular (row, col) via a float sqrt;
    # on GPUs whose sqrt undershoots perfect squares (Apple Metal: sqrt(15129) -> 122.999 instead of 123) a naive
    # inversion lands one row short on every j=0 boundary and silently drops the long-range coupling entries, leaving
    # the assembled mass matrix indefinite. A real joint-space mass matrix is always symmetric positive-definite.
    mass_mat_chain = tensor_to_array(long_chain.get_mass_mat(decompose=False))
    assert_allclose(mass_mat_chain, mass_mat_chain.T, tol=tol)
    assert np.linalg.eigvalsh(0.5 * (mass_mat_chain + mass_mat_chain.T)).min() > 0.0

    # On GPU the high-DOF chain factors through the register-tiled path (auto-enabled above the shared-memory cap when
    # RigidOptions.register_tiled_mass is left to its default); its LTDL factor must reconstruct the mass matrix to the
    # same accuracy as the under-cap path.
    mass_mat_chain_L, mass_mat_chain_D_inv = long_chain.get_mass_mat(decompose=True)
    mass_mat_chain_rec = mass_mat_chain_L.T @ torch.diag(1.0 / mass_mat_chain_D_inv) @ mass_mat_chain_L
    assert_allclose(mass_mat_chain_rec, mass_mat_chain, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["two_fixed_branches"])
def test_mass_block_partition(xml_path, show_viewer, tol):
    # Two chains rigidly attached to the fixed world are kinematically independent: the mass matrix is block-diagonal,
    # so it must partition into one mass block per branch (factoring two n/2 blocks instead of one dense n block).
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
        ),
        show_viewer=show_viewer,
    )
    entity = scene.add_entity(
        gs.morphs.MJCF(
            file=xml_path,
        ),
    )
    scene.build(n_envs=0)

    n_dofs = entity.n_dofs
    branch = n_dofs // 2
    block_start = qd_to_numpy(scene.rigid_solver._rigid_global_info.dofs_mass_block_start)
    block_end = qd_to_numpy(scene.rigid_solver._rigid_global_info.dofs_mass_block_end)
    assert_allclose(block_start, [0] * branch + [branch] * branch, tol=0)
    assert_allclose(block_end, [branch] * branch + [n_dofs] * branch, tol=0)

    # The two branches do not couple, and the LTDL factor reconstructs the (block-diagonal) mass matrix.
    mass_mat = tensor_to_array(entity.get_mass_mat(decompose=False))
    assert_allclose(mass_mat[:branch, branch:], 0.0, tol=tol)
    mass_mat_L, mass_mat_D_inv = entity.get_mass_mat(decompose=True)
    assert_allclose(mass_mat_L.T @ torch.diag(1.0 / mass_mat_D_inv) @ mass_mat_L, mass_mat, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["hinge_slide"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
def test_set_dofs_frictionloss_physics(gs_sim, tol):
    (robot,) = gs_sim.entities

    initial_velocity = np.array([1.0, 0.0])
    robot.set_dofs_velocity(initial_velocity)

    robot.set_dofs_frictionloss(np.array([0.0, 0.0]))
    frictionloss = robot.get_dofs_frictionloss()
    assert_allclose(frictionloss, np.array([0.0, 0.0]), atol=tol)
    for _ in range(10):
        gs_sim.step()
    velocity_zero = gs_sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]

    robot.set_dofs_velocity(initial_velocity)
    robot.set_dofs_frictionloss(np.array([1.0, 0.0]))
    frictionloss = robot.get_dofs_frictionloss()
    assert_allclose(frictionloss, np.array([1.0, 0.0]), atol=tol)
    for _ in range(10):
        gs_sim.step()
    velocity_high = gs_sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]

    np.testing.assert_array_less(velocity_high[0], velocity_zero[0])
    np.testing.assert_array_less(velocity_high[1], velocity_zero[1])

    robot.set_dofs_velocity(initial_velocity)
    robot.set_dofs_frictionloss(np.array([0.5]), dofs_idx_local=[0])
    frictionloss = robot.get_dofs_frictionloss(dofs_idx_local=[0])
    assert_allclose(frictionloss, np.array([0.5]), atol=tol)
    for _ in range(10):
        gs_sim.step()
    velocity_medium = gs_sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]

    np.testing.assert_array_less(velocity_high[0], velocity_medium[0])
    np.testing.assert_array_less(velocity_medium[0], velocity_zero[0])

    friction_effect = velocity_zero[0] - velocity_high[0]
    np.testing.assert_array_less(tol, friction_effect)

    slide_friction_effect = velocity_zero[1] - velocity_high[1]
    np.testing.assert_array_less(tol, slide_friction_effect)


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_frictionloss_advanced(show_viewer, tol):
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.4, 0.7, 1.4),
            camera_lookat=(0.6, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(gs.morphs.Plane())
    asset_path = get_hf_dataset(pattern="SO101/*")
    robot = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=f"{asset_path}/SO101/so101_new_calib.xml",
        ),
        # vis_mode="collision",
    )
    box = scene.add_entity(
        gs.morphs.Box(
            pos=(0.1, 0.0, 0.6),
            size=(0.025, 0.025, 0.025),
        ),
    )
    scene.build()

    scene.reset()
    for _ in range(230):
        scene.step()

    assert_allclose(robot.get_contacts()["position"][:, 2].min(), 0.0, tol=1e-4)
    assert_allclose(robot.get_AABB()[0, 2], 0.0, tol=2e-4)
    box_pos = box.get_pos()
    assert box_pos[0] > 0.4

    # This is to check collision detection is working correctly on Apple Metal.
    # The box should collide with the robot and roll on the ground within a reasonable range without not blow up.
    assert_allclose(box_pos[1:], 0.0, tol=0.05)
    assert_allclose(box.get_dofs_velocity(), 0.0, tol=50 * tol)


# Force CPU because it would be too slow otherwise
@pytest.mark.required
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
            euler=(100, -10, 0),
            convexify=False,
        ),
        # vis_mode="collision",
    )
    ball = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.05,
            pos=(0.0, 0.0, 0.75),
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
    for i in range(500):
        scene.step()
        if i > 450:
            qvel = scene.sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]
            assert_allclose(qvel, 0, atol=0.05)


# Force CPU because it would be too slow otherwise
@pytest.mark.parametrize("backend", [gs.cpu])
def test_nonconvex_nonwatertight_collision(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.002,
        ),
        rigid_options=gs.options.RigidOptions(
            max_collision_pairs=20,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, 15.0, 3.0),
            camera_lookat=(2.0, 0.0, -2.0),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    asset_path = get_hf_dataset(pattern="spacecraft.obj")
    scene.add_entity(
        gs.morphs.Mesh(
            file=f"{asset_path}/spacecraft.obj",
            pos=(-0.4, 0.0, -4.0),
            euler=(90.0, 0.0, 0.0),
            scale=3.0,
            convexify=False,
            fixed=True,
        ),
        vis_mode="collision",
    )
    obj = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
        ),
        surface=gs.surfaces.Default(
            color=(0.5, 0.7, 0.9, 1.0),
        ),
        visualize_contact=True,
    )
    scene.build(n_envs=64)

    obj.set_pos(
        torch.cartesian_prod(
            torch.linspace(-6.25, 9.05, 8),
            torch.linspace(-5.2, 5.5, 8),
            torch.tensor((0.39,)),
        )
    )
    for _ in range(750):
        scene.step()

    # The velocity is fairly large for boxes whose contact set is stable at keep changing (border of a cliff)
    assert_allclose(obj.get_dofs_velocity(), 0.0, tol=0.08)


@pytest.mark.parametrize("obj_shape", ["box", "sphere_mesh"])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_nonconvex_inner_corner_multi_contact(obj_shape, show_viewer, tmp_path):
    INIT_GAP = 1e-4  # initial gap between the body and the L-mesh surfaces (no overlap)
    # An object wedged at the inner corner of a non-convex L-shaped mesh under gravity tilted into both surfaces.
    # The object must settle in the corner with at least one contact on each surface (floor and wall). A single
    # contact with a mixed floor+wall normal is what the perturbation-only path returned, and it lets the object
    # squirt out of the corner along the tilted normal instead of staying wedged.
    # Parametrised over a primitive BOX and a generic mesh (tessellated icosphere via MeshSet, so it is *not*
    # classified as a SPHERE primitive) to exercise both the primitive-geom and generic-mesh dispatch paths.
    floor = trimesh.creation.box(extents=(4.0, 4.0, 0.2))
    floor.apply_translation((0.0, 0.0, -0.1))
    wall = trimesh.creation.box(extents=(0.2, 4.0, 2.0))
    wall.apply_translation((1.0, 0.0, 1.0))
    mesh_path = tmp_path / "L.obj"
    trimesh.util.concatenate([floor, wall]).export(mesh_path)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.002,
            gravity=(5.0, 0.0, -9.81),
        ),
        rigid_options=gs.options.RigidOptions(
            max_collision_pairs=20,
        ),
        viewer_options=gs.options.ViewerOptions(
            # Frame the L-corner (wall at x=0.9, floor at z=0) where the object settles.
            camera_pos=(0.6, -2.2, 0.55),
            camera_lookat=(0.85, 0.0, 0.15),
            camera_fov=35,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    world = scene.add_entity(
        gs.morphs.Mesh(
            file=str(mesh_path),
            pos=(0.0, 0.0, 0.0),
            convexify=False,
            fixed=True,
        ),
        visualize_contact=True,
        vis_mode="collision",
    )
    obj_surface = gs.surfaces.Default(color=(0.5, 0.7, 0.9, 1.0))
    if obj_shape == "box":
        obj = scene.add_entity(
            gs.morphs.Box(
                size=(0.2, 0.2, 0.2),
                pos=(0.8 - INIT_GAP, 0.0, 0.1 + INIT_GAP),
            ),
            surface=obj_surface,
            vis_mode="collision",
        )
    else:
        sphere_radius = 0.1
        obj = scene.add_entity(
            morph=gs.morphs.MeshSet(
                files=(trimesh.creation.icosphere(radius=sphere_radius, subdivisions=3),),
                pos=(0.8 - INIT_GAP, 0.0, sphere_radius + INIT_GAP),
                decimate=False,
                convexify=False,
            ),
            surface=obj_surface,
            vis_mode="collision",
        )
    scene.build()

    # Run 10 warm-up steps so the body resolves its initial fall/impact transient, then monitor the velocity at every
    # subsequent step: a wedged body must never spike (detect simulation blow-up). Final velocity must be near zero
    # (body must actually settle).
    for _ in range(10):
        scene.step()
    max_v_seen = 0.0
    for _ in range(200):
        scene.step()
        v = tensor_to_array(obj.get_dofs_velocity())
        max_v_seen = max(max_v_seen, float(np.abs(v).max()))
    assert max_v_seen < 0.05, f"velocity spike during settling: max |v| = {max_v_seen:.4f}"

    contacts = scene.rigid_solver.collider._collider_state.contact_data
    n_contacts = int(scene.rigid_solver.collider._collider_state.n_contacts[0])
    normals = qd_to_numpy(contacts.normal, transpose=True)
    positions = qd_to_numpy(contacts.pos, transpose=True)
    ga = qd_to_numpy(contacts.geom_a, transpose=True)
    obj_idx = obj.geoms[0].idx
    floor_contacts = []
    wall_contacts = []
    for k in range(n_contacts):
        sign = +1 if ga[0, k] == obj_idx else -1
        n = sign * normals[0, k]
        p = positions[0, k]
        if n[2] > 0.7:
            floor_contacts.append((p, n))
        elif n[0] < -0.7:
            wall_contacts.append((p, n))
    # Both shapes settle wedged in the L-corner with zero residual velocity. Equilibrium has the body centred at
    # (0.8, 0, 0.1): bottom touching the floor (z=0) and right touching the wall (x=0.9). Any drift means a spurious
    # tangential force from a non-axis-aligned contact normal.
    assert_allclose(obj.get_pos(), (0.8, 0.0, 0.1), tol=1e-3)
    assert_allclose(obj.get_dofs_velocity(), 0.0, tol=0.05)
    if obj_shape == "sphere_mesh":
        # The icosphere touches the floor at its bottom and the wall at its right; expect a single contact on each
        # surface with pure axis-aligned normal direction.
        assert n_contacts == 2, f"expected exactly 2 contacts (1 floor, 1 wall), got {n_contacts}"
        assert len(floor_contacts) == 1
        assert len(wall_contacts) == 1
        floor_pos, floor_normal = floor_contacts[0]
        wall_pos, wall_normal = wall_contacts[0]
        assert_allclose(floor_pos, (0.8, 0.0, 0.0), tol=5e-3)
        assert_allclose(floor_normal, (0.0, 0.0, 1.0), tol=1e-2)
        assert_allclose(wall_pos, (0.9, 0.0, 0.1), tol=5e-3)
        assert_allclose(wall_normal, (-1.0, 0.0, 0.0), tol=1e-2)
    # FIXME: The box test only checks that the body wedges at the L-corner equilibrium (position + zero velocity).
    # The detailed contact set is not asserted because the grid SDF emits an edge-regime contact at the bottom-right
    # corners with a non-axis-aligned normal; the resulting contact pattern works physically (the body wedges and stays
    # put) but does not match the clean 2-floor + 2-wall configuration the sphere case enforces.


# Force CPU because nonconvex SDF is slow on GPU
@pytest.mark.parametrize("backend", [gs.cpu])
def test_nonconvex_tunneling(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.002,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.2, 0.2, 1.0),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/tank.obj",
            euler=(95, 0, 0),
            scale=5.0,
            fixed=True,
            convexify=False,
        ),
        vis_mode="collision",
    )
    rod = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/stirrer.obj",
            pos=(0.0, 0.0, 0.2),
            convexify=False,
        ),
        vis_mode="collision",
        visualize_contact=True,
    )
    scene.build()

    # It collides with the tank bottom at step 200
    for step in range(250):
        scene.step()
    assert rod.get_pos()[..., 2] > -0.05
    assert_allclose(rod.get_dofs_velocity(dofs_idx_local=slice(None, 3)), 0, atol=0.08)


# Force CPU because nonconvex SDF is slow on GPU
@pytest.mark.parametrize("backend", [gs.cpu])
def test_nonconvex_overlap(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.001,
            gravity=(0, 0, 0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, 0.3, 0.15),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    a = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/stirrer.obj",
            pos=(-0.051, 0.0, 0.0),
            convexify=False,
        ),
        vis_mode="collision",
    )
    b = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/stirrer.obj",
            pos=(+0.05, 0.0, 0.0),
            euler=(0, 0, 90),
            convexify=False,
        ),
        vis_mode="collision",
        visualize_contact=True,
    )
    # Two compact solid meshes barely touching along their local OBB diagonal, away from the stirrers. The
    # closing-direction penetration floor must read the true submillimetric overlap along the axis; an OBB-projection
    # bound overestimates it by up to sqrt(3) off-axis and catapults the pair apart.
    asset_path = get_hf_dataset(pattern="apple_15/*")
    apples = []
    for i_apple in range(2):
        apples.append(
            scene.add_entity(
                gs.morphs.MJCF(
                    file=f"{asset_path}/apple_15/model.xml",
                    pos=(0.0, 0.5 + 0.2 * i_apple, 0.0),
                    convexify=False,
                ),
            )
        )
    scene.build()
    a.set_dofs_velocity(+1.0, dofs_idx_local=0)
    b.set_dofs_velocity(-1.0, dofs_idx_local=0)

    geom = apples[0].geoms[0]
    apples_overlap = 1e-3
    u_local = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    proj = (geom.init_verts - geom.init_verts.mean(axis=0)) @ u_local
    u_world = gu.quat_to_R(tensor_to_array(geom.get_quat())) @ u_local
    apples_offset = (proj.max() - proj.min() - apples_overlap) * u_world
    apples_pos_ref = [tensor_to_array(apple.get_pos()) for apple in apples]
    apples_pos_ref[1] = apples_pos_ref[0] + apples_offset
    apples[1].set_pos(apples_pos_ref[1])

    total_energy_history = []
    for _ in range(200):
        total_energy = tensor_to_array(a.get_total_energy() + b.get_total_energy())
        total_energy_history.append(total_energy)
        scene.step()

    # FIXME: The total energy should be not strictly decreasing but is not... relaxing the condition
    # assert (np.diff(total_energy_history, axis=0) < 0.0)
    assert total_energy_history[0] > 3.0 * total_energy_history[-1]

    # Constraint stabilization alone resolves the overlap, so it cannot separate the apples faster than
    # overlap / timeconst; a spurious deep contact catapults them an order of magnitude above that ceiling.
    # Contact impulses being internal to the pair, its total momentum must stay zero.
    v_sep_max = apples_overlap / float(geom.sol_params[0])
    assert np.linalg.norm(tensor_to_array(apples[1].get_vel() - apples[0].get_vel())) < v_sep_max
    assert_allclose(apples[0].get_vel() + apples[1].get_vel(), 0, atol=1e-6)
    # The apples must separate by at least the overlap, but no more than the stabilization drift accumulates
    # over the simulated horizon.
    apples_dist = np.linalg.norm(tensor_to_array(apples[1].get_pos() - apples[0].get_pos()))
    assert 0.5 * apples_overlap < apples_dist - np.linalg.norm(apples_offset) < v_sep_max * 200 * 0.001


# Force CPU because nonconvex SDF is slow on GPU
@pytest.mark.parametrize("backend", [gs.cpu])
@pytest.mark.xfail(reason="Recovery is too slow: the separating push is creep-rate-bound by the thin-shell pen cap.")
def test_nonconvex_shell_crossing_recovery(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.004,
            gravity=(0, 0, 0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.17, 0.21, 0.6),
            camera_lookat=(0.04, -0.02, 0.5),
        ),
        show_viewer=show_viewer,
    )
    asset_path = get_hf_dataset(pattern="cup_2/*")
    cups = []
    for i_cup in range(2):
        cups.append(
            scene.add_entity(
                gs.morphs.MJCF(
                    file=f"{asset_path}/cup_2/model.xml",
                    pos=(0.0, 0.0, 0.5 * i_cup),
                    convexify=False,
                ),
                vis_mode="collision",
                # visualize_contact=True,
            )
        )
    scene.build()

    # Spawn the pair deeply crossed: cup B perpendicular to cup A with its mouth rim 45mm past A's side wall, the
    # thin-shell equivalent of an overlapping spawn. The solver must recover by pushing the shells apart until they
    # separate; the failure mode is a wedged equilibrium where contacts on both sides of the crossing curve cancel
    # and hold the crossed pose forever.
    geom = cups[0].geoms[0]
    verts = torch.as_tensor(geom.init_verts, dtype=gs.tc_float)
    ext = verts.max(dim=0).values - verts.min(dim=0).values
    mesh_center = 0.5 * (verts.max(dim=0).values + verts.min(dim=0).values)
    i_axis = int(torch.argmax(ext))
    quat_a = geom.get_quat()
    rot_a = gu.quat_to_R(quat_a)
    radial_dir = rot_a[:, (i_axis + 1) % 3]
    euler_rot = torch.zeros(3, dtype=gs.tc_float)
    euler_rot[(i_axis + 2) % 3] = 0.5 * math.pi
    quat_b = gu.transform_quat_by_quat(gu.xyz_to_quat(euler_rot), quat_a)
    mesh_center_a = torch.tensor([0.0, 0.0, 0.5], dtype=gs.tc_float) + gu.transform_by_quat(mesh_center, quat_a)
    mesh_center_b = mesh_center_a + radial_dir * (0.5 * (ext[(i_axis + 1) % 3] + ext[i_axis]) - 0.045)
    for cup, quat_target, mesh_center_target in ((cups[0], quat_a, mesh_center_a), (cups[1], quat_b, mesh_center_b)):
        cup_geom = cup.geoms[0]
        corr_quat = gu.transform_quat_by_quat(gu.inv_quat(cup_geom.get_quat()), quat_target)
        cup.set_quat(gu.transform_quat_by_quat(cup.get_quat(), corr_quat))
        cup.set_pos(
            cup.get_pos() + mesh_center_target - cup_geom.get_pos() - gu.transform_by_quat(mesh_center, quat_target)
        )

    centers_dist_init = torch.linalg.norm(cups[1].get_pos() - cups[0].get_pos())
    for _ in range(200):
        scene.step()

    # Separation requires the centers to move apart by at least the spawn crossing depth; a wedged pair stays put
    # (residual velocities below 1e-2 m/s and unchanged distance)
    centers_dist = torch.linalg.norm(cups[1].get_pos() - cups[0].get_pos())
    assert centers_dist - centers_dist_init > 0.02


# Force CPU because nonconvex SDF is slow on GPU
@pytest.mark.parametrize("backend", [gs.cpu])
@pytest.mark.parametrize("direction", ["down", "up"])
def test_nonconvex_concentric_contact(direction, show_viewer):
    PITCH = 3.0e-3  # matches genesis/assets/meshes/bolt_nut/generate_bolt_nut.py
    PITCH_RATE = PITCH / (2.0 * np.pi)  # axial advance per radian of rotation
    # Head top is at z = 11 mm, so the 18 mm nut seats with its center at z ~ 20 mm. Driving down, release just above
    # that so the nut coasts onto the head rather than being driven into it. Driving up, release just below the shaft
    # tip (z ~ 48 mm) where only a turn of thread is left engaged, so the nut spins off and falls rather than stalling.
    SEAT_RELEASE_Z = 0.0202
    TIP_RELEASE_Z = 0.048
    # Advance-per-revolution is only meaningful while enough of the nut is still threaded. Past ~2/3 unscrewed (less
    # than a third of the 18 mm nut still gripping below the 43 mm shaft tip) the few remaining threads slip, so the
    # pitch is checked only up to that engagement.
    NUT_HEIGHT = 0.018
    SHAFT_TIP_Z = 0.043
    # 4.5 N*m screws down and 5.0 N*m unscrews up within the step budget below. At this test's single-substep dt the
    # solve diverges past ~5 N*m (the example tolerates more only because its substeps make each step stiffer).
    torque = -4.5 if direction == "down" else 5.0
    # Per-step thread coupling carries a contact jitter of a few mm/s; unscrewing jitters more, so its bound is looser.
    coupling_atol = 5e-3 if direction == "down" else 1e-2

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
        ),
        rigid_options=gs.options.RigidOptions(
            # Fine-thread contact needs a stiff constraint (the default 0.01 is too soft, letting the nut sink through
            # the flanks and advance faster than the pitch).
            constraint_timeconst=4e-3,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -0.2, 0.1),
            camera_lookat=(0.0, 0.0, 0.03),
            camera_fov=35,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(gs.morphs.Plane())
    # Realistic steel density so the nut carries a real fastener's mass and inertia.
    steel = gs.materials.Rigid(rho=7850.0)
    scene.add_entity(
        gs.morphs.Mesh(
            # Head bottom rests on the plane (z = 0), head top at z = 11 mm, shaft tip at z = 43 mm.
            pos=(0.0, 0.0, 0.011),
            file="meshes/bolt_nut/bolt.stl",
            decimate=False,
            convexify=False,
            fixed=True,
        ),
        material=steel,
        vis_mode="collision",
    )
    nut = scene.add_entity(
        gs.morphs.Mesh(
            # Pre-engaged near the top of the shaft (base ~24 mm) so it has the whole thread to travel.
            pos=(0.0, 0.0, 0.024),
            file="meshes/bolt_nut/nut.stl",
            decimate=False,
            convexify=False,
        ),
        material=steel,
        vis_mode="collision",
        # visualize_contact=True,
    )
    scene.build()

    # Drive a steady torque about z (a wrench) until the nut reaches its release height, then let go: negative torque
    # screws it down onto the head, positive torque unscrews it up and off the shaft tip.
    z0 = nut.get_pos(relative=False)[..., 2]
    prev_yaw = gu.quat_to_xyz(nut.get_quat(relative=False))[..., 2]
    total_turn = 0.0
    released_step = None
    z_engaged = z0
    turn_engaged = total_turn
    z_history = []
    horizon = 4100 if direction == "down" else 4800
    for step in range(horizon):
        z = nut.get_pos(relative=False)[..., 2]
        if released_step is None:
            reached = (z < SEAT_RELEASE_Z).all() if direction == "down" else (z > TIP_RELEASE_Z).all()
            if reached:
                released_step = step
        driving = released_step is None
        nut.control_dofs_force([torque if driving else 0.0], dofs_idx_local=(5,))
        scene.step()

        pos = nut.get_pos(relative=False)
        rpy = gu.quat_to_xyz(nut.get_quat(relative=False))
        vel = nut.get_dofs_velocity()
        z_history.append(pos[..., 2])
        yaw = rpy[..., 2]
        total_turn = total_turn + ((yaw - prev_yaw + np.pi) % (2.0 * np.pi) - np.pi)
        prev_yaw = yaw
        # Fraction of the nut height still threaded below the shaft tip (capped at 1 while fully on the shaft). Track
        # the last driven sample with more than a third engaged for the advance-per-revolution check below.
        engaged = torch.clamp((SHAFT_TIP_Z - (pos[..., 2] - NUT_HEIGHT / 2.0)) / NUT_HEIGHT, max=1.0)
        if driving and (engaged > 0.5).all():
            z_engaged = pos[..., 2]
            turn_engaged = total_turn
        # While steadily screwing through the middle of the thread (past the initial spin-up, away from the seat and
        # the tip where engagement thins) the nut stays coaxial and upright and its axial speed stays locked to its
        # rotation by the pitch (vz = wz * pitch/2pi). The bound is loose because of the per-step contact jitter; it
        # guards against a flank dropping out and letting vz decouple from wz by tens of mm/s (the nut spins without
        # translating, stripping).
        if driving and step > 100 and (0.025 < pos[..., 2]).all() and (pos[..., 2] < 0.043).all():
            assert (torch.linalg.norm(pos[..., :2], dim=-1) < 5e-4).all()
            assert_allclose(rpy[..., :2], 0.0, atol=0.02)
            assert_allclose(vel[..., 2], vel[..., 5] * PITCH_RATE, atol=coupling_atol)

    # The nut travelled the thread and reached its release height.
    assert released_step is not None
    # Over the well-engaged phase the axial advance per revolution tracks the thread pitch (it really screwed along
    # the thread rather than slipping).
    travel = torch.abs(z_engaged - z0)
    revolutions = torch.abs(turn_engaged) / (2.0 * np.pi)
    assert_allclose(travel / revolutions, PITCH, rtol=0.1)

    if direction == "down":
        # Comes to a clean rest seated on the head - no bounce, no drift: over the final settle window the nut height
        # holds within a tight band, since a bounce or a strip would show as a large z excursion. The seated nut keeps
        # a small steady contact jitter in velocity, so the position band is the robust at-rest signal.
        z_window = torch.stack(z_history[-200:], dim=0)
        assert ((z_window.amax(dim=0) - z_window.amin(dim=0)) < 1e-4).all()
        z_final = nut.get_pos(relative=False)[..., 2]
        assert ((0.019 < z_final) & (z_final < 0.021)).all()
    else:
        # Spun off the tip, fell, and came to rest flat on the ground: its bounding box now sits on the plane and all
        # of its velocities have decayed to zero.
        aabb = nut.get_AABB()
        assert (aabb[..., 0, 2] < 1.0e-3).all()
        assert_allclose(nut.get_dofs_velocity(), 0.0, atol=0.07)


# Force CPU because nonconvex SDF is slow on GPU
@pytest.mark.debug(False)  # Disable debug for speedup
@pytest.mark.parametrize("backend", [gs.cpu])
@pytest.mark.parametrize(
    "timestep, decimate",
    [
        pytest.param(0.01, True, marks=pytest.mark.required),
        (0.001, False),
    ],
)
def test_nonconvex_concave_slanted_wall(timestep, decimate, show_viewer):
    BOWL_THICKNESS = 0.013
    NUM_BOWLS = 32

    timeconst = max(0.005, 2 * timestep)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=timestep,
        ),
        rigid_options=gs.options.RigidOptions(
            # The pyramidal cone cannot hold the pile: its regularized friction creeps tangentially under the
            # sustained shear of the nested stack, and the tower topples within a few thousand steps.
            friction_cone=gs.friction_cone.elliptic,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(-0.6, 0.6, 0.5),
            camera_lookat=(-0.25, 0.0, 0.3),
        ),
        renderer=gs.renderers.Rasterizer(),
        show_viewer=show_viewer,
    )
    scene.add_entity(morph=gs.morphs.Plane())
    asset_path = get_hf_dataset(pattern="glb/orange_plastic_bowl.glb")
    for i in range(NUM_BOWLS):
        scene.add_entity(
            morph=gs.morphs.Mesh(
                file=f"{asset_path}/glb/orange_plastic_bowl.glb",
                pos=(0, 0, 0.0 + i * (BOWL_THICKNESS - 0.15 * timeconst)),
                euler=(90, 0, 0),
                convexify=False,
                decimate=decimate,
                file_meshes_are_zup=True,
            ),
            vis_mode="collision",
            # visualize_contact=(i in (0, NUM_BOWLS - 1)),
        )
    scene.build()

    # Make sure that the pile stays upright, with bowls stay tightly packed together during the entire motion
    bowls_link_idx = [entity.base_link_idx for entity in scene.entities[-NUM_BOWLS:]]
    # The spawn drop sways the stack laterally before it settles; assert once the transient has decayed.
    for _ in range(700):
        scene.step()
    for _ in range(1000):
        scene.step()
        bowls_pos = tensor_to_array(scene.rigid_solver.get_links_pos(bowls_link_idx, relative=True))
        bowls_dist_abs = np.linalg.norm(bowls_pos[:, :2] - bowls_pos[0, :2], axis=-1)
        assert (bowls_dist_abs < 0.025).all()
        bowls_dist_rel = np.linalg.norm(np.diff(bowls_pos, axis=0), axis=-1)
        assert ((BOWL_THICKNESS - 0.5 * timeconst) < bowls_dist_rel).all()
        assert (bowls_dist_rel < BOWL_THICKNESS + 1e-3).all()


@pytest.mark.required
@pytest.mark.parametrize("convexify", [True, False])
@pytest.mark.parametrize("gjk_collision", [True, False])
def test_mesh_repair(convexify, show_viewer, gjk_collision):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.004,
        ),
        rigid_options=gs.options.RigidOptions(
            use_gjk_collision=gjk_collision,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.3, 0.4, 0.01),
            camera_lookat=(0.3, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    asset_path = get_hf_dataset(pattern="work_table.glb")
    scene.add_entity(
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
            pos=(0.3, 0, 0.009),
            euler=(0.0, -2.5 if convexify else 0.0, 0.0),
            convexify=convexify,
            scale=1.0,
        ),
        vis_mode="collision",
        visualize_contact=True,
    )
    scene.build()

    if show_viewer:
        obj_com = obj.get_links_pos(ref="link_com")[0]
        scene.draw_debug_sphere(pos=obj_com, radius=0.003, color=(1, 1, 1, 1))
        scene.visualizer.update(force=True)

    for geom in obj.geoms:
        assert ("decomposed" in geom.metadata) ^ (not convexify)
        max_faces = obj._morph.decimate_face_num if convexify else 5000
        num_faces = geom.face_end - geom.face_start
        assert num_faces <= max_faces
        assert ("convexified" in geom.metadata) ^ (not convexify)

    # MPR collision detection is less reliable than SDF and GJK in terms of penetration depth estimation
    is_mpr = convexify and not gjk_collision
    tol_pos = 0.05 if is_mpr else 0.005
    tol_rot = 1.25 if is_mpr else 0.5
    init_pos = obj.geoms[0].get_pos()
    for _ in range(50):
        scene.step()
    for _ in range(100):
        scene.step()
        qvel = obj.get_dofs_velocity()
        assert_allclose(qvel[:3], 0, atol=tol_pos)
        assert_allclose(qvel[3:], 0, atol=tol_rot)
    assert_allclose(obj.geoms[0].get_pos()[:2], init_pos[:2], atol=2e-3)


@pytest.mark.required
@pytest.mark.parametrize("euler", [(90, 0, 90), (74, 15, 90)])
@pytest.mark.parametrize("gjk_collision", [True, False])
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_convexify(euler, show_viewer, gjk_collision):
    OBJ_OFFSET_X = 0.0  # 0.02
    OBJ_OFFSET_Y = 0.15
    N_SETTLE = 1000

    # The test check that the volume difference is under a given threshold and that convex decomposition is only used
    # whenever it is necessary. Then run a simulation to see if it explodes, i.e. objects are at reset inside tank.
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.004,
        ),
        rigid_options=gs.options.RigidOptions(
            use_gjk_collision=gjk_collision,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, 0.5, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
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
    scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/tank.obj",
            scale=5.0,
            fixed=True,
            pos=(0.05, -0.05, 0.0),
            euler=euler,
            # coacd_options=gs.options.CoacdOptions(
            #     threshold=0.08,
            # ),
        ),
        vis_mode="collision",
    )
    objs = []
    for i, (asset_name, xml_file) in enumerate(
        (("mug_1", "output.xml"), ("donut_0", "output.xml"), ("cup_2", "model.xml"), ("apple_15", "model.xml"))
    ):
        asset_path = get_hf_dataset(pattern=f"{asset_name}/*")
        obj = scene.add_entity(
            gs.morphs.MJCF(
                file=f"{asset_path}/{asset_name}/{xml_file}",
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
    assert not any(geom.metadata.get("decomposed", False) for geom in apple.geoms)
    assert not any(geom.metadata.get("decomposed", False) for geom in cup.geoms)
    assert all(geom.metadata["decomposed"] for geom in donut.geoms) and 5 <= len(donut.geoms) <= 10
    assert all(geom.metadata["decomposed"] for geom in mug.geoms) and 5 <= len(mug.geoms) <= 40
    assert all(geom.metadata["decomposed"] for geom in box.geoms) and 5 <= len(box.geoms) <= 20

    # Check that all the objects settle at rest after a while, without spurious jumps
    # cam.start_recording()
    vel_lin_all, vel_ang_all = [], []
    for i in range(N_SETTLE + 100):
        scene.step()
        # cam.render()
        if i > N_SETTLE:
            vel_lin_all.append(gs_sim.rigid_solver.get_links_vel(ref="link_com"))
            vel_ang_all.append(gs_sim.rigid_solver.get_links_ang())
    # cam.stop_recording(save_to_filename="video.mp4", fps=60)
    # FIXME: There is spurious residual motion on both paths that prevents the objects from truly settling
    assert_allclose(torch.quantile(torch.stack(vel_lin_all, dim=0), 0.5, dim=0), 0.0, tol=0.01)
    assert_allclose(torch.quantile(torch.stack(vel_ang_all, dim=0), 0.5, dim=0), 0.0, tol=0.1)

    for obj in objs:
        obj_pos = tensor_to_array(obj.get_pos())
        np.testing.assert_array_less(-0.1, obj_pos[2])
        np.testing.assert_array_less(obj_pos[2], 0.15)
        np.testing.assert_array_less(np.linalg.norm(obj_pos[:2]), 0.5)

    # Check that the mug, donut and cup are landing straight if the tank is horizontal.
    # FIXME: The cup is falling on Windows OS because the convex decomposition provided by CoACD is different than
    # other platform, and much worst in practice, with the bottom of the tank that is not planar (even discontinuous).
    if euler == (90, 0, 90):
        for i, obj in enumerate((mug, donut, *(() if sys.platform == "win32" else (cup,)))):
            obj_pos = obj.get_pos()
            assert_allclose(obj_pos[:2], (OBJ_OFFSET_X * (1.5 - i), OBJ_OFFSET_Y * (i - 1.5)), atol=6e-3)


@pytest.mark.required
@pytest.mark.parametrize("convexify, watertighten", [(True, 5), (False, 5), (False, None)])
@pytest.mark.parametrize("model_name", ["decompose_fusion_groups"])
def test_convexify_fusion_groups(convexify, watertighten, xml_path):
    scene = gs.Scene()
    entity = scene.add_entity(
        gs.morphs.MJCF(
            file=xml_path,
            convexify=convexify,
            watertighten=watertighten,
        ),
    )
    scene.build()

    # The plane can never be merged nor watertightened.
    (geom_plane,) = [geom for geom in entity.geoms if geom.type == gs.GEOM_TYPE.PLANE]
    assert len(geom_plane.init_verts) == 4
    assert not geom_plane.metadata.get("watertightened", False)

    if convexify:
        # Only the L-shape sub-group may be decomposed, with its primitive box merged along: the mesh boxes must
        # survive as four separate convex geoms instead of one hull spanning their gap, and the primitive boxes must
        # pass through untouched.
        geoms_decomposed = [geom for geom in entity.geoms if geom.metadata.get("decomposed", False)]
        geoms_box = [
            geom
            for geom in entity.geoms
            if geom.type == gs.GEOM_TYPE.MESH and not geom.metadata.get("decomposed", False)
        ]
        assert len(geoms_decomposed) >= 2
        assert len(geoms_box) == 4
        for geom in geoms_box:
            assert geom.is_convex
            assert not geom.metadata.get("merged", False)
            assert_allclose(geom.init_verts.max(axis=0) - geom.init_verts.min(axis=0), 0.1, tol=gs.EPS)
        assert len([geom for geom in entity.geoms if geom.type == gs.GEOM_TYPE.BOX]) == 2
    elif watertighten is not None:
        # All multi-geom sub-groups are fused systematically, including bare primitives, and every fused geom is
        # watertightened even when its sub-meshes are individually watertight. The mesh boxes with adjacent collision
        # masks belong to distinct sub-groups and must survive as two separate geoms.
        assert all(geom.type in (gs.GEOM_TYPE.PLANE, gs.GEOM_TYPE.MESH) for geom in entity.geoms)
        geoms_merged = [geom for geom in entity.geoms if geom.metadata.get("merged", False)]
        assert len(geoms_merged) == 3
        assert len(entity.geoms) == 6
        assert all(geom.metadata.get("watertightened", False) for geom in geoms_merged)
    else:
        # Disabling watertightening on the nonconvex path opts out of fusion entirely: every geom passes through.
        assert len(entity.geoms) == 9
        assert not any(geom.metadata.get("merged", False) for geom in entity.geoms)
        assert len([geom for geom in entity.geoms if geom.type == gs.GEOM_TYPE.BOX]) == 3


@pytest.mark.debug(False)  # Disable debug for speedup
@pytest.mark.slow
@pytest.mark.precision("32")
@pytest.mark.parametrize("backend", [gs.cpu])
@pytest.mark.parametrize("convexify", [False, True])
def test_many_objects_collision(convexify, show_viewer, tol):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.004,
        ),
        rigid_options=gs.options.RigidOptions(
            max_collision_pairs=8000,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, 0.2, 1.6),
            camera_lookat=(0.0, 0.0, 0.3),
        ),
        show_viewer=show_viewer,
    )
    tank = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/tank.obj",
            scale=5.0,
            fixed=True,
            euler=(90, 0, 90),
            convexify=convexify,
        ),
        vis_mode="collision",
    )
    assets = (("mug_1", "output.xml"), ("donut_0", "output.xml"), ("cup_2", "model.xml"), ("apple_15", "model.xml"))
    asset_files = {name: f"{get_hf_dataset(pattern=f'{name}/*')}/{name}/{xml}" for name, xml in assets}
    objs = []
    obj_names = []
    for i in range(80):
        gx, gy, gz = i % 4, (i // 4) % 4, i // 16
        name = assets[(gx + gy + gz) % len(assets)][0]
        obj_names.append(name)
        base_pos = ((gx + 0.5 * (gz % 2)) * 0.1 - 0.18, (gy + 0.5 * (gz % 2)) * 0.145 - 0.265, 0.11 + gz * 0.08)
        objs.append(
            scene.add_entity(
                gs.morphs.MJCF(
                    file=asset_files[name],
                    pos=base_pos + np.random.uniform(-2e-4, 2e-4, 3),
                    euler=(90.0, 0.0, 0.0) + np.random.uniform(-0.2, 0.2, 3),
                    convexify=convexify,
                ),
                vis_mode="collision",
            )
        )
    scene.build()

    # Wait for the pile to collapse and settle at rest
    vmax_trace, wmax_trace, energy_trace = [], [], []
    for i in range(1300):
        scene.step()
        energy_trace.append(tensor_to_array(scene.rigid_solver.get_total_energy()))
        if show_viewer:
            vmax_trace.append(scene.rigid_solver.get_links_vel(ref="link_com").norm(dim=-1).max())
            wmax_trace.append(scene.rigid_solver.get_links_ang().norm(dim=-1).max())

    # The pile has settled at rest, fully contained in the tank (no ground/tank penetration, no ejection)
    for obj in objs:
        obj_pos = tensor_to_array(obj.get_pos())
        np.testing.assert_array_less(-0.1, obj_pos[2])
        np.testing.assert_array_less(obj_pos[2], 0.6)
        np.testing.assert_array_less(np.linalg.norm(obj_pos[:2]), 0.5)

    # Make sure that there is no interpenetration among the settled objects
    links, link_names = [], []
    for obj, name in zip(objs, obj_names):
        for link in obj.links:
            links.append([(geom.get_verts(), geom.get_trimesh().faces) for geom in link.geoms])
            link_names.append(name)
    max_penetration, crossings = get_genuine_interpenetration(links)
    # FIXME: Rare (~5% of initial-pose draws) stem-through-wall traps exceed this bound by design: a thin feature
    # creeping through a sub-cell wall is a known nonconvex detection limitation, excluded from the bound.
    assert max_penetration < (5e-4 if convexify else 5e-3)

    # Over a 100-step window, record the residual velocities and the net energy produced per contact
    vel_lin_all, vel_ang_all = [], []
    contact_energy = {}
    for i in range(100):
        scene.step()
        com_pos = scene.rigid_solver.get_links_pos(ref="link_com")
        com_vel = scene.rigid_solver.get_links_vel(ref="link_com")
        ang = scene.rigid_solver.get_links_ang()
        vel_lin_all.append(com_vel.norm(dim=-1))
        vel_ang_all.append(ang.norm(dim=-1))
        contacts = scene.rigid_solver.collider.get_contacts(as_tensor=True)
        link_a, link_b = contacts["link_a"], contacts["link_b"]
        pos, force = contacts["position"], contacts["force"]
        v_rel = (
            com_vel[link_b]
            + torch.linalg.cross(ang[link_b], pos - com_pos[link_b])
            - com_vel[link_a]
            - torch.linalg.cross(ang[link_a], pos - com_pos[link_a])
        )
        power = (force * v_rel).sum(dim=-1)
        keys = zip(link_a.tolist(), link_b.tolist(), map(tuple, (pos / 2e-3).round().tolist()))
        for key, contact_power in zip(keys, power.tolist()):
            contact_energy[key] = contact_energy.get(key, 0.0) + contact_power * scene.sim_options.dt
        energy_trace.append(tensor_to_array(scene.rigid_solver.get_total_energy()))
        if show_viewer:
            vmax_trace.append(com_vel.norm(dim=-1).max())
            wmax_trace.append(ang.norm(dim=-1).max())

    # Make sure that all objects are settling at rest.
    # Note that it is not possible to be stricter than quantile because there is legitimate residual motion.
    # FIXME: Why the angular velocity threshold has to be so large without any visual effect?!
    assert_allclose(torch.quantile(torch.stack(vel_lin_all, dim=0), 0.7, dim=0), 0.0, tol=0.1 if convexify else 0.2)
    assert_allclose(torch.quantile(torch.stack(vel_ang_all, dim=0), 0.7, dim=0), 0.0, tol=5.0 if convexify else 8.0)

    # Contacts at zero restitution must dissipate over their lifetime, so net positive contact energy is the
    # solver pumping; contact_data.force acts as -F on link_a and +F on link_b.
    # FIXME: Both path pumps net positive contact energy over this window.
    assert sum(max(energy, 0.0) for energy in contact_energy.values()) < (0.1 if convexify else 1.0)
    # Total mechanical energy (KE+PE) is a state function, so its per-step rise isolates fictitious energy the
    # solver injected at contacts (a strictly dissipative pile can only lose energy).
    # FIXME: Both paths suffer from fictitious energy injection.
    assert np.quantile(np.maximum(np.diff(energy_trace), 0.0), 0.95 if convexify else 0.75) < tol

    if show_viewer:
        _fig, (ax_v, ax_w, ax_e) = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
        ax_v.semilogy(vmax_trace)
        ax_v.set_ylabel("max |linear velocity| [m/s]")
        ax_w.semilogy(wmax_trace)
        ax_w.set_ylabel("max |angular velocity| [rad/s]")
        ax_w.set_ylim(bottom=1e-3)
        ax_e.plot(np.maximum(np.diff(energy_trace), 0.0))
        ax_e.set_ylabel("energy injected dE+ [J]")
        ax_e.set_xlabel("step")
        for ax in (ax_v, ax_w, ax_e):
            ax.set_xlim(0, len(vmax_trace) - 1)
            ax.grid(True)
        plt.tight_layout()
        plt.show(block=True)

        pairs = []
        for crossing in crossings:
            a, b = crossing.link_a, crossing.link_b
            label = f"{link_names[a]}#{a} vs {link_names[b]}#{b} ({crossing.depth * 1e3:.1f}mm)"
            pairs.append((links[a], links[b], label))
        if pairs:
            display_collision_pairs(pairs)


@pytest.mark.slow("gpu")  # gpu ~250s
@pytest.mark.parametrize(
    "scene_kind, max_collision_pairs, max_contacts, error_pattern",
    [
        # Post-pruning contact budget overflow, with the candidate buffer large enough (2x margin) that it cannot
        # trip first. The automatic budget resolves to 32 contact points per link pair floored at 512, far below
        # what the piled-up bowls produce.
        pytest.param("bowls", 1_000, None, "max number of post-pruning contact points", marks=pytest.mark.required),
        # Candidate contact buffer overflow. The explicit contact budget is clamped down to the buffer size, so only
        # the buffer itself can overflow.
        ("bowls", 150, 1_000, "max number of candidate contact points"),
        # Buffers large enough for the whole pile: no overflow at all. Both values keep a 2x margin over the peaks
        # reached within the stepped window (about 500 colliding geom pairs and 1040 post-pruning contact points).
        ("bowls", 1_000, 2_000, None),
        # Two contacts against a budget of one: the clamp must also run when the contact count is below the pruning
        # gate (n_contacts < 3), in both the serial and the GPU cooperative kernel variants.
        ("spheres", 150, 1, "max number of post-pruning contact points"),
    ],
)
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_num_contact_overflow(scene_kind, max_collision_pairs, max_contacts, error_pattern, show_viewer):
    from genesis.engine.simulator import RATE_CHECK_ERRNO

    N_BOWLS = 4
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            max_collision_pairs=max_collision_pairs,
            max_contacts=max_contacts,
        ),
        show_viewer=show_viewer,
        renderer=gs.renderers.Rasterizer(),
    )
    scene.add_entity(morph=gs.morphs.Plane())
    if scene_kind == "bowls":
        asset_path = get_hf_dataset(pattern="glb/orange_plastic_bowl.glb")
        for _ in range(N_BOWLS):
            scene.add_entity(
                morph=gs.morphs.Mesh(
                    file=f"{asset_path}/glb/orange_plastic_bowl.glb",
                    pos=(0, 0, 0.5),
                    euler=(90, 0, 0),
                    convexify=True,
                    file_meshes_are_zup=True,
                ),
            )
    else:
        # Non-contacting nonconvex mesh: makes the scene prunable so that the GPU cooperative kernel is exercised.
        scene.add_entity(
            morph=gs.morphs.Mesh(
                file="meshes/duck.obj",
                scale=0.04,
                pos=(5.0, 5.0, 5.0),
                convexify=False,
            ),
        )
        for i in range(2):
            scene.add_entity(
                morph=gs.morphs.Sphere(
                    pos=(0.5 * i, 0.0, 0.0999),
                    radius=0.1,
                ),
            )
    scene.build()
    assert scene.rigid_solver.collider._collider_static_config.has_prunable_contacts

    # The resolved contact budget must match the documented resolution: 32 contact points per link pair floored at
    # 512 when automatic (every link pair here has more than 32 candidate contact points), the explicit value clamped
    # to the candidate buffer size otherwise. The constraint buffers are sized accordingly, with 4 constraint rows
    # per contact point (all joints are free so there is no joint-limit term).
    solver = scene.rigid_solver
    collider_info = solver.collider._collider_info
    if max_contacts is None:
        n_link_pairs = (N_BOWLS + 1) * N_BOWLS // 2
        expected_max_contacts = max(32 * n_link_pairs, 512)
    else:
        expected_max_contacts = min(max_contacts, int(collider_info.max_candidate_contacts[None]))
    assert int(collider_info.max_contacts[None]) == expected_max_contacts
    expected_len_constraints = 4 * expected_max_contacts + solver.n_dofs + 6 * solver.n_candidate_equalities_
    assert solver.constraint_solver.len_constraints == expected_len_constraints

    # All overflows occur on the very first step (the bowls start fully overlapping, the spheres start resting on the
    # plane), but errno is only polled every RATE_CHECK_ERRNO substeps, so one extra step is required to guarantee
    # that the error gets raised.
    with nullcontext() if error_pattern is None else pytest.raises(gs.GenesisException, match=error_pattern):
        for _ in range(RATE_CHECK_ERRNO + 1):
            scene.step()


@pytest.mark.required
@pytest.mark.mujoco_compatibility(False)
@pytest.mark.parametrize("mode", range(9))
@pytest.mark.parametrize("model_name", ["collision_edge_cases"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
@pytest.mark.parametrize("gjk_collision", [True, False])
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_collision_edge_cases(gs_sim, mode):
    qpos_0 = gs_sim.rigid_solver.get_dofs_position()
    for _ in range(200):
        gs_sim.scene.step()

    qvel = gs_sim.rigid_solver.get_dofs_velocity()
    assert_allclose(qvel, 0, atol=1e-2)
    qpos = gs_sim.rigid_solver.get_dofs_position()
    atol = 1e-3 if mode in (4, 6) else 1e-4
    assert_allclose(qpos[[0, 1, 3, 4, 5]], qpos_0[[0, 1, 3, 4, 5]], atol=atol)


@pytest.mark.slow  # ~200s
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
@pytest.mark.parametrize("precision", ["32"])
def test_mpr_thin_box_stack_no_lateral_phantom(show_viewer, tol):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            use_gjk_collision=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.1, -0.08, 0.06),
            camera_lookat=(0.0, 0.0, 0.01),
            camera_fov=20,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        gs.morphs.Box(
            pos=(0.0, 0.0, 0.005),
            size=(0.002, 0.02, 0.01),
            fixed=True,
        ),
        surface=gs.surfaces.Default(
            color=(0, 0, 1),
        ),
    )
    box = scene.add_entity(
        gs.morphs.Box(
            pos=(0.0, 0.0, 0.01495),
            size=(0.002, 0.0199, 0.01),
        ),
        surface=gs.surfaces.Default(
            color=(1, 0, 0),
        ),
        visualize_contact=True,
    )
    scene.build()

    scene.step()
    contacts = scene.rigid_solver.collider.get_contacts(to_torch=False)
    normals = contacts["normal"]
    assert len(normals) > 0
    assert_allclose(np.abs(normals[..., 2]), 1, atol=1e2 * tol)

    for _ in range(100):
        scene.step()
    pos = box.get_pos()
    assert_allclose(pos[..., :2], 0, atol=1e1 * tol)
    assert_allclose(pos[..., 2], 0.015, atol=1e1 * tol)


@pytest.mark.required
def test_box_on_terrain_no_spurious_spin(show_viewer):
    BOX_SIZE = (0.12, 0.06, 0.025)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
        ),
        rigid_options=gs.options.RigidOptions(
            box_box_detection=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, -3.5, 2.0),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=30,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(
        morph=gs.morphs.Terrain(
            pos=(-1.5, -1.5, 0.0),
            height_field=np.zeros((4, 4), dtype=np.float32),
            horizontal_scale=1.0,
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.8, 0.4),
        ),
    )
    box = scene.add_entity(
        morph=(
            gs.morphs.Box(size=BOX_SIZE),
            gs.morphs.MeshSet(
                files=(trimesh.creation.box(extents=BOX_SIZE),),
                decimate=False,
            ),
        ),
        surface=gs.surfaces.Default(
            color=(0.95, 0.2, 0.2),
        ),
    )

    # 4x4 grid spread across the 3 m x 3 m terrain.
    grid = np.linspace(-1.2, 1.2, 4)
    xy = np.stack(np.meshgrid(grid, grid, indexing="ij"), axis=-1).reshape(-1, 2)
    n_envs = xy.shape[0]
    scene.build(n_envs=n_envs)

    z = BOX_SIZE[2] / 2
    pos = np.concatenate([xy, np.full((n_envs, 1), z)], axis=-1).astype(np.float32)
    box.set_pos(torch.from_numpy(pos))
    box.set_dofs_velocity(torch.zeros((n_envs, 6)))
    quat_initial = tensor_to_array(box.get_quat())

    for _ in range(500):
        scene.step()

    quat_delta = gu.transform_quat_by_quat(tensor_to_array(box.get_quat()), gu.inv_quat(quat_initial))
    assert_allclose(gu.quat_to_rotvec(quat_delta), 0.0, tol=0.02)


@pytest.mark.required
def test_multicontact_sphere_vs_terrain(show_viewer, tol):
    GRID_N = 13
    APEX_IDX = GRID_N // 2
    VERTICAL_SCALE = 0.04
    HORIZONTAL_SCALE = 0.05
    SPHERE_RADIUS = 0.1

    ii, jj = np.meshgrid(np.arange(GRID_N), np.arange(GRID_N), indexing="ij")
    hf = (np.abs(ii - APEX_IDX) + np.abs(jj - APEX_IDX)).astype(np.int16)
    terrain_pos = (-APEX_IDX * HORIZONTAL_SCALE, -APEX_IDX * HORIZONTAL_SCALE, 0.0)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, -1.0, 0.8),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=30,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(
        morph=gs.morphs.Terrain(
            pos=terrain_pos,
            height_field=hf,
            horizontal_scale=HORIZONTAL_SCALE,
            vertical_scale=VERTICAL_SCALE,
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.8, 0.4),
        ),
    )
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(radius=SPHERE_RADIUS),
        surface=gs.surfaces.Default(
            color=(0.95, 0.2, 0.2),
        ),
        visualize_contact=True,
    )

    scene.build()

    sphere.set_pos(torch.tensor([0.0, 0.0, 0.25]))

    for _ in range(80):
        scene.step()
        print(tensor_to_array(sphere.get_dofs_velocity()))

    # Sphere is at rest at the apex of the pit. Equilibrium height is set by the four-wall solid angle: contacts on the
    # opposing pyramid walls push the sphere upward, so it sits above the apex vertex but well below the rim.
    pos_final = tensor_to_array(sphere.get_pos())
    assert_allclose(pos_final[:2], 0.0, tol=2.0 * HORIZONTAL_SCALE)
    assert SPHERE_RADIUS < pos_final[2] < APEX_IDX * VERTICAL_SCALE + SPHERE_RADIUS

    vel_final = tensor_to_array(sphere.get_dofs_velocity())
    assert_allclose(vel_final, 0.0, tol=1e-5)


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(gs.gpu, marks=pytest.mark.required),
        gs.cpu,  # This test takes too much time of CPU (~1000s)
    ],
)
@pytest.mark.parametrize("is_named", [True, False])
def test_terrain_generation(is_named, show_viewer, tol):
    TERRAIN_PATTERN = [
        ["flat_terrain", "flat_terrain", "flat_terrain", "flat_terrain", "flat_terrain"],
        ["flat_terrain", "fractal_terrain", "random_uniform_terrain", "sloped_terrain", "flat_terrain"],
        ["flat_terrain", "pyramid_sloped_terrain", "discrete_obstacles_terrain", "wave_terrain", "flat_terrain"],
        ["flat_terrain", "stairs_terrain", "pyramid_stairs_terrain", "stepping_stones_terrain", "flat_terrain"],
        ["flat_terrain", "flat_terrain", "flat_terrain", "flat_terrain", "flat_terrain"],
    ]
    TERRAIN_OFFSET = (10.0, -10.0, -1.0)
    TERRAIN_SIZE = 10.0
    SUBTERRAIN_GRID_SIZE = 15
    OBJ_SIZE = 0.1
    OBJ_HEIGHT_INIT = 0.3
    NUM_OBJ_SQRT = 15

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.006,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(-5.0 + TERRAIN_OFFSET[0], -5.0 + TERRAIN_OFFSET[1], 10.0 + TERRAIN_OFFSET[2]),
            camera_lookat=(5.0 + TERRAIN_OFFSET[0], 5.0 + TERRAIN_OFFSET[1], 0.0 + TERRAIN_OFFSET[2]),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    terrain_kwargs = dict(
        pos=TERRAIN_OFFSET,
        n_subterrains=(len(TERRAIN_PATTERN),) * 2,
        subterrain_size=(TERRAIN_SIZE / len(TERRAIN_PATTERN),) * 2,
        horizontal_scale=TERRAIN_SIZE / len(TERRAIN_PATTERN) / SUBTERRAIN_GRID_SIZE,
        vertical_scale=0.05,
        subterrain_types=TERRAIN_PATTERN,
        randomize=False,
        name="my_terrain" if is_named else None,
    )
    # FIXME: Collision detection is very unstable for 'stepping_stones' pattern.
    terrain = scene.add_entity(gs.morphs.Terrain(**terrain_kwargs))
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
    obj.set_pos(obj_pos_init_rel + torch.tensor(TERRAIN_OFFSET))

    # Drop the objects and simulate for a while.
    for _ in range(600):
        scene.step()

    # Check that objects are not moving anymore
    assert_allclose(obj.get_vel(), 0.0, tol=0.1)

    # Check the the terrain is not entirely flat and has the expected size
    terrain_min_corner, terrain_max_corner = tensor_to_array(terrain.geoms[0].get_AABB()) - TERRAIN_OFFSET
    assert_allclose(terrain_min_corner[:2], 0.0, tol=gs.EPS)
    assert_allclose(terrain_max_corner[:2], TERRAIN_SIZE, tol=gs.EPS)
    assert terrain_min_corner[2] < -1.0  # Stepping stone depth
    assert terrain_max_corner[2] > 0.01  # FIXME: It should not be larger than 'vertical_scale'

    # Check that all objects are in contact with the terrain
    obj_pos = tensor_to_array(obj.get_pos()) - TERRAIN_OFFSET
    terrain_mesh = terrain.geoms[0].mesh
    signed_distance, *_ = igl.signed_distance(obj_pos, terrain_mesh.verts, terrain_mesh.faces)
    assert (signed_distance > 0.0).all()
    assert (signed_distance < 2 * OBJ_SIZE).all()

    # Check if cache is being reloaded as expected
    if is_named:
        scene = gs.Scene()
        terrain_2 = scene.add_entity(gs.morphs.Terrain(**{**terrain_kwargs, **dict(randomize=True)}))
        terrain_2_mesh = terrain_2.geoms[0].mesh
        assert_allclose(terrain_mesh.verts, terrain_2_mesh.verts, tol=tol)


@pytest.mark.required
def test_terrain_discrete_obstacles():
    scene = gs.Scene()
    terrain = scene.add_entity(
        gs.morphs.Terrain(
            n_subterrains=(1, 1),
            subterrain_size=(6.0, 6.0),
            horizontal_scale=0.5,
            vertical_scale=0.5,
            subterrain_types=[["discrete_obstacles_terrain"]],
            subterrain_parameters={
                "discrete_obstacles_terrain": {
                    "max_height": 1.0,
                    "platform_size": 1.0,
                }
            },
        )
    )
    scene.build()
    height_field = terrain.geoms[0].metadata["height_field"]
    platform = height_field[5:7, 5:7]

    assert height_field.max() == 2.0
    assert height_field.min() == -2.0
    assert (platform < gs.EPS).all()


def test_mesh_to_heightfield(tmp_path, show_viewer):
    horizontal_scale = 2.0
    path_terrain = os.path.join(get_assets_dir(), "meshes", "terrain_45.obj")

    hf_terrain, xs, ys = tu.mesh_to_heightfield(path_terrain, spacing=horizontal_scale, oversample=1)

    # default heightfield starts at 0, 0, 0
    # translate to the center of the mesh
    translation = np.array([np.nanmin(xs), np.nanmin(ys), 0])

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(5, 0, -5),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -5, 7),
            camera_lookat=(10, 15, 4),
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
            pos=(10, 15, 7),
            radius=1,
        ),
        vis_mode="collision",
    )
    scene.build()

    for i in range(70):
        scene.step()

    # The ball is at rest (on the terrain)
    assert_allclose(ball.get_dofs_velocity(), 0, tol=1e-3)


@pytest.mark.required
def test_subterrain_parameters(show_viewer):
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

    assert_allclose((height_ref * 2.0), height_test, tol=gs.EPS)


@pytest.mark.required
@pytest.mark.merge_fixed_links(False)
@pytest.mark.parametrize("model_name", ["pendulum"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
def test_jacobian(gs_sim, tol):
    (pendulum,) = gs_sim.entities

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
@pytest.mark.parametrize("model_name", ["compound_joint"])
def test_jacobian_compound_joints(xml_path, tol):
    scene = gs.Scene(show_viewer=False)
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file=xml_path,
            requires_jac_and_IK=True,
        ),
    )
    scene.build()
    end_link = robot.get_link("seg2")

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    end_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "seg2")
    jacp = np.empty((3, mj_model.nv), dtype=np.float64)
    jacr = np.empty((3, mj_model.nv), dtype=np.float64)

    for qpos in (np.zeros(3), np.array([0.3, -0.5, 0.7])):
        robot.set_qpos(qpos.astype(gs.np_float))
        mj_data.qpos[:] = qpos
        mujoco.mj_forward(mj_model, mj_data)
        mujoco.mj_jacBody(mj_model, mj_data, jacp, jacr, end_body_id)

        assert_allclose(robot.get_jacobian(end_link), np.concatenate([jacp, jacr]), tol=tol)


@pytest.mark.slow  # ~250s
@pytest.mark.required
def test_mjcf_parsing_with_include():
    scene = gs.Scene()
    robot1 = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/scene.xml"))
    robot2 = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    robot3 = scene.add_entity(gs.morphs.MJCF(file="xml/franka_sim/franka_panda.xml"))
    scene.build()
    assert_allclose(robot1.get_qpos(), robot2.get_qpos(), tol=gs.EPS)
    assert_allclose(robot1.get_qpos(), robot3.get_qpos(), tol=gs.EPS)


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_urdf_parsing(show_viewer, tol):
    POS_OFFSET = 0.8
    WOLRD_QUAT = np.array([1.0, 1.0, -0.3, +0.3])
    DOOR_JOINT_DAMPING = 1.5

    scene = gs.Scene(
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

    def _check_entity_positions(expected_y_spacing, tol):
        # The four parsing configs are laid out 'expected_y_spacing' apart in y, so their world AABBs must coincide once
        # that spacing is removed. AABBs are world-frame, so this check is independent of the relative-getter frame.
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
                AABB_i = tensor_to_array(geom.get_AABB())
                AABB[0] = np.minimum(AABB[0], AABB_i[0])
                AABB[1] = np.maximum(AABB[1], AABB_i[1])
            AABB_all.append(AABB)
        AABB_diff = np.diff(AABB_all, axis=0)
        AABB_diff[..., 1] -= expected_y_spacing
        assert_allclose(AABB_diff, 0.0, tol=tol)

    # Check that `set_pos` / `set_quat` applies the same transform in all cases. Both frames place every config at the
    # same pose, so the world AABBs coincide with no residual spacing.
    for relative in (False, True):
        for key in ((False, False), (False, True), (True, False), (True, True)):
            entities[key].set_pos(np.array([0.5, 0.0, 0.0]), relative=relative)
            entities[key].set_quat(np.array([0.0, 0.0, 0.0, 1.0]), relative=relative)
        if show_viewer:
            scene.visualizer.update()
        _check_entity_positions(0.0, tol=tol)

    # Check that `set_qpos` applies the same absolute transform in all cases. The fixed roots have no free joint to
    # take a base pose via qpos, so they are placed at the matching absolute world pose with the (relative=False)
    # setters. All four configs then sit POS_OFFSET apart in y, as at creation.
    door_angle = np.array([1.1])
    world_quat = tuple(WOLRD_QUAT / np.linalg.norm(WOLRD_QUAT))
    for i, key in enumerate(((False, False), (False, True))):
        qpos = np.concatenate(((0.0, (i - 1.5) * POS_OFFSET, 0.0), world_quat, door_angle))
        entities[key].set_qpos(qpos)
    for i, key in enumerate(((True, False), (True, True))):
        config_y = ((i + 2) - 1.5) * POS_OFFSET
        entities[key].set_pos(np.array([0.0, config_y, 0.0]), relative=False)
        entities[key].set_quat(np.array(world_quat), relative=False)
        entities[key].set_qpos(door_angle)
    if show_viewer:
        scene.visualizer.update()
    _check_entity_positions(POS_OFFSET, tol=tol)

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
        door_pos_diff = torch.diff(torch.concatenate(door_pos_all))
        assert_allclose(door_pos_diff, 0, tol=5e-3)
    assert_allclose(scene.rigid_solver.dofs_state.vel.to_numpy(), 0.0, tol=1e-3)
    _check_entity_positions(POS_OFFSET, tol=2e-3)


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize("model_name", ["undefined_inertia"])
def test_urdf_parsing_undefined_inertia(xml_path, show_viewer):
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, 0.5, 0.5),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())

    entity = scene.add_entity(
        morph=gs.morphs.URDF(
            file=xml_path,
            pos=(0.0, 0.0, 0.1),
        )
    )

    scene.build()

    for i in range(30):
        scene.step()
    assert_allclose(entity.get_pos(), (0, 0, 0.03), tol=1e-3)


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize("urdf_path", ["chain.urdf", "dual_arms_glb/dual_arms_glb.urdf", "dual_arms_primitives.urdf"])
@pytest.mark.parametrize("fixed", [False, True])
def test_urdf_parsing_merge_fixed_links(urdf_path, fixed, show_viewer, tol):
    POS = (0.0, -0.2, 0.5)
    EULER = (0.0, 90.0, 45.0)

    scene = gs.Scene(
        show_viewer=show_viewer,
    )
    urdf_rootdir = os.path.dirname(urdf_path)
    asset_path = get_hf_dataset(pattern=os.path.join(urdf_rootdir, "*") if urdf_rootdir else urdf_path)
    robot_1 = scene.add_entity(
        gs.morphs.URDF(
            file=os.path.join(asset_path, urdf_path),
            pos=POS,
            euler=EULER,
            fixed=fixed,
            recompute_inertia=True,
            merge_fixed_links=False,
        ),
        surface=gs.surfaces.Default(
            color=(1, 0, 0, 0.5),
        ),
    )
    robot_2 = scene.add_entity(
        gs.morphs.URDF(
            file=os.path.join(asset_path, urdf_path),
            pos=POS,
            euler=EULER,
            fixed=fixed,
            recompute_inertia=True,
            merge_fixed_links=True,
        ),
        surface=gs.surfaces.Default(
            color=(0, 1, 0, 0.5),
        ),
    )
    scene.build()

    assert_allclose(robot_1.get_pos(), POS, tol=tol)
    assert_allclose(robot_1.get_quat(), gu.euler_to_quat(EULER), tol=tol)

    for _ in range(2):
        assert_allclose(robot_1.get_pos(), robot_2.get_pos(), tol=tol)
        assert_allclose(robot_1.get_quat(), robot_2.get_quat(), tol=tol)
        for link_2 in robot_2.links:
            link_1 = robot_1.get_link(link_2.name)
            assert_allclose(link_1.get_pos(), link_2.get_pos(), tol=tol)
            quat_1, quat_2 = link_1.get_quat(), link_2.get_quat()
            if quat_1[0] * quat_2[0] < 0.0:
                quat_2[:] *= -1.0
            assert_allclose(quat_1, quat_2, tol=tol)

        pos0 = np.random.rand(3)
        quat0 = np.random.rand(4)
        for robot in (robot_1, robot_2):
            robot.set_pos(pos0)
            robot.set_quat(quat0)

    com_robot_1, com_robot_2 = scene.rigid_solver.get_links_root_COM(
        links_idx=(robot_1.base_link_idx, robot_2.base_link_idx)
    )
    assert_allclose(com_robot_1, com_robot_2, tol=tol)


@pytest.fixture(scope="session")
def box_freejoint_offset():
    mjcf = ET.Element("mujoco", model="test_freejoint")
    worldbody = ET.SubElement(mjcf, "worldbody")

    base_body = ET.SubElement(worldbody, "body", name="base", pos="0 0 1.0", quat="1.0 0 0 1.0")
    ET.SubElement(base_body, "freejoint", name="root")
    ET.SubElement(base_body, "inertial", pos="0 0 0", mass="1.0", diaginertia="0.01 0.01 0.01")
    ET.SubElement(base_body, "geom", type="box", size="0.05 0.05 0.05")

    child_body = ET.SubElement(base_body, "body", name="child", pos="0 0 0.1")
    ET.SubElement(child_body, "inertial", pos="0 0 0", mass="0.5", diaginertia="0.001 0.001 0.001")
    ET.SubElement(child_body, "joint", name="joint1", type="hinge", axis="0 1 0")
    ET.SubElement(child_body, "geom", type="box", size="0.03 0.03 0.05")

    return mjcf


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["box_freejoint_offset"])
def test_mjcf_parsing_merge_fixed_links(xml_path, show_viewer):
    """Test that get_pos reflects set_qpos for MJCF robots with freejoint and non-zero initial body position."""
    POS = (1.0, 2.0, 3.0)
    QUAT = (0.0, 1.0, 0.0, 0.0)

    scene = gs.Scene(
        show_viewer=show_viewer,
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file=xml_path,
        )
    )
    scene.build()

    assert_allclose(robot.get_pos(), (0.0, 0.0, 1.0), tol=gs.EPS)
    assert_allclose(robot.get_quat(), np.array([1.0, 0.0, 0.0, 1.0]) / math.sqrt(2), tol=gs.EPS)

    robot.set_qpos((*POS, *QUAT), qs_idx_local=slice(None, 7))
    assert_allclose(robot.get_pos(), POS, tol=gs.EPS)
    assert_allclose(robot.get_quat(), QUAT, tol=gs.EPS)

    scene.reset()
    assert_allclose(robot.get_pos(), (0.0, 0.0, 1.0), tol=gs.EPS)
    assert_allclose(robot.get_quat(), np.array([1.0, 0.0, 0.0, 1.0]) / math.sqrt(2), tol=gs.EPS)

    robot.set_pos(POS)
    robot.set_quat(QUAT)
    assert_allclose(robot.get_pos(), POS, tol=gs.EPS)
    assert_allclose(robot.get_quat(), QUAT, tol=gs.EPS)


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_urdf_capsule(tmp_path, show_viewer, tol):
    urdf_path = tmp_path / "capsule.urdf"
    with open(urdf_path, "w") as f:
        f.write(
            """
            <robot name="urdf_robot">
                <link name="base_link">
                    <inertial>
                        <origin rpy="0 0 0" xyz="0 0 0"/>
                        <mass value=".1"/>
                        <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
                    </inertial>
                    <collision>
                        <origin rpy="0 0 0" xyz="0 0 0"/>
                        <geometry>
                            <capsule length="0.1" radius="0.02"/>
                        </geometry>
                    </collision>
                    <visual>
                        <origin rpy="0 0 0" xyz="0 0 0"/>
                        <geometry>
                            <capsule length="0.06" radius="0.03"/>
                        </geometry>
                    </visual>
                </link>
            </robot>
            """
        )

    scene = gs.Scene(show_viewer=show_viewer)
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=urdf_path,
            pos=(0.0, 0.0, 0.3),
        ),
        vis_mode="collision",
    )
    scene.build()

    (geom,) = robot.geoms
    assert geom.type == gs.GEOM_TYPE.CAPSULE
    assert_allclose(geom.data[:2], (0.02, 0.1), tol=gs.EPS)

    for _ in range(40):
        scene.step()
    geom_verts = tensor_to_array(geom.get_verts())
    assert np.linalg.norm(geom_verts - (0.0, 0.0, 0.0), axis=-1, ord=np.inf).min() < 1e-3
    assert np.linalg.norm(geom_verts - (0.0, 0.0, 0.14), axis=-1, ord=np.inf).min() < 1e-3

    (vgeom,) = robot.vgeoms
    vgeom_verts = tensor_to_array(vgeom.get_vverts())
    # Visual is a capsule (length=0.06, radius=0.03, total height 0.12) centered on the link, so after
    # the collision capsule settles against the plane (link at z=0.07), the visual spans z in [0.01, 0.13].
    assert np.linalg.norm(vgeom_verts - (0.0, 0.0, 0.01), axis=-1, ord=np.inf).min() < 1e-3
    assert np.linalg.norm(vgeom_verts - (0.0, 0.0, 0.13), axis=-1, ord=np.inf).min() < 1e-3


@pytest.mark.required
@pytest.mark.required
@pytest.mark.parametrize("overwrite", [False, True])
def test_color_overwrite(overwrite, show_viewer):
    scene = gs.Scene(show_viewer=show_viewer)
    box = scene.add_entity(
        gs.morphs.URDF(
            file="genesis/assets/urdf/blue_box/model.urdf",
            convexify=False,
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.0, 0.0, 1.0) if overwrite else None,
        ),
    )
    asset_path = get_hf_dataset(pattern="chain.urdf")
    chain = scene.add_entity(
        gs.morphs.URDF(
            file=f"{asset_path}/chain.urdf",
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0, 0, 1.0) if overwrite else None,
        ),
    )
    axis = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            convexify=False,
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.0, 0.0, 1.0) if overwrite else None,
        ),
    )
    asset_path = get_hf_dataset(pattern="work_table.glb")
    table = scene.add_entity(
        gs.morphs.Mesh(
            file=f"{asset_path}/work_table.glb",
            convexify=False,
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.0, 0.0, 1.0) if overwrite else None,
        ),
    )
    asset_path = get_hf_dataset(pattern="humanoid.xml")
    humanoid = scene.add_entity(
        gs.morphs.MJCF(
            file=f"{asset_path}/humanoid.xml",
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.0, 0.0, 1.0) if overwrite else None,
        ),
    )
    if show_viewer:
        scene.build()
    for vgeom in box.vgeoms:
        assert vgeom.vmesh.metadata["is_visual_overwritten"] == overwrite
        visual = vgeom.vmesh.trimesh.visual
        assert visual.defined
        color = np.unique(visual.vertex_colors, axis=0)
        assert_equal(color, (255, 0, 0, 255) if overwrite else (0, 0, 255, 255))
    for vgeom in chain.vgeoms:
        assert vgeom.vmesh.metadata["is_visual_overwritten"] == overwrite
        visual = vgeom.vmesh.trimesh.visual
        assert visual.defined
        color = np.unique(visual.vertex_colors, axis=0)
        assert_equal(color, (255, 0, 0, 255) if overwrite else (51, 51, 51, 255))
    for vgeom in humanoid.vgeoms:
        # FIXME: The original material is lost because the visuals are collision geometries that has been duplicated as
        # visual to circumvent the lack of dedicated visuals.
        is_true_visual = vgeom.vmesh.metadata["name"] == "nose"
        assert vgeom.vmesh.metadata["is_visual_overwritten"] == overwrite or not is_true_visual
        visual = vgeom.vmesh.trimesh.visual
        assert visual.defined
        color = np.unique(visual.vertex_colors, axis=0)
        if is_true_visual:
            if overwrite:
                assert_equal(color, (255, 0, 0, 255))
            else:
                with pytest.raises(AssertionError):
                    assert_equal(color, (128, 128, 128, 255))
        else:
            assert_equal(color, (255, 0, 0, 255) if overwrite else (128, 128, 128, 255))
    for vgeom in axis.vgeoms:
        assert vgeom.vmesh.metadata["is_visual_overwritten"] == overwrite
        visual = vgeom.vmesh.trimesh.visual
        assert visual.defined
        color = np.unique(visual.vertex_colors, axis=0)
        if overwrite:
            assert_equal(color, (255, 0, 0, 255))
        else:
            assert_equal(color, [[0, 0, 178, 255], [0, 178, 0, 255], [178, 0, 0, 255], [255, 255, 255, 255]])
    for vgeom in table.vgeoms:
        assert vgeom.vmesh.metadata["is_visual_overwritten"] == overwrite
        visual = vgeom.vmesh.trimesh.visual
        assert visual.defined
        if overwrite:
            color = np.unique(visual.vertex_colors, axis=0)
            assert_equal(color, (255, 0, 0, 255))
    for entity in scene.entities:
        for geom in entity.geoms:
            assert geom.mesh.metadata["is_visual_overwritten"]
            visual = geom.mesh.trimesh.visual
            assert visual.defined
            color = np.unique(visual.vertex_colors, axis=0)
            # Collision geometry meshes have randomized colors with partial transparency to ease debugging
            with pytest.raises(AssertionError):
                assert_equal(color, (255, 0, 0, 255))


@pytest.mark.required
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
@pytest.mark.parametrize("model_name", ["pendulum_with_joint_dynamics"])
@pytest.mark.parametrize("joint_damping, joint_friction", [(1.0, 2.0)])
def test_urdf_joint_dynamics(joint_damping, joint_friction, xml_path):
    scene = gs.Scene()
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=xml_path,
            pos=(0, 0, 0.8),
            convexify=True,
        ),
    )
    assert_allclose(robot.joints[0].dofs_damping, 0.0, tol=gs.EPS)
    assert_allclose(robot.joints[1].dofs_damping, joint_damping, tol=gs.EPS)
    assert_allclose(robot.joints[0].dofs_frictionloss, 0.0, tol=gs.EPS)
    assert_allclose(robot.joints[1].dofs_frictionloss, joint_friction, tol=gs.EPS)


@pytest.fixture(scope="session")
def freeflyer_mjcf():
    mjcf = ET.Element("mujoco", model="freeflyer")
    worldbody = ET.SubElement(mjcf, "worldbody")
    body = ET.SubElement(worldbody, "body", name="base", pos="0 0 1")
    ET.SubElement(body, "joint", type="free")
    ET.SubElement(body, "inertial", pos="0 0 0", mass="1.0", diaginertia="0.01 0.01 0.01")
    ET.SubElement(body, "geom", type="sphere", size="0.05")
    child = ET.SubElement(body, "body", name="child", pos="0 0 0.1")
    ET.SubElement(child, "joint", type="hinge", axis="0 1 0")
    ET.SubElement(child, "inertial", pos="0 0 0", mass="0.5", diaginertia="0.001 0.001 0.001")
    ET.SubElement(child, "geom", type="sphere", size="0.02")
    grandchild = ET.SubElement(child, "body", name="grandchild", pos="0 0 0.1")
    ET.SubElement(grandchild, "joint", type="slide", axis="1 0 0", armature="42.0")
    ET.SubElement(grandchild, "inertial", pos="0 0 0", mass="0.1", diaginertia="0.0001 0.0001 0.0001")
    ET.SubElement(grandchild, "geom", type="sphere", size="0.01")
    return mjcf


@pytest.fixture(scope="session")
def freeflyer_urdf():
    robot = ET.Element("robot", name="freeflyer")
    ET.SubElement(robot, "link", name="world")
    base_link = ET.SubElement(robot, "link", name="base_link")
    inertial = ET.SubElement(base_link, "inertial")
    ET.SubElement(inertial, "origin", rpy="0 0 0", xyz="0 0 0")
    ET.SubElement(inertial, "mass", value="1.0")
    ET.SubElement(inertial, "inertia", ixx="0.01", ixy="0", ixz="0", iyy="0.01", iyz="0", izz="0.01")
    collision = ET.SubElement(base_link, "collision")
    ET.SubElement(ET.SubElement(collision, "geometry"), "sphere", radius="0.05")
    root_joint = ET.SubElement(robot, "joint", name="root", type="floating")
    ET.SubElement(root_joint, "parent", link="world")
    ET.SubElement(root_joint, "child", link="base_link")
    child_link = ET.SubElement(robot, "link", name="child_link")
    child_inertial = ET.SubElement(child_link, "inertial")
    ET.SubElement(child_inertial, "origin", rpy="0 0 0", xyz="0 0 0")
    ET.SubElement(child_inertial, "mass", value="0.5")
    ET.SubElement(child_inertial, "inertia", ixx="0.001", ixy="0", ixz="0", iyy="0.001", iyz="0", izz="0.001")
    child_collision = ET.SubElement(child_link, "collision")
    ET.SubElement(ET.SubElement(child_collision, "geometry"), "sphere", radius="0.02")
    arm_joint = ET.SubElement(robot, "joint", name="arm", type="revolute")
    ET.SubElement(arm_joint, "parent", link="base_link")
    ET.SubElement(arm_joint, "child", link="child_link")
    ET.SubElement(arm_joint, "origin", rpy="0 0 0", xyz="0 0 0.1")
    ET.SubElement(arm_joint, "axis", xyz="0 1 0")
    ET.SubElement(arm_joint, "limit", lower="-3.14", upper="3.14", effort="10", velocity="10")
    return robot


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize("model_name", ["freeflyer_mjcf", "freeflyer_urdf"])
def test_default_armature_freeflyer(xml_path):
    DEFAULT_ARMATURE = 1000.0

    if xml_path.endswith(".urdf"):
        morph = gs.morphs.URDF(
            file=xml_path,
            default_armature=DEFAULT_ARMATURE,
        )
    else:
        morph = gs.morphs.MJCF(
            file=xml_path,
            default_armature=DEFAULT_ARMATURE,
        )

    scene = gs.Scene()
    robot = scene.add_entity(morph)
    scene.build()

    armature = robot.get_dofs_armature()
    assert_allclose(armature[:6], 0.0, tol=gs.EPS)
    assert_allclose(armature[6], DEFAULT_ARMATURE, tol=gs.EPS)
    if xml_path.endswith(".mjcf"):
        assert abs(armature[7]) > gs.EPS and abs(armature[7] - DEFAULT_ARMATURE) > gs.EPS


@pytest.mark.required
def test_gravity(show_viewer, tol):
    scene = gs.Scene(
        show_viewer=show_viewer,
    )

    sphere = scene.add_entity(gs.morphs.Sphere())
    scene.build(n_envs=3)

    scene.sim.set_gravity(torch.tensor([0.0, 0.0, 0.0]))
    scene.sim.set_gravity(torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]), envs_idx=[0, 1])
    scene.sim.set_gravity(torch.tensor([0.0, 0.0, 3.0]), envs_idx=2)
    with np.testing.assert_raises(RuntimeError):
        scene.sim.set_gravity(torch.tensor([0.0, -10.0]))
    with np.testing.assert_raises(RuntimeError):
        scene.sim.set_gravity(torch.tensor([[0.0, 0.0, -10.0], [0.0, 0.0, -10.0]]), envs_idx=1)

    scene.step()

    assert_allclose(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        sphere.get_links_acc()[..., 0, :],
        tol=tol,
    )


@pytest.mark.slow  # ~350s
@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_scene_saver_franka(tmp_path, show_viewer, tol):
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

    ckpt_path = tmp_path / "franka_unit.pkl"
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
def test_drone_propellers_force_substep_consistency(show_viewer, tol):
    BASE_RPM = 15000

    scene_ref = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.004,
            substeps=1,
        ),
        show_viewer=show_viewer,
    )
    drone_ref = scene_ref.add_entity(
        morph=gs.morphs.Drone(
            file="urdf/drones/cf2x.urdf",
            pos=(0, 0, 1),
        ),
    )
    scene_ref.build(n_envs=2)

    # This not only tests setter, but also proper reset (tracking and clearing applied external force)
    drone_ref.set_propellers_rpm(BASE_RPM)
    with np.testing.assert_raises(gs.GenesisException):
        drone_ref.set_propellers_rpm(BASE_RPM)
    scene_ref.reset()
    drone_ref.set_propellers_rpm((BASE_RPM,) * 4)
    scene_ref.reset()
    drone_ref.set_propellers_rpm(torch.full((scene_ref.n_envs, 4), fill_value=BASE_RPM))
    scene_ref.reset()

    for _ in range(500):
        drone_ref.set_propellers_rpm(BASE_RPM)
        scene_ref.step()

    scene_test = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.02,
            substeps=5,
        ),
        show_viewer=show_viewer,
    )
    drone_test = scene_test.add_entity(
        morph=gs.morphs.Drone(
            file="urdf/drones/cf2x.urdf",
            pos=(0, 0, 1.0),
        ),
    )
    scene_test.build()
    for _ in range(100):
        drone_test.set_propellers_rpm(BASE_RPM)
        scene_test.step()

    pos_ref = drone_ref.get_dofs_position()
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
            drone.set_propellers_rpm(50000.0)
        scene.step()
        if i > 350:
            assert scene.rigid_solver.collider._collider_state.n_contacts.to_numpy()[0] == 2
            assert_allclose(scene.rigid_solver.get_dofs_velocity(), 0, tol=2e-3)

    # Push the drones symmetrically and wait for them to collide
    drones[0].set_dofs_velocity([0.2], [1])
    drones[1].set_dofs_velocity([-0.2], [1])
    for i in range(150):
        for drone in drones:
            drone.set_propellers_rpm(50000.0)
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


@pytest.mark.slow  # ~200s
@pytest.mark.required
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
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(0.2, 0.0, 0.05),
        )
    )
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


@pytest.mark.slow  # ~500s
@pytest.mark.required
@pytest.mark.parametrize("precision", ["32", "64"])
@pytest.mark.parametrize("backend", [gs.gpu])
def test_cholesky_tiling(monkeypatch, tol):
    import genesis.engine.solvers

    rigid_solver_build_orig = genesis.engine.solvers.RigidSolver.build

    values = []
    for enable_tiled_cholesky in (True, False):

        def rigid_solver_build(self):
            nonlocal enable_tiled_cholesky

            rigid_solver_build_orig(self)
            self._static_rigid_sim_config.enable_tiled_cholesky_mass_matrix = enable_tiled_cholesky
            self._static_rigid_sim_config.enable_tiled_cholesky_hessian = enable_tiled_cholesky
            if enable_tiled_cholesky:
                self._static_rigid_sim_config.tiled_n_dofs_per_entity = 32
                self._static_rigid_sim_config.tiled_n_dofs = 32

        monkeypatch.setattr("genesis.engine.solvers.RigidSolver.build", rigid_solver_build)

        scene = gs.Scene(
            rigid_options=gs.options.RigidOptions(
                constraint_solver=gs.constraint_solver.Newton,
                sparse_solve=False,
                iterations=1,
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
        scene.build(n_envs=2)
        assert scene.rigid_solver._static_rigid_sim_config.enable_tiled_cholesky_mass_matrix == enable_tiled_cholesky
        assert scene.rigid_solver._static_rigid_sim_config.enable_tiled_cholesky_hessian == enable_tiled_cholesky

        scene.step()
        assert not scene.rigid_solver.get_error_envs_mask().any()
        assert (scene.rigid_solver.constraint_solver.constraint_state.n_constraints.to_numpy() > 0).all()

        Mgrad = scene.rigid_solver.constraint_solver.constraint_state.Mgrad.to_numpy()
        assert np.linalg.norm(Mgrad) > 5.0
        values.append(Mgrad)

    # analysis for choice tolerance: https://github.com/Genesis-Embodied-AI/Genesis/pull/2659#discussion_r3041684256
    assert_allclose(*values, tol=5e-4)


@pytest.mark.slow  # ~200s
@pytest.mark.precision("32")
@pytest.mark.parametrize("backend", [gs.cuda])
def test_cholesky_tiling_large_shared_memory(show_viewer):
    if gs.device.type != "cuda":
        pytest.skip("Requires CUDA device")

    from cuda.bindings import runtime  # Transitive dependency of torch CUDA

    _, max_shared_mem = runtime.cudaDeviceGetAttribute(
        runtime.cudaDeviceAttr.cudaDevAttrMaxSharedMemoryPerBlockOptin, gs.device.index
    )
    if max_shared_mem <= 49152:
        pytest.skip("GPU does not support opt-in shared memory beyond the default 48kB")

    # Stack 17 free boxes (6 DOFs each = 102 total) to exceed the default 48kB tiling limit of 96 DOFs for f32
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 1.0, 2.5),
            camera_lookat=(0.0, 0.0, 1.2),
        ),
        rigid_options=gs.options.RigidOptions(
            constraint_solver=gs.constraint_solver.Newton,
            sparse_solve=False,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(gs.morphs.Plane())
    for i in range(17):
        scene.add_entity(
            gs.morphs.Box(
                size=(0.1, 0.1, 0.1),
                pos=(0, 0, 0.5 + i * 0.15),
            )
        )
    scene.build(n_envs=2)

    assert scene.rigid_solver.n_dofs == 102
    assert scene.rigid_solver._static_rigid_sim_config.enable_tiled_cholesky_hessian

    scene.step()
    assert not scene.rigid_solver.get_error_envs_mask().any()


@pytest.mark.slow  # ~200s
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
    gs_geom = gs_link.geoms[0]
    gs_vgeom = gs_link.vgeoms[0]
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
    # * Compare first 'Get' ouput with Quadrants value
    # Then, for any possible combinations of row and column masking:
    # * Call 'Get' -> Call 'Set' with 'Get' output -> Call 'Get'
    # * Compare first 'Get' output with last 'Get' output
    # * Compare last 'Get' output with corresponding slice of non-masking 'Get' output
    def get_all_supported_masks(i, max_length):
        if max_length <= 0 or i > max_length - 1:
            return (None,)
        if i == max_length - 1:
            return (
                i,
                [i],
                slice(i, i + 1),
                range(i, i + 1),
                np.array([i], dtype=np.int32),
                torch.tensor([i], dtype=torch.int64),
                torch.tensor([i], dtype=gs.tc_int, device=gs.device),
            )
        return (
            [i, i + 1],
            slice(i, i + 2),
            range(i, i + 2),
            np.array([i, i + 1], dtype=np.int32),
            torch.tensor([i, i + 1], dtype=torch.int64),
            torch.tensor([i, i + 1], dtype=gs.tc_int, device=gs.device),
        )

    def must_cast(value, dtype):
        return not (
            isinstance(value, torch.Tensor)
            and value.is_contiguous()
            and value.dtype == dtype
            and value.device == gs.device
        )

    for arg1_max, arg2_max, getter_or_spec, setter, qd_data in (
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
        (gs_s.n_dofs, -1, gs_s.get_dofs_limit, gs_s.set_dofs_limit, gs_s.dofs_info.limit),
        (gs_s.n_dofs, -1, gs_s.get_dofs_stiffness, gs_s.set_dofs_stiffness, gs_s.dofs_info.stiffness),
        (gs_s.n_dofs, -1, gs_s.get_dofs_invweight, None, gs_s.dofs_info.invweight),
        (gs_s.n_dofs, -1, gs_s.get_dofs_armature, gs_s.set_dofs_armature, gs_s.dofs_info.armature),
        (gs_s.n_dofs, -1, gs_s.get_dofs_damping, gs_s.set_dofs_damping, gs_s.dofs_info.damping),
        (gs_s.n_dofs, -1, gs_s.get_dofs_frictionloss, gs_s.set_dofs_frictionloss, gs_s.dofs_info.frictionloss),
        (gs_s.n_dofs, -1, gs_s.get_dofs_kp, gs_s.set_dofs_kp, gs_s.dofs_info.act_gain),
        (gs_s.n_dofs, -1, gs_s.get_dofs_kv, gs_s.set_dofs_kv, None),
        (gs_s.n_dofs, -1, gs_s.get_dofs_act_bias, gs_s.set_dofs_act_bias, gs_s.dofs_info.act_bias),
        (gs_s.n_dofs, -1, gs_s.get_dofs_act_gain, gs_s.set_dofs_act_gain, gs_s.dofs_info.act_gain),
        (gs_s.n_geoms, n_envs, gs_s.get_geoms_pos, None, gs_s.geoms_state.pos),
        (gs_s.n_geoms, n_envs, gs_s.get_geoms_quat, None, gs_s.geoms_state.quat),
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
        (gs_robot.n_dofs, -1, gs_robot.get_dofs_frictionloss, gs_robot.set_dofs_frictionloss, None),
        (gs_robot.n_dofs, -1, gs_robot.get_dofs_kp, gs_robot.set_dofs_kp, None),
        (gs_robot.n_dofs, -1, gs_robot.get_dofs_kv, gs_robot.set_dofs_kv, None),
        (gs_robot.n_dofs, -1, gs_robot.get_dofs_act_bias, gs_robot.set_dofs_act_bias, None),
        (gs_robot.n_dofs, -1, gs_robot.get_dofs_act_gain, gs_robot.set_dofs_act_gain, None),
        (gs_robot.n_qs, n_envs, gs_robot.get_qpos, gs_robot.set_qpos, None),
        (-1, n_envs, gs_robot.get_mass_mat, None, None),
        (-1, n_envs, gs_robot.get_links_net_contact_force, None, None),
        (-1, n_envs, gs_robot.get_pos, gs_robot.set_pos, None),
        (-1, n_envs, gs_robot.get_quat, gs_robot.set_quat, None),
        (-1, -1, gs_robot.get_mass, gs_robot.set_mass, None),
        (-1, -1, gs_robot.get_verts, None, None),
        (-1, -1, gs_robot.get_AABB, None, None),
        (-1, -1, gs_robot.get_vAABB, None, None),
        # LINK
        (-1, -1, gs_link.get_pos, None, None),
        (-1, -1, gs_link.get_quat, None, None),
        (-1, -1, gs_link.get_mass, gs_link.set_mass, None),
        (-1, -1, gs_link.get_verts, None, None),
        (-1, -1, gs_link.get_AABB, None, None),
        (-1, -1, gs_link.get_vAABB, None, None),
        # GEOM
        (-1, -1, gs_geom.get_pos, None, None),
        (-1, -1, gs_geom.get_quat, None, None),
        (-1, -1, gs_geom.get_verts, None, None),
        (-1, -1, gs_geom.get_AABB, None, None),
        # VGEOM
        (-1, -1, gs_vgeom.get_pos, None, None),
        (-1, -1, gs_vgeom.get_quat, None, None),
        (-1, -1, gs_vgeom.get_vAABB, None, None),
    ):
        getter, spec = (getter_or_spec, None) if callable(getter_or_spec) else (None, getter_or_spec)

        # Restore PD consistency before each iteration (act_gain/act_bias setters may have broken it)
        gs_s.set_dofs_kp(0.0)
        gs_s.set_dofs_kv(0.0)

        # Check getter and setter without row or column masking
        if getter is not None:
            datas = deepcopy(getter())
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
        if qd_data is not None:
            true = qd_to_torch(qd_data)
            qd_ndim = getattr(qd_data, "ndim", len(getattr(qd_data, "element_shape", ())))
            true = true.movedim(true.ndim - qd_ndim - 1, 0)
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
                datas = torch.as_tensor(datas, dtype=gs.tc_float)
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
            if i is not None:
                mask_i = [i, i + 1] if i < arg1_max - 1 else [i]
            for arg1 in get_all_supported_masks(i, arg1_max):
                for j in range(max(arg2_max, 1)) if arg2_max >= 0 else (None,):
                    if j is not None:
                        mask_j = [j, j + 1] if j < arg2_max - 1 else [j]
                    for arg2 in get_all_supported_masks(j, arg2_max):
                        if arg1 is None and arg2 is not None:
                            if getter is not None:
                                data = deepcopy(getter(arg2))
                            else:
                                if is_tuple:
                                    data = [torch.ones((len(mask_j), *shape)) for shape in spec]
                                else:
                                    data = torch.ones((len(mask_j), *spec))
                            if setter is not None:
                                setter(data, arg2)
                            if n_envs:
                                if is_tuple:
                                    data_ = [val[mask_j] for val in datas]
                                else:
                                    data_ = datas[mask_j]
                            else:
                                data_ = datas
                        elif arg1 is not None and arg2 is None:
                            if getter is not None:
                                data = deepcopy(getter(arg1))
                            else:
                                if is_tuple:
                                    data = [torch.ones((len(mask_i), *shape)) for shape in spec]
                                else:
                                    data = torch.ones((len(mask_i), *spec))
                            if setter is not None:
                                if is_tuple:
                                    setter(*data, arg1)
                                else:
                                    setter(data, arg1)
                            if is_tuple:
                                data_ = [val[mask_i] for val in datas]
                            else:
                                data_ = datas[mask_i]
                        else:
                            if getter is not None:
                                data = deepcopy(getter(arg1, arg2))
                            else:
                                if is_tuple:
                                    data = [torch.ones((len(mask_j), len(mask_i), *shape)) for shape in spec]
                                else:
                                    data = torch.ones((len(mask_j), len(mask_i), *spec))
                            if setter is not None:
                                setter(data, arg1, arg2)
                            if is_tuple:
                                data_ = [val[mask_j, :][:, mask_i] for val in datas]
                            else:
                                data_ = datas[mask_j, :][:, mask_i]
                        # FIXME: Not sure why tolerance must be increased for tests to pass
                        assert_allclose(data_, data, tol=(5.0 * tol))

    for dofs_idx in (*get_all_supported_masks(0, gs_s.n_dofs), None):
        for envs_idx in (*(get_all_supported_masks(0, gs_s.n_dofs) if n_envs > 0 else ()), None):
            dofs_pos = gs_s.get_dofs_position(dofs_idx, envs_idx)
            dofs_vel = gs_s.get_dofs_velocity(dofs_idx, envs_idx)
            gs_s.control_dofs_position(dofs_pos, dofs_idx, envs_idx)
            gs_s.control_dofs_velocity(dofs_vel, dofs_idx, envs_idx)

    # Must be tested independently because of non-trival return type
    gs_robot.get_contacts()


@pytest.mark.required
def test_deprecated_properties(caplog):
    scene = gs.Scene(
        show_viewer=False,
        show_FPS=False,
    )
    box = scene.add_entity(
        gs.morphs.Box(
            size=(1.0, 1.0, 1.0),
            pos=(0.0, 0.0, 0.0),
        )
    )
    scene.build()

    joint = box.joints[0]

    # Verify introspection doesn't trigger warnings
    caplog.clear()
    with caplog.at_level("WARNING"):
        repr(joint)
        vars(joint)
    assert len(caplog.records) == 0

    for name_old, name_new in (
        ("dof_idx", "dofs_idx"),
        ("dof_idx_local", "dofs_idx_local"),
        ("q_idx", "qs_idx"),
        ("q_idx_local", "qs_idx_local"),
    ):
        # Make sure that deprecated properties are hidden
        assert name_old not in dir(joint)

        # Verify deprecated properties emit warnings but work correctly
        caplog.clear()
        with caplog.at_level("WARNING"):
            deprecated_value = getattr(joint, name_old)
        assert len(caplog.records) > 0
        assert_allclose(deprecated_value, getattr(joint, name_new), tol=gs.EPS)


@pytest.mark.required
@pytest.mark.parametrize("enable_mujoco_compatibility", [True, False])
def test_getter_vs_state_post_step_consistency(enable_mujoco_compatibility):
    DT = 0.01
    GRAVITY = 10.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, GRAVITY),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_mujoco_compatibility=enable_mujoco_compatibility,
        ),
    )
    box = scene.add_entity(
        gs.morphs.Box(
            size=(1.0, 1.0, 1.0),
            pos=(0.0, 0.0, 0.0),
        )
    )
    (box_link,) = box.links
    scene.build()

    scene.step()
    dof_vel = scene.rigid_solver.get_dofs_velocity()
    assert_allclose(dof_vel[:3], (0.0, 0.0, GRAVITY * DT), atol=gs.EPS)
    vel = box_link.get_vel()
    with pytest.raises(AssertionError) if enable_mujoco_compatibility else nullcontext():
        assert_allclose(dof_vel[:3], vel, atol=gs.EPS)
    dof_pos = scene.rigid_solver.get_qpos()
    assert_allclose(dof_pos[:3], (0.0, 0.0, GRAVITY * DT**2), atol=gs.EPS)
    pos = box_link.get_pos()
    with pytest.raises(AssertionError) if enable_mujoco_compatibility else nullcontext():
        assert_allclose(dof_pos[:3], pos, atol=gs.EPS)


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_extended_broadcasting():
    scene = gs.Scene(
        show_viewer=False,
    )
    for i in range(4):
        scene.add_entity(
            gs.morphs.Box(
                size=(1.0, 1.0, 1.0),
                pos=(0.0, 0.0, i),
            )
        )
    scene.build(n_envs=2)

    envs_idx = torch.tensor([0, 1], dtype=gs.tc_int, device=gs.device)
    for entity in scene.entities:
        entity.zero_all_dofs_velocity(envs_idx)
    assert_allclose(entity.get_dofs_velocity(), 0.0, tol=gs.EPS)
    entity.set_dofs_velocity(1.0)
    assert_allclose(entity.get_dofs_velocity(), 1.0, tol=gs.EPS)
    entity.set_dofs_velocity((1.0, 2.0))
    assert_allclose(entity.get_dofs_velocity(), np.array([(1.0,) * 6, (2.0,) * 6]), tol=gs.EPS)
    entity.set_dofs_velocity((3.0,) * 6)
    assert_allclose(entity.get_dofs_velocity(), 3.0, tol=gs.EPS)
    entity.zero_all_dofs_velocity(torch.tensor([False, True], dtype=torch.bool, device=gs.device))
    assert_allclose(entity.get_dofs_velocity(), np.array([(3.0,) * 6, (0.0,) * 6]), tol=gs.EPS)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_geom_pos_quat(n_envs, show_viewer):
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
    scene.build(n_envs=n_envs)
    batch_shape = (n_envs,) if n_envs > 0 else ()

    box.set_dofs_position(np.random.rand(*batch_shape, 6))
    scene.rigid_solver.update_vgeoms()

    for link in box.links:
        for vgeom, geom in zip(link.vgeoms, link.geoms):
            geom_pos, geom_quat = geom.get_pos(), geom.get_quat()
            assert geom_pos.shape == (*batch_shape, 3)
            assert geom_quat.shape == (*batch_shape, 4)
            vgeom_pos, vgeom_quat = vgeom.get_pos(), vgeom.get_quat()
            assert vgeom_pos.shape == (*batch_shape, 3)
            assert vgeom_quat.shape == (*batch_shape, 4)
            assert_allclose(geom_pos, vgeom_pos, atol=gs.EPS)
            assert_allclose(geom_quat, vgeom_quat, atol=gs.EPS)


@pytest.mark.slow  # ~200s
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
        visualize_contact=True,
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
        visualize_contact=True,
    )
    scene.build()

    for _ in range(80):
        scene.step()

    assert_allclose(box1.get_pos(), (0.0, 0.0, 0.25), atol=5e-4)
    assert_allclose(box2.get_pos(), (0.0, 0.0, 0.75), atol=2e-3)
    assert_allclose(box2.get_pos(), box3.get_pos(), atol=2e-3)
    assert_allclose(scene.rigid_solver.get_links_acc(slice(box4.link_start, box4.link_end)), GRAVITY, atol=tol)


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_mesh_primitive_COM(show_viewer):
    scene = gs.Scene(
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        gs.morphs.Plane(),
    )
    bunny = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/bunny.obj",
            pos=(-1.0, -1.0, 0.55),
        ),
        vis_mode="collision",
        visualize_contact=True,
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.5, 0.5, 0.5),
            pos=(1.0, 1.0, 0.55),
        ),
        vis_mode="collision",
        visualize_contact=True,
    )

    scene.build()
    rigid = scene.sim.rigid_solver
    for _ in range(50):
        scene.step()
    scene.rigid_solver.update_vgeoms()

    _, bunny_COM, cube_COM = rigid.get_links_pos(ref="link_com")
    _, root_bunny_COM, root_cube_COM = rigid.get_links_pos(ref="root_com")
    assert_allclose(bunny_COM, bunny.get_links_pos(links_idx_local=[0], ref="link_com"), atol=gs.EPS)
    assert_allclose(cube_COM, cube.get_links_pos(links_idx_local=[0], ref="link_com"), atol=gs.EPS)
    assert_allclose(root_bunny_COM, bunny_COM, atol=gs.EPS)
    assert_allclose(root_cube_COM, cube_COM, atol=gs.EPS)

    bunny_vgeom = bunny.vgeoms[0]
    bunny_vgeom_COM = tensor_to_array(bunny_vgeom.get_pos()) + bunny_vgeom.vmesh.trimesh.center_mass
    assert_allclose(scene.rigid_solver.get_dofs_velocity(), 0.0, atol=0.05)
    assert_allclose(bunny_COM, bunny_vgeom_COM, tol=5e-3)
    assert_allclose(cube_COM[2], 0.25, atol=2e-3)


@pytest.mark.slow("gpu")  # gpu ~250s
@pytest.mark.debug(False)  # Disable debug for speedup
@pytest.mark.parametrize(
    "backend, mode, friction, n_boxes, solver, scale, mesh_boxes",
    [
        # Two floating boxes (the original noslip scenario): a balanced half-fraction of the backend x friction x
        # scale x geometry matrix - every axis value appears four times and every axis-value pair twice.
        pytest.param(gs.cpu, "noslip", 0.5, 2, gs.constraint_solver.Newton, 0.04, False, marks=pytest.mark.required),
        (gs.cpu, "noslip", 0.5, 2, gs.constraint_solver.Newton, 1.0, True),
        pytest.param(gs.cpu, "noslip", 2.0, 2, gs.constraint_solver.Newton, 0.04, True, marks=pytest.mark.required),
        (gs.cpu, "noslip", 2.0, 2, gs.constraint_solver.Newton, 1.0, False),
        (gs.gpu, "noslip", 0.5, 2, gs.constraint_solver.Newton, 0.04, True),
        pytest.param(gs.gpu, "noslip", 0.5, 2, gs.constraint_solver.Newton, 1.0, False, marks=pytest.mark.required),
        (gs.gpu, "noslip", 2.0, 2, gs.constraint_solver.Newton, 0.04, False),
        pytest.param(gs.gpu, "noslip", 2.0, 2, gs.constraint_solver.Newton, 1.0, True, marks=pytest.mark.required),
        # Constraint-solver coverage: the CG configs document the baseline users can expect from CG. It holds the
        # two-box chain (elliptic at the near-exact Coulomb push here, noslip on CPU below); the three-box chain at
        # the same pushes is beyond its convergence and stays on Newton.
        (gs.gpu, "elliptic", 2.0, 2, gs.constraint_solver.CG, 1.0, False),
        # Three floating boxes: the longer friction chain both mechanisms must hold. At 18 DOF the chain turns
        # islands on and, on GPU past the 16-DOF cooperative threshold, engages the decomposed arm; the islands-off
        # elliptic arms are covered by test_elliptic_cone_coulomb_isotropy. CG rides the lighter-load configs; the
        # stiff high-load cases stay on Newton, which CG cannot hold as tightly. The small-scale mesh configs cover
        # scale sensitivity and mesh contacts.
        pytest.param(gs.cpu, "elliptic", 2.0, 3, gs.constraint_solver.Newton, 1.0, False, marks=pytest.mark.required),
        (gs.cpu, "elliptic", 0.5, 3, gs.constraint_solver.Newton, 0.04, True),
        pytest.param(gs.gpu, "elliptic", 2.0, 3, gs.constraint_solver.Newton, 1.0, False, marks=pytest.mark.required),
        (gs.gpu, "elliptic", 0.5, 3, gs.constraint_solver.Newton, 0.04, True),
        pytest.param(gs.cpu, "noslip", 2.0, 3, gs.constraint_solver.Newton, 0.04, True, marks=pytest.mark.required),
        (gs.cpu, "noslip", 0.5, 3, gs.constraint_solver.CG, 1.0, False),
        pytest.param(gs.gpu, "noslip", 2.0, 3, gs.constraint_solver.Newton, 1.0, False, marks=pytest.mark.required),
    ],
)
def test_static_friction(mode, friction, n_boxes, solver, scale, mesh_boxes, show_viewer, asset_tmp_path):
    # A shear-loaded stack of n_boxes floating boxes braced against a fixed wall must stay static under either
    # creep-suppression mechanism: noslip (pyramidal cone + noslip post-iterations) or the elliptic cone (high
    # tangential impedance). Regularized friction alone lets the stack slowly creep under sustained shear; both hold.
    GRAVITY = -9.81
    # SAFETY_FACTOR scales the applied shear above the theoretical minimum (weight / mu) that braces the stack. The
    # pyramidal cone inscribes the true friction cone and its regularized friction creeps, so noslip must over-push
    # ~2.5x; the elliptic cone enforces the exact Coulomb limit and holds at nearly the theoretical force (the static
    # hold breaks down just below ~1.08, since the fixed wall braces the stack only through the inter-box friction
    # chain). Residual creep shrinks monotonically with the tangential impedance ratio impratio: 20 still creeps past
    # tolerance over this horizon, ~50 holds marginally, and the default 100 holds with margin.
    SAFETY_FACTOR = 1.1 if mode == "elliptic" else 2.5
    # The noslip iteration count is tuned per chain length to match the elliptic cone's static hold: 5 iterations
    # converge the two-box chain at every scale, while the three-box chain at small scale starves at 5 (steady
    # residual creep, solver-independent) and converges from ~15.
    NOSLIP_ITERATIONS = 5 if n_boxes == 2 else 15

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            constraint_solver=solver,
            noslip_iterations=NOSLIP_ITERATIONS if mode == "noslip" else 0,
            friction_cone=gs.friction_cone.elliptic if mode == "elliptic" else gs.friction_cone.pyramidal,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=((0.5 * n_boxes + 4) * scale, (n_boxes + 1.5) * scale, 3 * scale),
            camera_lookat=(0.5 * n_boxes * scale, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )

    for i in range(n_boxes + 1):
        box_size = (scale, scale * (1 + 0.3 * (2 - i)), scale * (1 + 0.3 * (2 - i)))
        if mesh_boxes:
            mesh_path = str(asset_tmp_path / f"static_friction_box_{scale}_{i}.obj")
            trimesh.creation.box(extents=box_size).export(mesh_path, file_type="obj")
            morph = gs.morphs.Mesh(
                file=mesh_path,
                pos=(i * scale, 0, 0),
                fixed=(i == 0),
            )
        else:
            morph = gs.morphs.Box(
                size=box_size,
                pos=(i * (1 - 1e-3) * scale, 0, 0),
                fixed=(i == 0),
            )
        scene.add_entity(
            morph,
            material=gs.materials.Rigid(
                rho=200.0,
                friction=friction,
            ),
            visualize_contact=True,
        )

    floating_boxes = scene.entities[1:]
    scene.build()

    # The solver arms are provably exercised across the parametrization: a single floating box is one island on the
    # dense monolith path, multiple floating boxes turn islands on, and on GPU the cooperative decomposed arm - the
    # path that regressed the elliptic slip - engages once the floating chain reaches the 16-DOF threshold (3 boxes).
    # prefer_decomposed_solver is pinned by the test infra (1 on GPU, 0 on CPU) and the decomposed arm is kept only
    # where the cooperative kernels engage.
    rigid_solver = scene.sim.rigid_solver
    assert rigid_solver._use_contact_island == (n_boxes > 1)
    if gs.backend != gs.cpu:
        assert rigid_solver._static_rigid_sim_config.enable_cooperative_constraint_kernels == (6 * n_boxes >= 16)
        assert rigid_solver._static_rigid_sim_config.prefer_decomposed_solver == (6 * n_boxes >= 16)

    # Force needed to hold the floating boxes static without slipping
    total_mass = sum(box.get_mass() for box in floating_boxes)
    force_x = (total_mass * GRAVITY) / friction

    # Push the furthest floating box toward the fixed wall
    floating_boxes[-1].control_dofs_force(SAFETY_FACTOR * force_x, dofs_idx_local=0)

    # Position-based orientation control stabilizes the contacts
    for box in floating_boxes:
        box.set_dofs_kp(1000.0 * total_mass, dofs_idx_local=slice(3, 6))
        box.set_dofs_kv(100.0 * total_mass, dofs_idx_local=slice(3, 6))
        box.control_dofs_position(box.get_dofs_position(dofs_idx_local=slice(3, 6)), dofs_idx_local=slice(3, 6))

    # Record rest positions after warmup
    for _ in range(50):
        scene.step()
    boxes_pos_init = [box.get_pos() for box in floating_boxes]

    # Hold under sustained shear for 20 seconds
    for _ in range(2000):
        scene.step()

    # The floating boxes stay static
    assert_allclose([box.get_pos() for box in floating_boxes], boxes_pos_init, atol=5e-3)

    # Drop the force below the theoretical threshold; the stack loses its brace and falls
    floating_boxes[-1].control_dofs_force(0.95 * force_x, dofs_idx_local=0)
    for _ in range(300):
        scene.step()
    for box in floating_boxes:
        _, _, box_z = box.get_pos()
        assert box_z < -scale


@pytest.mark.required
@pytest.mark.parametrize(
    "sparse_solve, use_contact_island",
    [
        # Beyond the default arms, the explicit-sparse config pins the elliptic whole-env skyline factor (on CPU,
        # with islands off so the skyline envelope owns the factorization) and the GPU sparse build (which must
        # rebuild with the cone baked in each iteration since the CPU-only incremental cone update is compiled out).
        (None, True),
        (True, False),
    ],
)
def test_elliptic_cone_coulomb_isotropy(sparse_solve, use_contact_island, show_viewer):
    # With the box yaw and the tangential center-of-mass force in independent random directions across parallel envs, a
    # box on a plane must slide above the Coulomb threshold |F_t| = mu*N and hold static below it, identically per env.
    GRAVITY = -9.81
    MU = 1.0
    DT = 0.005
    N_ENVS = 16

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, GRAVITY),
        ),
        rigid_options=gs.options.RigidOptions(
            friction_cone=gs.friction_cone.elliptic,
            sparse_solve=sparse_solve,
            use_contact_island=use_contact_island,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, 1.0, 0.7),
            camera_lookat=(0.0, 0.0, 0.1),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            friction=MU,
        ),
    )
    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(0.0, 0.0, 0.1),
        ),
        material=gs.materials.Rigid(
            friction=MU,
        ),
    )
    scene.build(n_envs=N_ENVS)
    mass = box.get_mass()
    normal_force = MU * mass * (-GRAVITY)

    yaw = 2.0 * torch.pi * torch.rand(N_ENVS, device=gs.device)
    direction = 2.0 * torch.pi * torch.rand(N_ENVS, device=gs.device)
    zeros = torch.zeros(N_ENVS, device=gs.device)
    quat = torch.stack((torch.cos(0.5 * yaw), zeros, zeros, torch.sin(0.5 * yaw)), dim=1)
    force_dir = torch.stack((torch.cos(direction), torch.sin(direction)), dim=1)

    def settle():
        box.control_dofs_force(0.0, dofs_idx_local=[0, 1])
        box.set_pos((0.0, 0.0, 0.1))
        box.set_quat(quat)
        box.set_dofs_velocity(
            torch.cat(
                (0.02 * torch.randn(N_ENVS, 2, device=gs.device), torch.zeros(N_ENVS, 4, device=gs.device)), dim=1
            )
        )
        # Hold each orientation so the CoM force slides the box instead of tipping it about the contact.
        box.set_dofs_kp(1.0e3 * mass, dofs_idx_local=slice(3, 6))
        box.set_dofs_kv(1.0e2 * mass, dofs_idx_local=slice(3, 6))
        box.control_dofs_position(box.get_dofs_position(dofs_idx_local=slice(3, 6)), dofs_idx_local=slice(3, 6))
        for _ in range(25):
            scene.step()

    # Above the Coulomb threshold: the box slides, and the elliptic cone makes the sliding acceleration identical in
    # every direction. Skip the initial transient, then measure the acceleration over a fixed window.
    settle()
    box.control_dofs_force(1.5 * normal_force * force_dir, dofs_idx_local=[0, 1])
    for _ in range(10):
        scene.step()
    vel_0 = box.get_dofs_velocity(dofs_idx_local=[0, 1])
    for _ in range(20):
        scene.step()
    vel_1 = box.get_dofs_velocity(dofs_idx_local=[0, 1])
    accel = torch.linalg.norm(vel_1 - vel_0, dim=1) / (20 * DT)
    # The elliptic spread measures ~1e-5 relative; the pyramidal cone's anisotropy spreads it to ~0.5.
    assert accel.std() < 5e-5 * accel.mean()

    # Below the Coulomb threshold: friction holds the box static in every direction, with no slow tangential creep.
    # The elliptic residual measures ~1e-5; the pyramidal cone's regularized friction creeps at ~1e-3.
    settle()
    box.control_dofs_force(0.4 * normal_force * force_dir, dofs_idx_local=[0, 1])
    for _ in range(40):
        scene.step()
    assert (torch.linalg.norm(box.get_dofs_velocity(dofs_idx_local=[0, 1]), dim=1) < 5e-5).all()


@pytest.mark.required
def test_elliptic_cone_push_isotropy(show_viewer):
    N_ENVS = 8
    FRICTION = 0.5
    BOX_POS = (0.0, 0.0, 0.05)
    # Pusher path in the box's local frame; the shared +y offset gives the push a lever arm that spins the box.
    PUSH_START_LOCAL = (-0.15, 0.03, 0.05)
    PUSH_END_LOCAL = (0.02, 0.03, 0.05)
    POSE_TOL = 2e-4

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
        ),
        rigid_options=gs.options.RigidOptions(
            friction_cone=gs.friction_cone.elliptic,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.7, 0.7, 0.45),
            camera_lookat=(0.0, 0.0, 0.05),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            friction=FRICTION,
        ),
    )
    box = scene.add_entity(
        gs.morphs.Box(
            pos=BOX_POS,
            size=(0.1, 0.2, 0.1),
        ),
        material=gs.materials.Rigid(
            friction=FRICTION,
        ),
    )
    pusher = scene.add_entity(
        gs.morphs.Cylinder(
            pos=PUSH_START_LOCAL,
            height=0.1,
            radius=0.02,
        ),
        material=gs.materials.Rigid(
            friction=FRICTION,
        ),
    )
    scene.build(n_envs=N_ENVS)

    yaw = 2.0 * torch.pi * torch.arange(N_ENVS, device=gs.device) / N_ENVS
    box_quat = gu.xyz_to_quat(torch.stack((torch.zeros_like(yaw), torch.zeros_like(yaw), yaw), dim=1), rpy=True)
    box.set_quat(box_quat)

    # Rotate the local pusher path into each env's world frame by the box yaw, and PD-control the pusher's full pose.
    push_start = gu.transform_by_quat(torch.tensor(PUSH_START_LOCAL, device=gs.device).repeat(N_ENVS, 1), box_quat)
    push_end = gu.transform_by_quat(torch.tensor(PUSH_END_LOCAL, device=gs.device).repeat(N_ENVS, 1), box_quat)
    pusher.set_pos(push_start)
    pusher.set_dofs_kp(
        pusher.get_mass() * torch.tensor((2000.0, 2000.0, 2000.0, 500.0, 500.0, 500.0), device=gs.device)
    )
    pusher.set_dofs_kv(pusher.get_mass() * torch.tensor((200.0, 200.0, 200.0, 50.0, 50.0, 50.0), device=gs.device))

    # Let the box resolve its initial ground contact before the push starts, so the two transients do not couple.
    scene.step()

    # Drive the pusher forward through the box while holding its height and orientation.
    pusher.control_dofs_position(push_end, dofs_idx_local=[0, 1, 2])
    pusher.control_dofs_position(0.0, dofs_idx_local=[3, 4, 5])
    for _ in range(160):
        scene.step()

    # The box and pusher settle at rest by the end.
    assert_allclose(scene.rigid_solver.get_dofs_velocity(), 0.0, atol=0.01)

    # The final box pose in its own initial frame is identical across every initial yaw.
    rel_pos = gu.transform_by_quat(box.get_pos() - torch.tensor(BOX_POS, device=gs.device), gu.inv_quat(box_quat))
    rel_yaw = gu.quat_to_xyz(gu.transform_quat_by_quat(box.get_quat(), gu.inv_quat(box_quat)), rpy=True)[:, 2]
    assert_allclose(rel_pos, rel_pos.mean(dim=0), atol=POSE_TOL)
    assert_allclose(rel_yaw, rel_yaw.mean(), atol=POSE_TOL)


@pytest.mark.slow  # ~250s
@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 3])
def test_axis_aligned_bounding_boxes(n_envs):
    scene = gs.Scene()
    scene.add_entity(
        gs.morphs.Plane(
            normal=(0, 0, 1),
            pos=(0, 0, 0),
        ),
    )
    scene.add_entity(
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

    batch_shape = (n_envs,) if n_envs > 0 else ()
    aabb_shape = (*batch_shape, 2, 3)

    qpos = np.random.rand(*(*batch_shape, robot.n_dofs))
    robot.set_dofs_position(qpos)

    robot_aabb = robot.get_AABB()
    robot_geoms_aabb = torch.stack([geom.get_AABB().expand(aabb_shape) for geom in robot.geoms], dim=0)
    assert_allclose(torch.min(robot_geoms_aabb[..., 0, :], dim=0).values, robot_aabb[..., 0, :], tol=gs.EPS)
    assert_allclose(torch.max(robot_geoms_aabb[..., 1, :], dim=0).values, robot_aabb[..., 1, :], tol=gs.EPS)
    for link in robot.links:
        link_aabb = link.get_AABB()
        link_geoms_aabb = torch.stack([geom.get_AABB().expand(aabb_shape) for geom in link.geoms], dim=0)
        assert_allclose(torch.min(link_geoms_aabb[..., 0, :], dim=0).values, link_aabb[..., 0, :], tol=gs.EPS)
        assert_allclose(torch.max(link_geoms_aabb[..., 1, :], dim=0).values, link_aabb[..., 1, :], tol=gs.EPS)

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

    vaabbs = [vgeom.get_vAABB().expand(aabb_shape) for entity in scene.entities for vgeom in entity.vgeoms]
    if n_envs > 0:
        for entity in scene.entities:
            for vgeom in entity.vgeoms:
                assert_allclose(vgeom.get_vAABB(), [vgeom.get_vAABB(i)[0] for i in range(n_envs)], tol=gs.EPS)
    box_aabb_min, box_aabb_max = vaabbs[1].split(1, dim=-2)
    assert_allclose(box_aabb_min, (0.45, -0.05, 0.0), atol=gs.EPS)
    assert_allclose(box_aabb_max, (0.55, 0.05, 0.1), atol=gs.EPS)
    sphere_aabb_min, sphere_aabb_max = vaabbs[3].split(1, dim=-2)
    assert_allclose(sphere_aabb_min, (-0.55, -0.05, 0.0), atol=1e-3)
    assert_allclose(sphere_aabb_max, (-0.45, 0.05, 0.1), atol=1e-3)

    robot_vaabb = robot.get_vAABB()
    assert_allclose(robot_vaabb, robot_aabb, atol=1e-3)


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize("model_name", ["ellipsoid"])
def test_ellipsoid(xml_path, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.02,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.4, 0.4, 0.3),
            camera_lookat=(0.0, 0.0, 0.1),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    entity = scene.add_entity(
        gs.morphs.MJCF(
            file=xml_path,
            pos=(0, 0, 0.2),
        ),
        vis_mode="collision",
        visualize_contact=True,
    )
    scene.build()

    entity.set_dofs_velocity(20 * np.random.rand(3), dofs_idx_local=slice(3, 6))
    entity.set_dofs_kv(0.002, dofs_idx_local=slice(3, 6))
    entity.control_dofs_velocity(0.0, dofs_idx_local=slice(3, 6))

    # AABB must match the ellipsoid semi-axes
    aabb = entity.get_AABB()
    aabb_extent = aabb[1] - aabb[0]
    assert_allclose(aabb_extent, (0.10, 0.10, 0.04), atol=1e-3)

    # Free-fall onto plane: ellipsoid must come to rest
    for _ in range(100):
        scene.step()

    assert_allclose(entity.get_dofs_velocity(), 0, tol=5e-3)
    assert (-0.005 < entity.get_AABB()[0, 2] < 0.0).all()
    roll, pitch, _yaw = gu.quat_to_xyz(entity.get_quat(), rpy=True)
    assert_allclose((roll, pitch), (0.0, 0.0), tol=5e-3)


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_align_mesh(show_viewer, tol):
    INIT_POS = (0.0, 0.0, 0.1)

    mango_path = get_hf_dataset(pattern="glb/mango.glb")
    bowl_path = get_hf_dataset(pattern="glb/orange_plastic_bowl.glb")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.8, 0.8, 1.6),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    mango_morph = gs.morphs.Mesh(
        file=f"{mango_path}/glb/mango.glb",
        scale=0.045,
        pos=INIT_POS,
        align=True,
    )
    mango = scene.add_entity(
        mango_morph,
        material=gs.materials.Rigid(
            rho=1000.0,
        ),
        vis_mode="collision",
        visualize_contact=True,
    )
    ghost_mango = scene.add_entity(
        mango_morph,
        material=gs.materials.Kinematic(),
    )
    # Heterogeneous entity sharing one link across a bowl and a mango variant. Each variant must be aligned to its own
    # geometry (link origin at that variant's COM) and report its own offset per environment. The bowl carries a
    # hard-coded offset to check it composes with the alignment, and the mango variant must end up aligned exactly
    # like the standalone mango above.
    HET_POS = (0.5, 0.0, 0.1)
    het_obj = scene.add_entity(
        morph=(
            gs.morphs.Mesh(
                file=f"{bowl_path}/glb/orange_plastic_bowl.glb",
                scale=0.5,
                pos=HET_POS,
                offset_euler=(30.0, 0.0, 0.0),
                align=True,
            ),
            gs.morphs.Mesh(
                file=f"{mango_path}/glb/mango.glb",
                scale=0.045,
                pos=HET_POS,
                align=True,
            ),
        ),
        material=gs.materials.Rigid(rho=1000.0),
    )
    scene.build(n_envs=2)

    # Geoms/vgeoms are alignment-transparent, so their world pose equals the morph pose, and without a morph offset
    # the relative (user-frame) pose matches it too.
    geom, vgeom = mango.geoms[0], mango.vgeoms[0]
    for relative in (False, True):
        assert_allclose(geom.get_pos(relative=relative), INIT_POS, atol=1e-3)
        assert_allclose(geom.get_quat(relative=relative), gu.identity_quat(), atol=1e-3)
        assert_allclose(vgeom.get_pos(relative=relative), INIT_POS, atol=1e-3)
        assert_allclose(vgeom.get_quat(relative=relative), gu.identity_quat(), atol=1e-3)

    # The relative (user-frame) base pose strips the alignment back to the morph pose.
    assert_allclose(mango.get_pos(relative=True), INIT_POS, tol=tol)
    assert_allclose(mango.get_quat(relative=True), gu.identity_quat(), tol=tol)
    # The world-frame base pose places the link frame at the geometry COM and principal axes.
    assert_allclose(
        mango.get_links_pos(links_idx_local=[0], ref="link_com", relative=False),
        mango.get_links_pos(links_idx_local=[0], ref="link_origin", relative=False),
        tol=tol,
    )
    geom_inertia_i = qd_to_numpy(scene.rigid_solver.links_state.cinr_inertial, transpose=True)[0, 1]
    geom_quat = tensor_to_array(mango.get_quat(relative=False))
    assert_allclose(gu.R_to_xyz(gu.quat_to_R(geom_quat) @ uu.principal_axes_rot(geom_inertia_i).T), 0.0, tol=tol)

    # Both variants (env 0 bowl, env 1 mango) strip their own offset back to the user pose, and each variant's link
    # origin sits at its own COM. The bowl recovers the user frame despite its hard-coded offset, proving the offset
    # composes with the alignment.
    for i_env in (0, 1):
        assert_allclose(het_obj.get_pos(relative=True, envs_idx=i_env), HET_POS, tol=tol)
        assert_allclose(het_obj.get_quat(relative=True, envs_idx=i_env), gu.identity_quat(), tol=tol)
        assert_allclose(
            het_obj.get_links_pos(links_idx_local=[0], ref="link_com", envs_idx=i_env, relative=False),
            het_obj.get_links_pos(links_idx_local=[0], ref="link_origin", envs_idx=i_env, relative=False),
            tol=tol,
        )

    # The two variants have different geometry, so their aligned world origins differ.
    with np.testing.assert_raises(AssertionError):
        assert_allclose(
            het_obj.get_pos(relative=False, envs_idx=0), het_obj.get_pos(relative=False, envs_idx=1), tol=tol
        )

    # The heterogeneous mango variant (env 1) is aligned exactly like the standalone mango: the world<-user offset
    # (COM shift and principal-axis rotation) matches, independent of the base placement.
    het_mango_offset = het_obj.get_pos(relative=False, envs_idx=1) - het_obj.get_pos(relative=True, envs_idx=1)
    mango_offset = mango.get_pos(relative=False, envs_idx=0) - mango.get_pos(relative=True, envs_idx=0)
    assert_allclose(het_mango_offset, mango_offset, tol=tol)
    assert_allclose(het_obj.get_quat(relative=False, envs_idx=1), mango.get_quat(relative=False, envs_idx=0), tol=tol)

    # Same qpos on rigid and kinematic entities must yield matching vAABB
    qpos = (0.3, -0.2, 1.0, 0.6, 0.5, 0.3, 0.0)
    mango.set_qpos(qpos)
    ghost_mango.set_qpos(qpos)
    assert_allclose(mango.get_vAABB(), ghost_mango.get_vAABB(), tol=gs.EPS)
    scene.reset()

    # Simulate
    for _ in range(600):
        scene.step()

    assert_allclose(mango.get_dofs_velocity(dofs_idx_local=(0, 1, 2)), 0, tol=0.01)
    assert_allclose(mango.get_dofs_velocity(dofs_idx_local=(3, 4, 5)), 0, tol=0.05)
    assert_allclose(mango.get_dofs_velocity(), 0, tol=0.05)
    min_z = mango.get_AABB()[:, 0, 2]
    assert ((-0.005 < min_z) & (min_z < 0.0)).all()


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_align_urdf(show_viewer, tol):
    INIT_POS = (0.0, 0.0, 0.7)

    asset_path = get_hf_dataset(pattern="fork/*")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.8, 0.8, 0.6),
            camera_lookat=(-0.3, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    fork_morph = gs.morphs.URDF(
        file=f"{asset_path}/fork/fork.urdf",
        pos=INIT_POS,
    )
    fork = scene.add_entity(
        fork_morph,
        vis_mode="collision",
        visualize_contact=True,
    )
    ghost_fork = scene.add_entity(
        fork_morph,
        material=gs.materials.Kinematic(),
    )
    scene.build()

    # The relative (user-frame) base pose strips the alignment back to the morph pose, while the world-frame base
    # pose has its link frame origin at the collision geometry COM (auto-align for basic rigid objects).
    assert_allclose(fork.get_pos(relative=True), INIT_POS, tol=tol)
    assert_allclose(fork.get_links_pos(ref="link_com"), fork.get_pos(relative=False), tol=tol)

    # Same qpos on rigid and kinematic entities must yield matching vAABB
    qpos = (0.3, -0.2, 1.0, 0.6, 0.5, 0.3, 0.0)
    fork.set_qpos(qpos)
    ghost_fork.set_qpos(qpos)
    assert_allclose(fork.get_vAABB(), ghost_fork.get_vAABB(), tol=gs.EPS)
    scene.reset()

    # Simulate with initial angular velocity to check numerical stability
    fork.set_dofs_velocity(10.0, dofs_idx_local=slice(3, 6))
    for _ in range(200):
        scene.step()

    assert_allclose(fork.get_dofs_velocity(), 0, tol=0.05)
    assert (-0.002 < fork.get_AABB()[0, 2] < 0.0).all()


@pytest.mark.required
def test_align_mixed_mass_raises():
    # Mixing a user-specified mass with a geometry-estimated one in an aligned free body makes the anchor density-
    # dependent (so rigid and kinematic could align differently) and must raise. The fixed joint with
    # merge_fixed_links=False keeps the child a distinct fixed link with unspecified mass while the base specifies one.
    urdf = _build_two_link_revolute_urdf(
        "mixed_mass_align",
        "box",
        {"size": "0.06 0.06 0.06"},
        links_inertial=[{"mass": 1.0, "ixx": 0.01, "iyy": 0.01, "izz": 0.01, "origin_xyz": "0 0 0"}, None],
        joint_type="fixed",
    )
    for material in (gs.materials.Rigid(), gs.materials.Kinematic()):
        scene = gs.Scene(
            show_viewer=False,
            show_FPS=False,
        )
        scene.add_entity(
            gs.morphs.URDF(
                file=urdf,
                align=True,
                merge_fixed_links=False,
            ),
            material=material,
        )
        with pytest.raises(gs.GenesisException, match="mixes user-specified and geometry-estimated"):
            scene.build()


@pytest.mark.required
def test_align_relative_offset_on_link_relative_geoms(show_viewer, tol):
    # To exercise the geom-frame offset strip the geoms MUST sit at non-identity poses relative to their link (explicit
    # collision/visual <origin>) AND the morph offset MUST be a rotation that does not commute with them - otherwise the
    # conjugation degenerates to the plain morph offset and a naive (corrupted) strip would still pass. A convex-
    # decomposed mesh is useless here: its sub-geoms keep an identity frame (geometry lives in the vertices).
    robot = ET.Element("robot", name="posed_geoms")
    link = ET.SubElement(robot, "link", name="body")
    for origin_rpy, origin_xyz in (
        ("0 0 1.5708", "0.1 0 0"),
        ("0.7854 0 0", "0 0.1 0.05"),
        ("0 1.0472 0.5", "0 0 0.1"),
    ):
        for group_tag in ("collision", "visual"):
            group = ET.SubElement(link, group_tag)
            geom_el = ET.SubElement(group, "geometry")
            ET.SubElement(geom_el, "box", size="0.05 0.1 0.15")
            ET.SubElement(group, "origin", rpy=origin_rpy, xyz=origin_xyz)
    urdf = urdfpy.URDF._from_xml(robot, robot, get_assets_dir())

    BODY_POS = (0.0, 0.0, 0.2)
    OFFSET_EULER = (20.0, 35.0, 50.0)  # a generic rotation that does not commute with the geom poses
    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )
    body = scene.add_entity(
        gs.morphs.URDF(
            file=urdf,
            pos=BODY_POS,
            offset_euler=OFFSET_EULER,
            align=True,
        ),
        material=gs.materials.Rigid(),
    )
    scene.build()

    assert len(body.geoms) > 1, "expected multiple geoms posed relative to the link"

    # The user orientation is identity, so the world<-user offset rotates each geom about the link origin:
    # geom_world_pos = U_pos + R(offset) * (geom_user_pos - U_pos) and geom_world_quat = offset * geom_user_quat.
    assert_allclose(body.get_quat(relative=True), gu.identity_quat(), tol=tol)
    u_pos = tensor_to_array(body.get_pos(relative=True))
    offset_quat = gu.xyz_to_quat(np.array(OFFSET_EULER), rpy=True, degrees=True)
    for geom in body.geoms:
        geom_user_pos = tensor_to_array(geom.get_pos(relative=True))
        geom_user_quat = tensor_to_array(geom.get_quat(relative=True))
        expected_world_pos = u_pos + gu.transform_by_quat(geom_user_pos - u_pos, offset_quat)
        expected_world_quat = gu.transform_quat_by_quat(geom_user_quat, offset_quat)
        assert_allclose(geom.get_pos(relative=False), expected_world_pos, tol=tol)
        assert_allclose(geom.get_quat(relative=False), expected_world_quat, tol=tol)


def create_two_free_bodies_mjcf(name, pos_a, geom_a, pos_b, geom_b):
    """Helper to create an MJCF with two free root bodies, each a single box geom offset from its body origin."""
    mjcf = ET.Element("mujoco", model=name)
    worldbody = ET.SubElement(mjcf, "worldbody")
    body_a = ET.SubElement(worldbody, "body", name="a", pos=f"{pos_a[0]} {pos_a[1]} {pos_a[2]}")
    ET.SubElement(body_a, "joint", name="a_free", type="free")
    ET.SubElement(
        body_a, "geom", type="box", size="0.05 0.05 0.05", pos=f"{geom_a[0]} {geom_a[1]} {geom_a[2]}", density="1000"
    )
    body_b = ET.SubElement(worldbody, "body", name="b", pos=f"{pos_b[0]} {pos_b[1]} {pos_b[2]}")
    ET.SubElement(body_b, "joint", name="b_free", type="free")
    ET.SubElement(
        body_b, "geom", type="box", size="0.05 0.08 0.03", pos=f"{geom_b[0]} {geom_b[1]} {geom_b[2]}", density="1000"
    )
    return mjcf


@pytest.mark.required
def test_multi_root_offset(show_viewer, tol):
    # To exercise per-root offset tracking the entity MUST hold more than one free root (one MJCF, several free
    # bodies) with DISTINCT per-root geometry, so each root gets its own alignment offset; a single root - or
    # identical roots - would not surface the cross-contamination bug (one root's offset leaking into another).
    BODY_A_POS = (1.0, 0.0, 0.5)
    BODY_B_POS = (-1.0, 0.0, 0.5)
    GEOM_A_POS = (0.02, 0.01, 0.0)
    GEOM_B_POS = (0.0, 0.03, 0.02)

    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )
    entity = scene.add_entity(
        gs.morphs.MJCF(
            file=ET.tostring(
                create_two_free_bodies_mjcf("two_bodies", BODY_A_POS, GEOM_A_POS, BODY_B_POS, GEOM_B_POS),
                encoding="unicode",
            ),
            align=True,
        ),
    )
    scene.build()

    link_a, link_b = entity.links
    # Each root reports its own user-specified pose in the relative frame, independent of the other root.
    assert_allclose(link_a.get_pos(), BODY_A_POS, tol=tol)
    assert_allclose(link_b.get_pos(), BODY_B_POS, tol=tol)
    assert_allclose(link_a.get_quat(), gu.identity_quat(), tol=tol)
    assert_allclose(link_b.get_quat(), gu.identity_quat(), tol=tol)

    # The world frame carries each root's own COM shift (the box center), confirming the offsets are not shared.
    assert_allclose(link_a.get_pos(relative=False), np.add(BODY_A_POS, GEOM_A_POS), tol=tol)
    assert_allclose(link_b.get_pos(relative=False), np.add(BODY_B_POS, GEOM_B_POS), tol=tol)

    # Both roots free-fall under gravity: the relative getter tracks each user frame, holding x/y and dropping z
    # equally (free fall is mass-independent).
    for _ in range(20):
        scene.step()
    assert_allclose(link_a.get_pos()[..., :2], BODY_A_POS[:2], tol=tol)
    assert_allclose(link_b.get_pos()[..., :2], BODY_B_POS[:2], tol=tol)
    assert_allclose(link_a.get_pos()[..., 2], link_b.get_pos()[..., 2], tol=tol)


@pytest.fixture
def xacro_robot(tmp_path):
    """Generate a XACRO file with a two-link chain using macros, properties, overridable args, and a mesh geometry."""
    XACRO_NS = "http://www.ros.org/wiki/xacro"
    ET.register_namespace("xacro", XACRO_NS)

    # Symlink a mesh file into the tmp directory so the xacro can reference it with a relative path
    mesh_src = os.path.join(get_assets_dir(), "meshes", "sphere.obj")
    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()
    (mesh_dir / "sphere.obj").symlink_to(mesh_src)

    robot = ET.Element("robot", name="xacro_chain")

    # Overridable args with defaults
    ET.SubElement(robot, f"{{{XACRO_NS}}}arg", name="link_mass", default="1.0")
    ET.SubElement(robot, f"{{{XACRO_NS}}}arg", name="link_length", default="0.4")

    # Properties derived from args
    ET.SubElement(robot, f"{{{XACRO_NS}}}property", name="mass", value="$(arg link_mass)")
    ET.SubElement(robot, f"{{{XACRO_NS}}}property", name="length", value="$(arg link_length)")
    ET.SubElement(robot, f"{{{XACRO_NS}}}property", name="radius", value="0.05")

    # Macro for a cylindrical link with inertial
    macro = ET.SubElement(robot, f"{{{XACRO_NS}}}macro", name="cyl_link", params="name")
    link = ET.SubElement(macro, "link", name="${name}")
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "mass", value="${mass}")
    ET.SubElement(inertial, "inertia", ixx="0.01", ixy="0", ixz="0", iyy="0.01", iyz="0", izz="0.001")
    visual = ET.SubElement(link, "visual")
    ET.SubElement(ET.SubElement(visual, "geometry"), "cylinder", radius="${radius}", length="${length}")
    collision = ET.SubElement(link, "collision")
    ET.SubElement(ET.SubElement(collision, "geometry"), "cylinder", radius="${radius}", length="${length}")

    # Macro for a mesh link (uses relative path)
    mesh_macro = ET.SubElement(robot, f"{{{XACRO_NS}}}macro", name="mesh_link", params="name")
    mesh_link = ET.SubElement(mesh_macro, "link", name="${name}")
    mesh_inertial = ET.SubElement(mesh_link, "inertial")
    ET.SubElement(mesh_inertial, "mass", value="${mass}")
    ET.SubElement(mesh_inertial, "inertia", ixx="0.01", ixy="0", ixz="0", iyy="0.01", iyz="0", izz="0.001")
    for tag in ("visual", "collision"):
        group = ET.SubElement(mesh_link, tag)
        ET.SubElement(ET.SubElement(group, "geometry"), "mesh", filename="meshes/sphere.obj", scale="0.05 0.05 0.05")

    # Instantiate: two cylinder links + one mesh link
    ET.SubElement(robot, f"{{{XACRO_NS}}}cyl_link", name="base_link")
    ET.SubElement(robot, f"{{{XACRO_NS}}}cyl_link", name="child_link")
    ET.SubElement(robot, f"{{{XACRO_NS}}}mesh_link", name="mesh_link")

    # Revolute joint: base_link -> child_link
    joint = ET.SubElement(robot, "joint", name="joint_0", type="revolute")
    ET.SubElement(joint, "parent", link="base_link")
    ET.SubElement(joint, "child", link="child_link")
    ET.SubElement(joint, "origin", xyz="0 0 ${length}")
    ET.SubElement(joint, "axis", xyz="0 1 0")
    ET.SubElement(joint, "limit", lower="-1.57", upper="1.57", effort="100", velocity="1")

    # Fixed joint: child_link -> mesh_link
    joint2 = ET.SubElement(robot, "joint", name="joint_1", type="fixed")
    ET.SubElement(joint2, "parent", link="child_link")
    ET.SubElement(joint2, "child", link="mesh_link")
    ET.SubElement(joint2, "origin", xyz="0 0 ${length}")

    file_path = str(tmp_path / "two_link.urdf.xacro")
    ET.ElementTree(robot).write(file_path, encoding="utf-8", xml_declaration=True)
    return file_path


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_xacro_loading(xacro_robot, show_viewer, tol):
    """Test that .urdf.xacro files are preprocessed and loaded with correct structure and properties."""
    scene = gs.Scene(show_viewer=show_viewer)

    # Load with default args (mass=1.0, length=0.4)
    morph = gs.morphs.URDF(
        file=xacro_robot,
        fixed=True,
        merge_fixed_links=False,
    )

    # After xacro processing, morph.file is a urdfpy.URDF with absolute mesh paths
    assert isinstance(morph.file, urdfpy.URDF)
    for link in morph.file.links:
        for geom_prop in (*link.collisions, *link.visuals):
            if isinstance(geom_prop.geometry.geometry, urdfpy.Mesh):
                assert os.path.isabs(geom_prop.geometry.geometry.filename)

    entity = scene.add_entity(morph)

    # Load again with overridden mass via xacro_args
    heavy = scene.add_entity(
        gs.morphs.URDF(
            file=xacro_robot,
            fixed=True,
            merge_fixed_links=False,
            xacro_args={"link_mass": "5.0"},
        ),
    )
    scene.build()

    # Entity name from <robot name="xacro_chain">
    assert entity.name.startswith("xacro_chain_")

    # Three links (base_link + child_link + mesh_link), one revolute DOF
    assert entity.n_links == 3
    assert [l.name for l in entity.links] == ["base_link", "child_link", "mesh_link"]
    assert entity.n_dofs == 1
    assert entity.links[1].joints[0].type == gs.JOINT_TYPE.REVOLUTE

    # Geom types: cylinder on first two links, mesh on third
    assert entity.links[0].geoms[0].type == gs.GEOM_TYPE.CYLINDER
    assert entity.links[1].geoms[0].type == gs.GEOM_TYPE.CYLINDER
    assert entity.links[2].geoms[0].type == gs.GEOM_TYPE.MESH

    # Mass check: 3 links at 1.0 each (default) vs 5.0 each (overridden)
    assert_allclose(entity.get_mass(), 3.0, tol=tol)
    assert_allclose(heavy.get_mass(), 15.0, tol=tol)


@pytest.mark.slow  # ~250s
@pytest.mark.required
@pytest.mark.parametrize("batch_links_info", [False, True])
@pytest.mark.parametrize("batch_joints_info", [False, True])
@pytest.mark.parametrize("batch_dofs_info", [False, True])
def test_batched_info(batch_links_info, batch_joints_info, batch_dofs_info):
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
    act_gain = dofs_info.act_gain.to_numpy()
    assert act_gain.shape == (9, 2) if batch_dofs_info else (9,)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("robot_path", ["xml/franka_emika_panda/panda.xml"])
def test_reset_control(robot_path, tol):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
        ),
        show_viewer=False,
        show_FPS=False,
    )
    robot = scene.add_entity(gs.morphs.MJCF(file=robot_path))
    scene.build()
    qpos = np.random.rand(robot.n_dofs)
    robot.set_dofs_position(qpos)
    robot.control_dofs_position(torch.zeros((robot.n_dofs,), dtype=gs.tc_float, device=gs.device))
    old_control_force = robot.get_dofs_control_force()
    scene.reset()
    new_control_force = robot.get_dofs_control_force()
    assert old_control_force.abs().max() > gs.EPS
    assert_allclose(new_control_force, 0, tol=gs.EPS)


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_joint_get_anchor_pos_and_axis(n_envs):
    scene = gs.Scene(
        show_viewer=False,
        show_FPS=False,
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
    )
    scene.build(n_envs=n_envs)
    batch_shape = (n_envs,) if n_envs > 0 else ()

    joint = robot.joints[1]
    anchor_pos = joint.get_anchor_pos()
    assert anchor_pos.shape == (*batch_shape, 3)
    expected_pos = scene.rigid_solver.joints_state.xanchor.to_numpy()
    assert_allclose(anchor_pos, expected_pos[joint.idx], tol=gs.EPS)

    anchor_axis = joint.get_anchor_axis()
    assert anchor_axis.shape == (*batch_shape, 3)
    expected_axis = scene.rigid_solver.joints_state.xaxis.to_numpy()
    assert_allclose(anchor_axis, expected_axis[joint.idx], tol=gs.EPS)


@pytest.mark.required
@pytest.mark.parametrize("is_fixed", [False, True])
@pytest.mark.parametrize("merge_fixed_links", [False, True])
def test_merge_entities(is_fixed, merge_fixed_links, show_viewer, tol, monkeypatch):
    # Force parallelism on CPU to trigger any cross-entity race condition
    if gs.backend == gs.cpu:
        monkeypatch.setenv("GS_PARA_LEVEL", "2")
        monkeypatch.setenv("QD_NUM_THREADS", "3")

    EULER_OFFSET = (0, 0, 45)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=True,
            enable_neutral_collision=True,
            enable_adjacent_collision=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -3.5, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(gs.morphs.Plane())

    franka = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/panda_bullet/panda_nohand.urdf",
            merge_fixed_links=False,
            fixed=True,
        ),
        vis_mode="collision",
    )
    hand = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/panda_bullet/hand.urdf",
            euler=EULER_OFFSET,
            fixed=is_fixed,
            merge_fixed_links=merge_fixed_links,
            batch_fixed_verts=is_fixed,
        ),
        vis_mode="collision",
    )
    tool = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.005,
        ),
    )
    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.02, 0.02, 0.02),
            pos=(0.3, 0.0, 0.01),
        ),
    )
    with pytest.raises(gs.GenesisException):
        franka.attach(hand, "right_finger")
    hand.attach(franka, "attachment")
    tool.attach(hand, "right_finger")
    scene.build()
    with pytest.raises(gs.GenesisException):
        box.attach(hand, "right_finger")

    # Make sure that collision between hand base link and franka attachment point has been filtered out as adjacent
    collision_pair_idx = scene.rigid_solver.collider._collider_info.collision_pair_idx.to_numpy()
    assert collision_pair_idx[franka.get_link("attachment").idx, hand.base_link_idx] == -1

    with pytest.raises(gs.GenesisException):
        hand.set_pos(0.0)
    with pytest.raises(gs.GenesisException):
        hand.set_quat(0.0)

    # The free box is dynamically isolated from the robot, so its lateral position must stay put while the
    # gripper actuates. Attaching the floating-base hand re-indexes joints by dropping its free base joint;
    # the hand's mimic (joint-equality) references must follow that re-indexing, otherwise they alias this
    # box's free-joint DOFs and the corrupted constraint drags the box sideways as the fingers move.
    box_pos_init = box.get_pos()

    franka.control_dofs_position([-1, 0.8, 1, -2, 1, 0.5, -0.5])
    hand.control_dofs_position([0.04, 0.04])
    for _ in range(30):
        scene.step()

    assert_allclose(box.get_pos()[..., :2], box_pos_init[..., :2], tol=1e-3)

    attach_link = franka.get_link("attachment")
    assert_allclose(attach_link.get_pos(), hand.links[0].get_pos(), tol=gs.EPS)
    offset_quat = gu.transform_quat_by_quat(hand.links[0].get_quat(), gu.inv_quat(attach_link.get_quat()))
    assert_allclose(gu.quat_to_xyz(offset_quat, rpy=False, degrees=True), EULER_OFFSET, tol=tol)
    for link in hand.links[slice(0, None) if merge_fixed_links else slice(1, -1)]:
        assert torch.linalg.norm(link.get_pos() - attach_link.get_pos(), dim=-1) < 0.08
    if not merge_fixed_links:
        assert_allclose(torch.linalg.norm(hand.links[-1].get_pos() - attach_link.get_pos(), dim=-1), 0.105, tol=tol)

    assert_allclose(tool.get_pos(), hand.get_link("right_finger").get_pos(), tol=gs.EPS)


@pytest.mark.required
def test_heterogeneous_physics_parity(show_viewer, tol):
    # Uses the fixed-child mesh objects from 'test_convexify' (offset center of mass, distinct mass) so the per-env
    # parity check exercises the inertia alignment, not just trivially-symmetric primitives.
    N_STEPS = 100
    DROP_HEIGHT = 0.2
    VARIANTS = (("mug_1", "output.xml"), ("donut_0", "output.xml"), ("cup_2", "model.xml"), ("apple_15", "model.xml"))
    # Divergent per-variant yaw offset, stripped to identity by the relative getter and carried by the world frame.
    # Applied identically to the homogeneous reference so the dynamics still match.
    OFFSET_EULERS = ((0.0, 0.0, 30.0), (0.0, 0.0, -45.0), (0.0, 0.0, 90.0), (0.0, 0.0, -120.0))
    # Distinct per-variant placement, dispatched per environment.
    POSITIONS = ((0.0, 0.0, DROP_HEIGHT), (0.2, 0.0, DROP_HEIGHT), (0.0, 0.2, DROP_HEIGHT), (0.2, 0.2, DROP_HEIGHT))
    # The homogeneous references live in the same scene, offset far enough that no entity ever interacts with
    # another: a single build compiles one kernel set instead of one per scene.
    REFERENCE_OFFSETS = ((10.0, 0.0, 0.0), (20.0, 0.0, 0.0), (30.0, 0.0, 0.0), (40.0, 0.0, 0.0))

    asset_files = tuple(f"{get_hf_dataset(pattern=f'{name}/*')}/{name}/{xml}" for name, xml in VARIANTS)

    # One homogeneous reference entity per variant plus a single heterogeneous entity dispatching one variant per
    # environment, all in one scene.
    scene = gs.Scene(show_viewer=show_viewer)
    scene.add_entity(gs.morphs.Plane())
    ref_objs = []
    for file, pos, offset_euler, offset in zip(asset_files, POSITIONS, OFFSET_EULERS, REFERENCE_OFFSETS):
        ref_objs.append(
            scene.add_entity(
                gs.morphs.MJCF(
                    file=file,
                    pos=(pos[0] + offset[0], pos[1] + offset[1], pos[2] + offset[2]),
                    offset_euler=offset_euler,
                ),
            )
        )
    het_obj = scene.add_entity(
        morph=tuple(
            gs.morphs.MJCF(
                file=file,
                pos=pos,
                offset_euler=offset_euler,
            )
            for file, pos, offset_euler in zip(asset_files, POSITIONS, OFFSET_EULERS)
        )
    )
    scene.build(n_envs=len(VARIANTS))

    # At init each variant sits at its own placement; the relative getter strips its offset (and inertial alignment) to
    # identity in the user frame, while the world frame matches the homogeneous reference's world orientation.
    assert_allclose(gu.quat_to_xyz(het_obj.get_quat(relative=True)), 0.0, tol=tol)
    assert_allclose(het_obj.get_pos(), POSITIONS, tol=tol)
    # Matching the reference in both frames validates that the inertial alignment is applied identically to the
    # heterogeneous entity and the homogeneous references.
    for relative in (True, False):
        ref_quats = torch.cat(
            [ref_obj.get_quat(envs_idx=[i_env], relative=relative) for i_env, ref_obj in enumerate(ref_objs)]
        )
        assert_allclose(het_obj.get_quat(relative=relative), ref_quats, tol=tol)

    for _ in range(N_STEPS):
        scene.step()

    # After the drop each environment matches the homogeneous reference of its variant in pose, velocity and mass.
    ref_pos = torch.cat([ref_obj.get_pos(envs_idx=[i_env]) for i_env, ref_obj in enumerate(ref_objs)])
    ref_vel = torch.cat([ref_obj.get_vel(envs_idx=[i_env]) for i_env, ref_obj in enumerate(ref_objs)])
    assert_allclose(ref_pos - het_obj.get_pos(), REFERENCE_OFFSETS, tol=tol)
    assert_allclose(het_obj.get_vel(), ref_vel, tol=tol)
    assert_allclose(het_obj.get_mass(), [ref_obj.get_mass() for ref_obj in ref_objs], tol=tol)

    # The variants are genuinely distinct: their masses are not all equal.
    with pytest.raises(AssertionError):
        assert_allclose(het_obj.get_mass(), het_obj.get_mass()[0], tol=tol)


@pytest.mark.required
def test_heterogeneous_invalid_material_raises():
    """Test that heterogeneous morphs with unsupported material raises an exception."""
    scene = gs.Scene(
        show_viewer=False,
    )

    morphs_heterogeneous = (
        gs.morphs.Box(size=(1.0, 1.0, 1.0)),
        gs.morphs.Box(size=(1.0, 1.0, 1.0)),
    )

    # PBD material should raise an exception
    with pytest.raises(gs.GenesisException):
        scene.add_entity(
            morph=morphs_heterogeneous,
            material=gs.materials.PBD.Cloth(),
        )


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_heterogeneous_morph_property_raises():
    scene = gs.Scene(show_viewer=False)

    single_morph = gs.morphs.Box(size=(0.1, 0.1, 0.1))
    single_obj = scene.add_entity(morph=single_morph)

    rigid_morphs_heterogeneous = (
        gs.morphs.Box(size=(0.1, 0.1, 0.1)),
        gs.morphs.Cylinder(radius=0.05, height=0.2),
    )
    rigid_obj = scene.add_entity(morph=rigid_morphs_heterogeneous)
    kinematic_morphs_heterogeneous = (
        gs.morphs.Box(size=(0.2, 0.2, 0.2)),
        gs.morphs.Sphere(radius=0.1),
    )
    kinematic_obj = scene.add_entity(
        morph=kinematic_morphs_heterogeneous,
        material=gs.materials.Kinematic(),
    )

    assert single_obj.morph is single_morph
    assert rigid_obj.main_morph is rigid_morphs_heterogeneous[0]
    assert list(rigid_obj.morphs) == list(rigid_morphs_heterogeneous)
    with pytest.raises(gs.GenesisException, match=r"Heterogeneous.*\.morphs") as exc_info:
        _ = rigid_obj.morph
    assert ".main_morph" in str(exc_info.value)

    assert kinematic_obj.main_morph is kinematic_morphs_heterogeneous[0]
    assert list(kinematic_obj.morphs) == list(kinematic_morphs_heterogeneous)
    with pytest.raises(gs.GenesisException, match=r"Heterogeneous.*\.morphs"):
        _ = kinematic_obj.morph


@pytest.mark.required
def test_heterogeneous_fewer_envs_than_variants():
    """Test that having fewer environments than variants works correctly.

    Variant Assignment Rule (when n_envs < n_het):
        Environment i gets variant i (0-indexed). Variants beyond n_envs are unused.
        For example, with 4 variants and 2 environments:
        - Environment 0 -> Variant 0 (first morph in list)
        - Environment 1 -> Variant 1 (second morph in list)
        - Variants 2 and 3 are unused
    """
    scene = gs.Scene(
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())

    # 4 variants with different positions but only 2 environments
    morphs_heterogeneous = [
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.0, 0.0, 0.1)),
        gs.morphs.Box(size=(0.03, 0.03, 0.03), pos=(0.1, 0.0, 0.15)),
        gs.morphs.Box(size=(0.02, 0.02, 0.02), pos=(0.2, 0.0, 0.2)),
        gs.morphs.Sphere(radius=0.02, pos=(0.3, 0.0, 0.25)),
    ]
    het_obj = scene.add_entity(morph=morphs_heterogeneous)

    # Building with only 2 environments should work - each env gets a unique variant
    scene.build(n_envs=2)

    # Verify mass - env 0 gets variant 0 (0.04 box), env 1 gets variant 1 (0.03 box)
    mass = het_obj.get_mass()
    assert mass.shape == (scene.n_envs,)
    # Different box sizes should have different masses
    assert mass[0] != mass[1]


@pytest.mark.required
def test_mass_setters(tol):
    # Batched links info (default): entity- and link-level set_mass apply, link masses may differ per env, and a
    # wrong-length array is rejected. The heterogeneous entity gives each env a distinct starting mass.
    scene = gs.Scene(
        show_viewer=False,
    )
    het_obj = scene.add_entity(
        morph=[
            gs.morphs.Box(size=(0.01, 0.01, 0.01)),
            gs.morphs.Box(size=(0.02, 0.02, 0.02)),
            gs.morphs.Sphere(radius=0.01),
            gs.morphs.Sphere(radius=0.02),
        ],
    )
    scene.build(n_envs=4)
    link = next(link for link in het_obj.links if not link.is_fixed)
    with pytest.raises(gs.GenesisException):
        link.set_mass((1.0, 2.0))
    het_obj.set_mass(1.0)
    assert_allclose(het_obj.get_mass(), 1.0, tol=tol)
    target_mass = (0.2, 0.4, 0.6, 0.8)
    link.set_mass(target_mass)
    assert_allclose(link.get_mass(), target_mass, tol=tol)

    # Non-batched links info: link mass is shared across envs, so a scalar applies uniformly and a per-env array raises.
    scene = gs.Scene(
        show_viewer=False,
        rigid_options=gs.options.RigidOptions(
            batch_links_info=False,
        ),
    )
    obj = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
        )
    )
    scene.build(n_envs=4)
    link = next(link for link in obj.links if not link.is_fixed)
    link.set_mass(2.0)
    assert_allclose(link.get_mass(), 2.0, tol=tol)
    with pytest.raises(gs.GenesisException):
        link.set_mass((1.0, 2.0, 3.0, 4.0))


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_heterogeneous_aabb(tol):
    """Test that get_AABB and get_vAABB work correctly with heterogeneous simulation."""
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())

    # Box and sphere with different sizes and positions
    morphs_heterogeneous = (
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.0, 0.0, 0.1)),
        gs.morphs.Sphere(radius=0.01, pos=(0.1, 0.0, 0.15)),
    )
    het_obj = scene.add_entity(morph=morphs_heterogeneous)
    # 4 envs: envs 0-1 get box, envs 2-3 get sphere
    scene.build(n_envs=4)

    # Per-variant morph.pos should be correctly applied
    pos = het_obj.get_pos()
    assert_allclose(pos[[0, 1]], (0.0, 0.0, 0.1), tol=tol)
    assert_allclose(pos[[2, 3]], (0.1, 0.0, 0.15), tol=tol)

    # get_AABB should return correct shapes
    aabb = het_obj.get_AABB()
    assert aabb.shape == (scene.n_envs, 2, 3)  # (n_envs, min/max, xyz)
    for i in range(scene.n_envs):
        assert_allclose(aabb[i], het_obj.get_AABB(i), tol=gs.EPS)

    # Box envs should have same AABB, sphere envs should have same AABB
    assert_allclose(aabb[0], aabb[1], tol=gs.EPS)
    assert_allclose(aabb[2], aabb[3], tol=gs.EPS)

    # Box and sphere should have different AABBs (different sizes)
    with pytest.raises(AssertionError):
        assert_allclose(aabb[0], aabb[2], tol=1e-3)

    # get_vAABB should also work
    vaabb = het_obj.get_vAABB()
    assert vaabb.shape == (scene.n_envs, 2, 3)  # (n_envs, min/max, xyz) - same as AABB
    for i in range(scene.n_envs):
        assert_allclose(vaabb[i], het_obj.get_vAABB(i), tol=gs.EPS)

    # vAABB should have same structure as AABB (box envs same, sphere envs same)
    assert_allclose(vaabb[0], vaabb[1], tol=gs.EPS)
    assert_allclose(vaabb[2], vaabb[3], tol=gs.EPS)
    with pytest.raises(AssertionError):
        assert_allclose(vaabb[0], vaabb[2], tol=1e-3)

    # AABB and vAABB sizes should be approximately equal for each environment
    aabb_size_box = aabb[0, 1] - aabb[0, 0]
    vaabb_size_box = vaabb[0, 1] - vaabb[0, 0]
    assert_allclose(aabb_size_box, vaabb_size_box, tol=tol)

    aabb_size_sphere = aabb[2, 1] - aabb[2, 0]
    vaabb_size_sphere = vaabb[2, 1] - vaabb[2, 0]
    assert_allclose(aabb_size_sphere, vaabb_size_sphere, tol=1e-3)  # Allow small tolerance for decimation


# 30s
@pytest.mark.slow  # ~250s
@pytest.mark.parametrize("backend", [gs.gpu])  # Grasping physics requires GPU
def test_pick_heterogenous_objects(show_viewer):
    """Test heterogeneous simulation: CoM at rest, lifting, and gripper width differ per variant."""
    scene = gs.Scene(show_viewer=show_viewer)
    scene.add_entity(gs.morphs.Plane())
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))

    # 4 geometry variants: env i -> variant i
    # Sizes: box0=0.04, box1=0.02, sphere0=0.03, sphere1=0.025 (radius for spheres)
    # Note: spheres need larger radius to be reliably grasped by the Franka gripper
    sizes = [0.04, 0.02, 0.03, 0.025]  # box0, box1, sphere0, sphere1
    het_obj = scene.add_entity(
        morph=[
            gs.morphs.Box(size=(sizes[0],) * 3, pos=(0.65, 0.0, 0.02)),
            gs.morphs.Box(size=(sizes[1],) * 3, pos=(0.65, 0.0, 0.02)),
            gs.morphs.Sphere(radius=sizes[2], pos=(0.65, 0.0, 0.02)),
            gs.morphs.Sphere(radius=sizes[3], pos=(0.65, 0.0, 0.02)),
        ]
    )
    scene.build(n_envs=4, env_spacing=(1, 1))

    # Expected CoM z at rest: half-height for boxes, radius for spheres
    expected_com_z = np.array([sizes[0] / 2, sizes[1] / 2, sizes[2], sizes[3]])

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    init_qpos = np.array([[-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04]] * 4)

    # Set PD gains for finger joints (MJCF raw params have negligible control gain)
    franka.set_dofs_kp([100.0, 100.0], fingers_dof)
    franka.set_dofs_kv([10.0, 10.0], fingers_dof)

    # Initialize robot position
    franka.set_qpos(init_qpos)
    scene.step()

    # Test 1: CoM at rest matches expected heights based on shape
    # Control robot to hold position while objects settle
    for _ in range(30):
        franka.control_dofs_position(init_qpos[:, :7], motors_dof)
        franka.control_dofs_position(init_qpos[:, 7:9], fingers_dof)
        scene.step()
    assert_allclose(het_obj.get_pos()[:, 2], expected_com_z, tol=0.005)

    # Move to grasp position
    end_effector = franka.get_link("hand")
    qpos_grasp = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([[0.65, 0.0, 0.135]] * scene.n_envs),
        quat=np.array([[0, 1, 0, 0]] * scene.n_envs),
    )

    # Hold - approach with gripper open
    for _ in range(50):
        franka.control_dofs_position(qpos_grasp[:, :7], motors_dof)
        franka.control_dofs_position(np.array([[0.04, 0.04]] * scene.n_envs), fingers_dof)
        scene.step()

    # Grasp - close gripper
    for _ in range(50):
        franka.control_dofs_position(qpos_grasp[:, :7], motors_dof)
        franka.control_dofs_position(np.array([[0.0, 0.0]] * scene.n_envs), fingers_dof)
        scene.step()

    # Test 2: Gripper width matches object size (box width or sphere diameter)
    gripper_qpos = franka.get_qpos()[:, 7:9]
    gripper_widths = (gripper_qpos[:, 0] + gripper_qpos[:, 1]).cpu().numpy()
    expected_grip_widths = np.array([sizes[0], sizes[1], 2 * sizes[2], 2 * sizes[3]])  # box size or sphere diameter
    assert_allclose(gripper_widths, expected_grip_widths, tol=0.005)

    # Record positions before lifting
    pre_lift_z = het_obj.get_pos()[:, 2]

    # Lift
    qpos_lift = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([[0.65, 0.0, 0.3]] * scene.n_envs),
        quat=np.array([[0, 1, 0, 0]] * scene.n_envs),
    )
    for _ in range(50):
        franka.control_dofs_position(qpos_lift[:, :7], motors_dof)
        franka.control_dofs_position(np.array([[0.0, 0.0]] * scene.n_envs), fingers_dof)
        scene.step()

    # Test 3: All 4 objects were lifted
    post_lift_z = het_obj.get_pos()[:, 2]
    lift_deltas = tensor_to_array(post_lift_z - pre_lift_z)
    assert np.all(lift_deltas > 0.05), f"All objects should be lifted (deltas={lift_deltas})"


def _build_two_link_revolute_urdf(
    name, geom_tag=None, geom_attribs=None, *, links_geoms=None, links_inertial=None, joint_type="prismatic"
):
    """Build a 2-link URDF file (prismatic joint by default) and return its path.

    Geometry can be specified either uniformly via (geom_tag, geom_attribs) — applied identically
    to all links — or per-link via links_geoms for full control.

    Parameters
    ----------
    links_geoms : list of list of (tag, attribs, origin_xyz) or None
        Per-link geometry specs. Each link gets a list of (tag, attribs, origin_xyz) tuples.
    links_inertial : list of (dict or None) or None
        Per-link inertial overrides. Each dict may contain 'mass', 'ixx', 'iyy', 'izz',
        'ixy', 'ixz', 'iyz', 'origin_xyz'. A None entry leaves that link's inertial unspecified
        (recomputed from geometry); None for the whole list does so for every link.
    joint_type : str
        Type of the joint between the two links ('prismatic', 'revolute', 'fixed', ...). A fixed joint makes the
        second link a fixed child of the first (a single rigid body).
    """
    robot = ET.Element("robot", name=name)

    link_defs = [("base", None), ("moving", "0.1 0 0")]
    for i_link, (link_name, default_origin_xyz) in enumerate(link_defs):
        link = ET.SubElement(robot, "link", name=link_name)
        if links_geoms is not None:
            geoms = links_geoms[i_link]
        else:
            geoms = [(geom_tag, geom_attribs, default_origin_xyz)]
        for tag, attribs, origin_xyz in geoms:
            for group_tag in ("visual", "collision"):
                group = ET.SubElement(link, group_tag)
                geom_el = ET.SubElement(group, "geometry")
                ET.SubElement(geom_el, tag, **attribs)
                if origin_xyz:
                    ET.SubElement(group, "origin", xyz=origin_xyz)
        if links_inertial and links_inertial[i_link] is not None:
            inertial_props = links_inertial[i_link]
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "mass", value=str(inertial_props["mass"]))
            ET.SubElement(inertial, "origin", xyz=inertial_props["origin_xyz"])
            ET.SubElement(
                inertial,
                "inertia",
                ixx=str(inertial_props.get("ixx", 0)),
                ixy=str(inertial_props.get("ixy", 0)),
                ixz=str(inertial_props.get("ixz", 0)),
                iyy=str(inertial_props.get("iyy", 0)),
                iyz=str(inertial_props.get("iyz", 0)),
                izz=str(inertial_props.get("izz", 0)),
            )

    joint = ET.SubElement(robot, "joint", name="joint1", type=joint_type)
    ET.SubElement(joint, "parent", link="base")
    ET.SubElement(joint, "child", link="moving")
    ET.SubElement(joint, "origin", xyz="0.1 0 0")
    if joint_type != "fixed":
        ET.SubElement(joint, "axis", xyz="1 0 0")
        ET.SubElement(joint, "limit", lower="-1.0", upper="1.0", effort="100", velocity="1.0")

    return urdfpy.URDF._from_xml(robot, robot, get_assets_dir())


def _build_free_body_urdf(name, com_xyz):
    """Build a single free-floating link URDF with a box geom and an off-center COM, returning its path."""
    robot = ET.Element("robot", name=name)
    link = ET.SubElement(robot, "link", name="body")
    for group_tag in ("visual", "collision"):
        group = ET.SubElement(link, group_tag)
        geom_el = ET.SubElement(group, "geometry")
        ET.SubElement(geom_el, "box", size="0.04 0.04 0.04")
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "mass", value="0.5")
    ET.SubElement(inertial, "origin", xyz=com_xyz)
    ET.SubElement(inertial, "inertia", ixx="1e-3", iyy="1e-3", izz="1e-3", ixy="0", ixz="0", iyz="0")
    return urdfpy.URDF._from_xml(robot, robot, get_assets_dir())


def _build_wrapped_free_body_urdf(name, child_mass):
    """Build a free body whose root link is empty (no geom, no inertial) and whose mass and geometry live on a fixed
    child link, returning its path. Exercises the empty-free-root wrapping a fixed massive child topology."""
    robot = ET.Element("robot", name=name)
    ET.SubElement(robot, "link", name="root")
    child = ET.SubElement(robot, "link", name="payload")
    for group_tag in ("visual", "collision"):
        group = ET.SubElement(child, group_tag)
        geom_el = ET.SubElement(group, "geometry")
        ET.SubElement(geom_el, "box", size="0.04 0.04 0.04")
    inertial = ET.SubElement(child, "inertial")
    ET.SubElement(inertial, "mass", value=str(child_mass))
    ET.SubElement(inertial, "origin", xyz="0 0 0")
    ET.SubElement(inertial, "inertia", ixx="1e-3", iyy="1e-3", izz="1e-3", ixy="0", ixz="0", iyz="0")
    joint = ET.SubElement(robot, "joint", name="weld", type="fixed")
    ET.SubElement(joint, "parent", link="root")
    ET.SubElement(joint, "child", link="payload")
    ET.SubElement(joint, "origin", xyz="0 0 0")
    return urdfpy.URDF._from_xml(robot, robot, get_assets_dir())


@pytest.mark.slow  # ~250s
@pytest.mark.required
def test_align_heterogeneous_inertial(show_viewer, tol):
    GRAVITY = -9.81

    # Variant A: sphere mesh collision with explicit inertial properties per link
    sphere_base_mass, sphere_moving_mass = 0.5, 0.3
    sphere_base_com = (0.0, 0.01, 0.0)
    sphere_moving_com = (0.05, 0.0, 0.0)
    sphere_base_inertia_diag = 1e-4
    sphere_moving_inertia_diag = 5e-5
    urdf_spheres = _build_two_link_revolute_urdf(
        "two_sphere_revolute",
        "mesh",
        {"filename": os.path.join(get_assets_dir(), "meshes", "sphere.obj"), "scale": "0.08 0.08 0.08"},
        links_inertial=[
            {
                "mass": sphere_base_mass,
                "ixx": sphere_base_inertia_diag,
                "iyy": sphere_base_inertia_diag,
                "izz": sphere_base_inertia_diag,
                "origin_xyz": " ".join(map(str, sphere_base_com)),
            },
            {
                "mass": sphere_moving_mass,
                "ixx": sphere_moving_inertia_diag,
                "iyy": sphere_moving_inertia_diag,
                "izz": sphere_moving_inertia_diag,
                "origin_xyz": " ".join(map(str, sphere_moving_com)),
            },
        ],
    )

    # Variant B: 2 half-height box primitives per link.
    # Setting zero inertial to test recomputation from geometry for non-primary morph.
    half_box = {"size": "0.04 0.04 0.02"}
    urdf_boxes = _build_two_link_revolute_urdf(
        "two_box_revolute",
        links_geoms=[
            [("box", half_box, "0 0 0.01"), ("box", half_box, "0 0 -0.01")],
            [("box", half_box, "0.1 0 0.01"), ("box", half_box, "0.1 0 -0.01")],
        ],
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, GRAVITY),
        ),
        rigid_options=gs.options.RigidOptions(
            # Allow specifying different controller gains for each env
            batch_dofs_info=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, 1.0, 1.0),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(gs.morphs.Plane())
    # align=True is requested but must be ignored for these articulated robots: a free base with a DOF-bearing child is
    # not a single rigid body, so its link frames and joint-space mass coupling must be left intact (aligning it would
    # misplace the moving child and drop the base coupling). The link-spacing and settling assertions below verify this.
    het_morph = (
        gs.morphs.URDF(file=urdf_spheres, pos=(0.5, 0, 0.08), align=True),
        gs.morphs.URDF(file=urdf_boxes, pos=(0, 0, 0.02), align=True),
    )
    het_obj = scene.add_entity(
        morph=het_morph,
        material=gs.materials.Rigid(
            rho=200.0,
            friction=1e-2,
        ),
    )
    # Kinematic heterogeneous URDF entity (same variants, different positions)
    het_kin = scene.add_entity(
        morph=het_morph,
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Default(
            color=(0.0, 0.0, 1.0, 0.4),
        ),
    )
    # Free-floating single-link URDF objects with different off-center COMs. Unlike the articulated robots above (whose
    # requested alignment is ignored), each variant is a basic rigid object, so its link frame is moved to its own COM.
    FREE_POS = (3.0, 0.0, 0.2)
    free_morph = (
        gs.morphs.URDF(file=_build_free_body_urdf("free_body_a", "0.02 0 0"), pos=FREE_POS, align=True),
        gs.morphs.URDF(file=_build_free_body_urdf("free_body_b", "0 0 0.03"), pos=FREE_POS, align=True),
    )
    free_het = scene.add_entity(morph=free_morph, material=gs.materials.Rigid(rho=200.0))
    # Kinematic counterpart of the aligned free bodies. The COM/principal anchoring is applied in the base entity, so
    # for the same qpos a kinematic visualization and the rigid body it tracks must place identical world geometry.
    free_kin = scene.add_entity(morph=free_morph, material=gs.materials.Kinematic())
    # A free entity whose two variants are identical, so the solver resolves a single shared offset and takes the
    # broadcast path (the base link offset) rather than the per-env variant offset. That shared base offset must still
    # carry the COM anchoring, else the relative getter cannot strip the alignment on this path.
    dup_morph = (
        gs.morphs.URDF(file=_build_free_body_urdf("free_dup_a", "0.02 0 0"), pos=FREE_POS, align=True),
        gs.morphs.URDF(file=_build_free_body_urdf("free_dup_b", "0.02 0 0"), pos=FREE_POS, align=True),
    )
    free_dup = scene.add_entity(morph=dup_morph, material=gs.materials.Rigid())
    # Free bodies whose root link is empty and whose mass lives on a fixed child (merge_fixed_links=False keeps the
    # wrapper). Alignment folds the child's mass onto the root; the subsumed child keeps only the gs.EPS placeholder.
    WRAP_MASS_A, WRAP_MASS_B = 0.5, 0.25
    wrapped_morph = (
        gs.morphs.URDF(
            file=_build_wrapped_free_body_urdf("wrap_a", WRAP_MASS_A),
            pos=(5.0, 0.0, 0.2),
            align=True,
            merge_fixed_links=False,
        ),
        gs.morphs.URDF(
            file=_build_wrapped_free_body_urdf("wrap_b", WRAP_MASS_B),
            pos=(5.0, 0.0, 0.2),
            align=True,
            merge_fixed_links=False,
        ),
    )
    free_wrapped = scene.add_entity(morph=wrapped_morph)
    scene.build(n_envs=4, env_spacing=(0.0, 0.5))

    # Same absolute qpos must map to the same world geometry for the aligned rigid body and its kinematic counterpart;
    # an unanchored kinematic entity would interpret the qpos in a different link frame and diverge.
    free_qpos = (1.0, 0.5, 0.8, 0.6, 0.5, 0.3, 0.0)
    free_het.set_qpos(free_qpos)
    free_kin.set_qpos(free_qpos)
    assert_allclose(free_het.get_vAABB(), free_kin.get_vAABB(), tol=tol)
    scene.reset()

    # Each free-body variant (env 0 variant A, env 2 variant B) is aligned to its own COM: the link origin coincides
    # with the COM, and the relative getter strips the alignment back to the user pose.
    for i_env in (0, 2):
        assert_allclose(
            free_het.get_links_pos(links_idx_local=[0], ref="link_com", envs_idx=i_env, relative=False),
            free_het.get_links_pos(links_idx_local=[0], ref="link_origin", envs_idx=i_env, relative=False),
            tol=tol,
        )
        assert_allclose(free_het.get_pos(relative=True, envs_idx=i_env), FREE_POS, tol=tol)

    # The duplicate-variant entity takes the broadcast offset path; its relative getter must still strip the shared COM
    # anchoring back to the user pose in every env (a non-anchored base offset would leave it COM-shifted).
    assert_allclose(
        free_dup.get_links_pos(links_idx_local=[0], ref="link_com", relative=False),
        free_dup.get_links_pos(links_idx_local=[0], ref="link_origin", relative=False),
        tol=tol,
    )
    assert_allclose(free_dup.get_pos(relative=True), FREE_POS, tol=tol)
    assert_allclose(gu.quat_to_xyz(free_dup.get_quat(relative=True)), 0.0, tol=tol)

    # Relative set_pos on a boolean-masked subset of envs: each selected env's relative getter must report its target
    # back, stripping its own per-variant offset.
    mask = torch.tensor([True, False, True, False], device=gs.device)
    free_new_pos = torch.tensor([[1.0, 0.0, 0.5], [2.0, 0.0, 0.5]], dtype=gs.tc_float, device=gs.device)
    free_het.set_pos(free_new_pos, envs_idx=mask, relative=True)
    free_pos = free_het.get_pos(relative=True)
    assert_allclose(free_pos[[0, 2]], free_new_pos, tol=tol)
    assert_allclose(free_pos[[1, 3]], FREE_POS, tol=tol)

    # Joint structure: both variants share the same joints (root_joint + joint1)
    assert len(het_obj.joints) == 2
    assert len(het_obj.links) == 2
    assert het_obj.get_qpos().shape == (4, 8)  # free joint (7) + prismatic (1)
    assert het_obj.get_dofs_velocity().shape == (4, 7)  # free joint (6) + prismatic (1)

    # Check that kinematic vAABB matches rigid
    assert_allclose(het_kin.get_vAABB(), het_obj.get_vAABB(), tol=1e-3)

    # Verify initial z-positions match per-variant morph.pos (z is unaffected by env_spacing)
    het_pos_init = het_obj.get_pos()
    assert_allclose(het_pos_init[0, 2], 0.08, tol=tol)
    assert_allclose(het_pos_init[1, 2], 0.08, tol=tol)
    assert_allclose(het_pos_init[2, 2], 0.02, tol=tol)
    assert_allclose(het_pos_init[3, 2], 0.02, tol=tol)
    # Variant B has x-offset relative to variant A
    assert_allclose(het_pos_init[0, 0] - het_pos_init[2, 0], 0.5, tol=tol)
    assert_allclose(het_pos_init[1, 0] - het_pos_init[3, 0], 0.5, tol=tol)
    het_links_pos_init = het_obj.get_links_pos(relative=False)
    assert_allclose(het_links_pos_init.diff(dim=-2), (0.1, 0, 0), tol=tol)

    # Same-variant envs produce identical results (balanced block [A, A, B, B])
    het_pos = het_obj.get_pos()
    het_qpos = het_obj.get_qpos()
    assert_allclose(het_pos[0], het_pos[1], tol=tol)
    assert_allclose(het_pos[2], het_pos[3], tol=tol)
    assert_allclose(het_qpos[0], het_qpos[1], tol=tol)
    assert_allclose(het_qpos[2], het_qpos[3], tol=tol)

    # Different-variant envs produce different results
    with pytest.raises(AssertionError):
        assert_allclose(het_pos[0], het_pos[2], tol=tol)
    with pytest.raises(AssertionError):
        assert_allclose(het_qpos[0], het_qpos[2], tol=tol)

    # Mass differs between variants
    mass = het_obj.get_mass()
    assert mass.shape == (scene.n_envs,)
    assert_allclose(mass[0], mass[1], tol=tol)
    assert_allclose(mass[2], mass[3], tol=tol)
    assert not np.allclose(mass[0], mass[2], atol=tol, rtol=tol), "Variant A and B masses should differ"
    # Variant B total mass should match the explicit URDF inertial values
    assert_allclose(mass[0], sphere_base_mass + sphere_moving_mass, tol=tol)

    # CoM position: variant B should match explicit URDF inertial origin_xyz
    com_pos = het_obj.get_links_pos(ref="link_com", relative=False)
    origin_pos = het_obj.get_links_pos(ref="link_origin", relative=False)
    com_offset = com_pos - origin_pos
    # Variant A: CoM offset matches URDF inertial origin
    assert_allclose(com_offset[0, 0], sphere_base_com, tol=tol)
    assert_allclose(com_offset[0, 1], sphere_moving_com, tol=tol)
    assert_allclose(com_offset[1, 0], sphere_base_com, tol=tol)
    assert_allclose(com_offset[1, 1], sphere_moving_com, tol=tol)
    # Variant B: symmetric split boxes => CoM at link origin for base, at geometry center for moving
    assert_allclose(com_offset[2, 0], 0.0, tol=tol)
    assert_allclose(com_offset[2, 1, 0], 0.1, tol=tol)  # x-offset of moving link geometry
    # Same-variant consistency
    assert_allclose(com_offset[2], com_offset[3], tol=tol)
    # CoM differs between variants on base link (non-zero y-offset for variant B)
    with pytest.raises(AssertionError):
        assert_allclose(com_offset[0, 0], com_offset[2, 0], tol=tol)

    # Inertia matrix: variant B should match explicit URDF values
    links_idx = slice(het_obj.link_start, het_obj.link_end)
    inertial_i = qd_to_numpy(scene.rigid_solver.links_info.inertial_i, None, links_idx, transpose=True)
    # Variant A: diagonal inertia matches URDF
    assert_allclose(inertial_i[[0, 1], 0], np.eye(3) * sphere_base_inertia_diag, tol=tol)
    assert_allclose(inertial_i[[0, 1], 1], np.eye(3) * sphere_moving_inertia_diag, tol=tol)
    # Variant B: Recomputed inertia from geometry
    assert_allclose(inertial_i[[2, 3]], np.eye(3) * ((0.04**5 / 6.0) * het_obj.material.rho), tol=tol)
    # Same-variant consistency
    assert_allclose(inertial_i[2], inertial_i[3], tol=tol)
    # Variants differ
    with pytest.raises(AssertionError):
        assert_allclose(inertial_i[0, 0], inertial_i[2, 0], tol=tol)

    # Empty-free-root wrapping a fixed massive child: alignment folds the composite mass onto the root (link 0),
    # leaving the subsumed child (link 1) with only the gs.EPS placeholder. The root must carry exactly the child's
    # mass; a prior bug summed the root's own gs.EPS placeholder into the composite, inflating it by one gs.EPS
    # (hence the sub-EPS tolerance below). Envs are dispatched as [A, A, B, B].
    wrapped_idx = slice(free_wrapped.link_start, free_wrapped.link_end)
    wrapped_mass = qd_to_numpy(scene.rigid_solver.links_info.inertial_mass, None, wrapped_idx, transpose=True)
    assert_allclose(wrapped_mass[[0, 1], 0], WRAP_MASS_A, atol=gs.EPS * 0.5)
    assert_allclose(wrapped_mass[[2, 3], 0], WRAP_MASS_B, atol=gs.EPS * 0.5)
    assert_allclose(wrapped_mass[:, 1], gs.EPS, atol=gs.EPS * 1e-3)

    # Check contacts
    for i in range(4):
        for _ in range(10):
            scene.step()
        pos = het_obj.get_pos()
        assert_allclose(pos[:2, [0, 2]], het_pos_init[:2, [0, 2]], tol=1e-3)
        assert_allclose(pos[:2, 1], het_pos_init[:2, 1], tol=0.02)
        assert_allclose(pos[2:], het_pos_init[2:], tol=2e-4)
        het_obj.set_quat(gu.euler_to_quat((90 * i, 0, 0)))

    # Apply control and simulate for a while
    target_dof_pos = np.array((0.05, 0.1, 0.01, 0.02), dtype=gs.np_float)
    het_obj.set_dofs_kp((1000.0, 1000.0, 100.0, 100.0), dofs_idx_local=-1)
    het_obj.set_dofs_kv((100.0, 100.0, 10.0, 10.0), dofs_idx_local=-1)
    het_obj.control_dofs_position(target_dof_pos, dofs_idx_local=-1)
    for _ in range(100):
        scene.step()

    # Velocity should be near zero (settled)
    assert_allclose(het_obj.get_vel(), 0.0, tol=0.05)

    # All objects should be near their initial z-positions (settled on ground)
    pos = het_obj.get_pos()
    assert_allclose(pos[..., 2], het_pos_init[..., 2], tol=1e-3)

    # Check that dof position is correct
    dof_pos = het_obj.get_dofs_position()
    assert_allclose(dof_pos[..., -1], target_dof_pos, tol=1e-3)
    het_links_pos = het_obj.get_links_pos(relative=False)
    assert_allclose(het_links_pos[..., 1, 0] - het_links_pos[..., 0, 0], target_dof_pos + 0.1, tol=1e-3)
    assert_allclose(het_links_pos[..., 1, 1:], het_links_pos[..., 0, 1:], tol=5e-3)

    # Check that the acceleration is matching the analytical formula
    links_mass = qd_to_numpy(scene.rigid_solver.links_info.inertial_mass, None, links_idx, transpose=True)
    force = np.zeros((scene.n_envs, 2, 3))
    force[..., 2] = -links_mass * GRAVITY
    het_obj.set_pos((0, 0, 0.2))
    het_obj.control_dofs_force(0.0, dofs_idx_local=-1)
    scene.step()
    assert_allclose(het_obj.get_links_acc()[..., 2], GRAVITY, tol=tol)
    het_obj.zero_all_dofs_velocity()
    for _ in range(10):
        scene.rigid_solver.apply_links_external_force(force, links_idx=links_idx, ref="link_com")
        scene.step()
        assert_allclose(het_obj.get_links_acc(), 0.0, tol=tol)


@pytest.mark.required
def test_heterogeneous_articulated_structure_mismatch():
    """Test that mismatched joint structure raises an exception."""
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())

    # two_cube_revolute has 1 revolute joint; two_link_arm has 2 continuous joints
    with pytest.raises(gs.GenesisException):
        scene.add_entity(
            morph=[
                gs.morphs.URDF(file="urdf/simple/two_cube_revolute.urdf", pos=(0, 0, 0.1)),
                gs.morphs.URDF(file="urdf/simple/two_link_arm.urdf", pos=(0, 0, 0.1)),
            ]
        )


@pytest.mark.required
@pytest.mark.parametrize("integrator", [gs.integrator.Euler, gs.integrator.approximate_implicitfast])
def test_energy_analytical_and_conservation(show_viewer, tol, integrator):
    g = 9.81
    dt = 0.001
    h0 = 0.5
    radius = 0.1
    n_steps = 400
    undamped_sol_params = [10.0, 0.001, 0.9, 0.95, 0.001, 0.5, 2.0]

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            gravity=(0, 0, -g),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.25, 1.5, 0.7),
            camera_lookat=(0.25, 0.0, 0.2),
        ),
        rigid_options=gs.options.RigidOptions(
            integrator=integrator,
        ),
        show_viewer=show_viewer,
    )
    plane = scene.add_entity(gs.morphs.Plane())
    sphere_a = scene.add_entity(
        gs.morphs.Sphere(
            radius=radius,
            pos=(0, 0, h0),
        ),
    )
    sphere_b = scene.add_entity(
        gs.morphs.Sphere(
            radius=radius,
            pos=(0.5, 0, h0),
        ),
    )
    scene.build()

    # Nearly undamped contact for sphere_a: small dampratio gives very stiff elastic spring with minimal damping.
    # Contact sol_params are averaged: 0.5*(geom_a + geom_b), so both geoms must share the same params.
    plane.geoms[0].set_sol_params(undamped_sol_params)
    sphere_a.geoms[0].set_sol_params(undamped_sol_params)

    mass = sphere_a.get_links_inertial_mass()
    te_initial = sphere_a.get_total_energy()

    ke_a, pe_a, ke_b, pe_b = [], [], [], []
    impact_step = -1
    for i in range(n_steps):
        scene.step()
        ke_a.append(sphere_a.get_kinetic_energy())
        pe_a.append(sphere_a.get_potential_energy())
        ke_b.append(sphere_b.get_kinetic_energy())
        pe_b.append(sphere_b.get_potential_energy())
        if impact_step < 0 and scene.rigid_solver.collider._collider_state.n_contacts.to_numpy().any():
            impact_step = i
    assert impact_step > 0

    # Free fall: verify analytical KE and PE (semi-implicit Euler)
    # After step n: v_n = n*g*dt, z_n = h0 - g*dt^2*n*(n+1)/2
    for i in range(impact_step):
        n = i + 1
        expected_ke = 0.5 * mass * (n * g * dt) ** 2
        expected_pe = mass * g * (h0 - g * dt**2 * n * (n + 1) / 2)
        assert_allclose(ke_a[i], expected_ke, tol=tol)
        assert_allclose(pe_a[i], expected_pe, tol=tol)
        assert_allclose(ke_b[i], expected_ke, tol=tol)
        assert_allclose(pe_b[i], expected_pe, tol=tol)

    # Undamped sphere_a: energy conserved after bouncing (drift < 1%)
    te_a_final = ke_a[-1] + pe_a[-1]
    assert_allclose(te_a_final, te_initial, tol=0.01)

    # Damped sphere_b: energy strictly decreased
    te_b_final = ke_b[-1] + pe_b[-1]
    assert te_b_final < te_initial
