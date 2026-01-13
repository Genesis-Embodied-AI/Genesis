"""
Test USD parsing and comparison with MJCF scenes.

This module tests that USD files can be parsed correctly and that scenes
loaded from USD files match equivalent scenes loaded from MJCF files.
"""

import os
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import get_assets_dir, tensor_to_array
from genesis.utils import geom as gu

from .utils import assert_allclose, get_hf_dataset

# Check for USD support
try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    HAS_USD_SUPPORT = True
except ImportError:
    HAS_USD_SUPPORT = False


def compare_links(mjcf_links, usd_links, tol):
    """
    Generic function to compare links between two scenes.
    Compares as much link data as possible including positions, orientations,
    inertial properties, structural properties, etc.

    Parameters
    ----------
    mjcf_links : list
        List of links from MJCF scene
    usd_links : list
        List of links from USD scene
    tol : float, optional
        Tolerance for numerical comparisons. Defaults to 1e-5.
    """
    # Check number of links
    assert len(mjcf_links) == len(usd_links), f"Number of links mismatch: MJCF={len(mjcf_links)}, USD={len(usd_links)}"

    # Create dictionaries keyed by link name for comparison
    mjcf_links_by_name = {link.name: link for link in mjcf_links}
    usd_links_by_name = {link.name: link for link in usd_links}

    # Create index to name mappings for parent comparison
    mjcf_idx_to_name = {i: link.name for i, link in enumerate(mjcf_links)}
    usd_idx_to_name = {i: link.name for i, link in enumerate(usd_links)}

    # Check that we have matching link names
    mjcf_link_names = set(mjcf_links_by_name.keys())
    usd_link_names = set(usd_links_by_name.keys())
    assert mjcf_link_names == usd_link_names, f"Link names mismatch: MJCF={mjcf_link_names}, USD={usd_link_names}"

    # Compare all link properties by name
    for link_name in sorted(mjcf_link_names):
        mjcf_link = mjcf_links_by_name[link_name]
        usd_link = usd_links_by_name[link_name]

        # Compare position
        assert_allclose(
            mjcf_link.pos,
            usd_link.pos,
            tol=tol,
            err_msg=f"Link '{link_name}' position mismatch: MJCF={mjcf_link.pos}, USD={usd_link.pos}",
        )

        # Compare quaternion
        assert_allclose(
            mjcf_link.quat,
            usd_link.quat,
            tol=tol,
            err_msg=f"Link '{link_name}' quat mismatch: MJCF={mjcf_link.quat}, USD={usd_link.quat}",
        )

        # Compare is_fixed
        assert mjcf_link.is_fixed == usd_link.is_fixed, (
            f"Link '{link_name}' is_fixed mismatch: MJCF={mjcf_link.is_fixed}, USD={usd_link.is_fixed}"
        )

        # Compare number of geoms
        assert len(mjcf_link.geoms) == len(usd_link.geoms), (
            f"Link '{link_name}' number of geoms mismatch: MJCF={len(mjcf_link.geoms)}, USD={len(usd_link.geoms)}"
        )

        # Compare number of joints
        assert mjcf_link.n_joints == usd_link.n_joints, (
            f"Link '{link_name}' number of joints mismatch: MJCF={mjcf_link.n_joints}, USD={usd_link.n_joints}"
        )

        # Compare number of visual geoms
        assert len(mjcf_link.vgeoms) == len(usd_link.vgeoms), (
            f"Link '{link_name}' number of vgeoms mismatch: MJCF={len(mjcf_link.vgeoms)}, USD={len(usd_link.vgeoms)}"
        )

        # Compare parent link by name (mapping indices to names)
        mjcf_parent_idx = mjcf_link.parent_idx
        usd_parent_idx = usd_link.parent_idx

        if mjcf_parent_idx == -1:
            mjcf_parent_name = None
        else:
            mjcf_parent_name = mjcf_idx_to_name.get(mjcf_parent_idx, f"<unknown idx {mjcf_parent_idx}>")

        if usd_parent_idx == -1:
            usd_parent_name = None
        else:
            usd_parent_name = usd_idx_to_name.get(usd_parent_idx, f"<unknown idx {usd_parent_idx}>")

        assert mjcf_parent_name == usd_parent_name, (
            f"Link '{link_name}' parent mismatch: MJCF parent_idx={mjcf_parent_idx}, parent_name={mjcf_parent_name}, USD parent_idx={usd_parent_idx}, parent_name={usd_parent_name}"
        )

        # Compare inertial properties if available
        mjcf_inertial_pos = mjcf_link.inertial_pos
        usd_inertial_pos = usd_link.inertial_pos
        assert_allclose(
            mjcf_inertial_pos,
            usd_inertial_pos,
            tol=tol,
            err_msg=f"Link '{link_name}' inertial_pos mismatch: MJCF={mjcf_inertial_pos}, USD={usd_inertial_pos}",
        )

        mjcf_inertial_quat = mjcf_link.inertial_quat
        usd_inertial_quat = usd_link.inertial_quat
        assert_allclose(
            mjcf_inertial_quat,
            usd_inertial_quat,
            tol=tol,
            err_msg=f"Link '{link_name}' inertial_quat mismatch: MJCF={mjcf_inertial_quat}, USD={usd_inertial_quat}",
        )

        # Skip mass and inertia checks for fixed links - they're not used in simulation
        if not mjcf_link.is_fixed:
            mjcf_inertial_mass = mjcf_link.inertial_mass
            usd_inertial_mass = usd_link.inertial_mass
            # Both scenes now use the same material density (1000 kg/m³), so values should match closely
            assert_allclose(
                mjcf_inertial_mass,
                usd_inertial_mass,
                atol=tol,
                err_msg=f"Link '{link_name}' inertial_mass mismatch: MJCF={mjcf_inertial_mass}, USD={usd_inertial_mass}",
            )

            mjcf_inertial_i = mjcf_link.inertial_i
            usd_inertial_i = usd_link.inertial_i
            # Both scenes now use the same material density (1000 kg/m³), so values should match closely
            assert_allclose(
                mjcf_inertial_i,
                usd_inertial_i,
                atol=tol,
                err_msg=f"Link '{link_name}' inertial_i mismatch: MJCF={mjcf_inertial_i}, USD={usd_inertial_i}",
            )


def compare_joints(mjcf_joints, usd_joints, tol):
    """
    Generic function to compare joints between two scenes.
    Compares as much joint data as possible including positions, orientations,
    degrees of freedom, limits, dynamics properties, etc.

    Parameters
    ----------
    mjcf_joints : list
        List of joints from MJCF scene
    usd_joints : list
        List of joints from USD scene
    tol : float, optional
        Tolerance for numerical comparisons. Defaults to 1e-5.
    """
    # Check number of joints
    assert len(mjcf_joints) == len(usd_joints), (
        f"Number of joints mismatch: MJCF={len(mjcf_joints)}, USD={len(usd_joints)}"
    )

    # Create dictionaries keyed by joint name for comparison
    mjcf_joints_by_name = {joint.name: joint for joint in mjcf_joints}
    usd_joints_by_name = {joint.name: joint for joint in usd_joints}

    # Check that we have matching joint names
    mjcf_joint_names = set(mjcf_joints_by_name.keys())
    usd_joint_names = set(usd_joints_by_name.keys())
    assert mjcf_joint_names == usd_joint_names, f"Joint names mismatch: MJCF={mjcf_joint_names}, USD={usd_joint_names}"

    # Compare all joint properties by name
    for joint_name in sorted(mjcf_joint_names):
        mjcf_joint = mjcf_joints_by_name[joint_name]
        usd_joint = usd_joints_by_name[joint_name]

        # Compare joint type
        assert mjcf_joint.type == usd_joint.type, (
            f"Joint '{joint_name}' type mismatch: MJCF={mjcf_joint.type}, USD={usd_joint.type}"
        )

        # Compare position
        assert_allclose(
            mjcf_joint.pos,
            usd_joint.pos,
            tol=tol,
            err_msg=f"Joint '{joint_name}' position mismatch: MJCF={mjcf_joint.pos}, USD={usd_joint.pos}",
        )

        # Compare quaternion
        assert_allclose(
            mjcf_joint.quat,
            usd_joint.quat,
            tol=tol,
            err_msg=f"Joint '{joint_name}' quat mismatch: MJCF={mjcf_joint.quat}, USD={usd_joint.quat}",
        )

        # Compare number of qs and dofs
        assert mjcf_joint.n_qs == usd_joint.n_qs, (
            f"Joint '{joint_name}' n_qs mismatch: MJCF={mjcf_joint.n_qs}, USD={usd_joint.n_qs}"
        )

        assert mjcf_joint.n_dofs == usd_joint.n_dofs, (
            f"Joint '{joint_name}' n_dofs mismatch: MJCF={mjcf_joint.n_dofs}, USD={usd_joint.n_dofs}"
        )

        # Compare initial qpos
        assert_allclose(
            mjcf_joint.init_qpos,
            usd_joint.init_qpos,
            tol=tol,
            err_msg=f"Joint '{joint_name}' init_qpos mismatch: MJCF={mjcf_joint.init_qpos}, USD={usd_joint.init_qpos}",
        )

        # Skip mass/inertia-dependent property checks for fixed joints - they're not used in simulation
        if mjcf_joint.type != gs.JOINT_TYPE.FIXED:
            # Compare dof limits
            assert_allclose(
                mjcf_joint.dofs_limit,
                usd_joint.dofs_limit,
                tol=tol,
                err_msg=f"Joint '{joint_name}' dofs_limit mismatch: MJCF={mjcf_joint.dofs_limit}, USD={usd_joint.dofs_limit}",
            )

            # Compare dof motion properties
            assert_allclose(
                mjcf_joint.dofs_motion_ang,
                usd_joint.dofs_motion_ang,
                tol=tol,
                err_msg=f"Joint '{joint_name}' dofs_motion_ang mismatch: MJCF={mjcf_joint.dofs_motion_ang}, USD={usd_joint.dofs_motion_ang}",
            )

            assert_allclose(
                mjcf_joint.dofs_motion_vel,
                usd_joint.dofs_motion_vel,
                tol=tol,
                err_msg=f"Joint '{joint_name}' dofs_motion_vel mismatch: MJCF={mjcf_joint.dofs_motion_vel}, USD={usd_joint.dofs_motion_vel}",
            )

            assert_allclose(
                mjcf_joint.dofs_frictionloss,
                usd_joint.dofs_frictionloss,
                tol=tol,
                err_msg=f"Joint '{joint_name}' dofs_frictionloss mismatch: MJCF={mjcf_joint.dofs_frictionloss}, USD={usd_joint.dofs_frictionloss}",
            )

            assert_allclose(
                mjcf_joint.dofs_stiffness,
                usd_joint.dofs_stiffness,
                tol=tol,
                err_msg=f"Joint '{joint_name}' dofs_stiffness mismatch: MJCF={mjcf_joint.dofs_stiffness}, USD={usd_joint.dofs_stiffness}",
            )

            assert_allclose(
                mjcf_joint.dofs_damping,
                usd_joint.dofs_damping,
                tol=tol,
                err_msg=f"Joint '{joint_name}' dofs_damping mismatch: MJCF={mjcf_joint.dofs_damping}, USD={usd_joint.dofs_damping}",
            )

            assert_allclose(
                mjcf_joint.dofs_armature,
                usd_joint.dofs_armature,
                tol=tol,
                err_msg=f"Joint '{joint_name}' dofs_armature mismatch: MJCF={mjcf_joint.dofs_armature}, USD={usd_joint.dofs_armature}",
            )

            # Compare dof control properties
            assert_allclose(
                mjcf_joint.dofs_kp,
                usd_joint.dofs_kp,
                tol=tol,
                err_msg=f"Joint '{joint_name}' dofs_kp mismatch: MJCF={mjcf_joint.dofs_kp}, USD={usd_joint.dofs_kp}",
            )

            assert_allclose(
                mjcf_joint.dofs_kv,
                usd_joint.dofs_kv,
                tol=tol,
                err_msg=f"Joint '{joint_name}' dofs_kv mismatch: MJCF={mjcf_joint.dofs_kv}, USD={usd_joint.dofs_kv}",
            )

            # Compare dof force range
            assert_allclose(
                mjcf_joint.dofs_force_range,
                usd_joint.dofs_force_range,
                tol=tol,
                err_msg=f"Joint '{joint_name}' dofs_force_range mismatch: MJCF={mjcf_joint.dofs_force_range}, USD={usd_joint.dofs_force_range}",
            )


def compare_geoms(mjcf_geoms, usd_geoms, tol):
    """
    Generic function to compare geoms between two scenes.
    Compares as much geom data as possible including positions, orientations,
    sizes, etc.
    """
    assert len(mjcf_geoms) == len(usd_geoms), f"Number of geoms mismatch: MJCF={len(mjcf_geoms)}, USD={len(usd_geoms)}"

    # Sort geoms by link name for consistent comparison
    mjcf_geoms_sorted = sorted(mjcf_geoms, key=lambda g: (g.link.name, g._idx))
    usd_geoms_sorted = sorted(usd_geoms, key=lambda g: (g.link.name, g._idx))

    for mjcf_geom, usd_geom in zip(mjcf_geoms_sorted, usd_geoms_sorted):
        assert mjcf_geom.type == usd_geom.type, (
            f"Geom type mismatch: MJCF={mjcf_geom.type}, USD={usd_geom.type}, link={mjcf_geom.link.name}"
        )


def compare_scene(mjcf_scene: gs.Scene, usd_scene: gs.Scene, tol):
    """Compare structure and data between MJCF and USD scenes."""
    mjcf_entities = mjcf_scene.entities
    usd_entities = usd_scene.entities

    mjcf_links = []
    for entity in mjcf_entities:
        mjcf_links.extend(entity.links)

    usd_links = []
    for entity in usd_entities:
        usd_links.extend(entity.links)

    compare_links(mjcf_links, usd_links, tol=tol)

    mjcf_geoms = []
    for entity in mjcf_entities:
        mjcf_geoms.extend(entity.geoms)

    usd_geoms = []
    for entity in usd_entities:
        usd_geoms.extend(entity.geoms)

    compare_geoms(mjcf_geoms, usd_geoms, tol=tol)

    mjcf_joints = []
    for entity in mjcf_entities:
        mjcf_joints.extend(entity.joints)

    usd_joints = []
    for entity in usd_entities:
        usd_joints.extend(entity.joints)

    compare_joints(mjcf_joints, usd_joints, tol=tol)


@pytest.fixture(scope="session")
def mjcf_file(box_plane_mjcf):
    """Create a temporary MJCF file from the fixture."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        mjcf_file = f.name
        ET.ElementTree(box_plane_mjcf).write(mjcf_file, encoding="utf-8", xml_declaration=True)
    return mjcf_file


@pytest.fixture(scope="session")
def box_plane_mjcf():
    """
    Generate an MJCF model for a box on a plane.

    - Using the USD path syntax for the names of the bodies and joints to keep track of the hierarchy.
    """
    mjcf = ET.Element("mujoco", model="one_box")
    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "joint", armature="0.0")

    worldbody = ET.SubElement(mjcf, "worldbody")
    floor = ET.SubElement(worldbody, "body", name="/worldbody/floor")
    ET.SubElement(floor, "geom", type="plane", pos="0. 0. 0.", size="40. 40. 40.")

    box = ET.SubElement(worldbody, "body", name="/worldbody/box", pos="0. 0. 0.3")
    ET.SubElement(box, "geom", type="box", size="0.2 0.2 0.2", pos="0. 0. 0.")
    ET.SubElement(box, "joint", name="/worldbody/box_joint", type="free")

    return mjcf


@pytest.fixture(scope="session")
def box_plane_usd(box_plane_mjcf: ET.ElementTree):
    """Generate a USD file equivalent to the MJCF box_plane_mjcf fixture.
    Extracts data from the MJCF XML structure to build the USD file.
    """
    # Extract data from MJCF XML structure
    worldbody = box_plane_mjcf.find("worldbody")

    # Floor: body contains a geom with pos and size
    floor_body = worldbody.find("body[@name='/worldbody/floor']")
    floor_geom = floor_body.find("geom[@type='plane']")
    floor_pos_str = floor_geom.get("pos", "0. 0. 0.")
    floor_pos = [float(x) for x in floor_pos_str.split()]
    floor_size_str = floor_geom.get("size", "40. 40. 40.")
    floor_size = [float(x) for x in floor_size_str.split()]

    # Box: body has pos, geom inside has size
    box_body = worldbody.find("body[@name='/worldbody/box']")
    box_pos_str = box_body.get("pos", "0. 0. 0.")
    box_pos = [float(x) for x in box_pos_str.split()]
    box_geom = box_body.find("geom[@type='box']")
    box_size_str = box_geom.get("size", "0.2 0.2 0.2")
    box_size = [float(x) for x in box_size_str.split()]

    # Create temporary USD file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".usda", delete=False) as f:
        usd_file = f.name

    # Create USD stage
    stage = Usd.Stage.CreateNew(usd_file)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Create root prim
    root_prim = stage.DefinePrim("/worldbody", "Xform")
    stage.SetDefaultPrim(root_prim)

    # Create floor plane (fixed, collision-only)
    # In MJCF: plane at floor_pos with size floor_size
    # In USD: Create a plane geometry with CollisionAPI (fixed rigid body)
    floor = UsdGeom.Plane.Define(stage, "/worldbody/floor")
    floor.GetAxisAttr().Set("Z")
    floor.AddTranslateOp().Set(Gf.Vec3d(floor_pos[0], floor_pos[1], floor_pos[2]))
    # MJCF plane size - the third value is typically ignored for plane
    # For USD Plane, we use width and length
    floor.GetWidthAttr().Set(floor_size[0] * 2)  # size[0] * 2
    floor.GetLengthAttr().Set(floor_size[1] * 2)  # size[1] * 2

    # Make it a fixed collision-only rigid body
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    # No RigidBodyAPI means it's kinematic/fixed

    # Create box (free rigid body)
    # In MJCF: box at box_pos with size box_size (half-extent), free joint
    box = UsdGeom.Cube.Define(stage, "/worldbody/box")
    box.AddTranslateOp().Set(Gf.Vec3d(box_pos[0], box_pos[1], box_pos[2]))
    # MJCF size is half-extent, USD size is full edge length
    # So we need to multiply by 2
    box.GetSizeAttr().Set(box_size[0] * 2.0)

    # Make it a free rigid body (no joint means free in USD parser)
    rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(box.GetPrim())
    rigid_body_api.GetKinematicEnabledAttr().Set(False)

    stage.Save()
    return usd_file


@pytest.mark.skipif(not HAS_USD_SUPPORT, reason="USD support not available")
def test_box_plane_mjcf_vs_usd(mjcf_file, box_plane_usd):
    """Test that MJCF and USD scenes produce equivalent Genesis entities."""
    # Create MJCF scene
    mjcf_scene = gs.Scene()
    mjcf_morph = gs.morphs.MJCF(file=mjcf_file, convexify=False)
    mjcf_entity = mjcf_scene.add_entity(mjcf_morph)
    mjcf_scene.build()

    # Create USD scene
    # USD parser now uses material with density=1000.0 to match MuJoCo's default
    usd_scene = gs.Scene()
    usd_morph = gs.morphs.USD(file=box_plane_usd, convexify=False)
    usd_entities = usd_scene.add_stage(usd_morph, vis_mode="collision")
    usd_scene.build()

    compare_scene(mjcf_scene, usd_scene, tol=1e-5)

    # Clean up
    if os.path.exists(mjcf_file):
        os.unlink(mjcf_file)
    if os.path.exists(box_plane_usd):
        os.unlink(box_plane_usd)


# ==================== Prismatic Joint Tests ====================


@pytest.fixture(scope="session")
def prismatic_joint_mjcf():
    """
    Generate an MJCF model for a box with a prismatic (sliding) joint.
    The box can slide along the Z axis.
    """
    mjcf = ET.Element("mujoco", model="prismatic_joint")
    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "joint", armature="0.0")

    worldbody = ET.SubElement(mjcf, "worldbody")
    floor = ET.SubElement(worldbody, "body", name="/worldbody/floor")
    ET.SubElement(floor, "geom", type="plane", pos="0. 0. 0.", size="40. 40. 40.")

    base = ET.SubElement(worldbody, "body", name="/worldbody/base", pos="0. 0. 0.1")
    ET.SubElement(base, "geom", type="box", size="0.1 0.1 0.1", pos="0. 0. 0.")

    box = ET.SubElement(base, "body", name="/worldbody/base/box", pos="0. 0. 0.2")
    ET.SubElement(box, "geom", type="box", size="0.2 0.2 0.2", pos="0. 0. 0.")
    ET.SubElement(
        box,
        "joint",
        name="/worldbody/base/box_joint",
        type="slide",
        axis="0. 0. 1.",
        range="-0.1 0.3",
        stiffness="120.0",
        damping="12.0",
    )
    return mjcf


@pytest.fixture(scope="session")
def prismatic_joint_usd(prismatic_joint_mjcf: ET.ElementTree):
    """Generate a USD file equivalent to the prismatic joint MJCF fixture."""
    worldbody = prismatic_joint_mjcf.find("worldbody")

    # Floor
    floor_body = worldbody.find("body[@name='/worldbody/floor']")
    floor_geom = floor_body.find("geom[@type='plane']")
    floor_pos_str = floor_geom.get("pos")
    floor_pos = [float(x) for x in floor_pos_str.split()]
    floor_size_str = floor_geom.get("size", "40. 40. 40.")
    floor_size = [float(x) for x in floor_size_str.split()]

    # Base
    base_body = worldbody.find("body[@name='/worldbody/base']")
    base_pos_str = base_body.get("pos")
    base_pos = [float(x) for x in base_pos_str.split()]
    base_geom = base_body.find("geom[@type='box']")
    base_size_str = base_geom.get("size")
    base_size = [float(x) for x in base_size_str.split()]

    # Box with prismatic joint
    box_body = base_body.find("body[@name='/worldbody/base/box']")
    box_pos_str = box_body.get("pos")
    box_pos = [float(x) for x in box_pos_str.split()]
    box_geom = box_body.find("geom[@type='box']")
    box_size_str = box_geom.get("size")
    box_size = [float(x) for x in box_size_str.split()]

    # Joint limits
    joint = box_body.find("joint[@name='/worldbody/base/box_joint']")
    range_str = joint.get("range")
    range_vals = [float(x) for x in range_str.split()]
    lower_limit = range_vals[0]
    upper_limit = range_vals[1]

    # Create temporary USD file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".usda", delete=False) as f:
        usd_file = f.name

    # Create USD stage
    stage = Usd.Stage.CreateNew(usd_file)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Create root prim
    root_prim = stage.DefinePrim("/worldbody", "Xform")
    stage.SetDefaultPrim(root_prim)

    # Create floor plane (fixed, collision-only)
    floor = UsdGeom.Plane.Define(stage, "/worldbody/floor")
    floor.GetAxisAttr().Set("Z")
    floor.AddTranslateOp().Set(Gf.Vec3d(floor_pos[0], floor_pos[1], floor_pos[2]))
    floor.GetWidthAttr().Set(floor_size[0] * 2)
    floor.GetLengthAttr().Set(floor_size[1] * 2)
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())

    # Create base (fixed, collision-only)
    base = UsdGeom.Cube.Define(stage, "/worldbody/base")
    UsdPhysics.ArticulationRootAPI.Apply(base.GetPrim())
    base.AddTranslateOp().Set(Gf.Vec3d(base_pos[0], base_pos[1], base_pos[2]))
    base.GetSizeAttr().Set(base_size[0] * 2.0)
    UsdPhysics.CollisionAPI.Apply(base.GetPrim())

    # Create box
    box = UsdGeom.Cube.Define(stage, "/worldbody/base/box")

    box_world_pos = [box_pos[i] for i in range(3)]
    box.AddTranslateOp().Set(Gf.Vec3d(box_world_pos[0], box_world_pos[1], box_world_pos[2]))
    box.GetSizeAttr().Set(box_size[0] * 2.0)
    box_rigid = UsdPhysics.RigidBodyAPI.Apply(box.GetPrim())
    box_rigid.GetKinematicEnabledAttr().Set(False)

    # Create prismatic joint
    joint_prim = UsdPhysics.PrismaticJoint.Define(stage, "/worldbody/base/box_joint")
    joint_prim.CreateBody0Rel().SetTargets([base.GetPrim().GetPath()])
    joint_prim.CreateBody1Rel().SetTargets([box.GetPrim().GetPath()])
    joint_prim.CreateAxisAttr().Set("Z")
    joint_prim.CreateLowerLimitAttr().Set(lower_limit)
    joint_prim.CreateUpperLimitAttr().Set(upper_limit)
    joint_prim.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint_prim.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    # Create drive API
    drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, "linear")
    drive_api.CreateStiffnessAttr().Set(120.0)
    drive_api.CreateDampingAttr().Set(12.0)

    stage.Save()
    return usd_file


@pytest.fixture(scope="session")
def prismatic_joint_mjcf_file(prismatic_joint_mjcf):
    """Create a temporary MJCF file from the prismatic joint fixture."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        mjcf_file = f.name
        ET.ElementTree(prismatic_joint_mjcf).write(mjcf_file, encoding="utf-8", xml_declaration=True)
    return mjcf_file


@pytest.mark.skipif(not HAS_USD_SUPPORT, reason="USD support not available")
def test_prismatic_joint_mjcf_vs_usd(prismatic_joint_mjcf_file, prismatic_joint_usd):
    """Test that MJCF and USD scenes with prismatic joints produce equivalent Genesis entities."""
    # Create MJCF scene
    mjcf_scene = gs.Scene()
    mjcf_morph = gs.morphs.MJCF(file=prismatic_joint_mjcf_file, convexify=False)
    mjcf_entity = mjcf_scene.add_entity(mjcf_morph)
    mjcf_scene.build()

    # Create USD scene
    usd_scene = gs.Scene()
    usd_morph = gs.morphs.USD(file=prismatic_joint_usd, convexify=False)
    usd_entities = usd_scene.add_stage(usd_morph, vis_mode="collision")
    usd_scene.build()

    compare_scene(mjcf_scene, usd_scene, tol=1e-5)

    # Clean up
    if os.path.exists(prismatic_joint_mjcf_file):
        os.unlink(prismatic_joint_mjcf_file)
    if os.path.exists(prismatic_joint_usd):
        os.unlink(prismatic_joint_usd)


# ==================== Revolute Joint Tests ====================


@pytest.fixture(scope="session")
def revolute_joint_mjcf():
    """
    Generate an MJCF model for a box with a revolute (hinge) joint.
    The box can rotate around the Z axis.
    """
    mjcf = ET.Element("mujoco", model="revolute_joint")
    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "joint", armature="0.0")

    worldbody = ET.SubElement(mjcf, "worldbody")
    floor = ET.SubElement(worldbody, "body", name="/worldbody/floor")
    ET.SubElement(floor, "geom", type="plane", pos="0. 0. 0.", size="40. 40. 40.")

    base = ET.SubElement(worldbody, "body", name="/worldbody/base", pos="0. 0. 0.1")
    ET.SubElement(base, "geom", type="box", size="0.1 0.1 0.1", pos="0. 0. 0.")

    box = ET.SubElement(base, "body", name="/worldbody/base/box", pos="0. 0. 0.2")
    ET.SubElement(box, "geom", type="box", size="0.2 0.2 0.2", pos="0. 0. 0.")

    ET.SubElement(
        box,
        "joint",
        name="/worldbody/base/box_joint",
        type="hinge",
        axis="0. 0. 1.",
        range="-45 45",
        stiffness="120.0",
        damping="12.0",
    )
    return mjcf


@pytest.fixture(scope="session")
def revolute_joint_usd(revolute_joint_mjcf: ET.ElementTree):
    """Generate a USD file equivalent to the revolute joint MJCF fixture."""
    worldbody = revolute_joint_mjcf.find("worldbody")

    # Floor
    floor_body = worldbody.find("body[@name='/worldbody/floor']")
    floor_geom = floor_body.find("geom[@type='plane']")
    floor_pos_str = floor_geom.get("pos")
    floor_pos = [float(x) for x in floor_pos_str.split()]
    floor_size_str = floor_geom.get("size", "40. 40. 40.")
    floor_size = [float(x) for x in floor_size_str.split()]

    # Base
    base_body = worldbody.find("body[@name='/worldbody/base']")
    base_pos_str = base_body.get("pos")
    base_pos = [float(x) for x in base_pos_str.split()]
    base_geom = base_body.find("geom[@type='box']")
    base_size_str = base_geom.get("size")
    base_size = [float(x) for x in base_size_str.split()]

    # Box with revolute joint
    box_body = base_body.find("body[@name='/worldbody/base/box']")
    box_pos_str = box_body.get("pos")
    box_pos = [float(x) for x in box_pos_str.split()]
    box_geom = box_body.find("geom[@type='box']")
    box_size_str = box_geom.get("size")
    box_size = [float(x) for x in box_size_str.split()]

    # Joint limits
    joint = box_body.find("joint[@name='/worldbody/base/box_joint']")
    range_str = joint.get("range")
    range_vals = [float(x) for x in range_str.split()]
    lower_limit_deg = range_vals[0]
    upper_limit_deg = range_vals[1]

    # Create temporary USD file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".usda", delete=False) as f:
        usd_file = f.name

    # Create USD stage
    stage = Usd.Stage.CreateNew(usd_file)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Create root prim
    root_prim = stage.DefinePrim("/worldbody", "Xform")
    stage.SetDefaultPrim(root_prim)

    # Create floor plane (fixed, collision-only)
    floor = UsdGeom.Plane.Define(stage, "/worldbody/floor")
    floor.GetAxisAttr().Set("Z")
    floor.AddTranslateOp().Set(Gf.Vec3d(floor_pos[0], floor_pos[1], floor_pos[2]))
    floor.GetWidthAttr().Set(floor_size[0] * 2)
    floor.GetLengthAttr().Set(floor_size[1] * 2)
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())

    # Create base (fixed, collision-only)
    base = UsdGeom.Cube.Define(stage, "/worldbody/base")
    UsdPhysics.ArticulationRootAPI.Apply(base.GetPrim())
    base.AddTranslateOp().Set(Gf.Vec3d(base_pos[0], base_pos[1], base_pos[2]))
    base.GetSizeAttr().Set(base_size[0] * 2.0)
    UsdPhysics.CollisionAPI.Apply(base.GetPrim())

    # Create box
    box = UsdGeom.Cube.Define(stage, "/worldbody/base/box")

    box_world_pos = [box_pos[i] for i in range(3)]
    box.AddTranslateOp().Set(Gf.Vec3d(box_world_pos[0], box_world_pos[1], box_world_pos[2]))
    box.GetSizeAttr().Set(box_size[0] * 2.0)
    box_rigid = UsdPhysics.RigidBodyAPI.Apply(box.GetPrim())
    box_rigid.GetKinematicEnabledAttr().Set(False)

    # Create revolute joint
    joint_prim = UsdPhysics.RevoluteJoint.Define(stage, "/worldbody/base/box_joint")
    joint_prim.CreateBody0Rel().SetTargets([base.GetPrim().GetPath()])
    joint_prim.CreateBody1Rel().SetTargets([box.GetPrim().GetPath()])
    joint_prim.CreateAxisAttr().Set("Z")
    joint_prim.CreateLowerLimitAttr().Set(lower_limit_deg)
    joint_prim.CreateUpperLimitAttr().Set(upper_limit_deg)
    joint_prim.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint_prim.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    # Create drive API (use "angular" for revolute joints)
    drive_api = UsdPhysics.DriveAPI.Apply(joint_prim.GetPrim(), "angular")
    drive_api.CreateStiffnessAttr().Set(120.0)
    drive_api.CreateDampingAttr().Set(12.0)

    stage.Save()
    return usd_file


@pytest.fixture(scope="session")
def revolute_joint_mjcf_file(revolute_joint_mjcf):
    """Create a temporary MJCF file from the revolute joint fixture."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        mjcf_file = f.name
        ET.ElementTree(revolute_joint_mjcf).write(mjcf_file, encoding="utf-8", xml_declaration=True)
    return mjcf_file


@pytest.mark.skipif(not HAS_USD_SUPPORT, reason="USD support not available")
def test_revolute_joint_mjcf_vs_usd(revolute_joint_mjcf_file, revolute_joint_usd):
    """Test that MJCF and USD scenes with revolute joints produce equivalent Genesis entities."""
    # Create MJCF scene
    mjcf_scene = gs.Scene()
    mjcf_morph = gs.morphs.MJCF(file=revolute_joint_mjcf_file, convexify=False)
    mjcf_entity = mjcf_scene.add_entity(mjcf_morph)
    mjcf_scene.build()

    # Create USD scene
    usd_scene = gs.Scene()
    usd_morph = gs.morphs.USD(file=revolute_joint_usd, convexify=False)
    usd_entities = usd_scene.add_stage(usd_morph, vis_mode="collision")
    usd_scene.build()

    compare_scene(mjcf_scene, usd_scene, tol=1e-5)

    # Clean up
    if os.path.exists(revolute_joint_mjcf_file):
        os.unlink(revolute_joint_mjcf_file)
    if os.path.exists(revolute_joint_usd):
        os.unlink(revolute_joint_usd)


# ==================== Spherical Joint Tests ====================


@pytest.fixture(scope="session")
def spherical_joint_mjcf():
    """
    Generate an MJCF model for a box with a spherical (ball) joint.
    The box can rotate freely around all three axes.
    """
    mjcf = ET.Element("mujoco", model="spherical_joint")
    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "joint", armature="0.0")

    worldbody = ET.SubElement(mjcf, "worldbody")
    floor = ET.SubElement(worldbody, "body", name="/worldbody/floor")
    ET.SubElement(floor, "geom", type="plane", pos="0. 0. 0.", size="40. 40. 40.")

    base = ET.SubElement(worldbody, "body", name="/worldbody/base", pos="0. 0. 0.1")
    ET.SubElement(base, "geom", type="box", size="0.1 0.1 0.1", pos="0. 0. 0.")

    box = ET.SubElement(base, "body", name="/worldbody/base/box", pos="0. 0. 0.2")
    ET.SubElement(box, "geom", type="box", size="0.2 0.2 0.2", pos="0. 0. 0.")
    # Spherical joint (ball) - no limits, can rotate freely
    ET.SubElement(box, "joint", name="/worldbody/base/box_joint", type="ball")
    return mjcf


@pytest.fixture(scope="session")
def spherical_joint_usd(spherical_joint_mjcf: ET.ElementTree):
    """Generate a USD file equivalent to the spherical joint MJCF fixture."""
    worldbody = spherical_joint_mjcf.find("worldbody")

    # Floor
    floor_body = worldbody.find("body[@name='/worldbody/floor']")
    floor_geom = floor_body.find("geom[@type='plane']")
    floor_pos_str = floor_geom.get("pos")
    floor_pos = [float(x) for x in floor_pos_str.split()]
    floor_size_str = floor_geom.get("size", "40. 40. 40.")
    floor_size = [float(x) for x in floor_size_str.split()]

    # Base
    base_body = worldbody.find("body[@name='/worldbody/base']")
    base_pos_str = base_body.get("pos")
    base_pos = [float(x) for x in base_pos_str.split()]
    base_geom = base_body.find("geom[@type='box']")
    base_size_str = base_geom.get("size")
    base_size = [float(x) for x in base_size_str.split()]

    # Box with spherical joint
    box_body = base_body.find("body[@name='/worldbody/base/box']")
    box_pos_str = box_body.get("pos")
    box_pos = [float(x) for x in box_pos_str.split()]
    box_geom = box_body.find("geom[@type='box']")
    box_size_str = box_geom.get("size")
    box_size = [float(x) for x in box_size_str.split()]

    # Create temporary USD file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".usda", delete=False) as f:
        usd_file = f.name

    # Create USD stage
    stage = Usd.Stage.CreateNew(usd_file)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Create root prim
    root_prim = stage.DefinePrim("/worldbody", "Xform")
    stage.SetDefaultPrim(root_prim)

    # Create floor plane (fixed, collision-only)
    floor = UsdGeom.Plane.Define(stage, "/worldbody/floor")
    floor.GetAxisAttr().Set("Z")
    floor.AddTranslateOp().Set(Gf.Vec3d(floor_pos[0], floor_pos[1], floor_pos[2]))
    floor.GetWidthAttr().Set(floor_size[0] * 2)
    floor.GetLengthAttr().Set(floor_size[1] * 2)
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())

    # Create base (fixed, collision-only)
    base = UsdGeom.Cube.Define(stage, "/worldbody/base")
    UsdPhysics.ArticulationRootAPI.Apply(base.GetPrim())
    base.AddTranslateOp().Set(Gf.Vec3d(base_pos[0], base_pos[1], base_pos[2]))
    base.GetSizeAttr().Set(base_size[0] * 2.0)
    UsdPhysics.CollisionAPI.Apply(base.GetPrim())

    # Create box
    box = UsdGeom.Cube.Define(stage, "/worldbody/base/box")

    box_world_pos = [box_pos[i] for i in range(3)]
    box.AddTranslateOp().Set(Gf.Vec3d(box_world_pos[0], box_world_pos[1], box_world_pos[2]))
    box.GetSizeAttr().Set(box_size[0] * 2.0)
    box_rigid = UsdPhysics.RigidBodyAPI.Apply(box.GetPrim())
    box_rigid.GetKinematicEnabledAttr().Set(False)

    # Create spherical joint
    joint_prim = UsdPhysics.SphericalJoint.Define(stage, "/worldbody/base/box_joint")
    joint_prim.CreateBody0Rel().SetTargets([base.GetPrim().GetPath()])
    joint_prim.CreateBody1Rel().SetTargets([box.GetPrim().GetPath()])
    joint_prim.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint_prim.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    stage.Save()
    return usd_file


@pytest.fixture(scope="session")
def spherical_joint_mjcf_file(spherical_joint_mjcf):
    """Create a temporary MJCF file from the spherical joint fixture."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        mjcf_file = f.name
        ET.ElementTree(spherical_joint_mjcf).write(mjcf_file, encoding="utf-8", xml_declaration=True)
    return mjcf_file


@pytest.mark.skipif(not HAS_USD_SUPPORT, reason="USD support not available")
def test_spherical_joint_mjcf_vs_usd(spherical_joint_mjcf_file, spherical_joint_usd):
    """Test that MJCF and USD scenes with spherical joints produce equivalent Genesis entities."""
    # Create MJCF scene
    mjcf_scene = gs.Scene()
    mjcf_morph = gs.morphs.MJCF(file=spherical_joint_mjcf_file, convexify=False)
    mjcf_entity = mjcf_scene.add_entity(mjcf_morph)
    mjcf_scene.build()

    # Create USD scene
    usd_scene = gs.Scene()
    usd_morph = gs.morphs.USD(file=spherical_joint_usd, convexify=False)
    usd_entities = usd_scene.add_stage(usd_morph, vis_mode="collision")
    usd_scene.build()

    compare_scene(mjcf_scene, usd_scene, tol=1e-5)

    # Clean up
    if os.path.exists(spherical_joint_mjcf_file):
        os.unlink(spherical_joint_mjcf_file)
    if os.path.exists(spherical_joint_usd):
        os.unlink(spherical_joint_usd)
