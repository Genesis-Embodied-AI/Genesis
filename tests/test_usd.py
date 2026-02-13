"""
Test USD parsing and comparison with compared scenes.

This module tests that USD files can be parsed correctly and that scenes
loaded from USD files match equivalent scenes loaded from compared files.
"""

import os
import time
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import genesis as gs
import genesis.utils.geom as gu

from .utils import assert_allclose, get_hf_dataset
from .test_mesh import check_gs_meshes, check_gs_surfaces

# Check for USD support
try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    from genesis.utils.usd import UsdContext, HAS_OMNIVERSE_KIT_SUPPORT

    HAS_USD_SUPPORT = True
except ImportError as e:
    HAS_USD_SUPPORT = False
    HAS_OMNIVERSE_KIT_SUPPORT = False


# Conversion from .usd to .glb significantly affects precision
USD_COLOR_TOL = 1e-07
USD_NORMALS_TOL = 1e-02


def to_array(s: str) -> np.ndarray:
    """Convert a string of space-separated floats to a numpy array."""
    return np.array([float(x) for x in s.split()])


def compare_links(compared_links, usd_links, tol):
    """Compare links between two scenes."""
    # Check number of links
    assert len(compared_links) == len(usd_links)

    # Create dictionaries keyed by link name for comparison
    compared_links_by_name = {link.name: link for link in compared_links}
    usd_links_by_name = {link.name: link for link in usd_links}

    # Create index to name mappings for parent comparison
    compared_idx_to_name = {i: link.name for i, link in enumerate(compared_links)}
    usd_idx_to_name = {i: link.name for i, link in enumerate(usd_links)}

    # Check that we have matching link names
    compared_link_names = set(compared_links_by_name.keys())
    usd_link_names = set(usd_links_by_name.keys())
    assert compared_link_names == usd_link_names

    # Compare all link properties by name
    for link_name in sorted(compared_link_names):
        compared_link = compared_links_by_name[link_name]
        usd_link = usd_links_by_name[link_name]
        err_msg = f"Properties mismatched for link {link_name}"

        # Compare link properties
        assert_allclose(compared_link.pos, usd_link.pos, tol=tol, err_msg=err_msg)
        assert_allclose(compared_link.quat, usd_link.quat, tol=tol, err_msg=err_msg)
        assert compared_link.is_fixed == usd_link.is_fixed, err_msg
        assert len(compared_link.geoms) == len(usd_link.geoms), err_msg
        assert compared_link.n_joints == usd_link.n_joints, err_msg
        assert len(compared_link.vgeoms) == len(usd_link.vgeoms), err_msg

        # Compare parent link by name (mapping indices to names)
        compared_parent_idx = compared_link.parent_idx
        usd_parent_idx = usd_link.parent_idx
        if compared_parent_idx == -1:
            compared_parent_name = None
        else:
            compared_parent_name = compared_idx_to_name.get(compared_parent_idx, f"<unknown idx {compared_parent_idx}>")
        if usd_parent_idx == -1:
            usd_parent_name = None
        else:
            usd_parent_name = usd_idx_to_name.get(usd_parent_idx, f"<unknown idx {usd_parent_idx}>")
        assert compared_parent_name == usd_parent_name, err_msg

        # Compare inertial properties if available
        assert_allclose(compared_link.inertial_pos, usd_link.inertial_pos, tol=tol, err_msg=err_msg)
        assert_allclose(compared_link.inertial_quat, usd_link.inertial_quat, tol=tol, err_msg=err_msg)

        # Skip mass and inertia checks for fixed links - they're not used in simulation
        if not compared_link.is_fixed:
            assert_allclose(compared_link.inertial_mass, usd_link.inertial_mass, atol=tol, err_msg=err_msg)
            assert_allclose(compared_link.inertial_i, usd_link.inertial_i, atol=tol, err_msg=err_msg)


def compare_joints(compared_joints, usd_joints, tol):
    """Compare joints between two scenes."""
    # Check number of joints
    assert len(compared_joints) == len(usd_joints)

    # Create dictionaries keyed by joint name for comparison
    compared_joints_by_name = {joint.name: joint for joint in compared_joints}
    usd_joints_by_name = {joint.name: joint for joint in usd_joints}

    # Check that we have matching joint names
    compared_joint_names = set(compared_joints_by_name.keys())
    usd_joint_names = set(usd_joints_by_name.keys())
    assert compared_joint_names == usd_joint_names

    # Compare all joint properties by name
    for joint_name in sorted(compared_joint_names):
        compared_joint = compared_joints_by_name[joint_name]
        usd_joint = usd_joints_by_name[joint_name]

        # Compare joint properties
        assert compared_joint.type == usd_joint.type
        err_msg = f"Properties mismatched for joint type {compared_joint.type}"

        assert_allclose(compared_joint.pos, usd_joint.pos, tol=tol, err_msg=err_msg)
        assert_allclose(compared_joint.quat, usd_joint.quat, tol=tol, err_msg=err_msg)
        assert compared_joint.n_qs == usd_joint.n_qs, err_msg
        assert compared_joint.n_dofs == usd_joint.n_dofs, err_msg

        # Compare initial qpos
        assert_allclose(compared_joint.init_qpos, usd_joint.init_qpos, tol=tol, err_msg=err_msg)

        # Skip mass/inertia-dependent property checks for fixed joints - they're not used in simulation
        if compared_joint.type != gs.JOINT_TYPE.FIXED:
            # Compare dof limits
            assert_allclose(compared_joint.dofs_limit, usd_joint.dofs_limit, tol=tol, err_msg=err_msg)

            # Compare dof motion properties
            assert_allclose(compared_joint.dofs_motion_ang, usd_joint.dofs_motion_ang, tol=tol, err_msg=err_msg)
            assert_allclose(compared_joint.dofs_motion_vel, usd_joint.dofs_motion_vel, tol=tol, err_msg=err_msg)
            assert_allclose(compared_joint.dofs_frictionloss, usd_joint.dofs_frictionloss, tol=tol, err_msg=err_msg)
            assert_allclose(compared_joint.dofs_stiffness, usd_joint.dofs_stiffness, tol=tol, err_msg=err_msg)
            assert_allclose(compared_joint.dofs_frictionloss, usd_joint.dofs_frictionloss, tol=tol, err_msg=err_msg)
            assert_allclose(compared_joint.dofs_force_range, usd_joint.dofs_force_range, tol=tol, err_msg=err_msg)
            assert_allclose(compared_joint.dofs_damping, usd_joint.dofs_damping, tol=tol, err_msg=err_msg)
            assert_allclose(compared_joint.dofs_armature, usd_joint.dofs_armature, tol=tol, err_msg=err_msg)

            # Compare dof control properties
            assert_allclose(compared_joint.dofs_kp, usd_joint.dofs_kp, tol=tol, err_msg=err_msg)
            assert_allclose(compared_joint.dofs_kv, usd_joint.dofs_kv, tol=tol, err_msg=err_msg)
            assert_allclose(compared_joint.dofs_force_range, usd_joint.dofs_force_range, tol=tol, err_msg=err_msg)


def compare_geoms(compared_geoms, usd_geoms, tol):
    """Compare geoms between two scenes."""
    assert len(compared_geoms) == len(usd_geoms)

    # Sort geoms by link name for consistent comparison
    compared_geoms_sorted = sorted(compared_geoms, key=lambda g: (g.link.name, g.idx))
    usd_geoms_sorted = sorted(usd_geoms, key=lambda g: (g.link.name, g.idx))

    for compared_geom, usd_geom in zip(compared_geoms_sorted, usd_geoms_sorted):
        assert compared_geom.type == usd_geom.type
        err_msg = f"Properties mismatched for geom type {compared_geom.type}"

        assert_allclose(compared_geom.init_pos, usd_geom.init_pos, tol=tol, err_msg=err_msg)
        assert_allclose(compared_geom.init_quat, usd_geom.init_quat, tol=tol, err_msg=err_msg)
        assert_allclose(compared_geom.get_AABB(), usd_geom.get_AABB(), tol=tol, err_msg=err_msg)


def compare_vgeoms(compared_vgeoms, usd_vgeoms, tol):
    """Compare visual geoms between two scenes."""
    assert len(compared_vgeoms) == len(usd_vgeoms)

    # Sort geoms by link name for consistent comparison
    compared_vgeoms_sorted = sorted(compared_vgeoms, key=lambda g: g.vmesh.metadata["name"])
    usd_vgeoms_sorted = sorted(usd_vgeoms, key=lambda g: g.vmesh.metadata["name"].split("/")[-1])

    for compared_vgeom, usd_vgeom in zip(compared_vgeoms_sorted, usd_vgeoms_sorted):
        compared_vgeom_pos, compared_vgeom_quat = gu.transform_pos_quat_by_trans_quat(
            compared_vgeom.init_pos, compared_vgeom.init_quat, compared_vgeom.link.pos, compared_vgeom.link.quat
        )
        usd_vgeom_pos, usd_vgeom_quat = gu.transform_pos_quat_by_trans_quat(
            usd_vgeom.init_pos, usd_vgeom.init_quat, usd_vgeom.link.pos, usd_vgeom.link.quat
        )
        compared_vgeom_T = gu.trans_quat_to_T(compared_vgeom_pos, compared_vgeom_quat)
        usd_vgeom_T = gu.trans_quat_to_T(usd_vgeom_pos, usd_vgeom_quat)

        compared_vgeom_mesh = compared_vgeom.vmesh.copy()
        usd_vgeom_mesh = usd_vgeom.vmesh.copy()
        mesh_name = usd_vgeom_mesh.metadata["name"]
        compared_vgeom_mesh.apply_transform(compared_vgeom_T)
        usd_vgeom_mesh.apply_transform(usd_vgeom_T)
        check_gs_meshes(compared_vgeom_mesh, usd_vgeom_mesh, mesh_name, tol, USD_NORMALS_TOL)

        compared_vgeom_surface = compared_vgeom_mesh.surface
        usd_vgeom_surface = usd_vgeom_mesh.surface
        check_gs_surfaces(compared_vgeom_surface, usd_vgeom_surface, mesh_name)


def compare_scene(compared_scene: gs.Scene, usd_scene: gs.Scene, tol: float):
    """Compare structure and data between compared scene and USD scene."""
    compared_entities = compared_scene.entities
    usd_entities = usd_scene.entities

    compared_geoms = [geom for entity in compared_entities for geom in entity.geoms]
    usd_geoms = [geom for entity in usd_entities for geom in entity.geoms]
    compare_geoms(compared_geoms, usd_geoms, tol=tol)

    compared_joints = [joint for entity in compared_entities for joint in entity.joints]
    usd_joints = [joint for entity in usd_entities for joint in entity.joints]
    compare_joints(compared_joints, usd_joints, tol=tol)

    compared_links = [link for entity in compared_entities for link in entity.links]
    usd_links = [link for entity in usd_entities for link in entity.links]
    compare_links(compared_links, usd_links, tol=tol)


def compare_mesh_scene(compared_scene: gs.Scene, usd_scene: gs.Scene, tol: float):
    """Compare mesh data between mesh scene and USD scene."""
    compared_entities = compared_scene.entities
    usd_entities = usd_scene.entities
    compared_vgeoms = [vgeom for entity in compared_entities for vgeom in entity.vgeoms]
    usd_vgeoms = [vgeom for entity in usd_entities for vgeom in entity.vgeoms]
    compare_vgeoms(compared_vgeoms, usd_vgeoms, tol=tol)


def build_mjcf_scene(xml_path: str, scale: float):
    """Build a MJCF scene from its file path."""
    # Create MJCF scene
    mjcf_scene = gs.Scene()

    mjcf_scene.add_entity(
        gs.morphs.MJCF(
            file=xml_path,
            scale=scale,
            convexify=False,
        ),
        material=gs.materials.Rigid(
            rho=1000.0,
        ),
    )

    mjcf_scene.build()
    return mjcf_scene


def build_usd_scene(
    usd_file: str,
    scale: float,
    vis_mode: str = "collision",
    is_stage: bool = True,
    fixed: bool = False,
):
    """Build a USD scene from its file path."""
    # Create USD scene
    scene = gs.Scene()

    kwargs = dict(
        morph=gs.morphs.USD(
            usd_ctx=UsdContext(
                usd_file,
                use_bake_cache=False,
            ),
            scale=scale,
            fixed=fixed,
            convexify=False,
        ),
        material=gs.materials.Rigid(
            rho=1000.0,
        ),
        vis_mode=vis_mode,
    )

    if is_stage:
        scene.add_stage(**kwargs)
    else:
        scene.add_entity(**kwargs)

    # Note that it is necessary to build the scene because spatial inertia of some geometries may not be specified.
    # In such a case, it will be estimated from the geometry during build (RigidLink._build to be specific).
    scene.build()

    return scene


def build_mesh_scene(mesh_file: str, scale: float):
    """Build a mesh scene from its file path."""
    mesh_scene = gs.Scene()
    mesh_morph = gs.morphs.Mesh(
        file=mesh_file,
        scale=scale,
        euler=(-90, 0, 0),
        group_by_material=False,
        convexify=False,
    )
    mesh_scene.add_entity(
        mesh_morph,
        material=gs.materials.Rigid(
            rho=1000.0,
        ),
    )
    mesh_scene.build()
    return mesh_scene


@pytest.fixture
def xml_path(request, tmp_path, model_name):
    """Create a temporary MJCF/XML file from the fixture."""
    mjcf = request.getfixturevalue(model_name)
    xml_tree = ET.ElementTree(mjcf)
    file_name = f"{model_name}.xml"
    file_path = str(tmp_path / file_name)
    xml_tree.write(file_path, encoding="utf-8", xml_declaration=True)
    return file_path


# ==================== Primitive Tests ====================


@pytest.fixture(scope="session")
def all_primitives_mjcf():
    """Generate an MJCF model with various geometric primitives on a plane."""
    mjcf = ET.Element("mujoco", model="primitives")
    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "joint", armature="0.0")

    worldbody = ET.SubElement(mjcf, "worldbody")
    floor = ET.SubElement(worldbody, "body", name="/worldbody/floor")
    ET.SubElement(floor, "geom", type="plane", pos="0. 0. 0.", size="40. 40. 40.")

    # Box
    box = ET.SubElement(worldbody, "body", name="/worldbody/box", pos="-0.6 0. 0.3")
    ET.SubElement(box, "geom", type="box", size="0.2 0.2 0.2", pos="0. 0. 0.")
    ET.SubElement(box, "joint", name="/worldbody/box_joint", type="free")

    # Cylinder
    cylinder = ET.SubElement(worldbody, "body", name="/worldbody/cylinder", pos="-0.2 0. 0.3")
    ET.SubElement(cylinder, "geom", type="cylinder", size="0.15 0.2", pos="0. 0. 0.")
    ET.SubElement(cylinder, "joint", name="/worldbody/cylinder_joint", type="free")

    # Capsule
    capsule = ET.SubElement(worldbody, "body", name="/worldbody/capsule", pos="0.2 0. 0.3")
    ET.SubElement(capsule, "geom", type="capsule", size="0.15 0.2", pos="0. 0. 0.")
    ET.SubElement(capsule, "joint", name="/worldbody/capsule_joint", type="free")

    # Sphere
    sphere = ET.SubElement(worldbody, "body", name="/worldbody/sphere", pos="0.6 0. 0.3")
    ET.SubElement(sphere, "geom", type="sphere", size="0.2", pos="0. 0. 0.")
    ET.SubElement(sphere, "joint", name="/worldbody/sphere_joint", type="free")

    return mjcf


@pytest.fixture(scope="session")
def all_primitives_usd(asset_tmp_path, all_primitives_mjcf: ET.ElementTree):
    """Generate a USD file equivalent to the MJCF all_primitives_mjcf fixture."""
    # Extract data from MJCF XML structure
    worldbody = all_primitives_mjcf.find("worldbody")

    # Floor: body contains a geom with pos and size
    floor_body = worldbody.find("body[@name='/worldbody/floor']")
    floor_geom = floor_body.find("geom[@type='plane']")
    floor_pos_str = floor_geom.get("pos", "0. 0. 0.")
    floor_pos = to_array(floor_pos_str)
    floor_size = to_array(floor_geom.get("size", "40. 40. 40."))

    # Box: body has pos, geom inside has size
    box_body = worldbody.find("body[@name='/worldbody/box']")
    box_pos_str = box_body.get("pos", "0. 0. 0.")
    box_pos = to_array(box_pos_str)
    box_geom = box_body.find("geom[@type='box']")
    box_size_str = box_geom.get("size", "0.2 0.2 0.2")
    box_size = to_array(box_size_str)

    # Cylinder: body has pos, geom has size (radius, half-height)
    cylinder_body = worldbody.find("body[@name='/worldbody/cylinder']")
    cylinder_pos_str = cylinder_body.get("pos", "0. 0. 0.")
    cylinder_pos = to_array(cylinder_pos_str)
    cylinder_geom = cylinder_body.find("geom[@type='cylinder']")
    cylinder_size_str = cylinder_geom.get("size", "0.15 0.2")
    cylinder_size = to_array(cylinder_size_str)
    cylinder_radius = cylinder_size[0]
    cylinder_half_height = cylinder_size[1]

    # Capsule: body has pos, geom has size (radius, half-height)
    capsule_body = worldbody.find("body[@name='/worldbody/capsule']")
    capsule_pos_str = capsule_body.get("pos", "0. 0. 0.")
    capsule_pos = to_array(capsule_pos_str)
    capsule_geom = capsule_body.find("geom[@type='capsule']")
    capsule_size_str = capsule_geom.get("size", "0.15 0.2")
    capsule_size = to_array(capsule_size_str)
    capsule_radius = capsule_size[0]
    capsule_half_height = capsule_size[1]

    # Sphere: body has pos, geom has size (radius)
    sphere_body = worldbody.find("body[@name='/worldbody/sphere']")
    sphere_pos_str = sphere_body.get("pos", "0. 0. 0.")
    sphere_pos = to_array(sphere_pos_str)
    sphere_geom = sphere_body.find("geom[@type='sphere']")
    sphere_size_str = sphere_geom.get("size", "0.2")
    sphere_radius = float(sphere_size_str) if isinstance(sphere_size_str, str) else sphere_size_str[0]

    # Create temporary USD file
    usd_file = str(asset_tmp_path / "all_primitives.usda")

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
    box_rigid = UsdPhysics.RigidBodyAPI.Apply(box.GetPrim())
    box_rigid.GetKinematicEnabledAttr().Set(False)

    # Create cylinder (free rigid body)
    # In MJCF: cylinder size is (radius, half-height)
    # In USD: cylinder has radius and height (full height)
    cylinder = UsdGeom.Cylinder.Define(stage, "/worldbody/cylinder")
    cylinder.AddTranslateOp().Set(Gf.Vec3d(cylinder_pos[0], cylinder_pos[1], cylinder_pos[2]))
    cylinder.GetRadiusAttr().Set(cylinder_radius)
    cylinder.GetHeightAttr().Set(cylinder_half_height * 2.0)  # Convert half-height to full height
    cylinder.GetAxisAttr().Set("Z")
    cylinder_rigid = UsdPhysics.RigidBodyAPI.Apply(cylinder.GetPrim())
    cylinder_rigid.GetKinematicEnabledAttr().Set(False)

    # Create capsule (free rigid body)
    # In MJCF: capsule size is (radius, half-height)
    # In USD: capsule has radius and height (full height)
    capsule = UsdGeom.Capsule.Define(stage, "/worldbody/capsule")
    capsule.AddTranslateOp().Set(Gf.Vec3d(capsule_pos[0], capsule_pos[1], capsule_pos[2]))
    capsule.GetRadiusAttr().Set(capsule_radius)
    capsule.GetHeightAttr().Set(capsule_half_height * 2.0)  # Convert half-height to full height
    capsule.GetAxisAttr().Set("Z")
    capsule_rigid = UsdPhysics.RigidBodyAPI.Apply(capsule.GetPrim())
    capsule_rigid.GetKinematicEnabledAttr().Set(False)

    # Create sphere (free rigid body)
    # In MJCF: sphere size is radius
    # In USD: sphere has radius
    sphere = UsdGeom.Sphere.Define(stage, "/worldbody/sphere")
    sphere.AddTranslateOp().Set(Gf.Vec3d(sphere_pos[0], sphere_pos[1], sphere_pos[2]))
    sphere.GetRadiusAttr().Set(sphere_radius)
    sphere_rigid = UsdPhysics.RigidBodyAPI.Apply(sphere.GetPrim())
    sphere_rigid.GetKinematicEnabledAttr().Set(False)

    stage.Save()

    return usd_file


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["all_primitives_mjcf"])
@pytest.mark.parametrize("scale", [1.0, 2.0])
@pytest.mark.skipif(not HAS_USD_SUPPORT, reason="USD support not available")
def test_primitives_mjcf_vs_usd(xml_path, all_primitives_usd, scale, tol):
    """Test that MJCF and USD scenes produce equivalent Genesis entities."""
    mjcf_scene = build_mjcf_scene(xml_path, scale=scale)
    usd_scene = build_usd_scene(all_primitives_usd, scale=scale)
    compare_scene(mjcf_scene, usd_scene, tol=tol)


# ==================== Joint Tests ====================


@pytest.fixture(scope="session")
def all_joints_mjcf():
    """Generate an MJCF model with all joint types: prismatic, revolute, spherical, fixed, and free."""
    mjcf = ET.Element("mujoco", model="all_joints")
    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "joint", armature="0.0")

    worldbody = ET.SubElement(mjcf, "worldbody")
    floor = ET.SubElement(worldbody, "body", name="/worldbody/floor")
    ET.SubElement(floor, "geom", type="plane", pos="0. 0. 0.", size="40. 40. 40.")

    base = ET.SubElement(worldbody, "body", name="/worldbody/base", pos="0. 0. 0.1")
    ET.SubElement(base, "geom", type="box", size="0.1 0.1 0.1", pos="0. 0. 0.")

    # Prismatic joint branch
    prismatic_box = ET.SubElement(base, "body", name="/worldbody/base/prismatic_box", pos="-0.5 0. 0.2")
    ET.SubElement(prismatic_box, "geom", type="box", size="0.2 0.2 0.2", pos="0. 0. 0.")
    ET.SubElement(
        prismatic_box,
        "joint",
        name="/worldbody/base/prismatic_box_joint",
        type="slide",
        axis="0. 0. 1.",
        range="-0.1 0.4",
        stiffness="50.0",
        damping="5.0",
    )

    # Revolute joint branch
    # Add actuator for PD controller (maps to dofs_kp and dofs_kv)
    # The parser uses: dofs_kp = -gear * biasprm[1] * scale^3
    # So to get dofs_kp=120.0, we need biasprm[1] = -120.0 (with gear=1, scale=1)
    actuator = ET.SubElement(mjcf, "actuator")
    revolute_box = ET.SubElement(base, "body", name="/worldbody/base/revolute_box", pos="0. 0. 0.2")
    ET.SubElement(revolute_box, "geom", type="box", size="0.2 0.2 0.2", pos="0. 0. 0.")
    ET.SubElement(
        revolute_box,
        "joint",
        name="/worldbody/base/revolute_box_joint",
        type="hinge",
        axis="0. 0. 1.",
        range="-45 45",
        stiffness="50.0",
        damping="5.0",
    )

    # Spherical joint branch
    spherical_box = ET.SubElement(base, "body", name="/worldbody/base/spherical_box", pos="0.5 0. 0.2")
    ET.SubElement(spherical_box, "geom", type="box", size="0.2 0.2 0.2", pos="0. 0. 0.")
    ET.SubElement(spherical_box, "joint", name="/worldbody/base/spherical_box_joint", type="ball")

    # Fixed joint branch (no joint element means fixed in MJCF)
    fixed_box = ET.SubElement(base, "body", name="/worldbody/base/fixed_box", pos="-0.5 0.5 0.2")
    ET.SubElement(fixed_box, "geom", type="box", size="0.2 0.2 0.2", pos="0. 0. 0.")
    # No joint element = fixed joint

    # Free joint branch (must be at top level in MJCF - directly under worldbody)
    free_box = ET.SubElement(worldbody, "body", name="/worldbody/free_box", pos="0.5 0.5 0.3")
    ET.SubElement(free_box, "geom", type="box", size="0.2 0.2 0.2", pos="0. 0. 0.")
    ET.SubElement(free_box, "joint", name="/worldbody/free_box_joint", type="free")

    # Add actuators for PD controllers (prismatic and revolute only)
    actuator = ET.SubElement(mjcf, "actuator")
    ET.SubElement(
        actuator,
        "general",
        name="/worldbody/base/prismatic_box_joint_actuator",
        joint="/worldbody/base/prismatic_box_joint",
        biastype="affine",
        gainprm="120.0 0 0",  # gainprm[0] must equal -biasprm[1] to avoid warning
        biasprm="0 -120.0 -12.0",  # biasprm format: [b0, b1, b2] where b1=kp, b2=kv (negated)
    )
    ET.SubElement(
        actuator,
        "general",
        name="/worldbody/base/revolute_box_joint_actuator",
        joint="/worldbody/base/revolute_box_joint",
        biastype="affine",
        gainprm="120.0 0 0",
        biasprm="0 -120.0 -12.0",
    )

    return mjcf


@pytest.fixture(scope="session")
def all_joints_usd(asset_tmp_path, all_joints_mjcf: ET.ElementTree):
    """Generate a USD file equivalent to the all joints MJCF fixture."""
    worldbody = all_joints_mjcf.find("worldbody")

    # Floor
    floor_body = worldbody.find("body[@name='/worldbody/floor']")
    floor_geom = floor_body.find("geom[@type='plane']")
    floor_pos_str = floor_geom.get("pos")
    floor_pos = to_array(floor_pos_str)
    floor_size = to_array(floor_geom.get("size", "40. 40. 40."))

    # Base
    base_body = worldbody.find("body[@name='/worldbody/base']")
    base_pos_str = base_body.get("pos")
    base_pos = to_array(base_pos_str)
    base_geom = base_body.find("geom[@type='box']")
    base_size_str = base_geom.get("size")
    base_size = to_array(base_size_str)

    # Prismatic box
    prismatic_box_body = base_body.find("body[@name='/worldbody/base/prismatic_box']")
    prismatic_box_pos_str = prismatic_box_body.get("pos")
    prismatic_box_pos = to_array(prismatic_box_pos_str)
    prismatic_box_geom = prismatic_box_body.find("geom[@type='box']")
    prismatic_box_size = to_array(prismatic_box_geom.get("size"))
    prismatic_joint = prismatic_box_body.find("joint[@name='/worldbody/base/prismatic_box_joint']")
    prismatic_range = to_array(prismatic_joint.get("range"))

    # Revolute box
    revolute_box_body = base_body.find("body[@name='/worldbody/base/revolute_box']")
    revolute_box_pos_str = revolute_box_body.get("pos")
    revolute_box_pos = to_array(revolute_box_pos_str)
    revolute_box_geom = revolute_box_body.find("geom[@type='box']")
    revolute_box_size = to_array(revolute_box_geom.get("size"))
    revolute_joint = revolute_box_body.find("joint[@name='/worldbody/base/revolute_box_joint']")
    revolute_range = to_array(revolute_joint.get("range"))

    # Spherical box
    spherical_box_body = base_body.find("body[@name='/worldbody/base/spherical_box']")
    spherical_box_pos_str = spherical_box_body.get("pos")
    spherical_box_pos = to_array(spherical_box_pos_str)
    spherical_box_geom = spherical_box_body.find("geom[@type='box']")
    spherical_box_size = to_array(spherical_box_geom.get("size"))

    # Fixed box (no joint in MJCF means fixed)
    fixed_box_body = base_body.find("body[@name='/worldbody/base/fixed_box']")
    fixed_box_pos_str = fixed_box_body.get("pos")
    fixed_box_pos = to_array(fixed_box_pos_str)
    fixed_box_geom = fixed_box_body.find("geom[@type='box']")
    fixed_box_size = to_array(fixed_box_geom.get("size"))

    # Free box (at top level in MJCF)
    free_box_body = worldbody.find("body[@name='/worldbody/free_box']")
    free_box_pos_str = free_box_body.get("pos")
    free_box_pos = to_array(free_box_pos_str)
    free_box_geom = free_box_body.find("geom[@type='box']")
    free_box_size = to_array(free_box_geom.get("size"))

    # Create temporary USD file
    usd_file = str(asset_tmp_path / "all_joints.usda")

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

    # Create prismatic box
    prismatic_box = UsdGeom.Cube.Define(stage, "/worldbody/base/prismatic_box")
    prismatic_box.AddTranslateOp().Set(Gf.Vec3d(prismatic_box_pos[0], prismatic_box_pos[1], prismatic_box_pos[2]))
    prismatic_box.GetSizeAttr().Set(prismatic_box_size[0] * 2.0)
    prismatic_box_rigid = UsdPhysics.RigidBodyAPI.Apply(prismatic_box.GetPrim())
    prismatic_box_rigid.GetKinematicEnabledAttr().Set(False)

    # Create prismatic joint
    prismatic_joint_prim = UsdPhysics.PrismaticJoint.Define(stage, "/worldbody/base/prismatic_box_joint")
    prismatic_joint_prim.CreateBody0Rel().SetTargets([base.GetPrim().GetPath()])
    prismatic_joint_prim.CreateBody1Rel().SetTargets([prismatic_box.GetPrim().GetPath()])
    prismatic_joint_prim.CreateAxisAttr().Set("Z")
    prismatic_joint_prim.CreateLowerLimitAttr().Set(prismatic_range[0])
    prismatic_joint_prim.CreateUpperLimitAttr().Set(prismatic_range[1])
    prismatic_joint_prim.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    prismatic_joint_prim.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    prismatic_joint_prim.GetPrim().CreateAttribute("linear:stiffness", Sdf.ValueTypeNames.Float).Set(50.0)
    prismatic_joint_prim.GetPrim().CreateAttribute("linear:damping", Sdf.ValueTypeNames.Float).Set(5.0)
    prismatic_drive_api = UsdPhysics.DriveAPI.Apply(prismatic_joint_prim.GetPrim(), "linear")
    prismatic_drive_api.CreateStiffnessAttr().Set(120.0)
    prismatic_drive_api.CreateDampingAttr().Set(12.0)

    # Create revolute box
    revolute_box = UsdGeom.Cube.Define(stage, "/worldbody/base/revolute_box")
    revolute_box.AddTranslateOp().Set(Gf.Vec3d(revolute_box_pos[0], revolute_box_pos[1], revolute_box_pos[2]))
    revolute_box.GetSizeAttr().Set(revolute_box_size[0] * 2.0)
    revolute_box_rigid = UsdPhysics.RigidBodyAPI.Apply(revolute_box.GetPrim())
    revolute_box_rigid.GetKinematicEnabledAttr().Set(False)

    # Create revolute joint
    revolute_joint_prim = UsdPhysics.RevoluteJoint.Define(stage, "/worldbody/base/revolute_box_joint")
    revolute_joint_prim.CreateBody0Rel().SetTargets([base.GetPrim().GetPath()])
    revolute_joint_prim.CreateBody1Rel().SetTargets([revolute_box.GetPrim().GetPath()])
    revolute_joint_prim.CreateAxisAttr().Set("Z")
    revolute_joint_prim.CreateLowerLimitAttr().Set(revolute_range[0])
    revolute_joint_prim.CreateUpperLimitAttr().Set(revolute_range[1])
    revolute_joint_prim.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    revolute_joint_prim.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    revolute_joint_prim.GetPrim().CreateAttribute("stiffness", Sdf.ValueTypeNames.Float).Set(50.0)
    revolute_joint_prim.GetPrim().CreateAttribute("angular:damping", Sdf.ValueTypeNames.Float).Set(5.0)
    revolute_drive_api = UsdPhysics.DriveAPI.Apply(revolute_joint_prim.GetPrim(), "angular")
    revolute_drive_api.CreateStiffnessAttr().Set(120.0)
    revolute_drive_api.CreateDampingAttr().Set(12.0)

    # Create spherical box
    spherical_box = UsdGeom.Cube.Define(stage, "/worldbody/base/spherical_box")
    spherical_box.AddTranslateOp().Set(Gf.Vec3d(spherical_box_pos[0], spherical_box_pos[1], spherical_box_pos[2]))
    spherical_box.GetSizeAttr().Set(spherical_box_size[0] * 2.0)
    spherical_box_rigid = UsdPhysics.RigidBodyAPI.Apply(spherical_box.GetPrim())
    spherical_box_rigid.GetKinematicEnabledAttr().Set(False)

    # Create spherical joint
    spherical_joint_prim = UsdPhysics.SphericalJoint.Define(stage, "/worldbody/base/spherical_box_joint")
    spherical_joint_prim.CreateBody0Rel().SetTargets([base.GetPrim().GetPath()])
    spherical_joint_prim.CreateBody1Rel().SetTargets([spherical_box.GetPrim().GetPath()])
    spherical_joint_prim.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    spherical_joint_prim.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    # Create fixed box
    fixed_box = UsdGeom.Cube.Define(stage, "/worldbody/base/fixed_box")
    fixed_box.AddTranslateOp().Set(Gf.Vec3d(fixed_box_pos[0], fixed_box_pos[1], fixed_box_pos[2]))
    fixed_box.GetSizeAttr().Set(fixed_box_size[0] * 2.0)
    fixed_box_rigid = UsdPhysics.RigidBodyAPI.Apply(fixed_box.GetPrim())
    fixed_box_rigid.GetKinematicEnabledAttr().Set(False)

    # Create fixed joint
    fixed_joint_prim = UsdPhysics.FixedJoint.Define(stage, "/worldbody/base/fixed_box_joint")
    fixed_joint_prim.CreateBody0Rel().SetTargets([base.GetPrim().GetPath()])
    fixed_joint_prim.CreateBody1Rel().SetTargets([fixed_box.GetPrim().GetPath()])
    fixed_joint_prim.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    fixed_joint_prim.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    # Create free box (at top level, not under base)
    free_box = UsdGeom.Cube.Define(stage, "/worldbody/free_box")
    free_box.AddTranslateOp().Set(Gf.Vec3d(free_box_pos[0], free_box_pos[1], free_box_pos[2]))
    free_box.GetSizeAttr().Set(free_box_size[0] * 2.0)
    free_box_rigid = UsdPhysics.RigidBodyAPI.Apply(free_box.GetPrim())
    free_box_rigid.GetKinematicEnabledAttr().Set(False)

    # Create free joint (PhysicsJoint type) - connects to worldbody root, not base
    free_joint_prim = UsdPhysics.Joint.Define(stage, "/worldbody/free_box_joint")
    free_joint_prim.CreateBody0Rel().SetTargets([root_prim.GetPath()])
    free_joint_prim.CreateBody1Rel().SetTargets([free_box.GetPrim().GetPath()])
    free_joint_prim.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    free_joint_prim.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    stage.Save()

    return usd_file


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["all_joints_mjcf"])
@pytest.mark.parametrize("scale", [1.0, 2.0])
@pytest.mark.skipif(not HAS_USD_SUPPORT, reason="USD support not available")
def test_joints_mjcf_vs_usd(xml_path, all_joints_usd, scale, tol):
    """
    Test that MJCF and USD scenes with all joint types (prismatic, revolute, spherical, fixed, free)
    produce equivalent Genesis entities.

    This test verifies that all five joint types are correctly parsed from both
    MJCF and USD formats and produce equivalent results.
    """
    mjcf_scene = build_mjcf_scene(xml_path, scale=scale)
    usd_scene = build_usd_scene(all_joints_usd, scale=scale)

    # Compare entire scenes - this will check all joints via compare_joints
    compare_scene(mjcf_scene, usd_scene, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["usd/sneaker_airforce", "usd/RoughnessTest"])
@pytest.mark.skipif(not HAS_USD_SUPPORT, reason="USD support not available")
def test_usd_visual_parse(model_name, tol):
    glb_asset_path = get_hf_dataset(pattern=f"{model_name}.glb")
    glb_file = os.path.join(glb_asset_path, f"{model_name}.glb")
    usd_asset_path = get_hf_dataset(pattern=f"{model_name}.usdz")
    usd_file = os.path.join(usd_asset_path, f"{model_name}.usdz")

    mesh_scene = build_mesh_scene(glb_file, scale=1.0)
    usd_scene = build_usd_scene(usd_file, scale=1.0, vis_mode="visual", is_stage=False)

    compare_mesh_scene(mesh_scene, usd_scene, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("usd_file", ["usd/nodegraph.usda"])
@pytest.mark.skipif(not HAS_USD_SUPPORT, reason="USD support not available")
def test_usd_parse_nodegraph(usd_file):
    asset_path = get_hf_dataset(pattern=usd_file)
    usd_file = os.path.join(asset_path, usd_file)

    usd_scene = build_usd_scene(usd_file, scale=1.0, vis_mode="visual", is_stage=False)

    texture0 = usd_scene.entities[0].vgeoms[0].vmesh.surface.diffuse_texture
    texture1 = usd_scene.entities[0].vgeoms[1].vmesh.surface.diffuse_texture
    assert isinstance(texture0, gs.textures.ColorTexture)
    assert isinstance(texture1, gs.textures.ColorTexture)
    assert_allclose(texture0.color, (0.8, 0.2, 0.2), rtol=USD_COLOR_TOL)
    assert_allclose(texture1.color, (0.2, 0.6, 0.9), rtol=USD_COLOR_TOL)


@pytest.mark.required
@pytest.mark.parametrize(
    "usd_file", ["usd/WoodenCrate/WoodenCrate_D1_1002.usda", "usd/franka_mocap_teleop/table_scene.usd"]
)
@pytest.mark.parametrize("backend", [gs.cuda])
@pytest.mark.skipif(not HAS_USD_SUPPORT, reason="USD support not available")
@pytest.mark.skipif(not HAS_OMNIVERSE_KIT_SUPPORT, reason="omniverse-kit support not available")
def test_usd_bake(usd_file, tmp_path):
    RETRY_NUM = 3 if "PYTEST_XDIST_WORKER" in os.environ else 0
    RETRY_DELAY = 30.0

    asset_path = get_hf_dataset(pattern=os.path.join(os.path.dirname(usd_file), "*"), local_dir=tmp_path)
    usd_file = os.path.join(asset_path, usd_file)

    # Note that bootstrapping omni-kit by multiple workers concurrently is causing failure.
    # There is no easy way to get around this limitation except retrying after some delay...
    retry_idx = 0
    while True:
        usd_scene = build_usd_scene(usd_file, scale=1.0, vis_mode="visual", is_stage=False, fixed=True)

        is_any_baked = False
        for vgeom in usd_scene.entities[0].vgeoms:
            vmesh = vgeom.vmesh
            bake_success = vmesh.metadata["bake_success"]
            try:
                assert bake_success is None or bake_success
            except AssertionError:
                if retry_idx < RETRY_NUM:
                    usd_scene.destroy()
                    print(f"Failed to bake usd. Trying again in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                    break
                raise
            is_any_baked |= bake_success
        else:
            assert is_any_baked
            break


@pytest.mark.required
@pytest.mark.parametrize("scale", [1.0, 2.0])
@pytest.mark.skipif(not HAS_USD_SUPPORT, reason="USD support not available")
def test_massapi_invalid_defaults_mjcf_vs_usd(asset_tmp_path, scale, tol):
    """
    Test that USD MassAPI with invalid default values produces equivalent results to MJCF.

    USD Physics MassAPI defines some attributes with invalid default values:
    - centerOfMass: default (-inf, -inf, -inf) - invalid, should be recomputed
    - principalAxes: default (0, 0, 0, 0) - invalid quaternion, should be recomputed
    - diagonalInertia: default (0, 0, 0) - valid but means ignored, should be recomputed
    - mass: default 0 - valid but means ignored, should be recomputed

    This test creates equivalent MJCF and USD scenes where mass properties are computed
    from geometry (MJCF has no inertial element, USD has MassAPI with invalid defaults).
    Both should produce equivalent results.
    """
    mjcf = ET.Element("mujoco", model="massapi_test")
    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "joint", armature="0.0")

    worldbody = ET.SubElement(mjcf, "worldbody")

    floor = ET.SubElement(worldbody, "body", name="/worldbody/floor")
    ET.SubElement(floor, "geom", type="plane", pos="0. 0. 0.", size="40. 40. 40.")

    box = ET.SubElement(worldbody, "body", name="/worldbody/test_box", pos="0. 0. 0.3")
    ET.SubElement(box, "geom", type="box", size="0.2 0.2 0.2", pos="0. 0. 0.")
    ET.SubElement(box, "joint", name="/worldbody/test_box_joint", type="free")

    xml_tree = ET.ElementTree(mjcf)
    xml_file = str(asset_tmp_path / "massapi_test.xml")
    xml_tree.write(xml_file, encoding="utf-8", xml_declaration=True)

    usd_file = str(asset_tmp_path / "massapi_test.usda")

    stage = Usd.Stage.CreateNew(usd_file)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root_prim = stage.DefinePrim("/worldbody", "Xform")
    stage.SetDefaultPrim(root_prim)

    floor = UsdGeom.Plane.Define(stage, "/worldbody/floor")
    floor.GetAxisAttr().Set("Z")
    floor.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    floor.GetWidthAttr().Set(80.0)
    floor.GetLengthAttr().Set(80.0)
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())

    box = UsdGeom.Cube.Define(stage, "/worldbody/test_box")
    box.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.3))
    box.GetSizeAttr().Set(0.4)  # 0.2 half-extent * 2

    box_rigid = UsdPhysics.RigidBodyAPI.Apply(box.GetPrim())
    box_rigid.GetKinematicEnabledAttr().Set(False)

    mass_api = UsdPhysics.MassAPI.Apply(box.GetPrim())

    stage.Save()

    mjcf_scene = build_mjcf_scene(xml_file, scale=scale)
    usd_scene = build_usd_scene(usd_file, scale=scale)

    compare_scene(mjcf_scene, usd_scene, tol=tol)
