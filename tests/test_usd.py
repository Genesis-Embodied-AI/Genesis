"""
Test USD parsing and comparison with compared scenes.

This module tests that USD files can be parsed correctly and that scenes
loaded from USD files match equivalent scenes loaded from compared files.
"""

import os

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

USD_COLOR_TOL = 1e-07  # Parsing from .usd loses a little precision in color


def to_array(s: str) -> np.ndarray:
    """
    Convert a string of space-separated floats to a numpy array.
    """
    return np.array([float(x) for x in s.split()])


def compare_links(compared_links, usd_links, tol):
    """
    Generic function to compare links between two scenes.
    Compares as much link data as possible including positions, orientations,
    inertial properties, structural properties, etc.

    Parameters
    ----------
    compared_links : list
        List of links from compared scene
    usd_links : list
        List of links from USD scene
    tol : float, optional
        Tolerance for numerical comparisons.
    """
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

        # Compare link properties
        assert_allclose(compared_link.pos, usd_link.pos, tol=tol)
        assert_allclose(compared_link.quat, usd_link.quat, tol=tol)
        assert compared_link.is_fixed == usd_link.is_fixed
        assert len(compared_link.geoms) == len(usd_link.geoms)
        assert compared_link.n_joints == usd_link.n_joints
        assert len(compared_link.vgeoms) == len(usd_link.vgeoms)

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
        assert compared_parent_name == usd_parent_name

        # Compare inertial properties if available
        assert_allclose(compared_link.inertial_pos, usd_link.inertial_pos, tol=tol)
        assert_allclose(compared_link.inertial_quat, usd_link.inertial_quat, tol=tol)

        # Skip mass and inertia checks for fixed links - they're not used in simulation
        if not compared_link.is_fixed:
            # Both scenes now use the same material density (1000 kg/m³), so values should match closely
            assert_allclose(compared_link.inertial_mass, usd_link.inertial_mass, atol=tol)
            assert_allclose(compared_link.inertial_i, usd_link.inertial_i, atol=tol)


def compare_joints(compared_joints, usd_joints, tol):
    """
    Generic function to compare joints between two scenes.
    Compares as much joint data as possible including positions, orientations,
    degrees of freedom, limits, dynamics properties, etc.

    Parameters
    ----------
    compared_joints : list
        List of joints from compared scene
    usd_joints : list
        List of joints from USD scene
    tol : float, optional
        Tolerance for numerical comparisons.
    """
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
        assert_allclose(compared_joint.pos, usd_joint.pos, tol=tol)
        assert_allclose(compared_joint.quat, usd_joint.quat, tol=tol)
        assert compared_joint.n_qs == usd_joint.n_qs
        assert compared_joint.n_dofs == usd_joint.n_dofs

        # Compare initial qpos
        assert_allclose(compared_joint.init_qpos, usd_joint.init_qpos, tol=tol)

        # Skip mass/inertia-dependent property checks for fixed joints - they're not used in simulation
        if compared_joint.type != gs.JOINT_TYPE.FIXED:
            # Compare dof limits
            assert_allclose(compared_joint.dofs_limit, usd_joint.dofs_limit, tol=tol)

            # Compare dof motion properties
            assert_allclose(compared_joint.dofs_motion_ang, usd_joint.dofs_motion_ang, tol=tol)
            assert_allclose(compared_joint.dofs_motion_vel, usd_joint.dofs_motion_vel, tol=tol)
            assert_allclose(compared_joint.dofs_frictionloss, usd_joint.dofs_frictionloss, tol=tol)
            assert_allclose(compared_joint.dofs_stiffness, usd_joint.dofs_stiffness, tol=tol)
            assert_allclose(compared_joint.dofs_frictionloss, usd_joint.dofs_frictionloss, tol=tol)
            assert_allclose(compared_joint.dofs_force_range, usd_joint.dofs_force_range, tol=tol)
            assert_allclose(compared_joint.dofs_damping, usd_joint.dofs_damping, tol=tol)
            assert_allclose(compared_joint.dofs_armature, usd_joint.dofs_armature, tol=tol)

            # Compare dof control properties
            assert_allclose(compared_joint.dofs_kp, usd_joint.dofs_kp, tol=tol)
            assert_allclose(compared_joint.dofs_kv, usd_joint.dofs_kv, tol=tol)
            assert_allclose(compared_joint.dofs_force_range, usd_joint.dofs_force_range, tol=tol)


def compare_geoms(compared_geoms, usd_geoms, tol):
    """
    Generic function to compare geoms between two scenes.
    Compares as much geom data as possible including positions, orientations, sizes, etc.
    """
    assert len(compared_geoms) == len(usd_geoms)

    # Sort geoms by link name for consistent comparison
    compared_geoms_sorted = sorted(compared_geoms, key=lambda g: (g.link.name, g._idx))
    usd_geoms_sorted = sorted(usd_geoms, key=lambda g: (g.link.name, g._idx))

    for compared_geom, usd_geom in zip(compared_geoms_sorted, usd_geoms_sorted):
        assert compared_geom.type == usd_geom.type
        assert_allclose(compared_geom.init_pos, usd_geom.init_pos, tol=tol)
        assert_allclose(compared_geom.init_quat, usd_geom.init_quat, tol=tol)
        assert_allclose(compared_geom.get_AABB(), usd_geom.get_AABB(), tol=tol)


def compare_vgeoms(compared_vgeoms, usd_vgeoms, tol, strict=True):
    assert len(compared_vgeoms) == len(usd_vgeoms)

    # Sort geoms by link name for consistent comparison
    compared_vgeoms_sorted = sorted(compared_vgeoms, key=lambda g: g.vmesh.metadata["name"])
    usd_vgeoms_sorted = sorted(usd_vgeoms, key=lambda g: g.vmesh.metadata["name"].split("/")[-1])

    for compared_vgeom, usd_vgeom in zip(compared_vgeoms_sorted, usd_vgeoms_sorted):
        if strict:
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
            check_gs_meshes(compared_vgeom_mesh, usd_vgeom_mesh, mesh_name)
        else:
            assert_allclose(compared_vgeom.get_AABB(), usd_vgeom.get_AABB(), tol=tol)

        compared_vgeom_surface = compared_vgeom_mesh.surface
        usd_vgeom_surface = usd_vgeom_mesh.surface
        check_gs_surfaces(compared_vgeom_surface, usd_vgeom_surface, mesh_name)


def compare_scene(compared_scene: gs.Scene, usd_scene: gs.Scene, tol: float):
    """Compare structure and data between compared scene and USD scene."""
    compared_entities = compared_scene.entities
    usd_entities = usd_scene.entities

    compared_links = [link for entity in compared_entities for link in entity.links]
    usd_links = [link for entity in usd_entities for link in entity.links]
    compare_links(compared_links, usd_links, tol=tol)

    compared_geoms = [geom for entity in compared_entities for geom in entity.geoms]
    usd_geoms = [geom for entity in usd_entities for geom in entity.geoms]
    compare_geoms(compared_geoms, usd_geoms, tol=tol)

    compared_joints = [joint for entity in compared_entities for joint in entity.joints]
    usd_joints = [joint for entity in usd_entities for joint in entity.joints]
    compare_joints(compared_joints, usd_joints, tol=tol)


def compare_mesh_scene(compared_scene: gs.Scene, usd_scene: gs.Scene, tol: float):
    """
    Compare mesh data between mesh scene and USD scene.
    Meshes are loaded with transformations baked. Therefore, we only compare mesh data.
    """
    compared_entities = compared_scene.entities
    usd_entities = usd_scene.entities
    compared_vgeoms = [vgeom for entity in compared_entities for vgeom in entity.vgeoms]
    usd_vgeoms = [vgeom for entity in usd_entities for vgeom in entity.vgeoms]
    compare_vgeoms(compared_vgeoms, usd_vgeoms, tol=tol)


def build_mjcf_scene(xml_path: str, scale: float):
    """
    Build a MJCF scene from its file path.

    Parameters
    ----------
    xml_path : str
        Path to the MJCF/XML file
    scale : float
        Scale factor to apply to the scene
    Returns
    -------
    mjcf_scene : gs.Scene
        The MJCF scene
    """
    # Create MJCF scene
    mjcf_scene = gs.Scene()
    mjcf_morph = gs.morphs.MJCF(file=xml_path, scale=scale)
    mjcf_scene.add_entity(mjcf_morph, material=gs.materials.Rigid(rho=1000.0))
    mjcf_scene.build()
    return mjcf_scene


def build_usd_scene(
    usd_file: str,
    scale: float,
    vis_mode: str = "collision",
    is_stage: bool = True,
):
    """
    Build a USD scene from its file path.

    Parameters
    ----------
    usd_file : str
        Path to the USD file
    scale : float
        Scale factor to apply to the scene
    vis_mode : str
        The visualization mode of the scene
    is_stage : bool
        Whether to add the USD file as a stage or as an entity
    Returns
    -------
    usd_scene : gs.Scene
        The USD scene
    """
    # Create USD scene
    usd_scene = gs.Scene()
    usd_morph = gs.morphs.USD(
        usd_ctx=UsdContext(usd_file, use_bake_cache=False),
        scale=scale,
    )
    if is_stage:
        usd_scene.add_stage(usd_morph, vis_mode=vis_mode, material=gs.materials.Rigid(rho=1000.0))
    else:
        usd_scene.add_entity(usd_morph, vis_mode=vis_mode, material=gs.materials.Rigid(rho=1000.0))
    usd_scene.build()
    return usd_scene


def build_mesh_scene(mesh_file: str, scale: float):
    """
    Build a mesh scene from its file path.

    Parameters
    ----------
    mesh_file : str
        Path to the mesh file
    scale : float
        Scale factor to apply to the scene
    Returns
    -------
    mesh_scene : gs.Scene
        The mesh scene
    """
    mesh_scene = gs.Scene()
    mesh_morph = gs.morphs.Mesh(
        file=mesh_file,
        scale=scale,
        euler=(-90, 0, 0),
        group_by_material=False,
    )
    mesh_scene.add_entity(mesh_morph, material=gs.materials.Rigid(rho=1000.0))
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
def box_plane_usd(asset_tmp_path, box_plane_mjcf: ET.ElementTree):
    """Generate a USD file equivalent to the MJCF box_plane_mjcf fixture.

    Extracts data from the MJCF XML structure to build the USD file.
    """
    # Extract data from MJCF XML structure
    worldbody = box_plane_mjcf.find("worldbody")

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

    # Create temporary USD file
    usd_file = str(asset_tmp_path / "box_plane.usda")

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


@pytest.mark.parametrize("precision", ["32"])
@pytest.mark.parametrize("model_name", ["box_plane_mjcf"])
@pytest.mark.parametrize("scale", [1.0, 2.0])
@pytest.mark.skipif(not HAS_USD_SUPPORT, reason="USD support not available")
def test_box_plane_mjcf_vs_usd(xml_path, box_plane_usd, scale, tol):
    """Test that MJCF and USD scenes produce equivalent Genesis entities."""
    mjcf_scene = build_mjcf_scene(xml_path, scale=scale)
    usd_scene = build_usd_scene(box_plane_usd, scale=scale)
    compare_scene(mjcf_scene, usd_scene, tol=tol)


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
        range="-0.1 0.4",
        stiffness="50.0",
        damping="5.0",
    )

    # Add actuator for PD controller (maps to dofs_kp and dofs_kv)
    # The parser uses: dofs_kp = -gear * biasprm[1] * scale^3
    # So to get dofs_kp=120.0, we need biasprm[1] = -120.0 (with gear=1, scale=1)
    actuator = ET.SubElement(mjcf, "actuator")
    ET.SubElement(
        actuator,
        "general",
        name="/worldbody/base/box_joint_actuator",
        joint="/worldbody/base/box_joint",
        biastype="affine",
        gainprm="120.0 0 0",  # gainprm[0] must equal -biasprm[1] to avoid warning
        biasprm="0 -120.0 -12.0",  # biasprm format: [b0, b1, b2] where b1=kp, b2=kv (negated)
    )

    return mjcf


@pytest.fixture(scope="session")
def prismatic_joint_usd(asset_tmp_path, prismatic_joint_mjcf: ET.ElementTree):
    """Generate a USD file equivalent to the prismatic joint MJCF fixture."""
    worldbody = prismatic_joint_mjcf.find("worldbody")

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

    # Box with prismatic joint
    box_body = base_body.find("body[@name='/worldbody/base/box']")
    box_pos_str = box_body.get("pos")
    box_pos = to_array(box_pos_str)
    box_geom = box_body.find("geom[@type='box']")
    box_size_str = box_geom.get("size")
    box_size = to_array(box_size_str)

    # Joint limits
    joint = box_body.find("joint[@name='/worldbody/base/box_joint']")
    range_str = joint.get("range")
    range_vals = to_array(range_str)
    lower_limit = range_vals[0]
    upper_limit = range_vals[1]

    # Create temporary USD file
    usd_file = str(asset_tmp_path / "prismatic_joint.usda")

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

    # Add stiffness and damping attributes (using last candidate name)
    joint_prim.GetPrim().CreateAttribute("linear:stiffness", Sdf.ValueTypeNames.Float).Set(50.0)
    joint_prim.GetPrim().CreateAttribute("linear:damping", Sdf.ValueTypeNames.Float).Set(5.0)

    # Create drive API
    drive_api = UsdPhysics.DriveAPI.Apply(joint_prim.GetPrim(), "linear")
    drive_api.CreateStiffnessAttr().Set(120.0)
    drive_api.CreateDampingAttr().Set(12.0)

    stage.Save()
    return usd_file


@pytest.mark.parametrize("precision", ["32"])
@pytest.mark.parametrize("model_name", ["prismatic_joint_mjcf"])
@pytest.mark.parametrize("scale", [1.0, 2.0])
@pytest.mark.skipif(not HAS_USD_SUPPORT, reason="USD support not available")
def test_prismatic_joint_mjcf_vs_usd(xml_path, prismatic_joint_usd, scale, tol):
    """Test that MJCF and USD scenes with prismatic joints produce equivalent Genesis entities."""
    mjcf_scene = build_mjcf_scene(xml_path, scale=scale)
    usd_scene = build_usd_scene(prismatic_joint_usd, scale=scale)
    compare_scene(mjcf_scene, usd_scene, tol=tol)


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
        stiffness="50.0",
        damping="5.0",
    )

    # Add actuator for PD controller (maps to dofs_kp and dofs_kv)
    # The parser uses: dofs_kp = -gear * biasprm[1] * scale^3
    # So to get dofs_kp=120.0, we need biasprm[1] = -120.0 (with gear=1, scale=1)
    actuator = ET.SubElement(mjcf, "actuator")
    ET.SubElement(
        actuator,
        "general",
        name="/worldbody/base/box_joint_actuator",
        joint="/worldbody/base/box_joint",
        biastype="affine",
        gainprm="120.0 0 0",  # gainprm[0] must equal -biasprm[1] to avoid warning
        biasprm="0 -120.0 -12.0",  # biasprm format: [b0, b1, b2] where b1=kp, b2=kv (negated)
    )

    return mjcf


@pytest.fixture(scope="session")
def revolute_joint_usd(asset_tmp_path, revolute_joint_mjcf: ET.ElementTree):
    """Generate a USD file equivalent to the revolute joint MJCF fixture."""
    worldbody = revolute_joint_mjcf.find("worldbody")

    # Floor
    floor_body = worldbody.find("body[@name='/worldbody/floor']")
    floor_geom = floor_body.find("geom[@type='plane']")
    floor_pos_str = floor_geom.get("pos")
    floor_pos = to_array(floor_pos_str)
    floor_size_str = floor_geom.get("size", "40. 40. 40.")
    floor_size = to_array(floor_size_str)

    # Base
    base_body = worldbody.find("body[@name='/worldbody/base']")
    base_pos_str = base_body.get("pos")
    base_pos = to_array(base_pos_str)
    base_geom = base_body.find("geom[@type='box']")
    base_size_str = base_geom.get("size")
    base_size = to_array(base_size_str)

    # Box with revolute joint
    box_body = base_body.find("body[@name='/worldbody/base/box']")
    box_pos_str = box_body.get("pos")
    box_pos = to_array(box_pos_str)
    box_geom = box_body.find("geom[@type='box']")
    box_size_str = box_geom.get("size")
    box_size = to_array(box_size_str)

    # Joint limits
    joint = box_body.find("joint[@name='/worldbody/base/box_joint']")
    range_str = joint.get("range")
    range_vals = to_array(range_str)
    lower_limit_deg = range_vals[0]
    upper_limit_deg = range_vals[1]

    # Create temporary USD file
    usd_file = str(asset_tmp_path / "revolute_joint.usda")

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

    # Add stiffness and damping attributes (using last candidate name)
    joint_prim.GetPrim().CreateAttribute("stiffness", Sdf.ValueTypeNames.Float).Set(50.0)
    joint_prim.GetPrim().CreateAttribute("angular:damping", Sdf.ValueTypeNames.Float).Set(5.0)

    # Create drive API (use "angular" for revolute joints)
    drive_api = UsdPhysics.DriveAPI.Apply(joint_prim.GetPrim(), "angular")
    drive_api.CreateStiffnessAttr().Set(120.0)
    drive_api.CreateDampingAttr().Set(12.0)

    stage.Save()
    return usd_file


@pytest.mark.parametrize("precision", ["32"])
@pytest.mark.parametrize("model_name", ["revolute_joint_mjcf"])
@pytest.mark.parametrize("scale", [1.0, 2.0])
@pytest.mark.skipif(not HAS_USD_SUPPORT, reason="USD support not available")
def test_revolute_joint_mjcf_vs_usd(xml_path, revolute_joint_usd, scale, tol):
    """Test that MJCF and USD scenes with revolute joints produce equivalent Genesis entities."""
    mjcf_scene = build_mjcf_scene(xml_path, scale=scale)
    usd_scene = build_usd_scene(revolute_joint_usd, scale=scale)
    compare_scene(mjcf_scene, usd_scene, tol=tol)


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
def spherical_joint_usd(asset_tmp_path, spherical_joint_mjcf: ET.ElementTree):
    """Generate a USD file equivalent to the spherical joint MJCF fixture."""
    worldbody = spherical_joint_mjcf.find("worldbody")

    # Floor
    floor_body = worldbody.find("body[@name='/worldbody/floor']")
    floor_geom = floor_body.find("geom[@type='plane']")
    floor_pos_str = floor_geom.get("pos")
    floor_pos = to_array(floor_pos_str)
    floor_size_str = floor_geom.get("size", "40. 40. 40.")
    floor_size = to_array(floor_size_str)

    # Base
    base_body = worldbody.find("body[@name='/worldbody/base']")
    base_pos_str = base_body.get("pos")
    base_pos = to_array(base_pos_str)
    base_geom = base_body.find("geom[@type='box']")
    base_size_str = base_geom.get("size")
    base_size = to_array(base_size_str)

    # Box with spherical joint
    box_body = base_body.find("body[@name='/worldbody/base/box']")
    box_pos_str = box_body.get("pos")
    box_pos = to_array(box_pos_str)
    box_geom = box_body.find("geom[@type='box']")
    box_size_str = box_geom.get("size")
    box_size = to_array(box_size_str)

    # Create temporary USD file
    usd_file = str(asset_tmp_path / "spherical_joint.usda")

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


@pytest.mark.parametrize("precision", ["32"])
@pytest.mark.parametrize("model_name", ["spherical_joint_mjcf"])
@pytest.mark.parametrize("scale", [1.0, 2.0])
@pytest.mark.skipif(not HAS_USD_SUPPORT, reason="USD support not available")
def test_spherical_joint_mjcf_vs_usd(xml_path, spherical_joint_usd, scale, tol):
    """Test that MJCF and USD scenes with spherical joints produce equivalent Genesis entities."""
    mjcf_scene = build_mjcf_scene(xml_path, scale=scale)
    usd_scene = build_usd_scene(spherical_joint_usd, scale=scale)
    compare_scene(mjcf_scene, usd_scene, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("precision", ["32"])
@pytest.mark.parametrize("model_name", ["usd/sneaker_airforce", "usd/RoughnessTest"])
@pytest.mark.skipif(not HAS_USD_SUPPORT, reason="USD support not available")
def test_usd_visual_parse(model_name, tol):
    glb_file = os.path.join(get_hf_dataset(pattern=f"{model_name}.glb"), f"{model_name}.glb")
    usd_file = os.path.join(get_hf_dataset(pattern=f"{model_name}.usdz"), f"{model_name}.usdz")
    mesh_scene = build_mesh_scene(glb_file, scale=1.0)
    usd_scene = build_usd_scene(usd_file, scale=1.0, vis_mode="visual", is_stage=False)
    compare_mesh_scene(mesh_scene, usd_scene, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("precision", ["32"])
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
@pytest.mark.parametrize("precision", ["32"])
@pytest.mark.parametrize(
    "usd_file", ["usd/WoodenCrate/WoodenCrate_D1_1002.usda", "usd/franka_mocap_teleop/table_scene.usd"]
)
@pytest.mark.parametrize("backend", [gs.cuda])
@pytest.mark.skipif(not HAS_USD_SUPPORT, reason="USD support not available")
@pytest.mark.skipif(not HAS_OMNIVERSE_KIT_SUPPORT, reason="omniverse-kit support not available")
def test_usd_bake(usd_file):
    asset_path = get_hf_dataset(pattern=os.path.join(os.path.dirname(usd_file), "*"), local_dir_use_symlinks=False)
    usd_file = os.path.join(asset_path, usd_file)
    usd_scene = build_usd_scene(usd_file, scale=1.0, vis_mode="visual", is_stage=False)

    success_count = 0
    for vgeom in usd_scene.entities[0].vgeoms:
        vmesh = vgeom.vmesh
        bake_success = vmesh.metadata["bake_success"]
        assert bake_success is None or bake_success
        success_count += 1 if bake_success else 0
    assert success_count > 0
