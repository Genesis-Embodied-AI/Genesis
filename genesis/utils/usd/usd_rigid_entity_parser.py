"""
USD Rigid Entity Parser

Unified parser for extracting rigid entity information from USD stages.
Treats both articulations and rigid bodies as rigid entities, where rigid bodies
are treated as articulation roots with no child links.

The parser is agnostic to genesis structures, focusing only on USD structure.
"""

import copy
import re
import collections.abc
from typing import Dict, List, Literal, Tuple, TYPE_CHECKING

import numpy as np
from pxr import Sdf, Usd, UsdPhysics

import genesis as gs
from genesis.options.morphs import USD
from .. import geom as gu
from .. import urdf as urdf_utils
from .usd_geo_adapter import create_geo_info_from_prim, create_geo_infos_from_subtree
from .usd_parser_context import UsdParserContext
from .usd_parser_utils import (
    compute_gs_global_transform,
    compute_gs_relative_transform,
    compute_gs_joint_pos_from_usd_prim,
    compute_gs_joint_axis_and_pos_from_usd_prim,
    usd_quat_to_numpy,
)

if TYPE_CHECKING:
    from genesis.engine.entities.base_entity import Entity


# ==================== Helper Functions ====================


def _is_rigid_body(prim: Usd.Prim) -> bool:
    """
    Check if a prim should be regarded as a rigid body.

    Note: We regard CollisionAPI also as rigid body (they are fixed rigid body).

    Parameters
    ----------
    prim : Usd.Prim
        The prim to check.

    Returns
    -------
    bool
        True if the prim should be regarded as a rigid body, False otherwise.
    """
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        return False

    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        return True

    if prim.HasAPI(UsdPhysics.CollisionAPI):
        return True

    return False


# ==================== Geometry Collection Functions ====================

# Pattern matching for geometry collection
_visual_pattern = re.compile(r"^(visual|Visual).*")
_collision_pattern = re.compile(r"^(collision|Collision).*")
_all_pattern = re.compile(r"^.*")


def _create_geo_infos(
    context: UsdParserContext, link: Usd.Prim, pattern: re.Pattern, mesh_type: Literal["mesh", "vmesh"]
) -> List[Dict]:
    """
    Create geometry info dictionaries from a link prim and its children that match the pattern.

    Parameters
    ----------
    context : UsdParserContext
        The parser context.
    link : Usd.Prim
        The link prim.
    pattern : re.Pattern
        Pattern to match child prim names.
    mesh_type : Literal["mesh", "vmesh"]
        The mesh type to create geometry info for.

    Returns
    -------
    List[Dict]
        List of geometry info dictionaries.
    """
    # if the link itself is a geometry
    geo_infos: List[Dict] = []
    link_geo_info = create_geo_info_from_prim(context, link, link, mesh_type)
    if link_geo_info is not None:
        geo_infos.append(link_geo_info)

    # - Link
    #     - Visuals
    #     - Collisions
    search_roots: list[Usd.Prim] = []
    for child in link.GetChildren():
        child: Usd.Prim
        if pattern.match(str(child.GetName())):
            search_roots.append(child)

    for search_root in search_roots:
        geo_infos.extend(create_geo_infos_from_subtree(context, search_root, link, mesh_type))

    return geo_infos


def _create_visual_geo_infos(link: Usd.Prim, context: UsdParserContext) -> List[Dict]:
    """
    Create visual geometry info dictionaries from a link prim.

    Parameters
    ----------
    link : Usd.Prim
        The link prim.
    context : UsdParserContext
        The parser context.

    Returns
    -------
    List[Dict]
        List of visual geometry info dictionaries.
    """
    if context.vis_mode == "visual":
        vis_geo_infos = _create_geo_infos(context, link, _visual_pattern, "vmesh")
        if len(vis_geo_infos) == 0:
            # if no visual geometries found, use any pattern to find visual geometries
            gs.logger.info(
                f"No visual geometries found, using any pattern to find visual geometries in {link.GetPath()}"
            )
            vis_geo_infos = _create_geo_infos(context, link, _all_pattern, "vmesh")
    elif context.vis_mode == "collision":
        vis_geo_infos = _create_geo_infos(context, link, _collision_pattern, "vmesh")
    else:
        gs.raise_exception(f"Unsupported visualization mode {context.vis_mode}.")
    return vis_geo_infos


def _create_collision_geo_infos(link: Usd.Prim, context: UsdParserContext) -> List[Dict]:
    """
    Create collision geometry info dictionaries from a link prim.

    Parameters
    ----------
    link : Usd.Prim
        The link prim.
    context : UsdParserContext
        The parser context.

    Returns
    -------
    List[Dict]
        List of collision geometry info dictionaries.
    """
    return _create_geo_infos(context, link, _collision_pattern, "mesh")


# ==================== Helper Functions for Joint Parsing ====================


def _axis_str_to_vector(axis_str: str) -> np.ndarray:
    """
    Convert a joint axis string to a vector.

    Parameters
    ----------
    axis_str : str
        The axis string ('X', 'Y', or 'Z').
    """
    if axis_str == "X":
        return np.array([1.0, 0.0, 0.0], dtype=gs.np_float)
    elif axis_str == "Y":
        return np.array([0.0, 1.0, 0.0], dtype=gs.np_float)
    elif axis_str == "Z":
        return np.array([0.0, 0.0, 1.0], dtype=gs.np_float)
    else:
        gs.raise_exception(f"Unsupported joint axis {axis_str}.")


def _compute_child_link_local_axis_pos(
    joint: UsdPhysics.PrismaticJoint | UsdPhysics.RevoluteJoint, child_link: Usd.Prim
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the local axis and position of a joint in the child link local space.

    Parameters
    ----------
    joint : UsdPhysics.PrismaticJoint | UsdPhysics.RevoluteJoint
    child_link : Usd.Prim
    """
    axis_attr = joint.GetAxisAttr()
    axis_str = axis_attr.Get()
    axis = _axis_str_to_vector(axis_str)

    pos_attr = joint.GetLocalPos1Attr()
    usd_local_pos = pos_attr.Get()

    rotation_attr = joint.GetLocalRot1Attr()
    usd_local_rotation = usd_quat_to_numpy(rotation_attr.Get())
    usd_local_axis = gu.quat_to_R(usd_local_rotation) @ axis

    return compute_gs_joint_axis_and_pos_from_usd_prim(usd_local_axis, usd_local_pos, child_link)


def _compute_child_link_local_pos(joint: UsdPhysics.SphericalJoint, child_link: Usd.Prim) -> np.ndarray:
    """
    Compute the local position of a spherical joint in the child link local space.

    Parameters
    ----------
    joint : UsdPhysics.SphericalJoint
        The spherical joint API.
    child_link : Usd.Prim
        The child link prim.
    """
    pos_attr = joint.GetLocalPos1Attr()
    usd_local_pos = pos_attr.Get() if pos_attr else gu.zero_pos()
    gs_local_pos = compute_gs_joint_pos_from_usd_prim(usd_local_pos, child_link)
    return gs_local_pos


def _parse_revolute_joint(
    revolute_joint: UsdPhysics.RevoluteJoint, parent_link: Usd.Prim, child_link: Usd.Prim
) -> Dict:
    """
    Parse a revolute joint and create joint info dictionary.

    Parameters
    ----------
    revolute_joint : UsdPhysics.RevoluteJoint
        The revolute joint API.
    parent_link : Usd.Prim or None
        The parent link prim.
    child_link : Usd.Prim
        The child link prim.

    Returns
    -------
    dict
        Joint info dictionary.
    """
    j_info = dict()
    axis, pos = _compute_child_link_local_axis_pos(revolute_joint, child_link)

    # Normalize the axis
    unit_axis = axis / np.linalg.norm(axis)
    assert np.linalg.norm(unit_axis) == 1.0, f"Can not normalize the axis {axis}."

    # Get joint limits (angle limits are preserved under proportional scaling)
    # NOTE: I have no idea how we can scale the angle limits under non-uniform scaling.
    lower_limit_attr = revolute_joint.GetLowerLimitAttr()
    upper_limit_attr = revolute_joint.GetUpperLimitAttr()
    deg_lower_limit = lower_limit_attr.Get() if lower_limit_attr else -np.inf
    deg_upper_limit = upper_limit_attr.Get() if upper_limit_attr else np.inf
    lower_limit = np.deg2rad(deg_lower_limit)
    upper_limit = np.deg2rad(deg_upper_limit)

    # Fill the joint info
    j_info["pos"] = pos
    j_info["dofs_motion_ang"] = np.array([unit_axis])
    j_info["dofs_motion_vel"] = np.zeros((1, 3))
    j_info["dofs_limit"] = np.array([[lower_limit, upper_limit]])
    j_info["dofs_stiffness"] = np.array([0.0])
    j_info["type"] = gs.JOINT_TYPE.REVOLUTE
    j_info["n_qs"] = 1
    j_info["n_dofs"] = 1
    j_info["init_qpos"] = np.zeros(1)

    return j_info


def _parse_prismatic_joint(
    prismatic_joint: UsdPhysics.PrismaticJoint, parent_link: Usd.Prim | None, child_link: Usd.Prim
) -> Dict:
    """
    Parse a prismatic joint and create joint info dictionary.

    Parameters
    ----------
    prismatic_joint : UsdPhysics.PrismaticJoint
        The prismatic joint API.
    parent_link : Usd.Prim or None
        The parent link prim.
    child_link : Usd.Prim
        The child link prim.

    Returns
    -------
    dict
        Joint info dictionary.
    """

    j_info = dict()
    axis, pos = _compute_child_link_local_axis_pos(prismatic_joint, child_link)

    # Normalize the axis
    unit_axis = axis / np.linalg.norm(axis)
    assert np.linalg.norm(unit_axis) == 1.0, f"Can not normalize the axis {axis}."

    # Get joint limits (in linear units, not degrees)
    lower_limit_attr = prismatic_joint.GetLowerLimitAttr()
    upper_limit_attr = prismatic_joint.GetUpperLimitAttr()
    lower_limit = lower_limit_attr.Get() if lower_limit_attr else -np.inf
    upper_limit = upper_limit_attr.Get() if upper_limit_attr else np.inf

    # Fill the joint info
    j_info["pos"] = pos
    # Prismatic joints use dofs_motion_vel (linear motion) instead of dofs_motion_ang
    j_info["dofs_motion_ang"] = np.zeros((1, 3))
    j_info["dofs_motion_vel"] = np.array([unit_axis])
    j_info["dofs_limit"] = np.array([[lower_limit, upper_limit]])
    j_info["dofs_stiffness"] = np.array([0.0])
    j_info["type"] = gs.JOINT_TYPE.PRISMATIC
    j_info["n_qs"] = 1
    j_info["n_dofs"] = 1
    j_info["init_qpos"] = np.zeros(1)

    return j_info


def _parse_spherical_joint(
    spherical_joint: UsdPhysics.SphericalJoint, parent_link: Usd.Prim | None, child_link: Usd.Prim
) -> Dict:
    """
    Parse a spherical joint and create joint info dictionary.

    Parameters
    ----------
    spherical_joint : UsdPhysics.SphericalJoint
        The spherical joint API.
    parent_link : Usd.Prim or None
        The parent link prim.
    child_link : Usd.Prim
        The child link prim.

    Returns
    -------
    dict
        Joint info dictionary.
    """
    j_info = dict()

    pos = _compute_child_link_local_pos(spherical_joint, child_link)

    # Fill the joint info
    j_info["pos"] = pos
    # Spherical joints have 3 DOF (rotation around all 3 axes)
    j_info["dofs_motion_ang"] = np.eye(3)  # Identity matrix for 3 rotational axes
    j_info["dofs_motion_vel"] = np.zeros((3, 3))
    # Spherical joints typically don't have simple limits
    # If limits exist, they would be complex (cone limits), which we don't support yet
    j_info["dofs_limit"] = np.tile([-np.inf, np.inf], (3, 1))
    j_info["dofs_stiffness"] = np.zeros(3)
    j_info["type"] = gs.JOINT_TYPE.SPHERICAL
    j_info["n_qs"] = 4  # Quaternion representation
    j_info["n_dofs"] = 3  # 3 rotational DOF
    j_info["init_qpos"] = gu.identity_quat()  # Initial quaternion

    return j_info


def _parse_fixed_joint(joint_prim: Usd.Prim, parent_link: Usd.Prim, child_link: Usd.Prim) -> Dict:
    """
    Parse a fixed joint and create joint info dictionary.

    Parameters
    ----------
    joint_prim : Usd.Prim
        The joint prim.
    parent_link : Usd.Prim
        The parent link.
    child_link : Usd.Prim
        The child link.

    Returns
    -------
    dict
        Joint info dictionary.
    """
    j_info = dict()

    if not parent_link:
        gs.logger.debug(f"Root Fixed Joint detected {joint_prim.GetPath()}")
    else:
        gs.logger.debug(f"Fixed Joint detected {joint_prim.GetPath()}")

    j_info["dofs_motion_ang"] = np.zeros((0, 3))
    j_info["dofs_motion_vel"] = np.zeros((0, 3))
    j_info["dofs_limit"] = np.zeros((0, 2))
    j_info["dofs_stiffness"] = np.zeros((0))
    j_info["type"] = gs.JOINT_TYPE.FIXED
    j_info["n_qs"] = 0
    j_info["n_dofs"] = 0
    j_info["init_qpos"] = np.zeros(0)

    return j_info


def _create_joint_info_for_base_link(l_info: Dict) -> Dict:
    """
    Create a joint info dictionary for base links that have no incoming joints.

    Parameters
    ----------
    l_info : Dict
        Link info dictionary.

    Returns
    -------
    dict
        Joint info dictionary for FREE joint.
    """
    j_info = dict()
    l_name = l_info["name"]
    # NOTE: Any naming convention for base link joints?
    j_name = f"{l_name}_joint"
    j_info["name"] = j_name
    j_info["sol_params"] = gu.default_solver_params()

    if l_info["is_fixed"]:
        j_info["type"] = gs.JOINT_TYPE.FIXED
        j_info["n_qs"] = 0
        j_info["n_dofs"] = 0
        j_info["init_qpos"] = np.zeros(0)
        j_info["dofs_motion_ang"] = np.zeros((0, 3))
        j_info["dofs_motion_vel"] = np.zeros((0, 3))
        j_info["dofs_limit"] = np.zeros((0, 2))
        j_info["dofs_stiffness"] = np.zeros(0)
        j_info["dofs_invweight"] = np.zeros(0)
        j_info["dofs_frictionloss"] = np.zeros(0)
        j_info["dofs_damping"] = np.zeros(0)
        j_info["dofs_armature"] = np.zeros(0)
        j_info["dofs_kp"] = np.zeros((0,), dtype=gs.np_float)
        j_info["dofs_kv"] = np.zeros((0,), dtype=gs.np_float)
        j_info["dofs_force_range"] = np.zeros((0, 2))
    else:
        j_info["type"] = gs.JOINT_TYPE.FREE
        j_info["n_qs"] = 7
        j_info["n_dofs"] = 6
        j_info["init_qpos"] = np.concatenate([l_info["pos"], l_info["quat"]])
        j_info["dofs_motion_ang"] = np.eye(6, 3, -3)
        j_info["dofs_motion_vel"] = np.eye(6, 3)
        j_info["dofs_limit"] = np.tile([-np.inf, np.inf], (6, 1))
        j_info["dofs_stiffness"] = np.zeros(6)
        j_info["dofs_invweight"] = np.zeros(6)
        j_info["dofs_frictionloss"] = np.zeros(6)
        j_info["dofs_damping"] = np.zeros(6)
        j_info["dofs_armature"] = np.zeros(6)
        j_info["dofs_kp"] = np.zeros((6,), dtype=gs.np_float)
        j_info["dofs_kv"] = np.zeros((6,), dtype=gs.np_float)
        j_info["dofs_force_range"] = np.tile([-np.inf, np.inf], (6, 1))
    return j_info


def _parse_joint_dynamics(joint_prim: Usd.Prim, n_dofs: int) -> Dict:
    """
    Parse joint dynamics properties (friction, damping, armature) from a joint prim.

    Parameters
    ----------
    joint_prim : Usd.Prim
        The joint prim.
    n_dofs : int
        Number of degrees of freedom for the joint.

    Returns
    -------
    dict
        Dictionary with joint dynamics parameters (dofs_frictionloss, dofs_armature).
        Always contains numpy arrays (either from USD or defaults).
    """
    # Initialize with default values (only override if found in USD)
    dynamics_params = {
        "dofs_frictionloss": np.full(n_dofs, 0.0),
        "dofs_armature": np.zeros(n_dofs),
    }

    # Check for friction attribute
    friction_attr = joint_prim.GetAttribute("physics:jointFriction")
    if not friction_attr or not friction_attr.IsValid():
        friction_attr = joint_prim.GetAttribute("physics:friction")
    if not friction_attr or not friction_attr.IsValid():
        friction_attr = joint_prim.GetAttribute("jointFriction")
    if not friction_attr or not friction_attr.IsValid():
        friction_attr = joint_prim.GetAttribute("friction")

    if friction_attr and friction_attr.IsValid() and friction_attr.HasAuthoredValue():
        friction = friction_attr.Get()
        if friction is not None:
            dynamics_params["dofs_frictionloss"] = np.full((n_dofs,), float(friction))

    # Check for armature attribute
    armature_attr = joint_prim.GetAttribute("physics:armature")
    if not armature_attr or not armature_attr.IsValid():
        armature_attr = joint_prim.GetAttribute("armature")

    if armature_attr and armature_attr.IsValid() and armature_attr.HasAuthoredValue():
        armature = armature_attr.Get()
        if armature is not None:
            dynamics_params["dofs_armature"] = np.full((n_dofs,), float(armature))

    return dynamics_params


def _parse_drive_api(joint_prim: Usd.Prim, joint_type: str, n_dofs: int) -> Dict:
    """
    Parse UsdPhysics.DriveAPI attributes from a joint prim.

    Including:
    - Stiffness (maps to dofs_stiffness)
    - Damping (maps to dofs_damping)
    - Max Force (maps to dofs_force_range - max force range)

    Parameters
    ----------
    joint_prim : Usd.Prim
        The joint prim.
    joint_type : str
        The joint type (REVOLUTE, PRISMATIC, SPHERICAL, etc.).
    n_dofs : int
        Number of degrees of freedom for the joint.

    Returns
    -------
    dict
        Dictionary with drive parameters (dofs_stiffness, dofs_damping, dofs_kp, dofs_kv, dofs_force_range).
        Always contains numpy arrays (either from DriveAPI or defaults).
    """
    # Initialize with default values
    drive_params = {
        "dofs_stiffness": np.full((n_dofs,), fill_value=0.0),
        "dofs_damping": np.full((n_dofs,), fill_value=0.0),
        "dofs_force_range": np.tile([-np.inf, np.inf], (n_dofs, 1)),
    }

    # Determine the primary drive name based on joint type
    # For revolute and spherical joints, use "angular" drive
    # For prismatic joints, use "linear" drive
    if joint_type == gs.JOINT_TYPE.REVOLUTE or joint_type == gs.JOINT_TYPE.SPHERICAL:
        primary_drive_name = "angular"
        fallback_drive_names = ["linear"]  # Try linear as fallback
    elif joint_type == gs.JOINT_TYPE.PRISMATIC:
        primary_drive_name = "linear"
        fallback_drive_names = ["angular"]  # Try angular as fallback
    else:
        # For fixed or other joint types, try both
        primary_drive_name = "angular"
        fallback_drive_names = ["linear"]

    # Try primary drive name first, then fallbacks
    drive_names_to_try = [primary_drive_name] + fallback_drive_names
    drive_api = None

    for drive_name in drive_names_to_try:
        if joint_prim.HasAPI(UsdPhysics.DriveAPI, drive_name):
            drive_api = UsdPhysics.DriveAPI(joint_prim, drive_name)
            break

    # If no DriveAPI found, return defaults
    if drive_api is None:
        return drive_params

    # Extract stiffness (maps to both dofs_stiffness and dofs_kp for PD control)
    stiffness_attr = drive_api.GetStiffnessAttr()
    if stiffness_attr and stiffness_attr.HasAuthoredValue():
        stiffness = stiffness_attr.Get()
        if stiffness is not None:
            stiffness_val = float(stiffness)
            # For multi-DOF joints (like spherical), apply to all DOFs
            drive_params["dofs_stiffness"] = np.full((n_dofs,), stiffness_val, dtype=gs.np_float)

    # Extract damping (maps to both dofs_damping and dofs_kv for PD control)
    damping_attr = drive_api.GetDampingAttr()
    if damping_attr and damping_attr.HasAuthoredValue():
        damping = damping_attr.Get()
        if damping is not None:
            damping_val = float(damping)
            # For multi-DOF joints (like spherical), apply to all DOFs
            drive_params["dofs_damping"] = np.full((n_dofs,), damping_val, dtype=gs.np_float)

    # Extract maxForce (maps to dofs_force_range)
    max_force_attr = drive_api.GetMaxForceAttr()
    if max_force_attr and max_force_attr.HasAuthoredValue():
        max_force = max_force_attr.Get()
        if max_force is not None:
            max_force_val = float(max_force)
            # Convert single maxForce value to range [-maxForce, maxForce]
            # For multi-DOF joints (like spherical), apply to all DOFs
            drive_params["dofs_force_range"] = np.tile([-max_force_val, max_force_val], (n_dofs, 1))

    return drive_params


def _parse_joint_target(joint_prim: Usd.Prim, joint_type: str) -> np.ndarray | None:
    """
    Parse the target value from UsdPhysics.DriveAPI to set initial joint position.
    The target in USD is relative to the lower limit, so we add the lower limit to get the absolute position.

    Parameters
    ----------
    joint_prim : Usd.Prim
        The joint prim.
    joint_type : str
        The joint type (REVOLUTE, PRISMATIC, SPHERICAL, etc.).

    Returns
    -------
    np.ndarray or None
        Target value as numpy array if found, None otherwise.
        For revolute joints: target in radians (scalar), relative to lower limit
        For prismatic joints: target in linear units (scalar), relative to lower limit
        For spherical joints: target as quaternion (4 elements), absolute
    """
    # Determine the primary drive name based on joint type
    # For revolute and spherical joints, use "angular" drive
    # For prismatic joints, use "linear" drive
    drive_name = "linear"
    if joint_type == gs.JOINT_TYPE.REVOLUTE or joint_type == gs.JOINT_TYPE.SPHERICAL:
        drive_name = "angular"
    elif joint_type == gs.JOINT_TYPE.PRISMATIC:
        drive_name = "linear"

    drive_api = None

    if joint_prim.HasAPI(UsdPhysics.DriveAPI, drive_name):
        drive_api = UsdPhysics.DriveAPI(joint_prim, drive_name)

    if drive_api is None:
        return None

    # Extract target value
    target_attr = drive_api.GetTargetPositionAttr()
    if not target_attr or not target_attr.IsValid():
        return None

    target = target_attr.Get()

    if joint_type == gs.JOINT_TYPE.SPHERICAL:
        if not isinstance(target, collections.abc.Sequence) or len(target) != 4:
            gs.raise_exception(f"Spherical joint target at {joint_prim.GetPath()} is not a quaternion.")
        return usd_quat_to_numpy(target)
    elif joint_type == gs.JOINT_TYPE.REVOLUTE:
        return np.array([np.deg2rad(target)], dtype=gs.np_float)
    elif joint_type == gs.JOINT_TYPE.PRISMATIC:
        return np.array([target], dtype=gs.np_float)
    else:
        gs.logger.warning(f"Unsupported joint type: {joint_type} for target parsing. Ignoring target value.")
        return None


def _get_parent_child_links(stage: Usd.Stage, joint: UsdPhysics.Joint) -> Tuple[Usd.Prim, Usd.Prim]:
    """
    Get the parent and child links from a joint.

    Parameters
    ----------
    stage : Usd.Stage
        The USD stage.
    joint : UsdPhysics.Joint
        The joint.
    """
    body0_targets = joint.GetBody0Rel().GetTargets()  # optional target
    body1_targets = joint.GetBody1Rel().GetTargets()  # mandatory target

    parent_link: Usd.Prim = None
    child_link: Usd.Prim = None

    if body0_targets and len(body0_targets) > 0:
        parent_link = stage.GetPrimAtPath(body0_targets[0])

    if body1_targets and len(body1_targets) > 0:
        child_link = stage.GetPrimAtPath(body1_targets[0])

    return parent_link, child_link


# ==================== Finding Functions ====================


def _find_all_rigid_entities(stage: Usd.Stage, context: UsdParserContext = None) -> Dict[str, List[Usd.Prim]]:
    """
    Find all articulation roots and rigid bodies in the stage.

    Rigid bodies are treated as articulation roots with no child links.
    This function distinguishes them at the finding level but they will be
    processed similarly in the parsing part.

    Parameters
    ----------
    stage : Usd.Stage
        The USD stage.
    context : UsdParserContext, optional
        If provided, articulation roots and rigid bodies will be added to the context.

    Returns
    -------
    Dict[str, List[Usd.Prim]]
        Dictionary with keys:
        - "articulation_roots": List of articulation root prims
        - "rigid_bodies": List of rigid body prims
    """
    articulation_roots = []
    rigid_bodies = []

    # Use Usd.PrimRange for traversal
    it = iter(Usd.PrimRange(stage.GetPseudoRoot()))
    for prim in it:
        # Early break if we come across an ArticulationRootAPI (don't go deeper)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(prim)
            if context:
                context.add_articulation_root(prim)
            # Skip descendants (they are part of this articulation)
            it.PruneChildren()
            continue

        # Early break if we come across a rigid body
        if _is_rigid_body(prim):
            rigid_bodies.append(prim)
            if context:
                context.add_rigid_body(prim)
            # Skip descendants (they will be merged, not treated as separate rigid bodies)
            it.PruneChildren()

    return {
        "articulation_roots": articulation_roots,
        "rigid_bodies": rigid_bodies,
    }


# ==================== Collection Functions: Joints and Links ====================


def _collect_joints(root_prim: Usd.Prim) -> Dict[str, List]:
    """
    Collect all joints in the articulation subtree.

    Parameters
    ----------
    root_prim : Usd.Prim
        The root prim of the articulation or rigid body.

    Returns
    -------
    Dict[str, List]
        Dictionary with keys:
        - "joints": List of all UsdPhysics.Joint
        - "fixed_joints": List of UsdPhysics.FixedJoint
        - "revolute_joints": List of UsdPhysics.RevoluteJoint
        - "prismatic_joints": List of UsdPhysics.PrismaticJoint
        - "spherical_joints": List of UsdPhysics.SphericalJoint
    """
    joints = []
    fixed_joints = []
    revolute_joints = []
    prismatic_joints = []
    spherical_joints = []

    for child in Usd.PrimRange(root_prim):
        if child.IsA(UsdPhysics.Joint):
            joint_api = UsdPhysics.Joint(child)
            joints.append(joint_api)
            if child.IsA(UsdPhysics.RevoluteJoint):
                revolute_joint_api = UsdPhysics.RevoluteJoint(child)
                revolute_joints.append(revolute_joint_api)
            elif child.IsA(UsdPhysics.FixedJoint):
                fixed_joint_api = UsdPhysics.FixedJoint(child)
                fixed_joints.append(fixed_joint_api)
            elif child.IsA(UsdPhysics.PrismaticJoint):
                prismatic_joint_api = UsdPhysics.PrismaticJoint(child)
                prismatic_joints.append(prismatic_joint_api)
            elif child.IsA(UsdPhysics.SphericalJoint):
                spherical_joint_api = UsdPhysics.SphericalJoint(child)
                spherical_joints.append(spherical_joint_api)

    return {
        "joints": joints,
        "fixed_joints": fixed_joints,
        "revolute_joints": revolute_joints,
        "prismatic_joints": prismatic_joints,
        "spherical_joints": spherical_joints,
    }


def _collect_links(stage: Usd.Stage, joints: List[UsdPhysics.Joint]) -> List[Usd.Prim]:
    """
    Collect all links connected by joints.

    Parameters
    ----------
    stage : Usd.Stage
        The USD stage.
    joints : List[UsdPhysics.Joint]
        List of joints to extract links from.

    Returns
    -------
    List[Usd.Prim]
        List of link prims.
    """
    links = []
    paths = set()
    for joint in joints:
        body0_targets = joint.GetBody0Rel().GetTargets()
        body1_targets = joint.GetBody1Rel().GetTargets()
        for target_path in body0_targets + body1_targets:
            # Check target is valid
            if stage.GetPrimAtPath(target_path):
                paths.add(target_path)
            else:
                gs.raise_exception(f"Joint {joint.GetPath()} has invalid target body reference {target_path}.")
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        links.append(prim)
    return links


def _is_fixed_rigid_body(prim: Usd.Prim) -> bool:
    """
    Check if a rigid body prim is fixed (kinematic or collision-only).

    Parameters
    ----------
    prim : Usd.Prim
        The rigid body prim.

    Returns
    -------
    bool
        True if the rigid body is fixed, False otherwise.
    """
    collision_api_only = prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.HasAPI(UsdPhysics.RigidBodyAPI)
    kinematic_enabled = False
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
        kinematic_enabled = bool(rigid_body_api.GetKinematicEnabledAttr().Get())
    return collision_api_only or kinematic_enabled


def _parse_joints(
    stage: Usd.Stage,
    joints: List[UsdPhysics.Joint],
    l_infos: List[Dict],
    links_j_infos: List[List[Dict]],
    link_name_to_idx: Dict,
):
    """
    Parse all joints and update link transforms.

    Parameters
    ----------
    stage : Usd.Stage
        The USD stage.
    joints : List[UsdPhysics.Joint]
        List of joints to parse.
    l_infos : List[Dict]
        List of link info dictionaries.
    links_j_infos : List[List[Dict]]
        List of lists of joint info dictionaries.
    link_name_to_idx : Dict
        Dictionary mapping link paths to indices.
    """
    for joint in joints:
        parent_link, child_link = _get_parent_child_links(stage, joint)
        child_link_path = child_link.GetPath()

        idx = link_name_to_idx.get(child_link_path)
        if idx is None:
            gs.raise_exception(f"Joint {joint.GetPath()} references unknown child link {child_link_path}.")

        l_info = l_infos[idx]

        trans_mat, _ = compute_gs_relative_transform(child_link, parent_link)

        l_info["pos"] = trans_mat[:3, 3]
        l_info["quat"] = gu.R_to_quat(trans_mat[:3, :3])

        if parent_link:
            parent_link_path = parent_link.GetPath()
            l_info["parent_idx"] = link_name_to_idx.get(parent_link_path, -1)

        j_info = dict()
        links_j_infos[idx].append(j_info)

        j_info["name"] = str(joint.GetPath())
        j_info["sol_params"] = gu.default_solver_params()
        joint_prim = joint.GetPrim()

        if joint_prim.IsA(UsdPhysics.RevoluteJoint):
            revolute_joint = UsdPhysics.RevoluteJoint(joint_prim)
            joint_type_info = _parse_revolute_joint(revolute_joint, parent_link, child_link)
            j_info.update(joint_type_info)
        elif joint_prim.IsA(UsdPhysics.PrismaticJoint):
            prismatic_joint = UsdPhysics.PrismaticJoint(joint_prim)
            joint_type_info = _parse_prismatic_joint(prismatic_joint, parent_link, child_link)
            j_info.update(joint_type_info)
        elif joint_prim.IsA(UsdPhysics.SphericalJoint):
            spherical_joint = UsdPhysics.SphericalJoint(joint_prim)
            joint_type_info = _parse_spherical_joint(spherical_joint, parent_link, child_link)
            j_info.update(joint_type_info)
        else:
            if not joint_prim.IsA(UsdPhysics.FixedJoint):
                gs.logger.warning(
                    f"Unsupported USD joint type: <{joint_prim.GetTypeName()}> in joint {joint_prim.GetPath()}. "
                    "Treating as fixed joint."
                )
            joint_type_info = _parse_fixed_joint(joint_prim, parent_link, child_link)
            j_info.update(joint_type_info)

        n_dofs = j_info["n_dofs"]
        n_qs = j_info["n_qs"]

        # NOTE: Because we don't implement all the joint physics properties, we need to finalize the joint info with
        # common properties.
        # TODO: Implement all the joint physics properties.
        j_info["dofs_invweight"] = np.full((n_dofs,), fill_value=-1.0)

        # Default values
        j_info["dofs_frictionloss"] = np.full((n_dofs,), fill_value=0.0)
        j_info["dofs_damping"] = np.full((n_dofs,), fill_value=0.0)
        j_info["dofs_armature"] = np.full((n_dofs,), fill_value=0.0)
        j_info["dofs_kp"] = np.full((n_dofs,), fill_value=0.0)
        j_info["dofs_kv"] = np.full((n_dofs,), fill_value=0.0)
        j_info["dofs_force_range"] = np.tile([-np.inf, np.inf], (n_dofs, 1))
        j_info["dofs_stiffness"] = np.full((n_dofs,), fill_value=0.0)

        target = _parse_joint_target(joint_prim, j_info["type"])
        if target is not None:
            if target.shape[0] == n_dofs:
                # TODO: Implement target solving in rigid solver.
                j_info["dofs_target"] = target
            else:
                gs.raise_exception(f"Joint target at {joint_prim.GetPath()} has shape {target.shape}")

        j_info.update(_parse_joint_dynamics(joint_prim, n_dofs))
        j_info.update(_parse_drive_api(joint_prim, j_info["type"], n_dofs))


def _parse_link(link: Usd.Prim) -> Dict:
    """
    Parse a link and return a link info dictionary.
    """
    l_info = dict()
    l_info["name"] = str(link.GetPath())
    l_info["parent_idx"] = -1  # No parent by default, will be overwritten later if appropriate

    Q, S = compute_gs_global_transform(link)
    global_pos = Q[:3, 3]
    global_quat = gu.R_to_quat(Q[:3, :3])
    l_info["pos"] = global_pos
    l_info["quat"] = global_quat
    l_info["is_fixed"] = _is_fixed_rigid_body(link)

    # We will compute the inertial information later based on the geometry
    l_info["invweight"] = np.full((2,), fill_value=-1.0)
    l_info["inertial_pos"] = gu.zero_pos()
    l_info["inertial_quat"] = gu.identity_quat()
    l_info["inertial_i"] = None
    l_info["inertial_mass"] = None
    return l_info


# ==================== Main Parsing Function ====================


def parse_usd_rigid_entity(morph: gs.morphs.USD, surface: gs.surfaces.Surface):
    """
    Unified parser for USD rigid entities (both articulations and rigid bodies).

    Treats rigid bodies as articulation roots with no child links.
    Automatically detects whether the prim is an articulation (has joints) or
    a rigid body (no joints) and processes accordingly.

    Parameters
    ----------
    morph : gs.morphs.USD
        USD morph configuration.
    surface : gs.surfaces.Surface
        Surface configuration.

    Returns
    -------
    l_infos : list
        List of link info dictionaries.
    links_j_infos : list
        List of lists of joint info dictionaries.
    links_g_infos : list
        List of lists of geometry info dictionaries.
    eqs_info : list
        List of equality constraint info dictionaries.
    """
    # Validate scale
    if morph.scale is not None and morph.scale != 1.0:
        gs.logger.warning("USD rigid entity parsing currently only supports scale=1.0. Scale will be set to 1.0.")
    morph.scale = 1.0

    assert morph.parser_ctx is not None, "USDRigidEntity must have a parser context."
    assert morph.prim_path is not None, "USDRigidEntity must have a prim path."

    context: UsdParserContext = morph.parser_ctx
    stage: Usd.Stage = context.stage
    root_prim: Usd.Prim = stage.GetPrimAtPath(Sdf.Path(morph.prim_path))
    assert root_prim.IsValid(), f"Invalid prim path {morph.prim_path} in USD file {morph.file}."

    # Validate that the prim is either an articulation root or a rigid body
    is_articulation_root = root_prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    is_rigid_body = _is_rigid_body(root_prim)

    if not is_articulation_root and not is_rigid_body:
        gs.raise_exception(
            f"Provided prim {root_prim.GetPath()} is neither an articulation root nor a rigid body. "
            f"APIs found: {root_prim.GetAppliedSchemas()}"
        )

    gs.logger.debug(f"Parsing USD rigid entity from {root_prim.GetPath()}.")

    joints = []
    if is_articulation_root:
        joint_data = _collect_joints(root_prim)
        joints = joint_data["joints"]

    has_joints = len(joints) > 0

    if has_joints:
        links = _collect_links(stage, joints)
        link_name_to_idx = {link.GetPath(): idx for idx, link in enumerate(links)}
    else:
        links = [root_prim]

    n_links = len(links)
    l_infos = []
    links_j_infos = [[] for _ in range(n_links)]
    links_g_infos = [[] for _ in range(n_links)]

    for link, link_g_infos in zip(links, links_g_infos):
        l_info = _parse_link(link)
        l_infos.append(l_info)
        visual_g_infos = _create_visual_geo_infos(link, context)
        collision_g_infos = _create_collision_geo_infos(link, context)
        if len(visual_g_infos) == 0 and len(collision_g_infos) == 0:
            gs.logger.warning(f"No visual or collision geometries found for link {link.GetPath()}, skipping.")
            continue
        if len(collision_g_infos) == 0:
            gs.logger.warning(
                f"No collision geometries found for link {link.GetPath()}, using visual geometries instead."
            )
        link_g_infos.extend(visual_g_infos)
        link_g_infos.extend(collision_g_infos)

    if has_joints:
        _parse_joints(stage, joints, l_infos, links_j_infos, link_name_to_idx)

    for l_info, link_j_infos in zip(l_infos, links_j_infos):
        if l_info["parent_idx"] == -1 and len(link_j_infos) == 0:
            j_info = _create_joint_info_for_base_link(l_info)
            link_j_infos.append(j_info)

    if has_joints:
        l_infos, links_j_infos, links_g_infos, _ = urdf_utils._order_links(l_infos, links_j_infos, links_g_infos)

    # USD doesn't support equality constraints.
    eqs_info = []

    return l_infos, links_j_infos, links_g_infos, eqs_info


# ==================== Stage-Level Parsing Function ====================


def parse_all_rigid_entities(
    scene: gs.Scene,
    stage: Usd.Stage,
    context: UsdParserContext,
    usd_morph: USD,
    vis_mode: Literal["visual", "collision"],
    visualize_contact: bool = False,
) -> Dict[str, "Entity"]:
    """
    Find and parse all rigid entities (articulations and rigid bodies) from a USD stage.

    Parameters
    ----------
    scene : gs.Scene
        The scene to add entities to.
    stage : Usd.Stage
        The USD stage.
    context : UsdParserContext
        The parser context.
    usd_morph : USD
        USD morph configuration.
    vis_mode : Literal["visual", "collision"]
        Visualization mode.
    visualize_contact : bool, optional
        Whether to visualize contact, by default False.

    Returns
    -------
    Dict[str, Entity]
        Dictionary of created entities (both articulations and rigid bodies) keyed by prim path.
    """
    from genesis.engine.entities.base_entity import Entity as GSEntity

    entities: Dict[str, GSEntity] = {}

    # Find all rigid entities (articulations and rigid bodies)
    rigid_entities = _find_all_rigid_entities(stage, context)
    articulation_roots = rigid_entities["articulation_roots"]
    rigid_bodies = rigid_entities["rigid_bodies"]

    gs.logger.debug(
        f"Found {len(articulation_roots)} articulation(s) and {len(rigid_bodies)} rigid body(ies) in USD stage."
    )

    # Use material with density=1000.0 to match MuJoCo's default density when computing inertia from geometry
    # Usd can rewrite the density of the material later
    material = gs.materials.Rigid(rho=1000.0)

    # Process articulation roots
    for articulation_root in articulation_roots:
        morph = copy.copy(usd_morph)
        morph.prim_path = str(articulation_root.GetPath())
        morph.parsing_type = "articulation"
        entity = scene.add_entity(morph, material=material, vis_mode=vis_mode, visualize_contact=visualize_contact)
        entities[str(articulation_root.GetPath())] = entity
        gs.logger.debug(f"Imported articulation from prim: {articulation_root.GetPath()}")

    # Process rigid bodies (treated as articulation roots with no child links)
    for rigid_body_prim in rigid_bodies:
        morph = copy.copy(usd_morph)
        morph.prim_path = str(rigid_body_prim.GetPath())
        morph.parsing_type = "rigid_body"
        entity = scene.add_entity(morph, material=material, vis_mode=vis_mode, visualize_contact=visualize_contact)
        entities[str(rigid_body_prim.GetPath())] = entity
        gs.logger.debug(f"Imported rigid body from prim: {rigid_body_prim.GetPath()}")

    return entities
