"""
USD Articulation Parser

Parser for extracting articulation information from USD stages.
The parser is agnostic to genesis structures, focusing only on USD articulation structure.

Also includes Genesis-specific parsing functions that translate USD structures into Genesis info structures.
"""

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf
from typing import List, Dict, Tuple, Literal
import genesis as gs
import numpy as np
import re
from scipy.spatial.transform import Rotation as R
from .usd_parser_context import UsdParserContext
from .usd_parser_utils import (
    bfs_iterator,
    compute_gs_related_transform,
    extract_quat_from_transform,
    compute_gs_global_transform,
    convert_usd_joint_axis_to_gs,
    usd_quat_to_numpy,
    convert_usd_joint_pos_to_gs,
)
from .usd_geo_adapter import BatchedUsdGeometryAdapater, UsdGeometryAdapter
from .. import geom as gu
from .. import urdf as urdf_utils


class UsdArticulationParser:
    """
    A parser to extract articulation information from a USD stage.
    The Parser is agnostic to genesis structures, it only focuses on USD articulation structure.
    """

    def __init__(self, stage: Usd.Stage, articulation_root_prim: Usd.Prim):
        """
        Initialize the articulation parser.

        Parameters
        ----------
        stage : Usd.Stage
            The USD stage.
        articulation_root_prim : Usd.Prim
            The root prim of the articulation (must have ArticulationRootAPI).
        """
        self._stage: Usd.Stage = stage
        self._root: Usd.Prim = articulation_root_prim
        if not articulation_root_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            gs.raise_exception(
                f"Provided prim {articulation_root_prim.GetPath()} is not an Articulation Root. Now we only support articulation parsing from ArticulationRootAPI."
            )

        gs.logger.info(f"Parsing USD articulation from {articulation_root_prim.GetPath()}.")

        self.joints: List[UsdPhysics.Joint] = []
        self.fixed_joints: List[UsdPhysics.FixedJoint] = []
        self.revolute_joints: List[UsdPhysics.RevoluteJoint] = []
        self.prismatic_joints: List[UsdPhysics.PrismaticJoint] = []
        self.spherical_joints: List[UsdPhysics.SphericalJoint] = []
        self._collect_joints()

        self.links: List[Usd.Prim] = []
        self._collect_links()

    # ==================== Static Methods: Finding Articulation Roots ====================

    @staticmethod
    def find_all_articulation_roots(stage: Usd.Stage, context: UsdParserContext = None) -> List[Usd.Prim]:
        """
        Find all prims with ArticulationRootAPI in the stage.

        Parameters
        ----------
        stage : Usd.Stage
            The USD stage.
        context : UsdParserContext, optional
            If provided, articulation roots will be added to the context.

        Returns
        -------
        List[Usd.Prim]
            List of articulation root prims.
        """
        articulation_roots = []
        for prim in bfs_iterator(stage.GetPseudoRoot()):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulation_roots.append(prim)
                if context:
                    context.add_articulation_root(prim)
        return articulation_roots

    # ==================== Collection Methods: Joints and Links ====================

    def _collect_joints(self):
        """Collect all joints in the articulation."""
        for child in bfs_iterator(self._root):
            if child.IsA(UsdPhysics.Joint):
                joint_api = UsdPhysics.Joint(child)
                self.joints.append(joint_api)
                if child.IsA(UsdPhysics.RevoluteJoint):
                    revolute_joint_api = UsdPhysics.RevoluteJoint(child)
                    self.revolute_joints.append(revolute_joint_api)
                elif child.IsA(UsdPhysics.FixedJoint):
                    fixed_joint_api = UsdPhysics.FixedJoint(child)
                    self.fixed_joints.append(fixed_joint_api)
                elif child.IsA(UsdPhysics.PrismaticJoint):
                    prismatic_joint_api = UsdPhysics.PrismaticJoint(child)
                    self.prismatic_joints.append(prismatic_joint_api)
                elif child.IsA(UsdPhysics.SphericalJoint):
                    spherical_joint_api = UsdPhysics.SphericalJoint(child)
                    self.spherical_joints.append(spherical_joint_api)

    def _collect_links(self):
        """Collect all links connected by joints in the articulation."""
        # Now we have joints collected, we can find links connected by these joints
        paths = set()
        for joint in self.joints:
            body0_targets = joint.GetBody0Rel().GetTargets()
            body1_targets = joint.GetBody1Rel().GetTargets()
            for target_path in body0_targets + body1_targets:
                # Check target is valid
                if self._stage.GetPrimAtPath(target_path):
                    paths.add(target_path)
                else:
                    gs.raise_exception(f"Joint {joint.GetPath()} has invalid target body reference {target_path}.")
        for path in paths:
            prim = self._stage.GetPrimAtPath(path)
            self.links.append(prim)

    # ==================== Geometry Collection Methods ====================

    visual_pattern = re.compile(r"^(visual|Visual).*")
    collision_pattern = re.compile(r"^(collision|Collision).*")
    all_pattern = re.compile(r"^.*")

    @staticmethod
    def create_gs_geo_infos(
        context: UsdParserContext, link: Usd.Prim, pattern, mesh_type: Literal["mesh", "vmesh"]
    ) -> List[Dict]:
        # if the link itself is a geometry
        geo_infos: List[Dict] = []
        link_geo_adapter = UsdGeometryAdapter(context, link, link, mesh_type)
        link_geo_info = link_geo_adapter.create_gs_geo_info()
        if link_geo_info is not None:
            geo_infos.append(link_geo_info)

        # - Link
        #     - Visuals
        #     - Collisions
        search_roots: list[Usd.Prim] = []
        for child in link.GetChildren():
            if pattern.match(child.GetName()):
                search_roots.append(child)

        for search_root in search_roots:
            adapter = BatchedUsdGeometryAdapater(context, search_root, link, mesh_type)
            geo_infos.extend(adapter.create_gs_geo_infos())

        return geo_infos

    @staticmethod
    def get_visual_geometries(link: Usd.Prim, context: UsdParserContext) -> List[Dict]:
        if context.vis_mode == "visual":
            vis_geo_infos = UsdArticulationParser.create_gs_geo_infos(
                context, link, UsdArticulationParser.visual_pattern, "vmesh"
            )
            if len(vis_geo_infos) == 0:
                # if no visual geometries found, use any pattern to find visual geometries
                gs.logger.info(
                    f"No visual geometries found, using any pattern to find visual geometries in {link.GetPath()}"
                )
                vis_geo_infos = UsdArticulationParser.create_gs_geo_infos(
                    context, link, UsdArticulationParser.all_pattern, "vmesh"
                )
        elif context.vis_mode == "collision":
            vis_geo_infos = UsdArticulationParser.create_gs_geo_infos(
                context, link, UsdArticulationParser.collision_pattern, "vmesh"
            )
        else:
            gs.raise_exception(f"Unsupported visualization mode {context.vis_mode}.")
        return vis_geo_infos

    @staticmethod
    def get_collision_geometries(link: Usd.Prim, context: UsdParserContext) -> List[Dict]:
        col_geo_infos = UsdArticulationParser.create_gs_geo_infos(
            context, link, UsdArticulationParser.collision_pattern, "mesh"
        )
        return col_geo_infos


# ==================== Helper Functions for Genesis Parsing ====================


def _axis_str_to_vector(axis_str: str) -> np.ndarray:
    """
    Convert a joint axis string to a vector.

    Parameters
    ----------
    axis_str : str
        The axis string ('X', 'Y', or 'Z').
    """
    if axis_str == "X":
        return np.array([1.0, 0.0, 0.0])
    elif axis_str == "Y":
        return np.array([0.0, 1.0, 0.0])
    elif axis_str == "Z":
        return np.array([0.0, 0.0, 1.0])
    else:
        gs.raise_exception(f"Unsupported joint axis {axis_str}.")


def _compute_child_link_local_axis_pos(
    joint: UsdPhysics.PrismaticJoint | UsdPhysics.RevoluteJoint, child_link: Usd.Prim
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the local axis and position of a joint in the parent link local space.

    Parameters
    ----------
    joint : UsdPhysics.Joint
    parent_link : Usd.Prim
    """
    axis_attr = joint.GetAxisAttr()
    axis_str = axis_attr.Get() if axis_attr else "X"
    pos_attr = joint.GetLocalPos1Attr()
    axis = _axis_str_to_vector(axis_str)
    usd_local_pos = pos_attr.Get() if pos_attr else gu.zero_pos()

    # rotate the orth-normal axis (X/Y/Z) to the local space
    rotation_attr = joint.GetLocalRot1Attr()
    usd_local_rotation = usd_quat_to_numpy(rotation_attr.Get()) if rotation_attr.Get() else gu.identity_quat()
    usd_local_axis = gu.quat_to_R(usd_local_rotation) @ axis

    # convert to gs
    gs_local_axis = convert_usd_joint_axis_to_gs(usd_local_axis, child_link)
    gs_local_pos = convert_usd_joint_pos_to_gs(usd_local_pos, child_link)
    return gs_local_axis, gs_local_pos


def _compute_child_link_local_pos(joint: UsdPhysics.SphericalJoint, child_link: Usd.Prim) -> np.ndarray:
    """
    Compute the local position of a spherical joint in the parent link local space.

    Parameters
    ----------
    joint : UsdPhysics.SphericalJoint
    parent_link : Usd.Prim
    """
    pos_attr = joint.GetLocalPos1Attr()
    usd_local_pos = pos_attr.Get() if pos_attr else gu.zero_pos()
    gs_local_pos = convert_usd_joint_pos_to_gs(usd_local_pos, child_link)
    return gs_local_pos


def _create_link_info(link: Usd.Prim) -> Dict:
    """
    Create a basic link info dictionary with default values.

    Parameters
    ----------
    link : Usd.Prim
        The link prim.

    Returns
    -------
    dict
        Link info dictionary with default values.
    """
    l_info = dict()
    l_info["name"] = link.GetPath()
    l_info["parent_idx"] = -1  # No parent by default, will be overwritten later if appropriate

    Q, S = compute_gs_global_transform(link)

    global_pos = Q[:3, 3]
    global_quat = gu.R_to_quat(Q[:3, :3])
    l_info["pos"] = global_pos
    l_info["quat"] = global_quat
    l_info["invweight"] = np.full((2,), fill_value=-1.0)
    l_info["inertial_pos"] = gu.zero_pos()
    l_info["inertial_quat"] = gu.identity_quat()
    l_info["inertial_i"] = None
    l_info["inertial_mass"] = None
    return l_info


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
    spherial_joint: UsdPhysics.SphericalJoint, parent_link: Usd.Prim | None, child_link: Usd.Prim
) -> Dict:
    """
    Parse a spherical joint and create joint info dictionary.

    Parameters
    ----------
    spherial_joint : UsdPhysics.SphericalJoint
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
    joint_prim = spherial_joint.GetPrim()

    pos = _compute_child_link_local_pos(joint_prim, child_link)

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
        The body0 targets (to determine if it's a root fixed joint).

    Returns
    -------
    dict
        Joint info dictionary.
    """
    j_info = dict()

    if not parent_link:
        gs.logger.info(f"Root Fixed Joint detected {joint_prim.GetPath()}")
    else:
        gs.logger.info(f"Fixed Joint detected {joint_prim.GetPath()}")

    j_info["dofs_motion_ang"] = np.zeros((0, 3))
    j_info["dofs_motion_vel"] = np.zeros((0, 3))
    j_info["dofs_limit"] = np.zeros((0, 2))
    j_info["dofs_stiffness"] = np.zeros((0))
    j_info["type"] = gs.JOINT_TYPE.FIXED
    j_info["n_qs"] = 0
    j_info["n_dofs"] = 0
    j_info["init_qpos"] = np.zeros(0)

    return j_info


def _create_free_joint_for_base_link(l_info: Dict) -> Dict:
    """
    Create a FREE joint for base links that have no incoming joints.

    Parameters
    ----------
    l_info : dict
        Link info dictionary.

    Returns
    -------
    dict
        Joint info dictionary for FREE joint.
    """
    j_info = dict()
    # NOTE: Any naming convention for root joints?
    j_info["name"] = f"{l_info['name']}_joint"
    j_info["type"] = gs.JOINT_TYPE.FREE
    j_info["n_qs"] = 7
    j_info["n_dofs"] = 6
    j_info["init_qpos"] = np.concatenate([l_info["pos"], l_info["quat"]])

    j_info["pos"] = gu.zero_pos()
    j_info["quat"] = gu.identity_quat()
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
    j_info["sol_params"] = gu.default_solver_params()
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
        Dictionary with joint dynamics parameters (dofs_frictionloss, dofs_damping, dofs_armature).
        Always contains numpy arrays (either from USD or defaults).
    """
    # Initialize with default values
    dynamics_params = {
        "dofs_frictionloss": np.full(n_dofs, 0.0),
        "dofs_damping": np.full(n_dofs, 0.0),
        "dofs_armature": np.zeros(n_dofs),
    }

    # Check for damping attribute on the joint
    # Note: Standard USD Physics may not have damping directly on Joint,
    # but some implementations might extend it
    damping_attr = joint_prim.GetAttribute("physics:damping")
    if not damping_attr or not damping_attr.IsValid():
        # Try alternative attribute names
        damping_attr = joint_prim.GetAttribute("damping")

    if damping_attr and damping_attr.IsValid() and damping_attr.HasAuthoredValue():
        damping = damping_attr.Get()
        if damping is not None:
            dynamics_params["dofs_damping"] = np.full((n_dofs,), float(damping))

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
        Dictionary with drive parameters (dofs_kp, dofs_kv, dofs_force_range).
        Always contains numpy arrays (either from DriveAPI or defaults).
    """
    # Initialize with default values
    drive_params = {
        "dofs_kp": gu.default_dofs_kp(n_dofs),
        "dofs_kv": gu.default_dofs_kv(n_dofs),
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

    # Extract stiffness (maps to dofs_kp - position gain)
    stiffness_attr = drive_api.GetStiffnessAttr()
    if stiffness_attr and stiffness_attr.HasAuthoredValue():
        stiffness = stiffness_attr.Get()
        if stiffness is not None:
            # For multi-DOF joints (like spherical), apply to all DOFs
            drive_params["dofs_kp"] = np.full((n_dofs,), float(stiffness), dtype=gs.np_float)

    # Extract damping (maps to dofs_kv - velocity gain)
    damping_attr = drive_api.GetDampingAttr()
    if damping_attr and damping_attr.HasAuthoredValue():
        damping = damping_attr.Get()
        if damping is not None:
            # For multi-DOF joints (like spherical), apply to all DOFs
            drive_params["dofs_kv"] = np.full((n_dofs,), float(damping), dtype=gs.np_float)

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
    n_qs : int
        Number of position coordinates for the joint.
    dofs_limit : np.ndarray
        Joint limits array with shape (n_dofs, 2) where each row is [lower_limit, upper_limit].

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

    # If no DriveAPI found, return None
    if drive_api is None:
        return None

    # Extract target value
    target_attr = drive_api.GetTargetPositionAttr()
    if target_attr and target_attr.HasAuthoredValue():
        target = target_attr.Get()
        if target is not None:
            # Convert target to numpy array
            if joint_type == gs.JOINT_TYPE.SPHERICAL:
                # For spherical joints, target is absolute quaternion (not relative to lower limit)
                # Try to get as quaternion first
                if hasattr(target, "__len__") and len(target) == 4:
                    return np.array(target, dtype=gs.np_float)
                elif hasattr(target, "__len__") and len(target) == 3:
                    # If it's a rotation vector (axis-angle), convert to quaternion
                    # For now, just return identity quaternion and log warning
                    gs.logger.warning(
                        f"Spherical joint target at {joint_prim.GetPath()} is axis-angle format. "
                        "Quaternion conversion not yet implemented. Using identity quaternion."
                    )
                    return gu.identity_quat()
                else:
                    # Single value - treat as angle around some axis (not fully supported)
                    gs.logger.warning(
                        f"Spherical joint target at {joint_prim.GetPath()} has unexpected format. "
                        "Using identity quaternion."
                    )
                    return gu.identity_quat()
            else:
                # For revolute and prismatic joints, target is a scalar relative to lower limit
                target_val = float(target)

                # For revolute joints, target is typically in degrees in USD, convert to radians
                if joint_type == gs.JOINT_TYPE.REVOLUTE:
                    target_val = np.deg2rad(target_val)

                absolute_target = target_val
                return np.array([absolute_target], dtype=gs.np_float)

    return None


def _get_parent_child_links(stage: Usd.Stage, joint: UsdPhysics.Joint) -> Tuple[Usd.Prim, Usd.Prim]:
    """
    Get the parent and child links from a joint.

    Parameters
    ----------
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


# ==================== Parsing Helper Functions ====================


def _parse_link_geometries(
    robot: UsdArticulationParser, l_infos: List[Dict], links_g_infos: List[List[Dict]], context: UsdParserContext
):
    """
    Parse geometries (visual and collision) for all links.

    Parameters
    ----------
    robot : UsdArticulationParser
        The articulation parser instance.
    l_infos : List[Dict]
        List of link info dictionaries.
    links_g_infos : List[List[Dict]]
        List of lists of geometry info dictionaries.
    context : UsdParserContext
        The parser context.
    """
    for link, l_info, link_g_infos in zip(robot.links, l_infos, links_g_infos):
        visual_g_infos = UsdArticulationParser.get_visual_geometries(link, context)
        collision_g_infos = UsdArticulationParser.get_collision_geometries(link, context)
        if len(visual_g_infos) == 0 and len(collision_g_infos) == 0:
            gs.logger.warning(f"No visual or collision geometries found for link {link.GetPath()}, skipping.")
            continue
        if len(collision_g_infos) == 0:
            gs.logger.warning(
                f"No collision geometries found for link {link.GetPath()}, using visual geometries instead."
            )
        # Add all visual geometries
        link_g_infos.extend(visual_g_infos)
        # Add all collision geometries
        link_g_infos.extend(collision_g_infos)


def _parse_joints(
    robot: UsdArticulationParser,
    stage: Usd.Stage,
    l_infos: List[Dict],
    links_j_infos: List[List[Dict]],
    link_name_to_idx: Dict,
):
    """
    Parse all joints and update link transforms.

    Parameters
    ----------
    robot : UsdArticulationParser
        The articulation parser instance.
    stage : Usd.Stage
        The USD stage.
    l_infos : List[Dict]
        List of link info dictionaries.
    links_j_infos : List[List[Dict]]
        List of lists of joint info dictionaries.
    link_name_to_idx : Dict
        Dictionary mapping link paths to indices.
    """
    for joint in robot.joints:
        parent_link, child_link = _get_parent_child_links(stage, joint)
        child_link_path = child_link.GetPath()
        # Find the child link index
        idx = link_name_to_idx.get(child_link_path)
        if idx is None:
            gs.raise_exception(f"Joint {joint.GetPath()} references unknown child link {child_link_path}.")

        l_info = l_infos[idx]

        # Update link transform
        trans_mat, _ = compute_gs_related_transform(child_link, parent_link)

        l_info["pos"] = trans_mat[:3, 3]
        l_info["quat"] = extract_quat_from_transform(trans_mat)

        # Set parent link index
        if parent_link:
            parent_link_path = parent_link.GetPath()
            l_info["parent_idx"] = link_name_to_idx.get(parent_link_path, -1)

        # Common joint properties
        j_info = dict()
        links_j_infos[idx].append(j_info)

        j_info["name"] = joint.GetPath()
        j_info["sol_params"] = gu.default_solver_params()
        joint_prim = joint.GetPrim()

        # Parse joint type-specific properties
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

        # NOTE: Cuz we don't implement all the joint physics properties, we need to finalize the joint info with common properties.
        # TODO: Implement all the joint physics properties.
        j_info["dofs_invweight"] = np.full((n_dofs,), fill_value=-1.0)

        # Default values
        j_info["dofs_frictionloss"] = np.full((n_dofs,), fill_value=0.0)
        j_info["dofs_damping"] = np.full((n_dofs,), fill_value=0.0)
        j_info["dofs_armature"] = np.full((n_dofs,), fill_value=0.0)
        j_info["dofs_kp"] = np.full((n_dofs,), fill_value=0.0)
        j_info["dofs_kv"] = np.full((n_dofs,), fill_value=0.0)
        j_info["dofs_force_range"] = np.tile([-np.inf, np.inf], (n_dofs, 1))

        # Parse joint target from DriveAPI to set initial position
        # Target is relative to lower limit, so we pass dofs_limit to add it
        target = _parse_joint_target(joint_prim, j_info["type"])
        if target is not None:
            # Override init_qpos with target value if found
            if target.shape[0] == n_qs:
                j_info["dofs_stiffness"] = np.full((n_dofs,), fill_value=10.0)
                j_info["dofs_damping"] = np.full((n_dofs,), fill_value=10.0)
                j_info["init_qpos"] = np.full((n_dofs,), fill_value=0.0)
                j_info["dofs_target"] = target
            else:
                gs.logger.warning(
                    f"Joint target at {joint_prim.GetPath()} has shape {target.shape}, "
                    f"but expected {n_qs} elements. Ignoring target value."
                )

        # Parse joint dynamics properties (friction, damping, armature)
        dynamics_params = _parse_joint_dynamics(joint_prim, n_dofs)
        j_info.update(dynamics_params)

        # Parse DriveAPI
        drive_params = _parse_drive_api(joint_prim, j_info["type"], n_dofs)
        j_info.update(drive_params)


def _setup_free_joints_for_base_links(l_infos: List[Dict], links_j_infos: List[List[Dict]]):
    """
    Add FREE joints to base links that have no incoming joints.

    Parameters
    ----------
    l_infos : List[Dict]
        List of link info dictionaries.
    links_j_infos : List[List[Dict]]
        List of lists of joint info dictionaries.
    """
    for idx, (l_info, link_j_infos) in enumerate(zip(l_infos, links_j_infos)):
        # Base link (no parent) with no joints should get a FREE joint
        if l_info["parent_idx"] == -1 and len(link_j_infos) == 0:
            j_info = _create_free_joint_for_base_link(l_info)
            link_j_infos.append(j_info)


# ==================== Main Parsing Function ====================


def parse_usd_articulation(morph: gs.morphs.USDArticulation, surface: gs.surfaces.Surface):
    """
    Parse USD articulation from the given USD file and prim path, translating it into genesis structures.

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
    # Validate inputs and setup
    if morph.scale is not None and morph.scale != 1.0:
        gs.logger.warning("USD articulation parsing currently only supports scale=1.0. Scale will be set to 1.0.")
    morph.scale = 1.0

    assert morph.scale == 1.0, "Currently we only support scale=1.0 for USD articulation parsing."
    assert morph.parser_ctx is not None, "USDArticulation must have a parser context."
    assert morph.prim_path is not None, "USDArticulation must have a prim path."

    context: UsdParserContext = morph.parser_ctx
    stage: Usd.Stage = context.stage
    root_prim: Usd.Prim = stage.GetPrimAtPath(Sdf.Path(morph.prim_path))
    assert root_prim.IsValid(), f"Invalid prim path {morph.prim_path} in USD file {morph.file}."

    # Create parser and build link name mapping
    robot = UsdArticulationParser(stage, root_prim)
    link_name_to_idx = {link.GetPath(): idx for idx, link in enumerate(robot.links)}
    n_links = len(robot.links)

    # Initialize data structures
    l_infos = [_create_link_info(link) for link in robot.links]
    links_j_infos = [[] for _ in range(n_links)]
    links_g_infos = [[] for _ in range(n_links)]

    # Parse geometry for each link
    _parse_link_geometries(robot, l_infos, links_g_infos, context)

    # Parse joints and update link transforms
    _parse_joints(robot, stage, l_infos, links_j_infos, link_name_to_idx)

    # Add FREE joint to base links that have no incoming joints
    _setup_free_joints_for_base_links(l_infos, links_j_infos)

    # Order links
    l_infos, links_j_infos, links_g_infos, _ = urdf_utils._order_links(l_infos, links_j_infos, links_g_infos)

    # For now, no equalities
    eqs_info = []

    return l_infos, links_j_infos, links_g_infos, eqs_info
