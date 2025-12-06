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
from genesis.utils.usd.usd_parser_context import UsdParserContext
from genesis.utils.usd.usd_parser_utils import (
    bfs_iterator,
    compute_gs_related_transform,
    extract_quat_from_transform,
    compute_gs_global_transform,
    convert_usd_joint_axis_to_gs,
    usd_quat_to_numpy,
    convert_usd_joint_pos_to_gs,
)
from genesis.utils.usd.usd_geo_adapter import BatchedUsdGeometryAdapater, UsdGeometryAdapter
from genesis.utils import geom as gu


class UsdArticulationParser:
    """
    A parser to extract articulation information from a USD stage.
    The Parser is agnostic to genesis structures, it only focuses on USD articulation structure.
    """

    class MeshLikePrims:
        def __init__(self):
            self.meshes: List[UsdGeom.Mesh] = []
            self.planes: List[UsdGeom.Plane] = []
            self.spheres: List[UsdGeom.Sphere] = []
            self.capsules: List[UsdGeom.Capsule] = []
            self.cubes: List[UsdGeom.Cube] = []

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


def _create_joint_info(joint: UsdPhysics.Joint) -> Dict:
    """
    Create a joint info dictionary with default values.

    Parameters
    ----------
    joint : UsdPhysics.Joint
        The joint API.

    Returns
    -------
    dict
        Joint info dictionary with default values.
    """
    j_info = dict()
    j_info["name"] = joint.GetPath()
    j_info["sol_params"] = gu.default_solver_params()
    return j_info


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
    for link, l_info, link_g_infos in zip(robot.links, l_infos, links_g_infos):
        visual_g_infos = UsdArticulationParser.get_visual_geometries(link, context)
        collision_g_infos = UsdArticulationParser.get_collision_geometries(link, context)

        if len(visual_g_infos) == 0 and len(collision_g_infos) == 0:
            continue

        # Add all visual geometries
        link_g_infos.extend(visual_g_infos)

        # Add all collision geometries
        link_g_infos.extend(collision_g_infos)

    # Parse joints and update link transforms
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
        j_info = _create_joint_info(joint)

        links_j_infos[idx].append(j_info)

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

        # Finalize joint info with common properties
        # NOTE: Cuz we don't implement all the joint physics properties, we need to finalize the joint info with common properties.
        # TODO: Implement all the joint physics properties.
        n_dofs = j_info["n_dofs"]
        j_info["dofs_invweight"] = np.full((n_dofs,), fill_value=-1.0)
        j_info["dofs_frictionloss"] = np.full(n_dofs, 0.0)
        j_info["dofs_damping"] = np.full(n_dofs, 0.0)
        j_info["dofs_armature"] = np.zeros(n_dofs)
        j_info["dofs_kp"] = gu.default_dofs_kp(n_dofs)
        j_info["dofs_kv"] = gu.default_dofs_kv(n_dofs)
        j_info["dofs_force_range"] = np.tile([-np.inf, np.inf], (n_dofs, 1))

    # Add FREE joint to base links that have no incoming joints
    for idx, (l_info, link_j_infos) in enumerate(zip(l_infos, links_j_infos)):
        # Base link (no parent) with no joints should get a FREE joint
        if l_info["parent_idx"] == -1 and len(link_j_infos) == 0:
            j_info = _create_free_joint_for_base_link(l_info)
            link_j_infos.append(j_info)

    # Post-process: Re-order kinematic tree info
    from .. import urdf as urdf_utils

    l_infos, links_j_infos, links_g_infos, _ = urdf_utils._order_links(l_infos, links_j_infos, links_g_infos)

    # For now, no equalities
    eqs_info = []

    return l_infos, links_j_infos, links_g_infos, eqs_info
