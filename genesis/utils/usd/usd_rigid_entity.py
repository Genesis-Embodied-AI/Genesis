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
from typing import Dict, List, Literal, Optional, Tuple, TYPE_CHECKING

import numpy as np
from pxr import Sdf, Usd, UsdPhysics, Gf

import genesis as gs
from genesis.options.morphs import USD
from genesis.utils import geom as gu
from genesis.utils import urdf as urdf_utils

from .usd_geo_adapter import create_geo_info_from_prim, create_geo_infos_from_subtree
from .usd_context import UsdContext
from .usd_geometry import parse_prim_geoms
from .usd_parser_utils import (
    AXES_VECTOR,
    get_attr_value_by_candidates,
    compute_gs_joint_pos_from_usd_prim,
    compute_gs_joint_axis_and_posfrom_usd_prim,
    usd_pos_to_numpy,
    usd_quat_to_numpy,
)

if TYPE_CHECKING:
    from genesis.engine.entities.base_entity import Entity


# ==================== Joint/Link Default Values ====================

DRIVE_NAMES = {
    gs.JOINT_TYPE.REVOLUTE: ("angular",),
    gs.JOINT_TYPE.PRISMATIC: ("linear",),
    gs.JOINT_TYPE.SPHERICAL: ("rotX", "rotY", "rotZ"),
    gs.JOINT_TYPE.FIXED: (),
    gs.JOINT_TYPE.FREE: ("transX", "transY", "transZ", "rotX", "rotY", "rotZ"),
}



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


def _create_geo_infos(
    context: UsdParserContext, link: Usd.Prim, patterns: List[str], mesh_type: Literal["mesh", "vmesh"]
) -> List[Dict]:
    """
    Create geometry info dictionaries from a link prim and its children that match the patterns.

    Parameters
    ----------
    context : UsdParserContext
        The parser context.
    link : Usd.Prim
        The link prim.
    patterns : List[str]
        List of regex patterns to match child prim names. Patterns are tried in order.
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

    # Try each pattern in order
    for pattern in patterns:
        for child in link.GetChildren():
            child: Usd.Prim
            child_name = str(child.GetName())
            if re.match(pattern, child_name) and child not in context.link_prims:
                search_roots.append(child)
        if len(search_roots) > 0:
            break

    for search_root in search_roots:
        geo_infos.extend(create_geo_infos_from_subtree(context, search_root, link, mesh_type))

    return geo_infos


def _parse_joint_axis_pos(
    context: UsdContext, joint: UsdPhysics.Joint, child_link: Usd.Prim, is_body1: bool
) -> Tuple[np.ndarray, np.ndarray]:
    joint_pos = usd_pos_to_numpy(
        (joint.GetLocalPos1Attr() if is_body1 else joint.GetLocalPos0Attr()).Get()
    )
    T = context.compute_transform(child_link)
    joint_pos = gu.transform_by_T(joint_pos, T)
    Q, S = context.compute_gs_transform(child_link)
    Q_inv = np.linalg.inv(Q)
    joint_pos = Q_inv[:3, :3] @ (joint_pos - Q[:3, 3])

    if isinstance(joint, (UsdPhysics.PrismaticJoint, UsdPhysics.RevoluteJoint)):
        joint_quat = usd_quat_to_numpy(
            (joint.GetLocalRot1Attr() if is_body1 else joint.GetLocalRot0Attr()).Get()
        )
        joint_axis = gu.transform_by_quat(AXES_VECTOR[joint.GetAxisAttr().Get() or "X"], joint_quat) 
        joint_axis = gu.transform_by_R(joint_axis, T[:3, :3])
        joint_axis = Q_inv[:3, :3] @ joint_axis
        if np.linalg.norm(joint_axis) < gs.EPS:
            gs.raise_exception(f"Joint axis is zero for joint {joint.GetPath()}.")
        joint_axis /= np.linalg.norm(joint_axis)
    else:
        joint_axis = None

    return joint_axis, joint_pos


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
    l_name = l_info["name"]
    # NOTE: Any naming convention for base link joints?
    j_name = f"{l_name}_joint"

    if l_info["is_fixed"]:
        n_dofs = 0
        n_qs = 0
    else:
        n_dofs = 6
        n_qs = 7

    j_info = _create_joint_default_values(n_dofs)

    j_info["name"] = j_name

    if l_info["is_fixed"]:
        j_info["type"] = gs.JOINT_TYPE.FIXED
        j_info["n_qs"] = n_qs
        j_info["n_dofs"] = n_dofs
        j_info["init_qpos"] = np.zeros(0)
        j_info["dofs_motion_ang"] = np.zeros((0, 3))
        j_info["dofs_motion_vel"] = np.zeros((0, 3))
        j_info["dofs_limit"] = np.zeros((0, 2))
    else:
        j_info["type"] = gs.JOINT_TYPE.FREE
        j_info["n_qs"] = n_qs
        j_info["n_dofs"] = n_dofs
        j_info["init_qpos"] = np.concatenate([l_info["pos"], l_info["quat"]])
        j_info["dofs_motion_ang"] = np.eye(6, 3, -3)
        j_info["dofs_motion_vel"] = np.eye(6, 3)
        j_info["dofs_limit"] = np.tile([-np.inf, np.inf], (6, 1))

    return j_info




def _parse_joints(
    stage: Usd.Stage,
    joints: List[UsdPhysics.Joint],
    l_infos: List[Dict],
    links_j_infos: List[List[Dict]],
    link_name_to_idx: Dict,
    morph: gs.morphs.USD,
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
    morph : gs.morphs.USD
        USD morph configuration containing joint friction attribute candidates.
    """
    pass



def _parse_link(
    context: UsdContext,
    link: Usd.Prim,
    joints: List[Tuple(Usd.Prim, int, bool)],
    links: List[Usd.Prim],
    entity_prim: Usd.Prim,
    morph: gs.morphs.USD,
):
    """
    Parse a link and return a link info dictionary.
    """
    l_info = {}
    l_info["name"] = str(link.GetPath())
    l_info["invweight"] = np.full((2,), fill_value=-1.0)

    # parse link fixed state
    link_fixed = False
    if link.HasAPI(UsdPhysics.RigidBodyAPI):
        rigid_body_api = UsdPhysics.RigidBodyAPI(link)
        if rigid_body_api.GetKinematicEnabledAttr().Get():
            link_fixed = True
        elif rigid_body_api.GetRigidBodyEnabledAttr().Get() is False:
            link_fixed = True
    elif link.HasAPI(UsdPhysics.CollisionAPI):
        link_fixed = True

    # parse link mass properties
    if link.HasAPI(UsdPhysics.MassAPI):
        mass_api = UsdPhysics.MassAPI(link)
        l_info["inertial_pos"] = usd_pos_to_numpy(mass_api.GetCenterOfMassAttr().Get())
        l_info["inertial_quat"] = usd_quat_to_numpy(mass_api.GetPrincipalAxesAttr().Get())
        l_info["inertial_i"] = np.diag(usd_pos_to_numpy(mass_api.GetDiagonalInertiaAttr().Get()))
        l_info["inertial_mass"] = float(mass_api.GetMassAttr().Get() or 0.0)
    else:
        l_info["inertial_pos"] = gu.zero_pos()
        l_info["inertial_quat"] = gu.identity_quat()
        l_info["inertial_i"] = None
        l_info["inertial_mass"] = None

    j_infos = []
    for joint_prim, parent_idx, is_body1 in joints:
        if l_info.setdefault("parent_idx", parent_idx) != parent_idx:
            gs.raise_exception(f"Link {link.GetPath()} has multiple parent links: {l_info['parent_idx']} and {parent_idx}.")

        if parent_idx != -1:
            parent_link = links[parent_idx]
        else:
            parent_link = entity_prim
        Q, S = context.compute_gs_transform(link, parent_link)
        l_info["pos"] = Q[:3, 3]
        l_info["quat"] = gu.R_to_quat(Q[:3, :3])

        joint_type = gs.JOINT_TYPE.FIXED
        n_dofs, n_qs = 0, 0
        if not link_fixed:
            if joint_prim.IsA(UsdPhysics.RevoluteJoint):
                joint_type = gs.JOINT_TYPE.REVOLUTE
                joint = UsdPhysics.RevoluteJoint(joint_prim)
                n_dofs, n_qs = 1, 1
            elif joint_prim.IsA(UsdPhysics.PrismaticJoint):
                joint_type = gs.JOINT_TYPE.PRISMATIC
                joint = UsdPhysics.PrismaticJoint(joint_prim)
                n_dofs, n_qs = 1, 1
            elif joint_prim.IsA(UsdPhysics.SphericalJoint):
                joint_type = gs.JOINT_TYPE.SPHERICAL
                joint = UsdPhysics.SphericalJoint(joint_prim)
                n_dofs, n_qs = 3, 4
            elif not joint_prim.IsA(UsdPhysics.FixedJoint):
                gs.logger.warning(
                    f"Unsupported USD joint type: {joint_prim.GetTypeName()} for {joint_prim.GetPath()}. "
                    "Parsed as fixed joint."
                )
        if joint_type == gs.JOINT_TYPE.FIXED:
            joint = UsdPhysics.Joint(joint_prim)

        joint_axis, joint_pos = _parse_joint_axis_pos(context, joint, link, is_body1)
        j_info = {
            "name": str(joint_prim.GetPath()),
            "sol_params": gu.default_solver_params(),
            "n_qs": n_qs,
            "n_dofs": n_dofs,
            "type": joint_type,
            "pos": joint_pos,
            "dofs_invweight": np.full(n_dofs, -1.0, dtype=gs.np_float),
        }

        if joint_type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC):
            lower_limit = np.deg2rad(joint.GetLowerLimitAttr().Get() or -np.inf)
            upper_limit = np.deg2rad(joint.GetUpperLimitAttr().Get() or np.inf)
            j_info["dofs_limit"] = np.asarray([[lower_limit, upper_limit]], dtype=gs.np_float)
            j_info["init_qpos"] = np.zeros(n_qs, dtype=gs.np_float)

            if joint_type == gs.JOINT_TYPE.REVOLUTE:
                j_info["dofs_motion_ang"] = joint_axis[None]
                j_info["dofs_motion_vel"] = np.zeros((1, 3), dtype=gs.np_float)
                j_info["dofs_stiffness"] = np.array([
                    get_attr_value_by_candidates(
                        joint_prim,
                        candidates=morph.revolute_joint_stiffness_attr_candidates,
                        attr_name="dofs_stiffness",
                        default_value=0.0,
                    )
                ], dtype=gs.np_float)
                j_info["dofs_damping"] = np.array([
                    get_attr_value_by_candidates(
                        joint_prim,
                        candidates=morph.revolute_joint_damping_attr_candidates,
                        attr_name="dofs_damping",
                        default_value=0.0,
                    )
                ], dtype=gs.np_float)
                # NOTE: No idea how to scale the angle limits under non-uniform scaling now.
            else: # joint_type == gs.JOINT_TYPE.PRISMATIC
                j_info["dofs_motion_ang"] = np.zeros((1, 3), dtype=gs.np_float)
                j_info["dofs_motion_vel"] = joint_axis[None]
                j_info["dofs_stiffness"] = np.array([
                    get_attr_value_by_candidates(
                        joint_prim,
                        candidates=morph.prismatic_joint_stiffness_attr_candidates,
                        attr_name="dofs_stiffness",
                        default_value=0.0,
                    )
                ], dtype=gs.np_float)
                j_info["dofs_damping"] = np.array([
                    get_attr_value_by_candidates(
                        joint_prim,
                        candidates=morph.prismatic_joint_damping_attr_candidates,
                        attr_name="dofs_damping",
                        default_value=0.0,
                    )
                ], dtype=gs.np_float)
                j_info["dofs_limit"] *= morph.scale
                j_info["init_qpos"] *= morph.scale

        else:
            j_info["dofs_stiffness"] = np.zeros(n_dofs, dtype=gs.np_float)
            j_info["dofs_damping"] = np.zeros(n_dofs, dtype=gs.np_float)

            if joint_type == gs.JOINT_TYPE.SPHERICAL:
                j_info["dofs_motion_ang"] = np.eye(3)
                j_info["dofs_motion_vel"] = np.zeros((3, 3))
                j_info["dofs_limit"] = np.tile([-np.inf, np.inf], (3, 1))
                j_info["init_qpos"] = gu.identity_quat()
            else:
                j_info["dofs_motion_ang"] = np.zeros((0, 3))
                j_info["dofs_motion_vel"] = np.zeros((0, 3))
                j_info["dofs_limit"] = np.zeros((0, 2))
                j_info["init_qpos"] = np.zeros(0)

        # Only parse joint dynamics and drive API for non-fixed and non-free joints
        if joint_type not in (gs.JOINT_TYPE.FIXED, gs.JOINT_TYPE.FREE):
            j_info["dofs_frictionloss"] = np.full((n_dofs,), get_attr_value_by_candidates(
                joint_prim,
                candidates=morph.joint_friction_attr_candidates,
                attr_name="dofs_frictionloss",
                default_value=0.0,
            ), dtype=gs.np_float)
            j_info["dofs_armature"] = np.full((n_dofs,), get_attr_value_by_candidates(
                joint_prim,
                candidates=morph.joint_armature_attr_candidates,
                attr_name="dofs_armature",
                default_value=0.0,
            ), dtype=gs.np_float)

            # parse drive API
            # References: https://openusd.org/release/api/class_usd_physics_drive_a_p_i.html
            # Note: dofs_stiffness and dofs_damping are NOT set here - they are passive joint properties
            # that come from joint dynamics, not from DriveAPI (which is an active control system)
            drives = UsdPhysics.DriveAPI.GetAll(joint_prim)
            drive_params = {}
            for d in drives:
                name = d.GetName()
                force_range = d.GetMaxForceAttr().Get() or np.inf
                drive_params[name] = {
                    "type": d.GetTypeAttr().Get(),
                    "dofs_kp": d.GetStiffnessAttr().Get() or 0.0,
                    "dofs_kv": d.GetDampingAttr().Get() or 0.0,
                    "dofs_force_range": [-force_range, force_range],
                }
            dofs_kp, dofs_kv, dofs_force_range = [], [], []
            for drive_component in DRIVE_NAMES[joint_type]:
                dofs_kp.append(drive_params.get(drive_component, {}).get("dofs_kp", 0.0))
                dofs_kv.append(drive_params.get(drive_component, {}).get("dofs_kv", 0.0))
                dofs_force_range.append(
                    drive_params.get(drive_component, {}).get("dofs_force_range", [-np.inf, np.inf])
                )
            # TODO: Implement target solving in rigid solver. (GetTargetPositionAttr())
            j_info["dofs_kp"] = np.asarray(dofs_kp, dtype=gs.np_float)
            j_info["dofs_kv"] = np.asarray(dofs_kv, dtype=gs.np_float)
            j_info["dofs_force_range"] = np.asarray(dofs_force_range, dtype=gs.np_float)
        else:
            j_info["dofs_frictionloss"] = np.zeros(n_dofs, dtype=gs.np_float)
            j_info["dofs_armature"] = np.zeros(n_dofs, dtype=gs.np_float)
            j_info["dofs_kp"] = np.zeros(n_dofs, dtype=gs.np_float)
            j_info["dofs_kv"] = np.zeros(n_dofs, dtype=gs.np_float)
            j_info["dofs_force_range"] = np.tile([-np.inf, np.inf], (n_dofs, 1))

        j_infos.append(j_info)

    if not j_infos:
        if not link_fixed:
            j_infos.append(_create_joint_info_for_base_link(l_info))
        else:
            j_infos.append(_create_joint_info_for_fixed_link(l_info))

    if abs(1.0 - morph.scale) > gs.EPS:
        l_info["pos"] *= morph.scale
        l_info["inertial_pos"] *= morph.scale
        l_info["inertial_mass"] *= morph.scale**3
        l_info["inertial_i"] *= morph.scale**5
        l_info["invweight"][:] = -1.0
        for j_info in j_infos:
            j_info["pos"] *= morph.scale
            # TODO: parse actuator in USD articulation
            # j_info["dofs_kp"] *= morph.scale**3
            # j_info["dofs_kv"] *= morph.scale**3
            j_info["dofs_invweight"][:] = -1.0

    return l_info, j_infos

# Rigidbody requirements: https://docs.omniverse.nvidia.com/kit/docs/asset-requirements/latest/capabilities/physics_bodies/physics_rigid_bodies/capability-physics_rigid_bodies.html
# Joint requirements: https://docs.omniverse.nvidia.com/kit/docs/asset-requirements/latest/capabilities/physics_bodies/physics_joints/capability-physics_joints.html

def _parse_articulation_structure(stage: Usd.Stage, entity_prim: Usd.Prim):
    # TODO: we only assume that all links are under the subtree of the articulation root now.
    # To parse the accurate articulation structure, we need to search through the BodyRel.
    link_path_joints = {}
    for prim in Usd.PrimRange(entity_prim):
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            body0_targets = joint.GetBody0Rel().GetTargets() # parent
            body1_targets = joint.GetBody1Rel().GetTargets() # child
            body0_target_path = body0_targets[0] if body0_targets else None 
            body1_target_path = body1_targets[0] if body1_targets else None
            if body1_target_path:
                if body0_target_path:
                    link_path_joints.setdefault(body0_target_path, [])
                link_path_joints.setdefault(body1_target_path, []).append((prim, body0_target_path, True))
            elif body0_target_path:
                link_path_joints.setdefault(body0_target_path, []).append((prim, None, False))

    links, link_joints = [], []
    if link_path_joints:
        link_path_to_idx = {link_path: idx for idx, link_path in enumerate(link_path_joints.keys())}
        link_path_to_idx[None] = -1
        for link_path, joints in link_path_joints.items():
            if not stage.GetPrimAtPath(link_path):
                gs.raise_exception(f"Link {link_path} not found in stage.")
            links.append(stage.GetPrimAtPath(link_path))
            link_joints.append([
                (joint, link_path_to_idx[parent_path], is_body1) for joint, parent_path, is_body1 in joints
            ])
    else:
        links = [entity_prim]
        link_joints = [[]]

    return links, link_joints


def _parse_geoms(
    context: UsdContext,
    links: List[Usd.Prim],
    morph: gs.morphs.USD,
    surface: gs.surfaces.Surface,
) -> List[List[Dict]]:
    links_g_infos = []
    for link in links:
        links_g_infos.append(parse_prim_geoms(context, link, link, morph, surface))
    return links_g_infos


def _parse_links(
    context: UsdContext,
    links: List[Usd.Prim],
    link_joints: List[List[Tuple(Usd.Prim, int, bool)]],
    entity_prim: Usd.Prim,
    morph: gs.morphs.USD,
) -> Tuple[List[Dict], List[List[Dict]]]:
    l_infos = []
    links_j_infos = []
    for link, joints in zip(links, link_joints):
        l_info, link_j_info = _parse_link(context, link, joints, links, entity_prim, morph)
        l_infos.append(l_info)
        links_j_infos.append(link_j_info)
    return l_infos, links_j_infos


# ==================== Main Parsing Function ====================
def parse_usd(morph: gs.morphs.USD, surface: gs.surfaces.Surface):
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
    # if morph.scale is not None and morph.scale != 1.0:
    #     gs.logger.warning("USD rigid entity parsing currently only supports scale=1.0. Scale will be set to 1.0.")
    # morph.scale = 1.0
    context: UsdContext = morph.usd_ctx
    context.find_all_materials()
    stage: Usd.Stage = context.stage

    if morph.prim_path is None:
        gs.raise_exception("USD rigid entity must have a prim path.")
    entity_prim: Usd.Prim = stage.GetPrimAtPath(morph.prim_path)
    if not entity_prim.IsValid():
        gs.raise_exception(f"Invalid prim path {morph.prim_path} in USD file {morph.file}.")
    if not (
        entity_prim.HasAPI(UsdPhysics.RigidBodyAPI) or
        entity_prim.HasAPI(UsdPhysics.CollisionAPI) or
        entity_prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ):
        gs.raise_exception(
            f"Provided prim {entity_prim.GetPath()} is neither an articulation root nor a rigid body. "
            f"APIs found: {entity_prim.GetAppliedSchemas()}"
        )

    # find joints
    links, link_joints = _parse_articulation_structure(stage, entity_prim)
    links_g_infos = _parse_geoms(context, links, morph, surface)
    l_infos, links_j_infos = _parse_links(context, links, link_joints, entity_prim, morph)
    l_infos, links_j_infos, links_g_infos, _ = uu._order_links(l_infos, links_j_infos, links_g_infos)
    eqs_info = parse_equalities(mj, morph.scale)

        
    l_info = _parse_link(link)
    l_infos.append(l_info)
    visual_g_infos = _create_visual_geo_infos(link, context, morph)
    collision_g_infos = _create_collision_geo_infos(link, context, morph)
    if len(visual_g_infos) == 0 and len(collision_g_infos) == 0:
        gs.logger.warning(f"No visual or collision geometries found for link {link.GetPath()}, skipping.")
        continue
    link_g_infos.extend(visual_g_infos)
    link_g_infos.extend(collision_g_infos)

    if has_joints:
        _parse_joints(stage, joints, l_infos, links_j_infos, link_name_to_idx, morph)

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


# def parse_all_rigid_entities(
#     scene: gs.Scene,
#     stage: Usd.Stage,
#     context: UsdParserContext,
#     usd_morph: USD,
#     vis_mode: Literal["visual", "collision"],
#     visualize_contact: bool = False,
# ) -> Dict[str, "Entity"]:
#     """
#     Find and parse all rigid entities (articulations and rigid bodies) from a USD stage.

#     Parameters
#     ----------
#     scene : gs.Scene
#         The scene to add entities to.
#     stage : Usd.Stage
#         The USD stage.
#     context : UsdParserContext
#         The parser context.
#     usd_morph : USD
#         USD morph configuration.
#     vis_mode : Literal["visual", "collision"]
#         Visualization mode.
#     visualize_contact : bool, optional
#         Whether to visualize contact, by default False.

#     Returns
#     -------
#     Dict[str, Entity]
#         Dictionary of created entities (both articulations and rigid bodies) keyed by prim path.
#     """
#     from genesis.engine.entities.base_entity import Entity as GSEntity

#     entities: Dict[str, GSEntity] = {}

#     # Find all rigid entities (articulations and rigid bodies)
#     rigid_entities = _find_all_rigid_entities(stage, context)
#     articulation_roots = rigid_entities["articulation_roots"]
#     rigid_bodies = rigid_entities["rigid_bodies"]

#     gs.logger.debug(
#         f"Found {len(articulation_roots)} articulation(s) and {len(rigid_bodies)} rigid body(ies) in USD stage."
#     )

#     # Process articulation roots
#     for articulation_root in articulation_roots:
#         morph = copy.copy(usd_morph)
#         morph.prim_path = str(articulation_root.GetPath())
#         # NOTE: Now only support per-entity density, not per-geometry density.
#         density = _parse_density(articulation_root)
#         entity = scene.add_entity(
#             morph, material=gs.materials.Rigid(rho=density), vis_mode=vis_mode, visualize_contact=visualize_contact
#         )
#         entities[str(articulation_root.GetPath())] = entity
#         gs.logger.debug(f"Imported articulation from prim: {articulation_root.GetPath()} with density: {density}")

#     # Process rigid bodies (treated as articulation roots with no child links)
#     for rigid_body_prim in rigid_bodies:
#         morph = copy.copy(usd_morph)
#         morph.prim_path = str(rigid_body_prim.GetPath())
#         # NOTE: Now only support per-entity density, not per-geometry density.
#         density = _parse_density(rigid_body_prim)
#         entity = scene.add_entity(
#             morph, material=gs.materials.Rigid(rho=density), vis_mode=vis_mode, visualize_contact=visualize_contact
#         )
#         entities[str(rigid_body_prim.GetPath())] = entity
#         gs.logger.debug(f"Imported rigid body from prim: {rigid_body_prim.GetPath()} with density: {density}")

#     return entities
