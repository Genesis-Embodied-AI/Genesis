from typing import Dict, List, Tuple

import numpy as np
from pxr import Usd, UsdPhysics

import genesis as gs
from genesis.utils import geom as gu
from genesis.utils import urdf as urdf_utils

from .usd_context import UsdContext
from .usd_geometry import parse_prim_geoms
from .usd_utils import (
    AXES_VECTOR,
    get_attr_value_by_candidates,
    usd_center_of_mass_to_numpy,
    usd_diagonal_inertia_to_numpy,
    usd_inertia_to_numpy,
    usd_mass_to_float,
    usd_pos_to_numpy,
    usd_principal_axes_to_numpy,
    usd_quat_to_numpy,
)


DRIVE_NAMES = {
    gs.JOINT_TYPE.REVOLUTE: (("angular",),),
    gs.JOINT_TYPE.PRISMATIC: (("linear",),),
    gs.JOINT_TYPE.SPHERICAL: (
        ("rotX",),
        ("rotY",),
        ("rotZ",),
    ),
    gs.JOINT_TYPE.FIXED: (),
    gs.JOINT_TYPE.FREE: (
        ("transX",),
        ("transY",),
        ("transZ",),
        ("rotX",),
        ("rotY",),
        ("rotZ",),
    ),
}


def _parse_joint_axis_pos(
    context: UsdContext, joint: UsdPhysics.Joint, child_link: Usd.Prim, is_body1: bool
) -> Tuple[str, np.ndarray, np.ndarray]:
    joint_pos_attr = joint.GetLocalPos1Attr() if is_body1 else joint.GetLocalPos0Attr()
    joint_pos = usd_pos_to_numpy(joint_pos_attr.Get()) if joint_pos_attr.HasValue() else gu.zero_pos()
    T = context.compute_transform(child_link)
    joint_pos = gu.transform_by_T(joint_pos, T)
    Q, S = context.compute_gs_transform(child_link)
    Q_inv = np.linalg.inv(Q)
    joint_pos = Q_inv[:3, :3] @ (joint_pos - Q[:3, 3])

    if isinstance(joint, (UsdPhysics.PrismaticJoint, UsdPhysics.RevoluteJoint)):
        joint_quat = usd_quat_to_numpy((joint.GetLocalRot1Attr() if is_body1 else joint.GetLocalRot0Attr()).Get())
        joint_axis_str = joint.GetAxisAttr().Get() or "X"
        joint_axis = gu.transform_by_quat(AXES_VECTOR[joint_axis_str], joint_quat)
        joint_axis = gu.transform_by_R(joint_axis, T[:3, :3])
        joint_axis = Q_inv[:3, :3] @ joint_axis
        if np.linalg.norm(joint_axis) < gs.EPS:
            gs.raise_exception(f"Joint axis is zero for joint {joint.GetPath()}.")
        joint_axis /= np.linalg.norm(joint_axis)
    else:
        joint_axis_str, joint_axis = None, None

    return joint_axis_str, joint_axis, joint_pos


def _parse_link(
    context: UsdContext,
    link: Usd.Prim,
    joints: List[Tuple[Usd.Prim, int, bool]],
    links: List[Usd.Prim],
    morph: gs.morphs.USD,
):
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

    if morph.fixed:
        link_fixed = any(parent_idx == -1 for _, parent_idx, _ in joints)

    # parse link mass properties
    if link.HasAPI(UsdPhysics.MassAPI):
        mass_api = UsdPhysics.MassAPI(link)

        com_attr = mass_api.GetCenterOfMassAttr()
        l_info["inertial_pos"] = usd_center_of_mass_to_numpy(com_attr.Get())
        principal_axes_attr = mass_api.GetPrincipalAxesAttr()
        l_info["inertial_quat"] = usd_principal_axes_to_numpy(principal_axes_attr.Get())
        inertia_attr = mass_api.GetDiagonalInertiaAttr()
        l_info["inertial_i"] = usd_inertia_to_numpy(inertia_attr.Get())
        mass_attr = mass_api.GetMassAttr()
        l_info["inertial_mass"] = usd_mass_to_float(mass_attr.Get())

    j_infos = []
    for joint_prim, parent_idx, is_body1 in joints:
        if "parent_idx" not in l_info:
            parent_link = None if parent_idx == -1 else links[parent_idx]
            Q, S = context.compute_gs_transform(link, parent_link)
            l_info["parent_idx"] = parent_idx
            l_info["pos"] = Q[:3, 3]
            l_info["quat"] = gu.R_to_quat(Q[:3, :3])

        elif l_info["parent_idx"] != parent_idx:
            gs.raise_exception(f"Link {link.GetPath()} has multiple parents: {l_info['parent_idx']} and {parent_idx}.")

        if joint_prim is not None:
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
                elif joint_prim.GetTypeName() == "PhysicsJoint":
                    joint_type = gs.JOINT_TYPE.FREE
                    joint = UsdPhysics.Joint(joint_prim)
                    n_dofs, n_qs = 6, 7
                elif not joint_prim.IsA(UsdPhysics.FixedJoint):
                    gs.logger.warning(
                        f"Unsupported USD joint type: {joint_prim.GetTypeName()} for {joint_prim.GetPath()}. "
                        "Parsed as fixed joint."
                    )
            if joint_type == gs.JOINT_TYPE.FIXED:
                joint = UsdPhysics.Joint(joint_prim)
            joint_axis_str, joint_axis, joint_pos = _parse_joint_axis_pos(context, joint, link, is_body1)
            joint_name = str(joint_prim.GetPath())
        else:
            if link_fixed:
                joint_type = gs.JOINT_TYPE.FIXED
                n_dofs, n_qs = 0, 0
            else:
                joint_type = gs.JOINT_TYPE.FREE
                n_dofs, n_qs = 6, 7
            joint = None
            joint_axis_str, joint_axis, joint_pos = None, None, gu.zero_pos()
            joint_name = f"{l_info['name']}_joint"

        j_info = {
            "name": joint_name,
            "sol_params": gu.default_solver_params(),
            "n_qs": n_qs,
            "n_dofs": n_dofs,
            "type": joint_type,
            "pos": joint_pos,
            "dofs_invweight": np.full(n_dofs, -1.0, dtype=gs.np_float),
        }

        if joint_type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC):
            # TODO: use attribute "state:<INSTANCE_NAME>:physics:positiion" to parse init_qpos (but it is IsaacSim specific)
            j_info["init_qpos"] = np.zeros(n_qs, dtype=gs.np_float)

            if joint_type == gs.JOINT_TYPE.REVOLUTE:
                j_info["dofs_motion_ang"] = joint_axis[None]
                j_info["dofs_motion_vel"] = np.zeros((1, 3), dtype=gs.np_float)
                j_info["dofs_stiffness"] = np.array(
                    [
                        get_attr_value_by_candidates(
                            joint_prim,
                            candidates=morph.revolute_joint_stiffness_attr_candidates,
                            attr_name="dofs_stiffness",
                            default_value=0.0,
                        )
                    ],
                    dtype=gs.np_float,
                )
                j_info["dofs_damping"] = np.array(
                    [
                        get_attr_value_by_candidates(
                            joint_prim,
                            candidates=morph.revolute_joint_damping_attr_candidates,
                            attr_name="dofs_damping",
                            default_value=0.0,
                        )
                    ],
                    dtype=gs.np_float,
                )
                # NOTE: No idea how to scale the angle limits under non-uniform scaling now.
                lower_limit_attr = joint.GetLowerLimitAttr()
                upper_limit_attr = joint.GetUpperLimitAttr()
                lower_limit = np.deg2rad(lower_limit_attr.Get()) if lower_limit_attr.HasValue() else -np.inf
                upper_limit = np.deg2rad(upper_limit_attr.Get()) if upper_limit_attr.HasValue() else np.inf
                j_info["dofs_limit"] = np.asarray([[lower_limit, upper_limit]], dtype=gs.np_float)
            else:  # joint_type == gs.JOINT_TYPE.PRISMATIC
                j_info["dofs_motion_ang"] = np.zeros((1, 3), dtype=gs.np_float)
                j_info["dofs_motion_vel"] = joint_axis[None]
                j_info["dofs_stiffness"] = np.array(
                    [
                        get_attr_value_by_candidates(
                            joint_prim,
                            candidates=morph.prismatic_joint_stiffness_attr_candidates,
                            attr_name="dofs_stiffness",
                            default_value=0.0,
                        )
                    ],
                    dtype=gs.np_float,
                )
                j_info["dofs_damping"] = np.array(
                    [
                        get_attr_value_by_candidates(
                            joint_prim,
                            candidates=morph.prismatic_joint_damping_attr_candidates,
                            attr_name="dofs_damping",
                            default_value=0.0,
                        )
                    ],
                    dtype=gs.np_float,
                )
                lower_limit_attr = joint.GetLowerLimitAttr()
                upper_limit_attr = joint.GetUpperLimitAttr()
                lower_limit = lower_limit_attr.Get() if lower_limit_attr.HasValue() else -np.inf
                upper_limit = upper_limit_attr.Get() if upper_limit_attr.HasValue() else np.inf
                j_info["dofs_limit"] = np.asarray([[lower_limit, upper_limit]], dtype=gs.np_float) * morph.scale
                j_info["init_qpos"] *= morph.scale

        else:
            j_info["dofs_stiffness"] = np.zeros(n_dofs, dtype=gs.np_float)
            j_info["dofs_damping"] = np.zeros(n_dofs, dtype=gs.np_float)

            if joint_type == gs.JOINT_TYPE.SPHERICAL:
                j_info["dofs_motion_ang"] = np.eye(3)
                j_info["dofs_motion_vel"] = np.zeros((3, 3))
                j_info["dofs_limit"] = np.tile([-np.inf, np.inf], (3, 1))
                j_info["init_qpos"] = gu.identity_quat()
            elif joint_type == gs.JOINT_TYPE.FIXED:
                j_info["dofs_motion_ang"] = np.zeros((0, 3))
                j_info["dofs_motion_vel"] = np.zeros((0, 3))
                j_info["dofs_limit"] = np.zeros((0, 2))
                j_info["init_qpos"] = np.zeros(0)
            else:  # joint_type == gs.JOINT_TYPE.FREE
                j_info["dofs_motion_ang"] = np.eye(6, 3, -3)
                j_info["dofs_motion_vel"] = np.eye(6, 3)
                j_info["dofs_limit"] = np.tile([-np.inf, np.inf], (6, 1))
                j_info["init_qpos"] = np.concatenate([l_info["pos"] * morph.scale, l_info["quat"]])

        # Only parse joint dynamics and drive API for non-fixed and non-free joints
        if joint_type not in (gs.JOINT_TYPE.FIXED, gs.JOINT_TYPE.FREE):
            j_info["dofs_frictionloss"] = np.full(
                (n_dofs,),
                get_attr_value_by_candidates(
                    joint_prim,
                    candidates=morph.joint_friction_attr_candidates,
                    attr_name="dofs_frictionloss",
                    default_value=0.0,
                ),
                dtype=gs.np_float,
            )
            j_info["dofs_armature"] = np.full(
                (n_dofs,),
                get_attr_value_by_candidates(
                    joint_prim,
                    candidates=morph.joint_armature_attr_candidates,
                    attr_name="dofs_armature",
                    default_value=0.0,
                ),
                dtype=gs.np_float,
            )

            # parse drive API
            # References: https://openusd.org/release/api/class_usd_physics_drive_a_p_i.html
            # Note: dofs_stiffness and dofs_damping are NOT set here - they are passive joint properties
            # that come from joint dynamics, not from DriveAPI (which is an active control system)
            dofs_kp, dofs_kv, dofs_force_range = [], [], []

            for drive_components in DRIVE_NAMES[joint_type]:
                dof_kp, dof_kv, max_force = 0.0, 0.0, np.inf
                if joint_axis_str:
                    drive_components = drive_components + (joint_axis_str,)
                for drive_component in drive_components:
                    if joint_prim.HasAPI(UsdPhysics.DriveAPI, drive_component):
                        drive = UsdPhysics.DriveAPI.Get(joint_prim, drive_component)
                        # TODO: use drive.GetTypeAttr().Get() to parse force or velocity.
                        # Note: Defaults are 0 (stiffness/damping) and inf (maxForce), which are valid.
                        # Using 'or' is safe here since fallback values match the defaults.
                        dof_kp = drive.GetStiffnessAttr().Get() or dof_kp
                        dof_kv = drive.GetDampingAttr().Get() or dof_kv
                        max_force = drive.GetMaxForceAttr().Get() or max_force
                        break

                dofs_kp.append(dof_kp)
                dofs_kv.append(dof_kv)
                dofs_force_range.append([-max_force, max_force])

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

    if abs(1.0 - morph.scale) > gs.EPS:
        l_info["pos"] *= morph.scale
        if l_info.get("inertial_pos") is not None:
            l_info["inertial_pos"] *= morph.scale
        if l_info.get("inertial_mass") is not None:
            l_info["inertial_mass"] *= morph.scale**3
        if l_info.get("inertial_i") is not None:
            l_info["inertial_i"] *= morph.scale**5
        l_info["invweight"][:] = -1.0
        for j_info in j_infos:
            j_info["pos"] *= morph.scale
            # TODO: parse actuator in USD articulation, now all joints are considered to have actuators
            j_info["dofs_kp"] *= morph.scale**3
            j_info["dofs_kv"] *= morph.scale**3
            j_info["dofs_invweight"][:] = -1.0

    return l_info, j_infos


# Rigidbody requirements: https://docs.omniverse.nvidia.com/kit/docs/asset-requirements/latest/capabilities/physics_bodies/physics_rigid_bodies/capability-physics_rigid_bodies.html
# Joint requirements: https://docs.omniverse.nvidia.com/kit/docs/asset-requirements/latest/capabilities/physics_bodies/physics_joints/capability-physics_joints.html
def _parse_articulation_structure(stage: Usd.Stage, entity_prim: Usd.Prim):
    # TODO: we only assume that all links are under the subtree of the articulation root now.
    # To parse the accurate articulation structure, we need to search through the BodyRel.
    link_path_joints = {}
    if entity_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        for prim in Usd.PrimRange(entity_prim):  # will not iterate inactive prims
            if prim.IsA(UsdPhysics.Joint):
                joint = UsdPhysics.Joint(prim)
                body0_targets = joint.GetBody0Rel().GetTargets()  # parent
                body1_targets = joint.GetBody1Rel().GetTargets()  # child
                body0_target_path = str(body0_targets[0]) if body0_targets else None
                body1_target_path = str(body1_targets[0]) if body1_targets else None
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
            link_joints.append(
                [(joint, link_path_to_idx[parent_path], is_body1) for joint, parent_path, is_body1 in joints]
            )
    else:
        links = [entity_prim]
        link_joints = [[]]
        link_path_to_idx = {None: -1, str(entity_prim.GetPath()): 0}
    for joints in link_joints:
        if not joints:
            joints.append((None, -1, False))

    return links, link_joints, link_path_to_idx


def _parse_geoms(
    context: UsdContext,
    entity_prim: Usd.Prim,
    link_path_to_idx: Dict[str, int],
    morph: gs.morphs.USD,
    surface: gs.surfaces.Surface,
) -> List[List[Dict]]:
    links_g_infos = [[] for _ in range(len(link_path_to_idx))]
    parse_prim_geoms(context, entity_prim, None, links_g_infos, link_path_to_idx, morph, surface)
    return links_g_infos


def _parse_links(
    context: UsdContext,
    links: List[Usd.Prim],
    link_joints: List[List[Tuple[Usd.Prim, int, bool]]],
    morph: gs.morphs.USD,
) -> Tuple[List[Dict], List[List[Dict]]]:
    l_infos = []
    links_j_infos = []
    for link, joints in zip(links, link_joints):
        l_info, link_j_info = _parse_link(context, link, joints, links, morph)
        l_infos.append(l_info)
        links_j_infos.append(link_j_info)
    return l_infos, links_j_infos


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
    context: UsdContext = morph.usd_ctx
    context.find_all_materials()
    stage: Usd.Stage = context.stage

    if morph.prim_path is None:
        gs.logger.debug("USD morph has no prim path. Fallback to its default prim path.")
        entity_prim = stage.GetDefaultPrim()
    else:
        entity_prim = stage.GetPrimAtPath(morph.prim_path)
    if not entity_prim.IsValid():
        if morph.prim_path is None:
            err_msg = (
                f"Invalid default prim path {entity_prim} in USD file {morph.file}. Please specify 'morph.prim_path'."
            )
        else:
            err_msg = f"Invalid user-specified prim path {entity_prim} in USD file {morph.file}."
        gs.raise_exception(err_msg)

    # find joints
    links, link_joints, link_path_to_idx = _parse_articulation_structure(stage, entity_prim)
    links_g_infos = _parse_geoms(context, entity_prim, link_path_to_idx, morph, surface)
    l_infos, links_j_infos = _parse_links(context, links, link_joints, morph)
    l_infos, links_j_infos, links_g_infos, _ = urdf_utils._order_links(l_infos, links_j_infos, links_g_infos)
    eqs_info = []  # USD doesn't support equality constraints

    return l_infos, links_j_infos, links_g_infos, eqs_info
