"""Utility functions for the IPC coupler."""

import numpy as np
from scipy.spatial.transform import Rotation as R

import genesis as gs
import genesis.utils.geom as gu

try:
    from uipc import Scene
except ImportError:
    pass


def find_target_link_for_fixed_merge(rigid_solver, link_idx):
    """Walk up the kinematic tree, skipping FIXED joints, to find the merge-target link."""
    target_idx = link_idx

    while True:
        link = rigid_solver.links[target_idx]

        if link.parent_idx < 0:
            break

        joints = link.joints
        if len(joints) == 0:
            target_idx = link.parent_idx
            continue

        if not all(joint.type == gs.JOINT_TYPE.FIXED for joint in joints):
            break

        target_idx = link.parent_idx

    return target_idx


def compute_link_to_link_transform(rigid_solver, from_link_idx, to_link_idx):
    """Compute the relative (R, t) transform from ``from_link`` to ``to_link`` up the tree."""
    R_acc = np.eye(3, dtype=np.float32)
    t_acc = np.zeros(3, dtype=np.float32)

    current_idx = from_link_idx
    while current_idx != to_link_idx:
        link = rigid_solver.links[current_idx]

        if link.parent_idx < 0:
            gs.logger.error(f"Cannot compute transform from link {from_link_idx} to {to_link_idx}")
            break

        link_rot = gu.quat_to_R(link.quat)
        R_acc = link_rot @ R_acc
        t_acc = link_rot @ t_acc + link.pos
        current_idx = link.parent_idx

    return R_acc, t_acc


def compute_link_init_world_rotation(rigid_solver, link_idx):
    """Recursively compute the world rotation of a link in the initial configuration."""
    link = rigid_solver.links[link_idx]
    if link.parent_idx < 0:
        return gu.quat_to_R(link.quat)

    parent_rot = compute_link_init_world_rotation(rigid_solver, link.parent_idx)
    return parent_rot @ gu.quat_to_R(link.quat)


def extract_articulated_joints(entity):
    """Extract revolute and prismatic joints from a RigidEntity."""
    revolute_joints = []
    prismatic_joints = []
    joint_qpos_indices = []
    joint_dof_indices = []

    for link_joints in entity._joints:
        for joint in link_joints:
            if joint.type == gs.JOINT_TYPE.FIXED:
                continue
            if joint.type == gs.JOINT_TYPE.REVOLUTE:
                revolute_joints.append(joint)
                joint_qpos_indices.append(joint.qs_idx_local[0])
                joint_dof_indices.append(joint.dofs_idx_local[0])
            elif joint.type == gs.JOINT_TYPE.PRISMATIC:
                prismatic_joints.append(joint)
                joint_qpos_indices.append(joint.qs_idx_local[0])
                joint_dof_indices.append(joint.dofs_idx_local[0])

    n_joints = len(revolute_joints) + len(prismatic_joints)

    if n_joints == 0:
        gs.logger.warning(
            f"Entity {entity.idx} has no revolute or prismatic joints. "
            "External articulation coupling requires at least one 1-DOF joint."
        )

    return {
        "revolute_joints": revolute_joints,
        "prismatic_joints": prismatic_joints,
        "joint_qpos_indices": joint_qpos_indices,
        "joint_dof_indices": joint_dof_indices,
        "n_joints": n_joints,
    }


def categorize_entities_by_coupling_type(entity_coupling_types):
    """Partition entity indices by coupling mode."""
    result = {
        "two_way_soft_constraint": [],
        "external_articulation": [],
        "ipc_only": [],
    }

    for entity_idx, coupling_type in entity_coupling_types.items():
        if coupling_type in result:
            result[coupling_type].append(entity_idx)

    return result


def build_ipc_scene_config(options, sim_options):
    """Build a ``uipc.Scene`` config dict from IPCCouplerOptions and SimOptions."""
    config = Scene.default_config()

    config["dt"] = sim_options.dt
    g = sim_options.gravity
    config["gravity"] = [[g[0]], [g[1]], [g[2]]]

    # contact
    config["contact"]["d_hat"] = options.contact_d_hat
    if options.contact_enable is not None:
        config["contact"]["enable"] = options.contact_enable
    if options.contact_friction_enable is not None:
        config["contact"]["friction"]["enable"] = options.contact_friction_enable
    if options.contact_eps_velocity is not None:
        config["contact"]["eps_velocity"] = options.contact_eps_velocity
    if options.contact_constitution is not None:
        config["contact"]["constitution"] = options.contact_constitution

    # newton solver
    config["newton"]["velocity_tol"] = options.newton_tolerance
    if options.newton_max_iterations is not None:
        config["newton"]["max_iter"] = options.newton_max_iterations
    if options.newton_min_iterations is not None:
        config["newton"]["min_iter"] = options.newton_min_iterations
    if options.newton_ccd_tolerance is not None:
        config["newton"]["ccd_tol"] = options.newton_ccd_tolerance
    if options.newton_use_adaptive_tolerance is not None:
        config["newton"]["use_adaptive_tol"] = options.newton_use_adaptive_tolerance
    if options.newton_translation_tolerance is not None:
        config["newton"]["transrate_tol"] = options.newton_translation_tolerance
    if options.newton_semi_implicit_enable is not None:
        config["newton"]["semi_implicit"]["enable"] = options.newton_semi_implicit_enable
    if options.newton_semi_implicit_beta_tolerance is not None:
        config["newton"]["semi_implicit"]["beta_tol"] = options.newton_semi_implicit_beta_tolerance

    # line search
    config["line_search"]["max_iter"] = options.n_linesearch_iterations
    if options.linesearch_report_energy is not None:
        config["line_search"]["report_energy"] = options.linesearch_report_energy

    # linear system
    config["linear_system"]["tol_rate"] = options.linear_system_tolerance
    if options.linear_system_solver is not None:
        config["linear_system"]["solver"] = options.linear_system_solver

    # collision detection
    if options.collision_detection_method is not None:
        config["collision_detection"]["method"] = options.collision_detection_method

    # CFL condition
    if options.cfl_enable is not None:
        config["cfl"]["enable"] = options.cfl_enable

    # sanity check
    if options.sanity_check_enable is not None:
        config["sanity_check"]["enable"] = options.sanity_check_enable

    return config


def read_ipc_geometry_metadata(geo):
    """Read (solver_type, env_idx, idx) from IPC geometry metadata, or None."""
    try:
        meta_attrs = geo.meta()
        solver_type_attr = meta_attrs.find("solver_type")

        if not solver_type_attr or solver_type_attr.name() != "solver_type":
            return None

        solver_type_view = solver_type_attr.view()
        if len(solver_type_view) == 0:
            return None
        solver_type = str(solver_type_view[0])

        env_idx_attr = meta_attrs.find("env_idx")
        if not env_idx_attr:
            return None
        env_idx = int(str(env_idx_attr.view()[0]))

        if solver_type == "rigid":
            link_idx_attr = meta_attrs.find("link_idx")
            if not link_idx_attr:
                return None
            idx = int(str(link_idx_attr.view()[0]))
        elif solver_type in ("fem", "cloth"):
            entity_idx_attr = meta_attrs.find("entity_idx")
            if not entity_idx_attr:
                return None
            idx = int(str(entity_idx_attr.view()[0]))
        else:
            return None

        return (solver_type, env_idx, idx)
    except Exception:
        return None


def decompose_transform_matrix(transform_4x4):
    """Decompose a 4x4 homogeneous matrix into (position, quaternion_wxyz)."""
    pos = transform_4x4[:3, 3]
    quat_xyzw = R.from_matrix(transform_4x4[:3, :3]).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return pos, quat_wxyz
