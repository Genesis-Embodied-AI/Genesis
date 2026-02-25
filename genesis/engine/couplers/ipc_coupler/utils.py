"""
Utility functions for IPC coupler.

Stateless helper functions extracted from IPCCoupler for clarity.
"""

import numba as nb
import numpy as np

import genesis as gs
import genesis.utils.geom as gu

try:
    from uipc.core import Scene as _UIPCScene

    _UIPC_AVAILABLE = True
except ImportError:
    _UIPC_AVAILABLE = False


def find_target_link_for_fixed_merge(rigid_solver, link_idx):
    """
    Find the target link for merging fixed joints.

    Walks up the kinematic tree, skipping links connected via FIXED joints,
    until finding a link with a non-FIXED joint (or the root).

    This is similar to _merge_target_id in mjcf.py.

    Returns
    -------
    int
        The target link index to merge into
    """
    target_idx = link_idx

    while True:
        link = rigid_solver.links[target_idx]

        # If this is the root link (no parent), stop
        if link.parent_idx < 0:
            break

        # Check if all joints connecting this link to parent are FIXED
        # In Genesis/MuJoCo convention:
        # - len(joints) == 0: means this link has no joints (fixed to parent)
        # - all joints are FIXED type: also means fixed to parent
        joints = link.joints
        if len(joints) == 0:
            # No joints means this is a fixed joint, continue merging
            target_idx = link.parent_idx
            continue

        # Check if all joints are FIXED
        all_fixed = all(joint.type == gs.JOINT_TYPE.FIXED for joint in joints)

        if not all_fixed:
            # Found a link with non-FIXED joint, this is our target
            break

        # All joints are FIXED, move up to parent
        target_idx = link.parent_idx

    return target_idx


def compute_link_to_link_transform(rigid_solver, from_link_idx, to_link_idx):
    """
    Compute the relative transform from from_link to to_link.

    Similar to _accumulate_body_to_parent_transform in mjcf.py, but computed
    using Genesis link positions and quaternions.

    Returns
    -------
    tuple
        (rotation_matrix, translation_vector) transforming points from
        from_link frame to to_link frame
    """
    # Accumulate transforms going up from from_link to common ancestor (to_link)
    R_acc = np.eye(3, dtype=np.float32)
    t_acc = np.zeros(3, dtype=np.float32)

    current_idx = from_link_idx
    while current_idx != to_link_idx:
        link = rigid_solver.links[current_idx]

        if link.parent_idx < 0:
            # Reached root without finding to_link - this shouldn't happen
            gs.logger.error(f"Cannot compute transform from link {from_link_idx} to {to_link_idx}")
            break

        # Get link's local transform (relative to parent)
        link_quat = link.quat
        link_pos = link.pos
        link_rot = gu.quat_to_R(link_quat)

        # Accumulate: transform from current link to its parent
        # New point = R_link @ old_point + t_link
        # Accumulated: R_acc_new = R_link @ R_acc_old
        #              t_acc_new = R_link @ t_acc_old + t_link
        R_acc = link_rot @ R_acc
        t_acc = link_rot @ t_acc + link_pos

        current_idx = link.parent_idx

    return R_acc, t_acc


def is_robot_entity(entity):
    """Check if entity is a robot (has non-fixed, non-free joints)."""
    return any(j.type not in (gs.JOINT_TYPE.FIXED, gs.JOINT_TYPE.FREE) for j in entity.joints)


def compute_link_init_world_rotation(rigid_solver, link_idx):
    """
    Compute the world rotation matrix for a link in its initial configuration.
    This recursively computes the rotation by traversing up the kinematic tree.

    Note: In MuJoCo/URDF, the joint origin rotation (rpy) is baked into the child
    link's body transformation. Therefore, we use link.quat (not joint.quat) which
    contains the complete transformation including the joint origin rotation.
    """
    link = rigid_solver.links[link_idx]
    if link.parent_idx < 0:
        # Root link, just use its own orientation
        link_quat = link.quat
        link_rot = gu.quat_to_R(link_quat)
        return link_rot

    parent_rot = compute_link_init_world_rotation(rigid_solver, link.parent_idx)

    # Use link.quat which contains the joint origin rotation (from URDF <origin rpy="..."/>)
    # Note: joint.quat is hardcoded to [1,0,0,0] in Genesis's MJCF parser and is NOT used
    link_quat = link.quat
    link_local_rot = gu.quat_to_R(link_quat)
    link_world_rot = parent_rot @ link_local_rot

    return link_world_rot


def extract_articulated_joints(entity):
    """
    Extract revolute and prismatic joints from a RigidEntity.

    Parameters
    ----------
    entity : RigidEntity
        The rigid entity to extract joints from

    Returns
    -------
    dict
        Dictionary containing:
        - 'revolute_joints': List of revolute joints
        - 'prismatic_joints': List of prismatic joints
        - 'joint_qpos_indices': List of local q-space indices for each joint
        - 'joint_dof_indices': List of local DOF indices for each joint
        - 'n_joints': Total number of joints
    """
    revolute_joints = []
    prismatic_joints = []
    joint_qpos_indices = []
    joint_dof_indices = []

    for link_joints in entity._joints:
        if len(link_joints) == 0:
            continue

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
            f"External articulation coupling requires at least one 1-DOF joint."
        )

    return {
        "revolute_joints": revolute_joints,
        "prismatic_joints": prismatic_joints,
        "joint_qpos_indices": joint_qpos_indices,
        "joint_dof_indices": joint_dof_indices,
        "n_joints": n_joints,
    }


def build_ipc_scene_config(options, sim_options):
    """
    Build IPC Scene config dict from IPCCouplerOptions and SimOptions.

    Parameters
    ----------
    options : IPCCouplerOptions
        The coupler options
    sim_options : SimOptions
        The simulation options (provides dt, gravity, requires_grad)

    Returns
    -------
    dict
        Scene config dict ready to pass to Scene(config)
    """
    config = _UIPCScene.default_config()

    # Basic simulation parameters (derived from SimOptions)
    config["dt"] = sim_options.dt
    gravity = sim_options.gravity
    config["gravity"] = [[float(e)] for e in gravity]

    # Newton solver options (only set if specified)
    _set_if_not_none(config, ["newton", "max_iter"], options.newton_max_iterations)
    _set_if_not_none(config, ["newton", "min_iter"], options.newton_min_iterations)
    _set_if_not_none(config, ["newton", "velocity_tol"], options.newton_tolerance)
    _set_if_not_none(config, ["newton", "ccd_tol"], options.newton_ccd_tolerance)
    _set_if_not_none(config, ["newton", "use_adaptive_tol"], options.newton_use_adaptive_tolerance)
    _set_if_not_none(config, ["newton", "transrate_tol"], options.newton_translation_tolerance)
    _set_if_not_none(config, ["newton", "semi_implicit", "enable"], options.newton_semi_implicit_enable)
    _set_if_not_none(config, ["newton", "semi_implicit", "beta_tol"], options.newton_semi_implicit_beta_tolerance)

    # Line search options
    _set_if_not_none(config, ["line_search", "max_iter"], options.n_linesearch_iterations)
    _set_if_not_none(config, ["line_search", "report_energy"], options.linesearch_report_energy)

    # Linear system options
    _set_if_not_none(config, ["linear_system", "solver"], options.linear_system_solver)
    _set_if_not_none(config, ["linear_system", "tol_rate"], options.linear_system_tolerance)

    # Contact options
    _set_if_not_none(config, ["contact", "enable"], options.contact_enable)
    _set_if_not_none(config, ["contact", "d_hat"], options.contact_d_hat)
    _set_if_not_none(config, ["contact", "friction", "enable"], options.contact_friction_enable)
    _set_if_not_none(config, ["contact", "eps_velocity"], options.contact_eps_velocity)
    _set_if_not_none(config, ["contact", "constitution"], options.contact_constitution)

    # Collision detection options
    _set_if_not_none(config, ["collision_detection", "method"], options.collision_detection_method)

    # CFL options
    _set_if_not_none(config, ["cfl", "enable"], options.cfl_enable)

    # Sanity check options
    _set_if_not_none(config, ["sanity_check", "enable"], options.sanity_check_enable)

    # Differential simulation options (derived from SimOptions)
    _set_if_not_none(config, ["diff_sim", "enable"], sim_options.requires_grad)

    return config


def _set_if_not_none(config, keys, value):
    """Set a nested config value only if it's not None."""
    if value is None:
        return
    # Cast to native Python types — UIPC pybind11 rejects numpy scalars and Python bool.
    # bool check must come before int since bool is a subclass of int.
    if isinstance(value, (bool, np.bool_)):
        value = int(value)
    elif isinstance(value, (int, np.integer)):
        value = int(value)
    elif isinstance(value, (float, np.floating)):
        value = float(value)
    d = config
    for key in keys[:-1]:
        d = d[key]
    d[keys[-1]] = value


def read_ipc_geometry_metadata(geo):
    """
    Read solver_type, env_idx, and entity/link index from IPC geometry metadata.

    Returns (solver_type, env_idx, idx) where idx is entity_idx for fem/cloth
    or link_idx for rigid. Returns None if the geometry has no solver_type
    metadata (i.e. not a Genesis-created geometry).
    """
    meta_attrs = geo.meta()
    solver_type_attr = meta_attrs.find("solver_type")
    if solver_type_attr is None:
        return None

    (solver_type,) = solver_type_attr.view()
    solver_type = str(solver_type)

    (env_idx,) = map(int, meta_attrs.find("env_idx").view())

    if solver_type == "rigid":
        (idx,) = map(int, meta_attrs.find("link_idx").view())
    elif solver_type in ("fem", "cloth"):
        (idx,) = map(int, meta_attrs.find("entity_idx").view())
    else:
        gs.raise_exception(f"Unknown IPC geometry solver_type: {solver_type!r}")

    return (solver_type, env_idx, idx)


# ============================================================
# Numpy computation functions (replacing Quadrants kernels)
# ============================================================


@nb.jit(nopython=True, cache=True)
def compute_coupling_forces(
    ipc_transforms,
    aim_transforms,
    link_masses,
    inertia_tensors,
    translation_strength,
    rotation_strength,
    dt2,
):
    """
    Compute coupling forces and torques for all links.

    Parameters
    ----------
    ipc_transforms : np.ndarray, shape (n, 4, 4)
    aim_transforms : np.ndarray, shape (n, 4, 4)
    link_masses : np.ndarray, shape (n,)
    inertia_tensors : np.ndarray, shape (n, 3, 3)
    translation_strength : float
    rotation_strength : float
    dt2 : float

    Returns
    -------
    tuple of (forces, torques), each shape (n, 3)
    """
    n = ipc_transforms.shape[0]
    out_forces = np.zeros((n, 3), dtype=ipc_transforms.dtype)
    out_torques = np.zeros((n, 3), dtype=ipc_transforms.dtype)

    for i in range(n):
        pos_current = ipc_transforms[i, :3, 3]
        pos_aim = aim_transforms[i, :3, 3]

        R_current = ipc_transforms[i, :3, :3]
        R_aim = aim_transforms[i, :3, :3]

        # Linear force: F = strength * mass * delta_pos / dt^2
        mass = link_masses[i]
        for k in range(3):
            out_forces[i, k] = translation_strength * mass * (pos_current[k] - pos_aim[k]) / dt2

        # Relative rotation: R_rel = R_current @ R_aim^T
        R_rel = R_current @ R_aim.T

        # Rodrigues: extract rotation vector
        trace = R_rel[0, 0] + R_rel[1, 1] + R_rel[2, 2]
        cos_val = (trace - 1.0) / 2.0
        cos_val = max(-1.0, min(1.0, cos_val))
        theta = np.arccos(cos_val)

        rotvec = np.zeros(3, dtype=ipc_transforms.dtype)
        if theta > 1e-6:
            ax0 = R_rel[2, 1] - R_rel[1, 2]
            ax1 = R_rel[0, 2] - R_rel[2, 0]
            ax2 = R_rel[1, 0] - R_rel[0, 1]
            norm = np.sqrt(ax0 * ax0 + ax1 * ax1 + ax2 * ax2)
            if norm > 1e-8:
                s = theta / norm
                rotvec[0] = s * ax0
                rotvec[1] = s * ax1
                rotvec[2] = s * ax2

        # Transform inertia to world frame: I_world = R @ I @ R^T
        I_world = R_current @ inertia_tensors[i] @ R_current.T

        # Torque = (rotation_strength / dt^2) * I_world @ rotvec
        scale = rotation_strength / dt2
        for k in range(3):
            val = 0.0
            for m in range(3):
                val += I_world[k, m] * rotvec[m]
            out_torques[i, k] = scale * val

    return out_forces, out_torques
