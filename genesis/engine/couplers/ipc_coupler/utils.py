"""
Utility functions for IPC coupler.

Stateless helper functions extracted from IPCCoupler for clarity.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

import genesis as gs
import genesis.utils.geom as gu


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
    """Heuristic: treat URDF/MJCF/Drone morphs as robots."""
    try:
        return isinstance(entity.morph, (gs.morphs.URDF, gs.morphs.MJCF, gs.morphs.Drone))
    except AttributeError:
        return False


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
    from uipc import Scene

    config = Scene.default_config()

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

    Parameters
    ----------
    geo : Geometry
        An IPC geometry with meta attributes

    Returns
    -------
    tuple or None
        (solver_type, env_idx, idx) where idx is entity_idx for fem/cloth
        or link_idx for rigid. Returns None if metadata is missing/invalid.
    """
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
    """
    Decompose a 4x4 transformation matrix into position and quaternion (wxyz).

    Parameters
    ----------
    transform_4x4 : np.ndarray
        4x4 homogeneous transformation matrix

    Returns
    -------
    tuple
        (pos, quat_wxyz) where pos is (3,) and quat_wxyz is (4,)
    """
    pos = transform_4x4[:3, 3]
    rot_mat = transform_4x4[:3, :3]
    quat_xyzw = R.from_matrix(rot_mat).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return pos, quat_wxyz


def build_link_transform_matrix(pos_3, quat_4):
    """
    Build a 4x4 transformation matrix from position and quaternion.
    Uses uipc Transform for consistency with IPC.

    Parameters
    ----------
    pos_3 : array-like
        Position (3,)
    quat_4 : array-like
        Quaternion in wxyz order (4,)

    Returns
    -------
    np.ndarray
        4x4 transformation matrix
    """
    from uipc import Transform, Vector3, Quaternion

    t = Transform.Identity()
    t.translate(Vector3.Values((float(pos_3[0]), float(pos_3[1]), float(pos_3[2]))))
    uipc_quat = Quaternion(quat_4)
    t.rotate(uipc_quat)
    return t.matrix().copy()


# ============================================================
# Numpy computation functions (replacing Quadrants kernels)
# ============================================================


def compute_external_force_12d(contact_forces, contact_torques, abd_transforms):
    """
    Compute 12D external force from contact forces and torques.

    force_12d = [force (3), M_affine (9)]
    where M_affine = skew(torque) @ A, A is the rotation part of ABD transform.

    Parameters
    ----------
    contact_forces : np.ndarray, shape (n, 3)
    contact_torques : np.ndarray, shape (n, 3)
    abd_transforms : np.ndarray, shape (n, 4, 4)

    Returns
    -------
    np.ndarray, shape (n, 12)
    """
    n = contact_forces.shape[0]
    out = np.zeros((n, 12), dtype=contact_forces.dtype)

    forces = -0.5 * contact_forces
    torques = -0.5 * contact_torques
    out[:, :3] = forces

    # Build skew-symmetric matrices for all torques at once: (n, 3, 3)
    S = np.zeros((n, 3, 3), dtype=torques.dtype)
    S[:, 0, 1] = -torques[:, 2]
    S[:, 0, 2] = torques[:, 1]
    S[:, 1, 0] = torques[:, 2]
    S[:, 1, 2] = -torques[:, 0]
    S[:, 2, 0] = -torques[:, 1]
    S[:, 2, 1] = torques[:, 0]

    # M_affine = S @ A (rotation part of transform)
    A = abd_transforms[:, :3, :3]
    M = np.einsum("nij,njk->nik", S, A)
    out[:, 3:] = M.reshape(n, 9)

    return out


def compute_coupling_forces(
    ipc_transforms, aim_transforms, link_masses, inertia_tensors,
    translation_strength, rotation_strength, dt2,
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
        delta_pos = pos_current - pos_aim

        R_current = ipc_transforms[i, :3, :3]
        R_aim = aim_transforms[i, :3, :3]

        # Linear force
        mass = link_masses[i]
        out_forces[i] = translation_strength * mass * delta_pos / dt2

        # Relative rotation: R_rel = R_current @ R_aim^T
        R_rel = R_current @ R_aim.T

        # Rodrigues: extract rotation vector
        trace = R_rel[0, 0] + R_rel[1, 1] + R_rel[2, 2]
        theta = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))

        rotvec = np.zeros(3, dtype=ipc_transforms.dtype)
        if theta > 1e-6:
            axis = np.array([
                R_rel[2, 1] - R_rel[1, 2],
                R_rel[0, 2] - R_rel[2, 0],
                R_rel[1, 0] - R_rel[0, 1],
            ], dtype=ipc_transforms.dtype)
            norm = np.linalg.norm(axis)
            if norm > 1e-8:
                rotvec = theta * axis / norm

        # Transform inertia to world frame
        I_world = R_current @ inertia_tensors[i] @ R_current.T

        out_torques[i] = rotation_strength / dt2 * (I_world @ rotvec)

    return out_forces, out_torques


def compute_link_contact_forces(
    force_gradients, link_indices, env_indices, vert_positions, link_centers,
    max_links, max_envs,
):
    """
    Compute contact forces and torques for rigid links from vertex gradients.
    Uses np.add.at for accumulation (equivalent to atomic add).

    Parameters
    ----------
    force_gradients : np.ndarray, shape (n, 3)
    link_indices : np.ndarray, shape (n,), int
    env_indices : np.ndarray, shape (n,), int
    vert_positions : np.ndarray, shape (n, 3)
    link_centers : np.ndarray, shape (n, 3)
    max_links : int
    max_envs : int

    Returns
    -------
    tuple of (out_forces, out_torques), each shape (max_links, max_envs, 3)
    """
    out_forces = np.zeros((max_links, max_envs, 3), dtype=force_gradients.dtype)
    out_torques = np.zeros((max_links, max_envs, 3), dtype=force_gradients.dtype)

    forces = -force_gradients  # (n, 3)
    r = vert_positions - link_centers  # (n, 3)
    torques = np.cross(r, forces)  # (n, 3)

    # Accumulate using np.add.at (unbuffered, like atomic add)
    np.add.at(out_forces, (link_indices, env_indices), forces)
    np.add.at(out_torques, (link_indices, env_indices), torques)

    return out_forces, out_torques


def extract_joint_mass_matrix(mass_mat, dof_start, joint_dof_indices, n_joints):
    """
    Extract the joint mass matrix submatrix from the full DOF mass matrix.

    Parameters
    ----------
    mass_mat : np.ndarray, shape (n_total_dofs, n_total_dofs) or (n_total_dofs, n_total_dofs, n_envs)
        Full mass matrix from rigid solver
    dof_start : int
        DOF start index for this entity
    joint_dof_indices : array-like of int
        DOF indices for each joint (local to entity)
    n_joints : int
        Number of joints

    Returns
    -------
    np.ndarray, shape (n_joints * n_joints,)
        Mass matrix submatrix in column-major (Fortran) order
    """
    # Build absolute DOF indices
    abs_indices = np.array([dof_start + d for d in joint_dof_indices[:n_joints]], dtype=int)

    # Extract submatrix using fancy indexing
    sub = mass_mat[np.ix_(abs_indices, abs_indices)]

    # Return flattened in column-major order (Fortran order = transpose then flatten)
    return sub.T.flatten()
