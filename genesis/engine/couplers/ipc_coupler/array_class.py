"""Quadrants data structures and kernels for the IPC coupler."""

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import DATA_ORIENTED, BASE_METACLASS, V_ANNOTATION, V, V_VEC


# ==================================== Quadrants data structures ====================================


@DATA_ORIENTED
class ArticulationState(metaclass=BASE_METACLASS):
    """GPU state for external articulation coupling (ref qpos + IPC joint displacements)."""

    ref_dof_prev: V_ANNOTATION
    delta_theta_ipc: V_ANNOTATION


def get_articulation_state(total_articu_qs, total_articu_joints, n_envs):
    """Allocate and return an ArticulationState with exact sizes."""
    return ArticulationState(
        ref_dof_prev=V(dtype=gs.qd_float, shape=(total_articu_qs, n_envs)),
        delta_theta_ipc=V(dtype=gs.qd_float, shape=(total_articu_joints, n_envs)),
    )


# ==================================== Kernels ====================================


@qd.kernel(fastcache=gs.use_fastcache)
def compute_external_force_kernel(
    n_links: gs.qd_int,
    contact_forces: V_ANNOTATION,
    contact_torques: V_ANNOTATION,
    abd_transforms: V_ANNOTATION,
    out_forces: V_ANNOTATION,
):
    """
    Compute 12D external force from contact forces and torques for each link.

    The 12D force layout is: [force (3), M_affine (9)] where
    M_affine = skew(torque) @ A and A is the rotation block of the ABD transform.
    """
    for i in range(n_links):
        force = -0.5 * contact_forces[i]
        for j in qd.static(range(3)):
            out_forces[i][j] = force[j]

        tau = -0.5 * contact_torques[i]

        A = qd.Matrix.zero(gs.qd_float, 3, 3)
        for row in range(3):
            for col in range(3):
                A[row, col] = abd_transforms[i][row, col]

        S = qd.Matrix.zero(gs.qd_float, 3, 3)
        S[0, 1] = -tau[2]
        S[0, 2] = tau[1]
        S[1, 0] = tau[2]
        S[1, 2] = -tau[0]
        S[2, 0] = -tau[1]
        S[2, 1] = tau[0]

        M_affine = S @ A

        idx = 3
        for row in range(3):
            for col in range(3):
                out_forces[i][idx] = M_affine[row, col]
                idx += 1


@qd.kernel(fastcache=gs.use_fastcache)
def compute_coupling_forces_kernel(
    n_links: gs.qd_int,
    ipc_transforms: qd.types.ndarray(),
    aim_transforms: qd.types.ndarray(),
    link_masses: qd.types.ndarray(),
    inertia_tensors: qd.types.ndarray(),
    translation_strength: gs.qd_float,
    rotation_strength: gs.qd_float,
    dt2: gs.qd_float,
    out_forces: qd.types.ndarray(),
    out_torques: qd.types.ndarray(),
):
    """Compute soft-constraint coupling forces and torques for all links in parallel."""
    for i in range(n_links):
        pos_current = qd.Vector([ipc_transforms[i, 0, 3], ipc_transforms[i, 1, 3], ipc_transforms[i, 2, 3]])
        pos_aim = qd.Vector([aim_transforms[i, 0, 3], aim_transforms[i, 1, 3], aim_transforms[i, 2, 3]])
        delta_pos = pos_current - pos_aim

        R_current = qd.Matrix.zero(gs.qd_float, 3, 3)
        R_aim = qd.Matrix.zero(gs.qd_float, 3, 3)
        for row in range(3):
            for col in range(3):
                R_current[row, col] = ipc_transforms[i, row, col]
                R_aim[row, col] = aim_transforms[i, row, col]

        mass = link_masses[i]
        linear_force = translation_strength * mass * delta_pos / dt2

        R_rel = R_current @ R_aim.transpose()
        trace = R_rel[0, 0] + R_rel[1, 1] + R_rel[2, 2]
        theta = qd.acos(qd.min(qd.max((trace - 1.0) / 2.0, -1.0), 1.0))

        rotvec = qd.Vector.zero(gs.qd_float, 3)
        if theta > 1e-6:
            axis_x = R_rel[2, 1] - R_rel[1, 2]
            axis_y = R_rel[0, 2] - R_rel[2, 0]
            axis_z = R_rel[1, 0] - R_rel[0, 1]
            norm = qd.sqrt(axis_x * axis_x + axis_y * axis_y + axis_z * axis_z)
            if norm > 1e-8:
                rotvec = theta * qd.Vector([axis_x, axis_y, axis_z]) / norm

        I_local = qd.Matrix.zero(gs.qd_float, 3, 3)
        for row in range(3):
            for col in range(3):
                I_local[row, col] = inertia_tensors[i, row, col]

        I_world = R_current @ I_local @ R_current.transpose()
        angular_torque = rotation_strength / dt2 * (I_world @ rotvec)

        for j in qd.static(range(3)):
            out_forces[i, j] = linear_force[j]
            out_torques[i, j] = angular_torque[j]


@qd.kernel(fastcache=gs.use_fastcache)
def compute_link_contact_forces_kernel(
    n_force_entries: gs.qd_int,
    force_gradients: V_ANNOTATION,
    vert_to_link_idx: V_ANNOTATION,
    vert_to_env_idx: V_ANNOTATION,
    vert_positions: V_ANNOTATION,
    link_centers: V_ANNOTATION,
    out_forces: V_ANNOTATION,
    out_torques: V_ANNOTATION,
):
    """Accumulate contact forces/torques on rigid links from per-vertex IPC gradients."""
    for i in range(n_force_entries):
        link_idx = vert_to_link_idx[i]
        env_idx = vert_to_env_idx[i]

        force = -force_gradients[i]

        for j in qd.static(range(3)):
            qd.atomic_add(out_forces[link_idx, env_idx][j], force[j])

        r = vert_positions[i] - link_centers[i]
        torque = r.cross(force)

        for j in qd.static(range(3)):
            qd.atomic_add(out_torques[link_idx, env_idx][j], torque[j])
