"""
Data structures, kernels, and utility functions for the IPC coupler.

Quadrants structs hold only state that cannot be derived from the rigid solver at runtime,
allocated with exact sizes at build time. Simple data access uses torch/numpy directly on
the rigid solver's existing fields. Kernels handle GPU-parallel math.
"""

import dataclasses

import numpy as np
import quadrants as qd
from scipy.spatial.transform import Rotation as R

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.array_class import DATA_ORIENTED, BASE_METACLASS, V_ANNOTATION, V, V_VEC


# ==================================== Quadrants data structures ====================================


@DATA_ORIENTED
class ArticulationState(metaclass=BASE_METACLASS):
    """
    GPU state for external articulation coupling.

    Stores only the data that cannot be read directly from the rigid solver at runtime:
    the reference qpos at the start of each IPC step and the IPC-computed joint
    displacements. Allocated with exact sizes at build time.

    Layout mirrors the rigid solver's flat qpos layout:
    ``ref_dof_prev`` shape is ``(total_articu_qs, n_envs)`` where ``total_articu_qs``
    is the sum of DOF counts across all articulated entities.
    ``delta_theta_ipc`` shape is ``(total_articu_joints, n_envs)`` where
    ``total_articu_joints`` is the sum of joint counts.
    """

    ref_dof_prev: V_ANNOTATION
    delta_theta_ipc: V_ANNOTATION


def get_articulation_state(total_articu_qs, total_articu_joints, n_envs):
    """Allocate and return an ArticulationState with exact sizes."""
    return ArticulationState(
        ref_dof_prev=V(dtype=gs.qd_float, shape=(total_articu_qs, n_envs)),
        delta_theta_ipc=V(dtype=gs.qd_float, shape=(total_articu_joints, n_envs)),
    )


@dataclasses.dataclass
class IPCCouplingData:
    """Pre-allocated numpy buffers for soft-constraint coupling force computation."""

    link_indices: np.ndarray
    env_indices: np.ndarray
    ipc_transforms: np.ndarray
    aim_transforms: np.ndarray
    link_masses: np.ndarray
    inertia_tensors: np.ndarray
    out_forces: np.ndarray
    out_torques: np.ndarray
    n_items: int = 0


def get_ipc_coupling_data(n_coupled_links):
    """Allocate and return an IPCCouplingData with exact sizes."""
    return IPCCouplingData(
        link_indices=np.empty(n_coupled_links, dtype=np.int32),
        env_indices=np.empty(n_coupled_links, dtype=np.int32),
        ipc_transforms=np.empty((n_coupled_links, 4, 4), dtype=gs.np_float),
        aim_transforms=np.empty((n_coupled_links, 4, 4), dtype=gs.np_float),
        link_masses=np.empty(n_coupled_links, dtype=gs.np_float),
        inertia_tensors=np.empty((n_coupled_links, 3, 3), dtype=gs.np_float),
        out_forces=np.empty((n_coupled_links, 3), dtype=gs.np_float),
        out_torques=np.empty((n_coupled_links, 3), dtype=gs.np_float),
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
    """
    Compute soft-constraint coupling forces and torques for all links in parallel.

    Uses numpy arrays for zero-copy performance.
    """
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
    """
    Accumulate contact forces and torques on rigid links from per-vertex IPC gradients.

    Uses atomic operations to accumulate contributions from multiple vertices per link.
    """
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


# ==================================== Utility functions ====================================


def find_target_link_for_fixed_merge(rigid_solver, link_idx):
    """
    Walk the kinematic tree upward, skipping FIXED joints, to find the effective
    parent link for IPC geometry merging.

    Returns
    -------
    int
        Index of the target link to merge into.
    """
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
    """
    Compute the relative rigid transform from ``from_link`` to ``to_link`` by walking
    up the kinematic tree.

    Returns
    -------
    tuple
        ``(R_acc, t_acc)`` — 3×3 rotation matrix and 3-vector translation that map
        points expressed in ``from_link`` frame into ``to_link`` frame.
    """
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


def is_robot_entity(entity):
    """Return True if the entity is a URDF/MJCF/Drone morph (heuristic for robots)."""
    try:
        return isinstance(entity.morph, (gs.morphs.URDF, gs.morphs.MJCF, gs.morphs.Drone))
    except Exception:
        return False


def compute_link_init_world_rotation(rigid_solver, link_idx):
    """
    Recursively compute the world rotation matrix of a link in the initial configuration.

    Uses ``link.quat`` (not joint.quat) because URDF joint-origin rotations are baked
    into the child link's body transformation in Genesis.
    """
    link = rigid_solver.links[link_idx]
    if link.parent_idx < 0:
        return gu.quat_to_R(link.quat)

    parent_rot = compute_link_init_world_rotation(rigid_solver, link.parent_idx)
    return parent_rot @ gu.quat_to_R(link.quat)


def extract_articulated_joints(entity):
    """
    Extract revolute and prismatic joints from a RigidEntity.

    Returns
    -------
    dict
        Keys: ``'revolute_joints'``, ``'prismatic_joints'``, ``'joint_qpos_indices'``,
        ``'joint_dof_indices'``, ``'n_joints'``.
    """
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
    """
    Partition entity indices by coupling mode.

    Parameters
    ----------
    entity_coupling_types : dict
        Maps ``entity_idx`` → coupling mode string.

    Returns
    -------
    dict
        Maps each coupling mode string to a list of entity indices.
    """
    result = {
        "two_way": [],
        "external_articulation": [],
        "ipc_only": [],
    }

    for entity_idx, coupling_type in entity_coupling_types.items():
        if coupling_type in result:
            result[coupling_type].append(entity_idx)

    return result


def build_ipc_scene_config(options, sim_options):
    """
    Build the ``uipc.Scene`` config dict from IPCCouplerOptions and SimOptions.

    Parameters
    ----------
    options : IPCCouplerOptions
    sim_options : SimOptions

    Returns
    -------
    dict
        Config dict ready to pass to ``Scene(config)``.
    """
    from uipc import Scene

    config = Scene.default_config()

    config["dt"] = sim_options.dt
    g = sim_options.gravity
    config["gravity"] = [[g[0]], [g[1]], [g[2]]]

    _set_if_not_none(config, ["newton", "velocity_tol"], options.newton_velocity_tol)
    _set_if_not_none(config, ["line_search", "max_iter"], options.line_search_max_iter)
    _set_if_not_none(config, ["linear_system", "tol_rate"], options.linear_system_tol_rate)
    _set_if_not_none(config, ["contact", "d_hat"], options.contact_d_hat)

    return config


def _set_if_not_none(config, keys, value):
    """Set a nested config value only if it is not None."""
    if value is None:
        return
    d = config
    for key in keys[:-1]:
        d = d[key]
    d[keys[-1]] = value


def read_ipc_geometry_metadata(geo):
    """
    Read ``solver_type``, ``env_idx``, and entity/link index from IPC geometry metadata.

    Returns
    -------
    tuple or None
        ``(solver_type, env_idx, idx)`` where ``idx`` is ``entity_idx`` for FEM/cloth
        or ``link_idx`` for rigid. Returns ``None`` if metadata is missing or invalid.
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
    Decompose a 4×4 homogeneous matrix into position and quaternion (wxyz).

    Returns
    -------
    tuple
        ``(pos, quat_wxyz)`` — arrays of shape ``(3,)`` and ``(4,)``.
    """
    pos = transform_4x4[:3, 3]
    quat_xyzw = R.from_matrix(transform_4x4[:3, :3]).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return pos, quat_wxyz


def build_link_transform_matrix(pos_3, quat_4):
    """
    Build a 4×4 transformation matrix from position and quaternion (wxyz).

    Uses the uipc Transform class for consistency with IPC.

    Returns
    -------
    np.ndarray
        4×4 transformation matrix.
    """
    from uipc import Transform, Vector3, Quaternion

    t = Transform.Identity()
    t.translate(Vector3.Values((float(pos_3[0]), float(pos_3[1]), float(pos_3[2]))))
    t.rotate(Quaternion(quat_4))
    return t.matrix().copy()
