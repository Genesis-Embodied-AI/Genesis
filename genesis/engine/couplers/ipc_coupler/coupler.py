from typing import TYPE_CHECKING

import numpy as np
import gstaichi as ti
from uipc.constitution import ElasticModuli2D, ElasticModuli

import genesis as gs
from genesis.options.solvers import IPCCouplerOptions
from genesis.repr_base import RBC
from .data import IPCTransformData, IPCCouplingData, ArticulationData
from .utils import (
    find_target_link_for_fixed_merge,
    compute_link_to_link_transform,
    compute_link_init_world_rotation,
    is_robot_entity,
    categorize_entities_by_coupling_type,
    build_ipc_scene_config,
    extract_articulated_joints,
    read_ipc_geometry_metadata,
    decompose_transform_matrix,
    build_link_transform_matrix,
)

if TYPE_CHECKING:
    from genesis.engine.simulator import Simulator

# Check if libuipc is available
try:
    import uipc

    UIPC_AVAILABLE = True
except ImportError:
    UIPC_AVAILABLE = False
    uipc = None


@ti.data_oriented
class IPCCoupler(RBC):
    """
    Coupler class for handling Incremental Potential Contact (IPC) simulation coupling.

    This coupler manages the communication between Genesis solvers and the IPC system,
    including rigid bodies (as ABD objects) and FEM bodies in a unified contact framework.
    """

    # ============================================================
    # Class Constants: Pre-allocated buffer sizes
    # ============================================================
    # These constants define the maximum capacities for Taichi field allocation.
    # Increase these values if you encounter capacity warnings during simulation.

    MAX_CONTACTS = 1000  # Maximum contact force entries
    MAX_LINKS = 200  # Maximum number of rigid links
    MAX_ENVS = 100  # Maximum number of parallel environments
    MAX_ABD_LINKS = 1000  # Maximum ABD link entries
    MAX_VERTEX_CONTACTS = 5000  # Maximum vertex-level contact entries
    MAX_QPOS_SIZE = 500  # Maximum qpos size per entity
    MAX_QPOS_BUFFER_LARGE = 2000  # Large qpos buffer for complex entities
    MAX_ARTICULATED_ENTITIES = 50  # Maximum articulated entities
    MAX_DOFS_PER_ENTITY = 100  # Maximum DOFs per articulated entity
    MAX_JOINTS_PER_ENTITY = 50  # Maximum joints per articulated entity

    # ============================================================
    # Section 1: Taichi Kernels
    # ============================================================
    # 1.1 Force computation kernels
    # 1.2 Batch data kernels (qpos read/write/compare)
    # 1.3 Transform and initialization kernels
    # 1.4 Articulation coupling kernels
    # ============================================================

    # ---------- 1.1 Force computation kernels ----------

    @ti.kernel
    def _compute_external_force_kernel(
        self,
        n_links: ti.i32,
        contact_forces: ti.template(),  # Taichi Vector field (max_contacts, 3)
        contact_torques: ti.template(),  # Taichi Vector field (max_contacts, 3)
        abd_transforms: ti.template(),  # Taichi Matrix field (max_contacts, 4, 4)
        out_forces: ti.template(),  # Taichi Vector field (max_contacts, 12)
    ):
        """
        Compute 12D external force from contact forces and torques.
        force = [force (3), M_affine (9)]
        where M_affine = skew(torque) * A, A is the rotation part of ABD transform
        """
        for i in range(n_links):
            # Copy force directly to first 3 components
            force = -0.5 * contact_forces[i]
            for j in ti.static(range(3)):
                out_forces[i][j] = force[j]

            # Extract torque
            tau = -0.5 * contact_torques[i]

            # Extract rotation matrix A (3x3) from ABD transform (first 3x3 block)
            A = ti.Matrix.zero(ti.f32, 3, 3)
            for row in range(3):
                for col in range(3):
                    A[row, col] = abd_transforms[i][row, col]

            # Compute skew-symmetric matrix S of tau
            S = ti.Matrix.zero(ti.f32, 3, 3)
            S[0, 0] = 0.0
            S[0, 1] = -tau[2]
            S[0, 2] = tau[1]
            S[1, 0] = tau[2]
            S[1, 1] = 0.0
            S[1, 2] = -tau[0]
            S[2, 0] = -tau[1]
            S[2, 1] = tau[0]
            S[2, 2] = 0.0

            # Compute M_affine = S * A (row-major order)
            M_affine = S @ A

            # Write M_affine to last 9 components (row-major)
            idx = 3
            for row in range(3):
                for col in range(3):
                    out_forces[i][idx] = M_affine[row, col]
                    idx += 1

    @ti.kernel
    def _compute_coupling_forces_kernel(
        self,
        n_links: ti.i32,
        ipc_transforms: ti.template(),  # Taichi Matrix field
        aim_transforms: ti.template(),  # Taichi Matrix field
        link_masses: ti.template(),  # Taichi field
        inertia_tensors: ti.template(),  # Taichi Matrix field
        translation_strength: ti.f32,
        rotation_strength: ti.f32,
        dt2: ti.f32,
        out_forces: ti.template(),  # Taichi Vector field
        out_torques: ti.template(),  # Taichi Vector field
    ):
        """
        Compute coupling forces and torques for all links in parallel.
        """
        for i in range(n_links):
            # Extract positions (Matrix field: ipc_transforms[i] returns a 4x4 matrix)
            pos_current = ti.Vector([ipc_transforms[i][0, 3], ipc_transforms[i][1, 3], ipc_transforms[i][2, 3]])
            pos_aim = ti.Vector([aim_transforms[i][0, 3], aim_transforms[i][1, 3], aim_transforms[i][2, 3]])
            delta_pos = pos_current - pos_aim

            # Extract rotation matrices
            R_current = ti.Matrix.zero(ti.f32, 3, 3)
            R_aim = ti.Matrix.zero(ti.f32, 3, 3)
            for row in range(3):
                for col in range(3):
                    R_current[row, col] = ipc_transforms[i][row, col]
                    R_aim[row, col] = aim_transforms[i][row, col]

            # Compute linear force
            mass = link_masses[i]

            linear_force = translation_strength * mass * delta_pos / dt2
            # Compute relative rotation: R_rel = R_current @ R_aim^T
            R_rel = R_current @ R_aim.transpose()

            # Extract rotation vector from R_rel using Rodrigues formula
            # trace(R) = 1 + 2*cos(theta)
            trace = R_rel[0, 0] + R_rel[1, 1] + R_rel[2, 2]
            theta = ti.acos(ti.min(ti.max((trace - 1.0) / 2.0, -1.0), 1.0))

            # Rotation axis (when theta != 0)
            rotvec = ti.Vector.zero(ti.f32, 3)
            if theta > 1e-6:
                axis_x = R_rel[2, 1] - R_rel[1, 2]
                axis_y = R_rel[0, 2] - R_rel[2, 0]
                axis_z = R_rel[1, 0] - R_rel[0, 1]
                norm = ti.sqrt(axis_x * axis_x + axis_y * axis_y + axis_z * axis_z)
                if norm > 1e-8:
                    rotvec = theta * ti.Vector([axis_x, axis_y, axis_z]) / norm

            # Load inertia tensor (Matrix field: inertia_tensors[i] returns a 3x3 matrix)
            I_local = ti.Matrix.zero(ti.f32, 3, 3)
            for row in range(3):
                for col in range(3):
                    I_local[row, col] = inertia_tensors[i][row, col]

            # Transform to world frame: I_world = R_current @ I_local @ R_current^T
            I_world = R_current @ I_local @ R_current.transpose()

            # Compute angular torque
            angular_torque = rotation_strength / dt2 * (I_world @ rotvec)

            # Store results (Vector field: out_forces[i] returns a 3D vector)
            for j in ti.static(range(3)):
                out_forces[i][j] = linear_force[j]
                out_torques[i][j] = angular_torque[j]

    @ti.kernel
    def _accumulate_contact_forces_kernel(
        self,
        n_contacts: ti.i32,
        contact_vert_indices: ti.template(),  # Taichi field (max_contacts,)
        contact_gradients: ti.template(),  # Taichi Vector field (max_contacts, 3)
        vert_to_link: ti.template(),  # Taichi field (max_vertices,) mapping vert -> link_idx (-1 if invalid)
        vert_positions: ti.template(),  # Taichi Vector field (max_vertices, 3)
        link_centers: ti.template(),  # Taichi Vector field (max_links, 3)
        out_forces: ti.template(),  # Taichi Vector field (max_links, 3)
        out_torques: ti.template(),  # Taichi Vector field (max_links, 3)
    ):
        """
        Accumulate contact forces and torques for all vertices in parallel.
        LEGACY: Currently unused, but kept for potential future use.
        """
        for i in range(n_contacts):
            vert_idx = contact_vert_indices[i]
            link_idx = vert_to_link[vert_idx]

            if link_idx >= 0:  # Valid link
                # Force is negative gradient
                force = -contact_gradients[i]

                # Atomic add force
                for j in ti.static(range(3)):
                    ti.atomic_add(out_forces[link_idx][j], force[j])

                # Compute torque: τ = r × F
                contact_pos = vert_positions[vert_idx]
                center_pos = link_centers[link_idx]
                r = contact_pos - center_pos
                torque = r.cross(force)

                # Atomic add torque
                for j in ti.static(range(3)):
                    ti.atomic_add(out_torques[link_idx][j], torque[j])

    @ti.kernel
    def _compute_link_contact_forces_kernel(
        self,
        n_force_entries: ti.i32,
        force_gradients: ti.template(),  # Taichi Vector field (max_vertex_contacts, 3)
        vert_to_link_idx: ti.template(),  # Taichi field (max_vertex_contacts,)
        vert_to_env_idx: ti.template(),  # Taichi field (max_vertex_contacts,)
        vert_positions: ti.template(),  # Taichi Vector field (max_vertex_contacts, 3)
        link_centers: ti.template(),  # Taichi Vector field (max_vertex_contacts, 3)
        out_forces: ti.template(),  # Taichi Vector field (max_links, max_envs, 3)
        out_torques: ti.template(),  # Taichi Vector field (max_links, max_envs, 3)
    ):
        """
        Compute contact forces and torques for rigid links from vertex gradients.
        Uses atomic operations to accumulate forces from multiple vertices per link.
        """
        for i in range(n_force_entries):
            link_idx = vert_to_link_idx[i]
            env_idx = vert_to_env_idx[i]

            # Force is negative gradient
            force = -force_gradients[i]

            # Atomic add force
            for j in ti.static(range(3)):
                ti.atomic_add(out_forces[link_idx, env_idx][j], force[j])

            # Compute torque: τ = r × F
            contact_pos = vert_positions[i]
            center_pos = link_centers[i]
            r = contact_pos - center_pos
            torque = r.cross(force)

            # Atomic add torque
            for j in ti.static(range(3)):
                ti.atomic_add(out_torques[link_idx, env_idx][j], torque[j])

    # ---------- 1.2 Batch data kernels (qpos read/write/compare) ----------

    @ti.kernel
    def _batch_read_qpos_kernel(
        self,
        qpos_field: ti.types.ndarray(),  # Source qpos field (n_dofs, n_envs) - from Genesis, must be ndarray
        q_start: ti.i32,
        n_qs: ti.i32,
        env_idx: ti.i32,
        out_qpos: ti.template(),  # Output Taichi field (max_qpos_size,)
    ):
        """
        Batch read qpos for a specific entity and environment.
        Input must be ndarray because it comes from Genesis (PyTorch -> numpy).
        Output is Taichi field to avoid one copy.
        """
        for i in range(n_qs):
            out_qpos[i] = qpos_field[q_start + i, env_idx]

    @ti.kernel
    def _compare_qpos_kernel(
        self,
        n_entries: ti.i32,
        qpos_current: ti.template(),  # Taichi field (max_qpos_size,)
        qpos_stored: ti.types.ndarray(),  # (n_entries,) - must be ndarray (from stored dict)
        tolerance: ti.f32,
        out_modified: ti.template(),  # Taichi field (1,) - output 1 if modified, 0 otherwise
    ):
        """
        Compare two qpos arrays and detect if modified beyond tolerance.
        qpos_current is Taichi field (from qpos_buffer).
        qpos_stored is ndarray (from external dict storage).
        Output is Taichi field to avoid one copy.
        """
        modified = 0
        for i in range(n_entries):
            diff = ti.abs(qpos_current[i] - qpos_stored[i])
            if diff > tolerance:
                modified = 1
        out_modified[0] = modified

    @ti.kernel
    def _compare_qpos_field_kernel(
        self,
        n_entries: ti.i32,
        qpos_current: ti.template(),  # Taichi field (max_qpos_size,) - current qpos buffer
        stored_qpos: ti.template(),  # Taichi field (max_entities, max_envs, max_qpos_size)
        entity_idx: ti.i32,
        env_idx: ti.i32,
        tolerance: ti.f32,
        out_modified: ti.template(),  # Taichi field (1,) - output 1 if modified, 0 otherwise
    ):
        """
        Compare current qpos buffer with stored qpos in TransformData field.
        Both inputs are Taichi fields (zero numpy copy).
        """
        modified = 0
        for i in range(n_entries):
            diff = ti.abs(qpos_current[i] - stored_qpos[entity_idx, env_idx, i])
            if diff > tolerance:
                modified = 1
        out_modified[0] = modified

    # ---------- 1.3 Transform and initialization kernels ----------

    @ti.kernel
    def _batch_pos_quat_to_transform_kernel(
        self,
        n_links: ti.i32,
        positions: ti.template(),  # Taichi Vector field (max_links, 3)
        quaternions: ti.template(),  # Taichi Vector field (max_links, 4) - wxyz format
        out_transforms: ti.template(),  # Taichi Matrix field (max_links, 4, 4)
    ):
        """
        Convert batch of positions and quaternions to 4x4 transform matrices.
        Quaternion format: [w, x, y, z]
        Uses Taichi fields to avoid numpy copying overhead.
        """
        for i in range(n_links):
            # Extract quaternion components
            w = quaternions[i][0]
            x = quaternions[i][1]
            y = quaternions[i][2]
            z = quaternions[i][3]

            # Compute rotation matrix from quaternion
            # R = I + 2*s*K + 2*K^2, where s = w, K = skew(x,y,z)
            xx = x * x
            yy = y * y
            zz = z * z
            xy = x * y
            xz = x * z
            yz = y * z
            wx = w * x
            wy = w * y
            wz = w * z

            # Rotation matrix (3x3)
            R00 = 1.0 - 2.0 * (yy + zz)
            R01 = 2.0 * (xy - wz)
            R02 = 2.0 * (xz + wy)

            R10 = 2.0 * (xy + wz)
            R11 = 1.0 - 2.0 * (xx + zz)
            R12 = 2.0 * (yz - wx)

            R20 = 2.0 * (xz - wy)
            R21 = 2.0 * (yz + wx)
            R22 = 1.0 - 2.0 * (xx + yy)

            # Build 4x4 transform matrix
            # [R  t]
            # [0  1]
            out_transforms[i][0, 0] = R00
            out_transforms[i][0, 1] = R01
            out_transforms[i][0, 2] = R02
            out_transforms[i][0, 3] = positions[i][0]

            out_transforms[i][1, 0] = R10
            out_transforms[i][1, 1] = R11
            out_transforms[i][1, 2] = R12
            out_transforms[i][1, 3] = positions[i][1]

            out_transforms[i][2, 0] = R20
            out_transforms[i][2, 1] = R21
            out_transforms[i][2, 2] = R22
            out_transforms[i][2, 3] = positions[i][2]

            out_transforms[i][3, 0] = 0.0
            out_transforms[i][3, 1] = 0.0
            out_transforms[i][3, 2] = 0.0
            out_transforms[i][3, 3] = 1.0

    @ti.kernel
    def _store_link_states_kernel(
        self,
        transform_data: ti.template(),
        links_pos: ti.types.ndarray(),  # (n_links, 3) from Genesis
        links_quat: ti.types.ndarray(),  # (n_links, 4) from Genesis
        env_idx: ti.i32,
        n_links: ti.i32,
    ):
        """
        Store link positions and quaternions to Taichi fields.
        """
        for link_idx in range(n_links):
            # Store position
            transform_data.stored_link_pos[link_idx, env_idx] = ti.Vector(
                [links_pos[link_idx, 0], links_pos[link_idx, 1], links_pos[link_idx, 2]]
            )

            # Store quaternion
            transform_data.stored_link_quat[link_idx, env_idx] = ti.Vector(
                [
                    links_quat[link_idx, 0],
                    links_quat[link_idx, 1],
                    links_quat[link_idx, 2],
                    links_quat[link_idx, 3],
                ]
            )

            # Mark as valid
            transform_data.stored_link_valid[link_idx, env_idx] = 1

    @ti.kernel
    def _store_qpos_kernel(
        self,
        transform_data: ti.template(),
        solver_qpos: ti.types.ndarray(),
        entity_idx: ti.i32,
        env_idx: ti.i32,
        q_start: ti.i32,
        n_qs: ti.i32,
    ):
        """
        Store qpos for a single entity to Taichi fields.
        Input is ndarray from rigid solver (when gs.use_ndarray=True).
        """
        for i in range(n_qs):
            transform_data.stored_qpos[entity_idx, env_idx, i] = solver_qpos[q_start + i, env_idx]

    @ti.kernel
    def _store_qpos_kernel_field(
        self,
        transform_data: ti.template(),
        solver_qpos: ti.template(),  # ti.field (performance_mode=True)
        entity_idx: ti.i32,
        env_idx: ti.i32,
        q_start: ti.i32,
        n_qs: ti.i32,
    ):
        """
        Store qpos from ti.field (for performance_mode=True).
        Same logic as _store_qpos_kernel but accepts ti.field.
        """
        for i in range(n_qs):
            transform_data.stored_qpos[entity_idx, env_idx, i] = solver_qpos[q_start + i, env_idx]

    @ti.kernel
    def _copy_stored_qpos_to_articulation_kernel(
        self,
        transform_data: ti.template(),
        articulation_data: ti.template(),
        n_envs: ti.i32,
    ):
        """
        Copy qpos from stored_qpos (TransformData) to qpos_current (ArticulationData).
        Parallelized over all entities and environments.
        """
        n_entities = articulation_data.n_entities[None]

        for idx in range(n_entities):
            entity_idx = articulation_data.entity_indices[idx]
            n_dofs = transform_data.stored_qpos_size[entity_idx]

            for env_idx, dof_idx in ti.ndrange(n_envs, n_dofs):
                # Copy from stored_qpos to qpos_current
                articulation_data.qpos_current[idx, env_idx, dof_idx] = transform_data.stored_qpos[
                    entity_idx, env_idx, dof_idx
                ]

    @ti.kernel
    def _init_mappings_and_flags_kernel(self, transform_data: ti.template(), max_links: ti.i32):
        """Initialize all mapping and flag arrays to default values."""
        for link_idx in range(max_links):
            transform_data.link_to_entity_map[link_idx] = -1
            transform_data.entity_base_link_map[link_idx] = -1
            transform_data.entity_n_links_map[link_idx] = 0
            transform_data.ipc_only_flags[link_idx] = 0
            transform_data.ipc_filter_flags[link_idx] = 0

    @ti.kernel
    def _init_user_modified_flags_kernel(self, transform_data: ti.template(), max_entities: ti.i32, max_envs: ti.i32):
        """Initialize user-modified entity flags to 0."""
        for entity_idx, env_idx in ti.ndrange(max_entities, max_envs):
            transform_data.user_modified_flags[entity_idx, env_idx] = 0

    @ti.kernel
    def _init_filter_input_kernel(self, transform_data: ti.template(), n_items: ti.i32, max_envs: ti.i32):
        """Initialize filter input validity flags to 0."""
        for i, env in ti.ndrange(n_items, max_envs):
            transform_data.input_valid[i, env] = 0

    @ti.kernel
    def _filter_and_collect_batch_outputs_kernel(
        self,
        transform_data: ti.template(),
        n_items: ti.i32,
        max_envs: ti.i32,
        ipc_only: ti.i32,  # 1 for True, 0 for False
    ):
        """
        Complete pipeline kernel:
        1. Filter links based on ipc_only flag
        2. Extract pos/quat from transform matrices
        3. Separate simple vs complex cases
        4. Compact output per environment into batch arrays
        """
        # Reset batch output counts
        for env in range(max_envs):
            transform_data.output_count_per_env[env] = 0

        # Reset complex case flags
        for entity_idx, env in ti.ndrange(200, max_envs):  # max_links as max entities
            transform_data.complex_case_flags[entity_idx, env] = 0

        # Process all (link, env) pairs in parallel
        for i, env in ti.ndrange(n_items, max_envs):
            if transform_data.input_valid[i, env] == 0:
                continue

            link_idx = transform_data.input_link_indices[i]
            env_idx = transform_data.input_env_indices[i, env]
            entity_idx = transform_data.link_to_entity_map[link_idx]

            if entity_idx < 0:
                continue

            # Check user modification flag
            if transform_data.user_modified_flags[entity_idx, env_idx] == 1:
                continue

            # Check filtering criteria
            passes_filter = 0
            if ipc_only == 1:
                # Must be both IPC-only AND in IPC filters
                if transform_data.ipc_only_flags[link_idx] == 1 and transform_data.ipc_filter_flags[link_idx] == 1:
                    passes_filter = 1
            else:
                # Must be in IPC filters
                if transform_data.ipc_filter_flags[link_idx] == 1:
                    passes_filter = 1

            if passes_filter == 0:
                continue

            # Check if this is a simple case: single base link
            base_link_idx = transform_data.entity_base_link_map[entity_idx]
            n_links = transform_data.entity_n_links_map[entity_idx]

            is_simple_case = n_links == 1 and link_idx == base_link_idx

            if is_simple_case:
                # Extract rotation matrix (3x3)
                R = ti.Matrix(
                    [
                        [
                            transform_data.input_transforms[i, env][0, 0],
                            transform_data.input_transforms[i, env][0, 1],
                            transform_data.input_transforms[i, env][0, 2],
                        ],
                        [
                            transform_data.input_transforms[i, env][1, 0],
                            transform_data.input_transforms[i, env][1, 1],
                            transform_data.input_transforms[i, env][1, 2],
                        ],
                        [
                            transform_data.input_transforms[i, env][2, 0],
                            transform_data.input_transforms[i, env][2, 1],
                            transform_data.input_transforms[i, env][2, 2],
                        ],
                    ]
                )

                # Extract position
                pos = ti.Vector(
                    [
                        transform_data.input_transforms[i, env][0, 3],
                        transform_data.input_transforms[i, env][1, 3],
                        transform_data.input_transforms[i, env][2, 3],
                    ]
                )

                # Convert rotation matrix to quaternion using Shepperd's method
                trace = R[0, 0] + R[1, 1] + R[2, 2]

                qw = 0.0
                qx = 0.0
                qy = 0.0
                qz = 0.0

                if trace > 0.0:
                    s = ti.sqrt(trace + 1.0)
                    qw = s * 0.5
                    s = 0.5 / s
                    qx = (R[2, 1] - R[1, 2]) * s
                    qy = (R[0, 2] - R[2, 0]) * s
                    qz = (R[1, 0] - R[0, 1]) * s
                else:
                    if R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
                        s = ti.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                        qx = s * 0.5
                        s = 0.5 / s
                        qw = (R[2, 1] - R[1, 2]) * s
                        qy = (R[0, 1] + R[1, 0]) * s
                        qz = (R[0, 2] + R[2, 0]) * s
                    elif R[1, 1] > R[2, 2]:
                        s = ti.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                        qy = s * 0.5
                        s = 0.5 / s
                        qw = (R[0, 2] - R[2, 0]) * s
                        qx = (R[0, 1] + R[1, 0]) * s
                        qz = (R[1, 2] + R[2, 1]) * s
                    else:
                        s = ti.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                        qz = s * 0.5
                        s = 0.5 / s
                        qw = (R[1, 0] - R[0, 1]) * s
                        qx = (R[0, 2] + R[2, 0]) * s
                        qy = (R[1, 2] + R[2, 1]) * s

                # Atomically add to batch output for this environment
                idx = ti.atomic_add(transform_data.output_count_per_env[env_idx], 1)
                transform_data.output_link_idx[env_idx, idx] = link_idx
                transform_data.output_pos[env_idx, idx] = pos
                transform_data.output_quat[env_idx, idx] = ti.Vector([qw, qx, qy, qz])
                transform_data.output_entity_idx[env_idx, idx] = entity_idx
            else:
                # Complex case: mark for later IK processing
                transform_data.complex_case_flags[entity_idx, env_idx] = 1

    # ==================== Articulation Coupling Kernels ====================

    @ti.kernel
    def _compute_delta_theta_tilde_kernel(
        self,
        articulation_data: ti.template(),
        n_envs: ti.i32,
    ):
        """
        Compute target joint displacement: delta_theta_tilde = qpos_current - ref_dof_prev
        This represents the displacement Genesis wants to achieve.
        Parallelized over all entities and environments.
        """
        n_entities = articulation_data.n_entities[None]

        for entity_idx, env_idx in ti.ndrange(n_entities, n_envs):
            n_joints = articulation_data.entity_n_joints[entity_idx]

            for joint_idx in range(n_joints):
                # Get the DOF index for this joint
                dof_idx = articulation_data.joint_dof_indices[entity_idx, joint_idx]

                # Compute delta_theta_tilde = qpos_current - ref_dof_prev
                qpos_curr = articulation_data.qpos_current[entity_idx, env_idx, dof_idx]
                qpos_prev = articulation_data.ref_dof_prev[entity_idx, env_idx, dof_idx]
                articulation_data.delta_theta_tilde[entity_idx, env_idx, joint_idx] = qpos_curr - qpos_prev

    @ti.kernel
    def _compute_qpos_new_kernel(
        self,
        articulation_data: ti.template(),
        n_envs: ti.i32,
    ):
        """
        Compute new qpos from ref_dof_prev and delta_theta_ipc.
        Parallelized over all entities, environments, and DOFs.

        qpos_new[dof_idx] = ref_dof_prev[dof_idx] + delta_theta_ipc[joint_idx]
        (for DOFs corresponding to joints, other DOFs keep ref_dof_prev)
        """
        n_entities = articulation_data.n_entities[None]

        for entity_idx, env_idx in ti.ndrange(n_entities, n_envs):
            n_dofs = articulation_data.entity_n_dofs[entity_idx]
            n_joints = articulation_data.entity_n_joints[entity_idx]

            # First, copy all ref_dof_prev to qpos_new
            for dof_idx in range(n_dofs):
                articulation_data.qpos_new[entity_idx, env_idx, dof_idx] = articulation_data.ref_dof_prev[
                    entity_idx, env_idx, dof_idx
                ]

            # Then, update DOFs corresponding to joints
            for joint_idx in range(n_joints):
                dof_idx = articulation_data.joint_dof_indices[entity_idx, joint_idx]
                if dof_idx < n_dofs:
                    delta_theta = articulation_data.delta_theta_ipc[entity_idx, env_idx, joint_idx]
                    articulation_data.qpos_new[entity_idx, env_idx, dof_idx] = (
                        articulation_data.ref_dof_prev[entity_idx, env_idx, dof_idx] + delta_theta
                    )

    @ti.kernel
    def _batch_read_qpos_from_solver_kernel(
        self,
        articulation_data: ti.template(),
        solver_qpos: ti.types.ndarray(),  # ndarray/field: qpos (n_total_dofs, n_envs)
        n_envs: ti.i32,
    ):
        """
        Batch read qpos directly from rigid solver's qpos (ndarray or field).
        Parallelized over all entities and environments.
        Works with both ti.ndarray (use_ndarray=True) and ti.field (performance_mode=True).
        """
        n_entities = articulation_data.n_entities[None]

        for entity_idx in range(n_entities):
            n_dofs = articulation_data.entity_n_dofs[entity_idx]
            dof_start = articulation_data.entity_dof_start[entity_idx]

            for env_idx, dof_idx in ti.ndrange(n_envs, n_dofs):
                # Read from solver: solver_qpos[dof_start + dof_idx, env_idx]
                articulation_data.qpos_current[entity_idx, env_idx, dof_idx] = solver_qpos[dof_start + dof_idx, env_idx]

    @ti.kernel
    def _batch_write_qpos_kernel(
        self,
        articulation_data: ti.template(),
        qpos_out: ti.types.ndarray(),  # (n_entities, max_envs, max_dofs)
        n_envs: ti.i32,
    ):
        """
        Batch write qpos_new from Taichi fields to output array.
        Parallelized over all entities and environments.
        """
        n_entities = articulation_data.n_entities[None]

        for entity_idx, env_idx in ti.ndrange(n_entities, n_envs):
            n_dofs = articulation_data.entity_n_dofs[entity_idx]

            for dof_idx in range(n_dofs):
                qpos_out[entity_idx, env_idx, dof_idx] = articulation_data.qpos_new[entity_idx, env_idx, dof_idx]

    @ti.kernel
    def _extract_joint_mass_matrix_kernel(
        self,
        articulation_data: ti.template(),
        solver_mass_mat: ti.types.ndarray(),  # ndarray/field: mass matrix (n_total_dofs, n_total_dofs, n_envs)
        entity_idx: ti.i32,
        env_idx: ti.i32,
    ):
        """
        Extract the mass matrix submatrix for joints from the full DOF mass matrix.
        Stores result in column-major order for IPC (transposed).
        Works with both ti.ndarray (use_ndarray=True) and ti.field (performance_mode=True).

        Parameters:
        - solver_mass_mat: Full mass matrix from rigid solver (n_total_dofs, n_total_dofs, n_envs)
        - entity_idx: Index of the articulated entity
        - env_idx: Environment index
        """
        dof_start = articulation_data.entity_dof_start[entity_idx]
        n_joints = articulation_data.entity_n_joints[entity_idx]
        # print("Extracting mass matrix for entity ", entity_idx, " in env ", env_idx)
        # print("DOF start: ", dof_start, ", Number of joints: ", n_joints)
        # Extract joint submatrix and store in column-major order
        for i in range(n_joints):
            dof_i = articulation_data.joint_dof_indices[entity_idx, i]
            for j in range(n_joints):
                dof_j = articulation_data.joint_dof_indices[entity_idx, j]
                # Store in column-major order: mass_matrix[j * n_joints + i] = M[i, j]
                # This is equivalent to transposing during flatten
                articulation_data.mass_matrix[entity_idx, j * n_joints + i] = solver_mass_mat[
                    dof_start + dof_i, dof_start + dof_j, env_idx
                ]
                # print mass matrix access for debugging
                # print(
                #     "Mass matrix access: M[",
                #     dof_start + dof_i,
                #     ",",
                #     dof_start + dof_j,
                #     ",",
                #     env_idx,
                #     "] = ",
                #     solver_mass_mat[dof_start + dof_i, dof_start + dof_j, env_idx],
                # )

    @ti.kernel
    def _extract_joint_mass_matrix_kernel_field(
        self,
        articulation_data: ti.template(),
        solver_mass_mat: ti.template(),  # ti.field: mass matrix (performance_mode=True)
        entity_idx: ti.i32,
        env_idx: ti.i32,
    ):
        """
        Extract mass matrix from ti.field (for performance_mode=True).
        Same logic as _extract_joint_mass_matrix_kernel but accepts ti.field.
        """
        dof_start = articulation_data.entity_dof_start[entity_idx]
        n_joints = articulation_data.entity_n_joints[entity_idx]

        # print("Extracting mass matrix for entity ", entity_idx, " in env ", env_idx)
        # print("DOF start: ", dof_start, ", Number of joints: ", n_joints)

        for i in range(n_joints):
            dof_i = articulation_data.joint_dof_indices[entity_idx, i]
            for j in range(n_joints):
                dof_j = articulation_data.joint_dof_indices[entity_idx, j]
                articulation_data.mass_matrix[entity_idx, j * n_joints + i] = solver_mass_mat[
                    dof_start + dof_i, dof_start + dof_j, env_idx
                ]
                # print mass matrix access for debugging
                # print(
                #     "Mass matrix access: M[",
                #     dof_start + dof_i,
                #     ",",
                #     dof_start + dof_j,
                #     ",",
                #     env_idx,
                #     "] = ",
                #     solver_mass_mat[dof_start + dof_i, dof_start + dof_j, env_idx],
                # )

    @ti.kernel
    def _update_ref_dof_prev_kernel(
        self,
        articulation_data: ti.template(),
        n_envs: ti.i32,
    ):
        """
        Update ref_dof_prev from qpos_new for next timestep.
        Parallelized over all entities and environments.
        """
        n_entities = articulation_data.n_entities[None]

        for entity_idx, env_idx in ti.ndrange(n_entities, n_envs):
            n_dofs = articulation_data.entity_n_dofs[entity_idx]

            for dof_idx in range(n_dofs):
                articulation_data.ref_dof_prev[entity_idx, env_idx, dof_idx] = articulation_data.qpos_new[
                    entity_idx, env_idx, dof_idx
                ]

    # ============================================================
    # Section 2: Initialization & Setup
    # ============================================================
    # Data classes are defined in ipc_data.py:
    # - IPCTransformData, IPCCouplingData, ArticulationData

    def __init__(self, simulator: "Simulator", options: "IPCCouplerOptions") -> None:
        """
        Initialize IPC Coupler.

        Parameters
        ----------
        simulator : Simulator
            The simulator containing all solvers
        options : IPCCouplerOptions
            IPC configuration options
        """
        # Check if uipc is available
        if not UIPC_AVAILABLE:
            raise ImportError(
                "libuipc is required for IPC coupling but not found.\n"
                "Please build and install libuipc from source:\n"
                "https://github.com/spiriMirror/libuipc"
            )

        self.sim = simulator
        self.options = options

        # Store solver references
        self.rigid_solver = self.sim.rigid_solver
        self.fem_solver = self.sim.fem_solver

        # IPC system components (will be initialized in build)
        self._ipc_engine = None
        self._ipc_world = None
        self._ipc_scene = None
        self._ipc_abd = None
        self._ipc_stk = None
        self._ipc_abd_contact = None
        self._ipc_fem_contact = None
        self._ipc_scene_subscenes = {}
        self._use_subscenes = False  # Will be set in _init_ipc based on number of environments

        # Per-entity coupling type: maps entity_idx -> coupling_type
        # Valid types: None (default, not in IPC), "two_way_soft_constraint", "external_articulation", "ipc_only"
        self._entity_coupling_types = {}

        # IPC link filter: maps entity_idx -> set of link_idx to include in IPC
        # Only used for "two_way_soft_constraint" type entities to filter which links participate
        # If entity_idx not in dict, all links of that entity participate
        self._ipc_link_filters = {}

        # Storage for Genesis rigid body states before IPC advance
        # Maps link_idx -> {env_idx: transform_matrix}
        self._genesis_stored_states = {}

        # Storage for IPC contact forces on rigid links (both coupling mode)
        # Maps link_idx -> {env_idx: {'force': np.array, 'torque': np.array}}
        self._ipc_contact_forces = {}

        # Storage for external force data for rigid links
        # Maps (link_idx, env_idx) -> force_vector (12D numpy array)
        self._external_force_data = {}

        # Pre-computed mapping from vertex index to rigid link (built once during IPC setup)
        # Maps global_vertex_idx -> (link_idx, env_idx, local_vertex_idx)
        self._vertex_to_link_mapping = {}
        # Global vertex offset for tracking vertex indices across all geometries
        self._global_vertex_offset = 0

        # Pre-allocated Taichi fields for contact force processing
        # Use class constants for buffer sizes (can be overridden by subclassing)
        self.contact_forces_ti = ti.Vector.field(3, dtype=gs.ti_float, shape=self.MAX_CONTACTS)
        self.contact_torques_ti = ti.Vector.field(3, dtype=gs.ti_float, shape=self.MAX_CONTACTS)
        self.abd_transforms_ti = ti.Matrix.field(4, 4, dtype=gs.ti_float, shape=self.MAX_CONTACTS)
        self.out_forces_ti = ti.Vector.field(12, dtype=gs.ti_float, shape=self.MAX_CONTACTS)
        self.link_indices_ti = ti.field(dtype=ti.i32, shape=self.MAX_CONTACTS)
        self.env_indices_ti = ti.field(dtype=ti.i32, shape=self.MAX_CONTACTS)

        # Pre-allocated fields for link contact force computation
        self.link_contact_forces_out = ti.Vector.field(3, dtype=gs.ti_float, shape=(self.MAX_LINKS, self.MAX_ENVS))
        self.link_contact_torques_out = ti.Vector.field(3, dtype=gs.ti_float, shape=(self.MAX_LINKS, self.MAX_ENVS))

        # Fields for storing vertex-level contact data (for kernel processing)
        self.vertex_force_gradients = ti.Vector.field(3, dtype=gs.ti_float, shape=self.MAX_VERTEX_CONTACTS)
        self.vertex_link_indices = ti.field(dtype=ti.i32, shape=self.MAX_VERTEX_CONTACTS)
        self.vertex_env_indices = ti.field(dtype=ti.i32, shape=self.MAX_VERTEX_CONTACTS)
        self.vertex_positions_world = ti.Vector.field(3, dtype=gs.ti_float, shape=self.MAX_VERTEX_CONTACTS)
        self.vertex_link_centers = ti.Vector.field(3, dtype=gs.ti_float, shape=self.MAX_VERTEX_CONTACTS)

        # Fields for batch qpos operations
        self.qpos_buffer = ti.field(dtype=gs.ti_float, shape=self.MAX_QPOS_SIZE)
        self.qpos_comparison_result = ti.field(dtype=ti.i32, shape=1)

        # Fields for batch transform operations
        self.batch_positions = ti.Vector.field(3, dtype=gs.ti_float, shape=self.MAX_LINKS)
        self.batch_quaternions = ti.Vector.field(4, dtype=gs.ti_float, shape=self.MAX_LINKS)
        self.batch_transforms = ti.Matrix.field(4, 4, dtype=gs.ti_float, shape=self.MAX_LINKS)

        # Initialize data-oriented transform data structure
        self.transform_data = IPCTransformData(self.MAX_LINKS, self.MAX_ENVS, self.MAX_ABD_LINKS, self.MAX_QPOS_SIZE)

        # Initialize data-oriented coupling data structure
        self.coupling_data = IPCCouplingData(self.MAX_LINKS)

        # ============ External Articulation Coupling Data ============
        # Articulated entities participating in joint-level coupling
        # Structure: {entity_idx: articulation_data_dict}
        self._articulated_entities = {}
        # Each articulation_data_dict contains:
        # {
        #     'entity': RigidEntity,
        #     'env_idx': int,
        #     'revolute_joints': List[RigidJoint],
        #     'prismatic_joints': List[RigidJoint],
        #     'joint_geo_slots': List[GeometrySlot],
        #     'articulation_geo': Geometry,
        #     'articulation_object': Object,
        #     'ref_dof_prev': np.ndarray,  # (n_dofs,)
        #     'delta_theta_tilde': np.ndarray,  # (n_joints,)
        #     'delta_theta': np.ndarray,  # (n_joints,)
        #     'joint_dof_indices': List[int],  # Local DOF indices for each joint
        #     'mass_matrix': np.ndarray,  # (n_joints, n_joints)
        # }

        # ExternalArticulationConstraint instance (created in _init_ipc if needed)
        self._ipc_eac = None

        # Mapping from link_idx to ABD geometry for articulation constraint
        # Structure: {(env_idx, link_idx): abd_geometry}
        self._link_to_abd_geo = {}
        # Mapping from link_idx to ABD geometry slot for articulation constraint
        # Structure: {(env_idx, link_idx): abd_geometry_slot}
        self._link_to_abd_slot = {}

        # Link collision settings for IPC
        # Structure: {entity_idx: {link_idx: bool}} - True to enable collision, False to disable
        self._link_collision_settings = {}

        # Initialize Taichi data structure for articulation coupling
        self.articulation_data = ArticulationData(
            self.MAX_ARTICULATED_ENTITIES,
            self.MAX_DOFS_PER_ENTITY,
            self.MAX_JOINTS_PER_ENTITY,
            self.MAX_ENVS,
        )

    def build(self) -> None:
        """Build IPC system"""
        # Initialize IPC system
        self._init_ipc()
        self._add_objects_to_ipc()
        self._finalize_ipc()
        if self.options.enable_ipc_gui:
            self._init_ipc_gui()

    def _init_ipc(self):
        """Initialize IPC system components"""
        from uipc.core import Engine, World, Scene
        from uipc.constitution import (
            AffineBodyConstitution,
            StableNeoHookean,
            NeoHookeanShell,
            StrainLimitingBaraffWitkinShell,
            DiscreteShellBending,
        )

        # Disable IPC logging if requested
        if self.options.disable_ipc_logging:
            from uipc import Logger, Timer

            Logger.set_level(Logger.Level.Error)
            Timer.disable_all()

        # Create IPC engine and world
        import os
        import tempfile

        # Create workspace directory for IPC output
        workspace = os.path.join(tempfile.gettempdir(), "genesis_ipc_workspace")
        os.makedirs(workspace, exist_ok=True)

        # Note: gpu_device option may need to be set via CUDA environment variables (CUDA_VISIBLE_DEVICES)
        # before Genesis initialization, as libuipc Engine does not expose device selection in constructor
        self._ipc_engine = Engine("cuda", workspace)
        self._ipc_world = World(self._ipc_engine)

        # Create IPC scene with configuration
        config = build_ipc_scene_config(self.options)
        self._ipc_scene = Scene(config)

        # Create constitutions
        self._ipc_abd = AffineBodyConstitution()
        self._ipc_stk = StableNeoHookean()
        self._ipc_nks = StrainLimitingBaraffWitkinShell()  # For cloth
        self._ipc_dsb = DiscreteShellBending()  # For cloth bending

        # Add constitutions to scene
        self._ipc_scene.constitution_tabular().insert(self._ipc_abd)
        self._ipc_scene.constitution_tabular().insert(self._ipc_stk)
        # Note: Shell constitutions are added on-demand when cloth entities exist

        # Set up contact model (physical parameters)
        self._ipc_scene.contact_tabular().default_model(
            self.options.contact_friction_mu, self.options.contact_resistance
        )

        # Create separate contact elements for ABD, FEM, Cloth, and Ground to control their interactions
        self._ipc_abd_contact = self._ipc_scene.contact_tabular().create("abd_contact")
        self._ipc_fem_contact = self._ipc_scene.contact_tabular().create("fem_contact")
        self._ipc_cloth_contact = self._ipc_scene.contact_tabular().create("cloth_contact")
        self._ipc_ground_contact = self._ipc_scene.contact_tabular().create("ground_contact")
        # Create no_collision contact element for links with collision disabled
        self._ipc_no_collision_contact = self._ipc_scene.contact_tabular().create("no_collision_contact")

        # Configure contact interactions based on IPC coupler options
        # FEM-FEM: always enabled
        self._ipc_scene.contact_tabular().insert(
            self._ipc_fem_contact,
            self._ipc_fem_contact,
            self.options.fem_fem_friction_mu,
            self.options.contact_resistance,
            True,
        )
        # FEM-ABD: always enabled
        self._ipc_scene.contact_tabular().insert(
            self._ipc_fem_contact,
            self._ipc_abd_contact,
            self.options.contact_friction_mu,
            self.options.contact_resistance,
            True,
        )
        # ABD-ABD: controlled by IPC_self_contact option
        self._ipc_scene.contact_tabular().insert(
            self._ipc_abd_contact,
            self._ipc_abd_contact,
            self.options.contact_friction_mu,
            self.options.contact_resistance,
            self.options.IPC_self_contact,
        )
        # Cloth-Cloth: always enabled for cloth self-collision (necessary to prevent self-penetration)
        self._ipc_scene.contact_tabular().insert(
            self._ipc_cloth_contact,
            self._ipc_cloth_contact,
            self.options.fem_fem_friction_mu,
            self.options.contact_resistance,
            True,
        )  # Always enable cloth self-collision
        # Cloth-FEM: always enabled
        self._ipc_scene.contact_tabular().insert(
            self._ipc_cloth_contact,
            self._ipc_fem_contact,
            self.options.fem_fem_friction_mu,
            self.options.contact_resistance,
            True,
        )
        # Cloth-ABD: always enabled
        self._ipc_scene.contact_tabular().insert(
            self._ipc_cloth_contact,
            self._ipc_abd_contact,
            self.options.contact_friction_mu,
            self.options.contact_resistance,
            True,
        )

        # Ground contact interactions
        # Ground-ABD (rigid bodies): controlled by disable_ipc_ground_contact option
        self._ipc_scene.contact_tabular().insert(
            self._ipc_ground_contact,
            self._ipc_abd_contact,
            self.options.contact_friction_mu,
            self.options.contact_resistance,
            not self.options.disable_ipc_ground_contact,
        )
        # Ground-FEM: always enabled
        self._ipc_scene.contact_tabular().insert(
            self._ipc_ground_contact,
            self._ipc_fem_contact,
            self.options.contact_friction_mu,
            self.options.contact_resistance,
            True,
        )
        # Ground-Cloth: always enabled
        self._ipc_scene.contact_tabular().insert(
            self._ipc_ground_contact,
            self._ipc_cloth_contact,
            self.options.contact_friction_mu,
            self.options.contact_resistance,
            True,
        )

        # Configure no_collision contact element: disable all interactions
        # No collision with ABD (disabled)
        self._ipc_scene.contact_tabular().insert(
            self._ipc_no_collision_contact,
            self._ipc_abd_contact,
            self.options.contact_friction_mu,
            self.options.contact_resistance,
            False,  # Disabled
        )
        # No collision with FEM (disabled)
        self._ipc_scene.contact_tabular().insert(
            self._ipc_no_collision_contact,
            self._ipc_fem_contact,
            self.options.contact_friction_mu,
            self.options.contact_resistance,
            False,  # Disabled
        )
        # No collision with Cloth (disabled)
        self._ipc_scene.contact_tabular().insert(
            self._ipc_no_collision_contact,
            self._ipc_cloth_contact,
            self.options.contact_friction_mu,
            self.options.contact_resistance,
            False,  # Disabled
        )
        # No collision with Ground (disabled)
        self._ipc_scene.contact_tabular().insert(
            self._ipc_no_collision_contact,
            self._ipc_ground_contact,
            self.options.contact_friction_mu,
            self.options.contact_resistance,
            False,  # Disabled
        )
        # No self-collision (disabled)
        self._ipc_scene.contact_tabular().insert(
            self._ipc_no_collision_contact,
            self._ipc_no_collision_contact,
            self.options.contact_friction_mu,
            self.options.contact_resistance,
            False,  # Disabled
        )

        # Set up subscenes for multi-environment (scene grouping)
        # Only use subscenes when B > 1 to avoid issues with ground collision
        # (ground's subscene support is incomplete in libuipc)
        B = self.sim._B
        self._ipc_scene_subscenes = {}
        self._use_subscenes = B > 1

        if self._use_subscenes:
            for i in range(B):
                self._ipc_scene_subscenes[i] = self._ipc_scene.subscene_tabular().create(f"subscene{i}")

            # Disable contact between different environments
            for i in range(B):
                for j in range(B):
                    if i != j:
                        self._ipc_scene.subscene_tabular().insert(
                            self._ipc_scene_subscenes[i], self._ipc_scene_subscenes[j], False
                        )

        self.abd_data_by_link = {}

    # _extract_articulated_joints is now extract_articulated_joints() in utils.py

    def _add_objects_to_ipc(self):
        """Add objects from solvers to IPC system"""
        # Add FEM entities to IPC
        if self.fem_solver.is_active:
            self._add_fem_entities_to_ipc()

        # Add rigid geoms to IPC based on per-entity coupling types
        if self.rigid_solver.is_active:
            self._add_rigid_geoms_to_ipc()

        # Add articulated entities for entities with external_articulation coupling type
        has_articulation_entities = any(ct == "external_articulation" for ct in self._entity_coupling_types.values())
        if has_articulation_entities and self.rigid_solver.is_active:
            self._add_articulated_entities_to_ipc()

    def _add_fem_entities_to_ipc(self):
        """Add FEM entities to the existing IPC scene (includes both volumetric FEM and cloth)"""
        from uipc.constitution import ElasticModuli
        from uipc.geometry import label_surface, tetmesh, trimesh
        from genesis.engine.materials.FEM.cloth import Cloth as ClothMaterial

        fem_solver = self.fem_solver
        scene = self._ipc_scene
        stk = self._ipc_stk  # StableNeoHookean for volumetric FEM
        nks = self._ipc_nks  # NeoHookeanShell for cloth
        dsb = self._ipc_dsb  # DiscreteShellBending for cloth
        scene_subscenes = self._ipc_scene_subscenes

        fem_solver._mesh_handles = {}
        fem_solver.list_env_obj = []
        fem_solver.list_env_mesh = []

        for i_b in range(self.sim._B):
            fem_solver.list_env_obj.append([])
            fem_solver.list_env_mesh.append([])
            for i_e, entity in enumerate(fem_solver._entities):
                is_cloth = isinstance(entity.material, ClothMaterial)

                # Create object in IPC
                obj_name = f"cloth_{i_b}_{i_e}" if is_cloth else f"fem_{i_b}_{i_e}"
                fem_solver.list_env_obj[i_b].append(scene.objects().create(obj_name))

                # Create mesh: trimesh for cloth (2D shell), tetmesh for volumetric FEM (3D)
                if is_cloth:
                    # Cloth: use surface triangles only
                    verts = entity.init_positions.cpu().numpy().astype(np.float64)
                    faces = entity.surface_triangles.astype(np.int32)
                    mesh = trimesh(verts, faces)
                else:
                    # Volumetric FEM: use tetrahedral mesh
                    mesh = tetmesh(entity.init_positions.cpu().numpy(), entity.elems)

                fem_solver.list_env_mesh[i_b].append(mesh)

                # Add to contact subscene (only for multi-environment)
                if self._use_subscenes:
                    scene_subscenes[i_b].apply_to(mesh)

                # Apply contact element based on type
                if is_cloth:
                    self._ipc_cloth_contact.apply_to(mesh)
                else:
                    self._ipc_fem_contact.apply_to(mesh)

                label_surface(mesh)

                # Apply material constitution based on type
                if is_cloth:
                    moduli = ElasticModuli2D.youngs_poisson(entity.material.E, entity.material.nu)
                    # Apply shell material for cloth
                    nks.apply_to(
                        mesh, moduli=moduli, mass_density=entity.material.rho, thickness=entity.material.thickness
                    )
                    # Apply bending stiffness if specified
                    if entity.material.bending_stiffness is not None:
                        dsb.apply_to(mesh, bending_stiffness=entity.material.bending_stiffness)
                else:
                    # Apply volumetric material for FEM
                    moduli = ElasticModuli.youngs_poisson(entity.material.E, entity.material.nu)
                    stk.apply_to(mesh, moduli, mass_density=entity.material.rho)

                # Add metadata to identify geometry type
                meta_attrs = mesh.meta()
                meta_attrs.create("solver_type", "cloth" if is_cloth else "fem")
                meta_attrs.create("env_idx", str(i_b))
                meta_attrs.create("entity_idx", str(i_e))

                # Create geometry in IPC scene
                fem_solver.list_env_obj[i_b][i_e].geometries().create(mesh)
                fem_solver._mesh_handles[f"gs_ipc_{i_b}_{i_e}"] = mesh

                # Update global vertex offset (FEM vertices occupy index space but aren't in mapping)
                self._global_vertex_offset += mesh.vertices().size()

    def _add_rigid_geoms_to_ipc(self):
        """Add rigid geoms to the existing IPC scene as ABD objects, merging geoms by link_idx"""
        from uipc.geometry import tetmesh, label_surface, label_triangle_orient, flip_inward_triangles, merge, ground
        from uipc.constitution import AffineBodyExternalBodyForce
        from genesis.utils import mesh as mu
        import trimesh

        rigid_solver = self.rigid_solver
        scene = self._ipc_scene
        abd = self._ipc_abd
        scene_subscenes = self._ipc_scene_subscenes

        # Create and register AffineBodyExternalBodyForce constitution
        if not hasattr(self, "_ipc_ext_force"):
            self._ipc_ext_force = AffineBodyExternalBodyForce()
            scene.constitution_tabular().insert(self._ipc_ext_force)

        # Initialize lists following FEM solver pattern
        rigid_solver.list_env_obj = []
        rigid_solver.list_env_mesh = []
        rigid_solver._mesh_handles = {}
        rigid_solver._abd_transforms = {}

        # Debug: print all registered entity coupling types
        gs.logger.info(f"Registered entity coupling types: {self._entity_coupling_types}")

        for i_b in range(self.sim._B):
            rigid_solver.list_env_obj.append([])
            rigid_solver.list_env_mesh.append([])

            # ========== First, handle planes (independent of coupling type) ==========
            # Planes are static collision geometry, they don't need any coupling
            plane_geoms = []  # list of (geom_idx, plane_geom)
            for i_g in range(rigid_solver.n_geoms_):
                geom_type = rigid_solver.geoms_info.type[i_g]
                if geom_type == gs.GEOM_TYPE.PLANE:
                    # Handle planes as static IPC geometry (no coupling needed)
                    pos = rigid_solver.geoms_info.pos[i_g].to_numpy()
                    normal = np.array([0.0, 0.0, 1.0])  # Z-up
                    height = np.dot(pos, normal)
                    plane_geom = ground(height, normal)
                    plane_geoms.append((i_g, plane_geom))

            # Create plane objects in IPC scene
            for geom_idx, plane_geom in plane_geoms:
                plane_obj = scene.objects().create(f"rigid_plane_{i_b}_{geom_idx}")
                rigid_solver.list_env_obj[i_b].append(plane_obj)
                rigid_solver.list_env_mesh[i_b].append(None)  # Planes are ImplicitGeometry

                # Apply ground contact element to plane
                self._ipc_ground_contact.apply_to(plane_geom)

                plane_obj.geometries().create(plane_geom)
                rigid_solver._mesh_handles[f"rigid_plane_{i_b}_{geom_idx}"] = plane_geom

            # ========== Then, handle non-plane geoms (requires coupling type) ==========
            # Group geoms by link_idx for merging
            # IMPORTANT: We merge geoms by target_link_idx (after fixed joint merging)
            # This matches the behavior of mjcf.py where geoms from fixed-joint children
            # are merged into the parent body's mesh
            link_geoms = (
                {}
            )  # target_link_idx -> dict with 'meshes', 'link_world_pos', 'link_world_quat', 'entity_idx', 'original_to_target'

            # First pass: collect and group geoms by target_link_idx (merging fixed joints)
            for i_g in range(rigid_solver.n_geoms_):
                geom_type = rigid_solver.geoms_info.type[i_g]

                # Skip planes (already handled above)
                if geom_type == gs.GEOM_TYPE.PLANE:
                    continue

                link_idx = rigid_solver.geoms_info.link_idx[i_g]
                entity_idx = rigid_solver.links_info.entity_idx[link_idx]
                entity = rigid_solver._entities[entity_idx]

                # Check if this entity has a coupling type (None means skip entirely)
                entity_coupling_type = self._entity_coupling_types.get(entity_idx)
                gs.logger.debug(
                    f"Geom {i_g}: link_idx={link_idx}, entity_idx={entity_idx}, coupling_type={entity_coupling_type}"
                )
                if entity_coupling_type is None:
                    continue  # Entity not in IPC

                # For two_way_soft_constraint, check link filter
                if entity_coupling_type == "two_way_soft_constraint":
                    if entity_idx in self._ipc_link_filters:
                        link_filter = self._ipc_link_filters[entity_idx]
                        if link_filter is not None and link_idx not in link_filter:
                            continue  # Skip this geom/link

                # Find target link for fixed joint merging
                # This walks up the tree, skipping FIXED joints
                target_link_idx = find_target_link_for_fixed_merge(self.rigid_solver, link_idx)

                # Initialize target link group if not exists
                if target_link_idx not in link_geoms:
                    link_geoms[target_link_idx] = {
                        "meshes": [],
                        "link_world_pos": None,
                        "link_world_quat": None,
                        "entity_idx": entity_idx,
                        "original_to_target": {},  # Maps original link_idx to (R, t) transform
                    }

                # Compute transform from original link to target link (for fixed joint merging)
                if link_idx != target_link_idx:
                    if link_idx not in link_geoms[target_link_idx]["original_to_target"]:
                        # Compute relative transform
                        R_link_to_target, t_link_to_target = compute_link_to_link_transform(
                            self.rigid_solver, link_idx, target_link_idx
                        )
                        link_geoms[target_link_idx]["original_to_target"][link_idx] = (
                            R_link_to_target,
                            t_link_to_target,
                        )
                        gs.logger.info(
                            f"Merging link {link_idx} ({rigid_solver.links[link_idx].name}) "
                            f"into target link {target_link_idx} ({rigid_solver.links[target_link_idx].name}) "
                            f"via fixed joint"
                        )

                try:
                    # For all non-plane geoms, create trimesh
                    vert_num = rigid_solver.geoms_info.vert_num[i_g]
                    if vert_num == 0:
                        continue  # Skip geoms without vertices

                    # Extract vertex and face data
                    vert_start = rigid_solver.geoms_info.vert_start[i_g]
                    vert_end = rigid_solver.geoms_info.vert_end[i_g]
                    face_start = rigid_solver.geoms_info.face_start[i_g]
                    face_end = rigid_solver.geoms_info.face_end[i_g]

                    # Get vertices and faces
                    geom_verts = rigid_solver.verts_info.init_pos.to_numpy()[vert_start:vert_end]
                    geom_faces = rigid_solver.faces_info.verts_idx.to_numpy()[face_start:face_end]
                    geom_faces = geom_faces - vert_start  # Adjust indices

                    # Apply geom-relative transform to vertices (needed for merging)
                    geom_rel_pos = rigid_solver.geoms_info.pos[i_g].to_numpy()
                    geom_rel_quat = rigid_solver.geoms_info.quat[i_g].to_numpy()

                    # Transform vertices by geom relative transform
                    import genesis.utils.geom as gu

                    geom_rot_mat = gu.quat_to_R(geom_rel_quat)
                    transformed_verts = geom_verts @ geom_rot_mat.T + geom_rel_pos

                    # If this geom belongs to a link that was merged via fixed joint,
                    # apply additional transform to target link frame
                    if link_idx != target_link_idx:
                        R_link_to_target, t_link_to_target = link_geoms[target_link_idx]["original_to_target"][link_idx]
                        # Transform vertices: v' = R @ v + t
                        transformed_verts = transformed_verts @ R_link_to_target.T + t_link_to_target

                    # Create uipc trimesh for rigid body (ABD doesn't need tetmesh)
                    try:
                        from uipc.geometry import trimesh as uipc_trimesh

                        # Create uipc trimesh directly (dim=2, surface mesh for ABD)
                        rigid_mesh = uipc_trimesh(transformed_verts.astype(np.float64), geom_faces.astype(np.int32))

                        # Store uipc mesh (SimplicialComplex) for merging
                        link_geoms[target_link_idx]["meshes"].append((i_g, rigid_mesh))

                    except Exception as e:
                        gs.logger.warning(f"Failed to convert trimesh to tetmesh for geom {i_g}: {e}")
                        continue

                    # Store target link transform info (same for all geoms merged into this target)
                    if link_geoms[target_link_idx]["link_world_pos"] is None:
                        link_geoms[target_link_idx]["link_world_pos"] = rigid_solver.links_state.pos[
                            target_link_idx, i_b
                        ]
                        link_geoms[target_link_idx]["link_world_quat"] = rigid_solver.links_state.quat[
                            target_link_idx, i_b
                        ]

                except Exception as e:
                    gs.logger.warning(f"Failed to process geom {i_g}: {e}")
                    continue

            # Second pass: merge geoms per link and create IPC objects
            link_obj_counter = 0
            for link_idx, link_data in link_geoms.items():
                try:
                    # Handle regular meshes (merge if multiple)
                    if link_data["meshes"]:
                        if len(link_data["meshes"]) == 1:
                            # Single mesh in link
                            geom_idx, merged_mesh = link_data["meshes"][0]
                        else:
                            # Multiple meshes in link - merge them using uipc.geometry.merge
                            meshes_to_merge = [mesh for geom_idx, mesh in link_data["meshes"]]
                            merged_mesh = merge(meshes_to_merge)
                            geom_idx = link_data["meshes"][0][0]  # Use first geom's index for metadata

                        # Apply link world transform
                        from uipc import view, Transform, Vector3, Quaternion

                        trans_view = view(merged_mesh.transforms())
                        t = Transform.Identity()

                        link_world_pos = link_data["link_world_pos"]
                        link_world_quat = link_data["link_world_quat"]

                        # Ensure numpy format
                        link_world_pos = link_world_pos.to_numpy()
                        link_world_quat = link_world_quat.to_numpy()

                        t.translate(Vector3.Values((link_world_pos[0], link_world_pos[1], link_world_pos[2])))
                        uipc_link_quat = Quaternion(link_world_quat)
                        t.rotate(uipc_link_quat)
                        trans_view[0] = t.matrix()

                        # Process surface for contact
                        label_surface(merged_mesh)

                        # Create rigid object
                        rigid_obj = scene.objects().create(f"rigid_link_{i_b}_{link_idx}")
                        rigid_solver.list_env_obj[i_b].append(rigid_obj)
                        rigid_solver.list_env_mesh[i_b].append(merged_mesh)

                        # Add to contact subscene and apply contact element based on collision settings
                        if self._use_subscenes:
                            scene_subscenes[i_b].apply_to(merged_mesh)

                        # Check if collision is disabled for this link
                        collision_enabled = True
                        entity_idx = link_data["entity_idx"]
                        if entity_idx in self._link_collision_settings:
                            if link_idx in self._link_collision_settings[entity_idx]:
                                collision_enabled = self._link_collision_settings[entity_idx][link_idx]

                        # Apply appropriate contact element
                        if collision_enabled:
                            self._ipc_abd_contact.apply_to(merged_mesh)
                        else:
                            self._ipc_no_collision_contact.apply_to(merged_mesh)

                        from uipc.unit import MPa

                        # Get entity coupling type for this link's entity
                        entity_coupling_type = self._entity_coupling_types.get(entity_idx)

                        # Check if this entity is IPC-only
                        is_ipc_only = entity_coupling_type == "ipc_only"

                        entity_rho = rigid_solver._entities[link_data["entity_idx"]].material.rho

                        # Always use full density (no mass splitting)
                        abd.apply_to(
                            merged_mesh,
                            kappa=100.0 * MPa,
                            mass_density=entity_rho,
                        )

                        # Set external_kinetic=1 for all ABD objects
                        from uipc import builtin, view

                        external_kinetic_attr = merged_mesh.instances().find(builtin.external_kinetic)
                        if external_kinetic_attr is not None:
                            external_kinetic_view = view(external_kinetic_attr)
                            external_kinetic_view[:] = 1

                        # Set is_fixed attribute for base link (when link.is_fixed=True)
                        # This fixes the base link in IPC, matching test_external_articulation_constraint.py
                        link = rigid_solver.links[link_idx]
                        is_link_fixed = link.is_fixed

                        is_fixed_attr = merged_mesh.instances().find(builtin.is_fixed)
                        if is_fixed_attr is not None:
                            is_fixed_view = view(is_fixed_attr)
                            # Fix link if it's fixed in Genesis
                            is_fixed_view[0] = 1 if is_link_fixed else 0

                        # For external_articulation mode, create ref_dof_prev attribute
                        if entity_coupling_type == "external_articulation" and self.options.sync_dof_enable:
                            from uipc.geometry import affine_body
                            from uipc import Vector12

                            # Create ref_dof_prev attribute on instances
                            ref_dof_prev_attr = merged_mesh.instances().create("ref_dof_prev", Vector12.Zero())
                            ref_dof_prev_view = view(ref_dof_prev_attr)
                            # Initialize with current transform (convert transform matrix to q)
                            initial_transform = trans_view[0]
                            ref_dof_prev_view[0] = affine_body.transform_to_q(initial_transform)

                        # Apply soft transform constraints for non-IPC-only links
                        if not is_ipc_only:
                            from uipc.constitution import SoftTransformConstraint

                            if not hasattr(self, "_ipc_stc"):
                                self._ipc_stc = SoftTransformConstraint()
                                scene.constitution_tabular().insert(self._ipc_stc)

                            strength_tuple = self.options.ipc_constraint_strength
                            constraint_strength = np.array(
                                [
                                    strength_tuple[0],  # translation strength
                                    strength_tuple[1],  # rotation strength
                                ]
                            )
                            self._ipc_stc.apply_to(merged_mesh, constraint_strength)

                        # Apply external force (initially zero, can be modified by animator)
                        initial_force = np.zeros(12, dtype=np.float64)  # Vector12: [fx, fy, fz, dS/dt (9 components)]
                        self._ipc_ext_force.apply_to(merged_mesh, initial_force)

                        # Add metadata
                        meta_attrs = merged_mesh.meta()
                        meta_attrs.create("solver_type", "rigid")
                        meta_attrs.create("env_idx", str(i_b))
                        meta_attrs.create("link_idx", str(link_idx))  # Use link_idx instead of geom_idx

                        # Build vertex-to-link mapping for contact force computation
                        # Only for two_way_soft_constraint (needs contact force feedback)
                        if entity_coupling_type == "two_way_soft_constraint":
                            n_verts = merged_mesh.vertices().size()
                            for local_idx in range(n_verts):
                                global_idx = self._global_vertex_offset + local_idx
                                self._vertex_to_link_mapping[global_idx] = (link_idx, i_b, local_idx)

                        # Update global vertex offset
                        self._global_vertex_offset += merged_mesh.vertices().size()

                        # Create geometry and get slot
                        abd_slot, _ = rigid_obj.geometries().create(merged_mesh)

                        # Set up animator for this link (only for two_way_soft_constraint)
                        # For external_articulation, the ExternalArticulationConstraint handles the coupling
                        # For ipc_only, no animator needed - transforms are directly set from IPC
                        if entity_coupling_type == "two_way_soft_constraint":
                            if not hasattr(self, "_ipc_animator"):
                                self._ipc_animator = scene.animator()

                            def create_animate_function(env_idx, link_idx, coupler_ref):
                                def animate_rigid_link(info):
                                    from uipc import view, builtin

                                    geo_slots = info.geo_slots()
                                    if len(geo_slots) == 0:
                                        return
                                    geo = geo_slots[0].geometry()

                                    try:
                                        # Read stored Genesis transform (q_genesis^n)
                                        # This was stored in _store_genesis_rigid_states() before advance()
                                        if hasattr(coupler_ref, "_genesis_stored_states"):
                                            stored_states = coupler_ref._genesis_stored_states
                                            if link_idx in stored_states and env_idx in stored_states[link_idx]:
                                                transform_matrix = stored_states[link_idx][env_idx]

                                                # Enable constraint and set target transform
                                                is_constrained = geo.instances().find(builtin.is_constrained)
                                                aim_transform_attr = geo.instances().find(builtin.aim_transform)

                                                if is_constrained and aim_transform_attr:
                                                    view(is_constrained)[0] = 1
                                                    view(aim_transform_attr)[:] = transform_matrix

                                        # Update external force if user has set it
                                        if hasattr(coupler_ref, "_external_force_data"):
                                            force_data = coupler_ref._external_force_data
                                            force_attr = geo.instances().find("external_force")
                                            is_constrained_attr = geo.instances().find("is_constrained")
                                            key = (link_idx, env_idx)
                                            if key in force_data:
                                                if force_attr is not None:
                                                    force_vector = force_data[key]
                                                    view(force_attr)[:] = force_vector.reshape(-1, 1)

                                                if is_constrained_attr is not None:
                                                    view(is_constrained_attr)[:] = 1
                                            else:
                                                if force_attr is not None:
                                                    view(force_attr)[:] = np.zeros((12, 1), dtype=np.float64)

                                    except Exception as e:
                                        gs.logger.warning(f"Error setting IPC animation target: {e}")

                                return animate_rigid_link

                            animate_func = create_animate_function(i_b, link_idx, self)
                            self._ipc_animator.insert(rigid_obj, animate_func)

                        rigid_solver._mesh_handles[f"rigid_link_{i_b}_{link_idx}"] = merged_mesh

                        # Store ABD geometry and slot mapping for articulation constraint
                        # Store for target link
                        self._link_to_abd_geo[(i_b, link_idx)] = merged_mesh
                        self._link_to_abd_slot[(i_b, link_idx)] = abd_slot

                        # Also store mappings for all original links that were merged into this target
                        # This allows joints connecting to child links (via fixed joints) to find the merged ABD
                        if "original_to_target" in link_data:
                            for original_link_idx in link_data["original_to_target"].keys():
                                self._link_to_abd_geo[(i_b, original_link_idx)] = merged_mesh
                                self._link_to_abd_slot[(i_b, original_link_idx)] = abd_slot
                                gs.logger.info(
                                    f"Created ABD slot mapping: link {original_link_idx} -> target link {link_idx} (merged via fixed joint)"
                                )

                        link_obj_counter += 1

                except Exception as e:
                    gs.logger.warning(f"Failed to create IPC object for link {link_idx}: {e}")
                    continue

        # NOTE: Mass scaling removed - now using external_kinetic=1 instead
        # All mass is handled by IPC, Genesis uses external_kinetic for kinematic coupling

    def _finalize_ipc(self):
        """Finalize IPC setup"""
        self._ipc_world.init(self._ipc_scene)
        gs.logger.info("IPC world initialized successfully")

    # ============================================================
    # Section 3: Configuration API
    # ============================================================

    @property
    def is_active(self) -> bool:
        """Check if IPC coupling is active"""
        return self._ipc_world is not None

    def set_entity_coupling_type(self, entity, coupling_type: str):
        """
        Set IPC coupling type for an entire entity.

        Parameters
        ----------
        entity : RigidEntity
            The rigid entity to configure
        coupling_type : str or None
            Type of coupling:
            - None: Entity not processed by IPC (default)
            - 'two_way_soft_constraint': Two-way coupling using SoftTransformConstraint
            - 'external_articulation': Joint-level coupling using ExternalArticulationConstraint
            - 'ipc_only': IPC controls entity, transforms copied to Genesis (one-way)
                         Only allowed for single base-link entities (free rigid bodies)

        Notes
        -----
        - None: Entity is completely ignored by IPC coupler
        - 'two_way_soft_constraint': Uses SoftTransformConstraint for bidirectional coupling,
          can use set_ipc_coupling_link_filter to select specific links
        - 'external_articulation': Uses ExternalArticulationConstraint for articulated bodies,
          joint positions are coupled at the DOF level
        - 'ipc_only': Entity only simulated in IPC, transforms directly set to Genesis.
          Only allowed for entities with a single base link (no joints).

        This must be called before scene.build().
        """
        # Use solver-level index for consistency with rigid_solver.links_info.entity_idx
        entity_idx = entity._idx_in_solver

        # Validate coupling type
        valid_types = [None, "two_way_soft_constraint", "external_articulation", "ipc_only"]
        if coupling_type not in valid_types:
            raise ValueError(f"Invalid coupling_type '{coupling_type}'. Must be one of {valid_types}.")

        # For ipc_only, validate that entity has only base link (single rigid body)
        if coupling_type == "ipc_only":
            if entity.n_links != 1:
                raise ValueError(
                    f"'ipc_only' coupling type only allowed for single base-link entities. "
                    f"Entity {entity_idx} has {entity.n_links} links."
                )

        # Store coupling type
        if coupling_type is None:
            # Remove from dict if present
            if entity_idx in self._entity_coupling_types:
                del self._entity_coupling_types[entity_idx]
            # Also remove from link filters
            if entity_idx in self._ipc_link_filters:
                del self._ipc_link_filters[entity_idx]
        else:
            self._entity_coupling_types[entity_idx] = coupling_type

        gs.logger.info(f"Rigid entity (solver idx={entity_idx}): coupling type set to '{coupling_type}'")

    def set_ipc_coupling_link_filter(self, entity, link_names=None, link_indices=None):
        """
        Set which links of an entity participate in IPC coupling.

        This is only applicable for entities with 'two_way_soft_constraint' coupling type.
        By default, all links participate. Use this to filter to specific links.

        Parameters
        ----------
        entity : RigidEntity
            The rigid entity to configure
        link_names : list of str, optional
            Names of links to include in IPC coupling.
        link_indices : list of int, optional
            Local indices of links to include in IPC coupling.

        Notes
        -----
        - If both link_names and link_indices are None, all links participate (removes filter)
        - Links not in the filter will not be simulated in IPC
        - This must be called before scene.build()
        - Only valid for 'two_way_soft_constraint' entities
        """
        # Use solver-level index for consistency with rigid_solver.links_info.entity_idx
        entity_idx = entity._idx_in_solver

        # Check that entity has appropriate coupling type
        coupling_type = self._entity_coupling_types.get(entity_idx)
        if coupling_type != "two_way_soft_constraint":
            gs.logger.warning(
                f"set_ipc_coupling_link_filter only applies to 'two_way_soft_constraint' entities. "
                f"Entity {entity_idx} has coupling type '{coupling_type}'. Ignoring."
            )
            return

        # Determine which links to include
        if link_names is None and link_indices is None:
            # Remove filter - all links participate
            if entity_idx in self._ipc_link_filters:
                del self._ipc_link_filters[entity_idx]
            gs.logger.info(f"Entity {entity_idx}: IPC link filter removed (all links participate)")
            return

        # Build set of links to include
        target_links = set()

        if link_names is not None:
            for name in link_names:
                try:
                    link = entity.get_link(name=name)
                    target_links.add(link.idx)
                except Exception as e:
                    gs.logger.warning(f"Link name '{name}' not found in entity")

        if link_indices is not None:
            for local_idx in link_indices:
                solver_link_idx = local_idx + entity._link_start
                target_links.add(solver_link_idx)

        # Store filter
        self._ipc_link_filters[entity_idx] = target_links
        gs.logger.info(f"Entity {entity_idx}: IPC link filter set to {len(target_links)} link(s)")

    def has_coupling_type(self, coupling_type: str) -> bool:
        """
        Check if any entity has the specified coupling type.

        Parameters
        ----------
        coupling_type : str
            The coupling type to check for. Valid values: "two_way_soft_constraint",
            "external_articulation", "ipc_only"

        Returns
        -------
        bool
            True if at least one entity has the specified coupling type.
        """
        return coupling_type in self._entity_coupling_types.values()

    def has_any_rigid_coupling(self) -> bool:
        """
        Check if any rigid entity is coupled to IPC.

        Returns
        -------
        bool
            True if at least one rigid entity has a coupling type (two_way_soft_constraint,
            external_articulation, or ipc_only).
        """
        return len(self._entity_coupling_types) > 0

    def set_link_ipc_collision(self, entity, enabled: bool, link_names=None, link_indices=None):
        """
        Enable or disable IPC collision for specific links of an entity.

        This method allows fine-grained control over which links participate in IPC collision detection.
        Links with collision disabled will still be simulated in IPC (if entity has a coupling type set),
        but will not generate contact forces with other objects.

        Parameters
        ----------
        entity : RigidEntity
            The rigid entity to configure
        enabled : bool
            Whether to enable (True) or disable (False) collision for the specified links
        link_names : list of str, optional
            Names of links to configure. If None and link_indices is None, applies to all links.
        link_indices : list of int, optional
            Local indices of links to configure. If None and link_names is None, applies to all links.

        Notes
        -----
        - This setting only affects IPC collision detection, not Genesis collision
        - Entity must have a coupling type set (via set_entity_coupling_type) for this to have effect
        - Disabled collision links will have their own contact element with no interactions enabled
        - This must be called before scene.build()

        Examples
        --------
        # Disable collision for robot fingers
        coupler.set_link_ipc_collision(robot, enabled=False, link_names=['finger1', 'finger2'])

        # Enable collision for all links (default behavior)
        coupler.set_link_ipc_collision(robot, enabled=True)
        """
        # Use solver-level index for consistency with rigid_solver.links_info.entity_idx
        entity_idx = entity._idx_in_solver

        # Determine which links to configure
        if link_names is None and link_indices is None:
            # Apply to all links
            target_links = set()
            for local_idx in range(entity.n_links):
                solver_link_idx = local_idx + entity._link_start
                target_links.add(solver_link_idx)
        else:
            # Apply to specified links
            target_links = set()

            if link_names is not None:
                for name in link_names:
                    try:
                        link = entity.get_link(name=name)
                        target_links.add(link.idx)
                    except Exception as e:
                        gs.logger.warning(f"Link name '{name}' not found in entity")

            if link_indices is not None:
                for local_idx in link_indices:
                    solver_link_idx = local_idx + entity._link_start
                    target_links.add(solver_link_idx)

        # Store collision settings
        if entity_idx not in self._link_collision_settings:
            self._link_collision_settings[entity_idx] = {}

        for link_idx in target_links:
            self._link_collision_settings[entity_idx][link_idx] = enabled

        status = "enabled" if enabled else "disabled"
        gs.logger.info(f"Entity {entity_idx}: {len(target_links)} link(s) set to collision {status}")

    # ============================================================
    # Section 4: Main Coupling Loop & Shared Helpers
    # ============================================================
    # preprocess, couple (dispatcher), couple_grad, reset
    # _store_genesis_rigid_states, _categorize_entities_by_coupling_type
    # _retrieve_fem_states, _retrieve_rigid_states
    # _get_genesis_link_transform, _apply_abd_coupling_forces

    def preprocess(self, f):
        """Preprocessing step before coupling"""
        pass

    def _store_genesis_rigid_states(self):
        """
        Store current Genesis rigid body states before IPC advance.

        OPTIMIZED VERSION: Uses Taichi kernels for batch reading qpos.

        These stored states will be used by:
        1. Animator: to set aim_transform for IPC soft constraints
        2. Force computation: to ensure action-reaction force consistency
        3. User modification detection: to detect if user called set_qpos
        """
        if not self.rigid_solver.is_active:
            return

        rigid_solver = self.rigid_solver

        # Clear previous stored states
        self._genesis_stored_states.clear()

        # Store qpos for all entities using Taichi fields (used by all coupling strategies)
        for entity_idx, entity in enumerate(rigid_solver._entities):
            if entity.n_qs > 0:  # Skip entities without dofs
                q_start = entity._q_start
                n_qs = entity.n_qs

                # Store metadata
                self.transform_data.stored_qpos_size[entity_idx] = n_qs
                self.transform_data.stored_qpos_start[entity_idx] = q_start

                # Check capacity
                if n_qs > self.MAX_QPOS_SIZE:
                    gs.logger.warning(
                        f"Entity {entity_idx} qpos size {n_qs} exceeds max {self.MAX_QPOS_SIZE}. Skipping qpos storage."
                    )
                    continue

                # Use kernel to store qpos for all environments
                # Choose kernel based on use_ndarray setting
                kernel_func = self._store_qpos_kernel if gs.use_ndarray else self._store_qpos_kernel_field

                for env_idx in range(self.sim._B):
                    kernel_func(self.transform_data, rigid_solver.qpos, entity_idx, env_idx, q_start, n_qs)

        # Store transforms for all rigid links using Taichi fields
        # OPTIMIZED VERSION: Batch get all link states at once per environment
        is_parallelized = self.sim._scene.n_envs > 0

        for env_idx in range(self.sim._B):
            # Batch get all link positions and quaternions for this environment
            if is_parallelized:
                all_links_pos = rigid_solver.get_links_pos(envs_idx=env_idx).detach().cpu().numpy()
                all_links_quat = rigid_solver.get_links_quat(envs_idx=env_idx).detach().cpu().numpy()
            else:
                all_links_pos = rigid_solver.get_links_pos().detach().cpu().numpy()
                all_links_quat = rigid_solver.get_links_quat().detach().cpu().numpy()

            # Some backends still return (n_envs, n_links, dim); slice to 2D for kernels.
            if all_links_pos.ndim == 3:
                all_links_pos = all_links_pos[env_idx, :, :]
            if all_links_quat.ndim == 3:
                all_links_quat = all_links_quat[env_idx, :, :]

            # Get number of links
            n_links = all_links_pos.shape[0]

            # Use kernel to store to Taichi fields (for external_articulation to read pos/quat)
            self._store_link_states_kernel(self.transform_data, all_links_pos, all_links_quat, env_idx, n_links)

            # Also convert to transform matrices and store in _genesis_stored_states (for two_way_soft_constraint)
            # Only store for links that have mesh handles (are in the IPC scene)
            if hasattr(rigid_solver, "_mesh_handles"):
                for handle_key in rigid_solver._mesh_handles.keys():
                    if handle_key.startswith("rigid_link_"):
                        # Parse: "rigid_link_{env_idx}_{link_idx}"
                        parts = handle_key.split("_")
                        if len(parts) >= 4:
                            handle_env_idx = int(parts[2])
                            link_idx = int(parts[3])

                            # Only process if this is the current environment
                            if handle_env_idx == env_idx and link_idx < n_links:
                                link_transform = build_link_transform_matrix(
                                    all_links_pos[link_idx], all_links_quat[link_idx]
                                )

                                # Store transform in _genesis_stored_states
                                if link_idx not in self._genesis_stored_states:
                                    self._genesis_stored_states[link_idx] = {}
                                self._genesis_stored_states[link_idx][env_idx] = link_transform

    def _categorize_entities_by_coupling_type(self):
        """
        Categorize entities by their coupling type.
        Called once during build or first couple() call, cached for subsequent calls.
        """
        if hasattr(self, "_entities_by_coupling_type"):
            return  # Already categorized

        self._entities_by_coupling_type = categorize_entities_by_coupling_type(self._entity_coupling_types)

    def couple(self, f):
        """
        Execute IPC coupling step with per-entity coupling types.

        This unified coupling flow handles all entity types:
        - 'two_way_soft_constraint': Uses Animator + SoftTransformConstraint
        - 'external_articulation': Uses ExternalArticulationConstraint at joint level
        - 'ipc_only': One-way coupling, IPC controls rigid body transforms

        Flow:
        1. Store Genesis rigid states (common)
        2. Pre-advance processing (per entity type)
        3. IPC advance + retrieve (common, only once)
        4. Retrieve FEM states (common)
        5. Post-advance processing (per entity type)
        """
        if not self.is_active:
            return

        # Ensure entities are categorized (cached after first call)
        self._categorize_entities_by_coupling_type()

        two_way_entities = self._entities_by_coupling_type["two_way_soft_constraint"]
        articulation_entities = self._entities_by_coupling_type["external_articulation"]
        ipc_only_entities = self._entities_by_coupling_type["ipc_only"]

        # ========== Step 1: Store Genesis rigid states (common) ==========
        self._store_genesis_rigid_states()

        # ========== Step 2: Pre-advance processing (per entity type) ==========
        # For two_way_soft_constraint: Animator handles aim_transform during advance
        # (no explicit pre-processing needed here)

        # For external_articulation: prepare articulation data before advance
        if articulation_entities:
            self._pre_advance_external_articulation(articulation_entities)

        # For ipc_only: no pre-processing needed

        # ========== Step 3: IPC advance + retrieve (common, only once) ==========
        self._ipc_world.advance()
        self._ipc_world.retrieve()

        # ========== Step 4: Retrieve FEM states (common) ==========
        self._retrieve_fem_states(f)

        # ========== Step 5: Post-advance processing (per entity type) ==========
        # First, retrieve rigid states for all entities that need it (two_way + ipc_only)
        rigid_entities = two_way_entities + ipc_only_entities
        if rigid_entities:
            self._retrieve_rigid_states(f, set(rigid_entities))

        # For two_way_soft_constraint: apply coupling forces
        if two_way_entities:
            if self.options.two_way_coupling:
                self._apply_abd_coupling_forces(set(two_way_entities))
            if self.options.use_contact_proxy:
                self._record_ipc_contact_forces()
                self._apply_ipc_contact_forces()

        # For external_articulation: read delta_theta and update Genesis qpos
        if articulation_entities:
            self._post_advance_external_articulation(articulation_entities)

        # For ipc_only: directly set Genesis transforms from IPC
        if ipc_only_entities:
            self._post_advance_ipc_only(ipc_only_entities)

    # ============================================================
    # Section 6: External Articulation Coupling
    # ============================================================

    def _pre_advance_external_articulation(self, entity_indices):
        """
        Pre-advance processing for external_articulation entities.
        Prepares articulation data and updates IPC geometry before advance().
        """
        from uipc import view
        from uipc.geometry import affine_body

        if len(self._articulated_entities) == 0:
            return

        ad = self.articulation_data

        # Initialize metadata on first call
        if not hasattr(self, "_articulation_metadata_initialized"):
            n_entities = len(self._articulated_entities)
            ad.n_entities[None] = n_entities

            for idx, (entity_idx, art_data) in enumerate(self._articulated_entities.items()):
                entity = art_data["entity"]
                env_idx = art_data["env_idx"]
                n_joints = art_data["n_joints"]
                joint_dof_indices = art_data["joint_dof_indices"]

                # Get actual qpos size from the entity
                if self.sim._B > 1:
                    actual_qpos = entity.get_qpos(envs_idx=env_idx).cpu().numpy()
                else:
                    actual_qpos = entity.get_qpos().cpu().numpy()
                n_dofs_actual = len(actual_qpos)

                # Store the actual size
                art_data["n_dofs_actual"] = n_dofs_actual

                # Fill metadata
                ad.entity_indices[idx] = entity_idx
                ad.entity_env_indices[idx] = env_idx
                ad.entity_n_dofs[idx] = n_dofs_actual
                ad.entity_n_joints[idx] = n_joints
                ad.entity_dof_start[idx] = entity.dof_start

                # Fill joint to DOF mapping
                for j_idx, dof_idx in enumerate(joint_dof_indices):
                    ad.joint_dof_indices[idx, j_idx] = dof_idx

            self._articulation_metadata_initialized = True

        # Get dimensions for batching
        is_parallelized = self.sim._scene.n_envs > 0
        n_envs = self.sim._scene.n_envs if is_parallelized else 1

        # Copy qpos from transform_data.stored_qpos to articulation_data.qpos_current
        self._copy_stored_qpos_to_articulation_kernel(self.transform_data, ad, n_envs)

        # Compute delta_theta_tilde = qpos_current - ref_dof_prev
        self._compute_delta_theta_tilde_kernel(ad, n_envs)

        # Update IPC geometry for each articulated entity
        for idx, (entity_idx, art_data) in enumerate(self._articulated_entities.items()):
            articulation_slot = art_data["articulation_slot"]
            articulation_geo = articulation_slot.geometry()
            env_idx = art_data["env_idx"]
            n_joints = art_data["n_joints"]

            # Update ref_dof_prev on all ABD instances
            for joint_idx, joint in enumerate(art_data["revolute_joints"] + art_data["prismatic_joints"]):
                child_link_idx = joint.link.idx
                abd_geo_slot = self._find_abd_geometry_slot_by_link(child_link_idx, env_idx)
                abd_geo = abd_geo_slot.geometry()

                if abd_geo is not None and self.options.sync_dof_enable:
                    ref_dof_prev_attr = abd_geo.instances().find("ref_dof_prev")
                    if ref_dof_prev_attr is not None:
                        ref_dof_prev_view = view(ref_dof_prev_attr)

                        key = (idx, joint_idx, env_idx)
                        if key in ad.prev_link_transforms:
                            link_transform = ad.prev_link_transforms[key]
                            q = affine_body.transform_to_q(link_transform)
                            ref_dof_prev_view[0] = q
                        else:
                            if (
                                child_link_idx in self._genesis_stored_states
                                and env_idx in self._genesis_stored_states[child_link_idx]
                            ):
                                link_transform = self._genesis_stored_states[child_link_idx][env_idx]
                                q = affine_body.transform_to_q(link_transform)
                                ref_dof_prev_view[0] = q

            # Set delta_theta_tilde to IPC geometry
            delta_theta_tilde_attr = articulation_geo["joint"].find("delta_theta_tilde")
            delta_theta_tilde_view = view(delta_theta_tilde_attr)
            for joint_idx in range(n_joints):
                delta_theta_tilde_view[joint_idx] = ad.delta_theta_tilde[idx, env_idx, joint_idx]

            # Update mass matrix from Genesis
            mass_kernel_func = (
                self._extract_joint_mass_matrix_kernel
                if gs.use_ndarray
                else self._extract_joint_mass_matrix_kernel_field
            )
            mass_kernel_func(ad, self.rigid_solver.mass_mat, idx, env_idx)

            # Transfer mass matrix to IPC
            mass_attr = articulation_geo["joint_joint"].find("mass")
            if mass_attr is not None:
                mass_view = view(mass_attr)
                mass_size = n_joints * n_joints
                for i in range(mass_size):
                    mass_view[i] = ad.mass_matrix[idx, i]

    def _post_advance_external_articulation(self, entity_indices):
        """
        Post-advance processing for external_articulation entities.
        Reads delta_theta from IPC and updates Genesis qpos.
        """
        from uipc import view

        if len(self._articulated_entities) == 0:
            return

        ad = self.articulation_data
        is_parallelized = self.sim._scene.n_envs > 0
        n_envs = self.sim._scene.n_envs if is_parallelized else 1

        # Read delta_theta_ipc from IPC
        for idx, (entity_idx, art_data) in enumerate(self._articulated_entities.items()):
            articulation_slot = art_data["articulation_slot"]
            env_idx = art_data["env_idx"]
            n_joints = art_data["n_joints"]

            scene_art_geo = articulation_slot.geometry()

            delta_theta_attr = scene_art_geo["joint"].find("delta_theta")
            delta_theta_view = view(delta_theta_attr)
            for joint_idx in range(n_joints):
                ad.delta_theta_ipc[idx, env_idx, joint_idx] = delta_theta_view[joint_idx]

        # Compute qpos_new using kernel
        self._compute_qpos_new_kernel(ad, n_envs)

        # Export qpos_new to numpy once (avoid per-element access in loop)
        qpos_new_all = ad.qpos_new.to_numpy()  # (max_entities, max_envs, max_dofs)

        # Write qpos_new back to Genesis using numpy slices
        for idx, (entity_idx, art_data) in enumerate(self._articulated_entities.items()):
            entity = art_data["entity"]
            env_idx = art_data["env_idx"]
            n_dofs = ad.entity_n_dofs[idx]

            # Use slice instead of list comprehension
            qpos_new_np = qpos_new_all[idx, env_idx, :n_dofs].astype(np.float32)
            qpos_tensor = gs.torch.as_tensor(qpos_new_np, dtype=gs.tc_float, device=gs.device)

            if self.sim._B > 1:
                entity.set_qpos(qpos_tensor, envs_idx=env_idx, zero_velocity=False)
            else:
                entity.set_qpos(qpos_tensor, zero_velocity=False)

        # Update ref_dof_prev for next timestep
        self._update_ref_dof_prev_kernel(ad, n_envs)

        # Store current link transforms to prev_link_transforms
        for idx, (entity_idx, art_data) in enumerate(self._articulated_entities.items()):
            env_idx = art_data["env_idx"]
            for joint_idx, joint in enumerate(art_data["revolute_joints"] + art_data["prismatic_joints"]):
                child_link_idx = joint.link.idx
                if (
                    child_link_idx in self._genesis_stored_states
                    and env_idx in self._genesis_stored_states[child_link_idx]
                ):
                    key = (idx, joint_idx, env_idx)
                    ad.prev_link_transforms[key] = self._genesis_stored_states[child_link_idx][env_idx].copy()

    # ============================================================
    # Section 7: IPC-Only Coupling
    # ============================================================

    def _apply_ipc_only_robot_qpos(self, entity_indices, env_idx: int, is_parallelized: bool):
        """Update robot qpos from IPC transforms; returns list of updated entities."""
        if not hasattr(self, "abd_data_by_link"):
            return []

        rigid_solver = self.rigid_solver
        updated_entities = []

        for entity_idx in entity_indices:
            entity = rigid_solver._entities[entity_idx]
            link_idx = entity.base_link_idx

            if link_idx not in self.abd_data_by_link:
                continue
            env_data = self.abd_data_by_link[link_idx]
            if env_idx not in env_data:
                continue

            ipc_transform = env_data[env_idx].get("transform")
            if ipc_transform is None:
                continue

            pos, quat_wxyz = decompose_transform_matrix(ipc_transform)

            if entity.n_qs < 7:
                gs.logger.warning(
                    f"ipc_only entity {entity_idx} has n_qs={entity.n_qs}; expected at least 7 for base pose."
                )
                continue

            if is_parallelized:
                qpos_current = entity.get_qpos(envs_idx=env_idx).detach().cpu().numpy()
            else:
                qpos_current = entity.get_qpos().detach().cpu().numpy()

            if qpos_current.ndim > 1:
                qpos_current = qpos_current[0]

            qpos_new = qpos_current.copy()
            qpos_new[:3] = pos
            qpos_new[3:7] = quat_wxyz

            qpos_tensor = gs.torch.as_tensor(qpos_new, dtype=gs.tc_float, device=gs.device)
            if is_parallelized:
                entity.set_qpos(qpos_tensor, envs_idx=env_idx, zero_velocity=True, skip_forward=True)
            else:
                entity.set_qpos(qpos_tensor, envs_idx=None, zero_velocity=True, skip_forward=True)

            updated_entities.append(entity_idx)

        return updated_entities

    def _finalize_ipc_only_robot_fk(self, entity_indices, env_idx: int, is_parallelized: bool):
        """Run forward kinematics/geoms update for robot entities only."""
        if not entity_indices:
            return

        envs_idx = self.sim._scene._sanitize_envs_idx(env_idx if is_parallelized else None)
        for entity_idx in entity_indices:
            self.rigid_solver._func_forward_kinematics_entity(entity_idx, envs_idx)
        self.rigid_solver._func_update_geoms(envs_idx)
        self.rigid_solver._is_forward_pos_updated = True
        self.rigid_solver._is_forward_vel_updated = True

    def _apply_ipc_only_transforms(self, entity_indices, env_idx: int, is_parallelized: bool):
        """Apply IPC transforms directly to base links (non-robots)."""
        if not hasattr(self, "abd_data_by_link"):
            return

        rigid_solver = self.rigid_solver
        pos_list = []
        quat_list = []
        link_idx_list = []

        for entity_idx in entity_indices:
            entity = rigid_solver._entities[entity_idx]
            link_idx = entity.base_link_idx

            if link_idx not in self.abd_data_by_link:
                continue
            env_data = self.abd_data_by_link[link_idx]
            if env_idx not in env_data:
                continue

            ipc_transform = env_data[env_idx].get("transform")
            if ipc_transform is None:
                continue

            pos, quat_wxyz = decompose_transform_matrix(ipc_transform)

            pos_list.append(pos)
            quat_list.append(quat_wxyz)
            link_idx_list.append(link_idx)

        if not pos_list:
            return

        pos_tensor = gs.torch.as_tensor(np.array(pos_list), dtype=gs.tc_float, device=gs.device)
        quat_tensor = gs.torch.as_tensor(np.array(quat_list), dtype=gs.tc_float, device=gs.device)
        link_idx_tensor = gs.torch.as_tensor(np.array(link_idx_list), dtype=gs.tc_int, device=gs.device)

        if is_parallelized:
            rigid_solver.set_base_links_pos(pos_tensor, link_idx_tensor, envs_idx=env_idx, relative=False)
            rigid_solver.set_base_links_quat(quat_tensor, link_idx_tensor, envs_idx=env_idx, relative=False)
        else:
            rigid_solver.set_base_links_pos(pos_tensor, link_idx_tensor, envs_idx=None, relative=False)
            rigid_solver.set_base_links_quat(quat_tensor, link_idx_tensor, envs_idx=None, relative=False)

        for entity_idx in entity_indices:
            entity = rigid_solver._entities[entity_idx]
            if is_parallelized:
                entity.zero_all_dofs_velocity(envs_idx=env_idx)
            else:
                entity.zero_all_dofs_velocity(envs_idx=None)

    def _post_advance_ipc_only(self, entity_indices):
        """
        Post-advance processing for ipc_only entities.
        Directly sets Genesis transforms from IPC results.
        Only handles simple case (single base link entities).
        """
        rigid_solver = self.rigid_solver
        is_parallelized = self.sim._scene.n_envs > 0
        n_envs = self.sim._scene.n_envs if is_parallelized else 1

        for env_idx in range(n_envs):
            robot_entities = []
            non_robot_entities = []

            for entity_idx in entity_indices:
                entity = rigid_solver._entities[entity_idx]
                if is_robot_entity(entity):
                    robot_entities.append(entity_idx)
                else:
                    non_robot_entities.append(entity_idx)

            updated_robot_entities = self._apply_ipc_only_robot_qpos(robot_entities, env_idx, is_parallelized)

            if non_robot_entities:
                self._apply_ipc_only_transforms(non_robot_entities, env_idx, is_parallelized)
            else:
                # Only robots are present; run FK update explicitly.
                self._finalize_ipc_only_robot_fk(updated_robot_entities, env_idx, is_parallelized)

    # ============================================================
    # Section 8: FEM State Retrieval
    # ============================================================

    def _retrieve_fem_states(self, f):
        # IPC world advance/retrieve is handled at Scene level
        # This method handles both volumetric FEM (3D) and cloth (2D) post-processing

        if not self.fem_solver.is_active:
            return

        # Gather FEM states (both volumetric and cloth) using metadata filtering
        from uipc import builtin
        from uipc.backend import SceneVisitor
        from uipc.geometry import SimplicialComplexSlot, apply_transform, merge

        visitor = SceneVisitor(self._ipc_scene)

        # Collect FEM and cloth geometries using metadata
        fem_geo_by_entity = {}
        for geo_slot in visitor.geometries():
            if isinstance(geo_slot, SimplicialComplexSlot):
                geo = geo_slot.geometry()
                if geo.dim() in [2, 3]:
                    meta = read_ipc_geometry_metadata(geo)
                    if meta is None:
                        continue
                    solver_type, env_idx, entity_idx = meta
                    if solver_type not in ("fem", "cloth"):
                        continue

                    try:
                        if entity_idx not in fem_geo_by_entity:
                            fem_geo_by_entity[entity_idx] = {}

                        proc_geo = geo
                        if geo.instances().size() >= 1:
                            proc_geo = merge(apply_transform(geo))
                        pos = proc_geo.positions().view().reshape(-1, 3)
                        fem_geo_by_entity[entity_idx][env_idx] = pos
                    except Exception:
                        continue

        # Update FEM entities using filtered geometries
        for entity_idx, env_positions in fem_geo_by_entity.items():
            if entity_idx < len(self.fem_solver._entities):
                entity = self.fem_solver._entities[entity_idx]
                env_pos_list = []

                for env_idx in range(self.sim._B):
                    if env_idx in env_positions:
                        env_pos_list.append(env_positions[env_idx])
                    else:
                        # Fallback for missing environment
                        env_pos_list.append(np.zeros((0, 3)))

                if env_pos_list:
                    all_env_pos = np.stack(env_pos_list, axis=0, dtype=gs.np_float)
                    entity.set_pos(0, all_env_pos)

    def _retrieve_rigid_states(self, f, entity_set=None):
        """
        Handle rigid body IPC: Retrieve ABD transforms/affine matrices after IPC step
        and apply coupling forces back to Genesis rigid bodies.

        Parameters
        ----------
        f : int
            Frame number
        entity_set : set, optional
            Set of entity indices to process. If None, process all.
        """
        # IPC world advance/retrieve is handled at Scene level
        # Retrieve ABD transform matrices after IPC simulation

        if not hasattr(self, "_ipc_scene") or not hasattr(self.rigid_solver, "list_env_mesh"):
            return

        from uipc import builtin, view
        from uipc.backend import SceneVisitor
        from uipc.geometry import SimplicialComplexSlot
        import genesis.utils.geom as gu

        rigid_solver = self.rigid_solver
        visitor = SceneVisitor(self._ipc_scene)

        # Collect ABD geometries and their constraint data using metadata
        abd_data_by_link = {}  # link_idx -> {env_idx: {transform, gradient, mass}}

        for geo_slot in visitor.geometries():
            if isinstance(geo_slot, SimplicialComplexSlot):
                geo = geo_slot.geometry()
                if geo.dim() in [2, 3]:
                    meta = read_ipc_geometry_metadata(geo)
                    if meta is None:
                        continue
                    solver_type, env_idx, link_idx = meta
                    if solver_type != "rigid":
                        continue

                    try:
                        # Filter by entity_set if specified
                        if entity_set is not None:
                            entity_idx = rigid_solver.links_info.entity_idx[link_idx]
                            if entity_idx not in entity_set:
                                continue

                        if link_idx not in abd_data_by_link:
                            abd_data_by_link[link_idx] = {}

                        # Get current transform matrix from ABD object (after IPC solve)
                        transforms = geo.transforms()
                        transform_matrix = None
                        if transforms.size() > 0:
                            transform_matrix = view(transforms)[0].copy()

                        # Get aim transform (q_genesis^n stored before advance)
                        aim_transform = None
                        if link_idx in self._genesis_stored_states and env_idx in self._genesis_stored_states[link_idx]:
                            aim_transform = self._genesis_stored_states[link_idx][env_idx]

                        abd_data_by_link[link_idx][env_idx] = {
                            "transform": transform_matrix,
                            "aim_transform": aim_transform,
                        }

                    except Exception as e:
                        gs.logger.warning(f"Failed to retrieve ABD geometry data: {e}")
                        continue

        # Store transforms for later access
        self.abd_data_by_link = abd_data_by_link

    def _apply_abd_coupling_forces(self, entity_set=None):
        """
        Apply coupling forces from IPC ABD constraint to Genesis rigid bodies using taichi kernel.

        This ensures action-reaction force consistency:
        - IPC constraint force: G_ipc = M * (q_ipc^{n+1} - q_genesis^n)
        - Genesis reaction force: F_genesis = M * (q_ipc^{n+1} - q_genesis^n) = G_ipc

        Where:
        - q_ipc^{n+1}: IPC ABD position after solve (from geo.transforms())
        - q_genesis^n: Genesis position before IPC advance (stored in _genesis_stored_states)
        - M: Mass matrix scaled by constraint strengths

        Parameters
        ----------
        entity_set : set, optional
            Set of entity indices to process. If None, process all two_way_soft_constraint entities.
        """
        import torch

        rigid_solver = self.rigid_solver
        strength_tuple = self.options.ipc_constraint_strength
        translation_strength = float(strength_tuple[0])
        rotation_strength = float(strength_tuple[1])

        dt = self.sim._dt
        dt2 = dt * dt

        # Collect all link data directly into pre-allocated Taichi buffers
        cd = self.coupling_data
        n_items = 0

        for link_idx, env_data in self.abd_data_by_link.items():
            # Filter by entity_set if specified
            entity_idx = rigid_solver.links_info.entity_idx[link_idx]
            if entity_set is not None and entity_idx not in entity_set:
                continue

            for env_idx, data in env_data.items():
                ipc_transform = data.get("transform")  # Current transform after IPC solve
                aim_transform = data.get("aim_transform")  # Target from Genesis

                if ipc_transform is None or aim_transform is None:
                    continue

                try:
                    # Write directly to Taichi fields
                    cd.link_indices[n_items] = link_idx
                    cd.env_indices[n_items] = env_idx

                    # Copy transform matrices
                    for row in range(4):
                        for col in range(4):
                            cd.ipc_transforms[n_items][row, col] = ipc_transform[row, col]
                            cd.aim_transforms[n_items][row, col] = aim_transform[row, col]

                    cd.link_masses[n_items] = float(rigid_solver.links_info.inertial_mass[link_idx])

                    # Copy inertia tensor
                    inertia = rigid_solver.links_info.inertial_i[link_idx]
                    for row in range(3):
                        for col in range(3):
                            cd.inertia_tensors[n_items][row, col] = inertia[row, col]

                    n_items += 1
                except Exception as e:
                    gs.logger.warning(f"Failed to collect data for link {link_idx}, env {env_idx}: {e}")
                    continue

        if n_items == 0:
            return  # No links to process

        cd.n_items[None] = n_items

        # Call taichi kernel with pre-allocated fields
        # IMPORTANT: Pass Taichi fields directly, not numpy arrays, so kernel can write results
        self._compute_coupling_forces_kernel(
            n_items,
            cd.ipc_transforms,
            cd.aim_transforms,
            cd.link_masses,
            cd.inertia_tensors,
            translation_strength,
            rotation_strength,
            dt2,
            cd.out_forces,
            cd.out_torques,
        )

        # Apply forces to Genesis rigid bodies - OPTIMIZED batch processing
        is_parallelized = self.sim._scene.n_envs > 0

        # Export Taichi fields to numpy once (avoid per-element access in loops)
        out_forces_np = cd.out_forces.to_numpy()[:n_items]  # (n_items, 3)
        out_torques_np = cd.out_torques.to_numpy()[:n_items]  # (n_items, 3)
        link_indices_np = cd.link_indices.to_numpy()[:n_items]
        env_indices_np = cd.env_indices.to_numpy()[:n_items]

        if is_parallelized:
            # Group by environment using numpy arrays
            env_batches = {}  # {env_idx: {'link_indices': [], 'forces': [], 'torques': []}}
            for i in range(n_items):
                env_idx = int(env_indices_np[i])
                if env_idx not in env_batches:
                    env_batches[env_idx] = {"link_indices": [], "forces": [], "torques": []}
                env_batches[env_idx]["link_indices"].append(int(link_indices_np[i]))
                env_batches[env_idx]["forces"].append(out_forces_np[i])
                env_batches[env_idx]["torques"].append(out_torques_np[i])

            # Apply forces per environment
            for env_idx, batch in env_batches.items():
                for j, link_idx in enumerate(batch["link_indices"]):
                    try:
                        force_input = batch["forces"][j].reshape(1, 1, 3)
                        torque_input = batch["torques"][j].reshape(1, 1, 3)
                        # check if force_input is nan
                        if np.isnan(force_input).any() or np.isnan(torque_input).any():
                            gs.logger.warning(
                                f"NaN detected in ABD coupling force/torque for link {link_idx}, env {env_idx}. Skipping application."
                            )
                            continue
                        rigid_solver.apply_links_external_force(force=force_input, links_idx=link_idx, envs_idx=env_idx)
                        rigid_solver.apply_links_external_torque(
                            torque=torque_input, links_idx=link_idx, envs_idx=env_idx
                        )
                    except Exception as e:
                        gs.logger.warning(f"Failed to apply ABD coupling force for link {link_idx}, env {env_idx}: {e}")
                        continue
        else:
            # Non-parallelized: apply all forces using numpy slices
            for i in range(n_items):
                link_idx = int(link_indices_np[i])
                try:
                    force_input = out_forces_np[i].reshape(1, 3)
                    torque_input = out_torques_np[i].reshape(1, 3)
                    # check if force_input is nan
                    if np.isnan(force_input).any() or np.isnan(torque_input).any():
                        gs.logger.warning(
                            f"NaN detected in ABD coupling force/torque for link {link_idx}. Skipping application."
                        )
                        continue
                    rigid_solver.apply_links_external_force(force=force_input, links_idx=link_idx)
                    rigid_solver.apply_links_external_torque(torque=torque_input, links_idx=link_idx)
                except Exception as e:
                    gs.logger.warning(f"Failed to apply ABD coupling force for link {link_idx}: {e}")
                    continue

    def couple_grad(self, f):
        """Gradient computation for coupling"""
        # IPC doesn't support gradients yet
        pass

    def reset(self, envs_idx=None):
        """Reset coupling state"""
        # IPC doesn't need special reset logic currently
        pass

    # ============================================================
    # Section 9: GUI
    # ============================================================

    def _init_ipc_gui(self):
        """Initialize IPC GUI for debugging"""
        try:
            import polyscope as ps
            from uipc.gui import SceneGUI

            self.ps = ps

            # Initialize SceneGUI for IPC scene
            self._ipc_scene_gui = SceneGUI(self._ipc_scene, "split")

            # Initialize polyscope if not already done
            if not ps.is_initialized():
                ps.init()

            # Register IPC GUI with polyscope
            self._ipc_scene_gui.register()
            self._ipc_scene_gui.set_edge_width(1)

            # Set up ground plane display in polyscope to match Genesis z=0
            ps.set_up_dir("z_up")
            ps.set_ground_plane_height(0.0)  # Set at z=0 to match Genesis

            # Show polyscope window for first frame to initialize properly
            ps.show(forFrames=1)
            # Flag to control GUI updates
            self.sim._scene._ipc_gui_enabled = True

            gs.logger.info("IPC GUI initialized successfully")

        except Exception as e:
            gs.logger.warning(f"Failed to initialize IPC GUI: {e}")
            self.sim._scene._ipc_gui_enabled = False

    def update_ipc_gui(self):
        """Update IPC GUI"""
        self.ps.frame_tick()  # Non-blocking frame update
        self._ipc_scene_gui.update()

    # ============================================================
    # Section 10: Contact Forces
    # ============================================================

    def _compute_link_contact_forces_and_torques(self, total_force_dict, vertex_to_link, link_vertex_positions):
        """
        Compute total contact forces and torques for each rigid link from vertex gradients.

        OPTIMIZED VERSION: Uses Taichi kernel for parallel computation.

        This function follows the pattern from test_affine_body_contact_force.py, computing
        the total force and torque acting on each rigid body by accumulating contributions
        from all contact vertices.

        Parameters
        ----------
        total_force_dict : dict
            Dictionary mapping vertex indices to contact force gradients {vertex_idx: force_gradient_vector}
        vertex_to_link : dict
            Mapping from global vertex index to (link_idx, env_idx, local_idx)
        link_vertex_positions : dict
            Mapping from (link_idx, env_idx) to list of vertex positions in world space

        Returns
        -------
        dict
            Dictionary mapping (link_idx, env_idx) to {'force': np.array, 'torque': np.array, 'center': np.array}

        Notes
        -----
        - Force is computed as the negative sum of contact gradients: F = -∑grad
        - Torque is computed as τ = ∑(r × F) where r is the vector from link center to contact point
        - Link center is computed as the average of all vertex positions
        """
        if not total_force_dict:
            return {}

        # Step 1: Prepare link centers (compute once per (link_idx, env_idx))
        link_centers_dict = {}  # {(link_idx, env_idx): center}
        for (link_idx, env_idx), verts in link_vertex_positions.items():
            link_centers_dict[(link_idx, env_idx)] = np.mean(verts, axis=0)

        # Step 2: Prepare data for kernel - collect all vertex force entries
        vertex_data = []
        for vert_idx, force_grad in total_force_dict.items():
            if vert_idx not in vertex_to_link:
                continue  # Vertex doesn't belong to a both-coupling link

            link_idx, env_idx, local_idx = vertex_to_link[vert_idx]

            # Get vertex position and link center
            if (link_idx, env_idx) in link_vertex_positions:
                contact_pos = link_vertex_positions[(link_idx, env_idx)][local_idx]
                center_pos = link_centers_dict.get((link_idx, env_idx))
                if center_pos is not None:
                    vertex_data.append(
                        {
                            "force_grad": force_grad,
                            "link_idx": link_idx,
                            "env_idx": env_idx,
                            "contact_pos": contact_pos,
                            "center_pos": center_pos,
                        }
                    )

        if not vertex_data:
            return {}

        # Check capacity
        n_entries = len(vertex_data)
        if n_entries > self.MAX_VERTEX_CONTACTS:
            gs.logger.warning(
                f"Vertex contact capacity exceeded: {n_entries} > {self.MAX_VERTEX_CONTACTS}. "
                f"Truncating to {self.MAX_VERTEX_CONTACTS}."
            )
            n_entries = self.MAX_VERTEX_CONTACTS
            vertex_data = vertex_data[:n_entries]

        # Step 3: Populate Taichi fields
        for i, data in enumerate(vertex_data):
            self.vertex_force_gradients[i] = data["force_grad"]
            self.vertex_link_indices[i] = data["link_idx"]
            self.vertex_env_indices[i] = data["env_idx"]
            self.vertex_positions_world[i] = data["contact_pos"]
            self.vertex_link_centers[i] = data["center_pos"]

        # Step 4: Clear output arrays
        self.link_contact_forces_out.fill(0.0)
        self.link_contact_torques_out.fill(0.0)

        # Step 5: Call kernel directly on Taichi fields (no numpy conversion)
        self._compute_link_contact_forces_kernel(
            n_entries,
            self.vertex_force_gradients,
            self.vertex_link_indices,
            self.vertex_env_indices,
            self.vertex_positions_world,
            self.vertex_link_centers,
            self.link_contact_forces_out,
            self.link_contact_torques_out,
        )

        # Step 6: Extract results using to_numpy() once (avoid per-element access)
        forces_all = self.link_contact_forces_out.to_numpy()  # (max_links, max_envs, 3)
        torques_all = self.link_contact_torques_out.to_numpy()  # (max_links, max_envs, 3)

        link_forces = {}  # {(link_idx, env_idx): {'force': np.array, 'torque': np.array, 'center': np.array}}

        for (link_idx, env_idx), center in link_centers_dict.items():
            # Use numpy slices instead of list comprehension
            force = forces_all[link_idx, env_idx]
            torque = torques_all[link_idx, env_idx]

            # Only include if there's non-zero force/torque
            if np.any(force != 0.0) or np.any(torque != 0.0):
                link_forces[(link_idx, env_idx)] = {
                    "force": force,
                    "torque": torque,
                    "center": center,
                }

        return link_forces

    def _record_ipc_contact_forces(self):
        """
        Record contact forces from IPC for two_way_soft_constraint coupling links.

        This method extracts contact forces and torques from IPC's contact system
        and stores them for later application to Genesis rigid bodies.
        """
        from uipc import view
        from uipc.geometry import Geometry

        # Clear previous contact forces
        self._ipc_contact_forces.clear()
        self._external_force_data.clear()

        # Get contact feature from IPC world
        features = self._ipc_world.features()
        contact_feature = features.find("core/contact_system")

        if contact_feature is None:
            return  # No contact system available

        # Get available contact primitive types
        prim_types = contact_feature.contact_primitive_types()

        # Accumulate contact gradients (forces) for all vertices
        # NOTE: IPC gradients are actually force * dt^2, so we need to divide by dt^2
        dt = self.options.dt
        dt2 = dt * dt
        total_force_dict = {}  # {vertex_index: force_vector}

        for prim_type in prim_types:
            # Get contact gradient for this primitive type
            vert_grad = Geometry()
            contact_feature.contact_gradient(prim_type, vert_grad)

            # Extract gradient data from instances
            instances = vert_grad.instances()
            i_attr = instances.find("i")  # Vertex indices
            grad_attr = instances.find("grad")  # Gradient vectors

            if i_attr is not None and grad_attr is not None:
                # view() returns numpy array directly - no conversion needed
                indices = view(i_attr)  # shape: (n,)
                gradients = view(grad_attr)  # shape may be (n, 3) or (n, 3, 3)

                # Skip if empty
                if len(indices) == 0 or gradients.size == 0:
                    continue

                # Handle different gradient shapes - could be vector (n, 3) or matrix (n, 3, 3)
                # Flatten each gradient and take first 3 elements
                if gradients.ndim == 3:
                    # Matrix gradients (n, 3, 3) -> flatten to (n, 9) -> take first 3
                    scaled_grads = gradients.reshape(len(gradients), -1)[:, :3] / dt2
                else:
                    # Vector gradients (n, 3) -> use directly
                    scaled_grads = gradients[:, :3] / dt2

                # Accumulate forces per vertex index
                for i, idx in enumerate(indices):
                    if idx not in total_force_dict:
                        total_force_dict[idx] = np.zeros(3)
                    total_force_dict[idx] += scaled_grads[i]

        if not total_force_dict:
            return  # No contact forces to process

        # Use pre-built vertex-to-link mapping (built once during IPC setup)
        vertex_to_link = self._vertex_to_link_mapping

        # Get current vertex positions for contact force computation
        from uipc.backend import SceneVisitor
        from uipc.geometry import SimplicialComplexSlot

        visitor = SceneVisitor(self._ipc_scene)
        link_vertex_positions = {}  # {(link_idx, env_idx): [vertex_positions]}

        global_vertex_offset = 0

        for geo_slot in visitor.geometries():
            if isinstance(geo_slot, SimplicialComplexSlot):
                geo = geo_slot.geometry()
                if geo.dim() in [2, 3]:
                    n_verts = geo.vertices().size()

                    try:
                        meta = read_ipc_geometry_metadata(geo)
                        if meta is not None:
                            solver_type, env_idx, link_idx = meta
                            if solver_type == "rigid":
                                # Check if any vertex of this geometry is in the mapping
                                first_vertex_idx = global_vertex_offset
                                if first_vertex_idx in vertex_to_link:
                                    transforms = geo.transforms()
                                    if transforms.size() > 0:
                                        transform_matrix = view(transforms)[0]
                                        positions = view(geo.positions())

                                        if (link_idx, env_idx) not in link_vertex_positions:
                                            link_vertex_positions[(link_idx, env_idx)] = []

                                        for local_idx in range(n_verts):
                                            local_pos = np.array(positions[local_idx]).flatten()[:3]
                                            local_pos_homogeneous = np.append(local_pos, 1.0)
                                            world_pos = (transform_matrix @ local_pos_homogeneous)[:3]
                                            link_vertex_positions[(link_idx, env_idx)].append(world_pos)

                    except Exception as e:
                        gs.logger.warning(f"Failed to process geometry for contact forces: {e}")
                    finally:
                        global_vertex_offset += n_verts

        # Compute contact forces and torques for each link
        link_forces = self._compute_link_contact_forces_and_torques(
            total_force_dict, vertex_to_link, link_vertex_positions
        )

        # Store forces in the proper format
        for (link_idx, env_idx), data in link_forces.items():
            if link_idx not in self._ipc_contact_forces:
                self._ipc_contact_forces[link_idx] = {}

            self._ipc_contact_forces[link_idx][env_idx] = {"force": data["force"], "torque": data["torque"]}

        # Compute external force from contact forces using taichi kernel
        # Collect data directly into pre-allocated Taichi fields
        contact_idx = 0

        for link_idx, env_data in self._ipc_contact_forces.items():
            for env_idx, force_data in env_data.items():
                # Check if we've exceeded pre-allocated capacity
                if contact_idx >= self.contact_forces_ti.shape[0]:
                    gs.logger.warning(
                        f"Contact capacity exceeded: {contact_idx} >= {self.contact_forces_ti.shape[0]}. "
                        f"Increase max_contacts in __init__."
                    )
                    break

                # Get contact force and torque
                force = force_data["force"]
                torque = force_data["torque"]

                # Get ABD transform for this link
                if hasattr(self, "abd_data_by_link") and link_idx in self.abd_data_by_link:
                    if env_idx in self.abd_data_by_link[link_idx]:
                        abd_transform = self.abd_data_by_link[link_idx][env_idx].get("transform")
                        if abd_transform is not None:
                            # Write directly to pre-allocated Taichi fields
                            self.contact_forces_ti[contact_idx] = force
                            self.contact_torques_ti[contact_idx] = torque
                            self.abd_transforms_ti[contact_idx] = abd_transform
                            self.link_indices_ti[contact_idx] = link_idx
                            self.env_indices_ti[contact_idx] = env_idx
                            contact_idx += 1

        if contact_idx > 0:
            # Use a view/slice of the pre-allocated arrays for the kernel call
            # Call taichi kernel to compute forces (no numpy conversion needed)
            n_links = contact_idx
            self._compute_external_force_kernel(
                n_links,
                self.contact_forces_ti,
                self.contact_torques_ti,
                self.abd_transforms_ti,
                self.out_forces_ti,
            )

            # Export to numpy once (avoid per-element access in loop)
            out_forces_np = self.out_forces_ti.to_numpy()[:n_links]  # (n_links, 12)
            link_indices_np = self.link_indices_ti.to_numpy()[:n_links]
            env_indices_np = self.env_indices_ti.to_numpy()[:n_links]

            # Store forces in _external_force_data for animator to use
            for i in range(n_links):
                link_idx = int(link_indices_np[i])
                env_idx = int(env_indices_np[i])
                # Use numpy slice instead of list comprehension
                force_vector = out_forces_np[i].astype(np.float64)
                self._external_force_data[(link_idx, env_idx)] = force_vector

    def _apply_ipc_contact_forces(self):
        """
        Apply recorded IPC contact forces to Genesis rigid bodies.

        OPTIMIZED VERSION: Batches forces and torques to reduce API call overhead.

        This method takes the contact forces and torques recorded by _record_ipc_contact_forces
        and applies them to the corresponding Genesis rigid links.
        """
        import torch

        if not self._ipc_contact_forces:
            return  # No contact forces to apply

        rigid_solver = self.rigid_solver
        is_parallelized = self.sim._scene.n_envs > 0

        # Collect all forces and torques into batches
        # Group by environment for parallelized scenes, or single batch for non-parallelized
        if is_parallelized:
            # For parallelized scenes, we need to group by env_idx
            env_batches = {}  # {env_idx: {'link_indices': [], 'forces': [], 'torques': []}}

            for link_idx, env_data in self._ipc_contact_forces.items():
                for env_idx, force_data in env_data.items():
                    if env_idx not in env_batches:
                        env_batches[env_idx] = {"link_indices": [], "forces": [], "torques": []}

                    env_batches[env_idx]["link_indices"].append(link_idx)
                    env_batches[env_idx]["forces"].append(force_data["force"] * 0.5)
                    env_batches[env_idx]["torques"].append(force_data["torque"] * 0.5)

            # Apply forces for each environment batch
            for env_idx, batch_data in env_batches.items():
                if not batch_data["link_indices"]:
                    continue

                # Convert to torch tensors for batch application
                forces_tensor = torch.as_tensor(
                    np.array(batch_data["forces"], dtype=np.float32),
                    dtype=gs.tc_float,
                    device=gs.device,
                )  # (n_links, 3)
                torques_tensor = torch.as_tensor(
                    np.array(batch_data["torques"], dtype=np.float32),
                    dtype=gs.tc_float,
                    device=gs.device,
                )  # (n_links, 3)
                link_indices = batch_data["link_indices"]

                # Apply forces/torques in batch
                rigid_solver.apply_links_external_force(
                    force=forces_tensor,
                    links_idx=link_indices,
                    envs_idx=env_idx,
                    local=False,
                )
                rigid_solver.apply_links_external_torque(
                    torque=torques_tensor,
                    links_idx=link_indices,
                    envs_idx=env_idx,
                    local=False,
                )

        else:
            # For non-parallelized scenes, batch all together
            all_link_indices = []
            all_forces = []
            all_torques = []

            for link_idx, env_data in self._ipc_contact_forces.items():
                for env_idx, force_data in env_data.items():
                    all_link_indices.append(link_idx)
                    all_forces.append(force_data["force"] * 0.5)
                    all_torques.append(force_data["torque"] * 0.5)

            if not all_link_indices:
                return

            # Convert to torch tensors for batch application
            forces_tensor = torch.as_tensor(
                np.array(all_forces, dtype=np.float32), dtype=gs.tc_float, device=gs.device
            )  # (n_links, 3)
            torques_tensor = torch.as_tensor(
                np.array(all_torques, dtype=np.float32), dtype=gs.tc_float, device=gs.device
            )  # (n_links, 3)

            # Apply forces/torques in batch
            rigid_solver.apply_links_external_force(
                force=forces_tensor,
                links_idx=all_link_indices,
                local=False,
            )
            rigid_solver.apply_links_external_torque(
                torque=torques_tensor,
                links_idx=all_link_indices,
                local=False,
            )

    # ============================================================
    # Section 11: Articulation IPC Setup & Geometry Lookup
    # ============================================================

    def _add_articulated_entities_to_ipc(self):
        """
        Add articulated robot entities to IPC using ExternalArticulationConstraint.
        This enables joint-level coupling between Genesis and IPC.
        """
        from uipc.constitution import (
            ExternalArticulationConstraint,
            AffineBodyConstitution,
            AffineBodyRevoluteJoint,
            AffineBodyPrismaticJoint,
        )
        from uipc.geometry import label_surface

        from uipc import view

        # Create ExternalArticulationConstraint if not already created
        if self._ipc_eac is None:
            self._ipc_eac = ExternalArticulationConstraint()
            self._ipc_scene.constitution_tabular().insert(self._ipc_eac)

        rigid_solver = self.rigid_solver
        scene = self._ipc_scene

        # Process each rigid entity with external_articulation coupling type
        for entity_idx in range(len(rigid_solver._entities)):
            # Only process entities with external_articulation coupling type
            entity_coupling_type = self._entity_coupling_types.get(entity_idx)
            if entity_coupling_type != "external_articulation":
                continue

            entity = rigid_solver._entities[entity_idx]

            # Extract joints from the entity
            joint_info = extract_articulated_joints(entity)

            if joint_info["n_joints"] == 0:
                continue  # Skip entities without joints

            gs.logger.info(
                f"Adding articulated entity {entity_idx} with {joint_info['n_joints']} joints "
                f"({len(joint_info['revolute_joints'])} revolute, {len(joint_info['prismatic_joints'])} prismatic)"
            )

            # Create joint geometries and slots for libuipc (following test_external_articulation_constraint.py)
            from uipc.geometry import linemesh

            joint_geo_slots = []
            joint_objects = []  # Store joint objects for later reference

            # Create constitutions for joints
            abrj = AffineBodyRevoluteJoint()
            abpj = AffineBodyPrismaticJoint()

            # Add revolute joints
            for joint in joint_info["revolute_joints"]:
                # Get parent and child links
                # joint.link is the child link (the one that moves)
                # joint.link.parent_idx is the parent link index
                child_link_idx = joint.link.idx
                parent_link_idx_original = joint.link.parent_idx if joint.link.parent_idx >= 0 else 0

                # IMPORTANT: Following mjcf.py logic (line 187):
                # Parent link index should be the MERGED TARGET (for fixed joint merging)
                # Child link index stays ORIGINAL (used for axis/position calculation)
                parent_link_idx = find_target_link_for_fixed_merge(self.rigid_solver, parent_link_idx_original)

                # Find the corresponding ABD geometry SLOTS in IPC scene
                # NOTE: parent_abd_slot uses target link (merged), child uses original mapping
                # Both will work because _link_to_abd_slot has mappings for both original and target
                parent_abd_slot = self._find_abd_geometry_slot_by_link(parent_link_idx, env_idx=0)
                child_abd_slot = self._find_abd_geometry_slot_by_link(child_link_idx, env_idx=0)

                if parent_abd_slot is None or child_abd_slot is None:
                    gs.logger.warning(
                        f"Skipping joint {joint.name}: ABD geometry slots not found for links {parent_link_idx}, {child_link_idx}"
                    )
                    continue

                # Get joint axis and position in world coordinates
                # libuipc uses ABSOLUTE world coordinates for linemesh vertices
                joint_axis_local = joint.dofs_motion_ang[0]  # (3,) array - rotation axis in joint frame
                child_link = self.rigid_solver.links[child_link_idx]
                parent_link = self.rigid_solver.links[parent_link_idx]

                gs.logger.info(f"\n--- Processing revolute joint: {joint.name} ---")
                gs.logger.info(f"  Parent link: {parent_link_idx} ({parent_link.name})")
                gs.logger.info(f"  Child link: {child_link_idx} ({child_link.name})")
                gs.logger.info(f"  Joint axis (joint frame): {joint_axis_local}")

                # Transform joint axis from joint frame to world frame
                # In MuJoCo/URDF convention, the axis is defined in the joint frame
                child_rot_matrix = compute_link_init_world_rotation(self.rigid_solver, child_link_idx)
                joint_axis = child_rot_matrix @ joint_axis_local  # Transform to world coordinates

                gs.logger.info(f"  Child link rotation matrix:\n{child_rot_matrix}")
                gs.logger.info(f"  Joint axis (world): {joint_axis}")

                # Get joint world position from joints_state.xanchor (computed by FK)
                joint_idx = joint.idx
                joint_pos = rigid_solver.joints_state.xanchor.to_numpy()[joint_idx, 0]  # env_idx=0
                gs.logger.info(f"  Joint position (world): {joint_pos}")

                # Create linemesh for revolute joint in world coordinates
                # Line segment centered at joint_pos, aligned with joint_axis
                axis_length = 1.0
                v1 = joint_pos - (axis_length / 2) * joint_axis
                v2 = joint_pos + (axis_length / 2) * joint_axis
                vertices = np.array([v2, v1], dtype=np.float64)
                edges = np.array([[0, 1]], dtype=np.int32)
                revolute_mesh = linemesh(vertices, edges)

                # Apply revolute joint constitution
                # abrj.apply_to(mesh, parent_slots, parent_instance_ids, child_slots, child_instance_ids, stiffnesses)
                abrj.apply_to(revolute_mesh, [parent_abd_slot], [0], [child_abd_slot], [0], [100.0])

                # Create geometry in IPC scene
                joint_obj = scene.objects().create(f"revolute_joint_{entity_idx}_{joint.name}")
                revolute_slot, _ = joint_obj.geometries().create(revolute_mesh)

                joint_geo_slots.append(revolute_slot)
                joint_objects.append(joint_obj)

            # Add prismatic joints
            for joint in joint_info["prismatic_joints"]:
                # Get parent and child links (same as revolute joints)
                child_link_idx = joint.link.idx
                parent_link_idx_original = joint.link.parent_idx if joint.link.parent_idx >= 0 else 0

                # IMPORTANT: Following mjcf.py logic (line 187):
                # Parent link index should be the MERGED TARGET (for fixed joint merging)
                # Child link index stays ORIGINAL (used for axis/position calculation)
                parent_link_idx = find_target_link_for_fixed_merge(self.rigid_solver, parent_link_idx_original)

                parent_abd_slot = self._find_abd_geometry_slot_by_link(parent_link_idx, env_idx=0)
                child_abd_slot = self._find_abd_geometry_slot_by_link(child_link_idx, env_idx=0)

                if parent_abd_slot is None or child_abd_slot is None:
                    gs.logger.warning(
                        f"Skipping joint {joint.name}: ABD geometry slots not found for links {parent_link_idx}, {child_link_idx}"
                    )
                    continue

                # Get joint axis and position in world coordinates
                # libuipc uses ABSOLUTE world coordinates for linemesh vertices
                joint_axis_local = joint.dofs_motion_vel[0]  # (3,) array - rotation axis in joint frame

                # Get link objects first
                child_link = self.rigid_solver.links[child_link_idx]
                parent_link = self.rigid_solver.links[parent_link_idx]

                gs.logger.info(f"\n--- Processing prismatic joint: {joint.name} ---")
                gs.logger.info(f"  Parent link: {parent_link_idx} ({parent_link.name})")
                gs.logger.info(f"  Child link: {child_link_idx} ({child_link.name})")
                gs.logger.info(f"  Joint axis (joint frame): {joint_axis_local}")

                child_rot_matrix = compute_link_init_world_rotation(self.rigid_solver, child_link_idx)
                joint_axis = child_rot_matrix @ joint_axis_local  # Transform to world coordinates

                # Get joint world position from joints_state.xanchor (computed by FK)
                joint_idx = joint.idx
                joint_pos = rigid_solver.joints_state.xanchor.to_numpy()[joint_idx, 0]  # env_idx=0

                gs.logger.info(f"  Child link rotation matrix:\n{child_rot_matrix}")
                gs.logger.info(f"  Joint axis (world): {joint_axis}")

                # Create linemesh for prismatic joint in world coordinates
                # Line segment centered at joint_pos, aligned with translation axis
                # Use same length as revolute joints for consistency
                axis_length = 1.0
                v1 = joint_pos - (axis_length / 2) * joint_axis
                v2 = joint_pos + (axis_length / 2) * joint_axis
                vertices = np.array([v1, v2], dtype=np.float64)
                edges = np.array([[0, 1]], dtype=np.int32)
                prismatic_mesh = linemesh(vertices, edges)

                # Apply prismatic joint constitution
                abpj.apply_to(prismatic_mesh, [parent_abd_slot], [0], [child_abd_slot], [0], [100.0])

                # Create geometry in IPC scene
                joint_obj = scene.objects().create(f"prismatic_joint_{entity_idx}_{joint.name}")
                prismatic_slot, _ = joint_obj.geometries().create(prismatic_mesh)

                joint_geo_slots.append(prismatic_slot)
                joint_objects.append(joint_obj)

            if len(joint_geo_slots) == 0:
                gs.logger.warning(f"Entity {entity_idx}: No valid joint geometry slots created")
                continue

            # Create articulation geometry using ExternalArticulationConstraint
            # eac.create_geometry(joint_geos, indices) - indices specify which joint from each geometry
            indices = [0] * len(joint_geo_slots)  # First (and only) joint from each geometry
            articulation_geo = self._ipc_eac.create_geometry(joint_geo_slots, indices)

            # Initialize mass matrix (n_joints x n_joints) as identity for now
            n_joints = len(joint_geo_slots)
            mass_matrix = np.eye(n_joints, dtype=np.float64) * 1e4  # Default stiffness

            # Set mass matrix (column-major storage)
            mass_attr = articulation_geo["joint_joint"].find("mass")
            mass_view = view(mass_attr)
            mass_view[:] = mass_matrix.T.flatten()  # Column-major
            # Create object in IPC scene and get the articulation geometry slot
            articulation_object = scene.objects().create(f"articulation_entity_{entity_idx}")
            articulation_slot, _ = articulation_object.geometries().create(articulation_geo)

            # Store articulation data
            self._articulated_entities[entity_idx] = {
                "entity": entity,
                "env_idx": 0,  # TODO: support multi-environment
                "revolute_joints": joint_info["revolute_joints"],
                "prismatic_joints": joint_info["prismatic_joints"],
                "joint_geo_slots": joint_geo_slots,
                "articulation_geo": articulation_geo,
                "articulation_slot": articulation_slot,  # Store the slot for later updates
                "articulation_object": articulation_object,
                "n_joints": n_joints,
                "ref_dof_prev": np.zeros(entity.n_dofs, dtype=np.float64),
                "delta_theta_tilde": np.zeros(n_joints, dtype=np.float64),
                "delta_theta": np.zeros(n_joints, dtype=np.float64),
                "joint_dof_indices": joint_info["joint_dof_indices"],
                "mass_matrix": mass_matrix,
            }

            gs.logger.info(f"Successfully added articulated entity {entity_idx} to IPC")

    def _find_abd_geometry_by_link(self, link_idx, env_idx=0):
        """
        Find the ABD geometry corresponding to a link_idx in the IPC scene.

        Parameters
        ----------
        link_idx : int
            Genesis link index
        env_idx : int
            Environment index (default: 0)

        Returns
        -------
        Geometry or None
            The ABD geometry object if found, None otherwise
        """
        # Look up in the mapping created during _add_rigid_geoms_to_ipc
        return self._link_to_abd_geo.get((env_idx, link_idx), None)

    def _find_abd_geometry_slot_by_link(self, link_idx, env_idx=0):
        """
        Find the ABD geometry slot corresponding to a link_idx in the IPC scene.

        Parameters
        ----------
        link_idx : int
            Genesis link index
        env_idx : int
            Environment index (default: 0)

        Returns
        -------
        GeometrySlot or None
            The ABD geometry slot if found, None otherwise
        """
        # Look up in the mapping created during _add_rigid_geoms_to_ipc
        return self._link_to_abd_slot.get((env_idx, link_idx), None)
