"""
Data classes for IPC coupler.

Numpy-backed data structures used for IPC coupling computations.
"""

import numpy as np

import genesis as gs


class IPCCouplingData:
    """Pre-allocated numpy arrays for coupling force computation."""

    def __init__(self, max_links):
        self.link_indices = np.empty(max_links, dtype=np.int32)
        self.env_indices = np.empty(max_links, dtype=np.int32)
        self.ipc_transforms = np.empty((max_links, 4, 4), dtype=gs.np_float)
        self.aim_transforms = np.empty((max_links, 4, 4), dtype=gs.np_float)
        self.link_masses = np.empty(max_links, dtype=gs.np_float)
        self.inertia_tensors = np.empty((max_links, 3, 3), dtype=gs.np_float)
        self.out_forces = np.empty((max_links, 3), dtype=gs.np_float)
        self.out_torques = np.empty((max_links, 3), dtype=gs.np_float)
        self.n_items = 0


class ArticulationData:
    """Numpy-backed data for joint articulation coupling."""

    def __init__(self, n_entities, max_dofs_per_entity, max_joints_per_entity, n_envs):
        # Entity-level metadata
        self.n_entities = n_entities
        self.entity_indices = np.zeros(n_entities, dtype=np.int32)
        self.entity_n_dofs = np.zeros(n_entities, dtype=np.int32)
        self.entity_n_joints = np.zeros(n_entities, dtype=np.int32)
        self.entity_dof_start = np.zeros(n_entities, dtype=np.int32)

        # Joint to qpos and DOF mapping (per entity)
        self.joint_qpos_indices = np.zeros((n_entities, max_joints_per_entity), dtype=np.int32)
        self.joint_dof_indices = np.zeros((n_entities, max_joints_per_entity), dtype=np.int32)

        # DOF data (per entity, per environment)
        self.ref_dof_prev = np.zeros((n_entities, n_envs, max_dofs_per_entity), dtype=gs.np_float)
        self.qpos_current = np.zeros((n_entities, n_envs, max_dofs_per_entity), dtype=gs.np_float)
        self.qpos_new = np.zeros((n_entities, n_envs, max_dofs_per_entity), dtype=gs.np_float)

        # Joint data (per entity, per environment)
        self.delta_theta_tilde = np.zeros((n_entities, n_envs, max_joints_per_entity), dtype=gs.np_float)
        self.delta_theta_ipc = np.zeros((n_entities, n_envs, max_joints_per_entity), dtype=gs.np_float)

        # Mass matrix (per entity, flattened column-major)
        self.mass_matrix = np.zeros((n_entities, max_joints_per_entity * max_joints_per_entity), dtype=gs.np_float)

        # Previous timestep link transforms for ref_dof_prev computation
        # {(entity_idx, joint_idx, env_idx): transform_matrix_4x4}
        self.prev_link_transforms = {}
