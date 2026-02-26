"""
Data classes for IPC coupler.

Numpy-backed data structures used for IPC coupling computations.
"""

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

import genesis as gs


class ABDLinkEntry(NamedTuple):
    """Per-link, per-env ABD state retrieved from IPC after advance."""

    transform: np.ndarray  # (4, 4) IPC transform
    aim_transform: np.ndarray  # (4, 4) Genesis stored transform
    velocity: np.ndarray | None = None  # (4, 4) velocity matrix, optional


class ContactForceEntry(NamedTuple):
    """Per-link, per-env contact force/torque from IPC."""

    force: np.ndarray  # (3,) force vector
    torque: np.ndarray  # (3,) torque vector


@dataclass
class ArticulatedEntityData:
    """Typed container for per-entity articulation coupling data."""

    entity: object
    revolute_joints: list
    prismatic_joints: list
    joint_geo_slots_by_env: dict
    articulation_geos_by_env: dict
    articulation_slots_by_env: dict
    articulation_objects_by_env: dict
    n_joints: int
    ref_dof_prev: np.ndarray
    delta_theta_tilde: np.ndarray
    delta_theta: np.ndarray
    joint_qpos_indices: list
    joint_dof_indices: list
    mass_matrix: np.ndarray
    has_free_base: bool
    base_link_idx: int
    n_dofs_actual: int = 0


@dataclass
class ForceBatch:
    """Batched forces/torques for a single environment, to be applied to Genesis rigid links."""

    link_indices: list = field(default_factory=list)
    forces: list = field(default_factory=list)
    torques: list = field(default_factory=list)


class IPCCouplingData:
    """Pre-allocated arrays for coupling force computation, sized exactly at build time."""

    def __init__(self, n):
        self.link_indices = np.empty(n, dtype=gs.np_int)
        self.env_indices = np.empty(n, dtype=gs.np_int)
        self.ipc_transforms = np.empty((n, 4, 4), dtype=gs.np_float)
        self.aim_transforms = np.empty((n, 4, 4), dtype=gs.np_float)
        self.link_masses = np.empty(n, dtype=gs.np_float)
        self.inertia_tensors = np.empty((n, 3, 3), dtype=gs.np_float)
        self.out_forces = np.empty((n, 3), dtype=gs.np_float)
        self.out_torques = np.empty((n, 3), dtype=gs.np_float)


class ArticulationData:
    """Numpy-backed data for joint articulation coupling."""

    def __init__(self, n_entities, max_dofs_per_entity, max_joints_per_entity, n_envs):
        # Entity-level metadata
        self.n_entities = n_entities
        self.entity_indices = np.zeros(n_entities, dtype=gs.np_int)
        self.entity_n_dofs = np.zeros(n_entities, dtype=gs.np_int)
        self.entity_n_joints = np.zeros(n_entities, dtype=gs.np_int)
        self.entity_dof_start = np.zeros(n_entities, dtype=gs.np_int)

        # Joint to qpos and DOF mapping (per entity)
        self.joint_qpos_indices = np.zeros((n_entities, max_joints_per_entity), dtype=gs.np_int)
        self.joint_dof_indices = np.zeros((n_entities, max_joints_per_entity), dtype=gs.np_int)

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
