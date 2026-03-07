"""
Data classes for IPC coupler.
"""

from enum import IntEnum
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from uipc.geometry import GeometrySlot

import genesis as gs

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidLink


class COUPLING_TYPE(IntEnum):
    TWO_WAY_SOFT_CONSTRAINT = 0
    EXTERNAL_ARTICULATION = 1
    IPC_ONLY = 2
    NONE = 3


@dataclass
class ABDLinkData:
    """Per-link ABD data across all envs."""

    # Build-time (set in _add_rigid_entities_to_ipc)
    slots: list[GeometrySlot]  # per env

    # Per-step inputs (populated by _store_gs_rigid_states)
    aim_transforms: np.ndarray | None = None  # (B, 4, 4)

    # Per-step outputs (populated by _retrieve_ipc_rigid_states); only for coupling links
    ipc_transforms: np.ndarray | None = None  # (B, 4, 4)
    ipc_velocities: np.ndarray | None = None  # (B, 4, 4)


@dataclass
class ArticulatedEntityData:
    """Typed container for per-entity articulation coupling data."""

    # Topology (set at build time, ext-art always has fixed base)
    slots: list[GeometrySlot]
    q_slice: slice  # slice into global qpos array
    dof_slice: slice  # slice into global dofs array
    joints_child_link: list["RigidLink"]
    joints_qs_idx_local: list[int]

    # Per-step inputs (populated by _store_gs_rigid_states)
    delta_theta_tilde: np.ndarray | None = None  # (B, n_joints)
    prev_qpos: np.ndarray | None = None  # (B, n_qs)
    mass_matrix: np.ndarray | None = None  # (B, n_dofs, n_dofs)

    # Per-step outputs (populated by _post_advance_external_articulation)
    ipc_qpos: np.ndarray | None = None  # (B, n_qs)
