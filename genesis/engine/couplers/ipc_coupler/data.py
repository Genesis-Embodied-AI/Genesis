"""
Data classes for IPC coupler.
"""

from enum import IntEnum
from dataclasses import dataclass
from typing import NamedTuple, TYPE_CHECKING

import numpy as np

from uipc.core import Object
from uipc.geometry import Geometry, GeometrySlot

import genesis as gs

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidLink


class COUPLING_TYPE(IntEnum):
    TWO_WAY_SOFT_CONSTRAINT = 0
    EXTERNAL_ARTICULATION = 1
    IPC_ONLY = 2
    NONE = 3


class ABDLinkEntry(NamedTuple):
    """Per-link, per-env ABD state retrieved from IPC after advance."""

    transform: np.ndarray  # (4, 4) IPC transform
    aim_transform: np.ndarray  # (4, 4) Genesis stored transform
    velocity: np.ndarray  # (4, 4) velocity matrix


@dataclass
class ArticulatedEntityData:
    """Typed container for per-entity articulation coupling data."""

    joints_child_link: list["RigidLink"]
    joints_q_idx_local: list[int]

    joints_geom_slot_by_env: list[list[GeometrySlot]]
    articulation_geoms_by_env: list[Geometry]
    articulation_slots_by_env: list[GeometrySlot]
    articulation_objects_by_env: list[Object]

    ref_dof_prev: np.ndarray
    qpos_stored: np.ndarray
    qpos_current: np.ndarray
    qpos_new: np.ndarray
    delta_theta_tilde: np.ndarray
    delta_theta_ipc: np.ndarray

    # Previous timestep link transforms for ref_dof_prev computation {(joint, env_idx): transform_matrix_4x4}
    prev_links_transform: list[list[np.ndarray | None]]


class IPCCouplingData:
    """Pre-allocated arrays for coupling force computation."""

    def __init__(self, coupling_entries: list[tuple[int, "RigidLink", int]]):
        links = list(dict.fromkeys(link for _, link, _ in coupling_entries))

        n_links = len(links)
        n_envs = len(set([env_idx for _, _, env_idx in coupling_entries]))
        assert len(coupling_entries) == n_links * n_envs

        self.links_idx = [link.idx for link in links]
        self.links_mass = np.array([link.inertial_mass for link in links], dtype=gs.np_float)
        if links:
            self.links_inertia_i = np.stack([link.inertial_i for link in links], axis=0, dtype=gs.np_float)
        else:
            self.links_inertia_i = np.empty((0, 0, 3, 3), dtype=gs.np_float)

        self.ipc_transforms = np.empty((n_envs, n_links, 4, 4), dtype=gs.np_float)
        self.aim_transforms = np.empty((n_envs, n_links, 4, 4), dtype=gs.np_float)
        self.out_forces = np.empty((n_envs, n_links, 3), dtype=gs.np_float)
        self.out_torques = np.empty((n_envs, n_links, 3), dtype=gs.np_float)
