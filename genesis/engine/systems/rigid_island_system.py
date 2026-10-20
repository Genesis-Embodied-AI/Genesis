"""Owner of rigid island partition and hibernation data shared by dynamics, contact and constraints."""

from typing import TYPE_CHECKING

import numpy as np

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.core import Data, Require, System, register_system
from genesis.utils.misc import qd_to_numpy

from .rigid_solver_system import RigidSolverSystem
from .simulator_engine import SimulatorEngine

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


@qd.data_oriented
class RigidIslandData(Data):
    """Rigid topology partition, hibernation state and dynamic equality count."""

    def __init__(self, island_state, qd_n_equalities) -> None:
        self.island = island_state
        self.qd_n_equalities = qd_n_equalities


class RigidIslandSystem(System):
    """Own partition data used independently of whether contact or constraint solving is enabled."""

    rigid_solver_system = Require(RigidSolverSystem)

    def build(self) -> None:
        from .constraint_solver_system import ConstraintSolverSystem
        from .rigid_rigid_contact_system import RigidRigidContactSystem

        self.contact_system = self.require(RigidRigidContactSystem)
        self.constraint_system = self.find(ConstraintSolverSystem)

    @System.action(kind="host")
    def initialize(self):
        return self.initialize_data

    def initialize_data(self) -> None:
        rigid = self.rigid_solver_system
        max_contacts = 0
        len_constraints = 1
        if self.constraint_system is not None:
            max_contacts = self.contact_system.collider_data.collider_info.max_contacts[None]
            len_constraints = (
                rigid.rigid_config.rows_per_contact * max_contacts
                + sum(joint.type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC) for joint in rigid.joints)
                + rigid.n_dofs
                + rigid.n_candidate_equalities_ * 6
            )
        island_state = array_class.get_island_state(rigid, max_contacts, max(1, len_constraints))
        qd_n_equalities = array_class.V(dtype=gs.qd_int, shape=(rigid._B,))
        qd_n_equalities.from_numpy(np.full((rigid._B,), rigid.n_equalities, dtype=gs.np_int))
        self.data = RigidIslandData(island_state, qd_n_equalities)

        island_state.links_island_idx.fill(-1)
        if rigid._use_hibernation:
            island_state.hibernated_next_link.fill(-1)
            if rigid.rigid_config.is_single_island:
                links_idx = np.flatnonzero(qd_to_numpy(rigid.rigid_info.links_tree_idx) == 0)
                link_id = qd_to_numpy(island_state.link_id)
                link_id[: len(links_idx)] = links_idx[:, None]
                island_state.link_id.from_numpy(link_id)
                links_island_idx = qd_to_numpy(island_state.links_island_idx)
                links_island_idx[links_idx] = 0
                island_state.links_island_idx.from_numpy(links_island_idx)
                for link_slice in (island_state.link_slices.n, island_state.link_slices.curr):
                    link_slice_np = qd_to_numpy(link_slice)
                    link_slice_np[0] = len(links_idx)
                    link_slice.from_numpy(link_slice_np)


@register_system(SimulatorEngine)
def create_rigid_island_system(scene: "Scene") -> RigidIslandSystem:
    return RigidIslandSystem(scene)
