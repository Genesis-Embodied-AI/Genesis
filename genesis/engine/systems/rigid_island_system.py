"""Owner of rigid island partition and hibernation data shared by dynamics, contact and constraints."""

from collections.abc import Iterator

import numpy as np
import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.core import Data, Require, System, register_system
from genesis.utils.misc import qd_to_numpy

from .rigid_solver_system import RigidSolverSystem
from ..simulator_engine import SimulatorEngine


class RigidIslandSystem(System):
    """Own partition data used independently of whether contact or constraint solving is enabled."""

    # -------------------------------- Data --------------------------------

    @qd.data_oriented
    class Data(Data):
        """Rigid topology partition, hibernation state and dynamic equality count.

        ``@qd.data_oriented`` is a QD discovery workaround: ``GlobalDataContext`` can expose this root only in a form
        QD traverses. Remove it when QD derives ndarray parameters through ordinary non-data-oriented objects.
        """

        island: array_class.IslandState
        qd_n_equalities: qd.Tensor

    data: Data

    # -------------------------------- Dependencies --------------------------------

    rigid_solver_system = Require(RigidSolverSystem)

    def __init__(self) -> None:
        super().__init__()
        self.data = self.Data()

    def build(self) -> None:
        from .constraint_solver_system import ConstraintSolverSystem
        from .rigid_rigid_contact_system import RigidRigidContactSystem

        self.contact_system = self.find(RigidRigidContactSystem)
        self.constraint_system = self.find(ConstraintSolverSystem)

    # -------------------------------- Actions --------------------------------

    @System.action(kind="host")
    def initialize(self):
        return self.initialize_data

    # -------------------------------- Public get/set API --------------------------------

    def iter_data(self) -> Iterator[array_class.DataItem | array_class.ConfigItem]:
        yield from array_class.iter_data(self.data, "island")
        yield from array_class.iter_data(self.data, "qd_n_equalities", array_class.DataKind.STATE)

    # -------------------------------- Implementation --------------------------------

    def initialize_data(self) -> None:
        rigid = self.rigid_solver_system
        max_contacts = 0
        if self.constraint_system is not None and self.contact_system is not None:
            max_contacts = self.contact_system.data.collider_info.max_contacts[None]
        island_state = array_class.get_island_state(
            batch_size=rigid._B,
            n_trees=rigid.n_trees_,
            n_links_actual=rigid.n_links,
            n_dofs_actual=rigid.n_dofs,
            rigid_config=rigid.rigid_config,
            use_hibernation=rigid._use_hibernation,
            max_contacts=max_contacts,
        )
        qd_n_equalities = array_class.V(dtype=gs.qd_int, shape=(rigid._B,))
        qd_n_equalities.from_numpy(np.full((rigid._B,), rigid.n_equalities, dtype=gs.np_int))
        self.data.island = island_state
        self.data.qd_n_equalities = qd_n_equalities

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
def create_rigid_island_system(engine: SimulatorEngine) -> RigidIslandSystem | None:
    from genesis.engine.entities import RigidEntity

    return (
        RigidIslandSystem()
        if any(isinstance(entity, RigidEntity) for entity in engine.scene.entities)
        else None
    )
