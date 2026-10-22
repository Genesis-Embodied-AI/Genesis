"""Owner of rigid island partition and hibernation data shared by dynamics, contact and constraints."""

from collections.abc import Iterator

import numpy as np
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.core import Data, InlineAction, Require, System, register_system
from genesis.engine.solvers.rigid.abd.accessor import (
    kernel_wake_up_entities_by_dofs,
    kernel_wake_up_entities_by_links,
    kernel_wake_up_entities_by_qs,
    kernel_wakeup_links_island,
)
from genesis.engine.solvers.rigid.constraint.island import func_substep_partition_islands
from genesis.engine.solvers.rigid.substep import func_substep_hibernate_settled_islands
from genesis.utils.misc import qd_to_numpy

from .rigid_dynamics import RigidDynamics
from ..simulator_engine import SimulatorEngine


class RigidIsland(System):
    """Own partition data used independently of whether contact or constraint solving is enabled.

    The partition groups the kinematic trees into islands, the components of the trees under the edges the edge
    sources join (see on_partition_edges), the equality constraints and the hibernation chains. An island at rest falls
    asleep after the solve (see settle), and the bodies of a sleeping island an awake one reaches wake in the partition.
    """

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

    rigid_dynamics = Require(RigidDynamics)

    def __init__(self) -> None:
        super().__init__()
        self.data = self.Data()

    # -------------------------------- Actions --------------------------------

    @System.action(kind="host")
    def initialize(self):
        return self.initialize_data

    @System.action(kind="stage")
    def partition(self):
        rigid = self.rigid_dynamics
        return func_substep_partition_islands, self.data, rigid.data, rigid.rigid_config, self.partition_edges

    @System.action(kind="stage")
    def settle(self):
        rigid = self.rigid_dynamics
        return func_substep_hibernate_settled_islands, self.data, rigid.data, rigid.rigid_config

    # -------------------------------- Action collections --------------------------------

    @System.action_collection(kind="inline", call_args=("i_b", "tid"))
    def partition_edges(self):
        """The edge sources of the partition, see on_partition_edges."""

    # -------------------------------- Protocols --------------------------------

    @System.protocol
    def on_partition_edges(self, edges: InlineAction) -> None:
        """Join the trees an edge source couples into the islands of the partition.

        Signature  edges(*bound_args, i_b, tid).
        Invoked    inside the partition loop of env i_b, in the one thread of the serial build (tid 0), or in every
                   lane tid of the 32 of the block of the cooperative build, once per hooking round, before the barrier
                   that ends the round.
        Must       pass every edge of env i_b to func_union_link_trees, the lanes of a block striding over the edges.
        Must not   write the partition otherwise, or read it beyond the roots func_union_link_trees resolves.
        """
        self.partition_edges += edges

    # -------------------------------- Public get/set API --------------------------------

    def iter_data(self) -> Iterator[array_class.DataItem | array_class.ConfigItem]:
        yield from array_class.iter_data(self.data, "island")
        yield from array_class.iter_data(self.data, "qd_n_equalities", array_class.DataKind.STATE)

    def wake_entities_by_links(self, links_idx: torch.Tensor, envs_idx: torch.Tensor) -> None:
        """Wake the hibernated entities holding these links, so that a state written to them is not dropped."""
        rigid = self.rigid_dynamics
        if rigid.rigid_config.use_hibernation:
            kernel_wake_up_entities_by_links(
                links_idx,
                envs_idx,
                rigid.data.dyn_state,
                self.data.island,
                rigid.data.dyn_info,
                rigid.data.rigid_info,
                rigid.rigid_config,
            )

    def wake_entities_by_dofs(self, dofs_idx: torch.Tensor, envs_idx: torch.Tensor) -> None:
        """Wake the hibernated entities owning these dofs: forward dynamics and integration act only on awake dofs, so a
        state written to a sleeping body would be dropped until something else woke it."""
        rigid = self.rigid_dynamics
        if rigid.rigid_config.use_hibernation:
            kernel_wake_up_entities_by_dofs(
                dofs_idx,
                envs_idx,
                rigid.data.dyn_state,
                self.data.island,
                rigid.data.dyn_info,
                rigid.data.rigid_info,
                rigid.rigid_config,
            )

    def wake_entities_by_qs(self, qs_idx: torch.Tensor, envs_idx: torch.Tensor) -> None:
        """Wake the hibernated entities owning these generalized coordinates."""
        rigid = self.rigid_dynamics
        if rigid.rigid_config.use_hibernation:
            kernel_wake_up_entities_by_qs(
                qs_idx,
                envs_idx,
                rigid.data.dyn_state,
                self.data.island,
                rigid.data.dyn_info,
                rigid.data.rigid_info,
                rigid.rigid_config,
            )

    def wake_link_islands(self, links_idx: torch.Tensor, envs_idx: torch.Tensor) -> None:
        """Wake the island of every hibernated link among these."""
        rigid = self.rigid_dynamics
        kernel_wakeup_links_island(
            links_idx,
            envs_idx,
            rigid.data.dyn_state,
            self.data.island,
            rigid.data.dyn_info,
            rigid.data.rigid_info,
            rigid.rigid_config,
        )

    # -------------------------------- Implementation --------------------------------

    def initialize_data(self) -> None:
        rigid = self.rigid_dynamics
        island_state = array_class.get_island_state(
            batch_size=rigid.batch_size,
            n_trees=rigid.n_trees_,
            n_links_actual=rigid.n_links,
            n_dofs_actual=rigid.n_dofs,
            rigid_config=rigid.rigid_config,
            use_hibernation=rigid.use_hibernation,
        )
        qd_n_equalities = array_class.V(dtype=gs.qd_int, shape=(rigid.batch_size,))
        qd_n_equalities.from_numpy(np.full((rigid.batch_size,), rigid.n_equalities, dtype=gs.np_int))
        self.data.island = island_state
        self.data.qd_n_equalities = qd_n_equalities

        island_state.links_island_idx.fill(-1)
        if rigid.use_hibernation:
            island_state.hibernated_next_link.fill(-1)
            if rigid.rigid_config.is_single_island:
                links_idx = np.flatnonzero(qd_to_numpy(rigid.data.rigid_info.links_tree_idx) == 0)
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
def create_rigid_island(engine: SimulatorEngine) -> RigidIsland | None:
    from genesis.engine.entities import RigidEntity

    return RigidIsland() if any(isinstance(entity, RigidEntity) for entity in engine.scene.entities) else None
