"""Participant System of the rigid dof frictionloss, the dry friction of the dofs."""

from collections.abc import Iterator
from typing import TypeAlias

from numpy.typing import ArrayLike
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.core import Data, Require, System, register_system
from genesis.engine.solvers.rigid.constraint.backward import kernel_manual_add_frictionloss_constraints_bw
from genesis.engine.solvers.rigid.constraint.solver import func_substep_add_frictionloss_rows
from genesis.utils.misc import qd_to_torch

from .rigid_constraint import RigidConstraint, RowCategory
from .rigid_dynamics import RigidDynamics
from ..simulator_engine import SimulatorEngine

EnvIndex: TypeAlias = ArrayLike | torch.Tensor | None


class RigidFrictionLossConstraint(System):
    """Append one Huber row per dof with frictionloss in every env, and own the query of their forces."""

    # -------------------------------- Data --------------------------------

    @qd.data_oriented
    class Data(Data):
        """Where the frictionloss rows start in every env.

        ``@qd.data_oriented`` is a QD discovery workaround: ``GlobalDataContext`` can expose this root only in a form
        QD traverses. Remove it when QD derives ndarray parameters through ordinary non-data-oriented objects.
        """

        segment: array_class.ConstraintRowSegment

    data: Data

    # -------------------------------- Dependencies --------------------------------

    rigid_dynamics = Require(RigidDynamics)

    rigid_constraint = Require(RigidConstraint)

    def __init__(self) -> None:
        super().__init__()
        self.data = self.Data()

    def build(self) -> None:
        self.rigid_constraint.on_rows(
            RowCategory.FRICTIONLOSS,
            initialize=self.initialize,
            assemble=self.assemble,
            reverse=self.reverse,
        )

    # -------------------------------- Actions --------------------------------

    @System.action(kind="host")
    def initialize(self):
        return self.initialize_data

    @System.action(kind="stage")
    def assemble(self):
        rigid = self.rigid_dynamics
        return (
            func_substep_add_frictionloss_rows,
            self.data,
            self.rigid_constraint.data,
            rigid.data,
            rigid.rigid_config,
        )

    @System.action(kind="host")
    def reverse(self):
        rigid = self.rigid_dynamics
        return (
            kernel_manual_add_frictionloss_constraints_bw,
            self.data.segment,
            rigid.dyn_state,
            self.rigid_constraint.data.constraint_state,
            rigid.dyn_info,
            rigid.rigid_info,
            rigid.rigid_config,
        )

    # -------------------------------- Public get/set API --------------------------------

    def iter_data(self) -> Iterator[array_class.DataItem | array_class.ConfigItem]:
        for item in array_class.iter_data(self.data, "segment"):
            yield item._replace(name=f"frictionloss_rows.{item.name}")

    def get_dofs_frictionloss_force(self, envs_idx: EnvIndex = None) -> torch.Tensor:
        """The solved frictionloss force of every dof in the given envs, zero for a dof without frictionloss.

        The rows of the dofs with frictionloss follow one another in dof order from the start of the segment.
        """
        rigid = self.rigid_dynamics
        if rigid._options.batch_dofs_info:
            frictionloss = qd_to_torch(rigid.data.dyn_info.dofs.frictionloss, envs_idx, transpose=True)
        else:
            frictionloss = qd_to_torch(rigid.data.dyn_info.dofs.frictionloss, transpose=True)
        efc_force = qd_to_torch(self.rigid_constraint.data.constraint_state.efc_force, envs_idx, transpose=True)
        row_start = qd_to_torch(self.data.segment.row_start, envs_idx)
        has_frictionloss = frictionloss > gs.EPS
        rank = torch.cumsum(has_frictionloss, dim=-1) - 1
        gather_idx = (row_start[:, None] + rank).clamp_(min=0)
        return torch.gather(efc_force, 1, gather_idx) * has_frictionloss

    # -------------------------------- Implementation --------------------------------

    def initialize_data(self) -> None:
        """Allocate the segment record and reserve one row per dof."""
        rigid = self.rigid_dynamics
        self.data.segment = array_class.get_constraint_row_segment(batch_size=rigid._B)
        self.rigid_constraint.reserve_rows(RowCategory.FRICTIONLOSS, rigid.n_dofs)


@register_system(SimulatorEngine)
def create_rigid_frictionloss_constraint(engine: SimulatorEngine) -> RigidFrictionLossConstraint | None:
    """Create the frictionloss rows whenever a Rigid scene solves constraints, since set_dofs_frictionloss may give any
    dof frictionloss at run time."""
    from genesis.engine.entities import RigidEntity

    has_rigid = any(isinstance(entity, RigidEntity) for entity in engine.scene.entities)
    return (
        RigidFrictionLossConstraint()
        if has_rigid and not engine.scene.options.rigid.disable_constraint
        else None
    )
