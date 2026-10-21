"""Participant System of the rigid contact constraints, the rows of the contacts the rigid-rigid detection found."""

from collections.abc import Iterator

import quadrants as qd

import genesis.utils.array_class as array_class
from genesis.engine.core import Data, Require, System, register_system
from genesis.engine.solvers.rigid.constraint.backward import kernel_manual_add_collision_constraints_bw
from genesis.engine.solvers.rigid.constraint.solver import (
    func_substep_add_contact_rows,
    func_substep_update_contact_force,
)

from .rigid_constraint_system import RigidConstraintSystem, RowCategory
from .rigid_island_system import RigidIslandSystem
from .rigid_rigid_contact_system import RigidRigidContactSystem
from .rigid_solver_system import RigidSolverSystem
from ..simulator_engine import SimulatorEngine


class RigidContactConstraintSystem(System):
    """Append rows_per_contact contact rows per live contact in every env, and read the contact forces back from them.

    Before assembling, it lets the contact detection order the live contacts and, on the CPU skyline path, reorders
    the dofs of each island by their contact adjacency.
    """

    # -------------------------------- Data --------------------------------

    @qd.data_oriented
    class Data(Data):
        """Where the contact rows start in every env, and the scratch of the dof reorder of the CPU skyline path.

        ``@qd.data_oriented`` is a QD discovery workaround: ``GlobalDataContext`` can expose this root only in a form
        QD traverses. Remove it when QD derives ndarray parameters through ordinary non-data-oriented objects.
        """

        segment: array_class.ConstraintRowSegment
        reorder: array_class.IslandReorderState

    data: Data

    # -------------------------------- Dependencies --------------------------------

    rigid_solver_system = Require(RigidSolverSystem)

    island_system = Require(RigidIslandSystem)

    contact_system = Require(RigidRigidContactSystem)

    constraint_system = Require(RigidConstraintSystem)

    def __init__(self) -> None:
        super().__init__()
        self.data = self.Data()

    def build(self) -> None:
        self.constraint_system.on_rows(
            RowCategory.CONTACT,
            initialize=self.initialize,
            assemble=self.assemble,
            reverse=self.reverse,
            post_solve=self.post_solve,
        )

    # -------------------------------- Actions --------------------------------

    @System.action(kind="host")
    def initialize(self):
        return self.initialize_data

    @System.action(kind="stage")
    def assemble(self):
        rigid = self.rigid_solver_system
        return (
            func_substep_add_contact_rows,
            self.data,
            self.constraint_system.data,
            self.island_system.data,
            self.contact_system.data,
            rigid.data,
            rigid.rigid_config,
            self.contact_system.order_contacts,
        )

    @System.action(kind="stage")
    def post_solve(self):
        rigid = self.rigid_solver_system
        return (
            func_substep_update_contact_force,
            self.data,
            self.constraint_system.data,
            self.contact_system.data,
            rigid.data,
            rigid.rigid_config,
        )

    @System.action(kind="host")
    def reverse(self):
        return self.reverse_rows

    # -------------------------------- Public get/set API --------------------------------

    def iter_data(self) -> Iterator[array_class.DataItem | array_class.ConfigItem]:
        for attr in ("segment", "reorder"):
            for item in array_class.iter_data(self.data, attr):
                yield item._replace(name=f"contact_rows.{item.name}")

    # -------------------------------- Implementation --------------------------------

    def initialize_data(self) -> None:
        """Allocate the segment record and the reorder scratch, and reserve the rows of the most contacts the detection
        keeps, which are also the rows of the elliptic cone groups."""
        rigid = self.rigid_solver_system
        max_contacts = self.contact_system.get_max_contacts()
        self.data.segment = array_class.get_constraint_row_segment(batch_size=rigid._B)
        self.data.reorder = array_class.get_island_reorder_state(
            batch_size=rigid._B,
            n_trees=rigid.n_trees_,
            max_contacts=max_contacts,
            rigid_config=rigid.rigid_config,
        )
        self.constraint_system.reserve_rows(RowCategory.CONTACT, rigid.rigid_config.rows_per_contact * max_contacts)

    def reverse_rows(self) -> None:
        """Reverse the contact rows into the adjoints of the contacts, then the narrowphase that detected them."""
        rigid = self.rigid_solver_system
        self.contact_system.clear_backward_input_grads()
        kernel_manual_add_collision_constraints_bw(
            self.data.segment,
            rigid.dyn_state,
            self.contact_system.data.collider_state,
            self.constraint_system.data.constraint_state,
            rigid.dyn_info,
            rigid.rigid_info,
            rigid.rigid_config,
        )
        self.contact_system.backward_narrowphase()


@register_system(SimulatorEngine)
def create_rigid_contact_constraint_system(engine: SimulatorEngine) -> RigidContactConstraintSystem | None:
    """Create the contact rows when a Rigid scene solves constraints and detects rigid-rigid contacts."""
    from genesis.engine.entities import RigidEntity

    rigid_options = engine.scene.options.rigid
    has_rigid = any(isinstance(entity, RigidEntity) for entity in engine.scene.entities)
    return (
        RigidContactConstraintSystem()
        if has_rigid and not rigid_options.disable_constraint and rigid_options.enable_collision
        else None
    )
