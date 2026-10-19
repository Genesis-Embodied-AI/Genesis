"""System of the constraint solve of the rigid solver, which spans all three graphs of every rigid substep."""

from typing import TYPE_CHECKING

import quadrants as qd

from genesis.engine.core import Data, Require, System, register_system
from genesis.engine.solvers.rigid.constraint import ConstraintSolver
from genesis.engine.solvers.rigid.constraint.solver import (
    func_add_equality_constraints,
    func_add_inequality_constraints,
    func_resolve_post,
    func_solve_body,
)

from .rigid_solver_system import RigidSolverSystem
from .simulator_engine import SimulatorEngine

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


@qd.data_oriented
class ConstraintSolverData(Data):
    """Device state owned by the rigid constraint solve."""

    def __init__(self, constraint_solver: ConstraintSolver) -> None:
        self.constraint_state = constraint_solver.constraint_state


class ConstraintSolverSystem(System):
    """Own the rigid constraint data and run the solve when constraints are enabled.

    It assembles the equality constraints after the forward dynamics and the inequality constraints after collision
    detection, solves them on the host, then turns the solved forces into accelerations and contact forces. The System
    exists with a padded data layout even when constraints are disabled because rigid dynamics use its scratch arrays.
    """

    rigid_solver_system = Require(RigidSolverSystem)

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)
        rigid_solver = scene.sim.rigid_solver
        self.enabled = rigid_solver.is_active and not rigid_solver._disable_constraint

    @System.action(kind="host")
    def initialize(self):
        return self.initialize_data

    @System.action(kind="host")
    def prepare(self):
        return self.clear_equality_info_cache

    @System.action(kind="stage")
    def assemble_equalities(self):
        return func_substep_add_equality_constraints, self

    @System.action(kind="stage")
    def assemble_inequalities(self):
        return func_substep_add_inequality_constraints, self

    @System.action(kind="host")
    def solve(self):
        return self.solve_constraints

    @System.action(kind="stage")
    def update_forces(self):
        return func_substep_resolve_post, self

    def build(self):
        # Imported here to avoid the contact/constraint module cycle.
        from .rigid_rigid_contact_system import RigidRigidContactSystem

        self.contact_system = self.require(RigidRigidContactSystem)
        if self.enabled:
            self.rigid_solver_system.on_constrain(
                prepare=self.prepare,
                assemble_equalities=self.assemble_equalities,
                assemble_inequalities=self.assemble_inequalities,
                solve=self.solve,
                update_forces=self.update_forces,
            )

    def initialize_data(self) -> None:
        """Allocate constraint data from the rigid model and the contact layout."""
        rigid_solver_system = self.rigid_solver_system
        self.constraint_solver = ConstraintSolver(rigid_solver_system, self.contact_system.collider_data)
        self.data = ConstraintSolverData(self.constraint_solver)
        self.collider_config = self.contact_system.collider_data.collider_config
        self.n_iterations = self.constraint_solver._n_iterations
        self.noslip = rigid_solver_system._options.noslip_iterations > 0

    def clear_equality_info_cache(self) -> None:
        """Drop the equality constraints read back during the previous substep, which the assembly is about to
        replace."""
        self.constraint_solver._eq_const_info_cache.clear()

    def solve_constraints(self) -> None:
        """Solve the constraints of a substep, through the solver arm func_solve_body selects at run time."""
        rigid_data = self.rigid_solver_system.data
        func_solve_body(
            rigid_data.dyn_state,
            self.data.constraint_state,
            rigid_data.dyn_info,
            rigid_data.rigid_info,
            self.rigid_solver_system.rigid_config,
            self.n_iterations,
        )


@register_system(SimulatorEngine)
def create_constraint_solver_system(scene: "Scene") -> ConstraintSolverSystem:
    """Create the owner of the scene's rigid constraint data."""
    return ConstraintSolverSystem(scene)


@qd.func(requires_top_level=True)
def func_substep_add_equality_constraints(
    constraint_solver_system: qd.template(),
    contact_data: qd.template(),
    constraint_data: qd.template(),
):
    """Assemble the equality constraints of a substep."""
    rigid_data = constraint_solver_system.rigid_solver_system.data
    func_add_equality_constraints(
        rigid_data.dyn_state,
        contact_data.collider_state,
        constraint_data.constraint_state,
        rigid_data.dyn_info,
        rigid_data.rigid_info,
        constraint_solver_system.rigid_solver_system.rigid_config,
    )


@qd.func(requires_top_level=True)
def func_substep_add_inequality_constraints(
    constraint_solver_system: qd.template(),
    contact_data: qd.template(),
    constraint_data: qd.template(),
):
    """Assemble the inequality constraints of a substep, from its detected contacts among others."""
    rigid_data = constraint_solver_system.rigid_solver_system.data
    func_add_inequality_constraints(
        rigid_data.dyn_state,
        contact_data.collider_state,
        constraint_data.constraint_state,
        rigid_data.dyn_info,
        rigid_data.rigid_info,
        constraint_solver_system.rigid_solver_system.rigid_config,
        constraint_solver_system.collider_config,
    )


@qd.func(requires_top_level=True)
def func_substep_resolve_post(
    constraint_solver_system: qd.template(),
    contact_data: qd.template(),
    constraint_data: qd.template(),
):
    """Update the accelerations and the contact forces of a substep from its solved constraint forces (see
    func_resolve_post for noslip)."""
    rigid_data = constraint_solver_system.rigid_solver_system.data
    func_resolve_post(
        rigid_data.dyn_state,
        contact_data.collider_state,
        constraint_data.constraint_state,
        rigid_data.dyn_info,
        rigid_data.rigid_info,
        constraint_solver_system.rigid_solver_system.rigid_config,
        constraint_solver_system.noslip,
        rigid_data.errno,
    )
