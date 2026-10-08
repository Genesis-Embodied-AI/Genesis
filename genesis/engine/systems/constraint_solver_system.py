"""System of the constraint solve of the rigid solver, which spans all three graphs of every rigid substep."""

from typing import TYPE_CHECKING

import quadrants as qd

from genesis.engine.core import Require, System
from genesis.engine.solvers.rigid.constraint.solver import (
    func_add_equality_constraints,
    func_add_inequality_constraints,
    func_resolve_post,
    func_solve_body,
)

from .rigid_solver_system import RigidSolverData, RigidSolverSystem

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


class ConstraintSolverSystem(System):
    """System running the constraint solve of the rigid solver, which the scene adds unless it disables constraints.

    The constraint solver of the rigid solver keeps owning its data, which RigidSolverData exposes to the substep
    graphs. It assembles the equality constraints after the forward dynamics and the inequality constraints after the
    collision detection, solves them on the host, then turns the solved forces into accelerations and contact forces.
    """

    rigid_solver_system = Require(RigidSolverSystem)

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)
        self.constraint_solver = scene.sim.rigid_solver.constraint_solver

    @System.action(kind="host")
    def prepare(self):
        return self.clear_equality_info_cache

    @System.action(kind="stage")
    def assemble_equalities(self):
        return (
            func_substep_add_equality_constraints,
            self.rigid_solver_system.data,
            self.rigid_solver_system.rigid_config,
        )

    @System.action(kind="stage")
    def assemble_inequalities(self):
        return (
            func_substep_add_inequality_constraints,
            self.rigid_solver_system.data,
            self.rigid_solver_system.rigid_config,
            self.rigid_solver_system.solver.collider.collider_config,
        )

    @System.action(kind="host")
    def solve(self):
        return (
            solve_substep_constraints,
            self.rigid_solver_system.data,
            self.rigid_solver_system.rigid_config,
            self.constraint_solver._n_iterations,
        )

    @System.action(kind="stage")
    def update_forces(self):
        return (
            func_substep_resolve_post,
            self.rigid_solver_system.data,
            self.rigid_solver_system.rigid_config,
            self.rigid_solver_system.solver._options.noslip_iterations > 0,
        )

    def build(self):
        self.rigid_solver_system.on_constrain(
            prepare=self.prepare,
            assemble_equalities=self.assemble_equalities,
            assemble_inequalities=self.assemble_inequalities,
            solve=self.solve,
            update_forces=self.update_forces,
        )

    def clear_equality_info_cache(self) -> None:
        """Drop the equality constraints read back during the previous substep, which the assembly is about to
        replace."""
        self.constraint_solver._eq_const_info_cache.clear()


@qd.func(requires_top_level=True)
def func_substep_add_equality_constraints(rigid_data: qd.template(), rigid_config: qd.template()):
    """Assemble the equality constraints of a substep."""
    func_add_equality_constraints(
        rigid_data.dyn_state,
        rigid_data.collider_state,
        rigid_data.constraint_state,
        rigid_data.dyn_info,
        rigid_data.rigid_info,
        rigid_config,
    )


@qd.func(requires_top_level=True)
def func_substep_add_inequality_constraints(
    rigid_data: qd.template(), rigid_config: qd.template(), collider_static_config: qd.template()
):
    """Assemble the inequality constraints of a substep, from its detected contacts among others."""
    func_add_inequality_constraints(
        rigid_data.dyn_state,
        rigid_data.collider_state,
        rigid_data.constraint_state,
        rigid_data.dyn_info,
        rigid_data.rigid_info,
        rigid_config,
        collider_static_config,
    )


def solve_substep_constraints(rigid_data: RigidSolverData, rigid_config, n_iterations: int) -> None:
    """Solve the constraints of a substep, through the solver arm func_solve_body selects at run time."""
    func_solve_body(
        rigid_data.dyn_state,
        rigid_data.constraint_state,
        rigid_data.dyn_info,
        rigid_data.rigid_info,
        rigid_config,
        n_iterations,
    )


@qd.func(requires_top_level=True)
def func_substep_resolve_post(rigid_data: qd.template(), rigid_config: qd.template(), noslip: qd.template()):
    """Update the accelerations and the contact forces of a substep from its solved constraint forces (see
    func_resolve_post for noslip)."""
    func_resolve_post(
        rigid_data.dyn_state,
        rigid_data.collider_state,
        rigid_data.constraint_state,
        rigid_data.dyn_info,
        rigid_data.rigid_info,
        rigid_config,
        noslip,
        rigid_data.errno,
    )
