"""Engine of the simulator, which steps the legacy solvers and coupler through Systems."""

import functools
from typing import TYPE_CHECKING

from genesis.engine.core import Engine, Find, Pipeline, Require, System, register_system

from .collider_system import ColliderSystem
from .constraint_solver_system import ConstraintSolverSystem
from .rigid_solver_system import RigidSolverSystem
from .solver_systems import (
    CouplerSystem,
    FEMSolverSystem,
    KinematicSolverSystem,
    MPMSolverSystem,
    PBDSolverSystem,
    SFSolverSystem,
    SPHSolverSystem,
    ToolSolverSystem,
)
from .substepper import Substepper

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


class SimulatorEngine(Engine):
    """The engine of one simulator, which runs the substep phases of its solvers and of its coupler.

    The simulator keeps the step loop, the clock and the checkpoints, and launches one pipeline per entry point.
    """

    substepper = Require(Substepper)
    rigid_solver_system = Find(RigidSolverSystem)

    def build(self):
        super().build()
        self.input_pipeline = Pipeline(self.substepper.process_input)
        self.substep_pipeline = Pipeline(self.substepper.substep)
        # A scene simulating rigid entities alone advances the rigid solver directly, skipping the inputs, which only
        # feed the differentiable tape, and the coupler, which has nothing to exchange
        self.rigid_substep_pipeline = None
        if self.rigid_solver_system is not None:
            self.rigid_substep_pipeline = Pipeline(self.rigid_solver_system.substep)


def create_active_solver_system(system_cls: type[System], scene: "Scene") -> System | None:
    """Create the System wrapping one solver of the scene, whenever that solver simulates an entity."""
    system = system_cls(scene)
    return system if system.solver.is_active else None


def create_rigid_solver_system(scene: "Scene") -> RigidSolverSystem | None:
    """Create the manager of the rigid substep whenever the rigid solver simulates an entity, before which its data
    does not exist."""
    return RigidSolverSystem(scene) if scene.sim.rigid_solver.is_active else None


def create_collider_system(scene: "Scene") -> ColliderSystem | None:
    """Create the collision detection of the rigid substep, unless the scene disables collisions."""
    rigid_solver = scene.sim.rigid_solver
    return ColliderSystem(scene) if rigid_solver.is_active and rigid_solver._enable_collision else None


def create_constraint_solver_system(scene: "Scene") -> ConstraintSolverSystem | None:
    """Create the constraint solve of the rigid substep, unless the scene disables constraints."""
    rigid_solver = scene.sim.rigid_solver
    return ConstraintSolverSystem(scene) if rigid_solver.is_active and not rigid_solver._disable_constraint else None


# The registration order fixes the order the systems are added in, and the ranks fix the order their phases run in
register_system(SimulatorEngine)(functools.partial(create_active_solver_system, ToolSolverSystem))
for create_fun in (create_rigid_solver_system, create_collider_system, create_constraint_solver_system):
    register_system(SimulatorEngine)(create_fun)
for solver_system_cls in (
    KinematicSolverSystem,
    MPMSolverSystem,
    SPHSolverSystem,
    PBDSolverSystem,
    FEMSolverSystem,
    SFSolverSystem,
):
    register_system(SimulatorEngine)(functools.partial(create_active_solver_system, solver_system_cls))
register_system(SimulatorEngine)(CouplerSystem)
