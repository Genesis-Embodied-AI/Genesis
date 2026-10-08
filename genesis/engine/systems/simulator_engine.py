"""Engine of the simulator, which steps the legacy solvers and coupler through Systems."""

import functools

from genesis.engine.core import Engine, Find, Pipeline, Require, System, register_system

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
            self.rigid_substep_pipeline = Pipeline(self.rigid_solver_system.solver.substep)


def create_active_solver_system(system_cls: type[System], scene) -> System | None:
    """Create the System wrapping one solver of the scene, whenever that solver simulates an entity."""
    system = system_cls(scene)
    return system if system.solver.is_active else None


# The registration order fixes the order the systems are added in, and the ranks fix the order their phases run in
for solver_system_cls in (
    ToolSolverSystem,
    RigidSolverSystem,
    KinematicSolverSystem,
    MPMSolverSystem,
    SPHSolverSystem,
    PBDSolverSystem,
    FEMSolverSystem,
    SFSolverSystem,
):
    register_system(SimulatorEngine)(functools.partial(create_active_solver_system, solver_system_cls))
register_system(SimulatorEngine)(CouplerSystem)
