"""System wrapping the MPM solver, whose phases the Substepper calls.

The MPM solver keeps owning its data and running its own kernels; the System hands its phases to the Substepper as host
Actions.
"""

from typing import TYPE_CHECKING

from genesis.engine.core import System, register_system

from .simulator_engine import SimulatorEngine

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


class MPMSolverSystem(System):
    """System wrapping the MPM solver of the scene."""

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)
        self.solver = scene.sim.mpm_solver

    @System.action(kind="host")
    def process_input(self):
        return self.solver.process_input

    @System.action(kind="host")
    def pre_coupling(self):
        return self.solver.substep_pre_coupling

    @System.action(kind="host")
    def post_coupling(self):
        return self.solver.substep_post_coupling


@register_system(SimulatorEngine)
def create_mpm_solver_system(scene: "Scene") -> MPMSolverSystem | None:
    """Create the System wrapping the MPM solver whenever it simulates an entity."""
    return MPMSolverSystem(scene) if scene.sim.mpm_solver.is_active else None
