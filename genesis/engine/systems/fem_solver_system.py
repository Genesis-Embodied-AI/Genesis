"""System wrapping the FEM solver, whose phases run through the Substepper.

The FEM solver keeps owning its data and running its own kernels; the System hands its phases to the Substepper as host
Actions.
"""

from typing import TYPE_CHECKING

from genesis.engine.core import Require, System, register_system

from .simulator_engine import SimulatorEngine
from .substepper import Substepper

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


class FEMSolverSystem(System):
    """System wrapping the FEM solver of the scene."""

    substepper = Require(Substepper)

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)
        self.solver = scene.sim.fem_solver

    @System.action(kind="host")
    def process_input(self):
        return self.solver.process_input

    @System.action(kind="host")
    def pre_coupling(self):
        return self.solver.substep_pre_coupling

    @System.action(kind="host")
    def post_coupling(self):
        return self.solver.substep_post_coupling

    def build(self):
        self.substepper.on_substep(
            process_input=self.process_input,
            pre_coupling=self.pre_coupling,
            post_coupling=self.post_coupling,
            rank=self._scene.sim.solvers.index(self.solver),
        )


@register_system(SimulatorEngine)
def create_fem_solver_system(scene: "Scene") -> FEMSolverSystem | None:
    """Create the System wrapping the FEM solver whenever it simulates an entity."""
    return FEMSolverSystem(scene) if scene.sim.fem_solver.is_active else None
