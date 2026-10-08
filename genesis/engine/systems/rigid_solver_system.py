"""System wrapping the rigid solver, whose substep phases run through the Substepper."""

from typing import TYPE_CHECKING

from genesis.engine.core import Require, System

from .substepper import Substepper

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


class RigidSolverSystem(System):
    """System wrapping the rigid solver of the scene, which keeps owning its data and running its own kernels."""

    substepper = Require(Substepper)

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)
        self.solver = scene.sim.rigid_solver

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
