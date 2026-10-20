"""Engine owning and composing the Rigid solver of a simulator."""

from genesis.engine.core import Engine, Pipeline


class SimulatorEngine(Engine):
    """The Engine owning Rigid's active or padded inactive data in every simulator.

    Each System registers the function that creates it for this Engine from its own module (see register_system). The
    simulator keeps the step loop, clock and checkpoints; mixed/coupled scenes enter the Systems through legacy APIs.
    """

    def build(self):
        super().build()
        # The Systems import this module to register, so they are imported here rather than declared on the class
        from .initializer import Initializer
        from .rigid_solver_system import RigidSolverSystem

        self.init_pipeline = Pipeline(self.require(Initializer).initialize)
        # A scene simulating rigid entities alone advances the rigid solver directly, skipping the inputs, which only
        # feed the differentiable tape, and the coupler, which has nothing to exchange
        self.rigid_substep_pipeline = Pipeline(self.require(RigidSolverSystem).substep)
