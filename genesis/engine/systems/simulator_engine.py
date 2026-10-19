"""Engine of a rigid-only simulator."""

from genesis.engine.core import Engine, Pipeline


class SimulatorEngine(Engine):
    """The engine used when Rigid is the simulator's only active solver.

    Each System registers the function that creates it for this Engine from its own module (see register_system). The
    simulator keeps the step loop, the clock and the checkpoints.
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
