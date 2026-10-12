"""Engine of the simulator, which steps the legacy solvers and coupler through Systems."""

from genesis.engine.core import Engine


class SimulatorEngine(Engine):
    """The engine of one simulator, which runs the substep phases of its solvers and of its coupler.

    Each System registers the function that creates it for this Engine from its own module (see register_system). The
    simulator keeps the step loop, the clock and the checkpoints, and launches the pipelines of the Substepper.
    """
