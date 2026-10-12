"""System wrapping the coupler of the scene, whose phases the Substepper calls."""

from typing import TYPE_CHECKING

from genesis.engine.core import System, register_system

from .simulator_engine import SimulatorEngine

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


def preprocess_coupler(coupler, f):
    """Prepare the coupling of substep f."""
    coupler.preprocess(f)


def couple_coupler(coupler, f):
    """Exchange state between the solvers within substep f."""
    coupler.couple(f)


class CouplerSystem(System):
    """System wrapping the coupler of the scene, whichever coupler the scene options select."""

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)
        self.coupler = scene.sim.coupler

    # The couplers name the substep argument differently, so free functions give every coupler the `f` argument the
    # Substepper calls them with
    @System.action(kind="host")
    def preprocess(self):
        return preprocess_coupler, self.coupler

    @System.action(kind="host")
    def couple(self):
        return couple_coupler, self.coupler


@register_system(SimulatorEngine)
def create_coupler_system(scene: "Scene") -> CouplerSystem:
    """Create the System wrapping the coupler, which every scene stepped by the engine has."""
    return CouplerSystem(scene)
