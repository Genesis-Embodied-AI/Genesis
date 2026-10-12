"""System wrapping the coupler of the scene, whose phases run through the Substepper."""

from typing import TYPE_CHECKING

from genesis.engine.core import Require, System, register_system

from .simulator_engine import SimulatorEngine
from .substepper import Substepper

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


def preprocess_coupler(coupler, f):
    """Prepare the coupling of substep f."""
    coupler.preprocess(f)


def couple_coupler(coupler, f):
    """Exchange state between the solvers within substep f."""
    coupler.couple(f)


@register_system(SimulatorEngine)
class CouplerSystem(System):
    """System wrapping the coupler of the scene, whichever coupler the scene options select."""

    substepper = Require(Substepper)

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

    def build(self):
        self.substepper.on_couple(preprocess=self.preprocess, couple=self.couple)
