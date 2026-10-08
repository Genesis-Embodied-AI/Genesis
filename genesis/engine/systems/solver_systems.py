"""Systems wrapping the solvers other than the rigid one and the coupler, whose phases run through the Substepper.

Each System keeps its legacy object, which still owns its data and runs its own kernels, and hands the phases of that
object to the Substepper as host Actions.
"""

from typing import TYPE_CHECKING

from genesis.engine.core import Require, System

from .substepper import Substepper

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


class ToolSolverSystem(System):
    """System wrapping the tool solver of the scene."""

    substepper = Require(Substepper)

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)
        self.solver = scene.sim.tool_solver

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


class KinematicSolverSystem(System):
    """System wrapping the kinematic solver of the scene."""

    substepper = Require(Substepper)

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)
        self.solver = scene.sim.kinematic_solver

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


class MPMSolverSystem(System):
    """System wrapping the MPM solver of the scene."""

    substepper = Require(Substepper)

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

    def build(self):
        self.substepper.on_substep(
            process_input=self.process_input,
            pre_coupling=self.pre_coupling,
            post_coupling=self.post_coupling,
            rank=self._scene.sim.solvers.index(self.solver),
        )


class SPHSolverSystem(System):
    """System wrapping the SPH solver of the scene."""

    substepper = Require(Substepper)

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)
        self.solver = scene.sim.sph_solver

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


class PBDSolverSystem(System):
    """System wrapping the PBD solver of the scene."""

    substepper = Require(Substepper)

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)
        self.solver = scene.sim.pbd_solver

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


class SFSolverSystem(System):
    """System wrapping the stable fluid solver of the scene."""

    substepper = Require(Substepper)

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)
        self.solver = scene.sim.sf_solver

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


def preprocess_coupler(coupler, f):
    """Prepare the coupling of substep f."""
    coupler.preprocess(f)


def couple_coupler(coupler, f):
    """Exchange state between the solvers within substep f."""
    coupler.couple(f)


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
