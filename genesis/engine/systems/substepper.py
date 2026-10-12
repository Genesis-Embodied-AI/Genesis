"""Manager System that runs the phases of the solvers and of the coupler in the order of one simulator substep."""

from typing import TYPE_CHECKING

from genesis.engine.core import Find, Pipeline, Require, System, register_system

from .coupler_system import CouplerSystem
from .fem_solver_system import FEMSolverSystem
from .kinematic_solver_system import KinematicSolverSystem
from .mpm_solver_system import MPMSolverSystem
from .pbd_solver_system import PBDSolverSystem
from .rigid_solver_system import RigidSolverSystem
from .sf_solver_system import SFSolverSystem
from .simulator_engine import SimulatorEngine
from .sph_solver_system import SPHSolverSystem
from .tool_solver_system import ToolSolverSystem

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


class Substepper(System):
    """Manager of one substep: the preprocess of the coupler, every pre-coupling phase, the coupling, then every
    post-coupling phase.

    Every solver System it finds takes part in the input and in both phases around the coupling, in the order build()
    states.
    """

    tool = Find(ToolSolverSystem)
    rigid = Find(RigidSolverSystem)
    kinematic = Find(KinematicSolverSystem)
    mpm = Find(MPMSolverSystem)
    sph = Find(SPHSolverSystem)
    pbd = Find(PBDSolverSystem)
    fem = Find(FEMSolverSystem)
    sf = Find(SFSolverSystem)
    coupler = Require(CouplerSystem)

    def build(self):
        # The rigid solver advances before the MPM solver, whose p2g reads the rigid geoms and links that the rigid
        # substep moves. Within one phase the other solvers touch nothing of one another, so their order only keeps the
        # results reproducible.
        solver_systems = (self.tool, self.rigid, self.kinematic, self.mpm, self.sph, self.pbd, self.fem, self.sf)
        solver_systems = tuple(system for system in solver_systems if system is not None)
        # Actions are created only while the build window is open, so the substep runs the ones gathered here
        self.inputs = tuple(system.process_input for system in solver_systems)
        self.pre_couplings = tuple(system.pre_coupling for system in solver_systems)
        self.post_couplings = tuple(system.post_coupling for system in solver_systems)
        self.preprocess = self.coupler.preprocess
        self.couple = self.coupler.couple
        self.input_pipeline = Pipeline(self.process_input)
        self.substep_pipeline = Pipeline(self.substep)

    def process_input(self, in_backward: bool) -> None:
        """Apply the external commands of one step to every solver."""
        for action in self.inputs:
            action.invoke((in_backward,))

    # FIXME: the substep runs on the host, since the phases it calls launch their own kernels around host logic. The
    # DFSPH and implicit FEM iterations read their convergence back on the host every iteration, the rigid constraint
    # solve selects its kernels on the host and one of them is a graph itself, the SAP coupler iterates Newton, PCG and
    # line search on the host, and the phases of the legacy coupler are kernels rather than funcs. The substep becomes
    # one graph once every phase is a stage Action whose loops run as device-side graph loops, and once quadrants#966
    # lets such a graph use fastcache.
    def substep(self, f: int) -> None:
        """Advance every solver by substep f: the coupler preprocess, every pre-coupling phase, the coupling, then every
        post-coupling phase. A scene simulating rigid entities alone advances through RigidSolverSystem.substep_pipeline
        instead."""
        self.preprocess.invoke((f,))
        for action in self.pre_couplings:
            action.invoke((f,))
        self.couple.invoke((f,))
        for action in self.post_couplings:
            action.invoke((f,))


@register_system(SimulatorEngine)
def create_substepper(scene: "Scene") -> Substepper:
    """Create the manager of the substep, which every scene stepped by the engine has."""
    return Substepper(scene)
