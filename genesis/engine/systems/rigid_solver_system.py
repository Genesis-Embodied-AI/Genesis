"""Manager System of the rigid solver, which runs one substep as three graphs of stage Actions around a host solve."""

from typing import TYPE_CHECKING

import quadrants as qd

from genesis.engine.core import Data, HostAction, Require, StageAction, System
from genesis.engine.couplers import IPCCoupler, SAPCoupler
from genesis.engine.solvers.rigid.rigid_solver import func_step_1, func_step_2

from .substepper import Substepper

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver


# WORKAROUND: Quadrants discovers the tensors of the array_class structs reached from a kernel parameter only through a
# @qd.data_oriented object, so the structs the rigid solver owns are gathered here rather than bound one by one.
@qd.data_oriented
class RigidSolverData(Data):
    """The array_class structs of the rigid solver, its collider and its constraint solver that a substep reads and
    writes."""

    def __init__(self, solver: "RigidSolver") -> None:
        collider, constraint_solver = solver.collider, solver.constraint_solver
        self.geoms_init_AABB = solver.geoms_init_AABB
        self.dyn_state = solver.dyn_state
        self.dyn_info = solver.dyn_info
        self.rigid_info = solver.rigid_info
        self.collider_state = collider.collider_state
        self.collider_info = collider.collider_info
        self.mpr_state = collider.mpr.mpr_state
        self.contact0_mpr_state = collider.mpr.contact0_mpr_state
        self.multicontact_mpr_state = collider.mpr.multicontact_mpr_state
        self.gjk_state = collider.gjk.gjk_state
        self.contact0_gjk_state = collider.gjk.contact0_gjk_state
        self.multicontact_gjk_state = collider.gjk.multicontact_gjk_state
        self.constraint_state = constraint_solver.constraint_state
        self.errno = solver._errno


@qd.data_oriented
class RigidSolverSystem(System):
    """Manager of the rigid substep, wrapping the rigid solver of the scene, which keeps owning its data.

    A substep runs three graphs around a host solve, which selects its kernels at run time:
        graph 1  the forward dynamics, then every equality assembly
        graph 2  every detection, then every inequality assembly
        host     every solve
        graph 3  every constraint force update, then the integration
    The work before the solve is cut ahead of the detection, since the compile time of a kernel grows faster than its
    size while one more graph replay costs next to nothing. The collision detection and the constraint solve join
    through on_detect and on_constrain, from Systems the scene adds only when it enables them.
    """

    substepper = Require(Substepper)

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)
        self.solver = scene.sim.rigid_solver
        # The static configuration every graph of the substep is compiled against
        self.rigid_config = self.solver.rigid_config
        self.data = RigidSolverData(self.solver)

    @System.action_collection(kind="host")
    def preparations(self):
        """Host Actions run before the first graph of every substep."""

    @System.action_collection(kind="stage")
    def equality_assemblies(self):
        """Stage Actions that follow the forward dynamics in the first graph."""

    @System.action_collection(kind="stage")
    def detections(self):
        """Stage Actions that open the second graph."""

    @System.action_collection(kind="stage")
    def inequality_assemblies(self):
        """Stage Actions that follow every detection in the second graph."""

    @System.action_collection(kind="host")
    def solves(self):
        """Host Actions run between the second and the third graph."""

    @System.action_collection(kind="stage")
    def force_updates(self):
        """Stage Actions that open the third graph, ahead of the integration."""

    @System.protocol
    def on_detect(self, *, prepare: HostAction, detect: StageAction):
        """Join the rigid substep with a collision detection, calling it once from build().

        prepare  HostAction   `function(*bound_args)`, before the first graph
        detect   StageAction  `function(*bound_args)`, opening the second graph, once the forward dynamics has updated
                              the cartesian space

        No graph runs backward, since a scene tracking gradients advances the rigid solver through RigidSolver.substep.
        """
        self.preparations += prepare
        self.detections += detect

    @System.protocol
    def on_constrain(
        self,
        *,
        prepare: HostAction,
        assemble_equalities: StageAction,
        assemble_inequalities: StageAction,
        solve: HostAction,
        update_forces: StageAction,
    ):
        """Join the rigid substep with a constraint solve, calling it once from build().

        prepare                HostAction   `function(*bound_args)`, before the first graph
        assemble_equalities    StageAction  `function(*bound_args)`, in the first graph, after the forward dynamics
        assemble_inequalities  StageAction  `function(*bound_args)`, in the second graph, after every detection
        solve                  HostAction   `function(*bound_args)`, between the second and the third graph
        update_forces          StageAction  `function(*bound_args)`, in the third graph, before the integration

        No graph runs backward, since a scene tracking gradients advances the rigid solver through RigidSolver.substep.
        """
        self.preparations += prepare
        self.equality_assemblies += assemble_equalities
        self.inequality_assemblies += assemble_inequalities
        self.solves += solve
        self.force_updates += update_forces

    @System.action(kind="host")
    def process_input(self):
        return self.solver.process_input

    @System.action(kind="host")
    def pre_coupling(self):
        return self.substep_pre_coupling

    @System.action(kind="host")
    def post_coupling(self):
        return self.substep_post_coupling

    def build(self):
        self.substepper.on_substep(
            process_input=self.process_input,
            pre_coupling=self.pre_coupling,
            post_coupling=self.post_coupling,
            rank=self._scene.sim.solvers.index(self.solver),
        )

    def substep(self, f: int) -> None:
        """Advance the rigid solver by substep f."""
        solver = self.solver
        if isinstance(solver.sim.coupler, SAPCoupler) or solver._requires_grad:
            solver.substep(f)
            return
        solver.wakeup_coupled_links()
        for action in self.preparations.actions:
            action.invoke()
        kernel_substep_dynamics(self, solver._is_forward_pos_updated, solver._is_forward_vel_updated)
        kernel_substep_collision(self)
        for action in self.solves.actions:
            action.invoke()
        kernel_substep_post(self)
        solver.mark_forward_updated()

    def substep_pre_coupling(self, f: int) -> None:
        """Advance the rigid solver by substep f ahead of the coupling, unless an IPC coupler simulates rigid entities,
        which the rigid solver then advances after the coupling."""
        coupler = self.solver.sim.coupler
        if not (isinstance(coupler, IPCCoupler) and coupler.has_any_rigid_coupling):
            self.substep(f)

    def substep_post_coupling(self, f: int) -> None:
        """Complete substep f once the coupling has run."""
        coupler = self.solver.sim.coupler
        if isinstance(coupler, SAPCoupler):
            self.solver.finish_sap_substep()
        elif isinstance(coupler, IPCCoupler) and coupler.has_any_rigid_coupling:
            # The collider excludes the links coupled to IPC from its collisions at build time
            self.substep(f)


@qd.func(requires_top_level=True)
def func_substep_forward_dynamics(
    rigid_data: qd.template(),
    rigid_config: qd.template(),
    is_forward_pos_updated: qd.template(),
    is_forward_vel_updated: qd.template(),
):
    """Run the forward dynamics of a substep, after updating the cartesian space and the velocities unless they are
    already up to date."""
    func_step_1(
        rigid_data.dyn_state,
        rigid_data.constraint_state,
        rigid_data.dyn_info,
        rigid_data.rigid_info,
        rigid_config,
        is_forward_pos_updated,
        is_forward_vel_updated,
        False,
    )


@qd.func(requires_top_level=True)
def func_substep_integrate(rigid_data: qd.template(), rigid_config: qd.template()):
    """Integrate a substep, then update the cartesian space and the velocities the next substep starts from."""
    func_step_2(
        rigid_data.dyn_state,
        rigid_data.constraint_state,
        rigid_data.dyn_info,
        rigid_data.rigid_info,
        rigid_config,
        False,
        rigid_data.errno,
    )


# Each graph is a kernel taking the System, since Quadrants discovers the Data of every Action through it, and unrolls
# its frozen collections at compile time.
# FIXME: quadrants#966 - fastcache keys a kernel on its arguments without the constants an Action binds, so it could
# load a kernel compiled for another configuration of the solver. These kernels stay out of it until then.
@qd.kernel(graph=True)
def kernel_substep_dynamics(
    rigid_solver_system: qd.template(), is_forward_pos_updated: qd.template(), is_forward_vel_updated: qd.template()
):
    func_substep_forward_dynamics(
        rigid_solver_system.data, rigid_solver_system.rigid_config, is_forward_pos_updated, is_forward_vel_updated
    )
    for action in qd.static(rigid_solver_system.equality_assemblies.actions):
        action.invoke()


@qd.kernel(graph=True)
def kernel_substep_collision(rigid_solver_system: qd.template()):
    for action in qd.static(rigid_solver_system.detections.actions):
        action.invoke()
    for action in qd.static(rigid_solver_system.inequality_assemblies.actions):
        action.invoke()


@qd.kernel(graph=True)
def kernel_substep_post(rigid_solver_system: qd.template()):
    for action in qd.static(rigid_solver_system.force_updates.actions):
        action.invoke()
    func_substep_integrate(rigid_solver_system.data, rigid_solver_system.rigid_config)
