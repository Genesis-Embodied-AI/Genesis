"""System wrapping the rigid solver, which runs its substep as three graphs of stage Actions around a host solve."""

from typing import TYPE_CHECKING

import quadrants as qd

from genesis.engine.core import Data, HostAction, Require, StageAction, System
from genesis.engine.couplers import IPCCoupler, SAPCoupler
from genesis.engine.solvers.rigid.collider.collider import func_detection
from genesis.engine.solvers.rigid.constraint.solver import (
    func_add_equality_constraints,
    func_add_inequality_constraints,
    func_resolve_post,
    func_solve_body,
)
from genesis.engine.solvers.rigid.rigid_solver import func_step_1, func_step_2

from .substepper import Substepper

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver


# WORKAROUND: Quadrants discovers the tensors of the array_class structs reached from a kernel parameter only through a
# @qd.data_oriented object, so the structs the rigid solver owns are gathered here rather than bound one by one.
@qd.data_oriented
class RigidSolverData(Data):
    """The array_class structs of the rigid solver that one substep reads and writes."""

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
    """System wrapping the rigid solver of the scene, which keeps owning its data.

    A substep runs three graphs, each unrolling one collection of stage Actions, around the constraint solve, which
    selects its kernels on the host at run time. The work before the solve is cut ahead of the collision detection,
    since the compile time of a kernel grows faster than its size while one more graph replay costs next to nothing.
    """

    substepper = Require(Substepper)

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)
        self.solver = scene.sim.rigid_solver

    @System.action_collection(kind="stage", call_args=("is_forward_pos_updated", "is_forward_vel_updated"))
    def dynamics_stages(self):
        """Stage Actions of the first graph, which runs the forward dynamics."""

    @System.action_collection(kind="stage")
    def collision_stages(self):
        """Stage Actions of the second graph, which detects the collisions."""

    @System.action_collection(kind="host")
    def solves(self):
        """Host Actions run between the second and the third graph, which solve the constraints."""

    @System.action_collection(kind="stage")
    def post_stages(self):
        """Stage Actions of the third graph, which applies the constraint forces and integrates."""

    @System.protocol
    def on_rigid_substep(self, *, dynamics: StageAction, collision: StageAction, solve: HostAction, post: StageAction):
        """Join the rigid substep with one Action per phase, calling it once from build().

        dynamics   StageAction  `function(*bound_args, is_forward_pos_updated, is_forward_vel_updated)`, in the first
                                graph, where both flags state whether the cartesian space and the velocities are
                                already up to date
        collision  StageAction  `function(*bound_args)`, in the second graph
        solve      HostAction   `function(*bound_args)`, on the host between the second and the third graph
        post       StageAction  `function(*bound_args)`, in the third graph, which ends the substep

        Every phase runs its Actions in registration order. No phase runs backward, since a scene tracking gradients
        advances the rigid solver through RigidSolver.substep instead.
        """
        self.dynamics_stages += dynamics
        self.collision_stages += collision
        self.solves += solve
        self.post_stages += post

    @System.action(kind="host")
    def process_input(self):
        return self.solver.process_input

    @System.action(kind="host")
    def pre_coupling(self):
        return self.substep_pre_coupling

    @System.action(kind="host")
    def post_coupling(self):
        return self.substep_post_coupling

    @System.action(kind="stage")
    def dynamics(self):
        return func_substep_dynamics, self.data, self.solver.rigid_config, not self.solver._disable_constraint

    @System.action(kind="stage")
    def collision(self):
        collider = self.solver.collider
        return (
            func_substep_collision,
            self.data,
            self.solver.rigid_config,
            collider.collider_config,
            collider.gjk.gjk_config,
            not self.solver._disable_constraint,
            collider._n_possible_pairs > 0,
            collider._use_split_narrowphase,
            collider._use_coop_dedup,
        )

    @System.action(kind="host")
    def solve(self):
        return (
            solve_constraints,
            self.data,
            self.solver.rigid_config,
            not self.solver._disable_constraint,
            self.solver.constraint_solver._n_iterations,
        )

    @System.action(kind="stage")
    def post(self):
        return (
            func_substep_post,
            self.data,
            self.solver.rigid_config,
            not self.solver._disable_constraint,
            self.solver._options.noslip_iterations > 0,
        )

    def build(self):
        self.data = RigidSolverData(self.solver)
        self.on_rigid_substep(dynamics=self.dynamics, collision=self.collision, solve=self.solve, post=self.post)
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
        solver.collider._contact_data_cache.clear()
        solver.constraint_solver._eq_const_info_cache.clear()
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
def func_substep_dynamics(
    rigid_data: qd.template(),
    rigid_config: qd.template(),
    enable_constraint: qd.template(),
    is_forward_pos_updated: qd.template(),
    is_forward_vel_updated: qd.template(),
):
    """Run the forward dynamics of a substep, then assemble its equality constraints when enable_constraint is set."""
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
    if qd.static(enable_constraint):
        func_add_equality_constraints(
            rigid_data.dyn_state,
            rigid_data.collider_state,
            rigid_data.constraint_state,
            rigid_data.dyn_info,
            rigid_data.rigid_info,
            rigid_config,
        )


@qd.func(requires_top_level=True)
def func_substep_collision(
    rigid_data: qd.template(),
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    gjk_static_config: qd.template(),
    enable_constraint: qd.template(),
    has_possible_pairs: qd.template(),
    split_narrowphase: qd.template(),
    coop_dedup: qd.template(),
):
    """Detect the collisions of a substep, then assemble its inequality constraints when enable_constraint is set.

    has_possible_pairs, split_narrowphase and coop_dedup configure the detection (see func_detection).
    """
    if qd.static(rigid_config.enable_collision):
        func_detection(
            geoms_init_AABB=rigid_data.geoms_init_AABB,
            dyn_state=rigid_data.dyn_state,
            collider_state=rigid_data.collider_state,
            mpr_state=rigid_data.mpr_state,
            gjk_state=rigid_data.gjk_state,
            diff_contact_input=rigid_data.gjk_state.diff_contact_input,
            contact0_mpr_state=rigid_data.contact0_mpr_state,
            contact0_gjk_state=rigid_data.contact0_gjk_state,
            multicontact_mpr_state=rigid_data.multicontact_mpr_state,
            multicontact_gjk_state=rigid_data.multicontact_gjk_state,
            constraint_state=rigid_data.constraint_state,
            dyn_info=rigid_data.dyn_info,
            rigid_info=rigid_data.rigid_info,
            collider_info=rigid_data.collider_info,
            rigid_config=rigid_config,
            collider_static_config=collider_static_config,
            gjk_static_config=gjk_static_config,
            has_possible_pairs=has_possible_pairs,
            split_narrowphase=split_narrowphase,
            coop_dedup=coop_dedup,
            errno=rigid_data.errno,
        )
    if qd.static(enable_constraint):
        func_add_inequality_constraints(
            rigid_data.dyn_state,
            rigid_data.collider_state,
            rigid_data.constraint_state,
            rigid_data.dyn_info,
            rigid_data.rigid_info,
            rigid_config,
            collider_static_config,
        )


def solve_constraints(rigid_data: RigidSolverData, rigid_config, enable_constraint: bool, n_iterations: int) -> None:
    """Solve the constraints of a substep, through the solver arm func_solve_body selects at run time."""
    if enable_constraint:
        func_solve_body(
            rigid_data.dyn_state,
            rigid_data.constraint_state,
            rigid_data.dyn_info,
            rigid_data.rigid_info,
            rigid_config,
            n_iterations,
        )


@qd.func(requires_top_level=True)
def func_substep_post(
    rigid_data: qd.template(),
    rigid_config: qd.template(),
    enable_constraint: qd.template(),
    noslip: qd.template(),
):
    """Update the accelerations and the contact forces from the solved constraint forces when enable_constraint is set
    (see func_resolve_post for noslip), then integrate."""
    if qd.static(enable_constraint):
        func_resolve_post(
            rigid_data.dyn_state,
            rigid_data.collider_state,
            rigid_data.constraint_state,
            rigid_data.dyn_info,
            rigid_data.rigid_info,
            rigid_config,
            noslip,
            rigid_data.errno,
        )
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
# its frozen collection at compile time.
# FIXME: quadrants#966 - fastcache keys a kernel on its arguments without the constants an Action binds, so it could
# load a kernel compiled for another configuration of the solver. These kernels stay out of it until then.
@qd.kernel(graph=True)
def kernel_substep_dynamics(
    rigid_solver_system: qd.template(), is_forward_pos_updated: qd.template(), is_forward_vel_updated: qd.template()
):
    for action in qd.static(rigid_solver_system.dynamics_stages.actions):
        action.invoke((is_forward_pos_updated, is_forward_vel_updated))


@qd.kernel(graph=True)
def kernel_substep_collision(rigid_solver_system: qd.template()):
    for action in qd.static(rigid_solver_system.collision_stages.actions):
        action.invoke()


@qd.kernel(graph=True)
def kernel_substep_post(rigid_solver_system: qd.template()):
    for action in qd.static(rigid_solver_system.post_stages.actions):
        action.invoke()
