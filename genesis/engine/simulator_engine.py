"""Concrete Engine composing the Rigid Systems of a Simulator."""

from typing import TYPE_CHECKING

import quadrants as qd

from genesis.engine.core import Engine, Pipeline

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


class SimulatorEngine(Engine):
    """Engine owning and composing the admitted Systems of one Simulator."""

    def __init__(self, scene: "Scene") -> None:
        super().__init__()
        self.scene = scene

    def build(self) -> None:
        # Importing each module registers its factory for this concrete Engine; local imports avoid module cycles.
        from .systems.rigid_sdf_system import RigidSdfSystem
        from .systems.rigid_solver_system import RigidSolverSystem
        from .systems.rigid_island_system import RigidIslandSystem
        from .systems.rigid_rigid_contact_system import RigidRigidContactSystem
        from .systems.constraint_solver_system import ConstraintSolverSystem

        super().build()
        self.rigid_system = self.find(RigidSolverSystem)
        if self.rigid_system is None:
            self.rigid_sdf_system = None
            self.island_system = None
            self.contact_system = None
            self.constraint_system = None
            return
        self.rigid_sdf_system = self.require(RigidSdfSystem)
        self.island_system = self.require(RigidIslandSystem)
        self.contact_system = self.find(RigidRigidContactSystem)
        self.constraint_system = self.find(ConstraintSolverSystem)
        context = self.create_context()

        # Dynamics graph: wake coupled bodies, advance dynamics, then assemble equality constraints.
        dynamics_schedule = self.action_schedule(
            wakeup=self.rigid_system.wake_coupled_links_stage,
            dynamics=self.rigid_system.forward_dynamics_stage,
            equalities=self.constraint_system.assemble_equalities if self.constraint_system else None,
        )
        self._dynamics_pipeline = Pipeline(kernel_substep_dynamics, context, dynamics_schedule)

        # Collision graph: detect contacts, then assemble inequality constraints.
        collision_schedule = self.action_schedule(
            collision=self.contact_system.detect if self.contact_system else None,
            constraints=self.constraint_system.assemble_inequalities if self.constraint_system else None,
        )
        self._collision_pipeline = Pipeline(kernel_substep_collision, context, collision_schedule)

        # Post-solve graph: apply the host-solved constraint forces, then integrate.
        post_schedule = self.action_schedule(
            force_updates=self.constraint_system.update_forces if self.constraint_system else None,
            integration=self.rigid_system.integrate_stage,
        )
        self._post_pipeline = Pipeline(kernel_substep_post, context, post_schedule)

    def initialize(self) -> None:
        if self.rigid_system is None:
            return
        self.rigid_system.collect.invoke()
        self.rigid_sdf_system.initialize.invoke()
        self.rigid_system.initialize.invoke()
        if self.contact_system is not None:
            self.contact_system.initialize.invoke()
        self.island_system.initialize.invoke()
        if self.constraint_system is not None:
            self.constraint_system.initialize.invoke()
        self.rigid_system.finalize.invoke()

    def substep_graph(self) -> None:
        """Advance one Engine-level simulation substep."""
        if self.rigid_system is None:
            return
        if self.contact_system is not None:
            self.contact_system.prepare.invoke()
        if self.constraint_system is not None:
            self.constraint_system.prepare.invoke()
        is_forward_pos_updated, is_forward_vel_updated = self.rigid_system.get_forward_update_flags()
        self._dynamics_pipeline.run(is_forward_pos_updated, is_forward_vel_updated)
        self._collision_pipeline.run()
        if self.constraint_system is not None:
            self.constraint_system.solve.invoke()
        self._post_pipeline.run()
        self.rigid_system.mark_forward_updated()

    def substep_host(self, f: int) -> None:
        """Advance through host-launched kernels.

        Retained for the current Legacy/IPC coupling lifecycle and reused by backward replay. It can disappear once
        those participants join the Engine-level graph.
        """
        rigid = self.rigid_system
        if rigid is None:
            return
        if rigid.requires_grad and f == 0:
            rigid.save_adjoint_cache(f)
        if self.contact_system is not None:
            self.contact_system.prepare.invoke()
        if self.constraint_system is not None:
            self.constraint_system.prepare.invoke()
        rigid.wake_coupled_links_kernel.invoke()
        rigid.forward_dynamics_kernel.invoke(rigid.get_forward_kernel_args())
        self.constraint_force_host()
        rigid.integrate_kernel.invoke((rigid.is_backward,))
        rigid.mark_forward_updated()
        if rigid.requires_grad:
            rigid.save_adjoint_cache(f + 1)

    def constraint_force_host(self) -> None:
        """Run collision and constraint phases for the temporary host coupling/backward path."""
        rigid = self.rigid_system
        if rigid is None:
            return
        if self.constraint_system is not None:
            self.constraint_system.assemble_equalities_kernel.invoke()
        if self.contact_system is not None:
            self.contact_system.detect_kernel.invoke()
        if self.constraint_system is not None:
            self.constraint_system.assemble_inequalities_kernel.invoke()
            self.constraint_system.solve.invoke()
            self.constraint_system.update_forces_kernel.invoke()

    def substep_to_velocity(self) -> None:
        """Advance to the velocity cut retained for the current SAP coupling lifecycle."""
        rigid = self.rigid_system
        if rigid is None:
            return
        rigid.wake_coupled_links_kernel.invoke()
        rigid.forward_dynamics_kernel.invoke(rigid.get_forward_kernel_args())
        rigid.update_velocity_for_coupling()

    def substep_after_velocity_coupling(self) -> None:
        """Complete integration after the velocity cut retained for the current SAP coupling lifecycle."""
        rigid = self.rigid_system
        if rigid is None:
            return
        rigid.update_acceleration_after_coupling()
        rigid.integrate_kernel.invoke((rigid.is_backward,))

    def substep_backward(self, f: int) -> None:
        """Replay the composed forward phases, then reverse owner stages."""
        rigid = self.rigid_system
        if rigid is None:
            return
        rigid.prepare_backward_replay(f)
        self.substep_host(f)
        rigid.begin_backward(f)
        if self.constraint_system is None:
            rigid.reverse_smooth_acceleration()
        else:
            self.constraint_system.reverse_solve.invoke()
            if self.contact_system is not None:
                self.contact_system.prepare_backward.invoke()
                self.constraint_system.reverse_collision_rows.invoke()
                self.contact_system.backward_narrowphase()
            self.constraint_system.reverse_joint_limit_rows.invoke()
        rigid.finish_backward(f)


# These graph adapter kernels belong to Engine scheduling. The participating Systems provide only Actions and Data.
# FIXME: quadrants#966 - fastcache keys a kernel on its arguments without the constants an Action binds, so it could
# load a kernel compiled for another configuration. Keep these graph kernels out of fastcache until then.
@qd.kernel(graph=True)
def kernel_substep_dynamics(
    context: qd.template(),
    action_schedule: qd.template(),
    is_forward_pos_updated: qd.template(),
    is_forward_vel_updated: qd.template(),
):
    for action in qd.static(action_schedule.wakeup):
        action.invoke()
    for action in qd.static(action_schedule.dynamics):
        action.invoke((is_forward_pos_updated, is_forward_vel_updated, False))
    for action in qd.static(action_schedule.equalities):
        action.invoke()


@qd.kernel(graph=True)
def kernel_substep_collision(
    context: qd.template(),
    action_schedule: qd.template(),
):
    for action in qd.static(action_schedule.collision):
        action.invoke()
    for action in qd.static(action_schedule.constraints):
        action.invoke()


@qd.kernel(graph=True)
def kernel_substep_post(
    context: qd.template(),
    action_schedule: qd.template(),
):
    for action in qd.static(action_schedule.force_updates):
        action.invoke()
    for action in qd.static(action_schedule.integration):
        action.invoke((False,))
