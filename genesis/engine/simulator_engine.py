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
        from .systems.rigid_sdf import RigidSdf
        from .systems.rigid_dynamics import RigidDynamics
        from .systems.rigid_island import RigidIsland
        from .systems.rigid_rigid_contact import RigidRigidContact
        from .systems.rigid_constraint import RigidConstraint

        super().build()
        self.rigid_dynamics = self.find(RigidDynamics)
        if self.rigid_dynamics is None:
            self.rigid_sdf = None
            self.rigid_island = None
            self.rigid_rigid_contact = None
            self.rigid_constraint = None
            return
        self.rigid_sdf = self.require(RigidSdf)
        self.rigid_island = self.require(RigidIsland)
        self.rigid_rigid_contact = self.find(RigidRigidContact)
        self.rigid_constraint = self.find(RigidConstraint)
        context = self.create_context()

        # Dynamics graph: wake coupled bodies, then advance dynamics.
        dynamics_schedule = self.action_schedule(
            wakeup=self.rigid_dynamics.wake_coupled_links_stage,
            dynamics=self.rigid_dynamics.forward_dynamics_stage,
        )
        self._dynamics_pipeline = Pipeline(kernel_substep_dynamics, context, dynamics_schedule)

        # Collision graph: detect contacts, then assemble the constraint rows.
        collision_schedule = self.action_schedule(
            collision=self.rigid_rigid_contact.detect if self.rigid_rigid_contact else None,
            constraints=self.rigid_constraint.assemble if self.rigid_constraint else None,
        )
        self._collision_pipeline = Pipeline(kernel_substep_collision, context, collision_schedule)

        # Post-solve graph: apply the host-solved constraint forces, then integrate.
        post_schedule = self.action_schedule(
            constraints=self.rigid_constraint.post_solve if self.rigid_constraint else None,
            integration=self.rigid_dynamics.integrate_stage,
        )
        self._post_pipeline = Pipeline(kernel_substep_post, context, post_schedule)

        # The host path runs the same collision and post-solve phases, launched as plain kernels.
        self._collision_host_pipeline = None
        if self.rigid_rigid_contact is not None or self.rigid_constraint is not None:
            self._collision_host_pipeline = Pipeline(kernel_substep_collision_host, context, collision_schedule)
        self._post_solve_host_pipeline = None
        if self.rigid_constraint is not None:
            post_solve_schedule = self.action_schedule(constraints=self.rigid_constraint.post_solve)
            self._post_solve_host_pipeline = Pipeline(kernel_substep_post_solve_host, context, post_solve_schedule)

    def initialize(self) -> None:
        if self.rigid_dynamics is None:
            return
        self.rigid_dynamics.collect.invoke()
        self.rigid_sdf.initialize.invoke()
        self.rigid_dynamics.initialize.invoke()
        if self.rigid_rigid_contact is not None:
            self.rigid_rigid_contact.initialize.invoke()
        self.rigid_island.initialize.invoke()
        if self.rigid_constraint is not None:
            self.rigid_constraint.initialize.invoke()
        self.rigid_dynamics.finalize.invoke()

    def substep_graph(self) -> None:
        """Advance one Engine-level simulation substep."""
        if self.rigid_dynamics is None:
            return
        if self.rigid_rigid_contact is not None:
            self.rigid_rigid_contact.prepare.invoke()
        is_forward_pos_updated, is_forward_vel_updated = self.rigid_dynamics.get_forward_update_flags()
        self._dynamics_pipeline.run(is_forward_pos_updated, is_forward_vel_updated)
        self._collision_pipeline.run()
        if self.rigid_constraint is not None:
            self.rigid_constraint.solve.invoke()
        self._post_pipeline.run()
        self.rigid_dynamics.mark_forward_updated()

    def substep_host(self, f: int) -> None:
        """Advance through host-launched kernels.

        Retained for the current Legacy/IPC coupling lifecycle and reused by backward replay. It can disappear once
        those participants join the Engine-level graph.
        """
        rigid = self.rigid_dynamics
        if rigid is None:
            return
        if rigid.requires_grad and f == 0:
            rigid.save_adjoint_cache(f)
        if self.rigid_rigid_contact is not None:
            self.rigid_rigid_contact.prepare.invoke()
        rigid.wake_coupled_links_kernel.invoke()
        rigid.forward_dynamics_kernel.invoke(rigid.get_forward_kernel_args())
        self.constraint_force_host()
        rigid.integrate_kernel.invoke((rigid.is_backward,))
        rigid.mark_forward_updated()
        if rigid.requires_grad:
            rigid.save_adjoint_cache(f + 1)

    def assemble_constraints_host(self) -> None:
        """Detect the contacts and assemble the constraint rows through host-launched kernels."""
        if self._collision_host_pipeline is not None:
            self._collision_host_pipeline.run()

    def constraint_force_host(self) -> None:
        """Run collision and constraint phases for the temporary host coupling/backward path."""
        if self.rigid_dynamics is None:
            return
        self.assemble_constraints_host()
        if self.rigid_constraint is not None:
            self.rigid_constraint.solve.invoke()
            self._post_solve_host_pipeline.run()

    def substep_to_velocity(self) -> None:
        """Advance to the velocity cut retained for the current SAP coupling lifecycle."""
        rigid = self.rigid_dynamics
        if rigid is None:
            return
        rigid.wake_coupled_links_kernel.invoke()
        rigid.forward_dynamics_kernel.invoke(rigid.get_forward_kernel_args())
        rigid.update_velocity_for_coupling()

    def substep_after_velocity_coupling(self) -> None:
        """Complete integration after the velocity cut retained for the current SAP coupling lifecycle."""
        rigid = self.rigid_dynamics
        if rigid is None:
            return
        rigid.update_acceleration_after_coupling()
        rigid.integrate_kernel.invoke((rigid.is_backward,))

    def substep_backward(self, f: int) -> None:
        """Replay the composed forward phases, then reverse owner stages."""
        rigid = self.rigid_dynamics
        if rigid is None:
            return
        rigid.prepare_backward_replay(f)
        self.substep_host(f)
        rigid.begin_backward(f)
        if self.rigid_constraint is None:
            rigid.reverse_smooth_acceleration()
        else:
            self.rigid_constraint.reverse.invoke()
        rigid.finish_backward(f)


# These adapter kernels belong to Engine scheduling. The participating Systems provide only Actions and Data. A
# composite Action binds the Data of its children inside nested Actions, which no owner kernel exposes as parameters,
# so every adapter takes the context first, graph or not.
# FIXME: quadrants#966 - fastcache keys a kernel on its arguments without the constants an Action binds, so it could
# load a kernel compiled for another configuration. Keep these kernels out of fastcache until then.
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
    for action in qd.static(action_schedule.constraints):
        action.invoke()
    for action in qd.static(action_schedule.integration):
        action.invoke((False,))


@qd.kernel
def kernel_substep_collision_host(
    context: qd.template(),
    action_schedule: qd.template(),
):
    for action in qd.static(action_schedule.collision):
        action.invoke()
    for action in qd.static(action_schedule.constraints):
        action.invoke()


@qd.kernel
def kernel_substep_post_solve_host(
    context: qd.template(),
    action_schedule: qd.template(),
):
    for action in qd.static(action_schedule.constraints):
        action.invoke()
