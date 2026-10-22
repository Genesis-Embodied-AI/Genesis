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
        from .entities import RigidEntity
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
        rigid, contact, constraint = self.rigid_dynamics, self.rigid_rigid_contact, self.rigid_constraint
        context = self.create_context()

        # Another solver may leave coupling forces on hibernated links, which wake before dynamics consumes them.
        self._coupled = any(not isinstance(entity, RigidEntity) for entity in self.scene.entities)

        # Dynamics graph: wake coupled bodies, advance dynamics, then begin the constraint rows with the equality ones.
        dynamics_schedule = self.action_schedule(
            wakeup=rigid.wake_coupled_links_stage if self._coupled else None,
            dynamics=rigid.forward_dynamics_stage,
            constraints=constraint.assemble_equality_rows if constraint else None,
        )
        self._dynamics_pipeline = Pipeline(kernel_substep_dynamics, context, dynamics_schedule)

        # Collision graph: detect contacts, then append the other constraint rows over the island partition.
        collision_schedule = self.action_schedule(
            collision=contact.detect if contact else None,
            constraints=constraint.assemble_inequality_rows if constraint else None,
        )
        self._collision_pipeline = Pipeline(kernel_substep_collision, context, collision_schedule)

        # Post-solve graph: apply the host-solved constraint forces, then integrate.
        post_schedule = self.action_schedule(
            constraints=constraint.apply_forces if constraint else None,
            integration=rigid.integrate_stage,
        )
        self._post_pipeline = Pipeline(kernel_substep_post, context, post_schedule)

        # The host path runs the same phases, launched as plain kernels.
        self._collision_host_pipeline = None
        if contact is not None or constraint is not None:
            self._collision_host_pipeline = Pipeline(kernel_substep_collision_host, context, collision_schedule)
        self._equality_rows_host_pipeline = self._apply_forces_host_pipeline = None
        if constraint is not None:
            equality_rows_schedule = self.action_schedule(constraints=constraint.assemble_equality_rows)
            self._equality_rows_host_pipeline = Pipeline(kernel_constraints_host, context, equality_rows_schedule)
            apply_forces_schedule = self.action_schedule(constraints=constraint.apply_forces)
            self._apply_forces_host_pipeline = Pipeline(kernel_constraints_host, context, apply_forces_schedule)

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
        self.rigid_dynamics.initialize_inverse_weights.invoke()

    def substep_graph(self) -> None:
        """Advance one Engine-level simulation substep."""
        if self.rigid_dynamics is None:
            return
        self._clear_readbacks()
        is_forward_pos_updated, is_forward_vel_updated = self.rigid_dynamics.get_forward_update_flags()
        self._dynamics_pipeline.run(is_forward_pos_updated, is_forward_vel_updated)
        self._collision_pipeline.run()
        if self.rigid_constraint is not None:
            self.rigid_constraint.solve.invoke()
        self._post_pipeline.run()
        self.rigid_dynamics.mark_forward_updated.invoke()

    def substep_host(self, f: int) -> None:
        """Advance through host-launched kernels.

        Retained for the current Legacy/IPC coupling lifecycle and reused by backward replay. It can disappear once
        those participants join the Engine-level graph.
        """
        rigid = self.rigid_dynamics
        if rigid is None:
            return
        if rigid.requires_grad and f == 0:
            rigid.save_adjoint_cache.invoke((f,))
        if self._coupled:
            rigid.wake_coupled_links_kernel.invoke()
        rigid.forward_dynamics_kernel.invoke(rigid.get_forward_kernel_args())
        self.constraint_force_host()
        rigid.integrate_kernel.invoke((rigid.is_backward,))
        rigid.mark_forward_updated.invoke()
        if rigid.requires_grad:
            rigid.save_adjoint_cache.invoke((f + 1,))

    def assemble_constraints_host(self) -> None:
        """Assemble the constraint rows through host-launched kernels: the equality rows, the collision detection, then
        the other rows."""
        self._clear_readbacks()
        if self._equality_rows_host_pipeline is not None:
            self._equality_rows_host_pipeline.run()
        if self._collision_host_pipeline is not None:
            self._collision_host_pipeline.run()

    def constraint_force_host(self) -> None:
        """Run collision and constraint phases for the temporary host coupling/backward path."""
        if self.rigid_dynamics is None:
            return
        self.assemble_constraints_host()
        if self.rigid_constraint is not None:
            self.rigid_constraint.solve.invoke()
            self._apply_forces_host_pipeline.run()

    def substep_to_velocity(self) -> None:
        """Advance to the velocity cut retained for the current SAP coupling lifecycle."""
        rigid = self.rigid_dynamics
        if rigid is None:
            return
        if self._coupled:
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
        rigid.restore_substep_inputs.invoke((f,))
        self.substep_host(f)
        rigid.reverse_integration.invoke((f,))
        if self.rigid_constraint is None:
            rigid.reverse_smooth_acceleration.invoke()
        else:
            self.rigid_constraint.reverse.invoke()
        rigid.reverse_forward_dynamics.invoke((f,))

    def _clear_readbacks(self) -> None:
        """Drop what the contact and constraint owners read back, before a substep recomputes it."""
        if self.rigid_rigid_contact is not None:
            self.rigid_rigid_contact.clear_readback.invoke()
        if self.rigid_constraint is not None:
            self.rigid_constraint.clear_readbacks.invoke()


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
    for action in qd.static(action_schedule.constraints):
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
def kernel_constraints_host(
    context: qd.template(),
    action_schedule: qd.template(),
):
    for action in qd.static(action_schedule.constraints):
        action.invoke()
