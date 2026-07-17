"""
Rigid solver control, getter, and setter kernel functions.

This module contains Quadrants kernel functions for controlling rigid body simulations,
including state getters/setters, link position/quaternion manipulation, DOF control,
and drone-specific operations.

These functions are used by the RigidSolver class to interface with the Quadrants
data structures for rigid body dynamics simulation.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class
from .misc import func_apply_link_external_force, func_apply_link_external_torque, func_wakeup_island


@qd.kernel(fastcache=True)
def kernel_get_kinematic_state(
    i_pos_shift: qd.types.ndarray(),
    qpos: qd.types.ndarray(),
    vel: qd.types.ndarray(),
    links_pos: qd.types.ndarray(),
    links_quat: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    n_qs = qpos.shape[1]
    n_dofs = vel.shape[1]
    n_links = links_pos.shape[1]
    _B = qpos.shape[0]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_q, i_b in qd.ndrange(n_qs, _B):
        qpos[i_b, i_q] = rigid_info.qpos[i_q, i_b]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        vel[i_b, i_d] = dyn_state.dofs.vel[i_d, i_b]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b in qd.ndrange(n_links, _B):
        for j in qd.static(range(3)):
            links_pos[i_b, i_l, j] = dyn_state.links.pos[i_l, i_b][j]
            i_pos_shift[i_b, i_l, j] = dyn_state.links.i_pos_shift[i_l, i_b][j]
        for j in qd.static(range(4)):
            links_quat[i_b, i_l, j] = dyn_state.links.quat[i_l, i_b][j]


@qd.kernel(fastcache=True)
def kernel_set_kinematic_state(
    envs_idx: qd.types.ndarray(),
    i_pos_shift: qd.types.ndarray(),
    qpos: qd.types.ndarray(),
    dofs_vel: qd.types.ndarray(),
    links_pos: qd.types.ndarray(),
    links_quat: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    n_qs = qpos.shape[1]
    n_dofs = dofs_vel.shape[1]
    n_links = links_pos.shape[1]
    _B = envs_idx.shape[0]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_q, i_b_ in qd.ndrange(n_qs, _B):
        rigid_info.qpos[i_q, envs_idx[i_b_]] = qpos[envs_idx[i_b_], i_q]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b_ in qd.ndrange(n_dofs, _B):
        dyn_state.dofs.vel[i_d, envs_idx[i_b_]] = dofs_vel[envs_idx[i_b_], i_d]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b_ in qd.ndrange(n_links, _B):
        for j in qd.static(range(3)):
            dyn_state.links.pos[i_l, envs_idx[i_b_]][j] = links_pos[envs_idx[i_b_], i_l, j]
            dyn_state.links.i_pos_shift[i_l, envs_idx[i_b_]][j] = i_pos_shift[envs_idx[i_b_], i_l, j]
        for j in qd.static(range(4)):
            dyn_state.links.quat[i_l, envs_idx[i_b_]][j] = links_quat[envs_idx[i_b_], i_l, j]


@qd.kernel(fastcache=True)
def kernel_get_state(
    i_pos_shift: qd.types.ndarray(),
    qpos: qd.types.ndarray(),
    vel: qd.types.ndarray(),
    acc: qd.types.ndarray(),
    links_pos: qd.types.ndarray(),
    links_quat: qd.types.ndarray(),
    mass_shift: qd.types.ndarray(),
    friction_ratio: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    n_qs = qpos.shape[1]
    n_dofs = vel.shape[1]
    n_links = links_pos.shape[1]
    n_geoms = friction_ratio.shape[1]
    _B = qpos.shape[0]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_q, i_b in qd.ndrange(n_qs, _B):
        qpos[i_b, i_q] = rigid_info.qpos[i_q, i_b]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        vel[i_b, i_d] = dyn_state.dofs.vel[i_d, i_b]
        acc[i_b, i_d] = dyn_state.dofs.acc[i_d, i_b]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b in qd.ndrange(n_links, _B):
        for j in qd.static(range(3)):
            links_pos[i_b, i_l, j] = dyn_state.links.pos[i_l, i_b][j]
            i_pos_shift[i_b, i_l, j] = dyn_state.links.i_pos_shift[i_l, i_b][j]
        for j in qd.static(range(4)):
            links_quat[i_b, i_l, j] = dyn_state.links.quat[i_l, i_b][j]
        mass_shift[i_b, i_l] = dyn_state.links.mass_shift[i_l, i_b]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b in qd.ndrange(n_geoms, _B):
        friction_ratio[i_b, i_l] = dyn_state.geoms.friction_ratio[i_l, i_b]


@qd.kernel(fastcache=True)
def kernel_set_state(
    envs_idx: qd.types.ndarray(),
    i_pos_shift: qd.types.ndarray(),
    qpos: qd.types.ndarray(),
    dofs_vel: qd.types.ndarray(),
    dofs_acc: qd.types.ndarray(),
    links_pos: qd.types.ndarray(),
    links_quat: qd.types.ndarray(),
    mass_shift: qd.types.ndarray(),
    friction_ratio: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    n_qs = qpos.shape[1]
    n_dofs = dofs_vel.shape[1]
    n_links = links_pos.shape[1]
    n_geoms = friction_ratio.shape[1]
    _B = envs_idx.shape[0]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_q, i_b_ in qd.ndrange(n_qs, _B):
        rigid_info.qpos[i_q, envs_idx[i_b_]] = qpos[envs_idx[i_b_], i_q]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b_ in qd.ndrange(n_dofs, _B):
        dyn_state.dofs.vel[i_d, envs_idx[i_b_]] = dofs_vel[envs_idx[i_b_], i_d]
        dyn_state.dofs.acc[i_d, envs_idx[i_b_]] = dofs_acc[envs_idx[i_b_], i_d]
        dyn_state.dofs.ctrl_force[i_d, envs_idx[i_b_]] = gs.qd_float(0.0)
        dyn_state.dofs.ctrl_mode[i_d, envs_idx[i_b_]] = gs.CTRL_MODE.FORCE

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b_ in qd.ndrange(n_links, _B):
        for j in qd.static(range(3)):
            dyn_state.links.pos[i_l, envs_idx[i_b_]][j] = links_pos[envs_idx[i_b_], i_l, j]
            dyn_state.links.i_pos_shift[i_l, envs_idx[i_b_]][j] = i_pos_shift[envs_idx[i_b_], i_l, j]
            dyn_state.links.cfrc_applied_vel[i_l, envs_idx[i_b_]][j] = gs.qd_float(0.0)
            dyn_state.links.cfrc_applied_ang[i_l, envs_idx[i_b_]][j] = gs.qd_float(0.0)
        for j in qd.static(range(4)):
            dyn_state.links.quat[i_l, envs_idx[i_b_]][j] = links_quat[envs_idx[i_b_], i_l, j]
        dyn_state.links.mass_shift[i_l, envs_idx[i_b_]] = mass_shift[envs_idx[i_b_], i_l]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b_ in qd.ndrange(n_geoms, _B):
        dyn_state.geoms.friction_ratio[i_l, envs_idx[i_b_]] = friction_ratio[envs_idx[i_b_], i_l]


@qd.kernel(fastcache=True)
def kernel_get_state_grad(
    qpos_grad: qd.types.ndarray(),
    vel_grad: qd.types.ndarray(),
    links_pos_grad: qd.types.ndarray(),
    links_quat_grad: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    n_qs = qpos_grad.shape[1]
    n_dofs = vel_grad.shape[1]
    n_links = links_pos_grad.shape[1]
    _B = qpos_grad.shape[0]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_q, i_b in qd.ndrange(n_qs, _B):
        qd.atomic_add(rigid_info.qpos.grad[i_q, i_b], qpos_grad[i_b, i_q])

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        qd.atomic_add(dyn_state.dofs.vel.grad[i_d, i_b], vel_grad[i_b, i_d])

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b in qd.ndrange(n_links, _B):
        for j in qd.static(range(3)):
            qd.atomic_add(dyn_state.links.pos.grad[i_l, i_b][j], links_pos_grad[i_b, i_l, j])
        for j in qd.static(range(4)):
            qd.atomic_add(dyn_state.links.quat.grad[i_l, i_b][j], links_quat_grad[i_b, i_l, j])


@qd.kernel(fastcache=True)
def kernel_set_links_pos(
    links_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    pos: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l_, i_b_ in qd.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_l = links_idx[i_l_]
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l

        if dyn_info.links.parent_idx[I_l] == -1 and dyn_info.links.is_fixed[I_l]:
            for j in qd.static(range(3)):
                dyn_state.links.pos[i_l, i_b][j] = pos[i_b_, i_l_, j]
        else:
            q_start = dyn_info.links.q_start[I_l]
            for j in qd.static(range(3)):
                rigid_info.qpos[q_start + j, i_b] = pos[i_b_, i_l_, j]


@qd.kernel(fastcache=True)
def kernel_wake_up_entities_by_links(
    links_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Wake up the entities owning the specified links, along with the other entities of their hibernated islands.

    Waking up the whole island is necessary to clear its daisy-chain links, which would otherwise keep re-connecting
    the woken entities to their previous islands at the next island partition build."""
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l_, i_b_ in qd.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_l = links_idx[i_l_]
        if dyn_state.links.is_hibernated[i_l, i_b]:
            func_wakeup_island(
                constraint_state.island.links_island_idx[i_l, i_b],
                i_b,
                dyn_state,
                constraint_state,
                dyn_info,
                rigid_info,
                rigid_config,
            )


@qd.kernel(fastcache=True)
def kernel_wake_up_entities_by_dofs(
    dofs_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Wake up the component-island owning each specified DOF, so writing a sleeping body's position or velocity
    revives it (and clears its daisy chain) rather than being silently dropped. The by-DOF analogue of
    kernel_wake_up_entities_by_links, used by the DOF-level state setters."""
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d_, i_b_ in qd.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_d = dofs_idx[i_d_]

        if dyn_state.dofs.is_hibernated[i_d, i_b]:
            func_wakeup_island(
                constraint_state.island.dofs_island_idx[i_d, i_b],
                i_b,
                dyn_state,
                constraint_state,
                dyn_info,
                rigid_info,
                rigid_config,
            )


@qd.kernel(fastcache=True)
def kernel_wake_up_entities_by_qs(
    qs_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Wake up the entities owning the specified generalized coordinates (qs), located via the link whose q-range
    contains each qs. The by-qs analogue of kernel_wake_up_entities_by_links, used by set_qpos."""
    n_links = dyn_info.links.q_start.shape[0]
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_q_, i_b_ in qd.ndrange(qs_idx.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_q = qs_idx[i_q_]
        for i_l in range(n_links):
            I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
            if dyn_info.links.q_start[I_l] <= i_q and i_q < dyn_info.links.q_end[I_l]:
                if dyn_state.links.is_hibernated[i_l, i_b]:
                    func_wakeup_island(
                        constraint_state.island.links_island_idx[i_l, i_b],
                        i_b,
                        dyn_state,
                        constraint_state,
                        dyn_info,
                        rigid_info,
                        rigid_config,
                    )


@qd.kernel(fastcache=True)
def kernel_wake_up_entities_on_new_contact(
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Wake a sleeping body when an awake body collides with it, so it responds dynamically instead of acting as an
    immovable obstacle. Runs after collision detection and before the solve, so the woken body joins the island
    partition and is solved this step. Only a contact whose partner is an awake dynamic body wakes the sleeper:
    hibernated-fixed (resting on the ground) and hibernated-hibernated (one sleeping island) contacts are left
    asleep, as they generate no new motion."""
    _B = collider_state.n_contacts.shape[0]
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        for i_c in range(collider_state.n_contacts[i_b]):
            i_la = collider_state.contact_data.link_a[i_c, i_b]
            i_lb = collider_state.contact_data.link_b[i_c, i_b]
            I_la = [i_la, i_b] if qd.static(rigid_config.batch_links_info) else i_la
            I_lb = [i_lb, i_b] if qd.static(rigid_config.batch_links_info) else i_lb
            is_a_hibernated = dyn_state.links.is_hibernated[i_la, i_b]
            is_b_hibernated = dyn_state.links.is_hibernated[i_lb, i_b]

            # Wake the sleeping side only when its partner is an awake dynamic body. Checking the per-link flag (not the
            # owning entity's) is what lets a single Genesis entity's settled free body wake when another of its free
            # bodies - or any awake body - strikes it.
            if is_a_hibernated and not is_b_hibernated and not dyn_info.links.is_fixed[I_lb]:
                func_wakeup_island(
                    constraint_state.island.links_island_idx[i_la, i_b],
                    i_b,
                    dyn_state,
                    constraint_state,
                    dyn_info,
                    rigid_info,
                    rigid_config,
                )
            if is_b_hibernated and not is_a_hibernated and not dyn_info.links.is_fixed[I_la]:
                func_wakeup_island(
                    constraint_state.island.links_island_idx[i_lb, i_b],
                    i_b,
                    dyn_state,
                    constraint_state,
                    dyn_info,
                    rigid_info,
                    rigid_config,
                )


@qd.kernel(fastcache=True)
def kernel_set_links_pos_grad(
    links_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    pos_grad: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l_, i_b_ in qd.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_l = links_idx[i_l_]
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l

        if dyn_info.links.parent_idx[I_l] == -1 and dyn_info.links.is_fixed[I_l]:
            for j in qd.static(range(3)):
                pos_grad[i_b_, i_l_, j] = dyn_state.links.pos.grad[i_l, i_b][j]
                dyn_state.links.pos.grad[i_l, i_b][j] = 0.0
        else:
            q_start = dyn_info.links.q_start[I_l]
            for j in qd.static(range(3)):
                pos_grad[i_b_, i_l_, j] = rigid_info.qpos.grad[q_start + j, i_b]
                rigid_info.qpos.grad[q_start + j, i_b] = 0.0


@qd.kernel(fastcache=True)
def kernel_set_links_quat(
    links_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    quat: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l_, i_b_ in qd.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_l = links_idx[i_l_]
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l

        if dyn_info.links.parent_idx[I_l] == -1 and dyn_info.links.is_fixed[I_l]:
            for j in qd.static(range(4)):
                dyn_state.links.quat[i_l, i_b][j] = quat[i_b_, i_l_, j]
        else:
            q_start = dyn_info.links.q_start[I_l]
            for j in qd.static(range(4)):
                rigid_info.qpos[q_start + j + 3, i_b] = quat[i_b_, i_l_, j]


@qd.kernel(fastcache=True)
def kernel_set_links_quat_grad(
    links_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    quat_grad: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l_, i_b_ in qd.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_l = links_idx[i_l_]
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l

        if dyn_info.links.parent_idx[I_l] == -1 and dyn_info.links.is_fixed[I_l]:
            for j in qd.static(range(4)):
                quat_grad[i_b_, i_l_, j] = dyn_state.links.quat.grad[i_l, i_b][j]
                dyn_state.links.quat.grad[i_l, i_b][j] = 0.0
        else:
            q_start = dyn_info.links.q_start[I_l]
            for j in qd.static(range(4)):
                quat_grad[i_b_, i_l_, j] = rigid_info.qpos.grad[q_start + j + 3, i_b]
                rigid_info.qpos.grad[q_start + j + 3, i_b] = 0.0


@qd.kernel(fastcache=True)
def kernel_set_links_mass_shift(
    links_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    mass: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l_, i_b_ in qd.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        dyn_state.links.mass_shift[links_idx[i_l_], envs_idx[i_b_]] = mass[i_b_, i_l_]


@qd.kernel(fastcache=True)
def kernel_set_links_COM_shift(
    links_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    com: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l_, i_b_ in qd.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        for j in qd.static(range(3)):
            dyn_state.links.i_pos_shift[links_idx[i_l_], envs_idx[i_b_]][j] = com[i_b_, i_l_, j]


@qd.kernel(fastcache=True)
def kernel_set_links_inertial_mass(
    links_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    inertial_mass: qd.types.ndarray(),
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    if qd.static(rigid_config.batch_links_info):
        for i_l_, i_b_ in qd.ndrange(links_idx.shape[0], envs_idx.shape[0]):
            dyn_info.links.inertial_mass[links_idx[i_l_], envs_idx[i_b_]] = inertial_mass[i_b_, i_l_]
    else:
        for i_l_ in range(links_idx.shape[0]):
            dyn_info.links.inertial_mass[links_idx[i_l_]] = inertial_mass[i_l_]


@qd.kernel(fastcache=True)
def kernel_adjust_link_inertia(
    links_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    ratio: qd.types.ndarray(),
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    if qd.static(rigid_config.batch_links_info):
        for i_l_, i_b_ in qd.ndrange(links_idx.shape[0], envs_idx.shape[0]):
            r = ratio[i_b_, i_l_]
            dyn_info.links.inertial_mass[links_idx[i_l_], envs_idx[i_b_]] *= r
            for j1, j2 in qd.static(qd.ndrange(3, 3)):
                dyn_info.links.inertial_i[links_idx[i_l_], envs_idx[i_b_]][j1, j2] *= r
            for j in qd.static(range(2)):
                dyn_info.links.invweight[links_idx[i_l_], envs_idx[i_b_]][j] /= r
    else:
        for i_l_ in range(links_idx.shape[0]):
            r = ratio[i_l_]
            dyn_info.links.inertial_mass[links_idx[i_l_]] *= r
            for j1, j2 in qd.static(qd.ndrange(3, 3)):
                dyn_info.links.inertial_i[links_idx[i_l_]][j1, j2] *= r
            for j in qd.static(range(2)):
                dyn_info.links.invweight[links_idx[i_l_]][j] /= r


@qd.kernel(fastcache=True)
def kernel_set_geoms_friction_ratio(
    geoms_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    friction_ratio: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_g_, i_b_ in qd.ndrange(geoms_idx.shape[0], envs_idx.shape[0]):
        dyn_state.geoms.friction_ratio[geoms_idx[i_g_], envs_idx[i_b_]] = friction_ratio[i_b_, i_g_]


@qd.kernel(fastcache=True)
def kernel_set_qpos(
    qs_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    qpos: qd.types.ndarray(),
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_q_, i_b_ in qd.ndrange(qs_idx.shape[0], envs_idx.shape[0]):
        rigid_info.qpos[qs_idx[i_q_], envs_idx[i_b_]] = qpos[i_b_, i_q_]


@qd.kernel(fastcache=True)
def kernel_set_global_sol_params(
    sol_params: qd.types.ndarray(), dyn_info: array_class.DynInfo, rigid_config: qd.template()
):
    n_geoms = dyn_info.geoms.sol_params.shape[0]
    n_joints = dyn_info.joints.sol_params.shape[0]
    n_equalities = dyn_info.equalities.sol_params.shape[0]
    _B = dyn_info.equalities.sol_params.shape[1]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_g in range(n_geoms):
        for j in qd.static(range(7)):
            dyn_info.geoms.sol_params[i_g][j] = sol_params[j]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_j, i_b in qd.ndrange(n_joints, _B):
        I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
        for j in qd.static(range(7)):
            dyn_info.joints.sol_params[I_j][j] = sol_params[j]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_eq, i_b in qd.ndrange(n_equalities, _B):
        for j in qd.static(range(7)):
            dyn_info.equalities.sol_params[i_eq, i_b][j] = sol_params[j]


@qd.kernel(fastcache=True)
def kernel_set_sol_params(
    inputs_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    sol_params: qd.types.ndarray(),
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
    constraint_type: qd.template(),
):
    if qd.static(constraint_type == 0):  # geometries
        qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
        for i_g_ in range(inputs_idx.shape[0]):
            for j in qd.static(range(7)):
                dyn_info.geoms.sol_params[inputs_idx[i_g_]][j] = sol_params[i_g_, j]
    if qd.static(constraint_type == 1):  # joints
        qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
        if qd.static(rigid_config.batch_joints_info):
            for i_j_, i_b_ in qd.ndrange(inputs_idx.shape[0], envs_idx.shape[0]):
                for j in qd.static(range(7)):
                    dyn_info.joints.sol_params[inputs_idx[i_j_], envs_idx[i_b_]][j] = sol_params[i_b_, i_j_, j]
        else:
            for i_j_ in range(inputs_idx.shape[0]):
                for j in qd.static(range(7)):
                    dyn_info.joints.sol_params[inputs_idx[i_j_]][j] = sol_params[i_j_, j]
    if qd.static(constraint_type == 2):  # equalities
        qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
        for i_eq_, i_b_ in qd.ndrange(inputs_idx.shape[0], envs_idx.shape[0]):
            for j in qd.static(range(7)):
                dyn_info.equalities.sol_params[inputs_idx[i_eq_], envs_idx[i_b_]][j] = sol_params[i_b_, i_eq_, j]


@qd.kernel(fastcache=True)
def kernel_set_dofs_kp(
    dofs_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    kp: qd.types.ndarray(),
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    if qd.static(rigid_config.batch_dofs_info):
        for i_d_, i_b_ in qd.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            dyn_info.dofs.act_gain[dofs_idx[i_d_], envs_idx[i_b_]] = kp[i_b_, i_d_]
            dyn_info.dofs.act_bias[dofs_idx[i_d_], envs_idx[i_b_]][0] = 0.0
            dyn_info.dofs.act_bias[dofs_idx[i_d_], envs_idx[i_b_]][1] = -kp[i_b_, i_d_]
    else:
        for i_d_ in range(dofs_idx.shape[0]):
            dyn_info.dofs.act_gain[dofs_idx[i_d_]] = kp[i_d_]
            dyn_info.dofs.act_bias[dofs_idx[i_d_]][0] = 0.0
            dyn_info.dofs.act_bias[dofs_idx[i_d_]][1] = -kp[i_d_]


@qd.kernel(fastcache=True)
def kernel_set_dofs_kv(
    dofs_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    kv: qd.types.ndarray(),
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    if qd.static(rigid_config.batch_dofs_info):
        for i_d_, i_b_ in qd.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            dyn_info.dofs.act_bias[dofs_idx[i_d_], envs_idx[i_b_]][2] = -kv[i_b_, i_d_]
    else:
        for i_d_ in range(dofs_idx.shape[0]):
            dyn_info.dofs.act_bias[dofs_idx[i_d_]][2] = -kv[i_d_]


@qd.kernel(fastcache=True)
def kernel_set_dofs_act_gain(
    dofs_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    act_gain: qd.types.ndarray(),
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    if qd.static(rigid_config.batch_dofs_info):
        for i_d_, i_b_ in qd.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            dyn_info.dofs.act_gain[dofs_idx[i_d_], envs_idx[i_b_]] = act_gain[i_b_, i_d_]
    else:
        for i_d_ in range(dofs_idx.shape[0]):
            dyn_info.dofs.act_gain[dofs_idx[i_d_]] = act_gain[i_d_]


@qd.kernel(fastcache=True)
def kernel_set_dofs_act_bias(
    dofs_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    bias0: qd.types.ndarray(),
    bias1: qd.types.ndarray(),
    bias2: qd.types.ndarray(),
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    if qd.static(rigid_config.batch_dofs_info):
        for i_d_, i_b_ in qd.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            dyn_info.dofs.act_bias[dofs_idx[i_d_], envs_idx[i_b_]][0] = bias0[i_b_, i_d_]
            dyn_info.dofs.act_bias[dofs_idx[i_d_], envs_idx[i_b_]][1] = bias1[i_b_, i_d_]
            dyn_info.dofs.act_bias[dofs_idx[i_d_], envs_idx[i_b_]][2] = bias2[i_b_, i_d_]
    else:
        for i_d_ in range(dofs_idx.shape[0]):
            dyn_info.dofs.act_bias[dofs_idx[i_d_]][0] = bias0[i_d_]
            dyn_info.dofs.act_bias[dofs_idx[i_d_]][1] = bias1[i_d_]
            dyn_info.dofs.act_bias[dofs_idx[i_d_]][2] = bias2[i_d_]


@qd.kernel(fastcache=True)
def kernel_set_dofs_force_range(
    dofs_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    lower: qd.types.ndarray(),
    upper: qd.types.ndarray(),
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    if qd.static(rigid_config.batch_dofs_info):
        for i_d_, i_b_ in qd.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            dyn_info.dofs.force_range[dofs_idx[i_d_], envs_idx[i_b_]][0] = lower[i_b_, i_d_]
            dyn_info.dofs.force_range[dofs_idx[i_d_], envs_idx[i_b_]][1] = upper[i_b_, i_d_]
    else:
        for i_d_ in range(dofs_idx.shape[0]):
            dyn_info.dofs.force_range[dofs_idx[i_d_]][0] = lower[i_d_]
            dyn_info.dofs.force_range[dofs_idx[i_d_]][1] = upper[i_d_]


@qd.kernel(fastcache=True)
def kernel_set_dofs_stiffness(
    dofs_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    stiffness: qd.types.ndarray(),
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    if qd.static(rigid_config.batch_dofs_info):
        for i_d_, i_b_ in qd.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            dyn_info.dofs.stiffness[dofs_idx[i_d_], envs_idx[i_b_]] = stiffness[i_b_, i_d_]
    else:
        for i_d_ in range(dofs_idx.shape[0]):
            dyn_info.dofs.stiffness[dofs_idx[i_d_]] = stiffness[i_d_]


@qd.kernel(fastcache=True)
def kernel_set_dofs_armature(
    dofs_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    armature: qd.types.ndarray(),
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    if qd.static(rigid_config.batch_dofs_info):
        for i_d_, i_b_ in qd.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            dyn_info.dofs.armature[dofs_idx[i_d_], envs_idx[i_b_]] = armature[i_b_, i_d_]
    else:
        for i_d_ in range(dofs_idx.shape[0]):
            dyn_info.dofs.armature[dofs_idx[i_d_]] = armature[i_d_]


@qd.kernel(fastcache=True)
def kernel_set_dofs_damping(
    dofs_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    damping: qd.types.ndarray(),
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    if qd.static(rigid_config.batch_dofs_info):
        for i_d_, i_b_ in qd.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            dyn_info.dofs.damping[dofs_idx[i_d_], envs_idx[i_b_]] = damping[i_b_, i_d_]
    else:
        for i_d_ in range(dofs_idx.shape[0]):
            dyn_info.dofs.damping[dofs_idx[i_d_]] = damping[i_d_]


@qd.kernel(fastcache=True)
def kernel_set_dofs_frictionloss(
    dofs_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    frictionloss: qd.types.ndarray(),
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    if qd.static(rigid_config.batch_dofs_info):
        for i_d_, i_b_ in qd.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            dyn_info.dofs.frictionloss[dofs_idx[i_d_], envs_idx[i_b_]] = frictionloss[i_b_, i_d_]
    else:
        for i_d_ in range(dofs_idx.shape[0]):
            dyn_info.dofs.frictionloss[dofs_idx[i_d_]] = frictionloss[i_d_]


@qd.kernel(fastcache=True)
def kernel_set_dofs_limit(
    dofs_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    lower: qd.types.ndarray(),
    upper: qd.types.ndarray(),
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    if qd.static(rigid_config.batch_dofs_info):
        for i_d_, i_b_ in qd.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            dyn_info.dofs.limit[dofs_idx[i_d_], envs_idx[i_b_]][0] = lower[i_b_, i_d_]
            dyn_info.dofs.limit[dofs_idx[i_d_], envs_idx[i_b_]][1] = upper[i_b_, i_d_]
    else:
        for i_d_ in range(dofs_idx.shape[0]):
            dyn_info.dofs.limit[dofs_idx[i_d_]][0] = lower[i_d_]
            dyn_info.dofs.limit[dofs_idx[i_d_]][1] = upper[i_d_]


@qd.kernel(fastcache=True)
def kernel_set_dofs_velocity(
    dofs_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    velocity: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_d_, i_b_ in qd.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        dyn_state.dofs.vel[dofs_idx[i_d_], envs_idx[i_b_]] = velocity[i_b_, i_d_]


@qd.kernel(fastcache=True)
def kernel_set_dofs_velocity_grad(
    dofs_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    velocity_grad: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d_, i_b_ in qd.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        velocity_grad[i_b_, i_d_] = dyn_state.dofs.vel.grad[dofs_idx[i_d_], envs_idx[i_b_]]
        dyn_state.dofs.vel.grad[dofs_idx[i_d_], envs_idx[i_b_]] = 0.0


@qd.kernel(fastcache=True)
def kernel_set_dofs_zero_velocity(
    dofs_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_d_, i_b_ in qd.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        dyn_state.dofs.vel[dofs_idx[i_d_], envs_idx[i_b_]] = 0.0


@qd.kernel(fastcache=True)
def kernel_set_dofs_position(
    dofs_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    position: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    n_entities = dyn_info.entities.link_start.shape[0]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_d_, i_b_ in qd.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        dyn_state.dofs.pos[dofs_idx[i_d_], envs_idx[i_b_]] = position[i_b_, i_d_]

    # Note that qpos must be updated, as dofs_state.pos is not used for actual IK.
    # TODO: Make this more efficient by only taking care of releavant qs/dofs.
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_e, i_b_ in qd.ndrange(n_entities, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        for i_l in range(dyn_info.entities.link_start[i_e], dyn_info.entities.link_end[i_e]):
            I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
            if dyn_info.links.n_dofs[I_l] == 0:
                continue

            dof_start = dyn_info.links.dof_start[I_l]
            q_start = dyn_info.links.q_start[I_l]

            i_j = dyn_info.links.joint_start[I_l]
            I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
            joint_type = dyn_info.joints.type[I_j]

            if joint_type == gs.JOINT_TYPE.FIXED:
                pass
            elif joint_type == gs.JOINT_TYPE.FREE:
                xyz = qd.Vector(
                    [
                        dyn_state.dofs.pos[0 + 3 + dof_start, i_b],
                        dyn_state.dofs.pos[1 + 3 + dof_start, i_b],
                        dyn_state.dofs.pos[2 + 3 + dof_start, i_b],
                    ],
                    dt=gs.qd_float,
                )
                quat = gu.qd_xyz_to_quat(xyz)

                for j in qd.static(range(3)):
                    rigid_info.qpos[j + q_start, i_b] = dyn_state.dofs.pos[j + dof_start, i_b]

                for j in qd.static(range(4)):
                    rigid_info.qpos[j + 3 + q_start, i_b] = quat[j]
            elif joint_type == gs.JOINT_TYPE.SPHERICAL:
                xyz = qd.Vector(
                    [
                        dyn_state.dofs.pos[0 + dof_start, i_b],
                        dyn_state.dofs.pos[1 + dof_start, i_b],
                        dyn_state.dofs.pos[2 + dof_start, i_b],
                    ],
                    dt=gs.qd_float,
                )
                quat = gu.qd_xyz_to_quat(xyz)
                for i_q_ in qd.static(range(4)):
                    i_q = q_start + i_q_
                    rigid_info.qpos[i_q, i_b] = quat[i_q_]
            else:  # (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC)
                for i_d_ in range(dyn_info.links.dof_end[I_l] - dof_start):
                    i_q = q_start + i_d_
                    i_d = dof_start + i_d_
                    rigid_info.qpos[i_q, i_b] = rigid_info.qpos0[i_q, i_b] + dyn_state.dofs.pos[i_d, i_b]


@qd.kernel(fastcache=True)
def kernel_control_dofs_force(
    dofs_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    force: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_d_, i_b_ in qd.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        dyn_state.dofs.ctrl_mode[dofs_idx[i_d_], envs_idx[i_b_]] = gs.CTRL_MODE.FORCE
        dyn_state.dofs.ctrl_force[dofs_idx[i_d_], envs_idx[i_b_]] = force[i_b_, i_d_]


@qd.kernel(fastcache=True)
def kernel_control_dofs_velocity(
    dofs_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    velocity: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_d_, i_b_ in qd.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        i_d = dofs_idx[i_d_]
        i_b = envs_idx[i_b_]

        dyn_state.dofs.ctrl_mode[i_d, i_b] = gs.CTRL_MODE.VELOCITY
        dyn_state.dofs.ctrl_vel[i_d, i_b] = velocity[i_b_, i_d_]


@qd.kernel(fastcache=True)
def kernel_control_dofs_position(
    dofs_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    position: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_d_, i_b_ in qd.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        i_d = dofs_idx[i_d_]
        i_b = envs_idx[i_b_]

        dyn_state.dofs.ctrl_mode[i_d, i_b] = gs.CTRL_MODE.POSITION
        dyn_state.dofs.ctrl_pos[i_d, i_b] = position[i_b_, i_d_]
        dyn_state.dofs.ctrl_vel[i_d, i_b] = 0.0


@qd.kernel(fastcache=True)
def kernel_control_dofs_position_velocity(
    dofs_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    position: qd.types.ndarray(),
    velocity: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_d_, i_b_ in qd.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        i_d = dofs_idx[i_d_]
        i_b = envs_idx[i_b_]

        dyn_state.dofs.ctrl_mode[i_d, i_b] = gs.CTRL_MODE.POSITION
        dyn_state.dofs.ctrl_pos[i_d, i_b] = position[i_b_, i_d_]
        dyn_state.dofs.ctrl_vel[i_d, i_b] = velocity[i_b_, i_d_]


@qd.kernel(fastcache=True)
def kernel_get_links_vel(
    links_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    tensor: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    rigid_config: qd.template(),
    ref: qd.template(),
):
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_l_, i_b_ in qd.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        # This is the velocity in world coordinates expressed at global com-position
        vel = dyn_state.links.cd_vel[links_idx[i_l_], envs_idx[i_b_]]  # entity's CoM

        # Translate to get the velocity expressed at a different position if necessary link-position
        if qd.static(ref == 1):  # link's CoM
            vel = vel + dyn_state.links.cd_ang[links_idx[i_l_], envs_idx[i_b_]].cross(
                dyn_state.links.i_pos[links_idx[i_l_], envs_idx[i_b_]]
            )
        if qd.static(ref == 2):  # link's origin
            vel = vel + dyn_state.links.cd_ang[links_idx[i_l_], envs_idx[i_b_]].cross(
                dyn_state.links.pos[links_idx[i_l_], envs_idx[i_b_]]
                - dyn_state.links.root_COM[links_idx[i_l_], envs_idx[i_b_]]
            )

        for j in qd.static(range(3)):
            tensor[i_b_, i_l_, j] = vel[j]


@qd.kernel(fastcache=True)
def kernel_get_links_acc(
    links_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    tensor: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_l_, i_b_ in qd.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        i_l = links_idx[i_l_]
        i_b = envs_idx[i_b_]

        # Compute links spatial acceleration expressed at links origin in world coordinates
        cpos = dyn_state.links.pos[i_l, i_b] - dyn_state.links.root_COM[i_l, i_b]
        acc_ang = dyn_state.links.cacc_ang[i_l, i_b]
        acc_lin = dyn_state.links.cacc_lin[i_l, i_b] + acc_ang.cross(cpos)

        # Compute links classical linear acceleration expressed at links origin in world coordinates
        ang = dyn_state.links.cd_ang[i_l, i_b]
        vel = dyn_state.links.cd_vel[i_l, i_b] + ang.cross(cpos)
        acc_classic_lin = acc_lin + ang.cross(vel)

        for j in qd.static(range(3)):
            tensor[i_b_, i_l_, j] = acc_classic_lin[j]


@qd.kernel(fastcache=True)
def kernel_get_dofs_control_force(
    dofs_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    tensor: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    # we need to compute control force here because this won't be computed until the next actual simulation step
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_d_, i_b_ in qd.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        i_d = dofs_idx[i_d_]
        i_b = envs_idx[i_b_]
        I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d
        force = gs.qd_float(0.0)
        if dyn_state.dofs.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.FORCE:
            force = dyn_state.dofs.ctrl_force[i_d, i_b]
        elif dyn_state.dofs.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.VELOCITY:
            force = -dyn_info.dofs.act_bias[I_d][2] * (dyn_state.dofs.ctrl_vel[i_d, i_b] - dyn_state.dofs.vel[i_d, i_b])
        elif dyn_state.dofs.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.POSITION:
            force = (
                dyn_info.dofs.act_gain[I_d] * (dyn_state.dofs.ctrl_pos[i_d, i_b] - dyn_state.dofs.pos[i_d, i_b])
                + dyn_info.dofs.act_bias[I_d][0]
                + (dyn_info.dofs.act_gain[I_d] + dyn_info.dofs.act_bias[I_d][1]) * dyn_state.dofs.pos[i_d, i_b]
                + dyn_info.dofs.act_bias[I_d][2] * (dyn_state.dofs.vel[i_d, i_b] - dyn_state.dofs.ctrl_vel[i_d, i_b])
            )
        tensor[i_b_, i_d_] = qd.math.clamp(force, dyn_info.dofs.force_range[I_d][0], dyn_info.dofs.force_range[I_d][1])


@qd.kernel(fastcache=True)
def kernel_set_drone_rpm(
    propellers_link_idx: qd.types.ndarray(),
    propellers_rpm: qd.types.ndarray(),
    propellers_spin: qd.types.ndarray(),
    KF: qd.float32,
    KM: qd.float32,
    dyn_state: array_class.DynState,
    rigid_config: qd.template(),
    invert: qd.i32,
):
    """
    Set the RPM of propellers of a drone entity.

    This method should only be called by drone entities.
    """
    n_propellers = propellers_link_idx.shape[0]
    _B = propellers_rpm.shape[0]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        for i_prop in range(n_propellers):
            i_l = propellers_link_idx[i_prop]

            force = qd.Vector([0.0, 0.0, propellers_rpm[i_b, i_prop] ** 2 * KF], dt=gs.qd_float)
            torque = qd.Vector(
                [0.0, 0.0, propellers_rpm[i_b, i_prop] ** 2 * KM * propellers_spin[i_prop]], dt=gs.qd_float
            )
            if invert:
                torque = -torque

            func_apply_link_external_force(i_l, i_b, force, dyn_state, 1, 1)
            func_apply_link_external_torque(i_l, i_b, torque, dyn_state, 1, 1)


@qd.kernel(fastcache=True)
def kernel_update_drone_propeller_vgeoms(
    propellers_vgeom_idxs: qd.types.ndarray(),
    propellers_revs: qd.types.ndarray(),
    propellers_spin: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """
    Update the angle of the vgeom in the propellers of a drone entity.
    """
    EPS = rigid_info.EPS[None]

    n_propellers = propellers_vgeom_idxs.shape[0]
    _B = propellers_revs.shape[1]

    for i_pp, i_b in qd.ndrange(n_propellers, _B):
        i_vg = propellers_vgeom_idxs[i_pp]
        rad = propellers_revs[i_pp, i_b] * propellers_spin[i_pp] * rigid_info.substep_dt[None] * qd.math.pi / 30.0
        dyn_state.vgeoms.quat[i_vg, i_b] = gu.qd_transform_quat_by_quat(
            gu.qd_rotvec_to_quat(qd.Vector([0.0, 0.0, rad], dt=gs.qd_float), EPS), dyn_state.vgeoms.quat[i_vg, i_b]
        )


@qd.kernel(fastcache=True)
def kernel_set_geom_friction(geoms_idx: qd.i32, dyn_info: array_class.DynInfo, friction: qd.f32):
    dyn_info.geoms.friction[geoms_idx] = friction


@qd.kernel(fastcache=True)
def kernel_set_geoms_friction(
    geoms_idx: qd.types.ndarray(),
    friction: qd.types.ndarray(),
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_g_ in range(geoms_idx.shape[0]):
        dyn_info.geoms.friction[geoms_idx[i_g_]] = friction[i_g_]


@qd.kernel(fastcache=True)
def kernel_set_vverts(
    vvert_start: qd.i32,
    envs_idx: qd.types.ndarray(),
    vverts: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    rigid_config: qd.template(),
):
    n_envs_in = envs_idx.shape[0]
    n_vverts_in = vverts.shape[1]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b_, i_vv_ in qd.ndrange(n_envs_in, n_vverts_in):
        i_b = envs_idx[i_b_]
        i_vv = vvert_start + i_vv_
        for j in qd.static(range(3)):
            dyn_state.vverts.pos[i_vv, i_b][j] = vverts[i_b_, i_vv_, j]
