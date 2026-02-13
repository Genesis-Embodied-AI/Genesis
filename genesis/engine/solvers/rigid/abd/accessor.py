"""
Rigid solver control, getter, and setter kernel functions.

This module contains Taichi kernel functions for controlling rigid body simulations,
including state getters/setters, link position/quaternion manipulation, DOF control,
and drone-specific operations.

These functions are used by the RigidSolver class to interface with the Taichi
data structures for rigid body dynamics simulation.
"""

import quadrants as ti

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class
from .misc import func_apply_link_external_force, func_apply_link_external_torque


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_get_state(
    qpos: ti.types.ndarray(),
    vel: ti.types.ndarray(),
    acc: ti.types.ndarray(),
    links_pos: ti.types.ndarray(),
    links_quat: ti.types.ndarray(),
    i_pos_shift: ti.types.ndarray(),
    mass_shift: ti.types.ndarray(),
    friction_ratio: ti.types.ndarray(),
    links_state: array_class.LinksState,
    dofs_state: array_class.DofsState,
    geoms_state: array_class.GeomsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    n_qs = qpos.shape[1]
    n_dofs = vel.shape[1]
    n_links = links_pos.shape[1]
    n_geoms = friction_ratio.shape[1]
    _B = qpos.shape[0]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_q, i_b in ti.ndrange(n_qs, _B):
        qpos[i_b, i_q] = rigid_global_info.qpos[i_q, i_b]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in ti.ndrange(n_dofs, _B):
        vel[i_b, i_d] = dofs_state.vel[i_d, i_b]
        acc[i_b, i_d] = dofs_state.acc[i_d, i_b]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b in ti.ndrange(n_links, _B):
        for j in ti.static(range(3)):
            links_pos[i_b, i_l, j] = links_state.pos[i_l, i_b][j]
            i_pos_shift[i_b, i_l, j] = links_state.i_pos_shift[i_l, i_b][j]
        for j in ti.static(range(4)):
            links_quat[i_b, i_l, j] = links_state.quat[i_l, i_b][j]
        mass_shift[i_b, i_l] = links_state.mass_shift[i_l, i_b]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b in ti.ndrange(n_geoms, _B):
        friction_ratio[i_b, i_l] = geoms_state.friction_ratio[i_l, i_b]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_state(
    qpos: ti.types.ndarray(),
    dofs_vel: ti.types.ndarray(),
    dofs_acc: ti.types.ndarray(),
    links_pos: ti.types.ndarray(),
    links_quat: ti.types.ndarray(),
    i_pos_shift: ti.types.ndarray(),
    mass_shift: ti.types.ndarray(),
    friction_ratio: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    links_state: array_class.LinksState,
    dofs_state: array_class.DofsState,
    geoms_state: array_class.GeomsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    n_qs = qpos.shape[1]
    n_dofs = dofs_vel.shape[1]
    n_links = links_pos.shape[1]
    n_geoms = friction_ratio.shape[1]
    _B = envs_idx.shape[0]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_q, i_b_ in ti.ndrange(n_qs, _B):
        rigid_global_info.qpos[i_q, envs_idx[i_b_]] = qpos[envs_idx[i_b_], i_q]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b_ in ti.ndrange(n_dofs, _B):
        dofs_state.vel[i_d, envs_idx[i_b_]] = dofs_vel[envs_idx[i_b_], i_d]
        dofs_state.acc[i_d, envs_idx[i_b_]] = dofs_acc[envs_idx[i_b_], i_d]
        dofs_state.ctrl_force[i_d, envs_idx[i_b_]] = gs.ti_float(0.0)
        dofs_state.ctrl_mode[i_d, envs_idx[i_b_]] = gs.CTRL_MODE.FORCE

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b_ in ti.ndrange(n_links, _B):
        for j in ti.static(range(3)):
            links_state.pos[i_l, envs_idx[i_b_]][j] = links_pos[envs_idx[i_b_], i_l, j]
            links_state.i_pos_shift[i_l, envs_idx[i_b_]][j] = i_pos_shift[envs_idx[i_b_], i_l, j]
            links_state.cfrc_applied_vel[i_l, envs_idx[i_b_]][j] = gs.ti_float(0.0)
            links_state.cfrc_applied_ang[i_l, envs_idx[i_b_]][j] = gs.ti_float(0.0)
        for j in ti.static(range(4)):
            links_state.quat[i_l, envs_idx[i_b_]][j] = links_quat[envs_idx[i_b_], i_l, j]
        links_state.mass_shift[i_l, envs_idx[i_b_]] = mass_shift[envs_idx[i_b_], i_l]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b_ in ti.ndrange(n_geoms, _B):
        geoms_state.friction_ratio[i_l, envs_idx[i_b_]] = friction_ratio[envs_idx[i_b_], i_l]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_get_state_grad(
    qpos_grad: ti.types.ndarray(),
    vel_grad: ti.types.ndarray(),
    links_pos_grad: ti.types.ndarray(),
    links_quat_grad: ti.types.ndarray(),
    links_state: array_class.LinksState,
    dofs_state: array_class.DofsState,
    geoms_state: array_class.GeomsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    n_qs = qpos_grad.shape[1]
    n_dofs = vel_grad.shape[1]
    n_links = links_pos_grad.shape[1]
    _B = qpos_grad.shape[0]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_q, i_b in ti.ndrange(n_qs, _B):
        ti.atomic_add(rigid_global_info.qpos.grad[i_q, i_b], qpos_grad[i_b, i_q])

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in ti.ndrange(n_dofs, _B):
        ti.atomic_add(dofs_state.vel.grad[i_d, i_b], vel_grad[i_b, i_d])

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b in ti.ndrange(n_links, _B):
        for j in ti.static(range(3)):
            ti.atomic_add(links_state.pos.grad[i_l, i_b][j], links_pos_grad[i_b, i_l, j])
        for j in ti.static(range(4)):
            ti.atomic_add(links_state.quat.grad[i_l, i_b][j], links_quat_grad[i_b, i_l, j])


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_links_pos(
    relative: ti.i32,
    pos: ti.types.ndarray(),
    links_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_l = links_idx[i_l_]
        I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

        if links_info.parent_idx[I_l] == -1 and links_info.is_fixed[I_l]:
            for j in ti.static(range(3)):
                links_state.pos[i_l, i_b][j] = pos[i_b_, i_l_, j]
            if relative:
                for j in ti.static(range(3)):
                    links_state.pos[i_l, i_b][j] = links_state.pos[i_l, i_b][j] + links_info.pos[I_l][j]
        else:
            q_start = links_info.q_start[I_l]
            for j in ti.static(range(3)):
                rigid_global_info.qpos[q_start + j, i_b] = pos[i_b_, i_l_, j]
            if relative:
                for j in ti.static(range(3)):
                    rigid_global_info.qpos[q_start + j, i_b] = (
                        rigid_global_info.qpos[q_start + j, i_b] + rigid_global_info.qpos0[q_start + j, i_b]
                    )


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_wake_up_entities_by_links(
    links_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    entities_state: array_class.EntitiesState,
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    geoms_state: array_class.GeomsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    """Wake up entities that own the specified links by setting their hibernated flags to False."""
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_l = links_idx[i_l_]
        I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
        i_e = links_info.entity_idx[I_l]

        # Wake up the entity and all its components
        if entities_state.hibernated[i_e, i_b]:
            entities_state.hibernated[i_e, i_b] = False

            # Add entity to awake_entities list
            n_awake = ti.atomic_add(rigid_global_info.n_awake_entities[i_b], 1)
            rigid_global_info.awake_entities[n_awake, i_b] = i_e

            # Wake up all links of this entity and add to awake_links
            for i_l2 in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
                links_state.hibernated[i_l2, i_b] = False
                n_awake_links = ti.atomic_add(rigid_global_info.n_awake_links[i_b], 1)
                rigid_global_info.awake_links[n_awake_links, i_b] = i_l2

            # Wake up all DOFs of this entity and add to awake_dofs
            for i_d in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                dofs_state.hibernated[i_d, i_b] = False
                n_awake_dofs = ti.atomic_add(rigid_global_info.n_awake_dofs[i_b], 1)
                rigid_global_info.awake_dofs[n_awake_dofs, i_b] = i_d

            # Wake up all geoms of this entity
            for i_g in range(entities_info.geom_start[i_e], entities_info.geom_end[i_e]):
                geoms_state.hibernated[i_g, i_b] = False


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_links_pos_grad(
    relative: ti.i32,
    pos_grad: ti.types.ndarray(),
    links_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_l = links_idx[i_l_]
        I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

        if links_info.parent_idx[I_l] == -1 and links_info.is_fixed[I_l]:
            for j in ti.static(range(3)):
                pos_grad[i_b_, i_l_, j] = links_state.pos.grad[i_l, i_b][j]
                links_state.pos.grad[i_l, i_b][j] = 0.0
        else:
            q_start = links_info.q_start[I_l]
            for j in ti.static(range(3)):
                pos_grad[i_b_, i_l_, j] = rigid_global_info.qpos.grad[q_start + j, i_b]
                rigid_global_info.qpos.grad[q_start + j, i_b] = 0.0


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_links_quat(
    relative: ti.i32,
    quat: ti.types.ndarray(),
    links_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_l = links_idx[i_l_]
        I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

        if relative:
            quat_ = ti.Vector(
                [
                    quat[i_b_, i_l_, 0],
                    quat[i_b_, i_l_, 1],
                    quat[i_b_, i_l_, 2],
                    quat[i_b_, i_l_, 3],
                ],
                dt=gs.ti_float,
            )
            if links_info.parent_idx[I_l] == -1 and links_info.is_fixed[I_l]:
                links_state.quat[i_l, i_b] = gu.ti_transform_quat_by_quat(links_info.quat[I_l], quat_)
            else:
                q_start = links_info.q_start[I_l]
                quat0 = ti.Vector(
                    [
                        rigid_global_info.qpos0[q_start + 3, i_b],
                        rigid_global_info.qpos0[q_start + 4, i_b],
                        rigid_global_info.qpos0[q_start + 5, i_b],
                        rigid_global_info.qpos0[q_start + 6, i_b],
                    ],
                    dt=gs.ti_float,
                )
                quat_ = gu.ti_transform_quat_by_quat(quat0, quat_)
                for j in ti.static(range(4)):
                    rigid_global_info.qpos[q_start + j + 3, i_b] = quat_[j]
        else:
            if links_info.parent_idx[I_l] == -1 and links_info.is_fixed[I_l]:
                for j in ti.static(range(4)):
                    links_state.quat[i_l, i_b][j] = quat[i_b_, i_l_, j]
            else:
                q_start = links_info.q_start[I_l]
                for j in ti.static(range(4)):
                    rigid_global_info.qpos[q_start + j + 3, i_b] = quat[i_b_, i_l_, j]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_links_quat_grad(
    relative: ti.i32,
    quat_grad: ti.types.ndarray(),
    links_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_l = links_idx[i_l_]
        I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

        if links_info.parent_idx[I_l] == -1 and links_info.is_fixed[I_l]:
            for j in ti.static(range(4)):
                quat_grad[i_b_, i_l_, j] = links_state.quat.grad[i_l, i_b][j]
                links_state.quat.grad[i_l, i_b][j] = 0.0
        else:
            q_start = links_info.q_start[I_l]
            for j in ti.static(range(4)):
                quat_grad[i_b_, i_l_, j] = rigid_global_info.qpos.grad[q_start + j + 3, i_b]
                rigid_global_info.qpos.grad[q_start + j + 3, i_b] = 0.0


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_links_mass_shift(
    mass: ti.types.ndarray(),
    links_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    links_state: array_class.LinksState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        links_state.mass_shift[links_idx[i_l_], envs_idx[i_b_]] = mass[i_b_, i_l_]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_links_COM_shift(
    com: ti.types.ndarray(),
    links_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    links_state: array_class.LinksState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        for j in ti.static(range(3)):
            links_state.i_pos_shift[links_idx[i_l_], envs_idx[i_b_]][j] = com[i_b_, i_l_, j]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_links_inertial_mass(
    inertial_mass: ti.types.ndarray(),
    links_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    links_info: array_class.LinksInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    if ti.static(static_rigid_sim_config.batch_links_info):
        for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
            links_info.inertial_mass[links_idx[i_l_], envs_idx[i_b_]] = inertial_mass[i_b_, i_l_]
    else:
        for i_l_ in range(links_idx.shape[0]):
            links_info.inertial_mass[links_idx[i_l_]] = inertial_mass[i_l_]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_geoms_friction_ratio(
    friction_ratio: ti.types.ndarray(),
    geoms_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    geoms_state: array_class.GeomsState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_g_, i_b_ in ti.ndrange(geoms_idx.shape[0], envs_idx.shape[0]):
        geoms_state.friction_ratio[geoms_idx[i_g_], envs_idx[i_b_]] = friction_ratio[i_b_, i_g_]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_qpos(
    qpos: ti.types.ndarray(),
    qs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_q_, i_b_ in ti.ndrange(qs_idx.shape[0], envs_idx.shape[0]):
        rigid_global_info.qpos[qs_idx[i_q_], envs_idx[i_b_]] = qpos[i_b_, i_q_]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_global_sol_params(
    sol_params: ti.types.ndarray(),
    geoms_info: array_class.GeomsInfo,
    joints_info: array_class.JointsInfo,
    equalities_info: array_class.EqualitiesInfo,
    static_rigid_sim_config: ti.template(),
):
    n_geoms = geoms_info.sol_params.shape[0]
    n_joints = joints_info.sol_params.shape[0]
    n_equalities = equalities_info.sol_params.shape[0]
    _B = equalities_info.sol_params.shape[1]

    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_g in range(n_geoms):
        for j in ti.static(range(7)):
            geoms_info.sol_params[i_g][j] = sol_params[j]

    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_j, i_b in ti.ndrange(n_joints, _B):
        I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
        for j in ti.static(range(7)):
            joints_info.sol_params[I_j][j] = sol_params[j]

    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_eq, i_b in ti.ndrange(n_equalities, _B):
        for j in ti.static(range(7)):
            equalities_info.sol_params[i_eq, i_b][j] = sol_params[j]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_sol_params(
    constraint_type: ti.template(),
    sol_params: ti.types.ndarray(),
    inputs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    geoms_info: array_class.GeomsInfo,
    joints_info: array_class.JointsInfo,
    equalities_info: array_class.EqualitiesInfo,
    static_rigid_sim_config: ti.template(),
):
    if ti.static(constraint_type == 0):  # geometries
        ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
        for i_g_ in range(inputs_idx.shape[0]):
            for j in ti.static(range(7)):
                geoms_info.sol_params[inputs_idx[i_g_]][j] = sol_params[i_g_, j]
    if ti.static(constraint_type == 1):  # joints
        ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
        if ti.static(static_rigid_sim_config.batch_joints_info):
            for i_j_, i_b_ in ti.ndrange(inputs_idx.shape[0], envs_idx.shape[0]):
                for j in ti.static(range(7)):
                    joints_info.sol_params[inputs_idx[i_j_], envs_idx[i_b_]][j] = sol_params[i_b_, i_j_, j]
        else:
            for i_j_ in range(inputs_idx.shape[0]):
                for j in ti.static(range(7)):
                    joints_info.sol_params[inputs_idx[i_j_]][j] = sol_params[i_j_, j]
    if ti.static(constraint_type == 2):  # equalities
        ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
        for i_eq_, i_b_ in ti.ndrange(inputs_idx.shape[0], envs_idx.shape[0]):
            for j in ti.static(range(7)):
                equalities_info.sol_params[inputs_idx[i_eq_], envs_idx[i_b_]][j] = sol_params[i_b_, i_eq_, j]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_dofs_kp(
    kp: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_info: array_class.DofsInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    if ti.static(static_rigid_sim_config.batch_dofs_info):
        for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            dofs_info.kp[dofs_idx[i_d_], envs_idx[i_b_]] = kp[i_b_, i_d_]
    else:
        for i_d_ in range(dofs_idx.shape[0]):
            dofs_info.kp[dofs_idx[i_d_]] = kp[i_d_]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_dofs_kv(
    kv: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_info: array_class.DofsInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    if ti.static(static_rigid_sim_config.batch_dofs_info):
        for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            dofs_info.kv[dofs_idx[i_d_], envs_idx[i_b_]] = kv[i_b_, i_d_]
    else:
        for i_d_ in range(dofs_idx.shape[0]):
            dofs_info.kv[dofs_idx[i_d_]] = kv[i_d_]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_dofs_force_range(
    lower: ti.types.ndarray(),
    upper: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_info: array_class.DofsInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    if ti.static(static_rigid_sim_config.batch_dofs_info):
        for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            dofs_info.force_range[dofs_idx[i_d_], envs_idx[i_b_]][0] = lower[i_b_, i_d_]
            dofs_info.force_range[dofs_idx[i_d_], envs_idx[i_b_]][1] = upper[i_b_, i_d_]
    else:
        for i_d_ in range(dofs_idx.shape[0]):
            dofs_info.force_range[dofs_idx[i_d_]][0] = lower[i_d_]
            dofs_info.force_range[dofs_idx[i_d_]][1] = upper[i_d_]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_dofs_stiffness(
    stiffness: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_info: array_class.DofsInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    if ti.static(static_rigid_sim_config.batch_dofs_info):
        for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            dofs_info.stiffness[dofs_idx[i_d_], envs_idx[i_b_]] = stiffness[i_b_, i_d_]
    else:
        for i_d_ in range(dofs_idx.shape[0]):
            dofs_info.stiffness[dofs_idx[i_d_]] = stiffness[i_d_]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_dofs_armature(
    armature: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_info: array_class.DofsInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    if ti.static(static_rigid_sim_config.batch_dofs_info):
        for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            dofs_info.armature[dofs_idx[i_d_], envs_idx[i_b_]] = armature[i_b_, i_d_]
    else:
        for i_d_ in range(dofs_idx.shape[0]):
            dofs_info.armature[dofs_idx[i_d_]] = armature[i_d_]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_dofs_damping(
    damping: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_info: array_class.DofsInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    if ti.static(static_rigid_sim_config.batch_dofs_info):
        for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            dofs_info.damping[dofs_idx[i_d_], envs_idx[i_b_]] = damping[i_b_, i_d_]
    else:
        for i_d_ in range(dofs_idx.shape[0]):
            dofs_info.damping[dofs_idx[i_d_]] = damping[i_d_]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_dofs_frictionloss(
    frictionloss: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_info: array_class.DofsInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    if ti.static(static_rigid_sim_config.batch_dofs_info):
        for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            dofs_info.frictionloss[dofs_idx[i_d_], envs_idx[i_b_]] = frictionloss[i_b_, i_d_]
    else:
        for i_d_ in range(dofs_idx.shape[0]):
            dofs_info.frictionloss[dofs_idx[i_d_]] = frictionloss[i_d_]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_dofs_limit(
    lower: ti.types.ndarray(),
    upper: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_info: array_class.DofsInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    if ti.static(static_rigid_sim_config.batch_dofs_info):
        for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            dofs_info.limit[dofs_idx[i_d_], envs_idx[i_b_]][0] = lower[i_b_, i_d_]
            dofs_info.limit[dofs_idx[i_d_], envs_idx[i_b_]][1] = upper[i_b_, i_d_]
    else:
        for i_d_ in range(dofs_idx.shape[0]):
            dofs_info.limit[dofs_idx[i_d_]][0] = lower[i_d_]
            dofs_info.limit[dofs_idx[i_d_]][1] = upper[i_d_]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_dofs_velocity(
    velocity: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_state: array_class.DofsState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        dofs_state.vel[dofs_idx[i_d_], envs_idx[i_b_]] = velocity[i_b_, i_d_]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_dofs_velocity_grad(
    velocity_grad: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_state: array_class.DofsState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        velocity_grad[i_b_, i_d_] = dofs_state.vel.grad[dofs_idx[i_d_], envs_idx[i_b_]]
        dofs_state.vel.grad[dofs_idx[i_d_], envs_idx[i_b_]] = 0.0


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_dofs_zero_velocity(
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_state: array_class.DofsState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        dofs_state.vel[dofs_idx[i_d_], envs_idx[i_b_]] = 0.0


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_dofs_position(
    position: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_state: array_class.DofsState,
    links_info: array_class.LinksInfo,
    joints_info: array_class.JointsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    n_entities = entities_info.link_start.shape[0]

    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        dofs_state.pos[dofs_idx[i_d_], envs_idx[i_b_]] = position[i_b_, i_d_]

    # Note that qpos must be updated, as dofs_state.pos is not used for actual IK.
    # TODO: Make this more efficient by only taking care of releavant qs/dofs.
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_e, i_b_ in ti.ndrange(n_entities, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
            if links_info.n_dofs[I_l] == 0:
                continue

            dof_start = links_info.dof_start[I_l]
            q_start = links_info.q_start[I_l]

            i_j = links_info.joint_start[I_l]
            I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
            joint_type = joints_info.type[I_j]

            if joint_type == gs.JOINT_TYPE.FIXED:
                pass
            elif joint_type == gs.JOINT_TYPE.FREE:
                xyz = ti.Vector(
                    [
                        dofs_state.pos[0 + 3 + dof_start, i_b],
                        dofs_state.pos[1 + 3 + dof_start, i_b],
                        dofs_state.pos[2 + 3 + dof_start, i_b],
                    ],
                    dt=gs.ti_float,
                )
                quat = gu.ti_xyz_to_quat(xyz)

                for j in ti.static(range(3)):
                    rigid_global_info.qpos[j + q_start, i_b] = dofs_state.pos[j + dof_start, i_b]

                for j in ti.static(range(4)):
                    rigid_global_info.qpos[j + 3 + q_start, i_b] = quat[j]
            elif joint_type == gs.JOINT_TYPE.SPHERICAL:
                xyz = ti.Vector(
                    [
                        dofs_state.pos[0 + dof_start, i_b],
                        dofs_state.pos[1 + dof_start, i_b],
                        dofs_state.pos[2 + dof_start, i_b],
                    ],
                    dt=gs.ti_float,
                )
                quat = gu.ti_xyz_to_quat(xyz)
                for i_q_ in ti.static(range(4)):
                    i_q = q_start + i_q_
                    rigid_global_info.qpos[i_q, i_b] = quat[i_q_]
            else:  # (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC)
                for i_d_ in range(links_info.dof_end[I_l] - dof_start):
                    i_q = q_start + i_d_
                    i_d = dof_start + i_d_
                    rigid_global_info.qpos[i_q, i_b] = rigid_global_info.qpos0[i_q, i_b] + dofs_state.pos[i_d, i_b]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_control_dofs_force(
    force: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_state: array_class.DofsState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        dofs_state.ctrl_mode[dofs_idx[i_d_], envs_idx[i_b_]] = gs.CTRL_MODE.FORCE
        dofs_state.ctrl_force[dofs_idx[i_d_], envs_idx[i_b_]] = force[i_b_, i_d_]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_control_dofs_velocity(
    velocity: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_state: array_class.DofsState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        i_d = dofs_idx[i_d_]
        i_b = envs_idx[i_b_]

        dofs_state.ctrl_mode[i_d, i_b] = gs.CTRL_MODE.VELOCITY
        dofs_state.ctrl_vel[i_d, i_b] = velocity[i_b_, i_d_]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_control_dofs_position(
    position: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_state: array_class.DofsState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        i_d = dofs_idx[i_d_]
        i_b = envs_idx[i_b_]

        dofs_state.ctrl_mode[i_d, i_b] = gs.CTRL_MODE.POSITION
        dofs_state.ctrl_pos[i_d, i_b] = position[i_b_, i_d_]
        dofs_state.ctrl_vel[i_d, i_b] = 0.0


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_control_dofs_position_velocity(
    position: ti.types.ndarray(),
    velocity: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_state: array_class.DofsState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        i_d = dofs_idx[i_d_]
        i_b = envs_idx[i_b_]

        dofs_state.ctrl_mode[i_d, i_b] = gs.CTRL_MODE.POSITION
        dofs_state.ctrl_pos[i_d, i_b] = position[i_b_, i_d_]
        dofs_state.ctrl_vel[i_d, i_b] = velocity[i_b_, i_d_]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_get_links_vel(
    tensor: ti.types.ndarray(),
    links_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    ref: ti.template(),
    links_state: array_class.LinksState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        # This is the velocity in world coordinates expressed at global com-position
        vel = links_state.cd_vel[links_idx[i_l_], envs_idx[i_b_]]  # entity's CoM

        # Translate to get the velocity expressed at a different position if necessary link-position
        if ti.static(ref == 1):  # link's CoM
            vel = vel + links_state.cd_ang[links_idx[i_l_], envs_idx[i_b_]].cross(
                links_state.i_pos[links_idx[i_l_], envs_idx[i_b_]]
            )
        if ti.static(ref == 2):  # link's origin
            vel = vel + links_state.cd_ang[links_idx[i_l_], envs_idx[i_b_]].cross(
                links_state.pos[links_idx[i_l_], envs_idx[i_b_]] - links_state.root_COM[links_idx[i_l_], envs_idx[i_b_]]
            )

        for j in ti.static(range(3)):
            tensor[i_b_, i_l_, j] = vel[j]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_get_links_acc(
    tensor: ti.types.ndarray(),
    links_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    links_state: array_class.LinksState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        i_l = links_idx[i_l_]
        i_b = envs_idx[i_b_]

        # Compute links spatial acceleration expressed at links origin in world coordinates
        cpos = links_state.pos[i_l, i_b] - links_state.root_COM[i_l, i_b]
        acc_ang = links_state.cacc_ang[i_l, i_b]
        acc_lin = links_state.cacc_lin[i_l, i_b] + acc_ang.cross(cpos)

        # Compute links classical linear acceleration expressed at links origin in world coordinates
        ang = links_state.cd_ang[i_l, i_b]
        vel = links_state.cd_vel[i_l, i_b] + ang.cross(cpos)
        acc_classic_lin = acc_lin + ang.cross(vel)

        for j in ti.static(range(3)):
            tensor[i_b_, i_l_, j] = acc_classic_lin[j]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_get_dofs_control_force(
    tensor: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    static_rigid_sim_config: ti.template(),
):
    # we need to compute control force here because this won't be computed until the next actual simulation step
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        i_d = dofs_idx[i_d_]
        i_b = envs_idx[i_b_]
        I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
        force = gs.ti_float(0.0)
        if dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.FORCE:
            force = dofs_state.ctrl_force[i_d, i_b]
        elif dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.VELOCITY:
            force = dofs_info.kv[I_d] * (dofs_state.ctrl_vel[i_d, i_b] - dofs_state.vel[i_d, i_b])
        elif dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.POSITION:
            force = dofs_info.kp[I_d] * (dofs_state.ctrl_pos[i_d, i_b] - dofs_state.pos[i_d, i_b]) + dofs_info.kv[
                I_d
            ] * (dofs_state.ctrl_vel[i_d, i_b] - dofs_state.vel[i_d, i_b])
        tensor[i_b_, i_d_] = ti.math.clamp(
            force,
            dofs_info.force_range[I_d][0],
            dofs_info.force_range[I_d][1],
        )


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_drone_rpm(
    propellers_link_idx: ti.types.ndarray(),
    propellers_rpm: ti.types.ndarray(),
    propellers_spin: ti.types.ndarray(),
    KF: ti.float32,
    KM: ti.float32,
    invert: ti.i32,
    links_state: array_class.LinksState,
    static_rigid_sim_config: ti.template(),
):
    """
    Set the RPM of propellers of a drone entity.

    This method should only be called by drone entities.
    """
    n_propellers = propellers_link_idx.shape[0]
    _B = propellers_rpm.shape[0]

    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        for i_prop in range(n_propellers):
            i_l = propellers_link_idx[i_prop]

            force = ti.Vector([0.0, 0.0, propellers_rpm[i_b, i_prop] ** 2 * KF], dt=gs.ti_float)
            torque = ti.Vector(
                [0.0, 0.0, propellers_rpm[i_b, i_prop] ** 2 * KM * propellers_spin[i_prop]], dt=gs.ti_float
            )
            if invert:
                torque = -torque

            func_apply_link_external_force(force, i_l, i_b, 1, 1, links_state)
            func_apply_link_external_torque(torque, i_l, i_b, 1, 1, links_state)


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_update_drone_propeller_vgeoms(
    propellers_vgeom_idxs: ti.types.ndarray(),
    propellers_revs: ti.types.ndarray(),
    propellers_spin: ti.types.ndarray(),
    vgeoms_state: array_class.VGeomsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    """
    Update the angle of the vgeom in the propellers of a drone entity.
    """
    EPS = rigid_global_info.EPS[None]

    n_propellers = propellers_vgeom_idxs.shape[0]
    _B = propellers_revs.shape[1]

    for i_pp, i_b in ti.ndrange(n_propellers, _B):
        i_vg = propellers_vgeom_idxs[i_pp]
        rad = (
            propellers_revs[i_pp, i_b] * propellers_spin[i_pp] * rigid_global_info.substep_dt[None] * ti.math.pi / 30.0
        )
        vgeoms_state.quat[i_vg, i_b] = gu.ti_transform_quat_by_quat(
            gu.ti_rotvec_to_quat(ti.Vector([0.0, 0.0, rad], dt=gs.ti_float), EPS),
            vgeoms_state.quat[i_vg, i_b],
        )


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_geom_friction(geoms_idx: ti.i32, friction: ti.f32, geoms_info: array_class.GeomsInfo):
    geoms_info.friction[geoms_idx] = friction


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_geoms_friction(
    friction: ti.types.ndarray(),
    geoms_idx: ti.types.ndarray(),
    geoms_info: array_class.GeomsInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_g_ in range(geoms_idx.shape[0]):
        geoms_info.friction[geoms_idx[i_g_]] = friction[i_g_]
