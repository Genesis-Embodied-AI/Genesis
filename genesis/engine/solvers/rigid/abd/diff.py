"""
Backward pass functions for the rigid body solver.

This module contains functions used during the backward pass (gradient computation)
of the rigid body simulation. These functions handle:
- Copying state between next and current time steps
- Saving and loading adjoint cache for gradient computation
- Preparing and beginning backward substeps
- Gradient validity checking
- Cartesian space copying for adjoint computation
- Acceleration copying and dq integration

These functions are extracted from the main rigid_solver module to improve
code organization and maintainability.
"""

import quadrants as ti

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class
from .forward_kinematics import func_update_cartesian_space


@ti.func
def func_copy_next_to_curr(
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    errno: array_class.V_ANNOTATION,
):
    n_qs = rigid_global_info.qpos.shape[0]
    n_dofs = dofs_state.vel.shape[0]
    _B = dofs_state.vel.shape[1]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        # Prevent nan propagation
        is_valid = True
        for i_d in range(n_dofs):
            e = dofs_state.vel_next[i_d, i_b]
            is_valid &= not ti.math.isnan(e)
        for i_q in range(n_qs):
            e = rigid_global_info.qpos_next[i_q, i_b]
            is_valid &= not ti.math.isnan(e)

        if is_valid:
            for i_d in range(n_dofs):
                dofs_state.vel[i_d, i_b] = dofs_state.vel_next[i_d, i_b]

            for i_q in range(n_qs):
                rigid_global_info.qpos[i_q, i_b] = rigid_global_info.qpos_next[i_q, i_b]
        else:
            errno[i_b] = errno[i_b] | array_class.ErrorCode.INVALID_ACC_NAN


@ti.func
def func_copy_next_to_curr_grad(
    f: ti.int32,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    rigid_adjoint_cache: array_class.RigidAdjointCache,
    static_rigid_sim_config: ti.template(),
):
    n_dofs = dofs_state.vel.shape[0]
    n_qs = rigid_global_info.qpos.shape[0]
    _B = dofs_state.vel.shape[1]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in ti.ndrange(n_dofs, _B):
        dofs_state.vel_next.grad[i_d, i_b] = dofs_state.vel.grad[i_d, i_b]
        dofs_state.vel.grad[i_d, i_b] = 0.0
        dofs_state.vel[i_d, i_b] = rigid_adjoint_cache.dofs_vel[f, i_d, i_b]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_q, i_b in ti.ndrange(n_qs, _B):
        rigid_global_info.qpos_next.grad[i_q, i_b] = rigid_global_info.qpos.grad[i_q, i_b]
        rigid_global_info.qpos.grad[i_q, i_b] = 0.0
        rigid_global_info.qpos[i_q, i_b] = rigid_adjoint_cache.qpos[f, i_q, i_b]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_save_adjoint_cache(
    f: ti.int32,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    rigid_adjoint_cache: array_class.RigidAdjointCache,
    static_rigid_sim_config: ti.template(),
):
    func_save_adjoint_cache(f, dofs_state, rigid_global_info, rigid_adjoint_cache, static_rigid_sim_config)


@ti.func
def func_save_adjoint_cache(
    f: ti.int32,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    rigid_adjoint_cache: array_class.RigidAdjointCache,
    static_rigid_sim_config: ti.template(),
):
    n_dofs = dofs_state.vel.shape[0]
    n_qs = rigid_global_info.qpos.shape[0]
    _B = dofs_state.vel.shape[1]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in ti.ndrange(n_dofs, _B):
        rigid_adjoint_cache.dofs_vel[f, i_d, i_b] = dofs_state.vel[i_d, i_b]
        rigid_adjoint_cache.dofs_acc[f, i_d, i_b] = dofs_state.acc[i_d, i_b]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_q, i_b in ti.ndrange(n_qs, _B):
        rigid_adjoint_cache.qpos[f, i_q, i_b] = rigid_global_info.qpos[i_q, i_b]


@ti.func
def func_load_adjoint_cache(
    f: ti.int32,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    rigid_adjoint_cache: array_class.RigidAdjointCache,
    static_rigid_sim_config: ti.template(),
):
    n_dofs = dofs_state.vel.shape[0]
    n_qs = rigid_global_info.qpos.shape[0]
    _B = dofs_state.vel.shape[1]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in ti.ndrange(n_dofs, _B):
        dofs_state.vel[i_d, i_b] = rigid_adjoint_cache.dofs_vel[f, i_d, i_b]
        dofs_state.acc[i_d, i_b] = rigid_adjoint_cache.dofs_acc[f, i_d, i_b]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_q, i_b in ti.ndrange(n_qs, _B):
        rigid_global_info.qpos[i_q, i_b] = rigid_adjoint_cache.qpos[f, i_q, i_b]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_prepare_backward_substep(
    f: ti.int32,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    dofs_state_adjoint_cache: array_class.DofsState,
    links_state_adjoint_cache: array_class.LinksState,
    joints_state_adjoint_cache: array_class.JointsState,
    geoms_state_adjoint_cache: array_class.GeomsState,
    rigid_adjoint_cache: array_class.RigidAdjointCache,
    static_rigid_sim_config: ti.template(),
):
    # Load the current state from adjoint cache
    func_load_adjoint_cache(
        f=f,
        dofs_state=dofs_state,
        rigid_global_info=rigid_global_info,
        rigid_adjoint_cache=rigid_adjoint_cache,
        static_rigid_sim_config=static_rigid_sim_config,
    )

    # If mujoco compatibility is disabled, update the cartesian space and save the results to adjoint cache. This is
    # because the cartesian space is overwritten later by other kernels if mujoco compatibility was disabled.
    if ti.static(not static_rigid_sim_config.enable_mujoco_compatibility):
        func_update_cartesian_space(
            links_state=links_state,
            links_info=links_info,
            joints_state=joints_state,
            joints_info=joints_info,
            dofs_state=dofs_state,
            dofs_info=dofs_info,
            geoms_state=geoms_state,
            geoms_info=geoms_info,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
            force_update_fixed_geoms=False,
            is_backward=True,
        )

        # FIXME: Parameter pruning for ndarray is buggy for now and requires match variable and arg names.
        # Save results of [update_cartesian_space] to adjoint cache
        func_copy_cartesian_space(
            dofs_state=dofs_state,
            links_state=links_state,
            joints_state=joints_state,
            geoms_state=geoms_state,
            dofs_state_adjoint_cache=dofs_state_adjoint_cache,
            links_state_adjoint_cache=links_state_adjoint_cache,
            joints_state_adjoint_cache=joints_state_adjoint_cache,
            geoms_state_adjoint_cache=geoms_state_adjoint_cache,
            static_rigid_sim_config=static_rigid_sim_config,
        )


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_begin_backward_substep(
    f: ti.int32,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    dofs_state_adjoint_cache: array_class.DofsState,
    links_state_adjoint_cache: array_class.LinksState,
    joints_state_adjoint_cache: array_class.JointsState,
    geoms_state_adjoint_cache: array_class.GeomsState,
    rigid_adjoint_cache: array_class.RigidAdjointCache,
    static_rigid_sim_config: ti.template(),
) -> ti.i32:
    is_grad_valid = func_is_grad_valid(
        rigid_global_info=rigid_global_info,
        dofs_state=dofs_state,
        static_rigid_sim_config=static_rigid_sim_config,
    )
    if is_grad_valid:
        func_copy_next_to_curr_grad(
            f=f,
            dofs_state=dofs_state,
            rigid_global_info=rigid_global_info,
            rigid_adjoint_cache=rigid_adjoint_cache,
            static_rigid_sim_config=static_rigid_sim_config,
        )

        if not static_rigid_sim_config.enable_mujoco_compatibility:
            # FIXME: Parameter pruning for ndarray is buggy for now and requires match variable and arg names.
            # Save results of [update_cartesian_space] to adjoint cache
            func_copy_cartesian_space(
                dofs_state=dofs_state,
                links_state=links_state,
                joints_state=joints_state,
                geoms_state=geoms_state,
                dofs_state_adjoint_cache=dofs_state_adjoint_cache,
                links_state_adjoint_cache=links_state_adjoint_cache,
                joints_state_adjoint_cache=joints_state_adjoint_cache,
                geoms_state_adjoint_cache=geoms_state_adjoint_cache,
                static_rigid_sim_config=static_rigid_sim_config,
            )

    return is_grad_valid


@ti.func
def func_is_grad_valid(
    rigid_global_info: array_class.RigidGlobalInfo,
    dofs_state: array_class.DofsState,
    static_rigid_sim_config: ti.template(),
):
    is_valid = True
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for I in ti.grouped(ti.ndrange(*rigid_global_info.qpos.shape)):
        if ti.math.isnan(rigid_global_info.qpos.grad[I]):
            is_valid = False

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for I in ti.grouped(ti.ndrange(*dofs_state.vel.shape)):
        if ti.math.isnan(dofs_state.vel.grad[I]):
            is_valid = False

    return is_valid


@ti.func
def func_copy_cartesian_space(
    dofs_state: array_class.DofsState,
    links_state: array_class.LinksState,
    joints_state: array_class.JointsState,
    geoms_state: array_class.GeomsState,
    dofs_state_adjoint_cache: array_class.DofsState,
    links_state_adjoint_cache: array_class.LinksState,
    joints_state_adjoint_cache: array_class.JointsState,
    geoms_state_adjoint_cache: array_class.GeomsState,
    static_rigid_sim_config: ti.template(),
):
    # Copy outputs of [kernel_update_cartesian_space] among [dofs, links, joints, geoms] states. This is used to restore
    # the outputs that were overwritten if we disabled mujoco compatibility for backward pass.

    # dofs state
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for I in ti.grouped(ti.ndrange(*dofs_state.pos.shape)):
        # pos, cdof_ang, cdof_vel, cdofvel_ang, cdofvel_vel, cdofd_ang, cdofd_vel
        dofs_state_adjoint_cache.pos[I] = dofs_state.pos[I]
        dofs_state_adjoint_cache.cdof_ang[I] = dofs_state.cdof_ang[I]
        dofs_state_adjoint_cache.cdof_vel[I] = dofs_state.cdof_vel[I]
        dofs_state_adjoint_cache.cdofvel_ang[I] = dofs_state.cdofvel_ang[I]
        dofs_state_adjoint_cache.cdofvel_vel[I] = dofs_state.cdofvel_vel[I]
        dofs_state_adjoint_cache.cdofd_ang[I] = dofs_state.cdofd_ang[I]
        dofs_state_adjoint_cache.cdofd_vel[I] = dofs_state.cdofd_vel[I]

    # links state
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for I in ti.grouped(ti.ndrange(*links_state.pos.shape)):
        # pos, quat, root_COM, mass_sum, i_pos, i_quat, cinr_inertial, cinr_pos, cinr_quat, cinr_mass, j_pos, j_quat,
        # cd_vel, cd_ang
        links_state_adjoint_cache.pos[I] = links_state.pos[I]
        links_state_adjoint_cache.quat[I] = links_state.quat[I]
        links_state_adjoint_cache.root_COM[I] = links_state.root_COM[I]
        links_state_adjoint_cache.mass_sum[I] = links_state.mass_sum[I]
        links_state_adjoint_cache.i_pos[I] = links_state.i_pos[I]
        links_state_adjoint_cache.i_quat[I] = links_state.i_quat[I]
        links_state_adjoint_cache.cinr_inertial[I] = links_state.cinr_inertial[I]
        links_state_adjoint_cache.cinr_pos[I] = links_state.cinr_pos[I]
        links_state_adjoint_cache.cinr_quat[I] = links_state.cinr_quat[I]
        links_state_adjoint_cache.cinr_mass[I] = links_state.cinr_mass[I]
        links_state_adjoint_cache.j_pos[I] = links_state.j_pos[I]
        links_state_adjoint_cache.j_quat[I] = links_state.j_quat[I]
        links_state_adjoint_cache.cd_vel[I] = links_state.cd_vel[I]
        links_state_adjoint_cache.cd_ang[I] = links_state.cd_ang[I]

    # joints state
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for I in ti.grouped(ti.ndrange(*joints_state.xanchor.shape)):
        # xanchor, xaxis
        joints_state_adjoint_cache.xanchor[I] = joints_state.xanchor[I]
        joints_state_adjoint_cache.xaxis[I] = joints_state.xaxis[I]

    # geoms state
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for I in ti.grouped(ti.ndrange(*geoms_state.pos.shape)):
        # pos, quat, verts_updated
        geoms_state_adjoint_cache.pos[I] = geoms_state.pos[I]
        geoms_state_adjoint_cache.quat[I] = geoms_state.quat[I]
        geoms_state_adjoint_cache.verts_updated[I] = geoms_state.verts_updated[I]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_copy_acc(
    f: ti.int32,
    dofs_state: array_class.DofsState,
    rigid_adjoint_cache: array_class.RigidAdjointCache,
    static_rigid_sim_config: ti.template(),
):
    n_dofs = dofs_state.vel.shape[0]
    _B = dofs_state.vel.shape[1]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in ti.ndrange(n_dofs, _B):
        dofs_state.acc[i_d, i_b] = rigid_adjoint_cache.dofs_acc[f, i_d, i_b]


@ti.func
def func_integrate_dq_entity(
    dq,
    i_e,
    i_b,
    respect_joint_limit,
    links_info: array_class.LinksInfo,
    joints_info: array_class.JointsInfo,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    EPS = rigid_global_info.EPS[None]

    for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
        I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
        if links_info.n_dofs[I_l] == 0:
            continue

        i_j = links_info.joint_start[I_l]
        I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
        joint_type = joints_info.type[I_j]

        q_start = links_info.q_start[I_l]
        dof_start = links_info.dof_start[I_l]
        dq_start = links_info.dof_start[I_l] - entities_info.dof_start[i_e]

        if joint_type == gs.JOINT_TYPE.FREE:
            pos = ti.Vector(
                [
                    rigid_global_info.qpos[q_start, i_b],
                    rigid_global_info.qpos[q_start + 1, i_b],
                    rigid_global_info.qpos[q_start + 2, i_b],
                ]
            )
            dpos = ti.Vector([dq[dq_start, i_b], dq[dq_start + 1, i_b], dq[dq_start + 2, i_b]])
            pos = pos + dpos

            quat = ti.Vector(
                [
                    rigid_global_info.qpos[q_start + 3, i_b],
                    rigid_global_info.qpos[q_start + 4, i_b],
                    rigid_global_info.qpos[q_start + 5, i_b],
                    rigid_global_info.qpos[q_start + 6, i_b],
                ]
            )
            dquat = gu.ti_rotvec_to_quat(
                ti.Vector([dq[dq_start + 3, i_b], dq[dq_start + 4, i_b], dq[dq_start + 5, i_b]], dt=gs.ti_float), EPS
            )
            quat = gu.ti_transform_quat_by_quat(
                quat, dquat
            )  # Note that this order is different from integrateing vel. Here dq is w.r.t to world.

            for j in ti.static(range(3)):
                rigid_global_info.qpos[q_start + j, i_b] = pos[j]

            for j in ti.static(range(4)):
                rigid_global_info.qpos[q_start + j + 3, i_b] = quat[j]

        elif joint_type == gs.JOINT_TYPE.FIXED:
            pass

        else:
            for i_d_ in range(links_info.n_dofs[I_l]):
                rigid_global_info.qpos[q_start + i_d_, i_b] = (
                    rigid_global_info.qpos[q_start + i_d_, i_b] + dq[dq_start + i_d_, i_b]
                )

                if respect_joint_limit:
                    I_d = (
                        [dof_start + i_d_, i_b]
                        if ti.static(static_rigid_sim_config.batch_dofs_info)
                        else dof_start + i_d_
                    )
                    rigid_global_info.qpos[q_start + i_d_, i_b] = ti.math.clamp(
                        rigid_global_info.qpos[q_start + i_d_, i_b],
                        dofs_info.limit[I_d][0],
                        dofs_info.limit[I_d][1],
                    )
