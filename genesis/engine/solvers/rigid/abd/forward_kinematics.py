"""
Forward kinematics, velocity propagation, and geometry updates for rigid body simulation.

This module contains Taichi kernels and functions for:
- Forward kinematics computation (link and joint pose updates)
- Velocity propagation through kinematic chains
- Geometry pose and vertex updates
- Center of mass calculations
- AABB updates for collision detection
- Hibernation management for inactive entities
"""

import quadrants as ti

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class
from .misc import (
    func_check_index_range,
    func_read_field_if,
    func_write_field_if,
    func_write_and_read_field_if,
    func_atomic_add_if,
)


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_forward_kinematics_links_geoms(
    envs_idx: ti.types.ndarray(),
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
    static_rigid_sim_config: ti.template(),
):
    func_update_cartesian_space(
        links_state=links_state,
        links_info=links_info,
        joints_state=joints_state,
        joints_info=joints_info,
        dofs_state=dofs_state,
        dofs_info=dofs_info,
        geoms_info=geoms_info,
        geoms_state=geoms_state,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        force_update_fixed_geoms=True,
        is_backward=False,
    )
    func_forward_velocity(
        entities_info=entities_info,
        links_info=links_info,
        links_state=links_state,
        joints_info=joints_info,
        dofs_state=dofs_state,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        is_backward=False,
    )


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_masked_forward_kinematics_links_geoms(
    envs_mask: ti.types.ndarray(),
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
    static_rigid_sim_config: ti.template(),
):
    for i_b in range(envs_mask.shape[0]):
        if envs_mask[i_b]:
            func_update_cartesian_space_batch(
                i_b=i_b,
                links_state=links_state,
                links_info=links_info,
                joints_state=joints_state,
                joints_info=joints_info,
                dofs_state=dofs_state,
                dofs_info=dofs_info,
                geoms_info=geoms_info,
                geoms_state=geoms_state,
                entities_info=entities_info,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
                force_update_fixed_geoms=True,
                is_backward=False,
            )
            func_forward_velocity_batch(
                i_b=i_b,
                entities_info=entities_info,
                links_info=links_info,
                links_state=links_state,
                joints_info=joints_info,
                dofs_state=dofs_state,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
                is_backward=False,
            )


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_forward_velocity(
    envs_idx: ti.types.ndarray(),
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        func_forward_velocity_batch(
            i_b=i_b,
            entities_info=entities_info,
            links_info=links_info,
            links_state=links_state,
            joints_info=joints_info,
            dofs_state=dofs_state,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
            is_backward=is_backward,
        )


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_masked_forward_velocity(
    envs_mask: ti.types.ndarray(),
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    for i_b in range(envs_mask.shape[0]):
        if envs_mask[i_b]:
            func_forward_velocity_batch(
                i_b=i_b,
                entities_info=entities_info,
                links_info=links_info,
                links_state=links_state,
                joints_info=joints_info,
                dofs_state=dofs_state,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
                is_backward=is_backward,
            )


@ti.func
def func_COM_links(
    i_b,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    BW = ti.static(is_backward)

    for i_e_ in (
        (
            range(rigid_global_info.n_awake_entities[i_b])
            if ti.static(static_rigid_sim_config.use_hibernation)
            else range(entities_info.n_links.shape[0])
        )
        if ti.static(not BW)
        else (
            ti.static(range(static_rigid_sim_config.max_n_awake_entities))
            if ti.static(static_rigid_sim_config.use_hibernation)
            else ti.static(range(static_rigid_sim_config.n_entities))
        )
    ):
        if func_check_index_range(
            i_e_, 0, rigid_global_info.n_awake_entities[i_b], static_rigid_sim_config.use_hibernation
        ):
            i_e = (
                rigid_global_info.awake_entities[i_e_, i_b]
                if ti.static(static_rigid_sim_config.use_hibernation)
                else i_e_
            )

            func_COM_links_entity(
                i_e,
                i_b,
                links_state,
                links_info,
                joints_state,
                joints_info,
                dofs_state,
                dofs_info,
                entities_info,
                rigid_global_info,
                static_rigid_sim_config,
                is_backward,
            )


@ti.func
def func_COM_links_entity(
    i_e,
    i_b,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    BW = ti.static(is_backward)

    # Becomes static loop in backward pass, because we assume this loop is an inner loop
    for i_l_ in (
        range(entities_info.link_start[i_e], entities_info.link_end[i_e])
        if ti.static(not BW)
        else ti.static(range(static_rigid_sim_config.max_n_links_per_entity))
    ):
        i_l = i_l_ if ti.static(not BW) else (i_l_ + entities_info.link_start[i_e])

        if func_check_index_range(i_l, entities_info.link_start[i_e], entities_info.link_end[i_e], BW):
            links_state.root_COM_bw[i_l, i_b].fill(0.0)
            links_state.mass_sum[i_l, i_b] = 0.0

    for i_l_ in (
        range(entities_info.link_start[i_e], entities_info.link_end[i_e])
        if ti.static(not BW)
        else ti.static(range(static_rigid_sim_config.max_n_links_per_entity))
    ):
        i_l = i_l_ if ti.static(not BW) else (i_l_ + entities_info.link_start[i_e])

        if func_check_index_range(i_l, entities_info.link_start[i_e], entities_info.link_end[i_e], BW):
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

            mass = links_info.inertial_mass[I_l] + links_state.mass_shift[i_l, i_b]
            (
                links_state.i_pos_bw[i_l, i_b],
                links_state.i_quat[i_l, i_b],
            ) = gu.ti_transform_pos_quat_by_trans_quat(
                links_info.inertial_pos[I_l] + links_state.i_pos_shift[i_l, i_b],
                links_info.inertial_quat[I_l],
                links_state.pos[i_l, i_b],
                links_state.quat[i_l, i_b],
            )

            i_r = links_info.root_idx[I_l]
            links_state.mass_sum[i_r, i_b] = links_state.mass_sum[i_r, i_b] + mass
            ti.atomic_add(links_state.root_COM_bw[i_r, i_b], mass * links_state.i_pos_bw[i_l, i_b])

    for i_l_ in (
        range(entities_info.link_start[i_e], entities_info.link_end[i_e])
        if ti.static(not BW)
        else ti.static(range(static_rigid_sim_config.max_n_links_per_entity))
    ):
        i_l = i_l_ if ti.static(not BW) else (i_l_ + entities_info.link_start[i_e])

        if func_check_index_range(i_l, entities_info.link_start[i_e], entities_info.link_end[i_e], BW):
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

            i_r = links_info.root_idx[I_l]
            if i_l == i_r:
                links_state.root_COM[i_l, i_b] = links_state.root_COM_bw[i_l, i_b] / links_state.mass_sum[i_l, i_b]

    for i_l_ in (
        range(entities_info.link_start[i_e], entities_info.link_end[i_e])
        if ti.static(not BW)
        else ti.static(range(static_rigid_sim_config.max_n_links_per_entity))
    ):
        i_l = i_l_ if ti.static(not BW) else (i_l_ + entities_info.link_start[i_e])

        if func_check_index_range(i_l, entities_info.link_start[i_e], entities_info.link_end[i_e], BW):
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

            i_r = links_info.root_idx[I_l]
            links_state.root_COM[i_l, i_b] = links_state.root_COM[i_r, i_b]

    for i_l_ in (
        range(entities_info.link_start[i_e], entities_info.link_end[i_e])
        if ti.static(not BW)
        else ti.static(range(static_rigid_sim_config.max_n_links_per_entity))
    ):
        i_l = i_l_ if ti.static(not BW) else (i_l_ + entities_info.link_start[i_e])

        if func_check_index_range(i_l, entities_info.link_start[i_e], entities_info.link_end[i_e], BW):
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

            i_r = links_info.root_idx[I_l]
            links_state.i_pos[i_l, i_b] = links_state.i_pos_bw[i_l, i_b] - links_state.root_COM[i_l, i_b]

            i_inertial = links_info.inertial_i[I_l]
            i_mass = links_info.inertial_mass[I_l] + links_state.mass_shift[i_l, i_b]
            (
                links_state.cinr_inertial[i_l, i_b],
                links_state.cinr_pos[i_l, i_b],
                links_state.cinr_quat[i_l, i_b],
                links_state.cinr_mass[i_l, i_b],
            ) = gu.ti_transform_inertia_by_trans_quat(
                i_inertial,
                i_mass,
                links_state.i_pos[i_l, i_b],
                links_state.i_quat[i_l, i_b],
                rigid_global_info.EPS[None],
            )

    for i_l_ in (
        range(entities_info.link_start[i_e], entities_info.link_end[i_e])
        if ti.static(not BW)
        else ti.static(range(static_rigid_sim_config.max_n_links_per_entity))
    ):
        i_l = i_l_ if ti.static(not BW) else (i_l_ + entities_info.link_start[i_e])

        if func_check_index_range(i_l, entities_info.link_start[i_e], entities_info.link_end[i_e], BW):
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

            if links_info.n_dofs[I_l] > 0:
                i_p = links_info.parent_idx[I_l]

                _i_j = links_info.joint_start[I_l]
                _I_j = [_i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else _i_j
                joint_type = joints_info.type[_I_j]

                p_pos = ti.Vector.zero(gs.ti_float, 3)
                p_quat = gu.ti_identity_quat()
                if i_p != -1:
                    p_pos = links_state.pos[i_p, i_b]
                    p_quat = links_state.quat[i_p, i_b]

                if joint_type == gs.JOINT_TYPE.FREE or (links_info.is_fixed[I_l] and i_p == -1):
                    links_state.j_pos[i_l, i_b] = links_state.pos[i_l, i_b]
                    links_state.j_quat[i_l, i_b] = links_state.quat[i_l, i_b]
                else:
                    (
                        links_state.j_pos_bw[i_l, 0, i_b],
                        links_state.j_quat_bw[i_l, 0, i_b],
                    ) = gu.ti_transform_pos_quat_by_trans_quat(links_info.pos[I_l], links_info.quat[I_l], p_pos, p_quat)

                    n_joints = links_info.joint_end[I_l] - links_info.joint_start[I_l]

                    for i_j_ in (
                        range(n_joints)
                        if ti.static(not BW)
                        else ti.static(range(static_rigid_sim_config.max_n_joints_per_link))
                    ):
                        i_j = i_j_ + links_info.joint_start[I_l]

                        curr_i_j = 0 if ti.static(not BW) else i_j_
                        next_i_j = 0 if ti.static(not BW) else i_j_ + 1

                        if func_check_index_range(
                            i_j,
                            links_info.joint_start[I_l],
                            links_info.joint_end[I_l],
                            BW,
                        ):
                            I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j

                            (
                                links_state.j_pos_bw[i_l, next_i_j, i_b],
                                links_state.j_quat_bw[i_l, next_i_j, i_b],
                            ) = gu.ti_transform_pos_quat_by_trans_quat(
                                joints_info.pos[I_j],
                                gu.ti_identity_quat(),
                                links_state.j_pos_bw[i_l, curr_i_j, i_b],
                                links_state.j_quat_bw[i_l, curr_i_j, i_b],
                            )

                    i_j_ = 0 if ti.static(not BW) else n_joints
                    links_state.j_pos[i_l, i_b] = links_state.j_pos_bw[i_l, i_j_, i_b]
                    links_state.j_quat[i_l, i_b] = links_state.j_quat_bw[i_l, i_j_, i_b]

    for i_l_ in (
        range(entities_info.link_start[i_e], entities_info.link_end[i_e])
        if ti.static(not BW)
        else ti.static(range(static_rigid_sim_config.max_n_links_per_entity))
    ):
        i_l = i_l_ if ti.static(not BW) else (i_l_ + entities_info.link_start[i_e])

        if func_check_index_range(i_l, entities_info.link_start[i_e], entities_info.link_end[i_e], BW):
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

            if links_info.n_dofs[I_l] > 0:
                for i_j_ in (
                    range(links_info.joint_start[I_l], links_info.joint_end[I_l])
                    if ti.static(not BW)
                    else ti.static(range(static_rigid_sim_config.max_n_joints_per_link))
                ):
                    i_j = i_j_ if ti.static(not BW) else (i_j_ + links_info.joint_start[I_l])

                    if func_check_index_range(i_j, links_info.joint_start[I_l], links_info.joint_end[I_l], BW):
                        EPS = rigid_global_info.EPS[None]

                        offset_pos = links_state.root_COM[i_l, i_b] - joints_state.xanchor[i_j, i_b]
                        I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
                        joint_type = joints_info.type[I_j]

                        dof_start = joints_info.dof_start[I_j]

                        if joint_type == gs.JOINT_TYPE.REVOLUTE:
                            dofs_state.cdof_ang[dof_start, i_b] = joints_state.xaxis[i_j, i_b]
                            dofs_state.cdof_vel[dof_start, i_b] = joints_state.xaxis[i_j, i_b].cross(offset_pos)
                        elif joint_type == gs.JOINT_TYPE.PRISMATIC:
                            dofs_state.cdof_ang[dof_start, i_b] = ti.Vector.zero(gs.ti_float, 3)
                            dofs_state.cdof_vel[dof_start, i_b] = joints_state.xaxis[i_j, i_b]
                        elif joint_type == gs.JOINT_TYPE.SPHERICAL:
                            xmat_T = gu.ti_quat_to_R(links_state.quat[i_l, i_b], EPS).transpose()
                            for i in ti.static(range(3)):
                                dofs_state.cdof_ang[i + dof_start, i_b] = xmat_T[i, :]
                                dofs_state.cdof_vel[i + dof_start, i_b] = xmat_T[i, :].cross(offset_pos)
                        elif joint_type == gs.JOINT_TYPE.FREE:
                            for i in ti.static(range(3)):
                                dofs_state.cdof_ang[i + dof_start, i_b] = ti.Vector.zero(gs.ti_float, 3)
                                dofs_state.cdof_vel[i + dof_start, i_b] = ti.Vector.zero(gs.ti_float, 3)
                                dofs_state.cdof_vel[i + dof_start, i_b][i] = 1.0

                            xmat_T = gu.ti_quat_to_R(links_state.quat[i_l, i_b], EPS).transpose()
                            for i in ti.static(range(3)):
                                dofs_state.cdof_ang[i + dof_start + 3, i_b] = xmat_T[i, :]
                                dofs_state.cdof_vel[i + dof_start + 3, i_b] = xmat_T[i, :].cross(offset_pos)

                        for i_d_ in (
                            range(dof_start, joints_info.dof_end[I_j])
                            if ti.static(not BW)
                            else ti.static(range(static_rigid_sim_config.max_n_dofs_per_joint))
                        ):
                            i_d = i_d_ if ti.static(not BW) else (i_d_ + dof_start)
                            if func_check_index_range(i_d, dof_start, joints_info.dof_end[I_j], BW):
                                dofs_state.cdofvel_ang[i_d, i_b] = (
                                    dofs_state.cdof_ang[i_d, i_b] * dofs_state.vel[i_d, i_b]
                                )
                                dofs_state.cdofvel_vel[i_d, i_b] = (
                                    dofs_state.cdof_vel[i_d, i_b] * dofs_state.vel[i_d, i_b]
                                )


@ti.func
def func_forward_kinematics_entity(
    i_e,
    i_b,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    BW = ti.static(is_backward)
    W = ti.static(func_write_field_if)
    R = ti.static(func_read_field_if)
    WR = ti.static(func_write_and_read_field_if)

    # Becomes static loop in backward pass, because we assume this loop is an inner loop
    for i_l_ in (
        range(entities_info.link_start[i_e], entities_info.link_end[i_e])
        if ti.static(not BW)
        else ti.static(range(static_rigid_sim_config.max_n_links_per_entity))
    ):
        i_l = i_l_ if ti.static(not BW) else (i_l_ + entities_info.link_start[i_e])

        if func_check_index_range(i_l, entities_info.link_start[i_e], entities_info.link_end[i_e], BW):
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
            I_l0 = (i_l, 0, i_b)

            pos = W(links_state.pos_bw, I_l0, links_info.pos[I_l], BW)
            quat = W(links_state.quat_bw, I_l0, links_info.quat[I_l], BW)
            if links_info.parent_idx[I_l] != -1:
                parent_pos = links_state.pos[links_info.parent_idx[I_l], i_b]
                parent_quat = links_state.quat[links_info.parent_idx[I_l], i_b]
                pos_ = parent_pos + gu.ti_transform_by_quat(links_info.pos[I_l], parent_quat)
                quat_ = gu.ti_transform_quat_by_quat(links_info.quat[I_l], parent_quat)

                pos = W(links_state.pos_bw, I_l0, pos_, BW)
                quat = W(links_state.quat_bw, I_l0, quat_, BW)

            n_joints = links_info.joint_end[I_l] - links_info.joint_start[I_l]

            for i_j_ in (
                range(n_joints)
                if ti.static(not BW)
                else ti.static(range(static_rigid_sim_config.max_n_joints_per_link))
            ):
                i_j = i_j_ + links_info.joint_start[I_l]

                curr_I = (i_l, 0 if ti.static(not BW) else i_j_, i_b)
                next_I = (i_l, 0 if ti.static(not BW) else i_j_ + 1, i_b)

                if func_check_index_range(i_j, links_info.joint_start[I_l], links_info.joint_end[I_l], BW):
                    I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
                    joint_type = joints_info.type[I_j]
                    q_start = joints_info.q_start[I_j]
                    dof_start = joints_info.dof_start[I_j]
                    I_d = [dof_start, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else dof_start

                    # compute axis and anchor
                    if joint_type == gs.JOINT_TYPE.FREE:
                        joints_state.xanchor[i_j, i_b] = ti.Vector(
                            [
                                rigid_global_info.qpos[q_start, i_b],
                                rigid_global_info.qpos[q_start + 1, i_b],
                                rigid_global_info.qpos[q_start + 2, i_b],
                            ]
                        )
                        joints_state.xaxis[i_j, i_b] = ti.Vector([0.0, 0.0, 1.0])
                    elif joint_type == gs.JOINT_TYPE.FIXED:
                        pass
                    else:
                        axis = ti.Vector([0.0, 0.0, 1.0], dt=gs.ti_float)
                        if joint_type == gs.JOINT_TYPE.REVOLUTE:
                            axis = dofs_info.motion_ang[I_d]
                        elif joint_type == gs.JOINT_TYPE.PRISMATIC:
                            axis = dofs_info.motion_vel[I_d]

                        pos_ = R(links_state.pos_bw, curr_I, pos, BW)
                        quat_ = R(links_state.quat_bw, curr_I, quat, BW)

                        joints_state.xanchor[i_j, i_b] = gu.ti_transform_by_quat(joints_info.pos[I_j], quat_) + pos_
                        joints_state.xaxis[i_j, i_b] = gu.ti_transform_by_quat(axis, quat_)

                    if joint_type == gs.JOINT_TYPE.FREE:
                        pos_ = ti.Vector(
                            [
                                rigid_global_info.qpos[q_start, i_b],
                                rigid_global_info.qpos[q_start + 1, i_b],
                                rigid_global_info.qpos[q_start + 2, i_b],
                            ],
                            dt=gs.ti_float,
                        )
                        quat_ = ti.Vector(
                            [
                                rigid_global_info.qpos[q_start + 3, i_b],
                                rigid_global_info.qpos[q_start + 4, i_b],
                                rigid_global_info.qpos[q_start + 5, i_b],
                                rigid_global_info.qpos[q_start + 6, i_b],
                            ],
                            dt=gs.ti_float,
                        )
                        quat_ = quat_ / quat_.norm()
                        pos = WR(links_state.pos_bw, next_I, pos_, BW)
                        quat = WR(links_state.quat_bw, next_I, quat_, BW)

                        xyz = gu.ti_quat_to_xyz(quat, rigid_global_info.EPS[None])
                        for j in ti.static(range(3)):
                            dofs_state.pos[dof_start + j, i_b] = pos[j]
                            dofs_state.pos[dof_start + 3 + j, i_b] = xyz[j]
                    elif joint_type == gs.JOINT_TYPE.FIXED:
                        pass
                    elif joint_type == gs.JOINT_TYPE.SPHERICAL:
                        qloc = ti.Vector(
                            [
                                rigid_global_info.qpos[q_start, i_b],
                                rigid_global_info.qpos[q_start + 1, i_b],
                                rigid_global_info.qpos[q_start + 2, i_b],
                                rigid_global_info.qpos[q_start + 3, i_b],
                            ],
                            dt=gs.ti_float,
                        )
                        xyz = gu.ti_quat_to_xyz(qloc, rigid_global_info.EPS[None])
                        for j in ti.static(range(3)):
                            dofs_state.pos[dof_start + j, i_b] = xyz[j]
                        quat_ = gu.ti_transform_quat_by_quat(qloc, R(links_state.quat_bw, curr_I, quat, BW))
                        quat = WR(links_state.quat_bw, next_I, quat_, BW)
                        pos_ = joints_state.xanchor[i_j, i_b] - gu.ti_transform_by_quat(joints_info.pos[I_j], quat)
                        pos = W(links_state.pos_bw, next_I, pos_, BW)
                    elif joint_type == gs.JOINT_TYPE.REVOLUTE:
                        axis = dofs_info.motion_ang[I_d]
                        dofs_state.pos[dof_start, i_b] = (
                            rigid_global_info.qpos[q_start, i_b] - rigid_global_info.qpos0[q_start, i_b]
                        )
                        qloc = gu.ti_rotvec_to_quat(axis * dofs_state.pos[dof_start, i_b], rigid_global_info.EPS[None])
                        quat_ = gu.ti_transform_quat_by_quat(qloc, R(links_state.quat_bw, curr_I, quat, BW))
                        quat = WR(links_state.quat_bw, next_I, quat_, BW)
                        pos_ = joints_state.xanchor[i_j, i_b] - gu.ti_transform_by_quat(joints_info.pos[I_j], quat)
                        pos = W(links_state.pos_bw, next_I, pos_, BW)
                    else:  # joint_type == gs.JOINT_TYPE.PRISMATIC:
                        dofs_state.pos[dof_start, i_b] = (
                            rigid_global_info.qpos[q_start, i_b] - rigid_global_info.qpos0[q_start, i_b]
                        )
                        pos_ = (
                            R(links_state.pos_bw, curr_I, pos, BW)
                            + joints_state.xaxis[i_j, i_b] * dofs_state.pos[dof_start, i_b]
                        )
                        pos = W(links_state.pos_bw, next_I, pos_, BW)

            # Skip link pose update for fixed root links to let users manually overwrite them
            I_jf = (i_l, 0 if ti.static(not BW) else n_joints, i_b)
            if not (links_info.parent_idx[I_l] == -1 and links_info.is_fixed[I_l]):
                links_state.pos[i_l, i_b] = R(links_state.pos_bw, I_jf, pos, BW)
                links_state.quat[i_l, i_b] = R(links_state.quat_bw, I_jf, quat, BW)


@ti.func
def func_forward_kinematics_batch(
    i_b,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    BW = ti.static(is_backward)

    for i_e_ in (
        (
            range(rigid_global_info.n_awake_entities[i_b])
            if ti.static(static_rigid_sim_config.use_hibernation)
            else range(entities_info.n_links.shape[0])
        )
        if ti.static(not BW)
        else (
            ti.static(range(static_rigid_sim_config.max_n_awake_entities))
            if ti.static(static_rigid_sim_config.use_hibernation)
            else ti.static(range(static_rigid_sim_config.max_n_links_per_entity))
        )
    ):
        if func_check_index_range(
            i_e_, 0, rigid_global_info.n_awake_entities[i_b], static_rigid_sim_config.use_hibernation
        ):
            i_e = (
                rigid_global_info.awake_entities[i_e_, i_b]
                if ti.static(static_rigid_sim_config.use_hibernation)
                else i_e_
            )

            func_forward_kinematics_entity(
                i_e,
                i_b,
                links_state,
                links_info,
                joints_state,
                joints_info,
                dofs_state,
                dofs_info,
                entities_info,
                rigid_global_info,
                static_rigid_sim_config,
                is_backward,
            )


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_forward_kinematics_entity(
    i_e: ti.int32,
    envs_idx: ti.types.ndarray(),
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]

        func_forward_kinematics_entity(
            i_e,
            i_b,
            links_state,
            links_info,
            joints_state,
            joints_info,
            dofs_state,
            dofs_info,
            entities_info,
            rigid_global_info,
            static_rigid_sim_config,
            is_backward=False,
        )


@ti.func
def func_update_geoms_entity(
    i_e,
    i_b,
    entities_info: array_class.EntitiesInfo,
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    force_update_fixed_geoms: ti.template(),
    is_backward: ti.template(),
):
    """
    NOTE: this only update geom pose, not its verts and else.
    """
    BW = ti.static(is_backward)

    for i_g_ in (
        # Dynamic inner loop for forward pass
        range(entities_info.n_geoms[i_e])
        if ti.static(not BW)
        else ti.static(range(static_rigid_sim_config.max_n_geoms_per_entity))  # Static inner loop for backward pass
    ):
        i_g = entities_info.geom_start[i_e] + i_g_
        if func_check_index_range(i_g, entities_info.geom_start[i_e], entities_info.geom_end[i_e], BW):
            if force_update_fixed_geoms or not geoms_info.is_fixed[i_g]:
                (
                    geoms_state.pos[i_g, i_b],
                    geoms_state.quat[i_g, i_b],
                ) = gu.ti_transform_pos_quat_by_trans_quat(
                    geoms_info.pos[i_g],
                    geoms_info.quat[i_g],
                    links_state.pos[geoms_info.link_idx[i_g], i_b],
                    links_state.quat[geoms_info.link_idx[i_g], i_b],
                )
                geoms_state.verts_updated[i_g, i_b] = False


@ti.func
def func_update_geoms_batch(
    i_b,
    entities_info: array_class.EntitiesInfo,
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    force_update_fixed_geoms: ti.template(),
    is_backward: ti.template(),
):
    """
    NOTE: this only update geom pose, not its verts and else.
    """
    BW = ti.static(is_backward)

    for i_e_ in (
        (
            # Dynamic inner loop for forward pass
            range(rigid_global_info.n_awake_entities[i_b])
            if ti.static(static_rigid_sim_config.use_hibernation)
            else range(entities_info.n_links.shape[0])
        )
        if ti.static(not BW)
        else (
            ti.static(range(static_rigid_sim_config.max_n_awake_entities))  # Static inner loop for backward pass
            if ti.static(static_rigid_sim_config.use_hibernation)
            else ti.static(range(static_rigid_sim_config.max_n_links_per_entity))
        )
    ):
        if func_check_index_range(
            i_e_, 0, rigid_global_info.n_awake_entities[i_b], static_rigid_sim_config.use_hibernation
        ):
            i_e = (
                rigid_global_info.awake_entities[i_e_, i_b]
                if ti.static(static_rigid_sim_config.use_hibernation)
                else i_e_
            )

            func_update_geoms_entity(
                i_e,
                i_b,
                entities_info,
                geoms_info,
                geoms_state,
                links_state,
                rigid_global_info,
                static_rigid_sim_config,
                force_update_fixed_geoms,
                is_backward,
            )


@ti.func
def func_update_geoms(
    entities_info: array_class.EntitiesInfo,
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    force_update_fixed_geoms: ti.template(),
    is_backward: ti.template(),
):
    # This loop must be the outermost loop to be differentiable
    if ti.static(static_rigid_sim_config.use_hibernation):
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(links_state.pos.shape[1]):
            func_update_geoms_batch(
                i_b,
                entities_info,
                geoms_info,
                geoms_state,
                links_state,
                rigid_global_info,
                static_rigid_sim_config,
                force_update_fixed_geoms,
                is_backward,
            )
    else:
        ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_e, i_b in ti.ndrange(entities_info.n_links.shape[0], links_state.pos.shape[1]):
            func_update_geoms_entity(
                i_e,
                i_b,
                entities_info,
                geoms_info,
                geoms_state,
                links_state,
                rigid_global_info,
                static_rigid_sim_config,
                force_update_fixed_geoms,
                is_backward,
            )


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_update_geoms(
    envs_idx: ti.types.ndarray(),
    entities_info: array_class.EntitiesInfo,
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    force_update_fixed_geoms: ti.template(),
):
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]

        func_update_geoms_batch(
            i_b,
            entities_info,
            geoms_info,
            geoms_state,
            links_state,
            rigid_global_info,
            static_rigid_sim_config,
            force_update_fixed_geoms,
            is_backward=False,
        )


@ti.func
def func_forward_velocity_entity(
    i_e,
    i_b,
    entities_info: array_class.EntitiesInfo,
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    BW = ti.static(is_backward)
    W = ti.static(func_write_field_if)
    R = ti.static(func_read_field_if)
    A = ti.static(func_atomic_add_if)

    for i_l_ in (
        range(entities_info.link_start[i_e], entities_info.link_end[i_e])
        if ti.static(not BW)
        else ti.static(range(static_rigid_sim_config.max_n_links_per_entity))
    ):
        i_l = i_l_ if ti.static(not BW) else (i_l_ + entities_info.link_start[i_e])

        if func_check_index_range(i_l, entities_info.link_start[i_e], entities_info.link_end[i_e], BW):
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
            n_joints = links_info.joint_end[I_l] - links_info.joint_start[I_l]

            I_j0 = (i_l, 0, i_b)
            cvel_vel = W(links_state.cd_vel_bw, I_j0, ti.Vector.zero(gs.ti_float, 3), BW)
            cvel_ang = W(links_state.cd_ang_bw, I_j0, ti.Vector.zero(gs.ti_float, 3), BW)

            if links_info.parent_idx[I_l] != -1:
                cvel_vel = W(links_state.cd_vel_bw, I_j0, links_state.cd_vel[links_info.parent_idx[I_l], i_b], BW)
                cvel_ang = W(links_state.cd_ang_bw, I_j0, links_state.cd_ang[links_info.parent_idx[I_l], i_b], BW)

            for i_j_ in (
                range(n_joints)
                if ti.static(not BW)
                else ti.static(range(static_rigid_sim_config.max_n_joints_per_link))
            ):
                i_j = i_j_ + links_info.joint_start[I_l]

                if func_check_index_range(i_j, links_info.joint_start[I_l], links_info.joint_end[I_l], BW):
                    I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
                    joint_type = joints_info.type[I_j]
                    dof_start = joints_info.dof_start[I_j]

                    curr_I = (i_l, 0 if ti.static(not BW) else i_j_, i_b)
                    next_I = (i_l, 0 if ti.static(not BW) else i_j_ + 1, i_b)

                    if joint_type == gs.JOINT_TYPE.FREE:
                        for i_3 in ti.static(range(3)):
                            _vel = dofs_state.cdof_vel[dof_start + i_3, i_b] * dofs_state.vel[dof_start + i_3, i_b]
                            _ang = dofs_state.cdof_ang[dof_start + i_3, i_b] * dofs_state.vel[dof_start + i_3, i_b]

                            cvel_vel = cvel_vel + A(links_state.cd_vel_bw, curr_I, _vel, BW)
                            cvel_ang = cvel_ang + A(links_state.cd_ang_bw, curr_I, _ang, BW)

                        for i_3 in ti.static(range(3)):
                            (
                                dofs_state.cdofd_ang[dof_start + i_3, i_b],
                                dofs_state.cdofd_vel[dof_start + i_3, i_b],
                            ) = ti.Vector.zero(gs.ti_float, 3), ti.Vector.zero(gs.ti_float, 3)

                            (
                                dofs_state.cdofd_ang[dof_start + i_3 + 3, i_b],
                                dofs_state.cdofd_vel[dof_start + i_3 + 3, i_b],
                            ) = gu.motion_cross_motion(
                                R(links_state.cd_ang_bw, curr_I, cvel_ang, BW),
                                R(links_state.cd_vel_bw, curr_I, cvel_vel, BW),
                                dofs_state.cdof_ang[dof_start + i_3 + 3, i_b],
                                dofs_state.cdof_vel[dof_start + i_3 + 3, i_b],
                            )

                        if ti.static(BW):
                            links_state.cd_vel_bw[next_I] = links_state.cd_vel_bw[curr_I]
                            links_state.cd_ang_bw[next_I] = links_state.cd_ang_bw[curr_I]

                        for i_3 in ti.static(range(3)):
                            _vel = (
                                dofs_state.cdof_vel[dof_start + i_3 + 3, i_b] * dofs_state.vel[dof_start + i_3 + 3, i_b]
                            )
                            _ang = (
                                dofs_state.cdof_ang[dof_start + i_3 + 3, i_b] * dofs_state.vel[dof_start + i_3 + 3, i_b]
                            )
                            cvel_vel = cvel_vel + A(links_state.cd_vel_bw, next_I, _vel, BW)
                            cvel_ang = cvel_ang + A(links_state.cd_ang_bw, next_I, _ang, BW)

                    else:
                        for i_d_ in (
                            range(dof_start, joints_info.dof_end[I_j])
                            if ti.static(not BW)
                            else ti.static(range(static_rigid_sim_config.max_n_dofs_per_joint))
                        ):
                            i_d = i_d_ if ti.static(not BW) else (i_d_ + dof_start)
                            if func_check_index_range(i_d, dof_start, joints_info.dof_end[I_j], BW):
                                dofs_state.cdofd_ang[i_d, i_b], dofs_state.cdofd_vel[i_d, i_b] = gu.motion_cross_motion(
                                    R(links_state.cd_ang_bw, curr_I, cvel_ang, BW),
                                    R(links_state.cd_vel_bw, curr_I, cvel_vel, BW),
                                    dofs_state.cdof_ang[i_d, i_b],
                                    dofs_state.cdof_vel[i_d, i_b],
                                )

                        if ti.static(BW):
                            links_state.cd_vel_bw[next_I] = links_state.cd_vel_bw[curr_I]
                            links_state.cd_ang_bw[next_I] = links_state.cd_ang_bw[curr_I]

                        for i_d_ in (
                            range(dof_start, joints_info.dof_end[I_j])
                            if ti.static(not BW)
                            else ti.static(range(static_rigid_sim_config.max_n_dofs_per_joint))
                        ):
                            i_d = i_d_ if ti.static(not BW) else (i_d_ + dof_start)
                            if func_check_index_range(i_d, dof_start, joints_info.dof_end[I_j], BW):
                                _vel = dofs_state.cdof_vel[i_d, i_b] * dofs_state.vel[i_d, i_b]
                                _ang = dofs_state.cdof_ang[i_d, i_b] * dofs_state.vel[i_d, i_b]
                                cvel_vel = cvel_vel + A(links_state.cd_vel_bw, next_I, _vel, BW)
                                cvel_ang = cvel_ang + A(links_state.cd_ang_bw, next_I, _ang, BW)

            I_jf = (i_l, 0 if ti.static(not BW) else n_joints, i_b)
            links_state.cd_vel[i_l, i_b] = R(links_state.cd_vel_bw, I_jf, cvel_vel, BW)
            links_state.cd_ang[i_l, i_b] = R(links_state.cd_ang_bw, I_jf, cvel_ang, BW)


@ti.func
def func_forward_velocity_batch(
    i_b,
    entities_info: array_class.EntitiesInfo,
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    BW = ti.static(is_backward)

    for i_e_ in (
        (
            # Dynamic inner loop for forward pass
            range(rigid_global_info.n_awake_entities[i_b])
            if ti.static(static_rigid_sim_config.use_hibernation)
            else range(entities_info.n_links.shape[0])
        )
        if ti.static(not BW)
        else (
            ti.static(range(static_rigid_sim_config.max_n_awake_entities))  # Static inner loop for backward pass
            if ti.static(static_rigid_sim_config.use_hibernation)
            else ti.static(range(static_rigid_sim_config.max_n_links_per_entity))
        )
    ):
        if func_check_index_range(
            i_e_, 0, rigid_global_info.n_awake_entities[i_b], static_rigid_sim_config.use_hibernation
        ):
            i_e = (
                rigid_global_info.awake_entities[i_e_, i_b]
                if ti.static(static_rigid_sim_config.use_hibernation)
                else i_e_
            )

            func_forward_velocity_entity(
                i_e=i_e,
                i_b=i_b,
                entities_info=entities_info,
                links_info=links_info,
                links_state=links_state,
                joints_info=joints_info,
                dofs_state=dofs_state,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
                is_backward=is_backward,
            )


@ti.func
def func_forward_velocity(
    entities_info: array_class.EntitiesInfo,
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    # This loop must be the outermost loop to be differentiable
    if ti.static(static_rigid_sim_config.use_hibernation):
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(links_state.pos.shape[1]):
            func_forward_velocity_batch(
                i_b,
                entities_info,
                links_info,
                links_state,
                joints_info,
                dofs_state,
                rigid_global_info,
                static_rigid_sim_config,
                is_backward,
            )
    else:
        ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_e, i_b in ti.ndrange(entities_info.n_links.shape[0], links_state.pos.shape[1]):
            func_forward_velocity_entity(
                i_e,
                i_b,
                entities_info,
                links_info,
                links_state,
                joints_info,
                dofs_state,
                rigid_global_info,
                static_rigid_sim_config,
                is_backward,
            )


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_update_verts_for_geoms(
    geoms_idx: ti.types.ndarray(),
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    free_verts_state: array_class.VertsState,
    fixed_verts_state: array_class.VertsState,
    static_rigid_sim_config: ti.template(),
):
    n_geoms = geoms_idx.shape[0]
    _B = geoms_state.verts_updated.shape[1]

    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_g_, i_b in ti.ndrange(n_geoms, _B):
        i_g = geoms_idx[i_g_]
        func_update_verts_for_geom(i_g, i_b, geoms_state, geoms_info, verts_info, free_verts_state, fixed_verts_state)


@ti.func
def func_update_verts_for_geom(
    i_g: ti.i32,
    i_b: ti.i32,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    free_verts_state: array_class.VertsState,
    fixed_verts_state: array_class.VertsState,
):
    _B = geoms_state.verts_updated.shape[1]

    if not geoms_state.verts_updated[i_g, i_b]:
        i_v_start = geoms_info.vert_start[i_g]
        if verts_info.is_fixed[i_v_start]:
            for i_v in range(i_v_start, geoms_info.vert_end[i_g]):
                verts_state_idx = verts_info.verts_state_idx[i_v]
                fixed_verts_state.pos[verts_state_idx] = gu.ti_transform_by_trans_quat(
                    verts_info.init_pos[i_v], geoms_state.pos[i_g, i_b], geoms_state.quat[i_g, i_b]
                )
            for j_b in range(_B):
                geoms_state.verts_updated[i_g, j_b] = True
        else:
            for i_v in range(i_v_start, geoms_info.vert_end[i_g]):
                verts_state_idx = verts_info.verts_state_idx[i_v]
                free_verts_state.pos[verts_state_idx, i_b] = gu.ti_transform_by_trans_quat(
                    verts_info.init_pos[i_v], geoms_state.pos[i_g, i_b], geoms_state.quat[i_g, i_b]
                )
            geoms_state.verts_updated[i_g, i_b] = True


@ti.func
def func_update_all_verts(
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    verts_info: array_class.VertsInfo,
    free_verts_state: array_class.VertsState,
    fixed_verts_state: array_class.VertsState,
    static_rigid_sim_config: ti.template(),
):
    n_geoms, _B = geoms_state.pos.shape

    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_g, i_b in ti.ndrange(n_geoms, _B):
        func_update_verts_for_geom(i_g, i_b, geoms_state, geoms_info, verts_info, free_verts_state, fixed_verts_state)


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_update_all_verts(
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    verts_info: array_class.VertsInfo,
    free_verts_state: array_class.VertsState,
    fixed_verts_state: array_class.VertsState,
    static_rigid_sim_config: ti.template(),
):
    func_update_all_verts(
        geoms_info, geoms_state, verts_info, free_verts_state, fixed_verts_state, static_rigid_sim_config
    )


@ti.kernel
def kernel_update_geom_aabbs(
    geoms_state: array_class.GeomsState,
    geoms_init_AABB: array_class.GeomsInitAABB,
    static_rigid_sim_config: ti.template(),
):
    n_geoms = geoms_state.pos.shape[0]
    _B = geoms_state.pos.shape[1]

    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_g, i_b in ti.ndrange(n_geoms, _B):
        g_pos = geoms_state.pos[i_g, i_b]
        g_quat = geoms_state.quat[i_g, i_b]

        lower = gu.ti_vec3(ti.math.inf)
        upper = gu.ti_vec3(-ti.math.inf)
        for i_corner in ti.static(range(8)):
            corner_pos = gu.ti_transform_by_trans_quat(geoms_init_AABB[i_g, i_corner], g_pos, g_quat)
            lower = ti.min(lower, corner_pos)
            upper = ti.max(upper, corner_pos)

        geoms_state.aabb_min[i_g, i_b] = lower
        geoms_state.aabb_max[i_g, i_b] = upper


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_update_vgeoms(
    vgeoms_info: array_class.VGeomsInfo,
    vgeoms_state: array_class.VGeomsState,
    links_state: array_class.LinksState,
    static_rigid_sim_config: ti.template(),
):
    """
    Vgeoms are only for visualization purposes.
    """
    n_vgeoms = vgeoms_info.link_idx.shape[0]
    _B = links_state.pos.shape[1]

    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_g, i_b in ti.ndrange(n_vgeoms, _B):
        i_l = vgeoms_info.link_idx[i_g]
        vgeoms_state.pos[i_g, i_b], vgeoms_state.quat[i_g, i_b] = gu.ti_transform_pos_quat_by_trans_quat(
            vgeoms_info.pos[i_g], vgeoms_info.quat[i_g], links_state.pos[i_l, i_b], links_state.quat[i_l, i_b]
        )


@ti.func
def func_hibernate__for_all_awake_islands_either_hiberanate_or_update_aabb_sort_buffer(
    dofs_state: array_class.DofsState,
    entities_state: array_class.EntitiesState,
    entities_info: array_class.EntitiesInfo,
    links_state: array_class.LinksState,
    geoms_state: array_class.GeomsState,
    collider_state: array_class.ColliderState,
    unused__rigid_global_info: array_class.RigidGlobalInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    contact_island_state: array_class.ContactIslandState,
    errno: array_class.V_ANNOTATION,
):
    _B = entities_state.hibernated.shape[1]

    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        for island_idx in range(contact_island_state.n_islands[i_b]):
            was_island_hibernated = contact_island_state.island_hibernated[island_idx, i_b]

            if not was_island_hibernated:
                are_all_entities_okay_for_hibernation = True
                entity_ref_n = contact_island_state.island_entity.n[island_idx, i_b]
                entity_ref_start = contact_island_state.island_entity.start[island_idx, i_b]

                # Invariant check: ensure entity_id access won't exceed buffer
                if entity_ref_start + entity_ref_n > contact_island_state.entity_id.shape[0]:
                    errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_HIBERNATION_ISLANDS
                    continue

                for i_entity_ref_offset_ in range(entity_ref_n):
                    entity_ref = entity_ref_start + i_entity_ref_offset_
                    entity_idx = contact_island_state.entity_id[entity_ref, i_b]

                    # Hibernated entities already have zero dofs_state.acc/vel
                    is_entity_hibernated = entities_state.hibernated[entity_idx, i_b]
                    if is_entity_hibernated:
                        continue

                    for i_d in range(entities_info.dof_start[entity_idx], entities_info.dof_end[entity_idx]):
                        max_acc = rigid_global_info.hibernation_thresh_acc[None]
                        max_vel = rigid_global_info.hibernation_thresh_vel[None]
                        if ti.abs(dofs_state.acc[i_d, i_b]) > max_acc or ti.abs(dofs_state.vel[i_d, i_b]) > max_vel:
                            are_all_entities_okay_for_hibernation = False
                            break

                    if not are_all_entities_okay_for_hibernation:
                        break

                if not are_all_entities_okay_for_hibernation:
                    # update collider sort_buffer with aabb extents along x-axis
                    for i_entity_ref_offset_ in range(entity_ref_n):
                        entity_ref = entity_ref_start + i_entity_ref_offset_
                        entity_idx = contact_island_state.entity_id[entity_ref, i_b]
                        for i_g in range(entities_info.geom_start[entity_idx], entities_info.geom_end[entity_idx]):
                            min_idx, min_val = geoms_state.min_buffer_idx[i_g, i_b], geoms_state.aabb_min[i_g, i_b][0]
                            max_idx, max_val = geoms_state.max_buffer_idx[i_g, i_b], geoms_state.aabb_max[i_g, i_b][0]
                            collider_state.sort_buffer.value[min_idx, i_b] = min_val
                            collider_state.sort_buffer.value[max_idx, i_b] = max_val
                else:
                    # perform hibernation
                    # Guard: only process if there are entities in this island
                    if entity_ref_n > 0:
                        prev_entity_ref = entity_ref_start + entity_ref_n - 1
                        prev_entity_idx = contact_island_state.entity_id[prev_entity_ref, i_b]

                        for i_entity_ref_offset_ in range(entity_ref_n):
                            entity_ref = entity_ref_start + i_entity_ref_offset_
                            entity_idx = contact_island_state.entity_id[entity_ref, i_b]

                            func_hibernate_entity_and_zero_dof_velocities(
                                entity_idx,
                                i_b,
                                entities_state=entities_state,
                                entities_info=entities_info,
                                dofs_state=dofs_state,
                                links_state=links_state,
                                geoms_state=geoms_state,
                            )

                            # store entities in the hibernated islands by daisy chaining them
                            contact_island_state.entity_idx_to_next_entity_idx_in_hibernated_island[
                                prev_entity_idx, i_b
                            ] = entity_idx
                            prev_entity_idx = entity_idx


@ti.func
def func_aggregate_awake_entities(
    entities_state: array_class.EntitiesState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    n_entities = entities_state.hibernated.shape[0]
    _B = entities_state.hibernated.shape[1]

    # Reset counts once per batch (not per entity!)
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        rigid_global_info.n_awake_entities[i_b] = 0
        rigid_global_info.n_awake_links[i_b] = 0
        rigid_global_info.n_awake_dofs[i_b] = 0

    # Count awake entities
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_e, i_b in ti.ndrange(n_entities, _B):
        if entities_state.hibernated[i_e, i_b] or entities_info.n_dofs[i_e] == 0:
            continue

        next_awake_entity_idx = ti.atomic_add(rigid_global_info.n_awake_entities[i_b], 1)
        rigid_global_info.awake_entities[next_awake_entity_idx, i_b] = i_e

        n_dofs = entities_info.n_dofs[i_e]
        entity_dofs_base_idx: ti.int32 = entities_info.dof_start[i_e]
        awake_dofs_base_idx = ti.atomic_add(rigid_global_info.n_awake_dofs[i_b], n_dofs)
        for i_d_ in range(n_dofs):
            rigid_global_info.awake_dofs[awake_dofs_base_idx + i_d_, i_b] = entity_dofs_base_idx + i_d_

        n_links = entities_info.n_links[i_e]
        entity_links_base_idx: ti.int32 = entities_info.link_start[i_e]
        awake_links_base_idx = ti.atomic_add(rigid_global_info.n_awake_links[i_b], n_links)
        for i_l_ in range(n_links):
            rigid_global_info.awake_links[awake_links_base_idx + i_l_, i_b] = entity_links_base_idx + i_l_


@ti.func
def func_hibernate_entity_and_zero_dof_velocities(
    i_e: int,
    i_b: int,
    entities_state: array_class.EntitiesState,
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    links_state: array_class.LinksState,
    geoms_state: array_class.GeomsState,
):
    """
    Mark RigidEnity, individual DOFs in DofsState, RigidLinks, and RigidGeoms as hibernated.

    Also, zero out DOF velocitities and accelerations.
    """
    entities_state.hibernated[i_e, i_b] = True

    for i_d in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
        dofs_state.hibernated[i_d, i_b] = True
        dofs_state.vel[i_d, i_b] = 0.0
        dofs_state.acc[i_d, i_b] = 0.0

    for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
        links_state.hibernated[i_l, i_b] = True

    for i_g in range(entities_info.geom_start[i_e], entities_info.geom_end[i_e]):
        geoms_state.hibernated[i_g, i_b] = True


@ti.func
def func_update_cartesian_space_entity(
    i_e,
    i_b,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    force_update_fixed_geoms: ti.template(),
    is_backward: ti.template(),
):
    func_forward_kinematics_entity(
        i_e,
        i_b,
        links_state=links_state,
        links_info=links_info,
        joints_state=joints_state,
        joints_info=joints_info,
        dofs_state=dofs_state,
        dofs_info=dofs_info,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        is_backward=is_backward,
    )
    func_COM_links_entity(
        i_e,
        i_b,
        links_state=links_state,
        links_info=links_info,
        joints_state=joints_state,
        joints_info=joints_info,
        dofs_state=dofs_state,
        dofs_info=dofs_info,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        is_backward=is_backward,
    )
    func_update_geoms_entity(
        i_e,
        i_b,
        entities_info=entities_info,
        geoms_info=geoms_info,
        geoms_state=geoms_state,
        links_state=links_state,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        force_update_fixed_geoms=force_update_fixed_geoms,
        is_backward=is_backward,
    )


@ti.func
def func_update_cartesian_space_batch(
    i_b,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    force_update_fixed_geoms: ti.template(),
    is_backward: ti.template(),
):
    BW = ti.static(is_backward)

    # This loop is considered an inner loop
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_0 in (
        (
            # Dynamic inner loop for forward pass
            range(rigid_global_info.n_awake_entities[i_b])
            if ti.static(static_rigid_sim_config.use_hibernation)
            else range(entities_info.n_links.shape[0])
        )
        if ti.static(not BW)
        else (
            ti.static(range(static_rigid_sim_config.max_n_awake_entities))  # Static inner loop for backward pass
            if ti.static(static_rigid_sim_config.use_hibernation)
            else ti.static(range(static_rigid_sim_config.max_n_links_per_entity))
        )
    ):
        if func_check_index_range(i_0, 0, rigid_global_info.n_awake_entities[i_b], BW):
            i_e = (
                rigid_global_info.awake_entities[i_0, i_b]
                if ti.static(static_rigid_sim_config.use_hibernation)
                else i_0
            )

            func_update_cartesian_space_entity(
                i_e,
                i_b,
                links_state,
                links_info,
                joints_state,
                joints_info,
                dofs_state,
                dofs_info,
                geoms_info,
                geoms_state,
                entities_info,
                rigid_global_info,
                static_rigid_sim_config,
                force_update_fixed_geoms,
                is_backward,
            )


@ti.func
def func_update_cartesian_space(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    force_update_fixed_geoms: ti.template(),
    is_backward: ti.template(),
):
    BW = ti.static(is_backward)

    # This loop must be the outermost loop to be differentiable
    if ti.static(static_rigid_sim_config.use_hibernation):
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(links_state.pos.shape[1]):
            func_update_cartesian_space_batch(
                i_b,
                links_state,
                links_info,
                joints_state,
                joints_info,
                dofs_state,
                dofs_info,
                geoms_info,
                geoms_state,
                entities_info,
                rigid_global_info,
                static_rigid_sim_config,
                force_update_fixed_geoms,
                is_backward,
            )
    else:
        # FIXME: Implement parallelization at tree-level (based on root_idx) instead of entity-level
        ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_e, i_b in ti.ndrange(entities_info.n_links.shape[0], links_state.pos.shape[1]):
            i_l_start = entities_info.link_start[i_e]
            I_l_start = [i_l_start, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l_start
            if links_info.root_idx[I_l_start] == i_l_start:
                for j_e in (
                    range(i_e, entities_info.n_links.shape[0])
                    if ti.static(not BW)
                    else ti.static(range(static_rigid_sim_config.n_entities))
                ):
                    if func_check_index_range(j_e, i_e, static_rigid_sim_config.n_entities, BW):
                        j_l_start = entities_info.link_start[j_e]
                        J_l_start = (
                            [j_l_start, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else j_l_start
                        )
                        if links_info.root_idx[J_l_start] == i_l_start:
                            func_update_cartesian_space_entity(
                                j_e,
                                i_b,
                                links_state,
                                links_info,
                                joints_state,
                                joints_info,
                                dofs_state,
                                dofs_info,
                                geoms_info,
                                geoms_state,
                                entities_info,
                                rigid_global_info,
                                static_rigid_sim_config,
                                force_update_fixed_geoms,
                                is_backward,
                            )


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_update_cartesian_space(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    force_update_fixed_geoms: ti.template(),
    is_backward: ti.template(),
):
    func_update_cartesian_space(
        links_state=links_state,
        links_info=links_info,
        joints_state=joints_state,
        joints_info=joints_info,
        dofs_state=dofs_state,
        dofs_info=dofs_info,
        geoms_info=geoms_info,
        geoms_state=geoms_state,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        force_update_fixed_geoms=force_update_fixed_geoms,
        is_backward=is_backward,
    )
