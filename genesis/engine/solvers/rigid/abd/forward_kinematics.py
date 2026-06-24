"""
Forward kinematics, velocity propagation, and geometry updates for rigid body simulation.

This module contains Quadrants kernels and functions for:
- Forward kinematics computation (link and joint pose updates)
- Velocity propagation through kinematic chains
- Geometry pose and vertex updates
- Center of mass calculations
- AABB updates for collision detection
- Hibernation management for inactive entities
"""

import quadrants as qd

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


@qd.kernel(fastcache=True)
def kernel_forward_kinematics_links_geoms(
    envs_idx: qd.types.ndarray(),
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
    static_rigid_sim_config: qd.template(),
):
    for i_b_ in range(envs_idx.shape[0]):
        i_b = qd.cast(envs_idx[i_b_], qd.i32)
        func_update_cartesian_space_batch(
            i_b=i_b,
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


@qd.kernel(fastcache=True)
def kernel_masked_forward_kinematics_links_geoms(
    envs_mask: qd.types.ndarray(),
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
    static_rigid_sim_config: qd.template(),
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
                geoms_state=geoms_state,
                geoms_info=geoms_info,
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


@qd.kernel(fastcache=True)
def kernel_forward_kinematics(
    envs_idx: qd.types.ndarray(),
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    for i_b_ in range(envs_idx.shape[0]):
        i_b = qd.cast(envs_idx[i_b_], qd.i32)
        func_forward_kinematics_batch(
            i_b=i_b,
            links_state=links_state,
            links_info=links_info,
            joints_state=joints_state,
            joints_info=joints_info,
            dofs_state=dofs_state,
            dofs_info=dofs_info,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
            is_backward=False,
        )
        func_COM_links(
            i_b=i_b,
            links_state=links_state,
            links_info=links_info,
            joints_state=joints_state,
            joints_info=joints_info,
            dofs_state=dofs_state,
            dofs_info=dofs_info,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
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


@qd.kernel(fastcache=True)
def kernel_masked_forward_kinematics(
    envs_mask: qd.types.ndarray(),
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    for i_b in range(envs_mask.shape[0]):
        if envs_mask[i_b]:
            func_forward_kinematics_batch(
                i_b=i_b,
                links_state=links_state,
                links_info=links_info,
                joints_state=joints_state,
                joints_info=joints_info,
                dofs_state=dofs_state,
                dofs_info=dofs_info,
                entities_info=entities_info,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
                is_backward=False,
            )
            func_COM_links(
                i_b=i_b,
                links_state=links_state,
                links_info=links_info,
                joints_state=joints_state,
                joints_info=joints_info,
                dofs_state=dofs_state,
                dofs_info=dofs_info,
                entities_info=entities_info,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
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


@qd.kernel(fastcache=True)
def kernel_forward_velocity(
    envs_idx: qd.types.ndarray(),
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    is_backward: qd.template(),
):
    for i_b_ in range(envs_idx.shape[0]):
        i_b = qd.cast(envs_idx[i_b_], qd.i32)
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


@qd.kernel(fastcache=True)
def kernel_masked_forward_velocity(
    envs_mask: qd.types.ndarray(),
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    is_backward: qd.template(),
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


@qd.func
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
    static_rigid_sim_config: qd.template(),
    is_backward: qd.template(),
):
    BW = qd.static(is_backward)
    i_b = qd.cast(i_b, qd.i32)

    for i_e_ in (
        range(rigid_global_info.n_awake_entities[i_b])
        if qd.static(static_rigid_sim_config.use_hibernation)
        else range(entities_info.n_links.shape[0])
    ):
        if func_check_index_range(
            i_e_, 0, rigid_global_info.n_awake_entities[i_b], static_rigid_sim_config.use_hibernation
        ):
            i_e = (
                rigid_global_info.awake_entities[i_e_, i_b]
                if qd.static(static_rigid_sim_config.use_hibernation)
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


@qd.func
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
    static_rigid_sim_config: qd.template(),
    is_backward: qd.template(),
):
    EPS = rigid_global_info.EPS[None]
    BW = qd.static(is_backward)
    i_b = qd.cast(i_b, qd.i32)

    for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
        links_state.root_COM_bw[i_l, i_b].fill(0.0)
        links_state.mass_sum[i_l, i_b] = 0.0

    for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
        I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l

        mass = links_info.inertial_mass[I_l] + links_state.mass_shift[i_l, i_b]
        (
            links_state.i_pos_bw[i_l, i_b],
            links_state.i_quat[i_l, i_b],
        ) = gu.qd_transform_pos_quat_by_trans_quat(
            links_info.inertial_pos[I_l] + links_state.i_pos_shift[i_l, i_b],
            links_info.inertial_quat[I_l],
            links_state.pos[i_l, i_b],
            links_state.quat[i_l, i_b],
        )

        i_r = links_info.root_idx[I_l]
        links_state.mass_sum[i_r, i_b] = links_state.mass_sum[i_r, i_b] + mass
        qd.atomic_add(links_state.root_COM_bw[i_r, i_b], mass * links_state.i_pos_bw[i_l, i_b])

    for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
        I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l

        i_r = links_info.root_idx[I_l]
        if i_l == i_r:
            mass_sum = links_state.mass_sum[i_l, i_b]
            if mass_sum > EPS:
                links_state.root_COM[i_l, i_b] = links_state.root_COM_bw[i_l, i_b] / links_state.mass_sum[i_l, i_b]
            else:
                links_state.root_COM[i_l, i_b] = links_state.i_pos_bw[i_r, i_b]

    for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
        I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l

        i_r = links_info.root_idx[I_l]
        links_state.root_COM[i_l, i_b] = links_state.root_COM[i_r, i_b]

    for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
        I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l

        i_r = links_info.root_idx[I_l]
        links_state.i_pos[i_l, i_b] = links_state.i_pos_bw[i_l, i_b] - links_state.root_COM[i_l, i_b]

        i_inertial = links_info.inertial_i[I_l]
        i_mass = links_info.inertial_mass[I_l] + links_state.mass_shift[i_l, i_b]
        (
            links_state.cinr_inertial[i_l, i_b],
            links_state.cinr_pos[i_l, i_b],
            links_state.cinr_quat[i_l, i_b],
            links_state.cinr_mass[i_l, i_b],
        ) = gu.qd_transform_inertia_by_trans_quat(
            i_inertial,
            i_mass,
            links_state.i_pos[i_l, i_b],
            links_state.i_quat[i_l, i_b],
            rigid_global_info.EPS[None],
        )

    # j_pos/j_quat are only read by the backward adjoint cache; skip in forward-only mode.
    if qd.static(static_rigid_sim_config.requires_grad):
        for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
            I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l

            if links_info.n_dofs[I_l] > 0:
                i_p = links_info.parent_idx[I_l]

                _i_j = links_info.joint_start[I_l]
                _I_j = [_i_j, i_b] if qd.static(static_rigid_sim_config.batch_joints_info) else _i_j
                joint_type = joints_info.type[_I_j]

                p_pos = qd.Vector.zero(gs.qd_float, 3)
                p_quat = gu.qd_identity_quat()
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
                    ) = gu.qd_transform_pos_quat_by_trans_quat(links_info.pos[I_l], links_info.quat[I_l], p_pos, p_quat)

                    n_joints = links_info.joint_end[I_l] - links_info.joint_start[I_l]

                    for i_j_ in range(n_joints):
                        i_j = i_j_ + links_info.joint_start[I_l]

                        curr_i_j = 0 if qd.static(not BW) else i_j_
                        next_i_j = 0 if qd.static(not BW) else i_j_ + 1

                        if func_check_index_range(
                            i_j,
                            links_info.joint_start[I_l],
                            links_info.joint_end[I_l],
                            BW,
                        ):
                            I_j = [i_j, i_b] if qd.static(static_rigid_sim_config.batch_joints_info) else i_j

                            (
                                links_state.j_pos_bw[i_l, next_i_j, i_b],
                                links_state.j_quat_bw[i_l, next_i_j, i_b],
                            ) = gu.qd_transform_pos_quat_by_trans_quat(
                                joints_info.pos[I_j],
                                gu.qd_identity_quat(),
                                links_state.j_pos_bw[i_l, curr_i_j, i_b],
                                links_state.j_quat_bw[i_l, curr_i_j, i_b],
                            )

                    i_j_ = 0 if qd.static(not BW) else n_joints
                    links_state.j_pos[i_l, i_b] = links_state.j_pos_bw[i_l, i_j_, i_b]
                    links_state.j_quat[i_l, i_b] = links_state.j_quat_bw[i_l, i_j_, i_b]

    for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
        I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l

        if links_info.n_dofs[I_l] > 0:
            for i_j in range(links_info.joint_start[I_l], links_info.joint_end[I_l]):
                offset_pos = links_state.root_COM[i_l, i_b] - joints_state.xanchor[i_j, i_b]
                I_j = [i_j, i_b] if qd.static(static_rigid_sim_config.batch_joints_info) else i_j
                joint_type = joints_info.type[I_j]

                dof_start = joints_info.dof_start[I_j]

                if joint_type == gs.JOINT_TYPE.REVOLUTE:
                    dofs_state.cdof_ang[dof_start, i_b] = joints_state.xaxis[i_j, i_b]
                    dofs_state.cdof_vel[dof_start, i_b] = joints_state.xaxis[i_j, i_b].cross(offset_pos)
                elif joint_type == gs.JOINT_TYPE.PRISMATIC:
                    dofs_state.cdof_ang[dof_start, i_b] = qd.Vector.zero(gs.qd_float, 3)
                    dofs_state.cdof_vel[dof_start, i_b] = joints_state.xaxis[i_j, i_b]
                elif joint_type == gs.JOINT_TYPE.SPHERICAL:
                    xmat_T = gu.qd_quat_to_R(links_state.quat[i_l, i_b], EPS).transpose()
                    for i in qd.static(range(3)):
                        dofs_state.cdof_ang[i + dof_start, i_b] = xmat_T[i, :]
                        dofs_state.cdof_vel[i + dof_start, i_b] = xmat_T[i, :].cross(offset_pos)
                elif joint_type == gs.JOINT_TYPE.FREE:
                    for i in qd.static(range(3)):
                        dofs_state.cdof_ang[i + dof_start, i_b] = qd.Vector.zero(gs.qd_float, 3)
                        dofs_state.cdof_vel[i + dof_start, i_b] = qd.Vector.zero(gs.qd_float, 3)
                        dofs_state.cdof_vel[i + dof_start, i_b][i] = 1.0

                    xmat_T = gu.qd_quat_to_R(links_state.quat[i_l, i_b], EPS).transpose()
                    for i in qd.static(range(3)):
                        dofs_state.cdof_ang[i + dof_start + 3, i_b] = xmat_T[i, :]
                        dofs_state.cdof_vel[i + dof_start + 3, i_b] = xmat_T[i, :].cross(offset_pos)

                for i_d in range(dof_start, joints_info.dof_end[I_j]):
                    dofs_state.cdofvel_ang[i_d, i_b] = dofs_state.cdof_ang[i_d, i_b] * dofs_state.vel[i_d, i_b]
                    dofs_state.cdofvel_vel[i_d, i_b] = dofs_state.cdof_vel[i_d, i_b] * dofs_state.vel[i_d, i_b]


@qd.func
def func_fk_link(
    i_l,
    i_b,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    is_backward: qd.template(),
):
    """Forward kinematics for a SINGLE link: its world pose from its parent's.

    Depends only on the parent link (already finalized one tree level up), so links
    at the same depth are independent and can run in parallel -- see
    func_fk_levels_split. The body is identical to the old per-link loop iteration in
    func_forward_kinematics_entity.
    """
    BW = qd.static(is_backward)
    W = qd.static(func_write_field_if)
    R = qd.static(func_read_field_if)
    WR = qd.static(func_write_and_read_field_if)
    i_b = qd.cast(i_b, qd.i32)

    I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
    I_l0 = (i_l, 0, i_b)

    pos = W(links_state.pos_bw, I_l0, links_info.pos[I_l], BW)
    quat = W(links_state.quat_bw, I_l0, links_info.quat[I_l], BW)
    if links_info.parent_idx[I_l] != -1:
        parent_pos = links_state.pos[links_info.parent_idx[I_l], i_b]
        parent_quat = links_state.quat[links_info.parent_idx[I_l], i_b]
        pos_ = parent_pos + gu.qd_transform_by_quat(links_info.pos[I_l], parent_quat)
        quat_ = gu.qd_transform_quat_by_quat(links_info.quat[I_l], parent_quat)

        pos = W(links_state.pos_bw, I_l0, pos_, BW)
        quat = W(links_state.quat_bw, I_l0, quat_, BW)

    n_joints = links_info.joint_end[I_l] - links_info.joint_start[I_l]

    for i_j_ in range(n_joints):
        i_j = i_j_ + links_info.joint_start[I_l]

        curr_I = (i_l, 0 if qd.static(not BW) else i_j_, i_b)
        next_I = (i_l, 0 if qd.static(not BW) else i_j_ + 1, i_b)

        I_j = [i_j, i_b] if qd.static(static_rigid_sim_config.batch_joints_info) else i_j
        joint_type = joints_info.type[I_j]
        q_start = joints_info.q_start[I_j]
        dof_start = joints_info.dof_start[I_j]
        I_d = [dof_start, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else dof_start

        # compute axis and anchor
        if joint_type == gs.JOINT_TYPE.FREE:
            joints_state.xanchor[i_j, i_b] = qd.Vector(
                [
                    rigid_global_info.qpos[q_start, i_b],
                    rigid_global_info.qpos[q_start + 1, i_b],
                    rigid_global_info.qpos[q_start + 2, i_b],
                ]
            )
            joints_state.xaxis[i_j, i_b] = qd.Vector([0.0, 0.0, 1.0])
        elif joint_type == gs.JOINT_TYPE.FIXED:
            pass
        else:
            axis = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)
            if joint_type == gs.JOINT_TYPE.REVOLUTE:
                axis = dofs_info.motion_ang[I_d]
            elif joint_type == gs.JOINT_TYPE.PRISMATIC:
                axis = dofs_info.motion_vel[I_d]

            pos_ = R(links_state.pos_bw, curr_I, pos, BW)
            quat_ = R(links_state.quat_bw, curr_I, quat, BW)

            joints_state.xanchor[i_j, i_b] = gu.qd_transform_by_quat(joints_info.pos[I_j], quat_) + pos_
            joints_state.xaxis[i_j, i_b] = gu.qd_transform_by_quat(axis, quat_)

        if joint_type == gs.JOINT_TYPE.FREE:
            pos_ = qd.Vector(
                [
                    rigid_global_info.qpos[q_start, i_b],
                    rigid_global_info.qpos[q_start + 1, i_b],
                    rigid_global_info.qpos[q_start + 2, i_b],
                ],
                dt=gs.qd_float,
            )
            quat_ = qd.Vector(
                [
                    rigid_global_info.qpos[q_start + 3, i_b],
                    rigid_global_info.qpos[q_start + 4, i_b],
                    rigid_global_info.qpos[q_start + 5, i_b],
                    rigid_global_info.qpos[q_start + 6, i_b],
                ],
                dt=gs.qd_float,
            )
            quat_ = quat_ / quat_.norm()
            pos = WR(links_state.pos_bw, next_I, pos_, BW)
            quat = WR(links_state.quat_bw, next_I, quat_, BW)

            xyz = gu.qd_quat_to_xyz(quat, rigid_global_info.EPS[None])
            for j in qd.static(range(3)):
                dofs_state.pos[dof_start + j, i_b] = pos[j]
                dofs_state.pos[dof_start + 3 + j, i_b] = xyz[j]
        elif joint_type == gs.JOINT_TYPE.FIXED:
            pass
        elif joint_type == gs.JOINT_TYPE.SPHERICAL:
            qloc = qd.Vector(
                [
                    rigid_global_info.qpos[q_start, i_b],
                    rigid_global_info.qpos[q_start + 1, i_b],
                    rigid_global_info.qpos[q_start + 2, i_b],
                    rigid_global_info.qpos[q_start + 3, i_b],
                ],
                dt=gs.qd_float,
            )
            xyz = gu.qd_quat_to_xyz(qloc, rigid_global_info.EPS[None])
            for j in qd.static(range(3)):
                dofs_state.pos[dof_start + j, i_b] = xyz[j]
            quat_ = gu.qd_transform_quat_by_quat(qloc, R(links_state.quat_bw, curr_I, quat, BW))
            quat = WR(links_state.quat_bw, next_I, quat_, BW)
            pos_ = joints_state.xanchor[i_j, i_b] - gu.qd_transform_by_quat(joints_info.pos[I_j], quat)
            pos = W(links_state.pos_bw, next_I, pos_, BW)
        elif joint_type == gs.JOINT_TYPE.REVOLUTE:
            axis = dofs_info.motion_ang[I_d]
            dofs_state.pos[dof_start, i_b] = (
                rigid_global_info.qpos[q_start, i_b] - rigid_global_info.qpos0[q_start, i_b]
            )
            qloc = gu.qd_rotvec_to_quat(axis * dofs_state.pos[dof_start, i_b], rigid_global_info.EPS[None])
            quat_ = gu.qd_transform_quat_by_quat(qloc, R(links_state.quat_bw, curr_I, quat, BW))
            quat = WR(links_state.quat_bw, next_I, quat_, BW)
            pos_ = joints_state.xanchor[i_j, i_b] - gu.qd_transform_by_quat(joints_info.pos[I_j], quat)
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
    I_jf = (i_l, 0 if qd.static(not BW) else n_joints, i_b)
    if not (links_info.parent_idx[I_l] == -1 and links_info.is_fixed[I_l]):
        links_state.pos[i_l, i_b] = R(links_state.pos_bw, I_jf, pos, BW)
        links_state.quat[i_l, i_b] = R(links_state.quat_bw, I_jf, quat, BW)


@qd.func
def func_fk_levels_split(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    static_rigid_sim_config: qd.template(),
    rigid_global_info: array_class.RigidGlobalInfo,
    is_backward: qd.template(),
):
    """Level-scheduled forward kinematics: one per-(link, env)-parallel pass per tree
    depth, with the implicit kernel barrier between passes ordering parent before
    child. Replaces the one-thread-per-env serial chain walk for the (non-hibernation)
    forward pass; raises occupancy from ~1 thread/env to n_links*n_envs."""
    n_links = links_state.pos.shape[0]
    _B = links_state.pos.shape[1]
    for d in qd.static(range(static_rigid_sim_config.max_link_depth + 1)):
        qd.loop_config(
            name="fk_level",
            serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL),
            block_dim=64,
        )
        for i_l, i_b in qd.ndrange(n_links, _B):
            I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
            if links_info.depth[I_l] == d:
                func_fk_link(
                    i_l,
                    i_b,
                    links_state,
                    links_info,
                    joints_state,
                    joints_info,
                    dofs_state,
                    dofs_info,
                    rigid_global_info,
                    static_rigid_sim_config,
                    is_backward,
                )


@qd.func
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
    static_rigid_sim_config: qd.template(),
    is_backward: qd.template(),
):
    i_b = qd.cast(i_b, qd.i32)
    for i_l_ in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
        func_fk_link(
            gs.qd_int(i_l_),
            i_b,
            links_state,
            links_info,
            joints_state,
            joints_info,
            dofs_state,
            dofs_info,
            rigid_global_info,
            static_rigid_sim_config,
            is_backward,
        )


@qd.func
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
    static_rigid_sim_config: qd.template(),
    is_backward: qd.template(),
):
    BW = qd.static(is_backward)
    i_b = qd.cast(i_b, qd.i32)

    for i_e_ in (
        range(rigid_global_info.n_awake_entities[i_b])
        if qd.static(static_rigid_sim_config.use_hibernation)
        else range(entities_info.n_links.shape[0])
    ):
        if func_check_index_range(
            i_e_, 0, rigid_global_info.n_awake_entities[i_b], static_rigid_sim_config.use_hibernation
        ):
            i_e = (
                rigid_global_info.awake_entities[i_e_, i_b]
                if qd.static(static_rigid_sim_config.use_hibernation)
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


@qd.kernel(fastcache=True)
def kernel_forward_kinematics_entity(
    i_e: qd.int32,
    envs_idx: qd.types.ndarray(),
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    for i_b_ in range(envs_idx.shape[0]):
        i_b = qd.cast(envs_idx[i_b_], qd.i32)

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


@qd.func
def func_update_geoms_entity(
    i_e,
    i_b,
    entities_info: array_class.EntitiesInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    force_update_fixed_geoms: qd.template(),
    is_backward: qd.template(),
):
    """
    NOTE: this only update geom pose, not its verts and else.
    """
    BW = qd.static(is_backward)
    i_b = qd.cast(i_b, qd.i32)

    for i_g_ in (
        # Dynamic inner loop for forward pass
        range(entities_info.n_geoms[i_e])
        if qd.static(not BW)
        else qd.static(range(static_rigid_sim_config.max_n_geoms_per_entity))  # Static inner loop for backward pass
    ):
        i_g = entities_info.geom_start[i_e] + i_g_
        if func_check_index_range(i_g, entities_info.geom_start[i_e], entities_info.geom_end[i_e], BW):
            if force_update_fixed_geoms or not geoms_info.is_fixed[i_g]:
                (
                    geoms_state.pos[i_g, i_b],
                    geoms_state.quat[i_g, i_b],
                ) = gu.qd_transform_pos_quat_by_trans_quat(
                    geoms_info.pos[i_g],
                    geoms_info.quat[i_g],
                    links_state.pos[geoms_info.link_idx[i_g], i_b],
                    links_state.quat[geoms_info.link_idx[i_g], i_b],
                )
                geoms_state.verts_updated[i_g, i_b] = False


@qd.func
def func_update_geoms_batch(
    i_b,
    entities_info: array_class.EntitiesInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    force_update_fixed_geoms: qd.template(),
    is_backward: qd.template(),
):
    """
    NOTE: this only update geom pose, not its verts and else.
    """
    BW = qd.static(is_backward)
    i_b = qd.cast(i_b, qd.i32)

    for i_e_ in (
        range(rigid_global_info.n_awake_entities[i_b])
        if qd.static(static_rigid_sim_config.use_hibernation)
        else range(entities_info.n_links.shape[0])
    ):
        if func_check_index_range(
            i_e_, 0, rigid_global_info.n_awake_entities[i_b], static_rigid_sim_config.use_hibernation
        ):
            i_e = (
                rigid_global_info.awake_entities[i_e_, i_b]
                if qd.static(static_rigid_sim_config.use_hibernation)
                else i_e_
            )

            func_update_geoms_entity(
                i_e,
                i_b,
                entities_info,
                geoms_state,
                geoms_info,
                links_state,
                rigid_global_info,
                static_rigid_sim_config,
                force_update_fixed_geoms,
                is_backward,
            )


@qd.func
def func_update_geoms(
    entities_info: array_class.EntitiesInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    force_update_fixed_geoms: qd.template(),
    is_backward: qd.template(),
):
    # This loop must be the outermost loop to be differentiable
    if qd.static(static_rigid_sim_config.use_hibernation):
        qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(links_state.pos.shape[1]):
            func_update_geoms_batch(
                i_b,
                entities_info,
                geoms_state,
                geoms_info,
                links_state,
                rigid_global_info,
                static_rigid_sim_config,
                force_update_fixed_geoms,
                is_backward,
            )
    else:
        qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_e, i_b in qd.ndrange(entities_info.n_links.shape[0], links_state.pos.shape[1]):
            func_update_geoms_entity(
                i_e,
                i_b,
                entities_info,
                geoms_state,
                geoms_info,
                links_state,
                rigid_global_info,
                static_rigid_sim_config,
                force_update_fixed_geoms,
                is_backward,
            )


@qd.kernel(fastcache=True)
def kernel_update_geoms(
    envs_idx: qd.types.ndarray(),
    entities_info: array_class.EntitiesInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    force_update_fixed_geoms: qd.template(),
):
    for i_b_ in range(envs_idx.shape[0]):
        i_b = qd.cast(envs_idx[i_b_], qd.i32)

        func_update_geoms_batch(
            i_b,
            entities_info,
            geoms_state,
            geoms_info,
            links_state,
            rigid_global_info,
            static_rigid_sim_config,
            force_update_fixed_geoms,
            is_backward=False,
        )


@qd.func
def func_forward_velocity_entity(
    i_e,
    i_b,
    entities_info: array_class.EntitiesInfo,
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    is_backward: qd.template(),
):
    BW = qd.static(is_backward)
    W = qd.static(func_write_field_if)
    R = qd.static(func_read_field_if)
    A = qd.static(func_atomic_add_if)
    i_b = qd.cast(i_b, qd.i32)

    for i_l_ in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
        i_l = gs.qd_int(i_l_)

        I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
        n_joints = links_info.joint_end[I_l] - links_info.joint_start[I_l]

        I_j0 = (i_l, 0, i_b)
        cvel_vel = W(links_state.cd_vel_bw, I_j0, qd.Vector.zero(gs.qd_float, 3), BW)
        cvel_ang = W(links_state.cd_ang_bw, I_j0, qd.Vector.zero(gs.qd_float, 3), BW)

        if links_info.parent_idx[I_l] != -1:
            cvel_vel = W(links_state.cd_vel_bw, I_j0, links_state.cd_vel[links_info.parent_idx[I_l], i_b], BW)
            cvel_ang = W(links_state.cd_ang_bw, I_j0, links_state.cd_ang[links_info.parent_idx[I_l], i_b], BW)

        for i_j_ in range(n_joints):
            i_j = i_j_ + links_info.joint_start[I_l]

            I_j = [i_j, i_b] if qd.static(static_rigid_sim_config.batch_joints_info) else i_j
            joint_type = joints_info.type[I_j]
            dof_start = joints_info.dof_start[I_j]

            curr_I = (i_l, 0 if qd.static(not BW) else i_j_, i_b)
            next_I = (i_l, 0 if qd.static(not BW) else i_j_ + 1, i_b)

            if joint_type == gs.JOINT_TYPE.FREE:
                for i_3 in qd.static(range(3)):
                    _vel = dofs_state.cdof_vel[dof_start + i_3, i_b] * dofs_state.vel[dof_start + i_3, i_b]
                    _ang = dofs_state.cdof_ang[dof_start + i_3, i_b] * dofs_state.vel[dof_start + i_3, i_b]

                    cvel_vel = cvel_vel + A(links_state.cd_vel_bw, curr_I, _vel, BW)
                    cvel_ang = cvel_ang + A(links_state.cd_ang_bw, curr_I, _ang, BW)

                for i_3 in qd.static(range(3)):
                    (
                        dofs_state.cdofd_ang[dof_start + i_3, i_b],
                        dofs_state.cdofd_vel[dof_start + i_3, i_b],
                    ) = qd.Vector.zero(gs.qd_float, 3), qd.Vector.zero(gs.qd_float, 3)

                    (
                        dofs_state.cdofd_ang[dof_start + i_3 + 3, i_b],
                        dofs_state.cdofd_vel[dof_start + i_3 + 3, i_b],
                    ) = gu.motion_cross_motion(
                        R(links_state.cd_ang_bw, curr_I, cvel_ang, BW),
                        R(links_state.cd_vel_bw, curr_I, cvel_vel, BW),
                        dofs_state.cdof_ang[dof_start + i_3 + 3, i_b],
                        dofs_state.cdof_vel[dof_start + i_3 + 3, i_b],
                    )

                if qd.static(BW):
                    links_state.cd_vel_bw[next_I] = links_state.cd_vel_bw[curr_I]
                    links_state.cd_ang_bw[next_I] = links_state.cd_ang_bw[curr_I]

                for i_3 in qd.static(range(3)):
                    _vel = dofs_state.cdof_vel[dof_start + i_3 + 3, i_b] * dofs_state.vel[dof_start + i_3 + 3, i_b]
                    _ang = dofs_state.cdof_ang[dof_start + i_3 + 3, i_b] * dofs_state.vel[dof_start + i_3 + 3, i_b]
                    cvel_vel = cvel_vel + A(links_state.cd_vel_bw, next_I, _vel, BW)
                    cvel_ang = cvel_ang + A(links_state.cd_ang_bw, next_I, _ang, BW)

            else:
                for i_d in range(dof_start, joints_info.dof_end[I_j]):
                    dofs_state.cdofd_ang[i_d, i_b], dofs_state.cdofd_vel[i_d, i_b] = gu.motion_cross_motion(
                        R(links_state.cd_ang_bw, curr_I, cvel_ang, BW),
                        R(links_state.cd_vel_bw, curr_I, cvel_vel, BW),
                        dofs_state.cdof_ang[i_d, i_b],
                        dofs_state.cdof_vel[i_d, i_b],
                    )

                if qd.static(BW):
                    links_state.cd_vel_bw[next_I] = links_state.cd_vel_bw[curr_I]
                    links_state.cd_ang_bw[next_I] = links_state.cd_ang_bw[curr_I]

                for i_d in range(dof_start, joints_info.dof_end[I_j]):
                    _vel = dofs_state.cdof_vel[i_d, i_b] * dofs_state.vel[i_d, i_b]
                    _ang = dofs_state.cdof_ang[i_d, i_b] * dofs_state.vel[i_d, i_b]
                    cvel_vel = cvel_vel + A(links_state.cd_vel_bw, next_I, _vel, BW)
                    cvel_ang = cvel_ang + A(links_state.cd_ang_bw, next_I, _ang, BW)

        I_jf = (i_l, 0 if qd.static(not BW) else n_joints, i_b)
        links_state.cd_vel[i_l, i_b] = R(links_state.cd_vel_bw, I_jf, cvel_vel, BW)
        links_state.cd_ang[i_l, i_b] = R(links_state.cd_ang_bw, I_jf, cvel_ang, BW)


@qd.func
def func_forward_velocity_batch(
    i_b,
    entities_info: array_class.EntitiesInfo,
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    is_backward: qd.template(),
):
    BW = qd.static(is_backward)
    i_b = qd.cast(i_b, qd.i32)

    for i_e_ in (
        range(rigid_global_info.n_awake_entities[i_b])
        if qd.static(static_rigid_sim_config.use_hibernation)
        else range(entities_info.n_links.shape[0])
    ):
        if func_check_index_range(
            i_e_, 0, rigid_global_info.n_awake_entities[i_b], static_rigid_sim_config.use_hibernation
        ):
            i_e = (
                rigid_global_info.awake_entities[i_e_, i_b]
                if qd.static(static_rigid_sim_config.use_hibernation)
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


@qd.func
def func_forward_velocity(
    entities_info: array_class.EntitiesInfo,
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    is_backward: qd.template(),
):
    # This loop must be the outermost loop to be differentiable
    if qd.static(static_rigid_sim_config.use_hibernation):
        qd.loop_config(name="forward_velocity_batch", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
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
        # OPT-1: when a contiguous dynamic-entity window is set (hibernation off), restrict the outer
        # ndrange to [dynamic_entity_offset_, dynamic_entity_offset_ + n_dynamic_entities_), skipping
        # static 0-DOF entities whose velocity is always zero and whose FK outputs are constant.
        _HAS_WINDOW_FV = qd.static(
            static_rigid_sim_config.n_dynamic_entities_ > 0 and not static_rigid_sim_config.use_hibernation
        )
        _ENTITY_BASE_FV = qd.static(static_rigid_sim_config.dynamic_entity_offset_ if _HAS_WINDOW_FV else 0)
        _N_GRID_ENTITIES_FV = qd.static(static_rigid_sim_config.n_dynamic_entities_) if _HAS_WINDOW_FV else entities_info.n_links.shape[0]
        qd.loop_config(
            name="forward_velocity_entity",
            serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL),
        )
        for i_e_local, i_b in qd.ndrange(_N_GRID_ENTITIES_FV, links_state.pos.shape[1]):
            i_e = i_e_local + _ENTITY_BASE_FV
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


@qd.kernel(fastcache=True)
def kernel_update_verts_for_geoms(
    geoms_idx: qd.types.ndarray(),
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    free_verts_state: array_class.VertsState,
    fixed_verts_state: array_class.VertsState,
    static_rigid_sim_config: qd.template(),
):
    n_geoms = geoms_idx.shape[0]
    _B = geoms_state.verts_updated.shape[1]

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_g_, i_b in qd.ndrange(n_geoms, _B):
        i_g = geoms_idx[i_g_]
        func_update_verts_for_geom(i_g, i_b, geoms_state, geoms_info, verts_info, free_verts_state, fixed_verts_state)


@qd.func
def func_update_verts_for_geom(
    i_g: qd.i32,
    i_b: qd.i32,
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
                fixed_verts_state.pos[verts_state_idx] = gu.qd_transform_by_trans_quat(
                    verts_info.init_pos[i_v], geoms_state.pos[i_g, i_b], geoms_state.quat[i_g, i_b]
                )
            for j_b in range(_B):
                geoms_state.verts_updated[i_g, j_b] = True
        else:
            for i_v in range(i_v_start, geoms_info.vert_end[i_g]):
                verts_state_idx = verts_info.verts_state_idx[i_v]
                free_verts_state.pos[verts_state_idx, i_b] = gu.qd_transform_by_trans_quat(
                    verts_info.init_pos[i_v], geoms_state.pos[i_g, i_b], geoms_state.quat[i_g, i_b]
                )
            geoms_state.verts_updated[i_g, i_b] = True


@qd.func
def func_update_all_verts(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    free_verts_state: array_class.VertsState,
    fixed_verts_state: array_class.VertsState,
    static_rigid_sim_config: qd.template(),
):
    n_geoms, _B = geoms_state.pos.shape

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_g, i_b in qd.ndrange(n_geoms, _B):
        func_update_verts_for_geom(i_g, i_b, geoms_state, geoms_info, verts_info, free_verts_state, fixed_verts_state)


@qd.kernel(fastcache=True)
def kernel_update_all_verts(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    free_verts_state: array_class.VertsState,
    fixed_verts_state: array_class.VertsState,
    static_rigid_sim_config: qd.template(),
):
    func_update_all_verts(
        geoms_state, geoms_info, verts_info, free_verts_state, fixed_verts_state, static_rigid_sim_config
    )


@qd.kernel
def kernel_update_geom_aabbs(
    geoms_state: array_class.GeomsState,
    geoms_init_AABB: array_class.GeomsInitAABB,
    static_rigid_sim_config: qd.template(),
):
    n_geoms = geoms_state.pos.shape[0]
    _B = geoms_state.pos.shape[1]

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_g, i_b in qd.ndrange(n_geoms, _B):
        g_pos = geoms_state.pos[i_g, i_b]
        g_quat = geoms_state.quat[i_g, i_b]

        lower = gu.qd_vec3(qd.math.inf)
        upper = gu.qd_vec3(-qd.math.inf)
        for i_corner in qd.static(range(8)):
            corner_pos = gu.qd_transform_by_trans_quat(geoms_init_AABB[i_g, i_corner], g_pos, g_quat)
            lower = qd.min(lower, corner_pos)
            upper = qd.max(upper, corner_pos)

        geoms_state.aabb_min[i_g, i_b] = lower
        geoms_state.aabb_max[i_g, i_b] = upper


@qd.kernel(fastcache=True)
def kernel_update_vgeoms(
    vgeoms_info: array_class.VGeomsInfo,
    vgeoms_state: array_class.VGeomsState,
    links_state: array_class.LinksState,
    static_rigid_sim_config: qd.template(),
):
    """
    Vgeoms are only for visualization purposes. Updates vgeom world transforms from link state.
    """
    n_vgeoms = vgeoms_info.link_idx.shape[0]
    _B = links_state.pos.shape[1]

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_g, i_b in qd.ndrange(n_vgeoms, _B):
        i_l = vgeoms_info.link_idx[i_g]
        vgeoms_state.pos[i_g, i_b], vgeoms_state.quat[i_g, i_b] = gu.qd_transform_pos_quat_by_trans_quat(
            vgeoms_info.pos[i_g], vgeoms_info.quat[i_g], links_state.pos[i_l, i_b], links_state.quat[i_l, i_b]
        )


@qd.kernel(fastcache=True)
def kernel_update_vverts_for_vgeoms(
    vgeoms_idx: qd.types.ndarray(),
    vgeoms_info: array_class.VGeomsInfo,
    vgeoms_state: array_class.VGeomsState,
    vverts_info: array_class.VVertsInfo,
    vverts_state: array_class.VVertsState,
    static_rigid_sim_config: qd.template(),
):
    """
    Refresh vverts_state.pos for the requested vgeom range from FK output. Only iterates vverts that have a slot in
    the custom buffer (vverts_state_idx != -1); other vverts are computed on the fly by their consumers, so they have
    no persistent storage here.
    """
    n_vgeoms_in = vgeoms_idx.shape[0]
    _B = vgeoms_state.pos.shape[1]

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_vg_, i_b in qd.ndrange(n_vgeoms_in, _B):
        i_vg = vgeoms_idx[i_vg_]
        v_start = vgeoms_info.vvert_start[i_vg]
        v_end = vgeoms_info.vvert_end[i_vg]
        for i_vv in range(v_start, v_end):
            i_state = vverts_info.vverts_state_idx[i_vv]
            if i_state >= 0:
                vverts_state.pos[i_state, i_b] = gu.qd_transform_by_trans_quat(
                    vverts_info.init_pos[i_vv], vgeoms_state.pos[i_vg, i_b], vgeoms_state.quat[i_vg, i_b]
                )


@qd.func
def func_hibernate__for_all_awake_islands_either_hiberanate_or_update_aabb_sort_buffer(
    dofs_state: array_class.DofsState,
    entities_state: array_class.EntitiesState,
    entities_info: array_class.EntitiesInfo,
    links_state: array_class.LinksState,
    geoms_state: array_class.GeomsState,
    collider_state: array_class.ColliderState,
    unused__rigid_global_info: array_class.RigidGlobalInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    contact_island_state: array_class.ContactIslandState,
    errno: qd.Tensor,
):
    _B = entities_state.hibernated.shape[1]

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
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
                        if qd.abs(dofs_state.acc[i_d, i_b]) > max_acc or qd.abs(dofs_state.vel[i_d, i_b]) > max_vel:
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


@qd.func
def func_aggregate_awake_entities(
    entities_state: array_class.EntitiesState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    n_entities = entities_state.hibernated.shape[0]
    _B = entities_state.hibernated.shape[1]

    # Reset counts once per batch (not per entity!)
    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        rigid_global_info.n_awake_entities[i_b] = 0
        rigid_global_info.n_awake_links[i_b] = 0
        rigid_global_info.n_awake_dofs[i_b] = 0

    # Count awake entities
    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_e, i_b in qd.ndrange(n_entities, _B):
        if entities_state.hibernated[i_e, i_b] or entities_info.n_dofs[i_e] == 0:
            continue

        next_awake_entity_idx = qd.atomic_add(rigid_global_info.n_awake_entities[i_b], 1)
        rigid_global_info.awake_entities[next_awake_entity_idx, i_b] = i_e

        n_dofs = entities_info.n_dofs[i_e]
        entity_dofs_base_idx: qd.int32 = entities_info.dof_start[i_e]
        awake_dofs_base_idx = qd.atomic_add(rigid_global_info.n_awake_dofs[i_b], n_dofs)
        for i_d_ in range(n_dofs):
            rigid_global_info.awake_dofs[awake_dofs_base_idx + i_d_, i_b] = entity_dofs_base_idx + i_d_

        n_links = entities_info.n_links[i_e]
        entity_links_base_idx: qd.int32 = entities_info.link_start[i_e]
        awake_links_base_idx = qd.atomic_add(rigid_global_info.n_awake_links[i_b], n_links)
        for i_l_ in range(n_links):
            rigid_global_info.awake_links[awake_links_base_idx + i_l_, i_b] = entity_links_base_idx + i_l_


@qd.func
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


@qd.func
def func_com_pass1_zero_link(
    i_l,
    i_b,
    links_state: array_class.LinksState,
):
    links_state.root_COM_bw[i_l, i_b].fill(0.0)
    links_state.mass_sum[i_l, i_b] = 0.0


@qd.func
def func_com_pass2_accumulate_entity(
    i_e,
    i_b,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    entities_info: array_class.EntitiesInfo,
    static_rigid_sim_config: qd.template(),
    is_backward: qd.template(),
):
    """
    Pass-2 inertial transform + mass/CoM accumulation for all links of a single
    entity.

    Body is lifted verbatim from the second `for i_l_` block of the original
    `func_COM_links_entity` so the per-entity link traversal order, the plain
    RMW on `mass_sum`, and the `qd.atomic_add` on `root_COM_bw` all match the
    baseline bit-exact. The caller is expected to wrap this in the same
    2-level outer loop (root-entity check, then all entities sharing that
    root) that the original entity-level cartesian-update launch used.
    """
    BW = qd.static(is_backward)
    i_b = qd.cast(i_b, qd.i32)

    for i_l_ in (
        range(entities_info.link_start[i_e], entities_info.link_end[i_e])
        if qd.static(not BW)
        else qd.static(range(static_rigid_sim_config.max_n_links_per_entity))
    ):
        i_l = i_l_ if qd.static(not BW) else (i_l_ + entities_info.link_start[i_e])

        if func_check_index_range(i_l, entities_info.link_start[i_e], entities_info.link_end[i_e], BW):
            I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l

            mass = links_info.inertial_mass[I_l] + links_state.mass_shift[i_l, i_b]
            (
                links_state.i_pos_bw[i_l, i_b],
                links_state.i_quat[i_l, i_b],
            ) = gu.qd_transform_pos_quat_by_trans_quat(
                links_info.inertial_pos[I_l] + links_state.i_pos_shift[i_l, i_b],
                links_info.inertial_quat[I_l],
                links_state.pos[i_l, i_b],
                links_state.quat[i_l, i_b],
            )

            i_r = links_info.root_idx[I_l]
            links_state.mass_sum[i_r, i_b] = links_state.mass_sum[i_r, i_b] + mass
            links_state.root_COM_bw[i_r, i_b] = links_state.root_COM_bw[i_r, i_b] + mass * links_state.i_pos_bw[i_l, i_b]


@qd.func
def func_com_pass3_normalize_root_link(
    i_l,
    i_b,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    EPS = rigid_global_info.EPS[None]
    I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l

    i_r = links_info.root_idx[I_l]
    if i_l == i_r:
        mass_sum = links_state.mass_sum[i_l, i_b]
        if mass_sum > EPS:
            links_state.root_COM[i_l, i_b] = links_state.root_COM_bw[i_l, i_b] / mass_sum
        else:
            links_state.root_COM[i_l, i_b] = links_state.i_pos_bw[i_r, i_b]


@qd.func
def func_com_pass4_broadcast_link(
    i_l,
    i_b,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    static_rigid_sim_config: qd.template(),
):
    I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
    i_r = links_info.root_idx[I_l]
    # For the root link, `i_l == i_r` so this is a self-assignment (no-op), but
    # making the branch unconditional keeps the kernel uniform and avoids a
    # divergent write path.
    links_state.root_COM[i_l, i_b] = links_state.root_COM[i_r, i_b]


@qd.func
def func_com_pass5_inertial_link(
    i_l,
    i_b,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l

    links_state.i_pos[i_l, i_b] = links_state.i_pos_bw[i_l, i_b] - links_state.root_COM[i_l, i_b]

    i_inertial = links_info.inertial_i[I_l]
    i_mass = links_info.inertial_mass[I_l] + links_state.mass_shift[i_l, i_b]
    (
        links_state.cinr_inertial[i_l, i_b],
        links_state.cinr_pos[i_l, i_b],
        links_state.cinr_quat[i_l, i_b],
        links_state.cinr_mass[i_l, i_b],
    ) = gu.qd_transform_inertia_by_trans_quat(
        i_inertial,
        i_mass,
        links_state.i_pos[i_l, i_b],
        links_state.i_quat[i_l, i_b],
        rigid_global_info.EPS[None],
    )


@qd.func
def func_com_pass6_joint_pose_link(
    i_l,
    i_b,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_info: array_class.JointsInfo,
    static_rigid_sim_config: qd.template(),
    is_backward: qd.template(),
):
    BW = qd.static(is_backward)
    I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l

    # j_pos/j_quat are only read by the backward adjoint cache; skip in forward-only mode.
    if qd.static(static_rigid_sim_config.requires_grad):
        if links_info.n_dofs[I_l] > 0:
            i_p = links_info.parent_idx[I_l]

            _i_j = links_info.joint_start[I_l]
            _I_j = [_i_j, i_b] if qd.static(static_rigid_sim_config.batch_joints_info) else _i_j
            joint_type = joints_info.type[_I_j]

            p_pos = qd.Vector.zero(gs.qd_float, 3)
            p_quat = gu.qd_identity_quat()
            if i_p != -1:
                p_pos = links_state.pos[i_p, i_b]
                p_quat = links_state.quat[i_p, i_b]

            if joint_type == gs.JOINT_TYPE.FREE or (links_info.is_fixed[I_l] and i_p == -1):
                links_state.j_pos[i_l, i_b] = links_state.pos[i_l, i_b]
                links_state.j_quat[i_l, i_b] = links_state.quat[i_l, i_b]
            else:
                acc_pos, acc_quat = gu.qd_transform_pos_quat_by_trans_quat(
                    links_info.pos[I_l], links_info.quat[I_l], p_pos, p_quat,
                )
                if qd.static(BW):
                    links_state.j_pos_bw[i_l, 0, i_b] = acc_pos
                    links_state.j_quat_bw[i_l, 0, i_b] = acc_quat

                n_joints = links_info.joint_end[I_l] - links_info.joint_start[I_l]

                for i_j_ in (
                    range(n_joints)
                    if qd.static(not BW)
                    else qd.static(range(static_rigid_sim_config.max_n_joints_per_link))
                ):
                    i_j = i_j_ + links_info.joint_start[I_l]

                    if func_check_index_range(
                        i_j,
                        links_info.joint_start[I_l],
                        links_info.joint_end[I_l],
                        BW,
                    ):
                        I_j = [i_j, i_b] if qd.static(static_rigid_sim_config.batch_joints_info) else i_j

                        if qd.static(not BW):
                            acc_pos = acc_pos + gu.qd_transform_by_quat(joints_info.pos[I_j], acc_quat)
                        else:
                            curr_i_j = i_j_
                            next_i_j = i_j_ + 1
                            prev_quat = links_state.j_quat_bw[i_l, curr_i_j, i_b]
                            links_state.j_pos_bw[i_l, next_i_j, i_b] = (
                                links_state.j_pos_bw[i_l, curr_i_j, i_b]
                                + gu.qd_transform_by_quat(joints_info.pos[I_j], prev_quat)
                            )
                            links_state.j_quat_bw[i_l, next_i_j, i_b] = prev_quat

                if qd.static(not BW):
                    links_state.j_pos[i_l, i_b] = acc_pos
                    links_state.j_quat[i_l, i_b] = acc_quat
                else:
                    links_state.j_pos[i_l, i_b] = links_state.j_pos_bw[i_l, n_joints, i_b]
                    links_state.j_quat[i_l, i_b] = links_state.j_quat_bw[i_l, n_joints, i_b]


@qd.func
def func_com_pass7_cdof_link(
    i_l,
    i_b,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    is_backward: qd.template(),
):
    EPS = rigid_global_info.EPS[None]
    BW = qd.static(is_backward)
    I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l

    if links_info.n_dofs[I_l] > 0:
        for i_j_ in (
            range(links_info.joint_start[I_l], links_info.joint_end[I_l])
            if qd.static(not BW)
            else qd.static(range(static_rigid_sim_config.max_n_joints_per_link))
        ):
            i_j = i_j_ if qd.static(not BW) else (i_j_ + links_info.joint_start[I_l])

            if func_check_index_range(i_j, links_info.joint_start[I_l], links_info.joint_end[I_l], BW):
                offset_pos = links_state.root_COM[i_l, i_b] - joints_state.xanchor[i_j, i_b]
                I_j = [i_j, i_b] if qd.static(static_rigid_sim_config.batch_joints_info) else i_j
                joint_type = joints_info.type[I_j]

                dof_start = joints_info.dof_start[I_j]

                if joint_type == gs.JOINT_TYPE.REVOLUTE:
                    dofs_state.cdof_ang[dof_start, i_b] = joints_state.xaxis[i_j, i_b]
                    dofs_state.cdof_vel[dof_start, i_b] = joints_state.xaxis[i_j, i_b].cross(offset_pos)
                elif joint_type == gs.JOINT_TYPE.PRISMATIC:
                    dofs_state.cdof_ang[dof_start, i_b] = qd.Vector.zero(gs.qd_float, 3)
                    dofs_state.cdof_vel[dof_start, i_b] = joints_state.xaxis[i_j, i_b]
                elif joint_type == gs.JOINT_TYPE.SPHERICAL:
                    xmat_T = gu.qd_quat_to_R(links_state.quat[i_l, i_b], EPS).transpose()
                    for i in qd.static(range(3)):
                        dofs_state.cdof_ang[i + dof_start, i_b] = xmat_T[i, :]
                        dofs_state.cdof_vel[i + dof_start, i_b] = xmat_T[i, :].cross(offset_pos)
                elif joint_type == gs.JOINT_TYPE.FREE:
                    for i in qd.static(range(3)):
                        dofs_state.cdof_ang[i + dof_start, i_b] = qd.Vector.zero(gs.qd_float, 3)
                        dofs_state.cdof_vel[i + dof_start, i_b] = qd.Vector.zero(gs.qd_float, 3)
                        dofs_state.cdof_vel[i + dof_start, i_b][i] = 1.0

                    xmat_T = gu.qd_quat_to_R(links_state.quat[i_l, i_b], EPS).transpose()
                    for i in qd.static(range(3)):
                        dofs_state.cdof_ang[i + dof_start + 3, i_b] = xmat_T[i, :]
                        dofs_state.cdof_vel[i + dof_start + 3, i_b] = xmat_T[i, :].cross(offset_pos)

                for i_d_ in (
                    range(dof_start, joints_info.dof_end[I_j])
                    if qd.static(not BW)
                    else qd.static(range(static_rigid_sim_config.max_n_dofs_per_joint))
                ):
                    i_d = i_d_ if qd.static(not BW) else (i_d_ + dof_start)
                    if func_check_index_range(i_d, dof_start, joints_info.dof_end[I_j], BW):
                        dofs_state.cdofvel_ang[i_d, i_b] = (
                            dofs_state.cdof_ang[i_d, i_b] * dofs_state.vel[i_d, i_b]
                        )
                        dofs_state.cdofvel_vel[i_d, i_b] = (
                            dofs_state.cdof_vel[i_d, i_b] * dofs_state.vel[i_d, i_b]
                        )


@qd.func
def func_com_links_split(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    is_backward: qd.template(),
):
    """
    Split replacement for `func_COM_links_entity` fused into the cartesian-update
    path.

    Passes 1, 3, 4, 5, 6, 7 run per-link-parallel (`ndrange(n_links, B)`).

    Pass 2 runs with the same 2-level outer loop as the original entity-level
    cartesian-update launch: one thread per (root-entity, batch), which then
    walks every entity sharing that root in ascending entity-index order and
    invokes the per-entity Pass-2 body on it

    Each top-level `for` below is emitted by quadrant as its own offloaded
    sub-kernel with an implicit device-wide barrier between them, so the
    original sequential ordering between passes is preserved.
    """
    BW = qd.static(is_backward)
    serialize = qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    n_links = links_state.pos.shape[0]
    _B = links_state.pos.shape[1]

    # Pass 1: zero-init `root_COM_bw` and `mass_sum` for every link.
    qd.loop_config(serialize=serialize, block_dim=64)
    for i_l, i_b in qd.ndrange(n_links, _B):
        func_com_pass1_zero_link(i_l, i_b, links_state)

    # Pass 2: 2-level outer loop over (root-entity, batch)
    qd.loop_config(serialize=serialize, block_dim=64)
    for i_e, i_b in qd.ndrange(entities_info.n_links.shape[0], _B):
        i_l_start = entities_info.link_start[i_e]
        I_l_start = [i_l_start, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l_start
        if links_info.root_idx[I_l_start] == i_l_start:
            for j_e in (
                range(i_e, entities_info.n_links.shape[0])
                if qd.static(not BW)
                else qd.static(range(static_rigid_sim_config.n_entities))
            ):
                if func_check_index_range(j_e, i_e, static_rigid_sim_config.n_entities, BW):
                    j_l_start = entities_info.link_start[j_e]
                    J_l_start = (
                        [j_l_start, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else j_l_start
                    )
                    if links_info.root_idx[J_l_start] == i_l_start:
                        func_com_pass2_accumulate_entity(
                            j_e, i_b, links_state, links_info, entities_info,
                            static_rigid_sim_config, is_backward,
                        )

    # Pass 3: only the root link normalizes its own `root_COM`.
    qd.loop_config(serialize=serialize, block_dim=64)
    for i_l, i_b in qd.ndrange(n_links, _B):
        func_com_pass3_normalize_root_link(
            i_l, i_b, links_state, links_info, rigid_global_info, static_rigid_sim_config,
        )

    # Pass 4: broadcast root's `root_COM` to every link in its tree.
    qd.loop_config(serialize=serialize, block_dim=64)
    for i_l, i_b in qd.ndrange(n_links, _B):
        func_com_pass4_broadcast_link(
            i_l, i_b, links_state, links_info, static_rigid_sim_config,
        )

    # Pass 5: per-link `i_pos` + `cinr_*` (reads pass-4 `root_COM`).
    qd.loop_config(serialize=serialize, block_dim=64)
    for i_l, i_b in qd.ndrange(n_links, _B):
        func_com_pass5_inertial_link(
            i_l, i_b, links_state, links_info, rigid_global_info, static_rigid_sim_config,
        )

    # Pass 6: per-link joint pose (`j_pos`/`j_quat`). Only reads FK outputs, so
    # this pass has no data dependency on passes 1-5 and could in principle run
    # earlier, but we keep it here to minimize structural churn.
    qd.loop_config(serialize=serialize, block_dim=64)
    for i_l, i_b in qd.ndrange(n_links, _B):
        func_com_pass6_joint_pose_link(
            i_l, i_b, links_state, links_info, joints_info, static_rigid_sim_config, is_backward,
        )

    # Pass 7: per-link motion subspace (`cdof_*`/`cdofvel_*`); reads pass-4
    # `root_COM` at the joint anchor.
    qd.loop_config(serialize=serialize, block_dim=64)
    for i_l, i_b in qd.ndrange(n_links, _B):
        func_com_pass7_cdof_link(
            i_l, i_b, links_state, links_info, joints_state, joints_info, dofs_state,
            rigid_global_info, static_rigid_sim_config, is_backward,
        )


@qd.func
def func_update_cartesian_space_entity(
    i_e,
    i_b,
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
    static_rigid_sim_config: qd.template(),
    force_update_fixed_geoms: qd.template(),
    is_backward: qd.template(),
    include_com: qd.template(),
    include_geoms: qd.template(),
    include_fk: qd.template(),
):
    # The non-hibernation forward launch level-schedules FK (func_fk_levels_split) for
    # higher occupancy and skips it here (include_fk=False); the serial chain walk is
    # kept for the autodiff backward pass and the hibernation path.
    if qd.static(include_fk):
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
    # The non-hibernation cartesian-update launch moves CoM out of this per-entity device
    # function and runs it afterwards as 7 per-link-parallel sub-kernels via
    # ``func_com_links_split`` (much higher occupancy than one thread per entity). Only the
    # hibernation path passes ``include_com=True`` here.
    if qd.static(include_com):
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
    # Geom-pose update is dependency-free (each geom transformed by its link's already
    # computed pose); the non-hibernation forward launch skips it here (include_geoms=
    # False) and runs it afterwards as a per-geom-parallel sub-kernel for much higher
    # occupancy, mirroring the CoM split above.
    if qd.static(include_geoms):
        func_update_geoms_entity(
            i_e,
            i_b,
            entities_info=entities_info,
            geoms_state=geoms_state,
            geoms_info=geoms_info,
            links_state=links_state,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
            force_update_fixed_geoms=force_update_fixed_geoms,
            is_backward=is_backward,
        )


@qd.func
def func_update_cartesian_space_batch(
    i_b,
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
    static_rigid_sim_config: qd.template(),
    force_update_fixed_geoms: qd.template(),
    is_backward: qd.template(),
):
    BW = qd.static(is_backward)
    i_b = qd.cast(i_b, qd.i32)

    # This loop is considered an inner loop
    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_0 in (
        range(rigid_global_info.n_awake_entities[i_b])
        if qd.static(static_rigid_sim_config.use_hibernation)
        else range(entities_info.n_links.shape[0])
    ):
        i_e = rigid_global_info.awake_entities[i_0, i_b] if qd.static(static_rigid_sim_config.use_hibernation) else i_0

        func_update_cartesian_space_entity(
            i_e,
            i_b,
            links_state,
            links_info,
            joints_state,
            joints_info,
            dofs_state,
            dofs_info,
            geoms_state,
            geoms_info,
            entities_info,
            rigid_global_info,
            static_rigid_sim_config,
            force_update_fixed_geoms,
            is_backward,
            True,  # include_com
            True,  # include_geoms
            True,  # include_fk
        )


@qd.func
def func_update_cartesian_space(
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
    static_rigid_sim_config: qd.template(),
    force_update_fixed_geoms: qd.template(),
    is_backward: qd.template(),
):
    BW = qd.static(is_backward)

    # This loop must be the outermost loop to be differentiable
    if qd.static(static_rigid_sim_config.use_hibernation):
        qd.loop_config(
            name="update_carteisan_space_batch", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL
        )
        for i_b in range(links_state.pos.shape[1]):
            func_update_cartesian_space_batch(
                i_b,
                links_state,
                links_info,
                joints_state,
                joints_info,
                dofs_state,
                dofs_info,
                geoms_state,
                geoms_info,
                entities_info,
                rigid_global_info,
                static_rigid_sim_config,
                force_update_fixed_geoms,
                is_backward,
            )
    else:
        # Forward pass: level-schedule FK (one per-(link,env)-parallel pass per tree
        # depth) instead of the one-thread-per-env serial chain walk. The backward pass
        # keeps the serial walk (via include_fk below). CoM is split out further down.
        if qd.static(not BW):
            func_fk_levels_split(
                links_state,
                links_info,
                joints_state,
                joints_info,
                dofs_state,
                dofs_info,
                static_rigid_sim_config,
                rigid_global_info,
                is_backward,
            )
        # Per-entity pass for geom poses (and the serial FK on the backward pass).
        # OPT-1: when a contiguous dynamic-entity window is set (hibernation off), restrict the outer
        # ndrange to [dynamic_entity_offset_, dynamic_entity_offset_ + n_dynamic_entities_), skipping
        # static 0-DOF entities whose FK outputs are constant world-frame values.
        _HAS_WINDOW = qd.static(
            static_rigid_sim_config.n_dynamic_entities_ > 0 and not static_rigid_sim_config.use_hibernation
        )
        _ENTITY_BASE = qd.static(static_rigid_sim_config.dynamic_entity_offset_ if _HAS_WINDOW else 0)
        _N_GRID_ENTITIES = qd.static(static_rigid_sim_config.n_dynamic_entities_) if _HAS_WINDOW else entities_info.n_links.shape[0]
        _N_TOTAL_ENTITIES = entities_info.n_links.shape[0]
        qd.loop_config(
            name="update_cartesian_space",
            serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL),
            block_dim=64,
        )
        for i_e_local, i_b in qd.ndrange(_N_GRID_ENTITIES, links_state.pos.shape[1]):
            i_e = i_e_local + _ENTITY_BASE
            i_l_start = entities_info.link_start[i_e]
            I_l_start = [i_l_start, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l_start
            if links_info.root_idx[I_l_start] == i_l_start:
                for j_e in range(i_e, _N_TOTAL_ENTITIES):
                    j_l_start = entities_info.link_start[j_e]
                    J_l_start = [j_l_start, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else j_l_start
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
                            geoms_state,
                            geoms_info,
                            entities_info,
                            rigid_global_info,
                            static_rigid_sim_config,
                            force_update_fixed_geoms,
                            is_backward,
                            False,
                            # include_geoms: inline only on the backward pass (the per-geom
                            # split below covers the forward pass).
                            is_backward,
                            # include_fk: serial chain walk only on the backward pass;
                            # the forward pass is handled by func_fk_levels_split above.
                            is_backward,
                        )

        # CoM split into per-link-parallel sub-kernels (passes 1, 3-7) plus the bit-exact
        # per-entity pass-2 reduction. Replaces the serial one-thread-per-entity CoM that ran
        # at ~183ms (vg=4) on gfx942; the split is the fork's AMD-optimized G68 path.
        func_com_links_split(
            links_state=links_state,
            links_info=links_info,
            joints_state=joints_state,
            joints_info=joints_info,
            dofs_state=dofs_state,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
            is_backward=is_backward,
        )

        # Geom-pose update as a per-geom-parallel sub-kernel (forward pass). The poses
        # only depend on the already-computed link transforms, so this launches
        # n_geoms * n_envs threads instead of the one-thread-per-entity inline update,
        # which was severely under-occupied at large batch sizes.
        if qd.static(not BW):
            qd.loop_config(
                name="update_geoms_split",
                serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL),
                block_dim=64,
            )
            for i_g, i_b in qd.ndrange(geoms_state.pos.shape[0], links_state.pos.shape[1]):
                if force_update_fixed_geoms or not geoms_info.is_fixed[i_g]:
                    geoms_state.pos[i_g, i_b], geoms_state.quat[i_g, i_b] = gu.qd_transform_pos_quat_by_trans_quat(
                        geoms_info.pos[i_g],
                        geoms_info.quat[i_g],
                        links_state.pos[geoms_info.link_idx[i_g], i_b],
                        links_state.quat[geoms_info.link_idx[i_g], i_b],
                    )
                    geoms_state.verts_updated[i_g, i_b] = False


@qd.kernel(fastcache=True)
def kernel_update_cartesian_space(
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
    static_rigid_sim_config: qd.template(),
    force_update_fixed_geoms: qd.template(),
    is_backward: qd.template(),
):
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
        force_update_fixed_geoms=force_update_fixed_geoms,
        is_backward=is_backward,
    )
