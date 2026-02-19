"""
Inverse kinematics for rigid body entities.

This module contains the inverse kinematics kernel for computing joint configurations
that achieve desired end-effector poses.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.linalg as lu
import genesis.utils.array_class as array_class


# FIXME: RigidEntity is not compatible with fast cache
@qd.kernel(fastcache=False)
def kernel_rigid_entity_inverse_kinematics(
    rigid_entity: qd.template(),
    links_idx: qd.types.ndarray(),
    poss: qd.types.ndarray(),
    quats: qd.types.ndarray(),
    local_points: qd.types.ndarray(),
    n_links: qd.i32,
    dofs_idx: qd.types.ndarray(),
    n_dofs: qd.i32,
    links_idx_by_dofs: qd.types.ndarray(),
    n_links_by_dofs: qd.i32,
    custom_init_qpos: qd.i32,
    init_qpos: qd.types.ndarray(),
    max_samples: qd.i32,
    max_solver_iters: qd.i32,
    damping: qd.f32,
    pos_tol: qd.f32,
    rot_tol: qd.f32,
    pos_mask_: qd.types.ndarray(),
    rot_mask_: qd.types.ndarray(),
    link_pos_mask: qd.types.ndarray(),
    link_rot_mask: qd.types.ndarray(),
    max_step_size: qd.f32,
    respect_joint_limit: qd.i32,
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
    EPS = rigid_global_info.EPS[None]

    # convert to qd Vector
    pos_mask = qd.Vector([pos_mask_[0], pos_mask_[1], pos_mask_[2]], dt=gs.qd_float)
    rot_mask = qd.Vector([rot_mask_[0], rot_mask_[1], rot_mask_[2]], dt=gs.qd_float)
    n_error_dims = 6 * n_links

    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]

        # save original qpos
        for i_q in range(rigid_entity.n_qs):
            rigid_entity._IK_qpos_orig[i_q, i_b] = rigid_global_info.qpos[i_q + rigid_entity._q_start, i_b]

        if custom_init_qpos:
            for i_q in range(rigid_entity.n_qs):
                rigid_global_info.qpos[i_q + rigid_entity._q_start, i_b] = init_qpos[i_b_, i_q]

        for i_error in range(n_error_dims):
            rigid_entity._IK_err_pose_best[i_error, i_b] = 1e4

        solved = False
        for i_sample in range(max_samples):
            for _ in range(max_solver_iters):
                # run FK to update link states using current q
                gs.engine.solvers.rigid.rigid_solver.func_forward_kinematics_entity(
                    rigid_entity._idx_in_solver,
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
                # compute error
                solved = True
                for i_ee in range(n_links):
                    i_l_ee = links_idx[i_ee]

                    tgt_pos_i = qd.Vector([poss[i_ee, i_b_, 0], poss[i_ee, i_b_, 1], poss[i_ee, i_b_, 2]])
                    local_point_i = qd.Vector([local_points[i_ee, 0], local_points[i_ee, 1], local_points[i_ee, 2]])
                    pos_curr_i = links_state.pos[i_l_ee, i_b] + gu.qd_transform_by_quat(
                        local_point_i, links_state.quat[i_l_ee, i_b]
                    )
                    err_pos_i = tgt_pos_i - pos_curr_i
                    for k in range(3):
                        err_pos_i[k] *= pos_mask[k] * link_pos_mask[i_ee]
                    if err_pos_i.norm() > pos_tol:
                        solved = False

                    tgt_quat_i = qd.Vector(
                        [quats[i_ee, i_b_, 0], quats[i_ee, i_b_, 1], quats[i_ee, i_b_, 2], quats[i_ee, i_b_, 3]]
                    )
                    err_rot_i = gu.qd_quat_to_rotvec(
                        gu.qd_transform_quat_by_quat(gu.qd_inv_quat(links_state.quat[i_l_ee, i_b]), tgt_quat_i), EPS
                    )
                    for k in range(3):
                        err_rot_i[k] *= rot_mask[k] * link_rot_mask[i_ee]
                    if err_rot_i.norm() > rot_tol:
                        solved = False

                    # put into multi-link error array
                    for k in range(3):
                        rigid_entity._IK_err_pose[i_ee * 6 + k, i_b] = err_pos_i[k]
                        rigid_entity._IK_err_pose[i_ee * 6 + k + 3, i_b] = err_rot_i[k]

                if solved:
                    break

                # compute multi-link jacobian
                for i_ee in range(n_links):
                    # update jacobian for ee link
                    i_l_ee = links_idx[i_ee]
                    local_point_i = qd.Vector([local_points[i_ee, 0], local_points[i_ee, 1], local_points[i_ee, 2]])
                    rigid_entity._func_get_jacobian(
                        tgt_link_idx=i_l_ee,
                        i_b=i_b,
                        p_local=local_point_i,
                        pos_mask=pos_mask,
                        rot_mask=rot_mask,
                        dofs_info=dofs_info,
                        joints_info=joints_info,
                        links_info=links_info,
                        links_state=links_state,
                    )  # NOTE: we still compute jacobian for all dofs as we haven't found a clean way to implement this

                    # copy to multi-link jacobian (only for the effective n_dofs instead of self.n_dofs)
                    for i_dof in range(n_dofs):
                        for i_error in qd.static(range(6)):
                            i_row = i_ee * 6 + i_error
                            i_dof_ = dofs_idx[i_dof]
                            rigid_entity._IK_jacobian[i_row, i_dof, i_b] = rigid_entity._jacobian[i_error, i_dof_, i_b]

                # compute dq = jac.T @ inverse(jac @ jac.T + diag) @ error (only for the effective n_dofs instead of self.n_dofs)
                lu.mat_transpose(rigid_entity._IK_jacobian, rigid_entity._IK_jacobian_T, n_error_dims, n_dofs, i_b)
                lu.mat_mul(
                    rigid_entity._IK_jacobian,
                    rigid_entity._IK_jacobian_T,
                    rigid_entity._IK_mat,
                    n_error_dims,
                    n_dofs,
                    n_error_dims,
                    i_b,
                )
                lu.mat_add_eye(rigid_entity._IK_mat, damping**2, n_error_dims, i_b)
                lu.mat_inverse(
                    rigid_entity._IK_mat,
                    rigid_entity._IK_L,
                    rigid_entity._IK_U,
                    rigid_entity._IK_y,
                    rigid_entity._IK_inv,
                    n_error_dims,
                    i_b,
                )
                lu.mat_mul_vec(
                    rigid_entity._IK_inv,
                    rigid_entity._IK_err_pose,
                    rigid_entity._IK_vec,
                    n_error_dims,
                    n_error_dims,
                    i_b,
                )

                for i_d in range(rigid_entity.n_dofs):  # IK_delta_qpos = IK_jacobian_T @ IK_vec
                    rigid_entity._IK_delta_qpos[i_d, i_b] = 0
                for i_d in range(n_dofs):
                    for j in range(n_error_dims):
                        # NOTE: IK_delta_qpos uses the original indexing instead of the effective n_dofs
                        i_d_ = dofs_idx[i_d]
                        rigid_entity._IK_delta_qpos[i_d_, i_b] += (
                            rigid_entity._IK_jacobian_T[i_d, j, i_b] * rigid_entity._IK_vec[j, i_b]
                        )

                for i_d in range(rigid_entity.n_dofs):
                    rigid_entity._IK_delta_qpos[i_d, i_b] = qd.math.clamp(
                        rigid_entity._IK_delta_qpos[i_d, i_b], -max_step_size, max_step_size
                    )

                # update q
                gs.engine.solvers.rigid.rigid_solver.func_integrate_dq_entity(
                    rigid_entity._IK_delta_qpos,
                    rigid_entity._idx_in_solver,
                    i_b,
                    respect_joint_limit,
                    links_info,
                    joints_info,
                    dofs_info,
                    entities_info,
                    rigid_global_info,
                    static_rigid_sim_config,
                )

            if not solved:
                # re-compute final error if exited not due to solved
                gs.engine.solvers.rigid.rigid_solver.func_forward_kinematics_entity(
                    rigid_entity._idx_in_solver,
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
                solved = True
                for i_ee in range(n_links):
                    i_l_ee = links_idx[i_ee]

                    tgt_pos_i = qd.Vector([poss[i_ee, i_b_, 0], poss[i_ee, i_b_, 1], poss[i_ee, i_b_, 2]])
                    local_point_i = qd.Vector([local_points[i_ee, 0], local_points[i_ee, 1], local_points[i_ee, 2]])
                    pos_curr_i = links_state.pos[i_l_ee, i_b] + gu.qd_transform_by_quat(
                        local_point_i, links_state.quat[i_l_ee, i_b]
                    )
                    err_pos_i = tgt_pos_i - pos_curr_i
                    for k in range(3):
                        err_pos_i[k] *= pos_mask[k] * link_pos_mask[i_ee]
                    if err_pos_i.norm() > pos_tol:
                        solved = False

                    tgt_quat_i = qd.Vector(
                        [quats[i_ee, i_b_, 0], quats[i_ee, i_b_, 1], quats[i_ee, i_b_, 2], quats[i_ee, i_b_, 3]]
                    )
                    err_rot_i = gu.qd_quat_to_rotvec(
                        gu.qd_transform_quat_by_quat(gu.qd_inv_quat(links_state.quat[i_l_ee, i_b]), tgt_quat_i), EPS
                    )
                    for k in range(3):
                        err_rot_i[k] *= rot_mask[k] * link_rot_mask[i_ee]
                    if err_rot_i.norm() > rot_tol:
                        solved = False

                    # put into multi-link error array
                    for k in range(3):
                        rigid_entity._IK_err_pose[i_ee * 6 + k, i_b] = err_pos_i[k]
                        rigid_entity._IK_err_pose[i_ee * 6 + k + 3, i_b] = err_rot_i[k]

            if solved:
                for i_q in range(rigid_entity.n_qs):
                    rigid_entity._IK_qpos_best[i_q, i_b] = rigid_global_info.qpos[i_q + rigid_entity._q_start, i_b]
                for i_error in range(n_error_dims):
                    rigid_entity._IK_err_pose_best[i_error, i_b] = rigid_entity._IK_err_pose[i_error, i_b]
                break

            else:
                # copy to _IK_qpos if this sample is better
                improved = True
                for i_ee in range(n_links):
                    error_pos_i = qd.Vector(
                        [rigid_entity._IK_err_pose[i_ee * 6 + i_error, i_b] for i_error in range(3)]
                    )
                    error_rot_i = qd.Vector(
                        [rigid_entity._IK_err_pose[i_ee * 6 + i_error, i_b] for i_error in range(3, 6)]
                    )
                    error_pos_best = qd.Vector(
                        [rigid_entity._IK_err_pose_best[i_ee * 6 + i_error, i_b] for i_error in range(3)]
                    )
                    error_rot_best = qd.Vector(
                        [rigid_entity._IK_err_pose_best[i_ee * 6 + i_error, i_b] for i_error in range(3, 6)]
                    )
                    if error_pos_i.norm() > error_pos_best.norm() or error_rot_i.norm() > error_rot_best.norm():
                        improved = False
                        break

                if improved:
                    for i_q in range(rigid_entity.n_qs):
                        rigid_entity._IK_qpos_best[i_q, i_b] = rigid_global_info.qpos[i_q + rigid_entity._q_start, i_b]
                    for i_error in range(n_error_dims):
                        rigid_entity._IK_err_pose_best[i_error, i_b] = rigid_entity._IK_err_pose[i_error, i_b]

                # Resample init q
                if respect_joint_limit and i_sample < max_samples - 1:
                    for i_l_ in range(n_links_by_dofs):
                        i_l = links_idx_by_dofs[i_l_]
                        I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l

                        for i_j in range(links_info.joint_start[I_l], links_info.joint_end[I_l]):
                            I_j = [i_j, i_b] if qd.static(static_rigid_sim_config.batch_joints_info) else i_j
                            i_d = joints_info.dof_start[I_j]
                            I_d = [i_d, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else i_d

                            dof_limit = dofs_info.limit[I_d]
                            if (
                                joints_info.type[I_j] == gs.JOINT_TYPE.REVOLUTE
                                or joints_info.type[I_j] == gs.JOINT_TYPE.PRISMATIC
                            ) and not (qd.math.isinf(dof_limit[0]) or qd.math.isinf(dof_limit[1])):
                                q_start = joints_info.q_start[I_j]
                                rigid_global_info.qpos[q_start, i_b] = dof_limit[0] + qd.random() * (
                                    dof_limit[1] - dof_limit[0]
                                )
                else:
                    pass  # When respect_joint_limit=False, we can simply continue from the last solution

        # restore original qpos and link state
        for i_q in range(rigid_entity.n_qs):
            rigid_global_info.qpos[i_q + rigid_entity._q_start, i_b] = rigid_entity._IK_qpos_orig[i_q, i_b]
        gs.engine.solvers.rigid.rigid_solver.func_forward_kinematics_entity(
            rigid_entity._idx_in_solver,
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
