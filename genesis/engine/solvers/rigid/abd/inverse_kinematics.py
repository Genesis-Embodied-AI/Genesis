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


@qd.func
def func_ik_fk(
    i_col_out,
    i_col_q,
    i_b,
    entity_idx,
    link_offset,
    joint_offset,
    q_offset,
    qpos: qd.Tensor,
    links_pos: qd.Tensor,
    links_quat: qd.Tensor,
    joints_xanchor: qd.Tensor,
    joints_xaxis: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Forward kinematics of one entity on caller-owned scratch buffers, for one configuration column.

    Reads the configuration from qpos[:, i_col_q] (entity-local q coordinates keyed by q_offset) and writes link
    poses and joint frames at column i_col_out of the given tensors; the live solver state is only read (model
    info and the fixed root pose). REVOLUTE / PRISMATIC / FIXED joints only - callers rejecting FREE / SPHERICAL
    joints (like the planner) can use this scratch forward kinematics without mutating solver state.
    """
    for i_l_ in range(dyn_info.entities.link_start[entity_idx], dyn_info.entities.link_end[entity_idx]):
        i_l = gs.qd_int(i_l_)
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l

        pos = dyn_info.links.pos[I_l]
        quat = dyn_info.links.quat[I_l]
        if dyn_info.links.parent_idx[I_l] != -1:
            parent_pos = links_pos[dyn_info.links.parent_idx[I_l] - link_offset, i_col_out]
            parent_quat = links_quat[dyn_info.links.parent_idx[I_l] - link_offset, i_col_out]
            pos = parent_pos + gu.qd_transform_by_quat(dyn_info.links.pos[I_l], parent_quat)
            quat = gu.qd_transform_quat_by_quat(dyn_info.links.quat[I_l], parent_quat)

        for i_j_ in range(dyn_info.links.joint_start[I_l], dyn_info.links.joint_end[I_l]):
            i_j = gs.qd_int(i_j_)
            I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
            joint_type = dyn_info.joints.type[I_j]
            q_start = dyn_info.joints.q_start[I_j]
            dof_start = dyn_info.joints.dof_start[I_j]
            I_d = [dof_start, i_b] if qd.static(rigid_config.batch_dofs_info) else dof_start

            if joint_type != gs.JOINT_TYPE.FIXED:
                axis = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)
                if joint_type == gs.JOINT_TYPE.REVOLUTE:
                    axis = dyn_info.dofs.motion_ang[I_d]
                else:
                    axis = dyn_info.dofs.motion_vel[I_d]
                joints_xanchor[i_j - joint_offset, i_col_out] = (
                    gu.qd_transform_by_quat(dyn_info.joints.pos[I_j], quat) + pos
                )
                joints_xaxis[i_j - joint_offset, i_col_out] = gu.qd_transform_by_quat(axis, quat)

                q_delta = qpos[q_start - q_offset, i_col_q] - rigid_info.qpos0[q_start, i_b]
                if joint_type == gs.JOINT_TYPE.REVOLUTE:
                    qloc = gu.qd_rotvec_to_quat(axis * q_delta, rigid_info.EPS[None])
                    quat = gu.qd_transform_quat_by_quat(qloc, quat)
                    pos = joints_xanchor[i_j - joint_offset, i_col_out] - gu.qd_transform_by_quat(
                        dyn_info.joints.pos[I_j], quat
                    )
                else:
                    pos = pos + joints_xaxis[i_j - joint_offset, i_col_out] * q_delta

        # Fixed root links keep their live pose, exactly like the solver FK (users may overwrite them manually).
        if dyn_info.links.parent_idx[I_l] == -1 and dyn_info.links.is_fixed[I_l]:
            pos = dyn_state.links.pos[i_l, i_b]
            quat = dyn_state.links.quat[i_l, i_b]
        links_pos[i_l - link_offset, i_col_out] = pos
        links_quat[i_l - link_offset, i_col_out] = quat


@qd.func
def func_ik_jacobian(
    i_col,
    i_b,
    entity_idx,
    joint_offset,
    ee_link,
    ee_pos,
    jacobian: qd.Tensor,
    joints_xanchor: qd.Tensor,
    joints_xaxis: qd.Tensor,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
    n_dp: qd.template(),
):
    """Spatial Jacobian (6 x n_dp) of the goal point at column i_col, from the per-column scratch FK joint frames.

    Walks the kinematic path from the target link to the entity root, filling each REVOLUTE / PRISMATIC joint's
    column from its world axis and anchor (already placed by func_ik_fk into joints_xaxis / joints_xanchor).
    Unmasked; callers rejecting FREE / SPHERICAL joints leave only these two contributing.
    """
    dof_offset = dyn_info.entities.dof_start[entity_idx]
    for i_row, i_d in qd.ndrange(6, qd.static(n_dp)):
        jacobian[i_row, i_d, i_col] = 0.0

    i_l = ee_link
    while i_l != -1:
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        for i_j_ in range(dyn_info.links.joint_start[I_l], dyn_info.links.joint_end[I_l]):
            i_j = gs.qd_int(i_j_)
            I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
            i_d_jac = dyn_info.joints.dof_start[I_j] - dof_offset
            if dyn_info.joints.type[I_j] == gs.JOINT_TYPE.REVOLUTE:
                rotation = joints_xaxis[i_j - joint_offset, i_col]
                translation = rotation.cross(ee_pos - joints_xanchor[i_j - joint_offset, i_col])
                for i in qd.static(range(3)):
                    jacobian[i, i_d_jac, i_col] = translation[i]
                    jacobian[i + 3, i_d_jac, i_col] = rotation[i]
            elif dyn_info.joints.type[I_j] == gs.JOINT_TYPE.PRISMATIC:
                translation = joints_xaxis[i_j - joint_offset, i_col]
                for i in qd.static(range(3)):
                    jacobian[i, i_d_jac, i_col] = translation[i]
        i_l = dyn_info.links.parent_idx[I_l]


@qd.func
def func_get_jacobian(
    tgt_link_idx,
    i_b,
    dof_start,
    p_local,
    jacobian: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    """Full spatial Jacobian (6 x n_dofs) of a target link's local point, written into column i_b of the buffer.

    Sums the velocity contribution of each joint on the path to the root; dof_start offsets global joint DOFs into
    the buffer rows. Axis masking is left to the caller (the inverse-kinematics solver masks the stacked copy).
    """
    n_dofs = jacobian.shape[1]
    for i_row, i_d in qd.ndrange(6, n_dofs):
        jacobian[i_row, i_d, i_b] = 0.0

    tgt_link_pos = dyn_state.links.pos[tgt_link_idx, i_b] + gu.qd_transform_by_quat(
        p_local, dyn_state.links.quat[tgt_link_idx, i_b]
    )
    i_l = tgt_link_idx
    while i_l > -1:
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l

        for i_j in range(dyn_info.links.joint_start[I_l], dyn_info.links.joint_end[I_l]):
            I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j

            if dyn_info.joints.type[I_j] == gs.JOINT_TYPE.FIXED:
                pass

            elif dyn_info.joints.type[I_j] == gs.JOINT_TYPE.REVOLUTE:
                i_d_jac = dyn_info.joints.dof_start[I_j] - dof_start
                rotation = dyn_state.joints.xaxis[i_j, i_b]
                translation = rotation.cross(tgt_link_pos - dyn_state.joints.xanchor[i_j, i_b])

                for i in qd.static(range(3)):
                    jacobian[i, i_d_jac, i_b] = translation[i]
                    jacobian[i + 3, i_d_jac, i_b] = rotation[i]

            elif dyn_info.joints.type[I_j] == gs.JOINT_TYPE.PRISMATIC:
                i_d_jac = dyn_info.joints.dof_start[I_j] - dof_start
                translation = dyn_state.joints.xaxis[i_j, i_b]

                for i in qd.static(range(3)):
                    jacobian[i, i_d_jac, i_b] = translation[i]

            elif dyn_info.joints.type[I_j] == gs.JOINT_TYPE.FREE:
                for i_d_ in qd.static(range(3)):
                    i_d_jac = dyn_info.joints.dof_start[I_j] + i_d_ - dof_start
                    jacobian[i_d_, i_d_jac, i_b] = 1.0

                for i_d_ in qd.static(range(3)):
                    i_d = dyn_info.joints.dof_start[I_j] + i_d_ + 3
                    i_d_jac = i_d - dof_start
                    I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d
                    rotation = dyn_info.dofs.motion_ang[I_d]
                    translation = rotation.cross(tgt_link_pos - dyn_state.links.pos[i_l, i_b])

                    for i in qd.static(range(3)):
                        jacobian[i, i_d_jac, i_b] = translation[i]
                        jacobian[i + 3, i_d_jac, i_b] = rotation[i]

        i_l = dyn_info.links.parent_idx[I_l]


@qd.kernel
def kernel_get_jacobian(
    tgt_link_idx: int,
    dof_start: int,
    p_local: qd.types.ndarray(),
    jacobian: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
    n_batch: int,
):
    """Full spatial Jacobian of a link-local point, for every environment column."""
    p_vec = qd.Vector([p_local[0], p_local[1], p_local[2]], dt=gs.qd_float)
    for i_b in range(n_batch):
        func_get_jacobian(tgt_link_idx, i_b, dof_start, p_vec, jacobian, dyn_state, dyn_info, rigid_config)


@qd.kernel
def kernel_get_jacobian_zero(
    tgt_link_idx: int,
    dof_start: int,
    jacobian: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
    n_batch: int,
):
    """Full spatial Jacobian of a link origin, for every environment column."""
    for i_b in range(n_batch):
        func_get_jacobian(
            tgt_link_idx, i_b, dof_start, qd.Vector.zero(gs.qd_float, 3), jacobian, dyn_state, dyn_info, rigid_config
        )


@qd.func
def func_inverse_kinematics(
    idx_in_solver,
    entity_q_start,
    dof_start,
    targets: array_class.IKTargets,
    ik_state: array_class.IKState,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    n_qs,
    entity_n_dofs,
    custom_init_qpos,
    max_samples,
    max_solver_iters,
    damping,
    pos_tol,
    rot_tol,
    max_step_size,
    seed,
    respect_joint_limit,
):
    """Damped-least-squares inverse kinematics for the target links, writing the best configuration per column
    into ik_state.qpos_best.

    Runs up to max_samples restarts of at most max_solver_iters Gauss-Newton steps against the stacked
    multi-target pose error, keeping the least-error configuration. The solver qpos serves as working state and is
    restored on return; unconverged restarts resample joint-limited DOFs with a counter-based draw so the result is
    deterministic across parallel schedules.
    """
    EPS = rigid_info.EPS[None]

    n_dofs = targets.dofs_idx.shape[0]
    n_links = targets.links_idx.shape[0]
    n_error_dims = 6 * n_links

    for i_b_ in range(targets.envs_idx.shape[0]):
        i_b = targets.envs_idx[i_b_]

        # save original qpos
        for i_q in range(n_qs):
            ik_state.qpos_orig[i_q, i_b] = rigid_info.qpos[i_q + entity_q_start, i_b]

        if custom_init_qpos:
            for i_q in range(n_qs):
                rigid_info.qpos[i_q + entity_q_start, i_b] = targets.init_qpos[i_b_, i_q]

        for i_error in range(n_error_dims):
            ik_state.err_pose_best[i_error, i_b] = 1e4

        solved = False
        for i_sample in range(max_samples):
            for _ in range(max_solver_iters):
                # run FK to update link states using current q
                gs.engine.solvers.rigid.rigid_solver.func_forward_kinematics_entity(
                    idx_in_solver, i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward=False
                )
                # compute error
                solved = True
                for i_ee in range(n_links):
                    i_l_ee = targets.links_idx[i_ee]

                    tgt_pos_i = targets.pos[i_ee, i_b_]
                    local_point_i = targets.local_point[i_ee]
                    pos_curr_i = dyn_state.links.pos[i_l_ee, i_b] + gu.qd_transform_by_quat(
                        local_point_i, dyn_state.links.quat[i_l_ee, i_b]
                    )
                    err_pos_i = tgt_pos_i - pos_curr_i
                    for k in range(3):
                        err_pos_i[k] *= targets.pos_mask[k] * targets.link_pos_mask[i_ee]
                    if err_pos_i.norm() > pos_tol:
                        solved = False

                    tgt_quat_i = targets.quat[i_ee, i_b_]
                    err_rot_i = gu.qd_quat_to_rotvec(
                        gu.qd_transform_quat_by_quat(gu.qd_inv_quat(dyn_state.links.quat[i_l_ee, i_b]), tgt_quat_i), EPS
                    )
                    for k in range(3):
                        err_rot_i[k] *= targets.rot_mask[k] * targets.link_rot_mask[i_ee]
                    if err_rot_i.norm() > rot_tol:
                        solved = False

                    # put into multi-link error array
                    for k in range(3):
                        ik_state.err_pose[i_ee * 6 + k, i_b] = err_pos_i[k]
                        ik_state.err_pose[i_ee * 6 + k + 3, i_b] = err_rot_i[k]

                if solved:
                    break

                # compute multi-link jacobian
                for i_ee in range(n_links):
                    # update jacobian for ee link
                    i_l_ee = targets.links_idx[i_ee]
                    local_point_i = targets.local_point[i_ee]
                    # NOTE: the jacobian covers all entity dofs, as we haven't found a clean way to restrict it.
                    func_get_jacobian(
                        i_l_ee,
                        i_b,
                        dof_start,
                        local_point_i,
                        ik_state.jacobian,
                        dyn_state,
                        dyn_info,
                        rigid_config,
                    )

                    # Copy the full single-link Jacobian into the stacked multi-link block for the effective DOFs,
                    # masking the position rows by pos_mask and the rotation rows by rot_mask (the maskless
                    # func_get_jacobian leaves axis selection to here).
                    for i_d_ in range(n_dofs):
                        i_d = targets.dofs_idx[i_d_]
                        for i_error in qd.static(range(3)):
                            ik_state.jacobian_stacked[i_ee * 6 + i_error, i_d_, i_b] = (
                                ik_state.jacobian[i_error, i_d, i_b] * targets.pos_mask[i_error]
                            )
                            ik_state.jacobian_stacked[i_ee * 6 + i_error + 3, i_d_, i_b] = (
                                ik_state.jacobian[i_error + 3, i_d, i_b] * targets.rot_mask[i_error]
                            )

                # compute dq = jac.T @ inverse(jac @ jac.T + diag) @ error (only for the effective n_dofs instead of self.n_dofs)
                lu.mat_transpose(ik_state.jacobian_stacked, ik_state.jacobian_stacked_t, n_error_dims, n_dofs, i_b)
                lu.mat_mul(
                    ik_state.jacobian_stacked,
                    ik_state.jacobian_stacked_t,
                    ik_state.mat,
                    n_error_dims,
                    n_dofs,
                    n_error_dims,
                    i_b,
                )
                lu.mat_add_eye(ik_state.mat, damping**2, n_error_dims, i_b)
                lu.mat_inverse(
                    ik_state.mat,
                    ik_state.lu_lower,
                    ik_state.lu_upper,
                    ik_state.lu_y,
                    ik_state.inv,
                    n_error_dims,
                    i_b,
                )
                lu.mat_mul_vec(
                    ik_state.inv,
                    ik_state.err_pose,
                    ik_state.vec,
                    n_error_dims,
                    n_error_dims,
                    i_b,
                )

                for i_d_ in range(entity_n_dofs):  # IK_delta_qpos = IK_jacobian_T @ IK_vec
                    ik_state.delta_qpos[i_d_, i_b] = 0
                for i_d_ in range(n_dofs):
                    i_d = targets.dofs_idx[i_d_]
                    for j in range(n_error_dims):
                        # NOTE: IK_delta_qpos uses the original indexing instead of the effective n_dofs
                        ik_state.delta_qpos[i_d, i_b] += (
                            ik_state.jacobian_stacked_t[i_d_, j, i_b] * ik_state.vec[j, i_b]
                        )

                for i_d_ in range(entity_n_dofs):
                    ik_state.delta_qpos[i_d_, i_b] = qd.math.clamp(
                        ik_state.delta_qpos[i_d_, i_b], -max_step_size, max_step_size
                    )

                # update q
                gs.engine.solvers.rigid.rigid_solver.func_integrate_dq_entity(
                    idx_in_solver,
                    i_b,
                    ik_state.delta_qpos,
                    dyn_info,
                    rigid_info,
                    rigid_config,
                    respect_joint_limit,
                )

            if not solved:
                # re-compute final error if exited not due to solved
                gs.engine.solvers.rigid.rigid_solver.func_forward_kinematics_entity(
                    idx_in_solver, i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward=False
                )
                solved = True
                for i_ee in range(n_links):
                    i_l_ee = targets.links_idx[i_ee]

                    tgt_pos_i = targets.pos[i_ee, i_b_]
                    local_point_i = targets.local_point[i_ee]
                    pos_curr_i = dyn_state.links.pos[i_l_ee, i_b] + gu.qd_transform_by_quat(
                        local_point_i, dyn_state.links.quat[i_l_ee, i_b]
                    )
                    err_pos_i = tgt_pos_i - pos_curr_i
                    for k in range(3):
                        err_pos_i[k] *= targets.pos_mask[k] * targets.link_pos_mask[i_ee]
                    if err_pos_i.norm() > pos_tol:
                        solved = False

                    tgt_quat_i = targets.quat[i_ee, i_b_]
                    err_rot_i = gu.qd_quat_to_rotvec(
                        gu.qd_transform_quat_by_quat(gu.qd_inv_quat(dyn_state.links.quat[i_l_ee, i_b]), tgt_quat_i), EPS
                    )
                    for k in range(3):
                        err_rot_i[k] *= targets.rot_mask[k] * targets.link_rot_mask[i_ee]
                    if err_rot_i.norm() > rot_tol:
                        solved = False

                    # put into multi-link error array
                    for k in range(3):
                        ik_state.err_pose[i_ee * 6 + k, i_b] = err_pos_i[k]
                        ik_state.err_pose[i_ee * 6 + k + 3, i_b] = err_rot_i[k]

            if solved:
                for i_q in range(n_qs):
                    ik_state.qpos_best[i_q, i_b] = rigid_info.qpos[i_q + entity_q_start, i_b]
                for i_error in range(n_error_dims):
                    ik_state.err_pose_best[i_error, i_b] = ik_state.err_pose[i_error, i_b]
                break

            else:
                # copy to _IK_qpos if this sample is better
                improved = True
                for i_ee in range(n_links):
                    error_pos_i = qd.Vector([ik_state.err_pose[i_ee * 6 + i_error, i_b] for i_error in range(3)])
                    error_rot_i = qd.Vector([ik_state.err_pose[i_ee * 6 + i_error, i_b] for i_error in range(3, 6)])
                    error_pos_best = qd.Vector(
                        [ik_state.err_pose_best[i_ee * 6 + i_error, i_b] for i_error in range(3)]
                    )
                    error_rot_best = qd.Vector(
                        [ik_state.err_pose_best[i_ee * 6 + i_error, i_b] for i_error in range(3, 6)]
                    )
                    if error_pos_i.norm() > error_pos_best.norm() or error_rot_i.norm() > error_rot_best.norm():
                        improved = False
                        break

                if improved:
                    for i_q in range(n_qs):
                        ik_state.qpos_best[i_q, i_b] = rigid_info.qpos[i_q + entity_q_start, i_b]
                    for i_error in range(n_error_dims):
                        ik_state.err_pose_best[i_error, i_b] = ik_state.err_pose[i_error, i_b]

                # Resample init q
                if respect_joint_limit and i_sample < max_samples - 1:
                    i_e = idx_in_solver
                    entity_dof_start = dyn_info.entities.dof_start[i_e]
                    for i_l in range(dyn_info.entities.link_start[i_e], dyn_info.entities.link_end[i_e]):
                        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l

                        must_resample = False
                        for i_d_ in range(n_dofs):
                            i_d = targets.dofs_idx[i_d_]
                            link_dof_start_local = dyn_info.links.dof_start[I_l] - entity_dof_start
                            link_dof_end_local = dyn_info.links.dof_end[I_l] - entity_dof_start
                            if link_dof_start_local <= i_d and i_d < link_dof_end_local:
                                must_resample = True
                                break
                        if not must_resample:
                            continue

                        for i_j in range(dyn_info.links.joint_start[I_l], dyn_info.links.joint_end[I_l]):
                            I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
                            i_d = dyn_info.joints.dof_start[I_j]
                            I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d

                            dof_limit = dyn_info.dofs.limit[I_d]
                            if (
                                dyn_info.joints.type[I_j] == gs.JOINT_TYPE.REVOLUTE
                                or dyn_info.joints.type[I_j] == gs.JOINT_TYPE.PRISMATIC
                            ) and not (qd.math.isinf(dof_limit[0]) or qd.math.isinf(dof_limit[1])):
                                q_start = dyn_info.joints.q_start[I_j]
                                # Counter-based draw: qd.random() per-thread stream ordering depends on the
                                # parallel schedule, which would make the resampled solutions - and therefore
                                # the returned configuration - nondeterministic across runs.
                                rigid_info.qpos[q_start, i_b] = dof_limit[0] + gu.qd_hash01(
                                    seed, i_b, i_sample, q_start
                                ) * (dof_limit[1] - dof_limit[0])
                else:
                    pass  # When respect_joint_limit=False, we can simply continue from the last solution

        # restore original qpos and link state
        for i_q in range(n_qs):
            rigid_info.qpos[i_q + entity_q_start, i_b] = ik_state.qpos_orig[i_q, i_b]
        gs.engine.solvers.rigid.rigid_solver.func_forward_kinematics_entity(
            idx_in_solver, i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward=False
        )


@qd.kernel(fastcache=True)
def kernel_rigid_entity_inverse_kinematics(
    idx_in_solver: int,
    entity_q_start: int,
    dof_start: int,
    targets: array_class.IKTargets,
    ik_state: array_class.IKState,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    n_qs: int,
    entity_n_dofs: int,
    custom_init_qpos: int,
    max_samples: int,
    max_solver_iters: int,
    damping: float,
    pos_tol: float,
    rot_tol: float,
    max_step_size: float,
    seed: int,
    respect_joint_limit: int,
):
    """Entity-facing launch wrapper forwarding the IK scratch and model scalars to func_inverse_kinematics."""
    func_inverse_kinematics(
        idx_in_solver,
        entity_q_start,
        dof_start,
        targets,
        ik_state,
        dyn_state,
        dyn_info,
        rigid_info,
        rigid_config,
        n_qs,
        entity_n_dofs,
        custom_init_qpos,
        max_samples,
        max_solver_iters,
        damping,
        pos_tol,
        rot_tol,
        max_step_size,
        seed,
        respect_joint_limit,
    )


@qd.kernel(fastcache=True)
def kernel_set_ik_targets(
    links_idx: qd.types.ndarray(),
    dofs_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    poss: qd.types.ndarray(),
    quats: qd.types.ndarray(),
    local_points: qd.types.ndarray(),
    init_qpos: qd.types.ndarray(),
    pos_mask: qd.types.ndarray(),
    rot_mask: qd.types.ndarray(),
    link_pos_mask: qd.types.ndarray(),
    link_rot_mask: qd.types.ndarray(),
    targets: array_class.IKTargets,
):
    """Copy externally supplied inverse-kinematics targets into the solver-side struct (non-zero-copy fallback)."""
    for i_ee in range(links_idx.shape[0]):
        targets.links_idx[i_ee] = links_idx[i_ee]
        targets.link_pos_mask[i_ee] = link_pos_mask[i_ee]
        targets.link_rot_mask[i_ee] = link_rot_mask[i_ee]
        targets.local_point[i_ee] = qd.Vector([local_points[i_ee, 0], local_points[i_ee, 1], local_points[i_ee, 2]])
        for i_c in range(envs_idx.shape[0]):
            targets.pos[i_ee, i_c] = qd.Vector([poss[i_ee, i_c, 0], poss[i_ee, i_c, 1], poss[i_ee, i_c, 2]])
            targets.quat[i_ee, i_c] = qd.Vector(
                [quats[i_ee, i_c, 0], quats[i_ee, i_c, 1], quats[i_ee, i_c, 2], quats[i_ee, i_c, 3]]
            )
    for i_d in range(dofs_idx.shape[0]):
        targets.dofs_idx[i_d] = dofs_idx[i_d]
    for i_c in range(envs_idx.shape[0]):
        targets.envs_idx[i_c] = envs_idx[i_c]
        for i_q in range(init_qpos.shape[1]):
            targets.init_qpos[i_c, i_q] = init_qpos[i_c, i_q]
    for k in qd.static(range(3)):
        targets.pos_mask[k] = pos_mask[k]
        targets.rot_mask[k] = rot_mask[k]
