import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils import array_class


@qd.func
def func_planner_fk(
    i_col_out,
    i_col_q,
    i_b,
    qpos: qd.Tensor,
    links_pos: qd.Tensor,
    links_quat: qd.Tensor,
    joints_xanchor: qd.Tensor,
    joints_xaxis: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    planner_config: qd.template(),
):
    """
    Forward kinematics (FK) of the planned entity on planner-owned buffers, for one configuration column.

    Reads the configuration from ``qpos[:, i_col_q]`` (entity-local q coordinates) and writes link poses and joint
    frames at column ``i_col_out`` of the given planner tensors; the live solver state is only read (model info and
    the fixed root pose). Strict subset of the solver FK (see func_forward_kinematics_entity in
    abd/forward_kinematics.py): REVOLUTE / PRISMATIC / FIXED joints only, planning rejects FREE / SPHERICAL up
    front.
    """
    i_e = qd.static(planner_config.entity_idx)
    link_offset = qd.static(planner_config.link_offset)
    joint_offset = qd.static(planner_config.joint_offset)
    q_offset = qd.static(planner_config.q_offset)

    for i_l_ in range(dyn_info.entities.link_start[i_e], dyn_info.entities.link_end[i_e]):
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
def func_planner_spheres(
    i_col,
    i_b,
    links_pos: qd.Tensor,
    links_quat: qd.Tensor,
    spheres_pos: qd.Tensor,
    plan_info: array_class.PlannerEntityInfo,
    planner_config: qd.template(),
):
    """Place the collision-proxy spheres (robot + active attached) in world frame for one FK column."""
    for i_s in range(qd.static(planner_config.n_spheres)):
        i_l = plan_info.spheres_link_idx[i_s]
        spheres_pos[i_s, i_col] = links_pos[i_l, i_col] + gu.qd_transform_by_quat(
            plan_info.spheres_pos_local[i_s], links_quat[i_l, i_col]
        )
    for i_s in range(qd.static(planner_config.n_attach_max)):
        if plan_info.attach_spheres_is_active[i_s, i_b]:
            i_l = plan_info.attach_spheres_link_idx[i_s]
            spheres_pos[qd.static(planner_config.n_spheres) + i_s, i_col] = links_pos[
                i_l, i_col
            ] + gu.qd_transform_by_quat(plan_info.attach_spheres_pos_local[i_s, i_b], links_quat[i_l, i_col])


@qd.kernel
def kernel_planner_fk_cache(
    envs_idx: qd.types.ndarray(),
    plan_state: array_class.PlannerState,
    plan_info: array_class.PlannerEntityInfo,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    planner_config: qd.template(),
):
    """FK + sphere placement of every active (candidate, knot) column into the gradient-path cache."""
    n_knots = qd.static(planner_config.n_knots)
    n_seeds = qd.static(planner_config.n_seeds)

    qd.loop_config(serialize=qd.static(planner_config.para_level < gs.PARA_LEVEL.ALL))
    for i_cw in range(envs_idx.shape[0] * n_seeds * n_knots):
        i_c = i_cw // n_knots
        if plan_state.is_active[i_c]:
            i_b = envs_idx[i_c // n_seeds]
            func_planner_fk(
                i_cw,
                i_cw,
                i_b,
                qpos=plan_state.qpos_traj,
                links_pos=plan_state.links_pos,
                links_quat=plan_state.links_quat,
                joints_xanchor=plan_state.joints_xanchor,
                joints_xaxis=plan_state.joints_xaxis,
                dyn_state=dyn_state,
                dyn_info=dyn_info,
                rigid_info=rigid_info,
                rigid_config=rigid_config,
                planner_config=planner_config,
            )
            func_planner_spheres(
                i_cw,
                i_b,
                links_pos=plan_state.links_pos,
                links_quat=plan_state.links_quat,
                spheres_pos=plan_state.spheres_pos,
                plan_info=plan_info,
                planner_config=planner_config,
            )
