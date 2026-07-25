import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils import array_class
from genesis.engine.solvers.rigid.abd.inverse_kinematics import func_forward_kinematics_scratch, func_jacobian_scratch


@qd.func
def func_forward_kinematics(
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
    """Forward kinematics of the planned entity on planner-owned buffers.

    Binds the planner's link, joint and q offsets to the shared scratch forward kinematics (see
    func_forward_kinematics_scratch).
    """
    func_forward_kinematics_scratch(
        i_col_out,
        i_col_q,
        i_b,
        planner_config.entity_idx,
        planner_config.link_offset,
        planner_config.joint_offset,
        planner_config.q_offset,
        qpos,
        links_pos,
        links_quat,
        joints_xanchor,
        joints_xaxis,
        dyn_state,
        dyn_info,
        rigid_info,
        rigid_config,
    )


@qd.func
def func_inverse_kinematics_jacobian(
    i_col,
    i_b,
    ee_link,
    ee_pos,
    jacobian: qd.Tensor,
    joints_xanchor: qd.Tensor,
    joints_xaxis: qd.Tensor,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
    planner_config: qd.template(),
):
    """Spatial Jacobian of the goal point, from the per-column planner forward-kinematics frames.

    Binds the planner's link, joint and q offsets to the shared scratch Jacobian (see func_jacobian_scratch).
    """
    func_jacobian_scratch(
        i_col,
        i_b,
        planner_config.entity_idx,
        planner_config.joint_offset,
        ee_link,
        ee_pos,
        jacobian,
        joints_xanchor,
        joints_xaxis,
        dyn_info,
        rigid_config,
    )


@qd.func
def func_sphere_positions(
    i_col,
    i_b,
    links_pos: qd.Tensor,
    links_quat: qd.Tensor,
    spheres_pos: qd.Tensor,
    planner_info: array_class.PlannerEntityInfo,
    planner_config: qd.template(),
):
    """Place the collision-proxy spheres (robot + active attached) in world frame for one FK column."""
    for i_s in range(qd.static(planner_config.n_spheres)):
        i_l = planner_info.fk.spheres.link_idx[i_s]
        spheres_pos[i_s, i_col] = links_pos[i_l, i_col] + gu.qd_transform_by_quat(
            planner_info.fk.spheres.pos_local[i_s], links_quat[i_l, i_col]
        )
    for i_s in range(qd.static(planner_config.n_attach_max)):
        if planner_info.fk.attach.is_active[i_s, i_b]:
            i_l = planner_info.fk.attach.link_idx[i_s]
            spheres_pos[qd.static(planner_config.n_spheres) + i_s, i_col] = links_pos[
                i_l, i_col
            ] + gu.qd_transform_by_quat(planner_info.fk.attach.pos_local[i_s, i_b], links_quat[i_l, i_col])
