"""Manual reverse-mode kernels for the rigid backward pass.

Where Quadrants AD silently drops the reverse chain, we compute
the Jacobian-transpose by hand instead.

Kernels:
  - kernel_manual_forward_kinematics_bw          : forward-kinematics reverse
        (link pos/quat grad -> qpos / dofs_vel grad).
  - kernel_manual_forward_velocity_bw : link-velocity propagation reverse.
  - kernel_manual_compute_qacc_bw     : reverse of `acc = M^{-1} force` via the
        Implicit Function Theorem (writes force.grad + mass_mat.grad).

Hibernation is not supported and sets the `errno` field
(`ErrorCode.MANUAL_BW_UNIMPLEMENTED`) rather than silently corrupting
gradients.

The `@qd.func` helpers below (`d_transform_by_quat__dq`, `d_quat_mul__dlhs` /
`d_quat_mul__drhs`, `d_rotvec_to_quat__drotvec`, `d_motion_cross_motion`) are
the hand-written chain-rule derivatives of the corresponding
`genesis/utils/geom.py` forward functions, used as building blocks above.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu


@qd.func
def d_transform_by_quat__dq(v, quat, out_grad):
    """Gradient w.r.t. `quat` of `qd_transform_by_quat(v, quat)`.

    Forward (geom.py:294):
        out[0] = v0 * (qw^2 + qx^2 - qy^2 - qz^2) + v1 * (2qxy - 2qwz) + v2 * (2qxz + 2qwy)
        out[1] = v0 * (2qxy + 2qwz) + v1 * (qw^2 - qx^2 + qy^2 - qz^2) + v2 * (2qyz - 2qwx)
        out[2] = v0 * (2qxz - 2qwy) + v1 * (2qyz + 2qwx) + v2 * (qw^2 - qx^2 - qy^2 + qz^2)

    Returns Vec4 = (dL/dqw, dL/dqx, dL/dqy, dL/dqz) where
    L is whatever scalar seeded `out_grad`. (No normalization assumed.)
    """
    qw = quat[0]
    qx = quat[1]
    qy = quat[2]
    qz = quat[3]
    v0 = v[0]
    v1 = v[1]
    v2 = v[2]
    og0 = out_grad[0]
    og1 = out_grad[1]
    og2 = out_grad[2]

    # dout[0]/d{w,x,y,z}
    do0_dqw = 2.0 * (qw * v0 - qz * v1 + qy * v2)
    do0_dqx = 2.0 * (qx * v0 + qy * v1 + qz * v2)
    do0_dqy = 2.0 * (-qy * v0 + qx * v1 + qw * v2)
    do0_dqz = 2.0 * (-qz * v0 - qw * v1 + qx * v2)

    # dout[1]/d{w,x,y,z}
    do1_dqw = 2.0 * (qz * v0 + qw * v1 - qx * v2)
    do1_dqx = 2.0 * (qy * v0 - qx * v1 - qw * v2)
    do1_dqy = 2.0 * (qx * v0 + qy * v1 + qz * v2)
    do1_dqz = 2.0 * (qw * v0 - qz * v1 + qy * v2)

    # dout[2]/d{w,x,y,z}
    do2_dqw = 2.0 * (-qy * v0 + qx * v1 + qw * v2)
    do2_dqx = 2.0 * (qz * v0 + qw * v1 - qx * v2)
    do2_dqy = 2.0 * (-qw * v0 + qz * v1 - qy * v2)
    do2_dqz = 2.0 * (qx * v0 + qy * v1 + qz * v2)

    return qd.Vector(
        [
            og0 * do0_dqw + og1 * do1_dqw + og2 * do2_dqw,
            og0 * do0_dqx + og1 * do1_dqx + og2 * do2_dqx,
            og0 * do0_dqy + og1 * do1_dqy + og2 * do2_dqy,
            og0 * do0_dqz + og1 * do1_dqz + og2 * do2_dqz,
        ],
        dt=gs.qd_float,
    )


@qd.func
def d_quat_mul__dlhs(a, b, out_grad):
    """Gradient w.r.t. `a` of `quat_mul(a, b)` (Hamilton convention).

    Forward (geom.py qd_quat_mul):
        out_w = aw * bw - ax * bx - ay * by - az * bz
        out_x = aw * bx + ax * bw + ay * bz - az * by
        out_y = aw * by - ax * bz + ay * bw + az * bx
        out_z = aw * bz + ax * by - ay * bx + az * bw
    """
    bw = b[0]
    bx = b[1]
    by = b[2]
    bz = b[3]
    ogw = out_grad[0]
    ogx = out_grad[1]
    ogy = out_grad[2]
    ogz = out_grad[3]
    return qd.Vector(
        [
            # dL/daw
            ogw * bw + ogx * bx + ogy * by + ogz * bz,
            # dL/dax
            -ogw * bx + ogx * bw - ogy * bz + ogz * by,
            # dL/day
            -ogw * by + ogx * bz + ogy * bw - ogz * bx,
            # dL/daz
            -ogw * bz - ogx * by + ogy * bx + ogz * bw,
        ],
        dt=gs.qd_float,
    )


@qd.func
def d_quat_mul__drhs(a, b, out_grad):
    """Gradient w.r.t. `b` of `quat_mul(a, b)`."""
    aw = a[0]
    ax = a[1]
    ay = a[2]
    az = a[3]
    ogw = out_grad[0]
    ogx = out_grad[1]
    ogy = out_grad[2]
    ogz = out_grad[3]
    return qd.Vector(
        [
            # dL/dbw
            ogw * aw + ogx * ax + ogy * ay + ogz * az,
            # dL/dbx
            -ogw * ax + ogx * aw + ogy * az - ogz * ay,
            # dL/dby
            -ogw * ay - ogx * az + ogy * aw + ogz * ax,
            # dL/dbz
            -ogw * az + ogx * ay - ogy * ax + ogz * aw,
        ],
        dt=gs.qd_float,
    )


@qd.func
def d_rotvec_to_quat__drotvec(rotvec, eps, quat_grad):
    """Gradient w.r.t. `rotvec` of `qd_rotvec_to_quat(rotvec, eps)`.

    Forward:
        thetasq   = rx^2 + ry^2 + rz^2
        theta_reg = sqrt(thetasq + eps^2)
        c         = cos(theta_reg / 2)
        sinc      = sin(theta_reg / 2) / theta_reg
        quat      = (c, sinc * rx, sinc * ry, sinc * rz)

    Backward - by chain rule on theta_reg(rx, ry, rz):
        dtheta_reg/dri  = ri / theta_reg
        dc/dri          = -0.5 * sin(theta_reg/2) * ri/theta_reg
                        = -0.5 * (sin * ri)/theta_reg
        dsinc/dri       = [(0.5 * cos(theta_reg/2))/theta_reg
                            - sin(theta_reg/2)/theta_reg^2]  *  ri/theta_reg
                        = ri * (0.5 * c/theta_reg^2 - sinc/theta_reg^2)

    dquat[0]/dri = dc/dri = -0.5 * sin * ri/theta_reg
    dquat[1+j]/dri = d(sinc * r_j)/dri
                  = delta(i,j) * sinc + r_j * dsinc/dri

    So rotvec_grad[i] = quat_grad[0] * (-0.5 * sin * ri/theta_reg)
                      + sum_j quat_grad[1+j]  *  [delta(i,j) * sinc + r_j * dsinc/dri]
                      = quat_grad[0] * (-0.5 * sin * ri/theta_reg)
                      + sinc * quat_grad[1+i]
                      + dsinc/dri  *  sum_j quat_grad[1+j] * r_j
    """
    rx = rotvec[0]
    ry = rotvec[1]
    rz = rotvec[2]
    thetasq = rx * rx + ry * ry + rz * rz
    theta_reg = qd.sqrt(thetasq + eps * eps)
    theta_half = 0.5 * theta_reg
    sin_h = qd.sin(theta_half)
    cos_h = qd.cos(theta_half)
    sinc = sin_h / theta_reg
    # dsinc/dtheta_reg = (0.5 * cos_h - sinc) / theta_reg
    dsinc_dtheta = (0.5 * cos_h - sinc) / theta_reg

    qg_w = quat_grad[0]
    qg_x = quat_grad[1]
    qg_y = quat_grad[2]
    qg_z = quat_grad[3]

    # sum_j quat_grad[1+j]  *  r_j
    qg_dot_r = qg_x * rx + qg_y * ry + qg_z * rz

    # dquat[0]/dri = -0.5 * sin_h * ri/theta_reg
    # d(sinc * rj)/dri = delta_ij * sinc + r_j * (dsinc_dtheta  *  ri/theta_reg)
    # so total per i:
    #   ri * [ -0.5 * sin_h/theta_reg  *  qg_w + dsinc_dtheta/theta_reg  *  qg_dot_r ] + sinc * qg_{x,y,z}[i]
    coeff = -0.5 * sin_h / theta_reg * qg_w + dsinc_dtheta / theta_reg * qg_dot_r
    return qd.Vector(
        [
            coeff * rx + sinc * qg_x,
            coeff * ry + sinc * qg_y,
            coeff * rz + sinc * qg_z,
        ],
        dt=gs.qd_float,
    )


@qd.kernel(fastcache=True)
def kernel_manual_forward_kinematics_bw(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    errno: qd.Tensor,
):
    """Single-call manual reverse of `kernel_forward_kinematics_replay`.
    Iterates each entity's links leaf->root in one launch (so a child's
    `parent.{pos,quat}.grad` write lands before the parent consumes it), and
    within each link reverses the *full joint chain*.

    A link may carry more than one joint (e.g. a planar floating base =
    slide-x + slide-z + hinge-y on one body). The forward composes all of them
    in sequence and caches the per-joint intermediate pose in
    `dyn_state.links.{pos,quat}_bw[i_l, k]`: slot 0 = the "arm base" (parent pose
    composed with the link's fixed offset), slot k+1 = pose after joint k, slot
    n_joints = the final link pose. We walk those slots in reverse: seed the
    grad on the final pose, reverse joint k for k = n_joints-1 .. 0 (each step
    consumes the grad on slot k+1, emits qpos.grad for that joint and the grad
    on slot k), then reverse the arm-base composition (slot 0) into the parent's
    pose grad.

    Each joint also feeds `dyn_state.joints.{xanchor,xaxis}` downstream (velocity
    FK), so we fold those accumulated grads back through slot k as well.

    Joint types: FREE / REVOLUTE / PRISMATIC / SPHERICAL / FIXED.
    """
    qd.loop_config(
        name="manual_fk_only_bw",
        serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL),
    )
    for i_e, i_b in qd.ndrange(dyn_info.entities.n_links.shape[0], dyn_state.links.pos.shape[1]):
        n_in_e = dyn_info.entities.n_links[i_e]
        for i_l_rev in range(n_in_e):
            i_l = dyn_info.entities.link_end[i_e] - 1 - i_l_rev
            I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
            parent_idx = dyn_info.links.parent_idx[I_l]
            n_joints = dyn_info.links.joint_end[I_l] - dyn_info.links.joint_start[I_l]

            # Grad seeded on the final link pose (= slot n_joints). Carried
            # backward through the joint chain; after the loop it holds the grad
            # on slot 0 (arm base).
            g_pos = dyn_state.links.pos.grad[i_l, i_b]
            g_quat = dyn_state.links.quat.grad[i_l, i_b]

            for k_rev in range(n_joints):
                k = n_joints - 1 - k_rev
                i_j = dyn_info.links.joint_start[I_l] + k
                I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
                joint_type = dyn_info.joints.type[I_j]
                q_start = dyn_info.joints.q_start[I_j]
                dof_start = dyn_info.joints.dof_start[I_j]
                I_d = [dof_start, i_b] if qd.static(rigid_config.batch_dofs_info) else dof_start

                # Input pose to joint k (slot k), cached by the forward replay.
                pos_in = dyn_state.links.pos_bw[i_l, k, i_b]
                quat_in = dyn_state.links.quat_bw[i_l, k, i_b]
                joint_pos_off = dyn_info.joints.pos[I_j]
                xanchor_grad = dyn_state.joints.xanchor.grad[i_j, i_b]
                xaxis_grad = dyn_state.joints.xaxis.grad[i_j, i_b]

                if joint_type == gs.JOINT_TYPE.FREE:
                    # Final pose is set absolutely from qpos (slot in unused);
                    # xanchor = qpos[0:3].
                    for j in qd.static(range(3)):
                        rigid_info.qpos.grad[q_start + j, i_b] = (
                            rigid_info.qpos.grad[q_start + j, i_b] + g_pos[j] + xanchor_grad[j]
                        )
                    for j in qd.static(range(4)):
                        rigid_info.qpos.grad[q_start + 3 + j, i_b] = (
                            rigid_info.qpos.grad[q_start + 3 + j, i_b] + g_quat[j]
                        )
                    g_pos = qd.Vector([0.0, 0.0, 0.0], dt=gs.qd_float)
                    g_quat = qd.Vector([0.0, 0.0, 0.0, 0.0], dt=gs.qd_float)

                elif joint_type == gs.JOINT_TYPE.REVOLUTE:
                    axis = dyn_info.dofs.motion_ang[I_d]
                    angle = rigid_info.qpos[q_start, i_b] - rigid_info.qpos0[q_start, i_b]
                    rotvec = axis * angle
                    qloc = gu.qd_rotvec_to_quat(rotvec, rigid_info.EPS[None])
                    # quat_out = transform_quat_by_quat(qloc, quat_in) = quat_mul(quat_in, qloc)
                    quat_out = gu.qd_transform_quat_by_quat(qloc, quat_in)

                    # pos_out = xanchor - transform(joint_pos_off, quat_out)
                    # xanchor = transform(joint_pos_off, quat_in) + pos_in
                    gq_out = g_quat - d_transform_by_quat__dq(joint_pos_off, quat_out, g_pos)
                    g_qloc = d_quat_mul__drhs(quat_in, qloc, gq_out)
                    g_quat_in_apply = d_quat_mul__dlhs(quat_in, qloc, gq_out)
                    rotvec_grad = d_rotvec_to_quat__drotvec(rotvec, rigid_info.EPS[None], g_qloc)
                    angle_grad = axis[0] * rotvec_grad[0] + axis[1] * rotvec_grad[1] + axis[2] * rotvec_grad[2]
                    rigid_info.qpos.grad[q_start, i_b] = rigid_info.qpos.grad[q_start, i_b] + angle_grad

                    # grad into xanchor = g_pos (from pos_out) + downstream xanchor_grad
                    g_xanchor = g_pos + xanchor_grad
                    g_quat_in = (
                        g_quat_in_apply
                        + d_transform_by_quat__dq(joint_pos_off, quat_in, g_xanchor)
                        + d_transform_by_quat__dq(axis, quat_in, xaxis_grad)
                    )
                    g_pos = g_xanchor
                    g_quat = g_quat_in

                elif joint_type == gs.JOINT_TYPE.PRISMATIC:
                    axis = dyn_info.dofs.motion_vel[I_d]
                    displacement = rigid_info.qpos[q_start, i_b] - rigid_info.qpos0[q_start, i_b]
                    xaxis = gu.qd_transform_by_quat(axis, quat_in)
                    # pos_out = pos_in + xaxis * displacement ; quat_out = quat_in
                    displacement_grad = xaxis[0] * g_pos[0] + xaxis[1] * g_pos[1] + xaxis[2] * g_pos[2]
                    rigid_info.qpos.grad[q_start, i_b] = rigid_info.qpos.grad[q_start, i_b] + displacement_grad
                    g_xaxis = qd.Vector(
                        [
                            g_pos[0] * displacement + xaxis_grad[0],
                            g_pos[1] * displacement + xaxis_grad[1],
                            g_pos[2] * displacement + xaxis_grad[2],
                        ],
                        dt=gs.qd_float,
                    )
                    g_xanchor = g_pos + xanchor_grad
                    g_quat_in = (
                        g_quat
                        + d_transform_by_quat__dq(axis, quat_in, g_xaxis)
                        + d_transform_by_quat__dq(joint_pos_off, quat_in, g_xanchor)
                    )
                    g_pos = g_xanchor
                    g_quat = g_quat_in

                elif joint_type == gs.JOINT_TYPE.SPHERICAL:
                    # qloc = qpos[q_start:q_start+4] (direct); quat_out = quat_mul(quat_in, qloc).
                    # axis defaults to [0,0,1] (xaxis = transform(axis, quat_in)).
                    axis = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)
                    qloc = qd.Vector(
                        [
                            rigid_info.qpos[q_start, i_b],
                            rigid_info.qpos[q_start + 1, i_b],
                            rigid_info.qpos[q_start + 2, i_b],
                            rigid_info.qpos[q_start + 3, i_b],
                        ],
                        dt=gs.qd_float,
                    )
                    quat_out = gu.qd_transform_quat_by_quat(qloc, quat_in)
                    gq_out = g_quat - d_transform_by_quat__dq(joint_pos_off, quat_out, g_pos)
                    g_qloc = d_quat_mul__drhs(quat_in, qloc, gq_out)
                    g_quat_in_apply = d_quat_mul__dlhs(quat_in, qloc, gq_out)
                    for j in qd.static(range(4)):
                        rigid_info.qpos.grad[q_start + j, i_b] = rigid_info.qpos.grad[q_start + j, i_b] + g_qloc[j]
                    g_xanchor = g_pos + xanchor_grad
                    g_quat_in = (
                        g_quat_in_apply
                        + d_transform_by_quat__dq(joint_pos_off, quat_in, g_xanchor)
                        + d_transform_by_quat__dq(axis, quat_in, xaxis_grad)
                    )
                    g_pos = g_xanchor
                    g_quat = g_quat_in

                else:  # gs.JOINT_TYPE.FIXED - pose passes through unchanged.
                    pass

                for j in qd.static(range(3)):
                    dyn_state.joints.xanchor.grad[i_j, i_b][j] = 0.0
                    dyn_state.joints.xaxis.grad[i_j, i_b][j] = 0.0

            # Reverse the arm-base composition (slot 0):
            #   arm_base_pos  = parent_pos + transform(link_offset_pos, parent_quat)
            #   arm_base_quat = quat_mul(parent_quat, link_offset_quat)
            # propagating slot-0 grad (g_pos, g_quat) into the parent's pose grad.
            if parent_idx != -1:
                parent_quat = dyn_state.links.quat[parent_idx, i_b]
                link_off_pos = dyn_info.links.pos[I_l]
                link_off_quat = dyn_info.links.quat[I_l]
                parent_quat_grad_from_pos = d_transform_by_quat__dq(link_off_pos, parent_quat, g_pos)
                parent_quat_grad_from_quat = d_quat_mul__dlhs(parent_quat, link_off_quat, g_quat)
                for j in qd.static(range(3)):
                    dyn_state.links.pos.grad[parent_idx, i_b][j] = (
                        dyn_state.links.pos.grad[parent_idx, i_b][j] + g_pos[j]
                    )
                for j in qd.static(range(4)):
                    dyn_state.links.quat.grad[parent_idx, i_b][j] = (
                        dyn_state.links.quat.grad[parent_idx, i_b][j]
                        + parent_quat_grad_from_pos[j]
                        + parent_quat_grad_from_quat[j]
                    )

            for j in qd.static(range(3)):
                dyn_state.links.pos.grad[i_l, i_b][j] = 0.0
            for j in qd.static(range(4)):
                dyn_state.links.quat.grad[i_l, i_b][j] = 0.0


@qd.func
def d_motion_cross_motion(s_ang, s_vel, m_ang, m_vel, ang_g, vel_g):
    """Reverse of motion_cross_motion(s_ang, s_vel, m_ang, m_vel).

    Forward (geom.py:437):
        vel = s_ang x m_vel + s_vel x m_ang
        ang = s_ang x m_ang

    Chain rule (c=axb => a.g += b x c.g, b.g += c.g x a):
        s_ang.g += m_ang x ang.g + m_vel x vel.g
        s_vel.g += m_ang x vel.g
        m_ang.g += ang.g x s_ang + vel.g x s_vel
        m_vel.g += vel.g x s_ang

    Returns (s_ang_g, s_vel_g, m_ang_g, m_vel_g) - additive deltas.
    """
    return (
        m_ang.cross(ang_g) + m_vel.cross(vel_g),
        m_ang.cross(vel_g),
        ang_g.cross(s_ang) + vel_g.cross(s_vel),
        vel_g.cross(s_ang),
    )


@qd.kernel(fastcache=True)
def kernel_manual_forward_velocity_bw(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    errno: qd.Tensor,
):
    """Manual reverse of `kernel_forward_velocity` - single-call (no per-link
    split). Replaces the diagnostic per-link split in `substep_pre_coupling_grad`
    by computing the cross-link `cd_{vel,ang}[parent_idx]` chain explicitly.

    Inputs (read .grad seeds):
      - cd_vel.grad[i_l, i_b], cd_ang.grad[i_l, i_b]
      - cd_vel_bw.grad[i_l, k, i_b], cd_ang_bw.grad[i_l, k, i_b]
      - cdofd_ang.grad[i_d, i_b], cdofd_vel.grad[i_d, i_b]

    Outputs (accumulated .grad):
      - dyn_state.dofs.vel.grad[i_d, i_b]
      - dyn_state.dofs.cdof_ang.grad[i_d, i_b], dyn_state.dofs.cdof_vel.grad[i_d, i_b]
      - dyn_state.links.cd_vel.grad[parent_idx, i_b], dyn_state.links.cd_ang.grad[parent_idx, i_b]
        (cross-link chain - equivalent to forward replay's BW=True
        `cd_*_bw[i_l, 0] = parent.cd_*`)
    """
    qd.loop_config(
        name="manual_forward_velocity_bw",
        serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL),
    )
    for i_e, i_b in qd.ndrange(dyn_info.entities.n_links.shape[0], dyn_state.links.pos.shape[1]):
        if qd.static(rigid_config.use_hibernation):
            errno[i_b] = errno[i_b] | array_class.ErrorCode.MANUAL_BW_UNIMPLEMENTED
        else:
            n_in_e = dyn_info.entities.n_links[i_e]
            # Leaf -> root iteration so each link's cd_*_bw[0].grad (which
            # accumulates into parent.cd_*.grad) is propagated *before* the
            # parent's own iteration uses it.
            for i_l_rev in range(n_in_e):
                i_l = dyn_info.entities.link_end[i_e] - 1 - i_l_rev
                I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
                n_joints = dyn_info.links.joint_end[I_l] - dyn_info.links.joint_start[I_l]
                i_p = dyn_info.links.parent_idx[I_l]

                # --- Step 1 reverse: cd_*[i_l].grad -> cd_*_bw[i_l, n_joints].grad
                for k in qd.static(range(3)):
                    dyn_state.links.cd_vel_bw.grad[i_l, n_joints, i_b][k] = (
                        dyn_state.links.cd_vel_bw.grad[i_l, n_joints, i_b][k] + dyn_state.links.cd_vel.grad[i_l, i_b][k]
                    )
                    dyn_state.links.cd_ang_bw.grad[i_l, n_joints, i_b][k] = (
                        dyn_state.links.cd_ang_bw.grad[i_l, n_joints, i_b][k] + dyn_state.links.cd_ang.grad[i_l, i_b][k]
                    )
                # consume cd_vel/cd_ang.grad[i_l]
                for k in qd.static(range(3)):
                    dyn_state.links.cd_vel.grad[i_l, i_b][k] = 0.0
                    dyn_state.links.cd_ang.grad[i_l, i_b][k] = 0.0

                # --- Step 2: iterate joints in reverse
                for i_j_rev in range(n_joints):
                    i_j_ = n_joints - 1 - i_j_rev
                    i_j = i_j_ + dyn_info.links.joint_start[I_l]
                    I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
                    jt = dyn_info.joints.type[I_j]
                    ds = dyn_info.joints.dof_start[I_j]
                    de = dyn_info.joints.dof_end[I_j]
                    curr_idx = i_j_
                    next_idx = i_j_ + 1

                    # --- [d-rev] cd_*_bw[next].grad -> cdof_*.grad / vel.grad
                    # Forward (FREE angular: i_3=0..2 at d=ds+3+i_3; else: d in ds..de):
                    #   _vel = cdof_vel[d] * vel[d];  atomic_add(cd_vel_bw[next], _vel)
                    #   _ang = cdof_ang[d] * vel[d];  atomic_add(cd_ang_bw[next], _ang)
                    cvg_next = dyn_state.links.cd_vel_bw.grad[i_l, next_idx, i_b]
                    cag_next = dyn_state.links.cd_ang_bw.grad[i_l, next_idx, i_b]
                    if jt == gs.JOINT_TYPE.FREE:
                        for i_3 in qd.static(range(3)):
                            d_i = ds + 3 + i_3
                            v_at_d = dyn_state.dofs.vel[d_i, i_b]
                            cdv = dyn_state.dofs.cdof_vel[d_i, i_b]
                            cda = dyn_state.dofs.cdof_ang[d_i, i_b]
                            for k in qd.static(range(3)):
                                dyn_state.dofs.cdof_vel.grad[d_i, i_b][k] = (
                                    dyn_state.dofs.cdof_vel.grad[d_i, i_b][k] + cvg_next[k] * v_at_d
                                )
                                dyn_state.dofs.cdof_ang.grad[d_i, i_b][k] = (
                                    dyn_state.dofs.cdof_ang.grad[d_i, i_b][k] + cag_next[k] * v_at_d
                                )
                            dot_vel = cdv[0] * cvg_next[0] + cdv[1] * cvg_next[1] + cdv[2] * cvg_next[2]
                            dot_ang = cda[0] * cag_next[0] + cda[1] * cag_next[1] + cda[2] * cag_next[2]
                            dyn_state.dofs.vel.grad[d_i, i_b] = dyn_state.dofs.vel.grad[d_i, i_b] + dot_vel + dot_ang
                    else:
                        for i_d in range(ds, de):
                            v_at_d = dyn_state.dofs.vel[i_d, i_b]
                            cdv = dyn_state.dofs.cdof_vel[i_d, i_b]
                            cda = dyn_state.dofs.cdof_ang[i_d, i_b]
                            for k in qd.static(range(3)):
                                dyn_state.dofs.cdof_vel.grad[i_d, i_b][k] = (
                                    dyn_state.dofs.cdof_vel.grad[i_d, i_b][k] + cvg_next[k] * v_at_d
                                )
                                dyn_state.dofs.cdof_ang.grad[i_d, i_b][k] = (
                                    dyn_state.dofs.cdof_ang.grad[i_d, i_b][k] + cag_next[k] * v_at_d
                                )
                            dot_vel = cdv[0] * cvg_next[0] + cdv[1] * cvg_next[1] + cdv[2] * cvg_next[2]
                            dot_ang = cda[0] * cag_next[0] + cda[1] * cag_next[1] + cda[2] * cag_next[2]
                            dyn_state.dofs.vel.grad[i_d, i_b] = dyn_state.dofs.vel.grad[i_d, i_b] + dot_vel + dot_ang

                    # --- [c-rev] cd_*_bw[next] = cd_*_bw[curr] -> curr.grad += next.grad
                    for k in qd.static(range(3)):
                        dyn_state.links.cd_vel_bw.grad[i_l, curr_idx, i_b][k] = (
                            dyn_state.links.cd_vel_bw.grad[i_l, curr_idx, i_b][k] + cvg_next[k]
                        )
                        dyn_state.links.cd_ang_bw.grad[i_l, curr_idx, i_b][k] = (
                            dyn_state.links.cd_ang_bw.grad[i_l, curr_idx, i_b][k] + cag_next[k]
                        )
                    # consume next
                    for k in qd.static(range(3)):
                        dyn_state.links.cd_vel_bw.grad[i_l, next_idx, i_b][k] = 0.0
                        dyn_state.links.cd_ang_bw.grad[i_l, next_idx, i_b][k] = 0.0

                    # --- [b-rev] motion_cross_motion reverse:
                    # Forward: (cdofd_ang[d_i], cdofd_vel[d_i]) =
                    #     motion_cross_motion(cd_ang_bw[curr], cd_vel_bw[curr], cdof_ang[d_i], cdof_vel[d_i])
                    # Reverse via d_motion_cross_motion(s_ang, s_vel, m_ang, m_vel, ang_g, vel_g)
                    s_ang_primal = dyn_state.links.cd_ang_bw[i_l, curr_idx, i_b]
                    s_vel_primal = dyn_state.links.cd_vel_bw[i_l, curr_idx, i_b]
                    if jt == gs.JOINT_TYPE.FREE:
                        # Angular dofs i_3=0..2 at d_i = ds + 3 + i_3 (linear cdofd_* are explicit 0)
                        for i_3 in qd.static(range(3)):
                            d_i = ds + 3 + i_3
                            ang_g = dyn_state.dofs.cdofd_ang.grad[d_i, i_b]
                            vel_g = dyn_state.dofs.cdofd_vel.grad[d_i, i_b]
                            cda = dyn_state.dofs.cdof_ang[d_i, i_b]
                            cdv = dyn_state.dofs.cdof_vel[d_i, i_b]
                            s_ang_g, s_vel_g, m_ang_g, m_vel_g = d_motion_cross_motion(
                                s_ang_primal, s_vel_primal, cda, cdv, ang_g, vel_g
                            )
                            for k in qd.static(range(3)):
                                dyn_state.links.cd_ang_bw.grad[i_l, curr_idx, i_b][k] = (
                                    dyn_state.links.cd_ang_bw.grad[i_l, curr_idx, i_b][k] + s_ang_g[k]
                                )
                                dyn_state.links.cd_vel_bw.grad[i_l, curr_idx, i_b][k] = (
                                    dyn_state.links.cd_vel_bw.grad[i_l, curr_idx, i_b][k] + s_vel_g[k]
                                )
                                dyn_state.dofs.cdof_ang.grad[d_i, i_b][k] = (
                                    dyn_state.dofs.cdof_ang.grad[d_i, i_b][k] + m_ang_g[k]
                                )
                                dyn_state.dofs.cdof_vel.grad[d_i, i_b][k] = (
                                    dyn_state.dofs.cdof_vel.grad[d_i, i_b][k] + m_vel_g[k]
                                )
                            # consume cdofd_*.grad[d_i]
                            for k in qd.static(range(3)):
                                dyn_state.dofs.cdofd_ang.grad[d_i, i_b][k] = 0.0
                                dyn_state.dofs.cdofd_vel.grad[d_i, i_b][k] = 0.0
                        # Linear dofs (i_3=0..2 at d_i = ds + i_3): cdofd_* set to 0
                        # (constant), reverse is no-op; just consume to mirror P8.
                        for i_3 in qd.static(range(3)):
                            d_i = ds + i_3
                            for k in qd.static(range(3)):
                                dyn_state.dofs.cdofd_ang.grad[d_i, i_b][k] = 0.0
                                dyn_state.dofs.cdofd_vel.grad[d_i, i_b][k] = 0.0
                    else:
                        for i_d in range(ds, de):
                            ang_g = dyn_state.dofs.cdofd_ang.grad[i_d, i_b]
                            vel_g = dyn_state.dofs.cdofd_vel.grad[i_d, i_b]
                            cda = dyn_state.dofs.cdof_ang[i_d, i_b]
                            cdv = dyn_state.dofs.cdof_vel[i_d, i_b]
                            s_ang_g, s_vel_g, m_ang_g, m_vel_g = d_motion_cross_motion(
                                s_ang_primal, s_vel_primal, cda, cdv, ang_g, vel_g
                            )
                            for k in qd.static(range(3)):
                                dyn_state.links.cd_ang_bw.grad[i_l, curr_idx, i_b][k] = (
                                    dyn_state.links.cd_ang_bw.grad[i_l, curr_idx, i_b][k] + s_ang_g[k]
                                )
                                dyn_state.links.cd_vel_bw.grad[i_l, curr_idx, i_b][k] = (
                                    dyn_state.links.cd_vel_bw.grad[i_l, curr_idx, i_b][k] + s_vel_g[k]
                                )
                                dyn_state.dofs.cdof_ang.grad[i_d, i_b][k] = (
                                    dyn_state.dofs.cdof_ang.grad[i_d, i_b][k] + m_ang_g[k]
                                )
                                dyn_state.dofs.cdof_vel.grad[i_d, i_b][k] = (
                                    dyn_state.dofs.cdof_vel.grad[i_d, i_b][k] + m_vel_g[k]
                                )
                            for k in qd.static(range(3)):
                                dyn_state.dofs.cdofd_ang.grad[i_d, i_b][k] = 0.0
                                dyn_state.dofs.cdofd_vel.grad[i_d, i_b][k] = 0.0

                    # --- [a-rev] (FREE only) cd_*_bw[curr].grad -> linear cdof_*.grad / vel.grad
                    # Forward (FREE linear pre-motion_cross_motion): for i_3=0..2 at d_i = ds + i_3,
                    #   _vel = cdof_vel[d_i] * vel[d_i];  atomic_add(cd_vel_bw[curr], _vel)
                    #   _ang = cdof_ang[d_i] * vel[d_i];  atomic_add(cd_ang_bw[curr], _ang)
                    # (cdof_vel[linear] = e_i_3 constant; cdof_ang[linear] = 0 constant)
                    if jt == gs.JOINT_TYPE.FREE:
                        cvg_curr = dyn_state.links.cd_vel_bw.grad[i_l, curr_idx, i_b]
                        cag_curr = dyn_state.links.cd_ang_bw.grad[i_l, curr_idx, i_b]
                        for i_3 in qd.static(range(3)):
                            d_i = ds + i_3
                            v_at_d = dyn_state.dofs.vel[d_i, i_b]
                            cdv = dyn_state.dofs.cdof_vel[d_i, i_b]
                            cda = dyn_state.dofs.cdof_ang[d_i, i_b]
                            for k in qd.static(range(3)):
                                dyn_state.dofs.cdof_vel.grad[d_i, i_b][k] = (
                                    dyn_state.dofs.cdof_vel.grad[d_i, i_b][k] + cvg_curr[k] * v_at_d
                                )
                                dyn_state.dofs.cdof_ang.grad[d_i, i_b][k] = (
                                    dyn_state.dofs.cdof_ang.grad[d_i, i_b][k] + cag_curr[k] * v_at_d
                                )
                            dot_vel = cdv[0] * cvg_curr[0] + cdv[1] * cvg_curr[1] + cdv[2] * cvg_curr[2]
                            dot_ang = cda[0] * cag_curr[0] + cda[1] * cag_curr[1] + cda[2] * cag_curr[2]
                            dyn_state.dofs.vel.grad[d_i, i_b] = dyn_state.dofs.vel.grad[d_i, i_b] + dot_vel + dot_ang

                # --- Step 1 (initial cvel setup) reverse:
                # Forward: cd_*_bw[i_l, 0, i_b] = parent.cd_*[i_p, i_b] (if i_p != -1) else 0
                # Reverse: parent.cd_*.grad[i_p] += cd_*_bw[i_l, 0].grad; consume slot 0
                slot0_v_g = dyn_state.links.cd_vel_bw.grad[i_l, 0, i_b]
                slot0_a_g = dyn_state.links.cd_ang_bw.grad[i_l, 0, i_b]
                if i_p != -1:
                    for k in qd.static(range(3)):
                        dyn_state.links.cd_vel.grad[i_p, i_b][k] = (
                            dyn_state.links.cd_vel.grad[i_p, i_b][k] + slot0_v_g[k]
                        )
                        dyn_state.links.cd_ang.grad[i_p, i_b][k] = (
                            dyn_state.links.cd_ang.grad[i_p, i_b][k] + slot0_a_g[k]
                        )
                # consume slot 0
                for k in qd.static(range(3)):
                    dyn_state.links.cd_vel_bw.grad[i_l, 0, i_b][k] = 0.0
                    dyn_state.links.cd_ang_bw.grad[i_l, 0, i_b][k] = 0.0


@qd.kernel(fastcache=True)
def kernel_manual_compute_qacc_bw(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Manual backward for func_compute_qacc via the implicit function theorem (IFT).

    Forward chain (func_compute_qacc):
        acc_smooth = M^{-1} . force   (per-block LDL^T solve in func_solve_mass)
        acc[i]     = acc_smooth[i]    (identity copy)

    Reverse chain (manual, by IFT and symmetry of M = L^T D L):
        acc_smooth.grad += acc.grad   (reverse of the identity copy; acc.grad is then consumed since the forward
                                       copy overwrites acc)
        force_contrib    = M^{-1} . acc_smooth.grad   (M is symmetric, so M^{-T} = M^{-1})
        force.grad      += force_contrib
        mass_mat[i, i].grad += -force_contrib[i] * acc_smooth[i]
        mass_mat[i, j].grad += -(force_contrib[i] * acc_smooth[j] + force_contrib[j] * acc_smooth[i])    (i > j)
    mass_mat is stored lower-triangular with the upper half implicit by symmetry, so each off-diagonal parameter
    combines the chain terms of both its (i, j) and (j, i) occurrences. The forward factored the dense mass_mat into
    mass_mat_L / mass_mat_D_inv already, so only mass_mat.grad is touched here; this kernel is the single place the
    backward path populates it, and kernel_forward_dynamics_without_qacc.grad then reverses it into link poses.

    Like func_solve_mass_entity, the triangular solves and the IFT outer product are restricted to the mass blocks
    rooted in each entity (see entities_mass_block_dof_start in array_class.py): elimination never crosses a block,
    and cross-block mass entries are structural zeros whose grads must stay zero.
    """
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_e, i_b in qd.ndrange(dyn_info.entities.n_links.shape[0], dyn_state.dofs.force.shape[1]):
        if rigid_info.mass_mat_mask[i_e, i_b]:
            blocks_dof_start = rigid_info.entities_mass_block_dof_start[i_e]
            blocks_dof_end = rigid_info.entities_mass_block_dof_end[i_e]

            # Reverse of acc[i] = acc_smooth[i]: drain acc.grad into the acc_smooth.grad seed, stashed in
            # acc_smooth_bw[0] as the input of the LDL^T reverse solve. acc.grad is consumed since the forward copy
            # overwrites acc.
            for i_d in range(blocks_dof_start, blocks_dof_end):
                dyn_state.dofs.acc_smooth_bw[0, i_d, i_b] = (
                    dyn_state.dofs.acc_smooth.grad[i_d, i_b] + dyn_state.dofs.acc.grad[i_d, i_b]
                )
                dyn_state.dofs.acc.grad[i_d, i_b] = 0.0
                dyn_state.dofs.acc_smooth.grad[i_d, i_b] = 0.0

            # Step 1: solve L^T . u = seed (input from [0], output to [1])
            #   u[i] = seed[i] - sum_{j>i} L[j,i] * u[j]
            for i_d_ in range(blocks_dof_end - blocks_dof_start):
                i_d = blocks_dof_end - i_d_ - 1
                block_end = rigid_info.dofs_mass_block_end[i_d]
                curr = dyn_state.dofs.acc_smooth_bw[0, i_d, i_b]
                for j_d in range(i_d + 1, block_end):
                    curr = curr - rigid_info.mass_mat_L[j_d, i_d, i_b] * dyn_state.dofs.acc_smooth_bw[1, j_d, i_b]
                dyn_state.dofs.acc_smooth_bw[1, i_d, i_b] = curr

            # Step 2: v = D^{-1} . u (output to [0], overwriting input)
            for i_d in range(blocks_dof_start, blocks_dof_end):
                dyn_state.dofs.acc_smooth_bw[0, i_d, i_b] = (
                    dyn_state.dofs.acc_smooth_bw[1, i_d, i_b] * rigid_info.mass_mat_D_inv[i_d, i_b]
                )

            # Step 3: solve L . delta = v (input from [0], output to [1])
            #   delta[i] = v[i] - sum_{j<i} L[i,j] * delta[j]
            for i_d in range(blocks_dof_start, blocks_dof_end):
                block_start = rigid_info.dofs_mass_block_start[i_d]
                curr = dyn_state.dofs.acc_smooth_bw[0, i_d, i_b]
                for j_d in range(block_start, i_d):
                    curr = curr - rigid_info.mass_mat_L[i_d, j_d, i_b] * dyn_state.dofs.acc_smooth_bw[1, j_d, i_b]
                dyn_state.dofs.acc_smooth_bw[1, i_d, i_b] = curr

            # Accumulate into force.grad.
            for i_d in range(blocks_dof_start, blocks_dof_end):
                dyn_state.dofs.force.grad[i_d, i_b] = (
                    dyn_state.dofs.force.grad[i_d, i_b] + dyn_state.dofs.acc_smooth_bw[1, i_d, i_b]
                )

            # IFT seed for mass_mat.grad, restricted to each lower-triangular in-block pair (see the docstring).
            for i_d in range(blocks_dof_start, blocks_dof_end):
                block_start = rigid_info.dofs_mass_block_start[i_d]
                force_contrib_i = dyn_state.dofs.acc_smooth_bw[1, i_d, i_b]
                acc_smooth_i = dyn_state.dofs.acc_smooth[i_d, i_b]
                rigid_info.mass_mat.grad[i_d, i_d, i_b] = (
                    rigid_info.mass_mat.grad[i_d, i_d, i_b] - force_contrib_i * acc_smooth_i
                )
                for j_d in range(block_start, i_d):
                    force_contrib_j = dyn_state.dofs.acc_smooth_bw[1, j_d, i_b]
                    acc_smooth_j = dyn_state.dofs.acc_smooth[j_d, i_b]
                    rigid_info.mass_mat.grad[i_d, j_d, i_b] = rigid_info.mass_mat.grad[i_d, j_d, i_b] - (
                        force_contrib_i * acc_smooth_j + force_contrib_j * acc_smooth_i
                    )
