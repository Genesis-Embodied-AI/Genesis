"""
Constraint building functions for the rigid body constraint solver.

This module contains functions for adding and managing equality constraints,
inequality constraints (collision, joint limits, friction loss), and weld constraints.
"""

import gstaichi as ti

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class


@ti.func
def add_collision_constraints(
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    collider_state: array_class.ColliderState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    EPS = rigid_global_info.EPS[None]

    _B = dofs_state.ctrl_mode.shape[1]
    n_dofs = dofs_state.ctrl_mode.shape[0]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        for i_col in range(collider_state.n_contacts[i_b]):
            contact_data_link_a = collider_state.contact_data.link_a[i_col, i_b]
            contact_data_link_b = collider_state.contact_data.link_b[i_col, i_b]

            contact_data_pos = collider_state.contact_data.pos[i_col, i_b]
            contact_data_normal = collider_state.contact_data.normal[i_col, i_b]
            contact_data_friction = collider_state.contact_data.friction[i_col, i_b]
            contact_data_sol_params = collider_state.contact_data.sol_params[i_col, i_b]
            contact_data_penetration = collider_state.contact_data.penetration[i_col, i_b]

            link_a = contact_data_link_a
            link_b = contact_data_link_b
            link_a_maybe_batch = [link_a, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else link_a
            link_b_maybe_batch = [link_b, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else link_b

            d1, d2 = gu.ti_orthogonals(contact_data_normal)

            invweight = links_info.invweight[link_a_maybe_batch][0]
            if link_b > -1:
                invweight = invweight + links_info.invweight[link_b_maybe_batch][0]

            for i in range(4):
                d = (2 * (i % 2) - 1) * (d1 if i < 2 else d2)
                n = d * contact_data_friction - contact_data_normal

                n_con = ti.atomic_add(constraint_state.n_constraints[i_b], 1)
                if ti.static(static_rigid_sim_config.sparse_solve):
                    for i_d_ in range(constraint_state.jac_n_relevant_dofs[n_con, i_b]):
                        i_d = constraint_state.jac_relevant_dofs[n_con, i_d_, i_b]
                        constraint_state.jac[n_con, i_d, i_b] = gs.ti_float(0.0)
                else:
                    for i_d in range(n_dofs):
                        constraint_state.jac[n_con, i_d, i_b] = gs.ti_float(0.0)

                con_n_relevant_dofs = 0
                jac_qvel = gs.ti_float(0.0)
                for i_ab in range(2):
                    sign = gs.ti_float(-1.0)
                    link = link_a
                    if i_ab == 1:
                        sign = gs.ti_float(1.0)
                        link = link_b

                    while link > -1:
                        link_maybe_batch = [link, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else link

                        # reverse order to make sure dofs in each row of self.jac_relevant_dofs is strictly descending
                        for i_d_ in range(links_info.n_dofs[link_maybe_batch]):
                            i_d = links_info.dof_end[link_maybe_batch] - 1 - i_d_

                            cdof_ang = dofs_state.cdof_ang[i_d, i_b]
                            cdot_vel = dofs_state.cdof_vel[i_d, i_b]

                            t_quat = gu.ti_identity_quat()
                            t_pos = contact_data_pos - links_state.root_COM[link, i_b]
                            _, vel = gu.ti_transform_motion_by_trans_quat(cdof_ang, cdot_vel, t_pos, t_quat)

                            diff = sign * vel
                            jac = diff @ n
                            jac_qvel = jac_qvel + jac * dofs_state.vel[i_d, i_b]
                            constraint_state.jac[n_con, i_d, i_b] = constraint_state.jac[n_con, i_d, i_b] + jac

                            if ti.static(static_rigid_sim_config.sparse_solve):
                                constraint_state.jac_relevant_dofs[n_con, con_n_relevant_dofs, i_b] = i_d
                                con_n_relevant_dofs = con_n_relevant_dofs + 1

                        link = links_info.parent_idx[link_maybe_batch]

                if ti.static(static_rigid_sim_config.sparse_solve):
                    constraint_state.jac_n_relevant_dofs[n_con, i_b] = con_n_relevant_dofs
                imp, aref = gu.imp_aref(
                    contact_data_sol_params, -contact_data_penetration, jac_qvel, -contact_data_penetration
                )

                diag = invweight + contact_data_friction * contact_data_friction * invweight
                diag *= 2 * contact_data_friction * contact_data_friction * (1 - imp) / imp
                diag = ti.max(diag, EPS)

                constraint_state.diag[n_con, i_b] = diag
                constraint_state.aref[n_con, i_b] = aref
                constraint_state.efc_D[n_con, i_b] = 1 / diag


@ti.func
def func_equality_connect(
    i_b,
    i_e,
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    dofs_state: array_class.DofsState,
    equalities_info: array_class.EqualitiesInfo,
    constraint_state: array_class.ConstraintState,
    collider_state: array_class.ColliderState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    EPS = rigid_global_info.EPS[None]

    n_dofs = dofs_state.ctrl_mode.shape[0]

    link1_idx = equalities_info.eq_obj1id[i_e, i_b]
    link2_idx = equalities_info.eq_obj2id[i_e, i_b]
    link_a_maybe_batch = [link1_idx, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else link1_idx
    link_b_maybe_batch = [link2_idx, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else link2_idx
    anchor1_pos = gs.ti_vec3(
        [
            equalities_info.eq_data[i_e, i_b][0],
            equalities_info.eq_data[i_e, i_b][1],
            equalities_info.eq_data[i_e, i_b][2],
        ]
    )
    anchor2_pos = gs.ti_vec3(
        [
            equalities_info.eq_data[i_e, i_b][3],
            equalities_info.eq_data[i_e, i_b][4],
            equalities_info.eq_data[i_e, i_b][5],
        ]
    )
    sol_params = equalities_info.sol_params[i_e, i_b]

    # Transform anchor positions to global coordinates
    global_anchor1 = gu.ti_transform_by_trans_quat(
        pos=anchor1_pos,
        trans=links_state.pos[link1_idx, i_b],
        quat=links_state.quat[link1_idx, i_b],
    )
    global_anchor2 = gu.ti_transform_by_trans_quat(
        pos=anchor2_pos,
        trans=links_state.pos[link2_idx, i_b],
        quat=links_state.quat[link2_idx, i_b],
    )

    invweight = links_info.invweight[link_a_maybe_batch][0] + links_info.invweight[link_b_maybe_batch][0]

    for i_3 in range(3):
        n_con = ti.atomic_add(constraint_state.n_constraints[i_b], 1)
        ti.atomic_add(constraint_state.n_constraints_equality[i_b], 1)
        con_n_relevant_dofs = 0

        if ti.static(static_rigid_sim_config.sparse_solve):
            for i_d_ in range(constraint_state.jac_n_relevant_dofs[n_con, i_b]):
                i_d = constraint_state.jac_relevant_dofs[n_con, i_d_, i_b]
                constraint_state.jac[n_con, i_d, i_b] = gs.ti_float(0.0)
        else:
            for i_d in range(n_dofs):
                constraint_state.jac[n_con, i_d, i_b] = gs.ti_float(0.0)

        jac_qvel = gs.ti_float(0.0)
        for i_ab in range(2):
            sign = gs.ti_float(1.0)
            link = link1_idx
            pos = global_anchor1
            if i_ab == 1:
                sign = gs.ti_float(-1.0)
                link = link2_idx
                pos = global_anchor2

            while link > -1:
                link_maybe_batch = [link, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else link

                for i_d_ in range(links_info.n_dofs[link_maybe_batch]):
                    i_d = links_info.dof_end[link_maybe_batch] - 1 - i_d_

                    cdof_ang = dofs_state.cdof_ang[i_d, i_b]
                    cdot_vel = dofs_state.cdof_vel[i_d, i_b]

                    t_quat = gu.ti_identity_quat()
                    t_pos = pos - links_state.root_COM[link, i_b]
                    ang, vel = gu.ti_transform_motion_by_trans_quat(cdof_ang, cdot_vel, t_pos, t_quat)

                    diff = sign * vel
                    jac = diff[i_3]
                    jac_qvel = jac_qvel + jac * dofs_state.vel[i_d, i_b]
                    constraint_state.jac[n_con, i_d, i_b] = constraint_state.jac[n_con, i_d, i_b] + jac

                    if ti.static(static_rigid_sim_config.sparse_solve):
                        constraint_state.jac_relevant_dofs[n_con, con_n_relevant_dofs, i_b] = i_d
                        con_n_relevant_dofs = con_n_relevant_dofs + 1

                link = links_info.parent_idx[link_maybe_batch]

        if ti.static(static_rigid_sim_config.sparse_solve):
            constraint_state.jac_n_relevant_dofs[n_con, i_b] = con_n_relevant_dofs

        pos_diff = global_anchor1 - global_anchor2
        penetration = pos_diff.norm()

        imp, aref = gu.imp_aref(sol_params, -penetration, jac_qvel, pos_diff[i_3])

        diag = ti.max(invweight * (1.0 - imp) / imp, EPS)

        constraint_state.diag[n_con, i_b] = diag
        constraint_state.aref[n_con, i_b] = aref
        constraint_state.efc_D[n_con, i_b] = 1.0 / diag


@ti.func
def func_equality_joint(
    i_b,
    i_e,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    equalities_info: array_class.EqualitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    EPS = rigid_global_info.EPS[None]

    n_dofs = constraint_state.jac.shape[1]

    sol_params = equalities_info.sol_params[i_e, i_b]

    I_joint1 = (
        [equalities_info.eq_obj1id[i_e, i_b], i_b]
        if ti.static(static_rigid_sim_config.batch_joints_info)
        else equalities_info.eq_obj1id[i_e, i_b]
    )
    I_joint2 = (
        [equalities_info.eq_obj2id[i_e, i_b], i_b]
        if ti.static(static_rigid_sim_config.batch_joints_info)
        else equalities_info.eq_obj2id[i_e, i_b]
    )
    i_qpos1 = joints_info.q_start[I_joint1]
    i_qpos2 = joints_info.q_start[I_joint2]
    i_dof1 = joints_info.dof_start[I_joint1]
    i_dof2 = joints_info.dof_start[I_joint2]
    I_dof1 = [i_dof1, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_dof1
    I_dof2 = [i_dof2, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_dof2

    n_con = ti.atomic_add(constraint_state.n_constraints[i_b], 1)
    ti.atomic_add(constraint_state.n_constraints_equality[i_b], 1)

    if ti.static(static_rigid_sim_config.sparse_solve):
        for i_d_ in range(constraint_state.jac_n_relevant_dofs[n_con, i_b]):
            i_d = constraint_state.jac_relevant_dofs[n_con, i_d_, i_b]
            constraint_state.jac[n_con, i_d, i_b] = gs.ti_float(0.0)
    else:
        for i_d in range(n_dofs):
            constraint_state.jac[n_con, i_d, i_b] = gs.ti_float(0.0)

    pos1 = rigid_global_info.qpos[i_qpos1, i_b]
    pos2 = rigid_global_info.qpos[i_qpos2, i_b]
    ref1 = rigid_global_info.qpos0[i_qpos1, i_b]
    ref2 = rigid_global_info.qpos0[i_qpos2, i_b]

    # TODO: zero objid2
    diff = pos2 - ref2
    pos = pos1 - ref1
    deriv = gs.ti_float(0.0)

    # y - y0 = a0 + a1 * (x-x0) + a2 * (x-x0)^2 + a3 * (x-fx0)^3 + a4 * (x-x0)^4
    for i_5 in range(5):
        diff_power = diff**i_5
        pos = pos - diff_power * equalities_info.eq_data[i_e, i_b][i_5]
        if i_5 < 4:
            deriv = deriv + equalities_info.eq_data[i_e, i_b][i_5 + 1] * diff_power * (i_5 + 1)

    constraint_state.jac[n_con, i_dof1, i_b] = gs.ti_float(1.0)
    constraint_state.jac[n_con, i_dof2, i_b] = -deriv
    jac_qvel = (
        constraint_state.jac[n_con, i_dof1, i_b] * dofs_state.vel[i_dof1, i_b]
        + constraint_state.jac[n_con, i_dof2, i_b] * dofs_state.vel[i_dof2, i_b]
    )
    invweight = dofs_info.invweight[I_dof1] + dofs_info.invweight[I_dof2]

    imp, aref = gu.imp_aref(sol_params, -ti.abs(pos), jac_qvel, pos)

    diag = ti.max(invweight * (1.0 - imp) / imp, EPS)

    constraint_state.diag[n_con, i_b] = diag
    constraint_state.aref[n_con, i_b] = aref
    constraint_state.efc_D[n_con, i_b] = 1.0 / diag


@ti.kernel(fastcache=gs.use_fastcache)
def add_equality_constraints(
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    joints_info: array_class.JointsInfo,
    equalities_info: array_class.EqualitiesInfo,
    constraint_state: array_class.ConstraintState,
    collider_state: array_class.ColliderState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    _B = dofs_state.ctrl_mode.shape[1]

    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        constraint_state.n_constraints[i_b] = 0
        constraint_state.n_constraints_equality[i_b] = 0

        for i_e in range(constraint_state.ti_n_equalities[i_b]):
            if equalities_info.eq_type[i_e, i_b] == gs.EQUALITY_TYPE.CONNECT:
                func_equality_connect(
                    i_b,
                    i_e,
                    links_info=links_info,
                    links_state=links_state,
                    dofs_state=dofs_state,
                    equalities_info=equalities_info,
                    constraint_state=constraint_state,
                    collider_state=collider_state,
                    rigid_global_info=rigid_global_info,
                    static_rigid_sim_config=static_rigid_sim_config,
                )

            elif equalities_info.eq_type[i_e, i_b] == gs.EQUALITY_TYPE.WELD:
                func_equality_weld(
                    i_b,
                    i_e,
                    links_info=links_info,
                    links_state=links_state,
                    dofs_state=dofs_state,
                    equalities_info=equalities_info,
                    constraint_state=constraint_state,
                    rigid_global_info=rigid_global_info,
                    static_rigid_sim_config=static_rigid_sim_config,
                )
            elif equalities_info.eq_type[i_e, i_b] == gs.EQUALITY_TYPE.JOINT:
                func_equality_joint(
                    i_b,
                    i_e,
                    joints_info=joints_info,
                    dofs_state=dofs_state,
                    dofs_info=dofs_info,
                    equalities_info=equalities_info,
                    constraint_state=constraint_state,
                    rigid_global_info=rigid_global_info,
                    static_rigid_sim_config=static_rigid_sim_config,
                )


@ti.kernel(fastcache=gs.use_fastcache)
def add_inequality_constraints(
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    joints_info: array_class.JointsInfo,
    constraint_state: array_class.ConstraintState,
    collider_state: array_class.ColliderState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    add_frictionloss_constraints(
        links_info=links_info,
        joints_info=joints_info,
        dofs_info=dofs_info,
        dofs_state=dofs_state,
        rigid_global_info=rigid_global_info,
        constraint_state=constraint_state,
        static_rigid_sim_config=static_rigid_sim_config,
    )
    if ti.static(static_rigid_sim_config.enable_collision):
        add_collision_constraints(
            links_info=links_info,
            links_state=links_state,
            dofs_state=dofs_state,
            constraint_state=constraint_state,
            collider_state=collider_state,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )
    if ti.static(static_rigid_sim_config.enable_joint_limit):
        add_joint_limit_constraints(
            links_info=links_info,
            joints_info=joints_info,
            dofs_info=dofs_info,
            dofs_state=dofs_state,
            rigid_global_info=rigid_global_info,
            constraint_state=constraint_state,
            static_rigid_sim_config=static_rigid_sim_config,
        )


@ti.func
def func_equality_weld(
    i_b,
    i_e,
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    dofs_state: array_class.DofsState,
    equalities_info: array_class.EqualitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    EPS = rigid_global_info.EPS[None]

    n_dofs = dofs_state.ctrl_mode.shape[0]

    # TODO: sparse mode
    # Get equality info for this constraint
    link1_idx = equalities_info.eq_obj1id[i_e, i_b]
    link2_idx = equalities_info.eq_obj2id[i_e, i_b]
    link_a_maybe_batch = [link1_idx, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else link1_idx
    link_b_maybe_batch = [link2_idx, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else link2_idx

    # For weld, eq_data layout:
    # [0:3]  : anchor2 (local pos in body2)
    # [3:6]  : anchor1 (local pos in body1)
    # [6:10] : relative pose (quat) of body 2 related to body 1 to match orientations
    # [10]   : torquescale
    anchor1_pos = gs.ti_vec3(
        [
            equalities_info.eq_data[i_e, i_b][3],
            equalities_info.eq_data[i_e, i_b][4],
            equalities_info.eq_data[i_e, i_b][5],
        ]
    )
    anchor2_pos = gs.ti_vec3(
        [
            equalities_info.eq_data[i_e, i_b][0],
            equalities_info.eq_data[i_e, i_b][1],
            equalities_info.eq_data[i_e, i_b][2],
        ]
    )
    relpose = gs.ti_vec4(
        [
            equalities_info.eq_data[i_e, i_b][6],
            equalities_info.eq_data[i_e, i_b][7],
            equalities_info.eq_data[i_e, i_b][8],
            equalities_info.eq_data[i_e, i_b][9],
        ]
    )
    torquescale = equalities_info.eq_data[i_e, i_b][10]
    sol_params = equalities_info.sol_params[i_e, i_b]

    # Transform anchor positions to global coordinates
    global_anchor1 = gu.ti_transform_by_trans_quat(
        pos=anchor1_pos,
        trans=links_state.pos[link1_idx, i_b],
        quat=links_state.quat[link1_idx, i_b],
    )
    global_anchor2 = gu.ti_transform_by_trans_quat(
        pos=anchor2_pos,
        trans=links_state.pos[link2_idx, i_b],
        quat=links_state.quat[link2_idx, i_b],
    )

    pos_error = global_anchor1 - global_anchor2

    # Compute orientation error.
    # For weld: compute q = body1_quat * relpose, then error = (inv(body2_quat) * q)
    quat_body1 = links_state.quat[link1_idx, i_b]
    quat_body2 = links_state.quat[link2_idx, i_b]
    q = gu.ti_quat_mul(quat_body1, relpose)
    inv_quat_body2 = gu.ti_inv_quat(quat_body2)
    error_quat = gu.ti_quat_mul(inv_quat_body2, q)
    # Take the vector (axis) part and scale by torquescale.
    rot_error = gs.ti_vec3([error_quat[1], error_quat[2], error_quat[3]]) * torquescale

    all_error = gs.ti_vec6([pos_error[0], pos_error[1], pos_error[2], rot_error[0], rot_error[1], rot_error[2]])
    pos_imp = all_error.norm()

    # Compute inverse weight from both bodies.
    invweight = links_info.invweight[link_a_maybe_batch] + links_info.invweight[link_b_maybe_batch]

    # --- Position part (first 3 constraints) ---
    for i in range(3):
        n_con = ti.atomic_add(constraint_state.n_constraints[i_b], 1)
        ti.atomic_add(constraint_state.n_constraints_equality[i_b], 1)
        con_n_relevant_dofs = 0

        if ti.static(static_rigid_sim_config.sparse_solve):
            for i_d_ in range(constraint_state.jac_n_relevant_dofs[n_con, i_b]):
                i_d = constraint_state.jac_relevant_dofs[n_con, i_d_, i_b]
                constraint_state.jac[n_con, i_d, i_b] = gs.ti_float(0.0)
        else:
            for i_d in range(n_dofs):
                constraint_state.jac[n_con, i_d, i_b] = gs.ti_float(0.0)

        jac_qvel = gs.ti_float(0.0)
        for i_ab in range(2):
            sign = gs.ti_float(1.0) if i_ab == 0 else gs.ti_float(-1.0)
            link = link1_idx if i_ab == 0 else link2_idx
            pos_anchor = global_anchor1 if i_ab == 0 else global_anchor2

            # Accumulate jacobian contributions along the kinematic chain.
            # (Assuming similar structure to equality_connect.)
            while link > -1:
                link_maybe_batch = [link, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else link

                for i_d_ in range(links_info.n_dofs[link_maybe_batch]):
                    i_d = links_info.dof_end[link_maybe_batch] - 1 - i_d_
                    cdof_ang = dofs_state.cdof_ang[i_d, i_b]
                    cdot_vel = dofs_state.cdof_vel[i_d, i_b]

                    t_quat = gu.ti_identity_quat()
                    t_pos = pos_anchor - links_state.root_COM[link, i_b]
                    ang, vel = gu.ti_transform_motion_by_trans_quat(cdof_ang, cdot_vel, t_pos, t_quat)
                    diff = sign * vel
                    jac = diff[i]
                    jac_qvel = jac_qvel + jac * dofs_state.vel[i_d, i_b]
                    constraint_state.jac[n_con, i_d, i_b] = constraint_state.jac[n_con, i_d, i_b] + jac

                    if ti.static(static_rigid_sim_config.sparse_solve):
                        constraint_state.jac_relevant_dofs[n_con, con_n_relevant_dofs, i_b] = i_d
                        con_n_relevant_dofs = con_n_relevant_dofs + 1
                link = links_info.parent_idx[link_maybe_batch]

        if ti.static(static_rigid_sim_config.sparse_solve):
            constraint_state.jac_n_relevant_dofs[n_con, i_b] = con_n_relevant_dofs

        imp, aref = gu.imp_aref(sol_params, -pos_imp, jac_qvel, pos_error[i])
        diag = ti.max(invweight[0] * (1 - imp) / imp, EPS)

        constraint_state.diag[n_con, i_b] = diag
        constraint_state.aref[n_con, i_b] = aref
        constraint_state.efc_D[n_con, i_b] = 1.0 / diag

    # --- Orientation part (next 3 constraints) ---
    n_con = ti.atomic_add(constraint_state.n_constraints[i_b], 3)
    ti.atomic_add(constraint_state.n_constraints_equality[i_b], 3)
    con_n_relevant_dofs = 0
    for i_con in range(n_con, n_con + 3):
        for i_d in range(n_dofs):
            constraint_state.jac[i_con, i_d, i_b] = gs.ti_float(0.0)

    for i_ab in range(2):
        sign = gs.ti_float(1.0) if i_ab == 0 else gs.ti_float(-1.0)
        link = link1_idx if i_ab == 0 else link2_idx
        # For rotation, we use the body's orientation (here we use its quaternion)
        # and a suitable reference frame. (You may need a more detailed implementation.)
        while link > -1:
            link_maybe_batch = [link, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else link

            for i_d_ in range(links_info.n_dofs[link_maybe_batch]):
                i_d = links_info.dof_end[link_maybe_batch] - 1 - i_d_
                jac = sign * dofs_state.cdof_ang[i_d, i_b]

                for i_con in range(n_con, n_con + 3):
                    constraint_state.jac[i_con, i_d, i_b] = constraint_state.jac[i_con, i_d, i_b] + jac[i_con - n_con]
            link = links_info.parent_idx[link_maybe_batch]

    jac_qvel = ti.Vector([0.0, 0.0, 0.0])
    for i_d in range(n_dofs):
        # quat2 = neg(q1)*(jac0-jac1)
        # quat3 = neg(q1)*(jac0-jac1)*q0*relpose
        jac_diff_r = ti.Vector(
            [
                constraint_state.jac[n_con, i_d, i_b],
                constraint_state.jac[n_con + 1, i_d, i_b],
                constraint_state.jac[n_con + 2, i_d, i_b],
            ]
        )
        quat2 = gu.ti_quat_mul_axis(inv_quat_body2, jac_diff_r)
        quat3 = gu.ti_quat_mul(quat2, q)

        for i_con in range(n_con, n_con + 3):
            constraint_state.jac[i_con, i_d, i_b] = 0.5 * quat3[i_con - n_con + 1] * torquescale
            jac_qvel[i_con - n_con] = (
                jac_qvel[i_con - n_con] + constraint_state.jac[i_con, i_d, i_b] * dofs_state.vel[i_d, i_b]
            )

    if ti.static(static_rigid_sim_config.sparse_solve):
        for i_con in range(n_con, n_con + 3):
            constraint_state.jac_n_relevant_dofs[i_con, i_b] = con_n_relevant_dofs

    for i_con in range(n_con, n_con + 3):
        imp, aref = gu.imp_aref(sol_params, -pos_imp, jac_qvel[i_con - n_con], rot_error[i_con - n_con])
        diag = ti.max(invweight[1] * (1.0 - imp) / imp, EPS)

        constraint_state.diag[i_con, i_b] = diag
        constraint_state.aref[i_con, i_b] = aref
        constraint_state.efc_D[i_con, i_b] = 1.0 / diag


@ti.func
def add_joint_limit_constraints(
    links_info: array_class.LinksInfo,
    joints_info: array_class.JointsInfo,
    dofs_info: array_class.DofsInfo,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
):
    EPS = rigid_global_info.EPS[None]

    _B = constraint_state.jac.shape[2]
    n_links = links_info.root_idx.shape[0]
    n_dofs = dofs_state.ctrl_mode.shape[0]

    # TODO: sparse mode
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        for i_l in range(n_links):
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

            for i_j in range(links_info.joint_start[I_l], links_info.joint_end[I_l]):
                I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j

                if joints_info.type[I_j] == gs.JOINT_TYPE.REVOLUTE or joints_info.type[I_j] == gs.JOINT_TYPE.PRISMATIC:
                    i_q = joints_info.q_start[I_j]
                    i_d = joints_info.dof_start[I_j]
                    I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
                    pos_delta_min = rigid_global_info.qpos[i_q, i_b] - dofs_info.limit[I_d][0]
                    pos_delta_max = dofs_info.limit[I_d][1] - rigid_global_info.qpos[i_q, i_b]
                    pos_delta = ti.min(pos_delta_min, pos_delta_max)

                    if pos_delta < 0:
                        jac = (pos_delta_min < pos_delta_max) * 2 - 1
                        jac_qvel = jac * dofs_state.vel[i_d, i_b]
                        imp, aref = gu.imp_aref(joints_info.sol_params[I_j], pos_delta, jac_qvel, pos_delta)
                        diag = ti.max(dofs_info.invweight[I_d] * (1 - imp) / imp, EPS)

                        n_con = ti.atomic_add(constraint_state.n_constraints[i_b], 1)
                        constraint_state.diag[n_con, i_b] = diag
                        constraint_state.aref[n_con, i_b] = aref
                        constraint_state.efc_D[n_con, i_b] = 1 / diag

                        if ti.static(static_rigid_sim_config.sparse_solve):
                            for i_d2_ in range(constraint_state.jac_n_relevant_dofs[n_con, i_b]):
                                i_d2 = constraint_state.jac_relevant_dofs[n_con, i_d2_, i_b]
                                constraint_state.jac[n_con, i_d2, i_b] = gs.ti_float(0.0)
                        else:
                            for i_d2 in range(n_dofs):
                                constraint_state.jac[n_con, i_d2, i_b] = gs.ti_float(0.0)
                        constraint_state.jac[n_con, i_d, i_b] = jac

                        if ti.static(static_rigid_sim_config.sparse_solve):
                            constraint_state.jac_n_relevant_dofs[n_con, i_b] = 1
                            constraint_state.jac_relevant_dofs[n_con, 0, i_b] = i_d


@ti.func
def add_frictionloss_constraints(
    links_info: array_class.LinksInfo,
    joints_info: array_class.JointsInfo,
    dofs_info: array_class.DofsInfo,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
):
    EPS = rigid_global_info.EPS[None]

    _B = constraint_state.jac.shape[2]
    n_links = links_info.root_idx.shape[0]
    n_dofs = dofs_state.ctrl_mode.shape[0]

    # TODO: sparse mode
    # FIXME: The condition `if dofs_info.frictionloss[I_d] > EPS:` is not correctly evaluated on Apple Metal
    # if `serialize=True`...
    ti.loop_config(
        serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL and gs.backend != gs.metal)
    )
    for i_b in range(_B):
        constraint_state.n_constraints_frictionloss[i_b] = 0

        for i_l in range(n_links):
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

            for i_j in range(links_info.joint_start[I_l], links_info.joint_end[I_l]):
                I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j

                for i_d in range(joints_info.dof_start[I_j], joints_info.dof_end[I_j]):
                    I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d

                    if dofs_info.frictionloss[I_d] > EPS:
                        jac = 1.0
                        jac_qvel = jac * dofs_state.vel[i_d, i_b]
                        imp, aref = gu.imp_aref(joints_info.sol_params[I_j], 0.0, jac_qvel, 0.0)
                        diag = ti.max(dofs_info.invweight[I_d] * (1.0 - imp) / imp, EPS)

                        i_con = ti.atomic_add(constraint_state.n_constraints[i_b], 1)
                        ti.atomic_add(constraint_state.n_constraints_frictionloss[i_b], 1)

                        constraint_state.diag[i_con, i_b] = diag
                        constraint_state.aref[i_con, i_b] = aref
                        constraint_state.efc_D[i_con, i_b] = 1.0 / diag
                        constraint_state.efc_frictionloss[i_con, i_b] = dofs_info.frictionloss[I_d]
                        for i_d2 in range(n_dofs):
                            constraint_state.jac[i_con, i_d2, i_b] = gs.ti_float(0.0)
                        constraint_state.jac[i_con, i_d, i_b] = jac


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_add_weld_constraint(
    link1_idx: ti.i32,
    link2_idx: ti.i32,
    envs_idx: ti.types.ndarray(),
    equalities_info: array_class.EqualitiesInfo,
    constraint_state: array_class.ConstraintState,
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
) -> ti.i32:
    overflow = gs.ti_bool(False)

    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_e = constraint_state.ti_n_equalities[i_b]
        if i_e == rigid_global_info.n_candidate_equalities[None]:
            overflow = True
        else:
            shared_pos = links_state.pos[link1_idx, i_b]
            pos1 = gu.ti_inv_transform_by_trans_quat(
                shared_pos, links_state.pos[link1_idx, i_b], links_state.quat[link1_idx, i_b]
            )
            pos2 = gu.ti_inv_transform_by_trans_quat(
                shared_pos, links_state.pos[link2_idx, i_b], links_state.quat[link2_idx, i_b]
            )

            equalities_info.eq_type[i_e, i_b] = gs.ti_int(gs.EQUALITY_TYPE.WELD)
            equalities_info.eq_obj1id[i_e, i_b] = link1_idx
            equalities_info.eq_obj2id[i_e, i_b] = link2_idx

            for i_3 in ti.static(range(3)):
                equalities_info.eq_data[i_e, i_b][i_3 + 3] = pos1[i_3]
                equalities_info.eq_data[i_e, i_b][i_3] = pos2[i_3]

            relpose = gu.ti_quat_mul(gu.ti_inv_quat(links_state.quat[link1_idx, i_b]), links_state.quat[link2_idx, i_b])

            for i_4 in ti.static(range(4)):
                equalities_info.eq_data[i_e, i_b][i_4 + 6] = relpose[i_4]

            equalities_info.eq_data[i_e, i_b][10] = 1.0

            equalities_info.sol_params[i_e, i_b] = ti.Vector(
                [2 * rigid_global_info.substep_dt[None], 1.0, 0.9, 0.95, 0.001, 0.5, 2.0]
            )

            constraint_state.ti_n_equalities[i_b] = constraint_state.ti_n_equalities[i_b] + 1
    return overflow


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_delete_weld_constraint(
    link1_idx: ti.i32,
    link2_idx: ti.i32,
    envs_idx: ti.types.ndarray(),
    equalities_info: array_class.EqualitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        for i_e in range(rigid_global_info.n_equalities[None], constraint_state.ti_n_equalities[i_b]):
            if (
                equalities_info.eq_type[i_e, i_b] == gs.EQUALITY_TYPE.WELD
                and equalities_info.eq_obj1id[i_e, i_b] == link1_idx
                and equalities_info.eq_obj2id[i_e, i_b] == link2_idx
            ):
                if i_e < constraint_state.ti_n_equalities[i_b] - 1:
                    equalities_info.eq_type[i_e, i_b] = equalities_info.eq_type[
                        constraint_state.ti_n_equalities[i_b] - 1, i_b
                    ]
                constraint_state.ti_n_equalities[i_b] = constraint_state.ti_n_equalities[i_b] - 1


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_get_equality_constraints(
    is_padded: ti.template(),
    iout: ti.types.ndarray(),
    fout: ti.types.ndarray(),
    constraint_state: array_class.ConstraintState,
    equalities_info: array_class.EqualitiesInfo,
    static_rigid_sim_config: ti.template(),
):
    _B = constraint_state.ti_n_equalities.shape[0]
    n_eqs_max = gs.ti_int(0)

    # this is a reduction operation (global max), we have to serialize it
    # TODO: a good unittest and a better implementation from gstaichi for this kind of reduction
    ti.loop_config(serialize=True)
    for i_b in range(_B):
        n_eqs = constraint_state.ti_n_equalities[i_b]
        if n_eqs > n_eqs_max:
            n_eqs_max = n_eqs

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        i_c_start = gs.ti_int(0)
        i_e_start = gs.ti_int(0)
        if ti.static(is_padded):
            i_e_start = i_b * n_eqs_max
        else:
            for j_b in range(i_b):
                i_e_start = i_e_start + constraint_state.ti_n_equalities[j_b]

        for i_e_ in range(constraint_state.ti_n_equalities[i_b]):
            i_e = i_e_start + i_e_

            iout[i_e, 0] = equalities_info.eq_type[i_e_, i_b]
            iout[i_e, 1] = equalities_info.eq_obj1id[i_e_, i_b]
            iout[i_e, 2] = equalities_info.eq_obj2id[i_e_, i_b]

            if equalities_info.eq_type[i_e_, i_b] == gs.EQUALITY_TYPE.CONNECT:
                for i_c_ in ti.static(range(3)):
                    i_c = i_c_start + i_c_
                    fout[i_e, i_c_] = constraint_state.efc_force[i_c, i_b]
                i_c_start = i_c_start + 3
            elif equalities_info.eq_type[i_e_, i_b] == gs.EQUALITY_TYPE.WELD:
                for i_c_ in ti.static(range(6)):
                    i_c = i_c_start + i_c_
                    fout[i_e, i_c_] = constraint_state.efc_force[i_c, i_b]
                i_c_start = i_c_start + 6
            elif equalities_info.eq_type[i_e_, i_b] == gs.EQUALITY_TYPE.JOINT:
                fout[i_e, 0] = constraint_state.efc_force[i_c_start, i_b]
                i_c_start = i_c_start + 1
