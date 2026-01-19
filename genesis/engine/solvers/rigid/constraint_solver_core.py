"""
Constraint solver core functions for the rigid body constraint solver.

This module contains the core solving algorithms including Newton solver,
line search, Hessian computation, and gradient updates.
"""

import gstaichi as ti

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.engine.solvers.rigid.rigid_solver as rigid_solver


@ti.func
def func_nt_hessian_incremental(
    i_b,
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    EPS = rigid_global_info.EPS[None]

    n_dofs = constraint_state.nt_H.shape[1]

    is_degenerated = False
    for i_c in range(constraint_state.n_constraints[i_b]):
        is_active = constraint_state.active[i_c, i_b]
        is_active_prev = constraint_state.prev_active[i_c, i_b]
        if is_active ^ is_active_prev:
            sign = 1.0 if is_active else -1.0

            efc_D_sqrt = ti.sqrt(constraint_state.efc_D[i_c, i_b])
            if ti.static(static_rigid_sim_config.sparse_solve):
                for i_d_ in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                    i_d = constraint_state.jac_relevant_dofs[i_c, i_d_, i_b]
                    constraint_state.nt_vec[i_d, i_b] = constraint_state.jac[i_c, i_d, i_b] * efc_D_sqrt

                for k_ in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                    k = constraint_state.jac_relevant_dofs[i_c, k_, i_b]
                    Lkk = constraint_state.nt_H[i_b, k, k]
                    tmp = Lkk**2 + sign * constraint_state.nt_vec[k, i_b] ** 2
                    if tmp < EPS:
                        is_degenerated = True
                        break
                    r = ti.sqrt(tmp)
                    c = r / Lkk
                    cinv = 1 / c
                    s = constraint_state.nt_vec[k, i_b] / Lkk
                    constraint_state.nt_H[i_b, k, k] = r
                    for i_ in range(k_):
                        i = constraint_state.jac_relevant_dofs[i_c, i_, i_b]  # i is strictly > k
                        constraint_state.nt_H[i_b, i, k] = (
                            constraint_state.nt_H[i_b, i, k] + s * constraint_state.nt_vec[i, i_b] * sign
                        ) * cinv

                    for i_ in range(k_):
                        i = constraint_state.jac_relevant_dofs[i_c, i_, i_b]  # i is strictly > k
                        constraint_state.nt_vec[i, i_b] = (
                            constraint_state.nt_vec[i, i_b] * c - s * constraint_state.nt_H[i_b, i, k]
                        )
            else:
                for i_d in range(n_dofs):
                    constraint_state.nt_vec[i_d, i_b] = constraint_state.jac[i_c, i_d, i_b] * efc_D_sqrt

                for k in range(n_dofs):
                    if ti.abs(constraint_state.nt_vec[k, i_b]) > EPS:
                        Lkk = constraint_state.nt_H[i_b, k, k]
                        tmp = Lkk**2 + sign * constraint_state.nt_vec[k, i_b] ** 2
                        if tmp < EPS:
                            is_degenerated = True
                            break
                        r = ti.sqrt(tmp)
                        c = r / Lkk
                        cinv = 1 / c
                        s = constraint_state.nt_vec[k, i_b] / Lkk
                        constraint_state.nt_H[i_b, k, k] = r
                        for i in range(k + 1, n_dofs):
                            constraint_state.nt_H[i_b, i, k] = (
                                constraint_state.nt_H[i_b, i, k] + s * constraint_state.nt_vec[i, i_b] * sign
                            ) * cinv

                        for i in range(k + 1, n_dofs):
                            constraint_state.nt_vec[i, i_b] = (
                                constraint_state.nt_vec[i, i_b] * c - s * constraint_state.nt_H[i_b, i, k]
                            )

    if is_degenerated:
        func_nt_hessian_direct(
            i_b,
            entities_info=entities_info,
            constraint_state=constraint_state,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )


@ti.func
def func_nt_hessian_direct(
    i_b,
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    EPS = rigid_global_info.EPS[None]

    n_dofs = constraint_state.nt_H.shape[1]
    n_entities = entities_info.n_links.shape[0]

    # H = M + J'*D*J
    for i_d1 in range(n_dofs):
        for i_d2 in range(i_d1 + 1):
            constraint_state.nt_H[i_b, i_d1, i_d2] = gs.ti_float(0.0)

    if ti.static(static_rigid_sim_config.sparse_solve):
        for i_c in range(constraint_state.n_constraints[i_b]):
            jac_n_relevant_dofs = constraint_state.jac_n_relevant_dofs[i_c, i_b]
            for i_d1_ in range(jac_n_relevant_dofs):
                i_d1 = constraint_state.jac_relevant_dofs[i_c, i_d1_, i_b]
                if ti.abs(constraint_state.jac[i_c, i_d1, i_b]) > EPS:
                    for i_d2_ in range(i_d1_, jac_n_relevant_dofs):
                        i_d2 = constraint_state.jac_relevant_dofs[i_c, i_d2_, i_b]  # i_d2 is strictly <= i_d1
                        constraint_state.nt_H[i_b, i_d1, i_d2] = (
                            constraint_state.nt_H[i_b, i_d1, i_d2]
                            + constraint_state.jac[i_c, i_d2, i_b]
                            * constraint_state.jac[i_c, i_d1, i_b]
                            * constraint_state.efc_D[i_c, i_b]
                            * constraint_state.active[i_c, i_b]
                        )
    else:
        for i_d1, i_c in ti.ndrange(n_dofs, constraint_state.n_constraints[i_b]):
            if ti.abs(constraint_state.jac[i_c, i_d1, i_b]) > EPS:
                for i_d2 in range(i_d1 + 1):
                    constraint_state.nt_H[i_b, i_d1, i_d2] = (
                        constraint_state.nt_H[i_b, i_d1, i_d2]
                        + constraint_state.jac[i_c, i_d2, i_b]
                        * constraint_state.jac[i_c, i_d1, i_b]
                        * constraint_state.efc_D[i_c, i_b]
                        * constraint_state.active[i_c, i_b]
                    )

    for i_e in range(n_entities):
        for i_d1 in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
            for i_d2 in range(entities_info.dof_start[i_e], i_d1 + 1):
                constraint_state.nt_H[i_b, i_d1, i_d2] = (
                    constraint_state.nt_H[i_b, i_d1, i_d2] + rigid_global_info.mass_mat[i_d1, i_d2, i_b]
                )

    func_nt_chol_factor(i_b, constraint_state, rigid_global_info)


@ti.func
def func_nt_chol_factor(
    i_b,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
):
    EPS = rigid_global_info.EPS[None]

    n_dofs = constraint_state.nt_H.shape[1]
    for i_d in range(n_dofs):
        tmp = constraint_state.nt_H[i_b, i_d, i_d]
        for j_d in range(i_d):
            tmp = tmp - constraint_state.nt_H[i_b, i_d, j_d] ** 2
        constraint_state.nt_H[i_b, i_d, i_d] = ti.sqrt(ti.max(tmp, EPS))

        tmp = 1.0 / constraint_state.nt_H[i_b, i_d, i_d]
        for j_d in range(i_d + 1, n_dofs):
            dot = gs.ti_float(0.0)
            for k_d in range(i_d):
                dot = dot + constraint_state.nt_H[i_b, j_d, k_d] * constraint_state.nt_H[i_b, i_d, k_d]
            constraint_state.nt_H[i_b, j_d, i_d] = (constraint_state.nt_H[i_b, j_d, i_d] - dot) * tmp


@ti.func
def func_nt_chol_solve(
    i_b,
    constraint_state: array_class.ConstraintState,
):
    n_dofs = constraint_state.Mgrad.shape[0]

    for i_d in range(n_dofs):
        curr_out = constraint_state.grad[i_d, i_b]
        for j_d in range(i_d):
            curr_out = curr_out - constraint_state.nt_H[i_b, i_d, j_d] * constraint_state.Mgrad[j_d, i_b]
        constraint_state.Mgrad[i_d, i_b] = curr_out / constraint_state.nt_H[i_b, i_d, i_d]

    for i_d_ in range(n_dofs):
        i_d = n_dofs - 1 - i_d_
        curr_out = constraint_state.Mgrad[i_d, i_b]
        for j_d in range(i_d + 1, n_dofs):
            curr_out = curr_out - constraint_state.nt_H[i_b, j_d, i_d] * constraint_state.Mgrad[j_d, i_b]
        constraint_state.Mgrad[i_d, i_b] = curr_out / constraint_state.nt_H[i_b, i_d, i_d]


@ti.kernel(fastcache=gs.use_fastcache)
def func_solve(
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    _B = constraint_state.grad.shape[1]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    for i_b in range(_B):
        # t0_start = ti.clock_counter()
        if constraint_state.n_constraints[i_b] > 0:
            for _ in range(rigid_global_info.iterations[None]):
                func_solve_iter(
                    i_b,
                    entities_info=entities_info,
                    dofs_state=dofs_state,
                    rigid_global_info=rigid_global_info,
                    constraint_state=constraint_state,
                    static_rigid_sim_config=static_rigid_sim_config,
                )
                if not constraint_state.improved[i_b]:
                    break
        else:
            constraint_state.improved[i_b] = False
        # t0_end = ti.clock_counter()
        # constraint_state.timers[0, i_b_] = t0_end - t0_start


@ti.func
def func_ls_init(
    i_b,
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    n_dofs = constraint_state.search.shape[0]
    n_entities = entities_info.dof_start.shape[0]
    # mv and jv
    for i_e in range(n_entities):
        for i_d1 in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
            mv = gs.ti_float(0.0)
            for i_d2 in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                mv = mv + rigid_global_info.mass_mat[i_d1, i_d2, i_b] * constraint_state.search[i_d2, i_b]
            constraint_state.mv[i_d1, i_b] = mv

    for i_c in range(constraint_state.n_constraints[i_b]):
        jv = gs.ti_float(0.0)
        if ti.static(static_rigid_sim_config.sparse_solve):
            for i_d_ in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                i_d = constraint_state.jac_relevant_dofs[i_c, i_d_, i_b]
                jv = jv + constraint_state.jac[i_c, i_d, i_b] * constraint_state.search[i_d, i_b]
        else:
            for i_d in range(n_dofs):
                jv = jv + constraint_state.jac[i_c, i_d, i_b] * constraint_state.search[i_d, i_b]
        constraint_state.jv[i_c, i_b] = jv

    # quad and quad_gauss
    quad_gauss_1 = gs.ti_float(0.0)
    quad_gauss_2 = gs.ti_float(0.0)
    for i_d in range(n_dofs):
        quad_gauss_1 = quad_gauss_1 + (
            constraint_state.search[i_d, i_b] * constraint_state.Ma[i_d, i_b]
            - constraint_state.search[i_d, i_b] * dofs_state.force[i_d, i_b]
        )
        quad_gauss_2 = quad_gauss_2 + 0.5 * constraint_state.search[i_d, i_b] * constraint_state.mv[i_d, i_b]
    for _i0 in range(1):
        constraint_state.quad_gauss[_i0 + 0, i_b] = constraint_state.gauss[i_b]
        constraint_state.quad_gauss[_i0 + 1, i_b] = quad_gauss_1
        constraint_state.quad_gauss[_i0 + 2, i_b] = quad_gauss_2

        for i_c in range(constraint_state.n_constraints[i_b]):
            constraint_state.quad[i_c, _i0 + 0, i_b] = constraint_state.efc_D[i_c, i_b] * (
                0.5 * constraint_state.Jaref[i_c, i_b] * constraint_state.Jaref[i_c, i_b]
            )
            constraint_state.quad[i_c, _i0 + 1, i_b] = constraint_state.efc_D[i_c, i_b] * (
                constraint_state.jv[i_c, i_b] * constraint_state.Jaref[i_c, i_b]
            )
            constraint_state.quad[i_c, _i0 + 2, i_b] = constraint_state.efc_D[i_c, i_b] * (
                0.5 * constraint_state.jv[i_c, i_b] * constraint_state.jv[i_c, i_b]
            )


@ti.func
def func_ls_point_fn(
    i_b,
    alpha,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
):
    ne = constraint_state.n_constraints_equality[i_b]
    nef = ne + constraint_state.n_constraints_frictionloss[i_b]

    tmp_quad_total_0, tmp_quad_total_1, tmp_quad_total_2 = gs.ti_float(0.0), gs.ti_float(0.0), gs.ti_float(0.0)
    tmp_quad_total_0 = constraint_state.quad_gauss[0, i_b]
    tmp_quad_total_1 = constraint_state.quad_gauss[1, i_b]
    tmp_quad_total_2 = constraint_state.quad_gauss[2, i_b]
    for i_c in range(constraint_state.n_constraints[i_b]):
        x = constraint_state.Jaref[i_c, i_b] + alpha * constraint_state.jv[i_c, i_b]
        qf_0 = constraint_state.quad[i_c, 0, i_b]
        qf_1 = constraint_state.quad[i_c, 1, i_b]
        qf_2 = constraint_state.quad[i_c, 2, i_b]

        active = gs.ti_bool(True)  # Equality constraints
        if ne <= i_c and i_c < nef:  # Friction constraints
            f = constraint_state.efc_frictionloss[i_c, i_b]
            r = constraint_state.diag[i_c, i_b]
            rf = r * f
            linear_neg = x <= -rf
            linear_pos = x >= rf

            if linear_neg or linear_pos:
                qf_0 = linear_neg * f * (-0.5 * rf - constraint_state.Jaref[i_c, i_b]) + linear_pos * f * (
                    -0.5 * rf + constraint_state.Jaref[i_c, i_b]
                )
                qf_1 = linear_neg * (-f * constraint_state.jv[i_c, i_b]) + linear_pos * (
                    f * constraint_state.jv[i_c, i_b]
                )
                qf_2 = 0.0
        elif nef <= i_c:  # Contact constraints
            active = x < 0

        tmp_quad_total_0 = tmp_quad_total_0 + qf_0 * active
        tmp_quad_total_1 = tmp_quad_total_1 + qf_1 * active
        tmp_quad_total_2 = tmp_quad_total_2 + qf_2 * active

    cost = alpha * alpha * tmp_quad_total_2 + alpha * tmp_quad_total_1 + tmp_quad_total_0

    deriv_0 = 2 * alpha * tmp_quad_total_2 + tmp_quad_total_1
    deriv_1 = 2 * tmp_quad_total_2
    if deriv_1 <= 0.0:
        deriv_1 = rigid_global_info.EPS[None]

    constraint_state.ls_it[i_b] = constraint_state.ls_it[i_b] + 1

    return alpha, cost, deriv_0, deriv_1


@ti.func
def func_no_linesearch(i_b, constraint_state: array_class.ConstraintState):
    func_ls_init(i_b)
    n_dofs = constraint_state.search.shape[0]

    constraint_state.improved[i_b] = True
    for i_d in range(n_dofs):
        constraint_state.qacc[i_d, i_b] = constraint_state.qacc[i_d, i_b] + constraint_state.search[i_d, i_b]
        constraint_state.Ma[i_d, i_b] = constraint_state.Ma[i_d, i_b] + constraint_state.mv[i_d, i_b]
    for i_c in range(constraint_state.n_constraints[i_b]):
        constraint_state.Jaref[i_c, i_b] = constraint_state.Jaref[i_c, i_b] + constraint_state.jv[i_c, i_b]


@ti.func
def func_linesearch(
    i_b,
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
):
    n_dofs = constraint_state.search.shape[0]
    ## use adaptive linesearch tolerance
    snorm = gs.ti_float(0.0)
    for jd in range(n_dofs):
        snorm = snorm + constraint_state.search[jd, i_b] ** 2
    snorm = ti.sqrt(snorm)
    scale = rigid_global_info.meaninertia[i_b] * ti.max(1, n_dofs)
    gtol = rigid_global_info.tolerance[None] * rigid_global_info.ls_tolerance[None] * snorm * scale
    constraint_state.gtol[i_b] = gtol

    constraint_state.ls_it[i_b] = 0
    constraint_state.ls_result[i_b] = 0

    res_alpha = gs.ti_float(0.0)
    done = False

    if snorm < rigid_global_info.EPS[None]:
        constraint_state.ls_result[i_b] = 1
        res_alpha = 0.0
    else:
        func_ls_init(
            i_b,
            entities_info=entities_info,
            dofs_state=dofs_state,
            constraint_state=constraint_state,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )

        p0_alpha, p0_cost, p0_deriv_0, p0_deriv_1 = func_ls_point_fn(
            i_b, gs.ti_float(0.0), constraint_state, rigid_global_info
        )
        p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = func_ls_point_fn(
            i_b, p0_alpha - p0_deriv_0 / p0_deriv_1, constraint_state, rigid_global_info
        )

        if p0_cost < p1_cost:
            p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = p0_alpha, p0_cost, p0_deriv_0, p0_deriv_1

        if ti.abs(p1_deriv_0) < gtol:
            if ti.abs(p1_alpha) < rigid_global_info.EPS[None]:
                constraint_state.ls_result[i_b] = 2
            else:
                constraint_state.ls_result[i_b] = 0
            res_alpha = p1_alpha
        else:
            direction = (p1_deriv_0 < 0) * 2 - 1
            p2update = 0
            p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1 = p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1
            while (
                p1_deriv_0 * direction <= -gtol and constraint_state.ls_it[i_b] < rigid_global_info.ls_iterations[None]
            ):
                p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1 = p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1
                p2update = 1

                p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = func_ls_point_fn(
                    i_b, p1_alpha - p1_deriv_0 / p1_deriv_1, constraint_state, rigid_global_info
                )
                if ti.abs(p1_deriv_0) < gtol:
                    res_alpha = p1_alpha
                    done = True
                    break
            if not done:
                if constraint_state.ls_it[i_b] >= rigid_global_info.ls_iterations[None]:
                    constraint_state.ls_result[i_b] = 3
                    res_alpha = p1_alpha
                    done = True

                if not p2update and not done:
                    constraint_state.ls_result[i_b] = 6
                    res_alpha = p1_alpha
                    done = True

                if not done:
                    p2_next_alpha, p2_next_cost, p2_next_deriv_0, p2_next_deriv_1 = (
                        p1_alpha,
                        p1_cost,
                        p1_deriv_0,
                        p1_deriv_1,
                    )

                    p1_next_alpha, p1_next_cost, p1_next_deriv_0, p1_next_deriv_1 = func_ls_point_fn(
                        i_b, p1_alpha - p1_deriv_0 / p1_deriv_1, constraint_state, rigid_global_info
                    )

                    while constraint_state.ls_it[i_b] < rigid_global_info.ls_iterations[None]:
                        pmid_alpha, pmid_cost, pmid_deriv_0, pmid_deriv_1 = func_ls_point_fn(
                            i_b, (p1_alpha + p2_alpha) * 0.5, constraint_state, rigid_global_info
                        )

                        i = 0
                        (
                            constraint_state.candidates[4 * i + 0, i_b],
                            constraint_state.candidates[4 * i + 1, i_b],
                            constraint_state.candidates[4 * i + 2, i_b],
                            constraint_state.candidates[4 * i + 3, i_b],
                        ) = (p1_next_alpha, p1_next_cost, p1_next_deriv_0, p1_next_deriv_1)
                        i = 1
                        (
                            constraint_state.candidates[4 * i + 0, i_b],
                            constraint_state.candidates[4 * i + 1, i_b],
                            constraint_state.candidates[4 * i + 2, i_b],
                            constraint_state.candidates[4 * i + 3, i_b],
                        ) = (p2_next_alpha, p2_next_cost, p2_next_deriv_0, p2_next_deriv_1)
                        i = 2
                        (
                            constraint_state.candidates[4 * i + 0, i_b],
                            constraint_state.candidates[4 * i + 1, i_b],
                            constraint_state.candidates[4 * i + 2, i_b],
                            constraint_state.candidates[4 * i + 3, i_b],
                        ) = (pmid_alpha, pmid_cost, pmid_deriv_0, pmid_deriv_1)

                        best_i = -1
                        best_cost = gs.ti_float(0.0)
                        for ii in range(3):
                            if ti.abs(constraint_state.candidates[4 * ii + 2, i_b]) < gtol and (
                                best_i < 0 or constraint_state.candidates[4 * ii + 1, i_b] < best_cost
                            ):
                                best_cost = constraint_state.candidates[4 * ii + 1, i_b]
                                best_i = ii
                        if best_i >= 0:
                            res_alpha = constraint_state.candidates[4 * best_i + 0, i_b]
                            done = True
                        else:
                            (
                                b1,
                                p1_alpha,
                                p1_cost,
                                p1_deriv_0,
                                p1_deriv_1,
                                p1_next_alpha,
                                p1_next_cost,
                                p1_next_deriv_0,
                                p1_next_deriv_1,
                            ) = update_bracket(
                                p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1, i_b, constraint_state, rigid_global_info
                            )
                            (
                                b2,
                                p2_alpha,
                                p2_cost,
                                p2_deriv_0,
                                p2_deriv_1,
                                p2_next_alpha,
                                p2_next_cost,
                                p2_next_deriv_0,
                                p2_next_deriv_1,
                            ) = update_bracket(
                                p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1, i_b, constraint_state, rigid_global_info
                            )

                            if b1 == 0 and b2 == 0:
                                if pmid_cost < p0_cost:
                                    constraint_state.ls_result[i_b] = 0
                                else:
                                    constraint_state.ls_result[i_b] = 7
                                res_alpha = pmid_alpha
                                done = True

                    if not done:
                        if p1_cost <= p2_cost and p1_cost < p0_cost:
                            constraint_state.ls_result[i_b] = 4
                            res_alpha = p1_alpha
                        elif p2_cost <= p1_cost and p2_cost < p1_cost:
                            constraint_state.ls_result[i_b] = 4
                            res_alpha = p2_alpha
                        else:
                            constraint_state.ls_result[i_b] = 5
                            res_alpha = 0.0
    return res_alpha


@ti.func
def update_bracket(
    p_alpha,
    p_cost,
    p_deriv_0,
    p_deriv_1,
    i_b,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
):
    flag = 0

    for i in range(3):
        if (
            p_deriv_0 < 0
            and constraint_state.candidates[4 * i + 2, i_b] < 0
            and p_deriv_0 < constraint_state.candidates[4 * i + 2, i_b]
        ):
            p_alpha, p_cost, p_deriv_0, p_deriv_1 = (
                constraint_state.candidates[4 * i + 0, i_b],
                constraint_state.candidates[4 * i + 1, i_b],
                constraint_state.candidates[4 * i + 2, i_b],
                constraint_state.candidates[4 * i + 3, i_b],
            )

            flag = 1

        elif (
            p_deriv_0 > 0
            and constraint_state.candidates[4 * i + 2, i_b] > 0
            and p_deriv_0 > constraint_state.candidates[4 * i + 2, i_b]
        ):
            p_alpha, p_cost, p_deriv_0, p_deriv_1 = (
                constraint_state.candidates[4 * i + 0, i_b],
                constraint_state.candidates[4 * i + 1, i_b],
                constraint_state.candidates[4 * i + 2, i_b],
                constraint_state.candidates[4 * i + 3, i_b],
            )
            flag = 2

    p_next_alpha, p_next_cost, p_next_deriv_0, p_next_deriv_1 = p_alpha, p_cost, p_deriv_0, p_deriv_1

    if flag > 0:
        p_next_alpha, p_next_cost, p_next_deriv_0, p_next_deriv_1 = func_ls_point_fn(
            i_b, p_alpha - p_deriv_0 / p_deriv_1, constraint_state, rigid_global_info
        )
    return flag, p_alpha, p_cost, p_deriv_0, p_deriv_1, p_next_alpha, p_next_cost, p_next_deriv_0, p_next_deriv_1


@ti.func
def func_solve_iter(
    i_b,
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
):
    n_dofs = constraint_state.qacc.shape[0]
    alpha = func_linesearch(
        i_b,
        entities_info=entities_info,
        dofs_state=dofs_state,
        rigid_global_info=rigid_global_info,
        constraint_state=constraint_state,
        static_rigid_sim_config=static_rigid_sim_config,
    )

    if ti.abs(alpha) < rigid_global_info.EPS[None]:
        constraint_state.improved[i_b] = False
    else:
        for i_d in range(n_dofs):
            constraint_state.qacc[i_d, i_b] = (
                constraint_state.qacc[i_d, i_b] + constraint_state.search[i_d, i_b] * alpha
            )
            constraint_state.Ma[i_d, i_b] = constraint_state.Ma[i_d, i_b] + constraint_state.mv[i_d, i_b] * alpha

        for i_c in range(constraint_state.n_constraints[i_b]):
            constraint_state.Jaref[i_c, i_b] = constraint_state.Jaref[i_c, i_b] + constraint_state.jv[i_c, i_b] * alpha

        if ti.static(static_rigid_sim_config.solver_type == gs.constraint_solver.CG):
            for i_d in range(n_dofs):
                constraint_state.cg_prev_grad[i_d, i_b] = constraint_state.grad[i_d, i_b]
                constraint_state.cg_prev_Mgrad[i_d, i_b] = constraint_state.Mgrad[i_d, i_b]

        func_update_constraint(
            i_b,
            qacc=constraint_state.qacc,
            Ma=constraint_state.Ma,
            cost=constraint_state.cost,
            dofs_state=dofs_state,
            constraint_state=constraint_state,
            static_rigid_sim_config=static_rigid_sim_config,
        )

        if ti.static(static_rigid_sim_config.solver_type == gs.constraint_solver.Newton):
            func_nt_hessian_incremental(
                i_b,
                entities_info=entities_info,
                constraint_state=constraint_state,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
            )

        func_update_gradient(
            i_b,
            dofs_state=dofs_state,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            constraint_state=constraint_state,
            static_rigid_sim_config=static_rigid_sim_config,
        )

        tol_scaled = (rigid_global_info.meaninertia[i_b] * ti.max(1, n_dofs)) * rigid_global_info.tolerance[None]
        improvement = constraint_state.prev_cost[i_b] - constraint_state.cost[i_b]
        gradient = gs.ti_float(0.0)
        for i_d in range(n_dofs):
            gradient = gradient + constraint_state.grad[i_d, i_b] * constraint_state.grad[i_d, i_b]
        gradient = ti.sqrt(gradient)
        if gradient < tol_scaled or improvement < tol_scaled:
            constraint_state.improved[i_b] = False
        else:
            constraint_state.improved[i_b] = True

            if ti.static(static_rigid_sim_config.solver_type == gs.constraint_solver.Newton):
                for i_d in range(n_dofs):
                    constraint_state.search[i_d, i_b] = -constraint_state.Mgrad[i_d, i_b]
            else:
                cg_beta = gs.ti_float(0.0)
                cg_pg_dot_pMg = gs.ti_float(0.0)

                for i_d in range(n_dofs):
                    cg_beta = cg_beta + constraint_state.grad[i_d, i_b] * (
                        constraint_state.Mgrad[i_d, i_b] - constraint_state.cg_prev_Mgrad[i_d, i_b]
                    )
                    cg_pg_dot_pMg = cg_pg_dot_pMg + (
                        constraint_state.cg_prev_Mgrad[i_d, i_b] * constraint_state.cg_prev_grad[i_d, i_b]
                    )
                cg_beta = ti.max(cg_beta / ti.max(rigid_global_info.EPS[None], cg_pg_dot_pMg), 0.0)

                constraint_state.cg_pg_dot_pMg[i_b] = cg_pg_dot_pMg
                constraint_state.cg_beta[i_b] = cg_beta

                for i_d in range(n_dofs):
                    constraint_state.search[i_d, i_b] = (
                        -constraint_state.Mgrad[i_d, i_b] + cg_beta * constraint_state.search[i_d, i_b]
                    )


@ti.func
def func_update_constraint(
    i_b,
    qacc: array_class.V_ANNOTATION,
    Ma: array_class.V_ANNOTATION,
    cost: array_class.V_ANNOTATION,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
):
    n_dofs = constraint_state.qfrc_constraint.shape[0]
    ne = constraint_state.n_constraints_equality[i_b]
    nef = ne + constraint_state.n_constraints_frictionloss[i_b]

    constraint_state.prev_cost[i_b] = cost[i_b]
    cost_i = gs.ti_float(0.0)
    gauss_i = gs.ti_float(0.0)

    # Beware 'active' does not refer to whether a constraint is active, but rather whether its quadratic cost is active
    for i_c in range(constraint_state.n_constraints[i_b]):
        if ti.static(static_rigid_sim_config.solver_type == gs.constraint_solver.Newton):
            constraint_state.prev_active[i_c, i_b] = constraint_state.active[i_c, i_b]
        constraint_state.active[i_c, i_b] = True

        floss_force = gs.ti_float(0.0)
        if ne <= i_c and i_c < nef:  # Friction constraints
            f = constraint_state.efc_frictionloss[i_c, i_b]
            r = constraint_state.diag[i_c, i_b]
            rf = r * f
            linear_neg = constraint_state.Jaref[i_c, i_b] <= -rf
            linear_pos = constraint_state.Jaref[i_c, i_b] >= rf
            constraint_state.active[i_c, i_b] = not (linear_neg or linear_pos)
            floss_force = linear_neg * f + linear_pos * -f
            floss_cost_local = linear_neg * f * (-0.5 * rf - constraint_state.Jaref[i_c, i_b])
            floss_cost_local = floss_cost_local + linear_pos * f * (-0.5 * rf + constraint_state.Jaref[i_c, i_b])
            cost_i = cost_i + floss_cost_local
        elif nef <= i_c:  # Contact constraints
            constraint_state.active[i_c, i_b] = constraint_state.Jaref[i_c, i_b] < 0

        constraint_state.efc_force[i_c, i_b] = floss_force + (
            -constraint_state.Jaref[i_c, i_b] * constraint_state.efc_D[i_c, i_b] * constraint_state.active[i_c, i_b]
        )

    if ti.static(static_rigid_sim_config.sparse_solve):
        for i_d in range(n_dofs):
            constraint_state.qfrc_constraint[i_d, i_b] = gs.ti_float(0.0)
        for i_c in range(constraint_state.n_constraints[i_b]):
            for i_d_ in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                i_d = constraint_state.jac_relevant_dofs[i_c, i_d_, i_b]
                constraint_state.qfrc_constraint[i_d, i_b] = (
                    constraint_state.qfrc_constraint[i_d, i_b]
                    + constraint_state.jac[i_c, i_d, i_b] * constraint_state.efc_force[i_c, i_b]
                )
    else:
        for i_d in range(n_dofs):
            qfrc_constraint = gs.ti_float(0.0)
            for i_c in range(constraint_state.n_constraints[i_b]):
                qfrc_constraint = (
                    qfrc_constraint + constraint_state.jac[i_c, i_d, i_b] * constraint_state.efc_force[i_c, i_b]
                )
            constraint_state.qfrc_constraint[i_d, i_b] = qfrc_constraint

    # (Mx - Mx') * (x - x')
    for i_d in range(n_dofs):
        v = 0.5 * (Ma[i_d, i_b] - dofs_state.force[i_d, i_b]) * (qacc[i_d, i_b] - dofs_state.acc_smooth[i_d, i_b])
        gauss_i = gauss_i + v
        cost_i = cost_i + v

    # D * (Jx - aref) ** 2
    for i_c in range(constraint_state.n_constraints[i_b]):
        cost_i = cost_i + 0.5 * (
            constraint_state.Jaref[i_c, i_b] ** 2 * constraint_state.efc_D[i_c, i_b] * constraint_state.active[i_c, i_b]
        )

    constraint_state.gauss[i_b] = gauss_i
    cost[i_b] = cost_i


@ti.func
def func_update_gradient(
    i_b,
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
):
    n_dofs = constraint_state.grad.shape[0]

    for i_d in range(n_dofs):
        constraint_state.grad[i_d, i_b] = (
            constraint_state.Ma[i_d, i_b] - dofs_state.force[i_d, i_b] - constraint_state.qfrc_constraint[i_d, i_b]
        )

    if ti.static(static_rigid_sim_config.solver_type == gs.constraint_solver.CG):
        rigid_solver.func_solve_mass_batch(
            i_b,
            constraint_state.grad,
            constraint_state.Mgrad,
            array_class.PLACEHOLDER,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
            is_backward=False,
        )
    if ti.static(static_rigid_sim_config.solver_type == gs.constraint_solver.Newton):
        func_nt_chol_solve(i_b, constraint_state=constraint_state)


@ti.func
def initialize_Jaref(
    qacc: array_class.V_ANNOTATION,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
):
    _B = constraint_state.jac.shape[2]
    n_dofs = constraint_state.jac.shape[1]
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        for i_c in range(constraint_state.n_constraints[i_b]):
            Jaref = -constraint_state.aref[i_c, i_b]
            if ti.static(static_rigid_sim_config.sparse_solve):
                for i_d_ in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                    i_d = constraint_state.jac_relevant_dofs[i_c, i_d_, i_b]
                    Jaref = Jaref + constraint_state.jac[i_c, i_d, i_b] * qacc[i_d, i_b]
            else:
                for i_d in range(n_dofs):
                    Jaref = Jaref + constraint_state.jac[i_c, i_d, i_b] * qacc[i_d, i_b]
            constraint_state.Jaref[i_c, i_b] = Jaref


@ti.func
def initialize_Ma(
    Ma: array_class.V_ANNOTATION,
    qacc: array_class.V_ANNOTATION,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    _B = rigid_global_info.mass_mat.shape[2]
    n_dofs = qacc.shape[0]

    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d1, i_b in ti.ndrange(n_dofs, _B):
        I_d1 = [i_d1, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d1
        i_e = dofs_info.entity_idx[I_d1]
        Ma_ = gs.ti_float(0.0)
        for i_d2 in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
            Ma_ = Ma_ + rigid_global_info.mass_mat[i_d1, i_d2, i_b] * qacc[i_d2, i_b]
        Ma[i_d1, i_b] = Ma_


@ti.kernel(fastcache=gs.use_fastcache)
def func_init_solver(
    dofs_info: array_class.DofsInfo,
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    EPS = rigid_global_info.EPS[None]

    _B = dofs_state.acc_smooth.shape[1]
    n_dofs = dofs_state.acc_smooth.shape[0]

    if ti.static(static_rigid_sim_config.enable_mujoco_compatibility):
        # Compute cost for warmstart state (i.e. acceleration at previous timestep)
        initialize_Ma(
            Ma=constraint_state.Ma_ws,
            qacc=constraint_state.qacc_ws,
            dofs_info=dofs_info,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )

        initialize_Jaref(
            qacc=constraint_state.qacc_ws,
            constraint_state=constraint_state,
            static_rigid_sim_config=static_rigid_sim_config,
        )
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(_B):
            func_update_constraint(
                i_b,
                qacc=constraint_state.qacc_ws,
                Ma=constraint_state.Ma_ws,
                cost=constraint_state.cost_ws,
                dofs_state=dofs_state,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )

        # Compute cost for current state (assuming constraint-free acceleration)
        initialize_Ma(
            Ma=constraint_state.Ma,
            qacc=dofs_state.acc_smooth,
            dofs_info=dofs_info,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )

        initialize_Jaref(
            qacc=dofs_state.acc_smooth,
            constraint_state=constraint_state,
            static_rigid_sim_config=static_rigid_sim_config,
        )
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(_B):
            func_update_constraint(
                i_b,
                qacc=dofs_state.acc_smooth,
                Ma=constraint_state.Ma,
                cost=constraint_state.cost,
                dofs_state=dofs_state,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )

        # Pick the best starting point between current state and warmstart
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_d, i_b in ti.ndrange(n_dofs, _B):
            if constraint_state.cost_ws[i_b] < constraint_state.cost[i_b]:
                constraint_state.qacc[i_d, i_b] = constraint_state.qacc_ws[i_d, i_b]
                constraint_state.Ma[i_d, i_b] = constraint_state.Ma_ws[i_d, i_b]
            else:
                constraint_state.qacc[i_d, i_b] = dofs_state.acc_smooth[i_d, i_b]
    else:
        # Always initialize from warmstart
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_d, i_b in ti.ndrange(n_dofs, _B):
            if constraint_state.n_constraints[i_b] > 0 and constraint_state.is_warmstart[i_b]:
                constraint_state.qacc[i_d, i_b] = constraint_state.qacc_ws[i_d, i_b]
            else:
                constraint_state.qacc[i_d, i_b] = dofs_state.acc_smooth[i_d, i_b]

        initialize_Ma(
            Ma=constraint_state.Ma,
            qacc=constraint_state.qacc,
            dofs_info=dofs_info,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )

    # Initialize solver accordingly
    initialize_Jaref(
        qacc=constraint_state.qacc,
        constraint_state=constraint_state,
        static_rigid_sim_config=static_rigid_sim_config,
    )
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        func_update_constraint(
            i_b,
            qacc=constraint_state.qacc,
            Ma=constraint_state.Ma,
            cost=constraint_state.cost,
            dofs_state=dofs_state,
            constraint_state=constraint_state,
            static_rigid_sim_config=static_rigid_sim_config,
        )

    if ti.static(
        static_rigid_sim_config.solver_type != gs.constraint_solver.Newton
        or static_rigid_sim_config.sparse_solve
        or static_rigid_sim_config.backend == gs.cpu
    ):
        if ti.static(static_rigid_sim_config.solver_type == gs.constraint_solver.Newton):
            ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
            for i_b in range(_B):
                func_nt_hessian_direct(
                    i_b,
                    entities_info=entities_info,
                    rigid_global_info=rigid_global_info,
                    constraint_state=constraint_state,
                    static_rigid_sim_config=static_rigid_sim_config,
                )

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(_B):
            func_update_gradient(
                i_b,
                dofs_state=dofs_state,
                entities_info=entities_info,
                rigid_global_info=rigid_global_info,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )
    else:
        # Performance is optimal for BLOCK_DIM = MAX_DOFS_PER_BLOCK = 64.
        BLOCK_DIM = ti.static(64)
        MAX_DOFS_PER_BLOCK = ti.static(64)
        MAX_CONSTRAINTS_PER_BLOCK = ti.static(32)

        n_dofs_2 = n_dofs**2
        n_lower_tri = n_dofs * (n_dofs + 1) // 2

        # FIXME: Adding `serialize=False` is causing sync failing for some reason...
        ti.loop_config(block_dim=BLOCK_DIM)
        for i in range(_B * BLOCK_DIM):
            tid = i % BLOCK_DIM
            i_b = i // BLOCK_DIM
            if i_b >= _B:
                continue

            jac_row = ti.simt.block.SharedArray((MAX_CONSTRAINTS_PER_BLOCK, MAX_DOFS_PER_BLOCK), gs.ti_float)
            jac_col = ti.simt.block.SharedArray((MAX_CONSTRAINTS_PER_BLOCK, MAX_DOFS_PER_BLOCK), gs.ti_float)
            efc = ti.simt.block.SharedArray((MAX_CONSTRAINTS_PER_BLOCK,), gs.ti_float)

            i_c_start = 0
            n_c = constraint_state.n_constraints[i_b]
            while i_c_start < n_c:
                i_c_ = tid
                n_conts_tile = ti.min(MAX_CONSTRAINTS_PER_BLOCK, n_c - i_c_start)
                while i_c_ < n_conts_tile:
                    efc[i_c_] = (
                        constraint_state.efc_D[i_c_start + i_c_, i_b] * constraint_state.active[i_c_start + i_c_, i_b]
                    )
                    i_c_ = i_c_ + BLOCK_DIM

                i_d1_start = 0
                while i_d1_start < n_dofs:
                    n_dofs_tile_row = ti.min(MAX_DOFS_PER_BLOCK, n_dofs - i_d1_start)

                    i_c_ = tid
                    while i_c_ < n_conts_tile:
                        for i_d_ in range(n_dofs_tile_row):
                            jac_row[i_c_, i_d_] = constraint_state.jac[i_c_start + i_c_, i_d1_start + i_d_, i_b]
                        i_c_ = i_c_ + BLOCK_DIM
                    ti.simt.block.sync()

                    i_d2_start = 0
                    while i_d2_start <= i_d1_start:
                        n_dofs_tile_col = ti.min(MAX_DOFS_PER_BLOCK, n_dofs - i_d2_start)
                        is_diag_tile = i_d1_start == i_d2_start

                        if not is_diag_tile:
                            i_c_ = tid
                            while i_c_ < n_conts_tile:
                                for i_d_ in range(n_dofs_tile_col):
                                    jac_col[i_c_, i_d_] = constraint_state.jac[i_c_start + i_c_, i_d2_start + i_d_, i_b]
                                i_c_ = i_c_ + BLOCK_DIM
                            ti.simt.block.sync()

                        pid = tid
                        numel = n_dofs_tile_row * n_dofs_tile_col
                        while pid < numel:
                            i_d1_ = pid // n_dofs_tile_col
                            i_d2_ = pid % n_dofs_tile_col
                            i_d1 = i_d1_ + i_d1_start
                            i_d2 = i_d2_ + i_d2_start
                            if i_d1 >= i_d2:
                                coef = gs.ti_float(0.0)
                                if i_c_start == 0:
                                    coef = rigid_global_info.mass_mat[i_d1, i_d2, i_b]
                                if is_diag_tile:
                                    for j_c_ in range(n_conts_tile):
                                        coef = coef + jac_row[j_c_, i_d1_] * jac_row[j_c_, i_d2_] * efc[j_c_]
                                else:
                                    for j_c_ in range(n_conts_tile):
                                        coef = coef + jac_row[j_c_, i_d1_] * jac_col[j_c_, i_d2_] * efc[j_c_]
                                if i_c_start == 0:
                                    constraint_state.nt_H[i_b, i_d1, i_d2] = coef
                                else:
                                    constraint_state.nt_H[i_b, i_d1, i_d2] = (
                                        constraint_state.nt_H[i_b, i_d1, i_d2] + coef
                                    )
                            pid = pid + BLOCK_DIM
                        ti.simt.block.sync()

                        i_d2_start = i_d2_start + MAX_DOFS_PER_BLOCK
                    i_d1_start = i_d1_start + MAX_DOFS_PER_BLOCK
                i_c_start = i_c_start + MAX_CONSTRAINTS_PER_BLOCK

            if n_c == 0:
                i_pair = tid
                while i_pair < n_lower_tri:
                    i_d1 = ti.cast(ti.floor((-1.0 + ti.sqrt(1.0 + 8.0 * i_pair)) / 2.0), gs.ti_int)
                    i_d2 = i_pair - i_d1 * (i_d1 + 1) // 2
                    constraint_state.nt_H[i_b, i_d1, i_d2] = rigid_global_info.mass_mat[i_d1, i_d2, i_b]
                    i_pair = i_pair + BLOCK_DIM

        if ti.static(static_rigid_sim_config.enable_tiled_cholesky_hessian):
            BLOCK_DIM = ti.static(64)
            MAX_DOFS = ti.static(static_rigid_sim_config.tiled_n_dofs)
            ENABLE_WARP_REDUCTION = ti.static(static_rigid_sim_config.backend == gs.cuda and gs.ti_float == ti.f32)
            WARP_SIZE = ti.static(32)
            NUM_WARPS = ti.static(BLOCK_DIM // WARP_SIZE)

            ti.loop_config(block_dim=BLOCK_DIM)
            for i in range(_B * BLOCK_DIM):
                tid = i % BLOCK_DIM
                i_b = i // BLOCK_DIM
                if i_b >= _B:
                    continue

                # Padding +1 to avoid memory bank conflicts that would cause access serialization
                H = ti.simt.block.SharedArray((MAX_DOFS, MAX_DOFS + 1), gs.ti_float)

                i_pair = tid
                while i_pair < n_lower_tri:
                    i_d1 = ti.cast((ti.sqrt(8 * i_pair + 1) - 1) // 2, ti.i32)
                    i_d2 = i_pair - i_d1 * (i_d1 + 1) // 2
                    H[i_d1, i_d2] = constraint_state.nt_H[i_b, i_d1, i_d2]
                    i_pair = i_pair + BLOCK_DIM
                ti.simt.block.sync()

                for i_d in range(n_dofs):
                    if tid == 0:
                        tmp = H[i_d, i_d]
                        for j_d in range(i_d):
                            tmp = tmp - H[i_d, j_d] ** 2
                        H[i_d, i_d] = ti.sqrt(ti.max(tmp, EPS))
                    ti.simt.block.sync()

                    inv_diag = 1.0 / H[i_d, i_d]
                    j_d = i_d + 1 + tid
                    while j_d < n_dofs:
                        dot = gs.ti_float(0.0)
                        for k_d in range(i_d):
                            dot = dot + H[j_d, k_d] * H[i_d, k_d]
                        H[j_d, i_d] = (H[j_d, i_d] - dot) * inv_diag
                        j_d = j_d + BLOCK_DIM
                    ti.simt.block.sync()

                i_pair = tid
                while i_pair < n_lower_tri:
                    i_d1 = ti.cast((ti.sqrt(8 * i_pair + 1) - 1) // 2, ti.i32)
                    i_d2 = i_pair - i_d1 * (i_d1 + 1) // 2
                    constraint_state.nt_H[i_b, i_d1, i_d2] = H[i_d1, i_d2]
                    i_pair = i_pair + BLOCK_DIM

            ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
            for i_d, i_b in ti.ndrange(n_dofs, _B):
                constraint_state.grad[i_d, i_b] = (
                    constraint_state.Ma[i_d, i_b]
                    - dofs_state.force[i_d, i_b]
                    - constraint_state.qfrc_constraint[i_d, i_b]
                )
                constraint_state.Mgrad[i_d, i_b] = constraint_state.grad[i_d, i_b]

            ti.loop_config(block_dim=BLOCK_DIM)
            for i in range(_B * BLOCK_DIM):
                tid = i % BLOCK_DIM
                i_b = i // BLOCK_DIM
                warp_id = tid // WARP_SIZE
                lane_id = tid % WARP_SIZE
                if i_b >= _B:
                    continue

                H = ti.simt.block.SharedArray((MAX_DOFS, MAX_DOFS + 1), gs.ti_float)
                v = ti.simt.block.SharedArray((MAX_DOFS,), gs.ti_float)
                partial = ti.simt.block.SharedArray(
                    (NUM_WARPS if ti.static(ENABLE_WARP_REDUCTION) else BLOCK_DIM,), gs.ti_float
                )

                i_flat = tid
                while i_flat < n_dofs_2:
                    i_d1 = i_flat // n_dofs
                    i_d2 = i_flat % n_dofs
                    if i_d2 <= i_d1:
                        H[i_d1, i_d2] = constraint_state.nt_H[i_b, i_d1, i_d2]
                    i_flat = i_flat + BLOCK_DIM
                k_d = tid
                while k_d < n_dofs:
                    v[k_d] = constraint_state.Mgrad[k_d, i_b]
                    k_d = k_d + BLOCK_DIM
                ti.simt.block.sync()

                for i_d in range(n_dofs):
                    dot = gs.ti_float(0.0)
                    j_d = tid
                    while j_d < i_d:
                        dot = dot + H[i_d, j_d] * v[j_d]
                        j_d = j_d + BLOCK_DIM
                    if ti.static(ENABLE_WARP_REDUCTION):
                        for offset in ti.static([16, 8, 4, 2, 1]):
                            dot = dot + ti.simt.warp.shfl_down_f32(ti.u32(0xFFFFFFFF), dot, offset)
                        if lane_id == 0:
                            partial[warp_id] = dot
                    else:
                        partial[tid] = dot
                    ti.simt.block.sync()

                    if tid == 0:
                        total = gs.ti_float(0.0)
                        for k in ti.static(range(NUM_WARPS)) if ti.static(ENABLE_WARP_REDUCTION) else range(BLOCK_DIM):
                            total = total + partial[k]
                        v[i_d] = (v[i_d] - total) / H[i_d, i_d]
                    ti.simt.block.sync()

                for i_d_ in range(n_dofs):
                    i_d = n_dofs - 1 - i_d_
                    dot = gs.ti_float(0.0)
                    j_d = i_d + 1 + tid
                    while j_d < n_dofs:
                        dot = dot + H[j_d, i_d] * v[j_d]
                        j_d = j_d + BLOCK_DIM

                    if ti.static(ENABLE_WARP_REDUCTION):
                        for offset in ti.static([16, 8, 4, 2, 1]):
                            dot = dot + ti.simt.warp.shfl_down_f32(ti.u32(0xFFFFFFFF), dot, offset)
                        if lane_id == 0:
                            partial[warp_id] = dot
                    else:
                        partial[tid] = dot
                    ti.simt.block.sync()

                    if tid == 0:
                        total = gs.ti_float(0.0)
                        for k in ti.static(range(NUM_WARPS)) if ti.static(ENABLE_WARP_REDUCTION) else range(BLOCK_DIM):
                            total = total + partial[k]
                        v[i_d] = (v[i_d] - total) / H[i_d, i_d]
                    ti.simt.block.sync()

                k_d = tid
                while k_d < n_dofs:
                    constraint_state.Mgrad[k_d, i_b] = v[k_d]
                    k_d = k_d + BLOCK_DIM
        else:
            ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
            for i_b in range(_B):
                func_nt_chol_factor(i_b, constraint_state, rigid_global_info)

            ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
            for i_b in range(_B):
                for i_d in range(n_dofs):
                    constraint_state.grad[i_d, i_b] = (
                        constraint_state.Ma[i_d, i_b]
                        - dofs_state.force[i_d, i_b]
                        - constraint_state.qfrc_constraint[i_d, i_b]
                    )
                func_nt_chol_solve(i_b, constraint_state)

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in ti.ndrange(n_dofs, _B):
        constraint_state.search[i_d, i_b] = -constraint_state.Mgrad[i_d, i_b]
