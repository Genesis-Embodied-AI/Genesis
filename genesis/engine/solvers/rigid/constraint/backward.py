import quadrants as ti

import genesis as gs
import genesis.utils.array_class as array_class


@ti.func
def func_matvec_Ap(
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    i_b,
):
    """
    Compute Ap = (M + J^T * diag(D) * J) * p on the current active set, which is used for solving the adjoint u.

    Specifically, M = mass matrix, J = Jacobian, diag(D) = diagonal matrix of efc_D, and p = search direction.
    """
    n_dofs = constraint_state.bw_p.shape[0]
    for i_d in range(n_dofs):
        constraint_state.bw_Ap[i_d, i_b] = 0.0

    # Mp: Block multiplication
    n_entities = entities_info.n_links.shape[0]
    for i_e in range(n_entities):
        for i_d1 in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
            acc = gs.ti_float(0.0)
            for i_d2 in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                acc += rigid_global_info.mass_mat[i_d1, i_d2, i_b] * constraint_state.bw_p[i_d2, i_b]
            constraint_state.bw_Ap[i_d1, i_b] += acc

    # tmp = J v
    for i_c in range(constraint_state.n_constraints[i_b]):
        jv = gs.ti_float(0.0)
        if ti.static(static_rigid_sim_config.sparse_solve):
            for k in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                i_d = constraint_state.jac_relevant_dofs[i_c, k, i_b]
                jv += constraint_state.jac[i_c, i_d, i_b] * constraint_state.bw_p[i_d, i_b]
        else:
            for i_d in range(n_dofs):
                jv += constraint_state.jac[i_c, i_d, i_b] * constraint_state.bw_p[i_d, i_b]
        # only active constraints contribute
        jv *= constraint_state.efc_D[i_c, i_b] * constraint_state.active[i_c, i_b]
        # out += J^T (D * J v)
        if ti.static(static_rigid_sim_config.sparse_solve):
            for k in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                i_d = constraint_state.jac_relevant_dofs[i_c, k, i_b]
                constraint_state.bw_Ap[i_d, i_b] += constraint_state.jac[i_c, i_d, i_b] * jv
        else:
            for i_d in range(n_dofs):
                constraint_state.bw_Ap[i_d, i_b] += constraint_state.jac[i_c, i_d, i_b] * jv


@ti.kernel
def kernel_solve_adjoint_u(
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
):
    r"""
    Solve for the adjoint vector [u] from Au = g, where A = dF/dqacc (primal Hessian on the active set) and g = dL/dqacc.
    Intuitively, [u] is a sensitivity vector that translates the upstream gradient dL/dqacc into the primal space.
    This adjoint vector [u] can be used an intermediate variable to compute the downstream gradients. Since A is a
    Semi-Positive Definite (SPD) matrix, we can solve A * u = g using either Cholesky decomposition or CG solver.
    When Newton solver was used, we reuse the Cholesky decomposition of A (= L * L^T) to solve A * u = g. Otherwise,
    we use CG solver.

    Specifically, A = M + J^T * diag(D) * J, where M = mass matrix, J = Jacobian, diag(D) = diagonal matrix of efc_D.
    """
    n_dofs = constraint_state.bw_u.shape[0]
    _B = constraint_state.bw_u.shape[1]

    # Initialize u
    for i_d, i_b in ti.ndrange(n_dofs, _B):
        constraint_state.bw_u[i_d, i_b] = 0.0

    if ti.static(static_rigid_sim_config.solver_type == gs.constraint_solver.Newton):
        # Since we already have the Cholesky decomposition of A (= L * L^T), we can use it to solve A * u = g.
        for i_b in range(_B):
            # z = L^{-1} g  (forward substitution)
            # Save solution to bw_r
            for i_d in range(n_dofs):
                z = constraint_state.dL_dqacc[i_d, i_b]
                for j_d in range(i_d):
                    z -= constraint_state.nt_H[i_b, i_d, j_d] * constraint_state.bw_r[j_d, i_b]
                z /= constraint_state.nt_H[i_b, i_d, i_d]
                constraint_state.bw_r[i_d, i_b] = z

            # u = L^{-T} z  (back substitution)
            for i_d_ in range(n_dofs):
                i_d = n_dofs - 1 - i_d_
                u = constraint_state.bw_r[i_d, i_b]
                for j_d in range(i_d + 1, n_dofs):
                    u -= constraint_state.nt_H[i_b, j_d, i_d] * constraint_state.bw_u[j_d, i_b]
                u /= constraint_state.nt_H[i_b, i_d, i_d]
                constraint_state.bw_u[i_d, i_b] = u
    else:
        # Use CG solver for solving A * u = g.
        # 2. Local buffers for solving A * u = g
        # Initialize r, p with dL_dqacc
        for i_d, i_b in ti.ndrange(n_dofs, _B):
            # Residual: g - A * 0 (u = 0)
            constraint_state.bw_r[i_d, i_b] = constraint_state.dL_dqacc[i_d, i_b]
            # Search direction: p = r
            constraint_state.bw_p[i_d, i_b] = constraint_state.bw_r[i_d, i_b]

        # 3. Solve A * u = g, parallelized over batch dimension
        for i_b in range(_B):
            # Compute Ap for the current search direction
            for it in range(static_rigid_sim_config.iterations):
                func_matvec_Ap(
                    entities_info=entities_info,
                    rigid_global_info=rigid_global_info,
                    constraint_state=constraint_state,
                    static_rigid_sim_config=static_rigid_sim_config,
                    i_b=i_b,
                )

                # alpha = (r,r)/(p,Hp)
                num = gs.ti_float(0.0)
                den = gs.ti_float(0.0)
                for i_d in range(n_dofs):
                    num += constraint_state.bw_r[i_d, i_b] * constraint_state.bw_r[i_d, i_b]
                    den += constraint_state.bw_p[i_d, i_b] * constraint_state.bw_Ap[i_d, i_b]
                alpha = num / ti.max(den, rigid_global_info.EPS[None])

                # u += alpha p ; r -= alpha Hp
                for i_d in range(n_dofs):
                    constraint_state.bw_u[i_d, i_b] += alpha * constraint_state.bw_p[i_d, i_b]
                    constraint_state.bw_r[i_d, i_b] -= alpha * constraint_state.bw_Ap[i_d, i_b]

                # check tol (optional: per-batch)
                # TODO: Might need lower tolerance?
                if num < rigid_global_info.EPS[None]:
                    break

                # beta = (r_new,r_new)/(r_old,r_old)
                num_new = gs.ti_float(0.0)
                for i_d in range(n_dofs):
                    num_new += constraint_state.bw_r[i_d, i_b] * constraint_state.bw_r[i_d, i_b]
                beta = num_new / ti.max(num, rigid_global_info.EPS[None])

                # p = r + beta p
                for i_d in range(n_dofs):
                    constraint_state.bw_p[i_d, i_b] = (
                        constraint_state.bw_r[i_d, i_b] + beta * constraint_state.bw_p[i_d, i_b]
                    )


@ti.kernel
def kernel_compute_gradients(
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
):
    r"""
    Compute gradients of the loss with respect to the input variables to this solver. Note that we use the intermediate
    adjoint vector [u] computed in [kernel_solve_adjoint_u] to compute these gradients.

    Specifically, the gradients are computed as follows:
    - dL_dM = -u * qacc^T
    - dL_djac = -[u * y^T + qacc * (D \odot (Ju))^T] (y = D \odot w, w = (Jqacc - aref))
    - dL_daref = Ju \odot D
    - dL_defc_D = -Ju \odot (Jqacc - aref)
    - dL_dforce = u
    """
    _B = constraint_state.bw_u.shape[1]
    n_dofs = constraint_state.bw_u.shape[0]
    n_constraints = constraint_state.bw_Ju.shape[0]

    # clear grads
    for i_d0, i_d1, i_b in ti.ndrange(n_dofs, n_dofs, _B):
        constraint_state.dL_dM[i_d0, i_d1, i_b] = gs.ti_float(0.0)
    for i_c, i_d, i_b in ti.ndrange(n_constraints, n_dofs, _B):
        constraint_state.dL_djac[i_c, i_d, i_b] = gs.ti_float(0.0)
    for i_c, i_b in ti.ndrange(n_constraints, _B):
        constraint_state.dL_daref[i_c, i_b] = gs.ti_float(0.0)
        constraint_state.dL_defc_D[i_c, i_b] = gs.ti_float(0.0)
        constraint_state.bw_Ju[i_c, i_b] = gs.ti_float(0.0)
        constraint_state.bw_y[i_c, i_b] = gs.ti_float(0.0)
        constraint_state.bw_w[i_c, i_b] = gs.ti_float(0.0)
    for i_d, i_b in ti.ndrange(n_dofs, _B):
        constraint_state.dL_dforce[i_d, i_b] = gs.ti_float(0.0)

    # Ju, w, y
    for i_b in range(_B):
        # Ju
        for i_c in range(constraint_state.n_constraints[i_b]):
            s = gs.ti_float(0.0)
            if ti.static(static_rigid_sim_config.sparse_solve):
                for k in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                    i_d = constraint_state.jac_relevant_dofs[i_c, k, i_b]
                    s += constraint_state.jac[i_c, i_d, i_b] * constraint_state.bw_u[i_d, i_b]
            else:
                for i_d in range(n_dofs):
                    s += constraint_state.jac[i_c, i_d, i_b] * constraint_state.bw_u[i_d, i_b]
            constraint_state.bw_Ju[i_c, i_b] = s

        # w = J qacc - aref
        # y = D \odot w
        for i_c in range(constraint_state.n_constraints[i_b]):
            t = gs.ti_float(0.0)
            if ti.static(static_rigid_sim_config.sparse_solve):
                for k in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                    i_d = constraint_state.jac_relevant_dofs[i_c, k, i_b]
                    t += constraint_state.jac[i_c, i_d, i_b] * constraint_state.qacc[i_d, i_b]
            else:
                for i_d in range(n_dofs):
                    t += constraint_state.jac[i_c, i_d, i_b] * constraint_state.qacc[i_d, i_b]
            constraint_state.bw_w[i_c, i_b] = t - constraint_state.aref[i_c, i_b]
            constraint_state.bw_y[i_c, i_b] = constraint_state.efc_D[i_c, i_b] * constraint_state.bw_w[i_c, i_b]

        # grads
        # force: u
        for i_d in range(n_dofs):
            constraint_state.dL_dforce[i_d, i_b] += constraint_state.bw_u[i_d, i_b]

        # per-constraint (mask by active)
        # aref: Ju \odot D
        # D: -Ju \odot w
        # J: -[u * y^T + qacc * (D \odot (Ju)^T)]
        for i_c in range(constraint_state.n_constraints[i_b]):
            if constraint_state.active[i_c, i_b] != 0:
                # aref: Ju \odot D
                constraint_state.dL_daref[i_c, i_b] += (
                    constraint_state.efc_D[i_c, i_b] * constraint_state.bw_Ju[i_c, i_b]
                )
                # D: -Ju \odot w
                constraint_state.dL_defc_D[i_c, i_b] -= (
                    constraint_state.bw_Ju[i_c, i_b] * constraint_state.bw_w[i_c, i_b]
                )

                # J: -[u * y^T + qacc * (D \odot (Ju))^T]
                DJu_i = constraint_state.efc_D[i_c, i_b] * constraint_state.bw_Ju[i_c, i_b]
                y_i = constraint_state.bw_y[i_c, i_b]

                if ti.static(static_rigid_sim_config.sparse_solve):
                    for k in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                        i_d = constraint_state.jac_relevant_dofs[i_c, k, i_b]
                        constraint_state.dL_djac[i_c, i_d, i_b] += -(
                            constraint_state.bw_u[i_d, i_b] * y_i + constraint_state.qacc[i_d, i_b] * DJu_i
                        )
                else:
                    for i_d in range(n_dofs):
                        constraint_state.dL_djac[i_c, i_d, i_b] += -(
                            constraint_state.bw_u[i_d, i_b] * y_i + constraint_state.qacc[i_d, i_b] * DJu_i
                        )

        # M: -u * qacc^T
        n_entities = entities_info.n_links.shape[0]
        for i_e in range(n_entities):
            s = entities_info.dof_start[i_e]
            e = entities_info.dof_end[i_e]
            for i in range(s, e):
                for j in range(s, e):
                    val0 = -constraint_state.bw_u[i, i_b] * constraint_state.qacc[j, i_b]
                    val1 = -constraint_state.bw_u[j, i_b] * constraint_state.qacc[i, i_b]
                    constraint_state.dL_dM[i, j, i_b] += (val0 + val1) * 0.5  # symmetrize
