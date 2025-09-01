import gstaichi as ti

import genesis as gs
import genesis.utils.array_class as array_class

import genesis.engine.solvers.rigid.rigid_solver_decomp as rigid_solver


@gs.maybe_pure
@ti.kernel
def kernel_build_efc_AR(
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
    static_rigid_sim_cache_key: array_class.StaticRigidSimCacheKey,
):
    _B = constraint_state.jac.shape[2]
    n_dofs = constraint_state.jac.shape[1]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_b in range(_B):
        nefc = constraint_state.n_constraints[i_b]
        # zero AR
        for r in range(nefc):
            for c in range(nefc):
                constraint_state.efc_AR[r, c, i_b] = gs.ti_float(0.0)

        # build AR = J * inv(M) * J^T
        # do it row-by-row: for each row r, tmp = inv(M) * J[r]^T, then AR[r,:] = J * tmp
        for r in range(nefc):
            # tmp = M^{-1} * Jr^T
            for i_d in range(n_dofs):
                constraint_state.Mgrad[i_d, i_b] = constraint_state.jac[r, i_d, i_b]

            rigid_solver.func_solve_mass_batched(
                constraint_state.Mgrad,
                constraint_state.Mgrad,
                i_b,
                entities_info=entities_info,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
            )

            # AR[r, c] = J[c, :] dot tmp
            for c in range(nefc):
                s = gs.ti_float(0.0)
                for i_d in range(n_dofs):
                    s += constraint_state.jac[c, i_d, i_b] * constraint_state.Mgrad[i_d, i_b]
                constraint_state.efc_AR[r, c, i_b] = s

        # add R to diagonal: AR[ii] += R[i]
        for r in range(nefc):
            constraint_state.efc_AR[r, r, i_b] += constraint_state.efc_R[r, i_b]


def kernel_noslip(
    entities_info: array_class.EntitiesInfo,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
    static_rigid_sim_cache_key: array_class.StaticRigidSimCacheKey,
):
    _B = constraint_state.jac.shape[2]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):

        # temp variables
        res = ti.Vector.zero(gs.ti_float, 5)
        old_force = ti.Vector.zero(gs.ti_float, 5)
        bc = ti.Vector.zero(gs.ti_float, 5)
        Ac = ti.Vector.zero(gs.ti_float, 9)

        const_start = nef
        n_con = collider_state.n_contacts[i_b]
        ne = constraint_state.n_constraints_equality[i_b]
        nef = ne + constraint_state.n_constraints_frictionloss[i_b]

        for i_iter in range(static_rigid_sim_config.noslip_iterations):
            # Residual-stepped dry joint friction-loss with diagonal A
            # TODO: friction loss
            # for i_c in range(ne, nef):
            #     res = constraint_state.efc_resid[i_c, i_b]
            #     adiag = ti.max(constraint_state.efc_A_diag[i_c, i_b], gs.EPS)
            #     f = constraint_state.efc_force[i_c, i_b] - res / adiag
            #     floss = constraint_state.efc_frictionloss[i_c, i_b]
            #     if f > floss:
            #         f = floss
            #     elif f < -floss:
            #         f = -floss
            #     constraint_state.efc_force[i_c, i_b] = f

            # Project contact friction (pyramidal 4-edge) with normal fixed
            for i_col in range(n_con):
                base = const_start + i_col * 4
                mu = collider_state.contact_data.friction[i_col, i_b]

                for j_efc in range(base, base + 4, 2):
                    func_residual(
                        res=res,
                        i_b=i_b,
                        i_efc=j_efc,
                        dim=2,
                        flg_subR=True,
                        constraint_state=constraint_state,
                    )
                    for i2 in range(2):
                        old_force[i2] = constraint_state.efc_force[j_efc + i2, i_b]
                    func_extract_block(
                        Ac=Ac,
                        i_b=i_b,
                        start=j_efc,
                        n=2,
                        flg_subR=True,
                    )
                    for i2 in range(2):
                        bc[i2] = res[i2]
                        for i3 in range(2):
                            bc[i2] -= Ac[i2 * 2 + i3] * old_force[i3]
                    mid = 0.5 * (
                        constraint_state.efc_force[j_efc + 0, i_b] + constraint_state.efc_force[j_efc + 1, i_b]
                    )
                    y = 0.5 * (constraint_state.efc_force[j_efc + 0, i_b] - constraint_state.efc_force[j_efc + 1, i_b])
                    K1 = Ac[0] + Ac[3] - Ac[1] - Ac[2]
                    K0 = mid * (Ac[0] - Ac[3]) + bc[0] - bc[1]
                    if K1 < gs.EPS:
                        constraint_state.efc_force[j_efc + 0, i_b] = constraint_state.efc_force[j_efc + 1, i_b] = mid
                    else:
                        y = -K0 / K1
                        if y < -mid:
                            constraint_state.efc_force[j_efc + 0, i_b] = 0
                            constraint_state.efc_force[j_efc + 1, i_b] = 2 * mid
                        elif y > mid:
                            constraint_state.efc_force[j_efc + 0, i_b] = 2 * mid
                            constraint_state.efc_force[j_efc + 1, i_b] = 0
                        else:
                            constraint_state.efc_force[j_efc + 0, i_b] = mid + y
                            constraint_state.efc_force[j_efc + 1, i_b] = mid - y
                    improvement -= func_cost_change(
                        Ac, constraint_state.efc_force[j_efc : j_efc + 2, i_b], j_efc, old_force, res, 2
                    )
            # start solve

            # TODO: efc_state
            # // process state
            # mju_copyInt(oldstate, d->efc_state, nefc);
            # int nactive = dualState(m, d, d->efc_state);
            # int nchange = 0;
            # for (int i=0; i < nefc; i++) {
            # nchange += (oldstate[i] != d->efc_state[i]);
            # }

            # // scale improvement, save stats
            # improvement *= scale;

            # // save noslip stats after all the entries from regular solver
            # int stats_iter = iter + d->solver_niter[island];
            # saveStats(m, d, island, stats_iter, improvement, 0, 0, nactive, nchange, 0, 0);

            # // increment iteration count
            # iter++;

            # // terminate
            # if (improvement < m->opt.noslip_tolerance) {
            #     break;
            # }
            improvement *= scale
            if improvement < static_rigid_sim_config.tolerance:
                break


@gs.maybe_pure
@ti.kernel
def kernel_dual_finish(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
    static_rigid_sim_cache_key: array_class.StaticRigidSimCacheKey,
):
    n_dofs = constraint_state.qfrc_constraint.shape[0]
    _B = constraint_state.qfrc_constraint.shape[1]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        # zero
        for i_d in range(n_dofs):
            constraint_state.qfrc_constraint[i_d, i_b] = gs.ti_float(0.0)

            for i_c in range(constraint_state.n_constraints[i_b]):
                for i_d in range(n_dofs):
                    constraint_state.qfrc_constraint[i_d, i_b] = (
                        constraint_state.qfrc_constraint[i_d, i_b]
                        + constraint_state.jac[i_c, i_d, i_b] * constraint_state.efc_force[i_c, i_b]
                    )

    rigid_solver.func_solve_mass(
        vec=constraint_state.qfrc_constraint,
        out=constraint_state.qacc,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
    )

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        for i_d in range(n_dofs):
            dofs_state.acc[i_d, i_b] = constraint_state.qacc[i_d, i_b] + dofs_state.acc_smooth[i_d, i_b]


@ti.func
def compute_efc_b(
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
):
    _B = dofs_state.acc_smooth.shape[1]
    n_dofs = dofs_state.acc_smooth.shape[0]
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        for i_c in range(constraint_state.n_constraints[i_b]):
            v = -constraint_state.aref[i_c, i_b]
            if ti.static(static_rigid_sim_config.sparse_solve):
                for i_d_ in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                    i_d = constraint_state.jac_relevant_dofs[i_c, i_d_, i_b]
                    v += constraint_state.jac[i_c, i_d, i_b] * dofs_state.acc_smooth[i_d, i_b]
            else:
                for i_d in range(n_dofs):
                    v += constraint_state.jac[i_c, i_d, i_b] * dofs_state.acc_smooth[i_d, i_b]
            constraint_state.efc_b[i_c, i_b] = v


@ti.func
def func_extract_block(
    Ac,
    i_b: int,
    start: int,
    n: int,
    flg_subR: bool,
):
    for j in range(n):
        for k in range(n):
            Ac[j * n + k] = constraint_state.efc_AR[start + j, start + k, i_b]
    if flg_subR:
        for j in range(n):
            Ac[j * (n + 1)] -= constraint_state.efc_R[start + j, i_b]
            Ac[j * (n + 1)] = ti.max(1e-10, Ac[j * (n + 1)])


@ti.func
def func_residual(
    res,
    i_b: int,
    i_efc: int,
    dim: int,
    flg_subR: bool,
    constraint_state: array_class.ConstraintState,
):
    for j in range(dim):
        res[j] = constraint_state.efc_b[i_efc + j, i_b]
        for k in range(constraint_state.n_constraints[i_b]):
            res[j] += constraint_state.efc_AR[i_efc + j, k, i_b] * constraint_state.efc_force[k, i_b]
    if flg_subR:
        for j in range(dim):
            res[j] -= constraint_state.efc_R[i_efc + j, i_b] * constraint_state.efc_force[i_efc + j, i_b]


@ti.func
def func_cost_change(
    A,
    force,
    force_start: int,
    old_force,
    res,
    dim: int,
):
    if dim == 1:
        delta = force[force_start + 0] - old_force[0]
        change = 0.5 * A[0] * delta * delta + delta * res[0]
    else:
        delta = ti.Vector.zero(gs.ti_float, 6)
        for i in range(dim):
            delta[i] = force[force_start + i] - old_force[i]
        change = gs.ti_float(0.0)
        # change = 0.5*mju_mulVecMatVec(delta, A, delta, dim) + mju_dot(delta, res, dim);
        for i in range(dim):
            for j in range(dim):
                change += A[i * dim + j] * delta[i] * delta[j]
            change += delta[i] * res[i]
    if change > 1e-10:
        for i in range(dim):
            force[force_start + i] = old_force[i]
        change = 0.0
    return change


@gs.maybe_pure
@ti.kernel
def compute_A_diag(
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
    static_rigid_sim_cache_key: array_class.StaticRigidSimCacheKey,
):
    _B = constraint_state.jac.shape[2]
    n_dofs = constraint_state.jac.shape[1]
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        # For each constraint row i: Ai = Ji * M^{-1} * Ji^T
        for i_c in range(constraint_state.n_constraints[i_b]):
            # tmp = M^{-1} * Ji^T
            for i_d in range(n_dofs):
                constraint_state.Mgrad[i_d, i_b] = constraint_state.jac[i_c, i_d, i_b]

            rigid_solver.func_solve_mass_batched(
                constraint_state.Mgrad,
                constraint_state.Mgrad,
                i_b,
                entities_info=entities_info,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
            )

            # Ai = Ji * tmp
            aii = gs.ti_float(0.0)
            if ti.static(static_rigid_sim_config.sparse_solve):
                for i_d_ in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                    i_d = constraint_state.jac_relevant_dofs[i_c, i_d_, i_b]
                    aii += constraint_state.jac[i_c, i_d, i_b] * constraint_state.Mgrad[i_d, i_b]
            else:
                for i_d in range(n_dofs):
                    aii += constraint_state.jac[i_c, i_d, i_b] * constraint_state.Mgrad[i_d, i_b]
            constraint_state.efc_A_diag[i_c, i_b] = aii
