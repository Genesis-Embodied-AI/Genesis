import gstaichi as ti

import genesis as gs
import genesis.utils.array_class as array_class

import genesis.engine.solvers.rigid.rigid_solver_decomp as rigid_solver


@ti.kernel(pure=gs.use_pure)
def kernel_build_efc_AR_b(
    dofs_state: array_class.DofsState,
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
        for i_row in range(nefc):
            for i_col in range(nefc):
                constraint_state.efc_AR[i_row, i_col, i_b] = gs.ti_float(0.0)

        # build AR = J * inv(M) * J^T
        # do it row-by-row: for each row r, tmp = inv(M) * J[r]^T, then AR[r,:] = J * tmp
        for i_row in range(nefc):
            # tmp = M^{-1} * Jr^T
            for i_d in range(n_dofs):
                constraint_state.Mgrad[i_d, i_b] = constraint_state.jac[i_row, i_d, i_b]

            rigid_solver.func_solve_mass_batched(
                constraint_state.Mgrad,
                constraint_state.Mgrad,
                None,
                i_b,
                entities_info=entities_info,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
                is_backward=False,
            )

            # AR[r, c] = J[c, :] * tmp
            for i_col in range(nefc):
                s = gs.ti_float(0.0)
                for i_d in range(n_dofs):
                    s += constraint_state.jac[i_col, i_d, i_b] * constraint_state.Mgrad[i_d, i_b]
                constraint_state.efc_AR[i_row, i_col, i_b] = s

        for i_c in range(constraint_state.n_constraints[i_b]):
            v = -constraint_state.aref[i_c, i_b]
            for i_d in range(n_dofs):
                v += constraint_state.jac[i_c, i_d, i_b] * dofs_state.acc_smooth[i_d, i_b]
            constraint_state.efc_b[i_c, i_b] = v


@ti.kernel(pure=gs.use_pure)
def kernel_noslip(
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    static_rigid_sim_cache_key: array_class.StaticRigidSimCacheKey,
):
    _B = constraint_state.jac.shape[2]
    n_dofs = constraint_state.jac.shape[1]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        # temp variables
        res = ti.Vector.zero(gs.ti_float, 5)
        old_force = ti.Vector.zero(gs.ti_float, 5)
        bc = ti.Vector.zero(gs.ti_float, 5)
        Ac = ti.Vector.zero(gs.ti_float, 9)

        n_con = collider_state.n_contacts[i_b]
        ne = constraint_state.n_constraints_equality[i_b]
        nf = constraint_state.n_constraints_frictionloss[i_b]
        const_start = ne + nf

        scale = 1.0 / (rigid_global_info.meaninertia[i_b] * ti.max(1.0, n_dofs))

        for i_iter in range(rigid_global_info.noslip_iterations[None]):
            improvement = gs.ti_float(0.0)
            if i_iter == 0:
                for i_c in range(constraint_state.n_constraints[i_b]):
                    improvement += 0.5 * constraint_state.efc_force[i_c, i_b] ** 2 * constraint_state.diag[i_c, i_b]

            for i_c in range(ne, ne + nf):
                res = func_residual_constraint_force(
                    res=res,
                    i_b=i_b,
                    i_efc=i_c,
                    dim=1,
                    constraint_state=constraint_state,
                )
                old_force[0] = constraint_state.efc_force[i_c, i_b]
                constraint_state.efc_force[i_c, i_b] -= res[0] / constraint_state.efc_AR[i_c, i_c, i_b]
                if constraint_state.efc_force[i_c, i_b] < -constraint_state.efc_frictionloss[i_c, i_b]:
                    constraint_state.efc_force[i_c, i_b] = -constraint_state.efc_frictionloss[i_c, i_b]
                elif constraint_state.efc_force[i_c, i_b] > constraint_state.efc_frictionloss[i_c, i_b]:
                    constraint_state.efc_force[i_c, i_b] = constraint_state.efc_frictionloss[i_c, i_b]
                delta = constraint_state.efc_force[i_c, i_b] - old_force[0]
                improvement -= 0.5 * delta**2 / constraint_state.efc_AR[i_c, i_c, i_b] + delta * res[0]

            # Project contact friction (pyramidal 4-edge) with normal fixed
            for i_col in range(n_con):
                base = const_start + i_col * 4
                mu = collider_state.contact_data.friction[i_col, i_b]
                for j2 in ti.static(range(2)):
                    j_efc = base + j2 * 2
                    res = func_residual_constraint_force(
                        res=res,
                        i_b=i_b,
                        i_efc=j_efc,
                        dim=2,
                        constraint_state=constraint_state,
                    )
                    for i2 in ti.static(range(2)):
                        old_force[i2] = constraint_state.efc_force[j_efc + i2, i_b]
                    Ac = func_extract_block_matrix_from_AR(
                        Ac=Ac,
                        i_b=i_b,
                        start=j_efc,
                        n=2,
                        constraint_state=constraint_state,
                    )
                    for j in ti.static(range(2)):
                        bc[j] = res[j]
                        for k in ti.static(range(2)):
                            bc[j] -= Ac[j * 2 + k] * old_force[k]
                    mid = 0.5 * (constraint_state.efc_force[j_efc, i_b] + constraint_state.efc_force[j_efc + 1, i_b])
                    y = 0.5 * (constraint_state.efc_force[j_efc, i_b] - constraint_state.efc_force[j_efc + 1, i_b])
                    K1 = Ac[0] + Ac[3] - Ac[1] - Ac[2]
                    K0 = mid * (Ac[0] - Ac[3]) + bc[0] - bc[1]
                    if K1 < gs.EPS:
                        constraint_state.efc_force[j_efc, i_b] = constraint_state.efc_force[j_efc + 1, i_b] = mid
                    else:
                        y = -K0 / K1
                        if y < -mid:
                            constraint_state.efc_force[j_efc, i_b] = 0
                            constraint_state.efc_force[j_efc + 1, i_b] = 2 * mid
                        elif y > mid:
                            constraint_state.efc_force[j_efc, i_b] = 2 * mid
                            constraint_state.efc_force[j_efc + 1, i_b] = 0
                        else:
                            constraint_state.efc_force[j_efc, i_b] = mid + y
                            constraint_state.efc_force[j_efc + 1, i_b] = mid - y
                    cost_change = func_cost_change(
                        i_b=i_b,
                        Ac=Ac,
                        force=constraint_state.efc_force,
                        force_start=j_efc,
                        old_force=old_force,
                        res=res,
                        dim=2,
                    )

                    improvement -= cost_change
            improvement *= scale

            if improvement < rigid_global_info.noslip_tolerance[None]:
                break


@ti.kernel(pure=gs.use_pure)
def kernel_dual_finish(
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
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
                constraint_state.qfrc_constraint[i_d, i_b] = (
                    constraint_state.qfrc_constraint[i_d, i_b]
                    + constraint_state.jac[i_c, i_d, i_b] * constraint_state.efc_force[i_c, i_b]
                )

        rigid_solver.func_solve_mass_batched(
            vec=constraint_state.qfrc_constraint,
            out=constraint_state.qacc,
            out_bw=None,
            i_b=i_b,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
            is_backward=False,
        )

        for i_d in range(n_dofs):
            constraint_state.qacc[i_d, i_b] = constraint_state.qacc[i_d, i_b] + dofs_state.acc_smooth[i_d, i_b]
            dofs_state.acc[i_d, i_b] = constraint_state.qacc[i_d, i_b]

            dofs_state.qf_constraint[i_d, i_b] = constraint_state.qfrc_constraint[i_d, i_b]
            dofs_state.force[i_d, i_b] = dofs_state.qf_smooth[i_d, i_b] + constraint_state.qfrc_constraint[i_d, i_b]


@ti.func
def func_extract_block_matrix_from_AR(
    Ac,
    i_b: int,
    start: int,
    n: int,
    constraint_state: array_class.ConstraintState,
):
    for j in range(n):
        for k in range(n):
            Ac[j * n + k] = constraint_state.efc_AR[start + j, start + k, i_b]
    return Ac


@ti.func
def func_residual_constraint_force(
    res,
    i_b: int,
    i_efc: int,
    dim: int,
    constraint_state: array_class.ConstraintState,
):
    for j in range(dim):
        res[j] = constraint_state.efc_b[i_efc + j, i_b]
        for k in range(constraint_state.n_constraints[i_b]):
            res[j] += constraint_state.efc_AR[i_efc + j, k, i_b] * constraint_state.efc_force[k, i_b]
    return res


@ti.func
def func_cost_change(
    i_b: int,
    Ac,
    force: array_class.V_ANNOTATION,
    force_start: int,
    old_force,
    res,
    dim: int,
):
    change = gs.ti_float(0.0)
    if dim == 1:
        delta = force[force_start, i_b] - old_force[0]
        change = 0.5 * Ac[0] * delta * delta + delta * res[0]
    else:
        delta = ti.Vector.zero(gs.ti_float, 2)
        for i in range(dim):
            delta[i] = force[force_start + i, i_b] - old_force[i]
        for i in range(dim):
            for j in range(dim):
                change += 0.5 * Ac[i * dim + j] * delta[i] * delta[j]
            change += delta[i] * res[i]
    if change > gs.EPS:
        for i in range(dim):
            force[force_start + i, i_b] = old_force[i]
        change = 0.0
    return change


@ti.kernel(pure=gs.use_pure)
def compute_A_diag(
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
                None,
                i_b,
                entities_info=entities_info,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
                is_backward=False,
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
