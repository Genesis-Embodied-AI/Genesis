import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class

import genesis.engine.solvers.rigid.rigid_solver as rigid_solver


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_build_efc_AR_b(
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    _B = constraint_state.jac.shape[2]
    n_dofs = constraint_state.jac.shape[1]

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        nefc = constraint_state.n_constraints[i_b]
        # zero AR
        for i_row in range(nefc):
            for i_col in range(nefc):
                constraint_state.efc_AR[i_row, i_col, i_b] = gs.qd_float(0.0)

        # build AR = J * inv(M) * J^T
        # do it row-by-row: for each row r, tmp = inv(M) * J[r]^T, then AR[r,:] = J * tmp
        for i_row in range(nefc):
            # tmp = M^{-1} * Jr^T
            for i_d in range(n_dofs):
                constraint_state.Mgrad[i_d, i_b] = constraint_state.jac[i_row, i_d, i_b]

            rigid_solver.func_solve_mass_batch(
                i_b,
                constraint_state.Mgrad,
                constraint_state.Mgrad,
                array_class.PLACEHOLDER,
                entities_info=entities_info,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
                is_backward=False,
            )

            # AR[r, c] = J[c, :] * tmp
            for i_col in range(nefc):
                s = gs.qd_float(0.0)
                for i_d in range(n_dofs):
                    s += constraint_state.jac[i_col, i_d, i_b] * constraint_state.Mgrad[i_d, i_b]
                constraint_state.efc_AR[i_row, i_col, i_b] = s

        for i_c in range(constraint_state.n_constraints[i_b]):
            v = -constraint_state.aref[i_c, i_b]
            for i_d in range(n_dofs):
                v += constraint_state.jac[i_c, i_d, i_b] * dofs_state.acc_smooth[i_d, i_b]
            constraint_state.efc_b[i_c, i_b] = v


@qd.func
def func_solve_mass_entity_row(
    i_row: qd.int32,
    i_e: qd.int32,
    i_b: qd.int32,
    buf: array_class.V_ANNOTATION,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
):
    """LDL^T forward-backward substitution on buf[i_row, :, i_b].

    Same algorithm as func_solve_mass_entity (forward-only path), but operates
    on a 3D buffer indexed by (constraint_row, dof, batch). This allows
    different constraint rows to be solved in parallel since each row uses
    a separate memory slice.
    """
    if rigid_global_info.mass_mat_mask[i_e, i_b]:
        entity_dof_start = entities_info.dof_start[i_e]
        entity_dof_end = entities_info.dof_end[i_e]
        n_dofs = entities_info.n_dofs[i_e]

        # Step 1: Solve w s.t. L^T @ w = y (backward substitution)
        for i_d_ in range(n_dofs):
            i_d = entity_dof_end - i_d_ - 1
            curr_out = buf[i_row, i_d, i_b]
            for j_d in range(i_d + 1, entity_dof_end):
                curr_out = curr_out - rigid_global_info.mass_mat_L[j_d, i_d, i_b] * buf[i_row, j_d, i_b]
            buf[i_row, i_d, i_b] = curr_out

        # Step 2: z = D^{-1} @ w
        for i_d in range(entity_dof_start, entity_dof_end):
            buf[i_row, i_d, i_b] = buf[i_row, i_d, i_b] * rigid_global_info.mass_mat_D_inv[i_d, i_b]

        # Step 3: Solve x s.t. L @ x = z (forward substitution)
        for i_d in range(entity_dof_start, entity_dof_end):
            curr_out = buf[i_row, i_d, i_b]
            for j_d in range(entity_dof_start, i_d):
                curr_out = curr_out - rigid_global_info.mass_mat_L[i_d, j_d, i_b] * buf[i_row, j_d, i_b]
            buf[i_row, i_d, i_b] = curr_out


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_compute_MinvJT(
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """Compute MinvJT[row, :, i_b] = M^{-1} @ J[row, :, i_b] for each constraint row.

    Parallelized over (row, batch) via ndrange — each thread independently
    copies J[row] into MinvJT[row] and solves M^{-1} in-place using the
    row-indexed LDL^T substitution. No shared buffers between rows.
    """
    len_c = constraint_state.MinvJT.shape[0]
    _B = constraint_state.jac.shape[2]
    n_dofs = constraint_state.jac.shape[1]

    for i_row, i_b in qd.ndrange(len_c, _B):
        if i_row < constraint_state.n_constraints[i_b]:
            # Copy J[row] into MinvJT[row] (per-row buffer)
            for i_d in range(n_dofs):
                constraint_state.MinvJT[i_row, i_d, i_b] = constraint_state.jac[i_row, i_d, i_b]

            # In-place solve: MinvJT[row] = M^{-1} @ J[row]
            for i_0 in (
                range(rigid_global_info.n_awake_entities[i_b])
                if qd.static(static_rigid_sim_config.use_hibernation)
                else range(entities_info.n_links.shape[0])
            ):
                i_e = (
                    rigid_global_info.awake_entities[i_0, i_b]
                    if qd.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )
                func_solve_mass_entity_row(i_row, i_e, i_b, constraint_state.MinvJT, entities_info, rigid_global_info)


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_compute_AR_and_b(
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """Phase 2: Compute AR = J @ MinvJT and efc_b (GPU multi-env only).

    AR[row, col, i_b] = sum_d J[col, d, i_b] * MinvJT[row, d, i_b]

    Uses ndrange for full GPU parallelism across (row, col, batch) — gives
    nefc^2 * n_envs independent threads (~490K for typical scenes).

    This kernel is only called when para_level >= PARA_LEVEL.ALL (GPU multi-env).
    For CPU and GPU single-env, the fused kernel_build_efc_AR_b is used instead.
    """
    len_c = constraint_state.efc_AR.shape[0]
    _B = constraint_state.jac.shape[2]
    n_dofs = constraint_state.jac.shape[1]

    for i_row, i_col, i_b in qd.ndrange(len_c, len_c, _B):
        nefc = constraint_state.n_constraints[i_b]
        if i_row < nefc and i_col < nefc:
            s = gs.qd_float(0.0)
            for i_d in range(n_dofs):
                s += constraint_state.jac[i_col, i_d, i_b] * constraint_state.MinvJT[i_row, i_d, i_b]
            constraint_state.efc_AR[i_row, i_col, i_b] = s
        else:
            constraint_state.efc_AR[i_row, i_col, i_b] = gs.qd_float(0.0)

    for i_c, i_b in qd.ndrange(len_c, _B):
        if i_c < constraint_state.n_constraints[i_b]:
            v = -constraint_state.aref[i_c, i_b]
            for i_d in range(n_dofs):
                v += constraint_state.jac[i_c, i_d, i_b] * dofs_state.acc_smooth[i_d, i_b]
            constraint_state.efc_b[i_c, i_b] = v


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_noslip(
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    EPS = rigid_global_info.EPS[None]
    _B = constraint_state.jac.shape[2]
    n_dofs = constraint_state.jac.shape[1]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        # temp variables
        res = qd.Vector.zero(gs.qd_float, 5)
        old_force = qd.Vector.zero(gs.qd_float, 5)
        bc = qd.Vector.zero(gs.qd_float, 5)
        Ac = qd.Vector.zero(gs.qd_float, 9)

        n_con = collider_state.n_contacts[i_b]
        ne = constraint_state.n_constraints_equality[i_b]
        nf = constraint_state.n_constraints_frictionloss[i_b]
        const_start = ne + nf

        scale = 1.0 / (rigid_global_info.meaninertia[i_b] * qd.max(1.0, n_dofs))

        for i_iter in range(rigid_global_info.noslip_iterations[None]):
            improvement = gs.qd_float(0.0)
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
                for j2 in qd.static(range(2)):
                    j_efc = base + j2 * 2
                    res = func_residual_constraint_force(
                        res=res,
                        i_b=i_b,
                        i_efc=j_efc,
                        dim=2,
                        constraint_state=constraint_state,
                    )
                    for i2 in qd.static(range(2)):
                        old_force[i2] = constraint_state.efc_force[j_efc + i2, i_b]
                    Ac = func_extract_block_matrix_from_AR(
                        Ac=Ac,
                        i_b=i_b,
                        start=j_efc,
                        n=2,
                        constraint_state=constraint_state,
                    )
                    for j in qd.static(range(2)):
                        bc[j] = res[j]
                        for k in qd.static(range(2)):
                            bc[j] -= Ac[j * 2 + k] * old_force[k]
                    mid = 0.5 * (constraint_state.efc_force[j_efc, i_b] + constraint_state.efc_force[j_efc + 1, i_b])
                    y = 0.5 * (constraint_state.efc_force[j_efc, i_b] - constraint_state.efc_force[j_efc + 1, i_b])
                    K1 = Ac[0] + Ac[3] - Ac[1] - Ac[2]
                    K0 = mid * (Ac[0] - Ac[3]) + bc[0] - bc[1]
                    if K1 < EPS:
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
                        eps=EPS,
                    )

                    improvement -= cost_change
            improvement *= scale

            if improvement < rigid_global_info.noslip_tolerance[None]:
                break


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_dual_finish(
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    n_dofs = constraint_state.qfrc_constraint.shape[0]
    _B = constraint_state.qfrc_constraint.shape[1]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        # zero
        for i_d in range(n_dofs):
            constraint_state.qfrc_constraint[i_d, i_b] = gs.qd_float(0.0)

            for i_c in range(constraint_state.n_constraints[i_b]):
                constraint_state.qfrc_constraint[i_d, i_b] = (
                    constraint_state.qfrc_constraint[i_d, i_b]
                    + constraint_state.jac[i_c, i_d, i_b] * constraint_state.efc_force[i_c, i_b]
                )

        rigid_solver.func_solve_mass_batch(
            i_b=i_b,
            vec=constraint_state.qfrc_constraint,
            out=constraint_state.qacc,
            out_bw=array_class.PLACEHOLDER,
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


@qd.func
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


@qd.func
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


@qd.func
def func_cost_change(
    i_b: int,
    Ac,
    force: array_class.V_ANNOTATION,
    force_start: int,
    old_force,
    res,
    dim: int,
    eps,
):
    change = gs.qd_float(0.0)
    if dim == 1:
        delta = force[force_start, i_b] - old_force[0]
        change = 0.5 * Ac[0] * delta * delta + delta * res[0]
    else:
        delta = qd.Vector.zero(gs.qd_float, 2)
        for i in range(dim):
            delta[i] = force[force_start + i, i_b] - old_force[i]
        for i in range(dim):
            for j in range(dim):
                change += 0.5 * Ac[i * dim + j] * delta[i] * delta[j]
            change += delta[i] * res[i]
    if change > eps:
        for i in range(dim):
            force[force_start + i, i_b] = old_force[i]
        change = 0.0
    return change


@qd.kernel(fastcache=gs.use_fastcache)
def compute_A_diag(
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    _B = constraint_state.jac.shape[2]
    n_dofs = constraint_state.jac.shape[1]
    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        # For each constraint row i: Ai = Ji * M^{-1} * Ji^T
        for i_c in range(constraint_state.n_constraints[i_b]):
            # tmp = M^{-1} * Ji^T
            for i_d in range(n_dofs):
                constraint_state.Mgrad[i_d, i_b] = constraint_state.jac[i_c, i_d, i_b]

            rigid_solver.func_solve_mass_batch(
                i_b,
                constraint_state.Mgrad,
                constraint_state.Mgrad,
                array_class.PLACEHOLDER,
                entities_info=entities_info,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
                is_backward=False,
            )

            # Ai = Ji * tmp
            aii = gs.qd_float(0.0)
            if qd.static(static_rigid_sim_config.sparse_solve):
                for i_d_ in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                    i_d = constraint_state.jac_relevant_dofs[i_c, i_d_, i_b]
                    aii += constraint_state.jac[i_c, i_d, i_b] * constraint_state.Mgrad[i_d, i_b]
            else:
                for i_d in range(n_dofs):
                    aii += constraint_state.jac[i_c, i_d, i_b] * constraint_state.Mgrad[i_d, i_b]
            constraint_state.efc_A_diag[i_c, i_b] = aii
