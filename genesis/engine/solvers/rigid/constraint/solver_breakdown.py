import quadrants as ti

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.solvers.rigid.constraint import solver

LS_PARALLEL_K = 8
LS_PARALLEL_MIN_STEP = 1e-6
_P0_BLOCK = 32
_JV_BLOCK = 32


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_linesearch(
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    _B = constraint_state.grad.shape[1]
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            solver.func_linesearch_and_apply_alpha(
                i_b,
                entities_info=entities_info,
                dofs_state=dofs_state,
                rigid_global_info=rigid_global_info,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )
        else:
            constraint_state.improved[i_b] = False


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_parallel_linesearch_mv(
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    """Compute mv = M @ search, parallelized over (dof, env).

    Uses per-dof entity lookup to find the entity block boundaries, giving n_dofs * B
    threads (each computing a single ~6-element dot product) instead of n_entities * B
    threads (each computing the full block matvec).
    """
    n_dofs = constraint_state.search.shape[0]
    _B = constraint_state.grad.shape[1]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d1, i_b in ti.ndrange(n_dofs, _B):
        if constraint_state.n_constraints[i_b] > 0:
            I_d1 = [i_d1, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d1
            i_e = dofs_info.entity_idx[I_d1]
            mv = gs.qd_float(0.0)
            for i_d2 in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                mv = mv + rigid_global_info.mass_mat[i_d1, i_d2, i_b] * constraint_state.search[i_d2, i_b]
            constraint_state.mv[i_d1, i_b] = mv


# @ti.kernel(fastcache=gs.use_fastcache)
# def _kernel_parallel_linesearch_jv(
#     constraint_state: array_class.ConstraintState,
#     static_rigid_sim_config: ti.template(),
# ):
#     """Compute jv = J @ search. T threads per env, search loaded into shared memory."""
#     n_dofs = constraint_state.search.shape[0]
#     _SHMEM_N_DOFS = ti.static(static_rigid_sim_config.tiled_n_dofs)
#     _B = constraint_state.grad.shape[1]
#     _T = ti.static(_JV_BLOCK)

#     ti.loop_config(block_dim=_T)
#     for i_ in range(_B * _T):
#         tid = i_ % _T
#         i_b = i_ // _T

#         # Shared memory: load search once per block instead of once per constraint
#         sh_search = ti.simt.block.SharedArray((_SHMEM_N_DOFS,), gs.qd_float)

#         if constraint_state.n_constraints[i_b] > 0:
#             # Cooperatively load search into shared memory
#             i_d = tid
#             while i_d < n_dofs:
#                 sh_search[i_d] = constraint_state.search[i_d, i_b]
#                 i_d += _T

#             ti.simt.block.sync()

#             # Each thread handles constraints in strided pattern
#             n_con = constraint_state.n_constraints[i_b]
#             i_c = tid
#             while i_c < n_con:
#                 jv = gs.qd_float(0.0)
#                 if ti.static(static_rigid_sim_config.sparse_solve):
#                     for i_d_ in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
#                         i_dd = constraint_state.jac_relevant_dofs[i_c, i_d_, i_b]
#                         jv += constraint_state.jac[i_c, i_dd, i_b] * sh_search[i_dd]
#                 else:
#                     for i_dd in range(n_dofs):
#                         jv += constraint_state.jac[i_c, i_dd, i_b] * sh_search[i_dd]
#                 constraint_state.jv[i_c, i_b] = jv
#                 i_c += _T


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_parallel_linesearch_jv(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
):
    """Compute jv = J @ search, parallelized over (constraint, env)."""
    n_dofs = constraint_state.search.shape[0]
    len_constraints = constraint_state.jac.shape[0]
    _B = constraint_state.grad.shape[1]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_c, i_b in ti.ndrange(len_constraints, _B):
        if i_c < constraint_state.n_constraints[i_b]:
            jv = gs.qd_float(0.0)
            if ti.static(static_rigid_sim_config.sparse_solve):
                for i_d_ in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                    i_d = constraint_state.jac_relevant_dofs[i_c, i_d_, i_b]
                    jv = jv + constraint_state.jac[i_c, i_d, i_b] * constraint_state.search[i_d, i_b]
            else:
                for i_d in range(n_dofs):
                    jv = jv + constraint_state.jac[i_c, i_d, i_b] * constraint_state.search[i_d, i_b]
            constraint_state.jv[i_c, i_b] = jv


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_parallel_linesearch_p0(
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    """Snorm check, quad_gauss, eq_sum, and p0_cost. T threads per env with shared memory reductions.

    Phase 1: Fused snorm + quad_gauss parallel reduction over n_dofs (Options A+B).
    Phase 2: Parallel reduction over n_constraints for eq_sum and p0_cost.
    """
    _B = constraint_state.grad.shape[1]
    _T = ti.static(_P0_BLOCK)

    ti.loop_config(block_dim=_T)
    for i_ in range(_B * _T):
        tid = i_ % _T
        i_b = i_ // _T

        # 4 shared arrays for parallel reductions (reused across phases)
        sh_a = ti.simt.block.SharedArray((_T,), gs.qd_float)
        sh_b = ti.simt.block.SharedArray((_T,), gs.qd_float)
        sh_c = ti.simt.block.SharedArray((_T,), gs.qd_float)
        sh_d = ti.simt.block.SharedArray((_T,), gs.qd_float)

        if constraint_state.n_constraints[i_b] > 0:
            n_dofs = constraint_state.search.shape[0]

            # === Phase 1: Fused snorm + quad_gauss, parallel over n_dofs ===
            local_snorm_sq = gs.qd_float(0.0)
            local_qg1 = gs.qd_float(0.0)
            local_qg2 = gs.qd_float(0.0)

            i_d = tid
            while i_d < n_dofs:
                s = constraint_state.search[i_d, i_b]
                local_snorm_sq += s * s
                local_qg1 += s * constraint_state.Ma[i_d, i_b] - s * dofs_state.force[i_d, i_b]
                local_qg2 += 0.5 * s * constraint_state.mv[i_d, i_b]
                i_d += _T

            sh_a[tid] = local_snorm_sq
            sh_b[tid] = local_qg1
            sh_c[tid] = local_qg2

            ti.simt.block.sync()

            # Tree reduction for 3 accumulators
            stride = _T // 2
            while stride > 0:
                if tid < stride:
                    sh_a[tid] += sh_a[tid + stride]
                    sh_b[tid] += sh_b[tid + stride]
                    sh_c[tid] += sh_c[tid + stride]
                ti.simt.block.sync()
                stride //= 2

            # All threads read the reduced snorm
            snorm = ti.sqrt(sh_a[0])

            if snorm < rigid_global_info.EPS[None]:
                # Converged — only thread 0 writes
                if tid == 0:
                    constraint_state.candidates[0, i_b] = 0.0
                    constraint_state.candidates[1, i_b] = 0.0
                    constraint_state.improved[i_b] = False
            else:
                # Thread 0 writes quad_gauss to global memory
                if tid == 0:
                    constraint_state.improved[i_b] = True
                    constraint_state.quad_gauss[0, i_b] = constraint_state.gauss[i_b]
                    constraint_state.quad_gauss[1, i_b] = sh_b[0]
                    constraint_state.quad_gauss[2, i_b] = sh_c[0]

                # === Phase 2: Constraint cost, parallel over n_constraints ===
                ne = constraint_state.n_constraints_equality[i_b]
                nef = ne + constraint_state.n_constraints_frictionloss[i_b]
                n_con = constraint_state.n_constraints[i_b]

                local_eq0 = gs.qd_float(0.0)
                local_eq1 = gs.qd_float(0.0)
                local_eq2 = gs.qd_float(0.0)
                local_tmp0 = gs.qd_float(0.0)

                i_c = tid
                while i_c < n_con:
                    Jaref_c = constraint_state.Jaref[i_c, i_b]
                    D = constraint_state.efc_D[i_c, i_b]
                    qf_0 = D * (0.5 * Jaref_c * Jaref_c)

                    if i_c < ne:
                        # Equality: always active, need jv for eq_sum
                        jv_c = constraint_state.jv[i_c, i_b]
                        qf_1 = D * (jv_c * Jaref_c)
                        qf_2 = D * (0.5 * jv_c * jv_c)
                        local_eq0 += qf_0
                        local_eq1 += qf_1
                        local_eq2 += qf_2
                        local_tmp0 += qf_0
                    elif i_c < nef:
                        # Friction: only qf_0 needed (qf_1/qf_2 not stored)
                        f = constraint_state.efc_frictionloss[i_c, i_b]
                        r = constraint_state.diag[i_c, i_b]
                        rf = r * f
                        linear_neg = Jaref_c <= -rf
                        linear_pos = Jaref_c >= rf
                        if linear_neg or linear_pos:
                            qf_0 = linear_neg * f * (-0.5 * rf - Jaref_c) + linear_pos * f * (-0.5 * rf + Jaref_c)
                        local_tmp0 += qf_0
                    else:
                        # Contact: active if Jaref < 0
                        active = Jaref_c < 0
                        local_tmp0 += qf_0 * active

                    i_c += _T

                # Reuse shared arrays for Phase 2 reduction
                sh_a[tid] = local_eq0
                sh_b[tid] = local_eq1
                sh_c[tid] = local_eq2
                sh_d[tid] = local_tmp0

                ti.simt.block.sync()

                # Tree reduction for 4 accumulators
                stride = _T // 2
                while stride > 0:
                    if tid < stride:
                        sh_a[tid] += sh_a[tid + stride]
                        sh_b[tid] += sh_b[tid + stride]
                        sh_c[tid] += sh_c[tid + stride]
                        sh_d[tid] += sh_d[tid + stride]
                    ti.simt.block.sync()
                    stride //= 2

                if tid == 0:
                    constraint_state.eq_sum[0, i_b] = sh_a[0]
                    constraint_state.eq_sum[1, i_b] = sh_b[0]
                    constraint_state.eq_sum[2, i_b] = sh_c[0]
                    constraint_state.ls_it[i_b] = 1
                    constraint_state.candidates[1, i_b] = constraint_state.gauss[i_b] + sh_d[0]


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_parallel_linesearch_eval(
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    """Evaluate K candidate alphas in parallel per env, pick the best via reduction."""
    _B = constraint_state.grad.shape[1]
    _K = ti.static(LS_PARALLEL_K)
    _MIN_STEP = ti.static(LS_PARALLEL_MIN_STEP)

    ti.loop_config(block_dim=_K)
    for i_ in range(_B * _K):
        tid = i_ % _K
        i_b = i_ // _K

        # Shared memory for argmin reduction
        sh_cost = ti.simt.block.SharedArray((_K,), gs.qd_float)
        sh_idx = ti.simt.block.SharedArray((_K,), ti.i32)

        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            ne = constraint_state.n_constraints_equality[i_b]
            nef = ne + constraint_state.n_constraints_frictionloss[i_b]
            n_con = constraint_state.n_constraints[i_b]

            # Generate log-spaced alpha: alpha[0]=MIN_STEP ... alpha[K-1]=1.0
            alpha = solver._log_scale(_MIN_STEP, 1.0, _K, tid)

            # Evaluate cost at this alpha
            cost = (
                alpha * alpha * constraint_state.quad_gauss[2, i_b]
                + alpha * constraint_state.quad_gauss[1, i_b]
                + constraint_state.quad_gauss[0, i_b]
            )

            # Equality constraints (always active) - use eq_sum precomputed during init
            cost = (
                cost
                + alpha * alpha * constraint_state.eq_sum[2, i_b]
                + alpha * constraint_state.eq_sum[1, i_b]
                + constraint_state.eq_sum[0, i_b]
            )

            # Friction constraints
            for i_c in range(ne, nef):
                Jaref_c = constraint_state.Jaref[i_c, i_b]
                jv_c = constraint_state.jv[i_c, i_b]
                D = constraint_state.efc_D[i_c, i_b]
                f = constraint_state.efc_frictionloss[i_c, i_b]
                r = constraint_state.diag[i_c, i_b]
                x = Jaref_c + alpha * jv_c
                rf = r * f
                linear_neg = x <= -rf
                linear_pos = x >= rf
                if linear_neg or linear_pos:
                    cost = cost + linear_neg * f * (-0.5 * rf - Jaref_c - alpha * jv_c)
                    cost = cost + linear_pos * f * (-0.5 * rf + Jaref_c + alpha * jv_c)
                else:
                    cost = cost + D * 0.5 * x * x

            # Contact constraints (active if x < 0)
            for i_c in range(nef, n_con):
                Jaref_c = constraint_state.Jaref[i_c, i_b]
                jv_c = constraint_state.jv[i_c, i_b]
                D = constraint_state.efc_D[i_c, i_b]
                x = Jaref_c + alpha * jv_c
                if x < 0:
                    cost += D * 0.5 * x * x

            sh_cost[tid] = cost
            sh_idx[tid] = tid
        else:
            sh_cost[tid] = gs.qd_float(1e30)
            sh_idx[tid] = tid

        ti.simt.block.sync()

        # Tree reduction for argmin
        stride = _K // 2
        while stride > 0:
            if tid < stride:
                if sh_cost[tid + stride] < sh_cost[tid]:
                    sh_cost[tid] = sh_cost[tid + stride]
                    sh_idx[tid] = sh_idx[tid + stride]
            ti.simt.block.sync()
            stride = stride // 2

        # Thread 0: acceptance check and write result
        if tid == 0:
            if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
                p0_cost = constraint_state.candidates[1, i_b]
                best_tid = sh_idx[0]
                best_cost = sh_cost[0]
                best_alpha = solver._log_scale(_MIN_STEP, 1.0, _K, best_tid)
                if best_cost < p0_cost:
                    constraint_state.candidates[0, i_b] = best_alpha
                else:
                    constraint_state.candidates[0, i_b] = 0.0
            else:
                constraint_state.candidates[0, i_b] = 0.0


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_parallel_linesearch_apply_alpha_dofs(
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    """Apply best alpha to qacc and Ma, parallelized over (dof, env)."""
    n_dofs = constraint_state.qacc.shape[0]
    _B = constraint_state.grad.shape[1]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in ti.ndrange(n_dofs, _B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            alpha = constraint_state.candidates[0, i_b]
            if ti.abs(alpha) < rigid_global_info.EPS[None]:
                if i_d == 0:
                    constraint_state.improved[i_b] = False
            else:
                constraint_state.qacc[i_d, i_b] += constraint_state.search[i_d, i_b] * alpha
                constraint_state.Ma[i_d, i_b] += constraint_state.mv[i_d, i_b] * alpha


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_parallel_linesearch_apply_alpha_constraints(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
):
    """Apply best alpha to Jaref, parallelized over (constraint, env)."""
    len_constraints = constraint_state.Jaref.shape[0]
    _B = constraint_state.grad.shape[1]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_c, i_b in ti.ndrange(len_constraints, _B):
        if i_c < constraint_state.n_constraints[i_b] and constraint_state.improved[i_b]:
            alpha = constraint_state.candidates[0, i_b]
            constraint_state.Jaref[i_c, i_b] += constraint_state.jv[i_c, i_b] * alpha


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_cg_only_save_prev_grad(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
):
    """Save prev_grad and prev_Mgrad (CG only)"""
    _B = constraint_state.grad.shape[1]
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            solver.func_save_prev_grad(i_b, constraint_state=constraint_state)


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_update_constraint(
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    _B = constraint_state.grad.shape[1]
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            solver.func_update_constraint_batch(
                i_b,
                qacc=constraint_state.qacc,
                Ma=constraint_state.Ma,
                cost=constraint_state.cost,
                dofs_state=dofs_state,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_update_constraint_forces(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
):
    """Compute active flags and efc_force, parallelized over (constraint, env)."""
    len_constraints = constraint_state.active.shape[0]
    _B = constraint_state.grad.shape[1]

    for i_c, i_b in ti.ndrange(len_constraints, _B):
        if i_c < constraint_state.n_constraints[i_b] and constraint_state.improved[i_b]:
            ne = constraint_state.n_constraints_equality[i_b]
            nef = ne + constraint_state.n_constraints_frictionloss[i_b]

            if ti.static(static_rigid_sim_config.solver_type == gs.constraint_solver.Newton):
                constraint_state.prev_active[i_c, i_b] = constraint_state.active[i_c, i_b]

            constraint_state.active[i_c, i_b] = True
            floss_force = gs.qd_float(0.0)

            if ne <= i_c and i_c < nef:
                f = constraint_state.efc_frictionloss[i_c, i_b]
                r = constraint_state.diag[i_c, i_b]
                rf = r * f
                linear_neg = constraint_state.Jaref[i_c, i_b] <= -rf
                linear_pos = constraint_state.Jaref[i_c, i_b] >= rf
                constraint_state.active[i_c, i_b] = not (linear_neg or linear_pos)
                floss_force = linear_neg * f + linear_pos * -f
            elif nef <= i_c:
                constraint_state.active[i_c, i_b] = constraint_state.Jaref[i_c, i_b] < 0

            constraint_state.efc_force[i_c, i_b] = floss_force + (
                -constraint_state.Jaref[i_c, i_b] * constraint_state.efc_D[i_c, i_b] * constraint_state.active[i_c, i_b]
            )


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_update_constraint_qfrc(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
):
    """Compute qfrc_constraint = J^T @ efc_force, parallelized over (dof, env)."""
    n_dofs = constraint_state.qfrc_constraint.shape[0]
    _B = constraint_state.grad.shape[1]

    for i_d, i_b in ti.ndrange(n_dofs, _B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            n_con = constraint_state.n_constraints[i_b]
            qfrc = gs.qd_float(0.0)
            for i_c in range(n_con):
                qfrc += constraint_state.jac[i_c, i_d, i_b] * constraint_state.efc_force[i_c, i_b]
            constraint_state.qfrc_constraint[i_d, i_b] = qfrc


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_update_constraint_cost(
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
):
    """Compute gauss and cost (reductions over dofs and constraints). One thread per env."""
    _B = constraint_state.grad.shape[1]

    ti.loop_config(block_dim=32)
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            n_dofs = constraint_state.qfrc_constraint.shape[0]
            ne = constraint_state.n_constraints_equality[i_b]
            nef = ne + constraint_state.n_constraints_frictionloss[i_b]
            n_con = constraint_state.n_constraints[i_b]

            constraint_state.prev_cost[i_b] = constraint_state.cost[i_b]

            cost_i = gs.qd_float(0.0)
            gauss_i = gs.qd_float(0.0)

            # Gauss cost from dofs
            for i_d in range(n_dofs):
                v = (
                    0.5
                    * (constraint_state.Ma[i_d, i_b] - dofs_state.force[i_d, i_b])
                    * (constraint_state.qacc[i_d, i_b] - dofs_state.acc_smooth[i_d, i_b])
                )
                gauss_i += v
                cost_i += v

            # Constraint cost: quadratic + friction linear
            for i_c in range(n_con):
                cost_i += 0.5 * (
                    constraint_state.Jaref[i_c, i_b] ** 2
                    * constraint_state.efc_D[i_c, i_b]
                    * constraint_state.active[i_c, i_b]
                )
                if ne <= i_c and i_c < nef:
                    f = constraint_state.efc_frictionloss[i_c, i_b]
                    r = constraint_state.diag[i_c, i_b]
                    rf = r * f
                    linear_neg = constraint_state.Jaref[i_c, i_b] <= -rf
                    linear_pos = constraint_state.Jaref[i_c, i_b] >= rf
                    cost_i += linear_neg * f * (-0.5 * rf - constraint_state.Jaref[i_c, i_b]) + linear_pos * f * (
                        -0.5 * rf + constraint_state.Jaref[i_c, i_b]
                    )

            constraint_state.gauss[i_b] = gauss_i
            constraint_state.cost[i_b] = cost_i


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_newton_only_nt_hessian_incremental(
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    """Step 4: Newton Hessian update (Newton only)"""
    solver.func_hessian_direct_tiled(constraint_state=constraint_state, rigid_global_info=rigid_global_info)
    if ti.static(static_rigid_sim_config.enable_tiled_cholesky_hessian):
        solver.func_cholesky_factor_direct_tiled(
            constraint_state=constraint_state,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )
    else:
        _B = constraint_state.jac.shape[2]
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
        for i_b in range(_B):
            if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
                solver.func_cholesky_factor_direct_batch(
                    i_b=i_b, constraint_state=constraint_state, rigid_global_info=rigid_global_info
                )


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_update_gradient(
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    """Step 5: Update gradient"""
    _B = constraint_state.grad.shape[1]
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            solver.func_update_gradient_batch(
                i_b,
                dofs_state=dofs_state,
                entities_info=entities_info,
                rigid_global_info=rigid_global_info,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_update_search_direction(
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    """Step 6: Check convergence and update search direction"""
    _B = constraint_state.grad.shape[1]
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            solver.func_terminate_or_update_descent_batch(
                i_b,
                rigid_global_info=rigid_global_info,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )


# ================================================ Init kernels ================================================


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_init_warmstart(
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
):
    """Select qacc from warmstart or acc_smooth, parallelized over (dof, env)."""
    n_dofs = dofs_state.acc_smooth.shape[0]
    _B = dofs_state.acc_smooth.shape[1]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in ti.ndrange(n_dofs, _B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.is_warmstart[i_b]:
            constraint_state.qacc[i_d, i_b] = constraint_state.qacc_ws[i_d, i_b]
        else:
            constraint_state.qacc[i_d, i_b] = dofs_state.acc_smooth[i_d, i_b]


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_init_Ma(
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    """Compute Ma = M @ qacc, parallelized over (dof, env)."""
    solver.initialize_Ma(
        Ma=constraint_state.Ma,
        qacc=constraint_state.qacc,
        dofs_info=dofs_info,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
    )


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_init_Jaref(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
):
    """Compute Jaref = -aref + J @ qacc, parallelized over (constraint, env)."""
    len_constraints = constraint_state.Jaref.shape[0]
    n_dofs = constraint_state.jac.shape[1]
    _B = constraint_state.grad.shape[1]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_c, i_b in ti.ndrange(len_constraints, _B):
        if i_c < constraint_state.n_constraints[i_b]:
            Jaref = -constraint_state.aref[i_c, i_b]
            if ti.static(static_rigid_sim_config.sparse_solve):
                for i_d_ in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                    i_d = constraint_state.jac_relevant_dofs[i_c, i_d_, i_b]
                    Jaref += constraint_state.jac[i_c, i_d, i_b] * constraint_state.qacc[i_d, i_b]
            else:
                for i_d in range(n_dofs):
                    Jaref += constraint_state.jac[i_c, i_d, i_b] * constraint_state.qacc[i_d, i_b]
            constraint_state.Jaref[i_c, i_b] = Jaref


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_init_improved(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
):
    """Set improved = (n_constraints > 0) for each env."""
    _B = constraint_state.grad.shape[1]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        constraint_state.improved[i_b] = constraint_state.n_constraints[i_b] > 0


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_init_search(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
):
    """Set search = -Mgrad, parallelized over (dof, env)."""
    n_dofs = constraint_state.search.shape[0]
    _B = constraint_state.grad.shape[1]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in ti.ndrange(n_dofs, _B):
        constraint_state.search[i_d, i_b] = -constraint_state.Mgrad[i_d, i_b]


def func_solve_init_decomposed(
    dofs_info,
    dofs_state,
    entities_info,
    constraint_state,
    rigid_global_info,
    static_rigid_sim_config,
):
    """
    Decomposed version of func_solve_init for CUDA backend (non-mujoco path).

    Breaks the monolithic init kernel into separate kernel launches:
    1. Warmstart selection (ndrange over dofs)
    2. Ma = M @ qacc (ndrange over dofs with entity lookup)
    3. Jaref = -aref + J @ qacc (ndrange over constraints — main optimization)
    4. Set improved flags
    5. Update constraint (forces / qfrc / cost — reuse decomposed kernels)
    6. Newton hessian (Newton only — reuse existing kernel)
    7. Update gradient (reuse existing kernel)
    8. search = -Mgrad (ndrange over dofs)
    """
    # 1. Warmstart selection
    _kernel_init_warmstart(dofs_state, constraint_state, static_rigid_sim_config)

    # 2. Ma = M @ qacc
    _kernel_init_Ma(dofs_info, entities_info, constraint_state, rigid_global_info, static_rigid_sim_config)

    # 3. Jaref = -aref + J @ qacc (parallelized over constraints)
    _kernel_init_Jaref(constraint_state, static_rigid_sim_config)

    # 4. Set improved flags (needed by decomposed update_constraint kernels)
    _kernel_init_improved(constraint_state, static_rigid_sim_config)

    # 5. Update constraint (reuse decomposed kernels)
    _kernel_update_constraint_forces(constraint_state, static_rigid_sim_config)
    _kernel_update_constraint_qfrc(constraint_state, static_rigid_sim_config)
    _kernel_update_constraint_cost(dofs_state, constraint_state, static_rigid_sim_config)

    # 6. Newton hessian (Newton only)
    if static_rigid_sim_config.solver_type == gs.constraint_solver.Newton:
        _kernel_newton_only_nt_hessian_incremental(
            entities_info, constraint_state, rigid_global_info, static_rigid_sim_config
        )

    # 7. Update gradient
    _kernel_update_gradient(entities_info, dofs_state, constraint_state, rigid_global_info, static_rigid_sim_config)

    # 8. search = -Mgrad
    _kernel_init_search(constraint_state, static_rigid_sim_config)


def func_solve_decomposed_macrokernels(
    entities_info,
    dofs_info,
    dofs_state,
    constraint_state,
    rigid_global_info,
    static_rigid_sim_config,
    use_parallel_ls=False,
):
    """
    Uses separate kernels for each solver step per iteration.

    This maximizes kernel granularity, potentially allowing better GPU scheduling
    and more flexibility in execution, at the cost of more Python→C++ boundary crossings.
    """
    iterations = rigid_global_info.iterations[None]
    for _it in range(iterations):
        if use_parallel_ls:
            _kernel_parallel_linesearch_mv(
                dofs_info,
                entities_info,
                constraint_state,
                rigid_global_info,
                static_rigid_sim_config,
            )
            _kernel_parallel_linesearch_jv(
                constraint_state,
                static_rigid_sim_config,
            )
            _kernel_parallel_linesearch_p0(
                dofs_state,
                constraint_state,
                rigid_global_info,
                static_rigid_sim_config,
            )
            _kernel_parallel_linesearch_eval(
                constraint_state,
                rigid_global_info,
                static_rigid_sim_config,
            )
            _kernel_parallel_linesearch_apply_alpha_dofs(
                constraint_state,
                rigid_global_info,
                static_rigid_sim_config,
            )
            _kernel_parallel_linesearch_apply_alpha_constraints(
                constraint_state,
                static_rigid_sim_config,
            )
        else:
            _kernel_linesearch(
                entities_info,
                dofs_state,
                constraint_state,
                rigid_global_info,
                static_rigid_sim_config,
            )
        if static_rigid_sim_config.solver_type == gs.constraint_solver.CG:
            _kernel_cg_only_save_prev_grad(
                constraint_state,
                static_rigid_sim_config,
            )
        if use_parallel_ls:
            _kernel_update_constraint_forces(
                constraint_state,
                static_rigid_sim_config,
            )
            _kernel_update_constraint_qfrc(
                constraint_state,
                static_rigid_sim_config,
            )
            _kernel_update_constraint_cost(
                dofs_state,
                constraint_state,
                static_rigid_sim_config,
            )
        else:
            _kernel_update_constraint(
                entities_info,
                dofs_state,
                constraint_state,
                rigid_global_info,
                static_rigid_sim_config,
            )
        if static_rigid_sim_config.solver_type == gs.constraint_solver.Newton:
            _kernel_newton_only_nt_hessian_incremental(
                entities_info,
                constraint_state,
                rigid_global_info,
                static_rigid_sim_config,
            )
        _kernel_update_gradient(
            entities_info,
            dofs_state,
            constraint_state,
            rigid_global_info,
            static_rigid_sim_config,
        )
        _kernel_update_search_direction(
            constraint_state,
            rigid_global_info,
            static_rigid_sim_config,
        )
