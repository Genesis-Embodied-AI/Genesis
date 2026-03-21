import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.solvers.rigid.constraint import solver

# --- Parallel linesearch constants ---
# Number of candidate step sizes evaluated simultaneously per env.
# Each CUDA block processes one env with K threads, using shared memory for the argmin reduction.
# Similar to BLOCK_DIM in func_hessian_direct_tiled: determines parallelism and shared memory layout.
LS_PARALLEL_K = 32

# Floor for the Newton step estimate used to center the log-spaced search range.
# When |grad/hess| is near-zero the search range [alpha*1e-2, alpha*1e2] would collapse;
# this clamp keeps the range meaningful. The value is well below typical linesearch tolerances
# (ls_tolerance * tolerance ~ 1e-2 * 1e-8 for double, ~ 1e-2 * 1e-5 for float) so it never
# masks a genuinely small optimal step.
LS_PARALLEL_MIN_STEP = 1e-6

# Number of successive refinement passes: after picking the best of K candidates the search
# range is narrowed around the winner and re-evaluated. 1 pass (K=16 candidates) already
# gives sufficient resolution; increase for tighter convergence at the cost of more kernels.
LS_PARALLEL_N_REFINE = 3

# Block sizes for shared-memory reductions in _kernel_parallel_linesearch_p0 and _jv.
_P0_BLOCK = 32
_JV_BLOCK = 32


@qd.kernel(fastcache=gs.use_fastcache)
def _kernel_parallel_linesearch_mv(
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Compute mv = M @ search, parallelized over (dof, env).

    Uses per-dof entity lookup to find the entity block boundaries, giving n_dofs * B
    threads (each computing a single ~6-element dot product) instead of n_entities * B
    threads (each computing the full block matvec).
    """
    n_dofs = constraint_state.search.shape[0]
    _B = constraint_state.grad.shape[1]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d1, i_b in qd.ndrange(n_dofs, _B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            I_d1 = [i_d1, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else i_d1
            i_e = dofs_info.entity_idx[I_d1]
            mv = gs.qd_float(0.0)
            for i_d2 in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                mv = mv + rigid_global_info.mass_mat[i_d1, i_d2, i_b] * constraint_state.search[i_d2, i_b]
            constraint_state.mv[i_d1, i_b] = mv


@qd.kernel(fastcache=gs.use_fastcache)
def _kernel_parallel_linesearch_jv(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """Compute jv = J @ search, parallelized over (constraint, env)."""
    n_dofs = constraint_state.search.shape[0]
    len_constraints = constraint_state.jac.shape[0]
    _B = constraint_state.grad.shape[1]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_c, i_b in qd.ndrange(len_constraints, _B):
        if i_c < constraint_state.n_constraints[i_b] and constraint_state.improved[i_b]:
            jv = gs.qd_float(0.0)
            if qd.static(static_rigid_sim_config.sparse_solve):
                for i_d_ in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                    i_d = constraint_state.jac_relevant_dofs[i_c, i_d_, i_b]
                    jv = jv + constraint_state.jac[i_c, i_d, i_b] * constraint_state.search[i_d, i_b]
            else:
                for i_d in range(n_dofs):
                    jv = jv + constraint_state.jac[i_c, i_d, i_b] * constraint_state.search[i_d, i_b]
            constraint_state.jv[i_c, i_b] = jv


@qd.kernel(fastcache=gs.use_fastcache)
def _kernel_parallel_linesearch_p0(
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Snorm check, quad_gauss, eq_sum, and p0_cost. T threads per env with shared memory reductions.

    Phase 1: Fused snorm + quad_gauss parallel reduction over n_dofs (Options A+B).
    Phase 2: Parallel reduction over n_constraints for eq_sum and p0_cost.
    """
    _B = constraint_state.grad.shape[1]
    _T = qd.static(_P0_BLOCK)

    qd.loop_config(block_dim=_T)
    for i_flat in range(_B * _T):
        tid = i_flat % _T
        i_b = i_flat // _T

        # 6 shared arrays for parallel reductions (reused across phases)
        sh_snorm_sq = qd.simt.block.SharedArray((_T,), gs.qd_float)
        sh_qg_grad = qd.simt.block.SharedArray((_T,), gs.qd_float)
        sh_qg_hess = qd.simt.block.SharedArray((_T,), gs.qd_float)
        sh_p0_cost = qd.simt.block.SharedArray((_T,), gs.qd_float)
        sh_constraint_grad = qd.simt.block.SharedArray((_T,), gs.qd_float)
        sh_constraint_hess = qd.simt.block.SharedArray((_T,), gs.qd_float)

        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            n_dofs = constraint_state.search.shape[0]

            # === Phase 1: Fused snorm + quad_gauss, parallel over n_dofs ===
            local_snorm_sq = gs.qd_float(0.0)
            local_qg_grad = gs.qd_float(0.0)
            local_qg_hess = gs.qd_float(0.0)

            i_d = tid
            while i_d < n_dofs:
                s = constraint_state.search[i_d, i_b]
                local_snorm_sq += s * s
                local_qg_grad += s * constraint_state.Ma[i_d, i_b] - s * dofs_state.force[i_d, i_b]
                local_qg_hess += 0.5 * s * constraint_state.mv[i_d, i_b]
                i_d += _T

            sh_snorm_sq[tid] = local_snorm_sq
            sh_qg_grad[tid] = local_qg_grad
            sh_qg_hess[tid] = local_qg_hess

            qd.simt.block.sync()

            # Tree reduction for 3 accumulators
            stride = _T // 2
            while stride > 0:
                if tid < stride:
                    sh_snorm_sq[tid] += sh_snorm_sq[tid + stride]
                    sh_qg_grad[tid] += sh_qg_grad[tid + stride]
                    sh_qg_hess[tid] += sh_qg_hess[tid + stride]
                qd.simt.block.sync()
                stride //= 2

            # All threads read the reduced snorm
            snorm = qd.sqrt(sh_snorm_sq[0])

            if snorm < rigid_global_info.EPS[None]:
                # Converged — only thread 0 writes
                if tid == 0:
                    constraint_state.candidates[0, i_b] = 0.0
                    constraint_state.candidates[1, i_b] = 0.0
                    constraint_state.improved[i_b] = False
            else:
                # Thread 0 writes quad_gauss to global memory
                if tid == 0:
                    constraint_state.quad_gauss[0, i_b] = constraint_state.gauss[i_b]
                    constraint_state.quad_gauss[1, i_b] = sh_qg_grad[0]
                    constraint_state.quad_gauss[2, i_b] = sh_qg_hess[0]

                # === Phase 2: Constraint cost, parallel over n_constraints ===
                ne = constraint_state.n_constraints_equality[i_b]
                nef = ne + constraint_state.n_constraints_frictionloss[i_b]
                n_con = constraint_state.n_constraints[i_b]

                local_eq_cost = gs.qd_float(0.0)
                local_eq_grad = gs.qd_float(0.0)
                local_eq_hess = gs.qd_float(0.0)
                local_p0_cost = gs.qd_float(0.0)
                local_constraint_grad = gs.qd_float(0.0)
                local_constraint_hess = gs.qd_float(0.0)

                i_c = tid
                while i_c < n_con:
                    Jaref_c = constraint_state.Jaref[i_c, i_b]
                    jv_c = constraint_state.jv[i_c, i_b]
                    D = constraint_state.efc_D[i_c, i_b]
                    qf_0 = D * (0.5 * Jaref_c * Jaref_c)
                    qf_1 = D * (jv_c * Jaref_c)
                    qf_2 = D * (0.5 * jv_c * jv_c)

                    if i_c < ne:
                        # Equality: always active
                        local_eq_cost += qf_0
                        local_eq_grad += qf_1
                        local_eq_hess += qf_2
                        local_p0_cost += qf_0
                        local_constraint_grad += qf_1
                        local_constraint_hess += qf_2
                    elif i_c < nef:
                        # Friction: check linear regime at alpha=0
                        f = constraint_state.efc_frictionloss[i_c, i_b]
                        r = constraint_state.diag[i_c, i_b]
                        rf = r * f
                        linear_neg = Jaref_c <= -rf
                        linear_pos = Jaref_c >= rf
                        if linear_neg or linear_pos:
                            qf_0 = linear_neg * f * (-0.5 * rf - Jaref_c) + linear_pos * f * (-0.5 * rf + Jaref_c)
                            qf_1 = linear_neg * (-f * jv_c) + linear_pos * (f * jv_c)
                            qf_2 = 0.0
                        local_p0_cost += qf_0
                        local_constraint_grad += qf_1
                        local_constraint_hess += qf_2
                    else:
                        # Contact: active if Jaref < 0
                        active = Jaref_c < 0
                        local_p0_cost += qf_0 * active
                        local_constraint_grad += qf_1 * active
                        local_constraint_hess += qf_2 * active

                    i_c += _T

                # Reuse shared arrays for Phase 2 reduction
                sh_snorm_sq[tid] = local_eq_cost
                sh_qg_grad[tid] = local_eq_grad
                sh_qg_hess[tid] = local_eq_hess
                sh_p0_cost[tid] = local_p0_cost
                sh_constraint_grad[tid] = local_constraint_grad
                sh_constraint_hess[tid] = local_constraint_hess

                qd.simt.block.sync()

                # Tree reduction for 6 accumulators
                stride = _T // 2
                while stride > 0:
                    if tid < stride:
                        sh_snorm_sq[tid] += sh_snorm_sq[tid + stride]
                        sh_qg_grad[tid] += sh_qg_grad[tid + stride]
                        sh_qg_hess[tid] += sh_qg_hess[tid + stride]
                        sh_p0_cost[tid] += sh_p0_cost[tid + stride]
                        sh_constraint_grad[tid] += sh_constraint_grad[tid + stride]
                        sh_constraint_hess[tid] += sh_constraint_hess[tid + stride]
                    qd.simt.block.sync()
                    stride //= 2

                if tid == 0:
                    constraint_state.eq_sum[0, i_b] = sh_snorm_sq[0]
                    constraint_state.eq_sum[1, i_b] = sh_qg_grad[0]
                    constraint_state.eq_sum[2, i_b] = sh_qg_hess[0]
                    constraint_state.ls_it[i_b] = 1
                    constraint_state.candidates[1, i_b] = constraint_state.gauss[i_b] + sh_p0_cost[0]
                    # Use full Newton step as initial best guess + search range center.
                    total_hess = 2.0 * (constraint_state.quad_gauss[2, i_b] + sh_constraint_hess[0])
                    total_grad = constraint_state.quad_gauss[1, i_b] + sh_constraint_grad[0]
                    p0_cost_val = constraint_state.gauss[i_b] + sh_p0_cost[0]

                    if total_hess > 0.0:
                        alpha_newton = qd.max(qd.abs(total_grad / total_hess), gs.qd_float(LS_PARALLEL_MIN_STEP))
                        # Initialize with Newton step as the best alpha so far.
                        # Its quadratic-approximated cost is p0 - grad²/(2*hess), always < p0.
                        newton_cost = p0_cost_val - 0.5 * total_grad * total_grad / total_hess
                        constraint_state.candidates[0, i_b] = alpha_newton
                        constraint_state.candidates[4, i_b] = newton_cost
                        constraint_state.candidates[2, i_b] = alpha_newton * 1e-2
                        constraint_state.candidates[3, i_b] = alpha_newton * 10.0
                    else:
                        constraint_state.candidates[0, i_b] = 0.0
                        constraint_state.candidates[4, i_b] = gs.qd_float(1e30)
                        constraint_state.candidates[2, i_b] = 1e-6
                        constraint_state.candidates[3, i_b] = 1e2


@qd.kernel(fastcache=gs.use_fastcache)
def _kernel_parallel_linesearch_eval(
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Evaluate K candidate alphas in parallel per env, pick the best via reduction.

    Reads the search range from candidates[2] (lo) and candidates[3] (hi).
    Writes narrowed range back to candidates[2,3] for successive refinement.
    """
    _B = constraint_state.grad.shape[1]
    _K = qd.static(LS_PARALLEL_K)

    qd.loop_config(block_dim=_K)
    for i_flat in range(_B * _K):
        tid = i_flat % _K
        i_b = i_flat // _K

        # Shared memory for argmin reduction
        sh_cost = qd.simt.block.SharedArray((_K,), gs.qd_float)
        sh_idx = qd.simt.block.SharedArray((_K,), qd.i32)

        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            ne = constraint_state.n_constraints_equality[i_b]
            nef = ne + constraint_state.n_constraints_frictionloss[i_b]
            n_con = constraint_state.n_constraints[i_b]

            lo = constraint_state.candidates[2, i_b]
            hi = constraint_state.candidates[3, i_b]

            # Generate log-spaced alpha within [lo, hi]
            _step = (qd.log(hi) - qd.log(lo)) / qd.max(1.0, qd.cast(_K - 1, gs.qd_float))
            alpha = qd.exp(qd.log(lo) + qd.cast(tid, gs.qd_float) * _step)

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

        qd.simt.block.sync()

        # Tree reduction for argmin
        stride = _K // 2
        while stride > 0:
            if tid < stride:
                if sh_cost[tid + stride] < sh_cost[tid]:
                    sh_cost[tid] = sh_cost[tid + stride]
                    sh_idx[tid] = sh_idx[tid + stride]
            qd.simt.block.sync()
            stride = stride // 2

        # Thread 0: acceptance check, write result, and narrow range for next pass
        if tid == 0:
            if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
                p0_cost = constraint_state.candidates[1, i_b]
                best_tid = sh_idx[0]
                best_cost = sh_cost[0]
                lo = constraint_state.candidates[2, i_b]
                hi = constraint_state.candidates[3, i_b]
                # Recover best alpha from log-spaced grid
                _step = (qd.log(hi) - qd.log(lo)) / qd.max(1.0, qd.cast(_K - 1, gs.qd_float))
                best_alpha = qd.exp(qd.log(lo) + qd.cast(best_tid, gs.qd_float) * _step)

                # Only update best alpha if this pass improved over ALL previous passes
                best_cost_prev = constraint_state.candidates[4, i_b]
                if best_cost < p0_cost and best_cost < best_cost_prev:
                    constraint_state.candidates[0, i_b] = best_alpha
                    constraint_state.candidates[4, i_b] = best_cost

                    # Narrow range around accepted point for next refinement pass
                    lo_idx = qd.max(0, best_tid - 1)
                    hi_idx = qd.min(_K - 1, best_tid + 1)
                    # Narrow search range to neighbors of best candidate
                    _step = (qd.log(hi) - qd.log(lo)) / qd.max(1.0, qd.cast(_K - 1, gs.qd_float))
                    constraint_state.candidates[2, i_b] = qd.exp(qd.log(lo) + qd.cast(lo_idx, gs.qd_float) * _step)
                    constraint_state.candidates[3, i_b] = qd.exp(qd.log(lo) + qd.cast(hi_idx, gs.qd_float) * _step)
            else:
                constraint_state.candidates[0, i_b] = 0.0


@qd.kernel(fastcache=gs.use_fastcache)
def _kernel_parallel_linesearch_apply_alpha(
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Apply best alpha to qacc, Ma, and Jaref. Fuses dof and constraint updates."""
    n_dofs = constraint_state.qacc.shape[0]
    len_constraints = constraint_state.Jaref.shape[0]
    _B = constraint_state.grad.shape[1]
    n_items = qd.max(n_dofs, len_constraints)

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i, i_b in qd.ndrange(n_items, _B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            alpha = constraint_state.candidates[0, i_b]
            if qd.abs(alpha) < rigid_global_info.EPS[None]:
                if i == 0:
                    constraint_state.improved[i_b] = False
            else:
                # Apply to dofs
                if i < n_dofs:
                    constraint_state.qacc[i, i_b] += constraint_state.search[i, i_b] * alpha
                    constraint_state.Ma[i, i_b] += constraint_state.mv[i, i_b] * alpha
                # Apply to constraints
                if i < constraint_state.n_constraints[i_b]:
                    constraint_state.Jaref[i, i_b] += constraint_state.jv[i, i_b] * alpha


# ============================================== Shared iteration kernels ==============================================


@qd.kernel(fastcache=gs.use_fastcache)
def _kernel_cg_only_save_prev_grad(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """Save prev_grad and prev_Mgrad (CG only)"""
    _B = constraint_state.grad.shape[1]
    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            solver.func_save_prev_grad(i_b, constraint_state=constraint_state)


@qd.kernel(fastcache=gs.use_fastcache)
def _kernel_update_constraint_forces(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """Compute active flags and efc_force, parallelized over (constraint, env)."""
    len_constraints = constraint_state.active.shape[0]
    _B = constraint_state.grad.shape[1]

    for i_c, i_b in qd.ndrange(len_constraints, _B):
        if i_c < constraint_state.n_constraints[i_b] and constraint_state.improved[i_b]:
            ne = constraint_state.n_constraints_equality[i_b]
            nef = ne + constraint_state.n_constraints_frictionloss[i_b]

            if qd.static(static_rigid_sim_config.solver_type == gs.constraint_solver.Newton):
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


@qd.kernel(fastcache=gs.use_fastcache)
def _kernel_update_constraint_qfrc(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """Compute qfrc_constraint = J^T @ efc_force, parallelized over (dof, env)."""
    n_dofs = constraint_state.qfrc_constraint.shape[0]
    _B = constraint_state.grad.shape[1]

    for i_d, i_b in qd.ndrange(n_dofs, _B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            n_con = constraint_state.n_constraints[i_b]
            qfrc = gs.qd_float(0.0)
            for i_c in range(n_con):
                qfrc += constraint_state.jac[i_c, i_d, i_b] * constraint_state.efc_force[i_c, i_b]
            constraint_state.qfrc_constraint[i_d, i_b] = qfrc


@qd.kernel(fastcache=gs.use_fastcache)
def _kernel_update_constraint_cost(
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """Compute gauss and cost (reductions over dofs and constraints). One thread per env."""
    _B = constraint_state.grad.shape[1]

    qd.loop_config(block_dim=32)
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


@qd.kernel(fastcache=gs.use_fastcache)
def _kernel_newton_only_nt_hessian(
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Step 4: Newton Hessian update (Newton only)"""
    solver.func_hessian_direct_tiled(constraint_state=constraint_state, rigid_global_info=rigid_global_info)
    if qd.static(static_rigid_sim_config.enable_tiled_cholesky_hessian):
        solver.func_cholesky_factor_direct_tiled(
            constraint_state=constraint_state,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )
    else:
        _B = constraint_state.jac.shape[2]
        qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
        for i_b in range(_B):
            if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
                solver.func_cholesky_factor_direct_batch(
                    i_b=i_b, constraint_state=constraint_state, rigid_global_info=rigid_global_info
                )


@qd.kernel(fastcache=gs.use_fastcache)
def _kernel_update_gradient(
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Step 5: Update gradient"""
    _B = constraint_state.grad.shape[1]
    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
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


@qd.kernel(fastcache=gs.use_fastcache)
def _kernel_update_search_direction(
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Step 6: Check convergence and update search direction"""
    _B = constraint_state.grad.shape[1]
    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            solver.func_terminate_or_update_descent_batch(
                i_b,
                rigid_global_info=rigid_global_info,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )


# ============================================== Solve body dispatch ================================================


@solver.func_solve_body.register(
    is_compatible=lambda *args, **kwargs: (
        # Note: we do not use parallel linesearch for finite difference gradient validation, as it is highly
        # sensitive to numerical precision and GPU float64 rounding errors can accumulate over many trials.
        gs.backend in {gs.cuda} and not (args[5] if len(args) > 5 else kwargs["static_rigid_sim_config"]).requires_grad
    )
)
def func_solve_decomposed(
    entities_info,
    dofs_info,
    dofs_state,
    constraint_state,
    rigid_global_info,
    static_rigid_sim_config,
    _n_iterations,
):
    """
    Uses separate kernels for each solver step per iteration.

    This maximizes kernel granularity, potentially allowing better GPU scheduling
    and more flexibility in execution, at the cost of more Python->C++ boundary crossings.
    """
    # _n_iterations is a Python-native int to avoid CPU-GPU sync (vs rigid_global_info.iterations[None])
    for _it in range(_n_iterations):
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
        # Successive refinement: each pass narrows the search range around the best alpha.
        for _refine in range(LS_PARALLEL_N_REFINE):
            _kernel_parallel_linesearch_eval(
                constraint_state,
                rigid_global_info,
                static_rigid_sim_config,
            )
        _kernel_parallel_linesearch_apply_alpha(
            constraint_state,
            rigid_global_info,
            static_rigid_sim_config,
        )
        if static_rigid_sim_config.solver_type == gs.constraint_solver.CG:
            _kernel_cg_only_save_prev_grad(
                constraint_state,
                static_rigid_sim_config,
            )
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

        if static_rigid_sim_config.solver_type == gs.constraint_solver.Newton:
            _kernel_newton_only_nt_hessian(
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
