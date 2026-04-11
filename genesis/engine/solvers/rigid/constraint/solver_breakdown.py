import sys

import numpy as np
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

# Block sizes for shared-memory reductions in _kernel_parallel_linesearch_p0 and _jv.
_P0_BLOCK = 32
_JV_BLOCK = 32

# Maximum bisection iterations for gradient-guided refinement after grid search.
LS_BISECT_STEPS = 12

# Number of alpha candidates evaluated via cooperative constraint reduction.
# Each candidate is evaluated by ALL K threads cooperating on the constraint sum,
# reducing per-thread work from O(n_constraints) to O(n_constraints/K).
LS_N_CANDIDATES = 6

# Maximum allowed alpha (prevents divergence from degenerate steps).
LS_ALPHA_MAX = 1e4


@qd.func
def _func_parallel_linesearch_p0(
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Parallel linesearch P0 kernel: fused mv + jv + snorm + quad_gauss + eq_sum + p0_cost.

    Parallel grid-search linesearch algorithm overview
    --------------------------------------------------
    A block of K=32 threads cooperates on each env. Both approaches are O(n_constraints) per
    evaluation, but the grid search parallelizes each evaluation across 32 threads
    (n_constraints/32 work per thread), whereas the iterative approach runs each evaluation on
    a single thread.

    The algorithm is split across two kernels:

    P0 kernel (this function):
        Phase 0a: Compute mv = M @ search (cooperative over DOFs, 32 threads).
        Phase 0b: Compute jv = J @ search (cooperative over constraints, 32 threads).
        Phase 1: Fused snorm + quad_gauss parallel reduction over n_dofs.
        Phase 2: Parallel reduction over n_constraints for eq_sum and p0_cost.

    Eval kernel (_kernel_parallel_linesearch_eval):
        a) Grid search: Evaluate N_CANDIDATES=6 log-spaced alphas plus the Newton step,
           all 32 threads cooperating on each candidate's constraint reduction.
        b) Newton correction: One Newton step from the best grid candidate. Accepted if it
           improves cost.
        c) Bisection fallback: If Newton fails, bracket the zero-crossing of the gradient
           and bisect up to LS_BISECT_STEPS=12 times.
        d) Apply: Update qacc, Ma, Jaref with the chosen alpha (cooperative over DOFs).

    Post-linesearch: Separate kernels for constraint force update, cost update, gradient
    update, Hessian update (Newton only), and search direction update. These reuse the
    batch-level functions from solver.py.
    """
    _B = constraint_state.grad.shape[1]
    _T = qd.static(_P0_BLOCK)

    qd.loop_config(name="parallel_linesearch_p0", block_dim=_T)
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
            n_con = constraint_state.n_constraints[i_b]

            # === Phase 0a: Compute mv = M @ search (cooperative over DOFs) ===
            i_d1 = tid
            while i_d1 < n_dofs:
                I_d1 = [i_d1, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else i_d1
                i_e = dofs_info.entity_idx[I_d1]
                mv_val = gs.qd_float(0.0)
                for i_d2 in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                    mv_val = mv_val + rigid_global_info.mass_mat[i_d1, i_d2, i_b] * constraint_state.search[i_d2, i_b]
                constraint_state.mv[i_d1, i_b] = mv_val
                i_d1 += _T

            # === Phase 0b: Compute jv = J @ search (cooperative over constraints) ===
            i_c = tid
            while i_c < n_con:
                jv_val = gs.qd_float(0.0)
                if qd.static(static_rigid_sim_config.sparse_solve):
                    for i_d_ in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                        i_d = constraint_state.jac_relevant_dofs[i_c, i_d_, i_b]
                        jv_val = jv_val + constraint_state.jac[i_c, i_d, i_b] * constraint_state.search[i_d, i_b]
                else:
                    for i_d in range(n_dofs):
                        jv_val = jv_val + constraint_state.jac[i_c, i_d, i_b] * constraint_state.search[i_d, i_b]
                constraint_state.jv[i_c, i_b] = jv_val
                i_c += _T

            qd.simt.block.sync()  # Ensure mv and jv are written before Phase 1 reads them

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
                    # Initialize best alpha, search range, and best-cost tracker for parallel linesearch
                    constraint_state.candidates[0, i_b] = 0.0  # default: no step

                    # Use full Newton step (DOF + all constraints) as the range center.
                    total_hess = 2.0 * (constraint_state.quad_gauss[2, i_b] + sh_constraint_hess[0])
                    if total_hess > 0.0:
                        total_grad = constraint_state.quad_gauss[1, i_b] + sh_constraint_grad[0]
                        alpha_newton = qd.max(
                            qd.abs(total_grad / total_hess), gs.qd_float(qd.static(LS_PARALLEL_MIN_STEP))
                        )
                        constraint_state.candidates[2, i_b] = alpha_newton * 1e-2
                        constraint_state.candidates[3, i_b] = alpha_newton * 10.0
                        constraint_state.candidates[5, i_b] = alpha_newton  # exact Newton step for eval
                    else:
                        constraint_state.candidates[2, i_b] = 1e-6
                        constraint_state.candidates[3, i_b] = 1e2
                        constraint_state.candidates[5, i_b] = 0.0
                    constraint_state.candidates[4, i_b] = gs.qd_float(1e30)  # best cost across passes
                    # Store gtol for gradient-guided bisection after grid search
                    n_dofs_val = constraint_state.search.shape[0]
                    scale = rigid_global_info.meaninertia[i_b] * qd.max(1, n_dofs_val)
                    constraint_state.candidates[7, i_b] = (
                        rigid_global_info.tolerance[None] * rigid_global_info.ls_tolerance[None] * snorm * scale
                    )


@qd.func
def _func_iterative_linesearch_and_apply(
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Cooperative iterative Newton linesearch + cooperative alpha apply.

    Same algorithm as the monolith's func_linesearch_batch, but each constraint evaluation
    uses all K threads cooperatively for K-fold speedup. Thread 0 manages the Newton
    algorithm's control flow via shared memory st[].

    Phases (st[30]): 0=eval_p1, 1=bracket, 2=refine_a0, 3=refine_a1, 4=refine_a2, 5=refine_process.
    """
    _B = constraint_state.grad.shape[1]
    _K = qd.static(LS_PARALLEL_K)

    qd.loop_config(name="parallel_linesearch_eval", block_dim=_K)
    for i_flat in range(_B * _K):
        tid = i_flat % _K
        i_b = i_flat // _K

        sh_qt0 = qd.simt.block.SharedArray((_K,), gs.qd_float)
        sh_qt1 = qd.simt.block.SharedArray((_K,), gs.qd_float)
        sh_qt2 = qd.simt.block.SharedArray((_K,), gs.qd_float)
        st = qd.simt.block.SharedArray((32,), gs.qd_float)
        # st layout: [0]=alpha_to_eval  [1]=p1_a [2]=p1_c [3]=p1_g [4]=p1_h
        #            [5]=p2_a [6]=p2_c [7]=p2_g [8]=p2_h  [9]=done [10]=direction
        #            [11]=p0_cost [12]=result_alpha [13]=ls_it [14]=p2update [15]=gtol
        #            [16]=snorm_ok [17..19]=costs[0..2] [20..22]=grads[0..2] [23..25]=hess[0..2]
        #            [26..28]=alphas[0..2] [29]=unused [30]=phase [31]=converged_flag

        active = constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]
        if not active:
            if tid == 0:
                constraint_state.candidates[0, i_b] = 0.0
        if active:
            ne = constraint_state.n_constraints_equality[i_b]
            nef = ne + constraint_state.n_constraints_frictionloss[i_b]
            n_con = constraint_state.n_constraints[i_b]

            # Thread 0 initializes all state
            if tid == 0:
                for _si in range(32):
                    st[_si] = 0.0
                n_dofs = constraint_state.search.shape[0]
                snorm = gs.qd_float(0.0)
                for jd in range(n_dofs):
                    snorm = snorm + constraint_state.search[jd, i_b] ** 2
                snorm = qd.sqrt(snorm)
                scale = rigid_global_info.meaninertia[i_b] * qd.max(1, n_dofs)
                st[15] = rigid_global_info.tolerance[None] * rigid_global_info.ls_tolerance[None] * snorm * scale
                st[16] = gs.qd_float(snorm >= rigid_global_info.EPS[None])
            qd.simt.block.sync()

            if st[16] > 0.5:
                # === Cooperative p0 eval (alpha=0) ===
                p0_lc0 = gs.qd_float(0.0)
                p0_lc1 = gs.qd_float(0.0)
                p0_lc2 = gs.qd_float(0.0)
                i_c = ne + tid
                while i_c < n_con:
                    Ja = constraint_state.Jaref[i_c, i_b]
                    jv = constraint_state.jv[i_c, i_b]
                    D = constraint_state.efc_D[i_c, i_b]
                    qf0 = D * (0.5 * Ja * Ja)
                    qf1 = D * (jv * Ja)
                    qf2 = D * (0.5 * jv * jv)
                    if i_c < nef:
                        f = constraint_state.efc_frictionloss[i_c, i_b]
                        r = constraint_state.diag[i_c, i_b]
                        rf = r * f
                        ln = Ja <= -rf
                        lp = Ja >= rf
                        if ln or lp:
                            qf0 = ln * f * (-0.5 * rf - Ja) + lp * f * (-0.5 * rf + Ja)
                            qf1 = ln * (-f * jv) + lp * (f * jv)
                            qf2 = 0.0
                        p0_lc0 = p0_lc0 + qf0
                        p0_lc1 = p0_lc1 + qf1
                        p0_lc2 = p0_lc2 + qf2
                    else:
                        act = Ja < 0
                        p0_lc0 = p0_lc0 + qf0 * act
                        p0_lc1 = p0_lc1 + qf1 * act
                        p0_lc2 = p0_lc2 + qf2 * act
                    i_c += _K
                sh_qt0[tid] = p0_lc0
                sh_qt1[tid] = p0_lc1
                sh_qt2[tid] = p0_lc2
                qd.simt.block.sync()
                stride = _K // 2
                while stride > 0:
                    if tid < stride:
                        sh_qt0[tid] += sh_qt0[tid + stride]
                        sh_qt1[tid] += sh_qt1[tid + stride]
                        sh_qt2[tid] += sh_qt2[tid + stride]
                    qd.simt.block.sync()
                    stride //= 2

                if tid == 0:
                    t0 = constraint_state.quad_gauss[0, i_b] + constraint_state.eq_sum[0, i_b] + sh_qt0[0]
                    t1 = constraint_state.quad_gauss[1, i_b] + constraint_state.eq_sum[1, i_b] + sh_qt1[0]
                    t2 = constraint_state.quad_gauss[2, i_b] + constraint_state.eq_sum[2, i_b] + sh_qt2[0]
                    p0_h = 2.0 * t2
                    if p0_h <= 0.0:
                        p0_h = rigid_global_info.EPS[None]
                    st[11] = t0  # p0_cost
                    st[0] = -t1 / p0_h  # p1_alpha (Newton from p0)
                    st[13] = 1.0  # ls_it
                    st[30] = 0.0  # phase: eval_p1
                qd.simt.block.sync()

                # === Main cooperative linesearch loop ===
                _MAX_LS = qd.static(60)
                for _ls_iter in range(_MAX_LS):
                    if tid == 0:
                        if st[9] > 0.5 or st[13] >= rigid_global_info.ls_iterations[None]:
                            st[0] = -1.0
                    qd.simt.block.sync()

                    if st[0] < 0.0:
                        break

                    alpha_ev = st[0]

                    # === Cooperative constraint eval at alpha_ev ===
                    ev_lc0 = gs.qd_float(0.0)
                    ev_lc1 = gs.qd_float(0.0)
                    ev_lc2 = gs.qd_float(0.0)
                    i_c = ne + tid
                    while i_c < n_con:
                        Ja = constraint_state.Jaref[i_c, i_b]
                        jv = constraint_state.jv[i_c, i_b]
                        D = constraint_state.efc_D[i_c, i_b]
                        x = Ja + alpha_ev * jv
                        qf0 = D * (0.5 * Ja * Ja)
                        qf1 = D * (jv * Ja)
                        qf2 = D * (0.5 * jv * jv)
                        if i_c < nef:
                            f = constraint_state.efc_frictionloss[i_c, i_b]
                            r = constraint_state.diag[i_c, i_b]
                            rf = r * f
                            ln = x <= -rf
                            lp = x >= rf
                            if ln or lp:
                                qf0 = ln * f * (-0.5 * rf - Ja) + lp * f * (-0.5 * rf + Ja)
                                qf1 = ln * (-f * jv) + lp * (f * jv)
                                qf2 = 0.0
                            ev_lc0 = ev_lc0 + qf0
                            ev_lc1 = ev_lc1 + qf1
                            ev_lc2 = ev_lc2 + qf2
                        else:
                            act = x < 0
                            ev_lc0 = ev_lc0 + qf0 * act
                            ev_lc1 = ev_lc1 + qf1 * act
                            ev_lc2 = ev_lc2 + qf2 * act
                        i_c += _K
                    sh_qt0[tid] = ev_lc0
                    sh_qt1[tid] = ev_lc1
                    sh_qt2[tid] = ev_lc2
                    qd.simt.block.sync()
                    stride = _K // 2
                    while stride > 0:
                        if tid < stride:
                            sh_qt0[tid] += sh_qt0[tid + stride]
                            sh_qt1[tid] += sh_qt1[tid + stride]
                            sh_qt2[tid] += sh_qt2[tid + stride]
                        qd.simt.block.sync()
                        stride //= 2

                    # Thread 0 processes result based on current phase
                    if tid == 0:
                        t0 = constraint_state.quad_gauss[0, i_b] + constraint_state.eq_sum[0, i_b] + sh_qt0[0]
                        t1 = constraint_state.quad_gauss[1, i_b] + constraint_state.eq_sum[1, i_b] + sh_qt1[0]
                        t2 = constraint_state.quad_gauss[2, i_b] + constraint_state.eq_sum[2, i_b] + sh_qt2[0]
                        ev_c = alpha_ev * alpha_ev * t2 + alpha_ev * t1 + t0
                        ev_g = 2.0 * alpha_ev * t2 + t1
                        ev_h = 2.0 * t2
                        if ev_h <= 0.0:
                            ev_h = rigid_global_info.EPS[None]
                        st[13] = st[13] + 1.0
                        gtol = st[15]
                        phase = st[30]

                        if phase < 0.5:
                            # Phase 0: p1 eval result
                            if st[11] < ev_c:
                                # p0 better → use p0 as p1
                                p0_t1 = constraint_state.quad_gauss[1, i_b] + constraint_state.eq_sum[1, i_b]
                                p0_t2 = constraint_state.quad_gauss[2, i_b] + constraint_state.eq_sum[2, i_b]
                                p0_g = p0_t1  # grad at alpha=0
                                p0_h = 2.0 * p0_t2
                                if p0_h <= 0.0:
                                    p0_h = rigid_global_info.EPS[None]
                                st[1] = 0.0
                                st[2] = st[11]
                                st[3] = p0_g
                                st[4] = p0_h
                            else:
                                st[1] = alpha_ev
                                st[2] = ev_c
                                st[3] = ev_g
                                st[4] = ev_h

                            if qd.abs(st[3]) < gtol:
                                st[12] = st[1]
                                st[9] = 1.0
                            else:
                                # Start bracketing
                                st[10] = (st[3] < 0.0) * 2.0 - 1.0
                                st[5] = st[1]  # p2 = p1
                                st[6] = st[2]
                                st[7] = st[3]
                                st[8] = st[4]
                                st[30] = 1.0  # phase: bracket
                                st[0] = st[1] - st[3] / st[4]

                        elif phase < 1.5:
                            # Phase 1: bracketing
                            if qd.abs(ev_g) < gtol:
                                st[12] = alpha_ev
                                st[9] = 1.0
                            elif st[3] * st[10] <= -gtol:
                                # Gradient still same sign → continue bracketing
                                st[5] = st[1]  # p2 = old p1
                                st[6] = st[2]
                                st[7] = st[3]
                                st[8] = st[4]
                                st[14] = 1.0  # p2update
                                st[1] = alpha_ev
                                st[2] = ev_c
                                st[3] = ev_g
                                st[4] = ev_h
                                st[0] = alpha_ev - ev_g / ev_h
                            else:
                                # Gradient changed sign → transition to refinement
                                # First update p2 with current p1, p1 with eval result
                                if st[14] < 0.5:
                                    # No p2 update happened → accept p1
                                    st[12] = st[1]
                                    st[9] = 1.0
                                else:
                                    st[5] = st[1]
                                    st[6] = st[2]
                                    st[7] = st[3]
                                    st[8] = st[4]
                                    st[1] = alpha_ev
                                    st[2] = ev_c
                                    st[3] = ev_g
                                    st[4] = ev_h
                                    # Set up 3 alphas for refinement
                                    st[26] = st[1] - st[3] / st[4]
                                    st[27] = st[1]
                                    st[28] = (st[1] + st[5]) * 0.5
                                    st[30] = 2.0  # phase: refine_a0
                                    st[0] = st[26]

                        elif phase < 2.5:
                            # Phase 2: refine alpha_0 result
                            st[17] = ev_c
                            st[20] = ev_g
                            st[23] = ev_h
                            st[30] = 3.0
                            st[0] = st[27]

                        elif phase < 3.5:
                            # Phase 3: refine alpha_1 result
                            st[18] = ev_c
                            st[21] = ev_g
                            st[24] = ev_h
                            st[30] = 4.0
                            st[0] = st[28]

                        elif phase < 4.5:
                            # Phase 4: refine alpha_2 result → process all 3
                            st[19] = ev_c
                            st[22] = ev_g
                            st[25] = ev_h

                            # Check convergence among 3 candidates
                            best_a = gs.qd_float(0.0)
                            best_c = gs.qd_float(0.0)
                            found = False
                            for ci in qd.static(range(3)):
                                if qd.abs(st[20 + ci]) < gtol and (not found or st[17 + ci] < best_c):
                                    best_a = st[26 + ci]
                                    best_c = st[17 + ci]
                                    found = True
                            if found:
                                st[12] = best_a
                                st[9] = 1.0
                            else:
                                # Update p1 bracket
                                b1 = False
                                for ci in qd.static(range(3)):
                                    if st[3] < 0.0 and st[20 + ci] < 0.0 and st[3] < st[20 + ci]:
                                        st[1] = st[26 + ci]
                                        st[2] = st[17 + ci]
                                        st[3] = st[20 + ci]
                                        st[4] = st[23 + ci]
                                        b1 = True
                                    elif st[3] > 0.0 and st[20 + ci] > 0.0 and st[3] > st[20 + ci]:
                                        st[1] = st[26 + ci]
                                        st[2] = st[17 + ci]
                                        st[3] = st[20 + ci]
                                        st[4] = st[23 + ci]
                                        b1 = True
                                p1_next = st[1] - st[3] / st[4] if b1 else st[26]

                                # Update p2 bracket
                                b2 = False
                                for ci in qd.static(range(3)):
                                    if st[7] < 0.0 and st[20 + ci] < 0.0 and st[7] < st[20 + ci]:
                                        st[5] = st[26 + ci]
                                        st[6] = st[17 + ci]
                                        st[7] = st[20 + ci]
                                        st[8] = st[23 + ci]
                                        b2 = True
                                    elif st[7] > 0.0 and st[20 + ci] > 0.0 and st[7] > st[20 + ci]:
                                        st[5] = st[26 + ci]
                                        st[6] = st[17 + ci]
                                        st[7] = st[20 + ci]
                                        st[8] = st[23 + ci]
                                        b2 = True
                                p2_next = st[5] - st[7] / st[8] if b2 else st[27]

                                if not b1 and not b2:
                                    if st[19] < st[11]:
                                        st[12] = st[28]
                                    else:
                                        st[12] = 0.0
                                    st[9] = 1.0
                                else:
                                    st[26] = p1_next
                                    st[27] = p2_next
                                    st[28] = (st[1] + st[5]) * 0.5
                                    st[30] = 2.0  # back to refine_a0
                                    st[0] = st[26]
                    qd.simt.block.sync()

                # Finalize
                if tid == 0:
                    if st[9] < 0.5:
                        # Not converged: pick best of p1, p2
                        if st[2] <= st[6] and st[2] < st[11]:
                            st[12] = st[1]
                        elif st[6] <= st[2] and st[6] < st[11]:
                            st[12] = st[5]
                    constraint_state.candidates[0, i_b] = st[12]
            else:
                if tid == 0:
                    constraint_state.candidates[0, i_b] = 0.0
            qd.simt.block.sync()

        # Cooperative apply alpha (all K threads)
        if active:
            n_dofs_apply = constraint_state.qacc.shape[0]
            n_con_apply = constraint_state.n_constraints[i_b]
            alpha_apply = constraint_state.candidates[0, i_b]
            if qd.abs(alpha_apply) < rigid_global_info.EPS[None]:
                if tid == 0:
                    constraint_state.improved[i_b] = False
            else:
                i_d = tid
                while i_d < n_dofs_apply:
                    constraint_state.qacc[i_d, i_b] += constraint_state.search[i_d, i_b] * alpha_apply
                    constraint_state.Ma[i_d, i_b] += constraint_state.mv[i_d, i_b] * alpha_apply
                    i_d += _K
                i_c = tid
                while i_c < n_con_apply:
                    constraint_state.Jaref[i_c, i_b] += constraint_state.jv[i_c, i_b] * alpha_apply
                    i_c += _K


# ============================================== Shared iteration funcs ================================================


@qd.func
def _func_cg_only_save_prev_grad(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """Save prev_grad and prev_Mgrad (CG only)"""
    _B = constraint_state.grad.shape[1]
    qd.loop_config(
        name="cg_only_save_prev_grag", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32
    )
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            solver.func_save_prev_grad(i_b, constraint_state=constraint_state)


@qd.func
def _func_update_constraint_forces(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """Compute active flags and efc_force, parallelized over (constraint, env)."""
    len_constraints = constraint_state.active.shape[0]
    _B = constraint_state.grad.shape[1]

    qd.loop_config(name="update_constraint_forces")
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


@qd.func
def _func_update_constraint_qfrc(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """Compute qfrc_constraint = J^T @ efc_force, parallelized over (dof, env)."""
    n_dofs = constraint_state.qfrc_constraint.shape[0]
    _B = constraint_state.grad.shape[1]

    qd.loop_config(name="update_constraint_qfrc")
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            n_con = constraint_state.n_constraints[i_b]
            qfrc = gs.qd_float(0.0)
            for i_c in range(n_con):
                qfrc += constraint_state.jac[i_c, i_d, i_b] * constraint_state.efc_force[i_c, i_b]
            constraint_state.qfrc_constraint[i_d, i_b] = qfrc


@qd.func
def _func_update_constraint_cost(
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """Compute gauss and cost (reductions over dofs and constraints). One thread per env."""
    _B = constraint_state.grad.shape[1]

    qd.loop_config(name="update_constraint_cost", block_dim=32)
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


@qd.func
def _func_newton_only_nt_hessian(
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
        qd.loop_config(
            name="cholesky_factor_direct_batch",
            serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL,
            block_dim=32,
        )
        for i_b in range(_B):
            if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
                solver.func_cholesky_factor_direct_batch(
                    i_b=i_b, constraint_state=constraint_state, rigid_global_info=rigid_global_info
                )


@qd.func
def _func_update_gradient(
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Step 5: Update gradient"""
    _B = constraint_state.grad.shape[1]
    qd.loop_config(
        name="update_gradient", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32
    )
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


@qd.func
def _func_update_search_direction(
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Step 6: Check convergence and update search direction"""
    _B = constraint_state.grad.shape[1]
    qd.loop_config(
        name="update_search_direction", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32
    )
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            solver.func_terminate_or_update_descent_batch(
                i_b,
                rigid_global_info=rigid_global_info,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )


@qd.func
def _func_check_early_exit(
    constraint_state: array_class.ConstraintState,
    graph_counter: qd.types.ndarray(qd.i32, ndim=0),
):
    """Decrement iteration counter and exit early if no batch element improved."""
    qd.loop_config(name="check_early_exit_reset_flag")
    for _ in range(1):
        graph_counter[()] = graph_counter[()] - 1
        constraint_state.early_exit_flag[()] = 0

    _B = constraint_state.grad.shape[1]
    qd.loop_config(name="check_early_exit_scan_values")
    for i_b in range(_B):
        if constraint_state.improved[i_b]:
            qd.atomic_max(constraint_state.early_exit_flag[()], 1)

    qd.loop_config(name="check_early_exit_set_counter")
    for _ in range(1):
        if constraint_state.early_exit_flag[()] == 0:
            graph_counter[()] = 0


# ============================================== Solve body dispatch ================================================


@qd.kernel(graph=True, fastcache=gs.use_fastcache)
def _kernel_solve_graph(
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    graph_counter: qd.types.ndarray(qd.i32, ndim=0),
):
    while qd.graph_do_while(graph_counter):
        # Fused: mv + jv + snorm + quad_gauss + eq_sum + p0_cost
        _func_parallel_linesearch_p0(
            dofs_info, entities_info, dofs_state, constraint_state, rigid_global_info, static_rigid_sim_config
        )
        # Iterative Newton linesearch (cooperative) + cooperative apply
        _func_iterative_linesearch_and_apply(
            dofs_info, entities_info, dofs_state, constraint_state, rigid_global_info, static_rigid_sim_config
        )
        if qd.static(static_rigid_sim_config.solver_type == gs.constraint_solver.CG):
            _func_cg_only_save_prev_grad(constraint_state, static_rigid_sim_config)
        _func_update_constraint_forces(constraint_state, static_rigid_sim_config)
        _func_update_constraint_qfrc(constraint_state, static_rigid_sim_config)
        _func_update_constraint_cost(dofs_state, constraint_state, static_rigid_sim_config)
        if qd.static(static_rigid_sim_config.solver_type == gs.constraint_solver.Newton):
            _func_newton_only_nt_hessian(constraint_state, rigid_global_info, static_rigid_sim_config)
        _func_update_gradient(entities_info, dofs_state, constraint_state, rigid_global_info, static_rigid_sim_config)
        _func_update_search_direction(constraint_state, rigid_global_info, static_rigid_sim_config)
        _func_check_early_exit(constraint_state, graph_counter)


@solver.func_solve_body.register(
    is_compatible=lambda *args, **kwargs: (
        not (static_rigid_sim_config := solver._get_static_config(*args, **kwargs)).requires_grad
        and static_rigid_sim_config.prefer_parallel_linesearch != 0
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
    GPU graph accelerated solver loop with parallel grid-search linesearch and GPU-side iteration via graph_do_while.

    On CUDA SM 9.0+ (Hopper), the entire iteration loop runs on the GPU with no host involvement. On older CUDA GPUs,
    falls back to a host-side do-while loop that still benefits from CUDA graph kernel launch batching. On other GPUs,
    falls back to a host-side C++-side loop, that still reduces python launch overhead.

    Early exits when all batch elements have converged (no improved[i_b] is True).
    """
    if _n_iterations <= 0:
        return
    constraint_state.graph_counter.from_numpy(np.array(_n_iterations, dtype=np.int32))
    _kernel_solve_graph(
        dofs_info,
        entities_info,
        dofs_state,
        constraint_state,
        rigid_global_info,
        static_rigid_sim_config,
        constraint_state.graph_counter,
    )
