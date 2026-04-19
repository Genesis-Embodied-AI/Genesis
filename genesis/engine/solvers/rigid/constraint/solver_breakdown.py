import os

_LINESEARCH_TIGHTER_BRACKET = os.environ.get("GS_LINESEARCH_TIGHTER_BRACKET", "0") == "1"

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

# Block sizes for shared-memory reductions in _kernel_parallel_linesearch_p0 and _jv.
_P0_BLOCK = 32
_JV_BLOCK = 32

# Maximum allowed alpha (prevents divergence from degenerate steps).
LS_ALPHA_MAX = 1e4


@qd.func
def _eval_constraint_at_alpha(
    alpha,
    i_c,
    i_b,
    ne,
    nef,
    constraint_state: array_class.ConstraintState,
):
    """Evaluate a single constraint's cost/grad/hess contribution at a given alpha.

    Returns (cost, grad, hess) for the variable part (friction + contact only).
    Equality constraints are handled via precomputed constant coefficients.
    """
    Jaref_c = constraint_state.Jaref[i_c, i_b]
    jv_c = constraint_state.jv[i_c, i_b]
    D = constraint_state.efc_D[i_c, i_b]
    x = Jaref_c + alpha * jv_c
    jvD = jv_c * D
    jv2D = jv_c * jvD

    ec = gs.qd_float(0.0)
    eg = gs.qd_float(0.0)
    eh = gs.qd_float(0.0)

    if i_c < nef:
        f_val = constraint_state.efc_frictionloss[i_c, i_b]
        r_val = constraint_state.diag[i_c, i_b]
        rf = r_val * f_val
        if x <= -rf:
            ec = f_val * (-0.5 * rf - x)
            eg = -f_val * jv_c
        elif x >= rf:
            ec = f_val * (-0.5 * rf + x)
            eg = f_val * jv_c
        else:
            ec = 0.5 * D * x * x
            eg = jvD * x
            eh = jv2D
    else:
        if x < 0:
            ec = 0.5 * D * x * x
            eg = jvD * x
            eh = jv2D

    return ec, eg, eh


@qd.func
def _reduce_3(sh_a, sh_b, sh_c, tid):
    """Tree-reduce 3 shared arrays of size LS_PARALLEL_K in-place. Result in index 0."""
    _K = qd.static(LS_PARALLEL_K)
    qd.simt.block.sync()
    stride = _K // 2
    while stride > 0:
        if tid < stride:
            sh_a[tid] += sh_a[tid + stride]
            sh_b[tid] += sh_b[tid + stride]
            sh_c[tid] += sh_c[tid + stride]
        qd.simt.block.sync()
        stride //= 2


@qd.func
def _reduce_9(sh_a, sh_b, sh_c, sh_d, sh_e, sh_f, sh_g, sh_h, sh_i, tid):
    """Tree-reduce 9 shared arrays of size LS_PARALLEL_K in-place. Result in index 0."""
    _K = qd.static(LS_PARALLEL_K)
    qd.simt.block.sync()
    stride = _K // 2
    while stride > 0:
        if tid < stride:
            sh_a[tid] += sh_a[tid + stride]
            sh_b[tid] += sh_b[tid + stride]
            sh_c[tid] += sh_c[tid + stride]
            sh_d[tid] += sh_d[tid + stride]
            sh_e[tid] += sh_e[tid + stride]
            sh_f[tid] += sh_f[tid + stride]
            sh_g[tid] += sh_g[tid + stride]
            sh_h[tid] += sh_h[tid + stride]
            sh_i[tid] += sh_i[tid + stride]
        qd.simt.block.sync()
        stride //= 2


@qd.func
def _tighter_bracket(old_grad, new_grad):
    """True if new_grad is closer to zero from the same sign as old_grad."""
    return (old_grad < new_grad and new_grad < 0.0) or (old_grad > new_grad and new_grad > 0.0)


@qd.func
def _func_parallel_linesearch_p0(
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Decomposed constraint solver P0 kernel: fused mv + jv + snorm + quad_gauss + eq_sum + p0_cost.

    Decomposed solver algorithm overview
    -------------------------------------
    A block of K=32 threads cooperates on each env for all phases including linesearch refinement.

    P0 kernel (this function):
        Phase 0a: Compute mv = M @ search (cooperative over DOFs, 32 threads).
        Phase 0b: Compute jv = J @ search (cooperative over constraints, 32 threads).
        Phase 1: Fused snorm + quad_gauss parallel reduction over n_dofs.
        Phase 2: Parallel reduction over n_constraints for eq_sum and p0_cost. Also computes alpha_newton.

    Eval kernel (_func_parallel_linesearch_eval):
        a) Cooperative refinement (all K threads): Newton chase + 3-alpha bracket refinement with cooperative
           constraint reductions. Two bracket-swap strategies available via GS_LINESEARCH_TIGHTER_BRACKET.
        b) Apply: Update qacc, Ma, Jaref with the chosen alpha (cooperative over DOFs).

    Post-linesearch: Separate kernels for constraint force update, cost update, gradient update, Hessian update (Newton
    only), and search direction update. These reuse the batch-level functions from solver.py.
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
                    constraint_state.ls_alpha[i_b] = 0.0
                    constraint_state.ls_p0_cost[i_b] = 0.0
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
                    constraint_state.ls_p0_cost[i_b] = constraint_state.gauss[i_b] + sh_p0_cost[0]
                    # Initialize best alpha, search range, and best-cost tracker for parallel linesearch
                    constraint_state.ls_alpha[i_b] = 0.0  # default: no step

                    # Newton step estimate from the full DOF + constraint gradient/hessian
                    total_hess = 2.0 * (constraint_state.quad_gauss[2, i_b] + sh_constraint_hess[0])
                    if total_hess > 0.0:
                        total_grad = constraint_state.quad_gauss[1, i_b] + sh_constraint_grad[0]
                        constraint_state.ls_alpha_newton[i_b] = qd.abs(total_grad / total_hess)
                    else:
                        constraint_state.ls_alpha_newton[i_b] = 0.0
                    # Store gtol for gradient-guided refinement
                    n_dofs_val = constraint_state.search.shape[0]
                    scale = rigid_global_info.meaninertia[i_b] * qd.max(1, n_dofs_val)
                    constraint_state.ls_gtol[i_b] = (
                        rigid_global_info.tolerance[None] * rigid_global_info.ls_tolerance[None] * snorm * scale
                    )


@qd.func
def _func_parallel_linesearch_eval(
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
):
    """Decomposed solver eval kernel: cooperative refinement from Newton step + cooperative apply.

    The P0 kernel precomputes a Newton step (ls_alpha_newton). This kernel refines it via Newton chase and 3-alpha
    bracket refinement, with all K=32 threads cooperating on constraint reductions. Two bracket-swap strategies are
    available, selected by GS_LINESEARCH_TIGHTER_BRACKET:
      - 0 (default): dual-bracket update (same logic as update_bracket_no_eval_local / monolith path)
      - 1: tighter-bracket heuristic (swaps bracket endpoint when candidate gradient is closer to zero from same sign)
    """
    _B = constraint_state.grad.shape[1]
    _K = qd.static(LS_PARALLEL_K)

    qd.loop_config(name="parallel_linesearch_eval", block_dim=_K)
    for i_flat in range(_B * _K):
        tid = i_flat % _K
        i_b = i_flat // _K

        sh0 = qd.simt.block.SharedArray((_K,), gs.qd_float)
        sh1 = qd.simt.block.SharedArray((_K,), gs.qd_float)
        sh2 = qd.simt.block.SharedArray((_K,), gs.qd_float)
        sh3 = qd.simt.block.SharedArray((_K,), gs.qd_float)
        sh4 = qd.simt.block.SharedArray((_K,), gs.qd_float)
        sh5 = qd.simt.block.SharedArray((_K,), gs.qd_float)
        sh6 = qd.simt.block.SharedArray((_K,), gs.qd_float)
        sh7 = qd.simt.block.SharedArray((_K,), gs.qd_float)
        sh8 = qd.simt.block.SharedArray((_K,), gs.qd_float)

        active = constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]

        if active:
            p0_cost = constraint_state.ls_p0_cost[i_b]
            gtol = constraint_state.ls_gtol[i_b]
            alpha_newton = constraint_state.ls_alpha_newton[i_b]
            ne = constraint_state.n_constraints_equality[i_b]
            nef = ne + constraint_state.n_constraints_frictionloss[i_b]
            n_con = constraint_state.n_constraints[i_b]
            EPS = rigid_global_info.EPS[None]
            max_ls_iter = rigid_global_info.ls_iterations[None]

            const_0 = constraint_state.quad_gauss[0, i_b] + constraint_state.eq_sum[0, i_b]
            const_1 = constraint_state.quad_gauss[1, i_b] + constraint_state.eq_sum[1, i_b]
            const_2 = constraint_state.quad_gauss[2, i_b] + constraint_state.eq_sum[2, i_b]

            best_alpha = gs.qd_float(0.0)

            if alpha_newton > 0.0:
                # ── Cooperative eval at alpha_newton ──────────────────────────────────────────────────────
                loc_vc = gs.qd_float(0.0)
                loc_vg = gs.qd_float(0.0)
                loc_vh = gs.qd_float(0.0)
                i_c = ne + tid
                while i_c < n_con:
                    ec, eg, eh = _eval_constraint_at_alpha(alpha_newton, i_c, i_b, ne, nef, constraint_state)
                    loc_vc += ec
                    loc_vg += eg
                    loc_vh += eh
                    i_c += _K
                sh0[tid] = loc_vc
                sh1[tid] = loc_vg
                sh2[tid] = loc_vh
                _reduce_3(sh0, sh1, sh2, tid)

                p1_alpha = alpha_newton
                p1_cost = const_0 + p1_alpha * const_1 + p1_alpha * p1_alpha * const_2 + sh0[0]
                p1_grad = const_1 + 2.0 * p1_alpha * const_2 + sh1[0]
                p1_hess = 2.0 * const_2 + sh2[0]
                if p1_hess <= 0.0:
                    p1_hess = EPS

                # If Newton step worsened cost, re-evaluate at alpha=0
                if p0_cost < p1_cost:
                    loc_vc = gs.qd_float(0.0)
                    loc_vg = gs.qd_float(0.0)
                    loc_vh = gs.qd_float(0.0)
                    i_c = ne + tid
                    while i_c < n_con:
                        ec, eg, eh = _eval_constraint_at_alpha(
                            gs.qd_float(0.0), i_c, i_b, ne, nef, constraint_state
                        )
                        loc_vc += ec
                        loc_vg += eg
                        loc_vh += eh
                        i_c += _K
                    sh0[tid] = loc_vc
                    sh1[tid] = loc_vg
                    sh2[tid] = loc_vh
                    _reduce_3(sh0, sh1, sh2, tid)

                    p1_alpha = gs.qd_float(0.0)
                    p1_cost = const_0 + sh0[0]
                    p1_grad = const_1 + sh1[0]
                    p1_hess = 2.0 * const_2 + sh2[0]
                    if p1_hess <= 0.0:
                        p1_hess = EPS

                if p1_cost < p0_cost:
                    best_alpha = p1_alpha

                if qd.abs(p1_grad) > gtol:
                    # ── Phase 1: Newton chase — follow Newton steps until gradient sign flips ─────────
                    direction = (p1_grad < 0) * 2 - 1
                    p2_alpha = p1_alpha
                    p2_cost = p1_cost
                    p2_grad = p1_grad
                    p2_hess = p1_hess
                    p2update = 0
                    chase_done = False
                    ls_iter = 1

                    while p1_grad * direction <= -gtol and ls_iter < max_ls_iter:
                        ls_iter += 1
                        p2_alpha = p1_alpha
                        p2_cost = p1_cost
                        p2_grad = p1_grad
                        p2_hess = p1_hess
                        p2update = 1

                        next_a = p1_alpha - p1_grad / p1_hess

                        loc_vc = gs.qd_float(0.0)
                        loc_vg = gs.qd_float(0.0)
                        loc_vh = gs.qd_float(0.0)
                        i_c = ne + tid
                        while i_c < n_con:
                            ec, eg, eh = _eval_constraint_at_alpha(next_a, i_c, i_b, ne, nef, constraint_state)
                            loc_vc += ec
                            loc_vg += eg
                            loc_vh += eh
                            i_c += _K
                        sh0[tid] = loc_vc
                        sh1[tid] = loc_vg
                        sh2[tid] = loc_vh
                        _reduce_3(sh0, sh1, sh2, tid)

                        p1_alpha = next_a
                        p1_cost = const_0 + next_a * const_1 + next_a * next_a * const_2 + sh0[0]
                        p1_grad = const_1 + 2.0 * next_a * const_2 + sh1[0]
                        p1_hess = 2.0 * const_2 + sh2[0]
                        if p1_hess <= 0.0:
                            p1_hess = EPS

                        if qd.abs(p1_grad) < gtol:
                            best_alpha = p1_alpha
                            chase_done = True

                    if not chase_done:
                        if ls_iter >= max_ls_iter:
                            if p1_cost < p0_cost:
                                best_alpha = p1_alpha
                        elif not p2update:
                            if p1_cost < p0_cost:
                                best_alpha = p1_alpha
                        else:
                            # ── Phase 2: 3-alpha bracket refinement ───────────────────────────────────
                            if qd.static(_LINESEARCH_TIGHTER_BRACKET):
                                # Tighter-bracket: set up lo/hi from chase endpoints
                                lo_a = p2_alpha
                                lo_c = p2_cost
                                lo_g = p2_grad
                                lo_h = p2_hess
                                hi_a = p1_alpha
                                hi_c = p1_cost
                                hi_g = p1_grad
                                hi_h = p1_hess
                                if p1_grad < p2_grad:
                                    lo_a = p1_alpha
                                    lo_c = p1_cost
                                    lo_g = p1_grad
                                    lo_h = p1_hess
                                    hi_a = p2_alpha
                                    hi_c = p2_cost
                                    hi_g = p2_grad
                                    hi_h = p2_hess

                                bracket_done = False
                                while not bracket_done and ls_iter < max_ls_iter:
                                    ls_iter += 1

                                    cand_a = lo_a - lo_g / lo_h
                                    cand_b = hi_a - hi_g / hi_h
                                    cand_c = 0.5 * (lo_a + hi_a)

                                    # Cooperative 3-alpha constraint eval
                                    loc_ac = gs.qd_float(0.0)
                                    loc_ag = gs.qd_float(0.0)
                                    loc_ah = gs.qd_float(0.0)
                                    loc_bc = gs.qd_float(0.0)
                                    loc_bg = gs.qd_float(0.0)
                                    loc_bh = gs.qd_float(0.0)
                                    loc_cc = gs.qd_float(0.0)
                                    loc_cg = gs.qd_float(0.0)
                                    loc_ch = gs.qd_float(0.0)

                                    i_c = ne + tid
                                    while i_c < n_con:
                                        Jaref_c = constraint_state.Jaref[i_c, i_b]
                                        jv_c = constraint_state.jv[i_c, i_b]
                                        D = constraint_state.efc_D[i_c, i_b]
                                        jvD = jv_c * D
                                        jv2D = jv_c * jvD

                                        xa = Jaref_c + cand_a * jv_c
                                        xb = Jaref_c + cand_b * jv_c
                                        xc = Jaref_c + cand_c * jv_c

                                        if i_c < nef:
                                            f_val = constraint_state.efc_frictionloss[i_c, i_b]
                                            r_val = constraint_state.diag[i_c, i_b]
                                            rf = r_val * f_val

                                            if xa <= -rf:
                                                loc_ac += f_val * (-0.5 * rf - xa)
                                                loc_ag += -f_val * jv_c
                                            elif xa >= rf:
                                                loc_ac += f_val * (-0.5 * rf + xa)
                                                loc_ag += f_val * jv_c
                                            else:
                                                loc_ac += 0.5 * D * xa * xa
                                                loc_ag += jvD * xa
                                                loc_ah += jv2D

                                            if xb <= -rf:
                                                loc_bc += f_val * (-0.5 * rf - xb)
                                                loc_bg += -f_val * jv_c
                                            elif xb >= rf:
                                                loc_bc += f_val * (-0.5 * rf + xb)
                                                loc_bg += f_val * jv_c
                                            else:
                                                loc_bc += 0.5 * D * xb * xb
                                                loc_bg += jvD * xb
                                                loc_bh += jv2D

                                            if xc <= -rf:
                                                loc_cc += f_val * (-0.5 * rf - xc)
                                                loc_cg += -f_val * jv_c
                                            elif xc >= rf:
                                                loc_cc += f_val * (-0.5 * rf + xc)
                                                loc_cg += f_val * jv_c
                                            else:
                                                loc_cc += 0.5 * D * xc * xc
                                                loc_cg += jvD * xc
                                                loc_ch += jv2D
                                        else:
                                            if xa < 0:
                                                loc_ac += 0.5 * D * xa * xa
                                                loc_ag += jvD * xa
                                                loc_ah += jv2D
                                            if xb < 0:
                                                loc_bc += 0.5 * D * xb * xb
                                                loc_bg += jvD * xb
                                                loc_bh += jv2D
                                            if xc < 0:
                                                loc_cc += 0.5 * D * xc * xc
                                                loc_cg += jvD * xc
                                                loc_ch += jv2D

                                        i_c += _K

                                    sh0[tid] = loc_ac
                                    sh1[tid] = loc_ag
                                    sh2[tid] = loc_ah
                                    sh3[tid] = loc_bc
                                    sh4[tid] = loc_bg
                                    sh5[tid] = loc_bh
                                    sh6[tid] = loc_cc
                                    sh7[tid] = loc_cg
                                    sh8[tid] = loc_ch
                                    _reduce_9(sh0, sh1, sh2, sh3, sh4, sh5, sh6, sh7, sh8, tid)

                                    a_cost = const_0 + cand_a * const_1 + cand_a * cand_a * const_2 + sh0[0]
                                    a_grad = const_1 + 2.0 * cand_a * const_2 + sh1[0]
                                    a_hess = 2.0 * const_2 + sh2[0]
                                    if a_hess <= 0.0:
                                        a_hess = EPS

                                    b_cost = const_0 + cand_b * const_1 + cand_b * cand_b * const_2 + sh3[0]
                                    b_grad = const_1 + 2.0 * cand_b * const_2 + sh4[0]
                                    b_hess = 2.0 * const_2 + sh5[0]
                                    if b_hess <= 0.0:
                                        b_hess = EPS

                                    c_cost = const_0 + cand_c * const_1 + cand_c * cand_c * const_2 + sh6[0]
                                    c_grad = const_1 + 2.0 * cand_c * const_2 + sh7[0]
                                    c_hess = 2.0 * const_2 + sh8[0]
                                    if c_hess <= 0.0:
                                        c_hess = EPS

                                    # Convergence check
                                    best_found = False
                                    best_cand_cost = gs.qd_float(0.0)
                                    if qd.abs(a_grad) < gtol and (not best_found or a_cost < best_cand_cost):
                                        best_alpha = cand_a
                                        best_cand_cost = a_cost
                                        best_found = True
                                    if qd.abs(b_grad) < gtol and (not best_found or b_cost < best_cand_cost):
                                        best_alpha = cand_b
                                        best_cand_cost = b_cost
                                        best_found = True
                                    if qd.abs(c_grad) < gtol and (not best_found or c_cost < best_cand_cost):
                                        best_alpha = cand_c
                                        best_cand_cost = c_cost
                                        best_found = True

                                    if best_found:
                                        bracket_done = True
                                    else:
                                        swap_lo = False
                                        if _tighter_bracket(lo_g, a_grad):
                                            lo_a, lo_c, lo_g, lo_h = cand_a, a_cost, a_grad, a_hess
                                            swap_lo = True
                                        if _tighter_bracket(lo_g, c_grad):
                                            lo_a, lo_c, lo_g, lo_h = cand_c, c_cost, c_grad, c_hess
                                            swap_lo = True
                                        if _tighter_bracket(lo_g, b_grad):
                                            lo_a, lo_c, lo_g, lo_h = cand_b, b_cost, b_grad, b_hess
                                            swap_lo = True

                                        swap_hi = False
                                        if _tighter_bracket(hi_g, b_grad):
                                            hi_a, hi_c, hi_g, hi_h = cand_b, b_cost, b_grad, b_hess
                                            swap_hi = True
                                        if _tighter_bracket(hi_g, c_grad):
                                            hi_a, hi_c, hi_g, hi_h = cand_c, c_cost, c_grad, c_hess
                                            swap_hi = True
                                        if _tighter_bracket(hi_g, a_grad):
                                            hi_a, hi_c, hi_g, hi_h = cand_a, a_cost, a_grad, a_hess
                                            swap_hi = True

                                        if not swap_lo and not swap_hi:
                                            best_alpha = cand_c
                                            bracket_done = True
                                        elif (lo_g < 0.0 and lo_g > -gtol) or (hi_g > 0.0 and hi_g < gtol):
                                            if lo_c < p0_cost or hi_c < p0_cost:
                                                if lo_c <= hi_c:
                                                    best_alpha = lo_a
                                                else:
                                                    best_alpha = hi_a
                                            bracket_done = True

                                if not bracket_done:
                                    if lo_c <= hi_c and lo_c < p0_cost:
                                        best_alpha = lo_a
                                    elif hi_c <= lo_c and hi_c < p0_cost:
                                        best_alpha = hi_a

                            else:
                                # Dual-bracket (Alexis's update_bracket_no_eval_local logic)
                                alpha_0 = p1_alpha - p1_grad / p1_hess
                                alpha_1 = p1_alpha
                                alpha_2 = (p1_alpha + p2_alpha) * 0.5

                                bracket_done = False
                                while not bracket_done and ls_iter < max_ls_iter:
                                    ls_iter += 1

                                    # Cooperative 3-alpha constraint eval
                                    loc_ac = gs.qd_float(0.0)
                                    loc_ag = gs.qd_float(0.0)
                                    loc_ah = gs.qd_float(0.0)
                                    loc_bc = gs.qd_float(0.0)
                                    loc_bg = gs.qd_float(0.0)
                                    loc_bh = gs.qd_float(0.0)
                                    loc_cc = gs.qd_float(0.0)
                                    loc_cg = gs.qd_float(0.0)
                                    loc_ch = gs.qd_float(0.0)

                                    i_c = ne + tid
                                    while i_c < n_con:
                                        Jaref_c = constraint_state.Jaref[i_c, i_b]
                                        jv_c = constraint_state.jv[i_c, i_b]
                                        D = constraint_state.efc_D[i_c, i_b]
                                        jvD = jv_c * D
                                        jv2D = jv_c * jvD

                                        xa = Jaref_c + alpha_0 * jv_c
                                        xb = Jaref_c + alpha_1 * jv_c
                                        xc = Jaref_c + alpha_2 * jv_c

                                        if i_c < nef:
                                            f_val = constraint_state.efc_frictionloss[i_c, i_b]
                                            r_val = constraint_state.diag[i_c, i_b]
                                            rf = r_val * f_val

                                            if xa <= -rf:
                                                loc_ac += f_val * (-0.5 * rf - xa)
                                                loc_ag += -f_val * jv_c
                                            elif xa >= rf:
                                                loc_ac += f_val * (-0.5 * rf + xa)
                                                loc_ag += f_val * jv_c
                                            else:
                                                loc_ac += 0.5 * D * xa * xa
                                                loc_ag += jvD * xa
                                                loc_ah += jv2D

                                            if xb <= -rf:
                                                loc_bc += f_val * (-0.5 * rf - xb)
                                                loc_bg += -f_val * jv_c
                                            elif xb >= rf:
                                                loc_bc += f_val * (-0.5 * rf + xb)
                                                loc_bg += f_val * jv_c
                                            else:
                                                loc_bc += 0.5 * D * xb * xb
                                                loc_bg += jvD * xb
                                                loc_bh += jv2D

                                            if xc <= -rf:
                                                loc_cc += f_val * (-0.5 * rf - xc)
                                                loc_cg += -f_val * jv_c
                                            elif xc >= rf:
                                                loc_cc += f_val * (-0.5 * rf + xc)
                                                loc_cg += f_val * jv_c
                                            else:
                                                loc_cc += 0.5 * D * xc * xc
                                                loc_cg += jvD * xc
                                                loc_ch += jv2D
                                        else:
                                            if xa < 0:
                                                loc_ac += 0.5 * D * xa * xa
                                                loc_ag += jvD * xa
                                                loc_ah += jv2D
                                            if xb < 0:
                                                loc_bc += 0.5 * D * xb * xb
                                                loc_bg += jvD * xb
                                                loc_bh += jv2D
                                            if xc < 0:
                                                loc_cc += 0.5 * D * xc * xc
                                                loc_cg += jvD * xc
                                                loc_ch += jv2D

                                        i_c += _K

                                    sh0[tid] = loc_ac
                                    sh1[tid] = loc_ag
                                    sh2[tid] = loc_ah
                                    sh3[tid] = loc_bc
                                    sh4[tid] = loc_bg
                                    sh5[tid] = loc_bh
                                    sh6[tid] = loc_cc
                                    sh7[tid] = loc_cg
                                    sh8[tid] = loc_ch
                                    _reduce_9(sh0, sh1, sh2, sh3, sh4, sh5, sh6, sh7, sh8, tid)

                                    costs_0 = const_0 + alpha_0 * const_1 + alpha_0 * alpha_0 * const_2 + sh0[0]
                                    grads_0 = const_1 + 2.0 * alpha_0 * const_2 + sh1[0]
                                    hess_0 = 2.0 * const_2 + sh2[0]
                                    if hess_0 <= 0.0:
                                        hess_0 = EPS

                                    costs_1 = const_0 + alpha_1 * const_1 + alpha_1 * alpha_1 * const_2 + sh3[0]
                                    grads_1 = const_1 + 2.0 * alpha_1 * const_2 + sh4[0]
                                    hess_1 = 2.0 * const_2 + sh5[0]
                                    if hess_1 <= 0.0:
                                        hess_1 = EPS

                                    costs_2 = const_0 + alpha_2 * const_1 + alpha_2 * alpha_2 * const_2 + sh6[0]
                                    grads_2 = const_1 + 2.0 * alpha_2 * const_2 + sh7[0]
                                    hess_2 = 2.0 * const_2 + sh8[0]
                                    if hess_2 <= 0.0:
                                        hess_2 = EPS

                                    # Convergence check
                                    best_found = False
                                    best_c = gs.qd_float(0.0)
                                    res_alpha = gs.qd_float(0.0)
                                    if qd.abs(grads_0) < gtol and (not best_found or costs_0 < best_c):
                                        res_alpha = alpha_0
                                        best_c = costs_0
                                        best_found = True
                                    if qd.abs(grads_1) < gtol and (not best_found or costs_1 < best_c):
                                        res_alpha = alpha_1
                                        best_c = costs_1
                                        best_found = True
                                    if qd.abs(grads_2) < gtol and (not best_found or costs_2 < best_c):
                                        res_alpha = alpha_2
                                        best_c = costs_2
                                        best_found = True

                                    p1_next = alpha_0
                                    p2_next = alpha_1

                                    if best_found:
                                        best_alpha = res_alpha
                                        bracket_done = True
                                    else:
                                        # Dual-bracket update for p1
                                        alphas = qd.Vector([alpha_0, alpha_1, alpha_2])
                                        costs_v = qd.Vector([costs_0, costs_1, costs_2])
                                        grads_v = qd.Vector([grads_0, grads_1, grads_2])
                                        hess_v = qd.Vector([hess_0, hess_1, hess_2])

                                        b1 = 0
                                        for i in qd.static(range(3)):
                                            if p1_grad < 0 and grads_v[i] < 0 and p1_grad < grads_v[i]:
                                                p1_alpha = alphas[i]
                                                p1_cost = costs_v[i]
                                                p1_grad = grads_v[i]
                                                p1_hess = hess_v[i]
                                                b1 = 1
                                            elif p1_grad > 0 and grads_v[i] > 0 and p1_grad > grads_v[i]:
                                                p1_alpha = alphas[i]
                                                p1_cost = costs_v[i]
                                                p1_grad = grads_v[i]
                                                p1_hess = hess_v[i]
                                                b1 = 2
                                        if b1 > 0:
                                            p1_next = p1_alpha - p1_grad / p1_hess

                                        # Dual-bracket update for p2
                                        b2 = 0
                                        for i in qd.static(range(3)):
                                            if p2_grad < 0 and grads_v[i] < 0 and p2_grad < grads_v[i]:
                                                p2_alpha = alphas[i]
                                                p2_cost = costs_v[i]
                                                p2_grad = grads_v[i]
                                                p2_hess = hess_v[i]
                                                b2 = 1
                                            elif p2_grad > 0 and grads_v[i] > 0 and p2_grad > grads_v[i]:
                                                p2_alpha = alphas[i]
                                                p2_cost = costs_v[i]
                                                p2_grad = grads_v[i]
                                                p2_hess = hess_v[i]
                                                b2 = 2
                                        if b2 > 0:
                                            p2_next = p2_alpha - p2_grad / p2_hess

                                        if b1 == 0 and b2 == 0:
                                            if costs_2 < p0_cost:
                                                best_alpha = alpha_2
                                            bracket_done = True

                                    if not bracket_done:
                                        alpha_0 = p1_next
                                        alpha_1 = p2_next
                                        alpha_2 = (p1_alpha + p2_alpha) * 0.5

                                if not bracket_done:
                                    if p1_cost <= p2_cost and p1_cost < p0_cost:
                                        best_alpha = p1_alpha
                                    elif p2_cost <= p1_cost and p2_cost < p0_cost:
                                        best_alpha = p2_alpha

            if tid == 0:
                constraint_state.ls_alpha[i_b] = best_alpha
            qd.simt.block.sync()
        else:
            if tid == 0:
                constraint_state.ls_alpha[i_b] = 0.0
            qd.simt.block.sync()

        # === Phase 4: Cooperative apply alpha (fused, saves 1 kernel launch) ===
        qd.simt.block.sync()
        if active:
            n_dofs_apply = constraint_state.qacc.shape[0]
            n_con_apply = constraint_state.n_constraints[i_b]
            alpha_apply = constraint_state.ls_alpha[i_b]
            if qd.abs(alpha_apply) < rigid_global_info.EPS[None]:
                if tid == 0:
                    constraint_state.improved[i_b] = False
            else:
                # Apply to dofs (strided over threads)
                i_d = tid
                while i_d < n_dofs_apply:
                    constraint_state.qacc[i_d, i_b] += constraint_state.search[i_d, i_b] * alpha_apply
                    constraint_state.Ma[i_d, i_b] += constraint_state.mv[i_d, i_b] * alpha_apply
                    i_d += _K
                # Apply to constraints (strided over threads)
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


# Number of full Hessian+Cholesky rebuilds at the start of the solver loop (after the init's iter-0 full rebuild).
# 0 = all incremental, 2 = full for loop iters 0-1 then incremental, 999 = always full.
@qd.func
def _func_build_changed_and_decide_hessian_mode(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """Build changed-constraint lists and set per-env use_full_hessian flag.

    Adaptive policy: use full rebuild if more than half the constraints changed, otherwise patch. Init (iter 0) always
    does full rebuild via func_solve_init.
    """
    qd.loop_config(name="increment_iter_counter")
    for _ in range(1):
        constraint_state.solver_iter_counter[()] = constraint_state.solver_iter_counter[()] + 1

    _B = constraint_state.grad.shape[1]
    iter_count = constraint_state.solver_iter_counter[()]
    qd.loop_config(name="build_changed_decide", block_dim=32)
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            solver.func_build_changed_constraint_list(i_b, constraint_state=constraint_state)
            # First graph iteration must do full rebuild: nt_H contains L from func_solve_init's Cholesky, not H.
            # Patching L would be wrong.
            if iter_count <= 1:
                constraint_state.use_full_hessian[i_b] = 1
            else:
                n_changed = constraint_state.incr_n_changed[i_b]
                n_total = constraint_state.n_constraints[i_b]
                if n_changed * 2 > n_total:
                    constraint_state.use_full_hessian[i_b] = 1
                else:
                    constraint_state.use_full_hessian[i_b] = 0


@qd.func
def _func_patch_hessian_delta(
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
):
    """Incrementally update H with delta contributions from changed constraints.

    Adds or subtracts each changed constraint's J^T D J contribution depending on whether it became active or inactive.
    Only runs on envs where use_full_hessian == 0 (others get a full rebuild instead).
    """
    _B = constraint_state.grad.shape[1]
    n_dofs = constraint_state.nt_H.shape[1]
    n_lower_tri = n_dofs * (n_dofs + 1) // 2

    BLOCK_DIM = qd.static(128)

    qd.loop_config(name="patch_hessian_delta", block_dim=BLOCK_DIM)
    for i in range(_B * BLOCK_DIM):
        tid = i % BLOCK_DIM
        i_b = i // BLOCK_DIM
        if i_b >= _B:
            continue
        if constraint_state.n_constraints[i_b] == 0 or not constraint_state.improved[i_b]:
            continue
        if constraint_state.use_full_hessian[i_b] != 0:
            continue

        n_changed = constraint_state.incr_n_changed[i_b]
        if n_changed == 0:
            continue

        elem = tid
        while elem < n_lower_tri:
            i_d1, i_d2 = solver.linear_to_lower_tri(elem)

            delta = gs.qd_float(0.0)
            for idx in range(n_changed):
                i_c = constraint_state.incr_changed_idx[idx, i_b]
                Ji = constraint_state.jac[i_c, i_d1, i_b]
                if Ji != 0.0:
                    Jj = constraint_state.jac[i_c, i_d2, i_b]
                    if Jj != 0.0:
                        D = constraint_state.efc_D[i_c, i_b]
                        if constraint_state.active[i_c, i_b]:
                            delta = delta + D * Ji * Jj
                        else:
                            delta = delta - D * Ji * Jj

            if delta != 0.0:
                constraint_state.nt_H[i_b, i_d1, i_d2] = constraint_state.nt_H[i_b, i_d1, i_d2] + delta
            elem = elem + BLOCK_DIM


@qd.func
def _func_newton_only_nt_hessian(
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
):
    """Full tiled Hessian rebuild for envs with use_full_hessian == 1 (skips others)."""
    solver.func_hessian_direct_tiled(
        constraint_state=constraint_state, rigid_global_info=rigid_global_info, check_full_hessian=True
    )


@qd.func
def _func_newton_only_nt_hessian_and_cholesky(
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Full Hessian rebuild + Cholesky for ALL improved envs (non-fused path).

    Matches origin/main behavior: H is rebuilt from scratch every iteration, then Cholesky overwrites nt_H with L
    in-place.  H patching is not used because the subsequent Cholesky would destroy H anyway.
    """
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
def _func_update_gradient_no_solve(
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Compute gradient only (no Cholesky solve) — used with fused Cholesky+Solve."""
    _B = constraint_state.grad.shape[1]
    n_dofs = constraint_state.grad.shape[0]
    qd.loop_config(name="update_gradient_no_solve", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            constraint_state.grad[i_d, i_b] = (
                constraint_state.Ma[i_d, i_b] - dofs_state.force[i_d, i_b] - constraint_state.qfrc_constraint[i_d, i_b]
            )


@qd.func
def _func_cholesky_and_solve_fused(
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Fused Cholesky factorization + solve. L stays in shared memory."""
    solver.func_cholesky_and_solve_fused_tiled(
        constraint_state=constraint_state,
        rigid_global_info=rigid_global_info,
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
        # Fused: refinement + apply alpha
        _func_parallel_linesearch_eval(constraint_state, rigid_global_info)
        if qd.static(static_rigid_sim_config.solver_type == gs.constraint_solver.CG):
            _func_cg_only_save_prev_grad(constraint_state, static_rigid_sim_config)
        _func_update_constraint_forces(constraint_state, static_rigid_sim_config)
        _func_update_constraint_qfrc(constraint_state, static_rigid_sim_config)
        _func_update_constraint_cost(dofs_state, constraint_state, static_rigid_sim_config)
        if qd.static(
            static_rigid_sim_config.solver_type == gs.constraint_solver.Newton
            and static_rigid_sim_config.enable_tiled_cholesky_hessian
        ):
            # Fused path: H patching + fused Cholesky+Solve (L in shmem, H preserved in nt_H)
            _func_build_changed_and_decide_hessian_mode(constraint_state, static_rigid_sim_config)
            _func_newton_only_nt_hessian(constraint_state, rigid_global_info)
            _func_patch_hessian_delta(constraint_state, rigid_global_info)
            _func_update_gradient_no_solve(
                entities_info, dofs_state, constraint_state, rigid_global_info, static_rigid_sim_config
            )
            _func_cholesky_and_solve_fused(constraint_state, rigid_global_info, static_rigid_sim_config)
        elif qd.static(static_rigid_sim_config.solver_type == gs.constraint_solver.Newton):
            # Non-fused path: full H rebuild + separate Cholesky every iteration (Cholesky overwrites nt_H with L,
            # so H patching is not possible)
            _func_newton_only_nt_hessian_and_cholesky(constraint_state, rigid_global_info, static_rigid_sim_config)
            _func_update_gradient(
                entities_info, dofs_state, constraint_state, rigid_global_info, static_rigid_sim_config
            )
        else:
            _func_update_gradient(
                entities_info, dofs_state, constraint_state, rigid_global_info, static_rigid_sim_config
            )
        _func_update_search_direction(constraint_state, rigid_global_info, static_rigid_sim_config)
        _func_check_early_exit(constraint_state, graph_counter)


@solver.func_solve_body.register(
    is_compatible=lambda *args, **kwargs: (
        not (static_rigid_sim_config := solver._get_static_config(*args, **kwargs)).requires_grad
        and static_rigid_sim_config.prefer_decomposed_solver != 0
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
