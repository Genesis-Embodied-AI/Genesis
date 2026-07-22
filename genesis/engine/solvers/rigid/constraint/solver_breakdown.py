import sys

import numpy as np
import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
from . import solver

# Shared-memory reduction block size for the _jv path.
_JV_BLOCK = 32

# Maximum allowed alpha (prevents divergence from degenerate steps).
LS_ALPHA_MAX = 1e4


@qd.func
def _func_decomp_linesearch_p0(
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Decomposed constraint solver P0 kernel: fused mv + jv + snorm + quad_gauss + eq_sum.

    Decomposed solver algorithm overview
    -------------------------------------
    A block of K=32 threads cooperates on each env for setup and apply; the linesearch refinement runs serially on
    thread 0 using func_linesearch_refine (shared with the monolith path).

    P0 kernel (this function):
        Phase 0a: Compute mv = M @ search (cooperative over DOFs, 32 threads).
        Phase 0b: Compute jv = J @ search (cooperative over constraints, 32 threads).
        Phase 1: Fused snorm + quad_gauss parallel reduction over n_dofs.
        Phase 2: Parallel reduction over n_constraints for eq_sum. Also computes alpha_newton.

    Eval kernel (_kernel_parallel_linesearch_eval):
        a) Serial refinement (thread 0): re-evaluate the Newton step via func_linesearch_refine.
        b) Apply: Update qacc, Ma, Jaref with the chosen alpha (cooperative over DOFs).

    Post-linesearch: Separate kernels for constraint force update, gradient update, Hessian update (Newton only), and
    search direction update. These reuse the batch-level functions from solver.py.
    """
    # Block size for shared-memory reductions.
    _T = qd.static(32)

    _B = constraint_state.grad.shape[1]

    qd.loop_config(name="parallel_linesearch_p0", block_dim=_T)
    for i_flat in range(_B * _T):
        tid = i_flat % _T
        i_b = i_flat // _T

        # 5 shared arrays for parallel reductions (reused across phases)
        sh_snorm_sq = qd.simt.block.SharedArray((_T,), gs.qd_float)
        sh_qg_grad = qd.simt.block.SharedArray((_T,), gs.qd_float)
        sh_qg_hess = qd.simt.block.SharedArray((_T,), gs.qd_float)
        sh_constraint_grad = qd.simt.block.SharedArray((_T,), gs.qd_float)
        sh_constraint_hess = qd.simt.block.SharedArray((_T,), gs.qd_float)

        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            n_dofs = constraint_state.search.shape[0]
            n_con = constraint_state.n_constraints[i_b]

            # === Phase 0a: Compute mv = M @ search (cooperative over DOFs) ===
            i_d1 = tid
            while i_d1 < n_dofs:
                I_d1 = [i_d1, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d1
                i_e = dyn_info.dofs.entity_idx[I_d1]
                mv_val = gs.qd_float(0.0)
                for i_d2 in range(dyn_info.entities.dof_start[i_e], dyn_info.entities.dof_end[i_e]):
                    mv_val = mv_val + rigid_info.mass_mat[i_d1, i_d2, i_b] * constraint_state.search[i_d2, i_b]
                constraint_state.mv[i_d1, i_b] = mv_val
                i_d1 += _T

            # === Phase 0b: Compute jv = J @ search (cooperative over constraints). Sparse over each constraint's
            # coupled DOFs (jac_dofs_idx) for CPU skyline / per-island GPU; islands-OFF GPU iterates dense to keep
            # the per-lane trip count uniform (no warp divergence), matching the non-island baseline. ===
            i_c = tid
            while i_c < n_con:
                jv_val = gs.qd_float(0.0)
                if qd.static(rigid_config.sparse_solve or rigid_config.enable_per_island_solve):
                    for i_d_ in range(constraint_state.jac_n_dofs[i_c, i_b]):
                        i_d = constraint_state.jac_dofs_idx[i_c, i_d_, i_b]
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
                local_qg_grad += s * constraint_state.Ma[i_d, i_b] - s * dyn_state.dofs.force[i_d, i_b]
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

            if snorm < rigid_info.EPS[None]:
                # Converged — only thread 0 writes
                if tid == 0:
                    constraint_state.ls_alpha[i_b] = 0.0
                    constraint_state.improved[i_b] = False
            else:
                # Thread 0 writes quad_gauss to global memory
                if tid == 0:
                    constraint_state.quad_gauss[0, i_b] = sh_qg_grad[0]
                    constraint_state.quad_gauss[1, i_b] = sh_qg_hess[0]

                # === Phase 2: Constraint cost, parallel over n_constraints ===
                ne = constraint_state.n_constraints_equality[i_b]
                nef = ne + constraint_state.n_constraints_frictionloss[i_b]
                ncone = nef
                if qd.static(rigid_config.enable_elliptic_friction):
                    ncone = ncone + constraint_state.n_constraints_cone[i_b]
                n_con = constraint_state.n_constraints[i_b]

                local_eq_grad = gs.qd_float(0.0)
                local_eq_hess = gs.qd_float(0.0)
                local_constraint_grad = gs.qd_float(0.0)
                local_constraint_hess = gs.qd_float(0.0)

                i_c = tid
                while i_c < n_con:
                    if qd.static(rigid_config.enable_elliptic_friction) and (nef <= i_c and i_c < ncone):
                        # Elliptic cone: the head thread carries the exact coupled grad/hess of the whole cone
                        # (evaluated at alpha=0 through the shared per-alpha routine, matching the serial linesearch);
                        # the friction-row threads are no-ops so the cone is counted once. The cone cost itself
                        # cancels in the shifted convention (see quad_gauss in array_class.py).
                        if (i_c - nef) % qd.static(rigid_config.rows_per_contact) == 0:
                            n_rows = qd.static(rigid_config.rows_per_contact)
                            rows_efc_D, rows_friction, con_mu, rows_jaref = solver._func_cone_head_load(
                                i_c, i_b, constraint_state, rigid_config
                            )
                            rows_jv = qd.Vector.zero(gs.qd_float, n_rows)
                            for i_r in qd.static(range(n_rows)):
                                rows_jv[i_r] = constraint_state.jv[i_c + i_r, i_b]
                            _c_cost, c_grad, c_hess = solver._func_cone_cost_along_alpha(
                                rows_jaref, rows_jv, 0.0, rows_efc_D, con_mu, rows_friction
                            )
                            local_constraint_grad += c_grad
                            local_constraint_hess += 0.5 * c_hess
                    else:
                        Jaref_c = constraint_state.Jaref[i_c, i_b]
                        jv_c = constraint_state.jv[i_c, i_b]
                        D = constraint_state.efc_D[i_c, i_b]
                        qf_1 = D * (jv_c * Jaref_c)
                        qf_2 = D * (0.5 * jv_c * jv_c)

                        if i_c < ne:
                            # Equality: always active
                            local_eq_grad += qf_1
                            local_eq_hess += qf_2
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
                                qf_1 = linear_neg * (-f * jv_c) + linear_pos * (f * jv_c)
                                qf_2 = 0.0
                            local_constraint_grad += qf_1
                            local_constraint_hess += qf_2
                        else:
                            # Contact / joint-limit: active if Jaref < 0
                            active = Jaref_c < 0
                            local_constraint_grad += qf_1 * active
                            local_constraint_hess += qf_2 * active

                    i_c += _T

                # Reuse shared arrays for Phase 2 reduction
                sh_qg_grad[tid] = local_eq_grad
                sh_qg_hess[tid] = local_eq_hess
                sh_constraint_grad[tid] = local_constraint_grad
                sh_constraint_hess[tid] = local_constraint_hess

                qd.simt.block.sync()

                # Tree reduction for 4 accumulators
                stride = _T // 2
                while stride > 0:
                    if tid < stride:
                        sh_qg_grad[tid] += sh_qg_grad[tid + stride]
                        sh_qg_hess[tid] += sh_qg_hess[tid + stride]
                        sh_constraint_grad[tid] += sh_constraint_grad[tid + stride]
                        sh_constraint_hess[tid] += sh_constraint_hess[tid + stride]
                    qd.simt.block.sync()
                    stride //= 2

                if tid == 0:
                    constraint_state.eq_sum[0, i_b] = sh_qg_grad[0]
                    constraint_state.eq_sum[1, i_b] = sh_qg_hess[0]
                    constraint_state.ls_it[i_b] = 1
                    # Initialize best alpha and improvement for parallel linesearch
                    constraint_state.ls_alpha[i_b] = 0.0  # default: no step
                    constraint_state.ls_improvement[i_b] = 0.0

                    # Newton step estimate from the full DOF + constraint gradient/hessian
                    total_hess = 2.0 * (constraint_state.quad_gauss[1, i_b] + sh_constraint_hess[0])
                    if total_hess > 0.0:
                        total_grad = constraint_state.quad_gauss[0, i_b] + sh_constraint_grad[0]
                        constraint_state.ls_alpha_newton[i_b] = qd.abs(total_grad / total_hess)
                    else:
                        constraint_state.ls_alpha_newton[i_b] = 0.0
                    # Store gtol for gradient-guided refinement
                    n_dofs_val = constraint_state.search.shape[0]
                    scale = rigid_info.meaninertia[i_b] * qd.max(1, n_dofs_val)
                    constraint_state.ls_gtol[i_b] = (
                        rigid_info.tolerance[None] * rigid_info.ls_tolerance[None] * snorm * scale
                    )


@qd.func
def _func_decomp_linesearch_refine_coop(
    i_b,
    tid,
    alpha_newton,
    gtol,
    constraint_state: array_class.ConstraintState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Warp-cooperative variant of ``_func_decomp_linesearch_refine``: all 32 lanes drive the unified
    ``_func_linesearch_eval_at_alpha`` / ``func_linesearch_refine`` (called with the Python literal ``True`` for the ``coop``
    template arg). Writes to ``ls_alpha`` / ``ls_improvement`` are tid==0-guarded since the result is per-env, not
    per-lane."""
    # Gated: skip when the Newton step is zero (degenerate hessian).
    if alpha_newton > 0.0:
        if tid == 0:
            constraint_state.ls_alpha[i_b] = 0.0
        p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = solver._func_linesearch_eval_at_alpha(
            i_b, tid, alpha_newton, constraint_state, rigid_info, rigid_config, coop=True
        )
        # Costs are shifted deltas from alpha=0 (see quad_gauss in array_class.py): positive means no improvement
        if p1_cost > 0.0:
            p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = solver._func_linesearch_eval_at_alpha(
                i_b, tid, 0.0, constraint_state, rigid_info, rigid_config, coop=True
            )
        if p1_cost < 0.0 and tid == 0:
            constraint_state.ls_alpha[i_b] = p1_alpha
            constraint_state.ls_improvement[i_b] = -p1_cost
        if qd.abs(p1_deriv_0) > gtol:
            res_alpha, res_cost, ls_result = solver.func_linesearch_refine(
                i_b,
                tid,
                p1_alpha,
                p1_cost,
                p1_deriv_0,
                p1_deriv_1,
                gtol,
                constraint_state,
                rigid_info,
                rigid_config,
                coop=True,
            )
            # Skip status 7 (brackets stalled, midpoint non-improving) to preserve the validated
            # p1_alpha already written above.
            if qd.abs(res_alpha) > rigid_info.EPS[None] and ls_result != 7 and tid == 0:
                constraint_state.ls_alpha[i_b] = res_alpha
                constraint_state.ls_improvement[i_b] = -res_cost


@qd.func
def _func_decomp_linesearch_refine_serial(
    i_b,
    tid,
    alpha_newton,
    gtol,
    constraint_state: array_class.ConstraintState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """1-thread-per-env variant of ``_func_decomp_linesearch_refine``: bit-identical to the pre-coop baseline.
    Only ``tid == 0`` runs the work; the unified helpers are called with the Python literal ``False`` for ``coop``."""
    # Gated: skip when the Newton step is zero (degenerate hessian).
    if alpha_newton > 0.0 and tid == 0:
        constraint_state.ls_alpha[i_b] = 0.0
        p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = solver._func_linesearch_eval_at_alpha(
            i_b, tid, alpha_newton, constraint_state, rigid_info, rigid_config, coop=False
        )
        # Costs are shifted deltas from alpha=0 (see quad_gauss in array_class.py): positive means no improvement
        if p1_cost > 0.0:
            p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = solver._func_linesearch_eval_at_alpha(
                i_b, tid, 0.0, constraint_state, rigid_info, rigid_config, coop=False
            )
        if p1_cost < 0.0:
            constraint_state.ls_alpha[i_b] = p1_alpha
            constraint_state.ls_improvement[i_b] = -p1_cost
        if qd.abs(p1_deriv_0) > gtol:
            res_alpha, res_cost, ls_result = solver.func_linesearch_refine(
                i_b,
                tid,
                p1_alpha,
                p1_cost,
                p1_deriv_0,
                p1_deriv_1,
                gtol,
                constraint_state,
                rigid_info,
                rigid_config,
                coop=False,
            )
            # Skip status 7 (brackets stalled, midpoint non-improving) to preserve the validated
            # p1_alpha already written above.
            if qd.abs(res_alpha) > rigid_info.EPS[None] and ls_result != 7:
                constraint_state.ls_alpha[i_b] = res_alpha
                constraint_state.ls_improvement[i_b] = -res_cost


@qd.func
def _func_decomp_linesearch_refine(
    i_b,
    tid,
    alpha_newton,
    gtol,
    constraint_state: array_class.ConstraintState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Linesearch refinement body, called per-env from ``_func_decomp_linesearch_refine_and_apply``. Dispatches at
    compile time on ``enable_cooperative_constraint_kernels`` to ``_func_decomp_linesearch_refine_coop``
    (warp-cooperative, all 32 lanes) or ``_func_decomp_linesearch_refine_serial`` (1-thread-per-env, bit-identical
    baseline).

    Each branch passes the literal Python ``True`` / ``False`` for the ``coop`` template arg of the unified
    ``_func_linesearch_eval_at_alpha`` / ``func_linesearch_refine``: Quadrants' ``qd.template()`` machinery does not
    auto-promote a struct member access to a compile-time value, so we cannot share a single call site here."""
    if qd.static(rigid_config.enable_cooperative_constraint_kernels):
        _func_decomp_linesearch_refine_coop(i_b, tid, alpha_newton, gtol, constraint_state, rigid_info, rigid_config)
    else:
        _func_decomp_linesearch_refine_serial(i_b, tid, alpha_newton, gtol, constraint_state, rigid_info, rigid_config)


@qd.func
def _func_decomp_linesearch_refine_and_apply(
    constraint_state: array_class.ConstraintState, rigid_info: array_class.RigidInfo, rigid_config: qd.template()
):
    """Decomposed solver eval kernel: linesearch refinement from Newton step + cooperative apply.

    The P0 kernel precomputes a Newton step (ls_alpha_newton). This kernel refines it via the unified
    ``func_linesearch_refine`` (templated on ``coop``: serial-on-tid-0 when False, cooperative across the 32-lane
    warp when True), gated on ``enable_cooperative_constraint_kernels``. It then cooperatively applies the chosen alpha
    to qacc, Ma, and Jaref.

    The cooperative path is only safe when the layout-flippable constraint-state tensors are stored with
    ``layout=(1, 0)`` (so per-lane strided reads of ``Jaref[i_c, i_b]`` etc. are coalesced across constraints for a
    fixed env). The qd.Tensor layout rewrite makes the canonical indexing identical in both paths; only the access
    pattern changes.
    """
    # Number of candidate step sizes evaluated simultaneously per env.
    # Each CUDA block processes one env with K threads, using shared memory for the argmin reduction.
    # Similar to BLOCK_DIM in func_hessian_direct_tiled: determines parallelism and shared memory layout.
    _K = qd.static(32)

    _B = constraint_state.grad.shape[1]

    qd.loop_config(name="parallel_linesearch_eval", block_dim=_K)
    for i_flat in range(_B * _K):
        tid = i_flat % _K
        i_b = i_flat // _K

        active = constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]

        if active:
            gtol = constraint_state.ls_gtol[i_b]
            alpha_newton = constraint_state.ls_alpha_newton[i_b]

            _func_decomp_linesearch_refine(i_b, tid, alpha_newton, gtol, constraint_state, rigid_info, rigid_config)
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
            if qd.abs(alpha_apply) < rigid_info.EPS[None]:
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
def _func_cg_only_save_prev_grad(constraint_state: array_class.ConstraintState, rigid_config: qd.template()):
    """Save prev_grad and prev_Mgrad (CG only)"""
    _B = constraint_state.grad.shape[1]
    qd.loop_config(name="cg_only_save_prev_grad", serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            solver.func_save_prev_grad(i_b, constraint_state)


@qd.func
def _func_update_constraint_forces_body(
    i_c, i_b, constraint_state: array_class.ConstraintState, rigid_config: qd.template()
):
    """Per-element body for ``_func_update_constraint_forces``. Factored out so the two
    ndrange orderings (coalescing-optimal for each layout) share a single implementation."""
    ne = constraint_state.n_constraints_equality[i_b]
    nef = ne + constraint_state.n_constraints_frictionloss[i_b]
    ncone = nef
    if qd.static(rigid_config.enable_elliptic_friction):
        ncone = ncone + constraint_state.n_constraints_cone[i_b]

    if qd.static(rigid_config.enable_elliptic_friction) and (nef <= i_c and i_c < ncone):
        # Elliptic cone (one-thread-per-row): only the head thread resolves the coupled rows and writes all of
        # them; the friction-row threads are no-ops (race-free). The coupled middle-zone cost is discarded here; the
        # linesearch evaluates the cone cost delta directly.
        if (i_c - nef) % qd.static(rigid_config.rows_per_contact) == 0:
            solver.func_cone_update_rows(i_c, i_b, constraint_state, rigid_config)
    else:
        if qd.static(
            rigid_config.solver_type == gs.constraint_solver.Newton and not rigid_config.enable_elliptic_friction
        ):
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
def _func_update_constraint_forces(constraint_state: array_class.ConstraintState, rigid_config: qd.template()):
    """Compute active flags and efc_force, parallelized over (constraint, env).

    Iteration order is picked at compile time so adjacent lanes always cover the *physical* contiguous dimension of the
    layout-flippable constraint-state tensors:
      - layout False (canonical [i_c, i_b], physical [i_c, i_b]):  ndrange(len_constraints, _B)
      - layout True  (canonical [i_c, i_b], physical [i_b, i_c]):  ndrange(_B, len_constraints)
    """
    len_constraints = constraint_state.active.shape[0]
    _B = constraint_state.grad.shape[1]

    # Snapshot prev_active in its own parallel pass so every row is captured before any active recompute: the cone head
    # thread rewrites its two tangent rows' active, which would otherwise race the tangent threads capturing
    # prev_active. Pyramidal threads only write their own row, so they snapshot inline in the body (no extra pass).
    if qd.static(rigid_config.solver_type == gs.constraint_solver.Newton and rigid_config.enable_elliptic_friction):
        qd.loop_config(name="snapshot_prev_active")
        for i_c, i_b in qd.ndrange(
            len_constraints, _B, axes=qd.static((1, 0) if rigid_config.enable_cooperative_constraint_kernels else None)
        ):
            if i_c < constraint_state.n_constraints[i_b] and constraint_state.improved[i_b]:
                constraint_state.prev_active[i_c, i_b] = constraint_state.active[i_c, i_b]

    qd.loop_config(name="update_constraint_forces")
    for i_c, i_b in qd.ndrange(
        len_constraints, _B, axes=qd.static((1, 0) if rigid_config.enable_cooperative_constraint_kernels else None)
    ):
        if i_c < constraint_state.n_constraints[i_b] and constraint_state.improved[i_b]:
            _func_update_constraint_forces_body(i_c, i_b, constraint_state, rigid_config)


@qd.func
def _func_update_qfrc_constraint_per_dof(constraint_state: array_class.ConstraintState, rigid_config: qd.template()):
    """Compute qfrc_constraint = J^T @ efc_force with one thread per (dof, env).

    With islands, a DOF only couples to constraints in its own island (a constraint touching the DOF is always in
    its island), so the sum runs over that island's constraints (constraint_id) rather than all n_con - identical
    result, but O(nnz) instead of O(n_dofs * n_con). The per-step constraint order is fixed, so the sum stays
    deterministic. Without islands it falls back to the dense scan over all constraints.

    Under ``enable_cooperative_constraint_kernels`` the outer ndrange is swapped so adjacent lanes vary i_d: the
    qfrc_constraint write coalesces under the flipped DOF-vec layout.
    """
    n_dofs = constraint_state.qfrc_constraint.shape[0]
    _B = constraint_state.grad.shape[1]

    qd.loop_config(name="update_constraint_qfrc")
    for i_d, i_b in qd.ndrange(
        n_dofs, _B, axes=qd.static((1, 0) if rigid_config.enable_cooperative_constraint_kernels else None)
    ):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            qfrc = gs.qd_float(0.0)
            if qd.static(rigid_config.enable_per_island_solve):
                i_island = constraint_state.island.dofs_island_idx[i_d, i_b]
                if i_island >= 0:
                    con_base = constraint_state.island.constraint_slices.start[i_island, i_b]
                    con_n = constraint_state.island.constraint_slices.n[i_island, i_b]
                    for i_lcon in range(con_n):
                        i_c = constraint_state.island.constraint_id[con_base + i_lcon, i_b]
                        qfrc += constraint_state.jac[i_c, i_d, i_b] * constraint_state.efc_force[i_c, i_b]
            else:
                n_con = constraint_state.n_constraints[i_b]
                for i_c in range(n_con):
                    qfrc += constraint_state.jac[i_c, i_d, i_b] * constraint_state.efc_force[i_c, i_b]
            constraint_state.qfrc_constraint[i_d, i_b] = qfrc


# Number of full Hessian+Cholesky rebuilds at the start of the solver loop (after the init's iter-0 full rebuild).
# 0 = all incremental, 2 = full for loop iters 0-1 then incremental, 999 = always full.
@qd.func
def _func_build_changed_and_decide_hessian_mode(
    constraint_state: array_class.ConstraintState, rigid_config: qd.template()
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
            solver.func_build_changed_constraint_list(i_b, constraint_state)
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
def _func_patch_hessian_delta(constraint_state: array_class.ConstraintState, rigid_info: array_class.RigidInfo):
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
def _func_newton_only_nt_hessian(constraint_state: array_class.ConstraintState, rigid_info: array_class.RigidInfo):
    """Full tiled Hessian rebuild for envs with use_full_hessian == 1 (skips others)."""
    solver.func_hessian_direct_tiled(constraint_state, rigid_info, check_full_hessian=True)


@qd.func
def _func_wrap_cone_hessian(
    constraint_state: array_class.ConstraintState, rigid_config: qd.template(), scale: qd.template()
):
    """Add (scale=+1) or remove (scale=-1) the coupled elliptic-cone Hessian block across all improved envs.

    Bracketing a non-destructive factor+solve (whole-env fused or per-island tiled) with +1 then -1 lets the cone ride
    the incrementally maintained nt_H without a per-iteration full rebuild: the current cone block is present while
    the factor reads nt_H, then removed so the next iteration patches a cone-free Hessian. A no-op unless the elliptic
    cone is active.
    """
    if qd.static(rigid_config.enable_elliptic_friction):
        _B = constraint_state.jac.shape[2]
        qd.loop_config(name="wrap_cone_hessian", serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
        for i_b in range(_B):
            if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
                solver.func_add_cone_hessian_block(i_b, constraint_state, rigid_config, scale)


@qd.func
def _func_newton_only_nt_hessian_and_cholesky(
    constraint_state: array_class.ConstraintState, rigid_info: array_class.RigidInfo, rigid_config: qd.template()
):
    """Full Hessian rebuild + Cholesky for ALL improved envs (non-fused path).

    Matches origin/main behavior: H is rebuilt from scratch every iteration, then Cholesky overwrites nt_H with L
    in-place.  H patching is not used because the subsequent Cholesky would destroy H anyway.
    """
    solver.func_hessian_direct_tiled(constraint_state, rigid_info)
    # func_hessian_direct_tiled assembles M + J^T D J only; add the coupled elliptic-cone block as an additive
    # post-pass before the factor reads nt_H, matching the monolith tiled path (this path rebuilds every improved env).
    if qd.static(rigid_config.enable_elliptic_friction):
        _B_envs = constraint_state.jac.shape[2]
        qd.loop_config(
            name="add_cone_hessian_block", serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32
        )
        for i_b in range(_B_envs):
            if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
                solver.func_add_cone_hessian_block(i_b, constraint_state, rigid_config)
    if qd.static(rigid_config.enable_tiled_cholesky_hessian):
        solver.func_cholesky_factor_direct_tiled(constraint_state, rigid_info, rigid_config)
    else:
        _B = constraint_state.jac.shape[2]
        qd.loop_config(
            name="cholesky_factor_direct_batch", serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32
        )
        for i_b in range(_B):
            if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
                # Decomposed arm is non-island: i_island = 0 is the full-env work-unit (island branch is dead).
                solver.func_cholesky_factor_direct_batch(i_b, 0, constraint_state, rigid_info, rigid_config)


@qd.func
def _func_update_gradient(
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Step 5: Update gradient"""
    _B = constraint_state.grad.shape[1]
    qd.loop_config(name="update_gradient", serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            solver.func_update_gradient_batch(i_b, dyn_state, constraint_state, dyn_info, rigid_info, rigid_config)


@qd.func
def _func_update_search_direction(
    constraint_state: array_class.ConstraintState, rigid_info: array_class.RigidInfo, rigid_config: qd.template()
):
    """Step 6: Check convergence and update search direction"""
    _B = constraint_state.grad.shape[1]
    qd.loop_config(name="update_search_direction", serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            solver.func_terminate_or_update_descent_batch(i_b, constraint_state, rigid_info, rigid_config)


@qd.func
def _func_check_early_exit(
    graph_counter: qd.types.ndarray(qd.i32, ndim=0), constraint_state: array_class.ConstraintState
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


@qd.kernel(graph=True, fastcache=True)
def _kernel_solve_graph(
    graph_counter: qd.types.ndarray(qd.i32, ndim=0),
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    while qd.graph_do_while(graph_counter):
        # Fused: mv + jv + snorm + quad_gauss + eq_sum
        _func_decomp_linesearch_p0(dyn_state, constraint_state, dyn_info, rigid_info, rigid_config)
        # Fused: refinement + apply alpha
        _func_decomp_linesearch_refine_and_apply(constraint_state, rigid_info, rigid_config)
        if qd.static(rigid_config.solver_type == gs.constraint_solver.CG):
            _func_cg_only_save_prev_grad(constraint_state, rigid_config)
        _func_update_constraint_forces(constraint_state, rigid_config)
        _func_update_qfrc_constraint_per_dof(constraint_state, rigid_config)
        if qd.static(rigid_config.solver_type == gs.constraint_solver.Newton and rigid_config.hessian_fits_shared):
            # Incremental Hessian assembly - full rebuild when the active set changed a lot, delta patch otherwise -
            # then a tiled factor + solve reading the maintained nt_H. Each changed constraint's J^T D J lands inside
            # its island's diagonal block (no constraint couples DOFs across islands), so the patch is island-correct.
            _func_build_changed_and_decide_hessian_mode(constraint_state, rigid_config)
            _func_newton_only_nt_hessian(constraint_state, rigid_info)
            _func_patch_hessian_delta(constraint_state, rigid_info)
            solver.func_update_gradient_no_solve(dyn_state, constraint_state, dyn_info, rigid_info, rigid_config)
            if qd.static(rigid_config.enable_per_island_solve):
                # Hibernation needs the per-island grid to skip asleep islands, so factor + solve each awake island in
                # its own tile over the (env, island) grid. The per-island tile stages nt_H non-destructively, so the
                # elliptic cone rides the same add/factor+solve/remove bracket as the whole-env fused path below and
                # the incremental patch keeps maintaining the cone-free Hessian.
                _func_wrap_cone_hessian(constraint_state, rigid_config, 1.0)
                solver.func_island_tiled_factor_solve_all(
                    constraint_state,
                    dyn_info,
                    rigid_info,
                    rigid_config,
                    qd.simt.Tile32x32 if qd.static(rigid_config.cholesky_tile_size == 32) else qd.simt.Tile16x16,
                )
                _func_wrap_cone_hessian(constraint_state, rigid_config, -1.0)
            else:
                # Islands OFF, or islands ON without hibernation: the whole-env Hessian is block-diagonal by island, so
                # its Cholesky is itself block-diagonal - the whole-env fused factor+solve (L in shared memory) yields
                # the exact per-island result with none of the per-(env, island) grid/indirection overhead, which is
                # pure cost at the env counts where the env dimension alone already saturates the GPU. The per-island
                # grid only pays off when the whole-env Hessian does not fit shared (the cooperative branch below).
                # For the elliptic cone, add its coupled block to the maintained nt_H, factor+solve, then remove it, so
                # the incremental patch keeps working on the cone-free Hessian (no per-iteration rebuild).
                _func_wrap_cone_hessian(constraint_state, rigid_config, 1.0)
                solver.func_cholesky_and_solve_fused_tiled(constraint_state, rigid_info, rigid_config)
                _func_wrap_cone_hessian(constraint_state, rigid_config, -1.0)
        elif qd.static(
            rigid_config.solver_type == gs.constraint_solver.Newton
            and rigid_config.enable_per_island_solve
            and rigid_config.enable_cooperative_constraint_kernels
            # The in-tile assembly builds M + J^T D J only and overwrites nt_H, so the coupled elliptic-cone block can
            # neither be baked beforehand nor bracketed around it; elliptic falls through to the whole-env rebuild
            # below, whose post-pass bakes the cone before the factor.
            and not rigid_config.enable_elliptic_friction
        ):
            # Hibernation with a whole-env Hessian too big for shared but each island's block fitting the per-island
            # tile: assemble + factor + solve each awake island in its own tile (do_assemble=True), with NO whole-env
            # assemble + factor + solve each island in its own tile (do_assemble=True), with NO whole-env Hessian
            # touched. This keeps the cost at sum-of-per-island-blocks instead of the whole-env O(n_dofs^3) factor the
            # non-fused path below would do - the regime of many small islands whose total dof count exceeds the shared
            # cap. An island larger than the per-island tile falls back to the scalar per-island solve inside the factor.
            solver.func_update_gradient_no_solve(dyn_state, constraint_state, dyn_info, rigid_info, rigid_config)
            solver.func_island_tiled_factor_solve_all(
                constraint_state,
                dyn_info,
                rigid_info,
                rigid_config,
                qd.simt.Tile32x32 if qd.static(rigid_config.cholesky_tile_size == 32) else qd.simt.Tile16x16,
                do_assemble=True,
            )
        elif qd.static(rigid_config.solver_type == gs.constraint_solver.Newton):
            # Non-fused path: full whole-env H rebuild + separate Cholesky every iteration (Cholesky overwrites nt_H
            # with L, so H patching is not possible). Reached when the whole-env Hessian does not fit shared and either
            # islands are OFF (a single whole-env factor, matching the non-island baseline) or the cooperative kernels
            # are disabled (tiny n_dofs or huge env count), where the whole-env factor is cheap or the monolith wins.
            _func_newton_only_nt_hessian_and_cholesky(constraint_state, rigid_info, rigid_config)
            _func_update_gradient(dyn_state, constraint_state, dyn_info, rigid_info, rigid_config)
        else:
            _func_update_gradient(dyn_state, constraint_state, dyn_info, rigid_info, rigid_config)
        _func_update_search_direction(constraint_state, rigid_info, rigid_config)
        _func_check_early_exit(graph_counter, constraint_state)


@solver.func_solve_body.register(
    is_compatible=lambda *args, **kwargs: (
        not (rigid_config := solver._get_static_config(*args, **kwargs)).requires_grad
        and rigid_config.prefer_decomposed_solver != 0
    )
)
def func_solve_decomposed(dyn_state, constraint_state, dyn_info, rigid_info, rigid_config, _n_iterations):
    """
    GPU graph accelerated solver loop with parallel grid-search linesearch and GPU-side iteration via graph_do_while.

    On CUDA SM 9.0+ (Hopper), the entire iteration loop runs on the GPU with no host involvement. On older CUDA GPUs,
    falls back to a host-side do-while loop that still benefits from CUDA graph kernel launch batching. On other GPUs,
    falls back to a host-side C++-side loop, that still reduces python launch overhead.

    Early exits when all batch elements have converged (no improved[i_b] is True).

    Islands ON/OFF share this same graph loop: with islands the per-iteration factor/solve runs per-island over the
    (env, island) grid, while an unpartitioned env is a single island spanning every dof. Early exits when all batch
    elements have converged (no improved[i_b] is True).
    """
    # This entrypoint statically IS the decomposed arm, so it owns its init: it forwards is_decomposed=True to
    # func_solve_init, which builds the island partition but then skips the init Hessian factor + gradient. The graph
    # rebuilds the Hessian on its first iteration regardless (iter_count <= 1 -> use_full_hessian), so the init factor
    # would be pure waste, and skipping it makes the decomposed arm behave identically for islands ON and OFF.
    solver.func_solve_init(dyn_state, constraint_state, dyn_info, rigid_info, rigid_config, is_decomposed=True)
    if _n_iterations <= 0:
        return
    constraint_state.graph_counter.from_numpy(np.array(_n_iterations, dtype=np.int32))
    _kernel_solve_graph(constraint_state.graph_counter, dyn_state, constraint_state, dyn_info, rigid_info, rigid_config)
