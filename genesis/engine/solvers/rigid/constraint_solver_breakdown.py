import gstaichi as ti

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.engine.solvers.rigid.constraint_solver as constraint_solver


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_solve_body_decomposed(
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    """
    Single kernel containing all solver steps as separate top-level loops.

    This reduces Python→C++ boundary crossing overhead (1 call per iteration instead of 6)
    while still allowing Taichi to launch each step as a separate GPU kernel internally.
    """
    _B = constraint_state.grad.shape[1]

    # Step 1: Linesearch and update qacc, Ma, Jaref
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    # Index: 0
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            constraint_solver.func_linesearch_top_level(
                i_b,
                entities_info=entities_info,
                dofs_state=dofs_state,
                rigid_global_info=rigid_global_info,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )
        else:
            constraint_state.improved[i_b] = False

    # Step 2: Save prev_grad and prev_Mgrad (CG only)
    if ti.static(static_rigid_sim_config.solver_type == gs.constraint_solver.CG):
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
        # Index: 1 if CG
        for i_b in range(_B):
            if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
                constraint_solver.func_save_prev_grad(i_b, constraint_state=constraint_state)

    # Step 3: Update constraints
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    # Index: 1 if Newton else 2
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            constraint_solver.func_update_constraint(
                i_b,
                qacc=constraint_state.qacc,
                Ma=constraint_state.Ma,
                cost=constraint_state.cost,
                dofs_state=dofs_state,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )

    # Step 4: Newton Hessian update (Newton only)
    if ti.static(static_rigid_sim_config.solver_type == gs.constraint_solver.Newton):
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
        # Index: 2 if Newton
        for i_b in range(_B):
            if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
                constraint_solver.func_nt_hessian_incremental(
                    i_b,
                    entities_info=entities_info,
                    constraint_state=constraint_state,
                    rigid_global_info=rigid_global_info,
                    static_rigid_sim_config=static_rigid_sim_config,
                )

    # Step 5: Update gradient
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    # Index: 3
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            constraint_solver.func_update_gradient(
                i_b,
                dofs_state=dofs_state,
                entities_info=entities_info,
                rigid_global_info=rigid_global_info,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )

    # Step 6: Check convergence and update search direction
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    # Index: 4
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            constraint_solver.func_update_search_direction(
                i_b,
                rigid_global_info=rigid_global_info,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )


def func_solve_decomposed_microkernels(
    entities_info,
    dofs_state,
    constraint_state,
    rigid_global_info,
    static_rigid_sim_config,
):
    """
    Uses a single kernel with multiple top-level for loops per iteration, reducing
    Python→C++ boundary crossing overhead from 6× to 1× per iteration.
    """
    iterations = rigid_global_info.iterations[None]
    for _it in range(iterations):
        _kernel_solve_body_decomposed(
            entities_info,
            dofs_state,
            constraint_state,
            rigid_global_info,
            static_rigid_sim_config,
        )


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_linesearch_top_level(
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
            constraint_solver.func_linesearch_top_level(
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
def _kernel_cg_only_save_prev_grad(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
):
    """Save prev_grad and prev_Mgrad (CG only)"""
    _B = constraint_state.grad.shape[1]
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            constraint_solver.func_save_prev_grad(i_b, constraint_state=constraint_state)


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
            constraint_solver.func_update_constraint(
                i_b,
                qacc=constraint_state.qacc,
                Ma=constraint_state.Ma,
                cost=constraint_state.cost,
                dofs_state=dofs_state,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_newton_only_nt_hessian_incremental(
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    """Step 4: Newton Hessian update (Newton only)"""
    _B = constraint_state.grad.shape[1]
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            constraint_solver.func_nt_hessian_incremental(
                i_b,
                entities_info=entities_info,
                constraint_state=constraint_state,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
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
            constraint_solver.func_update_gradient(
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
            constraint_solver.func_update_search_direction(
                i_b,
                rigid_global_info=rigid_global_info,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )


_decomposed_timer = None
_decomposed_call_count = 0


def func_solve_decomposed_macrokernels(
    entities_info,
    dofs_state,
    constraint_state,
    rigid_global_info,
    static_rigid_sim_config,
):
    """
    Uses separate kernels for each solver step per iteration.

    This maximizes kernel granularity, potentially allowing better GPU scheduling
    and more flexibility in execution, at the cost of more Python→C++ boundary crossings
    (6× per iteration instead of 1×).
    """
    global _decomposed_timer, _decomposed_call_count
    import os

    do_profile = os.environ.get("GS_PROFILE_DECOMPOSED", "0") == "1"
    if do_profile:
        from genesis.utils.tools import create_timer

        _decomposed_call_count += 1
        skip = (_decomposed_call_count % 100) != 0
        _decomposed_timer = create_timer("decomposed", new=True, ti_sync=True, skip_first_call=skip)

    iterations = rigid_global_info.iterations[None]
    for _it in range(iterations):
        _kernel_linesearch_top_level(
            entities_info,
            dofs_state,
            constraint_state,
            rigid_global_info,
            static_rigid_sim_config,
        )
        if do_profile and _decomposed_timer:
            _decomposed_timer._stamp("linesearch_top_level")
        if static_rigid_sim_config.solver_type == gs.constraint_solver.CG:
            _kernel_cg_only_save_prev_grad(
                constraint_state,
                static_rigid_sim_config,
            )
            if do_profile and _decomposed_timer:
                _decomposed_timer._stamp("cg_save_prev_grad")
        _kernel_update_constraint(
            entities_info,
            dofs_state,
            constraint_state,
            rigid_global_info,
            static_rigid_sim_config,
        )
        if do_profile and _decomposed_timer:
            _decomposed_timer._stamp("update_constraint")
        if static_rigid_sim_config.solver_type == gs.constraint_solver.Newton:
            _kernel_newton_only_nt_hessian_incremental(
                entities_info,
                constraint_state,
                rigid_global_info,
                static_rigid_sim_config,
            )
            if do_profile and _decomposed_timer:
                _decomposed_timer._stamp("nt_hessian")
        _kernel_update_gradient(
            entities_info,
            dofs_state,
            constraint_state,
            rigid_global_info,
            static_rigid_sim_config,
        )
        if do_profile and _decomposed_timer:
            _decomposed_timer._stamp("update_gradient")
        _kernel_update_search_direction(
            constraint_state,
            rigid_global_info,
            static_rigid_sim_config,
        )
        if do_profile and _decomposed_timer:
            _decomposed_timer._stamp("update_search_dir")


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_linesearch_multi_probe(
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
            constraint_solver.func_linesearch_multi_probe_top_level(
                i_b,
                entities_info=entities_info,
                dofs_state=dofs_state,
                rigid_global_info=rigid_global_info,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )
        else:
            constraint_state.improved[i_b] = False


def func_solve_multi_probe_macrokernels(
    entities_info,
    dofs_state,
    constraint_state,
    rigid_global_info,
    static_rigid_sim_config,
):
    """
    Uses multi-probe linesearch with separate kernels for each solver step per iteration.
    The linesearch evaluates multiple candidate alphas per iteration for faster convergence.
    """
    iterations = rigid_global_info.iterations[None]
    for _it in range(iterations):
        _kernel_linesearch_multi_probe(
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


# ======================== Parallel-probe linesearch kernels ========================
#
# Parallelized version of func_linesearch (constraint_solver.py:1672).
#
# The original func_linesearch runs entirely inside a single i_b thread:
#   - Bracketing while-loop (line 1729): 1 Newton step per iteration (sequential)
#   - Refinement while-loop (line 1765): 3 candidates per iteration (sequential)
# In Taichi, only top-level `for` loops are parallelized, so all linesearch work
# for a given i_b was serial.
#
# This parallel version splits the linesearch into 4 kernels:
#   _kernel_par_ls_init   (1D: for i_b)            → constraint_solver.py:1680-1737
#   _kernel_par_ls_eval   (2D: for i_b, i_p)       → constraint_solver.py:1550-1601 (func_ls_point_fn)
#   _kernel_par_ls_select (1D: for i_b)             → constraint_solver.py:1729-1848 + 1888-1933 (update_bracket)
#   _kernel_par_ls_apply  (1D: for i_b)             → constraint_solver.py:1852-1884 (func_linesearch_top_level)
#
# The key optimization is _kernel_par_ls_eval: by using `for i_b, i_p in ti.ndrange(_B, 3)`,
# we evaluate 3 candidate alphas in parallel per batch element. Both phases use all 3 probes:
#   - Bracketing: Newton step, half-Newton step, double-Newton step
#   - Refinement: p1_next, p2_next, pmid (same 3 candidates as original)
#
# State is communicated between kernels via ls_par_* fields in ConstraintState.
# A phase field (DONE/BRACKETING/REFINEMENT) encodes the state machine per batch element.

N_PAR_PROBES = 3

# Phase constants
_PAR_PHASE_DONE = 0
_PAR_PHASE_BRACKETING = 1
_PAR_PHASE_REFINEMENT = 2


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_par_ls_init(
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    """
    Initialize parallel-probe linesearch.

    Maps to constraint_solver.py func_linesearch (line 1672), lines 1680-1737:
      - Lines 1682-1688 → snorm/gtol computation
      - Line  1696-1698 → early exit if snorm < EPS (ls_result=1)
      - Lines 1700-1707 → func_ls_init(): precompute mv, jv, quad_gauss, quad
      - Lines 1709-1713 → eval p0 (alpha=0) and p1 (Newton step) via func_ls_point_fn
      - Line  1716-1717 → if p0 is cheaper, reset p1 = p0
      - Lines 1719-1724 → convergence: if |p1_d0| < gtol → DONE
      - Lines 1726-1728 → compute direction, set p2 = p1, p2update = 0
      - Line  1735      → [CHANGED] original generates 1 Newton candidate;
                           parallel version generates 3: Newton, half-Newton, double-Newton

    Also mirrors func_linesearch_top_level (line 1852) guard: skip if !improved.
    """
    _B = constraint_state.grad.shape[1]
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            n_dofs = constraint_state.search.shape[0]

            # snorm, gtol — lines 1682-1688
            snorm = gs.ti_float(0.0)
            for jd in range(n_dofs):
                snorm = snorm + constraint_state.search[jd, i_b] ** 2
            snorm = ti.sqrt(snorm)
            scale = rigid_global_info.meaninertia[i_b] * ti.max(1, n_dofs)
            gtol = rigid_global_info.tolerance[None] * rigid_global_info.ls_tolerance[None] * snorm * scale
            constraint_state.gtol[i_b] = gtol

            constraint_state.ls_it[i_b] = 0
            constraint_state.ls_result[i_b] = 0

            if snorm < rigid_global_info.EPS[None]:
                # line 1697
                constraint_state.ls_result[i_b] = 1
                constraint_state.ls_par_res_alpha[i_b] = 0.0
                constraint_state.ls_par_phase[i_b] = _PAR_PHASE_DONE
            else:
                # lines 1700-1707: func_ls_init
                constraint_solver.func_ls_init(
                    i_b,
                    entities_info=entities_info,
                    dofs_state=dofs_state,
                    constraint_state=constraint_state,
                    rigid_global_info=rigid_global_info,
                    static_rigid_sim_config=static_rigid_sim_config,
                )

                # lines 1709-1713: eval p0 and p1
                p0_alpha, p0_cost, p0_d0, p0_d1 = constraint_solver.func_ls_point_fn(
                    i_b, gs.ti_float(0.0), constraint_state, rigid_global_info
                )
                p1_alpha, p1_cost, p1_d0, p1_d1 = constraint_solver.func_ls_point_fn(
                    i_b, p0_alpha - p0_d0 / p0_d1, constraint_state, rigid_global_info
                )

                # line 1716
                if p0_cost < p1_cost:
                    p1_alpha, p1_cost, p1_d0, p1_d1 = p0_alpha, p0_cost, p0_d0, p0_d1

                constraint_state.ls_par_p0_cost[i_b] = p0_cost

                # lines 1719-1724: convergence check
                if ti.abs(p1_d0) < gtol:
                    if ti.abs(p1_alpha) < rigid_global_info.EPS[None]:
                        constraint_state.ls_result[i_b] = 2
                    else:
                        constraint_state.ls_result[i_b] = 0
                    constraint_state.ls_par_res_alpha[i_b] = p1_alpha
                    constraint_state.ls_par_phase[i_b] = _PAR_PHASE_DONE
                else:
                    # lines 1726-1728: direction, p2 = p1
                    direction = (p1_d0 < 0) * 2 - 1
                    constraint_state.ls_par_direction[i_b] = direction
                    constraint_state.ls_par_p2update[i_b] = 0

                    constraint_state.ls_par_p1[0, i_b] = p1_alpha
                    constraint_state.ls_par_p1[1, i_b] = p1_cost
                    constraint_state.ls_par_p1[2, i_b] = p1_d0
                    constraint_state.ls_par_p1[3, i_b] = p1_d1

                    constraint_state.ls_par_p2[0, i_b] = p1_alpha
                    constraint_state.ls_par_p2[1, i_b] = p1_cost
                    constraint_state.ls_par_p2[2, i_b] = p1_d0
                    constraint_state.ls_par_p2[3, i_b] = p1_d1

                    # Generate 3 bracketing candidates: Newton, half-Newton, double-Newton
                    step = -p1_d0 / p1_d1
                    constraint_state.ls_par_cand_alpha[0, i_b] = p1_alpha + step
                    constraint_state.ls_par_cand_alpha[1, i_b] = p1_alpha + 0.5 * step
                    constraint_state.ls_par_cand_alpha[2, i_b] = p1_alpha + 2.0 * step
                    constraint_state.ls_par_phase[i_b] = _PAR_PHASE_BRACKETING
        else:
            constraint_state.improved[i_b] = False
            constraint_state.ls_par_phase[i_b] = _PAR_PHASE_DONE


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_par_ls_eval(
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
):
    """
    Parallel evaluation of candidate alphas across (batch, probe) pairs.

    Maps to constraint_solver.py func_ls_point_fn (line 1550):
      - Lines 1556-1557 → ne, nef: constraint type boundaries
      - Lines 1560-1562 → init tmp from quad_gauss[0..2, i_b]
      - Lines 1563-1590 → loop over constraints, accumulate quadratic cost terms:
          - Lines 1569-1584 → friction constraints: linearization when |x| >= rf
          - Lines 1585-1586 → contact constraints: active only when x < 0
      - Lines 1592-1597 → cost = α²·tmp2 + α·tmp1 + tmp0, derivs

    Key difference from original:
      - Original: func_ls_point_fn is called sequentially, one alpha at a time,
        inside the bracketing loop (line 1735) or refinement loop (lines 1761, 1766).
      - Parallel: all 3 candidate alphas are evaluated simultaneously via
        `for i_b, i_p in ti.ndrange(_B, 3)`, parallelizing across batch × probe.
      - ls_it is NOT incremented here (deferred to _kernel_par_ls_select);
        the original func_ls_point_fn increments ls_it at line 1599.
    """
    _B = constraint_state.grad.shape[1]
    for i_b, i_p in ti.ndrange(_B, N_PAR_PROBES):
        phase = constraint_state.ls_par_phase[i_b]
        if phase != _PAR_PHASE_DONE:
            alpha = constraint_state.ls_par_cand_alpha[i_p, i_b]

            ne = constraint_state.n_constraints_equality[i_b]
            nef = ne + constraint_state.n_constraints_frictionloss[i_b]

            tmp_0 = constraint_state.quad_gauss[0, i_b]
            tmp_1 = constraint_state.quad_gauss[1, i_b]
            tmp_2 = constraint_state.quad_gauss[2, i_b]

            for i_c in range(constraint_state.n_constraints[i_b]):
                x = constraint_state.Jaref[i_c, i_b] + alpha * constraint_state.jv[i_c, i_b]
                qf_0 = constraint_state.quad[i_c, 0, i_b]
                qf_1 = constraint_state.quad[i_c, 1, i_b]
                qf_2 = constraint_state.quad[i_c, 2, i_b]

                active = gs.ti_bool(True)
                if ne <= i_c and i_c < nef:
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
                elif nef <= i_c:
                    active = x < 0

                tmp_0 = tmp_0 + qf_0 * active
                tmp_1 = tmp_1 + qf_1 * active
                tmp_2 = tmp_2 + qf_2 * active

            cost = alpha * alpha * tmp_2 + alpha * tmp_1 + tmp_0
            deriv_0 = 2 * alpha * tmp_2 + tmp_1
            deriv_1 = 2 * tmp_2
            if deriv_1 <= 0.0:
                deriv_1 = rigid_global_info.EPS[None]

            constraint_state.ls_par_cand_cost[i_p, i_b] = cost
            constraint_state.ls_par_cand_deriv0[i_p, i_b] = deriv_0
            constraint_state.ls_par_cand_deriv1[i_p, i_b] = deriv_1


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_par_ls_select(
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
):
    """
    Select best candidate and advance the linesearch state machine.

    BRACKETING phase — maps to constraint_solver.py func_linesearch lines 1729-1763:
      - Lines 1729-1731 → while-loop condition: p1_d0 * direction <= -gtol && ls_it < max
        [CHANGED] Original checks 1 candidate per iteration. Parallel checks all 3.
      - Lines 1732-1733 → p2 = old p1, p2update = 1
      - Line  1735      → p1 = func_ls_point_fn(Newton step)
        [CHANGED] Original takes 1 Newton step. Parallel picks best of 3 candidates:
        candidates still on bracketing side (d0*direction <= -gtol) advance p1 toward
        zero (same sign, closest to zero — same logic as update_bracket line 1888);
        candidates that crossed become bracket boundaries.
      - Lines 1738-1741 → convergence: if |p1_d0| < gtol → DONE
      - Lines 1743-1746 → max iterations → DONE (ls_result=3)
      - Lines 1748-1751 → if !p2update → DONE (ls_result=6)
      - Lines 1753-1763 → transition to REFINEMENT: generate p1_next, p2_next=p1, pmid

    REFINEMENT phase — maps to constraint_solver.py func_linesearch lines 1765-1848
    and update_bracket (line 1888):
      - Lines 1770-1790 → 3 candidates: [0]=p1_next, [1]=p2_next, [2]=pmid
      - Lines 1792-1801 → convergence check on all 3, pick lowest cost
      - Lines 1804-1816 → update_bracket for p1 (line 1888-1933):
        iterate over 3 candidates, find same-sign closest-to-zero derivative.
        [CHANGED] Original calls func_ls_point_fn(Newton) immediately (line 1930);
        parallel version defers eval — writes alpha to ls_par_cand_alpha for next round.
      - Lines 1817-1829 → update_bracket for p2 (same logic)
      - Lines 1831-1837 → if b1==0 and b2==0: use pmid, DONE (ls_result=0 or 7)
      - Lines 1839-1848 → max iterations: pick best of p1/p2 (ls_result=4 or 5)
      - Otherwise: generate next 3 candidates (p1_next, p2_next, pmid) for next round
    """
    _B = constraint_state.grad.shape[1]
    ti.loop_config(serialize=True)
    for i_b in range(_B):
        phase = constraint_state.ls_par_phase[i_b]

        if phase == _PAR_PHASE_BRACKETING:
            # --- Paralleled version of bracketing while-loop (original lines 1729-1763) ---
            gtol = constraint_state.gtol[i_b]
            direction = constraint_state.ls_par_direction[i_b]

            # Read current p1
            p1_alpha = constraint_state.ls_par_p1[0, i_b]
            p1_cost = constraint_state.ls_par_p1[1, i_b]
            p1_d0 = constraint_state.ls_par_p1[2, i_b]
            p1_d1 = constraint_state.ls_par_p1[3, i_b]

            # Read all 3 bracketing candidates (Newton, half-Newton, double-Newton)
            c0_a = constraint_state.ls_par_cand_alpha[0, i_b]
            c0_c = constraint_state.ls_par_cand_cost[0, i_b]
            c0_d0 = constraint_state.ls_par_cand_deriv0[0, i_b]
            c0_d1 = constraint_state.ls_par_cand_deriv1[0, i_b]

            c1_a = constraint_state.ls_par_cand_alpha[1, i_b]
            c1_c = constraint_state.ls_par_cand_cost[1, i_b]
            c1_d0 = constraint_state.ls_par_cand_deriv0[1, i_b]
            c1_d1 = constraint_state.ls_par_cand_deriv1[1, i_b]

            c2_a = constraint_state.ls_par_cand_alpha[2, i_b]
            c2_c = constraint_state.ls_par_cand_cost[2, i_b]
            c2_d0 = constraint_state.ls_par_cand_deriv0[2, i_b]
            c2_d1 = constraint_state.ls_par_cand_deriv1[2, i_b]

            # ls_it += 3 (3 parallel evaluations; original line 1599 increments by 1)
            constraint_state.ls_it[i_b] = constraint_state.ls_it[i_b] + N_PAR_PROBES

            # Check convergence on all 3 (original line 1738: checks 1 candidate)
            best_conv_i = -1
            best_conv_cost = gs.ti_float(0.0)
            best_conv_alpha = gs.ti_float(0.0)

            if ti.abs(c0_d0) < gtol and (best_conv_i < 0 or c0_c < best_conv_cost):
                best_conv_i = 0
                best_conv_cost = c0_c
                best_conv_alpha = c0_a
            if ti.abs(c1_d0) < gtol and (best_conv_i < 0 or c1_c < best_conv_cost):
                best_conv_i = 1
                best_conv_cost = c1_c
                best_conv_alpha = c1_a
            if ti.abs(c2_d0) < gtol and (best_conv_i < 0 or c2_c < best_conv_cost):
                best_conv_i = 2
                best_conv_cost = c2_c
                best_conv_alpha = c2_a

            if best_conv_i >= 0:
                # Original lines 1738-1741: convergence found
                if ti.abs(best_conv_alpha) < rigid_global_info.EPS[None]:
                    constraint_state.ls_result[i_b] = 2  # line 1721
                else:
                    constraint_state.ls_result[i_b] = 0   # line 1723
                constraint_state.ls_par_res_alpha[i_b] = best_conv_alpha
                constraint_state.ls_par_phase[i_b] = _PAR_PHASE_DONE
            elif constraint_state.ls_it[i_b] >= rigid_global_info.ls_iterations[None]:
                # Original lines 1743-1746: max iterations reached
                constraint_state.ls_result[i_b] = 3
                # Pick lowest-cost candidate as result
                best_a = c0_a
                best_c = c0_c
                if c1_c < best_c:
                    best_a = c1_a
                    best_c = c1_c
                if c2_c < best_c:
                    best_a = c2_a
                    best_c = c2_c
                constraint_state.ls_par_res_alpha[i_b] = best_a
                constraint_state.ls_par_phase[i_b] = _PAR_PHASE_DONE
            else:
                # [CHANGED from original] Original lines 1729-1737 process 1 Newton step:
                #   p2 = p1; p1 = func_ls_point_fn(Newton)
                # Parallel version processes 3 candidates at once. For each candidate:
                #   - If d0*direction <= -gtol: still on bracketing side → advance p1
                #     toward zero derivative (same-sign, closest-to-zero — mirrors
                #     update_bracket logic at line 1888)
                #   - Otherwise: crossed bracket boundary → transition to refinement
                new_p1_alpha = p1_alpha
                new_p1_cost = p1_cost
                new_p1_d0 = p1_d0
                new_p1_d1 = p1_d1
                bracket_found = gs.ti_bool(False)

                # Check candidate 0
                if c0_d0 * direction <= -gtol:
                    # Still on the bracketing side — advance p1 if closer to zero
                    if (p1_d0 < 0 and c0_d0 > new_p1_d0) or (p1_d0 > 0 and c0_d0 < new_p1_d0):
                        new_p1_alpha = c0_a
                        new_p1_cost = c0_c
                        new_p1_d0 = c0_d0
                        new_p1_d1 = c0_d1
                elif not bracket_found or c0_c < new_p1_cost:
                    # Crossed bracket — candidate for p1 at bracket crossing
                    bracket_found = True
                    new_p1_alpha = c0_a
                    new_p1_cost = c0_c
                    new_p1_d0 = c0_d0
                    new_p1_d1 = c0_d1

                # Check candidate 1
                if c1_d0 * direction <= -gtol:
                    if not bracket_found:
                        if (p1_d0 < 0 and c1_d0 > new_p1_d0) or (p1_d0 > 0 and c1_d0 < new_p1_d0):
                            new_p1_alpha = c1_a
                            new_p1_cost = c1_c
                            new_p1_d0 = c1_d0
                            new_p1_d1 = c1_d1
                elif not bracket_found or c1_c < new_p1_cost:
                    bracket_found = True
                    new_p1_alpha = c1_a
                    new_p1_cost = c1_c
                    new_p1_d0 = c1_d0
                    new_p1_d1 = c1_d1

                # Check candidate 2
                if c2_d0 * direction <= -gtol:
                    if not bracket_found:
                        if (p1_d0 < 0 and c2_d0 > new_p1_d0) or (p1_d0 > 0 and c2_d0 < new_p1_d0):
                            new_p1_alpha = c2_a
                            new_p1_cost = c2_c
                            new_p1_d0 = c2_d0
                            new_p1_d1 = c2_d1
                elif not bracket_found or c2_c < new_p1_cost:
                    bracket_found = True
                    new_p1_alpha = c2_a
                    new_p1_cost = c2_c
                    new_p1_d0 = c2_d0
                    new_p1_d1 = c2_d1

                # Original lines 1732-1733: p2 = old p1, p2update = 1
                constraint_state.ls_par_p2[0, i_b] = p1_alpha
                constraint_state.ls_par_p2[1, i_b] = p1_cost
                constraint_state.ls_par_p2[2, i_b] = p1_d0
                constraint_state.ls_par_p2[3, i_b] = p1_d1
                constraint_state.ls_par_p2update[i_b] = 1

                # Update p1 to best candidate
                constraint_state.ls_par_p1[0, i_b] = new_p1_alpha
                constraint_state.ls_par_p1[1, i_b] = new_p1_cost
                constraint_state.ls_par_p1[2, i_b] = new_p1_d0
                constraint_state.ls_par_p1[3, i_b] = new_p1_d1

                if bracket_found:
                    # Original lines 1748-1763: transition to refinement
                    p2update = constraint_state.ls_par_p2update[i_b]
                    if not p2update:
                        # Original line 1749: no valid p2 → give up
                        constraint_state.ls_result[i_b] = 6
                        constraint_state.ls_par_res_alpha[i_b] = new_p1_alpha
                        constraint_state.ls_par_phase[i_b] = _PAR_PHASE_DONE
                    else:
                        # Original lines 1753-1763: enter refinement with 3 candidates
                        # [0]=Newton(p1) (line 1761), [1]=p2_next=p1 (line 1754), [2]=pmid (line 1766)
                        p2_alpha = constraint_state.ls_par_p2[0, i_b]
                        p1_next_alpha = new_p1_alpha - new_p1_d0 / new_p1_d1
                        mid_alpha = (new_p1_alpha + p2_alpha) * 0.5

                        constraint_state.ls_par_cand_alpha[0, i_b] = p1_next_alpha
                        constraint_state.ls_par_cand_alpha[1, i_b] = new_p1_alpha  # p2_next = p1
                        constraint_state.ls_par_cand_alpha[2, i_b] = mid_alpha
                        constraint_state.ls_par_phase[i_b] = _PAR_PHASE_REFINEMENT
                else:
                    # Original line 1735 [CHANGED]: generate next 3 bracketing candidates
                    step = -new_p1_d0 / new_p1_d1
                    constraint_state.ls_par_cand_alpha[0, i_b] = new_p1_alpha + step
                    constraint_state.ls_par_cand_alpha[1, i_b] = new_p1_alpha + 0.5 * step
                    constraint_state.ls_par_cand_alpha[2, i_b] = new_p1_alpha + 2.0 * step

        elif phase == _PAR_PHASE_REFINEMENT:
            # --- Paralleled version of refinement while-loop (original lines 1765-1848) ---
            gtol = constraint_state.gtol[i_b]

            # ls_it += 3 (3 parallel evals; original line 1599 increments by 1 per call)
            constraint_state.ls_it[i_b] = constraint_state.ls_it[i_b] + 3

            # Read p1, p2
            p1_alpha = constraint_state.ls_par_p1[0, i_b]
            p1_cost = constraint_state.ls_par_p1[1, i_b]
            p1_d0 = constraint_state.ls_par_p1[2, i_b]
            p1_d1 = constraint_state.ls_par_p1[3, i_b]

            p2_alpha = constraint_state.ls_par_p2[0, i_b]
            p2_cost = constraint_state.ls_par_p2[1, i_b]
            p2_d0 = constraint_state.ls_par_p2[2, i_b]
            p2_d1 = constraint_state.ls_par_p2[3, i_b]

            p0_cost = constraint_state.ls_par_p0_cost[i_b]

            # Read all 3 evaluated candidates: [0]=p1_next, [1]=p2_next, [2]=pmid
            # (original lines 1770-1790: stored in constraint_state.candidates[12])
            c0_a = constraint_state.ls_par_cand_alpha[0, i_b]
            c0_c = constraint_state.ls_par_cand_cost[0, i_b]
            c0_d0 = constraint_state.ls_par_cand_deriv0[0, i_b]
            c0_d1 = constraint_state.ls_par_cand_deriv1[0, i_b]

            c1_a = constraint_state.ls_par_cand_alpha[1, i_b]
            c1_c = constraint_state.ls_par_cand_cost[1, i_b]
            c1_d0 = constraint_state.ls_par_cand_deriv0[1, i_b]
            c1_d1 = constraint_state.ls_par_cand_deriv1[1, i_b]

            c2_a = constraint_state.ls_par_cand_alpha[2, i_b]
            c2_c = constraint_state.ls_par_cand_cost[2, i_b]
            c2_d0 = constraint_state.ls_par_cand_deriv0[2, i_b]
            c2_d1 = constraint_state.ls_par_cand_deriv1[2, i_b]

            # Original lines 1792-1801: check convergence on all 3, pick lowest cost
            best_i = -1
            best_cost = gs.ti_float(0.0)
            best_alpha_val = gs.ti_float(0.0)

            if ti.abs(c0_d0) < gtol and (best_i < 0 or c0_c < best_cost):
                best_i = 0
                best_cost = c0_c
                best_alpha_val = c0_a
            if ti.abs(c1_d0) < gtol and (best_i < 0 or c1_c < best_cost):
                best_i = 1
                best_cost = c1_c
                best_alpha_val = c1_a
            if ti.abs(c2_d0) < gtol and (best_i < 0 or c2_c < best_cost):
                best_i = 2
                best_cost = c2_c
                best_alpha_val = c2_a

            if best_i >= 0:
                # Original line 1801: converged candidate found
                constraint_state.ls_par_res_alpha[i_b] = best_alpha_val
                constraint_state.ls_par_phase[i_b] = _PAR_PHASE_DONE
            else:
                # Original lines 1804-1816 → update_bracket for p1
                # Inlined from update_bracket (line 1888): iterate over 3 candidates,
                # find one whose derivative has same sign as p_d0 but closer to zero.
                # [CHANGED] Original calls func_ls_point_fn(Newton) immediately (line 1930);
                # parallel version defers eval — writes alpha to ls_par_cand_alpha.
                b1 = 0
                # candidate 0
                if p1_d0 < 0 and c0_d0 < 0 and p1_d0 < c0_d0:
                    p1_alpha, p1_cost, p1_d0, p1_d1 = c0_a, c0_c, c0_d0, c0_d1
                    b1 = 1
                elif p1_d0 > 0 and c0_d0 > 0 and p1_d0 > c0_d0:
                    p1_alpha, p1_cost, p1_d0, p1_d1 = c0_a, c0_c, c0_d0, c0_d1
                    b1 = 2
                # candidate 1
                if p1_d0 < 0 and c1_d0 < 0 and p1_d0 < c1_d0:
                    p1_alpha, p1_cost, p1_d0, p1_d1 = c1_a, c1_c, c1_d0, c1_d1
                    b1 = 1
                elif p1_d0 > 0 and c1_d0 > 0 and p1_d0 > c1_d0:
                    p1_alpha, p1_cost, p1_d0, p1_d1 = c1_a, c1_c, c1_d0, c1_d1
                    b1 = 2
                # candidate 2
                if p1_d0 < 0 and c2_d0 < 0 and p1_d0 < c2_d0:
                    p1_alpha, p1_cost, p1_d0, p1_d1 = c2_a, c2_c, c2_d0, c2_d1
                    b1 = 1
                elif p1_d0 > 0 and c2_d0 > 0 and p1_d0 > c2_d0:
                    p1_alpha, p1_cost, p1_d0, p1_d1 = c2_a, c2_c, c2_d0, c2_d1
                    b1 = 2

                # Original line 1930: compute Newton step for updated p1
                # (deferred eval — will be candidate 0 next round)
                p1_next_alpha = p1_alpha
                if b1 > 0:
                    p1_next_alpha = p1_alpha - p1_d0 / p1_d1

                # Original lines 1817-1829 → update_bracket for p2 (same logic as p1 above)
                # Uses same 3 candidates (not the updated p1 values)
                b2 = 0
                if p2_d0 < 0 and c0_d0 < 0 and p2_d0 < c0_d0:
                    p2_alpha, p2_cost, p2_d0, p2_d1 = c0_a, c0_c, c0_d0, c0_d1
                    b2 = 1
                elif p2_d0 > 0 and c0_d0 > 0 and p2_d0 > c0_d0:
                    p2_alpha, p2_cost, p2_d0, p2_d1 = c0_a, c0_c, c0_d0, c0_d1
                    b2 = 2
                if p2_d0 < 0 and c1_d0 < 0 and p2_d0 < c1_d0:
                    p2_alpha, p2_cost, p2_d0, p2_d1 = c1_a, c1_c, c1_d0, c1_d1
                    b2 = 1
                elif p2_d0 > 0 and c1_d0 > 0 and p2_d0 > c1_d0:
                    p2_alpha, p2_cost, p2_d0, p2_d1 = c1_a, c1_c, c1_d0, c1_d1
                    b2 = 2
                if p2_d0 < 0 and c2_d0 < 0 and p2_d0 < c2_d0:
                    p2_alpha, p2_cost, p2_d0, p2_d1 = c2_a, c2_c, c2_d0, c2_d1
                    b2 = 1
                elif p2_d0 > 0 and c2_d0 > 0 and p2_d0 > c2_d0:
                    p2_alpha, p2_cost, p2_d0, p2_d1 = c2_a, c2_c, c2_d0, c2_d1
                    b2 = 2

                p2_next_alpha = p2_alpha
                if b2 > 0:
                    p2_next_alpha = p2_alpha - p2_d0 / p2_d1

                # Original lines 1831-1837: if both brackets failed, use midpoint
                if b1 == 0 and b2 == 0:
                    # pmid is candidate 2
                    if c2_c < p0_cost:
                        constraint_state.ls_result[i_b] = 0
                    else:
                        constraint_state.ls_result[i_b] = 7
                    constraint_state.ls_par_res_alpha[i_b] = c2_a
                    constraint_state.ls_par_phase[i_b] = _PAR_PHASE_DONE
                elif constraint_state.ls_it[i_b] >= rigid_global_info.ls_iterations[None]:
                    # Original lines 1839-1848: max iterations — pick best of p1/p2
                    if p1_cost <= p2_cost and p1_cost < p0_cost:
                        constraint_state.ls_result[i_b] = 4
                        constraint_state.ls_par_res_alpha[i_b] = p1_alpha
                    elif p2_cost <= p1_cost and p2_cost < p1_cost:
                        constraint_state.ls_result[i_b] = 4
                        constraint_state.ls_par_res_alpha[i_b] = p2_alpha
                    else:
                        constraint_state.ls_result[i_b] = 5
                        constraint_state.ls_par_res_alpha[i_b] = 0.0
                    constraint_state.ls_par_phase[i_b] = _PAR_PHASE_DONE
                else:
                    # Continue refinement: store updated p1/p2, generate next 3 candidates
                    constraint_state.ls_par_p1[0, i_b] = p1_alpha
                    constraint_state.ls_par_p1[1, i_b] = p1_cost
                    constraint_state.ls_par_p1[2, i_b] = p1_d0
                    constraint_state.ls_par_p1[3, i_b] = p1_d1

                    constraint_state.ls_par_p2[0, i_b] = p2_alpha
                    constraint_state.ls_par_p2[1, i_b] = p2_cost
                    constraint_state.ls_par_p2[2, i_b] = p2_d0
                    constraint_state.ls_par_p2[3, i_b] = p2_d1

                    # Generate next 3 candidates for parallel eval:
                    # [0]=Newton(p1) (deferred from line 1930)
                    # [1]=Newton(p2) (deferred from line 1930)
                    # [2]=midpoint   (original line 1766)
                    constraint_state.ls_par_cand_alpha[0, i_b] = p1_next_alpha
                    constraint_state.ls_par_cand_alpha[1, i_b] = p2_next_alpha
                    constraint_state.ls_par_cand_alpha[2, i_b] = (p1_alpha + p2_alpha) * 0.5


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_par_ls_apply(
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
):
    """
    Apply the result alpha to qacc, Ma, Jaref.

    Maps to constraint_solver.py func_linesearch_top_level (line 1852):
      - Line  1870-1871 → if |alpha| < EPS: improved = False (no progress)
      - Lines 1876-1880 → qacc += search * alpha, Ma += mv * alpha
      - Lines 1883-1884 → Jaref += jv * alpha

    Only difference: original reads alpha as return value of func_linesearch;
    parallel version reads from ls_par_res_alpha[i_b].
    """
    _B = constraint_state.grad.shape[1]
    ti.loop_config(serialize=True)
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            alpha = constraint_state.ls_par_res_alpha[i_b]
            n_dofs = constraint_state.qacc.shape[0]
            if ti.abs(alpha) < rigid_global_info.EPS[None]:
                constraint_state.improved[i_b] = False
            else:
                for i_d in range(n_dofs):
                    constraint_state.qacc[i_d, i_b] = (
                        constraint_state.qacc[i_d, i_b] + constraint_state.search[i_d, i_b] * alpha
                    )
                    constraint_state.Ma[i_d, i_b] = (
                        constraint_state.Ma[i_d, i_b] + constraint_state.mv[i_d, i_b] * alpha
                    )
                for i_c in range(constraint_state.n_constraints[i_b]):
                    constraint_state.Jaref[i_c, i_b] = (
                        constraint_state.Jaref[i_c, i_b] + constraint_state.jv[i_c, i_b] * alpha
                    )


_par_probe_timer = None
_par_probe_call_count = 0


def func_solve_parallel_probe_macrokernels(
    entities_info,
    dofs_state,
    constraint_state,
    rigid_global_info,
    static_rigid_sim_config,
):
    """
    Uses parallel-probe linesearch based on the original func_linesearch algorithm,
    split into init/eval/select/apply kernels. Both bracketing and refinement phases
    evaluate 3 candidates in parallel across (batch, probe) pairs via ti.ndrange.
    """
    global _par_probe_timer, _par_probe_call_count
    import os

    do_profile = os.environ.get("GS_PROFILE_PARALLEL_PROBE", "0") == "1"
    if do_profile:
        from genesis.utils.tools import create_timer

        _par_probe_call_count += 1
        # Only print every 100th call to avoid flooding
        skip = (_par_probe_call_count % 100) != 0
        _par_probe_timer = create_timer("par_probe", new=True, ti_sync=True, skip_first_call=skip)

    iterations = rigid_global_info.iterations[None]
    max_ls_rounds = (rigid_global_info.ls_iterations[None] + N_PAR_PROBES - 1) // N_PAR_PROBES + 2
    # TODO: now set to 1 for debugging
    # max_ls_rounds = 1
    for _it in range(iterations):
        _kernel_par_ls_init(
            entities_info,
            dofs_state,
            constraint_state,
            rigid_global_info,
            static_rigid_sim_config,
        )
        if do_profile and _par_probe_timer:
            _par_probe_timer._stamp("par_ls_init")
        for _ls_round in range(max_ls_rounds):
            _kernel_par_ls_eval(
                constraint_state,
                rigid_global_info,
            )
            if do_profile and _par_probe_timer:
                _par_probe_timer._stamp("par_ls_eval")
            _kernel_par_ls_select(
                constraint_state,
                rigid_global_info,
            )
            if do_profile and _par_probe_timer:
                _par_probe_timer._stamp("par_ls_select")
        _kernel_par_ls_apply(
            constraint_state,
            rigid_global_info,
        )
        if do_profile and _par_probe_timer:
            _par_probe_timer._stamp("par_ls_apply")
        if static_rigid_sim_config.solver_type == gs.constraint_solver.CG:
            _kernel_cg_only_save_prev_grad(
                constraint_state,
                static_rigid_sim_config,
            )
            if do_profile and _par_probe_timer:
                _par_probe_timer._stamp("cg_save_prev_grad")
        _kernel_update_constraint(
            entities_info,
            dofs_state,
            constraint_state,
            rigid_global_info,
            static_rigid_sim_config,
        )
        if do_profile and _par_probe_timer:
            _par_probe_timer._stamp("update_constraint")
        if static_rigid_sim_config.solver_type == gs.constraint_solver.Newton:
            _kernel_newton_only_nt_hessian_incremental(
                entities_info,
                constraint_state,
                rigid_global_info,
                static_rigid_sim_config,
            )
            if do_profile and _par_probe_timer:
                _par_probe_timer._stamp("nt_hessian")
        _kernel_update_gradient(
            entities_info,
            dofs_state,
            constraint_state,
            rigid_global_info,
            static_rigid_sim_config,
        )
        if do_profile and _par_probe_timer:
            _par_probe_timer._stamp("update_gradient")
        _kernel_update_search_direction(
            constraint_state,
            rigid_global_info,
            static_rigid_sim_config,
        )
        if do_profile and _par_probe_timer:
            _par_probe_timer._stamp("update_search_dir")
