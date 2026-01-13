"""
Decomposed Constraint Solver with Multi-dimensional Parallelization.

This module contains decomposed kernels for the constraint solver that parallelize
over (batch, dofs) and (batch, constraints) dimensions for better GPU utilization.
"""

import gstaichi as ti

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.engine.solvers.rigid.rigid_solver_decomp as rigid_solver


# =============================================================================
# Decomposed Solver Kernels - Multi-dimensional parallelization
# =============================================================================


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_compute_jv(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
):
    """Compute jv = J @ search. Parallelizes over (batch, constraints)."""
    max_constraints = constraint_state.jac.shape[0]
    n_dofs = constraint_state.jac.shape[1]
    _B = constraint_state.jac.shape[2]

    for i_c, i_b in ti.ndrange(max_constraints, _B):
        if i_c < constraint_state.n_constraints[i_b] and constraint_state.improved[i_b]:
            jv = gs.ti_float(0.0)
            if ti.static(static_rigid_sim_config.sparse_solve):
                for i_d_ in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                    i_d = constraint_state.jac_relevant_dofs[i_c, i_d_, i_b]
                    jv = jv + constraint_state.jac[i_c, i_d, i_b] * constraint_state.search[i_d, i_b]
            else:
                for i_d in range(n_dofs):
                    jv = jv + constraint_state.jac[i_c, i_d, i_b] * constraint_state.search[i_d, i_b]
            constraint_state.jv[i_c, i_b] = jv


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_compute_mv(
    constraint_state: array_class.ConstraintState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    """Compute mv = M @ search. Parallelizes over (batch, entities)."""
    n_entities = entities_info.dof_start.shape[0]
    _B = constraint_state.search.shape[1]

    for i_e, i_b in ti.ndrange(n_entities, _B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            for i_d1 in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                mv = gs.ti_float(0.0)
                for i_d2 in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                    mv = mv + rigid_global_info.mass_mat[i_d1, i_d2, i_b] * constraint_state.search[i_d2, i_b]
                constraint_state.mv[i_d1, i_b] = mv


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_update_constraint_efc(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
):
    """Update efc_force and active flags. Parallelizes over (batch, constraints)."""
    max_constraints = constraint_state.active.shape[0]
    _B = constraint_state.active.shape[1]

    for i_c, i_b in ti.ndrange(max_constraints, _B):
        if i_c < constraint_state.n_constraints[i_b] and constraint_state.improved[i_b]:
            ne = constraint_state.n_constraints_equality[i_b]
            nef = ne + constraint_state.n_constraints_frictionloss[i_b]

            if ti.static(static_rigid_sim_config.solver_type == gs.constraint_solver.Newton):
                constraint_state.prev_active[i_c, i_b] = constraint_state.active[i_c, i_b]

            constraint_state.active[i_c, i_b] = True
            Jaref = constraint_state.Jaref[i_c, i_b]
            floss_force = gs.ti_float(0.0)

            if ne <= i_c and i_c < nef:  # Friction constraints
                f = constraint_state.efc_frictionloss[i_c, i_b]
                r = constraint_state.diag[i_c, i_b]
                rf = r * f
                linear_neg = Jaref <= -rf
                linear_pos = Jaref >= rf
                constraint_state.active[i_c, i_b] = not (linear_neg or linear_pos)
                floss_force = linear_neg * f + linear_pos * (-f)
            elif nef <= i_c:  # Contact constraints
                constraint_state.active[i_c, i_b] = Jaref < 0

            constraint_state.efc_force[i_c, i_b] = floss_force + (
                -Jaref * constraint_state.efc_D[i_c, i_b] * constraint_state.active[i_c, i_b]
            )


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_update_qfrc_constraint(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
):
    """Compute qfrc_constraint = J^T @ efc_force. Parallelizes over (batch, dofs)."""
    n_dofs = constraint_state.qfrc_constraint.shape[0]
    _B = constraint_state.qfrc_constraint.shape[1]

    if ti.static(static_rigid_sim_config.sparse_solve):
        # Sparse path: zero out first, then accumulate from constraints
        for i_d, i_b in ti.ndrange(n_dofs, _B):
            if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
                constraint_state.qfrc_constraint[i_d, i_b] = gs.ti_float(0.0)
        for i_c, i_b in ti.ndrange(constraint_state.jac.shape[0], _B):
            if i_c < constraint_state.n_constraints[i_b] and constraint_state.improved[i_b]:
                for i_d_ in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                    i_d = constraint_state.jac_relevant_dofs[i_c, i_d_, i_b]
                    constraint_state.qfrc_constraint[i_d, i_b] = (
                        constraint_state.qfrc_constraint[i_d, i_b]
                        + constraint_state.jac[i_c, i_d, i_b] * constraint_state.efc_force[i_c, i_b]
                    )
    else:
        # Dense path: iterate over dofs, sum over constraints
        for i_d, i_b in ti.ndrange(n_dofs, _B):
            if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
                qfrc = gs.ti_float(0.0)
                for i_c in range(constraint_state.n_constraints[i_b]):
                    qfrc = qfrc + constraint_state.jac[i_c, i_d, i_b] * constraint_state.efc_force[i_c, i_b]
                constraint_state.qfrc_constraint[i_d, i_b] = qfrc


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_compute_cost(
    constraint_state: array_class.ConstraintState,
    dofs_state: array_class.DofsState,
):
    """Compute total cost. Two-phase: constraints then DOFs with atomic adds."""
    max_constraints = constraint_state.active.shape[0]
    n_dofs = constraint_state.grad.shape[0]
    _B = constraint_state.cost.shape[0]

    # Initialize cost and gauss
    for i_b in range(_B):
        if constraint_state.improved[i_b]:
            constraint_state.prev_cost[i_b] = constraint_state.cost[i_b]
            constraint_state.cost[i_b] = gs.ti_float(0.0)
            constraint_state.gauss[i_b] = gs.ti_float(0.0)

    # Constraint cost contributions
    for i_c, i_b in ti.ndrange(max_constraints, _B):
        if i_c < constraint_state.n_constraints[i_b] and constraint_state.improved[i_b]:
            ne = constraint_state.n_constraints_equality[i_b]
            nef = ne + constraint_state.n_constraints_frictionloss[i_b]
            Jaref = constraint_state.Jaref[i_c, i_b]

            cost_c = gs.ti_float(0.0)
            if ne <= i_c and i_c < nef:  # Friction
                f = constraint_state.efc_frictionloss[i_c, i_b]
                r = constraint_state.diag[i_c, i_b]
                rf = r * f
                linear_neg = Jaref <= -rf
                linear_pos = Jaref >= rf
                cost_c = linear_neg * f * (-0.5 * rf - Jaref) + linear_pos * f * (-0.5 * rf + Jaref)

            cost_c = cost_c + 0.5 * Jaref * Jaref * constraint_state.efc_D[i_c, i_b] * constraint_state.active[i_c, i_b]
            ti.atomic_add(constraint_state.cost[i_b], cost_c)

    # DOF contributions (gauss term)
    for i_d, i_b in ti.ndrange(n_dofs, _B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            v = (
                0.5
                * (constraint_state.Ma[i_d, i_b] - dofs_state.force[i_d, i_b])
                * (constraint_state.qacc[i_d, i_b] - dofs_state.acc_smooth[i_d, i_b])
            )
            ti.atomic_add(constraint_state.gauss[i_b], v)
            ti.atomic_add(constraint_state.cost[i_b], v)


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_compute_grad(
    constraint_state: array_class.ConstraintState,
    dofs_state: array_class.DofsState,
):
    """Compute grad = Ma - force - qfrc_constraint. Parallelizes over (batch, dofs)."""
    n_dofs = constraint_state.grad.shape[0]
    _B = constraint_state.grad.shape[1]

    for i_d, i_b in ti.ndrange(n_dofs, _B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            constraint_state.grad[i_d, i_b] = (
                constraint_state.Ma[i_d, i_b] - dofs_state.force[i_d, i_b] - constraint_state.qfrc_constraint[i_d, i_b]
            )


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_update_search_cg(
    constraint_state: array_class.ConstraintState,
):
    """Update CG search direction. Parallelizes over (batch, dofs)."""
    n_dofs = constraint_state.search.shape[0]
    _B = constraint_state.search.shape[1]

    for i_d, i_b in ti.ndrange(n_dofs, _B):
        if constraint_state.improved[i_b]:
            beta = constraint_state.cg_beta[i_b]
            constraint_state.search[i_d, i_b] = (
                -constraint_state.Mgrad[i_d, i_b] + beta * constraint_state.search[i_d, i_b]
            )


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_cg_beta(
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
):
    """Compute CG beta coefficient (Polak-Ribiere). Per-batch with DOF reduction."""
    n_dofs = constraint_state.grad.shape[0]
    _B = constraint_state.cg_beta.shape[0]
    EPS = rigid_global_info.EPS[None]

    for i_b in range(_B):
        if constraint_state.improved[i_b]:
            numerator = gs.ti_float(0.0)
            denominator = gs.ti_float(0.0)
            for i_d in range(n_dofs):
                numerator = numerator + constraint_state.grad[i_d, i_b] * (
                    constraint_state.Mgrad[i_d, i_b] - constraint_state.cg_prev_Mgrad[i_d, i_b]
                )
                denominator = (
                    denominator + constraint_state.cg_prev_Mgrad[i_d, i_b] * constraint_state.cg_prev_grad[i_d, i_b]
                )

            constraint_state.cg_pg_dot_pMg[i_b] = denominator
            constraint_state.cg_beta[i_b] = ti.max(numerator / ti.max(EPS, denominator), 0.0)


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_check_convergence(
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
):
    """Check convergence based on gradient norm and cost improvement."""
    n_dofs = constraint_state.grad.shape[0]
    _B = constraint_state.improved.shape[0]

    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            tol_scaled = rigid_global_info.meaninertia[i_b] * ti.max(1, n_dofs) * rigid_global_info.tolerance[None]
            improvement = constraint_state.prev_cost[i_b] - constraint_state.cost[i_b]

            grad_norm = gs.ti_float(0.0)
            for i_d in range(n_dofs):
                grad_norm = grad_norm + constraint_state.grad[i_d, i_b] * constraint_state.grad[i_d, i_b]
            grad_norm = ti.sqrt(grad_norm)

            if grad_norm < tol_scaled or improvement < tol_scaled:
                constraint_state.improved[i_b] = False


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_update_qacc_Ma_Jaref(
    constraint_state: array_class.ConstraintState,
):
    """Update qacc, Ma with alpha. Parallelizes over (batch, dofs) and (batch, constraints)."""
    n_dofs = constraint_state.qacc.shape[0]
    max_constraints = constraint_state.Jaref.shape[0]
    _B = constraint_state.qacc.shape[1]

    # Update qacc and Ma (DOF parallel)
    for i_d, i_b in ti.ndrange(n_dofs, _B):
        if constraint_state.improved[i_b]:
            alpha = constraint_state.alpha[i_b]
            constraint_state.qacc[i_d, i_b] = (
                constraint_state.qacc[i_d, i_b] + constraint_state.search[i_d, i_b] * alpha
            )
            constraint_state.Ma[i_d, i_b] = constraint_state.Ma[i_d, i_b] + constraint_state.mv[i_d, i_b] * alpha

    # Update Jaref (constraint parallel)
    for i_c, i_b in ti.ndrange(max_constraints, _B):
        if i_c < constraint_state.n_constraints[i_b] and constraint_state.improved[i_b]:
            alpha = constraint_state.alpha[i_b]
            constraint_state.Jaref[i_c, i_b] = constraint_state.Jaref[i_c, i_b] + constraint_state.jv[i_c, i_b] * alpha


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_save_prev_grad_Mgrad(
    constraint_state: array_class.ConstraintState,
):
    """Save grad and Mgrad for CG beta computation. Parallelizes over (batch, dofs).

    Note: This kernel is CG-specific. Should only be called when solver_type == CG.
    """
    n_dofs = constraint_state.grad.shape[0]
    _B = constraint_state.grad.shape[1]

    for i_d, i_b in ti.ndrange(n_dofs, _B):
        if constraint_state.improved[i_b]:
            constraint_state.cg_prev_grad[i_d, i_b] = constraint_state.grad[i_d, i_b]
            constraint_state.cg_prev_Mgrad[i_d, i_b] = constraint_state.Mgrad[i_d, i_b]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_update_search_newton(
    constraint_state: array_class.ConstraintState,
):
    """Update search direction for Newton solver: search = -Mgrad. Parallelizes over (batch, dofs)."""
    n_dofs = constraint_state.search.shape[0]
    _B = constraint_state.search.shape[1]

    for i_d, i_b in ti.ndrange(n_dofs, _B):
        if constraint_state.improved[i_b]:
            constraint_state.search[i_d, i_b] = -constraint_state.Mgrad[i_d, i_b]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_init_improved(
    constraint_state: array_class.ConstraintState,
):
    """Initialize improved flag for batches with constraints."""
    _B = constraint_state.improved.shape[0]

    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0:
            constraint_state.improved[i_b] = True
        else:
            constraint_state.improved[i_b] = False


# =============================================================================
# Linesearch wrapper and Mgrad computation
# These kernels require access to functions from constraint_solver_decomp
# =============================================================================


def create_linesearch_wrapper(func_linesearch):
    """Create linesearch wrapper kernel with access to func_linesearch."""

    @ti.kernel(fastcache=gs.use_fastcache)
    def kernel_linesearch_wrapper(
        entities_info: array_class.EntitiesInfo,
        dofs_state: array_class.DofsState,
        constraint_state: array_class.ConstraintState,
        rigid_global_info: array_class.RigidGlobalInfo,
        static_rigid_sim_config: ti.template(),
    ):
        """Run linesearch for each batch and store alpha. Per-batch operation."""
        _B = constraint_state.alpha.shape[0]
        EPS = rigid_global_info.EPS[None]

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(_B):
            if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
                alpha = func_linesearch(
                    i_b,
                    entities_info=entities_info,
                    dofs_state=dofs_state,
                    rigid_global_info=rigid_global_info,
                    constraint_state=constraint_state,
                    static_rigid_sim_config=static_rigid_sim_config,
                )
                constraint_state.alpha[i_b] = alpha
                if ti.abs(alpha) < EPS:
                    constraint_state.improved[i_b] = False
            else:
                constraint_state.alpha[i_b] = gs.ti_float(0.0)

    return kernel_linesearch_wrapper


def create_compute_Mgrad_kernel():
    """Create Mgrad computation kernel."""

    @ti.kernel(fastcache=gs.use_fastcache)
    def kernel_compute_Mgrad(
        entities_info: array_class.EntitiesInfo,
        constraint_state: array_class.ConstraintState,
        rigid_global_info: array_class.RigidGlobalInfo,
        static_rigid_sim_config: ti.template(),
    ):
        """Compute Mgrad = M^-1 @ grad for CG solver. Per-batch due to mass solve."""
        _B = constraint_state.grad.shape[1]

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(_B):
            if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
                rigid_solver.func_solve_mass_batch(
                    i_b,
                    constraint_state.grad,
                    constraint_state.Mgrad,
                    array_class.PLACEHOLDER,
                    entities_info=entities_info,
                    rigid_global_info=rigid_global_info,
                    static_rigid_sim_config=static_rigid_sim_config,
                    is_backward=False,
                )

    return kernel_compute_Mgrad


def create_nt_hessian_incremental_kernel(func_nt_hessian_incremental):
    """Create Newton Hessian incremental update kernel."""

    @ti.kernel(fastcache=gs.use_fastcache)
    def kernel_nt_hessian_incremental(
        entities_info: array_class.EntitiesInfo,
        constraint_state: array_class.ConstraintState,
        rigid_global_info: array_class.RigidGlobalInfo,
        static_rigid_sim_config: ti.template(),
    ):
        """Incremental Newton Hessian update. Per-batch operation."""
        _B = constraint_state.grad.shape[1]

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(_B):
            if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
                func_nt_hessian_incremental(
                    i_b,
                    entities_info=entities_info,
                    constraint_state=constraint_state,
                    rigid_global_info=rigid_global_info,
                    static_rigid_sim_config=static_rigid_sim_config,
                )

    return kernel_nt_hessian_incremental


def create_nt_chol_solve_kernel(func_nt_chol_solve):
    """Create Newton Cholesky solve kernel."""

    @ti.kernel(fastcache=gs.use_fastcache)
    def kernel_nt_chol_solve(
        constraint_state: array_class.ConstraintState,
        static_rigid_sim_config: ti.template(),
    ):
        """Newton Cholesky solve for Mgrad. Per-batch operation."""
        _B = constraint_state.grad.shape[1]

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(_B):
            if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
                func_nt_chol_solve(i_b, constraint_state=constraint_state)

    return kernel_nt_chol_solve


# =============================================================================
# Decomposed Solver Orchestration
# =============================================================================


def func_solve_decomposed(
    entities_info,
    dofs_state,
    constraint_state,
    rigid_global_info,
    static_rigid_sim_config,
    solver_type,
    iterations,
    kernel_linesearch_wrapper,
    kernel_compute_Mgrad,
    kernel_nt_hessian_incremental,
    kernel_nt_chol_solve,
):
    """
    Decomposed constraint solver with multi-dimensional parallelization.

    This function orchestrates the decomposed kernels to perform the same
    computation as func_solve, but with better GPU utilization through
    multi-dimensional parallelization over (batch, dofs) and (batch, constraints).

    Args:
        entities_info: Entity information array
        dofs_state: DOF state array
        constraint_state: Constraint state array
        rigid_global_info: Global rigid body info
        static_rigid_sim_config: Static configuration
        solver_type: gs.constraint_solver.CG or gs.constraint_solver.Newton
        iterations: Maximum number of iterations
        kernel_linesearch_wrapper: Linesearch wrapper kernel
        kernel_compute_Mgrad: Mgrad computation kernel
        kernel_nt_hessian_incremental: Newton Hessian incremental kernel
        kernel_nt_chol_solve: Newton Cholesky solve kernel
    """
    # Initialize improved flag for batches with constraints
    kernel_init_improved(constraint_state)

    for _ in range(iterations):
        # 1. Compute mv = M @ search (needed for linesearch)
        kernel_compute_mv(
            constraint_state,
            entities_info,
            rigid_global_info,
            static_rigid_sim_config,
        )

        # 2. Compute jv = J @ search (needed for linesearch)
        kernel_compute_jv(
            constraint_state,
            static_rigid_sim_config,
        )

        # 3. Linesearch to find optimal step size alpha
        kernel_linesearch_wrapper(
            entities_info,
            dofs_state,
            constraint_state,
            rigid_global_info,
            static_rigid_sim_config,
        )

        # 4. Update qacc, Ma, Jaref with alpha
        kernel_update_qacc_Ma_Jaref(constraint_state)

        # 5. Save prev_grad and prev_Mgrad for CG beta computation
        if solver_type == gs.constraint_solver.CG:
            kernel_save_prev_grad_Mgrad(constraint_state)

        # 6. Update constraint: efc_force and active flags
        kernel_update_constraint_efc(
            constraint_state,
            static_rigid_sim_config,
        )

        # 7. Compute qfrc_constraint = J^T @ efc_force
        kernel_update_qfrc_constraint(
            constraint_state,
            static_rigid_sim_config,
        )

        # 8. Compute cost
        kernel_compute_cost(
            constraint_state,
            dofs_state,
        )

        # 9. Newton Hessian update (Newton only)
        if solver_type == gs.constraint_solver.Newton:
            kernel_nt_hessian_incremental(
                entities_info,
                constraint_state,
                rigid_global_info,
                static_rigid_sim_config,
            )

        # 10. Compute grad = Ma - force - qfrc_constraint
        kernel_compute_grad(
            constraint_state,
            dofs_state,
        )

        # 11. Compute Mgrad = M^-1 @ grad (CG) or Cholesky solve (Newton)
        if solver_type == gs.constraint_solver.CG:
            kernel_compute_Mgrad(
                entities_info,
                constraint_state,
                rigid_global_info,
                static_rigid_sim_config,
            )
        else:  # Newton
            kernel_nt_chol_solve(constraint_state, static_rigid_sim_config)

        # 12. Check convergence
        kernel_check_convergence(
            constraint_state,
            rigid_global_info,
        )

        # 13. Update search direction
        if solver_type == gs.constraint_solver.CG:
            # CG: compute beta and update search = -Mgrad + beta * search
            kernel_cg_beta(
                constraint_state,
                rigid_global_info,
            )
            kernel_update_search_cg(constraint_state)
        else:  # Newton
            # Newton: search = -Mgrad
            kernel_update_search_newton(constraint_state)


# =============================================================================
# Decomposed Init Solver Kernels
# =============================================================================


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_init_qacc(
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
):
    """Initialize qacc from warmstart or acc_smooth. Parallelizes over (dofs, batch)."""
    n_dofs = dofs_state.acc_smooth.shape[0]
    _B = dofs_state.acc_smooth.shape[1]

    for i_d, i_b in ti.ndrange(n_dofs, _B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.is_warmstart[i_b]:
            constraint_state.qacc[i_d, i_b] = constraint_state.qacc_ws[i_d, i_b]
        else:
            constraint_state.qacc[i_d, i_b] = dofs_state.acc_smooth[i_d, i_b]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_init_Ma(
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    """Compute Ma = M @ qacc. Parallelizes over (entities, batch)."""
    n_entities = entities_info.dof_start.shape[0]
    _B = rigid_global_info.mass_mat.shape[2]

    for i_e, i_b in ti.ndrange(n_entities, _B):
        for i_d1 in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
            Ma_ = gs.ti_float(0.0)
            for i_d2 in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                Ma_ = Ma_ + rigid_global_info.mass_mat[i_d1, i_d2, i_b] * constraint_state.qacc[i_d2, i_b]
            constraint_state.Ma[i_d1, i_b] = Ma_


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_init_Jaref(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
):
    """Compute Jaref = J @ qacc - aref. Parallelizes over (constraints, batch)."""
    max_constraints = constraint_state.jac.shape[0]
    n_dofs = constraint_state.jac.shape[1]
    _B = constraint_state.jac.shape[2]

    for i_c, i_b in ti.ndrange(max_constraints, _B):
        if i_c < constraint_state.n_constraints[i_b]:
            Jaref = -constraint_state.aref[i_c, i_b]
            if ti.static(static_rigid_sim_config.sparse_solve):
                for i_d_ in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                    i_d = constraint_state.jac_relevant_dofs[i_c, i_d_, i_b]
                    Jaref = Jaref + constraint_state.jac[i_c, i_d, i_b] * constraint_state.qacc[i_d, i_b]
            else:
                for i_d in range(n_dofs):
                    Jaref = Jaref + constraint_state.jac[i_c, i_d, i_b] * constraint_state.qacc[i_d, i_b]
            constraint_state.Jaref[i_c, i_b] = Jaref


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_init_update_constraint(
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
):
    """
    Update efc_force, active, qfrc_constraint, cost for init_solver.
    Matches the original func_update_constraint exactly.
    Parallelizes over batch (sequential per-batch to match original precision).
    """
    n_dofs = constraint_state.qfrc_constraint.shape[0]
    _B = constraint_state.active.shape[1]

    for i_b in range(_B):
        ne = constraint_state.n_constraints_equality[i_b]
        nef = ne + constraint_state.n_constraints_frictionloss[i_b]

        constraint_state.prev_cost[i_b] = constraint_state.cost[i_b]
        cost_i = gs.ti_float(0.0)
        gauss_i = gs.ti_float(0.0)

        # Update active flags and efc_force
        for i_c in range(constraint_state.n_constraints[i_b]):
            if ti.static(static_rigid_sim_config.solver_type == gs.constraint_solver.Newton):
                constraint_state.prev_active[i_c, i_b] = constraint_state.active[i_c, i_b]
            constraint_state.active[i_c, i_b] = True

            floss_force = gs.ti_float(0.0)
            if ne <= i_c < nef:  # Friction constraints
                f = constraint_state.efc_frictionloss[i_c, i_b]
                r = constraint_state.diag[i_c, i_b]
                rf = r * f
                linear_neg = constraint_state.Jaref[i_c, i_b] <= -rf
                linear_pos = constraint_state.Jaref[i_c, i_b] >= rf
                constraint_state.active[i_c, i_b] = not (linear_neg or linear_pos)
                floss_force = linear_neg * f + linear_pos * (-f)
                floss_cost_local = linear_neg * f * (-0.5 * rf - constraint_state.Jaref[i_c, i_b])
                floss_cost_local = floss_cost_local + linear_pos * f * (-0.5 * rf + constraint_state.Jaref[i_c, i_b])
                cost_i = cost_i + floss_cost_local
            elif i_c >= nef:  # Contact constraints
                constraint_state.active[i_c, i_b] = constraint_state.Jaref[i_c, i_b] < 0

            constraint_state.efc_force[i_c, i_b] = floss_force + (
                -constraint_state.Jaref[i_c, i_b] * constraint_state.efc_D[i_c, i_b] * constraint_state.active[i_c, i_b]
            )

        # Compute qfrc_constraint = J^T @ efc_force
        if ti.static(static_rigid_sim_config.sparse_solve):
            for i_d in range(n_dofs):
                constraint_state.qfrc_constraint[i_d, i_b] = gs.ti_float(0.0)
            for i_c in range(constraint_state.n_constraints[i_b]):
                for i_d_ in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                    i_d = constraint_state.jac_relevant_dofs[i_c, i_d_, i_b]
                    constraint_state.qfrc_constraint[i_d, i_b] = (
                        constraint_state.qfrc_constraint[i_d, i_b]
                        + constraint_state.jac[i_c, i_d, i_b] * constraint_state.efc_force[i_c, i_b]
                    )
        else:
            for i_d in range(n_dofs):
                qfrc_constraint = gs.ti_float(0.0)
                for i_c in range(constraint_state.n_constraints[i_b]):
                    qfrc_constraint = (
                        qfrc_constraint + constraint_state.jac[i_c, i_d, i_b] * constraint_state.efc_force[i_c, i_b]
                    )
                constraint_state.qfrc_constraint[i_d, i_b] = qfrc_constraint

        # (Mx - Mx') * (x - x') - gauss term
        for i_d in range(n_dofs):
            v = (
                0.5
                * (constraint_state.Ma[i_d, i_b] - dofs_state.force[i_d, i_b])
                * (constraint_state.qacc[i_d, i_b] - dofs_state.acc_smooth[i_d, i_b])
            )
            gauss_i = gauss_i + v
            cost_i = cost_i + v

        # D * (Jx - aref) ** 2
        for i_c in range(constraint_state.n_constraints[i_b]):
            cost_i = cost_i + 0.5 * (
                constraint_state.Jaref[i_c, i_b] ** 2
                * constraint_state.efc_D[i_c, i_b]
                * constraint_state.active[i_c, i_b]
            )

        constraint_state.gauss[i_b] = gauss_i
        constraint_state.cost[i_b] = cost_i


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_hessian_JDJ_tiled(
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    """
    Compute Hessian H = M + J^T D J using block tiling over DOF pairs.
    Parallelizes over (batch, block_pairs) - NO ATOMICS needed!
    Each block pair is computed independently by looping over all constraints.
    """
    n_dofs = constraint_state.nt_H.shape[1]
    _B = constraint_state.nt_H.shape[0]
    n_entities = entities_info.dof_start.shape[0]
    EPS = rigid_global_info.EPS[None]

    TILE_SIZE = 8  # Smaller tiles for register efficiency

    n_blocks = (n_dofs + TILE_SIZE - 1) // TILE_SIZE
    n_block_pairs = n_blocks * (n_blocks + 1) // 2  # Lower triangle blocks

    for i_b, block_pair_id in ti.ndrange(_B, n_block_pairs):
        if constraint_state.n_constraints[i_b] == 0:
            continue

        # Decode block_pair_id to (block_i, block_j) in lower triangle
        block_i = ti.cast((ti.sqrt(8.0 * block_pair_id + 1.0) - 1.0) / 2.0, gs.ti_int)
        block_j = block_pair_id - block_i * (block_i + 1) // 2

        offset_i = block_i * TILE_SIZE
        offset_j = block_j * TILE_SIZE

        # Process each element in this block
        for di in range(TILE_SIZE):
            i_d1 = offset_i + di
            if i_d1 >= n_dofs:
                continue
            for dj in range(TILE_SIZE):
                i_d2 = offset_j + dj
                if i_d2 >= n_dofs:
                    continue
                if i_d1 < i_d2:  # Only lower triangle
                    continue

                # Start with mass matrix contribution
                H_val = rigid_global_info.mass_mat[i_d1, i_d2, i_b]

                # Loop over ALL constraints - no atomics needed!
                n_c = constraint_state.n_constraints[i_b]
                if ti.static(static_rigid_sim_config.sparse_solve):
                    # Sparse: check if Jacobian entries are non-zero
                    for i_c in range(n_c):
                        if constraint_state.active[i_c, i_b]:
                            jac_i = constraint_state.jac[i_c, i_d1, i_b]
                            jac_j = constraint_state.jac[i_c, i_d2, i_b]
                            if ti.abs(jac_i) > EPS and ti.abs(jac_j) > EPS:
                                H_val = H_val + jac_i * jac_j * constraint_state.efc_D[i_c, i_b]
                else:
                    # Dense: always accumulate
                    for i_c in range(n_c):
                        if constraint_state.active[i_c, i_b]:
                            jac_i = constraint_state.jac[i_c, i_d1, i_b]
                            jac_j = constraint_state.jac[i_c, i_d2, i_b]
                            H_val = H_val + jac_i * jac_j * constraint_state.efc_D[i_c, i_b]

                constraint_state.nt_H[i_b, i_d1, i_d2] = H_val


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_cholesky_factor(
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
):
    """In-place Cholesky factorization of H. Parallelizes over batch."""
    n_dofs = constraint_state.nt_H.shape[1]
    _B = constraint_state.nt_H.shape[0]
    EPS = rigid_global_info.EPS[None]

    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] == 0:
            continue

        for i_d in range(n_dofs):
            # Diagonal element
            tmp = constraint_state.nt_H[i_b, i_d, i_d]
            for j_d in range(i_d):
                tmp = tmp - constraint_state.nt_H[i_b, i_d, j_d] ** 2
            constraint_state.nt_H[i_b, i_d, i_d] = ti.sqrt(ti.max(tmp, EPS))

            # Off-diagonal elements
            inv_diag = 1.0 / constraint_state.nt_H[i_b, i_d, i_d]
            for j_d in range(i_d + 1, n_dofs):
                dot = gs.ti_float(0.0)
                for k_d in range(i_d):
                    dot = dot + constraint_state.nt_H[i_b, j_d, k_d] * constraint_state.nt_H[i_b, i_d, k_d]
                constraint_state.nt_H[i_b, j_d, i_d] = (constraint_state.nt_H[i_b, j_d, i_d] - dot) * inv_diag


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_init_gradient(
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
):
    """Compute grad = Ma - force - qfrc_constraint. Parallelizes over (dofs, batch)."""
    n_dofs = constraint_state.grad.shape[0]
    _B = constraint_state.grad.shape[1]

    for i_d, i_b in ti.ndrange(n_dofs, _B):
        constraint_state.grad[i_d, i_b] = (
            constraint_state.Ma[i_d, i_b] - dofs_state.force[i_d, i_b] - constraint_state.qfrc_constraint[i_d, i_b]
        )


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_init_Mgrad_cg(
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    """Compute Mgrad = M^-1 @ grad for CG solver via LDL solve. Parallelizes over batch."""
    _B = constraint_state.Mgrad.shape[1]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        rigid_solver.func_solve_mass_batch(
            i_b,
            constraint_state.grad,
            constraint_state.Mgrad,
            array_class.PLACEHOLDER,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
            is_backward=False,
        )


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_init_Mgrad_newton(
    constraint_state: array_class.ConstraintState,
):
    """Compute Mgrad = H^-1 @ grad for Newton solver via Cholesky solve. Parallelizes over batch."""
    n_dofs = constraint_state.nt_H.shape[1]
    _B = constraint_state.nt_H.shape[0]

    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] == 0:
            continue

        # Copy grad to Mgrad for in-place solve
        for i_d in range(n_dofs):
            constraint_state.Mgrad[i_d, i_b] = constraint_state.grad[i_d, i_b]

        # Forward substitution: L y = grad
        for i_d in range(n_dofs):
            tmp = constraint_state.Mgrad[i_d, i_b]
            for j_d in range(i_d):
                tmp = tmp - constraint_state.nt_H[i_b, i_d, j_d] * constraint_state.Mgrad[j_d, i_b]
            constraint_state.Mgrad[i_d, i_b] = tmp / constraint_state.nt_H[i_b, i_d, i_d]

        # Backward substitution: L^T x = y
        for i_d_ in range(n_dofs):
            i_d = n_dofs - 1 - i_d_
            tmp = constraint_state.Mgrad[i_d, i_b]
            for j_d in range(i_d + 1, n_dofs):
                tmp = tmp - constraint_state.nt_H[i_b, j_d, i_d] * constraint_state.Mgrad[j_d, i_b]
            constraint_state.Mgrad[i_d, i_b] = tmp / constraint_state.nt_H[i_b, i_d, i_d]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_init_search(
    constraint_state: array_class.ConstraintState,
):
    """Initialize search = -Mgrad. Parallelizes over (dofs, batch)."""
    n_dofs = constraint_state.search.shape[0]
    _B = constraint_state.search.shape[1]

    for i_d, i_b in ti.ndrange(n_dofs, _B):
        constraint_state.search[i_d, i_b] = -constraint_state.Mgrad[i_d, i_b]


# =============================================================================
# Decomposed Init Solver Orchestration
# =============================================================================


def func_init_solver_decomposed(
    dofs_info: array_class.DofsInfo,
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config,
    solver_type,
):
    """
    Decomposed init_solver with multi-dimensional parallelization.

    This function initializes the constraint solver state before iterations begin.
    It decomposes the monolithic func_init_solver into separate kernels that can
    be profiled and optimized independently.
    """
    # 1. Initialize qacc from warmstart or acc_smooth
    kernel_init_qacc(dofs_state, constraint_state)

    # 2. Compute Ma = M @ qacc
    kernel_init_Ma(
        dofs_info,
        entities_info,
        constraint_state,
        rigid_global_info,
        static_rigid_sim_config,
    )

    # 3. Compute Jaref = J @ qacc - aref
    kernel_init_Jaref(constraint_state, static_rigid_sim_config)

    # 4. Update constraint forces and cost
    kernel_init_update_constraint(dofs_state, constraint_state, static_rigid_sim_config)

    # 5. For Newton solver: compute Hessian and Cholesky factorization
    if solver_type == gs.constraint_solver.Newton:
        kernel_hessian_JDJ_tiled(
            entities_info,
            constraint_state,
            rigid_global_info,
            static_rigid_sim_config,
        )
        kernel_cholesky_factor(constraint_state, rigid_global_info)

    # 6. Compute gradient = Ma - force - qfrc_constraint
    kernel_init_gradient(dofs_state, constraint_state)

    # 7. Compute Mgrad = M^-1 @ grad (CG) or H^-1 @ grad (Newton)
    if solver_type == gs.constraint_solver.CG:
        kernel_init_Mgrad_cg(entities_info, constraint_state, rigid_global_info, static_rigid_sim_config)
    else:  # Newton
        kernel_init_Mgrad_newton(constraint_state)

    # 8. Initialize search = -Mgrad
    kernel_init_search(constraint_state)
