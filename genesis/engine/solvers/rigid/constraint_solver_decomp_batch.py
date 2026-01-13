"""
Decomposed Constraint Solver with Batch-level Parallelization.

This module contains decomposed kernels for the constraint solver that maintain
the original batch-level parallelization pattern for compatibility with the
existing codebase structure.

Unlike constraint_solver_breakdown.py which uses multi-dimensional parallelization
over (batch, dofs) and (batch, constraints), these kernels parallelize only over
batches, matching the original func_solve kernel's parallelization strategy.
"""

import gstaichi as ti

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.engine.solvers.rigid.constraint_solver_decomp as constraint_solver_decomp


# =============================================================================
# Decomposed Solver Kernels - Batch-level parallelization
# =============================================================================


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_linesearch_and_update(
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    """Compute linesearch alpha and update qacc, Ma, Jaref. Parallelizes over batches only."""
    _B = constraint_state.grad.shape[1]
    n_dofs = constraint_state.qacc.shape[0]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0:
            alpha = constraint_solver_decomp.func_linesearch(
                i_b,
                entities_info=entities_info,
                dofs_state=dofs_state,
                rigid_global_info=rigid_global_info,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )
            
            if ti.abs(alpha) < rigid_global_info.EPS[None]:
                constraint_state.improved[i_b] = False
            else:
                # Update qacc and Ma
                for i_d in range(n_dofs):
                    constraint_state.qacc[i_d, i_b] = (
                        constraint_state.qacc[i_d, i_b] + constraint_state.search[i_d, i_b] * alpha
                    )
                    constraint_state.Ma[i_d, i_b] = constraint_state.Ma[i_d, i_b] + constraint_state.mv[i_d, i_b] * alpha

                # Update Jaref
                for i_c in range(constraint_state.n_constraints[i_b]):
                    constraint_state.Jaref[i_c, i_b] = constraint_state.Jaref[i_c, i_b] + constraint_state.jv[i_c, i_b] * alpha


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_save_prev_grad_Mgrad(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
):
    """Save prev_grad and prev_Mgrad for CG. Parallelizes over batches only."""
    _B = constraint_state.grad.shape[1]
    n_dofs = constraint_state.grad.shape[0]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0:
            for i_d in range(n_dofs):
                constraint_state.cg_prev_grad[i_d, i_b] = constraint_state.grad[i_d, i_b]
                constraint_state.cg_prev_Mgrad[i_d, i_b] = constraint_state.Mgrad[i_d, i_b]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_update_constraint_wrapper(
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
):
    """Update constraints. Parallelizes over batches only."""
    _B = constraint_state.grad.shape[1]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0:
            constraint_solver_decomp.func_update_constraint(
                i_b,
                qacc=constraint_state.qacc,
                Ma=constraint_state.Ma,
                cost=constraint_state.cost,
                dofs_state=dofs_state,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_nt_hessian_incremental_wrapper(
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    """Update Newton Hessian incrementally. Parallelizes over batches only."""
    _B = constraint_state.grad.shape[1]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0:
            constraint_solver_decomp.func_nt_hessian_incremental(
                i_b,
                entities_info=entities_info,
                constraint_state=constraint_state,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
            )


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_update_gradient_wrapper(
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
):
    """Update gradient. Parallelizes over batches only."""
    _B = constraint_state.grad.shape[1]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0:
            constraint_solver_decomp.func_update_gradient(
                i_b,
                dofs_state=dofs_state,
                entities_info=entities_info,
                rigid_global_info=rigid_global_info,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_check_convergence_and_update_search(
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    """Check convergence and update search direction. Parallelizes over batches only."""
    _B = constraint_state.grad.shape[1]
    n_dofs = constraint_state.grad.shape[0]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0:
            # Check convergence
            tol_scaled = (rigid_global_info.meaninertia[i_b] * ti.max(1, n_dofs)) * rigid_global_info.tolerance[None]
            improvement = constraint_state.prev_cost[i_b] - constraint_state.cost[i_b]
            gradient = gs.ti_float(0.0)
            for i_d in range(n_dofs):
                gradient = gradient + constraint_state.grad[i_d, i_b] * constraint_state.grad[i_d, i_b]
            gradient = ti.sqrt(gradient)

            if gradient < tol_scaled or improvement < tol_scaled:
                constraint_state.improved[i_b] = False
            else:
                constraint_state.improved[i_b] = True

                # Update search direction
                if ti.static(static_rigid_sim_config.solver_type == gs.constraint_solver.Newton):
                    # Newton: search = -Mgrad
                    for i_d in range(n_dofs):
                        constraint_state.search[i_d, i_b] = -constraint_state.Mgrad[i_d, i_b]
                else:
                    # CG: compute beta and update search = -Mgrad + beta * search
                    cg_beta = gs.ti_float(0.0)
                    cg_pg_dot_pMg = gs.ti_float(0.0)

                    for i_d in range(n_dofs):
                        cg_beta = cg_beta + constraint_state.grad[i_d, i_b] * (
                            constraint_state.Mgrad[i_d, i_b] - constraint_state.cg_prev_Mgrad[i_d, i_b]
                        )
                        cg_pg_dot_pMg = cg_pg_dot_pMg + (
                            constraint_state.cg_prev_Mgrad[i_d, i_b] * constraint_state.cg_prev_grad[i_d, i_b]
                        )
                    cg_beta = ti.max(cg_beta / ti.max(rigid_global_info.EPS[None], cg_pg_dot_pMg), 0.0)

                    constraint_state.cg_pg_dot_pMg[i_b] = cg_pg_dot_pMg
                    constraint_state.cg_beta[i_b] = cg_beta

                    for i_d in range(n_dofs):
                        constraint_state.search[i_d, i_b] = (
                            -constraint_state.Mgrad[i_d, i_b] + cg_beta * constraint_state.search[i_d, i_b]
                        )


# =============================================================================
# Decomposed Solver Orchestration
# =============================================================================


def func_solve_decomposed(
    entities_info,
    dofs_state,
    constraint_state,
    rigid_global_info,
    static_rigid_sim_config,
):
    """
    Decomposed constraint solver maintaining original batch-level parallelization.
    
    This function orchestrates the decomposed kernels to perform the same computation
    as func_solve, but split into separate kernel calls for better profiling and
    potential optimization, while maintaining the original parallelization over batches only.
    
    Args:
        entities_info: Entity information array
        dofs_state: DOF state array
        constraint_state: Constraint state array
        rigid_global_info: Global rigid body info
        static_rigid_sim_config: Static configuration
    """
    iterations = rigid_global_info.iterations[None]
    
    for _ in range(iterations):
        # 1. Compute linesearch to find optimal step size alpha and update qacc, Ma, Jaref
        kernel_linesearch_and_update(
            entities_info,
            dofs_state,
            constraint_state,
            rigid_global_info,
            static_rigid_sim_config,
        )
        
        # 2. Save prev_grad and prev_Mgrad for CG beta computation (CG only)
        if static_rigid_sim_config.solver_type == gs.constraint_solver.CG:
            kernel_save_prev_grad_Mgrad(
                constraint_state,
                static_rigid_sim_config,
            )
        
        # 3. Update constraint: efc_force and active flags
        kernel_update_constraint_wrapper(
            dofs_state,
            constraint_state,
            static_rigid_sim_config,
        )
        
        # 4. Newton Hessian update (Newton only)
        if static_rigid_sim_config.solver_type == gs.constraint_solver.Newton:
            kernel_nt_hessian_incremental_wrapper(
                entities_info,
                constraint_state,
                rigid_global_info,
                static_rigid_sim_config,
            )
        
        # 5. Update gradient
        kernel_update_gradient_wrapper(
            dofs_state,
            entities_info,
            rigid_global_info,
            constraint_state,
            static_rigid_sim_config,
        )
        
        # 6. Check convergence and update search direction
        kernel_check_convergence_and_update_search(
            constraint_state,
            rigid_global_info,
            static_rigid_sim_config,
        )

