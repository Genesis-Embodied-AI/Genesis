"""
Decomposed Constraint Solver with Batch-level Parallelization.

This module contains decomposed kernels for the constraint solver that maintain
the original batch-level parallelization pattern for compatibility with the
existing codebase structure.

Unlike constraint_solver_breakdown.py which uses multi-dimensional parallelization
over (batch, dofs) and (batch, constraints), these kernels parallelize only over
batches, matching the original func_solve kernel's parallelization strategy.

The solver is decomposed into a single kernel with multiple top-level for loops,
which reduces Python→C++ boundary crossing overhead while still allowing each step
to be profiled separately as Taichi can launch them as separate GPU kernels internally.
"""

import gstaichi as ti

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.engine.solvers.rigid.constraint_solver_decomp as constraint_solver_decomp


# =============================================================================
# Decomposed Solver Kernel - Single kernel with multiple top-level loops
# =============================================================================


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_solve_body_decomposed(
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    """
    Single kernel containing all solver steps as separate top-level loops.

    This reduces Python→C++ boundary crossing overhead (1 call per iteration instead of 6)
    while still allowing Taichi to launch each step as a separate GPU kernel internally
    for profiling purposes.
    """
    _B = constraint_state.grad.shape[1]
    n_dofs = constraint_state.qacc.shape[0]

    # Step 1: Linesearch and update qacc, Ma, Jaref
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    # Index: 0
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
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
                # we need alpha for this, so stay in same top level for loop
                # (though we could store alpha in a new tensor of course, if we wanted to split this)
                for i_d in range(n_dofs):
                    constraint_state.qacc[i_d, i_b] = (
                        constraint_state.qacc[i_d, i_b] + constraint_state.search[i_d, i_b] * alpha
                    )
                    constraint_state.Ma[i_d, i_b] = (
                        constraint_state.Ma[i_d, i_b] + constraint_state.mv[i_d, i_b] * alpha
                    )

                # Update Jaref
                for i_c in range(constraint_state.n_constraints[i_b]):
                    constraint_state.Jaref[i_c, i_b] = (
                        constraint_state.Jaref[i_c, i_b] + constraint_state.jv[i_c, i_b] * alpha
                    )
        else:
            constraint_state.improved[i_b] = False

    # Step 2: Save prev_grad and prev_Mgrad (CG only)
    if ti.static(static_rigid_sim_config.solver_type == gs.constraint_solver.CG):
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
        # Index: 1 if CG
        for i_b in range(_B):
            if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
                for i_d in range(n_dofs):
                    constraint_state.cg_prev_grad[i_d, i_b] = constraint_state.grad[i_d, i_b]
                    constraint_state.cg_prev_Mgrad[i_d, i_b] = constraint_state.Mgrad[i_d, i_b]

    # Step 3: Update constraints
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    # Index: 1 if Newton else 2
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            constraint_solver_decomp.func_update_constraint(
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
                constraint_solver_decomp.func_nt_hessian_incremental(
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
            constraint_solver_decomp.func_update_gradient(
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

    Uses a single kernel with multiple top-level for loops per iteration, reducing
    Python→C++ boundary crossing overhead from 6× to 1× per iteration.

    This provides much better CPU performance (~3-6x faster than separate kernels)
    while still allowing profiling of individual steps on GPU.

    Args:
        entities_info: Entity information array
        dofs_state: DOF state array
        constraint_state: Constraint state array
        rigid_global_info: Global rigid body info
        static_rigid_sim_config: Static configuration
    """
    iterations = rigid_global_info.iterations[None]
    for _it in range(iterations):
        # Single kernel call containing all 6 steps as separate top-level loops
        # This reduces overhead: 1 Python→C++ crossing instead of 6
        kernel_solve_body_decomposed(
            entities_info,
            dofs_state,
            constraint_state,
            rigid_global_info,
            static_rigid_sim_config,
        )
