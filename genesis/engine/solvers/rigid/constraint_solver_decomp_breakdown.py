import gstaichi as ti

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.engine.solvers.rigid.constraint_solver_decomp as constraint_solver_decomp


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
            constraint_solver_decomp.func_linesearch_top_level(
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
                constraint_solver_decomp.func_save_prev_grad(i_b, constraint_state=constraint_state)

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
            constraint_solver_decomp.func_update_search_direction(
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
            constraint_solver_decomp.func_linesearch_top_level(
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
            constraint_solver_decomp.func_save_prev_grad(i_b, constraint_state=constraint_state)


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
            constraint_solver_decomp.func_nt_hessian_incremental(
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
            constraint_solver_decomp.func_update_gradient(
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
            constraint_solver_decomp.func_update_search_direction(
                i_b,
                rigid_global_info=rigid_global_info,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )


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
    iterations = rigid_global_info.iterations[None]
    for _it in range(iterations):
        _kernel_linesearch_top_level(
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
            num_env = 4096
            import time
            ti.sync()
            start = time.time()
            _kernel_newton_only_nt_hessian_incremental(
                entities_info,
                constraint_state,
                rigid_global_info,
                static_rigid_sim_config,
            )
            ti.sync()
            end = time.time()
            elapsed = end - start
            if _it >= 1:
                active_changed = (prev_active != constraint_state.active.to_numpy()).sum()
            print(f"n_improved {constraint_state.improved.to_numpy().sum()} "
                  f"n_constraints_avg {constraint_state.n_constraints.to_numpy().sum() / num_env:.2f} "
                  f"nt_dofs {constraint_state.nt_H.shape[1]} "
                  f"n_active_avg {constraint_state.active.to_numpy().sum() / num_env:.4f} ",
                  end='',
            )
            if _it >= 1:
                  print(f"active_changed_avg {active_changed / num_env:.2f} ", end='')
            print(f"time {elapsed * 1e6:.0f}us ",
                  end='')
            print('')
            prev_active = constraint_state.active.to_numpy()
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
