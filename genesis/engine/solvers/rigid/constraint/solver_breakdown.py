import quadrants as ti

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.solvers.rigid.constraint import solver


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
def _kernel_newton_only_nt_hessian(
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


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_init_update_constraint(
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
):
    """Init-only constraint update — wraps monolith's func_update_constraint for exact FP match."""
    solver.func_update_constraint(
        qacc=constraint_state.qacc,
        Ma=constraint_state.Ma,
        cost=constraint_state.cost,
        dofs_state=dofs_state,
        constraint_state=constraint_state,
        static_rigid_sim_config=static_rigid_sim_config,
    )


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_init_update_gradient(
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    """Init-only gradient update — wraps monolith's func_update_gradient (dispatches to tiled on GPU)."""
    solver.func_update_gradient(
        dofs_state=dofs_state,
        entities_info=entities_info,
        constraint_state=constraint_state,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
    )


@solver.func_solve_init.register(is_compatible=lambda *args, **kwargs: gs.backend in {gs.cuda})
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
    5. Update constraint (wraps monolith's func_update_constraint for exact FP match)
    6. Newton hessian (Newton only — reuse existing kernel)
    7. Update gradient (wraps monolith's func_update_gradient — uses tiled on GPU)
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

    # 5. Update constraint (init-specific: wraps monolith's func_update_constraint for exact FP match)
    _kernel_init_update_constraint(dofs_state, constraint_state, static_rigid_sim_config)

    # 6. Newton hessian (Newton only)
    if static_rigid_sim_config.solver_type == gs.constraint_solver.Newton:
        _kernel_newton_only_nt_hessian(constraint_state, rigid_global_info, static_rigid_sim_config)

    # 7. Update gradient (init-specific: wraps monolith's func_update_gradient, dispatches to tiled on GPU)
    _kernel_init_update_gradient(
        entities_info, dofs_state, constraint_state, rigid_global_info, static_rigid_sim_config
    )

    # 8. search = -Mgrad
    _kernel_init_search(constraint_state, static_rigid_sim_config)


# ============================================== Solve body kernels ================================================


@solver.func_solve_body.register(is_compatible=lambda *args, **kwargs: gs.backend in {gs.cuda})
def func_solve_decomposed(
    entities_info,
    dofs_state,
    constraint_state,
    rigid_global_info,
    static_rigid_sim_config,
    _n_iterations,
):
    """
    Uses separate kernels for each solver step per iteration.

    This maximizes kernel granularity, potentially allowing better GPU scheduling
    and more flexibility in execution, at the cost of more Python→C++ boundary crossings.
    """
    # _n_iterations is a Python-native int to avoid CPU-GPU sync (vs rigid_global_info.iterations[None])
    for _it in range(_n_iterations):
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
