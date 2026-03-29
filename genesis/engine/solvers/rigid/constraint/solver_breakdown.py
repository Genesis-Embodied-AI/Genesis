import numpy as np
import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.solvers.rigid.constraint import solver


@qd.func
def _func_linesearch(
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    _B = constraint_state.grad.shape[1]
    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
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


@qd.func
def _func_cg_only_save_prev_grad(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    _B = constraint_state.grad.shape[1]
    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            solver.func_save_prev_grad(i_b, constraint_state=constraint_state)


@qd.func
def _func_update_constraint_forces(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
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


@qd.func
def _func_update_constraint_qfrc(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    n_dofs = constraint_state.qfrc_constraint.shape[0]
    _B = constraint_state.grad.shape[1]

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

            for i_d in range(n_dofs):
                v = (
                    0.5
                    * (constraint_state.Ma[i_d, i_b] - dofs_state.force[i_d, i_b])
                    * (constraint_state.qacc[i_d, i_b] - dofs_state.acc_smooth[i_d, i_b])
                )
                gauss_i += v
                cost_i += v

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


@qd.func
def _func_update_gradient(
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
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


@qd.func
def _func_update_search_direction(
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
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


@qd.func
def _func_check_early_exit(
    constraint_state: array_class.ConstraintState,
    graph_continue_loop: qd.types.ndarray(qd.i32, ndim=0),
):
    """Decrement iteration counter and exit early if no batch element improved."""
    for _ in range(1):
        graph_continue_loop[()] = graph_continue_loop[()] - 1
        _B = constraint_state.grad.shape[1]
        any_improved = False
        for i_b in range(_B):
            if constraint_state.improved[i_b]:
                any_improved = True
        if not any_improved:
            graph_continue_loop[()] = 0


@qd.kernel(cuda_graph=True, fastcache=gs.use_fastcache)
def _kernel_solve_cuda_graph(
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    graph_continue_loop: qd.types.ndarray(qd.i32, ndim=0),
):
    while qd.graph_do_while(graph_continue_loop):
        _func_linesearch(entities_info, dofs_state, constraint_state, rigid_global_info, static_rigid_sim_config)
        if qd.static(static_rigid_sim_config.solver_type == gs.constraint_solver.CG):
            _func_cg_only_save_prev_grad(constraint_state, static_rigid_sim_config)
        _func_update_constraint_forces(constraint_state, static_rigid_sim_config)
        _func_update_constraint_qfrc(constraint_state, static_rigid_sim_config)
        _func_update_constraint_cost(dofs_state, constraint_state, static_rigid_sim_config)
        if qd.static(static_rigid_sim_config.solver_type == gs.constraint_solver.Newton):
            _func_newton_only_nt_hessian(constraint_state, rigid_global_info, static_rigid_sim_config)
        _func_update_gradient(entities_info, dofs_state, constraint_state, rigid_global_info, static_rigid_sim_config)
        _func_update_search_direction(constraint_state, rigid_global_info, static_rigid_sim_config)
        _func_check_early_exit(constraint_state, graph_continue_loop)


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
    CUDA-graph accelerated solver loop with GPU-side iteration via graph_do_while.

    On SM 9.0+ (Hopper), the entire iteration loop runs on the GPU with no host
    involvement. On older GPUs, falls back to a host-side do-while loop that still
    benefits from CUDA graph kernel launch batching.

    Early exits when all batch elements have converged (no improved[i_b] is True).
    """
    if _n_iterations <= 0:
        return
    constraint_state.graph_continue_loop.from_numpy(np.array(_n_iterations, dtype=np.int32))
    _kernel_solve_cuda_graph(
        entities_info,
        dofs_state,
        constraint_state,
        rigid_global_info,
        static_rigid_sim_config,
        constraint_state.graph_continue_loop,
    )
