import gstaichi as ti

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.engine.solvers.rigid.constraint_solver_decomp as constraint_solver_decomp


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
def _kernel_cholesky_warp_level(
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    """
    Single-kernel Cholesky factorization using warp-level synchronization.

    Each warp (32 threads) processes one batch through all columns sequentially.
    Warp-level sync eliminates the need for 70 separate kernel launches.

    Thread organization:
    - 4096 warps (one per batch)
    - 32 threads per warp
    - Sequential over columns with warp sync between phases
    - Grid-stride loop for off-diagonal elements
    """
    _B = constraint_state.grad.shape[1]
    n_dofs = constraint_state.nt_H.shape[1]
    EPS = rigid_global_info.EPS[None]
    WARP_SIZE = ti.static(32)

    # Launch with 32 threads per block (one warp per batch)
    ti.loop_config(block_dim=WARP_SIZE)
    for i in range(_B * WARP_SIZE):
        tid = i % WARP_SIZE
        i_b = i // WARP_SIZE

        if i_b < _B and constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            # Process all columns sequentially with warp-level synchronization
            for i_d in range(n_dofs):
                # Phase 1: Compute diagonal element L[i_b, i_d, i_d]
                # Only thread 0 computes this
                if tid == 0:
                    tmp = constraint_state.nt_H[i_b, i_d, i_d]
                    for j_d in range(i_d):
                        tmp = tmp - constraint_state.nt_H[i_b, i_d, j_d] ** 2
                    constraint_state.nt_H[i_b, i_d, i_d] = ti.sqrt(ti.max(tmp, EPS))

                # Warp sync: All threads wait for diagonal to be computed
                ti.simt.warp.sync(ti.u32(0xFFFFFFFF))

                # Phase 2: Compute off-diagonal elements L[i_b, j_d, i_d] for j_d > i_d
                # Use grid-stride loop to handle any number of DOFs
                inv_diag = gs.ti_float(1.0) / constraint_state.nt_H[i_b, i_d, i_d]
                j_d = i_d + 1 + tid
                while j_d < n_dofs:
                    # Compute dot product
                    dot = gs.ti_float(0.0)
                    for k_d in range(i_d):
                        dot = dot + constraint_state.nt_H[i_b, j_d, k_d] * constraint_state.nt_H[i_b, i_d, k_d]
                    # Update element
                    constraint_state.nt_H[i_b, j_d, i_d] = (constraint_state.nt_H[i_b, j_d, i_d] - dot) * inv_diag
                    # Stride to next element for this thread
                    j_d = j_d + WARP_SIZE

                # Warp sync: All threads wait for off-diagonal to be computed
                ti.simt.warp.sync(ti.u32(0xFFFFFFFF))


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_newton_only_nt_hessian_direct_parallel(
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    n_dofs: ti.i32,
):
    """
    Fully parallelized Newton Hessian matrix computation: H = J'*D*J + M

    This kernel parallelizes over (batch, dof1, dof2) to maximize GPU utilization.
    Computes only the Hessian matrix. Cholesky factorization is done separately
    using column-by-column parallel kernels.
    """
    _B = constraint_state.grad.shape[1]

    # Compute H = J'*D*J + M in parallel over all batch-dof pairs
    ti.loop_config(serialize=False, block_dim=256)
    for i_b, i_d1, i_d2 in ti.ndrange(_B, n_dofs, n_dofs):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b] and i_d2 <= i_d1:
            constraint_solver_decomp.func_nt_hessian_direct_B_dofs_dofs(
                i_b,
                i_d1,
                i_d2,
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
    and more flexibility in execution, at the cost of more Python→C++ boundary crossings.
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
            n_dofs = constraint_state.nt_H.shape[1]
            _kernel_newton_only_nt_hessian_direct_parallel(
                entities_info,
                constraint_state,
                rigid_global_info,
                static_rigid_sim_config,
                n_dofs,
            )

            _kernel_cholesky_warp_level(
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
