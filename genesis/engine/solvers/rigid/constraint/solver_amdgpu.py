"""
AMDGPU-specific variants of `func_solve_body`, registered with the perf
dispatch system in `solver.func_solve_body`. Each variant targets a
different bottleneck of the baked-in monolithic kernel
`func_solve_body_monolith` on gfx942 (wave64 hardware):

- `func_solve_body_split_amdgpu` (per-iteration 2-kernel split,
  linesearch + apply-alpha | post-linesearch). Iteration loop lifted
  to Python. Halves the inlined call chain seen by the LLVM AMDGPU
  backend inside each kernel, giving the register allocator a smaller
  live-range graph to budget. Targets register pressure.

- `func_solve_body_lifted_loop_amdgpu` (per-iteration single-kernel
  launch, full `func_solve_iter` body, iteration loop lifted to
  Python). Keeps the body monolithic but kills the cross-iteration
  live ranges the compiler would otherwise have to carry through the
  `for _ in range(iterations):` loop. Targets register pressure.

- `func_solve_body_decomposed_amdgpu` (7-kernel-per-iter decomposition
  that exposes 2D `(n_dofs, _B)` and `(n_constraints, _B)` parallelism
  for the two biggest inner loops -- J^T @ efc_force and efc_force
  update). Mirrors the CUDA `func_solve_decomposed` in
  solver_breakdown.py but pins `block_dim=64` for wave64 lane
  utilization on gfx942. Targets monolith SIMD under-utilization on
  workloads where the wavefront count from the monolith dispatch is
  small relative to the device's SIMD count.

- `func_solve_body_wavecoop_amdgpu` (one wavefront per env, all 64
  lanes cooperate on the same env's CG iter). Cross-lane reductions
  go through LDS scratch. Targets workloads with large per-env work
  (e.g. high n_dofs) where the cooperative reduction beats the
  scalar-per-env path. See the lane-utilization breakdown in the
  block comment above the variant for when this wins/loses vs the
  monolith.

- `func_solve_body_tiled_wc_amdgpu` (tiled wave-cooperative, 8 envs *
  8 lanes-per-env packed into one wave64). Sits between the monolith
  (1 env per thread, 100% lane use on every phase) and the full
  wave-coop (1 env per wavefront, 1 active lane during per-env scalar
  phases). Recovers per-env scalar-phase lane utilization while
  keeping the cooperative reduction benefit for the per-DOF and
  per-constraint loops.

Per-batch convergence semantics: every per-iter kernel gates on
`constraint_state.improved[i_b]` (a device-side read, no D2H sync) and
skips work for batches that have already converged. Without this gate,
converged batches would re-run linesearch using a stale `search`
direction and inject FP noise into qacc/Ma/Jaref that accumulates over
many sim steps. This matches the existing CUDA `func_solve_decomposed`
behavior in `solver_breakdown.py`.

All variants pin `block_dim=64` to avoid the 50% VALU-lane-masking
penalty wave64 hardware imposes on 32-thread workgroups (see comment on
`block_dim` in `func_solve_body_monolith`).
"""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.solvers.rigid.constraint import solver


# ---------------------------------------------------------------------------
# B3: 2-kernel split (linesearch | post-linesearch)
# ---------------------------------------------------------------------------


@qd.kernel(fastcache=gs.use_fastcache)
def _kernel_linesearch_amdgpu(
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    _B = static_rigid_sim_config.n_envs
    qd.loop_config(
        serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL,
        block_dim=64,
    )
    for i_b in range(_B):
        # Gate linesearch on improved[i_b] to mirror baseline `if not improved: break`.
        # Without this gate, a converged batch (improved=False) would re-run linesearch on the
        # next iteration using a stale `search` direction; if the resulting alpha is non-trivial
        # the qacc/Ma/Jaref updates inject noise that accumulates over many sim steps.
        # improved[i_b] is read device-side, so this gate adds no host sync cost.
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


@qd.kernel(fastcache=gs.use_fastcache)
def _kernel_solve_iter_post_linesearch_amdgpu(
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    _B = static_rigid_sim_config.n_envs
    qd.loop_config(
        serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL,
        block_dim=64,
    )
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            solver.func_solve_iter_post_linesearch(
                i_b,
                entities_info=entities_info,
                dofs_state=dofs_state,
                rigid_global_info=rigid_global_info,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )


@solver.func_solve_body.register(is_compatible=lambda *args, **kwargs: gs.backend in {gs.amdgpu})
def func_solve_body_split_amdgpu(
    entities_info,
    dofs_state,
    constraint_state,
    rigid_global_info,
    static_rigid_sim_config,
    _n_iterations,
):
    # _n_iterations is a Python-native int (avoids GPU sync that
    # rigid_global_info.iterations[None] would force).
    for _it in range(_n_iterations):
        _kernel_linesearch_amdgpu(
            entities_info,
            dofs_state,
            constraint_state,
            rigid_global_info,
            static_rigid_sim_config,
        )
        _kernel_solve_iter_post_linesearch_amdgpu(
            entities_info,
            dofs_state,
            constraint_state,
            rigid_global_info,
            static_rigid_sim_config,
        )


# ---------------------------------------------------------------------------
# B4: lifted iteration loop, monolithic per-iter body
# ---------------------------------------------------------------------------


@qd.kernel(fastcache=gs.use_fastcache)
def _kernel_solve_one_iter_amdgpu(
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    _B = static_rigid_sim_config.n_envs
    qd.loop_config(
        serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL,
        block_dim=64,
    )
    for i_b in range(_B):
        # Same gating rationale as the B3 linesearch kernel: skip work for batches that have
        # already converged so we don't apply spurious alpha steps using a stale search direction.
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            solver.func_solve_iter(
                i_b,
                entities_info=entities_info,
                dofs_state=dofs_state,
                rigid_global_info=rigid_global_info,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )
        else:
            constraint_state.improved[i_b] = False


@solver.func_solve_body.register(is_compatible=lambda *args, **kwargs: gs.backend in {gs.amdgpu})
def func_solve_body_lifted_loop_amdgpu(
    entities_info,
    dofs_state,
    constraint_state,
    rigid_global_info,
    static_rigid_sim_config,
    _n_iterations,
):
    for _it in range(_n_iterations):
        _kernel_solve_one_iter_amdgpu(
            entities_info,
            dofs_state,
            constraint_state,
            rigid_global_info,
            static_rigid_sim_config,
        )


# ---------------------------------------------------------------------------
# 7-kernel decomposition with 2D (n_dofs, _B) and (n_con, _B) parallelism
# ---------------------------------------------------------------------------
#
# Mirrors the CUDA `func_solve_decomposed` but tunes block_dim=64 for
# wave64 hardware on gfx942. The two highest-fan-out loops (J^T @
# efc_force and per-constraint force / active flag update) are
# flattened into 2D ndranges so the dispatch produces enough wavefronts
# to saturate the device's SIMDs, instead of leaving them mostly idle
# the way the per-env monolith dispatch does on small batch sizes.
#
# The per-env kernels (linesearch, gradient update, search-direction,
# cost) keep the same 1-thread-per-env shape as the monolith but use
# block_dim=64 to keep wave64 lanes fully utilized (the CUDA decomposed
# variant uses block_dim=32, which would mask 32 of the 64 lanes per
# wavefront on AMDGPU).
#
# This variant is incompatible with sparse_solve=True because
# `_kernel_update_constraint_qfrc_amdgpu` only handles the dense
# Jacobian path. Sparse layouts must continue to use the monolithic or
# split AMD variants until the sparse scatter-add is decomposed
# separately.


@qd.kernel(fastcache=gs.use_fastcache)
def _kernel_linesearch_amdgpu_decomposed(
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    _B = static_rigid_sim_config.n_envs
    qd.loop_config(
        serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL,
        block_dim=64,
    )
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


@qd.kernel(fastcache=gs.use_fastcache)
def _kernel_cg_save_prev_grad_amdgpu_decomposed(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    _B = static_rigid_sim_config.n_envs
    qd.loop_config(
        serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL,
        block_dim=64,
    )
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            solver.func_save_prev_grad(
                i_b,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )


@qd.kernel(fastcache=gs.use_fastcache)
def _kernel_update_constraint_forces_amdgpu_decomposed(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """Compute active flags and efc_force, parallelized over (i_c, i_b).

    The 2D fan-out produces enough work-items / wavefronts to saturate
    the device's SIMDs at typical batch sizes, enabling latency hiding
    on the chained Jaref/efc_D/diag/frictionloss loads through L2.
    """
    len_constraints = constraint_state.active.shape[0]
    _B = static_rigid_sim_config.n_envs

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
                -constraint_state.Jaref[i_c, i_b]
                * constraint_state.efc_D[i_c, i_b]
                * constraint_state.active[i_c, i_b]
            )


@qd.kernel(fastcache=gs.use_fastcache)
def _kernel_update_constraint_qfrc_amdgpu_decomposed(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """qfrc_constraint = J^T @ efc_force, parallelized over (i_d, i_b).

    The 2D fan-out is much wider than the per-env monolith dispatch and
    keeps the device's SIMDs busy on workloads where the monolith
    would leave most of them idle. Each thread does the n_constraints
    inner sum serially, which is the right granularity given the
    streaming `jac[*, i_d, i_b]` access pattern.

    Dense-only: this kernel reads `constraint_state.jac` directly and
    does NOT respect `sparse_solve=True`. The decomposed variant is
    registered with `is_compatible` that enforces `sparse_solve=False`.
    """
    n_dofs = constraint_state.qfrc_constraint.shape[0]
    _B = static_rigid_sim_config.n_envs

    for i_d, i_b in qd.ndrange(n_dofs, _B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            n_con = constraint_state.n_constraints[i_b]
            qfrc = gs.qd_float(0.0)
            for i_c in range(n_con):
                qfrc += constraint_state.jac[i_c, i_d, i_b] * constraint_state.efc_force[i_c, i_b]
            constraint_state.qfrc_constraint[i_d, i_b] = qfrc


@qd.kernel(fastcache=gs.use_fastcache)
def _kernel_update_constraint_cost_amdgpu_decomposed(
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """Compute gauss + cost (scalar reductions per env). One wave per env.

    Gauss reduction over n_dofs and constraint cost reduction over n_con
    are kept serial-per-thread because parallelizing them across a wave
    requires LDS reductions (and hits the same global-memory loads twice
    if not also fused with the qfrc kernel). This kernel is small
    relative to the qfrc/forces ones and is left at one-thread-per-env.
    """
    _B = static_rigid_sim_config.n_envs

    qd.loop_config(
        serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL,
        block_dim=64,
    )
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


@qd.kernel(fastcache=gs.use_fastcache)
def _kernel_update_gradient_amdgpu_decomposed(
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    _B = static_rigid_sim_config.n_envs
    qd.loop_config(
        serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL,
        block_dim=64,
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


@qd.kernel(fastcache=gs.use_fastcache)
def _kernel_update_search_direction_amdgpu_decomposed(
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    _B = static_rigid_sim_config.n_envs
    qd.loop_config(
        serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL,
        block_dim=64,
    )
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            prev_cost = constraint_state.prev_cost[i_b]
            solver.func_terminate_or_update_descent_batch(
                i_b,
                prev_cost,
                rigid_global_info=rigid_global_info,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )


def _decomposed_amdgpu_is_compatible(*args, **kwargs):
    # `func_solve_body` is invoked positionally everywhere (see solver.py
    # ConstraintSolver.solve), so static_rigid_sim_config arrives as args[4].
    if gs.backend not in {gs.amdgpu}:
        return False
    # Dense path only -- _kernel_update_constraint_qfrc_amdgpu_decomposed
    # reads constraint_state.jac directly. Sparse decomposition is a
    # separate kernel design (scatter-add) and not implemented here yet.
    cfg = kwargs.get("static_rigid_sim_config", args[4] if len(args) >= 5 else None)
    if cfg is None:
        return False
    if cfg.sparse_solve:
        return False
    # Newton solver needs the in-iter Hessian/Cholesky update which is only
    # inlined in the monolithic and split variants. CG (the default for our
    # perf workload) has no such requirement.
    if cfg.solver_type != gs.constraint_solver.CG:
        return False
    # Require a true batched workload. The decomposition's reduction order
    # differs from the monolith (per-DOF parallel scatter-add of J^T*efc_force
    # vs the monolith's serial accumulation), which produces ULP-level FP
    # drift. That drift is benign for the benchmarked batch sizes but
    # compounds into multi-step trajectory divergence on small/unbatched
    # workloads (n_envs=0 maps to cfg.n_envs=1 internally) that the unit
    # tests assert against tight tolerances. Gate to batched mode so the
    # variant only competes on the workloads it was designed and validated
    # for. The threshold matches the tiled-wc tile size for consistency.
    if cfg.n_envs < 8:
        return False
    return True


# ---------------------------------------------------------------------------
# Wave-cooperative monolith (1 wave / env)
# ---------------------------------------------------------------------------
#
# Structure: one workgroup of 64 threads per env. The 64 lanes of one
# wave cooperate on the same env's CG iter, partitioning per-DOF and
# per-constraint loops by tid stride. Cross-lane reductions go through
# LDS scratch. This expands the wavefront count of the monolith
# dispatch by ~64x (one wave per env instead of one thread per env),
# which can win on workloads with enough per-env work to amortize the
# LDS reduction overhead and where the monolith's wavefront count is
# small relative to the device's SIMD count.
#
# Phases parallelized across the 64 lanes:
#   * Apply-alpha: qacc/Ma update + Jaref update.
#   * Save-prev-grad: cg_prev_grad/cg_prev_Mgrad copy.
#   * Update-constraint-batch: active flags + efc_force + J^T @
#     efc_force + cost/gauss reductions; cost/gauss final sums via LDS
#     reduction.
#   * Linesearch (`func_linesearch_batch_wavecoop` below): every
#     reduction over n_dofs (snorm, quad_gauss) and n_con (mv/jv
#     matvecs, quad_total + eq_sum, point-fn evals, 3-alpha
#     refinement) is partitioned across 64 lanes with an LDS final
#     reduction; bracketing scalar logic runs redundantly on every
#     lane using broadcast inputs.
#
# Phases that stay scalar on lane 0:
#   * Gradient update + mass solve + terminate-or-update-descent.
#     For small n_dofs these phases are dominated by the per-DOF work
#     itself (LDS roundtrip costs more than the work), so cooperative
#     reduction doesn't help here. On small-n_dofs robots this also
#     means the wave-coop wavefront spends a large fraction of its
#     time with 63 of 64 lanes idle, which limits where this variant
#     wins; the tiled wave-coop variant below recovers that lane
#     utilization for small-n_dofs workloads.
#
# Compatibility: dense Jacobian (`sparse_solve=False`) and CG only.
# Same rationale as the decomposed variant: the dense J^T @ efc_force
# loop is what benefits most from 64-way intra-env parallelism, and
# the Newton path wants its in-iter Hessian/Cholesky update which is
# only inlined in the monolith and split variants.


# ---------------------------------------------------------------------------
# Wave-coop linesearch helpers
# ---------------------------------------------------------------------------
#
# Each helper takes the lane index `tid` (0..63) and the env index `i_b`,
# allocates LDS partial-sum + broadcast slots, performs cooperative
# reductions over n_dofs / n_con / n_entities, and returns broadcast
# scalar results to ALL 64 lanes. The scalar bracketing logic in
# `func_linesearch_batch_wavecoop` then runs redundantly on every lane:
# all lanes see identical reduction results so they compute identical
# next-alpha decisions without needing per-lane broadcasts.
#
# FP-determinism strategy: lane-strided partitioning + sequential
# lane-order accumulation on tid=0 (`for k in range(BLOCK_DIM): total +=
# red[k]`). This gives a fixed reduction order across runs but differs
# from monolith's pure scalar order in the regrouping induced by 64
# parallel partial accumulators -- expected ULP-level drift, same as
# the cost/gauss reduction in `update_constraint` below.


@qd.func
def _func_ls_pt_opt_wc(
    i_b,
    tid,
    alpha,
    base_0,
    base_1,
    base_2,
    ls_it,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
):
    """Wave-cooperative version of func_ls_point_fn_opt.

    Each lane handles a strided slice of the friction + contact constraint
    range and accumulates 3 partial sums (cost / grad / hess coefficients).
    LDS reduces them across the wavefront; all 64 lanes leave with
    identical broadcast (alpha, cost, grad, hess, ls_it).
    """
    BLOCK_DIM = qd.static(64)
    pt_red = qd.simt.block.SharedArray((3, BLOCK_DIM), gs.qd_float)
    pt_bcast = qd.simt.block.SharedArray((3,), gs.qd_float)

    ne = constraint_state.n_constraints_equality[i_b]
    nef = ne + constraint_state.n_constraints_frictionloss[i_b]
    n_con = constraint_state.n_constraints[i_b]

    my_t0 = gs.qd_float(0.0)
    my_t1 = gs.qd_float(0.0)
    my_t2 = gs.qd_float(0.0)

    # Friction constraints [ne, nef) - strided across lanes.
    i_c = ne + tid
    while i_c < nef:
        Jaref_c = constraint_state.Jaref[i_c, i_b]
        jv_c = constraint_state.jv[i_c, i_b]
        D = constraint_state.efc_D[i_c, i_b]
        f = constraint_state.efc_frictionloss[i_c, i_b]
        r = constraint_state.diag[i_c, i_b]
        qf_0 = D * (0.5 * Jaref_c * Jaref_c)
        qf_1 = D * (jv_c * Jaref_c)
        qf_2 = D * (0.5 * jv_c * jv_c)
        x = Jaref_c + alpha * jv_c
        rf = r * f
        linear_neg = x <= -rf
        linear_pos = x >= rf
        if linear_neg or linear_pos:
            qf_0 = linear_neg * f * (-0.5 * rf - Jaref_c) + linear_pos * f * (-0.5 * rf + Jaref_c)
            qf_1 = linear_neg * (-f * jv_c) + linear_pos * (f * jv_c)
            qf_2 = 0.0
        my_t0 = my_t0 + qf_0
        my_t1 = my_t1 + qf_1
        my_t2 = my_t2 + qf_2
        i_c = i_c + BLOCK_DIM

    # Contact constraints [nef, n_con) - strided across lanes.
    i_c = nef + tid
    while i_c < n_con:
        Jaref_c = constraint_state.Jaref[i_c, i_b]
        jv_c = constraint_state.jv[i_c, i_b]
        D = constraint_state.efc_D[i_c, i_b]
        x = Jaref_c + alpha * jv_c
        active = x < 0
        qf_0 = D * (0.5 * Jaref_c * Jaref_c)
        qf_1 = D * (jv_c * Jaref_c)
        qf_2 = D * (0.5 * jv_c * jv_c)
        my_t0 = my_t0 + qf_0 * active
        my_t1 = my_t1 + qf_1 * active
        my_t2 = my_t2 + qf_2 * active
        i_c = i_c + BLOCK_DIM

    pt_red[0, tid] = my_t0
    pt_red[1, tid] = my_t1
    pt_red[2, tid] = my_t2
    qd.simt.block.sync()
    if tid == 0:
        s0 = base_0
        s1 = base_1
        s2 = base_2
        for k in qd.static(range(BLOCK_DIM)):
            s0 = s0 + pt_red[0, k]
            s1 = s1 + pt_red[1, k]
            s2 = s2 + pt_red[2, k]
        cost = alpha * alpha * s2 + alpha * s1 + s0
        grad = 2 * alpha * s2 + s1
        hess = 2 * s2
        if hess <= 0.0:
            hess = rigid_global_info.EPS[None]
        pt_bcast[0] = cost
        pt_bcast[1] = grad
        pt_bcast[2] = hess
    qd.simt.block.sync()
    return alpha, pt_bcast[0], pt_bcast[1], pt_bcast[2], ls_it + 1


@qd.func
def _func_ls_pt_3a_wc(
    i_b,
    tid,
    alpha_0,
    alpha_1,
    alpha_2,
    base_0,
    base_1,
    base_2,
    ls_it,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
):
    """Wave-cooperative version of func_ls_point_fn_3alphas_opt.

    Returns 3 (cost, grad, hess) tuples broadcast to all lanes. The 9
    partial sums are accumulated per-lane in registers and reduced via
    a single 9-slot LDS array.
    """
    BLOCK_DIM = qd.static(64)
    pt3_red = qd.simt.block.SharedArray((9, BLOCK_DIM), gs.qd_float)
    pt3_bcast = qd.simt.block.SharedArray((9,), gs.qd_float)

    ne = constraint_state.n_constraints_equality[i_b]
    nef = ne + constraint_state.n_constraints_frictionloss[i_b]
    n_con = constraint_state.n_constraints[i_b]

    t0_0 = gs.qd_float(0.0)
    t0_1 = gs.qd_float(0.0)
    t0_2 = gs.qd_float(0.0)
    t1_0 = gs.qd_float(0.0)
    t1_1 = gs.qd_float(0.0)
    t1_2 = gs.qd_float(0.0)
    t2_0 = gs.qd_float(0.0)
    t2_1 = gs.qd_float(0.0)
    t2_2 = gs.qd_float(0.0)

    # Friction [ne, nef): 3 alphas evaluated together, lane-strided.
    i_c = ne + tid
    while i_c < nef:
        Jaref_c = constraint_state.Jaref[i_c, i_b]
        jv_c = constraint_state.jv[i_c, i_b]
        D = constraint_state.efc_D[i_c, i_b]
        f = constraint_state.efc_frictionloss[i_c, i_b]
        r = constraint_state.diag[i_c, i_b]
        qf_0 = D * (0.5 * Jaref_c * Jaref_c)
        qf_1 = D * (jv_c * Jaref_c)
        qf_2 = D * (0.5 * jv_c * jv_c)
        rf = r * f

        x0 = Jaref_c + alpha_0 * jv_c
        ln0 = x0 <= -rf
        lp0 = x0 >= rf
        a0_qf_0, a0_qf_1, a0_qf_2 = qf_0, qf_1, qf_2
        if ln0 or lp0:
            a0_qf_0 = ln0 * f * (-0.5 * rf - Jaref_c) + lp0 * f * (-0.5 * rf + Jaref_c)
            a0_qf_1 = ln0 * (-f * jv_c) + lp0 * (f * jv_c)
            a0_qf_2 = 0.0
        t0_0 = t0_0 + a0_qf_0
        t0_1 = t0_1 + a0_qf_1
        t0_2 = t0_2 + a0_qf_2

        x1 = Jaref_c + alpha_1 * jv_c
        ln1 = x1 <= -rf
        lp1 = x1 >= rf
        a1_qf_0, a1_qf_1, a1_qf_2 = qf_0, qf_1, qf_2
        if ln1 or lp1:
            a1_qf_0 = ln1 * f * (-0.5 * rf - Jaref_c) + lp1 * f * (-0.5 * rf + Jaref_c)
            a1_qf_1 = ln1 * (-f * jv_c) + lp1 * (f * jv_c)
            a1_qf_2 = 0.0
        t1_0 = t1_0 + a1_qf_0
        t1_1 = t1_1 + a1_qf_1
        t1_2 = t1_2 + a1_qf_2

        x2 = Jaref_c + alpha_2 * jv_c
        ln2 = x2 <= -rf
        lp2 = x2 >= rf
        a2_qf_0, a2_qf_1, a2_qf_2 = qf_0, qf_1, qf_2
        if ln2 or lp2:
            a2_qf_0 = ln2 * f * (-0.5 * rf - Jaref_c) + lp2 * f * (-0.5 * rf + Jaref_c)
            a2_qf_1 = ln2 * (-f * jv_c) + lp2 * (f * jv_c)
            a2_qf_2 = 0.0
        t2_0 = t2_0 + a2_qf_0
        t2_1 = t2_1 + a2_qf_1
        t2_2 = t2_2 + a2_qf_2
        i_c = i_c + BLOCK_DIM

    # Contact [nef, n_con).
    i_c = nef + tid
    while i_c < n_con:
        Jaref_c = constraint_state.Jaref[i_c, i_b]
        jv_c = constraint_state.jv[i_c, i_b]
        D = constraint_state.efc_D[i_c, i_b]
        qf_0 = D * (0.5 * Jaref_c * Jaref_c)
        qf_1 = D * (jv_c * Jaref_c)
        qf_2 = D * (0.5 * jv_c * jv_c)
        x0 = Jaref_c + alpha_0 * jv_c
        x1 = Jaref_c + alpha_1 * jv_c
        x2 = Jaref_c + alpha_2 * jv_c
        act0 = gs.qd_bool(x0 < 0)
        act1 = gs.qd_bool(x1 < 0)
        act2 = gs.qd_bool(x2 < 0)
        t0_0 = t0_0 + qf_0 * act0
        t0_1 = t0_1 + qf_1 * act0
        t0_2 = t0_2 + qf_2 * act0
        t1_0 = t1_0 + qf_0 * act1
        t1_1 = t1_1 + qf_1 * act1
        t1_2 = t1_2 + qf_2 * act1
        t2_0 = t2_0 + qf_0 * act2
        t2_1 = t2_1 + qf_1 * act2
        t2_2 = t2_2 + qf_2 * act2
        i_c = i_c + BLOCK_DIM

    pt3_red[0, tid] = t0_0
    pt3_red[1, tid] = t0_1
    pt3_red[2, tid] = t0_2
    pt3_red[3, tid] = t1_0
    pt3_red[4, tid] = t1_1
    pt3_red[5, tid] = t1_2
    pt3_red[6, tid] = t2_0
    pt3_red[7, tid] = t2_1
    pt3_red[8, tid] = t2_2
    qd.simt.block.sync()
    if tid == 0:
        s00 = base_0
        s01 = base_1
        s02 = base_2
        s10 = base_0
        s11 = base_1
        s12 = base_2
        s20 = base_0
        s21 = base_1
        s22 = base_2
        for k in qd.static(range(BLOCK_DIM)):
            s00 = s00 + pt3_red[0, k]
            s01 = s01 + pt3_red[1, k]
            s02 = s02 + pt3_red[2, k]
            s10 = s10 + pt3_red[3, k]
            s11 = s11 + pt3_red[4, k]
            s12 = s12 + pt3_red[5, k]
            s20 = s20 + pt3_red[6, k]
            s21 = s21 + pt3_red[7, k]
            s22 = s22 + pt3_red[8, k]
        EPS = rigid_global_info.EPS[None]
        c0 = alpha_0 * alpha_0 * s02 + alpha_0 * s01 + s00
        g0 = 2 * alpha_0 * s02 + s01
        h0 = 2 * s02
        if h0 <= 0.0:
            h0 = EPS
        c1 = alpha_1 * alpha_1 * s12 + alpha_1 * s11 + s10
        g1 = 2 * alpha_1 * s12 + s11
        h1 = 2 * s12
        if h1 <= 0.0:
            h1 = EPS
        c2 = alpha_2 * alpha_2 * s22 + alpha_2 * s21 + s20
        g2 = 2 * alpha_2 * s22 + s21
        h2 = 2 * s22
        if h2 <= 0.0:
            h2 = EPS
        pt3_bcast[0] = c0
        pt3_bcast[1] = g0
        pt3_bcast[2] = h0
        pt3_bcast[3] = c1
        pt3_bcast[4] = g1
        pt3_bcast[5] = h1
        pt3_bcast[6] = c2
        pt3_bcast[7] = g2
        pt3_bcast[8] = h2
    qd.simt.block.sync()
    return (
        pt3_bcast[0],
        pt3_bcast[1],
        pt3_bcast[2],
        pt3_bcast[3],
        pt3_bcast[4],
        pt3_bcast[5],
        pt3_bcast[6],
        pt3_bcast[7],
        pt3_bcast[8],
        ls_it + 3,
    )


@qd.func
def _func_ls_init_p0_wc(
    i_b,
    tid,
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Wave-cooperative version of func_ls_init_and_eval_p0_opt.

    Computes mv (M*search per entity), jv (J*search per constraint),
    quad_gauss_{1,2} reductions over n_dofs, and quad_total + eq_sum
    reductions over n_con. Returns p0 evaluation tuple broadcast to
    all 64 lanes.
    """
    BLOCK_DIM = qd.static(64)
    init_red = qd.simt.block.SharedArray((8, BLOCK_DIM), gs.qd_float)
    init_bcast = qd.simt.block.SharedArray((9,), gs.qd_float)

    n_dofs = constraint_state.search.shape[0]
    n_entities = static_rigid_sim_config.n_entities_
    ne = constraint_state.n_constraints_equality[i_b]
    nef = ne + constraint_state.n_constraints_frictionloss[i_b]
    n_con = constraint_state.n_constraints[i_b]

    # 1) mv[i_d1] = sum_d2 mass_mat[d1,d2] * search[d2] (per entity).
    # Partition outer i_d1 across lanes; inner sum stays serial per lane.
    for i_e in range(n_entities):
        d_start = entities_info.dof_start[i_e]
        d_end = entities_info.dof_end[i_e]
        i_d1 = d_start + tid
        while i_d1 < d_end:
            mv = gs.qd_float(0.0)
            for i_d2 in range(d_start, d_end):
                mv = mv + rigid_global_info.mass_mat[i_d1, i_d2, i_b] * constraint_state.search[i_d2, i_b]
            constraint_state.mv[i_d1, i_b] = mv
            i_d1 = i_d1 + BLOCK_DIM
    qd.simt.block.sync()  # ensure mv is visible before quad_gauss_2 reads it

    # 2) jv[i_c] = sum_d jac[c,d] * search[d] (DENSE only, see is_compatible).
    i_c = tid
    while i_c < n_con:
        jv = gs.qd_float(0.0)
        for i_d in range(n_dofs):
            jv = jv + constraint_state.jac[i_c, i_d, i_b] * constraint_state.search[i_d, i_b]
        constraint_state.jv[i_c, i_b] = jv
        i_c = i_c + BLOCK_DIM
    qd.simt.block.sync()  # ensure jv is visible before constraint quads read it

    # 3) quad_gauss_1, quad_gauss_2 reductions over n_dofs (lane-strided).
    my_qg1 = gs.qd_float(0.0)
    my_qg2 = gs.qd_float(0.0)
    i_d = tid
    while i_d < n_dofs:
        s = constraint_state.search[i_d, i_b]
        Ma_d = constraint_state.Ma[i_d, i_b]
        f_d = dofs_state.force[i_d, i_b]
        mv_d = constraint_state.mv[i_d, i_b]
        my_qg1 = my_qg1 + s * Ma_d - s * f_d
        my_qg2 = my_qg2 + 0.5 * s * mv_d
        i_d = i_d + BLOCK_DIM

    # 4) quad_total + eq_sum reductions over n_con (lane-strided).
    my_qt0 = gs.qd_float(0.0)
    my_qt1 = gs.qd_float(0.0)
    my_qt2 = gs.qd_float(0.0)
    my_eq0 = gs.qd_float(0.0)
    my_eq1 = gs.qd_float(0.0)
    my_eq2 = gs.qd_float(0.0)
    i_c = tid
    while i_c < n_con:
        Jaref_c = constraint_state.Jaref[i_c, i_b]
        jv_c = constraint_state.jv[i_c, i_b]
        D = constraint_state.efc_D[i_c, i_b]
        qf_0 = D * (0.5 * Jaref_c * Jaref_c)
        qf_1 = D * (jv_c * Jaref_c)
        qf_2 = D * (0.5 * jv_c * jv_c)
        if i_c < ne:
            my_eq0 = my_eq0 + qf_0
            my_eq1 = my_eq1 + qf_1
            my_eq2 = my_eq2 + qf_2
            my_qt0 = my_qt0 + qf_0
            my_qt1 = my_qt1 + qf_1
            my_qt2 = my_qt2 + qf_2
        elif i_c < nef:
            f = constraint_state.efc_frictionloss[i_c, i_b]
            r = constraint_state.diag[i_c, i_b]
            rf = r * f
            linear_neg = Jaref_c <= -rf
            linear_pos = Jaref_c >= rf
            if linear_neg or linear_pos:
                qf_0 = linear_neg * f * (-0.5 * rf - Jaref_c) + linear_pos * f * (-0.5 * rf + Jaref_c)
                qf_1 = linear_neg * (-f * jv_c) + linear_pos * (f * jv_c)
                qf_2 = 0.0
            my_qt0 = my_qt0 + qf_0
            my_qt1 = my_qt1 + qf_1
            my_qt2 = my_qt2 + qf_2
        else:
            active = Jaref_c < 0
            my_qt0 = my_qt0 + qf_0 * active
            my_qt1 = my_qt1 + qf_1 * active
            my_qt2 = my_qt2 + qf_2 * active
        i_c = i_c + BLOCK_DIM

    # 5) LDS reduce 8 sums (qg1, qg2, qt0/1/2, eq0/1/2) -> totals on tid=0.
    init_red[0, tid] = my_qg1
    init_red[1, tid] = my_qg2
    init_red[2, tid] = my_qt0
    init_red[3, tid] = my_qt1
    init_red[4, tid] = my_qt2
    init_red[5, tid] = my_eq0
    init_red[6, tid] = my_eq1
    init_red[7, tid] = my_eq2
    qd.simt.block.sync()

    if tid == 0:
        sg1 = gs.qd_float(0.0)
        sg2 = gs.qd_float(0.0)
        st0 = gs.qd_float(0.0)
        st1 = gs.qd_float(0.0)
        st2 = gs.qd_float(0.0)
        se0 = gs.qd_float(0.0)
        se1 = gs.qd_float(0.0)
        se2 = gs.qd_float(0.0)
        for k in qd.static(range(BLOCK_DIM)):
            sg1 = sg1 + init_red[0, k]
            sg2 = sg2 + init_red[1, k]
            st0 = st0 + init_red[2, k]
            st1 = st1 + init_red[3, k]
            st2 = st2 + init_red[4, k]
            se0 = se0 + init_red[5, k]
            se1 = se1 + init_red[6, k]
            se2 = se2 + init_red[7, k]
        quad_gauss_0 = constraint_state.gauss[i_b]
        quad_total_0 = quad_gauss_0 + st0
        quad_total_1 = sg1 + st1
        quad_total_2 = sg2 + st2
        base_0 = quad_gauss_0 + se0
        base_1 = sg1 + se1
        base_2 = sg2 + se2
        cost = quad_total_0
        grad = quad_total_1
        hess = 2 * quad_total_2
        if hess <= 0.0:
            hess = rigid_global_info.EPS[None]
        # Pack 9 broadcast slots: cost, grad, hess, base_0, base_1, base_2,
        # plus quad_gauss_0 unused but pad-friendly.
        init_bcast[0] = cost
        init_bcast[1] = grad
        init_bcast[2] = hess
        init_bcast[3] = base_0
        init_bcast[4] = base_1
        init_bcast[5] = base_2
        # Slots 6-8 reserved for future use.
    qd.simt.block.sync()

    return (
        gs.qd_float(0.0),  # p0_alpha
        init_bcast[0],     # p0_cost
        init_bcast[1],     # p0_deriv_0 (grad)
        init_bcast[2],     # p0_deriv_1 (hess)
        init_bcast[3],     # base_0
        init_bcast[4],     # base_1
        init_bcast[5],     # base_2
        gs.qd_int(1),      # ls_it
    )


@qd.func
def _func_snorm_wc(
    i_b,
    tid,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
):
    """Cooperative computation of snorm = sqrt(sum_d search[d]^2) and gtol.

    Returns (snorm, gtol) broadcast to all 64 lanes.
    """
    BLOCK_DIM = qd.static(64)
    sn_red = qd.simt.block.SharedArray((BLOCK_DIM,), gs.qd_float)
    sn_bcast = qd.simt.block.SharedArray((2,), gs.qd_float)

    n_dofs = constraint_state.search.shape[0]
    my_p = gs.qd_float(0.0)
    i_d = tid
    while i_d < n_dofs:
        s = constraint_state.search[i_d, i_b]
        my_p = my_p + s * s
        i_d = i_d + BLOCK_DIM
    sn_red[tid] = my_p
    qd.simt.block.sync()
    if tid == 0:
        total = gs.qd_float(0.0)
        for k in qd.static(range(BLOCK_DIM)):
            total = total + sn_red[k]
        snorm = qd.sqrt(total)
        scale = rigid_global_info.meaninertia[i_b] * qd.max(1, n_dofs)
        gtol = rigid_global_info.tolerance[None] * rigid_global_info.ls_tolerance[None] * snorm * scale
        sn_bcast[0] = snorm
        sn_bcast[1] = gtol
    qd.simt.block.sync()
    return sn_bcast[0], sn_bcast[1]


@qd.func
def func_linesearch_batch_wavecoop(
    i_b,
    tid,
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """Wave-cooperative linesearch.

    Equivalent to solver.func_linesearch_batch_global but with all
    `n_dofs` and `n_con` reductions parallelized across the 64 lanes
    of one wavefront. Bracketing / 3-alpha refinement control flow
    runs redundantly on every lane (all lanes have identical broadcast
    inputs and so compute identical state), avoiding extra broadcasts.

    All 64 lanes return the same `res_alpha`.
    """
    snorm, gtol = _func_snorm_wc(i_b, tid, constraint_state, rigid_global_info)
    ls_it = gs.qd_int(0)
    ls_result = gs.qd_int(0)
    res_alpha = gs.qd_float(0.0)
    done = False

    if snorm < rigid_global_info.EPS[None]:
        ls_result = 1
        res_alpha = gs.qd_float(0.0)
        done = True

    if not done:
        # Phase 1: Init + p0 + p1
        p0_alpha, p0_cost, p0_deriv_0, p0_deriv_1, base_0, base_1, base_2, ls_it = _func_ls_init_p0_wc(
            i_b,
            tid,
            entities_info=entities_info,
            dofs_state=dofs_state,
            constraint_state=constraint_state,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )
        p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1, ls_it = _func_ls_pt_opt_wc(
            i_b,
            tid,
            p0_alpha - p0_deriv_0 / p0_deriv_1,
            base_0,
            base_1,
            base_2,
            ls_it,
            constraint_state,
            rigid_global_info,
        )

        if p0_cost < p1_cost:
            p1_alpha = p0_alpha
            p1_cost = p0_cost
            p1_deriv_0 = p0_deriv_0
            p1_deriv_1 = p0_deriv_1

        if qd.abs(p1_deriv_0) < gtol:
            if qd.abs(p1_alpha) < rigid_global_info.EPS[None]:
                ls_result = 2
            else:
                ls_result = 0
            res_alpha = p1_alpha
            done = True
        else:
            # Phase 2: Bracketing
            direction = (p1_deriv_0 < 0) * 2 - 1
            p2update = 0
            p2_alpha = p1_alpha
            p2_cost = p1_cost
            p2_deriv_0 = p1_deriv_0
            p2_deriv_1 = p1_deriv_1
            phase2_break = False
            while p1_deriv_0 * direction <= -gtol and ls_it < rigid_global_info.ls_iterations[None]:
                p2_alpha = p1_alpha
                p2_cost = p1_cost
                p2_deriv_0 = p1_deriv_0
                p2_deriv_1 = p1_deriv_1
                p2update = 1

                p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1, ls_it = _func_ls_pt_opt_wc(
                    i_b,
                    tid,
                    p1_alpha - p1_deriv_0 / p1_deriv_1,
                    base_0,
                    base_1,
                    base_2,
                    ls_it,
                    constraint_state,
                    rigid_global_info,
                )
                if qd.abs(p1_deriv_0) < gtol:
                    res_alpha = p1_alpha
                    done = True
                    phase2_break = True
                    break

            if not phase2_break:
                if ls_it >= rigid_global_info.ls_iterations[None]:
                    ls_result = 3
                    res_alpha = p1_alpha
                    done = True

                if not p2update and not done:
                    ls_result = 6
                    res_alpha = p1_alpha
                    done = True

                if not done:
                    # Phase 3: 3-alpha refinement.
                    alpha_0 = p1_alpha - p1_deriv_0 / p1_deriv_1
                    alpha_1 = p1_alpha
                    alpha_2 = (p1_alpha + p2_alpha) * 0.5
                    p3_done = False
                    while ls_it < rigid_global_info.ls_iterations[None]:
                        c0, g0, h0, c1, g1, h1, c2, g2, h2, ls_it = _func_ls_pt_3a_wc(
                            i_b,
                            tid,
                            alpha_0,
                            alpha_1,
                            alpha_2,
                            base_0,
                            base_1,
                            base_2,
                            ls_it,
                            constraint_state,
                            rigid_global_info,
                        )

                        # Inline best-alpha-with-low-grad check (mirrors
                        # the qd.Vector loop in _func_linesearch_phase3_batch).
                        best_alpha = gs.qd_float(0.0)
                        best_cost = gs.qd_float(0.0)
                        best_found = False
                        if qd.abs(g0) < gtol:
                            best_alpha = alpha_0
                            best_cost = c0
                            best_found = True
                        if qd.abs(g1) < gtol and (not best_found or c1 < best_cost):
                            best_alpha = alpha_1
                            best_cost = c1
                            best_found = True
                        if qd.abs(g2) < gtol and (not best_found or c2 < best_cost):
                            best_alpha = alpha_2
                            best_cost = c2
                            best_found = True

                        if best_found:
                            res_alpha = best_alpha
                            done = True
                            p3_done = True
                            break

                        # Inline update_bracket_no_eval_local for p1 and p2.
                        # p1 update
                        b1 = 0
                        p1_next_alpha = alpha_0
                        if p1_deriv_0 < 0 and g0 < 0 and p1_deriv_0 < g0:
                            p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = alpha_0, c0, g0, h0
                            b1 = 1
                        elif p1_deriv_0 > 0 and g0 > 0 and p1_deriv_0 > g0:
                            p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = alpha_0, c0, g0, h0
                            b1 = 2
                        if p1_deriv_0 < 0 and g1 < 0 and p1_deriv_0 < g1:
                            p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = alpha_1, c1, g1, h1
                            b1 = 1
                        elif p1_deriv_0 > 0 and g1 > 0 and p1_deriv_0 > g1:
                            p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = alpha_1, c1, g1, h1
                            b1 = 2
                        if p1_deriv_0 < 0 and g2 < 0 and p1_deriv_0 < g2:
                            p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = alpha_2, c2, g2, h2
                            b1 = 1
                        elif p1_deriv_0 > 0 and g2 > 0 and p1_deriv_0 > g2:
                            p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = alpha_2, c2, g2, h2
                            b1 = 2
                        if b1 > 0:
                            p1_next_alpha = p1_alpha - p1_deriv_0 / p1_deriv_1

                        # p2 update
                        b2 = 0
                        p2_next_alpha = alpha_1
                        if p2_deriv_0 < 0 and g0 < 0 and p2_deriv_0 < g0:
                            p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1 = alpha_0, c0, g0, h0
                            b2 = 1
                        elif p2_deriv_0 > 0 and g0 > 0 and p2_deriv_0 > g0:
                            p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1 = alpha_0, c0, g0, h0
                            b2 = 2
                        if p2_deriv_0 < 0 and g1 < 0 and p2_deriv_0 < g1:
                            p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1 = alpha_1, c1, g1, h1
                            b2 = 1
                        elif p2_deriv_0 > 0 and g1 > 0 and p2_deriv_0 > g1:
                            p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1 = alpha_1, c1, g1, h1
                            b2 = 2
                        if p2_deriv_0 < 0 and g2 < 0 and p2_deriv_0 < g2:
                            p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1 = alpha_2, c2, g2, h2
                            b2 = 1
                        elif p2_deriv_0 > 0 and g2 > 0 and p2_deriv_0 > g2:
                            p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1 = alpha_2, c2, g2, h2
                            b2 = 2
                        if b2 > 0:
                            p2_next_alpha = p2_alpha - p2_deriv_0 / p2_deriv_1

                        if b1 == 0 and b2 == 0:
                            if c2 < p0_cost:
                                ls_result = 0
                            else:
                                ls_result = 7
                            res_alpha = alpha_2
                            done = True
                            p3_done = True
                            break

                        alpha_0 = p1_next_alpha
                        alpha_1 = p2_next_alpha
                        alpha_2 = (p1_alpha + p2_alpha) * 0.5

                    if not p3_done:
                        if p1_cost <= p2_cost and p1_cost < p0_cost:
                            ls_result = 4
                            res_alpha = p1_alpha
                        elif p2_cost <= p1_cost and p2_cost < p0_cost:
                            ls_result = 4
                            res_alpha = p2_alpha
                        else:
                            ls_result = 5
                            res_alpha = gs.qd_float(0.0)
    return res_alpha


@qd.kernel(
    fastcache=gs.use_fastcache,
    # Same occupancy hint as func_solve_init / monolith ("2,4"): avoid the
    # over-aggressive (8,10) target which forces VGPR count below ~64 and
    # causes scratch spilling (-> 2x scratch vs monolith, with each spill
    # access going through HBM rather than the register file).
)
def _kernel_solve_body_wavecoop_amdgpu(
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Wave-cooperative monolith: iteration loop is *inside* the kernel
    (mirrors solver.func_solve_body_monolith) so we pay 1 launch/step,
    not 15.  One wavefront (block_dim=64) per env; the 64 lanes cooperate
    on the same env's CG iter."""
    BLOCK_DIM = qd.static(64)
    N_DOFS = qd.static(static_rigid_sim_config.n_dofs_)
    _B = static_rigid_sim_config.n_envs

    qd.loop_config(
        serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL,
        block_dim=BLOCK_DIM,
    )
    for i in range(_B * BLOCK_DIM):
        tid = i % BLOCK_DIM
        i_b = i // BLOCK_DIM
        if i_b >= _B:
            continue

        # ----- shared LDS scratch (declared once per WG, before any branch) -----
        # `bcast` carries prev_cost from lane 0 to all lanes for the
        # apply-alpha phase; `cost_red` / `gauss_red` are the LDS
        # partials for the update-constraint cost/gauss reductions.
        # The wave-coop linesearch (`func_linesearch_batch_wavecoop`)
        # allocates its own LDS internally and returns alpha broadcast
        # to all 64 lanes.
        bcast = qd.simt.block.SharedArray((4,), gs.qd_float)
        cost_red = qd.simt.block.SharedArray((BLOCK_DIM,), gs.qd_float)
        gauss_red = qd.simt.block.SharedArray((BLOCK_DIM,), gs.qd_float)
        # Working vector for the wave-cooperative LDL^T mass solve (Phase 5).
        # Staged in LDS so the two triangular sweeps avoid per-step HBM
        # round-trips; sized to the static padded dof count.
        msolve = qd.simt.block.SharedArray((N_DOFS,), gs.qd_float)

        # ----- gate (matches monolith): WG-uniform on i_b. -----
        if constraint_state.n_constraints[i_b] == 0:
            if tid == 0:
                constraint_state.improved[i_b] = False
            continue

        # ===== CG iteration loop -- inside the kernel. =====
        for _it in range(rigid_global_info.iterations[None]):
            n_con = constraint_state.n_constraints[i_b]
            if not constraint_state.improved[i_b]:
                break

            # ===== Phase 1: linesearch (WAVE-COOPERATIVE) =====
            # All 64 lanes participate in the snorm, mv/jv matvecs,
            # quad reductions, point-fn reductions, and 3-alpha
            # refinement reductions. The bracketing scalar logic runs
            # redundantly on every lane (all lanes see identical
            # broadcast inputs => identical state), avoiding extra
            # LDS broadcast traffic for control flow.
            #
            # Returned `alpha_val` is identical on all 64 lanes -- no
            # broadcast needed before the apply-alpha phase below.
            alpha_val = func_linesearch_batch_wavecoop(
                i_b,
                tid,
                entities_info=entities_info,
                dofs_state=dofs_state,
                rigid_global_info=rigid_global_info,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )

            # prev_cost lives in constraint_state.cost; lane 0 captures
            # it before the apply-alpha phase (workgroup-uniform read).
            if tid == 0:
                bcast[1] = constraint_state.cost[i_b]
            qd.simt.block.sync()

            if qd.abs(alpha_val) < rigid_global_info.EPS[None]:
                if tid == 0:
                    constraint_state.improved[i_b] = False
                # `break` rather than `continue`: alpha~0 means linesearch
                # couldn't improve, mirroring monolith's `if not improved: break`.
                break

            alpha = alpha_val
            prev_cost = bcast[1]

            # ===== Phase 2a: apply alpha to qacc, Ma (parallel per-dof) =====
            i_d = tid
            while i_d < N_DOFS:
                constraint_state.qacc[i_d, i_b] = (
                    constraint_state.qacc[i_d, i_b] + constraint_state.search[i_d, i_b] * alpha
                )
                constraint_state.Ma[i_d, i_b] = (
                    constraint_state.Ma[i_d, i_b] + constraint_state.mv[i_d, i_b] * alpha
                )
                i_d = i_d + BLOCK_DIM

            # ===== Phase 2b: apply alpha to Jaref (parallel per-constraint) =====
            i_c = tid
            while i_c < n_con:
                constraint_state.Jaref[i_c, i_b] = (
                    constraint_state.Jaref[i_c, i_b] + constraint_state.jv[i_c, i_b] * alpha
                )
                i_c = i_c + BLOCK_DIM
            qd.simt.block.sync()

            # ===== Phase 3: save prev grad (CG only, parallel per-dof) =====
            if qd.static(static_rigid_sim_config.solver_type == gs.constraint_solver.CG):
                i_d = tid
                while i_d < N_DOFS:
                    constraint_state.cg_prev_grad[i_d, i_b] = constraint_state.grad[i_d, i_b]
                    constraint_state.cg_prev_Mgrad[i_d, i_b] = constraint_state.Mgrad[i_d, i_b]
                    i_d = i_d + BLOCK_DIM
                qd.simt.block.sync()

            # ===== Phase 4: update_constraint_batch (parallel) =====
            # Mirrors solver.func_update_constraint_batch but distributes the
            # four sub-loops across the 64 lanes of the wave. Cross-lane
            # partial sums of cost/gauss are reduced in LDS at the end.
            if tid == 0:
                constraint_state.prev_cost[i_b] = prev_cost
            ne = constraint_state.n_constraints_equality[i_b]
            nef = ne + constraint_state.n_constraints_frictionloss[i_b]

            my_cost_partial = gs.qd_float(0.0)
            my_gauss_partial = gs.qd_float(0.0)

            # 4a: per-constraint active flag + efc_force update.
            #     Friction-cost contribution to cost_partial happens here.
            i_c = tid
            while i_c < n_con:
                Jaref_c = constraint_state.Jaref[i_c, i_b]
                efc_D_c = constraint_state.efc_D[i_c, i_b]

                if qd.static(static_rigid_sim_config.solver_type == gs.constraint_solver.Newton):
                    constraint_state.prev_active[i_c, i_b] = constraint_state.active[i_c, i_b]

                active_c = True
                floss_force = gs.qd_float(0.0)
                floss_cost_local = gs.qd_float(0.0)

                if ne <= i_c and i_c < nef:  # Friction
                    f = constraint_state.efc_frictionloss[i_c, i_b]
                    r = constraint_state.diag[i_c, i_b]
                    rf = r * f
                    linear_neg = Jaref_c <= -rf
                    linear_pos = Jaref_c >= rf
                    active_c = not (linear_neg or linear_pos)
                    floss_force = linear_neg * f + linear_pos * -f
                    floss_cost_local = linear_neg * f * (-0.5 * rf - Jaref_c) + linear_pos * f * (
                        -0.5 * rf + Jaref_c
                    )
                elif nef <= i_c:  # Contact
                    active_c = Jaref_c < 0

                constraint_state.active[i_c, i_b] = active_c
                constraint_state.efc_force[i_c, i_b] = floss_force + (-Jaref_c * efc_D_c * active_c)

                # Constraint-cost contribution (loop 4d in baseline) fused here:
                #   cost += floss_cost_local + 0.5 * Jaref^2 * D * active
                my_cost_partial = (
                    my_cost_partial + floss_cost_local + 0.5 * Jaref_c * Jaref_c * efc_D_c * active_c
                )
                i_c = i_c + BLOCK_DIM
            qd.simt.block.sync()  # ensure efc_force writes visible before J^T@e reads

            # 4b: per-dof qfrc_constraint = J^T @ efc_force (DENSE only).
            # Each lane handles a stride of i_d; inner sum stays serial per lane.
            i_d = tid
            while i_d < N_DOFS:
                qfrc = gs.qd_float(0.0)
                for j_c in range(n_con):
                    qfrc = qfrc + constraint_state.jac[j_c, i_d, i_b] * constraint_state.efc_force[j_c, i_b]
                constraint_state.qfrc_constraint[i_d, i_b] = qfrc
                i_d = i_d + BLOCK_DIM

            # 4c: per-dof gauss / cost contribution from (Mx-Mx')*(x-x').
            i_d = tid
            while i_d < N_DOFS:
                v = (
                    0.5
                    * (constraint_state.Ma[i_d, i_b] - dofs_state.force[i_d, i_b])
                    * (constraint_state.qacc[i_d, i_b] - dofs_state.acc_smooth[i_d, i_b])
                )
                my_cost_partial = my_cost_partial + v
                my_gauss_partial = my_gauss_partial + v
                i_d = i_d + BLOCK_DIM

            # 4d: LDS reduce my_cost_partial / my_gauss_partial -> totals.
            cost_red[tid] = my_cost_partial
            gauss_red[tid] = my_gauss_partial
            qd.simt.block.sync()
            if tid == 0:
                total_cost = gs.qd_float(0.0)
                total_gauss = gs.qd_float(0.0)
                for k in qd.static(range(BLOCK_DIM)):
                    total_cost = total_cost + cost_red[k]
                    total_gauss = total_gauss + gauss_red[k]
                constraint_state.cost[i_b] = total_cost
                constraint_state.gauss[i_b] = total_gauss
            qd.simt.block.sync()

            # ===== Phase 5: update gradient (WAVE-COOPERATIVE) =====
            # Mirrors solver.func_update_gradient_batch but spreads the work
            # across all 64 lanes instead of running on lane 0 alone:
            #   5a: grad = Ma - force - qfrc_constraint        (parallel per-dof)
            #   5b: Mgrad = M^{-1} grad via a wave-cooperative LDL^T solve
            #       (CG only; Newton is excluded by _wavecoop_amdgpu_is_compatible)
            # 5a:
            i_d = tid
            while i_d < N_DOFS:
                constraint_state.grad[i_d, i_b] = (
                    constraint_state.Ma[i_d, i_b]
                    - dofs_state.force[i_d, i_b]
                    - constraint_state.qfrc_constraint[i_d, i_b]
                )
                i_d = i_d + BLOCK_DIM
            qd.simt.block.sync()

            # 5b: cooperative mass solve, per entity (block-diagonal M).
            # Column-sweep triangular solves: the critical path is O(n_dofs)
            # sequential steps (vs O(n_dofs^2) serial on lane 0), each step
            # fanning the just-finalized component out across the lanes.
            # Intra-wavefront syncs (block_dim=64 == one wave64) are cheap.
            if qd.static(static_rigid_sim_config.solver_type == gs.constraint_solver.CG):
                for i_e in range(qd.static(static_rigid_sim_config.n_entities_)):
                    if rigid_global_info.mass_mat_mask[i_e, i_b]:
                        e_ds = entities_info.dof_start[i_e]
                        e_de = entities_info.dof_end[i_e]
                        e_n = e_de - e_ds

                        # load y -> LDS
                        i_d = e_ds + tid
                        while i_d < e_de:
                            msolve[i_d] = constraint_state.grad[i_d, i_b]
                            i_d = i_d + BLOCK_DIM
                        qd.simt.block.sync()

                        # Step 1: solve L^T w = y (back-substitution, p high->low)
                        for pp in range(e_n):
                            p = e_de - 1 - pp
                            wp = msolve[p]
                            k = e_ds + tid
                            while k < p:
                                msolve[k] = msolve[k] - rigid_global_info.mass_mat_L[p, k, i_b] * wp
                                k = k + BLOCK_DIM
                            qd.simt.block.sync()

                        # Step 2: z = D^{-1} w (parallel per-dof)
                        i_d = e_ds + tid
                        while i_d < e_de:
                            msolve[i_d] = msolve[i_d] * rigid_global_info.mass_mat_D_inv[i_d, i_b]
                            i_d = i_d + BLOCK_DIM
                        qd.simt.block.sync()

                        # Step 3: solve L x = z (forward-substitution, p low->high)
                        for pp in range(e_n):
                            p = e_ds + pp
                            wp = msolve[p]
                            k = p + 1 + tid
                            while k < e_de:
                                msolve[k] = msolve[k] - rigid_global_info.mass_mat_L[k, p, i_b] * wp
                                k = k + BLOCK_DIM
                            qd.simt.block.sync()

                        # store x -> Mgrad
                        i_d = e_ds + tid
                        while i_d < e_de:
                            constraint_state.Mgrad[i_d, i_b] = msolve[i_d]
                            i_d = i_d + BLOCK_DIM
                        qd.simt.block.sync()

            # ===== Phase 6: terminate or update descent (WAVE-COOPERATIVE) =====
            # Mirrors solver.func_terminate_or_update_descent_batch; the
            # grad-norm and CG-beta dot products are lane-strided partials
            # reduced in LDS, the search update is embarrassingly parallel.
            my_gn = gs.qd_float(0.0)
            i_d = tid
            while i_d < N_DOFS:
                g_d = constraint_state.grad[i_d, i_b]
                my_gn = my_gn + g_d * g_d
                i_d = i_d + BLOCK_DIM
            cost_red[tid] = my_gn
            qd.simt.block.sync()
            if tid == 0:
                gn_sq = gs.qd_float(0.0)
                for k in qd.static(range(BLOCK_DIM)):
                    gn_sq = gn_sq + cost_red[k]
                bcast[2] = gn_sq
            qd.simt.block.sync()

            grad_norm = qd.sqrt(bcast[2])
            tol_scaled = (
                rigid_global_info.meaninertia[i_b] * qd.max(1, N_DOFS)
            ) * rigid_global_info.tolerance[None]
            improvement = prev_cost - constraint_state.cost[i_b]
            improved6 = (grad_norm > tol_scaled) and (improvement > tol_scaled)
            if tid == 0:
                constraint_state.improved[i_b] = improved6

            if improved6:
                # CG beta (two dot products), lane-strided + LDS reduce.
                my_beta = gs.qd_float(0.0)
                my_pgpm = gs.qd_float(0.0)
                i_d = tid
                while i_d < N_DOFS:
                    grad_d = constraint_state.grad[i_d, i_b]
                    Mgrad_d = constraint_state.Mgrad[i_d, i_b]
                    pMgrad_d = constraint_state.cg_prev_Mgrad[i_d, i_b]
                    pgrad_d = constraint_state.cg_prev_grad[i_d, i_b]
                    my_beta = my_beta + grad_d * (Mgrad_d - pMgrad_d)
                    my_pgpm = my_pgpm + pMgrad_d * pgrad_d
                    i_d = i_d + BLOCK_DIM
                cost_red[tid] = my_beta
                gauss_red[tid] = my_pgpm
                qd.simt.block.sync()
                if tid == 0:
                    t_beta = gs.qd_float(0.0)
                    t_pgpm = gs.qd_float(0.0)
                    for k in qd.static(range(BLOCK_DIM)):
                        t_beta = t_beta + cost_red[k]
                        t_pgpm = t_pgpm + gauss_red[k]
                    cg_beta = qd.max(t_beta / qd.max(rigid_global_info.EPS[None], t_pgpm), 0.0)
                    constraint_state.cg_pg_dot_pMg[i_b] = t_pgpm
                    constraint_state.cg_beta[i_b] = cg_beta
                    bcast[3] = cg_beta
                qd.simt.block.sync()

                cg_beta_b = bcast[3]
                i_d = tid
                while i_d < N_DOFS:
                    constraint_state.search[i_d, i_b] = (
                        -constraint_state.Mgrad[i_d, i_b] + cg_beta_b * constraint_state.search[i_d, i_b]
                    )
                    i_d = i_d + BLOCK_DIM
            qd.simt.block.sync()


def _wavecoop_amdgpu_is_compatible(*args, **kwargs):
    # Wave-cooperative monolith variant. Eligible when:
    #   - backend = AMDGPU
    #   - dense Jacobian (sparse_solve=False)
    #   - CG solver (Newton needs in-iter Hessian/Cholesky update)
    #   - batched workload (cfg.n_envs >= 8): the wave-coop reduction
    #     order differs from the monolith's serial accumulation by
    #     ULP-level FP drift, which is benign for batched benchmarks but
    #     compounds into multi-step trajectory divergence on small/
    #     unbatched workloads (n_envs=0 maps to cfg.n_envs=1 internally)
    #     that unit tests assert against tight tolerances. The threshold
    #     matches the tiled-wc tile size for consistency.
    if gs.backend not in {gs.amdgpu}:
        return False
    cfg = kwargs.get("static_rigid_sim_config", args[4] if len(args) >= 5 else None)
    if cfg is None:
        return False
    if cfg.sparse_solve:
        return False
    if cfg.solver_type != gs.constraint_solver.CG:
        return False
    if cfg.n_envs < 8:
        return False
    return True


@solver.func_solve_body.register(is_compatible=_wavecoop_amdgpu_is_compatible)
def func_solve_body_wavecoop_amdgpu(
    entities_info,
    dofs_state,
    constraint_state,
    rigid_global_info,
    static_rigid_sim_config,
    _n_iterations,
):
    # Iteration loop is *inside* the kernel (mirrors monolith), so we
    # launch once per step rather than once per CG iter. _n_iterations
    # is unused here -- the kernel reads rigid_global_info.iterations[None].
    _kernel_solve_body_wavecoop_amdgpu(
        entities_info,
        dofs_state,
        constraint_state,
        rigid_global_info,
        static_rigid_sim_config,
    )


@solver.func_solve_body.register(is_compatible=_decomposed_amdgpu_is_compatible)
def func_solve_body_decomposed_amdgpu(
    entities_info,
    dofs_state,
    constraint_state,
    rigid_global_info,
    static_rigid_sim_config,
    _n_iterations,
):
    for _it in range(_n_iterations):
        _kernel_linesearch_amdgpu_decomposed(
            entities_info,
            dofs_state,
            constraint_state,
            rigid_global_info,
            static_rigid_sim_config,
        )
        _kernel_cg_save_prev_grad_amdgpu_decomposed(
            constraint_state,
            static_rigid_sim_config,
        )
        _kernel_update_constraint_forces_amdgpu_decomposed(
            constraint_state,
            static_rigid_sim_config,
        )
        _kernel_update_constraint_qfrc_amdgpu_decomposed(
            constraint_state,
            static_rigid_sim_config,
        )
        _kernel_update_constraint_cost_amdgpu_decomposed(
            dofs_state,
            constraint_state,
            static_rigid_sim_config,
        )
        _kernel_update_gradient_amdgpu_decomposed(
            entities_info,
            dofs_state,
            constraint_state,
            rigid_global_info,
            static_rigid_sim_config,
        )
        _kernel_update_search_direction_amdgpu_decomposed(
            constraint_state,
            rigid_global_info,
            static_rigid_sim_config,
        )


# ---------------------------------------------------------------------------
# Tiled wave-cooperative monolith (8 envs x 8 lanes/env per wave64)
# ---------------------------------------------------------------------------
#
# The full wave-coop variant above is structurally fast on robots with
# large n_dofs but pays a heavy lane-utilization cost on small-n_dofs
# robots: the per-DOF "tid==0" phases (mass solve, gradient norm,
# terminate-or-update-descent) idle 63 of 64 lanes per wavefront.
#
# Tiled wave-coop sits between the monolith (1 env per thread, 100% lane
# use on every phase) and the full wave-coop (1 env per wavefront, all
# 64 lanes cooperating but only 1 lane active during the per-env scalar
# phases): it packs TWC_ENVS_PER_BLOCK=8 envs into one wavefront with
# TWC_COOP_FACTOR=8 lanes cooperating per env. The 8 envs share the
# wavefront's LDS in 8-slot stripes; the per-env scalar phases now run
# on 8 lanes in parallel (one lane per env) instead of 1.
#
#   |                       | lane util on per-env tid==0 work          |
#   |-----------------------|-------------------------------------------|
#   | Monolith (1 env/lane) | 100 %                                     |
#   | Wave-coop (64x1)      |   1.5 % (1 of 64 lanes)                   |
#   | Tiled (8x8)           |  12.5 % (8 of 64 lanes)                   |
#
# Lane partition inside one workgroup of BLOCK_DIM=64 threads:
#   tid -> (env_in_block, lane_in_env) = (tid // 8, tid % 8)
#   i_b = block_id * TWC_ENVS_PER_BLOCK + env_in_block
# All cooperative reductions (snorm, mv/jv, cost/gauss, linesearch sums)
# run in parallel for the 8 envs, each using its own 8-slot stripe of
# LDS. Per-env "tid==0" work runs on 8 lanes simultaneously (one per
# env in the block).
#
# Block-uniform sync constraint: per-env `continue` / `break` is
# forbidden because different envs in the same workgroup may have
# different convergence states, and divergent control flow across the
# workgroup deadlocks the per-iter `qd.simt.block.sync()` calls.
# Instead, each phase runs unconditionally for all 8 envs; the *write*
# sites are predicated on a per-env `is_active_env` flag. The CG
# iteration loop breaks only when all 8 envs in the block have
# converged (block-wide OR-reduce of the `is_active_env` flags).
#
# Compatibility (`_tiled_wc_amdgpu_is_compatible`):
#   - AMDGPU backend (wave64)
#   - dense Jacobian (sparse_solve=False)
#   - CG solver (Newton needs in-iter Hessian/Cholesky update which is
#     only inlined in the monolith / split / lifted-loop variants)
#   - n_envs % TWC_ENVS_PER_BLOCK == 0 (so the OOB tail-block gate
#     never fires inside a workgroup sync, which would otherwise
#     deadlock the lanes that took the gate path)


_TWC_BLOCK_DIM = 64
_TWC_COOP_FACTOR = 8
_TWC_ENVS_PER_BLOCK = 8


@qd.func
def _func_snorm_twc(
    i_b,
    tid,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
):
    """Tiled wave-coop snorm: 8 envs reduce in parallel, each using an
    8-slot LDS stripe. Returns (snorm, gtol) broadcast to the 8 lanes
    of the env's lane group.
    """
    BLOCK_DIM = qd.static(_TWC_BLOCK_DIM)
    COOP = qd.static(_TWC_COOP_FACTOR)
    ENVS = qd.static(_TWC_ENVS_PER_BLOCK)
    sn_red = qd.simt.block.SharedArray((BLOCK_DIM,), gs.qd_float)
    sn_bcast = qd.simt.block.SharedArray((ENVS, 2), gs.qd_float)

    env_in_block = tid // COOP
    lane_in_env = tid % COOP

    n_dofs = constraint_state.search.shape[0]
    my_p = gs.qd_float(0.0)
    i_d = lane_in_env
    while i_d < n_dofs:
        s = constraint_state.search[i_d, i_b]
        my_p = my_p + s * s
        i_d = i_d + COOP
    sn_red[tid] = my_p
    qd.simt.block.sync()
    if lane_in_env == 0:
        total = gs.qd_float(0.0)
        base = env_in_block * COOP
        for k in qd.static(range(COOP)):
            total = total + sn_red[base + k]
        snorm = qd.sqrt(total)
        scale = rigid_global_info.meaninertia[i_b] * qd.max(1, n_dofs)
        gtol = rigid_global_info.tolerance[None] * rigid_global_info.ls_tolerance[None] * snorm * scale
        sn_bcast[env_in_block, 0] = snorm
        sn_bcast[env_in_block, 1] = gtol
    qd.simt.block.sync()
    return sn_bcast[env_in_block, 0], sn_bcast[env_in_block, 1]


@qd.func
def _func_ls_init_p0_twc(
    i_b,
    tid,
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Tiled wave-coop init+p0: each of 8 envs cooperatively computes
    mv (M*search per entity), jv (J*search per constraint), and
    quad_gauss / quad_total / eq_sum reductions. Returns p0 tuple
    broadcast to the 8 lanes of the env's group.
    """
    BLOCK_DIM = qd.static(_TWC_BLOCK_DIM)
    COOP = qd.static(_TWC_COOP_FACTOR)
    ENVS = qd.static(_TWC_ENVS_PER_BLOCK)
    init_red = qd.simt.block.SharedArray((8, BLOCK_DIM), gs.qd_float)
    init_bcast = qd.simt.block.SharedArray((ENVS, 6), gs.qd_float)

    env_in_block = tid // COOP
    lane_in_env = tid % COOP

    n_dofs = constraint_state.search.shape[0]
    n_entities = static_rigid_sim_config.n_entities_
    ne = constraint_state.n_constraints_equality[i_b]
    nef = ne + constraint_state.n_constraints_frictionloss[i_b]
    n_con = constraint_state.n_constraints[i_b]

    # 1) mv[i_d1] = sum_d2 mass_mat[d1,d2] * search[d2] per entity.
    for i_e in range(n_entities):
        d_start = entities_info.dof_start[i_e]
        d_end = entities_info.dof_end[i_e]
        i_d1 = d_start + lane_in_env
        while i_d1 < d_end:
            mv = gs.qd_float(0.0)
            for i_d2 in range(d_start, d_end):
                mv = mv + rigid_global_info.mass_mat[i_d1, i_d2, i_b] * constraint_state.search[i_d2, i_b]
            constraint_state.mv[i_d1, i_b] = mv
            i_d1 = i_d1 + COOP
    qd.simt.block.sync()

    # 2) jv[i_c] = sum_d jac[c,d] * search[d] (DENSE only).
    i_c = lane_in_env
    while i_c < n_con:
        jv = gs.qd_float(0.0)
        for i_d in range(n_dofs):
            jv = jv + constraint_state.jac[i_c, i_d, i_b] * constraint_state.search[i_d, i_b]
        constraint_state.jv[i_c, i_b] = jv
        i_c = i_c + COOP
    qd.simt.block.sync()

    # 3) quad_gauss_1, quad_gauss_2 over n_dofs (lane_in_env-strided).
    my_qg1 = gs.qd_float(0.0)
    my_qg2 = gs.qd_float(0.0)
    i_d = lane_in_env
    while i_d < n_dofs:
        s = constraint_state.search[i_d, i_b]
        Ma_d = constraint_state.Ma[i_d, i_b]
        f_d = dofs_state.force[i_d, i_b]
        mv_d = constraint_state.mv[i_d, i_b]
        my_qg1 = my_qg1 + s * Ma_d - s * f_d
        my_qg2 = my_qg2 + 0.5 * s * mv_d
        i_d = i_d + COOP

    # 4) quad_total + eq_sum over n_con (lane_in_env-strided).
    my_qt0 = gs.qd_float(0.0)
    my_qt1 = gs.qd_float(0.0)
    my_qt2 = gs.qd_float(0.0)
    my_eq0 = gs.qd_float(0.0)
    my_eq1 = gs.qd_float(0.0)
    my_eq2 = gs.qd_float(0.0)
    i_c = lane_in_env
    while i_c < n_con:
        Jaref_c = constraint_state.Jaref[i_c, i_b]
        jv_c = constraint_state.jv[i_c, i_b]
        D = constraint_state.efc_D[i_c, i_b]
        qf_0 = D * (0.5 * Jaref_c * Jaref_c)
        qf_1 = D * (jv_c * Jaref_c)
        qf_2 = D * (0.5 * jv_c * jv_c)
        if i_c < ne:
            my_eq0 = my_eq0 + qf_0
            my_eq1 = my_eq1 + qf_1
            my_eq2 = my_eq2 + qf_2
            my_qt0 = my_qt0 + qf_0
            my_qt1 = my_qt1 + qf_1
            my_qt2 = my_qt2 + qf_2
        elif i_c < nef:
            f = constraint_state.efc_frictionloss[i_c, i_b]
            r = constraint_state.diag[i_c, i_b]
            rf = r * f
            linear_neg = Jaref_c <= -rf
            linear_pos = Jaref_c >= rf
            if linear_neg or linear_pos:
                qf_0 = linear_neg * f * (-0.5 * rf - Jaref_c) + linear_pos * f * (-0.5 * rf + Jaref_c)
                qf_1 = linear_neg * (-f * jv_c) + linear_pos * (f * jv_c)
                qf_2 = 0.0
            my_qt0 = my_qt0 + qf_0
            my_qt1 = my_qt1 + qf_1
            my_qt2 = my_qt2 + qf_2
        else:
            active = Jaref_c < 0
            my_qt0 = my_qt0 + qf_0 * active
            my_qt1 = my_qt1 + qf_1 * active
            my_qt2 = my_qt2 + qf_2 * active
        i_c = i_c + COOP

    init_red[0, tid] = my_qg1
    init_red[1, tid] = my_qg2
    init_red[2, tid] = my_qt0
    init_red[3, tid] = my_qt1
    init_red[4, tid] = my_qt2
    init_red[5, tid] = my_eq0
    init_red[6, tid] = my_eq1
    init_red[7, tid] = my_eq2
    qd.simt.block.sync()
    if lane_in_env == 0:
        sg1 = gs.qd_float(0.0)
        sg2 = gs.qd_float(0.0)
        st0 = gs.qd_float(0.0)
        st1 = gs.qd_float(0.0)
        st2 = gs.qd_float(0.0)
        se0 = gs.qd_float(0.0)
        se1 = gs.qd_float(0.0)
        se2 = gs.qd_float(0.0)
        base = env_in_block * COOP
        for k in qd.static(range(COOP)):
            sg1 = sg1 + init_red[0, base + k]
            sg2 = sg2 + init_red[1, base + k]
            st0 = st0 + init_red[2, base + k]
            st1 = st1 + init_red[3, base + k]
            st2 = st2 + init_red[4, base + k]
            se0 = se0 + init_red[5, base + k]
            se1 = se1 + init_red[6, base + k]
            se2 = se2 + init_red[7, base + k]
        quad_gauss_0 = constraint_state.gauss[i_b]
        quad_total_0 = quad_gauss_0 + st0
        quad_total_1 = sg1 + st1
        quad_total_2 = sg2 + st2
        base_0 = quad_gauss_0 + se0
        base_1 = sg1 + se1
        base_2 = sg2 + se2
        cost = quad_total_0
        grad = quad_total_1
        hess = 2 * quad_total_2
        if hess <= 0.0:
            hess = rigid_global_info.EPS[None]
        init_bcast[env_in_block, 0] = cost
        init_bcast[env_in_block, 1] = grad
        init_bcast[env_in_block, 2] = hess
        init_bcast[env_in_block, 3] = base_0
        init_bcast[env_in_block, 4] = base_1
        init_bcast[env_in_block, 5] = base_2
    qd.simt.block.sync()

    return (
        gs.qd_float(0.0),
        init_bcast[env_in_block, 0],
        init_bcast[env_in_block, 1],
        init_bcast[env_in_block, 2],
        init_bcast[env_in_block, 3],
        init_bcast[env_in_block, 4],
        init_bcast[env_in_block, 5],
        gs.qd_int(1),
    )


@qd.func
def _func_ls_pt_opt_twc(
    i_b,
    tid,
    alpha,
    base_0,
    base_1,
    base_2,
    ls_it,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
):
    """Tiled wave-coop point evaluation: 8 envs reduce in parallel."""
    BLOCK_DIM = qd.static(_TWC_BLOCK_DIM)
    COOP = qd.static(_TWC_COOP_FACTOR)
    ENVS = qd.static(_TWC_ENVS_PER_BLOCK)
    pt_red = qd.simt.block.SharedArray((3, BLOCK_DIM), gs.qd_float)
    pt_bcast = qd.simt.block.SharedArray((ENVS, 3), gs.qd_float)

    env_in_block = tid // COOP
    lane_in_env = tid % COOP

    ne = constraint_state.n_constraints_equality[i_b]
    nef = ne + constraint_state.n_constraints_frictionloss[i_b]
    n_con = constraint_state.n_constraints[i_b]

    my_t0 = gs.qd_float(0.0)
    my_t1 = gs.qd_float(0.0)
    my_t2 = gs.qd_float(0.0)

    # Friction [ne, nef).
    i_c = ne + lane_in_env
    while i_c < nef:
        Jaref_c = constraint_state.Jaref[i_c, i_b]
        jv_c = constraint_state.jv[i_c, i_b]
        D = constraint_state.efc_D[i_c, i_b]
        f = constraint_state.efc_frictionloss[i_c, i_b]
        r = constraint_state.diag[i_c, i_b]
        qf_0 = D * (0.5 * Jaref_c * Jaref_c)
        qf_1 = D * (jv_c * Jaref_c)
        qf_2 = D * (0.5 * jv_c * jv_c)
        x = Jaref_c + alpha * jv_c
        rf = r * f
        linear_neg = x <= -rf
        linear_pos = x >= rf
        if linear_neg or linear_pos:
            qf_0 = linear_neg * f * (-0.5 * rf - Jaref_c) + linear_pos * f * (-0.5 * rf + Jaref_c)
            qf_1 = linear_neg * (-f * jv_c) + linear_pos * (f * jv_c)
            qf_2 = 0.0
        my_t0 = my_t0 + qf_0
        my_t1 = my_t1 + qf_1
        my_t2 = my_t2 + qf_2
        i_c = i_c + COOP

    # Contact [nef, n_con).
    i_c = nef + lane_in_env
    while i_c < n_con:
        Jaref_c = constraint_state.Jaref[i_c, i_b]
        jv_c = constraint_state.jv[i_c, i_b]
        D = constraint_state.efc_D[i_c, i_b]
        x = Jaref_c + alpha * jv_c
        active = x < 0
        qf_0 = D * (0.5 * Jaref_c * Jaref_c)
        qf_1 = D * (jv_c * Jaref_c)
        qf_2 = D * (0.5 * jv_c * jv_c)
        my_t0 = my_t0 + qf_0 * active
        my_t1 = my_t1 + qf_1 * active
        my_t2 = my_t2 + qf_2 * active
        i_c = i_c + COOP

    pt_red[0, tid] = my_t0
    pt_red[1, tid] = my_t1
    pt_red[2, tid] = my_t2
    qd.simt.block.sync()
    if lane_in_env == 0:
        s0 = base_0
        s1 = base_1
        s2 = base_2
        base = env_in_block * COOP
        for k in qd.static(range(COOP)):
            s0 = s0 + pt_red[0, base + k]
            s1 = s1 + pt_red[1, base + k]
            s2 = s2 + pt_red[2, base + k]
        cost = alpha * alpha * s2 + alpha * s1 + s0
        grad = 2 * alpha * s2 + s1
        hess = 2 * s2
        if hess <= 0.0:
            hess = rigid_global_info.EPS[None]
        pt_bcast[env_in_block, 0] = cost
        pt_bcast[env_in_block, 1] = grad
        pt_bcast[env_in_block, 2] = hess
    qd.simt.block.sync()
    return alpha, pt_bcast[env_in_block, 0], pt_bcast[env_in_block, 1], pt_bcast[env_in_block, 2], ls_it + 1


@qd.func
def _func_ls_pt_3a_twc(
    i_b,
    tid,
    alpha_0,
    alpha_1,
    alpha_2,
    base_0,
    base_1,
    base_2,
    ls_it,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
):
    """Tiled wave-coop 3-alpha refinement reduction."""
    BLOCK_DIM = qd.static(_TWC_BLOCK_DIM)
    COOP = qd.static(_TWC_COOP_FACTOR)
    ENVS = qd.static(_TWC_ENVS_PER_BLOCK)
    pt3_red = qd.simt.block.SharedArray((9, BLOCK_DIM), gs.qd_float)
    pt3_bcast = qd.simt.block.SharedArray((ENVS, 9), gs.qd_float)

    env_in_block = tid // COOP
    lane_in_env = tid % COOP

    ne = constraint_state.n_constraints_equality[i_b]
    nef = ne + constraint_state.n_constraints_frictionloss[i_b]
    n_con = constraint_state.n_constraints[i_b]

    t0_0 = gs.qd_float(0.0)
    t0_1 = gs.qd_float(0.0)
    t0_2 = gs.qd_float(0.0)
    t1_0 = gs.qd_float(0.0)
    t1_1 = gs.qd_float(0.0)
    t1_2 = gs.qd_float(0.0)
    t2_0 = gs.qd_float(0.0)
    t2_1 = gs.qd_float(0.0)
    t2_2 = gs.qd_float(0.0)

    # Friction [ne, nef).
    i_c = ne + lane_in_env
    while i_c < nef:
        Jaref_c = constraint_state.Jaref[i_c, i_b]
        jv_c = constraint_state.jv[i_c, i_b]
        D = constraint_state.efc_D[i_c, i_b]
        f = constraint_state.efc_frictionloss[i_c, i_b]
        r = constraint_state.diag[i_c, i_b]
        qf_0 = D * (0.5 * Jaref_c * Jaref_c)
        qf_1 = D * (jv_c * Jaref_c)
        qf_2 = D * (0.5 * jv_c * jv_c)
        rf = r * f

        x0 = Jaref_c + alpha_0 * jv_c
        ln0 = x0 <= -rf
        lp0 = x0 >= rf
        a0_qf_0, a0_qf_1, a0_qf_2 = qf_0, qf_1, qf_2
        if ln0 or lp0:
            a0_qf_0 = ln0 * f * (-0.5 * rf - Jaref_c) + lp0 * f * (-0.5 * rf + Jaref_c)
            a0_qf_1 = ln0 * (-f * jv_c) + lp0 * (f * jv_c)
            a0_qf_2 = 0.0
        t0_0 = t0_0 + a0_qf_0
        t0_1 = t0_1 + a0_qf_1
        t0_2 = t0_2 + a0_qf_2

        x1 = Jaref_c + alpha_1 * jv_c
        ln1 = x1 <= -rf
        lp1 = x1 >= rf
        a1_qf_0, a1_qf_1, a1_qf_2 = qf_0, qf_1, qf_2
        if ln1 or lp1:
            a1_qf_0 = ln1 * f * (-0.5 * rf - Jaref_c) + lp1 * f * (-0.5 * rf + Jaref_c)
            a1_qf_1 = ln1 * (-f * jv_c) + lp1 * (f * jv_c)
            a1_qf_2 = 0.0
        t1_0 = t1_0 + a1_qf_0
        t1_1 = t1_1 + a1_qf_1
        t1_2 = t1_2 + a1_qf_2

        x2 = Jaref_c + alpha_2 * jv_c
        ln2 = x2 <= -rf
        lp2 = x2 >= rf
        a2_qf_0, a2_qf_1, a2_qf_2 = qf_0, qf_1, qf_2
        if ln2 or lp2:
            a2_qf_0 = ln2 * f * (-0.5 * rf - Jaref_c) + lp2 * f * (-0.5 * rf + Jaref_c)
            a2_qf_1 = ln2 * (-f * jv_c) + lp2 * (f * jv_c)
            a2_qf_2 = 0.0
        t2_0 = t2_0 + a2_qf_0
        t2_1 = t2_1 + a2_qf_1
        t2_2 = t2_2 + a2_qf_2
        i_c = i_c + COOP

    # Contact [nef, n_con).
    i_c = nef + lane_in_env
    while i_c < n_con:
        Jaref_c = constraint_state.Jaref[i_c, i_b]
        jv_c = constraint_state.jv[i_c, i_b]
        D = constraint_state.efc_D[i_c, i_b]
        qf_0 = D * (0.5 * Jaref_c * Jaref_c)
        qf_1 = D * (jv_c * Jaref_c)
        qf_2 = D * (0.5 * jv_c * jv_c)
        x0 = Jaref_c + alpha_0 * jv_c
        x1 = Jaref_c + alpha_1 * jv_c
        x2 = Jaref_c + alpha_2 * jv_c
        act0 = gs.qd_bool(x0 < 0)
        act1 = gs.qd_bool(x1 < 0)
        act2 = gs.qd_bool(x2 < 0)
        t0_0 = t0_0 + qf_0 * act0
        t0_1 = t0_1 + qf_1 * act0
        t0_2 = t0_2 + qf_2 * act0
        t1_0 = t1_0 + qf_0 * act1
        t1_1 = t1_1 + qf_1 * act1
        t1_2 = t1_2 + qf_2 * act1
        t2_0 = t2_0 + qf_0 * act2
        t2_1 = t2_1 + qf_1 * act2
        t2_2 = t2_2 + qf_2 * act2
        i_c = i_c + COOP

    pt3_red[0, tid] = t0_0
    pt3_red[1, tid] = t0_1
    pt3_red[2, tid] = t0_2
    pt3_red[3, tid] = t1_0
    pt3_red[4, tid] = t1_1
    pt3_red[5, tid] = t1_2
    pt3_red[6, tid] = t2_0
    pt3_red[7, tid] = t2_1
    pt3_red[8, tid] = t2_2
    qd.simt.block.sync()
    if lane_in_env == 0:
        s00 = base_0
        s01 = base_1
        s02 = base_2
        s10 = base_0
        s11 = base_1
        s12 = base_2
        s20 = base_0
        s21 = base_1
        s22 = base_2
        base = env_in_block * COOP
        for k in qd.static(range(COOP)):
            s00 = s00 + pt3_red[0, base + k]
            s01 = s01 + pt3_red[1, base + k]
            s02 = s02 + pt3_red[2, base + k]
            s10 = s10 + pt3_red[3, base + k]
            s11 = s11 + pt3_red[4, base + k]
            s12 = s12 + pt3_red[5, base + k]
            s20 = s20 + pt3_red[6, base + k]
            s21 = s21 + pt3_red[7, base + k]
            s22 = s22 + pt3_red[8, base + k]
        EPS = rigid_global_info.EPS[None]
        c0 = alpha_0 * alpha_0 * s02 + alpha_0 * s01 + s00
        g0 = 2 * alpha_0 * s02 + s01
        h0 = 2 * s02
        if h0 <= 0.0:
            h0 = EPS
        c1 = alpha_1 * alpha_1 * s12 + alpha_1 * s11 + s10
        g1 = 2 * alpha_1 * s12 + s11
        h1 = 2 * s12
        if h1 <= 0.0:
            h1 = EPS
        c2 = alpha_2 * alpha_2 * s22 + alpha_2 * s21 + s20
        g2 = 2 * alpha_2 * s22 + s21
        h2 = 2 * s22
        if h2 <= 0.0:
            h2 = EPS
        pt3_bcast[env_in_block, 0] = c0
        pt3_bcast[env_in_block, 1] = g0
        pt3_bcast[env_in_block, 2] = h0
        pt3_bcast[env_in_block, 3] = c1
        pt3_bcast[env_in_block, 4] = g1
        pt3_bcast[env_in_block, 5] = h1
        pt3_bcast[env_in_block, 6] = c2
        pt3_bcast[env_in_block, 7] = g2
        pt3_bcast[env_in_block, 8] = h2
    qd.simt.block.sync()
    return (
        pt3_bcast[env_in_block, 0],
        pt3_bcast[env_in_block, 1],
        pt3_bcast[env_in_block, 2],
        pt3_bcast[env_in_block, 3],
        pt3_bcast[env_in_block, 4],
        pt3_bcast[env_in_block, 5],
        pt3_bcast[env_in_block, 6],
        pt3_bcast[env_in_block, 7],
        pt3_bcast[env_in_block, 8],
        ls_it + 3,
    )


@qd.func
def func_linesearch_batch_tiled_wc(
    i_b,
    tid,
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """Tiled wave-cooperative linesearch.

    Equivalent to func_linesearch_batch_wavecoop but with COOP_FACTOR=8
    lanes per env (instead of 64) so that 8 envs run concurrently in
    one wave64. Bracketing/Phase-3 control flow runs redundantly on
    every lane within an env's group (all 8 lanes have identical
    broadcast inputs and so compute identical state).

    The 8 lanes belonging to one env return identical `res_alpha`.
    """
    snorm, gtol = _func_snorm_twc(i_b, tid, constraint_state, rigid_global_info)
    ls_it = gs.qd_int(0)
    ls_result = gs.qd_int(0)
    res_alpha = gs.qd_float(0.0)
    done = False

    if snorm < rigid_global_info.EPS[None]:
        ls_result = 1
        res_alpha = gs.qd_float(0.0)
        done = True

    if not done:
        p0_alpha, p0_cost, p0_deriv_0, p0_deriv_1, base_0, base_1, base_2, ls_it = _func_ls_init_p0_twc(
            i_b,
            tid,
            entities_info=entities_info,
            dofs_state=dofs_state,
            constraint_state=constraint_state,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )
        p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1, ls_it = _func_ls_pt_opt_twc(
            i_b,
            tid,
            p0_alpha - p0_deriv_0 / p0_deriv_1,
            base_0,
            base_1,
            base_2,
            ls_it,
            constraint_state,
            rigid_global_info,
        )

        if p0_cost < p1_cost:
            p1_alpha = p0_alpha
            p1_cost = p0_cost
            p1_deriv_0 = p0_deriv_0
            p1_deriv_1 = p0_deriv_1

        if qd.abs(p1_deriv_0) < gtol:
            if qd.abs(p1_alpha) < rigid_global_info.EPS[None]:
                ls_result = 2
            else:
                ls_result = 0
            res_alpha = p1_alpha
            done = True
        else:
            direction = (p1_deriv_0 < 0) * 2 - 1
            p2update = 0
            p2_alpha = p1_alpha
            p2_cost = p1_cost
            p2_deriv_0 = p1_deriv_0
            p2_deriv_1 = p1_deriv_1
            phase2_break = False
            while p1_deriv_0 * direction <= -gtol and ls_it < rigid_global_info.ls_iterations[None]:
                p2_alpha = p1_alpha
                p2_cost = p1_cost
                p2_deriv_0 = p1_deriv_0
                p2_deriv_1 = p1_deriv_1
                p2update = 1

                p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1, ls_it = _func_ls_pt_opt_twc(
                    i_b,
                    tid,
                    p1_alpha - p1_deriv_0 / p1_deriv_1,
                    base_0,
                    base_1,
                    base_2,
                    ls_it,
                    constraint_state,
                    rigid_global_info,
                )
                if qd.abs(p1_deriv_0) < gtol:
                    res_alpha = p1_alpha
                    done = True
                    phase2_break = True
                    break

            if not phase2_break:
                if ls_it >= rigid_global_info.ls_iterations[None]:
                    ls_result = 3
                    res_alpha = p1_alpha
                    done = True

                if not p2update and not done:
                    ls_result = 6
                    res_alpha = p1_alpha
                    done = True

                if not done:
                    alpha_0 = p1_alpha - p1_deriv_0 / p1_deriv_1
                    alpha_1 = p1_alpha
                    alpha_2 = (p1_alpha + p2_alpha) * 0.5
                    p3_done = False
                    while ls_it < rigid_global_info.ls_iterations[None]:
                        c0, g0, h0, c1, g1, h1, c2, g2, h2, ls_it = _func_ls_pt_3a_twc(
                            i_b,
                            tid,
                            alpha_0,
                            alpha_1,
                            alpha_2,
                            base_0,
                            base_1,
                            base_2,
                            ls_it,
                            constraint_state,
                            rigid_global_info,
                        )

                        best_alpha = gs.qd_float(0.0)
                        best_cost = gs.qd_float(0.0)
                        best_found = False
                        if qd.abs(g0) < gtol:
                            best_alpha = alpha_0
                            best_cost = c0
                            best_found = True
                        if qd.abs(g1) < gtol and (not best_found or c1 < best_cost):
                            best_alpha = alpha_1
                            best_cost = c1
                            best_found = True
                        if qd.abs(g2) < gtol and (not best_found or c2 < best_cost):
                            best_alpha = alpha_2
                            best_cost = c2
                            best_found = True

                        if best_found:
                            res_alpha = best_alpha
                            done = True
                            p3_done = True
                            break

                        b1 = 0
                        p1_next_alpha = alpha_0
                        if p1_deriv_0 < 0 and g0 < 0 and p1_deriv_0 < g0:
                            p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = alpha_0, c0, g0, h0
                            b1 = 1
                        elif p1_deriv_0 > 0 and g0 > 0 and p1_deriv_0 > g0:
                            p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = alpha_0, c0, g0, h0
                            b1 = 2
                        if p1_deriv_0 < 0 and g1 < 0 and p1_deriv_0 < g1:
                            p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = alpha_1, c1, g1, h1
                            b1 = 1
                        elif p1_deriv_0 > 0 and g1 > 0 and p1_deriv_0 > g1:
                            p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = alpha_1, c1, g1, h1
                            b1 = 2
                        if p1_deriv_0 < 0 and g2 < 0 and p1_deriv_0 < g2:
                            p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = alpha_2, c2, g2, h2
                            b1 = 1
                        elif p1_deriv_0 > 0 and g2 > 0 and p1_deriv_0 > g2:
                            p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = alpha_2, c2, g2, h2
                            b1 = 2
                        if b1 > 0:
                            p1_next_alpha = p1_alpha - p1_deriv_0 / p1_deriv_1

                        b2 = 0
                        p2_next_alpha = alpha_1
                        if p2_deriv_0 < 0 and g0 < 0 and p2_deriv_0 < g0:
                            p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1 = alpha_0, c0, g0, h0
                            b2 = 1
                        elif p2_deriv_0 > 0 and g0 > 0 and p2_deriv_0 > g0:
                            p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1 = alpha_0, c0, g0, h0
                            b2 = 2
                        if p2_deriv_0 < 0 and g1 < 0 and p2_deriv_0 < g1:
                            p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1 = alpha_1, c1, g1, h1
                            b2 = 1
                        elif p2_deriv_0 > 0 and g1 > 0 and p2_deriv_0 > g1:
                            p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1 = alpha_1, c1, g1, h1
                            b2 = 2
                        if p2_deriv_0 < 0 and g2 < 0 and p2_deriv_0 < g2:
                            p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1 = alpha_2, c2, g2, h2
                            b2 = 1
                        elif p2_deriv_0 > 0 and g2 > 0 and p2_deriv_0 > g2:
                            p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1 = alpha_2, c2, g2, h2
                            b2 = 2
                        if b2 > 0:
                            p2_next_alpha = p2_alpha - p2_deriv_0 / p2_deriv_1

                        if b1 == 0 and b2 == 0:
                            ls_result = 7
                            res_alpha = p1_alpha
                            done = True
                            p3_done = True
                            break

                        alpha_0 = p1_next_alpha
                        alpha_1 = p2_next_alpha
                        alpha_2 = (p1_alpha + p2_alpha) * 0.5

                    if not p3_done:
                        ls_result = 4
                        res_alpha = p1_alpha
                        done = True

    return res_alpha


@qd.kernel(fastcache=gs.use_fastcache)
def _kernel_solve_body_tiled_wc_amdgpu(
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Tiled wave-cooperative monolith. One workgroup of 64 threads
    handles 8 envs at a time. The 64 lanes are partitioned as 8 envs *
    8 lanes, with each env using its own 8-slot LDS stripe for
    cooperative reductions.
    """
    BLOCK_DIM = qd.static(_TWC_BLOCK_DIM)
    COOP = qd.static(_TWC_COOP_FACTOR)
    ENVS = qd.static(_TWC_ENVS_PER_BLOCK)
    N_DOFS = qd.static(static_rigid_sim_config.n_dofs_)
    _B = static_rigid_sim_config.n_envs
    N_BLOCKS = qd.static((static_rigid_sim_config.n_envs + _TWC_ENVS_PER_BLOCK - 1) // _TWC_ENVS_PER_BLOCK)

    qd.loop_config(
        serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL,
        block_dim=BLOCK_DIM,
    )
    for i in range(N_BLOCKS * BLOCK_DIM):
        tid = i % BLOCK_DIM
        block_id = i // BLOCK_DIM
        env_in_block = tid // COOP
        lane_in_env = tid % COOP
        i_b = block_id * ENVS + env_in_block

        # Per-block LDS scratch.
        # `bcast` (8 envs * 4 slots) carries prev_cost from lane_in_env=0
        # to all 8 lanes of an env's group during apply-alpha.
        # `cost_red` / `gauss_red` / `act_red` are the LDS partials
        # for the update-constraint cost/gauss reductions and the
        # per-iter active-env reduction.
        bcast = qd.simt.block.SharedArray((ENVS, 4), gs.qd_float)
        cost_red = qd.simt.block.SharedArray((BLOCK_DIM,), gs.qd_float)
        gauss_red = qd.simt.block.SharedArray((BLOCK_DIM,), gs.qd_float)
        act_red = qd.simt.block.SharedArray((BLOCK_DIM,), gs.qd_int)
        any_active_bcast = qd.simt.block.SharedArray((1,), gs.qd_int)
        # Per-env working vector for the cooperative LDL^T mass solve (Phase 5),
        # one N_DOFS stripe per env in the block (8 lanes/env cooperate on it).
        msolve_t = qd.simt.block.SharedArray((ENVS, N_DOFS), gs.qd_float)

        # Out-of-range guard (only the last block can have i_b >= _B
        # if _B isn't divisible by ENVS_PER_BLOCK; the is_compatible
        # gate enforces divisibility, so this is always false in the
        # benchmark path -- left here as a safety check). Note: we
        # MUST NOT `continue` here when out of range, because the
        # workgroup-uniform syncs below would deadlock. Instead we
        # treat OOB envs as inactive and let them participate in
        # syncs with zero contributions.
        oob = i_b >= _B

        # Initial per-env active flag (avoid reading constraint_state
        # for OOB lanes -- guard with `oob` mask).
        n_con_init = gs.qd_int(0)
        improved_init = False
        if not oob:
            n_con_init = constraint_state.n_constraints[i_b]
            improved_init = constraint_state.improved[i_b]

        # Envs with zero constraints: clear improved and never enter loop.
        if (not oob) and n_con_init == 0:
            if lane_in_env == 0:
                constraint_state.improved[i_b] = False
            improved_init = False

        is_active_env = (not oob) and (n_con_init > 0) and improved_init

        # Iteration loop -- block-wide. We keep iterating until ALL 8
        # envs in the block have converged (or we hit the iteration
        # cap). Per-env writes are predicated on is_active_env so that
        # already-converged envs are not corrupted by further updates.
        for _it in range(rigid_global_info.iterations[None]):
            # Block-wide reduction: any_active = OR over all 8 envs.
            act_red[tid] = gs.qd_int(is_active_env)
            qd.simt.block.sync()
            if tid == 0:
                tot = gs.qd_int(0)
                for k in qd.static(range(BLOCK_DIM)):
                    tot = tot + act_red[k]
                any_active_bcast[0] = tot
            qd.simt.block.sync()
            if any_active_bcast[0] == 0:
                break

            n_con = gs.qd_int(0)
            ne_local = gs.qd_int(0)
            nef_local = gs.qd_int(0)
            if is_active_env:
                n_con = constraint_state.n_constraints[i_b]
                ne_local = constraint_state.n_constraints_equality[i_b]
                nef_local = ne_local + constraint_state.n_constraints_frictionloss[i_b]

            # ===== Phase 1: linesearch (tiled wave-coop) =====
            # Note: linesearch helpers run unconditionally for all 8
            # envs in the block (because they contain workgroup syncs
            # that all 64 lanes must reach). For inactive envs, the
            # reductions get zeros (n_con=0 -> empty loops) and
            # alpha_val is effectively garbage; we predicate the
            # apply-alpha writes below on `is_active_env` so that
            # garbage alpha never touches inactive env state.
            alpha_val = func_linesearch_batch_tiled_wc(
                i_b,
                tid,
                entities_info=entities_info,
                dofs_state=dofs_state,
                rigid_global_info=rigid_global_info,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )

            # Capture prev_cost (per-env, written by lane_in_env=0).
            if is_active_env and lane_in_env == 0:
                bcast[env_in_block, 1] = constraint_state.cost[i_b]
            qd.simt.block.sync()

            # Tiny-alpha gate: per-env -> downgrade is_active_env.
            tiny_alpha = qd.abs(alpha_val) < rigid_global_info.EPS[None]
            if is_active_env and tiny_alpha:
                if lane_in_env == 0:
                    constraint_state.improved[i_b] = False
                is_active_env = False

            alpha = alpha_val
            prev_cost = bcast[env_in_block, 1]

            # ===== Phase 2a: apply alpha to qacc, Ma (per-dof, parallel) =====
            if is_active_env:
                i_d = lane_in_env
                while i_d < N_DOFS:
                    constraint_state.qacc[i_d, i_b] = (
                        constraint_state.qacc[i_d, i_b] + constraint_state.search[i_d, i_b] * alpha
                    )
                    constraint_state.Ma[i_d, i_b] = (
                        constraint_state.Ma[i_d, i_b] + constraint_state.mv[i_d, i_b] * alpha
                    )
                    i_d = i_d + COOP

                # ===== Phase 2b: apply alpha to Jaref (per-constraint) =====
                i_c = lane_in_env
                while i_c < n_con:
                    constraint_state.Jaref[i_c, i_b] = (
                        constraint_state.Jaref[i_c, i_b] + constraint_state.jv[i_c, i_b] * alpha
                    )
                    i_c = i_c + COOP
            qd.simt.block.sync()

            # ===== Phase 3: save prev grad (CG only) =====
            if qd.static(static_rigid_sim_config.solver_type == gs.constraint_solver.CG):
                if is_active_env:
                    i_d = lane_in_env
                    while i_d < N_DOFS:
                        constraint_state.cg_prev_grad[i_d, i_b] = constraint_state.grad[i_d, i_b]
                        constraint_state.cg_prev_Mgrad[i_d, i_b] = constraint_state.Mgrad[i_d, i_b]
                        i_d = i_d + COOP
                qd.simt.block.sync()

            # ===== Phase 4: update_constraint_batch =====
            if is_active_env and lane_in_env == 0:
                constraint_state.prev_cost[i_b] = prev_cost

            # 4a: per-constraint active flag + efc_force, fused with
            # friction/equality cost contribution.
            my_cost_partial = gs.qd_float(0.0)
            my_gauss_partial = gs.qd_float(0.0)
            if is_active_env:
                i_c = lane_in_env
                while i_c < n_con:
                    Jaref_c = constraint_state.Jaref[i_c, i_b]
                    efc_D_c = constraint_state.efc_D[i_c, i_b]

                    if qd.static(static_rigid_sim_config.solver_type == gs.constraint_solver.Newton):
                        constraint_state.prev_active[i_c, i_b] = constraint_state.active[i_c, i_b]

                    active_c = True
                    floss_force = gs.qd_float(0.0)
                    floss_cost_local = gs.qd_float(0.0)

                    if ne_local <= i_c and i_c < nef_local:
                        f = constraint_state.efc_frictionloss[i_c, i_b]
                        r = constraint_state.diag[i_c, i_b]
                        rf = r * f
                        linear_neg = Jaref_c <= -rf
                        linear_pos = Jaref_c >= rf
                        active_c = not (linear_neg or linear_pos)
                        floss_force = linear_neg * f + linear_pos * -f
                        floss_cost_local = linear_neg * f * (-0.5 * rf - Jaref_c) + linear_pos * f * (
                            -0.5 * rf + Jaref_c
                        )
                    elif nef_local <= i_c:
                        active_c = Jaref_c < 0

                    constraint_state.active[i_c, i_b] = active_c
                    constraint_state.efc_force[i_c, i_b] = floss_force + (-Jaref_c * efc_D_c * active_c)

                    my_cost_partial = (
                        my_cost_partial + floss_cost_local + 0.5 * Jaref_c * Jaref_c * efc_D_c * active_c
                    )
                    i_c = i_c + COOP
            qd.simt.block.sync()

            # 4b: per-dof qfrc_constraint = J^T @ efc_force.
            if is_active_env:
                i_d = lane_in_env
                while i_d < N_DOFS:
                    qfrc = gs.qd_float(0.0)
                    for j_c in range(n_con):
                        qfrc = qfrc + constraint_state.jac[j_c, i_d, i_b] * constraint_state.efc_force[j_c, i_b]
                    constraint_state.qfrc_constraint[i_d, i_b] = qfrc
                    i_d = i_d + COOP

                # 4c: per-dof gauss / cost contribution from (Mx-Mx')*(x-x').
                i_d = lane_in_env
                while i_d < N_DOFS:
                    v = (
                        0.5
                        * (constraint_state.Ma[i_d, i_b] - dofs_state.force[i_d, i_b])
                        * (constraint_state.qacc[i_d, i_b] - dofs_state.acc_smooth[i_d, i_b])
                    )
                    my_cost_partial = my_cost_partial + v
                    my_gauss_partial = my_gauss_partial + v
                    i_d = i_d + COOP

            # 4d: LDS reduce my_cost_partial / my_gauss_partial -> totals.
            cost_red[tid] = my_cost_partial
            gauss_red[tid] = my_gauss_partial
            qd.simt.block.sync()
            if is_active_env and lane_in_env == 0:
                total_cost = gs.qd_float(0.0)
                total_gauss = gs.qd_float(0.0)
                base = env_in_block * COOP
                for k in qd.static(range(COOP)):
                    total_cost = total_cost + cost_red[base + k]
                    total_gauss = total_gauss + gauss_red[base + k]
                constraint_state.cost[i_b] = total_cost
                constraint_state.gauss[i_b] = total_gauss
            qd.simt.block.sync()

            # ===== Phase 5: update gradient (tiled wave-coop, 8 lanes/env) =====
            # Previously ran on lane_in_env=0 of each env (1 lane/env, so the
            # O(n_dofs^2) mass solve stayed serial per env). Now the 8 lanes of
            # each env's group cooperate. All block.sync() calls below stay
            # workgroup-uniform (entity dof counts are per-entity, so sweep trip
            # counts match across all 8 envs); only the LDS/HBM work is gated.
            # 5a: grad = Ma - force - qfrc_constraint (per-dof, lane_in_env-strided)
            if is_active_env:
                i_d = lane_in_env
                while i_d < N_DOFS:
                    constraint_state.grad[i_d, i_b] = (
                        constraint_state.Ma[i_d, i_b]
                        - dofs_state.force[i_d, i_b]
                        - constraint_state.qfrc_constraint[i_d, i_b]
                    )
                    i_d = i_d + COOP
            qd.simt.block.sync()

            # 5b: cooperative LDL^T mass solve per entity (CG only).
            if qd.static(static_rigid_sim_config.solver_type == gs.constraint_solver.CG):
                for i_e in range(qd.static(static_rigid_sim_config.n_entities_)):
                    e_ds = entities_info.dof_start[i_e]
                    e_de = entities_info.dof_end[i_e]
                    e_n = e_de - e_ds
                    # do_e read only for active (non-OOB) lanes; mask honored.
                    do_e = False
                    if is_active_env:
                        do_e = bool(rigid_global_info.mass_mat_mask[i_e, i_b])

                    # load y -> LDS stripe
                    if do_e:
                        i_d = e_ds + lane_in_env
                        while i_d < e_de:
                            msolve_t[env_in_block, i_d] = constraint_state.grad[i_d, i_b]
                            i_d = i_d + COOP
                    qd.simt.block.sync()

                    # Step 1: solve L^T w = y (back-substitution, p high->low)
                    for pp in range(e_n):
                        p = e_de - 1 - pp
                        if do_e:
                            wp = msolve_t[env_in_block, p]
                            k = e_ds + lane_in_env
                            while k < p:
                                msolve_t[env_in_block, k] = (
                                    msolve_t[env_in_block, k] - rigid_global_info.mass_mat_L[p, k, i_b] * wp
                                )
                                k = k + COOP
                        qd.simt.block.sync()

                    # Step 2: z = D^{-1} w
                    if do_e:
                        i_d = e_ds + lane_in_env
                        while i_d < e_de:
                            msolve_t[env_in_block, i_d] = (
                                msolve_t[env_in_block, i_d] * rigid_global_info.mass_mat_D_inv[i_d, i_b]
                            )
                            i_d = i_d + COOP
                    qd.simt.block.sync()

                    # Step 3: solve L x = z (forward-substitution, p low->high)
                    for pp in range(e_n):
                        p = e_ds + pp
                        if do_e:
                            wp = msolve_t[env_in_block, p]
                            k = p + 1 + lane_in_env
                            while k < e_de:
                                msolve_t[env_in_block, k] = (
                                    msolve_t[env_in_block, k] - rigid_global_info.mass_mat_L[k, p, i_b] * wp
                                )
                                k = k + COOP
                        qd.simt.block.sync()

                    # store x -> Mgrad
                    if do_e:
                        i_d = e_ds + lane_in_env
                        while i_d < e_de:
                            constraint_state.Mgrad[i_d, i_b] = msolve_t[env_in_block, i_d]
                            i_d = i_d + COOP
                    qd.simt.block.sync()

            # ===== Phase 6: terminate or update descent (tiled wave-coop) =====
            # grad-norm + CG-beta dot products as 8-lane stripe reductions;
            # search update lane_in_env-strided. Syncs block-uniform.
            my_gn = gs.qd_float(0.0)
            if is_active_env:
                i_d = lane_in_env
                while i_d < N_DOFS:
                    g_d = constraint_state.grad[i_d, i_b]
                    my_gn = my_gn + g_d * g_d
                    i_d = i_d + COOP
            cost_red[tid] = my_gn
            qd.simt.block.sync()
            if is_active_env and lane_in_env == 0:
                gn_sq = gs.qd_float(0.0)
                base = env_in_block * COOP
                for k in qd.static(range(COOP)):
                    gn_sq = gn_sq + cost_red[base + k]
                grad_norm = qd.sqrt(gn_sq)
                tol_scaled = (
                    rigid_global_info.meaninertia[i_b] * qd.max(1, N_DOFS)
                ) * rigid_global_info.tolerance[None]
                improvement = prev_cost - constraint_state.cost[i_b]
                improved6 = (grad_norm > tol_scaled) and (improvement > tol_scaled)
                constraint_state.improved[i_b] = improved6
                bcast[env_in_block, 2] = gs.qd_float(1.0) if improved6 else gs.qd_float(0.0)
            qd.simt.block.sync()

            env_improved = bcast[env_in_block, 2] > 0.5
            if is_active_env and env_improved:
                my_beta = gs.qd_float(0.0)
                my_pgpm = gs.qd_float(0.0)
                i_d = lane_in_env
                while i_d < N_DOFS:
                    grad_d = constraint_state.grad[i_d, i_b]
                    Mgrad_d = constraint_state.Mgrad[i_d, i_b]
                    pMgrad_d = constraint_state.cg_prev_Mgrad[i_d, i_b]
                    pgrad_d = constraint_state.cg_prev_grad[i_d, i_b]
                    my_beta = my_beta + grad_d * (Mgrad_d - pMgrad_d)
                    my_pgpm = my_pgpm + pMgrad_d * pgrad_d
                    i_d = i_d + COOP
                cost_red[tid] = my_beta
                gauss_red[tid] = my_pgpm
            qd.simt.block.sync()
            if is_active_env and env_improved and lane_in_env == 0:
                t_beta = gs.qd_float(0.0)
                t_pgpm = gs.qd_float(0.0)
                base = env_in_block * COOP
                for k in qd.static(range(COOP)):
                    t_beta = t_beta + cost_red[base + k]
                    t_pgpm = t_pgpm + gauss_red[base + k]
                cg_beta = qd.max(t_beta / qd.max(rigid_global_info.EPS[None], t_pgpm), 0.0)
                constraint_state.cg_pg_dot_pMg[i_b] = t_pgpm
                constraint_state.cg_beta[i_b] = cg_beta
                bcast[env_in_block, 3] = cg_beta
            qd.simt.block.sync()
            if is_active_env and env_improved:
                cg_beta_b = bcast[env_in_block, 3]
                i_d = lane_in_env
                while i_d < N_DOFS:
                    constraint_state.search[i_d, i_b] = (
                        -constraint_state.Mgrad[i_d, i_b] + cg_beta_b * constraint_state.search[i_d, i_b]
                    )
                    i_d = i_d + COOP
            qd.simt.block.sync()

            # Refresh per-env active flag for the next iter -- the
            # scalar terminate above may have just set improved=False.
            if is_active_env:
                # constraint_state.improved[i_b] is workgroup-uniform
                # within the 8 lanes of an env's group.
                is_active_env = bool(constraint_state.improved[i_b])


def _tiled_wc_amdgpu_is_compatible(*args, **kwargs):
    # Tiled wave-coop variant. Eligible when:
    #   - backend = AMDGPU
    #   - dense Jacobian (sparse_solve=False)
    #   - CG solver (Newton needs in-iter Hessian/Cholesky update)
    #   - cfg.n_envs >= ENVS_PER_BLOCK and is a multiple of it. The
    #     ">=" lower bound is required because cfg.n_envs is internally
    #     `_B = max(1, user_n_envs)`, so the unbatched user-facing case
    #     (n_envs=0) arrives here as cfg.n_envs=1, which trivially
    #     satisfies `n_envs % 8 == 0`-style modulo checks but breaks the
    #     kernel's 8-env-per-workgroup partitioning. The kernel's
    #     reduction order also differs from the monolith's by ULP-level
    #     FP drift that compounds across simulation steps on small
    #     unbatched workloads.
    if gs.backend not in {gs.amdgpu}:
        return False
    cfg = kwargs.get("static_rigid_sim_config", args[4] if len(args) >= 5 else None)
    if cfg is None:
        return False
    if cfg.sparse_solve:
        return False
    if cfg.solver_type != gs.constraint_solver.CG:
        return False
    if cfg.n_envs < _TWC_ENVS_PER_BLOCK or cfg.n_envs % _TWC_ENVS_PER_BLOCK != 0:
        return False
    return True


@solver.func_solve_body.register(is_compatible=_tiled_wc_amdgpu_is_compatible)
def func_solve_body_tiled_wc_amdgpu(
    entities_info,
    dofs_state,
    constraint_state,
    rigid_global_info,
    static_rigid_sim_config,
    _n_iterations,
):
    # Iteration loop is *inside* the kernel (mirrors monolith), so we
    # launch once per step rather than once per CG iter.
    _kernel_solve_body_tiled_wc_amdgpu(
        entities_info,
        dofs_state,
        constraint_state,
        rigid_global_info,
        static_rigid_sim_config,
    )
