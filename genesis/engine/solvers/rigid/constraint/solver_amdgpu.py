"""
AMDGPU-specific variants of `func_solve_body`, registered with the perf
dispatch system in `solver.func_solve_body`.

Two strategies are registered, both targeting the high VGPR pressure
(~410 archived registers/wave -> 1 wave/SIMD = 12.5% theoretical occupancy)
of the baked-in monolithic kernel `func_solve_body_monolith` on gfx942:

- B3 / `func_solve_body_split_amdgpu`: per-iteration **2-kernel split**
  (linesearch + apply-alpha | post-linesearch). Iteration loop lifted to
  Python. Halves the inlined call chain seen by the LLVM AMDGPU backend
  inside each kernel, giving the register allocator a smaller live-range
  graph to budget.

- B4 / `func_solve_body_lifted_loop_amdgpu`: per-iteration **single-kernel
  launch** (full `func_solve_iter` body), iteration loop lifted to Python.
  Keeps the body monolithic but kills the cross-iteration live ranges the
  compiler would otherwise have to carry through the `for _ in
  range(iterations):` loop.

Both AMD variants lift the iteration loop into Python (instead of inside
the kernel) but preserve baseline per-batch convergence semantics: each
inner per-iter kernel gates on `constraint_state.improved[i_b]` (a
device-side read, no D2H sync) and skips both the linesearch+apply step
and the post-linesearch update for batches that have already converged.
Without this gate, converged batches would re-run linesearch using a
stale `search` direction and inject FP noise into qacc/Ma/Jaref that
accumulates over many sim steps (broke test_mesh_align). This matches
the existing CUDA `func_solve_decomposed` behavior in
`solver_breakdown.py`.

Both variants pin `block_dim=64` to avoid the 50% VALU-lane-masking
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
