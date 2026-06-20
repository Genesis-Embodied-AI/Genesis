# Island ON/OFF Unification — Working Log

> PERSISTENT progress/experiment log for the rigid-solver island unification effort.
> Branch: `island_hibernation_reboot`. Keep this file updated EVERY session. Read it FIRST.
> Companion (private) memory note: `project_monolith_host_orchestration_regression.md`.

## !!! DISCIPLINE (read before touching ANY code) !!!
- **HEED THE EXPERIMENT LOG.** Q2 (2026-06-20) failed because it directly contradicted experiment #6, which
  was ALREADY in this file: "skipping the OFF init makes OFF free-fall, OFF needs the seeded first search."
  The log was right; I ignored it and re-broke it. ALWAYS scan VERIFIED INVARIANTS + EXPERIMENT LOG first.
- **Separate VERIFIED from THEORIZED.** Anything not confirmed by a passing test or a kernel-print/probe is a
  THEORY - label it. Never state a theory as an invariant. Q2's premise ("OFF init is redundant") was an
  unverified theory written as an invariant; it was false.
- **INSTRUMENT BEFORE FIXING.** No code change to the solve path without probe/print evidence of which branch
  fires and what values it sees. Guessing has cost two full break-revert cycles (Q2, force_whole_env).
- **CPU green != GPU correct.** Decomposed/tiled is GPU-only. The monolith arm is NOT exercised by the CUDA
  suite for islands ON (autotuner picks decomposed), so a CUDA suite pass does NOT validate monolith-ON.

## GOAL
Make the rigid constraint solver behave identically for islands ON vs OFF, with two required properties:
1. **1 island ⇒ identical to islands OFF** (a single island spanning the whole env is the same problem).
2. **N islands ⇒ linear/parallel scaling** (each island solved independently, parallel over (env, island)).
Rule from the user: **NOTHING may be forced whole-env.** Everything parallel over (env, island).

## TL;DR CURRENT STATE (update each session)
- **2026-06-21 (SESSION: BOTH FIXES IMPLEMENTED + VALIDATED).** The single-island ON penalty (scalar
  one-thread-per-island init factor, ~2.2ms) is FIXED on BOTH arms by routing the GPU-island seed through the
  TILED PER-ISLAND factor (`func_island_tiled_factor_solve_all`, the same factor the decomposed graph already
  runs each iteration) instead of the scalar `func_hessian_and_cholesky_factor_direct` island branch.
  - DECOMPOSED: func_solve_init GPU-island branch -> no-solve gradient + `func_island_tiled_factor_solve_all(
    do_assemble=True, write_L=False)`. Init factor 2.209ms -> 0.605ms (CUDA, single 48-dof island). The graph
    rebuilds nt_H so the seed need not persist L.
  - MONOLITH: same seed branch with write_L=True (persists L into nt_H for the incremental rank-1 iterations);
    the scalar in-body self-init is dropped for the fits_shared case. Mirrors the existing NON-island monolith
    fused init (`enable_fused_factor_solve_init` + `write_L_to_nt_H`, rigid_solver.py:593).
  - GATE: `Newton and use_contact_island and gpu and hessian_fits_shared`. Above the shared cap or CPU it falls
    back to the existing scalar path (decomposed graph also uses its non-fused scalar path there, so consistent).
  - NEW: `write_L` template on `func_island_assemble_factor_solve_tiled` + `func_island_tiled_factor_solve_all`;
    `func_update_gradient_no_solve` moved solver_breakdown -> solver.py (shared by graph + seed).
  - VALIDATED: CUDA suite 26/26 (decomposed); physics bitwise ON==OFF (max|z_ON-z_OFF|=0.000000) for mono+dec,
    stack(1 island)+grid(16 islands), on BOTH CUDA and Metal (headless check_monolith_island.py). Metal pytest
    suite blocked by macOS visualizer/display flakiness (cocoa screens[0] IndexError at "Building visualizer",
    AFTER "Compiling simulation kernels" succeeds) - environmental, NOT solver code; the headless physics checks
    cover Metal correctness.
  - PENDING: arm-crossover sweep (re-tune rigid_solver.py pin from fresh numbers); commit+push.

- HEAD `a42db7707` (Step 2). **Decomposed arm now correctly seeded by func_solve_init for ON and OFF.**
  CONFIRMED CUDA: suite 26/26; forced-decomposed probe ON and OFF both REST (~0.049, was sinking 0.049->0.029
  i.e. diverging). This fixed BOTH the Step-1 OFF regression AND a PRE-EXISTING latent divergence in the
  decomposed island-ON path (its GPU-island seed had always been skipped; masked by tests using the monolith
  for ON + benchmarks not asserting physics). Original "decomposed ON 0.78 < OFF 0.96" = diverging vs correct,
  NOT a parity gap. Now both ~0.96 (correct). Monolith unchanged. Plus Q1 (single-island monolith incremental).
- REMAINING: monolith ON vs OFF parity for one big island (Q1 helped, gap remains ~13.6 vs 2.8 @256env -
  needs Step-2-monolith: unify monolith init onto tiled factor / address scalar self-init); monolith CPU/GPU
  scaling check; OPTIONAL perf: make the decomposed graph self-seed (direction before first linesearch) to
  drop the seed factor (~0.96 -> faster). See FIX PLAN.
- Earlier: Q1 (single-island monolith uses incremental factor). Q2-skip-decomposed-init was attempted, found
  to cause divergence (decomposed has no self-init), and resolved by the always-seed fix above (NOT reverted).
- **Decomposed arm already meets the goal**: (env,island)-tiled, 1 island = OFF, scales near-flat
  (8/32/128 islands grid @1env: 0.78/1.16/3.14 ms). Autotuner picks it for islands in production.
- **Q1 (DONE, partial)**: single awake island now uses incremental dense in the monolith (CPU-correct, 10/10).
  Helped monolith stack 4.5/20.6 -> 4.02/13.59 ms (1env/256env) but did NOT close the gap to OFF (1.30/2.78).
  Residual = scalar self-init + per-island indirection in the monolith's vector ops, both incremental.
- **Q2 (REVERTED, premise was WRONG)**: I theorized func_solve_init's factor is redundant for the decomposed
  arm so OFF could skip it. FALSE - skipping the OFF init makes the decomposed arm DIVERGE (boxes free-fall;
  test_solve_correctness[*-0]/[*-5] fail, ACTUAL rests, DESIRED=OFF at -0.96). ON tolerates the skip, OFF does
  NOT. The OFF init is REQUIRED. So decomposed ON<OFF is because ON skips a NEEDED init and gets away with it
  (probably the per-step partition resets state ON relies on), while OFF correctly does it. WHY ON tolerates
  the skip but OFF can't is NOT understood - do not retry the skip without answering that first.

## PERF FINDINGS (2026-06-21) - MONOLITH single-island ON cause + the REAL fix
- Split fix (Step 4, skip converging-iteration Cholesky): REVERTED. A/B ~1-3% (noise); settled solves do 1
  iter/step (probe), so there is no extra Cholesky to skip. Increased complexity for ~0 gain.
- Monolith perf (BENCH_ARM=0 forces the arm; stack 8 = single connected island; settled = 1 iter/step):
    monolith   stack8 1env  OFF 1.09ms  ON 3.27ms (3.0x)   256env OFF 2.32 ON 11.0
    decomposed stack8 1env  OFF 0.96ms  ON 3.26ms (3.4x)
  monolith ~= decomposed; the ~3x ON penalty is the SAME on both arms. NOT iteration count (1 both), NOT the
  partition (kernel_build_islands+group ~0.1ms).
- MONOLITH per-segment profile (qd.profiler DOES break down per-loop in a kernel) is DECISIVE:
    OFF: func_solve_init hessian_direct_TILED 0.075 + cholesky_fused_TILED 0.057 = 0.13ms ; _kernel_solve_monolith 0.37ms
    ON : _kernel_solve_monolith 2.56ms  (the tiled segments are ABSENT)
  MECHANISM: for islands ON the init factor takes the SCALAR per-island branch of
  func_hessian_and_cholesky_factor_direct (func_hessian_direct_batch + func_cholesky_factor_direct_batch,
  solver.py ~2748-2771, one thread per island). For a single 48-dof island that is a ONE-THREAD scalar
  Cholesky ~2.2ms; OFF uses the 32-lane TILED whole-env factor (func_hessian_direct_tiled +
  func_cholesky_factor_direct_tiled, ~0.13ms). The tiled path is the NON-island GPU branch (~2788) only.
- ATTEMPTED FIX (9c2fa478c, REVERTED): move the monolith GPU-island factor into func_solve_init. NO GAIN -
  re-profile showed the 2.2ms just relocated to `func_solve_init ... hess_cholesky_factor_direct_island`
  (the island branch is the SAME scalar code). Confirmed func_solve_init's island branch == scalar per-island.
- REAL FIX (proposed, NOT yet done): the MONOLITH is env-packed - it does NOT parallelize over islands, so the
  per-island factor buys it nothing and the scalar one-thread-per-island Cholesky is pure loss. For a SINGLE
  island (n_islands==1, no hibernation) the monolith should factor the WHOLE ENV with the TILED path
  (func_hessian_direct_tiled + func_cholesky_factor_direct_tiled) - identical to OFF - because 1 island = whole
  env. This mirrors Q1 exactly (single island -> whole-env incremental iterations). Keep per-island only for
  multi-island / hibernation (where asleep islands must be skipped and the whole-env factor would move them).
  Expect monolith single-island ON ~3.3ms -> ~1.1ms (== OFF).
- DECOMPOSED 3.4x RESOLVED (2026-06-21) - SAME ROOT CAUSE as the monolith. Full decomposed-ON profile:
  `func_solve_init ... hess_cholesky_factor_direct_island` = 2.209ms (73%). That is func_solve_init's
  always-seed factor taking the SCALAR per-island branch (func_hessian_direct_batch + func_cholesky_factor_
  direct_batch, solver.py 2748-2771) - the SAME scalar single-big-island Cholesky as the monolith. For the
  decomposed it is also DOUBLY WASTED: the graph re-factors per-island per iteration, so the seed factor's L
  is thrown away (only its gradient/search seed is used).
- ROOT (both arms): scalar ONE-THREAD-PER-ISLAND Cholesky of a SINGLE 48-dof island ~2.2ms; OFF uses the
  32-lane TILED whole-env factor ~0.13ms. This is bad ONLY for a single big island - for many small islands
  the per-island scalar is fine (tiny blocks, parallel over (env,island)) and whole-env there = the O(n^3)
  cubic blowup (the force_whole_env catastrophe). So the fix MUST gate on n_islands==1.
- ONE FIX (both arms, proposed): when n_islands==1 (single island = whole env) and not hibernation, use the
  TILED whole-env factor + gradient (func_hessian_direct_tiled + func_cholesky_factor_direct_tiled +
  func_update_gradient_tiled) instead of the scalar per-island path, in func_hessian_and_cholesky_factor_direct
  + func_update_gradient. Decomposed: seed becomes tiled (graph still re-factors per-island - per-iteration
  scaling unchanged). Monolith: single-island init becomes tiled (iter already whole-env incremental via Q1).
  COMPLICATION: the tiled factor is whole-env/cooperative; needs an n_islands==1 route that does NOT send
  many-island envs whole-env (cubic). Mixed batches (some envs single, some multi island) are the hard case.
  Expect both arms single-island ON ~3.3ms -> ~1.1ms. NOT yet implemented - confirm approach (cubic risk).

## REFINED FIX (2026-06-21, user steer "why not TILED PER-ISLAND?") - resolves the cubic risk
- The correct factor is NOT "tiled whole-env vs scalar per-island" - it is the THIRD variant that already
  exists: TILED PER-ISLAND `func_island_assemble_factor_solve_tiled` (solver.py:1982), driven over the
  (env,island) grid by `func_island_tiled_factor_solve_all` (solver.py:2130). It is the SAME register-streaming
  TileTxT Cholesky as the whole-env tiled factor, applied to each island's contiguous [gbase,gbase+n) block
  with T lanes cooperating PER ISLAND. So: single island spanning the env == whole-env tiled (ON==OFF); many
  small islands == each block factored by its own tile (NO cubic, NO whole-env cap). This is EXACTLY what the
  decomposed graph already runs every iteration (solver_breakdown.py:1024). NO n_islands==1 special-case, NO
  mixed-batch hard case, NO cubic risk. The scalar per-island was simply the wrong choice wired into the seed.
- VERIFIED L-WRITEBACK: `func_island_assemble_factor_solve_tiled` reads nt_H, factors into shared L_sh, solves
  grad->Mgrad (global), and does NOT write L back to nt_H (docstring:2008). The two arms maintain nt_H
  differently, so this matters:
    DECOMPOSED graph keeps nt_H = raw H (do_assemble=False per iter; re-assembled by _func_newton_only_nt_hessian
      on full-rebuild iters, delta-patched otherwise). It must NOT find L in nt_H. => tiled per-island with NO
      L-writeback is a PERFECT drop-in. The seed's nt_H is discarded by the graph's iter-0 full rebuild; the
      seed only needs to produce Mgrad/search.
    MONOLITH per-iter incremental rank-1 (func_hessian_and_cholesky_factor_incremental_batch) reads L IN PLACE
      from nt_H. => it needs L persisted, i.e. tiled per-island WITH an L-writeback. Also: the cooperative tiled
      factor cannot live inside the scalar env-packed _kernel_solve_monolith body - it must be hoisted to a
      separate cooperative kernel (like the seed already is), and the monolith body drops its scalar self-init.
- IMPLEMENTATION (step 1 = decomposed, clean; step 2 = monolith):
    STEP 1 (decomposed, in func_solve_init): when Newton and use_contact_island and gpu and hessian_fits_shared
      and is_decomposed, replace the scalar factor (func_hessian_and_cholesky_factor_direct) + scalar gradient
      (func_update_gradient) with: no-solve gradient (grad=Ma-force-qfrc) then
      func_island_tiled_factor_solve_all(do_assemble=True). The seed only runs for the DECOMPOSED arm in the
      island+gpu case (monolith skips it), so this only affects decomposed. Dedup: move
      _func_update_gradient_no_solve (breakdown.py:911) -> solver.func_update_gradient_no_solve so both call it.
    STEP 2 (monolith): add write_L template to func_island_assemble_factor_solve_tiled; let the monolith use the
      hoisted seed with write_L=True and drop its in-body scalar self-init. Gate on hessian_fits_shared (matches
      the graph's fused gate). Mixed/over-cap envs fall back to the existing scalar path (unchanged).

## BENCHMARK 2026-06-21 (BOTH fixes in; forced-arm A/B; grid = N single-box islands, 1 env)
Tiled per-island SEED win (single 48-dof island, CUDA, func_solve_init factor): scalar 2.209ms -> tiled 0.605ms.
Monolith single-island ON init now tiled too (kernel_15 0.614ms + body 0.438ms), was scalar 2.2ms.

Forced-arm crossover, island ON, ms/step (mono = prefer 0, dec = prefer 1):
| bodies | islands | n_dofs | tiled_fits | CUDA mono | CUDA dec | Metal mono | Metal dec |
|---|---|---|---|---|---|---|---|
|   4 |   4 |  24 | True  |  0.83 |  0.81 |  1.69 |  1.78 |
|  16 |  16 |  96 | True  |  2.18 |  1.57 |  4.72 |  3.91 |
|  36 |  36 | 216 | False |  7.26 |  6.88 | 10.01 | 14.06 |
|  64 |  64 | 384 | False | 14.49 | 26.36 | 19.03 | 56.93 |
| 100 | 100 | 600 | False | 22.34 | 90.93 | 28.79 | 195.29 |
Stack (1 big island, 8 boxes, 48 dofs, tiled_fits=True): CUDA 1env mono 1.74 / dec 1.55; 256env mono 3.78 / dec 2.65.
OFF reference stack: CUDA 1env mono 1.09 / dec 0.96; 256env mono 2.31 / dec 1.24.

READING: tiled_fits=True (whole-env <= ~96 dofs) -> decomposed slightly faster or tied (both arms tiled). Past the
shared cap (tiled_fits=False) the DECOMPOSED arm degrades badly with island count (up to 4x CUDA / 6.8x Metal SLOWER
than monolith). BOTH backends agree. This INVERTS the old report (decomposed-wins-at-scale).

## !!! NEXT TARGET: decomposed many-island tiled_fits=False regression (NOT YET FIXED) !!!
ROOT (CUDA grid-100 profile, decomposed): `func_solve_init ... hess_cholesky_factor_direct_island` = 10.257ms (67%).
That is func_solve_init's SCALAR per-island seed (`func_hessian_and_cholesky_factor_direct` island branch), taken
because the tiled seed is gated on `hessian_fits_shared` (WHOLE-ENV cap, False at 600 dofs) even though every island
is 6 dofs. The monolith stays per-island via its incremental rank-1 update, so it does not pay this and wins.
PROPOSED FIX (the real N-island linear-scaling fix; aligns with "nothing whole-env"): the per-island tiled factor's
shared tile is sized `MAX_DOFS = tiled_n_dofs` (whole-env) in `func_island_tiled_factor_solve_all`; it should be
`tiled_n_island_dofs` (per-island; rigid_solver.py:568 already computes it and documents it as "always usable
because islands are small"). Then:
  - size L_sh/v_sh to tiled_n_island_dofs (per-island, fits shared even when whole-env does not),
  - add a runtime guard in func_island_assemble_factor_solve_tiled: island n > tiled_n_island_dofs -> scalar fallback
    (a big single island still degrades to scalar, but that is rare and unavoidable; many SMALL islands get tiled),
  - gate the seed + the decomposed graph (solver_breakdown.py:1009) on a per-island-fits flag instead of
    hessian_fits_shared; the graph's tiled_fits=False branch (`_func_newton_only_nt_hessian_and_cholesky`, whole-env
    func_hessian_direct_tiled + func_cholesky_factor_direct_tiled) becomes a fallback only for big islands.
RISK: reworks the decomposed hot-path factor dispatch + shared sizing for ALL island runs; needs bitwise ON==OFF +
suite + determinism re-validation on both backends. tiled_fits=True cases are UNAFFECTED (tiled_n_island_dofs ==
tiled_n_dofs there). DEFERRED to a joint step with the user (hot-path + determinism sensitivity).

## PIN RE-TUNE (rigid_solver.py elif False) - PENDING the above fix
The fresh crossover would pin MONOLITH for tiled_fits=False islands and decomposed for tiled_fits=True. But that
encodes a WORKAROUND for the scalar-seed regression above; once the per-island-sizing fix lands, decomposed should
win broadly and the pin simplifies to "decomposed for islands". So the pin re-tune should FOLLOW the fix, not the
current (bugged) crossover. Left disabled (islands on the autotuner) for now.

## ARCHITECTURE (read before touching anything)
Two solve "arms", selected by `@qd.perf_dispatch func_solve_body` (registered in solver_breakdown.py):
- **monolith** (`func_solve_body_monolith`, solver.py): ONE @qd.kernel, env-packed (`block_dim=32`, 32
  envs/warp), scalar factor per env. Single GPU-side `for _ in range(iters)` loop (no per-iter launch).
  Used when `prefer_decomposed_solver != 1`. **REQUIRED for `requires_grad`** (decomposed excludes grad).
- **decomposed** (`func_solve_decomposed` -> `_kernel_solve_graph`, solver_breakdown.py): `graph_do_while`
  GPU-side loop; per-iteration (env,island)-tiled factor `func_island_tiled_factor_solve_all`. Cooperative
  (lane-parallel) vector ops. Used when `not requires_grad and prefer_decomposed != 0`. THE island solver.
- `prefer_decomposed_solver` (RigidSimStaticConfig): -1 auto, 0 monolith, 1 decomposed. It ONLY gates which
  candidates are REGISTERED/compatible (monolith is_compatible: `!= 1`; decomposed: `not requires_grad and
  != 0`). It is NOT the running arm and MUST NOT be used to detect it (see PERF_DISPATCH below).

### !!! PERF_DISPATCH ARM SELECTION (the fact I kept getting wrong) !!!
- The running arm is chosen by `@qd.perf_dispatch func_solve_body` (quadrants lang/_perf_dispatch.py). For
  `prefer == -1` BOTH arms are compatible and the autotuner picks the FASTEST after timing them.
- **During warmup the dispatcher runs DIFFERENT arms on consecutive steps** (to time each). So across steps
  the arm VARIES for the same config.
- **`func_solve_init` (solver.py:300) is launched in `resolve()` BEFORE the dispatch (solver.py:310), so it
  CANNOT know which arm runs this step - the arm may even differ step-to-step during warmup. `func_solve_init`
  MUST be arm-INDEPENDENT.** Branching it on the arm (my Q2: `prefer == 1`) is incoherent and breaks whichever
  arm the autotuner actually ran. Forcing the arm for tests/bench is via `QD_PERFDISPATCH_FORCE`
  (env, read at import) or by registration gating, NOT by reading prefer_decomposed_solver at runtime.
- **CORRECT pattern (per user):** each dispatch ENTRYPOINT (`func_solve_body_monolith`, `func_solve_decomposed`)
  statically KNOWS its arm. Anything arm-dependent must be reached FROM the entrypoint, forwarding a hardcoded
  `qd.static` bool downstream. `func_solve_init` is pre-dispatch => it can hold only arm-INDEPENDENT work; any
  arm-specific init must live in / be owned by the entrypoint.

### KEY INVARIANTS / GOTCHAS (these bit me repeatedly)
- **Tests default to CPU.** Decomposed arm is GPU-only (uses `block.sync`, fails on `Arch.arm64`). So CPU
  green says NOTHING about the decomposed arm. ALWAYS validate islands with `--backend gpu`.
- **Hibernation needs the per-island factor.** The per-island path SKIPS hibernated islands
  (`is_hibernated`). A whole-env factor includes asleep dofs and MOVES them -> boxes wake -> hibernation
  tests fail. (This broke commit f6f7f2ffc.)
- **[VERIFIED BY PROBE 2026-06-20] The decomposed arm has NO self-init; it free-rides on func_solve_init for
  its initial Newton direction.** The graph (solver_breakdown.py _kernel_solve_graph) does the LINESEARCH
  FIRST (line ~997, consumes constraint_state.search) and only computes the direction at the END of each
  iteration (_func_update_search_direction ~1061). So iteration 1's linesearch uses func_solve_init's search.
  If func_solve_init skips the gradient (is_decomposed=True), search==0 -> DEGENERATE first linesearch ->
  cost explodes (probe: 47k->13M over ~16 steps) -> bodies fall. The MONOLITH is immune because its kernel
  SELF-INITS the direction (factor_batch+gradient_batch+search, solver.py ~4789) before iterating.
  PROBE METHOD that finally worked: instrument func_solve_init to dump i_b=0 (use qd_to_torch / read fields
  host-side and print at END, NOT in-kernel print() - that forces a per-step GPU sync) and run the REAL test
  (conftest disables hibernation + sets prefer_decomposed_solver=int(backend!=cpu)). Earlier standalone probe
  FAILED to repro because it left hibernation ON (default), which froze the bodies and masked divergence
  (full-then-strip: I stripped the conftest conditions). The dump also showed dec=0 for ON / dec=1 for OFF,
  i.e. the test runs the MONOLITH for islands-ON and the DECOMPOSED arm for OFF (arm pinning, see
  rigid_solver.py ~615) - so the failure is the decomposed arm lacking a self-init, NOT an ON/OFF asymmetry.
- **THE IDEA IS STILL VALID (user, explicit): forward is_decomposed and let the decomposed arm own its init.**
- **[VERIFIED BY PROBE 2026-06-20, forced decomposed + hibernation OFF] ON and OFF are BIT-IDENTICAL and BOTH
  DIVERGE with the init skipped** (z 0.0490->0.0294 sinking+accelerating, both). So the decomposed ON/OFF
  invariant HOLDS; "island ON works in the test" is ARM SELECTION (test runs the MONOLITH for ON), NOT the
  decomposed arm tolerating the skip. COROLLARY: the decomposed island-ON path had a PRE-EXISTING latent
  divergence before Step 1 too (func_solve_init already skipped its seed for GPU-island) - masked by tests
  using the monolith for ON + benchmarks not asserting physics. The old "decomposed ON 0.78 < OFF 0.96" was a
  DIVERGING step vs a CORRECT step, not a parity gap. func_solve_init's factor is NOT redundant: graph is
  linesearch-first, so func_solve_init = iteration-0 seeding iter-1's linesearch; graph iter-1 seeds iter-2.
- **FIX (commit pending): func_solve_init ALWAYS seeds the decomposed arm.** Factor gate
  `Newton and (is_decomposed or not(use_contact_island and backend!=cpu))`; gradient gate skip only
  `Newton and not is_decomposed and use_contact_island and backend!=cpu` (monolith GPU-island self-inits).
  Fixes Step-1 OFF regression AND the pre-existing decomposed ON latent divergence => correct ON==OFF parity.
  Perf note: decomposed ON goes 0.78 (diverging) -> ~0.96 (correct, seeded). The seed factor is genuinely
  needed (no self-init in the decomposed kernel). A FOLLOW-UP could make the graph self-seed (compute
  direction before the first linesearch) to drop the seed factor - separate, riskier, optimize later.
- **The env-packed monolith CANNOT host the `block_dim=T` cooperative tiled factor** in its per-env loop.
- **CPU has no tiled factor** (tiled = GPU). So tiled-path changes are CUDA-only-validatable.
- On CUDA, `use_contact_island` defaults FALSE (rigid_solver.py ~259, `gs.backend != gs.cuda`); on CPU/Metal
  it defaults True. So the CUDA default (and `test_solve_correctness` OFF) is islands OFF.
- Quadrants: top-level `for` is auto-parallelized; `block_dim` is per-loop via `qd.loop_config`. A kernel
  CAN have multiple top-level loops with different block_dims.

## MECHANISMS (the two open questions, verified in code)
### Q1 — why monolith ON ≠ OFF for a SINGLE island
`func_hessian_and_cholesky_factor_incremental_batch` (solver.py ~2924):
`if use_contact_island: func_hessian_and_cholesky_factor_direct_batch(...); return False` — i.e. islands do a
FULL rebuild + SCALAR Cholesky EVERY Newton iteration (comment: "incremental rank-1 update assumes a single
dense system, so islands rebuild directly"). OFF does cheap incremental rank-1 updates. So for the SAME
48-dof matrix, ON rebuilds-from-scratch (scalar) each iter; OFF updates incrementally. The single island
doesn't reduce work — the `use_contact_island` flag just selects a worse factor strategy. (Plus init: OFF
tiled via func_solve_init, ON scalar via monolith self-init.)
DATA: stack 1 island, monolith ON 4.5ms(1env)/20.6ms(256env) vs OFF 1.09/2.78. At 1 env init is negligible
=> the gap is the per-iteration full-rebuild.

### Q2 — why decomposed ON < OFF for a SINGLE island
Decomposed Newton branch is gated on `hessian_fits_shared` (NOT use_contact_island), so ON and OFF take the
IDENTICAL per-island tiled solve in the graph loop. The only difference is `func_solve_init`: its factor gate
is `Newton and not(use_contact_island and backend != cpu)`. For OFF (use_contact_island=False) it RUNS a
whole-env tiled factor+gradient; for ON (GPU island) it SKIPS. So ON does ~0.18ms LESS work per step => ON<OFF.
DATA: decomposed ON 0.78/0.85 vs OFF 0.96/1.20 (1env/256env), gap ~ the init factor cost.
[CORRECTION 2026-06-20] That init is NOT redundant for OFF: skipping it (exp #10) makes OFF DIVERGE. So OFF
does that work because it MUST; ON skips it and survives for an as-yet-UNEXPLAINED reason (see VERIFIED
invariant on the asymmetry). Closing the gap requires understanding that asymmetry by PROBE, not skipping.

### SHARED ROOT
Two things keyed on `use_contact_island` that shouldn't be:
1. Incremental update disabled for ANY island (Q1) — but a SINGLE island IS a single dense system; incremental
   is valid there.
2. `func_solve_init` init-factor skip keyed on islands not on the ARM (Q2) — the factor is always redundant
   for the decomposed arm, always needed by the monolith.

## FIX PLAN
- **Q1 [DONE, partial, KEPT]**: in func_hessian_and_cholesky_factor_incremental_batch,
  `do_full_rebuild = (n_islands[i_b] > 1) or use_hibernation`; else incremental dense (single island = single
  dense system). CPU-correct (10/10). Monolith stack 4.5/20.6 -> 4.02/13.59 ms. Gap to OFF NOT closed; residual
  is the scalar self-init + per-island indirection in the monolith vector ops. NEEDS forced-monolith CUDA
  correctness check (suite uses decomposed for ON).
- **Q2 [CONFIRMED PLAN 2026-06-20, user-approved]**: the init is arm-specific but func_solve_init is
  pre-dispatch (can't know the arm). FIX = move the func_solve_init launch OUT of resolve() INTO the two
  dispatch entrypoints, each forwarding a hardcoded `is_decomposed: qd.template()`:
    * func_solve_decomposed (Python): func_solve_init(is_decomposed=True) -> skip factor+gradient (the graph
      rebuilds on its first iteration regardless - see solver_breakdown.py:767). Identical for ON and OFF =>
      decomposed ON==OFF.
    * func_solve_body_monolith: convert to a thin PYTHON WRAPPER (like func_solve_decomposed) that launches
      func_solve_init(is_decomposed=False) -> TILED factor+gradient -> then the renamed _kernel_solve_monolith.
      This REPLACES the monolith's scalar GPU-island self-init (~solver.py:4794) with the tiled init for BOTH
      ON and OFF (also helps Q1's monolith stack residual).
  func_solve_init factor gate becomes `Newton and not is_decomposed`; gradient gate `not (is_decomposed and
  Newton)`. Partition stays in resolve() (arm-independent). The init becomes PART of each candidate, so it
  matches the arm the dispatcher actually runs - even mid-warmup. This is exp #6 done CORRECTLY (it broke
  before because the skip was applied to a pre-dispatch kernel while the monolith - needing the factor - was
  the arm that actually ran).
- The FULL monolith parity for many islands / requires_grad needs the (env,island)-tiled factor in the
  monolith = converging onto the decomposed body (Python-loop _kernel_solve_graph). Substantial; deferred.
- **Step 3 [ATTEMPTED, BROKE, REVERTED c2526d8e5 -> back to a42db7707]**: drop the decomposed seed factor by
  rotating the graph to direction-first (factor->search->linesearch->apply->refresh). BROKE CUDA (14 fail,
  forced-decomposed probe diverges 0.049->0.029) AND Metal (14 fail). ROOT CAUSE (verified): the convergence
  check lives in `func_terminate_or_update_descent_batch` (solver.py ~4315) which sets
  `improved = grad_norm>tol AND improvement>tol` where `improvement = prev_cost - cost`. It is
  IMPROVEMENT-BASED, so it MUST run AFTER the move (linesearch) to measure cost reduction. Direction-first
  runs it BEFORE the move => improvement=0 => improved=False => search never updates => linesearch gated off
  => divergence. To make direction-first work you must SPLIT terminate_or_update_descent into (a) update-search
  (after factor) + (b) check-termination (after move) and re-thread the `improved` gate across iterations -
  another invasive change to the convergence core, for a payoff of ONE factor (~0.1ms on an already-0.8ms arm).
  NOT WORTH IT this session. DEFERRED. The seed factor is NOT wasted per se - the linesearch-first structure
  pairs it with the improvement-based termination cleanly. If revisited: do the split, validate on BOTH CUDA
  and Metal (Metal must not regress - user requirement), use benchmarks/probe_decomp.py (forced decomposed,
  hibernation OFF) for the divergence check.

## VALIDATION RECIPE (cluster)
- Code is editable locally; the decomposed/tiled path is GPU-only -> validate on CUDA cluster.
- Sync: `scp <file> genesis-coreweave:/mnt/home/duburcqa/workspace/src/genesis/<path>` then md5-compare.
- Run (login node, NOT inside gs-srun for git):
  `ssh genesis-coreweave 'bash -lc "gs-srun --partition=rtx-high --nodes=1 --gpus=1 bash -ilc \"bash <driver> >| <log> 2>&1\""'`
  NOTE: login shell has `noclobber` -> use `>|` not `>`. Do NOT pipe the background launch through tail/grep
  (it truncates the captured output).
- Drivers in benchmarks/: validate_island.sh (suite+bench), run_island_bench.sh (stack 2x2x2), scale_island.sh
  (grid island-count scaling), prof_monolith_island.sh (qd.profiler per-kernel). bench_island_vs_tiled.py
  `<n_bodies> <0|1 island> [grid|stack]`, env BENCH_ARM(0 mono/1 decomp)/BENCH_ENVS.
- Correctness: `pytest tests/test_rigid_physics_island.py --backend gpu -n 8` (26 tests). test_solve_correctness
  asserts ON positions == OFF positions (tol 5e-3).

## BENCHMARK DATA (stack=1 island; grid=N islands; CUDA, 8 bodies unless noted)
Stack OFF/ON (ms), envs 1 / 256:
  decomposed OFF 1.13/1.37  ON 0.78/0.85   (ON<OFF = Q2 redundant init)
  monolith   OFF 1.09/2.78  ON 4.5/20.6    (ON>>OFF = Q1 full-rebuild scalar)
Grid island-count scaling (1 env, islands ON, ms) 8/32/128 bodies:
  decomposed: 0.76 / 1.16 / 3.13   (near-flat, tiled per-island — THE GOAL)
  monolith:   2.2  / 8.0  / 38.7   (per-island scalar, scales linearly; was 3.1/45/1853 with force_whole_env)

## ARM SELECTION (verified 2026-06-20 - resolves the "monolith for ON" confusion)
- The running arm is `prefer_decomposed_solver` -> perf_dispatch is_compatible (monolith `!=1`, decomposed
  `not grad and !=0`). With `prefer=1` only decomposed is compatible; the dispatcher runs it unconditionally
  (lang/_perf_dispatch.py:305 - NO correctness/NaN fallback).
- CONFTEST forces `prefer = int(backend != cpu)` = 1 on gpu => decomposed for BOTH islands ON and OFF.
- The island ARM-PINNING block in rigid_solver.py (~629 `elif False: # self._use_contact_island`) is DISABLED.
  If it were ENABLED, `max_islands <= 16 -> prefer=0` (monolith) for small-island scenes.
- **The earlier "ON runs monolith (dec=0)" trace was a STALE CLUSTER rigid_solver.py** (pinning enabled in the
  stale copy -> ON=monolith for <=16 islands, OFF=decomposed). I had not scp'd rigid_solver.py this session.
  After syncing the current file (pinning disabled), ARM-PROBE confirms `prefer=1` and the dispatcher logs
  "chose func_solve_decomposed ... Only 1 was compatible" for BOTH ON and OFF on CUDA and Metal. LESSON
  (again): sync ALL touched files to the cluster before trusting a run (feedback_verify_runs_not_frugal).
- NET: current code runs the DECOMPOSED arm for islands ON and OFF in the unit tests; the always-seed fix is
  correctly exercised (26/26).

## EXPERIMENT LOG (chronological; DO NOT repeat these)
1. Route islands through `_kernel_solve_graph` + delete dead host-loop island solver. WORKS (d17e8343b).
2. Port incremental H-patch to decomposed island path (do_assemble flag, factor-only reads nt_H). WORKS (6fb3c37a4).
3. Islands-OFF via static 1-island partition (host numpy init in ConstraintSolver.__init__). WORKS (84703b294 superseded).
4. Decomposed gate -> `hessian_fits_shared` (drop use_contact_island/enable_tiled): ON/OFF same decomposed path. WORKS.
5. Delete monolith warp-per-env path (func_solve_islands_tiled_body/iter_warp/factor_solve_all_coop). WORKS.
6. [BROKE] func_solve_init skip ALL GPU (backend==cpu only factors) + monolith self-init: islands-OFF free-fell
   (4 fail). OFF picks a GPU arm needing the seeded first search. Fixed -> skip only GPU-ISLAND (f24eede76).
7. [BROKE] f6f7f2ffc: func_solve_init WHOLE-ENV factor for islands (force_whole_env) so monolith uses OFF's
   tiled init: 12 fail — hibernation (whole-env moves asleep dofs) + decomposed (whole-env init breaks its
   per-island graph). REVERTED (b52152b5c). LESSON: per-island factor is load-bearing for hibernation.
8. [BAD PERF] 6b04a5e7a: force_whole_env template threaded through 8 factor funcs so monolith factors whole-env.
   26/26 but monolith grid CATASTROPHIC O(total_dofs^3): 128 islands = 1853 ms. REVERTED (2fe9093e9).
   LESSON: "nothing forced whole-env" — whole-env is cubic in TOTAL dofs; per-island is the point.
9. [KEPT, partial] Q1: single awake island uses incremental dense in the monolith (not per-iter full rebuild).
   func_hessian_and_cholesky_factor_incremental_batch: do_full_rebuild = (n_islands>1) or use_hibernation.
   CPU-correct (10/10). Monolith stack 4.5/20.6 -> 4.02/13.59 ms. Did NOT close the gap to OFF (1.30/2.78).
   NOT yet validated for monolith correctness on CUDA (suite uses decomposed for ON - see DISCIPLINE).
10. [BROKE, REVERTED] Q2: skip func_solve_init factor+gradient also when prefer_decomposed_solver==1, on the
    theory the decomposed arm's init is redundant. 4 fail (test_solve_correctness OFF free-falls). DIRECTLY
    CONTRADICTED exp #6 (already logged: OFF needs the init). Reverted both gates. Premise was a guess.
11. [DONE Step 1, CPU 10/10, CUDA pending - commit 2282e0cd3] is_decomposed plumbing (user-approved arch).
    func_solve_init gains `is_decomposed: qd.template()`; the init launch moved OUT of resolve() INTO the two
    dispatch entrypoints, each forwarding its hardcoded constant:
      - func_solve_body_monolith is now a PYTHON WRAPPER: func_solve_init(is_decomposed=False) -> _kernel_solve_monolith
        (renamed from the old kernel). Monolith factor/gradient gate unchanged (still skips GPU-island where it
        self-inits) => monolith behavior IDENTICAL to before. Zero monolith risk.
      - func_solve_decomposed: func_solve_init(is_decomposed=True) -> skips init factor+gradient (graph rebuilds
        iter-1 regardless). Same for ON and OFF => decomposed ON==OFF (OFF no longer pays the redundant init).
    KEY CORRECTION to exp #6/#10: the earlier breaks were NOT "decomposed OFF needs the init". They were the
    skip applied to a PRE-DISPATCH kernel while the dispatcher actually ran the MONOLITH (which needs it). With
    the bool forwarded from the real entrypoint, each arm's init matches the arm that runs - no warmup race.
    EXPECTED CUDA: 26/26; decomposed stack OFF drops from 0.96/1.20 to ~ON (0.77/0.86); monolith unchanged.
12. (NEXT, Step 2) Unify the monolith init onto func_solve_init's tiled func_hessian_and_cholesky_factor_direct
    (it HAS a GPU island branch, hibernation-aware, at solver.py:2750) and DELETE the monolith self-init
    (~solver.py:4786). Needs a FORCED-MONOLITH (prefer_decomposed_solver=0) CUDA correctness check vs OFF - the
    suite uses decomposed for ON so it does NOT cover the monolith GPU-island init path.

GIT: history has the bad commits + reverts (6b04a5e7a, f6f7f2ffc + reverts, ff4109086 Q1+Q2, then Q2 revert).
SQUASH before pushing. NOT pushed.
