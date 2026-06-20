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
