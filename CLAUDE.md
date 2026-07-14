# Genesis Development Guidelines

## Miscellaneous

* getattr / hasattr should be prohibited in favor of None initialization and isinstance check
* external / non-owned instances are being hijacked to added attributes on the fly. This is strictly forbidden.
* Quadrants kernels are used whereas interop data are numpy arrays on CPU. It would be much faster and simpler to keep everything as torch tensors or numpy arrays from the start. Numpy implementation should be fully vectorized. Numba (single thread, CPU backend) should be used for computation heavy "kernels".
* Data preallocation based on max size is prohibited. The exact memory size that is needed should be allocated. Nothing more.
* Plain dict for packing attributes is prohibited. Strongly typed named data structures should be used instead, either via dataclasses or just simple NamedTuples.
* Local imports in functions is prohibited, unless strictly necessary to avoid circular dependencies.
* There are many prints that are commented out. This is prohibited. They should be converted to logging debug traces.
* There are code duplication everywhere. This should be avoided.
* Some variable names are not following our naming conventions. See Rigid engine for reference. You can also have a like to the description of PR#1053 (https://github.com/Genesis-Embodied-AI/Genesis/pull/1053).
* Domain objects are only 'entity', 'link', or 'geom'. Never invent a new noun ('body', 'object', 'piece') when one of these fits. This is also a correctness cue: the rigid-body unit is the link, so group geoms per link (iterate entity.links then link.geoms), not per entity.
* Never add a cast - float() / int() / .astype() / np.asarray() / dtype= - unless strictly necessary. numpy scalars index, compare, and do arithmetic fine; cast only where an external interface forces the dtype (igl float64/int64, native Python for kernel args), and never re-cast what a later step already casts.
* Many helpers that are defined in genesis/utils/misc.py and genesis/utils/geom.py are not used. This is a major flaw. In particular, 'detach().cpu().numpy()' should never be used and 'tensor_to_array' should be prefered.
* Calling Quadrants data instance method 'to_numpy()' is forbidden. 'qd_to_(torch|numpy)' should be used instead. Take advantage of torch-based zero-copy as much as possible to update the internal state of other solvers. Beware, numpy-based zero-copy is only supported on CPU backend, that is why relying on torch is probably preferable.
* Catch-all exceptions is prohibited, except for exception forwarding in multi-processing process. Moreover, critical errors should not be replaced by warnings. If something breaks the physics, it should just raise an exception, eventually via re-raise to clarify the error message.
* Direct use of numpy / torch / quadrants data types is prohibited, unless strictly necessary due to external interfaces enforcing specific dtypes. Our types are defined in genesis/__init__.py. You may also take inspiration / relying on our quadrants dataclasses dtypes defined in genesis/utils/array_class.py.
* Defining local-scope functions should be avoided unless very well motivated.
* attributes-like instances methods (eg has_any_rigid_coupling) should be converted to properties.
* scene.sim.coupler should not be considered part of public API. It cannot be expected from the user to call 'set_link_ipc_collision'. It should be replaced by some entity-level material options.
* Warnings for nan is not acceptable. Simulation should be halted. The correct way to do this is leveraging our existing errno mechanism. Define a new error code if necessary.
* Remove all legacy code. We do not handle deprecation / legacy compatibility for now since this coupler is still experimental.
* Unused helpers should be removed (eg has_coupling_type). No need to implement private helpers proactively if they would be easy to add later on.
* Prefer using dev getters/setters API in various solvers over directly accessing quadrants fields, mainly for maintainability concerns. Quadrants fields may be renamed or even removed, while getters/setters should be considered stable dev API.
* A quantity derived from several raw solver fields belongs in a single vectorized solver getter, not assembled by the consumer nor extracted via a bespoke per-element `@qd.kernel` that the caller then re-indexes. This keeps field-access coupling and batched / non-batched handling in one tested place.
* Batching based on batched / non-batched should be avoided. Deal with it at argument-level 'envs_idx=env_idx if batched else None'.
* Do not call getters/setters repeatedly for even env idx. This is inefficient. Deal with all envs at once as much as you can, even if it is necessary to fallback to for-loop for IPC-related getters/setters. Notably, calling 'qd_to_numpy' in tight loop is not acceptable. It should only be done once. Moreover, if it is absolutely necessary, then you should specify the slice to extract in input argument of 'qd_to_numpy' instead of slicing manually afterward.
* Avoid generic 'all_' prefix/suffix for containers if possible but rather be specific, eg 'joints_xanchor', 'links_inertia_i', 'geoms_pos', 'entities_quat', 'verts_idx'.
* Avoid pure varilable name assignment, eg 'mass_mat_env = mass_mat_all'.
* Prefer clear over fresh allocation for mutable containers, eg 'self.abd_data_by_link.clear()' instead of 'self.abd_data_by_link = {}'.
* Never use `flatten()`, `reshape((-1,))`, or `ravel()` in place of squashing. Moreover, prefer `[..., 0, :]` over `squash(axis=-2)`.
* Don't use np.asarray unless strictly necessary, ie for public API with no control on the data source. For internal dev or private code path, it should NEVER be the case.
* Don't use np.asarray unless strictly necessary, ie for public API with no control on the data source. For internal dev or private code path, it should NEVER be the case.

## Priorities

- **Code quality over source consistency.** When refactoring or porting code from PRs, external projects, or prior implementations, Genesis conventions always take priority over the original code's patterns. Use the original logic as reference, not the implementation style.
- **Build/init time: maintainability first** (as long as performance is reasonable). Use public APIs.
- **Runtime hot paths: efficiency first.** Minimize GPU-CPU transfers, use bulk operations.

## Data Access

- **Build/init time:** Iterate via entity/link/geom public API (`entity.links`, `link.geoms`, `geom.init_verts`, `link.get_pos()`, etc.). Only fall back to low-level solver field access when efficiency, conciseness, or maintainability specifically requires it.
- **Runtime:** Use `qd_to_numpy` / `qd_to_torch` (from `genesis.utils.misc`) for bulk GPU-CPU transfer. Never use `.to_numpy()` / `.to_torch()` directly. Never item-access qd fields (`field[i]`).
- **Always pass `transpose=True`** to `qd_to_numpy` / `qd_to_torch` to move the batch dimension to front (`[B, n, dim]`), aligning with public getter conventions.
- **Hoist conversions before loops.** Call `qd_to_numpy` / `qd_to_torch` once, then index the local array.
- **`qd_to_numpy` with `copy=False` fails on CUDA.** GPU data always requires a copy to numpy.
- **Do not pass `copy=` at all** to `qd_to_torch` / `qd_to_numpy` when the value returned to the caller is fresh arithmetic (nothing internal aliases out) — let it zero-copy or copy only as needed. Reserve `copy=True` for when a raw field view would otherwise escape and could be mutated.
- **Store Python references at build time** (entity, link objects in dicts) for runtime lookup, rather than re-deriving from solver fields every step.
- **Never pass numpy scalar types** (e.g., `numpy.float64`) to kernel arguments. Always cast to native Python types (`float()`, `int()`). Numpy scalars break quadrants fastcache (weak reference).
- **Never use `np.asarray` on torch tensors.** Use `tensor_to_array` from `genesis.utils.misc` instead, which properly handles GPU→CPU transfer via `tensor_to_cpu` first. Pass its `dtype` argument (`tensor_to_array(x, dtype=...)`) rather than chaining `.astype`, and convert once inside the consuming function, not at each call site.

## Kernels

- **New code:** Free function `@qd.kernel`, no `@qd.data_oriented`. Use `V_ANNOTATION` from `genesis.utils.array_class` for type-polymorphic parameters.
- **FEM solver:** Follows old `@qd.data_oriented` method pattern. Any kernel added to FEM solver must stay consistent with this.
- **No keyword arguments** when calling kernels from Python (slow). Use positional args.
- **Rigid solver** is the reference implementation for kernel and code quality standards.

## API Design

- **Don't force users to set options with a single correct value.** If an option has only one valid value in a given context (e.g., `enable_collision=False` is always required with IPCCoupler), set it automatically and raise an error if the user explicitly provides the wrong value. The right pattern is: default to `None`, resolve to the correct value at init time, raise on conflicting explicit value. The resolution logic can be based on other options, scene contents, or even runtime state — as long as it is well documented.
- **No single user helper.** Do not introduce any extra helper if it is used at once single place for one specific context. It is only acceptable to introduce a new single-use helper if it is standalone and completely generic, like some math function and conversion utility. Anything beyond that is fishy. In particular, avoid private instance methods, they are almost always a bad idea.
- **A unit-valued return must not be wrapped in a named structure.** Returning a one-field NamedTuple/dataclass where a bare value suffices is dead boilerplate; only introduce a named structure for genuinely multi-field returns.
- **Option fields for index/array inputs must use the project's array-like type aliases** (`IArrayType` / `OptionalIArrayType` / `FArrayType` / `Vec3FType`, etc. from `genesis.typing`), never a bare `tuple[int, ...]` / `list[...]`. `Options` is strict Pydantic (`ConfigDict(strict=True)`), so a bare typed-collection field rejects the lists and numpy arrays users naturally pass (`dofs_idx_local=[0, 1]`); the aliases carry `strict=False` to coerce array-like inputs to a tuple. Match how sibling option fields are typed (e.g. `filter_link_idx: OptionalIArrayType`).

## Style

- **Docstrings describe the current state, not history.** Write what a function does now, not what it used to do or what was added. Do not write "Preserves the original X scenario" or "Added Y for Z". If a behavior is required for a bug fix, explain why in terms of the current invariant (e.g., "Must handle Z case to avoid Y").
- **User-facing option docs state the TRADEOFF, stand alone, and stay behavioral.** An option's doc must let the user decide: give cost AND benefit and when to pick each value, not just what it does (if one setting looks strictly better, the downside is missing - the doc is useless). State the real cost in user terms (if speed is not it, say what is, e.g. numerical robustness). Explain it in Genesis terms only - no external-engine references (e.g. MuJoCo) and no internal/implementation mechanism (constraint "rows", "coupling", "cone projection", kernels, factor paths); those belong in dev code comments. Keep it compact.
- **Comments justify - generically.** A comment's purpose is to explain why the code is the way it is: the invariant it preserves, the failure mode the tempting alternative runs into. Keep the justification generic and present-tense; never anchor it to perishable specifics (measured values, benchmark figures, particular unit tests or examples) - those go stale long before the mechanism does.
- **State what IS, never what is NOT.** In comments, docstrings, and reports, drop "not X", "unlike Y", "does not ...", "it is X, not Y" unless the negation was explicitly asked for or is the literal spec. If a property is not mentioned, it does not exist - defending against an unraised concern is noise. Say the positive fact and stop.
- **Function-level description goes in a docstring, never a block of leading `#` comments** at the top of the body. Reserve inline comments for non-obvious implementation details at their point of use (e.g. a constraint-layout invariant a few lines rely on). A getter/method that opens with a paragraph of `#` prose is wrong; that prose is its docstring.
- **No local imports** unless strictly necessary (e.g., circular dependency avoidance). All imports at module top level.
- No import aliasing (`Cloth` not `ClothMaterial`).
- No temporary variables for `isinstance` checks. Use `isinstance(...)` directly in `if` statements.
- **Pass an anonymous literal by keyword** so its meaning shows at the call site (`_adaptive_params(verts, faces, aggressiveness=7)`, not a bare `7`). Positional is fine for self-named variables. Kernel calls are the exception - positional only (see Kernels).
- **Booleans read as predicates** - every boolean variable/field/attribute takes an `is_` or `has_` prefix, present tense preferred (`is_fixed`, `is_convex`, `has_multi_island_structure`). `was_` only for a genuine past-tense flag (prefer `is_cached_loaded` over `was_cached`). A bare name (`hibernated`, `placed`) or a `did_` prefix (`did_fuse`) is not valid.
- No `@qd.data_oriented` on non-solver classes (materials, couplers, entities).
- Naming consistency within directories (e.g., `ipc_*.py` for all IPC examples).
- **ASCII only.** No non-ASCII characters anywhere in source (code, comments, docstrings): write `tau`/`qddot` not `τ`/`q̈`, `*` not `·`, `-`/`--` not `—`, plain `# ---` not box-drawing separators.
- **Spell out acronyms on first use** in each docstring/comment, verbatim with the acronym in parentheses, e.g. "positive semi-definite (PSD)", "second-order cone (SOC)". After that first spell-out the acronym alone is fine within the same docstring/comment.
- **No unverifiable / stale external references.** Do not justify a design by citing external projects a reader cannot verify. When porting a known feature (e.g. MuJoCo's elliptic cone), reference it at the right granularity - name the feature, optionally the official doc section, and/or the math formula - but never the external repo's specific file / function / constant names (they are cross-repo and go stale). "matches MuJoCo's elliptic cone" good; "matches MuJoCo's `ellipticCostDif`" bad.
- **No torch dtype casts** (`.long()`, `.to(torch.int64)`, etc.). For advanced indexing use `indices_to_mask` (from `genesis.utils.misc`); for a `torch.gather` index, let `cumsum` over a boolean mask yield int64 instead of casting.
- **No useless or circular comments.** Do not restate what names/types/control-flow already convey, and never write tautologies ("`dt` is the timestep, so it must match the simulated timestep").
- **Match the established sibling pattern; do not invent structure.** Before adding any helper, fixture, option, cast, comment, or file layout, confirm a sibling in the same directory already does it that way and copy it. Most review friction here comes from deviating from the existing convention (e.g. `imu.py` / `imu_franka.py` / `test_imu_sensor` / `test_rigid_physics.py`), not from genuine novelty.

## Testing Guidelines

- **Never filter or truncate test output inline** (no `pytest ... | tail` / `| grep`). Redirect the full output to a log file, then extract from the file: a filter between pytest and disk destroys the only copy of the failure names and masks the exit code.
- **Never remove or weaken an existing assertion or measurement to silence a failure** (local or CI). Report the failure with the data and ask; a threshold that only holds on one machine is a calibration problem, not a license to delete the check.
- **Bug fix PRs** must include a regression test that fails on `main` and passes with the fix.
- **No deprecation tests.** Do not add unit tests that verify deprecation warnings are emitted.
- Feature tests should exercise the new behavior, not the internal warning machinery.
- **No docstrings in unit tests.** Use code comments only where the intent is non-obvious; never state regressions or history.
- **No dedicated single-feature test module.** Add new tests to the existing comprehensive file for that domain (e.g. `tests/test_sensors.py`), not a fresh `test_<one_thing>.py`. Complements "Pack tests" below.
- **Pack tests.** Prefer one comprehensive scene with diverse entities and options over many small single-option tests. Add entities with different configurations to the same scene instead of creating separate test functions.
- **Assert physics, not just execution.** "Simulation runs without error" is not a test. Check quantities with physical meaning: free-fall displacement (`z = z0 - 0.5*g*t^2`), no ground penetration (`min_z > -d_hat`), velocity → 0 at rest, contact stops fall.
- **Non-regression fallback.** If no analytical expectation is available, run the simulation once to get reference values, hardcode them, and assert with a loose tolerance. This is much better than checking nothing. Add a `FIXME` comment asking to replace with physics-informed assertions later.
- **Use `assert_allclose` from `tests/utils.py`.** It handles tensors, numpy arrays, and scalars uniformly.
- **FEM entity positions:** `entity.get_state().pos` has shape `[B, n_verts, 3]`. Use `[..., 2]` to select z across all envs and vertices.
- **Rigid entity positions:** `entity.get_pos()` returns `[B, 3]` or `[3]`. Use `np.atleast_1d(...)[..., 2]` and `.all()` for multi-env checks.
- **Parametrize batched tests over `n_envs=[0, 2]`.** Multi-env is where shape bugs hide; never test single-env only. Do not add a dead parametrize dimension that a conftest fixture already controls (e.g. `backend`).
- **Step budget (hard).** Use the strict minimum steps needed, measured not guessed; if the checked quantity is FP-sensitive, pad by whichever is larger: +10% or round up to 50. Per-test horizon tiers: `<100` fine, `100-300` grey (needs justification), `>300` near-prohibited (only 4-5 such tests in the whole suite). Also bound total env-steps (`steps * n_envs`) - a 500-step x 16-env settle (8000) is unacceptable; delete it or fit the budget. CI cost multiplies every step across the matrix.
- **Validate the math, not scale.** One constraint proves the theory (works for 1 => works for 1k); no large/high-DOF scene is needed for correctness. A real multi-body scene earns its steps only when it validates something a single constraint cannot (end-to-end stability, e.g. the bowl tower).
- **Prefer FP-robust checks.** A single constraint-solve comparison (one step from a fixed state) is robust; a multi-step trajectory comparison across numerically-distinct codepaths (different solver arms/factor paths) is not - fp32 compounds and correct paths diverge. Compare a single solve, or an analytical physical property. Cross-engine (MuJoCo) consistency = the real-world check; analytical closed-form = the math check. Regression tests are reactive (add when a bug surfaces), not speculative.
- **Tighten toward physically-exact conditions; slack hides bugs.** A padded safety factor or generous tolerance that makes a scenario pass trivially also masks real behavioral differences (across backends/arms/factor paths). Prefer the tightest thresholds the physics justifies - an over-pushed / loosely-checked scenario passes everywhere while a genuine discrepancy sits underneath. When tightening a test makes it start failing, that failure is signal working as intended: investigate the root, don't back off. And verify a fix against the EXACT assertion path (full horizon, all phases), never a cheaper proxy - the part a proxy drops is where behavior diverges.

### Scene construction in tests and examples

- **One option per line** for every `scene.add_entity` / `scene.add_sensor` / `gs.morphs.*` / `gs.options.*` call, even with a single option. ruff does not enforce this; it must be done by hand.
- **Only strictly-required options, at default values unless a reason forces otherwise.** Never set redundant or default-valued options (`align=False`, `ProfilingOptions(show_FPS=False)`), and never pick a non-default value without a load-bearing reason (e.g. use a smaller `dt` than the default only if precision or stability actually needs it). The one exception: an option whose value the test/example computation reads (`gravity`, `dt`) must be set explicitly even when it equals the default, so the simulated value matches the asserted one.
- **Pass scalars / lists directly.** Setters and IK accept array-like and broadcast a scalar across DOFs; never wrap in `np.full`, `np.array`, or a single-element `[...]` (`control_dofs_force(TAU)`, `set_dofs_kp([...])`, `inverse_kinematics(pos=[...], quat=[...])`). Inline one-shot target vectors instead of naming them.
- **Custom MJCF/URDF models:** build them with `xml.etree.ElementTree` in a fixture that returns `ET.tostring(mjcf, encoding="unicode")`, and pass that straight to the morph `file=` — the loader parses inline XML content. Follow `test_rigid_physics.py`. Never assemble XML via `write_text` / `textwrap.dedent` string literals or f-string concatenation.
- **No module-level test constants, helpers, or parametrize scenario-lists.** Put constants inside the test, build the scene inline, pack diverse configs into one scene (follow `test_imu_sensor`). Read derived params back from the built model (`get_dofs_armature`, `get_dofs_damping`, ...) rather than duplicating literals between fixture and assertions.
- **Do not gate per-step asserts** with `float(...)` / `.item()` casts on tensor reductions. Design the scenario (e.g. a warmup loop that spins joints up) so the assertion holds unconditionally, then assert on the full tensor with `.all()`.
- **Exact analytical dynamics checks:** force `gs.integrator.Euler` so the finite-differenced `qacc` equals the solver's, and account for both rigid-body rotational inertia and the implicit-damping first-order correction (`effective_inertia = I + damping*dt`, as in `test_position_control`) rather than loosening tolerances or distorting geometry.

## Testing on Cluster

- SSH: `ssh genesis-coreweave`
- Path: `/mnt/home/duburcqa/workspace/src/genesis`
- **Git operations (fetch, checkout, pull) must be done on the login node**, not inside `gs-srun`. The compute node container cannot reach GitHub.
- **Always use `gs-srun`** (Slurm wrapper) to allocate a GPU node. Never run pytest or python directly on the login node.
- **Never source a venv manually.** The container image already has the correct environment. The `bash -lc` login shell sets everything up automatically.
- Pattern for git + run: `ssh genesis-coreweave 'bash -lc "cd /mnt/home/duburcqa/workspace/src/genesis && git pull && gs-srun --partition=rtx-high --nodes=1 --gpus=1 bash -ilc \"cd /mnt/home/duburcqa/workspace/src/genesis && pytest -n 10 tests/test_ipc.py -v --no-header 2>&1\""'`
- Example tests need `-m examples` to override the default marker filter in `pyproject.toml`.
- Copy local files to cluster: `scp <local_path> genesis-coreweave:<remote_path>`. Then run with `gs-srun`.
