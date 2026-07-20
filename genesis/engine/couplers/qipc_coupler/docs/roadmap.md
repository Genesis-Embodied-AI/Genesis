# QIPCCoupler Roadmap

## Current Status (2026-07-20)

Multi-entity support with IPC ground contact fully working. Unified writeback kernel, absolute joint position control, free-base qpos writeback, strong-typed code. PR #3043 on `feat/qipc-coupler`.

### What Works

- Multi-entity: N ABD robots + M ground planes in one scene
- IPC contact enabled by default (halfplane ground auto-detection from Plane entities)
- Unified writeback kernel (links_state + dofs_state + free-base qpos in single launch)
- Entity classification upfront (plane vs abd), joint classification by type
- FREE joint handled as free ABD body with qpos writeback
- Fixed-joint merging (parallel axis theorem, relative-transform writeback)
- Home pose via QIPC init_theta (absolute joint-angle frame)
- Per-entity config via `gs.materials.Rigid(qipc_*=...)`
- Actuator gain resolution (MJCF-parsed > material override > defaults)
- Revolute, prismatic, and mixed joint configurations
- IK-based end-effector teleop example
- Strong-typed NamedTuples, full type annotations, no getattr/hasattr
- Stacked free-base collision (passes where IPC coupler xfails)

### What Remains

1. `reset()` implementation
2. `n_envs > 1` support
3. Velocity control mode (requires cuda-graph-qipc kernel change)
4. Per-pair contact tabular (per-entity ContactElement + friction/resistance)
5. Observation API validation (`get_pos`, `get_quat` at entity level)

---

## Next: Velocity Control (Priority 1)

QIPC currently supports position control and direct force control. Velocity
control is needed for bang-bang limit testing, locomotion policies, and Genesis
API compatibility (`control_dofs_velocity`).

### Design (see `cuda-graph-qipc/docs/joint_controller.md`)

Shift the implicit damping energy rest-velocity from zero to `target_velocity`:

$$E_{\mathrm{damp}} = \Delta t \cdot \frac{1}{2}\,k_v\,(\Delta\theta - \dot\theta_{\mathrm{target}}\,\Delta t)^2$$

When `target_velocity=0`, reduces to current behavior. Variational stability
preserved (Hessian structure unchanged).

### Implementation tasks

- [ ] `cuda-graph-qipc`: add `_joint_target_velocity` buffer to `Solver`
- [ ] `cuda-graph-qipc`: modify `revolute_damping_assemble_kernel` to shift `s`
- [ ] `cuda-graph-qipc`: modify `prismatic_damping_assemble_kernel` same shift
- [ ] `cuda-graph-qipc`: add `JointCollection.control_dofs_velocity()` Python API
- [ ] `genesis-world`: forward `dofs_state.ctrl_vel` in `QIPCCoupler.preprocess`
- [ ] `genesis-world`: remove xfail from `test_joint_position_limits_bang_bang`

---

## Priority 2: Test Suite

Restructure tests to `tests/qipc/` following the IPC pattern (`tests/ipc/`), and add physics-asserting test cases.

### Test cases

#### Alignment (current test_qipc.py)

- [x] init state alignment (revolute, prismatic, mixed, fixed-joint merge)
- [x] step alignment (gravity, no control)
- [x] control alignment (target tracking)
- [ ] alignment with home_qpos / init_theta

#### Rigid-body physics

- [ ] `test_freefall` -- object in freefall matches `z = z0 - 0.5*g*t^2`
- [x] `test_ground_contact` -- object on ground, `z > 0` (IPC no-penetration)
- [ ] `test_fixed_base_holds` -- fixed-base robot does not move under gravity
- [ ] `test_merged_body_coherence` -- links merged by fixed joints move as one rigid body
- [x] `test_stacked_free_base_collision` -- multiple free-base entities stack on ground

#### Joint control

- [ ] `test_single_joint_tracking` -- sinusoidal PD target, verify correlation and amplitude
- [x] `test_joint_position_limits` -- bang-bang velocity command respects limits (xfail: velocity control not yet implemented)
- [ ] `test_joint_type_matrix` -- parametrize over revolute/prismatic x fixed/free base
- [ ] `test_actuator_gains_from_mjcf` -- verify kp/kv match MJCF actuator section
- [ ] `test_home_qpos_offset` -- verify theta at init equals home_qpos, control at home_qpos holds pose
- [ ] `test_velocity_control` -- direct velocity control tracking (blocked on Priority 1)

#### Multi-entity

- [x] `test_multi_entity` -- two robots track opposite targets independently

---

## Priority 3: Reset Implementation

Implement `reset()` to restore QIPC scene state:
- Reset ABD q to initial transforms
- Reset joint theta to init_theta values
- Reset target_theta, target_velocity
- Writeback to Genesis

---

## Priority 4: Per-pair Contact Tabular

- Per-entity ContactElement registration
- Pairwise friction/resistance via geometric/harmonic mean (matching IPCCoupler)
- Genesis material `coup_friction` / `contact_resistance` forwarding

---

## Priority 5: n_envs > 1

- Batched simulation support

---

## Resolved

- Multi-entity support (2026-07-20)
- Ground contact via halfplane auto-detection (2026-07-20)
- Unified single-kernel writeback (2026-07-20)
- Free-base qpos writeback from ABD transform (2026-07-20)
- getattr/hasattr cleanup, strong typing (2026-07-20)
- Joint theta direction alignment
- Fixed-joint merging
- Per-entity material config (moved from QIPCCouplerOptions)
- Home pose via init_theta (removed offset hacking)
- Sign-preserving FK rotation (_rodrigues)
- MJCF prismatic FK body-quat bug (fixed in both coupler and cgq)
