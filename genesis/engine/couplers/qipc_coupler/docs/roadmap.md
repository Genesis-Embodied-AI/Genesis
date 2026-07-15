# QIPCCoupler Roadmap

## Current Status (2026-07-15)

Core data pipeline fully aligned with QIPC standalone. Fixed-joint merging, per-entity material config, home pose via init_theta, and actuator gain resolution all working. Draft PR #3043 submitted to `feat/ipc_coupler`.

### What Works

- Entity parsing, ABD body creation, joint topology with correct anchor/axis
- Fixed-joint merging (parallel axis theorem, relative-transform writeback)
- Home pose via QIPC init_theta (absolute joint-angle frame)
- Per-entity config via `gs.materials.Rigid(qipc_*=...)`
- Actuator gain resolution (MJCF-parsed > material override > defaults)
- Revolute, prismatic, and mixed joint configurations
- IK-based end-effector teleop example
- Initial state writeback at end of build()

### What Remains

1. `reset()` implementation
2. Multi-entity support (currently single entity only)
3. `n_envs > 1` support
4. Observation API validation (`get_pos`, `get_quat` at entity level)

---

## Next: Test Suite (Priority 1)

Restructure tests to `tests/qipc/` following the IPC pattern (`tests/ipc/`), and add physics-asserting test cases covering what IPC tests already validate plus QIPC-specific features.

### Structure

```
tests/qipc/
  __init__.py
  utils.py              # shared helpers (scene builders, comparison utilities)
  test_alignment.py     # solver state alignment vs standalone QIPC
  test_rigid.py         # rigid-body physics tests
  test_joint.py         # joint control and limits
```

### Test cases to implement

#### test_alignment.py (migrate from current test_qipc.py)

- [x] init state alignment (revolute, prismatic, mixed, fixed-joint merge)
- [x] step alignment (gravity, no control)
- [x] control alignment (target tracking)
- [ ] alignment with home_qpos / init_theta

#### test_rigid.py (analogous to tests/ipc/test_rigid.py)

- [ ] `test_freefall` — object in freefall matches `z = z0 - 0.5*g*t^2`
- [ ] `test_ground_contact` — object resting on ground does not penetrate (requires QIPC contact)
- [ ] `test_fixed_base_holds` — fixed-base robot does not move under gravity
- [ ] `test_merged_body_coherence` — links merged by fixed joints move as one rigid body

#### test_joint.py (analogous to tests/ipc/test_rigid.py joint tests)

- [ ] `test_single_joint_tracking` — sinusoidal PD target, verify correlation and amplitude
- [ ] `test_joint_position_limits` — bang-bang velocity command respects joint limits
- [ ] `test_joint_type_matrix` — parametrize over revolute/prismatic x fixed/free base
- [ ] `test_actuator_gains_from_mjcf` — verify kp/kv match MJCF actuator section
- [ ] `test_home_qpos_offset` — verify theta at init equals home_qpos, control at home_qpos holds pose

### IPC test gaps (not yet covered by IPC, opportunity for QIPC to lead)

- [ ] `test_stacked_free_base` — IPC marks this xfail; QIPC may handle it differently
- [ ] `test_velocity_control` — direct velocity control tracking
- [ ] `test_multi_entity` — multiple robots in one scene (blocked on multi-entity support)

---

## Priority 2: Reset Implementation

Implement `reset()` to restore QIPC scene state:
- Reset ABD q to initial transforms
- Reset joint theta to init_theta values
- Reset target_theta, velocities
- Writeback to Genesis

---

## Priority 3: Multi-entity and n_envs

- Support multiple entities per scene (iterate over all rigid entities in build)
- Support n_envs > 1 (batched simulation)

---

## Resolved

- Joint theta direction alignment
- Fixed-joint merging
- Per-entity material config (moved from QIPCCouplerOptions)
- Home pose via init_theta (removed offset hacking)
- Sign-preserving FK rotation (_rodrigues)
- MJCF prismatic FK body-quat bug (fixed in both coupler and cgq)
