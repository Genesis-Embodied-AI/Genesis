# QIPCCoupler Roadmap

## Current Status (2026-07-13)

**Core data pipeline is fully aligned with QIPC standalone.** All solver input state (ABD q, joint anchor/axis, kp/kv, target_theta) matches between Genesis+QIPCCoupler and standalone QIPC `add_urdf` with machine-epsilon precision. The full Genesis user-facing pipeline (`control_dofs_position` -> `step` -> `get_dofs_position`) produces identical results.

### What Works (Verified)

- Genesis entity parsing -> QIPC ABD body creation (mesh + mass/inertia from Genesis rigid info)
- Joint topology extraction -> QIPC joint creation with per-body `axis_left`/`axis_right`
- Mesh in link-local frame with `geo.transforms` for positioning
- Joint pivot/axis correctness verified against standalone QIPC
- QIPC viewer integration (`debug_viewer` option, `up_axis = "z"`)
- `JointCollection.merge()` for single-JC batch control
- Quadrants kernel for ABD q -> links_state writeback (no host transfer)
- Build-time skip of Genesis compilation step (`_skip_first_step`)
- Control forwarding (Genesis `ctrl_pos` -> QIPC target theta) — **verified correct**
- State writeback (QIPC theta -> Genesis `dofs_state.pos`) — **verified correct**
- Revolute joints — **verified, all parameters aligned**
- Prismatic joints — **verified, all parameters aligned**
- Mixed joint configurations (revolute + prismatic) — **verified**
- Non-trivial joint RPY (e.g., panda_joint2 with RPY=[-pi/2, 0, 0]) — **verified**
- End-to-end Genesis pipeline alignment (control -> step -> readback) — **verified**
- Fixed joint merging (links connected by FIXED joints fused into single body) — **verified**
- Links without collision geometry — handled via proxy body or mesh-only merge
- Per-link relative transform writeback for merged bodies — **verified**

### Alignment Test Results (123/123 checks pass)

| Test | Checks | Max Diff |
|---|---|---|
| simple_two_link (init) | 17/17 | 0.00e+00 |
| panda_2link (init, RPY) | 18/18 | 4.98e-17 |
| prismatic (init) | 9/9 | 0.00e+00 |
| mixed_joints (init) | 19/19 | 0.00e+00 |
| step alignment (gravity) | 10/10 | 0.00e+00 |
| control alignment (direct) | 20/20 | 2.64e-16 |
| E2E Genesis simple | 10/10 | 2.78e-17 |
| E2E Genesis panda_2link | 10/10 | 1.33e-11 |
| fixed_joint_merge | 17/17 | 0.00e+00 |
| fixed_chain | 18/18 | 0.00e+00 |

Test script: `genesis/engine/couplers/qipc_coupler/tests/test_alignment.py`

### What Remains

1. **`reset()` implementation** — not implemented yet, currently a no-op
2. **Home pose initialization** — not cleanly supported (theta=0 means init pose in QIPC)
3. **Observation API validation** — `get_pos`, `get_quat` may need validation at the Genesis entity level
4. **Multi-entity support** — currently only supports single entity
5. **n_envs > 1** — not supported

## Priority 1: Joint Theta Direction Alignment — RESOLVED

The previous concern about joint theta direction differences between Genesis and QIPC has been resolved. The alignment test suite verifies that:
- ABD body `q` (12-DOF per body): identical initial transforms
- Joint `target_theta`: identical control targets
- Joint `kp`/`kv`: identical gains
- Joint `axis_left`/`axis_right`: identical axis vectors
- Joint `anchor_left`/`anchor_right`: identical pivot positions
- Stepped theta readback: identical after 10 steps with control

The coupler's manual body-local anchor/axis computation produces the same results as QIPC standalone's `anchor_world`/`axis_world` → `_resolve_world_params` pipeline.

## Priority 2: Reset Implementation

Implement `reset()` to restore all QIPC scene state to initial conditions:
- Reset ABD q to initial transforms
- Reset joint theta to zero
- Reset target_theta, velocities
- Sync with Genesis entity init_qpos

## Priority 3: Robust Link Handling — RESOLVED

Fixed joint merging and proxy body support are now implemented:
- Links connected by FIXED joints are merged into single ABD bodies (parallel axis theorem for inertials)
- Jointless MJCF bodies (implicit fixed connection) are also detected and merged
- Meshless links contribute mass/inertia to merged body; fully meshless groups use proxy bodies
- Writeback kernel handles per-link relative transforms for merged members
