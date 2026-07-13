# QIPCCoupler Roadmap

## Current Status (2026-07-13)

**Only initial scene loading is functional.** All runtime features (control forwarding, state writeback, observation API) are in broken/unverified state and require the next agent to fix and validate.

### What Works

- Genesis entity parsing → QIPC ABD body creation (mesh + mass/inertia from Genesis rigid info)
- Joint topology extraction → QIPC joint creation with per-body `axis_left`/`axis_right`
- Mesh in link-local frame with `geo.transforms` for positioning
- Joint pivot/axis correctness verified visually for full Panda URDF (7 revolute + 2 prismatic)
- QIPC viewer integration (`debug_viewer` option, `up_axis = "z"`)
- `JointCollection.merge()` for single-JC batch control
- Quadrants kernel for ABD q → links_state writeback (no host transfer)
- Build-time skip of Genesis compilation step (`_skip_first_step`)
- ~117 FPS for full Panda with viewer

### What Is Broken / Unverified

1. **Joint theta direction alignment** — CRITICAL, first priority for next agent
2. Control forwarding (Genesis `ctrl_pos` → QIPC target theta) — direction may be inverted
3. State writeback (QIPC theta → Genesis `dofs_state.pos`) — direction/sign unknown
4. Observation API (`get_pos`, `get_quat`, `get_qpos`) — unverified against Genesis reference
5. `reset()` — not implemented
6. Links without collision geometry — currently returns `None, None` (should skip gracefully)
7. Home pose initialization — not cleanly supported (theta=0 means init pose in QIPC)

## Priority 1: Joint Theta Direction Alignment

### Problem

Genesis and QIPC may define joint theta rotation direction differently:
- Genesis follows URDF/MJCF convention (right-hand rule around axis)
- QIPC measures theta as signed rotation from initial configuration
- The sign convention and zero-point may differ

### Required Validation

Load the **same URDF** (e.g., `urdf/panda_bullet/panda_2link.urdf`) in both:
1. Genesis with QIPCCoupler (coupler's internal QIPC scene)
2. QIPC standalone (via `scene.add_urdf()` directly)

The correct comparison method: **before `scene.step()`, verify that the QIPC solver's internal state (ABD q, joint theta, target theta, kp, kv) is identical between the two setups.** If the solver input data matches, the solver output must also match — any divergence means the coupler is feeding incorrect data.

Specific checks before first step:
- ABD body `q` (12-DOF per body): same initial transforms
- Joint `target_theta`: same control targets
- Joint `kp`/`kv`: same gains
- Joint `axis_a`/`axis_b`: same axis vectors
- Joint `anchor_a`/`anchor_b`: same pivot positions

If pre-step state matches but theta readback differs, the issue is in the coupler's writeback mapping. If pre-step state already differs, the issue is in the coupler's `build()` or `preprocess()`.

### Test Script Pattern

```python
# Setup A: Genesis + QIPCCoupler
gs_scene = gs.Scene(coupler_options=QIPCCouplerOptions(...))
robot = gs_scene.add_entity(morph=gs.morphs.URDF(file=..., fixed=True))
gs_scene.build()
qipc_scene_A = gs_scene.sim._coupler._scene

# Setup B: standalone QIPC
from qipc import Scene
qipc_scene_B = Scene(dt=0.01, gravity=(0, 0, -9.81))
model_B = qipc_scene_B.add_urdf(file=..., fix_base=True, enable_controller=True, ...)
qipc_scene_B.init()

# Compare solver state before stepping
assert torch.allclose(qipc_scene_A.affine_body.q, qipc_scene_B.affine_body.q)
assert torch.allclose(qipc_scene_A.solver._joint_target_theta, qipc_scene_B.solver._joint_target_theta)
# ... etc for kp, kv, axis, anchor

# Apply same target, step both, compare theta
robot.control_dofs_position([0.5, -0.3])
model_B.control_dofs_position([0.5, -0.3])
gs_scene.step()
qipc_scene_B.step()

# Compare results
theta_A = gs_scene.sim._coupler._jc.get_dofs_position()
theta_B = model_B.get_dofs_position()
assert torch.allclose(theta_A, theta_B)
```

