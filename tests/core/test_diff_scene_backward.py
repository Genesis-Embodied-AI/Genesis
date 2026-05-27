"""Unit tests for the `scene.backward(loss)` API.

Within the differentiable-rigid test suite, this is the *plumbing* layer —
orthogonal to which physics is active. The other files check that the gradient
is numerically correct; this one checks that the backward *machinery* (state
snapshot/restore, gradient-tape clearing, no grad leak across chunked horizons)
is correct. Siblings:
  - test_diff_forward_kinematics : local FD of the FK + velocity gradient.
  - test_diff_joint_limit        : the joint-limit constraint gradient.
  - test_diff_contact            : the collision / diff-GJK contact gradient.
  - test_diff_scene_backward     : *this file* — scene.backward() + horizon truncation.
  - test_diff_optim              : end-to-end optimization convergence.

`scene.backward(loss)` folds the snapshot → backward → restore dance
(`scene.get_state()` → `loss.backward()` → `scene.reset(snapshot)`) into a single
call; the flagship test below exercises it directly.

The flagship test, ``test_horizon_truncation_matches_independent_scenes``,
runs three scenes in parallel:

    Scene A: single scene, 5-step horizon 1 → ``scene.backward(loss1)``
             (snapshot + backward + restore in one call) → 5-step horizon 2
             → ``scene.backward(loss2)``. Yields ``grad1_A`` and ``grad2_A``.
    Scene B: same as A's horizon 1 only; ``scene.backward`` returns the
             captured mid-trajectory snapshot. Yields ``grad1_B`` (compared
             to ``grad1_A``) and that snapshot.
    Scene C: fresh scene, starts from B's snapshot, runs 5-step horizon 2 →
             ``scene.backward(loss2)``. Yields ``grad2_C`` (compared to ``grad2_A``).

If `scene.backward(loss)` correctly (a) restores physics state, (b) clears
the gradient tape, and (c) doesn't leak grad accumulation across horizons,
then ``grad1_A == grad1_B`` and ``grad2_A == grad2_C`` exactly.

We parameterize over the 5 J1~J5 topologies from
`test_diff_forward_kinematics.py` to cover single freejoint, 1-DOF revolute /
prismatic, freejoint+revolute child, and revolute chain-3.
"""

import os
import tempfile

import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import qd_to_torch

from .utils import assert_allclose


pytestmark = [
    pytest.mark.debug(False),
]


# Parametrization params (mirrors `test_diff_forward_kinematics.py`).
_PRECISION_PARAMS = [
    pytest.param("64", marks=pytest.mark.precision("64"), id="fp64"),
    pytest.param("32", marks=pytest.mark.precision("32"), id="fp32"),
]

_N_ENVS_PARAMS = [
    pytest.param(0, id="single"),
    pytest.param(4, id="batched"),
]

_TOL = {
    "64": dict(atol=1e-12, rtol=1e-10),
    "32": dict(atol=1e-5, rtol=1e-4),
}


# ---------------------------------------------------------------------------
# MJCF topologies (copied from `tests/test_diff_forward_kinematics.py` to keep
# this file self-contained).
# ---------------------------------------------------------------------------

MJCF_FREE = """
<mujoco model="free">
  <worldbody>
    <body name="chassis" pos="0 0 0">
      <freejoint/>
      <inertial mass="1.0" pos="0 0 0" diaginertia="0.1 0.1 0.1"/>
      <geom type="box" size="0.1 0.1 0.1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""

MJCF_REVOLUTE = """
<mujoco model="revolute">
  <worldbody>
    <body name="arm" pos="0 0 0">
      <joint type="hinge" axis="0 1 0"/>
      <inertial mass="0.5" pos="0.1 0 0" diaginertia="0.01 0.01 0.01"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""

MJCF_PRISMATIC = """
<mujoco model="prismatic">
  <worldbody>
    <body name="slider" pos="0 0 0">
      <joint type="slide" axis="1 0 0"/>
      <inertial mass="0.5" pos="0 0 0" diaginertia="0.01 0.01 0.01"/>
      <geom type="box" size="0.05 0.05 0.05" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""

MJCF_FREE_REV = """
<mujoco model="free_with_child">
  <worldbody>
    <body name="chassis" pos="0 0 0">
      <freejoint/>
      <inertial mass="1.0" pos="0 0 0" diaginertia="0.1 0.1 0.1"/>
      <geom type="box" size="0.1 0.1 0.1" contype="0" conaffinity="0"/>
      <body name="arm" pos="0.2 0 0">
        <joint type="hinge" axis="0 1 0"/>
        <inertial mass="0.5" pos="0.1 0 0" diaginertia="0.01 0.01 0.01"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

MJCF_REV_CHAIN3 = """
<mujoco model="chain3">
  <worldbody>
    <body name="l1" pos="0 0 0">
      <joint type="hinge" axis="0 1 0"/>
      <inertial mass="0.3" pos="0.1 0 0" diaginertia="0.005 0.005 0.005"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" contype="0" conaffinity="0"/>
      <body name="l2" pos="0.2 0 0">
        <joint type="hinge" axis="0 1 0"/>
        <inertial mass="0.3" pos="0.1 0 0" diaginertia="0.005 0.005 0.005"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" contype="0" conaffinity="0"/>
        <body name="l3" pos="0.2 0 0">
          <joint type="hinge" axis="0 1 0"/>
          <inertial mass="0.3" pos="0.1 0 0" diaginertia="0.005 0.005 0.005"/>
          <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" contype="0" conaffinity="0"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


_TOPOLOGIES = [
    pytest.param(MJCF_FREE, 6, id="J1_free"),
    pytest.param(MJCF_REVOLUTE, 1, id="J2_revolute"),
    pytest.param(MJCF_PRISMATIC, 1, id="J3_prismatic"),
    pytest.param(MJCF_FREE_REV, 7, id="J4_free_rev"),
    pytest.param(MJCF_REV_CHAIN3, 3, id="J5_chain3"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mjcf_to_tmpfile(mjcf_str: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".xml")
    with os.fdopen(fd, "w") as f:
        f.write(mjcf_str)
    return path


def _build_scene(mjcf_str: str, n_envs: int = 0, substeps: int = 1):
    """Build a diff-rigid scene with the standard "no collision / no constraint" config."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=substeps,
            gravity=(0.0, 0.0, 0.0),
            requires_grad=True,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            enable_self_collision=False,
            enable_joint_limit=False,
            disable_constraint=True,
            use_hibernation=False,
            use_contact_island=False,
        ),
        show_viewer=False,
    )
    robot = scene.add_entity(gs.morphs.MJCF(file=_mjcf_to_tmpfile(mjcf_str)))
    scene.build(n_envs=n_envs)
    return scene, robot


def _make_velocity(n_envs: int, n_dofs: int, seed: int) -> np.ndarray:
    """Per-env-distinct velocity vector. Single env: shape (n_dofs,). Batched: (n_envs, n_dofs)."""
    rng = np.random.default_rng(seed)
    if n_envs == 0:
        return rng.standard_normal(n_dofs)
    return rng.standard_normal((n_envs, n_dofs))


def _rigid_qpos_loss(scene):
    """Differentiable scalar loss = sum((qpos)**2). Reads `state.qpos` via
    `scene.get_state()` so the resulting tensor is a gs.Tensor whose
    `.backward()` triggers `scene._backward()`."""
    state = scene.get_state()
    rigid_state = state.solvers_state[scene.solvers.index(scene.rigid_solver)]
    return (rigid_state.qpos**2).sum()


def _run_segment(scene, robot, v_tensor, n_steps: int):
    """Apply `set_dofs_velocity(v_tensor)` once, then step `n_steps` times.
    Returns the resulting (post-step) scalar loss."""
    robot.set_dofs_velocity(v_tensor)
    for _ in range(n_steps):
        scene.step()
    return _rigid_qpos_loss(scene)


def _read_qpos(scene) -> np.ndarray:
    """Read the simulator's current qpos field (detached)."""
    solver = scene.rigid_solver
    return qd_to_torch(solver._rigid_global_info.qpos, copy=True).cpu().numpy()


@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("precision_str", _PRECISION_PARAMS)
@pytest.mark.parametrize("substeps", [1, 4])
@pytest.mark.parametrize("n_envs", _N_ENVS_PARAMS)
@pytest.mark.parametrize("mjcf_str, n_dofs", _TOPOLOGIES)
def test_horizon_truncation_matches_independent_scenes(mjcf_str, n_dofs, n_envs, substeps, precision_str):
    """Two-segment trajectory in Scene A matches the same two
    segments run in independent Scene B (horizon 1) and Scene C (horizon 2,
    started from B's mid-trajectory snapshot via `scene.reset(state)`).

    Verifies that `scene.get_state()` + `scene.reset(state)` correctly
    isolates two consecutive horizons: physics state propagates seamlessly,
    but the autograd tapes are independent."""
    tol = _TOL[precision_str]
    rng_v1 = _make_velocity(n_envs, n_dofs, seed=101)
    rng_v2 = _make_velocity(n_envs, n_dofs, seed=202)
    H = 5

    # ----- Scene A: one scene, snapshot+reset between two horizons -----
    sceneA, robotA = _build_scene(mjcf_str, n_envs=n_envs, substeps=substeps)
    sceneA.reset()
    v1A = gs.tensor(rng_v1, dtype=gs.tc_float, requires_grad=True)
    loss_h1_A = _run_segment(sceneA, robotA, v1A, H)
    qpos_mid_A = _read_qpos(sceneA)
    # `scene.backward` snapshots the terminal state, runs the backward unroll,
    # and restores that state — so horizon 2 continues seamlessly from here.
    sceneA.backward(loss_h1_A)
    # backward consumes the adstack / input buffer, so the step & substep
    # counters reset to 0 (they index that buffer) — unlike the physics state,
    # which is restored. Horizon 2 below thus records a fresh tape from 0.
    assert sceneA._t == 0 and sceneA._sim._cur_substep_global == 0
    grad1_A = v1A.grad.detach().clone().cpu().numpy()

    v2A = gs.tensor(rng_v2, dtype=gs.tc_float, requires_grad=True)
    loss_h2_A = _run_segment(sceneA, robotA, v2A, H)
    qpos_end_A = _read_qpos(sceneA)
    sceneA.backward(loss_h2_A)
    grad2_A = v2A.grad.detach().clone().cpu().numpy()

    # ----- Scene B: same horizon 1 only -----
    sceneB, robotB = _build_scene(mjcf_str, n_envs=n_envs, substeps=substeps)
    sceneB.reset()
    v1B = gs.tensor(rng_v1, dtype=gs.tc_float, requires_grad=True)
    loss_h1_B = _run_segment(sceneB, robotB, v1B, H)
    qpos_mid_B = _read_qpos(sceneB)
    # `scene.backward` returns the terminal snapshot it captured; Scene C below
    # loads it into a fresh scene via `reset(snapshot_B)`.
    snapshot_B = sceneB.backward(loss_h1_B)
    grad1_B = v1B.grad.detach().clone().cpu().numpy()

    # Sanity: A and B end at the same intermediate state and produce the same loss.
    assert_allclose(qpos_mid_A, qpos_mid_B, atol=0, rtol=0)
    assert_allclose(loss_h1_A.detach().cpu().item(), loss_h1_B.detach().cpu().item(), atol=0, rtol=0)
    # Core assertion: horizon-1 gradient identical.
    assert_allclose(grad1_A, grad1_B, **tol)

    # ----- Scene C: fresh scene, start from B's mid-trajectory snapshot -----
    sceneC, robotC = _build_scene(mjcf_str, n_envs=n_envs, substeps=substeps)
    sceneC.reset(snapshot_B)
    v2C = gs.tensor(rng_v2, dtype=gs.tc_float, requires_grad=True)
    loss_h2_C = _run_segment(sceneC, robotC, v2C, H)
    qpos_end_C = _read_qpos(sceneC)
    sceneC.backward(loss_h2_C)
    grad2_C = v2C.grad.detach().clone().cpu().numpy()

    # Sanity: A and C end at the same final state and produce the same loss.
    assert_allclose(qpos_end_A, qpos_end_C, atol=0, rtol=0)
    assert_allclose(loss_h2_A.detach().cpu().item(), loss_h2_C.detach().cpu().item(), atol=0, rtol=0)
    # Core assertion: horizon-2 gradient identical.
    assert_allclose(grad2_A, grad2_C, **tol)
