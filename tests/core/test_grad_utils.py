import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from ..utils import assert_allclose


pytestmark = [
    pytest.mark.debug(False),
]


@pytest.fixture
def build_scene(show_viewer):
    def build(model_xml, n_envs=0, substeps=1):
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
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.5, -1.5, 1.0),
                camera_lookat=(0.0, 0.0, 0.0),
            ),
            show_viewer=show_viewer,
        )
        robot = scene.add_entity(
            gs.morphs.MJCF(
                file=model_xml,
            ),
        )
        scene.build(n_envs=n_envs)
        return scene, robot

    return build


@pytest.fixture
def run_segment():
    # The loss reads qpos through scene.get_state(), so it is a gs.Tensor wired to the scene gradient tape and its
    # backward triggers the scene backward unroll.
    def run(scene, robot, velocity, n_steps):
        robot.set_dofs_velocity(velocity)
        for _ in range(n_steps):
            scene.step()
        state = scene.get_state()
        rigid_state = state.solvers_state[scene.solvers.index(scene.rigid_solver)]
        return (rigid_state.qpos**2).sum()

    return run


@pytest.fixture
def read_qpos():
    def read(scene):
        return qd_to_numpy(scene.rigid_solver.rigid_info.qpos, copy=True)

    return read


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize(
    "precision_str",
    [
        pytest.param("64", marks=pytest.mark.precision("64"), id="fp64"),
        pytest.param("32", marks=pytest.mark.precision("32"), id="fp32"),
    ],
)
@pytest.mark.parametrize("substeps", [1, 4])
@pytest.mark.parametrize(
    "n_envs",
    [
        pytest.param(0, id="single"),
        pytest.param(2, id="batched"),
    ],
)
@pytest.mark.parametrize(
    "model_fixture, n_dofs",
    [
        pytest.param("grad_free", 6, id="J1_free"),
        pytest.param("grad_revolute", 1, id="J2_revolute"),
        pytest.param("grad_prismatic", 1, id="J3_prismatic"),
        pytest.param("grad_free_with_revolute", 7, id="J4_free_rev"),
        pytest.param("grad_revolute_chain3", 3, id="J5_chain3"),
    ],
)
def test_horizon_truncation_matches_independent_scenes(
    model_fixture, n_dofs, n_envs, substeps, precision_str, build_scene, run_segment, read_qpos, request
):
    tol = dict(atol=1e-12, rtol=1e-10) if precision_str == "64" else dict(atol=1e-5, rtol=1e-4)
    model_xml = request.getfixturevalue(model_fixture)
    # Per-env-distinct velocity vectors. Single env: shape (n_dofs,). Batched: (n_envs, n_dofs).
    shape = (n_dofs,) if n_envs == 0 else (n_envs, n_dofs)
    v1_np = np.random.default_rng(seed=101).standard_normal(shape)
    v2_np = np.random.default_rng(seed=202).standard_normal(shape)
    horizon = 5

    # ----- Scene A: one scene, snapshot+reset between two horizons -----
    sceneA, robotA = build_scene(model_xml, n_envs=n_envs, substeps=substeps)
    sceneA.reset()
    v1A = gs.tensor(v1_np, dtype=gs.tc_float, requires_grad=True)
    loss_h1_A = run_segment(sceneA, robotA, v1A, horizon)
    qpos_mid_A = read_qpos(sceneA)
    # scene.backward snapshots the terminal state, runs the backward unroll, and restores that state, so horizon 2
    # continues seamlessly from here.
    sceneA.backward(loss_h1_A)
    # backward consumes the adstack / input buffer, so the step and substep counters reset to 0 (they index that
    # buffer) while the restored physics state carries over. Horizon 2 below thus records a fresh tape from 0.
    assert sceneA._t == 0 and sceneA._sim._cur_substep_global == 0
    grad1_A = tensor_to_array(v1A.grad).copy()

    v2A = gs.tensor(v2_np, dtype=gs.tc_float, requires_grad=True)
    loss_h2_A = run_segment(sceneA, robotA, v2A, horizon)
    qpos_end_A = read_qpos(sceneA)
    sceneA.backward(loss_h2_A)
    grad2_A = tensor_to_array(v2A.grad).copy()

    # ----- Scene B: same horizon 1 only -----
    sceneB, robotB = build_scene(model_xml, n_envs=n_envs, substeps=substeps)
    sceneB.reset()
    v1B = gs.tensor(v1_np, dtype=gs.tc_float, requires_grad=True)
    loss_h1_B = run_segment(sceneB, robotB, v1B, horizon)
    qpos_mid_B = read_qpos(sceneB)
    # scene.backward returns the terminal snapshot it captured; Scene C below starts a fresh scene from it.
    snapshot_B = sceneB.backward(loss_h1_B)
    grad1_B = tensor_to_array(v1B.grad).copy()

    # Sanity: A and B end at the same intermediate state and produce the same loss.
    assert_allclose(qpos_mid_A, qpos_mid_B, atol=0, rtol=0)
    assert_allclose(float(tensor_to_array(loss_h1_A)), float(tensor_to_array(loss_h1_B)), atol=0, rtol=0)
    # Core assertion: horizon-1 gradient identical.
    assert_allclose(grad1_A, grad1_B, **tol)

    # ----- Scene C: fresh scene, start from B's mid-trajectory snapshot -----
    sceneC, robotC = build_scene(model_xml, n_envs=n_envs, substeps=substeps)
    sceneC.reset(snapshot_B)
    v2C = gs.tensor(v2_np, dtype=gs.tc_float, requires_grad=True)
    loss_h2_C = run_segment(sceneC, robotC, v2C, horizon)
    qpos_end_C = read_qpos(sceneC)
    sceneC.backward(loss_h2_C)
    grad2_C = tensor_to_array(v2C.grad).copy()

    # Sanity: A and C end at the same final state and produce the same loss.
    assert_allclose(qpos_end_A, qpos_end_C, atol=0, rtol=0)
    assert_allclose(float(tensor_to_array(loss_h2_A)), float(tensor_to_array(loss_h2_C)), atol=0, rtol=0)
    # Core assertion: horizon-2 gradient identical.
    assert_allclose(grad2_A, grad2_C, **tol)
