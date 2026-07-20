# Autodiff-tape behavior: horizon truncation (snapshot + reset between two backward windows must reproduce the
# gradients and states of independent fresh scenes), and gradient-source parity (entity state vs rigid-solver state).
import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from ..utils import assert_allclose
from .utils import make_diff_scene_pair


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["grad_free", "grad_revolute", "grad_free_with_revolute"])
def test_grad_horizon_truncation_matches_independent_scenes(model_name, request, show_viewer):
    mjcf = request.getfixturevalue(model_name)
    tol = dict(atol=1e-5, rtol=1e-4)
    horizon = 5
    B = 2

    def build(show_viewer=False):
        pair = make_diff_scene_pair(mjcf, n_envs=2, substeps=4, gravity=(0.0, 0.0, 0.0), show_viewer=show_viewer)
        return pair.scene_ana, pair.entity_ana

    def run_segment(scene, entity, velocity):
        entity.set_dofs_velocity(velocity)
        for _ in range(horizon):
            scene.step()
        return (scene.rigid_solver.get_state().qpos ** 2).sum()

    def read_qpos(scene):
        return qd_to_numpy(scene.rigid_solver.rigid_info.qpos, copy=True)

    # Scene A: one scene, snapshot + reset between two horizons.
    scene_a, robot_a = build(show_viewer=show_viewer)
    v1 = np.random.default_rng(seed=101).standard_normal((B, robot_a.n_dofs))
    v2 = np.random.default_rng(seed=202).standard_normal((B, robot_a.n_dofs))
    scene_a.reset()
    v1_a = gs.tensor(v1, dtype=gs.tc_float, requires_grad=True)
    loss_h1_a = run_segment(scene_a, robot_a, v1_a)
    qpos_mid_a = read_qpos(scene_a)
    scene_a.backward(loss_h1_a)
    # backward consumes the input buffer, so the step / substep counters (which index it) reset to 0 while the
    # restored physics state carries over; horizon 2 records a fresh tape from 0.
    assert scene_a._t == 0 and scene_a._sim._cur_substep_global == 0
    grad1_a = tensor_to_array(v1_a.grad).copy()

    v2_a = gs.tensor(v2, dtype=gs.tc_float, requires_grad=True)
    loss_h2_a = run_segment(scene_a, robot_a, v2_a)
    qpos_end_a = read_qpos(scene_a)
    scene_a.backward(loss_h2_a)
    grad2_a = tensor_to_array(v2_a.grad).copy()

    # Scene B: horizon 1 only; its backward returns the terminal snapshot Scene C resumes from.
    scene_b, robot_b = build()
    scene_b.reset()
    v1_b = gs.tensor(v1, dtype=gs.tc_float, requires_grad=True)
    loss_h1_b = run_segment(scene_b, robot_b, v1_b)
    qpos_mid_b = read_qpos(scene_b)
    snapshot_b = scene_b.backward(loss_h1_b)
    grad1_b = tensor_to_array(v1_b.grad).copy()

    assert_allclose(qpos_mid_a, qpos_mid_b, atol=0, rtol=0)
    assert_allclose(float(tensor_to_array(loss_h1_a)), float(tensor_to_array(loss_h1_b)), atol=0, rtol=0)
    assert_allclose(grad1_a, grad1_b, **tol)

    # Scene C: fresh scene resumed from B's mid-trajectory snapshot.
    scene_c, robot_c = build()
    scene_c.reset(snapshot_b)
    v2_c = gs.tensor(v2, dtype=gs.tc_float, requires_grad=True)
    loss_h2_c = run_segment(scene_c, robot_c, v2_c)
    qpos_end_c = read_qpos(scene_c)
    scene_c.backward(loss_h2_c)
    grad2_c = tensor_to_array(v2_c.grad).copy()

    assert_allclose(qpos_end_a, qpos_end_c, atol=0, rtol=0)
    assert_allclose(float(tensor_to_array(loss_h2_a)), float(tensor_to_array(loss_h2_c)), atol=0, rtol=0)
    assert_allclose(grad2_a, grad2_c, **tol)


@pytest.mark.slow
@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_rigid_sim_vs_solver_state_grad_parity(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0.0, 0.0, 0.0),
            requires_grad=True,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
        ),
        show_viewer=show_viewer,
    )
    robot = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0, 0, 0),
        )
    )
    scene.build()

    ctrl = gs.tensor(np.random.randn(robot.n_dofs), dtype=gs.tc_float, requires_grad=True)

    grads = []
    for is_sim_state_source in (False, True):
        scene.reset()
        robot.set_dofs_velocity(ctrl)
        scene.step()
        if is_sim_state_source:
            chassis_pos = scene.rigid_solver.get_state().links_pos[:, 0].squeeze()
        else:
            chassis_pos = robot.get_state().pos.squeeze()
        loss = torch.linalg.norm(chassis_pos)
        loss.backward()
        grads.append(ctrl.grad.detach().clone())
        ctrl.grad.zero_()
        assert (grads[-1][..., :3].abs() > gs.EPS).all()
        assert_allclose(grads[-1][..., 3:], 0.0, atol=gs.EPS)

    assert_allclose(*grads, atol=gs.EPS)
