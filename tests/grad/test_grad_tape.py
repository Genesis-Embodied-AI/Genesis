import gc
import weakref

import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array

from ..utils.assertions import assert_allclose, assert_equal
from .utils import make_diff_scene_pair


@pytest.mark.parametrize(
    "model_name", ["grad_free", "grad_revolute", pytest.param("grad_free_with_revolute", marks=pytest.mark.required)]
)
def test_horizon_truncation_matches_independent_scenes(model_name, request, show_viewer):
    mjcf = request.getfixturevalue(model_name)
    tol = dict(atol=1e-5, rtol=1e-4)
    horizon = 5
    B = 2

    def build(show_viewer=False):
        pair = make_diff_scene_pair(mjcf, n_envs=B, substeps=4, gravity=(0.0, 0.0, 0.0), show_viewer=show_viewer)
        return pair.scene_ana, pair.entity_ana

    def run_segment(scene, entity, velocity):
        entity.set_dofs_velocity(velocity)
        for _ in range(horizon):
            scene.step()
        return (scene.rigid_solver.get_state().qpos ** 2).sum()

    def read_qpos(scene):
        return tensor_to_array(scene.rigid_solver.get_state().qpos)

    # Scene A: one scene, snapshot + reset between two horizons.
    scene_a, robot_a = build(show_viewer=show_viewer)
    v1 = np.random.default_rng(seed=101).standard_normal((B, robot_a.n_dofs))
    v2 = np.random.default_rng(seed=202).standard_normal((B, robot_a.n_dofs))
    scene_a.reset()
    v1_a = gs.tensor(v1, dtype=gs.tc_float, requires_grad=True)
    loss_h1_a = run_segment(scene_a, robot_a, v1_a)
    qpos_mid_a = read_qpos(scene_a)
    time_mid_a = scene_a.get_time().clone()
    scene_a.backward(loss_h1_a)
    # backward restores the physics state it snapshotted, so the clocks describing it come back too, while the substep
    # counter indexes the input buffer that was consumed and rewinds to 0 for the tape horizon 2 records.
    assert_allclose(scene_a.get_time(), time_mid_a, tol=gs.EPS)
    assert scene_a._sim._cur_substep_global == 0
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

    assert_equal(qpos_mid_a, qpos_mid_b)
    assert_equal(loss_h1_a, loss_h1_b)
    assert_allclose(grad1_a, grad1_b, **tol)

    # Scene C: fresh scene resumed from B's mid-trajectory snapshot.
    scene_c, robot_c = build()
    scene_c.reset(snapshot_b)
    v2_c = gs.tensor(v2, dtype=gs.tc_float, requires_grad=True)
    loss_h2_c = run_segment(scene_c, robot_c, v2_c)
    qpos_end_c = read_qpos(scene_c)
    scene_c.backward(loss_h2_c)
    grad2_c = tensor_to_array(v2_c.grad).copy()

    assert_equal(qpos_end_a, qpos_end_c)
    assert_equal(loss_h2_a, loss_h2_c)
    assert_allclose(grad2_a, grad2_c, **tol)


@pytest.mark.required
def test_sim_vs_solver_state_grad_parity(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
            requires_grad=True,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.2, -1.2, 0.8),
            camera_lookat=(0.0, 0.0, 0.2),
        ),
        show_viewer=show_viewer,
    )
    robot = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
        ),
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
        # Rebuilding a checkpoint window replays steps the environments already simulated, so the clock has to follow
        # the physics the backward pass rewinds instead of counting that replay as newly simulated time.
        assert_allclose(scene.get_time(), 0.0, tol=gs.EPS)
        grads.append(ctrl.grad.detach().clone())
        ctrl.grad.zero_()
        assert (grads[-1][..., :3].abs() > gs.EPS).all()
        assert_allclose(grads[-1][..., 3:], 0.0, atol=gs.EPS)

    assert_allclose(*grads, atol=gs.EPS)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("closed_loop", [False, True])
def test_torch_policy_grad_matches_fd(closed_loop, grad_revolute, show_viewer):
    # The control forces are computed by a torch policy from plain torch parameters, not from tensors Genesis created,
    # and are penalized in the loss as well, so their graph is shared by the loss and the rollout. Closed loop, they
    # also read the state queried before each step. The rollout is part of the same autograd graph, so a plain
    # `loss.backward()` (no `retain_graph=True`) and `torch.autograd.grad` both differentiate it in one traversal.
    # Finite differences need 64-bit precision.
    horizon, dt = 50, 0.01
    pair = make_diff_scene_pair(grad_revolute, n_envs=2, dt=dt, show_viewer=show_viewer)
    target = torch.tensor([[0.4], [-0.3]], dtype=gs.tc_float, device=gs.device)
    params = np.array([0.6, 3.0, 8.0, 0.5])
    n_params = len(params) if closed_loop else 2  # open loop, the feedback gains are unused

    def rollout(scene, entity, params):
        amplitude, frequency, kp, kd = torch.as_tensor(params, dtype=gs.tc_float, device=gs.device)
        scene.reset()
        effort = 0.0
        for i_step in range(horizon):
            force = (amplitude * torch.sin(frequency * i_step * dt)).expand(scene.n_envs, 1)
            if closed_loop:
                state = scene.rigid_solver.get_state()
                force = force + kp * (target - state.qpos) - kd * state.dofs_vel
            entity.control_dofs_force(force)
            effort = effort + (force**2).sum()
            scene.step()
        return ((scene.rigid_solver.get_state().qpos - target) ** 2).sum() + 1e-4 * effort

    params_ana = torch.tensor(params, dtype=gs.tc_float, device=gs.device, requires_grad=True)
    rollout(pair.scene_ana, pair.entity_ana, params_ana).backward()
    (grad_functional,) = torch.autograd.grad(rollout(pair.scene_ana, pair.entity_ana, params_ana), params_ana)
    assert_allclose(grad_functional, params_ana.grad, tol=gs.EPS)

    eps = 1e-6
    fd_grad = np.zeros_like(params)
    for i in range(n_params):
        losses = []
        for sign in (+1, -1):
            perturbed = params.copy()
            perturbed[i] += sign * eps
            losses.append(float(rollout(pair.scene_fd, pair.entity_fd, perturbed)))
        fd_grad[i] = (losses[0] - losses[1]) / (2.0 * eps)
    assert (np.abs(fd_grad[:n_params]) > 1e-5).all()
    assert_allclose(params_ana.grad, fd_grad, rtol=1e-6, atol=1e-9)

    # A differentiable scene is freed once the user drops it, whatever rollout it recorded
    scene_ref = weakref.ref(pair.scene_ana)
    del pair
    gc.collect()
    assert scene_ref() is None


@pytest.mark.required
@pytest.mark.precision("64")
def test_input_shared_by_entities_grad_matches_fd(grad_revolute, show_viewer):
    # One input tensor given to two entities in the same step is one input of that step's autograd node, which returns
    # the gradient summed over both: counting it once per entity would double it.
    horizon = 5

    def build(requires_grad):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(gravity=(0.0, 0.0, 0.0), requires_grad=requires_grad),
            rigid_options=gs.options.RigidOptions(enable_collision=False, disable_constraint=True),
            show_viewer=show_viewer and requires_grad,
        )
        arms = [scene.add_entity(gs.morphs.MJCF(file=grad_revolute, pos=(0.0, y, 0.0))) for y in (-0.3, 0.3)]
        scene.build()
        return scene, arms

    def rollout(scene, arms, gain):
        scene.reset()
        for i_step in range(horizon):
            force = gain * (i_step + 1.0)
            for arm in arms:
                arm.control_dofs_force(force)
            scene.step()
        return (scene.rigid_solver.get_state().qpos ** 2).sum()

    gain = torch.tensor([0.7], dtype=gs.tc_float, device=gs.device, requires_grad=True)
    rollout(*build(requires_grad=True), gain).backward()

    eps = 1e-6
    scene_fd, arms_fd = build(requires_grad=False)
    losses = [float(rollout(scene_fd, arms_fd, np.array([0.7 + sign * eps]))) for sign in (+1, -1)]
    fd_grad = (losses[0] - losses[1]) / (2.0 * eps)
    assert abs(fd_grad) > 1e-3
    assert_allclose(gain.grad, fd_grad, rtol=1e-6, atol=1e-9)
