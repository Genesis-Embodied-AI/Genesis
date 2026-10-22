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
    v1_a = torch.tensor(v1, dtype=gs.tc_float, device=gs.device, requires_grad=True)
    loss_h1_a = run_segment(scene_a, robot_a, v1_a)
    qpos_mid_a = read_qpos(scene_a)
    time_mid_a = scene_a.get_time().clone()
    scene_a.backward(loss_h1_a)
    # backward restores the physics state it snapshotted, so the clocks describing it come back too, while the substep
    # counter indexes the input buffer that was consumed and rewinds to 0 for the tape horizon 2 records.
    assert_allclose(scene_a.get_time(), time_mid_a, tol=gs.EPS)
    assert scene_a._sim._cur_substep_global == 0
    grad1_a = tensor_to_array(v1_a.grad).copy()

    v2_a = torch.tensor(v2, dtype=gs.tc_float, device=gs.device, requires_grad=True)
    loss_h2_a = run_segment(scene_a, robot_a, v2_a)
    qpos_end_a = read_qpos(scene_a)
    scene_a.backward(loss_h2_a)
    grad2_a = tensor_to_array(v2_a.grad).copy()

    # Scene B: horizon 1 only; its backward returns the terminal snapshot Scene C resumes from.
    scene_b, robot_b = build()
    scene_b.reset()
    v1_b = torch.tensor(v1, dtype=gs.tc_float, device=gs.device, requires_grad=True)
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
    v2_c = torch.tensor(v2, dtype=gs.tc_float, device=gs.device, requires_grad=True)
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

    ctrl = torch.tensor(np.random.randn(robot.n_dofs), dtype=gs.tc_float, device=gs.device, requires_grad=True)

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

    # A state queried after a setter reflects it, even within the step of an earlier query
    scene.reset()
    scene.rigid_solver.get_state()
    robot.set_pos((0.1, -0.2, 0.3))
    assert_allclose(scene.rigid_solver.get_state().links_pos[0, 0], (0.1, -0.2, 0.3), tol=gs.EPS)


@pytest.mark.required
def test_closed_loop_policy_grad_matches_fd(grad_revolute, precision, show_viewer):
    # The control forces of a torch policy read the state queried before each step, are given to both arms and are
    # penalized in the loss, so their graph is shared by the loss and the rollout.
    HORIZON, DT = 50, 0.01

    scenes, arms = [], []
    for requires_grad in (True, False):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=DT,
                gravity=(0.0, 0.0, 0.0),
                requires_grad=requires_grad,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_collision=False,
                disable_constraint=True,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.2, -1.2, 0.8),
                camera_lookat=(0.0, 0.0, 0.2),
            ),
            show_viewer=show_viewer and requires_grad,
        )
        for y in (-0.3, 0.3):
            arms.append(
                scene.add_entity(
                    gs.morphs.MJCF(
                        file=grad_revolute,
                        pos=(0.0, y, 0.0),
                    ),
                )
            )
        scene.build(n_envs=2)
        scenes.append(scene)
    scene_ana, scene_fd = scenes
    arms_ana, arms_fd = arms[:2], arms[2:]

    target = torch.tensor([[0.4], [-0.3]], dtype=gs.tc_float, device=gs.device)
    params = np.array([0.6, 3.0, 8.0, 0.5])

    def rollout(scene, arms, params):
        amplitude, frequency, kp, kd = torch.as_tensor(params, dtype=gs.tc_float, device=gs.device)
        scene.reset()
        effort = 0.0
        for i_step in range(HORIZON):
            # A query without gradient recording, as for logging, leaves the next query of the step differentiable
            with torch.no_grad():
                scene.rigid_solver.get_state()
            state = scene.rigid_solver.get_state()
            force = amplitude * torch.sin(frequency * i_step * DT) + kp * (target - state.qpos[:, :1])
            force = force - kd * state.dofs_vel[:, :1]
            for arm in arms:
                arm.control_dofs_force(force)
            effort = effort + (force**2).sum()
            scene.step()
        return ((scene.rigid_solver.get_state().qpos - target) ** 2).sum() + 1e-4 * effort

    params_ana = torch.tensor(params, dtype=gs.tc_float, device=gs.device, requires_grad=True)
    rollout(scene_ana, arms_ana, params_ana).backward()
    (grad_functional,) = torch.autograd.grad(rollout(scene_ana, arms_ana, params_ana), params_ana)
    assert_allclose(grad_functional, params_ana.grad, tol=gs.EPS)

    # fp32 needs a large step to clear the state-noise floor. fp64 needs a small step to bound truncation error.
    eps = 3e-5 if precision == "64" else 3e-2
    fd_grad = np.zeros_like(params)
    for i in range(len(params)):
        losses = []
        for sign in (+1, -1):
            perturbed = params.copy()
            perturbed[i] += sign * eps
            losses.append(float(rollout(scene_fd, arms_fd, perturbed)))
        fd_grad[i] = (losses[0] - losses[1]) / (2.0 * eps)
    assert (np.abs(fd_grad) > 1e-3).all()
    tol = 2e-10 if precision == "64" else 1e-4
    assert_allclose(params_ana.grad, fd_grad, rtol=tol, atol=tol)

    # A differentiable scene is freed once the user drops it, whatever rollout it recorded
    scene_ref = weakref.ref(scene_ana)
    del scenes, arms, scene, scene_ana, arms_ana
    gc.collect()
    assert scene_ref() is None
