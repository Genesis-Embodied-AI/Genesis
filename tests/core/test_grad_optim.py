import os
from pathlib import Path

import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array

from ..utils import assert_allclose


@pytest.fixture
def build_cartpole_scene(grad_cartpole, show_viewer):
    def build(*, requires_grad: bool, n_envs: int = 0, substeps: int = 1):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=0.01,
                substeps=substeps,
                gravity=(0.0, 0.0, -9.81),
                requires_grad=requires_grad,
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
                camera_lookat=(0.0, 0.0, 0.5),
            ),
            show_viewer=show_viewer,
        )
        robot = scene.add_entity(
            gs.morphs.MJCF(
                file=grad_cartpole,
            ),
        )
        scene.build(n_envs=n_envs)
        return scene, robot

    return build


@pytest.fixture
def rigid_state():
    def read(scene):
        state = scene.get_state()
        return state.solvers_state[scene.solvers.index(scene.rigid_solver)]

    return read


@pytest.fixture
def rollout(rigid_state):
    # set_dofs_velocity is a tracked setter, so the returned (qpos, dofs_vel) tensors carry the gradient tape back
    # to init_vel.
    def run(scene, robot, init_vel, n_steps):
        scene.reset()
        robot.set_dofs_velocity(init_vel)
        for _ in range(n_steps):
            scene.step()
        state = rigid_state(scene)
        return state.qpos, state.dofs_vel

    return run


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize(
    "precision",
    [
        pytest.param("64", marks=pytest.mark.precision("64"), id="fp64"),
        pytest.param("32", marks=pytest.mark.precision("32"), id="fp32"),
    ],
)
@pytest.mark.parametrize(
    "n_envs",
    [
        pytest.param(0, id="single"),
        pytest.param(2, id="batched"),
    ],
)
def test_diff_optim_init_vel_cartpole(n_envs, precision, backend, build_cartpole_scene, rollout):
    N_STEPS = 32
    N_ITER = 200
    LR = 1e-2
    # cartpole: slider + hinge
    N_DOFS = 2
    B = n_envs if n_envs > 0 else 1

    # Precision-specific tolerances. fp32 has ~7 significant digits, so a rollout-trained scalar loss can't get
    # below ~1e-4 even at the optimum, and the rate of improvement plateaus earlier.
    if precision == "64":
        REL_REDUCTION = 1e-2
        ABS_THRESHOLD = 1e-4
    else:
        REL_REDUCTION = 1e-1
        ABS_THRESHOLD = 1e-2

    rng = np.random.default_rng(seed=11)
    input_shape = (N_DOFS,) if n_envs == 0 else (n_envs, N_DOFS)

    # Target trajectory from the requires_grad=False scene. Per-env distinct target_init_vel when n_envs > 0, so
    # each env converges to its own answer.
    target_init_vel_np = rng.normal(size=input_shape) * 0.5

    scene_ref, robot_ref = build_cartpole_scene(requires_grad=False, n_envs=n_envs)
    target_init_vel_t = gs.tensor(target_init_vel_np, dtype=gs.tc_float)
    with torch.no_grad():
        target_qpos, target_vel = rollout(scene_ref, robot_ref, target_init_vel_t, N_STEPS)
        target_qpos = target_qpos.detach().clone()
        target_vel = target_vel.detach().clone()

    # Differentiable scene to optimize on. Per-env distinct init noise.
    scene_opt, robot_opt = build_cartpole_scene(requires_grad=True, n_envs=n_envs)
    init_offset = rng.normal(size=input_shape) * 0.3
    init_vel_np = target_init_vel_np + init_offset
    init_vel = gs.tensor(init_vel_np, dtype=gs.tc_float, requires_grad=True)

    optimizer = torch.optim.Adam([init_vel], lr=LR)
    # Each entry is the (B,) per-env loss at one iteration.
    loss_history = []

    for it in range(N_ITER):
        optimizer.zero_grad(set_to_none=False)
        pred_qpos, pred_vel = rollout(scene_opt, robot_opt, init_vel, N_STEPS)
        diff_pos = (pred_qpos - target_qpos).reshape(B, -1)
        diff_vel = (pred_vel - target_vel).reshape(B, -1)
        loss_per_env = (diff_pos**2).sum(dim=-1) + (diff_vel**2).sum(dim=-1)
        loss = loss_per_env.sum()
        loss_history.append(tensor_to_array(loss_per_env).copy())
        loss.backward()
        assert init_vel.grad is not None, f"iter {it}: init_vel.grad is None"
        optimizer.step()

    # history has shape (N_ITER, B).
    history = np.asarray(loss_history)
    initial = history[0]
    final = history[-1]

    # Per-env assertions: every env must satisfy both criteria.
    rel_ratios = final / initial
    worst_rel_env = int(np.argmax(rel_ratios))
    assert (rel_ratios < REL_REDUCTION).all(), (
        f"loss reduction insufficient (worst env={worst_rel_env}): "
        f"initial={initial[worst_rel_env]:.3e}, final={final[worst_rel_env]:.3e}, "
        f"ratio={rel_ratios[worst_rel_env]:.3e} (>= {REL_REDUCTION:.0e})"
    )
    worst_abs_env = int(np.argmax(final))
    assert (final < ABS_THRESHOLD).all(), (
        f"final loss above absolute threshold (worst env={worst_abs_env}): "
        f"{final[worst_abs_env]:.3e} >= {ABS_THRESHOLD:.0e}"
    )


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize(
    "precision",
    [
        pytest.param("64", marks=pytest.mark.precision("64"), id="fp64"),
        pytest.param("32", marks=pytest.mark.precision("32"), id="fp32"),
    ],
)
@pytest.mark.parametrize(
    "n_envs",
    [
        pytest.param(0, id="single"),
        pytest.param(2, id="batched"),
    ],
)
def test_diff_optim_control_force_cartpole(n_envs, precision, backend, build_cartpole_scene, rigid_state):
    N_STEPS = 32
    N_ITER = 200
    LR = 1e-2
    N_DOFS = 2
    B = n_envs if n_envs > 0 else 1

    if precision == "64":
        REL_REDUCTION = 1e-2
        ABS_THRESHOLD = 1e-4
    else:
        REL_REDUCTION = 1e-1
        ABS_THRESHOLD = 1e-2

    rng = np.random.default_rng(seed=23)

    shape_per_step = (N_DOFS,) if n_envs == 0 else (n_envs, N_DOFS)
    target_force_np = rng.normal(size=(N_STEPS,) + shape_per_step) * 0.2

    # Target trajectory from the requires_grad=False scene.
    scene_ref, robot_ref = build_cartpole_scene(requires_grad=False, n_envs=n_envs)
    with torch.no_grad():
        scene_ref.reset()
        for t in range(N_STEPS):
            robot_ref.control_dofs_force(gs.tensor(target_force_np[t], dtype=gs.tc_float))
            scene_ref.step()
        state = rigid_state(scene_ref)
        target_qpos = state.qpos.detach().clone()
        target_vel = state.dofs_vel.detach().clone()

    # Differentiable scene with learnable per-step force tensors.
    scene_opt, robot_opt = build_cartpole_scene(requires_grad=True, n_envs=n_envs)
    init_offset = rng.normal(size=(N_STEPS,) + shape_per_step) * 0.1
    init_force_np = target_force_np + init_offset
    forces = [gs.tensor(init_force_np[t], dtype=gs.tc_float, requires_grad=True) for t in range(N_STEPS)]
    optimizer = torch.optim.Adam(forces, lr=LR)

    # Each entry is the (B,) per-env loss at one iteration.
    loss_history = []
    for it in range(N_ITER):
        optimizer.zero_grad(set_to_none=False)
        scene_opt.reset()
        for t in range(N_STEPS):
            robot_opt.control_dofs_force(forces[t])
            scene_opt.step()
        state = rigid_state(scene_opt)
        diff_pos = (state.qpos - target_qpos).reshape(B, -1)
        diff_vel = (state.dofs_vel - target_vel).reshape(B, -1)
        loss_per_env = (diff_pos**2).sum(dim=-1) + (diff_vel**2).sum(dim=-1)
        loss = loss_per_env.sum()
        loss_history.append(tensor_to_array(loss_per_env).copy())
        loss.backward()
        for t, force in enumerate(forces):
            assert force.grad is not None, f"iter {it} step {t}: force.grad is None"
        optimizer.step()

    # history has shape (N_ITER, B).
    history = np.asarray(loss_history)
    initial = history[0]
    final = history[-1]

    rel_ratios = final / initial
    worst_rel_env = int(np.argmax(rel_ratios))
    assert (rel_ratios < REL_REDUCTION).all(), (
        f"loss reduction insufficient (worst env={worst_rel_env}): "
        f"initial={initial[worst_rel_env]:.3e}, final={final[worst_rel_env]:.3e}, "
        f"ratio={rel_ratios[worst_rel_env]:.3e} (>= {REL_REDUCTION:.0e})"
    )
    worst_abs_env = int(np.argmax(final))
    assert (final < ABS_THRESHOLD).all(), (
        f"final loss above absolute threshold (worst env={worst_abs_env}): "
        f"{final[worst_abs_env]:.3e} >= {ABS_THRESHOLD:.0e}"
    )


@pytest.mark.slow
@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_differentiable_rigid(show_viewer):
    dt = 1e-2
    horizon = 100
    substeps = 1
    goal_pos = gs.tensor([0.7, 1.0, 0.05])
    goal_quat = gs.tensor([0.3, 0.2, 0.1, 0.9])
    goal_quat = goal_quat / torch.norm(goal_quat, dim=-1, keepdim=True)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=substeps,
            requires_grad=True,
            gravity=(0, 0, -1),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            enable_self_collision=False,
            enable_joint_limit=False,
            disable_constraint=True,
            use_contact_island=False,
            use_hibernation=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, -0.15, 2.42),
            camera_lookat=(0.5, 0.5, 0.1),
        ),
        show_viewer=show_viewer,
    )

    box = scene.add_entity(
        gs.morphs.Box(
            pos=(0, 0, 0),
            size=(0.1, 0.1, 0.2),
        ),
        surface=gs.surfaces.Default(
            color=(0.9, 0.0, 0.0, 1.0),
        ),
    )
    if show_viewer:
        target = scene.add_entity(
            gs.morphs.Box(
                pos=goal_pos,
                quat=goal_quat,
                size=(0.1, 0.1, 0.2),
            ),
            surface=gs.surfaces.Default(
                color=(0.0, 0.9, 0.0, 0.5),
            ),
        )

    scene.build()

    num_iter = 200
    lr = 1e-2

    init_pos = gs.tensor([0.3, 0.1, 0.28], requires_grad=True)
    init_quat = gs.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=True)
    optimizer = torch.optim.Adam([init_pos, init_quat], lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iter, eta_min=1e-3)

    for _ in range(num_iter):
        scene.reset()

        box.set_pos(init_pos)
        box.set_quat(init_quat)

        for _ in range(horizon):
            scene.step()
            if show_viewer:
                target.set_pos(goal_pos)
                target.set_quat(goal_quat)

        box_state = box.get_state()
        box_pos = box_state.pos
        box_quat = box_state.quat
        loss = torch.abs(box_pos - goal_pos).sum() + torch.abs(box_quat - goal_quat).sum()

        optimizer.zero_grad()
        # Gradient flows all the way back to the input tensors.
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            init_quat.data = init_quat / torch.norm(init_quat, dim=-1, keepdim=True)

    assert_allclose(loss, 0.0, atol=1e-2)
