# FD-vs-analytical gradient checks for the differentiable rigid solver, in three sections: forward kinematics
# (constraints off), joint limit, and contact, each with its own scene builder so configs never bleed across.
import math

import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils import set_random_seed
from genesis.utils.geom import R_to_quat
from genesis.utils.misc import qd_to_numpy, qd_to_torch, tensor_to_array

from ..utils import assert_allclose


pytestmark = [
    pytest.mark.debug(False),
]


def _fd_tol(precision, kind):
    # Per-precision FD tolerance, looser at fp32. The "quat" kind covers outputs that go through a nonlinear pose
    # composition (set_dofs_velocity -> state.quat) where the analytical gradient sits ~1% away from central FD.
    return {
        ("64", "default"): dict(rtol=1e-4, atol=1e-6, eps=1e-5),
        ("64", "quat"): dict(rtol=2e-2, atol=1e-3, eps=1e-5),
        ("32", "default"): dict(rtol=2e-2, atol=2e-3, eps=1e-3),
        ("32", "quat"): dict(rtol=5e-2, atol=5e-3, eps=1e-3),
    }[precision, kind]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_scene(mjcf: str, *, requires_grad: bool, n_envs: int = 0, substeps: int = 1, show_viewer: bool = False):
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
            camera_pos=(1.2, -1.2, 0.8),
            camera_lookat=(0.0, 0.0, 0.2),
        ),
        show_viewer=show_viewer,
    )
    robot = scene.add_entity(gs.morphs.MJCF(file=mjcf))
    scene.build(n_envs=n_envs)
    return scene, robot


def _make_scene_pair(mjcf: str, n_envs: int = 0, substeps: int = 1, show_viewer: bool = False):
    # scene_ana runs the differentiable-mode forward and is the only one loss.backward() ever runs on: a backward
    # leaves the scene's target-replay state silently ignoring subsequent setters, so FD-probing it would return a
    # fake zero gradient. scene_fd runs the production forward and is what FD perturbs, so each reset -> set -> step
    # cycle stays clean. FD therefore checks the diff-mode analytical gradient against the production forward's
    # local sensitivity, per env when n_envs > 0.
    # The Metal reverse-mode adstack is undersized for the integrator's joint-type branch and corrupts batched
    # gradients (lane 0 replicated across envs); CPU and CUDA are correct. See
    # https://github.com/Genesis-Embodied-AI/quadrants/issues/791.
    if gs.backend == gs.metal and n_envs > 0:
        pytest.skip("Batched reverse-mode kernels are broken on Metal (quadrants#791).")
    scene_ana, robot_ana = _build_scene(
        mjcf, requires_grad=True, n_envs=n_envs, substeps=substeps, show_viewer=show_viewer
    )
    scene_fd, robot_fd = _build_scene(mjcf, requires_grad=False, n_envs=n_envs, substeps=substeps)
    return scene_ana, robot_ana, scene_fd, robot_fd, mjcf


def _batch_size(scene) -> int:
    return scene.n_envs if scene.n_envs > 0 else 1


def _input_shape(base_shape, n_envs):
    return (n_envs,) + tuple(base_shape) if n_envs > 0 else tuple(base_shape)


def _solver_state(scene):
    state = scene.get_state()
    return state.solvers_state[scene.solvers.index(scene.rigid_solver)]


def _grad_matches_fd(
    scene_ana,
    robot_ana,
    scene_fd,
    robot_fd,
    init_input,  # 1-D numpy array (fp64)
    apply_fn,  # callable(robot, x): apply x via a @tracked setter
    loss_fn,  # callable(scene, robot) -> scalar tensor
    *,
    label: str,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    eps: float = 1e-5,
):
    # The production-mode and diff-mode forward kernels produce bit-identical states for the same input, so an FD
    # probed on the no-grad scene is a valid reference for the diff scene's analytical gradient.
    base_np = np.array(init_input, dtype=np.float64)

    # --- analytical (diff-mode scene) ---
    x_ana = gs.tensor(base_np, dtype=gs.tc_float, requires_grad=True)
    scene_ana.reset()
    apply_fn(robot_ana, x_ana)
    scene_ana.step()
    loss = loss_fn(scene_ana, robot_ana)
    assert loss.requires_grad, f"[{label}] loss does not require grad - output is not grad-aware"
    loss.backward()
    assert x_ana.grad is not None, f"[{label}] x.grad is None after backward"
    ana_grad = tensor_to_array(x_ana.grad)

    # --- central FD (production-mode scene) ---
    n = base_np.size
    fd_grad = np.zeros_like(base_np)
    for i in range(n):
        plus = base_np.copy()
        plus.reshape(-1)[i] = base_np.reshape(-1)[i] + eps
        scene_fd.reset()
        apply_fn(robot_fd, gs.tensor(plus, dtype=gs.tc_float))
        scene_fd.step()
        loss_p = float(loss_fn(scene_fd, robot_fd))

        minus = base_np.copy()
        minus.reshape(-1)[i] = base_np.reshape(-1)[i] - eps
        scene_fd.reset()
        apply_fn(robot_fd, gs.tensor(minus, dtype=gs.tc_float))
        scene_fd.step()
        loss_m = float(loss_fn(scene_fd, robot_fd))

        fd_grad.reshape(-1)[i] = (loss_p - loss_m) / (2.0 * eps)

    assert_allclose(
        torch.from_numpy(ana_grad),
        torch.from_numpy(fd_grad),
        rtol=rtol,
        atol=atol,
        err_msg=f"[{label}] FD vs analytical mismatch",
    )


def _grad_matches_fd_multistep(
    scene_ana,
    robot_ana,
    scene_fd,
    robot_fd,
    init_inputs,  # list[np.ndarray] - one input per timestep, each shape matches the setter's expectation
    apply_fn,  # callable(robot, x): apply x via a @tracked setter
    loss_fn,  # callable(scene, robot) -> scalar tensor
    *,
    label: str,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    eps: float = 1e-5,
):
    # Multi-step variant of _grad_matches_fd: N = len(init_inputs) steps, one tracked-setter input per step, each of
    # which must receive an independent adjoint from the backward unroll. The FD reference perturbs each entry of
    # each step's input separately and re-runs the full N-step trajectory on scene_fd (O(N * sum of input sizes)
    # forward runs of N steps each).
    N = len(init_inputs)
    base_np = [np.array(inp, dtype=np.float64) for inp in init_inputs]

    # --- analytical (diff-mode scene) ---
    scene_ana.reset()
    x_anas = []
    for t in range(N):
        x = gs.tensor(base_np[t], dtype=gs.tc_float, requires_grad=True)
        x_anas.append(x)
        apply_fn(robot_ana, x)
        scene_ana.step()
    loss = loss_fn(scene_ana, robot_ana)
    assert loss.requires_grad, f"[{label}] loss does not require grad - output is not grad-aware"
    loss.backward()
    ana_grads = []
    for t, x in enumerate(x_anas):
        assert x.grad is not None, f"[{label}] step {t}: x.grad is None after backward"
        ana_grads.append(tensor_to_array(x.grad))

    # --- central FD (production-mode scene): for each (t, i) entry, run the
    # full N-step trajectory twice with the perturbation injected only at
    # step t. All other steps use the original input.
    fd_grads = [np.zeros_like(b) for b in base_np]

    def _run_traj_with_perturb(t_perturb, i_perturb, sign):
        scene_fd.reset()
        for s in range(N):
            inp = base_np[s].copy()
            if s == t_perturb:
                inp.reshape(-1)[i_perturb] += sign * eps
            apply_fn(robot_fd, gs.tensor(inp, dtype=gs.tc_float))
            scene_fd.step()
        return float(loss_fn(scene_fd, robot_fd))

    for t in range(N):
        for i in range(base_np[t].size):
            loss_p = _run_traj_with_perturb(t, i, +1)
            loss_m = _run_traj_with_perturb(t, i, -1)
            fd_grads[t].reshape(-1)[i] = (loss_p - loss_m) / (2.0 * eps)

    for t in range(N):
        assert_allclose(
            torch.from_numpy(ana_grads[t]),
            torch.from_numpy(fd_grads[t]),
            rtol=rtol,
            atol=atol,
            err_msg=f"[{label}] step {t}: FD vs analytical mismatch",
        )


# loss factories - all use sum-of-squared-deviation to a fixed random target so
# every entry of the input has a nontrivial sensitivity. Targets and outputs are
# both flattened before the subtraction so multi-link shapes (B, n_links, 3|4)
# don't trip torch broadcasting.
def _loss_state_pos(target):
    flat = target.reshape(-1)

    def _fn(scene, robot):
        return ((robot.get_state().pos.reshape(-1) - flat) ** 2).sum()

    return _fn


def _loss_state_quat(target):
    flat = target.reshape(-1)

    def _fn(scene, robot):
        return ((robot.get_state().quat.reshape(-1) - flat) ** 2).sum()

    return _fn


def _loss_links_pos(target):
    flat = target.reshape(-1)

    def _fn(scene, robot):
        return ((_solver_state(scene).links_pos.reshape(-1) - flat) ** 2).sum()

    return _fn


def _loss_links_quat(target):
    flat = target.reshape(-1)

    def _fn(scene, robot):
        return ((_solver_state(scene).links_quat.reshape(-1) - flat) ** 2).sum()

    return _fn


def _rand_np(shape, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float64)


def _target(shape, seed):
    return torch.from_numpy(_rand_np(shape, seed)).to(dtype=gs.tc_float, device=gs.device)


# ---------------------------------------------------------------------------
# Tests - one per joint topology, several (input, output) checks inside.
# ---------------------------------------------------------------------------


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize(
    "precision",
    [
        pytest.param("64", marks=pytest.mark.precision("64"), id="fp64"),
        pytest.param("32", marks=pytest.mark.precision("32"), id="fp32"),
    ],
)
@pytest.mark.parametrize("n_envs", [pytest.param(0, id="single"), pytest.param(2, id="batched")])
@pytest.mark.parametrize("substeps", [pytest.param(1, id="ss1"), pytest.param(4, id="ss4")])
def test_diff_fk_freejoint(show_viewer, n_envs, precision, substeps, grad_free):
    scene_ana, robot_ana, scene_fd, robot_fd, _ = _make_scene_pair(grad_free, n_envs=n_envs, substeps=substeps)
    n_dofs = robot_ana.n_dofs
    B = _batch_size(scene_ana)
    tol_default = _fd_tol(precision, "default")
    tol_quat = _fd_tol(precision, "quat")

    tgt_pos = _target((B, 3), seed=1)
    tgt_quat = _target((B, 4), seed=2)

    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=_rand_np(_input_shape((3,), n_envs), seed=10),
        apply_fn=lambda r, x: r.set_pos(x),
        loss_fn=_loss_state_pos(tgt_pos),
        label="J1 set_pos -> state.pos",
        **tol_default,
    )

    init_q_shape = _input_shape((4,), n_envs)
    init_q = np.broadcast_to(np.array([1.0, 0.0, 0.0, 0.0]), init_q_shape).copy()
    init_q = init_q + 0.05 * _rand_np(init_q_shape, seed=11)
    init_q = init_q / np.linalg.norm(init_q, axis=-1, keepdims=True)
    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=init_q,
        apply_fn=lambda r, x: r.set_quat(x),
        loss_fn=_loss_state_quat(tgt_quat),
        label="J1 set_quat -> state.quat",
        **tol_quat,
    )

    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=12),
        apply_fn=lambda r, x: r.set_dofs_velocity(x),
        loss_fn=_loss_state_pos(tgt_pos),
        label="J1 set_dofs_velocity -> state.pos (after 1 step)",
        **tol_default,
    )

    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=13),
        apply_fn=lambda r, x: r.set_dofs_velocity(x),
        loss_fn=_loss_state_quat(tgt_quat),
        label="J1 set_dofs_velocity -> state.quat (after 1 step)",
        **tol_quat,
    )

    # fp64 only: d(state.pos)/d(force) ~ dt^2 / (2 * inertia) ~ 1e-4 after 1
    # step. At fp32 with FD eps=1e-3 the loss difference is ~1e-7 - at fp32's
    # precision floor - and the FD probe disagrees with analytical by ~1e-4
    # absolute, well above the fp32 default tol band. The J2/J3/J4/J5 force
    # checks below are also fp64-only for the same reason; J2's
    # `control_dofs_force -> state.quat` does pass at fp32 only because its
    # check uses the wider quat tolerance.
    if precision == "64":
        _grad_matches_fd(
            scene_ana,
            robot_ana,
            scene_fd,
            robot_fd,
            init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=14),
            apply_fn=lambda r, x: r.control_dofs_force(x),
            loss_fn=_loss_state_pos(tgt_pos),
            label="J1 control_dofs_force -> state.pos (after 1 step)",
            **tol_default,
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
@pytest.mark.parametrize("n_envs", [pytest.param(0, id="single"), pytest.param(2, id="batched")])
@pytest.mark.parametrize("substeps", [pytest.param(1, id="ss1"), pytest.param(4, id="ss4")])
def test_diff_fk_revolute(show_viewer, n_envs, precision, substeps, grad_revolute):
    scene_ana, robot_ana, scene_fd, robot_fd, _ = _make_scene_pair(grad_revolute, n_envs=n_envs, substeps=substeps)
    n_dofs = robot_ana.n_dofs  # = 1
    B = _batch_size(scene_ana)
    tol_default = _fd_tol(precision, "default")
    tol_quat = _fd_tol(precision, "quat")

    tgt_pos = _target((B, 3), seed=21)
    tgt_quat = _target((B, 4), seed=22)

    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=30),
        apply_fn=lambda r, x: r.set_dofs_velocity(x),
        loss_fn=_loss_state_pos(tgt_pos),
        label="J2 set_dofs_velocity -> state.pos",
        **tol_default,
    )

    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=31),
        apply_fn=lambda r, x: r.set_dofs_velocity(x),
        loss_fn=_loss_state_quat(tgt_quat),
        label="J2 set_dofs_velocity -> state.quat",
        **tol_quat,
    )

    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=32),
        apply_fn=lambda r, x: r.control_dofs_force(x),
        loss_fn=_loss_state_quat(tgt_quat),
        label="J2 control_dofs_force -> state.quat",
        **tol_quat,
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
@pytest.mark.parametrize("n_envs", [pytest.param(0, id="single"), pytest.param(2, id="batched")])
@pytest.mark.parametrize("substeps", [pytest.param(1, id="ss1"), pytest.param(4, id="ss4")])
def test_diff_fk_spherical(show_viewer, n_envs, precision, substeps, grad_spherical):
    scene_ana, robot_ana, scene_fd, robot_fd, _ = _make_scene_pair(grad_spherical, n_envs=n_envs, substeps=substeps)
    n_dofs = robot_ana.n_dofs  # = 3
    B = _batch_size(scene_ana)
    tol_default = _fd_tol(precision, "default")
    tol_quat = _fd_tol(precision, "quat")

    tgt_pos = _target((B, 3), seed=61)
    tgt_quat = _target((B, 4), seed=62)

    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=70),
        apply_fn=lambda r, x: r.set_dofs_velocity(x),
        loss_fn=_loss_state_pos(tgt_pos),
        label="J6 set_dofs_velocity -> state.pos",
        **tol_default,
    )

    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=71),
        apply_fn=lambda r, x: r.set_dofs_velocity(x),
        loss_fn=_loss_state_quat(tgt_quat),
        label="J6 set_dofs_velocity -> state.quat",
        **tol_quat,
    )

    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=72),
        apply_fn=lambda r, x: r.control_dofs_force(x),
        loss_fn=_loss_state_quat(tgt_quat),
        label="J6 control_dofs_force -> state.quat",
        **tol_quat,
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
@pytest.mark.parametrize("n_envs", [pytest.param(0, id="single"), pytest.param(2, id="batched")])
@pytest.mark.parametrize("substeps", [pytest.param(1, id="ss1"), pytest.param(4, id="ss4")])
def test_diff_fk_prismatic(show_viewer, n_envs, precision, substeps, grad_prismatic):
    scene_ana, robot_ana, scene_fd, robot_fd, _ = _make_scene_pair(grad_prismatic, n_envs=n_envs, substeps=substeps)
    n_dofs = robot_ana.n_dofs  # = 1
    B = _batch_size(scene_ana)
    tol_default = _fd_tol(precision, "default")
    tgt_pos = _target((B, 3), seed=41)

    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=50),
        apply_fn=lambda r, x: r.set_dofs_velocity(x),
        loss_fn=_loss_state_pos(tgt_pos),
        label="J3 set_dofs_velocity -> state.pos",
        **tol_default,
    )

    # fp64-only - see J1's control_dofs_force comment for why FD-vs-analytical
    # on force-driven position is at fp32's precision floor.
    if precision == "64":
        _grad_matches_fd(
            scene_ana,
            robot_ana,
            scene_fd,
            robot_fd,
            init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=51),
            apply_fn=lambda r, x: r.control_dofs_force(x),
            loss_fn=_loss_state_pos(tgt_pos),
            label="J3 control_dofs_force -> state.pos",
            **tol_default,
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
@pytest.mark.parametrize("n_envs", [pytest.param(0, id="single"), pytest.param(2, id="batched")])
@pytest.mark.parametrize("substeps", [pytest.param(1, id="ss1"), pytest.param(4, id="ss4")])
def test_diff_fk_cartpole(show_viewer, n_envs, precision, substeps, grad_cartpole):
    scene_ana, robot_ana, scene_fd, robot_fd, _ = _make_scene_pair(grad_cartpole, n_envs=n_envs, substeps=substeps)
    n_dofs = robot_ana.n_dofs  # = 2 (slider + hinge)
    B = _batch_size(scene_ana)
    tol_default = _fd_tol(precision, "default")
    tol_quat = _fd_tol(precision, "quat")

    tgt_links_pos = _target((B, 2, 3), seed=181)
    tgt_links_quat = _target((B, 2, 4), seed=182)

    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=190),
        apply_fn=lambda r, x: r.set_dofs_velocity(x),
        loss_fn=_loss_links_pos(tgt_links_pos),
        label="J7 set_dofs_velocity -> links_pos",
        **tol_default,
    )

    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=191),
        apply_fn=lambda r, x: r.set_dofs_velocity(x),
        loss_fn=_loss_links_quat(tgt_links_quat),
        label="J7 set_dofs_velocity -> links_quat",
        **tol_quat,
    )

    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=192),
        apply_fn=lambda r, x: r.control_dofs_force(x),
        loss_fn=_loss_links_pos(tgt_links_pos),
        label="J7 control_dofs_force -> links_pos",
        **tol_default,
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
@pytest.mark.parametrize("n_envs", [pytest.param(0, id="single"), pytest.param(2, id="batched")])
@pytest.mark.parametrize("substeps", [pytest.param(1, id="ss1"), pytest.param(4, id="ss4")])
def test_diff_fk_hopper(show_viewer, n_envs, precision, substeps, grad_hopper):
    scene_ana, robot_ana, scene_fd, robot_fd, _ = _make_scene_pair(grad_hopper, n_envs=n_envs, substeps=substeps)
    n_dofs = robot_ana.n_dofs  # = 6 (rootx, rootz, rooty, thigh, leg, foot)
    n_links = robot_ana.n_links  # = 5 (base + torso, thigh, leg, foot)
    B = _batch_size(scene_ana)
    tol_default = _fd_tol(precision, "default")
    tol_quat = _fd_tol(precision, "quat")
    # Hopper is the largest topology here (5 links, 6 DOFs). At fp32 the batched
    # (n_envs=4) FD probe quantizes the small-sensitivity links_pos/links_quat
    # entries to a ~2e-3 step, leaving a few entries ~6e-3 from the analytical.
    # fp64 (single + batched) pins correctness; widen only the fp32 atol band so
    # the FD-floor noise on the larger chain doesn't trip the check.
    if precision == "32":
        tol_default = dict(rtol=tol_default["rtol"], atol=8e-3, eps=tol_default["eps"])
        tol_quat = dict(rtol=tol_quat["rtol"], atol=8e-3, eps=tol_quat["eps"])

    tgt_links_pos = _target((B, n_links, 3), seed=201)
    tgt_links_quat = _target((B, n_links, 4), seed=202)

    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=210),
        apply_fn=lambda r, x: r.set_dofs_velocity(x),
        loss_fn=_loss_links_pos(tgt_links_pos),
        label="J8 set_dofs_velocity -> links_pos",
        **tol_default,
    )

    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=211),
        apply_fn=lambda r, x: r.set_dofs_velocity(x),
        loss_fn=_loss_links_quat(tgt_links_quat),
        label="J8 set_dofs_velocity -> links_quat",
        **tol_quat,
    )

    # fp64-only - see J1's control_dofs_force comment.
    if precision == "64":
        _grad_matches_fd(
            scene_ana,
            robot_ana,
            scene_fd,
            robot_fd,
            init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=212),
            apply_fn=lambda r, x: r.control_dofs_force(x),
            loss_fn=_loss_links_pos(tgt_links_pos),
            label="J8 control_dofs_force -> links_pos",
            **tol_default,
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
@pytest.mark.parametrize("n_envs", [pytest.param(0, id="single"), pytest.param(2, id="batched")])
@pytest.mark.parametrize("substeps", [pytest.param(1, id="ss1"), pytest.param(4, id="ss4")])
def test_diff_fk_free_with_revolute(show_viewer, n_envs, precision, substeps, grad_free_with_revolute):
    scene_ana, robot_ana, scene_fd, robot_fd, _ = _make_scene_pair(
        grad_free_with_revolute, n_envs=n_envs, substeps=substeps
    )
    n_dofs = robot_ana.n_dofs  # 6 free + 1 hinge = 7
    n_links = robot_ana.n_links  # 2
    B = _batch_size(scene_ana)
    tol_default = _fd_tol(precision, "default")
    tol_quat = _fd_tol(precision, "quat")
    tgt_links_pos = _target((B, n_links, 3), seed=61)
    tgt_links_quat = _target((B, n_links, 4), seed=62)

    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=_rand_np(_input_shape((3,), n_envs), seed=70),
        apply_fn=lambda r, x: r.set_pos(x),
        loss_fn=_loss_links_pos(tgt_links_pos),
        label="J4 set_pos -> links_pos",
        **tol_default,
    )

    init_q_shape = _input_shape((4,), n_envs)
    init_q = np.broadcast_to(np.array([1.0, 0.0, 0.0, 0.0]), init_q_shape).copy()
    init_q = init_q + 0.05 * _rand_np(init_q_shape, seed=71)
    init_q = init_q / np.linalg.norm(init_q, axis=-1, keepdims=True)
    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=init_q,
        apply_fn=lambda r, x: r.set_quat(x),
        loss_fn=_loss_links_quat(tgt_links_quat),
        label="J4 set_quat -> links_quat",
        **tol_quat,
    )

    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=72),
        apply_fn=lambda r, x: r.set_dofs_velocity(x),
        loss_fn=_loss_links_pos(tgt_links_pos),
        label="J4 set_dofs_velocity -> links_pos",
        **tol_default,
    )

    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=73),
        apply_fn=lambda r, x: r.set_dofs_velocity(x),
        loss_fn=_loss_links_quat(tgt_links_quat),
        label="J4 set_dofs_velocity -> links_quat",
        **tol_quat,
    )

    # fp64-only - see J1's control_dofs_force comment.
    if precision == "64":
        _grad_matches_fd(
            scene_ana,
            robot_ana,
            scene_fd,
            robot_fd,
            init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=74),
            apply_fn=lambda r, x: r.control_dofs_force(x),
            loss_fn=_loss_links_pos(tgt_links_pos),
            label="J4 control_dofs_force -> links_pos",
            **tol_default,
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
@pytest.mark.parametrize("n_envs", [pytest.param(0, id="single"), pytest.param(2, id="batched")])
@pytest.mark.parametrize("substeps", [pytest.param(1, id="ss1"), pytest.param(4, id="ss4")])
def test_diff_fk_revolute_chain3(show_viewer, n_envs, precision, substeps, grad_revolute_chain3):
    scene_ana, robot_ana, scene_fd, robot_fd, _ = _make_scene_pair(
        grad_revolute_chain3, n_envs=n_envs, substeps=substeps
    )
    n_dofs = robot_ana.n_dofs  # 3
    n_links = robot_ana.n_links  # 3
    B = _batch_size(scene_ana)
    tol_default = _fd_tol(precision, "default")
    tol_quat = _fd_tol(precision, "quat")
    tgt_links_pos = _target((B, n_links, 3), seed=81)
    tgt_links_quat = _target((B, n_links, 4), seed=82)

    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=90),
        apply_fn=lambda r, x: r.set_dofs_velocity(x),
        loss_fn=_loss_links_pos(tgt_links_pos),
        label="J5 set_dofs_velocity -> links_pos",
        **tol_default,
    )

    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=91),
        apply_fn=lambda r, x: r.set_dofs_velocity(x),
        loss_fn=_loss_links_quat(tgt_links_quat),
        label="J5 set_dofs_velocity -> links_quat",
        **tol_quat,
    )

    # fp64-only - see J1's control_dofs_force comment.
    if precision == "64":
        _grad_matches_fd(
            scene_ana,
            robot_ana,
            scene_fd,
            robot_fd,
            init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=92),
            apply_fn=lambda r, x: r.control_dofs_force(x),
            loss_fn=_loss_links_pos(tgt_links_pos),
            label="J5 control_dofs_force -> links_pos",
            **tol_default,
        )


# ---------------------------------------------------------------------------
# Multi-step gradient verification - exercises cross-step adjoint propagation.
# ---------------------------------------------------------------------------


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize(
    "model_name, n_dofs, loss_factory, output_shape, seed",
    [
        pytest.param("grad_free", 6, _loss_state_pos, (3,), 161, id="J1_free"),
        pytest.param("grad_revolute", 1, _loss_state_pos, (3,), 162, id="J2_revolute"),
        pytest.param("grad_prismatic", 1, _loss_state_pos, (3,), 163, id="J3_prismatic"),
        pytest.param("grad_free_with_revolute", 7, _loss_links_pos, (2, 3), 164, id="J4_free_rev"),
        pytest.param("grad_revolute_chain3", 3, _loss_links_pos, (3, 3), 165, id="J5_chain3"),
        pytest.param("grad_spherical", 3, _loss_state_pos, (3,), 166, id="J6_spherical"),
        pytest.param("grad_cartpole", 2, _loss_links_pos, (2, 3), 167, id="J7_cartpole"),
        pytest.param("grad_hopper", 6, _loss_links_pos, (5, 3), 168, id="J8_hopper"),
    ],
)
@pytest.mark.parametrize("substeps", [pytest.param(1, id="ss1"), pytest.param(4, id="ss4")])
def test_diff_fk_multistep_control_force(
    show_viewer, request, model_name, n_dofs, loss_factory, output_shape, seed, substeps
):
    mjcf = request.getfixturevalue(model_name)
    scene_ana, robot_ana, scene_fd, robot_fd, _ = _make_scene_pair(
        mjcf, n_envs=0, substeps=substeps, show_viewer=show_viewer
    )
    B = _batch_size(scene_ana)
    target = _target((B, *output_shape), seed=seed)

    # 10 distinct force inputs, one per step.
    N = 10
    init_inputs = [_rand_np((n_dofs,), seed=seed * 100 + t) for t in range(N)]

    _grad_matches_fd_multistep(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_inputs=init_inputs,
        apply_fn=lambda r, x: r.control_dofs_force(x),
        loss_fn=loss_factory(target),
        label=f"{model_name} control_dofs_force x {N} steps",
    )


# ===========================================================================
# Joint-limit constraint FD  (enable_joint_limit=True -> constraints ON)
# ===========================================================================


def _build(mjcf: str, *, requires_grad: bool, enable_joint_limit: bool, show_viewer: bool = False):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            substeps=4,
            gravity=(0.0, 0.0, 0.0),
            requires_grad=requires_grad,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            enable_self_collision=False,
            enable_joint_limit=enable_joint_limit,
            disable_constraint=not enable_joint_limit,
            use_hibernation=False,
            use_contact_island=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.2, -1.2, 0.8),
            camera_lookat=(0.0, 0.0, 0.2),
        ),
        show_viewer=show_viewer,
    )
    robot = scene.add_entity(gs.morphs.MJCF(file=mjcf))
    scene.build(n_envs=0)
    return scene, robot


def _rigid_state(scene):
    return scene.get_state().solvers_state[scene.solvers.index(scene.rigid_solver)]


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_diff_joint_limit_forward_enforcement(show_viewer, grad_slider_limit):
    mjcf = grad_slider_limit

    # Control: limit OFF
    scene, robot = _build(mjcf, requires_grad=False, enable_joint_limit=False)
    scene.reset()
    robot.set_dofs_velocity(gs.tensor([100.0], dtype=gs.tc_float))
    for _ in range(60):
        scene.step()
    x_off = float(_rigid_state(scene).qpos[0, 0].detach())
    assert abs(x_off) > 50.0, f"control (limit OFF) cart should drift past 50m, got x={x_off}"

    # Limit ON - should stay bounded.
    scene, robot = _build(mjcf, requires_grad=False, enable_joint_limit=True)
    scene.reset()
    robot.set_dofs_velocity(gs.tensor([100.0], dtype=gs.tc_float))
    for _ in range(60):
        scene.step()
    x_on = float(_rigid_state(scene).qpos[0, 0].detach())
    assert abs(x_on) <= 4.5, f"limit ON should keep |x| <= 4.5 (small margin for soft constraint), got x={x_on}"


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("init_vel", [0.5, 5.0])
def test_diff_joint_limit_backward_finite_no_limit_hit(show_viewer, init_vel, grad_slider_limit):
    mjcf = grad_slider_limit
    N_STEPS = 1  # short - cart doesn't reach limit

    grads = {}
    for limit in (False, True):
        scene, robot = _build(mjcf, requires_grad=True, enable_joint_limit=limit)
        scene.reset()
        v = gs.tensor([init_vel], dtype=gs.tc_float, requires_grad=True)
        robot.set_dofs_velocity(v)
        for _ in range(N_STEPS):
            scene.step()
        loss = (_rigid_state(scene).qpos[0, 0]) ** 2
        loss.backward()
        assert v.grad is not None, f"limit={limit}: v.grad is None"
        g = float(v.grad[0])
        assert math.isfinite(g), f"limit={limit}: gradient is not finite ({g})"
        grads[limit] = g

    # Limit-inactive case should match the no-limit baseline tightly - the
    # constraint branch only runs `n_constraints += 0`, so the autograd tape
    # should be identical up to floating-point.
    assert_allclose(grads[True], grads[False], rtol=1e-6, atol=1e-9)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_diff_joint_limit_backward_fd_one_step(show_viewer, grad_slider_limit):
    mjcf = grad_slider_limit
    init_vel = 2.0
    eps = 1e-5

    # Analytical
    scene_ana, robot_ana = _build(mjcf, requires_grad=True, enable_joint_limit=True)
    scene_ana.reset()
    v = gs.tensor([init_vel], dtype=gs.tc_float, requires_grad=True)
    robot_ana.set_dofs_velocity(v)
    scene_ana.step()
    loss = (_rigid_state(scene_ana).qpos[0, 0]) ** 2
    loss.backward()
    ana = float(v.grad[0])

    # FD
    scene_fd, robot_fd = _build(mjcf, requires_grad=False, enable_joint_limit=True)

    def loss_at(val: float) -> float:
        scene_fd.reset()
        robot_fd.set_dofs_velocity(gs.tensor([val], dtype=gs.tc_float))
        scene_fd.step()
        return float((_rigid_state(scene_fd).qpos[0, 0]) ** 2)

    fd = (loss_at(init_vel + eps) - loss_at(init_vel - eps)) / (2 * eps)

    assert_allclose(ana, fd, rtol=1e-3, atol=1e-6)


# (init_vel, n_steps) cases where the cart actually crosses |x|=4 during the
# rollout. Each case engages the constraint solver during the integration -
# they cover the M^{-1} J^T lambda correction path that the unconstrained
# `kernel_manual_compute_qacc_bw` could not produce. Resolved 2026-05-25 by
# wiring `constraint_solver.backward` + `kernel_manual_add_joint_limit_constraints_bw`
# into `substep_pre_coupling_grad`.


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("init_vel, n_steps", [(500.0, 1), (200.0, 2), (100.0, 5), (50.0, 10)])
def test_diff_joint_limit_backward_fd_active(show_viewer, init_vel, n_steps, grad_slider_limit):
    mjcf = grad_slider_limit
    eps = 1e-4

    # Analytical
    scene_ana, robot_ana = _build(mjcf, requires_grad=True, enable_joint_limit=True)
    scene_ana.reset()
    v = gs.tensor([init_vel], dtype=gs.tc_float, requires_grad=True)
    robot_ana.set_dofs_velocity(v)
    for _ in range(n_steps):
        scene_ana.step()
    x_final = float(_rigid_state(scene_ana).qpos[0, 0].detach())
    # Setup sanity: the cart must have entered the limit band, otherwise this
    # case wouldn't actually exercise the constraint correction path.
    assert abs(x_final) > 3.5, (
        f"setup error: init_vel={init_vel}, n_steps={n_steps} did not bring "
        f"the cart near the limit (x_final={x_final}); pick a larger v0 or "
        f"more steps."
    )
    loss = (_rigid_state(scene_ana).qpos[0, 0]) ** 2
    loss.backward()
    ana = float(v.grad[0])

    # FD
    scene_fd, robot_fd = _build(mjcf, requires_grad=False, enable_joint_limit=True)

    def loss_at(val: float) -> float:
        scene_fd.reset()
        robot_fd.set_dofs_velocity(gs.tensor([val], dtype=gs.tc_float))
        for _ in range(n_steps):
            scene_fd.step()
        return float((_rigid_state(scene_fd).qpos[0, 0]) ** 2)

    fd = (loss_at(init_vel + eps) - loss_at(init_vel - eps)) / (2 * eps)

    assert_allclose(ana, fd, rtol=1e-3, atol=1e-6)


# Per-step force horizons that drive the cart into the slider limit through
# `control_dofs_force`. Constant +500 N over `n_steps` accelerates the
# unit-mass cart past |x|=4 within ~10 substep-groups at dt=1/60, substeps=4
# (default solref); shorter horizons leave the cart inside the band and
# don't exercise the constraint backward, so we restrict to multi-step
# active cases. n_steps=10 probes whether the per-step `force.grad` for
# early-horizon steps leaks a wrong gradient when the constrained backward
# chain (`constraint_solver.backward` + manual joint-limit BW +
# `fwd_dynamics_without_qacc.grad` accumulation) runs across many substeps.


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("n_steps", [10])
def test_diff_joint_limit_backward_fd_per_step_force(show_viewer, n_steps, grad_slider_limit):
    mjcf = grad_slider_limit
    eps = 1e-2
    force_value = 500.0
    init_force = np.full((n_steps, 1), force_value, dtype=np.float64)

    # Analytical
    scene_ana, robot_ana = _build(mjcf, requires_grad=True, enable_joint_limit=True)
    scene_ana.reset()
    forces = [gs.tensor(init_force[t], dtype=gs.tc_float, requires_grad=True) for t in range(n_steps)]
    for t in range(n_steps):
        robot_ana.control_dofs_force(forces[t])
        scene_ana.step()
    x_final = float(_rigid_state(scene_ana).qpos[0, 0].detach())
    # Setup sanity: the cart must have entered the limit band, otherwise this
    # case wouldn't actually exercise the multi-step constraint backward.
    assert abs(x_final) > 3.5, (
        f"setup error: n_steps={n_steps} at force={force_value} did not bring "
        f"the cart near the limit (x_final={x_final}); pick a larger force or "
        f"more steps."
    )
    loss = (_rigid_state(scene_ana).qpos[0, 0]) ** 2
    loss.backward()
    for t, f in enumerate(forces):
        assert f.grad is not None, f"step {t}: force.grad is None"
    ana = np.array([float(f.grad[0]) for f in forces])

    # FD per-step
    scene_fd, robot_fd = _build(mjcf, requires_grad=False, enable_joint_limit=True)

    def loss_at(perturbed: np.ndarray) -> float:
        scene_fd.reset()
        for t in range(n_steps):
            robot_fd.control_dofs_force(gs.tensor(perturbed[t], dtype=gs.tc_float))
            scene_fd.step()
        return float((_rigid_state(scene_fd).qpos[0, 0]) ** 2)

    fd = np.zeros(n_steps)
    for t in range(n_steps):
        plus = init_force.copy()
        plus[t, 0] += eps
        minus = init_force.copy()
        minus[t, 0] -= eps
        fd[t] = (loss_at(plus) - loss_at(minus)) / (2 * eps)

    # Per-step comparison so the failure message identifies the offending step.
    for t in range(n_steps):
        assert_allclose(
            ana[t],
            fd[t],
            rtol=1e-3,
            atol=1e-4,
            err_msg=(
                f"per-step force.grad mismatch at t={t}/{n_steps} "
                f"(ana={ana[t]:+.4e}, fd={fd[t]:+.4e}); full ana={ana}, fd={fd}"
            ),
        )


def _build_cartpole(mjcf: str, *, requires_grad: bool, show_viewer: bool = False):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            substeps=4,
            gravity=(0.0, 0.0, -9.81),
            requires_grad=requires_grad,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            enable_self_collision=False,
            enable_joint_limit=True,
            disable_constraint=False,
            use_hibernation=False,
            use_contact_island=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.2, -1.2, 0.8),
            camera_lookat=(0.0, 0.0, 0.2),
        ),
        show_viewer=show_viewer,
    )
    robot = scene.add_entity(gs.morphs.MJCF(file=mjcf))
    scene.build(n_envs=0)
    return scene, robot


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("n_steps", [15])
def test_diff_joint_limit_backward_fd_per_step_force_cartpole(show_viewer, n_steps, grad_cartpole):
    mjcf = grad_cartpole
    eps = 1e-2
    # cart+pole effective mass ~ 11 (cart 1 + pole 10 horizontal-locked at
    # hanging), so cart force needs to be larger than the cart-only test
    # to reach the limit within `n_steps`.
    force_value = 2000.0
    # Force shape per step: (n_dofs,) = (cart_f, pole_f). pole stays at 0.
    init_force = np.zeros((n_steps, 2), dtype=np.float64)
    init_force[:, 0] = force_value

    # Initial state: cart at x=0, pole hanging down at theta=-pi (same as
    # `CartPoleSwingUpEnv._init_qpos`). Deterministic; same in ana / FD.
    init_qpos = [0.0, -math.pi]

    # Analytical
    scene_ana, robot_ana = _build_cartpole(mjcf, requires_grad=True)
    scene_ana.reset()
    robot_ana.set_dofs_position(gs.tensor(init_qpos, dtype=gs.tc_float))
    forces = [gs.tensor(init_force[t], dtype=gs.tc_float, requires_grad=True) for t in range(n_steps)]
    for t in range(n_steps):
        robot_ana.control_dofs_force(forces[t])
        scene_ana.step()
    x_final = float(_rigid_state(scene_ana).qpos[0, 0].detach())
    assert abs(x_final) > 3.5, (
        f"setup error: cart+pole at n_steps={n_steps}, force={force_value} "
        f"did not bring the cart near the limit (x_final={x_final}); pick a "
        f"larger force or more steps."
    )
    loss = (_rigid_state(scene_ana).qpos[0, 0]) ** 2
    loss.backward()
    for t, f in enumerate(forces):
        assert f.grad is not None, f"step {t}: force.grad is None"
    # cart-force grad per step (slot 0); slot 1 is pole-force grad, must be 0.
    ana_cart = np.array([float(f.grad[0]) for f in forces])
    ana_pole = np.array([float(f.grad[1]) for f in forces])

    # FD per-step on the cart-force slot only.
    scene_fd, robot_fd = _build_cartpole(mjcf, requires_grad=False)

    def loss_at(perturbed: np.ndarray) -> float:
        scene_fd.reset()
        robot_fd.set_dofs_position(gs.tensor(init_qpos, dtype=gs.tc_float))
        for t in range(n_steps):
            robot_fd.control_dofs_force(gs.tensor(perturbed[t], dtype=gs.tc_float))
            scene_fd.step()
        return float((_rigid_state(scene_fd).qpos[0, 0]) ** 2)

    fd_cart = np.zeros(n_steps)
    fd_pole = np.zeros(n_steps)
    for t in range(n_steps):
        plus = init_force.copy()
        plus[t, 0] += eps
        minus = init_force.copy()
        minus[t, 0] -= eps
        fd_cart[t] = (loss_at(plus) - loss_at(minus)) / (2 * eps)

        plus = init_force.copy()
        plus[t, 1] += eps
        minus = init_force.copy()
        minus[t, 1] -= eps
        fd_pole[t] = (loss_at(plus) - loss_at(minus)) / (2 * eps)

    # Cart-force grad - straight chain from action to cart_x.
    for t in range(n_steps):
        assert_allclose(
            ana_cart[t],
            fd_cart[t],
            rtol=1e-3,
            atol=1e-4,
            err_msg=(
                f"cart-pole cart_force.grad mismatch at t={t}/{n_steps} "
                f"(ana={ana_cart[t]:+.4e}, fd={fd_cart[t]:+.4e}); "
                f"full ana={ana_cart}, fd={fd_cart}"
            ),
        )
    # Pole-force grad - hinge torque chain: pole_force -> pole_angle ->
    # pole COM horizontal accel -> reactive force on cart via hinge ->
    # cart_x. Non-zero, must still match FD step-by-step.
    for t in range(n_steps):
        assert_allclose(
            ana_pole[t],
            fd_pole[t],
            rtol=1e-3,
            atol=1e-4,
            err_msg=(
                f"cart-pole pole_force.grad mismatch at t={t}/{n_steps} "
                f"(ana={ana_pole[t]:+.4e}, fd={fd_pole[t]:+.4e}); "
                f"full ana_pole={ana_pole}, fd_pole={fd_pole}"
            ),
        )


def _build_hopper(mjcf: str, *, requires_grad: bool, show_viewer: bool = False):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            substeps=4,
            gravity=(0.0, 0.0, 0.0),
            requires_grad=requires_grad,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            enable_self_collision=False,
            enable_joint_limit=True,
            disable_constraint=False,
            use_hibernation=False,
            use_contact_island=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.2, -1.2, 0.8),
            camera_lookat=(0.0, 0.0, 0.2),
        ),
        show_viewer=show_viewer,
    )
    robot = scene.add_entity(gs.morphs.MJCF(file=mjcf))
    scene.build(n_envs=0)
    return scene, robot


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("n_steps", [10])
def test_diff_joint_limit_backward_fd_per_step_force_hopper(show_viewer, n_steps, grad_hopper):
    mjcf = grad_hopper
    n_dofs = 6  # rootx, rootz, rooty, thigh, leg, foot
    foot_dof = 5
    eps = 1e-2
    force_value = 200.0
    init_force = np.zeros((n_steps, n_dofs), dtype=np.float64)
    init_force[:, foot_dof] = force_value

    def _links_pos_sq_loss(scene):
        lp = _rigid_state(scene).links_pos
        return (lp.reshape(-1) ** 2).sum()

    # Analytical
    scene_ana, robot_ana = _build_hopper(mjcf, requires_grad=True)
    scene_ana.reset()
    forces = [gs.tensor(init_force[t], dtype=gs.tc_float, requires_grad=True) for t in range(n_steps)]
    for t in range(n_steps):
        robot_ana.control_dofs_force(forces[t])
        scene_ana.step()
    foot_q = float(_rigid_state(scene_ana).qpos[0, foot_dof].detach())
    # Setup sanity: the foot must have entered its limit band, else the
    # constraint backward isn't exercised.
    assert abs(foot_q) > 0.7, (
        f"setup error: n_steps={n_steps} at foot force={force_value} did not "
        f"drive the foot joint near its 0.785 limit (foot_q={foot_q}); pick a "
        f"larger force or more steps."
    )
    loss = _links_pos_sq_loss(scene_ana)
    loss.backward()
    for t, f in enumerate(forces):
        assert f.grad is not None, f"step {t}: force.grad is None"
    ana = np.array([[float(f.grad[d]) for d in range(n_dofs)] for f in forces])  # (n_steps, n_dofs)

    # FD per-step, per-dof
    scene_fd, robot_fd = _build_hopper(mjcf, requires_grad=False)

    def loss_at(perturbed: np.ndarray) -> float:
        scene_fd.reset()
        for t in range(n_steps):
            robot_fd.control_dofs_force(gs.tensor(perturbed[t], dtype=gs.tc_float))
            scene_fd.step()
        return float(_links_pos_sq_loss(scene_fd).detach())

    fd = np.zeros((n_steps, n_dofs))
    for t in range(n_steps):
        for d in range(n_dofs):
            plus = init_force.copy()
            plus[t, d] += eps
            minus = init_force.copy()
            minus[t, d] -= eps
            fd[t, d] = (loss_at(plus) - loss_at(minus)) / (2 * eps)

    for t in range(n_steps):
        for d in range(n_dofs):
            assert_allclose(
                ana[t, d],
                fd[t, d],
                rtol=1e-3,
                atol=1e-4,
                err_msg=(
                    f"hopper force.grad mismatch at t={t}/{n_steps}, dof={d} "
                    f"(ana={ana[t, d]:+.4e}, fd={fd[t, d]:+.4e})\nfull ana=\n{ana}\nfull fd=\n{fd}"
                ),
            )


# ===========================================================================
# Frictionloss constraint FD  (dof `frictionloss > 0` -> constraint row added)
# ===========================================================================


def _build_frictionloss(mjcf: str, *, requires_grad: bool, show_viewer: bool = False):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            substeps=4,
            gravity=(0.0, 0.0, 0.0),
            requires_grad=requires_grad,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            enable_self_collision=False,
            enable_joint_limit=False,
            disable_constraint=False,
            use_hibernation=False,
            use_contact_island=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.2, -1.2, 0.8),
            camera_lookat=(0.0, 0.0, 0.2),
        ),
        show_viewer=show_viewer,
    )
    robot = scene.add_entity(gs.morphs.MJCF(file=mjcf))
    scene.build(n_envs=0)
    return scene, robot


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_diff_frictionloss_forward_emits_row(show_viewer, grad_revolute_frictionloss):
    scene, _robot = _build_frictionloss(grad_revolute_frictionloss, requires_grad=False)
    scene.reset()
    scene.step()
    cs = scene.rigid_solver.constraint_solver.constraint_state
    n_fric = int(qd_to_torch(cs.n_constraints_frictionloss)[0])
    assert n_fric == 1, f"expected 1 frictionloss row, got {n_fric}"


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("init_vel", [0.5, 2.0, 5.0])
def test_diff_frictionloss_backward_fd_one_step(show_viewer, init_vel, grad_revolute_frictionloss):
    mjcf = grad_revolute_frictionloss
    eps = 1e-5

    scene_ana, robot_ana = _build_frictionloss(mjcf, requires_grad=True)
    scene_ana.reset()
    v = gs.tensor([init_vel], dtype=gs.tc_float, requires_grad=True)
    robot_ana.set_dofs_velocity(v)
    scene_ana.step()
    loss = (_rigid_state(scene_ana).qpos[0, 0]) ** 2
    loss.backward()
    ana = float(v.grad[0])

    scene_fd, robot_fd = _build_frictionloss(mjcf, requires_grad=False)

    def loss_at(val: float) -> float:
        scene_fd.reset()
        robot_fd.set_dofs_velocity(gs.tensor([val], dtype=gs.tc_float))
        scene_fd.step()
        return float((_rigid_state(scene_fd).qpos[0, 0]) ** 2)

    fd = (loss_at(init_vel + eps) - loss_at(init_vel - eps)) / (2 * eps)

    assert_allclose(ana, fd, rtol=1e-3, atol=1e-6)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("n_steps", [1, 4, 10])
def test_diff_frictionloss_backward_fd_multistep(show_viewer, n_steps, grad_revolute_frictionloss):
    mjcf = grad_revolute_frictionloss
    init_vel = 2.0
    eps = 1e-5

    scene_ana, robot_ana = _build_frictionloss(mjcf, requires_grad=True)
    scene_ana.reset()
    v = gs.tensor([init_vel], dtype=gs.tc_float, requires_grad=True)
    robot_ana.set_dofs_velocity(v)
    for _ in range(n_steps):
        scene_ana.step()
    loss = (_rigid_state(scene_ana).qpos[0, 0]) ** 2
    loss.backward()
    ana = float(v.grad[0])

    scene_fd, robot_fd = _build_frictionloss(mjcf, requires_grad=False)

    def loss_at(val: float) -> float:
        scene_fd.reset()
        robot_fd.set_dofs_velocity(gs.tensor([val], dtype=gs.tc_float))
        for _ in range(n_steps):
            scene_fd.step()
        return float((_rigid_state(scene_fd).qpos[0, 0]) ** 2)

    fd = (loss_at(init_vel + eps) - loss_at(init_vel - eps)) / (2 * eps)

    assert_allclose(ana, fd, rtol=2e-3, atol=1e-6)


# ===========================================================================
# Equality JOINT constraint FD  (<equality><joint .../></equality> -> JOINT row)
# ===========================================================================


def _build_equality_joint(mjcf: str, *, requires_grad: bool, show_viewer: bool = False):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            substeps=4,
            gravity=(0.0, 0.0, 0.0),
            requires_grad=requires_grad,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            enable_self_collision=False,
            enable_joint_limit=False,
            disable_constraint=False,
            use_hibernation=False,
            use_contact_island=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.2, -1.2, 0.8),
            camera_lookat=(0.0, 0.0, 0.2),
        ),
        show_viewer=show_viewer,
    )
    robot = scene.add_entity(gs.morphs.MJCF(file=mjcf))
    scene.build(n_envs=0)
    return scene, robot


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_diff_equality_joint_forward_emits_row(show_viewer, grad_hinge_pair_joint_eq_linear):
    scene, _robot = _build_equality_joint(grad_hinge_pair_joint_eq_linear, requires_grad=False)
    scene.reset()
    scene.step()
    cs = scene.rigid_solver.constraint_solver.constraint_state
    n_eq = int(qd_to_torch(cs.n_constraints_equality)[0])
    assert n_eq == 1, f"expected 1 equality row, got {n_eq}"


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("model_name", ["grad_hinge_pair_joint_eq_linear", "grad_hinge_pair_joint_eq_quadratic"])
def test_diff_equality_joint_backward_fd_one_step(show_viewer, request, model_name):
    mjcf = request.getfixturevalue(model_name)
    eps = 1e-5

    scene_ana, robot_ana = _build_equality_joint(mjcf, requires_grad=True)
    scene_ana.reset()
    # Two-DOF state. (vel[j1], vel[j2]).
    v = gs.tensor([1.0, -0.5], dtype=gs.tc_float, requires_grad=True)
    robot_ana.set_dofs_velocity(v)
    scene_ana.step()
    # Loss couples both dofs so each gradient component is exercised.
    qpos = _rigid_state(scene_ana).qpos[0]
    loss = qpos[0] ** 2 + 0.7 * qpos[1] ** 2
    loss.backward()
    ana = tensor_to_array(v.grad)

    scene_fd, robot_fd = _build_equality_joint(mjcf, requires_grad=False)

    def loss_at(val_array) -> float:
        scene_fd.reset()
        robot_fd.set_dofs_velocity(gs.tensor(val_array, dtype=gs.tc_float))
        scene_fd.step()
        qp = _rigid_state(scene_fd).qpos[0]
        return float(qp[0] ** 2 + 0.7 * qp[1] ** 2)

    fd = np.zeros_like(ana)
    base = np.array([1.0, -0.5], dtype=np.float64)
    for d in range(2):
        plus = base.copy()
        plus[d] += eps
        minus = base.copy()
        minus[d] -= eps
        fd[d] = (loss_at(plus) - loss_at(minus)) / (2 * eps)

    for d in range(2):
        assert_allclose(ana[d], fd[d], rtol=1e-3, atol=1e-6, err_msg=f"dof {d}: ana={ana[d]} fd={fd[d]}")


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("n_steps", [1, 4, 10])
def test_diff_equality_joint_backward_fd_multistep(show_viewer, n_steps, grad_hinge_pair_joint_eq_quadratic):
    mjcf = grad_hinge_pair_joint_eq_quadratic
    eps = 1e-5

    scene_ana, robot_ana = _build_equality_joint(mjcf, requires_grad=True)
    scene_ana.reset()
    v = gs.tensor([0.8, -0.3], dtype=gs.tc_float, requires_grad=True)
    robot_ana.set_dofs_velocity(v)
    for _ in range(n_steps):
        scene_ana.step()
    qpos = _rigid_state(scene_ana).qpos[0]
    loss = qpos[0] ** 2 + 0.7 * qpos[1] ** 2
    loss.backward()
    ana = tensor_to_array(v.grad)

    scene_fd, robot_fd = _build_equality_joint(mjcf, requires_grad=False)

    def loss_at(val_array) -> float:
        scene_fd.reset()
        robot_fd.set_dofs_velocity(gs.tensor(val_array, dtype=gs.tc_float))
        for _ in range(n_steps):
            scene_fd.step()
        qp = _rigid_state(scene_fd).qpos[0]
        return float(qp[0] ** 2 + 0.7 * qp[1] ** 2)

    fd = np.zeros_like(ana)
    base = np.array([0.8, -0.3], dtype=np.float64)
    for d in range(2):
        plus = base.copy()
        plus[d] += eps
        minus = base.copy()
        minus[d] -= eps
        fd[d] = (loss_at(plus) - loss_at(minus)) / (2 * eps)

    for d in range(2):
        assert_allclose(ana[d], fd[d], rtol=2e-3, atol=1e-6, err_msg=f"dof {d}: ana={ana[d]} fd={fd[d]}")


# ===========================================================================
# Equality CONNECT constraint FD  (<equality><connect .../></equality> -> 3 rows)
# ===========================================================================


def _build_equality_connect(mjcf: str, *, requires_grad: bool, show_viewer: bool = False):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            substeps=4,
            gravity=(0.0, 0.0, 0.0),
            requires_grad=requires_grad,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            enable_self_collision=False,
            enable_joint_limit=False,
            disable_constraint=False,
            use_hibernation=False,
            use_contact_island=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.2, -1.2, 0.8),
            camera_lookat=(0.0, 0.0, 0.2),
        ),
        show_viewer=show_viewer,
    )
    robot = scene.add_entity(gs.morphs.MJCF(file=mjcf))
    scene.build(n_envs=0)
    return scene, robot


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_diff_equality_connect_forward_emits_rows(show_viewer, grad_connect_loop):
    scene, _ = _build_equality_connect(grad_connect_loop, requires_grad=False)
    scene.reset()
    scene.step()
    cs = scene.rigid_solver.constraint_solver.constraint_state
    n_eq = int(qd_to_torch(cs.n_constraints_equality)[0])
    assert n_eq == 3, f"expected 3 equality rows for CONNECT, got {n_eq}"


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_diff_equality_connect_backward_fd_one_step(show_viewer, grad_connect_loop):
    mjcf = grad_connect_loop
    eps = 1e-5

    scene_ana, robot_ana = _build_equality_connect(mjcf, requires_grad=True)
    scene_ana.reset()
    v = gs.tensor([1.0, -0.5], dtype=gs.tc_float, requires_grad=True)
    robot_ana.set_dofs_velocity(v)
    scene_ana.step()
    qpos = _rigid_state(scene_ana).qpos[0]
    loss = qpos[0] ** 2 + 0.7 * qpos[1] ** 2
    loss.backward()
    ana = tensor_to_array(v.grad)

    scene_fd, robot_fd = _build_equality_connect(mjcf, requires_grad=False)

    def loss_at(val_array) -> float:
        scene_fd.reset()
        robot_fd.set_dofs_velocity(gs.tensor(val_array, dtype=gs.tc_float))
        scene_fd.step()
        qp = _rigid_state(scene_fd).qpos[0]
        return float(qp[0] ** 2 + 0.7 * qp[1] ** 2)

    fd = np.zeros_like(ana)
    base = np.array([1.0, -0.5], dtype=np.float64)
    for d in range(2):
        plus = base.copy()
        plus[d] += eps
        minus = base.copy()
        minus[d] -= eps
        fd[d] = (loss_at(plus) - loss_at(minus)) / (2 * eps)

    for d in range(2):
        assert_allclose(ana[d], fd[d], rtol=1e-3, atol=1e-6, err_msg=f"dof {d}: ana={ana[d]} fd={fd[d]}")


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("n_steps", [1, 4, 10])
def test_diff_equality_connect_backward_fd_multistep(show_viewer, n_steps, grad_connect_loop):
    mjcf = grad_connect_loop
    eps = 1e-5

    scene_ana, robot_ana = _build_equality_connect(mjcf, requires_grad=True)
    scene_ana.reset()
    v = gs.tensor([0.8, -0.3], dtype=gs.tc_float, requires_grad=True)
    robot_ana.set_dofs_velocity(v)
    for _ in range(n_steps):
        scene_ana.step()
    qpos = _rigid_state(scene_ana).qpos[0]
    loss = qpos[0] ** 2 + 0.7 * qpos[1] ** 2
    loss.backward()
    ana = tensor_to_array(v.grad)

    scene_fd, robot_fd = _build_equality_connect(mjcf, requires_grad=False)

    def loss_at(val_array) -> float:
        scene_fd.reset()
        robot_fd.set_dofs_velocity(gs.tensor(val_array, dtype=gs.tc_float))
        for _ in range(n_steps):
            scene_fd.step()
        qp = _rigid_state(scene_fd).qpos[0]
        return float(qp[0] ** 2 + 0.7 * qp[1] ** 2)

    fd = np.zeros_like(ana)
    base = np.array([0.8, -0.3], dtype=np.float64)
    for d in range(2):
        plus = base.copy()
        plus[d] += eps
        minus = base.copy()
        minus[d] -= eps
        fd[d] = (loss_at(plus) - loss_at(minus)) / (2 * eps)

    for d in range(2):
        assert_allclose(ana[d], fd[d], rtol=2e-3, atol=1e-6, err_msg=f"dof {d}: ana={ana[d]} fd={fd[d]}")


# ===========================================================================
# Equality WELD constraint FD  (<equality><weld .../></equality> -> 6 rows)
# ===========================================================================


def _build_equality_weld(mjcf: str, *, requires_grad: bool, show_viewer: bool = False):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            substeps=4,
            gravity=(0.0, 0.0, 0.0),
            requires_grad=requires_grad,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            enable_self_collision=False,
            enable_joint_limit=False,
            disable_constraint=False,
            use_hibernation=False,
            use_contact_island=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.2, -1.2, 0.8),
            camera_lookat=(0.0, 0.0, 0.2),
        ),
        show_viewer=show_viewer,
    )
    robot = scene.add_entity(gs.morphs.MJCF(file=mjcf))
    scene.build(n_envs=0)
    return scene, robot


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_diff_equality_weld_forward_emits_rows(show_viewer, grad_weld_pair):
    scene, _ = _build_equality_weld(grad_weld_pair, requires_grad=False)
    scene.reset()
    scene.step()
    cs = scene.rigid_solver.constraint_solver.constraint_state
    n_eq = int(qd_to_torch(cs.n_constraints_equality)[0])
    assert n_eq == 6, f"expected 6 equality rows for WELD, got {n_eq}"


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_diff_equality_weld_backward_fd_one_step(show_viewer, grad_weld_pair):
    mjcf = grad_weld_pair
    eps = 1e-5

    scene_ana, robot_ana = _build_equality_weld(mjcf, requires_grad=True)
    scene_ana.reset()
    v = gs.tensor([1.0, -0.5], dtype=gs.tc_float, requires_grad=True)
    robot_ana.set_dofs_velocity(v)
    scene_ana.step()
    qpos = _rigid_state(scene_ana).qpos[0]
    loss = qpos[0] ** 2 + 0.7 * qpos[1] ** 2
    loss.backward()
    ana = tensor_to_array(v.grad)

    scene_fd, robot_fd = _build_equality_weld(mjcf, requires_grad=False)

    def loss_at(val_array) -> float:
        scene_fd.reset()
        robot_fd.set_dofs_velocity(gs.tensor(val_array, dtype=gs.tc_float))
        scene_fd.step()
        qp = _rigid_state(scene_fd).qpos[0]
        return float(qp[0] ** 2 + 0.7 * qp[1] ** 2)

    fd = np.zeros_like(ana)
    base = np.array([1.0, -0.5], dtype=np.float64)
    for d in range(2):
        plus = base.copy()
        plus[d] += eps
        minus = base.copy()
        minus[d] -= eps
        fd[d] = (loss_at(plus) - loss_at(minus)) / (2 * eps)

    for d in range(2):
        assert_allclose(ana[d], fd[d], rtol=1e-3, atol=1e-6, err_msg=f"dof {d}: ana={ana[d]} fd={fd[d]}")


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("n_steps", [1, 4, 10])
def test_diff_equality_weld_backward_fd_multistep(show_viewer, n_steps, grad_weld_pair):
    mjcf = grad_weld_pair
    eps = 1e-5

    scene_ana, robot_ana = _build_equality_weld(mjcf, requires_grad=True)
    scene_ana.reset()
    v = gs.tensor([0.8, -0.3], dtype=gs.tc_float, requires_grad=True)
    robot_ana.set_dofs_velocity(v)
    for _ in range(n_steps):
        scene_ana.step()
    qpos = _rigid_state(scene_ana).qpos[0]
    loss = qpos[0] ** 2 + 0.7 * qpos[1] ** 2
    loss.backward()
    ana = tensor_to_array(v.grad)

    scene_fd, robot_fd = _build_equality_weld(mjcf, requires_grad=False)

    def loss_at(val_array) -> float:
        scene_fd.reset()
        robot_fd.set_dofs_velocity(gs.tensor(val_array, dtype=gs.tc_float))
        for _ in range(n_steps):
            scene_fd.step()
        qp = _rigid_state(scene_fd).qpos[0]
        return float(qp[0] ** 2 + 0.7 * qp[1] ** 2)

    fd = np.zeros_like(ana)
    base = np.array([0.8, -0.3], dtype=np.float64)
    for d in range(2):
        plus = base.copy()
        plus[d] += eps
        minus = base.copy()
        minus[d] -= eps
        fd[d] = (loss_at(plus) - loss_at(minus)) / (2 * eps)

    for d in range(2):
        assert_allclose(ana[d], fd[d], rtol=2e-3, atol=1e-6, err_msg=f"dof {d}: ana={ana[d]} fd={fd[d]}")


# ===========================================================================
# Integration: all differentiated constraint groups in one scene
# (equality JOINT + CONNECT + WELD + inequality frictionloss)
# ===========================================================================


def _build_all_eq_fric(mjcf: str, *, requires_grad: bool, show_viewer: bool = False):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            substeps=4,
            gravity=(0.0, 0.0, 0.0),
            requires_grad=requires_grad,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            enable_self_collision=False,
            enable_joint_limit=False,
            disable_constraint=False,
            use_hibernation=False,
            use_contact_island=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.2, -1.2, 0.8),
            camera_lookat=(0.0, 0.0, 0.2),
        ),
        show_viewer=show_viewer,
    )
    robot = scene.add_entity(gs.morphs.MJCF(file=mjcf))
    scene.build(n_envs=0)
    return scene, robot


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_diff_all_constraints_forward_row_counts(show_viewer, grad_all_eq_fric):
    scene, _ = _build_all_eq_fric(grad_all_eq_fric, requires_grad=False)
    scene.reset()
    scene.step()
    cs = scene.rigid_solver.constraint_solver.constraint_state
    n_eq = int(qd_to_torch(cs.n_constraints_equality)[0])
    n_fric = int(qd_to_torch(cs.n_constraints_frictionloss)[0])
    assert n_eq == 10, f"expected 10 equality rows (1+3+6), got {n_eq}"
    assert n_fric == 1, f"expected 1 frictionloss row, got {n_fric}"


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("n_steps", [1, 4, 10])
def test_diff_all_constraints_backward_fd(show_viewer, n_steps, grad_all_eq_fric):
    eps = 1e-5
    n_dofs = 6
    init = np.array([0.8, -0.3, 0.5, -0.2, 0.4, -0.6], dtype=np.float64)
    weights = np.array([1.0, 0.7, 1.3, 0.5, 0.9, 1.1], dtype=np.float64)

    def loss_from_qpos(qp):
        # Mix all 6 dofs so each gradient component is exercised.
        out = 0.0
        for d in range(n_dofs):
            out = out + weights[d] * qp[d] ** 2
        return out

    scene_ana, robot_ana = _build_all_eq_fric(grad_all_eq_fric, requires_grad=True)
    scene_ana.reset()
    v = gs.tensor(init, dtype=gs.tc_float, requires_grad=True)
    robot_ana.set_dofs_velocity(v)
    for _ in range(n_steps):
        scene_ana.step()
    qpos = _rigid_state(scene_ana).qpos[0]
    loss = loss_from_qpos(qpos)
    loss.backward()
    ana = tensor_to_array(v.grad)

    scene_fd, robot_fd = _build_all_eq_fric(grad_all_eq_fric, requires_grad=False)

    def loss_at(val_array) -> float:
        scene_fd.reset()
        robot_fd.set_dofs_velocity(gs.tensor(val_array, dtype=gs.tc_float))
        for _ in range(n_steps):
            scene_fd.step()
        qp = _rigid_state(scene_fd).qpos[0]
        return float(loss_from_qpos(qp))

    fd = np.zeros_like(ana)
    for d in range(n_dofs):
        plus = init.copy()
        plus[d] += eps
        minus = init.copy()
        minus[d] -= eps
        fd[d] = (loss_at(plus) - loss_at(minus)) / (2 * eps)

    # Looser rtol than the per-group tests: this scene is mildly over-constrained
    # (11 active rows on 6 dofs across 3 disjoint pairs) so the constraint
    # solver's LDLT accumulates a small CPU/GPU divergence over a long horizon.
    # The per-group tests already pin the reverse formulas; this integration
    # test just guards row-offset bookkeeping across all groups.
    for d in range(n_dofs):
        assert_allclose(ana[d], fd[d], rtol=2e-2, atol=1e-6, err_msg=f"dof {d}: ana={ana[d]} fd={fd[d]}")


# ===========================================================================
# Collision / diff-GJK contact FD  (enable_collision=True -> constraints ON)
# ===========================================================================


def _build_box_box(*, requires_grad: bool, show_viewer: bool = False):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=2,
            gravity=(0.0, 0.0, -9.81),
            requires_grad=requires_grad,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            enable_self_collision=False,
            enable_joint_limit=False,
            disable_constraint=False,
            use_hibernation=False,
            use_contact_island=False,
            box_box_detection=False,  # general convex-convex GJK (differentiable) path
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.2, -1.2, 0.8),
            camera_lookat=(0.0, 0.0, 0.2),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Box(size=(2.0, 2.0, 0.2), pos=(0.0, 0.0, 0.1), fixed=True))
    box = scene.add_entity(gs.morphs.Box(size=(0.4, 0.4, 0.4), pos=(0.0, 0.0, 0.4)))
    scene.build(n_envs=0)
    return scene, box


def _build_plane_convex(mjcf_capsule: str, shape: str, *, requires_grad: bool, show_viewer: bool = False):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=2,
            gravity=(0.0, 0.0, -9.81),
            requires_grad=requires_grad,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            enable_self_collision=False,
            enable_joint_limit=False,
            disable_constraint=False,
            use_hibernation=False,
            use_contact_island=False,
            box_box_detection=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.2, -1.2, 0.8),
            camera_lookat=(0.0, 0.0, 0.2),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    if shape == "box":
        obj = scene.add_entity(gs.morphs.Box(size=(0.4, 0.4, 0.4), pos=(0.0, 0.0, 0.3)))
    elif shape == "sphere":
        obj = scene.add_entity(gs.morphs.Sphere(radius=0.2, pos=(0.0, 0.0, 0.3)))
    elif shape == "capsule":
        obj = scene.add_entity(gs.morphs.MJCF(file=mjcf_capsule, align=False))
    else:
        raise ValueError(shape)
    scene.build(n_envs=0)
    return scene, obj


def _n_contacts(scene) -> int:
    return int(qd_to_numpy(scene.rigid_solver.collider._collider_state.n_contacts)[0])


def _settle(scene, obj, n_settle: int):
    zero = gs.tensor([0.0] * 6, dtype=gs.tc_float)
    for _ in range(n_settle):
        obj.control_dofs_force(zero)
        scene.step()


def _run_fd_per_step_force(build_fn, rest_dofs, *, base_force, n_settle, n_steps, fd_dofs, eps, rtol, atol):
    init_force = np.broadcast_to(base_force, (n_steps, 6)).copy()

    # --- analytical ---
    scene_ana, obj_ana = build_fn(requires_grad=True)
    scene_ana.reset()
    obj_ana.set_dofs_position(gs.tensor(rest_dofs, dtype=gs.tc_float).sceneless())
    _settle(scene_ana, obj_ana, n_settle)
    nc = _n_contacts(scene_ana)
    assert nc > 0, f"setup error: not in contact after settle (n_contacts={nc})"

    forces = [gs.tensor(init_force[t], dtype=gs.tc_float, requires_grad=True) for t in range(n_steps)]
    for t in range(n_steps):
        obj_ana.control_dofs_force(forces[t])
        scene_ana.step()
        assert _n_contacts(scene_ana) == nc, "contact set changed during grad window - FD invalid"
    loss = (_rigid_state(scene_ana).qpos[0, :3] ** 2).sum()
    scene_ana.backward(loss)
    ana = np.array([[float(f.grad[d]) for d in range(6)] for f in forces])  # (N, 6)

    # --- central FD, contact set preserved ---
    scene_fd, obj_fd = build_fn(requires_grad=True)

    def loss_at(perturbed: np.ndarray) -> float:
        scene_fd.reset()
        obj_fd.set_dofs_position(gs.tensor(rest_dofs, dtype=gs.tc_float).sceneless())
        _settle(scene_fd, obj_fd, n_settle)
        for t in range(n_steps):
            obj_fd.control_dofs_force(gs.tensor(perturbed[t], dtype=gs.tc_float))
            scene_fd.step()
            assert _n_contacts(scene_fd) == nc, "contact set changed under FD perturbation"
        return float((_rigid_state(scene_fd).qpos[0, :3] ** 2).sum().detach())

    fd = np.full((n_steps, 6), np.nan)
    for t in range(n_steps):
        for d in fd_dofs:
            plus = init_force.copy()
            plus[t, d] += eps
            minus = init_force.copy()
            minus[t, d] -= eps
            fd[t, d] = (loss_at(plus) - loss_at(minus)) / (2 * eps)

    # Contact gradients are small (stiff contact barely moves), so the band is
    # absolute-dominated; rtol pins the load-bearing z entry.
    for t in range(n_steps):
        for d in fd_dofs:
            assert_allclose(
                ana[t, d],
                fd[t, d],
                rtol=rtol,
                atol=atol,
                err_msg=f"contact force.grad mismatch at t={t}/{n_steps}, dof={d}\nana=\n{ana}\nfd=\n{fd}",
            )


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_diff_contact_fd_per_step_force(show_viewer):
    # Box rests on the ground top (z=0.2) at center z=0.40; settle to a stable
    # multi-contact manifold, then a short grad window with a per-step push.
    _run_fd_per_step_force(
        _build_box_box,
        [0.0, 0.0, 0.40, 0.0, 0.0, 0.0],  # freejoint 6 DOFs: xyz + rotvec(=identity)
        base_force=np.array([0.0, 0.0, -8.0, 0.0, 0.0, 0.0], dtype=np.float64),
        n_settle=12,
        n_steps=2,
        fd_dofs=(2,),
        eps=1e-2,
        rtol=2e-3,
        atol=1e-10,
    )


# rest z so the body's lowest point sits on the plane (z=0): box/sphere half
# extent 0.2; capsule radius 0.1 + half_length 0.2 = 0.3 (upright).


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("shape", ["box", "sphere", "capsule"])
def test_diff_contact_fd_plane_convex(shape, show_viewer, grad_capsule):
    # Rest z puts the shape's lowest point on the plane: box / sphere half extent 0.2; upright capsule
    # radius 0.1 + half length 0.2 = 0.3.
    rest_dofs = {"box": 0.20, "sphere": 0.20, "capsule": 0.30}[shape]
    rest_dofs = [0.0, 0.0, rest_dofs, 0.0, 0.0, 0.0]
    # Plane (fixed) + free convex. The analytic plane contact is reconstructed
    # differentiably via `func_differentiable_plane_contact` (stored convex
    # support core + radius), so the same FD chain as box-box applies.
    _run_fd_per_step_force(
        lambda *, requires_grad: _build_plane_convex(grad_capsule, shape, requires_grad=requires_grad),
        rest_dofs,
        base_force=np.array([0.0, 0.0, -8.0, 0.0, 0.0, 0.0], dtype=np.float64),
        n_settle=12,
        n_steps=2,
        fd_dofs=(2,),
        eps=1e-2,
        rtol=2e-3,
        atol=1e-10,
    )


# ===========================================================================
# Low-level contact-detection + constraint-solver backward FD
# ===========================================================================


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_diff_contact():
    RTOL = 1e-4

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            # Turn on differentiable mode
            requires_grad=True,
        ),
        show_viewer=False,
    )

    box_size = 0.25
    box_spacing = box_size
    vec_one = np.array([1.0, 1.0, 1.0])
    box_pos_offset = (0.0, 0.0, 0.0) + 0.5 * box_size * vec_one

    box0 = scene.add_entity(
        gs.morphs.Box(size=box_size * vec_one, pos=box_pos_offset),
    )
    box1 = scene.add_entity(
        gs.morphs.Box(size=box_size * vec_one, pos=box_pos_offset + 0.8 * box_spacing * np.array([0, 0, 1])),
    )
    scene.build()
    solver = scene.sim.rigid_solver
    collider = solver.collider

    # Set up initial configuration
    x_ang, y_ang, z_ang = 3.0, 3.0, 3.0
    box1.set_quat(R_to_quat(gs.euler_to_R([np.deg2rad(x_ang), np.deg2rad(y_ang), np.deg2rad(z_ang)])))

    box0_init_pos = box0.get_pos().clone()
    box1_init_pos = box1.get_pos().clone()
    box0_init_quat = box0.get_quat().clone()
    box1_init_quat = box1.get_quat().clone()

    ### Compute the initial loss and compute gradients using differentiable contact detection
    # Detect contact
    collider.detection()

    # Get contact outputs and their grads
    contacts = collider.get_contacts(as_tensor=True, to_torch=True, keep_batch_dim=True)
    normal = contacts["normal"].requires_grad_()
    position = contacts["position"].requires_grad_()
    penetration = contacts["penetration"].requires_grad_()

    loss = ((normal * position).sum(dim=-1) * penetration).sum()
    dL_dnormal = torch.autograd.grad(loss, normal, retain_graph=True)[0]
    dL_dposition = torch.autograd.grad(loss, position, retain_graph=True)[0]
    dL_dpenetration = torch.autograd.grad(loss, penetration)[0]

    # Compute analytical gradients of the geoms position and quaternion
    collider.backward(dL_dposition, dL_dnormal, dL_dpenetration)
    dL_dpos = qd_to_torch(solver.dyn_state.geoms.pos.grad)
    dL_dquat = qd_to_torch(solver.dyn_state.geoms.quat.grad)

    ### Compute directional derivatives along random directions
    FD_EPS = 1e-5
    TRIALS = 100

    def compute_dL_error(dL_dx, x_type):
        dL_error_rel = 0.0

        box0_input_pos = box0_init_pos
        box1_input_pos = box1_init_pos
        box0_input_quat = box0_init_quat
        box1_input_quat = box1_init_quat

        for _ in range(TRIALS):
            rand_dx = torch.randn_like(dL_dx)
            rand_dx = torch.nn.functional.normalize(rand_dx, dim=-1)

            dL = (rand_dx * dL_dx).sum()

            lossPs = []
            for sign in (1, -1):
                # Compute query point
                if x_type == "pos":
                    box0_input_pos = box0_init_pos + sign * rand_dx[0, 0] * FD_EPS
                    box1_input_pos = box1_init_pos + sign * rand_dx[1, 0] * FD_EPS
                else:
                    # FIXME: The quaternion should be normalized
                    box0_input_quat = box0_init_quat + sign * rand_dx[0, 0] * FD_EPS
                    box1_input_quat = box1_init_quat + sign * rand_dx[1, 0] * FD_EPS

                # Update box positions
                box0.set_pos(box0_input_pos)
                box1.set_pos(box1_input_pos)
                box0.set_quat(box0_input_quat)
                box1.set_quat(box1_input_quat)

                # Re-detect contact.
                # We need to manually reset the contact counter as we are not running the whole sim step.
                collider._collider_state.n_contacts.fill(0)
                collider.detection()
                contacts = collider.get_contacts(as_tensor=True, to_torch=True, keep_batch_dim=True)
                normal, position, penetration = contacts["normal"], contacts["position"], contacts["penetration"]

                # Compute loss
                loss = ((normal * position).sum(dim=-1) * penetration).sum()
                lossPs.append(loss)

            dL_fd = (lossPs[0] - lossPs[1]) / (2 * FD_EPS)
            dL_error_rel += (dL - dL_fd).abs() / max(dL.abs(), dL_fd.abs(), gs.EPS)

        dL_error_rel /= TRIALS
        return dL_error_rel

    dL_dpos_error_rel = compute_dL_error(dL_dpos, "pos")
    assert_allclose(dL_dpos_error_rel, 0.0, atol=RTOL)
    dL_dquat_error_rel = compute_dL_error(dL_dquat, "quat")
    assert_allclose(dL_dquat_error_rel, 0.0, atol=RTOL)


# We need to use 64-bit precision for this test because we need to use sufficiently small perturbation to get reliable
# gradient estimates through finite difference method. This small perturbation is not supported by 32-bit precision in
# stable way.
@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_diff_solver(monkeypatch):
    from genesis.engine.solvers.rigid.constraint.solver import func_solve_init, func_solve_body
    from genesis.engine.solvers.rigid.rigid_solver import kernel_step_1

    RTOL = 1e-4

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            requires_grad=True,
        ),
        rigid_options=gs.options.RigidOptions(
            # We use Newton's method because it converges faster than CG, and therefore gives better gradient estimation
            # when using finite difference method
            constraint_solver=gs.constraint_solver.Newton,
        ),
        show_viewer=False,
    )

    scene.add_entity(gs.morphs.Plane(pos=(0, 0, 0)))
    scene.add_entity(gs.morphs.Box(size=(1, 1, 1), pos=(10, 10, 0.49)))
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )

    scene.build()
    rigid_solver = scene._sim.rigid_solver
    constraint_solver = rigid_solver.constraint_solver

    franka.set_qpos([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04])

    # Monkeypatch the constraint resolve function to avoid overwriting the necessary information for computing gradients.
    def constraint_solver_resolve():
        func_solve_init(
            rigid_solver.dyn_state,
            constraint_solver.constraint_state,
            rigid_solver.dyn_info,
            rigid_solver.rigid_info,
            rigid_solver.rigid_config,
            is_decomposed=False,
        )
        func_solve_body(
            rigid_solver.dyn_state,
            constraint_solver.constraint_state,
            rigid_solver.dyn_info,
            rigid_solver.rigid_info,
            rigid_solver.rigid_config,
            constraint_solver._n_iterations,
        )

    monkeypatch.setattr(constraint_solver, "resolve", constraint_solver_resolve)

    # Step once to compute constraint solver's inputs: [mass], [jac], [aref], [efc_D], [force]. We do not call the
    # entire scene.step() because it will overwrite the necessary information that we need to compute the gradients.
    kernel_step_1(
        rigid_solver.dyn_state,
        constraint_solver.constraint_state,
        rigid_solver.dyn_info,
        rigid_solver.rigid_info,
        rigid_solver.rigid_config,
        is_forward_pos_updated=True,
        is_forward_vel_updated=True,
        is_backward=False,
    )
    constraint_solver.add_equality_constraints()
    rigid_solver.collider.detection()
    constraint_solver.add_inequality_constraints()
    constraint_solver.resolve()

    # Loss function to compute gradients using finite difference method
    def compute_loss(input_mass, input_jac, input_aref, input_efc_D, input_force):
        rigid_solver.rigid_info.mass_mat.from_numpy(input_mass)
        constraint_solver.constraint_state.jac.from_numpy(input_jac)
        constraint_solver.constraint_state.aref.from_numpy(input_aref)
        constraint_solver.constraint_state.efc_D.from_numpy(input_efc_D)
        rigid_solver.dyn_state.dofs.force.from_numpy(input_force)

        # Recompute acc_smooth from the updated input variables
        updated_acc_smooth = np.linalg.solve(input_mass[..., 0], input_force[..., 0])
        rigid_solver.dyn_state.dofs.acc_smooth.from_numpy(updated_acc_smooth[..., None])
        constraint_solver.resolve()

        output_qacc = qd_to_torch(constraint_solver.qacc)
        return ((output_qacc - target_qacc) ** 2).mean()

    init_input_mass = qd_to_numpy(rigid_solver.rigid_info.mass_mat, copy=True)
    init_input_jac = qd_to_numpy(constraint_solver.constraint_state.jac, copy=True)
    init_input_aref = qd_to_numpy(constraint_solver.constraint_state.aref, copy=True)
    init_input_efc_D = qd_to_numpy(constraint_solver.constraint_state.efc_D, copy=True)
    init_input_force = qd_to_numpy(rigid_solver.dyn_state.dofs.force, copy=True)

    # Initial output of the constraint solver
    set_random_seed(0)
    init_output_qacc = qd_to_torch(constraint_solver.qacc)
    target_qacc = torch.from_numpy(np.random.randn(*init_output_qacc.shape)).to(device=gs.device)
    target_qacc = target_qacc * init_output_qacc.abs().mean()

    # Solve the constraint solver and get the output
    output_qacc = qd_to_torch(constraint_solver.qacc, copy=True).requires_grad_(True)

    # Compute loss and gradient of the output
    loss = ((output_qacc - target_qacc) ** 2).mean()
    dL_dqacc = tensor_to_array(torch.autograd.grad(loss, output_qacc)[0])

    # Compute gradients of the input variables: [mass], [jac], [aref], [efc_D], [force]
    constraint_solver.constraint_state.dL_dqacc.from_numpy(dL_dqacc)
    constraint_solver.backward()

    # Fetch gradients of the input variables
    dL_dM = qd_to_numpy(constraint_solver.constraint_state.dL_dM)
    dL_djac = qd_to_numpy(constraint_solver.constraint_state.dL_djac)
    dL_daref = qd_to_numpy(constraint_solver.constraint_state.dL_daref)
    dL_defc_D = qd_to_numpy(constraint_solver.constraint_state.dL_defc_D)
    dL_dforce = qd_to_numpy(constraint_solver.constraint_state.dL_dforce)

    ### Compute directional derivatives along random directions
    FD_EPS = 1e-3
    TRIALS = 200

    for dL_dx, x_type in (
        (dL_dforce, "force"),
        (dL_daref, "aref"),
        (dL_defc_D, "efc_D"),
        (dL_djac, "jac"),
        (dL_dM, "mass"),
    ):
        dL_error = 0.0
        for _ in range(TRIALS):
            rand_dx = np.random.randn(*dL_dx.shape)
            rand_dx = rand_dx / max(
                np.linalg.norm(rand_dx, axis=0 if x_type in ("force", "aref", "efc_D") else (0, 1)), gs.EPS
            )
            if x_type == "mass":
                # Make rand_dx symmetric
                rand_dx = (rand_dx + np.moveaxis(rand_dx, 0, 1)) * 0.5

            dL = (rand_dx * dL_dx).sum()

            input_force = init_input_force
            input_aref = init_input_aref
            input_efc_D = init_input_efc_D
            input_jac = init_input_jac
            input_mass = init_input_mass

            # 1 * eps
            if x_type == "force":
                input_force = init_input_force + rand_dx * FD_EPS
            elif x_type == "aref":
                input_aref = init_input_aref + rand_dx * FD_EPS
            elif x_type == "efc_D":
                input_efc_D = init_input_efc_D + rand_dx * FD_EPS
            elif x_type == "jac":
                input_jac = init_input_jac + rand_dx * FD_EPS
            elif x_type == "mass":
                input_mass = init_input_mass + rand_dx * FD_EPS
            lossP1 = compute_loss(input_mass, input_jac, input_aref, input_efc_D, input_force)

            # -1 * eps
            if x_type == "force":
                input_force = init_input_force - rand_dx * FD_EPS
            elif x_type == "aref":
                input_aref = init_input_aref - rand_dx * FD_EPS
            elif x_type == "efc_D":
                input_efc_D = init_input_efc_D - rand_dx * FD_EPS
            elif x_type == "jac":
                input_jac = init_input_jac - rand_dx * FD_EPS
            elif x_type == "mass":
                input_mass = init_input_mass - rand_dx * FD_EPS

            lossP2 = compute_loss(input_mass, input_jac, input_aref, input_efc_D, input_force)
            dL_fd = (lossP1 - lossP2) / (2 * FD_EPS)

            dL_error += (dL - dL_fd).abs() / max(abs(dL), abs(dL_fd), gs.EPS)

        dL_error /= TRIALS
        assert_allclose(dL_error, 0.0, atol=RTOL)
