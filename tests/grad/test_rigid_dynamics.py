# Finite-difference vs analytical reverse-mode gradient checks for rigid forward kinematics (constraints off), one
# packed test per joint topology exercising every tracked setter, and a multi-step control-force adjoint check.
import sys

import numpy as np
import pytest

import genesis as gs

from ..conftest import SKIP_METAL_GRAD
from .utils import assert_grad_matches_fd, make_diff_scene_pair


@pytest.mark.required
@pytest.mark.parametrize(
    "backend",
    [
        gs.cpu,
        # FIXME: Quadrants' released native-Metal reverse-mode autodiff collapses per-env adjoints (fixed upstream,
        # see Quadrants issue #805). Re-enable once the fix ships in a Quadrants release.
        pytest.param(gs.gpu, marks=pytest.mark.skipif(sys.platform == "darwin", reason=SKIP_METAL_GRAD)),
    ],
)
@pytest.mark.parametrize(
    "model_name",
    [
        "grad_free",
        "grad_revolute",
        "grad_prismatic",
        "grad_spherical",
        "grad_free_with_revolute",
        "grad_revolute_chain3",
        "grad_cartpole",
        "grad_hopper",
    ],
)
def test_rigid_fk_grad_matches_fd(model_name, request, precision, show_viewer):
    pair = make_diff_scene_pair(request.getfixturevalue(model_name), n_envs=2, substeps=4, show_viewer=show_viewer)
    n_dofs = pair.entity_ana.n_dofs
    n_links = pair.entity_ana.n_links
    B = 2

    # Single-link joints read the entity pose; multi-link topologies read the rigid-solver per-link pose.
    is_single_link = model_name in ("grad_free", "grad_revolute", "grad_prismatic", "grad_spherical")
    # (setter, output, input_seed) per joint, plus the position and quaternion target seeds. Single-step
    # force->position is only checked on cartpole, where its sensitivity clears the finite-difference floor.
    checks_by_joint = {
        "grad_free": ((("pos", "pos", 10), ("quat", "quat", 11), ("vel", "pos", 12), ("vel", "quat", 13)), 1, 2),
        "grad_revolute": ((("vel", "pos", 30), ("vel", "quat", 31), ("force", "quat", 32)), 21, 22),
        "grad_prismatic": ((("vel", "pos", 50),), 41, 0),
        "grad_spherical": ((("vel", "pos", 70), ("vel", "quat", 71), ("force", "quat", 72)), 61, 62),
        "grad_free_with_revolute": (
            (("pos", "pos", 70), ("quat", "quat", 71), ("vel", "pos", 72), ("vel", "quat", 73)),
            61,
            62,
        ),
        "grad_revolute_chain3": ((("vel", "pos", 90), ("vel", "quat", 91)), 81, 82),
        "grad_cartpole": ((("vel", "pos", 190), ("vel", "quat", 191), ("force", "pos", 192)), 181, 182),
        "grad_hopper": ((("vel", "pos", 210), ("vel", "quat", 211)), 201, 202),
    }
    checks, pos_seed, quat_seed = checks_by_joint[model_name]

    # Per-topology fp32 finite-difference floor (tolerance, step): quaternion and multi-link chain topologies
    # (free, free_with_revolute, hopper) are noisier at fp32 and need a smaller step, while the single-DOF cases
    # are far cleaner and pin down to ~1e-5. fp64 clears 1e-9 for every topology, so it stays a single band.
    fp32_tol, fp32_eps = {
        "grad_free": (2e-4, 1e-2),
        "grad_revolute": (5e-5, 3e-2),
        "grad_prismatic": (5e-6, 3e-2),
        "grad_spherical": (1e-4, 3e-2),
        "grad_free_with_revolute": (5e-4, 1e-2),
        "grad_revolute_chain3": (2e-4, 3e-2),
        "grad_cartpole": (2e-4, 3e-2),
        "grad_hopper": (5e-4, 3e-2),
    }[model_name]

    pos_shape = (B, 3) if is_single_link else (B, n_links, 3)
    quat_shape = (B, 4) if is_single_link else (B, n_links, 4)
    tgt_pos = gs.tensor(np.random.RandomState(pos_seed).standard_normal(pos_shape), dtype=gs.tc_float).reshape(-1)
    tgt_quat = gs.tensor(np.random.RandomState(quat_seed).standard_normal(quat_shape), dtype=gs.tc_float).reshape(-1)
    for setter, output, input_seed in checks:
        rng = np.random.default_rng(input_seed)
        if setter == "pos":
            step_input = rng.standard_normal((B, 3))
        elif setter == "quat":
            step_input = np.broadcast_to(np.array([1.0, 0.0, 0.0, 0.0]), (B, 4)).copy()
            step_input = step_input + 0.05 * rng.standard_normal((B, 4))
            step_input = step_input / np.linalg.norm(step_input, axis=-1, keepdims=True)
        else:
            step_input = rng.standard_normal((B, n_dofs))

        apply_fn = {
            "pos": lambda e, x: e.set_pos(x),
            "quat": lambda e, x: e.set_quat(x),
            "vel": lambda e, x: e.set_dofs_velocity(x),
            "force": lambda e, x: e.control_dofs_force(x),
        }[setter]

        target = tgt_pos if output == "pos" else tgt_quat

        def loss_fn(scene, entity, tgt=target, out=output, sl=is_single_link):
            if sl:
                pose = entity.get_state().pos if out == "pos" else entity.get_state().quat
            else:
                state = scene.rigid_solver.get_state()
                pose = state.links_pos if out == "pos" else state.links_quat
            return ((pose.reshape(-1) - tgt) ** 2).sum()

        assert_grad_matches_fd(
            pair,
            [step_input],
            apply_fn,
            loss_fn,
            rtol=1e-9 if precision == "64" else fp32_tol,
            atol=1e-9 if precision == "64" else fp32_tol,
            eps=3e-5 if precision == "64" else fp32_eps,
        )


@pytest.mark.required
@pytest.mark.parametrize(
    "model_name",
    [
        "grad_free",
        "grad_revolute",
        "grad_prismatic",
        "grad_free_with_revolute",
        "grad_revolute_chain3",
        "grad_spherical",
        "grad_cartpole",
        "grad_hopper",
    ],
)
@pytest.mark.debug(False)
def test_rigid_fk_multistep_force_grad_matches_fd(model_name, request, precision, show_viewer):
    # Ten distinct per-step control forces, each of which must receive an independent adjoint across the unroll.
    # (output kind: entity state vs rigid-solver links, per-link output shape, target seed, fp32 tolerance). The
    # per-topology fp32 floor spans 2e-6 (prismatic) to 2e-4 (hopper), tracking how far the ten-step unroll
    # amplifies fp32 noise; fp64 clears 5e-9 for all, set by the chaotic hopper/chain3 chains.
    output, output_shape, seed, fp32_tol = {
        "grad_free": ("state", (3,), 161, 2e-5),
        "grad_revolute": ("state", (3,), 162, 1e-5),
        "grad_prismatic": ("state", (3,), 163, 2e-6),
        "grad_free_with_revolute": ("links", (2, 3), 164, 5e-5),
        "grad_revolute_chain3": ("links", (3, 3), 165, 1e-4),
        "grad_spherical": ("state", (3,), 166, 1e-5),
        "grad_cartpole": ("links", (2, 3), 167, 2e-5),
        "grad_hopper": ("links", (5, 3), 168, 2e-4),
    }[model_name]
    pair = make_diff_scene_pair(request.getfixturevalue(model_name), n_envs=0, substeps=4, show_viewer=show_viewer)
    n_dofs = pair.entity_ana.n_dofs
    target = gs.tensor(np.random.RandomState(seed).standard_normal((1, *output_shape)), dtype=gs.tc_float).reshape(-1)
    inputs = [np.random.default_rng(seed * 100 + t).standard_normal((n_dofs,)) for t in range(10)]

    def loss_fn(scene, entity):
        pose = entity.get_state().pos if output == "state" else scene.rigid_solver.get_state().links_pos
        return ((pose.reshape(-1) - target) ** 2).sum()

    # fp32 needs a large step to clear the state-noise floor; fp64 needs a small step to bound truncation error.
    assert_grad_matches_fd(
        pair,
        inputs,
        lambda e, x: e.control_dofs_force(x),
        loss_fn,
        rtol=5e-9 if precision == "64" else fp32_tol,
        atol=5e-9 if precision == "64" else fp32_tol,
        eps=3e-5 if precision == "64" else 3e-2,
    )
