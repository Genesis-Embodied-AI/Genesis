# Finite-difference vs analytical reverse-mode gradient checks for rigid forward kinematics (constraints off), one
# packed test per joint topology exercising every tracked setter, and a multi-step control-force adjoint check.
import numpy as np
import pytest
import torch

import genesis as gs

from ..utils import assert_grad_matches_fd, make_diff_scene_pair, rigid_solver_state


pytestmark = [
    pytest.mark.debug(False),
]


@pytest.mark.required
@pytest.mark.precision("32")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
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
def test_rigid_fk_grad_matches_fd(model_name, request):
    pair = make_diff_scene_pair(request.getfixturevalue(model_name), n_envs=2, substeps=4)
    n_dofs = pair.entity_ana.n_dofs
    n_links = pair.entity_ana.n_links
    B = 2

    # Single-link joints read the entity pose; multi-link topologies read the rigid-solver per-link pose.
    single_link = model_name in ("grad_free", "grad_revolute", "grad_prismatic", "grad_spherical")
    # (setter, output, tol_kind, input_seed) per joint: the fp32-stable subset of the original per-joint FD checks.
    # force->position checks that sit at the fp32 FD floor are omitted; the cartpole force->links_pos check is kept
    # because it clears the band. (checks, pos_target_seed, quat_target_seed).
    checks_by_joint = {
        "grad_free": (
            (
                ("pos", "pos", "default", 10),
                ("quat", "quat", "quat", 11),
                ("vel", "pos", "default", 12),
                ("vel", "quat", "quat", 13),
            ),
            1,
            2,
        ),
        "grad_revolute": (
            (("vel", "pos", "default", 30), ("vel", "quat", "quat", 31), ("force", "quat", "quat", 32)),
            21,
            22,
        ),
        "grad_prismatic": ((("vel", "pos", "default", 50),), 41, 0),
        "grad_spherical": (
            (("vel", "pos", "default", 70), ("vel", "quat", "quat", 71), ("force", "quat", "quat", 72)),
            61,
            62,
        ),
        "grad_free_with_revolute": (
            (
                ("pos", "pos", "default", 70),
                ("quat", "quat", "quat", 71),
                ("vel", "pos", "default", 72),
                ("vel", "quat", "quat", 73),
            ),
            61,
            62,
        ),
        "grad_revolute_chain3": ((("vel", "pos", "default", 90), ("vel", "quat", "quat", 91)), 81, 82),
        "grad_cartpole": (
            (("vel", "pos", "default", 190), ("vel", "quat", "quat", 191), ("force", "pos", "default", 192)),
            181,
            182,
        ),
        "grad_hopper": ((("vel", "pos", "default", 210), ("vel", "quat", "quat", 211)), 201, 202),
    }
    checks, pos_seed, quat_seed = checks_by_joint[model_name]

    pos_shape = (B, 3) if single_link else (B, n_links, 3)
    quat_shape = (B, 4) if single_link else (B, n_links, 4)
    tgt_pos = (
        torch.from_numpy(np.random.default_rng(pos_seed).standard_normal(pos_shape))
        .to(dtype=gs.tc_float, device=gs.device)
        .reshape(-1)
    )
    tgt_quat = (
        torch.from_numpy(np.random.default_rng(quat_seed).standard_normal(quat_shape))
        .to(dtype=gs.tc_float, device=gs.device)
        .reshape(-1)
    )
    # Hopper is the largest chain; at fp32 its small-sensitivity links entries quantize to the FD step, so widen atol.
    atol_override = 8e-3 if model_name == "grad_hopper" else None

    for setter, output, tol_kind, input_seed in checks:
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

        def loss_fn(scene, entity, tgt=target, out=output, sl=single_link):
            if sl:
                pose = entity.get_state().pos if out == "pos" else entity.get_state().quat
            else:
                state = rigid_solver_state(scene)
                pose = state.links_pos if out == "pos" else state.links_quat
            return ((pose.reshape(-1) - tgt) ** 2).sum()

        if tol_kind == "default":
            tol = dict(rtol=2e-2, atol=2e-3, eps=1e-3)
        else:
            tol = dict(rtol=5e-2, atol=5e-3, eps=1e-3)
        if atol_override is not None:
            tol["atol"] = atol_override

        assert_grad_matches_fd(pair, [step_input], apply_fn, loss_fn, **tol)


@pytest.mark.required
@pytest.mark.precision("64")
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
def test_rigid_fk_multistep_force_grad_matches_fd(model_name, request):
    # Ten distinct per-step control forces, each of which must receive an independent adjoint across the unroll.
    # fp64 is required: the force->position FD sensitivity sits at the fp32 precision floor after a single step.
    # Output kind (entity state vs rigid-solver links), per-link output shape, and target seed per joint topology.
    output, output_shape, seed = {
        "grad_free": ("state", (3,), 161),
        "grad_revolute": ("state", (3,), 162),
        "grad_prismatic": ("state", (3,), 163),
        "grad_free_with_revolute": ("links", (2, 3), 164),
        "grad_revolute_chain3": ("links", (3, 3), 165),
        "grad_spherical": ("state", (3,), 166),
        "grad_cartpole": ("links", (2, 3), 167),
        "grad_hopper": ("links", (5, 3), 168),
    }[model_name]
    pair = make_diff_scene_pair(request.getfixturevalue(model_name), n_envs=0, substeps=4)
    n_dofs = pair.entity_ana.n_dofs
    target = (
        torch.from_numpy(np.random.default_rng(seed).standard_normal((1, *output_shape)))
        .to(dtype=gs.tc_float, device=gs.device)
        .reshape(-1)
    )
    inputs = [np.random.default_rng(seed * 100 + t).standard_normal((n_dofs,)) for t in range(10)]

    def loss_fn(scene, entity):
        pose = entity.get_state().pos if output == "state" else rigid_solver_state(scene).links_pos
        return ((pose.reshape(-1) - target) ** 2).sum()

    assert_grad_matches_fd(pair, inputs, lambda e, x: e.control_dofs_force(x), loss_fn, rtol=1e-4, atol=1e-6, eps=1e-5)
