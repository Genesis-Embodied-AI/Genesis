# FD-vs-analytical reverse-mode gradient checks through the rigid constraint solver: joint limits, dof frictionloss,
# and the three equality types (joint / connect / weld), plus an all-groups integration scene. All fp64: the FD
# probes use eps down to 1e-5, which is below the fp32 noise floor.
import math

import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import qd_to_torch

from ..utils import assert_grad_matches_fd, make_diff_scene_pair, rigid_solver_state


pytestmark = [
    pytest.mark.debug(False),
]


@pytest.mark.required
@pytest.mark.precision("64")
def test_rigid_joint_limit_grad_matches_fd(grad_slider_limit):
    # Forward: the slider limit must actually bound the cart (it drifts freely when the constraint is off).
    off = make_diff_scene_pair(
        grad_slider_limit,
        substeps=4,
        dt=1.0 / 60.0,
        gravity=(0.0, 0.0, 0.0),
        enable_joint_limit=False,
        disable_constraint=True,
    )
    off.scene_fd.reset()
    off.entity_fd.set_dofs_velocity(gs.tensor([100.0], dtype=gs.tc_float))
    for _ in range(60):
        off.scene_fd.step()
    assert abs(float(rigid_solver_state(off.scene_fd).qpos[0, 0])) > 50.0

    on = make_diff_scene_pair(
        grad_slider_limit,
        substeps=4,
        dt=1.0 / 60.0,
        gravity=(0.0, 0.0, 0.0),
        enable_joint_limit=True,
        disable_constraint=False,
    )
    on.scene_fd.reset()
    on.entity_fd.set_dofs_velocity(gs.tensor([100.0], dtype=gs.tc_float))
    for _ in range(60):
        on.scene_fd.step()
    assert abs(float(rigid_solver_state(on.scene_fd).qpos[0, 0])) <= 4.5

    # Backward: a rollout that drives the cart into the active |x|=4 limit, so the gradient flows through the
    # constraint correction. Sanity-check that the cart actually reaches the band first.
    on.scene_fd.reset()
    on.entity_fd.set_dofs_velocity(gs.tensor([100.0], dtype=gs.tc_float))
    for _ in range(5):
        on.scene_fd.step()
    assert abs(float(rigid_solver_state(on.scene_fd).qpos[0, 0])) > 3.5

    assert_grad_matches_fd(
        on,
        [np.array([100.0])],
        lambda e, x: e.set_dofs_velocity(x),
        lambda scene, entity: rigid_solver_state(scene).qpos[0, 0] ** 2,
        n_steps=5,
        rtol=1e-3,
        atol=1e-6,
        eps=1e-4,
    )


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("model_name", ["grad_slider_limit", "grad_cartpole", "grad_hopper"])
def test_rigid_per_step_force_grad_matches_fd(model_name, request):
    # Per-step control-force adjoint driving a joint into its limit, across three topologies. A constant force over
    # the horizon pushes the tracked dof into the active band; the setup-sanity assert guards against a vacuous run.
    mjcf = request.getfixturevalue(model_name)
    setup_fn = None
    # (gravity, n_steps, per-step force, loss reads links_pos, sanity dof, sanity threshold) per topology.
    if model_name == "grad_slider_limit":
        gravity, n_steps, per_step_force, loss_links, sanity_dof, sanity_thresh = (
            (0.0, 0.0, 0.0),
            10,
            [500.0],
            False,
            0,
            3.5,
        )
    elif model_name == "grad_cartpole":
        gravity, n_steps, per_step_force, loss_links, sanity_dof, sanity_thresh = (
            (0.0, 0.0, -9.81),
            15,
            [2000.0, 0.0],
            False,
            0,
            3.5,
        )

        def setup_fn(scene, entity):
            entity.set_dofs_position(gs.tensor([0.0, -math.pi], dtype=gs.tc_float))
    else:
        base_force = np.zeros(6)
        base_force[5] = 200.0
        gravity, n_steps, per_step_force, loss_links, sanity_dof, sanity_thresh = (
            (0.0, 0.0, 0.0),
            10,
            base_force,
            True,
            5,
            0.7,
        )

    pair = make_diff_scene_pair(
        mjcf, substeps=4, dt=1.0 / 60.0, gravity=gravity, enable_joint_limit=True, disable_constraint=False
    )
    forces = [np.array(per_step_force) for _ in range(n_steps)]

    def loss_fn(scene, entity):
        state = rigid_solver_state(scene)
        return (state.links_pos.reshape(-1) ** 2).sum() if loss_links else state.qpos[0, 0] ** 2

    pair.scene_fd.reset()
    if setup_fn is not None:
        setup_fn(pair.scene_fd, pair.entity_fd)
    for force in forces:
        pair.entity_fd.control_dofs_force(gs.tensor(force, dtype=gs.tc_float))
        pair.scene_fd.step()
    reached = abs(float(rigid_solver_state(pair.scene_fd).qpos[0, sanity_dof]))
    assert reached > sanity_thresh, f"setup error: {model_name} did not reach its limit band (q={reached})"

    assert_grad_matches_fd(
        pair,
        forces,
        lambda e, x: e.control_dofs_force(x),
        loss_fn,
        setup_fn=setup_fn,
        rtol=1e-3,
        atol=1e-4,
        eps=1e-2,
    )


@pytest.mark.required
@pytest.mark.precision("64")
def test_rigid_frictionloss_grad_matches_fd(grad_revolute_frictionloss):
    pair = make_diff_scene_pair(
        grad_revolute_frictionloss,
        substeps=4,
        dt=1.0 / 60.0,
        gravity=(0.0, 0.0, 0.0),
        enable_joint_limit=False,
        disable_constraint=False,
    )
    pair.scene_fd.reset()
    pair.scene_fd.step()
    cs = pair.scene_fd.rigid_solver.constraint_solver.constraint_state
    assert int(qd_to_torch(cs.n_constraints_frictionloss)[0]) == 1

    assert_grad_matches_fd(
        pair,
        [np.array([2.0])],
        lambda e, x: e.set_dofs_velocity(x),
        lambda scene, entity: rigid_solver_state(scene).qpos[0, 0] ** 2,
        n_steps=10,
        rtol=2e-3,
        atol=1e-6,
        eps=1e-5,
    )


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize(
    "model_name, n_rows",
    [
        ("grad_hinge_pair_joint_eq_quadratic", 1),
        ("grad_connect_loop", 3),
        ("grad_weld_pair", 6),
    ],
)
def test_rigid_equality_grad_matches_fd(model_name, n_rows, request):
    pair = make_diff_scene_pair(
        request.getfixturevalue(model_name),
        substeps=4,
        dt=1.0 / 60.0,
        gravity=(0.0, 0.0, 0.0),
        enable_joint_limit=False,
        disable_constraint=False,
    )
    pair.scene_fd.reset()
    pair.scene_fd.step()
    cs = pair.scene_fd.rigid_solver.constraint_solver.constraint_state
    assert int(qd_to_torch(cs.n_constraints_equality)[0]) == n_rows

    assert_grad_matches_fd(
        pair,
        [np.array([0.8, -0.3])],
        lambda e, x: e.set_dofs_velocity(x),
        lambda scene, entity: (
            rigid_solver_state(scene).qpos[0, 0] ** 2 + 0.7 * rigid_solver_state(scene).qpos[0, 1] ** 2
        ),
        n_steps=10,
        rtol=2e-3,
        atol=1e-6,
        eps=1e-5,
    )


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_rigid_all_constraints_grad_matches_fd(grad_all_eq_fric):
    # Integration scene: frictionloss + equality joint + connect + weld on disjoint link pairs. Guards row-offset
    # bookkeeping across every differentiated constraint group at once; per-group formulas are pinned elsewhere.
    pair = make_diff_scene_pair(
        grad_all_eq_fric,
        substeps=4,
        dt=1.0 / 60.0,
        gravity=(0.0, 0.0, 0.0),
        enable_joint_limit=False,
        disable_constraint=False,
    )
    pair.scene_fd.reset()
    pair.scene_fd.step()
    cs = pair.scene_fd.rigid_solver.constraint_solver.constraint_state
    assert int(qd_to_torch(cs.n_constraints_equality)[0]) == 10
    assert int(qd_to_torch(cs.n_constraints_frictionloss)[0]) == 1

    weights = np.array([1.0, 0.7, 1.3, 0.5, 0.9, 1.1])

    def loss_fn(scene, entity):
        qpos = rigid_solver_state(scene).qpos[0]
        return sum(weights[d] * qpos[d] ** 2 for d in range(6))

    assert_grad_matches_fd(
        pair,
        [np.array([0.8, -0.3, 0.5, -0.2, 0.4, -0.6])],
        lambda e, x: e.set_dofs_velocity(x),
        loss_fn,
        n_steps=10,
        rtol=2e-2,
        atol=1e-6,
        eps=1e-5,
    )
