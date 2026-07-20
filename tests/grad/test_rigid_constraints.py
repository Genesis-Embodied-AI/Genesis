# FD-vs-analytical reverse-mode gradient checks through the rigid constraint solver: joint limits, dof frictionloss,
# and the three equality types (joint / connect / weld), plus an all-groups integration scene.
import math

import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import qd_to_torch, tensor_to_array

from ..utils import assert_allclose
from .utils import assert_grad_matches_fd, make_diff_scene_pair


@pytest.mark.required
@pytest.mark.debug(False)
def test_rigid_joint_limit_grad_matches_fd(grad_slider_limit, precision, show_viewer):
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
    assert abs(off.scene_fd.rigid_solver.get_state().qpos[0, 0]) > 50.0

    on = make_diff_scene_pair(
        grad_slider_limit,
        substeps=4,
        dt=1.0 / 60.0,
        gravity=(0.0, 0.0, 0.0),
        enable_joint_limit=True,
        disable_constraint=False,
        show_viewer=show_viewer,
    )
    on.scene_fd.reset()
    on.entity_fd.set_dofs_velocity(gs.tensor([100.0], dtype=gs.tc_float))
    for _ in range(60):
        on.scene_fd.step()
    assert abs(on.scene_fd.rigid_solver.get_state().qpos[0, 0]) <= 4.5

    # Backward: a rollout that drives the cart into the active |x|=4 limit, so the gradient flows through the
    # constraint correction. Sanity-check that the cart actually reaches the band first.
    on.scene_fd.reset()
    on.entity_fd.set_dofs_velocity(gs.tensor([100.0], dtype=gs.tc_float))
    for _ in range(5):
        on.scene_fd.step()
    assert abs(on.scene_fd.rigid_solver.get_state().qpos[0, 0]) > 3.5

    assert_grad_matches_fd(
        on,
        [np.array([100.0])],
        lambda e, x: e.set_dofs_velocity(x),
        lambda scene, entity: scene.rigid_solver.get_state().qpos[0, 0] ** 2,
        n_steps=5,
        rtol=1e-10 if precision == "64" else 5e-4,
        atol=1e-10 if precision == "64" else 5e-4,
        eps=3e-4 if precision == "64" else 3e-2,
    )

    # Inside-limit single step: the cart stays well inside the range so the limit is present but inactive; the
    # constraint-inclusive forward+backward chain must still satisfy central FD (a smoother path than the crossing).
    assert_grad_matches_fd(
        on,
        [np.array([2.0])],
        lambda e, x: e.set_dofs_velocity(x),
        lambda scene, entity: scene.rigid_solver.get_state().qpos[0, 0] ** 2,
        n_steps=1,
        rtol=1e-10 if precision == "64" else 5e-4,
        atol=1e-10 if precision == "64" else 5e-4,
        eps=3e-4 if precision == "64" else 3e-2,
    )

    # Inactive-path parity: with the limit enabled but never hit, the adjoint must equal the no-limit baseline - the
    # inactive constraint branch must inject no spurious gradient.
    off_solver = make_diff_scene_pair(
        grad_slider_limit,
        substeps=4,
        dt=1.0 / 60.0,
        gravity=(0.0, 0.0, 0.0),
        enable_joint_limit=False,
        disable_constraint=False,
    )
    grads = {}
    for pair, key in ((off_solver, "off"), (on, "on")):
        pair.scene_ana.reset()
        v = gs.tensor([0.5], dtype=gs.tc_float, requires_grad=True)
        pair.entity_ana.set_dofs_velocity(v)
        pair.scene_ana.step()
        loss = pair.scene_ana.rigid_solver.get_state().qpos[0, 0] ** 2
        loss.backward()
        grads[key] = tensor_to_array(v.grad)
    assert_allclose(
        grads["on"],
        grads["off"],
        rtol=1e-6 if precision == "64" else 1e-4,
        atol=1e-9 if precision == "64" else 1e-6,
    )


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["grad_slider_limit", "grad_cartpole", "grad_hopper"])
@pytest.mark.debug(False)
def test_rigid_per_step_force_grad_matches_fd(model_name, request, precision, show_viewer):
    # Per-step control-force adjoint driving a joint into its limit, across three topologies. A constant force over
    # the horizon pushes the tracked dof into the active band; the setup-sanity assert guards against a vacuous run.
    hopper_force = np.zeros(6)
    hopper_force[5] = 200.0
    # (gravity, n_steps, per-step force, loss reads links_pos, sanity dof, sanity threshold, initial dof pose,
    # fp32 tolerance). The fp32 floor depends on the topology: the slider limit is the noisiest, the hopper the
    # cleanest; fp64 clears 5e-5 for all three (the cartpole limit kink sets that floor).
    gravity, n_steps, per_step_force, is_links_loss, sanity_dof, sanity_thresh, init_pos, fp32_tol = {
        "grad_slider_limit": ((0.0, 0.0, 0.0), 10, [500.0], False, 0, 3.5, None, 1e-3),
        "grad_cartpole": ((0.0, 0.0, -9.81), 15, [2000.0, 0.0], False, 0, 3.5, [0.0, -math.pi], 5e-4),
        "grad_hopper": ((0.0, 0.0, 0.0), 10, hopper_force, True, 5, 0.7, None, 2e-4),
    }[model_name]

    pair = make_diff_scene_pair(
        request.getfixturevalue(model_name),
        substeps=4,
        dt=1.0 / 60.0,
        gravity=gravity,
        enable_joint_limit=True,
        disable_constraint=False,
        show_viewer=show_viewer,
    )
    forces = [np.array(per_step_force) for _ in range(n_steps)]

    def setup_fn(scene, entity):
        if init_pos is not None:
            entity.set_dofs_position(gs.tensor(init_pos, dtype=gs.tc_float))

    def loss_fn(scene, entity):
        state = scene.rigid_solver.get_state()
        return (state.links_pos.reshape(-1) ** 2).sum() if is_links_loss else state.qpos[0, 0] ** 2

    pair.scene_fd.reset()
    setup_fn(pair.scene_fd, pair.entity_fd)
    for force in forces:
        pair.entity_fd.control_dofs_force(gs.tensor(force, dtype=gs.tc_float))
        pair.scene_fd.step()
    reached = abs(pair.scene_fd.rigid_solver.get_state().qpos[0, sanity_dof])
    assert reached > sanity_thresh, f"setup error: {model_name} did not reach its limit band (q={reached})"

    assert_grad_matches_fd(
        pair,
        forces,
        lambda e, x: e.control_dofs_force(x),
        loss_fn,
        setup_fn=setup_fn,
        rtol=5e-5 if precision == "64" else fp32_tol,
        atol=5e-5 if precision == "64" else fp32_tol,
        eps=3e-2,
    )


@pytest.mark.required
def test_rigid_frictionloss_grad_matches_fd(grad_revolute_frictionloss, precision, show_viewer):
    pair = make_diff_scene_pair(
        grad_revolute_frictionloss,
        substeps=4,
        dt=1.0 / 60.0,
        gravity=(0.0, 0.0, 0.0),
        enable_joint_limit=False,
        disable_constraint=False,
        show_viewer=show_viewer,
    )
    pair.scene_fd.reset()
    pair.scene_fd.step()
    cs = pair.scene_fd.rigid_solver.constraint_solver.constraint_state
    assert qd_to_torch(cs.n_constraints_frictionloss)[0] == 1

    assert_grad_matches_fd(
        pair,
        [np.array([2.0])],
        lambda e, x: e.set_dofs_velocity(x),
        lambda scene, entity: scene.rigid_solver.get_state().qpos[0, 0] ** 2,
        n_steps=10,
        rtol=1e-10 if precision == "64" else 5e-5,
        atol=1e-10 if precision == "64" else 5e-5,
        eps=3e-6 if precision == "64" else 1e-2,
    )


@pytest.mark.required
@pytest.mark.parametrize(
    "model_name, n_rows",
    [
        ("grad_hinge_pair_joint_eq_linear", 1),
        ("grad_hinge_pair_joint_eq_quadratic", 1),
        ("grad_connect_loop", 3),
        ("grad_weld_pair", 6),
    ],
)
def test_rigid_equality_grad_matches_fd(model_name, n_rows, request, precision, show_viewer):
    pair = make_diff_scene_pair(
        request.getfixturevalue(model_name),
        substeps=4,
        dt=1.0 / 60.0,
        gravity=(0.0, 0.0, 0.0),
        enable_joint_limit=False,
        disable_constraint=False,
        show_viewer=show_viewer,
    )
    pair.scene_fd.reset()
    pair.scene_fd.step()
    cs = pair.scene_fd.rigid_solver.constraint_solver.constraint_state
    assert qd_to_torch(cs.n_constraints_equality)[0] == n_rows

    assert_grad_matches_fd(
        pair,
        [np.array([0.8, -0.3])],
        lambda e, x: e.set_dofs_velocity(x),
        lambda scene, entity: (
            scene.rigid_solver.get_state().qpos[0, 0] ** 2 + 0.7 * scene.rigid_solver.get_state().qpos[0, 1] ** 2
        ),
        n_steps=10,
        rtol=2e-9 if precision == "64" else 5e-5,
        atol=2e-9 if precision == "64" else 5e-5,
        eps=1e-3 if precision == "64" else 3e-2,
    )


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_rigid_all_constraints_grad_matches_fd(grad_all_eq_fric, precision, show_viewer):
    # Integration scene: frictionloss + equality joint + connect + weld on disjoint link pairs. Guards row-offset
    # bookkeeping across every differentiated constraint group at once; per-group formulas are pinned elsewhere.
    pair = make_diff_scene_pair(
        grad_all_eq_fric,
        substeps=4,
        dt=1.0 / 60.0,
        gravity=(0.0, 0.0, 0.0),
        enable_joint_limit=False,
        disable_constraint=False,
        show_viewer=show_viewer,
    )
    pair.scene_fd.reset()
    pair.scene_fd.step()
    cs = pair.scene_fd.rigid_solver.constraint_solver.constraint_state
    assert qd_to_torch(cs.n_constraints_equality)[0] == 10
    assert qd_to_torch(cs.n_constraints_frictionloss)[0] == 1

    weights = np.array([1.0, 0.7, 1.3, 0.5, 0.9, 1.1])

    def loss_fn(scene, entity):
        qpos = scene.rigid_solver.get_state().qpos[0]
        return sum(weights[d] * qpos[d] ** 2 for d in range(6))

    assert_grad_matches_fd(
        pair,
        [np.array([0.8, -0.3, 0.5, -0.2, 0.4, -0.6])],
        lambda e, x: e.set_dofs_velocity(x),
        loss_fn,
        n_steps=10,
        rtol=5e-10 if precision == "64" else 5e-5,
        atol=5e-10 if precision == "64" else 5e-5,
        eps=3e-4 if precision == "64" else 3e-3,
    )
