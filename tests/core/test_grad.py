import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.misc import qd_to_numpy

from ..utils import assert_allclose


@pytest.mark.slow  # ~350s
@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_differentiable_push(show_viewer):
    HORIZON = 10

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
            substeps=10,
            requires_grad=True,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(0.0, -1.0, 0.0),
            upper_bound=(1.0, 1.0, 0.55),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, -0.15, 2.42),
            camera_lookat=(0.5, 0.5, 0.1),
        ),
        show_viewer=show_viewer,
    )

    plane = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/plane/plane.urdf",
            fixed=True,
        )
    )
    stick = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/stirrer.obj",
            scale=0.6,
            pos=(0.5, 0.5, 0.05),
            euler=(90.0, 0.0, 0.0),
        ),
        material=gs.materials.Tool(
            friction=8.0,
        ),
    )
    obj = scene.add_entity(
        morph=gs.morphs.Box(
            lower=(0.2, 0.1, 0.05),
            upper=(0.4, 0.3, 0.15),
        ),
        material=gs.materials.MPM.Elastic(
            rho=500,
        ),
    )
    scene.build(n_envs=2)

    init_pos = gs.tensor([[0.3, 0.1, 0.28], [0.3, 0.1, 0.5]], requires_grad=True)
    stick.set_position(init_pos)
    pos_obj_init = gs.tensor([0.3, 0.3, 0.1], requires_grad=True)
    obj.set_position(pos_obj_init)
    v_obj_init = gs.tensor([0.0, -1.0, 0.0], requires_grad=True)
    obj.set_velocity(v_obj_init)
    goal = gs.tensor([0.5, 0.8, 0.05])

    loss = 0.0
    v_list = []
    for i in range(HORIZON):
        v_i = gs.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], requires_grad=True)
        stick.set_velocity(vel=v_i)
        v_list.append(v_i)

        scene.step()

        if i == HORIZON // 2:
            mpm_particles = scene.get_state().solvers_state[scene.solvers.index(scene.mpm_solver)]
            loss += torch.pow(mpm_particles.pos[mpm_particles.active == 1] - goal, 2).sum()

        if i == HORIZON - 2:
            state = obj.get_state()
            loss += torch.pow(state.pos - goal, 2).sum()
    loss.backward()

    # TODO: It would be great to compare the gradient to its analytical or numerical value.
    for v_i in v_list[:-1]:
        assert (v_i.grad.abs() > gs.EPS).any()
    assert (v_list[-1].grad.abs() < gs.EPS).all()


@pytest.mark.required
def test_diff_convex_contact_forward(show_viewer):
    # The split narrowphase (GPU-only) used to skip GJK entirely when requires_grad is True, so convex-convex
    # contacts were never detected and bodies fell through each other. Differentiable contact detection must
    # route through the monolithic diff_gjk path, which produces the same forward contacts as the non-grad path.
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            requires_grad=True,
        ),
        rigid_options=gs.options.RigidOptions(
            integrator=gs.integrator.approximate_implicitfast,
            # Keep both boxes on the general convex-convex GJK path rather than the specialized box-box detector.
            box_box_detection=False,
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(
        gs.morphs.Plane(),
    )
    # Two independent stacks whose x-order (-0.8 < +0.8) is the reverse of their geom-pair detection order (the +0.8
    # stack is added first). On GPU the x-position spatial sort would permute contact_sort_idx into a non-identity
    # order; combined with the autodiff backward writing gradients by physical index, that attaches gradients to the
    # wrong contacts. The sort must therefore be disabled in autodiff mode, leaving contact_sort_idx the identity.
    tops = []
    for x in (0.8, -0.8):
        scene.add_entity(
            gs.morphs.Box(
                size=(0.6, 0.6, 0.4),
                pos=(x, 0.0, 0.2),
                fixed=True,
            ),
        )
        tops.append(
            scene.add_entity(
                gs.morphs.Box(
                    size=(0.4, 0.4, 0.4),
                    pos=(x, 0.0, 0.6),
                ),
            )
        )

    scene.build()

    for _ in range(20):
        scene.step()

    # Each top box rests on its fixed box (top face at z=0.4, half-height 0.2 -> center at 0.6) and never tunnels
    # through it. Without contact detection the boxes free-fall to large negative z.
    for top, x in zip(tops, (0.8, -0.8)):
        assert_allclose(top.get_pos(), (x, 0.0, 0.6), atol=2e-4)
        assert_allclose(top.get_dofs_velocity(), 0.0, atol=0.05)

    # In autodiff mode the contact permutation must stay the identity: collider.backward writes upstream gradients
    # back by physical contact index, while get_contacts returns them in contact_sort_idx (logical) order.
    collider = scene.sim.rigid_solver.collider
    assert not collider._collider_static_config.spatial_sort_supported
    n_contacts = int(np.atleast_1d(qd_to_numpy(collider._collider_state.n_contacts))[0])
    sort_idx = qd_to_numpy(collider._collider_state.contact_sort_idx)[:n_contacts, 0]
    assert_allclose(sort_idx, np.arange(n_contacts), atol=0)


@pytest.mark.required
def test_diff_smooth_pair_raises():
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            requires_grad=True,
        ),
        show_viewer=False,
    )
    scene.add_entity(
        gs.morphs.Sphere(
            radius=0.2,
            pos=(0.0, 0.0, 0.2),
            fixed=True,
        ),
    )
    scene.add_entity(
        gs.morphs.Sphere(
            radius=0.2,
            pos=(0.0, 0.0, 0.5),
        ),
    )

    # A sphere/ellipsoid pair has an everywhere-curved Minkowski boundary on which diff_gjk's EPA never converges,
    # so it would silently tunnel.
    with pytest.raises(gs.GenesisException):
        scene.build()


# We need to use 64-bit precision for this test because we need to use sufficiently small perturbation to get reliable
# gradient estimates through finite difference method. This small perturbation is not supported by 32-bit precision in
# stable way.
@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_diff_sim_vs_solver_state_grad_parity(show_viewer):
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
    for use_sim_state in (False, True):
        scene.reset()

        robot.set_dofs_velocity(ctrl)
        scene.step()

        if use_sim_state:
            solver_state = scene.get_state().solvers_state[scene.solvers.index(scene.rigid_solver)]
            chassis_pos = solver_state.links_pos[:, 0].squeeze()
        else:
            chassis_pos = robot.get_state().pos.squeeze()

        loss = torch.linalg.norm(chassis_pos)
        loss.backward()
        grad = ctrl.grad.detach().clone()
        ctrl.grad.zero_()

        # Basic sanity check
        assert (grad[..., :3].abs() > gs.EPS).all()
        assert (grad[..., 3:].abs() < gs.EPS).all()

        grads.append(grad)

    assert_allclose(*grads, atol=gs.EPS)
