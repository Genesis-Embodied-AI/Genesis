import sys

import pytest
import torch

import genesis as gs


pytestmark = [
    pytest.mark.field_only,
]


@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_differentiable_push(precision, show_viewer):
    # FIXME: Wait for fix to be merged in GsTaichi: https://github.com/Genesis-Embodied-AI/gstaichi/pull/225
    if sys.platform == "darwin" and gs.backend != gs.cpu:
        pytest.skip(reason="GsTaichi does not support AutoDiff on non-CPU backend on Mac OS for now.")
    if sys.platform == "linux" and gs.backend == gs.cpu and precision == "64":
        pytest.skip(reason="GsTaichi segfault when using AutoDiff on CPU backend on Linux for now.")

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
            mpm_particles = scene.get_state().solvers_state[3]
            loss += torch.pow(mpm_particles.pos[mpm_particles.active == 1] - goal, 2).sum()

        if i == HORIZON - 2:
            state = obj.get_state()
            loss += torch.pow(state.pos - goal, 2).sum()
    loss.backward()

    # TODO: It would be great to compare the gradient to its analytical or numerical value.
    for v_i in v_list[:-1]:
        assert (v_i.grad.abs() > gs.EPS).any()
    assert (v_list[-1].grad.abs() < gs.EPS).all()
