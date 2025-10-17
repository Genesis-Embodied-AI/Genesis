import sys

import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.geom import R_to_quat
from genesis.utils.misc import ti_to_torch

from .utils import assert_allclose


pytestmark = [
    pytest.mark.field_only,
]


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_diff_contact(backend):
    if gs.use_ndarray:
        pytest.skip(reason="GsTaichi dynamic array type does not support AutoDiff.")

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

    box0_init_pos = box0.get_pos()
    box1_init_pos = box1.get_pos()
    box0_init_quat = box0.get_quat()
    box1_init_quat = box1.get_quat()

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
    dL_dpos = ti_to_torch(solver.geoms_state.pos.grad)
    dL_dquat = ti_to_torch(solver.geoms_state.quat.grad)

    ### Compute directional derivatives along random directions
    FD_EPS = 1e-5
    TRIALS = 100

    def compute_dL_error(dL_dx, x_type):
        dL_error_rel = 0.0

        box0_input_pos = box0_init_pos
        box1_input_pos = box1_init_pos
        box0_input_quat = box0_init_quat
        box1_input_quat = box1_init_quat

        for trial in range(TRIALS):
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
    assert_allclose(dL_dpos_error_rel, 0.0, atol=RTOL)


@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_differentiable_push(precision, show_viewer):
    # FIXME: Wait for fix to be merged in GsTaichi: https://github.com/Genesis-Embodied-AI/gstaichi/pull/225
    if sys.platform == "darwin" and gs.backend != gs.cpu:
        pytest.skip(reason="GsTaichi does not support AutoDiff on non-CPU backend on Mac OS for now.")
    if sys.platform == "linux" and gs.backend == gs.cpu and precision == "64":
        pytest.skip(reason="GsTaichi segfault when using AutoDiff on CPU backend on Linux for now.")
    if gs.use_ndarray:
        pytest.skip(reason="GsTaichi dynamic array type does not support AutoDiff.")

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
