import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.geom import R_to_quat
from genesis.utils.misc import ti_to_torch


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu])
def test_diff_contact(backend):
    torch.manual_seed(0)
    rtol = 1e-4

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
        gs.morphs.Box(size=box_size * vec_one, pos=box_pos_offset, fixed=True),
    )
    box1 = scene.add_entity(
        gs.morphs.Box(
            size=box_size * vec_one, pos=box_pos_offset + 0.8 * box_spacing * np.array([0, 0, 1]), fixed=True
        ),
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

    def compute_loss(box0_pos, box1_pos, box0_quat, box1_quat):
        box0.set_pos(box0_pos)
        box1.set_pos(box1_pos)
        box0.set_quat(box0_quat)
        box1.set_quat(box1_quat)

        # Re-detect contact, we need to manually reset the contact counter as we are not running the whole sim step
        collider._collider_state.n_contacts.fill(0)
        collider.detection()
        contacts = collider.get_contacts(as_tensor=True, to_torch=True, keep_batch_dim=True)
        normal, position, penetration = contacts["normal"], contacts["position"], contacts["penetration"]

        loss = ((normal * position).sum(dim=-1) * penetration).sum()
        return loss

    def compute_dL_error(dL_dx, x_type):
        dL_error = 0.0

        box0_input_pos = box0_init_pos
        box1_input_pos = box1_init_pos
        box0_input_quat = box0_init_quat
        box1_input_quat = box1_init_quat

        for trial in range(TRIALS):
            rand_dx = torch.randn_like(dL_dx)
            rand_dx = torch.nn.functional.normalize(rand_dx, dim=-1)

            dL = (rand_dx * dL_dx).sum()

            # 1 * eps
            if x_type == "pos":
                box0_input_pos = box0_init_pos + rand_dx[0, 0] * FD_EPS
                box1_input_pos = box1_init_pos + rand_dx[1, 0] * FD_EPS
            else:
                box0_input_quat = box0_init_quat + rand_dx[0, 0] * FD_EPS
                box1_input_quat = box1_init_quat + rand_dx[1, 0] * FD_EPS
            lossP1 = compute_loss(box0_input_pos, box1_input_pos, box0_input_quat, box1_input_quat)

            # -1 * eps
            if x_type == "pos":
                box0_input_pos = box0_init_pos - rand_dx[0, 0] * FD_EPS
                box1_input_pos = box1_init_pos - rand_dx[1, 0] * FD_EPS
            else:
                box0_input_quat = box0_init_quat - rand_dx[0, 0] * FD_EPS
                box1_input_quat = box1_init_quat - rand_dx[1, 0] * FD_EPS
            lossP2 = compute_loss(box0_input_pos, box1_input_pos, box0_input_quat, box1_input_quat)

            dL_fd = (lossP1 - lossP2) / (2 * FD_EPS)
            dL_error += (dL - dL_fd).abs() / max(dL.abs(), dL_fd.abs(), gs.EPS)

        dL_error /= TRIALS
        return dL_error

    # dL_dpos
    dL_dpos_error = compute_dL_error(dL_dpos, "pos")
    assert dL_dpos_error < rtol, f"Relative error is too large: {dL_dpos_error * 100.0:.4g}% < {rtol * 100.0:.4g}%"

    # dL_dquat
    dL_dquat_error = compute_dL_error(dL_dquat, "quat")
    assert dL_dquat_error < rtol, f"Relative error is too large: {dL_dquat_error * 100.0:.4g}% < {rtol * 100.0:.4g}%"
