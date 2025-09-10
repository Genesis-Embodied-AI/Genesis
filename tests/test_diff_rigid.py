import numpy as np
import pytest
import torch

import genesis as gs
from scipy.spatial.transform import Rotation as R


@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu])
def test_diff_contact(backend):
    torch.manual_seed(0)
    rtol = 0.1

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            # Turn on differentiable mode
            requires_grad=True,
        ),
        rigid_options=gs.options.RigidOptions(
            box_box_detection=False,
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
    x_ang = 0.03
    y_ang = 0.03
    box1.set_quat(R.from_euler("xy", [np.deg2rad(x_ang), np.deg2rad(y_ang)]).as_quat(scalar_first=True))

    box0_init_pos = box0.get_pos()
    box1_init_pos = box1.get_pos()
    box0_init_quat = box0.get_quat()
    box1_init_quat = box1.get_quat()

    init_pos = torch.stack([box0_init_pos, box1_init_pos], dim=0)
    init_quat = torch.stack([box0_init_quat, box1_init_quat], dim=0)

    ### Compute the initial loss and compute gradients using differentiable contact detection
    # Detect contact
    collider.detection()

    # Get contact outputs and their grads
    contacts = collider.get_contacts(as_tensor=True, to_torch=True, keep_batch_dim=True)
    normal = contacts["normal"].clone().requires_grad_()
    position = contacts["position"].clone().requires_grad_()
    penetration = contacts["penetration"].clone().requires_grad_()
    num_init_contacts = len(normal[0])

    loss = ((normal * position).sum(dim=-1) * penetration).sum()
    dL_dnormal = torch.autograd.grad(loss, normal, retain_graph=True)[0]
    dL_dposition = torch.autograd.grad(loss, position, retain_graph=True)[0]
    dL_dpenetration = torch.autograd.grad(loss, penetration)[0]

    # Compute analytical gradients of the geoms position and quaternion
    collider.backward(dL_dposition, dL_dnormal, dL_dpenetration)
    dL_dpos = torch.from_numpy(solver.geoms_state.pos.grad.to_numpy())
    dL_dquat = torch.from_numpy(solver.geoms_state.quat.grad.to_numpy())

    def compute_loss(pos, quat):
        box0_pos = pos[0]
        box1_pos = pos[1]
        box0_quat = quat[0]
        box1_quat = quat[1]

        box0.set_pos(box0_pos)
        box1.set_pos(box1_pos)
        box0.set_quat(box0_quat)
        box1.set_quat(box1_quat)

        # Re-detect contact, we need to manually reset the contact counter as we are not running the whole sim step
        collider._collider_state.n_contacts.fill(0)
        collider._collider_state.n_diff_contacts.fill(0)
        collider.detection()
        contacts = collider.get_contacts(as_tensor=True, to_torch=True, keep_batch_dim=True)

        normal = contacts["normal"]
        position = contacts["position"]
        penetration = contacts["penetration"]
        num_contacts = len(normal[0])

        if num_contacts != num_init_contacts:
            raise ValueError(f"Number of contacts changed from {num_init_contacts} to {num_contacts}")

        loss = ((normal * position).sum(dim=-1) * penetration).sum()

        return loss

    ### Compute directional derivatives along random directions
    FD_EPS = 1e-5
    TRIALS = 100

    # dL_dpos
    dL_dpos_error = 0
    valid_trials = 0
    for trial in range(TRIALS):
        rand_dpos = torch.randn_like(dL_dpos)
        rand_dpos = torch.nn.functional.normalize(rand_dpos, dim=-1)

        dL = (rand_dpos * dL_dpos).sum()
        try:
            # 1 * eps
            input_pos = init_pos + rand_dpos.squeeze(1) * FD_EPS
            lossP1 = compute_loss(input_pos, init_quat)

            # -1 * eps
            input_pos = init_pos - rand_dpos.squeeze(1) * FD_EPS
            lossP2 = compute_loss(input_pos, init_quat)
        except ValueError as e:
            continue
        valid_trials += 1
        dL_fd = (lossP1 - lossP2) / (2 * FD_EPS)

        dL_error = (dL - dL_fd).abs() / max(dL.abs(), dL_fd.abs(), gs.EPS)
        dL_dpos_error += dL_error

    dL_dpos_error /= valid_trials
    assert dL_dpos_error < rtol, f"Relative error is too large: {dL_dpos_error * 100.0:.4g}% < {rtol * 100.0:.4g}%"

    # dL_dquat
    dL_dquat_error = 0
    valid_trials = 0
    for trial in range(TRIALS):
        rand_dquat = torch.randn_like(dL_dquat)
        rand_dquat = torch.nn.functional.normalize(rand_dquat, dim=-1)

        dL = (rand_dquat * dL_dquat).sum()
        try:
            # 1 * eps
            input_quat = init_quat + rand_dquat.squeeze(1) * FD_EPS
            input_quat = torch.nn.functional.normalize(input_quat, dim=-1)
            lossP1 = compute_loss(init_pos, input_quat)

            # -1 * eps
            input_quat = init_quat - rand_dquat.squeeze(1) * FD_EPS
            input_quat = torch.nn.functional.normalize(input_quat, dim=-1)
            lossP2 = compute_loss(init_pos, input_quat)
        except ValueError as e:
            continue
        valid_trials += 1
        dL_fd = (lossP1 - lossP2) / (2 * FD_EPS)
        dL_error = (dL - dL_fd).abs() / max(dL.abs(), dL_fd.abs(), gs.EPS)
        dL_dquat_error += dL_error
    dL_dquat_error /= valid_trials
    assert dL_dquat_error < rtol, f"Relative error is too large: {dL_dquat_error * 100.0:.4g}% < {rtol * 100.0:.4g}%"
