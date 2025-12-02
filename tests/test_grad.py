import sys

import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.geom import R_to_quat
from genesis.utils.misc import ti_to_torch, ti_to_numpy, tensor_to_array
from genesis.utils import set_random_seed

from .utils import assert_allclose


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_differentiable_push(precision, show_viewer):
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


@pytest.mark.required
@pytest.mark.field_only  # FIXME: Parameter pruning for ndarray is buggy...
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
@pytest.mark.field_only  # FIXME: Parameter pruning for ndarray is buggy...
@pytest.mark.precision("64")
def test_diff_solver(monkeypatch):
    from genesis.engine.solvers.rigid.constraint_solver_decomp import func_init_solver, func_solve
    from genesis.engine.solvers.rigid.rigid_solver_decomp import kernel_step_1

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

    plane = scene.add_entity(gs.morphs.Plane(pos=(0, 0, 0)))
    box = scene.add_entity(gs.morphs.Box(size=(1, 1, 1), pos=(10, 10, 0.49)))
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )

    scene.build()
    rigid_solver = scene._sim.rigid_solver
    constraint_solver = rigid_solver.constraint_solver

    franka.set_qpos([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04])

    # Monkeypatch the constraint resolve function to avoid overwriting the necessary information for computing gradients.
    def constraint_solver_resolve():
        func_init_solver(
            dofs_state=rigid_solver.dofs_state,
            entities_info=rigid_solver.entities_info,
            constraint_state=constraint_solver.constraint_state,
            rigid_global_info=rigid_solver._rigid_global_info,
            static_rigid_sim_config=rigid_solver._static_rigid_sim_config,
        )
        func_solve(
            entities_info=rigid_solver.entities_info,
            dofs_state=rigid_solver.dofs_state,
            constraint_state=constraint_solver.constraint_state,
            rigid_global_info=rigid_solver._rigid_global_info,
            static_rigid_sim_config=rigid_solver._static_rigid_sim_config,
        )

    monkeypatch.setattr(constraint_solver, "resolve", constraint_solver_resolve)

    # Step once to compute constraint solver's inputs: [mass], [jac], [aref], [efc_D], [force]. We do not call the
    # entire scene.step() because it will overwrite the necessary information that we need to compute the gradients.
    kernel_step_1(
        links_state=rigid_solver.links_state,
        links_info=rigid_solver.links_info,
        joints_state=rigid_solver.joints_state,
        joints_info=rigid_solver.joints_info,
        dofs_state=rigid_solver.dofs_state,
        dofs_info=rigid_solver.dofs_info,
        geoms_state=rigid_solver.geoms_state,
        geoms_info=rigid_solver.geoms_info,
        entities_state=rigid_solver.entities_state,
        entities_info=rigid_solver.entities_info,
        rigid_global_info=rigid_solver._rigid_global_info,
        static_rigid_sim_config=rigid_solver._static_rigid_sim_config,
        contact_island_state=constraint_solver.contact_island.contact_island_state,
    )
    constraint_solver.add_equality_constraints()
    rigid_solver.collider.detection()
    constraint_solver.add_inequality_constraints()
    constraint_solver.resolve()

    # Loss function to compute gradients using finite difference method
    def compute_loss(input_mass, input_jac, input_aref, input_efc_D, input_force):
        rigid_solver._rigid_global_info.mass_mat.from_numpy(tensor_to_array(input_mass))
        constraint_solver.constraint_state.jac.from_numpy(tensor_to_array(input_jac))
        constraint_solver.constraint_state.aref.from_numpy(tensor_to_array(input_aref))
        constraint_solver.constraint_state.efc_D.from_numpy(tensor_to_array(input_efc_D))
        rigid_solver.dofs_state.force.from_numpy(tensor_to_array(input_force))

        # Recompute acc_smooth from the updated input variables
        updated_acc_smooth = torch.linalg.solve(input_mass.squeeze(-1), input_force.squeeze(-1))
        input_acc_smooth = tensor_to_array(updated_acc_smooth.unsqueeze(-1))

        rigid_solver.dofs_state.acc_smooth.from_numpy(input_acc_smooth)

        constraint_solver.resolve()
        output_qacc = ti_to_numpy(constraint_solver.qacc)
        th_output_qacc = torch.from_numpy(output_qacc).to(device=gs.device)
        loss = ((th_output_qacc - target_qacc) ** 2).mean()
        return loss

    init_input_mass = ti_to_torch(rigid_solver._rigid_global_info.mass_mat)
    init_input_jac = ti_to_torch(constraint_solver.constraint_state.jac)
    init_input_aref = ti_to_torch(constraint_solver.constraint_state.aref)
    init_input_efc_D = ti_to_torch(constraint_solver.constraint_state.efc_D)
    init_input_force = ti_to_torch(rigid_solver.dofs_state.force)

    # Initial output of the constraint solver
    set_random_seed(0)
    init_output_qacc = ti_to_torch(constraint_solver.qacc)
    target_qacc = np.random.randn(*init_output_qacc.shape)
    target_qacc = torch.from_numpy(target_qacc).to(device=gs.device) * init_output_qacc.abs().mean()

    # Solve the constraint solver and get the output
    output_qacc = ti_to_numpy(constraint_solver.qacc)
    th_output_qacc = torch.from_numpy(output_qacc).to(device=gs.device).requires_grad_(True)

    # Compute loss and gradient of the output
    loss = ((th_output_qacc - target_qacc) ** 2).mean()
    dL_dqacc = tensor_to_array(torch.autograd.grad(loss, th_output_qacc)[0])

    # Compute gradients of the input variables: [mass], [jac], [aref], [efc_D], [force]
    constraint_solver.backward(dL_dqacc)

    # Fetch gradients of the input variables
    dL_dM = ti_to_torch(constraint_solver.constraint_state.dL_dM)
    dL_djac = ti_to_torch(constraint_solver.constraint_state.dL_djac)
    dL_daref = ti_to_torch(constraint_solver.constraint_state.dL_daref)
    dL_defc_D = ti_to_torch(constraint_solver.constraint_state.dL_defc_D)
    dL_dforce = ti_to_torch(constraint_solver.constraint_state.dL_dforce)

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
            rand_dx = torch.from_numpy(rand_dx).to(device=gs.device)
            rand_dx = rand_dx / max(
                torch.linalg.norm(rand_dx, dim=0 if x_type in ("force", "aref", "efc_D") else (0, 1)), gs.EPS
            )
            if x_type == "mass":
                # Make rand_dx symmetric
                rand_dx = (rand_dx + rand_dx.transpose(0, 1)) * 0.5

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

            dL_error += (dL - dL_fd).abs() / max(dL.abs(), dL_fd.abs(), gs.EPS)

        dL_error /= TRIALS
        assert_allclose(dL_error, 0.0, atol=RTOL)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_differentiable_rigid(show_viewer):
    dt = 1e-2
    horizon = 100
    substeps = 1
    goal_pos = gs.tensor([0.7, 1.0, 0.05])
    goal_quat = gs.tensor([0.3, 0.2, 0.1, 0.9])
    goal_quat = goal_quat / torch.norm(goal_quat, dim=-1, keepdim=True)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=substeps, requires_grad=True, gravity=(0, 0, -1)),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            enable_self_collision=False,
            enable_joint_limit=False,
            disable_constraint=True,
            use_contact_island=False,
            use_hibernation=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, -0.15, 2.42),
            camera_lookat=(0.5, 0.5, 0.1),
        ),
        show_viewer=show_viewer,
    )

    box = scene.add_entity(
        gs.morphs.Box(
            pos=(0, 0, 0),
            size=(0.1, 0.1, 0.2),
        ),
        surface=gs.surfaces.Default(
            color=(0.9, 0.0, 0.0, 1.0),
        ),
    )
    if show_viewer:
        target = scene.add_entity(
            gs.morphs.Box(
                pos=goal_pos,
                quat=goal_quat,
                size=(0.1, 0.1, 0.2),
            ),
            surface=gs.surfaces.Default(
                color=(0.0, 0.9, 0.0, 0.5),
            ),
        )

    scene.build()

    num_iter = 200
    lr = 1e-2

    init_pos = gs.tensor([0.3, 0.1, 0.28], requires_grad=True)
    init_quat = gs.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=True)
    optimizer = torch.optim.Adam([init_pos, init_quat], lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iter, eta_min=1e-3)

    for _ in range(num_iter):
        scene.reset()

        box.set_pos(init_pos)
        box.set_quat(init_quat)

        loss = 0
        for _ in range(horizon):
            scene.step()
            if show_viewer:
                target.set_pos(goal_pos)
                target.set_quat(goal_quat)

        box_state = box.get_state()
        box_pos = box_state.pos
        box_quat = box_state.quat
        loss = torch.abs(box_pos - goal_pos).sum() + torch.abs(box_quat - goal_quat).sum()

        optimizer.zero_grad()
        loss.backward()  # this lets gradient flow all the way back to tensor input
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            init_quat.data = init_quat / torch.norm(init_quat, dim=-1, keepdim=True)

    assert_allclose(loss, 0.0, atol=1e-2)
