import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.misc import ti_to_torch
from genesis.engine.solvers.rigid.constraint_solver_decomp import func_init_solver, func_solve
from genesis.engine.solvers.rigid.rigid_solver_decomp import kernel_step_1


@pytest.mark.required
@pytest.mark.field_only
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_diff_contact(backend):
    torch.manual_seed(0)
    rtol = 1e-4

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
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

    qpos = np.array([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04])
    franka.set_qpos(qpos)

    def constraint_solver_resolve():
        func_init_solver(
            dofs_state=rigid_solver.dofs_state,
            entities_info=rigid_solver.entities_info,
            constraint_state=constraint_solver.constraint_state,
            rigid_global_info=rigid_solver._rigid_global_info,
            static_rigid_sim_config=rigid_solver._static_rigid_sim_config,
            static_rigid_sim_cache_key=rigid_solver._static_rigid_sim_cache_key,
        )
        func_solve(
            entities_info=rigid_solver.entities_info,
            dofs_state=rigid_solver.dofs_state,
            constraint_state=constraint_solver.constraint_state,
            rigid_global_info=rigid_solver._rigid_global_info,
            static_rigid_sim_config=rigid_solver._static_rigid_sim_config,
            static_rigid_sim_cache_key=rigid_solver._static_rigid_sim_cache_key,
        )

    def scene_step():
        # Mock the scene step because if we call scene.step(), it will overwrite the necessary information that we need to
        # compute the gradients for
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
            static_rigid_sim_cache_key=rigid_solver._static_rigid_sim_cache_key,
        )
        rigid_solver._func_constraint_clear()
        constraint_solver.add_equality_constraints()
        rigid_solver.collider.detection()
        constraint_solver.add_frictionloss_constraints()
        constraint_solver.add_collision_constraints()
        constraint_solver.add_joint_limit_constraints()
        constraint_solver_resolve()

    def compute_loss(input_mass, input_jac, input_aref, input_efc_D, input_force):
        rigid_solver._rigid_global_info.mass_mat.from_numpy(input_mass.detach().cpu().numpy())
        constraint_solver.constraint_state.jac.from_numpy(input_jac.detach().cpu().numpy())
        constraint_solver.constraint_state.aref.from_numpy(input_aref.detach().cpu().numpy())
        constraint_solver.constraint_state.efc_D.from_numpy(input_efc_D.detach().cpu().numpy())
        rigid_solver.dofs_state.force.from_numpy(input_force.detach().cpu().numpy())

        # Recompute acc_smooth from the updated input variables
        updated_acc_smooth = torch.linalg.solve(input_mass.squeeze(-1), input_force.squeeze(-1))
        input_acc_smooth = updated_acc_smooth.unsqueeze(-1).detach().cpu().numpy()

        rigid_solver.dofs_state.acc_smooth.from_numpy(input_acc_smooth)

        constraint_solver_resolve()
        output_qacc = constraint_solver.qacc.to_numpy()
        th_output_qacc = torch.from_numpy(output_qacc).to(device=target_qacc.device)
        loss = ((th_output_qacc - target_qacc) ** 2).mean()
        return loss

    # Step once to compute constraint solver's inputs: [mass], [jac], [aref], [efc_D], [force]. We compute gradients for them.
    scene_step()
    init_input_mass = ti_to_torch(rigid_solver._rigid_global_info.mass_mat)
    init_input_jac = ti_to_torch(constraint_solver.constraint_state.jac)
    init_input_aref = ti_to_torch(constraint_solver.constraint_state.aref)
    init_input_efc_D = ti_to_torch(constraint_solver.constraint_state.efc_D)
    init_input_force = ti_to_torch(rigid_solver.dofs_state.force)

    # [acc_smooth] is dependent on [force], not an independent input ---> No need to compute gradient for it
    init_input_acc_smooth = ti_to_torch(rigid_solver.dofs_state.acc_smooth)

    # Initial output of the constraint solver
    init_output_qacc = ti_to_torch(constraint_solver.qacc)
    target_qacc = torch.rand_like(init_output_qacc) * init_output_qacc.abs().mean()

    # Number of constraints
    n_constraints = ti_to_torch(constraint_solver.n_constraints)

    # Solve the constraint solver and get the output
    output_qacc = constraint_solver.qacc.to_numpy()
    th_output_qacc = torch.from_numpy(output_qacc).to(device=target_qacc.device).requires_grad_(True)

    # Compute loss and gradient of the output
    loss = ((th_output_qacc - target_qacc) ** 2).mean()
    dL_dqacc = torch.autograd.grad(loss, th_output_qacc)[0].cpu().numpy()

    # Compute gradients of the input variables: [mass], [jac], [aref], [efc_D], [force]
    constraint_solver.backward(dL_dqacc)

    # Fetch gradients of the input variables
    dL_dM = ti_to_torch(constraint_solver.constraint_state.dL_dM)
    dL_djac = ti_to_torch(constraint_solver.constraint_state.dL_djac)
    dL_daref = ti_to_torch(constraint_solver.constraint_state.dL_daref)
    dL_defc_D = ti_to_torch(constraint_solver.constraint_state.dL_defc_D)
    dL_dforce = ti_to_torch(constraint_solver.constraint_state.dL_dforce)

    ### Compute directional derivatives along random directions
    FD_EPS = 1e-4
    TRIALS = 100
    errors = {}

    def compute_dL_error(dL_dx, x_type):
        dL_error = 0.0
        for trial in range(TRIALS):
            rand_dx = torch.randn_like(dL_dx)
            rand_dx = torch.nn.functional.normalize(rand_dx, dim=0 if x_type in ["force", "aref", "efc_D"] else [0, 1])
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
        return dL_error

    dL_dforce_error = compute_dL_error(dL_dforce, "force")
    dL_daref_error = compute_dL_error(dL_daref, "aref")
    dL_defc_D_error = compute_dL_error(dL_defc_D, "efc_D")
    dL_djac_error = compute_dL_error(dL_djac, "jac")
    dL_dM_error = compute_dL_error(dL_dM, "mass")

    assert (
        dL_dforce_error < rtol
    ), f"Relative error (dL_dforce) is too large: {dL_dforce_error * 100.0:.4g}% < {rtol * 100.0:.4g}%"
    assert (
        dL_daref_error < rtol
    ), f"Relative error (dL_daref) is too large: {dL_daref_error * 100.0:.4g}% < {rtol * 100.0:.4g}%"
    assert (
        dL_defc_D_error < rtol
    ), f"Relative error (dL_defc_D) is too large: {dL_defc_D_error * 100.0:.4g}% < {rtol * 100.0:.4g}%"
    assert (
        dL_djac_error < rtol
    ), f"Relative error (dL_djac) is too large: {dL_djac_error * 100.0:.4g}% < {rtol * 100.0:.4g}%"
    assert dL_dM_error < rtol, f"Relative error (dL_dM) is too large: {dL_dM_error * 100.0:.4g}% < {rtol * 100.0:.4g}%"
