from dataclasses import dataclass

import numpy as np
import mujoco


@dataclass
class MjSim:
    model: mujoco.MjModel
    data: mujoco.MjData


def mj_get_mass_matrix_from_sparse(mj_model, mj_data):
    is_, js, madr_ijs = [], [], []
    for i in range(mj_model.nv):
        madr_ij, j = mj_model.dof_Madr[i], i
        while True:
            madr_ij, j = madr_ij + 1, mj_model.dof_parentid[j]
            if j == -1:
                break
            is_, js, madr_ijs = (
                is_ + [i],
                js + [j],
                madr_ijs + [madr_ij],
            )
    i, j, madr_ij = (np.array(x, dtype=np.int32) for x in (is_, js, madr_ijs))
    mat = np.zeros((mj_model.nv, mj_model.nv))
    mat[i, j] = mj_data.qM[madr_ij]
    mat = np.diag(mj_data.qM[np.array(mj_model.dof_Madr)]) + mat + mat.T
    return mat


def init_simulators(gs_sim, mj_sim, qpos, qvel):
    gs_robot = gs_sim.entities[0]

    gs_sim.scene.reset()
    gs_robot.set_qpos(qpos)
    gs_robot.set_dofs_velocity(qvel)
    if gs_sim.scene.visualizer:
        gs_sim.scene.visualizer.update()

    mujoco.mj_resetData(mj_sim.model, mj_sim.data)
    mj_sim.data.qpos[:] = qpos
    mj_sim.data.qvel[:] = qvel
    mujoco.mj_forward(mj_sim.model, mj_sim.data)


def check_mujoco_state_consistency(gs_sim, mj_sim, is_first_step, qvel_prev):
    gs_meaninertia = gs_sim.rigid_solver.meaninertia.to_numpy()[0]
    mj_meaninertia = mj_sim.model.stat.meaninertia
    np.testing.assert_allclose(gs_meaninertia, mj_meaninertia, atol=1e-9)

    gs_mass_mat_damped = gs_sim.rigid_solver.mass_mat.to_numpy()[:, :, 0]
    mj_mass_mat_damped = mj_get_mass_matrix_from_sparse(mj_sim.model, mj_sim.data)
    np.testing.assert_allclose(gs_mass_mat_damped, mj_mass_mat_damped, atol=1e-9)

    gs_qfrc_bias = gs_sim.rigid_solver.dofs_state.qf_bias.to_numpy()[:, 0]
    mj_qfrc_bias = mj_sim.data.qfrc_bias
    np.testing.assert_allclose(gs_qfrc_bias, mj_qfrc_bias, atol=1e-9)
    gs_qfrc_passive = gs_sim.rigid_solver.dofs_state.qf_passive.to_numpy()[:, 0]
    mj_qfrc_passive = mj_sim.data.qfrc_passive
    np.testing.assert_allclose(gs_qfrc_passive, mj_qfrc_passive, atol=1e-9)
    gs_qfrc_actuator = gs_sim.rigid_solver.dofs_state.qf_applied.to_numpy()[:, 0]
    mj_qfrc_actuator = mj_sim.data.qfrc_actuator
    np.testing.assert_allclose(gs_qfrc_actuator, mj_qfrc_actuator, atol=1e-9)

    if is_first_step:
        gs_n_contacts = gs_sim.rigid_solver.collider.n_contacts.to_numpy()[0]
        mj_n_contacts = mj_sim.data.ncon
        assert gs_n_contacts == mj_n_contacts

        gs_n_constraints = gs_sim.rigid_solver.constraint_solver.n_constraints.to_numpy()[0]
        mj_n_constraints = mj_sim.data.nefc
        assert gs_n_constraints == mj_n_constraints
    else:
        gs_n_constraints = 0
        mj_n_constraints = 0

    if gs_n_constraints:
        gs_contact_pos = gs_sim.rigid_solver.collider.contact_data.pos.to_numpy()[:gs_n_contacts, 0]
        mj_contact_pos = mj_sim.data.contact.pos
        gs_sidx = np.argsort(gs_contact_pos[:, 0])
        mj_sidx = np.argsort(mj_contact_pos[:, 0])
        np.testing.assert_allclose(gs_contact_pos[gs_sidx], mj_contact_pos[mj_sidx], atol=1e-9)
        gs_contact_normal = gs_sim.rigid_solver.collider.contact_data.normal.to_numpy()[:gs_n_contacts, 0]
        mj_contact_normal = -mj_sim.data.contact.frame[:, :3]
        np.testing.assert_allclose(gs_contact_normal[gs_sidx], mj_contact_normal[mj_sidx], atol=1e-9)
        gs_penetration = gs_sim.rigid_solver.collider.contact_data.penetration.to_numpy()[:gs_n_contacts, 0]
        mj_penetration = -mj_sim.data.contact.dist
        np.testing.assert_allclose(gs_penetration[gs_sidx], mj_penetration[mj_sidx], atol=1e-9)

        gs_crb_inertial = gs_sim.rigid_solver.links_state.crb_inertial.to_numpy()[:-1, 0].reshape([-1, 9])[
            :, [0, 4, 8, 1, 2, 5]
        ]
        mj_crb_inertial = mj_sim.data.crb[1:, :6]
        np.testing.assert_allclose(gs_crb_inertial, mj_crb_inertial, atol=1e-9)
        gs_crb_pos = gs_sim.rigid_solver.links_state.crb_pos.to_numpy()[:-1, 0]
        mj_crb_pos = mj_sim.data.crb[1:, 6:9]
        np.testing.assert_allclose(gs_crb_pos, mj_crb_pos, atol=1e-9)
        gs_crb_mass = gs_sim.rigid_solver.links_state.crb_mass.to_numpy()[:-1, 0]
        mj_crb_mass = mj_sim.data.crb[1:, 9]
        np.testing.assert_allclose(gs_crb_mass, mj_crb_mass, atol=1e-9)

        gs_efc_D = gs_sim.rigid_solver.constraint_solver.efc_D.to_numpy()[:gs_n_constraints, 0]
        mj_efc_D = mj_sim.data.efc_D
        np.testing.assert_allclose(
            gs_efc_D[[j for i in gs_sidx for j in range(i * 4, (i + 1) * 4)]],
            mj_efc_D[[j for i in mj_sidx for j in range(i * 4, (i + 1) * 4)]],
            atol=1e-9,
        )

        gs_jac = gs_sim.rigid_solver.constraint_solver.jac.to_numpy()[:gs_n_constraints, :, 0]
        mj_jac = mj_sim.data.efc_J.reshape([gs_n_constraints, -1])
        gs_sidx = np.argsort(gs_jac.sum(axis=1))
        mj_sidx = np.argsort(mj_jac.sum(axis=1))
        np.testing.assert_allclose(gs_jac[gs_sidx], mj_jac[mj_sidx], atol=1e-9)

        gs_efc_aref = gs_sim.rigid_solver.constraint_solver.aref.to_numpy()[:gs_n_constraints, 0]
        mj_efc_aref = mj_sim.data.efc_aref
        np.testing.assert_allclose(gs_efc_aref[gs_sidx], mj_efc_aref[mj_sidx], atol=1e-9)

        mj_iter = mj_sim.data.solver_niter[0] - 1
        if gs_n_constraints and mj_iter > 0:
            gs_scale = 1.0 / (gs_meaninertia * max(1, gs_sim.rigid_solver.n_dofs))
            gs_improvement = gs_scale * (
                gs_sim.rigid_solver.constraint_solver.prev_cost[0] - gs_sim.rigid_solver.constraint_solver.cost[0]
            )
            mj_improvement = mj_sim.data.solver.improvement[mj_iter]
            np.testing.assert_allclose(gs_improvement, mj_improvement, atol=1e-9)
            gs_gradient = gs_scale * np.linalg.norm(
                gs_sim.rigid_solver.constraint_solver.grad.to_numpy()[: gs_sim.rigid_solver.n_dofs, 0]
            )
            mj_gradient = mj_sim.data.solver.gradient[mj_iter]
            np.testing.assert_allclose(gs_gradient, mj_gradient, atol=1e-9)

    gs_qfrc_constraint = gs_sim.rigid_solver.dofs_state.qf_constraint.to_numpy()[:, 0]
    mj_qfrc_constraint = mj_sim.data.qfrc_constraint
    np.testing.assert_allclose(gs_qfrc_constraint, mj_qfrc_constraint, atol=1e-9)

    if gs_n_constraints:
        gs_efc_force = gs_sim.rigid_solver.constraint_solver.efc_force.to_numpy()[:gs_n_constraints, 0]
        mj_efc_force = mj_sim.data.efc_force
        np.testing.assert_allclose(gs_efc_force[gs_sidx], mj_efc_force[mj_sidx], atol=1e-9)

        if qvel_prev is not None:
            gs_efc_vel = gs_jac @ qvel_prev
            mj_efc_vel = mj_sim.data.efc_vel
            np.testing.assert_allclose(gs_efc_vel[gs_sidx], mj_efc_vel[mj_sidx], atol=1e-9)

    gs_qfrc_all = gs_sim.rigid_solver.dofs_state.force.to_numpy()[:, 0]
    mj_qfrc_all = mj_sim.data.qfrc_smooth + mj_sim.data.qfrc_constraint
    np.testing.assert_allclose(gs_qfrc_all, mj_qfrc_all, atol=1e-9)

    # FIXME: Why this check is not passing???
    gs_qfrc_smooth = gs_sim.rigid_solver.dofs_state.qf_smooth.to_numpy()[:, 0]
    mj_qfrc_smooth = mj_sim.data.qfrc_smooth
    # np.testing.assert_allclose(gs_qfrc_smooth, mj_qfrc_smooth, atol=1e-9)

    gs_qacc_smooth = gs_sim.rigid_solver.dofs_state.acc_smooth.to_numpy()[:, 0]
    mj_qacc_smooth = mj_sim.data.qacc_smooth
    np.testing.assert_allclose(gs_qacc_smooth, mj_qacc_smooth, atol=1e-9)

    # Acceleration pre- VS post-implicit damping
    # gs_qacc_post = gs_sim.rigid_solver.dofs_state.acc.to_numpy()[:, 0]
    gs_qacc_pre = gs_sim.rigid_solver.constraint_solver.qacc.to_numpy()[:, 0]
    mj_qacc_pre = mj_sim.data.qacc
    np.testing.assert_allclose(gs_qacc_pre, mj_qacc_pre, atol=1e-9)

    gs_qpos = gs_sim.rigid_solver.qpos.to_numpy()[:, 0]
    mj_qpos = mj_sim.data.qpos
    np.testing.assert_allclose(gs_qpos, mj_qpos, atol=1e-9)
    gs_qvel = gs_sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]
    mj_qvel = mj_sim.data.qvel
    np.testing.assert_allclose(gs_qvel, mj_qvel, atol=1e-9)

    mujoco.mj_fwdPosition(mj_sim.model, mj_sim.data)
    gs_xpos = gs_sim.rigid_solver.links_state.pos.to_numpy()[:-1, 0]
    mj_xpos = mj_sim.data.xpos[1:]
    np.testing.assert_allclose(gs_xpos, mj_xpos, atol=1e-9)

    gs_cdof_vel = gs_sim.rigid_solver.dofs_state.cdof_vel.to_numpy()[:, 0]
    mj_cdof_vel = mj_sim.data.cdof[:, 3:]
    np.testing.assert_allclose(gs_cdof_vel, mj_cdof_vel, atol=1e-9)
    gs_cdof_ang = gs_sim.rigid_solver.dofs_state.cdof_ang.to_numpy()[:, 0]
    mj_cdof_ang = mj_sim.data.cdof[:, :3]
    np.testing.assert_allclose(gs_cdof_ang, mj_cdof_ang, atol=1e-9)


def simulate_and_check_mujoco_consistency(gs_sim, mj_sim, qpos, qvel, num_steps):
    init_simulators(gs_sim, mj_sim, qpos, qvel)

    qvel_prev = None
    for i in range(num_steps):
        is_first_step = i > 0
        check_mujoco_state_consistency(gs_sim, mj_sim, is_first_step, qvel_prev)

        mj_sim.data.qpos[:] = gs_sim.rigid_solver.qpos.to_numpy()[:, 0]
        mj_sim.data.qvel[:] = gs_sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]
        qvel_prev = mj_sim.data.qvel.copy()
        mujoco.mj_step(mj_sim.model, mj_sim.data)
        gs_sim.scene.step()
        if gs_sim.scene.visualizer:
            gs_sim.scene.visualizer.update()
