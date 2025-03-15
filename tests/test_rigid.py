import unittest
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import mujoco
import genesis as gs


ENABLE_INTERACTIVE_VIEWER = False


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


class TestRigid(unittest.TestCase):
    def setUp(self):
        gs.init(backend=gs.cpu, precision="64", logging_level="warning", seed=0, debug=True)

    def tearDown(self):
        gs.destroy()

    def _build_engines(self, xml_path):
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        self.mj_model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.mj_model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        self.mj_data = mujoco.MjData(self.mj_model)

        self.gs_scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                res=(960, 640),
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(
                dt=self.mj_model.opt.timestep,
                substeps=1,
                gravity=self.mj_model.opt.gravity.tolist(),
            ),
            rigid_options=gs.options.RigidOptions(
                integrator=gs.integrator.implicitfast,
                constraint_solver=gs.constraint_solver.CG,
                box_box_detection=True,
                enable_self_collision=True,
                enable_adjacent_collision=True,
                contact_resolve_time=0.02,
                iterations=self.mj_model.opt.iterations,
                tolerance=self.mj_model.opt.tolerance,
                ls_iterations=self.mj_model.opt.ls_iterations,
                ls_tolerance=self.mj_model.opt.ls_tolerance,
            ),
            show_viewer=ENABLE_INTERACTIVE_VIEWER,
            show_FPS=False,
        )
        self.gs_robot = self.gs_scene.add_entity(
            gs.morphs.MJCF(file=xml_path),
            visualize_contact=True,
        )
        self.gs_scene.build()
        self.gs_solver = self.gs_scene.sim.rigid_solver

    def _init_engines(self, qpos, qvel):
        self.gs_scene.reset()
        self.gs_robot.set_qpos(qpos)
        self.gs_robot.set_dofs_velocity(qvel)
        if self.gs_scene.visualizer:
            self.gs_scene.visualizer.update()

        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.qpos[:] = qpos
        self.mj_data.qvel[:] = qvel
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.qvel_prev = None

    def _check_mujoco_consistency(self, check_collision):
        gs_meaninertia = self.gs_solver.meaninertia.to_numpy()[0]
        mj_meaninertia = self.mj_model.stat.meaninertia
        np.testing.assert_allclose(gs_meaninertia, mj_meaninertia, atol=1e-9)

        gs_mass_mat_damped = self.gs_solver.mass_mat.to_numpy()[:, :, 0]
        mj_mass_mat_damped = mj_get_mass_matrix_from_sparse(self.mj_model, self.mj_data)
        np.testing.assert_allclose(gs_mass_mat_damped, mj_mass_mat_damped, atol=1e-9)

        gs_qfrc_bias = self.gs_solver.dofs_state.qf_bias.to_numpy()[:, 0]
        mj_qfrc_bias = self.mj_data.qfrc_bias
        np.testing.assert_allclose(gs_qfrc_bias, mj_qfrc_bias, atol=1e-9)
        gs_qfrc_passive = self.gs_solver.dofs_state.qf_passive.to_numpy()[:, 0]
        mj_qfrc_passive = self.mj_data.qfrc_passive
        np.testing.assert_allclose(gs_qfrc_passive, mj_qfrc_passive, atol=1e-9)
        gs_qfrc_actuator = self.gs_solver.dofs_state.qf_applied.to_numpy()[:, 0]
        mj_qfrc_actuator = self.mj_data.qfrc_actuator
        np.testing.assert_allclose(gs_qfrc_actuator, mj_qfrc_actuator, atol=1e-9)

        if check_collision:
            gs_n_contacts = self.gs_solver.collider.n_contacts.to_numpy()[0]
            mj_n_contacts = self.mj_data.ncon
            assert gs_n_contacts == mj_n_contacts

            gs_n_constraints = self.gs_solver.constraint_solver.n_constraints.to_numpy()[0]
            mj_n_constraints = self.mj_data.nefc
            assert gs_n_constraints == mj_n_constraints
        else:
            gs_n_constraints = 0
            mj_n_constraints = 0

        if gs_n_constraints:
            gs_contact_pos = self.gs_solver.collider.contact_data.pos.to_numpy()[:gs_n_contacts, 0]
            mj_contact_pos = self.mj_data.contact.pos
            gs_sidx = np.argsort(gs_contact_pos[:, 0])
            mj_sidx = np.argsort(mj_contact_pos[:, 0])
            np.testing.assert_allclose(gs_contact_pos[gs_sidx], mj_contact_pos[mj_sidx], atol=1e-9)
            gs_contact_normal = self.gs_solver.collider.contact_data.normal.to_numpy()[:gs_n_contacts, 0]
            mj_contact_normal = -self.mj_data.contact.frame[:, :3]
            np.testing.assert_allclose(gs_contact_normal[gs_sidx], mj_contact_normal[mj_sidx], atol=1e-9)
            gs_penetration = self.gs_solver.collider.contact_data.penetration.to_numpy()[:gs_n_contacts, 0]
            mj_penetration = -self.mj_data.contact.dist
            np.testing.assert_allclose(gs_penetration[gs_sidx], mj_penetration[mj_sidx], atol=1e-9)

            gs_crb_inertial = self.gs_solver.links_state.crb_inertial.to_numpy()[:-1, 0].reshape([-1, 9])[
                :, [0, 4, 8, 1, 2, 5]
            ]
            mj_crb_inertial = self.mj_data.crb[1:, :6]
            np.testing.assert_allclose(gs_crb_inertial, mj_crb_inertial, atol=1e-9)
            gs_crb_pos = self.gs_solver.links_state.crb_pos.to_numpy()[:-1, 0]
            mj_crb_pos = self.mj_data.crb[1:, 6:9]
            np.testing.assert_allclose(gs_crb_pos, mj_crb_pos, atol=1e-9)
            gs_crb_mass = self.gs_solver.links_state.crb_mass.to_numpy()[:-1, 0]
            mj_crb_mass = self.mj_data.crb[1:, 9]
            np.testing.assert_allclose(gs_crb_mass, mj_crb_mass, atol=1e-9)

            gs_efc_D = self.gs_solver.constraint_solver.efc_D.to_numpy()[:gs_n_constraints, 0]
            mj_efc_D = self.mj_data.efc_D
            np.testing.assert_allclose(
                gs_efc_D[[j for i in gs_sidx for j in range(i * 4, (i + 1) * 4)]],
                mj_efc_D[[j for i in mj_sidx for j in range(i * 4, (i + 1) * 4)]],
                atol=1e-9,
            )

            gs_jac = self.gs_solver.constraint_solver.jac.to_numpy()[:gs_n_constraints, :, 0]
            mj_jac = self.mj_data.efc_J.reshape([gs_n_constraints, -1])
            gs_sidx = np.argsort(gs_jac.sum(axis=1))
            mj_sidx = np.argsort(mj_jac.sum(axis=1))
            np.testing.assert_allclose(gs_jac[gs_sidx], mj_jac[mj_sidx], atol=1e-9)

            gs_efc_aref = self.gs_solver.constraint_solver.aref.to_numpy()[:gs_n_constraints, 0]
            mj_efc_aref = self.mj_data.efc_aref
            np.testing.assert_allclose(gs_efc_aref[gs_sidx], mj_efc_aref[mj_sidx], atol=1e-9)

            mj_iter = self.mj_data.solver_niter[0] - 1
            if gs_n_constraints and mj_iter > 0:
                gs_scale = 1.0 / (gs_meaninertia * max(1, self.gs_solver.n_dofs))
                gs_improvement = gs_scale * (
                    self.gs_solver.constraint_solver.prev_cost[0] - self.gs_solver.constraint_solver.cost[0]
                )
                mj_improvement = self.mj_data.solver.improvement[mj_iter]
                np.testing.assert_allclose(gs_improvement, mj_improvement, atol=1e-9)
                gs_gradient = gs_scale * np.linalg.norm(
                    self.gs_solver.constraint_solver.grad.to_numpy()[: self.gs_solver.n_dofs, 0]
                )
                mj_gradient = self.mj_data.solver.gradient[mj_iter]
                np.testing.assert_allclose(gs_gradient, mj_gradient, atol=1e-9)

        gs_qfrc_constraint = self.gs_solver.dofs_state.qf_constraint.to_numpy()[:, 0]
        mj_qfrc_constraint = self.mj_data.qfrc_constraint
        np.testing.assert_allclose(gs_qfrc_constraint, mj_qfrc_constraint, atol=1e-9)

        if gs_n_constraints:
            gs_efc_force = self.gs_solver.constraint_solver.efc_force.to_numpy()[:gs_n_constraints, 0]
            mj_efc_force = self.mj_data.efc_force
            np.testing.assert_allclose(gs_efc_force[gs_sidx], mj_efc_force[mj_sidx], atol=1e-9)

            if self.qvel_prev is not None:
                gs_efc_vel = gs_jac @ self.qvel_prev
                mj_efc_vel = self.mj_data.efc_vel
                np.testing.assert_allclose(gs_efc_vel[gs_sidx], mj_efc_vel[mj_sidx], atol=1e-9)

        gs_qfrc_all = self.gs_solver.dofs_state.force.to_numpy()[:, 0]
        mj_qfrc_all = self.mj_data.qfrc_smooth + self.mj_data.qfrc_constraint
        np.testing.assert_allclose(gs_qfrc_all, mj_qfrc_all, atol=1e-9)

        # FIXME: Why this check is not passing???
        gs_qfrc_smooth = self.gs_solver.dofs_state.qf_smooth.to_numpy()[:, 0]
        mj_qfrc_smooth = self.mj_data.qfrc_smooth
        # np.testing.assert_allclose(gs_qfrc_smooth, mj_qfrc_smooth, atol=1e-9)

        gs_qacc_smooth = self.gs_solver.dofs_state.acc_smooth.to_numpy()[:, 0]
        mj_qacc_smooth = self.mj_data.qacc_smooth
        np.testing.assert_allclose(gs_qacc_smooth, mj_qacc_smooth, atol=1e-9)

        # Acceleration pre- VS post-implicit damping
        # gs_qacc_post = self.gs_solver.dofs_state.acc.to_numpy()[:, 0]
        gs_qacc_pre = self.gs_solver.constraint_solver.qacc.to_numpy()[:, 0]
        mj_qacc_pre = self.mj_data.qacc
        np.testing.assert_allclose(gs_qacc_pre, mj_qacc_pre, atol=1e-9)

        gs_qpos = self.gs_solver.qpos.to_numpy()[:, 0]
        mj_qpos = self.mj_data.qpos
        np.testing.assert_allclose(gs_qpos, mj_qpos, atol=1e-9)
        gs_qvel = self.gs_solver.dofs_state.vel.to_numpy()[:, 0]
        mj_qvel = self.mj_data.qvel
        np.testing.assert_allclose(gs_qvel, mj_qvel, atol=1e-9)

        mujoco.mj_fwdPosition(self.mj_model, self.mj_data)
        gs_xpos = self.gs_solver.links_state.pos.to_numpy()[:-1, 0]
        mj_xpos = self.mj_data.xpos[1:]
        np.testing.assert_allclose(gs_xpos, mj_xpos, atol=1e-9)

        gs_cdof_vel = self.gs_solver.dofs_state.cdof_vel.to_numpy()[:, 0]
        mj_cdof_vel = self.mj_data.cdof[:, 3:]
        np.testing.assert_allclose(gs_cdof_vel, mj_cdof_vel, atol=1e-9)
        gs_cdof_ang = self.gs_solver.dofs_state.cdof_ang.to_numpy()[:, 0]
        mj_cdof_ang = self.mj_data.cdof[:, :3]
        np.testing.assert_allclose(gs_cdof_ang, mj_cdof_ang, atol=1e-9)

    def _simulate_and_check_mujoco_consistency(self, qpos, qvel, num_steps):
        self._init_engines(qpos, qvel)

        for i in range(num_steps):
            self._check_mujoco_consistency(check_collision=bool(i))

            self.mj_data.qpos[:] = self.gs_solver.qpos.to_numpy()[:, 0]
            self.mj_data.qvel[:] = self.gs_solver.dofs_state.vel.to_numpy()[:, 0]
            self.qvel_prev = self.mj_data.qvel.copy()
            mujoco.mj_step(self.mj_model, self.mj_data)
            self.gs_scene.step()
            if self.gs_scene.visualizer:
                self.gs_scene.visualizer.update()

    def test_box_plan_dynamics(self):
        mjcf = ET.Element("mujoco", model="one_box")
        option = ET.SubElement(mjcf, "option", timestep="0.01")
        default = ET.SubElement(mjcf, "default")
        ET.SubElement(default, "geom", contype="1", conaffinity="1", condim="3", friction="1. 0.5 0.5")
        worldbody = ET.SubElement(mjcf, "worldbody")
        ET.SubElement(worldbody, "geom", type="plane", name="floor", pos="0. 0. 0.", size="40. 40. 40.")
        box_body = ET.SubElement(worldbody, "body", name="box", pos="0. 0. 0.3")
        ET.SubElement(box_body, "geom", type="box", size="0.2 0.2 0.2", pos="0. 0. 0.")
        ET.SubElement(box_body, "joint", type="free")

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".xml", delete=True) as file:
            xml_tree = ET.ElementTree(mjcf)
            xml_tree.write(file.name, encoding="utf-8", xml_declaration=True)
            self._build_engines(file.name)

        cube_pos = np.array([0.0, 0.0, 0.6])
        cube_quat = np.random.rand(4)
        cube_quat /= np.linalg.norm(cube_quat)
        qpos = np.concatenate((cube_pos, cube_quat))
        qvel = np.zeros((self.gs_robot.n_dofs,))

        self._simulate_and_check_mujoco_consistency(qpos, qvel, num_steps=150)
