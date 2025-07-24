from typing import TYPE_CHECKING
import numpy as np
import taichi as ti
import numpy.typing as npt

import genesis as gs
import genesis.utils.geom as gu
import genesis.engine.solvers.rigid.array_class as array_class

if TYPE_CHECKING:
    from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver


@ti.data_oriented
class ConstraintSolver:
    def __init__(self, rigid_solver: "RigidSolver"):
        self._solver = rigid_solver
        self._collider = rigid_solver.collider
        self._B = rigid_solver._B
        self._para_level = rigid_solver._para_level

        self._solver_type = rigid_solver._options.constraint_solver
        self.iterations = rigid_solver._options.iterations
        self.tolerance = rigid_solver._options.tolerance
        self.ls_iterations = rigid_solver._options.ls_iterations
        self.ls_tolerance = rigid_solver._options.ls_tolerance
        self.sparse_solve = rigid_solver._options.sparse_solve

        # 4 constraints per contact, 1 constraints per joint limit (upper and lower, if not inf), and 3 constraints per equality
        self.len_constraints = (
            5 * rigid_solver.collider._collider_info._max_contact_pairs[None]
            + np.logical_not(np.isinf(self._solver.dofs_info.limit.to_numpy()[:, 0])).sum()
            + self._solver.n_equalities_candidate * 6
        )
        self.len_constraints_ = max(1, self.len_constraints)

        self.constraint_state = array_class.ConstraintState(self, self._solver)

        # self.ti_n_equalities = ti.field(gs.ti_int, shape=self._solver._batch_shape())
        # self.ti_n_equalities.from_numpy(np.full((self._solver._B,), self._solver.n_equalities, dtype=gs.np_int))

        # jac_shape = self._solver._batch_shape((self.len_constraints_, self._solver.n_dofs_))
        # if (jac_shape[0] * jac_shape[1] * jac_shape[2]) > np.iinfo(np.int32).max:
        #     raise ValueError(
        #         f"Jacobian shape {jac_shape} is too large for int32. "
        #         "Consider reducing the number of constraints or the number of degrees of freedom."
        #     )

        # self.jac = ti.field(
        #     dtype=gs.ti_float, shape=self._solver._batch_shape((self.len_constraints_, self._solver.n_dofs_))
        # )
        # self.diag = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self.len_constraints_))
        # self.aref = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self.len_constraints_))

        # self.jac_relevant_dofs = ti.field(
        #     gs.ti_int, shape=self._solver._batch_shape((self.len_constraints_, self._solver.n_dofs_))
        # )
        # self.jac_n_relevant_dofs = ti.field(gs.ti_int, shape=self._solver._batch_shape(self.len_constraints_))

        # self.n_constraints = ti.field(gs.ti_int, shape=self._solver._batch_shape())
        # self.n_constraints_equality = ti.field(gs.ti_int, shape=self._solver._batch_shape())
        # self.improved = ti.field(gs.ti_int, shape=self._solver._batch_shape())

        # self.Jaref = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self.len_constraints_))
        # self.Ma = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self._solver.n_dofs_))
        # self.Ma_ws = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self._solver.n_dofs_))
        # self.grad = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self._solver.n_dofs_))
        # self.Mgrad = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self._solver.n_dofs_))
        # self.search = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self._solver.n_dofs_))

        # self.efc_D = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self.len_constraints_))
        # self.efc_force = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self.len_constraints_))
        # self.active = ti.field(dtype=gs.ti_int, shape=self._solver._batch_shape(self.len_constraints_))
        # self.prev_active = ti.field(dtype=gs.ti_int, shape=self._solver._batch_shape(self.len_constraints_))
        # self.qfrc_constraint = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self._solver.n_dofs_))
        # self.qacc = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self._solver.n_dofs_))
        # self.qacc_ws = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self._solver.n_dofs_))
        # self.qacc_prev = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self._solver.n_dofs_))

        # self.cost_ws = ti.field(gs.ti_float, shape=self._solver._batch_shape())

        # self.gauss = ti.field(gs.ti_float, shape=self._solver._batch_shape())
        # self.cost = ti.field(gs.ti_float, shape=self._solver._batch_shape())
        # self.prev_cost = ti.field(gs.ti_float, shape=self._solver._batch_shape())

        # ## line search
        # self.gtol = ti.field(gs.ti_float, shape=self._solver._batch_shape())

        # self.mv = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self._solver.n_dofs_))
        # self.jv = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self.len_constraints_))
        # self.quad_gauss = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(3))
        # self.quad = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape((self.len_constraints_, 3)))

        # self.candidates = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(12))
        # self.ls_its = ti.field(gs.ti_float, shape=self._solver._batch_shape())
        # self.ls_result = ti.field(gs.ti_int, shape=self._solver._batch_shape())

        # if self._solver_type == gs.constraint_solver.CG:
        #     self.cg_prev_grad = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self._solver.n_dofs_))
        #     self.cg_prev_Mgrad = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self._solver.n_dofs_))
        #     self.cg_beta = ti.field(gs.ti_float, shape=self._solver._batch_shape())
        #     self.cg_pg_dot_pMg = ti.field(gs.ti_float, shape=self._solver._batch_shape())

        # if self._solver_type == gs.constraint_solver.Newton:
        #     self.nt_H = ti.field(
        #         dtype=gs.ti_float, shape=self._solver._batch_shape((self._solver.n_dofs_, self._solver.n_dofs_))
        #     )
        #     self.nt_vec = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self._solver.n_dofs_))

        cs = self.constraint_state
        self.ti_n_equalities = cs.ti_n_equalities
        self.jac = cs.jac
        self.diag = cs.diag
        self.aref = cs.aref
        self.jac_n_relevant_dofs = cs.jac_n_relevant_dofs
        self.jac_relevant_dofs = cs.jac_relevant_dofs
        self.n_constraints = cs.n_constraints
        self.n_constraints_equality = cs.n_constraints_equality
        self.improved = cs.improved
        self.Jaref = cs.Jaref
        self.Ma = cs.Ma
        self.Ma_ws = cs.Ma_ws
        self.grad = cs.grad
        self.Mgrad = cs.Mgrad
        self.search = cs.search
        self.efc_D = cs.efc_D
        self.efc_force = cs.efc_force
        self.active = cs.active
        self.prev_active = cs.prev_active
        self.qfrc_constraint = cs.qfrc_constraint
        self.qacc = cs.qacc
        self.qacc_ws = cs.qacc_ws
        self.qacc_prev = cs.qacc_prev
        self.cost_ws = cs.cost_ws
        self.gauss = cs.gauss
        self.cost = cs.cost
        self.prev_cost = cs.prev_cost
        self.gtol = cs.gtol
        self.mv = cs.mv
        self.jv = cs.jv
        self.quad_gauss = cs.quad_gauss
        self.quad = cs.quad
        self.candidates = cs.candidates
        self.ls_its = cs.ls_its
        self.ls_result = cs.ls_result
        if self._solver_type == gs.constraint_solver.CG:
            self.cg_prev_grad = cs.cg_prev_grad
            self.cg_prev_Mgrad = cs.cg_prev_Mgrad
            self.cg_beta = cs.cg_beta
            self.cg_pg_dot_pMg = cs.cg_pg_dot_pMg
        if self._solver_type == gs.constraint_solver.Newton:
            self.nt_H = cs.nt_H
            self.nt_vec = cs.nt_vec

        self.reset()

        #

        # self.constraint_state.ti_n_equalities = self.ti_n_equalities
        # self.constraint_state.jac = self.jac
        # self.constraint_state.diag = self.diag
        # self.constraint_state.aref = self.aref
        # self.constraint_state.jac_n_relevant_dofs = self.jac_n_relevant_dofs
        # self.constraint_state.jac_relevant_dofs = self.jac_relevant_dofs
        # self.constraint_state.n_constraints = self.n_constraints
        # self.constraint_state.n_constraints_equality = self.n_constraints_equality
        # self.constraint_state.improved = self.improved
        # self.constraint_state.Jaref = self.Jaref
        # self.constraint_state.Ma = self.Ma
        # self.constraint_state.Ma_ws = self.Ma_ws
        # self.constraint_state.grad = self.grad
        # self.constraint_state.Mgrad = self.Mgrad
        # self.constraint_state.search = self.search
        # self.constraint_state.efc_D = self.efc_D
        # self.constraint_state.efc_force = self.efc_force
        # self.constraint_state.active = self.active
        # self.constraint_state.prev_active = self.prev_active
        # self.constraint_state.qfrc_constraint = self.qfrc_constraint
        # self.constraint_state.qacc = self.qacc
        # self.constraint_state.qacc_ws = self.qacc_ws
        # self.constraint_state.qacc_prev = self.qacc_prev
        # self.constraint_state.cost_ws = self.cost_ws
        # self.constraint_state.gauss = self.gauss
        # self.constraint_state.cost = self.cost
        # self.constraint_state.prev_cost = self.prev_cost
        # self.constraint_state.gtol = self.gtol
        # self.constraint_state.mv = self.mv
        # self.constraint_state.jv = self.jv
        # self.constraint_state.quad_gauss = self.quad_gauss
        # self.constraint_state.quad = self.quad
        # self.constraint_state.candidates = self.candidates
        # self.constraint_state.ls_its = self.ls_its
        # self.constraint_state.ls_result = self.ls_result
        # if self._solver_type == gs.constraint_solver.CG:
        #     self.constraint_state.cg_prev_grad = self.cg_prev_grad
        #     self.constraint_state.cg_prev_Mgrad = self.cg_prev_Mgrad
        #     self.constraint_state.cg_beta = self.cg_beta
        #     self.constraint_state.cg_pg_dot_pMg = self.cg_pg_dot_pMg
        # if self._solver_type == gs.constraint_solver.Newton:
        #     self.constraint_state.nt_H = self.nt_H
        #     self.constraint_state.nt_vec = self.nt_vec

    def clear(self, envs_idx: npt.NDArray[np.int32] | None = None):
        if envs_idx is None:
            envs_idx = self._solver._scene._envs_idx
        self._kernel_clear(envs_idx)

    @ti.kernel
    def _kernel_clear(self, envs_idx: ti.types.ndarray()):
        ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b_ in range(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            self.n_constraints[i_b] = 0
            self.n_constraints_equality[i_b] = 0

    @ti.kernel
    def add_collision_constraints(
        self_unused,
        links_info: array_class.LinksInfo,
        links_state: array_class.LinksState,
        dofs_state: array_class.DofsState,
        constraint_state: ti.template(),
        collider_state: ti.template(),
        static_rigid_sim_config: ti.template(),
    ):
        _B = dofs_state.ctrl_mode.shape[1]
        n_dofs = dofs_state.ctrl_mode.shape[0]

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(_B):
            for i_col in range(collider_state.n_contacts[i_b]):
                contact_data = collider_state.contact_data[i_col, i_b]
                link_a = contact_data.link_a
                link_b = contact_data.link_b
                link_a_maybe_batch = [link_a, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else link_a
                link_b_maybe_batch = [link_b, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else link_b

                d1, d2 = gu.ti_orthogonals(contact_data.normal)

                invweight = links_info.invweight[link_a_maybe_batch][0]
                if link_b > -1:
                    invweight = invweight + links_info.invweight[link_b_maybe_batch][0]

                for i in range(4):
                    d = (2 * (i % 2) - 1) * (d1 if i < 2 else d2)
                    n = d * contact_data.friction - contact_data.normal

                    n_con = ti.atomic_add(constraint_state.n_constraints[i_b], 1)
                    if ti.static(static_rigid_sim_config.sparse_solve):
                        for i_d_ in range(constraint_state.jac_n_relevant_dofs[n_con, i_b]):
                            i_d = constraint_state.jac_relevant_dofs[n_con, i_d_, i_b]
                            constraint_state.jac[n_con, i_d, i_b] = gs.ti_float(0.0)
                    else:
                        for i_d in range(n_dofs):
                            constraint_state.jac[n_con, i_d, i_b] = gs.ti_float(0.0)

                    con_n_relevant_dofs = 0
                    jac_qvel = gs.ti_float(0.0)
                    for i_ab in range(2):
                        sign = gs.ti_float(-1.0)
                        link = link_a
                        if i_ab == 1:
                            sign = gs.ti_float(1.0)
                            link = link_b

                        while link > -1:
                            link_maybe_batch = (
                                [link, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else link
                            )

                            # reverse order to make sure dofs in each row of self.jac_relevant_dofs is strictly descending
                            for i_d_ in range(links_info.n_dofs[link_maybe_batch]):
                                i_d = links_info.dof_end[link_maybe_batch] - 1 - i_d_

                                cdof_ang = dofs_state.cdof_ang[i_d, i_b]
                                cdot_vel = dofs_state.cdof_vel[i_d, i_b]

                                t_quat = gu.ti_identity_quat()
                                t_pos = contact_data.pos - links_state.COM[link, i_b]
                                _, vel = gu.ti_transform_motion_by_trans_quat(cdof_ang, cdot_vel, t_pos, t_quat)

                                diff = sign * vel
                                jac = diff @ n
                                jac_qvel = jac_qvel + jac * dofs_state.vel[i_d, i_b]
                                constraint_state.jac[n_con, i_d, i_b] = constraint_state.jac[n_con, i_d, i_b] + jac

                                if ti.static(static_rigid_sim_config.sparse_solve):
                                    constraint_state.jac_relevant_dofs[n_con, con_n_relevant_dofs, i_b] = i_d
                                    con_n_relevant_dofs += 1

                            link = links_info.parent_idx[link_maybe_batch]

                    if ti.static(static_rigid_sim_config.sparse_solve):
                        constraint_state.jac_n_relevant_dofs[n_con, i_b] = con_n_relevant_dofs
                    imp, aref = gu.imp_aref(
                        contact_data.sol_params, -contact_data.penetration, jac_qvel, -contact_data.penetration
                    )

                    diag = invweight + contact_data.friction * contact_data.friction * invweight
                    diag *= 2 * contact_data.friction * contact_data.friction * (1 - imp) / imp
                    diag = ti.max(diag, gs.EPS)

                    constraint_state.diag[n_con, i_b] = diag
                    constraint_state.aref[n_con, i_b] = aref
                    constraint_state.efc_D[n_con, i_b] = 1 / diag

    @ti.func
    def _func_equality_connect(
        self_unused,
        i_b,
        i_e,
        links_info,
        links_state,
        dofs_state,
        equalities_info,
        constraint_state,
        collider_state,
        static_rigid_sim_config: ti.template(),
    ):
        n_dofs = dofs_state.ctrl_mode.shape[0]

        link1_idx = equalities_info.eq_obj1id[i_e, i_b]
        link2_idx = equalities_info.eq_obj2id[i_e, i_b]
        link_a_maybe_batch = [link1_idx, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else link1_idx
        link_b_maybe_batch = [link2_idx, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else link2_idx
        anchor1_pos = gs.ti_vec3(
            [
                equalities_info.eq_data[i_e, i_b][0],
                equalities_info.eq_data[i_e, i_b][1],
                equalities_info.eq_data[i_e, i_b][2],
            ]
        )
        anchor2_pos = gs.ti_vec3(
            [
                equalities_info.eq_data[i_e, i_b][3],
                equalities_info.eq_data[i_e, i_b][4],
                equalities_info.eq_data[i_e, i_b][5],
            ]
        )
        sol_params = equalities_info.sol_params[i_e, i_b]

        # Transform anchor positions to global coordinates
        global_anchor1 = gu.ti_transform_by_trans_quat(
            pos=anchor1_pos,
            trans=links_state.pos[link1_idx, i_b],
            quat=links_state.quat[link1_idx, i_b],
        )
        global_anchor2 = gu.ti_transform_by_trans_quat(
            pos=anchor2_pos,
            trans=links_state.pos[link2_idx, i_b],
            quat=links_state.quat[link2_idx, i_b],
        )

        invweight = links_info.invweight[link_a_maybe_batch][0] + links_info.invweight[link_b_maybe_batch][0]

        for i_3 in range(3):
            n_con = ti.atomic_add(constraint_state.n_constraints[i_b], 1)
            ti.atomic_add(constraint_state.n_constraints_equality[i_b], 1)

            if ti.static(static_rigid_sim_config.sparse_solve):
                for i_d_ in range(collider_state.jac_n_relevant_dofs[n_con, i_b]):
                    i_d = constraint_state.jac_relevant_dofs[n_con, i_d_, i_b]
                    constraint_state.jac[n_con, i_d, i_b] = gs.ti_float(0.0)
            else:
                for i_d in range(n_dofs):
                    constraint_state.jac[n_con, i_d, i_b] = gs.ti_float(0.0)

            jac_qvel = gs.ti_float(0.0)
            for i_ab in range(2):
                sign = gs.ti_float(1.0)
                link = link1_idx
                pos = global_anchor1
                if i_ab == 1:
                    sign = gs.ti_float(-1.0)
                    link = link2_idx
                    pos = global_anchor2

                while link > -1:
                    link_maybe_batch = [link, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else link

                    for i_d_ in range(links_info.n_dofs[link_maybe_batch]):
                        i_d = links_info.dof_end[link_maybe_batch] - 1 - i_d_

                        cdof_ang = dofs_state.cdof_ang[i_d, i_b]
                        cdot_vel = dofs_state.cdof_vel[i_d, i_b]

                        t_quat = gu.ti_identity_quat()
                        t_pos = pos - links_state.COM[link, i_b]
                        ang, vel = gu.ti_transform_motion_by_trans_quat(cdof_ang, cdot_vel, t_pos, t_quat)

                        diff = sign * vel
                        jac = diff[i_3]
                        jac_qvel = jac_qvel + jac * dofs_state.vel[i_d, i_b]
                        constraint_state.jac[n_con, i_d, i_b] = constraint_state.jac[n_con, i_d, i_b] + jac

                        if ti.static(static_rigid_sim_config.sparse_solve):
                            constraint_state.jac_relevant_dofs[n_con, con_n_relevant_dofs, i_b] = i_d
                            con_n_relevant_dofs += 1

                    link = links_info.parent_idx[link_maybe_batch]

            if ti.static(static_rigid_sim_config.sparse_solve):
                constraint_state.jac_n_relevant_dofs[n_con, i_b] = con_n_relevant_dofs

            pos_diff = global_anchor1 - global_anchor2
            penetration = pos_diff.norm()

            imp, aref = gu.imp_aref(sol_params, -penetration, jac_qvel, pos_diff[i_3])

            diag = ti.max(invweight * (1 - imp) / imp, gs.EPS)

            constraint_state.diag[n_con, i_b] = diag
            constraint_state.aref[n_con, i_b] = aref
            constraint_state.efc_D[n_con, i_b] = 1 / diag

    @ti.func
    def _func_equality_joint(
        self_unused,
        i_b,
        i_e,
        joints_info,
        dofs_state,
        dofs_info,
        equalities_info,
        constraint_state,
        rigid_global_info,
        static_rigid_sim_config: ti.template(),
    ):
        n_dofs = constraint_state.jac.shape[1]

        rgi = rigid_global_info

        sol_params = equalities_info.sol_params[i_e, i_b]

        I_joint1 = (
            [equalities_info.eq_obj1id[i_e, i_b], i_b]
            if ti.static(static_rigid_sim_config.batch_joints_info)
            else equalities_info.eq_obj1id[i_e, i_b]
        )
        I_joint2 = (
            [equalities_info.eq_obj2id[i_e, i_b], i_b]
            if ti.static(static_rigid_sim_config.batch_joints_info)
            else equalities_info.eq_obj2id[i_e, i_b]
        )
        i_qpos1 = joints_info.q_start[I_joint1]
        i_qpos2 = joints_info.q_start[I_joint2]
        i_dof1 = joints_info.dof_start[I_joint1]
        i_dof2 = joints_info.dof_start[I_joint2]
        I_dof1 = [i_dof1, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_dof1
        I_dof2 = [i_dof2, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_dof2

        n_con = ti.atomic_add(constraint_state.n_constraints[i_b], 1)
        ti.atomic_add(constraint_state.n_constraints_equality[i_b], 1)

        if ti.static(static_rigid_sim_config.sparse_solve):
            for i_d_ in range(constraint_state.jac_n_relevant_dofs[n_con, i_b]):
                i_d = constraint_state.jac_relevant_dofs[n_con, i_d_, i_b]
                constraint_state.jac[n_con, i_d, i_b] = gs.ti_float(0.0)
        else:
            for i_d in range(n_dofs):
                constraint_state.jac[n_con, i_d, i_b] = gs.ti_float(0.0)

        pos1 = rgi.qpos[i_qpos1, i_b]
        pos2 = rgi.qpos[i_qpos2, i_b]
        ref1 = rgi.qpos0[i_qpos1, i_b]
        ref2 = rgi.qpos0[i_qpos2, i_b]

        # TODO: zero objid2
        diff = pos2 - ref2
        pos = pos1 - ref1
        deriv = gs.ti_float(0.0)

        # y - y0 = a0 + a1 * (x-x0) + a2 * (x-x0)^2 + a3 * (x-fx0)^3 + a4 * (x-x0)^4
        for i_5 in range(5):
            diff_power = diff**i_5
            pos = pos - diff_power * equalities_info.eq_data[i_e, i_b][i_5]
            if i_5 < 4:
                deriv = deriv + equalities_info.eq_data[i_e, i_b][i_5 + 1] * diff_power * (i_5 + 1)

        constraint_state.jac[n_con, i_dof1, i_b] = gs.ti_float(1.0)
        constraint_state.jac[n_con, i_dof2, i_b] = -deriv
        jac_qvel = (
            constraint_state.jac[n_con, i_dof1, i_b] * dofs_state.vel[i_dof1, i_b]
            + constraint_state.jac[n_con, i_dof2, i_b] * dofs_state.vel[i_dof2, i_b]
        )
        invweight = dofs_info.invweight[I_dof1] + dofs_info.invweight[I_dof2]

        imp, aref = gu.imp_aref(sol_params, -ti.abs(pos), jac_qvel, pos)

        diag = ti.max(invweight * (1 - imp) / imp, gs.EPS)

        constraint_state.diag[n_con, i_b] = diag
        constraint_state.aref[n_con, i_b] = aref
        constraint_state.efc_D[n_con, i_b] = 1 / diag

    @ti.kernel
    def add_equality_constraints(
        self_unused,
        links_info: array_class.LinksInfo,
        links_state: array_class.LinksState,
        dofs_state: array_class.DofsState,
        dofs_info: array_class.DofsInfo,
        joints_info: array_class.JointsInfo,
        equalities_info: array_class.EqualitiesInfo,
        constraint_state: ti.template(),
        collider_state: ti.template(),
        rigid_global_info: ti.template(),
        static_rigid_sim_config: ti.template(),
    ):
        _B = dofs_state.ctrl_mode.shape[1]
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_b in range(_B):
            for i_e in range(constraint_state.ti_n_equalities[i_b]):
                if equalities_info.eq_type[i_e, i_b] == gs.EQUALITY_TYPE.CONNECT:
                    self_unused._func_equality_connect(
                        i_b,
                        i_e,
                        links_info=links_info,
                        links_state=links_state,
                        dofs_state=dofs_state,
                        equalities_info=equalities_info,
                        constraint_state=constraint_state,
                        collider_state=collider_state,
                        static_rigid_sim_config=static_rigid_sim_config,
                    )

                elif equalities_info.eq_type[i_e, i_b] == gs.EQUALITY_TYPE.WELD:
                    self_unused._func_equality_weld(
                        i_b,
                        i_e,
                        links_info=links_info,
                        links_state=links_state,
                        dofs_state=dofs_state,
                        equalities_info=equalities_info,
                        constraint_state=constraint_state,
                        static_rigid_sim_config=static_rigid_sim_config,
                    )
                elif equalities_info.eq_type[i_e, i_b] == gs.EQUALITY_TYPE.JOINT:
                    self_unused._func_equality_joint(
                        i_b,
                        i_e,
                        joints_info=joints_info,
                        dofs_state=dofs_state,
                        dofs_info=dofs_info,
                        equalities_info=equalities_info,
                        constraint_state=constraint_state,
                        rigid_global_info=rigid_global_info,
                        static_rigid_sim_config=static_rigid_sim_config,
                    )

    @ti.func
    def _func_equality_weld(
        self_unused,
        i_b,
        i_e,
        links_info,
        links_state,
        dofs_state,
        equalities_info,
        constraint_state,
        static_rigid_sim_config: ti.template(),
    ):
        n_dofs = dofs_state.ctrl_mode.shape[0]

        # TODO: sparse mode
        # Get equality info for this constraint
        link1_idx = equalities_info.eq_obj1id[i_e, i_b]
        link2_idx = equalities_info.eq_obj2id[i_e, i_b]
        link_a_maybe_batch = [link1_idx, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else link1_idx
        link_b_maybe_batch = [link2_idx, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else link2_idx

        # For weld, eq_data layout:
        # [0:3]  : anchor2 (local pos in body2)
        # [3:6]  : anchor1 (local pos in body1)
        # [6:10] : relative pose (quat) of body 2 related to body 1 to match orientations
        # [10]   : torquescale
        anchor1_pos = gs.ti_vec3(
            [
                equalities_info.eq_data[i_e, i_b][3],
                equalities_info.eq_data[i_e, i_b][4],
                equalities_info.eq_data[i_e, i_b][5],
            ]
        )
        anchor2_pos = gs.ti_vec3(
            [
                equalities_info.eq_data[i_e, i_b][0],
                equalities_info.eq_data[i_e, i_b][1],
                equalities_info.eq_data[i_e, i_b][2],
            ]
        )
        relpose = gs.ti_vec4(
            [
                equalities_info.eq_data[i_e, i_b][6],
                equalities_info.eq_data[i_e, i_b][7],
                equalities_info.eq_data[i_e, i_b][8],
                equalities_info.eq_data[i_e, i_b][9],
            ]
        )
        torquescale = equalities_info.eq_data[i_e, i_b][10]
        sol_params = equalities_info.sol_params[i_e, i_b]

        # Transform anchor positions to global coordinates
        global_anchor1 = gu.ti_transform_by_trans_quat(
            pos=anchor1_pos,
            trans=links_state.pos[link1_idx, i_b],
            quat=links_state.quat[link1_idx, i_b],
        )
        global_anchor2 = gu.ti_transform_by_trans_quat(
            pos=anchor2_pos,
            trans=links_state.pos[link2_idx, i_b],
            quat=links_state.quat[link2_idx, i_b],
        )

        pos_error = global_anchor1 - global_anchor2

        # Compute orientation error.
        # For weld: compute q = body1_quat * relpose, then error = (inv(body2_quat) * q)
        quat_body1 = links_state.quat[link1_idx, i_b]
        quat_body2 = links_state.quat[link2_idx, i_b]
        q = gu.ti_quat_mul(quat_body1, relpose)
        inv_quat_body2 = gu.ti_inv_quat(quat_body2)
        error_quat = gu.ti_quat_mul(inv_quat_body2, q)
        # Take the vector (axis) part and scale by torquescale.
        rot_error = gs.ti_vec3([error_quat[1], error_quat[2], error_quat[3]]) * torquescale

        all_error = gs.ti_vec6([pos_error[0], pos_error[1], pos_error[2], rot_error[0], rot_error[1], rot_error[2]])
        pos_imp = all_error.norm()

        # Compute inverse weight from both bodies.
        invweight = links_info.invweight[link_a_maybe_batch] + links_info.invweight[link_b_maybe_batch]

        # --- Position part (first 3 constraints) ---
        for i in range(3):
            n_con = ti.atomic_add(constraint_state.n_constraints[i_b], 1)
            ti.atomic_add(constraint_state.n_constraints_equality[i_b], 1)
            con_n_relevant_dofs = 0

            if ti.static(static_rigid_sim_config.sparse_solve):
                for i_d_ in range(constraint_state.jac_n_relevant_dofs[n_con, i_b]):
                    i_d = constraint_state.jac_relevant_dofs[n_con, i_d_, i_b]
                    constraint_state.jac[n_con, i_d, i_b] = gs.ti_float(0.0)
            else:
                for i_d in range(n_dofs):
                    constraint_state.jac[n_con, i_d, i_b] = gs.ti_float(0.0)

            jac_qvel = gs.ti_float(0.0)
            for i_ab in range(2):
                sign = gs.ti_float(1.0) if i_ab == 0 else gs.ti_float(-1.0)
                link = link1_idx if i_ab == 0 else link2_idx
                pos_anchor = global_anchor1 if i_ab == 0 else global_anchor2

                # Accumulate jacobian contributions along the kinematic chain.
                # (Assuming similar structure to equality_connect.)
                while link > -1:
                    link_maybe_batch = [link, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else link

                    for i_d_ in range(links_info.n_dofs[link_maybe_batch]):
                        i_d = links_info.dof_end[link_maybe_batch] - 1 - i_d_
                        cdof_ang = dofs_state.cdof_ang[i_d, i_b]
                        cdot_vel = dofs_state.cdof_vel[i_d, i_b]

                        t_quat = gu.ti_identity_quat()
                        t_pos = pos_anchor - links_state.COM[link, i_b]
                        ang, vel = gu.ti_transform_motion_by_trans_quat(cdof_ang, cdot_vel, t_pos, t_quat)
                        diff = sign * vel
                        jac = diff[i]
                        jac_qvel += jac * dofs_state.vel[i_d, i_b]
                        constraint_state.jac[n_con, i_d, i_b] += jac

                        if ti.static(static_rigid_sim_config.sparse_solve):
                            constraint_state.jac_relevant_dofs[n_con, con_n_relevant_dofs, i_b] = i_d
                            con_n_relevant_dofs += 1
                    link = links_info.parent_idx[link_maybe_batch]

            if ti.static(static_rigid_sim_config.sparse_solve):
                constraint_state.jac_n_relevant_dofs[n_con, i_b] = con_n_relevant_dofs

            imp, aref = gu.imp_aref(sol_params, -pos_imp, jac_qvel, pos_error[i])
            diag = ti.max(invweight[0] * (1 - imp) / imp, gs.EPS)

            constraint_state.diag[n_con, i_b] = diag
            constraint_state.aref[n_con, i_b] = aref
            constraint_state.efc_D[n_con, i_b] = 1.0 / diag

        # --- Orientation part (next 3 constraints) ---
        n_con = ti.atomic_add(constraint_state.n_constraints[i_b], 3)
        ti.atomic_add(constraint_state.n_constraints_equality[i_b], 3)
        con_n_relevant_dofs = 0
        for i_con in range(n_con, n_con + 3):
            for i_d in range(n_dofs):
                constraint_state.jac[i_con, i_d, i_b] = gs.ti_float(0.0)

        for i_ab in range(2):
            sign = gs.ti_float(1.0) if i_ab == 0 else gs.ti_float(-1.0)
            link = link1_idx if i_ab == 0 else link2_idx
            # For rotation, we use the body’s orientation (here we use its quaternion)
            # and a suitable reference frame. (You may need a more detailed implementation.)
            while link > -1:
                link_maybe_batch = [link, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else link

                for i_d_ in range(links_info.n_dofs[link_maybe_batch]):
                    i_d = links_info.dof_end[link_maybe_batch] - 1 - i_d_
                    jac = sign * dofs_state.cdof_ang[i_d, i_b]

                    for i_con in range(n_con, n_con + 3):
                        constraint_state.jac[i_con, i_d, i_b] = (
                            constraint_state.jac[i_con, i_d, i_b] + jac[i_con - n_con]
                        )
                link = links_info.parent_idx[link_maybe_batch]

        jac_qvel = ti.Vector([0.0, 0.0, 0.0])
        for i_d in range(n_dofs):
            # quat2 = neg(q1)*(jac0-jac1)
            # quat3 = neg(q1)*(jac0-jac1)*q0*relpose
            jac_diff_r = ti.Vector(
                [
                    constraint_state.jac[n_con, i_d, i_b],
                    constraint_state.jac[n_con + 1, i_d, i_b],
                    constraint_state.jac[n_con + 2, i_d, i_b],
                ]
            )
            quat2 = gu.ti_quat_mul_axis(inv_quat_body2, jac_diff_r)
            quat3 = gu.ti_quat_mul(quat2, q)

            for i_con in range(n_con, n_con + 3):
                constraint_state.jac[i_con, i_d, i_b] = 0.5 * quat3[i_con - n_con + 1] * torquescale
                jac_qvel[i_con - n_con] = (
                    jac_qvel[i_con - n_con] + constraint_state.jac[i_con, i_d, i_b] * dofs_state.vel[i_d, i_b]
                )

        for i_con in range(n_con, n_con + 3):
            constraint_state.jac_n_relevant_dofs[i_con, i_b] = con_n_relevant_dofs

        for i_con in range(n_con, n_con + 3):
            imp, aref = gu.imp_aref(sol_params, -pos_imp, jac_qvel[i_con - n_con], rot_error[i_con - n_con])
            diag = ti.max(invweight[1] * (1.0 - imp) / imp, gs.EPS)

            constraint_state.diag[i_con, i_b] = diag
            constraint_state.aref[i_con, i_b] = aref
            constraint_state.efc_D[i_con, i_b] = 1.0 / diag

    @ti.kernel
    def add_joint_limit_constraints(
        self_unused,
        links_info: array_class.LinksInfo,
        joints_info: array_class.JointsInfo,
        dofs_info: array_class.DofsInfo,
        dofs_state: array_class.DofsState,
        rigid_global_info: ti.template(),
        constraint_state: ti.template(),
        static_rigid_sim_config: ti.template(),
    ):
        _B = constraint_state.jac.shape[2]
        n_links = links_info.root_idx.shape[0]
        n_dofs = dofs_state.ctrl_mode.shape[0]
        rgi = rigid_global_info
        # TODO: sparse mode
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_b in range(_B):
            for i_l in range(n_links):
                I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

                for i_j in range(links_info.joint_start[I_l], links_info.joint_end[I_l]):
                    I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j

                    if (
                        joints_info.type[I_j] == gs.JOINT_TYPE.REVOLUTE
                        or joints_info.type[I_j] == gs.JOINT_TYPE.PRISMATIC
                    ):
                        i_q = joints_info.q_start[I_j]
                        i_d = joints_info.dof_start[I_j]
                        I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
                        pos_delta_min = rgi.qpos[i_q, i_b] - dofs_info.limit[I_d][0]
                        pos_delta_max = dofs_info.limit[I_d][1] - rgi.qpos[i_q, i_b]
                        pos_delta = min(pos_delta_min, pos_delta_max)

                        if pos_delta < 0:
                            jac = (pos_delta_min < pos_delta_max) * 2 - 1
                            jac_qvel = jac * dofs_state.vel[i_d, i_b]
                            imp, aref = gu.imp_aref(joints_info.sol_params[I_j], pos_delta, jac_qvel, pos_delta)
                            diag = ti.max(dofs_info.invweight[I_d] * (1 - imp) / imp, gs.EPS)

                            n_con = constraint_state.n_constraints[i_b]
                            constraint_state.n_constraints[i_b] = n_con + 1
                            constraint_state.diag[n_con, i_b] = diag
                            constraint_state.aref[n_con, i_b] = aref
                            constraint_state.efc_D[n_con, i_b] = 1 / diag

                            if ti.static(static_rigid_sim_config.sparse_solve):
                                for i_d2_ in range(constraint_state.jac_n_relevant_dofs[n_con, i_b]):
                                    i_d2 = constraint_state.jac_relevant_dofs[n_con, i_d2_, i_b]
                                    constraint_state.jac[n_con, i_d2, i_b] = gs.ti_float(0.0)
                            else:
                                for i_d2 in range(n_dofs):
                                    constraint_state.jac[n_con, i_d2, i_b] = gs.ti_float(0.0)
                            constraint_state.jac[n_con, i_d, i_b] = jac

                            if ti.static(static_rigid_sim_config.sparse_solve):
                                constraint_state.jac_n_relevant_dofs[n_con, i_b] = 1
                                constraint_state.jac_relevant_dofs[n_con, 0, i_b] = i_d

    @ti.func
    def _func_nt_hessian_incremental(
        self_unused,
        i_b,
        entities_info,
        constraint_state,
        rigid_global_info,
        static_rigid_sim_config: ti.template(),
    ):
        n_dofs = constraint_state.nt_H.shape[0]
        rank = n_dofs
        updated = False

        for i_c in range(constraint_state.n_constraints[i_b]):
            if not updated:
                flag_update = -1
                # add quad
                if constraint_state.prev_active[i_c, i_b] == 0 and constraint_state.active[i_c, i_b] == 1:
                    flag_update = 1
                # sub quad
                if constraint_state.prev_active[i_c, i_b] == 1 and constraint_state.active[i_c, i_b] == 0:
                    flag_update = 0

                if ti.static(static_rigid_sim_config.sparse_solve):
                    if flag_update != -1:
                        for i_d_ in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                            i_d = constraint_state.jac_relevant_dofs[i_c, i_d_, i_b]
                            constraint_state.nt_vec[i_d, i_b] = constraint_state.jac[i_c, i_d, i_b] * ti.sqrt(
                                constraint_state.efc_D[i_c, i_b]
                            )

                        rank = n_dofs
                        for k_ in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                            k = constraint_state.jac_relevant_dofs[i_c, k_, i_b]
                            Lkk = constraint_state.nt_H[k, k, i_b]
                            tmp = Lkk * Lkk + constraint_state.nt_vec[k, i_b] * constraint_state.nt_vec[k, i_b] * (
                                flag_update * 2 - 1
                            )
                            if tmp < gs.EPS:
                                tmp = gs.EPS
                                rank = rank - 1
                            r = ti.sqrt(tmp)
                            c = r / Lkk
                            cinv = 1 / c
                            s = constraint_state.nt_vec[k, i_b] / Lkk
                            constraint_state.nt_H[k, k, i_b] = r
                            for i_ in range(k_):
                                i = constraint_state.jac_relevant_dofs[i_c, i_, i_b]  # i is strictly > k
                                constraint_state.nt_H[i, k, i_b] = (
                                    constraint_state.nt_H[i, k, i_b]
                                    + s * constraint_state.nt_vec[i, i_b] * (flag_update * 2 - 1)
                                ) * cinv

                            for i_ in range(k_):
                                i = constraint_state.jac_relevant_dofs[i_c, i_, i_b]  # i is strictly > k
                                constraint_state.nt_vec[i, i_b] = (
                                    constraint_state.nt_vec[i, i_b] * c - s * constraint_state.nt_H[i, k, i_b]
                                )

                        if rank < n_dofs:
                            self_unused._func_nt_hessian_direct(
                                i_b,
                                entities_info=entities_info,
                                constraint_state=constraint_state,
                                rigid_global_info=rigid_global_info,
                                static_rigid_sim_config=static_rigid_sim_config,
                            )
                            updated = True
                else:
                    if flag_update != -1:
                        for i_d in range(n_dofs):
                            constraint_state.nt_vec[i_d, i_b] = constraint_state.jac[i_c, i_d, i_b] * ti.sqrt(
                                constraint_state.efc_D[i_c, i_b]
                            )

                        rank = n_dofs
                        for k in range(n_dofs):
                            if ti.abs(constraint_state.nt_vec[k, i_b]) > gs.EPS:
                                Lkk = constraint_state.nt_H[k, k, i_b]
                                tmp = Lkk * Lkk + constraint_state.nt_vec[k, i_b] * constraint_state.nt_vec[k, i_b] * (
                                    flag_update * 2 - 1
                                )
                                if tmp < gs.EPS:
                                    tmp = gs.EPS
                                    rank = rank - 1
                                r = ti.sqrt(tmp)
                                c = r / Lkk
                                cinv = 1 / c
                                s = constraint_state.nt_vec[k, i_b] / Lkk
                                constraint_state.nt_H[k, k, i_b] = r
                                for i in range(k + 1, n_dofs):
                                    constraint_state.nt_H[i, k, i_b] = (
                                        constraint_state.nt_H[i, k, i_b]
                                        + s * constraint_state.nt_vec[i, i_b] * (flag_update * 2 - 1)
                                    ) * cinv

                                for i in range(k + 1, n_dofs):
                                    constraint_state.nt_vec[i, i_b] = (
                                        constraint_state.nt_vec[i, i_b] * c - s * constraint_state.nt_H[i, k, i_b]
                                    )

                        if rank < n_dofs:
                            self_unused._func_nt_hessian_direct(
                                i_b,
                                entities_info=entities_info,
                                constraint_state=constraint_state,
                                rigid_global_info=rigid_global_info,
                                static_rigid_sim_config=static_rigid_sim_config,
                            )
                            updated = True

    @ti.func
    def _func_nt_hessian_direct(
        self_unused,
        i_b,
        entities_info,
        constraint_state,
        rigid_global_info,
        static_rigid_sim_config: ti.template(),
    ):
        n_dofs = constraint_state.nt_H.shape[0]
        n_entities = entities_info.n_links.shape[0]
        # H = M + J'*D*J
        for i_d1 in range(n_dofs):
            for i_d2 in range(n_dofs):
                constraint_state.nt_H[i_d1, i_d2, i_b] = gs.ti_float(0.0)

        if ti.static(static_rigid_sim_config.sparse_solve):
            for i_c in range(constraint_state.n_constraints[i_b]):
                jac_n_relevant_dofs = constraint_state.jac_n_relevant_dofs[i_c, i_b]
                for i_d1_ in range(jac_n_relevant_dofs):
                    i_d1 = constraint_state.jac_relevant_dofs[i_c, i_d1_, i_b]
                    if ti.abs(constraint_state.jac[i_c, i_d1, i_b]) > gs.EPS:
                        for i_d2_ in range(i_d1_, jac_n_relevant_dofs):
                            i_d2 = constraint_state.jac_relevant_dofs[i_c, i_d2_, i_b]  # i_d2 is strictly <= i_d1
                            constraint_state.nt_H[i_d1, i_d2, i_b] = (
                                constraint_state.nt_H[i_d1, i_d2, i_b]
                                + constraint_state.jac[i_c, i_d2, i_b]
                                * constraint_state.jac[i_c, i_d1, i_b]
                                * constraint_state.efc_D[i_c, i_b]
                                * constraint_state.active[i_c, i_b]
                            )
        else:
            for i_c in range(constraint_state.n_constraints[i_b]):
                for i_d1 in range(n_dofs):
                    if ti.abs(constraint_state.jac[i_c, i_d1, i_b]) > gs.EPS:
                        for i_d2 in range(i_d1 + 1):
                            constraint_state.nt_H[i_d1, i_d2, i_b] = (
                                constraint_state.nt_H[i_d1, i_d2, i_b]
                                + constraint_state.jac[i_c, i_d2, i_b]
                                * constraint_state.jac[i_c, i_d1, i_b]
                                * constraint_state.efc_D[i_c, i_b]
                                * constraint_state.active[i_c, i_b]
                            )

        for i_d1 in range(n_dofs):
            for i_d2 in range(i_d1 + 1, n_dofs):
                constraint_state.nt_H[i_d1, i_d2, i_b] = constraint_state.nt_H[i_d2, i_d1, i_b]

        for i_e in range(n_entities):
            for i_d1 in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                for i_d2 in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                    constraint_state.nt_H[i_d1, i_d2, i_b] = (
                        constraint_state.nt_H[i_d1, i_d2, i_b] + rigid_global_info.mass_mat[i_d1, i_d2, i_b]
                    )
                # self.nt_ori_H[i_d1, i_d2, i_b] = self.nt_H[i_d1, i_d2, i_b]

        self_unused._func_nt_chol_factor(i_b, constraint_state)

    @ti.func
    def _func_nt_chol_factor(
        self_unused,
        i_b,
        constraint_state,
    ):
        n_dofs = constraint_state.nt_H.shape[0]
        rank = n_dofs
        for i_d in range(n_dofs):
            tmp = constraint_state.nt_H[i_d, i_d, i_b]
            for j_d in range(i_d):
                tmp = tmp - (constraint_state.nt_H[i_d, j_d, i_b] * constraint_state.nt_H[i_d, j_d, i_b])

            if tmp < gs.EPS:
                tmp = gs.EPS
                rank = rank - 1
            constraint_state.nt_H[i_d, i_d, i_b] = ti.sqrt(tmp)

            tmp = 1.0 / constraint_state.nt_H[i_d, i_d, i_b]

            for j_d in range(i_d + 1, n_dofs):
                dot = gs.ti_float(0.0)
                for k_d in range(i_d):
                    dot = dot + constraint_state.nt_H[j_d, k_d, i_b] * constraint_state.nt_H[i_d, k_d, i_b]

                constraint_state.nt_H[j_d, i_d, i_b] = (constraint_state.nt_H[j_d, i_d, i_b] - dot) * tmp

    @ti.func
    def _func_nt_chol_solve(
        self_unused,
        i_b,
        constraint_state,
    ):
        n_dofs = constraint_state.Mgrad.shape[0]
        for i_d in range(n_dofs):
            constraint_state.Mgrad[i_d, i_b] = constraint_state.grad[i_d, i_b]

        for i_d in range(n_dofs):
            for j_d in range(i_d):
                constraint_state.Mgrad[i_d, i_b] = constraint_state.Mgrad[i_d, i_b] - (
                    constraint_state.nt_H[i_d, j_d, i_b] * constraint_state.Mgrad[j_d, i_b]
                )

            constraint_state.Mgrad[i_d, i_b] = constraint_state.Mgrad[i_d, i_b] / constraint_state.nt_H[i_d, i_d, i_b]

        for i_d_ in range(n_dofs):
            i_d = n_dofs - 1 - i_d_
            for j_d in range(i_d + 1, n_dofs):
                constraint_state.Mgrad[i_d, i_b] = (
                    constraint_state.Mgrad[i_d, i_b]
                    - constraint_state.nt_H[j_d, i_d, i_b] * constraint_state.Mgrad[j_d, i_b]
                )

            constraint_state.Mgrad[i_d, i_b] = constraint_state.Mgrad[i_d, i_b] / constraint_state.nt_H[i_d, i_d, i_b]

    def reset(self, envs_idx=None):
        if envs_idx is None:
            envs_idx = self._solver._scene._envs_idx
        self._kernel_reset(envs_idx)

    @ti.kernel
    def _kernel_reset(self, envs_idx: ti.types.ndarray()):
        ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b_ in range(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            for i_d in range(self._solver.n_dofs_):
                self.qacc_ws[i_d, i_b] = 0
                for i_c in range(self.len_constraints_):
                    self.jac[i_c, i_d, i_b] = 0
            for i_c in range(self.len_constraints_):
                self.jac_n_relevant_dofs[i_c, i_b] = 0

    def handle_constraints(self):
        self.add_equality_constraints(
            links_info=self._solver.links_info,
            links_state=self._solver.links_state,
            dofs_state=self._solver.dofs_state,
            dofs_info=self._solver.dofs_info,
            joints_info=self._solver.joints_info,
            equalities_info=self._solver.equalities_info,
            constraint_state=self.constraint_state,
            collider_state=self._collider._collider_state,
            rigid_global_info=self._solver._rigid_global_info,
            static_rigid_sim_config=self._solver._static_rigid_sim_config,
        )

        if self._solver._enable_collision:
            self.add_collision_constraints(
                links_info=self._solver.links_info,
                links_state=self._solver.links_state,
                dofs_state=self._solver.dofs_state,
                constraint_state=self.constraint_state,
                collider_state=self._collider._collider_state,
                static_rigid_sim_config=self._solver._static_rigid_sim_config,
            )

        if self._solver._enable_joint_limit:
            self.add_joint_limit_constraints(
                links_info=self._solver.links_info,
                joints_info=self._solver.joints_info,
                dofs_info=self._solver.dofs_info,
                dofs_state=self._solver.dofs_state,
                rigid_global_info=self._solver._rigid_global_info,
                constraint_state=self.constraint_state,
                static_rigid_sim_config=self._solver._static_rigid_sim_config,
            )

        if self._solver._enable_collision or self._solver._enable_joint_limit or self._solver.n_equalities > 0:
            self.resolve()

    def resolve(self):
        # from genesis.utils.tools import create_timer

        # timer = create_timer(name="resolve", level=3, ti_sync=True, skip_first_call=True)
        self._func_init_solver(
            dofs_state=self._solver.dofs_state,
            entities_info=self._solver.entities_info,
            constraint_state=self.constraint_state,
            rigid_global_info=self._solver._rigid_global_info,
            static_rigid_sim_config=self._solver._static_rigid_sim_config,
        )
        # timer.stamp("_func_init_solver")
        self._func_solve(
            entities_info=self._solver.entities_info,
            dofs_state=self._solver.dofs_state,
            constraint_state=self.constraint_state,
            rigid_global_info=self._solver._rigid_global_info,
            static_rigid_sim_config=self._solver._static_rigid_sim_config,
        )
        # timer.stamp("_func_solve")
        self._func_update_qacc(
            dofs_state=self._solver.dofs_state,
            constraint_state=self.constraint_state,
            static_rigid_sim_config=self._solver._static_rigid_sim_config,
        )
        # timer.stamp("_func_update_qacc")
        self._func_update_contact_force(
            links_state=self._solver.links_state,
            collider_state=self._collider._collider_state,
            constraint_state=self.constraint_state,
            static_rigid_sim_config=self._solver._static_rigid_sim_config,
        )
        # timer.stamp("compute force")

    @ti.kernel
    def _func_update_contact_force(
        self_unused,
        links_state: array_class.LinksState,
        collider_state: ti.template(),
        constraint_state: ti.template(),
        static_rigid_sim_config: ti.template(),
    ):
        n_links = links_state.contact_force.shape[0]
        _B = links_state.contact_force.shape[1]

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_l, i_b in ti.ndrange(n_links, _B):
            links_state.contact_force[i_l, i_b] = ti.Vector.zero(gs.ti_float, 3)

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(_B):
            const_start = constraint_state.n_constraints_equality[i_b]
            for i_c in range(collider_state.n_contacts[i_b]):
                contact_data = collider_state.contact_data[i_c, i_b]

                force = ti.Vector.zero(gs.ti_float, 3)
                d1, d2 = gu.ti_orthogonals(contact_data.normal)
                for i_dir in range(4):
                    d = (2 * (i_dir % 2) - 1) * (d1 if i_dir < 2 else d2)
                    n = d * contact_data.friction - contact_data.normal
                    force += n * constraint_state.efc_force[i_c * 4 + i_dir + const_start, i_b]

                collider_state.contact_data[i_c, i_b].force = force

                links_state.contact_force[contact_data.link_a, i_b] = (
                    links_state.contact_force[contact_data.link_a, i_b] - force
                )
                links_state.contact_force[contact_data.link_b, i_b] = (
                    links_state.contact_force[contact_data.link_b, i_b] + force
                )

    @ti.kernel
    def _func_update_qacc(
        self_unused,
        dofs_state: array_class.DofsState,
        constraint_state: ti.template(),
        static_rigid_sim_config: ti.template(),
    ):
        n_dofs = dofs_state.acc.shape[0]
        _B = dofs_state.acc.shape[1]
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_d, i_b in ti.ndrange(n_dofs, _B):
            dofs_state.acc[i_d, i_b] = constraint_state.qacc[i_d, i_b]
            dofs_state.qf_constraint[i_d, i_b] = constraint_state.qfrc_constraint[i_d, i_b]
            dofs_state.force[i_d, i_b] += constraint_state.qfrc_constraint[i_d, i_b]

        for i_d, i_b in ti.ndrange(n_dofs, _B):
            self_unused.qacc_ws[i_d, i_b] = constraint_state.qacc[i_d, i_b]

    @ti.kernel
    def _func_solve(
        self_unused,
        entities_info: array_class.EntitiesInfo,
        dofs_state: array_class.DofsState,
        constraint_state: ti.template(),
        rigid_global_info: ti.template(),
        static_rigid_sim_config: ti.template(),
    ):
        _B = constraint_state.grad.shape[1]
        n_dofs = constraint_state.grad.shape[0]
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(_B):
            # this safeguard seems not necessary in normal execution
            # if self.n_constraints[i_b] > 0 or self.cost_ws[i_b] < self.cost[i_b]:
            if constraint_state.n_constraints[i_b] > 0:
                tol_scaled = (
                    rigid_global_info.meaninertia[i_b] * ti.max(1, n_dofs)
                ) * static_rigid_sim_config.tolerance
                for it in range(static_rigid_sim_config.iterations):
                    self_unused._func_solve_body(
                        i_b,
                        entities_info=entities_info,
                        dofs_state=dofs_state,
                        rigid_global_info=rigid_global_info,
                        constraint_state=constraint_state,
                        static_rigid_sim_config=static_rigid_sim_config,
                    )
                    if constraint_state.improved[i_b] < 1:
                        break

                    gradient = gs.ti_float(0.0)
                    for i_d in range(n_dofs):
                        gradient += constraint_state.grad[i_d, i_b] * constraint_state.grad[i_d, i_b]
                    gradient = ti.sqrt(gradient)
                    improvement = constraint_state.prev_cost[i_b] - constraint_state.cost[i_b]
                    if gradient < tol_scaled or improvement < tol_scaled:
                        break

    @ti.func
    def _func_ls_init(
        self_unused,
        i_b,
        entities_info,
        dofs_state,
        constraint_state,
        rigid_global_info,
        static_rigid_sim_config: ti.template(),
    ):
        rgi = rigid_global_info
        n_dofs = constraint_state.search.shape[0]
        n_entities = entities_info.dof_start.shape[0]
        # mv and jv
        for i_e in range(n_entities):
            for i_d1 in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                mv = gs.ti_float(0.0)
                for i_d2 in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                    mv += rgi.mass_mat[i_d1, i_d2, i_b] * constraint_state.search[i_d2, i_b]
                constraint_state.mv[i_d1, i_b] = mv

        for i_c in range(constraint_state.n_constraints[i_b]):
            jv = gs.ti_float(0.0)
            if ti.static(static_rigid_sim_config.sparse_solve):
                for i_d_ in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                    i_d = constraint_state.jac_relevant_dofs[i_c, i_d_, i_b]
                    jv += constraint_state.jac[i_c, i_d, i_b] * constraint_state.search[i_d, i_b]
            else:
                for i_d in range(n_dofs):
                    jv += constraint_state.jac[i_c, i_d, i_b] * constraint_state.search[i_d, i_b]
            constraint_state.jv[i_c, i_b] = jv

        # quad and quad_gauss
        quad_gauss_1 = gs.ti_float(0.0)
        quad_gauss_2 = gs.ti_float(0.0)
        for i_d in range(n_dofs):
            quad_gauss_1 += (
                constraint_state.search[i_d, i_b] * constraint_state.Ma[i_d, i_b]
                - constraint_state.search[i_d, i_b] * dofs_state.force[i_d, i_b]
            )
            quad_gauss_2 += 0.5 * constraint_state.search[i_d, i_b] * constraint_state.mv[i_d, i_b]
        for _i0 in range(1):
            constraint_state.quad_gauss[_i0 + 0, i_b] = constraint_state.gauss[i_b]
            constraint_state.quad_gauss[_i0 + 1, i_b] = quad_gauss_1
            constraint_state.quad_gauss[_i0 + 2, i_b] = quad_gauss_2

            for i_c in range(constraint_state.n_constraints[i_b]):
                constraint_state.quad[i_c, _i0 + 0, i_b] = constraint_state.efc_D[i_c, i_b] * (
                    0.5 * constraint_state.Jaref[i_c, i_b] * constraint_state.Jaref[i_c, i_b]
                )
                constraint_state.quad[i_c, _i0 + 1, i_b] = constraint_state.efc_D[i_c, i_b] * (
                    constraint_state.jv[i_c, i_b] * constraint_state.Jaref[i_c, i_b]
                )
                constraint_state.quad[i_c, _i0 + 2, i_b] = constraint_state.efc_D[i_c, i_b] * (
                    0.5 * constraint_state.jv[i_c, i_b] * constraint_state.jv[i_c, i_b]
                )

    @ti.func
    def _func_ls_point_fn(
        self_unused,
        i_b,
        alpha,
        constraint_state,
    ):
        tmp_quad_total0, tmp_quad_total1, tmp_quad_total2 = gs.ti_float(0.0), gs.ti_float(0.0), gs.ti_float(0.0)
        for _i0 in range(1):
            tmp_quad_total0 = constraint_state.quad_gauss[_i0 + 0, i_b]
            tmp_quad_total1 = constraint_state.quad_gauss[_i0 + 1, i_b]
            tmp_quad_total2 = constraint_state.quad_gauss[_i0 + 2, i_b]
            for i_c in range(constraint_state.n_constraints[i_b]):
                active = 1
                if i_c >= constraint_state.n_constraints_equality[i_b]:
                    active = constraint_state.Jaref[i_c, i_b] + alpha * constraint_state.jv[i_c, i_b] < 0
                tmp_quad_total0 += constraint_state.quad[i_c, _i0 + 0, i_b] * active
                tmp_quad_total1 += constraint_state.quad[i_c, _i0 + 1, i_b] * active
                tmp_quad_total2 += constraint_state.quad[i_c, _i0 + 2, i_b] * active

        cost = alpha * alpha * tmp_quad_total2 + alpha * tmp_quad_total1 + tmp_quad_total0

        deriv_0 = 2 * alpha * tmp_quad_total2 + tmp_quad_total1
        deriv_1 = 2 * tmp_quad_total2 + gs.EPS * (ti.abs(tmp_quad_total2) < gs.EPS)

        constraint_state.ls_its[i_b] = constraint_state.ls_its[i_b] + 1

        return alpha, cost, deriv_0, deriv_1

    @ti.func
    def _func_no_linesearch(self, i_b):
        self._func_ls_init(i_b)

        self.improved[i_b] = 1
        for i_d in range(self._solver.n_dofs):
            self.qacc[i_d, i_b] = self.qacc[i_d, i_b] + self.search[i_d, i_b]
            self.Ma[i_d, i_b] = self.Ma[i_d, i_b] + self.mv[i_d, i_b]
        for i_c in range(self.n_constraints[i_b]):
            self.Jaref[i_c, i_b] = self.Jaref[i_c, i_b] + self.jv[i_c, i_b]

    @ti.func
    def _func_linesearch(
        self_unused,
        i_b,
        entities_info,
        dofs_state,
        rigid_global_info,
        constraint_state,
        static_rigid_sim_config: ti.template(),
    ):
        n_dofs = constraint_state.search.shape[0]
        ## use adaptive linesearch tolerance
        snorm = gs.ti_float(0.0)
        for jd in range(n_dofs):
            snorm += constraint_state.search[jd, i_b] ** 2
        snorm = ti.sqrt(snorm)
        scale = 1.0 / (rigid_global_info.meaninertia[i_b] * ti.max(1, n_dofs))
        gtol = static_rigid_sim_config.tolerance * static_rigid_sim_config.ls_tolerance * snorm / scale
        slopescl = scale / snorm
        constraint_state.gtol[i_b] = gtol

        constraint_state.ls_its[i_b] = 0
        constraint_state.ls_result[i_b] = 0
        ls_slope = gs.ti_float(1.0)

        res_alpha = gs.ti_float(0.0)
        done = False

        if snorm < gs.EPS:
            constraint_state.ls_result[i_b] = 1
            res_alpha = 0.0
        else:
            self_unused._func_ls_init(
                i_b,
                entities_info=entities_info,
                dofs_state=dofs_state,
                constraint_state=constraint_state,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
            )

            p0_alpha, p0_cost, p0_deriv_0, p0_deriv_1 = self_unused._func_ls_point_fn(
                i_b, gs.ti_float(0.0), constraint_state
            )
            p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = self_unused._func_ls_point_fn(
                i_b, p0_alpha - p0_deriv_0 / p0_deriv_1, constraint_state
            )

            if p0_cost < p1_cost:
                p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = p0_alpha, p0_cost, p0_deriv_0, p0_deriv_1

            if ti.abs(p1_deriv_0) < gtol:
                if ti.abs(p1_alpha) < gs.EPS:
                    constraint_state.ls_result[i_b] = 2
                else:
                    constraint_state.ls_result[i_b] = 0
                ls_slope = ti.abs(p1_deriv_0) * slopescl
                res_alpha = p1_alpha
            else:
                direction = (p1_deriv_0 < 0) * 2 - 1
                p2update = 0
                p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1 = p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1
                while (
                    p1_deriv_0 * direction <= -gtol
                    and constraint_state.ls_its[i_b] < static_rigid_sim_config.ls_iterations
                ):
                    p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1 = p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1
                    p2update = 1

                    p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = self_unused._func_ls_point_fn(
                        i_b, p1_alpha - p1_deriv_0 / p1_deriv_1, constraint_state
                    )
                    if ti.abs(p1_deriv_0) < gtol:
                        ls_slope = ti.abs(p1_deriv_0) * slopescl
                        res_alpha = p1_alpha
                        done = True
                        break
                if not done:
                    if constraint_state.ls_its[i_b] >= static_rigid_sim_config.ls_iterations:
                        constraint_state.ls_result[i_b] = 3
                        ls_slope = ti.abs(p1_deriv_0) * slopescl
                        res_alpha = p1_alpha
                        done = True

                    if not p2update and not done:
                        constraint_state.ls_result[i_b] = 6
                        ls_slope = ti.abs(p1_deriv_0) * slopescl
                        res_alpha = p1_alpha
                        done = True

                    if not done:
                        p2_next_alpha, p2_next_cost, p2_next_deriv_0, p2_next_deriv_1 = (
                            p1_alpha,
                            p1_cost,
                            p1_deriv_0,
                            p1_deriv_1,
                        )

                        p1_next_alpha, p1_next_cost, p1_next_deriv_0, p1_next_deriv_1 = self_unused._func_ls_point_fn(
                            i_b, p1_alpha - p1_deriv_0 / p1_deriv_1, constraint_state
                        )

                        while constraint_state.ls_its[i_b] < static_rigid_sim_config.ls_iterations:
                            pmid_alpha, pmid_cost, pmid_deriv_0, pmid_deriv_1 = self_unused._func_ls_point_fn(
                                i_b, (p1_alpha + p2_alpha) * 0.5, constraint_state
                            )

                            i = 0
                            (
                                constraint_state.candidates[4 * i + 0, i_b],
                                constraint_state.candidates[4 * i + 1, i_b],
                                constraint_state.candidates[4 * i + 2, i_b],
                                constraint_state.candidates[4 * i + 3, i_b],
                            ) = (p1_next_alpha, p1_next_cost, p1_next_deriv_0, p1_next_deriv_1)
                            i = 1
                            (
                                constraint_state.candidates[4 * i + 0, i_b],
                                constraint_state.candidates[4 * i + 1, i_b],
                                constraint_state.candidates[4 * i + 2, i_b],
                                constraint_state.candidates[4 * i + 3, i_b],
                            ) = (p2_next_alpha, p2_next_cost, p2_next_deriv_0, p2_next_deriv_1)
                            i = 2
                            (
                                constraint_state.candidates[4 * i + 0, i_b],
                                constraint_state.candidates[4 * i + 1, i_b],
                                constraint_state.candidates[4 * i + 2, i_b],
                                constraint_state.candidates[4 * i + 3, i_b],
                            ) = (pmid_alpha, pmid_cost, pmid_deriv_0, pmid_deriv_1)

                            best_i = -1
                            best_cost = gs.ti_float(0.0)
                            for ii in range(3):
                                if ti.abs(constraint_state.candidates[4 * ii + 2, i_b]) < gtol and (
                                    best_i < 0 or constraint_state.candidates[4 * ii + 1, i_b] < best_cost
                                ):
                                    best_cost = constraint_state.candidates[4 * ii + 1, i_b]
                                    best_i = ii
                            if best_i >= 0:
                                ls_slope = ti.abs(constraint_state.candidates[4 * i + 2, i_b]) * slopescl
                                res_alpha = constraint_state.candidates[4 * best_i + 0, i_b]
                                done = True
                            else:
                                (
                                    b1,
                                    p1_alpha,
                                    p1_cost,
                                    p1_deriv_0,
                                    p1_deriv_1,
                                    p1_next_alpha,
                                    p1_next_cost,
                                    p1_next_deriv_0,
                                    p1_next_deriv_1,
                                ) = self_unused.update_bracket(
                                    p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1, i_b, constraint_state
                                )
                                (
                                    b2,
                                    p2_alpha,
                                    p2_cost,
                                    p2_deriv_0,
                                    p2_deriv_1,
                                    p2_next_alpha,
                                    p2_next_cost,
                                    p2_next_deriv_0,
                                    p2_next_deriv_1,
                                ) = self_unused.update_bracket(
                                    p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1, i_b, constraint_state
                                )

                                if b1 == 0 and b2 == 0:
                                    if pmid_cost < p0_cost:
                                        constraint_state.ls_result[i_b] = 0
                                    else:
                                        constraint_state.ls_result[i_b] = 7

                                    ls_slope = ti.abs(pmid_deriv_0) * slopescl

                                    res_alpha = pmid_alpha
                                    done = True

                        if not done:
                            if p1_cost <= p2_cost and p1_cost < p0_cost:
                                constraint_state.ls_result[i_b] = 4
                                ls_slope = ti.abs(p1_deriv_0) * slopescl
                                res_alpha = p1_alpha
                            elif p2_cost <= p1_cost and p2_cost < p1_cost:
                                constraint_state.ls_result[i_b] = 4
                                ls_slope = ti.abs(p2_deriv_0) * slopescl
                                res_alpha = p2_alpha
                            else:
                                constraint_state.ls_result[i_b] = 5
                                res_alpha = 0.0
        return res_alpha

    @ti.func
    def update_bracket(
        self_unused,
        p_alpha,
        p_cost,
        p_deriv_0,
        p_deriv_1,
        i_b,
        constraint_state,
    ):
        flag = 0

        for i in range(3):
            if (
                p_deriv_0 < 0
                and constraint_state.candidates[4 * i + 2, i_b] < 0
                and p_deriv_0 < constraint_state.candidates[4 * i + 2, i_b]
            ):
                p_alpha, p_cost, p_deriv_0, p_deriv_1 = (
                    constraint_state.candidates[4 * i + 0, i_b],
                    constraint_state.candidates[4 * i + 1, i_b],
                    constraint_state.candidates[4 * i + 2, i_b],
                    constraint_state.candidates[4 * i + 3, i_b],
                )

                flag = 1

            elif (
                p_deriv_0 > 0
                and constraint_state.candidates[4 * i + 2, i_b] > 0
                and p_deriv_0 > constraint_state.candidates[4 * i + 2, i_b]
            ):
                p_alpha, p_cost, p_deriv_0, p_deriv_1 = (
                    constraint_state.candidates[4 * i + 0, i_b],
                    constraint_state.candidates[4 * i + 1, i_b],
                    constraint_state.candidates[4 * i + 2, i_b],
                    constraint_state.candidates[4 * i + 3, i_b],
                )
                flag = 2
            else:
                pass

        p_next_alpha, p_next_cost, p_next_deriv_0, p_next_deriv_1 = p_alpha, p_cost, p_deriv_0, p_deriv_1

        if flag > 0:
            p_next_alpha, p_next_cost, p_next_deriv_0, p_next_deriv_1 = self_unused._func_ls_point_fn(
                i_b, p_alpha - p_deriv_0 / p_deriv_1, constraint_state
            )
        return flag, p_alpha, p_cost, p_deriv_0, p_deriv_1, p_next_alpha, p_next_cost, p_next_deriv_0, p_next_deriv_1

    @ti.func
    def _func_solve_body(
        self_unused,
        i_b,
        entities_info,
        dofs_state,
        rigid_global_info,
        constraint_state,
        static_rigid_sim_config: ti.template(),
    ):
        n_dofs = constraint_state.qacc.shape[0]
        alpha = self_unused._func_linesearch(
            i_b,
            entities_info=entities_info,
            dofs_state=dofs_state,
            rigid_global_info=rigid_global_info,
            constraint_state=constraint_state,
            static_rigid_sim_config=static_rigid_sim_config,
        )

        if ti.abs(alpha) < gs.EPS:
            constraint_state.improved[i_b] = 0
        else:
            constraint_state.improved[i_b] = 1
            for i_d in range(n_dofs):
                constraint_state.qacc[i_d, i_b] = (
                    constraint_state.qacc[i_d, i_b] + constraint_state.search[i_d, i_b] * alpha
                )
                constraint_state.Ma[i_d, i_b] = constraint_state.Ma[i_d, i_b] + constraint_state.mv[i_d, i_b] * alpha

            for i_c in range(constraint_state.n_constraints[i_b]):
                constraint_state.Jaref[i_c, i_b] = (
                    constraint_state.Jaref[i_c, i_b] + constraint_state.jv[i_c, i_b] * alpha
                )

            if ti.static(static_rigid_sim_config.solver_type == gs.constraint_solver.CG):
                for i_d in range(n_dofs):
                    constraint_state.cg_prev_grad[i_d, i_b] = constraint_state.grad[i_d, i_b]
                    constraint_state.cg_prev_Mgrad[i_d, i_b] = constraint_state.Mgrad[i_d, i_b]

            self_unused._func_update_constraint(
                i_b,
                qacc=constraint_state.qacc,
                Ma=constraint_state.Ma,
                cost=constraint_state.cost,
                dofs_state=dofs_state,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )

            if ti.static(static_rigid_sim_config.solver_type == gs.constraint_solver.CG):
                self_unused._func_update_gradient(
                    i_b,
                    dofs_state=dofs_state,
                    entities_info=entities_info,
                    rigid_global_info=rigid_global_info,
                    constraint_state=constraint_state,
                    static_rigid_sim_config=static_rigid_sim_config,
                )

                constraint_state.cg_beta[i_b] = gs.ti_float(0.0)
                constraint_state.cg_pg_dot_pMg[i_b] = gs.ti_float(0.0)

                for i_d in range(n_dofs):
                    constraint_state.cg_beta[i_b] += constraint_state.grad[i_d, i_b] * (
                        constraint_state.Mgrad[i_d, i_b] - constraint_state.cg_prev_Mgrad[i_d, i_b]
                    )
                    constraint_state.cg_pg_dot_pMg[i_b] += (
                        constraint_state.cg_prev_Mgrad[i_d, i_b] * constraint_state.cg_prev_grad[i_d, i_b]
                    )

                constraint_state.cg_beta[i_b] = ti.max(
                    0.0, constraint_state.cg_beta[i_b] / ti.max(gs.EPS, constraint_state.cg_pg_dot_pMg[i_b])
                )
                for i_d in range(n_dofs):
                    constraint_state.search[i_d, i_b] = (
                        -constraint_state.Mgrad[i_d, i_b]
                        + constraint_state.cg_beta[i_b] * constraint_state.search[i_d, i_b]
                    )

            elif ti.static(static_rigid_sim_config.solver_type == gs.constraint_solver.Newton):
                improvement = constraint_state.prev_cost[i_b] - constraint_state.cost[i_b]
                if improvement > 0:
                    self_unused._func_nt_hessian_incremental(
                        i_b,
                        entities_info=entities_info,
                        constraint_state=constraint_state,
                        rigid_global_info=rigid_global_info,
                        static_rigid_sim_config=static_rigid_sim_config,
                    )
                    self_unused._func_update_gradient(
                        i_b,
                        dofs_state=dofs_state,
                        entities_info=entities_info,
                        rigid_global_info=rigid_global_info,
                        constraint_state=constraint_state,
                        static_rigid_sim_config=static_rigid_sim_config,
                    )
                    for i_d in range(n_dofs):
                        constraint_state.search[i_d, i_b] = -constraint_state.Mgrad[i_d, i_b]

    @ti.func
    def _func_update_constraint(
        self_unused,
        i_b,
        qacc,
        Ma,
        cost,
        dofs_state,
        constraint_state,
        static_rigid_sim_config: ti.template(),
    ):
        n_dofs = constraint_state.qfrc_constraint.shape[0]

        constraint_state.prev_cost[i_b] = cost[i_b]
        cost[i_b] = gs.ti_float(0.0)
        constraint_state.gauss[i_b] = gs.ti_float(0.0)

        for i_c in range(constraint_state.n_constraints[i_b]):
            if ti.static(static_rigid_sim_config.solver_type == gs.constraint_solver.Newton):
                constraint_state.prev_active[i_c, i_b] = constraint_state.active[i_c, i_b]
            constraint_state.active[i_c, i_b] = 1
            if i_c >= constraint_state.n_constraints_equality[i_b]:
                constraint_state.active[i_c, i_b] = constraint_state.Jaref[i_c, i_b] < 0
            constraint_state.efc_force[i_c, i_b] = (
                -constraint_state.efc_D[i_c, i_b] * constraint_state.Jaref[i_c, i_b] * constraint_state.active[i_c, i_b]
            )

        if ti.static(static_rigid_sim_config.sparse_solve):
            for i_d in range(n_dofs):
                constraint_state.qfrc_constraint[i_d, i_b] = gs.ti_float(0.0)
            for i_c in range(constraint_state.n_constraints[i_b]):
                for i_d_ in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                    i_d = constraint_state.jac_relevant_dofs[i_c, i_d_, i_b]
                    constraint_state.qfrc_constraint[i_d, i_b] = (
                        constraint_state.qfrc_constraint[i_d, i_b]
                        + constraint_state.jac[i_c, i_d, i_b] * constraint_state.efc_force[i_c, i_b]
                    )
        else:
            for i_d in range(n_dofs):
                qfrc_constraint = gs.ti_float(0.0)
                for i_c in range(constraint_state.n_constraints[i_b]):
                    qfrc_constraint += constraint_state.jac[i_c, i_d, i_b] * constraint_state.efc_force[i_c, i_b]
                constraint_state.qfrc_constraint[i_d, i_b] = qfrc_constraint
        # (Mx - Mx') * (x - x')
        for i_d in range(n_dofs):
            v = 0.5 * (Ma[i_d, i_b] - dofs_state.force[i_d, i_b]) * (qacc[i_d, i_b] - dofs_state.acc_smooth[i_d, i_b])
            constraint_state.gauss[i_b] = constraint_state.gauss[i_b] + v
            cost[i_b] = cost[i_b] + v

        # D * (Jx - aref) ** 2
        for i_c in range(constraint_state.n_constraints[i_b]):
            cost[i_b] = cost[i_b] + 0.5 * (
                constraint_state.efc_D[i_c, i_b]
                * constraint_state.Jaref[i_c, i_b]
                * constraint_state.Jaref[i_c, i_b]
                * constraint_state.active[i_c, i_b]
            )

    @ti.func
    def _func_update_gradient(
        self_unused,
        i_b,
        dofs_state,
        entities_info,
        rigid_global_info,
        constraint_state,
        static_rigid_sim_config: ti.template(),
    ):
        n_dofs = constraint_state.grad.shape[0]

        for i_d in range(n_dofs):
            constraint_state.grad[i_d, i_b] = (
                constraint_state.Ma[i_d, i_b] - dofs_state.force[i_d, i_b] - constraint_state.qfrc_constraint[i_d, i_b]
            )

        if ti.static(static_rigid_sim_config.solver_type == gs.constraint_solver.CG):
            self_unused._solver._func_solve_mass_batched(
                constraint_state.grad,
                constraint_state.Mgrad,
                i_b,
                entities_info=entities_info,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
            )

        elif ti.static(static_rigid_sim_config.solver_type == gs.constraint_solver.Newton):
            self_unused._func_nt_chol_solve(
                i_b,
                constraint_state=constraint_state,
            )

    @ti.func
    def initialize_Jaref(
        self_unused,
        qacc,
        constraint_state,
        static_rigid_sim_config: ti.template(),
    ):
        _B = constraint_state.jac.shape[2]
        n_dofs = constraint_state.jac.shape[1]
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(_B):
            for i_c in range(constraint_state.n_constraints[i_b]):
                Jaref = -constraint_state.aref[i_c, i_b]
                if ti.static(static_rigid_sim_config.sparse_solve):
                    for i_d_ in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                        i_d = constraint_state.jac_relevant_dofs[i_c, i_d_, i_b]
                        Jaref += constraint_state.jac[i_c, i_d, i_b] * qacc[i_d, i_b]
                else:
                    for i_d in range(n_dofs):
                        Jaref += constraint_state.jac[i_c, i_d, i_b] * qacc[i_d, i_b]
                constraint_state.Jaref[i_c, i_b] = Jaref

    @ti.func
    def initialize_Ma(
        self_unused,
        Ma,
        qacc,
        entities_info,
        rigid_global_info,
        static_rigid_sim_config: ti.template(),
    ):
        rgi = rigid_global_info
        _B = rgi.mass_mat.shape[2]
        n_entities = entities_info.n_links.shape[0]
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_e, i_b in ti.ndrange(n_entities, _B):
            for i_d1_ in range(entities_info.n_dofs[i_e]):
                i_d1 = entities_info.dof_start[i_e] + i_d1_
                Ma_ = gs.ti_float(0.0)
                for i_d2 in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                    Ma_ += rgi.mass_mat[i_d1, i_d2, i_b] * qacc[i_d2, i_b]
                Ma[i_d1, i_b] = Ma_

    @ti.kernel
    def _func_init_solver(
        self_unused,
        dofs_state: array_class.DofsState,
        entities_info: array_class.EntitiesInfo,
        constraint_state: ti.template(),
        rigid_global_info: ti.template(),
        static_rigid_sim_config: ti.template(),
    ):
        _B = dofs_state.acc_smooth.shape[1]
        n_dofs = dofs_state.acc_smooth.shape[0]
        # check if warm start
        self_unused.initialize_Jaref(
            qacc=constraint_state.qacc_ws,
            constraint_state=constraint_state,
            static_rigid_sim_config=static_rigid_sim_config,
        )

        self_unused.initialize_Ma(
            Ma=constraint_state.Ma_ws,
            qacc=constraint_state.qacc_ws,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(_B):
            self_unused._func_update_constraint(
                i_b,
                qacc=constraint_state.qacc_ws,
                Ma=constraint_state.Ma_ws,
                cost=constraint_state.cost_ws,
                dofs_state=dofs_state,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )
        self_unused.initialize_Jaref(
            qacc=dofs_state.acc_smooth,
            constraint_state=constraint_state,
            static_rigid_sim_config=static_rigid_sim_config,
        )

        self_unused.initialize_Ma(
            Ma=constraint_state.Ma,
            qacc=dofs_state.acc_smooth,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(_B):
            self_unused._func_update_constraint(
                i_b,
                qacc=dofs_state.acc_smooth,
                Ma=constraint_state.Ma,
                cost=constraint_state.cost,
                dofs_state=dofs_state,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_d, i_b in ti.ndrange(n_dofs, _B):
            if constraint_state.cost_ws[i_b] < constraint_state.cost[i_b]:
                constraint_state.qacc[i_d, i_b] = constraint_state.qacc_ws[i_d, i_b]
                constraint_state.Ma[i_d, i_b] = constraint_state.Ma_ws[i_d, i_b]
            else:
                constraint_state.qacc[i_d, i_b] = dofs_state.acc_smooth[i_d, i_b]
        self_unused.initialize_Jaref(
            qacc=constraint_state.qacc,
            constraint_state=constraint_state,
            static_rigid_sim_config=static_rigid_sim_config,
        )
        # end warm start

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(_B):
            self_unused._func_update_constraint(
                i_b,
                qacc=constraint_state.qacc,
                Ma=constraint_state.Ma,
                cost=constraint_state.cost,
                dofs_state=dofs_state,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )
            if ti.static(static_rigid_sim_config.solver_type == gs.constraint_solver.Newton):
                self_unused._func_nt_hessian_direct(
                    i_b,
                    entities_info=entities_info,
                    rigid_global_info=rigid_global_info,
                    constraint_state=constraint_state,
                    static_rigid_sim_config=static_rigid_sim_config,
                )

            self_unused._func_update_gradient(
                i_b,
                dofs_state=dofs_state,
                entities_info=entities_info,
                rigid_global_info=rigid_global_info,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_d, i_b in ti.ndrange(n_dofs, _B):
            constraint_state.search[i_d, i_b] = -constraint_state.Mgrad[i_d, i_b]
