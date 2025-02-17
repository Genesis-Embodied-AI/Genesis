import numpy as np
import taichi as ti

import genesis as gs
import genesis.utils.geom as gu


@ti.data_oriented
class ConstraintSolver:
    def __init__(self, rigid_solver):
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
            5 * self._collider._max_contact_pairs
            + np.logical_not(np.isinf(self._solver.dofs_info.limit.to_numpy()[:, 0])).sum()
            + self._solver.n_equalities * 3
        )
        self.len_constraints_ = max(1, self.len_constraints)

        self.jac = ti.field(
            dtype=gs.ti_float, shape=self._solver._batch_shape((self.len_constraints_, self._solver.n_dofs_))
        )
        self.diag = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self.len_constraints_))
        self.aref = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self.len_constraints_))

        if self.sparse_solve:
            self.jac_relevant_dofs = ti.field(
                gs.ti_int, shape=self._solver._batch_shape((self.len_constraints_, self._solver.n_dofs_))
            )
            self.jac_n_relevant_dofs = ti.field(gs.ti_int, shape=self._solver._batch_shape(self.len_constraints_))

        self.n_constraints = ti.field(gs.ti_int, shape=self._solver._batch_shape())
        self.n_constraints_equality = ti.field(gs.ti_int, shape=self._solver._batch_shape())
        self.improved = ti.field(gs.ti_int, shape=self._solver._batch_shape())

        self.Jaref = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self.len_constraints_))
        self.Ma = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self._solver.n_dofs_))
        self.Ma_ws = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self._solver.n_dofs_))
        self.grad = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self._solver.n_dofs_))
        self.Mgrad = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self._solver.n_dofs_))
        self.search = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self._solver.n_dofs_))

        self.efc_D = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self.len_constraints_))
        self.efc_force = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self.len_constraints_))
        self.active = ti.field(dtype=gs.ti_int, shape=self._solver._batch_shape(self.len_constraints_))
        self.prev_active = ti.field(dtype=gs.ti_int, shape=self._solver._batch_shape(self.len_constraints_))
        self.qfrc_constraint = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self._solver.n_dofs_))
        self.qacc = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self._solver.n_dofs_))
        self.qacc_ws = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self._solver.n_dofs_))
        self.qacc_prev = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self._solver.n_dofs_))

        self.cost_ws = ti.field(gs.ti_float, shape=self._solver._batch_shape())

        self.gauss = ti.field(gs.ti_float, shape=self._solver._batch_shape())
        self.cost = ti.field(gs.ti_float, shape=self._solver._batch_shape())
        self.prev_cost = ti.field(gs.ti_float, shape=self._solver._batch_shape())

        ## line search
        self.gtol = ti.field(gs.ti_float, shape=self._solver._batch_shape())
        self.meaninertia = ti.field(gs.ti_float, shape=self._solver._batch_shape())

        self.mv = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self._solver.n_dofs_))
        self.jv = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self.len_constraints_))
        self.quad_gauss = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(3))
        self.quad = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape((self.len_constraints_, 3)))

        self.candidates = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(12))
        self.ls_its = ti.field(gs.ti_float, shape=self._solver._batch_shape())
        self.ls_result = ti.field(gs.ti_int, shape=self._solver._batch_shape())

        if self._solver_type == gs.constraint_solver.CG:
            self.cg_prev_grad = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self._solver.n_dofs_))
            self.cg_prev_Mgrad = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self._solver.n_dofs_))
            self.cg_beta = ti.field(gs.ti_float, shape=self._solver._batch_shape())
            self.cg_pg_dot_pMg = ti.field(gs.ti_float, shape=self._solver._batch_shape())

        if self._solver_type == gs.constraint_solver.Newton:
            self.nt_H = ti.field(
                dtype=gs.ti_float, shape=self._solver._batch_shape((self._solver.n_dofs_, self._solver.n_dofs_))
            )
            self.nt_vec = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self._solver.n_dofs_))

        self.reset()

    @ti.kernel
    def clear(self):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
        for b in range(self._B):
            self.n_constraints[b] = 0
            self.n_constraints_equality[b] = 0

    @ti.kernel
    def add_collision_constraints(self):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(self._B):
            for i_col in range(self._collider.n_contacts[i_b]):
                impact = self._collider.contact_data[i_col, i_b]
                link_a = impact.link_a
                link_b = impact.link_b
                link_a_maybe_batch = [link_a, i_b] if ti.static(self._solver._options.batch_links_info) else link_a
                link_b_maybe_batch = [link_b, i_b] if ti.static(self._solver._options.batch_links_info) else link_b
                f = impact.friction
                pos = impact.pos

                d1, d2 = gu.orthogonals(impact.normal)

                t = self._solver.links_info[link_a_maybe_batch].invweight + self._solver.links_info[
                    link_b_maybe_batch
                ].invweight * (link_b > -1)
                for i in range(4):
                    n = -d1 * f - impact.normal
                    if i == 1:
                        n = d1 * f - impact.normal
                    elif i == 2:
                        n = -d2 * f - impact.normal
                    elif i == 3:
                        n = d2 * f - impact.normal

                    n_con = ti.atomic_add(self.n_constraints[i_b], 1)
                    if ti.static(self.sparse_solve):
                        for i_d_ in range(self.jac_n_relevant_dofs[n_con, i_b]):
                            i_d = self.jac_relevant_dofs[n_con, i_d_, i_b]
                            self.jac[n_con, i_d, i_b] = gs.ti_float(0.0)
                    else:
                        for i_d in range(self._solver.n_dofs):
                            self.jac[n_con, i_d, i_b] = gs.ti_float(0.0)

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
                                [link, i_b] if ti.static(self._solver._options.batch_links_info) else link
                            )

                            # reverse order to make sure dofs in each row of self.jac_relevant_dofs is strictly descending
                            for i_d_ in range(self._solver.links_info[link_maybe_batch].n_dofs):
                                i_d = self._solver.links_info[link_maybe_batch].dof_end - 1 - i_d_

                                cdof_ang = self._solver.dofs_state[i_d, i_b].cdof_ang
                                cdot_vel = self._solver.dofs_state[i_d, i_b].cdof_vel

                                t_quat = gu.ti_identity_quat()
                                t_pos = pos - self._solver.links_state[link, i_b].root_COM
                                ang, vel = gu.ti_transform_motion_by_trans_quat(cdof_ang, cdot_vel, t_pos, t_quat)

                                diff = sign * vel
                                jac = diff @ n
                                jac_qvel = jac_qvel + jac * self._solver.dofs_state[i_d, i_b].vel
                                self.jac[n_con, i_d, i_b] = self.jac[n_con, i_d, i_b] + jac

                                if ti.static(self.sparse_solve):
                                    self.jac_relevant_dofs[n_con, con_n_relevant_dofs, i_b] = i_d
                                    con_n_relevant_dofs += 1

                            link = self._solver.links_info[link_maybe_batch].parent_idx

                    if ti.static(self.sparse_solve):
                        self.jac_n_relevant_dofs[n_con, i_b] = con_n_relevant_dofs
                    imp, aref = gu.imp_aref(impact.sol_params, -impact.penetration, jac_qvel, -impact.penetration)

                    diag = t + impact.friction * impact.friction * t
                    diag *= 2 * impact.friction * impact.friction * (1 - imp) / ti.max(imp, gs.EPS)

                    self.diag[n_con, i_b] = diag
                    self.aref[n_con, i_b] = aref

                    self.efc_D[n_con, i_b] = 1 / ti.max(diag, gs.EPS)

    @ti.kernel
    def add_equality_constraints(self):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_b in range(self._B):
            for i_e in range(self._solver.n_equalities):
                eq_info = self._solver.equality_info[i_e]
                link1_idx = eq_info.link1_idx
                link2_idx = eq_info.link2_idx
                anchor1_pos = eq_info.anchor1_pos
                anchor2_pos = eq_info.anchor2_pos
                sol_params = eq_info.sol_params

                # Transform anchor positions to global coordinates
                global_anchor1 = gu.ti_transform_by_trans_quat(
                    pos=anchor1_pos,
                    trans=self._solver.links_state[link1_idx, i_b].pos,
                    quat=self._solver.links_state[link1_idx, i_b].quat,
                )
                global_anchor2 = gu.ti_transform_by_trans_quat(
                    pos=anchor2_pos,
                    trans=self._solver.links_state[link2_idx, i_b].pos,
                    quat=self._solver.links_state[link2_idx, i_b].quat,
                )

                link_a_maybe_batch = (
                    [link1_idx, i_b] if ti.static(self._solver._options.batch_links_info) else link1_idx
                )
                link_b_maybe_batch = (
                    [link2_idx, i_b] if ti.static(self._solver._options.batch_links_info) else link2_idx
                )
                invweight = (
                    self._solver.links_info[link_a_maybe_batch].invweight
                    + self._solver.links_info[link_b_maybe_batch].invweight
                )

                for i_3 in range(3):
                    n_con = ti.atomic_add(self.n_constraints[i_b], 1)
                    ti.atomic_add(self.n_constraints_equality[i_b], 1)

                    if ti.static(self.sparse_solve):
                        for i_d_ in range(self.jac_n_relevant_dofs[n_con, i_b]):
                            i_d = self.jac_relevant_dofs[n_con, i_d_, i_b]
                            self.jac[n_con, i_d, i_b] = gs.ti_float(0.0)
                    else:
                        for i_d in range(self._solver.n_dofs):
                            self.jac[n_con, i_d, i_b] = gs.ti_float(0.0)

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
                            link_maybe_batch = (
                                [link, i_b] if ti.static(self._solver._options.batch_links_info) else link
                            )

                            for i_d_ in range(self._solver.links_info[link_maybe_batch].n_dofs):
                                i_d = self._solver.links_info[link_maybe_batch].dof_end - 1 - i_d_

                                cdof_ang = self._solver.dofs_state[i_d, i_b].cdof_ang
                                cdot_vel = self._solver.dofs_state[i_d, i_b].cdof_vel

                                t_quat = gu.ti_identity_quat()
                                t_pos = pos - self._solver.links_state[link, i_b].root_COM
                                ang, vel = gu.ti_transform_motion_by_trans_quat(cdof_ang, cdot_vel, t_pos, t_quat)

                                diff = sign * vel
                                jac = diff[i_3]
                                jac_qvel = jac_qvel + jac * self._solver.dofs_state[i_d, i_b].vel
                                self.jac[n_con, i_d, i_b] = self.jac[n_con, i_d, i_b] + jac

                                if ti.static(self.sparse_solve):
                                    self.jac_relevant_dofs[n_con, con_n_relevant_dofs, i_b] = i_d
                                    con_n_relevant_dofs += 1

                            link = self._solver.links_info[link_maybe_batch].parent_idx

                    if ti.static(self.sparse_solve):
                        self.jac_n_relevant_dofs[n_con, i_b] = con_n_relevant_dofs

                    pos_diff = global_anchor1 - global_anchor2
                    penetration = pos_diff.norm()

                    imp, aref = gu.imp_aref(sol_params, -penetration, jac_qvel, pos_diff[i_3])

                    diag = invweight * (1 - imp) / (imp + gs.EPS)

                    self.diag[n_con, i_b] = diag
                    self.aref[n_con, i_b] = aref

                    self.efc_D[n_con, i_b] = 1 / ti.max(diag, gs.EPS)

    @ti.kernel
    def add_joint_limit_constraints(self):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_b in range(self._B):
            for i_l in range(self._solver.n_links):
                I_l = [i_l, i_b] if ti.static(self._solver._options.batch_links_info) else i_l
                l_info = self._solver.links_info[I_l]
                if l_info.joint_type == gs.JOINT_TYPE.REVOLUTE or l_info.joint_type == gs.JOINT_TYPE.PRISMATIC:
                    i_q = l_info.q_start
                    i_d = l_info.dof_start
                    I_d = [i_d, i_b] if ti.static(self._solver._options.batch_dofs_info) else i_d
                    pos_min = self._solver.qpos[i_q, i_b] - self._solver.dofs_info[I_d].limit[0]
                    pos_max = self._solver.dofs_info[I_d].limit[1] - self._solver.qpos[i_q, i_b]
                    pos = min(min(pos_min, pos_max), 0)

                    side = ((pos_min < pos_max) * 2 - 1) * (pos < 0)

                    jac = side
                    jac_qvel = jac * self._solver.dofs_state[i_d, i_b].vel
                    imp, aref = gu.imp_aref(self._solver.dofs_info[I_d].sol_params, pos, jac_qvel, pos)
                    diag = self._solver.dofs_info[I_d].invweight * (pos < 0) * (1 - imp) / (imp + gs.EPS)
                    aref = aref * (pos < 0)
                    if pos < 0:
                        n_con = self.n_constraints[i_b]
                        self.n_constraints[i_b] = n_con + 1
                        self.diag[n_con, i_b] = diag
                        self.aref[n_con, i_b] = aref

                        if ti.static(self.sparse_solve):
                            for i_d2_ in range(self.jac_n_relevant_dofs[n_con, i_b]):
                                i_d2 = self.jac_relevant_dofs[n_con, i_d2_, i_b]
                                self.jac[n_con, i_d2, i_b] = gs.ti_float(0.0)
                        else:
                            for i_d2 in range(self._solver.n_dofs):
                                self.jac[n_con, i_d2, i_b] = gs.ti_float(0.0)
                        self.jac[n_con, i_d, i_b] = jac

                        if ti.static(self.sparse_solve):
                            self.jac_n_relevant_dofs[n_con, i_b] = 1
                            self.jac_relevant_dofs[n_con, 0, i_b] = i_d

                        self.efc_D[n_con, i_b] = 1 / ti.max(gs.EPS, diag)

    @ti.func
    def _func_nt_hessian_incremental(self, i_b):
        rank = self._solver.n_dofs
        updated = False

        for i_c in range(self.n_constraints[i_b]):
            if not updated:
                flag_update = -1
                # add quad
                if self.prev_active[i_c, i_b] == 0 and self.active[i_c, i_b] == 1:
                    flag_update = 1
                # sub quad
                if self.prev_active[i_c, i_b] == 1 and self.active[i_c, i_b] == 0:
                    flag_update = 0

                if ti.static(self.sparse_solve):
                    if flag_update != -1:
                        for i_d_ in range(self.jac_n_relevant_dofs[i_c, i_b]):
                            i_d = self.jac_relevant_dofs[i_c, i_d_, i_b]
                            self.nt_vec[i_d, i_b] = self.jac[i_c, i_d, i_b] * ti.sqrt(self.efc_D[i_c, i_b])

                        rank = self._solver.n_dofs
                        for k_ in range(self.jac_n_relevant_dofs[i_c, i_b]):
                            k = self.jac_relevant_dofs[i_c, k_, i_b]
                            Lkk = self.nt_H[k, k, i_b]
                            tmp = Lkk * Lkk + self.nt_vec[k, i_b] * self.nt_vec[k, i_b] * (flag_update * 2 - 1)
                            if tmp < gs.EPS:
                                tmp = gs.EPS
                                rank = rank - 1
                            r = ti.sqrt(tmp)
                            c = r / Lkk
                            cinv = 1 / c
                            s = self.nt_vec[k, i_b] / Lkk
                            self.nt_H[k, k, i_b] = r
                            for i_ in range(k_):
                                i = self.jac_relevant_dofs[i_c, i_, i_b]  # i is strictly > k
                                self.nt_H[i, k, i_b] = (
                                    self.nt_H[i, k, i_b] + s * self.nt_vec[i, i_b] * (flag_update * 2 - 1)
                                ) * cinv

                            for i_ in range(k_):
                                i = self.jac_relevant_dofs[i_c, i_, i_b]  # i is strictly > k
                                self.nt_vec[i, i_b] = self.nt_vec[i, i_b] * c - s * self.nt_H[i, k, i_b]

                        if rank < self._solver.n_dofs:
                            self._func_nt_hessian_direct(i_b)
                            updated = True
                else:
                    if flag_update != -1:
                        for i_d in range(self._solver.n_dofs):
                            self.nt_vec[i_d, i_b] = self.jac[i_c, i_d, i_b] * ti.sqrt(self.efc_D[i_c, i_b])

                        rank = self._solver.n_dofs
                        for k in range(self._solver.n_dofs):
                            if ti.abs(self.nt_vec[k, i_b]) > gs.EPS:
                                Lkk = self.nt_H[k, k, i_b]
                                tmp = Lkk * Lkk + self.nt_vec[k, i_b] * self.nt_vec[k, i_b] * (flag_update * 2 - 1)
                                if tmp < gs.EPS:
                                    tmp = gs.EPS
                                    rank = rank - 1
                                r = ti.sqrt(tmp)
                                c = r / Lkk
                                cinv = 1 / c
                                s = self.nt_vec[k, i_b] / Lkk
                                self.nt_H[k, k, i_b] = r
                                for i in range(k + 1, self._solver.n_dofs):
                                    self.nt_H[i, k, i_b] = (
                                        self.nt_H[i, k, i_b] + s * self.nt_vec[i, i_b] * (flag_update * 2 - 1)
                                    ) * cinv

                                for i in range(k + 1, self._solver.n_dofs):
                                    self.nt_vec[i, i_b] = self.nt_vec[i, i_b] * c - s * self.nt_H[i, k, i_b]

                        if rank < self._solver.n_dofs:
                            self._func_nt_hessian_direct(i_b)
                            updated = True

    @ti.func
    def _func_nt_hessian_direct(self, i_b):
        # H = M + J'*D*J
        for i_d1 in range(self._solver.n_dofs):
            for i_d2 in range(self._solver.n_dofs):
                self.nt_H[i_d1, i_d2, i_b] = gs.ti_float(0.0)

        if ti.static(self.sparse_solve):
            for i_c in range(self.n_constraints[i_b]):
                jac_n_relevant_dofs = self.jac_n_relevant_dofs[i_c, i_b]
                for i_d1_ in range(jac_n_relevant_dofs):
                    i_d1 = self.jac_relevant_dofs[i_c, i_d1_, i_b]
                    if ti.abs(self.jac[i_c, i_d1, i_b]) > gs.EPS:
                        for i_d2_ in range(i_d1_, jac_n_relevant_dofs):
                            i_d2 = self.jac_relevant_dofs[i_c, i_d2_, i_b]  # i_d2 is strictly <= i_d1
                            self.nt_H[i_d1, i_d2, i_b] = (
                                self.nt_H[i_d1, i_d2, i_b]
                                + self.jac[i_c, i_d2, i_b]
                                * self.jac[i_c, i_d1, i_b]
                                * self.efc_D[i_c, i_b]
                                * self.active[i_c, i_b]
                            )
        else:
            for i_c in range(self.n_constraints[i_b]):
                for i_d1 in range(self._solver.n_dofs):
                    if ti.abs(self.jac[i_c, i_d1, i_b]) > 1e-8:
                        for i_d2 in range(i_d1 + 1):
                            self.nt_H[i_d1, i_d2, i_b] = (
                                self.nt_H[i_d1, i_d2, i_b]
                                + self.jac[i_c, i_d2, i_b]
                                * self.jac[i_c, i_d1, i_b]
                                * self.efc_D[i_c, i_b]
                                * self.active[i_c, i_b]
                            )

        for i_d1 in range(self._solver.n_dofs):
            for i_d2 in range(i_d1 + 1, self._solver.n_dofs):
                self.nt_H[i_d1, i_d2, i_b] = self.nt_H[i_d2, i_d1, i_b]

        for i_e in range(self._solver.n_entities):
            e_info = self._solver.entities_info[i_e]
            for i_d1 in range(e_info.dof_start, e_info.dof_end):
                for i_d2 in range(e_info.dof_start, e_info.dof_end):
                    self.nt_H[i_d1, i_d2, i_b] = self.nt_H[i_d1, i_d2, i_b] + self._solver.mass_mat[i_d1, i_d2, i_b]
                # self.nt_ori_H[i_d1, i_d2, i_b] = self.nt_H[i_d1, i_d2, i_b]

        self._func_nt_chol_factor(i_b)

    @ti.func
    def _func_nt_chol_factor(self, i_b):
        rank = self._solver.n_dofs
        for i_d in range(self._solver.n_dofs):
            tmp = self.nt_H[i_d, i_d, i_b]
            for j_d in range(i_d):
                tmp = tmp - (self.nt_H[i_d, j_d, i_b] * self.nt_H[i_d, j_d, i_b])

            mindiag = 1e-8
            if tmp < mindiag:
                tmp = mindiag
                rank = rank - 1
            self.nt_H[i_d, i_d, i_b] = ti.sqrt(tmp)

            tmp = 1 / self.nt_H[i_d, i_d, i_b]

            for j_d in range(i_d + 1, self._solver.n_dofs):
                dot = gs.ti_float(0.0)
                for k_d in range(i_d):
                    dot = dot + self.nt_H[j_d, k_d, i_b] * self.nt_H[i_d, k_d, i_b]

                self.nt_H[j_d, i_d, i_b] = (self.nt_H[j_d, i_d, i_b] - dot) * tmp

    @ti.func
    def _func_nt_chol_solve(self, i_b):
        for i_d in range(self._solver.n_dofs):
            self.Mgrad[i_d, i_b] = self.grad[i_d, i_b]

        for i_d in range(self._solver.n_dofs):
            for j_d in range(i_d):
                self.Mgrad[i_d, i_b] = self.Mgrad[i_d, i_b] - (self.nt_H[i_d, j_d, i_b] * self.Mgrad[j_d, i_b])
            self.Mgrad[i_d, i_b] = self.Mgrad[i_d, i_b] / self.nt_H[i_d, i_d, i_b]

        for i_d_ in range(self._solver.n_dofs):
            i_d = self._solver.n_dofs - 1 - i_d_
            for j_d in range(i_d + 1, self._solver.n_dofs):
                self.Mgrad[i_d, i_b] = self.Mgrad[i_d, i_b] - self.nt_H[j_d, i_d, i_b] * self.Mgrad[j_d, i_b]

            self.Mgrad[i_d, i_b] = self.Mgrad[i_d, i_b] / self.nt_H[i_d, i_d, i_b]

    def reset(self):
        self.jac.fill(0)
        self.qacc_ws.fill(0)
        self.meaninertia.fill(1.0)  # TODO: this is not used

        if self.sparse_solve:
            self.jac_n_relevant_dofs.fill(0)

    def handle_constraints(self):
        self.add_equality_constraints()

        if self._solver._enable_collision:
            self.add_collision_constraints()

        if self._solver._enable_joint_limit:
            self.add_joint_limit_constraints()

        if self._solver._enable_collision or self._solver._enable_joint_limit:
            self.resolve()

    def resolve(self):
        from genesis.utils.tools import create_timer

        timer = create_timer(name="resolve", level=3, ti_sync=True, skip_first_call=True)
        self._func_init_solver()
        timer.stamp("_func_init_solver")
        self._func_solve()
        timer.stamp("_func_solve")
        self._func_update_qacc()
        timer.stamp("_func_update_qacc")
        self._func_update_contact_force()
        timer.stamp("compute force")

    @ti.kernel
    def _func_update_contact_force(self):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
        for i_l, i_b in ti.ndrange(self._solver.n_links, self._B):
            self._solver.links_state[i_l, i_b].contact_force = ti.Vector.zero(gs.ti_float, 3)

        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(self._B):
            for i_col in range(self._collider.n_contacts[i_b]):
                impact = self._collider.contact_data[i_col, i_b]

                f = impact.friction
                force = ti.Vector.zero(gs.ti_float, 3)
                d1, d2 = gu.orthogonals(impact.normal)
                for i in range(4):
                    n = -d1 * f - impact.normal
                    if i == 1:
                        n = d1 * f - impact.normal
                    elif i == 2:
                        n = -d2 * f - impact.normal
                    elif i == 3:
                        n = d2 * f - impact.normal
                    force += n * self.efc_force[i_col * 4 + i, i_b]

                self._collider.contact_data[i_col, i_b].force = force

                self._solver.links_state[impact.link_a, i_b].contact_force = (
                    self._solver.links_state[impact.link_a, i_b].contact_force - force
                )
                self._solver.links_state[impact.link_b, i_b].contact_force = (
                    self._solver.links_state[impact.link_b, i_b].contact_force + force
                )

    @ti.kernel
    def _func_update_qacc(self):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
        for i_d, i_b in ti.ndrange(self._solver.n_dofs, self._B):
            self._solver.dofs_state[i_d, i_b].acc = self.qacc[i_d, i_b]
            self.qacc_ws[i_d, i_b] = self.qacc[i_d, i_b]

            self._solver.dofs_state[i_d, i_b].qf_constraint = self.qfrc_constraint[i_d, i_b]

    @ti.kernel
    def _func_solve(self):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(self._B):
            # this safeguard seems not necessary in normal execution
            # if self.n_constraints[i_b] > 0 or self.cost_ws[i_b] < self.cost[i_b]:
            if self.n_constraints[i_b] > 0:
                # cnt = 0
                for it in range(self.iterations):
                    # cnt += 1
                    self._func_solve_body(i_b)
                    if self.improved[i_b] < 1:
                        break

                    gradient = gs.ti_float(0.0)

                    for i_d in range(self._solver.n_dofs):
                        gradient += self.grad[i_d, i_b] * self.grad[i_d, i_b]

                    gradient = ti.sqrt(gradient / self._solver.n_dofs_)
                    improvement = self.prev_cost[i_b] - self.cost[i_b]
                    if gradient < self.tolerance or improvement < self.tolerance:
                        break
                # print(cnt)

    @ti.func
    def _func_ls_init(self, i_b):
        # mv and jv
        for i_e in range(self._solver.n_entities):
            e_info = self._solver.entities_info[i_e]
            for i_d1 in range(e_info.dof_start, e_info.dof_end):
                mv = gs.ti_float(0.0)
                for i_d2 in range(e_info.dof_start, e_info.dof_end):
                    mv += self._solver.mass_mat[i_d1, i_d2, i_b] * self.search[i_d2, i_b]
                self.mv[i_d1, i_b] = mv

        for i_c in range(self.n_constraints[i_b]):
            jv = gs.ti_float(0.0)
            if ti.static(self.sparse_solve):
                for i_d_ in range(self.jac_n_relevant_dofs[i_c, i_b]):
                    i_d = self.jac_relevant_dofs[i_c, i_d_, i_b]
                    jv += self.jac[i_c, i_d, i_b] * self.search[i_d, i_b]
            else:
                for i_d in range(self._solver.n_dofs):
                    jv += self.jac[i_c, i_d, i_b] * self.search[i_d, i_b]
            self.jv[i_c, i_b] = jv

        # quad and quad_gauss
        quad_gauss_1 = gs.ti_float(0.0)
        quad_gauss_2 = gs.ti_float(0.0)
        for i_d in range(self._solver.n_dofs):
            quad_gauss_1 += (
                self.search[i_d, i_b] * self.Ma[i_d, i_b]
                - self.search[i_d, i_b] * self._solver.dofs_state[i_d, i_b].force
            )
            quad_gauss_2 += 0.5 * self.search[i_d, i_b] * self.mv[i_d, i_b]
        for _i0 in range(1):
            self.quad_gauss[_i0 + 0, i_b] = self.gauss[i_b]
            self.quad_gauss[_i0 + 1, i_b] = quad_gauss_1
            self.quad_gauss[_i0 + 2, i_b] = quad_gauss_2

            for i_c in range(self.n_constraints[i_b]):
                self.quad[i_c, _i0 + 0, i_b] = self.efc_D[i_c, i_b] * (
                    0.5 * self.Jaref[i_c, i_b] * self.Jaref[i_c, i_b]
                )
                self.quad[i_c, _i0 + 1, i_b] = self.efc_D[i_c, i_b] * (self.jv[i_c, i_b] * self.Jaref[i_c, i_b])
                self.quad[i_c, _i0 + 2, i_b] = self.efc_D[i_c, i_b] * (0.5 * self.jv[i_c, i_b] * self.jv[i_c, i_b])

    @ti.func
    def _func_ls_point_fn(self, i_b, alpha):
        tmp_quad_total0, tmp_quad_total1, tmp_quad_total2 = gs.ti_float(0.0), gs.ti_float(0.0), gs.ti_float(0.0)
        for _i0 in range(1):
            tmp_quad_total0 = self.quad_gauss[_i0 + 0, i_b]
            tmp_quad_total1 = self.quad_gauss[_i0 + 1, i_b]
            tmp_quad_total2 = self.quad_gauss[_i0 + 2, i_b]
            for i_c in range(self.n_constraints[i_b]):
                active = 1
                if i_c >= self.n_constraints_equality[i_b]:
                    active = self.Jaref[i_c, i_b] + alpha * self.jv[i_c, i_b] < 0
                tmp_quad_total0 += self.quad[i_c, _i0 + 0, i_b] * active
                tmp_quad_total1 += self.quad[i_c, _i0 + 1, i_b] * active
                tmp_quad_total2 += self.quad[i_c, _i0 + 2, i_b] * active

        cost = alpha * alpha * tmp_quad_total2 + alpha * tmp_quad_total1 + tmp_quad_total0

        deriv_0 = 2 * alpha * tmp_quad_total2 + tmp_quad_total1
        deriv_1 = 2 * tmp_quad_total2 + gs.EPS * (ti.abs(tmp_quad_total2) < gs.EPS)

        self.ls_its[i_b] = self.ls_its[i_b] + 1

        return alpha, cost, deriv_0, deriv_1

    @ti.func
    def _func_no_linesearch(self, i_b):
        self._func_ls_init(i_b)
        improved = 1

        self.improved[i_b] = improved
        alpha = 1.0

        for i_d in range(self._solver.n_dofs):
            self.qacc[i_d, i_b] = self.qacc[i_d, i_b] + improved * self.search[i_d, i_b] * alpha
            self.Ma[i_d, i_b] = self.Ma[i_d, i_b] + improved * self.mv[i_d, i_b] * alpha
        for i_c in range(self.n_constraints[i_b]):
            self.Jaref[i_c, i_b] = self.Jaref[i_c, i_b] + improved * self.jv[i_c, i_b] * alpha

    @ti.func
    def _func_linesearch(self, i_b):
        ## use adaptive linesearch tolerance
        snorm = gs.ti_float(0.0)
        for jd in range(self._solver.n_dofs):
            snorm += self.search[jd, i_b] ** 2
        self.meaninertia[i_b] = 1.0
        snorm = ti.sqrt(snorm / self._solver.n_dofs_) * self.meaninertia[i_b] * self._solver.n_dofs
        self.gtol[i_b] = self.tolerance * self.ls_tolerance * snorm
        gtol = self.tolerance * self.ls_tolerance * snorm
        ## use adaptive linesearch tolerance

        self.ls_its[i_b] = 0
        self.ls_result[i_b] = 0
        ls_slope = gs.ti_float(1.0)

        res_alpha = gs.ti_float(0.0)
        done = False

        if snorm < 1e-8:
            self.ls_result[i_b] = 1
            res_alpha = 0.0
        else:
            scale = 1 / (self.meaninertia[i_b] * ti.max(1, self._solver.n_dofs))
            gtol = self.tolerance * self.ls_tolerance * snorm
            slopescl = scale / snorm

            self._func_ls_init(i_b)

            p0_alpha, p0_cost, p0_deriv_0, p0_deriv_1 = self._func_ls_point_fn(i_b, gs.ti_float(0.0))
            p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = self._func_ls_point_fn(i_b, p0_alpha - p0_deriv_0 / p0_deriv_1)

            if p0_cost < p1_cost:
                p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = p0_alpha, p0_cost, p0_deriv_0, p0_deriv_1

            if ti.abs(p1_deriv_0) < gtol:
                if ti.abs(p1_alpha) < gs.EPS:
                    self.ls_result[i_b] = 2
                else:
                    self.ls_result[i_b] = 0
                ls_slope = ti.abs(p1_deriv_0) * slopescl
                res_alpha = p1_alpha
            else:
                direction = (p1_deriv_0 < 0) * 2 - 1
                p2update = 0
                p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1 = p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1
                while p1_deriv_0 * direction <= -gtol and self.ls_its[i_b] < self.ls_iterations:
                    p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1 = p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1
                    p2update = 1

                    p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = self._func_ls_point_fn(
                        i_b, p1_alpha - p1_deriv_0 / p1_deriv_1
                    )
                    if ti.abs(p1_deriv_0) < gtol:
                        ls_slope = ti.abs(p1_deriv_0) * slopescl
                        res_alpha = p1_alpha
                        done = True
                        break
                if not done:
                    if self.ls_its[i_b] >= self.ls_iterations:
                        self.ls_result[i_b] = 3
                        ls_slope = ti.abs(p1_deriv_0) * slopescl
                        res_alpha = p1_alpha
                        done = True

                    if not p2update and not done:
                        self.ls_result[i_b] = 6
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

                        p1_next_alpha, p1_next_cost, p1_next_deriv_0, p1_next_deriv_1 = self._func_ls_point_fn(
                            i_b, p1_alpha - p1_deriv_0 / p1_deriv_1
                        )

                        while self.ls_its[i_b] < self.ls_iterations:
                            pmid_alpha, pmid_cost, pmid_deriv_0, pmid_deriv_1 = self._func_ls_point_fn(
                                i_b, (p1_alpha + p2_alpha) * 0.5
                            )

                            i = 0
                            (
                                self.candidates[4 * i + 0, i_b],
                                self.candidates[4 * i + 1, i_b],
                                self.candidates[4 * i + 2, i_b],
                                self.candidates[4 * i + 3, i_b],
                            ) = (p1_next_alpha, p1_next_cost, p1_next_deriv_0, p1_next_deriv_1)
                            i = 1
                            (
                                self.candidates[4 * i + 0, i_b],
                                self.candidates[4 * i + 1, i_b],
                                self.candidates[4 * i + 2, i_b],
                                self.candidates[4 * i + 3, i_b],
                            ) = (p2_next_alpha, p2_next_cost, p2_next_deriv_0, p2_next_deriv_1)
                            i = 2
                            (
                                self.candidates[4 * i + 0, i_b],
                                self.candidates[4 * i + 1, i_b],
                                self.candidates[4 * i + 2, i_b],
                                self.candidates[4 * i + 3, i_b],
                            ) = (pmid_alpha, pmid_cost, pmid_deriv_0, pmid_deriv_1)

                            best_i = -1
                            best_cost = gs.ti_float(0.0)
                            for ii in range(3):
                                if ti.abs(self.candidates[4 * ii + 2, i_b]) < gtol and (
                                    best_i < 0 or self.candidates[4 * ii + 1, i_b] < best_cost
                                ):
                                    best_cost = self.candidates[4 * ii + 1, i_b]
                                    best_i = ii
                            if best_i >= 0:
                                ls_slope = ti.abs(self.candidates[4 * i + 2, i_b]) * slopescl
                                res_alpha = self.candidates[4 * best_i + 0, i_b]
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
                                ) = self.update_bracket(p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1, i_b)
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
                                ) = self.update_bracket(p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1, i_b)

                                if b1 == 0 and b2 == 0:
                                    if pmid_cost < p0_cost:
                                        self.ls_result[i_b] = 0
                                    else:
                                        self.ls_result[i_b] = 7

                                    ls_slope = ti.abs(pmid_deriv_0) * slopescl

                                    res_alpha = pmid_alpha
                                    done = True

                        if not done:
                            if p1_cost <= p2_cost and p1_cost < p0_cost:
                                self.ls_result[i_b] = 4
                                ls_slope = ti.abs(p1_deriv_0) * slopescl
                                res_alpha = p1_alpha
                            elif p2_cost <= p1_cost and p2_cost < p1_cost:
                                self.ls_result[i_b] = 4
                                ls_slope = ti.abs(p2_deriv_0) * slopescl
                                res_alpha = p2_alpha
                            else:
                                self.ls_result[i_b] = 5
                                res_alpha = 0.0
        return res_alpha

    @ti.func
    def update_bracket(self, p_alpha, p_cost, p_deriv_0, p_deriv_1, i_b):
        flag = 0

        for i in range(3):
            if p_deriv_0 < 0 and self.candidates[4 * i + 2, i_b] < 0 and p_deriv_0 < self.candidates[4 * i + 2, i_b]:
                p_alpha, p_cost, p_deriv_0, p_deriv_1 = (
                    self.candidates[4 * i + 0, i_b],
                    self.candidates[4 * i + 1, i_b],
                    self.candidates[4 * i + 2, i_b],
                    self.candidates[4 * i + 3, i_b],
                )

                flag = 1

            elif p_deriv_0 > 0 and self.candidates[4 * i + 2, i_b] > 0 and p_deriv_0 > self.candidates[4 * i + 2, i_b]:
                p_alpha, p_cost, p_deriv_0, p_deriv_1 = (
                    self.candidates[4 * i + 0, i_b],
                    self.candidates[4 * i + 1, i_b],
                    self.candidates[4 * i + 2, i_b],
                    self.candidates[4 * i + 3, i_b],
                )
                flag = 2
            else:
                pass

        p_next_alpha, p_next_cost, p_next_deriv_0, p_next_deriv_1 = p_alpha, p_cost, p_deriv_0, p_deriv_1

        if flag > 0:
            p_next_alpha, p_next_cost, p_next_deriv_0, p_next_deriv_1 = self._func_ls_point_fn(
                i_b, p_alpha - p_deriv_0 / p_deriv_1
            )
        return flag, p_alpha, p_cost, p_deriv_0, p_deriv_1, p_next_alpha, p_next_cost, p_next_deriv_0, p_next_deriv_1

    @ti.func
    def _func_solve_body(self, i_b):
        alpha = self._func_linesearch(i_b)

        if ti.abs(alpha) < gs.EPS:
            self.improved[i_b] = 0
        else:
            self.improved[i_b] = 1
            for i_d in range(self._solver.n_dofs):
                self.qacc[i_d, i_b] = self.qacc[i_d, i_b] + self.search[i_d, i_b] * alpha
                self.Ma[i_d, i_b] = self.Ma[i_d, i_b] + self.mv[i_d, i_b] * alpha

            for i_c in range(self.n_constraints[i_b]):
                self.Jaref[i_c, i_b] = self.Jaref[i_c, i_b] + self.jv[i_c, i_b] * alpha

            if ti.static(self._solver_type == gs.constraint_solver.CG):
                for i_d in range(self._solver.n_dofs):
                    self.cg_prev_grad[i_d, i_b] = self.grad[i_d, i_b]
                    self.cg_prev_Mgrad[i_d, i_b] = self.Mgrad[i_d, i_b]

            self._func_update_constraint(i_b, self.qacc, self.Ma, self.cost)

            if ti.static(self._solver_type == gs.constraint_solver.CG):
                self._func_update_gradient(i_b)

                self.cg_beta[i_b] = gs.ti_float(0.0)
                self.cg_pg_dot_pMg[i_b] = gs.ti_float(0.0)

                for i_d in range(self._solver.n_dofs):
                    self.cg_beta[i_b] += self.grad[i_d, i_b] * (self.Mgrad[i_d, i_b] - self.cg_prev_Mgrad[i_d, i_b])
                    self.cg_pg_dot_pMg[i_b] += self.cg_prev_Mgrad[i_d, i_b] * self.cg_prev_grad[i_d, i_b]

                self.cg_beta[i_b] = self.cg_beta[i_b] / ti.max(gs.EPS, self.cg_pg_dot_pMg[i_b])
                self.cg_beta[i_b] = ti.max(0.0, self.cg_beta[i_b])
                for i_d in range(self._solver.n_dofs):
                    self.search[i_d, i_b] = -self.Mgrad[i_d, i_b] + self.cg_beta[i_b] * self.search[i_d, i_b]

            elif ti.static(self._solver_type == gs.constraint_solver.Newton):
                improvement = self.prev_cost[i_b] - self.cost[i_b]
                if improvement > 0:
                    self._func_nt_hessian_incremental(i_b)
                    self._func_update_gradient(i_b)
                    for i_d in range(self._solver.n_dofs):
                        self.search[i_d, i_b] = -self.Mgrad[i_d, i_b]

    @ti.func
    def _func_update_constraint(self, i_b, qacc, Ma, cost):
        self.prev_cost[i_b] = cost[i_b]
        cost[i_b] = gs.ti_float(0.0)
        self.gauss[i_b] = gs.ti_float(0.0)

        for i_c in range(self.n_constraints[i_b]):
            if ti.static(self._solver_type == gs.constraint_solver.Newton):
                self.prev_active[i_c, i_b] = self.active[i_c, i_b]
            self.active[i_c, i_b] = 1
            if i_c >= self.n_constraints_equality[i_b]:
                self.active[i_c, i_b] = self.Jaref[i_c, i_b] < 0
            self.efc_force[i_c, i_b] = -self.efc_D[i_c, i_b] * self.Jaref[i_c, i_b] * self.active[i_c, i_b]

        if ti.static(self.sparse_solve):
            for i_d in range(self._solver.n_dofs):
                self.qfrc_constraint[i_d, i_b] = gs.ti_float(0.0)
            for i_c in range(self.n_constraints[i_b]):
                for i_d_ in range(self.jac_n_relevant_dofs[i_c, i_b]):
                    i_d = self.jac_relevant_dofs[i_c, i_d_, i_b]
                    self.qfrc_constraint[i_d, i_b] = (
                        self.qfrc_constraint[i_d, i_b] + self.jac[i_c, i_d, i_b] * self.efc_force[i_c, i_b]
                    )
        else:
            for i_d in range(self._solver.n_dofs):
                qfrc_constraint = gs.ti_float(0.0)
                for i_c in range(self.n_constraints[i_b]):
                    qfrc_constraint += self.jac[i_c, i_d, i_b] * self.efc_force[i_c, i_b]
                self.qfrc_constraint[i_d, i_b] = qfrc_constraint
        # (Mx - Mx') * (x - x')
        for i_d in range(self._solver.n_dofs):
            v = (
                0.5
                * (Ma[i_d, i_b] - self._solver.dofs_state[i_d, i_b].force)
                * (qacc[i_d, i_b] - self._solver.dofs_state[i_d, i_b].acc_smooth)
            )
            self.gauss[i_b] = self.gauss[i_b] + v
            cost[i_b] = cost[i_b] + v

        # D * (Jx - aref) ** 2
        for i_c in range(self.n_constraints[i_b]):
            cost[i_b] = cost[i_b] + 0.5 * (
                self.efc_D[i_c, i_b] * self.Jaref[i_c, i_b] * self.Jaref[i_c, i_b] * self.active[i_c, i_b]
            )

    @ti.func
    def _func_update_gradient(self, i_b):
        for i_d in range(self._solver.n_dofs):
            self.grad[i_d, i_b] = (
                self.Ma[i_d, i_b] - self._solver.dofs_state[i_d, i_b].force - self.qfrc_constraint[i_d, i_b]
            )

        if ti.static(self._solver_type == gs.constraint_solver.CG):
            for i_e in range(self._solver.n_entities):
                e_info = self._solver.entities_info[i_e]
                for i_d1 in range(e_info.dof_start, e_info.dof_end):
                    Mgrad = gs.ti_float(0.0)
                    for i_d2 in range(e_info.dof_start, e_info.dof_end):
                        Mgrad += self._solver.mass_mat_inv[i_d1, i_d2, i_b] * self.grad[i_d2, i_b]
                    self.Mgrad[i_d1, i_b] = Mgrad

        elif ti.static(self._solver_type == gs.constraint_solver.Newton):
            self._func_nt_chol_solve(i_b)

    @ti.func
    def initialize_Jaref(self, qacc):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(self._B):
            for i_c in range(self.n_constraints[i_b]):
                Jaref = -self.aref[i_c, i_b]
                if ti.static(self.sparse_solve):
                    for i_d_ in range(self.jac_n_relevant_dofs[i_c, i_b]):
                        i_d = self.jac_relevant_dofs[i_c, i_d_, i_b]
                        Jaref += self.jac[i_c, i_d, i_b] * qacc[i_d, i_b]
                else:
                    for i_d in range(self._solver.n_dofs):
                        Jaref += self.jac[i_c, i_d, i_b] * qacc[i_d, i_b]
                self.Jaref[i_c, i_b] = Jaref

    @ti.func
    def initialize_Ma(self, Ma, qacc):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_e, i_d1_, i_b in ti.ndrange(self._solver.n_entities, self._solver.entity_max_dofs, self._B):
            e_info = self._solver.entities_info[i_e]
            if i_d1_ < e_info.n_dofs:
                i_d1 = e_info.dof_start + i_d1_
                Ma_ = gs.ti_float(0.0)
                for i_d2 in range(e_info.dof_start, e_info.dof_end):
                    Ma_ += self._solver.mass_mat[i_d1, i_d2, i_b] * qacc[i_d2, i_b]
                Ma[i_d1, i_b] = Ma_

    @ti.kernel
    def _func_init_solver(self):
        # check if warm start
        self.initialize_Jaref(self.qacc_ws)
        self.initialize_Ma(self.Ma_ws, self.qacc_ws)
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(self._B):
            self._func_update_constraint(i_b, self.qacc_ws, self.Ma_ws, self.cost_ws)

        self.initialize_Jaref(self._solver.dofs_state.acc)
        self.initialize_Ma(self.Ma, self._solver.dofs_state.acc)
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(self._B):
            self._func_update_constraint(i_b, self._solver.dofs_state.acc, self.Ma, self.cost)

        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
        for i_d, i_b in ti.ndrange(self._solver.n_dofs, self._B):
            if self.cost_ws[i_b] < self.cost[i_b]:
                self.qacc[i_d, i_b] = self.qacc_ws[i_d, i_b]
                self.Ma[i_d, i_b] = self.Ma_ws[i_d, i_b]
            else:
                self.qacc[i_d, i_b] = self._solver.dofs_state.acc[i_d, i_b]
        self.initialize_Jaref(self.qacc)
        # end warm start

        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(self._B):
            self._func_update_constraint(i_b, self.qacc, self.Ma, self.cost)

            if ti.static(self._solver_type == gs.constraint_solver.Newton):
                self._func_nt_hessian_direct(i_b)

            self._func_update_gradient(i_b)

        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
        for i_d, i_b in ti.ndrange(self._solver.n_dofs, self._B):
            self.search[i_d, i_b] = -self.Mgrad[i_d, i_b]
