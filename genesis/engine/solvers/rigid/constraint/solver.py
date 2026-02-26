import os
from typing import TYPE_CHECKING

import numpy as np
import quadrants as qd
import torch
from frozendict import frozendict

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.engine.solvers.rigid.abd import func_solve_mass_batch
from genesis.utils.misc import qd_to_torch

from ..collider.contact_island import ContactIsland
from . import backward as backward_constraint_solver
from . import noslip as constraint_noslip

if TYPE_CHECKING:
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver


IS_OLD_TORCH = tuple(map(int, torch.__version__.split(".")[:2])) < (2, 8)


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

        # Note that it must be over-estimated because friction parameters and joint limits may be updated dynamically.
        # * 4 constraints per contact
        # * 1 constraint per 1DoF joint limit (upper and lower, if not inf)
        # * 1 constraint per dof frictionloss
        # * up to 6 constraints per equality (weld)
        self.len_constraints = int(
            4 * rigid_solver.collider._collider_info.max_contact_pairs[None]
            + sum(joint.type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC) for joint in self._solver.joints)
            + self._solver.n_dofs
            + self._solver.n_candidate_equalities_ * 6
        )
        self.len_constraints_ = max(1, self.len_constraints)

        self.constraint_state = array_class.get_constraint_state(self, self._solver)
        self.constraint_state.qd_n_equalities.from_numpy(
            np.full((self._solver._B,), self._solver.n_equalities, dtype=gs.np_int)
        )

        self._eq_const_info_cache = {}

        cs = self.constraint_state
        self.qd_n_equalities = cs.qd_n_equalities
        self.jac = cs.jac
        self.diag = cs.diag
        self.aref = cs.aref
        self.jac_n_relevant_dofs = cs.jac_n_relevant_dofs
        self.jac_relevant_dofs = cs.jac_relevant_dofs
        self.n_constraints = cs.n_constraints
        self.n_constraints_equality = cs.n_constraints_equality
        self.n_constraints_frictionloss = cs.n_constraints_frictionloss
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

        self.candidates = cs.candidates
        self.ls_it = cs.ls_it
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

        # Creating a dummy ContactIsland, needed as param for some functions,
        # and not used when hibernation is not enabled.
        self.contact_island = ContactIsland(self._collider)

    def reset(self, envs_idx=None):
        self._eq_const_info_cache.clear()

        if gs.use_zerocopy:
            is_warmstart = qd_to_torch(self.constraint_state.is_warmstart, copy=False)
            qacc_ws = qd_to_torch(self.constraint_state.qacc_ws, copy=False)
            if isinstance(envs_idx, torch.Tensor) and (not IS_OLD_TORCH or envs_idx.dtype == torch.bool):
                if envs_idx.dtype == torch.bool:
                    is_warmstart.masked_fill_(envs_idx, False)
                    qacc_ws.masked_fill_(envs_idx[None], 0.0)
                else:
                    is_warmstart.scatter_(0, envs_idx, False)
                    qacc_ws.scatter_(1, envs_idx[None].expand((qacc_ws.shape[0], -1)), 0.0)
            else:
                is_warmstart[envs_idx] = False
                qacc_ws[:, envs_idx] = 0.0
            return

        if envs_idx is None:
            envs_idx = self._solver._scene._envs_idx
        constraint_solver_kernel_reset(
            envs_idx,
            self.constraint_state,
            self._solver._static_rigid_sim_config,
        )

    def clear(self, envs_idx=None):
        self.reset(envs_idx)

        if envs_idx is None:
            envs_idx = self._solver._scene._envs_idx
        constraint_solver_kernel_clear(
            envs_idx,
            self.constraint_state,
            self._solver._rigid_global_info,
            self._solver._static_rigid_sim_config,
        )

    def add_equality_constraints(self):
        self._eq_const_info_cache.clear()

        add_equality_constraints(
            self._solver.links_info,
            self._solver.links_state,
            self._solver.dofs_state,
            self._solver.dofs_info,
            self._solver.joints_info,
            self._solver.equalities_info,
            self.constraint_state,
            self._collider._collider_state,
            self._solver._rigid_global_info,
            self._solver._static_rigid_sim_config,
        )

    def add_inequality_constraints(self):
        add_inequality_constraints(
            self._solver.links_info,
            self._solver.links_state,
            self._solver.dofs_state,
            self._solver.dofs_info,
            self._solver.joints_info,
            self.constraint_state,
            self._collider._collider_state,
            self._solver._rigid_global_info,
            self._solver._static_rigid_sim_config,
        )

    def resolve(self):
        func_solve_init(
            self._solver.dofs_info,
            self._solver.dofs_state,
            self._solver.entities_info,
            self.constraint_state,
            self._solver._rigid_global_info,
            self._solver._static_rigid_sim_config,
        )

        func_solve_body(
            self._solver.entities_info,
            self._solver.dofs_state,
            self.constraint_state,
            self._solver._rigid_global_info,
            self._solver._static_rigid_sim_config,
        )

        func_update_qacc(
            self._solver.dofs_state,
            self.constraint_state,
            self._solver._static_rigid_sim_config,
            self._solver._errno,
        )

        if self._solver._options.noslip_iterations > 0:
            self.noslip()

        func_update_contact_force(
            self._solver.links_state,
            self._collider._collider_state,
            self.constraint_state,
            self._solver._static_rigid_sim_config,
        )

    def noslip(self):
        constraint_noslip.kernel_build_efc_AR_b(
            self._solver.dofs_state,
            self._solver.entities_info,
            self._solver._rigid_global_info,
            self.constraint_state,
            self._solver._static_rigid_sim_config,
        )

        constraint_noslip.kernel_noslip(
            self._collider._collider_state,
            self.constraint_state,
            self._solver._rigid_global_info,
            self._solver._static_rigid_sim_config,
        )

        constraint_noslip.kernel_dual_finish(
            self._solver.dofs_state,
            self._solver.entities_info,
            self._solver._rigid_global_info,
            self.constraint_state,
            self._solver._static_rigid_sim_config,
        )

    def get_equality_constraints(self, as_tensor: bool = True, to_torch: bool = True):
        # Early return if already pre-computed
        eq_const_info = self._eq_const_info_cache.get((as_tensor, to_torch))
        if eq_const_info is not None:
            return eq_const_info.copy()

        n_eqs = tuple(self.constraint_state.qd_n_equalities.to_numpy())
        n_envs = len(n_eqs)
        n_eqs_max = max(n_eqs)

        if as_tensor:
            out_size = n_envs * n_eqs_max
        else:
            *n_eqs_starts, out_size = np.cumsum(n_eqs)

        if to_torch:
            iout = torch.full((out_size, 3), -1, dtype=gs.tc_int, device=gs.device)
            fout = torch.zeros((out_size, 6), dtype=gs.tc_float, device=gs.device)
        else:
            iout = np.full((out_size, 3), -1, dtype=gs.np_int)
            fout = np.zeros((out_size, 6), dtype=gs.np_float)

        if n_eqs_max > 0:
            kernel_get_equality_constraints(
                as_tensor,
                iout,
                fout,
                self.constraint_state,
                self._solver.equalities_info,
                self._solver._static_rigid_sim_config,
            )

        if as_tensor:
            iout = iout.reshape((n_envs, n_eqs_max, 3))
            eq_type, obj_a, obj_b = (iout[..., i] for i in range(3))
            efc_force = fout.reshape((n_envs, n_eqs_max, 6))
            values = (eq_type, obj_a, obj_b, fout)
        else:
            if to_torch:
                iout_chunks = torch.split(iout, n_eqs)
                efc_force = torch.split(fout, n_eqs)
            else:
                iout_chunks = np.split(iout, n_eqs_starts)
                efc_force = np.split(fout, n_eqs_starts)
            eq_type, obj_a, obj_b = tuple(zip(*([data[..., i] for i in range(3)] for data in iout_chunks)))

        values = (eq_type, obj_a, obj_b, efc_force)
        eq_const_info = dict(zip(("type", "obj_a", "obj_b", "force"), values))

        # Cache equality constraint information before returning
        self._eq_const_info_cache[(as_tensor, to_torch)] = eq_const_info

        return eq_const_info.copy()

    def get_weld_constraints(self, as_tensor: bool = True, to_torch: bool = True):
        eq_const_info = self.get_equality_constraints(as_tensor, to_torch)
        eq_type = eq_const_info.pop("type")

        weld_const_info = {}
        if as_tensor:
            weld_mask = eq_type == gs.EQUALITY_TYPE.WELD
            n_envs = len(weld_mask)
            n_welds = weld_mask.sum(dim=-1) if to_torch else np.sum(weld_mask, axis=-1)
            n_welds_max = max(n_welds)
            for key, value in eq_const_info.items():
                shape = (n_envs, n_welds_max, *value.shape[2:])
                if to_torch:
                    if torch.is_floating_point(value):
                        weld_const_info[key] = torch.zeros(shape, dtype=value.dtype, device=value.device)
                    else:
                        weld_const_info[key] = torch.full(shape, -1, dtype=value.dtype, device=value.device)
                else:
                    if np.issubdtype(value.dtype, np.floating):
                        weld_const_info[key] = np.zeros(shape, dtype=value.dtype)
                    else:
                        weld_const_info[key] = np.full(shape, -1, dtype=value.dtype)
            for i_b, (n_welds_i, weld_mask_i) in enumerate(zip(n_welds, weld_mask)):
                for eq_value, weld_value in zip(eq_const_info.values(), weld_const_info.values()):
                    weld_value[i_b, :n_welds_i] = eq_value[i_b, weld_mask_i]
        else:
            weld_mask_chunks = tuple(eq_type_i == gs.EQUALITY_TYPE.WELD for eq_type_i in eq_type)
            for key, value in eq_const_info.items():
                weld_const_info[key] = tuple(data[weld_mask] for weld_mask, data in zip(weld_mask_chunks, value))

        weld_const_info["link_a"] = weld_const_info.pop("obj_a")
        weld_const_info["link_b"] = weld_const_info.pop("obj_b")

        return weld_const_info

    def add_weld_constraint(self, link1_idx, link2_idx, envs_idx=None):
        envs_idx = self._solver._scene._sanitize_envs_idx(envs_idx)
        link1_idx, link2_idx = int(link1_idx), int(link2_idx)

        assert link1_idx >= 0 and link2_idx >= 0
        weld_const_info = self.get_weld_constraints(as_tensor=True, to_torch=True)
        link_a = weld_const_info["link_a"]
        link_b = weld_const_info["link_b"]
        assert not (
            ((link_a == link1_idx) | (link_b == link1_idx)) & ((link_a == link2_idx) | (link_b == link2_idx))
        ).any()

        self._eq_const_info_cache.clear()
        overflow = kernel_add_weld_constraint(
            link1_idx,
            link2_idx,
            envs_idx,
            self._solver.equalities_info,
            self.constraint_state,
            self._solver.links_state,
            self._solver._rigid_global_info,
            self._solver._static_rigid_sim_config,
        )
        if overflow:
            gs.logger.warning(
                "Ignoring dynamically registered weld constraint to avoid exceeding max number of equality constraints"
                f"({self.rigid_global_info.n_candidate_equalities.to_numpy()}). Please increase the value of "
                "RigidSolver's option 'max_dynamic_constraints'."
            )

    def delete_weld_constraint(self, link1_idx, link2_idx, envs_idx=None):
        envs_idx = self._solver._scene._sanitize_envs_idx(envs_idx)
        self._eq_const_info_cache.clear()
        kernel_delete_weld_constraint(
            int(link1_idx),
            int(link2_idx),
            envs_idx,
            self._solver.equalities_info,
            self.constraint_state,
            self._solver._rigid_global_info,
            self._solver._static_rigid_sim_config,
        )

    def backward(self, dL_dqacc):
        if not self._solver._requires_grad:
            gs.raise_exception("Please set `requires_grad` to True in SimOptions to enable differentiable mode.")

        # Copy upstream gradients
        self.constraint_state.dL_dqacc.from_numpy(dL_dqacc)

        # 1. We first need to find a solution to A^T * u = g system.
        backward_constraint_solver.kernel_solve_adjoint_u(
            self._solver.entities_info,
            self._solver._rigid_global_info,
            self.constraint_state,
            self._solver._static_rigid_sim_config,
        )

        # 2. Using the solution u, we can compute the gradients of the input variables.
        backward_constraint_solver.kernel_compute_gradients(
            self._solver.entities_info,
            self.constraint_state,
            self._solver._static_rigid_sim_config,
        )


# =====================================================================================================================
# ================================================= Getters / Setters =================================================
# =====================================================================================================================


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_get_equality_constraints(
    is_padded: qd.template(),
    iout: qd.types.ndarray(),
    fout: qd.types.ndarray(),
    constraint_state: array_class.ConstraintState,
    equalities_info: array_class.EqualitiesInfo,
    static_rigid_sim_config: qd.template(),
):
    _B = constraint_state.qd_n_equalities.shape[0]
    n_eqs_max = gs.qd_int(0)

    # this is a reduction operation (global max), we have to serialize it
    # TODO: a good unittest and a better implementation from Quadrants for this kind of reduction
    qd.loop_config(serialize=True)
    for i_b in range(_B):
        n_eqs = constraint_state.qd_n_equalities[i_b]
        if n_eqs > n_eqs_max:
            n_eqs_max = n_eqs

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        i_c_start = gs.qd_int(0)
        i_e_start = gs.qd_int(0)
        if qd.static(is_padded):
            i_e_start = i_b * n_eqs_max
        else:
            for j_b in range(i_b):
                i_e_start = i_e_start + constraint_state.qd_n_equalities[j_b]

        for i_e_ in range(constraint_state.qd_n_equalities[i_b]):
            i_e = i_e_start + i_e_

            iout[i_e, 0] = equalities_info.eq_type[i_e_, i_b]
            iout[i_e, 1] = equalities_info.eq_obj1id[i_e_, i_b]
            iout[i_e, 2] = equalities_info.eq_obj2id[i_e_, i_b]

            if equalities_info.eq_type[i_e_, i_b] == gs.EQUALITY_TYPE.CONNECT:
                for i_c_ in qd.static(range(3)):
                    i_c = i_c_start + i_c_
                    fout[i_e, i_c_] = constraint_state.efc_force[i_c, i_b]
                i_c_start = i_c_start + 3
            elif equalities_info.eq_type[i_e_, i_b] == gs.EQUALITY_TYPE.WELD:
                for i_c_ in qd.static(range(6)):
                    i_c = i_c_start + i_c_
                    fout[i_e, i_c_] = constraint_state.efc_force[i_c, i_b]
                i_c_start = i_c_start + 6
            elif equalities_info.eq_type[i_e_, i_b] == gs.EQUALITY_TYPE.JOINT:
                fout[i_e, 0] = constraint_state.efc_force[i_c_start, i_b]
                i_c_start = i_c_start + 1


# =====================================================================================================================
# =================================================== Problem Setup ===================================================
# =====================================================================================================================

# ====================================== Reset and Clear Constraint Solver State ======================================


@qd.kernel(fastcache=gs.use_fastcache)
def constraint_solver_kernel_reset(
    envs_idx: qd.types.ndarray(),
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    n_dofs = constraint_state.qacc_ws.shape[0]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        constraint_state.is_warmstart[i_b] = False
        for i_d in range(n_dofs):
            constraint_state.qacc_ws[i_d, i_b] = 0.0


@qd.kernel(fastcache=gs.use_fastcache)
def constraint_solver_kernel_clear(
    envs_idx: qd.types.ndarray(),
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    n_dofs = constraint_state.qacc_ws.shape[0]
    len_constraints = constraint_state.jac.shape[0]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        constraint_state.n_constraints[i_b] = 0
        constraint_state.n_constraints_equality[i_b] = 0
        constraint_state.n_constraints_frictionloss[i_b] = 0
        # Reset dynamic equality count to static count to avoid stale constraints after partial reset
        constraint_state.qd_n_equalities[i_b] = rigid_global_info.n_equalities[None]
        for i_d, i_c in qd.ndrange(n_dofs, len_constraints):
            constraint_state.jac[i_c, i_d, i_b] = 0.0
        if qd.static(static_rigid_sim_config.sparse_solve):
            for i_c in range(len_constraints):
                constraint_state.jac_n_relevant_dofs[i_c, i_b] = 0


# ========================================= Register Pre-Defined Constraints ==========================================


@qd.func
def add_collision_constraints(
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    collider_state: array_class.ColliderState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    EPS = rigid_global_info.EPS[None]

    _B = dofs_state.ctrl_mode.shape[1]
    n_dofs = dofs_state.ctrl_mode.shape[0]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        for i_col in range(collider_state.n_contacts[i_b]):
            contact_data_link_a = collider_state.contact_data.link_a[i_col, i_b]
            contact_data_link_b = collider_state.contact_data.link_b[i_col, i_b]

            contact_data_pos = collider_state.contact_data.pos[i_col, i_b]
            contact_data_normal = collider_state.contact_data.normal[i_col, i_b]
            contact_data_friction = collider_state.contact_data.friction[i_col, i_b]
            contact_data_sol_params = collider_state.contact_data.sol_params[i_col, i_b]
            contact_data_penetration = collider_state.contact_data.penetration[i_col, i_b]

            link_a = contact_data_link_a
            link_b = contact_data_link_b
            link_a_maybe_batch = [link_a, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else link_a
            link_b_maybe_batch = [link_b, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else link_b

            d1, d2 = gu.qd_orthogonals(contact_data_normal)

            invweight = links_info.invweight[link_a_maybe_batch][0]
            if link_b > -1:
                invweight = invweight + links_info.invweight[link_b_maybe_batch][0]

            for i in range(4):
                d = (2 * (i % 2) - 1) * (d1 if i < 2 else d2)
                n = d * contact_data_friction - contact_data_normal

                n_con = qd.atomic_add(constraint_state.n_constraints[i_b], 1)
                if qd.static(static_rigid_sim_config.sparse_solve):
                    for i_d_ in range(constraint_state.jac_n_relevant_dofs[n_con, i_b]):
                        i_d = constraint_state.jac_relevant_dofs[n_con, i_d_, i_b]
                        constraint_state.jac[n_con, i_d, i_b] = gs.qd_float(0.0)
                else:
                    for i_d in range(n_dofs):
                        constraint_state.jac[n_con, i_d, i_b] = gs.qd_float(0.0)

                con_n_relevant_dofs = 0
                jac_qvel = gs.qd_float(0.0)
                for i_ab in range(2):
                    sign = gs.qd_float(-1.0)
                    link = link_a
                    if i_ab == 1:
                        sign = gs.qd_float(1.0)
                        link = link_b

                    while link > -1:
                        link_maybe_batch = [link, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else link

                        # reverse order to make sure dofs in each row of self.jac_relevant_dofs is strictly descending
                        for i_d_ in range(links_info.n_dofs[link_maybe_batch]):
                            i_d = links_info.dof_end[link_maybe_batch] - 1 - i_d_

                            cdof_ang = dofs_state.cdof_ang[i_d, i_b]
                            cdot_vel = dofs_state.cdof_vel[i_d, i_b]

                            t_quat = gu.qd_identity_quat()
                            t_pos = contact_data_pos - links_state.root_COM[link, i_b]
                            _, vel = gu.qd_transform_motion_by_trans_quat(cdof_ang, cdot_vel, t_pos, t_quat)

                            diff = sign * vel
                            jac = diff @ n
                            jac_qvel = jac_qvel + jac * dofs_state.vel[i_d, i_b]
                            constraint_state.jac[n_con, i_d, i_b] = constraint_state.jac[n_con, i_d, i_b] + jac

                            if qd.static(static_rigid_sim_config.sparse_solve):
                                constraint_state.jac_relevant_dofs[n_con, con_n_relevant_dofs, i_b] = i_d
                                con_n_relevant_dofs = con_n_relevant_dofs + 1

                        link = links_info.parent_idx[link_maybe_batch]

                if qd.static(static_rigid_sim_config.sparse_solve):
                    constraint_state.jac_n_relevant_dofs[n_con, i_b] = con_n_relevant_dofs
                imp, aref = gu.imp_aref(
                    contact_data_sol_params, -contact_data_penetration, jac_qvel, -contact_data_penetration
                )

                diag = invweight + contact_data_friction * contact_data_friction * invweight
                diag *= 2 * contact_data_friction * contact_data_friction * (1 - imp) / imp
                diag = qd.max(diag, EPS)

                constraint_state.diag[n_con, i_b] = diag
                constraint_state.aref[n_con, i_b] = aref
                constraint_state.efc_D[n_con, i_b] = 1 / diag


@qd.func
def func_equality_connect(
    i_b,
    i_e,
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    dofs_state: array_class.DofsState,
    equalities_info: array_class.EqualitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    EPS = rigid_global_info.EPS[None]

    n_dofs = dofs_state.ctrl_mode.shape[0]

    link1_idx = equalities_info.eq_obj1id[i_e, i_b]
    link2_idx = equalities_info.eq_obj2id[i_e, i_b]
    link_a_maybe_batch = [link1_idx, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else link1_idx
    link_b_maybe_batch = [link2_idx, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else link2_idx
    anchor1_pos = gs.qd_vec3(
        [
            equalities_info.eq_data[i_e, i_b][0],
            equalities_info.eq_data[i_e, i_b][1],
            equalities_info.eq_data[i_e, i_b][2],
        ]
    )
    anchor2_pos = gs.qd_vec3(
        [
            equalities_info.eq_data[i_e, i_b][3],
            equalities_info.eq_data[i_e, i_b][4],
            equalities_info.eq_data[i_e, i_b][5],
        ]
    )
    sol_params = equalities_info.sol_params[i_e, i_b]

    # Transform anchor positions to global coordinates
    global_anchor1 = gu.qd_transform_by_trans_quat(
        pos=anchor1_pos,
        trans=links_state.pos[link1_idx, i_b],
        quat=links_state.quat[link1_idx, i_b],
    )
    global_anchor2 = gu.qd_transform_by_trans_quat(
        pos=anchor2_pos,
        trans=links_state.pos[link2_idx, i_b],
        quat=links_state.quat[link2_idx, i_b],
    )

    invweight = links_info.invweight[link_a_maybe_batch][0] + links_info.invweight[link_b_maybe_batch][0]

    for i_3 in range(3):
        n_con = qd.atomic_add(constraint_state.n_constraints[i_b], 1)
        qd.atomic_add(constraint_state.n_constraints_equality[i_b], 1)
        con_n_relevant_dofs = 0

        if qd.static(static_rigid_sim_config.sparse_solve):
            for i_d_ in range(constraint_state.jac_n_relevant_dofs[n_con, i_b]):
                i_d = constraint_state.jac_relevant_dofs[n_con, i_d_, i_b]
                constraint_state.jac[n_con, i_d, i_b] = gs.qd_float(0.0)
        else:
            for i_d in range(n_dofs):
                constraint_state.jac[n_con, i_d, i_b] = gs.qd_float(0.0)

        jac_qvel = gs.qd_float(0.0)
        for i_ab in range(2):
            sign = gs.qd_float(1.0)
            link = link1_idx
            pos = global_anchor1
            if i_ab == 1:
                sign = gs.qd_float(-1.0)
                link = link2_idx
                pos = global_anchor2

            while link > -1:
                link_maybe_batch = [link, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else link

                for i_d_ in range(links_info.n_dofs[link_maybe_batch]):
                    i_d = links_info.dof_end[link_maybe_batch] - 1 - i_d_

                    cdof_ang = dofs_state.cdof_ang[i_d, i_b]
                    cdot_vel = dofs_state.cdof_vel[i_d, i_b]

                    t_quat = gu.qd_identity_quat()
                    t_pos = pos - links_state.root_COM[link, i_b]
                    ang, vel = gu.qd_transform_motion_by_trans_quat(cdof_ang, cdot_vel, t_pos, t_quat)

                    diff = sign * vel
                    jac = diff[i_3]
                    jac_qvel = jac_qvel + jac * dofs_state.vel[i_d, i_b]
                    constraint_state.jac[n_con, i_d, i_b] = constraint_state.jac[n_con, i_d, i_b] + jac

                    if qd.static(static_rigid_sim_config.sparse_solve):
                        constraint_state.jac_relevant_dofs[n_con, con_n_relevant_dofs, i_b] = i_d
                        con_n_relevant_dofs = con_n_relevant_dofs + 1

                link = links_info.parent_idx[link_maybe_batch]

        if qd.static(static_rigid_sim_config.sparse_solve):
            constraint_state.jac_n_relevant_dofs[n_con, i_b] = con_n_relevant_dofs

        pos_diff = global_anchor1 - global_anchor2
        penetration = pos_diff.norm()

        imp, aref = gu.imp_aref(sol_params, -penetration, jac_qvel, pos_diff[i_3])

        diag = qd.max(invweight * (1.0 - imp) / imp, EPS)

        constraint_state.diag[n_con, i_b] = diag
        constraint_state.aref[n_con, i_b] = aref
        constraint_state.efc_D[n_con, i_b] = 1.0 / diag


@qd.func
def func_equality_joint(
    i_b,
    i_e,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    equalities_info: array_class.EqualitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    EPS = rigid_global_info.EPS[None]

    n_dofs = constraint_state.jac.shape[1]

    sol_params = equalities_info.sol_params[i_e, i_b]

    I_joint1 = (
        [equalities_info.eq_obj1id[i_e, i_b], i_b]
        if qd.static(static_rigid_sim_config.batch_joints_info)
        else equalities_info.eq_obj1id[i_e, i_b]
    )
    I_joint2 = (
        [equalities_info.eq_obj2id[i_e, i_b], i_b]
        if qd.static(static_rigid_sim_config.batch_joints_info)
        else equalities_info.eq_obj2id[i_e, i_b]
    )
    i_qpos1 = joints_info.q_start[I_joint1]
    i_qpos2 = joints_info.q_start[I_joint2]
    i_dof1 = joints_info.dof_start[I_joint1]
    i_dof2 = joints_info.dof_start[I_joint2]
    I_dof1 = [i_dof1, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else i_dof1
    I_dof2 = [i_dof2, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else i_dof2

    n_con = qd.atomic_add(constraint_state.n_constraints[i_b], 1)
    qd.atomic_add(constraint_state.n_constraints_equality[i_b], 1)

    if qd.static(static_rigid_sim_config.sparse_solve):
        for i_d_ in range(constraint_state.jac_n_relevant_dofs[n_con, i_b]):
            i_d = constraint_state.jac_relevant_dofs[n_con, i_d_, i_b]
            constraint_state.jac[n_con, i_d, i_b] = gs.qd_float(0.0)
    else:
        for i_d in range(n_dofs):
            constraint_state.jac[n_con, i_d, i_b] = gs.qd_float(0.0)

    pos1 = rigid_global_info.qpos[i_qpos1, i_b]
    pos2 = rigid_global_info.qpos[i_qpos2, i_b]
    ref1 = rigid_global_info.qpos0[i_qpos1, i_b]
    ref2 = rigid_global_info.qpos0[i_qpos2, i_b]

    # TODO: zero objid2
    diff = pos2 - ref2
    pos = pos1 - ref1
    deriv = gs.qd_float(0.0)

    # y - y0 = a0 + a1 * (x-x0) + a2 * (x-x0)^2 + a3 * (x-fx0)^3 + a4 * (x-x0)^4
    for i_5 in range(5):
        diff_power = diff**i_5
        pos = pos - diff_power * equalities_info.eq_data[i_e, i_b][i_5]
        if i_5 < 4:
            deriv = deriv + equalities_info.eq_data[i_e, i_b][i_5 + 1] * diff_power * (i_5 + 1)

    constraint_state.jac[n_con, i_dof1, i_b] = gs.qd_float(1.0)
    constraint_state.jac[n_con, i_dof2, i_b] = -deriv
    jac_qvel = (
        constraint_state.jac[n_con, i_dof1, i_b] * dofs_state.vel[i_dof1, i_b]
        + constraint_state.jac[n_con, i_dof2, i_b] * dofs_state.vel[i_dof2, i_b]
    )
    invweight = dofs_info.invweight[I_dof1] + dofs_info.invweight[I_dof2]

    imp, aref = gu.imp_aref(sol_params, -qd.abs(pos), jac_qvel, pos)

    diag = qd.max(invweight * (1.0 - imp) / imp, EPS)

    constraint_state.diag[n_con, i_b] = diag
    constraint_state.aref[n_con, i_b] = aref
    constraint_state.efc_D[n_con, i_b] = 1.0 / diag


@qd.kernel(fastcache=gs.use_fastcache)
def add_equality_constraints(
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    joints_info: array_class.JointsInfo,
    equalities_info: array_class.EqualitiesInfo,
    constraint_state: array_class.ConstraintState,
    collider_state: array_class.ColliderState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    _B = dofs_state.ctrl_mode.shape[1]

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        constraint_state.n_constraints[i_b] = 0
        constraint_state.n_constraints_equality[i_b] = 0

        for i_e in range(constraint_state.qd_n_equalities[i_b]):
            if equalities_info.eq_type[i_e, i_b] == gs.EQUALITY_TYPE.CONNECT:
                func_equality_connect(
                    i_b,
                    i_e,
                    links_info=links_info,
                    links_state=links_state,
                    dofs_state=dofs_state,
                    equalities_info=equalities_info,
                    constraint_state=constraint_state,
                    rigid_global_info=rigid_global_info,
                    static_rigid_sim_config=static_rigid_sim_config,
                )

            elif equalities_info.eq_type[i_e, i_b] == gs.EQUALITY_TYPE.WELD:
                func_equality_weld(
                    i_b,
                    i_e,
                    links_info=links_info,
                    links_state=links_state,
                    dofs_state=dofs_state,
                    equalities_info=equalities_info,
                    constraint_state=constraint_state,
                    rigid_global_info=rigid_global_info,
                    static_rigid_sim_config=static_rigid_sim_config,
                )
            elif equalities_info.eq_type[i_e, i_b] == gs.EQUALITY_TYPE.JOINT:
                func_equality_joint(
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


@qd.kernel(fastcache=gs.use_fastcache)
def add_inequality_constraints(
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    joints_info: array_class.JointsInfo,
    constraint_state: array_class.ConstraintState,
    collider_state: array_class.ColliderState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    add_frictionloss_constraints(
        links_info=links_info,
        joints_info=joints_info,
        dofs_info=dofs_info,
        dofs_state=dofs_state,
        rigid_global_info=rigid_global_info,
        constraint_state=constraint_state,
        static_rigid_sim_config=static_rigid_sim_config,
    )
    if qd.static(static_rigid_sim_config.enable_collision):
        add_collision_constraints(
            links_info=links_info,
            links_state=links_state,
            dofs_state=dofs_state,
            constraint_state=constraint_state,
            collider_state=collider_state,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )
    if qd.static(static_rigid_sim_config.enable_joint_limit):
        add_joint_limit_constraints(
            links_info=links_info,
            joints_info=joints_info,
            dofs_info=dofs_info,
            dofs_state=dofs_state,
            rigid_global_info=rigid_global_info,
            constraint_state=constraint_state,
            static_rigid_sim_config=static_rigid_sim_config,
        )


@qd.func
def func_equality_weld(
    i_b,
    i_e,
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    dofs_state: array_class.DofsState,
    equalities_info: array_class.EqualitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    EPS = rigid_global_info.EPS[None]

    n_dofs = dofs_state.ctrl_mode.shape[0]

    # TODO: sparse mode
    # Get equality info for this constraint
    link1_idx = equalities_info.eq_obj1id[i_e, i_b]
    link2_idx = equalities_info.eq_obj2id[i_e, i_b]
    link_a_maybe_batch = [link1_idx, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else link1_idx
    link_b_maybe_batch = [link2_idx, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else link2_idx

    # For weld, eq_data layout:
    # [0:3]  : anchor2 (local pos in body2)
    # [3:6]  : anchor1 (local pos in body1)
    # [6:10] : relative pose (quat) of body 2 related to body 1 to match orientations
    # [10]   : torquescale
    anchor1_pos = gs.qd_vec3(
        [
            equalities_info.eq_data[i_e, i_b][3],
            equalities_info.eq_data[i_e, i_b][4],
            equalities_info.eq_data[i_e, i_b][5],
        ]
    )
    anchor2_pos = gs.qd_vec3(
        [
            equalities_info.eq_data[i_e, i_b][0],
            equalities_info.eq_data[i_e, i_b][1],
            equalities_info.eq_data[i_e, i_b][2],
        ]
    )
    relpose = gs.qd_vec4(
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
    global_anchor1 = gu.qd_transform_by_trans_quat(
        pos=anchor1_pos,
        trans=links_state.pos[link1_idx, i_b],
        quat=links_state.quat[link1_idx, i_b],
    )
    global_anchor2 = gu.qd_transform_by_trans_quat(
        pos=anchor2_pos,
        trans=links_state.pos[link2_idx, i_b],
        quat=links_state.quat[link2_idx, i_b],
    )

    pos_error = global_anchor1 - global_anchor2

    # Compute orientation error.
    # For weld: compute q = body1_quat * relpose, then error = (inv(body2_quat) * q)
    quat_body1 = links_state.quat[link1_idx, i_b]
    quat_body2 = links_state.quat[link2_idx, i_b]
    q = gu.qd_quat_mul(quat_body1, relpose)
    inv_quat_body2 = gu.qd_inv_quat(quat_body2)
    error_quat = gu.qd_quat_mul(inv_quat_body2, q)
    # Take the vector (axis) part and scale by torquescale.
    rot_error = gs.qd_vec3([error_quat[1], error_quat[2], error_quat[3]]) * torquescale

    all_error = gs.qd_vec6([pos_error[0], pos_error[1], pos_error[2], rot_error[0], rot_error[1], rot_error[2]])
    pos_imp = all_error.norm()

    # Compute inverse weight from both bodies.
    invweight = links_info.invweight[link_a_maybe_batch] + links_info.invweight[link_b_maybe_batch]

    # --- Position part (first 3 constraints) ---
    for i in range(3):
        n_con = qd.atomic_add(constraint_state.n_constraints[i_b], 1)
        qd.atomic_add(constraint_state.n_constraints_equality[i_b], 1)
        con_n_relevant_dofs = 0

        if qd.static(static_rigid_sim_config.sparse_solve):
            for i_d_ in range(constraint_state.jac_n_relevant_dofs[n_con, i_b]):
                i_d = constraint_state.jac_relevant_dofs[n_con, i_d_, i_b]
                constraint_state.jac[n_con, i_d, i_b] = gs.qd_float(0.0)
        else:
            for i_d in range(n_dofs):
                constraint_state.jac[n_con, i_d, i_b] = gs.qd_float(0.0)

        jac_qvel = gs.qd_float(0.0)
        for i_ab in range(2):
            sign = gs.qd_float(1.0) if i_ab == 0 else gs.qd_float(-1.0)
            link = link1_idx if i_ab == 0 else link2_idx
            pos_anchor = global_anchor1 if i_ab == 0 else global_anchor2

            # Accumulate jacobian contributions along the kinematic chain.
            # (Assuming similar structure to equality_connect.)
            while link > -1:
                link_maybe_batch = [link, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else link

                for i_d_ in range(links_info.n_dofs[link_maybe_batch]):
                    i_d = links_info.dof_end[link_maybe_batch] - 1 - i_d_
                    cdof_ang = dofs_state.cdof_ang[i_d, i_b]
                    cdot_vel = dofs_state.cdof_vel[i_d, i_b]
                    t_pos = pos_anchor - links_state.root_COM[link, i_b]
                    # t_quat = gu.qd_identity_quat()
                    # _ang, vel = gu.qd_transform_motion_by_trans_quat(cdof_ang, cdot_vel, t_pos, t_quat)
                    vel = cdot_vel - t_pos.cross(cdof_ang)
                    diff = sign * vel
                    jac = diff[i]
                    jac_qvel = jac_qvel + jac * dofs_state.vel[i_d, i_b]
                    constraint_state.jac[n_con, i_d, i_b] = constraint_state.jac[n_con, i_d, i_b] + jac

                    if qd.static(static_rigid_sim_config.sparse_solve):
                        constraint_state.jac_relevant_dofs[n_con, con_n_relevant_dofs, i_b] = i_d
                        con_n_relevant_dofs = con_n_relevant_dofs + 1
                link = links_info.parent_idx[link_maybe_batch]

        if qd.static(static_rigid_sim_config.sparse_solve):
            constraint_state.jac_n_relevant_dofs[n_con, i_b] = con_n_relevant_dofs

        imp, aref = gu.imp_aref(sol_params, -pos_imp, jac_qvel, pos_error[i])
        diag = qd.max(invweight[0] * (1 - imp) / imp, EPS)

        constraint_state.diag[n_con, i_b] = diag
        constraint_state.aref[n_con, i_b] = aref
        constraint_state.efc_D[n_con, i_b] = 1.0 / diag

    # --- Orientation part (next 3 constraints) ---
    n_con = qd.atomic_add(constraint_state.n_constraints[i_b], 3)
    qd.atomic_add(constraint_state.n_constraints_equality[i_b], 3)
    con_n_relevant_dofs = 0
    for i_con in range(n_con, n_con + 3):
        for i_d in range(n_dofs):
            constraint_state.jac[i_con, i_d, i_b] = gs.qd_float(0.0)

    for i_ab in range(2):
        sign = gs.qd_float(1.0) if i_ab == 0 else gs.qd_float(-1.0)
        link = link1_idx if i_ab == 0 else link2_idx
        # For rotation, we use the body's orientation (here we use its quaternion)
        # and a suitable reference frame. (You may need a more detailed implementation.)
        while link > -1:
            link_maybe_batch = [link, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else link

            for i_d_ in range(links_info.n_dofs[link_maybe_batch]):
                i_d = links_info.dof_end[link_maybe_batch] - 1 - i_d_
                jac = sign * dofs_state.cdof_ang[i_d, i_b]

                for i_con in range(n_con, n_con + 3):
                    constraint_state.jac[i_con, i_d, i_b] = constraint_state.jac[i_con, i_d, i_b] + jac[i_con - n_con]
            link = links_info.parent_idx[link_maybe_batch]

    jac_qvel = qd.Vector([0.0, 0.0, 0.0])
    for i_d in range(n_dofs):
        # quat2 = neg(q1)*(jac0-jac1)
        # quat3 = neg(q1)*(jac0-jac1)*q0*relpose
        jac_diff_r = qd.Vector(
            [
                constraint_state.jac[n_con, i_d, i_b],
                constraint_state.jac[n_con + 1, i_d, i_b],
                constraint_state.jac[n_con + 2, i_d, i_b],
            ]
        )
        quat2 = gu.qd_quat_mul_axis(inv_quat_body2, jac_diff_r)
        quat3 = gu.qd_quat_mul(quat2, q)

        for i_con in range(n_con, n_con + 3):
            constraint_state.jac[i_con, i_d, i_b] = 0.5 * quat3[i_con - n_con + 1] * torquescale
            jac_qvel[i_con - n_con] = (
                jac_qvel[i_con - n_con] + constraint_state.jac[i_con, i_d, i_b] * dofs_state.vel[i_d, i_b]
            )

    if qd.static(static_rigid_sim_config.sparse_solve):
        for i_con in range(n_con, n_con + 3):
            constraint_state.jac_n_relevant_dofs[i_con, i_b] = con_n_relevant_dofs

    for i_con in range(n_con, n_con + 3):
        imp, aref = gu.imp_aref(sol_params, -pos_imp, jac_qvel[i_con - n_con], rot_error[i_con - n_con])
        diag = qd.max(invweight[1] * (1.0 - imp) / imp, EPS)

        constraint_state.diag[i_con, i_b] = diag
        constraint_state.aref[i_con, i_b] = aref
        constraint_state.efc_D[i_con, i_b] = 1.0 / diag


@qd.func
def add_joint_limit_constraints(
    links_info: array_class.LinksInfo,
    joints_info: array_class.JointsInfo,
    dofs_info: array_class.DofsInfo,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    EPS = rigid_global_info.EPS[None]

    _B = constraint_state.jac.shape[2]
    n_links = links_info.root_idx.shape[0]
    n_dofs = dofs_state.ctrl_mode.shape[0]

    # TODO: sparse mode
    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        for i_l in range(n_links):
            I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l

            for i_j in range(links_info.joint_start[I_l], links_info.joint_end[I_l]):
                I_j = [i_j, i_b] if qd.static(static_rigid_sim_config.batch_joints_info) else i_j

                if joints_info.type[I_j] == gs.JOINT_TYPE.REVOLUTE or joints_info.type[I_j] == gs.JOINT_TYPE.PRISMATIC:
                    i_q = joints_info.q_start[I_j]
                    i_d = joints_info.dof_start[I_j]
                    I_d = [i_d, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else i_d
                    pos_delta_min = rigid_global_info.qpos[i_q, i_b] - dofs_info.limit[I_d][0]
                    pos_delta_max = dofs_info.limit[I_d][1] - rigid_global_info.qpos[i_q, i_b]
                    pos_delta = qd.min(pos_delta_min, pos_delta_max)

                    if pos_delta < 0:
                        jac = (pos_delta_min < pos_delta_max) * 2 - 1
                        jac_qvel = jac * dofs_state.vel[i_d, i_b]
                        imp, aref = gu.imp_aref(joints_info.sol_params[I_j], pos_delta, jac_qvel, pos_delta)
                        diag = qd.max(dofs_info.invweight[I_d] * (1 - imp) / imp, EPS)

                        n_con = qd.atomic_add(constraint_state.n_constraints[i_b], 1)
                        constraint_state.diag[n_con, i_b] = diag
                        constraint_state.aref[n_con, i_b] = aref
                        constraint_state.efc_D[n_con, i_b] = 1 / diag

                        if qd.static(static_rigid_sim_config.sparse_solve):
                            for i_d2_ in range(constraint_state.jac_n_relevant_dofs[n_con, i_b]):
                                i_d2 = constraint_state.jac_relevant_dofs[n_con, i_d2_, i_b]
                                constraint_state.jac[n_con, i_d2, i_b] = gs.qd_float(0.0)
                        else:
                            for i_d2 in range(n_dofs):
                                constraint_state.jac[n_con, i_d2, i_b] = gs.qd_float(0.0)
                        constraint_state.jac[n_con, i_d, i_b] = jac

                        if qd.static(static_rigid_sim_config.sparse_solve):
                            constraint_state.jac_n_relevant_dofs[n_con, i_b] = 1
                            constraint_state.jac_relevant_dofs[n_con, 0, i_b] = i_d


@qd.func
def add_frictionloss_constraints(
    links_info: array_class.LinksInfo,
    joints_info: array_class.JointsInfo,
    dofs_info: array_class.DofsInfo,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    EPS = rigid_global_info.EPS[None]

    _B = constraint_state.jac.shape[2]
    n_links = links_info.root_idx.shape[0]
    n_dofs = dofs_state.ctrl_mode.shape[0]

    # TODO: sparse mode
    # FIXME: The condition `if dofs_info.frictionloss[I_d] > EPS:` is not correctly evaluated on Apple Metal
    # if `serialize=True`...
    qd.loop_config(
        serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL and gs.backend != gs.metal)
    )
    for i_b in range(_B):
        constraint_state.n_constraints_frictionloss[i_b] = 0

        for i_l in range(n_links):
            I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l

            for i_j in range(links_info.joint_start[I_l], links_info.joint_end[I_l]):
                I_j = [i_j, i_b] if qd.static(static_rigid_sim_config.batch_joints_info) else i_j

                for i_d in range(joints_info.dof_start[I_j], joints_info.dof_end[I_j]):
                    I_d = [i_d, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else i_d

                    if dofs_info.frictionloss[I_d] > EPS:
                        jac = 1.0
                        jac_qvel = jac * dofs_state.vel[i_d, i_b]
                        imp, aref = gu.imp_aref(joints_info.sol_params[I_j], 0.0, jac_qvel, 0.0)
                        diag = qd.max(dofs_info.invweight[I_d] * (1.0 - imp) / imp, EPS)

                        i_con = qd.atomic_add(constraint_state.n_constraints[i_b], 1)
                        qd.atomic_add(constraint_state.n_constraints_frictionloss[i_b], 1)

                        constraint_state.diag[i_con, i_b] = diag
                        constraint_state.aref[i_con, i_b] = aref
                        constraint_state.efc_D[i_con, i_b] = 1.0 / diag
                        constraint_state.efc_frictionloss[i_con, i_b] = dofs_info.frictionloss[I_d]
                        for i_d2 in range(n_dofs):
                            constraint_state.jac[i_con, i_d2, i_b] = gs.qd_float(0.0)
                        constraint_state.jac[i_con, i_d, i_b] = jac


# ====================================== Runtime User-Specified Weld Constraints ======================================


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_add_weld_constraint(
    link1_idx: qd.i32,
    link2_idx: qd.i32,
    envs_idx: qd.types.ndarray(),
    equalities_info: array_class.EqualitiesInfo,
    constraint_state: array_class.ConstraintState,
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
) -> qd.i32:
    overflow = gs.qd_bool(False)

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_e = constraint_state.qd_n_equalities[i_b]
        if i_e == rigid_global_info.n_candidate_equalities[None]:
            overflow = True
        else:
            shared_pos = links_state.pos[link1_idx, i_b]
            pos1 = gu.qd_inv_transform_by_trans_quat(
                shared_pos, links_state.pos[link1_idx, i_b], links_state.quat[link1_idx, i_b]
            )
            pos2 = gu.qd_inv_transform_by_trans_quat(
                shared_pos, links_state.pos[link2_idx, i_b], links_state.quat[link2_idx, i_b]
            )

            equalities_info.eq_type[i_e, i_b] = gs.qd_int(gs.EQUALITY_TYPE.WELD)
            equalities_info.eq_obj1id[i_e, i_b] = link1_idx
            equalities_info.eq_obj2id[i_e, i_b] = link2_idx

            for i_3 in qd.static(range(3)):
                equalities_info.eq_data[i_e, i_b][i_3 + 3] = pos1[i_3]
                equalities_info.eq_data[i_e, i_b][i_3] = pos2[i_3]

            relpose = gu.qd_quat_mul(gu.qd_inv_quat(links_state.quat[link1_idx, i_b]), links_state.quat[link2_idx, i_b])

            for i_4 in qd.static(range(4)):
                equalities_info.eq_data[i_e, i_b][i_4 + 6] = relpose[i_4]

            equalities_info.eq_data[i_e, i_b][10] = 1.0

            equalities_info.sol_params[i_e, i_b] = qd.Vector(
                [2 * rigid_global_info.substep_dt[None], 1.0, 0.9, 0.95, 0.001, 0.5, 2.0]
            )

            constraint_state.qd_n_equalities[i_b] = constraint_state.qd_n_equalities[i_b] + 1
    return overflow


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_delete_weld_constraint(
    link1_idx: qd.i32,
    link2_idx: qd.i32,
    envs_idx: qd.types.ndarray(),
    equalities_info: array_class.EqualitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        for i_e in range(rigid_global_info.n_equalities[None], constraint_state.qd_n_equalities[i_b]):
            if (
                equalities_info.eq_type[i_e, i_b] == gs.EQUALITY_TYPE.WELD
                and equalities_info.eq_obj1id[i_e, i_b] == link1_idx
                and equalities_info.eq_obj2id[i_e, i_b] == link2_idx
            ):
                if i_e < constraint_state.qd_n_equalities[i_b] - 1:
                    equalities_info.eq_type[i_e, i_b] = equalities_info.eq_type[
                        constraint_state.qd_n_equalities[i_b] - 1, i_b
                    ]
                constraint_state.qd_n_equalities[i_b] = constraint_state.qd_n_equalities[i_b] - 1


# =====================================================================================================================
# ================================================= Solving Iteration =================================================
# =====================================================================================================================

# ====================================== Hessian Matrix & Cholesky Factorization ======================================


@qd.func
def func_hessian_direct_batch(
    i_b,
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Compute the Hessian matrix `H = M + J.T @ D @ J of the optimization problem for a given environment `i_b`.

    Note that only the lower triangular part will be updated for efficiency, because the Hessian matrix is symmetric.
    The upper triangular part is left as-is for efficiency. Accordingly, our solver's functions all leverage the
    symmetry property of the Hessian matrix and only ever use values from the upper triangle.
    """
    EPS = rigid_global_info.EPS[None]

    n_dofs = constraint_state.nt_H.shape[1]
    n_entities = entities_info.n_links.shape[0]

    # Reset Hessian matrix to zero
    for i_d1 in range(n_dofs):
        for i_d2 in range(i_d1 + 1):
            constraint_state.nt_H[i_b, i_d1, i_d2] = gs.qd_float(0.0)

    # Compute `H += J.T @ D @ J` using either dense or sparse implementation
    if qd.static(static_rigid_sim_config.sparse_solve):
        for i_c in range(constraint_state.n_constraints[i_b]):
            jac_n_relevant_dofs = constraint_state.jac_n_relevant_dofs[i_c, i_b]
            for i_d1_ in range(jac_n_relevant_dofs):
                i_d1 = constraint_state.jac_relevant_dofs[i_c, i_d1_, i_b]
                if qd.abs(constraint_state.jac[i_c, i_d1, i_b]) > EPS:
                    for i_d2_ in range(i_d1_, jac_n_relevant_dofs):
                        i_d2 = constraint_state.jac_relevant_dofs[i_c, i_d2_, i_b]  # i_d2 is strictly <= i_d1
                        constraint_state.nt_H[i_b, i_d1, i_d2] = (
                            constraint_state.nt_H[i_b, i_d1, i_d2]
                            + constraint_state.jac[i_c, i_d2, i_b]
                            * constraint_state.jac[i_c, i_d1, i_b]
                            * constraint_state.efc_D[i_c, i_b]
                            * constraint_state.active[i_c, i_b]
                        )
    else:
        for i_d1, i_c in qd.ndrange(n_dofs, constraint_state.n_constraints[i_b]):
            if qd.abs(constraint_state.jac[i_c, i_d1, i_b]) > EPS:
                for i_d2 in range(i_d1 + 1):
                    constraint_state.nt_H[i_b, i_d1, i_d2] = (
                        constraint_state.nt_H[i_b, i_d1, i_d2]
                        + constraint_state.jac[i_c, i_d2, i_b]
                        * constraint_state.jac[i_c, i_d1, i_b]
                        * constraint_state.efc_D[i_c, i_b]
                        * constraint_state.active[i_c, i_b]
                    )

    # Compute `H += M`
    for i_e in range(n_entities):
        for i_d1 in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
            for i_d2 in range(entities_info.dof_start[i_e], i_d1 + 1):
                constraint_state.nt_H[i_b, i_d1, i_d2] = (
                    constraint_state.nt_H[i_b, i_d1, i_d2] + rigid_global_info.mass_mat[i_d1, i_d2, i_b]
                )


@qd.func
def func_hessian_direct_tiled(
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
):
    """Compute the Hessian matrix `H = M + J.T @ D @ J of the optimization problem for all environment at once.

    This implementation is specialized for GPU backend and highly optimized for it using shared memory and cooperative
    threading.

    Under the hood, it implements a square-block matrix partitioned production algorithm to support arbitrary matrix
    sizes because shared memory storage is limited to 48kB. It boils down to classical matrix production if the entire
    optimization problem fits in a single block, i.e. n_constraints <= 32 and n_dofs <= 64.

    Note that only the lower triangular part will be updated for efficiency, because the Hessian matrix is symmetric.
    """
    _B = constraint_state.grad.shape[1]
    n_dofs = constraint_state.nt_H.shape[1]

    # Performance is optimal for BLOCK_DIM = MAX_DOFS_PER_BLOCK = 64
    BLOCK_DIM = qd.static(64)
    MAX_DOFS_PER_BLOCK = qd.static(64)
    MAX_CONSTRAINTS_PER_BLOCK = qd.static(32)

    n_lower_tri = n_dofs * (n_dofs + 1) // 2

    # FIXME: Adding `serialize=False` is causing sync failing for some reason...
    # TODO: Consider moving `H += M` in a dedicated CUDA kernel. It should be both simpler and faster.
    qd.loop_config(block_dim=BLOCK_DIM)
    for i in range(_B * BLOCK_DIM):
        tid = i % BLOCK_DIM
        i_b = i // BLOCK_DIM
        if i_b >= _B:
            continue
        if constraint_state.n_constraints[i_b] == 0 or not constraint_state.improved[i_b]:
            continue

        jac_row = qd.simt.block.SharedArray((MAX_CONSTRAINTS_PER_BLOCK, MAX_DOFS_PER_BLOCK), gs.qd_float)
        jac_col = qd.simt.block.SharedArray((MAX_CONSTRAINTS_PER_BLOCK, MAX_DOFS_PER_BLOCK), gs.qd_float)
        efc_D = qd.simt.block.SharedArray((MAX_CONSTRAINTS_PER_BLOCK,), gs.qd_float)

        # Loop over all the constraints and accumulate their respective contributions to the Hessian matrix
        i_c_start = 0
        n_c = constraint_state.n_constraints[i_b]
        while i_c_start < n_c:
            # Store masked `efc_D` in shared memory for fast access
            i_c_ = tid
            n_conts_tile = qd.min(MAX_CONSTRAINTS_PER_BLOCK, n_c - i_c_start)
            while i_c_ < n_conts_tile:
                efc_D[i_c_] = (
                    constraint_state.efc_D[i_c_start + i_c_, i_b] * constraint_state.active[i_c_start + i_c_, i_b]
                )
                i_c_ = i_c_ + BLOCK_DIM

            # Loop over all row blocks of the hessian matrix
            i_d1_start = 0
            while i_d1_start < n_dofs:
                n_dofs_tile_row = qd.min(MAX_DOFS_PER_BLOCK, n_dofs - i_d1_start)

                # Copy Jacobian row blocks to shared memory for fast access
                i_c_ = tid
                while i_c_ < n_conts_tile:
                    for i_d_ in range(n_dofs_tile_row):
                        jac_row[i_c_, i_d_] = constraint_state.jac[i_c_start + i_c_, i_d1_start + i_d_, i_b]
                    i_c_ = i_c_ + BLOCK_DIM
                qd.simt.block.sync()

                # Loop over all column blocks of the hessian matrix
                i_d2_start = 0
                while i_d2_start <= i_d1_start:
                    n_dofs_tile_col = qd.min(MAX_DOFS_PER_BLOCK, n_dofs - i_d2_start)
                    is_diag_tile = i_d1_start == i_d2_start

                    # Copy Jacobian column block to shared memory for fast access if necessary, i.e. the hessian block
                    # being considered is a diagonal block.
                    if not is_diag_tile:
                        i_c_ = tid
                        while i_c_ < n_conts_tile:
                            for i_d_ in range(n_dofs_tile_col):
                                jac_col[i_c_, i_d_] = constraint_state.jac[i_c_start + i_c_, i_d2_start + i_d_, i_b]
                            i_c_ = i_c_ + BLOCK_DIM
                        qd.simt.block.sync()

                    # Compute `H += J.T @ D @ J` for a single Hessian block
                    pid = tid
                    numel = n_dofs_tile_row * n_dofs_tile_col
                    while pid < numel:
                        i_d1_ = pid // n_dofs_tile_col
                        i_d2_ = pid % n_dofs_tile_col
                        i_d1 = i_d1_ + i_d1_start
                        i_d2 = i_d2_ + i_d2_start
                        if i_d1 >= i_d2:
                            coef = gs.qd_float(0.0)
                            if i_c_start == 0:
                                coef = rigid_global_info.mass_mat[i_d1, i_d2, i_b]
                            if is_diag_tile:
                                for j_c_ in range(n_conts_tile):
                                    coef = coef + jac_row[j_c_, i_d1_] * jac_row[j_c_, i_d2_] * efc_D[j_c_]
                            else:
                                for j_c_ in range(n_conts_tile):
                                    coef = coef + jac_row[j_c_, i_d1_] * jac_col[j_c_, i_d2_] * efc_D[j_c_]
                            if i_c_start == 0:
                                constraint_state.nt_H[i_b, i_d1, i_d2] = coef
                            else:
                                constraint_state.nt_H[i_b, i_d1, i_d2] = constraint_state.nt_H[i_b, i_d1, i_d2] + coef
                        pid = pid + BLOCK_DIM
                    qd.simt.block.sync()

                    i_d2_start = i_d2_start + MAX_DOFS_PER_BLOCK
                i_d1_start = i_d1_start + MAX_DOFS_PER_BLOCK
            i_c_start = i_c_start + MAX_CONSTRAINTS_PER_BLOCK

        # If there is no constraint, the main loop will be completely skipped, which means that the Hessian matrix must
        # be updated separately to store the lower triangular part  of the mass matrix M.
        if n_c == 0:
            i_pair = tid
            while i_pair < n_lower_tri:
                i_d1 = qd.cast(qd.floor((-1.0 + qd.sqrt(1.0 + 8.0 * i_pair)) / 2.0), gs.qd_int)
                i_d2 = i_pair - i_d1 * (i_d1 + 1) // 2
                constraint_state.nt_H[i_b, i_d1, i_d2] = rigid_global_info.mass_mat[i_d1, i_d2, i_b]
                i_pair = i_pair + BLOCK_DIM


@qd.func
def func_cholesky_factor_direct_batch(
    i_b,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
):
    """Compute the Cholesky factorization L of the Hessian matrix H = L @ L.T for all environments at once.

    Beware the Hessian matrix is re-purposed to store its Cholesky factorization to spare memory resources.

    Note that only the lower triangular part will be updated for efficiency, because the Hessian matrix is symmetric.
    """
    EPS = rigid_global_info.EPS[None]

    n_dofs = constraint_state.nt_H.shape[1]

    for i_d in range(n_dofs):
        tmp = constraint_state.nt_H[i_b, i_d, i_d]
        for j_d in range(i_d):
            tmp = tmp - constraint_state.nt_H[i_b, i_d, j_d] ** 2
        constraint_state.nt_H[i_b, i_d, i_d] = qd.sqrt(qd.max(tmp, EPS))

        tmp = 1.0 / constraint_state.nt_H[i_b, i_d, i_d]
        for j_d in range(i_d + 1, n_dofs):
            dot = gs.qd_float(0.0)
            for k_d in range(i_d):
                dot = dot + constraint_state.nt_H[i_b, j_d, k_d] * constraint_state.nt_H[i_b, i_d, k_d]
            constraint_state.nt_H[i_b, j_d, i_d] = (constraint_state.nt_H[i_b, j_d, i_d] - dot) * tmp


@qd.func
def func_cholesky_factor_direct_tiled(
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Compute the Cholesky factorization L of the Hessian matrix H = L @ L.T for a given environment `i_b`.

    This implementation is specialized for GPU backend and highly optimized for it using shared memory and cooperative
    threading. The current implementation only supports n_dofs <= 64 for 64bits precision and n_dofs <= 92 for 32bits
    precision due to shared memory storage being limited to 48kB. Note that the amount of shared memory available is
    hardware-specific, but the 48kB default limit without enabling dedicated GPU context flag is hardware-agnostic on
    modern GPUs.

    Beware the Hessian matrix is re-purposed to store its Cholesky factorization to sparse memory resources.

    Note that only the lower triangular part will be updated for efficiency, because the Hessian matrix is symmetric.
    """
    EPS = rigid_global_info.EPS[None]

    _B = constraint_state.grad.shape[1]
    n_dofs = constraint_state.nt_H.shape[1]

    # Performance is optimal for BLOCK_DIM = 64
    BLOCK_DIM = qd.static(64)
    MAX_DOFS = qd.static(static_rigid_sim_config.tiled_n_dofs)

    n_lower_tri = n_dofs * (n_dofs + 1) // 2

    qd.loop_config(block_dim=BLOCK_DIM)
    for i in range(_B * BLOCK_DIM):
        tid = i % BLOCK_DIM
        i_b = i // BLOCK_DIM
        if i_b >= _B:
            continue
        if constraint_state.n_constraints[i_b] == 0 or not constraint_state.improved[i_b]:
            continue

        # Padding +1 to avoid memory bank conflicts that would cause access serialization
        H = qd.simt.block.SharedArray((MAX_DOFS, MAX_DOFS + 1), gs.qd_float)

        # Copy the lower triangular part of the entire Hessian matrix to shared memory for efficiency
        i_pair = tid
        while i_pair < n_lower_tri:
            i_d1 = qd.cast((qd.sqrt(8 * i_pair + 1) - 1) // 2, qd.i32)
            i_d2 = i_pair - i_d1 * (i_d1 + 1) // 2
            H[i_d1, i_d2] = constraint_state.nt_H[i_b, i_d1, i_d2]
            i_pair = i_pair + BLOCK_DIM
        qd.simt.block.sync()

        # Loop over all columns sequentially, which is an integral part of Cholesky-Crout algorithm and cannot be
        # avoided.
        for i_d in range(n_dofs):
            # Compute the diagonal of the Cholesky factor L for the column i being considered, ie
            # L_{i,i} = sqrt(A_{i,i} - sum_{j=1}^{i-1}(L_{i,j} ** 2 ))
            if tid == 0:
                tmp = H[i_d, i_d]
                for j_d in range(i_d):
                    tmp = tmp - H[i_d, j_d] ** 2
                H[i_d, i_d] = qd.sqrt(qd.max(tmp, EPS))
            qd.simt.block.sync()

            # Compute all the off-diagonal terms of the Cholesky factor L for the column i being considered, ie
            # L_{j,i} = 1 / L_{i,i} (A_{j,i} - sum_{k=1}^{i-1}(L_{j,k} L_{i,k}), for j > i
            inv_diag = 1.0 / H[i_d, i_d]
            j_d = i_d + 1 + tid
            while j_d < n_dofs:
                dot = gs.qd_float(0.0)
                for k_d in range(i_d):
                    dot = dot + H[j_d, k_d] * H[i_d, k_d]
                H[j_d, i_d] = (H[j_d, i_d] - dot) * inv_diag
                j_d = j_d + BLOCK_DIM
            qd.simt.block.sync()

        # Copy the final result back from shared memory, only considered the lower triangular part
        i_pair = tid
        while i_pair < n_lower_tri:
            i_d1 = qd.cast((qd.sqrt(8 * i_pair + 1) - 1) // 2, qd.i32)
            i_d2 = i_pair - i_d1 * (i_d1 + 1) // 2
            constraint_state.nt_H[i_b, i_d1, i_d2] = H[i_d1, i_d2]
            i_pair = i_pair + BLOCK_DIM


@qd.func
def func_hessian_and_cholesky_factor_direct_batch(
    i_b,
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    func_hessian_direct_batch(i_b, entities_info, constraint_state, rigid_global_info, static_rigid_sim_config)
    func_cholesky_factor_direct_batch(i_b, constraint_state, rigid_global_info)


@qd.func
def func_hessian_and_cholesky_factor_direct(
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """
    Unified implementation of Hessian matrix computation with Cholesky factorization optimized for both CPU and GPU
    backends.

    The tiled optimization is only supported on GPU backend and specifically optimized for it, falling back to the
    classical batched implementation when running on CPU backend.

    Note that the sparse implementation has not been optimized using tiling for now, mainly it is largely designed to
    target CPU rather than GPU backend. In any event, its advantage over the dense implementation is still not clear.
    """
    _B = constraint_state.jac.shape[2]

    if qd.static(static_rigid_sim_config.backend == gs.cpu or static_rigid_sim_config.sparse_solve):
        # CPU
        qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
        for i_b in range(_B):
            func_hessian_and_cholesky_factor_direct_batch(
                i_b,
                entities_info=entities_info,
                rigid_global_info=rigid_global_info,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )
    else:
        # GPU
        func_hessian_direct_tiled(constraint_state, rigid_global_info)

        if qd.static(static_rigid_sim_config.enable_tiled_cholesky_hessian):
            func_cholesky_factor_direct_tiled(constraint_state, rigid_global_info, static_rigid_sim_config)
        else:
            qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
            for i_b in range(_B):
                func_cholesky_factor_direct_batch(i_b, constraint_state, rigid_global_info)


@qd.func
def func_build_changed_constraint_list(
    i_b,
    constraint_state: array_class.ConstraintState,
):
    """Build a compact list of constraint indices whose active state changed.

    This reduces GPU thread divergence in the subsequent incremental Cholesky update by ensuring threads iterate
    only over constraints that need processing, rather than branching over all constraints.
    """
    n_changed = 0
    for i_c in range(constraint_state.n_constraints[i_b]):
        if constraint_state.active[i_c, i_b] ^ constraint_state.prev_active[i_c, i_b]:
            constraint_state.incr_changed_idx[n_changed, i_b] = i_c
            n_changed += 1
    constraint_state.incr_n_changed[i_b] = n_changed


@qd.func
def func_hessian_and_cholesky_factor_incremental_dense_batch(
    i_b,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
) -> bool:
    EPS = rigid_global_info.EPS[None]

    n_dofs = constraint_state.nt_H.shape[1]

    is_degenerated = False
    for idx in range(constraint_state.incr_n_changed[i_b]):
        i_c = constraint_state.incr_changed_idx[idx, i_b]
        sign = 1.0 if constraint_state.active[i_c, i_b] else -1.0
        efc_D_sqrt = qd.sqrt(constraint_state.efc_D[i_c, i_b])

        for i_d in range(n_dofs):
            constraint_state.nt_vec[i_d, i_b] = constraint_state.jac[i_c, i_d, i_b] * efc_D_sqrt

        for k in range(n_dofs):
            if qd.abs(constraint_state.nt_vec[k, i_b]) > EPS:
                Lkk = constraint_state.nt_H[i_b, k, k]
                tmp = Lkk**2 + sign * constraint_state.nt_vec[k, i_b] ** 2
                if tmp < EPS:
                    is_degenerated = True
                    break
                r = qd.sqrt(tmp)
                c = r / Lkk
                cinv = 1 / c
                s = constraint_state.nt_vec[k, i_b] / Lkk
                constraint_state.nt_H[i_b, k, k] = r
                for i in range(k + 1, n_dofs):
                    constraint_state.nt_H[i_b, i, k] = (
                        constraint_state.nt_H[i_b, i, k] + s * constraint_state.nt_vec[i, i_b] * sign
                    ) * cinv

                for i in range(k + 1, n_dofs):
                    constraint_state.nt_vec[i, i_b] = (
                        constraint_state.nt_vec[i, i_b] * c - s * constraint_state.nt_H[i_b, i, k]
                    )

    return is_degenerated


@qd.func
def func_hessian_and_cholesky_factor_incremental_sparse_batch(
    i_b,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
) -> bool:
    EPS = rigid_global_info.EPS[None]

    is_degenerated = False
    for idx in range(constraint_state.incr_n_changed[i_b]):
        i_c = constraint_state.incr_changed_idx[idx, i_b]
        sign = 1.0 if constraint_state.active[i_c, i_b] else -1.0
        efc_D_sqrt = qd.sqrt(constraint_state.efc_D[i_c, i_b])

        for i_d_ in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
            i_d = constraint_state.jac_relevant_dofs[i_c, i_d_, i_b]
            constraint_state.nt_vec[i_d, i_b] = constraint_state.jac[i_c, i_d, i_b] * efc_D_sqrt

        for k_ in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
            k = constraint_state.jac_relevant_dofs[i_c, k_, i_b]
            Lkk = constraint_state.nt_H[i_b, k, k]
            tmp = Lkk**2 + sign * constraint_state.nt_vec[k, i_b] ** 2
            if tmp < EPS:
                is_degenerated = True
                break
            r = qd.sqrt(tmp)
            c = r / Lkk
            cinv = 1 / c
            s = constraint_state.nt_vec[k, i_b] / Lkk
            constraint_state.nt_H[i_b, k, k] = r
            for i_ in range(k_):
                i = constraint_state.jac_relevant_dofs[i_c, i_, i_b]  # i is strictly > k
                constraint_state.nt_H[i_b, i, k] = (
                    constraint_state.nt_H[i_b, i, k] + s * constraint_state.nt_vec[i, i_b] * sign
                ) * cinv

            for i_ in range(k_):
                i = constraint_state.jac_relevant_dofs[i_c, i_, i_b]  # i is strictly > k
                constraint_state.nt_vec[i, i_b] = (
                    constraint_state.nt_vec[i, i_b] * c - s * constraint_state.nt_H[i_b, i, k]
                )

    return is_degenerated


@qd.func
def func_hessian_and_cholesky_factor_incremental_batch(
    i_b,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
) -> bool:
    is_degenerated = False
    if qd.static(static_rigid_sim_config.sparse_solve):
        is_degenerated = func_hessian_and_cholesky_factor_incremental_sparse_batch(
            i_b, constraint_state, rigid_global_info
        )
    else:
        is_degenerated = func_hessian_and_cholesky_factor_incremental_dense_batch(
            i_b, constraint_state, rigid_global_info
        )
    return is_degenerated


# ======================================== Cholesky Factorization and Solving =========================================


@qd.func
def func_cholesky_solve_batch(
    i_b,
    constraint_state: array_class.ConstraintState,
):
    n_dofs = constraint_state.Mgrad.shape[0]

    for i_d in range(n_dofs):
        curr_out = constraint_state.grad[i_d, i_b]
        for j_d in range(i_d):
            curr_out = curr_out - constraint_state.nt_H[i_b, i_d, j_d] * constraint_state.Mgrad[j_d, i_b]
        constraint_state.Mgrad[i_d, i_b] = curr_out / constraint_state.nt_H[i_b, i_d, i_d]

    for i_d_ in range(n_dofs):
        i_d = n_dofs - 1 - i_d_
        curr_out = constraint_state.Mgrad[i_d, i_b]
        for j_d in range(i_d + 1, n_dofs):
            curr_out = curr_out - constraint_state.nt_H[i_b, j_d, i_d] * constraint_state.Mgrad[j_d, i_b]
        constraint_state.Mgrad[i_d, i_b] = curr_out / constraint_state.nt_H[i_b, i_d, i_d]


@qd.func
def func_cholesky_solve_tiled(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """Compute the solution of H @ grad = Mgrad st H = L @ L.T for all environments at once.

    This implementation is specialized for GPU backend and highly optimized for it using shared memory and cooperative
    threading. The current implementation only supports n_dofs <= 64 for 64bits precision and n_dofs <= 92 for 32bits
    precision. See `func_cholesky_factor_direct_tiled` documentation for details.

    Note that this implementation leverages warp-level reduction whenever supported, a generic fallback otherwise. At
    the time of writing, all warp-level intrinsics in `qd.simt.warp` sub-module are CUDA-specific, of which only
    `shfl_down_f32` is being used here. Although some of these warp-level instrinsics are supposed to be supported by
    all major GPUs if not all (incl. Apple Silicon chips under naming 'SIMD-group'), Quadrants does not provide a unified
    API for it yet. As a result, warp-level intrinsics are currently disabled if not running on CUDA backend. On top of
    that, most if not all, Warp-level intrinsics are only supporting 32bits precision.
    """
    # Performance is optimal for BLOCK_DIM = 64
    BLOCK_DIM = qd.static(64)
    MAX_DOFS = qd.static(static_rigid_sim_config.tiled_n_dofs)
    ENABLE_WARP_REDUCTION = qd.static(static_rigid_sim_config.backend == gs.cuda and gs.qd_float == qd.f32)
    WARP_SIZE = qd.static(32)
    NUM_WARPS = qd.static(BLOCK_DIM // WARP_SIZE)

    _B = constraint_state.jac.shape[2]
    n_dofs = constraint_state.jac.shape[1]
    n_dofs_2 = n_dofs**2

    qd.loop_config(block_dim=BLOCK_DIM)
    for i in range(_B * BLOCK_DIM):
        tid = i % BLOCK_DIM
        i_b = i // BLOCK_DIM
        warp_id = tid // WARP_SIZE
        lane_id = tid % WARP_SIZE
        if i_b >= _B:
            continue

        H = qd.simt.block.SharedArray((MAX_DOFS, MAX_DOFS + 1), gs.qd_float)
        v = qd.simt.block.SharedArray((MAX_DOFS,), gs.qd_float)
        partial = qd.simt.block.SharedArray(
            (NUM_WARPS if qd.static(ENABLE_WARP_REDUCTION) else BLOCK_DIM,), gs.qd_float
        )

        # Copy the lower triangular part of the entire Hessian matrix to shared memory for efficiency
        i_flat = tid
        while i_flat < n_dofs_2:
            i_d1 = i_flat // n_dofs
            i_d2 = i_flat % n_dofs
            if i_d2 <= i_d1:
                H[i_d1, i_d2] = constraint_state.nt_H[i_b, i_d1, i_d2]
            i_flat = i_flat + BLOCK_DIM

        # Copy the gradient to shared memory for efficiency
        k_d = tid
        while k_d < n_dofs:
            v[k_d] = constraint_state.grad[k_d, i_b]
            k_d = k_d + BLOCK_DIM
        qd.simt.block.sync()

        # Step 1: Solve w st. L^T @ w = y
        for i_d in range(n_dofs):
            dot = gs.qd_float(0.0)
            j_d = tid
            while j_d < i_d:
                dot = dot + H[i_d, j_d] * v[j_d]
                j_d = j_d + BLOCK_DIM
            if qd.static(ENABLE_WARP_REDUCTION):
                for offset in qd.static([16, 8, 4, 2, 1]):
                    dot = dot + qd.simt.warp.shfl_down_f32(qd.u32(0xFFFFFFFF), dot, offset)
                if lane_id == 0:
                    partial[warp_id] = dot
            else:
                partial[tid] = dot
            qd.simt.block.sync()

            if tid == 0:
                total = gs.qd_float(0.0)
                for k in qd.static(range(NUM_WARPS)) if qd.static(ENABLE_WARP_REDUCTION) else range(BLOCK_DIM):
                    total = total + partial[k]
                v[i_d] = (v[i_d] - total) / H[i_d, i_d]
            qd.simt.block.sync()

        # Step 2: Solve x st. L @ x = z
        for i_d_ in range(n_dofs):
            i_d = n_dofs - 1 - i_d_
            dot = gs.qd_float(0.0)
            j_d = i_d + 1 + tid
            while j_d < n_dofs:
                dot = dot + H[j_d, i_d] * v[j_d]
                j_d = j_d + BLOCK_DIM

            if qd.static(ENABLE_WARP_REDUCTION):
                for offset in qd.static([16, 8, 4, 2, 1]):
                    dot = dot + qd.simt.warp.shfl_down_f32(qd.u32(0xFFFFFFFF), dot, offset)
                if lane_id == 0:
                    partial[warp_id] = dot
            else:
                partial[tid] = dot
            qd.simt.block.sync()

            if tid == 0:
                total = gs.qd_float(0.0)
                for k in qd.static(range(NUM_WARPS)) if qd.static(ENABLE_WARP_REDUCTION) else range(BLOCK_DIM):
                    total = total + partial[k]
                v[i_d] = (v[i_d] - total) / H[i_d, i_d]
            qd.simt.block.sync()

        # Copy the final result back from shared memory
        k_d = tid
        while k_d < n_dofs:
            constraint_state.Mgrad[k_d, i_b] = v[k_d]
            k_d = k_d + BLOCK_DIM


# =====================================================================================================================
# ==================================================== Linesearch =====================================================
# =====================================================================================================================


@qd.func
def func_ls_init_and_eval_p0_opt(
    i_b,
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Fused linesearch initialization and first evaluation point (alpha=0) for a single environment.

    Merges init (computing mv, jv, quad_gauss) and alpha=0 evaluation into a single pass, and pre-computes eq_sum
    (the summed quadratic coefficients for always-active equality constraints) for reuse by subsequent evaluation calls.

    Bandwidth optimization: quad coefficients (D*Ja*Ja, D*jv*Ja, D*jv*jv) are recomputed on the fly from Jaref, jv,
    and efc_D (~8 FLOPs per constraint) instead of being precomputed and stored to a separate quad array. At 0.2%
    compute utilization (0.40 FLOPs/byte, 147x below roofline), this trades negligible compute for eliminating 3 global
    memory writes per constraint during init and 3 reads per constraint in every subsequent evaluation call — a 40%
    bandwidth reduction for contacts (5→3 loads) and 29% for friction (7→5 loads) in the hottest loop."""
    n_dofs = constraint_state.search.shape[0]
    n_entities = entities_info.dof_start.shape[0]
    ne = constraint_state.n_constraints_equality[i_b]
    nef = ne + constraint_state.n_constraints_frictionloss[i_b]
    n_con = constraint_state.n_constraints[i_b]

    # -- mv and jv (same as original func_ls_init) --
    for i_e in range(n_entities):
        for i_d1 in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
            mv = gs.qd_float(0.0)
            for i_d2 in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                mv = mv + rigid_global_info.mass_mat[i_d1, i_d2, i_b] * constraint_state.search[i_d2, i_b]
            constraint_state.mv[i_d1, i_b] = mv

    for i_c in range(n_con):
        jv = gs.qd_float(0.0)
        if qd.static(static_rigid_sim_config.sparse_solve):
            for i_d_ in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                i_d = constraint_state.jac_relevant_dofs[i_c, i_d_, i_b]
                jv = jv + constraint_state.jac[i_c, i_d, i_b] * constraint_state.search[i_d, i_b]
        else:
            for i_d in range(n_dofs):
                jv = jv + constraint_state.jac[i_c, i_d, i_b] * constraint_state.search[i_d, i_b]
        constraint_state.jv[i_c, i_b] = jv

    # -- quad_gauss (same as original func_ls_init) --
    quad_gauss_1 = gs.qd_float(0.0)
    quad_gauss_2 = gs.qd_float(0.0)
    for i_d in range(n_dofs):
        quad_gauss_1 = quad_gauss_1 + (
            constraint_state.search[i_d, i_b] * constraint_state.Ma[i_d, i_b]
            - constraint_state.search[i_d, i_b] * dofs_state.force[i_d, i_b]
        )
        quad_gauss_2 = quad_gauss_2 + 0.5 * constraint_state.search[i_d, i_b] * constraint_state.mv[i_d, i_b]
    constraint_state.quad_gauss[0, i_b] = constraint_state.gauss[i_b]
    constraint_state.quad_gauss[1, i_b] = quad_gauss_1
    constraint_state.quad_gauss[2, i_b] = quad_gauss_2

    # -- Compute quad per constraint and accumulate by type --
    quad_total_0 = constraint_state.gauss[i_b]
    quad_total_1 = quad_gauss_1
    quad_total_2 = quad_gauss_2
    eq_sum_0 = gs.qd_float(0.0)
    eq_sum_1 = gs.qd_float(0.0)
    eq_sum_2 = gs.qd_float(0.0)

    # Recompute quad on the fly from Jaref, jv, efc_D — avoids writing/reading the quad array entirely.
    # 3 loads per constraint (Jaref, jv, D) + ~8 FLOPs, vs 3 writes + 3 reads through global memory.
    for i_c in range(n_con):
        Jaref_c = constraint_state.Jaref[i_c, i_b]
        jv_c = constraint_state.jv[i_c, i_b]
        D = constraint_state.efc_D[i_c, i_b]
        qf_0 = D * (0.5 * Jaref_c * Jaref_c)
        qf_1 = D * (jv_c * Jaref_c)
        qf_2 = D * (0.5 * jv_c * jv_c)

        if i_c < ne:
            # Equality: always active
            eq_sum_0 = eq_sum_0 + qf_0
            eq_sum_1 = eq_sum_1 + qf_1
            eq_sum_2 = eq_sum_2 + qf_2
            quad_total_0 = quad_total_0 + qf_0
            quad_total_1 = quad_total_1 + qf_1
            quad_total_2 = quad_total_2 + qf_2
        elif i_c < nef:
            # Friction: check linear regime at x=Jaref (alpha=0)
            f = constraint_state.efc_frictionloss[i_c, i_b]
            r = constraint_state.diag[i_c, i_b]
            rf = r * f
            linear_neg = Jaref_c <= -rf
            linear_pos = Jaref_c >= rf
            if linear_neg or linear_pos:
                qf_0 = linear_neg * f * (-0.5 * rf - Jaref_c) + linear_pos * f * (-0.5 * rf + Jaref_c)
                qf_1 = linear_neg * (-f * jv_c) + linear_pos * (f * jv_c)
                qf_2 = 0.0
            quad_total_0 = quad_total_0 + qf_0
            quad_total_1 = quad_total_1 + qf_1
            quad_total_2 = quad_total_2 + qf_2
        else:
            # Contact: check Jaref < 0
            active = Jaref_c < 0
            quad_total_0 = quad_total_0 + qf_0 * active
            quad_total_1 = quad_total_1 + qf_1 * active
            quad_total_2 = quad_total_2 + qf_2 * active

    # Write eq_sum to global for subsequent calls
    constraint_state.eq_sum[0, i_b] = eq_sum_0
    constraint_state.eq_sum[1, i_b] = eq_sum_1
    constraint_state.eq_sum[2, i_b] = eq_sum_2

    # Return p0 result (alpha=0)
    cost = quad_total_0
    grad = quad_total_1
    hess = 2 * quad_total_2
    if hess <= 0.0:
        hess = rigid_global_info.EPS[None]

    constraint_state.ls_it[i_b] = 1

    return gs.qd_float(0.0), cost, grad, hess


@qd.func
def func_ls_point_fn_opt(
    i_b,
    alpha,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
):
    """Evaluate linesearch cost, gradient, and curvature at a single candidate alpha.

    Iterates over only friction and contact constraints — equality constraints are skipped by initializing accumulators
    from quad_gauss + eq_sum (pre-computed during init).

    Quad coefficients are recomputed on the fly from Jaref, jv, efc_D rather than read from a precomputed quad array.
    This reduces per-constraint loads from 5 to 3 (contacts) and 7 to 5 (friction), a 40%/29% bandwidth reduction.
    The ~8 FLOPs of recomputation per constraint are almost free."""
    ne = constraint_state.n_constraints_equality[i_b]
    nef = ne + constraint_state.n_constraints_frictionloss[i_b]
    n_con = constraint_state.n_constraints[i_b]

    # Start from quad_gauss + eq_sum (skips ne equality constraints)
    quad_total_0 = constraint_state.quad_gauss[0, i_b] + constraint_state.eq_sum[0, i_b]
    quad_total_1 = constraint_state.quad_gauss[1, i_b] + constraint_state.eq_sum[1, i_b]
    quad_total_2 = constraint_state.quad_gauss[2, i_b] + constraint_state.eq_sum[2, i_b]

    # Friction constraints [ne, nef): 5 loads (Jaref, jv, D, f, diag) + recompute quad
    for i_c in range(ne, nef):
        Jaref_c = constraint_state.Jaref[i_c, i_b]
        jv_c = constraint_state.jv[i_c, i_b]
        D = constraint_state.efc_D[i_c, i_b]
        f = constraint_state.efc_frictionloss[i_c, i_b]
        r = constraint_state.diag[i_c, i_b]
        qf_0 = D * (0.5 * Jaref_c * Jaref_c)
        qf_1 = D * (jv_c * Jaref_c)
        qf_2 = D * (0.5 * jv_c * jv_c)
        x = Jaref_c + alpha * jv_c
        rf = r * f
        linear_neg = x <= -rf
        linear_pos = x >= rf
        if linear_neg or linear_pos:
            qf_0 = linear_neg * f * (-0.5 * rf - Jaref_c) + linear_pos * f * (-0.5 * rf + Jaref_c)
            qf_1 = linear_neg * (-f * jv_c) + linear_pos * (f * jv_c)
            qf_2 = 0.0
        quad_total_0 = quad_total_0 + qf_0
        quad_total_1 = quad_total_1 + qf_1
        quad_total_2 = quad_total_2 + qf_2

    # Contact constraints [nef, n_con): 3 loads (Jaref, jv, D) + recompute quad
    for i_c in range(nef, n_con):
        Jaref_c = constraint_state.Jaref[i_c, i_b]
        jv_c = constraint_state.jv[i_c, i_b]
        D = constraint_state.efc_D[i_c, i_b]
        x = Jaref_c + alpha * jv_c
        active = x < 0
        qf_0 = D * (0.5 * Jaref_c * Jaref_c)
        qf_1 = D * (jv_c * Jaref_c)
        qf_2 = D * (0.5 * jv_c * jv_c)
        quad_total_0 = quad_total_0 + qf_0 * active
        quad_total_1 = quad_total_1 + qf_1 * active
        quad_total_2 = quad_total_2 + qf_2 * active

    cost = alpha * alpha * quad_total_2 + alpha * quad_total_1 + quad_total_0
    grad = 2 * alpha * quad_total_2 + quad_total_1
    hess = 2 * quad_total_2
    if hess <= 0.0:
        hess = rigid_global_info.EPS[None]

    constraint_state.ls_it[i_b] = constraint_state.ls_it[i_b] + 1

    return alpha, cost, grad, hess


@qd.func
def func_ls_point_fn_3alphas_opt(
    i_b,
    alpha_0,
    alpha_1,
    alpha_2,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
):
    """Evaluate linesearch cost, gradient, and curvature at three candidate alphas in a single constraint loop pass.

    Batches three candidate step sizes into one loop, amortizing per-constraint loads (Jaref, jv, efc_D, etc.) across
    all three evaluations. Equality constraints are skipped via quad_gauss + eq_sum.

    Quad coefficients are recomputed on the fly from Jaref, jv, efc_D — same bandwidth optimization as
    func_ls_point_fn_opt (3 loads per contact instead of 5, 5 per friction instead of 7). Combined with 3-alpha
    batching, each constraint's data is loaded once from global memory and reused for 3 alpha evaluations."""
    ne = constraint_state.n_constraints_equality[i_b]
    nef = ne + constraint_state.n_constraints_frictionloss[i_b]
    n_con = constraint_state.n_constraints[i_b]

    # Start from quad_gauss + eq_sum for all 3
    base_0 = constraint_state.quad_gauss[0, i_b] + constraint_state.eq_sum[0, i_b]
    base_1 = constraint_state.quad_gauss[1, i_b] + constraint_state.eq_sum[1, i_b]
    base_2 = constraint_state.quad_gauss[2, i_b] + constraint_state.eq_sum[2, i_b]

    t0_0, t0_1, t0_2 = base_0, base_1, base_2
    t1_0, t1_1, t1_2 = base_0, base_1, base_2
    t2_0, t2_1, t2_2 = base_0, base_1, base_2

    # Friction constraints [ne, nef): 5 loads (Jaref, jv, D, f, diag) + recompute quad, eval 3 alphas
    for i_c in range(ne, nef):
        Jaref_c = constraint_state.Jaref[i_c, i_b]
        jv_c = constraint_state.jv[i_c, i_b]
        D = constraint_state.efc_D[i_c, i_b]
        f = constraint_state.efc_frictionloss[i_c, i_b]
        r = constraint_state.diag[i_c, i_b]
        qf_0 = D * (0.5 * Jaref_c * Jaref_c)
        qf_1 = D * (jv_c * Jaref_c)
        qf_2 = D * (0.5 * jv_c * jv_c)
        rf = r * f

        x0 = Jaref_c + alpha_0 * jv_c
        ln0 = x0 <= -rf
        lp0 = x0 >= rf
        a0_qf_0, a0_qf_1, a0_qf_2 = qf_0, qf_1, qf_2
        if ln0 or lp0:
            a0_qf_0 = ln0 * f * (-0.5 * rf - Jaref_c) + lp0 * f * (-0.5 * rf + Jaref_c)
            a0_qf_1 = ln0 * (-f * jv_c) + lp0 * (f * jv_c)
            a0_qf_2 = 0.0
        t0_0 = t0_0 + a0_qf_0
        t0_1 = t0_1 + a0_qf_1
        t0_2 = t0_2 + a0_qf_2

        x1 = Jaref_c + alpha_1 * jv_c
        ln1 = x1 <= -rf
        lp1 = x1 >= rf
        a1_qf_0, a1_qf_1, a1_qf_2 = qf_0, qf_1, qf_2
        if ln1 or lp1:
            a1_qf_0 = ln1 * f * (-0.5 * rf - Jaref_c) + lp1 * f * (-0.5 * rf + Jaref_c)
            a1_qf_1 = ln1 * (-f * jv_c) + lp1 * (f * jv_c)
            a1_qf_2 = 0.0
        t1_0 = t1_0 + a1_qf_0
        t1_1 = t1_1 + a1_qf_1
        t1_2 = t1_2 + a1_qf_2

        x2 = Jaref_c + alpha_2 * jv_c
        ln2 = x2 <= -rf
        lp2 = x2 >= rf
        a2_qf_0, a2_qf_1, a2_qf_2 = qf_0, qf_1, qf_2
        if ln2 or lp2:
            a2_qf_0 = ln2 * f * (-0.5 * rf - Jaref_c) + lp2 * f * (-0.5 * rf + Jaref_c)
            a2_qf_1 = ln2 * (-f * jv_c) + lp2 * (f * jv_c)
            a2_qf_2 = 0.0
        t2_0 = t2_0 + a2_qf_0
        t2_1 = t2_1 + a2_qf_1
        t2_2 = t2_2 + a2_qf_2

    # Contact constraints [nef, n_con): 3 loads (Jaref, jv, D) + recompute quad, eval 3 alphas
    for i_c in range(nef, n_con):
        Jaref_c = constraint_state.Jaref[i_c, i_b]
        jv_c = constraint_state.jv[i_c, i_b]
        D = constraint_state.efc_D[i_c, i_b]
        qf_0 = D * (0.5 * Jaref_c * Jaref_c)
        qf_1 = D * (jv_c * Jaref_c)
        qf_2 = D * (0.5 * jv_c * jv_c)

        x0 = Jaref_c + alpha_0 * jv_c
        x1 = Jaref_c + alpha_1 * jv_c
        x2 = Jaref_c + alpha_2 * jv_c
        act0 = gs.qd_bool(x0 < 0)
        act1 = gs.qd_bool(x1 < 0)
        act2 = gs.qd_bool(x2 < 0)
        t0_0 = t0_0 + qf_0 * act0
        t0_1 = t0_1 + qf_1 * act0
        t0_2 = t0_2 + qf_2 * act0
        t1_0 = t1_0 + qf_0 * act1
        t1_1 = t1_1 + qf_1 * act1
        t1_2 = t1_2 + qf_2 * act1
        t2_0 = t2_0 + qf_0 * act2
        t2_1 = t2_1 + qf_1 * act2
        t2_2 = t2_2 + qf_2 * act2

    EPS = rigid_global_info.EPS[None]

    # Evaluate cost, gradient (1st derivative), and hessian (2nd derivative) for each alpha
    cost_0 = alpha_0 * alpha_0 * t0_2 + alpha_0 * t0_1 + t0_0
    grad_0 = 2 * alpha_0 * t0_2 + t0_1
    hess_0 = 2 * t0_2
    if hess_0 <= 0.0:
        hess_0 = EPS

    cost_1 = alpha_1 * alpha_1 * t1_2 + alpha_1 * t1_1 + t1_0
    grad_1 = 2 * alpha_1 * t1_2 + t1_1
    hess_1 = 2 * t1_2
    if hess_1 <= 0.0:
        hess_1 = EPS

    cost_2 = alpha_2 * alpha_2 * t2_2 + alpha_2 * t2_1 + t2_0
    grad_2 = 2 * alpha_2 * t2_2 + t2_1
    hess_2 = 2 * t2_2
    if hess_2 <= 0.0:
        hess_2 = EPS

    constraint_state.ls_it[i_b] = constraint_state.ls_it[i_b] + 3

    costs = qd.Vector([cost_0, cost_1, cost_2])
    grads = qd.Vector([grad_0, grad_1, grad_2])
    hess = qd.Vector([hess_0, hess_1, hess_2])
    return costs, grads, hess


@qd.func
def update_bracket_no_eval_local(
    p_alpha,
    p_cost,
    p_grad,
    p_hess,
    alphas,
    costs,
    grads,
    hess,
):
    """Bracket update using local candidate values. No global memory access or func_ls_point_fn call.

    Args:
        p_alpha, p_cost, p_grad, p_hess: current bracket point (scalar).
        alphas, costs, grads, hess: qd.Vector(3) of candidate values.
    """
    flag = 0

    for i in qd.static(range(3)):
        if p_grad < 0 and grads[i] < 0 and p_grad < grads[i]:
            p_alpha, p_cost, p_grad, p_hess = alphas[i], costs[i], grads[i], hess[i]
            flag = 1
        elif p_grad > 0 and grads[i] > 0 and p_grad > grads[i]:
            p_alpha, p_cost, p_grad, p_hess = alphas[i], costs[i], grads[i], hess[i]
            flag = 2

    p_next_alpha = p_alpha
    if flag > 0:
        p_next_alpha = p_alpha - p_grad / p_hess

    return flag, p_alpha, p_cost, p_grad, p_hess, p_next_alpha


@qd.func
def func_linesearch_and_apply_alpha(
    i_b,
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    alpha = func_linesearch_batch(
        i_b,
        entities_info=entities_info,
        dofs_state=dofs_state,
        rigid_global_info=rigid_global_info,
        constraint_state=constraint_state,
        static_rigid_sim_config=static_rigid_sim_config,
    )
    n_dofs = constraint_state.qacc.shape[0]
    if qd.abs(alpha) < rigid_global_info.EPS[None]:
        constraint_state.improved[i_b] = False
    else:
        # Update qacc and Ma
        # we need alpha for this, so stay in same top level for loop
        # (though we could store alpha in a new tensor of course, if we wanted to split this)
        for i_d in range(n_dofs):
            constraint_state.qacc[i_d, i_b] = (
                constraint_state.qacc[i_d, i_b] + constraint_state.search[i_d, i_b] * alpha
            )
            constraint_state.Ma[i_d, i_b] = constraint_state.Ma[i_d, i_b] + constraint_state.mv[i_d, i_b] * alpha

        # Update Jaref
        for i_c in range(constraint_state.n_constraints[i_b]):
            constraint_state.Jaref[i_c, i_b] = constraint_state.Jaref[i_c, i_b] + constraint_state.jv[i_c, i_b] * alpha


@qd.func
def func_linesearch_batch(
    i_b,
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    n_dofs = constraint_state.search.shape[0]
    ## use adaptive linesearch tolerance
    snorm = gs.qd_float(0.0)
    for jd in range(n_dofs):
        snorm = snorm + constraint_state.search[jd, i_b] ** 2
    snorm = qd.sqrt(snorm)
    scale = rigid_global_info.meaninertia[i_b] * qd.max(1, n_dofs)
    gtol = rigid_global_info.tolerance[None] * rigid_global_info.ls_tolerance[None] * snorm * scale
    constraint_state.gtol[i_b] = gtol

    constraint_state.ls_it[i_b] = 0
    constraint_state.ls_result[i_b] = 0

    res_alpha = gs.qd_float(0.0)
    done = False

    if snorm < rigid_global_info.EPS[None]:
        constraint_state.ls_result[i_b] = 1
        res_alpha = 0.0
    else:
        # Phase 1: Init + p0 + p1
        p0_alpha, p0_cost, p0_deriv_0, p0_deriv_1 = func_ls_init_and_eval_p0_opt(
            i_b,
            entities_info=entities_info,
            dofs_state=dofs_state,
            constraint_state=constraint_state,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )
        p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = func_ls_point_fn_opt(
            i_b, p0_alpha - p0_deriv_0 / p0_deriv_1, constraint_state, rigid_global_info
        )

        if p0_cost < p1_cost:
            p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = p0_alpha, p0_cost, p0_deriv_0, p0_deriv_1

        if qd.abs(p1_deriv_0) < gtol:
            if qd.abs(p1_alpha) < rigid_global_info.EPS[None]:
                constraint_state.ls_result[i_b] = 2
            else:
                constraint_state.ls_result[i_b] = 0
            res_alpha = p1_alpha
        else:
            # Phase 2: Bracketing
            direction = (p1_deriv_0 < 0) * 2 - 1
            p2update = 0
            p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1 = p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1
            while (
                p1_deriv_0 * direction <= -gtol and constraint_state.ls_it[i_b] < rigid_global_info.ls_iterations[None]
            ):
                p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1 = p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1
                p2update = 1

                p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = func_ls_point_fn_opt(
                    i_b, p1_alpha - p1_deriv_0 / p1_deriv_1, constraint_state, rigid_global_info
                )
                if qd.abs(p1_deriv_0) < gtol:
                    res_alpha = p1_alpha
                    done = True
                    break
            if not done:
                if constraint_state.ls_it[i_b] >= rigid_global_info.ls_iterations[None]:
                    constraint_state.ls_result[i_b] = 3
                    res_alpha = p1_alpha
                    done = True

                if not p2update and not done:
                    constraint_state.ls_result[i_b] = 6
                    res_alpha = p1_alpha
                    done = True

                if not done:
                    # Phase 3: Refinement with batched 3-alpha evaluation
                    alpha_0 = p1_alpha - p1_deriv_0 / p1_deriv_1  # Newton from p1
                    alpha_1 = p1_alpha  # p2_next (= current p1)
                    alpha_2 = (p1_alpha + p2_alpha) * 0.5  # midpoint

                    while constraint_state.ls_it[i_b] < rigid_global_info.ls_iterations[None]:
                        # Batch evaluate cost, gradient, hessian for all 3 alphas in one constraint loop
                        costs, grads, hess = func_ls_point_fn_3alphas_opt(
                            i_b, alpha_0, alpha_1, alpha_2, constraint_state, rigid_global_info
                        )
                        alphas = qd.Vector([alpha_0, alpha_1, alpha_2])

                        # Check convergence among 3 candidates
                        p1_next_alpha = alpha_0
                        p2_next_alpha = alpha_1

                        best_alpha = gs.qd_float(0.0)
                        best_cost = gs.qd_float(0.0)
                        best_found = False
                        for i in qd.static(range(3)):
                            if qd.abs(grads[i]) < gtol and (not best_found or costs[i] < best_cost):
                                best_alpha = alphas[i]
                                best_cost = costs[i]
                                best_found = True

                        if best_found:
                            res_alpha = best_alpha
                            done = True
                        else:
                            (
                                b1,
                                p1_alpha,
                                p1_cost,
                                p1_deriv_0,
                                p1_deriv_1,
                                p1_next_alpha,
                            ) = update_bracket_no_eval_local(
                                p1_alpha,
                                p1_cost,
                                p1_deriv_0,
                                p1_deriv_1,
                                alphas,
                                costs,
                                grads,
                                hess,
                            )
                            (
                                b2,
                                p2_alpha,
                                p2_cost,
                                p2_deriv_0,
                                p2_deriv_1,
                                p2_next_alpha,
                            ) = update_bracket_no_eval_local(
                                p2_alpha,
                                p2_cost,
                                p2_deriv_0,
                                p2_deriv_1,
                                alphas,
                                costs,
                                grads,
                                hess,
                            )

                            if b1 == 0 and b2 == 0:
                                if costs[2] < p0_cost:
                                    constraint_state.ls_result[i_b] = 0
                                else:
                                    constraint_state.ls_result[i_b] = 7
                                res_alpha = alpha_2
                                done = True

                        if done:
                            break

                        # Compute next 3 alphas for next iteration
                        alpha_0 = p1_next_alpha
                        alpha_1 = p2_next_alpha
                        alpha_2 = (p1_alpha + p2_alpha) * 0.5

                    if not done:
                        if p1_cost <= p2_cost and p1_cost < p0_cost:
                            constraint_state.ls_result[i_b] = 4
                            res_alpha = p1_alpha
                        elif p2_cost <= p1_cost and p2_cost < p0_cost:
                            constraint_state.ls_result[i_b] = 4
                            res_alpha = p2_alpha
                        else:
                            constraint_state.ls_result[i_b] = 5
                            res_alpha = 0.0
    return res_alpha


# =====================================================================================================================
# ================================================= Solving Algorithm =================================================
# =====================================================================================================================


# ====================================================== Helpers ======================================================


@qd.func
def func_save_prev_grad(
    i_b,
    constraint_state: array_class.ConstraintState,
):
    n_dofs = constraint_state.qacc.shape[0]
    for i_d in range(n_dofs):
        constraint_state.cg_prev_grad[i_d, i_b] = constraint_state.grad[i_d, i_b]
        constraint_state.cg_prev_Mgrad[i_d, i_b] = constraint_state.Mgrad[i_d, i_b]


@qd.func
def func_update_constraint_batch(
    i_b,
    qacc: array_class.V_ANNOTATION,
    Ma: array_class.V_ANNOTATION,
    cost: array_class.V_ANNOTATION,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    n_dofs = constraint_state.qfrc_constraint.shape[0]
    ne = constraint_state.n_constraints_equality[i_b]
    nef = ne + constraint_state.n_constraints_frictionloss[i_b]

    constraint_state.prev_cost[i_b] = cost[i_b]
    cost_i = gs.qd_float(0.0)
    gauss_i = gs.qd_float(0.0)

    # Beware 'active' does not refer to whether a constraint is active, but rather whether its quadratic cost is active
    for i_c in range(constraint_state.n_constraints[i_b]):
        if qd.static(static_rigid_sim_config.solver_type == gs.constraint_solver.Newton):
            constraint_state.prev_active[i_c, i_b] = constraint_state.active[i_c, i_b]
        constraint_state.active[i_c, i_b] = True

        floss_force = gs.qd_float(0.0)
        if ne <= i_c and i_c < nef:  # Friction constraints
            f = constraint_state.efc_frictionloss[i_c, i_b]
            r = constraint_state.diag[i_c, i_b]
            rf = r * f
            linear_neg = constraint_state.Jaref[i_c, i_b] <= -rf
            linear_pos = constraint_state.Jaref[i_c, i_b] >= rf
            constraint_state.active[i_c, i_b] = not (linear_neg or linear_pos)
            floss_force = linear_neg * f + linear_pos * -f
            floss_cost_local = linear_neg * f * (-0.5 * rf - constraint_state.Jaref[i_c, i_b])
            floss_cost_local = floss_cost_local + linear_pos * f * (-0.5 * rf + constraint_state.Jaref[i_c, i_b])
            cost_i = cost_i + floss_cost_local
        elif nef <= i_c:  # Contact constraints
            constraint_state.active[i_c, i_b] = constraint_state.Jaref[i_c, i_b] < 0

        constraint_state.efc_force[i_c, i_b] = floss_force + (
            -constraint_state.Jaref[i_c, i_b] * constraint_state.efc_D[i_c, i_b] * constraint_state.active[i_c, i_b]
        )

    if qd.static(static_rigid_sim_config.sparse_solve):
        for i_d in range(n_dofs):
            constraint_state.qfrc_constraint[i_d, i_b] = gs.qd_float(0.0)
        for i_c in range(constraint_state.n_constraints[i_b]):
            for i_d_ in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                i_d = constraint_state.jac_relevant_dofs[i_c, i_d_, i_b]
                constraint_state.qfrc_constraint[i_d, i_b] = (
                    constraint_state.qfrc_constraint[i_d, i_b]
                    + constraint_state.jac[i_c, i_d, i_b] * constraint_state.efc_force[i_c, i_b]
                )
    else:
        for i_d in range(n_dofs):
            qfrc_constraint = gs.qd_float(0.0)
            for i_c in range(constraint_state.n_constraints[i_b]):
                qfrc_constraint = (
                    qfrc_constraint + constraint_state.jac[i_c, i_d, i_b] * constraint_state.efc_force[i_c, i_b]
                )
            constraint_state.qfrc_constraint[i_d, i_b] = qfrc_constraint

    # (Mx - Mx') * (x - x')
    for i_d in range(n_dofs):
        v = 0.5 * (Ma[i_d, i_b] - dofs_state.force[i_d, i_b]) * (qacc[i_d, i_b] - dofs_state.acc_smooth[i_d, i_b])
        gauss_i = gauss_i + v
        cost_i = cost_i + v

    # D * (Jx - aref) ** 2
    for i_c in range(constraint_state.n_constraints[i_b]):
        cost_i = cost_i + 0.5 * (
            constraint_state.Jaref[i_c, i_b] ** 2 * constraint_state.efc_D[i_c, i_b] * constraint_state.active[i_c, i_b]
        )

    constraint_state.gauss[i_b] = gauss_i
    cost[i_b] = cost_i


@qd.func
def func_update_constraint(
    qacc: array_class.V_ANNOTATION,
    Ma: array_class.V_ANNOTATION,
    cost: array_class.V_ANNOTATION,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    _B = constraint_state.jac.shape[2]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        func_update_constraint_batch(
            i_b,
            qacc=qacc,
            Ma=Ma,
            cost=cost,
            dofs_state=dofs_state,
            constraint_state=constraint_state,
            static_rigid_sim_config=static_rigid_sim_config,
        )


@qd.func
def func_update_gradient_batch(
    i_b,
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    n_dofs = constraint_state.grad.shape[0]

    for i_d in range(n_dofs):
        constraint_state.grad[i_d, i_b] = (
            constraint_state.Ma[i_d, i_b] - dofs_state.force[i_d, i_b] - constraint_state.qfrc_constraint[i_d, i_b]
        )

    if qd.static(static_rigid_sim_config.solver_type == gs.constraint_solver.CG):
        func_solve_mass_batch(
            i_b,
            constraint_state.grad,
            constraint_state.Mgrad,
            array_class.PLACEHOLDER,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
            is_backward=False,
        )

    if qd.static(static_rigid_sim_config.solver_type == gs.constraint_solver.Newton):
        func_cholesky_solve_batch(i_b, constraint_state=constraint_state)


@qd.func
def func_update_gradient_tiled(
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    _B = constraint_state.jac.shape[2]
    n_dofs = constraint_state.jac.shape[1]

    # Compute Mgrad = H^{-1} @ grad, s.t. grad = M @ acc - q_force_ext - q_force_const
    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        constraint_state.grad[i_d, i_b] = (
            constraint_state.Ma[i_d, i_b] - dofs_state.force[i_d, i_b] - constraint_state.qfrc_constraint[i_d, i_b]
        )

    if qd.static(static_rigid_sim_config.solver_type == gs.constraint_solver.CG):
        qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
        for i_b in range(_B):
            func_solve_mass_batch(
                i_b,
                constraint_state.grad,
                constraint_state.Mgrad,
                array_class.PLACEHOLDER,
                entities_info=entities_info,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
                is_backward=False,
            )

    if qd.static(static_rigid_sim_config.solver_type == gs.constraint_solver.Newton):
        func_cholesky_solve_tiled(constraint_state, static_rigid_sim_config)


@qd.func
def func_update_gradient(
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """
    Unified implementation of gradient updated optimized for both CPU and GPU backends.

    The tiled optimization is only supported on GPU backend and specifically optimized for it, falling back to the
    classical batched implementation when running on CPU backend.

    Note that the tiled cholesky factorization and solving is not systematically enabled because it is not always
    superior in terms of performance and does not support arbitrary matrix sizes. More specifically, tiling gets more
    beneficial as n_dofs increases, but n_dofs>=96 is not supported for now. It is the responsibility of the calling
    code to configure the static global flag `enable_tiled_cholesky_hessian` accordingly. Failing to do so will cause
    the requested shared memory allocation to exceed 48kB and raise an exception.
    """
    _B = constraint_state.jac.shape[2]

    if qd.static(
        not static_rigid_sim_config.enable_tiled_cholesky_hessian or static_rigid_sim_config.backend == gs.cpu
    ):
        # CPU
        qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
        for i_b in range(_B):
            func_update_gradient_batch(
                i_b,
                dofs_state=dofs_state,
                entities_info=entities_info,
                constraint_state=constraint_state,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
            )
    else:
        # GPU
        func_update_gradient_tiled(
            dofs_state=dofs_state,
            entities_info=entities_info,
            constraint_state=constraint_state,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )


@qd.func
def func_terminate_or_update_descent_batch(
    i_b,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    n_dofs = constraint_state.jac.shape[1]

    # Check convergence, i.e. whether the cost function is not longer decreasing or the gradient is flat
    tol_scaled = (rigid_global_info.meaninertia[i_b] * qd.max(1, n_dofs)) * rigid_global_info.tolerance[None]
    improvement = constraint_state.prev_cost[i_b] - constraint_state.cost[i_b]
    grad_norm = gs.qd_float(0.0)
    for i_d in range(n_dofs):
        grad_norm = grad_norm + constraint_state.grad[i_d, i_b] * constraint_state.grad[i_d, i_b]
    grad_norm = qd.sqrt(grad_norm)
    improved = grad_norm > tol_scaled and improvement > tol_scaled
    constraint_state.improved[i_b] = improved

    # Update search direction if necessary
    if improved:
        if qd.static(static_rigid_sim_config.solver_type == gs.constraint_solver.Newton):
            for i_d in range(n_dofs):
                constraint_state.search[i_d, i_b] = -constraint_state.Mgrad[i_d, i_b]
        else:
            cg_beta = gs.qd_float(0.0)
            cg_pg_dot_pMg = gs.qd_float(0.0)

            for i_d in range(n_dofs):
                cg_beta = cg_beta + constraint_state.grad[i_d, i_b] * (
                    constraint_state.Mgrad[i_d, i_b] - constraint_state.cg_prev_Mgrad[i_d, i_b]
                )
                cg_pg_dot_pMg = cg_pg_dot_pMg + (
                    constraint_state.cg_prev_Mgrad[i_d, i_b] * constraint_state.cg_prev_grad[i_d, i_b]
                )
            cg_beta = qd.max(cg_beta / qd.max(rigid_global_info.EPS[None], cg_pg_dot_pMg), 0.0)

            constraint_state.cg_pg_dot_pMg[i_b] = cg_pg_dot_pMg
            constraint_state.cg_beta[i_b] = cg_beta

            for i_d in range(n_dofs):
                constraint_state.search[i_d, i_b] = (
                    -constraint_state.Mgrad[i_d, i_b] + cg_beta * constraint_state.search[i_d, i_b]
                )


@qd.func
def initialize_Jaref(
    qacc: array_class.V_ANNOTATION,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    _B = constraint_state.jac.shape[2]
    n_dofs = constraint_state.jac.shape[1]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        for i_c in range(constraint_state.n_constraints[i_b]):
            Jaref = -constraint_state.aref[i_c, i_b]
            if qd.static(static_rigid_sim_config.sparse_solve):
                for i_d_ in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                    i_d = constraint_state.jac_relevant_dofs[i_c, i_d_, i_b]
                    Jaref = Jaref + constraint_state.jac[i_c, i_d, i_b] * qacc[i_d, i_b]
            else:
                for i_d in range(n_dofs):
                    Jaref = Jaref + constraint_state.jac[i_c, i_d, i_b] * qacc[i_d, i_b]
            constraint_state.Jaref[i_c, i_b] = Jaref


@qd.func
def initialize_Ma(
    Ma: array_class.V_ANNOTATION,
    qacc: array_class.V_ANNOTATION,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    _B = rigid_global_info.mass_mat.shape[2]
    n_dofs = qacc.shape[0]

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d1, i_b in qd.ndrange(n_dofs, _B):
        I_d1 = [i_d1, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else i_d1
        i_e = dofs_info.entity_idx[I_d1]
        Ma_ = gs.qd_float(0.0)
        for i_d2 in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
            Ma_ = Ma_ + rigid_global_info.mass_mat[i_d1, i_d2, i_b] * qacc[i_d2, i_b]
        Ma[i_d1, i_b] = Ma_


# ======================================================= Core ========================================================


@qd.kernel(fastcache=gs.use_fastcache)
def func_solve_init(
    dofs_info: array_class.DofsInfo,
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    _B = dofs_state.acc_smooth.shape[1]
    n_dofs = dofs_state.acc_smooth.shape[0]

    if qd.static(static_rigid_sim_config.enable_mujoco_compatibility):
        # Compute cost for warmstart state (i.e. acceleration at previous timestep)
        initialize_Ma(
            Ma=constraint_state.Ma_ws,
            qacc=constraint_state.qacc_ws,
            dofs_info=dofs_info,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )

        initialize_Jaref(
            qacc=constraint_state.qacc_ws,
            constraint_state=constraint_state,
            static_rigid_sim_config=static_rigid_sim_config,
        )
        func_update_constraint(
            qacc=constraint_state.qacc_ws,
            Ma=constraint_state.Ma_ws,
            cost=constraint_state.cost_ws,
            dofs_state=dofs_state,
            constraint_state=constraint_state,
            static_rigid_sim_config=static_rigid_sim_config,
        )

        # Compute cost for current state (assuming constraint-free acceleration)
        initialize_Ma(
            Ma=constraint_state.Ma,
            qacc=dofs_state.acc_smooth,
            dofs_info=dofs_info,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )

        initialize_Jaref(
            qacc=dofs_state.acc_smooth,
            constraint_state=constraint_state,
            static_rigid_sim_config=static_rigid_sim_config,
        )
        func_update_constraint(
            qacc=dofs_state.acc_smooth,
            Ma=constraint_state.Ma,
            cost=constraint_state.cost,
            dofs_state=dofs_state,
            constraint_state=constraint_state,
            static_rigid_sim_config=static_rigid_sim_config,
        )

        # Pick the best starting point between current state and warmstart
        qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_d, i_b in qd.ndrange(n_dofs, _B):
            if constraint_state.cost_ws[i_b] < constraint_state.cost[i_b]:
                constraint_state.qacc[i_d, i_b] = constraint_state.qacc_ws[i_d, i_b]
                constraint_state.Ma[i_d, i_b] = constraint_state.Ma_ws[i_d, i_b]
            else:
                constraint_state.qacc[i_d, i_b] = dofs_state.acc_smooth[i_d, i_b]
    else:
        # Always initialize from warmstart
        qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_d, i_b in qd.ndrange(n_dofs, _B):
            if constraint_state.n_constraints[i_b] > 0 and constraint_state.is_warmstart[i_b]:
                constraint_state.qacc[i_d, i_b] = constraint_state.qacc_ws[i_d, i_b]
            else:
                constraint_state.qacc[i_d, i_b] = dofs_state.acc_smooth[i_d, i_b]

        initialize_Ma(
            Ma=constraint_state.Ma,
            qacc=constraint_state.qacc,
            dofs_info=dofs_info,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )

    # Initialize solver accordingly
    initialize_Jaref(
        qacc=constraint_state.qacc,
        constraint_state=constraint_state,
        static_rigid_sim_config=static_rigid_sim_config,
    )
    func_update_constraint(
        qacc=constraint_state.qacc,
        Ma=constraint_state.Ma,
        cost=constraint_state.cost,
        dofs_state=dofs_state,
        constraint_state=constraint_state,
        static_rigid_sim_config=static_rigid_sim_config,
    )

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in qd.ndrange(_B):
        constraint_state.improved[i_b] = constraint_state.n_constraints[i_b] > 0

    if qd.static(static_rigid_sim_config.solver_type == gs.constraint_solver.Newton):
        func_hessian_and_cholesky_factor_direct(
            entities_info=entities_info,
            constraint_state=constraint_state,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )

    func_update_gradient(
        dofs_state=dofs_state,
        entities_info=entities_info,
        constraint_state=constraint_state,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
    )

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        constraint_state.search[i_d, i_b] = -constraint_state.Mgrad[i_d, i_b]


@qd.func
def func_solve_iter(
    i_b,
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    n_dofs = constraint_state.qacc.shape[0]
    alpha = func_linesearch_batch(
        i_b,
        entities_info=entities_info,
        dofs_state=dofs_state,
        rigid_global_info=rigid_global_info,
        constraint_state=constraint_state,
        static_rigid_sim_config=static_rigid_sim_config,
    )

    if qd.abs(alpha) < rigid_global_info.EPS[None]:
        constraint_state.improved[i_b] = False
    else:
        for i_d in range(n_dofs):
            constraint_state.qacc[i_d, i_b] = (
                constraint_state.qacc[i_d, i_b] + constraint_state.search[i_d, i_b] * alpha
            )
            constraint_state.Ma[i_d, i_b] = constraint_state.Ma[i_d, i_b] + constraint_state.mv[i_d, i_b] * alpha

        for i_c in range(constraint_state.n_constraints[i_b]):
            constraint_state.Jaref[i_c, i_b] = constraint_state.Jaref[i_c, i_b] + constraint_state.jv[i_c, i_b] * alpha

        if qd.static(static_rigid_sim_config.solver_type == gs.constraint_solver.CG):
            for i_d in range(n_dofs):
                constraint_state.cg_prev_grad[i_d, i_b] = constraint_state.grad[i_d, i_b]
                constraint_state.cg_prev_Mgrad[i_d, i_b] = constraint_state.Mgrad[i_d, i_b]

        func_update_constraint_batch(
            i_b,
            qacc=constraint_state.qacc,
            Ma=constraint_state.Ma,
            cost=constraint_state.cost,
            dofs_state=dofs_state,
            constraint_state=constraint_state,
            static_rigid_sim_config=static_rigid_sim_config,
        )

        if qd.static(static_rigid_sim_config.solver_type == gs.constraint_solver.Newton):
            func_build_changed_constraint_list(i_b, constraint_state=constraint_state)
            is_degenerated = func_hessian_and_cholesky_factor_incremental_batch(
                i_b,
                constraint_state=constraint_state,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
            )
            if is_degenerated:
                func_hessian_and_cholesky_factor_direct_batch(
                    i_b,
                    entities_info=entities_info,
                    constraint_state=constraint_state,
                    rigid_global_info=rigid_global_info,
                    static_rigid_sim_config=static_rigid_sim_config,
                )

        func_update_gradient_batch(
            i_b,
            dofs_state=dofs_state,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            constraint_state=constraint_state,
            static_rigid_sim_config=static_rigid_sim_config,
        )

        func_terminate_or_update_descent_batch(
            i_b,
            constraint_state=constraint_state,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )


@qd.perf_dispatch(
    get_geometry_hash=lambda *args, **kwargs: (*args, frozendict(kwargs)), warmup=3, active=3, repeat_after_seconds=1.0
)
def func_solve_body(
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
) -> None: ...


@func_solve_body.register(is_compatible=lambda *args, **kwargs: True)
@qd.kernel(fastcache=gs.use_fastcache)
def func_solve_body_monolith(
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    _B = constraint_state.grad.shape[1]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0:
            for _ in range(rigid_global_info.iterations[None]):
                func_solve_iter(
                    i_b,
                    entities_info=entities_info,
                    dofs_state=dofs_state,
                    rigid_global_info=rigid_global_info,
                    constraint_state=constraint_state,
                    static_rigid_sim_config=static_rigid_sim_config,
                )
                if not constraint_state.improved[i_b]:
                    break
        else:
            constraint_state.improved[i_b] = False


# =====================================================================================================================
# ==================================================== Finalization ===================================================
# =====================================================================================================================


@qd.kernel(fastcache=gs.use_fastcache)
def func_update_contact_force(
    links_state: array_class.LinksState,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    n_links = links_state.contact_force.shape[0]
    _B = links_state.contact_force.shape[1]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b in qd.ndrange(n_links, _B):
        links_state.contact_force[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        const_start = constraint_state.n_constraints_equality[i_b] + constraint_state.n_constraints_frictionloss[i_b]

        # contact constraints should be after equality and frictionloss constraints and before joint limit constraints
        for i_c in range(collider_state.n_contacts[i_b]):
            contact_data_normal = collider_state.contact_data.normal[i_c, i_b]
            contact_data_friction = collider_state.contact_data.friction[i_c, i_b]
            contact_data_link_a = collider_state.contact_data.link_a[i_c, i_b]
            contact_data_link_b = collider_state.contact_data.link_b[i_c, i_b]

            force = qd.Vector.zero(gs.qd_float, 3)
            d1, d2 = gu.qd_orthogonals(contact_data_normal)
            for i_dir in range(4):
                d = (2 * (i_dir % 2) - 1) * (d1 if i_dir < 2 else d2)
                n = d * contact_data_friction - contact_data_normal
                force = force + n * constraint_state.efc_force[i_c * 4 + i_dir + const_start, i_b]

            collider_state.contact_data.force[i_c, i_b] = force

            links_state.contact_force[contact_data_link_a, i_b] = (
                links_state.contact_force[contact_data_link_a, i_b] - force
            )
            links_state.contact_force[contact_data_link_b, i_b] = (
                links_state.contact_force[contact_data_link_b, i_b] + force
            )


@qd.kernel(fastcache=gs.use_fastcache)
def func_update_qacc(
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
    errno: array_class.V_ANNOTATION,
):
    n_dofs = dofs_state.acc.shape[0]
    _B = dofs_state.acc.shape[1]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        dofs_state.acc[i_d, i_b] = constraint_state.qacc[i_d, i_b]
        dofs_state.qf_constraint[i_d, i_b] = constraint_state.qfrc_constraint[i_d, i_b]
        dofs_state.force[i_d, i_b] = dofs_state.qf_smooth[i_d, i_b] + constraint_state.qfrc_constraint[i_d, i_b]
        constraint_state.qacc_ws[i_d, i_b] = constraint_state.qacc[i_d, i_b]
        if qd.math.isnan(constraint_state.qacc[i_d, i_b]):
            errno[i_b] = errno[i_b] | array_class.ErrorCode.INVALID_FORCE_NAN

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        constraint_state.is_warmstart[i_b] = True


from genesis.utils.deprecated_module_wrapper import create_virtual_deprecated_module

create_virtual_deprecated_module(__name__, "genesis.engine.solvers.rigid.constraint_solver_decomp")
