"""
Constraint solver for rigid body dynamics.

This module contains the main ConstraintSolver class and related kernel functions
for handling equality and inequality constraints in rigid body simulation.
"""

from typing import TYPE_CHECKING

import gstaichi as ti
import numpy as np
import torch

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class
import genesis.engine.solvers.rigid.backward_constraint_solver as backward_constraint_solver
import genesis.engine.solvers.rigid.constraint_noslip as constraint_noslip
from genesis.engine.solvers.rigid.contact_island import ContactIsland
from genesis.utils.misc import ti_to_torch

# Import from submodules for internal use
from genesis.engine.solvers.rigid.constraint_builder import (
    add_collision_constraints,
    func_equality_connect,
    func_equality_joint,
    add_equality_constraints,
    add_inequality_constraints,
    func_equality_weld,
    add_joint_limit_constraints,
    add_frictionloss_constraints,
    kernel_add_weld_constraint,
    kernel_delete_weld_constraint,
    kernel_get_equality_constraints,
)

from genesis.engine.solvers.rigid.constraint_solver_core import (
    func_nt_hessian_incremental,
    func_nt_hessian_direct,
    func_nt_chol_factor,
    func_nt_chol_solve,
    func_solve,
    func_ls_init,
    func_ls_point_fn,
    func_no_linesearch,
    func_linesearch,
    update_bracket,
    func_solve_iter,
    func_update_constraint,
    func_update_gradient,
    initialize_Jaref,
    initialize_Ma,
    func_init_solver,
)

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
        self.constraint_state.ti_n_equalities.from_numpy(
            np.full((self._solver._B,), self._solver.n_equalities, dtype=gs.np_int)
        )

        self._eq_const_info_cache = {}

        cs = self.constraint_state
        self.ti_n_equalities = cs.ti_n_equalities
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
        self.quad = cs.quad
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
            is_warmstart = ti_to_torch(self.constraint_state.is_warmstart, copy=False)
            qacc_ws = ti_to_torch(self.constraint_state.qacc_ws, copy=False)
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
        func_init_solver(
            self._solver.dofs_info,
            self._solver.dofs_state,
            self._solver.entities_info,
            self.constraint_state,
            self._solver._rigid_global_info,
            self._solver._static_rigid_sim_config,
        )
        func_solve(
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

        n_eqs = tuple(self.constraint_state.ti_n_equalities.to_numpy())
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


@ti.kernel(fastcache=gs.use_fastcache)
def constraint_solver_kernel_reset(
    envs_idx: ti.types.ndarray(),
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
):
    n_dofs = constraint_state.qacc_ws.shape[0]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        constraint_state.is_warmstart[i_b] = False
        for i_d in range(n_dofs):
            constraint_state.qacc_ws[i_d, i_b] = 0.0


@ti.kernel(fastcache=gs.use_fastcache)
def constraint_solver_kernel_clear(
    envs_idx: ti.types.ndarray(),
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    n_dofs = constraint_state.qacc_ws.shape[0]
    len_constraints = constraint_state.jac.shape[0]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        constraint_state.n_constraints[i_b] = 0
        constraint_state.n_constraints_equality[i_b] = 0
        constraint_state.n_constraints_frictionloss[i_b] = 0
        # Reset dynamic equality count to static count to avoid stale constraints after partial reset
        constraint_state.ti_n_equalities[i_b] = rigid_global_info.n_equalities[None]
        for i_d, i_c in ti.ndrange(n_dofs, len_constraints):
            constraint_state.jac[i_c, i_d, i_b] = 0.0
        if ti.static(static_rigid_sim_config.sparse_solve):
            for i_c in range(len_constraints):
                constraint_state.jac_n_relevant_dofs[i_c, i_b] = 0


@ti.kernel(fastcache=gs.use_fastcache)
def func_update_contact_force(
    links_state: array_class.LinksState,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
):
    n_links = links_state.contact_force.shape[0]
    _B = links_state.contact_force.shape[1]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b in ti.ndrange(n_links, _B):
        links_state.contact_force[i_l, i_b] = ti.Vector.zero(gs.ti_float, 3)

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        const_start = constraint_state.n_constraints_equality[i_b] + constraint_state.n_constraints_frictionloss[i_b]

        # contact constraints should be after equality and frictionloss constraints and before joint limit constraints
        for i_c in range(collider_state.n_contacts[i_b]):
            contact_data_normal = collider_state.contact_data.normal[i_c, i_b]
            contact_data_friction = collider_state.contact_data.friction[i_c, i_b]
            contact_data_link_a = collider_state.contact_data.link_a[i_c, i_b]
            contact_data_link_b = collider_state.contact_data.link_b[i_c, i_b]

            force = ti.Vector.zero(gs.ti_float, 3)
            d1, d2 = gu.ti_orthogonals(contact_data_normal)
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


@ti.kernel(fastcache=gs.use_fastcache)
def func_update_qacc(
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: ti.template(),
    errno: array_class.V_ANNOTATION,
):
    n_dofs = dofs_state.acc.shape[0]
    _B = dofs_state.acc.shape[1]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in ti.ndrange(n_dofs, _B):
        dofs_state.acc[i_d, i_b] = constraint_state.qacc[i_d, i_b]
        dofs_state.qf_constraint[i_d, i_b] = constraint_state.qfrc_constraint[i_d, i_b]
        dofs_state.force[i_d, i_b] = dofs_state.qf_smooth[i_d, i_b] + constraint_state.qfrc_constraint[i_d, i_b]
        constraint_state.qacc_ws[i_d, i_b] = constraint_state.qacc[i_d, i_b]
        if ti.math.isnan(constraint_state.qacc[i_d, i_b]):
            errno[i_b] = errno[i_b] | 0b00000000000000000000000000000100

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        constraint_state.is_warmstart[i_b] = True


from genesis.utils.deprecated_module_wrapper import create_virtual_deprecated_module

create_virtual_deprecated_module(__name__, "genesis.engine.solvers.rigid.constraint_solver_decomp")
