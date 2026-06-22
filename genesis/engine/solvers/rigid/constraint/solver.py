from typing import TYPE_CHECKING

import numpy as np
import quadrants as qd
import torch
from frozendict import frozendict

import genesis as gs

import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.engine.solvers.rigid.abd import func_solve_mass_batch
from genesis.engine.solvers.rigid.abd.misc import linear_to_lower_tri
from genesis.utils.misc import qd_to_torch, indices_to_mask, assign_indexed_tensor

from .island import (
    _sort_island_contacts,
    func_build_islands,
    func_constraint_island,
    func_group_constraints_by_island,
    func_island_contacts_total,
)
from . import backward as backward_constraint_solver
from . import noslip as constraint_noslip


@qd.func
def _sort_relevant_dofs_descending(
    constraint_state: array_class.ConstraintState,
    i_con: qd.int32,
    n: qd.int32,
    i_b: qd.int32,
    static_rigid_sim_config: qd.template(),
):
    """Insertion sort jac_dofs_idx[i_con, :n, i_b] in descending order.

    Only the sparse skyline / incremental Cholesky relies on globally descending DOF order; the J.v / J^T.v products
    and the per-island solves are order-independent. So the sort is skipped unless sparse_solve is set - it is a
    serial (data-dependent) loop that would otherwise serialize the parallel contact-assembly kernel on GPU. The
    array is typically <= 14 elements, so O(n^2) is fine.
    """
    if qd.static(static_rigid_sim_config.sparse_solve):
        for i in range(1, n):
            key = constraint_state.jac_dofs_idx[i_con, i, i_b]
            j = i - 1
            while j >= 0 and constraint_state.jac_dofs_idx[i_con, j, i_b] < key:
                constraint_state.jac_dofs_idx[i_con, j + 1, i_b] = constraint_state.jac_dofs_idx[i_con, j, i_b]
                j -= 1
            constraint_state.jac_dofs_idx[i_con, j + 1, i_b] = key


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
        self._n_iterations = int(
            rigid_solver._options.iterations
        )  # Python-native; passed to Python-scope functions to avoid CPU-GPU sync
        self.tolerance = rigid_solver._options.tolerance
        self.ls_iterations = rigid_solver._options.ls_iterations
        self.ls_tolerance = rigid_solver._options.ls_tolerance
        # Effective (CPU-gated) sparsity flag, resolved in the static config; the raw option may differ on GPU.
        self.sparse_solve = rigid_solver._static_rigid_sim_config.sparse_solve

        # Note that it must be over-estimated because friction parameters and joint limits may be updated dynamically.
        # * 4 constraints per contact, bounded by the post-pruning contact budget enforced by the collider
        # * 1 constraint per 1DoF joint limit (upper and lower, if not inf)
        # * 1 constraint per dof frictionloss
        # * up to 6 constraints per equality (weld)
        # When 'max_contacts' is set, it overrides the post-pruning contact budget enforced by the collider.
        collider_info = rigid_solver.collider._collider_info
        if rigid_solver._options.max_contacts is not None:
            collider_info.max_contacts[None] = min(
                rigid_solver._options.max_contacts, collider_info.max_candidate_contacts[None]
            )
        self.len_constraints = int(
            4 * collider_info.max_contacts[None]
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
        self.jac_n_dofs = cs.jac_n_dofs
        self.jac_dofs_idx = cs.jac_dofs_idx
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

        self.ls_alpha = cs.ls_alpha
        self.ls_p0_cost = cs.ls_p0_cost
        self.ls_alpha_newton = cs.ls_alpha_newton
        self.ls_gtol = cs.ls_gtol
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

        # Island partition consumed by the per-island Newton solve (use_contact_island) and, when hibernation is
        # enabled, by the hibernation decision/wakeup. Every field is read only inside qd.static(use_contact_island)
        # branches, so with islands off it is allocated as scalars (see get_island_state) and never touched.
        self.island_state = array_class.get_island_state(self._solver, self._collider)
        # The hibernated-island daisy chain must start empty (-1 = no successor); it persists across steps, written
        # when an island hibernates and cleared on wakeup.
        if self._solver._use_hibernation:
            self.island_state.hibernated_next_link.fill(-1)

        # Fill-reducing DOF permutation for the skyline Cholesky: a structural choice fixed once from the initial
        # body layout (forward kinematics has already run at this point), never recomputed in the step loop. The
        # reorder (COM sort) only kicks in for the CPU envelope; otherwise this initializes the identity permutation,
        # which the sparse Hessian assembly still indexes through (including the explicit GPU sparse path).
        if self.sparse_solve:
            func_compute_dof_perm(
                self._solver.dofs_info,
                self._solver.entities_info,
                self._solver.links_state,
                self.constraint_state,
                self._solver._static_rigid_sim_config,
            )

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
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        envs_idx = self._solver._scene._sanitize_envs_idx(envs_idx)
        constraint_solver_kernel_reset(envs_idx, self.constraint_state, self._solver._static_rigid_sim_config)

    def clear(self, envs_idx=None):
        self.reset(envs_idx)

        if gs.use_zerocopy and (
            not isinstance(envs_idx, torch.Tensor) or (not IS_OLD_TORCH or envs_idx.dtype == torch.bool)
        ):
            n_constraints = qd_to_torch(self.constraint_state.n_constraints, copy=False)
            n_constraints_equality = qd_to_torch(self.constraint_state.n_constraints_equality, copy=False)
            n_constraints_frictionloss = qd_to_torch(self.constraint_state.n_constraints_frictionloss, copy=False)
            qd_n_equalities = qd_to_torch(self.constraint_state.qd_n_equalities, copy=False)
            n_eq = self._solver._n_equalities
            if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
                n_constraints.masked_fill_(envs_idx, 0)
                n_constraints_equality.masked_fill_(envs_idx, 0)
                n_constraints_frictionloss.masked_fill_(envs_idx, 0)
                qd_n_equalities.masked_fill_(envs_idx, n_eq)
            elif isinstance(envs_idx, torch.Tensor):
                n_constraints.scatter_(0, envs_idx, 0)
                n_constraints_equality.scatter_(0, envs_idx, 0)
                n_constraints_frictionloss.scatter_(0, envs_idx, 0)
                qd_n_equalities.scatter_(0, envs_idx, n_eq)
            else:
                env_mask = indices_to_mask(envs_idx)
                assign_indexed_tensor(n_constraints, env_mask, 0)
                assign_indexed_tensor(n_constraints_equality, env_mask, 0)
                assign_indexed_tensor(n_constraints_frictionloss, env_mask, 0)
                assign_indexed_tensor(qd_n_equalities, env_mask, n_eq)
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        if not isinstance(envs_idx, torch.Tensor):
            envs_idx = self._solver._scene._sanitize_envs_idx(envs_idx)
        if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
            fn = constraint_solver_kernel_masked_clear
        else:
            fn = constraint_solver_kernel_clear
        fn(
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
            self._solver.equalities_info,
            self.constraint_state,
            self._collider._collider_state,
            self.island_state,
            self._solver._rigid_global_info,
            self._solver._static_rigid_sim_config,
            self._collider._collider_static_config,
        )

    def resolve(self, entities_info=None, rigid_global_info=None):
        # func_solve_init is launched by each dispatch entrypoint (func_solve_body_monolith / func_solve_decomposed),
        # not here: only the entrypoint statically knows its arm, which determines whether the init factor/gradient is
        # done (monolith) or skipped (decomposed re-factors in-loop).
        func_solve_body(
            self._solver.entities_info,
            self._solver.dofs_info,
            self._solver.dofs_state,
            self.constraint_state,
            self._solver._rigid_global_info,
            self._solver._static_rigid_sim_config,
            self._n_iterations,
            self.island_state,
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
        if self._solver._para_level >= gs.PARA_LEVEL.PARTIAL:
            # GPU (any n_envs): one kernel decomposed into per-phase offloaded tasks, so that each phase keeps its
            # own parallel launch shape.
            constraint_noslip.kernel_noslip_decomposed(
                self._collider._collider_state,
                self._solver.dofs_state,
                self._solver.entities_info,
                self._solver._rigid_global_info,
                self.constraint_state,
                self._solver._static_rigid_sim_config,
            )
        else:
            # Serialized (CPU): single fused kernel processing each env end-to-end, so that the per-env AR scratch stays
            # cache-hot between the AR build and the force-update sweep (func_noslip_batch).
            constraint_noslip.kernel_noslip_fused(
                self._collider._collider_state,
                self._solver.dofs_state,
                self._solver.entities_info,
                self._solver._rigid_global_info,
                self.constraint_state,
                self.island_state,
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


@qd.kernel(fastcache=True)
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


@qd.kernel(fastcache=True)
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


@qd.func
def func_clear_constraint_at_env(
    i_b,
    n_dofs,
    len_constraints,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    constraint_state.n_constraints[i_b] = 0
    constraint_state.n_constraints_equality[i_b] = 0
    constraint_state.n_constraints_frictionloss[i_b] = 0
    constraint_state.qd_n_equalities[i_b] = rigid_global_info.n_equalities[None]
    for i_d, i_c in qd.ndrange(n_dofs, len_constraints):
        constraint_state.jac[i_c, i_d, i_b] = 0.0
    for i_c in range(len_constraints):
        constraint_state.jac_n_dofs[i_c, i_b] = 0


@qd.kernel(fastcache=True)
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
        func_clear_constraint_at_env(
            i_b, n_dofs, len_constraints, constraint_state, rigid_global_info, static_rigid_sim_config
        )


@qd.kernel(fastcache=True)
def constraint_solver_kernel_masked_clear(
    envs_mask: qd.types.ndarray(),
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    n_dofs = constraint_state.qacc_ws.shape[0]
    len_constraints = constraint_state.jac.shape[0]

    for i_b in range(envs_mask.shape[0]):
        if envs_mask[i_b]:
            func_clear_constraint_at_env(
                i_b, n_dofs, len_constraints, constraint_state, rigid_global_info, static_rigid_sim_config
            )


# ========================================= Register Pre-Defined Constraints ==========================================


@qd.func
def _add_friction_constraint(
    i_b,
    i_col_,
    i_friction,
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    collider_state: array_class.ColliderState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Add one friction-basis row to the constraint Jacobian and write its matching diag/aref/efc_D scalars."""
    EPS = rigid_global_info.EPS[None]
    n_dofs = dofs_state.ctrl_mode.shape[0]

    collision_con_start = constraint_state.n_constraints[i_b]

    i_col = collider_state.contact_sort_idx[i_col_, i_b]
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

    d = (2 * (i_friction % 2) - 1) * (d1 if i_friction < 2 else d2)
    n = d * contact_data_friction - contact_data_normal

    n_con = collision_con_start + i_col_ * 4 + i_friction
    if qd.static(static_rigid_sim_config.sparse_solve):
        for i_d_ in range(constraint_state.jac_n_dofs[n_con, i_b]):
            i_d = constraint_state.jac_dofs_idx[n_con, i_d_, i_b]
            constraint_state.jac[n_con, i_d, i_b] = gs.qd_float(0.0)
    else:
        for i_d in range(n_dofs):
            constraint_state.jac[n_con, i_d, i_b] = gs.qd_float(0.0)

    con_n_dofs = 0
    jac_qvel = gs.qd_float(0.0)
    for i_ab in range(2):
        sign = gs.qd_float(-1.0)
        link = link_a
        if i_ab == 1:
            sign = gs.qd_float(1.0)
            link = link_b

        while link > -1:
            link_maybe_batch = [link, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else link

            # reverse order to make sure dofs in each row of self.jac_dofs_idx are strictly descending
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

                constraint_state.jac_dofs_idx[n_con, con_n_dofs, i_b] = i_d
                con_n_dofs = con_n_dofs + 1

            link = links_info.parent_idx[link_maybe_batch]

    constraint_state.jac_n_dofs[n_con, i_b] = con_n_dofs
    _sort_relevant_dofs_descending(constraint_state, n_con, con_n_dofs, i_b, static_rigid_sim_config)
    imp, aref = gu.imp_aref(contact_data_sol_params, -contact_data_penetration, jac_qvel, -contact_data_penetration)

    diag = invweight + contact_data_friction * contact_data_friction * invweight
    diag *= 2 * contact_data_friction * contact_data_friction * (1 - imp) / imp
    diag = qd.max(diag, EPS)

    constraint_state.diag[n_con, i_b] = diag
    constraint_state.aref[n_con, i_b] = aref
    constraint_state.efc_D[n_con, i_b] = 1 / diag


@qd.func
def _add_collision_constraints_per_friction(
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    collider_state: array_class.ColliderState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Build all collision-contact constraints with one GPU thread per friction-basis constraint.

    Per-friction threading: 4x more threads than the legacy path; adjacent lanes vary the friction slot
    i_col_ * 4 + i_friction so within a warp adjacent threads write adjacent n_con values. Under the flipped jac
    layout (_B, n_dofs, n_constraints), n_con is stride-1, so jac writes coalesce.
    """
    _B = dofs_state.ctrl_mode.shape[1]
    max_candidate_contacts = collider_state.contact_data.link_a.shape[0]

    qd.loop_config(
        name="add_collision_constraints", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL
    )
    for flat_idx in range(_B * max_candidate_contacts * 4):
        slot = flat_idx % (max_candidate_contacts * 4)
        i_b = flat_idx // (max_candidate_contacts * 4)
        i_col_ = slot // 4
        i_friction = slot % 4
        if i_col_ < collider_state.n_contacts[i_b]:
            _add_friction_constraint(
                i_b,
                i_col_,
                i_friction,
                links_info=links_info,
                links_state=links_state,
                dofs_state=dofs_state,
                constraint_state=constraint_state,
                collider_state=collider_state,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
            )


@qd.func
def _add_collision_constraints_per_contact(
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    collider_state: array_class.ColliderState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Build all collision-contact constraints with one GPU thread per contact."""
    EPS = rigid_global_info.EPS[None]
    _B = dofs_state.ctrl_mode.shape[1]
    n_dofs = dofs_state.ctrl_mode.shape[0]
    max_candidate_contacts = collider_state.contact_data.link_a.shape[0]

    # Iteration order follows the jac layout: batch-outer keeps every write within one env's batch-first block, while
    # the batch-inner order keeps consecutive GPU threads on consecutive envs (coalesced batch-last).
    qd.loop_config(
        name="add_collision_constraints", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL
    )
    for i_col_, i_b in qd.ndrange(
        max_candidate_contacts,
        _B,
        axes=qd.static((1, 0) if static_rigid_sim_config.constraint_layout_batch_first else None),
    ):
        if i_col_ < collider_state.n_contacts[i_b]:
            collision_con_start = constraint_state.n_constraints[i_b]

            i_col = collider_state.contact_sort_idx[i_col_, i_b]
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

            # A contact needs a constraint only when at least one endpoint is an awake dynamic body; if both are
            # hibernated or fixed, no awake dof is acted upon and the contact carries no constraint. A sleeper struck
            # by an awake body was already revived in the broad phase, so it does not reach this branch. The slots are
            # reused by index across steps, so a skipped contact must actively clear its slots and mark them inert:
            # leaving the stale jacobian of a prior step (when those dofs were awake and in contact) would leak that
            # contact force into the qfrc_constraint of a since-woken body that now shares the slot.
            if qd.static(static_rigid_sim_config.use_hibernation):
                is_a_awake = not (links_info.is_fixed[link_a_maybe_batch] or links_state.is_hibernated[link_a, i_b])
                is_b_awake = link_b >= 0 and not (
                    links_info.is_fixed[link_b_maybe_batch] or links_state.is_hibernated[link_b, i_b]
                )
                if not is_a_awake and not is_b_awake:
                    for i_friction in range(4):
                        n_con = collision_con_start + i_col_ * 4 + i_friction
                        if qd.static(static_rigid_sim_config.sparse_solve):
                            for i_d_ in range(constraint_state.jac_n_dofs[n_con, i_b]):
                                i_d = constraint_state.jac_dofs_idx[n_con, i_d_, i_b]
                                constraint_state.jac[n_con, i_d, i_b] = gs.qd_float(0.0)
                            constraint_state.jac_n_dofs[n_con, i_b] = 0
                        else:
                            for i_d in range(n_dofs):
                                constraint_state.jac[n_con, i_d, i_b] = gs.qd_float(0.0)
                        constraint_state.diag[n_con, i_b] = gs.qd_float(1.0)
                        constraint_state.aref[n_con, i_b] = gs.qd_float(0.0)
                        constraint_state.efc_D[n_con, i_b] = gs.qd_float(0.0)
                    continue

            d1, d2 = gu.qd_orthogonals(contact_data_normal)

            invweight = links_info.invweight[link_a_maybe_batch][0]
            if link_b > -1:
                invweight = invweight + links_info.invweight[link_b_maybe_batch][0]

            for i_friction in range(4):
                d = (2 * (i_friction % 2) - 1) * (d1 if i_friction < 2 else d2)
                n = d * contact_data_friction - contact_data_normal

                n_con = collision_con_start + i_col_ * 4 + i_friction
                if qd.static(static_rigid_sim_config.sparse_solve):
                    for i_d_ in range(constraint_state.jac_n_dofs[n_con, i_b]):
                        i_d = constraint_state.jac_dofs_idx[n_con, i_d_, i_b]
                        constraint_state.jac[n_con, i_d, i_b] = gs.qd_float(0.0)
                else:
                    for i_d in range(n_dofs):
                        constraint_state.jac[n_con, i_d, i_b] = gs.qd_float(0.0)

                con_n_dofs = 0
                jac_qvel = gs.qd_float(0.0)
                for i_ab in range(2):
                    sign = gs.qd_float(-1.0)
                    link = link_a
                    if i_ab == 1:
                        sign = gs.qd_float(1.0)
                        link = link_b

                    while link > -1:
                        link_maybe_batch = [link, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else link

                        # reverse order to make sure dofs in each row of self.jac_dofs_idx are strictly descending
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

                            constraint_state.jac_dofs_idx[n_con, con_n_dofs, i_b] = i_d
                            con_n_dofs = con_n_dofs + 1

                        link = links_info.parent_idx[link_maybe_batch]

                constraint_state.jac_n_dofs[n_con, i_b] = con_n_dofs
                _sort_relevant_dofs_descending(constraint_state, n_con, con_n_dofs, i_b, static_rigid_sim_config)
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
def add_collision_constraints(
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    collider_state: array_class.ColliderState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    _B = dofs_state.ctrl_mode.shape[1]

    if qd.static(static_rigid_sim_config.enable_cooperative_constraint_kernels):
        _add_collision_constraints_per_friction(
            links_info=links_info,
            links_state=links_state,
            dofs_state=dofs_state,
            constraint_state=constraint_state,
            collider_state=collider_state,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )
    else:
        _add_collision_constraints_per_contact(
            links_info=links_info,
            links_state=links_state,
            dofs_state=dofs_state,
            constraint_state=constraint_state,
            collider_state=collider_state,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )

    qd.loop_config(name="add_collision_count", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        constraint_state.n_constraints[i_b] = constraint_state.n_constraints[i_b] + collider_state.n_contacts[i_b] * 4


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
        con_n_dofs = 0

        if qd.static(static_rigid_sim_config.sparse_solve):
            for i_d_ in range(constraint_state.jac_n_dofs[n_con, i_b]):
                i_d = constraint_state.jac_dofs_idx[n_con, i_d_, i_b]
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

                    constraint_state.jac_dofs_idx[n_con, con_n_dofs, i_b] = i_d
                    con_n_dofs = con_n_dofs + 1

                link = links_info.parent_idx[link_maybe_batch]

        constraint_state.jac_n_dofs[n_con, i_b] = con_n_dofs
        # Sort needed: DOFs from two entities are only descending within each
        # entity. Incremental Cholesky requires globally descending order.
        _sort_relevant_dofs_descending(constraint_state, n_con, con_n_dofs, i_b, static_rigid_sim_config)

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
        for i_d_ in range(constraint_state.jac_n_dofs[n_con, i_b]):
            i_d = constraint_state.jac_dofs_idx[n_con, i_d_, i_b]
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

    # Populate jac_dofs_idx for this joint-equality constraint, so the sparse-Jacobian iterations see its relevant
    # DOFs (otherwise they would see 0 and produce zero forces, leading to NaN in the solver).
    con_n_dofs = 0
    constraint_state.jac_dofs_idx[n_con, con_n_dofs, i_b] = i_dof1
    con_n_dofs += 1
    if i_dof2 != i_dof1:
        constraint_state.jac_dofs_idx[n_con, con_n_dofs, i_b] = i_dof2
        con_n_dofs += 1
    constraint_state.jac_n_dofs[n_con, i_b] = con_n_dofs
    _sort_relevant_dofs_descending(constraint_state, n_con, con_n_dofs, i_b, static_rigid_sim_config)


@qd.kernel(fastcache=True)
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


@qd.func
def _sort_contacts_per_island(
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    joints_info: array_class.JointsInfo,
    equalities_info: array_class.EqualitiesInfo,
    constraint_state: array_class.ConstraintState,
    collider_state: array_class.ColliderState,
    island_state: array_class.IslandState,
    static_rigid_sim_config: qd.template(),
    collider_static_config: qd.template(),
):
    """Build the island partition, order each island's contacts, and gather them into contact_sort_idx.

    The island build always runs (the per-island solve needs it) and the gather always runs (the solve reads contacts
    island-grouped); only the sort is gated on spatial_sort_supported. This makes the contact order match the
    islands-off path exactly when there is a single island: per-island grouping in physical scan order then equals the
    identity order the off path keeps, and the same comparator orders both.
    """
    _B = constraint_state.jac.shape[2]
    if qd.static(static_rigid_sim_config.enable_cooperative_constraint_kernels):
        # Warp-per-env: union-find island construction is serial per env, so lane 0 builds the partition; the per-island
        # contact sorts are independent (disjoint contact_id slices) so the warp's lanes take one island each in
        # parallel; then the island-grouped permutation is gathered back into contact_sort_idx (lane-strided).
        # block.sync fences each phase. The constraints assembled by the caller read the result.
        _K = qd.static(32)
        # Reset the (env, island) work-list counter before the per-env builds append to it (atomic reservation).
        for _i in range(1):
            island_state.factor_worklist_size[0] = 0
        qd.loop_config(name="build_and_sort_islands", block_dim=_K)
        for i_flat in range(_B * _K):
            tid = i_flat % _K
            i_b = i_flat // _K
            if tid == 0:
                func_build_islands(
                    i_b,
                    links_info,
                    links_state,
                    joints_info,
                    equalities_info,
                    constraint_state,
                    collider_state,
                    island_state,
                    static_rigid_sim_config,
                )
                # Append this env's islands to the work-list the cooperative factor+solve grid-strides over. Reserve a
                # contiguous block of slots with one atomic so the appends across envs do not interleave per island.
                n_islands = island_state.n_islands[i_b]
                base = qd.atomic_add(island_state.factor_worklist_size[0], n_islands)
                for i_island in range(n_islands):
                    island_state.factor_worklist_i_b[base + i_island] = i_b
                    island_state.factor_worklist_i_island[base + i_island] = i_island
            qd.simt.block.sync()
            if qd.static(collider_static_config.spatial_sort_supported):
                i_island = tid
                while i_island < island_state.n_islands[i_b]:
                    _sort_island_contacts(
                        i_b,
                        island_state.contact_slices.start[i_island, i_b],
                        island_state.contact_slices.n[i_island, i_b],
                        island_state.contact_id,
                        collider_state.contact_data.pos,
                        collider_state.contact_data.geom_a,
                        collider_state.contact_data.geom_b,
                    )
                    i_island = i_island + _K
                qd.simt.block.sync()
            total = func_island_contacts_total(i_b, island_state)
            i_c = tid
            while i_c < total:
                collider_state.contact_sort_idx[i_c, i_b] = island_state.contact_id[i_c, i_b]
                i_c = i_c + _K
            if tid == 0:
                collider_state.n_contacts[i_b] = total
    else:
        # CPU / non-cooperative: one thread per env builds the partition and sorts each island serially, then gathers
        # the permutation. Same result as the warp-per-env path (the lanes only parallelize independent islands), so the
        # contact order is identical regardless of backend.
        qd.loop_config(
            name="build_and_sort_islands",
            serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL),
        )
        for i_b in range(_B):
            func_build_islands(
                i_b,
                links_info,
                links_state,
                joints_info,
                equalities_info,
                constraint_state,
                collider_state,
                island_state,
                static_rigid_sim_config,
            )
            if qd.static(collider_static_config.spatial_sort_supported):
                for i_island in range(island_state.n_islands[i_b]):
                    _sort_island_contacts(
                        i_b,
                        island_state.contact_slices.start[i_island, i_b],
                        island_state.contact_slices.n[i_island, i_b],
                        island_state.contact_id,
                        collider_state.contact_data.pos,
                        collider_state.contact_data.geom_a,
                        collider_state.contact_data.geom_b,
                    )
            total = func_island_contacts_total(i_b, island_state)
            for i_c in range(total):
                collider_state.contact_sort_idx[i_c, i_b] = island_state.contact_id[i_c, i_b]
            collider_state.n_contacts[i_b] = total


@qd.kernel(fastcache=True)
def add_inequality_constraints(
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    joints_info: array_class.JointsInfo,
    equalities_info: array_class.EqualitiesInfo,
    constraint_state: array_class.ConstraintState,
    collider_state: array_class.ColliderState,
    island_state: array_class.IslandState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    collider_static_config: qd.template(),
):
    # Order the contacts deterministically BEFORE assembling the contact constraints below: the contact-constraint
    # index i_c follows the logical contact order (contact_sort_idx), so fixing that order here makes both the solve
    # order and get_contacts deterministic despite the racy atomic_add narrowphase layout. Done here rather than in the
    # collider (which has no notion of constraints) or in func_solve_init (too late - contacts are consumed just below).
    # With islands the order is built per-island (O(sum island^2)); without islands it is a single global pass. Both are
    # gated on spatial_sort_supported, and both use the same comparator, so a single island matches the off path
    # exactly. The off path still builds nothing - the collider's compacted contact_sort_idx is sorted in place.
    if qd.static(static_rigid_sim_config.enable_per_island_solve):
        _sort_contacts_per_island(
            links_info,
            links_state,
            joints_info,
            equalities_info,
            constraint_state,
            collider_state,
            island_state,
            static_rigid_sim_config,
            collider_static_config,
        )
    elif qd.static(collider_static_config.spatial_sort_supported):
        _B = constraint_state.jac.shape[2]
        qd.loop_config(
            name="sort_contacts",
            serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL),
        )
        for i_b in range(_B):
            _sort_island_contacts(
                i_b,
                0,
                collider_state.n_contacts[i_b],
                collider_state.contact_sort_idx,
                collider_state.contact_data.pos,
                collider_state.contact_data.geom_a,
                collider_state.contact_data.geom_b,
            )

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
        con_n_dofs = 0

        if qd.static(static_rigid_sim_config.sparse_solve):
            for i_d_ in range(constraint_state.jac_n_dofs[n_con, i_b]):
                i_d = constraint_state.jac_dofs_idx[n_con, i_d_, i_b]
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

                    constraint_state.jac_dofs_idx[n_con, con_n_dofs, i_b] = i_d
                    con_n_dofs = con_n_dofs + 1
                link = links_info.parent_idx[link_maybe_batch]

        constraint_state.jac_n_dofs[n_con, i_b] = con_n_dofs
        _sort_relevant_dofs_descending(constraint_state, n_con, con_n_dofs, i_b, static_rigid_sim_config)

        imp, aref = gu.imp_aref(sol_params, -pos_imp, jac_qvel, pos_error[i])
        diag = qd.max(invweight[0] * (1 - imp) / imp, EPS)

        constraint_state.diag[n_con, i_b] = diag
        constraint_state.aref[n_con, i_b] = aref
        constraint_state.efc_D[n_con, i_b] = 1.0 / diag

    # --- Orientation part (next 3 constraints) ---
    n_con = qd.atomic_add(constraint_state.n_constraints[i_b], 3)
    qd.atomic_add(constraint_state.n_constraints_equality[i_b], 3)
    con_n_dofs = 0
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

                # The 3 orientation constraints share the same support (the DOFs along both kinematic chains); record
                # it so sparse assembly does not drop them. (The position part above does the same per constraint.)
                for i_con in range(n_con, n_con + 3):
                    constraint_state.jac_dofs_idx[i_con, con_n_dofs, i_b] = i_d
                con_n_dofs = con_n_dofs + 1
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

    for i_con in range(n_con, n_con + 3):
        constraint_state.jac_n_dofs[i_con, i_b] = con_n_dofs
        _sort_relevant_dofs_descending(constraint_state, i_con, con_n_dofs, i_b, static_rigid_sim_config)

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
    qd.loop_config(
        name="add_joint_limit_constraints", serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    )
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
                            for i_d2_ in range(constraint_state.jac_n_dofs[n_con, i_b]):
                                i_d2 = constraint_state.jac_dofs_idx[n_con, i_d2_, i_b]
                                constraint_state.jac[n_con, i_d2, i_b] = gs.qd_float(0.0)
                        else:
                            for i_d2 in range(n_dofs):
                                constraint_state.jac[n_con, i_d2, i_b] = gs.qd_float(0.0)
                        constraint_state.jac[n_con, i_d, i_b] = jac

                        constraint_state.jac_n_dofs[n_con, i_b] = 1
                        constraint_state.jac_dofs_idx[n_con, 0, i_b] = i_d


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
        name="add_frictionloss_constraints",
        serialize=qd.static(
            static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL and static_rigid_sim_config.backend != gs.metal
        ),
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

                        constraint_state.jac_dofs_idx[i_con, 0, i_b] = i_d
                        constraint_state.jac_n_dofs[i_con, i_b] = 1


# ====================================== Runtime User-Specified Weld Constraints ======================================


@qd.kernel(fastcache=True)
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


@qd.kernel(fastcache=True)
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


@qd.kernel
def func_compute_dof_perm(
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    links_state: array_class.LinksState,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """Compute a fill-reducing DOF permutation by sorting DOFs on their body's COM (gravity axis first).

    Coupling in `H = M + J.T @ D @ J` is between spatially-near bodies (contacts) and within a body (M). Ordering DOFs
    by body position therefore keeps coupled DOFs index-adjacent, which bounds the skyline band regardless of the
    order bodies were added in. Each DOF keys on its entity's COM, so a whole entity's DOFs share a key and stay
    contiguous (ties broken by original index). The factorization runs in this permuted order; grad/Mgrad are mapped
    through dof_perm at the solve boundary so the rest of the solver is unchanged. Computed once from the initial
    layout, so the per-env insertion sort runs a single time and never in the step loop.
    """
    _B = constraint_state.grad.shape[1]
    n_dofs = constraint_state.nt_H.shape[1]

    qd.loop_config(name="compute_dof_perm", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        for i_d in range(n_dofs):
            constraint_state.dof_perm[i_b, i_d] = i_d

        # Reorder only when the envelope is active and not under MuJoCo compatibility (which needs the natural DOF
        # order); otherwise the permutation stays identity and everything downstream reduces to natural order.
        if qd.static(
            static_rigid_sim_config.sparse_envelope and not static_rigid_sim_config.enable_mujoco_compatibility
        ):
            for i_d in range(n_dofs):
                I_d = [i_d, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else i_d
                i_l = entities_info.link_start[dofs_info.entity_idx[I_d]]
                com = links_state.pos[i_l, i_b]
                constraint_state.dof_sort_key[i_b, i_d] = com[2] * 1.0e6 + com[1] * 1.0e3 + com[0]

            # Insertion sort dof_perm ascending by (key, original index); same-entity DOFs (equal key) keep order.
            for a in range(1, n_dofs):
                d = constraint_state.dof_perm[i_b, a]
                ka = constraint_state.dof_sort_key[i_b, d]
                j = a - 1
                while j >= 0:
                    dj = constraint_state.dof_perm[i_b, j]
                    kj = constraint_state.dof_sort_key[i_b, dj]
                    if kj < ka or (kj == ka and dj < d):
                        break
                    constraint_state.dof_perm[i_b, j + 1] = dj
                    j = j - 1
                constraint_state.dof_perm[i_b, j + 1] = d

        for p in range(n_dofs):
            constraint_state.dof_iperm[i_b, constraint_state.dof_perm[i_b, p]] = p


@qd.func
def func_compute_sparsity_pattern(
    i_b,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
):
    """Compute the skyline envelope start of each Hessian row analytically, without inspecting the assembled matrix.

    `nt_H_env_start[i_b, i_d]` is the smallest column index that can be structurally nonzero in row `i_d` of
    `H = M + J.T @ D @ J`. It is determined by the two sources of coupling, both known a priori:
    - the kinematic tree (`mass_parent_mask`): `M` couples two DOFs only if one supports the other (same branch);
    - the constraint supports (`jac_dofs_idx`): a constraint couples all DOFs it depends on, so its smallest
      relevant DOF bounds the envelope of all the others.

    Cholesky fill-in stays within this envelope, so the factor and solve only ever visit `[env_start, i_d]`. The
    pattern is structural (independent of which constraints are active), so it is recomputed once per step. All
    coupling is mapped through dof_iperm to permuted positions (identity when reordering is off). This is a device
    function so it can run inside func_solve_init's launch rather than as a separate kernel dispatch per step.
    """
    n_dofs = constraint_state.nt_H.shape[1]

    for p in range(n_dofs):
        constraint_state.nt_H_env_start[i_b, p] = p

    # M part: kinematic-tree coupling (same branch), scatter-min onto the permuted positions. The mass matrix is
    # block-diagonal per kinematic tree, so coupling only occurs within a DOF's block; restricting the inner loop to
    # that block makes this scale with the sum of per-tree blocks instead of the whole env.
    for i_d in range(n_dofs):
        for j_d in range(rigid_global_info.dofs_mass_block_start[i_d], rigid_global_info.dofs_mass_block_end[i_d]):
            if rigid_global_info.mass_parent_mask[i_d, j_d] > 0.5:
                p_i = constraint_state.dof_iperm[i_b, i_d]
                p_j = constraint_state.dof_iperm[i_b, j_d]
                row = qd.max(p_i, p_j)
                col = qd.min(p_i, p_j)
                if col < constraint_state.nt_H_env_start[i_b, row]:
                    constraint_state.nt_H_env_start[i_b, row] = col

    # J.T @ D @ J part: each constraint couples all DOFs in its support; the smallest permuted index bounds the rest.
    for i_c in range(constraint_state.n_constraints[i_b]):
        n_rel = constraint_state.jac_n_dofs[i_c, i_b]
        col_min = n_dofs
        for k in range(n_rel):
            p = constraint_state.dof_iperm[i_b, constraint_state.jac_dofs_idx[i_c, k, i_b]]
            if p < col_min:
                col_min = p
        for k in range(n_rel):
            p = constraint_state.dof_iperm[i_b, constraint_state.jac_dofs_idx[i_c, k, i_b]]
            if col_min < constraint_state.nt_H_env_start[i_b, p]:
                constraint_state.nt_H_env_start[i_b, p] = col_min


@qd.func
def func_compute_island_envelope(
    i_b,
    i_island,
    island_state: array_class.IslandState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
):
    """Compute one island's skyline envelope: the smallest island-local column that can be structurally nonzero in
    each local row of its Hessian block H = M + J.T @ D @ J. The two coupling sources are known a priori:
    - constraint supports: a constraint couples all DOFs in its support, so its smallest local DOF bounds the others;
    - mass: the kinematic tree (mass_parent_mask) couples a DOF with its ancestors and same-link DOFs.

    The island's DOFs are gathered in ascending global order (dof_id), so local order matches global order and the
    envelope is a valid band. Computed once per step (structural) and reused across Newton iterations.
    """
    EPS = rigid_global_info.EPS[None]
    n = island_state.dof_slices.n[i_island, i_b]
    dof_base = island_state.dof_slices.start[i_island, i_b]
    con_base = island_state.constraint_slices.start[i_island, i_b]
    con_n = island_state.constraint_slices.n[i_island, i_b]

    for ld in range(n):
        island_state.dof_env_start_local[dof_base + ld, i_b] = ld

    # Constraint coupling: scanning the island's DOFs in ascending order, the first DOF a constraint touches is its
    # smallest local column and bounds the envelope of every other DOF it touches.
    for i_lcon in range(con_n):
        i_c = island_state.constraint_id[con_base + i_lcon, i_b]
        col_min = n
        for ld in range(n):
            i_dg = island_state.dof_id[dof_base + ld, i_b]
            if qd.abs(constraint_state.jac[i_c, i_dg, i_b]) > EPS:
                if col_min == n:
                    col_min = ld
                elif col_min < island_state.dof_env_start_local[dof_base + ld, i_b]:
                    island_state.dof_env_start_local[dof_base + ld, i_b] = col_min

    # Mass coupling: the kinematic-tree mask is directional (descendant -> ancestor) plus full intra-link, so check
    # both orientations. DOFs are ascending, so the first coupled lower column is the smallest.
    for ld in range(n):
        i_dg = island_state.dof_id[dof_base + ld, i_b]
        for ld2 in range(ld):
            j_dg = island_state.dof_id[dof_base + ld2, i_b]
            if (
                rigid_global_info.mass_parent_mask[i_dg, j_dg] > 0.5
                or rigid_global_info.mass_parent_mask[j_dg, i_dg] > 0.5
            ):
                if ld2 < island_state.dof_env_start_local[dof_base + ld, i_b]:
                    island_state.dof_env_start_local[dof_base + ld, i_b] = ld2
                break


@qd.func
def func_hessian_direct_batch(
    i_b,
    i_island,
    island_state: array_class.IslandState,
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Compute the Hessian H = M + J.T @ D @ J of one work-unit. Only the lower triangle is written (H is
    symmetric); the solver always reads from the lower triangle.

    With islands, the unit is island i_island: its n x n block is assembled in place into the lower triangle of
    nt_H[i_b] at the island's global DOF rows/cols (dof_id maps the island's local index to its global dof, in
    ascending order). The global Hessian is block-diagonal by island, so the off-block entries are left untouched.
    The island's constraints are listed in constraint_id[constraint.start : +constraint.n]. Without islands, the
    unit is the whole env and i_island is unused: H is nt_H[i_b], with sparse_solve permuting storage via dof_iperm.
    """
    EPS = rigid_global_info.EPS[None]

    if qd.static(static_rigid_sim_config.enable_per_island_solve):
        n = island_state.dof_slices.n[i_island, i_b]
        dof_base = island_state.dof_slices.start[i_island, i_b]
        con_base = island_state.constraint_slices.start[i_island, i_b]
        con_n = island_state.constraint_slices.n[i_island, i_b]
        # The island's DOFs are gathered in ascending global order (dof_id), so its block lives in the lower
        # triangle of nt_H at those global rows/cols. Off-block (cross-island) entries are left untouched: the
        # Hessian is block-diagonal by island and the per-island factor/solve only ever read within the block.
        # All three passes visit only the row's skyline envelope [env_start, i_d]: entries below env_start are
        # structurally zero (no constraint or mass coupling reaches them), so zeroing, the J.T D J add and the mass
        # add can all skip them. The factor and solve read within the same band, so the block factors as a band.
        for i_d in range(n):
            i_dg = island_state.dof_id[dof_base + i_d, i_b]
            env_i = island_state.dof_env_start_local[dof_base + i_d, i_b]
            for j_d in range(env_i, i_d + 1):
                j_dg = island_state.dof_id[dof_base + j_d, i_b]
                constraint_state.nt_H[i_b, i_dg, j_dg] = gs.qd_float(0.0)
        # H += J.T @ D @ J by scattering each island constraint's rank update over the DOF pairs in its support
        # (jac_dofs_idx), writing the lower triangle at global rows/cols. This is O(sum n_support^2) instead of the
        # O(n_dofs * n_constraints) row-by-constraint scan, which matters when an island carries many contacts.
        for i_lcon in range(con_n):
            i_c = island_state.constraint_id[con_base + i_lcon, i_b]
            jac_n = constraint_state.jac_n_dofs[i_c, i_b]
            for i_d1_ in range(jac_n):
                i_d1 = constraint_state.jac_dofs_idx[i_c, i_d1_, i_b]
                if qd.abs(constraint_state.jac[i_c, i_d1, i_b]) > EPS:
                    for i_d2_ in range(i_d1_, jac_n):
                        i_d2 = constraint_state.jac_dofs_idx[i_c, i_d2_, i_b]
                        row = qd.max(i_d1, i_d2)
                        col = qd.min(i_d1, i_d2)
                        constraint_state.nt_H[i_b, row, col] = (
                            constraint_state.nt_H[i_b, row, col]
                            + constraint_state.jac[i_c, i_d1, i_b]
                            * constraint_state.jac[i_c, i_d2, i_b]
                            * constraint_state.efc_D[i_c, i_b]
                            * constraint_state.active[i_c, i_b]
                        )
        # H += M, restricted to the island's DOFs. Mass couples only DOFs within the same component (one island), so
        # the block is dense over the island's gathered DOFs and never reaches another island's block. Iterating the
        # gathered dof_id (rather than an entity's contiguous range) keeps the add inside this island even when a
        # component's DOFs are non-contiguous (e.g. an entity whose free bodies interleave in DOF order).
        for i_d in range(n):
            i_dg = island_state.dof_id[dof_base + i_d, i_b]
            env_i = island_state.dof_env_start_local[dof_base + i_d, i_b]
            for j_d in range(env_i, i_d + 1):
                j_dg = island_state.dof_id[dof_base + j_d, i_b]
                constraint_state.nt_H[i_b, i_dg, j_dg] = (
                    constraint_state.nt_H[i_b, i_dg, j_dg] + rigid_global_info.mass_mat[i_dg, j_dg, i_b]
                )
        return

    n_dofs = constraint_state.nt_H.shape[1]
    n_entities = entities_info.n_links.shape[0]

    # Reset Hessian matrix to zero
    for i_d1 in range(n_dofs):
        for i_d2 in range(i_d1 + 1):
            constraint_state.nt_H[i_b, i_d1, i_d2] = gs.qd_float(0.0)

    # Compute `H += J.T @ D @ J` using either dense or sparse implementation
    if qd.static(static_rigid_sim_config.sparse_solve):
        for i_c in range(constraint_state.n_constraints[i_b]):
            jac_n_dofs = constraint_state.jac_n_dofs[i_c, i_b]
            for i_d1_ in range(jac_n_dofs):
                i_d1 = constraint_state.jac_dofs_idx[i_c, i_d1_, i_b]
                if qd.abs(constraint_state.jac[i_c, i_d1, i_b]) > EPS:
                    for i_d2_ in range(i_d1_, jac_n_dofs):
                        i_d2 = constraint_state.jac_dofs_idx[i_c, i_d2_, i_b]
                        # Write to permuted positions (identity when reordering is off). jac/efc_D are read in natural
                        # DOF order; only the Hessian storage position is permuted.
                        p1 = constraint_state.dof_iperm[i_b, i_d1]
                        p2 = constraint_state.dof_iperm[i_b, i_d2]
                        row = qd.max(p1, p2)
                        col = qd.min(p1, p2)
                        constraint_state.nt_H[i_b, row, col] = (
                            constraint_state.nt_H[i_b, row, col]
                            + constraint_state.jac[i_c, i_d1, i_b]
                            * constraint_state.jac[i_c, i_d2, i_b]
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

    # Compute `H += M`. With sparse_solve the storage position is permuted via dof_iperm; otherwise it is natural
    # (dof_iperm is only populated on the sparse path).
    for i_e in range(n_entities):
        for i_d1 in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
            # Mass couples only DOFs within the same kinematic-tree block, so cross-block entries are zero and skipped.
            for i_d2 in range(rigid_global_info.dofs_mass_block_start[i_d1], i_d1 + 1):
                if qd.static(static_rigid_sim_config.sparse_solve):
                    p1 = constraint_state.dof_iperm[i_b, i_d1]
                    p2 = constraint_state.dof_iperm[i_b, i_d2]
                    row = qd.max(p1, p2)
                    col = qd.min(p1, p2)
                    constraint_state.nt_H[i_b, row, col] = (
                        constraint_state.nt_H[i_b, row, col] + rigid_global_info.mass_mat[i_d1, i_d2, i_b]
                    )
                else:
                    constraint_state.nt_H[i_b, i_d1, i_d2] = (
                        constraint_state.nt_H[i_b, i_d1, i_d2] + rigid_global_info.mass_mat[i_d1, i_d2, i_b]
                    )


@qd.func
def func_island_assemble_factor_solve_tiled(
    i_b,
    i_island,
    tid,
    L_sh,
    v_sh,
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    island_state: array_class.IslandState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    do_assemble: qd.template(),
    TileCls: qd.template(),
    write_L: qd.template() = False,
):
    """Barrier-free tiled Cholesky factor + triangular solve of one island's Newton system.

    Operates on the island's contiguous block [gbase, gbase+n) of nt_H with the same register-streaming TileTxT
    primitives as the non-island whole-env factor (func_cholesky_factor_direct_tiled). For a single island spanning
    the whole env this IS that path, so islands-on matches islands-off; for multiple islands each contiguous block is
    factored independently. The rare non-contiguous island (a connected cluster whose gathered DOFs are not a single
    ascending run) falls back to the scalar per-island solve on lane 0.

    When do_assemble is True the contiguous block is assembled from jac/efc into nt_H first (the caller has not built
    H); when False nt_H already holds the incrementally maintained Hessian and only the factor + solve run. The scalar
    fallback always assembles, as its scattered DOFs are not a readable contiguous block.

    L stays in the shared tile L_sh (local island indices); grad/Mgrad are global, reached at gbase + local. block.sync
    fences the assembly before the cooperative factor and the result before the caller's termination test.

    write_L persists L from L_sh back into the island's nt_H block, needed when a later step reads L from nt_H rather
    than re-factoring (the monolith's incremental rank-1 iterations); the decomposed graph re-factors every iteration
    so it leaves write_L False and keeps nt_H holding the raw Hessian.
    """
    T = qd.static(static_rigid_sim_config.cholesky_tile_size)
    LOG2_T = qd.static(T.bit_length() - 1)
    EPS = rigid_global_info.EPS[None]

    n = island_state.dof_slices.n[i_island, i_b]
    dof_base = island_state.dof_slices.start[i_island, i_b]
    con_base = island_state.constraint_slices.start[i_island, i_b]
    con_n = island_state.constraint_slices.n[i_island, i_b]
    gbase = island_state.dof_id[dof_base, i_b]

    # The island's gathered DOFs are ascending, so the block is contiguous iff first and last span exactly n indices.
    # A contiguous block factors with the register-streaming tiled algorithm: if it fits the shared tile
    # (n <= tiled_n_island_dofs) L stays in shared memory (fused factor + solve); otherwise L stays in nt_H global
    # (no shared-memory DOF cap), so even a whole-body-sized island avoids the serial scalar solve. A non-contiguous
    # island (gathered DOFs are not a single ascending run) falls back to the scalar per-island solve. Quadrants
    # forbids `return` inside a runtime branch, so these are an if/elif/else.
    is_contiguous = island_state.dof_id[dof_base + n - 1, i_b] == gbase + n - 1
    if is_contiguous and n <= qd.static(static_rigid_sim_config.tiled_n_island_dofs):
        # --- Assemble the full lower triangle of the island block into nt_H (T threads, row-striped) ---
        if qd.static(do_assemble):
            i_d = tid
            while i_d < n:
                gi = gbase + i_d
                for j in range(i_d + 1):
                    constraint_state.nt_H[i_b, gi, gbase + j] = gs.qd_float(0.0)
                for i_lcon in range(con_n):
                    i_c = island_state.constraint_id[con_base + i_lcon, i_b]
                    jac_i = constraint_state.jac[i_c, gi, i_b]
                    if qd.abs(jac_i) > EPS:
                        w = jac_i * constraint_state.efc_D[i_c, i_b] * constraint_state.active[i_c, i_b]
                        for j in range(i_d + 1):
                            gj = gbase + j
                            constraint_state.nt_H[i_b, gi, gj] = (
                                constraint_state.nt_H[i_b, gi, gj] + constraint_state.jac[i_c, gj, i_b] * w
                            )
                for j in range(i_d + 1):
                    gj = gbase + j
                    constraint_state.nt_H[i_b, gi, gj] = (
                        constraint_state.nt_H[i_b, gi, gj] + rigid_global_info.mass_mat[gi, gj, i_b]
                    )
                i_d = i_d + T
            qd.simt.block.sync()

        # --- Blocked left-looking Cholesky into L_sh (register tiles, no block sync), reading the island block ---
        N_BLOCKS = (n + T - 1) // T
        for kb in range(N_BLOCKS):
            lk0 = kb * T
            lk1 = qd.min(lk0 + T, n)
            gk0 = gbase + lk0
            gk1 = gbase + lk1
            L_kk = TileCls.eye(dtype=gs.qd_float)
            L_kk[:] = constraint_state.nt_H[i_b, gk0:gk1, gk0:gk1]
            for jb in range(kb):
                lj0 = jb * T
                for t in range(T):
                    v = L_sh[lk0:lk1, lj0 + t]
                    L_kk -= qd.outer(v, v)
            L_kk.cholesky_(EPS)
            for ib in range(kb + 1, N_BLOCKS):
                li0 = ib * T
                li1 = qd.min(li0 + T, n)
                gi0 = gbase + li0
                gi1 = gbase + li1
                L_ik = TileCls.zeros(dtype=gs.qd_float)
                L_ik[:] = constraint_state.nt_H[i_b, gi0:gi1, gk0:gk1]
                for jb in range(kb):
                    lj0 = jb * T
                    for t in range(T):
                        v_own = L_sh[li0:li1, lj0 + t]
                        v_diag = L_sh[lk0:lk1, lj0 + t]
                        L_ik -= qd.outer(v_own, v_diag)
                L_kk.solve_triangular_(L_ik)
                L_sh[li0:li1, lk0:lk1] = L_ik
            L_sh[lk0:lk1, lk0:lk1] = L_kk

        # --- Triangular solve grad -> Mgrad from L_sh (local indices; grad/Mgrad global at gbase + local) ---
        k = tid
        while k < n:
            v_sh[k] = constraint_state.grad[gbase + k, i_b]
            k = k + T
        qd.simt.block.sync()
        for i_r in range(n):
            dot = gs.qd_float(0.0)
            j = tid
            while j < i_r:
                dot = dot + L_sh[i_r, j] * v_sh[j]
                j = j + T
            dot = qd.simt.subgroup.reduce_all_add_tiled(dot, LOG2_T)
            if tid == 0:
                v_sh[i_r] = (v_sh[i_r] - dot) / L_sh[i_r, i_r]
            qd.simt.block.sync()
        for i_r_ in range(n):
            i_r = n - 1 - i_r_
            dot = gs.qd_float(0.0)
            j = i_r + 1 + tid
            while j < n:
                dot = dot + L_sh[j, i_r] * v_sh[j]
                j = j + T
            dot = qd.simt.subgroup.reduce_all_add_tiled(dot, LOG2_T)
            if tid == 0:
                v_sh[i_r] = (v_sh[i_r] - dot) / L_sh[i_r, i_r]
            qd.simt.block.sync()

        # Write the solved Mgrad back to global memory (local v_sh -> global at gbase + local).
        k = tid
        while k < n:
            constraint_state.Mgrad[gbase + k, i_b] = v_sh[k]
            k = k + T
        qd.simt.block.sync()

        # Persist the factor: store L's lower triangle (local L_sh) into the island's nt_H block so a caller that
        # reads L from nt_H instead of re-factoring (the monolith's incremental rank-1 iterations) finds it there.
        if qd.static(write_L):
            i_r = tid
            while i_r < n:
                gi = gbase + i_r
                for j in range(i_r + 1):
                    constraint_state.nt_H[i_b, gi, gbase + j] = L_sh[i_r, j]
                i_r = i_r + T
            qd.simt.block.sync()
    elif is_contiguous:
        # Contiguous island too large for the shared tile: factor with the same register-streaming tiled algorithm as
        # the whole-env path (func_cholesky_factor_direct_tiled), but keep L in nt_H global so there is no DOF cap.
        # A T-threaded triangular solve then reads L from nt_H using Mgrad as the working vector. This replaces the
        # serial scalar solve, whose O(n^3) factor on a single lane dominates for big islands (e.g. a humanoid body).
        if qd.static(do_assemble):
            i_d = tid
            while i_d < n:
                gi = gbase + i_d
                for j in range(i_d + 1):
                    constraint_state.nt_H[i_b, gi, gbase + j] = gs.qd_float(0.0)
                for i_lcon in range(con_n):
                    i_c = island_state.constraint_id[con_base + i_lcon, i_b]
                    jac_i = constraint_state.jac[i_c, gi, i_b]
                    if qd.abs(jac_i) > EPS:
                        w = jac_i * constraint_state.efc_D[i_c, i_b] * constraint_state.active[i_c, i_b]
                        for j in range(i_d + 1):
                            gj = gbase + j
                            constraint_state.nt_H[i_b, gi, gj] = (
                                constraint_state.nt_H[i_b, gi, gj] + constraint_state.jac[i_c, gj, i_b] * w
                            )
                for j in range(i_d + 1):
                    gj = gbase + j
                    constraint_state.nt_H[i_b, gi, gj] = (
                        constraint_state.nt_H[i_b, gi, gj] + rigid_global_info.mass_mat[gi, gj, i_b]
                    )
                i_d = i_d + T
            qd.simt.block.sync()

        # Left-looking blocked Cholesky with register tiles, prior L columns read back from nt_H (block_dim == T == one
        # subgroup, so the cooperative tile loads/stores are lockstep - no block.sync between column blocks needed).
        N_BLOCKS = (n + T - 1) // T
        for kb in range(N_BLOCKS):
            lk0 = kb * T
            lk1 = qd.min(lk0 + T, n)
            gk0 = gbase + lk0
            gk1 = gbase + lk1
            L_kk = TileCls.eye(dtype=gs.qd_float)
            L_kk[:] = constraint_state.nt_H[i_b, gk0:gk1, gk0:gk1]
            for jb in range(kb):
                lj0 = jb * T
                for t in range(T):
                    v = constraint_state.nt_H[i_b, gk0:gk1, gbase + lj0 + t]
                    L_kk -= qd.outer(v, v)
            L_kk.cholesky_(EPS)
            for ib in range(kb + 1, N_BLOCKS):
                li0 = ib * T
                li1 = qd.min(li0 + T, n)
                gi0 = gbase + li0
                gi1 = gbase + li1
                L_ik = TileCls.zeros(dtype=gs.qd_float)
                L_ik[:] = constraint_state.nt_H[i_b, gi0:gi1, gk0:gk1]
                for jb in range(kb):
                    lj0 = jb * T
                    for t in range(T):
                        v_own = constraint_state.nt_H[i_b, gi0:gi1, gbase + lj0 + t]
                        v_diag = constraint_state.nt_H[i_b, gk0:gk1, gbase + lj0 + t]
                        L_ik -= qd.outer(v_own, v_diag)
                L_kk.solve_triangular_(L_ik)
                constraint_state.nt_H[i_b, gi0:gi1, gk0:gk1] = L_ik
            constraint_state.nt_H[i_b, gk0:gk1, gk0:gk1] = L_kk
        qd.simt.block.sync()

        # Triangular solve L L^T x = grad -> Mgrad, reading L from nt_H. Mgrad is the working vector (no shared tile,
        # so no DOF cap); the T threads stripe each row's dot product and lane 0 writes the solved entry.
        k = tid
        while k < n:
            constraint_state.Mgrad[gbase + k, i_b] = constraint_state.grad[gbase + k, i_b]
            k = k + T
        qd.simt.block.sync()
        for i_r in range(n):
            dot = gs.qd_float(0.0)
            j = tid
            while j < i_r:
                dot = dot + constraint_state.nt_H[i_b, gbase + i_r, gbase + j] * constraint_state.Mgrad[gbase + j, i_b]
                j = j + T
            dot = qd.simt.subgroup.reduce_all_add_tiled(dot, LOG2_T)
            if tid == 0:
                constraint_state.Mgrad[gbase + i_r, i_b] = (
                    constraint_state.Mgrad[gbase + i_r, i_b] - dot
                ) / constraint_state.nt_H[i_b, gbase + i_r, gbase + i_r]
            qd.simt.block.sync()
        for i_r_ in range(n):
            i_r = n - 1 - i_r_
            dot = gs.qd_float(0.0)
            j = i_r + 1 + tid
            while j < n:
                dot = dot + constraint_state.nt_H[i_b, gbase + j, gbase + i_r] * constraint_state.Mgrad[gbase + j, i_b]
                j = j + T
            dot = qd.simt.subgroup.reduce_all_add_tiled(dot, LOG2_T)
            if tid == 0:
                constraint_state.Mgrad[gbase + i_r, i_b] = (
                    constraint_state.Mgrad[gbase + i_r, i_b] - dot
                ) / constraint_state.nt_H[i_b, gbase + i_r, gbase + i_r]
            qd.simt.block.sync()
    else:
        # Non-contiguous island (gathered DOFs are not a single ascending run): scalar per-island solve on lane 0,
        # which writes both L (func_cholesky_factor_direct_batch) and Mgrad (func_cholesky_solve_batch) to global.
        if tid == 0:
            func_hessian_direct_batch(
                i_b, i_island, island_state, entities_info, constraint_state, rigid_global_info, static_rigid_sim_config
            )
            func_cholesky_factor_direct_batch(
                i_b, i_island, island_state, constraint_state, rigid_global_info, static_rigid_sim_config
            )
            func_cholesky_solve_batch(i_b, i_island, island_state, constraint_state, static_rigid_sim_config)
        qd.simt.block.sync()


@qd.func
def func_island_tiled_factor_solve_all(
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    island_state: array_class.IslandState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    TileCls: qd.template(),
    do_assemble: qd.template() = False,
    write_L: qd.template() = False,
):
    # Barrier-free per-island factor + solve over the compact (env, island) work-list. Drives
    # func_island_assemble_factor_solve_tiled, which for a single island spanning the whole env is exactly the
    # non-island whole-env tiled factor - so an unpartitioned (1-island) island solves identically to the legacy
    # non-island path. grad must already hold M*acc - force - qfrc (the no-solve gradient). do_assemble=True builds each
    # island's Hessian block into nt_H first (used by the seed, where nt_H is not yet populated); the graph leaves it
    # False because it maintains nt_H incrementally before calling this. write_L persists L into nt_H for a caller that
    # reads the factor back (the monolith seed); the graph re-factors so it leaves it False.
    # A static grid of T-lane blocks grid-strides over the compact (env, island) work-list the partition build
    # materialized, so the block count (island_factor_n_blocks, sized to saturate the GPU) is decoupled from the env
    # count: a small batch with many islands fans its islands across blocks instead of serializing them inside a single
    # block-per-env, while a large batch reuses each block - and its one shared-tile reservation - across several work
    # items. Gridding per-(env, island) over max_islands = n_links instead launched a tile-reserving block for every
    # POTENTIAL island, whose idle reservations collapsed occupancy at many envs. The shared tile is sized to the
    # per-island capacity tiled_n_island_dofs (always fits shared); an island larger than that cap falls back to the
    # scalar per-island solve inside func_island_assemble_factor_solve_tiled. All T lanes of a block read the same work
    # item, so n_constraints/improved and the hibernation/contiguity branches are uniform and the per-island block.sync
    # is well-formed.
    N_BLOCKS = qd.static(static_rigid_sim_config.island_factor_n_blocks)
    MAX_DOFS = qd.static(static_rigid_sim_config.tiled_n_island_dofs)
    T = qd.static(static_rigid_sim_config.cholesky_tile_size)
    n_work = island_state.factor_worklist_size[0]
    qd.loop_config(block_dim=T)
    for i in range(N_BLOCKS * T):
        blk = i // T
        tid = i % T
        L_sh = qd.simt.block.SharedArray((MAX_DOFS, MAX_DOFS + 1), gs.qd_float)
        v_sh = qd.simt.block.SharedArray((MAX_DOFS,), gs.qd_float)
        i_work = blk
        while i_work < n_work:
            i_b = island_state.factor_worklist_i_b[i_work]
            i_island = island_state.factor_worklist_i_island[i_work]
            if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
                do_island = True
                if qd.static(static_rigid_sim_config.use_hibernation):
                    if island_state.is_hibernated[i_island, i_b]:
                        do_island = False
                if do_island:
                    func_island_assemble_factor_solve_tiled(
                        i_b,
                        i_island,
                        tid,
                        L_sh,
                        v_sh,
                        entities_info,
                        constraint_state,
                        island_state,
                        rigid_global_info,
                        static_rigid_sim_config,
                        do_assemble,
                        TileCls,
                        write_L,
                    )
            # Fence the shared tile before this block reuses it for its next work item.
            qd.simt.block.sync()
            i_work = i_work + N_BLOCKS


@qd.func
def func_hessian_direct_tiled(
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    check_full_hessian: qd.template() = False,
):
    """Compute the Hessian matrix `H = M + J.T @ D @ J of the optimization problem for all environment at once.

    This implementation is specialized for GPU backend and highly optimized for it using shared memory and cooperative
    threading.

    Under the hood, it implements a square-block matrix partitioned production algorithm to support arbitrary matrix
    sizes because shared memory storage is limited to 48kB. It boils down to classical matrix production if the entire
    optimization problem fits in a single block, i.e. n_constraints <= 32 and n_dofs <= 64.

    Note that only the lower triangular part will be updated for efficiency, because the Hessian matrix is symmetric.

    When check_full_hessian is True (used with H patching), skips envs where use_full_hessian == 0 (those get patched
    instead of rebuilt).
    """
    _B = constraint_state.grad.shape[1]
    n_dofs = constraint_state.nt_H.shape[1]

    # BLOCK_DIM = 128 is optimal, after grid searching ofter block_dim = 64, 128, 256, and evaluating
    # the test_rigid_benchmarks.py in production.yml for each value.
    BLOCK_DIM = qd.static(128)
    MAX_DOFS_PER_BLOCK = qd.static(64)
    # Note: setting MAX_CONSTRAINTS_PER_BLOCK to 64 provides a benefit for anymal_uniform_kinematic cpu
    # bs=0 (+14%), but a regression on anymal_uniform cuda ndarray (-9%). Generally gives better
    # performance on CPU, but worse on CUDA.
    MAX_CONSTRAINTS_PER_BLOCK = qd.static(32)

    n_lower_tri = n_dofs * (n_dofs + 1) // 2

    # FIXME: Adding `serialize=False` is causing sync failing for some reason...
    # TODO: Consider moving `H += M` in a dedicated CUDA kernel. It should be both simpler and faster.
    qd.loop_config(name="hessian_direct_tiled", block_dim=BLOCK_DIM)
    for i in range(_B * BLOCK_DIM):
        tid = i % BLOCK_DIM
        i_b = i // BLOCK_DIM
        if i_b >= _B:
            continue
        if constraint_state.n_constraints[i_b] == 0 or not constraint_state.improved[i_b]:
            continue
        if qd.static(check_full_hessian):
            if constraint_state.use_full_hessian[i_b] == 0:
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
                    if is_diag_tile:
                        n_lower_tri_tile = n_dofs_tile_row * (n_dofs_tile_row + 1) // 2
                        pid = tid
                        while pid < n_lower_tri_tile:
                            i_d1_, i_d2_ = linear_to_lower_tri(pid)
                            i_d1 = i_d1_ + i_d1_start
                            i_d2 = i_d2_ + i_d2_start
                            coef = gs.qd_float(0.0)
                            if i_c_start == 0:
                                coef = rigid_global_info.mass_mat[i_d1, i_d2, i_b]
                            for j_c_ in range(n_conts_tile):
                                coef = coef + jac_row[j_c_, i_d1_] * jac_row[j_c_, i_d2_] * efc_D[j_c_]
                            if i_c_start == 0:
                                constraint_state.nt_H[i_b, i_d1, i_d2] = coef
                            else:
                                constraint_state.nt_H[i_b, i_d1, i_d2] = constraint_state.nt_H[i_b, i_d1, i_d2] + coef
                            pid = pid + BLOCK_DIM
                    else:
                        numel = n_dofs_tile_row * n_dofs_tile_col
                        pid = tid
                        while pid < numel:
                            i_d1_ = pid // n_dofs_tile_col
                            i_d2_ = pid % n_dofs_tile_col
                            i_d1 = i_d1_ + i_d1_start
                            i_d2 = i_d2_ + i_d2_start
                            coef = gs.qd_float(0.0)
                            if i_c_start == 0:
                                coef = rigid_global_info.mass_mat[i_d1, i_d2, i_b]
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
                i_d1, i_d2 = linear_to_lower_tri(i_pair)
                constraint_state.nt_H[i_b, i_d1, i_d2] = rigid_global_info.mass_mat[i_d1, i_d2, i_b]
                i_pair = i_pair + BLOCK_DIM


@qd.func
def func_cholesky_factor_direct_batch(
    i_b,
    i_island,
    island_state: array_class.IslandState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Compute the Cholesky factorization L of one work-unit's Hessian H = L @ L.T in place.

    With islands, the unit is island i_island: H is its packed n x n tile at tile_start (element (a, b)
    at nt_H[i_b, tile_base + a * n + b]), and i_island selects it. Without islands, the unit is the whole
    env and i_island is unused: H is nt_H[i_b], and with the skyline envelope the factorization is restricted to
    each row's envelope (nt_H_env_start), confining Cholesky fill-in to the envelope.

    Beware the Hessian matrix is re-purposed to store its Cholesky factorization to spare memory resources. Only
    the lower triangular part is updated, because the Hessian matrix is symmetric.
    """
    EPS = rigid_global_info.EPS[None]

    if qd.static(static_rigid_sim_config.enable_per_island_solve):
        n = island_state.dof_slices.n[i_island, i_b]
        dof_base = island_state.dof_slices.start[i_island, i_b]
        # Factor the island's block in place at its global DOF rows/cols (dof_id is ascending, so all accesses below
        # stay in the lower triangle). The factorization is confined to each row's skyline envelope
        # (dof_env_start_local): a row's columns below its envelope start are structurally zero and fill-in stays
        # within the envelope, so a large island factors as a band instead of densely.
        for i_d in range(n):
            i_dg = island_state.dof_id[dof_base + i_d, i_b]
            i_start = island_state.dof_env_start_local[dof_base + i_d, i_b]
            tmp = constraint_state.nt_H[i_b, i_dg, i_dg]
            for j_d in range(i_start, i_d):
                j_dg = island_state.dof_id[dof_base + j_d, i_b]
                tmp = tmp - constraint_state.nt_H[i_b, i_dg, j_dg] ** 2
            constraint_state.nt_H[i_b, i_dg, i_dg] = qd.sqrt(qd.max(tmp, EPS))
            inv = 1.0 / constraint_state.nt_H[i_b, i_dg, i_dg]
            for j_d in range(i_d + 1, n):
                j_start = island_state.dof_env_start_local[dof_base + j_d, i_b]
                if j_start <= i_d:
                    j_dg = island_state.dof_id[dof_base + j_d, i_b]
                    dot = gs.qd_float(0.0)
                    for k_d in range(qd.max(i_start, j_start), i_d):
                        k_dg = island_state.dof_id[dof_base + k_d, i_b]
                        dot = dot + (constraint_state.nt_H[i_b, j_dg, k_dg] * constraint_state.nt_H[i_b, i_dg, k_dg])
                    constraint_state.nt_H[i_b, j_dg, i_dg] = (constraint_state.nt_H[i_b, j_dg, i_dg] - dot) * inv
        return

    n_dofs = constraint_state.nt_H.shape[1]

    # In-place factorization on nt_H (batch path never uses H patching)
    if qd.static(static_rigid_sim_config.sparse_envelope):
        for i_d in range(n_dofs):
            i_start = constraint_state.nt_H_env_start[i_b, i_d]
            tmp = constraint_state.nt_H[i_b, i_d, i_d]
            for k_d in range(i_start, i_d):
                tmp = tmp - constraint_state.nt_H[i_b, i_d, k_d] ** 2
            constraint_state.nt_H[i_b, i_d, i_d] = qd.sqrt(qd.max(tmp, EPS))

            tmp = 1.0 / constraint_state.nt_H[i_b, i_d, i_d]
            for j_d in range(i_d + 1, n_dofs):
                j_start = constraint_state.nt_H_env_start[i_b, j_d]
                if j_start <= i_d:
                    dot = gs.qd_float(0.0)
                    for k_d in range(qd.max(i_start, j_start), i_d):
                        dot = dot + constraint_state.nt_H[i_b, j_d, k_d] * constraint_state.nt_H[i_b, i_d, k_d]
                    constraint_state.nt_H[i_b, j_d, i_d] = (constraint_state.nt_H[i_b, j_d, i_d] - dot) * tmp
    else:
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
def _cholesky_factor_direct_tiled_impl(
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    TileCls: qd.template(),
):
    """Compute the Cholesky factorization L of the Hessian matrix H = L @ L.T for a given environment `i_b`.

    This implementation is specialized for GPU backend and highly optimized for it using a left-looking blocked algorithm
    with TileTxT primitives (potrf, trsm, syr_sub, ger_sub), all operating entirely in registers via subgroup shuffles.
    No shared memory or block synchronization needed. This function has no inherent DOF limit, but the fused variant
    (func_cholesky_and_solve_fused_tiled) requires shared memory for L, so the caller gates both behind the same
    shared-memory-based DOF threshold: n_dofs <= 64 (f64) or 96 (f32) with 48kB default shared memory, higher with
    opt-in shared memory (e.g. 160/224 on RTX PRO 6000).

    The tile size T (16 or 32) is dispatched at build time from static_rigid_sim_config.cholesky_tile_size based on
    n_dofs (see rigid_solver.py): T=16 for n_dofs in [1..16] or [33..48], T=32 for n_dofs in [17..32] or [49..].
    Confirmed at the endpoints by dex_hand (n_dofs=62, T=32 +2.6 %) and g1_fall (n_dofs=35, T=16 +2.9 %). TileCls is
    passed as a qd.template() so the value is part of the kernel's compile-time signature (selecting a Tile type as a
    local via ternary fails type inference); the func_cholesky_factor_direct_tiled wrapper guarantees TileCls matches T.

    Beware the Hessian matrix is re-purposed to store its Cholesky factorization to spare memory resources.

    Note that only the lower triangular part will be updated for efficiency, because the Hessian matrix is symmetric.
    When n_dofs is not a multiple of T, partial tiles are padded with identity (diagonal=1, off-diagonal=0) so the
    factorization is correct for the original n_dofs x n_dofs submatrix. Tile slice ops handle the per-thread bounds
    internally, so no `if tid < ...` guards are needed at the call site.
    """
    T = qd.static(static_rigid_sim_config.cholesky_tile_size)

    EPS = rigid_global_info.EPS[None]

    _B = constraint_state.grad.shape[1]
    n_dofs = constraint_state.nt_H.shape[1]
    N_BLOCKS = (n_dofs + T - 1) // T

    qd.loop_config(name="cholesky_factor_direct_tiled", block_dim=T)
    for i in range(_B * T):
        i_b = i // T
        if i_b >= _B:
            continue
        if constraint_state.n_constraints[i_b] == 0 or not constraint_state.improved[i_b]:
            continue

        # Loop over column blocks sequentially: each column block depends on all prior columns (inherent to
        # left-looking Cholesky). Within each column, the diagonal is factored first, then off-diagonal rows
        # are processed sequentially (they only depend on the diagonal, but each tile uses all threads).
        for kb in range(N_BLOCKS):
            k0 = kb * T
            k1 = qd.min(k0 + T, n_dofs)

            # Load diagonal tile H[k,k] (rows beyond n_dofs stay as identity from the .eye() init)
            L_kk = TileCls.eye(dtype=gs.qd_float)
            L_kk[:] = constraint_state.nt_H[i_b, k0:k1, k0:k1]

            # Subtract prior-column contributions: L_kk -= sum_j L[k,j] @ L[k,j]^T
            for jb in range(kb):
                j0 = jb * T
                for t in range(T):
                    v = constraint_state.nt_H[i_b, k0:k1, j0 + t]
                    L_kk -= qd.outer(v, v)

            # Factor diagonal tile in-place
            L_kk.cholesky_(EPS)

            # Solve off-diagonal tiles: L[i,k] = (H[i,k] - sum_j L[i,j] L[k,j]^T) @ inv(L[k,k]^T)
            for ib in range(kb + 1, N_BLOCKS):
                i0 = ib * T
                i1 = qd.min(i0 + T, n_dofs)

                # Load off-diagonal tile H[i,k] (rows beyond n_dofs stay as zero from the .zeros() init)
                L_ik = TileCls.zeros(dtype=gs.qd_float)
                L_ik[:] = constraint_state.nt_H[i_b, i0:i1, k0:k1]

                # Subtract prior-column contributions L[i,j] @ L[k,j]^T
                for jb in range(kb):
                    j0 = jb * T
                    for t in range(T):
                        v_own = constraint_state.nt_H[i_b, i0:i1, j0 + t]
                        v_diag = constraint_state.nt_H[i_b, k0:k1, j0 + t]
                        L_ik -= qd.outer(v_own, v_diag)

                # Triangular solve: L[i,k] = L_ik @ inv(L[k,k]^T)
                L_kk.solve_triangular_(L_ik)

                # Write L[i,k] back to global memory
                constraint_state.nt_H[i_b, i0:i1, k0:k1] = L_ik

            # Write L[k,k] back to global memory
            constraint_state.nt_H[i_b, k0:k1, k0:k1] = L_kk


@qd.func
def _cholesky_and_solve_fused_tiled_impl(
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    TileCls: qd.template(),
    write_L_to_nt_H: qd.template() = False,
):
    """Fused Cholesky factorization and triangular solve, keeping L in shared memory.

    Factorizes H = L L^T using register-resident TxT tiles, storing completed L tiles in shared memory. Then solves
    L L^T x = g (forward + backward substitution) in-place and writes the result to Mgrad, without ever writing L to
    global memory.

    Tile size T and TileCls are dispatched by the func_cholesky_and_solve_fused_tiled wrapper; see
    _cholesky_factor_direct_tiled_impl for the rule.

    When ``write_L_to_nt_H`` is True, L is also written back to ``constraint_state.nt_H`` at the end of the kernel.
    This is required by the warm-start dispatch (``enable_fused_factor_solve_init``) so the monolith body's incremental
    rank-1 Cholesky update finds L (not H) in nt_H.
    """
    T = qd.static(static_rigid_sim_config.cholesky_tile_size)
    LOG2_T = qd.static(T.bit_length() - 1)

    EPS = rigid_global_info.EPS[None]
    MAX_DOFS = qd.static(static_rigid_sim_config.tiled_n_dofs)

    _B = constraint_state.grad.shape[1]
    n_dofs = constraint_state.nt_H.shape[1]
    N_BLOCKS = (n_dofs + T - 1) // T

    qd.loop_config(name="cholesky_and_solve_fused_tiled", block_dim=T)
    for i in range(_B * T):
        tid = i % T
        i_b = i // T
        if i_b >= _B:
            continue
        if constraint_state.n_constraints[i_b] == 0 or not constraint_state.improved[i_b]:
            continue

        # +1 padding avoids shared memory bank conflicts on column-wise access (backward substitution, factorization)
        L_sh = qd.simt.block.SharedArray((MAX_DOFS, MAX_DOFS + 1), gs.qd_float)
        v_sh = qd.simt.block.SharedArray((MAX_DOFS,), gs.qd_float)

        # --- Blocked Cholesky factorization (same algorithm as func_cholesky_factor_direct_tiled) ---
        # Loop over column blocks sequentially: each column block depends on all prior columns (inherent to
        # left-looking Cholesky). Within each column, the diagonal is factored first, then off-diagonal rows
        # are processed sequentially (they only depend on the diagonal, but each tile uses all threads).
        for kb in range(N_BLOCKS):
            k0 = kb * T
            k1 = qd.min(k0 + T, n_dofs)

            # Load diagonal tile H[k,k] (rows beyond n_dofs stay as identity from the .eye() init)
            L_kk = TileCls.eye(dtype=gs.qd_float)
            L_kk[:] = constraint_state.nt_H[i_b, k0:k1, k0:k1]

            # Subtract prior-column contributions from shared memory
            for jb in range(kb):
                j0 = jb * T
                for t in range(T):
                    v = L_sh[k0:k1, j0 + t]
                    L_kk -= qd.outer(v, v)

            # Factor diagonal tile in-place
            L_kk.cholesky_(EPS)

            # Solve off-diagonal tiles and store in shared memory (not global)
            for ib in range(kb + 1, N_BLOCKS):
                i0 = ib * T
                i1 = qd.min(i0 + T, n_dofs)

                # Load off-diagonal tile H[i,k] (rows beyond n_dofs stay as zero from the .zeros() init)
                L_ik = TileCls.zeros(dtype=gs.qd_float)
                L_ik[:] = constraint_state.nt_H[i_b, i0:i1, k0:k1]

                # Subtract prior-column contributions from shared memory
                for jb in range(kb):
                    j0 = jb * T
                    for t in range(T):
                        v_own = L_sh[i0:i1, j0 + t]
                        v_diag = L_sh[k0:k1, j0 + t]
                        L_ik -= qd.outer(v_own, v_diag)

                # Triangular solve: L[i,k] = L_ik @ inv(L[k,k]^T)
                L_kk.solve_triangular_(L_ik)

                # Write L[i,k] to shared memory
                L_sh[i0:i1, k0:k1] = L_ik

            # Write L[k,k] to shared memory
            L_sh[k0:k1, k0:k1] = L_kk

        # --- Scalar triangular solve using L from shared memory ---
        # No longer using TxT tiles; the T threads parallelize each row's dot product by striping across columns,
        # then subgroup-reduce to sum the partial products. Thread 0 writes each solved element.

        # Load gradient into v_sh
        k = tid
        while k < n_dofs:
            v_sh[k] = constraint_state.grad[k, i_b]
            k = k + T
        qd.simt.block.sync()

        # Forward substitution: solve L @ y = grad (parallel dot with T threads)
        for i_d in range(n_dofs):
            dot = gs.qd_float(0.0)
            j = tid
            while j < i_d:
                dot = dot + L_sh[i_d, j] * v_sh[j]
                j = j + T
            dot = qd.simt.subgroup.reduce_all_add_tiled(dot, LOG2_T)
            if tid == 0:
                v_sh[i_d] = (v_sh[i_d] - dot) / L_sh[i_d, i_d]
            qd.simt.block.sync()

        # Backward substitution: solve L^T @ x = y (parallel dot with T threads)
        for i_d_ in range(n_dofs):
            i_d = n_dofs - 1 - i_d_
            dot = gs.qd_float(0.0)
            j = i_d + 1 + tid
            while j < n_dofs:
                dot = dot + L_sh[j, i_d] * v_sh[j]
                j = j + T
            dot = qd.simt.subgroup.reduce_all_add_tiled(dot, LOG2_T)
            if tid == 0:
                v_sh[i_d] = (v_sh[i_d] - dot) / L_sh[i_d, i_d]
            qd.simt.block.sync()

        # Write Mgrad to global memory
        k = tid
        while k < n_dofs:
            constraint_state.Mgrad[k, i_b] = v_sh[k]
            k = k + T

        # When dispatched from the warm-start in func_solve_init, the monolith body's first iter expects nt_H to hold L
        # (it runs an incremental rank-1 Cholesky update on it). The fused kernel keeps L only in shmem, so restore the
        # post-condition with a tid-strided writeback over the full n_dofs * n_dofs grid. The wasted upper-triangle
        # writes are harmless (no nt_H reader touches them) and avoid a per-element predicate that would idle half the
        # warp on small rows.
        if qd.static(write_L_to_nt_H):
            i_flat = tid
            n_dofs_sq = n_dofs * n_dofs
            while i_flat < n_dofs_sq:
                i_d1 = i_flat // n_dofs
                i_d2 = i_flat % n_dofs
                constraint_state.nt_H[i_b, i_d1, i_d2] = L_sh[i_d1, i_d2]
                i_flat = i_flat + T


@qd.func
def func_cholesky_factor_direct_tiled(
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Tile-size dispatcher; see _cholesky_factor_direct_tiled_impl for the algorithm and dispatch rule."""
    _cholesky_factor_direct_tiled_impl(
        constraint_state,
        rigid_global_info,
        static_rigid_sim_config,
        qd.simt.Tile32x32 if qd.static(static_rigid_sim_config.cholesky_tile_size == 32) else qd.simt.Tile16x16,
    )


@qd.func
def func_cholesky_and_solve_fused_tiled(
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    write_L_to_nt_H: qd.template() = False,
):
    """Tile-size dispatcher; see _cholesky_and_solve_fused_tiled_impl for the algorithm and dispatch rule."""
    _cholesky_and_solve_fused_tiled_impl(
        constraint_state,
        rigid_global_info,
        static_rigid_sim_config,
        qd.simt.Tile32x32 if qd.static(static_rigid_sim_config.cholesky_tile_size == 32) else qd.simt.Tile16x16,
        write_L_to_nt_H,
    )


@qd.func
def func_hessian_and_cholesky_factor_direct_batch(
    i_b,
    island_state: array_class.IslandState,
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    compute_envelope: qd.template() = False,
):
    # Combined Hessian build + Cholesky factor for one env. With islands the block-diagonal Hessian is assembled and
    # factored per island; otherwise the whole env is the single work-unit i_island = 0. compute_envelope sets each
    # island's structural skyline envelope first (callers do this once per step, then leave it False).
    if qd.static(static_rigid_sim_config.enable_per_island_solve):
        for i_island in range(island_state.n_islands[i_b]):
            if qd.static(static_rigid_sim_config.use_hibernation):
                if island_state.is_hibernated[i_island, i_b]:
                    continue
            if qd.static(compute_envelope):
                func_compute_island_envelope(i_b, i_island, island_state, constraint_state, rigid_global_info)
            func_hessian_direct_batch(
                i_b,
                i_island,
                island_state,
                entities_info,
                constraint_state,
                rigid_global_info,
                static_rigid_sim_config,
            )
            func_cholesky_factor_direct_batch(
                i_b, i_island, island_state, constraint_state, rigid_global_info, static_rigid_sim_config
            )
    else:
        func_hessian_direct_batch(
            i_b, 0, island_state, entities_info, constraint_state, rigid_global_info, static_rigid_sim_config
        )
        func_cholesky_factor_direct_batch(
            i_b, 0, island_state, constraint_state, rigid_global_info, static_rigid_sim_config
        )


@qd.func
def func_hessian_and_cholesky_factor_direct(
    island_state: array_class.IslandState,
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    compute_envelope: qd.template() = False,
):
    """
    Unified implementation of Hessian matrix computation with Cholesky factorization optimized for both CPU and GPU
    backends.

    With contact islands the Hessian is block-diagonal, so each island's block is assembled and factored independently,
    parallelized over the flat (env, island) grid. Otherwise the whole env is one work-unit, factored by the GPU tiled
    path or the CPU batched path (the sparse skyline-envelope factor is CPU-only and runs through the batched path).

    compute_envelope computes each island's structural skyline envelope before factoring; it is structural, so callers
    set it once per step (func_solve_init) and leave it False for the per-iteration re-factorizations.
    """
    _B = constraint_state.jac.shape[2]

    if qd.static(static_rigid_sim_config.enable_per_island_solve):
        # The block-diagonal Hessian factors per island; spread the islands across the (env, island) grid so they run
        # concurrently rather than serially within each env. max_islands bounds the per-env island count (at most one
        # island per link); the guard skips the unused tail.
        max_islands = island_state.dof_slices.start.shape[0]
        qd.loop_config(
            name="hess_cholesky_factor_direct_island",
            serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL),
            block_dim=32,
        )
        for i_b, i_island in qd.ndrange(_B, max_islands):
            if i_island < island_state.n_islands[i_b]:
                if qd.static(static_rigid_sim_config.use_hibernation):
                    if island_state.is_hibernated[i_island, i_b]:
                        continue
                if qd.static(compute_envelope):
                    func_compute_island_envelope(i_b, i_island, island_state, constraint_state, rigid_global_info)
                func_hessian_direct_batch(
                    i_b,
                    i_island,
                    island_state,
                    entities_info,
                    constraint_state,
                    rigid_global_info,
                    static_rigid_sim_config,
                )
                func_cholesky_factor_direct_batch(
                    i_b, i_island, island_state, constraint_state, rigid_global_info, static_rigid_sim_config
                )
    elif qd.static(static_rigid_sim_config.backend == gs.cpu):
        # CPU
        qd.loop_config(
            name="hess_cholesky_factor_direct",
            serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL,
            block_dim=32,
        )
        for i_b in range(_B):
            func_hessian_and_cholesky_factor_direct_batch(
                i_b,
                island_state=island_state,
                entities_info=entities_info,
                rigid_global_info=rigid_global_info,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )
    else:
        # GPU
        func_hessian_direct_tiled(constraint_state, rigid_global_info)

        if qd.static(static_rigid_sim_config.enable_tiled_cholesky_hessian):
            # The register-streaming tiled factor has no shared-memory DOF cap, so it replaces the scalar one-thread-
            # per-env Cholesky (O(n_dofs^3) serial) for any n_dofs >= 16. Above the shared cap (hessian_fits_shared is
            # False) the triangular solve falls back to the scalar batch path and the per-iteration incremental rank-1
            # update stays scalar, both reading L back from nt_H. When the fused warm-start dispatch is on, the factor
            # is folded into the fused kernel (called from func_update_gradient_tiled below), so the standalone factor
            # is skipped to avoid doing it twice.
            if qd.static(not static_rigid_sim_config.enable_fused_factor_solve_init):
                func_cholesky_factor_direct_tiled(constraint_state, rigid_global_info, static_rigid_sim_config)
        else:
            qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
            for i_b in range(_B):
                func_cholesky_factor_direct_batch(
                    i_b, 0, island_state, constraint_state, rigid_global_info, static_rigid_sim_config
                )


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

        for i_d_ in range(constraint_state.jac_n_dofs[i_c, i_b]):
            i_d = constraint_state.jac_dofs_idx[i_c, i_d_, i_b]
            constraint_state.nt_vec[i_d, i_b] = constraint_state.jac[i_c, i_d, i_b] * efc_D_sqrt

        for k_ in range(constraint_state.jac_n_dofs[i_c, i_b]):
            k = constraint_state.jac_dofs_idx[i_c, k_, i_b]
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
                i = constraint_state.jac_dofs_idx[i_c, i_, i_b]  # i is strictly > k
                constraint_state.nt_H[i_b, i, k] = (
                    constraint_state.nt_H[i_b, i, k] + s * constraint_state.nt_vec[i, i_b] * sign
                ) * cinv

            for i_ in range(k_):
                i = constraint_state.jac_dofs_idx[i_c, i_, i_b]  # i is strictly > k
                constraint_state.nt_vec[i, i_b] = (
                    constraint_state.nt_vec[i, i_b] * c - s * constraint_state.nt_H[i_b, i, k]
                )

    return is_degenerated


@qd.func
def func_hessian_and_cholesky_factor_incremental_batch(
    i_b,
    island_state: array_class.IslandState,
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
) -> bool:
    # Per-island full rebuild only when there are MULTIPLE islands (the incremental rank-1 update assumes a single
    # dense system, which a block-diagonal multi-island Hessian is not) or when hibernation is active (the rebuild
    # skips asleep islands; the incremental path would move their DOFs). A SINGLE awake island spans the whole env -
    # it IS a single dense system - so it uses the same incremental update as islands OFF: identical work, no
    # per-iteration full rebuild. This is what makes monolith island-ON match island-OFF for one island.
    do_full_rebuild = False
    if qd.static(static_rigid_sim_config.enable_per_island_solve):
        do_full_rebuild = island_state.n_islands[i_b] > 1
        if qd.static(static_rigid_sim_config.use_hibernation):
            do_full_rebuild = True
    is_degenerated = False
    if do_full_rebuild:
        func_hessian_and_cholesky_factor_direct_batch(
            i_b, island_state, entities_info, constraint_state, rigid_global_info, static_rigid_sim_config
        )
    else:
        func_build_changed_constraint_list(i_b, constraint_state=constraint_state)
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
    i_island,
    island_state: array_class.IslandState,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    # Solve L @ L.T @ Mgrad = grad for one work-unit. With islands, the unit is island i_island: its factored
    # L is the packed n x n tile at tile_start, and grad/Mgrad stay global-indexed via the block-gather
    # dof_id (local row ld -> global dof). The global Hessian is block-diagonal by island, so each island's
    # solve is independent and equals the single dense solve. Without islands, the unit is the whole env and
    # i_island is unused: L is in nt_H (in-place factorization), and with the skyline envelope the triangular
    # solves visit only the envelope of L (its nonzeros match the factorization's). Gated on enable_per_island_solve
    # (not use_contact_island): the per-island solve reads the per-island skyline envelope, which is only built by
    # the per-island factor - when the factor ran whole-env (enable_per_island_solve False) the solve must too.
    if qd.static(static_rigid_sim_config.enable_per_island_solve):
        n = island_state.dof_slices.n[i_island, i_b]
        dof_base = island_state.dof_slices.start[i_island, i_b]
        # L is stored in the island's block of nt_H at its global DOF rows/cols (dof_id ascending -> lower
        # triangle). grad/Mgrad stay global-indexed; the global Hessian is block-diagonal so this island solve
        # is independent and equals the single dense solve.
        # Forward then backward substitution, confined to L's skyline envelope (matching the factorization).
        for ld in range(n):
            gd = island_state.dof_id[dof_base + ld, i_b]
            curr_out = constraint_state.grad[gd, i_b]
            for j_d in range(island_state.dof_env_start_local[dof_base + ld, i_b], ld):
                g_jd = island_state.dof_id[dof_base + j_d, i_b]
                curr_out = curr_out - constraint_state.nt_H[i_b, gd, g_jd] * constraint_state.Mgrad[g_jd, i_b]
            constraint_state.Mgrad[gd, i_b] = curr_out / constraint_state.nt_H[i_b, gd, gd]
        for ld_ in range(n):
            ld = n - 1 - ld_
            gd = island_state.dof_id[dof_base + ld, i_b]
            curr_out = constraint_state.Mgrad[gd, i_b]
            for j_d in range(ld + 1, n):
                if island_state.dof_env_start_local[dof_base + j_d, i_b] <= ld:
                    g_jd = island_state.dof_id[dof_base + j_d, i_b]
                    curr_out = curr_out - constraint_state.nt_H[i_b, g_jd, gd] * constraint_state.Mgrad[g_jd, i_b]
            constraint_state.Mgrad[gd, i_b] = curr_out / constraint_state.nt_H[i_b, gd, gd]
    elif qd.static(static_rigid_sim_config.sparse_envelope):
        n_dofs = constraint_state.Mgrad.shape[0]
        # i_d / j_d index permuted positions; grad/Mgrad are stored in natural DOF order, so map through dof_perm
        # (identity when reordering is off).
        for i_d in range(n_dofs):
            d_i = constraint_state.dof_perm[i_b, i_d]
            curr_out = constraint_state.grad[d_i, i_b]
            for j_d in range(constraint_state.nt_H_env_start[i_b, i_d], i_d):
                d_j = constraint_state.dof_perm[i_b, j_d]
                curr_out = curr_out - constraint_state.nt_H[i_b, i_d, j_d] * constraint_state.Mgrad[d_j, i_b]
            constraint_state.Mgrad[d_i, i_b] = curr_out / constraint_state.nt_H[i_b, i_d, i_d]

        for i_d_ in range(n_dofs):
            i_d = n_dofs - 1 - i_d_
            d_i = constraint_state.dof_perm[i_b, i_d]
            curr_out = constraint_state.Mgrad[d_i, i_b]
            for j_d in range(i_d + 1, n_dofs):
                if constraint_state.nt_H_env_start[i_b, j_d] <= i_d:
                    d_j = constraint_state.dof_perm[i_b, j_d]
                    curr_out = curr_out - constraint_state.nt_H[i_b, j_d, i_d] * constraint_state.Mgrad[d_j, i_b]
            constraint_state.Mgrad[d_i, i_b] = curr_out / constraint_state.nt_H[i_b, i_d, i_d]
    else:
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

    qd.loop_config(name="cholesky_solve_tiled", block_dim=BLOCK_DIM)
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

        # Copy the lower triangular part of L (Cholesky factor) to shared memory for efficiency
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
def func_ls_init_and_eval_p0(
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
    # mv = M @ search. Mass couples only DOFs within the same kinematic-tree block, so restrict the inner loop to
    # i_d1's block (cross-block entries are zero). For one entity holding many free bodies this is the difference
    # between O(entity_dofs^2) and the sum of per-tree blocks.
    for i_e in range(n_entities):
        for i_d1 in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
            mv = gs.qd_float(0.0)
            for i_d2 in range(
                rigid_global_info.dofs_mass_block_start[i_d1], rigid_global_info.dofs_mass_block_end[i_d1]
            ):
                mv = mv + rigid_global_info.mass_mat[i_d1, i_d2, i_b] * constraint_state.search[i_d2, i_b]
            constraint_state.mv[i_d1, i_b] = mv

    for i_c in range(n_con):
        jv = gs.qd_float(0.0)
        if qd.static(static_rigid_sim_config.sparse_solve or static_rigid_sim_config.enable_per_island_solve):
            for i_d_ in range(constraint_state.jac_n_dofs[i_c, i_b]):
                i_d = constraint_state.jac_dofs_idx[i_c, i_d_, i_b]
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
def _func_linesearch_eval_constraints_at_n_alphas_serial(
    i_b,
    alphas,
    constraint_state: array_class.ConstraintState,
    n_alphas: qd.template(),
):
    """Reduce the quadratic-coefficient triplets (const, linear, quad) for up to ``n_alphas`` candidate alphas (passed
    as a ``qd.Vector(3)`` ``alphas``; only the first ``n_alphas`` slots are read) in a single pass over all friction +
    contact constraints. Returns 3 ``qd.Vector(3)``s ``(t0, t1, t2)`` where ``tk`` is alpha-slot ``k``'s
    ``[const, linear, quad]``. Slots beyond ``n_alphas`` hold the equality-only seed and should be ignored by the
    caller.

    Equality constraints are skipped via ``quad_gauss + eq_sum`` (pre-computed during init). Quad coefficients are
    recomputed on the fly from Jaref, jv, efc_D rather than read from a precomputed quad array, costing 3 loads per
    contact (vs 5) and 5 per friction (vs 7), a 40%/29% bandwidth reduction. The ~8 FLOPs of recomputation per
    constraint are almost free. With ``n_alphas == 3``, each constraint's loaded data is reused for all 3 alpha
    evaluations.
    """
    ne = constraint_state.n_constraints_equality[i_b]
    nef = ne + constraint_state.n_constraints_frictionloss[i_b]
    n_con = constraint_state.n_constraints[i_b]

    # Start from quad_gauss + eq_sum (skips ne equality constraints)
    base_0 = constraint_state.quad_gauss[0, i_b] + constraint_state.eq_sum[0, i_b]
    base_1 = constraint_state.quad_gauss[1, i_b] + constraint_state.eq_sum[1, i_b]
    base_2 = constraint_state.quad_gauss[2, i_b] + constraint_state.eq_sum[2, i_b]

    t_0 = [base_0, base_0, base_0]
    t_1 = [base_1, base_1, base_1]
    t_2 = [base_2, base_2, base_2]

    # Friction constraints [ne, nef): 5 loads (Jaref, jv, D, f, diag) + recompute quad, eval n_alphas
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
        for k in qd.static(range(n_alphas)):
            alpha_k = alphas[k]
            x = Jaref_c + alpha_k * jv_c
            ln = x <= -rf
            lp = x >= rf
            ak_qf_0, ak_qf_1, ak_qf_2 = qf_0, qf_1, qf_2
            if ln or lp:
                ak_qf_0 = ln * f * (-0.5 * rf - Jaref_c) + lp * f * (-0.5 * rf + Jaref_c)
                ak_qf_1 = ln * (-f * jv_c) + lp * (f * jv_c)
                ak_qf_2 = 0.0
            t_0[k] = t_0[k] + ak_qf_0
            t_1[k] = t_1[k] + ak_qf_1
            t_2[k] = t_2[k] + ak_qf_2

    # Contact constraints [nef, n_con): 3 loads (Jaref, jv, D) + recompute quad, eval n_alphas
    for i_c in range(nef, n_con):
        Jaref_c = constraint_state.Jaref[i_c, i_b]
        jv_c = constraint_state.jv[i_c, i_b]
        D = constraint_state.efc_D[i_c, i_b]
        qf_0 = D * (0.5 * Jaref_c * Jaref_c)
        qf_1 = D * (jv_c * Jaref_c)
        qf_2 = D * (0.5 * jv_c * jv_c)
        for k in qd.static(range(n_alphas)):
            alpha_k = alphas[k]
            x = Jaref_c + alpha_k * jv_c
            act = gs.qd_bool(x < 0)
            t_0[k] = t_0[k] + qf_0 * act
            t_1[k] = t_1[k] + qf_1 * act
            t_2[k] = t_2[k] + qf_2 * act

    t0 = qd.Vector([t_0[0], t_1[0], t_2[0]])
    t1 = qd.Vector([t_0[1], t_1[1], t_2[1]])
    t2 = qd.Vector([t_0[2], t_1[2], t_2[2]])
    return t0, t1, t2


@qd.func
def _func_linesearch_eval_quadratic_at_alpha(
    i_b,
    tid,
    alpha,
    t,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    coop: qd.template(),
):
    """Given the reduced quadratic-coefficient triple ``t`` (a ``qd.Vector(3)`` packed as ``[const, linear, quad]``),
    plug ``alpha`` into ``cost(alpha) = c + l*alpha + q*alpha**2`` and its first/second derivatives, and return
    ``(alpha, cost, grad, hess)``. The hessian is floored at ``EPS`` so downstream Newton steps stay finite. Increments
    ``ls_it`` by 1; under ``coop=True`` the increment is gated to a single thread because lanes share the same per-env
    counter."""
    cost = alpha * alpha * t[2] + alpha * t[1] + t[0]
    grad = 2 * alpha * t[2] + t[1]
    hess = 2 * t[2]
    if hess <= 0.0:
        hess = rigid_global_info.EPS[None]

    if qd.static(not coop) or tid == 0:
        constraint_state.ls_it[i_b] = constraint_state.ls_it[i_b] + 1

    return alpha, cost, grad, hess


@qd.func
def _func_linesearch_eval_at_alpha(
    i_b,
    tid,
    alpha,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    coop: qd.template(),
):
    """Single-alpha linesearch evaluator. ``coop=True`` runs cooperatively across the 32-lane warp (caller passes the
    lane id as ``tid``); ``coop=False`` runs serially and the caller is responsible for ensuring only one thread per
    env enters this function (typically by gating on ``tid == 0`` upstream).

    Note: the reducer call and the post-reduction call live inside the same ``qd.static(coop)`` branch and end with
    ``return``, because Quadrants' AST transformer doesn't propagate locals across ``if qd.static`` branches; naming
    a variable in the unified ``return`` statement raises ``Name "t0" is not defined`` even when one branch is
    DCE'd. Self-contained per-branch returns sidestep this."""
    alphas = qd.Vector([alpha, alpha, alpha])
    if qd.static(coop):
        t0, _u1, _u2 = _func_linesearch_eval_constraints_at_n_alphas_coop(
            i_b, tid, alphas, constraint_state, n_alphas=1
        )
        return _func_linesearch_eval_quadratic_at_alpha(
            i_b, tid, alpha, t0, constraint_state, rigid_global_info, coop=True
        )
    else:
        t0, _u1, _u2 = _func_linesearch_eval_constraints_at_n_alphas_serial(i_b, alphas, constraint_state, n_alphas=1)
        return _func_linesearch_eval_quadratic_at_alpha(
            i_b, tid, alpha, t0, constraint_state, rigid_global_info, coop=False
        )


@qd.func
def _func_linesearch_eval_constraints_at_n_alphas_coop(
    i_b,
    tid,
    alphas,
    constraint_state: array_class.ConstraintState,
    n_alphas: qd.template(),
):
    """Cooperative (32-lane subgroup) variant of ``_func_linesearch_eval_constraints_at_n_alphas_serial``.

    All 32 lanes call this with their own ``tid``; the constraint loop is strided by 32, then each
    accumulator is reduced across the warp via ``subgroup.reduce_all_add_tiled(_, 5)`` so every lane ends
    up with identical return values. Returns the same 3 ``qd.Vector(3)``s ``(t0, t1, t2)`` as the serial inner.
    """
    ne = constraint_state.n_constraints_equality[i_b]
    nef = ne + constraint_state.n_constraints_frictionloss[i_b]
    n_con = constraint_state.n_constraints[i_b]

    # Start from quad_gauss + eq_sum (skips ne equality constraints); only lane 0 holds the seed,
    # the warp-tree reduction at the end implicitly broadcasts it back to all lanes.
    base_0 = gs.qd_float(0.0)
    base_1 = gs.qd_float(0.0)
    base_2 = gs.qd_float(0.0)
    if tid == 0:
        base_0 = constraint_state.quad_gauss[0, i_b] + constraint_state.eq_sum[0, i_b]
        base_1 = constraint_state.quad_gauss[1, i_b] + constraint_state.eq_sum[1, i_b]
        base_2 = constraint_state.quad_gauss[2, i_b] + constraint_state.eq_sum[2, i_b]

    t_0 = [base_0, base_0, base_0]
    t_1 = [base_1, base_1, base_1]
    t_2 = [base_2, base_2, base_2]

    # Friction constraints [ne, nef): 5 loads (Jaref, jv, D, f, diag) + recompute quad, eval n_alphas;
    # constraint loop strided by 32 across the warp.
    i_c = ne + tid
    while i_c < nef:
        Jaref_c = constraint_state.Jaref[i_c, i_b]
        jv_c = constraint_state.jv[i_c, i_b]
        D = constraint_state.efc_D[i_c, i_b]
        f = constraint_state.efc_frictionloss[i_c, i_b]
        r = constraint_state.diag[i_c, i_b]
        qf_0 = D * (0.5 * Jaref_c * Jaref_c)
        qf_1 = D * (jv_c * Jaref_c)
        qf_2 = D * (0.5 * jv_c * jv_c)
        rf = r * f
        for k in qd.static(range(n_alphas)):
            alpha_k = alphas[k]
            x = Jaref_c + alpha_k * jv_c
            ln = x <= -rf
            lp = x >= rf
            ak_qf_0, ak_qf_1, ak_qf_2 = qf_0, qf_1, qf_2
            if ln or lp:
                ak_qf_0 = ln * f * (-0.5 * rf - Jaref_c) + lp * f * (-0.5 * rf + Jaref_c)
                ak_qf_1 = ln * (-f * jv_c) + lp * (f * jv_c)
                ak_qf_2 = 0.0
            t_0[k] = t_0[k] + ak_qf_0
            t_1[k] = t_1[k] + ak_qf_1
            t_2[k] = t_2[k] + ak_qf_2
        i_c = i_c + 32

    # Contact constraints [nef, n_con): 3 loads (Jaref, jv, D) + recompute quad, eval n_alphas;
    # constraint loop strided by 32 across the warp.
    i_c = nef + tid
    while i_c < n_con:
        Jaref_c = constraint_state.Jaref[i_c, i_b]
        jv_c = constraint_state.jv[i_c, i_b]
        D = constraint_state.efc_D[i_c, i_b]
        qf_0 = D * (0.5 * Jaref_c * Jaref_c)
        qf_1 = D * (jv_c * Jaref_c)
        qf_2 = D * (0.5 * jv_c * jv_c)
        for k in qd.static(range(n_alphas)):
            alpha_k = alphas[k]
            x = Jaref_c + alpha_k * jv_c
            act = gs.qd_bool(x < 0)
            t_0[k] = t_0[k] + qf_0 * act
            t_1[k] = t_1[k] + qf_1 * act
            t_2[k] = t_2[k] + qf_2 * act
        i_c = i_c + 32

    # Warp-tree reduction: every lane's 9 partial sums collapse into the per-env totals; after this
    # all 32 lanes hold identical scalars. The `5` is log2(32) tree levels.
    for k in qd.static(range(n_alphas)):
        t_0[k] = qd.simt.subgroup.reduce_all_add_tiled(t_0[k], 5)
        t_1[k] = qd.simt.subgroup.reduce_all_add_tiled(t_1[k], 5)
        t_2[k] = qd.simt.subgroup.reduce_all_add_tiled(t_2[k], 5)

    t0 = qd.Vector([t_0[0], t_1[0], t_2[0]])
    t1 = qd.Vector([t_0[1], t_1[1], t_2[1]])
    t2 = qd.Vector([t_0[2], t_1[2], t_2[2]])
    return t0, t1, t2


@qd.func
def _func_linesearch_eval_quadratic_at_3_alphas(
    i_b,
    tid,
    alphas,
    t0,
    t1,
    t2,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    coop: qd.template(),
):
    """Given three reduced quadratic-coefficient triples (one per candidate alpha; ``t0``, ``t1``, ``t2`` are each a
    ``qd.Vector(3)`` packed as ``[const, linear, quad]``) and a ``qd.Vector(3)`` of candidate ``alphas``, plug each
    alpha into ``cost(alpha) = c + l*alpha + q*alpha**2`` and its first/second derivatives. Returns three
    ``qd.Vector(3)``s ``(costs, grads, hess)`` indexed by alpha slot. The hessian is floored at ``EPS`` so downstream
    Newton steps stay finite. Increments ``ls_it`` by 3 (one per evaluated alpha); the increment is gated to a single
    thread under ``coop=True`` since lanes share the same per-env counter."""
    EPS = rigid_global_info.EPS[None]

    cost_0 = alphas[0] * alphas[0] * t0[2] + alphas[0] * t0[1] + t0[0]
    grad_0 = 2 * alphas[0] * t0[2] + t0[1]
    hess_0 = 2 * t0[2]
    if hess_0 <= 0.0:
        hess_0 = EPS

    cost_1 = alphas[1] * alphas[1] * t1[2] + alphas[1] * t1[1] + t1[0]
    grad_1 = 2 * alphas[1] * t1[2] + t1[1]
    hess_1 = 2 * t1[2]
    if hess_1 <= 0.0:
        hess_1 = EPS

    cost_2 = alphas[2] * alphas[2] * t2[2] + alphas[2] * t2[1] + t2[0]
    grad_2 = 2 * alphas[2] * t2[2] + t2[1]
    hess_2 = 2 * t2[2]
    if hess_2 <= 0.0:
        hess_2 = EPS

    if qd.static(not coop) or tid == 0:
        constraint_state.ls_it[i_b] = constraint_state.ls_it[i_b] + 3

    costs = qd.Vector([cost_0, cost_1, cost_2])
    grads = qd.Vector([grad_0, grad_1, grad_2])
    hess = qd.Vector([hess_0, hess_1, hess_2])
    return costs, grads, hess


@qd.func
def _func_linesearch_eval_at_3_alphas(
    i_b,
    tid,
    alphas,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    coop: qd.template(),
):
    """Evaluate linesearch cost, gradient, and curvature at three candidate alphas in a single constraint-loop pass.
    Batches the three step sizes into one loop over constraints so each constraint's heavy work (load Jaref/jv/efc_D
    plus, for friction, efc_frictionloss/diag; recompute the per-constraint quad coefficients) is paid once and reused
    for all three alpha evaluations. Combined with the on-the-fly quad recompute (3 loads/contact, 5 loads/friction;
    same bandwidth optimisation as the 1-alpha evaluator) this means each constraint's data is loaded once from global
    memory and feeds three (cost, grad, hess) results. ``alphas`` is a ``qd.Vector(3)`` of candidate step sizes.

    See ``_func_linesearch_eval_at_alpha`` for the serial-vs-cooperative contract (forwarded via ``coop``) and the
    rationale for the per-branch return."""
    if qd.static(coop):
        t0, t1, t2 = _func_linesearch_eval_constraints_at_n_alphas_coop(i_b, tid, alphas, constraint_state, n_alphas=3)
        return _func_linesearch_eval_quadratic_at_3_alphas(
            i_b, tid, alphas, t0, t1, t2, constraint_state, rigid_global_info, coop=True
        )
    else:
        t0, t1, t2 = _func_linesearch_eval_constraints_at_n_alphas_serial(i_b, alphas, constraint_state, n_alphas=3)
        return _func_linesearch_eval_quadratic_at_3_alphas(
            i_b, tid, alphas, t0, t1, t2, constraint_state, rigid_global_info, coop=False
        )


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
    """Bracket update using local candidate values. No global memory access or _func_linesearch_eval_at_alpha call.

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
def func_linesearch_refine(
    i_b,
    tid,
    p1_alpha,
    p1_cost,
    p1_deriv_0,
    p1_deriv_1,
    p0_cost,
    gtol,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    coop: qd.template(),
):
    """Bracketing walk + 3-alpha dual-bracket refinement.

    Shared by the monolith linesearch (``func_linesearch_batch``) and the decomposed path's Phase 3
    (``solver_breakdown._func_decomp_linesearch_refine``). Takes an initial point (p1_alpha, p1_cost, p1_deriv_0,
    p1_deriv_1) and refines it via Newton steps until the gradient sign flips, then polishes with batched 3-alpha
    evaluation. Returns (res_alpha, ls_result) where ls_result is a status code for diagnostics.

    ``coop=True`` runs cooperatively across the 32-lane warp (caller passes the lane id as ``tid``); ``coop=False`` runs
    serially (1-thread-per-env, caller is responsible for ensuring only ``tid == 0`` enters this function). The inner
    cost evaluators dispatch on the same ``coop`` flag, so ``coop`` is forwarded unchanged.

    The loop predicates use a lane-uniform local ``ls_it_local`` rather than rereading
    ``constraint_state.ls_it[i_b]``: in cooperative mode only ``tid == 0`` writes the global counter from the inner
    evaluators, and there is no warp sync between that gated store and the next-iter read of the global counter, so
    different lanes could otherwise observe different iteration counts and diverge on the predicate (which would
    deadlock the next ``subgroup.reduce_all_add``). We snapshot once at entry, broadcast lane-0's value across the
    warp, and bump locally on each eval call (eval helpers still update the global counter for downstream readers)."""
    res_alpha = gs.qd_float(0.0)
    ls_result = 0
    done = False

    ls_it_local = constraint_state.ls_it[i_b]
    if qd.static(coop):
        ls_it_local = qd.simt.subgroup.broadcast(ls_it_local, qd.u32(0))
    ls_iter_limit = rigid_global_info.ls_iterations[None]

    direction = (p1_deriv_0 < 0) * 2 - 1
    p2update = 0
    p2_alpha = p1_alpha
    p2_cost = p1_cost
    p2_deriv_0 = p1_deriv_0
    p2_deriv_1 = p1_deriv_1
    while p1_deriv_0 * direction <= -gtol and ls_it_local < ls_iter_limit:
        p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1 = p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1
        p2update = 1
        p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = _func_linesearch_eval_at_alpha(
            i_b, tid, p1_alpha - p1_deriv_0 / p1_deriv_1, constraint_state, rigid_global_info, coop=coop
        )
        ls_it_local = ls_it_local + 1
        if qd.abs(p1_deriv_0) < gtol:
            res_alpha = p1_alpha
            done = True
            break
    if not done:
        if ls_it_local >= ls_iter_limit:
            ls_result = 3
            res_alpha = p1_alpha
            done = True
        if not p2update and not done:
            ls_result = 6
            res_alpha = p1_alpha
            done = True
        if not done:
            alpha_0 = p1_alpha - p1_deriv_0 / p1_deriv_1
            alpha_1 = p1_alpha
            alpha_2 = (p1_alpha + p2_alpha) * 0.5
            while ls_it_local < ls_iter_limit:
                alphas = qd.Vector([alpha_0, alpha_1, alpha_2])
                costs, grads, hess = _func_linesearch_eval_at_3_alphas(
                    i_b, tid, alphas, constraint_state, rigid_global_info, coop=coop
                )
                ls_it_local = ls_it_local + 3
                p1_next = alpha_0
                p2_next = alpha_1
                best_a = gs.qd_float(0.0)
                best_c = gs.qd_float(0.0)
                best_found = False
                for i in qd.static(range(3)):
                    if qd.abs(grads[i]) < gtol and (not best_found or costs[i] < best_c):
                        best_a = alphas[i]
                        best_c = costs[i]
                        best_found = True
                if best_found:
                    res_alpha = best_a
                    done = True
                else:
                    b1, p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1, p1_next = update_bracket_no_eval_local(
                        p1_alpha,
                        p1_cost,
                        p1_deriv_0,
                        p1_deriv_1,
                        alphas,
                        costs,
                        grads,
                        hess,
                    )
                    b2, p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1, p2_next = update_bracket_no_eval_local(
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
                            ls_result = 0
                        else:
                            ls_result = 7
                        res_alpha = alpha_2
                        done = True
                if done:
                    break
                alpha_0 = p1_next
                alpha_1 = p2_next
                alpha_2 = (p1_alpha + p2_alpha) * 0.5
            if not done:
                if p1_cost <= p2_cost and p1_cost < p0_cost:
                    ls_result = 4
                    res_alpha = p1_alpha
                elif p2_cost <= p1_cost and p2_cost < p0_cost:
                    ls_result = 4
                    res_alpha = p2_alpha
                else:
                    ls_result = 5
                    res_alpha = 0.0
    return res_alpha, ls_result


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
        p0_alpha, p0_cost, p0_deriv_0, p0_deriv_1 = func_ls_init_and_eval_p0(
            i_b,
            entities_info=entities_info,
            dofs_state=dofs_state,
            constraint_state=constraint_state,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )
        p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = _func_linesearch_eval_at_alpha(
            i_b,
            tid=0,
            alpha=p0_alpha - p0_deriv_0 / p0_deriv_1,
            constraint_state=constraint_state,
            rigid_global_info=rigid_global_info,
            coop=False,
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
            res_alpha, ls_result = func_linesearch_refine(
                i_b,
                tid=0,
                p1_alpha=p1_alpha,
                p1_cost=p1_cost,
                p1_deriv_0=p1_deriv_0,
                p1_deriv_1=p1_deriv_1,
                p0_cost=p0_cost,
                gtol=gtol,
                constraint_state=constraint_state,
                rigid_global_info=rigid_global_info,
                coop=False,
            )
            constraint_state.ls_result[i_b] = ls_result
            # Status 7: both brackets stalled and midpoint cost >= p0_cost. Reject the non-improving alpha.
            if ls_result == 7:
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
    qacc: qd.Tensor,
    Ma: qd.Tensor,
    cost: qd.Tensor,
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

    # qfrc_constraint = J^T @ efc_force. Sparse scatter over each constraint's coupled DOFs (jac_dofs_idx) when that
    # helps (CPU skyline / per-island GPU); islands-OFF GPU gathers per-DOF (bit-identical to the non-island baseline)
    # to keep the 32-env-packed warp's trip count uniform.
    if qd.static(static_rigid_sim_config.sparse_solve or static_rigid_sim_config.enable_per_island_solve):
        for i_d in range(n_dofs):
            constraint_state.qfrc_constraint[i_d, i_b] = gs.qd_float(0.0)
        for i_c in range(constraint_state.n_constraints[i_b]):
            for i_d_ in range(constraint_state.jac_n_dofs[i_c, i_b]):
                i_d = constraint_state.jac_dofs_idx[i_c, i_d_, i_b]
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
def _func_update_efc_force_body(
    i_c,
    i_b,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """Compute active and write efc_force for one (constraint, env) pair.

    Same semantics as the per-constraint loop in ``func_update_constraint_batch`` (lines computing ``active``,
    ``floss_force``, ``efc_force``). Friction cost contribution is *not* accumulated here; it's recomputed in
    ``_func_update_cost_coop`` together with the quadratic term to avoid an extra atomic or shared-memory exchange
    between kernels.
    """
    ne = constraint_state.n_constraints_equality[i_b]
    nef = ne + constraint_state.n_constraints_frictionloss[i_b]

    if qd.static(static_rigid_sim_config.solver_type == gs.constraint_solver.Newton):
        constraint_state.prev_active[i_c, i_b] = constraint_state.active[i_c, i_b]
    constraint_state.active[i_c, i_b] = True

    floss_force = gs.qd_float(0.0)
    if ne <= i_c and i_c < nef:
        f = constraint_state.efc_frictionloss[i_c, i_b]
        r = constraint_state.diag[i_c, i_b]
        rf = r * f
        linear_neg = constraint_state.Jaref[i_c, i_b] <= -rf
        linear_pos = constraint_state.Jaref[i_c, i_b] >= rf
        constraint_state.active[i_c, i_b] = not (linear_neg or linear_pos)
        floss_force = linear_neg * f + linear_pos * -f
    elif nef <= i_c:
        constraint_state.active[i_c, i_b] = constraint_state.Jaref[i_c, i_b] < 0

    constraint_state.efc_force[i_c, i_b] = floss_force + (
        -constraint_state.Jaref[i_c, i_b] * constraint_state.efc_D[i_c, i_b] * constraint_state.active[i_c, i_b]
    )


@qd.func
def _func_update_efc_force(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """Compute active and efc_force for every (constraint, env) with one thread per pair (qd.ndrange-parallel).

    Iteration order picks the coalesced ndrange under each layout: under transposed jac/Jaref/efc_force, lanes vary
    i_c so adjacent reads of the flipped per-constraint tensors stride 1; under canonical, lanes vary i_b.
    """
    len_constraints = constraint_state.active.shape[0]
    _B = constraint_state.grad.shape[1]

    qd.loop_config(
        name="update_constraint_forces", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL
    )
    for i_c, i_b in qd.ndrange(
        len_constraints, _B, axes=qd.static((1, 0) if static_rigid_sim_config.constraint_layout_batch_first else None)
    ):
        if i_c < constraint_state.n_constraints[i_b]:
            _func_update_efc_force_body(i_c, i_b, constraint_state, static_rigid_sim_config)


@qd.func
def _func_update_qfrc_constraint_coop(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """Compute qfrc_constraint = J^T @ efc_force with one cooperating warp per (env, dof).

    32 lanes stride i_c so adjacent reads of jac[i_c, i_d, i_b] and efc_force[i_c, i_b] are stride-1 under the flipped
    jac and flipped efc_force layouts; each (env, dof) warp reduces over its constraints with one warp-reduce.

    Gridding over (env, dof) - one warp per dof rather than one warp per env looping all dofs - keeps the GPU busy when
    the env count alone does not fill it (a single env with many dofs leaves all but one warp idle in the per-env
    layout). The per-lane summation order is unchanged, so the result is bit-identical to the per-env loop.
    """
    n_dofs = constraint_state.qfrc_constraint.shape[0]
    _B = constraint_state.grad.shape[1]
    _K = qd.static(32)

    qd.loop_config(name="update_constraint_qfrc", block_dim=_K)
    for i_flat in range(_B * n_dofs * _K):
        tid = i_flat % _K
        work = i_flat // _K
        i_d = work % n_dofs
        i_b = work // n_dofs
        n_con = constraint_state.n_constraints[i_b]

        qfrc_lane = gs.qd_float(0.0)
        i_c = tid
        while i_c < n_con:
            qfrc_lane = qfrc_lane + constraint_state.jac[i_c, i_d, i_b] * constraint_state.efc_force[i_c, i_b]
            i_c = i_c + _K
        qfrc_total = qd.simt.subgroup.reduce_all_add_tiled(qfrc_lane, 5)
        if tid == 0:
            constraint_state.qfrc_constraint[i_d, i_b] = qfrc_total


@qd.func
def _func_update_cost_coop(
    qacc: qd.template(),
    Ma: qd.template(),
    cost: qd.template(),
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """Compute the linesearch cost (M-norm Gauss + quadratic constraint terms) using one cooperating warp per env.

    Inner loop over dofs (lanes stride i_d): DOF-vec family is canonical (n_dofs, _B) so reads here are *not*
    coalesced under the flipped layout, but the working set is small enough to live in cache. Inner loop over
    constraints (lanes stride i_c): coalesced under flipped Jaref/efc_D/active. One reduce_all_add_tiled per scalar at
    the end.
    """
    _B = constraint_state.grad.shape[1]
    _K = qd.static(32)

    qd.loop_config(name="update_constraint_cost", block_dim=_K)
    for i_flat in range(_B * _K):
        tid = i_flat % _K
        i_b = i_flat // _K
        n_dofs = constraint_state.qfrc_constraint.shape[0]
        ne = constraint_state.n_constraints_equality[i_b]
        nef = ne + constraint_state.n_constraints_frictionloss[i_b]
        n_con = constraint_state.n_constraints[i_b]

        if tid == 0:
            constraint_state.prev_cost[i_b] = cost[i_b]

        cost_i = gs.qd_float(0.0)
        gauss_i = gs.qd_float(0.0)

        i_d = tid
        while i_d < n_dofs:
            v = 0.5 * (Ma[i_d, i_b] - dofs_state.force[i_d, i_b]) * (qacc[i_d, i_b] - dofs_state.acc_smooth[i_d, i_b])
            gauss_i = gauss_i + v
            cost_i = cost_i + v
            i_d = i_d + _K

        i_c = tid
        while i_c < n_con:
            Jaref_c = constraint_state.Jaref[i_c, i_b]
            cost_i = cost_i + 0.5 * (
                Jaref_c * Jaref_c * constraint_state.efc_D[i_c, i_b] * constraint_state.active[i_c, i_b]
            )
            if ne <= i_c and i_c < nef:
                f = constraint_state.efc_frictionloss[i_c, i_b]
                r = constraint_state.diag[i_c, i_b]
                rf = r * f
                linear_neg = Jaref_c <= -rf
                linear_pos = Jaref_c >= rf
                cost_i = cost_i + linear_neg * f * (-0.5 * rf - Jaref_c) + linear_pos * f * (-0.5 * rf + Jaref_c)
            i_c = i_c + _K

        cost_i = qd.simt.subgroup.reduce_all_add_tiled(cost_i, 5)
        gauss_i = qd.simt.subgroup.reduce_all_add_tiled(gauss_i, 5)

        if tid == 0:
            constraint_state.gauss[i_b] = gauss_i
            cost[i_b] = cost_i


@qd.func
def func_update_constraint(
    qacc: qd.Tensor,
    Ma: qd.Tensor,
    cost: qd.Tensor,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """Compute active / efc_force / qfrc_constraint / gauss / cost.

    Under ``enable_cooperative_constraint_kernels=True`` we run three sub-kernels (``_func_update_efc_force``,
    ``_func_update_qfrc_constraint_coop``, ``_func_update_cost_coop``) so per-constraint reads/writes coalesce against
    the flipped jac and Tier-1 constraint-state tensors. Under canonical we keep the original 1-thread-per-env loop
    (bit-identical to the previous code path). The transpose heuristic disables the flip entirely under sparse_solve,
    so sparse runs always take the canonical path here.
    """
    if qd.static(static_rigid_sim_config.enable_cooperative_constraint_kernels):
        _func_update_efc_force(constraint_state, static_rigid_sim_config)
        _func_update_qfrc_constraint_coop(constraint_state, static_rigid_sim_config)
        _func_update_cost_coop(
            qacc=qacc,
            Ma=Ma,
            cost=cost,
            dofs_state=dofs_state,
            constraint_state=constraint_state,
            static_rigid_sim_config=static_rigid_sim_config,
        )
    else:
        _B = constraint_state.jac.shape[2]
        qd.loop_config(name="update_constraint", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
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
    island_state: array_class.IslandState,
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
            None,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
            is_backward=False,
        )

    if qd.static(static_rigid_sim_config.solver_type == gs.constraint_solver.Newton):
        if qd.static(static_rigid_sim_config.enable_per_island_solve):
            # Mgrad = H^{-1} @ grad solved per island on each island's local tile (factored above).
            for i_island in range(island_state.n_islands[i_b]):
                if qd.static(static_rigid_sim_config.use_hibernation):
                    if island_state.is_hibernated[i_island, i_b]:
                        continue
                func_cholesky_solve_batch(i_b, i_island, island_state, constraint_state, static_rigid_sim_config)
        else:
            # Whole-env solve (matching the whole-env factor): the block-diagonal L's per-island blocks are solved
            # together as one dense system; i_island is unused.
            func_cholesky_solve_batch(i_b, 0, island_state, constraint_state, static_rigid_sim_config)


@qd.func
def func_update_gradient_no_solve(
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Compute the gradient only (no Cholesky solve), used with a fused factor+solve that consumes grad directly.

    Under enable_cooperative_constraint_kernels the ndrange is swapped so adjacent lanes vary i_d - 3 of 4 in-loop
    accesses (grad, Ma, qfrc_constraint) are DOF-vec flipped; only dofs_state.force stays canonical.
    """
    _B = constraint_state.grad.shape[1]
    n_dofs = constraint_state.grad.shape[0]
    qd.loop_config(
        name="update_gradient_no_solve", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL
    )
    for i_d, i_b in qd.ndrange(
        n_dofs, _B, axes=qd.static((1, 0) if static_rigid_sim_config.enable_cooperative_constraint_kernels else None)
    ):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            constraint_state.grad[i_d, i_b] = (
                constraint_state.Ma[i_d, i_b] - dofs_state.force[i_d, i_b] - constraint_state.qfrc_constraint[i_d, i_b]
            )


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

    # Compute Mgrad = H^{-1} @ grad, s.t. grad = M @ acc - q_force_ext - q_force_const.
    # Under the DOF-vec flip, 3 of 4 in-loop accesses (grad, Ma, qfrc_constraint) are flipped and one (dofs_state.force)
    # is canonical — swap the ndrange so adjacent lanes vary i_d.
    qd.loop_config(name="update_gradient_tiled", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_d, i_b in qd.ndrange(
        n_dofs, _B, axes=qd.static((1, 0) if static_rigid_sim_config.constraint_layout_batch_first else None)
    ):
        constraint_state.grad[i_d, i_b] = (
            constraint_state.Ma[i_d, i_b] - dofs_state.force[i_d, i_b] - constraint_state.qfrc_constraint[i_d, i_b]
        )

    if qd.static(static_rigid_sim_config.solver_type == gs.constraint_solver.CG):
        qd.loop_config(
            name="update_gradient_tiled", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32
        )
        for i_b in range(_B):
            func_solve_mass_batch(
                i_b,
                constraint_state.grad,
                constraint_state.Mgrad,
                None,
                entities_info=entities_info,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
                is_backward=False,
            )

    if qd.static(static_rigid_sim_config.solver_type == gs.constraint_solver.Newton):
        # Warm-start path: dispatch through the fused factor+solve kernel so L stays in shared memory between factor
        # and solve. ``write_L_to_nt_H=True`` also writes L back to ``nt_H``, which the monolith body's first iter
        # needs for its incremental rank-1 Cholesky update.
        if qd.static(static_rigid_sim_config.enable_fused_factor_solve_init):
            func_cholesky_and_solve_fused_tiled(
                constraint_state, rigid_global_info, static_rigid_sim_config, write_L_to_nt_H=True
            )
        else:
            func_cholesky_solve_tiled(constraint_state, static_rigid_sim_config)


@qd.func
def func_update_gradient(
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    island_state: array_class.IslandState,
    static_rigid_sim_config: qd.template(),
):
    """
    Unified implementation of gradient updated optimized for both CPU and GPU backends.

    The tiled optimization is only supported on GPU backend and specifically optimized for it, falling back to the
    classical batched implementation when running on CPU backend.

    Note that the tiled cholesky factorization and solving is not systematically enabled because it is not always
    superior in terms of performance and does not support arbitrary matrix sizes. More specifically, tiling gets more
    beneficial as n_dofs increases, but n_dofs>=96 is not supported for now. It is the responsibility of the calling
    code to configure the static global flag `hessian_fits_shared` accordingly. Failing to do so will cause the
    requested shared memory allocation to exceed 48kB and raise an exception.
    """
    _B = constraint_state.jac.shape[2]

    if qd.static(
        not (static_rigid_sim_config.enable_tiled_cholesky_hessian and static_rigid_sim_config.hessian_fits_shared)
        or static_rigid_sim_config.backend == gs.cpu
        or static_rigid_sim_config.use_contact_island
    ):
        # CPU, or islands: the tiled factor/solve operates on the whole-env dense Hessian, but with islands nt_H
        # holds the per-island block-diagonal factor, so the gradient solve must go per-island (func_cholesky_solve_batch).
        qd.loop_config(
            name="update_gradient", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32
        )
        for i_b in range(_B):
            func_update_gradient_batch(
                i_b,
                dofs_state=dofs_state,
                entities_info=entities_info,
                constraint_state=constraint_state,
                rigid_global_info=rigid_global_info,
                island_state=island_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )
    else:
        # GPU
        qd.loop_config(name="update_gradient")
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
    qacc: qd.Tensor,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    if qd.static(static_rigid_sim_config.parallel_init):
        _initialize_Jaref_parallel(
            qacc=qacc,
            constraint_state=constraint_state,
            static_rigid_sim_config=static_rigid_sim_config,
        )
    else:
        _initialize_Jaref_per_env(
            qacc=qacc,
            constraint_state=constraint_state,
            static_rigid_sim_config=static_rigid_sim_config,
        )


@qd.func
def _initialize_Jaref_body(
    i_c,
    i_b,
    n_dofs,
    qacc: qd.template(),
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    Jaref = -constraint_state.aref[i_c, i_b]
    # Sparse support (jac_dofs_idx) helps the CPU skyline solve and the per-island GPU solve, but its variable trip
    # count diverges the 32-env-packed warp; islands-OFF GPU iterates dense (uniform), matching the non-island baseline.
    if qd.static(static_rigid_sim_config.sparse_solve or static_rigid_sim_config.enable_per_island_solve):
        for i_d_ in range(constraint_state.jac_n_dofs[i_c, i_b]):
            i_d = constraint_state.jac_dofs_idx[i_c, i_d_, i_b]
            Jaref = Jaref + constraint_state.jac[i_c, i_d, i_b] * qacc[i_d, i_b]
    else:
        for i_d in range(n_dofs):
            Jaref = Jaref + constraint_state.jac[i_c, i_d, i_b] * qacc[i_d, i_b]
    constraint_state.Jaref[i_c, i_b] = Jaref


@qd.func
def _initialize_Jaref_per_env(
    qacc: qd.template(),
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    _B = constraint_state.jac.shape[2]
    n_dofs = constraint_state.jac.shape[1]

    qd.loop_config(name="init_jaref", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        for i_c in range(constraint_state.n_constraints[i_b]):
            _initialize_Jaref_body(i_c, i_b, n_dofs, qacc, constraint_state, static_rigid_sim_config)


@qd.func
def _initialize_Jaref_parallel(
    qacc: qd.template(),
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """Initialize Jaref = J @ qacc, parallelised over (constraint, env)."""
    _B = constraint_state.jac.shape[2]
    n_dofs = constraint_state.jac.shape[1]
    len_constraints = constraint_state.Jaref.shape[0]

    # Innermost ndrange axis matches the stride-1 axis of jac so jac loads coalesce: i_c-innermost under the flipped
    # layout, i_b-innermost under canonical.
    qd.loop_config(name="init_jaref_parallel", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_c, i_b in qd.ndrange(
        len_constraints, _B, axes=qd.static((1, 0) if static_rigid_sim_config.constraint_layout_batch_first else None)
    ):
        if i_c < constraint_state.n_constraints[i_b]:
            _initialize_Jaref_body(i_c, i_b, n_dofs, qacc, constraint_state, static_rigid_sim_config)


@qd.func
def initialize_Ma(
    Ma: qd.Tensor,
    qacc: qd.Tensor,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    _B = rigid_global_info.mass_mat.shape[2]
    n_dofs = qacc.shape[0]

    # Flipped mass_mat layout=(2,1,0): physical (_B, n_dofs, n_dofs) with i_d1 stride-1. Make i_d1 the innermost
    # ndrange axis so adjacent lanes vary i_d1 -> coalesced reads of mass_mat[i_d1, i_d2, i_b]. qacc[i_d2, i_b] is
    # constant within the warp -> broadcast load.
    qd.loop_config(name="init_ma", serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d1, i_b in qd.ndrange(
        n_dofs, _B, axes=qd.static((1, 0) if static_rigid_sim_config.constraint_layout_batch_first else None)
    ):
        Ma_ = gs.qd_float(0.0)
        # Mass couples only DOFs within the same kinematic-tree block, so restrict to i_d1's block (cross-block is zero).
        for i_d2 in range(rigid_global_info.dofs_mass_block_start[i_d1], rigid_global_info.dofs_mass_block_end[i_d1]):
            Ma_ = Ma_ + rigid_global_info.mass_mat[i_d1, i_d2, i_b] * qacc[i_d2, i_b]
        Ma[i_d1, i_b] = Ma_


# ======================================================= Core ========================================================


@qd.kernel(fastcache=True)
def func_solve_init(
    dofs_info: array_class.DofsInfo,
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    island_state: array_class.IslandState,
    static_rigid_sim_config: qd.template(),
    is_decomposed: qd.template(),
):
    # is_decomposed is a hardcoded constant forwarded by the dispatch entrypoint that calls this (the decomposed arm
    # passes True, the monolith passes False). func_solve_init runs as a separate kernel before the perf-dispatcher
    # picks an arm, so it CANNOT detect the arm itself - the entrypoint must declare it. The decomposed arm rebuilds
    # the Hessian on its first graph iteration regardless, so it skips the init factor/gradient here entirely.
    _B = dofs_state.acc_smooth.shape[1]
    n_dofs = dofs_state.acc_smooth.shape[0]

    # Group the assembled constraints by island. The island partition itself (links_island_idx / dof_id / contact
    # ordering) is built earlier, in add_inequality_constraints, before the contact constraints are assembled; here we
    # only resolve each constraint's island (parallel per-(env, constraint)) and gather them into contiguous per-island
    # ranges (per-env), which needs the assembled jac and so cannot move earlier.
    if qd.static(static_rigid_sim_config.enable_per_island_solve):
        EPS = rigid_global_info.EPS[None]
        capacity = island_state.constraint_island_idx.shape[0]
        qd.loop_config(name="resolve_constraint_island", serialize=False)
        for i_flat in range(_B * capacity):
            i_b = i_flat // capacity
            i_c = i_flat % capacity
            # A single-island env groups by identity (func_group_constraints_by_island), so its per-constraint island
            # label is unused - skip resolving it.
            if island_state.n_islands[i_b] > 1 and i_c < constraint_state.n_constraints[i_b]:
                island_state.constraint_island_idx[i_c, i_b] = func_constraint_island(
                    constraint_state, island_state, i_c, i_b, n_dofs, EPS, static_rigid_sim_config
                )
        qd.loop_config(
            name="group_constraints_by_island",
            serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL),
        )
        for i_b in range(_B):
            func_group_constraints_by_island(i_b, island_state, constraint_state, static_rigid_sim_config)

    # Skyline envelope for the CPU sparse Cholesky, recomputed each step (the fill-reducing DOF permutation it builds
    # on is fixed at build time). Folded here rather than a standalone kernel to avoid a per-step launch.
    if qd.static(static_rigid_sim_config.sparse_envelope):
        qd.loop_config(
            name="solve_init_sparsity_pattern", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL
        )
        for i_b in range(_B):
            func_compute_sparsity_pattern(i_b, constraint_state, rigid_global_info)

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
        qd.loop_config(
            name="solve_init_pick_warmstart", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL
        )
        for i_d, i_b in qd.ndrange(n_dofs, _B):
            if constraint_state.cost_ws[i_b] < constraint_state.cost[i_b]:
                constraint_state.qacc[i_d, i_b] = constraint_state.qacc_ws[i_d, i_b]
                constraint_state.Ma[i_d, i_b] = constraint_state.Ma_ws[i_d, i_b]
            else:
                constraint_state.qacc[i_d, i_b] = dofs_state.acc_smooth[i_d, i_b]
    else:
        # Always initialize from warmstart.
        # Under the DOF-vec flip, both qacc and qacc_ws are env-leading; swap the ndrange so adjacent lanes vary i_d
        # to coalesce those writes/reads. The dofs_state.acc_smooth read remains canonical (small per-env working
        # set, dominated by the qacc write).
        qd.loop_config(name="from_warmstart", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_d, i_b in qd.ndrange(
            n_dofs, _B, axes=qd.static((1, 0) if static_rigid_sim_config.constraint_layout_batch_first else None)
        ):
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

    qd.loop_config(name="init_improved", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in qd.ndrange(_B):
        constraint_state.improved[i_b] = constraint_state.n_constraints[i_b] > 0
        constraint_state.use_full_hessian[i_b] = 1
    constraint_state.solver_iter_counter[()] = 0

    if qd.static(
        static_rigid_sim_config.solver_type == gs.constraint_solver.Newton
        and static_rigid_sim_config.use_contact_island
        and static_rigid_sim_config.enable_fused_factor_solve_init
        and not static_rigid_sim_config.use_hibernation
    ):
        # Islands ON, whole env fits shared, no hibernation: seed via the whole-env fused path, identical to islands
        # off. The Hessian is block-diagonal by island, so its whole-env Cholesky is the exact per-island result, and
        # the fused factor+solve (L in shared) has none of the per-(env, island) overhead - which is pure cost at the
        # env counts where the env dimension alone saturates the GPU. func_hessian_direct_tiled assembles the full H;
        # func_update_gradient_tiled builds grad and runs the fused factor+solve, writing L back to nt_H for the
        # monolith body's incremental iterations (write_L_to_nt_H inside, gated on enable_fused_factor_solve_init).
        func_hessian_direct_tiled(constraint_state, rigid_global_info)
        func_update_gradient_tiled(
            dofs_state=dofs_state,
            entities_info=entities_info,
            constraint_state=constraint_state,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )
    elif qd.static(
        static_rigid_sim_config.solver_type == gs.constraint_solver.Newton
        and static_rigid_sim_config.enable_per_island_solve
        and static_rigid_sim_config.enable_cooperative_constraint_kernels
    ):
        # GPU-island seed (hibernation): the per-island tiled assemble+factor+solve - the same barrier-free factor the
        # decomposed graph runs every iteration. The shared tile is sized per-island (tiled_n_island_dofs), so this runs
        # whenever the cooperative kernels are enabled, NOT only when the whole env fits shared - many small islands all
        # factor in their own tile even when the whole env is large (no whole-env cubic). A single island spanning the
        # env factors with the full T-lane tile (identical to islands-off); an island exceeding the per-island shared
        # capacity falls back to the scalar per-island solve inside the factor. It
        # solves grad -> Mgrad directly, subsuming the separate gradient solve. The monolith reads L back from nt_H in
        # its incremental iterations so it persists L (write_L=True); the decomposed graph re-factors each iteration so
        # it keeps nt_H holding the raw Hessian (write_L=False).
        func_update_gradient_no_solve(
            entities_info=entities_info,
            dofs_state=dofs_state,
            constraint_state=constraint_state,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )
        func_island_tiled_factor_solve_all(
            entities_info,
            constraint_state,
            island_state,
            rigid_global_info,
            static_rigid_sim_config,
            qd.simt.Tile32x32 if qd.static(static_rigid_sim_config.cholesky_tile_size == 32) else qd.simt.Tile16x16,
            do_assemble=True,
            write_L=qd.static(not is_decomposed),
        )
    else:
        if qd.static(
            static_rigid_sim_config.solver_type == gs.constraint_solver.Newton
            and (
                is_decomposed
                or not (
                    static_rigid_sim_config.use_contact_island
                    and static_rigid_sim_config.backend != gs.cpu
                    and not static_rigid_sim_config.enable_cooperative_constraint_kernels
                )
            )
        ):
            # Seed the initial Hessian factor. The decomposed arm has no self-init: its graph is linesearch-first, so
            # its first linesearch consumes the search direction computed here (this kernel is its "iteration 0"; the
            # graph then computes each subsequent direction at the end of an iteration). So it ALWAYS needs this seed,
            # islands on or off. The monolith seeds it here except in the one case where its body self-inits the factor
            # per-env: the GPU island arm with the cooperative kernels disabled (the only case keyed below). With the
            # cooperative kernels enabled the body does NOT self-init, so the seed must run here even for GPU islands -
            # otherwise a shared-fitting Hessian at an env count that oversaturates the GPU (where neither the fused nor
            # the per-island seed branch above fires) would leave Mgrad stale.
            # compute_envelope=True computes each island's structural skyline envelope once, reused per iteration.
            func_hessian_and_cholesky_factor_direct(
                island_state=island_state,
                entities_info=entities_info,
                constraint_state=constraint_state,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
                compute_envelope=True,
            )

        if qd.static(
            not (
                static_rigid_sim_config.solver_type == gs.constraint_solver.Newton
                and not is_decomposed
                and static_rigid_sim_config.use_contact_island
                and static_rigid_sim_config.backend != gs.cpu
                and not static_rigid_sim_config.enable_cooperative_constraint_kernels
            )
        ):
            # Initial gradient (Mgrad = H^-1 grad for Newton, grad for CG). Seeds the decomposed arm's first search
            # direction, so it runs for the decomposed arm in all cases. Skipped only for the GPU island monolith with
            # the cooperative kernels disabled, which self-inits the gradient per-env in its own body; with them enabled
            # the body does not self-init, so the seed must run here.
            func_update_gradient(
                dofs_state=dofs_state,
                entities_info=entities_info,
                constraint_state=constraint_state,
                rigid_global_info=rigid_global_info,
                island_state=island_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )

    qd.loop_config(name="assign_search", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_d, i_b in qd.ndrange(
        n_dofs, _B, axes=qd.static((1, 0) if static_rigid_sim_config.constraint_layout_batch_first else None)
    ):
        constraint_state.search[i_d, i_b] = -constraint_state.Mgrad[i_d, i_b]


@qd.func
def func_solve_iter(
    i_b,
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    island_state: array_class.IslandState,
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
            # Islands (if any) are handled inside the per-env factor funcs. The sparse path always rebuilds the
            # Hessian directly (the incremental rank-1 update assumes globally descending DOF order in jac_dofs_idx,
            # which does not hold for cross-entity constraints); the dense path uses the incremental update, falling
            # back to a direct rebuild when it degenerates.
            if qd.static(static_rigid_sim_config.sparse_solve):
                func_hessian_and_cholesky_factor_direct_batch(
                    i_b,
                    island_state=island_state,
                    entities_info=entities_info,
                    constraint_state=constraint_state,
                    rigid_global_info=rigid_global_info,
                    static_rigid_sim_config=static_rigid_sim_config,
                )
            else:
                is_degenerated = func_hessian_and_cholesky_factor_incremental_batch(
                    i_b,
                    island_state=island_state,
                    entities_info=entities_info,
                    constraint_state=constraint_state,
                    rigid_global_info=rigid_global_info,
                    static_rigid_sim_config=static_rigid_sim_config,
                )
                if is_degenerated:
                    func_hessian_and_cholesky_factor_direct_batch(
                        i_b,
                        island_state=island_state,
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
            island_state=island_state,
            static_rigid_sim_config=static_rigid_sim_config,
        )

        func_terminate_or_update_descent_batch(
            i_b,
            constraint_state=constraint_state,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )


def _get_static_config(*args, **kwargs):
    return args[5] if len(args) > 5 else kwargs["static_rigid_sim_config"]


@qd.perf_dispatch(
    get_geometry_hash=lambda *args, **kwargs: (*args, frozendict(kwargs)),
    first_warmup=1,
    warmup=0,
    active=2,
    repeat_after_seconds=5,
)
def func_solve_body(
    entities_info: array_class.EntitiesInfo,
    dofs_info: array_class.DofsInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    _n_iterations: int,
    island_state: array_class.IslandState,
) -> None: ...


@qd.kernel(fastcache=True)
def _kernel_solve_monolith(
    entities_info: array_class.EntitiesInfo,
    dofs_info: array_class.DofsInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    _n_iterations: int,
    island_state: array_class.IslandState,
):
    _B = constraint_state.grad.shape[1]
    n_dofs = constraint_state.qacc.shape[0]

    # The monolith arm solves each env whole (32 envs packed per warp); islands change only the per-env factor's block
    # structure, handled inside func_solve_iter, not the iteration scheme. Per-island parallelism is the decomposed
    # arm's job, so there is no separate island body here - ON and OFF run the identical packed-env solve.
    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    for i_b in range(_B):
        # A fully-asleep env has no awake DOF to move, so its Newton solve is a no-op. Skip the whole iteration loop
        # so step time tracks the awake set, not the total body count.
        has_awake_work = constraint_state.n_constraints[i_b] > 0
        if qd.static(static_rigid_sim_config.use_hibernation):
            has_awake_work = has_awake_work and rigid_global_info.n_awake_dofs[i_b] > 0
        if has_awake_work:
            if qd.static(
                static_rigid_sim_config.use_contact_island
                and static_rigid_sim_config.backend != gs.cpu
                and static_rigid_sim_config.solver_type == gs.constraint_solver.Newton
                and not static_rigid_sim_config.enable_cooperative_constraint_kernels
            ):
                # Without the cooperative kernels, func_solve_init skips the GPU-island tiled seed, so the monolith
                # seeds this env's initial scalar per-island factor + gradient + search here. Gives the first
                # linesearch a valid Newton direction; once per step, then iterate. With cooperative kernels enabled,
                # func_solve_init already seeded the factor (L persisted to nt_H) and the search direction.
                func_hessian_and_cholesky_factor_direct_batch(
                    i_b,
                    island_state=island_state,
                    entities_info=entities_info,
                    constraint_state=constraint_state,
                    rigid_global_info=rigid_global_info,
                    static_rigid_sim_config=static_rigid_sim_config,
                    compute_envelope=True,
                )
                func_update_gradient_batch(
                    i_b,
                    dofs_state=dofs_state,
                    entities_info=entities_info,
                    rigid_global_info=rigid_global_info,
                    constraint_state=constraint_state,
                    island_state=island_state,
                    static_rigid_sim_config=static_rigid_sim_config,
                )
                for i_d in range(n_dofs):
                    constraint_state.search[i_d, i_b] = -constraint_state.Mgrad[i_d, i_b]
            for _ in range(rigid_global_info.iterations[None]):
                func_solve_iter(
                    i_b,
                    entities_info=entities_info,
                    dofs_state=dofs_state,
                    rigid_global_info=rigid_global_info,
                    constraint_state=constraint_state,
                    island_state=island_state,
                    static_rigid_sim_config=static_rigid_sim_config,
                )
                if not constraint_state.improved[i_b]:
                    break
        else:
            constraint_state.improved[i_b] = False


@func_solve_body.register(
    # Runs whenever the decomposed arm is not specifically preferred. Solves each env whole with 32 envs packed per
    # warp; islands only reshape the per-env factor into block-diagonal blocks, they do not change the solve scheme.
    is_compatible=lambda *args, **kwargs: (
        (static_rigid_sim_config := _get_static_config(*args, **kwargs)).prefer_decomposed_solver != 1
    )
)
def func_solve_body_monolith(
    entities_info,
    dofs_info,
    dofs_state,
    constraint_state,
    rigid_global_info,
    static_rigid_sim_config,
    _n_iterations,
    island_state,
):
    # This entrypoint statically IS the monolith arm, so it owns its init: it forwards is_decomposed=False to
    # func_solve_init (which groups the constraints by island, factors, and seeds the gradient the packed-env body
    # consumes), then runs the solve kernel. Keeping the init inside the entrypoint (rather than in resolve, before the
    # dispatch) is what lets each arm declare its own init behavior - the dispatcher may run a different arm on the next
    # step during autotuning.
    func_solve_init(
        dofs_info,
        dofs_state,
        entities_info,
        constraint_state,
        rigid_global_info,
        island_state,
        static_rigid_sim_config,
        is_decomposed=False,
    )
    _kernel_solve_monolith(
        entities_info,
        dofs_info,
        dofs_state,
        constraint_state,
        rigid_global_info,
        static_rigid_sim_config,
        _n_iterations,
        island_state,
    )


# =====================================================================================================================
# ==================================================== Finalization ===================================================
# =====================================================================================================================


@qd.kernel(fastcache=True)
def func_update_contact_force(
    links_state: array_class.LinksState,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    n_links = links_state.contact_force.shape[0]
    _B = links_state.contact_force.shape[1]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_l, i_b in qd.ndrange(n_links, _B):
        links_state.contact_force[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        const_start = constraint_state.n_constraints_equality[i_b] + constraint_state.n_constraints_frictionloss[i_b]

        # contact constraints should be after equality and frictionloss constraints and before joint limit constraints
        for i_c in range(collider_state.n_contacts[i_b]):
            i_col = collider_state.contact_sort_idx[i_c, i_b]
            contact_data_normal = collider_state.contact_data.normal[i_col, i_b]
            contact_data_friction = collider_state.contact_data.friction[i_col, i_b]
            contact_data_link_a = collider_state.contact_data.link_a[i_col, i_b]
            contact_data_link_b = collider_state.contact_data.link_b[i_col, i_b]

            force = qd.Vector.zero(gs.qd_float, 3)
            d1, d2 = gu.qd_orthogonals(contact_data_normal)
            for i_dir in range(4):
                d = (2 * (i_dir % 2) - 1) * (d1 if i_dir < 2 else d2)
                n = d * contact_data_friction - contact_data_normal
                force = force + n * constraint_state.efc_force[i_c * 4 + i_dir + const_start, i_b]

            collider_state.contact_data.force[i_col, i_b] = force

            links_state.contact_force[contact_data_link_a, i_b] = (
                links_state.contact_force[contact_data_link_a, i_b] - force
            )
            links_state.contact_force[contact_data_link_b, i_b] = (
                links_state.contact_force[contact_data_link_b, i_b] + force
            )


@qd.kernel(fastcache=True)
def func_update_qacc(
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
    errno: qd.Tensor,
):
    n_dofs = dofs_state.acc.shape[0]
    _B = dofs_state.acc.shape[1]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
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
