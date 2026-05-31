"""
ComFree (Complementarity-Free) constraint solver.

Replaces iterative complementarity-based constraint solving (Newton/CG) with a
single-pass analytical contact-force computation. It reuses the existing Genesis
constraint-assembly infrastructure (Jacobians, collision detection, equality /
joint-limit / frictionloss rows) but resolves forces with an impedance-style
prediction-correction update instead of an optimization loop.

Reference implementation:
  comfree_warp/comfree_core/_src/forward.py  (compute_qfrc_total / forward_comfree)
  comfree_warp/comfree_core/_src/constraint.py  (_efc_row -> efc_dist / efc_mass)
"""

from typing import TYPE_CHECKING

import numpy as np
import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class

from ..abd import func_solve_mass_batch
from ..collider.contact_island import ContactIsland

if TYPE_CHECKING:
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver


class ComFreeSolver:
    """Complementarity-free analytical constraint solver.

    For each constraint row ``c`` with Jacobian ``J_c``, signed position ``efc_dist``
    and effective mass ``efc_mass`` the per-step update is (reference forward.py:67-85)::

        v_pred   = v + acc_smooth * dt                       # smooth predicted velocity
        efc_vel  = J_c @ v_pred
        efc_pen  = efc_vel * dt + efc_dist                   # predictive penetration
        force    = max(efc_mass * (-d*efc_vel - k*efc_pen), 0)
        qfrc    += J_c^T * force

    then ``qacc = M^{-1} (qf_smooth + qfrc)``. ``k`` and ``d`` are the user stiffness
    and damping scaled by ``1/dt``.

    Parameters
    ----------
    rigid_solver : RigidSolver
        The parent rigid body solver.
    comfree_stiffness : float
        Global stiffness parameter ``k_user`` (default 0.2).
    comfree_damping : float
        Global damping parameter ``d_user`` (default 0.001).
    """

    def __init__(self, rigid_solver: "RigidSolver", comfree_stiffness: float = 0.2, comfree_damping: float = 0.001):
        self._solver = rigid_solver
        self._collider = rigid_solver.collider
        self._B = rigid_solver._B

        self._comfree_stiffness = float(comfree_stiffness)
        self._comfree_damping = float(comfree_damping)

        # Over-estimate the constraint count exactly like the standard solver:
        #   * 4 constraints per contact
        #   * 1 constraint per 1-DoF joint limit
        #   * 1 constraint per DoF frictionloss
        #   * up to 6 constraints per equality (weld)
        self.len_constraints = int(
            4 * rigid_solver.collider._collider_info.max_contact_pairs[None]
            + sum(joint.type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC) for joint in self._solver.joints)
            + self._solver.n_dofs
            + self._solver.n_candidate_equalities_ * 6
        )
        self.len_constraints_ = max(1, self.len_constraints)

        # ComFree never builds a sparse Jacobian; needed by get_constraint_state allocation.
        self.sparse_solve = False

        # Reuse the standard constraint-state allocation (jac, efc_dist, efc_mass, efc_force, ...).
        self.constraint_state = array_class.get_constraint_state(self, self._solver)
        self.constraint_state.qd_n_equalities.from_numpy(
            np.full((self._solver._B,), self._solver.n_equalities, dtype=gs.np_int)
        )

        self.reset()

        # ContactIsland is required as a parameter by some kernels (e.g. kernel_forward_dynamics);
        # it is inert when hibernation / contact-island mode is disabled.
        self.contact_island = ContactIsland(self._collider)

    def reset(self, envs_idx=None):
        envs_idx = self._solver._scene._sanitize_envs_idx(envs_idx)
        _comfree_solver_kernel_reset(envs_idx, self.constraint_state, self._solver._static_rigid_sim_config)

    def clear(self, envs_idx=None):
        self.reset(envs_idx)
        envs_idx = self._solver._scene._sanitize_envs_idx(envs_idx)
        _comfree_solver_kernel_clear(
            envs_idx,
            self.constraint_state,
            self._solver._rigid_global_info,
            self._solver._static_rigid_sim_config,
        )

    def add_equality_constraints(self):
        from ..constraint.solver import add_equality_constraints

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
        from ..constraint.solver import add_inequality_constraints

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

    def add_constraints(self):
        """Contact-island assembly path (kept for interface parity with ConstraintSolverIsland)."""
        from ..constraint.solver_island import add_constraints

        add_constraints(
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
            self.contact_island.contact_island_state,
        )

    def resolve(self, entities_info=None, rigid_global_info=None):
        """Resolve constraints with the ComFree analytical update, then publish contact forces."""
        from ..constraint.solver import func_update_contact_force

        _comfree_resolve(
            self._comfree_stiffness,
            self._comfree_damping,
            self._solver.dofs_state,
            self._solver.entities_info,
            self.constraint_state,
            self._solver._rigid_global_info,
            self._solver._static_rigid_sim_config,
            self._solver._errno,
        )

        func_update_contact_force(
            self._solver.links_state,
            self._collider._collider_state,
            self.constraint_state,
            self._solver._static_rigid_sim_config,
        )


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@qd.kernel(fastcache=True)
def _comfree_solver_kernel_reset(
    envs_idx: qd.types.ndarray(),
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    n_dofs = constraint_state.qacc_ws.shape[0]

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        constraint_state.is_warmstart[i_b] = False
        for i_d in range(n_dofs):
            constraint_state.qacc_ws[i_d, i_b] = 0.0


@qd.kernel(fastcache=True)
def _comfree_solver_kernel_clear(
    envs_idx: qd.types.ndarray(),
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        constraint_state.n_constraints[i_b] = 0
        constraint_state.n_constraints_equality[i_b] = 0
        constraint_state.n_constraints_frictionloss[i_b] = 0
        constraint_state.qd_n_equalities[i_b] = rigid_global_info.n_equalities[None]


@qd.kernel(fastcache=True)
def _comfree_resolve(
    comfree_stiffness: qd.template(),
    comfree_damping: qd.template(),
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    errno: qd.Tensor,
):
    """Single-pass analytical constraint resolution (reference forward.py:43-97 + solve_m).

    1. Predict smooth velocity ``v_pred = v + acc_smooth * dt``.
    2. For each active constraint compute its analytical force and accumulate ``qfrc_constraint``.
    3. Solve ``M @ qacc = qf_smooth + qfrc_constraint`` with the pre-factored mass matrix.
    """
    n_dofs = dofs_state.acc.shape[0]
    _B = dofs_state.acc.shape[1]
    substep_dt = rigid_global_info.substep_dt[None]

    # Stiffness / damping scaled by 1/dt (reference forward.py:72-73).
    stiffness = qd.static(comfree_stiffness) / substep_dt
    damping = qd.static(comfree_damping) / substep_dt

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        n_c = constraint_state.n_constraints[i_b]

        for i_d in range(n_dofs):
            constraint_state.qfrc_constraint[i_d, i_b] = 0.0

        if n_c == 0:
            # No constraints: the constrained acceleration is just the smooth acceleration.
            for i_d in range(n_dofs):
                constraint_state.qacc[i_d, i_b] = dofs_state.acc_smooth[i_d, i_b]
        else:
            for i_c in range(n_c):
                # efc_vel = J_c @ (v + acc_smooth * dt)
                efc_vel = gs.qd_float(0.0)
                for i_d in range(n_dofs):
                    v_pred = dofs_state.vel[i_d, i_b] + dofs_state.acc_smooth[i_d, i_b] * substep_dt
                    efc_vel += constraint_state.jac[i_c, i_d, i_b] * v_pred

                efc_dist = constraint_state.efc_dist[i_c, i_b]
                efc_mass = constraint_state.efc_mass[i_c, i_b]

                # Predictive penetration and analytical, one-sided constraint force.
                efc_penetration = efc_vel * substep_dt + efc_dist
                efc_acc = -damping * efc_vel - stiffness * efc_penetration
                efc_frc = qd.max(efc_mass * efc_acc, 0.0)

                constraint_state.efc_force[i_c, i_b] = efc_frc

                # qfrc_constraint += J_c^T * force
                for i_d in range(n_dofs):
                    constraint_state.qfrc_constraint[i_d, i_b] += constraint_state.jac[i_c, i_d, i_b] * efc_frc

            # qfrc_total = qf_smooth + qfrc_constraint
            for i_d in range(n_dofs):
                dofs_state.force[i_d, i_b] = dofs_state.qf_smooth[i_d, i_b] + constraint_state.qfrc_constraint[i_d, i_b]

            # qacc = M^{-1} @ qfrc_total
            func_solve_mass_batch(
                i_b,
                dofs_state.force,
                constraint_state.qacc,
                None,
                entities_info=entities_info,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
                is_backward=False,
            )

    # Publish results back to dofs_state (mirrors func_update_qacc).
    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        dofs_state.acc[i_d, i_b] = constraint_state.qacc[i_d, i_b]
        dofs_state.qf_constraint[i_d, i_b] = constraint_state.qfrc_constraint[i_d, i_b]
        dofs_state.force[i_d, i_b] = dofs_state.qf_smooth[i_d, i_b] + constraint_state.qfrc_constraint[i_d, i_b]
        constraint_state.qacc_ws[i_d, i_b] = constraint_state.qacc[i_d, i_b]
        if qd.math.isnan(constraint_state.qacc[i_d, i_b]):
            errno[i_b] = errno[i_b] | array_class.ErrorCode.INVALID_FORCE_NAN

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        constraint_state.is_warmstart[i_b] = True
