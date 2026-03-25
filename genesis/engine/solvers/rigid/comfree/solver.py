"""
ComFree (Complementarity-Free) constraint solver.

Replaces iterative complementarity-based constraint solving with a single-pass
analytical contact force computation. Uses the existing Genesis constraint
assembly infrastructure (Jacobians, collision detection) but resolves forces
via an impedance-style prediction-correction update.

Reference implementation: references/comfree_warp/comfree_warp/comfree_core/_src/forward.py
"""

from typing import TYPE_CHECKING

import numpy as np
import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class

from ..abd.forward_dynamics import func_solve_mass_batch
from ..collider.contact_island import ContactIsland

if TYPE_CHECKING:
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver


class ComFreeSolver:
    """Complementarity-free analytical constraint solver.

    Instead of iteratively solving a complementarity problem (Newton/CG),
    ComFree computes constraint forces in closed form:

        1. Predict smooth velocity: v_smooth = v + acc_smooth * dt
        2. Compute constraint velocity: efc_vel = J @ v_smooth
        3. Compute penetration: efc_pen = efc_vel * dt + efc_dist
        4. Compute force: efc_force = max(efc_mass * (-d*efc_vel - k*efc_pen), 0)
        5. Accumulate: qfrc_constraint = J^T @ efc_force
        6. Solve: qacc = M^{-1} @ (qf_smooth + qfrc_constraint)

    Parameters
    ----------
    rigid_solver : RigidSolver
        The parent rigid body solver.
    comfree_stiffness : float
        Global stiffness parameter k_user (default 0.2).
    comfree_damping : float
        Global damping parameter d_user (default 0.001).
    """

    def __init__(self, rigid_solver: "RigidSolver", comfree_stiffness: float = 0.2, comfree_damping: float = 0.001):
        self._solver = rigid_solver
        self._collider = rigid_solver.collider
        self._B = rigid_solver._B

        self._comfree_stiffness = comfree_stiffness
        self._comfree_damping = comfree_damping

        # Estimate constraint count same way as the standard solver
        self.len_constraints = int(
            4 * rigid_solver.collider._collider_info.max_contact_pairs[None]
            + sum(joint.type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC) for joint in self._solver.joints)
            + self._solver.n_dofs
            + self._solver.n_candidate_equalities_ * 6
        )
        self.len_constraints_ = max(1, self.len_constraints)

        # Need these for compatibility with existing constraint assembly
        self.sparse_solve = False

        # Allocate constraint state (reuses standard allocation)
        self.constraint_state = array_class.get_constraint_state(self, self._solver)
        self.constraint_state.qd_n_equalities.from_numpy(
            np.full((self._solver._B,), self._solver.n_equalities, dtype=gs.np_int)
        )

        # Convenient aliases (needed by constraint assembly functions)
        cs = self.constraint_state
        self.qd_n_equalities = cs.qd_n_equalities
        self.jac = cs.jac
        self.diag = cs.diag
        self.aref = cs.aref
        self.n_constraints = cs.n_constraints
        self.n_constraints_equality = cs.n_constraints_equality
        self.n_constraints_frictionloss = cs.n_constraints_frictionloss
        self.efc_D = cs.efc_D
        self.efc_force = cs.efc_force
        self.active = cs.active
        self.prev_active = cs.prev_active
        self.qfrc_constraint = cs.qfrc_constraint
        self.qacc = cs.qacc
        self.qacc_ws = cs.qacc_ws
        self.qacc_prev = cs.qacc_prev
        self.Ma = cs.Ma
        self.Ma_ws = cs.Ma_ws
        self.Jaref = cs.Jaref
        self.efc_frictionloss = cs.efc_frictionloss
        self.jac_relevant_dofs = cs.jac_relevant_dofs
        self.jac_n_relevant_dofs = cs.jac_n_relevant_dofs
        self.improved = cs.improved
        self.grad = cs.grad
        self.Mgrad = cs.Mgrad
        self.search = cs.search
        self.mv = cs.mv
        self.jv = cs.jv
        self.cost_ws = cs.cost_ws
        self.gauss = cs.gauss
        self.cost = cs.cost
        self.prev_cost = cs.prev_cost
        self.gtol = cs.gtol
        self.quad_gauss = cs.quad_gauss
        self.candidates = cs.candidates
        self.ls_it = cs.ls_it
        self.ls_result = cs.ls_result

        self.reset()

        # ContactIsland needed for compatibility with kernel_step_2
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
        """For contact island mode compatibility."""
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

    def resolve(self):
        """Resolve constraints using the ComFree analytical method."""
        from ..constraint.solver import func_update_contact_force

        _comfree_resolve(
            self._comfree_stiffness,
            self._comfree_damping,
            self._solver.dofs_state,
            self._solver.dofs_info,
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
# Quadrants Kernels
# ---------------------------------------------------------------------------


@qd.kernel(fastcache=gs.use_fastcache)
def _comfree_solver_kernel_reset(
    envs_idx: array_class.V_ANNOTATION,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    _B = constraint_state.is_warmstart.shape[0]
    n_dofs = constraint_state.qacc_ws.shape[0]

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        constraint_state.is_warmstart[i_b] = False
        for i_d in range(n_dofs):
            constraint_state.qacc_ws[i_d, i_b] = 0.0


@qd.kernel(fastcache=gs.use_fastcache)
def _comfree_solver_kernel_clear(
    envs_idx: array_class.V_ANNOTATION,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    _B = constraint_state.n_constraints.shape[0]

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        constraint_state.n_constraints[i_b] = 0
        constraint_state.n_constraints_equality[i_b] = 0
        constraint_state.n_constraints_frictionloss[i_b] = 0
        constraint_state.qd_n_equalities[i_b] = rigid_global_info.n_equalities[None]


@qd.kernel(fastcache=gs.use_fastcache)
def _comfree_resolve(
    comfree_stiffness: qd.template(),
    comfree_damping: qd.template(),
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    errno: array_class.V_ANNOTATION,
):
    """ComFree constraint resolution: single-pass analytical force computation.

    Implements the core ComFree algorithm (arXiv:2603.12185):
    1. Predict smooth velocity: v_smooth_pred = v + acc_smooth * dt
    2. For each constraint: compute force analytically
    3. Accumulate generalized constraint forces
    4. Solve for constrained acceleration: qacc = M^{-1} @ (qf_smooth + qfrc_constraint)

    Reference: comfree_warp/comfree_core/_src/forward.py:26-97
    """
    n_dofs = dofs_state.acc.shape[0]
    _B = dofs_state.acc.shape[1]
    substep_dt = rigid_global_info.substep_dt[None]

    # Stiffness and damping scaled by 1/dt (as in reference: forward.py:72-73)
    stiffness = qd.static(comfree_stiffness) / substep_dt
    damping = qd.static(comfree_damping) / substep_dt

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        n_c = constraint_state.n_constraints[i_b]

        # Zero constraint forces
        for i_d in range(n_dofs):
            constraint_state.qfrc_constraint[i_d, i_b] = 0.0

        if n_c == 0:
            # No constraints: qacc = acc_smooth
            for i_d in range(n_dofs):
                constraint_state.qacc[i_d, i_b] = dofs_state.acc_smooth[i_d, i_b]
        else:
            # For each constraint, compute ComFree force analytically
            for i_c in range(n_c):
                # Compute constraint velocity: efc_vel = J @ v_smooth_pred
                # where v_smooth_pred = v + acc_smooth * dt
                efc_vel = float(0.0)
                for i_d in range(n_dofs):
                    v_smooth_pred = dofs_state.vel[i_d, i_b] + dofs_state.acc_smooth[i_d, i_b] * substep_dt
                    efc_vel += constraint_state.jac[i_c, i_d, i_b] * v_smooth_pred

                # efc_dist = signed distance/constraint violation
                # efc_D = constraint impedance (includes friction scaling for contacts)
                efc_dist = constraint_state.efc_dist[i_c, i_b]
                efc_mass = constraint_state.efc_D[i_c, i_b]

                # Predictive penetration: efc_vel * dt + efc_dist
                efc_penetration = efc_vel * substep_dt + efc_dist

                # Analytical force: max(efc_mass * (-damping * efc_vel - stiffness * efc_penetration), 0)
                efc_acc = -damping * efc_vel - stiffness * efc_penetration
                efc_frc = efc_mass * efc_acc
                efc_frc = qd.max(efc_frc, 0.0)

                # Store force
                constraint_state.efc_force[i_c, i_b] = efc_frc

                # Accumulate generalized constraint force: qfrc_constraint += J^T @ efc_force
                for i_d in range(n_dofs):
                    constraint_state.qfrc_constraint[i_d, i_b] += constraint_state.jac[i_c, i_d, i_b] * efc_frc

            # Update total force and solve for constrained acceleration
            # qfrc_total = qf_smooth + qfrc_constraint
            for i_d in range(n_dofs):
                dofs_state.force[i_d, i_b] = dofs_state.qf_smooth[i_d, i_b] + constraint_state.qfrc_constraint[i_d, i_b]

            # Solve M @ qacc = qfrc_total using pre-factored mass matrix
            func_solve_mass_batch(
                i_b,
                vec=dofs_state.force,
                out=constraint_state.qacc,
                out_bw=constraint_state.qacc,
                entities_info=entities_info,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
                is_backward=False,
            )

    # Write results back to dofs_state (same as standard solver's func_update_qacc)
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
