"""ComFree (complementarity-free) constraint solver.

Computes every constraint force analytically in a single pass (https://arxiv.org/abs/2603.12185) instead of
iterating a complementarity solve to convergence. It shares the whole constraint-assembly pipeline (Jacobians, contact,
equality, joint-limit, and frictionloss rows) with the iterative solvers and only replaces the resolution.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class

from ..abd import func_solve_mass
from .solver import ConstraintSolver, func_update_contact_force, func_update_qacc


class ComFreeSolver(ConstraintSolver):
    """Analytical constraint solver: per-row impedance forces in closed form, then one mass solve.

    For each constraint row c with Jacobian J_c, signed position efc_dist and effective mass efc_mass:

        v_pred  = v + acc_smooth * dt
        efc_vel = J_c @ v_pred
        force   = efc_mass * (-d * efc_vel - k * (efc_vel * dt + efc_dist))

    with k and d the user stiffness and damping scaled by 1 / dt. Each force is then projected onto its row's
    admissible set (see kernel_comfree_resolve), and qacc = M^-1 (qf_smooth + J^T @ force).
    """

    def resolve(self):
        kernel_comfree_resolve(
            self._solver.dyn_state,
            self.constraint_state,
            self._solver.dyn_info,
            self._solver.rigid_info,
            self._solver.rigid_config,
        )

        func_update_qacc(self._solver.dyn_state, self.constraint_state, self._solver.rigid_config, self._solver._errno)

        func_update_contact_force(
            self._solver.dyn_state, self._collider._collider_state, self.constraint_state, self._solver.rigid_config
        )


@qd.kernel(fastcache=True)
def kernel_comfree_resolve(
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Single-pass analytical constraint resolution.

    The resolution needs no iteration, so it runs as a fixed sequence of fully data-parallel passes: the
    analytical per-row force, the J^T @ force gather, and the pre-factored mass solve. Each force is projected
    onto its row's admissible set: contact and joint-limit rows push only (the one-sided forces realize the
    friction pyramid facet-by-facet), frictionloss rows are box-bounded, and equality rows stay two-sided.
    With no constraint the mass solve reduces to qacc = M^-1 @ qf_smooth = acc_smooth.
    """
    n_dofs = dyn_state.dofs.acc.shape[0]
    _B = dyn_state.dofs.acc.shape[1]
    len_constraints = constraint_state.efc_force.shape[0]
    substep_dt = rigid_info.substep_dt[None]
    stiffness = rigid_info.comfree_stiffness[None] / substep_dt
    damping = rigid_info.comfree_damping[None] / substep_dt

    # Smooth predicted velocity, staged in the mv scratch (unused by ComFree) so the per-row pass reads it
    # instead of recomputing it per (row, dof) pair.
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        constraint_state.mv[i_d, i_b] = dyn_state.dofs.vel[i_d, i_b] + dyn_state.dofs.acc_smooth[i_d, i_b] * substep_dt

    # Analytical per-row force. Iteration order follows the physical layout (see constraint_layout_batch_first
    # in rigid_solver.py). With sparse_solve only the relevant-DOF entries of a Jacobian row are maintained, so
    # the row product must go through the sparse indices.
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_c, i_b in qd.ndrange(
        len_constraints, _B, axes=qd.static((1, 0) if rigid_config.constraint_layout_batch_first else None)
    ):
        if i_c < constraint_state.n_constraints[i_b]:
            efc_vel = gs.qd_float(0.0)
            if qd.static(rigid_config.sparse_solve):
                for i_d_ in range(constraint_state.jac_n_dofs[i_c, i_b]):
                    i_d = constraint_state.jac_dofs_idx[i_c, i_d_, i_b]
                    efc_vel = efc_vel + constraint_state.jac[i_c, i_d, i_b] * constraint_state.mv[i_d, i_b]
            else:
                for i_d in range(n_dofs):
                    efc_vel = efc_vel + constraint_state.jac[i_c, i_d, i_b] * constraint_state.mv[i_d, i_b]

            efc_acc = -damping * efc_vel - stiffness * (efc_vel * substep_dt + constraint_state.efc_dist[i_c, i_b])
            force = constraint_state.efc_mass[i_c, i_b] * efc_acc

            n_eq = constraint_state.n_constraints_equality[i_b]
            if i_c >= n_eq + constraint_state.n_constraints_frictionloss[i_b]:
                # Contact and joint-limit rows push only.
                force = qd.max(force, 0.0)
            elif i_c >= n_eq:
                # Frictionloss rows oppose motion up to their friction loss bound.
                frictionloss = constraint_state.efc_frictionloss[i_c, i_b]
                force = qd.math.clamp(force, -frictionloss, frictionloss)
            constraint_state.efc_force[i_c, i_b] = force

    # qfrc_constraint = J^T @ efc_force and total force. The dense gather is one thread per (dof, env) with the
    # batch as the coalesced axis; with sparse_solve (CPU only) each env instead scatters its rows through the
    # sparse indices sequentially, which keeps the accumulation deterministic without atomics.
    if qd.static(rigid_config.sparse_solve):
        qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(_B):
            for i_d in range(n_dofs):
                constraint_state.qfrc_constraint[i_d, i_b] = 0.0
            for i_c in range(constraint_state.n_constraints[i_b]):
                force = constraint_state.efc_force[i_c, i_b]
                for i_d_ in range(constraint_state.jac_n_dofs[i_c, i_b]):
                    i_d = constraint_state.jac_dofs_idx[i_c, i_d_, i_b]
                    constraint_state.qfrc_constraint[i_d, i_b] = (
                        constraint_state.qfrc_constraint[i_d, i_b] + constraint_state.jac[i_c, i_d, i_b] * force
                    )
            for i_d in range(n_dofs):
                dyn_state.dofs.force[i_d, i_b] = (
                    dyn_state.dofs.qf_smooth[i_d, i_b] + constraint_state.qfrc_constraint[i_d, i_b]
                )
    else:
        qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_d, i_b in qd.ndrange(n_dofs, _B):
            qfrc = gs.qd_float(0.0)
            for i_c in range(constraint_state.n_constraints[i_b]):
                qfrc = qfrc + constraint_state.jac[i_c, i_d, i_b] * constraint_state.efc_force[i_c, i_b]
            constraint_state.qfrc_constraint[i_d, i_b] = qfrc
            dyn_state.dofs.force[i_d, i_b] = dyn_state.dofs.qf_smooth[i_d, i_b] + qfrc

    # qacc = M^-1 @ force through the pre-factored mass matrix.
    func_solve_mass(dyn_state.dofs.force, constraint_state.qacc, dyn_info, rigid_info, rigid_config)
