"""Complementarity-free analytical contact solver.

Implements the ComFree-Sim approach (arXiv:2603.12185): closed-form contact
force computation via dual-cone impedance, replacing the iterative CG/Newton
solver for contact constraints.

Contact forces decouple across pairs, making the computation embarrassingly
parallel with O(n) scaling in contact count.
"""

from typing import TYPE_CHECKING

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.engine.solvers.rigid.abd import func_solve_mass_batch

from ..collider.contact_island import ContactIsland

if TYPE_CHECKING:
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver


class ComFreeSolver:
    def __init__(self, rigid_solver: "RigidSolver"):
        self._solver = rigid_solver
        self._collider = rigid_solver.collider
        self._B = rigid_solver._B

        self.constraint_state = rigid_solver.constraint_solver.constraint_state
        self.contact_island = ContactIsland(self._collider)

        self._k_user = rigid_solver._options.comfree_stiffness
        self._d_user = rigid_solver._options.comfree_damping

    def resolve(self):
        kernel_comfree_resolve(
            self._solver.links_info,
            self._solver.links_state,
            self._solver.dofs_state,
            self._solver.dofs_info,
            self._solver.entities_info,
            self._collider._collider_state,
            self.constraint_state,
            self._solver._rigid_global_info,
            self._solver._static_rigid_sim_config,
        )

        func_comfree_update_qacc(
            self._solver.dofs_state,
            self.constraint_state,
            self._solver._static_rigid_sim_config,
            self._solver._errno,
        )

        func_comfree_update_contact_force(
            self._solver.links_state,
            self._collider._collider_state,
            self._solver._static_rigid_sim_config,
        )


# =============================================================================
# Core ComFree kernel
# =============================================================================


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_comfree_resolve(
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Closed-form contact force computation.

    For each contact detected by narrowphase, computes contact forces using
    MuJoCo-style impedance (imp/aref/diag) but in a single closed-form
    evaluation instead of iterative optimization.

    Uses the same Jacobian, impedance, and pyramid construction as the
    existing solver (add_collision_constraints), but replaces the iterative
    CG/Newton solve with: efc_force = max(0, -(J·acc_smooth - aref) / diag).
    """
    EPS = rigid_global_info.EPS[None]
    n_dofs = dofs_state.vel.shape[0]
    _B = dofs_state.vel.shape[1]
    dt = rigid_global_info.substep_dt[None]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        for i_d in range(n_dofs):
            constraint_state.qfrc_constraint[i_d, i_b] = gs.qd_float(0.0)

        for i_col in range(collider_state.n_contacts[i_b]):
            contact_data_normal = collider_state.contact_data.normal[i_col, i_b]
            contact_data_pos = collider_state.contact_data.pos[i_col, i_b]
            contact_data_penetration = collider_state.contact_data.penetration[i_col, i_b]
            contact_data_friction = collider_state.contact_data.friction[i_col, i_b]
            contact_data_sol_params = collider_state.contact_data.sol_params[i_col, i_b]
            link_a = collider_state.contact_data.link_a[i_col, i_b]
            link_b = collider_state.contact_data.link_b[i_col, i_b]

            link_a_maybe_batch = (
                [link_a, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else link_a
            )
            link_b_maybe_batch = (
                [link_b, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else link_b
            )

            d1, d2 = gu.qd_orthogonals(contact_data_normal)

            invweight = links_info.invweight[link_a_maybe_batch][0]
            if link_b > -1:
                invweight = invweight + links_info.invweight[link_b_maybe_batch][0]

            contact_force_3d = qd.Vector.zero(gs.qd_float, 3)

            for i_dir in range(4):
                d = (2 * (i_dir % 2) - 1) * (d1 if i_dir < 2 else d2)
                n_tilde = d * contact_data_friction - contact_data_normal

                # Walk kinematic chain: compute J·vel (for impedance) and J·acc_smooth
                jac_qvel = gs.qd_float(0.0)
                jac_qacc_smooth = gs.qd_float(0.0)

                for i_ab in range(2):
                    sign = gs.qd_float(-1.0)
                    link = link_a
                    if i_ab == 1:
                        sign = gs.qd_float(1.0)
                        link = link_b

                    while link > -1:
                        link_maybe_batch = (
                            [link, i_b]
                            if qd.static(static_rigid_sim_config.batch_links_info)
                            else link
                        )

                        for i_d_ in range(links_info.n_dofs[link_maybe_batch]):
                            i_d = links_info.dof_end[link_maybe_batch] - 1 - i_d_

                            cdof_ang = dofs_state.cdof_ang[i_d, i_b]
                            cdot_vel = dofs_state.cdof_vel[i_d, i_b]

                            t_quat = gu.qd_identity_quat()
                            t_pos = contact_data_pos - links_state.root_COM[link, i_b]
                            _, vel = gu.qd_transform_motion_by_trans_quat(
                                cdof_ang, cdot_vel, t_pos, t_quat
                            )

                            diff = sign * vel
                            jac = diff @ n_tilde

                            jac_qvel = jac_qvel + jac * dofs_state.vel[i_d, i_b]
                            jac_qacc_smooth = (
                                jac_qacc_smooth + jac * dofs_state.acc_smooth[i_d, i_b]
                            )

                        link = links_info.parent_idx[link_maybe_batch]

                # MuJoCo-style impedance and reference acceleration
                imp, aref = gu.imp_aref(
                    contact_data_sol_params,
                    -contact_data_penetration,
                    jac_qvel,
                    -contact_data_penetration,
                )

                # Constraint-space inverse mass for this facet.
                # Scale by n_pyramid_facets (4) to account for all facets
                # coupling through the same body DOFs.
                mu2 = contact_data_friction * contact_data_friction
                JMinvJT = invweight * (1.0 + mu2) * 16.0
                JMinvJT = qd.max(JMinvJT, EPS)

                # Exact one-step force: set J·qacc = aref
                # f = (aref - J·acc_smooth) / (J M⁻¹ Jᵀ)
                Jaref = jac_qacc_smooth - aref
                efc_force = qd.max(gs.qd_float(0.0), -Jaref / JMinvJT)

                contact_force_3d = contact_force_3d + n_tilde * efc_force

                # Scatter Jᵀ · force into generalized constraint forces
                if efc_force > 0.0:
                    for i_ab in range(2):
                        sign = gs.qd_float(-1.0)
                        link = link_a
                        if i_ab == 1:
                            sign = gs.qd_float(1.0)
                            link = link_b

                        while link > -1:
                            link_maybe_batch = (
                                [link, i_b]
                                if qd.static(static_rigid_sim_config.batch_links_info)
                                else link
                            )

                            for i_d_ in range(links_info.n_dofs[link_maybe_batch]):
                                i_d = links_info.dof_end[link_maybe_batch] - 1 - i_d_

                                cdof_ang = dofs_state.cdof_ang[i_d, i_b]
                                cdot_vel = dofs_state.cdof_vel[i_d, i_b]

                                t_quat = gu.qd_identity_quat()
                                t_pos = contact_data_pos - links_state.root_COM[link, i_b]
                                _, vel = gu.qd_transform_motion_by_trans_quat(
                                    cdof_ang, cdot_vel, t_pos, t_quat
                                )

                                diff = sign * vel
                                jac = diff @ n_tilde

                                constraint_state.qfrc_constraint[i_d, i_b] = (
                                    constraint_state.qfrc_constraint[i_d, i_b]
                                    + jac * efc_force
                                )

                            link = links_info.parent_idx[link_maybe_batch]

            collider_state.contact_data.force[i_col, i_b] = contact_force_3d

        # Compute qacc = acc_smooth + M⁻¹ · qfrc_constraint
        func_solve_mass_batch(
            i_b,
            constraint_state.qfrc_constraint,
            constraint_state.qacc,
            array_class.PLACEHOLDER,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
            is_backward=False,
        )
        for i_d in range(n_dofs):
            constraint_state.qacc[i_d, i_b] = (
                dofs_state.acc_smooth[i_d, i_b] + constraint_state.qacc[i_d, i_b]
            )


@qd.kernel(fastcache=gs.use_fastcache)
def func_comfree_update_qacc(
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
    errno: array_class.V_ANNOTATION,
):
    """Copy ComFree results to dofs_state (same as func_update_qacc)."""
    n_dofs = dofs_state.acc.shape[0]
    _B = dofs_state.acc.shape[1]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        dofs_state.acc[i_d, i_b] = constraint_state.qacc[i_d, i_b]
        dofs_state.qf_constraint[i_d, i_b] = constraint_state.qfrc_constraint[i_d, i_b]
        dofs_state.force[i_d, i_b] = (
            dofs_state.qf_smooth[i_d, i_b] + constraint_state.qfrc_constraint[i_d, i_b]
        )
        if qd.math.isnan(constraint_state.qacc[i_d, i_b]):
            errno[i_b] = errno[i_b] | array_class.ErrorCode.INVALID_FORCE_NAN


@qd.kernel(fastcache=gs.use_fastcache)
def func_comfree_update_contact_force(
    links_state: array_class.LinksState,
    collider_state: array_class.ColliderState,
    static_rigid_sim_config: qd.template(),
):
    """Extract per-link 3D contact forces from ComFree results.

    Unlike the existing solver which reconstructs forces from efc_force,
    ComFree stores per-contact λ values directly during the resolve kernel.
    Here we recompute the 3D force from the contact data for reporting.
    """
    n_links = links_state.contact_force.shape[0]
    _B = links_state.contact_force.shape[1]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b in qd.ndrange(n_links, _B):
        links_state.contact_force[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)

    # Recompute per-contact 3D forces (same pyramid reconstruction)
    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        for i_c in range(collider_state.n_contacts[i_b]):
            force = collider_state.contact_data.force[i_c, i_b]
            link_a = collider_state.contact_data.link_a[i_c, i_b]
            link_b = collider_state.contact_data.link_b[i_c, i_b]

            links_state.contact_force[link_a, i_b] = (
                links_state.contact_force[link_a, i_b] - force
            )
            if link_b > -1:
                links_state.contact_force[link_b, i_b] = (
                    links_state.contact_force[link_b, i_b] + force
                )
