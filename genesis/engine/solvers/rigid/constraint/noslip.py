import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class

import genesis.engine.solvers.rigid.rigid_solver as rigid_solver


@qd.func
def _func_noslip_block_reduce_add(sh_warp, tid: qd.i32, val, static_rigid_sim_config: qd.template()):
    """Sum ``val`` across all ``noslip_coop_block_dim`` lanes of the env's block, returning the total to every lane.

    Hybrid reduction: a free warp-shuffle reduce within each 32-lane warp, then a shared-memory combine across the
    (few) warps. ``sh_warp`` is a per-block SharedArray of length ``noslip_coop_block_dim // 32``, reused across calls;
    the leading and trailing block syncs make back-to-back calls safe."""
    n_warps = qd.static(static_rigid_sim_config.noslip_coop_block_dim // 32)
    w = tid // 32
    lane = tid % 32
    warp_sum = qd.simt.subgroup.reduce_all_add_tiled(val, 5)
    if lane == 0:
        sh_warp[w] = warp_sum
    qd.simt.block.sync()
    total = gs.qd_float(0.0)
    for i in qd.static(range(n_warps)):
        total += sh_warp[i]
    qd.simt.block.sync()
    return total


@qd.func
def func_build_efc_AR_b_batch(
    i_b,
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    n_dofs = constraint_state.jac.shape[1]
    # On the fused serialized path, efc_AR/efc_b are allocated with a single batch slot shared by all envs.
    i_b_AR = 0 if qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL) else i_b
    nefc = constraint_state.n_constraints[i_b]

    # build AR = J * inv(M) * J^T
    # do it row-by-row: for each row r, tmp = inv(M) * J[r]^T, then AR[r,:] = J * tmp.
    # No zeroing pass is needed: the symmetric lower-triangle fill writes every entry of [0, nefc)^2 and
    # consumers never read beyond nefc.
    for i_row in range(nefc):
        # tmp = M^{-1} * Jr^T
        if qd.static(static_rigid_sim_config.sparse_solve):
            # Sparse: zero buffer, copy only relevant DOFs
            for i_d in range(n_dofs):
                constraint_state.Mgrad[i_d, i_b] = gs.qd_float(0.0)
            for i_d_ in range(constraint_state.jac_n_dofs[i_row, i_b]):
                i_d = constraint_state.jac_dofs_idx[i_row, i_d_, i_b]
                constraint_state.Mgrad[i_d, i_b] = constraint_state.jac[i_row, i_d, i_b]
        else:
            for i_d in range(n_dofs):
                constraint_state.Mgrad[i_d, i_b] = constraint_state.jac[i_row, i_d, i_b]

        rigid_solver.func_solve_mass_batch(
            i_b,
            constraint_state.Mgrad,
            constraint_state.Mgrad,
            None,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
            is_backward=False,
        )

        # TODO: For consistency with other usages, migrate to either the lower or upper variant
        # and update all remaining use cases that still read both.
        # AR[r, c] = J[c, :] * Mgrad, only compute lower triangle
        for i_col in range(i_row + 1):
            s = gs.qd_float(0.0)
            if qd.static(static_rigid_sim_config.sparse_solve):
                for i_d_ in range(constraint_state.jac_n_dofs[i_col, i_b]):
                    i_d = constraint_state.jac_dofs_idx[i_col, i_d_, i_b]
                    s += constraint_state.jac[i_col, i_d, i_b] * constraint_state.Mgrad[i_d, i_b]
            else:
                for i_d in range(n_dofs):
                    s += constraint_state.jac[i_col, i_d, i_b] * constraint_state.Mgrad[i_d, i_b]
            constraint_state.efc_AR[i_row, i_col, i_b_AR] = s
            constraint_state.efc_AR[i_col, i_row, i_b_AR] = s

    # Build efc_b
    for i_c in range(nefc):
        v = -constraint_state.aref[i_c, i_b]
        if qd.static(static_rigid_sim_config.sparse_solve):
            for i_d_ in range(constraint_state.jac_n_dofs[i_c, i_b]):
                i_d = constraint_state.jac_dofs_idx[i_c, i_d_, i_b]
                v += constraint_state.jac[i_c, i_d, i_b] * dofs_state.acc_smooth[i_d, i_b]
        else:
            for i_d in range(n_dofs):
                v += constraint_state.jac[i_c, i_d, i_b] * dofs_state.acc_smooth[i_d, i_b]
        constraint_state.efc_b[i_c, i_b_AR] = v


@qd.func
def func_solve_mass_entity_row(
    i_row: qd.int32,
    i_e: qd.int32,
    i_b: qd.int32,
    buf: qd.Tensor,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
):
    """LDL^T forward-backward substitution on buf[i_row, :, i_b].

    Same algorithm as func_solve_mass_entity (forward-only path), but operates
    on a 3D buffer indexed by (constraint_row, dof, batch). This allows
    different constraint rows to be solved in parallel since each row uses
    a separate memory slice.
    """
    if rigid_global_info.mass_mat_mask[i_e, i_b]:
        entity_dof_start = entities_info.dof_start[i_e]
        entity_dof_end = entities_info.dof_end[i_e]
        n_dofs = entities_info.n_dofs[i_e]

        # Step 1: Solve w s.t. L^T @ w = y (backward substitution)
        for i_d_ in range(n_dofs):
            i_d = entity_dof_end - i_d_ - 1
            curr_out = buf[i_row, i_d, i_b]
            for j_d in range(i_d + 1, entity_dof_end):
                curr_out = curr_out - rigid_global_info.mass_mat_L[j_d, i_d, i_b] * buf[i_row, j_d, i_b]
            buf[i_row, i_d, i_b] = curr_out

        # Step 2: z = D^{-1} @ w
        for i_d in range(entity_dof_start, entity_dof_end):
            buf[i_row, i_d, i_b] = buf[i_row, i_d, i_b] * rigid_global_info.mass_mat_D_inv[i_d, i_b]

        # Step 3: Solve x s.t. L @ x = z (forward substitution)
        for i_d in range(entity_dof_start, entity_dof_end):
            curr_out = buf[i_row, i_d, i_b]
            for j_d in range(entity_dof_start, i_d):
                curr_out = curr_out - rigid_global_info.mass_mat_L[i_d, j_d, i_b] * buf[i_row, j_d, i_b]
            buf[i_row, i_d, i_b] = curr_out


@qd.func
def func_noslip_batch(
    i_b,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    EPS = rigid_global_info.EPS[None]
    n_dofs = constraint_state.jac.shape[1]
    # On the fused serialized path, efc_AR/efc_b are allocated with a single batch slot shared by all envs.
    i_b_AR = 0 if qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL) else i_b

    # temp variables
    res = qd.Vector.zero(gs.qd_float, 5)
    old_force = qd.Vector.zero(gs.qd_float, 5)
    bc = qd.Vector.zero(gs.qd_float, 5)
    Ac = qd.Vector.zero(gs.qd_float, 9)

    n_con = collider_state.n_contacts[i_b]
    ne = constraint_state.n_constraints_equality[i_b]
    nf = constraint_state.n_constraints_frictionloss[i_b]
    const_start = ne + nf

    scale = 1.0 / (rigid_global_info.meaninertia[i_b] * qd.max(1.0, n_dofs))

    for i_iter in range(rigid_global_info.noslip_iterations[None]):
        improvement = gs.qd_float(0.0)
        if i_iter == 0:
            for i_c in range(constraint_state.n_constraints[i_b]):
                improvement += 0.5 * constraint_state.efc_force[i_c, i_b] ** 2 * constraint_state.diag[i_c, i_b]

        for i_c in range(ne, ne + nf):
            res = func_residual_constraint_force(
                res=res,
                i_b=i_b,
                i_efc=i_c,
                dim=1,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )
            old_force[0] = constraint_state.efc_force[i_c, i_b]
            constraint_state.efc_force[i_c, i_b] -= res[0] / constraint_state.efc_AR[i_c, i_c, i_b_AR]
            if constraint_state.efc_force[i_c, i_b] < -constraint_state.efc_frictionloss[i_c, i_b]:
                constraint_state.efc_force[i_c, i_b] = -constraint_state.efc_frictionloss[i_c, i_b]
            elif constraint_state.efc_force[i_c, i_b] > constraint_state.efc_frictionloss[i_c, i_b]:
                constraint_state.efc_force[i_c, i_b] = constraint_state.efc_frictionloss[i_c, i_b]
            delta = constraint_state.efc_force[i_c, i_b] - old_force[0]
            improvement -= 0.5 * delta**2 / constraint_state.efc_AR[i_c, i_c, i_b_AR] + delta * res[0]

        # Project contact friction (pyramidal 4-edge) with normal fixed
        for i_col in range(n_con):
            base = const_start + i_col * 4
            for j2 in qd.static(range(2)):
                j_efc = base + j2 * 2
                res = func_residual_constraint_force(
                    res=res,
                    i_b=i_b,
                    i_efc=j_efc,
                    dim=2,
                    constraint_state=constraint_state,
                    static_rigid_sim_config=static_rigid_sim_config,
                )
                for i2 in qd.static(range(2)):
                    old_force[i2] = constraint_state.efc_force[j_efc + i2, i_b]
                Ac = func_extract_block_matrix_from_AR(
                    Ac=Ac,
                    i_b=i_b,
                    start=j_efc,
                    n=2,
                    constraint_state=constraint_state,
                    static_rigid_sim_config=static_rigid_sim_config,
                )
                for j in qd.static(range(2)):
                    bc[j] = res[j]
                    for k in qd.static(range(2)):
                        bc[j] -= Ac[j * 2 + k] * old_force[k]
                mid = 0.5 * (constraint_state.efc_force[j_efc, i_b] + constraint_state.efc_force[j_efc + 1, i_b])
                y = 0.5 * (constraint_state.efc_force[j_efc, i_b] - constraint_state.efc_force[j_efc + 1, i_b])
                K1 = Ac[0] + Ac[3] - Ac[1] - Ac[2]
                K0 = mid * (Ac[0] - Ac[3]) + bc[0] - bc[1]
                if K1 < EPS:
                    constraint_state.efc_force[j_efc, i_b] = constraint_state.efc_force[j_efc + 1, i_b] = mid
                else:
                    y = -K0 / K1
                    if y < -mid:
                        constraint_state.efc_force[j_efc, i_b] = 0
                        constraint_state.efc_force[j_efc + 1, i_b] = 2 * mid
                    elif y > mid:
                        constraint_state.efc_force[j_efc, i_b] = 2 * mid
                        constraint_state.efc_force[j_efc + 1, i_b] = 0
                    else:
                        constraint_state.efc_force[j_efc, i_b] = mid + y
                        constraint_state.efc_force[j_efc + 1, i_b] = mid - y
                cost_change = func_cost_change(
                    i_b=i_b,
                    Ac=Ac,
                    force=constraint_state.efc_force,
                    force_start=j_efc,
                    old_force=old_force,
                    res=res,
                    dim=2,
                    eps=EPS,
                )

                improvement -= cost_change
        improvement *= scale

        if improvement < rigid_global_info.noslip_tolerance[None]:
            break


@qd.func
def func_noslip_batch_coop(
    tid: qd.i32,
    i_b,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Block-per-env cooperative version of ``func_noslip_batch`` (approach A in ``perso_hugh/doc/noslip.md``).

    All ``noslip_coop_block_dim`` lanes of the env's block run identical control flow; only the two reductions are split
    across lanes: the dominant per-constraint residual dot product (via ``func_residual_constraint_force_coop``) and the
    iter-0 improvement sum. The projected Gauss-Seidel order is unchanged - every lane recomputes the scalar projection
    and writes ``efc_force`` redundantly with the identical value, so each lane reads back its own writes in program
    order and no cross-lane fence is needed. Results match the serial sweep up to reduction-associativity rounding."""
    EPS = rigid_global_info.EPS[None]
    n_dofs = constraint_state.jac.shape[1]
    i_b_AR = 0 if qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL) else i_b
    _BLOCK = qd.static(static_rigid_sim_config.noslip_coop_block_dim)

    # Per-block scratch for the cross-warp part of the residual / improvement reductions.
    sh_warp = qd.simt.block.SharedArray((static_rigid_sim_config.noslip_coop_block_dim // 32,), gs.qd_float)

    # temp variables
    res = qd.Vector.zero(gs.qd_float, 5)
    old_force = qd.Vector.zero(gs.qd_float, 5)
    bc = qd.Vector.zero(gs.qd_float, 5)
    Ac = qd.Vector.zero(gs.qd_float, 9)

    n_con = collider_state.n_contacts[i_b]
    ne = constraint_state.n_constraints_equality[i_b]
    nf = constraint_state.n_constraints_frictionloss[i_b]
    const_start = ne + nf

    scale = 1.0 / (rigid_global_info.meaninertia[i_b] * qd.max(1.0, n_dofs))

    for i_iter in range(rigid_global_info.noslip_iterations[None]):
        improvement = gs.qd_float(0.0)
        if i_iter == 0:
            partial = gs.qd_float(0.0)
            i_c = tid
            while i_c < constraint_state.n_constraints[i_b]:
                partial += 0.5 * constraint_state.efc_force[i_c, i_b] ** 2 * constraint_state.diag[i_c, i_b]
                i_c += _BLOCK
            improvement += _func_noslip_block_reduce_add(sh_warp, tid, partial, static_rigid_sim_config)

        for i_c in range(ne, ne + nf):
            res = func_residual_constraint_force_coop(
                res=res,
                tid=tid,
                i_b=i_b,
                i_efc=i_c,
                dim=1,
                sh_warp=sh_warp,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )
            old_force[0] = constraint_state.efc_force[i_c, i_b]
            constraint_state.efc_force[i_c, i_b] -= res[0] / constraint_state.efc_AR[i_c, i_c, i_b_AR]
            if constraint_state.efc_force[i_c, i_b] < -constraint_state.efc_frictionloss[i_c, i_b]:
                constraint_state.efc_force[i_c, i_b] = -constraint_state.efc_frictionloss[i_c, i_b]
            elif constraint_state.efc_force[i_c, i_b] > constraint_state.efc_frictionloss[i_c, i_b]:
                constraint_state.efc_force[i_c, i_b] = constraint_state.efc_frictionloss[i_c, i_b]
            delta = constraint_state.efc_force[i_c, i_b] - old_force[0]
            improvement -= 0.5 * delta**2 / constraint_state.efc_AR[i_c, i_c, i_b_AR] + delta * res[0]

        # Project contact friction (pyramidal 4-edge) with normal fixed
        for i_col in range(n_con):
            base = const_start + i_col * 4
            for j2 in qd.static(range(2)):
                j_efc = base + j2 * 2
                res = func_residual_constraint_force_coop(
                    res=res,
                    tid=tid,
                    i_b=i_b,
                    i_efc=j_efc,
                    dim=2,
                    sh_warp=sh_warp,
                    constraint_state=constraint_state,
                    static_rigid_sim_config=static_rigid_sim_config,
                )
                for i2 in qd.static(range(2)):
                    old_force[i2] = constraint_state.efc_force[j_efc + i2, i_b]
                Ac = func_extract_block_matrix_from_AR(
                    Ac=Ac,
                    i_b=i_b,
                    start=j_efc,
                    n=2,
                    constraint_state=constraint_state,
                    static_rigid_sim_config=static_rigid_sim_config,
                )
                for j in qd.static(range(2)):
                    bc[j] = res[j]
                    for k in qd.static(range(2)):
                        bc[j] -= Ac[j * 2 + k] * old_force[k]
                mid = 0.5 * (constraint_state.efc_force[j_efc, i_b] + constraint_state.efc_force[j_efc + 1, i_b])
                y = 0.5 * (constraint_state.efc_force[j_efc, i_b] - constraint_state.efc_force[j_efc + 1, i_b])
                K1 = Ac[0] + Ac[3] - Ac[1] - Ac[2]
                K0 = mid * (Ac[0] - Ac[3]) + bc[0] - bc[1]
                if K1 < EPS:
                    constraint_state.efc_force[j_efc, i_b] = constraint_state.efc_force[j_efc + 1, i_b] = mid
                else:
                    y = -K0 / K1
                    if y < -mid:
                        constraint_state.efc_force[j_efc, i_b] = 0
                        constraint_state.efc_force[j_efc + 1, i_b] = 2 * mid
                    elif y > mid:
                        constraint_state.efc_force[j_efc, i_b] = 2 * mid
                        constraint_state.efc_force[j_efc + 1, i_b] = 0
                    else:
                        constraint_state.efc_force[j_efc, i_b] = mid + y
                        constraint_state.efc_force[j_efc + 1, i_b] = mid - y
                cost_change = func_cost_change(
                    i_b=i_b,
                    Ac=Ac,
                    force=constraint_state.efc_force,
                    force_start=j_efc,
                    old_force=old_force,
                    res=res,
                    dim=2,
                    eps=EPS,
                )

                improvement -= cost_change
        improvement *= scale

        if improvement < rigid_global_info.noslip_tolerance[None]:
            break


@qd.func
def func_dual_finish_batch(
    i_b,
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    n_dofs = constraint_state.qfrc_constraint.shape[0]

    # zero
    for i_d in range(n_dofs):
        constraint_state.qfrc_constraint[i_d, i_b] = gs.qd_float(0.0)

        for i_c in range(constraint_state.n_constraints[i_b]):
            constraint_state.qfrc_constraint[i_d, i_b] = (
                constraint_state.qfrc_constraint[i_d, i_b]
                + constraint_state.jac[i_c, i_d, i_b] * constraint_state.efc_force[i_c, i_b]
            )

    rigid_solver.func_solve_mass_batch(
        i_b=i_b,
        vec=constraint_state.qfrc_constraint,
        out=constraint_state.qacc,
        out_bw=None,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        is_backward=False,
    )

    for i_d in range(n_dofs):
        constraint_state.qacc[i_d, i_b] = constraint_state.qacc[i_d, i_b] + dofs_state.acc_smooth[i_d, i_b]
        dofs_state.acc[i_d, i_b] = constraint_state.qacc[i_d, i_b]

        dofs_state.qf_constraint[i_d, i_b] = constraint_state.qfrc_constraint[i_d, i_b]
        dofs_state.force[i_d, i_b] = dofs_state.qf_smooth[i_d, i_b] + constraint_state.qfrc_constraint[i_d, i_b]


@qd.func
def func_dual_finish_batch_coop(
    tid: qd.i32,
    i_b,
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """Warp-per-env cooperative version of ``func_dual_finish_batch`` (approach C in ``perso_hugh/doc/noslip.md``).

    The 32 lanes split the per-dof work: the ``qfrc = J^T f`` accumulation and the final acc/force write-back are
    lane-strided over dofs (the dominant O(n_dofs * nefc) term). The block-diagonal mass solve runs redundantly on all
    lanes - on a GPU a warp executes its lanes in lockstep, so identical work across 32 lanes costs the same wall time
    as one lane, while every lane reading back its own qacc writes keeps it correct without an extra fence."""
    n_dofs = constraint_state.qfrc_constraint.shape[0]
    n_con = constraint_state.n_constraints[i_b]
    _BLOCK = qd.static(static_rigid_sim_config.noslip_coop_block_dim)

    # qfrc_constraint = J^T @ efc_force (lane-strided over dofs)
    i_d = tid
    while i_d < n_dofs:
        qfrc = gs.qd_float(0.0)
        for i_c in range(n_con):
            qfrc += constraint_state.jac[i_c, i_d, i_b] * constraint_state.efc_force[i_c, i_b]
        constraint_state.qfrc_constraint[i_d, i_b] = qfrc
        i_d += _BLOCK

    # All lanes must see every qfrc entry before the mass solve reads the full vector.
    qd.simt.block.sync()

    # The mass solve is an in-place block-diagonal LDL on qacc; running it redundantly is only safe within a single
    # warp (32 lanes in lockstep read/write identical values). With _BLOCK > 32 the extra warps are not lockstep with
    # warp 0, so confine the solve to the first warp and fence its result to the rest of the block.
    if tid < 32:
        rigid_solver.func_solve_mass_batch(
            i_b=i_b,
            vec=constraint_state.qfrc_constraint,
            out=constraint_state.qacc,
            out_bw=None,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
            is_backward=False,
        )

    qd.simt.block.sync()

    i_d = tid
    while i_d < n_dofs:
        constraint_state.qacc[i_d, i_b] = constraint_state.qacc[i_d, i_b] + dofs_state.acc_smooth[i_d, i_b]
        dofs_state.acc[i_d, i_b] = constraint_state.qacc[i_d, i_b]
        dofs_state.qf_constraint[i_d, i_b] = constraint_state.qfrc_constraint[i_d, i_b]
        dofs_state.force[i_d, i_b] = dofs_state.qf_smooth[i_d, i_b] + constraint_state.qfrc_constraint[i_d, i_b]
        i_d += _BLOCK


@qd.func
def func_noslip_sweep_serial(
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """1-thread-per-env force-update sweep (the legacy serial path)."""
    _B = constraint_state.jac.shape[2]
    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        func_noslip_batch(i_b, collider_state, constraint_state, rigid_global_info, static_rigid_sim_config)


@qd.func
def func_noslip_sweep_coop(
    collider_state: array_class.ColliderState,
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Block-per-env force-update sweep: a ``noslip_coop_block_dim``-lane block per env, cooperating on the residual
    reductions. A wider-than-warp block hides the dense efc_AR read latency (the bs=1 bottleneck) by keeping more
    loads in flight. With ``noslip_component_parallel`` set, the block instead splits the constraint graph's independent
    components across its warps (E47)."""
    _BLOCK = qd.static(static_rigid_sim_config.noslip_coop_block_dim)
    _B = constraint_state.jac.shape[2]
    if qd.static(static_rigid_sim_config.noslip_component_parallel):
        # Block-per-(env, component): launch noslip_max_components blocks per env, each owning one independent
        # constraint-graph component (trailing blocks beyond the env's live component count return immediately).
        _MAXC = qd.static(static_rigid_sim_config.noslip_max_components)
        qd.loop_config(name="noslip_sweep", block_dim=_BLOCK)
        for i_flat in range(_B * _MAXC * _BLOCK):
            tid = i_flat % _BLOCK
            blk = i_flat // _BLOCK
            i_b = blk // _MAXC
            comp = blk % _MAXC
            func_noslip_batch_comp_block(
                tid, i_b, comp, collider_state, entities_info, constraint_state, rigid_global_info,
                static_rigid_sim_config,
            )
    else:
        qd.loop_config(name="noslip_sweep", block_dim=_BLOCK)
        for i_flat in range(_B * _BLOCK):
            tid = i_flat % _BLOCK
            i_b = i_flat // _BLOCK
            func_noslip_batch_coop(
                tid, i_b, collider_state, constraint_state, rigid_global_info, static_rigid_sim_config
            )


@qd.func
def func_noslip_sweep(
    collider_state: array_class.ColliderState,
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Force-update sweep, dispatched at compile time on ``enable_cooperative_constraint_kernels`` to the warp-per-env
    cooperative variant (small n_envs, where one thread per env starves the GPU) or the 1-thread-per-env serial one."""
    if qd.static(static_rigid_sim_config.enable_cooperative_constraint_kernels):
        func_noslip_sweep_coop(
            collider_state, entities_info, constraint_state, rigid_global_info, static_rigid_sim_config
        )
    else:
        func_noslip_sweep_serial(collider_state, constraint_state, rigid_global_info, static_rigid_sim_config)


@qd.func
def func_dual_finish_serial(
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """1-thread-per-env dual finish (the legacy serial path)."""
    _B = constraint_state.jac.shape[2]
    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        func_dual_finish_batch(
            i_b, dofs_state, entities_info, rigid_global_info, constraint_state, static_rigid_sim_config
        )


@qd.func
def func_dual_finish_coop(
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """Warp-per-env dual finish: a 32-lane block per env, cooperating on the per-dof J^T f and write-back."""
    _B = constraint_state.jac.shape[2]
    _BLOCK = qd.static(static_rigid_sim_config.noslip_coop_block_dim)
    qd.loop_config(name="noslip_dual_finish", block_dim=_BLOCK)
    for i_flat in range(_B * _BLOCK):
        tid = i_flat % _BLOCK
        i_b = i_flat // _BLOCK
        func_dual_finish_batch_coop(
            tid, i_b, dofs_state, entities_info, rigid_global_info, constraint_state, static_rigid_sim_config
        )


@qd.func
def func_dual_finish(
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """Dual finish (qfrc = J^T f, mass solve, acc/force write-back), dispatched at compile time on
    ``enable_cooperative_constraint_kernels`` to the warp-per-env cooperative or 1-thread-per-env serial variant."""
    if qd.static(static_rigid_sim_config.enable_cooperative_constraint_kernels):
        func_dual_finish_coop(dofs_state, entities_info, rigid_global_info, constraint_state, static_rigid_sim_config)
    else:
        func_dual_finish_serial(dofs_state, entities_info, rigid_global_info, constraint_state, static_rigid_sim_config)


@qd.kernel(fastcache=True)
def kernel_noslip_fused(
    collider_state: array_class.ColliderState,
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """Serialized noslip pass: build AR/b, run the force-update sweep, and finish, one env at a time.

    Processing each env end-to-end keeps its efc_AR block (written by the build, consumed by the sweep) cache-hot, and
    allows efc_AR/efc_b to be allocated with a single batch slot shared by all envs.
    """
    _B = constraint_state.jac.shape[2]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        func_build_efc_AR_b_batch(
            i_b, dofs_state, entities_info, rigid_global_info, constraint_state, static_rigid_sim_config
        )
        func_noslip_batch(i_b, collider_state, constraint_state, rigid_global_info, static_rigid_sim_config)
        func_dual_finish_batch(
            i_b, dofs_state, entities_info, rigid_global_info, constraint_state, static_rigid_sim_config
        )


@qd.kernel(fastcache=True)
def kernel_noslip_decomposed(
    collider_state: array_class.ColliderState,
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """Decomposed noslip pass: parallel MinvJT solve, parallel AR/b build, force-update sweep, and dual finish.

    Each top-level loop is an independent offloaded task with its own launch shape, with implicit barriers in
    between. The MinvJT solve runs one thread per (row, env): each thread copies J[row] into its own MinvJT row and
    solves M^{-1} in place via the row-indexed LDL^T substitution, with no shared buffers between rows. The AR build
    runs one thread per (row, col, env) - nefc^2 * n_envs independent threads (~490K for typical scenes) - computing
    AR[row, col, i_b] = sum_d J[col, d, i_b] * MinvJT[row, d, i_b]. On the serialized path, kernel_noslip_fused is
    used instead.
    """
    len_c = constraint_state.MinvJT.shape[0]
    _B = constraint_state.jac.shape[2]
    n_dofs = constraint_state.jac.shape[1]

    for i_row, i_b in qd.ndrange(len_c, _B):
        if i_row < constraint_state.n_constraints[i_b]:
            # Copy J[row] into MinvJT[row] (per-row buffer)
            for i_d in range(n_dofs):
                constraint_state.MinvJT[i_row, i_d, i_b] = constraint_state.jac[i_row, i_d, i_b]

            # In-place solve: MinvJT[row] = M^{-1} @ J[row]
            for i_0 in (
                range(rigid_global_info.n_awake_entities[i_b])
                if qd.static(static_rigid_sim_config.use_hibernation)
                else range(entities_info.n_links.shape[0])
            ):
                i_e = (
                    rigid_global_info.awake_entities[i_0, i_b]
                    if qd.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )
                # A contact row's Jacobian only touches the 1-2 bodies in the contact, so most entity blocks have an
                # all-zero RHS. M^-1 @ 0 = 0 and MinvJT already holds 0 there, so solving those blocks is pure waste
                # (the dense robot block is the worst offender). Skip them - bit-identical, just fewer LDL solves.
                row_touches_e = False
                for i_d in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                    if constraint_state.MinvJT[i_row, i_d, i_b] != 0.0:
                        row_touches_e = True
                if row_touches_e:
                    func_solve_mass_entity_row(
                        i_row, i_e, i_b, constraint_state.MinvJT, entities_info, rigid_global_info
                    )

    for i_row, i_col, i_b in qd.ndrange(len_c, len_c, _B):
        nefc = constraint_state.n_constraints[i_b]
        if i_row < nefc and i_col < nefc:
            s = gs.qd_float(0.0)
            for i_d in range(n_dofs):
                s += constraint_state.jac[i_col, i_d, i_b] * constraint_state.MinvJT[i_row, i_d, i_b]
            constraint_state.efc_AR[i_row, i_col, i_b] = s
        else:
            constraint_state.efc_AR[i_row, i_col, i_b] = gs.qd_float(0.0)

    # Build efc_b
    for i_c, i_b in qd.ndrange(len_c, _B):
        if i_c < constraint_state.n_constraints[i_b]:
            v = -constraint_state.aref[i_c, i_b]
            for i_d in range(n_dofs):
                v += constraint_state.jac[i_c, i_d, i_b] * dofs_state.acc_smooth[i_d, i_b]
            constraint_state.efc_b[i_c, i_b] = v

    func_noslip_sweep(collider_state, entities_info, constraint_state, rigid_global_info, static_rigid_sim_config)

    func_dual_finish(dofs_state, entities_info, rigid_global_info, constraint_state, static_rigid_sim_config)


@qd.func
def func_extract_block_matrix_from_AR(
    Ac,
    i_b: int,
    start: int,
    n: int,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    # On the fused serialized path, efc_AR/efc_b are allocated with a single batch slot shared by all envs.
    i_b_AR = 0 if qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL) else i_b
    for j in range(n):
        for k in range(n):
            Ac[j * n + k] = constraint_state.efc_AR[start + j, start + k, i_b_AR]
    return Ac


@qd.func
def func_residual_constraint_force(
    res,
    i_b: int,
    i_efc: int,
    dim: int,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    # On the fused serialized path, efc_AR/efc_b are allocated with a single batch slot shared by all envs.
    i_b_AR = 0 if qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL) else i_b
    for j in range(dim):
        res[j] = constraint_state.efc_b[i_efc + j, i_b_AR]
        for k in range(constraint_state.n_constraints[i_b]):
            res[j] += constraint_state.efc_AR[i_efc + j, k, i_b_AR] * constraint_state.efc_force[k, i_b]
    return res


@qd.func
def func_residual_constraint_force_coop(
    res,
    tid: qd.i32,
    i_b: int,
    i_efc: int,
    dim: int,
    sh_warp,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """Block-cooperative variant of ``func_residual_constraint_force``: the ``noslip_coop_block_dim`` lanes of the env's
    block stride over the constraint index ``k`` of the residual dot product ``res = b + AR @ f`` and the per-lane
    partial sums are combined with ``_func_noslip_block_reduce_add``, so every lane ends up with the full residual.
    ``dim`` is uniform across the block (1 for frictionloss rows, 2 for a pyramidal pair), so all lanes call the
    reduction the same number of times."""
    i_b_AR = 0 if qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL) else i_b
    _BLOCK = qd.static(static_rigid_sim_config.noslip_coop_block_dim)
    n = constraint_state.n_constraints[i_b]
    for j in range(dim):
        partial = gs.qd_float(0.0)
        k = tid
        while k < n:
            partial += constraint_state.efc_AR[i_efc + j, k, i_b_AR] * constraint_state.efc_force[k, i_b]
            k += _BLOCK
        res[j] = constraint_state.efc_b[i_efc + j, i_b_AR] + _func_noslip_block_reduce_add(
            sh_warp, tid, partial, static_rigid_sim_config
        )
    return res


@qd.func
def _func_residual_constraint_force_warp_unused(
    res,
    lane: qd.i32,
    i_b: int,
    i_efc: int,
    dim: int,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """DEAD CODE (kept for reference). Warp-scoped residual reduction tried for the component-per-warp sweep (E47).
    Abandoned: warp-shuffle ``reduce_all_add_tiled`` proved fragile in this kernel context (gross ~4e2 force error vs
    the serial sweep, despite passing isolated microbenchmarks - likely a warp-reconvergence interaction after the
    per-lane variable-trip stride loop). Replaced by the block-per-component design, which reuses the proven block-scope
    ``func_residual_constraint_force_coop`` (``block.reduce_all_add``). Not referenced by any kernel."""
    i_b_AR = 0 if qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL) else i_b
    n = constraint_state.n_constraints[i_b]
    for j in range(dim):
        partial = gs.qd_float(0.0)
        k = lane
        while k < n:
            partial += constraint_state.efc_AR[i_efc + j, k, i_b_AR] * constraint_state.efc_force[k, i_b]
            k += 32
        # Reconverge the warp after the (per-lane variable trip-count) stride loop: the shuffle-based reduce assumes a
        # fully-active 32-lane warp (__shfl_sync(0xFFFFFFFF)), and Volta+ independent thread scheduling does not
        # guarantee reconvergence after a divergent loop without an explicit warp sync.
        qd.simt.subgroup.sync()
        warp_sum = qd.simt.subgroup.reduce_all_add_tiled(partial, 5)
        res[j] = constraint_state.efc_b[i_efc + j, i_b_AR] + warp_sum
    return res


@qd.func
def func_noslip_batch_comp_block(
    tid: qd.i32,
    i_b,
    comp: qd.i32,
    collider_state: array_class.ColliderState,
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Block-per-component cooperative no-slip sweep (E47).

    The contact-friction constraint graph splits into independent connected components (two contacts couple only if they
    share a moving entity -> their ``efc_AR`` cross-block is nonzero; the inverse mass matrix is block-diagonal per
    entity). One block is launched per (env, component): block ``comp`` runs the projected Gauss-Seidel sub-sweep over
    exactly the contacts whose component label == ``comp``, reusing the proven block-cooperative residual reduction
    (``func_residual_constraint_force_coop``, i.e. the block-scope ``reduce_all_add``). Components are disjoint so blocks
    never write each other's forces, and convergence is per-component (exact, since components do not interact).

    For bs=1 this turns the single-block-per-env sweep (1 SM busy, ``nefc``-deep serial GS) into ~``n_components`` blocks
    (more SMs busy, only ``max_component``-deep) -- the win on small env counts.

    Each block redundantly recomputes the (cheap) component labels in shared memory via an entity union-find, then
    selects its own component. Equality/frictionloss rows or a component count exceeding ``noslip_max_components`` fall
    back to the dense cooperative sweep, run by component-block 0 only. Every branch here is block-uniform (``comp``,
    ``n_comp`` and the per-contact label are identical across the block's threads), so the block syncs inside the
    residual reduction are always reached by all threads.
    """
    EPS = rigid_global_info.EPS[None]
    n_dofs = constraint_state.jac.shape[1]
    _BLOCK = qd.static(static_rigid_sim_config.noslip_coop_block_dim)
    _NCON_MAX = qd.static(static_rigid_sim_config.noslip_comp_max_contacts)
    _NENT = qd.static(static_rigid_sim_config.n_entities)
    _MAXC = qd.static(static_rigid_sim_config.noslip_max_components)

    n_con = collider_state.n_contacts[i_b]
    ne = constraint_state.n_constraints_equality[i_b]
    nf = constraint_state.n_constraints_frictionloss[i_b]
    const_start = ne + nf

    if ne + nf > 0 or n_con > _NCON_MAX:
        # Equality / frictionloss rows (not covered by the entity-based contact grouping) or a contact count exceeding
        # the shared-memory scratch cap: component-block 0 runs the dense cooperative sweep (correct, not parallel).
        if comp == 0:
            func_noslip_batch_coop(
                tid, i_b, collider_state, constraint_state, rigid_global_info, static_rigid_sim_config
            )
    else:
        # codegen allows only one *large* shared array per kernel, so the per-contact / per-entity int scratch is packed
        # into a single int SharedArray with manual sub-region offsets. Regions: e0=[0,N), e1=[N,2N), label=[2N,3N)
        # (N=_NCON_MAX); parent=[3N,3N+E), root_id=[3N+E,3N+2E) (E=_NENT).
        _E1 = qd.static(_NCON_MAX)
        _LAB = qd.static(2 * _NCON_MAX)
        _PAR = qd.static(3 * _NCON_MAX)
        _ROOT = qd.static(3 * _NCON_MAX + _NENT)
        sh_warp = qd.simt.block.SharedArray((static_rigid_sim_config.noslip_coop_block_dim // 32,), gs.qd_float)
        sh_i = qd.simt.block.SharedArray((3 * _NCON_MAX + 2 * _NENT,), gs.qd_int)
        sh_ncomp = qd.simt.block.SharedArray((1,), gs.qd_int)

        # --- Phase 0a (cooperative): per-contact moving entities via the jac row dof support. ---
        i_c = tid
        while i_c < n_con:
            r = const_start + i_c * 4
            e0 = qd.i32(-1)
            e1 = qd.i32(-1)
            for i_e in range(_NENT):
                if rigid_global_info.mass_mat_mask[i_e, i_b]:
                    touch = False
                    for i_d in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                        if constraint_state.jac[r, i_d, i_b] != 0.0:
                            touch = True
                    if touch:
                        if e0 < 0:
                            e0 = i_e
                        else:
                            e1 = i_e
            sh_i[i_c] = e0
            sh_i[_E1 + i_c] = e1
            i_c += _BLOCK
        qd.simt.block.sync()

        # --- Phase 0b (thread 0): union-find over entities, then dense component label per contact. ---
        if tid == 0:
            for i_e in range(_NENT):
                sh_i[_PAR + i_e] = i_e
                sh_i[_ROOT + i_e] = -1
            # Union the two moving entities of each contact.
            for i_cc in range(n_con):
                a = sh_i[i_cc]
                b = sh_i[_E1 + i_cc]
                if a >= 0 and b >= 0:
                    # find(a)
                    ra = a
                    while sh_i[_PAR + ra] != ra:
                        sh_i[_PAR + ra] = sh_i[_PAR + sh_i[_PAR + ra]]
                        ra = sh_i[_PAR + ra]
                    rb = b
                    while sh_i[_PAR + rb] != rb:
                        sh_i[_PAR + rb] = sh_i[_PAR + sh_i[_PAR + rb]]
                        rb = sh_i[_PAR + rb]
                    if ra != rb:
                        sh_i[_PAR + ra] = rb
            n_lab = qd.i32(0)
            for i_cc in range(n_con):
                a = sh_i[i_cc]
                if a < 0:
                    a = sh_i[_E1 + i_cc]
                if a < 0:
                    # Contact with no moving entity (fixed-vs-fixed); give it its own singleton component.
                    sh_i[_LAB + i_cc] = n_lab
                    n_lab += 1
                else:
                    ra = a
                    while sh_i[_PAR + ra] != ra:
                        sh_i[_PAR + ra] = sh_i[_PAR + sh_i[_PAR + ra]]
                        ra = sh_i[_PAR + ra]
                    if sh_i[_ROOT + ra] < 0:
                        sh_i[_ROOT + ra] = n_lab
                        n_lab += 1
                    sh_i[_LAB + i_cc] = sh_i[_ROOT + ra]
            sh_ncomp[0] = n_lab
        qd.simt.block.sync()

        n_comp = sh_ncomp[0]
        if n_comp > _MAXC:
            # More components than launched blocks: component-block 0 runs the dense cooperative sweep as a fallback.
            if comp == 0:
                func_noslip_batch_coop(
                    tid, i_b, collider_state, constraint_state, rigid_global_info, static_rigid_sim_config
                )
        elif comp < n_comp:
            # --- This block's component: block-cooperative projected Gauss-Seidel, per-component convergence. ---
            res = qd.Vector.zero(gs.qd_float, 5)
            old_force = qd.Vector.zero(gs.qd_float, 5)
            bc = qd.Vector.zero(gs.qd_float, 5)
            Ac = qd.Vector.zero(gs.qd_float, 9)
            scale = 1.0 / (rigid_global_info.meaninertia[i_b] * qd.max(1.0, n_dofs))

            for i_iter in range(rigid_global_info.noslip_iterations[None]):
                improvement = gs.qd_float(0.0)
                if i_iter == 0:
                    # iter-0 baseline cost over THIS component's contact rows only (block-strided over contacts).
                    partial = gs.qd_float(0.0)
                    i_cc = tid
                    while i_cc < n_con:
                        if sh_i[_LAB + i_cc] == comp:
                            for r2 in qd.static(range(4)):
                                row = const_start + i_cc * 4 + r2
                                partial += (
                                    0.5 * constraint_state.efc_force[row, i_b] ** 2 * constraint_state.diag[row, i_b]
                                )
                        i_cc += _BLOCK
                    improvement += _func_noslip_block_reduce_add(sh_warp, tid, partial, static_rigid_sim_config)

                for i_col in range(n_con):
                    if sh_i[_LAB + i_col] == comp:
                        base = const_start + i_col * 4
                        for j2 in qd.static(range(2)):
                            j_efc = base + j2 * 2
                            res = func_residual_constraint_force_coop(
                                res=res,
                                tid=tid,
                                i_b=i_b,
                                i_efc=j_efc,
                                dim=2,
                                sh_warp=sh_warp,
                                constraint_state=constraint_state,
                                static_rigid_sim_config=static_rigid_sim_config,
                            )
                            for i2 in qd.static(range(2)):
                                old_force[i2] = constraint_state.efc_force[j_efc + i2, i_b]
                            Ac = func_extract_block_matrix_from_AR(
                                Ac=Ac,
                                i_b=i_b,
                                start=j_efc,
                                n=2,
                                constraint_state=constraint_state,
                                static_rigid_sim_config=static_rigid_sim_config,
                            )
                            for j in qd.static(range(2)):
                                bc[j] = res[j]
                                for k in qd.static(range(2)):
                                    bc[j] -= Ac[j * 2 + k] * old_force[k]
                            mid = 0.5 * (
                                constraint_state.efc_force[j_efc, i_b] + constraint_state.efc_force[j_efc + 1, i_b]
                            )
                            y = 0.5 * (
                                constraint_state.efc_force[j_efc, i_b] - constraint_state.efc_force[j_efc + 1, i_b]
                            )
                            K1 = Ac[0] + Ac[3] - Ac[1] - Ac[2]
                            K0 = mid * (Ac[0] - Ac[3]) + bc[0] - bc[1]
                            if K1 < EPS:
                                constraint_state.efc_force[j_efc, i_b] = constraint_state.efc_force[j_efc + 1, i_b] = mid
                            else:
                                y = -K0 / K1
                                if y < -mid:
                                    constraint_state.efc_force[j_efc, i_b] = 0
                                    constraint_state.efc_force[j_efc + 1, i_b] = 2 * mid
                                elif y > mid:
                                    constraint_state.efc_force[j_efc, i_b] = 2 * mid
                                    constraint_state.efc_force[j_efc + 1, i_b] = 0
                                else:
                                    constraint_state.efc_force[j_efc, i_b] = mid + y
                                    constraint_state.efc_force[j_efc + 1, i_b] = mid - y
                            cost_change = func_cost_change(
                                i_b=i_b,
                                Ac=Ac,
                                force=constraint_state.efc_force,
                                force_start=j_efc,
                                old_force=old_force,
                                res=res,
                                dim=2,
                                eps=EPS,
                            )
                            improvement -= cost_change
                improvement *= scale
                if improvement < rigid_global_info.noslip_tolerance[None]:
                    break


@qd.func
def func_cost_change(
    i_b: int,
    Ac,
    force: qd.Tensor,
    force_start: int,
    old_force,
    res,
    dim: int,
    eps,
):
    change = gs.qd_float(0.0)
    if dim == 1:
        delta = force[force_start, i_b] - old_force[0]
        change = 0.5 * Ac[0] * delta * delta + delta * res[0]
    else:
        delta = qd.Vector.zero(gs.qd_float, 2)
        for i in range(dim):
            delta[i] = force[force_start + i, i_b] - old_force[i]
        for i in range(dim):
            for j in range(dim):
                change += 0.5 * Ac[i * dim + j] * delta[i] * delta[j]
            change += delta[i] * res[i]
    if change > eps:
        for i in range(dim):
            force[force_start + i, i_b] = old_force[i]
        change = 0.0
    return change


@qd.kernel(fastcache=True)
def compute_A_diag(
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    _B = constraint_state.jac.shape[2]
    n_dofs = constraint_state.jac.shape[1]
    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        # For each constraint row i: Ai = Ji * M^{-1} * Ji^T
        for i_c in range(constraint_state.n_constraints[i_b]):
            # tmp = M^{-1} * Ji^T
            for i_d in range(n_dofs):
                constraint_state.Mgrad[i_d, i_b] = constraint_state.jac[i_c, i_d, i_b]

            rigid_solver.func_solve_mass_batch(
                i_b,
                constraint_state.Mgrad,
                constraint_state.Mgrad,
                None,
                entities_info=entities_info,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
                is_backward=False,
            )

            # Ai = Ji * tmp
            aii = gs.qd_float(0.0)
            if qd.static(static_rigid_sim_config.sparse_solve):
                for i_d_ in range(constraint_state.jac_n_dofs[i_c, i_b]):
                    i_d = constraint_state.jac_dofs_idx[i_c, i_d_, i_b]
                    aii += constraint_state.jac[i_c, i_d, i_b] * constraint_state.Mgrad[i_d, i_b]
            else:
                for i_d in range(n_dofs):
                    aii += constraint_state.jac[i_c, i_d, i_b] * constraint_state.Mgrad[i_d, i_b]
            constraint_state.efc_A_diag[i_c, i_b] = aii
