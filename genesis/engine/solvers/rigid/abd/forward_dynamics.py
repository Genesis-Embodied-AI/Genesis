"""
Rigid solver dynamics kernel and function definitions.

This module contains Quadrants kernel and function definitions for rigid body dynamics
simulation, including:
- Mass matrix computation and factorization
- Force calculations (torque, passive, bias, actuation)
- Forward dynamics computation
- Velocity and acceleration updates
- Integration schemes (Euler, implicit damping)
- Cartesian space updates

These functions are used by the RigidSolver class to perform physics simulation
of articulated rigid body systems.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class
from .misc import (
    func_wakeup_island,
    func_check_index_range,
    func_add_safe_backward,
    linear_to_lower_tri,
)

# Block size (warp width) for the cooperative mass_mat_assemble path. Used only when
# enable_cooperative_constraint_kernels=True (and not use_hibernation). One warp per (entity, env); lanes stride i_d_
# within the entity dof block to coalesce the flipped mass_mat writes.
_MASS_MAT_BLOCK = 32


@qd.kernel
def update_qacc_from_qvel_delta(
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    is_backward: qd.template(),
):
    BW = qd.static(is_backward)

    n_dofs = dofs_state.ctrl_mode.shape[0]
    _B = dofs_state.ctrl_mode.shape[1]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in qd.ndrange(1, _B) if qd.static(static_rigid_sim_config.use_hibernation) else qd.ndrange(n_dofs, _B):
        for i_1 in (
            range(rigid_global_info.n_awake_dofs[i_b])
            if qd.static(static_rigid_sim_config.use_hibernation)
            else qd.static(range(1))
        ):
            if i_1 < (rigid_global_info.n_awake_dofs[i_b] if qd.static(static_rigid_sim_config.use_hibernation) else 1):
                i_d = (
                    rigid_global_info.awake_dofs[i_1, i_b]
                    if qd.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )
                dofs_state.acc[i_d, i_b] = (
                    dofs_state.vel[i_d, i_b] - dofs_state.vel_prev[i_d, i_b]
                ) / rigid_global_info.substep_dt[None]
                dofs_state.vel[i_d, i_b] = dofs_state.vel_prev[i_d, i_b]


@qd.kernel
def update_qvel(
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    is_backward: qd.template(),
):
    BW = qd.static(is_backward)

    _B = dofs_state.vel.shape[1]
    n_dofs = dofs_state.vel.shape[0]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in qd.ndrange(1, _B) if qd.static(static_rigid_sim_config.use_hibernation) else qd.ndrange(n_dofs, _B):
        for i_1 in (
            range(rigid_global_info.n_awake_dofs[i_b])
            if qd.static(static_rigid_sim_config.use_hibernation)
            else qd.static(range(1))
        ):
            if i_1 < (rigid_global_info.n_awake_dofs[i_b] if qd.static(static_rigid_sim_config.use_hibernation) else 1):
                i_d = (
                    rigid_global_info.awake_dofs[i_1, i_b]
                    if qd.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )
                dofs_state.vel_prev[i_d, i_b] = dofs_state.vel[i_d, i_b]
                dofs_state.vel[i_d, i_b] = (
                    dofs_state.vel[i_d, i_b] + dofs_state.acc[i_d, i_b] * rigid_global_info.substep_dt[None]
                )


@qd.kernel(fastcache=True)
def kernel_compute_mass_matrix(
    # Quadrants variables
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    decompose: qd.template(),
):
    func_compute_mass_matrix(
        implicit_damping=False,
        links_state=links_state,
        links_info=links_info,
        dofs_state=dofs_state,
        dofs_info=dofs_info,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        is_backward=False,
    )
    if decompose:
        func_factor_mass(
            implicit_damping=False,
            entities_info=entities_info,
            dofs_state=dofs_state,
            dofs_info=dofs_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
            is_backward=False,
        )


# @@@@@@@@@ Composer starts here
# decomposed kernels should happen in the block below. This block will be handled by composer and composed into a single kernel
@qd.func
def func_forward_dynamics(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    joints_info: array_class.JointsInfo,
    entities_state: array_class.EntitiesState,
    entities_info: array_class.EntitiesInfo,
    geoms_state: array_class.GeomsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    island_state: array_class.IslandState,
    is_backward: qd.template(),
):
    func_compute_mass_matrix(
        implicit_damping=qd.static(static_rigid_sim_config.integrator == gs.integrator.approximate_implicitfast),
        links_state=links_state,
        links_info=links_info,
        dofs_state=dofs_state,
        dofs_info=dofs_info,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        is_backward=is_backward,
    )
    func_factor_mass(
        implicit_damping=False,
        entities_info=entities_info,
        dofs_state=dofs_state,
        dofs_info=dofs_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        is_backward=is_backward,
    )
    func_torque_and_passive_force(
        entities_state=entities_state,
        entities_info=entities_info,
        dofs_state=dofs_state,
        dofs_info=dofs_info,
        links_state=links_state,
        links_info=links_info,
        joints_info=joints_info,
        geoms_state=geoms_state,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        island_state=island_state,
        is_backward=is_backward,
    )
    func_update_acc(
        update_cacc=False,
        dofs_state=dofs_state,
        links_info=links_info,
        links_state=links_state,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        is_backward=is_backward,
    )
    func_update_force(
        links_state=links_state,
        links_info=links_info,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        is_backward=is_backward,
    )
    func_bias_force(
        dofs_state=dofs_state,
        links_state=links_state,
        links_info=links_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        is_backward=is_backward,
    )
    func_compute_qacc(
        dofs_state=dofs_state,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        is_backward=is_backward,
    )


@qd.kernel(fastcache=True)
def kernel_forward_dynamics(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    joints_info: array_class.JointsInfo,
    entities_state: array_class.EntitiesState,
    entities_info: array_class.EntitiesInfo,
    geoms_state: array_class.GeomsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    island_state: array_class.IslandState,
):
    func_forward_dynamics(
        links_state=links_state,
        links_info=links_info,
        dofs_state=dofs_state,
        dofs_info=dofs_info,
        joints_info=joints_info,
        entities_state=entities_state,
        entities_info=entities_info,
        geoms_state=geoms_state,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        island_state=island_state,
        is_backward=False,
    )


@qd.kernel(fastcache=True)
def kernel_update_acc(
    dofs_state: array_class.DofsState,
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    func_update_acc(
        update_cacc=True,
        dofs_state=dofs_state,
        links_info=links_info,
        links_state=links_state,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        is_backward=False,
    )


@qd.func
def func_vel_at_point(pos_world, link_idx, i_b, links_state: array_class.LinksState):
    """
    Velocity of a certain point on a rigid link.
    """
    vel_rot = links_state.cd_ang[link_idx, i_b].cross(pos_world - links_state.root_COM[link_idx, i_b])
    vel_lin = links_state.cd_vel[link_idx, i_b]
    return vel_rot + vel_lin


@qd.func
def func_compute_mass_matrix(
    implicit_damping: qd.template(),
    # Quadrants variables
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    is_backward: qd.template(),
):
    BW = qd.static(is_backward)

    # crb initialize
    qd.loop_config(name="crb_initialize", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        qd.ndrange(1, links_state.pos.shape[1])
        if qd.static(static_rigid_sim_config.use_hibernation)
        else qd.ndrange(links_state.pos.shape[0], links_state.pos.shape[1])
    ):
        for i_1 in (
            range(rigid_global_info.n_awake_links[i_b])
            if qd.static(static_rigid_sim_config.use_hibernation)
            else qd.static(range(1))
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_links[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_l = (
                    rigid_global_info.awake_links[i_1, i_b]
                    if qd.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )

                links_state.crb_inertial[i_l, i_b] = links_state.cinr_inertial[i_l, i_b]
                links_state.crb_pos[i_l, i_b] = links_state.cinr_pos[i_l, i_b]
                links_state.crb_quat[i_l, i_b] = links_state.cinr_quat[i_l, i_b]
                links_state.crb_mass[i_l, i_b] = links_state.cinr_mass[i_l, i_b]

    # crb
    qd.loop_config(name="crb", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        qd.ndrange(1, links_state.pos.shape[1])
        if qd.static(static_rigid_sim_config.use_hibernation)
        else qd.ndrange(entities_info.n_links.shape[0], links_state.pos.shape[1])
    ):
        for i_1 in (
            range(rigid_global_info.n_awake_entities[i_b])
            if qd.static(static_rigid_sim_config.use_hibernation)
            else qd.static(range(1))
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_entities[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_e = (
                    rigid_global_info.awake_entities[i_1, i_b]
                    if qd.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )

                for i in range(entities_info.n_links[i_e]):
                    i_l = entities_info.link_end[i_e] - 1 - i
                    I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
                    i_p = links_info.parent_idx[I_l]
                    I_p = [i_p, i_b]

                    if i_p != -1:
                        func_add_safe_backward(links_state.crb_inertial, I_p, links_state.crb_inertial[i_l, i_b], BW)
                        func_add_safe_backward(links_state.crb_mass, I_p, links_state.crb_mass[i_l, i_b], BW)
                        func_add_safe_backward(links_state.crb_pos, I_p, links_state.crb_pos[i_l, i_b], BW)
                        func_add_safe_backward(links_state.crb_quat, I_p, links_state.crb_quat[i_l, i_b], BW)

    # mass_mat
    qd.loop_config(name="mass_mat", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        qd.ndrange(1, links_state.pos.shape[1])
        if qd.static(static_rigid_sim_config.use_hibernation)
        else qd.ndrange(links_state.pos.shape[0], links_state.pos.shape[1])
    ):
        for i_1 in (
            range(rigid_global_info.n_awake_links[i_b])
            if qd.static(static_rigid_sim_config.use_hibernation)
            else qd.static(range(1))
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_links[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_l = (
                    rigid_global_info.awake_links[i_1, i_b]
                    if qd.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )
                I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l

                for i_d in range(links_info.dof_start[I_l], links_info.dof_end[I_l]):
                    dofs_state.f_ang[i_d, i_b], dofs_state.f_vel[i_d, i_b] = gu.inertial_mul(
                        links_state.crb_pos[i_l, i_b],
                        links_state.crb_inertial[i_l, i_b],
                        links_state.crb_mass[i_l, i_b],
                        dofs_state.cdof_vel[i_d, i_b],
                        dofs_state.cdof_ang[i_d, i_b],
                    )

    if qd.static(
        static_rigid_sim_config.enable_cooperative_constraint_kernels and not static_rigid_sim_config.use_hibernation
    ):
        # Cooperative warp-per-(entity, env) writer over the lower triangle (inclusive of diagonal). Each cell's
        # symmetric value is computed once via the sqrt-formula compressed pair index and written to both
        # `[i_d, j_d, i_b]` and `[j_d, i_d, i_b]` inline, saving the upper-tri dot products that the previous
        # two-pass path computed and then overwrote, and removing the separate mirror pass. Under the flipped
        # mass_mat layout (i_d stride-1) the primary write coalesces; the inline mirror write is strided but
        # replaces the previous mirror-pass read-write at similar cost.
        _T = qd.static(_MASS_MAT_BLOCK)
        n_entities = entities_info.n_links.shape[0]
        _B_assemble = links_state.pos.shape[1]
        qd.loop_config(name="mass_mat_assemble", block_dim=_T)
        for i_flat in range(n_entities * _B_assemble * _T):
            tid = i_flat % _T
            i_eb = i_flat // _T
            i_e = i_eb % n_entities
            i_b = i_eb // n_entities

            d_s = entities_info.dof_start[i_e]
            d_e = entities_info.dof_end[i_e]
            n_e_e = d_e - d_s
            n_lower_tri = n_e_e * (n_e_e + 1) // 2

            i_pair = tid
            while i_pair < n_lower_tri:
                # Compressed lower-tri-inclusive index (matches tiled func_factor_mass): i_pair = i_d_ * (i_d_ + 1) / 2
                # + j_d_, with j_d_ in [0, i_d_]. The fast-math-robust inversion is required here: a raw sqrt drops the
                # j=0 entry of every perfect-square row on GPU, leaving M missing long-range coupling -> indefinite.
                i_d_, j_d_ = linear_to_lower_tri(i_pair)
                i_d = d_s + i_d_
                j_d = d_s + j_d_
                # The mass matrix is block-diagonal per kinematic tree, so only within-block (j_d in i_d's block) pairs
                # can be non-zero. Skipping cross-block pairs avoids their dot products; those entries stay zero
                # (mass_mat is zeroed and nothing else writes them). This makes the assemble cost scale with the sum of
                # per-tree blocks instead of the whole (possibly multi-body) entity.
                if j_d >= rigid_global_info.dofs_mass_block_start[i_d]:
                    val = (
                        dofs_state.f_ang[i_d, i_b].dot(dofs_state.cdof_ang[j_d, i_b])
                        + dofs_state.f_vel[i_d, i_b].dot(dofs_state.cdof_vel[j_d, i_b])
                    ) * rigid_global_info.mass_parent_mask[i_d, j_d]
                    rigid_global_info.mass_mat[i_d, j_d, i_b] = val
                    if i_d_ != j_d_:
                        rigid_global_info.mass_mat[j_d, i_d, i_b] = val
                i_pair += _T
    else:
        qd.loop_config(
            name="mass_mat_assemble", serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        )
        for i_0, i_b in (
            qd.ndrange(1, links_state.pos.shape[1])
            if qd.static(static_rigid_sim_config.use_hibernation)
            else qd.ndrange(entities_info.n_links.shape[0], links_state.pos.shape[1])
        ):
            for i_1 in (
                range(rigid_global_info.n_awake_entities[i_b])
                if qd.static(static_rigid_sim_config.use_hibernation)
                else qd.static(range(1))
            ):
                if func_check_index_range(
                    i_1, 0, rigid_global_info.n_awake_entities[i_b], static_rigid_sim_config.use_hibernation
                ):
                    i_e = (
                        rigid_global_info.awake_entities[i_1, i_b]
                        if qd.static(static_rigid_sim_config.use_hibernation)
                        else i_0
                    )

                    for i_d in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                        for j_d in range(
                            rigid_global_info.dofs_mass_block_start[i_d], rigid_global_info.dofs_mass_block_end[i_d]
                        ):
                            rigid_global_info.mass_mat[i_d, j_d, i_b] = (
                                dofs_state.f_ang[i_d, i_b].dot(dofs_state.cdof_ang[j_d, i_b])
                                + dofs_state.f_vel[i_d, i_b].dot(dofs_state.cdof_vel[j_d, i_b])
                            ) * rigid_global_info.mass_parent_mask[i_d, j_d]

                    for i_d in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                        for j_d in range(i_d + 1, rigid_global_info.dofs_mass_block_end[i_d]):
                            rigid_global_info.mass_mat[i_d, j_d, i_b] = rigid_global_info.mass_mat[j_d, i_d, i_b]

    # Take into account motor armature
    qd.loop_config(name="armature", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_d, i_b in qd.ndrange(dofs_state.f_ang.shape[0], links_state.pos.shape[1]):
        I_d = [i_d, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else i_d
        func_add_safe_backward(rigid_global_info.mass_mat, (i_d, i_d, i_b), dofs_info.armature[I_d], BW)

    # Take into account first-order correction terms for implicit integration scheme right away
    if qd.static(implicit_damping):
        qd.loop_config(name="impint_order_1_corr", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_d, i_b in qd.ndrange(dofs_state.f_ang.shape[0], links_state.pos.shape[1]):
            I_d = [i_d, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else i_d
            rigid_global_info.mass_mat[i_d, i_d, i_b] = (
                rigid_global_info.mass_mat[i_d, i_d, i_b] + dofs_info.damping[I_d] * rigid_global_info.substep_dt[None]
            )
            if dofs_state.ctrl_mode[i_d, i_b] <= gs.CTRL_MODE.VELOCITY:
                # qM += d qfrc_actuator / d qvel = -act_bias[2] * dt
                rigid_global_info.mass_mat[i_d, i_d, i_b] = (
                    rigid_global_info.mass_mat[i_d, i_d, i_b]
                    - dofs_info.act_bias[I_d][2] * rigid_global_info.substep_dt[None]
                )


@qd.func
def func_factor_mass_tiled(
    implicit_damping: qd.template(),
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    TileCls: qd.template(),
):
    """Register-streaming tiled per-entity mass factor for the >shared-cap branch (GPU forward only).

    Replaces the shared-pivot cooperative LDL^T when an entity's mass submatrix exceeds GPU shared memory. M is
    block-diagonal per kinematic tree, so one warp of T lanes factors each of the entity's mass blocks independently
    (a single-tree entity has just one block spanning it) via the same qd.simt.TileNxN blocked Cholesky as the
    constraint Hessian.

    func_solve_mass consumes the LTDL form M = L^T D L (L unit-lower), produced by eliminating DOFs last-to-first, not
    the standard L D L^T. The tile primitive does forward Cholesky M = G G^T, so each block's reverse-indexed matrix
    M_rev[a, b] = M[n-1-a, n-1-b] (n the block size) is factored and its factor mapped back to the block's LTDL factor:
      L[i,j] = G_rev[n-1-j, n-1-i] / G_rev[n-1-i, n-1-i]  (i > j),  D_inv[i] = 1 / G_rev[n-1-i, n-1-i]^2,  diag(L) = 1.
    See test_rigid_physics for the parity check against the cooperative factor.

    The qd.simt tile ops are batch-first while mass_mat_L is canonical batch-last (n_dofs, n_dofs, _B), so the
    factorization runs in each mass block's region of the batch-first scratch
    rigid_global_info.mass_mat_tiled_scratch and is scattered into mass_mat_L / mass_mat_D_inv. To avoid a dedicated
    allocation, that scratch aliases the constraint Hessian buffer nt_H (same shape, and free at mass-factor time since
    the constraint solve only populates it later in the step); see get_constraint_state. The scratch and mass_mat_L are
    distinct buffers, so the scatter is race-free. Backward keeps its own branch in func_factor_mass.
    """
    # Reuse the Hessian's tile width; TileCls is dispatched to match it at the call site, so T and the tile class stay
    # consistent for either value. In practice this path only runs for per-entity blocks exceeding shared memory (total
    # n_dofs > 48), where the rule lands on 32.
    T = qd.static(static_rigid_sim_config.cholesky_tile_size)
    EPS = rigid_global_info.EPS[None]

    n_entities = entities_info.n_links.shape[0]
    _B = dofs_state.ctrl_mode.shape[1]

    qd.loop_config(name="factor_mass", block_dim=T)
    for i in range(n_entities * _B * T):
        tid = i % T
        i_e = (i // T) % n_entities
        i_b = i // (T * n_entities)
        if i_b >= _B:
            continue
        # Skip hibernated entities: their mass matrix is unchanged, so the factor from the last awake step stays valid.
        # The slot remaps to an awake entity, so the work scales with the awake entity count. Distinct (awake) entities
        # own disjoint DOF ranges, so their mass_mat_tiled_scratch block-diagonal scratch regions never alias.
        if qd.static(static_rigid_sim_config.use_hibernation):
            if i_e >= rigid_global_info.n_awake_entities[i_b]:
                continue
            i_e = rigid_global_info.awake_entities[i_e, i_b]
        if not rigid_global_info.mass_mat_mask[i_e, i_b]:
            continue

        # Factor each mass block (kinematic tree) independently: a multi-tree entity has several blocks, a single-tree
        # entity (the common case) just one spanning the whole entity. This matches the cooperative path, which likewise
        # restricts to [block_start, block_end). The block's M and factor live at its own DOFs [block_start, ...), but
        # the tile workspace reuses the entity's region [d_s, d_s + n_block_dofs) across the entity's blocks (processed
        # sequentially by this warp; disjoint from other entities' regions), keeping the scratch indices short.
        d_s = entities_info.dof_start[i_e]
        entity_dof_end = entities_info.dof_end[i_e]
        block_start = d_s
        while block_start < entity_dof_end:
            n_block_dofs = rigid_global_info.dofs_mass_block_end[block_start] - block_start
            n_blocks = (n_block_dofs + T - 1) // T

            # Phase 1: copy the reverse-indexed symmetric M block (+ implicit damping) into the scratch workspace.
            # mass_mat stores M's lower triangle, so M[ri_, rj_] with ri_ <= rj_ is read from the stored M[rj_, ri_].
            i_d_ = tid
            while i_d_ < n_block_dofs:
                ri_ = n_block_dofs - 1 - i_d_
                for j_d_ in range(i_d_ + 1):
                    rj_ = n_block_dofs - 1 - j_d_  # i_d_ >= j_d_  =>  ri_ <= rj_
                    m = rigid_global_info.mass_mat[block_start + rj_, block_start + ri_, i_b]
                    rigid_global_info.mass_mat_tiled_scratch[i_b, d_s + i_d_, d_s + j_d_] = m
                    rigid_global_info.mass_mat_tiled_scratch[i_b, d_s + j_d_, d_s + i_d_] = m
                if qd.static(implicit_damping):
                    # Reverse-diagonal slot i_d_ holds M[ri_, ri_]; damping/act_bias index the original DOF.
                    i_d = block_start + ri_
                    I_d = [i_d, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else i_d
                    rigid_global_info.mass_mat_tiled_scratch[i_b, d_s + i_d_, d_s + i_d_] = (
                        rigid_global_info.mass_mat_tiled_scratch[i_b, d_s + i_d_, d_s + i_d_]
                        + dofs_info.damping[I_d] * rigid_global_info.substep_dt[None]
                    )
                    if qd.static(static_rigid_sim_config.integrator == gs.integrator.implicitfast):
                        if dofs_state.ctrl_mode[i_d, i_b] <= gs.CTRL_MODE.VELOCITY:
                            rigid_global_info.mass_mat_tiled_scratch[i_b, d_s + i_d_, d_s + i_d_] = (
                                rigid_global_info.mass_mat_tiled_scratch[i_b, d_s + i_d_, d_s + i_d_]
                                - dofs_info.act_bias[I_d][2] * rigid_global_info.substep_dt[None]
                            )
                i_d_ = i_d_ + T
            qd.simt.block.sync()

            # Phase 2: blocked Cholesky G_rev G_rev^T = M_rev in the scratch workspace (mirrors the constraint Hessian's
            # func_cholesky_factor_direct_tiled; the tile ops are warp-synchronous, so no sync inside the loop).
            for kb in range(n_blocks):
                k0 = kb * T
                k1 = qd.min(k0 + T, n_block_dofs)

                L_kk = TileCls.eye(dtype=gs.qd_float)  # rows past n_block_dofs stay identity
                L_kk[:] = rigid_global_info.mass_mat_tiled_scratch[i_b, d_s + k0 : d_s + k1, d_s + k0 : d_s + k1]
                for jb in range(kb):
                    j0 = jb * T
                    for t in range(T):
                        v = rigid_global_info.mass_mat_tiled_scratch[i_b, d_s + k0 : d_s + k1, d_s + j0 + t]
                        L_kk -= qd.outer(v, v)
                L_kk.cholesky_(EPS)

                for ib in range(kb + 1, n_blocks):
                    i0 = ib * T
                    i1 = qd.min(i0 + T, n_block_dofs)

                    L_ik = TileCls.zeros(dtype=gs.qd_float)
                    L_ik[:] = rigid_global_info.mass_mat_tiled_scratch[i_b, d_s + i0 : d_s + i1, d_s + k0 : d_s + k1]
                    for jb in range(kb):
                        j0 = jb * T
                        for t in range(T):
                            v_own = rigid_global_info.mass_mat_tiled_scratch[i_b, d_s + i0 : d_s + i1, d_s + j0 + t]
                            v_diag = rigid_global_info.mass_mat_tiled_scratch[i_b, d_s + k0 : d_s + k1, d_s + j0 + t]
                            L_ik -= qd.outer(v_own, v_diag)
                    L_kk.solve_triangular_(L_ik)
                    rigid_global_info.mass_mat_tiled_scratch[i_b, d_s + i0 : d_s + i1, d_s + k0 : d_s + k1] = L_ik

                rigid_global_info.mass_mat_tiled_scratch[i_b, d_s + k0 : d_s + k1, d_s + k0 : d_s + k1] = L_kk
            qd.simt.block.sync()

            # Phase 3: scatter the LTDL factor of M from G_rev (scratch) into canonical mass_mat_L / mass_mat_D_inv.
            # Reads the scratch, writes the distinct mass_mat_L (no in-place hazard). Only the strict-lower triangle and
            # unit diagonal are meaningful to the solve; the upper triangle is left untouched.
            n_strict_lower = n_block_dofs * (n_block_dofs - 1) // 2
            i_pair = tid
            while i_pair < n_strict_lower:
                i_d_, j_d_ = linear_to_lower_tri(i_pair, strict=True)
                ri_ = n_block_dofs - 1 - i_d_
                rj_ = n_block_dofs - 1 - j_d_  # i_d_ > j_d_  =>  rj_ > ri_  (a lower G_rev entry)
                g_num = rigid_global_info.mass_mat_tiled_scratch[i_b, d_s + rj_, d_s + ri_]
                g_den = rigid_global_info.mass_mat_tiled_scratch[i_b, d_s + ri_, d_s + ri_]
                rigid_global_info.mass_mat_L[block_start + i_d_, block_start + j_d_, i_b] = g_num / g_den
                i_pair = i_pair + T

            i_d_ = tid
            while i_d_ < n_block_dofs:
                ri_ = n_block_dofs - 1 - i_d_
                g_den = rigid_global_info.mass_mat_tiled_scratch[i_b, d_s + ri_, d_s + ri_]
                rigid_global_info.mass_mat_D_inv[block_start + i_d_, i_b] = 1.0 / (g_den * g_den)
                rigid_global_info.mass_mat_L[block_start + i_d_, block_start + i_d_, i_b] = 1.0
                i_d_ = i_d_ + T

            block_start = rigid_global_info.dofs_mass_block_end[block_start]


@qd.func
def func_factor_mass(
    implicit_damping: qd.template(),
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    is_backward: qd.template(),
):
    BW = qd.static(is_backward)

    if qd.static(not BW):
        n_entities = entities_info.n_links.shape[0]
        _B = dofs_state.ctrl_mode.shape[1]

        if qd.static(static_rigid_sim_config.enable_register_tiled_mass):
            # Register-streaming tiled per-entity factor for the >shared-cap path (same primitive as the constraint
            # Hessian). Implies enable_tiled_cholesky_mass_matrix and not mass_matrix_fits_shared; see
            # func_factor_mass_tiled. Replaces the cooperative LDL^T in the elif below.
            func_factor_mass_tiled(
                implicit_damping,
                entities_info,
                dofs_state,
                dofs_info,
                rigid_global_info,
                static_rigid_sim_config,
                qd.simt.Tile32x32 if qd.static(static_rigid_sim_config.cholesky_tile_size == 32) else qd.simt.Tile16x16,
            )
        elif qd.static(
            static_rigid_sim_config.enable_tiled_cholesky_mass_matrix
            and not static_rigid_sim_config.mass_matrix_fits_shared
        ):
            # Uncapped cooperative per-entity LDL^T (entity submatrix does not fit shared memory): factors the entity
            # mass submatrix in-place in global memory (mass_mat_L) over a block of BLOCK_DIM threads. Each elimination
            # step snapshots the pivot row into a small shared vector (O(n_dofs), not O(n_dofs^2)) before updating the
            # trailing submatrix, so the parallel per-row updates only READ the pivot row (from shared) -- race-free
            # regardless of scheduling. Numerically identical to the scalar branch below; only parallelization differs.
            BLOCK_DIM = qd.static(32)
            MAX_DOFS_PER_ENTITY = qd.static(static_rigid_sim_config.tiled_n_dofs_per_entity)

            qd.loop_config(name="factor_mass", block_dim=BLOCK_DIM)
            for i in range(n_entities * _B * BLOCK_DIM):
                tid = i % BLOCK_DIM
                i_e = (i // BLOCK_DIM) % n_entities
                i_b = i // (BLOCK_DIM * n_entities)
                if i_b >= _B:
                    continue
                # Skip hibernated entities: their mass matrix is unchanged, so the factor from the last awake step
                # stays valid. The slot remaps to an awake entity, so the work scales with the awake entity count.
                if qd.static(static_rigid_sim_config.use_hibernation):
                    if i_e >= rigid_global_info.n_awake_entities[i_b]:
                        continue
                    i_e = rigid_global_info.awake_entities[i_e, i_b]

                if rigid_global_info.mass_mat_mask[i_e, i_b]:
                    entity_dof_start = entities_info.dof_start[i_e]
                    entity_dof_end = entities_info.dof_end[i_e]
                    n_dofs = entities_info.n_dofs[i_e]

                    pivot_row = qd.simt.block.SharedArray((MAX_DOFS_PER_ENTITY,), gs.qd_float)

                    # Copy the lower triangle of M into mass_mat_L (+ implicit damping on the diagonal), cooperatively.
                    # The mass matrix is block-diagonal per kinematic tree, so only the within-block lower triangle is
                    # non-zero; restricting to it makes the factorization cost the sum of per-tree cubes instead of the
                    # whole (possibly multi-body) entity cube. Cross-block entries stay zero (mass_mat_L is zeroed).
                    i_d_ = tid
                    while i_d_ < n_dofs:
                        i_d = entity_dof_start + i_d_
                        block_start = rigid_global_info.dofs_mass_block_start[i_d]
                        for j_d in range(block_start, i_d + 1):
                            rigid_global_info.mass_mat_L[i_d, j_d, i_b] = rigid_global_info.mass_mat[i_d, j_d, i_b]
                        if qd.static(implicit_damping):
                            I_d = [i_d, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else i_d
                            rigid_global_info.mass_mat_L[i_d, i_d, i_b] = (
                                rigid_global_info.mass_mat_L[i_d, i_d, i_b]
                                + dofs_info.damping[I_d] * rigid_global_info.substep_dt[None]
                            )
                            if qd.static(static_rigid_sim_config.integrator == gs.integrator.implicitfast):
                                if dofs_state.ctrl_mode[i_d, i_b] <= gs.CTRL_MODE.VELOCITY:
                                    rigid_global_info.mass_mat_L[i_d, i_d, i_b] = (
                                        rigid_global_info.mass_mat_L[i_d, i_d, i_b]
                                        - dofs_info.act_bias[I_d][2] * rigid_global_info.substep_dt[None]
                                    )
                        i_d_ = i_d_ + BLOCK_DIM
                    qd.simt.block.sync()

                    # In-place LDL^T, eliminating dofs from last to first (matches the scalar branch). Each pivot only
                    # touches the trailing submatrix within its own block, so blocks factor independently.
                    for j in range(n_dofs):
                        i_d = entity_dof_end - j - 1
                        block_start = rigid_global_info.dofs_mass_block_start[i_d]
                        i_d_local = i_d - block_start
                        D_inv = 1.0 / rigid_global_info.mass_mat_L[i_d, i_d, i_b]
                        if tid == 0:
                            rigid_global_info.mass_mat_D_inv[i_d, i_b] = D_inv

                        # Phase A: snapshot the (Schur-updated) pivot-row entries below the diagonal into shared.
                        j_d_ = tid
                        while j_d_ < i_d_local:
                            pivot_row[j_d_] = rigid_global_info.mass_mat_L[i_d, block_start + j_d_, i_b]
                            j_d_ = j_d_ + BLOCK_DIM
                        qd.simt.block.sync()

                        # Phase B: each lane eliminates one column j_d, updating its own row j_d of the trailing
                        # submatrix from the read-only snapshot. Distinct rows per lane => no write conflicts, and
                        # the pivot row is only read (from shared) => no read/write race on row i_d.
                        j_d_ = tid
                        while j_d_ < i_d_local:
                            a = pivot_row[j_d_] * D_inv
                            j_d = block_start + j_d_
                            for k_d_ in range(j_d_ + 1):
                                rigid_global_info.mass_mat_L[j_d, block_start + k_d_, i_b] = (
                                    rigid_global_info.mass_mat_L[j_d, block_start + k_d_, i_b] - a * pivot_row[k_d_]
                                )
                            rigid_global_info.mass_mat_L[i_d, j_d, i_b] = a
                            j_d_ = j_d_ + BLOCK_DIM
                        qd.simt.block.sync()

                        # Diagonal coeffs of L are ignored downstream (see scalar branch) but set to 1.0 to match.
                        if tid == 0:
                            rigid_global_info.mass_mat_L[i_d, i_d, i_b] = 1.0
        elif qd.static(
            not static_rigid_sim_config.enable_tiled_cholesky_mass_matrix or static_rigid_sim_config.backend == gs.cpu
        ):
            qd.loop_config(name="factor_mass", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
            for i_slot, i_b in qd.ndrange(n_entities, _B):
                # Skip hibernated entities: their mass matrix is unchanged, so the factor from the last awake step
                # stays valid. This makes the factorization cost scale with the awake entity count.
                i_e = i_slot
                if qd.static(static_rigid_sim_config.use_hibernation):
                    if i_slot >= rigid_global_info.n_awake_entities[i_b]:
                        continue
                    i_e = rigid_global_info.awake_entities[i_slot, i_b]
                if rigid_global_info.mass_mat_mask[i_e, i_b]:
                    entity_dof_start = entities_info.dof_start[i_e]
                    entity_dof_end = entities_info.dof_end[i_e]
                    n_dofs = entities_info.n_dofs[i_e]

                    for i_d in range(entity_dof_start, entity_dof_end):
                        block_start = rigid_global_info.dofs_mass_block_start[i_d]
                        for j_d in range(block_start, i_d + 1):
                            rigid_global_info.mass_mat_L[i_d, j_d, i_b] = rigid_global_info.mass_mat[i_d, j_d, i_b]

                        if qd.static(implicit_damping):
                            I_d = [i_d, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else i_d
                            rigid_global_info.mass_mat_L[i_d, i_d, i_b] = (
                                rigid_global_info.mass_mat_L[i_d, i_d, i_b]
                                + dofs_info.damping[I_d] * rigid_global_info.substep_dt[None]
                            )
                            if qd.static(static_rigid_sim_config.integrator == gs.integrator.implicitfast):
                                if dofs_state.ctrl_mode[i_d, i_b] <= gs.CTRL_MODE.VELOCITY:
                                    rigid_global_info.mass_mat_L[i_d, i_d, i_b] = (
                                        rigid_global_info.mass_mat_L[i_d, i_d, i_b]
                                        - dofs_info.act_bias[I_d][2] * rigid_global_info.substep_dt[None]
                                    )

                    for i_d_ in range(n_dofs):
                        i_d = entity_dof_end - i_d_ - 1
                        block_start = rigid_global_info.dofs_mass_block_start[i_d]
                        D_inv = 1.0 / rigid_global_info.mass_mat_L[i_d, i_d, i_b]
                        rigid_global_info.mass_mat_D_inv[i_d, i_b] = D_inv

                        for j_d_ in range(i_d - block_start):
                            j_d = i_d - j_d_ - 1
                            a = rigid_global_info.mass_mat_L[i_d, j_d, i_b] * D_inv
                            for k_d in range(block_start, j_d + 1):
                                rigid_global_info.mass_mat_L[j_d, k_d, i_b] -= (
                                    a * rigid_global_info.mass_mat_L[i_d, k_d, i_b]
                                )
                            rigid_global_info.mass_mat_L[i_d, j_d, i_b] = a

                        # FIXME: Diagonal coeffs of L are ignored in computations, so no need to update them.
                        rigid_global_info.mass_mat_L[i_d, i_d, i_b] = 1.0
        else:
            BLOCK_DIM = qd.static(32)
            MAX_DOFS_PER_ENTITY = qd.static(static_rigid_sim_config.tiled_n_dofs_per_entity)
            WARP_SIZE = qd.static(32)

            qd.loop_config(name="factor_mass", block_dim=BLOCK_DIM)
            for i in range(n_entities * _B * BLOCK_DIM):
                tid = i % BLOCK_DIM
                i_e = (i // BLOCK_DIM) % n_entities
                i_b = i // (BLOCK_DIM * n_entities)
                if i_b >= _B:
                    continue
                # Skip hibernated entities: their mass matrix is unchanged, so the factor from the last awake step
                # stays valid. The slot remaps to an awake entity, so the work scales with the awake entity count.
                if qd.static(static_rigid_sim_config.use_hibernation):
                    if i_e >= rigid_global_info.n_awake_entities[i_b]:
                        continue
                    i_e = rigid_global_info.awake_entities[i_e, i_b]

                if rigid_global_info.mass_mat_mask[i_e, i_b]:
                    entity_dof_start = entities_info.dof_start[i_e]
                    entity_dof_end = entities_info.dof_end[i_e]
                    n_dofs = entities_info.n_dofs[i_e]
                    n_lower_tri = n_dofs * (n_dofs + 1) // 2

                    mass_mat = qd.simt.block.SharedArray((MAX_DOFS_PER_ENTITY, MAX_DOFS_PER_ENTITY + 1), gs.qd_float)

                    i_pair = tid
                    while i_pair < n_lower_tri:
                        i_d_, j_d_ = linear_to_lower_tri(i_pair)
                        i_d = entity_dof_start + i_d_
                        j_d = entity_dof_start + j_d_
                        mass_mat[i_d_, j_d_] = rigid_global_info.mass_mat[i_d, j_d, i_b]
                        i_pair = i_pair + BLOCK_DIM
                    qd.simt.block.sync()

                    if qd.static(implicit_damping):
                        i_d_ = tid
                        while i_d_ < n_dofs:
                            i_d = entity_dof_start + i_d_
                            I_d = [i_d, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else i_d
                            mass_mat[i_d_, i_d_] = (
                                mass_mat[i_d_, i_d_] + dofs_info.damping[I_d] * rigid_global_info.substep_dt[None]
                            )
                            if qd.static(static_rigid_sim_config.integrator == gs.integrator.implicitfast):
                                if dofs_state.ctrl_mode[i_d, i_b] <= gs.CTRL_MODE.VELOCITY:
                                    mass_mat[i_d_, i_d_] = (
                                        mass_mat[i_d_, i_d_]
                                        - dofs_info.act_bias[I_d][2] * rigid_global_info.substep_dt[None]
                                    )
                            i_d_ = i_d_ + BLOCK_DIM
                        qd.simt.block.sync()

                    for j in range(n_dofs):
                        i_d_ = n_dofs - j - 1
                        i_d = entity_dof_end - j - 1
                        # Block-local lower bound (in entity-local shared-memory indices): the mass matrix is
                        # block-diagonal per kinematic tree, so each pivot only eliminates within its own block.
                        block_start_ = rigid_global_info.dofs_mass_block_start[i_d] - entity_dof_start

                        D_inv = 1.0 / mass_mat[i_d_, i_d_]
                        if tid == 0:
                            rigid_global_info.mass_mat_D_inv[i_d, i_b] = D_inv
                            # FIXME: Diagonal coeffs of L are ignored in computations, so no need to update them.
                            rigid_global_info.mass_mat_L[i_d, i_d, i_b] = 1.0

                        j_d_ = i_d_ - 1 - tid
                        while j_d_ >= block_start_:
                            a = mass_mat[i_d_, j_d_] * D_inv
                            for k_d in range(block_start_, j_d_ + 1):
                                mass_mat[j_d_, k_d] = mass_mat[j_d_, k_d] - a * mass_mat[i_d_, k_d]
                            mass_mat[i_d_, j_d_] = a
                            j_d_ = j_d_ - BLOCK_DIM
                        if qd.static(static_rigid_sim_config.backend == gs.cuda):
                            if i_d_ <= WARP_SIZE:
                                qd.simt.warp.sync(qd.u32(0xFFFFFFFF))
                            else:
                                qd.simt.block.sync()
                        else:
                            qd.simt.block.sync()

                    i_pair = tid
                    n_strict_lower_tri = n_dofs * (n_dofs - 1) // 2
                    while i_pair < n_strict_lower_tri:
                        i_d_, j_d_ = linear_to_lower_tri(i_pair, strict=True)
                        i_d = entity_dof_start + i_d_
                        j_d = entity_dof_start + j_d_
                        rigid_global_info.mass_mat_L[i_d, j_d, i_b] = mass_mat[i_d_, j_d_]
                        i_pair = i_pair + BLOCK_DIM
    else:
        # Cholesky decomposition that has safe access pattern and robust handling of divide by zero for AD. Even though
        # it is logically equivalent to the above block, it shows slightly numerical difference in the result, and thus
        # it fails for a unit test ("test_urdf_rope"), while passing all the others. TODO: Investigate if we can fix this
        # and only use this block.

        # Assume this is the outermost loop
        qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_e, i_b in qd.ndrange(entities_info.n_links.shape[0], dofs_state.ctrl_mode.shape[1]):
            if rigid_global_info.mass_mat_mask[i_e, i_b]:
                EPS = rigid_global_info.EPS[None]

                entity_dof_start = entities_info.dof_start[i_e]
                entity_dof_end = entities_info.dof_end[i_e]
                n_dofs = entities_info.n_dofs[i_e]

                for i_d0 in range(n_dofs):
                    i_d = entity_dof_start + i_d0
                    i_pr = (entity_dof_start + entity_dof_end - 1) - i_d
                    for j_d in range(entity_dof_start, i_d + 1):
                        j_pr = (entity_dof_start + entity_dof_end - 1) - j_d
                        rigid_global_info.mass_mat_L_bw[0, i_pr, j_pr, i_b] = rigid_global_info.mass_mat[i_d, j_d, i_b]
                        rigid_global_info.mass_mat_L_bw[0, j_pr, i_pr, i_b] = rigid_global_info.mass_mat[i_d, j_d, i_b]

                    if qd.static(implicit_damping):
                        I_d = [i_d, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else i_d
                        qd.atomic_add(
                            rigid_global_info.mass_mat_L_bw[0, i_pr, i_pr, i_b],
                            (dofs_info.damping[I_d] * rigid_global_info.substep_dt[None]),
                        )
                        if qd.static(static_rigid_sim_config.integrator == gs.integrator.implicitfast):
                            if dofs_state.ctrl_mode[i_d, i_b] <= gs.CTRL_MODE.VELOCITY:
                                qd.atomic_add(
                                    rigid_global_info.mass_mat_L_bw[0, i_pr, i_pr, i_b],
                                    -dofs_info.act_bias[I_d][2] * rigid_global_info.substep_dt[None],
                                )

                # Cholesky-Banachiewicz algorithm (in the perturbed indices), access pattern is safe for autodiff
                # https://en.wikipedia.org/wiki/Cholesky_decomposition
                for p_i0 in range(n_dofs):
                    for p_j0 in range(p_i0 + 1):
                        # j_pr <= i_pr
                        i_pr = entity_dof_start + p_i0
                        j_pr = entity_dof_start + p_j0

                        sum = gs.qd_float(0.0)
                        for p_k0 in range(p_j0):
                            # k_pr < j_pr
                            k_pr = entity_dof_start + p_k0
                            sum = sum + (
                                rigid_global_info.mass_mat_L_bw[1, i_pr, k_pr, i_b]
                                * rigid_global_info.mass_mat_L_bw[1, j_pr, k_pr, i_b]
                            )

                        a = rigid_global_info.mass_mat_L_bw[0, i_pr, j_pr, i_b] - sum
                        b = qd.math.clamp(
                            rigid_global_info.mass_mat_L_bw[1, j_pr, j_pr, i_b],
                            EPS,
                            qd.math.inf,
                        )
                        if p_i0 == p_j0:
                            rigid_global_info.mass_mat_L_bw[1, i_pr, j_pr, i_b] = qd.sqrt(
                                qd.math.clamp(a, EPS, qd.math.inf)
                            )
                        else:
                            rigid_global_info.mass_mat_L_bw[1, i_pr, j_pr, i_b] = a / b

                for i_d0 in range(n_dofs):
                    for i_d1 in range(i_d0 + 1):
                        i_d = entity_dof_start + i_d0
                        j_d = entity_dof_start + i_d1
                        i_pr = (entity_dof_start + entity_dof_end - 1) - i_d
                        j_pr = (entity_dof_start + entity_dof_end - 1) - j_d

                        a = rigid_global_info.mass_mat_L_bw[1, i_pr, i_pr, i_b]
                        rigid_global_info.mass_mat_L[i_d, j_d, i_b] = rigid_global_info.mass_mat_L_bw[
                            1, j_pr, i_pr, i_b
                        ] / qd.math.clamp(a, EPS, qd.math.inf)

                        if i_d == j_d:
                            rigid_global_info.mass_mat_D_inv[i_d, i_b] = 1.0 / (qd.math.clamp(a**2, EPS, qd.math.inf))


@qd.func
def func_solve_mass_entity(
    i_e: qd.int32,
    i_b: qd.int32,
    vec: qd.Tensor,
    out: qd.Tensor,
    out_bw: qd.template(),
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    is_backward: qd.template(),
):
    BW = qd.static(is_backward)

    if rigid_global_info.mass_mat_mask[i_e, i_b]:
        entity_dof_start = entities_info.dof_start[i_e]
        entity_dof_end = entities_info.dof_end[i_e]
        n_dofs = entities_info.n_dofs[i_e]

        # Step 1: Solve w st. L^T @ w = y
        for i_d_ in range(n_dofs):
            i_d = entity_dof_end - i_d_ - 1
            curr_out = vec[i_d, i_b]
            if qd.static(BW):
                out_bw[0, i_d, i_b] = vec[i_d, i_b]

            for j_d in range(i_d + 1, rigid_global_info.dofs_mass_block_end[i_d]):
                # Since we read out[j_d, i_b], and j_d > i_d, which means that out[j_d, i_b] is already
                # finalized at this point, we don't need to care about AD mutation rule.
                if qd.static(BW):
                    out_bw[0, i_d, i_b] = (
                        out_bw[0, i_d, i_b] - rigid_global_info.mass_mat_L[j_d, i_d, i_b] * out_bw[0, j_d, i_b]
                    )
                else:
                    curr_out = curr_out - rigid_global_info.mass_mat_L[j_d, i_d, i_b] * out[j_d, i_b]

            if qd.static(not BW):
                out[i_d, i_b] = curr_out

        # Step 2: z = D^{-1} w
        for i_d in range(entity_dof_start, entity_dof_end):
            if qd.static(BW):
                out_bw[1, i_d, i_b] = out_bw[0, i_d, i_b] * rigid_global_info.mass_mat_D_inv[i_d, i_b]
            else:
                out[i_d, i_b] = out[i_d, i_b] * rigid_global_info.mass_mat_D_inv[i_d, i_b]

        # Step 3: Solve x st. L @ x = z
        for i_d in range(entity_dof_start, entity_dof_end):
            curr_out = out[i_d, i_b]
            if qd.static(BW):
                curr_out = out_bw[1, i_d, i_b]

            for j_d in range(rigid_global_info.dofs_mass_block_start[i_d], i_d):
                curr_out = curr_out - rigid_global_info.mass_mat_L[i_d, j_d, i_b] * out[j_d, i_b]

            out[i_d, i_b] = curr_out


@qd.func
def func_solve_mass_batch(
    i_b: qd.int32,
    vec: qd.Tensor,
    out: qd.Tensor,
    out_bw: qd.template(),
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    is_backward: qd.template(),
):
    BW = qd.static(is_backward)

    # This loop is considered an inner loop
    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_0 in (
        range(rigid_global_info.n_awake_entities[i_b])
        if qd.static(static_rigid_sim_config.use_hibernation)
        else range(entities_info.n_links.shape[0])
    ):
        i_e = rigid_global_info.awake_entities[i_0, i_b] if qd.static(static_rigid_sim_config.use_hibernation) else i_0
        func_solve_mass_entity(
            i_e, i_b, vec, out, out_bw, entities_info, rigid_global_info, static_rigid_sim_config, is_backward
        )


@qd.func
def func_solve_mass(
    vec: qd.Tensor,
    out: qd.Tensor,
    out_bw: qd.template(),  # None in forward mode, real tensor in backward mode
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    is_backward: qd.template(),
):
    # This loop must be the outermost loop to be differentiable
    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_e, i_b in qd.ndrange(entities_info.n_links.shape[0], out.shape[1]):
        func_solve_mass_entity(
            i_e, i_b, vec, out, out_bw, entities_info, rigid_global_info, static_rigid_sim_config, is_backward
        )


@qd.func
def func_torque_and_passive_force(
    entities_state: array_class.EntitiesState,
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_info: array_class.JointsInfo,
    geoms_state: array_class.GeomsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    island_state: array_class.IslandState,
    is_backward: qd.template(),
):
    BW = qd.static(is_backward)

    # compute force based on each dof's ctrl mode
    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_e, i_b in qd.ndrange(entities_info.n_links.shape[0], dofs_state.ctrl_mode.shape[1]):
        EPS = rigid_global_info.EPS[None]

        wakeup = False
        for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
            I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
            if links_info.n_dofs[I_l] > 0:
                i_j = links_info.joint_start[I_l]
                I_j = [i_j, i_b] if qd.static(static_rigid_sim_config.batch_joints_info) else i_j
                joint_type = joints_info.type[I_j]

                for i_d in range(links_info.dof_start[I_l], links_info.dof_end[I_l]):
                    I_d = [i_d, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else i_d
                    force = gs.qd_float(0.0)
                    if dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.FORCE:
                        force = dofs_state.ctrl_force[i_d, i_b]
                    elif dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.VELOCITY:
                        force = -dofs_info.act_bias[I_d][2] * (dofs_state.ctrl_vel[i_d, i_b] - dofs_state.vel[i_d, i_b])
                    elif dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.POSITION and not (
                        joint_type == gs.JOINT_TYPE.FREE and i_d >= links_info.dof_start[I_l] + 3
                    ):
                        # Unified formula for GENERAL and POSITION modes, factored for float32 stability.
                        # For PD (act_gain == -act_bias[1], act_bias[0] == 0), the residual terms vanish.
                        force = (
                            dofs_info.act_gain[I_d] * (dofs_state.ctrl_pos[i_d, i_b] - dofs_state.pos[i_d, i_b])
                            + dofs_info.act_bias[I_d][0]
                            + (dofs_info.act_gain[I_d] + dofs_info.act_bias[I_d][1]) * dofs_state.pos[i_d, i_b]
                            + dofs_info.act_bias[I_d][2] * (dofs_state.vel[i_d, i_b] - dofs_state.ctrl_vel[i_d, i_b])
                        )

                    dofs_state.qf_applied[i_d, i_b] = qd.math.clamp(
                        force,
                        dofs_info.force_range[I_d][0],
                        dofs_info.force_range[I_d][1],
                    )

                    if qd.abs(force) > EPS:
                        wakeup = True

                dof_start = links_info.dof_start[I_l]
                if joint_type == gs.JOINT_TYPE.FREE and (
                    dofs_state.ctrl_mode[dof_start + 3, i_b] == gs.CTRL_MODE.POSITION
                    or dofs_state.ctrl_mode[dof_start + 4, i_b] == gs.CTRL_MODE.POSITION
                    or dofs_state.ctrl_mode[dof_start + 5, i_b] == gs.CTRL_MODE.POSITION
                ):
                    xyz = qd.Vector(
                        [
                            dofs_state.pos[0 + 3 + dof_start, i_b],
                            dofs_state.pos[1 + 3 + dof_start, i_b],
                            dofs_state.pos[2 + 3 + dof_start, i_b],
                        ],
                        dt=gs.qd_float,
                    )

                    ctrl_xyz = qd.Vector(
                        [
                            dofs_state.ctrl_pos[0 + 3 + dof_start, i_b],
                            dofs_state.ctrl_pos[1 + 3 + dof_start, i_b],
                            dofs_state.ctrl_pos[2 + 3 + dof_start, i_b],
                        ],
                        dt=gs.qd_float,
                    )

                    quat = gu.qd_xyz_to_quat(xyz)
                    ctrl_quat = gu.qd_xyz_to_quat(ctrl_xyz)

                    q_diff = gu.qd_transform_quat_by_quat(ctrl_quat, gu.qd_inv_quat(quat))
                    rotvec = gu.qd_quat_to_rotvec(q_diff, EPS)

                    for j in qd.static(range(3)):
                        i_d = dof_start + 3 + j
                        I_d = [i_d, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else i_d
                        force = (
                            dofs_info.act_gain[I_d] * rotvec[j]
                            + dofs_info.act_bias[I_d][0]
                            + (dofs_info.act_gain[I_d] + dofs_info.act_bias[I_d][1]) * dofs_state.pos[i_d, i_b]
                            + dofs_info.act_bias[I_d][2] * (dofs_state.vel[i_d, i_b] - dofs_state.ctrl_vel[i_d, i_b])
                        )

                        dofs_state.qf_applied[i_d, i_b] = qd.math.clamp(
                            force, dofs_info.force_range[I_d][0], dofs_info.force_range[I_d][1]
                        )

                        if qd.abs(force) > EPS:
                            wakeup = True

        if qd.static(static_rigid_sim_config.use_hibernation):
            if wakeup:
                # Actuation may target any sleeping component of this entity; wake each one's island (a single call
                # revives the whole island, so already-awake links are skipped).
                for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
                    if links_state.is_hibernated[i_l, i_b]:
                        func_wakeup_island(
                            island_state.links_island_idx[i_l, i_b],
                            i_b,
                            entities_state,
                            entities_info,
                            links_info,
                            dofs_state,
                            links_state,
                            geoms_state,
                            rigid_global_info,
                            island_state,
                            static_rigid_sim_config,
                        )

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        qd.ndrange(1, dofs_state.ctrl_mode.shape[1])
        if qd.static(static_rigid_sim_config.use_hibernation)
        else qd.ndrange(dofs_state.ctrl_mode.shape[0], dofs_state.ctrl_mode.shape[1])
    ):
        for i_1 in (
            range(rigid_global_info.n_awake_dofs[i_b])
            if qd.static(static_rigid_sim_config.use_hibernation)
            else qd.static(range(1))
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_dofs[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_d = (
                    rigid_global_info.awake_dofs[i_1, i_b]
                    if qd.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )

                I_d = [i_d, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else i_d
                dofs_state.qf_passive[i_d, i_b] = -dofs_info.damping[I_d] * dofs_state.vel[i_d, i_b]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        qd.ndrange(1, dofs_state.ctrl_mode.shape[1])
        if qd.static(static_rigid_sim_config.use_hibernation)
        else qd.ndrange(links_info.root_idx.shape[0], dofs_state.ctrl_mode.shape[1])
    ):
        for i_1 in (
            range(rigid_global_info.n_awake_links[i_b])
            if qd.static(static_rigid_sim_config.use_hibernation)
            else qd.static(range(1))
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_links[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_l = (
                    rigid_global_info.awake_links[i_1, i_b]
                    if qd.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )
                I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l

                if links_info.n_dofs[I_l] > 0:
                    i_j = links_info.joint_start[I_l]
                    I_j = [i_j, i_b] if qd.static(static_rigid_sim_config.batch_joints_info) else i_j
                    joint_type = joints_info.type[I_j]

                    if joint_type != gs.JOINT_TYPE.FREE and joint_type != gs.JOINT_TYPE.FIXED:
                        dof_start = links_info.dof_start[I_l]
                        dof_end = links_info.dof_end[I_l]

                        for j_d in range(dof_end - dof_start):
                            I_d = (
                                [dof_start + j_d, i_b]
                                if qd.static(static_rigid_sim_config.batch_dofs_info)
                                else dof_start + j_d
                            )
                            # Note that using dofs_state instead of qpos here allows qpos to be pulled into qpos0
                            # instead 0: dofs_state.pos = qpos - qpos0
                            func_add_safe_backward(
                                dofs_state.qf_passive,
                                [dof_start + j_d, i_b],
                                -dofs_state.pos[dof_start + j_d, i_b] * dofs_info.stiffness[I_d],
                                BW,
                            )


@qd.func
def func_update_acc(
    update_cacc: qd.template(),
    dofs_state: array_class.DofsState,
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    is_backward: qd.template(),
):
    BW = qd.static(is_backward)

    # Assume this is the outermost loop
    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        qd.ndrange(1, dofs_state.ctrl_mode.shape[1])
        if qd.static(static_rigid_sim_config.use_hibernation)
        else qd.ndrange(entities_info.n_links.shape[0], dofs_state.ctrl_mode.shape[1])
    ):
        for i_1 in (
            range(rigid_global_info.n_awake_entities[i_b])
            if qd.static(static_rigid_sim_config.use_hibernation)
            else qd.static(range(1))
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_entities[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_e = (
                    rigid_global_info.awake_entities[i_1, i_b]
                    if qd.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )

                for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
                    I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
                    i_p = links_info.parent_idx[I_l]

                    if i_p == -1:
                        links_state.cdd_vel[i_l, i_b] = -rigid_global_info.gravity[i_b] * (
                            1 - entities_info.gravity_compensation[i_e]
                        )
                        links_state.cdd_ang[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)
                        if qd.static(update_cacc):
                            links_state.cacc_lin[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)
                            links_state.cacc_ang[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)
                    else:
                        links_state.cdd_vel[i_l, i_b] = links_state.cdd_vel[i_p, i_b]
                        links_state.cdd_ang[i_l, i_b] = links_state.cdd_ang[i_p, i_b]
                        if qd.static(update_cacc):
                            links_state.cacc_lin[i_l, i_b] = links_state.cacc_lin[i_p, i_b]
                            links_state.cacc_ang[i_l, i_b] = links_state.cacc_ang[i_p, i_b]

                    for i_d in range(links_info.dof_start[I_l], links_info.dof_end[I_l]):
                        # cacc = cacc_parent + cdofdot * qvel + cdof * qacc
                        local_cdd_vel = dofs_state.cdofd_vel[i_d, i_b] * dofs_state.vel[i_d, i_b]
                        local_cdd_ang = dofs_state.cdofd_ang[i_d, i_b] * dofs_state.vel[i_d, i_b]

                        func_add_safe_backward(links_state.cdd_vel, [i_l, i_b], local_cdd_vel, BW)
                        func_add_safe_backward(links_state.cdd_ang, [i_l, i_b], local_cdd_ang, BW)
                        if qd.static(update_cacc):
                            func_add_safe_backward(
                                links_state.cacc_lin,
                                [i_l, i_b],
                                local_cdd_vel + dofs_state.cdof_vel[i_d, i_b] * dofs_state.acc[i_d, i_b],
                                BW,
                            )
                            func_add_safe_backward(
                                links_state.cacc_ang,
                                [i_l, i_b],
                                local_cdd_ang + dofs_state.cdof_ang[i_d, i_b] * dofs_state.acc[i_d, i_b],
                                BW,
                            )


@qd.func
def func_update_force(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    is_backward: qd.template(),
):
    BW = qd.static(is_backward)

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        qd.ndrange(1, links_state.pos.shape[1])
        if qd.static(static_rigid_sim_config.use_hibernation)
        else qd.ndrange(links_info.root_idx.shape[0], links_state.pos.shape[1])
    ):
        for i_1 in (
            range(rigid_global_info.n_awake_links[i_b])
            if qd.static(static_rigid_sim_config.use_hibernation)
            else qd.static(range(1))
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_links[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_l = (
                    rigid_global_info.awake_links[i_1, i_b]
                    if qd.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )

                f1_ang, f1_vel = gu.inertial_mul(
                    links_state.cinr_pos[i_l, i_b],
                    links_state.cinr_inertial[i_l, i_b],
                    links_state.cinr_mass[i_l, i_b],
                    links_state.cdd_vel[i_l, i_b],
                    links_state.cdd_ang[i_l, i_b],
                )
                f2_ang, f2_vel = gu.inertial_mul(
                    links_state.cinr_pos[i_l, i_b],
                    links_state.cinr_inertial[i_l, i_b],
                    links_state.cinr_mass[i_l, i_b],
                    links_state.cd_vel[i_l, i_b],
                    links_state.cd_ang[i_l, i_b],
                )
                f3_ang, f3_vel = gu.motion_cross_force(
                    links_state.cd_ang[i_l, i_b], links_state.cd_vel[i_l, i_b], f2_ang, f2_vel
                )

                links_state.cfrc_vel[i_l, i_b] = (
                    f1_vel + f3_vel + links_state.cfrc_applied_vel[i_l, i_b] + links_state.cfrc_coupling_vel[i_l, i_b]
                )
                links_state.cfrc_ang[i_l, i_b] = (
                    f1_ang + f3_ang + links_state.cfrc_applied_ang[i_l, i_b] + links_state.cfrc_coupling_ang[i_l, i_b]
                )

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        qd.ndrange(1, links_state.pos.shape[1])
        if qd.static(static_rigid_sim_config.use_hibernation)
        else qd.ndrange(entities_info.n_links.shape[0], links_state.pos.shape[1])
    ):
        for i_1 in (
            range(rigid_global_info.n_awake_entities[i_b])
            if qd.static(static_rigid_sim_config.use_hibernation)
            else qd.static(range(1))
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_entities[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_e = (
                    rigid_global_info.awake_entities[i_1, i_b]
                    if qd.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )

                for i_l_ in range(entities_info.n_links[i_e]):
                    i_l = entities_info.link_end[i_e] - 1 - i_l_
                    I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
                    i_p = links_info.parent_idx[I_l]
                    I_p = [i_p, i_b]
                    if i_p != -1:
                        func_add_safe_backward(links_state.cfrc_vel, I_p, links_state.cfrc_vel[i_l, i_b], BW)
                        func_add_safe_backward(links_state.cfrc_ang, I_p, links_state.cfrc_ang[i_l, i_b], BW)

    # Clear coupling forces after use
    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for I in qd.grouped(qd.ndrange(*links_state.cfrc_coupling_ang.shape)):
        links_state.cfrc_coupling_ang[I] = qd.Vector.zero(gs.qd_float, 3)
        links_state.cfrc_coupling_vel[I] = qd.Vector.zero(gs.qd_float, 3)


@qd.func
def func_actuation(self):
    if qd.static(self._use_hibernation):
        pass
    else:
        qd.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_l, i_b in qd.ndrange(self.n_links, self._B):
            I_l = [i_l, i_b] if qd.static(self._options.batch_links_info) else i_l
            for i_j in range(self.links_info.joint_start[I_l], self.links_info.joint_end[I_l]):
                I_j = [i_j, i_b] if qd.static(self._options.batch_joints_info) else i_j
                joint_type = self.joints_info.type[I_j]
                q_start = self.joints_info.q_start[I_j]

                if joint_type == gs.JOINT_TYPE.REVOLUTE or joint_type == gs.JOINT_TYPE.PRISMATIC:
                    gear = -1  # TODO
                    i_d = self.links_info.dof_start[I_l]
                    self.dofs_state.act_length[i_d, i_b] = gear * self.qpos[q_start, i_b]
                    self.dofs_state.qf_actuator[i_d, i_b] = self.dofs_state.act_length[i_d, i_b]
                else:
                    for i_d in range(self.links_info.dof_start[I_l], self.links_info.dof_end[I_l]):
                        self.dofs_state.act_length[i_d, i_b] = 0.0
                        self.dofs_state.qf_actuator[i_d, i_b] = self.dofs_state.act_length[i_d, i_b]


@qd.func
def func_bias_force(
    dofs_state: array_class.DofsState,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    is_backward: qd.template(),
):
    BW = qd.static(is_backward)

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        qd.ndrange(1, dofs_state.ctrl_mode.shape[1])
        if qd.static(static_rigid_sim_config.use_hibernation)
        else qd.ndrange(links_info.root_idx.shape[0], dofs_state.ctrl_mode.shape[1])
    ):
        for i_1 in (
            range(rigid_global_info.n_awake_links[i_b])
            if qd.static(static_rigid_sim_config.use_hibernation)
            else qd.static(range(1))
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_links[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_l = (
                    rigid_global_info.awake_links[i_1, i_b]
                    if qd.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )
                I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l

                for i_d in range(links_info.dof_start[I_l], links_info.dof_end[I_l]):
                    dofs_state.qf_bias[i_d, i_b] = dofs_state.cdof_ang[i_d, i_b].dot(
                        links_state.cfrc_ang[i_l, i_b]
                    ) + dofs_state.cdof_vel[i_d, i_b].dot(links_state.cfrc_vel[i_l, i_b])

                    dofs_state.force[i_d, i_b] = (
                        dofs_state.qf_passive[i_d, i_b] - dofs_state.qf_bias[i_d, i_b] + dofs_state.qf_applied[i_d, i_b]
                        # + self.dofs_state.qf_actuator[i_d, i_b]
                    )

                    dofs_state.qf_smooth[i_d, i_b] = dofs_state.force[i_d, i_b]


@qd.kernel
def kernel_compute_qacc(
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    is_backward: qd.template(),
):
    func_compute_qacc(
        dofs_state=dofs_state,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        is_backward=is_backward,
    )


@qd.func
def func_compute_qacc(
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    is_backward: qd.template(),
):
    BW = qd.static(is_backward)

    func_solve_mass(
        vec=dofs_state.force,
        out=dofs_state.acc_smooth,
        out_bw=dofs_state.acc_smooth_bw,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        is_backward=is_backward,
    )

    # Assume this is the outermost loop
    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_0, i_b in (
        qd.ndrange(1, dofs_state.ctrl_mode.shape[1])
        if qd.static(static_rigid_sim_config.use_hibernation)
        else qd.ndrange(entities_info.n_links.shape[0], dofs_state.ctrl_mode.shape[1])
    ):
        for i_1 in (
            range(rigid_global_info.n_awake_entities[i_b])
            if qd.static(static_rigid_sim_config.use_hibernation)
            else qd.static(range(1))
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_entities[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_e = (
                    rigid_global_info.awake_entities[i_1, i_b]
                    if qd.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )

                for i_d1_ in range(entities_info.n_dofs[i_e]):
                    i_d1 = entities_info.dof_start[i_e] + i_d1_
                    dofs_state.acc[i_d1, i_b] = dofs_state.acc_smooth[i_d1, i_b]


@qd.func
def func_integrate(
    dofs_state: array_class.DofsState,
    links_info: array_class.LinksInfo,
    joints_info: array_class.JointsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    is_backward: qd.template(),
):
    BW = qd.static(is_backward)

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        (qd.ndrange(1, dofs_state.ctrl_mode.shape[1]))
        if qd.static(static_rigid_sim_config.use_hibernation)
        else (qd.ndrange(dofs_state.ctrl_mode.shape[0], dofs_state.ctrl_mode.shape[1]))
    ):
        for i_1 in (
            range(rigid_global_info.n_awake_dofs[i_b])
            if qd.static(static_rigid_sim_config.use_hibernation)
            else qd.static(range(1))
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_dofs[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_d = (
                    rigid_global_info.awake_dofs[i_1, i_b]
                    if qd.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )

                dofs_state.vel_next[i_d, i_b] = (
                    dofs_state.vel[i_d, i_b] + dofs_state.acc[i_d, i_b] * rigid_global_info.substep_dt[None]
                )

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        (qd.ndrange(1, dofs_state.ctrl_mode.shape[1]))
        if qd.static(static_rigid_sim_config.use_hibernation)
        else (qd.ndrange(links_info.root_idx.shape[0], dofs_state.ctrl_mode.shape[1]))
    ):
        for i_1 in (
            range(rigid_global_info.n_awake_links[i_b])
            if qd.static(static_rigid_sim_config.use_hibernation)
            else qd.static(range(1))
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_links[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_l = (
                    rigid_global_info.awake_links[i_1, i_b]
                    if qd.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )
                I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
                if links_info.n_dofs[I_l] > 0:
                    EPS = rigid_global_info.EPS[None]

                    dof_start = links_info.dof_start[I_l]
                    q_start = links_info.q_start[I_l]
                    q_end = links_info.q_end[I_l]

                    i_j = links_info.joint_start[I_l]
                    I_j = [i_j, i_b] if qd.static(static_rigid_sim_config.batch_joints_info) else i_j
                    joint_type = joints_info.type[I_j]

                    if joint_type == gs.JOINT_TYPE.FREE:
                        pos = qd.Vector(
                            [
                                rigid_global_info.qpos[q_start, i_b],
                                rigid_global_info.qpos[q_start + 1, i_b],
                                rigid_global_info.qpos[q_start + 2, i_b],
                            ]
                        )
                        vel = qd.Vector(
                            [
                                dofs_state.vel_next[dof_start, i_b],
                                dofs_state.vel_next[dof_start + 1, i_b],
                                dofs_state.vel_next[dof_start + 2, i_b],
                            ]
                        )
                        # Backward pass requires atomic add
                        if qd.static(BW):
                            qd.atomic_add(pos, vel * rigid_global_info.substep_dt[None])
                        else:
                            pos = pos + vel * rigid_global_info.substep_dt[None]
                        for j in qd.static(range(3)):
                            rigid_global_info.qpos_next[q_start + j, i_b] = pos[j]
                    if joint_type == gs.JOINT_TYPE.SPHERICAL or joint_type == gs.JOINT_TYPE.FREE:
                        rot_offset = 3 if joint_type == gs.JOINT_TYPE.FREE else 0
                        rot0 = qd.Vector(
                            [
                                rigid_global_info.qpos[q_start + rot_offset + 0, i_b],
                                rigid_global_info.qpos[q_start + rot_offset + 1, i_b],
                                rigid_global_info.qpos[q_start + rot_offset + 2, i_b],
                                rigid_global_info.qpos[q_start + rot_offset + 3, i_b],
                            ]
                        )
                        ang = (
                            qd.Vector(
                                [
                                    dofs_state.vel_next[dof_start + rot_offset + 0, i_b],
                                    dofs_state.vel_next[dof_start + rot_offset + 1, i_b],
                                    dofs_state.vel_next[dof_start + rot_offset + 2, i_b],
                                ]
                            )
                            * rigid_global_info.substep_dt[None]
                        )
                        qrot = gu.qd_rotvec_to_quat(ang, EPS)
                        rot = gu.qd_transform_quat_by_quat(qrot, rot0)
                        for j in qd.static(range(4)):
                            rigid_global_info.qpos_next[q_start + j + rot_offset, i_b] = rot[j]
                    else:
                        for j_ in range(q_end - q_start):
                            j = q_start + j_
                            if j < q_end:
                                rigid_global_info.qpos_next[j, i_b] = (
                                    rigid_global_info.qpos[j, i_b]
                                    + dofs_state.vel_next[dof_start + j_, i_b] * rigid_global_info.substep_dt[None]
                                )


@qd.kernel
def kernel_forward_dynamics_without_qacc(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    joints_info: array_class.JointsInfo,
    entities_state: array_class.EntitiesState,
    entities_info: array_class.EntitiesInfo,
    geoms_state: array_class.GeomsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    island_state: array_class.IslandState,
    is_backward: qd.template(),
):
    func_compute_mass_matrix(
        implicit_damping=qd.static(static_rigid_sim_config.integrator == gs.integrator.approximate_implicitfast),
        links_state=links_state,
        links_info=links_info,
        dofs_state=dofs_state,
        dofs_info=dofs_info,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        is_backward=is_backward,
    )
    func_factor_mass(
        implicit_damping=False,
        entities_info=entities_info,
        dofs_state=dofs_state,
        dofs_info=dofs_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        is_backward=is_backward,
    )
    func_torque_and_passive_force(
        entities_state=entities_state,
        entities_info=entities_info,
        dofs_state=dofs_state,
        dofs_info=dofs_info,
        links_state=links_state,
        links_info=links_info,
        joints_info=joints_info,
        geoms_state=geoms_state,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        island_state=island_state,
        is_backward=is_backward,
    )
    func_update_acc(
        update_cacc=False,
        dofs_state=dofs_state,
        links_info=links_info,
        links_state=links_state,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        is_backward=is_backward,
    )
    func_update_force(
        links_state=links_state,
        links_info=links_info,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        is_backward=is_backward,
    )
    func_bias_force(
        dofs_state=dofs_state,
        links_state=links_state,
        links_info=links_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        is_backward=is_backward,
    )


@qd.func
def func_implicit_damping(
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    is_backward: qd.template(),
):
    BW = qd.static(is_backward)

    EPS = rigid_global_info.EPS[None]

    n_entities = entities_info.dof_start.shape[0]
    _B = dofs_state.ctrl_mode.shape[1]

    # Determine whether the mass matrix must be re-computed to take into account first-order correction terms.
    # Note that avoiding inverting the mass matrix twice would not only speed up simulation but also improving
    # numerical stability as computing post-damping accelerations from forces is not necessary anymore.
    if qd.static(
        not static_rigid_sim_config.enable_mujoco_compatibility
        or static_rigid_sim_config.integrator == gs.integrator.Euler
    ):
        for i_e, i_b in qd.ndrange(n_entities, _B):
            rigid_global_info.mass_mat_mask[i_e, i_b] = False

        qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_e, i_b in qd.ndrange(n_entities, _B):
            entity_dof_start = entities_info.dof_start[i_e]
            entity_dof_end = entities_info.dof_end[i_e]
            for i_d_ in range(entity_dof_start, entity_dof_end):
                i_d = i_d_
                if i_d < entity_dof_end:
                    I_d = [i_d, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else i_d
                    if dofs_info.damping[I_d] > EPS:
                        rigid_global_info.mass_mat_mask[i_e, i_b] = True
                    if qd.static(static_rigid_sim_config.integrator != gs.integrator.Euler):
                        if (
                            dofs_state.ctrl_mode[i_d, i_b] <= gs.CTRL_MODE.VELOCITY
                            and qd.abs(dofs_info.act_bias[I_d][2]) > EPS
                        ):
                            rigid_global_info.mass_mat_mask[i_e, i_b] = True

    func_factor_mass(
        implicit_damping=True,
        entities_info=entities_info,
        dofs_state=dofs_state,
        dofs_info=dofs_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        is_backward=is_backward,
    )
    func_solve_mass(
        vec=dofs_state.force,
        out=dofs_state.acc,
        out_bw=dofs_state.acc_bw,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        is_backward=is_backward,
    )

    # Disable pre-computed factorization mask right away
    if qd.static(
        not static_rigid_sim_config.enable_mujoco_compatibility
        or static_rigid_sim_config.integrator == gs.integrator.Euler
    ):
        for i_e, i_b in qd.ndrange(n_entities, _B):
            rigid_global_info.mass_mat_mask[i_e, i_b] = True


from genesis.utils.deprecated_module_wrapper import create_virtual_deprecated_module

create_virtual_deprecated_module(__name__, "genesis.engine.solvers.rigid.rigid_solver_dynamics_decomp")
