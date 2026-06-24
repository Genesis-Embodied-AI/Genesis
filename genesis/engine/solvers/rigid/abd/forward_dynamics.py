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
    func_wakeup_entity_and_its_temp_island,
    func_check_index_range,
    func_add_safe_backward,
)

# Block size (warp width) for the cooperative mass_mat_assemble path. Used only when constraint_layout_transposed=True
# (and not use_hibernation). One warp per (entity, env); lanes stride i_d_ within the entity dof block to coalesce the
# flipped mass_mat writes.
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
    contact_island_state: array_class.ContactIslandState,
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
        contact_island_state=contact_island_state,
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
    contact_island_state: array_class.ContactIslandState,
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
        contact_island_state=contact_island_state,
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
def _linear_to_lower_tri(i_pair: qd.i32):
    """Linear index -> (row, col) of a lower-triangular matrix including diagonal.
    Sequence: (0,0), (1,0), (1,1), (2,0), (2,1), (2,2), ...
    Uses f32 sqrt (fast on all backends) with integer post-correction for
    GPUs whose sqrt is not correctly rounded on perfect squares.
    """
    i_d = qd.cast(qd.floor((qd.sqrt(qd.cast(8 * i_pair + 1, qd.f32)) - 1.0) / 2.0), qd.i32)
    if (i_d + 1) * (i_d + 2) // 2 <= i_pair:
        i_d = i_d + 1
    j_d = i_pair - i_d * (i_d + 1) // 2
    return i_d, j_d


@qd.func
def func_compute_mass_matrix_lds(
    implicit_damping: qd.template(),
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

    # LDS-optimized GPU implementation using a fixed block size and the
    # configured per-entity tiled DoF bound.
    BLOCK_DIM = qd.static(64)
    MAX_DOFS_PER_ENTITY = qd.static(static_rigid_sim_config.tiled_n_dofs_per_entity)

    n_entities = static_rigid_sim_config.n_entities_
    _B = static_rigid_sim_config.n_envs
    # OPT-1: when a contiguous dynamic-entity window is set (hibernation off), launch only over that
    # window and offset the entity index, skipping static 0-DOF entities.
    _HAS_WINDOW = qd.static(
        static_rigid_sim_config.n_dynamic_entities_ > 0 and not static_rigid_sim_config.use_hibernation
    )
    _ENTITY_BASE = qd.static(static_rigid_sim_config.dynamic_entity_offset_ if _HAS_WINDOW else 0)
    n_thread_entities = (
        qd.static(static_rigid_sim_config.n_dynamic_entities_)
        if _HAS_WINDOW
        else (static_rigid_sim_config.n_entities_ if qd.static(static_rigid_sim_config.use_hibernation) else n_entities)
    )

    qd.loop_config(block_dim=BLOCK_DIM)
    for i in range(n_thread_entities * _B * BLOCK_DIM):
        tid = i % BLOCK_DIM
        i_e_local = (i // BLOCK_DIM) % n_thread_entities
        i_b = i // (BLOCK_DIM * n_thread_entities)

        if i_b >= _B:
            continue

        i_e = i_e_local + _ENTITY_BASE

        if qd.static(static_rigid_sim_config.use_hibernation):
            if not func_check_index_range(
                i_e_local, 0, static_rigid_sim_config.n_entities_, static_rigid_sim_config.use_hibernation
            ):
                continue
            i_e = rigid_global_info.awake_entities[i_e_local, i_b]
        entity_dof_start = entities_info.dof_start[i_e]
        entity_dof_end = entities_info.dof_end[i_e]
        n_dofs = entities_info.n_dofs[i_e]

        # This kernel uses a packed lower-triangular mass matrix in shared memory.
        # Total bytes = 4 * (n*(n+1)/2 + 12n), where:
        #   - n*(n+1)/2  comes from mass_mat_packed (1D lower-triangle, OPT-2 vs old n^2)
        #   - 12n        comes from 4 vector caches of shape [n, 3]
        # For n=49: 1225 + 12*49 = 1813 floats = 7252B (vs old 2401+588*4 = 11,956B).
        # Constraining that to ~64KB gives n <= 103 for 4-byte floats (OPT-2 limit).
        KERNEL_MAX_DOFS_PER_ENTITY = qd.static(122 if MAX_DOFS_PER_ENTITY > 122 else MAX_DOFS_PER_ENTITY)

        if n_dofs <= 0 or n_dofs > KERNEL_MAX_DOFS_PER_ENTITY:
            continue

        f_ang_cache = qd.simt.block.SharedArray((KERNEL_MAX_DOFS_PER_ENTITY, 3), gs.qd_float)
        f_vel_cache = qd.simt.block.SharedArray((KERNEL_MAX_DOFS_PER_ENTITY, 3), gs.qd_float)
        cdof_ang_cache = qd.simt.block.SharedArray((KERNEL_MAX_DOFS_PER_ENTITY, 3), gs.qd_float)
        cdof_vel_cache = qd.simt.block.SharedArray((KERNEL_MAX_DOFS_PER_ENTITY, 3), gs.qd_float)
        # Packed lower-triangular storage. Only the lower triangle of this tile is ever
        # written or read (the symmetric mirror below reads global memory, not this LDS),
        # so storing n(n+1)/2 instead of n*n ~halves the dominant LDS term, raising
        # occupancy (this kernel is occupancy/latency-bound: VALU ~19%, LDS-capped).
        mass_mat_local = qd.simt.block.SharedArray(
            (KERNEL_MAX_DOFS_PER_ENTITY * (KERNEL_MAX_DOFS_PER_ENTITY + 1) // 2,), gs.qd_float
        )

        # Cooperative loading into LDS
        for load_round in range((n_dofs + BLOCK_DIM - 1) // BLOCK_DIM):
            i_d_local = load_round * BLOCK_DIM + tid
            if i_d_local < n_dofs:
                i_d_global = entity_dof_start + i_d_local

                # Coalesced loading of all vector components
                f_ang_cache[i_d_local, 0] = dofs_state.f_ang[i_d_global, i_b][0]
                f_ang_cache[i_d_local, 1] = dofs_state.f_ang[i_d_global, i_b][1]
                f_ang_cache[i_d_local, 2] = dofs_state.f_ang[i_d_global, i_b][2]

                f_vel_cache[i_d_local, 0] = dofs_state.f_vel[i_d_global, i_b][0]
                f_vel_cache[i_d_local, 1] = dofs_state.f_vel[i_d_global, i_b][1]
                f_vel_cache[i_d_local, 2] = dofs_state.f_vel[i_d_global, i_b][2]

                cdof_ang_cache[i_d_local, 0] = dofs_state.cdof_ang[i_d_global, i_b][0]
                cdof_ang_cache[i_d_local, 1] = dofs_state.cdof_ang[i_d_global, i_b][1]
                cdof_ang_cache[i_d_local, 2] = dofs_state.cdof_ang[i_d_global, i_b][2]

                cdof_vel_cache[i_d_local, 0] = dofs_state.cdof_vel[i_d_global, i_b][0]
                cdof_vel_cache[i_d_local, 1] = dofs_state.cdof_vel[i_d_global, i_b][1]
                cdof_vel_cache[i_d_local, 2] = dofs_state.cdof_vel[i_d_global, i_b][2]

        qd.simt.block.sync()  # Ensure all data is loaded

        # Compute mass matrix using LDS - optimal performance
        n_pairs = n_dofs * (n_dofs + 1) // 2
        pair_idx = tid

        while pair_idx < n_pairs:
            # Convert linear index to (i, j) lower triangular
            i_d_, j_d_ = _linear_to_lower_tri(pair_idx)

            # Fast LDS-based computation
            ang_dot = (f_ang_cache[i_d_, 0] * cdof_ang_cache[j_d_, 0] +
                      f_ang_cache[i_d_, 1] * cdof_ang_cache[j_d_, 1] +
                      f_ang_cache[i_d_, 2] * cdof_ang_cache[j_d_, 2])

            vel_dot = (f_vel_cache[i_d_, 0] * cdof_vel_cache[j_d_, 0] +
                      f_vel_cache[i_d_, 1] * cdof_vel_cache[j_d_, 1] +
                      f_vel_cache[i_d_, 2] * cdof_vel_cache[j_d_, 2])

            # Store in packed lower-triangular LDS (pair_idx == i_d_*(i_d_+1)/2 + j_d_)
            mass_mat_local[pair_idx] = ang_dot + vel_dot
            pair_idx += BLOCK_DIM

        qd.simt.block.sync()

        # Write results to global memory with masking
        global_pair_idx = tid
        while global_pair_idx < n_pairs:
            i_d_ = qd.cast(qd.floor((qd.sqrt(qd.cast(8 * global_pair_idx + 1, qd.f32)) - 1.0) / 2.0), qd.i32)
            if (i_d_ + 1) * (i_d_ + 2) // 2 <= global_pair_idx:
                i_d_ = i_d_ + 1
            j_d_ = global_pair_idx - i_d_ * (i_d_ + 1) // 2

            i_d_global = entity_dof_start + i_d_
            j_d_global = entity_dof_start + j_d_

            # Apply masking and store (global_pair_idx == i_d_*(i_d_+1)/2 + j_d_ = packed index)
            rigid_global_info.mass_mat[i_b, i_d_global, j_d_global] = (
                mass_mat_local[global_pair_idx] * rigid_global_info.mass_parent_mask[i_d_global, j_d_global]
            )

            global_pair_idx += BLOCK_DIM

        # Mirror upper triangle for symmetric matrix in both forward and backward passes
        qd.simt.block.sync()  # Ensure lower-triangle stores are complete

        n_upper_pairs = n_dofs * (n_dofs - 1) // 2
        upper_idx = tid

        while upper_idx < n_upper_pairs:
            # Convert to upper triangle indices
            i_d_ = qd.cast(qd.floor((qd.sqrt(qd.cast(8 * upper_idx + 1, qd.f32)) + 1.0) / 2.0), qd.i32)
            if i_d_ * (i_d_ + 1) // 2 <= upper_idx:
                i_d_ = i_d_ + 1
            j_d_ = upper_idx - i_d_ * (i_d_ - 1) // 2

            i_d_global = entity_dof_start + j_d_  # Note: swapped for upper triangle
            j_d_global = entity_dof_start + i_d_

            # Mirror from lower triangle
            rigid_global_info.mass_mat[i_b, i_d_global, j_d_global] = rigid_global_info.mass_mat[i_b, j_d_global, i_d_global]

            upper_idx += BLOCK_DIM


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
        static_rigid_sim_config.enable_tiled_cholesky_mass_matrix and static_rigid_sim_config.backend != gs.cpu
    ):
        # LDS-fused assembly (the fork's AMD-optimized path; staged f_ang/f_vel/cdof in shared memory and
        # computes the lower triangle once, then mirrors). ~1.5x faster than the no-LDS cooperative writer
        # below on gfx942 (mass_mat_assemble 224ms -> ~146ms in the G68 baseline).
        func_compute_mass_matrix_lds(
            implicit_damping=implicit_damping,
            links_state=links_state,
            links_info=links_info,
            dofs_state=dofs_state,
            dofs_info=dofs_info,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
            is_backward=BW,
        )
    elif qd.static(static_rigid_sim_config.constraint_layout_transposed and not static_rigid_sim_config.use_hibernation):
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
                # + j_d_, with j_d_ in [0, i_d_].
                i_d_ = qd.cast((qd.sqrt(8 * i_pair + 1) - 1) // 2, qd.i32)
                j_d_ = i_pair - i_d_ * (i_d_ + 1) // 2
                i_d = d_s + i_d_
                j_d = d_s + j_d_
                val = (
                    dofs_state.f_ang[i_d, i_b].dot(dofs_state.cdof_ang[j_d, i_b])
                    + dofs_state.f_vel[i_d, i_b].dot(dofs_state.cdof_vel[j_d, i_b])
                ) * rigid_global_info.mass_parent_mask[i_d, j_d]
                rigid_global_info.mass_mat[i_b, i_d, j_d] = val
                if i_d_ != j_d_:
                    rigid_global_info.mass_mat[i_b, j_d, i_d] = val
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

                    for i_d, j_d in qd.ndrange(
                        (entities_info.dof_start[i_e], entities_info.dof_end[i_e]),
                        (entities_info.dof_start[i_e], entities_info.dof_end[i_e]),
                    ):
                        val = (
                            dofs_state.f_ang[i_d, i_b].dot(dofs_state.cdof_ang[j_d, i_b])
                            + dofs_state.f_vel[i_d, i_b].dot(dofs_state.cdof_vel[j_d, i_b])
                        ) * rigid_global_info.mass_parent_mask[i_d, j_d]
                        rigid_global_info.mass_mat[i_b, i_d, j_d] = val

                    if qd.static(not BW):
                        _e_start_m = entities_info.dof_start[i_e]
                        _e_nd = entities_info.n_dofs[i_e]
                        _n_upper = _e_nd * (_e_nd - 1) // 2
                        for _pair_idx in range(_n_upper):
                            _row = qd.cast((qd.sqrt(8.0 * qd.cast(_pair_idx, gs.qd_float) + 1.0) + 1.0) // 2.0, qd.i32)
                            _col = _pair_idx - _row * (_row - 1) // 2
                            rigid_global_info.mass_mat[i_b, _e_start_m + _col, _e_start_m + _row] = rigid_global_info.mass_mat[i_b, _e_start_m + _row, _e_start_m + _col]
                    else:
                        for i_d_, j_d_ in qd.static(
                            qd.ndrange(
                                static_rigid_sim_config.max_n_dofs_per_entity,
                                static_rigid_sim_config.max_n_dofs_per_entity,
                            )
                        ):
                            i_d = entities_info.dof_start[i_e] + i_d_
                            j_d = entities_info.dof_start[i_e] + j_d_

                            if i_d < entities_info.dof_end[i_e] and j_d < entities_info.dof_end[i_e] and j_d > i_d:
                                rigid_global_info.mass_mat[i_b, i_d, j_d] = rigid_global_info.mass_mat[i_b, j_d, i_d]

    # Take into account motor armature
    qd.loop_config(name="armature", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in qd.ndrange(dofs_state.f_ang.shape[0], links_state.pos.shape[1]):
        I_d = [i_d, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else i_d
        func_add_safe_backward(rigid_global_info.mass_mat, (i_b, i_d, i_d), dofs_info.armature[I_d], BW)

    # Take into account first-order correction terms for implicit integration scheme right away
    if qd.static(implicit_damping):
        qd.loop_config(name="impint_order_1_corr", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_d, i_b in qd.ndrange(dofs_state.f_ang.shape[0], links_state.pos.shape[1]):
            I_d = [i_d, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else i_d
            # Single write: combine damping and (conditional) act_bias correction to avoid
            # a double-write on a needs_grad field, which would drop the first write's gradient.
            rigid_global_info.mass_mat[i_b, i_d, i_d] = (
                rigid_global_info.mass_mat[i_b, i_d, i_d]
                + dofs_info.damping[I_d] * rigid_global_info.substep_dt[None]
                - dofs_info.act_bias[I_d][2]
                * rigid_global_info.substep_dt[None]
                * (1.0 if dofs_state.ctrl_mode[i_d, i_b] <= gs.CTRL_MODE.VELOCITY else 0.0)
            )


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

        if qd.static(
            not static_rigid_sim_config.enable_tiled_cholesky_mass_matrix or static_rigid_sim_config.backend == gs.cpu
        ):
            qd.loop_config(name="factor_mass", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
            for i_e, i_b in qd.ndrange(n_entities, _B):
                if rigid_global_info.mass_mat_mask[i_e, i_b]:
                    entity_dof_start = entities_info.dof_start[i_e]
                    entity_dof_end = entities_info.dof_end[i_e]
                    n_dofs = entities_info.n_dofs[i_e]

                    for i_d in range(entity_dof_start, entity_dof_end):
                        for j_d in range(entity_dof_start, i_d + 1):
                            rigid_global_info.mass_mat_L[i_b, i_d, j_d] = rigid_global_info.mass_mat[i_b, i_d, j_d]

                        if qd.static(implicit_damping):
                            I_d = [i_d, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else i_d
                            diag_delta = dofs_info.damping[I_d] * rigid_global_info.substep_dt[None]
                            if qd.static(static_rigid_sim_config.integrator == gs.integrator.implicitfast):
                                # Single write: fold act_bias correction into the same expression to avoid
                                # a double-write on a needs_grad field, which would drop the first gradient.
                                diag_delta = diag_delta - dofs_info.act_bias[I_d][2] * rigid_global_info.substep_dt[
                                    None
                                ] * (1.0 if dofs_state.ctrl_mode[i_d, i_b] <= gs.CTRL_MODE.VELOCITY else 0.0)
                            rigid_global_info.mass_mat_L[i_b, i_d, i_d] = (
                                rigid_global_info.mass_mat_L[i_b, i_d, i_d] + diag_delta
                            )

                    for i_d_ in range(n_dofs):
                        i_d = entity_dof_end - i_d_ - 1
                        D_inv = 1.0 / rigid_global_info.mass_mat_L[i_b, i_d, i_d]
                        rigid_global_info.mass_mat_D_inv[i_d, i_b] = D_inv

                        for j_d_ in range(i_d - entity_dof_start):
                            j_d = i_d - j_d_ - 1
                            a = rigid_global_info.mass_mat_L[i_b, i_d, j_d] * D_inv
                            for k_d in range(entity_dof_start, j_d + 1):
                                rigid_global_info.mass_mat_L[i_b, j_d, k_d] -= (
                                    a * rigid_global_info.mass_mat_L[i_b, i_d, k_d]
                                )
                            rigid_global_info.mass_mat_L[i_b, i_d, j_d] = a

                        # FIXME: Diagonal coeffs of L are ignored in computations, so no need to update them.
                        rigid_global_info.mass_mat_L[i_b, i_d, i_d] = 1.0
        else:
            BLOCK_DIM = qd.static(64)
            MAX_DOFS_PER_ENTITY = qd.static(static_rigid_sim_config.tiled_n_dofs_per_entity)
            WARP_SIZE = qd.static(64)

            # OPT-1: restrict the (entity, env) grid to the contiguous dynamic-entity window when set,
            # skipping static 0-DOF entities (their factor is a no-op but still launches a block).
            _HAS_WINDOW = qd.static(static_rigid_sim_config.n_dynamic_entities_ > 0)
            _ENTITY_BASE = qd.static(static_rigid_sim_config.dynamic_entity_offset_ if _HAS_WINDOW else 0)
            n_grid_entities = qd.static(static_rigid_sim_config.n_dynamic_entities_) if _HAS_WINDOW else n_entities

            qd.loop_config(name="factor_mass", block_dim=BLOCK_DIM)
            for i in range(n_grid_entities * _B * BLOCK_DIM):
                tid = i % BLOCK_DIM
                i_e = (i // BLOCK_DIM) % n_grid_entities + _ENTITY_BASE
                i_b = i // (BLOCK_DIM * n_grid_entities)
                if i_b >= _B:
                    continue

                if rigid_global_info.mass_mat_mask[i_e, i_b]:
                    entity_dof_start = entities_info.dof_start[i_e]
                    entity_dof_end = entities_info.dof_end[i_e]
                    n_dofs = entities_info.n_dofs[i_e]
                    n_lower_tri = n_dofs * (n_dofs + 1) // 2

                    # OPT-2: use 1D packed lower-triangle storage for mass_mat to reduce LDS.
                    # mass_mat_packed[i_d_*(i_d_+1)//2 + j_d_] = mass_mat[i_d_, j_d_] (lower tri only).
                    # LDS saved: MAX_DOFS*(MAX_DOFS+1)*4B → MAX_DOFS*(MAX_DOFS+1)//2*4B for n=49:
                    # 49*50*4=9800B → 1225*4=4900B; total LDS: ~10KB → ~5.1KB → more WGs per CU.
                    MAX_LOWER_TRI_CHOL = qd.static(MAX_DOFS_PER_ENTITY * (MAX_DOFS_PER_ENTITY + 1) // 2)
                    mass_mat = qd.simt.block.SharedArray((MAX_LOWER_TRI_CHOL,), gs.qd_float)

                    # Load phase: i_pair is the 1D packed index; write directly without 2D encode.
                    i_pair = tid
                    while i_pair < n_lower_tri:
                        i_d_ = qd.cast((qd.sqrt(8 * i_pair + 1) - 1) // 2, qd.i32)
                        j_d_ = i_pair - i_d_ * (i_d_ + 1) // 2
                        i_d = entity_dof_start + i_d_
                        j_d = entity_dof_start + j_d_
                        mass_mat[i_pair] = rigid_global_info.mass_mat[i_b, i_d, j_d]
                        i_pair = i_pair + BLOCK_DIM
                    qd.simt.block.sync()

                    if qd.static(implicit_damping):
                        i_d_ = tid
                        while i_d_ < n_dofs:
                            i_d = entity_dof_start + i_d_
                            I_d = [i_d, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else i_d
                            diag_idx = i_d_ * (i_d_ + 1) // 2 + i_d_
                            mass_mat[diag_idx] = (
                                mass_mat[diag_idx] + dofs_info.damping[I_d] * rigid_global_info.substep_dt[None]
                            )
                            if qd.static(static_rigid_sim_config.integrator == gs.integrator.implicitfast):
                                if dofs_state.ctrl_mode[i_d, i_b] <= gs.CTRL_MODE.VELOCITY:
                                    mass_mat[diag_idx] = (
                                        mass_mat[diag_idx]
                                        - dofs_info.act_bias[I_d][2] * rigid_global_info.substep_dt[None]
                                    )
                            i_d_ = i_d_ + BLOCK_DIM
                        qd.simt.block.sync()

                    # Pivot-row cache sized to the LDS tile so it is valid even when an entity
                    # has more dofs than the wave width (n_dofs > BLOCK_DIM); the strided loops
                    # below then cover the full row.
                    sh_pivot = qd.simt.block.SharedArray((MAX_DOFS_PER_ENTITY,), gs.qd_float)

                    for j in range(n_dofs):
                        i_d_ = n_dofs - j - 1
                        i_d = entity_dof_end - j - 1

                        # Diagonal element in packed 1D: index = i_d_*(i_d_+1)//2 + i_d_
                        diag_packed = i_d_ * (i_d_ + 1) // 2 + i_d_
                        D_inv = 1.0 / mass_mat[diag_packed]
                        if tid == 0:
                            rigid_global_info.mass_mat_D_inv[i_d, i_b] = D_inv
                            # FIXME: Diagonal coeffs of L are ignored in computations, so no need to update them.
                            rigid_global_info.mass_mat_L[i_b, i_d, i_d] = 1.0

                        # Pivot row base index in packed 1D: i_d_*(i_d_+1)//2
                        # Elements i_d_*(i_d_+1)//2 .. i_d_*(i_d_+1)//2 + (i_d_-1) are row i_d_, cols 0..i_d_-1.
                        piv_base = i_d_ * (i_d_ + 1) // 2

                        # Cache the original pivot row into LDS before it is overwritten. On a
                        # wave64 block (BLOCK_DIM=WARP_SIZE=64) all threads run in lockstep, so the
                        # cache is visible to the cross-thread reads below without an extra barrier.
                        # Fast single-store path when the tile fits one wave (the common case);
                        # strided fallback keeps it correct for entities wider than BLOCK_DIM.
                        if qd.static(MAX_DOFS_PER_ENTITY <= BLOCK_DIM):
                            if tid < i_d_:
                                sh_pivot[tid] = mass_mat[piv_base + tid]
                        else:
                            _p = tid
                            while _p < i_d_:
                                sh_pivot[_p] = mass_mat[piv_base + _p]
                                _p = _p + BLOCK_DIM

                        # Row-major rank-1 update: each thread owns rows [tid, tid+BLOCK_DIM, ...].
                        # For row _r: packed base = _r*(_r+1)//2; elements are at base+_c for _c=0.._r.
                        # As _c increments by 1, the packed index increments by 1 — no extra multiply needed.
                        _r = tid
                        while _r < i_d_:
                            piv_r = sh_pivot[_r] * D_inv
                            row_base = _r * (_r + 1) // 2
                            _c = 0
                            while _c <= _r:
                                mass_mat[row_base + _c] = mass_mat[row_base + _c] - piv_r * sh_pivot[_c]
                                _c = _c + 1
                            _r = _r + BLOCK_DIM

                        # Write L factors back to the pivot row.
                        if qd.static(MAX_DOFS_PER_ENTITY <= BLOCK_DIM):
                            if tid < i_d_:
                                mass_mat[piv_base + tid] = sh_pivot[tid] * D_inv
                        else:
                            _p = tid
                            while _p < i_d_:
                                mass_mat[piv_base + _p] = sh_pivot[_p] * D_inv
                                _p = _p + BLOCK_DIM

                        if qd.static(
                            static_rigid_sim_config.backend == gs.cuda
                            or static_rigid_sim_config.backend == gs.amdgpu
                        ):
                            if i_d_ <= WARP_SIZE:
                                qd.simt.warp.sync(qd.u32(0xFFFFFFFF))
                            else:
                                qd.simt.block.sync()
                        else:
                            qd.simt.block.sync()

                    # HBM write-back: i_pair is again the 1D packed index, read mass_mat[i_pair] directly.
                    i_pair = tid
                    n_strict_lower_tri = n_dofs * (n_dofs - 1) // 2
                    while i_pair < n_strict_lower_tri:
                        i_d_ = qd.cast((qd.sqrt(8 * i_pair + 1) + 1) // 2, qd.i32)
                        j_d_ = i_pair - i_d_ * (i_d_ - 1) // 2
                        i_d = entity_dof_start + i_d_
                        j_d = entity_dof_start + j_d_
                        rigid_global_info.mass_mat_L[i_b, i_d, j_d] = mass_mat[i_d_ * (i_d_ + 1) // 2 + j_d_]
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

                for i_d0 in (
                    range(n_dofs)
                    if qd.static(not BW)
                    else qd.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                ):
                    if func_check_index_range(i_d0, 0, n_dofs, BW):
                        i_d = entity_dof_start + i_d0
                        i_pr = (entity_dof_start + entity_dof_end - 1) - i_d
                        for j_d_ in (
                            range(entity_dof_start, i_d + 1)
                            if qd.static(not BW)
                            else qd.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                        ):
                            j_d = j_d_ if qd.static(not BW) else (j_d_ + entities_info.dof_start[i_e])
                            j_pr = (entity_dof_start + entity_dof_end - 1) - j_d
                            if func_check_index_range(j_d, entity_dof_start, i_d + 1, BW):
                                rigid_global_info.mass_mat_L_bw[0, i_b, i_pr, j_pr] = rigid_global_info.mass_mat[
                                    i_b, i_d, j_d
                                ]
                                rigid_global_info.mass_mat_L_bw[0, i_b, j_pr, i_pr] = rigid_global_info.mass_mat[
                                    i_b, i_d, j_d
                                ]

                        if qd.static(implicit_damping):
                            I_d = [i_d, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else i_d
                            qd.atomic_add(
                                rigid_global_info.mass_mat_L_bw[0, i_b, i_pr, i_pr],
                                (dofs_info.damping[I_d] * rigid_global_info.substep_dt[None]),
                            )
                            if qd.static(static_rigid_sim_config.integrator == gs.integrator.implicitfast):
                                if dofs_state.ctrl_mode[i_d, i_b] <= gs.CTRL_MODE.VELOCITY:
                                    qd.atomic_add(
                                        rigid_global_info.mass_mat_L_bw[0, i_b, i_pr, i_pr],
                                        -dofs_info.act_bias[I_d][2] * rigid_global_info.substep_dt[None],
                                    )

                # Cholesky-Banachiewicz algorithm (in the perturbed indices), access pattern is safe for autodiff
                # https://en.wikipedia.org/wiki/Cholesky_decomposition
                for p_i0 in (
                    range(n_dofs)
                    if qd.static(not BW)
                    else qd.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                ):
                    for p_j0 in (
                        range(p_i0 + 1)
                        if qd.static(not BW)
                        else qd.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                    ):
                        if func_check_index_range(p_i0, 0, n_dofs, BW) and func_check_index_range(p_j0, 0, p_i0 + 1, BW):
                            # j_pr <= i_pr
                            i_pr = entity_dof_start + p_i0
                            j_pr = entity_dof_start + p_j0

                            sum = gs.qd_float(0.0)
                            for p_k0 in (
                                range(p_j0)
                                if qd.static(not BW)
                                else qd.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                            ):
                                # k_pr < j_pr
                                if func_check_index_range(p_k0, 0, p_j0, BW):
                                    k_pr = entity_dof_start + p_k0
                                    sum = sum + (
                                        rigid_global_info.mass_mat_L_bw[1, i_b, i_pr, k_pr]
                                        * rigid_global_info.mass_mat_L_bw[1, i_b, j_pr, k_pr]
                                    )

                            a = rigid_global_info.mass_mat_L_bw[0, i_b, i_pr, j_pr] - sum
                            b = qd.math.clamp(
                                rigid_global_info.mass_mat_L_bw[1, i_b, j_pr, j_pr],
                                EPS,
                                qd.math.inf,
                            )
                            if p_i0 == p_j0:
                                rigid_global_info.mass_mat_L_bw[1, i_b, i_pr, j_pr] = qd.sqrt(
                                    qd.math.clamp(a, EPS, qd.math.inf)
                                )
                            else:
                                rigid_global_info.mass_mat_L_bw[1, i_b, i_pr, j_pr] = a / b

                for i_d0 in (
                    range(n_dofs)
                    if qd.static(not BW)
                    else qd.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                ):
                    for i_d1 in (
                        range(i_d0 + 1)
                        if qd.static(not BW)
                        else qd.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                    ):
                        if func_check_index_range(i_d0, 0, n_dofs, BW) and func_check_index_range(i_d1, 0, i_d0 + 1, BW):
                            i_d = entity_dof_start + i_d0
                            j_d = entity_dof_start + i_d1
                            i_pr = (entity_dof_start + entity_dof_end - 1) - i_d
                            j_pr = (entity_dof_start + entity_dof_end - 1) - j_d

                            a = rigid_global_info.mass_mat_L_bw[1, i_b, i_pr, i_pr]
                            rigid_global_info.mass_mat_L[i_b, i_d, j_d] = rigid_global_info.mass_mat_L_bw[
                                1, i_b, j_pr, i_pr
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

            for j_d_ in (
                range(i_d + 1, entity_dof_end)
                if qd.static(not BW)
                else qd.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
            ):
                j_d = j_d_ if qd.static(not BW) else (j_d_ + entities_info.dof_start[i_e])
                if func_check_index_range(j_d, i_d + 1, entity_dof_end, BW):
                    # Since we read out[j_d, i_b], and j_d > i_d, which means that out[j_d, i_b] is already
                    # finalized at this point, we don't need to care about AD mutation rule.
                    if qd.static(BW):
                        out_bw[0, i_d, i_b] = (
                            out_bw[0, i_d, i_b] - rigid_global_info.mass_mat_L[i_b, j_d, i_d] * out_bw[0, j_d, i_b]
                        )
                    else:
                        curr_out = curr_out - rigid_global_info.mass_mat_L[i_b, j_d, i_d] * out[j_d, i_b]

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

            for j_d_ in (
                range(entity_dof_start, i_d)
                if qd.static(not BW)
                else qd.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
            ):
                j_d = j_d_ if qd.static(not BW) else (j_d_ + entities_info.dof_start[i_e])
                if func_check_index_range(j_d, entity_dof_start, i_d, BW):
                    curr_out = curr_out - rigid_global_info.mass_mat_L[i_b, i_d, j_d] * out[j_d, i_b]

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


# Upper bound on total scene DOFs for the cooperative tiled M^-1 solve. The kernel
# stages the whole flat DOF vector for 8 envs in LDS (msolve = 8 x n_dofs_ floats =
# 32 * n_dofs_ bytes). gfx942 has 64 KB LDS/workgroup, so at n_dofs_ > 2048 the tile
# no longer fits (launch/compile failure) and well before that it caps occupancy
# (n_dofs_=512 -> 16 KB -> 4 WG/CU). Large-DOF / multi-entity batched scenes above
# this bound fall back to the serial func_solve_mass / func_solve_mass_batch, which
# use no oversized LDS and handle arbitrary DOF counts. 512 keeps the tile <=16 KB
# (>=4 WG/CU on LDS) while comfortably covering the single-/few-robot batched regime.
COOP_MASS_SOLVE_MAX_DOFS = 512


@qd.func
def func_solve_mass_coop_tiled(
    vec: qd.Tensor,
    out: qd.Tensor,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Cooperative tiled forward M^-1 solve for AMDGPU (out = M^-1 @ vec).

    The serial `func_solve_mass` launches 1 thread/env -> only ~n_envs/32
    wavefronts (~2.6% occupancy at 8192 envs), leaving the GPU idle on the
    dependent triangular-solve chain. This tiles 8 envs/block x 8 lanes/env (8x
    the wavefronts) and reads mass_mat_L[i_b, p, k] with lanes over column k
    (coalesced under the [env,row,col] layout), software-pipelined -- mirroring
    the tiled_wc Phase 5b solve. Forward-only (no autodiff backward path).
    """
    BLOCK_DIM = qd.static(64)
    COOP = qd.static(8)
    ENVS = qd.static(8)
    N_DOFS = qd.static(static_rigid_sim_config.n_dofs_)
    N_BLOCKS = qd.static((static_rigid_sim_config.n_envs + 8 - 1) // 8)

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=BLOCK_DIM)
    for i_t in range(N_BLOCKS * BLOCK_DIM):
        tid = i_t % BLOCK_DIM
        block_id = i_t // BLOCK_DIM
        env_in_block = tid // COOP
        lane_in_env = tid % COOP
        i_b = block_id * ENVS + env_in_block
        msolve = qd.simt.block.SharedArray((ENVS, N_DOFS), gs.qd_float)
        # OOB guard: never `continue` (would deadlock the workgroup syncs).
        oob = i_b >= static_rigid_sim_config.n_envs

        for i_e in range(qd.static(static_rigid_sim_config.n_entities_)):
            e_ds = entities_info.dof_start[i_e]
            e_de = entities_info.dof_end[i_e]
            e_n = e_de - e_ds
            do_e = False
            if not oob:
                do_e = bool(rigid_global_info.mass_mat_mask[i_e, i_b])

            # load y -> LDS stripe
            if do_e:
                i_d = e_ds + lane_in_env
                while i_d < e_de:
                    msolve[env_in_block, i_d] = vec[i_d, i_b]
                    i_d = i_d + COOP
            qd.simt.block.sync()

            # Step 1: solve L^T w = y (back-substitution), software-pipelined
            for pp in range(e_n):
                p = e_de - 1 - pp
                if do_e:
                    wp = msolve[env_in_block, p]
                    k = e_ds + lane_in_env
                    l_pf = gs.qd_float(0.0)
                    if k < p:
                        l_pf = rigid_global_info.mass_mat_L[i_b, p, k]
                    while k < p:
                        l_cur = l_pf
                        k = k + COOP
                        l_pf = gs.qd_float(0.0)
                        if k < p:
                            l_pf = rigid_global_info.mass_mat_L[i_b, p, k]
                        msolve[env_in_block, k - COOP] = msolve[env_in_block, k - COOP] - l_cur * wp
                qd.simt.block.sync()

            # Step 2: z = D^{-1} w
            if do_e:
                i_d = e_ds + lane_in_env
                while i_d < e_de:
                    msolve[env_in_block, i_d] = msolve[env_in_block, i_d] * rigid_global_info.mass_mat_D_inv[i_d, i_b]
                    i_d = i_d + COOP
            qd.simt.block.sync()

            # Step 3: solve L x = z (forward-substitution), software-pipelined
            for pp in range(e_n):
                p = e_ds + pp
                if do_e:
                    wp = msolve[env_in_block, p]
                    k = p + 1 + lane_in_env
                    l_pf = gs.qd_float(0.0)
                    if k < e_de:
                        l_pf = rigid_global_info.mass_mat_L[i_b, k, p]
                    while k < e_de:
                        l_cur = l_pf
                        k = k + COOP
                        l_pf = gs.qd_float(0.0)
                        if k < e_de:
                            l_pf = rigid_global_info.mass_mat_L[i_b, k, p]
                        msolve[env_in_block, k - COOP] = msolve[env_in_block, k - COOP] - l_cur * wp
                qd.simt.block.sync()

            # store x -> out
            if do_e:
                i_d = e_ds + lane_in_env
                while i_d < e_de:
                    out[i_d, i_b] = msolve[env_in_block, i_d]
                    i_d = i_d + COOP
            qd.simt.block.sync()


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
    contact_island_state: array_class.ContactIslandState,
    is_backward: qd.template(),
):
    BW = qd.static(is_backward)

    # compute force based on each dof's ctrl mode. The non-hibernation launch runs
    # this per-link-parallel (ndrange(n_links, _B)) instead of one-thread-per-entity
    # walking every link serially -- much higher occupancy at large batch sizes, and
    # the per-link applied force is independent so this is a flat relaunch (no extra
    # launches, no cross-link dependency). The hibernation path keeps the per-entity
    # walk because it needs the per-entity wakeup reduction.
    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_x, i_b in (
        qd.ndrange(entities_info.n_links.shape[0], dofs_state.ctrl_mode.shape[1])
        if qd.static(static_rigid_sim_config.use_hibernation)
        else qd.ndrange(links_info.root_idx.shape[0], dofs_state.ctrl_mode.shape[1])
    ):
        EPS = rigid_global_info.EPS[None]

        wakeup = False
        for i_l in (
            range(entities_info.link_start[i_x], entities_info.link_end[i_x])
            if qd.static(static_rigid_sim_config.use_hibernation)
            else range(i_x, i_x + 1)
        ):
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
            if entities_state.hibernated[i_x, i_b] and wakeup:
                # TODO: migrate this function
                func_wakeup_entity_and_its_temp_island(
                    i_x,
                    i_b,
                    entities_state,
                    entities_info,
                    dofs_state,
                    links_state,
                    geoms_state,
                    rigid_global_info,
                    contact_island_state,
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
    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for I in qd.grouped(qd.ndrange(*links_state.cfrc_coupling_ang.shape)):
        links_state.cfrc_coupling_ang[I] = qd.Vector.zero(gs.qd_float, 3)
        links_state.cfrc_coupling_vel[I] = qd.Vector.zero(gs.qd_float, 3)


@qd.func
def func_actuation(self):
    if qd.static(self._use_hibernation):
        pass
    else:
        qd.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
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

    # Forward acc_smooth = M^-1 @ force. On AMDGPU the serial 1-thread/env solve
    # is severely under-occupied (~2.6% at 8192 envs); use the cooperative tiled
    # solve. It is a block-cooperative kernel (block_dim + shared memory + block
    # syncs), so it is only correct under a fully parallel launch (para_level == ALL,
    # i.e. batched scenes); non-batched scenes (PARTIAL) would serialize it and break
    # the cooperative reduction. Keep the serial path there, for the autodiff backward
    # pass (cooperative version is forward-only), and for non-AMDGPU backends.
    if qd.static(
        static_rigid_sim_config.backend == gs.amdgpu
        and not is_backward
        and static_rigid_sim_config.para_level == gs.PARA_LEVEL.ALL
        and static_rigid_sim_config.n_dofs_ <= COOP_MASS_SOLVE_MAX_DOFS
    ):
        func_solve_mass_coop_tiled(
            dofs_state.force,
            dofs_state.acc_smooth,
            entities_info,
            rigid_global_info,
            static_rigid_sim_config,
        )
    else:
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
    contact_island_state: array_class.ContactIslandState,
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
        contact_island_state=contact_island_state,
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

        qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
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
