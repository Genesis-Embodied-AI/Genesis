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
from .misc import func_wakeup_island, func_check_index_range, func_add_safe_backward, linear_to_lower_tri


@qd.kernel
def update_qacc_from_qvel_delta(
    dyn_state: array_class.DynState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    n_dofs = dyn_state.dofs.ctrl_mode.shape[0]
    _B = dyn_state.dofs.ctrl_mode.shape[1]

    # Read shape constants from static config (qd.template) so they're compile-time literals,
    # avoiding _serial shape-lookup dispatches that the ndarray descriptor path would otherwise emit.
    n_dofs = static_rigid_sim_config.n_dofs_
    _B = static_rigid_sim_config.n_envs

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in qd.ndrange(1, _B) if qd.static(static_rigid_sim_config.use_hibernation) else qd.ndrange(n_dofs, _B):
        for i_1 in (
            range(rigid_info.n_awake_dofs[i_b]) if qd.static(rigid_config.use_hibernation) else qd.static(range(1))
        ):
            if i_1 < (rigid_info.n_awake_dofs[i_b] if qd.static(rigid_config.use_hibernation) else 1):
                i_d = rigid_info.awake_dofs[i_1, i_b] if qd.static(rigid_config.use_hibernation) else i_0
                dyn_state.dofs.acc[i_d, i_b] = (
                    dyn_state.dofs.vel[i_d, i_b] - dyn_state.dofs.vel_prev[i_d, i_b]
                ) / rigid_info.substep_dt[None]
                dyn_state.dofs.vel[i_d, i_b] = dyn_state.dofs.vel_prev[i_d, i_b]


@qd.kernel
def update_qvel(
    dyn_state: array_class.DynState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    _B = dyn_state.dofs.vel.shape[1]
    n_dofs = dyn_state.dofs.vel.shape[0]

    _B = static_rigid_sim_config.n_envs
    n_dofs = static_rigid_sim_config.n_dofs_

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in qd.ndrange(1, _B) if qd.static(static_rigid_sim_config.use_hibernation) else qd.ndrange(n_dofs, _B):
        for i_1 in (
            range(rigid_info.n_awake_dofs[i_b]) if qd.static(rigid_config.use_hibernation) else qd.static(range(1))
        ):
            if i_1 < (rigid_info.n_awake_dofs[i_b] if qd.static(rigid_config.use_hibernation) else 1):
                i_d = rigid_info.awake_dofs[i_1, i_b] if qd.static(rigid_config.use_hibernation) else i_0
                dyn_state.dofs.vel_prev[i_d, i_b] = dyn_state.dofs.vel[i_d, i_b]
                dyn_state.dofs.vel[i_d, i_b] = (
                    dyn_state.dofs.vel[i_d, i_b] + dyn_state.dofs.acc[i_d, i_b] * rigid_info.substep_dt[None]
                )


@qd.kernel(fastcache=True)
def kernel_compute_mass_matrix(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    decompose: qd.template(),
):
    func_compute_mass_matrix(dyn_state, dyn_info, rigid_info, rigid_config, implicit_damping=False, is_backward=False)
    if decompose:
        func_factor_mass(dyn_state, dyn_info, rigid_info, rigid_config, implicit_damping=False)


# @@@@@@@@@ Composer starts here
# decomposed kernels should happen in the block below. This block will be handled by composer and composed into a single kernel
@qd.func
def func_forward_dynamics(
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    func_compute_mass_matrix(
        dyn_state,
        dyn_info,
        rigid_info,
        rigid_config,
        qd.static(rigid_config.integrator == gs.integrator.approximate_implicitfast),
        is_backward,
    )
    func_factor_mass(dyn_state, dyn_info, rigid_info, rigid_config, implicit_damping=False)
    func_torque_and_passive_force(dyn_state, constraint_state, dyn_info, rigid_info, rigid_config, is_backward)
    func_update_acc(dyn_state, dyn_info, rigid_info, rigid_config, update_cacc=False, is_backward=is_backward)
    func_update_force(dyn_state, dyn_info, rigid_info, rigid_config, is_backward)
    func_bias_force(dyn_state, dyn_info, rigid_info, rigid_config, is_backward)
    func_compute_qacc(dyn_state, dyn_info, rigid_info, rigid_config)


@qd.kernel(fastcache=True)
def kernel_forward_dynamics(
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    func_forward_dynamics(dyn_state, constraint_state, dyn_info, rigid_info, rigid_config, is_backward=False)


@qd.kernel(fastcache=True)
def kernel_update_acc(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    func_update_acc(dyn_state, dyn_info, rigid_info, rigid_config, update_cacc=True, is_backward=False)


@qd.func
def func_vel_at_point(link_idx, i_b, pos_world, links_state: array_class.LinksState):
    """
    Velocity of a certain point on a rigid link.
    """
    vel_rot = links_state.cd_ang[link_idx, i_b].cross(pos_world - links_state.root_COM[link_idx, i_b])
    vel_lin = links_state.cd_vel[link_idx, i_b]
    return vel_rot + vel_lin



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
    n_thread_entities = static_rigid_sim_config.n_entities_ if qd.static(static_rigid_sim_config.use_hibernation) else n_entities

    qd.loop_config(block_dim=BLOCK_DIM)
    for i in range(n_thread_entities * _B * BLOCK_DIM):
        tid = i % BLOCK_DIM
        i_e_local = (i // BLOCK_DIM) % n_thread_entities
        i_b = i // (BLOCK_DIM * n_thread_entities)

        if i_b >= _B:
            continue

        i_e = i_e_local

        if qd.static(static_rigid_sim_config.use_hibernation):
            if not func_check_index_range(i_e_local, static_rigid_sim_config.n_entities_):
                continue
            i_e = rigid_global_info.awake_entities[i_e_local, i_b]
        entity_dof_start = entities_info.dof_start[i_e]
        entity_dof_end = entities_info.dof_end[i_e]
        n_dofs = entities_info.n_dofs[i_e]

        # This kernel uses a dense local mass matrix in shared memory, so its LDS
        # footprint is larger than kernels that use packed lower-triangular storage.
        # Total bytes = 4 * (n^2 + 12n), where:
        #   - n^2      comes from mass_mat_local[n, n]
        #   - 12n      comes from 4 vector caches of shape [n, 3]
        # Constraining that to ~64KB gives n <= 122 for 4-byte floats.
        KERNEL_MAX_DOFS_PER_ENTITY = qd.static(122 if MAX_DOFS_PER_ENTITY > 122 else MAX_DOFS_PER_ENTITY)

        if n_dofs <= 0 or n_dofs > KERNEL_MAX_DOFS_PER_ENTITY:
            continue

        # Shared-memory allocation sized to this kernel's actual dense-matrix budget.
        f_ang_cache = qd.simt.block.SharedArray((KERNEL_MAX_DOFS_PER_ENTITY, 3), gs.qd_float)
        f_vel_cache = qd.simt.block.SharedArray((KERNEL_MAX_DOFS_PER_ENTITY, 3), gs.qd_float)
        cdof_ang_cache = qd.simt.block.SharedArray((KERNEL_MAX_DOFS_PER_ENTITY, 3), gs.qd_float)
        cdof_vel_cache = qd.simt.block.SharedArray((KERNEL_MAX_DOFS_PER_ENTITY, 3), gs.qd_float)
        mass_mat_local = qd.simt.block.SharedArray(
            (KERNEL_MAX_DOFS_PER_ENTITY, KERNEL_MAX_DOFS_PER_ENTITY), gs.qd_float
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

            # Store in local matrix
            mass_mat_local[i_d_, j_d_] = ang_dot + vel_dot
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

            # Apply masking and store
            rigid_global_info.mass_mat[i_d_global, j_d_global, i_b] = (
                mass_mat_local[i_d_, j_d_] * rigid_global_info.mass_parent_mask[i_d_global, j_d_global]
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
            rigid_global_info.mass_mat[i_d_global, j_d_global, i_b] = rigid_global_info.mass_mat[j_d_global, i_d_global, i_b]

            upper_idx += BLOCK_DIM
@qd.func
def func_compute_mass_matrix(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    implicit_damping: qd.template(),
    is_backward: qd.template(),
):
    # Block size (warp width) for the cooperative mass_mat_assemble path. Used only when
    # enable_cooperative_constraint_kernels=True (and not use_hibernation). One warp per (entity, env); lanes stride
    # i_d_ within the entity dof range to coalesce the flipped mass_mat writes.
    _MASS_MAT_BLOCK = qd.static(32)

    BW = qd.static(is_backward)

    # crb initialize
    qd.loop_config(name="crb_initialize", serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        qd.ndrange(1, static_rigid_sim_config.n_envs)
        if qd.static(static_rigid_sim_config.use_hibernation)
        else qd.ndrange(static_rigid_sim_config.n_links_, static_rigid_sim_config.n_envs)
    ):
        for i_1 in (
            range(rigid_info.n_awake_links[i_b]) if qd.static(rigid_config.use_hibernation) else qd.static(range(1))
        ):
            if func_check_index_range(i_1, 0, rigid_info.n_awake_links[i_b], rigid_config.use_hibernation):
                i_l = rigid_info.awake_links[i_1, i_b] if qd.static(rigid_config.use_hibernation) else i_0

                dyn_state.links.crb_inertial[i_l, i_b] = dyn_state.links.cinr_inertial[i_l, i_b]
                dyn_state.links.crb_pos[i_l, i_b] = dyn_state.links.cinr_pos[i_l, i_b]
                dyn_state.links.crb_quat[i_l, i_b] = dyn_state.links.cinr_quat[i_l, i_b]
                dyn_state.links.crb_mass[i_l, i_b] = dyn_state.links.cinr_mass[i_l, i_b]

    # crb: composite-rigid-body inertia, folded leaf-to-root, one thread per kinematic tree (root_idx == itself) over
    # its link span in descending order so children fold before their parent propagates, gating each link on the
    # thread's root (see links_tree_end in array_class.py). A tree's top link may fold into a fixed 0-DOF anchor of
    # another tree, whose crb is unused. Mirrors the root_idx tree walk in func_update_cartesian_space.
    qd.loop_config(name="crb", serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        qd.ndrange(1, static_rigid_sim_config.n_envs)
        if qd.static(static_rigid_sim_config.use_hibernation)
        else qd.ndrange(static_rigid_sim_config.n_entities_, static_rigid_sim_config.n_envs)
    ):
        for i_1 in (
            range(rigid_info.n_awake_links[i_b]) if qd.static(rigid_config.use_hibernation) else qd.static(range(1))
        ):
            if func_check_index_range(i_1, 0, rigid_info.n_awake_links[i_b], rigid_config.use_hibernation):
                i_l_root = rigid_info.awake_links[i_1, i_b] if qd.static(rigid_config.use_hibernation) else i_0
                I_l_root = [i_l_root, i_b] if qd.static(rigid_config.batch_links_info) else i_l_root
                if dyn_info.links.root_idx[I_l_root] == i_l_root:
                    tree_end = rigid_info.links_tree_end[i_l_root]
                    for k in range(tree_end - i_l_root):
                        i_l = tree_end - 1 - k
                        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
                        i_p = dyn_info.links.parent_idx[I_l]
                        I_p = [i_p, i_b]

                        if dyn_info.links.root_idx[I_l] == i_l_root and i_p != -1:
                            func_add_safe_backward(
                                I_p, dyn_state.links.crb_inertial[i_l, i_b], dyn_state.links.crb_inertial, BW
                            )
                            func_add_safe_backward(
                                I_p, dyn_state.links.crb_mass[i_l, i_b], dyn_state.links.crb_mass, BW
                            )
                            func_add_safe_backward(I_p, dyn_state.links.crb_pos[i_l, i_b], dyn_state.links.crb_pos, BW)
                            func_add_safe_backward(
                                I_p, dyn_state.links.crb_quat[i_l, i_b], dyn_state.links.crb_quat, BW
                            )

    # mass_mat
    qd.loop_config(name="mass_mat", serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        qd.ndrange(1, static_rigid_sim_config.n_envs)
        if qd.static(static_rigid_sim_config.use_hibernation)
        else qd.ndrange(static_rigid_sim_config.n_links_, static_rigid_sim_config.n_envs)
    ):
        for i_1 in (
            range(rigid_info.n_awake_links[i_b]) if qd.static(rigid_config.use_hibernation) else qd.static(range(1))
        ):
            if func_check_index_range(i_1, 0, rigid_info.n_awake_links[i_b], rigid_config.use_hibernation):
                i_l = rigid_info.awake_links[i_1, i_b] if qd.static(rigid_config.use_hibernation) else i_0
                I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l

                for i_d_ in (
                    range(links_info.dof_start[I_l], links_info.dof_end[I_l])
                    if qd.static(not BW)
                    else qd.static(range(static_rigid_sim_config.max_n_dofs_per_link))
                ):
                    i_d = i_d_ if qd.static(not BW) else links_info.dof_start[I_l] + i_d_

                    if func_check_index_range(i_d, links_info.dof_start[I_l], links_info.dof_end[I_l], BW):
                        dofs_state.f_ang[i_d, i_b], dofs_state.f_vel[i_d, i_b] = gu.inertial_mul(
                            links_state.crb_pos[i_l, i_b],
                            links_state.crb_inertial[i_l, i_b],
                            links_state.crb_mass[i_l, i_b],
                            dofs_state.cdof_vel[i_d, i_b],
                            dofs_state.cdof_ang[i_d, i_b],
                        )

    # Choose mass matrix computation implementation
    if qd.static(
        static_rigid_sim_config.enable_tiled_cholesky_mass_matrix and static_rigid_sim_config.backend != gs.cpu
    ):
        # Use isolated LDS-optimized function
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
    else:
        # Original CPU/simple implementation
        qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
        for i_0, i_b in (
            qd.ndrange(1, static_rigid_sim_config.n_envs)
            if qd.static(static_rigid_sim_config.use_hibernation)
            else qd.ndrange(static_rigid_sim_config.n_entities_, static_rigid_sim_config.n_envs)
        ):
            for i_1 in (
                (
                    # Dynamic inner loop for forward pass
                    range(rigid_global_info.n_awake_entities[i_b])
                    if qd.static(static_rigid_sim_config.use_hibernation)
                    else qd.static(range(1))
                )
                if qd.static(not BW)
                else (
                    qd.static(range(static_rigid_sim_config.max_n_awake_entities))  # Static inner loop for backward pass
                    if qd.static(static_rigid_sim_config.use_hibernation)
                    else qd.static(range(1))
                )
            ):
                if func_check_index_range(
                    i_1, 0, rigid_global_info.n_awake_entities[i_b], static_rigid_sim_config.use_hibernation
                ):
                    i_e = (
                        rigid_global_info.awake_entities[i_1, i_b]
                        if qd.static(static_rigid_sim_config.use_hibernation)
                        else i_0
                    )

    if qd.static(rigid_config.enable_cooperative_constraint_kernels and not rigid_config.use_hibernation):
        # Cooperative warp-per-(entity, env) writer over the lower triangle (inclusive of diagonal). Each cell's
        # symmetric value is computed once via the sqrt-formula compressed pair index and written to both
        # `[i_d, j_d, i_b]` and `[j_d, i_d, i_b]` inline, saving the upper-tri dot products that the previous
        # two-pass path computed and then overwrote, and removing the separate mirror pass. Under the flipped
        # mass_mat layout (i_d stride-1) the primary write coalesces; the inline mirror write is strided but
        # replaces the previous mirror-pass read-write at similar cost.
        n_entities = dyn_info.entities.n_links.shape[0]
        _B_assemble = dyn_state.links.pos.shape[1]
        qd.loop_config(name="mass_mat_assemble", block_dim=_MASS_MAT_BLOCK)
        for i_flat in range(n_entities * _B_assemble * _MASS_MAT_BLOCK):
            tid = i_flat % _MASS_MAT_BLOCK
            i_eb = i_flat // _MASS_MAT_BLOCK
            i_e = i_eb % n_entities
            i_b = i_eb // n_entities

            # Assemble each mass block whose root DOF lies in this entity over its full lower triangle: a merged child
            # owns no root and assembles nothing, while the parent assembles the whole coupled block (its columns
            # extend past the entity). mass_parent_mask zeroes the within-block ancestor gaps.
            entity_dof_start = dyn_info.entities.dof_start[i_e]
            entity_dof_end = dyn_info.entities.dof_end[i_e]
            block_start = entity_dof_start
            while block_start < entity_dof_end:
                block_end = rigid_info.dofs_mass_block_end[block_start]
                if rigid_info.dofs_mass_block_start[block_start] == block_start:
                    n_block_dofs = block_end - block_start
                    n_lower_tri = n_block_dofs * (n_block_dofs + 1) // 2
                    i_pair = tid
                    while i_pair < n_lower_tri:
                        # Compressed lower-tri-inclusive index: i_pair = i_d_ * (i_d_ + 1) / 2 + j_d_, with j_d_ in
                        # [0, i_d_]. The fast-math-robust inversion is required: a raw sqrt drops the j=0 entry of
                        # every perfect-square row on GPU, leaving M indefinite.
                        i_d_, j_d_ = linear_to_lower_tri(i_pair)
                        i_d = block_start + i_d_
                        j_d = block_start + j_d_
                        val = (
                            dyn_state.dofs.f_ang[i_d, i_b].dot(dyn_state.dofs.cdof_ang[j_d, i_b])
                            + dyn_state.dofs.f_vel[i_d, i_b].dot(dyn_state.dofs.cdof_vel[j_d, i_b])
                        ) * rigid_info.mass_parent_mask[i_d, j_d]
                        rigid_info.mass_mat[i_d, j_d, i_b] = val
                        if i_d_ != j_d_:
                            rigid_info.mass_mat[j_d, i_d, i_b] = val
                        i_pair += _MASS_MAT_BLOCK
                block_start = block_end
    else:
        qd.loop_config(name="mass_mat_assemble", serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
        for i_0, i_b in (
            qd.ndrange(1, dyn_state.links.pos.shape[1])
            if qd.static(rigid_config.use_hibernation)
            else qd.ndrange(dyn_info.entities.n_links.shape[0], dyn_state.links.pos.shape[1])
        ):
            for i_1 in (
                range(rigid_info.n_awake_entities[i_b])
                if qd.static(rigid_config.use_hibernation)
                else qd.static(range(1))
            ):
                if func_check_index_range(i_1, 0, rigid_info.n_awake_entities[i_b], rigid_config.use_hibernation):
                    i_e = rigid_info.awake_entities[i_1, i_b] if qd.static(rigid_config.use_hibernation) else i_0

                    # Assemble each mass block rooted in this entity over its full range (see
                    # entities_mass_block_dof_start in array_class.py): the mirror pass reads rows of the whole block,
                    # so the block-root entity must write them all itself before mirroring.
                    blocks_dof_start = rigid_info.entities_mass_block_dof_start[i_e]
                    blocks_dof_end = rigid_info.entities_mass_block_dof_end[i_e]
                    for i_d in range(blocks_dof_start, blocks_dof_end):
                        for j_d in range(rigid_info.dofs_mass_block_start[i_d], rigid_info.dofs_mass_block_end[i_d]):
                            rigid_info.mass_mat[i_d, j_d, i_b] = (
                                dyn_state.dofs.f_ang[i_d, i_b].dot(dyn_state.dofs.cdof_ang[j_d, i_b])
                                + dyn_state.dofs.f_vel[i_d, i_b].dot(dyn_state.dofs.cdof_vel[j_d, i_b])
                            ) * rigid_info.mass_parent_mask[i_d, j_d]

                    for i_d in range(blocks_dof_start, blocks_dof_end):
                        for j_d in range(i_d + 1, rigid_info.dofs_mass_block_end[i_d]):
                            rigid_info.mass_mat[i_d, j_d, i_b] = rigid_info.mass_mat[j_d, i_d, i_b]

    # Take into account motor armature
    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in qd.ndrange(static_rigid_sim_config.n_dofs_, static_rigid_sim_config.n_envs):
        I_d = [i_d, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else i_d
        func_add_safe_backward(rigid_global_info.mass_mat, (i_d, i_d, i_b), dofs_info.armature[I_d], BW)

    # Take into account first-order correction terms for implicit integration scheme right away
    if qd.static(implicit_damping):
        qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_d, i_b in qd.ndrange(static_rigid_sim_config.n_dofs_, static_rigid_sim_config.n_envs):
            I_d = [i_d, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else i_d
            rigid_global_info.mass_mat[i_d, i_d, i_b] = (
                rigid_global_info.mass_mat[i_d, i_d, i_b] + dofs_info.damping[I_d] * rigid_global_info.substep_dt[None]
            )
            if dyn_state.dofs.ctrl_mode[i_d, i_b] <= gs.CTRL_MODE.VELOCITY:
                # qM += d qfrc_actuator / d qvel = -act_bias[2] * dt
                rigid_info.mass_mat[i_d, i_d, i_b] = (
                    rigid_info.mass_mat[i_d, i_d, i_b] - dyn_info.dofs.act_bias[I_d][2] * rigid_info.substep_dt[None]
                )


@qd.func
def func_factor_mass_tiled(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    implicit_damping: qd.template(),
    TileCls: qd.template(),
):
    """Register-streaming tiled per-entity mass factor for the >shared-cap branch (GPU forward only).

    Replaces the shared-pivot cooperative LDL^T when an entity's mass submatrix exceeds GPU shared memory. M is
    block-diagonal per mass block (see dofs_mass_block_start in array_class.py), so one warp of T lanes factors each
    of the entity's blocks independently (a single-block entity has just one block spanning it) via the same
    qd.simt.TileNxN blocked Cholesky as the constraint Hessian.

    func_solve_mass consumes the LTDL form M = L^T D L (L unit-lower), produced by eliminating DOFs last-to-first, not
    the standard L D L^T. The tile primitive does forward Cholesky M = G G^T, so each block's reverse-indexed matrix
    M_rev[a, b] = M[n-1-a, n-1-b] (n the block's DOF count) is factored and its factor mapped back to the block's LTDL
    factor:
      L[i,j] = G_rev[n-1-j, n-1-i] / G_rev[n-1-i, n-1-i]  (i > j),  D_inv[i] = 1 / G_rev[n-1-i, n-1-i]^2,  diag(L) = 1.

    The qd.simt tile ops are batch-first while mass_mat_L is canonical batch-last (n_dofs, n_dofs, _B), so the
    factorization runs in each block's region of the batch-first scratch rigid_info.mass_mat_tiled_scratch and is
    scattered into mass_mat_L / mass_mat_D_inv. To avoid a dedicated allocation, that scratch aliases the constraint
    Hessian buffer nt_H (same shape, and free at mass-factor time since the constraint solve only populates it later in
    the step); see get_constraint_state. The scratch and mass_mat_L are distinct buffers, so the scatter is race-free.
    """
    # Reuse the Hessian's tile width; TileCls is dispatched to match it at the call site, so T and the tile class stay
    # consistent for either value. In practice this path only runs for mass blocks exceeding shared memory (total
    # n_dofs > 48), where the rule lands on 32.
    T = qd.static(rigid_config.cholesky_tile_size)
    EPS = rigid_info.EPS[None]

    n_entities = dyn_info.entities.n_links.shape[0]
    _B = dyn_state.dofs.ctrl_mode.shape[1]

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
        if qd.static(rigid_config.use_hibernation):
            if i_e >= rigid_info.n_awake_entities[i_b]:
                continue
            i_e = rigid_info.awake_entities[i_e, i_b]
        if not rigid_info.mass_mat_mask[i_e, i_b]:
            continue

        # Factor each mass block whose root DOF lies in this entity over its full range: a merged child owns no root
        # and factors nothing, while the parent factors the whole coupled block. Each block owns its own scratch
        # region [block_start, block_end), disjoint across entities too, so the scatter stays race-free.
        entity_dof_start = dyn_info.entities.dof_start[i_e]
        entity_dof_end = dyn_info.entities.dof_end[i_e]
        block_start = entity_dof_start
        while block_start < entity_dof_end:
            block_end = rigid_info.dofs_mass_block_end[block_start]
            if rigid_info.dofs_mass_block_start[block_start] == block_start:
                n_block_dofs = block_end - block_start
                n_blocks = (n_block_dofs + T - 1) // T

                # Phase 1: copy the reverse-indexed symmetric M block (+ implicit damping) into the scratch workspace.
                # mass_mat stores M's lower triangle, so M[ri_, rj_] with ri_ <= rj_ is read from the stored M[rj_, ri_].
                i_d_ = tid
                while i_d_ < n_block_dofs:
                    ri_ = n_block_dofs - 1 - i_d_
                    for j_d_ in range(i_d_ + 1):
                        rj_ = n_block_dofs - 1 - j_d_  # i_d_ >= j_d_  =>  ri_ <= rj_
                        m = rigid_info.mass_mat[block_start + rj_, block_start + ri_, i_b]
                        rigid_info.mass_mat_tiled_scratch[i_b, block_start + i_d_, block_start + j_d_] = m
                        rigid_info.mass_mat_tiled_scratch[i_b, block_start + j_d_, block_start + i_d_] = m
                    if qd.static(implicit_damping):
                        # Reverse-diagonal slot i_d_ holds M[ri_, ri_]; damping/act_bias index the original DOF.
                        i_d = block_start + ri_
                        I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d
                        rigid_info.mass_mat_tiled_scratch[i_b, block_start + i_d_, block_start + i_d_] = (
                            rigid_info.mass_mat_tiled_scratch[i_b, block_start + i_d_, block_start + i_d_]
                            + dyn_info.dofs.damping[I_d] * rigid_info.substep_dt[None]
                        )
                        if qd.static(rigid_config.integrator == gs.integrator.implicitfast):
                            if dyn_state.dofs.ctrl_mode[i_d, i_b] <= gs.CTRL_MODE.VELOCITY:
                                rigid_info.mass_mat_tiled_scratch[i_b, block_start + i_d_, block_start + i_d_] = (
                                    rigid_info.mass_mat_tiled_scratch[i_b, block_start + i_d_, block_start + i_d_]
                                    - dyn_info.dofs.act_bias[I_d][2] * rigid_info.substep_dt[None]
                                )
                    i_d_ = i_d_ + T
                qd.simt.block.sync()

                # Phase 2: blocked Cholesky G_rev G_rev^T = M_rev in the scratch workspace (mirrors the constraint
                # Hessian's func_cholesky_factor_direct_tiled; the tile ops are warp-synchronous, so no sync in loop).
                for kb in range(n_blocks):
                    k0 = block_start + kb * T
                    k1 = qd.min(k0 + T, block_end)

                    L_kk = TileCls.eye(dtype=gs.qd_float)  # rows past n_block_dofs stay identity
                    L_kk[:] = rigid_info.mass_mat_tiled_scratch[i_b, k0:k1, k0:k1]
                    for jb in range(kb):
                        j0 = block_start + jb * T
                        for t in range(T):
                            v = rigid_info.mass_mat_tiled_scratch[i_b, k0:k1, j0 + t]
                            L_kk -= qd.outer(v, v)
                    # Floor each pivot relative to the row's original diagonal: a pivot carries the inertia of its DOF,
                    # which falls with the fifth power of the body size, so an absolute floor takes over the
                    # factorization below unit scale. The diagonal read here is still the original one - the
                    # left-looking factor writes this block only after the call - and 'eps' is read per lane, so a
                    # per-lane value costs nothing.
                    d_row = k0 + tid
                    diag_orig = gs.qd_float(1.0)
                    if d_row < k1:
                        diag_orig = rigid_info.mass_mat_tiled_scratch[i_b, d_row, d_row]
                    L_kk.cholesky_(EPS * qd.max(diag_orig, EPS))

                    for ib in range(kb + 1, n_blocks):
                        i0 = block_start + ib * T
                        i1 = qd.min(i0 + T, block_end)

                        L_ik = TileCls.zeros(dtype=gs.qd_float)
                        L_ik[:] = rigid_info.mass_mat_tiled_scratch[i_b, i0:i1, k0:k1]
                        for jb in range(kb):
                            j0 = block_start + jb * T
                            for t in range(T):
                                v_own = rigid_info.mass_mat_tiled_scratch[i_b, i0:i1, j0 + t]
                                v_diag = rigid_info.mass_mat_tiled_scratch[i_b, k0:k1, j0 + t]
                                L_ik -= qd.outer(v_own, v_diag)
                        L_kk.solve_triangular_(L_ik)
                        rigid_info.mass_mat_tiled_scratch[i_b, i0:i1, k0:k1] = L_ik

                    rigid_info.mass_mat_tiled_scratch[i_b, k0:k1, k0:k1] = L_kk
                qd.simt.block.sync()

                # Phase 3: scatter the LTDL factor of M from G_rev (scratch) into canonical mass_mat_L / mass_mat_D_inv.
                # Reads the scratch, writes the distinct mass_mat_L (no in-place hazard). Only the strict-lower triangle
                # and unit diagonal are meaningful to the solve; the upper triangle is left untouched.
                n_strict_lower = n_block_dofs * (n_block_dofs - 1) // 2
                i_pair = tid
                while i_pair < n_strict_lower:
                    i_d_, j_d_ = linear_to_lower_tri(i_pair, strict=True)
                    ri_ = n_block_dofs - 1 - i_d_
                    rj_ = n_block_dofs - 1 - j_d_  # i_d_ > j_d_  =>  rj_ > ri_  (a lower G_rev entry)
                    g_num = rigid_info.mass_mat_tiled_scratch[i_b, block_start + rj_, block_start + ri_]
                    g_den = rigid_info.mass_mat_tiled_scratch[i_b, block_start + ri_, block_start + ri_]
                    rigid_info.mass_mat_L[block_start + i_d_, block_start + j_d_, i_b] = g_num / g_den
                    i_pair = i_pair + T

                i_d_ = tid
                while i_d_ < n_block_dofs:
                    ri_ = n_block_dofs - 1 - i_d_
                    g_den = rigid_info.mass_mat_tiled_scratch[i_b, block_start + ri_, block_start + ri_]
                    rigid_info.mass_mat_D_inv[block_start + i_d_, i_b] = 1.0 / (g_den * g_den)
                    rigid_info.mass_mat_L[block_start + i_d_, block_start + i_d_, i_b] = 1.0
                    i_d_ = i_d_ + T
                qd.simt.block.sync()
            block_start = block_end


@qd.func
def func_factor_mass(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    implicit_damping: qd.template(),
):
    n_entities = dyn_info.entities.n_links.shape[0]
    _B = dyn_state.dofs.ctrl_mode.shape[1]

    if qd.static(not BW):
        n_entities = static_rigid_sim_config.n_entities_
        _B = static_rigid_sim_config.n_envs

        qd.loop_config(name="factor_mass", block_dim=BLOCK_DIM)
        for i in range(n_entities * _B * BLOCK_DIM):
            tid = i % BLOCK_DIM
            i_e = (i // BLOCK_DIM) % n_entities
            i_b = i // (BLOCK_DIM * n_entities)
            if i_b >= _B:
                continue
            # Skip hibernated entities: their mass matrix is unchanged, so the factor from the last awake step
            # stays valid. The slot remaps to an awake entity, so the work scales with the awake entity count.
            if qd.static(rigid_config.use_hibernation):
                if i_e >= rigid_info.n_awake_entities[i_b]:
                    continue
                i_e = rigid_info.awake_entities[i_e, i_b]

            if rigid_info.mass_mat_mask[i_e, i_b]:
                entity_dof_start = dyn_info.entities.dof_start[i_e]
                entity_dof_end = dyn_info.entities.dof_end[i_e]

                pivot_row = qd.simt.block.SharedArray((MAX_DOFS_PER_BLOCK,), gs.qd_float)

                # Factor each mass block rooted in this entity in-place in global memory over its full range,
                # block-relative so shared indices stay >= 0; a merged child owns no root and factors nothing.
                block_start = entity_dof_start
                while block_start < entity_dof_end:
                    block_end = rigid_info.dofs_mass_block_end[block_start]
                    if rigid_info.dofs_mass_block_start[block_start] == block_start:
                        n_block_dofs = block_end - block_start

                        # Copy the block's lower triangle into mass_mat_L (+ implicit damping on the diagonal),
                        # cooperatively. Restricting to the block makes the factorization cost the sum of per-block
                        # cubes instead of the whole (possibly multi-block) entity cube.
                        i_d_ = tid
                        while i_d_ < n_block_dofs:
                            i_d = block_start + i_d_
                            for j_d in range(block_start, i_d + 1):
                                rigid_info.mass_mat_L[i_d, j_d, i_b] = rigid_info.mass_mat[i_d, j_d, i_b]
                            if qd.static(implicit_damping):
                                I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d
                                rigid_info.mass_mat_L[i_d, i_d, i_b] = (
                                    rigid_info.mass_mat_L[i_d, i_d, i_b]
                                    + dyn_info.dofs.damping[I_d] * rigid_info.substep_dt[None]
                                )
                                if qd.static(rigid_config.integrator == gs.integrator.implicitfast):
                                    if dyn_state.dofs.ctrl_mode[i_d, i_b] <= gs.CTRL_MODE.VELOCITY:
                                        rigid_info.mass_mat_L[i_d, i_d, i_b] = (
                                            rigid_info.mass_mat_L[i_d, i_d, i_b]
                                            - dyn_info.dofs.act_bias[I_d][2] * rigid_info.substep_dt[None]
                                        )
                            i_d_ = i_d_ + BLOCK_DIM
                        qd.simt.block.sync()

                        # In-place LDL^T, eliminating dofs from last to first (matches the scalar branch).
                        for j in range(n_block_dofs):
                            i_d = block_end - j - 1
                            i_d_local = i_d - block_start
                            D_inv = 1.0 / rigid_info.mass_mat_L[i_d, i_d, i_b]
                            if tid == 0:
                                rigid_info.mass_mat_D_inv[i_d, i_b] = D_inv

                    for j in range(n_dofs):
                        i_d_ = n_dofs - j - 1
                        i_d = entity_dof_end - j - 1

                        D_inv = 1.0 / mass_mat[i_d_, i_d_]
                        if tid == 0:
                            rigid_global_info.mass_mat_D_inv[i_d, i_b] = D_inv
                            # FIXME: Diagonal coeffs of L are ignored in computations, so no need to update them.
                            rigid_global_info.mass_mat_L[i_d, i_d, i_b] = 1.0

                        # Cache original pivot row values before modification.
                        # wave64: all threads in lockstep, no explicit sync needed.
                        if tid < i_d_:
                            sh_pivot[tid] = mass_mat[i_d_, tid]

                        # Row-major rank-1 update: each thread owns rows [tid, tid+BLOCK_DIM, ...].
                        # Eliminates the per-update sqrt needed by the flat-index decode, and keeps
                        # sh_pivot[_r] in a VGPR register across the inner column loop.
                        _r = tid
                        while _r < i_d_:
                            piv_r = sh_pivot[_r] * D_inv
                            _c = 0
                            while _c <= _r:
                                mass_mat[_r, _c] = mass_mat[_r, _c] - piv_r * sh_pivot[_c]
                                _c = _c + 1
                            _r = _r + BLOCK_DIM

                        # Write L factors to pivot row
                        if tid < i_d_:
                            mass_mat[i_d_, tid] = sh_pivot[tid] * D_inv

                        if qd.static(
                            static_rigid_sim_config.backend == gs.cuda
                            or static_rigid_sim_config.backend == gs.amdgpu
                        ):
                            if i_d_ <= WARP_SIZE:
                                qd.simt.warp.sync(qd.u32(0xFFFFFFFF))
                            elif qd.static(BLOCK_DIM > 64):
                                qd.simt.block.sync()
                        else:
                            qd.simt.block.sync()

                            # Phase B: each lane eliminates one column j_d, updating its own row j_d of the trailing
                            # submatrix from the read-only snapshot. Distinct rows per lane => no write conflicts,
                            # and the pivot row is only read (from shared) => no read/write race on row i_d.
                            j_d_ = tid
                            while j_d_ < i_d_local:
                                a = pivot_row[j_d_] * D_inv
                                j_d = block_start + j_d_
                                for k_d_ in range(j_d_ + 1):
                                    rigid_info.mass_mat_L[j_d, block_start + k_d_, i_b] = (
                                        rigid_info.mass_mat_L[j_d, block_start + k_d_, i_b] - a * pivot_row[k_d_]
                                    )
                                rigid_info.mass_mat_L[i_d, j_d, i_b] = a
                                j_d_ = j_d_ + BLOCK_DIM
                            qd.simt.block.sync()

                            # Diagonal coeffs of L are ignored downstream (see scalar branch) but set to 1.0 to match.
                            if tid == 0:
                                rigid_info.mass_mat_L[i_d, i_d, i_b] = 1.0
                    block_start = block_end
    elif qd.static(not rigid_config.enable_tiled_cholesky_mass_matrix or rigid_config.backend == gs.cpu):
        qd.loop_config(name="factor_mass", serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_slot, i_b in qd.ndrange(n_entities, _B):
            # Skip hibernated entities: their mass matrix is unchanged, so the factor from the last awake step
            # stays valid. This makes the factorization cost scale with the awake entity count.
            i_e = i_slot
            if qd.static(rigid_config.use_hibernation):
                if i_slot >= rigid_info.n_awake_entities[i_b]:
                    continue
                i_e = rigid_info.awake_entities[i_slot, i_b]
            if rigid_info.mass_mat_mask[i_e, i_b]:
                # Factor each mass block rooted in this entity, iterated flat over the rooted range with per-DOF
                # block bounds (see entities_mass_block_dof_start in array_class.py): elimination never leaves a
                # block, so interleaving independent blocks in one descending scan is exact.
                blocks_dof_start = rigid_info.entities_mass_block_dof_start[i_e]
                blocks_dof_end = rigid_info.entities_mass_block_dof_end[i_e]
                for i_d in range(blocks_dof_start, blocks_dof_end):
                    for j_d in range(rigid_info.dofs_mass_block_start[i_d], i_d + 1):
                        rigid_info.mass_mat_L[i_d, j_d, i_b] = rigid_info.mass_mat[i_d, j_d, i_b]

                    if qd.static(implicit_damping):
                        I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d
                        rigid_info.mass_mat_L[i_d, i_d, i_b] = (
                            rigid_info.mass_mat_L[i_d, i_d, i_b]
                            + dyn_info.dofs.damping[I_d] * rigid_info.substep_dt[None]
                        )
                        if qd.static(rigid_config.integrator == gs.integrator.implicitfast):
                            if dyn_state.dofs.ctrl_mode[i_d, i_b] <= gs.CTRL_MODE.VELOCITY:
                                rigid_info.mass_mat_L[i_d, i_d, i_b] = (
                                    rigid_info.mass_mat_L[i_d, i_d, i_b]
                                    - dyn_info.dofs.act_bias[I_d][2] * rigid_info.substep_dt[None]
                                )

                for i_d_ in range(blocks_dof_end - blocks_dof_start):
                    i_d = blocks_dof_end - i_d_ - 1
                    block_start = rigid_info.dofs_mass_block_start[i_d]
                    D_inv = 1.0 / rigid_info.mass_mat_L[i_d, i_d, i_b]
                    rigid_info.mass_mat_D_inv[i_d, i_b] = D_inv

                    for j_d_ in range(i_d - block_start):
                        j_d = i_d - j_d_ - 1
                        a = rigid_info.mass_mat_L[i_d, j_d, i_b] * D_inv
                        for k_d in range(block_start, j_d + 1):
                            rigid_info.mass_mat_L[j_d, k_d, i_b] -= a * rigid_info.mass_mat_L[i_d, k_d, i_b]
                        rigid_info.mass_mat_L[i_d, j_d, i_b] = a

                    # FIXME: Diagonal coeffs of L are ignored in computations, so no need to update them.
                    rigid_info.mass_mat_L[i_d, i_d, i_b] = 1.0
    else:
        BLOCK_DIM = qd.static(32)
        MAX_DOFS_PER_BLOCK = qd.static(rigid_config.tiled_n_dofs_per_block)
        WARP_SIZE = qd.static(32)

        # Assume this is the outermost loop
        qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_e, i_b in qd.ndrange(static_rigid_sim_config.n_entities_, static_rigid_sim_config.n_envs):
            if rigid_global_info.mass_mat_mask[i_e, i_b]:
                EPS = rigid_global_info.EPS[None]

            if rigid_info.mass_mat_mask[i_e, i_b]:
                entity_dof_start = dyn_info.entities.dof_start[i_e]
                entity_dof_end = dyn_info.entities.dof_end[i_e]

                mass_mat = qd.simt.block.SharedArray((MAX_DOFS_PER_BLOCK, MAX_DOFS_PER_BLOCK + 1), gs.qd_float)

                # Factor each mass block rooted in this entity in shared memory, indexed block-relative so
                # shared indices stay >= 0 (a merged child's block starts before its entity); the child owns no
                # root and factors nothing, while the parent factors the whole coupled block.
                block_start = entity_dof_start
                while block_start < entity_dof_end:
                    block_end = rigid_info.dofs_mass_block_end[block_start]
                    if rigid_info.dofs_mass_block_start[block_start] == block_start:
                        n_block_dofs = block_end - block_start
                        n_lower_tri = n_block_dofs * (n_block_dofs + 1) // 2

                        i_pair = tid
                        while i_pair < n_lower_tri:
                            i_d_, j_d_ = linear_to_lower_tri(i_pair)
                            mass_mat[i_d_, j_d_] = rigid_info.mass_mat[block_start + i_d_, block_start + j_d_, i_b]
                            i_pair = i_pair + BLOCK_DIM
                        qd.simt.block.sync()

                        if qd.static(implicit_damping):
                            i_d_ = tid
                            while i_d_ < n_block_dofs:
                                i_d = block_start + i_d_
                                I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d
                                mass_mat[i_d_, i_d_] = (
                                    mass_mat[i_d_, i_d_] + dyn_info.dofs.damping[I_d] * rigid_info.substep_dt[None]
                                )
                                if qd.static(rigid_config.integrator == gs.integrator.implicitfast):
                                    if dyn_state.dofs.ctrl_mode[i_d, i_b] <= gs.CTRL_MODE.VELOCITY:
                                        mass_mat[i_d_, i_d_] = (
                                            mass_mat[i_d_, i_d_]
                                            - dyn_info.dofs.act_bias[I_d][2] * rigid_info.substep_dt[None]
                                        )
                                i_d_ = i_d_ + BLOCK_DIM
                            qd.simt.block.sync()

                        for j in range(n_block_dofs):
                            i_d_ = n_block_dofs - j - 1
                            i_d = block_end - j - 1

                            D_inv = 1.0 / mass_mat[i_d_, i_d_]
                            if tid == 0:
                                rigid_info.mass_mat_D_inv[i_d, i_b] = D_inv
                                # FIXME: Diagonal coeffs of L are ignored in computations, so no need to update them.
                                rigid_info.mass_mat_L[i_d, i_d, i_b] = 1.0

                            j_d_ = i_d_ - 1 - tid
                            while j_d_ >= 0:
                                a = mass_mat[i_d_, j_d_] * D_inv
                                for k_d in range(j_d_ + 1):
                                    mass_mat[j_d_, k_d] = mass_mat[j_d_, k_d] - a * mass_mat[i_d_, k_d]
                                mass_mat[i_d_, j_d_] = a
                                j_d_ = j_d_ - BLOCK_DIM
                            if qd.static(rigid_config.backend == gs.cuda):
                                if i_d_ <= WARP_SIZE:
                                    qd.simt.warp.sync(qd.u32(0xFFFFFFFF))
                                else:
                                    qd.simt.block.sync()
                            else:
                                qd.simt.block.sync()

                        i_pair = tid
                        n_strict_lower_tri = n_block_dofs * (n_block_dofs - 1) // 2
                        while i_pair < n_strict_lower_tri:
                            i_d_, j_d_ = linear_to_lower_tri(i_pair, strict=True)
                            rigid_info.mass_mat_L[block_start + i_d_, block_start + j_d_, i_b] = mass_mat[i_d_, j_d_]
                            i_pair = i_pair + BLOCK_DIM
                        qd.simt.block.sync()
                    block_start = block_end


@qd.func
def func_solve_mass_entity(
    i_e: qd.int32,
    i_b: qd.int32,
    vec: qd.Tensor,
    out: qd.Tensor,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    if rigid_info.mass_mat_mask[i_e, i_b]:
        # Solve M x = y for each mass block rooted in this entity, iterated flat over the rooted range with per-DOF
        # block bounds (see entities_mass_block_dof_start in array_class.py); the substitutions never cross block
        # boundaries, and a merged child owns no root and solves nothing.
        blocks_dof_start = rigid_info.entities_mass_block_dof_start[i_e]
        blocks_dof_end = rigid_info.entities_mass_block_dof_end[i_e]

        # Step 1: Solve w st. L^T @ w = y. Reading out[j_d] (j_d > i_d) from the buffer being written is safe: those
        # entries were finalized in earlier (larger i_d) iterations. This func is never auto-reversed; the backward
        # pass seeds mass_mat.grad directly via the implicit function theorem (see kernel_manual_compute_qacc_bw in
        # manual_bw.py).
        for i_d_ in range(blocks_dof_end - blocks_dof_start):
            i_d = blocks_dof_end - i_d_ - 1
            block_end = rigid_info.dofs_mass_block_end[i_d]
            curr_out = vec[i_d, i_b]
            for j_d in range(i_d + 1, block_end):
                curr_out = curr_out - rigid_info.mass_mat_L[j_d, i_d, i_b] * out[j_d, i_b]
            out[i_d, i_b] = curr_out

        # Step 2: z = D^{-1} w
        for i_d in range(blocks_dof_start, blocks_dof_end):
            out[i_d, i_b] = out[i_d, i_b] * rigid_info.mass_mat_D_inv[i_d, i_b]

        # Step 3: Solve x st. L @ x = z
        for i_d in range(blocks_dof_start, blocks_dof_end):
            block_start = rigid_info.dofs_mass_block_start[i_d]
            curr_out = out[i_d, i_b]
            for j_d in range(block_start, i_d):
                curr_out = curr_out - rigid_info.mass_mat_L[i_d, j_d, i_b] * out[j_d, i_b]
            out[i_d, i_b] = curr_out


@qd.func
def func_solve_mass_batch(
    i_b: qd.int32,
    vec: qd.Tensor,
    out: qd.Tensor,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_0 in (
        (
            # Dynamic inner loop for forward pass
            range(rigid_global_info.n_awake_entities[i_b])
            if qd.static(static_rigid_sim_config.use_hibernation)
            else range(static_rigid_sim_config.n_entities_)
        )
        if qd.static(not BW)
        else (
            qd.static(range(static_rigid_sim_config.max_n_awake_entities))  # Static inner loop for backward pass
            if qd.static(static_rigid_sim_config.use_hibernation)
            else qd.static(range(static_rigid_sim_config.max_n_links_per_entity))
        )
    ):
        i_e = rigid_info.awake_entities[i_0, i_b] if qd.static(rigid_config.use_hibernation) else i_0
        func_solve_mass_entity(i_e, i_b, vec, out, dyn_info, rigid_info, rigid_config)


@qd.func
def func_solve_mass(
    vec: qd.Tensor,
    out: qd.Tensor,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    # This loop must be the outermost loop to be differentiable
    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_e, i_b in qd.ndrange(static_rigid_sim_config.n_entities_, static_rigid_sim_config.n_envs):
        func_solve_mass_entity(
            i_e, i_b, vec, out, out_bw, entities_info, rigid_global_info, static_rigid_sim_config, is_backward
        )


@qd.func
def func_torque_and_passive_force(
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    BW = qd.static(is_backward)

    # compute force based on each dof's ctrl mode
    if qd.static(static_rigid_sim_config.use_hibernation):
        # Hibernation path: keep the per-entity outer loop because the wakeup
        # bookkeeping (`wakeup` flag, `func_wakeup_entity_and_its_temp_island`)
        # is shared across all the dofs of an entity.
        qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_e, i_b in qd.ndrange(static_rigid_sim_config.n_entities_, static_rigid_sim_config.n_envs):
            EPS = rigid_global_info.EPS[None]

            wakeup = False
            for i_l_ in (
                range(entities_info.link_start[i_e], entities_info.link_end[i_e])
                if qd.static(not BW)
                else qd.static(range(static_rigid_sim_config.max_n_links_per_entity))
            ):
                i_l = i_l_ if qd.static(not BW) else (i_l_ + entities_info.link_start[i_e])

                if func_check_index_range(i_l, entities_info.link_start[i_e], entities_info.link_end[i_e], BW):
                    I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
                    if links_info.n_dofs[I_l] > 0:
                        i_j = links_info.joint_start[I_l]
                        I_j = [i_j, i_b] if qd.static(static_rigid_sim_config.batch_joints_info) else i_j
                        joint_type = joints_info.type[I_j]

                        for i_d_ in (
                            range(links_info.dof_start[I_l], links_info.dof_end[I_l])
                            if qd.static(not BW)
                            else qd.static(range(static_rigid_sim_config.max_n_dofs_per_link))
                        ):
                            i_d = i_d_ if qd.static(not BW) else (i_d_ + links_info.dof_start[I_l])

                            if func_check_index_range(i_d, links_info.dof_start[I_l], links_info.dof_end[I_l], BW):
                                I_d = [i_d, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else i_d
                                force = gs.qd_float(0.0)
                                if dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.FORCE:
                                    force = dofs_state.ctrl_force[i_d, i_b]
                                elif dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.VELOCITY:
                                    force = dofs_info.kv[I_d] * (dofs_state.ctrl_vel[i_d, i_b] - dofs_state.vel[i_d, i_b])
                                elif dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.POSITION and not (
                                    joint_type == gs.JOINT_TYPE.FREE and i_d >= links_info.dof_start[I_l] + 3
                                ):
                                    force = dofs_info.kp[I_d] * (
                                        dofs_state.ctrl_pos[i_d, i_b] - dofs_state.pos[i_d, i_b]
                                    ) + dofs_info.kv[I_d] * (dofs_state.ctrl_vel[i_d, i_b] - dofs_state.vel[i_d, i_b])

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
                                force = dofs_info.kp[I_d] * rotvec[j] - dofs_info.kv[I_d] * dofs_state.vel[i_d, i_b]

                                dofs_state.qf_applied[i_d, i_b] = qd.math.clamp(
                                    force, dofs_info.force_range[I_d][0], dofs_info.force_range[I_d][1]
                                )

                                if qd.abs(force) > EPS:
                                    wakeup = True

            if entities_state.hibernated[i_e, i_b] and wakeup:
                # TODO: migrate this function
                func_wakeup_entity_and_its_temp_island(
                    i_e,
                    i_b,
                    entities_state,
                    entities_info,
                    dofs_state,
                    links_state,
                    geoms_state,
                    rigid_global_info,
                    contact_island_state,
                )
    else:
        # Non-hibernation path: parallelize over (i_l, i_b) instead of
        # (i_e, i_b). For G1 this lifts the launch from
        # n_entities * n_envs (= ~16k threads / ~256 waves on MI300X, severely
        # under-occupying gfx942's 304 CUs * 4 SIMDs = 1216 SIMDs) to
        # n_links * n_envs (= ~245k threads / ~3840 waves), which actually
        # fills the chip and gives the memory subsystem enough in-flight
        # wavefronts to hide latency. block_dim=64 keeps lane utilization at
        # 100% on wave64 hardware. Bit-exact with the per-entity version
        # because each link writes only its own dofs' qf_applied entries.
        n_links = static_rigid_sim_config.n_links_
        qd.loop_config(
            serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL),
            block_dim=64,
        )
        for i_l, i_b in qd.ndrange(n_links, static_rigid_sim_config.n_envs):
            I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
            if links_info.n_dofs[I_l] > 0:
                EPS = rigid_global_info.EPS[None]
                i_j = links_info.joint_start[I_l]
                I_j = [i_j, i_b] if qd.static(static_rigid_sim_config.batch_joints_info) else i_j
                joint_type = joints_info.type[I_j]
                dof_start = links_info.dof_start[I_l]
                dof_end = links_info.dof_end[I_l]

                for i_d in range(dof_start, dof_end):
                    I_d = [i_d, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else i_d
                    force = gs.qd_float(0.0)
                    if dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.FORCE:
                        force = dofs_state.ctrl_force[i_d, i_b]
                    elif dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.VELOCITY:
                        force = dofs_info.kv[I_d] * (dofs_state.ctrl_vel[i_d, i_b] - dofs_state.vel[i_d, i_b])
                    elif dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.POSITION and not (
                        joint_type == gs.JOINT_TYPE.FREE and i_d >= dof_start + 3
                    ):
                        force = dofs_info.kp[I_d] * (
                            dofs_state.ctrl_pos[i_d, i_b] - dofs_state.pos[i_d, i_b]
                        ) + dofs_info.kv[I_d] * (dofs_state.ctrl_vel[i_d, i_b] - dofs_state.vel[i_d, i_b])

                    dofs_state.qf_applied[i_d, i_b] = qd.math.clamp(
                        force,
                        dofs_info.force_range[I_d][0],
                        dofs_info.force_range[I_d][1],
                    )

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
                        force = dofs_info.kp[I_d] * rotvec[j] - dofs_info.kv[I_d] * dofs_state.vel[i_d, i_b]
                        dofs_state.qf_applied[i_d, i_b] = qd.math.clamp(
                            force, dofs_info.force_range[I_d][0], dofs_info.force_range[I_d][1]
                        )

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        qd.ndrange(1, static_rigid_sim_config.n_envs)
        if qd.static(static_rigid_sim_config.use_hibernation)
        else qd.ndrange(static_rigid_sim_config.n_dofs_, static_rigid_sim_config.n_envs)
    ):
        for i_1 in (
            range(rigid_info.n_awake_dofs[i_b]) if qd.static(rigid_config.use_hibernation) else qd.static(range(1))
        ):
            if func_check_index_range(i_1, 0, rigid_info.n_awake_dofs[i_b], rigid_config.use_hibernation):
                i_d = rigid_info.awake_dofs[i_1, i_b] if qd.static(rigid_config.use_hibernation) else i_0

                I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d
                dyn_state.dofs.qf_passive[i_d, i_b] = -dyn_info.dofs.damping[I_d] * dyn_state.dofs.vel[i_d, i_b]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        qd.ndrange(1, static_rigid_sim_config.n_envs)
        if qd.static(static_rigid_sim_config.use_hibernation)
        else qd.ndrange(static_rigid_sim_config.n_links_, static_rigid_sim_config.n_envs)
    ):
        for i_1 in (
            range(rigid_info.n_awake_links[i_b]) if qd.static(rigid_config.use_hibernation) else qd.static(range(1))
        ):
            if func_check_index_range(i_1, 0, rigid_info.n_awake_links[i_b], rigid_config.use_hibernation):
                i_l = rigid_info.awake_links[i_1, i_b] if qd.static(rigid_config.use_hibernation) else i_0
                I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l

                if dyn_info.links.n_dofs[I_l] > 0:
                    i_j = dyn_info.links.joint_start[I_l]
                    I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
                    joint_type = dyn_info.joints.type[I_j]

                    if joint_type != gs.JOINT_TYPE.FREE and joint_type != gs.JOINT_TYPE.FIXED:
                        dof_start = dyn_info.links.dof_start[I_l]
                        dof_end = dyn_info.links.dof_end[I_l]

                        for j_d in range(dof_end - dof_start):
                            I_d = [dof_start + j_d, i_b] if qd.static(rigid_config.batch_dofs_info) else dof_start + j_d
                            # Note that using dofs_state instead of qpos here allows qpos to be pulled into qpos0
                            # instead 0: dofs_state.pos = qpos - qpos0
                            func_add_safe_backward(
                                [dof_start + j_d, i_b],
                                -dyn_state.dofs.pos[dof_start + j_d, i_b] * dyn_info.dofs.stiffness[I_d],
                                dyn_state.dofs.qf_passive,
                                BW,
                            )


@qd.func
def func_update_acc(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    update_cacc: qd.template(),
    is_backward: qd.template(),
):
    BW = qd.static(is_backward)

    # Assume this is the outermost loop
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        qd.ndrange(1, static_rigid_sim_config.n_envs)
        if qd.static(static_rigid_sim_config.use_hibernation)
        else qd.ndrange(static_rigid_sim_config.n_entities_, static_rigid_sim_config.n_envs)
    ):
        for i_1 in (
            range(rigid_info.n_awake_entities[i_b]) if qd.static(rigid_config.use_hibernation) else qd.static(range(1))
        ):
            if func_check_index_range(i_1, 0, rigid_info.n_awake_entities[i_b], rigid_config.use_hibernation):
                i_e = rigid_info.awake_entities[i_1, i_b] if qd.static(rigid_config.use_hibernation) else i_0

                for i_l in range(dyn_info.entities.link_start[i_e], dyn_info.entities.link_end[i_e]):
                    I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
                    i_p = dyn_info.links.parent_idx[I_l]

                    if i_p == -1:
                        dyn_state.links.cdd_vel[i_l, i_b] = -rigid_info.gravity[i_b] * (
                            1 - dyn_info.entities.gravity_compensation[i_e]
                        )
                        dyn_state.links.cdd_ang[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)
                        if qd.static(update_cacc):
                            dyn_state.links.cacc_lin[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)
                            dyn_state.links.cacc_ang[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)
                    else:
                        dyn_state.links.cdd_vel[i_l, i_b] = dyn_state.links.cdd_vel[i_p, i_b]
                        dyn_state.links.cdd_ang[i_l, i_b] = dyn_state.links.cdd_ang[i_p, i_b]
                        if qd.static(update_cacc):
                            dyn_state.links.cacc_lin[i_l, i_b] = dyn_state.links.cacc_lin[i_p, i_b]
                            dyn_state.links.cacc_ang[i_l, i_b] = dyn_state.links.cacc_ang[i_p, i_b]

                    for i_d in range(dyn_info.links.dof_start[I_l], dyn_info.links.dof_end[I_l]):
                        # cacc = cacc_parent + cdofdot * qvel + cdof * qacc
                        local_cdd_vel = dyn_state.dofs.cdofd_vel[i_d, i_b] * dyn_state.dofs.vel[i_d, i_b]
                        local_cdd_ang = dyn_state.dofs.cdofd_ang[i_d, i_b] * dyn_state.dofs.vel[i_d, i_b]

                        func_add_safe_backward([i_l, i_b], local_cdd_vel, dyn_state.links.cdd_vel, BW)
                        func_add_safe_backward([i_l, i_b], local_cdd_ang, dyn_state.links.cdd_ang, BW)
                        if qd.static(update_cacc):
                            func_add_safe_backward(
                                [i_l, i_b],
                                local_cdd_vel + dyn_state.dofs.cdof_vel[i_d, i_b] * dyn_state.dofs.acc[i_d, i_b],
                                dyn_state.links.cacc_lin,
                                BW,
                            )
                            func_add_safe_backward(
                                [i_l, i_b],
                                local_cdd_ang + dyn_state.dofs.cdof_ang[i_d, i_b] * dyn_state.dofs.acc[i_d, i_b],
                                dyn_state.links.cacc_ang,
                                BW,
                            )


@qd.func
def func_update_force(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    BW = qd.static(is_backward)

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        qd.ndrange(1, static_rigid_sim_config.n_envs)
        if qd.static(static_rigid_sim_config.use_hibernation)
        else qd.ndrange(static_rigid_sim_config.n_links_, static_rigid_sim_config.n_envs)
    ):
        for i_1 in (
            range(rigid_info.n_awake_links[i_b]) if qd.static(rigid_config.use_hibernation) else qd.static(range(1))
        ):
            if func_check_index_range(i_1, 0, rigid_info.n_awake_links[i_b], rigid_config.use_hibernation):
                i_l = rigid_info.awake_links[i_1, i_b] if qd.static(rigid_config.use_hibernation) else i_0

                f1_ang, f1_vel = gu.inertial_mul(
                    dyn_state.links.cinr_pos[i_l, i_b],
                    dyn_state.links.cinr_inertial[i_l, i_b],
                    dyn_state.links.cinr_mass[i_l, i_b],
                    dyn_state.links.cdd_vel[i_l, i_b],
                    dyn_state.links.cdd_ang[i_l, i_b],
                )
                f2_ang, f2_vel = gu.inertial_mul(
                    dyn_state.links.cinr_pos[i_l, i_b],
                    dyn_state.links.cinr_inertial[i_l, i_b],
                    dyn_state.links.cinr_mass[i_l, i_b],
                    dyn_state.links.cd_vel[i_l, i_b],
                    dyn_state.links.cd_ang[i_l, i_b],
                )
                f3_ang, f3_vel = gu.motion_cross_force(
                    dyn_state.links.cd_ang[i_l, i_b], dyn_state.links.cd_vel[i_l, i_b], f2_ang, f2_vel
                )

                dyn_state.links.cfrc_vel[i_l, i_b] = (
                    f1_vel
                    + f3_vel
                    + dyn_state.links.cfrc_applied_vel[i_l, i_b]
                    + dyn_state.links.cfrc_coupling_vel[i_l, i_b]
                )
                dyn_state.links.cfrc_ang[i_l, i_b] = (
                    f1_ang
                    + f3_ang
                    + dyn_state.links.cfrc_applied_ang[i_l, i_b]
                    + dyn_state.links.cfrc_coupling_ang[i_l, i_b]
                )

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        qd.ndrange(1, static_rigid_sim_config.n_envs)
        if qd.static(static_rigid_sim_config.use_hibernation)
        else qd.ndrange(static_rigid_sim_config.n_entities_, static_rigid_sim_config.n_envs)
    ):
        for i_1 in (
            range(rigid_info.n_awake_entities[i_b]) if qd.static(rigid_config.use_hibernation) else qd.static(range(1))
        ):
            if func_check_index_range(i_1, 0, rigid_info.n_awake_entities[i_b], rigid_config.use_hibernation):
                i_e = rigid_info.awake_entities[i_1, i_b] if qd.static(rigid_config.use_hibernation) else i_0

                for i_l_ in range(dyn_info.entities.n_links[i_e]):
                    i_l = dyn_info.entities.link_end[i_e] - 1 - i_l_
                    I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
                    i_p = dyn_info.links.parent_idx[I_l]
                    I_p = [i_p, i_b]
                    if i_p != -1:
                        func_add_safe_backward(I_p, dyn_state.links.cfrc_vel[i_l, i_b], dyn_state.links.cfrc_vel, BW)
                        func_add_safe_backward(I_p, dyn_state.links.cfrc_ang[i_l, i_b], dyn_state.links.cfrc_ang, BW)

    # Clear coupling forces after use
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for I in qd.grouped(qd.ndrange(*dyn_state.links.cfrc_coupling_ang.shape)):
        dyn_state.links.cfrc_coupling_ang[I] = qd.Vector.zero(gs.qd_float, 3)
        dyn_state.links.cfrc_coupling_vel[I] = qd.Vector.zero(gs.qd_float, 3)


@qd.func
def func_actuation(self):
    if qd.static(self._use_hibernation):
        pass
    else:
        qd.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_l, i_b in qd.ndrange(self.n_links, self._B):
            I_l = [i_l, i_b] if qd.static(self._options.batch_links_info) else i_l
            for i_j in range(self.dyn_info.links.joint_start[I_l], self.dyn_info.links.joint_end[I_l]):
                I_j = [i_j, i_b] if qd.static(self._options.batch_joints_info) else i_j
                joint_type = self.dyn_info.joints.type[I_j]
                q_start = self.dyn_info.joints.q_start[I_j]

                if joint_type == gs.JOINT_TYPE.REVOLUTE or joint_type == gs.JOINT_TYPE.PRISMATIC:
                    gear = -1  # TODO
                    i_d = self.dyn_info.links.dof_start[I_l]
                    self.dyn_state.dofs.act_length[i_d, i_b] = gear * self.qpos[q_start, i_b]
                    self.dyn_state.dofs.qf_actuator[i_d, i_b] = self.dyn_state.dofs.act_length[i_d, i_b]
                else:
                    for i_d in range(self.dyn_info.links.dof_start[I_l], self.dyn_info.links.dof_end[I_l]):
                        self.dyn_state.dofs.act_length[i_d, i_b] = 0.0
                        self.dyn_state.dofs.qf_actuator[i_d, i_b] = self.dyn_state.dofs.act_length[i_d, i_b]


@qd.func
def func_bias_force(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    BW = qd.static(is_backward)

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        qd.ndrange(1, static_rigid_sim_config.n_envs)
        if qd.static(static_rigid_sim_config.use_hibernation)
        else qd.ndrange(static_rigid_sim_config.n_links_, static_rigid_sim_config.n_envs)
    ):
        for i_1 in (
            range(rigid_info.n_awake_links[i_b]) if qd.static(rigid_config.use_hibernation) else qd.static(range(1))
        ):
            if func_check_index_range(i_1, 0, rigid_info.n_awake_links[i_b], rigid_config.use_hibernation):
                i_l = rigid_info.awake_links[i_1, i_b] if qd.static(rigid_config.use_hibernation) else i_0
                I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l

                for i_d in range(dyn_info.links.dof_start[I_l], dyn_info.links.dof_end[I_l]):
                    dyn_state.dofs.qf_bias[i_d, i_b] = dyn_state.dofs.cdof_ang[i_d, i_b].dot(
                        dyn_state.links.cfrc_ang[i_l, i_b]
                    ) + dyn_state.dofs.cdof_vel[i_d, i_b].dot(dyn_state.links.cfrc_vel[i_l, i_b])

                    dyn_state.dofs.force[i_d, i_b] = (
                        dyn_state.dofs.qf_passive[i_d, i_b]
                        - dyn_state.dofs.qf_bias[i_d, i_b]
                        + dyn_state.dofs.qf_applied[i_d, i_b]
                        # + self.dyn_state.dofs.qf_actuator[i_d, i_b]
                    )

                    dyn_state.dofs.qf_smooth[i_d, i_b] = dyn_state.dofs.force[i_d, i_b]


@qd.func
def func_compute_qacc(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    func_solve_mass(dyn_state.dofs.force, dyn_state.dofs.acc_smooth, dyn_info, rigid_info, rigid_config)

    # Assume this is the outermost loop
    if qd.static(static_rigid_sim_config.use_hibernation):
        qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
        for i_b in range(static_rigid_sim_config.n_envs):
            for i_1 in (
                range(rigid_global_info.n_awake_entities[i_b])
                if qd.static(not BW)
                else qd.static(range(static_rigid_sim_config.max_n_awake_entities))
            ):
                if func_check_index_range(i_1, 0, rigid_global_info.n_awake_entities[i_b], True):
                    i_e = rigid_global_info.awake_entities[i_1, i_b]
                    for i_d1_ in (
                        range(entities_info.n_dofs[i_e])
                        if qd.static(not BW)
                        else qd.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                    ):
                        i_d1 = entities_info.dof_start[i_e] + i_d1_
                        if func_check_index_range(i_d1, entities_info.dof_start[i_e], entities_info.dof_end[i_e], BW):
                            dofs_state.acc[i_d1, i_b] = dofs_state.acc_smooth[i_d1, i_b]
    else:
        # Non-hibernation: this is just a per-dof copy, so parallelize directly
        # over (i_d, i_b). For G1 this lifts the launch from
        # n_entities * n_envs (~16k threads) to n_dofs * n_envs (~287k threads),
        # turning a memory-bound copy into a fully bandwidth-saturating kernel.
        n_dofs = static_rigid_sim_config.n_dofs_
        qd.loop_config(
            serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL),
            block_dim=64,
        )
        for i_d, i_b in qd.ndrange(n_dofs, static_rigid_sim_config.n_envs):
            dofs_state.acc[i_d, i_b] = dofs_state.acc_smooth[i_d, i_b]


@qd.func
def func_integrate(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    BW = qd.static(is_backward)

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        (qd.ndrange(1, static_rigid_sim_config.n_envs))
        if qd.static(static_rigid_sim_config.use_hibernation)
        else (qd.ndrange(static_rigid_sim_config.n_dofs_, static_rigid_sim_config.n_envs))
    ):
        for i_1 in (
            range(rigid_info.n_awake_dofs[i_b]) if qd.static(rigid_config.use_hibernation) else qd.static(range(1))
        ):
            if func_check_index_range(i_1, 0, rigid_info.n_awake_dofs[i_b], rigid_config.use_hibernation):
                i_d = rigid_info.awake_dofs[i_1, i_b] if qd.static(rigid_config.use_hibernation) else i_0

                dyn_state.dofs.vel_next[i_d, i_b] = (
                    dyn_state.dofs.vel[i_d, i_b] + dyn_state.dofs.acc[i_d, i_b] * rigid_info.substep_dt[None]
                )

    # Standalone free bodies advance with the implicit midpoint rule under the velocity-implicit integrators: their
    # acc / vel_next are overwritten here so the position loop below integrates with the midpoint velocity, and the
    # loop after it recovers the true next velocity. Gated out of the differentiable path: the Newton iteration
    # carries no adjoint.
    if qd.static(not is_backward and not rigid_config.requires_grad and rigid_config.integrator != gs.integrator.Euler):
        qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
        for i_0, i_b in (
            (qd.ndrange(1, dyn_state.dofs.ctrl_mode.shape[1]))
            if qd.static(rigid_config.use_hibernation)
            else (qd.ndrange(dyn_info.links.root_idx.shape[0], dyn_state.dofs.ctrl_mode.shape[1]))
        ):
            for i_1 in (
                range(rigid_info.n_awake_links[i_b]) if qd.static(rigid_config.use_hibernation) else qd.static(range(1))
            ):
                if func_check_index_range(i_1, 0, rigid_info.n_awake_links[i_b], rigid_config.use_hibernation):
                    i_l = rigid_info.awake_links[i_1, i_b] if qd.static(rigid_config.use_hibernation) else i_0
                    if func_midpoint_eligible(i_l, i_b, dyn_state, dyn_info, rigid_config):
                        func_midpoint_free_body(i_l, i_b, dyn_state, dyn_info, rigid_info, rigid_config)

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        (qd.ndrange(1, static_rigid_sim_config.n_envs))
        if qd.static(static_rigid_sim_config.use_hibernation)
        else (qd.ndrange(static_rigid_sim_config.n_links_, static_rigid_sim_config.n_envs))
    ):
        for i_1 in (
            range(rigid_info.n_awake_links[i_b]) if qd.static(rigid_config.use_hibernation) else qd.static(range(1))
        ):
            if func_check_index_range(i_1, 0, rigid_info.n_awake_links[i_b], rigid_config.use_hibernation):
                i_l = rigid_info.awake_links[i_1, i_b] if qd.static(rigid_config.use_hibernation) else i_0
                I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
                if dyn_info.links.n_dofs[I_l] > 0:
                    EPS = rigid_info.EPS[None]

                    dof_start = dyn_info.links.dof_start[I_l]
                    q_start = dyn_info.links.q_start[I_l]
                    q_end = dyn_info.links.q_end[I_l]

                    i_j = dyn_info.links.joint_start[I_l]
                    I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
                    joint_type = dyn_info.joints.type[I_j]

                    if joint_type == gs.JOINT_TYPE.FREE:
                        pos = qd.Vector(
                            [
                                rigid_info.qpos[q_start, i_b],
                                rigid_info.qpos[q_start + 1, i_b],
                                rigid_info.qpos[q_start + 2, i_b],
                            ]
                        )
                        vel = qd.Vector(
                            [
                                dyn_state.dofs.vel_next[dof_start, i_b],
                                dyn_state.dofs.vel_next[dof_start + 1, i_b],
                                dyn_state.dofs.vel_next[dof_start + 2, i_b],
                            ]
                        )
                        pos = pos + vel * rigid_info.substep_dt[None]
                        for j in qd.static(range(3)):
                            rigid_info.qpos_next[q_start + j, i_b] = pos[j]
                    if joint_type == gs.JOINT_TYPE.SPHERICAL or joint_type == gs.JOINT_TYPE.FREE:
                        rot_offset = 3 if joint_type == gs.JOINT_TYPE.FREE else 0
                        rot0 = qd.Vector(
                            [
                                rigid_info.qpos[q_start + rot_offset + 0, i_b],
                                rigid_info.qpos[q_start + rot_offset + 1, i_b],
                                rigid_info.qpos[q_start + rot_offset + 2, i_b],
                                rigid_info.qpos[q_start + rot_offset + 3, i_b],
                            ]
                        )
                        ang = (
                            qd.Vector(
                                [
                                    dyn_state.dofs.vel_next[dof_start + rot_offset + 0, i_b],
                                    dyn_state.dofs.vel_next[dof_start + rot_offset + 1, i_b],
                                    dyn_state.dofs.vel_next[dof_start + rot_offset + 2, i_b],
                                ]
                            )
                            * rigid_info.substep_dt[None]
                        )
                        qrot = gu.qd_rotvec_to_quat(ang, EPS)
                        rot = gu.qd_transform_quat_by_quat(qrot, rot0)
                        for j in qd.static(range(4)):
                            rigid_info.qpos_next[q_start + j + rot_offset, i_b] = rot[j]
                    else:
                        for j_ in range(q_end - q_start):
                            j = q_start + j_
                            if j < q_end:
                                rigid_info.qpos_next[j, i_b] = (
                                    rigid_info.qpos[j, i_b]
                                    + dyn_state.dofs.vel_next[dof_start + j_, i_b] * rigid_info.substep_dt[None]
                                )

    # Recover the true next velocity of the midpoint-integrated free bodies, whose vel_next held the midpoint
    # velocity for the position update above.
    if qd.static(not is_backward and not rigid_config.requires_grad and rigid_config.integrator != gs.integrator.Euler):
        qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
        for i_0, i_b in (
            (qd.ndrange(1, dyn_state.dofs.ctrl_mode.shape[1]))
            if qd.static(rigid_config.use_hibernation)
            else (qd.ndrange(dyn_info.links.root_idx.shape[0], dyn_state.dofs.ctrl_mode.shape[1]))
        ):
            for i_1 in (
                range(rigid_info.n_awake_links[i_b]) if qd.static(rigid_config.use_hibernation) else qd.static(range(1))
            ):
                if func_check_index_range(i_1, 0, rigid_info.n_awake_links[i_b], rigid_config.use_hibernation):
                    i_l = rigid_info.awake_links[i_1, i_b] if qd.static(rigid_config.use_hibernation) else i_0
                    if func_midpoint_eligible(i_l, i_b, dyn_state, dyn_info, rigid_config):
                        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
                        # Linear DOFs are midpoint-integrated only when the center of mass is off the joint origin
                        # (see func_midpoint_free_body)
                        ipos = dyn_info.links.inertial_pos[I_l]
                        j_start = 3
                        if not (ipos[0] == 0.0 and ipos[1] == 0.0 and ipos[2] == 0.0):
                            j_start = 0
                        for j in range(j_start, 6):
                            i_d = dyn_info.links.dof_start[I_l] + j
                            dyn_state.dofs.vel_next[i_d, i_b] = (
                                2.0 * dyn_state.dofs.vel_next[i_d, i_b] - dyn_state.dofs.vel[i_d, i_b]
                            )


@qd.kernel
def kernel_forward_dynamics_without_qacc(
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    # Backward-only kernel. func_factor_mass is omitted: its reverse is unneeded since the backward pass seeds
    # mass_mat.grad directly via the implicit function theorem (see kernel_manual_compute_qacc_bw in manual_bw.py),
    # skipping the LDL^T factor chain. func_compute_mass_matrix is kept so Quadrants autodiff auto-reverses
    # mass_mat -> links pos / quat.
    func_compute_mass_matrix(
        dyn_state,
        dyn_info,
        rigid_info,
        rigid_config,
        qd.static(rigid_config.integrator == gs.integrator.approximate_implicitfast),
        is_backward,
    )
    func_torque_and_passive_force(dyn_state, constraint_state, dyn_info, rigid_info, rigid_config, is_backward)
    func_update_acc(dyn_state, dyn_info, rigid_info, rigid_config, update_cacc=False, is_backward=is_backward)
    func_update_force(dyn_state, dyn_info, rigid_info, rigid_config, is_backward)
    func_bias_force(dyn_state, dyn_info, rigid_info, rigid_config, is_backward)


@qd.func
def func_implicit_damping(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    EPS = rigid_info.EPS[None]

    EPS = rigid_global_info.EPS[None]

    n_entities = static_rigid_sim_config.n_entities_
    _B = static_rigid_sim_config.n_envs

    # Determine whether the mass matrix must be re-computed to take into account first-order correction terms.
    # Note that avoiding inverting the mass matrix twice would not only speed up simulation but also improving
    # numerical stability as computing post-damping accelerations from forces is not necessary anymore.
    if qd.static(not rigid_config.enable_mujoco_compatibility or rigid_config.integrator == gs.integrator.Euler):
        for i_e, i_b in qd.ndrange(n_entities, _B):
            rigid_info.mass_mat_mask[i_e, i_b] = False

        qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_e, i_b in qd.ndrange(n_entities, _B):
            # Set the mask over the mass blocks ROOTED in this entity (see entities_mass_block_dof_start in
            # array_class.py): the block-root entity factors the whole coupled block, so damping/act_bias anywhere in
            # it - including a merged child - must trigger the refactor.
            blocks_dof_start = rigid_info.entities_mass_block_dof_start[i_e]
            blocks_dof_end = rigid_info.entities_mass_block_dof_end[i_e]
            for i_d in range(blocks_dof_start, blocks_dof_end):
                I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d
                if dyn_info.dofs.damping[I_d] > EPS:
                    rigid_info.mass_mat_mask[i_e, i_b] = True
                if qd.static(rigid_config.integrator != gs.integrator.Euler):
                    if (
                        dyn_state.dofs.ctrl_mode[i_d, i_b] <= gs.CTRL_MODE.VELOCITY
                        and qd.abs(dyn_info.dofs.act_bias[I_d][2]) > EPS
                    ):
                        rigid_info.mass_mat_mask[i_e, i_b] = True

    func_factor_mass(dyn_state, dyn_info, rigid_info, rigid_config, implicit_damping=True)
    func_solve_mass(dyn_state.dofs.force, dyn_state.dofs.acc, dyn_info, rigid_info, rigid_config)

    # Disable pre-computed factorization mask right away
    if qd.static(not rigid_config.enable_mujoco_compatibility or rigid_config.integrator == gs.integrator.Euler):
        for i_e, i_b in qd.ndrange(n_entities, _B):
            rigid_info.mass_mat_mask[i_e, i_b] = True


from genesis.utils.deprecated_module_wrapper import create_virtual_deprecated_module

create_virtual_deprecated_module(__name__, "genesis.engine.solvers.rigid.rigid_solver_dynamics_decomp")
