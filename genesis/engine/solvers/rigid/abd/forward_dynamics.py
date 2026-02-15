"""
Rigid solver dynamics kernel and function definitions.

This module contains Taichi kernel and function definitions for rigid body dynamics
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

import quadrants as ti

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class
from .misc import (
    func_wakeup_entity_and_its_temp_island,
    func_check_index_range,
    func_add_safe_backward,
)


@ti.kernel
def update_qacc_from_qvel_delta(
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    BW = ti.static(is_backward)

    n_dofs = dofs_state.ctrl_mode.shape[0]
    _B = dofs_state.ctrl_mode.shape[1]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in ti.ndrange(1, _B) if ti.static(static_rigid_sim_config.use_hibernation) else ti.ndrange(n_dofs, _B):
        for i_1 in (
            (
                # Dynamic inner loop for forward pass
                range(rigid_global_info.n_awake_dofs[i_b])
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
            if ti.static(not BW)
            else (
                ti.static(range(static_rigid_sim_config.max_n_awake_dofs))  # Static inner loop for backward pass
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
        ):
            if i_1 < (rigid_global_info.n_awake_dofs[i_b] if ti.static(static_rigid_sim_config.use_hibernation) else 1):
                i_d = (
                    rigid_global_info.awake_dofs[i_1, i_b]
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )
                dofs_state.acc[i_d, i_b] = (
                    dofs_state.vel[i_d, i_b] - dofs_state.vel_prev[i_d, i_b]
                ) / rigid_global_info.substep_dt[None]
                dofs_state.vel[i_d, i_b] = dofs_state.vel_prev[i_d, i_b]


@ti.kernel
def update_qvel(
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    BW = ti.static(is_backward)

    _B = dofs_state.vel.shape[1]
    n_dofs = dofs_state.vel.shape[0]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in ti.ndrange(1, _B) if ti.static(static_rigid_sim_config.use_hibernation) else ti.ndrange(n_dofs, _B):
        for i_1 in (
            (
                # Dynamic inner loop for forward pass
                range(rigid_global_info.n_awake_dofs[i_b])
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
            if ti.static(not BW)
            else (
                ti.static(range(static_rigid_sim_config.max_n_awake_dofs))  # Static inner loop for backward pass
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
        ):
            if i_1 < (rigid_global_info.n_awake_dofs[i_b] if ti.static(static_rigid_sim_config.use_hibernation) else 1):
                i_d = (
                    rigid_global_info.awake_dofs[i_1, i_b]
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )
                dofs_state.vel_prev[i_d, i_b] = dofs_state.vel[i_d, i_b]
                dofs_state.vel[i_d, i_b] = (
                    dofs_state.vel[i_d, i_b] + dofs_state.acc[i_d, i_b] * rigid_global_info.substep_dt[None]
                )


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_compute_mass_matrix(
    # taichi variables
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    decompose: ti.template(),
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
@ti.func
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
    static_rigid_sim_config: ti.template(),
    contact_island_state: array_class.ContactIslandState,
    is_backward: ti.template(),
):
    func_compute_mass_matrix(
        implicit_damping=ti.static(static_rigid_sim_config.integrator == gs.integrator.approximate_implicitfast),
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


@ti.kernel(fastcache=gs.use_fastcache)
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
    static_rigid_sim_config: ti.template(),
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


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_update_acc(
    dofs_state: array_class.DofsState,
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
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


@ti.func
def func_vel_at_point(pos_world, link_idx, i_b, links_state: array_class.LinksState):
    """
    Velocity of a certain point on a rigid link.
    """
    vel_rot = links_state.cd_ang[link_idx, i_b].cross(pos_world - links_state.root_COM[link_idx, i_b])
    vel_lin = links_state.cd_vel[link_idx, i_b]
    return vel_rot + vel_lin


@ti.func
def func_compute_mass_matrix(
    implicit_damping: ti.template(),
    # taichi variables
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    BW = ti.static(is_backward)

    # crb initialize
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        ti.ndrange(1, links_state.pos.shape[1])
        if ti.static(static_rigid_sim_config.use_hibernation)
        else ti.ndrange(links_state.pos.shape[0], links_state.pos.shape[1])
    ):
        for i_1 in (
            (
                # Dynamic inner loop for forward pass
                range(rigid_global_info.n_awake_links[i_b])
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
            if ti.static(not BW)
            else (
                ti.static(range(static_rigid_sim_config.max_n_awake_links))  # Static inner loop for backward pass
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_links[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_l = (
                    rigid_global_info.awake_links[i_1, i_b]
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )

                links_state.crb_inertial[i_l, i_b] = links_state.cinr_inertial[i_l, i_b]
                links_state.crb_pos[i_l, i_b] = links_state.cinr_pos[i_l, i_b]
                links_state.crb_quat[i_l, i_b] = links_state.cinr_quat[i_l, i_b]
                links_state.crb_mass[i_l, i_b] = links_state.cinr_mass[i_l, i_b]

    # crb
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        ti.ndrange(1, links_state.pos.shape[1])
        if ti.static(static_rigid_sim_config.use_hibernation)
        else ti.ndrange(entities_info.n_links.shape[0], links_state.pos.shape[1])
    ):
        for i_1 in (
            (
                # Dynamic inner loop for forward pass
                range(rigid_global_info.n_awake_entities[i_b])
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
            if ti.static(not BW)
            else (
                ti.static(range(static_rigid_sim_config.max_n_awake_entities))  # Static inner loop for backward pass
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_entities[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_e = (
                    rigid_global_info.awake_entities[i_1, i_b]
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )

                for i in (
                    range(entities_info.n_links[i_e])
                    if ti.static(not BW)
                    else ti.static(range(static_rigid_sim_config.max_n_links_per_entity))
                ):
                    if func_check_index_range(i, 0, entities_info.n_links[i_e], BW):
                        i_l = entities_info.link_end[i_e] - 1 - i
                        I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
                        i_p = links_info.parent_idx[I_l]
                        I_p = [i_p, i_b]

                        if i_p != -1:
                            func_add_safe_backward(
                                links_state.crb_inertial, I_p, links_state.crb_inertial[i_l, i_b], BW
                            )
                            func_add_safe_backward(links_state.crb_mass, I_p, links_state.crb_mass[i_l, i_b], BW)
                            func_add_safe_backward(links_state.crb_pos, I_p, links_state.crb_pos[i_l, i_b], BW)
                            func_add_safe_backward(links_state.crb_quat, I_p, links_state.crb_quat[i_l, i_b], BW)

    # mass_mat
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        ti.ndrange(1, links_state.pos.shape[1])
        if ti.static(static_rigid_sim_config.use_hibernation)
        else ti.ndrange(links_state.pos.shape[0], links_state.pos.shape[1])
    ):
        for i_1 in (
            (
                # Dynamic inner loop for forward pass
                range(rigid_global_info.n_awake_links[i_b])
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
            if ti.static(not BW)
            else (
                ti.static(range(static_rigid_sim_config.max_n_awake_links))  # Static inner loop for backward pass
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_links[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_l = (
                    rigid_global_info.awake_links[i_1, i_b]
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )
                I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

                for i_d_ in (
                    range(links_info.dof_start[I_l], links_info.dof_end[I_l])
                    if ti.static(not BW)
                    else ti.static(range(static_rigid_sim_config.max_n_dofs_per_link))
                ):
                    i_d = i_d_ if ti.static(not BW) else links_info.dof_start[I_l] + i_d_

                    if func_check_index_range(i_d, links_info.dof_start[I_l], links_info.dof_end[I_l], BW):
                        dofs_state.f_ang[i_d, i_b], dofs_state.f_vel[i_d, i_b] = gu.inertial_mul(
                            links_state.crb_pos[i_l, i_b],
                            links_state.crb_inertial[i_l, i_b],
                            links_state.crb_mass[i_l, i_b],
                            dofs_state.cdof_vel[i_d, i_b],
                            dofs_state.cdof_ang[i_d, i_b],
                        )

    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_0, i_b in (
        ti.ndrange(1, links_state.pos.shape[1])
        if ti.static(static_rigid_sim_config.use_hibernation)
        else ti.ndrange(entities_info.n_links.shape[0], links_state.pos.shape[1])
    ):
        for i_1 in (
            (
                # Dynamic inner loop for forward pass
                range(rigid_global_info.n_awake_entities[i_b])
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
            if ti.static(not BW)
            else (
                ti.static(range(static_rigid_sim_config.max_n_awake_entities))  # Static inner loop for backward pass
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_entities[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_e = (
                    rigid_global_info.awake_entities[i_1, i_b]
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )

                for i_d_, j_d_ in (
                    (
                        # Dynamic inner loop for forward pass
                        ti.ndrange(
                            (entities_info.dof_start[i_e], entities_info.dof_end[i_e]),
                            (entities_info.dof_start[i_e], entities_info.dof_end[i_e]),
                        )
                    )
                    if ti.static(not BW)
                    else (
                        ti.static(  # Static inner loop for backward pass
                            ti.ndrange(
                                static_rigid_sim_config.max_n_dofs_per_entity,
                                static_rigid_sim_config.max_n_dofs_per_entity,
                            )
                        )
                    )
                ):
                    i_d = i_d_ if ti.static(not BW) else entities_info.dof_start[i_e] + i_d_
                    j_d = j_d_ if ti.static(not BW) else entities_info.dof_start[i_e] + j_d_

                    if func_check_index_range(
                        i_d,
                        entities_info.dof_start[i_e],
                        entities_info.dof_end[i_e],
                        BW,
                    ) and func_check_index_range(
                        j_d,
                        entities_info.dof_start[i_e],
                        entities_info.dof_end[i_e],
                        BW,
                    ):
                        rigid_global_info.mass_mat[i_d, j_d, i_b] = (
                            dofs_state.f_ang[i_d, i_b].dot(dofs_state.cdof_ang[j_d, i_b])
                            + dofs_state.f_vel[i_d, i_b].dot(dofs_state.cdof_vel[j_d, i_b])
                        ) * rigid_global_info.mass_parent_mask[i_d, j_d]

                if ti.static(not BW):
                    for i_d in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                        for j_d in range(i_d + 1, entities_info.dof_end[i_e]):
                            rigid_global_info.mass_mat[i_d, j_d, i_b] = rigid_global_info.mass_mat[j_d, i_d, i_b]
                else:
                    for i_d_, j_d_ in ti.static(
                        ti.ndrange(
                            static_rigid_sim_config.max_n_dofs_per_entity,
                            static_rigid_sim_config.max_n_dofs_per_entity,
                        )
                    ):
                        i_d = entities_info.dof_start[i_e] + i_d_
                        j_d = entities_info.dof_start[i_e] + j_d_

                        if i_d < entities_info.dof_end[i_e] and j_d < entities_info.dof_end[i_e] and j_d > i_d:
                            rigid_global_info.mass_mat[i_d, j_d, i_b] = rigid_global_info.mass_mat[j_d, i_d, i_b]

    # Take into account motor armature
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in ti.ndrange(dofs_state.f_ang.shape[0], links_state.pos.shape[1]):
        I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
        func_add_safe_backward(rigid_global_info.mass_mat, (i_d, i_d, i_b), dofs_info.armature[I_d], BW)

    # Take into account first-order correction terms for implicit integration scheme right away
    if ti.static(implicit_damping):
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_d, i_b in ti.ndrange(dofs_state.f_ang.shape[0], links_state.pos.shape[1]):
            I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
            rigid_global_info.mass_mat[i_d, i_d, i_b] = (
                rigid_global_info.mass_mat[i_d, i_d, i_b] + dofs_info.damping[I_d] * rigid_global_info.substep_dt[None]
            )
            if (
                dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.POSITION
                or dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.VELOCITY
            ):
                # qM += d qfrc_actuator / d qvel
                rigid_global_info.mass_mat[i_d, i_d, i_b] = (
                    rigid_global_info.mass_mat[i_d, i_d, i_b] + dofs_info.kv[I_d] * rigid_global_info.substep_dt[None]
                )


@ti.func
def func_factor_mass(
    implicit_damping: ti.template(),
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    BW = ti.static(is_backward)

    if ti.static(not BW):
        n_entities = entities_info.n_links.shape[0]
        _B = dofs_state.ctrl_mode.shape[1]

        if ti.static(
            not static_rigid_sim_config.enable_tiled_cholesky_mass_matrix or static_rigid_sim_config.backend == gs.cpu
        ):
            ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
            for i_e, i_b in ti.ndrange(n_entities, _B):
                if rigid_global_info.mass_mat_mask[i_e, i_b]:
                    entity_dof_start = entities_info.dof_start[i_e]
                    entity_dof_end = entities_info.dof_end[i_e]
                    n_dofs = entities_info.n_dofs[i_e]

                    for i_d in range(entity_dof_start, entity_dof_end):
                        for j_d in range(entity_dof_start, i_d + 1):
                            rigid_global_info.mass_mat_L[i_d, j_d, i_b] = rigid_global_info.mass_mat[i_d, j_d, i_b]

                        if ti.static(implicit_damping):
                            I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
                            rigid_global_info.mass_mat_L[i_d, i_d, i_b] = (
                                rigid_global_info.mass_mat_L[i_d, i_d, i_b]
                                + dofs_info.damping[I_d] * rigid_global_info.substep_dt[None]
                            )
                            if ti.static(static_rigid_sim_config.integrator == gs.integrator.implicitfast):
                                if (dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.POSITION) or (
                                    dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.VELOCITY
                                ):
                                    rigid_global_info.mass_mat_L[i_d, i_d, i_b] = (
                                        rigid_global_info.mass_mat_L[i_d, i_d, i_b]
                                        + dofs_info.kv[I_d] * rigid_global_info.substep_dt[None]
                                    )

                    for i_d_ in range(n_dofs):
                        i_d = entity_dof_end - i_d_ - 1
                        D_inv = 1.0 / rigid_global_info.mass_mat_L[i_d, i_d, i_b]
                        rigid_global_info.mass_mat_D_inv[i_d, i_b] = D_inv

                        for j_d_ in range(i_d - entity_dof_start):
                            j_d = i_d - j_d_ - 1
                            a = rigid_global_info.mass_mat_L[i_d, j_d, i_b] * D_inv
                            for k_d in range(entity_dof_start, j_d + 1):
                                rigid_global_info.mass_mat_L[j_d, k_d, i_b] -= (
                                    a * rigid_global_info.mass_mat_L[i_d, k_d, i_b]
                                )
                            rigid_global_info.mass_mat_L[i_d, j_d, i_b] = a

                        # FIXME: Diagonal coeffs of L are ignored in computations, so no need to update them.
                        rigid_global_info.mass_mat_L[i_d, i_d, i_b] = 1.0
        else:
            BLOCK_DIM = ti.static(32)
            MAX_DOFS_PER_ENTITY = ti.static(static_rigid_sim_config.tiled_n_dofs_per_entity)
            WARP_SIZE = ti.static(32)

            ti.loop_config(block_dim=BLOCK_DIM)
            for i in range(n_entities * _B * BLOCK_DIM):
                tid = i % BLOCK_DIM
                i_e = (i // BLOCK_DIM) % n_entities
                i_b = i // (BLOCK_DIM * n_entities)
                if i_b >= _B:
                    continue

                if rigid_global_info.mass_mat_mask[i_e, i_b]:
                    entity_dof_start = entities_info.dof_start[i_e]
                    entity_dof_end = entities_info.dof_end[i_e]
                    n_dofs = entities_info.n_dofs[i_e]
                    n_lower_tri = n_dofs * (n_dofs + 1) // 2

                    mass_mat = ti.simt.block.SharedArray((MAX_DOFS_PER_ENTITY, MAX_DOFS_PER_ENTITY + 1), gs.ti_float)

                    i_pair = tid
                    while i_pair < n_lower_tri:
                        i_d_ = ti.cast((ti.sqrt(8 * i_pair + 1) - 1) // 2, ti.i32)
                        j_d_ = i_pair - i_d_ * (i_d_ + 1) // 2
                        i_d = entity_dof_start + i_d_
                        j_d = entity_dof_start + j_d_
                        mass_mat[i_d_, j_d_] = rigid_global_info.mass_mat[i_d, j_d, i_b]
                        i_pair = i_pair + BLOCK_DIM
                    ti.simt.block.sync()

                    if ti.static(implicit_damping):
                        i_d_ = tid
                        while i_d_ < n_dofs:
                            i_d = entity_dof_start + i_d_
                            I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
                            mass_mat[i_d_, i_d_] = (
                                mass_mat[i_d_, i_d_] + dofs_info.damping[I_d] * rigid_global_info.substep_dt[None]
                            )
                            if ti.static(static_rigid_sim_config.integrator == gs.integrator.implicitfast):
                                if (dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.POSITION) or (
                                    dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.VELOCITY
                                ):
                                    mass_mat[i_d_, i_d_] = (
                                        mass_mat[i_d_, i_d_] + dofs_info.kv[I_d] * rigid_global_info.substep_dt[None]
                                    )
                            i_d_ = i_d_ + BLOCK_DIM
                        ti.simt.block.sync()

                    for j in range(n_dofs):
                        i_d_ = n_dofs - j - 1
                        i_d = entity_dof_end - j - 1

                        D_inv = 1.0 / mass_mat[i_d_, i_d_]
                        if tid == 0:
                            rigid_global_info.mass_mat_D_inv[i_d, i_b] = D_inv
                            # FIXME: Diagonal coeffs of L are ignored in computations, so no need to update them.
                            rigid_global_info.mass_mat_L[i_d, i_d, i_b] = 1.0

                        j_d_ = i_d_ - 1 - tid
                        while j_d_ >= 0:
                            a = mass_mat[i_d_, j_d_] * D_inv
                            for k_d in range(j_d_ + 1):
                                mass_mat[j_d_, k_d] = mass_mat[j_d_, k_d] - a * mass_mat[i_d_, k_d]
                            mass_mat[i_d_, j_d_] = a
                            j_d_ = j_d_ - BLOCK_DIM
                        if ti.static(static_rigid_sim_config.backend == gs.cuda):
                            if i_d_ <= WARP_SIZE:
                                ti.simt.warp.sync(ti.u32(0xFFFFFFFF))
                            else:
                                ti.simt.block.sync()
                        else:
                            ti.simt.block.sync()

                    i_pair = tid
                    n_strict_lower_tri = n_dofs * (n_dofs - 1) // 2
                    while i_pair < n_strict_lower_tri:
                        i_d_ = ti.cast((ti.sqrt(8 * i_pair + 1) + 1) // 2, ti.i32)
                        j_d_ = i_pair - i_d_ * (i_d_ - 1) // 2
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
        ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_e, i_b in ti.ndrange(entities_info.n_links.shape[0], dofs_state.ctrl_mode.shape[1]):
            if rigid_global_info.mass_mat_mask[i_e, i_b]:
                EPS = rigid_global_info.EPS[None]

                entity_dof_start = entities_info.dof_start[i_e]
                entity_dof_end = entities_info.dof_end[i_e]
                n_dofs = entities_info.n_dofs[i_e]

                for i_d0 in (
                    range(n_dofs)
                    if ti.static(not BW)
                    else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                ):
                    if func_check_index_range(i_d0, 0, n_dofs, BW):
                        i_d = entity_dof_start + i_d0
                        i_pr = (entity_dof_start + entity_dof_end - 1) - i_d
                        for j_d_ in (
                            range(entity_dof_start, i_d + 1)
                            if ti.static(not BW)
                            else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                        ):
                            j_d = j_d_ if ti.static(not BW) else (j_d_ + entities_info.dof_start[i_e])
                            j_pr = (entity_dof_start + entity_dof_end - 1) - j_d
                            if func_check_index_range(j_d, entity_dof_start, i_d + 1, BW):
                                rigid_global_info.mass_mat_L_bw[0, i_pr, j_pr, i_b] = rigid_global_info.mass_mat[
                                    i_d, j_d, i_b
                                ]
                                rigid_global_info.mass_mat_L_bw[0, j_pr, i_pr, i_b] = rigid_global_info.mass_mat[
                                    i_d, j_d, i_b
                                ]

                        if ti.static(implicit_damping):
                            I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
                            ti.atomic_add(
                                rigid_global_info.mass_mat_L_bw[0, i_pr, i_pr, i_b],
                                (dofs_info.damping[I_d] * rigid_global_info.substep_dt[None]),
                            )
                            if ti.static(static_rigid_sim_config.integrator == gs.integrator.implicitfast):
                                if (
                                    dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.POSITION
                                    or dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.VELOCITY
                                ):
                                    ti.atomic_add(
                                        rigid_global_info.mass_mat_L_bw[0, i_pr, i_pr, i_b],
                                        dofs_info.kv[I_d] * rigid_global_info.substep_dt[None],
                                    )

                # Cholesky-Banachiewicz algorithm (in the perturbed indices), access pattern is safe for autodiff
                # https://en.wikipedia.org/wiki/Cholesky_decomposition
                for p_i0 in (
                    range(n_dofs)
                    if ti.static(not BW)
                    else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                ):
                    for p_j0 in (
                        range(p_i0 + 1)
                        if ti.static(not BW)
                        else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                    ):
                        if func_check_index_range(p_i0, 0, n_dofs, BW) and func_check_index_range(
                            p_j0, 0, p_i0 + 1, BW
                        ):
                            # j_pr <= i_pr
                            i_pr = entity_dof_start + p_i0
                            j_pr = entity_dof_start + p_j0

                            sum = gs.ti_float(0.0)
                            for p_k0 in (
                                range(p_j0)
                                if ti.static(not BW)
                                else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                            ):
                                # k_pr < j_pr
                                if func_check_index_range(p_k0, 0, p_j0, BW):
                                    k_pr = entity_dof_start + p_k0
                                    sum = sum + (
                                        rigid_global_info.mass_mat_L_bw[1, i_pr, k_pr, i_b]
                                        * rigid_global_info.mass_mat_L_bw[1, j_pr, k_pr, i_b]
                                    )

                            a = rigid_global_info.mass_mat_L_bw[0, i_pr, j_pr, i_b] - sum
                            b = ti.math.clamp(
                                rigid_global_info.mass_mat_L_bw[1, j_pr, j_pr, i_b],
                                EPS,
                                ti.math.inf,
                            )
                            if p_i0 == p_j0:
                                rigid_global_info.mass_mat_L_bw[1, i_pr, j_pr, i_b] = ti.sqrt(
                                    ti.math.clamp(a, EPS, ti.math.inf)
                                )
                            else:
                                rigid_global_info.mass_mat_L_bw[1, i_pr, j_pr, i_b] = a / b

                for i_d0 in (
                    range(n_dofs)
                    if ti.static(not BW)
                    else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                ):
                    for i_d1 in (
                        range(i_d0 + 1)
                        if ti.static(not BW)
                        else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                    ):
                        if func_check_index_range(i_d0, 0, n_dofs, BW) and func_check_index_range(
                            i_d1, 0, i_d0 + 1, BW
                        ):
                            i_d = entity_dof_start + i_d0
                            j_d = entity_dof_start + i_d1
                            i_pr = (entity_dof_start + entity_dof_end - 1) - i_d
                            j_pr = (entity_dof_start + entity_dof_end - 1) - j_d

                            a = rigid_global_info.mass_mat_L_bw[1, i_pr, i_pr, i_b]
                            rigid_global_info.mass_mat_L[i_d, j_d, i_b] = rigid_global_info.mass_mat_L_bw[
                                1, j_pr, i_pr, i_b
                            ] / ti.math.clamp(a, EPS, ti.math.inf)

                            if i_d == j_d:
                                rigid_global_info.mass_mat_D_inv[i_d, i_b] = 1.0 / (
                                    ti.math.clamp(a**2, EPS, ti.math.inf)
                                )


@ti.func
def func_solve_mass_entity(
    i_e: ti.int32,
    i_b: ti.int32,
    vec: array_class.V_ANNOTATION,
    out: array_class.V_ANNOTATION,
    out_bw: array_class.V_ANNOTATION,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    BW = ti.static(is_backward)

    if rigid_global_info.mass_mat_mask[i_e, i_b]:
        entity_dof_start = entities_info.dof_start[i_e]
        entity_dof_end = entities_info.dof_end[i_e]
        n_dofs = entities_info.n_dofs[i_e]

        # Step 1: Solve w st. L^T @ w = y
        for i_d_ in (
            range(n_dofs) if ti.static(not BW) else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
        ):
            if func_check_index_range(i_d_, 0, n_dofs, BW):
                i_d = entity_dof_end - i_d_ - 1
                curr_out = vec[i_d, i_b]
                if ti.static(BW):
                    out_bw[0, i_d, i_b] = vec[i_d, i_b]

                for j_d_ in (
                    range(i_d + 1, entity_dof_end)
                    if ti.static(not BW)
                    else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                ):
                    j_d = j_d_ if ti.static(not BW) else (j_d_ + entities_info.dof_start[i_e])
                    if func_check_index_range(j_d, i_d + 1, entity_dof_end, BW):
                        # Since we read out[j_d, i_b], and j_d > i_d, which means that out[j_d, i_b] is already
                        # finalized at this point, we don't need to care about AD mutation rule.
                        if ti.static(BW):
                            out_bw[0, i_d, i_b] = (
                                out_bw[0, i_d, i_b] - rigid_global_info.mass_mat_L[j_d, i_d, i_b] * out_bw[0, j_d, i_b]
                            )
                        else:
                            curr_out = curr_out - rigid_global_info.mass_mat_L[j_d, i_d, i_b] * out[j_d, i_b]

                if ti.static(not BW):
                    out[i_d, i_b] = curr_out

        # Step 2: z = D^{-1} w
        for i_d_ in (
            range(entity_dof_start, entity_dof_end)
            if ti.static(not BW)
            else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
        ):
            i_d = i_d_ if ti.static(not BW) else (i_d_ + entities_info.dof_start[i_e])
            if func_check_index_range(i_d, entity_dof_start, entity_dof_end, BW):
                if ti.static(BW):
                    out_bw[1, i_d, i_b] = out_bw[0, i_d, i_b] * rigid_global_info.mass_mat_D_inv[i_d, i_b]
                else:
                    out[i_d, i_b] = out[i_d, i_b] * rigid_global_info.mass_mat_D_inv[i_d, i_b]

        # Step 3: Solve x st. L @ x = z
        for i_d_ in (
            range(entity_dof_start, entity_dof_end)
            if ti.static(not BW)
            else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
        ):
            i_d = i_d_ if ti.static(not BW) else (i_d_ + entities_info.dof_start[i_e])
            if func_check_index_range(i_d, entity_dof_start, entity_dof_end, BW):
                curr_out = out[i_d, i_b]
                if ti.static(BW):
                    curr_out = out_bw[1, i_d, i_b]

                for j_d_ in (
                    range(entity_dof_start, i_d)
                    if ti.static(not BW)
                    else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                ):
                    j_d = j_d_ if ti.static(not BW) else (j_d_ + entities_info.dof_start[i_e])
                    if func_check_index_range(j_d, entity_dof_start, i_d, BW):
                        curr_out = curr_out - rigid_global_info.mass_mat_L[i_d, j_d, i_b] * out[j_d, i_b]

                out[i_d, i_b] = curr_out


@ti.func
def func_solve_mass_batch(
    i_b: ti.int32,
    vec: array_class.V_ANNOTATION,
    out: array_class.V_ANNOTATION,
    out_bw: array_class.V_ANNOTATION,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    BW = ti.static(is_backward)

    # This loop is considered an inner loop
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_0 in (
        (
            # Dynamic inner loop for forward pass
            range(rigid_global_info.n_awake_entities[i_b])
            if ti.static(static_rigid_sim_config.use_hibernation)
            else range(entities_info.n_links.shape[0])
        )
        if ti.static(not BW)
        else (
            ti.static(range(static_rigid_sim_config.max_n_awake_entities))  # Static inner loop for backward pass
            if ti.static(static_rigid_sim_config.use_hibernation)
            else ti.static(range(static_rigid_sim_config.max_n_links_per_entity))
        )
    ):
        if func_check_index_range(i_0, 0, rigid_global_info.n_awake_entities[i_b], BW):
            i_e = (
                rigid_global_info.awake_entities[i_0, i_b]
                if ti.static(static_rigid_sim_config.use_hibernation)
                else i_0
            )
            func_solve_mass_entity(
                i_e, i_b, vec, out, out_bw, entities_info, rigid_global_info, static_rigid_sim_config, is_backward
            )


@ti.func
def func_solve_mass(
    vec: array_class.V_ANNOTATION,
    out: array_class.V_ANNOTATION,
    out_bw: array_class.V_ANNOTATION,  # Should not be None if backward
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    # This loop must be the outermost loop to be differentiable
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_e, i_b in ti.ndrange(entities_info.n_links.shape[0], out.shape[1]):
        func_solve_mass_entity(
            i_e, i_b, vec, out, out_bw, entities_info, rigid_global_info, static_rigid_sim_config, is_backward
        )


@ti.func
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
    static_rigid_sim_config: ti.template(),
    contact_island_state: array_class.ContactIslandState,
    is_backward: ti.template(),
):
    BW = ti.static(is_backward)

    # compute force based on each dof's ctrl mode
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_e, i_b in ti.ndrange(entities_info.n_links.shape[0], dofs_state.ctrl_mode.shape[1]):
        EPS = rigid_global_info.EPS[None]

        wakeup = False
        for i_l_ in (
            range(entities_info.link_start[i_e], entities_info.link_end[i_e])
            if ti.static(not BW)
            else ti.static(range(static_rigid_sim_config.max_n_links_per_entity))
        ):
            i_l = i_l_ if ti.static(not BW) else (i_l_ + entities_info.link_start[i_e])

            if func_check_index_range(i_l, entities_info.link_start[i_e], entities_info.link_end[i_e], BW):
                I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
                if links_info.n_dofs[I_l] > 0:
                    i_j = links_info.joint_start[I_l]
                    I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
                    joint_type = joints_info.type[I_j]

                    for i_d_ in (
                        range(links_info.dof_start[I_l], links_info.dof_end[I_l])
                        if ti.static(not BW)
                        else ti.static(range(static_rigid_sim_config.max_n_dofs_per_link))
                    ):
                        i_d = i_d_ if ti.static(not BW) else (i_d_ + links_info.dof_start[I_l])

                        if func_check_index_range(i_d, links_info.dof_start[I_l], links_info.dof_end[I_l], BW):
                            I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
                            force = gs.ti_float(0.0)
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

                            dofs_state.qf_applied[i_d, i_b] = ti.math.clamp(
                                force,
                                dofs_info.force_range[I_d][0],
                                dofs_info.force_range[I_d][1],
                            )

                            if ti.abs(force) > EPS:
                                wakeup = True

                    dof_start = links_info.dof_start[I_l]
                    if joint_type == gs.JOINT_TYPE.FREE and (
                        dofs_state.ctrl_mode[dof_start + 3, i_b] == gs.CTRL_MODE.POSITION
                        or dofs_state.ctrl_mode[dof_start + 4, i_b] == gs.CTRL_MODE.POSITION
                        or dofs_state.ctrl_mode[dof_start + 5, i_b] == gs.CTRL_MODE.POSITION
                    ):
                        xyz = ti.Vector(
                            [
                                dofs_state.pos[0 + 3 + dof_start, i_b],
                                dofs_state.pos[1 + 3 + dof_start, i_b],
                                dofs_state.pos[2 + 3 + dof_start, i_b],
                            ],
                            dt=gs.ti_float,
                        )

                        ctrl_xyz = ti.Vector(
                            [
                                dofs_state.ctrl_pos[0 + 3 + dof_start, i_b],
                                dofs_state.ctrl_pos[1 + 3 + dof_start, i_b],
                                dofs_state.ctrl_pos[2 + 3 + dof_start, i_b],
                            ],
                            dt=gs.ti_float,
                        )

                        quat = gu.ti_xyz_to_quat(xyz)
                        ctrl_quat = gu.ti_xyz_to_quat(ctrl_xyz)

                        q_diff = gu.ti_transform_quat_by_quat(ctrl_quat, gu.ti_inv_quat(quat))
                        rotvec = gu.ti_quat_to_rotvec(q_diff, EPS)

                        for j in ti.static(range(3)):
                            i_d = dof_start + 3 + j
                            I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
                            force = dofs_info.kp[I_d] * rotvec[j] - dofs_info.kv[I_d] * dofs_state.vel[i_d, i_b]

                            dofs_state.qf_applied[i_d, i_b] = ti.math.clamp(
                                force, dofs_info.force_range[I_d][0], dofs_info.force_range[I_d][1]
                            )

                            if ti.abs(force) > EPS:
                                wakeup = True

        if ti.static(static_rigid_sim_config.use_hibernation):
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

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        ti.ndrange(1, dofs_state.ctrl_mode.shape[1])
        if ti.static(static_rigid_sim_config.use_hibernation)
        else ti.ndrange(dofs_state.ctrl_mode.shape[0], dofs_state.ctrl_mode.shape[1])
    ):
        for i_1 in (
            (
                # Dynamic inner for forward pass
                range(rigid_global_info.n_awake_dofs[i_b])
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
            if ti.static(not BW)
            else (
                ti.static(range(static_rigid_sim_config.max_n_awake_dofs))  # Static inner for backward pass
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_dofs[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_d = (
                    rigid_global_info.awake_dofs[i_1, i_b]
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )

                I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
                dofs_state.qf_passive[i_d, i_b] = -dofs_info.damping[I_d] * dofs_state.vel[i_d, i_b]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        ti.ndrange(1, dofs_state.ctrl_mode.shape[1])
        if ti.static(static_rigid_sim_config.use_hibernation)
        else ti.ndrange(links_info.root_idx.shape[0], dofs_state.ctrl_mode.shape[1])
    ):
        for i_1 in (
            (
                # Dynamic inner for forward pass
                range(rigid_global_info.n_awake_links[i_b])
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
            if ti.static(not BW)
            else (
                ti.static(range(static_rigid_sim_config.max_n_awake_links))  # Static inner for backward pass
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_links[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_l = (
                    rigid_global_info.awake_links[i_1, i_b]
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )
                I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

                if links_info.n_dofs[I_l] > 0:
                    i_j = links_info.joint_start[I_l]
                    I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
                    joint_type = joints_info.type[I_j]

                    if joint_type != gs.JOINT_TYPE.FREE and joint_type != gs.JOINT_TYPE.FIXED:
                        dof_start = links_info.dof_start[I_l]
                        dof_end = links_info.dof_end[I_l]

                        for j_d in (
                            range(dof_end - dof_start)
                            if ti.static(not BW)
                            else ti.static(range(static_rigid_sim_config.max_n_dofs_per_link))
                        ):
                            if func_check_index_range(j_d, 0, dof_end - dof_start, BW):
                                I_d = (
                                    [dof_start + j_d, i_b]
                                    if ti.static(static_rigid_sim_config.batch_dofs_info)
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


@ti.func
def func_update_acc(
    update_cacc: ti.template(),
    dofs_state: array_class.DofsState,
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    BW = ti.static(is_backward)

    # Assume this is the outermost loop
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        ti.ndrange(1, dofs_state.ctrl_mode.shape[1])
        if ti.static(static_rigid_sim_config.use_hibernation)
        else ti.ndrange(entities_info.n_links.shape[0], dofs_state.ctrl_mode.shape[1])
    ):
        for i_1 in (
            (
                # Dynamic inner loop for forward pass
                range(rigid_global_info.n_awake_entities[i_b])
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
            if ti.static(not BW)
            else (
                ti.static(range(static_rigid_sim_config.max_n_awake_entities))  # Static inner loop for backward pass
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_entities[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_e = (
                    rigid_global_info.awake_entities[i_1, i_b]
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )

                for i_l_ in (
                    range(entities_info.link_start[i_e], entities_info.link_end[i_e])
                    if ti.static(not BW)
                    else ti.static(range(static_rigid_sim_config.max_n_links_per_entity))
                ):
                    i_l = i_l_ if ti.static(not BW) else (i_l_ + entities_info.link_start[i_e])

                    if func_check_index_range(
                        i_l,
                        entities_info.link_start[i_e],
                        entities_info.link_end[i_e],
                        BW,
                    ):
                        I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
                        i_p = links_info.parent_idx[I_l]

                        if i_p == -1:
                            links_state.cdd_vel[i_l, i_b] = -rigid_global_info.gravity[i_b] * (
                                1 - entities_info.gravity_compensation[i_e]
                            )
                            links_state.cdd_ang[i_l, i_b] = ti.Vector.zero(gs.ti_float, 3)
                            if ti.static(update_cacc):
                                links_state.cacc_lin[i_l, i_b] = ti.Vector.zero(gs.ti_float, 3)
                                links_state.cacc_ang[i_l, i_b] = ti.Vector.zero(gs.ti_float, 3)
                        else:
                            links_state.cdd_vel[i_l, i_b] = links_state.cdd_vel[i_p, i_b]
                            links_state.cdd_ang[i_l, i_b] = links_state.cdd_ang[i_p, i_b]
                            if ti.static(update_cacc):
                                links_state.cacc_lin[i_l, i_b] = links_state.cacc_lin[i_p, i_b]
                                links_state.cacc_ang[i_l, i_b] = links_state.cacc_ang[i_p, i_b]

                        for i_d_ in (
                            range(links_info.dof_start[I_l], links_info.dof_end[I_l])
                            if ti.static(not BW)
                            else ti.static(range(static_rigid_sim_config.max_n_dofs_per_link))
                        ):
                            i_d = i_d_ if ti.static(not BW) else (i_d_ + links_info.dof_start[I_l])

                            if func_check_index_range(i_d, links_info.dof_start[I_l], links_info.dof_end[I_l], BW):
                                # cacc = cacc_parent + cdofdot * qvel + cdof * qacc
                                local_cdd_vel = dofs_state.cdofd_vel[i_d, i_b] * dofs_state.vel[i_d, i_b]
                                local_cdd_ang = dofs_state.cdofd_ang[i_d, i_b] * dofs_state.vel[i_d, i_b]

                                func_add_safe_backward(links_state.cdd_vel, [i_l, i_b], local_cdd_vel, BW)
                                func_add_safe_backward(links_state.cdd_ang, [i_l, i_b], local_cdd_ang, BW)
                                if ti.static(update_cacc):
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


@ti.func
def func_update_force(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    BW = ti.static(is_backward)

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        ti.ndrange(1, links_state.pos.shape[1])
        if ti.static(static_rigid_sim_config.use_hibernation)
        else ti.ndrange(links_info.root_idx.shape[0], links_state.pos.shape[1])
    ):
        for i_1 in (
            (
                # Dynamic inner loop for forward pass
                range(rigid_global_info.n_awake_links[i_b])
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
            if ti.static(not BW)
            else (
                ti.static(range(static_rigid_sim_config.max_n_awake_links))  # Static inner loop for backward pass
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_links[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_l = (
                    rigid_global_info.awake_links[i_1, i_b]
                    if ti.static(static_rigid_sim_config.use_hibernation)
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

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        ti.ndrange(1, links_state.pos.shape[1])
        if ti.static(static_rigid_sim_config.use_hibernation)
        else ti.ndrange(entities_info.n_links.shape[0], links_state.pos.shape[1])
    ):
        for i_1 in (
            (
                # Dynamic inner loop for forward pass
                range(rigid_global_info.n_awake_entities[i_b])
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
            if ti.static(not BW)
            else (
                ti.static(range(static_rigid_sim_config.max_n_awake_entities))  # Static inner loop for backward pass
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_entities[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_e = (
                    rigid_global_info.awake_entities[i_1, i_b]
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )

                for i_l_ in (
                    range(entities_info.n_links[i_e])
                    if ti.static(not BW)
                    else ti.static(range(static_rigid_sim_config.max_n_links_per_entity))
                ):
                    if func_check_index_range(i_l_, 0, entities_info.n_links[i_e], BW):
                        i_l = entities_info.link_end[i_e] - 1 - i_l_
                        I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
                        i_p = links_info.parent_idx[I_l]
                        I_p = [i_p, i_b]
                        if i_p != -1:
                            func_add_safe_backward(links_state.cfrc_vel, I_p, links_state.cfrc_vel[i_l, i_b], BW)
                            func_add_safe_backward(links_state.cfrc_ang, I_p, links_state.cfrc_ang[i_l, i_b], BW)

    # Clear coupling forces after use
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for I in ti.grouped(ti.ndrange(*links_state.cfrc_coupling_ang.shape)):
        links_state.cfrc_coupling_ang[I] = ti.Vector.zero(gs.ti_float, 3)
        links_state.cfrc_coupling_vel[I] = ti.Vector.zero(gs.ti_float, 3)


@ti.func
def func_actuation(self):
    if ti.static(self._use_hibernation):
        pass
    else:
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
        for i_l, i_b in ti.ndrange(self.n_links, self._B):
            I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
            for i_j in range(self.links_info.joint_start[I_l], self.links_info.joint_end[I_l]):
                I_j = [i_j, i_b] if ti.static(self._options.batch_joints_info) else i_j
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


@ti.func
def func_bias_force(
    dofs_state: array_class.DofsState,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    BW = ti.static(is_backward)

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        ti.ndrange(1, dofs_state.ctrl_mode.shape[1])
        if ti.static(static_rigid_sim_config.use_hibernation)
        else ti.ndrange(links_info.root_idx.shape[0], dofs_state.ctrl_mode.shape[1])
    ):
        for i_1 in (
            (
                # Dynamic inner loop for forward pass
                range(rigid_global_info.n_awake_links[i_b])
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
            if ti.static(not BW)
            else (
                ti.static(range(static_rigid_sim_config.max_n_awake_links))  # Static inner loop for backward pass
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_links[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_l = (
                    rigid_global_info.awake_links[i_1, i_b]
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )
                I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

                for i_d_ in (
                    range(links_info.dof_start[I_l], links_info.dof_end[I_l])
                    if ti.static(not BW)
                    else ti.static(range(static_rigid_sim_config.max_n_dofs_per_link))
                ):
                    i_d = i_d_ if ti.static(not BW) else (i_d_ + links_info.dof_start[I_l])
                    if func_check_index_range(i_d, links_info.dof_start[I_l], links_info.dof_end[I_l], BW):
                        dofs_state.qf_bias[i_d, i_b] = dofs_state.cdof_ang[i_d, i_b].dot(
                            links_state.cfrc_ang[i_l, i_b]
                        ) + dofs_state.cdof_vel[i_d, i_b].dot(links_state.cfrc_vel[i_l, i_b])

                        dofs_state.force[i_d, i_b] = (
                            dofs_state.qf_passive[i_d, i_b]
                            - dofs_state.qf_bias[i_d, i_b]
                            + dofs_state.qf_applied[i_d, i_b]
                            # + self.dofs_state.qf_actuator[i_d, i_b]
                        )

                        dofs_state.qf_smooth[i_d, i_b] = dofs_state.force[i_d, i_b]


@ti.kernel
def kernel_compute_qacc(
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    func_compute_qacc(
        dofs_state=dofs_state,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        is_backward=is_backward,
    )


@ti.func
def func_compute_qacc(
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    BW = ti.static(is_backward)

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
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_0, i_b in (
        ti.ndrange(1, dofs_state.ctrl_mode.shape[1])
        if ti.static(static_rigid_sim_config.use_hibernation)
        else ti.ndrange(entities_info.n_links.shape[0], dofs_state.ctrl_mode.shape[1])
    ):
        for i_1 in (
            (
                # Dynamic inner loop for forward pass
                range(rigid_global_info.n_awake_entities[i_b])
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
            if ti.static(not BW)
            else (
                ti.static(range(static_rigid_sim_config.max_n_awake_entities))  # Static inner loop for backward pass
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_entities[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_e = (
                    rigid_global_info.awake_entities[i_1, i_b]
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )

                for i_d1_ in (
                    range(entities_info.n_dofs[i_e])
                    if ti.static(not BW)
                    else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                ):
                    i_d1 = entities_info.dof_start[i_e] + i_d1_
                    if func_check_index_range(i_d1, entities_info.dof_start[i_e], entities_info.dof_end[i_e], BW):
                        dofs_state.acc[i_d1, i_b] = dofs_state.acc_smooth[i_d1, i_b]


@ti.func
def func_integrate(
    dofs_state: array_class.DofsState,
    links_info: array_class.LinksInfo,
    joints_info: array_class.JointsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    BW = ti.static(is_backward)

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        (ti.ndrange(1, dofs_state.ctrl_mode.shape[1]))
        if ti.static(static_rigid_sim_config.use_hibernation)
        else (ti.ndrange(dofs_state.ctrl_mode.shape[0], dofs_state.ctrl_mode.shape[1]))
    ):
        for i_1 in (
            (
                # Dynamic inner loop for forward pass
                range(rigid_global_info.n_awake_dofs[i_b])
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
            if ti.static(not BW)
            else (
                ti.static(range(static_rigid_sim_config.max_n_awake_dofs))  # Static inner loop for backward pass
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_dofs[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_d = (
                    rigid_global_info.awake_dofs[i_1, i_b]
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )

                dofs_state.vel_next[i_d, i_b] = (
                    dofs_state.vel[i_d, i_b] + dofs_state.acc[i_d, i_b] * rigid_global_info.substep_dt[None]
                )

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        (ti.ndrange(1, dofs_state.ctrl_mode.shape[1]))
        if ti.static(static_rigid_sim_config.use_hibernation)
        else (ti.ndrange(links_info.root_idx.shape[0], dofs_state.ctrl_mode.shape[1]))
    ):
        for i_1 in (
            (
                # Dynamic inner loop for forward pass
                range(rigid_global_info.n_awake_links[i_b])
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
            if ti.static(not BW)
            else (
                ti.static(range(static_rigid_sim_config.max_n_awake_links))  # Static inner loop for backward pass
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_links[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_l = (
                    rigid_global_info.awake_links[i_1, i_b]
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )
                I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
                if links_info.n_dofs[I_l] > 0:
                    EPS = rigid_global_info.EPS[None]

                    dof_start = links_info.dof_start[I_l]
                    q_start = links_info.q_start[I_l]
                    q_end = links_info.q_end[I_l]

                    i_j = links_info.joint_start[I_l]
                    I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
                    joint_type = joints_info.type[I_j]

                    if joint_type == gs.JOINT_TYPE.FREE:
                        pos = ti.Vector(
                            [
                                rigid_global_info.qpos[q_start, i_b],
                                rigid_global_info.qpos[q_start + 1, i_b],
                                rigid_global_info.qpos[q_start + 2, i_b],
                            ]
                        )
                        vel = ti.Vector(
                            [
                                dofs_state.vel_next[dof_start, i_b],
                                dofs_state.vel_next[dof_start + 1, i_b],
                                dofs_state.vel_next[dof_start + 2, i_b],
                            ]
                        )
                        # Backward pass requires atomic add
                        if ti.static(BW):
                            ti.atomic_add(pos, vel * rigid_global_info.substep_dt[None])
                        else:
                            pos = pos + vel * rigid_global_info.substep_dt[None]
                        for j in ti.static(range(3)):
                            rigid_global_info.qpos_next[q_start + j, i_b] = pos[j]
                    if joint_type == gs.JOINT_TYPE.SPHERICAL or joint_type == gs.JOINT_TYPE.FREE:
                        rot_offset = 3 if joint_type == gs.JOINT_TYPE.FREE else 0
                        rot0 = ti.Vector(
                            [
                                rigid_global_info.qpos[q_start + rot_offset + 0, i_b],
                                rigid_global_info.qpos[q_start + rot_offset + 1, i_b],
                                rigid_global_info.qpos[q_start + rot_offset + 2, i_b],
                                rigid_global_info.qpos[q_start + rot_offset + 3, i_b],
                            ]
                        )
                        ang = (
                            ti.Vector(
                                [
                                    dofs_state.vel_next[dof_start + rot_offset + 0, i_b],
                                    dofs_state.vel_next[dof_start + rot_offset + 1, i_b],
                                    dofs_state.vel_next[dof_start + rot_offset + 2, i_b],
                                ]
                            )
                            * rigid_global_info.substep_dt[None]
                        )
                        qrot = gu.ti_rotvec_to_quat(ang, EPS)
                        rot = gu.ti_transform_quat_by_quat(qrot, rot0)
                        for j in ti.static(range(4)):
                            rigid_global_info.qpos_next[q_start + j + rot_offset, i_b] = rot[j]
                    else:
                        for j_ in (
                            (range(q_end - q_start))
                            if ti.static(not BW)
                            else (ti.static(range(static_rigid_sim_config.max_n_qs_per_link)))
                        ):
                            j = q_start + j_
                            if j < q_end:
                                rigid_global_info.qpos_next[j, i_b] = (
                                    rigid_global_info.qpos[j, i_b]
                                    + dofs_state.vel_next[dof_start + j_, i_b] * rigid_global_info.substep_dt[None]
                                )


@ti.kernel
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
    static_rigid_sim_config: ti.template(),
    contact_island_state: array_class.ContactIslandState,
    is_backward: ti.template(),
):
    func_compute_mass_matrix(
        implicit_damping=ti.static(static_rigid_sim_config.integrator == gs.integrator.approximate_implicitfast),
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


@ti.func
def func_implicit_damping(
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    BW = ti.static(is_backward)

    EPS = rigid_global_info.EPS[None]

    n_entities = entities_info.dof_start.shape[0]
    _B = dofs_state.ctrl_mode.shape[1]

    # Determine whether the mass matrix must be re-computed to take into account first-order correction terms.
    # Note that avoiding inverting the mass matrix twice would not only speed up simulation but also improving
    # numerical stability as computing post-damping accelerations from forces is not necessary anymore.
    if ti.static(
        not static_rigid_sim_config.enable_mujoco_compatibility
        or static_rigid_sim_config.integrator == gs.integrator.Euler
    ):
        for i_e, i_b in ti.ndrange(n_entities, _B):
            rigid_global_info.mass_mat_mask[i_e, i_b] = False

        ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
        for i_e, i_b in ti.ndrange(n_entities, _B):
            entity_dof_start = entities_info.dof_start[i_e]
            entity_dof_end = entities_info.dof_end[i_e]
            for i_d_ in (
                range(entity_dof_start, entity_dof_end)
                if ti.static(not BW)
                else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
            ):
                i_d = i_d_ if ti.static(not BW) else entities_info.dof_start[i_e] + i_d_
                if i_d < entity_dof_end:
                    I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
                    if dofs_info.damping[I_d] > EPS:
                        rigid_global_info.mass_mat_mask[i_e, i_b] = True
                    if ti.static(static_rigid_sim_config.integrator != gs.integrator.Euler):
                        if (
                            dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.POSITION
                            or dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.VELOCITY
                        ) and dofs_info.kv[I_d] > EPS:
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
    if ti.static(
        not static_rigid_sim_config.enable_mujoco_compatibility
        or static_rigid_sim_config.integrator == gs.integrator.Euler
    ):
        for i_e, i_b in ti.ndrange(n_entities, _B):
            rigid_global_info.mass_mat_mask[i_e, i_b] = True


from genesis.utils.deprecated_module_wrapper import create_virtual_deprecated_module

create_virtual_deprecated_module(__name__, "genesis.engine.solvers.rigid.rigid_solver_dynamics_decomp")
