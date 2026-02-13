"""
Contact management and utility functions for the rigid body collider.

This module contains functions for adding contacts, computing tolerances,
and managing contact data including reset/clear operations.
"""

import quadrants as ti

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu


@ti.func
def rotaxis(vecin, i0, i1, i2, f0, f1, f2):
    vecres = ti.Vector([0.0, 0.0, 0.0], dt=gs.ti_float)
    vecres[0] = vecin[i0] * f0
    vecres[1] = vecin[i1] * f1
    vecres[2] = vecin[i2] * f2
    return vecres


@ti.func
def rotmatx(matin, i0, i1, i2, f0, f1, f2):
    matres = ti.Matrix.zero(gs.ti_float, 3, 3)
    matres[0, :] = matin[i0, :] * f0
    matres[1, :] = matin[i1, :] * f1
    matres[2, :] = matin[i2, :] * f2
    return matres


@ti.kernel(fastcache=gs.use_fastcache)
def collider_kernel_reset(
    envs_idx: ti.types.ndarray(),
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    cache_only: ti.template(),
):
    max_possible_pairs = collider_state.contact_cache.normal.shape[0]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]

        if ti.static(not cache_only):
            collider_state.first_time[i_b] = True

        for i_pair in range(max_possible_pairs):
            collider_state.contact_cache.normal[i_pair, i_b] = ti.Vector.zero(gs.ti_float, 3)


# only used with hibernation ??
@ti.kernel(fastcache=gs.use_fastcache)
def kernel_collider_clear(
    envs_idx: ti.types.ndarray(),
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]

        if ti.static(static_rigid_sim_config.use_hibernation):
            collider_state.n_contacts_hibernated[i_b] = 0

            # advect hibernated contacts
            for i_c in range(collider_state.n_contacts[i_b]):
                i_la = collider_state.contact_data.link_a[i_c, i_b]
                i_lb = collider_state.contact_data.link_b[i_c, i_b]

                I_la = [i_la, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_la
                I_lb = [i_lb, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_lb

                # pair of hibernated-fixed links -> hibernated contact
                # TODO: we should also include hibernated-hibernated links and wake up the whole contact island
                # once a new collision is detected
                if (links_state.hibernated[i_la, i_b] and links_info.is_fixed[I_lb]) or (
                    links_state.hibernated[i_lb, i_b] and links_info.is_fixed[I_la]
                ):
                    i_c_hibernated = collider_state.n_contacts_hibernated[i_b]
                    if i_c != i_c_hibernated:
                        # Copying all fields of class StructContactData:
                        # fmt: off
                        collider_state.contact_data.geom_a[i_c_hibernated, i_b] = collider_state.contact_data.geom_a[i_c, i_b]
                        collider_state.contact_data.geom_b[i_c_hibernated, i_b] = collider_state.contact_data.geom_b[i_c, i_b]
                        collider_state.contact_data.penetration[i_c_hibernated, i_b] = collider_state.contact_data.penetration[i_c, i_b]
                        collider_state.contact_data.normal[i_c_hibernated, i_b] = collider_state.contact_data.normal[i_c, i_b]
                        collider_state.contact_data.pos[i_c_hibernated, i_b] = collider_state.contact_data.pos[i_c, i_b]
                        collider_state.contact_data.friction[i_c_hibernated, i_b] = collider_state.contact_data.friction[i_c, i_b]
                        collider_state.contact_data.sol_params[i_c_hibernated, i_b] = collider_state.contact_data.sol_params[i_c, i_b]
                        collider_state.contact_data.force[i_c_hibernated, i_b] = collider_state.contact_data.force[i_c, i_b]
                        collider_state.contact_data.link_a[i_c_hibernated, i_b] = collider_state.contact_data.link_a[i_c, i_b]
                        collider_state.contact_data.link_b[i_c_hibernated, i_b] = collider_state.contact_data.link_b[i_c, i_b]
                        # fmt: on

                    collider_state.n_contacts_hibernated[i_b] = i_c_hibernated + 1

        # Clear contacts: when hibernation is enabled, only clear non-hibernated contacts.
        # The hibernated contacts (positions 0 to n_contacts_hibernated-1) were just advected and should be preserved.
        for i_c in range(collider_state.n_contacts[i_b]):
            should_clear = True
            if ti.static(static_rigid_sim_config.use_hibernation):
                # Only clear if this is not a hibernated contact
                should_clear = i_c >= collider_state.n_contacts_hibernated[i_b]
            if should_clear:
                collider_state.contact_data.link_a[i_c, i_b] = -1
                collider_state.contact_data.link_b[i_c, i_b] = -1
                collider_state.contact_data.geom_a[i_c, i_b] = -1
                collider_state.contact_data.geom_b[i_c, i_b] = -1
                collider_state.contact_data.penetration[i_c, i_b] = 0.0
                collider_state.contact_data.pos[i_c, i_b] = ti.Vector.zero(gs.ti_float, 3)
                collider_state.contact_data.normal[i_c, i_b] = ti.Vector.zero(gs.ti_float, 3)
                collider_state.contact_data.force[i_c, i_b] = ti.Vector.zero(gs.ti_float, 3)

        if ti.static(static_rigid_sim_config.use_hibernation):
            collider_state.n_contacts[i_b] = collider_state.n_contacts_hibernated[i_b]
        else:
            collider_state.n_contacts[i_b] = 0


@ti.kernel(fastcache=gs.use_fastcache)
def collider_kernel_get_contacts(
    is_padded: ti.template(),
    iout: ti.types.ndarray(),
    fout: ti.types.ndarray(),
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
):
    _B = collider_state.active_buffer.shape[1]

    # TODO: Better implementation from quadrants for this kind of reduction.
    n_contacts_max = gs.ti_int(0)
    ti.loop_config(serialize=True)
    for i_b in range(_B):
        n_contacts = collider_state.n_contacts[i_b]
        if n_contacts > n_contacts_max:
            n_contacts_max = n_contacts

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        i_c_start = gs.ti_int(0)
        if ti.static(is_padded):
            i_c_start = i_b * n_contacts_max
        else:
            for j_b in range(i_b):
                i_c_start = i_c_start + collider_state.n_contacts[j_b]

        for i_c_ in range(collider_state.n_contacts[i_b]):
            i_c = i_c_start + i_c_

            iout[i_c, 0] = collider_state.contact_data.link_a[i_c_, i_b]
            iout[i_c, 1] = collider_state.contact_data.link_b[i_c_, i_b]
            iout[i_c, 2] = collider_state.contact_data.geom_a[i_c_, i_b]
            iout[i_c, 3] = collider_state.contact_data.geom_b[i_c_, i_b]
            fout[i_c, 0] = collider_state.contact_data.penetration[i_c_, i_b]
            for j in ti.static(range(3)):
                fout[i_c, 1 + j] = collider_state.contact_data.pos[i_c_, i_b][j]
                fout[i_c, 4 + j] = collider_state.contact_data.normal[i_c_, i_b][j]
                fout[i_c, 7 + j] = collider_state.contact_data.force[i_c_, i_b][j]


@ti.func
def func_add_contact(
    i_ga,
    i_gb,
    normal: ti.types.vector(3),
    contact_pos: ti.types.vector(3),
    penetration,
    i_b,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    errno: array_class.V_ANNOTATION,
):
    i_c = collider_state.n_contacts[i_b]
    if i_c < collider_info.max_contact_pairs[None]:
        friction_a = geoms_info.friction[i_ga] * geoms_state.friction_ratio[i_ga, i_b]
        friction_b = geoms_info.friction[i_gb] * geoms_state.friction_ratio[i_gb, i_b]

        # b to a
        collider_state.contact_data.geom_a[i_c, i_b] = i_ga
        collider_state.contact_data.geom_b[i_c, i_b] = i_gb
        collider_state.contact_data.normal[i_c, i_b] = normal
        collider_state.contact_data.pos[i_c, i_b] = contact_pos
        collider_state.contact_data.penetration[i_c, i_b] = penetration
        collider_state.contact_data.friction[i_c, i_b] = ti.max(ti.max(friction_a, friction_b), 1e-2)
        collider_state.contact_data.sol_params[i_c, i_b] = 0.5 * (
            geoms_info.sol_params[i_ga] + geoms_info.sol_params[i_gb]
        )
        collider_state.contact_data.link_a[i_c, i_b] = geoms_info.link_idx[i_ga]
        collider_state.contact_data.link_b[i_c, i_b] = geoms_info.link_idx[i_gb]

        collider_state.n_contacts[i_b] = i_c + 1
    else:
        errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_COLLISION_PAIRS


@ti.func
def func_set_contact(
    i_ga,
    i_gb,
    normal: ti.types.vector(3),
    contact_pos: ti.types.vector(3),
    penetration,
    i_b,
    i_c,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
):
    """
    Set the contact data for the contact [i_c]. This is used for the backward pass, which parallelizes over the entire
    contact data.
    """
    friction_a = geoms_info.friction[i_ga] * geoms_state.friction_ratio[i_ga, i_b]
    friction_b = geoms_info.friction[i_gb] * geoms_state.friction_ratio[i_gb, i_b]

    # b to a
    collider_state.contact_data.geom_a[i_c, i_b] = i_ga
    collider_state.contact_data.geom_b[i_c, i_b] = i_gb
    collider_state.contact_data.normal[i_c, i_b] = normal
    collider_state.contact_data.pos[i_c, i_b] = contact_pos
    collider_state.contact_data.penetration[i_c, i_b] = penetration
    collider_state.contact_data.friction[i_c, i_b] = ti.max(ti.max(friction_a, friction_b), 1e-2)
    collider_state.contact_data.sol_params[i_c, i_b] = 0.5 * (geoms_info.sol_params[i_ga] + geoms_info.sol_params[i_gb])
    collider_state.contact_data.link_a[i_c, i_b] = geoms_info.link_idx[i_ga]
    collider_state.contact_data.link_b[i_c, i_b] = geoms_info.link_idx[i_gb]


@ti.func
def func_add_diff_contact_input(
    i_ga,
    i_gb,
    i_b,
    i_d,
    gjk_state: array_class.GJKState,
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
):
    i_c = collider_state.n_contacts[i_b]
    if i_c < collider_info.max_contact_pairs[None]:
        collider_state.diff_contact_input.geom_a[i_b, i_c] = i_ga
        collider_state.diff_contact_input.geom_b[i_b, i_c] = i_gb
        collider_state.diff_contact_input.local_pos1_a[i_b, i_c] = gjk_state.diff_contact_input.local_pos1_a[i_b, i_d]
        collider_state.diff_contact_input.local_pos1_b[i_b, i_c] = gjk_state.diff_contact_input.local_pos1_b[i_b, i_d]
        collider_state.diff_contact_input.local_pos1_c[i_b, i_c] = gjk_state.diff_contact_input.local_pos1_c[i_b, i_d]
        collider_state.diff_contact_input.local_pos2_a[i_b, i_c] = gjk_state.diff_contact_input.local_pos2_a[i_b, i_d]
        collider_state.diff_contact_input.local_pos2_b[i_b, i_c] = gjk_state.diff_contact_input.local_pos2_b[i_b, i_d]
        collider_state.diff_contact_input.local_pos2_c[i_b, i_c] = gjk_state.diff_contact_input.local_pos2_c[i_b, i_d]
        collider_state.diff_contact_input.w_local_pos1[i_b, i_c] = gjk_state.diff_contact_input.w_local_pos1[i_b, i_d]
        collider_state.diff_contact_input.w_local_pos2[i_b, i_c] = gjk_state.diff_contact_input.w_local_pos2[i_b, i_d]
        # The first contact point is the reference contact point
        collider_state.diff_contact_input.ref_id[i_b, i_c] = i_c - i_d
        collider_state.diff_contact_input.ref_penetration[i_b, i_c] = gjk_state.diff_contact_input.ref_penetration[
            i_b, i_d
        ]


@ti.func
def func_compute_tolerance(
    i_ga,
    i_gb,
    i_b,
    tolerance,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
):
    # Note that the original world-aligned bounding box is used to computed the absolute tolerance from the
    # relative one. This way, it is a constant that does not depends on the orientation of the geometry, which
    # makes sense since the scale of the geometries is an intrinsic property and not something that is supposed
    # to change dynamically.
    aabb_size_b = (geoms_init_AABB[i_gb, 7] - geoms_init_AABB[i_gb, 0]).norm()
    aabb_size = aabb_size_b
    if geoms_info.type[i_ga] != gs.GEOM_TYPE.PLANE:
        aabb_size_a = (geoms_init_AABB[i_ga, 7] - geoms_init_AABB[i_ga, 0]).norm()
        aabb_size = ti.min(aabb_size_a, aabb_size_b)

    return 0.5 * tolerance * aabb_size


@ti.func
def func_contact_orthogonals(
    i_ga,
    i_gb,
    normal: ti.types.vector(3),
    i_b,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    EPS = rigid_global_info.EPS[None]

    axis_0 = ti.Vector.zero(gs.ti_float, 3)
    axis_1 = ti.Vector.zero(gs.ti_float, 3)

    if ti.static(static_rigid_sim_config.enable_mujoco_compatibility):
        # Choose between world axes Y or Z to avoid colinearity issue
        if ti.abs(normal[1]) < 0.5:
            axis_0[1] = 1.0
        else:
            axis_0[2] = 1.0

        # Project axis on orthogonal plane to contact normal
        axis_0 = (axis_0 - normal.dot(axis_0) * normal).normalized()

        # Perturb with some noise so that they do not align with world axes to avoid denegerated cases
        axis_1 = (normal.cross(axis_0) + 0.1 * axis_0).normalized()
        axis_0 = axis_1.cross(normal)
    else:
        # The reference geometry is the one that will have the largest impact on the position of
        # the contact point. Basically, the smallest one between the two, which can be approximated
        # by the volume of their respective bounding box.
        i_g = i_gb
        if geoms_info.type[i_ga] != gs.GEOM_TYPE.PLANE:
            size_ga = geoms_init_AABB[i_ga, 7]
            volume_ga = size_ga[0] * size_ga[1] * size_ga[2]
            size_gb = geoms_init_AABB[i_gb, 7]
            volume_gb = size_gb[0] * size_gb[1] * size_gb[2]
            i_g = i_ga if volume_ga < volume_gb else i_gb

        # Compute orthogonal basis mixing principal inertia axes of geometry with contact normal
        i_l = geoms_info.link_idx[i_g]
        rot = gu.ti_quat_to_R(links_state.i_quat[i_l, i_b], EPS)
        axis_idx = gs.ti_int(0)
        axis_angle_max = gs.ti_float(0.0)
        for i in ti.static(range(3)):
            axis_angle = ti.abs(rot[:, i].dot(normal))
            if axis_angle > axis_angle_max:
                axis_angle_max = axis_angle
                axis_idx = i
        axis_idx = (axis_idx + 1) % 3
        axis_0 = rot[:, axis_idx]
        axis_0 = (axis_0 - normal.dot(axis_0) * normal).normalized()
        axis_1 = normal.cross(axis_0)

    return axis_0, axis_1


@ti.func
def func_rotate_frame(
    pos: ti.types.vector(3, dtype=gs.ti_float),
    quat: ti.types.vector(4, dtype=gs.ti_float),
    contact_pos: ti.types.vector(3, dtype=gs.ti_float),
    qrot: ti.types.vector(4, dtype=gs.ti_float),
) -> tuple[
    ti.types.vector(3, dtype=gs.ti_float),
    ti.types.vector(4, dtype=gs.ti_float),
]:
    """
    Instead of modifying geoms_state in place, this function takes thread-local
    pos/quat and returns the updated values.
    """
    new_quat = gu.ti_transform_quat_by_quat(quat, qrot)

    rel = contact_pos - pos
    vec = gu.ti_transform_by_quat(rel, qrot)
    vec = vec - rel
    new_pos = pos - vec

    return new_pos, new_quat


@ti.kernel
def func_set_upstream_grad(
    dL_dposition: ti.types.ndarray(),
    dL_dnormal: ti.types.ndarray(),
    dL_dpenetration: ti.types.ndarray(),
    collider_state: array_class.ColliderState,
):
    _B = dL_dposition.shape[0]
    _C = dL_dposition.shape[1]
    for i_b, i_c in ti.ndrange(_B, _C):
        for j in ti.static(range(3)):
            collider_state.contact_data.pos.grad[i_c, i_b][j] = dL_dposition[i_b, i_c, j]
            collider_state.contact_data.normal.grad[i_c, i_b][j] = dL_dnormal[i_b, i_c, j]
        collider_state.contact_data.penetration.grad[i_c, i_b] = dL_dpenetration[i_b, i_c]
