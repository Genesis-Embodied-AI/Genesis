"""
Contact management and utility functions for the rigid body collider.

This module contains functions for adding contacts, computing tolerances,
and managing contact data including reset/clear operations.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu


@qd.func
def rotaxis(vecin, i0, i1, i2, f0, f1, f2):
    vecres = qd.Vector([0.0, 0.0, 0.0], dt=gs.qd_float)
    vecres[0] = vecin[i0] * f0
    vecres[1] = vecin[i1] * f1
    vecres[2] = vecin[i2] * f2
    return vecres


@qd.func
def rotmatx(matin, i0, i1, i2, f0, f1, f2):
    matres = qd.Matrix.zero(gs.qd_float, 3, 3)
    matres[0, :] = matin[i0, :] * f0
    matres[1, :] = matin[i1, :] * f1
    matres[2, :] = matin[i2, :] * f2
    return matres


@qd.kernel(fastcache=gs.use_fastcache)
def collider_kernel_reset(
    envs_idx: qd.types.ndarray(),
    static_rigid_sim_config: qd.template(),
    collider_state: array_class.ColliderState,
    cache_only: qd.template(),
):
    max_possible_pairs = collider_state.contact_cache.normal.shape[0]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]

        if qd.static(not cache_only):
            collider_state.first_time[i_b] = True

        for i_pair in range(max_possible_pairs):
            collider_state.contact_cache.normal[i_pair, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.func
def func_collider_clear_env(
    i_b: gs.qd_int,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    static_rigid_sim_config: qd.template(),
    collider_state: array_class.ColliderState,
):
    if qd.static(static_rigid_sim_config.use_hibernation):
        collider_state.n_contacts_hibernated[i_b] = 0

        for i_c in range(collider_state.n_contacts[i_b]):
            i_la = collider_state.contact_data.link_a[i_c, i_b]
            i_lb = collider_state.contact_data.link_b[i_c, i_b]

            I_la = [i_la, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_la
            I_lb = [i_lb, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_lb

            if (links_state.hibernated[i_la, i_b] and links_info.is_fixed[I_lb]) or (
                links_state.hibernated[i_lb, i_b] and links_info.is_fixed[I_la]
            ):
                i_c_hibernated = collider_state.n_contacts_hibernated[i_b]
                if i_c != i_c_hibernated:
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

    for i_c in range(collider_state.n_contacts[i_b]):
        should_clear = True
        if qd.static(static_rigid_sim_config.use_hibernation):
            should_clear = i_c >= collider_state.n_contacts_hibernated[i_b]
        if should_clear:
            collider_state.contact_data.link_a[i_c, i_b] = -1
            collider_state.contact_data.link_b[i_c, i_b] = -1
            collider_state.contact_data.geom_a[i_c, i_b] = -1
            collider_state.contact_data.geom_b[i_c, i_b] = -1
            collider_state.contact_data.penetration[i_c, i_b] = 0.0
            collider_state.contact_data.pos[i_c, i_b] = qd.Vector.zero(gs.qd_float, 3)
            collider_state.contact_data.normal[i_c, i_b] = qd.Vector.zero(gs.qd_float, 3)
            collider_state.contact_data.force[i_c, i_b] = qd.Vector.zero(gs.qd_float, 3)

    if qd.static(static_rigid_sim_config.use_hibernation):
        collider_state.n_contacts[i_b] = collider_state.n_contacts_hibernated[i_b]
    else:
        collider_state.n_contacts[i_b] = 0


# only used with hibernation ??
@qd.kernel(fastcache=gs.use_fastcache)
def kernel_collider_clear(
    envs_idx: qd.types.ndarray(),
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    static_rigid_sim_config: qd.template(),
    collider_state: array_class.ColliderState,
):
    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        func_collider_clear_env(i_b, links_state, links_info, static_rigid_sim_config, collider_state)


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_masked_collider_clear(
    envs_mask: qd.types.ndarray(),
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    static_rigid_sim_config: qd.template(),
    collider_state: array_class.ColliderState,
):
    for i_b in range(envs_mask.shape[0]):
        if envs_mask[i_b]:
            func_collider_clear_env(i_b, links_state, links_info, static_rigid_sim_config, collider_state)


@qd.kernel(fastcache=gs.use_fastcache)
def collider_kernel_get_contacts(
    is_padded: qd.template(),
    iout: qd.types.ndarray(),
    fout: qd.types.ndarray(),
    static_rigid_sim_config: qd.template(),
    collider_state: array_class.ColliderState,
):
    _B = collider_state.active_buffer.shape[1]

    # TODO: Better implementation from Quadrants for this kind of reduction.
    n_contacts_max = gs.qd_int(0)
    qd.loop_config(serialize=True)
    for i_b in range(_B):
        n_contacts = collider_state.n_contacts[i_b]
        if n_contacts > n_contacts_max:
            n_contacts_max = n_contacts

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        i_c_start = gs.qd_int(0)
        if qd.static(is_padded):
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
            for j in qd.static(range(3)):
                fout[i_c, 1 + j] = collider_state.contact_data.pos[i_c_, i_b][j]
                fout[i_c, 4 + j] = collider_state.contact_data.normal[i_c_, i_b][j]
                fout[i_c, 7 + j] = collider_state.contact_data.force[i_c_, i_b][j]


@qd.func
def func_add_contact(
    i_ga,
    i_gb,
    normal: qd.types.vector(3),
    contact_pos: qd.types.vector(3),
    penetration,
    i_b,
    i_pair,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    errno: array_class.V_ANNOTATION,
    use_atomic: qd.template() = False,
):
    i_c = 0
    if qd.static(use_atomic):
        i_c = qd.atomic_add(collider_state.n_contacts[i_b], 1)
    else:
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
        collider_state.contact_data.friction[i_c, i_b] = qd.max(qd.max(friction_a, friction_b), 1e-2)
        collider_state.contact_data.sol_params[i_c, i_b] = 0.5 * (
            geoms_info.sol_params[i_ga] + geoms_info.sol_params[i_gb]
        )
        collider_state.contact_data.link_a[i_c, i_b] = geoms_info.link_idx[i_ga]
        collider_state.contact_data.link_b[i_c, i_b] = geoms_info.link_idx[i_gb]
        collider_state.contact_data.pair_idx[i_c, i_b] = i_pair

        if not qd.static(use_atomic):
            collider_state.n_contacts[i_b] = i_c + 1
    else:
        errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_COLLISION_PAIRS


@qd.func
def func_set_contact(
    i_ga,
    i_gb,
    normal: qd.types.vector(3),
    contact_pos: qd.types.vector(3),
    penetration,
    i_b,
    i_c,
    i_pair,
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
    collider_state.contact_data.friction[i_c, i_b] = qd.max(qd.max(friction_a, friction_b), 1e-2)
    collider_state.contact_data.sol_params[i_c, i_b] = 0.5 * (geoms_info.sol_params[i_ga] + geoms_info.sol_params[i_gb])
    collider_state.contact_data.link_a[i_c, i_b] = geoms_info.link_idx[i_ga]
    collider_state.contact_data.link_b[i_c, i_b] = geoms_info.link_idx[i_gb]
    collider_state.contact_data.pair_idx[i_c, i_b] = i_pair


@qd.func
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


@qd.func
def func_compute_geom_rbound(
    i_g,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
):
    """Compute the bounding sphere radius for a geom, matching MuJoCo's geom_rbound."""
    geom_type = geoms_info.type[i_g]
    rbound = gs.qd_float(0.0)
    if geom_type == gs.GEOM_TYPE.SPHERE:
        rbound = geoms_info.data[i_g][0]
    elif geom_type == gs.GEOM_TYPE.CAPSULE:
        # radius + half_length (MuJoCo stores size as [radius, half_length])
        # Genesis stores data as [radius, full_length], so half_length = 0.5 * data[1]
        rbound = geoms_info.data[i_g][0] + 0.5 * geoms_info.data[i_g][1]
    elif geom_type == gs.GEOM_TYPE.ELLIPSOID:
        rbound = qd.max(geoms_info.data[i_g][0], qd.max(geoms_info.data[i_g][1], geoms_info.data[i_g][2]))
    elif geom_type == gs.GEOM_TYPE.BOX:
        d0 = geoms_info.data[i_g][0]
        d1 = geoms_info.data[i_g][1]
        d2 = geoms_info.data[i_g][2]
        rbound = qd.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
    else:
        # For mesh and other types, approximate as half AABB diagonal
        rbound = 0.5 * (geoms_init_AABB[i_g, 7] - geoms_init_AABB[i_g, 0]).norm()
    return rbound


@qd.func
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
        aabb_size = qd.min(aabb_size_a, aabb_size_b)

    return 0.5 * tolerance * aabb_size


@qd.func
def func_compute_mj_tolerance(
    i_ga,
    i_gb,
    tolerance,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
):
    """Compute tolerance matching MuJoCo's formula: relative_tolerance * min(rbound_g1, rbound_g2)."""
    rbound_a = func_compute_geom_rbound(i_ga, geoms_info, geoms_init_AABB)
    rbound_b = func_compute_geom_rbound(i_gb, geoms_info, geoms_init_AABB)
    return tolerance * qd.min(rbound_a, rbound_b)


@qd.func
def func_contact_orthogonals(
    i_ga,
    i_gb,
    normal: qd.types.vector(3),
    i_b,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    EPS = rigid_global_info.EPS[None]

    axis_0 = qd.Vector.zero(gs.qd_float, 3)
    axis_1 = qd.Vector.zero(gs.qd_float, 3)

    if qd.static(static_rigid_sim_config.enable_mujoco_compatibility):
        # Choose between world axes Y or Z to avoid colinearity issue
        if qd.abs(normal[1]) < 0.5:
            axis_0[1] = 1.0
        else:
            axis_0[2] = 1.0

        # Project axis on orthogonal plane to contact normal
        axis_0 = (axis_0 - normal.dot(axis_0) * normal).normalized()

        # Complete orthonormal frame (matching MuJoCo's mju_makeFrame)
        axis_1 = normal.cross(axis_0)
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
        rot = gu.qd_quat_to_R(links_state.i_quat[i_l, i_b], EPS)
        axis_idx = gs.qd_int(0)
        axis_angle_max = gs.qd_float(0.0)
        for i in qd.static(range(3)):
            axis_angle = qd.abs(rot[:, i].dot(normal))
            if axis_angle > axis_angle_max:
                axis_angle_max = axis_angle
                axis_idx = i
        axis_idx = (axis_idx + 1) % 3
        axis_0 = rot[:, axis_idx]
        axis_0 = (axis_0 - normal.dot(axis_0) * normal).normalized()
        axis_1 = normal.cross(axis_0)

    return axis_0, axis_1


@qd.func
def func_rotate_frame(
    pos: qd.types.vector(3, dtype=gs.qd_float),
    quat: qd.types.vector(4, dtype=gs.qd_float),
    contact_pos: qd.types.vector(3, dtype=gs.qd_float),
    qrot: qd.types.vector(4, dtype=gs.qd_float),
) -> tuple[
    qd.types.vector(3, dtype=gs.qd_float),
    qd.types.vector(4, dtype=gs.qd_float),
]:
    """
    Instead of modifying geoms_state in place, this function takes thread-local
    pos/quat and returns the updated values.
    """
    new_quat = gu.qd_transform_quat_by_quat(quat, qrot)

    rel = contact_pos - pos
    vec = gu.qd_transform_by_quat(rel, qrot)
    vec = vec - rel
    new_pos = pos - vec

    return new_pos, new_quat


@qd.kernel(fastcache=gs.use_fastcache)
def func_sort_contacts(
    collider_state: array_class.ColliderState,
    static_rigid_sim_config: qd.template(),
):
    """Sort contacts within each env spatially by x-coordinate, moving
    entire geom-pair groups as units.

    Contacts from the same geom pair are contiguous after narrowphase.
    We assign every contact in a group the x-position of the group's first
    contact.  The stable insertion sort then reorders groups spatially while
    preserving the narrowphase ordering within each group.

    Two-phase approach to minimise memory traffic:
    1. Insertion sort on a compact (key, index) pair — 8 bytes per swap
       instead of moving all 11 contact fields (~92 bytes).
    2. In-place cycle-following permutation that moves each contact record
       exactly once.
    """
    _B = collider_state.n_contacts.shape[0]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        n = collider_state.n_contacts[i_b]

        # Phase 1: initialise and insertion-sort the (key, idx) arrays.
        group_key = gs.qd_float(0.0)
        for i in range(n):
            ga = collider_state.contact_data.geom_a[i, i_b]
            gb = collider_state.contact_data.geom_b[i, i_b]
            if (
                i == 0
                or ga != collider_state.contact_data.geom_a[i - 1, i_b]
                or gb != collider_state.contact_data.geom_b[i - 1, i_b]
            ):
                group_key = collider_state.contact_data.pos[i, i_b][0]
            collider_state.contact_sort_key[i, i_b] = group_key
            collider_state.contact_sort_idx[i, i_b] = i

        for i in range(1, n):
            curr_key = collider_state.contact_sort_key[i, i_b]
            if collider_state.contact_sort_key[i - 1, i_b] <= curr_key:
                continue

            curr_idx = collider_state.contact_sort_idx[i, i_b]
            j = i - 1
            while j >= 0:
                if collider_state.contact_sort_key[j, i_b] <= curr_key:
                    break
                collider_state.contact_sort_key[j + 1, i_b] = collider_state.contact_sort_key[j, i_b]
                collider_state.contact_sort_idx[j + 1, i_b] = collider_state.contact_sort_idx[j, i_b]
                j = j - 1
            collider_state.contact_sort_key[j + 1, i_b] = curr_key
            collider_state.contact_sort_idx[j + 1, i_b] = curr_idx

        # Phase 2: apply permutation in-place via cycle decomposition.
        # Each contact is read and written exactly once.
        for i in range(n):
            if collider_state.contact_sort_idx[i, i_b] != i:
                tmp_geom_a = collider_state.contact_data.geom_a[i, i_b]
                tmp_geom_b = collider_state.contact_data.geom_b[i, i_b]
                tmp_penetration = collider_state.contact_data.penetration[i, i_b]
                tmp_normal = collider_state.contact_data.normal[i, i_b]
                tmp_pos = collider_state.contact_data.pos[i, i_b]
                tmp_friction = collider_state.contact_data.friction[i, i_b]
                tmp_sol_params = collider_state.contact_data.sol_params[i, i_b]
                tmp_force = collider_state.contact_data.force[i, i_b]
                tmp_link_a = collider_state.contact_data.link_a[i, i_b]
                tmp_link_b = collider_state.contact_data.link_b[i, i_b]
                tmp_pair_idx = collider_state.contact_data.pair_idx[i, i_b]

                j = i
                while collider_state.contact_sort_idx[j, i_b] != i:
                    src = collider_state.contact_sort_idx[j, i_b]
                    collider_state.contact_data.geom_a[j, i_b] = collider_state.contact_data.geom_a[src, i_b]
                    collider_state.contact_data.geom_b[j, i_b] = collider_state.contact_data.geom_b[src, i_b]
                    collider_state.contact_data.penetration[j, i_b] = collider_state.contact_data.penetration[src, i_b]
                    collider_state.contact_data.normal[j, i_b] = collider_state.contact_data.normal[src, i_b]
                    collider_state.contact_data.pos[j, i_b] = collider_state.contact_data.pos[src, i_b]
                    collider_state.contact_data.friction[j, i_b] = collider_state.contact_data.friction[src, i_b]
                    collider_state.contact_data.sol_params[j, i_b] = collider_state.contact_data.sol_params[src, i_b]
                    collider_state.contact_data.force[j, i_b] = collider_state.contact_data.force[src, i_b]
                    collider_state.contact_data.link_a[j, i_b] = collider_state.contact_data.link_a[src, i_b]
                    collider_state.contact_data.link_b[j, i_b] = collider_state.contact_data.link_b[src, i_b]
                    collider_state.contact_data.pair_idx[j, i_b] = collider_state.contact_data.pair_idx[src, i_b]
                    collider_state.contact_sort_idx[j, i_b] = j
                    j = src

                collider_state.contact_data.geom_a[j, i_b] = tmp_geom_a
                collider_state.contact_data.geom_b[j, i_b] = tmp_geom_b
                collider_state.contact_data.penetration[j, i_b] = tmp_penetration
                collider_state.contact_data.normal[j, i_b] = tmp_normal
                collider_state.contact_data.pos[j, i_b] = tmp_pos
                collider_state.contact_data.friction[j, i_b] = tmp_friction
                collider_state.contact_data.sol_params[j, i_b] = tmp_sol_params
                collider_state.contact_data.force[j, i_b] = tmp_force
                collider_state.contact_data.link_a[j, i_b] = tmp_link_a
                collider_state.contact_data.link_b[j, i_b] = tmp_link_b
                collider_state.contact_data.pair_idx[j, i_b] = tmp_pair_idx
                collider_state.contact_sort_idx[j, i_b] = j


@qd.kernel
def func_set_upstream_grad(
    dL_dposition: qd.types.ndarray(),
    dL_dnormal: qd.types.ndarray(),
    dL_dpenetration: qd.types.ndarray(),
    collider_state: array_class.ColliderState,
):
    _B = dL_dposition.shape[0]
    _C = dL_dposition.shape[1]
    for i_b, i_c in qd.ndrange(_B, _C):
        for j in qd.static(range(3)):
            collider_state.contact_data.pos.grad[i_c, i_b][j] = dL_dposition[i_b, i_c, j]
            collider_state.contact_data.normal.grad[i_c, i_b][j] = dL_dnormal[i_b, i_c, j]
        collider_state.contact_data.penetration.grad[i_c, i_b] = dL_dpenetration[i_b, i_c]
