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
def func_refine_smooth_contact_pos(
    geom_type,
    geom_data,
    geom_pos: qd.types.vector(3),
    geom_quat: qd.types.vector(4),
    normal: qd.types.vector(3),
    penetration,
    ccd_contact_pos: qd.types.vector(3),
):
    """
    Reconstruct the contact position analytically from the smooth side of the contact.

    MPR/GJK leave a position-dependent bias in the reported contact position that, on static contacts against
    rotationally-symmetric geometry, becomes torque on the smooth body and drives a persistent tangential drift (the
    lever arm becomes non-zero on what should be a face-aligned contact). For smooth primitives we have a closed-form
    surface point given the CCD-reported normal, so we can replace the biased contact position with the exact midpoint
    between that surface point and the inferred polytope-side surface. The result has the lever arm parallel to the
    contact normal, so the constraint force creates no spurious torque.

    Conventions: normal points from geom B to geom A (geom A is the one being refined). The refined contact position
    is the midpoint between A's surface (in the -normal direction from A's center) and the implicit B surface (offset
    by penetration along normal). Idempotent on the analytical paths (sphere-box, sphere-capsule, capsule-capsule)
    since those use the same closed-form expression.
    """
    refined = ccd_contact_pos
    if geom_type == gs.GEOM_TYPE.SPHERE:
        radius = geom_data[0]
        refined = geom_pos - (radius - 0.5 * penetration) * normal
    elif geom_type == gs.GEOM_TYPE.ELLIPSOID:
        # Surface point on ellipsoid in direction -normal, in local frame, is at p = -(a^2 n_x, b^2 n_y, c^2 n_z) /
        # sqrt(a^2 n_x^2 + b^2 n_y^2 + c^2 n_z^2). This comes from the Lagrangian "closest point in direction d" with
        # f(p) = (px/a)^2 + ... - 1 = 0.
        a = geom_data[0]
        b = geom_data[1]
        c = geom_data[2]
        n_local = gu.qd_inv_transform_by_quat(normal, geom_quat)
        denom = qd.sqrt(
            a * a * n_local[0] * n_local[0] + b * b * n_local[1] * n_local[1] + c * c * n_local[2] * n_local[2]
        )
        p_local = qd.Vector(
            [-a * a * n_local[0] / denom, -b * b * n_local[1] / denom, -c * c * n_local[2] / denom], dt=gs.qd_float
        )
        surface_pt = gu.qd_transform_by_trans_quat(p_local, geom_pos, geom_quat)
        refined = surface_pt + 0.5 * penetration * normal
    elif geom_type == gs.GEOM_TYPE.CAPSULE:
        # Capsule axis is along local +z. Project ccd_contact_pos onto the axis (clamped to the segment), then offset by
        # radius along -normal. The clamp lets cap contacts degenerate to the sphere case automatically. Barrel contacts
        # inherit the axial coordinate from ccd_contact_pos, which is only as good as the CCD's axial estimate.
        radius = geom_data[0]
        half_length = 0.5 * geom_data[1]
        axis_dir = gu.qd_transform_by_quat_fast(qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float), geom_quat)
        t_axial = (ccd_contact_pos - geom_pos).dot(axis_dir)
        t_clamped = qd.math.clamp(t_axial, -half_length, half_length)
        axis_point = geom_pos + t_clamped * axis_dir
        refined = axis_point - (radius - 0.5 * penetration) * normal
    return refined


@qd.func
def func_apply_smooth_refinement(
    i_ga,
    i_gb,
    normal: qd.types.vector(3),
    penetration,
    contact_pos: qd.types.vector(3),
    ga_pos: qd.types.vector(3),
    ga_quat: qd.types.vector(4),
    gb_pos: qd.types.vector(3),
    gb_quat: qd.types.vector(4),
    geoms_info: array_class.GeomsInfo,
    static_rigid_sim_config: qd.template(),
):
    """
    Reconstruct the contact position analytically from the smooth side when one of the geoms is a smooth primitive.

    Idempotent on analytical contact paths; on MPR/GJK paths it removes the position-dependent bias that drives
    spurious torque and drift on static smooth-vs-polytope contacts. Must be invoked right after collision detection
    and before any post-processing (deduplication, perturbation reversal, etc.), so downstream stages see contact
    positions in the same canonical frame the constraint solver will store. The pose inputs must match the pose CCD
    or the analytical formula actually saw - the perturbed pose under multi-contact, not the unperturbed state.
    """
    if qd.static(not static_rigid_sim_config.enable_mujoco_compatibility):
        # Geom pairs are sorted by ascending type, so smooth primitives (SPHERE/ELLIPSOID/CAPSULE) always sit on the
        # A side when paired with a polytope (BOX/MESH/TERRAIN/PLANE). Smooth-vs-smooth pairs go through analytical
        # fast paths and never reach this helper, so at most one side ever needs refinement.
        type_a = geoms_info.type[i_ga]
        type_b = geoms_info.type[i_gb]
        if type_a == gs.GEOM_TYPE.SPHERE or type_a == gs.GEOM_TYPE.ELLIPSOID or type_a == gs.GEOM_TYPE.CAPSULE:
            contact_pos = func_refine_smooth_contact_pos(
                type_a, geoms_info.data[i_ga], ga_pos, ga_quat, normal, penetration, contact_pos
            )
        elif type_b == gs.GEOM_TYPE.SPHERE or type_b == gs.GEOM_TYPE.ELLIPSOID or type_b == gs.GEOM_TYPE.CAPSULE:
            contact_pos = func_refine_smooth_contact_pos(
                type_b, geoms_info.data[i_gb], gb_pos, gb_quat, -normal, penetration, contact_pos
            )
    return contact_pos


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


@qd.kernel(fastcache=True)
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
    i_b,
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
@qd.kernel(fastcache=True)
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


@qd.kernel(fastcache=True)
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


@qd.kernel(fastcache=True)
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
    errno: qd.Tensor,
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
    contact data, and for the split narrowphase multi-contact writes.
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


@qd.kernel(fastcache=True)
def func_clamp_and_sort_contacts(
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    static_rigid_sim_config: qd.template(),
):
    """Sort contacts spatially by x-coordinate, moving entire geom-pair groups as units, while clamping number of
    contacts to avoid unbounded memory access.

    When contacts are added with use_atomic=True from parallel narrowphase kernels, the counter may exceed the array
    bounds even though only in-bounds entries are actually written. The clamp brings the counter back to the valid
    range before sorting.

    Contacts from the same geom pair are contiguous after narrowphase. We assign every contact in a group the
    x-position of the group's first contact. The stable insertion sort then reorders groups spatially while preserving
    the narrowphase ordering within each group.

    Two-phase approach to minimise memory traffic:
    1. Insertion sort on a compact (key, index) pair — 8 bytes per swap instead of moving all 11 contact fields
    2. In-place cycle-following permutation that moves each contact record exactly once
    """
    _B = collider_state.n_contacts.shape[0]
    max_contact_pairs = collider_info.max_contact_pairs[None]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        n_con = qd.min(collider_state.n_contacts[i_b], max_contact_pairs)
        collider_state.n_contacts[i_b] = n_con

        # Phase 1: initialise and insertion-sort the (key, idx) arrays.
        group_key = gs.qd_float(0.0)
        for i in range(n_con):
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

        for i in range(1, n_con):
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

        # Phase 2: apply permutation in-place via cycle decomposition. Each contact is read and written exactly once.
        # All 11 contact_data fields are permuted unconditionally (matching upstream): an earlier deskai6 opt-8b version
        # tried to gate force/pair_idx writes on requires_grad, but the resulting qd.static-scoped variables tripped a
        # Quadrants NameError in DEBUG-mode unit tests, and a hoist workaround caused subtle Metal-backend regressions
        # in test_smooth_box_no_drift. The 2 saved writes per non-grad cycle are not worth the platform risk.
        for i in range(n_con):
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


@qd.kernel(fastcache=True)
def func_prune_contacts(
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Prune redundant contacts per link-pair via support-polygon (2D convex hull on the contact-patch plane).

    Operates after func_clamp_and_sort_contacts. Groups contacts by canonical (min(link_a, link_b), max(link_a,
    link_b)) and, for each bucket of >= 5 contacts whose positions lie in a single plane (perpendicular to the
    bucket's folded mean normal), keeps only the 2D convex hull vertices of the projected positions. Buckets whose
    positions are not single-plane (e.g. multi-wall corner with contacts on perpendicular surfaces) are left
    untouched. The normal direction of each surviving contact is preserved verbatim; the bucket's mean normal is used
    only as the projection direction.

    The single ``tol`` parameter controls the depth gate as a dimensionless slop fraction:
      max |out-of-plane offset| / in-plane radius <= tol.

    Phases (per env, scratch sized to max_contact_pairs):
    1. Group by canonical link-pair: insertion-sort indices by (min_link, max_link) key, then apply the permutation
       in-place via cycle decomposition (11-field swap per contact record).
    2. Per bucket of >= 5 contacts: compute mean normal (folded to a common hemisphere). Check depth coplanarity of
       contact positions. If they share a plane, project to (u, v), Andrew's monotone chain. Mark survivors in
       contact_keep[]. Then restore any non-hull contact whose penetration is much deeper than the hull's average.
    3. Compact: copy kept contacts to the front, update n_contacts.
    """
    _B = collider_state.n_contacts.shape[0]
    max_contact_pairs = collider_info.max_contact_pairs[None]
    tol = collider_info.contact_pruning_tolerance[None]
    prune_deep_penetration_ratio = collider_info.prune_deep_penetration_ratio[None]
    prune_hull_collinear_tol = collider_info.prune_hull_collinear_tol[None]
    LP_KEY_STRIDE = gs.qd_float(1.0e7)
    EPS = rigid_global_info.EPS[None]

    # deskai6 opt 9d (real warp-coop, stage 1): pull contact_keep init and phase-1a key+idx init OUT of `if tid == 0:`
    # and parallelize them across the 32 lanes (stride-tid). Phase 1a sort, phase 2 bucket walk, and phase 3 (sort +
    # cycle-permute) stay serial on lane 0. Subsequent stages add coop reductions (phase-2 mean / centroid) and parallel
    # cycle-permute.
    _K = qd.static(32)
    qd.loop_config(name="prune_contacts_coop", block_dim=_K)
    for i_flat in range(_B * _K):
        tid = i_flat % _K
        i_b = i_flat // _K
        # All lanes compute n_con (cheap, no memory write on non-lane-0).
        n_con = qd.min(collider_state.n_contacts[i_b], max_contact_pairs)
        if tid == 0:
            collider_state.n_contacts[i_b] = n_con

        # PARALLEL: contact_keep init. Default-keep marked here unconditionally so the fused phase 3 (compact + spatial
        # sort) below produces a correct result for envs with n_con < 5 (no dedup buckets, but still need spatial sort).
        # 32 lanes stride.
        ii = tid
        while ii < n_con:
            collider_state.contact_keep[ii, i_b] = 1
            ii += _K

        if n_con >= 5:
            # PARALLEL: phase 1a key + idx init, 32 lanes stride.
            ii = tid
            while ii < n_con:
                la = collider_state.contact_data.link_a[ii, i_b]
                lb = collider_state.contact_data.link_b[ii, i_b]
                la_min = qd.min(la, lb)
                la_max = qd.max(la, lb)
                collider_state.contact_sort_key[ii, i_b] = qd.cast(la_min, gs.qd_float) * LP_KEY_STRIDE + qd.cast(
                    la_max, gs.qd_float
                )
                collider_state.contact_sort_idx[ii, i_b] = ii
                ii += _K

            if tid == 0:
                # SERIAL on lane 0: phase 1a insertion sort + phase 2 bucket walk.
                for i in range(1, n_con):
                    ck = collider_state.contact_sort_key[i, i_b]
                    if collider_state.contact_sort_key[i - 1, i_b] <= ck:
                        continue
                    ci = collider_state.contact_sort_idx[i, i_b]
                    j = i - 1
                    while j >= 0:
                        if collider_state.contact_sort_key[j, i_b] <= ck:
                            break
                        collider_state.contact_sort_key[j + 1, i_b] = collider_state.contact_sort_key[j, i_b]
                        collider_state.contact_sort_idx[j + 1, i_b] = collider_state.contact_sort_idx[j, i_b]
                        j = j - 1
                    collider_state.contact_sort_key[j + 1, i_b] = ck
                    collider_state.contact_sort_idx[j + 1, i_b] = ci

            qd.simt.subgroup.sync()

            # Phase 2 (deskai6 opt 9d stage 2): bucket walk runs on ALL 32 lanes. The outer control flow (find b_end,
            # iterate buckets) is duplicated across lanes since the inputs are all in DRAM (cache-friendly). Inside a
            # bucket, the mean-normal / centroid sum is done coop via 6 reduce_all_add_tiled calls. The rest of the
            # bucket processing (coplanarity check with early-exit, in-plane basis, projection, lex sort, hull build,
            # mark survivors) stays serial on lane 0.
            b_start = 0
            while b_start < n_con:
                key0 = collider_state.contact_sort_key[b_start, i_b]
                b_end = b_start + 1
                while b_end < n_con:
                    if collider_state.contact_sort_key[b_end, i_b] != key0:
                        break
                    b_end += 1
                b_size = b_end - b_start

                if b_size >= 5:
                    ref_src = collider_state.contact_sort_idx[b_start, i_b]
                    ref_n = collider_state.contact_data.normal[ref_src, i_b]
                    rnx = ref_n[0]
                    rny = ref_n[1]
                    rnz = ref_n[2]
                    mnx_l = gs.qd_float(0.0)
                    mny_l = gs.qd_float(0.0)
                    mnz_l = gs.qd_float(0.0)
                    cx_l = gs.qd_float(0.0)
                    cy_l = gs.qd_float(0.0)
                    cz_l = gs.qd_float(0.0)
                    jj = b_start + tid
                    while jj < b_end:
                        src_i = collider_state.contact_sort_idx[jj, i_b]
                        n_i = collider_state.contact_data.normal[src_i, i_b]
                        s = gs.qd_float(1.0)
                        if rnx * n_i[0] + rny * n_i[1] + rnz * n_i[2] < gs.qd_float(0.0):
                            s = gs.qd_float(-1.0)
                        mnx_l += s * n_i[0]
                        mny_l += s * n_i[1]
                        mnz_l += s * n_i[2]
                        p_i = collider_state.contact_data.pos[src_i, i_b]
                        cx_l += p_i[0]
                        cy_l += p_i[1]
                        cz_l += p_i[2]
                        jj += _K

                    mnx = qd.simt.subgroup.reduce_all_add_tiled(mnx_l, 5)
                    mny = qd.simt.subgroup.reduce_all_add_tiled(mny_l, 5)
                    mnz = qd.simt.subgroup.reduce_all_add_tiled(mnz_l, 5)
                    cx = qd.simt.subgroup.reduce_all_add_tiled(cx_l, 5)
                    cy = qd.simt.subgroup.reduce_all_add_tiled(cy_l, 5)
                    cz = qd.simt.subgroup.reduce_all_add_tiled(cz_l, 5)

                    # POST-REDUCE math runs on all 32 lanes (deterministic, cheap; redundant arithmetic is free vs.
                    # broadcasting the reduce results).
                    inv_n = gs.qd_float(1.0) / qd.cast(b_size, gs.qd_float)
                    cx *= inv_n
                    cy *= inv_n
                    cz *= inv_n
                    mnrm = qd.sqrt(mnx * mnx + mny * mny + mnz * mnz)

                    max_in_plane_r2 = gs.qd_float(0.0)
                    coplanar = mnrm > EPS
                    if coplanar:
                        mnx /= mnrm
                        mny /= mnrm
                        mnz /= mnrm

                        # COOP coplanarity check (stage 3). Each lane strides [b_start + tid, b_end) by _K, locally
                        # tracking max_depth / max_in_plane_r2. Wasted work per warp is at most b_size/_K contacts.
                        # The upstream algo no longer checks per-contact normals (a contact with a diagonal normal at
                        # the corner of a patch still participates in the 2D hull because its position is a vertex), so
                        # we only do the depth coplanarity gate here.
                        max_depth_l = gs.qd_float(0.0)
                        max_r2_l = gs.qd_float(0.0)
                        jj = b_start + tid
                        while jj < b_end:
                            src_i = collider_state.contact_sort_idx[jj, i_b]
                            p_i = collider_state.contact_data.pos[src_i, i_b]
                            dx = p_i[0] - cx
                            dy = p_i[1] - cy
                            dz = p_i[2] - cz
                            depth = qd.abs(dx * mnx + dy * mny + dz * mnz)
                            if depth > max_depth_l:
                                max_depth_l = depth
                            r2 = dx * dx + dy * dy + dz * dz - depth * depth
                            if r2 > max_r2_l:
                                max_r2_l = r2
                            jj += _K

                        max_depth = qd.simt.subgroup.reduce_all_max_tiled(max_depth_l, 5)
                        max_in_plane_r2 = qd.simt.subgroup.reduce_all_max_tiled(max_r2_l, 5)

                        if max_depth > tol * qd.sqrt(max_in_plane_r2):
                            coplanar = False

                    if coplanar:
                        # Basis on all lanes (deterministic from mnx/mny/mnz which the reduce broadcast to every lane).
                        abs_mnx = qd.abs(mnx)
                        abs_mny = qd.abs(mny)
                        abs_mnz = qd.abs(mnz)
                        ax = gs.qd_float(1.0)
                        ay = gs.qd_float(0.0)
                        az = gs.qd_float(0.0)
                        if abs_mny < abs_mnx and abs_mny < abs_mnz:
                            ax = gs.qd_float(0.0)
                            ay = gs.qd_float(1.0)
                            az = gs.qd_float(0.0)
                        elif abs_mnz < abs_mnx and abs_mnz <= abs_mny:
                            ax = gs.qd_float(0.0)
                            ay = gs.qd_float(0.0)
                            az = gs.qd_float(1.0)
                        adn = ax * mnx + ay * mny + az * mnz
                        ux = ax - adn * mnx
                        uy = ay - adn * mny
                        uz = az - adn * mnz
                        unrm = qd.sqrt(ux * ux + uy * uy + uz * uz)
                        ux /= unrm
                        uy /= unrm
                        uz /= unrm
                        vx = mny * uz - mnz * uy
                        vy = mnz * ux - mnx * uz
                        vz = mnx * uy - mny * ux

                        # COOP projection: 32 lanes stride writes to contact_sort_key + contact_proj_v.
                        jj = b_start + tid
                        while jj < b_end:
                            src_i = collider_state.contact_sort_idx[jj, i_b]
                            p_i = collider_state.contact_data.pos[src_i, i_b]
                            collider_state.contact_sort_key[jj, i_b] = p_i[0] * ux + p_i[1] * uy + p_i[2] * uz
                            collider_state.contact_proj_v[jj, i_b] = p_i[0] * vx + p_i[1] * vy + p_i[2] * vz
                            jj += _K

                        # COOP mark-drop: stride writes to contact_keep[orig].
                        jj = b_start + tid
                        while jj < b_end:
                            orig = collider_state.contact_sort_idx[jj, i_b]
                            collider_state.contact_keep[orig, i_b] = 0
                            jj += _K

                        # COOP lex_idx init: stride writes.
                        jj = b_start + tid
                        while jj < b_end:
                            collider_state.contact_lex_idx[jj, i_b] = jj
                            jj += _K

                        # SYNC between coop writes (sort_key, proj_v, lex_idx, contact_keep[orig]) and the lane-0 lex
                        # sort + hull build that reads them.
                        qd.simt.subgroup.sync()

                    if tid == 0 and coplanar:
                        # SERIAL on lane 0: lex sort + Andrew monotone-chain hull build + mark survivors. These walk a
                        # stack and have data-dependent inner loops that don't decompose across warp lanes.
                        #
                        # The sort_u_tol on the u comparison is critical for correctness, not just a perf tweak:
                        # contacts whose u values differ only by sub-millimeter MPR noise need to sort by v, otherwise
                        # mid-edge points get sorted between the two corners they sit between and survive the lower-
                        # hull pass as spurious hull vertices (cf. upstream test_contact_pruning regression when this
                        # tolerance is missing).
                        sort_u_tol = gs.qd_float(1e-3) * qd.sqrt(max_in_plane_r2)
                        for i in range(b_start + 1, b_end):
                            ci = collider_state.contact_lex_idx[i, i_b]
                            cu = collider_state.contact_sort_key[ci, i_b]
                            cv = collider_state.contact_proj_v[ci, i_b]
                            j = i - 1
                            while j >= b_start:
                                pj = collider_state.contact_lex_idx[j, i_b]
                                pu = collider_state.contact_sort_key[pj, i_b]
                                pv = collider_state.contact_proj_v[pj, i_b]
                                if (pu < cu - sort_u_tol) or (qd.abs(pu - cu) <= sort_u_tol and pv <= cv):
                                    break
                                collider_state.contact_lex_idx[j + 1, i_b] = pj
                                j -= 1
                            collider_state.contact_lex_idx[j + 1, i_b] = ci

                        # Upstream PR #2831 split the hull collinearity tolerance out of `tol`: tol is the depth
                        # coplanarity gate, prune_hull_collinear_tol is the cross-product slop for the monotone-chain
                        # popper. They were the same constant before; keeping them separate lets users tune one without
                        # changing the other.
                        hull_collinear_tol = prune_hull_collinear_tol * max_in_plane_r2

                        k = 0
                        for i in range(b_start, b_end):
                            ci = collider_state.contact_lex_idx[i, i_b]
                            cu = collider_state.contact_sort_key[ci, i_b]
                            cv = collider_state.contact_proj_v[ci, i_b]
                            while k >= 2:
                                idx_a = collider_state.contact_hull_stack[b_start + k - 2, i_b]
                                idx_b = collider_state.contact_hull_stack[b_start + k - 1, i_b]
                                au = collider_state.contact_sort_key[idx_a, i_b]
                                av = collider_state.contact_proj_v[idx_a, i_b]
                                bu = collider_state.contact_sort_key[idx_b, i_b]
                                bv = collider_state.contact_proj_v[idx_b, i_b]
                                cross = (bu - au) * (cv - av) - (bv - av) * (cu - au)
                                if cross <= hull_collinear_tol:
                                    k -= 1
                                else:
                                    break
                            collider_state.contact_hull_stack[b_start + k, i_b] = ci
                            k += 1

                        upper_start = k
                        # Memory-fence workaround for a Quadrants codegen issue on Metal _B >= 2 (genesis PR #2831):
                        # without an explicit scratch store between the lower- and upper-hull lane-0 ``for`` passes,
                        # the upper-hull pop-loop's reads of contact_hull_stack don't observe the lower-hull writes,
                        # so the cross-product check effectively runs on stale data and every candidate is kept (hull
                        # size == bucket size, no pruning). Writing any value to a non-overlapping slot here forces
                        # write-then-read ordering on the shared buffer. The value is unused. See
                        # perso_hugh/prot/qd_metal_hull_chain_visibility_repro.py for a standalone reproduction.
                        collider_state.contact_hull_stack[max_contact_pairs - 1, i_b] = 0
                        for k_step in range(b_size - 1):
                            ii_lex = b_end - 2 - k_step
                            ci = collider_state.contact_lex_idx[ii_lex, i_b]
                            cu = collider_state.contact_sort_key[ci, i_b]
                            cv = collider_state.contact_proj_v[ci, i_b]
                            while k >= upper_start + 1:
                                idx_a = collider_state.contact_hull_stack[b_start + k - 2, i_b]
                                idx_b = collider_state.contact_hull_stack[b_start + k - 1, i_b]
                                au = collider_state.contact_sort_key[idx_a, i_b]
                                av = collider_state.contact_proj_v[idx_a, i_b]
                                bu = collider_state.contact_sort_key[idx_b, i_b]
                                bv = collider_state.contact_proj_v[idx_b, i_b]
                                cross = (bu - au) * (cv - av) - (bv - av) * (cu - au)
                                if cross <= hull_collinear_tol:
                                    k -= 1
                                else:
                                    break
                            if ci != collider_state.contact_hull_stack[b_start, i_b] and k < b_size:
                                collider_state.contact_hull_stack[b_start + k, i_b] = ci
                                k += 1

                        for hk in range(k):
                            survivor_sort = collider_state.contact_hull_stack[b_start + hk, i_b]
                            survivor_orig = collider_state.contact_sort_idx[survivor_sort, i_b]
                            collider_state.contact_keep[survivor_orig, i_b] = 1

                        # Restore non-hull contacts whose penetration is much deeper than the hull boundary's max
                        # (PR #2831 switched from avg to max). Rationale: a contact whose penetration substantially
                        # exceeds the hull's deepest vertex represents a distinct physical support (deep body of a
                        # fork beyond its tines, deep middle of a long body) that the support-polygon argument
                        # doesn't actually authorize dropping (the argument only holds when all contacts share
                        # normal AND penetration). The 3x factor over the hull max is well above the typical ~1.x
                        # penetration spread on transient/rocking faces but well below the deep interior penetrations
                        # seen when a non-flat body rests inside its convex envelope. Indices here live in orig-space
                        # (because the cycle-permute is fused into the phase 3 below in opt 10 -- contact_data is
                        # still in pre-sort order, so we translate sort-space hull/bucket indices through
                        # contact_sort_idx).
                        hull_pen_max = gs.qd_float(0.0)
                        for hk in range(k):
                            survivor_sort = collider_state.contact_hull_stack[b_start + hk, i_b]
                            survivor_orig = collider_state.contact_sort_idx[survivor_sort, i_b]
                            p = collider_state.contact_data.penetration[survivor_orig, i_b]
                            if p > hull_pen_max:
                                hull_pen_max = p
                        deep_keep_threshold = prune_deep_penetration_ratio * hull_pen_max
                        for jj_idx in range(b_start, b_end):
                            orig = collider_state.contact_sort_idx[jj_idx, i_b]
                            if collider_state.contact_keep[orig, i_b] == 0:
                                if collider_state.contact_data.penetration[orig, i_b] > deep_keep_threshold:
                                    collider_state.contact_keep[orig, i_b] = 1

                b_start = b_end

        if tid == 0:
            # Phase 3 (deskai6 opt 10): FUSED compact + spatial sort. Replaces:
            #   - dedup phase 3 (compact based on contact_keep)
            #   - func_clamp_and_sort_contacts (assign group-x sort key + insertion sort + cycle-permute)
            # with a single permutation pass: compute a sort key per slot that pushes dropped contacts past the end
            # (sentinel +inf) and orders kept contacts by their geom-pair group's x-pos (anchored to the first kept
            # contact in the group, preserving narrowphase intra-group order via stable sort). Then sort sort-keys +
            # sort-indices, then cycle-permute the 9 contact_data fields exactly once.
            #
            # On dex_hand this saves ~25 displaced contacts worth of field copies that used to be done in dedup phase 3
            # before clamp_and_sort re-permuted everything. Same 9 fields per contact (force & pair_idx still skipped
            # per opt 8a; clamp_and_sort still runs in the grad path so its own opt 8b gate stays in place).
            SENTINEL_BIG = gs.qd_float(1e30)
            group_key = gs.qd_float(0.0)
            prev_ga = -1
            prev_gb = -1
            for i in range(n_con):
                if collider_state.contact_keep[i, i_b] != 0:
                    ga = collider_state.contact_data.geom_a[i, i_b]
                    gb = collider_state.contact_data.geom_b[i, i_b]
                    if ga != prev_ga or gb != prev_gb:
                        group_key = collider_state.contact_data.pos[i, i_b][0]
                        prev_ga = ga
                        prev_gb = gb
                    collider_state.contact_sort_key[i, i_b] = group_key
                else:
                    collider_state.contact_sort_key[i, i_b] = SENTINEL_BIG
                collider_state.contact_sort_idx[i, i_b] = i

            for i in range(1, n_con):
                ck = collider_state.contact_sort_key[i, i_b]
                if collider_state.contact_sort_key[i - 1, i_b] <= ck:
                    continue
                ci = collider_state.contact_sort_idx[i, i_b]
                j = i - 1
                while j >= 0:
                    if collider_state.contact_sort_key[j, i_b] <= ck:
                        break
                    collider_state.contact_sort_key[j + 1, i_b] = collider_state.contact_sort_key[j, i_b]
                    collider_state.contact_sort_idx[j + 1, i_b] = collider_state.contact_sort_idx[j, i_b]
                    j = j - 1
                collider_state.contact_sort_key[j + 1, i_b] = ck
                collider_state.contact_sort_idx[j + 1, i_b] = ci

            # n_contacts is the count of non-sentinel sort keys (the dropped contacts sit at the tail and are ignored
            # by downstream consumers via the new n_contacts).
            n_kept = 0
            for i in range(n_con):
                if collider_state.contact_sort_key[i, i_b] < SENTINEL_BIG:
                    n_kept += 1
                else:
                    break
            collider_state.n_contacts[i_b] = n_kept

            # Cycle-permute contact_data fields into their sorted positions. Only n_kept positions need correct values;
            # the tail (dropped) slots are scratched but not read again this step.
            for i in range(n_kept):
                if collider_state.contact_sort_idx[i, i_b] != i:
                    tmp_geom_a = collider_state.contact_data.geom_a[i, i_b]
                    tmp_geom_b = collider_state.contact_data.geom_b[i, i_b]
                    tmp_penetration = collider_state.contact_data.penetration[i, i_b]
                    tmp_normal = collider_state.contact_data.normal[i, i_b]
                    tmp_pos = collider_state.contact_data.pos[i, i_b]
                    tmp_friction = collider_state.contact_data.friction[i, i_b]
                    tmp_sol_params = collider_state.contact_data.sol_params[i, i_b]
                    tmp_link_a = collider_state.contact_data.link_a[i, i_b]
                    tmp_link_b = collider_state.contact_data.link_b[i, i_b]

                    j = i
                    while collider_state.contact_sort_idx[j, i_b] != i:
                        src = collider_state.contact_sort_idx[j, i_b]
                        collider_state.contact_data.geom_a[j, i_b] = collider_state.contact_data.geom_a[src, i_b]
                        collider_state.contact_data.geom_b[j, i_b] = collider_state.contact_data.geom_b[src, i_b]
                        collider_state.contact_data.penetration[j, i_b] = collider_state.contact_data.penetration[
                            src, i_b
                        ]
                        collider_state.contact_data.normal[j, i_b] = collider_state.contact_data.normal[src, i_b]
                        collider_state.contact_data.pos[j, i_b] = collider_state.contact_data.pos[src, i_b]
                        collider_state.contact_data.friction[j, i_b] = collider_state.contact_data.friction[src, i_b]
                        collider_state.contact_data.sol_params[j, i_b] = collider_state.contact_data.sol_params[
                            src, i_b
                        ]
                        collider_state.contact_data.link_a[j, i_b] = collider_state.contact_data.link_a[src, i_b]
                        collider_state.contact_data.link_b[j, i_b] = collider_state.contact_data.link_b[src, i_b]
                        collider_state.contact_sort_idx[j, i_b] = j
                        j = src

                    collider_state.contact_data.geom_a[j, i_b] = tmp_geom_a
                    collider_state.contact_data.geom_b[j, i_b] = tmp_geom_b
                    collider_state.contact_data.penetration[j, i_b] = tmp_penetration
                    collider_state.contact_data.normal[j, i_b] = tmp_normal
                    collider_state.contact_data.pos[j, i_b] = tmp_pos
                    collider_state.contact_data.friction[j, i_b] = tmp_friction
                    collider_state.contact_data.sol_params[j, i_b] = tmp_sol_params
                    collider_state.contact_data.link_a[j, i_b] = tmp_link_a
                    collider_state.contact_data.link_b[j, i_b] = tmp_link_b
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
