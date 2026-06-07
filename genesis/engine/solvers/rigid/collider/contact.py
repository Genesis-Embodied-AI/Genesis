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
    spurious torque and drift on static smooth-vs-polytope contacts. The pose inputs (ga_*/gb_*) must be in the same
    frame as contact_pos and normal: the detection pose for a directly-added contact, or the unperturbed pose for a
    multi-contact perturbed contact, which is refined only after the perturbation is reverted so the result lands in
    the canonical frame the constraint solver stores.
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
            i_col = collider_state.contact_sort_idx[i_c_, i_b]

            iout[i_c, 0] = collider_state.contact_data.link_a[i_col, i_b]
            iout[i_c, 1] = collider_state.contact_data.link_b[i_col, i_b]
            iout[i_c, 2] = collider_state.contact_data.geom_a[i_col, i_b]
            iout[i_c, 3] = collider_state.contact_data.geom_b[i_col, i_b]
            fout[i_c, 0] = collider_state.contact_data.penetration[i_col, i_b]
            for j in qd.static(range(3)):
                fout[i_c, 1 + j] = collider_state.contact_data.pos[i_col, i_b][j]
                fout[i_c, 4 + j] = collider_state.contact_data.normal[i_col, i_b][j]
                fout[i_c, 7 + j] = collider_state.contact_data.force[i_col, i_b][j]


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
def func_compute_geom_pair_scale(
    i_ga,
    i_gb,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
):
    # Intrinsic length scale of a geom pair: half the smaller geom's world-aligned bounding-box diagonal. The
    # original (rest-pose) AABB is used so the scale is a constant independent of the current orientation, which
    # makes sense since the size of the geometries is an intrinsic property. Multiply by a relative tolerance to
    # turn it into an absolute one.
    aabb_size_b = (geoms_init_AABB[i_gb, 7] - geoms_init_AABB[i_gb, 0]).norm()
    aabb_size = aabb_size_b
    if geoms_info.type[i_ga] != gs.GEOM_TYPE.PLANE:
        aabb_size_a = (geoms_init_AABB[i_ga, 7] - geoms_init_AABB[i_ga, 0]).norm()
        aabb_size = qd.min(aabb_size_a, aabb_size_b)

    return 0.5 * aabb_size


@qd.func
def func_compute_geom_pair_scale_mj(
    i_ga,
    i_gb,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
):
    """Geom-pair length scale matching MuJoCo's formula: min(rbound_g1, rbound_g2). Multiply by a relative tolerance
    to recover MuJoCo's absolute tolerance."""
    rbound_a = func_compute_geom_rbound(i_ga, geoms_info, geoms_init_AABB)
    rbound_b = func_compute_geom_rbound(i_gb, geoms_info, geoms_init_AABB)
    return qd.min(rbound_a, rbound_b)


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
    pos: qd.types.vector(3),
    quat: qd.types.vector(4),
    contact_pos: qd.types.vector(3),
    qrot: qd.types.vector(4),
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
def func_clamp_prune_and_sort_contacts(
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    collider_static_config: qd.template(),
):
    """Clamp + (optional) link-pair pruning + (optional) x-position sort, in one per-env loop pass.

    Builds a logical-to-physical contact permutation in ``contact_sort_idx`` rather than rewriting ``contact_data``.
    After this kernel runs, downstream consumers read contact i_col by indirecting through
    ``contact_data.X[contact_sort_idx[i_col, i_b], i_b]``. The physical layout of ``contact_data`` is left intact.

    Phases per env (gated at compile time by ``collider_static_config``):
    - Always: clamp ``n_contacts`` to ``max_contact_pairs``; initialise ``contact_sort_idx`` to the identity.
    - If ``has_prunable_contacts and not requires_grad``: prune redundant contacts via 2D convex hull on the
      contact-patch plane (skipped at runtime when ``contact_pruning_tolerance`` is 0). Drops are realised by
      compacting ``contact_sort_idx`` rather than ``contact_data``.
    - If ``has_non_box_plane_convex_convex and backend != cpu``: spatial sort the index permutation by x-position
      with geom-pair groups treated as units (provides spatial locality for downstream constraint-solver reads), with
      a (geom_a, geom_b) tie-break so groups sharing the same x sort deterministically regardless of physical layout.

    The pruning logic groups contacts by canonical (min(link_a, link_b), max(link_a, link_b)) and, for each bucket
    of >= 3 contacts whose positions lie in a single plane (perpendicular to the bucket's folded mean normal),
    keeps only the 2D convex hull vertices of the projected positions. Buckets whose positions are not single-plane
    (e.g. multi-wall corner with contacts on perpendicular surfaces) are left untouched. The normal direction of
    each surviving contact is preserved verbatim; the bucket's mean normal is used only as the projection direction.

    The single ``tol`` parameter controls the depth gate as a dimensionless slop fraction:
      max |out-of-plane offset| / in-plane radius <= tol.

    Phases (per env, scratch sized to max_contact_pairs):
    1. Group by canonical link-pair: insertion-sort ``contact_sort_idx`` by (min_link, max_link) key, reading link
       data through the current index permutation.
    2. Per bucket of >= 3 contacts: compute mean normal (folded to a common hemisphere). Check depth coplanarity of
       contact positions. If they share a plane, project to (u, v), Andrew's monotone chain. Mark survivors in
       contact_keep[] (indexed by bucket-logical position).
    3. Compact: squeeze dropped slots out of ``contact_sort_idx`` and update ``n_contacts``.

    During phase 2 the (u, v) bucket sort uses ``contact_keep`` itself as scratch for the per-bucket permutation,
    overwriting it with final keep flags before the bucket exits.
    """
    _B = collider_state.n_contacts.shape[0]
    max_contact_pairs = collider_info.max_contact_pairs[None]
    tol = collider_info.contact_pruning_tolerance[None]
    prune_deep_penetration_ratio = collider_info.prune_deep_penetration_ratio[None]
    LP_KEY_STRIDE = gs.qd_float(1.0e7)
    EPS = rigid_global_info.EPS[None]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        n_con = qd.min(collider_state.n_contacts[i_b], max_contact_pairs)
        collider_state.n_contacts[i_b] = n_con

        # Identity permutation. Required so downstream consumers can always indirect through contact_sort_idx,
        # even when neither pruning nor spatial sort is active.
        for i_c in range(n_con):
            collider_state.contact_sort_idx[i_c, i_b] = i_c

        # === Pruning phase (link-pair support polygon). Gated by static config: only emitted when the
        # scene has multi-geom links / nonconvex / terrain, and not in autodiff mode. Skipped at runtime
        # when contact_pruning_tolerance is 0.
        if qd.static(collider_static_config.has_prunable_contacts and not static_rigid_sim_config.requires_grad):
            if n_con >= 3 and tol > gs.qd_float(0.0):
                # Phase 1: insertion-sort contact_sort_idx by canonical (min_link, max_link) key. The sort_idx
                # already holds the identity from the unconditional init above, so the initial key read is direct.
                for i_c in range(n_con):
                    i_la = collider_state.contact_data.link_a[i_c, i_b]
                    i_lb = collider_state.contact_data.link_b[i_c, i_b]
                    i_l_min = qd.min(i_la, i_lb)
                    i_l_max = qd.max(i_la, i_lb)
                    collider_state.contact_sort_key[i_c, i_b] = qd.cast(i_l_min, gs.qd_float) * LP_KEY_STRIDE + qd.cast(
                        i_l_max, gs.qd_float
                    )

                for i_c in range(1, n_con):
                    key_p = collider_state.contact_sort_key[i_c, i_b]
                    if collider_state.contact_sort_key[i_c - 1, i_b] <= key_p:
                        continue
                    i_p = collider_state.contact_sort_idx[i_c, i_b]
                    j_c = i_c - 1
                    while j_c >= 0:
                        if collider_state.contact_sort_key[j_c, i_b] <= key_p:
                            break
                        collider_state.contact_sort_key[j_c + 1, i_b] = collider_state.contact_sort_key[j_c, i_b]
                        collider_state.contact_sort_idx[j_c + 1, i_b] = collider_state.contact_sort_idx[j_c, i_b]
                        j_c = j_c - 1
                    collider_state.contact_sort_key[j_c + 1, i_b] = key_p
                    collider_state.contact_sort_idx[j_c + 1, i_b] = i_p

                # Default: keep everything. Buckets that pass the gates flip their entries to drop and then mark
                # only hull-vertex contacts as keep again.
                for i_c in range(n_con):
                    collider_state.contact_keep[i_c, i_b] = 1

                # Phase 2: walk link-pair buckets (logical-contiguous after the sort above).
                i_cb_start = 0
                while i_cb_start < n_con:
                    i_pc0 = collider_state.contact_sort_idx[i_cb_start, i_b]
                    i_la0 = collider_state.contact_data.link_a[i_pc0, i_b]
                    i_lb0 = collider_state.contact_data.link_b[i_pc0, i_b]
                    i_l_min0 = qd.min(i_la0, i_lb0)
                    i_l_max0 = qd.max(i_la0, i_lb0)
                    i_cb_end = i_cb_start + 1
                    while i_cb_end < n_con:
                        i_pc = collider_state.contact_sort_idx[i_cb_end, i_b]
                        i_la = collider_state.contact_data.link_a[i_pc, i_b]
                        i_lb = collider_state.contact_data.link_b[i_pc, i_b]
                        if qd.min(i_la, i_lb) != i_l_min0 or qd.max(i_la, i_lb) != i_l_max0:
                            break
                        i_cb_end += 1
                    n_cb = i_cb_end - i_cb_start

                    if n_cb >= 3:
                        # Deterministic within-bucket order. Phase 1 only orders by the link-pair key, so contacts
                        # sharing a key keep the non-deterministic physical layout (atomic_add slot reservation,
                        # multi-pass narrowphase). The downstream (u, v) lex sort uses a non-transitive tolerance
                        # comparison, so its result - and thus the kept hull-vertex set - depends on that input order.
                        # Sorting the bucket by the contact's own position (a pure function of contact data) makes the
                        # survivor set reproducible.
                        for i_cb in range(i_cb_start + 1, i_cb_end):
                            i_p = collider_state.contact_sort_idx[i_cb, i_b]
                            pos_p = collider_state.contact_data.pos[i_p, i_b]
                            normal_p = collider_state.contact_data.normal[i_p, i_b]
                            geom_a_p = collider_state.contact_data.geom_a[i_p, i_b]
                            geom_b_p = collider_state.contact_data.geom_b[i_p, i_b]
                            pen_p = collider_state.contact_data.penetration[i_p, i_b]
                            j_cb = i_cb - 1
                            while j_cb >= i_cb_start:
                                j_p = collider_state.contact_sort_idx[j_cb, i_b]
                                pos_q = collider_state.contact_data.pos[j_p, i_b]
                                # Total order over the contact's intrinsic data: position, then geom pair, then normal,
                                # then penetration. Position alone leaves coincident contacts from different geoms (e.g.
                                # adjacent ring wedges touching the pole at one shared point) tied, so they keep the
                                # non-deterministic atomic-slot order and the downstream (u, v) hull dedup picks a
                                # different survivor run-to-run.
                                precedes = False
                                if pos_q[0] != pos_p[0]:
                                    precedes = pos_q[0] < pos_p[0]
                                elif pos_q[1] != pos_p[1]:
                                    precedes = pos_q[1] < pos_p[1]
                                elif pos_q[2] != pos_p[2]:
                                    precedes = pos_q[2] < pos_p[2]
                                else:
                                    geom_a_q = collider_state.contact_data.geom_a[j_p, i_b]
                                    geom_b_q = collider_state.contact_data.geom_b[j_p, i_b]
                                    normal_q = collider_state.contact_data.normal[j_p, i_b]
                                    if geom_a_q != geom_a_p:
                                        precedes = geom_a_q < geom_a_p
                                    elif geom_b_q != geom_b_p:
                                        precedes = geom_b_q < geom_b_p
                                    elif normal_q[0] != normal_p[0]:
                                        precedes = normal_q[0] < normal_p[0]
                                    elif normal_q[1] != normal_p[1]:
                                        precedes = normal_q[1] < normal_p[1]
                                    elif normal_q[2] != normal_p[2]:
                                        precedes = normal_q[2] < normal_p[2]
                                    else:
                                        precedes = collider_state.contact_data.penetration[j_p, i_b] <= pen_p
                                if precedes:
                                    break
                                collider_state.contact_sort_idx[j_cb + 1, i_b] = j_p
                                j_cb -= 1
                            collider_state.contact_sort_idx[j_cb + 1, i_b] = i_p
                        i_pc0 = collider_state.contact_sort_idx[i_cb_start, i_b]

                        # Mean normal (folded to the hemisphere of contact at i_cb_start) and centroid.
                        normal_ref = collider_state.contact_data.normal[i_pc0, i_b]
                        normal_ref_x = normal_ref[0]
                        normal_ref_y = normal_ref[1]
                        normal_ref_z = normal_ref[2]
                        mean_normal_x = gs.qd_float(0.0)
                        mean_normal_y = gs.qd_float(0.0)
                        mean_normal_z = gs.qd_float(0.0)
                        centroid_x = gs.qd_float(0.0)
                        centroid_y = gs.qd_float(0.0)
                        centroid_z = gs.qd_float(0.0)
                        for i_cb in range(i_cb_start, i_cb_end):
                            i_pc = collider_state.contact_sort_idx[i_cb, i_b]
                            normal_c = collider_state.contact_data.normal[i_pc, i_b]
                            dot_ref = (
                                normal_ref_x * normal_c[0] + normal_ref_y * normal_c[1] + normal_ref_z * normal_c[2]
                            )
                            sign = gs.qd_float(1.0)
                            if dot_ref < gs.qd_float(0.0):
                                sign = gs.qd_float(-1.0)
                            mean_normal_x += sign * normal_c[0]
                            mean_normal_y += sign * normal_c[1]
                            mean_normal_z += sign * normal_c[2]
                            pos_c = collider_state.contact_data.pos[i_pc, i_b]
                            centroid_x += pos_c[0]
                            centroid_y += pos_c[1]
                            centroid_z += pos_c[2]
                        inv_n_cb = gs.qd_float(1.0) / qd.cast(n_cb, gs.qd_float)
                        centroid_x *= inv_n_cb
                        centroid_y *= inv_n_cb
                        centroid_z *= inv_n_cb
                        mean_normal_norm = qd.sqrt(
                            mean_normal_x * mean_normal_x
                            + mean_normal_y * mean_normal_y
                            + mean_normal_z * mean_normal_z
                        )

                        # Hoisted out so the hull-build branch below can read it (quadrants scopes per if).
                        max_in_plane_r2 = gs.qd_float(0.0)

                        coplanar = mean_normal_norm > EPS
                        if coplanar:
                            mean_normal_x /= mean_normal_norm
                            mean_normal_y /= mean_normal_norm
                            mean_normal_z /= mean_normal_norm

                            # Depth coplanarity: positions must lie in a single plane perpendicular to the mean normal. No
                            # per-contact normal check: a contact whose normal is diagonal (e.g. an edge-vs-edge contact at a
                            # corner of the contact patch) still participates in the 2D hull because its position is a vertex of
                            # the patch; dropping a collinear-edge contact in the same bucket is justified by the positional
                            # support polygon regardless of that contact's normal direction.
                            max_depth = gs.qd_float(0.0)
                            for i_cb in range(i_cb_start, i_cb_end):
                                i_pc = collider_state.contact_sort_idx[i_cb, i_b]
                                pos_c = collider_state.contact_data.pos[i_pc, i_b]
                                delta_x = pos_c[0] - centroid_x
                                delta_y = pos_c[1] - centroid_y
                                delta_z = pos_c[2] - centroid_z
                                depth = qd.abs(
                                    delta_x * mean_normal_x + delta_y * mean_normal_y + delta_z * mean_normal_z
                                )
                                if depth > max_depth:
                                    max_depth = depth
                                radius_sq = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z - depth * depth
                                if radius_sq > max_in_plane_r2:
                                    max_in_plane_r2 = radius_sq

                            if max_depth > tol * qd.sqrt(max_in_plane_r2):
                                coplanar = False

                        if coplanar:
                            # In-plane basis (u, v): seed from the world axis least-aligned with mean normal.
                            abs_mean_normal_x = qd.abs(mean_normal_x)
                            abs_mean_normal_y = qd.abs(mean_normal_y)
                            abs_mean_normal_z = qd.abs(mean_normal_z)
                            axis_x = gs.qd_float(1.0)
                            axis_y = gs.qd_float(0.0)
                            axis_z = gs.qd_float(0.0)
                            if abs_mean_normal_y < abs_mean_normal_x and abs_mean_normal_y < abs_mean_normal_z:
                                axis_x = gs.qd_float(0.0)
                                axis_y = gs.qd_float(1.0)
                                axis_z = gs.qd_float(0.0)
                            elif abs_mean_normal_z < abs_mean_normal_x and abs_mean_normal_z <= abs_mean_normal_y:
                                axis_x = gs.qd_float(0.0)
                                axis_y = gs.qd_float(0.0)
                                axis_z = gs.qd_float(1.0)
                            axis_dot_normal = axis_x * mean_normal_x + axis_y * mean_normal_y + axis_z * mean_normal_z
                            u_x = axis_x - axis_dot_normal * mean_normal_x
                            u_y = axis_y - axis_dot_normal * mean_normal_y
                            u_z = axis_z - axis_dot_normal * mean_normal_z
                            u_norm = qd.sqrt(u_x * u_x + u_y * u_y + u_z * u_z)
                            u_x /= u_norm
                            u_y /= u_norm
                            u_z /= u_norm
                            v_x = mean_normal_y * u_z - mean_normal_z * u_y
                            v_y = mean_normal_z * u_x - mean_normal_x * u_z
                            v_z = mean_normal_x * u_y - mean_normal_y * u_x

                            # Project bucket contacts to (u, v). sort_key holds u, contact_proj_v holds v. Both are
                            # indexed by bucket-logical position so the (u, v) sort below can read them without another
                            # indirection.
                            for i_cb in range(i_cb_start, i_cb_end):
                                i_pc = collider_state.contact_sort_idx[i_cb, i_b]
                                pos_c = collider_state.contact_data.pos[i_pc, i_b]
                                collider_state.contact_sort_key[i_cb, i_b] = (
                                    pos_c[0] * u_x + pos_c[1] * u_y + pos_c[2] * u_z
                                )
                                collider_state.contact_proj_v[i_cb, i_b] = (
                                    pos_c[0] * v_x + pos_c[1] * v_y + pos_c[2] * v_z
                                )

                            # Sort bucket positions lexicographically by (u, v), with a tolerance on u so that contacts
                            # whose u values differ only by float noise (or by sub-millimeter physics noise from MPR
                            # perturbations) sort by v. Without the tolerance, the wrong point pops from a 3-collinear
                            # triplet when the corner and the mid-edge have u values that differ by a few microns and
                            # the mid-edge happens to sort first.
                            #
                            # The permutation lives in contact_keep[b_start..b_end). contact_keep is rewritten with the
                            # final keep flags below before this bucket exits, so reusing it as scratch is safe.
                            sort_u_tol = gs.qd_float(1e-3) * qd.sqrt(max_in_plane_r2)
                            for i_cb in range(i_cb_start, i_cb_end):
                                collider_state.contact_keep[i_cb, i_b] = i_cb
                            for i_cb in range(i_cb_start + 1, i_cb_end):
                                i_p = collider_state.contact_keep[i_cb, i_b]
                                u_p = collider_state.contact_sort_key[i_p, i_b]
                                v_p = collider_state.contact_proj_v[i_p, i_b]
                                j_cb = i_cb - 1
                                while j_cb >= i_cb_start:
                                    j_p = collider_state.contact_keep[j_cb, i_b]
                                    u_q = collider_state.contact_sort_key[j_p, i_b]
                                    v_q = collider_state.contact_proj_v[j_p, i_b]
                                    if (u_q < u_p - sort_u_tol) or (qd.abs(u_q - u_p) <= sort_u_tol and v_q <= v_p):
                                        break
                                    collider_state.contact_keep[j_cb + 1, i_b] = j_p
                                    j_cb -= 1
                                collider_state.contact_keep[j_cb + 1, i_b] = i_p

                            # Collinearity threshold for hull pops, scaled to the bucket extent. A pure "cross <= 0"
                            # check fails on numerically-near-collinear edge points (cross is a tiny positive epsilon
                            # from float roundoff), so genuine midpoints would survive as spurious  hull vertices.
                            hull_collinear_tol = tol * max_in_plane_r2

                            # Andrew's monotone chain. The (u, v) permutation lives in contact_keep; the hull stack
                            # lives in contact_hull_stack[i_cb_start..i_cb_start + n_hull). Both store bucket-logical
                            # indices in [i_cb_start, i_cb_end).
                            # Track the top two hull-stack entries in locals rather than re-reading the just-written
                            # contact_hull_stack slots. On Apple Metal, reading a slot written in the previous iteration
                            # can return a stale value (a compiler bug) that leaves collinear points unpruned. A sync
                            # fence between the passes helps in some cases but does not universally fix this family of
                            # bugs (especially with fields), so the re-read is avoided; only the deeper entry reloaded.
                            n_hull = 0
                            i_ht = qd.i32(-1)
                            i_hs = qd.i32(-1)
                            for i_cb in range(i_cb_start, i_cb_end):
                                i_p = collider_state.contact_keep[i_cb, i_b]
                                u_p = collider_state.contact_sort_key[i_p, i_b]
                                v_p = collider_state.contact_proj_v[i_p, i_b]
                                while n_hull >= 2:
                                    u_hs = collider_state.contact_sort_key[i_hs, i_b]
                                    v_hs = collider_state.contact_proj_v[i_hs, i_b]
                                    u_ht = collider_state.contact_sort_key[i_ht, i_b]
                                    v_ht = collider_state.contact_proj_v[i_ht, i_b]
                                    cross = (u_ht - u_hs) * (v_p - v_hs) - (v_ht - v_hs) * (u_p - u_hs)
                                    if cross <= hull_collinear_tol:
                                        n_hull -= 1
                                        i_ht = i_hs
                                        if n_hull >= 2:
                                            i_hs = collider_state.contact_hull_stack[i_cb_start + n_hull - 2, i_b]
                                    else:
                                        break
                                collider_state.contact_hull_stack[i_cb_start + n_hull, i_b] = i_p
                                i_hs = i_ht
                                i_ht = i_p
                                n_hull += 1

                            n_hull_lower = n_hull
                            for i_step in range(n_cb - 1):
                                i_cb = i_cb_end - 2 - i_step
                                i_p = collider_state.contact_keep[i_cb, i_b]
                                u_p = collider_state.contact_sort_key[i_p, i_b]
                                v_p = collider_state.contact_proj_v[i_p, i_b]
                                while n_hull >= n_hull_lower + 1:
                                    u_hs = collider_state.contact_sort_key[i_hs, i_b]
                                    v_hs = collider_state.contact_proj_v[i_hs, i_b]
                                    u_ht = collider_state.contact_sort_key[i_ht, i_b]
                                    v_ht = collider_state.contact_proj_v[i_ht, i_b]
                                    cross = (u_ht - u_hs) * (v_p - v_hs) - (v_ht - v_hs) * (u_p - u_hs)
                                    if cross <= hull_collinear_tol:
                                        n_hull -= 1
                                        i_ht = i_hs
                                        if n_hull >= n_hull_lower + 1:
                                            i_hs = collider_state.contact_hull_stack[i_cb_start + n_hull - 2, i_b]
                                    else:
                                        break
                                # The closing iteration of the upper hull visits the leftmost point, which already sits
                                # at stack[i_cb_start] from the lower hull. Skipping that push, plus the n_hull < n_cb
                                # guard, bounds n_hull to n_cb and keeps the write index within max_contact_pairs even
                                # for buckets where the lower-hull pass already kept all n_cb points (downward-convex
                                # layouts: every lex-sorted triple makes a left turn so nothing gets popped, then the
                                # upper-hull pass tries to push a duplicate of an already-kept lower-hull vertex).
                                if i_p != collider_state.contact_hull_stack[i_cb_start, i_b] and n_hull < n_cb:
                                    collider_state.contact_hull_stack[i_cb_start + n_hull, i_b] = i_p
                                    i_hs = i_ht
                                    i_ht = i_p
                                    n_hull += 1

                            # Overwrite contact_keep[b_start..b_end) (previously the (u, v) permutation scratch)
                            # with the final drop/keep flags: drop everything, then mark hull vertices keep.
                            for i_cb in range(i_cb_start, i_cb_end):
                                collider_state.contact_keep[i_cb, i_b] = 0
                            for i_h in range(n_hull):
                                i_hv = collider_state.contact_hull_stack[i_cb_start + i_h, i_b]
                                collider_state.contact_keep[i_hv, i_b] = 1

                            # Restore non-hull contacts whose penetration is much deeper than the hull boundary's
                            # average. The support-polygon argument says interior contacts are wrench-redundant only
                            # when ALL contacts share the same normal and penetration; a contact with substantially
                            # higher penetration than the hull's average represents a distinct physical support (the
                            # body of a fork resting beyond its tines, the deep middle of a long body) and dropping it
                            # lets the body sink into the surface. The 3x factor is well above the typical ~1.x
                            # penetration spread on transient/rocking faces (so non-uniform-penetration buckets like
                            # irregular mesh contacts keep only the hull) but well below the deep interior penetrations
                            # seen when a non-flat body rests inside its convex envelope (so genuine deep supports are
                            # restored).
                            hull_pen_max = gs.qd_float(0.0)
                            for i_h in range(n_hull):
                                i_hv = collider_state.contact_hull_stack[i_cb_start + i_h, i_b]
                                i_pc = collider_state.contact_sort_idx[i_hv, i_b]
                                pen = collider_state.contact_data.penetration[i_pc, i_b]
                                if pen > hull_pen_max:
                                    hull_pen_max = pen
                            deep_keep_threshold = prune_deep_penetration_ratio * hull_pen_max
                            for i_cb in range(i_cb_start, i_cb_end):
                                if collider_state.contact_keep[i_cb, i_b] == 0:
                                    i_pc = collider_state.contact_sort_idx[i_cb, i_b]
                                    if collider_state.contact_data.penetration[i_pc, i_b] > deep_keep_threshold:
                                        collider_state.contact_keep[i_cb, i_b] = 1

                    i_cb_start = i_cb_end

                # Phase 3: compact contact_sort_idx by squeezing out dropped slots.
                i_cw = 0
                for i_cr in range(n_con):
                    if collider_state.contact_keep[i_cr, i_b] != 0:
                        if i_cw != i_cr:
                            collider_state.contact_sort_idx[i_cw, i_b] = collider_state.contact_sort_idx[i_cr, i_b]
                        i_cw += 1
                collider_state.n_contacts[i_b] = i_cw

        # === Spatial sort by x-position with geom-pair grouping. Gated on collider_static_config.
        # spatial_sort_supported, which combines the narrowphase condition (has_non_box_plane_convex_convex on GPU)
        # with the use_contact_island override (forced off when the island path consumes contacts). Permutes
        # contact_sort_idx only; contact_data is never written.
        if qd.static(collider_static_config.spatial_sort_supported):
            n_con = collider_state.n_contacts[i_b]
            # Per-contact spatial key (own x-position). The key is a pure function of contact data, so the logical
            # order is independent of the non-deterministic physical contact layout (atomic_add slot reservation in
            # the narrowphase, plus the multi-pass narrowphase interleaving same-geom-pair contacts non-contiguously).
            for i_c in range(n_con):
                i_pc = collider_state.contact_sort_idx[i_c, i_b]
                collider_state.contact_sort_key[i_c, i_b] = collider_state.contact_data.pos[i_pc, i_b][0]

            # Insertion-sort contact_sort_idx by the total order (sort_key, geom_a, geom_b, pos_y, pos_z); (key, idx)
            # swap together, no contact_data writes. The position tie-break fully disambiguates contacts that share an
            # x sort_key (different geom pairs, or several contacts within one geom-pair manifold), which is required
            # for bit-reproducible simulation.
            for i_c in range(1, n_con):
                key_p = collider_state.contact_sort_key[i_c, i_b]
                i_p = collider_state.contact_sort_idx[i_c, i_b]
                geom_a_p = collider_state.contact_data.geom_a[i_p, i_b]
                geom_b_p = collider_state.contact_data.geom_b[i_p, i_b]
                pos_p = collider_state.contact_data.pos[i_p, i_b]
                pos_p_y = pos_p[1]
                pos_p_z = pos_p[2]
                j_c = i_c - 1
                while j_c >= 0:
                    j_p = collider_state.contact_sort_idx[j_c, i_b]
                    key_q = collider_state.contact_sort_key[j_c, i_b]
                    precedes = key_q < key_p
                    if not precedes and key_q == key_p:
                        geom_a_q = collider_state.contact_data.geom_a[j_p, i_b]
                        if geom_a_q < geom_a_p:
                            precedes = True
                        elif geom_a_q == geom_a_p:
                            geom_b_q = collider_state.contact_data.geom_b[j_p, i_b]
                            if geom_b_q < geom_b_p:
                                precedes = True
                            elif geom_b_q == geom_b_p:
                                pos_q = collider_state.contact_data.pos[j_p, i_b]
                                if pos_q[1] < pos_p_y:
                                    precedes = True
                                elif pos_q[1] == pos_p_y:
                                    precedes = pos_q[2] <= pos_p_z
                    if precedes:
                        break
                    collider_state.contact_sort_key[j_c + 1, i_b] = key_q
                    collider_state.contact_sort_idx[j_c + 1, i_b] = j_p
                    j_c = j_c - 1
                collider_state.contact_sort_key[j_c + 1, i_b] = key_p
                collider_state.contact_sort_idx[j_c + 1, i_b] = i_p


@qd.kernel(fastcache=True)
def func_clamp_prune_and_sort_contacts_coop(
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    collider_static_config: qd.template(),
):
    """GPU-only cooperative warp-per-env variant of `func_clamp_prune_and_sort_contacts`.

    Same contract (mandatory clamp + identity-init contact_sort_idx; gated prune; gated spatial sort) and same
    pruning algorithm as the serial fused kernel. Difference: 32 warp lanes split the per-env work:
      - PARALLEL: per-contact init, phase-2 mean-normal / centroid reductions, coplanarity reduction, in-plane
        projection writes, phase-1a bitonic sort (when n_con <= 32; falls back to serial insertion sort otherwise).
      - SERIAL on lane 0: bucket walk control, lex sort, Andrew's monotone chain, hull-mark, deep-pen restore, and
        the phase-3 compact (with fused spatial sort when `collider_static_config.spatial_sort_supported`).
    """
    _B = collider_state.n_contacts.shape[0]
    max_contact_pairs = collider_info.max_contact_pairs[None]
    tol = collider_info.contact_pruning_tolerance[None]
    prune_deep_penetration_ratio = collider_info.prune_deep_penetration_ratio[None]
    LP_KEY_STRIDE = gs.qd_float(1.0e7)
    EPS = rigid_global_info.EPS[None]

    _K = qd.static(32)
    _LOG2_K = qd.static(_K.bit_length() - 1)  # = log2(_K), assuming _K is a power of two.
    qd.loop_config(name="clamp_prune_and_sort_contacts_coop", block_dim=_K)
    for i_flat in range(_B * _K):
        tid = i_flat % _K
        i_b = i_flat // _K
        # All lanes compute n_con (cheap, no memory write on non-lane-0).
        n_con = qd.min(collider_state.n_contacts[i_b], max_contact_pairs)
        if tid == 0:
            collider_state.n_contacts[i_b] = n_con

        # PARALLEL: clamp+init. Mirrors the fused kernel's unconditional init block: every env (including n_con < 5
        # where the prune/sort branch below is skipped) needs contact_sort_idx set to identity so downstream consumers
        # that always indirect through contact_sort_idx (constraint solver, sensors) read valid permutations rather
        # than stale data from the previous step. contact_keep default-keep is set here for the same reason. 32 lanes
        # stride.
        i_c_ = tid
        while i_c_ < n_con:
            collider_state.contact_keep[i_c_, i_b] = 1
            collider_state.contact_sort_idx[i_c_, i_b] = i_c_
            i_c_ += _K

        if n_con >= 3:
            # PARALLEL: phase 1a key init, 32 lanes stride. contact_sort_idx identity was already written in the
            # unconditional init block above so the phase-1a sort can read+sort it in place.
            i_c_ = tid
            while i_c_ < n_con:
                i_la = collider_state.contact_data.link_a[i_c_, i_b]
                i_lb = collider_state.contact_data.link_b[i_c_, i_b]
                i_l_min = qd.min(i_la, i_lb)
                i_l_max = qd.max(i_la, i_lb)
                collider_state.contact_sort_key[i_c_, i_b] = qd.cast(i_l_min, gs.qd_float) * LP_KEY_STRIDE + qd.cast(
                    i_l_max, gs.qd_float
                )
                i_c_ += _K

            # Phase 1a sort: bitonic sort across _K lanes when n_con <= _K, serial-on-lane-0 insertion sort
            # otherwise.
            if n_con <= _K:
                # Load with sentinel for out-of-range lanes (pushes them to the end of ascending sort).
                my_key = qd.cast(gs.qd_float(1.0e30), gs.qd_float)
                my_idx = qd.i32(-1)
                if tid < n_con:
                    my_key = collider_state.contact_sort_key[tid, i_b]
                    my_idx = collider_state.contact_sort_idx[tid, i_b]

                my_key, my_idx = qd.simt.subgroup.bitonic_sort_kv_tiled(my_key, my_idx, _LOG2_K)

                # Write back the sorted values for the real range.
                if tid < n_con:
                    collider_state.contact_sort_key[tid, i_b] = my_key
                    collider_state.contact_sort_idx[tid, i_b] = my_idx
            elif tid == 0:
                # Serial fallback: insertion sort on lane 0 for n_con > 32.
                for i_c in range(1, n_con):
                    key_p = collider_state.contact_sort_key[i_c, i_b]
                    if collider_state.contact_sort_key[i_c - 1, i_b] <= key_p:
                        continue
                    i_p = collider_state.contact_sort_idx[i_c, i_b]
                    j_c = i_c - 1
                    while j_c >= 0:
                        if collider_state.contact_sort_key[j_c, i_b] <= key_p:
                            break
                        collider_state.contact_sort_key[j_c + 1, i_b] = collider_state.contact_sort_key[j_c, i_b]
                        collider_state.contact_sort_idx[j_c + 1, i_b] = collider_state.contact_sort_idx[j_c, i_b]
                        j_c = j_c - 1
                    collider_state.contact_sort_key[j_c + 1, i_b] = key_p
                    collider_state.contact_sort_idx[j_c + 1, i_b] = i_p

            qd.simt.subgroup.sync()

            # Phase 2: bucket walk control runs on all 32 lanes (inputs are DRAM-cached). Inside a bucket, mean-normal
            # / centroid sums and the coplanarity-check max-reduction run coop via subgroup reduce_all_*; the lex
            # sort, hull build, mark-survivors, and deep-pen restore stay serial on lane 0.
            i_cb_start = 0
            while i_cb_start < n_con:
                key_b = collider_state.contact_sort_key[i_cb_start, i_b]
                i_cb_end = i_cb_start + 1
                while i_cb_end < n_con:
                    if collider_state.contact_sort_key[i_cb_end, i_b] != key_b:
                        break
                    i_cb_end += 1
                n_cb = i_cb_end - i_cb_start

                if n_cb >= 3:
                    # Deterministic within-bucket order. Phase 1a only orders by the link-pair key, so contacts
                    # sharing a key keep the non-deterministic physical layout (atomic_add slot reservation, multi-pass
                    # narrowphase). The downstream (u, v) lex sort uses a non-transitive tolerance comparison, so its
                    # result - and thus the kept hull-vertex set - depends on that input order. Sorting the bucket by
                    # the contact's own position (a pure function of contact data) makes the survivor set reproducible.
                    # Serial on lane 0; sync so the strided coop reductions below read the reordered indices.
                    if tid == 0:
                        for i_cb in range(i_cb_start + 1, i_cb_end):
                            i_p = collider_state.contact_sort_idx[i_cb, i_b]
                            pos_p = collider_state.contact_data.pos[i_p, i_b]
                            normal_p = collider_state.contact_data.normal[i_p, i_b]
                            geom_a_p = collider_state.contact_data.geom_a[i_p, i_b]
                            geom_b_p = collider_state.contact_data.geom_b[i_p, i_b]
                            pen_p = collider_state.contact_data.penetration[i_p, i_b]
                            j_cb = i_cb - 1
                            while j_cb >= i_cb_start:
                                j_p = collider_state.contact_sort_idx[j_cb, i_b]
                                pos_q = collider_state.contact_data.pos[j_p, i_b]
                                # Total order over the contact's intrinsic data: position, then geom pair, then normal,
                                # then penetration. Position alone leaves coincident contacts from different geoms (e.g.
                                # adjacent ring wedges touching the pole at one shared point) tied, so they keep the
                                # non-deterministic atomic-slot order and the downstream (u, v) hull dedup picks a
                                # different survivor run-to-run.
                                precedes = False
                                if pos_q[0] != pos_p[0]:
                                    precedes = pos_q[0] < pos_p[0]
                                elif pos_q[1] != pos_p[1]:
                                    precedes = pos_q[1] < pos_p[1]
                                elif pos_q[2] != pos_p[2]:
                                    precedes = pos_q[2] < pos_p[2]
                                else:
                                    geom_a_q = collider_state.contact_data.geom_a[j_p, i_b]
                                    geom_b_q = collider_state.contact_data.geom_b[j_p, i_b]
                                    normal_q = collider_state.contact_data.normal[j_p, i_b]
                                    if geom_a_q != geom_a_p:
                                        precedes = geom_a_q < geom_a_p
                                    elif geom_b_q != geom_b_p:
                                        precedes = geom_b_q < geom_b_p
                                    elif normal_q[0] != normal_p[0]:
                                        precedes = normal_q[0] < normal_p[0]
                                    elif normal_q[1] != normal_p[1]:
                                        precedes = normal_q[1] < normal_p[1]
                                    elif normal_q[2] != normal_p[2]:
                                        precedes = normal_q[2] < normal_p[2]
                                    else:
                                        precedes = collider_state.contact_data.penetration[j_p, i_b] <= pen_p
                                if precedes:
                                    break
                                collider_state.contact_sort_idx[j_cb + 1, i_b] = j_p
                                j_cb -= 1
                            collider_state.contact_sort_idx[j_cb + 1, i_b] = i_p
                    qd.simt.subgroup.sync()

                    i_pc0 = collider_state.contact_sort_idx[i_cb_start, i_b]
                    normal_ref = collider_state.contact_data.normal[i_pc0, i_b]
                    normal_ref_x = normal_ref[0]
                    normal_ref_y = normal_ref[1]
                    normal_ref_z = normal_ref[2]
                    mean_normal_x_l = gs.qd_float(0.0)
                    mean_normal_y_l = gs.qd_float(0.0)
                    mean_normal_z_l = gs.qd_float(0.0)
                    centroid_x_l = gs.qd_float(0.0)
                    centroid_y_l = gs.qd_float(0.0)
                    centroid_z_l = gs.qd_float(0.0)
                    i_cb_ = i_cb_start + tid
                    while i_cb_ < i_cb_end:
                        i_pc = collider_state.contact_sort_idx[i_cb_, i_b]
                        normal_c = collider_state.contact_data.normal[i_pc, i_b]
                        dot_ref = normal_ref_x * normal_c[0] + normal_ref_y * normal_c[1] + normal_ref_z * normal_c[2]
                        sign = gs.qd_float(1.0)
                        if dot_ref < gs.qd_float(0.0):
                            sign = gs.qd_float(-1.0)
                        mean_normal_x_l += sign * normal_c[0]
                        mean_normal_y_l += sign * normal_c[1]
                        mean_normal_z_l += sign * normal_c[2]
                        pos_c = collider_state.contact_data.pos[i_pc, i_b]
                        centroid_x_l += pos_c[0]
                        centroid_y_l += pos_c[1]
                        centroid_z_l += pos_c[2]
                        i_cb_ += _K

                    mean_normal_x = qd.simt.subgroup.reduce_all_add_tiled(mean_normal_x_l, 5)
                    mean_normal_y = qd.simt.subgroup.reduce_all_add_tiled(mean_normal_y_l, 5)
                    mean_normal_z = qd.simt.subgroup.reduce_all_add_tiled(mean_normal_z_l, 5)
                    centroid_x = qd.simt.subgroup.reduce_all_add_tiled(centroid_x_l, 5)
                    centroid_y = qd.simt.subgroup.reduce_all_add_tiled(centroid_y_l, 5)
                    centroid_z = qd.simt.subgroup.reduce_all_add_tiled(centroid_z_l, 5)

                    # POST-REDUCE math runs on all 32 lanes (deterministic, cheap; redundant arithmetic is free vs.
                    # broadcasting the reduce results).
                    inv_n_cb = gs.qd_float(1.0) / qd.cast(n_cb, gs.qd_float)
                    centroid_x *= inv_n_cb
                    centroid_y *= inv_n_cb
                    centroid_z *= inv_n_cb
                    mean_normal_norm = qd.sqrt(
                        mean_normal_x * mean_normal_x + mean_normal_y * mean_normal_y + mean_normal_z * mean_normal_z
                    )

                    max_in_plane_r2 = gs.qd_float(0.0)
                    coplanar = mean_normal_norm > EPS
                    if coplanar:
                        mean_normal_x /= mean_normal_norm
                        mean_normal_y /= mean_normal_norm
                        mean_normal_z /= mean_normal_norm

                        # COOP coplanarity check (stage 3). Each lane strides [i_cb_start + tid, i_cb_end) by _K,
                        # locally tracking max_depth / max_in_plane_r2. Wasted work per warp is at most n_cb/_K.
                        # The upstream algo no longer checks per-contact normals (a contact with a diagonal normal at
                        # the corner of a patch still participates in the 2D hull because its position is a vertex), so
                        # we only do the depth coplanarity gate here.
                        max_depth_l = gs.qd_float(0.0)
                        max_radius_sq_l = gs.qd_float(0.0)
                        i_cb_ = i_cb_start + tid
                        while i_cb_ < i_cb_end:
                            i_pc = collider_state.contact_sort_idx[i_cb_, i_b]
                            pos_c = collider_state.contact_data.pos[i_pc, i_b]
                            delta_x = pos_c[0] - centroid_x
                            delta_y = pos_c[1] - centroid_y
                            delta_z = pos_c[2] - centroid_z
                            depth = qd.abs(delta_x * mean_normal_x + delta_y * mean_normal_y + delta_z * mean_normal_z)
                            if depth > max_depth_l:
                                max_depth_l = depth
                            radius_sq = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z - depth * depth
                            if radius_sq > max_radius_sq_l:
                                max_radius_sq_l = radius_sq
                            i_cb_ += _K

                        max_depth = qd.simt.subgroup.reduce_all_max_tiled(max_depth_l, 5)
                        max_in_plane_r2 = qd.simt.subgroup.reduce_all_max_tiled(max_radius_sq_l, 5)

                        if max_depth > tol * qd.sqrt(max_in_plane_r2):
                            coplanar = False

                    if coplanar:
                        # Basis on all lanes (deterministic from the mean normal the reduce broadcast to every lane).
                        abs_mean_normal_x = qd.abs(mean_normal_x)
                        abs_mean_normal_y = qd.abs(mean_normal_y)
                        abs_mean_normal_z = qd.abs(mean_normal_z)
                        axis_x = gs.qd_float(1.0)
                        axis_y = gs.qd_float(0.0)
                        axis_z = gs.qd_float(0.0)
                        if abs_mean_normal_y < abs_mean_normal_x and abs_mean_normal_y < abs_mean_normal_z:
                            axis_x = gs.qd_float(0.0)
                            axis_y = gs.qd_float(1.0)
                            axis_z = gs.qd_float(0.0)
                        elif abs_mean_normal_z < abs_mean_normal_x and abs_mean_normal_z <= abs_mean_normal_y:
                            axis_x = gs.qd_float(0.0)
                            axis_y = gs.qd_float(0.0)
                            axis_z = gs.qd_float(1.0)
                        axis_dot_normal = axis_x * mean_normal_x + axis_y * mean_normal_y + axis_z * mean_normal_z
                        u_x = axis_x - axis_dot_normal * mean_normal_x
                        u_y = axis_y - axis_dot_normal * mean_normal_y
                        u_z = axis_z - axis_dot_normal * mean_normal_z
                        u_norm = qd.sqrt(u_x * u_x + u_y * u_y + u_z * u_z)
                        u_x /= u_norm
                        u_y /= u_norm
                        u_z /= u_norm
                        v_x = mean_normal_y * u_z - mean_normal_z * u_y
                        v_y = mean_normal_z * u_x - mean_normal_x * u_z
                        v_z = mean_normal_x * u_y - mean_normal_y * u_x

                        # COOP projection: 32 lanes stride writes to contact_sort_key + contact_proj_v.
                        i_cb_ = i_cb_start + tid
                        while i_cb_ < i_cb_end:
                            i_pc = collider_state.contact_sort_idx[i_cb_, i_b]
                            pos_c = collider_state.contact_data.pos[i_pc, i_b]
                            collider_state.contact_sort_key[i_cb_, i_b] = (
                                pos_c[0] * u_x + pos_c[1] * u_y + pos_c[2] * u_z
                            )
                            collider_state.contact_proj_v[i_cb_, i_b] = pos_c[0] * v_x + pos_c[1] * v_y + pos_c[2] * v_z
                            i_cb_ += _K

                        # COOP mark-drop: stride writes to contact_keep[i_pc].
                        i_cb_ = i_cb_start + tid
                        while i_cb_ < i_cb_end:
                            i_pc = collider_state.contact_sort_idx[i_cb_, i_b]
                            collider_state.contact_keep[i_pc, i_b] = 0
                            i_cb_ += _K

                        # COOP lex_idx init: stride writes.
                        i_cb_ = i_cb_start + tid
                        while i_cb_ < i_cb_end:
                            collider_state.contact_lex_idx[i_cb_, i_b] = i_cb_
                            i_cb_ += _K

                        # SYNC between coop writes (sort_key, proj_v, lex_idx, contact_keep[i_pc]) and the lane-0 lex
                        # sort + hull build that reads them.
                        qd.simt.subgroup.sync()

                    if tid == 0 and coplanar:
                        sort_u_tol = gs.qd_float(1e-3) * qd.sqrt(max_in_plane_r2)
                        for i_cb in range(i_cb_start + 1, i_cb_end):
                            i_p = collider_state.contact_lex_idx[i_cb, i_b]
                            u_p = collider_state.contact_sort_key[i_p, i_b]
                            v_p = collider_state.contact_proj_v[i_p, i_b]
                            j_cb = i_cb - 1
                            while j_cb >= i_cb_start:
                                j_p = collider_state.contact_lex_idx[j_cb, i_b]
                                u_q = collider_state.contact_sort_key[j_p, i_b]
                                v_q = collider_state.contact_proj_v[j_p, i_b]
                                if (u_q < u_p - sort_u_tol) or (qd.abs(u_q - u_p) <= sort_u_tol and v_q <= v_p):
                                    break
                                collider_state.contact_lex_idx[j_cb + 1, i_b] = j_p
                                j_cb -= 1
                            collider_state.contact_lex_idx[j_cb + 1, i_b] = i_p

                        hull_collinear_tol = tol * max_in_plane_r2

                        # Track the top two hull-stack entries in locals rather than re-reading the just-written
                        # contact_hull_stack slots. On Apple Metal, reading a slot written in the previous iteration
                        # can return a stale value (a compiler bug) that leaves collinear points unpruned. A sync
                        # fence between the passes helps in some cases but does not universally fix this family of
                        # bugs, especially with fields, so the re-read is avoided; only the deeper entry is reloaded.
                        n_hull = 0
                        i_ht = qd.i32(-1)
                        i_hs = qd.i32(-1)
                        for i_cb in range(i_cb_start, i_cb_end):
                            i_p = collider_state.contact_lex_idx[i_cb, i_b]
                            u_p = collider_state.contact_sort_key[i_p, i_b]
                            v_p = collider_state.contact_proj_v[i_p, i_b]
                            while n_hull >= 2:
                                u_hs = collider_state.contact_sort_key[i_hs, i_b]
                                v_hs = collider_state.contact_proj_v[i_hs, i_b]
                                u_ht = collider_state.contact_sort_key[i_ht, i_b]
                                v_ht = collider_state.contact_proj_v[i_ht, i_b]
                                cross = (u_ht - u_hs) * (v_p - v_hs) - (v_ht - v_hs) * (u_p - u_hs)
                                if cross <= hull_collinear_tol:
                                    n_hull -= 1
                                    i_ht = i_hs
                                    if n_hull >= 2:
                                        i_hs = collider_state.contact_hull_stack[i_cb_start + n_hull - 2, i_b]
                                else:
                                    break
                            collider_state.contact_hull_stack[i_cb_start + n_hull, i_b] = i_p
                            i_hs = i_ht
                            i_ht = i_p
                            n_hull += 1

                        n_hull_lower = n_hull
                        for i_step in range(n_cb - 1):
                            i_cb = i_cb_end - 2 - i_step
                            i_p = collider_state.contact_lex_idx[i_cb, i_b]
                            u_p = collider_state.contact_sort_key[i_p, i_b]
                            v_p = collider_state.contact_proj_v[i_p, i_b]
                            while n_hull >= n_hull_lower + 1:
                                u_hs = collider_state.contact_sort_key[i_hs, i_b]
                                v_hs = collider_state.contact_proj_v[i_hs, i_b]
                                u_ht = collider_state.contact_sort_key[i_ht, i_b]
                                v_ht = collider_state.contact_proj_v[i_ht, i_b]
                                cross = (u_ht - u_hs) * (v_p - v_hs) - (v_ht - v_hs) * (u_p - u_hs)
                                if cross <= hull_collinear_tol:
                                    n_hull -= 1
                                    i_ht = i_hs
                                    if n_hull >= n_hull_lower + 1:
                                        i_hs = collider_state.contact_hull_stack[i_cb_start + n_hull - 2, i_b]
                                else:
                                    break
                            if i_p != collider_state.contact_hull_stack[i_cb_start, i_b] and n_hull < n_cb:
                                collider_state.contact_hull_stack[i_cb_start + n_hull, i_b] = i_p
                                i_hs = i_ht
                                i_ht = i_p
                                n_hull += 1

                        for i_h in range(n_hull):
                            i_hv = collider_state.contact_hull_stack[i_cb_start + i_h, i_b]
                            i_pc = collider_state.contact_sort_idx[i_hv, i_b]
                            collider_state.contact_keep[i_pc, i_b] = 1

                        # Lane-0 deep-penetration restore. See serial kernel for the rationale. Indices here live in
                        # orig-space because the cycle-permute is fused into phase 3 below (contact_data is still in
                        # pre-sort order, so we translate sort-space hull/bucket indices through contact_sort_idx).
                        hull_pen_max = gs.qd_float(0.0)
                        for i_h in range(n_hull):
                            i_hv = collider_state.contact_hull_stack[i_cb_start + i_h, i_b]
                            i_pc = collider_state.contact_sort_idx[i_hv, i_b]
                            pen = collider_state.contact_data.penetration[i_pc, i_b]
                            if pen > hull_pen_max:
                                hull_pen_max = pen
                        deep_keep_threshold = prune_deep_penetration_ratio * hull_pen_max
                        for i_cb in range(i_cb_start, i_cb_end):
                            i_pc = collider_state.contact_sort_idx[i_cb, i_b]
                            if collider_state.contact_keep[i_pc, i_b] == 0:
                                if collider_state.contact_data.penetration[i_pc, i_b] > deep_keep_threshold:
                                    collider_state.contact_keep[i_pc, i_b] = 1

                i_cb_start = i_cb_end

        if tid == 0:
            if qd.static(collider_static_config.spatial_sort_supported):
                # Phase 3 (with spatial sort): fused compact + spatial sort encoded entirely in contact_sort_idx.
                # Sentinel +inf sort_key pushes dropped slots to the tail; kept slots get their own x-position. The
                # key is a pure function of contact data, so the logical order is independent of the non-deterministic
                # physical contact layout (atomic_add slot reservation in the narrowphase, plus the multi-pass
                # narrowphase interleaving same-geom-pair contacts non-contiguously). Lock-step insertion sort on the
                # total order (sort_key, geom_a, geom_b, pos_y, pos_z) lands sort_idx as the final logical->physical
                # permutation. n_contacts = count of non-sentinel slots.
                SENTINEL_BIG = gs.qd_float(1e30)
                for i_c in range(n_con):
                    if collider_state.contact_keep[i_c, i_b] != 0:
                        collider_state.contact_sort_key[i_c, i_b] = collider_state.contact_data.pos[i_c, i_b][0]
                    else:
                        collider_state.contact_sort_key[i_c, i_b] = SENTINEL_BIG
                    collider_state.contact_sort_idx[i_c, i_b] = i_c

                # Insertion sort by (sort_key, geom_a, geom_b). The geom-pair tie-break makes the logical order
                # independent of the non-deterministic physical contact layout (atomic_add slot reservation in the
                # narrowphase) when several geom-pair groups share the same x sort_key, which is required for
                # bit-reproducible simulation. Dropped slots carry SENTINEL_BIG and sort to the tail.
                for i_c in range(1, n_con):
                    key_p = collider_state.contact_sort_key[i_c, i_b]
                    i_p = collider_state.contact_sort_idx[i_c, i_b]
                    geom_a_p = collider_state.contact_data.geom_a[i_p, i_b]
                    geom_b_p = collider_state.contact_data.geom_b[i_p, i_b]
                    pos_p = collider_state.contact_data.pos[i_p, i_b]
                    pos_p_y = pos_p[1]
                    pos_p_z = pos_p[2]
                    j_c = i_c - 1
                    while j_c >= 0:
                        j_p = collider_state.contact_sort_idx[j_c, i_b]
                        key_q = collider_state.contact_sort_key[j_c, i_b]
                        # (sort_key, geom_a, geom_b, pos_y, pos_z) lexicographic: stop once the left neighbour
                        # precedes-or-equals the current contact in this total order.
                        precedes = key_q < key_p
                        if not precedes and key_q == key_p:
                            geom_a_q = collider_state.contact_data.geom_a[j_p, i_b]
                            if geom_a_q < geom_a_p:
                                precedes = True
                            elif geom_a_q == geom_a_p:
                                geom_b_q = collider_state.contact_data.geom_b[j_p, i_b]
                                if geom_b_q < geom_b_p:
                                    precedes = True
                                elif geom_b_q == geom_b_p:
                                    pos_q = collider_state.contact_data.pos[j_p, i_b]
                                    if pos_q[1] < pos_p_y:
                                        precedes = True
                                    elif pos_q[1] == pos_p_y:
                                        precedes = pos_q[2] <= pos_p_z
                        if precedes:
                            break
                        collider_state.contact_sort_key[j_c + 1, i_b] = key_q
                        collider_state.contact_sort_idx[j_c + 1, i_b] = j_p
                        j_c = j_c - 1
                    collider_state.contact_sort_key[j_c + 1, i_b] = key_p
                    collider_state.contact_sort_idx[j_c + 1, i_b] = i_p

                n_kept = 0
                for i_c in range(n_con):
                    if collider_state.contact_sort_key[i_c, i_b] < SENTINEL_BIG:
                        n_kept += 1
                    else:
                        break
                collider_state.n_contacts[i_b] = n_kept
            else:
                # Phase 3 (compact-only): when spatial sort is statically disabled, preserve the serial kernel's
                # contract -- squeeze dropped orig-space slots out of contact_sort_idx in orig order and update
                # n_contacts. Kept slots map logical-position to physical-position (orig-space).
                i_cw = 0
                for i_c in range(n_con):
                    if collider_state.contact_keep[i_c, i_b] != 0:
                        collider_state.contact_sort_idx[i_cw, i_b] = i_c
                        i_cw += 1
                collider_state.n_contacts[i_b] = i_cw


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
