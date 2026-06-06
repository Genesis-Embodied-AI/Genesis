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
        for i in range(n_con):
            collider_state.contact_sort_idx[i, i_b] = i

        # === Pruning phase (link-pair support polygon). Gated by static config: only emitted when the
        # scene has multi-geom links / nonconvex / terrain, and not in autodiff mode. Skipped at runtime
        # when contact_pruning_tolerance is 0.
        if qd.static(collider_static_config.has_prunable_contacts and not static_rigid_sim_config.requires_grad):
            if n_con >= 3 and tol > gs.qd_float(0.0):
                # Phase 1: insertion-sort contact_sort_idx by canonical (min_link, max_link) key. The sort_idx
                # already holds the identity from the unconditional init above, so the initial key read is direct.
                for i in range(n_con):
                    la = collider_state.contact_data.link_a[i, i_b]
                    lb = collider_state.contact_data.link_b[i, i_b]
                    la_min = qd.min(la, lb)
                    la_max = qd.max(la, lb)
                    collider_state.contact_sort_key[i, i_b] = qd.cast(la_min, gs.qd_float) * LP_KEY_STRIDE + qd.cast(
                        la_max, gs.qd_float
                    )

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

                # Default: keep everything. Buckets that pass the gates flip their entries to drop and then mark
                # only hull-vertex contacts as keep again.
                for i in range(n_con):
                    collider_state.contact_keep[i, i_b] = 1

                # Phase 2: walk link-pair buckets (logical-contiguous after the sort above).
                b_start = 0
                while b_start < n_con:
                    phys0 = collider_state.contact_sort_idx[b_start, i_b]
                    la0 = collider_state.contact_data.link_a[phys0, i_b]
                    lb0 = collider_state.contact_data.link_b[phys0, i_b]
                    la0_min = qd.min(la0, lb0)
                    la0_max = qd.max(la0, lb0)
                    b_end = b_start + 1
                    while b_end < n_con:
                        phys_e = collider_state.contact_sort_idx[b_end, i_b]
                        la = collider_state.contact_data.link_a[phys_e, i_b]
                        lb = collider_state.contact_data.link_b[phys_e, i_b]
                        if qd.min(la, lb) != la0_min or qd.max(la, lb) != la0_max:
                            break
                        b_end += 1
                    b_size = b_end - b_start

                    if b_size >= 3:
                        # Mean normal (folded to the hemisphere of contact at b_start) and centroid.
                        ref_n = collider_state.contact_data.normal[phys0, i_b]
                        rnx = ref_n[0]
                        rny = ref_n[1]
                        rnz = ref_n[2]
                        mnx = gs.qd_float(0.0)
                        mny = gs.qd_float(0.0)
                        mnz = gs.qd_float(0.0)
                        cx = gs.qd_float(0.0)
                        cy = gs.qd_float(0.0)
                        cz = gs.qd_float(0.0)
                        for i in range(b_start, b_end):
                            phys_i = collider_state.contact_sort_idx[i, i_b]
                            n_i = collider_state.contact_data.normal[phys_i, i_b]
                            s = gs.qd_float(1.0)
                            if rnx * n_i[0] + rny * n_i[1] + rnz * n_i[2] < gs.qd_float(0.0):
                                s = gs.qd_float(-1.0)
                            mnx += s * n_i[0]
                            mny += s * n_i[1]
                            mnz += s * n_i[2]
                            p_i = collider_state.contact_data.pos[phys_i, i_b]
                            cx += p_i[0]
                            cy += p_i[1]
                            cz += p_i[2]
                        inv_n = gs.qd_float(1.0) / qd.cast(b_size, gs.qd_float)
                        cx *= inv_n
                        cy *= inv_n
                        cz *= inv_n
                        mnrm = qd.sqrt(mnx * mnx + mny * mny + mnz * mnz)

                        # Hoisted out so the hull-build branch below can read it (quadrants scopes per if).
                        max_in_plane_r2 = gs.qd_float(0.0)

                        coplanar = mnrm > EPS
                        if coplanar:
                            mnx /= mnrm
                            mny /= mnrm
                            mnz /= mnrm

                            # Depth coplanarity: positions must lie in a single plane perpendicular to the mean normal. No
                            # per-contact normal check: a contact whose normal is diagonal (e.g. an edge-vs-edge contact at a
                            # corner of the contact patch) still participates in the 2D hull because its position is a vertex of
                            # the patch; dropping a collinear-edge contact in the same bucket is justified by the positional
                            # support polygon regardless of that contact's normal direction.
                            max_depth = gs.qd_float(0.0)
                            for i in range(b_start, b_end):
                                phys_i = collider_state.contact_sort_idx[i, i_b]
                                p_i = collider_state.contact_data.pos[phys_i, i_b]
                                dx = p_i[0] - cx
                                dy = p_i[1] - cy
                                dz = p_i[2] - cz
                                depth = qd.abs(dx * mnx + dy * mny + dz * mnz)
                                if depth > max_depth:
                                    max_depth = depth
                                r2 = dx * dx + dy * dy + dz * dz - depth * depth
                                if r2 > max_in_plane_r2:
                                    max_in_plane_r2 = r2

                            if max_depth > tol * qd.sqrt(max_in_plane_r2):
                                coplanar = False

                        if coplanar:
                            # In-plane basis (u, v): seed from the world axis least-aligned with mean normal.
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

                            # Project bucket contacts to (u, v). sort_key holds u, contact_proj_v holds v. Both are
                            # indexed by bucket-logical position so the (u, v) sort below can read them without another
                            # indirection.
                            for i in range(b_start, b_end):
                                phys_i = collider_state.contact_sort_idx[i, i_b]
                                p_i = collider_state.contact_data.pos[phys_i, i_b]
                                collider_state.contact_sort_key[i, i_b] = p_i[0] * ux + p_i[1] * uy + p_i[2] * uz
                                collider_state.contact_proj_v[i, i_b] = p_i[0] * vx + p_i[1] * vy + p_i[2] * vz

                            # Sort bucket positions lexicographically by (u, v), with a tolerance on u so that contacts
                            # whose u values differ only by float noise (or by sub-millimeter physics noise from MPR
                            # perturbations) sort by v. Without the tolerance, the wrong point pops from a 3-collinear
                            # triplet when the corner and the mid-edge have u values that differ by a few microns and
                            # the mid-edge happens to sort first.
                            #
                            # The permutation lives in contact_keep[b_start..b_end). contact_keep is rewritten with the
                            # final keep flags below before this bucket exits, so reusing it as scratch is safe.
                            sort_u_tol = gs.qd_float(1e-3) * qd.sqrt(max_in_plane_r2)
                            for i in range(b_start, b_end):
                                collider_state.contact_keep[i, i_b] = i
                            for i in range(b_start + 1, b_end):
                                ci = collider_state.contact_keep[i, i_b]
                                cu = collider_state.contact_sort_key[ci, i_b]
                                cv = collider_state.contact_proj_v[ci, i_b]
                                j = i - 1
                                while j >= b_start:
                                    pj = collider_state.contact_keep[j, i_b]
                                    pu = collider_state.contact_sort_key[pj, i_b]
                                    pv = collider_state.contact_proj_v[pj, i_b]
                                    if (pu < cu - sort_u_tol) or (qd.abs(pu - cu) <= sort_u_tol and pv <= cv):
                                        break
                                    collider_state.contact_keep[j + 1, i_b] = pj
                                    j -= 1
                                collider_state.contact_keep[j + 1, i_b] = ci

                            # Collinearity threshold for hull pops, scaled to the bucket extent. A pure "cross <= 0"
                            # check fails on numerically-near-collinear edge points (cross is a tiny positive epsilon
                            # from float roundoff), so genuine midpoints would survive as spurious  hull vertices.
                            hull_collinear_tol = tol * max_in_plane_r2

                            # Andrew's monotone chain. The (u, v) permutation lives in contact_keep; the hull stack
                            # lives in contact_hull_stack[b_start..b_start + k). Both store bucket-logical indices
                            # in [b_start, b_end).
                            k = 0
                            for i in range(b_start, b_end):
                                ci = collider_state.contact_keep[i, i_b]
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
                            # Memory-fence for a Quadrants codegen issue on parallel envs (Metal backend, _B >= 2):
                            # without an explicit barrier between the lower-hull and upper-hull passes, the upper-
                            # hull pop-loop's reads of contact_hull_stack don't observe the writes from the lower
                            # hull, so its cross-product / pop-check effectively runs on stale data and every
                            # candidate is kept, producing a hull whose size equals the bucket size.
                            if qd.static(static_rigid_sim_config.backend == gs.metal):
                                qd.simt.block.sync()
                            for k_step in range(b_size - 1):
                                ii = b_end - 2 - k_step
                                ci = collider_state.contact_keep[ii, i_b]
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
                                # The closing iteration of the upper hull visits the leftmost point, which already sits
                                # at stack[b_start] from the lower hull. Skipping that push, plus the k < b_size guard,
                                # bounds k to b_size and keeps the write index within max_contact_pairs even for buckets
                                # where the lower-hull pass already kept all b_size points (downward-convex layouts:
                                # every lex-sorted triple makes a left turn so nothing gets popped, then the upper-hull
                                # pass tries to push a duplicate of an already-kept lower-hull vertex).
                                if ci != collider_state.contact_hull_stack[b_start, i_b] and k < b_size:
                                    collider_state.contact_hull_stack[b_start + k, i_b] = ci
                                    k += 1

                            # Overwrite contact_keep[b_start..b_end) (previously the (u, v) permutation scratch)
                            # with the final drop/keep flags: drop everything, then mark hull vertices keep.
                            for i in range(b_start, b_end):
                                collider_state.contact_keep[i, i_b] = 0
                            for hk in range(k):
                                survivor = collider_state.contact_hull_stack[b_start + hk, i_b]
                                collider_state.contact_keep[survivor, i_b] = 1

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
                            for hk in range(k):
                                survivor = collider_state.contact_hull_stack[b_start + hk, i_b]
                                phys_s = collider_state.contact_sort_idx[survivor, i_b]
                                p = collider_state.contact_data.penetration[phys_s, i_b]
                                if p > hull_pen_max:
                                    hull_pen_max = p
                            deep_keep_threshold = prune_deep_penetration_ratio * hull_pen_max
                            for i in range(b_start, b_end):
                                if collider_state.contact_keep[i, i_b] == 0:
                                    phys_i = collider_state.contact_sort_idx[i, i_b]
                                    if collider_state.contact_data.penetration[phys_i, i_b] > deep_keep_threshold:
                                        collider_state.contact_keep[i, i_b] = 1

                    b_start = b_end

                # Phase 3: compact contact_sort_idx by squeezing out dropped slots.
                write = 0
                for read in range(n_con):
                    if collider_state.contact_keep[read, i_b] != 0:
                        if write != read:
                            collider_state.contact_sort_idx[write, i_b] = collider_state.contact_sort_idx[read, i_b]
                        write += 1
                collider_state.n_contacts[i_b] = write

        # === Spatial sort by x-position with geom-pair grouping. Gated on collider_static_config.
        # spatial_sort_supported, which combines the narrowphase condition (has_non_box_plane_convex_convex on GPU)
        # with the use_contact_island override (forced off when the island path consumes contacts). Permutes
        # contact_sort_idx only; contact_data is never written.
        if qd.static(collider_static_config.spatial_sort_supported):
            n_con = collider_state.n_contacts[i_b]
            # Build per-logical-position spatial keys, treating consecutive same-geom-pair contacts as one group.
            group_key = gs.qd_float(0.0)
            for i in range(n_con):
                phys = collider_state.contact_sort_idx[i, i_b]
                ga = collider_state.contact_data.geom_a[phys, i_b]
                gb = collider_state.contact_data.geom_b[phys, i_b]
                new_group = i == 0
                if i > 0:
                    prev_phys = collider_state.contact_sort_idx[i - 1, i_b]
                    if (
                        ga != collider_state.contact_data.geom_a[prev_phys, i_b]
                        or gb != collider_state.contact_data.geom_b[prev_phys, i_b]
                    ):
                        new_group = True
                if new_group:
                    group_key = collider_state.contact_data.pos[phys, i_b][0]
                collider_state.contact_sort_key[i, i_b] = group_key

            # Insertion-sort contact_sort_idx by (sort_key, geom_a, geom_b); (key, idx) swap together, no contact_data
            # writes. The geom-pair tie-break makes the logical order independent of the non-deterministic physical
            # contact layout (atomic_add slot reservation in the narrowphase) when several geom-pair groups share the
            # same x sort_key, which is required for bit-reproducible simulation.
            for i in range(1, n_con):
                curr_key = collider_state.contact_sort_key[i, i_b]
                curr_idx = collider_state.contact_sort_idx[i, i_b]
                cga = collider_state.contact_data.geom_a[curr_idx, i_b]
                cgb = collider_state.contact_data.geom_b[curr_idx, i_b]
                j = i - 1
                while j >= 0:
                    pj = collider_state.contact_sort_idx[j, i_b]
                    pk = collider_state.contact_sort_key[j, i_b]
                    pga = collider_state.contact_data.geom_a[pj, i_b]
                    if pk < curr_key or (
                        pk == curr_key
                        and (pga < cga or (pga == cga and collider_state.contact_data.geom_b[pj, i_b] <= cgb))
                    ):
                        break
                    collider_state.contact_sort_key[j + 1, i_b] = pk
                    collider_state.contact_sort_idx[j + 1, i_b] = pj
                    j = j - 1
                collider_state.contact_sort_key[j + 1, i_b] = curr_key
                collider_state.contact_sort_idx[j + 1, i_b] = curr_idx


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
        ii = tid
        while ii < n_con:
            collider_state.contact_keep[ii, i_b] = 1
            collider_state.contact_sort_idx[ii, i_b] = ii
            ii += _K

        if n_con >= 3:
            # PARALLEL: phase 1a key init, 32 lanes stride. contact_sort_idx identity was already written in the
            # unconditional init block above so the phase-1a sort can read+sort it in place.
            ii = tid
            while ii < n_con:
                la = collider_state.contact_data.link_a[ii, i_b]
                lb = collider_state.contact_data.link_b[ii, i_b]
                la_min = qd.min(la, lb)
                la_max = qd.max(la, lb)
                collider_state.contact_sort_key[ii, i_b] = qd.cast(la_min, gs.qd_float) * LP_KEY_STRIDE + qd.cast(
                    la_max, gs.qd_float
                )
                ii += _K

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

            # Phase 2: bucket walk control runs on all 32 lanes (inputs are DRAM-cached). Inside a bucket, mean-normal
            # / centroid sums and the coplanarity-check max-reduction run coop via subgroup reduce_all_*; the lex
            # sort, hull build, mark-survivors, and deep-pen restore stay serial on lane 0.
            b_start = 0
            while b_start < n_con:
                key0 = collider_state.contact_sort_key[b_start, i_b]
                b_end = b_start + 1
                while b_end < n_con:
                    if collider_state.contact_sort_key[b_end, i_b] != key0:
                        break
                    b_end += 1
                b_size = b_end - b_start

                if b_size >= 3:
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

                        hull_collinear_tol = tol * max_in_plane_r2

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
                        # Lane-0 variant of the lower/upper hull memory-fence workaround used in the serial kernel
                        # (PR #2831): write to a non-overlapping scratch slot to force write-then-read ordering on
                        # contact_hull_stack between the two hull passes.
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

                        # Lane-0 deep-penetration restore. See serial kernel for the rationale. Indices here live in
                        # orig-space because the cycle-permute is fused into phase 3 below (contact_data is still in
                        # pre-sort order, so we translate sort-space hull/bucket indices through contact_sort_idx).
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
            if qd.static(collider_static_config.spatial_sort_supported):
                # Phase 3 (with spatial sort): fused compact + spatial sort encoded entirely in contact_sort_idx.
                # Sentinel +inf sort_key pushes dropped slots to the tail; kept slots get the geom-pair group's
                # x-pos for spatial locality. Lock-step insertion sort on (sort_key, sort_idx) lands sort_idx as
                # the final logical->physical permutation. n_contacts = count of non-sentinel slots.
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

                # Insertion sort by (sort_key, geom_a, geom_b). The geom-pair tie-break makes the logical order
                # independent of the non-deterministic physical contact layout (atomic_add slot reservation in the
                # narrowphase) when several geom-pair groups share the same x sort_key, which is required for
                # bit-reproducible simulation. Dropped slots carry SENTINEL_BIG and sort to the tail.
                for i in range(1, n_con):
                    ck = collider_state.contact_sort_key[i, i_b]
                    ci = collider_state.contact_sort_idx[i, i_b]
                    cga = collider_state.contact_data.geom_a[ci, i_b]
                    cgb = collider_state.contact_data.geom_b[ci, i_b]
                    j = i - 1
                    while j >= 0:
                        pj = collider_state.contact_sort_idx[j, i_b]
                        pk = collider_state.contact_sort_key[j, i_b]
                        pga = collider_state.contact_data.geom_a[pj, i_b]
                        if pk < ck or (
                            pk == ck
                            and (pga < cga or (pga == cga and collider_state.contact_data.geom_b[pj, i_b] <= cgb))
                        ):
                            break
                        collider_state.contact_sort_key[j + 1, i_b] = pk
                        collider_state.contact_sort_idx[j + 1, i_b] = pj
                        j = j - 1
                    collider_state.contact_sort_key[j + 1, i_b] = ck
                    collider_state.contact_sort_idx[j + 1, i_b] = ci

                n_kept = 0
                for i in range(n_con):
                    if collider_state.contact_sort_key[i, i_b] < SENTINEL_BIG:
                        n_kept += 1
                    else:
                        break
                collider_state.n_contacts[i_b] = n_kept
            else:
                # Phase 3 (compact-only): when spatial sort is statically disabled, preserve the serial kernel's
                # contract -- squeeze dropped orig-space slots out of contact_sort_idx in orig order and update
                # n_contacts. Kept slots map logical-position w to physical-position i (orig-space).
                write = 0
                for i in range(n_con):
                    if collider_state.contact_keep[i, i_b] != 0:
                        collider_state.contact_sort_idx[write, i_b] = i
                        write += 1
                collider_state.n_contacts[i_b] = write


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
