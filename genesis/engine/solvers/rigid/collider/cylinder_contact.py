import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class
from . import utils
from .contact import func_add_contact


@qd.func
def func_sphere_sphere_contact(
    i_ga,
    i_gb,
    i_b,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    errno: array_class.V_ANNOTATION,
):
    """Analytical sphere-sphere collision detection.

    Contact normal points from B to A.
    """
    EPS = rigid_global_info.EPS[None]
    ga_pos = geoms_state.pos[i_ga, i_b]
    gb_pos = geoms_state.pos[i_gb, i_b]

    radius_a = geoms_info.data[i_ga][0]
    radius_b = geoms_info.data[i_gb][0]

    diff = ga_pos - gb_pos
    dist_sq = diff.dot(diff)
    combined_radius = radius_a + radius_b

    if dist_sq < combined_radius * combined_radius:
        dist = qd.sqrt(dist_sq)

        normal = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
        if dist > EPS:
            normal = diff / dist
        penetration = combined_radius - dist
        contact_pos = ga_pos - (radius_a - 0.5 * penetration) * normal

        func_add_contact(
            i_ga,
            i_gb,
            normal,
            contact_pos,
            penetration,
            i_b,
            geoms_state,
            geoms_info,
            collider_state,
            collider_info,
            errno,
        )


@qd.func
def _closest_point_on_cylinder(
    cyl_pos,
    cyl_quat,
    cyl_radius: gs.qd_float,
    cyl_halflength: gs.qd_float,
    query_point,
    EPS,
):
    """Find the closest point on a cylinder surface (barrel + flat caps) to a query point.

    Returns (closest_point, is_on_cap).  The cylinder is oriented along local Z.
    """
    local_z = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)
    axis = gu.qd_transform_by_quat_fast(local_z, cyl_quat)

    delta = query_point - cyl_pos
    axial_proj = delta.dot(axis)
    radial_vec = delta - axial_proj * axis
    radial_dist = qd.sqrt(radial_vec.dot(radial_vec))

    is_on_cap = False
    closest = qd.Vector.zero(gs.qd_float, 3)
    perp = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)

    radial_dir = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
    has_radial = radial_dist > EPS

    if has_radial:
        radial_dir = radial_vec / radial_dist
    else:
        if qd.abs(axis[0]) < 0.9:
            perp = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
        else:
            perp = qd.Vector([0.0, 1.0, 0.0], dt=gs.qd_float)
        radial_dir = perp - perp.dot(axis) * axis
        radial_dir = radial_dir / qd.sqrt(radial_dir.dot(radial_dir) + 1e-30)

    beyond_cap = qd.abs(axial_proj) > cyl_halflength

    if beyond_cap:
        cap_sign = gs.qd_float(1.0)
        if axial_proj < 0.0:
            cap_sign = gs.qd_float(-1.0)
        cap_center = cyl_pos + cap_sign * cyl_halflength * axis
        is_on_cap = True
        if radial_dist <= cyl_radius:
            closest = cap_center + radial_dist * radial_dir
        else:
            closest = cap_center + cyl_radius * radial_dir
    else:
        axis_point = cyl_pos + axial_proj * axis
        if radial_dist <= cyl_radius:
            barrel_dist = cyl_radius - radial_dist
            cap_dist = cyl_halflength - qd.abs(axial_proj)
            if barrel_dist < cap_dist:
                closest = axis_point + cyl_radius * radial_dir
            else:
                cap_sign = gs.qd_float(1.0)
                if axial_proj < 0.0:
                    cap_sign = gs.qd_float(-1.0)
                closest = cyl_pos + cap_sign * cyl_halflength * axis + radial_dist * radial_dir
                is_on_cap = True
        else:
            closest = axis_point + cyl_radius * radial_dir

    return closest, is_on_cap


@qd.func
def func_cylinder_sphere_contact(
    i_ga,
    i_gb,
    i_b,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    errno: array_class.V_ANNOTATION,
):
    """Analytical cylinder-sphere collision detection.

    Either geom may be the cylinder or the sphere; the function normalises
    so that the normal always points from B to A (original ordering).
    """
    EPS = rigid_global_info.EPS[None]

    normal_sign = 1
    cyl_idx = i_ga
    cyl_pos = geoms_state.pos[i_ga, i_b]
    cyl_quat = geoms_state.quat[i_ga, i_b]
    sph_idx = i_gb
    sph_pos = geoms_state.pos[i_gb, i_b]
    if geoms_info.type[i_gb] == gs.GEOM_TYPE.CYLINDER:
        cyl_idx = i_gb
        cyl_pos = geoms_state.pos[i_gb, i_b]
        cyl_quat = geoms_state.quat[i_gb, i_b]
        sph_idx = i_ga
        sph_pos = geoms_state.pos[i_ga, i_b]
        normal_sign = -1

    cyl_radius = geoms_info.data[cyl_idx][0]
    cyl_halflength = 0.5 * geoms_info.data[cyl_idx][1]
    sph_radius = geoms_info.data[sph_idx][0]

    closest, _is_on_cap = _closest_point_on_cylinder(cyl_pos, cyl_quat, cyl_radius, cyl_halflength, sph_pos, EPS)

    diff = sph_pos - closest
    dist_sq = diff.dot(diff)

    if dist_sq < sph_radius * sph_radius:
        dist = qd.sqrt(dist_sq)
        normal = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
        if dist > EPS:
            normal = diff / dist
        penetration = sph_radius - dist
        contact_pos = sph_pos - (sph_radius - 0.5 * penetration) * normal

        normal = normal * gs.qd_float(normal_sign)

        func_add_contact(
            i_ga,
            i_gb,
            normal,
            contact_pos,
            penetration,
            i_b,
            geoms_state,
            geoms_info,
            collider_state,
            collider_info,
            errno,
        )


@qd.func
def _add_cylinder_cylinder_contact(
    i_ga,
    i_gb,
    i_b,
    normal,
    contact_pos,
    penetration,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    errno: array_class.V_ANNOTATION,
):
    func_add_contact(
        i_ga, i_gb, normal, contact_pos, penetration, i_b, geoms_state, geoms_info, collider_state, collider_info, errno
    )


@qd.func
def func_cylinder_cylinder_contact(
    i_ga,
    i_gb,
    i_b,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: qd.template(),
    errno: array_class.V_ANNOTATION,
):
    """Analytical cylinder-cylinder collision detection with native multi-contact.

    For parallel cylinders with barrel overlap, generates contacts at both ends
    and center of the overlap region.  Normal points from B to A.
    """
    EPS = rigid_global_info.EPS[None]

    ga_pos = geoms_state.pos[i_ga, i_b]
    ga_quat = geoms_state.quat[i_ga, i_b]
    gb_pos = geoms_state.pos[i_gb, i_b]
    gb_quat = geoms_state.quat[i_gb, i_b]

    radius_a = geoms_info.data[i_ga][0]
    halflength_a = 0.5 * geoms_info.data[i_ga][1]
    radius_b = geoms_info.data[i_gb][0]
    halflength_b = 0.5 * geoms_info.data[i_gb][1]

    local_z = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)
    axis_a = gu.qd_transform_by_quat_fast(local_z, ga_quat)
    axis_b = gu.qd_transform_by_quat_fast(local_z, gb_quat)

    A1 = ga_pos - halflength_a * axis_a
    A2 = ga_pos + halflength_a * axis_a
    B1 = gb_pos - halflength_b * axis_b
    B2 = gb_pos + halflength_b * axis_b

    combined_radius = radius_a + radius_b
    center_diff = ga_pos - gb_pos

    arb = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)

    axis_cross = axis_a.cross(axis_b)
    axis_cross_len_sq = axis_cross.dot(axis_cross)

    if axis_cross_len_sq < EPS * EPS:
        # Parallel axes
        perp_vec = center_diff - center_diff.dot(axis_a) * axis_a
        perp_dist = qd.sqrt(perp_vec.dot(perp_vec))

        b_on_a = -center_diff.dot(axis_a)
        overlap_start = qd.max(-halflength_a, b_on_a - halflength_b)
        overlap_end = qd.min(halflength_a, b_on_a + halflength_b)

        if overlap_end > overlap_start and perp_dist < combined_radius:
            penetration = combined_radius - perp_dist
            normal = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
            if perp_dist > EPS:
                normal = perp_vec / perp_dist
            else:
                if qd.abs(axis_a[0]) < 0.9:
                    arb = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
                else:
                    arb = qd.Vector([0.0, 1.0, 0.0], dt=gs.qd_float)
                normal = arb - arb.dot(axis_a) * axis_a
                normal = normal / qd.sqrt(normal.dot(normal) + 1e-30)

            radial_offset = -(radius_a - 0.5 * penetration) * normal

            # Native multi-contact: center + two endpoints of the overlap region
            overlap_center = 0.5 * (overlap_start + overlap_end)
            contact_pos_center = ga_pos + overlap_center * axis_a + radial_offset
            _add_cylinder_cylinder_contact(
                i_ga,
                i_gb,
                i_b,
                normal,
                contact_pos_center,
                penetration,
                geoms_state,
                geoms_info,
                collider_state,
                collider_info,
                errno,
            )

            overlap_len = overlap_end - overlap_start
            if overlap_len > EPS:
                n_max = qd.static(collider_static_config.n_contacts_per_pair)
                if n_max >= 3:
                    contact_pos_lo = ga_pos + overlap_start * axis_a + radial_offset
                    _add_cylinder_cylinder_contact(
                        i_ga,
                        i_gb,
                        i_b,
                        normal,
                        contact_pos_lo,
                        penetration,
                        geoms_state,
                        geoms_info,
                        collider_state,
                        collider_info,
                        errno,
                    )

                    contact_pos_hi = ga_pos + overlap_end * axis_a + radial_offset
                    _add_cylinder_cylinder_contact(
                        i_ga,
                        i_gb,
                        i_b,
                        normal,
                        contact_pos_hi,
                        penetration,
                        geoms_state,
                        geoms_info,
                        collider_state,
                        collider_info,
                        errno,
                    )

        elif perp_dist < combined_radius:
            Pa, Pb = utils.func_closest_points_on_segments(A1, A2, B1, B2, EPS)
            radial = Pa - Pb
            radial_dist = qd.sqrt(radial.dot(radial))
            if radial_dist > EPS and radial_dist < combined_radius:
                normal = radial / radial_dist
                penetration = combined_radius - radial_dist
                contact_pos = Pb + (radius_b - 0.5 * penetration) * normal
                _add_cylinder_cylinder_contact(
                    i_ga,
                    i_gb,
                    i_b,
                    normal,
                    contact_pos,
                    penetration,
                    geoms_state,
                    geoms_info,
                    collider_state,
                    collider_info,
                    errno,
                )
    else:
        Pa, Pb = utils.func_closest_points_on_segments(A1, A2, B1, B2, EPS)
        radial = Pa - Pb
        radial_dist = qd.sqrt(radial.dot(radial))

        if radial_dist > EPS:
            if radial_dist < combined_radius:
                normal = radial / radial_dist
                penetration = combined_radius - radial_dist
                contact_pos = Pb + (radius_b - 0.5 * penetration) * normal
                _add_cylinder_cylinder_contact(
                    i_ga,
                    i_gb,
                    i_b,
                    normal,
                    contact_pos,
                    penetration,
                    geoms_state,
                    geoms_info,
                    collider_state,
                    collider_info,
                    errno,
                )
        else:
            cross_len = qd.sqrt(axis_cross_len_sq)
            sep_dir = axis_cross / cross_len
            if sep_dir.dot(center_diff) < 0.0:
                sep_dir = -sep_dir
            normal = sep_dir
            penetration = combined_radius
            contact_pos = 0.5 * (Pa + Pb)
            _add_cylinder_cylinder_contact(
                i_ga,
                i_gb,
                i_b,
                normal,
                contact_pos,
                penetration,
                geoms_state,
                geoms_info,
                collider_state,
                collider_info,
                errno,
            )
