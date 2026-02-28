import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class
from . import utils


@qd.func
def func_sphere_sphere_contact(
    i_ga,
    i_gb,
    ga_pos,
    gb_pos,
    geoms_info: array_class.GeomsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
):
    """Analytical sphere-sphere collision detection.

    Contact normal points from B to A.
    """
    EPS = rigid_global_info.EPS[None]

    radius_a = geoms_info.data[i_ga][0]
    radius_b = geoms_info.data[i_gb][0]

    diff = ga_pos - gb_pos
    dist_sq = diff.dot(diff)
    combined_radius = radius_a + radius_b

    is_col = False
    normal = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
    contact_pos = qd.Vector.zero(gs.qd_float, 3)
    penetration = gs.qd_float(0.0)

    if dist_sq < combined_radius * combined_radius:
        is_col = True
        dist = qd.sqrt(dist_sq)

        if dist > EPS:
            normal = diff / dist
        penetration = combined_radius - dist
        contact_pos = ga_pos - (radius_a - 0.5 * penetration) * normal

    return is_col, normal, contact_pos, penetration


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
        # Axial projection is within barrel extent
        axis_point = cyl_pos + axial_proj * axis
        if radial_dist <= cyl_radius:
            # Inside cylinder: closest is barrel or nearest cap
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
    ga_pos,
    ga_quat,
    gb_pos,
    gb_quat,
    geoms_info: array_class.GeomsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
):
    """Analytical cylinder-sphere collision detection.

    Either geom may be the cylinder or the sphere; the function normalises
    so that the normal always points from B to A (original ordering).
    """
    EPS = rigid_global_info.EPS[None]

    # Identify which is the cylinder and which is the sphere
    normal_sign = 1
    cyl_idx = i_ga
    cyl_pos = ga_pos
    cyl_quat = ga_quat
    sph_idx = i_gb
    sph_pos = gb_pos
    if geoms_info.type[i_gb] == gs.GEOM_TYPE.CYLINDER:
        cyl_idx = i_gb
        cyl_pos = gb_pos
        cyl_quat = gb_quat
        sph_idx = i_ga
        sph_pos = ga_pos
        normal_sign = -1

    cyl_radius = geoms_info.data[cyl_idx][0]
    cyl_halflength = 0.5 * geoms_info.data[cyl_idx][1]
    sph_radius = geoms_info.data[sph_idx][0]

    closest, _is_on_cap = _closest_point_on_cylinder(cyl_pos, cyl_quat, cyl_radius, cyl_halflength, sph_pos, EPS)

    diff = sph_pos - closest
    dist_sq = diff.dot(diff)

    is_col = False
    normal = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
    contact_pos = qd.Vector.zero(gs.qd_float, 3)
    penetration = gs.qd_float(0.0)

    if dist_sq < sph_radius * sph_radius:
        is_col = True
        dist = qd.sqrt(dist_sq)
        if dist > EPS:
            normal = diff / dist
        penetration = sph_radius - dist
        contact_pos = sph_pos - (sph_radius - 0.5 * penetration) * normal

    # Flip normal to point from B to A in original ordering
    normal = normal * gs.qd_float(normal_sign)

    return is_col, normal, contact_pos, penetration


@qd.func
def func_cylinder_cylinder_contact(
    i_ga,
    i_gb,
    ga_pos,
    ga_quat,
    gb_pos,
    gb_quat,
    geoms_info: array_class.GeomsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
):
    """Analytical cylinder-cylinder collision detection.

    Decomposes into barrel-barrel (closest points on axes, clamped to barrel
    extent, then radial check), barrel-cap, and cap-cap sub-cases.  Normal
    points from B to A.
    """
    EPS = rigid_global_info.EPS[None]

    radius_a = geoms_info.data[i_ga][0]
    halflength_a = 0.5 * geoms_info.data[i_ga][1]
    radius_b = geoms_info.data[i_gb][0]
    halflength_b = 0.5 * geoms_info.data[i_gb][1]

    local_z = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)
    axis_a = gu.qd_transform_by_quat_fast(local_z, ga_quat)
    axis_b = gu.qd_transform_by_quat_fast(local_z, gb_quat)

    # Endpoints of each cylinder's axis segment
    A1 = ga_pos - halflength_a * axis_a
    A2 = ga_pos + halflength_a * axis_a
    B1 = gb_pos - halflength_b * axis_b
    B2 = gb_pos + halflength_b * axis_b

    combined_radius = radius_a + radius_b
    center_diff = ga_pos - gb_pos

    is_col = False
    normal = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
    contact_pos = qd.Vector.zero(gs.qd_float, 3)
    penetration = gs.qd_float(0.0)
    arb = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)

    axis_cross = axis_a.cross(axis_b)
    axis_cross_len_sq = axis_cross.dot(axis_cross)

    if axis_cross_len_sq < EPS * EPS:
        # Parallel (or nearly parallel) axes — use perpendicular distance and
        # check barrel overlap, since closest-points-on-segments gives misleading
        # endpoint distances for parallel segments.
        perp_vec = center_diff - center_diff.dot(axis_a) * axis_a
        perp_dist = qd.sqrt(perp_vec.dot(perp_vec))

        b_on_a = -center_diff.dot(axis_a)  # B's center projected onto A's axis
        overlap_start = qd.max(-halflength_a, b_on_a - halflength_b)
        overlap_end = qd.min(halflength_a, b_on_a + halflength_b)

        if overlap_end > overlap_start and perp_dist < combined_radius:
            is_col = True
            penetration = combined_radius - perp_dist
            if perp_dist > EPS:
                normal = perp_vec / perp_dist
            else:
                if qd.abs(axis_a[0]) < 0.9:
                    arb = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
                else:
                    arb = qd.Vector([0.0, 1.0, 0.0], dt=gs.qd_float)
                normal = arb - arb.dot(axis_a) * axis_a
                normal = normal / qd.sqrt(normal.dot(normal) + 1e-30)
            overlap_center = 0.5 * (overlap_start + overlap_end)
            contact_pos = ga_pos + overlap_center * axis_a - (radius_a - 0.5 * penetration) * normal
        elif perp_dist < combined_radius:
            # Barrels don't overlap but caps may interact (e.g., end-on approach).
            # Fall through to segment-endpoint closest-point check below.
            Pa, Pb = utils.func_closest_points_on_segments(A1, A2, B1, B2, EPS)
            radial = Pa - Pb
            radial_dist = qd.sqrt(radial.dot(radial))
            if radial_dist > EPS and radial_dist < combined_radius:
                is_col = True
                normal = radial / radial_dist
                penetration = combined_radius - radial_dist
                contact_pos = Pb + (radius_b - 0.5 * penetration) * normal
    else:
        # Non-parallel axes: closest-points-on-segments gives the true minimum.
        Pa, Pb = utils.func_closest_points_on_segments(A1, A2, B1, B2, EPS)
        radial = Pa - Pb
        radial_dist = qd.sqrt(radial.dot(radial))

        if radial_dist > EPS:
            if radial_dist < combined_radius:
                is_col = True
                normal = radial / radial_dist
                penetration = combined_radius - radial_dist
                contact_pos = Pb + (radius_b - 0.5 * penetration) * normal
        else:
            # Axes intersect: separation along cross product
            cross_len = qd.sqrt(axis_cross_len_sq)
            sep_dir = axis_cross / cross_len
            if sep_dir.dot(center_diff) < 0.0:
                sep_dir = -sep_dir
            normal = sep_dir
            penetration = combined_radius
            contact_pos = 0.5 * (Pa + Pb)
            is_col = True

    return is_col, normal, contact_pos, penetration
