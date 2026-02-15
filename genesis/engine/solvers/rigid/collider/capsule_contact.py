import gstaichi as ti

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class


vec3 = ti.types.vector(3, gs.ti_float)


@ti.func
def i_cross_vec(vec: vec3) -> vec3:
    return ti.Vector([0.0, -vec[2], vec[1]], dt=gs.ti_float)


@ti.func
def j_cross_vec(vec: vec3) -> vec3:
    return ti.Vector([vec[2], 0.0, -vec[0]], dt=gs.ti_float)


@ti.func
def k_cross_vec(vec: vec3) -> vec3:
    return ti.Vector([-vec[1], vec[0], 0.0], dt=gs.ti_float)


@ti.func
def transform_vec_by_normalized_quat_fast(
    v: vec3,
    quat: ti.types.vector(4, gs.ti_float),
) -> ti.types.vector[3, gs.ti_float]:
    """
    Assumptions:
    - quat must be normalized
    """
    q_w, q_x, q_y, q_z = quat
    u = ti.Vector([q_x, q_y, q_z])
    t = 2.0 * u.cross(v)
    return v + q_w * t + u.cross(t)


@ti.func
def func_closest_points_on_segments(
    seg_a_p1: ti.types.vector(3, gs.ti_float),
    seg_a_p2: ti.types.vector(3, gs.ti_float),
    seg_b_p1: ti.types.vector(3, gs.ti_float),
    seg_b_p2: ti.types.vector(3, gs.ti_float),
    EPS: gs.ti_float,
):
    """
    Compute closest points on two line segments using analytical solution.

    References
    ----------
    Real-Time Collision Detection by Christer Ericson, Chapter 5.1.9
    """
    segment_a_dir = seg_a_p2 - seg_a_p1
    segment_b_dir = seg_b_p2 - seg_b_p1
    vec_between_segment_origins = seg_a_p1 - seg_b_p1

    a_squared_len = segment_a_dir.dot(segment_a_dir)
    dot_product_dir = segment_a_dir.dot(segment_b_dir)
    b_squared_len = segment_b_dir.dot(segment_b_dir)
    d = segment_a_dir.dot(vec_between_segment_origins)
    e = segment_b_dir.dot(vec_between_segment_origins)

    denom = a_squared_len * b_squared_len - dot_product_dir * dot_product_dir

    s = gs.ti_float(0.0)
    t = gs.ti_float(0.0)

    if denom < EPS:
        # Segments are parallel or one/both are degenerate
        s = 0.0
        if b_squared_len > EPS:
            t = ti.math.clamp(e / b_squared_len, 0.0, 1.0)
        else:
            t = 0.0
    else:
        # General case: solve for optimal parameters
        s = (dot_product_dir * e - b_squared_len * d) / denom
        t = (a_squared_len * e - dot_product_dir * d) / denom

        s = ti.math.clamp(s, 0.0, 1.0)

        # Recompute t for clamped s
        t = ti.math.clamp((dot_product_dir * s + e) / b_squared_len if b_squared_len > EPS else 0.0, 0.0, 1.0)

        # Recompute s for clamped t (ensures we're on segment boundaries)
        s_new = ti.math.clamp((dot_product_dir * t - d) / a_squared_len if a_squared_len > EPS else 0.0, 0.0, 1.0)

        # Use refined s if it improves the solution
        if a_squared_len > EPS:
            s = s_new

    seg_a_closest = seg_a_p1 + s * segment_a_dir
    seg_b_closest = seg_b_p1 + t * segment_b_dir

    return seg_a_closest, seg_b_closest


@ti.func
def func_capsule_capsule_contact(
    i_ga,
    i_gb,
    ga_pos,
    ga_quat,
    gb_pos,
    gb_quat,
    geoms_info: array_class.GeomsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
):
    """
    Analytical capsule-capsule collision detection.

    A capsule is defined as a line segment plus a radius (swept sphere).
    Collision between two capsules reduces to:
      1. Find closest points on the two line segments (analytical)
      2. Check if distance < sum of radii
      3. Compute contact point and normal

    Parameters
    ----------
    ga_pos, ga_quat : Position and orientation of capsule A (may be perturbed for multicontact).
    gb_pos, gb_quat : Position and orientation of capsule B (may be perturbed for multicontact).
    """
    EPS = rigid_global_info.EPS[None]

    # Get capsule A parameters
    pos_a = ga_pos
    quat_a = ga_quat
    radius_a = geoms_info.data[i_ga][0]
    halflength_a = gs.ti_float(0.5) * geoms_info.data[i_ga][1]

    # Get capsule B parameters
    pos_b = gb_pos
    quat_b = gb_quat
    radius_b = geoms_info.data[i_gb][0]
    halflength_b = gs.ti_float(0.5) * geoms_info.data[i_gb][1]

    # Capsules are aligned along local Z-axis by convention
    local_z_unit = ti.Vector([0.0, 0.0, 1.0], dt=gs.ti_float)

    # Get segment axes in world space
    axis_a_unit = transform_vec_by_normalized_quat_fast(local_z_unit, quat_a)
    axis_b_unit = transform_vec_by_normalized_quat_fast(local_z_unit, quat_b)

    # Compute segment endpoints in world space
    P1 = pos_a - halflength_a * axis_a_unit
    P2 = pos_a + halflength_a * axis_a_unit
    Q1 = pos_b - halflength_b * axis_b_unit
    Q2 = pos_b + halflength_b * axis_b_unit

    Pa, Pb = func_closest_points_on_segments(P1, P2, Q1, Q2, EPS)

    # from B to A
    diff = Pa - Pb
    dist_sq = diff.dot(diff)
    combined_radius = radius_a + radius_b
    combined_radius_sq = combined_radius * combined_radius

    is_col = False
    normal_unit = ti.Vector([1.0, 0.0, 0.0], dt=gs.ti_float)
    contact_pos = ti.Vector.zero(gs.ti_float, 3)
    penetration = gs.ti_float(0.0)
    if dist_sq < combined_radius_sq:
        is_col = True
        dist = ti.sqrt(dist_sq)

        # Compute contact normal (from B to A, pointing into geom A)
        if dist > EPS:
            normal_unit = diff / dist
        else:
            # Segments are coincident, use arbitrary perpendicular direction
            # Try cross product with axis_a first
            normal_dir = axis_a_unit.cross(axis_b_unit)
            normal_dir_len = normal_dir.dot(normal_dir)
            if normal_dir_len > EPS:
                normal_unit = normal_dir / normal_dir_len
            else:
                # Axes are parallel, use any perpendicular
                if ti.abs(axis_a_unit[0]) < 0.9:
                    normal_unit = i_cross_vec(axis_a_unit)
                else:
                    normal_unit = j_cross_vec(axis_a_unit)

        penetration = combined_radius - dist
        # Contact position at midpoint between surfaces
        contact_pos = Pa - radius_a * normal_unit + gs.ti_float(0.5) * penetration * normal_unit

    return is_col, normal_unit, contact_pos, penetration


@ti.func
def func_sphere_capsule_contact(
    i_ga,
    i_gb,
    ga_pos,
    ga_quat,
    gb_pos,
    gb_quat,
    geoms_info: array_class.GeomsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
):
    """
    Analytical sphere-capsule collision detection.

    A sphere-capsule collision reduces to:
      1. Find closest point on the capsule's line segment to sphere center
      2. Check if distance < sum of radii
      3. Compute contact point and normal

    Parameters
    ----------
    ga_pos, ga_quat : Position and orientation of geom A (may be perturbed for multicontact).
    gb_pos, gb_quat : Position and orientation of geom B (may be perturbed for multicontact).
    """
    # Ensure sphere is always i_ga and capsule is i_gb
    normal_dir = 1
    sphere_pos = ga_pos
    capsule_pos = gb_pos
    capsule_q = gb_quat
    if geoms_info.type[i_gb] == gs.GEOM_TYPE.SPHERE:
        i_ga, i_gb = i_gb, i_ga
        sphere_pos = gb_pos
        capsule_pos = ga_pos
        capsule_q = ga_quat
        normal_dir = -1

    EPS = rigid_global_info.EPS[None]

    sphere_center = sphere_pos
    sphere_radius = geoms_info.data[i_ga][0]

    capsule_center = capsule_pos
    capsule_quat = capsule_q
    capsule_radius = geoms_info.data[i_gb][0]
    capsule_halflength = gs.ti_float(0.5) * geoms_info.data[i_gb][1]

    # Capsule is aligned along local Z-axis
    local_z_unit = ti.Vector([0.0, 0.0, 1.0], dt=gs.ti_float)
    capsule_axis = transform_vec_by_normalized_quat_fast(local_z_unit, capsule_quat)

    # Compute capsule segment endpoints
    P1 = capsule_center - capsule_halflength * capsule_axis
    P2 = capsule_center + capsule_halflength * capsule_axis

    # Find closest point on capsule segment to sphere center
    # Using parametric form: P(t) = P1 + t*(P2-P1), t ∈ [0,1]
    segment_vec = P2 - P1
    segment_length_sq = segment_vec.dot(segment_vec)

    # Project sphere center onto segment
    # Default for degenerate case
    t = gs.ti_float(0.5)
    if segment_length_sq > EPS:
        t = (sphere_center - P1).dot(segment_vec) / segment_length_sq
        t = ti.math.clamp(t, 0.0, 1.0)

    closest_point = P1 + t * segment_vec

    # Compute distance from sphere center to closest point
    diff = sphere_center - closest_point
    dist_sq = diff.dot(diff)
    combined_radius = sphere_radius + capsule_radius
    combined_radius_sq = combined_radius * combined_radius

    is_col = False
    normal_unit = ti.Vector([1.0, 0.0, 0.0], dt=gs.ti_float)
    contact_pos = ti.Vector.zero(gs.ti_float, 3)
    penetration = gs.ti_float(0.0)
    if dist_sq < combined_radius_sq:
        is_col = True
        dist = ti.sqrt(dist_sq)

        # Compute contact normal (from capsule to sphere, i.e., B to A)
        if dist > EPS:
            normal_unit = diff / dist
        else:
            # Sphere center is exactly on capsule axis
            # Use any perpendicular direction to the capsule axis
            if ti.abs(capsule_axis[0]) < 0.9:
                normal_unit = i_cross_vec(capsule_axis)
            else:
                normal_unit = j_cross_vec(capsule_axis)

        penetration = combined_radius - dist
        # Contact position at midpoint between surfaces
        contact_pos = sphere_center - (sphere_radius - gs.ti_float(0.5) * penetration) * normal_unit

    return is_col, normal_unit * normal_dir, contact_pos, penetration
