import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from . import utils
import genesis.utils.array_class as array_class


@qd.func
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
    local_z_unit = qd.Vector([0.0, 0.0, 1.0], dt=gs.ti_float)

    # Get segment axes in world space
    axis_a_unit = gu.transform_vec_by_normalized_quat_fast(local_z_unit, quat_a)
    axis_b_unit = gu.transform_vec_by_normalized_quat_fast(local_z_unit, quat_b)

    # Compute segment endpoints in world space
    P1 = pos_a - halflength_a * axis_a_unit
    P2 = pos_a + halflength_a * axis_a_unit
    Q1 = pos_b - halflength_b * axis_b_unit
    Q2 = pos_b + halflength_b * axis_b_unit

    Pa, Pb = utils.func_closest_points_on_segments(P1, P2, Q1, Q2, EPS)

    # from B to A
    diff = Pa - Pb
    dist_sq = diff.dot(diff)
    combined_radius = radius_a + radius_b
    combined_radius_sq = combined_radius * combined_radius

    is_col = False
    normal_unit = qd.Vector([1.0, 0.0, 0.0], dt=gs.ti_float)
    contact_pos = qd.Vector.zero(gs.ti_float, 3)
    penetration = gs.ti_float(0.0)
    if dist_sq < combined_radius_sq:
        is_col = True
        dist = qd.sqrt(dist_sq)

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
                if qd.abs(axis_a_unit[0]) < 0.9:
                    normal_unit = gu.i_cross_vec(axis_a_unit)
                else:
                    normal_unit = gu.j_cross_vec(axis_a_unit)

        penetration = combined_radius - dist
        # Contact position at midpoint between surfaces
        contact_pos = Pa - radius_a * normal_unit + gs.ti_float(0.5) * penetration * normal_unit

    return is_col, normal_unit, contact_pos, penetration


@qd.func
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
    EPS = rigid_global_info.EPS[None]

    # Ensure sphere is always i_ga and capsule is i_gb
    normal_dir = 1
    sphere_center = ga_pos
    capsule_center = gb_pos
    capsule_q = gb_quat
    if geoms_info.type[i_gb] == gs.GEOM_TYPE.SPHERE:
        i_ga, i_gb = i_gb, i_ga
        sphere_center = gb_pos
        capsule_center = ga_pos
        capsule_q = ga_quat
        normal_dir = -1

    sphere_radius = geoms_info.data[i_ga][0]

    capsule_quat = capsule_q
    capsule_radius = geoms_info.data[i_gb][0]
    capsule_halflength = gs.ti_float(0.5) * geoms_info.data[i_gb][1]

    # Capsule is aligned along local Z-axis
    local_z_unit = qd.Vector([0.0, 0.0, 1.0], dt=gs.ti_float)
    capsule_axis = gu.transform_vec_by_normalized_quat_fast(local_z_unit, capsule_quat)

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
        t = qd.math.clamp(t, 0.0, 1.0)

    closest_point = P1 + t * segment_vec

    # Compute distance from sphere center to closest point
    diff = sphere_center - closest_point
    dist_sq = diff.dot(diff)
    combined_radius = sphere_radius + capsule_radius
    combined_radius_sq = combined_radius * combined_radius

    is_col = False
    normal_unit = qd.Vector([1.0, 0.0, 0.0], dt=gs.ti_float)
    contact_pos = qd.Vector.zero(gs.ti_float, 3)
    penetration = gs.ti_float(0.0)
    if dist_sq < combined_radius_sq:
        is_col = True
        dist = qd.sqrt(dist_sq)

        # Compute contact normal (from capsule to sphere, i.e., B to A)
        if dist > EPS:
            normal_unit = diff / dist
        else:
            # Sphere center is exactly on capsule axis
            # Use any perpendicular direction to the capsule axis
            if qd.abs(capsule_axis[0]) < 0.9:
                normal_unit = gu.i_cross_vec(capsule_axis)
            else:
                normal_unit = gu.j_cross_vec(capsule_axis)

        penetration = combined_radius - dist
        # Contact position at midpoint between surfaces
        contact_pos = sphere_center - (sphere_radius - 0.5 * penetration) * normal_unit

    return is_col, normal_unit * normal_dir, contact_pos, penetration
