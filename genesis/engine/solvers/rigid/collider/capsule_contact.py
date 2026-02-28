import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from . import utils
import genesis.utils.array_class as array_class
from .contact import func_add_contact


@qd.func
def func_capsule_capsule_contact(
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
    """Analytical capsule-capsule collision detection with native multi-contact.

    For parallel capsules with axial overlap, generates contacts at both ends
    and center of the overlap region.
    """
    EPS = rigid_global_info.EPS[None]

    pos_a = geoms_state.pos[i_ga, i_b]
    quat_a = geoms_state.quat[i_ga, i_b]
    pos_b = geoms_state.pos[i_gb, i_b]
    quat_b = geoms_state.quat[i_gb, i_b]

    radius_a = geoms_info.data[i_ga][0]
    halflength_a = 0.5 * geoms_info.data[i_ga][1]
    radius_b = geoms_info.data[i_gb][0]
    halflength_b = 0.5 * geoms_info.data[i_gb][1]

    local_z_unit = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)
    axis_a_unit = gu.qd_transform_by_quat_fast(local_z_unit, quat_a)
    axis_b_unit = gu.qd_transform_by_quat_fast(local_z_unit, quat_b)

    P1 = pos_a - halflength_a * axis_a_unit
    P2 = pos_a + halflength_a * axis_a_unit
    Q1 = pos_b - halflength_b * axis_b_unit
    Q2 = pos_b + halflength_b * axis_b_unit

    combined_radius = radius_a + radius_b
    center_diff = pos_a - pos_b

    axis_cross = axis_a_unit.cross(axis_b_unit)
    axis_cross_len_sq = axis_cross.dot(axis_cross)

    if axis_cross_len_sq < EPS * EPS:
        # Parallel axes — check perpendicular distance and axial overlap
        perp_vec = center_diff - center_diff.dot(axis_a_unit) * axis_a_unit
        perp_dist = qd.sqrt(perp_vec.dot(perp_vec))

        b_on_a = -center_diff.dot(axis_a_unit)
        overlap_start = qd.max(-halflength_a, b_on_a - halflength_b)
        overlap_end = qd.min(halflength_a, b_on_a + halflength_b)

        if overlap_end > overlap_start and perp_dist < combined_radius:
            penetration = combined_radius - perp_dist
            normal_unit = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
            if perp_dist > EPS:
                normal_unit = perp_vec / perp_dist
            else:
                if qd.abs(axis_a_unit[0]) < 0.9:
                    normal_unit = gu.qd_i_cross_vec(axis_a_unit)
                else:
                    normal_unit = gu.qd_j_cross_vec(axis_a_unit)

            radial_offset = -(radius_a - 0.5 * penetration) * normal_unit

            overlap_center = 0.5 * (overlap_start + overlap_end)
            contact_pos_center = pos_a + overlap_center * axis_a_unit + radial_offset
            func_add_contact(
                i_ga,
                i_gb,
                normal_unit,
                contact_pos_center,
                penetration,
                i_b,
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
                    contact_pos_lo = pos_a + overlap_start * axis_a_unit + radial_offset
                    func_add_contact(
                        i_ga,
                        i_gb,
                        normal_unit,
                        contact_pos_lo,
                        penetration,
                        i_b,
                        geoms_state,
                        geoms_info,
                        collider_state,
                        collider_info,
                        errno,
                    )
                    contact_pos_hi = pos_a + overlap_end * axis_a_unit + radial_offset
                    func_add_contact(
                        i_ga,
                        i_gb,
                        normal_unit,
                        contact_pos_hi,
                        penetration,
                        i_b,
                        geoms_state,
                        geoms_info,
                        collider_state,
                        collider_info,
                        errno,
                    )

        elif perp_dist < combined_radius:
            # Barrels don't overlap axially but endpoints may interact
            Pa, Pb = utils.func_closest_points_on_segments(P1, P2, Q1, Q2, EPS)
            diff = Pa - Pb
            dist_sq = diff.dot(diff)
            if dist_sq < combined_radius * combined_radius:
                dist = qd.sqrt(dist_sq)
                normal_unit = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
                if dist > EPS:
                    normal_unit = diff / dist
                penetration = combined_radius - dist
                contact_pos = Pa - radius_a * normal_unit + 0.5 * penetration * normal_unit
                func_add_contact(
                    i_ga,
                    i_gb,
                    normal_unit,
                    contact_pos,
                    penetration,
                    i_b,
                    geoms_state,
                    geoms_info,
                    collider_state,
                    collider_info,
                    errno,
                )
    else:
        # Non-parallel axes
        Pa, Pb = utils.func_closest_points_on_segments(P1, P2, Q1, Q2, EPS)
        diff = Pa - Pb
        dist_sq = diff.dot(diff)
        combined_radius_sq = combined_radius * combined_radius

        if dist_sq < combined_radius_sq:
            dist = qd.sqrt(dist_sq)
            normal_unit = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)

            if dist > EPS:
                normal_unit = diff / dist
            else:
                normal_dir = axis_a_unit.cross(axis_b_unit)
                normal_dir_len = normal_dir.dot(normal_dir)
                if normal_dir_len > EPS:
                    normal_unit = normal_dir / normal_dir_len
                else:
                    if qd.abs(axis_a_unit[0]) < 0.9:
                        normal_unit = gu.qd_i_cross_vec(axis_a_unit)
                    else:
                        normal_unit = gu.qd_j_cross_vec(axis_a_unit)

            penetration = combined_radius - dist
            contact_pos = Pa - radius_a * normal_unit + 0.5 * penetration * normal_unit
            func_add_contact(
                i_ga,
                i_gb,
                normal_unit,
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
def func_sphere_capsule_contact(
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
    """Analytical sphere-capsule collision detection.

    Either geom may be the sphere or the capsule; the function normalises
    so that the normal always points from B to A (original ordering).
    """
    EPS = rigid_global_info.EPS[None]

    normal_dir = 1
    sphere_center = geoms_state.pos[i_ga, i_b]
    capsule_center = geoms_state.pos[i_gb, i_b]
    capsule_quat = geoms_state.quat[i_gb, i_b]
    sphere_idx = i_ga
    capsule_idx = i_gb
    if geoms_info.type[i_gb] == gs.GEOM_TYPE.SPHERE:
        sphere_idx = i_gb
        capsule_idx = i_ga
        sphere_center = geoms_state.pos[i_gb, i_b]
        capsule_center = geoms_state.pos[i_ga, i_b]
        capsule_quat = geoms_state.quat[i_ga, i_b]
        normal_dir = -1

    sphere_radius = geoms_info.data[sphere_idx][0]
    capsule_radius = geoms_info.data[capsule_idx][0]
    capsule_halflength = 0.5 * geoms_info.data[capsule_idx][1]

    local_z_unit = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)
    capsule_axis = gu.qd_transform_by_quat_fast(local_z_unit, capsule_quat)

    P1 = capsule_center - capsule_halflength * capsule_axis
    P2 = capsule_center + capsule_halflength * capsule_axis

    segment_vec = P2 - P1
    segment_length_sq = segment_vec.dot(segment_vec)

    t = gs.qd_float(0.5)
    if segment_length_sq > EPS:
        t = (sphere_center - P1).dot(segment_vec) / segment_length_sq
        t = qd.math.clamp(t, 0.0, 1.0)

    closest_point = P1 + t * segment_vec

    diff = sphere_center - closest_point
    dist_sq = diff.dot(diff)
    combined_radius = sphere_radius + capsule_radius
    combined_radius_sq = combined_radius * combined_radius

    if dist_sq < combined_radius_sq:
        dist = qd.sqrt(dist_sq)
        normal_unit = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)

        if dist > EPS:
            normal_unit = diff / dist
        else:
            if qd.abs(capsule_axis[0]) < 0.9:
                normal_unit = gu.qd_i_cross_vec(capsule_axis)
            else:
                normal_unit = gu.qd_j_cross_vec(capsule_axis)

        penetration = combined_radius - dist
        contact_pos = sphere_center - (sphere_radius - 0.5 * penetration) * normal_unit
        normal_unit = normal_unit * gs.qd_float(normal_dir)

        func_add_contact(
            i_ga,
            i_gb,
            normal_unit,
            contact_pos,
            penetration,
            i_b,
            geoms_state,
            geoms_info,
            collider_state,
            collider_info,
            errno,
        )
