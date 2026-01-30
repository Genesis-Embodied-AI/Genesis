"""
Thread-local versions of support field functions.

This module provides versions of support field functions that accept pos/quat
as direct parameters instead of reading from geoms_state. This enables
thread-local perturbations without modifying shared global geometry state.

These functions are used when parallelizing collision detection across collision
pairs within the same environment, where each thread needs to work with its own
perturbed geometry state.
"""

import math

import gstaichi as ti

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu


@ti.func
def _func_support_sphere_local(
    geoms_info: array_class.GeomsInfo,
    d,
    i_g,
    pos: ti.types.vector(3, dtype=gs.ti_float),
    quat: ti.types.vector(4, dtype=gs.ti_float),
    shrink,
):
    """
    Thread-local version of _func_support_sphere.

    Args:
        geoms_info: Geometry information (radii, etc.)
        d: Support direction in world frame
        i_g: Geometry index
        pos: Geometry position in world frame (thread-local, 28 bytes)
        quat: Geometry quaternion (thread-local, 28 bytes)
        shrink: Whether to shrink sphere to a point

    Returns:
        v: Support point in world frame
        v_: Support point in local frame
        vid: Vertex ID
    """
    sphere_center = pos
    sphere_radius = geoms_info.data[i_g][0]

    # Shrink the sphere to a point
    v = sphere_center
    v_ = ti.Vector.zero(gs.ti_float, 3)
    vid = -1

    if not shrink:
        v += d * sphere_radius

        # Local position of the support point
        local_d = gu.ti_inv_transform_by_quat(d, quat)
        v_ = local_d * sphere_radius

    return v, v_, vid


@ti.func
def _func_support_ellipsoid_local(
    geoms_info: array_class.GeomsInfo,
    d,
    i_g,
    pos: ti.types.vector(3, dtype=gs.ti_float),
    quat: ti.types.vector(4, dtype=gs.ti_float),
):
    """
    Thread-local version of _func_support_ellipsoid.

    Args:
        geoms_info: Geometry information (axis lengths)
        d: Support direction in world frame
        i_g: Geometry index
        pos: Geometry position in world frame (thread-local, 28 bytes)
        quat: Geometry quaternion (thread-local, 28 bytes)

    Returns:
        v: Support point in world frame
    """
    ellipsoid_center = pos
    ellipsoid_scaled_axis = ti.Vector(
        [
            geoms_info.data[i_g][0] ** 2,
            geoms_info.data[i_g][1] ** 2,
            geoms_info.data[i_g][2] ** 2,
        ],
        dt=gs.ti_float,
    )
    ellipsoid_scaled_axis = gu.ti_transform_by_quat(ellipsoid_scaled_axis, quat)
    dist = ellipsoid_scaled_axis / ti.sqrt(d.dot(1.0 / ellipsoid_scaled_axis))
    return ellipsoid_center + d * dist


@ti.func
def _func_support_capsule_local(
    geoms_info: array_class.GeomsInfo,
    d,
    i_g,
    pos: ti.types.vector(3, dtype=gs.ti_float),
    quat: ti.types.vector(4, dtype=gs.ti_float),
    shrink,
):
    """
    Thread-local version of _func_support_capsule.

    Args:
        geoms_info: Geometry information (radius, length)
        d: Support direction in world frame
        i_g: Geometry index
        pos: Geometry position in world frame (thread-local, 28 bytes)
        quat: Geometry quaternion (thread-local, 28 bytes)
        shrink: Whether to shrink capsule to a line

    Returns:
        v: Support point in world frame
    """
    res = gs.ti_vec3(0, 0, 0)
    capsule_center = pos
    capsule_radius = geoms_info.data[i_g][0]
    capsule_halflength = 0.5 * geoms_info.data[i_g][1]

    if shrink:
        local_dir = gu.ti_transform_by_quat(d, gu.ti_inv_quat(quat))
        res[2] = capsule_halflength if local_dir[2] >= 0.0 else -capsule_halflength
        res = gu.ti_transform_by_trans_quat(res, capsule_center, quat)
    else:
        capsule_axis = gu.ti_transform_by_quat(ti.Vector([0.0, 0.0, 1.0], dt=gs.ti_float), quat)
        capsule_endpoint_side = -1.0 if d.dot(capsule_axis) < 0.0 else 1.0
        capsule_endpoint = capsule_center + capsule_halflength * capsule_endpoint_side * capsule_axis
        res = capsule_endpoint + d * capsule_radius
    return res


@ti.func
def _func_support_box_local(
    geoms_info: array_class.GeomsInfo,
    d,
    i_g,
    pos: ti.types.vector(3, dtype=gs.ti_float),
    quat: ti.types.vector(4, dtype=gs.ti_float),
):
    """
    Thread-local version of _func_support_box.

    Args:
        geoms_info: Geometry information (box dimensions)
        d: Support direction in world frame
        i_g: Geometry index
        pos: Geometry position in world frame (thread-local, 28 bytes)
        quat: Geometry quaternion (thread-local, 28 bytes)

    Returns:
        v: Support point in world frame
        v_: Support point in local frame
        vid: Vertex ID
    """
    d_box = gu.ti_inv_transform_by_quat(d, quat)

    v_ = ti.Vector(
        [
            (-1.0 if d_box[0] < 0.0 else 1.0) * geoms_info.data[i_g][0] * 0.5,
            (-1.0 if d_box[1] < 0.0 else 1.0) * geoms_info.data[i_g][1] * 0.5,
            (-1.0 if d_box[2] < 0.0 else 1.0) * geoms_info.data[i_g][2] * 0.5,
        ],
        dt=gs.ti_float,
    )
    vid = (v_[0] > 0.0) * 1 + (v_[1] > 0.0) * 2 + (v_[2] > 0.0) * 4
    vid += geoms_info.vert_start[i_g]
    v = gu.ti_transform_by_trans_quat(v_, pos, quat)
    return v, v_, vid


@ti.func
def _func_support_world_local(
    geoms_info: array_class.GeomsInfo,
    support_field_info: array_class.SupportFieldInfo,
    d,
    i_g,
    pos: ti.types.vector(3, dtype=gs.ti_float),
    quat: ti.types.vector(4, dtype=gs.ti_float),
):
    """
    Thread-local version of _func_support_world.

    This function finds the support point for mesh geometries using a pre-computed
    support field. The support field itself is in local coordinates and doesn't need
    modification; only the transformation to world space uses pos/quat.

    Args:
        geoms_info: Geometry information
        support_field_info: Pre-computed support field data
        d: Support direction in world frame
        i_g: Geometry index
        pos: Geometry position in world frame (thread-local, 28 bytes)
        quat: Geometry quaternion (thread-local, 28 bytes)

    Returns:
        v: Support point in world frame
        v_: Support point in local frame
        vid: Vertex ID
    """
    # Transform direction to mesh frame
    d_mesh = gu.ti_transform_by_quat(d, gu.ti_inv_quat(quat))

    # Look up support point in mesh frame (uses pre-computed support field)
    v_, vid = _func_support_mesh(support_field_info, d_mesh, i_g)

    # Transform support point to world frame
    v = gu.ti_transform_by_trans_quat(v_, pos, quat)

    return v, v_, vid


@ti.func
def _func_support_mesh(support_field_info: array_class.SupportFieldInfo, d_mesh, i_g):
    """
    Support point at mesh frame coordinate.

    This function is unchanged from the original because it operates entirely
    in the mesh's local coordinate system and doesn't depend on world-space
    pos/quat. The support field lookup is purely geometric.

    Args:
        support_field_info: Pre-computed support field data
        d_mesh: Direction in mesh local coordinates
        i_g: Geometry index

    Returns:
        v: Support point in mesh frame
        vid: Vertex ID in the geometry's vertex list
    """
    theta = ti.atan2(d_mesh[1], d_mesh[0])  # [-pi, pi]
    phi = ti.acos(d_mesh[2])  # [0, pi]

    support_res = support_field_info.support_res[None]
    dot_max = gs.ti_float(-1e20)
    v = ti.Vector([0.0, 0.0, 0.0], dt=gs.ti_float)
    vid = 0

    ii = (theta + math.pi) / math.pi / 2 * support_res
    jj = phi / math.pi * support_res

    for i4 in range(4):
        i, j = gs.ti_int(0), gs.ti_int(0)
        if i4 % 2:
            i = gs.ti_int(ti.math.ceil(ii) % support_res)
        else:
            i = gs.ti_int(ti.math.floor(ii) % support_res)

        if i4 // 2 > 0:
            j = gs.ti_int(ti.math.clamp(ti.math.ceil(jj), 0, support_res - 1))
            if j == support_res - 1:
                j = support_res - 2
        else:
            j = gs.ti_int(ti.math.clamp(ti.math.floor(jj), 0, support_res - 1))
            if j == 0:
                j = 1

        support_idx = gs.ti_int(support_field_info.support_cell_start[i_g] + i * support_res + j)
        _vid = support_field_info.support_vid[support_idx]
        pos = support_field_info.support_v[support_idx]
        dot = pos.dot(d_mesh)

        if dot > dot_max:
            v = pos
            dot_max = dot
            vid = _vid

    return v, vid


@ti.func
def _func_count_supports_world_local(
    geoms_info: array_class.GeomsInfo,
    support_field_info: array_class.SupportFieldInfo,
    d,
    i_g,
    quat: ti.types.vector(4, dtype=gs.ti_float),
):
    """
    Thread-local version of _func_count_supports_world.

    Count the number of valid support points for the given world direction.
    Only needs quat since counting doesn't depend on position.

    Args:
        geoms_info: Geometry information
        support_field_info: Pre-computed support field data
        d: Support direction in world frame
        i_g: Geometry index
        quat: Geometry quaternion (thread-local, 28 bytes)

    Returns:
        count: Number of support points
    """
    d_mesh = gu.ti_transform_by_quat(d, gu.ti_inv_quat(quat))
    return _func_count_supports_mesh(support_field_info, d_mesh, i_g)


@ti.func
def _func_count_supports_mesh(
    support_field_info: array_class.SupportFieldInfo,
    d_mesh,
    i_g,
):
    """
    Count the number of valid support points for a mesh in the given direction.

    This function is unchanged from the original because it operates entirely
    in the mesh's local coordinate system.

    Args:
        support_field_info: Pre-computed support field data
        d_mesh: Direction in mesh local coordinates
        i_g: Geometry index

    Returns:
        count: Number of support points with the maximum dot product
    """
    theta = ti.atan2(d_mesh[1], d_mesh[0])  # [-pi, pi]
    phi = ti.acos(d_mesh[2])  # [0, pi]

    support_res = support_field_info.support_res[None]
    dot_max = gs.ti_float(-1e20)

    ii = (theta + math.pi) / math.pi / 2 * support_res
    jj = phi / math.pi * support_res

    count = gs.ti_int(0)
    for i4 in range(4):
        i, j = gs.ti_int(0), gs.ti_int(0)
        if i4 % 2:
            i = gs.ti_int(ti.math.ceil(ii) % support_res)
        else:
            i = gs.ti_int(ti.math.floor(ii) % support_res)

        if i4 // 2 > 0:
            j = gs.ti_int(ti.math.clamp(ti.math.ceil(jj), 0, support_res - 1))
            if j == support_res - 1:
                j = support_res - 2
        else:
            j = gs.ti_int(ti.math.clamp(ti.math.floor(jj), 0, support_res - 1))
            if j == 0:
                j = 1

        support_idx = gs.ti_int(support_field_info.support_cell_start[i_g] + i * support_res + j)
        _vid = support_field_info.support_vid[support_idx]
        pos = support_field_info.support_v[support_idx]
        dot = pos.dot(d_mesh)

        if dot > dot_max:
            count = 1
        elif dot == dot_max:
            count += 1

    return count


@ti.func
def _func_count_supports_box_local(
    d,
    quat: ti.types.vector(4, dtype=gs.ti_float),
):
    """
    Thread-local version of _func_count_supports_box.

    Count the number of valid support points for a box in the given direction.
    Only needs quat since counting doesn't depend on position or geometry-specific data.

    Args:
        d: Support direction in world frame
        quat: Geometry quaternion (thread-local)

    Returns:
        count: Number of support points (1, 2, 4, or 8)
    """
    d_box = gu.ti_inv_transform_by_quat(d, quat)

    return 2 ** (d_box == 0.0).cast(gs.ti_int).sum()
