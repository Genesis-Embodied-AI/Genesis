"""
Thread-local versions of support field functions.

This module provides versions of support field functions that accept pos/quat
as direct parameters instead of reading from geoms_state. This enables
thread-local perturbations without modifying shared global geometry state.

These functions are used when parallelizing collision detection across collision
pairs within the same environment, where each thread needs to work with its own
perturbed geometry state.

## Functions Included

Thread-local versions that accept pos/quat as parameters:
- `_func_support_sphere_local`
- `_func_support_ellipsoid_local`
- `_func_support_capsule_local`
- `_func_support_box_local`
- `_func_support_world_local`
- `_func_count_supports_world_local`
- `_func_count_supports_box_local`

## Functions NOT Included (and why)

### Shared Functions (Imported from support_field.py)

- `_func_support_mesh`
- `_func_count_supports_mesh`

**Why not duplicated:**
These functions operate entirely in mesh-local coordinates and don't access
geoms_state at all. They're pure geometric computations on pre-computed support
fields. Since they're identical in both files, they're imported from support_field.py
to avoid code duplication.

### Thread-Safe Functions (Used Directly)

- `_func_support_prism`

**Why not localized:**
This function is for terrain/prism geometries, which are global/static and not
perturbed during multi-contact. It only reads from `collider_state.prism[i, i_b]`
(indexed per-environment) and doesn't access `geoms_state.pos` or `geoms_state.quat`.
Already thread-safe, so both local and non-local code call it directly from
support_field.py without issues.
"""

import gstaichi as ti

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.engine.solvers.rigid.collider import support_field


@ti.func
def _func_support_world_local(
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
        support_field_info: Pre-computed support field data
        d: Support direction in world frame
        i_g: Geometry index
        pos: Geometry position in world frame (thread-local)
        quat: Geometry quaternion (thread-local)

    Returns:
        v: Support point in world frame
        v_: Support point in local frame
        vid: Vertex ID
    """
    # Transform direction to mesh frame
    d_mesh = gu.ti_transform_by_quat(d, gu.ti_inv_quat(quat))

    # Look up support point in mesh frame (uses pre-computed support field)
    v_, vid = support_field._func_support_mesh(support_field_info, d_mesh, i_g)

    # Transform support point to world frame
    v = gu.ti_transform_by_trans_quat(v_, pos, quat)

    return v, v_, vid


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
def _func_count_supports_world_local(
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
        support_field_info: Pre-computed support field data
        d: Support direction in world frame
        i_g: Geometry index
        quat: Geometry quaternion (thread-local)

    Returns:
        count: Number of support points
    """
    d_mesh = gu.ti_transform_by_quat(d, gu.ti_inv_quat(quat))
    return support_field._func_count_supports_mesh(support_field_info, d_mesh, i_g)


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
