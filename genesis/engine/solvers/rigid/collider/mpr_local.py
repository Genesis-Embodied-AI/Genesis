"""
Thread-local versions of MPR collision detection functions.

This module provides versions of MPR functions that accept pos/quat as direct
parameters instead of reading from geoms_state, enabling race-free multi-contact
detection when parallelizing across collision pairs within the same environment.
"""

import gstaichi as ti

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.solvers.rigid.collider import support_field, support_field_local
from genesis.utils import array_class


@ti.func
def support_driver_local(
    geoms_info: array_class.GeomsInfo,
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    direction,
    i_g,
    i_b,
    pos: ti.types.vector(3, dtype=gs.ti_float),
    quat: ti.types.vector(4, dtype=gs.ti_float),
):
    """
    Thread-local version of support_driver for MPR.

    Dispatches to the appropriate support function based on geometry type,
    using thread-local pos/quat instead of reading from geoms_state.

    Args:
        geoms_info: Geometry information (types, dimensions, etc.)
        collider_state: Collider state (for terrain prism)
        collider_static_config: Static configuration
        support_field_info: Pre-computed support field data
        direction: Support direction in world frame
        i_g: Geometry index
        i_b: Batch/environment index (for terrain prism)
        pos: Geometry position in world frame (thread-local, 28 bytes)
        quat: Geometry quaternion (thread-local, 28 bytes)

    Returns:
        v: Support point in world frame
    """
    v = ti.Vector.zero(gs.ti_float, 3)
    geom_type = geoms_info.type[i_g]

    if geom_type == gs.GEOM_TYPE.SPHERE:
        v, v_, vid = support_field_local._func_support_sphere_local(
            geoms_info, direction, i_g, pos, quat, False
        )
    elif geom_type == gs.GEOM_TYPE.ELLIPSOID:
        v = support_field_local._func_support_ellipsoid_local(geoms_info, direction, i_g, pos, quat)
    elif geom_type == gs.GEOM_TYPE.CAPSULE:
        v = support_field_local._func_support_capsule_local(geoms_info, direction, i_g, pos, quat, False)
    elif geom_type == gs.GEOM_TYPE.BOX:
        v, v_, vid = support_field_local._func_support_box_local(geoms_info, direction, i_g, pos, quat)
    elif geom_type == gs.GEOM_TYPE.TERRAIN:
        if ti.static(collider_static_config.has_terrain):
            # Terrain support doesn't depend on geometry pos/quat - uses collider_state.prism
            # Terrain is global and not perturbed, so we use the global state directly
            v, _ = support_field._func_support_prism(collider_state, direction, i_g, i_b)
    else:
        # Mesh geometries
        v, v_, vid = support_field_local._func_support_world_local(
            geoms_info, support_field_info, direction, i_g, pos, quat
        )

    return v


@ti.func
def compute_support_local(
    geoms_info: array_class.GeomsInfo,
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    direction,
    i_ga,
    i_gb,
    i_b,
    pos_a: ti.types.vector(3, dtype=gs.ti_float),
    quat_a: ti.types.vector(4, dtype=gs.ti_float),
    pos_b: ti.types.vector(3, dtype=gs.ti_float),
    quat_b: ti.types.vector(4, dtype=gs.ti_float),
):
    """
    Thread-local version of compute_support.

    Computes the support point on the Minkowski difference of two geometries.
    This is the fundamental operation for MPR collision detection.

    Args:
        geoms_info: Geometry information
        collider_state: Collider state (for terrain)
        collider_static_config: Static configuration
        support_field_info: Pre-computed support field data
        direction: Support direction in world frame
        i_ga: First geometry index
        i_gb: Second geometry index
        i_b: Batch/environment index
        pos_a: First geometry position (thread-local, 28 bytes)
        quat_a: First geometry quaternion (thread-local, 28 bytes)
        pos_b: Second geometry position (thread-local, 28 bytes)
        quat_b: Second geometry quaternion (thread-local, 28 bytes)

    Returns:
        v: Support point on Minkowski difference (v1 - v2)
        v1: Support point on first geometry
        v2: Support point on second geometry
    """
    v1 = support_driver_local(
        geoms_info,
        collider_state,
        collider_static_config,
        support_field_info,
        direction,
        i_ga,
        i_b,
        pos_a,
        quat_a,
    )
    v2 = support_driver_local(
        geoms_info,
        collider_state,
        collider_static_config,
        support_field_info,
        -direction,
        i_gb,
        i_b,
        pos_b,
        quat_b,
    )

    v = v1 - v2
    return v, v1, v2


@ti.func
def func_geom_support_local(
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    direction,
    i_g,
    pos: ti.types.vector(3, dtype=gs.ti_float),
    quat: ti.types.vector(4, dtype=gs.ti_float),
):
    """
    Thread-local version of func_geom_support.

    Finds the support point on a mesh geometry by exhaustively searching
    through all vertices. This is used as a fallback when support fields
    are not available.

    Args:
        geoms_info: Geometry information
        verts_info: Vertex information
        direction: Support direction in world frame
        i_g: Geometry index
        pos: Geometry position in world frame (thread-local, 28 bytes)
        quat: Geometry quaternion (thread-local, 28 bytes)

    Returns:
        v: Support point in world frame
        vid: Vertex ID of the support point
    """
    direction_in_init_frame = gu.ti_inv_transform_by_quat(direction, quat)

    dot_max = gs.ti_float(-1e10)
    v = ti.Vector.zero(gs.ti_float, 3)
    vid = 0

    for i_v in range(geoms_info.vert_start[i_g], geoms_info.vert_end[i_g]):
        pos_local = verts_info.init_pos[i_v]
        dot = pos_local.dot(direction_in_init_frame)
        if dot > dot_max:
            v = pos_local
            dot_max = dot
            vid = i_v

    v_world = gu.ti_transform_by_trans_quat(v, pos, quat)

    return v_world, vid
