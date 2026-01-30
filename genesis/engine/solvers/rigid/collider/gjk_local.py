"""
Thread-local versions of GJK collision detection functions.

This module provides versions of GJK functions that accept pos/quat as direct
parameters instead of reading from geoms_state, enabling race-free multi-contact
detection when parallelizing across collision pairs within the same environment.
"""

import gstaichi as ti

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.solvers.rigid.collider import gjk, support_field, support_field_local
from genesis.utils import array_class


@ti.func
def support_driver_local(
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    support_field_info: array_class.SupportFieldInfo,
    direction,
    i_g,
    pos: ti.types.vector(3, dtype=gs.ti_float),
    quat: ti.types.vector(4, dtype=gs.ti_float),
    i_b,
    i_o,
    shrink_sphere,
):
    """
    Thread-local version of support_driver for GJK.

    Dispatches to the appropriate support function based on geometry type,
    using thread-local pos/quat instead of reading from geoms_state.

    Args:
        geoms_info: Geometry information (types, dimensions, etc.)
        verts_info: Vertex information (for meshes)
        static_rigid_sim_config: Static simulation configuration
        collider_state: Collider state (for terrain prism)
        collider_static_config: Static collider configuration
        gjk_state: GJK algorithm state
        gjk_info: GJK algorithm parameters
        support_field_info: Pre-computed support field data
        direction: Support direction in world frame
        i_g: Geometry index
        pos: Geometry position in world frame (thread-local, 28 bytes)
        quat: Geometry quaternion (thread-local, 28 bytes)
        i_b: Batch/environment index
        i_o: Object index (for GJK state)
        shrink_sphere: If True, use point and line support for sphere and capsule

    Returns:
        v: Support point in world frame
        v_: Support point in local frame
        vid: Vertex ID
    """
    v = ti.Vector.zero(gs.ti_float, 3)
    v_ = ti.Vector.zero(gs.ti_float, 3)
    vid = -1

    geom_type = geoms_info.type[i_g]

    if geom_type == gs.GEOM_TYPE.SPHERE:
        v, v_, vid = support_field_local._func_support_sphere_local(
            geoms_info, direction, i_g, pos, quat, shrink_sphere
        )
    elif geom_type == gs.GEOM_TYPE.ELLIPSOID:
        v = support_field_local._func_support_ellipsoid_local(geoms_info, direction, i_g, pos, quat)
    elif geom_type == gs.GEOM_TYPE.CAPSULE:
        v = support_field_local._func_support_capsule_local(geoms_info, direction, i_g, pos, quat, shrink_sphere)
    elif geom_type == gs.GEOM_TYPE.BOX:
        v, v_, vid = support_field_local._func_support_box_local(geoms_info, direction, i_g, pos, quat)
    elif geom_type == gs.GEOM_TYPE.TERRAIN:
        if ti.static(collider_static_config.has_terrain):
            # Terrain support doesn't depend on geometry pos/quat - uses collider_state.prism
            v, vid = support_field._func_support_prism(collider_state, direction, i_g, i_b)
    elif geom_type == gs.GEOM_TYPE.MESH and static_rigid_sim_config.enable_mujoco_compatibility:
        # MuJoCo-compatible mesh support requires exhaustive vertex search
        v, vid = support_mesh_local(
            geoms_info,
            verts_info,
            gjk_state,
            gjk_info,
            direction,
            i_g,
            pos,
            quat,
            i_b,
            i_o,
        )
    else:
        # Mesh geometries with support field
        v, v_, vid = support_field_local._func_support_world_local(
            geoms_info, support_field_info, direction, i_g, pos, quat
        )

    return v, v_, vid


@ti.func
def support_mesh_local(
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    direction,
    i_g,
    pos: ti.types.vector(3, dtype=gs.ti_float),
    quat: ti.types.vector(4, dtype=gs.ti_float),
    i_b,
    i_o,
):
    """
    Thread-local version of support_mesh.

    Finds the support point on a mesh geometry using exhaustive vertex search.
    This is used for MuJoCo-compatible collision detection.

    Args:
        geoms_info: Geometry information
        verts_info: Vertex information
        gjk_state: GJK algorithm state (for caching previous vertex)
        gjk_info: GJK algorithm parameters
        direction: Support direction in world frame
        i_g: Geometry index
        pos: Geometry position in world frame (thread-local, 28 bytes)
        quat: Geometry quaternion (thread-local, 28 bytes)
        i_b: Batch/environment index
        i_o: Object index (0 for obj1, 1 for obj2)

    Returns:
        v: Support point in world frame
        vid: Vertex ID of the support point
    """
    d_mesh = gu.ti_transform_by_quat(direction, gu.ti_inv_quat(quat))

    # Exhaustively search for the vertex with maximum dot product
    fmax = -gjk_info.FLOAT_MAX[None]
    imax = 0

    vert_start = geoms_info.vert_start[i_g]
    vert_end = geoms_info.vert_end[i_g]

    # Use the previous maximum vertex if it is within the current range
    prev_imax = gjk_state.support_mesh_prev_vertex_id[i_b, i_o]
    if (prev_imax >= vert_start) and (prev_imax < vert_end):
        pos_local = verts_info.init_pos[prev_imax]
        fmax = d_mesh.dot(pos_local)
        imax = prev_imax

    for i in range(vert_start, vert_end):
        pos_local = verts_info.init_pos[i]
        vdot = d_mesh.dot(pos_local)
        if vdot > fmax:
            fmax = vdot
            imax = i

    v = verts_info.init_pos[imax]
    vid = imax

    gjk_state.support_mesh_prev_vertex_id[i_b, i_o] = vid

    v_world = gu.ti_transform_by_trans_quat(v, pos, quat)
    return v_world, vid


@ti.func
def func_support_local(
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    support_field_info: array_class.SupportFieldInfo,
    i_ga,
    i_gb,
    i_b,
    dir,
    pos_a: ti.types.vector(3, dtype=gs.ti_float),
    quat_a: ti.types.vector(4, dtype=gs.ti_float),
    pos_b: ti.types.vector(3, dtype=gs.ti_float),
    quat_b: ti.types.vector(4, dtype=gs.ti_float),
    shrink_sphere,
):
    """
    Thread-local version of func_support.

    Finds support points on two objects using the given direction, computing
    the Minkowski difference support point for GJK collision detection.

    Args:
        geoms_info: Geometry information
        verts_info: Vertex information
        static_rigid_sim_config: Static simulation configuration
        collider_state: Collider state
        collider_static_config: Static collider configuration
        gjk_state: GJK algorithm state
        gjk_info: GJK algorithm parameters
        support_field_info: Pre-computed support field data
        i_ga: First geometry index
        i_gb: Second geometry index
        i_b: Batch/environment index
        dir: Support direction (from obj1 to obj2)
        pos_a: First geometry position (thread-local, 28 bytes)
        quat_a: First geometry quaternion (thread-local, 28 bytes)
        pos_b: Second geometry position (thread-local, 28 bytes)
        quat_b: Second geometry quaternion (thread-local, 28 bytes)
        shrink_sphere: If True, use point and line support for sphere and capsule

    Returns:
        support_point_obj1: Support point on first object
        support_point_obj2: Support point on second object
        support_point_localpos1: Support point in first object's local frame
        support_point_localpos2: Support point in second object's local frame
        support_point_id_obj1: Vertex ID on first object
        support_point_id_obj2: Vertex ID on second object
        support_point_minkowski: Support point on Minkowski difference
    """
    support_point_obj1 = gs.ti_vec3(0, 0, 0)
    support_point_obj2 = gs.ti_vec3(0, 0, 0)
    support_point_localpos1 = gs.ti_vec3(0, 0, 0)
    support_point_localpos2 = gs.ti_vec3(0, 0, 0)
    support_point_id_obj1 = -1
    support_point_id_obj2 = -1

    for i in range(2):
        d = dir if i == 0 else -dir
        i_g = i_ga if i == 0 else i_gb
        pos = pos_a if i == 0 else pos_b
        quat = quat_a if i == 0 else quat_b

        sp, sp_, si = support_driver_local(
            geoms_info,
            verts_info,
            static_rigid_sim_config,
            collider_state,
            collider_static_config,
            gjk_state,
            gjk_info,
            support_field_info,
            d,
            i_g,
            pos,
            quat,
            i_b,
            i,
            shrink_sphere,
        )

        if i == 0:
            support_point_obj1 = sp
            support_point_id_obj1 = si
            support_point_localpos1 = sp_
        else:
            support_point_obj2 = sp
            support_point_id_obj2 = si
            support_point_localpos2 = sp_

    support_point_minkowski = support_point_obj1 - support_point_obj2

    return (
        support_point_obj1,
        support_point_obj2,
        support_point_localpos1,
        support_point_localpos2,
        support_point_id_obj1,
        support_point_id_obj2,
        support_point_minkowski,
    )


@ti.func
def func_get_discrete_geom_vertex_local(
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    i_g,
    pos: ti.types.vector(3, dtype=gs.ti_float),
    quat: ti.types.vector(4, dtype=gs.ti_float),
    i_v,
):
    """
    Thread-local version of func_get_discrete_geom_vertex.

    Gets the discrete vertex of a geometry for the given index, transforming
    it from local to world coordinates using thread-local pos/quat.

    Args:
        geoms_info: Geometry information (types, dimensions, vertex ranges)
        verts_info: Vertex information (initial positions)
        i_g: Geometry index
        pos: Geometry position in world frame (thread-local, 28 bytes)
        quat: Geometry quaternion (thread-local, 28 bytes)
        i_v: Vertex index (relative to geometry)

    Returns:
        v: Vertex position in world frame
        v_: Vertex position in local frame
    """
    geom_type = geoms_info.type[i_g]

    # Get the vertex position in the local frame of the geometry.
    v_ = ti.Vector([0.0, 0.0, 0.0], dt=gs.ti_float)
    if geom_type == gs.GEOM_TYPE.BOX:
        # For the consistency with the [func_support_box] function of [SupportField] class, we handle the box
        # vertex positions in a different way than the general mesh.
        v_ = ti.Vector(
            [
                (1.0 if (i_v & 1 == 1) else -1.0) * geoms_info.data[i_g][0] * 0.5,
                (1.0 if (i_v & 2 == 2) else -1.0) * geoms_info.data[i_g][1] * 0.5,
                (1.0 if (i_v & 4 == 4) else -1.0) * geoms_info.data[i_g][2] * 0.5,
            ],
            dt=gs.ti_float,
        )
    elif geom_type == gs.GEOM_TYPE.MESH:
        vert_start = geoms_info.vert_start[i_g]
        v_ = verts_info.init_pos[vert_start + i_v]

    # Transform the vertex position to the world frame using thread-local pos/quat
    v = gu.ti_transform_by_trans_quat(v_, pos, quat)

    return v, v_
