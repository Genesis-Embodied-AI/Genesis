"""
Support function utilities for GJK algorithm.

This module contains support point computation functions used by both GJK and EPA algorithms.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class
from . import support_field


@qd.func
def support_mesh(
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    direction,
    i_g,
    pos: qd.types.vector(3, dtype=gs.qd_float),
    quat: qd.types.vector(4, dtype=gs.qd_float),
    i_b,
    i_o,
):
    """
    Find the support point on a mesh in the given direction.
    """
    d_mesh = gu.qd_transform_by_quat(direction, gu.qd_inv_quat(quat))

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

    v_world = gu.qd_transform_by_trans_quat(v, pos, quat)
    return v_world, vid


@qd.func
def support_driver(
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    static_rigid_sim_config: qd.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: qd.template(),
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    support_field_info: array_class.SupportFieldInfo,
    direction,
    i_g,
    pos: qd.types.vector(3, dtype=gs.qd_float),
    quat: qd.types.vector(4, dtype=gs.qd_float),
    i_b,
    i_o,
    shrink_sphere,
):
    """
    @ shrink_sphere: If True, use point and line support for sphere and capsule.
    """
    v = qd.Vector.zero(gs.qd_float, 3)
    v_ = qd.Vector.zero(gs.qd_float, 3)
    vid = -1

    geom_type = geoms_info.type[i_g]
    if geom_type == gs.GEOM_TYPE.SPHERE:
        v, v_, vid = support_field._func_support_sphere(geoms_info, direction, i_g, pos, quat, shrink_sphere)
    elif geom_type == gs.GEOM_TYPE.ELLIPSOID:
        v = support_field._func_support_ellipsoid(geoms_info, direction, i_g, pos, quat)
    elif geom_type == gs.GEOM_TYPE.CAPSULE:
        v = support_field._func_support_capsule(geoms_info, direction, i_g, pos, quat, shrink_sphere)
    elif geom_type == gs.GEOM_TYPE.BOX:
        v, v_, vid = support_field._func_support_box(geoms_info, direction, i_g, pos, quat)
    elif geom_type == gs.GEOM_TYPE.TERRAIN:
        if qd.static(collider_static_config.has_terrain):
            v, vid = support_field._func_support_prism(collider_state, direction, i_b)
    elif geom_type == gs.GEOM_TYPE.MESH and static_rigid_sim_config.enable_mujoco_compatibility:
        # If mujoco-compatible, do exhaustive search for the vertex
        v, vid = support_mesh(geoms_info, verts_info, gjk_state, gjk_info, direction, i_g, pos, quat, i_b, i_o)
    else:
        v, v_, vid = support_field._func_support_world(support_field_info, direction, i_g, pos, quat)
    return v, v_, vid


@qd.func
def func_support(
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    static_rigid_sim_config: qd.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: qd.template(),
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    support_field_info: array_class.SupportFieldInfo,
    i_ga,
    i_gb,
    i_b,
    dir,
    pos_a: qd.types.vector(3, dtype=gs.qd_float),
    quat_a: qd.types.vector(4, dtype=gs.qd_float),
    pos_b: qd.types.vector(3, dtype=gs.qd_float),
    quat_b: qd.types.vector(4, dtype=gs.qd_float),
    shrink_sphere,
):
    """
    Find support points on the two objects using [dir].

    Parameters:
    ----------
    dir: gs.qd_vec3
        The direction in which to find the support points, from [ga] (obj 1) to [gb] (obj 2).
    """
    support_point_obj1 = gs.qd_vec3(0, 0, 0)
    support_point_obj2 = gs.qd_vec3(0, 0, 0)
    support_point_localpos1 = gs.qd_vec3(0, 0, 0)
    support_point_localpos2 = gs.qd_vec3(0, 0, 0)
    support_point_id_obj1 = -1
    support_point_id_obj2 = -1

    for i in range(2):
        d = dir if i == 0 else -dir
        i_g = i_ga if i == 0 else i_gb
        pos = pos_a if i == 0 else pos_b
        quat = quat_a if i == 0 else quat_b

        sp, sp_, si = support_driver(
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
