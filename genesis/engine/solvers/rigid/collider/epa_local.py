"""
Thread-local versions of EPA (Expanding Polytope Algorithm) functions.

This module provides versions of EPA functions that accept pos/quat as direct
parameters instead of reading from geoms_state, enabling race-free multi-contact
detection when parallelizing across collision pairs within the same environment.
"""

import gstaichi as ti

import genesis as gs
from genesis.engine.solvers.rigid.collider import epa, gjk_local
from genesis.utils import array_class


@ti.func
def func_epa_support_local(
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
    pos_a: ti.types.vector(3, dtype=gs.ti_float),
    quat_a: ti.types.vector(4, dtype=gs.ti_float),
    pos_b: ti.types.vector(3, dtype=gs.ti_float),
    quat_b: ti.types.vector(4, dtype=gs.ti_float),
    i_b,
    dir,
    dir_norm,
):
    """
    Thread-local version of func_epa_support.

    Find support points on the two objects using [dir] and insert them into the polytope,
    using thread-local pos/quat for both geometries.

    Parameters
    ----------
    dir: gs.ti_vec3
        Vector from [ga] (obj1) to [gb] (obj2).
    """
    d = gs.ti_vec3(1, 0, 0)
    if dir_norm > gjk_info.FLOAT_MIN[None]:
        d = dir / dir_norm

    (
        support_point_obj1,
        support_point_obj2,
        support_point_localpos1,
        support_point_localpos2,
        support_point_id_obj1,
        support_point_id_obj2,
        support_point_minkowski,
    ) = gjk_local.func_support_local(
        geoms_info,
        verts_info,
        static_rigid_sim_config,
        collider_state,
        collider_static_config,
        gjk_state,
        gjk_info,
        support_field_info,
        i_ga,
        i_gb,
        i_b,
        d,
        pos_a,
        quat_a,
        pos_b,
        quat_b,
        False,
    )

    # Insert the support points into the polytope
    v_index = epa.func_epa_insert_vertex_to_polytope(
        gjk_state,
        i_b,
        support_point_obj1,
        support_point_obj2,
        support_point_localpos1,
        support_point_localpos2,
        support_point_id_obj1,
        support_point_id_obj2,
        support_point_minkowski,
    )

    return v_index

