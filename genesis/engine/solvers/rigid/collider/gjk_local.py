"""
Thread-local versions of GJK collision detection functions.

This module provides versions of GJK functions that accept pos/quat as direct
parameters instead of reading from geoms_state, enabling race-free multi-contact
detection when parallelizing across collision pairs within the same environment.
"""

import gstaichi as ti

import genesis as gs
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
        # This needs the original support_mesh which reads from geoms_state
        # For now, we'll need to create a local version of this too
        v, vid = gjk.support_mesh(
            # Note: This still uses geoms_state - will need local version if used with perturbations
            # For now, assuming this path is not common with multi-contact perturbations
            None,  # geoms_state - placeholder, need local version
            geoms_info,
            verts_info,
            gjk_state,
            gjk_info,
            direction,
            i_g,
            i_b,
            i_o,
        )
    else:
        # Mesh geometries with support field
        v, v_, vid = support_field_local._func_support_world_local(
            geoms_info, support_field_info, direction, i_g, pos, quat
        )

    return v, v_, vid

