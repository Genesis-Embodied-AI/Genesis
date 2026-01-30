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


@ti.func
def func_gjk_local(
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
    pos_a: ti.types.vector(3, dtype=gs.ti_float),
    quat_a: ti.types.vector(4, dtype=gs.ti_float),
    pos_b: ti.types.vector(3, dtype=gs.ti_float),
    quat_b: ti.types.vector(4, dtype=gs.ti_float),
    shrink_sphere,
):
    """
    Thread-local version of func_gjk.

    GJK algorithm to compute the minimum distance between two convex objects.
    This implementation is based on the MuJoCo implementation.

    Args:
        geoms_info: Geometry information
        verts_info: Vertex information
        static_rigid_sim_config: Static simulation configuration
        collider_state: Collider state (for terrain)
        collider_static_config: Static configuration
        gjk_state: GJK algorithm state (simplex, witness points, etc.)
        gjk_info: GJK algorithm parameters (tolerances, iteration limits)
        support_field_info: Pre-computed support field data
        i_ga: First geometry index
        i_gb: Second geometry index
        i_b: Batch/environment index
        pos_a: First geometry position (thread-local, 28 bytes)
        quat_a: First geometry quaternion (thread-local, 28 bytes)
        pos_b: Second geometry position (thread-local, 28 bytes)
        quat_b: Second geometry quaternion (thread-local, 28 bytes)
        shrink_sphere: If True, use point and line support for sphere/capsule

    Returns:
        distance: Minimum distance between the two geometries
    """
    from genesis.engine.solvers.rigid.collider.gjk import (
        GJK_RETURN_CODE,
        func_gjk_subdistance,
        func_is_discrete_geoms,
        func_is_equal_vec,
        func_simplex_vertex_linear_comb,
    )

    # Simplex index
    n = gs.ti_int(0)
    # Final number of simplex vertices
    nsimplex = gs.ti_int(0)
    # Number of witness points and distance
    nx = gs.ti_int(0)
    dist = gs.ti_float(0.0)
    # Lambda for barycentric coordinates
    _lambda = gs.ti_vec4(1.0, 0.0, 0.0, 0.0)
    # Whether or not we need to compute the exact distance.
    get_dist = shrink_sphere
    # We can use GJK intersection algorithm only for collision detection if we do not have to compute the distance.
    backup_gjk = not get_dist
    # Support vector to compute the next support point.
    support_vector = gs.ti_vec3(0.0, 0.0, 0.0)
    support_vector_norm = gs.ti_float(0.0)
    # Whether or not the main loop finished early because intersection or seperation was detected.
    early_stop = False

    # Set initial guess of support vector using the thread-local positions, which should be a non-zero vector.
    approx_witness_point_obj1 = pos_a
    approx_witness_point_obj2 = pos_b
    support_vector = approx_witness_point_obj1 - approx_witness_point_obj2
    if support_vector.dot(support_vector) < gjk_info.FLOAT_MIN_SQ[None]:
        support_vector = gs.ti_vec3(1.0, 0.0, 0.0)

    # Epsilon for convergence check.
    epsilon = gs.ti_float(0.0)
    if not func_is_discrete_geoms(geoms_info, i_ga, i_gb, i_b):
        # If the objects are smooth, finite convergence is not guaranteed, so we need to set some epsilon
        # to determine convergence.
        epsilon = 0.5 * (gjk_info.tolerance[None] ** 2)

    for i in range(gjk_info.gjk_max_iterations[None]):
        # Compute the current support points
        support_vector_norm = support_vector.norm()
        if support_vector_norm < gjk_info.FLOAT_MIN[None]:
            # If the support vector is too small, it means that origin is located in the Minkowski difference
            # with high probability, so we can stop.
            break

        # Dir to compute the support point (pointing from obj1 to obj2)
        dir = -support_vector * (1.0 / support_vector_norm)

        (
            gjk_state.simplex_vertex.obj1[i_b, n],
            gjk_state.simplex_vertex.obj2[i_b, n],
            gjk_state.simplex_vertex.local_obj1[i_b, n],
            gjk_state.simplex_vertex.local_obj2[i_b, n],
            gjk_state.simplex_vertex.id1[i_b, n],
            gjk_state.simplex_vertex.id2[i_b, n],
            gjk_state.simplex_vertex.mink[i_b, n],
        ) = func_support_local(
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
            dir,
            pos_a,
            quat_a,
            pos_b,
            quat_b,
            shrink_sphere,
        )

        # Early stopping based on Frank-Wolfe duality gap. We need to find the minimum [support_vector_norm],
        # and if we denote it as [x], the problem formulation is: min_x |x|^2.
        # If we denote f(x) = |x|^2, then the Frank-Wolfe duality gap is:
        # |x - x_min|^2 <= < grad f(x), x - s> = < 2x, x - s >,
        # where s is the vertex of the Minkowski difference found by x. Here < 2x, x - s > is guaranteed to be
        # non-negative, and 2 is cancelled out in the definition of the epsilon.
        x_k = support_vector
        s_k = gjk_state.simplex_vertex.mink[i_b, n]
        diff = x_k - s_k
        if diff.dot(x_k) < epsilon:
            # Convergence condition is met, we can stop.
            if i == 0:
                n = 1
            break

        # Check if the objects are separated using support vector
        if not get_dist:
            is_separated = x_k.dot(s_k) > 0.0
            if is_separated:
                nsimplex = 0
                nx = 0
                dist = gjk_info.FLOAT_MAX[None]
                early_stop = True
                break

        if n == 3 and backup_gjk:
            # Tetrahedron is generated, try to detect collision if possible.
            intersect_code = func_gjk_intersect_local(
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
                pos_a,
                quat_a,
                pos_b,
                quat_b,
            )
            if intersect_code == GJK_RETURN_CODE.SEPARATED:
                # No intersection, objects are separated
                nx = 0
                dist = gjk_info.FLOAT_MAX[None]
                nsimplex = 0
                early_stop = True
                break
            elif intersect_code == GJK_RETURN_CODE.INTERSECT:
                # Intersection found
                nx = 0
                dist = 0.0
                nsimplex = 4
                early_stop = True
                break
            else:
                # Since gjk_intersect failed (e.g. origin is on the simplex face), fallback to distance computation
                backup_gjk = False

        # Compute the barycentric coordinates of the closest point to the origin in the simplex
        _lambda = func_gjk_subdistance(gjk_state, gjk_info, i_b, n + 1)

        # Remove vertices from the simplex with zero barycentric coordinates
        n = 0
        for j in ti.static(range(4)):
            if _lambda[j] > 0:
                gjk_state.simplex_vertex.obj1[i_b, n] = gjk_state.simplex_vertex.obj1[i_b, j]
                gjk_state.simplex_vertex.obj2[i_b, n] = gjk_state.simplex_vertex.obj2[i_b, j]
                gjk_state.simplex_vertex.id1[i_b, n] = gjk_state.simplex_vertex.id1[i_b, j]
                gjk_state.simplex_vertex.id2[i_b, n] = gjk_state.simplex_vertex.id2[i_b, j]
                gjk_state.simplex_vertex.mink[i_b, n] = gjk_state.simplex_vertex.mink[i_b, j]
                _lambda[n] = _lambda[j]
                n += 1

        # Should not occur
        if n < 1:
            nsimplex = 0
            nx = 0
            dist = gjk_info.FLOAT_MAX[None]
            early_stop = True
            break

        # Get the next support vector
        next_support_vector = func_simplex_vertex_linear_comb(gjk_state, i_b, 2, 0, 1, 2, 3, _lambda, n)
        if func_is_equal_vec(next_support_vector, support_vector, gjk_info.FLOAT_MIN[None]):
            # If the next support vector is equal to the previous one, we converged to the minimum distance
            break

        support_vector = next_support_vector

        if n == 4:
            # We have a tetrahedron containing the origin, so we can return early. This is because only when
            # the origin is inside the tetrahedron, the barycentric coordinates are all positive. While MuJoCo
            # does not set the [support_vector_norm] to zero as we do, it is necessary, because otherwise the
            # [support_vector_norm] could be non-zero value even if there is contact.
            support_vector_norm = 0
            break

    if not early_stop:
        # If [get_dist] was True and there was no numerical error, [return_code] would be SUCCESS.
        nx = 1
        nsimplex = n
        dist = support_vector_norm

        # Compute witness points
        for i in range(2):
            witness_point = func_simplex_vertex_linear_comb(gjk_state, i_b, i, 0, 1, 2, 3, _lambda, nsimplex)
            if i == 0:
                gjk_state.witness.point_obj1[i_b, 0] = witness_point
            else:
                gjk_state.witness.point_obj2[i_b, 0] = witness_point

    gjk_state.n_witness[i_b] = nx
    gjk_state.distance[i_b] = dist
    gjk_state.nsimplex[i_b] = nsimplex

    return gjk_state.distance[i_b]


@ti.func
def func_gjk_intersect_local(
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
    pos_a: ti.types.vector(3, dtype=gs.ti_float),
    quat_a: ti.types.vector(4, dtype=gs.ti_float),
    pos_b: ti.types.vector(3, dtype=gs.ti_float),
    quat_b: ti.types.vector(4, dtype=gs.ti_float),
):
    """
    Thread-local version of func_gjk_intersect.

    Check if two objects intersect using the GJK algorithm.

    This function refines the simplex until it contains the origin or it is determined
    that the objects are separated. It is used to check if objects intersect, not to
    find the minimum distance between them.

    Args:
        geoms_info: Geometry information
        verts_info: Vertex information
        static_rigid_sim_config: Static simulation configuration
        collider_state: Collider state (for terrain)
        collider_static_config: Static configuration
        gjk_state: GJK algorithm state (simplex, witness points, etc.)
        gjk_info: GJK algorithm parameters (tolerances, iteration limits)
        support_field_info: Pre-computed support field data
        i_ga: First geometry index
        i_gb: Second geometry index
        i_b: Batch/environment index
        pos_a: First geometry position (thread-local, 28 bytes)
        quat_a: First geometry quaternion (thread-local, 28 bytes)
        pos_b: Second geometry position (thread-local, 28 bytes)
        quat_b: Second geometry quaternion (thread-local, 28 bytes)

    Returns:
        flag: GJK_RETURN_CODE indicating result (INTERSECT, SEPARATED, NUM_ERROR)
    """
    from genesis.engine.solvers.rigid.collider.gjk import GJK_RETURN_CODE, func_gjk_triangle_info

    # Copy simplex to temporary storage
    for i in ti.static(range(4)):
        gjk_state.simplex_vertex_intersect.obj1[i_b, i] = gjk_state.simplex_vertex.obj1[i_b, i]
        gjk_state.simplex_vertex_intersect.obj2[i_b, i] = gjk_state.simplex_vertex.obj2[i_b, i]
        gjk_state.simplex_vertex_intersect.id1[i_b, i] = gjk_state.simplex_vertex.id1[i_b, i]
        gjk_state.simplex_vertex_intersect.id2[i_b, i] = gjk_state.simplex_vertex.id2[i_b, i]
        gjk_state.simplex_vertex_intersect.mink[i_b, i] = gjk_state.simplex_vertex.mink[i_b, i]

    # Simplex index
    si = ti.Vector([0, 1, 2, 3], dt=gs.ti_int)

    flag = GJK_RETURN_CODE.NUM_ERROR
    for i in range(gjk_info.gjk_max_iterations[None]):
        # Compute normal and signed distance of the triangle faces of the simplex with respect to the origin.
        # These normals are supposed to point outwards from the simplex.
        # If the origin is inside the plane, [sdist] will be positive.
        is_sdist_all_zero = True
        for j in range(4):
            s0, s1, s2 = si[2], si[1], si[3]
            if j == 1:
                s0, s1, s2 = si[0], si[2], si[3]
            elif j == 2:
                s0, s1, s2 = si[1], si[0], si[3]
            elif j == 3:
                s0, s1, s2 = si[0], si[1], si[2]

            n, s = func_gjk_triangle_info(gjk_state, gjk_info, i_b, s0, s1, s2)

            gjk_state.simplex_buffer_intersect.normal[i_b, j] = n
            gjk_state.simplex_buffer_intersect.sdist[i_b, j] = s

            if ti.abs(s) > gjk_info.FLOAT_MIN[None]:
                is_sdist_all_zero = False

        # If the origin is strictly on any affine hull of the faces, convergence will fail, so ignore this case
        if is_sdist_all_zero:
            break

        # Find the face with the smallest signed distance. We need to find [min_i] for the next iteration.
        min_i = 0
        for j in ti.static(range(1, 4)):
            if gjk_state.simplex_buffer_intersect.sdist[i_b, j] < gjk_state.simplex_buffer_intersect.sdist[i_b, min_i]:
                min_i = j

        min_si = si[min_i]
        min_normal = gjk_state.simplex_buffer_intersect.normal[i_b, min_i]
        min_sdist = gjk_state.simplex_buffer_intersect.sdist[i_b, min_i]

        # If origin is inside the simplex, the signed distances will all be positive
        if min_sdist >= 0:
            # Origin is inside the simplex, so we can stop
            flag = GJK_RETURN_CODE.INTERSECT

            # Copy the temporary simplex to the main simplex
            for j in ti.static(range(4)):
                gjk_state.simplex_vertex.obj1[i_b, j] = gjk_state.simplex_vertex_intersect.obj1[i_b, si[j]]
                gjk_state.simplex_vertex.obj2[i_b, j] = gjk_state.simplex_vertex_intersect.obj2[i_b, si[j]]
                gjk_state.simplex_vertex.id1[i_b, j] = gjk_state.simplex_vertex_intersect.id1[i_b, si[j]]
                gjk_state.simplex_vertex.id2[i_b, j] = gjk_state.simplex_vertex_intersect.id2[i_b, si[j]]
                gjk_state.simplex_vertex.mink[i_b, j] = gjk_state.simplex_vertex_intersect.mink[i_b, si[j]]
            break

        # Replace the worst vertex (which has the smallest signed distance) with new candidate
        (
            gjk_state.simplex_vertex_intersect.obj1[i_b, min_si],
            gjk_state.simplex_vertex_intersect.obj2[i_b, min_si],
            gjk_state.simplex_vertex_intersect.local_obj1[i_b, min_si],
            gjk_state.simplex_vertex_intersect.local_obj2[i_b, min_si],
            gjk_state.simplex_vertex_intersect.id1[i_b, min_si],
            gjk_state.simplex_vertex_intersect.id2[i_b, min_si],
            gjk_state.simplex_vertex_intersect.mink[i_b, min_si],
        ) = func_support_local(
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
            min_normal,
            pos_a,
            quat_a,
            pos_b,
            quat_b,
            False,
        )

        # Check if the origin is strictly outside of the Minkowski difference (which means there is no collision)
        new_minkowski = gjk_state.simplex_vertex_intersect.mink[i_b, min_si]

        is_no_collision = new_minkowski.dot(min_normal) < 0
        if is_no_collision:
            flag = GJK_RETURN_CODE.SEPARATED
            break

        # Swap vertices in the simplex to retain orientation
        m = (min_i + 1) % 4
        n = (min_i + 2) % 4
        swap = si[m]
        si[m] = si[n]
        si[n] = swap

    return flag


@ti.func
def count_support_driver_local(
    geoms_info: array_class.GeomsInfo,
    support_field_info: array_class.SupportFieldInfo,
    d,
    i_g,
    pos: ti.types.vector(3, dtype=gs.ti_float),
    quat: ti.types.vector(4, dtype=gs.ti_float),
    i_b,
):
    """
    Thread-local version of count_support_driver.

    Count the number of possible support points in the given direction,
    using thread-local pos/quat instead of reading from geoms_state.
    """
    geom_type = geoms_info.type[i_g]
    count = 1
    if geom_type == gs.GEOM_TYPE.BOX:
        count = support_field_local._func_count_supports_box_local(geoms_info, d, i_g, pos, quat)
    elif geom_type == gs.GEOM_TYPE.MESH:
        count = support_field_local._func_count_supports_world_local(
            geoms_info,
            support_field_info,
            d,
            i_g,
            pos,
            quat,
            i_b,
        )
    return count


@ti.func
def func_count_support_local(
    geoms_info: array_class.GeomsInfo,
    support_field_info: array_class.SupportFieldInfo,
    i_ga,
    i_gb,
    pos_a: ti.types.vector(3, dtype=gs.ti_float),
    quat_a: ti.types.vector(4, dtype=gs.ti_float),
    pos_b: ti.types.vector(3, dtype=gs.ti_float),
    quat_b: ti.types.vector(4, dtype=gs.ti_float),
    i_b,
    dir,
):
    """
    Thread-local version of func_count_support.

    Count the number of possible pairs of support points on the two objects
    in the given direction, using thread-local pos/quat for both geometries.
    """
    count = 1
    for i in range(2):
        if i == 0:
            count *= count_support_driver_local(
                geoms_info,
                support_field_info,
                dir,
                i_ga,
                pos_a,
                quat_a,
                i_b,
            )
        else:
            count *= count_support_driver_local(
                geoms_info,
                support_field_info,
                -dir,
                i_gb,
                pos_b,
                quat_b,
                i_b,
            )

    return count


@ti.func
def func_safe_gjk_support_local(
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
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
):
    """
    Thread-local version of func_safe_gjk_support.

    Find support points on the two objects using [dir] to use in the [safe_gjk] algorithm.
    Uses thread-local pos/quat for both geometries.

    This is a more robust version of the support function that finds only one pair of support points, because this
    function perturbs the support direction to find the best support points that guarantee non-degenerate simplex
    in the GJK algorithm.

    Parameters:
    ----------
    dir: gs.ti_vec3
        The unit direction in which to find the support points, from [ga] (obj 1) to [gb] (obj 2).
    """
    EPS = rigid_global_info.EPS[None]

    obj1 = gs.ti_vec3(0.0, 0.0, 0.0)
    obj2 = gs.ti_vec3(0.0, 0.0, 0.0)
    local_obj1 = gs.ti_vec3(0.0, 0.0, 0.0)
    local_obj2 = gs.ti_vec3(0.0, 0.0, 0.0)
    id1 = gs.ti_int(-1)
    id2 = gs.ti_int(-1)
    mink = obj1 - obj2

    for i in range(9):
        n_dir = dir
        if i > 0:
            j = i - 1
            n_dir[0] += -(1.0 - 2.0 * (j & 1)) * EPS
            n_dir[1] += -(1.0 - 2.0 * (j & 2)) * EPS
            n_dir[2] += -(1.0 - 2.0 * (j & 4)) * EPS

        # First order normalization based on Taylor series is accurate enough
        n_dir *= 2.0 - n_dir.dot(dir)

        num_supports = func_count_support_local(
            geoms_info,
            support_field_info,
            i_ga,
            i_gb,
            pos_a,
            quat_a,
            pos_b,
            quat_b,
            i_b,
            n_dir,
        )
        if i > 0 and num_supports > 1:
            # If this is a perturbed direction and we have more than one support point, we skip this iteration. If
            # it was the original direction, we continue to find the support points to keep it as the baseline.
            continue

        # Use the current direction to find the support points.
        for j in range(2):
            d = n_dir if j == 0 else -n_dir
            i_g = i_ga if j == 0 else i_gb
            pos = pos_a if j == 0 else pos_b
            quat = quat_a if j == 0 else quat_b

            sp, local_sp, si = support_driver_local(
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
                j,
                False,
            )
            if j == 0:
                obj1 = sp
                local_obj1 = local_sp
                id1 = si
            else:
                obj2 = sp
                local_obj2 = local_sp
                id2 = si

        mink = obj1 - obj2

        if i == 0:
            if num_supports > 1:
                # If there were multiple valid support points, we move on to the next iteration to perturb the
                # direction and find better support points.
                continue
            else:
                break

        # If it was a perturbed direction, check if the support points have been found before.
        if i == 8:
            # If this was the last iteration, we don't check if it has been found before.
            break

        # Check if the updated simplex would be a degenerate simplex.
        if gjk.func_is_new_simplex_vertex_valid(gjk_state, gjk_info, i_b, id1, id2, mink):
            break

    return obj1, obj2, local_obj1, local_obj2, id1, id2, mink


@ti.func
def func_search_valid_simplex_vertex_local(
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
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
):
    """
    Thread-local version of func_search_valid_simplex_vertex.

    Search for a valid simplex vertex (non-duplicate, non-degenerate) in the Minkowski difference,
    using thread-local pos/quat for both geometries.
    """
    obj1 = gs.ti_vec3(0.0, 0.0, 0.0)
    obj2 = gs.ti_vec3(0.0, 0.0, 0.0)
    local_obj1 = gs.ti_vec3(0.0, 0.0, 0.0)
    local_obj2 = gs.ti_vec3(0.0, 0.0, 0.0)
    id1 = -1
    id2 = -1
    minkowski = gs.ti_vec3(0.0, 0.0, 0.0)
    flag = gjk.RETURN_CODE.FAIL

    # If both geometries are discrete, we can use a brute-force search to find a valid simplex vertex.
    if gjk.func_is_discrete_geoms(geoms_info, i_ga, i_gb, i_b):
        geom_nverts = gs.ti_ivec2(0, 0)
        for i in range(2):
            geom_nverts[i] = gjk.func_num_discrete_geom_vertices(geoms_info, i_ga if i == 0 else i_gb, i_b)

        num_cases = geom_nverts[0] * geom_nverts[1]
        for k in range(num_cases):
            m = (k + gjk_state.last_searched_simplex_vertex_id[i_b]) % num_cases
            i = m // geom_nverts[1]
            j = m % geom_nverts[1]

            id1 = geoms_info.vert_start[i_ga] + i
            id2 = geoms_info.vert_start[i_gb] + j
            for p in range(2):
                if p == 0:
                    obj, local_obj = func_get_discrete_geom_vertex_local(geoms_info, verts_info, i_ga, pos_a, quat_a, i)
                    obj1 = obj
                    local_obj1 = local_obj
                else:
                    obj, local_obj = func_get_discrete_geom_vertex_local(geoms_info, verts_info, i_gb, pos_b, quat_b, j)
                    obj2 = obj
                    local_obj2 = local_obj
            minkowski = obj1 - obj2

            # Check if the new vertex is valid
            if gjk.func_is_new_simplex_vertex_valid(gjk_state, gjk_info, i_b, id1, id2, minkowski):
                flag = gjk.RETURN_CODE.SUCCESS
                # Update buffer
                gjk_state.last_searched_simplex_vertex_id[i_b] = (m + 1) % num_cases
                break
    else:
        # Try search direction based on the current simplex.
        nverts = gjk_state.simplex.nverts[i_b]
        if nverts == 3:
            # If we have a triangle, use its normal as the search direction.
            v1 = gjk_state.simplex_vertex.mink[i_b, 0]
            v2 = gjk_state.simplex_vertex.mink[i_b, 1]
            v3 = gjk_state.simplex_vertex.mink[i_b, 2]
            dir = (v3 - v1).cross(v2 - v1).normalized()

            for i in range(2):
                d = dir if i == 0 else -dir
                obj1, obj2, local_obj1, local_obj2, id1, id2, minkowski = func_safe_gjk_support_local(
                    geoms_info,
                    verts_info,
                    rigid_global_info,
                    static_rigid_sim_config,
                    collider_state,
                    collider_static_config,
                    gjk_state,
                    gjk_info,
                    support_field_info,
                    i_ga,
                    i_gb,
                    pos_a,
                    quat_a,
                    pos_b,
                    quat_b,
                    i_b,
                    d,
                )

                # Check if the new vertex is valid
                if gjk.func_is_new_simplex_vertex_valid(gjk_state, gjk_info, i_b, id1, id2, minkowski):
                    flag = gjk.RETURN_CODE.SUCCESS
                    break

    return obj1, obj2, local_obj1, local_obj2, id1, id2, minkowski, flag


@ti.func
def func_safe_gjk_local(
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
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
):
    """
    Thread-local version of func_safe_gjk.

    Safe GJK algorithm to compute the minimum distance between two convex objects,
    using thread-local pos/quat for both geometries.

    This implementation is safer than the one based on the MuJoCo implementation for the following reasons:
    1) It guarantees that the origin is strictly inside the tetrahedron when the intersection is detected.
    2) It guarantees to generate a non-degenerate tetrahedron if there is no numerical error, which is necessary
    for the following EPA algorithm to work correctly.
    3) When computing the face normals on the simplex, it uses a more robust method than using the origin.

    TODO: This implementation could be improved by using shrink_sphere option as the MuJoCo implementation does.
    TODO: This implementation could be further improved by referencing the follow-up work shown below.

    .. seealso::
    Original paper:
    Gilbert, Elmer G., Daniel W. Johnson, and S. Sathiya Keerthi.
    "A fast procedure for computing the distance between complex objects in three-dimensional space."
    IEEE Journal on Robotics and Automation 4.2 (2002): 193-203.

    Further improvements:
    Cameron, Stephen. "Enhancing GJK: Computing minimum and penetration distances between convex polyhedra."
    Proceedings of international conference on robotics and automation. Vol. 4. IEEE, 1997.
    https://www.cs.ox.ac.uk/people/stephen.cameron/distances/gjk2.4/

    Montaut, Louis, et al. "Collision detection accelerated: An optimization perspective."
    https://arxiv.org/abs/2205.09663
    """
    # Compute the initial tetrahedron using two random directions
    init_flag = gjk.RETURN_CODE.SUCCESS
    gjk_state.simplex.nverts[i_b] = 0
    for i in range(4):
        dir = ti.Vector.zero(gs.ti_float, 3)
        dir[2 - i // 2] = 1.0 - 2.0 * (i % 2)

        obj1, obj2, local_obj1, local_obj2, id1, id2, minkowski = func_safe_gjk_support_local(
            geoms_info,
            verts_info,
            rigid_global_info,
            static_rigid_sim_config,
            collider_state,
            collider_static_config,
            gjk_state,
            gjk_info,
            support_field_info,
            i_ga,
            i_gb,
            pos_a,
            quat_a,
            pos_b,
            quat_b,
            i_b,
            dir,
        )

        # Check if the new vertex would make a valid simplex.
        valid = gjk.func_is_new_simplex_vertex_valid(gjk_state, gjk_info, i_b, id1, id2, minkowski)

        # If this is not a valid vertex, fall back to a brute-force routine to find a valid vertex.
        if not valid:
            obj1, obj2, local_obj1, local_obj2, id1, id2, minkowski, init_flag = func_search_valid_simplex_vertex_local(
                geoms_info,
                verts_info,
                rigid_global_info,
                static_rigid_sim_config,
                collider_state,
                collider_static_config,
                gjk_state,
                gjk_info,
                support_field_info,
                i_ga,
                i_gb,
                pos_a,
                quat_a,
                pos_b,
                quat_b,
                i_b,
            )
            # If the brute-force search failed, we cannot proceed with GJK.
            if init_flag == gjk.RETURN_CODE.FAIL:
                break

        gjk_state.simplex_vertex.obj1[i_b, i] = obj1
        gjk_state.simplex_vertex.obj2[i_b, i] = obj2
        gjk_state.simplex_vertex.local_obj1[i_b, i] = local_obj1
        gjk_state.simplex_vertex.local_obj2[i_b, i] = local_obj2
        gjk_state.simplex_vertex.id1[i_b, i] = id1
        gjk_state.simplex_vertex.id2[i_b, i] = id2
        gjk_state.simplex_vertex.mink[i_b, i] = minkowski
        gjk_state.simplex.nverts[i_b] += 1

    gjk_flag = gjk.GJK_RETURN_CODE.SEPARATED
    if init_flag == gjk.RETURN_CODE.SUCCESS:
        # Simplex index
        si = ti.Vector([0, 1, 2, 3], dt=gs.ti_int)

        for i in range(gjk_info.gjk_max_iterations[None]):
            # Compute normal and signed distance of the triangle faces of the simplex with respect to the origin.
            # These normals are supposed to point outwards from the simplex. If the origin is inside the plane,
            # [sdist] will be positive.
            for j in range(4):
                s0, s1, s2, ap = si[2], si[1], si[3], si[0]
                if j == 1:
                    s0, s1, s2, ap = si[0], si[2], si[3], si[1]
                elif j == 2:
                    s0, s1, s2, ap = si[1], si[0], si[3], si[2]
                elif j == 3:
                    s0, s1, s2, ap = si[0], si[1], si[2], si[3]

                n, s = gjk.func_safe_gjk_triangle_info(gjk_state, i_b, s0, s1, s2, ap)

                gjk_state.simplex_buffer.normal[i_b, j] = n
                gjk_state.simplex_buffer.sdist[i_b, j] = s

            # Find the face with the smallest signed distance. We need to find [min_i] for the next iteration.
            min_i = 0
            for j in ti.static(range(1, 4)):
                if gjk_state.simplex_buffer.sdist[i_b, j] < gjk_state.simplex_buffer.sdist[i_b, min_i]:
                    min_i = j

            min_si = si[min_i]
            min_normal = gjk_state.simplex_buffer.normal[i_b, min_i]
            min_sdist = gjk_state.simplex_buffer.sdist[i_b, min_i]

            # If origin is inside the simplex, the signed distances will all be positive
            if min_sdist >= 0:
                # Origin is inside the simplex, so we can stop
                gjk_flag = gjk.GJK_RETURN_CODE.INTERSECT
                break

            # Check if the new vertex would make a valid simplex.
            gjk_state.simplex.nverts[i_b] = 3
            if min_si != 3:
                gjk_state.simplex_vertex.obj1[i_b, min_si] = gjk_state.simplex_vertex.obj1[i_b, 3]
                gjk_state.simplex_vertex.obj2[i_b, min_si] = gjk_state.simplex_vertex.obj2[i_b, 3]
                gjk_state.simplex_vertex.local_obj1[i_b, min_si] = gjk_state.simplex_vertex.local_obj1[i_b, 3]
                gjk_state.simplex_vertex.local_obj2[i_b, min_si] = gjk_state.simplex_vertex.local_obj2[i_b, 3]
                gjk_state.simplex_vertex.id1[i_b, min_si] = gjk_state.simplex_vertex.id1[i_b, 3]
                gjk_state.simplex_vertex.id2[i_b, min_si] = gjk_state.simplex_vertex.id2[i_b, 3]
                gjk_state.simplex_vertex.mink[i_b, min_si] = gjk_state.simplex_vertex.mink[i_b, 3]

            # Find a new candidate vertex to replace the worst vertex (which has the smallest signed distance)
            obj1, obj2, local_obj1, local_obj2, id1, id2, minkowski = func_safe_gjk_support_local(
                geoms_info,
                verts_info,
                rigid_global_info,
                static_rigid_sim_config,
                collider_state,
                collider_static_config,
                gjk_state,
                gjk_info,
                support_field_info,
                i_ga,
                i_gb,
                pos_a,
                quat_a,
                pos_b,
                quat_b,
                i_b,
                min_normal,
            )

            duplicate = gjk.func_is_new_simplex_vertex_duplicate(gjk_state, i_b, id1, id2)
            if duplicate:
                # If the new vertex is a duplicate, it means separation.
                gjk_flag = gjk.GJK_RETURN_CODE.SEPARATED
                break

            degenerate = gjk.func_is_new_simplex_vertex_degenerate(gjk_state, gjk_info, i_b, minkowski)
            if degenerate:
                # If the new vertex is degenerate, we cannot proceed with GJK.
                gjk_flag = gjk.GJK_RETURN_CODE.NUM_ERROR
                break

            # Check if the origin is strictly outside of the Minkowski difference (which means there is no collision)
            is_no_collision = minkowski.dot(min_normal) < 0.0
            if is_no_collision:
                gjk_flag = gjk.GJK_RETURN_CODE.SEPARATED
                break

            gjk_state.simplex_vertex.obj1[i_b, 3] = obj1
            gjk_state.simplex_vertex.obj2[i_b, 3] = obj2
            gjk_state.simplex_vertex.local_obj1[i_b, 3] = local_obj1
            gjk_state.simplex_vertex.local_obj2[i_b, 3] = local_obj2
            gjk_state.simplex_vertex.id1[i_b, 3] = id1
            gjk_state.simplex_vertex.id2[i_b, 3] = id2
            gjk_state.simplex_vertex.mink[i_b, 3] = minkowski
            gjk_state.simplex.nverts[i_b] = 4

    if gjk_flag == gjk.GJK_RETURN_CODE.INTERSECT:
        gjk_state.distance[i_b] = 0.0
    else:
        gjk_flag = gjk.GJK_RETURN_CODE.SEPARATED
        gjk_state.distance[i_b] = gjk_info.FLOAT_MAX[None]

    return gjk_flag
