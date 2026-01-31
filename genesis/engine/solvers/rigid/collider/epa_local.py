"""
Thread-local versions of EPA (Expanding Polytope Algorithm) functions.

This module provides versions of EPA functions that accept pos/quat as direct
parameters instead of reading from geoms_state, enabling race-free multi-contact
detection when parallelizing across collision pairs within the same environment.
"""

import gstaichi as ti

import genesis as gs
from genesis.engine.solvers.rigid.collider import epa, gjk
from genesis.utils import array_class
import genesis.utils.geom as gu
from genesis.engine.solvers.rigid import collider


@ti.func
def func_epa_local(
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
):
    """
    Thread-local version of func_epa.

    EPA algorithm to find the exact penetration depth and contact normal using the simplex constructed by GJK,
    using thread-local pos/quat for both geometries.

    Thread-safety note: Although this function receives geometry indices `i_ga` and `i_gb`, they are
    only used for thread-safe operations:
    1. Checking geometry types via `geoms_info.type[i_ga/i_gb]` (static metadata, read-only)
    2. Passing to `func_epa_witness` which only reads from pre-computed `gjk_state.polytope_verts`
    Neither operation accesses `geoms_state.pos` or `geoms_state.quat`, so there are no race conditions.

    .. seealso::
    MuJoCo's original implementation:
    https://github.com/google-deepmind/mujoco/blob/7dc7a349c5ba2db2d3f8ab50a367d08e2f1afbbc/src/engine/engine_collision_gjk.c#L1331
    """
    upper = gjk_info.FLOAT_MAX[None]
    upper2 = gjk_info.FLOAT_MAX_SQ[None]
    lower = 0.0
    tolerance = gjk_info.tolerance[None]

    # Index of the nearest face
    nearest_i_f = -1
    prev_nearest_i_f = -1

    discrete = gjk.func_is_discrete_geoms(geoms_info, i_ga, i_gb, i_b)
    if discrete:
        # If the objects are discrete, we do not use tolerance.
        tolerance = gjk_info.FLOAT_MIN[None]

    k_max = gjk_info.epa_max_iterations[None]
    for k in range(k_max):
        prev_nearest_i_f = nearest_i_f

        # Find the polytope face with the smallest distance to the origin
        lower2 = gjk_info.FLOAT_MAX_SQ[None]

        for i in range(gjk_state.polytope.nfaces_map[i_b]):
            i_f = gjk_state.polytope_faces_map[i_b, i]
            face_dist2 = gjk_state.polytope_faces.dist2[i_b, i_f]

            if face_dist2 < lower2:
                lower2 = face_dist2
                nearest_i_f = i_f

        if lower2 > upper2 or nearest_i_f < 0:
            # Invalid face found, stop the algorithm (lower bound of depth is larger than upper bound)
            nearest_i_f = prev_nearest_i_f
            break

        if lower2 <= gjk_info.FLOAT_MIN_SQ[None]:
            # Invalid lower bound (0), stop the algorithm (origin is on the affine hull of face)
            break

        # Find a new support point w from the nearest face's normal
        lower = ti.sqrt(lower2)
        dir = gjk_state.polytope_faces.normal[i_b, nearest_i_f]
        wi = func_epa_support_local(
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
            pos_a,
            quat_a,
            pos_b,
            quat_b,
            i_b,
            dir,
            lower,
        )
        w = gjk_state.polytope_verts.mink[i_b, wi]

        # The upper bound of depth at k-th iteration
        upper_k = w.dot(dir) / lower
        if upper_k < upper:
            upper = upper_k
            upper2 = upper**2

        # If the upper bound and lower bound are close enough, we can stop the algorithm
        if (upper - lower) < tolerance:
            break

        if discrete:
            repeated = False
            for i in range(gjk_state.polytope.nverts[i_b] - 1):
                if (
                    gjk_state.polytope_verts.id1[i_b, i] == gjk_state.polytope_verts.id1[i_b, wi]
                    and gjk_state.polytope_verts.id2[i_b, i] == gjk_state.polytope_verts.id2[i_b, wi]
                ):
                    # The vertex w is already in the polytope,
                    # so we do not need to add it again.
                    repeated = True
                    break
            if repeated:
                break

        gjk_state.polytope.horizon_w[i_b] = w

        # Compute horizon
        horizon_flag = epa.func_epa_horizon(gjk_state, gjk_info, i_b, nearest_i_f)

        if horizon_flag:
            # There was an error in the horizon construction, so the horizon edge is not a closed loop.
            nearest_i_f = -1
            break

        if gjk_state.polytope.horizon_nedges[i_b] < 3:
            # Should not happen, because at least three edges should be in the horizon from one deleted face.
            nearest_i_f = -1
            break

        # Check if the memory space is enough for attaching new faces
        nfaces = gjk_state.polytope.nfaces[i_b]
        nedges = gjk_state.polytope.horizon_nedges[i_b]
        if nfaces + nedges >= gjk_info.polytope_max_faces[None]:
            # If the polytope is full, we cannot insert new faces
            break

        # Attach the new faces
        for i in range(nedges):
            # Face id of the current face to attach
            i_f0 = nfaces + i
            # Face id of the next face to attach
            i_f1 = nfaces + (i + 1) % nedges

            horizon_i_f = gjk_state.polytope_horizon_data.face_idx[i_b, i]
            horizon_i_e = gjk_state.polytope_horizon_data.edge_idx[i_b, i]

            horizon_v1 = gjk_state.polytope_faces.verts_idx[i_b, horizon_i_f][horizon_i_e]
            horizon_v2 = gjk_state.polytope_faces.verts_idx[i_b, horizon_i_f][(horizon_i_e + 1) % 3]

            # Change the adjacent face index of the existing face
            gjk_state.polytope_faces.adj_idx[i_b, horizon_i_f][horizon_i_e] = i_f0

            # Attach the new face.
            # If this if the first face, will be adjacent to the face that will be attached last.
            adj_i_f_0 = i_f0 - 1 if (i > 0) else nfaces + nedges - 1
            adj_i_f_1 = horizon_i_f
            adj_i_f_2 = i_f1

            dist2 = epa.func_attach_face_to_polytope(
                gjk_state,
                gjk_info,
                i_b,
                wi,
                horizon_v2,
                horizon_v1,
                adj_i_f_2,  # Previous face id
                adj_i_f_1,
                adj_i_f_0,  # Next face id
            )
            if dist2 <= 0:
                # Unrecoverable numerical issue
                nearest_i_f = -1
                break

            if (dist2 >= lower2) and (dist2 <= upper2):
                # Store face in the map
                nfaces_map = gjk_state.polytope.nfaces_map[i_b]
                gjk_state.polytope_faces_map[i_b, nfaces_map] = i_f0
                gjk_state.polytope_faces.map_idx[i_b, i_f0] = nfaces_map
                gjk_state.polytope.nfaces_map[i_b] += 1

        # Clear the horizon data for the next iteration
        gjk_state.polytope.horizon_nedges[i_b] = 0

        if (gjk_state.polytope.nfaces_map[i_b] == 0) or (nearest_i_f == -1):
            # No face candidate left
            break

    if nearest_i_f != -1:
        # Nearest face found
        dist2 = gjk_state.polytope_faces.dist2[i_b, nearest_i_f]
        epa.func_epa_witness(gjk_state, i_ga, i_gb, i_b, nearest_i_f)
        gjk_state.n_witness[i_b] = 1
        gjk_state.distance[i_b] = -ti.sqrt(dist2)
    else:
        # No face found, so the objects are not colliding
        gjk_state.n_witness[i_b] = 0
        gjk_state.distance[i_b] = 0

    return nearest_i_f


@ti.func
def func_epa_init_polytope_2d_local(
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
    Thread-local version of func_epa_init_polytope_2d.

    Create the polytope for EPA from a 1-simplex (line segment),
    using thread-local pos/quat for both geometries.

    Thread-safety note: Geometry indices `i_ga` and `i_gb` are only used to pass through to
    `func_epa_support_local`, which uses them solely for read-only metadata access. They do
    not access `geoms_state.pos` or `geoms_state.quat`.

    Returns
    -------
    int
        0 when successful, or a flag indicating an error.
    """
    flag = epa.EPA_POLY_INIT_RETURN_CODE.SUCCESS

    # Get the simplex vertices
    v1 = gjk_state.simplex_vertex.mink[i_b, 0]
    v2 = gjk_state.simplex_vertex.mink[i_b, 1]
    diff = v2 - v1

    # Find the element in [diff] with the smallest magnitude, because it will give us the largest cross product
    min_val = ti.abs(diff[0])
    min_i = 0
    for i in ti.static(range(1, 3)):
        abs_diff_i = ti.abs(diff[i])
        if abs_diff_i < min_val:
            min_val = abs_diff_i
            min_i = i

    # Cross product with the found axis, then rotate it by 120 degrees around the axis [diff] to get three more
    # points spaced 120 degrees apart
    rotmat = gu.ti_rotvec_to_R(diff * ti.math.radians(120.0), rigid_global_info.EPS[None])
    e = gs.ti_vec3(0.0, 0.0, 0.0)
    e[min_i] = 1.0

    d1 = e.cross(diff)
    d2 = rotmat @ d1
    d3 = rotmat @ d2

    # Insert the first two vertices into the polytope
    vi = ti.Vector([0, 0, 0, 0, 0], dt=ti.i32)
    for i in range(2):
        vi[i] = epa.func_epa_insert_vertex_to_polytope(
            gjk_state,
            i_b,
            gjk_state.simplex_vertex.obj1[i_b, i],
            gjk_state.simplex_vertex.obj2[i_b, i],
            gjk_state.simplex_vertex.local_obj1[i_b, i],
            gjk_state.simplex_vertex.local_obj2[i_b, i],
            gjk_state.simplex_vertex.id1[i_b, i],
            gjk_state.simplex_vertex.id2[i_b, i],
            gjk_state.simplex_vertex.mink[i_b, i],
        )

    # Find three more vertices using [d1, d2, d3] as support vectors, and insert them into the polytope
    for i in range(3):
        di = d1
        if i == 1:
            di = d2
        elif i == 2:
            di = d3
        di_norm = di.norm()
        vi[i + 2] = func_epa_support_local(
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
            pos_a,
            quat_a,
            pos_b,
            quat_b,
            i_b,
            di,
            di_norm,
        )

    v3 = gjk_state.polytope_verts.mink[i_b, vi[2]]
    v4 = gjk_state.polytope_verts.mink[i_b, vi[3]]
    v5 = gjk_state.polytope_verts.mink[i_b, vi[4]]

    # Build hexahedron (6 faces) from the five vertices.
    # * This hexahedron would have line [v1, v2] as the central axis, and the other three vertices would be on the
    # sides of the hexahedron, as they are spaced 120 degrees apart.
    # * We already know the face and adjacent face indices in building this.
    # * While building the hexahedron by attaching faces, if the face is very close to the origin, we replace the
    # 1-simplex with the 2-simplex, and restart from it.
    for i in range(6):
        # Vertex indices for the faces in the hexahedron
        i_v1, i_v2, i_v3 = vi[0], vi[2], vi[3]
        # Adjacent face indices for the faces in the hexahedron
        i_a1, i_a2, i_a3 = 1, 3, 2
        if i == 1:
            i_v1, i_v2, i_v3 = vi[0], vi[4], vi[2]
            i_a1, i_a2, i_a3 = 2, 4, 0
        elif i == 2:
            i_v1, i_v2, i_v3 = vi[0], vi[3], vi[4]
            i_a1, i_a2, i_a3 = 0, 5, 1
        elif i == 3:
            i_v1, i_v2, i_v3 = vi[1], vi[3], vi[2]
            i_a1, i_a2, i_a3 = 5, 0, 4
        elif i == 4:
            i_v1, i_v2, i_v3 = vi[1], vi[2], vi[4]
            i_a1, i_a2, i_a3 = 3, 1, 5
        elif i == 5:
            i_v1, i_v2, i_v3 = vi[1], vi[4], vi[3]
            i_a1, i_a2, i_a3 = 4, 2, 3

        if (
            epa.func_attach_face_to_polytope(gjk_state, gjk_info, i_b, i_v1, i_v2, i_v3, i_a1, i_a2, i_a3)
            < gjk_info.FLOAT_MIN_SQ[None]
        ):
            epa.func_replace_simplex_3(gjk_state, i_b, i_v1, i_v2, i_v3)
            flag = epa.EPA_POLY_INIT_RETURN_CODE.P2_FALLBACK3
            break

    if flag == gjk.RETURN_CODE.SUCCESS:
        if not gjk.func_ray_triangle_intersection(v1, v2, v3, v4, v5):
            # The hexahedron should be convex by definition, but somehow if it is not, we return non-convex flag
            flag = epa.EPA_POLY_INIT_RETURN_CODE.P2_NONCONVEX

    if flag == gjk.RETURN_CODE.SUCCESS:
        # Initialize face map
        for i in ti.static(range(6)):
            gjk_state.polytope_faces_map[i_b, i] = i
            gjk_state.polytope_faces.map_idx[i_b, i] = i
        gjk_state.polytope.nfaces_map[i_b] = 6

    return flag


@ti.func
def func_epa_init_polytope_3d_local(
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
):
    """
    Thread-local version of func_epa_init_polytope_3d.

    Create the polytope for EPA from a 2-simplex (triangle),
    using thread-local pos/quat for both geometries.

    Thread-safety note: Geometry indices `i_ga` and `i_gb` are only used to pass through to
    `func_epa_support_local`, which uses them solely for read-only metadata access. They do
    not access `geoms_state.pos` or `geoms_state.quat`.

    Returns
    -------
    int
        0 when successful, or a flag indicating an error.
    """
    flag = epa.EPA_POLY_INIT_RETURN_CODE.SUCCESS

    # Get the simplex vertices
    v1 = gjk_state.simplex_vertex.mink[i_b, 0]
    v2 = gjk_state.simplex_vertex.mink[i_b, 1]
    v3 = gjk_state.simplex_vertex.mink[i_b, 2]

    # Get normal; if it is zero, we cannot proceed
    n = (v2 - v1).cross(v3 - v1)
    n_norm = n.norm()
    if n_norm < gjk_info.FLOAT_MIN[None]:
        flag = epa.EPA_POLY_INIT_RETURN_CODE.P3_BAD_NORMAL
    n_neg = -n

    # Save vertices in the polytope
    vi = ti.Vector([0, 0, 0, 0, 0], dt=ti.i32)
    for i in range(3):
        vi[i] = epa.func_epa_insert_vertex_to_polytope(
            gjk_state,
            i_b,
            gjk_state.simplex_vertex.obj1[i_b, i],
            gjk_state.simplex_vertex.obj2[i_b, i],
            gjk_state.simplex_vertex.local_obj1[i_b, i],
            gjk_state.simplex_vertex.local_obj2[i_b, i],
            gjk_state.simplex_vertex.id1[i_b, i],
            gjk_state.simplex_vertex.id2[i_b, i],
            gjk_state.simplex_vertex.mink[i_b, i],
        )

    # Find the fourth and fifth vertices using the normal
    # as the support vector. We form a hexahedron (6 faces)
    # with these five vertices.
    for i in range(2):
        dir = n if i == 0 else n_neg
        vi[i + 3] = func_epa_support_local(
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
            pos_a,
            quat_a,
            pos_b,
            quat_b,
            i_b,
            dir,
            n_norm,
        )
    v4 = gjk_state.polytope_verts.mink[i_b, vi[3]]
    v5 = gjk_state.polytope_verts.mink[i_b, vi[4]]

    # Check if v4 or v5 located inside the triangle.
    # If so, we do not proceed anymore.
    for i in range(2):
        v = v4 if i == 0 else v5
        if gjk.func_point_triangle_intersection(gjk_info, v, v1, v2, v3):
            flag = (
                epa.EPA_POLY_INIT_RETURN_CODE.P3_INVALID_V4 if i == 0 else epa.EPA_POLY_INIT_RETURN_CODE.P3_INVALID_V5
            )
            break

    if flag == epa.EPA_POLY_INIT_RETURN_CODE.SUCCESS:
        # If origin does not lie inside the triangle, we need to
        # check if the hexahedron contains the origin.

        tets_has_origin = gs.ti_ivec2(0, 0)
        for i in range(2):
            v = v4 if i == 0 else v5
            tets_has_origin[i] = (
                1 if gjk.func_origin_tetra_intersection(v1, v2, v3, v) == gjk.RETURN_CODE.SUCCESS else 0
            )

        # @TODO: It's possible for GJK to return a triangle with origin not contained in it but within tolerance
        # from it. In that case, the hexahedron could possibly be constructed that does ont contain the origin, but
        # there is penetration depth.
        if (
            gjk_state.simplex.dist[i_b] > 10 * gjk_info.FLOAT_MIN[None]
            and (not tets_has_origin[0])
            and (not tets_has_origin[1])
        ):
            flag = epa.EPA_POLY_INIT_RETURN_CODE.P3_MISSING_ORIGIN
        else:
            # Build hexahedron (6 faces) from the five vertices.
            for i in range(6):
                # Vertex indices for the faces in the hexahedron
                i_v1, i_v2, i_v3 = vi[3], vi[0], vi[1]
                # Adjacent face indices for the faces in the hexahedron
                i_a1, i_a2, i_a3 = 1, 3, 2
                if i == 1:
                    i_v1, i_v2, i_v3 = vi[3], vi[2], vi[0]
                    i_a1, i_a2, i_a3 = 2, 4, 0
                elif i == 2:
                    i_v1, i_v2, i_v3 = vi[3], vi[1], vi[2]
                    i_a1, i_a2, i_a3 = 0, 5, 1
                elif i == 3:
                    i_v1, i_v2, i_v3 = vi[4], vi[1], vi[0]
                    i_a1, i_a2, i_a3 = 5, 0, 4
                elif i == 4:
                    i_v1, i_v2, i_v3 = vi[4], vi[0], vi[2]
                    i_a1, i_a2, i_a3 = 3, 1, 5
                elif i == 5:
                    i_v1, i_v2, i_v3 = vi[4], vi[2], vi[1]
                    i_a1, i_a2, i_a3 = 4, 2, 3

                dist2 = epa.func_attach_face_to_polytope(gjk_state, gjk_info, i_b, i_v1, i_v2, i_v3, i_a1, i_a2, i_a3)
                if dist2 < gjk_info.FLOAT_MIN_SQ[None]:
                    flag = epa.EPA_POLY_INIT_RETURN_CODE.P3_ORIGIN_ON_FACE
                    break

    if flag == epa.EPA_POLY_INIT_RETURN_CODE.SUCCESS:
        # Initialize face map
        for i in ti.static(range(6)):
            gjk_state.polytope_faces_map[i_b, i] = i
            gjk_state.polytope_faces.map_idx[i_b, i] = i
        gjk_state.polytope.nfaces_map[i_b] = 6

    return flag


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

    Thread-safety note: Geometry indices `i_ga` and `i_gb` are only used to pass through to
    `func_support_local`, which uses them solely for read-only metadata access (geometry types,
    vertex ranges, etc.). They do not access `geoms_state.pos` or `geoms_state.quat`.

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
    ) = collider.gjk_local.func_support_local(
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
