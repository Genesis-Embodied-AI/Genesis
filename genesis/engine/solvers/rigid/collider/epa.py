"""
Expanding Polytope Algorithm (EPA) for penetration depth computation.

This module contains the EPA algorithm implementation for computing exact
penetration depth and contact normals for intersecting convex objects.
Includes both standard and numerically robust ("safe") variants.
"""

import gstaichi as ti

import genesis as gs
import genesis.utils.array_class as array_class
from . import gjk

# Note: Previously imported specific functions from gjk, but that caused circular import
# when gjk -> gjk_local -> epa_local -> epa -> gjk
# Now we import the module and access functions via gjk.function_name


@ti.func
def func_epa_witness(
    gjk_state: array_class.GJKState,
    i_ga,
    i_gb,
    i_b,
    i_f,
):
    """
    Compute the witness points from the geometries for the face i_f of the polytope.
    """
    # Find the affine coordinates of the origin's projection on the face i_f
    face_iv1 = gjk_state.polytope_faces.verts_idx[i_b, i_f][0]
    face_iv2 = gjk_state.polytope_faces.verts_idx[i_b, i_f][1]
    face_iv3 = gjk_state.polytope_faces.verts_idx[i_b, i_f][2]
    face_v1 = gjk_state.polytope_verts.mink[i_b, face_iv1]
    face_v2 = gjk_state.polytope_verts.mink[i_b, face_iv2]
    face_v3 = gjk_state.polytope_verts.mink[i_b, face_iv3]
    face_normal = gjk_state.polytope_faces.normal[i_b, i_f]

    _lambda = gjk.func_triangle_affine_coords(
        face_normal,
        face_v1,
        face_v2,
        face_v3,
    )

    # Point on geom 1
    v1 = gjk_state.polytope_verts.obj1[i_b, face_iv1]
    v2 = gjk_state.polytope_verts.obj1[i_b, face_iv2]
    v3 = gjk_state.polytope_verts.obj1[i_b, face_iv3]
    witness1 = v1 * _lambda[0] + v2 * _lambda[1] + v3 * _lambda[2]

    # Point on geom 2
    v1 = gjk_state.polytope_verts.obj2[i_b, face_iv1]
    v2 = gjk_state.polytope_verts.obj2[i_b, face_iv2]
    v3 = gjk_state.polytope_verts.obj2[i_b, face_iv3]
    witness2 = v1 * _lambda[0] + v2 * _lambda[1] + v3 * _lambda[2]

    gjk_state.witness.point_obj1[i_b, 0] = witness1
    gjk_state.witness.point_obj2[i_b, 0] = witness2


@ti.func
def func_epa_horizon(
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    i_b,
    nearest_i_f,
):
    """
    Compute the horizon, which represents the area of the polytope that is visible from the vertex w, and thus
    should be deleted for the expansion of the polytope.
    """
    w = gjk_state.polytope.horizon_w[i_b]

    # Initialize the stack by inserting the nearest face
    gjk_state.polytope_horizon_stack.face_idx[i_b, 0] = nearest_i_f
    gjk_state.polytope_horizon_stack.edge_idx[i_b, 0] = 0
    top = 1
    is_first = True

    flag = gjk.RETURN_CODE.SUCCESS
    while top > 0:
        # Pop the top face from the stack
        i_f = gjk_state.polytope_horizon_stack.face_idx[i_b, top - 1]
        i_e = gjk_state.polytope_horizon_stack.edge_idx[i_b, top - 1]
        i_v = gjk_state.polytope_faces.verts_idx[i_b, i_f][0]
        v = gjk_state.polytope_verts.mink[i_b, i_v]
        top -= 1

        # If the face is already deleted, skip it
        is_deleted = gjk_state.polytope_faces.map_idx[i_b, i_f] == -2
        if (not is_first) and (is_deleted):
            continue

        # Check visibility of the face. Two requirements for the face to be visible:
        # 1. The face normal should point towards the vertex w
        # 2. The vertex w should be on the other side of the face to the origin
        is_visible = gjk_state.polytope_faces.normal[i_b, i_f].dot(w - v) > gjk_info.FLOAT_MIN[None]

        # The first face is always considered visible.
        if is_visible or is_first:
            # If visible, delete the face from the polytope
            func_delete_face_from_polytope(gjk_state, i_b, i_f)

            # Add the other two or three edges of the face to the stack.
            # The order is important to form a closed loop.
            for k in range(0 if is_first else 1, 3):
                i_e2 = (i_e + k) % 3
                adj_face_idx = gjk_state.polytope_faces.adj_idx[i_b, i_f][i_e2]
                adj_face_is_deleted = gjk_state.polytope_faces.map_idx[i_b, adj_face_idx] == -2
                if not adj_face_is_deleted:
                    # Get the related edge id from the adjacent face. Since adjacent faces have different
                    # orientations, we need to use the ending vertex of the edge.
                    start_vert_idx = gjk_state.polytope_faces.verts_idx[i_b, i_f][(i_e2 + 1) % 3]
                    adj_edge_idx = func_get_edge_idx(gjk_state, i_b, adj_face_idx, start_vert_idx)

                    gjk_state.polytope_horizon_stack.face_idx[i_b, top] = adj_face_idx
                    gjk_state.polytope_horizon_stack.edge_idx[i_b, top] = adj_edge_idx
                    top += 1
        else:
            # If not visible, add the edge to the horizon.
            flag = func_add_edge_to_horizon(gjk_state, i_b, i_f, i_e)
            if flag:
                # If the edges do not form a closed loop, there is an error in the algorithm.
                break

        is_first = False

    return flag


@ti.func
def func_add_edge_to_horizon(
    gjk_state: array_class.GJKState,
    i_b,
    i_f,
    i_e,
):
    """
    Add an edge to the horizon data structure.
    """
    horizon_nedges = gjk_state.polytope.horizon_nedges[i_b]
    gjk_state.polytope_horizon_data.edge_idx[i_b, horizon_nedges] = i_e
    gjk_state.polytope_horizon_data.face_idx[i_b, horizon_nedges] = i_f
    gjk_state.polytope.horizon_nedges[i_b] += 1

    return gjk.RETURN_CODE.SUCCESS


@ti.func
def func_get_edge_idx(
    gjk_state: array_class.GJKState,
    i_b,
    i_f,
    i_v,
):
    """
    Get the edge index from the face, starting from the vertex i_v.

    If the face is comprised of [v1, v2, v3], the edges are: [v1, v2], [v2, v3], [v3, v1].
    Therefore, if i_v was v1, the edge index is 0, and if i_v was v2, the edge index is 1.
    """
    verts = gjk_state.polytope_faces.verts_idx[i_b, i_f]
    ret = gs.ti_int(2)
    if verts[0] == i_v:
        ret = 0
    elif verts[1] == i_v:
        ret = 1
    return ret


@ti.func
def func_delete_face_from_polytope(
    gjk_state: array_class.GJKState,
    i_b,
    i_f,
):
    """
    Delete the face from the polytope.
    """
    face_map_idx = gjk_state.polytope_faces.map_idx[i_b, i_f]
    if face_map_idx >= 0:
        last_face_idx = gjk_state.polytope_faces_map[i_b, gjk_state.polytope.nfaces_map[i_b] - 1]
        # Make the map to point to the last face
        gjk_state.polytope_faces_map[i_b, face_map_idx] = last_face_idx
        # Change map index of the last face
        gjk_state.polytope_faces.map_idx[i_b, last_face_idx] = face_map_idx

        # Decrease the number of faces in the polytope
        gjk_state.polytope.nfaces_map[i_b] -= 1

    # Mark the face as deleted
    gjk_state.polytope_faces.map_idx[i_b, i_f] = -2


@ti.func
def func_epa_insert_vertex_to_polytope(
    gjk_state: array_class.GJKState,
    i_b: int,
    obj1_point,
    obj2_point,
    obj1_localpos,
    obj2_localpos,
    obj1_id: int,
    obj2_id: int,
    minkowski_point,
):
    """
    Copy vertex information into the polytope.
    """
    n = gjk_state.polytope.nverts[i_b]
    gjk_state.polytope_verts.obj1[i_b, n] = obj1_point
    gjk_state.polytope_verts.obj2[i_b, n] = obj2_point
    gjk_state.polytope_verts.local_obj1[i_b, n] = obj1_localpos
    gjk_state.polytope_verts.local_obj2[i_b, n] = obj2_localpos
    gjk_state.polytope_verts.id1[i_b, n] = obj1_id
    gjk_state.polytope_verts.id2[i_b, n] = obj2_id
    gjk_state.polytope_verts.mink[i_b, n] = minkowski_point
    gjk_state.polytope.nverts[i_b] += 1
    return n


@ti.func
def func_epa_init_polytope_4d(
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    i_ga,
    i_gb,
    i_b,
):
    """
    Create the polytope for EPA from a 3-simplex (tetrahedron).

    Thread-safety note: This function is already thread-safe and does NOT require a local version.
    It only manipulates existing simplex vertices that were previously computed by GJK, without
    reading `geoms_state.pos` or `geoms_state.quat`, and without calling any support functions.
    All geometry state accesses happen through the pre-computed simplex data in `gjk_state`.

    Returns
    -------
    int
        0 when successful, or a flag indicating an error.
    """
    flag = gjk.EPA_POLY_INIT_RETURN_CODE.SUCCESS

    # Insert simplex vertices into the polytope
    vi = ti.Vector([0, 0, 0, 0], dt=ti.i32)
    for i in range(4):
        vi[i] = func_epa_insert_vertex_to_polytope(
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

    # If origin is on any face of the tetrahedron, replace the simplex with a 2-simplex (triangle)
    for i in range(4):
        # Vertex indices for the faces in the hexahedron
        v1, v2, v3 = vi[0], vi[1], vi[2]
        # Adjacent face indices for the faces in the hexahedron
        a1, a2, a3 = 1, 3, 2
        if i == 1:
            v1, v2, v3 = vi[0], vi[3], vi[1]
            a1, a2, a3 = 2, 3, 0
        elif i == 2:
            v1, v2, v3 = vi[0], vi[2], vi[3]
            a1, a2, a3 = 0, 3, 1
        elif i == 3:
            v1, v2, v3 = vi[3], vi[2], vi[1]
            a1, a2, a3 = 2, 0, 1

        dist2 = func_attach_face_to_polytope(gjk_state, gjk_info, i_b, v1, v2, v3, a1, a2, a3)

        if dist2 < gjk_info.FLOAT_MIN_SQ[None]:
            func_replace_simplex_3(gjk_state, i_b, v1, v2, v3)
            flag = gjk.EPA_POLY_INIT_RETURN_CODE.P4_FALLBACK3
            break

    if flag == gjk.EPA_POLY_INIT_RETURN_CODE.SUCCESS:
        # If the tetrahedron does not contain the origin, we do not proceed anymore.
        if (
            gjk.func_origin_tetra_intersection(
                gjk_state.polytope_verts.mink[i_b, vi[0]],
                gjk_state.polytope_verts.mink[i_b, vi[1]],
                gjk_state.polytope_verts.mink[i_b, vi[2]],
                gjk_state.polytope_verts.mink[i_b, vi[3]],
            )
            == gjk.RETURN_CODE.FAIL
        ):
            flag = gjk.EPA_POLY_INIT_RETURN_CODE.P4_MISSING_ORIGIN

    if flag == gjk.EPA_POLY_INIT_RETURN_CODE.SUCCESS:
        # Initialize face map
        for i in ti.static(range(4)):
            gjk_state.polytope_faces_map[i_b, i] = i
            gjk_state.polytope_faces.map_idx[i_b, i] = i
        gjk_state.polytope.nfaces_map[i_b] = 4

    return flag


@ti.func
def func_attach_face_to_polytope(
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    i_b,
    i_v1,
    i_v2,
    i_v3,
    i_a1,
    i_a2,
    i_a3,
):
    """
    Attach a face to the polytope.

    [i_v1, i_v2, i_v3] are the vertices of the face, [i_a1, i_a2, i_a3] are the adjacent faces.

    Returns
    -------
    float
        Squared distance of the face to the origin.
    """
    dist2 = 0.0

    n = gjk_state.polytope.nfaces[i_b]
    gjk_state.polytope_faces.verts_idx[i_b, n][0] = i_v1
    gjk_state.polytope_faces.verts_idx[i_b, n][1] = i_v2
    gjk_state.polytope_faces.verts_idx[i_b, n][2] = i_v3
    gjk_state.polytope_faces.adj_idx[i_b, n][0] = i_a1
    gjk_state.polytope_faces.adj_idx[i_b, n][1] = i_a2
    gjk_state.polytope_faces.adj_idx[i_b, n][2] = i_a3
    gjk_state.polytope.nfaces[i_b] += 1

    # Compute the squared distance of the face to the origin
    gjk_state.polytope_faces.normal[i_b, n], ret = gjk.func_project_origin_to_plane(
        gjk_info,
        gjk_state.polytope_verts.mink[i_b, i_v3],
        gjk_state.polytope_verts.mink[i_b, i_v2],
        gjk_state.polytope_verts.mink[i_b, i_v1],
    )
    if ret == gjk.RETURN_CODE.SUCCESS:
        normal = gjk_state.polytope_faces.normal[i_b, n]
        gjk_state.polytope_faces.dist2[i_b, n] = normal.dot(normal)
        gjk_state.polytope_faces.map_idx[i_b, n] = -1  # No map index yet
        dist2 = gjk_state.polytope_faces.dist2[i_b, n]

    return dist2


@ti.func
def func_replace_simplex_3(
    gjk_state: array_class.GJKState,
    i_b,
    i_v1,
    i_v2,
    i_v3,
):
    """
    Replace the simplex with a 2-simplex (triangle) from polytope vertices.

    Parameters
    ----------
    i_v1, i_v2, i_v3: int
        Indices of the vertices in the polytope that will be used to form the triangle.
    """
    gjk_state.simplex.nverts[i_b] = 3
    for i in ti.static(range(3)):
        i_v = i_v1
        if i == 1:
            i_v = i_v2
        elif i == 2:
            i_v = i_v3
        gjk_state.simplex_vertex.obj1[i_b, i] = gjk_state.polytope_verts.obj1[i_b, i_v]
        gjk_state.simplex_vertex.obj2[i_b, i] = gjk_state.polytope_verts.obj2[i_b, i_v]
        gjk_state.simplex_vertex.id1[i_b, i] = gjk_state.polytope_verts.id1[i_b, i_v]
        gjk_state.simplex_vertex.id2[i_b, i] = gjk_state.polytope_verts.id2[i_b, i_v]
        gjk_state.simplex_vertex.mink[i_b, i] = gjk_state.polytope_verts.mink[i_b, i_v]

    # Reset polytope
    gjk_state.polytope.nverts[i_b] = 0
    gjk_state.polytope.nfaces[i_b] = 0
    gjk_state.polytope.nfaces_map[i_b] = 0


@ti.func
def func_safe_epa_witness(
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    i_ga,
    i_gb,
    i_b,
    i_f,
):
    """
    Compute the witness points from the geometries for the face i_f of the polytope.
    """
    flag = gjk.RETURN_CODE.SUCCESS

    # Find the affine coordinates of the origin's projection on the face i_f
    face_iv1 = gjk_state.polytope_faces.verts_idx[i_b, i_f][0]
    face_iv2 = gjk_state.polytope_faces.verts_idx[i_b, i_f][1]
    face_iv3 = gjk_state.polytope_faces.verts_idx[i_b, i_f][2]
    face_v1 = gjk_state.polytope_verts.mink[i_b, face_iv1]
    face_v2 = gjk_state.polytope_verts.mink[i_b, face_iv2]
    face_v3 = gjk_state.polytope_verts.mink[i_b, face_iv3]

    # Project origin onto the face plane to get the barycentric coordinates
    proj_o, _ = gjk.func_project_origin_to_plane(gjk_info, face_v1, face_v2, face_v3)
    _lambda = gjk.func_triangle_affine_coords(proj_o, face_v1, face_v2, face_v3)

    # Check validity of affine coordinates through reprojection
    v1 = gjk_state.polytope_verts.mink[i_b, face_iv1]
    v2 = gjk_state.polytope_verts.mink[i_b, face_iv2]
    v3 = gjk_state.polytope_verts.mink[i_b, face_iv3]

    proj_o_lambda = v1 * _lambda[0] + v2 * _lambda[1] + v3 * _lambda[2]
    reprojection_error = (proj_o - proj_o_lambda).norm()

    # Take into account the face magnitude, as the error is relative to the face size.
    max_edge_len_inv = ti.rsqrt(
        max((v1 - v2).norm_sqr(), (v2 - v3).norm_sqr(), (v3 - v1).norm_sqr(), gjk_info.FLOAT_MIN_SQ[None])
    )
    rel_reprojection_error = reprojection_error * max_edge_len_inv
    if rel_reprojection_error > gjk_info.polytope_max_reprojection_error[None]:
        flag = gjk.RETURN_CODE.FAIL

    if flag == gjk.RETURN_CODE.SUCCESS:
        # Point on geom 1
        v1 = gjk_state.polytope_verts.obj1[i_b, face_iv1]
        v2 = gjk_state.polytope_verts.obj1[i_b, face_iv2]
        v3 = gjk_state.polytope_verts.obj1[i_b, face_iv3]
        witness1 = v1 * _lambda[0] + v2 * _lambda[1] + v3 * _lambda[2]

        # Point on geom 2
        v1 = gjk_state.polytope_verts.obj2[i_b, face_iv1]
        v2 = gjk_state.polytope_verts.obj2[i_b, face_iv2]
        v3 = gjk_state.polytope_verts.obj2[i_b, face_iv3]
        witness2 = v1 * _lambda[0] + v2 * _lambda[1] + v3 * _lambda[2]

        gjk_state.witness.point_obj1[i_b, 0] = witness1
        gjk_state.witness.point_obj2[i_b, 0] = witness2

    return flag


@ti.func
def func_safe_epa_init(
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    i_ga,
    i_gb,
    i_b,
):
    """
    Create the polytope for safe EPA from a 3-simplex (tetrahedron).

    Assume the tetrahedron is a non-degenerate simplex.
    """

    # Insert simplex vertices into the polytope
    vi = ti.Vector([0, 0, 0, 0], dt=ti.i32)
    for i in range(4):
        vi[i] = func_epa_insert_vertex_to_polytope(
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

    for i in range(4):
        # Vertex indices for the faces in the hexahedron
        v1, v2, v3 = vi[0], vi[1], vi[2]
        # Adjacent face indices for the faces in the hexahedron
        a1, a2, a3 = 1, 3, 2
        if i == 1:
            v1, v2, v3 = vi[0], vi[3], vi[1]
            a1, a2, a3 = 2, 3, 0
        elif i == 2:
            v1, v2, v3 = vi[0], vi[2], vi[3]
            a1, a2, a3 = 0, 3, 1
        elif i == 3:
            v1, v2, v3 = vi[3], vi[2], vi[1]
            a1, a2, a3 = 2, 0, 1

        func_safe_attach_face_to_polytope(gjk_state, gjk_info, i_b, v1, v2, v3, a1, a2, a3)

    # Initialize face map
    for i in ti.static(range(4)):
        gjk_state.polytope_faces_map[i_b, i] = i
        gjk_state.polytope_faces.map_idx[i_b, i] = i
    gjk_state.polytope.nfaces_map[i_b] = 4


@ti.func
def func_safe_attach_face_to_polytope(
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    i_b,
    i_v1,
    i_v2,
    i_v3,
    i_a1,
    i_a2,
    i_a3,
):
    """
    Attach a face to the polytope.

    While attaching the face, 1) determine its normal direction, and 2) estimate the lower bound of the penetration
    depth in robust manner.

    [i_v1, i_v2, i_v3] are the vertices of the face, [i_a1, i_a2, i_a3] are the adjacent faces.
    """
    n = gjk_state.polytope.nfaces[i_b]
    gjk_state.polytope_faces.verts_idx[i_b, n][0] = i_v1
    gjk_state.polytope_faces.verts_idx[i_b, n][1] = i_v2
    gjk_state.polytope_faces.verts_idx[i_b, n][2] = i_v3
    gjk_state.polytope_faces.adj_idx[i_b, n][0] = i_a1
    gjk_state.polytope_faces.adj_idx[i_b, n][1] = i_a2
    gjk_state.polytope_faces.adj_idx[i_b, n][2] = i_a3
    gjk_state.polytope_faces.visited[i_b, n] = 0
    gjk_state.polytope.nfaces[i_b] += 1

    # Compute the normal of the plane
    normal, flag = func_plane_normal(
        gjk_info,
        gjk_state.polytope_verts.mink[i_b, i_v3],
        gjk_state.polytope_verts.mink[i_b, i_v2],
        gjk_state.polytope_verts.mink[i_b, i_v1],
    )
    if flag == gjk.RETURN_CODE.SUCCESS:
        face_center = (
            gjk_state.polytope_verts.mink[i_b, i_v1]
            + gjk_state.polytope_verts.mink[i_b, i_v2]
            + gjk_state.polytope_verts.mink[i_b, i_v3]
        ) / 3.0

        # Use origin for initialization
        max_orient = -normal.dot(face_center)
        max_abs_orient = ti.abs(max_orient)

        # Consider other vertices in the polytope to reorient the normal
        nverts = gjk_state.polytope.nverts[i_b]
        for i_v in range(nverts):
            if i_v != i_v1 and i_v != i_v2 and i_v != i_v3:
                diff = gjk_state.polytope_verts.mink[i_b, i_v] - face_center
                orient = normal.dot(diff)
                if ti.abs(orient) > max_abs_orient:
                    max_abs_orient = ti.abs(orient)
                    max_orient = orient

        if max_orient > 0.0:
            normal = -normal

        gjk_state.polytope_faces.normal[i_b, n] = normal

        # Compute the safe lower bound of the penetration depth. We can do this by taking the minimum dot product
        # between the face normal and the vertices of the polytope face. This is safer than selecting one of the
        # vertices, because the face normal could be unstable, which ends up in significantly different dot product
        # values for different vertices.
        min_dist2 = gjk_info.FLOAT_MAX[None]
        for i in ti.static(range(3)):
            i_v = i_v1
            if i == 1:
                i_v = i_v2
            elif i == 2:
                i_v = i_v3
            v = gjk_state.polytope_verts.mink[i_b, i_v]
            dist2 = normal.dot(v) ** 2
            if dist2 < min_dist2:
                min_dist2 = dist2
        dist2 = min_dist2
        gjk_state.polytope_faces.dist2[i_b, n] = dist2
        gjk_state.polytope_faces.map_idx[i_b, n] = -1  # No map index yet

    return flag


@ti.func
def func_plane_normal(
    gjk_info: array_class.GJKInfo,
    v1,
    v2,
    v3,
):
    """
    Compute the reliable normal of the plane defined by three points.
    """
    normal, flag = gs.ti_vec3(0.0, 0.0, 0.0), gjk.RETURN_CODE.FAIL
    finished = False

    d21 = v2 - v1
    d31 = v3 - v1
    d32 = v3 - v2

    for i in ti.static(range(3)):
        if not finished:
            n = gs.ti_vec3(0.0, 0.0, 0.0)
            if i == 0:
                # Normal = (v1 - v2) x (v3 - v2)
                n = d32.cross(d21)
            elif i == 1:
                # Normal = (v2 - v1) x (v3 - v1)
                n = d21.cross(d31)
            else:
                # Normal = (v1 - v3) x (v2 - v3)
                n = d31.cross(d32)
            nn = n.norm()
            if nn == 0:
                # Zero normal, cannot project.
                flag = gjk.RETURN_CODE.FAIL
                finished = True
            elif nn > gjk_info.FLOAT_MIN[None]:
                normal = n.normalized()
                flag = gjk.RETURN_CODE.SUCCESS
                finished = True

    return normal, flag


from genesis.utils.deprecated_module_wrapper import create_virtual_deprecated_module

create_virtual_deprecated_module(__name__, "genesis.engine.solvers.rigid.gjk_decomp")
