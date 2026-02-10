"""
Multi-contact detection for collision handling.

This module contains the multi-contact detection algorithm based on
Sutherland-Hodgman polygon clipping for finding multiple contact points
between colliding geometric entities (face-face, edge-face pairs).
"""

import gstaichi as ti

import genesis as gs
import genesis.utils.array_class as array_class
from .gjk import func_is_equal_vec, RETURN_CODE
from . import multi_contact_local


@ti.func
def func_multi_contact(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    i_ga,
    i_gb,
    i_b,
    i_f,
    pos_a: ti.types.vector(3, dtype=gs.ti_float),
    quat_a: ti.types.vector(4, dtype=gs.ti_float),
    pos_b: ti.types.vector(3, dtype=gs.ti_float),
    quat_b: ti.types.vector(4, dtype=gs.ti_float),
):
    """
    Multi-contact detection algorithm based on Sutherland-Hodgman polygon clipping algorithm. For the two geometric
    entities that form the minimum distance (e.g. face-face, edge-face), this function tests if the pair is
    parallel, and if so, it clips one of the pair against the other to find the contact points.

    Parameters
    ----------
    i_f: int
        Index of the face in the EPA polytope where the minimum distance is found.
    pos_a, quat_a: ti.Vector
        Pose of geometry A (passed as parameters instead of reading from geoms_state)
    pos_b, quat_b: ti.Vector
        Pose of geometry B (passed as parameters instead of reading from geoms_state)

    .. seealso::
    MuJoCo's original implementation:
    https://github.com/google-deepmind/mujoco/blob/7dc7a349c5ba2db2d3f8ab50a367d08e2f1afbbc/src/engine/engine_collision_gjk.c#L2112
    """
    # Get vertices of the nearest face from EPA
    v11i = gjk_state.polytope_verts[i_b, gjk_state.polytope_faces[i_b, i_f].verts_idx[0]].id1
    v12i = gjk_state.polytope_verts[i_b, gjk_state.polytope_faces[i_b, i_f].verts_idx[1]].id1
    v13i = gjk_state.polytope_verts[i_b, gjk_state.polytope_faces[i_b, i_f].verts_idx[2]].id1
    v21i = gjk_state.polytope_verts[i_b, gjk_state.polytope_faces[i_b, i_f].verts_idx[0]].id2
    v22i = gjk_state.polytope_verts[i_b, gjk_state.polytope_faces[i_b, i_f].verts_idx[1]].id2
    v23i = gjk_state.polytope_verts[i_b, gjk_state.polytope_faces[i_b, i_f].verts_idx[2]].id2
    v11 = gjk_state.polytope_verts[i_b, gjk_state.polytope_faces[i_b, i_f].verts_idx[0]].obj1
    v12 = gjk_state.polytope_verts[i_b, gjk_state.polytope_faces[i_b, i_f].verts_idx[1]].obj1
    v13 = gjk_state.polytope_verts[i_b, gjk_state.polytope_faces[i_b, i_f].verts_idx[2]].obj1
    v21 = gjk_state.polytope_verts[i_b, gjk_state.polytope_faces[i_b, i_f].verts_idx[0]].obj2
    v22 = gjk_state.polytope_verts[i_b, gjk_state.polytope_faces[i_b, i_f].verts_idx[1]].obj2
    v23 = gjk_state.polytope_verts[i_b, gjk_state.polytope_faces[i_b, i_f].verts_idx[2]].obj2

    # Get the simplex dimension of geom 1 and 2
    nface1, nface2 = 0, 0
    for i in range(2):
        v1i, v2i, v3i, v1, v2, v3 = v11i, v12i, v13i, v11, v12, v13
        if i == 1:
            v1i, v2i, v3i, v1, v2, v3 = v21i, v22i, v23i, v21, v22, v23

        nface, v1i, v2i, v3i, v1, v2, v3 = func_simplex_dim(v1i, v2i, v3i, v1, v2, v3)
        if i == 0:
            nface1, v11i, v12i, v13i, v11, v12, v13 = nface, v1i, v2i, v3i, v1, v2, v3
        else:
            nface2, v21i, v22i, v23i, v21, v22, v23 = nface, v1i, v2i, v3i, v1, v2, v3
    dir = gjk_state.witness[i_b, 0].point_obj2 - gjk_state.witness[i_b, 0].point_obj1
    dir_neg = gjk_state.witness[i_b, 0].point_obj1 - gjk_state.witness[i_b, 0].point_obj2

    # Get all possible face normals for each geom
    nnorms1, nnorms2 = 0, 0
    geom_type_a = geoms_info.type[i_ga]
    geom_type_b = geoms_info.type[i_gb]

    for i_g0 in range(2):
        geom_type = geom_type_a if i_g0 == 0 else geom_type_b
        i_g = i_ga if i_g0 == 0 else i_gb
        nface = nface1 if i_g0 == 0 else nface2
        v1i = v11i if i_g0 == 0 else v21i
        v2i = v12i if i_g0 == 0 else v22i
        v3i = v13i if i_g0 == 0 else v23i
        t_dir = dir_neg if i_g0 == 0 else dir

        # Use pose parameters instead of reading from geoms_state
        quat_g = quat_a if i_g == i_ga else quat_b

        nnorms = 0
        if geom_type == gs.GEOM_TYPE.BOX:
            nnorms = multi_contact_local.func_potential_box_normals_local(
                geoms_info=geoms_info,
                gjk_state=gjk_state,
                gjk_info=gjk_info,
                i_g=i_g,
                quat=quat_g,
                i_b=i_b,
                dim=nface,
                v1=v1i,
                v2=v2i,
                v3=v3i,
                dir=t_dir,
            )
        elif geom_type == gs.GEOM_TYPE.MESH:
            nnorms = multi_contact_local.func_potential_mesh_normals_local(
                geoms_info=geoms_info,
                verts_info=verts_info,
                faces_info=faces_info,
                gjk_state=gjk_state,
                gjk_info=gjk_info,
                i_g=i_g,
                quat=quat_g,
                i_b=i_b,
                dim=nface,
                v1=v1i,
                v2=v2i,
                v3=v3i,
            )

        for i_n in range(nnorms):
            if i_g0 == 0:
                gjk_state.contact_faces[i_b, i_n].normal1 = gjk_state.contact_normals[i_b, i_n].normal
                gjk_state.contact_faces[i_b, i_n].id1 = gjk_state.contact_normals[i_b, i_n].id
                nnorms1 = nnorms
            else:
                gjk_state.contact_faces[i_b, i_n].normal2 = gjk_state.contact_normals[i_b, i_n].normal
                gjk_state.contact_faces[i_b, i_n].id2 = gjk_state.contact_normals[i_b, i_n].id
                nnorms2 = nnorms

    # Determine if any two face normals match
    aligned_faces_idx, aligned_faces_flag = func_find_aligned_faces(gjk_state, gjk_info, i_b, nnorms1, nnorms2)
    no_multiple_contacts = False
    edgecon1, edgecon2 = False, False

    if aligned_faces_flag == RETURN_CODE.FAIL:
        # No aligned faces found; check if there was edge-face collision
        # [is_edge_face]: geom1 is edge, geom2 is face
        # [is_face_edge]: geom1 is face, geom2 is edge
        is_edge_face = (nface1 < 3) and (nface1 <= nface2)
        is_face_edge = (not is_edge_face) and nface2 < 3

        if is_edge_face or is_face_edge:
            i_g = i_ga if is_edge_face else i_gb
            geom_type = geom_type_a if is_edge_face else geom_type_b
            nface = nface1 if is_edge_face else nface2
            v1 = v11 if is_edge_face else v21
            v2 = v12 if is_edge_face else v22
            v1i = v11i if is_edge_face else v21i
            v2i = v12i if is_edge_face else v22i

            # Use pose parameters instead of reading from geoms_state
            pos_g = pos_a if i_g == i_ga else pos_b
            quat_g = quat_a if i_g == i_ga else quat_b

            nnorms = 0
            if geom_type == gs.GEOM_TYPE.BOX:
                nnorms = multi_contact_local.func_potential_box_edge_normals_local(
                    geoms_info=geoms_info,
                    gjk_state=gjk_state,
                    gjk_info=gjk_info,
                    i_g=i_g,
                    pos=pos_g,
                    quat=quat_g,
                    i_b=i_b,
                    dim=nface,
                    v1=v1,
                    v2=v2,
                    v1i=v1i,
                    v2i=v2i,
                )
            elif geom_type == gs.GEOM_TYPE.MESH:
                nnorms = multi_contact_local.func_potential_mesh_edge_normals_local(
                    geoms_info=geoms_info,
                    verts_info=verts_info,
                    faces_info=faces_info,
                    gjk_state=gjk_state,
                    gjk_info=gjk_info,
                    i_g=i_g,
                    pos=pos_g,
                    quat=quat_g,
                    i_b=i_b,
                    dim=nface,
                    v1=v1,
                    v2=v2,
                    v1i=v1i,
                    v2i=v2i,
                )

            if is_edge_face:
                nnorms1 = nnorms
            else:
                nnorms2 = nnorms

            if nnorms > 0:
                for i_n in range(nnorms):
                    if is_edge_face:
                        gjk_state.contact_faces[i_b, i_n].normal1 = gjk_state.contact_normals[i_b, i_n].normal
                    else:
                        gjk_state.contact_faces[i_b, i_n].normal2 = gjk_state.contact_normals[i_b, i_n].normal

                    gjk_state.contact_faces[i_b, i_n].endverts = gjk_state.contact_normals[i_b, i_n].endverts

            # Check if any of the edge normals match
            nedges, nfaces = nnorms1, nnorms2
            if not is_edge_face:
                nedges, nfaces = nfaces, nedges
            aligned_faces_idx, aligned_edge_face_flag = func_find_aligned_edge_face(
                gjk_state, gjk_info, i_b, nedges, nfaces, is_edge_face
            )

            if aligned_edge_face_flag == RETURN_CODE.FAIL:
                no_multiple_contacts = True
            else:
                if is_edge_face:
                    edgecon1 = True
                else:
                    edgecon2 = True
        else:
            # No multiple contacts found
            no_multiple_contacts = True

    if not no_multiple_contacts:
        i, j = aligned_faces_idx[0], aligned_faces_idx[1]

        # Recover matching edge or face from geoms
        for k in range(2):
            edgecon = edgecon1 if k == 0 else edgecon2
            geom_type = geom_type_a if k == 0 else geom_type_b
            i_g = i_ga if k == 0 else i_gb

            # Use pose parameters instead of reading from geoms_state
            pos_g = pos_a if i_g == i_ga else pos_b
            quat_g = quat_a if i_g == i_ga else quat_b

            nface = 0
            if edgecon:
                if k == 0:
                    gjk_state.contact_faces[i_b, 0].vert1 = gjk_state.polytope_verts[
                        i_b, gjk_state.polytope_faces[i_b, i_f].verts_idx[0]
                    ].obj1
                    gjk_state.contact_faces[i_b, 1].vert1 = gjk_state.contact_faces[i_b, i].endverts
                else:
                    gjk_state.contact_faces[i_b, 0].vert2 = gjk_state.polytope_verts[
                        i_b, gjk_state.polytope_faces[i_b, i_f].verts_idx[0]
                    ].obj2
                    gjk_state.contact_faces[i_b, 1].vert2 = gjk_state.contact_faces[i_b, j].endverts

                nface = 2
            else:
                normal_face_idx = gjk_state.contact_faces[i_b, i].id1
                if k == 0 and edgecon2:
                    # Since [i] is the edge idx, use [j]
                    normal_face_idx = gjk_state.contact_faces[i_b, j].id1
                elif k == 1:
                    normal_face_idx = gjk_state.contact_faces[i_b, j].id2

                if geom_type == gs.GEOM_TYPE.BOX:
                    nface = multi_contact_local.func_box_face_local(
                        geoms_info=geoms_info,
                        gjk_state=gjk_state,
                        i_g=i_g,
                        pos=pos_g,
                        quat=quat_g,
                        i_b=i_b,
                        i_o=k,
                        face_idx=normal_face_idx,
                    )
                elif geom_type == gs.GEOM_TYPE.MESH:
                    nface = multi_contact_local.func_mesh_face_local(
                        verts_info=verts_info,
                        faces_info=faces_info,
                        gjk_state=gjk_state,
                        i_g=i_g,
                        pos=pos_g,
                        quat=quat_g,
                        i_b=i_b,
                        i_o=k,
                        face_idx=normal_face_idx,
                    )

            if k == 0:
                nface1 = nface
            else:
                nface2 = nface

        approx_dir = gs.ti_vec3(0.0, 0.0, 0.0)
        normal = gs.ti_vec3(0.0, 0.0, 0.0)
        if edgecon1:
            # Face 1 is an edge, so clip face 1 against face 2
            approx_dir = gjk_state.contact_faces[i_b, j].normal2 * dir.norm()
            normal = gjk_state.contact_faces[i_b, j].normal2
        elif edgecon2:
            # Face 2 is an edge, so clip face 2 against face 1
            approx_dir = gjk_state.contact_faces[i_b, j].normal1 * dir.norm()
            normal = gjk_state.contact_faces[i_b, j].normal1
        else:
            # Face-face contact
            approx_dir = gjk_state.contact_faces[i_b, j].normal2 * dir.norm()
            normal = gjk_state.contact_faces[i_b, i].normal1

        # Clip polygon
        func_clip_polygon(gjk_state, gjk_info, i_b, nface1, nface2, edgecon1, edgecon2, normal, approx_dir)


@ti.func
def func_simplex_dim(
    v1i,
    v2i,
    v3i,
    v1,
    v2,
    v3,
):
    """
    Determine the dimension of the given simplex (1-3).

    If every point is the same, 1-dim. If two points are the same, 2-dim. If all points are different, 3-dim.
    """
    dim = 0
    rv1i, rv2i, rv3i = v1i, v2i, v3i
    rv1, rv2, rv3 = v1, v2, v3
    if v1i != v2i:
        if (v1i == v3i) or (v2i == v3i):
            # Two points are the same
            dim = 2
        else:
            # All points are different
            dim = 3
    else:
        if v1i != v3i:
            # Two points are the same
            dim = 2
            # Swap v2 and v3
            rv2i, rv3i = rv3i, rv2i
            rv2, rv3 = rv3, rv2
        else:
            # All points are the same
            dim = 1

    return dim, rv1i, rv2i, rv3i, rv1, rv2, rv3


@ti.func
def func_cmp_bit(
    v1,
    v2,
    v3,
    n,
    shift,
):
    """
    Compare one bit of v1 and v2 that sits at position `shift` (shift = 0 for the LSB, 1 for the next bit, ...).

    Returns:
    -------
    int
        1  if both bits are 1
        -1 if both bits are 0
        0  if bits differ
    """

    b1 = (v1 >> shift) & 1  # 0 or 1
    b2 = (v2 >> shift) & 1  # 0 or 1
    b3 = (v3 >> shift) & 1  # 0 or 1

    res = 0
    if n == 3:
        both_set = b1 & b2 & b3  # 1 when 11, else 0
        both_clear = (b1 ^ 1) & (b2 ^ 1) & (b3 ^ 1)  # 1 when 00, else 0
        res = both_set - both_clear
    elif n == 2:
        both_set = b1 & b2  # 1 when 11, else 0
        both_clear = (b1 ^ 1) & (b2 ^ 1)  # 1 when 00, else 0
        res = both_set - both_clear
    elif n == 1:
        both_set = b1  # 1 when 1, else 0
        both_clear = b1 ^ 1  # 1 when 0, else 0
        res = both_set - both_clear

    return res


@ti.func
@ti.func
def func_find_aligned_faces(
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    i_b,
    nv,
    nw,
):
    """
    Find if any two faces from [contact_faces] are aligned.
    """
    res = gs.ti_ivec2(0, 0)
    flag = RETURN_CODE.FAIL

    for i, j in ti.ndrange(nv, nw):
        ni = gjk_state.contact_faces[i_b, i].normal1
        nj = gjk_state.contact_faces[i_b, j].normal2
        if ni.dot(nj) < -gjk_info.contact_face_tol[None]:
            res[0] = i
            res[1] = j
            flag = RETURN_CODE.SUCCESS
            break

    return res, flag


@ti.func
def func_safe_normalize(
    gjk_info: array_class.GJKInfo,
    v,
):
    """
    Normalize the vector [v] safely.
    """
    norm = v.norm()

    if norm < gjk_info.FLOAT_MIN[None]:
        # If the vector is too small, set it to a default value
        v[0] = 1.0
        v[1] = 0.0
        v[2] = 0.0
    else:
        # Normalize the vector
        inv_norm = 1.0 / norm
        v *= inv_norm
    return v


@ti.func
def func_find_aligned_edge_face(
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    i_b,
    nedge,
    nface,
    is_edge_face,
):
    """
    Find if an edge and face from [contact_faces] are aligned.
    """
    res = gs.ti_ivec2(0, 0)
    flag = RETURN_CODE.FAIL

    for i, j in ti.ndrange(nedge, nface):
        ni = gjk_state.contact_faces[i_b, i].normal1
        nj = gjk_state.contact_faces[i_b, j].normal2

        if not is_edge_face:
            # The first normal is the edge normal
            ni = gjk_state.contact_faces[i_b, i].normal2
        if not is_edge_face:
            # The second normal is the face normal
            nj = gjk_state.contact_faces[i_b, j].normal1

        if ti.abs(ni.dot(nj)) < gjk_info.contact_edge_tol[None]:
            res[0] = i
            res[1] = j
            flag = RETURN_CODE.SUCCESS
            break

    return res, flag


@ti.func
def func_clip_polygon(
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    i_b,
    nface1,
    nface2,
    edgecon1,
    edgecon2,
    normal,
    approx_dir,
):
    """
    Clip a polygon against the another polygon using Sutherland-Hodgman algorithm.

    Parameters:
    ----------
    normal: gs.ti_vec3
        The normal of the clipping polygon.
    approx_dir: gs.ti_vec3
        Preferred separation direction for the clipping.
    """
    clipping_polygon = 1 if not edgecon1 else 2
    clipping_polygon_nface = nface1 if clipping_polygon == 1 else nface2

    # The clipping polygon should be at least a triangle
    if clipping_polygon_nface >= 3:
        # For each edge of the clipping polygon, find the half-plane that is defined by the edge and the normal.
        # The normal of half-plane is perpendicular to the edge and face normal.
        for i in range(clipping_polygon_nface):
            v1 = gjk_state.contact_faces[i_b, i].vert1
            v2 = gjk_state.contact_faces[i_b, (i + 1) % clipping_polygon_nface].vert1
            v3 = gjk_state.contact_faces[i_b, (i + 2) % clipping_polygon_nface].vert1

            if clipping_polygon == 2:
                v1 = gjk_state.contact_faces[i_b, i].vert2
                v2 = gjk_state.contact_faces[i_b, (i + 1) % clipping_polygon_nface].vert2
                v3 = gjk_state.contact_faces[i_b, (i + 2) % clipping_polygon_nface].vert2

            # Plane normal
            res = (v2 - v1).cross(normal)

            # Reorient normal if needed
            inside_v3 = func_halfspace(gjk_info, v1, res, v3)
            if not inside_v3:
                res = -res

            gjk_state.contact_halfspaces[i_b, i].normal = res

            # Plane distance
            gjk_state.contact_halfspaces[i_b, i].dist = v1.dot(res)

        # Initialize buffers to store the clipped polygons
        nclipped = gs.ti_ivec2(0, 0)
        nclipped[0] = nface2 if clipping_polygon == 1 else nface1

        # These values are swapped during the clipping process.
        pi, ci = 0, 1

        for i in range(nclipped[pi]):
            if clipping_polygon == 1:
                gjk_state.contact_clipped_polygons[i_b, pi, i] = gjk_state.contact_faces[i_b, i].vert2
            else:
                gjk_state.contact_clipped_polygons[i_b, pi, i] = gjk_state.contact_faces[i_b, i].vert1

        # For each edge of the clipping polygon, clip the subject polygon against it.
        # Here we use the Sutherland-Hodgman algorithm.
        for e in range(clipping_polygon_nface):
            # Get the point [a] on the clipping polygon edge,
            # and the normal [n] of the half-plane defined by the edge.
            a = gjk_state.contact_faces[i_b, e].vert1
            if clipping_polygon == 2:
                a = gjk_state.contact_faces[i_b, e].vert2
            n = gjk_state.contact_halfspaces[i_b, e].normal
            d = gjk_state.contact_halfspaces[i_b, e].dist

            for i in range(nclipped[pi]):
                # Get edge PQ of the subject polygon
                P = gjk_state.contact_clipped_polygons[i_b, pi, i]
                Q = gjk_state.contact_clipped_polygons[i_b, pi, (i + 1) % nclipped[pi]]

                # Determine if P and Q are inside or outside the half-plane
                inside_P = func_halfspace(gjk_info, a, n, P)
                inside_Q = func_halfspace(gjk_info, a, n, Q)

                # PQ entirely outside the clipping edge, skip
                if not inside_P and not inside_Q:
                    continue

                # PQ entirely inside the clipping edge, add Q to the clipped polygon
                if inside_P and inside_Q:
                    gjk_state.contact_clipped_polygons[i_b, ci, nclipped[ci]] = Q
                    nclipped[ci] += 1
                    continue

                # PQ intersects the half-plane, add the intersection point
                t, ip = func_plane_intersect(gjk_info, n, d, P, Q)
                if t >= 0 and t <= 1:
                    gjk_state.contact_clipped_polygons[i_b, ci, nclipped[ci]] = ip
                    nclipped[ci] += 1

                # If Q is inside the half-plane, add it to the clipped polygon
                if inside_Q:
                    gjk_state.contact_clipped_polygons[i_b, ci, nclipped[ci]] = Q
                    nclipped[ci] += 1

            # Swap the buffers for the next edge clipping
            pi, ci = ci, pi

            # Reset the next clipped polygon count
            nclipped[ci] = 0

        nclipped_polygon = nclipped[pi]

        if nclipped_polygon >= 1:
            if gjk_info.max_contacts_per_pair[None] < 5 and nclipped_polygon > 4:
                # Approximate the clipped polygon with a convex quadrilateral
                gjk_state.n_witness[i_b] = 4
                rect = func_approximate_polygon_with_quad(gjk_state, i_b, pi, nclipped_polygon)

                for i in range(4):
                    witness2 = gjk_state.contact_clipped_polygons[i_b, pi, rect[i]]
                    witness1 = witness2 - approx_dir
                    gjk_state.witness[i_b, i].point_obj1 = witness1
                    gjk_state.witness[i_b, i].point_obj2 = witness2

            elif nclipped_polygon > gjk_info.max_contacts_per_pair[None]:
                # If the number of contacts exceeds the limit,
                # only use the first [max_contacts_per_pair] contacts.
                gjk_state.n_witness[i_b] = gjk_info.max_contacts_per_pair[None]

                for i in range(gjk_info.max_contacts_per_pair[None]):
                    witness2 = gjk_state.contact_clipped_polygons[i_b, pi, i]
                    witness1 = witness2 - approx_dir
                    gjk_state.witness[i_b, i].point_obj1 = witness1
                    gjk_state.witness[i_b, i].point_obj2 = witness2

            else:
                n_witness = 0
                # Just use every contact in the clipped polygon
                for i in range(nclipped_polygon):
                    skip = False

                    polygon_vert = gjk_state.contact_clipped_polygons[i_b, pi, i]

                    # Find if there were any duplicate contacts similar to [polygon_vert]
                    for j in range(n_witness):
                        prev_witness = gjk_state.witness[i_b, j].point_obj2
                        skip = func_is_equal_vec(polygon_vert, prev_witness, gjk_info.FLOAT_MIN[None])
                        if skip:
                            break

                    if not skip:
                        gjk_state.witness[i_b, n_witness].point_obj2 = polygon_vert
                        gjk_state.witness[i_b, n_witness].point_obj1 = polygon_vert - approx_dir
                        n_witness += 1

                gjk_state.n_witness[i_b] = n_witness


@ti.func
def func_halfspace(
    gjk_info: array_class.GJKInfo,
    a,
    n,
    p,
):
    """
    Check if the point [p] is inside the half-space defined by the plane with normal [n] and point [a].
    """
    return (p - a).dot(n) > -gjk_info.FLOAT_MIN[None]


@ti.func
def func_plane_intersect(
    gjk_info: array_class.GJKInfo,
    pn,
    pd,
    v1,
    v2,
):
    """
    Compute the intersection point of the line segment [v1, v2]
    with the plane defined by the normal [pn] and distance [pd].

    v1 + t * (v2 - v1) = intersection point

    Return:
    -------
    t: float
        The parameter t that defines the intersection point on the line segment.
    """
    t = gjk_info.FLOAT_MAX[None]
    ip = gs.ti_vec3(0, 0, 0)

    dir = v2 - v1
    normal_dot = pn.dot(dir)
    if ti.abs(normal_dot) > gjk_info.FLOAT_MIN[None]:
        t = (pd - pn.dot(v1)) / normal_dot
        if t >= 0 and t <= 1:
            ip = v1 + t * dir

    return t, ip


@ti.func
def func_approximate_polygon_with_quad(
    gjk_state: array_class.GJKState,
    i_b,
    polygon_start,
    nverts,
):
    """
    Find a convex quadrilateral that approximates the given N-gon [polygon]. We find it by selecting the four
    vertices in the polygon that form the maximum area quadrilateral.
    """
    i_v = gs.ti_ivec4(0, 1, 2, 3)
    i_v0 = gs.ti_ivec4(0, 1, 2, 3)
    m = func_quadrilateral_area(gjk_state, i_b, polygon_start, i_v[0], i_v[1], i_v[2], i_v[3])

    # 1: change b, 2: change c, 3: change d
    change_flag = 3

    while True:
        i_v0[0], i_v0[1], i_v0[2], i_v0[3] = i_v[0], i_v[1], i_v[2], i_v[3]
        if change_flag == 3:
            i_v0[3] = (i_v[3] + 1) % nverts
        elif change_flag == 2:
            i_v0[2] = (i_v[2] + 1) % nverts

        # Compute the area of the quadrilateral formed by the vertices
        m_next = func_quadrilateral_area(gjk_state, i_b, polygon_start, i_v0[0], i_v0[1], i_v0[2], i_v0[3])
        if m_next <= m:
            # If the area did not increase
            if change_flag == 3:
                if i_v[1] == i_v[0]:
                    i_v[1] = (i_v[1] + 1) % nverts
                if i_v[2] == i_v[1]:
                    i_v[2] = (i_v[2] + 1) % nverts
                if i_v[3] == i_v[2]:
                    i_v[3] = (i_v[3] + 1) % nverts
                # Change a if possible
                if i_v[0] == nverts - 1:
                    break
                i_v[0] = (i_v[0] + 1) % nverts
            elif change_flag == 2:
                # Now change b
                change_flag = 1
            elif change_flag == 1:
                # Now change d
                change_flag = 3
        else:
            # If the area increased
            m = m_next
            i_v[0], i_v[1], i_v[2], i_v[3] = i_v0[0], i_v0[1], i_v0[2], i_v0[3]
            if change_flag == 3:
                # Now change c
                change_flag = 2
            elif change_flag == 2:
                # Keep changing c
                pass
            elif change_flag == 1:
                # Keep changing b
                pass

    return i_v


@ti.func
def func_quadrilateral_area(
    gjk_state: array_class.GJKState,
    i_b,
    i_0,
    i_v0,
    i_v1,
    i_v2,
    i_v3,
):
    """
    Compute the area of the quadrilateral formed by vertices [i_v0, i_v1, i_v2, i_v3] in the [verts] array.
    """
    a = gjk_state.contact_clipped_polygons[i_b, i_0, i_v0]
    b = gjk_state.contact_clipped_polygons[i_b, i_0, i_v1]
    c = gjk_state.contact_clipped_polygons[i_b, i_0, i_v2]
    d = gjk_state.contact_clipped_polygons[i_b, i_0, i_v3]
    e = (d - a).cross(b - d) + (c - b).cross(a - c)

    return 0.5 * e.norm()
