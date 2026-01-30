"""
Thread-local versions of multi-contact detection functions.

This module provides versions of multi-contact functions that accept pos/quat as direct
parameters instead of reading from geoms_state, enabling race-free multi-contact
detection when parallelizing across collision pairs within the same environment.
"""

import gstaichi as ti

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.solvers.rigid.collider import multi_contact
from genesis.engine.solvers.rigid.collider.gjk import RETURN_CODE
from genesis.utils import array_class


@ti.func
def func_mesh_face_local(
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    gjk_state: array_class.GJKState,
    i_g,
    pos: ti.types.vector(3, dtype=gs.ti_float),
    quat: ti.types.vector(4, dtype=gs.ti_float),
    i_b,
    i_o,
    face_idx,
):
    """
    Thread-local version of func_mesh_face.

    Get the face vertices of the mesh, using thread-local pos/quat.

    Thread-safety note: Geometry index `i_g` is only used to pass through to `faces_info`
    and `verts_info` for read-only metadata access (face vertex indices, initial positions).
    It does not access `geoms_state.pos` or `geoms_state.quat`.

    Args:
        verts_info: Vertex information
        faces_info: Face information
        gjk_state: GJK algorithm state (for storing contact face vertices)
        i_g: Geometry index (for metadata only)
        pos: Thread-local position of the geometry
        quat: Thread-local quaternion of the geometry
        i_b: Batch/environment index
        i_o: Object index (0 or 1, determines which contact face to write to)
        face_idx: Index of the face to get vertices for

    Returns:
        Number of vertices in the face (always 3 for triangular meshes)
    """
    nvert = 3
    for i in range(nvert):
        i_v = faces_info[face_idx].verts_idx[i]
        v = verts_info.init_pos[i_v]
        v = gu.ti_transform_by_trans_quat(v, pos, quat)
        if i_o == 0:
            gjk_state.contact_faces[i_b, i].vert1 = v
        else:
            gjk_state.contact_faces[i_b, i].vert2 = v

    return nvert


@ti.func
def func_box_normal_from_collision_normal_local(
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    i_g,
    quat: ti.types.vector(4, dtype=gs.ti_float),
    i_b,
    dir,
):
    """
    Thread-local version of func_box_normal_from_collision_normal.

    Among the 6 faces of the box, find the one of which normal is closest to the [dir],
    using thread-local quat.

    Thread-safety note: Geometry index `i_g` is not used in this function at all
    (retained for API consistency with original). It does not access `geoms_state.pos`
    or `geoms_state.quat`.

    Args:
        gjk_state: GJK algorithm state (for storing contact normals)
        gjk_info: GJK algorithm parameters (contact face tolerance)
        i_g: Geometry index (unused, for API consistency)
        quat: Thread-local quaternion of the geometry
        i_b: Batch/environment index
        dir: Direction vector to match against box face normals

    Returns:
        RETURN_CODE.SUCCESS if a matching face normal was found, RETURN_CODE.FAIL otherwise
    """
    # Every box face normal
    normals = ti.Vector(
        [1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0],
        dt=gs.ti_float,
    )

    # Get local collision normal
    local_dir = gu.ti_transform_by_quat(dir, gu.ti_inv_quat(quat))
    local_dir = local_dir.normalized()

    # Determine the closest face normal
    flag = RETURN_CODE.FAIL
    for i in range(6):
        n = gs.ti_vec3(normals[3 * i + 0], normals[3 * i + 1], normals[3 * i + 2])
        if local_dir.dot(n) > gjk_info.contact_face_tol[None]:
            flag = RETURN_CODE.SUCCESS
            gjk_state.contact_normals[i_b, 0].normal = n
            gjk_state.contact_normals[i_b, 0].id = i
            break

    return flag


@ti.func
def func_safe_normalize(
    gjk_info: array_class.GJKInfo,
    v,
):
    """
    Safely normalize a vector (helper function for edge normal computation).

    This is a simple utility that doesn't access geoms_state, included here
    for convenience when using local functions.
    """
    norm = v.norm()
    if norm > gjk_info.FLOAT_MIN[None]:
        return v / norm
    else:
        return gs.ti_vec3(0.0, 0.0, 0.0)


@ti.func
def func_potential_box_edge_normals_local(
    geoms_info: array_class.GeomsInfo,
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    i_g,
    pos: ti.types.vector(3, dtype=gs.ti_float),
    quat: ti.types.vector(4, dtype=gs.ti_float),
    i_b,
    dim,
    v1,
    v2,
    v1i,
    v2i,
):
    """
    Thread-local version of func_potential_box_edge_normals.

    For a simplex defined on a box with two vertices [v1, v2],
    we find which edge normals are potentially related to the simplex,
    using thread-local pos/quat.

    If the simplex is a line, at most one edge normal are related.
    If the simplex is a point, at most three edge normals are related.

    We identify related edge normals to the simplex by checking the vertex indices of the simplex.

    Thread-safety note: Geometry index `i_g` is only used for read-only metadata access
    (geometry size data, vertex start index). It does not access `geoms_state.pos` or
    `geoms_state.quat`.

    Args:
        geoms_info: Geometry information
        gjk_state: GJK algorithm state (for storing contact normals)
        gjk_info: GJK algorithm parameters
        i_g: Geometry index (for metadata only)
        pos: Thread-local position of the geometry
        quat: Thread-local quaternion of the geometry
        i_b: Batch/environment index
        dim: Dimension of the simplex (1=point, 2=line)
        v1: First vertex position
        v2: Second vertex position
        v1i: First vertex index
        v2i: Second vertex index

    Returns:
        Number of edge normals found
    """
    g_size_x = geoms_info.data[i_g][0] * 0.5
    g_size_y = geoms_info.data[i_g][1] * 0.5
    g_size_z = geoms_info.data[i_g][2] * 0.5

    v1i -= geoms_info.vert_start[i_g]
    v2i -= geoms_info.vert_start[i_g]

    n_normals = 0

    if dim == 2:
        # If the nearest face is an edge
        gjk_state.contact_normals[i_b, 0].endverts = v2
        gjk_state.contact_normals[i_b, 0].normal = func_safe_normalize(gjk_info, v2 - v1)

        n_normals = 1
    elif dim == 1:
        # If the nearest face is a point, consider three adjacent edges
        x = g_size_x if (v1i & 1) else -g_size_x
        y = g_size_y if (v1i & 2) else -g_size_y
        z = g_size_z if (v1i & 4) else -g_size_z

        for i in range(3):
            bv = gs.ti_vec3(-x, y, z)
            if i == 1:
                bv = gs.ti_vec3(x, -y, z)
            elif i == 2:
                bv = gs.ti_vec3(x, y, -z)
            ev = gu.ti_transform_by_trans_quat(bv, pos, quat)
            r = func_safe_normalize(gjk_info, ev - v1)

            gjk_state.contact_normals[i_b, i].endverts = ev
            gjk_state.contact_normals[i_b, i].normal = r

        n_normals = 3

    return n_normals


@ti.func
def func_potential_mesh_edge_normals_local(
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    i_g,
    pos: ti.types.vector(3, dtype=gs.ti_float),
    quat: ti.types.vector(4, dtype=gs.ti_float),
    i_b,
    dim,
    v1,
    v2,
    v1i,
    v2i,
):
    """
    Thread-local version of func_potential_mesh_edge_normals.

    For a simplex defined on a mesh with two vertices [v1, v2],
    we find which edge normals are potentially related to the simplex,
    using thread-local pos/quat.

    If the simplex is a line, at most one edge normal are related.
    If the simplex is a point, multiple edges that are adjacent to the point could be related.

    We identify related edge normals to the simplex by checking the vertex indices of the simplex.

    Thread-safety note: Geometry index `i_g` is only used for read-only metadata access
    (face start/end indices). It does not access `geoms_state.pos` or `geoms_state.quat`.

    Args:
        geoms_info: Geometry information
        verts_info: Vertex information
        faces_info: Face information
        gjk_state: GJK algorithm state (for storing contact normals)
        gjk_info: GJK algorithm parameters
        i_g: Geometry index (for metadata only)
        pos: Thread-local position of the geometry
        quat: Thread-local quaternion of the geometry
        i_b: Batch/environment index
        dim: Dimension of the simplex (1=point, 2=line)
        v1: First vertex position
        v2: Second vertex position
        v1i: First vertex index
        v2i: Second vertex index

    Returns:
        Number of edge normals found
    """
    # Number of potential face normals
    n_normals = 0

    if dim == 2:
        # If the nearest face is an edge
        gjk_state.contact_normals[i_b, 0].endverts = v2
        gjk_state.contact_normals[i_b, 0].normal = func_safe_normalize(gjk_info, v2 - v1)

        n_normals = 1

    elif dim == 1:
        # If the nearest face is a point, consider every adjacent edge
        # Exhaustive search for the edge normals
        face_start = geoms_info.face_start[i_g]
        face_end = geoms_info.face_end[i_g]
        for i_f in range(face_start, face_end):
            face = faces_info[i_f].verts_idx

            v1_idx = -1
            if v1i == face[0]:
                v1_idx = 0
            elif v1i == face[1]:
                v1_idx = 1
            elif v1i == face[2]:
                v1_idx = 2

            if v1_idx != -1:
                # Consider the next vertex of [v1] in the face
                v2_idx = (v1_idx + 1) % 3
                t_v2i = face[v2_idx]

                # Compute the edge normal
                v2_pos = verts_info.init_pos[t_v2i]
                v2_pos = gu.ti_transform_by_trans_quat(v2_pos, pos, quat)
                t_res = func_safe_normalize(gjk_info, v2_pos - v1)

                gjk_state.contact_normals[i_b, n_normals].normal = t_res
                gjk_state.contact_normals[i_b, n_normals].endverts = v2_pos

                n_normals += 1
                if n_normals == gjk_info.max_contact_polygon_verts[None]:
                    break

    return n_normals


@ti.func
def func_potential_mesh_normals_local(
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    i_g,
    quat: ti.types.vector(4, dtype=gs.ti_float),
    i_b,
    dim,
    v1,
    v2,
    v3,
):
    """
    Thread-local version of func_potential_mesh_normals.

    For a simplex defined on a mesh with three vertices [v1, v2, v3],
    we find which face normals are potentially related to the simplex,
    using thread-local quat.

    If the simplex is a triangle, at most one face normal is related.
    If the simplex is a line, at most two face normals are related.
    If the simplex is a point, multiple faces that are adjacent to the point could be related.

    We identify related face normals to the simplex by checking the vertex indices of the simplex.

    Thread-safety note: Geometry index `i_g` is only used for read-only metadata access
    (face start/end indices). It does not access `geoms_state.pos` or `geoms_state.quat`.
    Note that this function only uses quat (not pos) since face normals are orientation-dependent
    but not position-dependent.

    Args:
        geoms_info: Geometry information
        verts_info: Vertex information
        faces_info: Face information
        gjk_state: GJK algorithm state (for storing contact normals)
        gjk_info: GJK algorithm parameters
        i_g: Geometry index (for metadata only)
        quat: Thread-local quaternion of the geometry
        i_b: Batch/environment index
        dim: Dimension of the simplex (1=point, 2=line, 3=triangle)
        v1: First vertex index
        v2: Second vertex index
        v3: Third vertex index

    Returns:
        Number of face normals found
    """
    # Number of potential face normals
    n_normals = 0

    # Exhaustive search for the face normals
    # @TODO: This would require a lot of cost if the mesh is large. It would be better to precompute adjacency
    # information in the solver and use it here.
    face_start = geoms_info.face_start[i_g]
    face_end = geoms_info.face_end[i_g]

    for i_f in range(face_start, face_end):
        face = faces_info[i_f].verts_idx
        has_vs = gs.ti_ivec3(0, 0, 0)
        if v1 == face[0] or v1 == face[1] or v1 == face[2]:
            has_vs[0] = 1
        if v2 == face[0] or v2 == face[1] or v2 == face[2]:
            has_vs[1] = 1
        if v3 == face[0] or v3 == face[1] or v3 == face[2]:
            has_vs[2] = 1

        compute_normal = True
        for j in range(dim):
            compute_normal = compute_normal and (has_vs[j] == 1)

        if compute_normal:
            v1pos = verts_info.init_pos[face[0]]
            v2pos = verts_info.init_pos[face[1]]
            v3pos = verts_info.init_pos[face[2]]

            # Compute the face normal
            n = (v2pos - v1pos).cross(v3pos - v1pos)
            n = n.normalized()
            n = gu.ti_transform_by_quat(n, quat)

            gjk_state.contact_normals[i_b, n_normals].normal = n
            gjk_state.contact_normals[i_b, n_normals].id = i_f
            n_normals += 1

            if dim == 3:
                break
            elif dim == 2:
                if n_normals == 2:
                    break
            else:
                if n_normals == gjk_info.max_contact_polygon_verts[None]:
                    break

    return n_normals


@ti.func
def func_box_face_local(
    geoms_info: array_class.GeomsInfo,
    gjk_state: array_class.GJKState,
    i_g,
    pos: ti.types.vector(3, dtype=gs.ti_float),
    quat: ti.types.vector(4, dtype=gs.ti_float),
    i_b,
    i_o,
    face_idx,
):
    """
    Thread-local version of func_box_face.

    Get the face vertices of the box geometry using thread-local pos/quat.

    Thread-safety note: Geometry index `i_g` is only used for read-only metadata access
    (geometry size data). It does not access `geoms_state.pos` or `geoms_state.quat`.

    Args:
        geoms_info: Geometry information
        gjk_state: GJK algorithm state (for storing contact faces)
        i_g: Geometry index (for metadata only)
        pos: Thread-local position of the geometry
        quat: Thread-local quaternion of the geometry
        i_b: Batch/environment index
        i_o: Object index (0 or 1) - determines which vert field to write to
        face_idx: Face index (0-5 for box faces)

    Returns:
        Number of face vertices (4 for valid face, 0 otherwise)
    """
    g_size_x = geoms_info.data[i_g][0]
    g_size_y = geoms_info.data[i_g][1]
    g_size_z = geoms_info.data[i_g][2]

    # Axis to fix, 0: x, 1: y, 2: z
    axis = face_idx // 2
    # Side of the fixed axis, 1: positive, -1: negative
    side = 1 - 2 * (face_idx & 1)

    nface = 4 if face_idx >= 0 and face_idx < 6 else 0

    vs = ti.Vector([0.0 for _ in range(3 * 4)], dt=gs.ti_float)
    if nface:
        for i in ti.static(range(4)):
            b0 = i & 1
            b1 = i >> 1
            # +1, +1, -1, -1
            su = 1 - 2 * b1
            # +1, -1, -1, +1
            sv = 1 - 2 * (b0 ^ b1)

            # Flip sv based on [side]
            sv = sv * side

            s = gs.ti_vec3(0, 0, 0)
            s[axis] = side
            s[(axis + 1) % 3] = su
            s[(axis + 2) % 3] = sv

            vs[3 * i + 0] = s[0] * g_size_x
            vs[3 * i + 1] = s[1] * g_size_y
            vs[3 * i + 2] = s[2] * g_size_z

    # Transform the vertices to the global coordinates
    for i in range(nface):
        v = gs.ti_vec3(vs[3 * i + 0], vs[3 * i + 1], vs[3 * i + 2]) * 0.5
        v = gu.ti_transform_by_trans_quat(v, pos, quat)
        if i_o == 0:
            gjk_state.contact_faces[i_b, i].vert1 = v
        else:
            gjk_state.contact_faces[i_b, i].vert2 = v

    return nface


@ti.func
def func_potential_box_normals_local(
    geoms_info: array_class.GeomsInfo,
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    i_g,
    quat: ti.types.vector(4, dtype=gs.ti_float),
    i_b,
    dim,
    v1,
    v2,
    v3,
    dir,
):
    """
    Thread-local version of func_potential_box_normals.

    For a simplex defined on a box with three vertices [v1, v2, v3], we find which face normals are potentially
    related to the simplex, using thread-local quat.

    If the simplex is a triangle, at most one face normal is related.
    If the simplex is a line, at most two face normals are related.
    If the simplex is a point, at most three face normals are related.

    We identify related face normals to the simplex by checking the vertex indices of the simplex.

    Thread-safety note: Geometry index `i_g` is only used for read-only metadata access
    (vertex start index). It does not access `geoms_state.pos` or `geoms_state.quat`.
    Note that this function only uses quat (not pos) since face normals are orientation-dependent
    but not position-dependent.

    Args:
        geoms_info: Geometry information
        gjk_state: GJK algorithm state (for storing contact normals)
        gjk_info: GJK algorithm parameters
        i_g: Geometry index (for metadata only)
        quat: Thread-local quaternion of the geometry
        i_b: Batch/environment index
        dim: Dimension of the simplex (1=point, 2=line, 3=triangle)
        v1: First vertex index
        v2: Second vertex index
        v3: Third vertex index
        dir: Collision direction vector

    Returns:
        Number of face normals found
    """
    # Change to local vertex indices
    v1 -= geoms_info.vert_start[i_g]
    v2 -= geoms_info.vert_start[i_g]
    v3 -= geoms_info.vert_start[i_g]

    # Number of potential face normals
    n_normals = 0

    # Fallback if the simplex is degenerate
    is_degenerate_simplex = False

    c = 0
    xyz = gs.ti_ivec3(0, 0, 0)
    for i in range(3):
        # 1 when every vertex has positive xyz coordinate,
        # -1 when every vertex has negative xyz coordinate,
        # 0 when vertices are mixed
        # Import func_cmp_bit from parent module
        xyz[i] = multi_contact.func_cmp_bit(v1, v2, v3, dim, i)

    for i in range(1 if dim == 3 else 3):
        # Determine the normal vector in the local space
        local_n = gs.ti_vec3(xyz[0], xyz[1], xyz[2])
        w = 1

        if dim == 2:
            w = xyz[i]

        if dim == 2 or dim == 1:
            local_n = gs.ti_vec3(0, 0, 0)
            local_n[i] = xyz[i]

        global_n = gu.ti_transform_by_quat(local_n, quat)

        if dim == 3:
            gjk_state.contact_normals[i_b, 0].normal = global_n

            # Note that only one of [x, y, z] could be non-zero, because the triangle is on the box face.
            sgn = xyz.sum()
            for j in range(3):
                if xyz[j]:
                    gjk_state.contact_normals[i_b, c].id = j * 2
                    c += 1

            if sgn == -1:
                # Flip if needed
                gjk_state.contact_normals[i_b, 0].id = gjk_state.contact_normals[i_b, 0].id + 1

        elif dim == 2:
            if w:
                if (i == 0) or (i == 1):
                    gjk_state.contact_normals[i_b, c].normal = global_n
                else:
                    gjk_state.contact_normals[i_b, 1].normal = global_n

                for j in range(3):
                    if i == j:
                        gjk_state.contact_normals[i_b, c].id = j * 2 if xyz[j] > 0 else j * 2 + 1
                        break

                c += 1

        elif dim == 1:
            gjk_state.contact_normals[i_b, c].normal = global_n

            for j in range(3):
                if i == j:
                    gjk_state.contact_normals[i_b, c].id = j * 2 if xyz[j] > 0 else j * 2 + 1
                    break
            c += 1

    # Check [c] for detecting degenerate cases
    if dim == 3:
        # [c] should be 1 in normal case, but if triangle does not lie on the box face, it could be other values.
        n_normals = 1
        is_degenerate_simplex = c != 1
    elif dim == 2:
        # [c] should be 2 in normal case, but if edge does not lie on the box edge, it could be other values.
        n_normals = 2
        is_degenerate_simplex = c != 2
    elif dim == 1:
        n_normals = 3
        is_degenerate_simplex = False

    # If the simplex was degenerate, find the face normal using collision normal
    if is_degenerate_simplex:
        n_normals = (
            1
            if func_box_normal_from_collision_normal_local(gjk_state, gjk_info, i_g, quat, i_b, dir)
            == multi_contact.RETURN_CODE.SUCCESS
            else 0
        )

    return n_normals

