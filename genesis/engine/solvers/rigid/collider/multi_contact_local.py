"""
Thread-local versions of multi-contact detection functions.

This module provides versions of multi-contact functions that accept pos/quat as direct
parameters instead of reading from geoms_state, enabling race-free multi-contact
detection when parallelizing across collision pairs within the same environment.
"""

import gstaichi as ti

import genesis as gs
import genesis.utils.geom as gu
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

