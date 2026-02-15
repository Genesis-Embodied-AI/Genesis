"""
Utility functions for GJK/EPA algorithms.

This module contains shared utility functions used by both GJK and EPA algorithms.
"""

import quadrants as ti

import genesis as gs
import genesis.utils.array_class as array_class
from .constants import RETURN_CODE
from .utils import func_det3


@ti.func
def func_ray_triangle_intersection(
    ray_v1,
    ray_v2,
    tri_v1,
    tri_v2,
    tri_v3,
):
    """
    Check if the ray intersects the triangle.

    Returns
    -------
    int
        True if the ray intersects the triangle, otherwise False.
    """
    ray = ray_v2 - ray_v1

    # Signed volumes of the tetrahedrons formed by the ray and triangle edges
    vols = gs.ti_vec3(0.0, 0.0, 0.0)
    for i in range(3):
        v1, v2 = gs.ti_vec3(0.0, 0.0, 0.0), gs.ti_vec3(0.0, 0.0, 0.0)
        if i == 0:
            v1, v2 = tri_v1 - ray_v1, tri_v2 - ray_v1
        if i == 1:
            v1, v2 = tri_v2 - ray_v1, tri_v3 - ray_v1
        elif i == 2:
            v1, v2 = tri_v3 - ray_v1, tri_v1 - ray_v1
        vols[i] = func_det3(v1, v2, ray)

    return (vols >= 0.0).all() or (vols <= 0.0).all()


@ti.func
def func_triangle_affine_coords(
    point,
    tri_v1,
    tri_v2,
    tri_v3,
):
    """
    Compute the affine coordinates of the point with respect to the triangle.
    """
    # Compute minors of the triangle vertices
    ms = gs.ti_vec3(0.0, 0.0, 0.0)
    for i in ti.static(range(3)):
        i1, i2 = (i + 1) % 3, (i + 2) % 3
        if i == 1:
            i1, i2 = i2, i1

        ms[i] = (
            tri_v2[i1] * tri_v3[i2]
            - tri_v2[i2] * tri_v3[i1]
            - tri_v1[i1] * tri_v3[i2]
            + tri_v1[i2] * tri_v3[i1]
            + tri_v1[i1] * tri_v2[i2]
            - tri_v1[i2] * tri_v2[i1]
        )

    # Exclude one of the axes with the largest projection using the minors of the above linear system.
    m_max = gs.ti_float(0.0)
    i_x, i_y = gs.ti_int(0), gs.ti_int(0)
    absms = ti.abs(ms)
    for i in range(3):
        if absms[i] >= absms[(i + 1) % 3] and absms[i] >= absms[(i + 2) % 3]:
            # Remove the i-th row
            m_max = ms[i]
            i_x, i_y = (i + 1) % 3, (i + 2) % 3
            if i == 1:
                i_x, i_y = i_y, i_x
            break

    cs = gs.ti_vec3(0.0, 0.0, 0.0)
    for i in range(3):
        tv1, tv2 = tri_v2, tri_v3
        if i == 1:
            tv1, tv2 = tri_v3, tri_v1
        elif i == 2:
            tv1, tv2 = tri_v1, tri_v2

        # Corresponds to the signed area of 2-simplex (triangle): (point, tv1, tv2)
        cs[i] = (
            point[i_x] * tv1[i_y]
            + point[i_y] * tv2[i_x]
            + tv1[i_x] * tv2[i_y]
            - point[i_x] * tv2[i_y]
            - point[i_y] * tv1[i_x]
            - tv2[i_x] * tv1[i_y]
        )

    # Affine coordinates are computed as: [ l1, l2, l3 ] = [ C1 / m_max, C2 / m_max, C3 / m_max ]
    return cs / m_max


@ti.func
def func_point_triangle_intersection(
    gjk_info: array_class.GJKInfo,
    point,
    tri_v1,
    tri_v2,
    tri_v3,
):
    """
    Check if the point is inside the triangle.
    """
    is_inside = False
    # Compute the affine coordinates of the point with respect to the triangle
    _lambda = func_triangle_affine_coords(point, tri_v1, tri_v2, tri_v3)

    # If any of the affine coordinates is negative, the point is outside the triangle
    if (_lambda >= 0).all():
        # Check if the point predicted by the affine coordinates is equal to the point itself
        pred = tri_v1 * _lambda[0] + tri_v2 * _lambda[1] + tri_v3 * _lambda[2]
        diff = pred - point
        is_inside = diff.norm_sqr() < gjk_info.FLOAT_MIN_SQ[None]

    return is_inside


@ti.func
def func_point_plane_same_side(
    point,
    plane_v1,
    plane_v2,
    plane_v3,
):
    """
    Check if the point is on the same side of the plane as the origin.
    """
    # Compute the normal of the plane
    edge1 = plane_v2 - plane_v1
    edge2 = plane_v3 - plane_v1
    normal = edge1.cross(edge2)

    diff1 = point - plane_v1
    dot1 = normal.dot(diff1)

    # origin - plane_v1
    diff2 = -plane_v1
    dot2 = normal.dot(diff2)

    return RETURN_CODE.SUCCESS if dot1 * dot2 > 0 else RETURN_CODE.FAIL


@ti.func
def func_origin_tetra_intersection(
    tet_v1,
    tet_v2,
    tet_v3,
    tet_v4,
):
    """
    Check if the origin is inside the tetrahedron.
    """
    flag = RETURN_CODE.SUCCESS
    for i in range(4):
        v1, v2, v3, v4 = tet_v1, tet_v2, tet_v3, tet_v4
        if i == 1:
            v1, v2, v3, v4 = tet_v2, tet_v3, tet_v4, tet_v1
        elif i == 2:
            v1, v2, v3, v4 = tet_v3, tet_v4, tet_v1, tet_v2
        elif i == 3:
            v1, v2, v3, v4 = tet_v4, tet_v1, tet_v2, tet_v3
        flag = func_point_plane_same_side(v1, v2, v3, v4)
        if flag == RETURN_CODE.FAIL:
            break
    return flag


@ti.func
def func_project_origin_to_plane(
    gjk_info: array_class.GJKInfo,
    v1,
    v2,
    v3,
):
    """
    Project the origin onto the plane defined by the simplex vertices.
    """
    point, flag = gs.ti_vec3(0, 0, 0), RETURN_CODE.SUCCESS

    d21 = v2 - v1
    d31 = v3 - v1
    d32 = v3 - v2

    for i in range(3):
        n = gs.ti_vec3(0, 0, 0)
        v = gs.ti_vec3(0, 0, 0)
        if i == 0:
            # Normal = (v1 - v2) x (v3 - v2)
            n = d32.cross(d21)
            v = v2
        elif i == 1:
            # Normal = (v2 - v1) x (v3 - v1)
            n = d21.cross(d31)
            v = v1
        else:
            # Normal = (v1 - v3) x (v2 - v3)
            n = d31.cross(d32)
            v = v3
        nv = n.dot(v)
        nn = n.norm_sqr()
        if nn == 0:
            # Zero normal, cannot project.
            flag = RETURN_CODE.FAIL
            break
        elif nn > gjk_info.FLOAT_MIN[None]:
            point = n * (nv / nn)
            flag = RETURN_CODE.SUCCESS
            break

        # Last fallback if no valid normal was found
        if i == 2:
            # If the normal is still unreliable, cannot project.
            if nn < gjk_info.FLOAT_MIN[None]:
                flag = RETURN_CODE.FAIL
            else:
                point = n * (nv / nn)
                flag = RETURN_CODE.SUCCESS

    return point, flag
