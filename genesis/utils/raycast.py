import gstaichi as ti

import genesis as gs


@ti.func
def ray_triangle_intersection(ray_start, ray_dir, v0, v1, v2):
    """
    Moller-Trumbore ray-triangle intersection.

    Returns: vec4(t, u, v, hit) where hit=1.0 if intersection found, 0.0 otherwise
    """
    result = ti.Vector.zero(gs.ti_float, 4)

    edge1 = v1 - v0
    edge2 = v2 - v0

    # Begin calculating determinant - also used to calculate u parameter
    h = ray_dir.cross(edge2)
    a = edge1.dot(h)

    # Check all conditions in sequence without early returns
    valid = True

    t = gs.ti_float(0.0)
    u = gs.ti_float(0.0)
    v = gs.ti_float(0.0)
    f = gs.ti_float(0.0)
    s = ti.Vector.zero(gs.ti_float, 3)
    q = ti.Vector.zero(gs.ti_float, 3)

    # If determinant is near zero, ray lies in plane of triangle
    if ti.abs(a) < gs.EPS:
        valid = False

    if valid:
        f = 1.0 / a
        s = ray_start - v0
        u = f * s.dot(h)

        if u < 0.0 or u > 1.0:
            valid = False

    if valid:
        q = s.cross(edge1)
        v = f * ray_dir.dot(q)

        if v < 0.0 or u + v > 1.0:
            valid = False

    if valid:
        # At this stage we can compute t to find out where the intersection point is on the line
        t = f * edge2.dot(q)

        # Ray intersection
        if t <= gs.EPS:
            valid = False

    if valid:
        result = ti.math.vec4(t, u, v, 1.0)

    return result


@ti.func
def ray_aabb_intersection(ray_start, ray_dir, aabb_min, aabb_max):
    """
    Fast ray-AABB intersection test.
    Returns the t value of intersection, or -1.0 if no intersection.
    """
    result = -1.0

    # Use the slab method for ray-AABB intersection
    sign = ti.select(ray_dir >= 0.0, 1.0, -1.0)
    ray_dir = sign * ti.max(ti.abs(ray_dir), gs.EPS)
    inv_dir = 1.0 / ray_dir

    t1 = (aabb_min - ray_start) * inv_dir
    t2 = (aabb_max - ray_start) * inv_dir

    tmin = ti.min(t1, t2)
    tmax = ti.max(t1, t2)

    t_near = ti.max(tmin.x, tmin.y, tmin.z, 0.0)
    t_far = ti.min(tmax.x, tmax.y, tmax.z)

    # Check if ray intersects AABB
    if t_near <= t_far:
        result = t_near

    return result


@ti.kernel
def kernel_update_aabbs(
    free_verts_state: ti.template(),
    fixed_verts_state: ti.template(),
    verts_info: ti.template(),
    faces_info: ti.template(),
    # FIXME: can't import array_class since it is before gs.init
    # free_verts_state: array_class.VertsState,
    # fixed_verts_state: array_class.VertsState,
    # verts_info: array_class.VertsInfo,
    # faces_info: array_class.FacesInfo,
    aabb_state: ti.template(),
):
    for i_b, i_f in ti.ndrange(free_verts_state.pos.shape[1], faces_info.verts_idx.shape[0]):
        aabb_state.aabbs[i_b, i_f].min.fill(ti.math.inf)
        aabb_state.aabbs[i_b, i_f].max.fill(-ti.math.inf)

        for i in ti.static(range(3)):
            i_v = faces_info.verts_idx[i_f][i]
            i_fv = verts_info.verts_state_idx[i_v]
            if verts_info.is_fixed[i_v]:
                pos_v = fixed_verts_state.pos[i_fv]
                aabb_state.aabbs[i_b, i_f].min = ti.min(aabb_state.aabbs[i_b, i_f].min, pos_v)
                aabb_state.aabbs[i_b, i_f].max = ti.max(aabb_state.aabbs[i_b, i_f].max, pos_v)
            else:
                pos_v = free_verts_state.pos[i_fv, i_b]
                aabb_state.aabbs[i_b, i_f].min = ti.min(aabb_state.aabbs[i_b, i_f].min, pos_v)
                aabb_state.aabbs[i_b, i_f].max = ti.max(aabb_state.aabbs[i_b, i_f].max, pos_v)
