import gstaichi as ti

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class
from . import mpr_local, support_field


class MPR:
    def __init__(self, rigid_solver, is_active: bool = True):
        self._solver = rigid_solver
        self._is_active = is_active

        self._mpr_info = array_class.get_mpr_info(
            # It has been observed in practice that increasing this threshold makes collision detection instable,
            # which is surprising since 1e-9 is above single precision (which has only 7 digits of precision).
            CCD_EPS=1e-9 if gs.ti_float == ti.f32 else 1e-10,
            CCD_TOLERANCE=1e-6,
            CCD_ITERATIONS=50,
        )
        self._mpr_state = array_class.get_mpr_state(self._solver._B)

    def reset(self):
        pass

    @property
    def is_active(self):
        return self._is_active


@ti.kernel
def clear(mpr_state: ti.template()):
    mpr_state.simplex_size.fill(0)


@ti.func
def func_point_in_geom_aabb(geoms_state: array_class.GeomsState, point, i_g, i_b):
    return (point < geoms_state.aabb_max[i_g, i_b]).all() and (point > geoms_state.aabb_min[i_g, i_b]).all()


@ti.func
def func_is_geom_aabbs_overlap(geoms_state: array_class.GeomsState, i_ga, i_gb, i_b):
    return not (
        (geoms_state.aabb_max[i_ga, i_b] <= geoms_state.aabb_min[i_gb, i_b]).any()
        or (geoms_state.aabb_min[i_ga, i_b] >= geoms_state.aabb_max[i_gb, i_b]).any()
    )


@ti.func
def func_find_intersect_midpoint(geoms_state: array_class.GeomsState, i_ga, i_gb, i_b):
    # return the center of the intersecting AABB of AABBs of two geoms
    intersect_lower = ti.max(geoms_state.aabb_min[i_ga, i_b], geoms_state.aabb_min[i_gb, i_b])
    intersect_upper = ti.min(geoms_state.aabb_max[i_ga, i_b], geoms_state.aabb_max[i_gb, i_b])
    return 0.5 * (intersect_lower + intersect_upper)


@ti.func
def mpr_swap(mpr_state: array_class.MPRState, i, j, i_ga, i_gb, i_b):
    mpr_state.simplex_support.v1[i, i_b], mpr_state.simplex_support.v1[j, i_b] = (
        mpr_state.simplex_support.v1[j, i_b],
        mpr_state.simplex_support.v1[i, i_b],
    )
    mpr_state.simplex_support.v2[i, i_b], mpr_state.simplex_support.v2[j, i_b] = (
        mpr_state.simplex_support.v2[j, i_b],
        mpr_state.simplex_support.v2[i, i_b],
    )
    mpr_state.simplex_support.v[i, i_b], mpr_state.simplex_support.v[j, i_b] = (
        mpr_state.simplex_support.v[j, i_b],
        mpr_state.simplex_support.v[i, i_b],
    )


@ti.func
def mpr_point_segment_dist2(mpr_info: array_class.MPRInfo, P, A, B):
    AB = B - A
    AP = P - A
    AB_AB = AB.dot(AB)
    AP_AB = AP.dot(AB)
    t = AP_AB / AB_AB
    if t < mpr_info.CCD_EPS[None]:
        t = gs.ti_float(0.0)
    elif t > 1.0 - mpr_info.CCD_EPS[None]:
        t = gs.ti_float(1.0)
    Q = A + AB * t

    return (P - Q).norm_sqr(), Q


@ti.func
def mpr_point_tri_depth(mpr_info: array_class.MPRInfo, P, x0, B, C):
    d1 = B - x0
    d2 = C - x0
    a = x0 - P
    v = d1.dot(d1)
    w = d2.dot(d2)
    p = a.dot(d1)
    q = a.dot(d2)
    r = d1.dot(d2)

    d = w * v - r * r
    dist = s = t = gs.ti_float(0.0)
    pdir = gs.ti_vec3([0.0, 0.0, 0.0])
    if ti.abs(d) < mpr_info.CCD_EPS[None]:
        s = t = -1.0
    else:
        s = (q * r - w * p) / d
        t = (-s * r - q) / w

    if (
        (s > -mpr_info.CCD_EPS[None])
        and (s < 1.0 + mpr_info.CCD_EPS[None])
        and (t > -mpr_info.CCD_EPS[None])
        and (t < 1.0 + mpr_info.CCD_EPS[None])
        and (t + s < 1.0 + mpr_info.CCD_EPS[None])
    ):
        pdir = x0 + d1 * s + d2 * t
        dist = (P - pdir).norm_sqr()
    else:
        dist, pdir = mpr_point_segment_dist2(mpr_info, P, x0, B)
        dist2, pdir2 = mpr_point_segment_dist2(mpr_info, P, x0, C)
        if dist2 < dist:
            dist = dist2
            pdir = pdir2

        dist2, pdir2 = mpr_point_segment_dist2(mpr_info, P, B, C)
        if dist2 < dist:
            dist = dist2
            pdir = pdir2

    return ti.sqrt(dist), pdir


@ti.func
def mpr_portal_dir(mpr_state: array_class.MPRState, i_ga, i_gb, i_b):
    v2v1 = mpr_state.simplex_support.v[2, i_b] - mpr_state.simplex_support.v[1, i_b]
    v3v1 = mpr_state.simplex_support.v[3, i_b] - mpr_state.simplex_support.v[1, i_b]
    direction = v2v1.cross(v3v1).normalized()
    return direction


@ti.func
def mpr_portal_encapsules_origin(
    mpr_state: array_class.MPRState, mpr_info: array_class.MPRInfo, direction, i_ga, i_gb, i_b
):
    dot = mpr_state.simplex_support.v[1, i_b].dot(direction)
    return dot > -mpr_info.CCD_EPS[None]


@ti.func
def mpr_portal_can_encapsule_origin(mpr_info: array_class.MPRInfo, v, direction):
    dot = v.dot(direction)
    return dot > -mpr_info.CCD_EPS[None]


@ti.func
def mpr_portal_reach_tolerance(
    mpr_state: array_class.MPRState, mpr_info: array_class.MPRInfo, v, direction, i_ga, i_gb, i_b
):
    dv1 = mpr_state.simplex_support.v[1, i_b].dot(direction)
    dv2 = mpr_state.simplex_support.v[2, i_b].dot(direction)
    dv3 = mpr_state.simplex_support.v[3, i_b].dot(direction)
    dv4 = v.dot(direction)
    dot1 = ti.min(dv4 - dv1, dv4 - dv2, dv4 - dv3)
    return dot1 < mpr_info.CCD_TOLERANCE[None] + mpr_info.CCD_EPS[None] * ti.max(1.0, dot1)


@ti.func
def support_driver(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    direction,
    i_g,
    i_b,
):
    pos = geoms_state.pos[i_g, i_b]
    quat = geoms_state.quat[i_g, i_b]
    return mpr_local.support_driver_local(
        geoms_info, collider_state, collider_static_config, support_field_info, direction, i_g, i_b, pos, quat
    )


@ti.func
def mpr_find_pos(
    static_rigid_sim_config: ti.template(),
    mpr_state: array_class.MPRState,
    mpr_info: array_class.MPRInfo,
    i_ga,
    i_gb,
    i_b,
):
    b = ti.Vector([0.0, 0.0, 0.0, 0.0], dt=gs.ti_float)

    # Only look into the direction of the portal for consistency with penetration depth computation
    if ti.static(static_rigid_sim_config.enable_mujoco_compatibility):
        for i in range(4):
            i1, i2, i3 = (i % 2) + 1, (i + 2) % 4, 3 * ((i + 1) % 2)
            vec = mpr_state.simplex_support.v[i1, i_b].cross(mpr_state.simplex_support.v[i2, i_b])
            b[i] = vec.dot(mpr_state.simplex_support.v[i3, i_b]) * (1 - 2 * (((i + 1) // 2) % 2))

    sum_ = b.sum()

    if sum_ < mpr_info.CCD_EPS[None]:
        direction = mpr_portal_dir(mpr_state, i_ga, i_gb, i_b)
        b[0] = 0.0
        for i in range(1, 4):
            i1, i2 = i % 3 + 1, (i + 1) % 3 + 1
            vec = mpr_state.simplex_support.v[i1, i_b].cross(mpr_state.simplex_support.v[i2, i_b])
            b[i] = vec.dot(direction)
        sum_ = b.sum()

    p1 = gs.ti_vec3([0.0, 0.0, 0.0])
    p2 = gs.ti_vec3([0.0, 0.0, 0.0])
    for i in range(4):
        p1 += b[i] * mpr_state.simplex_support.v1[i, i_b]
        p2 += b[i] * mpr_state.simplex_support.v2[i, i_b]

    return (0.5 / sum_) * (p1 + p2)


@ti.func
def mpr_find_penetr_touch(mpr_state: array_class.MPRState, i_ga, i_gb, i_b):
    is_col = True
    penetration = gs.ti_float(0.0)
    normal = -mpr_state.simplex_support.v[0, i_b].normalized()
    pos = (mpr_state.simplex_support.v1[1, i_b] + mpr_state.simplex_support.v2[1, i_b]) * 0.5
    return is_col, normal, penetration, pos


@ti.func
def mpr_find_penetr_segment(mpr_state: array_class.MPRState, i_ga, i_gb, i_b):
    is_col = True
    penetration = mpr_state.simplex_support.v[1, i_b].norm()
    normal = -mpr_state.simplex_support.v[1, i_b].normalized()
    pos = (mpr_state.simplex_support.v1[1, i_b] + mpr_state.simplex_support.v2[1, i_b]) * 0.5

    return is_col, normal, penetration, pos


@ti.func
def mpr_expand_portal(mpr_state: array_class.MPRState, v, v1, v2, i_ga, i_gb, i_b):
    v4v0 = v.cross(mpr_state.simplex_support.v[0, i_b])
    dot = mpr_state.simplex_support.v[1, i_b].dot(v4v0)

    i_s = gs.ti_int(0)
    if dot > 0:
        dot = mpr_state.simplex_support.v[2, i_b].dot(v4v0)
        i_s = 1 if dot > 0 else 3

    else:
        dot = mpr_state.simplex_support.v[3, i_b].dot(v4v0)
        i_s = 2 if dot > 0 else 1

    mpr_state.simplex_support.v1[i_s, i_b] = v1
    mpr_state.simplex_support.v2[i_s, i_b] = v2
    mpr_state.simplex_support.v[i_s, i_b] = v



from genesis.utils.deprecated_module_wrapper import create_virtual_deprecated_module

create_virtual_deprecated_module(__name__, "genesis.engine.solvers.rigid.mpr_decomp")
