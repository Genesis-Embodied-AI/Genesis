import numpy as np
import gstaichi as ti
import torch
from dataclasses import dataclass

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class
import genesis.engine.solvers.rigid.support_field_decomp as support_field


@ti.data_oriented
class MPR:

    @ti.data_oriented
    class MPRStaticConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    # @dataclass(frozen=True)
    # class MPRStaticConfig:
    #     # store static arguments here
    #     CCD_EPS: float = 1e-9
    #     CCD_TOLERANCE: float = 1e-6
    #     CCD_ITERATIONS: int = 50

    def __init__(self, rigid_solver):
        self._solver = rigid_solver
        self._mpr_static_config = MPR.MPRStaticConfig(
            # It has been observed in practice that increasing this threshold makes collision detection instable,
            # which is surprising since 1e-9 is above single precision (which has only 7 digits of precision).
            CCD_EPS=1e-9 if gs.ti_float == ti.f32 else 1e-10,
            CCD_TOLERANCE=1e-6,
            CCD_ITERATIONS=50,
        )
        self.init_state()

    def init_state(self):
        self._mpr_state = array_class.get_mpr_state(self._solver._batch_shape)

    def reset(self):
        pass


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
def mpr_point_segment_dist2(mpr_static_config: ti.template(), P, A, B):
    AB = B - A
    AP = P - A
    AB_AB = AB.dot(AB)
    AP_AB = AP.dot(AB)
    t = AP_AB / AB_AB
    if t < mpr_static_config.CCD_EPS:
        t = gs.ti_float(0.0)
    elif t > 1.0 - mpr_static_config.CCD_EPS:
        t = gs.ti_float(1.0)
    Q = A + AB * t

    return (P - Q).norm_sqr(), Q


@ti.func
def mpr_point_tri_depth(mpr_static_config: ti.template(), P, x0, B, C):
    d1 = B - x0
    d2 = C - x0
    a = x0 - P
    u = a.dot(a)
    v = d1.dot(d1)
    w = d2.dot(d2)
    p = a.dot(d1)
    q = a.dot(d2)
    r = d1.dot(d2)

    d = w * v - r * r
    dist = s = t = gs.ti_float(0.0)
    pdir = gs.ti_vec3([0.0, 0.0, 0.0])
    if ti.abs(d) < mpr_static_config.CCD_EPS:
        s = t = -1.0
    else:
        s = (q * r - w * p) / d
        t = (-s * r - q) / w

    if (
        (s > -mpr_static_config.CCD_EPS)
        and (s < 1.0 + mpr_static_config.CCD_EPS)
        and (t > -mpr_static_config.CCD_EPS)
        and (t < 1.0 + mpr_static_config.CCD_EPS)
        and (t + s < 1.0 + mpr_static_config.CCD_EPS)
    ):
        pdir = x0 + d1 * s + d2 * t
        dist = (P - pdir).norm_sqr()
    else:
        dist, pdir = mpr_point_segment_dist2(mpr_static_config, P, x0, B)
        dist2, pdir2 = mpr_point_segment_dist2(mpr_static_config, P, x0, C)
        if dist2 < dist:
            dist = dist2
            pdir = pdir2

        dist2, pdir2 = mpr_point_segment_dist2(mpr_static_config, P, B, C)
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
    mpr_state: array_class.MPRState, mpr_static_config: ti.template(), direction, i_ga, i_gb, i_b
):
    dot = mpr_state.simplex_support.v[1, i_b].dot(direction)
    return dot > -mpr_static_config.CCD_EPS


@ti.func
def mpr_portal_can_encapsule_origin(mpr_static_config: ti.template(), v, direction):
    dot = v.dot(direction)
    return dot > -mpr_static_config.CCD_EPS


@ti.func
def mpr_portal_reach_tolerance(
    mpr_state: array_class.MPRState, mpr_static_config: ti.template(), v, direction, i_ga, i_gb, i_b
):
    dv1 = mpr_state.simplex_support.v[1, i_b].dot(direction)
    dv2 = mpr_state.simplex_support.v[2, i_b].dot(direction)
    dv3 = mpr_state.simplex_support.v[3, i_b].dot(direction)
    dv4 = v.dot(direction)
    dot1 = ti.min(dv4 - dv1, dv4 - dv2, dv4 - dv3)
    return dot1 < mpr_static_config.CCD_TOLERANCE + mpr_static_config.CCD_EPS * ti.max(1.0, dot1)


@ti.func
def support_driver(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    direction,
    i_g,
    i_b,
):
    v = ti.Vector.zero(gs.ti_float, 3)
    geom_type = geoms_info.type[i_g]
    if geom_type == gs.GEOM_TYPE.SPHERE:
        v, v_, vid = support_field._func_support_sphere(geoms_state, geoms_info, direction, i_g, i_b, False)
    elif geom_type == gs.GEOM_TYPE.ELLIPSOID:
        v = support_field._func_support_ellipsoid(geoms_state, geoms_info, direction, i_g, i_b)
    elif geom_type == gs.GEOM_TYPE.CAPSULE:
        v = support_field._func_support_capsule(geoms_state, geoms_info, direction, i_g, i_b, False)
    elif geom_type == gs.GEOM_TYPE.BOX:
        v, v_, vid = support_field._func_support_box(geoms_state, geoms_info, direction, i_g, i_b)
    elif geom_type == gs.GEOM_TYPE.TERRAIN:
        if ti.static(collider_static_config.has_terrain):
            v, _ = support_field._func_support_prism(collider_state, direction, i_g, i_b)
    else:
        v, v_, vid = support_field._func_support_world(
            geoms_state, geoms_info, support_field_info, support_field_static_config, direction, i_g, i_b
        )
    return v


@ti.func
def compute_support(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    direction,
    i_ga,
    i_gb,
    i_b,
):
    v1 = support_driver(
        geoms_state,
        geoms_info,
        collider_state,
        collider_info,
        collider_static_config,
        support_field_info,
        support_field_static_config,
        direction,
        i_ga,
        i_b,
    )
    v2 = support_driver(
        geoms_state,
        geoms_info,
        collider_state,
        collider_info,
        collider_static_config,
        support_field_info,
        support_field_static_config,
        -direction,
        i_gb,
        i_b,
    )

    v = v1 - v2
    return v, v1, v2


@ti.func
def func_geom_support(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    direction,
    i_g,
    i_b,
):
    g_pos = geoms_state.pos[i_g, i_b]
    g_quat = geoms_state.quat[i_g, i_b]
    direction_in_init_frame = gu.ti_inv_transform_by_quat(direction, g_quat)

    dot_max = gs.ti_float(-1e10)
    v = ti.Vector.zero(gs.ti_float, 3)
    vid = 0

    for i_v in range(geoms_info.vert_start[i_g], geoms_info.vert_end[i_g]):
        pos = verts_info.init_pos[i_v]
        dot = pos.dot(direction_in_init_frame)
        if dot > dot_max:
            v = pos
            dot_max = dot
            vid = i_v
    v_ = gu.ti_transform_by_trans_quat(v, g_pos, g_quat)

    return v_, vid


@ti.func
def mpr_refine_portal(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: ti.template(),
    mpr_state: array_class.MPRState,
    mpr_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
):
    ret = 1
    while True:
        direction = mpr_portal_dir(mpr_state, i_ga, i_gb, i_b)

        if mpr_portal_encapsules_origin(mpr_state, mpr_static_config, direction, i_ga, i_gb, i_b):
            ret = 0
            break

        v, v1, v2 = compute_support(
            geoms_state,
            geoms_info,
            collider_state,
            collider_info,
            collider_static_config,
            support_field_info,
            support_field_static_config,
            direction,
            i_ga,
            i_gb,
            i_b,
        )

        if not mpr_portal_can_encapsule_origin(mpr_static_config, v, direction) or mpr_portal_reach_tolerance(
            mpr_state, mpr_static_config, v, direction, i_ga, i_gb, i_b
        ):
            ret = -1
            break

        mpr_expand_portal(mpr_state, v, v1, v2, i_ga, i_gb, i_b)
    return ret


@ti.func
def mpr_find_pos(
    static_rigid_sim_config: ti.template(),
    mpr_state: array_class.MPRState,
    mpr_static_config: ti.template(),
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

    if sum_ < mpr_static_config.CCD_EPS:
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
def mpr_find_penetration(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    static_rigid_sim_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: ti.template(),
    mpr_state: array_class.MPRState,
    mpr_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
):
    iterations = 0

    is_col = False
    pos = gs.ti_vec3([0.0, 0.0, 0.0])
    normal = gs.ti_vec3([0.0, 0.0, 0.0])
    penetration = gs.ti_float(0.0)

    while True:
        direction = mpr_portal_dir(mpr_state, i_ga, i_gb, i_b)
        v, v1, v2 = compute_support(
            geoms_state,
            geoms_info,
            collider_state,
            collider_info,
            collider_static_config,
            support_field_info,
            support_field_static_config,
            direction,
            i_ga,
            i_gb,
            i_b,
        )
        if (
            mpr_portal_reach_tolerance(mpr_state, mpr_static_config, v, direction, i_ga, i_gb, i_b)
            or iterations > mpr_static_config.CCD_ITERATIONS
        ):
            # The contact point is defined as the projection of the origin onto the portal, i.e. the closest point
            # to the origin that lies inside the portal.
            # Let's consider the portal as an infinite plane rather than a face triangle. This makes sense because
            # the projection of the origin must be strictly included into the portal triangle for it to correspond
            # to the true penetration depth.
            # For reference about this propery, see 'Collision Handling with Variable-Step Integrators' Theorem 4.2:
            # https://modiasim.github.io/Modia3D.jl/resources/documentation/CollisionHandling_Neumayr_Otter_2017.pdf
            #
            # In theory, the center should have been shifted until to end up with the one and only portal satisfying
            # this condition. However, a native implementation of this process must be avoided because it would be
            # very costly. In practice, assuming the portal is infinite provides a decent approximation of the true
            # penetration depth (it is actually a lower-bound estimate according to Theorem 4.3) and normal without
            # requiring any additional computations.
            # See: https://github.com/danfis/libccd/issues/71#issuecomment-660415008
            #
            # An improved version of MPR has been proposed to find the right portal in an efficient way.
            # See: https://arxiv.org/pdf/2304.07357
            # Implementation: https://github.com/weigao95/mind-fcl/blob/main/include/fcl/cvx_collide/mpr.h
            #
            # The original paper introducing MPR algorithm is available here:
            # https://archive.org/details/game-programming-gems-7
            if ti.static(static_rigid_sim_config.enable_mujoco_compatibility):
                penetration, pdir = mpr_point_tri_depth(
                    mpr_static_config,
                    gs.ti_vec3([0.0, 0.0, 0.0]),
                    mpr_state.simplex_support.v[1, i_b],
                    mpr_state.simplex_support.v[2, i_b],
                    mpr_state.simplex_support.v[3, i_b],
                )
                normal = -pdir.normalized()
            else:
                penetration = direction.dot(mpr_state.simplex_support.v[1, i_b])
                normal = -direction

            is_col = True
            pos = mpr_find_pos(static_rigid_sim_config, mpr_state, mpr_static_config, i_ga, i_gb, i_b)
            break

        mpr_expand_portal(mpr_state, v, v1, v2, i_ga, i_gb, i_b)
        iterations += 1

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


@ti.func
def mpr_discover_portal(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: ti.template(),
    mpr_state: array_class.MPRState,
    mpr_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
    center_a,
    center_b,
):
    mpr_state.simplex_support.v1[0, i_b] = center_a
    mpr_state.simplex_support.v2[0, i_b] = center_b
    mpr_state.simplex_support.v[0, i_b] = center_a - center_b
    mpr_state.simplex_size[i_b] = 1

    if (ti.abs(mpr_state.simplex_support.v[0, i_b]) < mpr_static_config.CCD_EPS).all():
        mpr_state.simplex_support.v[0, i_b][0] += 10.0 * mpr_static_config.CCD_EPS

    direction = -mpr_state.simplex_support.v[0, i_b].normalized()

    v, v1, v2 = compute_support(
        geoms_state,
        geoms_info,
        collider_state,
        collider_info,
        collider_static_config,
        support_field_info,
        support_field_static_config,
        direction,
        i_ga,
        i_gb,
        i_b,
    )

    mpr_state.simplex_support.v1[1, i_b] = v1
    mpr_state.simplex_support.v2[1, i_b] = v2
    mpr_state.simplex_support.v[1, i_b] = v
    mpr_state.simplex_size[i_b] = 2

    dot = v.dot(direction)

    ret = 0
    if dot < mpr_static_config.CCD_EPS:
        ret = -1
    else:
        direction = mpr_state.simplex_support.v[0, i_b].cross(mpr_state.simplex_support.v[1, i_b])
        if direction.dot(direction) < mpr_static_config.CCD_EPS:
            if (ti.abs(mpr_state.simplex_support.v[1, i_b]) < mpr_static_config.CCD_EPS).all():
                ret = 1
            else:
                ret = 2
        else:
            direction = direction.normalized()
            v, v1, v2 = compute_support(
                geoms_state,
                geoms_info,
                collider_state,
                collider_info,
                collider_static_config,
                support_field_info,
                support_field_static_config,
                direction,
                i_ga,
                i_gb,
                i_b,
            )
            dot = v.dot(direction)
            if dot < mpr_static_config.CCD_EPS:
                ret = -1
            else:
                mpr_state.simplex_support.v1[2, i_b] = v1
                mpr_state.simplex_support.v2[2, i_b] = v2
                mpr_state.simplex_support.v[2, i_b] = v
                mpr_state.simplex_size[i_b] = 3

                va = mpr_state.simplex_support.v[1, i_b] - mpr_state.simplex_support.v[0, i_b]
                vb = mpr_state.simplex_support.v[2, i_b] - mpr_state.simplex_support.v[0, i_b]
                direction = va.cross(vb)
                direction = direction.normalized()

                dot = direction.dot(mpr_state.simplex_support.v[0, i_b])
                if dot > 0:
                    mpr_swap(mpr_state, 1, 2, i_ga, i_gb, i_b)
                    direction = -direction

                # FIXME: This algorithm may get stuck in an infinite loop if the actually penetration is smaller
                # then `CCD_EPS` and at least one of the center of each geometry is outside their convex hull.
                # Since this deadlock happens very rarely, a simple fix is to abord computation after a few trials.
                num_trials = gs.ti_int(0)
                while mpr_state.simplex_size[i_b] < 4:
                    v, v1, v2 = compute_support(
                        geoms_state,
                        geoms_info,
                        collider_state,
                        collider_info,
                        collider_static_config,
                        support_field_info,
                        support_field_static_config,
                        direction,
                        i_ga,
                        i_gb,
                        i_b,
                    )
                    dot = v.dot(direction)
                    if dot < mpr_static_config.CCD_EPS:
                        ret = -1
                        break

                    cont = False

                    va = mpr_state.simplex_support.v[1, i_b].cross(v)
                    dot = va.dot(mpr_state.simplex_support.v[0, i_b])
                    if dot < -mpr_static_config.CCD_EPS:
                        mpr_state.simplex_support.v1[2, i_b] = v1
                        mpr_state.simplex_support.v2[2, i_b] = v2
                        mpr_state.simplex_support.v[2, i_b] = v
                        cont = True

                    if not cont:
                        va = v.cross(mpr_state.simplex_support.v[2, i_b])
                        dot = va.dot(mpr_state.simplex_support.v[0, i_b])
                        if dot < -mpr_static_config.CCD_EPS:
                            mpr_state.simplex_support.v1[1, i_b] = v1
                            mpr_state.simplex_support.v2[1, i_b] = v2
                            mpr_state.simplex_support.v[1, i_b] = v
                            cont = True

                    if cont:
                        va = mpr_state.simplex_support.v[1, i_b] - mpr_state.simplex_support.v[0, i_b]
                        vb = mpr_state.simplex_support.v[2, i_b] - mpr_state.simplex_support.v[0, i_b]
                        direction = va.cross(vb)
                        direction = direction.normalized()
                        num_trials = num_trials + 1
                        if num_trials == 15:
                            ret = -1
                            break
                    else:
                        mpr_state.simplex_support.v1[3, i_b] = v1
                        mpr_state.simplex_support.v2[3, i_b] = v2
                        mpr_state.simplex_support.v[3, i_b] = v
                        mpr_state.simplex_size[i_b] = 4

    return ret


@ti.func
def guess_geoms_center(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    static_rigid_sim_config: ti.template(),
    mpr_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
    normal_ws,
):
    # MPR algorithm was initially design to check whether a pair of convex geometries was colliding. The author
    # proposed to extend its application to collision detection as it can provide the contact normal and penetration
    # depth in some cases, i.e. when the original of the Minkowski difference can be projected inside the refined
    # portal. Beyond this specific scenario, it only provides an approximation, that gets worst and worst as the
    # ray casting and portal normal are misaligned.
    # For convex shape, one can show that everything should be fine for low penetration-to-size ratio for each
    # geometry, and the probability to accurately estimate the contact point decreases as this ratio increases.
    #
    # This issue can be avoided by initializing the algorithm with the good seach direction, basically the one
    # from the previous simulation timestep would do fine, as the penetration was smaller at that time and so the
    # likely for this direction to be valid was larger. Alternatively, the direction of the linear velocity would
    # be a good option.
    #
    # Enforcing a specific search direction to vanilla MPR is not straightforward, because the direction of the ray
    # control by v0, which is defined as the difference between the respective centers of each geometry.
    # The only option is to change the way the center of each geometry are defined, so as to make the ray casting
    # from origin to v0 as colinear as possible with the direction we are interested, while remaining included in
    # their respective geometry.
    # The idea is to offset the original centers of each geometry by a ratio that corresponds to their respective
    # (rotated) bounding box size along each axe. Each center cannot be moved more than half of its bound-box size
    # along each axe. This could lead to a center that is outside the geometries if they do not collide, but
    # should be fine otherwise. Anyway, this is not a big deal in practice and MPR is robust enough to converge to
    # a meaningful solution and if the center is slightly off of each geometry. Nevertheless, if it turns out this
    # is a real issue, one way to address it is to evaluate the exact signed distance of each center wrt their
    # respective geometry. If one of the center is off, its offset from the original center is divided by 2 and the
    # signed distance is computed once again until to find a valid point. This procedure should be cheap.

    g_pos_a = geoms_state.pos[i_ga, i_b]
    g_pos_b = geoms_state.pos[i_gb, i_b]
    g_quat_a = geoms_state.quat[i_ga, i_b]
    g_quat_b = geoms_state.quat[i_gb, i_b]
    center_a = gu.ti_transform_by_trans_quat(geoms_info.center[i_ga], g_pos_a, g_quat_a)
    center_b = gu.ti_transform_by_trans_quat(geoms_info.center[i_gb], g_pos_b, g_quat_b)

    # Completely different center logics if a normal guess is provided
    if ti.static(not static_rigid_sim_config.enable_mujoco_compatibility):
        if (ti.abs(normal_ws) > mpr_static_config.CCD_EPS).any():
            # Must start from the center of each bounding box
            center_a_local = 0.5 * (geoms_init_AABB[i_ga, 7] + geoms_init_AABB[i_ga, 0])
            center_a = gu.ti_transform_by_trans_quat(center_a_local, g_pos_a, g_quat_a)
            center_b_local = 0.5 * (geoms_init_AABB[i_gb, 7] + geoms_init_AABB[i_gb, 0])
            center_b = gu.ti_transform_by_trans_quat(center_b_local, g_pos_b, g_quat_b)
            delta = center_a - center_b

            # Skip offset if normal is roughly pointing in the same direction already.
            # Note that a threshold of 0.5 would probably make more sense, but this means that the center of each
            # geometry would significantly affect collision detection, which is undesirable.
            normal = delta.normalized()
            if normal_ws.cross(normal).norm() > 0.01:
                # Compute the target offset
                offset = delta.dot(normal_ws) * normal_ws - delta
                offset_norm = offset.norm()

                if offset_norm > gs.EPS:
                    # Compute the size of the bounding boxes along the target offset direction.
                    # First, move the direction in local box frame
                    dir_offset = offset / offset_norm
                    dir_offset_local_a = gu.ti_inv_transform_by_quat(dir_offset, g_quat_a)
                    dir_offset_local_b = gu.ti_inv_transform_by_quat(dir_offset, g_quat_b)
                    box_size_a = geoms_init_AABB[i_ga, 7] - geoms_init_AABB[i_ga, 0]
                    box_size_b = geoms_init_AABB[i_gb, 7] - geoms_init_AABB[i_gb, 0]
                    length_a = box_size_a.dot(ti.abs(dir_offset_local_a))
                    length_b = box_size_b.dot(ti.abs(dir_offset_local_b))

                    # Shift the center of each geometry
                    offset_ratio = ti.min(offset_norm / (length_a + length_b), 0.5)
                    center_a = center_a + dir_offset * length_a * offset_ratio
                    center_b = center_b - dir_offset * length_b * offset_ratio

    return center_a, center_b


@ti.func
def func_mpr_contact_from_centers(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: ti.template(),
    mpr_state: array_class.MPRState,
    mpr_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
    center_a,
    center_b,
):
    res = mpr_discover_portal(
        geoms_state=geoms_state,
        geoms_info=geoms_info,
        support_field_info=support_field_info,
        support_field_static_config=support_field_static_config,
        collider_state=collider_state,
        collider_info=collider_info,
        collider_static_config=collider_static_config,
        mpr_state=mpr_state,
        mpr_static_config=mpr_static_config,
        i_ga=i_ga,
        i_gb=i_gb,
        i_b=i_b,
        center_a=center_a,
        center_b=center_b,
    )

    is_col = False
    pos = gs.ti_vec3([0.0, 0.0, 0.0])
    normal = gs.ti_vec3([0.0, 0.0, 0.0])
    penetration = gs.ti_float(0.0)

    if res == 1:
        is_col, normal, penetration, pos = mpr_find_penetr_touch(mpr_state, i_ga, i_gb, i_b)
    elif res == 2:
        is_col, normal, penetration, pos = mpr_find_penetr_segment(mpr_state, i_ga, i_gb, i_b)
    elif res == 0:
        res = mpr_refine_portal(
            geoms_state,
            geoms_info,
            collider_state,
            collider_info,
            collider_static_config,
            mpr_state,
            mpr_static_config,
            support_field_info,
            support_field_static_config,
            i_ga,
            i_gb,
            i_b,
        )
        if res >= 0:
            is_col, normal, penetration, pos = mpr_find_penetration(
                geoms_state,
                geoms_info,
                static_rigid_sim_config,
                support_field_info,
                support_field_static_config,
                collider_state,
                collider_info,
                collider_static_config,
                mpr_state,
                mpr_static_config,
                i_ga,
                i_gb,
                i_b,
            )
    return is_col, normal, penetration, pos


@ti.func
def func_mpr_contact(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: ti.template(),
    mpr_state: array_class.MPRState,
    mpr_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
    normal_ws,
):
    center_a, center_b = guess_geoms_center(
        geoms_state,
        geoms_info,
        geoms_init_AABB,
        static_rigid_sim_config,
        mpr_static_config,
        i_ga,
        i_gb,
        i_b,
        normal_ws,
    )
    return func_mpr_contact_from_centers(
        geoms_state=geoms_state,
        geoms_info=geoms_info,
        static_rigid_sim_config=static_rigid_sim_config,
        collider_state=collider_state,
        collider_info=collider_info,
        collider_static_config=collider_static_config,
        mpr_state=mpr_state,
        mpr_static_config=mpr_static_config,
        support_field_info=support_field_info,
        support_field_static_config=support_field_static_config,
        i_ga=i_ga,
        i_gb=i_gb,
        i_b=i_b,
        center_a=center_a,
        center_b=center_b,
    )
