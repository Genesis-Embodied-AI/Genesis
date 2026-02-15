import quadrants as ti

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class
from . import support_field


class MPR:
    def __init__(self, rigid_solver):
        self._solver = rigid_solver

        self._mpr_info = array_class.get_mpr_info(
            # It has been observed in practice that increasing this threshold makes collision detection instable,
            # which is surprising since 1e-9 is above single precision (which has only 7 digits of precision).
            CCD_EPS=1e-9 if gs.ti_float == ti.f32 else 1e-10,
            CCD_TOLERANCE=1e-6,
            CCD_ITERATIONS=50,
        )
        self._mpr_state = array_class.get_mpr_state(self._solver._B)


@ti.kernel
def clear(mpr_state: ti.template()):
    mpr_state.simplex_size.fill(0)


@ti.func
def func_point_in_geom_aabb(geoms_state: array_class.GeomsState, point, i_g, i_b):
    return (point < geoms_state.aabb_max[i_g, i_b]).all() and (point > geoms_state.aabb_min[i_g, i_b]).all()


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
    geoms_info: array_class.GeomsInfo,
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    direction,
    i_g,
    i_b,
    pos: ti.types.vector(3, dtype=gs.ti_float),
    quat: ti.types.vector(4, dtype=gs.ti_float),
):
    v = ti.Vector.zero(gs.ti_float, 3)
    geom_type = geoms_info.type[i_g]
    if geom_type == gs.GEOM_TYPE.SPHERE:
        v, v_, vid = support_field._func_support_sphere(geoms_info, direction, i_g, pos, quat, False)
    elif geom_type == gs.GEOM_TYPE.ELLIPSOID:
        v = support_field._func_support_ellipsoid(geoms_info, direction, i_g, pos, quat)
    elif geom_type == gs.GEOM_TYPE.CAPSULE:
        v = support_field._func_support_capsule(geoms_info, direction, i_g, pos, quat, False)
    elif geom_type == gs.GEOM_TYPE.BOX:
        v, v_, vid = support_field._func_support_box(geoms_info, direction, i_g, pos, quat)
    elif geom_type == gs.GEOM_TYPE.TERRAIN:
        if ti.static(collider_static_config.has_terrain):
            # Terrain support doesn't depend on geometry pos/quat - uses collider_state.prism
            # Terrain is global and not perturbed, so we use the global state directly
            v, _ = support_field._func_support_prism(collider_state, direction, i_b)
    else:
        v, v_, vid = support_field._func_support_world(support_field_info, direction, i_g, pos, quat)

    return v


@ti.func
def compute_support(
    geoms_info: array_class.GeomsInfo,
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    direction,
    i_ga,
    i_gb,
    i_b,
    pos_a: ti.types.vector(3, dtype=gs.ti_float),
    quat_a: ti.types.vector(4, dtype=gs.ti_float),
    pos_b: ti.types.vector(3, dtype=gs.ti_float),
    quat_b: ti.types.vector(4, dtype=gs.ti_float),
):
    v1 = support_driver(
        geoms_info, collider_state, collider_static_config, support_field_info, direction, i_ga, i_b, pos_a, quat_a
    )
    v2 = support_driver(
        geoms_info, collider_state, collider_static_config, support_field_info, -direction, i_gb, i_b, pos_b, quat_b
    )

    v = v1 - v2
    return v, v1, v2


@ti.func
def func_geom_support(
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    direction,
    i_g,
    pos: ti.types.vector(3, dtype=gs.ti_float),
    quat: ti.types.vector(4, dtype=gs.ti_float),
):
    direction_in_init_frame = gu.ti_inv_transform_by_quat(direction, quat)

    dot_max = gs.ti_float(-1e10)
    v = ti.Vector.zero(gs.ti_float, 3)
    vid = 0

    for i_v in range(geoms_info.vert_start[i_g], geoms_info.vert_end[i_g]):
        pos_local = verts_info.init_pos[i_v]
        dot = pos_local.dot(direction_in_init_frame)
        if dot > dot_max:
            v = pos_local
            dot_max = dot
            vid = i_v
    v_world = gu.ti_transform_by_trans_quat(v, pos, quat)

    return v_world, vid


@ti.func
def mpr_refine_portal(
    geoms_info: array_class.GeomsInfo,
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    mpr_state: array_class.MPRState,
    mpr_info: array_class.MPRInfo,
    support_field_info: array_class.SupportFieldInfo,
    i_ga,
    i_gb,
    i_b,
    pos_a: ti.types.vector(3, dtype=gs.ti_float),
    quat_a: ti.types.vector(4, dtype=gs.ti_float),
    pos_b: ti.types.vector(3, dtype=gs.ti_float),
    quat_b: ti.types.vector(4, dtype=gs.ti_float),
):
    ret = 1
    while True:
        direction = mpr_portal_dir(mpr_state, i_ga, i_gb, i_b)

        if mpr_portal_encapsules_origin(mpr_state, mpr_info, direction, i_ga, i_gb, i_b):
            ret = 0
            break

        v, v1, v2 = compute_support(
            geoms_info,
            collider_state,
            collider_static_config,
            support_field_info,
            direction,
            i_ga,
            i_gb,
            i_b,
            pos_a,
            quat_a,
            pos_b,
            quat_b,
        )

        if not mpr_portal_can_encapsule_origin(mpr_info, v, direction) or mpr_portal_reach_tolerance(
            mpr_state, mpr_info, v, direction, i_ga, i_gb, i_b
        ):
            ret = -1
            break

        mpr_expand_portal(mpr_state, v, v1, v2, i_ga, i_gb, i_b)
    return ret


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
def mpr_find_penetration(
    geoms_info: array_class.GeomsInfo,
    static_rigid_sim_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    mpr_state: array_class.MPRState,
    mpr_info: array_class.MPRInfo,
    i_ga,
    i_gb,
    i_b,
    pos_a: ti.types.vector(3, dtype=gs.ti_float),
    quat_a: ti.types.vector(4, dtype=gs.ti_float),
    pos_b: ti.types.vector(3, dtype=gs.ti_float),
    quat_b: ti.types.vector(4, dtype=gs.ti_float),
):
    iterations = 0

    is_col = False
    pos = gs.ti_vec3([0.0, 0.0, 0.0])
    normal = gs.ti_vec3([0.0, 0.0, 0.0])
    penetration = gs.ti_float(0.0)

    while True:
        direction = mpr_portal_dir(mpr_state, i_ga, i_gb, i_b)
        v, v1, v2 = compute_support(
            geoms_info,
            collider_state,
            collider_static_config,
            support_field_info,
            direction,
            i_ga,
            i_gb,
            i_b,
            pos_a,
            quat_a,
            pos_b,
            quat_b,
        )
        if (
            mpr_portal_reach_tolerance(mpr_state, mpr_info, v, direction, i_ga, i_gb, i_b)
            or iterations > mpr_info.CCD_ITERATIONS[None]
        ):
            # The contact point is defined as the projection of the origin onto the portal, i.e. the closest point
            # to the origin that lies inside the portal.
            # Let's consider the portal as an infinite plane rather than a face triangle. This makes sense because
            # the projection of the origin must be strictly included into the portal triangle for it to correspond
            # to the true penetration depth.
            # For reference about this property, see 'Collision Handling with Variable-Step Integrators' Theorem 4.2:
            # https://modiasim.github.io/Modia3D.jl/resources/documentation/CollisionHandling_Neumayr_Otter_2017.pdf
            #
            # In theory, the center should have been shifted until to end up with the one and only portal satisfying
            # this condition. However, a naive implementation of this process must be avoided because it would be
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
                    mpr_info,
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
            pos = mpr_find_pos(static_rigid_sim_config, mpr_state, mpr_info, i_ga, i_gb, i_b)
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
    geoms_info: array_class.GeomsInfo,
    support_field_info: array_class.SupportFieldInfo,
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    mpr_state: array_class.MPRState,
    mpr_info: array_class.MPRInfo,
    i_ga,
    i_gb,
    i_b,
    center_a,
    center_b,
    pos_a: ti.types.vector(3, dtype=gs.ti_float),
    quat_a: ti.types.vector(4, dtype=gs.ti_float),
    pos_b: ti.types.vector(3, dtype=gs.ti_float),
    quat_b: ti.types.vector(4, dtype=gs.ti_float),
):
    mpr_state.simplex_support.v1[0, i_b] = center_a
    mpr_state.simplex_support.v2[0, i_b] = center_b
    mpr_state.simplex_support.v[0, i_b] = center_a - center_b
    mpr_state.simplex_size[i_b] = 1

    if (ti.abs(mpr_state.simplex_support.v[0, i_b]) < mpr_info.CCD_EPS[None]).all():
        mpr_state.simplex_support.v[0, i_b][0] += 10.0 * mpr_info.CCD_EPS[None]

    direction = -mpr_state.simplex_support.v[0, i_b].normalized()

    v, v1, v2 = compute_support(
        geoms_info,
        collider_state,
        collider_static_config,
        support_field_info,
        direction,
        i_ga,
        i_gb,
        i_b,
        pos_a,
        quat_a,
        pos_b,
        quat_b,
    )

    mpr_state.simplex_support.v1[1, i_b] = v1
    mpr_state.simplex_support.v2[1, i_b] = v2
    mpr_state.simplex_support.v[1, i_b] = v
    mpr_state.simplex_size[i_b] = 2

    dot = v.dot(direction)

    ret = 0
    if dot < mpr_info.CCD_EPS[None]:
        ret = -1
    else:
        direction = mpr_state.simplex_support.v[0, i_b].cross(mpr_state.simplex_support.v[1, i_b])
        if direction.dot(direction) < mpr_info.CCD_EPS[None]:
            if (ti.abs(mpr_state.simplex_support.v[1, i_b]) < mpr_info.CCD_EPS[None]).all():
                ret = 1
            else:
                ret = 2
        else:
            direction = direction.normalized()
            v, v1, v2 = compute_support(
                geoms_info,
                collider_state,
                collider_static_config,
                support_field_info,
                direction,
                i_ga,
                i_gb,
                i_b,
                pos_a,
                quat_a,
                pos_b,
                quat_b,
            )
            dot = v.dot(direction)
            if dot < mpr_info.CCD_EPS[None]:
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
                # Since this deadlock happens very rarely, a simple fix is to abort computation after a few trials.
                num_trials = gs.ti_int(0)
                while mpr_state.simplex_size[i_b] < 4:
                    v, v1, v2 = compute_support(
                        geoms_info,
                        collider_state,
                        collider_static_config,
                        support_field_info,
                        direction,
                        i_ga,
                        i_gb,
                        i_b,
                        pos_a,
                        quat_a,
                        pos_b,
                        quat_b,
                    )
                    dot = v.dot(direction)
                    if dot < mpr_info.CCD_EPS[None]:
                        ret = -1
                        break

                    cont = False

                    va = mpr_state.simplex_support.v[1, i_b].cross(v)
                    dot = va.dot(mpr_state.simplex_support.v[0, i_b])
                    if dot < -mpr_info.CCD_EPS[None]:
                        mpr_state.simplex_support.v1[2, i_b] = v1
                        mpr_state.simplex_support.v2[2, i_b] = v2
                        mpr_state.simplex_support.v[2, i_b] = v
                        cont = True

                    if not cont:
                        va = v.cross(mpr_state.simplex_support.v[2, i_b])
                        dot = va.dot(mpr_state.simplex_support.v[0, i_b])
                        if dot < -mpr_info.CCD_EPS[None]:
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
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    mpr_info: array_class.MPRInfo,
    i_ga,
    i_gb,
    pos_a: ti.types.vector(3, dtype=gs.ti_float),
    quat_a: ti.types.vector(4, dtype=gs.ti_float),
    pos_b: ti.types.vector(3, dtype=gs.ti_float),
    quat_b: ti.types.vector(4, dtype=gs.ti_float),
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
    EPS = rigid_global_info.EPS[None]

    # Transform geometry centers to world space using thread-local pos/quat
    center_a = gu.ti_transform_by_trans_quat(geoms_info.center[i_ga], pos_a, quat_a)
    center_b = gu.ti_transform_by_trans_quat(geoms_info.center[i_gb], pos_b, quat_b)

    # Completely different center logics if a normal guess is provided
    if ti.static(not static_rigid_sim_config.enable_mujoco_compatibility):
        if (ti.abs(normal_ws) > mpr_info.CCD_EPS[None]).any():
            # Must start from the center of each bounding box
            center_a_local = 0.5 * (geoms_init_AABB[i_ga, 7] + geoms_init_AABB[i_ga, 0])
            center_a = gu.ti_transform_by_trans_quat(center_a_local, pos_a, quat_a)
            center_b_local = 0.5 * (geoms_init_AABB[i_gb, 7] + geoms_init_AABB[i_gb, 0])
            center_b = gu.ti_transform_by_trans_quat(center_b_local, pos_b, quat_b)
            delta = center_a - center_b

            # Skip offset if normal is roughly pointing in the same direction already.
            # Note that a threshold of 0.5 would probably make more sense, but this means that the center of each
            # geometry would significantly affect collision detection, which is undesirable.
            normal = delta.normalized()
            if normal_ws.cross(normal).norm() > 0.01:
                # Compute the target offset
                offset = delta.dot(normal_ws) * normal_ws - delta
                offset_norm = offset.norm()

                if offset_norm > EPS:
                    # Compute the size of the bounding boxes along the target offset direction.
                    # First, move the direction in local box frame
                    dir_offset = offset / offset_norm
                    dir_offset_local_a = gu.ti_inv_transform_by_quat(dir_offset, quat_a)
                    dir_offset_local_b = gu.ti_inv_transform_by_quat(dir_offset, quat_b)
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
    geoms_info: array_class.GeomsInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    mpr_state: array_class.MPRState,
    mpr_info: array_class.MPRInfo,
    support_field_info: array_class.SupportFieldInfo,
    i_ga,
    i_gb,
    i_b,
    center_a,
    center_b,
    pos_a: ti.types.vector(3, dtype=gs.ti_float),
    quat_a: ti.types.vector(4, dtype=gs.ti_float),
    pos_b: ti.types.vector(3, dtype=gs.ti_float),
    quat_b: ti.types.vector(4, dtype=gs.ti_float),
):
    res = mpr_discover_portal(
        geoms_info=geoms_info,
        support_field_info=support_field_info,
        collider_state=collider_state,
        collider_static_config=collider_static_config,
        mpr_state=mpr_state,
        mpr_info=mpr_info,
        i_ga=i_ga,
        i_gb=i_gb,
        i_b=i_b,
        center_a=center_a,
        center_b=center_b,
        pos_a=pos_a,
        quat_a=quat_a,
        pos_b=pos_b,
        quat_b=quat_b,
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
            geoms_info,
            collider_state,
            collider_static_config,
            mpr_state,
            mpr_info,
            support_field_info,
            i_ga,
            i_gb,
            i_b,
            pos_a,
            quat_a,
            pos_b,
            quat_b,
        )
        if res >= 0:
            is_col, normal, penetration, pos = mpr_find_penetration(
                geoms_info,
                static_rigid_sim_config,
                support_field_info,
                collider_state,
                collider_static_config,
                mpr_state,
                mpr_info,
                i_ga,
                i_gb,
                i_b,
                pos_a,
                quat_a,
                pos_b,
                quat_b,
            )
    return is_col, normal, penetration, pos


@ti.func
def func_mpr_contact(
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    mpr_state: array_class.MPRState,
    mpr_info: array_class.MPRInfo,
    support_field_info: array_class.SupportFieldInfo,
    i_ga,
    i_gb,
    i_b,
    normal_ws,
    pos_a: ti.types.vector(3, dtype=gs.ti_float),
    quat_a: ti.types.vector(4, dtype=gs.ti_float),
    pos_b: ti.types.vector(3, dtype=gs.ti_float),
    quat_b: ti.types.vector(4, dtype=gs.ti_float),
):
    center_a, center_b = guess_geoms_center(
        geoms_info,
        geoms_init_AABB,
        rigid_global_info,
        static_rigid_sim_config,
        mpr_info,
        i_ga,
        i_gb,
        pos_a,
        quat_a,
        pos_b,
        quat_b,
        normal_ws,
    )
    return func_mpr_contact_from_centers(
        geoms_info=geoms_info,
        static_rigid_sim_config=static_rigid_sim_config,
        collider_state=collider_state,
        collider_static_config=collider_static_config,
        mpr_state=mpr_state,
        mpr_info=mpr_info,
        support_field_info=support_field_info,
        i_ga=i_ga,
        i_gb=i_gb,
        i_b=i_b,
        center_a=center_a,
        center_b=center_b,
        pos_a=pos_a,
        quat_a=quat_a,
        pos_b=pos_b,
        quat_b=quat_b,
    )


from genesis.utils.deprecated_module_wrapper import create_virtual_deprecated_module

create_virtual_deprecated_module(__name__, "genesis.engine.solvers.rigid.mpr_decomp")
