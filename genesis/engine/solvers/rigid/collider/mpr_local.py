"""
Thread-local versions of MPR collision detection functions.

This module provides versions of MPR functions that accept pos/quat as direct
parameters instead of reading from geoms_state, enabling race-free multi-contact
detection when parallelizing across collision pairs within the same environment.
"""

import gstaichi as ti

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.solvers.rigid.collider import support_field, support_field_local, mpr
from genesis.utils import array_class



@ti.func
def support_driver_local(
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
    """
    Thread-local version of support_driver for MPR.

    Dispatches to the appropriate support function based on geometry type,
    using thread-local pos/quat instead of reading from geoms_state.

    Args:
        geoms_info: Geometry information (types, dimensions, etc.)
        collider_state: Collider state (for terrain prism)
        collider_static_config: Static configuration
        support_field_info: Pre-computed support field data
        direction: Support direction in world frame
        i_g: Geometry index
        i_b: Batch/environment index (for terrain prism)
        pos: Geometry position in world frame (thread-local, 28 bytes)
        quat: Geometry quaternion (thread-local, 28 bytes)

    Returns:
        v: Support point in world frame
    """
    v = ti.Vector.zero(gs.ti_float, 3)
    geom_type = geoms_info.type[i_g]

    if geom_type == gs.GEOM_TYPE.SPHERE:
        v, v_, vid = support_field_local._func_support_sphere_local(geoms_info, direction, i_g, pos, quat, False)
    elif geom_type == gs.GEOM_TYPE.ELLIPSOID:
        v = support_field_local._func_support_ellipsoid_local(geoms_info, direction, i_g, pos, quat)
    elif geom_type == gs.GEOM_TYPE.CAPSULE:
        v = support_field_local._func_support_capsule_local(geoms_info, direction, i_g, pos, quat, False)
    elif geom_type == gs.GEOM_TYPE.BOX:
        v, v_, vid = support_field_local._func_support_box_local(geoms_info, direction, i_g, pos, quat)
    elif geom_type == gs.GEOM_TYPE.TERRAIN:
        if ti.static(collider_static_config.has_terrain):
            # Terrain support doesn't depend on geometry pos/quat - uses collider_state.prism
            # Terrain is global and not perturbed, so we use the global state directly
            v, _ = support_field._func_support_prism(collider_state, direction, i_g, i_b)
    else:
        # Mesh geometries
        v, v_, vid = support_field_local._func_support_world_local(support_field_info, direction, i_g, pos, quat)

    return v


@ti.func
def compute_support_local(
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
    """
    Thread-local version of compute_support.

    Computes the support point on the Minkowski difference of two geometries.
    This is the fundamental operation for MPR collision detection.

    Args:
        geoms_info: Geometry information
        collider_state: Collider state (for terrain)
        collider_static_config: Static configuration
        support_field_info: Pre-computed support field data
        direction: Support direction in world frame
        i_ga: First geometry index
        i_gb: Second geometry index
        i_b: Batch/environment index
        pos_a: First geometry position (thread-local, 28 bytes)
        quat_a: First geometry quaternion (thread-local, 28 bytes)
        pos_b: Second geometry position (thread-local, 28 bytes)
        quat_b: Second geometry quaternion (thread-local, 28 bytes)

    Returns:
        v: Support point on Minkowski difference (v1 - v2)
        v1: Support point on first geometry
        v2: Support point on second geometry
    """
    v1 = support_driver_local(
        geoms_info,
        collider_state,
        collider_static_config,
        support_field_info,
        direction,
        i_ga,
        i_b,
        pos_a,
        quat_a,
    )
    v2 = support_driver_local(
        geoms_info,
        collider_state,
        collider_static_config,
        support_field_info,
        -direction,
        i_gb,
        i_b,
        pos_b,
        quat_b,
    )

    v = v1 - v2
    return v, v1, v2


@ti.func
def func_geom_support_local(
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    direction,
    i_g,
    pos: ti.types.vector(3, dtype=gs.ti_float),
    quat: ti.types.vector(4, dtype=gs.ti_float),
):
    """
    Thread-local version of func_geom_support.

    Finds the support point on a mesh geometry by exhaustively searching
    through all vertices. This is used as a fallback when support fields
    are not available.

    Args:
        geoms_info: Geometry information
        verts_info: Vertex information
        direction: Support direction in world frame
        i_g: Geometry index
        pos: Geometry position in world frame (thread-local, 28 bytes)
        quat: Geometry quaternion (thread-local, 28 bytes)

    Returns:
        v: Support point in world frame
        vid: Vertex ID of the support point
    """
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
def mpr_refine_portal_local(
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
    """
    Thread-local version of mpr_refine_portal.

    Refines the MPR portal by iteratively expanding it towards the origin
    of the Minkowski difference. This is the main iteration loop of MPR.

    Args:
        geoms_info: Geometry information
        collider_state: Collider state (for terrain)
        collider_static_config: Static configuration
        mpr_state: MPR algorithm state (simplex, etc.)
        mpr_info: MPR algorithm parameters (tolerances, iteration limits)
        support_field_info: Pre-computed support field data
        i_ga: First geometry index
        i_gb: Second geometry index
        i_b: Batch/environment index
        pos_a: First geometry position (thread-local, 28 bytes)
        quat_a: First geometry quaternion (thread-local, 28 bytes)
        pos_b: Second geometry position (thread-local, 28 bytes)
        quat_b: Second geometry quaternion (thread-local, 28 bytes)

    Returns:
        ret: Status code (-1: refinement failed, >=0: success)
    """
    ret = 1
    while True:
        direction = mpr.mpr_portal_dir(mpr_state, i_ga, i_gb, i_b)

        if mpr.mpr_portal_encapsules_origin(mpr_state, mpr_info, direction, i_ga, i_gb, i_b):
            ret = 0
            break

        v, v1, v2 = compute_support_local(
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

        if not mpr.mpr_portal_can_encapsule_origin(mpr_info, v, direction) or mpr.mpr_portal_reach_tolerance(
            mpr_state, mpr_info, v, direction, i_ga, i_gb, i_b
        ):
            ret = -1
            break

        mpr.mpr_expand_portal(mpr_state, v, v1, v2, i_ga, i_gb, i_b)
    return ret


@ti.func
def mpr_find_penetration_local(
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
    """
    Thread-local version of mpr_find_penetration.

    Finds the penetration depth and contact information after the portal has
    been refined. This is the final step of MPR collision detection.

    Args:
        geoms_info: Geometry information
        static_rigid_sim_config: Static simulation configuration
        support_field_info: Pre-computed support field data
        collider_state: Collider state (for terrain)
        collider_static_config: Static configuration
        mpr_state: MPR algorithm state (simplex, etc.)
        mpr_info: MPR algorithm parameters (tolerances, iteration limits)
        i_ga: First geometry index
        i_gb: Second geometry index
        i_b: Batch/environment index
        pos_a: First geometry position (thread-local, 28 bytes)
        quat_a: First geometry quaternion (thread-local, 28 bytes)
        pos_b: Second geometry position (thread-local, 28 bytes)
        quat_b: Second geometry quaternion (thread-local, 28 bytes)

    Returns:
        is_col: True if collision detected
        normal: Contact normal in world space
        penetration: Penetration depth
        pos: Contact position in world space
    """
    iterations = 0

    is_col = False
    pos = gs.ti_vec3([0.0, 0.0, 0.0])
    normal = gs.ti_vec3([0.0, 0.0, 0.0])
    penetration = gs.ti_float(0.0)

    while True:
        direction = mpr.mpr_portal_dir(mpr_state, i_ga, i_gb, i_b)
        v, v1, v2 = compute_support_local(
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
            mpr.mpr_portal_reach_tolerance(mpr_state, mpr_info, v, direction, i_ga, i_gb, i_b)
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
            pos = mpr.mpr_find_pos(static_rigid_sim_config, mpr_state, mpr_info, i_ga, i_gb, i_b)
            break

        mpr.mpr_expand_portal(mpr_state, v, v1, v2, i_ga, i_gb, i_b)
        iterations += 1

    return is_col, normal, penetration, pos


@ti.func
def mpr_discover_portal_local(
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
    """
    Thread-local version of mpr_discover_portal.

    Discovers the initial portal (simplex) for MPR algorithm, which is used
    to determine if two geometries are colliding and to find the collision normal.

    This function builds up the MPR simplex by iteratively finding support points
    until it constructs a tetrahedron (4-simplex) that encloses the origin.

    Args:
        geoms_info: Geometry information
        support_field_info: Pre-computed support field data
        collider_state: Collider state (for terrain)
        collider_static_config: Static configuration
        mpr_state: MPR algorithm state (simplex, etc.)
        mpr_info: MPR algorithm parameters (tolerances, iteration limits)
        i_ga: First geometry index
        i_gb: Second geometry index
        i_b: Batch/environment index
        center_a: Center point for first geometry
        center_b: Center point for second geometry
        pos_a: First geometry position (thread-local, 28 bytes)
        quat_a: First geometry quaternion (thread-local, 28 bytes)
        pos_b: Second geometry position (thread-local, 28 bytes)
        quat_b: Second geometry quaternion (thread-local, 28 bytes)

    Returns:
        ret: Status code (-1: no collision, 0: portal found, 1: touching, 2: segment)
    """
    mpr_state.simplex_support.v1[0, i_b] = center_a
    mpr_state.simplex_support.v2[0, i_b] = center_b
    mpr_state.simplex_support.v[0, i_b] = center_a - center_b
    mpr_state.simplex_size[i_b] = 1

    if (ti.abs(mpr_state.simplex_support.v[0, i_b]) < mpr_info.CCD_EPS[None]).all():
        mpr_state.simplex_support.v[0, i_b][0] += 10.0 * mpr_info.CCD_EPS[None]

    direction = -mpr_state.simplex_support.v[0, i_b].normalized()

    v, v1, v2 = compute_support_local(
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
            v, v1, v2 = compute_support_local(
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
                    mpr.mpr_swap(mpr_state, 1, 2, i_ga, i_gb, i_b)
                    direction = -direction

                # FIXME: This algorithm may get stuck in an infinite loop if the actually penetration is smaller
                # then `CCD_EPS` and at least one of the center of each geometry is outside their convex hull.
                # Since this deadlock happens very rarely, a simple fix is to abort computation after a few trials.
                num_trials = gs.ti_int(0)
                while mpr_state.simplex_size[i_b] < 4:
                    v, v1, v2 = compute_support_local(
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
def guess_geoms_center_local(
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
    """
    Thread-local version of guess_geoms_center.

    Computes the center points for two geometries, optionally offsetting them based on
    a cached normal direction to improve MPR convergence. This warm-starting technique
    helps MPR find better initial search directions.

    The algorithm offsets geometry centers along the cached normal direction by a fraction
    of their bounding box sizes, ensuring the ray from origin to v0 aligns better with
    the expected collision normal.

    Args:
        geoms_info: Geometry information
        geoms_init_AABB: Initial axis-aligned bounding boxes
        rigid_global_info: Global simulation parameters (EPS)
        static_rigid_sim_config: Static simulation configuration
        mpr_info: MPR algorithm parameters
        i_ga: First geometry index
        i_gb: Second geometry index
        pos_a: First geometry position (thread-local, 28 bytes)
        quat_a: First geometry quaternion (thread-local, 28 bytes)
        pos_b: Second geometry position (thread-local, 28 bytes)
        quat_b: Second geometry quaternion (thread-local, 28 bytes)
        normal_ws: Cached normal from previous timestep (world space)

    Returns:
        center_a: Computed center for first geometry
        center_b: Computed center for second geometry
    """
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

    # Apply advanced warm-starting logic if MuJoCo compatibility is disabled
    # and a valid cached normal is available
    # Completely different center logics if a normal guess is provided
    if ti.static(not static_rigid_sim_config.enable_mujoco_compatibility):
        if (ti.abs(normal_ws) > mpr_info.CCD_EPS[None]).any():
            # Start from the center of each bounding box
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
def func_mpr_contact_from_centers_local(
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
    """
    Thread-local version of func_mpr_contact_from_centers.

    Performs MPR collision detection given pre-computed geometry centers.
    This is the main MPR algorithm orchestrator.

    Args:
        geoms_info: Geometry information
        static_rigid_sim_config: Static simulation configuration
        collider_state: Collider state (for terrain)
        collider_static_config: Static configuration
        mpr_state: MPR algorithm state (simplex, etc.)
        mpr_info: MPR algorithm parameters
        support_field_info: Pre-computed support field data
        i_ga: First geometry index
        i_gb: Second geometry index
        i_b: Batch/environment index
        center_a: Center point for first geometry
        center_b: Center point for second geometry
        pos_a: First geometry position (thread-local, 28 bytes)
        quat_a: First geometry quaternion (thread-local, 28 bytes)
        pos_b: Second geometry position (thread-local, 28 bytes)
        quat_b: Second geometry quaternion (thread-local, 28 bytes)

    Returns:
        is_col: True if collision detected
        normal: Contact normal in world space
        penetration: Penetration depth
        pos: Contact position in world space
    """
    res = mpr_discover_portal_local(
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
        is_col, normal, penetration, pos = mpr.mpr_find_penetr_touch(mpr_state, i_ga, i_gb, i_b)
    elif res == 2:
        is_col, normal, penetration, pos = mpr.mpr_find_penetr_segment(mpr_state, i_ga, i_gb, i_b)
    elif res == 0:
        res = mpr_refine_portal_local(
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
            is_col, normal, penetration, pos = mpr_find_penetration_local(
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
def func_mpr_contact_local(
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
    """
    Thread-local version of func_mpr_contact.

    Top-level MPR collision detection function. Computes geometry centers
    and then performs collision detection.

    This is the main entry point for thread-safe MPR collision detection that
    works with thread-local geometry poses, enabling race-free multi-contact
    detection when parallelizing across collision pairs.

    Thread-safety note: Geometry indices `i_ga` and `i_gb` are only used for read-only
    metadata access (geometry types, AABB bounds, etc.) and passing to support functions.
    They do not access `geoms_state.pos` or `geoms_state.quat`.

    Args:
        geoms_info: Geometry information
        geoms_init_AABB: Initial axis-aligned bounding boxes
        rigid_global_info: Global simulation parameters (EPS)
        static_rigid_sim_config: Static simulation configuration
        collider_state: Collider state (for terrain)
        collider_static_config: Static configuration
        mpr_state: MPR algorithm state (simplex, etc.)
        mpr_info: MPR algorithm parameters
        support_field_info: Pre-computed support field data
        i_ga: First geometry index
        i_gb: Second geometry index
        i_b: Batch/environment index
        normal_ws: Cached normal from previous timestep (for warm-starting)
        pos_a: First geometry position (thread-local, 28 bytes)
        quat_a: First geometry quaternion (thread-local, 28 bytes)
        pos_b: Second geometry position (thread-local, 28 bytes)
        quat_b: Second geometry quaternion (thread-local, 28 bytes)

    Returns:
        is_col: True if collision detected
        normal: Contact normal in world space
        penetration: Penetration depth
        pos: Contact position in world space
    """
    center_a, center_b = guess_geoms_center_local(
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
    return func_mpr_contact_from_centers_local(
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
