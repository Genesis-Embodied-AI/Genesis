import math
from enum import IntEnum

import gstaichi as ti

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class
from . import gjk_local, support_field


class RETURN_CODE(IntEnum):
    """
    Return codes for the general subroutines used in GJK and EPA algorithms.
    """

    SUCCESS = 0
    FAIL = 1


class GJK_RETURN_CODE(IntEnum):
    """
    Return codes for the GJK algorithm.
    """

    SEPARATED = 0
    INTERSECT = 1
    NUM_ERROR = 2


class EPA_POLY_INIT_RETURN_CODE(IntEnum):
    """
    Return codes for the EPA polytope initialization.
    """

    SUCCESS = 0
    P2_NONCONVEX = 1
    P2_FALLBACK3 = 2
    P3_BAD_NORMAL = 3
    P3_INVALID_V4 = 4
    P3_INVALID_V5 = 5
    P3_MISSING_ORIGIN = 6
    P3_ORIGIN_ON_FACE = 7
    P4_MISSING_ORIGIN = 8
    P4_FALLBACK3 = 9


class GJK:
    def __init__(self, rigid_solver, is_active: bool = True):
        # Initialize static configuration.
        # MuJoCo's multi-contact detection algorithm is disabled by default, because it is often less stable than the
        # other multi-contact detection algorithm. However, we keep the code here for compatibility with MuJoCo and for
        # possible future use.
        enable_mujoco_multi_contact = False
        gjk_max_iterations = 50
        epa_max_iterations = 50
        # 6 * epa_max_iterations is the maximum number of faces in the polytope.
        polytope_max_faces = 6 * epa_max_iterations

        if rigid_solver._static_rigid_sim_config.requires_grad:
            # For differentiable contact detection, we find multiple contact points for each pair.
            max_contacts_per_pair = 20
            max_contact_polygon_verts = 1
        elif enable_mujoco_multi_contact:
            # The maximum number of contacts per pair is related to the maximum number of contact manifold vertices.
            # MuJoCo sets [max_contacts_per_pair] to 50 and [max_contact_polygon_verts] to 150, when it uses
            # multi-contact detection algorithm, assuming that the faces could have more than 4 vertices. However, we
            # set them to smaller values, because we do not expect the faces to have more than 4 vertices in most cases,
            # and we want to keep the memory usage low.
            max_contacts_per_pair = 8
            max_contact_polygon_verts = 30
        else:
            max_contacts_per_pair = 1
            max_contact_polygon_verts = 1

        self._gjk_static_config = array_class.StructGJKStaticConfig(
            enable_mujoco_multi_contact=enable_mujoco_multi_contact,
        )

        # Initialize GJK info
        self._gjk_info = array_class.get_gjk_info(
            max_contacts_per_pair=max_contacts_per_pair,
            max_contact_polygon_verts=max_contact_polygon_verts,
            gjk_max_iterations=gjk_max_iterations,
            epa_max_iterations=epa_max_iterations,
            # When using larger minimum values (e.g. gs.EPS), unstability could occur for some examples (e.g. box
            # pyramid). Also, since different backends could have different precisions (e.g. computing vector norm),
            # we use a very small value, so that there is no discrepancy between backends.
            FLOAT_MIN=1e-15,
            FLOAT_MAX=1e15,
            tolerance=1e-6,
            collision_eps=1e-6,
            # This value has been experimentally determined based on the examples that we currently have (e.g. pyramid,
            # tower, ...), but it could be further tuned based on the future examples.
            simplex_max_degeneracy_sq=1e-5**2,
            polytope_max_faces=polytope_max_faces,
            # This value has been experimentally determined based on the examples that we currently have (e.g. pyramid,
            # tower, ...). We observed the error usually reaches around 5e-4, so we set the threshold to 1e-5 to be
            # safe. However, this value could be further tuned based on the future examples.
            polytope_max_reprojection_error=1e-5,
            # The values are matching MuJoCo for compatibility. Increasing this value could be useful for detecting
            # contact manifolds even when the normals are not perfectly aligned, but we observed that it leads to more
            # false positives and thus not a perfect solution for the multi-contact detection.
            contact_face_tol=math.cos(1.6e-3),
            contact_edge_tol=math.sin(1.6e-3),
            # FIXME: Adjust these values based on the case study.
            diff_contact_eps_boundary=1e-2,
            diff_contact_eps_distance=1e-2,
            diff_contact_eps_affine=1e-2,
            # We apply sqrt to the 10 * EPS value because we often use the square of the normal norm as the denominator.
            diff_contact_min_normal_norm=math.sqrt(gs.EPS * 10.0),
            diff_contact_min_penetration=gs.EPS * 100.0,
        )

        # Initialize GJK state
        self._gjk_state = array_class.get_gjk_state(
            rigid_solver, rigid_solver._static_rigid_sim_config, self._gjk_info, is_active
        )

        self._is_active = is_active

    def reset(self):
        pass

    @property
    def is_active(self):
        return self._is_active


@ti.func
def func_is_equal_vec(a, b, eps):
    """
    Check if two vectors are equal within a small tolerance.
    """
    return (ti.abs(a - b) < eps).all()


@ti.func
def func_compare_sign(a, b):
    """
    Compare the sign of two values.
    """
    ret = 0
    if a > 0 and b > 0:
        ret = 1
    elif a < 0 and b < 0:
        ret = -1
    return ret


@ti.func
def clear_cache(gjk_state: array_class.GJKState, i_b):
    """
    Clear the cache information to prepare for the next GJK-EPA run.

    The cache includes the temporary information about simplex consturction or multi-contact detection.
    """
    gjk_state.support_mesh_prev_vertex_id[i_b, 0] = -1
    gjk_state.support_mesh_prev_vertex_id[i_b, 1] = -1
    gjk_state.multi_contact_flag[i_b] = False
    gjk_state.last_searched_simplex_vertex_id[i_b] = 0


@ti.func
def func_gjk_contact(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    gjk_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    i_ga,
    i_gb,
    i_b,
):
    """
    Detect (possibly multiple) contact between two geometries using GJK and EPA algorithms.

    We first run the GJK algorithm to find the minimum distance between the two geometries. If the distance is
    smaller than the collision epsilon, we consider the geometries colliding. If they are colliding, we run the EPA
    algorithm to find the exact contact points and normals.

    .. seealso::
    MuJoCo's implementation:
    https://github.com/google-deepmind/mujoco/blob/7dc7a349c5ba2db2d3f8ab50a367d08e2f1afbbc/src/engine/engine_collision_gjk.c#L2259
    """
    # Clear the cache to prepare for this GJK-EPA run.
    clear_cache(gjk_state, i_b)

    # We use MuJoCo's GJK implementation when the compatibility mode is enabled. When it is disabled, we use more
    # robust GJK implementation which has the same overall structure as MuJoCo.
    if ti.static(static_rigid_sim_config.enable_mujoco_compatibility):
        # If any one of the geometries is a sphere or capsule, which are sphere-swept primitives, we can shrink them
        # to a point or line to detect shallow penetration faster.
        is_sphere_swept_geom_a, is_sphere_swept_geom_b = (
            func_is_sphere_swept_geom(geoms_info, i_ga, i_b),
            func_is_sphere_swept_geom(geoms_info, i_gb, i_b),
        )
        shrink_sphere = is_sphere_swept_geom_a or is_sphere_swept_geom_b

        # Run GJK
        for _ in range(2 if shrink_sphere else 1):
            distance = func_gjk(
                geoms_state,
                geoms_info,
                verts_info,
                static_rigid_sim_config,
                collider_state,
                collider_static_config,
                gjk_state,
                gjk_info,
                support_field_info,
                i_ga,
                i_gb,
                i_b,
                shrink_sphere,
            )

            if shrink_sphere:
                # If we shrunk the sphere and capsule to point and line and the distance is larger than the collision
                # epsilon, it means a shallow penetration. Thus we subtract the radius of the sphere and the capsule to
                # get the actual distance. If the distance is smaller than the collision epsilon, it means a deep
                # penetration, which requires the default GJK handling.
                if distance > gjk_info.collision_eps[None]:
                    radius_a, radius_b = 0.0, 0.0
                    if is_sphere_swept_geom_a:
                        radius_a = geoms_info.data[i_ga][0]
                    if is_sphere_swept_geom_b:
                        radius_b = geoms_info.data[i_gb][0]

                    wa = gjk_state.witness.point_obj1[i_b, 0]
                    wb = gjk_state.witness.point_obj2[i_b, 0]
                    n = func_safe_normalize(gjk_info, wb - wa)

                    gjk_state.distance[i_b] = distance - (radius_a + radius_b)
                    gjk_state.witness.point_obj1[i_b, 0] = wa + (radius_a * n)
                    gjk_state.witness.point_obj2[i_b, 0] = wb - (radius_b * n)

                    break

            # Only try shrinking the sphere once
            shrink_sphere = False

            distance = gjk_state.distance[i_b]
            nsimplex = gjk_state.nsimplex[i_b]
            collided = distance < gjk_info.collision_eps[None]

            # To run EPA, we need following conditions:
            # 1. We did not find min. distance with shrink_sphere flag
            # 2. We have a valid GJK simplex (nsimplex > 0)
            # 3. We have a collision (distance < collision_epsilon)
            do_epa = (not shrink_sphere) and collided and (nsimplex > 0)

            if do_epa:
                # Assume touching
                gjk_state.distance[i_b] = 0

                # Initialize polytope
                gjk_state.polytope.nverts[i_b] = 0
                gjk_state.polytope.nfaces[i_b] = 0
                gjk_state.polytope.nfaces_map[i_b] = 0
                gjk_state.polytope.horizon_nedges[i_b] = 0

                # Construct the initial polytope from the GJK simplex
                polytope_flag = EPA_POLY_INIT_RETURN_CODE.SUCCESS
                if nsimplex == 2:
                    polytope_flag = func_epa_init_polytope_2d(
                        geoms_state,
                        geoms_info,
                        verts_info,
                        rigid_global_info,
                        static_rigid_sim_config,
                        collider_state,
                        collider_static_config,
                        gjk_state,
                        gjk_info,
                        support_field_info,
                        i_ga,
                        i_gb,
                        i_b,
                    )
                elif nsimplex == 4:
                    polytope_flag = func_epa_init_polytope_4d(gjk_state, gjk_info, i_ga, i_gb, i_b)

                # Polytope 3D could be used as a fallback for 2D and 4D cases, but it is not necessary
                if (
                    nsimplex == 3
                    or (polytope_flag == EPA_POLY_INIT_RETURN_CODE.P2_FALLBACK3)
                    or (polytope_flag == EPA_POLY_INIT_RETURN_CODE.P4_FALLBACK3)
                ):
                    polytope_flag = func_epa_init_polytope_3d(
                        geoms_state,
                        geoms_info,
                        verts_info,
                        static_rigid_sim_config,
                        collider_state,
                        collider_static_config,
                        gjk_state,
                        gjk_info,
                        support_field_info,
                        i_ga,
                        i_gb,
                        i_b,
                    )

                # Run EPA from the polytope
                if polytope_flag == EPA_POLY_INIT_RETURN_CODE.SUCCESS:
                    i_f = func_epa(
                        geoms_state,
                        geoms_info,
                        verts_info,
                        static_rigid_sim_config,
                        collider_state,
                        collider_static_config,
                        gjk_state,
                        gjk_info,
                        support_field_info,
                        i_ga,
                        i_gb,
                        i_b,
                    )

                    if ti.static(gjk_static_config.enable_mujoco_multi_contact):
                        # To use MuJoCo's multi-contact detection algorithm,
                        # (1) [i_f] should be a valid face index in the polytope (>= 0),
                        # (2) Both of the geometries should be discrete,
                        # (3) [enable_mujoco_multi_contact] should be True. Default to False.
                        if i_f >= 0 and func_is_discrete_geoms(geoms_info, i_ga, i_gb, i_b):
                            func_multi_contact(
                                geoms_state,
                                geoms_info,
                                verts_info,
                                faces_info,
                                gjk_state,
                                gjk_info,
                                i_ga,
                                i_gb,
                                i_b,
                                i_f,
                            )
                            gjk_state.multi_contact_flag[i_b] = True
    else:
        gjk_flag = func_safe_gjk(
            geoms_state,
            geoms_info,
            verts_info,
            rigid_global_info,
            static_rigid_sim_config,
            collider_state,
            collider_static_config,
            gjk_state,
            gjk_info,
            support_field_info,
            i_ga,
            i_gb,
            i_b,
        )
        if gjk_flag == GJK_RETURN_CODE.INTERSECT:
            # Initialize polytope
            gjk_state.polytope.nverts[i_b] = 0
            gjk_state.polytope.nfaces[i_b] = 0
            gjk_state.polytope.nfaces_map[i_b] = 0
            gjk_state.polytope.horizon_nedges[i_b] = 0

            # Construct the initial polytope from the GJK simplex
            func_safe_epa_init(gjk_state, gjk_info, i_ga, i_gb, i_b)

            # Run EPA from the polytope
            func_safe_epa(
                geoms_state,
                geoms_info,
                verts_info,
                rigid_global_info,
                static_rigid_sim_config,
                collider_state,
                collider_static_config,
                gjk_state,
                gjk_info,
                support_field_info,
                i_ga,
                i_gb,
                i_b,
            )

    # Compute the final contact points and normals.
    n_contacts = 0
    gjk_state.is_col[i_b] = gjk_state.distance[i_b] < 0.0
    gjk_state.penetration[i_b] = -gjk_state.distance[i_b] if gjk_state.is_col[i_b] else 0.0

    if gjk_state.is_col[i_b]:
        for i in range(gjk_state.n_witness[i_b]):
            w1 = gjk_state.witness.point_obj1[i_b, i]
            w2 = gjk_state.witness.point_obj2[i_b, i]
            contact_pos = 0.5 * (w1 + w2)

            normal = w2 - w1
            normal_len = normal.norm()
            if normal_len < gjk_info.FLOAT_MIN[None]:
                continue

            normal = normal / normal_len

            gjk_state.contact_pos[i_b, n_contacts] = contact_pos
            gjk_state.normal[i_b, n_contacts] = normal
            n_contacts += 1

    gjk_state.n_contacts[i_b] = n_contacts
    # If there are no contacts, we set the penetration and is_col to 0.
    # FIXME: When we use if statement here, it leads to a bug in some backends (e.g. x86 cpu). Need to investigate.
    gjk_state.is_col[i_b] = False if n_contacts == 0 else gjk_state.is_col[i_b]
    gjk_state.penetration[i_b] = 0.0 if n_contacts == 0 else gjk_state.penetration[i_b]


@ti.func
def func_gjk(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    support_field_info: array_class.SupportFieldInfo,
    i_ga,
    i_gb,
    i_b,
    shrink_sphere,
):
    """
    GJK algorithm to compute the minimum distance between two convex objects.

    This is a thin wrapper that extracts geometry poses from global state
    and delegates to the thread-local version for the actual computation.

    This implementation is based on the MuJoCo implementation.

    Parameters
    ----------
    shrink_sphere: bool
        If True, use point and line support functions for sphere and capsule geometries, respectively.

    .. seealso::
    MuJoCo's original implementation:
    https://github.com/google-deepmind/mujoco/blob/7dc7a349c5ba2db2d3f8ab50a367d08e2f1afbbc/src/engine/engine_collision_gjk.c#L171
    """
    pos_a = geoms_state.pos[i_ga, i_b]
    quat_a = geoms_state.quat[i_ga, i_b]
    pos_b = geoms_state.pos[i_gb, i_b]
    quat_b = geoms_state.quat[i_gb, i_b]
    return gjk_local.func_gjk_local(
        geoms_info,
        verts_info,
        static_rigid_sim_config,
        collider_state,
        collider_static_config,
        gjk_state,
        gjk_info,
        support_field_info,
        i_ga,
        i_gb,
        i_b,
        pos_a,
        quat_a,
        pos_b,
        quat_b,
        shrink_sphere,
    )


@ti.func
def func_gjk_intersect(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    support_field_info: array_class.SupportFieldInfo,
    i_ga,
    i_gb,
    i_b,
):
    """
    Check if two objects intersect using the GJK algorithm.

    This is a thin wrapper that extracts geometry poses from global state
    and delegates to the thread-local version for the actual computation.

    This function refines the simplex until it contains the origin or it is determined
    that the objects are separated. It is used to check if objects intersect, not to
    find the minimum distance between them.
    """
    pos_a = geoms_state.pos[i_ga, i_b]
    quat_a = geoms_state.quat[i_ga, i_b]
    pos_b = geoms_state.pos[i_gb, i_b]
    quat_b = geoms_state.quat[i_gb, i_b]
    return gjk_local.func_gjk_intersect_local(
        geoms_info,
        verts_info,
        static_rigid_sim_config,
        collider_state,
        collider_static_config,
        gjk_state,
        gjk_info,
        support_field_info,
        i_ga,
        i_gb,
        i_b,
        pos_a,
        quat_a,
        pos_b,
        quat_b,
    )


@ti.func
def func_gjk_triangle_info(
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    i_b,
    i_va,
    i_vb,
    i_vc,
):
    """
    Compute normal and signed distance of the triangle face on the simplex from the origin.
    """
    vertex_1 = gjk_state.simplex_vertex_intersect.mink[i_b, i_va]
    vertex_2 = gjk_state.simplex_vertex_intersect.mink[i_b, i_vb]
    vertex_3 = gjk_state.simplex_vertex_intersect.mink[i_b, i_vc]

    normal = (vertex_3 - vertex_1).cross(vertex_2 - vertex_1)
    normal_length = normal.norm()

    sdist = 0.0
    if (normal_length > gjk_info.FLOAT_MIN[None]) and (normal_length < gjk_info.FLOAT_MAX[None]):
        normal = normal * (1.0 / normal_length)
        sdist = normal.dot(vertex_1)
    else:
        # If the normal length is unstable, return max distance.
        sdist = gjk_info.FLOAT_MAX[None]

    return normal, sdist


@ti.func
def func_gjk_subdistance(
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    i_b,
    n,
):
    """
    Compute the barycentric coordinates of the closest point to the origin in the n-simplex.

    .. seealso::
    Montanari, Mattia, Nik Petrinic, and Ettore Barbieri. "Improving the GJK algorithm for faster and more reliable
    distance queries between convex objects." ACM Transactions on Graphics (TOG) 36.3 (2017): 1-17.
    https://dl.acm.org/doi/10.1145/3072959.3083724
    """
    _lambda = ti.math.vec4(1.0, 0.0, 0.0, 0.0)

    # Whether or not the subdistance was computed successfully for the n-simplex.
    flag = RETURN_CODE.SUCCESS

    dmin = gjk_info.FLOAT_MAX[None]

    if n == 4:
        _lambda, flag3d = func_gjk_subdistance_3d(gjk_state, i_b, 0, 1, 2, 3)
        flag = flag3d

    if (flag == RETURN_CODE.FAIL) or n == 3:
        failed_3d = n == 4
        num_iter = 1
        if failed_3d:
            # Iterate through 4 faces of the tetrahedron
            num_iter = 4

        for i in range(num_iter):
            k_1, k_2, k_3 = i, (i + 1) % 4, (i + 2) % 4
            _lambda2d, flag2d = func_gjk_subdistance_2d(gjk_state, gjk_info, i_b, k_1, k_2, k_3)

            if failed_3d:
                if flag2d == RETURN_CODE.SUCCESS:
                    closest_point = func_simplex_vertex_linear_comb(gjk_state, i_b, 2, k_1, k_2, k_3, 0, _lambda2d, 3)
                    d = closest_point.dot(closest_point)
                    if d < dmin:
                        dmin = d
                        _lambda.fill(0.0)
                        _lambda[k_1] = _lambda2d[0]
                        _lambda[k_2] = _lambda2d[1]
                        _lambda[k_3] = _lambda2d[2]
            else:
                if flag2d == RETURN_CODE.SUCCESS:
                    _lambda = _lambda2d
                flag = flag2d

    if (flag == RETURN_CODE.FAIL) or n == 2:
        failed_3d = n == 4
        failed_2d = n == 3

        num_iter = 1
        if failed_3d:
            # Iterate through 6 edges of the tetrahedron
            num_iter = 6
        elif failed_2d:
            # Iterate through 3 edges of the triangle
            num_iter = 3

        for i in range(num_iter):
            k_1, k_2 = i, (i + 1) % 3
            if i >= 3:
                k_1, k_2 = i - 3, 3

            _lambda1d = func_gjk_subdistance_1d(gjk_state, i_b, k_1, k_2)

            if failed_3d or failed_2d:
                closest_point = func_simplex_vertex_linear_comb(gjk_state, i_b, 2, k_1, k_2, 0, 0, _lambda1d, 2)
                d = closest_point.dot(closest_point)
                if d < dmin:
                    dmin = d
                    _lambda.fill(0.0)
                    _lambda[k_1] = _lambda1d[0]
                    _lambda[k_2] = _lambda1d[1]
            else:
                _lambda = _lambda1d

    return _lambda


@ti.func
def func_gjk_subdistance_3d(
    gjk_state: array_class.GJKState,
    i_b,
    i_s1,
    i_s2,
    i_s3,
    i_s4,
):
    """
    Compute the barycentric coordinates of the closest point to the origin in the 3-simplex (tetrahedron).
    """
    flag = RETURN_CODE.FAIL
    _lambda = gs.ti_vec4(0, 0, 0, 0)

    # Simplex vertices
    s1 = gjk_state.simplex_vertex.mink[i_b, i_s1]
    s2 = gjk_state.simplex_vertex.mink[i_b, i_s2]
    s3 = gjk_state.simplex_vertex.mink[i_b, i_s3]
    s4 = gjk_state.simplex_vertex.mink[i_b, i_s4]

    # Compute the cofactors to find det(M), which corresponds to the signed volume of the tetrahedron
    Cs = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
    for i in range(4):
        v1, v2, v3 = s2, s3, s4
        if i == 1:
            v1, v2, v3 = s1, s3, s4
        elif i == 2:
            v1, v2, v3 = s1, s2, s4
        elif i == 3:
            v1, v2, v3 = s1, s2, s3
        Cs[i] = func_det3(v1, v2, v3)
    Cs[0], Cs[2] = -Cs[0], -Cs[2]
    m_det = Cs.sum()

    # Compare sign of the cofactors with the determinant
    scs = gs.ti_ivec4(0, 0, 0, 0)
    for i in range(4):
        scs[i] = func_compare_sign(Cs[i], m_det)

    if scs.all():
        # If all barycentric coordinates are positive, the origin is inside the tetrahedron
        _lambda = Cs / m_det
        flag = RETURN_CODE.SUCCESS

    return _lambda, flag


@ti.func
def func_gjk_subdistance_2d(
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    i_b,
    i_s1,
    i_s2,
    i_s3,
):
    """
    Compute the barycentric coordinates of the closest point to the origin in the 2-simplex (triangle).
    """
    _lambda = ti.math.vec4(0, 0, 0, 0)
    flag = RETURN_CODE.FAIL

    # Project origin onto affine hull of the simplex (triangle)
    proj_orig, proj_flag = func_project_origin_to_plane(
        gjk_info,
        gjk_state.simplex_vertex.mink[i_b, i_s1],
        gjk_state.simplex_vertex.mink[i_b, i_s2],
        gjk_state.simplex_vertex.mink[i_b, i_s3],
    )

    if proj_flag == RETURN_CODE.SUCCESS:
        # We should find the barycentric coordinates of the projected point, but the linear system is not square:
        # [ s1.x, s2.x, s3.x ] [ l1 ] = [ proj_o.x ]
        # [ s1.y, s2.y, s3.y ] [ l2 ] = [ proj_o.y ]
        # [ s1.z, s2.z, s3.z ] [ l3 ] = [ proj_o.z ]
        # [ 1,    1,    1,   ] [ ?  ] = [ 1.0 ]
        # So we remove one row before solving the system. We exclude the axis with the largest projection of the
        # simplex using the minors of the above linear system.
        s1 = gjk_state.simplex_vertex.mink[i_b, i_s1]
        s2 = gjk_state.simplex_vertex.mink[i_b, i_s2]
        s3 = gjk_state.simplex_vertex.mink[i_b, i_s3]

        ms = gs.ti_vec3(
            s2[1] * s3[2] - s2[2] * s3[1] - s1[1] * s3[2] + s1[2] * s3[1] + s1[1] * s2[2] - s1[2] * s2[1],
            s2[0] * s3[2] - s2[2] * s3[0] - s1[0] * s3[2] + s1[2] * s3[0] + s1[0] * s2[2] - s1[2] * s2[0],
            s2[0] * s3[1] - s2[1] * s3[0] - s1[0] * s3[1] + s1[1] * s3[0] + s1[0] * s2[1] - s1[1] * s2[0],
        )
        absms = ti.abs(ms)

        m_max = 0.0
        s1_2d, s2_2d, s3_2d = gs.ti_vec2(0, 0), gs.ti_vec2(0, 0), gs.ti_vec2(0, 0)
        proj_orig_2d = gs.ti_vec2(0, 0)

        for i in range(3):
            if absms[i] >= absms[(i + 1) % 3] and absms[i] >= absms[(i + 2) % 3]:
                # Remove the i-th row from the linear system
                m_max = ms[i]

                i0, i1 = (i + 1) % 3, (i + 2) % 3
                if i == 1:
                    i0, i1 = i1, i0

                s1_2d[0], s1_2d[1] = s1[i0], s1[i1]
                s2_2d[0], s2_2d[1] = s2[i0], s2[i1]
                s3_2d[0], s3_2d[1] = s3[i0], s3[i1]
                proj_orig_2d[0] = proj_orig[i0]
                proj_orig_2d[1] = proj_orig[i1]
                break

        # Now we find the barycentric coordinates of the projected point by solving the linear system:
        # [ s1_2d.x, s2_2d.x, s3_2d.x ] [ l1 ] = [ proj_orig_2d.x ]
        # [ s1_2d.y, s2_2d.y, s3_2d.y ] [ l2 ] = [ proj_orig_2d.y ]
        # [ 1,       1,       1,      ] [ l3 ] = [ 1.0 ]
        cs = gs.ti_vec3(0, 0, 0)
        for i in range(3):
            s2d0, s2d1 = s2_2d, s3_2d
            if i == 1:
                s2d0, s2d1 = s3_2d, s1_2d
            elif i == 2:
                s2d0, s2d1 = s1_2d, s2_2d
            # Corresponds to the signed area of 2-simplex (triangle): (proj_orig_2d, s2d0, s2d1)
            cs[i] = (
                proj_orig_2d[0] * s2d0[1]
                + proj_orig_2d[1] * s2d1[0]
                + s2d0[0] * s2d1[1]
                - proj_orig_2d[0] * s2d1[1]
                - proj_orig_2d[1] * s2d0[0]
                - s2d1[0] * s2d0[1]
            )

        # Compare sign of the cofactors with the determinant
        scs = gs.ti_ivec3(0, 0, 0)
        for i in range(3):
            scs[i] = func_compare_sign(cs[i], m_max)

        if scs.all():
            # If all barycentric coordinates are positive, the origin is inside the 2-simplex (triangle)
            for i in ti.static(range(3)):
                _lambda[i] = cs[i] / m_max
            flag = RETURN_CODE.SUCCESS

    return _lambda, flag


@ti.func
def func_gjk_subdistance_1d(
    gjk_state: array_class.GJKState,
    i_b,
    i_s1,
    i_s2,
):
    """
    Compute the barycentric coordinates of the closest point to the origin in the 1-simplex (line segment).
    """
    _lambda = gs.ti_vec4(0, 0, 0, 0)

    s1 = gjk_state.simplex_vertex.mink[i_b, i_s1]
    s2 = gjk_state.simplex_vertex.mink[i_b, i_s2]
    p_o = func_project_origin_to_line(s1, s2)

    mu_max = 0.0
    index = -1
    for i in range(3):
        mu = s1[i] - s2[i]
        if ti.abs(mu) >= ti.abs(mu_max):
            mu_max = mu
            index = i

    C1 = p_o[index] - s2[index]
    C2 = s1[index] - p_o[index]

    # Determine if projection of origin lies inside 1-simplex
    if func_compare_sign(mu_max, C1) and func_compare_sign(mu_max, C2):
        _lambda[0] = C1 / mu_max
        _lambda[1] = C2 / mu_max
    else:
        _lambda[0] = 0.0
        _lambda[1] = 1.0

    return _lambda


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
def func_is_discrete_geoms(
    geoms_info: array_class.GeomsInfo,
    i_ga,
    i_gb,
    i_b,
):
    """
    Check if the given geoms are discrete geometries.
    """
    return func_is_discrete_geom(geoms_info, i_ga, i_b) and func_is_discrete_geom(geoms_info, i_gb, i_b)


@ti.func
def func_is_discrete_geom(
    geoms_info: array_class.GeomsInfo,
    i_g,
    i_b,
):
    """
    Check if the given geom is a discrete geometry.
    """
    geom_type = geoms_info.type[i_g]
    return geom_type == gs.GEOM_TYPE.MESH or geom_type == gs.GEOM_TYPE.BOX


@ti.func
def func_is_sphere_swept_geom(
    geoms_info: array_class.GeomsInfo,
    i_g,
    i_b,
):
    """
    Check if the given geoms are sphere-swept geometries.
    """
    geom_type = geoms_info.type[i_g]
    return geom_type == gs.GEOM_TYPE.SPHERE or geom_type == gs.GEOM_TYPE.CAPSULE


@ti.func
def func_support(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    support_field_info: array_class.SupportFieldInfo,
    i_ga,
    i_gb,
    i_b,
    dir,
    shrink_sphere,
):
    """
    Find support points on the two objects using [dir].

    Parameters:
    ----------
    dir: gs.ti_vec3
        The direction in which to find the support points, from [ga] (obj 1) to [gb] (obj 2).
    """
    pos_a = geoms_state.pos[i_ga, i_b]
    quat_a = geoms_state.quat[i_ga, i_b]
    pos_b = geoms_state.pos[i_gb, i_b]
    quat_b = geoms_state.quat[i_gb, i_b]
    return gjk_local.func_support_local(
        geoms_info,
        verts_info,
        static_rigid_sim_config,
        collider_state,
        collider_static_config,
        gjk_state,
        gjk_info,
        support_field_info,
        i_ga,
        i_gb,
        i_b,
        dir,
        pos_a,
        quat_a,
        pos_b,
        quat_b,
        shrink_sphere,
    )


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


@ti.func
def func_project_origin_to_line(
    v1,
    v2,
):
    """
    Project the origin onto the line defined by the simplex vertices.

    P = v2 - ((v1 * diff) / (diff * diff)) * diff
    """
    diff = v2 - v1
    k = v2.dot(diff) / diff.dot(diff)
    P = v2 - k * diff

    return P


@ti.func
def func_simplex_vertex_linear_comb(
    gjk_state: array_class.GJKState,
    i_b,
    i_v,
    i_s1,
    i_s2,
    i_s3,
    i_s4,
    _lambda,
    n,
):
    """
    Compute the linear combination of the simplex vertices

    Parameters:
    ----------
    i_v: int
        Which vertex to use (0: obj1, 1: obj2, 2: minkowski)
    n: int
        Number of vertices to combine, combine the first n vertices
    """
    res = gs.ti_vec3(0, 0, 0)

    s1 = gjk_state.simplex_vertex.obj1[i_b, i_s1]
    s2 = gjk_state.simplex_vertex.obj1[i_b, i_s2]
    s3 = gjk_state.simplex_vertex.obj1[i_b, i_s3]
    s4 = gjk_state.simplex_vertex.obj1[i_b, i_s4]
    if i_v == 1:
        s1 = gjk_state.simplex_vertex.obj2[i_b, i_s1]
        s2 = gjk_state.simplex_vertex.obj2[i_b, i_s2]
        s3 = gjk_state.simplex_vertex.obj2[i_b, i_s3]
        s4 = gjk_state.simplex_vertex.obj2[i_b, i_s4]
    elif i_v == 2:
        s1 = gjk_state.simplex_vertex.mink[i_b, i_s1]
        s2 = gjk_state.simplex_vertex.mink[i_b, i_s2]
        s3 = gjk_state.simplex_vertex.mink[i_b, i_s3]
        s4 = gjk_state.simplex_vertex.mink[i_b, i_s4]

    c1 = _lambda[0]
    c2 = _lambda[1]
    c3 = _lambda[2]
    c4 = _lambda[3]

    if n == 1:
        res = s1 * c1
    elif n == 2:
        res = s1 * c1 + s2 * c2
    elif n == 3:
        res = s1 * c1 + s2 * c2 + s3 * c3
    else:
        res = s1 * c1 + s2 * c2 + s3 * c3 + s4 * c4
    return res


@ti.func
def func_det3(
    v1,
    v2,
    v3,
):
    """
    Compute the determinant of a 3x3 matrix M = [v1 | v2 | v3].
    """
    return (
        v1[0] * (v2[1] * v3[2] - v2[2] * v3[1])
        - v1[1] * (v2[0] * v3[2] - v2[2] * v3[0])
        + v1[2] * (v2[0] * v3[1] - v2[1] * v3[0])
    )


@ti.func
def support_mesh(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    direction,
    i_g,
    i_b,
    i_o,
):
    """
    Find the support point on a mesh in the given direction.
    """
    pos = geoms_state.pos[i_g, i_b]
    quat = geoms_state.quat[i_g, i_b]
    return gjk_local.support_mesh_local(
        geoms_info, verts_info, gjk_state, gjk_info, direction, i_g, pos, quat, i_b, i_o
    )


@ti.func
def support_driver(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    support_field_info: array_class.SupportFieldInfo,
    direction,
    i_g,
    i_b,
    i_o,
    shrink_sphere,
):
    """
    @ shrink_sphere: If True, use point and line support for sphere and capsule.
    """
    pos = geoms_state.pos[i_g, i_b]
    quat = geoms_state.quat[i_g, i_b]
    return gjk_local.support_driver_local(
        geoms_info,
        verts_info,
        static_rigid_sim_config,
        collider_state,
        collider_static_config,
        gjk_state,
        gjk_info,
        support_field_info,
        direction,
        i_g,
        pos,
        quat,
        i_b,
        i_o,
        shrink_sphere,
    )


@ti.func
def func_safe_gjk(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    support_field_info: array_class.SupportFieldInfo,
    i_ga,
    i_gb,
    i_b,
):
    """
    Safe GJK algorithm to compute the minimum distance between two convex objects.

    This is a thin wrapper that extracts geometry poses from global state
    and delegates to the thread-local version for the actual computation.

    This implementation is safer than the one based on the MuJoCo implementation for the following reasons:
    1) It guarantees that the origin is strictly inside the tetrahedron when the intersection is detected.
    2) It guarantees to generate a non-degenerate tetrahedron if there is no numerical error, which is necessary
    for the following EPA algorithm to work correctly.
    3) When computing the face normals on the simplex, it uses a more robust method than using the origin.

    TODO: This implementation could be improved by using shrink_sphere option as the MuJoCo implementation does.
    TODO: This implementation could be further improved by referencing the follow-up work shown below.

    .. seealso::
    Original paper:
    Gilbert, Elmer G., Daniel W. Johnson, and S. Sathiya Keerthi.
    "A fast procedure for computing the distance between complex objects in three-dimensional space."
    IEEE Journal on Robotics and Automation 4.2 (2002): 193-203.

    Further improvements:
    Cameron, Stephen. "Enhancing GJK: Computing minimum and penetration distances between convex polyhedra."
    Proceedings of international conference on robotics and automation. Vol. 4. IEEE, 1997.
    https://www.cs.ox.ac.uk/people/stephen.cameron/distances/gjk2.4/

    Montaut, Louis, et al. "Collision detection accelerated: An optimization perspective."
    https://arxiv.org/abs/2205.09663
    """
    pos_a = geoms_state.pos[i_ga, i_b]
    quat_a = geoms_state.quat[i_ga, i_b]
    pos_b = geoms_state.pos[i_gb, i_b]
    quat_b = geoms_state.quat[i_gb, i_b]
    return gjk_local.func_safe_gjk_local(
        geoms_info,
        verts_info,
        rigid_global_info,
        static_rigid_sim_config,
        collider_state,
        collider_static_config,
        gjk_state,
        gjk_info,
        support_field_info,
        i_ga,
        i_gb,
        pos_a,
        quat_a,
        pos_b,
        quat_b,
        i_b,
    )


@ti.func
def func_is_new_simplex_vertex_valid(
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    i_b,
    id1,
    id2,
    mink,
):
    """
    Check validity of the incoming simplex vertex (defined by id1, id2 and mink).

    To be a new valid simplex vertex, it should satisfy the following conditions:
    1) The vertex should not be already in the simplex.
    2) The simplex should not be degenerate after insertion.
    """
    return (not func_is_new_simplex_vertex_duplicate(gjk_state, i_b, id1, id2)) and (
        not func_is_new_simplex_vertex_degenerate(gjk_state, gjk_info, i_b, mink)
    )


@ti.func
def func_is_new_simplex_vertex_duplicate(
    gjk_state: array_class.GJKState,
    i_b,
    id1,
    id2,
):
    """
    Check if the incoming simplex vertex is already in the simplex.
    """
    nverts = gjk_state.simplex.nverts[i_b]
    found = False
    for i in range(nverts):
        if id1 == -1 or (gjk_state.simplex_vertex.id1[i_b, i] != id1):
            continue
        if id2 == -1 or (gjk_state.simplex_vertex.id2[i_b, i] != id2):
            continue
        found = True
        break
    return found


@ti.func
def func_is_new_simplex_vertex_degenerate(
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    i_b,
    mink,
):
    """
    Check if the simplex becomes degenerate after inserting a new vertex, assuming that the current simplex is okay.
    """
    is_degenerate = False

    # Check if the new vertex is not very close to the existing vertices
    nverts = gjk_state.simplex.nverts[i_b]
    for i in range(nverts):
        if (gjk_state.simplex_vertex.mink[i_b, i] - mink).norm_sqr() < (gjk_info.simplex_max_degeneracy_sq[None]):
            is_degenerate = True
            break

    if not is_degenerate:
        # Check the validity based on the simplex dimension
        if nverts == 2:
            # Becomes a triangle if valid, check if the three vertices are not collinear
            is_degenerate = func_is_colinear(
                gjk_info,
                gjk_state.simplex_vertex.mink[i_b, 0],
                gjk_state.simplex_vertex.mink[i_b, 1],
                mink,
            )
        elif nverts == 3:
            # Becomes a tetrahedron if valid, check if the four vertices are not coplanar
            is_degenerate = func_is_coplanar(
                gjk_info,
                gjk_state.simplex_vertex.mink[i_b, 0],
                gjk_state.simplex_vertex.mink[i_b, 1],
                gjk_state.simplex_vertex.mink[i_b, 2],
                mink,
            )

    return is_degenerate


@ti.func
def func_is_colinear(
    gjk_info: array_class.GJKInfo,
    v1,
    v2,
    v3,
):
    """
    Check if three points are collinear.

    This function assumes that every pair of points is non-degenerate, i.e. no pair of points is identical.
    """
    e1 = v2 - v1
    e2 = v3 - v1
    normal = e1.cross(e2)
    return normal.norm_sqr() < (gjk_info.simplex_max_degeneracy_sq[None]) * e1.norm_sqr() * e2.norm_sqr()


@ti.func
def func_is_coplanar(
    gjk_info: array_class.GJKInfo,
    v1,
    v2,
    v3,
    v4,
):
    """
    Check if four points are coplanar.

    This function assumes that every triplet of points is non-degenerate, i.e. no triplet of points is collinear.
    """
    e1 = (v2 - v1).normalized()
    e2 = (v3 - v1).normalized()
    normal = e1.cross(e2)
    diff = v4 - v1
    return (normal.dot(diff) ** 2) < (gjk_info.simplex_max_degeneracy_sq[None]) * normal.norm_sqr() * diff.norm_sqr()


@ti.func
def func_search_valid_simplex_vertex(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    support_field_info: array_class.SupportFieldInfo,
    i_ga,
    i_gb,
    i_b,
):
    """
    Search for a valid simplex vertex (non-duplicate, non-degenerate) in the Minkowski difference.

    This is a thin wrapper that extracts geometry poses from global state
    and delegates to the thread-local version for the actual computation.
    """
    pos_a = geoms_state.pos[i_ga, i_b]
    quat_a = geoms_state.quat[i_ga, i_b]
    pos_b = geoms_state.pos[i_gb, i_b]
    quat_b = geoms_state.quat[i_gb, i_b]
    return gjk_local.func_search_valid_simplex_vertex_local(
        geoms_info,
        verts_info,
        rigid_global_info,
        static_rigid_sim_config,
        collider_state,
        collider_static_config,
        gjk_state,
        gjk_info,
        support_field_info,
        i_ga,
        i_gb,
        pos_a,
        quat_a,
        pos_b,
        quat_b,
        i_b,
    )


@ti.func
def func_num_discrete_geom_vertices(
    geoms_info: array_class.GeomsInfo,
    i_g,
    i_b,
):
    """
    Count the number of discrete vertices in the geometry.
    """
    vert_start = geoms_info.vert_start[i_g]
    vert_end = geoms_info.vert_end[i_g]
    count = vert_end - vert_start
    return count


@ti.func
def func_get_discrete_geom_vertex(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    i_g,
    i_b,
    i_v,
):
    """
    Get the discrete vertex of the geometry for the given index [i_v].

    This is a thin wrapper that extracts geometry pose from global state
    and delegates to the thread-local version for the actual computation.
    """
    pos = geoms_state.pos[i_g, i_b]
    quat = geoms_state.quat[i_g, i_b]
    return gjk_local.func_get_discrete_geom_vertex_local(geoms_info, verts_info, i_g, pos, quat, i_v)


@ti.func
def func_safe_gjk_triangle_info(
    gjk_state: array_class.GJKState,
    i_b,
    i_ta,
    i_tb,
    i_tc,
    i_apex,
):
    """
    Compute normal and signed distance of the triangle face on the simplex from the origin.

    The triangle is defined by the vertices [i_ta], [i_tb], and [i_tc], and the apex is used to orient the triangle
    normal, so that it points outward from the simplex. Thus, if the origin is inside the simplex in terms of this
    triangle, the signed distance will be positive.
    """
    vertex_1 = gjk_state.simplex_vertex.mink[i_b, i_ta]
    vertex_2 = gjk_state.simplex_vertex.mink[i_b, i_tb]
    vertex_3 = gjk_state.simplex_vertex.mink[i_b, i_tc]
    apex_vertex = gjk_state.simplex_vertex.mink[i_b, i_apex]

    # This normal is guaranteed to be non-zero because we build the simplex avoiding degenerate vertices.
    normal = (vertex_3 - vertex_1).cross(vertex_2 - vertex_1).normalized()

    # Reorient the normal to point outward from the simplex
    if normal.dot(apex_vertex - vertex_1) > 0.0:
        normal = -normal

    # Compute the signed distance from the origin to the triangle plane
    sdist = normal.dot(vertex_1)

    return normal, sdist


@ti.func
def func_safe_gjk_support(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    support_field_info: array_class.SupportFieldInfo,
    i_ga,
    i_gb,
    i_b,
    dir,
):
    """
    Find support points on the two objects using [dir] to use in the [safe_gjk] algorithm.

    This is a thin wrapper that extracts geometry poses from global state
    and delegates to the thread-local version for the actual computation.

    This is a more robust version of the support function that finds only one pair of support points, because this
    function perturbs the support direction to find the best support points that guarantee non-degenerate simplex
    in the GJK algorithm.

    Parameters:
    ----------
    dir: gs.ti_vec3
        The unit direction in which to find the support points, from [ga] (obj 1) to [gb] (obj 2).
    """
    pos_a = geoms_state.pos[i_ga, i_b]
    quat_a = geoms_state.quat[i_ga, i_b]
    pos_b = geoms_state.pos[i_gb, i_b]
    quat_b = geoms_state.quat[i_gb, i_b]
    return gjk_local.func_safe_gjk_support_local(
        geoms_info,
        verts_info,
        rigid_global_info,
        static_rigid_sim_config,
        collider_state,
        collider_static_config,
        gjk_state,
        gjk_info,
        support_field_info,
        i_ga,
        i_gb,
        pos_a,
        quat_a,
        pos_b,
        quat_b,
        i_b,
        dir,
    )


@ti.func
def count_support_driver(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    support_field_info: array_class.SupportFieldInfo,
    d,
    i_g,
    i_b,
):
    """
    Count the number of possible support points in the given direction.

    This is a thin wrapper that extracts geometry pose from global state
    and delegates to the thread-local version for the actual computation.
    """
    quat = geoms_state.quat[i_g, i_b]
    return gjk_local.count_support_driver_local(
        geoms_info,
        support_field_info,
        d,
        i_g,
        quat,
    )


@ti.func
def func_count_support(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    support_field_info: array_class.SupportFieldInfo,
    i_ga,
    i_gb,
    i_b,
    dir,
):
    """
    Count the number of possible pairs of support points on the two objects in the given direction [d].

    This is a thin wrapper that extracts geometry poses from global state
    and delegates to the thread-local version for the actual computation.
    """
    pos_a = geoms_state.pos[i_ga, i_b]
    quat_a = geoms_state.quat[i_ga, i_b]
    pos_b = geoms_state.pos[i_gb, i_b]
    quat_b = geoms_state.quat[i_gb, i_b]
    return gjk_local.func_count_support_local(
        geoms_info,
        support_field_info,
        i_ga,
        i_gb,
        pos_a,
        quat_a,
        pos_b,
        quat_b,
        dir,
    )


# Import EPA functions (circular import resolved by importing after all gjk functions are defined)
from .epa import (
    func_epa,
    func_epa_witness,
    func_epa_horizon,
    func_add_edge_to_horizon,
    func_get_edge_idx,
    func_delete_face_from_polytope,
    func_epa_insert_vertex_to_polytope,
    func_epa_init_polytope_2d,
    func_epa_init_polytope_3d,
    func_epa_init_polytope_4d,
    func_epa_support,
    func_attach_face_to_polytope,
    func_replace_simplex_3,
    func_safe_epa,
    func_safe_epa_witness,
    func_safe_epa_init,
    func_safe_attach_face_to_polytope,
    func_plane_normal,
)

# Import EPA local functions for thread-safe multi-contact
from .epa_local import func_safe_epa_local

# Import multi-contact functions
from .multi_contact import (
    func_multi_contact,
    func_simplex_dim,
    func_potential_box_normals,
    func_cmp_bit,
    func_box_normal_from_collision_normal,
    func_potential_mesh_normals,
    func_find_aligned_faces,
    func_potential_box_edge_normals,
    func_potential_mesh_edge_normals,
    func_safe_normalize,
    func_find_aligned_edge_face,
    func_box_face,
    func_mesh_face,
    func_clip_polygon,
    func_halfspace,
    func_plane_intersect,
    func_approximate_polygon_with_quad,
    func_quadrilateral_area,
)


from genesis.utils.deprecated_module_wrapper import create_virtual_deprecated_module

create_virtual_deprecated_module(__name__, "genesis.engine.solvers.rigid.gjk_decomp")
