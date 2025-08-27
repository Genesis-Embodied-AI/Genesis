from enum import IntEnum
import gstaichi as ti
import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class
import genesis.engine.solvers.rigid.support_field_decomp as support_field
from dataclasses import dataclass


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


@ti.data_oriented
class GJK:
    @ti.data_oriented
    class GJKStaticConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def __init__(self, rigid_solver):
        # Initialize static configuration.
        # MuJoCo's multi-contact detection algorithm is disabled by default, because it is often less stable than the
        # other multi-contact detection algorithm. However, we keep the code here for compatibility with MuJoCo and for
        # possible future use.
        enable_mujoco_multi_contact = False
        gjk_max_iterations = 50
        epa_max_iterations = 50
        polytope_max_faces = 6 * epa_max_iterations

        self._gjk_static_config = GJK.GJKStaticConfig(
            # The maximum number of contacts per pair is related to the maximum number of contact manifold vertices.
            # MuJoCo sets [max_contacts_per_pair] to 50 and [max_contact_polygon_verts] to 150, when it uses
            # multi-contact detection algorithm, assuming that the faces could have more than 4 vertices. However, we
            # set them to smaller values, because we do not expect the faces to have more than 4 vertices in most cases,
            # and we want to keep the memory usage low.
            max_contacts_per_pair=8 if enable_mujoco_multi_contact else 1,
            max_contact_polygon_verts=30 if enable_mujoco_multi_contact else 1,
            # Maximum number of iterations for GJK and EPA algorithms
            gjk_max_iterations=gjk_max_iterations,
            epa_max_iterations=epa_max_iterations,
            # When using larger minimum values (e.g. gs.EPS), unstability could occur for some examples (e.g. box pyramid).
            # Also, since different backends could have different precisions (e.g. computing vector norm), we use a very
            # small value, so that there is no discrepancy between backends.
            FLOAT_MIN=gs.np_float(1e-15),
            FLOAT_MIN_SQ=gs.np_float(1e-15) ** 2,
            FLOAT_MAX=gs.np_float(1e15),
            FLOAT_MAX_SQ=gs.np_float(1e15) ** 2,
            # Tolerance for stopping GJK and EPA algorithms when they converge (only for non-discrete geometries).
            tolerance=gs.np_float(1e-6),
            # If the distance between two objects is smaller than this value, we consider them colliding.
            collision_eps=gs.np_float(1e-6),
            # In safe GJK, we do not allow degenerate simplex to happen, because it becomes the main reason of EPA errors.
            # To prevent degeneracy, we throw away the simplex that has smaller degeneracy measure (e.g. colinearity,
            # coplanarity) than this threshold. This value has been experimentally determined based on the examples that
            # we currently have (e.g. pyramid, tower, ...), but it could be further tuned based on the future examples.
            simplex_max_degeneracy_sq=gs.np_float(1e-5) ** 2,
            # 6 * epa_max_iterations is the maximum number of faces in the polytope.
            polytope_max_faces=polytope_max_faces,
            # Threshold for reprojection error when we compute the witness points from the polytope. In computing the
            # witness points, we project the origin onto the polytope faces and compute the barycentric coordinates of the
            # projected point. To confirm the projection is valid, we compute the projected point using the barycentric
            # coordinates and compare it with the original projected point. If the difference is larger than this threshold,
            # we consider the projection invalid, because it means numerical errors are too large. This value has been
            # experimentally determined based on the examples that we currently have (e.g. pyramid, tower, ...). We observed
            # the error usually reaches around 5e-4, so we set the threshold to 1e-5 to be safe. However, this value could
            # be further tuned based on the future examples.
            polytope_max_reprojection_error=gs.np_float(1e-5),
            # This is disabled by default, because it is often less stable than the other multi-contact detection algorithm.
            # However, we keep the code here for compatibility with MuJoCo and for possible future use.
            enable_mujoco_multi_contact=enable_mujoco_multi_contact,
            # Tolerance for normal alignment between (face-face) or (edge-face). The normals should align within this
            # tolerance to be considered as a valid parallel contact. The values are cosine and sine of 1.6e-3,
            # respectively, and brought from MuJoCo's implementation. Also keep them for compatibility with MuJoCo.
            # Increasing this value could be useful for detecting contact manifolds even when the normals are not
            # perfectly aligned, but we observed that it leads to more false positives and thus not a perfect solution
            # for the multi-contact detection.
            contact_face_tol=gs.np_float(0.99999872),
            contact_edge_tol=gs.np_float(0.00159999931),
        )

        # Initialize GJK state.
        self._gjk_state = array_class.get_gjk_state(
            rigid_solver, rigid_solver._static_rigid_sim_config, self._gjk_static_config
        )

    def reset(self):
        pass


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
    gjk_state.multi_contact_flag[i_b] = 0
    gjk_state.last_searched_simplex_vertex_id[i_b] = 0


@ti.func
def func_gjk_contact(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
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
                gjk_static_config,
                support_field_info,
                support_field_static_config,
                i_ga,
                i_gb,
                i_b,
                shrink_sphere,
            )

            if shrink_sphere:
                # If we shrinked the sphere and capsule to point and line and the distance is larger than the
                # collision epsilon, it means a shallow penetration. Thus we subtract the radius of the sphere and
                # the capsule to get the actual distance. If the distance is smaller than the collision epsilon, it
                # means a deep penetration, which requires the default GJK handling.
                if distance > gjk_static_config.collision_eps:
                    radius_a, radius_b = 0.0, 0.0
                    if is_sphere_swept_geom_a:
                        radius_a = geoms_info.data[i_ga][0]
                    if is_sphere_swept_geom_b:
                        radius_b = geoms_info.data[i_gb][0]

                    wa = gjk_state.witness.point_obj1[i_b, 0]
                    wb = gjk_state.witness.point_obj2[i_b, 0]
                    n = func_safe_normalize(gjk_static_config, wb - wa)

                    gjk_state.distance[i_b] = distance - (radius_a + radius_b)
                    gjk_state.witness.point_obj1[i_b, 0] = wa + (radius_a * n)
                    gjk_state.witness.point_obj2[i_b, 0] = wb - (radius_b * n)

                    break

            # Only try shrinking the sphere once
            shrink_sphere = False

            distance = gjk_state.distance[i_b]
            nsimplex = gjk_state.nsimplex[i_b]
            collided = distance < gjk_static_config.collision_eps

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
                        static_rigid_sim_config,
                        collider_state,
                        collider_static_config,
                        gjk_state,
                        gjk_static_config,
                        support_field_info,
                        support_field_static_config,
                        i_ga,
                        i_gb,
                        i_b,
                    )
                elif nsimplex == 4:
                    polytope_flag = func_epa_init_polytope_4d(gjk_state, gjk_static_config, i_ga, i_gb, i_b)

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
                        gjk_static_config,
                        support_field_info,
                        support_field_static_config,
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
                        gjk_static_config,
                        support_field_info,
                        support_field_static_config,
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
                                gjk_static_config,
                                i_ga,
                                i_gb,
                                i_b,
                                i_f,
                            )
                            gjk_state.multi_contact_flag[i_b] = 1
    else:
        gjk_flag = func_safe_gjk(
            geoms_state,
            geoms_info,
            verts_info,
            static_rigid_sim_config,
            collider_state,
            collider_static_config,
            gjk_state,
            gjk_static_config,
            support_field_info,
            support_field_static_config,
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
            func_safe_epa_init(gjk_state, gjk_static_config, i_ga, i_gb, i_b)

            # Run EPA from the polytope
            func_safe_epa(
                geoms_state,
                geoms_info,
                verts_info,
                static_rigid_sim_config,
                collider_state,
                collider_static_config,
                gjk_state,
                gjk_static_config,
                support_field_info,
                support_field_static_config,
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
            if normal_len < gjk_static_config.FLOAT_MIN:
                continue

            normal = normal / normal_len

            gjk_state.contact_pos[i_b, n_contacts] = contact_pos
            gjk_state.normal[i_b, n_contacts] = normal
            n_contacts += 1

    gjk_state.n_contacts[i_b] = n_contacts
    # If there are no contacts, we set the penetration and is_col to 0.
    # FIXME: When we use if statement here, it leads to a bug in some backends (e.g. x86 cpu). Need to investigate.
    gjk_state.is_col[i_b] = 0 if n_contacts == 0 else gjk_state.is_col[i_b]
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
    gjk_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
    shrink_sphere,
):
    """
    GJK algorithm to compute the minimum distance between two convex objects.

    This implementation is based on the MuJoCo implementation.

    TODO: This implementation could be further improved by referencing the follow-up work shown below.

    Parameters
    ----------
    shrink_sphere: bool
        If True, use point and line support functions for sphere and capsule geometries, respectively. It is more
        efficient and stable for shallow penetrations than the full GJK algorithm. However, if there is a deep
        penetration, we have to fallback to the full GJK algorithm by setting this parameter to False.

    .. seealso::
    MuJoCo's original implementation:
    https://github.com/google-deepmind/mujoco/blob/7dc7a349c5ba2db2d3f8ab50a367d08e2f1afbbc/src/engine/engine_collision_gjk.c#L171

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
    # Simplex index
    n = gs.ti_int(0)
    # Final number of simplex vertices
    nsimplex = gs.ti_int(0)
    # Number of witness points and distance
    nx = gs.ti_int(0)
    dist = gs.ti_float(0.0)
    # Lambda for barycentric coordinates
    _lambda = gs.ti_vec4(1.0, 0.0, 0.0, 0.0)
    # Whether or not we need to compute the exact distance.
    get_dist = shrink_sphere
    # We can use GJK intersection algorithm only for collision detection if we do not have to compute the distance.
    backup_gjk = not get_dist
    # Support vector to compute the next support point.
    support_vector = gs.ti_vec3(0.0, 0.0, 0.0)
    support_vector_norm = gs.ti_float(0.0)
    # Whether or not the main loop finished early because intersection or seperation was detected.
    early_stop = False

    # Set initial guess of support vector using the positions, which should be a non-zero vector.
    approx_witness_point_obj1 = geoms_state.pos[i_ga, i_b]
    approx_witness_point_obj2 = geoms_state.pos[i_gb, i_b]
    support_vector = approx_witness_point_obj1 - approx_witness_point_obj2
    if support_vector.dot(support_vector) < gjk_static_config.FLOAT_MIN_SQ:
        support_vector = gs.ti_vec3(1.0, 0.0, 0.0)

    # Epsilon for convergence check.
    epsilon = gs.ti_float(0.0)
    if not func_is_discrete_geoms(geoms_info, i_ga, i_gb, i_b):
        # If the objects are smooth, finite convergence is not guaranteed, so we need to set some epsilon
        # to determine convergence.
        epsilon = 0.5 * (gjk_static_config.tolerance**2)

    for i in range(gjk_static_config.gjk_max_iterations):
        # Compute the current support points
        support_vector_norm = support_vector.norm()
        if support_vector_norm < gjk_static_config.FLOAT_MIN:
            # If the support vector is too small, it means that origin is located in the Minkowski difference
            # with high probability, so we can stop.
            break

        # Dir to compute the support point (pointing from obj1 to obj2)
        dir = -support_vector * (1.0 / support_vector_norm)

        (
            gjk_state.simplex_vertex.obj1[i_b, n],
            gjk_state.simplex_vertex.obj2[i_b, n],
            gjk_state.simplex_vertex.id1[i_b, n],
            gjk_state.simplex_vertex.id2[i_b, n],
            gjk_state.simplex_vertex.mink[i_b, n],
        ) = func_support(
            geoms_state,
            geoms_info,
            verts_info,
            static_rigid_sim_config,
            collider_state,
            collider_static_config,
            gjk_state,
            gjk_static_config,
            support_field_info,
            support_field_static_config,
            i_ga,
            i_gb,
            i_b,
            dir,
            shrink_sphere,
        )

        # Early stopping based on Frank-Wolfe duality gap. We need to find the minimum [support_vector_norm],
        # and if we denote it as [x], the problem formulation is: min_x |x|^2.
        # If we denote f(x) = |x|^2, then the Frank-Wolfe duality gap is:
        # |x - x_min|^2 <= < grad f(x), x - s> = < 2x, x - s >,
        # where s is the vertex of the Minkowski difference found by x. Here < 2x, x - s > is guaranteed to be
        # non-negative, and 2 is cancelled out in the definition of the epsilon.
        x_k = support_vector
        s_k = gjk_state.simplex_vertex.mink[i_b, n]
        diff = x_k - s_k
        if diff.dot(x_k) < epsilon:
            # Convergence condition is met, we can stop.
            if i == 0:
                n = 1
            break

        # Check if the objects are separated using support vector
        if not get_dist:
            is_separated = x_k.dot(s_k) > 0.0
            if is_separated:
                nsimplex = 0
                nx = 0
                dist = gjk_static_config.FLOAT_MAX
                early_stop = True
                break

        if n == 3 and backup_gjk:
            # Tetrahedron is generated, try to detect collision if possible.
            intersect_code = func_gjk_intersect(
                geoms_state,
                geoms_info,
                verts_info,
                static_rigid_sim_config,
                collider_state,
                collider_static_config,
                gjk_state,
                gjk_static_config,
                support_field_info,
                support_field_static_config,
                i_ga,
                i_gb,
                i_b,
            )
            if intersect_code == GJK_RETURN_CODE.SEPARATED:
                # No intersection, objects are separated
                nx = 0
                dist = gjk_static_config.FLOAT_MAX
                nsimplex = 0
                early_stop = True
                break
            elif intersect_code == GJK_RETURN_CODE.INTERSECT:
                # Intersection found
                nx = 0
                dist = 0.0
                nsimplex = 4
                early_stop = True
                break
            else:
                # Since gjk_intersect failed (e.g. origin is on the simplex face), fallback to distance computation
                backup_gjk = False

        # Compute the barycentric coordinates of the closest point to the origin in the simplex
        _lambda = func_gjk_subdistance(gjk_state, gjk_static_config, i_b, n + 1)

        # Remove vertices from the simplex with zero barycentric coordinates
        n = 0
        for j in ti.static(range(4)):
            if _lambda[j] > 0:
                gjk_state.simplex_vertex.obj1[i_b, n] = gjk_state.simplex_vertex.obj1[i_b, j]
                gjk_state.simplex_vertex.obj2[i_b, n] = gjk_state.simplex_vertex.obj2[i_b, j]
                gjk_state.simplex_vertex.id1[i_b, n] = gjk_state.simplex_vertex.id1[i_b, j]
                gjk_state.simplex_vertex.id2[i_b, n] = gjk_state.simplex_vertex.id2[i_b, j]
                gjk_state.simplex_vertex.mink[i_b, n] = gjk_state.simplex_vertex.mink[i_b, j]
                _lambda[n] = _lambda[j]
                n += 1

        # Should not occur
        if n < 1:
            nsimplex = 0
            nx = 0
            dist = gjk_static_config.FLOAT_MAX
            early_stop = True
            break

        # Get the next support vector
        next_support_vector = func_simplex_vertex_linear_comb(gjk_state, i_b, 2, 0, 1, 2, 3, _lambda, n)
        if func_is_equal_vec(next_support_vector, support_vector, gjk_static_config.FLOAT_MIN):
            # If the next support vector is equal to the previous one, we converged to the minimum distance
            break

        support_vector = next_support_vector

        if n == 4:
            # We have a tetrahedron containing the origin, so we can return early. This is because only when
            # the origin is inside the tetrahedron, the barycentric coordinates are all positive. While MuJoCo
            # does not set the [support_vector_norm] to zero as we do, it is necessary, because otherwise the
            # [support_vector_norm] could be non-zero value even if there is contact.
            support_vector_norm = 0
            break

    if not early_stop:
        # If [get_dist] was True and there was no numerical error, [return_code] would be SUCCESS.
        nx = 1
        nsimplex = n
        dist = support_vector_norm

        # Compute witness points
        for i in range(2):
            witness_point = func_simplex_vertex_linear_comb(gjk_state, i_b, i, 0, 1, 2, 3, _lambda, nsimplex)
            if i == 0:
                gjk_state.witness.point_obj1[i_b, 0] = witness_point
            else:
                gjk_state.witness.point_obj2[i_b, 0] = witness_point

    gjk_state.n_witness[i_b] = nx
    gjk_state.distance[i_b] = dist
    gjk_state.nsimplex[i_b] = nsimplex

    return gjk_state.distance[i_b]


@ti.func
def func_gjk_intersect(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: ti.template(),
    gjk_static_config: ti.template(),
    support_field_info: ti.template(),
    support_field_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
):
    """
    Check if the two objects intersect using the GJK algorithm.

    This function refines the simplex until it contains the origin or it is determined that the objects are
    separated. It is used to check if the objects intersect, not to find the minimum distance between them.
    """
    # Copy simplex to temporary storage
    for i in ti.static(range(4)):
        gjk_state.simplex_vertex_intersect.obj1[i_b, i] = gjk_state.simplex_vertex.obj1[i_b, i]
        gjk_state.simplex_vertex_intersect.obj2[i_b, i] = gjk_state.simplex_vertex.obj2[i_b, i]
        gjk_state.simplex_vertex_intersect.id1[i_b, i] = gjk_state.simplex_vertex.id1[i_b, i]
        gjk_state.simplex_vertex_intersect.id2[i_b, i] = gjk_state.simplex_vertex.id2[i_b, i]
        gjk_state.simplex_vertex_intersect.mink[i_b, i] = gjk_state.simplex_vertex.mink[i_b, i]

    # Simplex index
    si = ti.Vector([0, 1, 2, 3], dt=gs.ti_int)

    flag = GJK_RETURN_CODE.NUM_ERROR
    for i in range(gjk_static_config.gjk_max_iterations):
        # Compute normal and signed distance of the triangle faces of the simplex with respect to the origin.
        # These normals are supposed to point outwards from the simplex.
        # If the origin is inside the plane, [sdist] will be positive.
        is_sdist_all_zero = True
        for j in range(4):
            s0, s1, s2 = si[2], si[1], si[3]
            if j == 1:
                s0, s1, s2 = si[0], si[2], si[3]
            elif j == 2:
                s0, s1, s2 = si[1], si[0], si[3]
            elif j == 3:
                s0, s1, s2 = si[0], si[1], si[2]

            n, s = func_gjk_triangle_info(gjk_state, gjk_static_config, i_b, s0, s1, s2)

            gjk_state.simplex_buffer_intersect.normal[i_b, j] = n
            gjk_state.simplex_buffer_intersect.sdist[i_b, j] = s

            if ti.abs(s) > gjk_static_config.FLOAT_MIN:
                is_sdist_all_zero = False

        # If the origin is strictly on any affine hull of the faces, convergence will fail, so ignore this case
        if is_sdist_all_zero:
            break

        # Find the face with the smallest signed distance. We need to find [min_i] for the next iteration.
        min_i = 0
        for j in ti.static(range(1, 4)):
            if gjk_state.simplex_buffer_intersect.sdist[i_b, j] < gjk_state.simplex_buffer_intersect.sdist[i_b, min_i]:
                min_i = j

        min_si = si[min_i]
        min_normal = gjk_state.simplex_buffer_intersect.normal[i_b, min_i]
        min_sdist = gjk_state.simplex_buffer_intersect.sdist[i_b, min_i]

        # If origin is inside the simplex, the signed distances will all be positive
        if min_sdist >= 0:
            # Origin is inside the simplex, so we can stop
            flag = GJK_RETURN_CODE.INTERSECT

            # Copy the temporary simplex to the main simplex
            for j in ti.static(range(4)):
                gjk_state.simplex_vertex.obj1[i_b, j] = gjk_state.simplex_vertex_intersect.obj1[i_b, si[j]]
                gjk_state.simplex_vertex.obj2[i_b, j] = gjk_state.simplex_vertex_intersect.obj2[i_b, si[j]]
                gjk_state.simplex_vertex.id1[i_b, j] = gjk_state.simplex_vertex_intersect.id1[i_b, si[j]]
                gjk_state.simplex_vertex.id2[i_b, j] = gjk_state.simplex_vertex_intersect.id2[i_b, si[j]]
                gjk_state.simplex_vertex.mink[i_b, j] = gjk_state.simplex_vertex_intersect.mink[i_b, si[j]]
            break

        # Replace the worst vertex (which has the smallest signed distance) with new candidate
        (
            gjk_state.simplex_vertex_intersect.obj1[i_b, min_si],
            gjk_state.simplex_vertex_intersect.obj2[i_b, min_si],
            gjk_state.simplex_vertex_intersect.id1[i_b, min_si],
            gjk_state.simplex_vertex_intersect.id2[i_b, min_si],
            gjk_state.simplex_vertex_intersect.mink[i_b, min_si],
        ) = func_support(
            geoms_state,
            geoms_info,
            verts_info,
            static_rigid_sim_config,
            collider_state,
            collider_static_config,
            gjk_state,
            gjk_static_config,
            support_field_info,
            support_field_static_config,
            i_ga,
            i_gb,
            i_b,
            min_normal,
            False,
        )

        # Check if the origin is strictly outside of the Minkowski difference (which means there is no collision)
        new_minkowski = gjk_state.simplex_vertex_intersect.mink[i_b, min_si]

        is_no_collision = new_minkowski.dot(min_normal) < 0
        if is_no_collision:
            flag = GJK_RETURN_CODE.SEPARATED
            break

        # Swap vertices in the simplex to retain orientation
        m = (min_i + 1) % 4
        n = (min_i + 2) % 4
        swap = si[m]
        si[m] = si[n]
        si[n] = swap

    return flag


@ti.func
def func_gjk_triangle_info(
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
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
    if (normal_length > gjk_static_config.FLOAT_MIN) and (normal_length < gjk_static_config.FLOAT_MAX):
        normal = normal * (1.0 / normal_length)
        sdist = normal.dot(vertex_1)
    else:
        # If the normal length is unstable, return max distance.
        sdist = gjk_static_config.FLOAT_MAX

    return normal, sdist


@ti.func
def func_gjk_subdistance(
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
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

    dmin = gjk_static_config.FLOAT_MAX

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
            _lambda2d, flag2d = func_gjk_subdistance_2d(gjk_state, gjk_static_config, i_b, k_1, k_2, k_3)

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
    gjk_static_config: ti.template(),
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
        gjk_static_config,
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
def func_epa(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
):
    """
    EPA algorithm to find the exact penetration depth and contact normal using the simplex constructed by GJK.

    .. seealso::
    MuJoCo's original implementation:
    https://github.com/google-deepmind/mujoco/blob/7dc7a349c5ba2db2d3f8ab50a367d08e2f1afbbc/src/engine/engine_collision_gjk.c#L1331
    """
    upper = gjk_static_config.FLOAT_MAX
    upper2 = gjk_static_config.FLOAT_MAX_SQ
    lower = 0.0
    tolerance = gjk_static_config.tolerance

    # Index of the nearest face
    nearest_i_f = -1
    prev_nearest_i_f = -1

    discrete = func_is_discrete_geoms(geoms_info, i_ga, i_gb, i_b)
    if discrete:
        # If the objects are discrete, we do not use tolerance.
        tolerance = gjk_static_config.FLOAT_MIN

    k_max = gjk_static_config.epa_max_iterations
    for k in range(k_max):
        prev_nearest_i_f = nearest_i_f

        # Find the polytope face with the smallest distance to the origin
        lower2 = gjk_static_config.FLOAT_MAX_SQ

        for i in range(gjk_state.polytope.nfaces_map[i_b]):
            i_f = gjk_state.polytope_faces_map[i_b, i]
            face_dist2 = gjk_state.polytope_faces.dist2[i_b, i_f]

            if face_dist2 < lower2:
                lower2 = face_dist2
                nearest_i_f = i_f

        if lower2 > upper2 or nearest_i_f < 0:
            # Invalid face found, stop the algorithm (lower bound of depth is larger than upper bound)
            nearest_i_f = prev_nearest_i_f
            break

        if lower2 <= gjk_static_config.FLOAT_MIN_SQ:
            # Invalid lower bound (0), stop the algorithm (origin is on the affine hull of face)
            break

        # Find a new support point w from the nearest face's normal
        lower = ti.sqrt(lower2)
        dir = gjk_state.polytope_faces.normal[i_b, nearest_i_f]
        wi = func_epa_support(
            geoms_state,
            geoms_info,
            verts_info,
            static_rigid_sim_config,
            collider_state,
            collider_static_config,
            gjk_state,
            gjk_static_config,
            support_field_info,
            support_field_static_config,
            i_ga,
            i_gb,
            i_b,
            dir,
            lower,
        )
        w = gjk_state.polytope_verts.mink[i_b, wi]

        # The upper bound of depth at k-th iteration
        upper_k = w.dot(dir) / lower
        if upper_k < upper:
            upper = upper_k
            upper2 = upper**2

        # If the upper bound and lower bound are close enough, we can stop the algorithm
        if (upper - lower) < tolerance:
            break

        if discrete:
            repeated = False
            for i in range(gjk_state.polytope.nverts[i_b] - 1):
                if (
                    gjk_state.polytope_verts.id1[i_b, i] == gjk_state.polytope_verts.id1[i_b, wi]
                    and gjk_state.polytope_verts.id2[i_b, i] == gjk_state.polytope_verts.id2[i_b, wi]
                ):
                    # The vertex w is already in the polytope,
                    # so we do not need to add it again.
                    repeated = True
                    break
            if repeated:
                break

        gjk_state.polytope.horizon_w[i_b] = w

        # Compute horizon
        horizon_flag = func_epa_horizon(gjk_state, gjk_static_config, i_b, nearest_i_f)

        if horizon_flag:
            # There was an error in the horizon construction, so the horizon edge is not a closed loop.
            nearest_i_f = -1
            break

        if gjk_state.polytope.horizon_nedges[i_b] < 3:
            # Should not happen, because at least three edges should be in the horizon from one deleted face.
            nearest_i_f = -1
            break

        # Check if the memory space is enough for attaching new faces
        nfaces = gjk_state.polytope.nfaces[i_b]
        nedges = gjk_state.polytope.horizon_nedges[i_b]
        if nfaces + nedges >= gjk_static_config.polytope_max_faces:
            # If the polytope is full, we cannot insert new faces
            break

        # Attach the new faces
        for i in range(nedges):
            # Face id of the current face to attach
            i_f0 = nfaces + i
            # Face id of the next face to attach
            i_f1 = nfaces + (i + 1) % nedges

            horizon_i_f = gjk_state.polytope_horizon_data.face_idx[i_b, i]
            horizon_i_e = gjk_state.polytope_horizon_data.edge_idx[i_b, i]

            horizon_v1 = gjk_state.polytope_faces.verts_idx[i_b, horizon_i_f][horizon_i_e]
            horizon_v2 = gjk_state.polytope_faces.verts_idx[i_b, horizon_i_f][(horizon_i_e + 1) % 3]

            # Change the adjacent face index of the existing face
            gjk_state.polytope_faces.adj_idx[i_b, horizon_i_f][horizon_i_e] = i_f0

            # Attach the new face.
            # If this if the first face, will be adjacent to the face that will be attached last.
            adj_i_f_0 = i_f0 - 1 if (i > 0) else nfaces + nedges - 1
            adj_i_f_1 = horizon_i_f
            adj_i_f_2 = i_f1

            dist2 = func_attach_face_to_polytope(
                gjk_state,
                gjk_static_config,
                i_b,
                wi,
                horizon_v2,
                horizon_v1,
                adj_i_f_2,  # Previous face id
                adj_i_f_1,
                adj_i_f_0,  # Next face id
            )
            if dist2 <= 0:
                # Unrecoverable numerical issue
                nearest_i_f = -1
                break

            if (dist2 >= lower2) and (dist2 <= upper2):
                # Store face in the map
                nfaces_map = gjk_state.polytope.nfaces_map[i_b]
                gjk_state.polytope_faces_map[i_b, nfaces_map] = i_f0
                gjk_state.polytope_faces.map_idx[i_b, i_f0] = nfaces_map
                gjk_state.polytope.nfaces_map[i_b] += 1

        # Clear the horizon data for the next iteration
        gjk_state.polytope.horizon_nedges[i_b] = 0

        if (gjk_state.polytope.nfaces_map[i_b] == 0) or (nearest_i_f == -1):
            # No face candidate left
            break

    if nearest_i_f != -1:
        # Nearest face found
        dist2 = gjk_state.polytope_faces.dist2[i_b, nearest_i_f]
        func_epa_witness(gjk_state, i_ga, i_gb, i_b, nearest_i_f)
        gjk_state.n_witness[i_b] = 1
        gjk_state.distance[i_b] = -ti.sqrt(dist2)
    else:
        # No face found, so the objects are not colliding
        gjk_state.n_witness[i_b] = 0
        gjk_state.distance[i_b] = 0

    return nearest_i_f


@ti.func
def func_epa_witness(
    gjk_state: array_class.GJKState,
    i_ga,
    i_gb,
    i_b,
    i_f,
):
    """
    Compute the witness points from the geometries for the face i_f of the polytope.
    """
    # Find the affine coordinates of the origin's projection on the face i_f
    face_iv1 = gjk_state.polytope_faces.verts_idx[i_b, i_f][0]
    face_iv2 = gjk_state.polytope_faces.verts_idx[i_b, i_f][1]
    face_iv3 = gjk_state.polytope_faces.verts_idx[i_b, i_f][2]
    face_v1 = gjk_state.polytope_verts.mink[i_b, face_iv1]
    face_v2 = gjk_state.polytope_verts.mink[i_b, face_iv2]
    face_v3 = gjk_state.polytope_verts.mink[i_b, face_iv3]
    face_normal = gjk_state.polytope_faces.normal[i_b, i_f]

    _lambda = func_triangle_affine_coords(
        face_normal,
        face_v1,
        face_v2,
        face_v3,
    )

    # Point on geom 1
    v1 = gjk_state.polytope_verts.obj1[i_b, face_iv1]
    v2 = gjk_state.polytope_verts.obj1[i_b, face_iv2]
    v3 = gjk_state.polytope_verts.obj1[i_b, face_iv3]
    witness1 = v1 * _lambda[0] + v2 * _lambda[1] + v3 * _lambda[2]

    # Point on geom 2
    v1 = gjk_state.polytope_verts.obj2[i_b, face_iv1]
    v2 = gjk_state.polytope_verts.obj2[i_b, face_iv2]
    v3 = gjk_state.polytope_verts.obj2[i_b, face_iv3]
    witness2 = v1 * _lambda[0] + v2 * _lambda[1] + v3 * _lambda[2]

    gjk_state.witness.point_obj1[i_b, 0] = witness1
    gjk_state.witness.point_obj2[i_b, 0] = witness2


@ti.func
def func_epa_horizon(
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    i_b,
    nearest_i_f,
):
    """
    Compute the horizon, which represents the area of the polytope that is visible from the vertex w, and thus
    should be deleted for the expansion of the polytope.
    """
    w = gjk_state.polytope.horizon_w[i_b]

    # Initialize the stack by inserting the nearest face
    gjk_state.polytope_horizon_stack.face_idx[i_b, 0] = nearest_i_f
    gjk_state.polytope_horizon_stack.edge_idx[i_b, 0] = 0
    top = 1
    is_first = True

    flag = RETURN_CODE.SUCCESS
    while top > 0:
        # Pop the top face from the stack
        i_f = gjk_state.polytope_horizon_stack.face_idx[i_b, top - 1]
        i_e = gjk_state.polytope_horizon_stack.edge_idx[i_b, top - 1]
        i_v = gjk_state.polytope_faces.verts_idx[i_b, i_f][0]
        v = gjk_state.polytope_verts.mink[i_b, i_v]
        top -= 1

        # If the face is already deleted, skip it
        is_deleted = gjk_state.polytope_faces.map_idx[i_b, i_f] == -2
        if (not is_first) and (is_deleted):
            continue

        # Check visibility of the face. Two requirements for the face to be visible:
        # 1. The face normal should point towards the vertex w
        # 2. The vertex w should be on the other side of the face to the origin
        is_visible = gjk_state.polytope_faces.normal[i_b, i_f].dot(w - v) > gjk_static_config.FLOAT_MIN

        # The first face is always considered visible.
        if is_visible or is_first:
            # If visible, delete the face from the polytope
            func_delete_face_from_polytope(gjk_state, i_b, i_f)

            # Add the other two or three edges of the face to the stack.
            # The order is important to form a closed loop.
            for k in range(0 if is_first else 1, 3):
                i_e2 = (i_e + k) % 3
                adj_face_idx = gjk_state.polytope_faces.adj_idx[i_b, i_f][i_e2]
                adj_face_is_deleted = gjk_state.polytope_faces.map_idx[i_b, adj_face_idx] == -2
                if not adj_face_is_deleted:
                    # Get the related edge id from the adjacent face. Since adjacent faces have different
                    # orientations, we need to use the ending vertex of the edge.
                    start_vert_idx = gjk_state.polytope_faces.verts_idx[i_b, i_f][(i_e2 + 1) % 3]
                    adj_edge_idx = func_get_edge_idx(gjk_state, i_b, adj_face_idx, start_vert_idx)

                    gjk_state.polytope_horizon_stack.face_idx[i_b, top] = adj_face_idx
                    gjk_state.polytope_horizon_stack.edge_idx[i_b, top] = adj_edge_idx
                    top += 1
        else:
            # If not visible, add the edge to the horizon.
            flag = func_add_edge_to_horizon(gjk_state, i_b, i_f, i_e)
            if flag:
                # If the edges do not form a closed loop, there is an error in the algorithm.
                break

        is_first = False

    return flag


@ti.func
def func_add_edge_to_horizon(
    gjk_state: array_class.GJKState,
    i_b,
    i_f,
    i_e,
):
    """
    Add an edge to the horizon data structure.
    """
    horizon_nedges = gjk_state.polytope.horizon_nedges[i_b]
    gjk_state.polytope_horizon_data.edge_idx[i_b, horizon_nedges] = i_e
    gjk_state.polytope_horizon_data.face_idx[i_b, horizon_nedges] = i_f
    gjk_state.polytope.horizon_nedges[i_b] += 1

    return RETURN_CODE.SUCCESS


@ti.func
def func_get_edge_idx(
    gjk_state: array_class.GJKState,
    i_b,
    i_f,
    i_v,
):
    """
    Get the edge index from the face, starting from the vertex i_v.

    If the face is comprised of [v1, v2, v3], the edges are: [v1, v2], [v2, v3], [v3, v1].
    Therefore, if i_v was v1, the edge index is 0, and if i_v was v2, the edge index is 1.
    """
    verts = gjk_state.polytope_faces.verts_idx[i_b, i_f]
    ret = gs.ti_int(2)
    if verts[0] == i_v:
        ret = 0
    elif verts[1] == i_v:
        ret = 1
    return ret


@ti.func
def func_delete_face_from_polytope(
    gjk_state: array_class.GJKState,
    i_b,
    i_f,
):
    """
    Delete the face from the polytope.
    """
    face_map_idx = gjk_state.polytope_faces.map_idx[i_b, i_f]
    if face_map_idx >= 0:
        last_face_idx = gjk_state.polytope_faces_map[i_b, gjk_state.polytope.nfaces_map[i_b] - 1]
        # Make the map to point to the last face
        gjk_state.polytope_faces_map[i_b, face_map_idx] = last_face_idx
        # Change map index of the last face
        gjk_state.polytope_faces.map_idx[i_b, last_face_idx] = face_map_idx

        # Decrease the number of faces in the polytope
        gjk_state.polytope.nfaces_map[i_b] -= 1

    # Mark the face as deleted
    gjk_state.polytope_faces.map_idx[i_b, i_f] = -2


@ti.func
def func_epa_insert_vertex_to_polytope(
    gjk_state: array_class.GJKState,
    i_b,
    obj1_point,
    obj2_point,
    obj1_id,
    obj2_id,
    minkowski_point,
):
    """
    Copy vertex information into the polytope.
    """
    n = gjk_state.polytope.nverts[i_b]
    gjk_state.polytope_verts.obj1[i_b, n] = obj1_point
    gjk_state.polytope_verts.obj2[i_b, n] = obj2_point
    gjk_state.polytope_verts.id1[i_b, n] = obj1_id
    gjk_state.polytope_verts.id2[i_b, n] = obj2_id
    gjk_state.polytope_verts.mink[i_b, n] = minkowski_point
    gjk_state.polytope.nverts[i_b] += 1
    return n


@ti.func
def func_epa_init_polytope_2d(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
):
    """
    Create the polytope for EPA from a 1-simplex (line segment).

    Returns
    -------
    int
        0 when successful, or a flag indicating an error.
    """
    flag = EPA_POLY_INIT_RETURN_CODE.SUCCESS

    # Get the simplex vertices
    v1 = gjk_state.simplex_vertex.mink[i_b, 0]
    v2 = gjk_state.simplex_vertex.mink[i_b, 1]
    diff = v2 - v1

    # Find the element in [diff] with the smallest magnitude, because it will give us the largest cross product
    min_val = ti.abs(diff[0])
    min_i = 0
    for i in ti.static(range(1, 3)):
        abs_diff_i = ti.abs(diff[i])
        if abs_diff_i < min_val:
            min_val = abs_diff_i
            min_i = i

    # Cross product with the found axis, then rotate it by 120 degrees around the axis [diff] to get three more
    # points spaced 120 degrees apart
    rotmat = gu.ti_rotvec_to_R(diff * ti.math.radians(120.0))
    e = gs.ti_vec3(0.0, 0.0, 0.0)
    e[min_i] = 1.0

    d1 = e.cross(diff)
    d2 = rotmat @ d1
    d3 = rotmat @ d2

    # Insert the first two vertices into the polytope
    vi = ti.Vector([0, 0, 0, 0, 0], dt=ti.i32)
    for i in range(2):
        vi[i] = func_epa_insert_vertex_to_polytope(
            gjk_state,
            i_b,
            gjk_state.simplex_vertex.obj1[i_b, i],
            gjk_state.simplex_vertex.obj2[i_b, i],
            gjk_state.simplex_vertex.id1[i_b, i],
            gjk_state.simplex_vertex.id2[i_b, i],
            gjk_state.simplex_vertex.mink[i_b, i],
        )

    # Find three more vertices using [d1, d2, d3] as support vectors, and insert them into the polytope
    for i in range(3):
        di = d1
        if i == 1:
            di = d2
        elif i == 2:
            di = d3
        di_norm = di.norm()
        vi[i + 2] = func_epa_support(
            geoms_state,
            geoms_info,
            verts_info,
            static_rigid_sim_config,
            collider_state,
            collider_static_config,
            gjk_state,
            gjk_static_config,
            support_field_info,
            support_field_static_config,
            i_ga,
            i_gb,
            i_b,
            di,
            di_norm,
        )

    v3 = gjk_state.polytope_verts.mink[i_b, vi[2]]
    v4 = gjk_state.polytope_verts.mink[i_b, vi[3]]
    v5 = gjk_state.polytope_verts.mink[i_b, vi[4]]

    # Build hexahedron (6 faces) from the five vertices.
    # * This hexahedron would have line [v1, v2] as the central axis, and the other three vertices would be on the
    # sides of the hexahedron, as they are spaced 120 degrees apart.
    # * We already know the face and adjacent face indices in building this.
    # * While building the hexahedron by attaching faces, if the face is very close to the origin, we replace the
    # 1-simplex with the 2-simplex, and restart from it.
    for i in range(6):
        # Vertex indices for the faces in the hexahedron
        i_v1, i_v2, i_v3 = vi[0], vi[2], vi[3]
        # Adjacent face indices for the faces in the hexahedron
        i_a1, i_a2, i_a3 = 1, 3, 2
        if i == 1:
            i_v1, i_v2, i_v3 = vi[0], vi[4], vi[2]
            i_a1, i_a2, i_a3 = 2, 4, 0
        elif i == 2:
            i_v1, i_v2, i_v3 = vi[0], vi[3], vi[4]
            i_a1, i_a2, i_a3 = 0, 5, 1
        elif i == 3:
            i_v1, i_v2, i_v3 = vi[1], vi[3], vi[2]
            i_a1, i_a2, i_a3 = 5, 0, 4
        elif i == 4:
            i_v1, i_v2, i_v3 = vi[1], vi[2], vi[4]
            i_a1, i_a2, i_a3 = 3, 1, 5
        elif i == 5:
            i_v1, i_v2, i_v3 = vi[1], vi[4], vi[3]
            i_a1, i_a2, i_a3 = 4, 2, 3

        if (
            func_attach_face_to_polytope(gjk_state, gjk_static_config, i_b, i_v1, i_v2, i_v3, i_a1, i_a2, i_a3)
            < gjk_static_config.FLOAT_MIN_SQ
        ):
            func_replace_simplex_3(gjk_state, i_b, i_v1, i_v2, i_v3)
            flag = EPA_POLY_INIT_RETURN_CODE.P2_FALLBACK3
            break

    if flag == RETURN_CODE.SUCCESS:
        if not func_ray_triangle_intersection(v1, v2, v3, v4, v5):
            # The hexahedron should be convex by definition, but somehow if it is not, we return non-convex flag
            flag = EPA_POLY_INIT_RETURN_CODE.P2_NONCONVEX

    if flag == RETURN_CODE.SUCCESS:
        # Initialize face map
        for i in ti.static(range(6)):
            gjk_state.polytope_faces_map[i_b, i] = i
            gjk_state.polytope_faces.map_idx[i_b, i] = i
        gjk_state.polytope.nfaces_map[i_b] = 6

    return flag


@ti.func
def func_epa_init_polytope_3d(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
):
    """
    Create the polytope for EPA from a 2-simplex (triangle).

    Returns
    -------
    int
        0 when successful, or a flag indicating an error.
    """
    flag = EPA_POLY_INIT_RETURN_CODE.SUCCESS

    # Get the simplex vertices
    v1 = gjk_state.simplex_vertex.mink[i_b, 0]
    v2 = gjk_state.simplex_vertex.mink[i_b, 1]
    v3 = gjk_state.simplex_vertex.mink[i_b, 2]

    # Get normal; if it is zero, we cannot proceed
    n = (v2 - v1).cross(v3 - v1)
    n_norm = n.norm()
    if n_norm < gjk_static_config.FLOAT_MIN:
        flag = EPA_POLY_INIT_RETURN_CODE.P3_BAD_NORMAL
    n_neg = -n

    # Save vertices in the polytope
    vi = ti.Vector([0, 0, 0, 0, 0], dt=ti.i32)
    for i in range(3):
        vi[i] = func_epa_insert_vertex_to_polytope(
            gjk_state,
            i_b,
            gjk_state.simplex_vertex.obj1[i_b, i],
            gjk_state.simplex_vertex.obj2[i_b, i],
            gjk_state.simplex_vertex.id1[i_b, i],
            gjk_state.simplex_vertex.id2[i_b, i],
            gjk_state.simplex_vertex.mink[i_b, i],
        )

    # Find the fourth and fifth vertices using the normal
    # as the support vector. We form a hexahedron (6 faces)
    # with these five vertices.
    for i in range(2):
        dir = n if i == 0 else n_neg
        vi[i + 3] = func_epa_support(
            geoms_state,
            geoms_info,
            verts_info,
            static_rigid_sim_config,
            collider_state,
            collider_static_config,
            gjk_state,
            gjk_static_config,
            support_field_info,
            support_field_static_config,
            i_ga,
            i_gb,
            i_b,
            dir,
            n_norm,
        )
    v4 = gjk_state.polytope_verts.mink[i_b, vi[3]]
    v5 = gjk_state.polytope_verts.mink[i_b, vi[4]]

    # Check if v4 or v5 located inside the triangle.
    # If so, we do not proceed anymore.
    for i in range(2):
        v = v4 if i == 0 else v5
        if func_point_triangle_intersection(gjk_static_config, v, v1, v2, v3):
            flag = EPA_POLY_INIT_RETURN_CODE.P3_INVALID_V4 if i == 0 else EPA_POLY_INIT_RETURN_CODE.P3_INVALID_V5
            break

    if flag == EPA_POLY_INIT_RETURN_CODE.SUCCESS:
        # If origin does not lie inside the triangle, we need to
        # check if the hexahedron contains the origin.

        tets_has_origin = gs.ti_ivec2(0, 0)
        for i in range(2):
            v = v4 if i == 0 else v5
            tets_has_origin[i] = 1 if func_origin_tetra_intersection(v1, v2, v3, v) == RETURN_CODE.SUCCESS else 0

        # @TODO: It's possible for GJK to return a triangle with origin not contained in it but within tolerance
        # from it. In that case, the hexahedron could possibly be constructed that does ont contain the origin, but
        # there is penetration depth.
        if (
            gjk_state.simplex.dist[i_b] > 10 * gjk_static_config.FLOAT_MIN
            and (not tets_has_origin[0])
            and (not tets_has_origin[1])
        ):
            flag = EPA_POLY_INIT_RETURN_CODE.P3_MISSING_ORIGIN
        else:
            # Build hexahedron (6 faces) from the five vertices.
            for i in range(6):
                # Vertex indices for the faces in the hexahedron
                i_v1, i_v2, i_v3 = vi[3], vi[0], vi[1]
                # Adjacent face indices for the faces in the hexahedron
                i_a1, i_a2, i_a3 = 1, 3, 2
                if i == 1:
                    i_v1, i_v2, i_v3 = vi[3], vi[2], vi[0]
                    i_a1, i_a2, i_a3 = 2, 4, 0
                elif i == 2:
                    i_v1, i_v2, i_v3 = vi[3], vi[1], vi[2]
                    i_a1, i_a2, i_a3 = 0, 5, 1
                elif i == 3:
                    i_v1, i_v2, i_v3 = vi[4], vi[1], vi[0]
                    i_a1, i_a2, i_a3 = 5, 0, 4
                elif i == 4:
                    i_v1, i_v2, i_v3 = vi[4], vi[0], vi[2]
                    i_a1, i_a2, i_a3 = 3, 1, 5
                elif i == 5:
                    i_v1, i_v2, i_v3 = vi[4], vi[2], vi[1]
                    i_a1, i_a2, i_a3 = 4, 2, 3

                dist2 = func_attach_face_to_polytope(
                    gjk_state, gjk_static_config, i_b, i_v1, i_v2, i_v3, i_a1, i_a2, i_a3
                )
                if dist2 < gjk_static_config.FLOAT_MIN_SQ:
                    flag = EPA_POLY_INIT_RETURN_CODE.P3_ORIGIN_ON_FACE
                    break

    if flag == EPA_POLY_INIT_RETURN_CODE.SUCCESS:
        # Initialize face map
        for i in ti.static(range(6)):
            gjk_state.polytope_faces_map[i_b, i] = i
            gjk_state.polytope_faces.map_idx[i_b, i] = i
        gjk_state.polytope.nfaces_map[i_b] = 6

    return flag


@ti.func
def func_epa_init_polytope_4d(
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
):
    """
    Create the polytope for EPA from a 3-simplex (tetrahedron).

    Returns
    -------
    int
        0 when successful, or a flag indicating an error.
    """
    flag = EPA_POLY_INIT_RETURN_CODE.SUCCESS

    # Insert simplex vertices into the polytope
    vi = ti.Vector([0, 0, 0, 0], dt=ti.i32)
    for i in range(4):
        vi[i] = func_epa_insert_vertex_to_polytope(
            gjk_state,
            i_b,
            gjk_state.simplex_vertex.obj1[i_b, i],
            gjk_state.simplex_vertex.obj2[i_b, i],
            gjk_state.simplex_vertex.id1[i_b, i],
            gjk_state.simplex_vertex.id2[i_b, i],
            gjk_state.simplex_vertex.mink[i_b, i],
        )

    # If origin is on any face of the tetrahedron, replace the simplex with a 2-simplex (triangle)
    for i in range(4):
        # Vertex indices for the faces in the hexahedron
        v1, v2, v3 = vi[0], vi[1], vi[2]
        # Adjacent face indices for the faces in the hexahedron
        a1, a2, a3 = 1, 3, 2
        if i == 1:
            v1, v2, v3 = vi[0], vi[3], vi[1]
            a1, a2, a3 = 2, 3, 0
        elif i == 2:
            v1, v2, v3 = vi[0], vi[2], vi[3]
            a1, a2, a3 = 0, 3, 1
        elif i == 3:
            v1, v2, v3 = vi[3], vi[2], vi[1]
            a1, a2, a3 = 2, 0, 1

        dist2 = func_attach_face_to_polytope(gjk_state, gjk_static_config, i_b, v1, v2, v3, a1, a2, a3)

        if dist2 < gjk_static_config.FLOAT_MIN_SQ:
            func_replace_simplex_3(gjk_state, i_b, v1, v2, v3)
            flag = EPA_POLY_INIT_RETURN_CODE.P4_FALLBACK3
            break

    if flag == EPA_POLY_INIT_RETURN_CODE.SUCCESS:
        # If the tetrahedron does not contain the origin, we do not proceed anymore.
        if (
            func_origin_tetra_intersection(
                gjk_state.polytope_verts.mink[i_b, vi[0]],
                gjk_state.polytope_verts.mink[i_b, vi[1]],
                gjk_state.polytope_verts.mink[i_b, vi[2]],
                gjk_state.polytope_verts.mink[i_b, vi[3]],
            )
            == RETURN_CODE.FAIL
        ):
            flag = EPA_POLY_INIT_RETURN_CODE.P4_MISSING_ORIGIN

    if flag == EPA_POLY_INIT_RETURN_CODE.SUCCESS:
        # Initialize face map
        for i in ti.static(range(4)):
            gjk_state.polytope_faces_map[i_b, i] = i
            gjk_state.polytope_faces.map_idx[i_b, i] = i
        gjk_state.polytope.nfaces_map[i_b] = 4

    return flag


@ti.func
def func_epa_support(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
    dir,
    dir_norm,
):
    """
    Find support points on the two objects using [dir] and insert them into the polytope.

    Parameters
    ----------
    dir: gs.ti_vec3
        Vector from [ga] (obj1) to [gb] (obj2).
    """
    d = gs.ti_vec3(1, 0, 0)
    if dir_norm > gjk_static_config.FLOAT_MIN:
        d = dir / dir_norm

    # Insert the support points into the polytope
    v_index = func_epa_insert_vertex_to_polytope(
        gjk_state,
        i_b,
        *func_support(
            geoms_state,
            geoms_info,
            verts_info,
            static_rigid_sim_config,
            collider_state,
            collider_static_config,
            gjk_state,
            gjk_static_config,
            support_field_info,
            support_field_static_config,
            i_ga,
            i_gb,
            i_b,
            d,
            False,
        ),
    )

    return v_index


@ti.func
def func_attach_face_to_polytope(
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    i_b,
    i_v1,
    i_v2,
    i_v3,
    i_a1,
    i_a2,
    i_a3,
):
    """
    Attach a face to the polytope.

    [i_v1, i_v2, i_v3] are the vertices of the face, [i_a1, i_a2, i_a3] are the adjacent faces.

    Returns
    -------
    float
        Squared distance of the face to the origin.
    """
    dist2 = 0.0

    n = gjk_state.polytope.nfaces[i_b]
    gjk_state.polytope_faces.verts_idx[i_b, n][0] = i_v1
    gjk_state.polytope_faces.verts_idx[i_b, n][1] = i_v2
    gjk_state.polytope_faces.verts_idx[i_b, n][2] = i_v3
    gjk_state.polytope_faces.adj_idx[i_b, n][0] = i_a1
    gjk_state.polytope_faces.adj_idx[i_b, n][1] = i_a2
    gjk_state.polytope_faces.adj_idx[i_b, n][2] = i_a3
    gjk_state.polytope.nfaces[i_b] += 1

    # Compute the squared distance of the face to the origin
    gjk_state.polytope_faces.normal[i_b, n], ret = func_project_origin_to_plane(
        gjk_static_config,
        gjk_state.polytope_verts.mink[i_b, i_v3],
        gjk_state.polytope_verts.mink[i_b, i_v2],
        gjk_state.polytope_verts.mink[i_b, i_v1],
    )
    if ret == RETURN_CODE.SUCCESS:
        normal = gjk_state.polytope_faces.normal[i_b, n]
        gjk_state.polytope_faces.dist2[i_b, n] = normal.dot(normal)
        gjk_state.polytope_faces.map_idx[i_b, n] = -1  # No map index yet
        dist2 = gjk_state.polytope_faces.dist2[i_b, n]

    return dist2


@ti.func
def func_replace_simplex_3(
    gjk_state: array_class.GJKState,
    i_b,
    i_v1,
    i_v2,
    i_v3,
):
    """
    Replace the simplex with a 2-simplex (triangle) from polytope vertices.

    Parameters
    ----------
    i_v1, i_v2, i_v3: int
        Indices of the vertices in the polytope that will be used to form the triangle.
    """
    gjk_state.simplex.nverts[i_b] = 3
    for i in ti.static(range(3)):
        i_v = i_v1
        if i == 1:
            i_v = i_v2
        elif i == 2:
            i_v = i_v3
        gjk_state.simplex_vertex.obj1[i_b, i] = gjk_state.polytope_verts.obj1[i_b, i_v]
        gjk_state.simplex_vertex.obj2[i_b, i] = gjk_state.polytope_verts.obj2[i_b, i_v]
        gjk_state.simplex_vertex.id1[i_b, i] = gjk_state.polytope_verts.id1[i_b, i_v]
        gjk_state.simplex_vertex.id2[i_b, i] = gjk_state.polytope_verts.id2[i_b, i_v]
        gjk_state.simplex_vertex.mink[i_b, i] = gjk_state.polytope_verts.mink[i_b, i_v]

    # Reset polytope
    gjk_state.polytope.nverts[i_b] = 0
    gjk_state.polytope.nfaces[i_b] = 0
    gjk_state.polytope.nfaces_map[i_b] = 0


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
    gjk_static_config: ti.template(),
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
        is_inside = diff.norm_sqr() < gjk_static_config.FLOAT_MIN_SQ

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
def func_multi_contact(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
    i_f,
):
    """
    Multi-contact detection algorithm based on Sutherland-Hodgman polygon clipping algorithm. For the two geometric
    entities that form the minimum distance (e.g. face-face, edge-face), this function tests if the pair is
    parallel, and if so, it clips one of the pair against the other to find the contact points.

    Parameters
    ----------
    i_f: int
        Index of the face in the EPA polytope where the minimum distance is found.

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

        nnorms = 0
        if geom_type == gs.GEOM_TYPE.BOX:
            nnorms = func_potential_box_normals(
                geoms_state, geoms_info, gjk_state, gjk_static_config, i_g, i_b, nface, v1i, v2i, v3i, t_dir
            )
        elif geom_type == gs.GEOM_TYPE.MESH:
            nnorms = func_potential_mesh_normals(
                geoms_state,
                geoms_info,
                verts_info,
                faces_info,
                gjk_state,
                gjk_static_config,
                i_g,
                i_b,
                nface,
                v1i,
                v2i,
                v3i,
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
    aligned_faces_idx, aligned_faces_flag = func_find_aligned_faces(gjk_state, gjk_static_config, i_b, nnorms1, nnorms2)
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

            nnorms = 0
            if geom_type == gs.GEOM_TYPE.BOX:
                nnorms = func_potential_box_edge_normals(
                    geoms_state, geoms_info, gjk_state, gjk_static_config, i_g, i_b, nface, v1, v2, v1i, v2i
                )
            elif geom_type == gs.GEOM_TYPE.MESH:
                nnorms = func_potential_mesh_edge_normals(
                    geoms_state,
                    geoms_info,
                    verts_info,
                    faces_info,
                    gjk_state,
                    gjk_static_config,
                    i_g,
                    i_b,
                    nface,
                    v1,
                    v2,
                    v1i,
                    v2i,
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
                gjk_state, gjk_static_config, i_b, nedges, nfaces, is_edge_face
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
                    nface = func_box_face(geoms_state, geoms_info, gjk_state, i_g, i_b, k, normal_face_idx)
                elif geom_type == gs.GEOM_TYPE.MESH:
                    nface = func_mesh_face(geoms_state, verts_info, faces_info, gjk_state, i_g, i_b, k, normal_face_idx)

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
        func_clip_polygon(gjk_state, gjk_static_config, i_b, nface1, nface2, edgecon1, edgecon2, normal, approx_dir)


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
def func_potential_box_normals(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    i_g,
    i_b,
    dim,
    v1,
    v2,
    v3,
    dir,
):
    """
    For a simplex defined on a box with three vertices [v1, v2, v3], we find which face normals are potentially
    related to the simplex.

    If the simplex is a triangle, at most one face normal is related.
    If the simplex is a line, at most two face normals are related.
    If the simplex is a point, at most three face normals are related.

    We identify related face normals to the simplex by checking the vertex indices of the simplex.
    """
    g_quat = geoms_state.quat[i_g, i_b]

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
        xyz[i] = func_cmp_bit(v1, v2, v3, dim, i)

    for i in range(1 if dim == 3 else 3):
        # Determine the normal vector in the local space
        local_n = gs.ti_vec3(xyz[0], xyz[1], xyz[2])
        w = 1

        if dim == 2:
            w = xyz[i]

        if dim == 2 or dim == 1:
            local_n = gs.ti_vec3(0, 0, 0)
            local_n[i] = xyz[i]

        global_n = gu.ti_transform_by_quat(local_n, g_quat)

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
            if func_box_normal_from_collision_normal(geoms_state, gjk_state, gjk_static_config, i_g, i_b, dir)
            == RETURN_CODE.SUCCESS
            else 0
        )

    return n_normals


@ti.func
def func_cmp_bit(
    v1,
    v2,
    v3,
    n,
    shift,
):
    """
    Compare one bit of v1 and v2 that sits at position `shift` (shift = 0 for the LSB, 1 for the next bit, …).

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
def func_box_normal_from_collision_normal(
    geoms_state: array_class.GeomsState,
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    i_g,
    i_b,
    dir,
):
    """
    Among the 6 faces of the box, find the one of which normal is closest to the [dir].
    """
    # Every box face normal
    normals = ti.Vector(
        [1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0],
        dt=gs.ti_float,
    )

    # Get local collision normal
    g_quat = geoms_state.quat[i_g, i_b]
    local_dir = gu.ti_transform_by_quat(dir, gu.ti_inv_quat(g_quat))
    local_dir = local_dir.normalized()

    # Determine the closest face normal
    flag = RETURN_CODE.FAIL
    for i in range(6):
        n = gs.ti_vec3(normals[3 * i + 0], normals[3 * i + 1], normals[3 * i + 2])
        if local_dir.dot(n) > gjk_static_config.contact_face_tol:
            flag = RETURN_CODE.SUCCESS
            gjk_state.contact_normals[i_b, 0].normal = n
            gjk_state.contact_normals[i_b, 0].id = i
            break

    return flag


@ti.func
def func_potential_mesh_normals(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    i_g,
    i_b,
    dim,
    v1,
    v2,
    v3,
):
    """
    For a simplex defined on a mesh with three vertices [v1, v2, v3],
    we find which face normals are potentially related to the simplex.

    If the simplex is a triangle, at most one face normal is related.
    If the simplex is a line, at most two face normals are related.
    If the simplex is a point, multiple faces that are adjacent to the point
    could be related.

    We identify related face normals to the simplex by checking the vertex indices of the simplex.
    """
    # Get the geometry state and quaternion
    g_quat = geoms_state.quat[i_g, i_b]

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
            n = gu.ti_transform_by_quat(n, g_quat)

            gjk_state.contact_normals[i_b, n_normals].normal = n
            gjk_state.contact_normals[i_b, n_normals].id = i_f
            n_normals += 1

            if dim == 3:
                break
            elif dim == 2:
                if n_normals == 2:
                    break
            else:
                if n_normals == gjk_static_config.max_contact_polygon_verts:
                    break

    return n_normals


@ti.func
def func_find_aligned_faces(
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
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
        if ni.dot(nj) < -gjk_static_config.contact_face_tol:
            res[0] = i
            res[1] = j
            flag = RETURN_CODE.SUCCESS
            break

    return res, flag


@ti.func
def func_potential_box_edge_normals(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    i_g,
    i_b,
    dim,
    v1,
    v2,
    v1i,
    v2i,
):
    """
    For a simplex defined on a box with two vertices [v1, v2],
    we find which edge normals are potentially related to the simplex.

    If the simplex is a line, at most one edge normal are related.
    If the simplex is a point, at most three edge normals are related.

    We identify related edge normals to the simplex by checking the vertex indices of the simplex.
    """
    # Get the geometry state and quaternion
    g_pos = geoms_state.pos[i_g, i_b]
    g_quat = geoms_state.quat[i_g, i_b]
    g_size_x = geoms_info.data[0] * 0.5
    g_size_y = geoms_info.data[1] * 0.5
    g_size_z = geoms_info.data[2] * 0.5

    v1i -= geoms_info.vert_start[i_g]
    v2i -= geoms_info.vert_start[i_g]

    n_normals = 0

    if dim == 2:
        # If the nearest face is an edge
        gjk_state.contact_normals[i_b, 0].endverts = v2
        gjk_state.contact_normals[i_b, 0].normal = func_safe_normalize(gjk_static_config, v2 - v1)

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
            ev = gu.ti_transform_by_trans_quat(bv, g_pos, g_quat)
            r = func_safe_normalize(gjk_static_config, ev - v1)

            gjk_state.contact_normals[i_b, i].endverts = ev
            gjk_state.contact_normals[i_b, i].normal = r

        n_normals = 3

    return n_normals


@ti.func
def func_potential_mesh_edge_normals(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    i_g,
    i_b,
    dim,
    v1,
    v2,
    v1i,
    v2i,
):
    """
    For a simplex defined on a mesh with two vertices [v1, v2],
    we find which edge normals are potentially related to the simplex.

    If the simplex is a line, at most one edge normal are related.
    If the simplex is a point, multiple edges that are adjacent to the point could be related.

    We identify related edge normals to the simplex by checking the vertex indices of the simplex.
    """
    # Get the geometry state and quaternion
    g_pos = geoms_state.pos[i_g, i_b]
    g_quat = geoms_state.quat[i_g, i_b]

    # Number of potential face normals
    n_normals = 0

    if dim == 2:
        # If the nearest face is an edge
        gjk_state.contact_normals[i_b, 0].endverts = v2
        gjk_state.contact_normals[i_b, 0].normal = func_safe_normalize(gjk_static_config, v2 - v1)

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
                v2_pos = gu.ti_transform_by_trans_quat(v2_pos, g_pos, g_quat)
                t_res = func_safe_normalize(gjk_static_config, v2_pos - v1)

                gjk_state.contact_normals[i_b, n_normals].normal = t_res
                gjk_state.contact_normals[i_b, n_normals].endverts = v2_pos

                n_normals += 1
                if n_normals == gjk_static_config.max_contact_polygon_verts:
                    break

    return n_normals


@ti.func
def func_safe_normalize(
    gjk_static_config: ti.template(),
    v,
):
    """
    Normalize the vector [v] safely.
    """
    norm = v.norm()

    if norm < gjk_static_config.FLOAT_MIN:
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
    gjk_static_config: ti.template(),
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

        if ti.abs(ni.dot(nj)) < gjk_static_config.contact_edge_tol:
            res[0] = i
            res[1] = j
            flag = RETURN_CODE.SUCCESS
            break

    return res, flag


@ti.func
def func_box_face(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    gjk_state: array_class.GJKState,
    i_g,
    i_b,
    i_o,
    face_idx,
):
    """
    Get the face vertices of the box geometry.
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

    # Get geometry position and quaternion
    g_pos = geoms_state.pos[i_g, i_b]
    g_quat = geoms_state.quat[i_g, i_b]

    # Transform the vertices to the global coordinates
    for i in range(nface):
        v = gs.ti_vec3(vs[3 * i + 0], vs[3 * i + 1], vs[3 * i + 2]) * 0.5
        v = gu.ti_transform_by_trans_quat(v, g_pos, g_quat)
        if i_o == 0:
            gjk_state.contact_faces[i_b, i].vert1 = v
        else:
            gjk_state.contact_faces[i_b, i].vert2 = v

    return nface


@ti.func
def func_mesh_face(
    geoms_state: array_class.GeomsState,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    gjk_state: array_class.GJKState,
    i_g,
    i_b,
    i_o,
    face_idx,
):
    """
    Get the face vertices of the mesh.
    """
    # Get geometry position and quaternion
    g_pos = geoms_state.pos[i_g, i_b]
    g_quat = geoms_state.quat[i_g, i_b]

    nvert = 3
    for i in range(nvert):
        i_v = faces_info[face_idx].verts_idx[i]
        v = verts_info.init_pos[i_v]
        v = gu.ti_transform_by_trans_quat(v, g_pos, g_quat)
        if i_o == 0:
            gjk_state.contact_faces[i_b, i].vert1 = v
        else:
            gjk_state.contact_faces[i_b, i].vert2 = v

    return nvert


@ti.func
def func_clip_polygon(
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
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
            inside_v3 = func_halfspace(gjk_static_config, v1, res, v3)
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
                inside_P = func_halfspace(gjk_static_config, a, n, P)
                inside_Q = func_halfspace(gjk_static_config, a, n, Q)

                # PQ entirely outside the clipping edge, skip
                if not inside_P and not inside_Q:
                    continue

                # PQ entirely inside the clipping edge, add Q to the clipped polygon
                if inside_P and inside_Q:
                    gjk_state.contact_clipped_polygons[i_b, ci, nclipped[ci]] = Q
                    nclipped[ci] += 1
                    continue

                # PQ intersects the half-plane, add the intersection point
                t, ip = func_plane_intersect(gjk_static_config, n, d, P, Q)
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
            if gjk_static_config.max_contacts_per_pair < 5 and nclipped_polygon > 4:
                # Approximate the clipped polygon with a convex quadrilateral
                gjk_state.n_witness[i_b] = 4
                rect = func_approximate_polygon_with_quad(gjk_state, i_b, pi, nclipped_polygon)

                for i in range(4):
                    witness2 = gjk_state.contact_clipped_polygons[i_b, pi, rect[i]]
                    witness1 = witness2 - approx_dir
                    gjk_state.witness[i_b, i].point_obj1 = witness1
                    gjk_state.witness[i_b, i].point_obj2 = witness2

            elif nclipped_polygon > gjk_static_config.max_contacts_per_pair:
                # If the number of contacts exceeds the limit,
                # only use the first [max_contacts_per_pair] contacts.
                gjk_state.n_witness[i_b] = gjk_static_config.max_contacts_per_pair

                for i in range(gjk_static_config.max_contacts_per_pair):
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
                        skip = func_is_equal_vec(polygon_vert, prev_witness, gjk_static_config.FLOAT_MIN)
                        if skip:
                            break

                    if not skip:
                        gjk_state.witness[i_b, n_witness].point_obj2 = polygon_vert
                        gjk_state.witness[i_b, n_witness].point_obj1 = polygon_vert - approx_dir
                        n_witness += 1

                gjk_state.n_witness[i_b] = n_witness


@ti.func
def func_halfspace(
    gjk_static_config: ti.template(),
    a,
    n,
    p,
):
    """
    Check if the point [p] is inside the half-space defined by the plane with normal [n] and point [a].
    """
    return (p - a).dot(n) > -gjk_static_config.FLOAT_MIN


@ti.func
def func_plane_intersect(
    gjk_static_config: ti.template(),
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
    t = gjk_static_config.FLOAT_MAX
    ip = gs.ti_vec3(0, 0, 0)

    dir = v2 - v1
    normal_dot = pn.dot(dir)
    if ti.abs(normal_dot) > gjk_static_config.FLOAT_MIN:
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
    gjk_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
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
    support_point_obj1 = gs.ti_vec3(0, 0, 0)
    support_point_obj2 = gs.ti_vec3(0, 0, 0)
    support_point_id_obj1 = -1
    support_point_id_obj2 = -1
    for i in range(2):
        d = dir if i == 0 else -dir
        i_g = i_ga if i == 0 else i_gb

        sp, si = support_driver(
            geoms_state,
            geoms_info,
            verts_info,
            static_rigid_sim_config,
            collider_state,
            collider_static_config,
            gjk_state,
            gjk_static_config,
            support_field_info,
            support_field_static_config,
            d,
            i_g,
            i_b,
            i,
            shrink_sphere,
        )
        if i == 0:
            support_point_obj1 = sp
            support_point_id_obj1 = si
        else:
            support_point_obj2 = sp
            support_point_id_obj2 = si
    support_point_minkowski = support_point_obj1 - support_point_obj2

    return (
        support_point_obj1,
        support_point_obj2,
        support_point_id_obj1,
        support_point_id_obj2,
        support_point_minkowski,
    )


@ti.func
def func_project_origin_to_plane(
    gjk_static_config: ti.template(),
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
        elif nn > gjk_static_config.FLOAT_MIN:
            point = n * (nv / nn)
            flag = RETURN_CODE.SUCCESS
            break

        # Last fallback if no valid normal was found
        if i == 2:
            # If the normal is still unreliable, cannot project.
            if nn < gjk_static_config.FLOAT_MIN:
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
    gjk_static_config: ti.template(),
    direction,
    i_g,
    i_b,
    i_o,
):
    """
    Find the support point on a mesh in the given direction.
    """
    g_quat = geoms_state.quat[i_g, i_b]
    g_pos = geoms_state.pos[i_g, i_b]
    d_mesh = gu.ti_transform_by_quat(direction, gu.ti_inv_quat(g_quat))

    # Exhaustively search for the vertex with maximum dot product
    fmax = -gjk_static_config.FLOAT_MAX
    imax = 0

    vert_start = geoms_info.vert_start[i_g]
    vert_end = geoms_info.vert_end[i_g]

    # Use the previous maximum vertex if it is within the current range
    prev_imax = gjk_state.support_mesh_prev_vertex_id[i_b, i_o]
    if (prev_imax >= vert_start) and (prev_imax < vert_end):
        pos = verts_info.init_pos[prev_imax]
        fmax = d_mesh.dot(pos)
        imax = prev_imax

    for i in range(vert_start, vert_end):
        pos = verts_info.init_pos[i]
        vdot = d_mesh.dot(pos)
        if vdot > fmax:
            fmax = vdot
            imax = i

    v = verts_info.init_pos[imax]
    vid = imax

    gjk_state.support_mesh_prev_vertex_id[i_b, i_o] = vid

    v_ = gu.ti_transform_by_trans_quat(v, g_pos, g_quat)
    return v_, vid


@ti.func
def support_driver(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    direction,
    i_g,
    i_b,
    i_o,
    shrink_sphere,
):
    """
    @ shrink_sphere: If True, use point and line support for sphere and capsule.
    """
    v = ti.Vector.zero(gs.ti_float, 3)
    vid = -1

    geom_type = geoms_info.type[i_g]
    if geom_type == gs.GEOM_TYPE.SPHERE:
        v = support_field._func_support_sphere(geoms_state, geoms_info, direction, i_g, i_b, shrink_sphere)
    elif geom_type == gs.GEOM_TYPE.ELLIPSOID:
        v = support_field._func_support_ellipsoid(geoms_state, geoms_info, direction, i_g, i_b)
    elif geom_type == gs.GEOM_TYPE.CAPSULE:
        v = support_field._func_support_capsule(geoms_state, geoms_info, direction, i_g, i_b, shrink_sphere)
    elif geom_type == gs.GEOM_TYPE.BOX:
        v, vid = support_field._func_support_box(geoms_state, geoms_info, direction, i_g, i_b)
    elif geom_type == gs.GEOM_TYPE.TERRAIN:
        if ti.static(collider_static_config.has_terrain):
            v, vid = support_field._func_support_prism(collider_state, direction, i_g, i_b)
    elif geom_type == gs.GEOM_TYPE.MESH and static_rigid_sim_config.enable_mujoco_compatibility:
        # If mujoco-compatible, do exhaustive search for the vertex
        v, vid = support_mesh(
            geoms_state, geoms_info, verts_info, gjk_state, gjk_static_config, direction, i_g, i_b, i_o
        )
    else:
        v, vid = support_field._func_support_world(
            geoms_state,
            geoms_info,
            support_field_info,
            support_field_static_config,
            direction,
            i_g,
            i_b,
        )
    return v, vid


@ti.func
def func_safe_gjk(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
):
    """
    Safe GJK algorithm to compute the minimum distance between two convex objects.

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
    # Compute the initial tetrahedron using two random directions
    init_flag = RETURN_CODE.SUCCESS
    gjk_state.simplex.nverts[i_b] = 0
    for i in range(4):
        dir = ti.Vector.zero(gs.ti_float, 3)
        dir[2 - i // 2] = 1.0 - 2.0 * (i % 2)

        obj1, obj2, id1, id2, minkowski = func_safe_gjk_support(
            geoms_state,
            geoms_info,
            verts_info,
            static_rigid_sim_config,
            collider_state,
            collider_static_config,
            gjk_state,
            gjk_static_config,
            support_field_info,
            support_field_static_config,
            i_ga,
            i_gb,
            i_b,
            dir,
        )

        # Check if the new vertex would make a valid simplex.
        valid = func_is_new_simplex_vertex_valid(gjk_state, gjk_static_config, i_b, id1, id2, minkowski)

        # If this is not a valid vertex, fall back to a brute-force routine to find a valid vertex.
        if not valid:
            obj1, obj2, id1, id2, minkowski, init_flag = func_search_valid_simplex_vertex(
                geoms_state,
                geoms_info,
                verts_info,
                static_rigid_sim_config,
                collider_state,
                collider_static_config,
                gjk_state,
                gjk_static_config,
                support_field_info,
                support_field_static_config,
                i_ga,
                i_gb,
                i_b,
            )
            # If the brute-force search failed, we cannot proceed with GJK.
            if init_flag == RETURN_CODE.FAIL:
                break

        gjk_state.simplex_vertex.obj1[i_b, i] = obj1
        gjk_state.simplex_vertex.obj2[i_b, i] = obj2
        gjk_state.simplex_vertex.id1[i_b, i] = id1
        gjk_state.simplex_vertex.id2[i_b, i] = id2
        gjk_state.simplex_vertex.mink[i_b, i] = minkowski
        gjk_state.simplex.nverts[i_b] += 1

    gjk_flag = GJK_RETURN_CODE.SEPARATED
    if init_flag == RETURN_CODE.SUCCESS:
        # Simplex index
        si = ti.Vector([0, 1, 2, 3], dt=gs.ti_int)

        for i in range(gjk_static_config.gjk_max_iterations):
            # Compute normal and signed distance of the triangle faces of the simplex with respect to the origin.
            # These normals are supposed to point outwards from the simplex. If the origin is inside the plane,
            # [sdist] will be positive.
            for j in range(4):
                s0, s1, s2, ap = si[2], si[1], si[3], si[0]
                if j == 1:
                    s0, s1, s2, ap = si[0], si[2], si[3], si[1]
                elif j == 2:
                    s0, s1, s2, ap = si[1], si[0], si[3], si[2]
                elif j == 3:
                    s0, s1, s2, ap = si[0], si[1], si[2], si[3]

                n, s = func_safe_gjk_triangle_info(gjk_state, i_b, s0, s1, s2, ap)

                gjk_state.simplex_buffer.normal[i_b, j] = n
                gjk_state.simplex_buffer.sdist[i_b, j] = s

            # Find the face with the smallest signed distance. We need to find [min_i] for the next iteration.
            min_i = 0
            for j in ti.static(range(1, 4)):
                if gjk_state.simplex_buffer.sdist[i_b, j] < gjk_state.simplex_buffer.sdist[i_b, min_i]:
                    min_i = j

            min_si = si[min_i]
            min_normal = gjk_state.simplex_buffer.normal[i_b, min_i]
            min_sdist = gjk_state.simplex_buffer.sdist[i_b, min_i]

            # If origin is inside the simplex, the signed distances will all be positive
            if min_sdist >= 0:
                # Origin is inside the simplex, so we can stop
                gjk_flag = GJK_RETURN_CODE.INTERSECT
                break

            # Check if the new vertex would make a valid simplex.
            gjk_state.simplex.nverts[i_b] = 3
            if min_si != 3:
                gjk_state.simplex_vertex.obj1[i_b, min_si] = gjk_state.simplex_vertex.obj1[i_b, 3]
                gjk_state.simplex_vertex.obj2[i_b, min_si] = gjk_state.simplex_vertex.obj2[i_b, 3]
                gjk_state.simplex_vertex.id1[i_b, min_si] = gjk_state.simplex_vertex.id1[i_b, 3]
                gjk_state.simplex_vertex.id2[i_b, min_si] = gjk_state.simplex_vertex.id2[i_b, 3]
                gjk_state.simplex_vertex.mink[i_b, min_si] = gjk_state.simplex_vertex.mink[i_b, 3]

            # Find a new candidate vertex to replace the worst vertex (which has the smallest signed distance)
            obj1, obj2, id1, id2, minkowski = func_safe_gjk_support(
                geoms_state,
                geoms_info,
                verts_info,
                static_rigid_sim_config,
                collider_state,
                collider_static_config,
                gjk_state,
                gjk_static_config,
                support_field_info,
                support_field_static_config,
                i_ga,
                i_gb,
                i_b,
                min_normal,
            )

            duplicate = func_is_new_simplex_vertex_duplicate(gjk_state, i_b, id1, id2)
            if duplicate:
                # If the new vertex is a duplicate, it means separation.
                gjk_flag = GJK_RETURN_CODE.SEPARATED
                break

            degenerate = func_is_new_simplex_vertex_degenerate(gjk_state, gjk_static_config, i_b, minkowski)
            if degenerate:
                # If the new vertex is degenerate, we cannot proceed with GJK.
                gjk_flag = GJK_RETURN_CODE.NUM_ERROR
                break

            # Check if the origin is strictly outside of the Minkowski difference (which means there is no collision)
            is_no_collision = minkowski.dot(min_normal) < 0.0
            if is_no_collision:
                gjk_flag = GJK_RETURN_CODE.SEPARATED
                break

            gjk_state.simplex_vertex.obj1[i_b, 3] = obj1
            gjk_state.simplex_vertex.obj2[i_b, 3] = obj2
            gjk_state.simplex_vertex.id1[i_b, 3] = id1
            gjk_state.simplex_vertex.id2[i_b, 3] = id2
            gjk_state.simplex_vertex.mink[i_b, 3] = minkowski
            gjk_state.simplex.nverts[i_b] = 4

    if gjk_flag == GJK_RETURN_CODE.INTERSECT:
        gjk_state.distance[i_b] = 0.0
    else:
        gjk_flag = GJK_RETURN_CODE.SEPARATED
        gjk_state.distance[i_b] = gjk_static_config.FLOAT_MAX

    return gjk_flag


@ti.func
def func_is_new_simplex_vertex_valid(
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
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
        not func_is_new_simplex_vertex_degenerate(gjk_state, gjk_static_config, i_b, mink)
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
    gjk_static_config: ti.template(),
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
        if (gjk_state.simplex_vertex.mink[i_b, i] - mink).norm_sqr() < (gjk_static_config.simplex_max_degeneracy_sq):
            is_degenerate = True
            break

    if not is_degenerate:
        # Check the validity based on the simplex dimension
        if nverts == 2:
            # Becomes a triangle if valid, check if the three vertices are not collinear
            is_degenerate = func_is_colinear(
                gjk_static_config,
                gjk_state.simplex_vertex.mink[i_b, 0],
                gjk_state.simplex_vertex.mink[i_b, 1],
                mink,
            )
        elif nverts == 3:
            # Becomes a tetrahedron if valid, check if the four vertices are not coplanar
            is_degenerate = func_is_coplanar(
                gjk_static_config,
                gjk_state.simplex_vertex.mink[i_b, 0],
                gjk_state.simplex_vertex.mink[i_b, 1],
                gjk_state.simplex_vertex.mink[i_b, 2],
                mink,
            )

    return is_degenerate


@ti.func
def func_is_colinear(
    gjk_static_config: ti.template(),
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
    return normal.norm_sqr() < (gjk_static_config.simplex_max_degeneracy_sq) * e1.norm_sqr() * e2.norm_sqr()


@ti.func
def func_is_coplanar(
    gjk_static_config: ti.template(),
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
    return (normal.dot(diff) ** 2) < (gjk_static_config.simplex_max_degeneracy_sq) * normal.norm_sqr() * diff.norm_sqr()


@ti.func
def func_search_valid_simplex_vertex(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
):
    """
    Search for a valid simplex vertex (non-duplicate, non-degenerate) in the Minkowski difference.
    """
    obj1 = gs.ti_vec3(0.0, 0.0, 0.0)
    obj2 = gs.ti_vec3(0.0, 0.0, 0.0)
    id1 = -1
    id2 = -1
    minkowski = gs.ti_vec3(0.0, 0.0, 0.0)
    flag = RETURN_CODE.FAIL

    # If both geometries are discrete, we can use a brute-force search to find a valid simplex vertex.
    if func_is_discrete_geoms(geoms_info, i_ga, i_gb, i_b):
        geom_nverts = gs.ti_ivec2(0, 0)
        for i in range(2):
            geom_nverts[i] = func_num_discrete_geom_vertices(geoms_info, i_ga if i == 0 else i_gb, i_b)

        num_cases = geom_nverts[0] * geom_nverts[1]
        for k in range(num_cases):
            m = (k + gjk_state.last_searched_simplex_vertex_id[i_b]) % num_cases
            i = m // geom_nverts[1]
            j = m % geom_nverts[1]

            id1 = geoms_info.vert_start[i_ga] + i
            id2 = geoms_info.vert_start[i_gb] + j
            for p in range(2):
                obj = func_get_discrete_geom_vertex(
                    geoms_state, geoms_info, verts_info, i_ga if p == 0 else i_gb, i_b, i if p == 0 else j
                )
                if p == 0:
                    obj1 = obj
                else:
                    obj2 = obj
            minkowski = obj1 - obj2

            # Check if the new vertex is valid
            if func_is_new_simplex_vertex_valid(gjk_state, gjk_static_config, i_b, id1, id2, minkowski):
                flag = RETURN_CODE.SUCCESS
                # Update buffer
                gjk_state.last_searched_simplex_vertex_id[i_b] = (m + 1) % num_cases
                break
    else:
        # Try search direction based on the current simplex.
        nverts = gjk_state.simplex.nverts[i_b]
        if nverts == 3:
            # If we have a triangle, use its normal as the search direction.
            v1 = gjk_state.simplex_vertex.mink[i_b, 0]
            v2 = gjk_state.simplex_vertex.mink[i_b, 1]
            v3 = gjk_state.simplex_vertex.mink[i_b, 2]
            dir = (v3 - v1).cross(v2 - v1).normalized()

            for i in range(2):
                d = dir if i == 0 else -dir
                obj1, obj2, id1, id2, minkowski = func_safe_gjk_support(
                    geoms_state,
                    geoms_info,
                    verts_info,
                    static_rigid_sim_config,
                    collider_state,
                    collider_static_config,
                    gjk_state,
                    gjk_static_config,
                    support_field_info,
                    support_field_static_config,
                    i_ga,
                    i_gb,
                    i_b,
                    d,
                )

                # Check if the new vertex is valid
                if func_is_new_simplex_vertex_valid(gjk_state, gjk_static_config, i_b, id1, id2, minkowski):
                    flag = RETURN_CODE.SUCCESS
                    break

    return obj1, obj2, id1, id2, minkowski, flag


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
    """
    geom_type = geoms_info.type[i_g]
    g_pos = geoms_state.pos[i_g, i_b]
    g_quat = geoms_state.quat[i_g, i_b]

    # Get the vertex position in the local frame of the geometry.
    v = ti.Vector([0.0, 0.0, 0.0], dt=gs.ti_float)
    if geom_type == gs.GEOM_TYPE.BOX:
        # For the consistency with the [func_support_box] function of [SupportField] class, we handle the box
        # vertex positions in a different way than the general mesh.
        v = ti.Vector(
            [
                (1.0 if (i_v & 1 == 1) else -1.0) * geoms_info.data[i_g][0] * 0.5,
                (1.0 if (i_v & 2 == 2) else -1.0) * geoms_info.data[i_g][1] * 0.5,
                (1.0 if (i_v & 4 == 4) else -1.0) * geoms_info.data[i_g][2] * 0.5,
            ],
            dt=gs.ti_float,
        )
    elif geom_type == gs.GEOM_TYPE.MESH:
        vert_start = geoms_info.vert_start[i_g]
        v = verts_info.init_pos[vert_start + i_v]

    # Transform the vertex position to the world frame
    v = gu.ti_transform_by_trans_quat(v, g_pos, g_quat)

    return v


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
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
    dir,
):
    """
    Find support points on the two objects using [dir] to use in the [safe_gjk] algorithm.

    This is a more robust version of the support function that finds only one pair of support points, because this
    function perturbs the support direction to find the best support points that guarantee non-degenerate simplex
    in the GJK algorithm.

    Parameters:
    ----------
    dir: gs.ti_vec3
        The unit direction in which to find the support points, from [ga] (obj 1) to [gb] (obj 2).
    """
    obj1 = gs.ti_vec3(0.0, 0.0, 0.0)
    obj2 = gs.ti_vec3(0.0, 0.0, 0.0)
    id1 = gs.ti_int(-1)
    id2 = gs.ti_int(-1)
    mink = obj1 - obj2

    for i in range(9):
        n_dir = dir
        if i > 0:
            j = i - 1
            n_dir[0] += -(1.0 - 2.0 * (j & 1)) * gs.EPS
            n_dir[1] += -(1.0 - 2.0 * (j & 2)) * gs.EPS
            n_dir[2] += -(1.0 - 2.0 * (j & 4)) * gs.EPS

        # First order normalization based on Taylor series is accurate enough
        n_dir *= 2.0 - n_dir.dot(dir)

        num_supports = func_count_support(
            geoms_state,
            geoms_info,
            support_field_info,
            support_field_static_config,
            i_ga,
            i_gb,
            i_b,
            n_dir,
        )
        if i > 0 and num_supports > 1:
            # If this is a perturbed direction and we have more than one support point, we skip this iteration. If
            # it was the original direction, we continue to find the support points to keep it as the baseline.
            continue

        # Use the current direction to find the support points.
        for j in range(2):
            d = n_dir if j == 0 else -n_dir
            i_g = i_ga if j == 0 else i_gb

            sp, si = support_driver(
                geoms_state,
                geoms_info,
                verts_info,
                static_rigid_sim_config,
                collider_state,
                collider_static_config,
                gjk_state,
                gjk_static_config,
                support_field_info,
                support_field_static_config,
                d,
                i_g,
                i_b,
                j,
                False,
            )
            if j == 0:
                obj1 = sp
                id1 = si
            else:
                obj2 = sp
                id2 = si

        mink = obj1 - obj2

        if i == 0:
            if num_supports > 1:
                # If there were multiple valid support points, we move on to the next iteration to perturb the
                # direction and find better support points.
                continue
            else:
                break

        # If it was a perturbed direction, check if the support points have been found before.
        if i == 8:
            # If this was the last iteration, we don't check if it has been found before.
            break

        # Check if the updated simplex would be a degenerate simplex.
        if func_is_new_simplex_vertex_valid(gjk_state, gjk_static_config, i_b, id1, id2, mink):
            break

    return obj1, obj2, id1, id2, mink


@ti.func
def count_support_driver(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    d,
    i_g,
    i_b,
):
    """
    Count the number of possible support points in the given direction.
    """
    geom_type = geoms_info.type[i_g]
    count = 1
    if geom_type == gs.GEOM_TYPE.BOX:
        count = support_field._func_count_supports_box(geoms_state, geoms_info, d, i_g, i_b)
    elif geom_type == gs.GEOM_TYPE.MESH:
        count = support_field._func_count_supports_world(
            geoms_state,
            geoms_info,
            support_field_info,
            support_field_static_config,
            d,
            i_g,
            i_b,
        )
    return count


@ti.func
def func_count_support(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
    dir,
):
    """
    Count the number of possible pairs of support points on the two objects in the given direction [d].
    """
    count = 1
    for i in range(2):
        count *= count_support_driver(
            geoms_state,
            geoms_info,
            support_field_info,
            support_field_static_config,
            dir if i == 0 else -dir,
            i_ga if i == 0 else i_gb,
            i_b,
        )

    return count


@ti.func
def func_safe_epa(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
):
    """
    Safe EPA algorithm to find the exact penetration depth and contact normal using the simplex constructed by GJK.

    This implementation is more robust than the one based on MuJoCo's implementation for the following reasons:
    1) It guarantees that the lower bound of the depth is always smaller than the upper bound, within the tolerance.
    2) This is because we acknowledge that polytope face normal could be unstable when the face is degenerate. Even
    in that case, we can robustly estimate the lower bound of the depth, which gives us more robust results.
    3) In determining the normal direction of a polytope face, we use origin and the polytope vertices altogether
    to get a more stable normal direction, rather than just the origin.
    """
    upper = gjk_static_config.FLOAT_MAX
    upper2 = gjk_static_config.FLOAT_MAX_SQ
    lower = gs.ti_float(0.0)
    tolerance = gjk_static_config.tolerance

    # Index of the nearest face
    nearest_i_f = gs.ti_int(-1)
    prev_nearest_i_f = gs.ti_int(-1)

    discrete = func_is_discrete_geoms(geoms_info, i_ga, i_gb, i_b)
    if discrete:
        # If the objects are discrete, we do not use tolerance.
        tolerance = gs.EPS

    k_max = gjk_static_config.epa_max_iterations
    for k in range(k_max):
        prev_nearest_i_f = nearest_i_f

        # Find the polytope face with the smallest distance to the origin
        lower2 = gjk_static_config.FLOAT_MAX_SQ

        for i in range(gjk_state.polytope.nfaces_map[i_b]):
            i_f = gjk_state.polytope_faces_map[i_b, i]
            face_dist2 = gjk_state.polytope_faces.dist2[i_b, i_f]

            if face_dist2 < lower2:
                lower2 = face_dist2
                nearest_i_f = i_f

        if lower2 > upper2 or nearest_i_f == -1:
            # Invalid face found, stop the algorithm (lower bound of depth is larger than upper bound)
            nearest_i_f = prev_nearest_i_f
            break

        # Find a new support point w from the nearest face's normal
        lower = ti.sqrt(lower2)
        dir = gjk_state.polytope_faces.normal[i_b, nearest_i_f]
        wi = func_epa_support(
            geoms_state,
            geoms_info,
            verts_info,
            static_rigid_sim_config,
            collider_state,
            collider_static_config,
            gjk_state,
            gjk_static_config,
            support_field_info,
            support_field_static_config,
            i_ga,
            i_gb,
            i_b,
            dir,
            1.0,
        )
        w = gjk_state.polytope_verts.mink[i_b, wi]

        # The upper bound of depth at k-th iteration
        upper_k = w.dot(dir)
        if upper_k < upper:
            upper = upper_k
            upper2 = upper**2

        # If the upper bound and lower bound are close enough, we can stop the algorithm
        if (upper - lower) < tolerance:
            break

        if discrete:
            repeated = False
            for i in range(gjk_state.polytope.nverts[i_b]):
                if i == wi:
                    continue
                elif (
                    gjk_state.polytope_verts.id1[i_b, i] == gjk_state.polytope_verts.id1[i_b, wi]
                    and gjk_state.polytope_verts.id2[i_b, i] == gjk_state.polytope_verts.id2[i_b, wi]
                ):
                    # The vertex w is already in the polytope, so we do not need to add it again.
                    repeated = True
                    break
            if repeated:
                break

        gjk_state.polytope.horizon_w[i_b] = w

        # Compute horizon
        horizon_flag = func_epa_horizon(gjk_state, gjk_static_config, i_b, nearest_i_f)

        if horizon_flag:
            # There was an error in the horizon construction, so the horizon edge is not a closed loop.
            nearest_i_f = -1
            break

        if gjk_state.polytope.horizon_nedges[i_b] < 3:
            # Should not happen, because at least three edges should be in the horizon from one deleted face.
            nearest_i_f = -1
            break

        # Check if the memory space is enough for attaching new faces
        nfaces = gjk_state.polytope.nfaces[i_b]
        nedges = gjk_state.polytope.horizon_nedges[i_b]
        if nfaces + nedges >= gjk_static_config.polytope_max_faces:
            # If the polytope is full, we cannot insert new faces
            break

        # Attach the new faces
        # print("Attaching new faces to the polytope")
        attach_flag = RETURN_CODE.SUCCESS
        for i in range(nedges):
            # Face id of the current face to attach
            i_f0 = nfaces + i
            # Face id of the next face to attach
            i_f1 = nfaces + (i + 1) % nedges

            horizon_i_f = gjk_state.polytope_horizon_data.face_idx[i_b, i]
            horizon_i_e = gjk_state.polytope_horizon_data.edge_idx[i_b, i]

            horizon_v1 = gjk_state.polytope_faces.verts_idx[i_b, horizon_i_f][horizon_i_e]
            horizon_v2 = gjk_state.polytope_faces.verts_idx[i_b, horizon_i_f][(horizon_i_e + 1) % 3]

            # Change the adjacent face index of the existing face
            gjk_state.polytope_faces.adj_idx[i_b, horizon_i_f][horizon_i_e] = i_f0

            # Attach the new face.
            # If this if the first face, will be adjacent to the face that will be attached last.
            adj_i_f_0 = i_f0 - 1 if (i > 0) else nfaces + nedges - 1
            adj_i_f_1 = horizon_i_f
            adj_i_f_2 = i_f1

            attach_flag = func_safe_attach_face_to_polytope(
                gjk_state,
                gjk_static_config,
                i_b,
                wi,
                horizon_v2,
                horizon_v1,
                adj_i_f_2,  # Previous face id
                adj_i_f_1,
                adj_i_f_0,  # Next face id
            )
            if attach_flag != RETURN_CODE.SUCCESS:
                # Unrecoverable numerical issue
                break

            dist2 = gjk_state.polytope_faces.dist2[i_b, gjk_state.polytope.nfaces[i_b] - 1]
            if (dist2 >= lower2 - gs.EPS) and (dist2 <= upper2 + gs.EPS):
                # Store face in the map
                nfaces_map = gjk_state.polytope.nfaces_map[i_b]
                gjk_state.polytope_faces_map[i_b, nfaces_map] = i_f0
                gjk_state.polytope_faces.map_idx[i_b, i_f0] = nfaces_map
                gjk_state.polytope.nfaces_map[i_b] += 1

        if attach_flag != RETURN_CODE.SUCCESS:
            break

        # Clear the horizon data for the next iteration
        gjk_state.polytope.horizon_nedges[i_b] = 0

        if (gjk_state.polytope.nfaces_map[i_b] == 0) or (nearest_i_f == -1):
            # No face candidate left
            break

    if nearest_i_f != -1:
        # Nearest face found
        dist2 = gjk_state.polytope_faces.dist2[i_b, nearest_i_f]
        flag = func_safe_epa_witness(gjk_state, gjk_static_config, i_ga, i_gb, i_b, nearest_i_f)
        if flag == RETURN_CODE.SUCCESS:
            gjk_state.n_witness[i_b] = 1
            gjk_state.distance[i_b] = -ti.sqrt(dist2)
        else:
            # Failed to compute witness points, so the objects are not colliding
            gjk_state.n_witness[i_b] = 0
            gjk_state.distance[i_b] = 0.0
    else:
        # No face found, so the objects are not colliding
        gjk_state.n_witness[i_b] = 0
        gjk_state.distance[i_b] = 0.0

    return nearest_i_f


@ti.func
def func_safe_epa_witness(
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
    i_f,
):
    """
    Compute the witness points from the geometries for the face i_f of the polytope.
    """
    flag = RETURN_CODE.SUCCESS

    # Find the affine coordinates of the origin's projection on the face i_f
    face_iv1 = gjk_state.polytope_faces.verts_idx[i_b, i_f][0]
    face_iv2 = gjk_state.polytope_faces.verts_idx[i_b, i_f][1]
    face_iv3 = gjk_state.polytope_faces.verts_idx[i_b, i_f][2]
    face_v1 = gjk_state.polytope_verts.mink[i_b, face_iv1]
    face_v2 = gjk_state.polytope_verts.mink[i_b, face_iv2]
    face_v3 = gjk_state.polytope_verts.mink[i_b, face_iv3]

    # Project origin onto the face plane to get the barycentric coordinates
    proj_o, _ = func_project_origin_to_plane(gjk_static_config, face_v1, face_v2, face_v3)
    _lambda = func_triangle_affine_coords(proj_o, face_v1, face_v2, face_v3)

    # Check validity of affine coordinates through reprojection
    v1 = gjk_state.polytope_verts.mink[i_b, face_iv1]
    v2 = gjk_state.polytope_verts.mink[i_b, face_iv2]
    v3 = gjk_state.polytope_verts.mink[i_b, face_iv3]

    proj_o_lambda = v1 * _lambda[0] + v2 * _lambda[1] + v3 * _lambda[2]
    reprojection_error = (proj_o - proj_o_lambda).norm()

    # Take into account the face magnitude, as the error is relative to the face size.
    max_edge_len_inv = ti.rsqrt(
        max((v1 - v2).norm_sqr(), (v2 - v3).norm_sqr(), (v3 - v1).norm_sqr(), gjk_static_config.FLOAT_MIN_SQ)
    )
    rel_reprojection_error = reprojection_error * max_edge_len_inv
    if rel_reprojection_error > gjk_static_config.polytope_max_reprojection_error:
        flag = RETURN_CODE.FAIL

    if flag == RETURN_CODE.SUCCESS:
        # Point on geom 1
        v1 = gjk_state.polytope_verts.obj1[i_b, face_iv1]
        v2 = gjk_state.polytope_verts.obj1[i_b, face_iv2]
        v3 = gjk_state.polytope_verts.obj1[i_b, face_iv3]
        witness1 = v1 * _lambda[0] + v2 * _lambda[1] + v3 * _lambda[2]

        # Point on geom 2
        v1 = gjk_state.polytope_verts.obj2[i_b, face_iv1]
        v2 = gjk_state.polytope_verts.obj2[i_b, face_iv2]
        v3 = gjk_state.polytope_verts.obj2[i_b, face_iv3]
        witness2 = v1 * _lambda[0] + v2 * _lambda[1] + v3 * _lambda[2]

        gjk_state.witness.point_obj1[i_b, 0] = witness1
        gjk_state.witness.point_obj2[i_b, 0] = witness2

    return flag


@ti.func
def func_safe_epa_init(
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
):
    """
    Create the polytope for safe EPA from a 3-simplex (tetrahedron).

    Assume the tetrahedron is a non-degenerate simplex.
    """

    # Insert simplex vertices into the polytope
    vi = ti.Vector([0, 0, 0, 0], dt=ti.i32)
    for i in range(4):
        vi[i] = func_epa_insert_vertex_to_polytope(
            gjk_state,
            i_b,
            gjk_state.simplex_vertex.obj1[i_b, i],
            gjk_state.simplex_vertex.obj2[i_b, i],
            gjk_state.simplex_vertex.id1[i_b, i],
            gjk_state.simplex_vertex.id2[i_b, i],
            gjk_state.simplex_vertex.mink[i_b, i],
        )

    for i in range(4):
        # Vertex indices for the faces in the hexahedron
        v1, v2, v3 = vi[0], vi[1], vi[2]
        # Adjacent face indices for the faces in the hexahedron
        a1, a2, a3 = 1, 3, 2
        if i == 1:
            v1, v2, v3 = vi[0], vi[3], vi[1]
            a1, a2, a3 = 2, 3, 0
        elif i == 2:
            v1, v2, v3 = vi[0], vi[2], vi[3]
            a1, a2, a3 = 0, 3, 1
        elif i == 3:
            v1, v2, v3 = vi[3], vi[2], vi[1]
            a1, a2, a3 = 2, 0, 1

        func_safe_attach_face_to_polytope(gjk_state, gjk_static_config, i_b, v1, v2, v3, a1, a2, a3)

    # Initialize face map
    for i in ti.static(range(4)):
        gjk_state.polytope_faces_map[i_b, i] = i
        gjk_state.polytope_faces.map_idx[i_b, i] = i
    gjk_state.polytope.nfaces_map[i_b] = 4


@ti.func
def func_safe_attach_face_to_polytope(
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    i_b,
    i_v1,
    i_v2,
    i_v3,
    i_a1,
    i_a2,
    i_a3,
):
    """
    Attach a face to the polytope.

    While attaching the face, 1) determine its normal direction, and 2) estimate the lower bound of the penetration
    depth in robust manner.

    [i_v1, i_v2, i_v3] are the vertices of the face, [i_a1, i_a2, i_a3] are the adjacent faces.
    """
    n = gjk_state.polytope.nfaces[i_b]
    gjk_state.polytope_faces.verts_idx[i_b, n][0] = i_v1
    gjk_state.polytope_faces.verts_idx[i_b, n][1] = i_v2
    gjk_state.polytope_faces.verts_idx[i_b, n][2] = i_v3
    gjk_state.polytope_faces.adj_idx[i_b, n][0] = i_a1
    gjk_state.polytope_faces.adj_idx[i_b, n][1] = i_a2
    gjk_state.polytope_faces.adj_idx[i_b, n][2] = i_a3
    gjk_state.polytope.nfaces[i_b] += 1

    # Compute the normal of the plane
    normal, flag = func_plane_normal(
        gjk_static_config,
        gjk_state.polytope_verts.mink[i_b, i_v3],
        gjk_state.polytope_verts.mink[i_b, i_v2],
        gjk_state.polytope_verts.mink[i_b, i_v1],
    )
    if flag == RETURN_CODE.SUCCESS:
        face_center = (
            gjk_state.polytope_verts.mink[i_b, i_v1]
            + gjk_state.polytope_verts.mink[i_b, i_v2]
            + gjk_state.polytope_verts.mink[i_b, i_v3]
        ) / 3.0

        # Use origin for initialization
        max_orient = -normal.dot(face_center)
        max_abs_orient = ti.abs(max_orient)

        # Consider other vertices in the polytope to reorient the normal
        nverts = gjk_state.polytope.nverts[i_b]
        for i_v in range(nverts):
            if i_v != i_v1 and i_v != i_v2 and i_v != i_v3:
                diff = gjk_state.polytope_verts.mink[i_b, i_v] - face_center
                orient = normal.dot(diff)
                if ti.abs(orient) > max_abs_orient:
                    max_abs_orient = ti.abs(orient)
                    max_orient = orient

        if max_orient > 0.0:
            normal = -normal

        gjk_state.polytope_faces.normal[i_b, n] = normal

        # Compute the safe lower bound of the penetration depth. We can do this by taking the minimum dot product
        # between the face normal and the vertices of the polytope face. This is safer than selecting one of the
        # vertices, because the face normal could be unstable, which ends up in significantly different dot product
        # values for different vertices.
        min_dist2 = gjk_static_config.FLOAT_MAX
        for i in ti.static(range(3)):
            i_v = i_v1
            if i == 1:
                i_v = i_v2
            elif i == 2:
                i_v = i_v3
            v = gjk_state.polytope_verts.mink[i_b, i_v]
            dist2 = normal.dot(v) ** 2
            if dist2 < min_dist2:
                min_dist2 = dist2
        dist2 = min_dist2
        gjk_state.polytope_faces.dist2[i_b, n] = dist2
        gjk_state.polytope_faces.map_idx[i_b, n] = -1  # No map index yet

    return flag


@ti.func
def func_plane_normal(
    gjk_static_config: ti.template(),
    v1,
    v2,
    v3,
):
    """
    Compute the reliable normal of the plane defined by three points.
    """
    normal, flag = gs.ti_vec3(0.0, 0.0, 0.0), RETURN_CODE.FAIL
    finished = False

    d21 = v2 - v1
    d31 = v3 - v1
    d32 = v3 - v2

    for i in ti.static(range(3)):
        if not finished:
            n = gs.ti_vec3(0.0, 0.0, 0.0)
            if i == 0:
                # Normal = (v1 - v2) x (v3 - v2)
                n = d32.cross(d21)
            elif i == 1:
                # Normal = (v2 - v1) x (v3 - v1)
                n = d21.cross(d31)
            else:
                # Normal = (v1 - v3) x (v2 - v3)
                n = d31.cross(d32)
            nn = n.norm()
            if nn == 0:
                # Zero normal, cannot project.
                flag = RETURN_CODE.FAIL
                finished = True
            elif nn > gjk_static_config.FLOAT_MIN:
                normal = n.normalized()
                flag = RETURN_CODE.SUCCESS
                finished = True

    return normal, flag
