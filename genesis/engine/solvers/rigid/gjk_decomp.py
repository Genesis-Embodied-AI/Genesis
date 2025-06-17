import taichi as ti
import genesis as gs
import genesis.utils.geom as gu
from .support_decomp import Support

EPA_P2_NONCONVEX = 2
EPA_P2_FALLBACK3 = 3
EPA_P3_BAD_NORMAL = 4
EPA_P3_INVALID_V4 = 5
EPA_P3_INVALID_V5 = 6
EPA_P3_MISSING_ORIGIN = 7
EPA_P3_ORIGIN_ON_FACE = 8
EPA_P4_MISSING_ORIGIN = 9
EPA_P4_FALLBACK3 = 10


@ti.data_oriented
class GJK:
    def __init__(self, rigid_solver):
        self._solver = rigid_solver
        self._B = rigid_solver._B

        self.FLOAT_MIN = gs.np_float(1e-15)
        self.FLOAT_MIN_SQ = self.FLOAT_MIN * self.FLOAT_MIN
        self.FLOAT_MAX = gs.np_float(1e15)
        self.FLOAT_MAX_SQ = self.FLOAT_MAX * self.FLOAT_MAX
        self.tolerance = gs.np_float(1e-6)
        self.collision_eps = gs.np_float(1e-6)

        # Maximum number of contact points to find per pair
        self.max_contacts_per_pair = 50

        ### GJK
        self.gjk_max_iterations = 50
        struct_simplex_vertex = ti.types.struct(
            # Support points on the two objects
            obj1=gs.ti_vec3,
            obj2=gs.ti_vec3,
            # Vertex on Minkowski difference
            mink=gs.ti_vec3,
            # Vertex ID
            id1=gs.ti_int,
            id2=gs.ti_int,
        )
        struct_simplex = ti.types.struct(
            # Number of vertices in the simplex
            nverts=gs.ti_int,
            dist=gs.ti_float,
        )
        self.gjk_simplex_vertex = struct_simplex_vertex.field(shape=(self._B, 4))
        self.gjk_simplex_vertex_intersect = struct_simplex_vertex.field(shape=(self._B, 4))
        self.gjk_simplex = struct_simplex.field(shape=(self._B,))
        self.gjk_nsimplex = ti.field(dtype=gs.ti_int, shape=(self._B,))

        ### EPA
        self.epa_max_iterations = 50
        struct_polytope_vertex = struct_simplex_vertex
        struct_polytope_face = ti.types.struct(
            # Indices of the vertices forming the face on the polytope
            verts_idx=gs.ti_ivec3,
            # Indices of adjacent faces, one for each edge: [v1,v2], [v2,v3], [v3,v1]
            adj_idx=gs.ti_ivec3,
            # Projection of the origin onto the face, can be used as face normal
            normal=gs.ti_vec3,
            # Square of 2-norm of the normal vector, negative means deleted face
            dist2=gs.ti_float,
            # Index of the face in the polytope map, -1 for not in the map, -2 for deleted
            map_idx=gs.ti_int,
        )
        # Horizon is used for representing the faces to delete
        # when the polytope is expanded by inserting a new vertex.
        struct_polytope_horizon_data = ti.types.struct(
            # Indices of faces on horizon
            face_idx=gs.ti_int,
            # Corresponding edge of each face on the horizon
            edge_idx=gs.ti_int,
        )
        struct_polytope = ti.types.struct(
            # Number of vertices in the polytope
            nverts=gs.ti_int,
            # Number of faces in the polytope
            # (it could include deleted faces)
            nfaces=gs.ti_int,
            # Number of faces in the polytope map
            # (only valid faces on polytope)
            nfaces_map=gs.ti_int,
            # Number of edges in the horizon
            horizon_nedges=gs.ti_int,
            # Support point on the Minkowski difference
            # where the horizon is created
            horizon_w=gs.ti_vec3,
        )
        self.polytope_max_faces = 6 * self.epa_max_iterations

        self.polytope = struct_polytope.field(shape=(self._B,))
        self.polytope_verts = struct_polytope_vertex.field(shape=(self._B, 5 + self.epa_max_iterations))
        self.polytope_faces = struct_polytope_face.field(shape=(self._B, self.polytope_max_faces))
        self.polytope_horizon_data = struct_polytope_horizon_data.field(shape=(self._B, 6 + self.epa_max_iterations))

        # Face indices that form the polytope
        # The first [nfaces_map] indices are the faces that form the polytope
        self.polytope_faces_map = ti.Vector.field(n=self.polytope_max_faces, dtype=gs.ti_int, shape=(self._B,))

        # Stack to use for visiting faces during the horizon construction.
        # The size is (# max faces * 3), because a face has 3 edges.
        self.polytope_horizon_stack = struct_polytope_horizon_data.field(shape=(self._B, self.polytope_max_faces * 3))

        ### Final results
        struct_witness = ti.types.struct(
            # Witness points on the two objects
            point_obj1=gs.ti_vec3,
            point_obj2=gs.ti_vec3,
        )
        self.witness = struct_witness.field(shape=(self._B, self.max_contacts_per_pair))

        # Number of witness points found for each pair
        self.n_witness = ti.field(dtype=gs.ti_int, shape=(self._B,))

        # Distance between the two objects
        # If the objects are separated, the distance is positive.
        # If the objects are intersecting, the distance is negative (depth).
        self.distance = ti.field(dtype=gs.ti_float, shape=(self._B,))

        # Normal vector of the contact point
        self.normal = gs.ti_vec3.field(shape=(self._B,))

        ### Multi-contact
        # Max. number of support vertices in a polygon face
        self.max_polygon_verts = 150
        struct_multi_contact_face = ti.types.struct(
            # Face vertices
            face1=gs.ti_vec3,
            face2=gs.ti_vec3,
            endverts=gs.ti_vec3,
            # Normals of face collisions
            normal1=gs.ti_vec3,
            normal2=gs.ti_vec3,
            # Face ID
            id1=gs.ti_int,
            id2=gs.ti_int,
        )
        # Struct for storing temp. contact normals
        struct_multi_contact_normal = ti.types.struct(
            endverts=gs.ti_vec3,
            # Normal vector of the contact point
            normal=gs.ti_vec3,
            # Face ID
            id=gs.ti_int,
        )
        struct_multi_contact_halfspace = ti.types.struct(
            # Halfspace normal
            normal=gs.ti_vec3,
            # Halfspace distance from the origin
            dist=gs.ti_float,
        )
        self.multi_contact_faces = struct_multi_contact_face.field(shape=(self._B, self.max_polygon_verts))
        self.multi_contact_normals = struct_multi_contact_normal.field(shape=(self._B, self.max_polygon_verts))
        self.multi_contact_halfspaces = struct_multi_contact_halfspace.field(shape=(self._B, self.max_polygon_verts))
        self.multi_contact_clipped_polygons = gs.ti_vec3.field(
            shape=(
                self._B,
                2,
                self.max_polygon_verts,
            )
        )

        # Tolerance for normal alignment of two faces (cosine of 1.6e-3)
        self.face_tol = 0.99999872
        # Tolerance for edge-face alignment (sine of 1.6e-3)
        self.edge_tol = 0.00159999931

        ### Cache for support vertex id
        self.support_vertex_id = ti.field(dtype=gs.ti_int, shape=(self._B, 2))

        ### Support
        self.support = Support(rigid_solver)

    @ti.func
    def clear_cache(self, i_b):
        """
        Clear the cache information for support point query.
        """
        self.support_vertex_id[i_b, 0] = -1
        self.support_vertex_id[i_b, 1] = -1

    """
    Main function
    """

    @ti.func
    def func_gjk_contact(self, i_ga, i_gb, i_b, multi_contact):
        """
        @ multi_contact: If True, use multi-contact detection algorithm if the geoms are discrete.
        """
        self.clear_cache(i_b)

        # Number of maximum contacts that we can find depending on the geometry types.
        is_discrete_geom_a, is_discrete_geom_b = (
            self.func_is_discrete_geom(i_ga, i_b),
            self.func_is_discrete_geom(i_gb, i_b),
        )
        is_discrete_geoms = is_discrete_geom_a and is_discrete_geom_b

        num_max_contacts = 1
        if multi_contact and is_discrete_geoms:
            # Not precise, just approximate.
            num_max_contacts = 4

        # If any one of the geometries is a sphere or capsule,
        # which are sphere-swept primitives, we can shrink them
        # to a point or line to detect shallow penetration faster.
        is_sphere_swept_geom_a, is_sphere_swept_geom_b = (
            self.func_is_sphere_swept_geom(i_ga, i_b),
            self.func_is_sphere_swept_geom(i_gb, i_b),
        )
        shrink_sphere = is_sphere_swept_geom_a or is_sphere_swept_geom_b
        num_gjk_iter = 2 if shrink_sphere else 1

        # Run GJK.
        for i_gjk in range(num_gjk_iter):
            distance = self.func_gjk(i_ga, i_gb, i_b, shrink_sphere)

            if shrink_sphere:
                # If we shrinked the sphere and capsule to point and line and
                # the distance is larger than the collision epsilon, it means
                # a shallow penetration. Thus we subtract the radius of the sphere
                # and the capsule to get the actual distance. If the distance is
                # smaller than the collision epsilon, it means a deep penetration,
                # which requires the default GJK handling.
                if distance > self.collision_eps:
                    radius_a, radius_b = 0.0, 0.0
                    if is_sphere_swept_geom_a:
                        radius_a = self._solver.geoms_info[i_ga].data[0]
                    if is_sphere_swept_geom_b:
                        radius_b = self._solver.geoms_info[i_gb].data[0]

                    wa = self.witness[i_b, 0].point_obj1
                    wb = self.witness[i_b, 0].point_obj2
                    n = self.func_safe_normalize(wb - wa)

                    self.distance[i_b] = distance - (radius_a + radius_b)
                    self.witness[i_b, 0].point_obj1 = wa + (radius_a * n)
                    self.witness[i_b, 0].point_obj2 = wb - (radius_b * n)

                    break

            # Only try shrinking the sphere once
            shrink_sphere = False

        distance = self.distance[i_b]
        nsimplex = self.gjk_nsimplex[i_b]
        collided = distance < self.collision_eps

        # To run EPA, we need following conditions:
        # 1. We did not find min. distance with shrink_sphere flag
        # 2. We have a valid GJK simplex (nsimplex > 0)
        # 3. We have a collision (distance < collision_epsilon)
        do_epa = (not shrink_sphere) and collided and (nsimplex > 0)

        if do_epa:

            # Assume touching
            self.distance[i_b] = 0

            # Initialize polytope
            self.polytope[i_b].nverts = 0
            self.polytope[i_b].nfaces = 0
            self.polytope[i_b].nfaces_map = 0
            self.polytope[i_b].horizon_nedges = 0

            # Construct the initial polytope from the GJK simplex
            polytope_flag = 0
            if nsimplex == 2:
                polytope_flag = self.func_epa_init_polytope_2d(i_ga, i_gb, i_b)
            elif nsimplex == 4:
                polytope_flag = self.func_epa_init_polytope_4d(i_ga, i_gb, i_b)

            # Polytope 3D could be used as a fallback for 2D and 4D cases, but it is not necessary
            if nsimplex == 3 or (polytope_flag == EPA_P2_FALLBACK3 or polytope_flag == EPA_P4_FALLBACK3):
                polytope_flag = self.func_epa_init_polytope_3d(i_ga, i_gb, i_b)

            # Run EPA from the polytope
            if polytope_flag == 0:
                i_f = self.func_epa(i_ga, i_gb, i_b)
                distance = self.distance[i_b]

                if num_max_contacts > 1 and i_f != -1:
                    # Detect multiple contacts
                    self.func_multi_contact(i_ga, i_gb, i_b, i_f)

    """
    GJK algorithms
    """

    @ti.func
    def func_gjk(self, i_ga, i_gb, i_b, shrink_sphere):
        """
        GJK algorithm to compute the minimum distance between two convex objects.
        """

        # Simplex index
        n = 0

        # Final number of simplex vertices
        nsimplex = 0

        # Number of witness points and distance
        nx = 0
        dist = 0.0

        # Lambda for barycentric coordinates
        _lambda = gs.ti_vec4(1.0, 0.0, 0.0, 0.0)

        # Final return flag
        return_flag = 0

        # Set initial guess of support vector using the positions.
        # The support vector should be a non-zero vector.
        approx_witness_point_obj1 = self._solver.geoms_state[i_ga, i_b].pos
        approx_witness_point_obj2 = self._solver.geoms_state[i_gb, i_b].pos
        support_vector = approx_witness_point_obj1 - approx_witness_point_obj2

        if support_vector.dot(support_vector) < self.FLOAT_MIN_SQ:
            approx_witness_point_obj1.x = approx_witness_point_obj1.x + 0.01
            approx_witness_point_obj2.x = approx_witness_point_obj2.x - 0.01
            support_vector = approx_witness_point_obj1 - approx_witness_point_obj2

        support_vector_norm = 0.0

        # Whether or not we need to compute the exact distance.
        # If we shrink the sphere and capsule, we need to compute the distance.
        get_dist = shrink_sphere

        # We can use GJK intersection algorithm only for collision detection
        # if we do not have to compute the distance.
        backup_gjk = not get_dist

        epsilon = 0.0
        if not self.func_is_discrete_geoms(i_ga, i_gb, i_b):
            # If the objects are smooth, finite convergence is not guaranteed,
            # so we need to set some epsilon to determine convergence
            epsilon = 0.5 * self.tolerance * self.tolerance

        for i in range(self.gjk_max_iterations):
            # Compute the current support points
            support_vector_norm = support_vector.dot(support_vector)
            if support_vector_norm < self.FLOAT_MIN_SQ:
                # If the support vector is too small, it means
                # that origin is located in the Minkowski difference
                # with high probability, so we can stop.
                support_vector_norm = ti.math.sqrt(support_vector_norm)
                break

            support_vector_norm = ti.math.sqrt(support_vector_norm)

            # Dir to compute the support point
            # (pointing from obj1 to obj2)
            dir = -support_vector * (1.0 / support_vector_norm)

            (
                self.gjk_simplex_vertex[i_b, n].obj1,
                self.gjk_simplex_vertex[i_b, n].obj2,
                self.gjk_simplex_vertex[i_b, n].id1,
                self.gjk_simplex_vertex[i_b, n].id2,
                self.gjk_simplex_vertex[i_b, n].mink,
            ) = self.func_support(i_ga, i_gb, i_b, dir, shrink_sphere)

            # Early stopping based on Frank-Wolfe duality gap.
            # We need to find the minimum [support_vector_norm], and if
            # we denote it as [x], the problem formulation is: min_x |x|^2.
            # If we denote f(x) = |x|^2, then the Frank-Wolfe duality gap is:
            # |x - x_min|^2 <= < grad f(x), x - s> = < 2x, x - s >,
            # where s is the vertex of the Minkowski difference found by x.
            # Here < 2x, x - s > is guaranteed to be non-negative, and 2
            # is cancelled out in the definition of the epsilon.
            x_k = support_vector
            s_k = self.gjk_simplex_vertex[i_b, n].mink
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
                    dist = self.FLOAT_MAX
                    return_flag = 2
                    break

            if n == 3 and backup_gjk:
                # Tetrahedron is generated, try to detect collision if possible.
                # It is guaranteed that [shrink_sphere] is False here.
                intersect_flag = self.func_gjk_intersect(i_ga, i_gb, i_b, False)
                if intersect_flag == 0:
                    # No intersection, objects are separated
                    nx = 0
                    dist = self.FLOAT_MAX
                    nsimplex = 0
                    return_flag = 2
                    break
                elif intersect_flag == 1:
                    # Intersection found
                    nx = 0
                    dist = 0.0
                    nsimplex = 4
                    return_flag = 1
                    break
                else:
                    # Since gjk_intersect failed (e.g. origin is on the simplex face),
                    # we fallback to minimum distance computation
                    backup_gjk = False

            # Run the distance subalgorithm to compute the barycentric
            # coordinates of the closest point to the origin in the simplex
            _lambda = self.func_simple_gjk_subdistance(i_b, n + 1)

            # Remove vertices from the simplex with zero barycentric coordinates
            # as they are not needed for the next iteration
            n = 0
            for j in range(4):
                if _lambda[j] > 0:
                    self.gjk_simplex_vertex[i_b, n] = self.gjk_simplex_vertex[i_b, j]
                    _lambda[n] = _lambda[j]
                    n += 1

            # Should not occur
            if n < 1:
                nsimplex = 0
                nx = 0
                dist = self.FLOAT_MAX
                return_flag = 3
                break

            # Get the next support vector
            next_support_vector = self.func_simplex_vertex_linear_comb(i_b, 2, 0, 1, 2, 3, _lambda, n)
            if self.func_is_equal_vec3(next_support_vector, support_vector):
                # If the next support vector is equal to the previous one,
                # we converged to the minimum distance
                break

            support_vector = next_support_vector

            if n == 4:
                # We have a tetrahedron containing the origin,
                # so we can return early.
                # This is because only when the origin is inside
                # the tetrahedron, the barycentric coordinates
                # are all positive.
                break

        if return_flag == 0:
            # If [get_dist] was True and there was no numerical error,
            # [return_flag] would be 0 and this logic works.
            nx = 1
            nsimplex = n
            dist = support_vector_norm

            # Compute witness points
            self.witness[i_b, 0].point_obj1 = self.func_simplex_vertex_linear_comb(
                i_b, 0, 0, 1, 2, 3, _lambda, nsimplex
            )
            self.witness[i_b, 0].point_obj2 = self.func_simplex_vertex_linear_comb(
                i_b, 1, 0, 1, 2, 3, _lambda, nsimplex
            )

        self.n_witness[i_b] = nx
        self.distance[i_b] = dist
        self.gjk_nsimplex[i_b] = nsimplex

        return self.distance[i_b]

    @ti.func
    def func_gjk_intersect(self, i_ga, i_gb, i_b, shrink_sphere):

        # copy simplex to temporary storage
        self.gjk_simplex_vertex_intersect[i_b, 0] = self.gjk_simplex_vertex[i_b, 0]
        self.gjk_simplex_vertex_intersect[i_b, 1] = self.gjk_simplex_vertex[i_b, 1]
        self.gjk_simplex_vertex_intersect[i_b, 2] = self.gjk_simplex_vertex[i_b, 2]
        self.gjk_simplex_vertex_intersect[i_b, 3] = self.gjk_simplex_vertex[i_b, 3]

        # simplex index
        si = ti.Vector([0, 1, 2, 3], dt=gs.ti_int)

        flag = -2
        for i in range(self.gjk_max_iterations):
            # Compute normal and signed distance of the triangle faces
            # of the simplex with respect to the origin.
            # These normals are supposed to point outwards from the simplex.
            # If the origin is inside the plane, [sdist] will be positive.
            normal_0, sdist_0 = self.func_gjk_triangle_info(i_b, si[2], si[1], si[3])
            normal_1, sdist_1 = self.func_gjk_triangle_info(i_b, si[0], si[2], si[3])
            normal_2, sdist_2 = self.func_gjk_triangle_info(i_b, si[1], si[0], si[3])
            normal_3, sdist_3 = self.func_gjk_triangle_info(i_b, si[0], si[1], si[2])

            # If the origin is strictly on any affine hull of the triangle faces,
            # convergence will fail, so ignore this case
            if sdist_0 == 0.0 or sdist_1 == 0.0 or sdist_2 == 0.0 or sdist_3 == 0.0:
                flag = -1
                break

            # Find the face with the smallest signed distance
            sdists = gs.ti_vec4(sdist_0, sdist_1, sdist_2, sdist_3)
            min_i = 0
            for j in range(1, 4):
                if sdists[j] < sdists[min_i]:
                    min_i = j
            min_si = si[min_i]

            min_normal = normal_0
            min_sdist = sdist_0
            if min_i == 1:
                min_normal = normal_1
                min_sdist = sdist_1
            elif min_i == 2:
                min_normal = normal_2
                min_sdist = sdist_2
            elif min_i == 3:
                min_normal = normal_3
                min_sdist = sdist_3

            # If origin is inside the simplex, the signed distances
            # will all be positive
            # @TODO: For numerical stability, we use small epsilon, but is it too much?
            if min_sdist >= 0:  # self.FLOAT_MIN:
                # Origin is inside the simplex, so we can stop
                flag = 1

                # Copy the temporary simplex to the main simplex
                self.gjk_simplex_vertex[i_b, 0] = self.gjk_simplex_vertex_intersect[i_b, si[0]]
                self.gjk_simplex_vertex[i_b, 1] = self.gjk_simplex_vertex_intersect[i_b, si[1]]
                self.gjk_simplex_vertex[i_b, 2] = self.gjk_simplex_vertex_intersect[i_b, si[2]]
                self.gjk_simplex_vertex[i_b, 3] = self.gjk_simplex_vertex_intersect[i_b, si[3]]
                break

            # Replace the worst vertex (which has the smallest signed distance)
            # with new candidate
            (
                self.gjk_simplex_vertex_intersect[i_b, min_si].obj1,
                self.gjk_simplex_vertex_intersect[i_b, min_si].obj2,
                self.gjk_simplex_vertex_intersect[i_b, min_si].id1,
                self.gjk_simplex_vertex_intersect[i_b, min_si].id2,
                self.gjk_simplex_vertex_intersect[i_b, min_si].mink,
            ) = self.func_support(i_ga, i_gb, i_b, min_normal, shrink_sphere)

            # Check if the origin is strictly outside of the Minkowski difference
            # (which means there is no collision)
            new_minkowski = self.gjk_simplex_vertex_intersect[i_b, min_si].mink

            # @TODO: For numerical stability, we use small epsilon, but is it too much?
            is_no_collision = new_minkowski.dot(min_normal) < 0  # -self.FLOAT_MIN)
            if is_no_collision:
                flag = 0
                break

            # Swap vertices in the simplex to retain orientation
            m = (min_i + 1) % 4
            n = (min_i + 2) % 4
            swap = si[m]
            si[m] = si[n]
            si[n] = swap

        # Never found origin
        if flag == -2:
            flag = -1

        return flag

    @ti.func
    def func_gjk_triangle_info(self, i_b, i_va, i_vb, i_vc):
        """
        Compute normal and signed distance of the triangle
        face on the simplex from the origin.
        """
        vertex_1 = self.gjk_simplex_vertex_intersect[i_b, i_va].mink
        vertex_2 = self.gjk_simplex_vertex_intersect[i_b, i_vb].mink
        vertex_3 = self.gjk_simplex_vertex_intersect[i_b, i_vc].mink

        edge_1 = vertex_3 - vertex_1
        edge_2 = vertex_2 - vertex_1
        normal = edge_1.cross(edge_2)

        normal_normsq = normal.dot(normal)
        sdist = 0.0
        if (normal_normsq > self.FLOAT_MIN_SQ) and (normal_normsq < self.FLOAT_MAX_SQ):
            normal = normal * (1.0 / ti.math.sqrt(normal_normsq))
            sdist = normal.dot(vertex_1)
        else:
            # if the normal length is unstable, return max distance
            sdist = self.FLOAT_MAX

        return normal, sdist

    @ti.func
    def func_gjk_subdistance(self, i_b, n):
        """
        Compute the barycentric coordinates of the
        closest point to the origin in the simplex.
        [Montanari et al, ToG 2017]
        """
        _lambda = ti.math.vec4(1.0, 0.0, 0.0, 0.0)

        if n == 4:
            _lambda = self.func_gjk_subdistance_3d(i_b, 0, 1, 2, 3)
        elif n == 3:
            _lambda = self.func_gjk_subdistance_2d(i_b, 0, 1, 2)
        elif n == 2:
            _lambda = self.func_gjk_subdistance_1d(i_b, 0, 1)

        return _lambda

    @ti.func
    def func_gjk_subdistance_3d(self, i_b, i_s1, i_s2, i_s3, i_s4):

        _lambda = gs.ti_vec4(0, 0, 0, 0)

        # Simplex vertices
        s1 = self.gjk_simplex_vertex[i_b, i_s1].mink
        s2 = self.gjk_simplex_vertex[i_b, i_s2].mink
        s3 = self.gjk_simplex_vertex[i_b, i_s3].mink
        s4 = self.gjk_simplex_vertex[i_b, i_s4].mink

        # Compute the cofactors to find det(M),
        # which corresponds to the signed volume of the tetrahedron
        Cs = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
        for i in range(4):
            v1, v2, v3 = s2, s3, s4
            if i == 1:
                v1, v2, v3 = s1, s3, s4
            elif i == 2:
                v1, v2, v3 = s1, s2, s4
            elif i == 3:
                v1, v2, v3 = s1, s2, s3
            Cs[i] = self.func_det3(v1, v2, v3)
        Cs[0], Cs[2] = -Cs[0], -Cs[2]
        C1, C2, C3, C4 = Cs[0], Cs[1], Cs[2], Cs[3]
        m_det = C1 + C2 + C3 + C4

        # Compare sign of the cofactors with the determinant
        scs = gs.ti_ivec4(0, 0, 0, 0)
        for i in range(4):
            scs[i] = self.func_compare_sign(Cs[i], m_det)
        sc1, sc2, sc3, sc4 = scs[0], scs[1], scs[2], scs[3]

        if sc1 and sc2 and sc3 and sc4:
            # If all barycentric coordinates are positive,
            # the origin is inside the tetrahedron
            for i in range(4):
                _lambda[i] = Cs[i] / m_det
        else:
            # Since the origin is outside the tetrahedron,
            # we need to find the closest point to the origin
            dmin = self.FLOAT_MAX

            # Project origin onto faces of the tetrahedron
            # of which apex point has negative barycentric coordinate
            for i in range(4):
                j_s1, j_s2, j_s3 = i_s2, i_s3, i_s4
                k_1, k_2, k_3 = 1, 2, 3
                if i == 1:
                    j_s1, j_s2, j_s3 = i_s1, i_s3, i_s4
                    k_1, k_2, k_3 = 0, 2, 3
                elif i == 2:
                    j_s1, j_s2, j_s3 = i_s1, i_s2, i_s4
                    k_1, k_2, k_3 = 0, 1, 3
                elif i == 3:
                    j_s1, j_s2, j_s3 = i_s1, i_s2, i_s3
                    k_1, k_2, k_3 = 0, 1, 2

                if not scs[i]:
                    _lambda2d = self.func_gjk_subdistance_2d(i_b, j_s1, j_s2, j_s3)
                    closest_point = self.func_simplex_vertex_linear_comb(i_b, 2, j_s1, j_s2, j_s3, 0, _lambda2d, 3)
                    d = closest_point.dot(closest_point)
                    if d < dmin:
                        dmin = d
                        _lambda.fill(0.0)
                        _lambda[k_1] = _lambda2d[0]
                        _lambda[k_2] = _lambda2d[1]
                        _lambda[k_3] = _lambda2d[2]

        return _lambda

    @ti.func
    def func_gjk_subdistance_2d(self, i_b, i_s1, i_s2, i_s3):

        _lambda = ti.math.vec4(0, 0, 0, 0)

        # Project origin onto affine hull of the simplex (triangle)
        proj_o, proj_flag = self.func_project_origin_to_plane(
            self.gjk_simplex_vertex[i_b, i_s1].mink,
            self.gjk_simplex_vertex[i_b, i_s2].mink,
            self.gjk_simplex_vertex[i_b, i_s3].mink,
        )
        if proj_flag:
            # If projection failed because the zero normal,
            # project on to the first edge of the triangle
            _lambda = self.func_gjk_subdistance_1d(i_b, i_s1, i_s2)
        else:
            # We should find the barycentric coordinates of the projected point,
            # but the linear system is not square:
            # [ s1.x, s2.x, s3.x ] [ l1 ] = [ proj_o.x ]
            # [ s1.y, s2.y, s3.y ] [ l2 ] = [ proj_o.y ]
            # [ s1.z, s2.z, s3.z ] [ l3 ] = [ proj_o.z ]
            # [ 1,    1,    1,   ] [ ?  ] = [ 1.0 ]
            # So we remove one row before solving the system
            # We exclude the axis with the largest projection of the simplex
            # using the minors of the above linear system.
            s1 = self.gjk_simplex_vertex[i_b, i_s1].mink
            s2 = self.gjk_simplex_vertex[i_b, i_s2].mink
            s3 = self.gjk_simplex_vertex[i_b, i_s3].mink

            m1 = s2[1] * s3[2] - s2[2] * s3[1] - s1[1] * s3[2] + s1[2] * s3[1] + s1[1] * s2[2] - s1[2] * s2[1]
            m2 = s2[0] * s3[2] - s2[2] * s3[0] - s1[0] * s3[2] + s1[2] * s3[0] + s1[0] * s2[2] - s1[2] * s2[0]
            m3 = s2[0] * s3[1] - s2[1] * s3[0] - s1[0] * s3[1] + s1[1] * s3[0] + s1[0] * s2[1] - s1[1] * s2[0]

            m_max = 0.0
            absm1, absm2, absm3 = ti.abs(m1), ti.abs(m2), ti.abs(m3)
            s1_2d, s2_2d, s3_2d = gs.ti_vec2(0, 0), gs.ti_vec2(0, 0), gs.ti_vec2(0, 0)
            proj_o_2d = gs.ti_vec2(0, 0)

            if absm1 >= absm2 and absm1 >= absm3:
                # Remove first row
                m_max = m1
                s1_2d[0] = s1[1]
                s1_2d[1] = s1[2]

                s2_2d[0] = s2[1]
                s2_2d[1] = s2[2]

                s3_2d[0] = s3[1]
                s3_2d[1] = s3[2]

                proj_o_2d[0] = proj_o[1]
                proj_o_2d[1] = proj_o[2]
            elif absm2 >= absm1 and absm2 >= absm3:
                # Remove second row
                m_max = m2
                s1_2d[0] = s1[0]
                s1_2d[1] = s1[2]

                s2_2d[0] = s2[0]
                s2_2d[1] = s2[2]

                s3_2d[0] = s3[0]
                s3_2d[1] = s3[2]

                proj_o_2d[0] = proj_o[0]
                proj_o_2d[1] = proj_o[2]
            else:
                # Remove third row
                m_max = m3
                s1_2d[0] = s1[0]
                s1_2d[1] = s1[1]

                s2_2d[0] = s2[0]
                s2_2d[1] = s2[1]

                s3_2d[0] = s3[0]
                s3_2d[1] = s3[1]

                proj_o_2d[0] = proj_o[0]
                proj_o_2d[1] = proj_o[1]

            # Now we find the barycentric coordinates of the projected point
            # by solving the linear system:
            # [ s1_2d.x, s2_2d.x, s3_2d.x ] [ l1 ] = [ proj_o_2d.x ]
            # [ s1_2d.y, s2_2d.y, s3_2d.y ] [ l2 ] = [ proj_o_2d.y ]
            # [ 1,       1,       1,      ] [ l3 ] = [ 1.0 ]

            # C1 corresponds to the signed area of 2-simplex (triangle): (proj_o_2d, s2_2d, s3_2d)
            C1 = (
                proj_o_2d[0] * s2_2d[1]
                + proj_o_2d[1] * s3_2d[0]
                + s2_2d[0] * s3_2d[1]
                - proj_o_2d[0] * s3_2d[1]
                - proj_o_2d[1] * s2_2d[0]
                - s3_2d[0] * s2_2d[1]
            )

            # C2 corresponds to the signed area of 2-simplex (triangle): (proj_o_2d, s1_2d, s3_2d)
            C2 = (
                proj_o_2d[0] * s3_2d[1]
                + proj_o_2d[1] * s1_2d[0]
                + s3_2d[0] * s1_2d[1]
                - proj_o_2d[0] * s1_2d[1]
                - proj_o_2d[1] * s3_2d[0]
                - s1_2d[0] * s3_2d[1]
            )

            # C3 corresponds to the signed area of 2-simplex (triangle): (proj_o_2d, s1_2d, s2_2d)
            C3 = (
                proj_o_2d[0] * s1_2d[1]
                + proj_o_2d[1] * s2_2d[0]
                + s1_2d[0] * s2_2d[1]
                - proj_o_2d[0] * s2_2d[1]
                - proj_o_2d[1] * s1_2d[0]
                - s2_2d[0] * s1_2d[1]
            )

            Cs = gs.ti_vec3(C1, C2, C3)

            # Compare sign of the cofactors with the determinant
            scs = gs.ti_ivec3(0, 0, 0)
            for i in range(3):
                scs[i] = self.func_compare_sign(Cs[i], m_max)
            sc1, sc2, sc3 = scs[0], scs[1], scs[2]

            if sc1 and sc2 and sc3:
                # If all barycentric coordinates are positive,
                # the origin is inside the 2-simplex (triangle)
                _lambda[0] = C1 / m_max
                _lambda[1] = C2 / m_max
                _lambda[2] = C3 / m_max
                _lambda[3] = 0.0
            else:
                # Since the origin is outside the 2-simplex (triangle),
                # we need to find the closest point to the origin
                dmin = self.FLOAT_MAX

                # Project origin onto edges of the triangle
                # of which apex point has negative barycentric coordinate
                for i in range(3):
                    j_s1, j_s2 = i_s2, i_s3
                    k_1, k_2 = 1, 2
                    if i == 1:
                        j_s1, j_s2 = i_s1, i_s3
                        k_1, k_2 = 0, 2
                    elif i == 2:
                        j_s1, j_s2 = i_s1, i_s2
                        k_1, k_2 = 0, 1

                    if not scs[i]:
                        _lambda1d = self.func_gjk_subdistance_1d(i_b, j_s1, j_s2)
                        closest_point = self.func_simplex_vertex_linear_comb(i_b, 2, j_s1, j_s2, 0, 0, _lambda1d, 2)
                        d = closest_point.dot(closest_point)
                        if d < dmin:
                            dmin = d
                            _lambda.fill(0.0)
                            _lambda[k_1] = _lambda1d[0]
                            _lambda[k_2] = _lambda1d[1]

        return _lambda

    @ti.func
    def func_gjk_subdistance_1d(self, i_b, i_s1, i_s2):

        _lambda = gs.ti_vec4(0, 0, 0, 0)

        s1 = self.gjk_simplex_vertex[i_b, i_s1].mink
        s2 = self.gjk_simplex_vertex[i_b, i_s2].mink
        p_o = self.func_project_origin_to_line(s1, s2)

        mu = s1[0] - s2[0]
        mu_max = mu
        index = 0

        mu = s1[1] - s2[1]
        if ti.abs(mu) >= ti.abs(mu_max):
            mu_max = mu
            index = 1

        mu = s1[2] - s2[2]
        if ti.abs(mu) >= ti.abs(mu_max):
            mu_max = mu
            index = 2

        C1 = p_o[index] - s2[index]
        C2 = s1[index] - p_o[index]

        # Determine if projection of origin lies inside 1-simplex
        same = self.func_compare_sign(mu_max, C1) and self.func_compare_sign(mu_max, C2)
        if same:
            _lambda[0] = C1 / mu_max
            _lambda[1] = C2 / mu_max
        else:
            _lambda[0] = 0.0
            _lambda[1] = 1.0

        return _lambda

    @ti.func
    def func_simple_gjk_subdistance(self, i_b, n):
        _lambda = ti.math.vec4(1.0, 0.0, 0.0, 0.0)

        # Whether or not the subdistance was computed
        # successfully for the n-simplex.
        flag = 1

        dmin = self.FLOAT_MAX

        if n == 4:
            _lambda, flag3d = self.func_simple_gjk_subdistance_3d(i_b, 0, 1, 2, 3)
            flag = flag3d

        if (flag == 0) or n == 3:
            failed_3d = n == 4
            num_iter = 1
            if failed_3d:
                # Iterate through 4 faces of the tetrahedron
                num_iter = 4

            for i in range(num_iter):
                k_1, k_2, k_3 = 0, 1, 2
                if i == 1:
                    k_1, k_2, k_3 = 1, 2, 3
                elif i == 2:
                    k_1, k_2, k_3 = 0, 2, 3
                elif i == 3:
                    k_1, k_2, k_3 = 0, 1, 3

                _lambda2d, flag2d = self.func_simple_gjk_subdistance_2d(i_b, k_1, k_2, k_3)

                if failed_3d:
                    if flag2d:
                        closest_point = self.func_simplex_vertex_linear_comb(i_b, 2, k_1, k_2, k_3, 0, _lambda2d, 3)
                        d = closest_point.dot(closest_point)
                        if d < dmin:
                            dmin = d
                            _lambda.fill(0.0)
                            _lambda[k_1] = _lambda2d[0]
                            _lambda[k_2] = _lambda2d[1]
                            _lambda[k_3] = _lambda2d[2]
                else:
                    if flag2d:
                        _lambda = _lambda2d
                    flag = flag2d

        if (flag == 0) or n == 2:
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
                k_1, k_2 = 0, 1
                if i == 1:
                    k_1, k_2 = 0, 2
                elif i == 2:
                    k_1, k_2 = 1, 2
                elif i == 3:
                    k_1, k_2 = 0, 3
                elif i == 4:
                    k_1, k_2 = 1, 3
                elif i == 5:
                    k_1, k_2 = 2, 3

                _lambda1d = self.func_simple_gjk_subdistance_1d(i_b, k_1, k_2)

                if failed_3d or failed_2d:
                    closest_point = self.func_simplex_vertex_linear_comb(i_b, 2, k_1, k_2, 0, 0, _lambda1d, 2)
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
    def func_simple_gjk_subdistance_3d(self, i_b, i_s1, i_s2, i_s3, i_s4):

        flag = 0
        _lambda = gs.ti_vec4(0, 0, 0, 0)

        # Simplex vertices
        s1 = self.gjk_simplex_vertex[i_b, i_s1].mink
        s2 = self.gjk_simplex_vertex[i_b, i_s2].mink
        s3 = self.gjk_simplex_vertex[i_b, i_s3].mink
        s4 = self.gjk_simplex_vertex[i_b, i_s4].mink

        # Compute the cofactors to find det(M),
        # which corresponds to the signed volume of the tetrahedron
        Cs = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
        for i in range(4):
            v1, v2, v3 = s2, s3, s4
            if i == 1:
                v1, v2, v3 = s1, s3, s4
            elif i == 2:
                v1, v2, v3 = s1, s2, s4
            elif i == 3:
                v1, v2, v3 = s1, s2, s3
            Cs[i] = self.func_det3(v1, v2, v3)
        Cs[0], Cs[2] = -Cs[0], -Cs[2]
        C1, C2, C3, C4 = Cs[0], Cs[1], Cs[2], Cs[3]
        m_det = C1 + C2 + C3 + C4

        # Compare sign of the cofactors with the determinant
        scs = gs.ti_ivec4(0, 0, 0, 0)
        for i in range(4):
            scs[i] = self.func_compare_sign(Cs[i], m_det)
        sc1, sc2, sc3, sc4 = scs[0], scs[1], scs[2], scs[3]

        if sc1 and sc2 and sc3 and sc4:
            # If all barycentric coordinates are positive,
            # the origin is inside the tetrahedron
            for i in range(4):
                _lambda[i] = Cs[i] / m_det
            flag = 1

        return _lambda, flag

    @ti.func
    def func_simple_gjk_subdistance_2d(self, i_b, i_s1, i_s2, i_s3):

        _lambda = ti.math.vec4(0, 0, 0, 0)
        flag = 0

        # Project origin onto affine hull of the simplex (triangle)
        proj_o, proj_flag = self.func_project_origin_to_plane(
            self.gjk_simplex_vertex[i_b, i_s1].mink,
            self.gjk_simplex_vertex[i_b, i_s2].mink,
            self.gjk_simplex_vertex[i_b, i_s3].mink,
        )

        if not proj_flag:
            # We should find the barycentric coordinates of the projected point,
            # but the linear system is not square:
            # [ s1.x, s2.x, s3.x ] [ l1 ] = [ proj_o.x ]
            # [ s1.y, s2.y, s3.y ] [ l2 ] = [ proj_o.y ]
            # [ s1.z, s2.z, s3.z ] [ l3 ] = [ proj_o.z ]
            # [ 1,    1,    1,   ] [ ?  ] = [ 1.0 ]
            # So we remove one row before solving the system
            # We exclude the axis with the largest projection of the simplex
            # using the minors of the above linear system.
            s1 = self.gjk_simplex_vertex[i_b, i_s1].mink
            s2 = self.gjk_simplex_vertex[i_b, i_s2].mink
            s3 = self.gjk_simplex_vertex[i_b, i_s3].mink

            m1 = s2[1] * s3[2] - s2[2] * s3[1] - s1[1] * s3[2] + s1[2] * s3[1] + s1[1] * s2[2] - s1[2] * s2[1]
            m2 = s2[0] * s3[2] - s2[2] * s3[0] - s1[0] * s3[2] + s1[2] * s3[0] + s1[0] * s2[2] - s1[2] * s2[0]
            m3 = s2[0] * s3[1] - s2[1] * s3[0] - s1[0] * s3[1] + s1[1] * s3[0] + s1[0] * s2[1] - s1[1] * s2[0]

            m_max = 0.0
            absm1, absm2, absm3 = ti.abs(m1), ti.abs(m2), ti.abs(m3)
            s1_2d, s2_2d, s3_2d = gs.ti_vec2(0, 0), gs.ti_vec2(0, 0), gs.ti_vec2(0, 0)
            proj_o_2d = gs.ti_vec2(0, 0)

            if absm1 >= absm2 and absm1 >= absm3:
                # Remove first row
                m_max = m1
                s1_2d[0] = s1[1]
                s1_2d[1] = s1[2]

                s2_2d[0] = s2[1]
                s2_2d[1] = s2[2]

                s3_2d[0] = s3[1]
                s3_2d[1] = s3[2]

                proj_o_2d[0] = proj_o[1]
                proj_o_2d[1] = proj_o[2]
            elif absm2 >= absm1 and absm2 >= absm3:
                # Remove second row
                m_max = m2
                s1_2d[0] = s1[0]
                s1_2d[1] = s1[2]

                s2_2d[0] = s2[0]
                s2_2d[1] = s2[2]

                s3_2d[0] = s3[0]
                s3_2d[1] = s3[2]

                proj_o_2d[0] = proj_o[0]
                proj_o_2d[1] = proj_o[2]
            else:
                # Remove third row
                m_max = m3
                s1_2d[0] = s1[0]
                s1_2d[1] = s1[1]

                s2_2d[0] = s2[0]
                s2_2d[1] = s2[1]

                s3_2d[0] = s3[0]
                s3_2d[1] = s3[1]

                proj_o_2d[0] = proj_o[0]
                proj_o_2d[1] = proj_o[1]

            # Now we find the barycentric coordinates of the projected point
            # by solving the linear system:
            # [ s1_2d.x, s2_2d.x, s3_2d.x ] [ l1 ] = [ proj_o_2d.x ]
            # [ s1_2d.y, s2_2d.y, s3_2d.y ] [ l2 ] = [ proj_o_2d.y ]
            # [ 1,       1,       1,      ] [ l3 ] = [ 1.0 ]

            # C1 corresponds to the signed area of 2-simplex (triangle): (proj_o_2d, s2_2d, s3_2d)
            C1 = (
                proj_o_2d[0] * s2_2d[1]
                + proj_o_2d[1] * s3_2d[0]
                + s2_2d[0] * s3_2d[1]
                - proj_o_2d[0] * s3_2d[1]
                - proj_o_2d[1] * s2_2d[0]
                - s3_2d[0] * s2_2d[1]
            )

            # C2 corresponds to the signed area of 2-simplex (triangle): (proj_o_2d, s1_2d, s3_2d)
            C2 = (
                proj_o_2d[0] * s3_2d[1]
                + proj_o_2d[1] * s1_2d[0]
                + s3_2d[0] * s1_2d[1]
                - proj_o_2d[0] * s1_2d[1]
                - proj_o_2d[1] * s3_2d[0]
                - s1_2d[0] * s3_2d[1]
            )

            # C3 corresponds to the signed area of 2-simplex (triangle): (proj_o_2d, s1_2d, s2_2d)
            C3 = (
                proj_o_2d[0] * s1_2d[1]
                + proj_o_2d[1] * s2_2d[0]
                + s1_2d[0] * s2_2d[1]
                - proj_o_2d[0] * s2_2d[1]
                - proj_o_2d[1] * s1_2d[0]
                - s2_2d[0] * s1_2d[1]
            )

            Cs = gs.ti_vec3(C1, C2, C3)

            # Compare sign of the cofactors with the determinant
            scs = gs.ti_ivec3(0, 0, 0)
            for i in range(3):
                scs[i] = self.func_compare_sign(Cs[i], m_max)
            sc1, sc2, sc3 = scs[0], scs[1], scs[2]

            if sc1 and sc2 and sc3:
                # If all barycentric coordinates are positive,
                # the origin is inside the 2-simplex (triangle)
                _lambda[0] = C1 / m_max
                _lambda[1] = C2 / m_max
                _lambda[2] = C3 / m_max
                _lambda[3] = 0.0
                flag = 1

        return _lambda, flag

    @ti.func
    def func_simple_gjk_subdistance_1d(self, i_b, i_s1, i_s2):

        _lambda = gs.ti_vec4(0, 0, 0, 0)

        s1 = self.gjk_simplex_vertex[i_b, i_s1].mink
        s2 = self.gjk_simplex_vertex[i_b, i_s2].mink
        p_o = self.func_project_origin_to_line(s1, s2)

        mu = s1[0] - s2[0]
        mu_max = mu
        index = 0

        mu = s1[1] - s2[1]
        if ti.abs(mu) >= ti.abs(mu_max):
            mu_max = mu
            index = 1

        mu = s1[2] - s2[2]
        if ti.abs(mu) >= ti.abs(mu_max):
            mu_max = mu
            index = 2

        C1 = p_o[index] - s2[index]
        C2 = s1[index] - p_o[index]

        # Determine if projection of origin lies inside 1-simplex
        same = self.func_compare_sign(mu_max, C1) and self.func_compare_sign(mu_max, C2)
        if same:
            _lambda[0] = C1 / mu_max
            _lambda[1] = C2 / mu_max
        else:
            _lambda[0] = 0.0
            _lambda[1] = 1.0

        return _lambda

    """
    EPA algorithms
    """

    @ti.func
    def func_epa(self, i_ga, i_gb, i_b):
        """
        Run EPA algorithm for finding the face on the expanded
        polytope that best approximates the penetration depth.
        """
        upper = self.FLOAT_MAX
        upper2 = self.FLOAT_MAX_SQ
        lower = 0.0
        tolerance = self.tolerance

        # Index of the nearest face
        nearest_i_f = -1
        prev_nearest_i_f = -1

        discrete = self.func_is_discrete_geoms(i_ga, i_gb, i_b)
        if discrete and gs.ti_float == ti.f64:
            # If the objects are discrete, we do not use tolerance.
            tolerance = self.FLOAT_MIN

        k_max = self.epa_max_iterations
        for k in range(k_max):
            prev_nearest_i_f = nearest_i_f

            # Find the polytope face with the smallest distance to the origin
            lower2 = self.FLOAT_MAX_SQ

            for i in range(self.polytope[i_b].nfaces_map):
                i_f = self.polytope_faces_map[i_b][i]
                face_dist2 = self.polytope_faces[i_b, i_f].dist2

                if (i == 0) or (face_dist2 < lower2):
                    lower2 = face_dist2
                    nearest_i_f = i_f

            if lower2 > upper2 or nearest_i_f < 0:
                # Invalid face found, stop the algorithm
                # (lower bound of depth is larger than upper bound)
                nearest_i_f = prev_nearest_i_f
                break

            if lower2 <= 0:
                # Invalid lower bound (0), stop the algorithm
                # (origin is on the affine hull of face)
                break

            # Find a new support point w from the nearest face's normal
            lower = ti.sqrt(lower2)
            dir = self.polytope_faces[i_b, nearest_i_f].normal
            wi = self.func_epa_support(i_ga, i_gb, i_b, dir, lower)
            w = self.polytope_verts[i_b, wi].mink

            # The upper bound of depth at k-th iteration
            upper_k = w.dot(dir) / lower
            if upper_k < upper:
                upper = upper_k
                upper2 = upper * upper

            # If the upper bound and lower bound are close enough,
            # we can stop the algorithm
            if (upper - lower) < tolerance:
                break

            # @TODO: Check the vertex w is already in the polytope
            if discrete:
                repeated = False
                for i in range(self.polytope[i_b].nverts - 1):
                    if (
                        self.polytope_verts[i_b, i].id1 == self.polytope_verts[i_b, wi].id1
                        and self.polytope_verts[i_b, i].id2 == self.polytope_verts[i_b, wi].id2
                    ):
                        # The vertex w is already in the polytope,
                        # so we do not need to add it again.
                        repeated = True
                        break
                if repeated:
                    break

            self.polytope[i_b].horizon_w = w

            # Compute horizon
            horizon_flag = self.func_epa_horizon(i_b, nearest_i_f)

            if horizon_flag:
                # There was an error in the horizon construction,
                # so the horizon edge is not a closed loop.
                nearest_i_f = -1
                break

            if self.polytope[i_b].horizon_nedges < 3:
                # Should not happen, because at least one face is deleted
                # and thus at least three edges should be in the horizon.
                nearest_i_f = -1
                break

            # Check if the memory space is enough for attaching new faces
            nfaces = self.polytope[i_b].nfaces
            nedges = self.polytope[i_b].horizon_nedges
            if nfaces + nedges >= self.polytope_max_faces:
                # If the polytope is full, we cannot insert new faces
                break

            # Attach the new faces
            for i in range(nedges):
                # Face id of the current face to attach
                i_f0 = nfaces + i
                # Face id of the next face to attach
                i_f1 = nfaces + (i + 1) % nedges

                horizon_i_f = self.polytope_horizon_data[i_b, i].face_idx
                horizon_i_e = self.polytope_horizon_data[i_b, i].edge_idx
                horizon_face = self.polytope_faces[i_b, horizon_i_f]
                horizon_v1 = horizon_face.verts_idx[horizon_i_e]
                horizon_v2 = horizon_face.verts_idx[(horizon_i_e + 1) % 3]

                # Change the adjacent face index of the existing face
                self.polytope_faces[i_b, horizon_i_f].adj_idx[horizon_i_e] = i_f0

                # Attach the new face
                # If this if the first face, will be adjacent to
                # the face that will be attached last
                adj_i_f_0 = i_f0 - 1 if (i > 0) else nfaces + nedges - 1
                adj_i_f_1 = horizon_i_f
                adj_i_f_2 = i_f1

                dist2 = self.func_attach_face_to_polytope(
                    i_b,
                    wi,
                    horizon_v2,
                    horizon_v1,
                    # Previous face id
                    adj_i_f_2,
                    adj_i_f_1,
                    # Next face id
                    adj_i_f_0,
                )
                if dist2 <= 0:
                    # Unrecoverable numerical issue
                    nearest_i_f = -1
                    break

                if (dist2 >= lower2) and (dist2 <= upper2):
                    # Store face in the map
                    nfaces_map = self.polytope[i_b].nfaces_map
                    self.polytope_faces_map[i_b][nfaces_map] = i_f0
                    self.polytope_faces[i_b, i_f0].map_idx = nfaces_map
                    self.polytope[i_b].nfaces_map += 1

            # Clear the horizon data for the next iteration
            self.polytope[i_b].horizon_nedges = 0

            if (self.polytope[i_b].nfaces_map == 0) or (nearest_i_f == -1):
                # No face candidate left
                break

        if nearest_i_f != -1:
            # Nearest face found
            dist2 = self.polytope_faces[i_b, nearest_i_f].dist2
            self.func_epa_witness(i_ga, i_gb, i_b, nearest_i_f)
            self.n_witness[i_b] = 1
            self.distance[i_b] = -ti.sqrt(dist2)
        else:
            # No face found, so the objects are not colliding
            self.n_witness[i_b] = 0
            self.distance[i_b] = 0

        return nearest_i_f

    @ti.func
    def func_epa_witness(self, i_ga, i_gb, i_b, i_f):
        """
        Compute the witness points from the geometries
        for the face i_f of the polytope.
        """
        # Find the affine coordinates of the origin's
        # projection on the face i_f
        face = self.polytope_faces[i_b, i_f]
        face_v1 = self.polytope_verts[i_b, face.verts_idx[0]].mink
        face_v2 = self.polytope_verts[i_b, face.verts_idx[1]].mink
        face_v3 = self.polytope_verts[i_b, face.verts_idx[2]].mink
        face_normal = face.normal

        _lambda = self.func_triangle_affine_coords(
            face_normal,
            face_v1,
            face_v2,
            face_v3,
        )

        # Point on geom 1
        v1 = self.polytope_verts[i_b, face.verts_idx[0]].obj1
        v2 = self.polytope_verts[i_b, face.verts_idx[1]].obj1
        v3 = self.polytope_verts[i_b, face.verts_idx[2]].obj1
        witness1 = v1 * _lambda[0] + v2 * _lambda[1] + v3 * _lambda[2]

        # Point on geom 2
        v1 = self.polytope_verts[i_b, face.verts_idx[0]].obj2
        v2 = self.polytope_verts[i_b, face.verts_idx[1]].obj2
        v3 = self.polytope_verts[i_b, face.verts_idx[2]].obj2
        witness2 = v1 * _lambda[0] + v2 * _lambda[1] + v3 * _lambda[2]

        self.witness[i_b, 0].point_obj1 = witness1
        self.witness[i_b, 0].point_obj2 = witness2

        # Safe measure for mujoco consistency
        normal = witness1 - witness2
        normal_length = ti.sqrt(normal.dot(normal))
        if normal_length < self.FLOAT_MIN:
            normal = gs.ti_vec3(1.0, 0.0, 0.0)  # Default normal if zero length
        else:
            normal = normal * (1.0 / normal_length)  # Normalize the normal vector
        self.normal[i_b] = normal

    @ti.func
    def func_epa_horizon(self, i_b, nearest_i_f):
        """
        Compute the horizon, which represents the area of the polytope
        that is visible from the vertex w, and thus should be deleted
        for the expansion of the polytope.
        """
        w = self.polytope[i_b].horizon_w

        # Initialize the stack by inserting the nearest face
        self.polytope_horizon_stack[i_b, 0].face_idx = nearest_i_f
        self.polytope_horizon_stack[i_b, 0].edge_idx = 0
        top = 1
        is_first = True

        flag = 0
        while top > 0:

            # Pop the top face from the stack
            i_f = self.polytope_horizon_stack[i_b, top - 1].face_idx
            i_e = self.polytope_horizon_stack[i_b, top - 1].edge_idx
            top -= 1

            # If the face is already deleted, skip it
            is_deleted = self.polytope_faces[i_b, i_f].map_idx == -2
            if (not is_first) and (is_deleted):
                continue

            face = self.polytope_faces[i_b, i_f]

            # Check visibility of the face.
            # Two requirements for the face to be visible:
            # 1. The face normal should point towards the vertex w
            # 2. The vertex w should be on the other side of the face
            #    to the origin
            is_visible = face.normal.dot(w) - face.dist2 > self.FLOAT_MIN

            # The first face is always visible.
            is_visible = is_visible or is_first

            if is_visible:

                # If visible, delete the face from the polytope
                self.func_delete_face_from_polytope(i_b, i_f)

                # If it is the first face, we need to add the three edges.
                k_beg = 0 if is_first else 1

                # Add the other two or three edges of the face to the stack.
                # The order is important to form a closed loop.
                for k in range(k_beg, 3):
                    i_e2 = (i_e + k) % 3
                    adj_face_idx = face.adj_idx[i_e2]
                    adj_face_is_deleted = self.polytope_faces[i_b, adj_face_idx].map_idx == -2
                    if not adj_face_is_deleted:
                        # Get the related edge id from the adjacent face.
                        # Since adjacent faces have different orientations,
                        # we need to use the ending vertex of the edge.
                        start_vert_idx = face.verts_idx[(i_e2 + 1) % 3]
                        adj_edge_idx = self.func_get_edge_idx(i_b, adj_face_idx, start_vert_idx)

                        self.polytope_horizon_stack[i_b, top].face_idx = adj_face_idx
                        self.polytope_horizon_stack[i_b, top].edge_idx = adj_edge_idx
                        top += 1

            else:

                # If not visible, add the edge to the horizon.
                flag = self.func_add_edge_to_horizon(i_b, i_f, i_e)
                if flag:
                    # If the edges do not form a closed loop,
                    # there is an error in the algorithm.
                    break

            is_first = False

        return flag

    @ti.func
    def func_add_edge_to_horizon(self, i_b, i_f, i_e):
        horizon_nedges = self.polytope[i_b].horizon_nedges
        self.polytope_horizon_data[i_b, horizon_nedges].edge_idx = i_e
        self.polytope_horizon_data[i_b, horizon_nedges].face_idx = i_f
        self.polytope[i_b].horizon_nedges += 1

        flag = 0
        return flag

    @ti.func
    def func_get_edge_idx(self, i_b, i_f, i_v):
        """
        Get the edge index from the face, starting from the vertex i_v.
        If the face is comprised of [v1, v2, v3], the edges are:
        [v1, v2], [v2, v3], [v3, v1].

        Therefore, if i_v is v1, the edge index is 0,
        and if i_v is v2, the edge index is 1,
        and if i_v is v3, the edge index is 2.
        """
        verts = self.polytope_faces[i_b, i_f].verts_idx
        ret = 2
        if verts[0] == i_v:
            ret = 0
        elif verts[1] == i_v:
            ret = 1
        return ret

    @ti.func
    def func_delete_face_from_polytope(self, i_b, i_f):
        """
        Delete the face from the polytope.
        """
        face_map_idx = self.polytope_faces[i_b, i_f].map_idx
        if face_map_idx >= 0:
            last_face_idx = self.polytope_faces_map[i_b][self.polytope[i_b].nfaces_map - 1]
            # Make the map to point to the last face
            self.polytope_faces_map[i_b][face_map_idx] = last_face_idx
            # Change map index of the last face
            self.polytope_faces[i_b, last_face_idx].map_idx = face_map_idx

            # Decrease the number of faces in the polytope
            self.polytope[i_b].nfaces_map -= 1

        # Mark the face as deleted
        self.polytope_faces[i_b, i_f].map_idx = -2

    @ti.func
    def func_epa_insert_vertex_to_polytope(self, i_b, obj1_point, obj2_point, obj1_id, obj2_id, minkowski_point):
        """
        Copy vertex information into the polytope.
        """
        n = self.polytope[i_b].nverts
        self.polytope_verts[i_b, n].obj1 = obj1_point
        self.polytope_verts[i_b, n].obj2 = obj2_point
        self.polytope_verts[i_b, n].id1 = obj1_id
        self.polytope_verts[i_b, n].id2 = obj2_id
        self.polytope_verts[i_b, n].mink = minkowski_point
        self.polytope[i_b].nverts += 1
        return n

    @ti.func
    def func_epa_init_polytope_2d(self, i_ga, i_gb, i_b):
        """
        Create the polytope for EPA from a 1-simplex (line segment).
        Return 0 when successful.
        """
        flag = 0

        # Get the simplex vertices
        v1 = self.gjk_simplex_vertex[i_b, 0].mink
        v2 = self.gjk_simplex_vertex[i_b, 1].mink
        diff = v2 - v1

        # Find the element in [diff] with the smallest magnitude,
        # because it will give us the largest cross product below
        min_val = ti.abs(diff[0])
        min_i = 0
        for i in range(1, 3):
            if ti.abs(diff[i]) < min_val:
                min_val = ti.abs(diff[i])
                min_i = i

        # Cross product with the found axis,
        # then rotate it by 120 degrees around the axis [diff]
        # to get three more points spaced 120 degrees apart
        rotmat = self.func_rotmat_120(diff)
        e = gs.ti_vec3(0.0, 0.0, 0.0)
        e[min_i] = 1.0

        d = ti.math.mat3(0, 0, 0, 0, 0, 0, 0, 0, 0)
        d1 = e.cross(diff)
        d[0, 0], d[0, 1], d[0, 2] = d1[0], d1[1], d1[2]
        d2 = rotmat @ d1
        d[1, 0], d[1, 1], d[1, 2] = d2[0], d2[1], d2[2]
        d3 = rotmat @ d2
        d[2, 0], d[2, 1], d[2, 2] = d3[0], d3[1], d3[2]

        # Insert the first two vertices into the polytope
        vi = ti.Vector([0, 0, 0, 0, 0], dt=ti.i32)
        for i in range(2):
            vi[i] = self.func_epa_insert_vertex_to_polytope(
                i_b,
                self.gjk_simplex_vertex[i_b, i].obj1,
                self.gjk_simplex_vertex[i_b, i].obj2,
                self.gjk_simplex_vertex[i_b, i].id1,
                self.gjk_simplex_vertex[i_b, i].id2,
                self.gjk_simplex_vertex[i_b, i].mink,
            )

        # Find three more vertices using [d1, d2, d3] as support vectors,
        # and insert them into the polytope
        for i in range(3):
            di = gs.ti_vec3(d[i, 0], d[i, 1], d[i, 2])
            di_norm = ti.math.length(di)
            vi[i + 2] = self.func_epa_support(i_ga, i_gb, i_b, di, di_norm)

        v3 = self.polytope_verts[i_b, vi[2]].mink
        v4 = self.polytope_verts[i_b, vi[3]].mink
        v5 = self.polytope_verts[i_b, vi[4]].mink

        # Build hexahedron (6 faces) from the five vertices.
        # * This hexahedron would have line [v1, v2] as the central axis,
        # and the other three vertices would be on the sides of the hexahedron,
        # as they are spaced 120 degrees apart.
        # * We already know the face and adjacent face indices in building this.
        # * While building the hexahedron by attaching faces, if the face is very
        # close to the origin, we replace the 1-simplex with the 2-simplex,
        # and restart from it.

        # Vertex indices for the faces in the hexahedron
        i_vs = ti.Matrix(
            [
                [vi[0], vi[2], vi[3]],
                [vi[0], vi[4], vi[2]],
                [vi[0], vi[3], vi[4]],
                [vi[1], vi[3], vi[2]],
                [vi[1], vi[2], vi[4]],
                [vi[1], vi[4], vi[3]],
            ],
            dt=ti.i32,
        )

        # Adjacent face indices for the faces in the hexahedron
        i_as = ti.Matrix([[1, 3, 2], [2, 4, 0], [0, 5, 1], [5, 0, 4], [3, 1, 5], [4, 2, 3]], dt=ti.i32)

        for i in range(6):
            i_v1, i_v2, i_v3 = i_vs[i, 0], i_vs[i, 1], i_vs[i, 2]
            i_a1, i_a2, i_a3 = i_as[i, 0], i_as[i, 1], i_as[i, 2]

            if self.func_attach_face_to_polytope(i_b, i_v1, i_v2, i_v3, i_a1, i_a2, i_a3) < self.FLOAT_MIN_SQ:
                self.func_replace_simplex_3(i_b, i_v1, i_v2, i_v3)
                flag = EPA_P2_FALLBACK3  # self.func_epa_init_polytope_3d(i_ga, i_gb, i_b)
                break

        if flag == 0:
            if not self.func_ray_triangle_intersection(v1, v2, v3, v4, v5):
                # The hexahedron should be convex by definition,
                # but somehow if it is not, we return non-convex flag
                flag = EPA_P2_NONCONVEX

        if flag == 0:
            # Initialize face map
            for i in range(6):
                self.polytope_faces_map[i_b][i] = i
                self.polytope_faces[i_b, i].map_idx = i
            self.polytope[i_b].nfaces_map = 6

        return flag

    @ti.func
    def func_epa_init_polytope_3d(self, i_ga, i_gb, i_b):
        """
        Create the polytope for EPA from a 2-simplex (triangle).
        Return 0 when successful.
        """
        flag = 0

        # Get the simplex vertices
        v1 = self.gjk_simplex_vertex[i_b, 0].mink
        v2 = self.gjk_simplex_vertex[i_b, 1].mink
        v3 = self.gjk_simplex_vertex[i_b, 2].mink

        # Get normal; if it is zero, we cannot proceed
        edge1 = v2 - v1
        edge2 = v3 - v1
        n = edge1.cross(edge2)
        n_norm = ti.math.length(n)
        if n_norm < self.FLOAT_MIN:
            flag = EPA_P3_BAD_NORMAL
        n_neg = -n

        # Save vertices in the polytope
        vi = ti.Vector([0, 0, 0, 0, 0], dt=ti.i32)
        for i in range(3):
            vi[i] = self.func_epa_insert_vertex_to_polytope(
                i_b,
                self.gjk_simplex_vertex[i_b, i].obj1,
                self.gjk_simplex_vertex[i_b, i].obj2,
                self.gjk_simplex_vertex[i_b, i].id1,
                self.gjk_simplex_vertex[i_b, i].id2,
                self.gjk_simplex_vertex[i_b, i].mink,
            )

        # Find the fourth and fifth vertices using the normal
        # as the support vector. We form a hexahedron (6 faces)
        # with these five vertices.
        for i in range(2):
            dir = n if i == 0 else n_neg
            vi[i + 3] = self.func_epa_support(i_ga, i_gb, i_b, dir, n_norm)
        v4 = self.polytope_verts[i_b, vi[3]].mink
        v5 = self.polytope_verts[i_b, vi[4]].mink

        # Check if v4 or v5 located inside the triangle.
        # If so, we do not proceed anymore.
        for i in range(2):
            v = v4 if i == 0 else v5
            if self.func_point_triangle_intersection(v, v1, v2, v3):
                flag = EPA_P3_INVALID_V4 if i == 0 else EPA_P3_INVALID_V5
                break

        if flag == 0:
            # If origin does not lie inside the triangle, we need to
            # check if the hexahedron contains the origin.

            tets_has_origin = gs.ti_ivec2(0, 0)
            for i in range(2):
                v = v4 if i == 0 else v5
                tets_has_origin[i] = self.func_origin_tetra_intersection(v1, v2, v3, v)

            # @TODO: It's possible for GJK to return a triangle with
            # origin not contained in it but within tolerance from it.
            # In that case, the hexahedron could possibly be constructed
            # that does ont contain the origin, but there is penetration depth.
            if (
                self.gjk_simplex[i_b].dist > 10 * self.FLOAT_MIN
                and (not tets_has_origin[0])
                and (not tets_has_origin[1])
            ):
                flag = EPA_P3_MISSING_ORIGIN
            else:
                # Vertex indices for the faces in the hexahedron
                i_vs = ti.Matrix(
                    [
                        [vi[3], vi[0], vi[1]],
                        [vi[3], vi[2], vi[0]],
                        [vi[3], vi[1], vi[2]],
                        [vi[4], vi[1], vi[0]],
                        [vi[4], vi[0], vi[2]],
                        [vi[4], vi[2], vi[1]],
                    ],
                    dt=ti.i32,
                )

                # Adjacent face indices for the faces in the hexahedron
                i_as = ti.Matrix([[1, 3, 2], [2, 4, 0], [0, 5, 1], [5, 0, 4], [3, 1, 5], [4, 2, 3]], dt=ti.i32)

                # Build hexahedron (6 faces) from the five vertices.
                for i in range(6):
                    i_v1, i_v2, i_v3 = i_vs[i, 0], i_vs[i, 1], i_vs[i, 2]
                    i_a1, i_a2, i_a3 = i_as[i, 0], i_as[i, 1], i_as[i, 2]

                    dist2 = self.func_attach_face_to_polytope(i_b, i_v1, i_v2, i_v3, i_a1, i_a2, i_a3)
                    if dist2 < self.FLOAT_MIN_SQ:
                        flag = EPA_P3_ORIGIN_ON_FACE
                        break

        if flag == 0:
            # Initialize face map
            for i in range(6):
                self.polytope_faces_map[i_b][i] = i
                self.polytope_faces[i_b, i].map_idx = i
            self.polytope[i_b].nfaces_map = 6

        return flag

    @ti.func
    def func_epa_init_polytope_4d(self, i_ga, i_gb, i_b):
        """
        Create the polytope for EPA from a 3-simplex (tetrahedron).
        Return 0 when successful.
        """
        flag = 0

        # Insert simplex vertices into the polytope
        vi = ti.Vector([0, 0, 0, 0], dt=ti.i32)
        for i in range(4):
            vi[i] = self.func_epa_insert_vertex_to_polytope(
                i_b,
                self.gjk_simplex_vertex[i_b, i].obj1,
                self.gjk_simplex_vertex[i_b, i].obj2,
                self.gjk_simplex_vertex[i_b, i].id1,
                self.gjk_simplex_vertex[i_b, i].id2,
                self.gjk_simplex_vertex[i_b, i].mink,
            )

        # If origin is on any face of the tetrahedron,
        # replace the simplex with a 2-simplex (triangle)

        # Vertex indices for the faces in the hexahedron
        i_vs = ti.Matrix(
            [[vi[0], vi[1], vi[2]], [vi[0], vi[3], vi[1]], [vi[0], vi[2], vi[3]], [vi[3], vi[2], vi[1]]], dt=ti.i32
        )

        # Adjacent face indices for the faces in the hexahedron
        i_as = ti.Matrix([[1, 3, 2], [2, 3, 0], [0, 3, 1], [2, 0, 1]], dt=ti.i32)

        for i in range(4):
            v1, v2, v3 = i_vs[i, 0], i_vs[i, 1], i_vs[i, 2]
            a1, a2, a3 = i_as[i, 0], i_as[i, 1], i_as[i, 2]

            dist2 = self.func_attach_face_to_polytope(i_b, v1, v2, v3, a1, a2, a3)

            if dist2 < self.FLOAT_MIN_SQ:
                self.func_replace_simplex_3(i_b, v1, v2, v3)
                flag = EPA_P4_FALLBACK3  # self.func_epa_init_polytope_3d(i_ga, i_gb, i_b)
                break

        if flag == 0:
            # If the tetrahedron does not contain the origin,
            # we do not proceed anymore.
            v1, v2, v3, v4 = vi[0], vi[1], vi[2], vi[3]
            if not self.func_origin_tetra_intersection(
                self.polytope_verts[i_b, v1].mink,
                self.polytope_verts[i_b, v2].mink,
                self.polytope_verts[i_b, v3].mink,
                self.polytope_verts[i_b, v4].mink,
            ):
                flag = EPA_P4_MISSING_ORIGIN

        if flag == 0:
            # Initialize face map
            for i in range(4):
                self.polytope_faces_map[i_b][i] = i
                self.polytope_faces[i_b, i].map_idx = i
            self.polytope[i_b].nfaces_map = 4

        return flag

    @ti.func
    def func_epa_support(self, i_ga, i_gb, i_b, dir, dir_norm):
        """
        Find support points on the two objects using [dir].
        [dir] should be a unit vector from [ga] (obj1) to [gb] (obj2).
        After finding them, insert them into the polytope.
        """
        d = gs.ti_vec3(1, 0, 0)
        if dir_norm > self.FLOAT_MIN:
            d = dir / dir_norm

        (
            support_point_obj1,
            support_point_obj2,
            support_point_id_obj1,
            support_point_id_obj2,
            support_point_minkowski,
        ) = self.func_support(i_ga, i_gb, i_b, d, False)

        # Insert the support points into the polytope
        v_index = self.func_epa_insert_vertex_to_polytope(
            i_b,
            support_point_obj1,
            support_point_obj2,
            support_point_id_obj1,
            support_point_id_obj2,
            support_point_minkowski,
        )

        return v_index

    @ti.func
    def func_attach_face_to_polytope(self, i_b, i_v1, i_v2, i_v3, i_a1, i_a2, i_a3):
        """
        Attach a face to the polytope.
        [i_v1, i_v2, i_v3] are the vertices of the face,
        [i_a1, i_a2, i_a3] are the adjacent faces.

        Also return the squared distance of the face to the origin.
        """
        flag = 0.0

        n = self.polytope[i_b].nfaces
        self.polytope_faces[i_b, n].verts_idx[0] = i_v1
        self.polytope_faces[i_b, n].verts_idx[1] = i_v2
        self.polytope_faces[i_b, n].verts_idx[2] = i_v3
        self.polytope_faces[i_b, n].adj_idx[0] = i_a1
        self.polytope_faces[i_b, n].adj_idx[1] = i_a2
        self.polytope_faces[i_b, n].adj_idx[2] = i_a3
        self.polytope[i_b].nfaces += 1

        # Compute the squared distance of the face to the origin
        self.polytope_faces[i_b, n].normal, ret = self.func_project_origin_to_plane(
            self.polytope_verts[i_b, i_v3].mink,
            self.polytope_verts[i_b, i_v2].mink,
            self.polytope_verts[i_b, i_v1].mink,
        )
        if not ret:
            normal = self.polytope_faces[i_b, n].normal
            self.polytope_faces[i_b, n].dist2 = normal.dot(normal)
            self.polytope_faces[i_b, n].map_idx = -1  # No map index yet
            flag = self.polytope_faces[i_b, n].dist2

        return flag

    @ti.func
    def func_replace_simplex_3(self, i_b, i_v1, i_v2, i_v3):
        """
        Replace the simplex with a 2-simplex (triangle) from polytope vertices.
        [i_v1, i_v2, i_v3] are the vertices that we will use from the polytope.
        """
        self.gjk_simplex[i_b].nverts = 3
        self.gjk_simplex_vertex[i_b, 0] = self.polytope_verts[i_b, i_v1]
        self.gjk_simplex_vertex[i_b, 1] = self.polytope_verts[i_b, i_v2]
        self.gjk_simplex_vertex[i_b, 2] = self.polytope_verts[i_b, i_v3]

        # Reset polytope
        self.polytope[i_b].nverts = 0
        self.polytope[i_b].nfaces = 0
        self.polytope[i_b].nfaces_map = 0

    @ti.func
    def func_ray_triangle_intersection(self, ray_v1, ray_v2, tri_v1, tri_v2, tri_v3):
        """
        Check if the ray intersects the triangle.
        Return Non-Zero value if it does, otherwise return Zero.
        """
        flag = 0

        ray = ray_v2 - ray_v1
        tri_ray_1 = tri_v1 - ray_v1
        tri_ray_2 = tri_v2 - ray_v1
        tri_ray_3 = tri_v3 - ray_v1

        # Signed volumes of the tetrahedrons formed by the ray and triangle edges
        vols = gs.ti_vec3(0.0, 0.0, 0.0)
        for i in range(3):
            v1, v2 = tri_ray_1, tri_ray_2
            if i == 1:
                v1, v2 = tri_ray_2, tri_ray_3
            elif i == 2:
                v1, v2 = tri_ray_3, tri_ray_1
            vols[i] = self.func_det3(v1, v2, ray)

        vol_1, vol_2, vol_3 = vols[0], vols[1], vols[2]

        if vol_1 >= 0 and vol_2 >= 0 and vol_3 >= 0:
            flag = 1
        elif vol_1 <= 0 and vol_2 <= 0 and vol_3 <= 0:
            flag = -1
        else:
            flag = 0

        return flag

    @ti.func
    def func_point_triangle_intersection(self, point, tri_v1, tri_v2, tri_v3):
        """
        Check if the point is inside the triangle.
        """
        flag = 0
        # Compute the affine coordinates of the point with respect to the triangle
        _lambda = self.func_triangle_affine_coords(point, tri_v1, tri_v2, tri_v3)

        # If any of the affine coordinates is negative,
        # the point is outside the triangle
        if _lambda[0] < 0 or _lambda[1] < 0 or _lambda[2] < 0:
            flag = 0
        else:
            # Check if the point predicted by the affine coordinates
            # is equal to the point itself
            pred = tri_v1 * _lambda[0] + tri_v2 * _lambda[1] + tri_v3 * _lambda[2]
            diff = pred - point
            flag = 1 if diff.dot(diff) < self.FLOAT_MIN_SQ else 0

        return flag

    @ti.func
    def func_triangle_affine_coords(self, point, tri_v1, tri_v2, tri_v3):
        """
        Compute the affine coordinates of the point with respect to the triangle.
        """
        # Compute minors of the triangle vertices
        m_1 = (
            tri_v2[1] * tri_v3[2]
            - tri_v2[2] * tri_v3[1]
            - tri_v1[1] * tri_v3[2]
            + tri_v1[2] * tri_v3[1]
            + tri_v1[1] * tri_v2[2]
            - tri_v1[2] * tri_v2[1]
        )

        m_2 = (
            tri_v2[0] * tri_v3[2]
            - tri_v2[2] * tri_v3[0]
            - tri_v1[0] * tri_v3[2]
            + tri_v1[2] * tri_v3[0]
            + tri_v1[0] * tri_v2[2]
            - tri_v1[2] * tri_v2[0]
        )

        m_3 = (
            tri_v2[0] * tri_v3[1]
            - tri_v2[1] * tri_v3[0]
            - tri_v1[0] * tri_v3[1]
            + tri_v1[1] * tri_v3[0]
            + tri_v1[0] * tri_v2[1]
            - tri_v1[1] * tri_v2[0]
        )

        # Exclude one of the axes with the largest projection of the triangle
        # using the minors of the above linear system.
        m_max = 0.0
        absm1, absm2, absm3 = ti.abs(m_1), ti.abs(m_2), ti.abs(m_3)
        x, y = 0, 0
        if absm1 >= absm2 and absm1 >= absm3:
            # Remove first row
            m_max = m_1
            x = 1
            y = 2
        elif absm2 >= absm1 and absm2 >= absm3:
            # Remove second row
            m_max = m_2
            x = 0
            y = 2
        else:
            # Remove third row
            m_max = m_3
            x = 0
            y = 1

        # C1 corresponds to the signed area of 2-simplex (triangle): (point, tri_v2, tri_v3)
        C1 = (
            point[x] * tri_v2[y]
            + point[y] * tri_v3[x]
            + tri_v2[x] * tri_v3[y]
            - point[x] * tri_v3[y]
            - point[y] * tri_v2[x]
            - tri_v3[x] * tri_v2[y]
        )

        # C2 corresponds to the signed area of 2-simplex (triangle): (point, tri_v1, tri_v3)
        C2 = (
            point[x] * tri_v3[y]
            + point[y] * tri_v1[x]
            + tri_v3[x] * tri_v1[y]
            - point[x] * tri_v1[y]
            - point[y] * tri_v3[x]
            - tri_v1[x] * tri_v3[y]
        )

        # C3 corresponds to the signed area of 2-simplex (triangle): (point, tri_v1, tri_v2)
        C3 = (
            point[x] * tri_v1[y]
            + point[y] * tri_v2[x]
            + tri_v1[x] * tri_v2[y]
            - point[x] * tri_v2[y]
            - point[y] * tri_v1[x]
            - tri_v2[x] * tri_v1[y]
        )

        # Affine coordinates are computed as:
        # [ l1, l2, l3 ] = [ C1 / m_max, C2 / m_max, C3 / m_max ]
        _lambda = gs.ti_vec3(0, 0, 0)
        _lambda[0] = C1 / m_max
        _lambda[1] = C2 / m_max
        _lambda[2] = C3 / m_max

        return _lambda

    @ti.func
    def func_origin_tetra_intersection(self, tet_v1, tet_v2, tet_v3, tet_v4):
        """
        Check if the origin is inside the tetrahedron.
        """
        flag = 1
        for i in range(4):
            v1, v2, v3, v4 = tet_v1, tet_v2, tet_v3, tet_v4
            if i == 1:
                v1, v2, v3, v4 = tet_v2, tet_v3, tet_v4, tet_v1
            elif i == 2:
                v1, v2, v3, v4 = tet_v3, tet_v4, tet_v1, tet_v2
            elif i == 3:
                v1, v2, v3, v4 = tet_v4, tet_v1, tet_v2, tet_v3
            flag = flag and self.func_point_plane_same_side(v1, v2, v3, v4)
            if not flag:
                break
        return flag

    @ti.func
    def func_point_plane_same_side(self, point, plane_v1, plane_v2, plane_v3):
        """
        Check if the point is on the same side of the plane as the origin.
        """
        # Compute the normal of the plane
        edge1 = plane_v2 - plane_v1
        edge2 = plane_v3 - plane_v1
        normal = edge1.cross(edge2)

        diff1 = point - plane_v1
        dot1 = normal.dot(diff1)

        diff2 = -plane_v1  # origin - plane_v1
        dot2 = normal.dot(diff2)

        flag = 1 if dot1 * dot2 > 0 else 0
        return flag

    """
    Multi-contact algorithms
    """

    @ti.func
    def func_multi_contact(self, i_ga, i_gb, i_b, i_f):
        # Get vertices of the nearest face from EPA
        v11i = self.polytope_verts[i_b, self.polytope_faces[i_b, i_f].verts_idx[0]].id1
        v12i = self.polytope_verts[i_b, self.polytope_faces[i_b, i_f].verts_idx[1]].id1
        v13i = self.polytope_verts[i_b, self.polytope_faces[i_b, i_f].verts_idx[2]].id1
        v21i = self.polytope_verts[i_b, self.polytope_faces[i_b, i_f].verts_idx[0]].id2
        v22i = self.polytope_verts[i_b, self.polytope_faces[i_b, i_f].verts_idx[1]].id2
        v23i = self.polytope_verts[i_b, self.polytope_faces[i_b, i_f].verts_idx[2]].id2
        v11 = self.polytope_verts[i_b, self.polytope_faces[i_b, i_f].verts_idx[0]].obj1
        v12 = self.polytope_verts[i_b, self.polytope_faces[i_b, i_f].verts_idx[1]].obj1
        v13 = self.polytope_verts[i_b, self.polytope_faces[i_b, i_f].verts_idx[2]].obj1
        v21 = self.polytope_verts[i_b, self.polytope_faces[i_b, i_f].verts_idx[0]].obj2
        v22 = self.polytope_verts[i_b, self.polytope_faces[i_b, i_f].verts_idx[1]].obj2
        v23 = self.polytope_verts[i_b, self.polytope_faces[i_b, i_f].verts_idx[2]].obj2

        # Get the simplex dimension of geom 1 and 2
        nface1, nface2 = 0, 0
        for i in range(2):
            v1i, v2i, v3i, v1, v2, v3 = v11i, v12i, v13i, v11, v12, v13
            if i == 1:
                v1i, v2i, v3i, v1, v2, v3 = v21i, v22i, v23i, v21, v22, v23

            nface, v1i, v2i, v3i, v1, v2, v3 = self.func_simplex_dim(v1i, v2i, v3i, v1, v2, v3)
            if i == 0:
                nface1, v11i, v12i, v13i, v11, v12, v13 = nface, v1i, v2i, v3i, v1, v2, v3
            else:
                nface2, v21i, v22i, v23i, v21, v22, v23 = nface, v1i, v2i, v3i, v1, v2, v3
        dir = self.witness[i_b, 0].point_obj2 - self.witness[i_b, 0].point_obj1
        dir_neg = self.witness[i_b, 0].point_obj1 - self.witness[i_b, 0].point_obj2

        # Get all possible face normals for each geom
        nnorms1, nnorms2 = 0, 0
        geom_type_a = self._solver.geoms_info[i_ga].type
        geom_type_b = self._solver.geoms_info[i_gb].type

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
                nnorms = self.func_potential_box_normals(i_g, i_b, nface, v1i, v2i, v3i, t_dir)
            elif geom_type == gs.GEOM_TYPE.MESH:
                nnorms = self.func_potential_mesh_normals(i_g, i_b, nface, v1i, v2i, v3i)

            for i_n in range(nnorms):
                if i_g0 == 0:
                    self.multi_contact_faces[i_b, i_n].normal1 = self.multi_contact_normals[i_b, i_n].normal
                    self.multi_contact_faces[i_b, i_n].id1 = self.multi_contact_normals[i_b, i_n].id
                    nnorms1 = nnorms
                else:
                    self.multi_contact_faces[i_b, i_n].normal2 = self.multi_contact_normals[i_b, i_n].normal
                    self.multi_contact_faces[i_b, i_n].id2 = self.multi_contact_normals[i_b, i_n].id
                    nnorms2 = nnorms

        # Determine if any two face normals match
        aligned_faces_idx, aligned_faces_flag = self.func_find_aligned_faces(i_b, nnorms1, nnorms2)
        no_multiple_contacts = False
        edgecon1, edgecon2 = False, False

        if not aligned_faces_flag:

            # No aligned faces found; check if there was edge-face collision
            # [is_edge_face]: geom1 is edge, geom2 is face
            # [is_face_edge]: geom1 is face, geom2 is edge
            is_edge_face = (nface1 < 3) and (nface1 <= nface2)
            is_face_edge = (not is_edge_face) and nface2 < 3

            if is_edge_face or is_face_edge:
                if is_edge_face:
                    nnorms1 = 0
                else:
                    nnorms2 = 0

                i_g = i_ga if is_edge_face else i_gb
                geom_type = geom_type_a if is_edge_face else geom_type_b
                nface = nface1 if is_edge_face else nface2
                v1 = v11 if is_edge_face else v21
                v2 = v12 if is_edge_face else v22
                v1i = v11i if is_edge_face else v21i
                v2i = v12i if is_edge_face else v22i

                nnorms = 0
                if geom_type == gs.GEOM_TYPE.BOX:
                    nnorms = self.func_potential_box_edge_normals(i_g, i_b, nface, v1, v2, v1i, v2i)
                elif geom_type == gs.GEOM_TYPE.MESH:
                    nnorms = self.func_potential_mesh_edge_normals(i_g, i_b, nface, v1, v2, v1i, v2i)

                if is_edge_face:
                    nnorms1 = nnorms
                else:
                    nnorms2 = nnorms

                if nnorms > 0:
                    for i_n in range(nnorms):
                        if is_edge_face:
                            self.multi_contact_faces[i_b, i_n].normal1 = self.multi_contact_normals[i_b, i_n].normal
                        else:
                            self.multi_contact_faces[i_b, i_n].normal2 = self.multi_contact_normals[i_b, i_n].normal

                        self.multi_contact_faces[i_b, i_n].endverts = self.multi_contact_normals[i_b, i_n].endverts

                # Check if any of the edge normals match
                nedges, nfaces = nnorms1, nnorms2
                if not is_edge_face:
                    nedges, nfaces = nfaces, nedges
                aligned_faces_idx, aligned_edge_face_flag = self.func_find_aligned_edge_face(
                    i_b, nedges, nfaces, is_edge_face
                )

                if not aligned_edge_face_flag:
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
            i = aligned_faces_idx[0]
            j = aligned_faces_idx[1]

            # Recover matching edge or face from geoms
            for k in range(2):
                edgecon = edgecon1 if k == 0 else edgecon2
                geom_type = geom_type_a if k == 0 else geom_type_b
                i_g = i_ga if k == 0 else i_gb

                nface = 0
                if edgecon:
                    if k == 0:
                        self.multi_contact_faces[i_b, 0].face1 = self.polytope_verts[
                            i_b, self.polytope_faces[i_b, i_f].verts_idx[0]
                        ].obj1
                        self.multi_contact_faces[i_b, 1].face1 = self.multi_contact_faces[i_b, i].endverts
                    else:
                        self.multi_contact_faces[i_b, 0].face2 = self.polytope_verts[
                            i_b, self.polytope_faces[i_b, i_f].verts_idx[0]
                        ].obj2
                        self.multi_contact_faces[i_b, 1].face2 = self.multi_contact_faces[i_b, j].endverts

                    nface = 2
                else:
                    normal_face_idx = self.multi_contact_faces[i_b, i].id1
                    if k == 0 and edgecon2:
                        # Since [i] is the edge idx, use [j]
                        normal_face_idx = self.multi_contact_faces[i_b, j].id1
                    elif k == 1:
                        normal_face_idx = self.multi_contact_faces[i_b, j].id2

                    if geom_type == gs.GEOM_TYPE.BOX:
                        nface = self.func_box_face(i_g, i_b, k, normal_face_idx)
                    elif geom_type == gs.GEOM_TYPE.MESH:
                        nface = self.func_mesh_face(i_g, i_b, k, normal_face_idx)

                if k == 0:
                    nface1 = nface
                else:
                    nface2 = nface

            approx_dir = gs.ti_vec3(0.0, 0.0, 0.0)
            normal = gs.ti_vec3(0.0, 0.0, 0.0)
            if edgecon1:
                # Face 1 is an edge, so clip face 1 against face 2
                approx_dir = self.multi_contact_faces[i_b, j].normal2 * ti.math.length(dir)
                normal = self.multi_contact_faces[i_b, j].normal2
            elif edgecon2:
                # Face 2 is an edge, so clip face 2 against face 1
                approx_dir = self.multi_contact_faces[i_b, j].normal1 * (-ti.math.length(dir))
                normal = self.multi_contact_faces[i_b, j].normal1
            else:
                # Face-face contact
                approx_dir = self.multi_contact_faces[i_b, j].normal2 * ti.math.length(dir)
                normal = self.multi_contact_faces[i_b, i].normal1

            # Clip polygon
            self.func_clip_polygon(i_b, nface1, nface2, edgecon1, edgecon2, normal, approx_dir)

    @ti.func
    def func_simplex_dim(self, v1i, v2i, v3i, v1, v2, v3):
        """
        Determine the dimension of the given simplex (1-3).
        If every point is the same, 1-dim.
        If two points are the same, 2-dim.
        If all points are different, 3-dim.
        """
        flag = 0
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
    def func_potential_box_normals(self, i_g, i_b, dim, v1, v2, v3, dir):
        """
        For a simplex defined on a box with three vertices [v1, v2, v3],
        we find which face normals are potentially related to the simplex.

        If the simplex is a triangle, at most one face normal is related.
        If the simplex is a line, at most two face normals are related.
        If the simplex is a point, at most three face normals are related.

        We identify which face normals are related to the simplex
        by checking the vertex indices of the simplex vertices.
        """
        g_state = self._solver.geoms_state[i_g, i_b]
        g_quat = g_state.quat

        # Change to local vertex indices
        v1 -= self._solver.geoms_info[i_g].vert_start
        v2 -= self._solver.geoms_info[i_g].vert_start
        v3 -= self._solver.geoms_info[i_g].vert_start

        # Number of potential face normals
        n_normals = 0

        # Fallback if the simplex is degenerate
        is_degenerate_simplex = False

        c = 0
        tmp = gs.ti_ivec3(0, 0, 0)
        for i in range(3):
            # 1 when every vertex has positive xyz coordinate,
            # -1 when every vertex has negative xyz coordinate,
            # 0 when vertices are mixed
            tmp[i] = self.func_cmp_bit(v1, v2, v3, dim, i)
        x, y, z = tmp[0], tmp[1], tmp[2]

        num_iter = 1 if dim == 3 else 3
        for i in range(num_iter):

            # Determine the normal vector in the local space
            local_n = gs.ti_vec3(x, y, z)
            w = 1

            if dim == 2:
                w = x
                if i == 1:
                    w = y
                elif i == 2:
                    w = z

            if dim == 2 or dim == 1:
                local_n = gs.ti_vec3(x, 0, 0)
                if i == 1:
                    local_n = gs.ti_vec3(0, y, 0)
                elif i == 2:
                    local_n = gs.ti_vec3(0, 0, z)

            global_n = gu.ti_transform_by_quat(local_n, g_quat)

            if dim == 3:
                self.multi_contact_normals[i_b, 0].normal = global_n

                # Note that only one of [x, y, z] could be non-zero,
                # because the triangle is on the box face.
                sgn = x + y + z
                if x:
                    self.multi_contact_normals[i_b, c].id = 0
                    c += 1
                if y:
                    self.multi_contact_normals[i_b, c].id = 2
                    c += 1
                if z:
                    self.multi_contact_normals[i_b, c].id = 4
                    c += 1

                if sgn == -1:
                    # Flip if needed
                    self.multi_contact_normals[i_b, 0].id = self.multi_contact_normals[i_b, 0].id + 1

            elif dim == 2:
                if w:
                    if (i == 0) or (i == 1):
                        self.multi_contact_normals[i_b, c].normal = global_n
                    else:
                        self.multi_contact_normals[i_b, 1].normal = global_n

                    if i == 0:
                        self.multi_contact_normals[i_b, c].id = 0 if x > 0 else 1
                    elif i == 1:
                        self.multi_contact_normals[i_b, c].id = 2 if y > 0 else 3
                    elif i == 2:
                        self.multi_contact_normals[i_b, c].id = 4 if z > 0 else 5
                    c += 1

            elif dim == 1:
                self.multi_contact_normals[i_b, c].normal = global_n
                if i == 0:
                    self.multi_contact_normals[i_b, c].id = 0 if x > 0 else 1
                elif i == 1:
                    self.multi_contact_normals[i_b, c].id = 2 if y > 0 else 3
                elif i == 2:
                    self.multi_contact_normals[i_b, c].id = 4 if z > 0 else 5
                c += 1

        # Check [c] for detecting degenerate cases
        if dim == 3:
            # [c] should be 1 in normal case,
            # but if triangle does not lie on the box face,
            # it could be other values.
            n_normals = 1
            is_degenerate_simplex = c != 1
        elif dim == 2:
            # [c] should be 2 in normal case,
            # but if edge does not lie on the box edge,
            # it could be other values.
            n_normals = 2
            is_degenerate_simplex = c != 2
        elif dim == 1:
            n_normals = 3
            is_degenerate_simplex = False

        # If the simplex was degenerate, find the face normal using collision normal
        if is_degenerate_simplex:
            n_normals = self.func_box_normal_from_collision_normal(i_g, i_b, dir)

        return n_normals

    @ti.func
    def func_cmp_bit(self, v1, v2, v3, n, shift):
        """
        Compare one bit of v1 and v2 that sits at position `shift`
        (shift = 0 for the LSB, 1 for the next bit, …).

        Returns:
            +1  if both bits are 1
            0  if bits differ
            -1  if both bits are 0
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
    def func_box_normal_from_collision_normal(self, i_g, i_b, dir):
        """
        Among the 6 faces of the box, find the one of which normal is closest to the [dir].
        """
        # Every box face normal
        normals = ti.Vector(
            [1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0],
            dt=gs.ti_float,
        )

        # Get local collision normal
        g_state = self._solver.geoms_state[i_g, i_b]
        g_quat = g_state.quat
        local_dir = gu.ti_transform_by_quat(dir, gu.ti_inv_quat(g_quat))
        local_dir_norm = ti.math.length(local_dir)
        local_dir = local_dir * (1.0 / local_dir_norm)

        # Determine the closest face normal
        flag = 0
        for i in range(6):
            n = gs.ti_vec3(normals[3 * i + 0], normals[3 * i + 1], normals[3 * i + 2])
            if local_dir.dot(n) > self.face_tol:
                flag = 1
                self.multi_contact_normals[i_b, 0].normal = n
                self.multi_contact_normals[i_b, 0].id = i
                break

        return flag

    @ti.func
    def func_potential_mesh_normals(self, i_g, i_b, dim, v1, v2, v3):
        """
        For a simplex defined on a mesh with three vertices [v1, v2, v3],
        we find which face normals are potentially related to the simplex.

        If the simplex is a triangle, at most one face normal is related.
        If the simplex is a line, at most two face normals are related.
        If the simplex is a point, multiple faces that are adjacent to the point
        could be related.

        We identify which face normals are related to the simplex
        by checking the vertex indices of the simplex vertices.
        """
        # Get the geometry state and quaternion
        g_state = self._solver.geoms_state[i_g, i_b]
        g_quat = g_state.quat

        # Number of potential face normals
        n_normals = 0

        # Exhaustive search for the face normals
        # @TODO: This would require a lot of cost if the mesh is large.
        # It would be better to precompute adjacency information in the solver.
        face_start = self._solver.geoms_info[i_g].face_start
        face_end = self._solver.geoms_info[i_g].face_end

        for i_f in range(face_start, face_end):
            face = self._solver.faces_info[i_f].verts_idx
            has_v1, has_v2, has_v3 = False, False, False
            if v1 == face[0] or v1 == face[1] or v1 == face[2]:
                has_v1 = True
            if v2 == face[0] or v2 == face[1] or v2 == face[2]:
                has_v2 = True
            if v3 == face[0] or v3 == face[1] or v3 == face[2]:
                has_v3 = True

            compute_normal = False
            if dim == 3 and (has_v1 and has_v2 and has_v3):
                compute_normal = True
            elif dim == 2 and (has_v1 and has_v2):
                compute_normal = True
            elif dim == 1 and (has_v1):
                compute_normal = True

            if compute_normal:
                v1pos = self._solver.verts_info[face[0]].init_pos
                v2pos = self._solver.verts_info[face[1]].init_pos
                v3pos = self._solver.verts_info[face[2]].init_pos

                # Compute the face normal
                n = (v2pos - v1pos).cross(v3pos - v1pos)
                n_norm = ti.math.length(n)
                n = n * (1.0 / n_norm)
                n = gu.ti_transform_by_quat(n, g_quat)

                if dim == 3:
                    self.multi_contact_normals[i_b, 0].normal = n
                    self.multi_contact_normals[i_b, 0].id = i_f
                    n_normals = 1
                    break
                elif dim == 2:
                    self.multi_contact_normals[i_b, n_normals].normal = n
                    self.multi_contact_normals[i_b, n_normals].id = i_f
                    n_normals += 1
                    if n_normals == 2:
                        break
                else:
                    self.multi_contact_normals[i_b, n_normals].normal = n
                    self.multi_contact_normals[i_b, n_normals].id = i_f
                    n_normals += 1
                    if n_normals == self.max_polygon_verts:
                        break

        return n_normals

    @ti.func
    def func_find_aligned_faces(self, i_b, nv, nw):
        """
        Find if any two faces from [multi_contact_faces] are aligned.
        """
        res = gs.ti_ivec2(0, 0)
        flag = 0

        for i in range(nv):
            ni = self.multi_contact_faces[i_b, i].normal1
            for j in range(nw):
                nj = self.multi_contact_faces[i_b, j].normal2
                if ni.dot(nj) < -self.face_tol:
                    res[0] = i
                    res[1] = j
                    flag = 1
                    break
            if flag:
                break

        return res, flag

    @ti.func
    def func_potential_box_edge_normals(self, i_g, i_b, dim, v1, v2, v1i, v2i):
        """
        For a simplex defined on a box with two vertices [v1, v2],
        we find which edge normals are potentially related to the simplex.

        If the simplex is a line, at most one edge normal are related.
        If the simplex is a point, at most three edge normals are related.

        We identify which edge normals are related to the simplex
        by checking the vertex indices of the simplex vertices.
        """
        # Get the geometry state and quaternion
        g_state = self._solver.geoms_state[i_g, i_b]
        g_pos = g_state.pos
        g_quat = g_state.quat
        g_size_x = self._solver.geoms_info[i_g].data[0] * 0.5
        g_size_y = self._solver.geoms_info[i_g].data[1] * 0.5
        g_size_z = self._solver.geoms_info[i_g].data[2] * 0.5

        v1i -= self._solver.geoms_info[i_g].vert_start
        v2i -= self._solver.geoms_info[i_g].vert_start

        n_normals = 0

        if dim == 2:
            # If the nearest face is an edge
            self.multi_contact_normals[i_b, 0].endverts = v2
            self.multi_contact_normals[i_b, 0].normal = self.func_safe_normalize(v2 - v1)

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
                r = self.func_safe_normalize(ev - v1)

                self.multi_contact_normals[i_b, i].endverts = ev
                self.multi_contact_normals[i_b, i].normal = r

            n_normals = 3

        return n_normals

    @ti.func
    def func_potential_mesh_edge_normals(self, i_g, i_b, dim, v1, v2, v1i, v2i):
        """
        For a simplex defined on a mesh with two vertices [v1, v2],
        we find which edge normals are potentially related to the simplex.

        If the simplex is a line, at most one edge normal are related.
        If the simplex is a point, multiple edges that are adjacent to the point
        could be related.

        We identify which edge normals are related to the simplex
        by checking the vertex indices of the simplex vertices.
        """
        # Get the geometry state and quaternion
        g_state = self._solver.geoms_state[i_g, i_b]
        g_pos = g_state.pos
        g_quat = g_state.quat

        # Number of potential face normals
        n_normals = 0

        if dim == 2:
            # If the nearest face is an edge
            self.multi_contact_normals[i_b, 0].endverts = v2
            self.multi_contact_normals[i_b, 0].normal = self.func_safe_normalize(v2 - v1)

            n_normals = 1

        elif dim == 1:
            # If the nearest face is a point, consider every adjacent edge
            # Exhaustive search for the edge normals
            face_start = self._solver.geoms_info[i_g].face_start
            face_end = self._solver.geoms_info[i_g].face_end
            for i_f in range(face_start, face_end):
                face = self._solver.faces_info[i_f].verts_idx

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
                    v2_pos = self._solver.verts_info[t_v2i].init_pos
                    v2_pos = gu.ti_transform_by_trans_quat(v2_pos, g_pos, g_quat)
                    t_res = self.func_safe_normalize(v2_pos - v1)

                    self.multi_contact_normals[i_b, n_normals].normal = t_res
                    self.multi_contact_normals[i_b, n_normals].endverts = v2_pos

                    n_normals += 1
                    if n_normals == self.max_polygon_verts:
                        break

        return n_normals

    @ti.func
    def func_safe_normalize(self, v):
        """
        Normalize the vector [v] safely.
        """
        norm = ti.math.length(v)

        if norm < self.FLOAT_MIN:
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
    def func_find_aligned_edge_face(self, i_b, nedge, nface, is_edge_face):
        """
        Find if an edge and face from [multi_contact_faces] are aligned.
        """
        res = gs.ti_ivec2(0, 0)
        flag = 0

        for i in range(nedge):
            ni = self.multi_contact_faces[i_b, i].normal1
            if not is_edge_face:
                # The second normal is the edge normal
                ni = self.multi_contact_faces[i_b, i].normal2

            for j in range(nface):
                nj = self.multi_contact_faces[i_b, j].normal2
                if not is_edge_face:
                    # The first normal is the face normal
                    nj = self.multi_contact_faces[i_b, j].normal1

                if ti.abs(ni.dot(nj)) < self.edge_tol:
                    res[0] = i
                    res[1] = j
                    flag = 1
                    break
            if flag:
                break

        return res, flag

    @ti.func
    def func_box_face(self, i_g, i_b, i_o, face_idx):
        """
        Get the face vertices of the box geometry.
        """
        g_size_x = self._solver.geoms_info[i_g].data[0]
        g_size_y = self._solver.geoms_info[i_g].data[1]
        g_size_z = self._solver.geoms_info[i_g].data[2]

        # Get geometry position and quaternion
        g_state = self._solver.geoms_state[i_g, i_b]
        g_pos = g_state.pos
        g_quat = g_state.quat

        nface = 4

        vs = ti.Vector([0.0 for _ in range(3 * 4)], dt=gs.ti_float)
        if face_idx == 0:
            # Right
            vs[0], vs[1], vs[2] = g_size_x, g_size_y, g_size_z
            vs[3], vs[4], vs[5] = g_size_x, g_size_y, -g_size_z
            vs[6], vs[7], vs[8] = g_size_x, -g_size_y, -g_size_z
            vs[9], vs[10], vs[11] = g_size_x, -g_size_y, g_size_z
        elif face_idx == 1:
            # Left
            vs[0], vs[1], vs[2] = -g_size_x, g_size_y, -g_size_z
            vs[3], vs[4], vs[5] = -g_size_x, g_size_y, g_size_z
            vs[6], vs[7], vs[8] = -g_size_x, -g_size_y, g_size_z
            vs[9], vs[10], vs[11] = -g_size_x, -g_size_y, -g_size_z
        elif face_idx == 2:
            # Top
            vs[0], vs[1], vs[2] = -g_size_x, g_size_y, -g_size_z
            vs[3], vs[4], vs[5] = g_size_x, g_size_y, -g_size_z
            vs[6], vs[7], vs[8] = g_size_x, g_size_y, g_size_z
            vs[9], vs[10], vs[11] = -g_size_x, g_size_y, g_size_z
        elif face_idx == 3:
            # Bottom
            vs[0], vs[1], vs[2] = -g_size_x, -g_size_y, g_size_z
            vs[3], vs[4], vs[5] = g_size_x, -g_size_y, g_size_z
            vs[6], vs[7], vs[8] = g_size_x, -g_size_y, -g_size_z
            vs[9], vs[10], vs[11] = -g_size_x, -g_size_y, -g_size_z
        elif face_idx == 4:
            # Front
            vs[0], vs[1], vs[2] = -g_size_x, g_size_y, g_size_z
            vs[3], vs[4], vs[5] = g_size_x, g_size_y, g_size_z
            vs[6], vs[7], vs[8] = g_size_x, -g_size_y, g_size_z
            vs[9], vs[10], vs[11] = -g_size_x, -g_size_y, g_size_z
        elif face_idx == 5:
            # Back
            vs[0], vs[1], vs[2] = g_size_x, g_size_y, -g_size_z
            vs[3], vs[4], vs[5] = -g_size_x, g_size_y, -g_size_z
            vs[6], vs[7], vs[8] = -g_size_x, -g_size_y, -g_size_z
            vs[9], vs[10], vs[11] = g_size_x, -g_size_y, -g_size_z
        else:
            nface = 0

        # Transform the vertices to the global coordinates
        for i in range(nface):
            v = gs.ti_vec3(vs[3 * i + 0], vs[3 * i + 1], vs[3 * i + 2]) * 0.5
            v = gu.ti_transform_by_trans_quat(v, g_pos, g_quat)
            if i_o == 0:
                self.multi_contact_faces[i_b, i].face1 = v
            else:
                self.multi_contact_faces[i_b, i].face2 = v

        return nface

    @ti.func
    def func_mesh_face(self, i_g, i_b, i_o, face_idx):
        """
        Get the face vertices of the mesh.
        """
        # Get geometry position and quaternion
        g_state = self._solver.geoms_state[i_g, i_b]
        g_pos = g_state.pos
        g_quat = g_state.quat

        nvert = 3
        for i in range(nvert):
            i_v = self._solver.faces_info[face_idx].verts_idx[i]
            v = self._solver.verts_info[i_v].init_pos
            v = gu.ti_transform_by_trans_quat(v, g_pos, g_quat)
            if i_o == 0:
                self.multi_contact_faces[i_b, i].face1 = v
            else:
                self.multi_contact_faces[i_b, i].face2 = v

        return nvert

    @ti.func
    def func_clip_polygon(self, i_b, nface1, nface2, edgecon1, edgecon2, normal, approx_dir):
        """
        Clip a polygon against the another polygon using Sutherland-Hodgman algorithm.

        @ normal: The normal of the clipping polygon.
        @ approx_dir: Preferred separation direction for the clipping.
        """
        clipping_polygon = 1 if not edgecon1 else 2
        clipping_polygon_nface = nface1 if clipping_polygon == 1 else nface2

        # The clipping polygon should be at least a triangle
        if clipping_polygon_nface >= 3:
            # For each edge of the clipping polygon, find the half-plane
            # that is defined by the edge and the normal. The normal of
            # half-plane is perpendicular to the edge and face normal.
            for i in range(clipping_polygon_nface):
                v1 = self.multi_contact_faces[i_b, i].face1
                v2 = self.multi_contact_faces[i_b, (i + 1) % clipping_polygon_nface].face1
                v3 = self.multi_contact_faces[i_b, (i + 2) % clipping_polygon_nface].face1

                if clipping_polygon == 2:
                    v1 = self.multi_contact_faces[i_b, i].face2
                    v2 = self.multi_contact_faces[i_b, (i + 1) % clipping_polygon_nface].face2
                    v3 = self.multi_contact_faces[i_b, (i + 2) % clipping_polygon_nface].face2

                diff1 = v2 - v1
                diff2 = normal

                # Plane normal
                res = diff1.cross(diff2)

                # Reorient normal if needed
                inside_v3 = self.func_halfspace(v1, res, v3)
                if not inside_v3:
                    res = -res

                self.multi_contact_halfspaces[i_b, i].normal = res

                # Plane distance
                self.multi_contact_halfspaces[i_b, i].dist = v1.dot(res)

            # Initialize buffers to store the clipped polygons
            nclipped = gs.ti_ivec2(0, 0)
            nclipped[0] = nface2 if clipping_polygon == 1 else nface1

            # These values are swapped during the clipping process.
            pi, ci = 0, 1

            for i in range(nclipped[pi]):
                if clipping_polygon == 1:
                    self.multi_contact_clipped_polygons[i_b, pi, i] = self.multi_contact_faces[i_b, i].face2
                else:
                    self.multi_contact_clipped_polygons[i_b, pi, i] = self.multi_contact_faces[i_b, i].face1

            # For each edge of the clipping polygon, clip the subject polygon against it.
            # Here we use the Sutherland-Hodgman algorithm.
            for e in range(clipping_polygon_nface):

                # Get the point [a] on the clipping polygon edge,
                # and the normal [n] of the half-plane defined by the edge.
                a = self.multi_contact_faces[i_b, e].face1
                if clipping_polygon == 2:
                    a = self.multi_contact_faces[i_b, e].face2
                n = self.multi_contact_halfspaces[i_b, e].normal
                d = self.multi_contact_halfspaces[i_b, e].dist

                for i in range(nclipped[pi]):
                    # Get edge PQ of the subject polygon
                    P = self.multi_contact_clipped_polygons[i_b, pi, i]
                    Q = self.multi_contact_clipped_polygons[i_b, pi, (i + 1) % nclipped[pi]]

                    # Determine if P and Q are inside or outside the half-plane
                    inside_P = self.func_halfspace(a, n, P)
                    inside_Q = self.func_halfspace(a, n, Q)

                    # PQ entirely outside the clipping edge, skip
                    if not inside_P and not inside_Q:
                        continue

                    # PQ entirely inside the clipping edge, add Q to the clipped polygon
                    if inside_P and inside_Q:
                        self.multi_contact_clipped_polygons[i_b, ci, nclipped[ci]] = Q
                        nclipped[ci] += 1
                        continue

                    # PQ intersects the half-plane, add the intersection point
                    t, ip = self.func_plane_intersect(n, d, P, Q)
                    if t >= 0 and t <= 1:
                        self.multi_contact_clipped_polygons[i_b, ci, nclipped[ci]] = ip
                        nclipped[ci] += 1

                    # If Q is inside the half-plane, add it to the clipped polygon
                    if inside_Q:
                        self.multi_contact_clipped_polygons[i_b, ci, nclipped[ci]] = Q
                        nclipped[ci] += 1

                # Swap the buffers for the next edge clipping
                pi, ci = ci, pi

                # Reset the next clipped polygon count
                nclipped[ci] = 0

            nclipped_polygon = nclipped[pi]

            if nclipped_polygon >= 1:

                if self.max_contacts_per_pair < 5 and nclipped_polygon > 4:
                    # Approximate the clipped polygon with a convex quadrilateral
                    self.n_witness[i_b] = 4
                    rect = self.func_approximate_polygon_with_quad(i_b, pi, nclipped_polygon)

                    for i in range(4):
                        witness2 = self.multi_contact_clipped_polygons[i_b, pi, rect[i]]
                        witness1 = witness2 - approx_dir
                        self.witness[i_b, i].point_obj1 = witness1
                        self.witness[i_b, i].point_obj2 = witness2

                elif nclipped_polygon > self.max_contacts_per_pair:
                    # If the number of contacts exceeds the limit,
                    # only use the first [max_contacts_per_pair] contacts.
                    self.n_witness[i_b] = self.max_contacts_per_pair

                    for i in range(self.max_contacts_per_pair):
                        witness2 = self.multi_contact_clipped_polygons[i_b, pi, i]
                        witness1 = witness2 - approx_dir
                        self.witness[i_b, i].point_obj1 = witness1
                        self.witness[i_b, i].point_obj2 = witness2

                else:
                    n_witness = 0
                    # Just use every contact in the clipped polygon
                    for i in range(nclipped_polygon):
                        skip = False

                        polygon_vert = self.multi_contact_clipped_polygons[i_b, pi, i]

                        # Find if there were any duplicate contacts similar to [polygon_vert]
                        for j in range(n_witness):
                            prev_witness = self.witness[i_b, j].point_obj2
                            skip = self.func_is_equal_vec3(polygon_vert, prev_witness)
                            if skip:
                                break

                        if not skip:
                            self.witness[i_b, n_witness].point_obj2 = polygon_vert
                            self.witness[i_b, n_witness].point_obj1 = polygon_vert - approx_dir
                            n_witness += 1

                    self.n_witness[i_b] = n_witness

    @ti.func
    def func_halfspace(self, a, n, p):
        """
        Check if the point [p] is inside the half-space defined by the plane
        with normal [n] and point [a].
        """
        diff = p - a
        return diff.dot(n) > -self.FLOAT_MIN

    @ti.func
    def func_plane_intersect(self, pn, pd, v1, v2):
        """
        Compute the intersection point of the line segment [v1, v2]
        with the plane defined by the normal [pn] and distance [pd].

        v1 + t * (v2 - v1) = intersection point

        Return t and the intersection point.
        """
        t = self.FLOAT_MAX
        ip = gs.ti_vec3(0, 0, 0)

        dir = v2 - v1
        temp = pn.dot(dir)
        if temp != 0:
            t = (pd - pn.dot(v1)) / temp
            if t >= 0 and t <= 1:
                ip = v1 + t * dir

        return t, ip

    @ti.func
    def func_approximate_polygon_with_quad(self, i_b, polygon_start, nverts):
        """
        Find a convex quadrilateral that approximates the given N-gon [polygon].
        We find it by selecting the four vertices in the polygon that
        form the maximum area quadrilateral.
        """
        i_v = gs.ti_ivec4(0, 1, 2, 3)
        i_v0 = gs.ti_ivec4(0, 1, 2, 3)
        m = self.func_quadrilateral_area(i_b, polygon_start, i_v[0], i_v[1], i_v[2], i_v[3])

        # 1: change b, 2: change c, 3: change d
        change_flag = 3

        while True:
            i_v0[0], i_v0[1], i_v0[2], i_v0[3] = i_v[0], i_v[1], i_v[2], i_v[3]
            if change_flag == 3:
                i_v0[3] = (i_v[3] + 1) % nverts
            elif change_flag == 2:
                i_v0[2] = (i_v[2] + 1) % nverts

            # Compute the area of the quadrilateral formed by the vertices
            m_next = self.func_quadrilateral_area(i_b, polygon_start, i_v0[0], i_v0[1], i_v0[2], i_v0[3])
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

            pass

        return i_v

    @ti.func
    def func_quadrilateral_area(self, i_b, i_0, i_v0, i_v1, i_v2, i_v3):
        """
        Compute the area of the quadrilateral formed by the vertices
        with indices [i_v0, i_v1, i_v2, i_v3] in the [verts] array.
        """
        a = self.multi_contact_clipped_polygons[i_b, i_0, i_v0]
        b = self.multi_contact_clipped_polygons[i_b, i_0, i_v1]
        c = self.multi_contact_clipped_polygons[i_b, i_0, i_v2]
        d = self.multi_contact_clipped_polygons[i_b, i_0, i_v3]

        ad = d - a
        db = b - d
        bc = c - b
        ca = a - c
        e = ad.cross(db)
        f = bc.cross(ca)
        g = e + f
        return 0.5 * ti.math.length(g)

    """
    Helpers
    """

    @ti.func
    def func_is_discrete_geoms(self, i_ga, i_gb, i_b):
        """
        Check if the given geoms are discrete geometries.
        """
        return self.func_is_discrete_geom(i_ga, i_b) and self.func_is_discrete_geom(i_gb, i_b)

    @ti.func
    def func_is_discrete_geom(self, i_g, i_b):
        """
        Check if the given geom is a discrete geometry.
        """
        geom_type = self._solver.geoms_info[i_g].type
        return geom_type == gs.GEOM_TYPE.MESH or geom_type == gs.GEOM_TYPE.BOX

    @ti.func
    def func_is_sphere_swept_geom(self, i_g, i_b):
        """
        Check if the given geoms are sphere-swept geometries.
        """
        geom_type = self._solver.geoms_info[i_g].type
        return geom_type == gs.GEOM_TYPE.SPHERE or geom_type == gs.GEOM_TYPE.CAPSULE

    @ti.func
    def func_support(self, i_ga, i_gb, i_b, dir, shrink_sphere):
        """
        Find support points on the two objects using [dir].
        [dir] should be a unit vector from [ga] (obj1) to [gb] (obj2).
        """
        support_point_obj1 = gs.ti_vec3(0, 0, 0)
        support_point_obj2 = gs.ti_vec3(0, 0, 0)
        support_point_id_obj1 = -1
        support_point_id_obj2 = -1
        for i in range(2):
            d = dir if i == 0 else -dir
            i_g = i_ga if i == 0 else i_gb

            sp, si = self.support_driver(d, i_g, i_b, i, shrink_sphere)
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
    def func_rotmat_120(self, axis):
        """
        Rotation matrix for 120 degrees rotation around the given axis.
        """
        n = ti.math.length(axis)
        u1 = axis[0] / n
        u2 = axis[1] / n
        u3 = axis[2] / n

        # sin and cos of 120 degrees
        sin = 0.86602540378
        cos = -0.5

        mat = ti.math.mat3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        mat[0, 0] = cos + u1 * u1 * (1 - cos)
        mat[0, 1] = u1 * u2 * (1 - cos) - u3 * sin
        mat[0, 2] = u1 * u3 * (1 - cos) + u2 * sin
        mat[1, 0] = u2 * u1 * (1 - cos) + u3 * sin
        mat[1, 1] = cos + u2 * u2 * (1 - cos)
        mat[1, 2] = u2 * u3 * (1 - cos) - u1 * sin
        mat[2, 0] = u3 * u1 * (1 - cos) - u2 * sin
        mat[2, 1] = u3 * u2 * (1 - cos) + u1 * sin
        mat[2, 2] = cos + u3 * u3 * (1 - cos)

        return mat

    @ti.func
    def func_project_origin_to_plane(self, v1, v2, v3):
        """
        Project the origin onto the plane defined by the simplex vertices.
        Find the projected point and return flag with it.
        """
        point, flag = gs.ti_vec3(0, 0, 0), -1

        d21 = v2 - v1
        d31 = v3 - v1
        d32 = v3 - v2

        # Normal = (v1 - v2) x (v3 - v2)
        n = d32.cross(d21)
        nv = n.dot(v2)
        nn = n.dot(n)
        if nn == 0:
            flag = 1
        elif nv != 0 and nn > self.FLOAT_MIN:
            point = n * (nv / nn)
            flag = 0

        if flag == -1:
            # If previous attempt was numerically unstable,
            # try use other normal estimations

            # Normal = (v2 - v1) x (v3 - v1)
            n = d21.cross(d31)
            nv = n.dot(v1)
            nn = n.dot(n)
            if nn == 0:
                flag = 1
            elif nv != 0 and nn > self.FLOAT_MIN:
                point = n * (nv / nn)
                flag = 0

        if flag == -1:
            # Last fallback

            # Normal = (v1 - v3) x (v2 - v3)
            n = d31.cross(d32)
            nv = n.dot(v3)
            nn = n.dot(n)
            point = n * (nv / nn)
            flag = 0

        return point, flag

    @ti.func
    def func_project_origin_to_line(self, v1, v2):
        """
        Project the origin onto the line defined by the simplex vertices.

        P = v2 - ((v1 * diff) / (diff * diff)) * diff
        """
        diff = v2 - v1
        k = v2.dot(diff) / diff.dot(diff)
        P = v2 - k * diff

        return P

    @ti.func
    def func_simplex_vertex_linear_comb(self, i_b, i_v, i_s1, i_s2, i_s3, i_s4, _lambda, n):
        """
        Compute the linear combination of the simplex vertices

        @ i_v: Which vertex to use (0: obj1, 1: obj2, 2: minkowski)
        @ n: Number of vertices to combine, combine the first n vertices
        """
        res = gs.ti_vec3(0, 0, 0)

        s1 = self.gjk_simplex_vertex[i_b, i_s1].obj1
        s2 = self.gjk_simplex_vertex[i_b, i_s2].obj1
        s3 = self.gjk_simplex_vertex[i_b, i_s3].obj1
        s4 = self.gjk_simplex_vertex[i_b, i_s4].obj1
        if i_v == 1:
            s1 = self.gjk_simplex_vertex[i_b, i_s1].obj2
            s2 = self.gjk_simplex_vertex[i_b, i_s2].obj2
            s3 = self.gjk_simplex_vertex[i_b, i_s3].obj2
            s4 = self.gjk_simplex_vertex[i_b, i_s4].obj2
        elif i_v == 2:
            s1 = self.gjk_simplex_vertex[i_b, i_s1].mink
            s2 = self.gjk_simplex_vertex[i_b, i_s2].mink
            s3 = self.gjk_simplex_vertex[i_b, i_s3].mink
            s4 = self.gjk_simplex_vertex[i_b, i_s4].mink

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
    def func_det3(self, v1, v2, v3):
        """
        Compute the determinant of a 3x3 matrix formed by the vectors v1, v2, v3.
        M = [v1 | v2 | v3]
        """
        return (
            v1[0] * (v2[1] * v3[2] - v2[2] * v3[1])
            - v1[1] * (v2[0] * v3[2] - v2[2] * v3[0])
            + v1[2] * (v2[0] * v3[1] - v2[1] * v3[0])
        )

    @ti.func
    def func_compare_sign(self, a, b):
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
    def func_is_equal_vec3(self, a, b):
        """
        Check if two vectors are equal within a small tolerance.
        """
        diff = ti.abs(a - b)
        return (diff < self.FLOAT_MIN).all()

    def reset(self):
        pass

    @ti.func
    def support_mesh(self, direction, i_g, i_b, i_o):
        g_state = self._solver.geoms_state[i_g, i_b]
        d_mesh = gu.ti_transform_by_quat(direction, gu.ti_inv_quat(g_state.quat))

        # Exhaustively search for the vertex with maximum dot product
        fmax = -self.FLOAT_MAX
        imax = 0

        vert_start = self._solver.geoms_info.vert_start[i_g]
        vert_end = self._solver.geoms_info.vert_end[i_g]

        # Use the previous maximum vertex if it is within the current range
        prev_imax = self.support_vertex_id[i_b, i_o]
        if (prev_imax >= vert_start) and (prev_imax < vert_end):
            pos = self._solver.verts_info[prev_imax].init_pos
            fmax = d_mesh.dot(pos)
            imax = prev_imax

        for i in range(vert_start, vert_end):
            pos = self._solver.verts_info[i].init_pos
            vdot = d_mesh.dot(pos)
            if vdot > fmax:
                fmax = vdot
                imax = i

        v = self._solver.verts_info[imax].init_pos
        vid = imax

        self.support_vertex_id[i_b, i_o] = vid

        v_ = gu.ti_transform_by_trans_quat(v, g_state.pos, g_state.quat)
        return v_, vid

    @ti.func
    def support_driver(self, direction, i_g, i_b, i_o, shrink_sphere):
        """
        @ shrink_sphere: If True, use point and line support for sphere and capsule.
        """
        v = ti.Vector.zero(gs.ti_float, 3)
        vid = -1

        geom_type = self._solver.geoms_info[i_g].type
        if geom_type == gs.GEOM_TYPE.SPHERE:
            v = self.support.support_sphere(direction, i_g, i_b, shrink_sphere)
        elif geom_type == gs.GEOM_TYPE.ELLIPSOID:
            v = self.support.support_ellipsoid(direction, i_g, i_b)
        elif geom_type == gs.GEOM_TYPE.CAPSULE:
            v = self.support.support_capsule(direction, i_g, i_b, shrink_sphere)
        elif geom_type == gs.GEOM_TYPE.BOX:
            v, vid = self.support.support_box(direction, i_g, i_b)
        elif geom_type == gs.GEOM_TYPE.TERRAIN:
            if ti.static(self._solver.collider._has_terrain):
                v, vid = self.support.support_prism(direction, i_g, i_b)
        elif geom_type == gs.GEOM_TYPE.MESH and self._solver._enable_mujoco_compatibility:
            # If mujoco-compatible, do exhaustive search for the vertex
            v, vid = self.support_mesh(direction, i_g, i_b, i_o)
        else:
            v, vid = self.support.support_field._func_support_world(direction, i_g, i_b)
        return v, vid
