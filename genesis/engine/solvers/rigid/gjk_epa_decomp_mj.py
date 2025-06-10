import numpy as np
import taichi as ti
import torch

import genesis as gs
import genesis.utils.geom as gu

from .support_field_decomp import SupportField
# @TODO: type checking for float, int, etc.

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
class GJKEPA:
    def __init__(self, rigid_solver):
        self._solver = rigid_solver
        self._max_contact_pairs = rigid_solver._max_collision_pairs
        self._B = rigid_solver._B
        self._para_level = rigid_solver._para_level
        
        self.FLOAT_MIN = gs.np_float(1e-15) # ti.sqrt(gs.EPS)
        self.FLOAT_MIN_SQ = self.FLOAT_MIN * self.FLOAT_MIN
        self.FLOAT_MAX = gs.np_float(1e15)
        self.FLOAT_MAX_SQ = self.FLOAT_MAX * self.FLOAT_MAX
        self.tolerance = gs.np_float(1e-6)
        self.collision_eps = gs.np_float(1e-6)

        # Maximum number of contact points to find per pair
        self.max_contacts_per_pair = 50
        
        ### Gjk
        self.gjk_max_iterations = 50
        struct_simplex_vertex = ti.types.struct(
            # Support points on the two objects
            obj1=gs.ti_vec3,
            obj2=gs.ti_vec3,
            # Vertex on Minkowski difference
            mink=gs.ti_vec3
        )
        struct_simplex = ti.types.struct(
            # Number of vertices in the simplex
            nverts=gs.ti_int,
            dist=gs.ti_float,
        )
        # @TODO: data arrangement?
        self.gjk_simplex_vertex = struct_simplex_vertex.field(shape=(self._B, 4))
        self.gjk_simplex_vertex_intersect = struct_simplex_vertex.field(shape=(self._B, 4))
        self.gjk_simplex = struct_simplex.field(shape=(self._B,))
        self.gjk_nsimplex = ti.field(dtype=gs.ti_int, shape=(self._B,))

        self.mw_gjk_plane = ti.Vector.field(n=3, dtype=gs.ti_float, shape=(self._B, 4))
        
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
            map_idx=gs.ti_int
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
        self.num_witness = ti.field(dtype=gs.ti_int, shape=(self._B,))

        # Distance between the two objects
        # If the objects are separated, the distance is positive.
        # If the objects are intersecting, the distance is negative (depth).
        self.distance = ti.field(dtype=gs.ti_float, shape=(self._B,))
        
        # Normal vector of the contact point
        self.normal = gs.ti_vec3.field(shape=(self._B,))

        ### Support field
        self.support_field = SupportField(rigid_solver)
    
    '''
    GJK algorithms
    '''
    @ti.func
    def func_gjk(self, i_ga, i_gb, i_b):
        '''
        GJK algorithm to compute the minimum distance between two convex objects.
        '''
        # Simplex index
        n = 0
        
        # Final number of simplex vertices
        nsimplex = 0

        # Number of witness points and distance
        nx = 0
        dist = 0.0

        # Final return flag
        return_flag = 0
        
        # Set initial guess of support vector using the positions.
        # The support vector should be a non-zero vector.
        approx_witness_point_obj1 = self._solver.geoms_state[i_ga, i_b].pos
        approx_witness_point_obj2 = self._solver.geoms_state[i_gb, i_b].pos
        print("Initial guess (x1_k): ", 
              f"{approx_witness_point_obj1[0]:.20g}",
              f"{approx_witness_point_obj1[1]:.20g}", 
              f"{approx_witness_point_obj1[2]:.20g}")
        print("Initial guess (x2_k): ", 
            f"{approx_witness_point_obj2[0]:.20g}",
            f"{approx_witness_point_obj2[1]:.20g}", 
            f"{approx_witness_point_obj2[2]:.20g}")
        support_vector = approx_witness_point_obj1 - approx_witness_point_obj2
        
        if support_vector.dot(support_vector) < self.FLOAT_MIN_SQ:
            approx_witness_point_obj1.x = approx_witness_point_obj1.x + 0.01
            approx_witness_point_obj2.x = approx_witness_point_obj2.x - 0.01
            support_vector = approx_witness_point_obj1 - approx_witness_point_obj2
        
        support_vector_norm = 0.0
        
        # Since we use GJK mainly for collision detection,
        # we use gjk_intersect when it is available
        backup_gjk = 1
        
        epsilon = 0.0
        if not self.func_is_discrete_geoms(i_ga, i_gb, i_b):
            # If the objects are smooth, finite convergence is not guaranteed,
            # so we need to set some epsilon to determine convergence
            epsilon = 0.5 * self.tolerance * self.tolerance
        
        for i in range(self.gjk_max_iterations):
            print("Support vector:", f"{support_vector[0]:.20g}", f"{support_vector[1]:.20g}", f"{support_vector[2]:.20g}")
            # Compute the current support points
            support_vector_norm = support_vector.dot(support_vector)
            if support_vector_norm < self.FLOAT_MIN_SQ:
                # If the support vector is too small, it means
                # that origin is located in the Minkowski difference
                # with high probability, so we can stop.
                # support_vector_norm = 0
                break
            
            support_vector_norm = ti.math.sqrt(support_vector_norm)
            
            # Dir to compute the support point
            # (pointing from obj1 to obj2)
            dir = -support_vector * (1.0 / support_vector_norm)
            
            self.gjk_simplex_vertex[i_b, n].obj1, \
            self.gjk_simplex_vertex[i_b, n].obj2, \
            self.gjk_simplex_vertex[i_b, n].mink = \
                self.func_support(i_ga, i_gb, i_b, dir)
            
            print("Obj1:", f"{self.gjk_simplex_vertex[i_b, n].obj1[0]:.20g}",
                  f"{self.gjk_simplex_vertex[i_b, n].obj1[1]:.20g}",
                  f"{self.gjk_simplex_vertex[i_b, n].obj1[2]:.20g}")
            print("Obj2:", f"{self.gjk_simplex_vertex[i_b, n].obj2[0]:.20g}",
                    f"{self.gjk_simplex_vertex[i_b, n].obj2[1]:.20g}",
                    f"{self.gjk_simplex_vertex[i_b, n].obj2[2]:.20g}")
            print("Mink:", f"{self.gjk_simplex_vertex[i_b, n].mink[0]:.20g}",
                    f"{self.gjk_simplex_vertex[i_b, n].mink[1]:.20g}",
                    f"{self.gjk_simplex_vertex[i_b, n].mink[2]:.20g}")

                        
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
                #print("GJK converged in", i, "iterations")
                break

            # Check if the objects are separated using support vector
            is_separated = x_k.dot(s_k) > 0.0
            if is_separated:
                nsimplex = 0
                nx = 0
                dist = self.FLOAT_MAX
                return_flag = 2
                break
            
            if n == 3 and backup_gjk:
                # Tetrahedron is generated, try to get contact info
                intersect_flag = self.func_gjk_intersect(i_ga, i_gb, i_b)
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
                    backup_gjk = 0
                    
            # Run the distance subalgorithm to compute the barycentric
            # coordinates of the closest point to the origin in the simplex
            _lambda = self.func_simple_gjk_subdistance(i_b, n + 1)
            print("HI")
            print("Lambda: ", f"{_lambda[0]:.20g}", f"{_lambda[1]:.20g}",
                  f"{_lambda[2]:.20g}", f"{_lambda[3]:.20g}")
            
            # Remove vertices from the simplex with zero barycentric coordinates
            # as they are not needed for the next iteration
            #print("Iteration", i, "lambda:", _lambda)
            n = 0
            for j in range(4):
                if _lambda[j] > 0:    
                    self.gjk_simplex_vertex[i_b, n] = self.gjk_simplex_vertex[i_b, j]
                    _lambda[n] = _lambda[j]
                    n += 1

            #print("Iteration", i, "n:", n, "lambda:", _lambda)

            # Should not occur
            if n < 1:
                nsimplex = 0
                nx = 0
                dist = self.FLOAT_MAX
                return_flag = 3
                break
            
            # Get the next support vector
            next_support_vector = self.func_simplex_vertex_linear_comb(
                i_b, 2, 0, 1, 2, 3, _lambda, n
            )
            print("Next support vector:", f"{next_support_vector[0]:.20g}",
                  f"{next_support_vector[1]:.20g}", f"{next_support_vector[2]:.20g}")
            if self.func_is_equal_vec3(next_support_vector, support_vector):
                # If the next support vector is equal to the previous one,
                # we converged to the minimum distance
                #print("GJK equal")
                break

            support_vector = next_support_vector
            
            if n == 4:
                # We have a tetrahedron containing the origin,
                # so we can return early.
                # This is because only when the origin is inside
                # the tetrahedron, the barycentric coordinates
                # are all positive.
                #print("GJK tetrahedron found")
                break

        #print("Return flag:", return_flag)
        
        if return_flag == 0:
            # If there was no conclusion until now...
            nx = 1
            nsimplex = n
            dist = support_vector_norm

        self.num_witness[i_b] = nx
        self.distance[i_b] = dist
        self.gjk_nsimplex[i_b] = nsimplex
        collision = dist < self.collision_eps

        return collision
    
    @ti.func
    def func_gjk_intersect(self, i_ga, i_gb, i_b):
        
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
            
            # # If the origin is strictly on any affine hull of the triangle faces, 
            # # convergence will fail, so ignore this case
            # if (
            #     sdist_0 == 0.0 or
            #     sdist_1 == 0.0 or
            #     sdist_2 == 0.0 or
            #     sdist_3 == 0.0
            # ):
            #     flag = -1
            #     break
            
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
            if min_sdist >= 0: #self.FLOAT_MIN:
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
            self.gjk_simplex_vertex_intersect[i_b, min_si].obj1, \
            self.gjk_simplex_vertex_intersect[i_b, min_si].obj2, \
            self.gjk_simplex_vertex_intersect[i_b, min_si].mink = \
                self.func_support(i_ga, i_gb, i_b, min_normal)
                
            # Check if the origin is strictly outside of the Minkowski difference
            # (which means there is no collision)
            new_minkowski = self.gjk_simplex_vertex_intersect[i_b, min_si].mink

            # @TODO: For numerical stability, we use small epsilon, but is it too much?
            is_no_collision = (new_minkowski.dot(min_normal) < 0) # -self.FLOAT_MIN)
            if is_no_collision:
                flag = 0
                break
            
            # Swap vertices in the simplex to retain orientation
            m = (min_i + 1) % 4
            n = (min_i + 2) % 4
            swap = si[m]
            si[m] = si[n]
            si[n] = swap

        #print("No collision flag:", flag)
        
        # Never found origin
        if flag == -2:
            flag = -1
        
        return flag
    
    @ti.func
    def func_gjk_contact(self, i_ga, i_gb, i_b):
        '''
        GJK algorithm to check collision between two convex geometries.
        '''
        collided = False

        dir = ti.math.vec3(0, 0, 1)
        dir_n = -dir
        depth = self.FLOAT_MAX
        
        # test two random directions and choose better one
        dist_max, sp_obj1_0, sp_obj2_0, sp_0 = self.gjk_compute_support(dir, i_ga, i_gb, i_b)
        dist_min, sp_obj1_1, sp_obj2_1, sp_1 = self.gjk_compute_support(dir_n, i_ga, i_gb, i_b)
        if dist_max < dist_min:
            depth = dist_max
        else:
            depth = dist_min
        
        sd = sp_0 - sp_1
        dir = self.gjk_orthonormal(sd)        # find a vector that lies in the plane orthogonal to sd

        dist_max, sp_obj1_2, sp_obj2_2, sp_3 = self.gjk_compute_support(dir, i_ga, i_gb, i_b)
        
        # Initialize a 2-simplex with simplex[2]==simplex[1]. This ensures the
        # correct winding order for face normals defined below. Face 0 and face 3
        # are degenerate, and face 1 and 2 have opposing normals.
        self.gjk_simplex_vertex[i_b, 0].obj1, \
        self.gjk_simplex_vertex[i_b, 0].obj2, \
        self.gjk_simplex_vertex[i_b, 0].mink = sp_obj1_0, sp_obj2_0, sp_0

        self.gjk_simplex_vertex[i_b, 1].obj1, \
        self.gjk_simplex_vertex[i_b, 1].obj2, \
        self.gjk_simplex_vertex[i_b, 1].mink = sp_obj1_1, sp_obj2_1, sp_1

        self.gjk_simplex_vertex[i_b, 2].obj1, \
        self.gjk_simplex_vertex[i_b, 2].obj2, \
        self.gjk_simplex_vertex[i_b, 2].mink = sp_obj1_1, sp_obj2_1, sp_1  # simplex[2] == simplex[1]

        self.gjk_simplex_vertex[i_b, 3].obj1, \
        self.gjk_simplex_vertex[i_b, 3].obj2, \
        self.gjk_simplex_vertex[i_b, 3].mink = sp_obj1_2, sp_obj2_2, sp_3
        
        if dist_max < depth:
            depth = dist_max
        if dist_min < depth:
            depth = dist_min
        
        for _ in range(self.gjk_max_iterations):
            # winding orders: plane[0] ccw, plane[1] cw, plane[2] ccw, plane[3] cw
            mink_0 = self.gjk_simplex_vertex[i_b, 0].mink
            mink_1 = self.gjk_simplex_vertex[i_b, 1].mink
            mink_2 = self.gjk_simplex_vertex[i_b, 2].mink
            mink_3 = self.gjk_simplex_vertex[i_b, 3].mink

            self.mw_gjk_plane[i_b, 0] = ti.math.cross(
                mink_3 - mink_2,
                mink_1 - mink_2
            )
            self.mw_gjk_plane[i_b, 1] = ti.math.cross(
                mink_3 - mink_0,
                mink_2 - mink_0
            )
            self.mw_gjk_plane[i_b, 2] = ti.math.cross(
                mink_3 - mink_1,
                mink_0 - mink_1
            )
            self.mw_gjk_plane[i_b, 3] = ti.math.cross(
                mink_2 - mink_0,
                mink_1 - mink_0
            )
            plane0, p0 = self.gjk_normalize(self.mw_gjk_plane[i_b, 0])
            plane1, p1 = self.gjk_normalize(self.mw_gjk_plane[i_b, 1])
            plane2, p2 = self.gjk_normalize(self.mw_gjk_plane[i_b, 2])
            plane3, p3 = self.gjk_normalize(self.mw_gjk_plane[i_b, 3])
            
            self.mw_gjk_plane[i_b, 0] = plane0
            self.mw_gjk_plane[i_b, 1] = plane1
            self.mw_gjk_plane[i_b, 2] = plane2
            self.mw_gjk_plane[i_b, 3] = plane3
            
            # Compute distance of each face halfspace to the origin. If dplane<0, then the
            # origin is outside the halfspace. If dplane>0 then the origin is inside
            # the halfspace defined by the face plane.
            dplane = ti.math.vec4(0, 0, 0, 0)
            dplane.fill(self.FLOAT_MAX)
            
            if p0:
                dplane.x = ti.math.dot(plane0, self.gjk_simplex_vertex[i_b, 2].mink)
            if p1:
                dplane.y = ti.math.dot(plane1, self.gjk_simplex_vertex[i_b, 0].mink)
            if p2:
                dplane.z = ti.math.dot(plane2, self.gjk_simplex_vertex[i_b, 1].mink)
            if p3:
                dplane.w = ti.math.dot(plane3, self.gjk_simplex_vertex[i_b, 0].mink)

            # pick plane normal with minimum distance to the origin
            i1 = 0 if dplane[0] < dplane[1] else 1
            i2 = 2 if dplane[2] < dplane[3] else 3
            index = i1 if dplane[i1] < dplane[i2] else i2

            if dplane[index] > 0.0:
                # origin is inside the simplex, objects are intersecting
                collided = True
                break

            # add new support point to the simplex
            iplane = self.mw_gjk_plane[i_b, index]
            dist, sp_obj1_i, sp_obj2_i, sp_i = self.gjk_compute_support(iplane, i_ga, i_gb, i_b)
            self.gjk_simplex_vertex[i_b, index].obj1 = sp_obj1_i
            self.gjk_simplex_vertex[i_b, index].obj2 = sp_obj2_i
            self.gjk_simplex_vertex[i_b, index].mink = sp_i
            
            if dist < depth:
                depth = dist
                
            # preserve winding order of the simplex faces
            index1 = (index + 1) % 4
            index2 = (index + 2) % 4
            swap = self.gjk_simplex_vertex[i_b, index1]
            self.gjk_simplex_vertex[i_b, index1] = self.gjk_simplex_vertex[i_b, index2]
            self.gjk_simplex_vertex[i_b, index2] = swap

            if dist < 0.0:
                collided = False
                break  # objects are likely non-intersecting

        # self.distance[i_b] = depth
        if collided:
            self.gjk_nsimplex[i_b] = 4
            self.gjk_simplex[i_b].nverts = 4
            self.gjk_simplex[i_b].dist = 0
        else:
            self.gjk_nsimplex[i_b] = 0
            self.gjk_simplex[i_b].nverts = 0
            self.gjk_simplex[i_b].dist = self.FLOAT_MAX

        return collided

    @ti.func
    def func_gjk_triangle_info(self, i_b, i_va, i_vb, i_vc):
        '''
        Compute normal and signed distance of the triangle 
        face on the simplex from the origin.
        '''
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
            # if sdist == 0:
            #     print("Zero signed distance in GJK triangle info!")
            #     print("Vertices:", vertex_1, vertex_2, vertex_3)
            #     print("Normal:", normal)
        else:
            # if the normal length is unstable, return max distance
            sdist = self.FLOAT_MAX
            
        return normal, sdist
    
    @ti.func
    def func_gjk_subdistance(self, i_b, n):
        '''
        Compute the barycentric coordinates of the 
        closest point to the origin in the simplex.
        [Montanari et al, ToG 2017]
        '''
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
        C1, C2, C3, C4 = Cs[0], Cs[1], Cs[2], Cs[3]
        m_det = C1 + C2 + C3 + C4
        
        # Compare sign of the cofactors with the determinant
        scs = gs.ti_ivec4(0, 0, 0, 0)
        for i in range(4):
            scs[i] = self.func_compare_sign(Cs[i], m_det)
        sc1, sc2, sc3, sc4 = scs[0], scs[1], scs[2], scs[3]
        
        if (sc1 and sc2 and sc3 and sc4):
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
                    closest_point = self.func_simplex_vertex_linear_comb(
                        i_b, 2, j_s1, j_s2, j_s3, 0, _lambda2d, 3
                    )
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
            print("Falling back to 1D projection in GJK subdistance 2D")
            # If projection failed because the zero normal,
            # project on to the first edge of the triangle
            _lambda = self.func_gjk_subdistance_1d(i_b, i_s1, i_s2)
        else:
            #print("Projected origin in GJK subdistance 2D:", proj_o)
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
            
            m1 = s2[1]*s3[2] - s2[2]*s3[1] \
                - s1[1]*s3[2] + s1[2]*s3[1] \
                + s1[1]*s2[2] - s1[2]*s2[1]
            m2 = s2[0]*s3[2] - s2[2]*s3[0] \
                - s1[0]*s3[2] + s1[2]*s3[0] \
                + s1[0]*s2[2] - s1[2]*s2[0]
            m3 = s2[0]*s3[1] - s2[1]*s3[0] \
                - s1[0]*s3[1] + s1[1]*s3[0] \
                + s1[0]*s2[1] - s1[1]*s2[0]
            
            #print("simplex vertices in GJK subdistance 2D:", s1, s2, s3)
            print("Minors in GJK subdistance 2D:", f"{m1:.20g}", f"{m2:.20g}", f"{m3:.20g}")
                
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
            C1 = proj_o_2d[0]*s2_2d[1] + proj_o_2d[1]*s3_2d[0] + s2_2d[0]*s3_2d[1] - \
                proj_o_2d[0]*s3_2d[1] - proj_o_2d[1]*s2_2d[0] - s3_2d[0]*s2_2d[1]
                
            # C2 corresponds to the signed area of 2-simplex (triangle): (proj_o_2d, s1_2d, s3_2d)
            C2 = proj_o_2d[0]*s3_2d[1] + proj_o_2d[1]*s1_2d[0] + s3_2d[0]*s1_2d[1] - \
                proj_o_2d[0]*s1_2d[1] - proj_o_2d[1]*s3_2d[0] - s1_2d[0]*s3_2d[1]
                
            # C3 corresponds to the signed area of 2-simplex (triangle): (proj_o_2d, s1_2d, s2_2d)
            C3 = proj_o_2d[0]*s1_2d[1] + proj_o_2d[1]*s2_2d[0] + s1_2d[0]*s2_2d[1] - \
                proj_o_2d[0]*s2_2d[1] - proj_o_2d[1]*s1_2d[0] - s2_2d[0]*s1_2d[1]
            
            Cs = gs.ti_vec3(C1, C2, C3)
                
            # Compare sign of the cofactors with the determinant
            scs = gs.ti_ivec3(0, 0, 0)
            for i in range(3):
                scs[i] = self.func_compare_sign(Cs[i], m_max)
            sc1, sc2, sc3 = scs[0], scs[1], scs[2]

            print("Cofactors:", f"{C1:.20g}", f"{C2:.20g}", f"{C3:.20g}")
            # print("Signs:", sc1, sc2, sc3)
            
            if (sc1 and sc2 and sc3):
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
                        closest_point = self.func_simplex_vertex_linear_comb(
                            i_b, 2, j_s1, j_s2, 0, 0, _lambda1d, 2
                        )
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
            failed_3d = (n == 4)
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
                        closest_point = self.func_simplex_vertex_linear_comb(
                            i_b, 2, k_1, k_2, k_3, 0, _lambda2d, 3
                        )
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
            failed_3d = (n == 4)
            failed_2d = (n == 3)

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
                    closest_point = self.func_simplex_vertex_linear_comb(
                        i_b, 2, k_1, k_2, 0, 0, _lambda1d, 2
                    )
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
        C1, C2, C3, C4 = Cs[0], Cs[1], Cs[2], Cs[3]
        m_det = C1 + C2 + C3 + C4
        
        # Compare sign of the cofactors with the determinant
        scs = gs.ti_ivec4(0, 0, 0, 0)
        for i in range(4):
            scs[i] = self.func_compare_sign(Cs[i], m_det)
        sc1, sc2, sc3, sc4 = scs[0], scs[1], scs[2], scs[3]
        
        if (sc1 and sc2 and sc3 and sc4):
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
            
            m1 = s2[1]*s3[2] - s2[2]*s3[1] \
                - s1[1]*s3[2] + s1[2]*s3[1] \
                + s1[1]*s2[2] - s1[2]*s2[1]
            m2 = s2[0]*s3[2] - s2[2]*s3[0] \
                - s1[0]*s3[2] + s1[2]*s3[0] \
                + s1[0]*s2[2] - s1[2]*s2[0]
            m3 = s2[0]*s3[1] - s2[1]*s3[0] \
                - s1[0]*s3[1] + s1[1]*s3[0] \
                + s1[0]*s2[1] - s1[1]*s2[0]
                
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
            C1 = proj_o_2d[0]*s2_2d[1] + proj_o_2d[1]*s3_2d[0] + s2_2d[0]*s3_2d[1] - \
                proj_o_2d[0]*s3_2d[1] - proj_o_2d[1]*s2_2d[0] - s3_2d[0]*s2_2d[1]
                
            # C2 corresponds to the signed area of 2-simplex (triangle): (proj_o_2d, s1_2d, s3_2d)
            C2 = proj_o_2d[0]*s3_2d[1] + proj_o_2d[1]*s1_2d[0] + s3_2d[0]*s1_2d[1] - \
                proj_o_2d[0]*s1_2d[1] - proj_o_2d[1]*s3_2d[0] - s1_2d[0]*s3_2d[1]
                
            # C3 corresponds to the signed area of 2-simplex (triangle): (proj_o_2d, s1_2d, s2_2d)
            C3 = proj_o_2d[0]*s1_2d[1] + proj_o_2d[1]*s2_2d[0] + s1_2d[0]*s2_2d[1] - \
                proj_o_2d[0]*s2_2d[1] - proj_o_2d[1]*s1_2d[0] - s2_2d[0]*s1_2d[1]
            
            Cs = gs.ti_vec3(C1, C2, C3)
                
            # Compare sign of the cofactors with the determinant
            scs = gs.ti_ivec3(0, 0, 0)
            for i in range(3):
                scs[i] = self.func_compare_sign(Cs[i], m_max)
            sc1, sc2, sc3 = scs[0], scs[1], scs[2]
    
            if (sc1 and sc2 and sc3):
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
    
    '''
    EPA algorithms
    '''
    @ti.func
    def func_epa(self, i_ga, i_gb, i_b):
        '''
        Run EPA algorithm for finding the face on the expanded
        polytope that best approximates the penetration depth.
        '''
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
            # print("EPA iteration:", k)
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
                print("EPA: Lower bound is larger than upper bound, stopping.")
                # print("Lower bound:", lower2, "Upper bound:", upper2)
                break

            if lower2 <= 0:
                # Invalid lower bound (0), stop the algorithm
                # (origin is on the affine hull of face)
                print("EPA: Lower bound is zero, stopping.")
                break

            # Find a new support point w from the nearest face's normal
            lower = ti.sqrt(lower2)
            dir = self.polytope_faces[i_b, nearest_i_f].normal
            wi = self.func_epa_support(i_ga, i_gb, i_b, dir, lower)
            w = self.polytope_verts[i_b, wi].mink
            
            print("EPA: dir =", f"{dir[0]:.20g}", f"{dir[1]:.20g}", f"{dir[2]:.20g}")
            print("EPA: w =", f"{w[0]:.20g}", f"{w[1]:.20g}", f"{w[2]:.20g}")
            # The upper bound of depth at k-th iteration
            upper_k = w.dot(dir) / lower
            if upper_k < upper:
                upper = upper_k
                upper2 = upper * upper

            # print("EPA Dir:", dir, f"Lower bound: {lower2:.15f} ", f"Upper bound: {upper2:.15f}")
            
            
            # If the upper bound and lower bound are close enough,
            # we can stop the algorithm
            print("EPA: Lower bound:", f"{lower:.20g}", "Upper bound:", f"{upper:.20g}")
            print("EPA: Upper bound minus lower bound:", f"{upper - lower:.20g} / {tolerance:.20g}")
            if (upper - lower) < tolerance:
                print("EPA: Upper and lower bounds are close enough, stopping.")
                break

            # @TODO: Check the vertex w is already in the polytope
            if discrete:
                pass

            self.polytope[i_b].horizon_w = w
            #print("Horizon vertex:", w)

            # Compute horizon
            horizon_flag = self.func_epa_horizon(i_b, nearest_i_f)
            # print("Horizon flag:", horizon_flag)
            
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
                print("EPA: Polytope is full, cannot insert new faces.")
                break

            # Attach the new faces
            # print("Number of edges in the horizon:", nedges)
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
                    wi, horizon_v2, horizon_v1,
                    # Previous face id
                    adj_i_f_2, 
                    adj_i_f_1,
                    # Next face id
                    adj_i_f_0
                )
                if dist2 <= 0:
                    # Unrecoverable numerical issue
                    nearest_i_f = -1
                    print("EPA: Unrecoverable numerical issue, stopping.")
                    # break

                    # We do not insert into the map, because the face
                    # is degenerate face and only needed for keeping topology
                    break

                if (dist2 >= lower2) and (dist2 <= upper2):
                    # Store face in the map
                    nfaces_map = self.polytope[i_b].nfaces_map
                    self.polytope_faces_map[i_b][nfaces_map] = i_f0
                    self.polytope_faces[i_b, i_f0].map_idx = nfaces_map
                    self.polytope[i_b].nfaces_map += 1

            # for i in range(self.polytope[i_b].nverts):
            #     print("Polytope vert", i, ":", self.polytope_verts[i_b, i].mink)
            # for i in range(self.polytope[i_b].nfaces):
            #     print("Polytope face", i, ":", self.polytope_faces[i_b, i].verts_idx, "adj_idx:", self.polytope_faces[i_b, i].adj_idx, f"dist2: {self.polytope_faces[i_b, i].dist2:.15f}")
            # for i in range(self.polytope[i_b].nfaces_map):
            #     print("Polytope face map", i, ":", self.polytope_faces_map[i_b][i])

            # Clear the horizon data for the next iteration
            self.polytope[i_b].horizon_nedges = 0

            if (self.polytope[i_b].nfaces_map == 0) or (nearest_i_f == -1):
                # No face candidate left
                print("EPA: No face candidate left, stopping.")
                # print("Nearest face index:", nearest_i_f)
                break

        #print("EPA finished with nearest face:", nearest_i_f)
        if nearest_i_f != -1:
            # Nearest face found
            dist2 = self.polytope_faces[i_b, nearest_i_f].dist2
            self.func_epa_witness(i_ga, i_gb, i_b, nearest_i_f)
            self.num_witness[i_b] = 1
            self.distance[i_b] = -ti.sqrt(dist2)
        else:
            # No face found, so the objects are not colliding
            self.num_witness[i_b] = 0
            self.distance[i_b] = 0
        
        return nearest_i_f

    @ti.func
    def func_epa_witness(self, i_ga, i_gb, i_b, i_f):
        '''
        Compute the witness points from the geometries
        for the face i_f of the polytope.
        '''
        # Find the affine coordinates of the origin's 
        # projection on the face i_f
        face = self.polytope_faces[i_b, i_f]
        face_v1 = self.polytope_verts[i_b, face.verts_idx[0]].mink
        face_v2 = self.polytope_verts[i_b, face.verts_idx[1]].mink
        face_v3 = self.polytope_verts[i_b, face.verts_idx[2]].mink
        face_normal = face.normal

        print("Witness v1:", f"{face_v1.x:.20g}", f"{face_v1.y:.20g}", f"{face_v1.z:.20g}")
        print("Witness v2:", f"{face_v2.x:.20g}", f"{face_v2.y:.20g}", f"{face_v2.z:.20g}")
        print("Witness v3:", f"{face_v3.x:.20g}", f"{face_v3.y:.20g}", f"{face_v3.z:.20g}")
        print("Witness face normal:", f"{face_normal.x:.20g}", f"{face_normal.y:.20g}", f"{face_normal.z:.20g}")
        
        _lambda = self.func_triangle_affine_coords(
            face_normal,
            face_v1, 
            face_v2, 
            face_v3,
        )
        print("Witness lambda:", f"{_lambda[0]:.20g}", f"{_lambda[1]:.20g}", f"{_lambda[2]:.20g}")

        # Point on geom 1
        v1 = self.polytope_verts[i_b, face.verts_idx[0]].obj1
        v2 = self.polytope_verts[i_b, face.verts_idx[1]].obj1
        v3 = self.polytope_verts[i_b, face.verts_idx[2]].obj1
        witness1 = v1 * _lambda[0] + v2 * _lambda[1] + v3 * _lambda[2]
        print("Witness point on geom 1:", f"{witness1.x:.20g}", f"{witness1.y:.20g}", f"{witness1.z:.20g}")

        # Point on geom 2
        v1 = self.polytope_verts[i_b, face.verts_idx[0]].obj2
        v2 = self.polytope_verts[i_b, face.verts_idx[1]].obj2
        v3 = self.polytope_verts[i_b, face.verts_idx[2]].obj2
        witness2 = v1 * _lambda[0] + v2 * _lambda[1] + v3 * _lambda[2]
        print("Witness point on geom 2:", f"{witness2.x:.20g}", f"{witness2.y:.20g}", f"{witness2.z:.20g}")

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
        '''
        Compute the horizon, which represents the area of the polytope
        that is visible from the vertex w, and thus should be deleted
        for the expansion of the polytope.
        '''
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
            is_visible = (face.normal.dot(w) - face.dist2 > self.FLOAT_MIN)

            # The first face is always visible.
            is_visible = is_visible or is_first

            # print("Checking face", i_f, "visibility:", is_visible)

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
                    # print("Checking adjacent face", adj_face_idx, "is deleted:", adj_face_is_deleted)
                    if not adj_face_is_deleted:
                        # Get the related edge id from the adjacent face.
                        # Since adjacent faces have different orientations,
                        # we need to use the ending vertex of the edge.
                        start_vert_idx = face.verts_idx[(i_e2 + 1) % 3]
                        adj_edge_idx = self.func_get_edge_idx(i_b, adj_face_idx, start_vert_idx)

                        self.polytope_horizon_stack[i_b, top].face_idx = adj_face_idx
                        self.polytope_horizon_stack[i_b, top].edge_idx = adj_edge_idx
                        top += 1

                        # print("Adding edge to stack:", adj_face_idx, adj_edge_idx)

            else:

                # If not visible, add the edge to the horizon.
                flag = self.func_add_edge_to_horizon(i_b, i_f, i_e)
                if flag:
                    # If the edges do not form a closed loop,
                    # there is an error in the algorithm.
                    break

            is_first = False

        # Sanity check: the first and the last edges in the horizon
        # should be connected to each other for a closed loop.
        # if flag == 0:
        #     face_idx_0 = self.polytope_horizon_data[i_b, 0].face_idx
        #     edge_idx_0 = self.polytope_horizon_data[i_b, 0].edge_idx
        #     face_idx_1 = self.polytope_horizon_data[i_b, self.polytope[i_b].horizon_nedges - 1].face_idx
        #     edge_idx_1 = self.polytope_horizon_data[i_b, self.polytope[i_b].horizon_nedges - 1].edge_idx
        #     beg_vert_idx = self.polytope_faces[i_b, face_idx_0].verts_idx[edge_idx_0]
        #     end_vert_idx = self.polytope_faces[i_b, face_idx_1].verts_idx[(edge_idx_1 + 1) % 3]
            
        #     if beg_vert_idx != end_vert_idx:
        #         # print("Horizon edges do not form a closed loop :", beg_vert_idx, end_vert_idx)
        #         flag = 1

        return flag
    
    @ti.func
    def func_add_edge_to_horizon(self, i_b, i_f, i_e):
        horizon_nedges = self.polytope[i_b].horizon_nedges
        self.polytope_horizon_data[i_b, horizon_nedges].edge_idx = i_e
        self.polytope_horizon_data[i_b, horizon_nedges].face_idx = i_f
        self.polytope[i_b].horizon_nedges += 1
        # print("Adding edge to horizon:", horizon_nedges, i_f, i_e)

        flag = 0
        # Sanity check: the edges should form a closed loop
        # if horizon_nedges > 0:
        #     prev_i_f = self.polytope_horizon_data[i_b, horizon_nedges - 1].face_idx
        #     prev_i_e = self.polytope_horizon_data[i_b, horizon_nedges - 1].edge_idx

        #     curr_beg_vert_idx = self.polytope_faces[i_b, i_f].verts_idx[i_e]
        #     prev_end_vert_idx = self.polytope_faces[i_b, prev_i_f].verts_idx[(prev_i_e + 1) % 3]
        #     if curr_beg_vert_idx != prev_end_vert_idx:
        #         # print("Horizon edges do not form a closed loop while add :", curr_beg_vert_idx, prev_end_vert_idx)
        #         flag = 1
        return flag

    @ti.func
    def func_get_edge_idx(self, i_b, i_f, i_v):
        '''
        Get the edge index from the face, starting from the vertex i_v.
        If the face is comprised of [v1, v2, v3], the edges are:
        [v1, v2], [v2, v3], [v3, v1].

        Therefore, if i_v is v1, the edge index is 0,
        and if i_v is v2, the edge index is 1,
        and if i_v is v3, the edge index is 2.
        '''
        verts = self.polytope_faces[i_b, i_f].verts_idx
        ret = 2
        if verts[0] == i_v:
            ret = 0
        elif verts[1] == i_v:
            ret = 1
        return ret

    @ti.func
    def func_delete_face_from_polytope(self, i_b, i_f):
        '''
        Delete the face from the polytope.
        '''
        # print("Deleting face", i_f)
        # print("Before deletion")
        # for i in range(self.polytope[i_b].nfaces_map):
        #     print("Face map", i, ":", self.polytope_faces_map[i_b][i])
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

        # print("After deletion")
        # for i in range(self.polytope[i_b].nfaces_map):
        #     print("Face map", i, ":", self.polytope_faces_map[i_b][i])

    @ti.func
    def func_epa_insert_vertex_to_polytope(self, i_b, obj1_point, obj2_point, minkowski_point):
        '''
        Copy vertex information into the polytope.
        '''
        n = self.polytope[i_b].nverts
        self.polytope_verts[i_b, n].obj1 = obj1_point
        self.polytope_verts[i_b, n].obj2 = obj2_point
        self.polytope_verts[i_b, n].mink = minkowski_point
        self.polytope[i_b].nverts += 1
        return n
    
    @ti.func
    def func_epa_init_polytope_2d(self, i_ga, i_gb, i_b):
        '''
        Create the polytope for EPA from a 1-simplex (line segment).
        Return 0 when successful.
        '''
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
                self.gjk_simplex_vertex[i_b, i].mink
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
            [[vi[0], vi[2], vi[3]],
             [vi[0], vi[4], vi[2]],
             [vi[0], vi[3], vi[4]],
             [vi[1], vi[3], vi[2]],
             [vi[1], vi[2], vi[4]],
            [vi[1], vi[4], vi[3]]],
            dt=ti.i32
        )

        # Adjacent face indices for the faces in the hexahedron
        i_as = ti.Matrix(
            [[1, 3, 2],
             [2, 4, 0],
             [0, 5, 1],
             [5, 0, 4],
             [3, 1, 5],
            [4, 2, 3]],
            dt=ti.i32
        )

        for i in range(6):
            i_v1, i_v2, i_v3 = i_vs[i, 0], i_vs[i, 1], i_vs[i, 2]
            i_a1, i_a2, i_a3 = i_as[i, 0], i_as[i, 1], i_as[i, 2]

            if self.func_attach_face_to_polytope(i_b, i_v1, i_v2, i_v3, i_a1, i_a2, i_a3) < self.FLOAT_MIN_SQ:
                self.func_replace_simplex_3(i_b, i_v1, i_v2, i_v3)
                flag = EPA_P2_FALLBACK3 # self.func_epa_init_polytope_3d(i_ga, i_gb, i_b)
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
        '''
        Create the polytope for EPA from a 2-simplex (triangle).
        Return 0 when successful.
        '''
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
                self.gjk_simplex_vertex[i_b, i].mink
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
            if self.gjk_simplex[i_b].dist > 10 * self.FLOAT_MIN and \
                (not tets_has_origin[0]) and (not tets_has_origin[1]):
                flag = EPA_P3_MISSING_ORIGIN
            else:
                # Vertex indices for the faces in the hexahedron
                i_vs = ti.Matrix(
                    [[vi[3], vi[0], vi[1]],
                    [vi[3], vi[2], vi[0]],
                    [vi[3], vi[1], vi[2]],
                    [vi[4], vi[1], vi[0]],
                    [vi[4], vi[0], vi[2]],
                    [vi[4], vi[2], vi[1]]],
                    dt=ti.i32
                )

                # Adjacent face indices for the faces in the hexahedron
                i_as = ti.Matrix(
                    [[1, 3, 2],
                    [2, 4, 0],
                    [0, 5, 1],
                    [5, 0, 4],
                    [3, 1, 5],
                    [4, 2, 3]],
                    dt=ti.i32
                )

                # Build hexahedron (6 faces) from the five vertices.
                for i in range(6):
                    i_v1, i_v2, i_v3 = i_vs[i, 0], i_vs[i, 1], i_vs[i, 2]
                    i_a1, i_a2, i_a3 = i_as[i, 0], i_as[i, 1], i_as[i, 2]

                    dist2 = self.func_attach_face_to_polytope(
                        i_b, i_v1, i_v2, i_v3, i_a1, i_a2, i_a3
                    )
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
        '''
        Create the polytope for EPA from a 3-simplex (tetrahedron).
        Return 0 when successful.
        '''
        flag = 0
        
        # Insert simplex vertices into the polytope
        vi = ti.Vector([0, 0, 0, 0], dt=ti.i32)
        for i in range(4):
            vi[i] = self.func_epa_insert_vertex_to_polytope(
                i_b,
                self.gjk_simplex_vertex[i_b, i].obj1,
                self.gjk_simplex_vertex[i_b, i].obj2,
                self.gjk_simplex_vertex[i_b, i].mink
            )
        
        # If origin is on any face of the tetrahedron,
        # replace the simplex with a 2-simplex (triangle)

        # Vertex indices for the faces in the hexahedron
        i_vs = ti.Matrix(
            [[vi[0], vi[1], vi[2]],
            [vi[0], vi[3], vi[1]],
            [vi[0], vi[2], vi[3]],
            [vi[3], vi[2], vi[1]]],
            dt=ti.i32
        )

        # Adjacent face indices for the faces in the hexahedron
        i_as = ti.Matrix(
            [[1, 3, 2],
            [2, 3, 0],
            [0, 3, 1],
            [2, 0, 1]],
            dt=ti.i32
        )

        for i in range(4):
            v1, v2, v3 = i_vs[i, 0], i_vs[i, 1], i_vs[i, 2]
            a1, a2, a3 = i_as[i, 0], i_as[i, 1], i_as[i, 2]

            dist2 = self.func_attach_face_to_polytope(i_b, v1, v2, v3, a1, a2, a3)

            if dist2 < self.FLOAT_MIN_SQ:
                self.func_replace_simplex_3(i_b, v1, v2, v3)
                flag = EPA_P4_FALLBACK3 # self.func_epa_init_polytope_3d(i_ga, i_gb, i_b)
                break
            
        if flag == 0:
            # If the tetrahedron does not contain the origin,
            # we do not proceed anymore.
            v1, v2, v3, v4 = vi[0], vi[1], vi[2], vi[3]
            if not self.func_origin_tetra_intersection(
                self.polytope_verts[i_b, v1].mink,
                self.polytope_verts[i_b, v2].mink,
                self.polytope_verts[i_b, v3].mink,
                self.polytope_verts[i_b, v4].mink
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
        '''
        Find support points on the two objects using [dir].
        [dir] should be a unit vector from [ga] (obj1) to [gb] (obj2).
        After finding them, insert them into the polytope.
        '''
        d = gs.ti_vec3(1, 0, 0)
        if dir_norm > self.FLOAT_MIN:
            d = dir / dir_norm
            
        support_point_obj1, support_point_obj2, support_point_minkowski = \
            self.func_support(i_ga, i_gb, i_b, d)
            
        # Insert the support points into the polytope
        v_index = self.func_epa_insert_vertex_to_polytope(
            i_b, support_point_obj1, support_point_obj2, support_point_minkowski
        )
        
        return v_index
    
    @ti.func
    def func_attach_face_to_polytope(self, i_b, i_v1, i_v2, i_v3, i_a1, i_a2, i_a3):
        '''
        Attach a face to the polytope.
        [i_v1, i_v2, i_v3] are the vertices of the face,
        [i_a1, i_a2, i_a3] are the adjacent faces.
        
        Also return the squared distance of the face to the origin.
        '''
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
            self.polytope_verts[i_b, i_v1].mink
        )
        print("project origin to plane result:", ret)
        print("Face verts 1: ", f"{self.polytope_verts[i_b, i_v1].mink.x:.20g}",
              f"{self.polytope_verts[i_b, i_v1].mink.y:.20g}", 
              f"{self.polytope_verts[i_b, i_v1].mink.z:.20g}")
        print("Face verts 2: ", f"{self.polytope_verts[i_b, i_v2].mink.x:.20g}",
                f"{self.polytope_verts[i_b, i_v2].mink.y:.20g}", 
                f"{self.polytope_verts[i_b, i_v2].mink.z:.20g}")
        print("Face verts 3: ", f"{self.polytope_verts[i_b, i_v3].mink.x:.20g}",
                f"{self.polytope_verts[i_b, i_v3].mink.y:.20g}", 
                f"{self.polytope_verts[i_b, i_v3].mink.z:.20g}")
        print("Face normal:", self.polytope_faces[i_b, n].normal)
        if not ret:
            normal = self.polytope_faces[i_b, n].normal
            self.polytope_faces[i_b, n].dist2 = normal.dot(normal)
            self.polytope_faces[i_b, n].map_idx = -1  # No map index yet
            flag = self.polytope_faces[i_b, n].dist2
            
        return flag
    
    @ti.func
    def func_replace_simplex_3(self, i_b, i_v1, i_v2, i_v3):
        '''
        Replace the simplex with a 2-simplex (triangle) from polytope vertices.
        [i_v1, i_v2, i_v3] are the vertices that we will use from the polytope.
        '''
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
        '''
        Check if the ray intersects the triangle.
        Return Non-Zero value if it does, otherwise return Zero.
        '''
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
        '''
        Check if the point is inside the triangle.
        '''
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
        '''
        Compute the affine coordinates of the point with respect to the triangle.
        '''
        # Compute minors of the triangle vertices
        m_1 = tri_v2[1]*tri_v3[2] - tri_v2[2]*tri_v3[1] \
            - tri_v1[1]*tri_v3[2] + tri_v1[2]*tri_v3[1] \
            + tri_v1[1]*tri_v2[2] - tri_v1[2]*tri_v2[1]
        
        m_2 = tri_v2[0]*tri_v3[2] - tri_v2[2]*tri_v3[0] \
            - tri_v1[0]*tri_v3[2] + tri_v1[2]*tri_v3[0] \
            + tri_v1[0]*tri_v2[2] - tri_v1[2]*tri_v2[0]
        
        m_3 = tri_v2[0]*tri_v3[1] - tri_v2[1]*tri_v3[0] \
            - tri_v1[0]*tri_v3[1] + tri_v1[1]*tri_v3[0] \
            + tri_v1[0]*tri_v2[1] - tri_v1[1]*tri_v2[0]
            
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
        C1 = point[x] * tri_v2[y] + point[y] * tri_v3[x] + tri_v2[x] * tri_v3[y] - \
            point[x] * tri_v3[y] - point[y] * tri_v2[x] - tri_v3[x] * tri_v2[y]
        
        # C2 corresponds to the signed area of 2-simplex (triangle): (point, tri_v1, tri_v3)
        C2 = point[x] * tri_v3[y] + point[y] * tri_v1[x] + tri_v3[x] * tri_v1[y] - \
            point[x] * tri_v1[y] - point[y] * tri_v3[x] - tri_v1[x] * tri_v3[y]
            
        # C3 corresponds to the signed area of 2-simplex (triangle): (point, tri_v1, tri_v2)
        C3 = point[x] * tri_v1[y] + point[y] * tri_v2[x] + tri_v1[x] * tri_v2[y] - \
            point[x] * tri_v2[y] - point[y] * tri_v1[x] - tri_v2[x] * tri_v1[y]
            
        # Affine coordinates are computed as:
        # [ l1, l2, l3 ] = [ C1 / m_max, C2 / m_max, C3 / m_max ]
        _lambda = gs.ti_vec3(0, 0, 0)
        _lambda[0] = C1 / m_max
        _lambda[1] = C2 / m_max
        _lambda[2] = C3 / m_max
        
        return _lambda
    
    @ti.func
    def func_origin_tetra_intersection(self, tet_v1, tet_v2, tet_v3, tet_v4):
        '''
        Check if the origin is inside the tetrahedron.
        '''
        flag = 1
        for i in range(4):
            v1, v2, v3, v4 = tet_v1, tet_v2, tet_v3, tet_v4
            if i == 1:
                v1, v2, v3, v4 = tet_v2, tet_v3, tet_v4, tet_v1
            elif i == 2:
                v1, v2, v3, v4 = tet_v3, tet_v4, tet_v1, tet_v2
            elif i == 3:
                v1, v2, v3, v4 = tet_v4, tet_v1, tet_v2, tet_v3
            flag = (flag and self.func_point_plane_same_side(v1, v2, v3, v4))
            if not flag:
                break
        return flag
    
    @ti.func
    def func_point_plane_same_side(self, point, plane_v1, plane_v2, plane_v3):
        '''
        Check if the point is on the same side of the plane as the origin.
        '''
        # Compute the normal of the plane
        edge1 = plane_v2 - plane_v1
        edge2 = plane_v3 - plane_v1
        normal = edge1.cross(edge2)
        
        diff1 = point - plane_v1
        dot1 = normal.dot(diff1)
        
        diff2 = -plane_v1       # origin - plane_v1
        dot2 = normal.dot(diff2)

        flag = 1 if dot1 * dot2 > 0 else 0
        return flag
    
    '''
    Helpers
    '''
    @ti.func
    def func_is_discrete_geoms(self, i_ga, i_gb, i_b):
        '''
        Check if the given geoms are discrete geometries.
        '''
        geom_type_a = self._solver.geoms_info[i_ga].type
        geom_type_b = self._solver.geoms_info[i_gb].type

        res = False
        if (geom_type_a == gs.GEOM_TYPE.BOX or geom_type_a == gs.GEOM_TYPE.MESH) and \
        (geom_type_b == gs.GEOM_TYPE.BOX or geom_type_b == gs.GEOM_TYPE.MESH):
            res = True
        
        return res

    @ti.func
    def func_support(self, i_ga, i_gb, i_b, dir):
        '''
        Find support points on the two objects using [dir].
        [dir] should be a unit vector from [ga] (obj1) to [gb] (obj2).
        '''
        support_point_obj1 = gs.ti_vec3(0, 0, 0)
        support_point_obj2 = gs.ti_vec3(0, 0, 0)
        for i in range(2):
            d = dir if i == 0 else -dir
            i_g = i_ga if i == 0 else i_gb

            sp = self.support_driver(d, i_g, i_b)
            if i == 0:
                support_point_obj1 = sp
            else:
                support_point_obj2 = sp
        support_point_minkowski = support_point_obj1 - support_point_obj2
    
        return support_point_obj1, support_point_obj2, support_point_minkowski
    
    @ti.func
    def func_rotmat_120(self, axis):
        '''
        Rotation matrix for 120 degrees rotation around the given axis.
        '''
        n = ti.math.length(axis)
        u1 = axis[0] / n
        u2 = axis[1] / n
        u3 = axis[2] / n
        
        # sin and cos of 120 degrees
        sin = 0.86602540378
        cos = -0.5
        
        mat = ti.math.mat3(0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0)
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
        '''
        Project the origin onto the plane defined by the simplex vertices.
        Find the projected point and return flag with it.
        '''
        point, flag = gs.ti_vec3(0, 0, 0), -1
        
        d21 = v2 - v1
        d31 = v3 - v1
        d32 = v3 - v2

        # print("v1:", v1, "v2:", v2, "v3:", v3)
        # print("d21:", d21, "d31:", d31, "d32:", d32)
        # print("normalized d21:", d21.normalized(), "normalized d31:", d31.normalized(), "normalized d32:", d32.normalized())
        # print("n1: ", d21.cross(d31), "n2:", d32.cross(d21), "n3:", d31.cross(d32))
        
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
        '''
        Project the origin onto the line defined by the simplex vertices.

        P = v2 - ((v1 * diff) / (diff * diff)) * diff
        '''
        diff = v2 - v1
        k = v2.dot(diff) / diff.dot(diff)
        P = v2 - k * diff

        return P

    @ti.func
    def func_simplex_vertex_linear_comb(self, i_b, i_v, i_s1, i_s2, i_s3, i_s4, _lambda, n):
        '''
        Compute the linear combination of the simplex vertices
        
        @ i_v: Which vertex to use (0: obj1, 1: obj2, 2: minkowski)
        @ n: Number of vertices to combine, combine the first n vertices
        '''
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
            # print("s1:", f"{s1[0]:.20g}", f"{s1[1]:.20g}", f"{s1[2]:.20g}")
            # print("s2:", f"{s2[0]:.20g}", f"{s2[1]:.20g}", f"{s2[2]:.20g}")
        elif n == 3:
            res = s1 * c1 + s2 * c2 + s3 * c3
        else:
            res = s1 * c1 + s2 * c2 + s3 * c3 + s4 * c4
        return res
    
    @ti.func
    def func_det3(self, v1, v2, v3):
        '''
        Compute the determinant of a 3x3 matrix formed by the vectors v1, v2, v3.
        M = [v1 | v2 | v3]
        '''
        return (
            v1[0] * (v2[1] * v3[2] - v2[2] * v3[1]) -
            v1[1] * (v2[0] * v3[2] - v2[2] * v3[0]) +
            v1[2] * (v2[0] * v3[1] - v2[1] * v3[0])
        )
        
    @ti.func
    def func_compare_sign(self, a, b):
        '''
        Compare the sign of two values.
        '''
        ret = 0
        if a > 0 and b > 0:
            ret = 1
        elif a < 0 and b < 0:
            ret = -1
        return ret
    
    @ti.func
    def func_is_equal_vec3(self, a, b):
        '''
        Check if two vectors are equal within a small tolerance.
        '''
        diff = ti.abs(a - b)
        amax = ti.max(ti.abs(a), ti.abs(b))
        return ((diff < self.FLOAT_MIN) + (diff < amax * self.FLOAT_MIN)).all()
    
    
    
    @ti.func
    def gjk_support_geom(self, direction, i_g, i_b):
        support_pt = self.support_driver(direction, i_g, i_b)
        dist = ti.math.dot(support_pt, direction)
        return dist, support_pt

    @ti.func
    def gjk_compute_support(self, direction, i_ga, i_gb, i_b):
        dist1, dist2 = 0.0, 0.0
        s1, s2 = gs.ti_vec3(0, 0, 0), gs.ti_vec3(0, 0, 0)
        for i in range(2):
            d = direction if i == 0 else -direction
            i_g = i_ga if i == 0 else i_gb
            
            dist, s = self.gjk_support_geom(d, i_g, i_b)
            if i == 0:
                dist1, s1 = dist, s
            else:
                dist2, s2 = dist, s
        
        support_pt = s1 - s2
        return dist1 + dist2, s1, s2, support_pt
    
    # @TODO: do we need this?
    @ti.func
    def gjk_normalize(self, a):
        norm = ti.math.length(a)
        b, success = a, 0
        if norm > float(1e-8) and norm < float(1e12):
            b = a / norm
            success = 1
        return b, success
        
    @ti.func
    def gjk_orthonormal(self, normal):
        dir = ti.math.vec3(0.0, 0.0, 0.0)
        if ti.abs(normal[0]) < ti.abs(normal[1]) and ti.abs(normal[0]) < ti.abs(normal[2]):
            dir = ti.math.vec3(1.0 - normal[0] * normal[0], -normal[0] * normal[1], -normal[0] * normal[2])
        elif ti.abs(normal[1]) < ti.abs(normal[2]):
            dir = ti.math.vec3(-normal[1] * normal[0], 1.0 - normal[1] * normal[1], -normal[1] * normal[2])
        else:
            dir = ti.math.vec3(-normal[2] * normal[0], -normal[2] * normal[1], 1.0 - normal[2] * normal[2])
        dir, _ = self.gjk_normalize(dir)
        return dir

    def reset(self):
        pass

    @ti.func
    def support_sphere(self, direction, i_g, i_b):
        sphere_center = self._solver.geoms_state[i_g, i_b].pos
        sphere_radius = self._solver.geoms_info[i_g].data[0]
        print("Sphere center:", f"{sphere_center[0]:.20g}", f"{sphere_center[1]:.20g}", f"{sphere_center[2]:.20g}")
        print("Sphere direction:", f"{direction[0]:.20g}", f"{direction[1]:.20g}", f"{direction[2]:.20g}")
        return sphere_center + direction * sphere_radius

    @ti.func
    def support_ellipsoid(self, direction, i_g, i_b):
        g_state = self._solver.geoms_state[i_g, i_b]
        ellipsoid_center = g_state.pos
        ellipsoid_scaled_axis = ti.Vector(
            [
                self._solver.geoms_info[i_g].data[0] ** 2,
                self._solver.geoms_info[i_g].data[1] ** 2,
                self._solver.geoms_info[i_g].data[2] ** 2,
            ],
            dt=gs.ti_float,
        )
        ellipsoid_scaled_axis = gu.ti_transform_by_quat(ellipsoid_scaled_axis, g_state.quat)
        dist = ellipsoid_scaled_axis / ti.sqrt(direction.dot(1.0 / ellipsoid_scaled_axis))
        return ellipsoid_center + direction * dist

    @ti.func
    def support_capsule(self, direction, i_g, i_b):
        g_state = self._solver.geoms_state[i_g, i_b]
        capule_center = g_state.pos
        capsule_axis = gu.ti_transform_by_quat(ti.Vector([0.0, 0.0, 1.0], dt=gs.ti_float), g_state.quat)
        capule_radius = self._solver.geoms_info[i_g].data[0]
        capule_halflength = 0.5 * self._solver.geoms_info[i_g].data[1]
        capule_endpoint_side = ti.math.sign(direction.dot(capsule_axis))
        if capule_endpoint_side == 0.0:
            capule_endpoint_side = 1.0
        capule_endpoint = capule_center + capule_halflength * capule_endpoint_side * capsule_axis
        return capule_endpoint + direction * capule_radius

    # @ti.func
    # def support_prism(self, direction, i_g, i_b):
    #     ibest = 0
    #     best = self._solver.collider.prism[ibest, i_b].dot(direction)
    #     for i in range(1, 6):
    #         dot = self._solver.collider.prism[i, i_b].dot(direction)
    #         if dot > best:
    #             ibest = i
    #             best = dot

    #     return self._solver.collider.prism[ibest, i_b], ibest

    @ti.func
    def support_prism(self, direction, i_g, i_b):
        istart = 3
        if direction[2] < 0:
            istart = 0

        ibest = istart
        best = self._solver.collider.prism[istart, i_b].dot(direction)
        for i in range(istart + 1, istart + 3):
            dot = self._solver.collider.prism[i, i_b].dot(direction)
            if dot > best:
                ibest = i
                best = dot

        return self._solver.collider.prism[ibest, i_b], ibest

    @ti.func
    def support_box(self, direction, i_g, i_b):
        g_state = self._solver.geoms_state[i_g, i_b]
        d_box = gu.ti_transform_by_quat(direction, gu.ti_inv_quat(g_state.quat))
        d_box_sign = ti.math.sign(d_box)
        for i in range(3):
            if d_box_sign[i] == 0.0:
                d_box_sign[i] = 1.0

        vid = (d_box[0] > 0) * 4 + (d_box[1] > 0) * 2 + (d_box[2] > 0) * 1
        v_ = ti.Vector(
            [
                d_box_sign[0] * self._solver.geoms_info[i_g].data[0] * 0.5,
                d_box_sign[1] * self._solver.geoms_info[i_g].data[1] * 0.5,
                d_box_sign[2] * self._solver.geoms_info[i_g].data[2] * 0.5,
            ],
            dt=gs.ti_float,
        )
        vid += self._solver.geoms_info[i_g].vert_start
        v = gu.ti_transform_by_trans_quat(v_, g_state.pos, g_state.quat)
        return v, vid

    @ti.func
    def support_mesh(self, direction, i_g, i_b):
        g_state = self._solver.geoms_state[i_g, i_b]
        d_mesh = gu.ti_transform_by_quat(direction, gu.ti_inv_quat(g_state.quat))
        
        # Exhaustively search for the vertex with maximum dot product
        fmax = -self.FLOAT_MAX
        imax = 0

        vert_start = self._solver.geoms_info.vert_start[i_g]
        vert_end = self._solver.geoms_info.vert_end[i_g]
    
        for i in range(vert_start, vert_end):
            pos = self._solver.verts_info[i].init_pos
            vdot = d_mesh.dot(pos)
            if vdot > fmax:
                fmax = vdot
                imax = i
        
        v = self._solver.verts_info[imax].init_pos
        vid = imax
        print("Support mesh dir: ", f"{d_mesh[0]:.20g}", f"{d_mesh[1]:.20g}", f"{d_mesh[2]:.20g}")
        print("Support mesh vertex: ", f"{v[0]:.20g}", f"{v[1]:.20g}", f"{v[2]:.20g}")
        
        v_ = gu.ti_transform_by_trans_quat(v, g_state.pos, g_state.quat)
        print("Support mesh vertex (global):", f"{v_[0]:.20g}", f"{v_[1]:.20g}", f"{v_[2]:.20g}")
        return v_, vid

    @ti.func
    def support_driver(self, direction, i_g, i_b):
        v = ti.Vector.zero(gs.ti_float, 3)
        geom_type = self._solver.geoms_info[i_g].type
        print("Support driver dir: ", f"{direction[0]:.20g}", f"{direction[1]:.20g}", f"{direction[2]:.20g}")
        if geom_type == gs.GEOM_TYPE.SPHERE:
            v = self.support_sphere(direction, i_g, i_b)
        elif geom_type == gs.GEOM_TYPE.ELLIPSOID:
            v = self.support_ellipsoid(direction, i_g, i_b)
        elif geom_type == gs.GEOM_TYPE.CAPSULE:
            v = self.support_capsule(direction, i_g, i_b)
        elif geom_type == gs.GEOM_TYPE.BOX:
            v, _ = self.support_box(direction, i_g, i_b)
        elif geom_type == gs.GEOM_TYPE.TERRAIN:
            if ti.static(self._solver.collider._has_terrain):
                v, _ = self.support_prism(direction, i_g, i_b)
        elif geom_type == gs.GEOM_TYPE.MESH:
            v, _ = self.support_mesh(direction, i_g, i_b)
        else:
            v, _ = self.support_field._func_support_world(direction, i_g, i_b)
        return v