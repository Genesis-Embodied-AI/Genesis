import numpy as np
import taichi as ti
import torch

import genesis as gs
import genesis.utils.geom as gu

from .support_field_decomp import SupportField
# @TODO: type checking for float, int, etc.

@ti.data_oriented
class GJKEPA:
    def __init__(self, rigid_solver):
        self._solver = rigid_solver
        self._max_contact_pairs = rigid_solver._max_collision_pairs
        self._B = rigid_solver._B
        self._para_level = rigid_solver._para_level
        
        #@TODO: remove [FLOAT_MAX]
        self.FLOAT_MAX = float(1e30)
        #@TODO: make this a parameter
        self.gjk_iterations = 1
        self.epa_depth_extension = 0.1
        self.epa_exact_neg_distance = False
        self.epa_iterations = 12
        
        ### constants
        self.EPS_BEST_COUNT = 12
        self.MULTI_CONTACT_COUNT = 4
        self.MULTI_POLYGON_COUNT = 8
        self.MULTI_TILT_ANGLE = 1.0
        self.TRIS_DIM = 3 * self.EPS_BEST_COUNT
        
        ### taichi fields
        self.gjk_simplex = ti.Vector.field(n=3, dtype=gs.ti_float, shape=(self._B, 4))
        self.gjk_normal = ti.Vector.field(n=3, dtype=gs.ti_float, shape=(self._B,))
        self.gjk_plane = ti.Vector.field(n=3, dtype=gs.ti_float, shape=(self._B, 4))
        
        self.epa_veci1 = ti.Vector.field(n=6, dtype=gs.ti_int, shape=())
        self.epa_veci2 = ti.Vector.field(n=6, dtype=gs.ti_int, shape=())
        self.epa_veci1[None] = ti.Vector([0, 0, 0, 1, 1, 2], dt=gs.ti_int)
        self.epa_veci2[None] = ti.Vector([1, 2, 3, 2, 3, 3], dt=gs.ti_int)
        
        self.epa_tris = ti.Vector.field(n=3, dtype=gs.ti_float, shape=(self._B, self.TRIS_DIM * 2))
        self.epa_p = ti.Vector.field(n=3, dtype=gs.ti_float, shape=(self._B, self.EPS_BEST_COUNT))          # supporting points for each triangle
        self.epa_dists = ti.Vector.field(n=self.EPS_BEST_COUNT * 3, dtype=gs.ti_float, shape=(self._B,))    # distance to the origin for candidate triangles
        
        self.epa_depth = ti.field(dtype=gs.ti_float, shape=(self._B,))  # depth of the contact point
        self.epa_normal = ti.Vector.field(n=3, dtype=gs.ti_float, shape=(self._B,))  # normal of the contact point
        
        self.mc_contact_points = ti.Vector.field(n=3, dtype=gs.ti_float, shape=(self._B, self.MULTI_CONTACT_COUNT))
        self.mc_contact_count = ti.field(dtype=gs.ti_int, shape=(self._B,))  # number of contact points for each body
        self.mc_v1 = ti.Vector.field(n=3, dtype=gs.ti_float, shape=(self._B, self.MULTI_POLYGON_COUNT))
        self.mc_v2 = ti.Vector.field(n=3, dtype=gs.ti_float, shape=(self._B, self.MULTI_POLYGON_COUNT))
        self.mc_out = ti.Vector.field(n=3, dtype=gs.ti_float, shape=(self._B, 4))
        
        self.support_field = SupportField(rigid_solver)
        
    @ti.func
    def func_gjk_contact(self, i_ga, i_gb, i_b):
        '''
        GJK algorithm to check collision between two convex geometries.
        '''
        dir = ti.math.vec3(0, 0, 1)
        dir_n = -dir
        depth = self.FLOAT_MAX
        normal = ti.math.vec3(0, 0, 0)

        # test two random directions and choose better one
        dist_max, simplex0 = self.compute_support(dir, i_ga, i_gb, i_b)
        dist_min, simplex1 = self.compute_support(dir_n, i_ga, i_gb, i_b)
        if dist_max < dist_min:
            depth = dist_max
            normal = dir
        else:
            depth = dist_min
            normal = dir_n

        sd = simplex0 - simplex1
        dir = self.gjk_orthonormal(sd)        # find a vector that lies in the plane orthogonal to sd

        dist_max, simplex3 = self.compute_support(dir, i_ga, i_gb, i_b)
        
        # Initialize a 2-simplex with simplex[2]==simplex[1]. This ensures the
        # correct winding order for face normals defined below. Face 0 and face 3
        # are degenerate, and face 1 and 2 have opposing normals.
        self.gjk_simplex[i_b, 0] = simplex0
        self.gjk_simplex[i_b, 1] = simplex1
        self.gjk_simplex[i_b, 2] = simplex1  # simplex[2] == simplex[1]
        self.gjk_simplex[i_b, 3] = simplex3
        
        if dist_max < depth:
            depth = dist_max
            normal = dir
        if dist_min < depth:
            depth = dist_min
            normal = dir_n
        
        for _ in range(self.gjk_iterations):
            # winding orders: plane[0] ccw, plane[1] cw, plane[2] ccw, plane[3] cw
            self.gjk_plane[i_b, 0] = ti.math.cross(
                self.gjk_simplex[i_b, 3] - self.gjk_simplex[i_b, 2],
                self.gjk_simplex[i_b, 1] - self.gjk_simplex[i_b, 2]
            )
            self.gjk_plane[i_b, 1] = ti.math.cross(
                self.gjk_simplex[i_b, 3] - self.gjk_simplex[i_b, 0],
                self.gjk_simplex[i_b, 2] - self.gjk_simplex[i_b, 0]
            )
            self.gjk_plane[i_b, 2] = ti.math.cross(
                self.gjk_simplex[i_b, 3] - self.gjk_simplex[i_b, 1],
                self.gjk_simplex[i_b, 0] - self.gjk_simplex[i_b, 1]
            )
            self.gjk_plane[i_b, 3] = ti.math.cross(
                self.gjk_simplex[i_b, 2] - self.gjk_simplex[i_b, 0],
                self.gjk_simplex[i_b, 1] - self.gjk_simplex[i_b, 0]
            )
            plane0, p0 = self.gjk_normalize(self.gjk_plane[i_b, 0])
            plane1, p1 = self.gjk_normalize(self.gjk_plane[i_b, 1])
            plane2, p2 = self.gjk_normalize(self.gjk_plane[i_b, 2])
            plane3, p3 = self.gjk_normalize(self.gjk_plane[i_b, 3])
            
            self.gjk_plane[i_b, 0] = plane0
            self.gjk_plane[i_b, 1] = plane1
            self.gjk_plane[i_b, 2] = plane2
            self.gjk_plane[i_b, 3] = plane3
            
            # Compute distance of each face halfspace to the origin. If dplane<0, then the
            # origin is outside the halfspace. If dplane>0 then the origin is inside
            # the halfspace defined by the face plane.
            dplane = ti.math.vec4(0, 0, 0, 0)
            dplane.fill(self.FLOAT_MAX)
            
            if p0:
                dplane.x = ti.math.dot(plane0, self.gjk_simplex[i_b, 2])
            if p1:
                dplane.y = ti.math.dot(plane1, self.gjk_simplex[i_b, 0])
            if p2:
                dplane.z = ti.math.dot(plane2, self.gjk_simplex[i_b, 1])
            if p3:
                dplane.w = ti.math.dot(plane3, self.gjk_simplex[i_b, 0])

            # pick plane normal with minimum distance to the origin
            i1 = 0 if dplane[0] < dplane[1] else 1
            i2 = 2 if dplane[2] < dplane[3] else 3
            index = i1 if dplane[i1] < dplane[i2] else i2

            if dplane[index] > 0.0:
                # origin is inside the simplex, objects are intersecting
                break

            # add new support point to the simplex
            iplane = self.gjk_plane[i_b, index]
            dist, simplex_i = self.compute_support(iplane, i_ga, i_gb, i_b)
            self.gjk_simplex[i_b, index] = simplex_i
            
            if dist < depth:
                depth = dist
                normal = self.gjk_plane[i_b, index]
                
            # preserve winding order of the simplex faces
            index1 = (index + 1) & 3
            index2 = (index + 2) & 3
            swap = self.gjk_simplex[i_b, index1]
            self.gjk_simplex[i_b, index1] = self.gjk_simplex[i_b, index2]
            self.gjk_simplex[i_b, index2] = swap

            if dist < 0.0:
                break  # objects are likely non-intersecting

        self.gjk_normal[i_b] = normal
    
    @ti.func
    def func_epa_contact(self, i_ga, i_gb, i_b):
        '''
        EPA algorithm to find exact collision depth and normal between two convex geometries.
        
        @ simplex: 4x3 matrix with simplex vertices
        @ normal: initial normal vector, which is used to compute the support
        '''
        normal = self.gjk_normal[i_b]
        # get the support, if depth < 0: objects do not intersect
        depth, _ = self.compute_support(normal, i_ga, i_gb, i_b)
        
        if depth < -self.epa_depth_extension:
            # Objects are not intersecting, and we do not obtain the closest points as
            # specified by depth_extension.
            depth = ti.math.nan
            normal = ti.math.vec3(ti.math.nan, ti.math.nan, ti.math.nan)
        else:
            if ti.static(self.epa_exact_neg_distance):
                # Check closest points to all edges of the simplex, rather than just the
                # face normals. This gives the exact depth/normal for the non-intersecting
                # case.
                for i in range(6):
                    i1 = self.epa_veci1[None][i]
                    i2 = self.epa_veci2[None][i]

                    si1 = self.gjk_simplex[i_b, i1]
                    si2 = self.gjk_simplex[i_b, i2]

                    if si1[0] != si2[0] or si1[1] != si2[1] or si1[2] != si2[2]:
                        v = si1 - si2
                        alpha = ti.math.dot(si1, v) / ti.math.dot(v, v)

                        # p0 is the closest segment point to the origin
                        p0 = ti.math.clamp(alpha, 0.0, 1.0) * v - si1
                        p0, pf = self.gjk_normalize(p0)

                        if pf:
                            depth2, _ = self.compute_support(p0, i_ga, i_gb, i_b)

                            if depth2 < depth:
                                depth = depth2
                                normal = p0

            simplex0 = self.gjk_simplex[i_b, 0]
            simplex1 = self.gjk_simplex[i_b, 1]
            simplex2 = self.gjk_simplex[i_b, 2]
            simplex3 = self.gjk_simplex[i_b, 3]
            
            self.epa_tris[i_b, 0] = simplex2
            self.epa_tris[i_b, 1] = simplex1
            self.epa_tris[i_b, 2] = simplex3
            
            self.epa_tris[i_b, 3] = simplex0
            self.epa_tris[i_b, 4] = simplex2
            self.epa_tris[i_b, 5] = simplex3
            
            self.epa_tris[i_b, 6] = simplex1
            self.epa_tris[i_b, 7] = simplex0
            self.epa_tris[i_b, 8] = simplex3
            
            self.epa_tris[i_b, 9] = simplex0
            self.epa_tris[i_b, 10] = simplex1
            self.epa_tris[i_b, 11] = simplex2
            
            # Calculate the total number of iterations to avoid nested loop
            # This is a hack to reduce compile time
            count = 4
            it = 0
            for ei in range(self.epa_iterations):
                it += count
                count = min(count * 3, self.EPS_BEST_COUNT)

            count = 4
            i = 0
            for iit in range(it):
                # Loop through all triangles, and obtain distances to the origin for each
                # new triangle candidate.
                Ti = 3 * i
                tris0 = self.epa_tris[i_b, Ti + 0]
                tris1 = self.epa_tris[i_b, Ti + 1]
                tris2 = self.epa_tris[i_b, Ti + 2]
                n = ti.math.cross(tris2 - tris0, tris1 - tris0)
                
                n, nf = self.gjk_normalize(n)
                if not nf:
                    for j in range(3):
                        self.epa_dists[i_b][i * 3 + j] = 2 * self.FLOAT_MAX
                    continue

                dist, pi = self.compute_support(n, i_ga, i_gb, i_b)
                self.epa_p[i_b, i] = pi

                if dist < depth:
                    depth = dist
                    normal = n
                    
                # iterate over edges and get distance using support point
                for j in range(3):
                    pii = self.epa_p[i_b, i]
                    tqj = self.epa_tris[i_b, Ti + j]
                    tqj1 = self.epa_tris[i_b, Ti + ((j + 1) % 3)]
                    tqj2 = self.epa_tris[i_b, Ti + ((j + 2) % 3)]
                    
                    if ti.static(self.epa_exact_neg_distance):
                        # obtain closest point between new triangle edge and origin
                        if (pii[0] != tqj[0]) or (pii[1] != tqj[1]) or (pii[2] != tqj[2]):
                            v = pii - tqj
                            alpha = ti.math.dot(pii, v) / ti.math.dot(v, v)
                            p0 = ti.math.clamp(alpha, 0.0, 1.0) * v - pii
                            p0, pf = self.gjk_normalize(p0)

                            if pf:
                                dist2, v = self.compute_support(p0, i_ga, i_gb, i_b)

                                if dist2 < depth:
                                    depth = dist2
                                    normal = p0
                          
                    plane = ti.math.cross(pii - tqj, tqj1 - tqj)
                    plane, pf = self.gjk_normalize(plane)

                    dd = 0.0
                    if pf:
                        dd = ti.math.dot(plane, tqj)
                    else:
                        dd = self.FLOAT_MAX

                    if (dd < 0 and depth >= 0) or (
                        tqj2[0] == pii[0] and tqj2[1] == pii[1] and tqj2[2] == pii[2]
                    ):
                        self.epa_dists[i_b][i * 3 + j] = self.FLOAT_MAX
                    else:
                        self.epa_dists[i_b][i * 3 + j] = dd

                if i == count - 1:
                    prev_count = count
                    count = min(count * 3, self.EPS_BEST_COUNT)
                    self.epa_expand_polytope(i_b, count, prev_count)
                    i = 0
                else:
                    i += 1
        
        self.epa_depth[i_b] = depth
        self.epa_normal[i_b] = normal
    
    @ti.func
    def epa_expand_polytope(self, i_b, count, prev_count):        
        # expand polytope greedily
        for j in range(count):
            best = 0
            dd = self.epa_dists[i_b][0]
            for i in range(1, 3 * prev_count):
                if self.epa_dists[i_b][i] < dd:
                    dd = self.epa_dists[i_b][i]
                    best = i
               
            self.epa_dists[i_b][best] = 2 * self.FLOAT_MAX

            parent_index = best // 3
            child_index = best % 3
            
            # fill in the new triangle at the next index
            self.epa_tris[i_b, self.TRIS_DIM + j * 3 + 0] = self.epa_tris[i_b, parent_index * 3 + child_index]
            self.epa_tris[i_b, self.TRIS_DIM + j * 3 + 1] = self.epa_tris[i_b, parent_index * 3 + ((child_index + 1) % 3)]
            self.epa_tris[i_b, self.TRIS_DIM + j * 3 + 2] = self.epa_p[i_b, parent_index]
        
        for r in range(self.EPS_BEST_COUNT * 3):
            # swap triangles
            swap = self.epa_tris[i_b, self.TRIS_DIM + r]
            self.epa_tris[i_b, self.TRIS_DIM + r] = self.epa_tris[i_b, r]
            self.epa_tris[i_b, r] = swap
        
    @ti.func
    def func_multiple_contacts(self, i_ga, i_gb, i_b):
        '''
        Calculates multiple contact points given the normal from EPA.
         1. Calculates the polygon on each shape by tiling the normal
            "MULTI_TILT_ANGLE" degrees in the orthogonal component of the normal.
            The "MULTI_TILT_ANGLE" can be changed to depend on the depth of the
            contact, in a future version.
         2. The normal is tilted "MULTI_POLYGON_COUNT" times in the directions evenly
           spaced in the orthogonal component of the normal.
           (works well for >= 6, default is 8).
         3. The intersection between these two polygons is calculated in 2D space
           (complement to the normal). If they intersect, extreme points in both
           directions are found. This can be modified to the extremes in the
           direction of eigenvectors of the variance of points of each polygon. If
           they do not intersect, the closest points of both polygons are found.
        '''
        depth = self.epa_depth[i_b]
        normal = self.epa_normal[i_b]
        
        contact_count = 0
        if depth >= -self.epa_depth_extension:
            dir = self.gjk_orthonormal(normal)
            dir2 = ti.math.cross(normal, dir)

            angle = self.MULTI_TILT_ANGLE * ti.math.pi / 180.0
            c = ti.cos(angle)
            s = ti.sin(angle)
            tc = 1.0 - c

            # Obtain points on the polygon determined by the support and tilt angle,
            # in the basis of the contact frame.
            v1count = 0
            v2count = 0
            angle_ratio = 2.0 * ti.math.pi / float(self.MULTI_POLYGON_COUNT)
            
            for i in range(self.MULTI_POLYGON_COUNT):
                angle = angle_ratio * float(i)
                axis = ti.math.cos(angle) * dir + ti.math.sin(angle) * dir2

                # Axis-angle rotation matrix. See
                # https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
                mat0 = c + axis[0] * axis[0] * tc
                mat5 = c + axis[1] * axis[1] * tc
                mat10 = c + axis[2] * axis[2] * tc
                t1 = axis[0] * axis[1] * tc
                t2 = axis[2] * s
                mat4 = t1 + t2
                mat1 = t1 - t2
                t1 = axis[0] * axis[2] * tc
                t2 = axis[1] * s
                mat8 = t1 - t2
                mat2 = t1 + t2
                t1 = axis[1] * axis[2] * tc
                t2 = axis[0] * s
                mat9 = t1 + t2
                mat6 = t1 - t2

                n = ti.math.vec3(
                    mat0 * normal[0] + mat1 * normal[1] + mat2 * normal[2],
                    mat4 * normal[0] + mat5 * normal[1] + mat6 * normal[2],
                    mat8 * normal[0] + mat9 * normal[1] + mat10 * normal[2],
                )
                
                _, p = self.gjk_support_geom(-n, i_ga, i_b)                 # since normal points toward [i_ga], should invert it to get contact point on [i_ga]
                self.mc_v1[i_b, v1count][0] = ti.math.dot(p, dir)
                self.mc_v1[i_b, v1count][1] = ti.math.dot(p, dir2)
                self.mc_v1[i_b, v1count][2] = ti.math.dot(p, normal)

                if i == 0:
                    v1count += 1
                elif self.any_different(self.mc_v1[i_b, v1count], self.mc_v1[i_b, v1count-1]):
                    v1count += 1

                _, p = self.gjk_support_geom(n, i_gb, i_b)
                self.mc_v2[i_b, v2count][0] = ti.math.dot(p, dir)
                self.mc_v2[i_b, v2count][1] = ti.math.dot(p, dir2)
                self.mc_v2[i_b, v2count][2] = ti.math.dot(p, normal)
                
                if i == 0:
                    v2count += 1
                elif self.any_different(self.mc_v2[i_b, v2count], self.mc_v2[i_b, v2count-1]):
                    v2count += 1

            # remove duplicate vertices on the array boundary
            if v1count > 1 and self.all_same(self.mc_v1[i_b, v1count-1], self.mc_v1[i_b, 0]):
                v1count -= 1

            if v2count > 1 and self.all_same(self.mc_v2[i_b, v2count-1], self.mc_v2[i_b, 0]):
                v2count -= 1
                
            # find an intersecting polygon between v1 and v2 in the 2D plane
            candCount = 0
            
            if v2count > 1:
                for i in range(v1count):
                    m1a = self.mc_v1[i_b, i]
                    is_in = 1
                    
                    # check if point m1a is inside the v2 polygon on the 2D plane
                    for j in range(v2count):
                        j2 = (j + 1) % v2count

                        # Checks that orientation of the triangle (v2[j], v2[j2], m1a) is
                        # counter-clockwise. If so, point m1a is inside the v2 polygon.
                        v2j = self.mc_v2[i_b, j]
                        v2j2 = self.mc_v2[i_b, j2]
                        is_in = is_in * ((v2j2[0] - v2j[0]) * (m1a[1] - v2j[1]) - (v2j2[1] - v2j[1]) * (m1a[0] - v2j[0]) >= 0.0)

                        if not is_in:
                            break

                    if is_in:
                        if not candCount or m1a[0] < self.mc_out[i_b, 0][0]:
                            self.mc_out[i_b, 0] = m1a
                        if not candCount or m1a[0] > self.mc_out[i_b, 1][0]:
                            self.mc_out[i_b, 1] = m1a
                        if not candCount or m1a[1] < self.mc_out[i_b, 2][1]:
                            self.mc_out[i_b, 2] = m1a
                        if not candCount or m1a[1] > self.mc_out[i_b, 3][1]:
                            self.mc_out[i_b, 3] = m1a
                        candCount += 1

            if v1count > 1:
                for i in range(v2count):
                    m1a = self.mc_v2[i_b, i]
                    is_in = 1
                    
                    for j in range(v1count):
                        j2 = (j + 1) % v1count
                        
                        v1j = self.mc_v1[i_b, j]
                        v1j2 = self.mc_v1[i_b, j2]
                        is_in = is_in * (v1j2[0] - v1j[0]) * (m1a[1] - v1j[1]) - (v1j2[1] - v1j[1]) * (m1a[0] - v1j[0]) >= 0.0
                        if not is_in:
                            break

                    if is_in:
                        if not candCount or m1a[0] < self.mc_out[i_b, 0][0]:
                            self.mc_out[i_b, 0] = m1a
                        if not candCount or m1a[0] > self.mc_out[i_b, 1][0]:
                            self.mc_out[i_b, 1] = m1a
                        if not candCount or m1a[1] < self.mc_out[i_b, 2][1]:
                            self.mc_out[i_b, 2] = m1a
                        if not candCount or m1a[1] > self.mc_out[i_b, 3][1]:
                            self.mc_out[i_b, 3] = m1a
                        candCount += 1

            if v1count > 1 and v2count > 1:
                # Check all edge pairs, and store line segment intersections if they are
                # on the edge of the boundary.
                for i in range(v1count):
                    for j in range(v2count):
                        m1a = self.mc_v1[i_b, i]
                        m1b = self.mc_v1[i_b, (i + 1) % v1count]
                        m2a = self.mc_v2[i_b, j]
                        m2b = self.mc_v2[i_b, (j + 1) % v2count]

                        det = (m2a[1] - m2b[1]) * (m1b[0] - m1a[0]) - (m1a[1] - m1b[1]) * (m2b[0] - m2a[0])

                        if ti.abs(det) > 1e-12:
                            a11 = (m2a[1] - m2b[1]) / det
                            a12 = (m2b[0] - m2a[0]) / det
                            a21 = (m1a[1] - m1b[1]) / det
                            a22 = (m1b[0] - m1a[0]) / det
                            b1 = m2a[0] - m1a[0]
                            b2 = m2a[1] - m1a[1]

                            alpha = a11 * b1 + a12 * b2
                            beta = a21 * b1 + a22 * b2
                            if alpha >= 0.0 and alpha <= 1.0 and beta >= 0.0 and beta <= 1.0:
                                m0 = ti.math.vec3(
                                    m1a[0] + alpha * (m1b[0] - m1a[0]),
                                    m1a[1] + alpha * (m1b[1] - m1a[1]),
                                    (m1a[2] + alpha * (m1b[2] - m1a[2]) + m2a[2] + beta * (m2b[2] - m2a[2])) * 0.5,
                                )
                                if not candCount or m0[0] < self.mc_out[i_b, 0][0]:
                                    self.mc_out[i_b, 0] = m0
                                if not candCount or m0[0] > self.mc_out[i_b, 1][0]:
                                    self.mc_out[i_b, 1] = m0
                                if not candCount or m0[1] < self.mc_out[i_b, 2][1]:
                                    self.mc_out[i_b, 2] = m0
                                if not candCount or m0[1] > self.mc_out[i_b, 3][1]:
                                    self.mc_out[i_b, 3] = m0
                                candCount += 1

            var_rx = ti.math.vec3(0, 0, 0)
            
            if candCount > 0:
                # Polygon intersection was found.
                # TODO(btaba): replace the above routine with the manifold point routine
                # from MJX. Deduplicate the points properly.
                last_pt = ti.math.vec3(self.FLOAT_MAX, self.FLOAT_MAX, self.FLOAT_MAX)

                for k in range(self.MULTI_CONTACT_COUNT):
                    outk = self.mc_out[i_b, k]
                    pt = outk[0] * dir + outk[1] * dir2 + outk[2] * normal
                    
                    # skip contact points that are too close
                    if ti.math.length(pt - last_pt) <= 1e-6:
                        continue
                    
                    self.mc_contact_points[i_b, contact_count] = pt
                    last_pt = pt
                    contact_count += 1      
            # @TODO: check stability of the below code
            '''
            else:            
                # Polygon intersection was not found. Loop through all vertex pairs and
                # calculate an approximate contact point.
                minDist = float(0.0)
                for i in range(v1count):
                    for j in range(v2count):
                        # Find the closest vertex pair. Calculate a contact point var_rx as the
                        # midpoint between the closest vertex pair.
                        m1 = self.mc_v1[i_b, i]
                        m2 = self.mc_v2[i_b, j]
                        dd = (m1[0] - m2[0]) * (m1[0] - m2[0]) + (m1[1] - m2[1]) * (m1[1] - m2[1])
                        
                        if i != 0 and j != 0 or dd < minDist:
                            minDist = dd
                            var_rx = ((m1[0] + m2[0]) * dir + (m1[1] + m2[1]) * dir2 + (m1[2] + m2[2]) * normal) * 0.5

                        # Check for a closer point between a point on v2 and an edge on v1.
                        m1b = self.mc_v1[i_b, (i + 1) % v1count]
                        m2b = self.mc_v2[i_b, (j + 1) % v2count]
                        
                        if v1count > 1:
                            dd = (m1b[0] - m1[0]) * (m1b[0] - m1[0]) + (m1b[1] - m1[1]) * (m1b[1] - m1[1])
                            t = ((m2[1] - m1[1]) * (m1b[0] - m1[0]) - (m2[0] - m1[0]) * (m1b[1] - m1[1])) / dd
                            dx = m2[0] + (m1b[1] - m1[1]) * t
                            dy = m2[1] - (m1b[0] - m1[0]) * t
                            dist = (dx - m2[0]) * (dx - m2[0]) + (dy - m2[1]) * (dy - m2[1])

                            if (
                                (dist < minDist)
                                and ((dx - m1[0]) * (m1b[0] - m1[0]) + (dy - m1[1]) * (m1b[1] - m1[1]) >= 0)
                                and ((dx - m1b[0]) * (m1[0] - m1b[0]) + (dy - m1b[1]) * (m1[1] - m1b[1]) >= 0)
                            ):
                                alpha = ti.math.sqrt(((dx - m1[0]) * (dx - m1[0]) + (dy - m1[1]) * (dy - m1[1])) / dd)
                                minDist = dist
                                w = ((1.0 - alpha) * m1 + alpha * m1b + m2) * 0.5
                                var_rx = w[0] * dir + w[1] * dir2 + w[2] * normal
                                
                        # check for a closer point between a point on v1 and an edge on v2
                        if v2count > 1:
                            dd = (m2b[0] - m2[0]) * (m2b[0] - m2[0]) + (m2b[1] - m2[1]) * (m2b[1] - m2[1])
                            t = ((m1[1] - m2[1]) * (m2b[0] - m2[0]) - (m1[0] - m2[0]) * (m2b[1] - m2[1])) / dd
                            dx = m1[0] + (m2b[1] - m2[1]) * t
                            dy = m1[1] - (m2b[0] - m2[0]) * t
                            dist = (dx - m1[0]) * (dx - m1[0]) + (dy - m1[1]) * (dy - m1[1])

                            if (
                                dist < minDist
                                and (dx - m2[0]) * (m2b[0] - m2[0]) + (dy - m2[1]) * (m2b[1] - m2[1]) >= 0
                                and (dx - m2b[0]) * (m2[0] - m2b[0]) + (dy - m2b[1]) * (m2[1] - m2b[1]) >= 0
                            ):
                                alpha = ti.math.sqrt(((dx - m2[0]) * (dx - m2[0]) + (dy - m2[1]) * (dy - m2[1])) / dd)
                                minDist = dist
                                w = (m1 + (1.0 - alpha) * m2 + alpha * m2b) * 0.5
                                var_rx = w[0] * dir + w[1] * dir2 + w[2] * normal

                for k in range(self.MULTI_CONTACT_COUNT):
                    self.mc_contact_points[i_b, k] = var_rx
                
                contact_count = 1
            '''
            
        self.mc_contact_count[i_b] = contact_count
    
    @ti.func
    def all_same(self, v0, v1):
        dx = abs(v0[0] - v1[0])
        dy = abs(v0[1] - v1[1])
        dz = abs(v0[2] - v1[2])
        
        return (
            (dx <= 1.0e-9 or dx <= max(abs(v0[0]), abs(v1[0])) * 1.0e-9)
            and (dy <= 1.0e-9 or dy <= max(abs(v0[1]), abs(v1[1])) * 1.0e-9)
            and (dz <= 1.0e-9 or dz <= max(abs(v0[2]), abs(v1[2])) * 1.0e-9)
        )
    
    @ti.func
    def any_different(self, v0, v1):
        dx = abs(v0[0] - v1[0])
        dy = abs(v0[1] - v1[1])
        dz = abs(v0[2] - v1[2])

        return (
            (dx > 1.0e-9 and dx > max(abs(v0[0]), abs(v1[0])) * 1.0e-9)
            or (dy > 1.0e-9 and dy > max(abs(v0[1]), abs(v1[1])) * 1.0e-9)
            or (dz > 1.0e-9 and dz > max(abs(v0[2]), abs(v1[2])) * 1.0e-9)
        )
        
    
    @ti.func
    def compute_support(self, direction, i_ga, i_gb, i_b):
        dist1, s1 = self.gjk_support_geom(direction, i_ga, i_b)
        dist2, s2 = self.gjk_support_geom(-direction, i_gb, i_b)
        
        support_pt = s1 - s2
        return dist1 + dist2, support_pt
    
    @ti.func
    def gjk_support_geom(self, direction, i_g, i_b):
        support_pt = self.support_driver(direction, i_g, i_b)
        dist = ti.math.dot(support_pt, direction)
        return dist, support_pt

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

        vid = (d_box[0] > 0) * 4 + (d_box[1] > 0) * 2 + (d_box[2] > 0) * 1
        v_ = ti.Vector(
            [
                ti.math.sign(d_box[0]) * self._solver.geoms_info[i_g].data[0] * 0.5,
                ti.math.sign(d_box[1]) * self._solver.geoms_info[i_g].data[1] * 0.5,
                ti.math.sign(d_box[2]) * self._solver.geoms_info[i_g].data[2] * 0.5,
            ],
            dt=gs.ti_float,
        )
        vid += self._solver.geoms_info[i_g].vert_start
        v = gu.ti_transform_by_trans_quat(v_, g_state.pos, g_state.quat)
        return v, vid

    @ti.func
    def support_driver(self, direction, i_g, i_b):
        v = ti.Vector.zero(gs.ti_float, 3)
        geom_type = self._solver.geoms_info[i_g].type
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
        else:
            v, _ = self.support_field._func_support_world(direction, i_g, i_b)
        return v