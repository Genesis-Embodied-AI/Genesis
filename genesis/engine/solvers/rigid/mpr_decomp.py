import numpy as np
import taichi as ti
import torch

import genesis as gs
import genesis.utils.geom as gu

from .support_field_decomp import SupportField


@ti.data_oriented
class MPR:
    def __init__(self, rigid_solver):
        self._solver = rigid_solver
        self._max_contact_pairs = rigid_solver._max_collision_pairs
        self._B = rigid_solver._B
        self._para_level = rigid_solver._para_level

        if gs.ti_float == ti.f32:
            self.CCD_EPS = 1e-6
        else:
            self.CCD_EPS = 1e-10

        self.support_field = SupportField(rigid_solver)
        self.init_support()

    def init_support(self):

        struct_support = ti.types.struct(
            v1=gs.ti_vec3,
            v2=gs.ti_vec3,
            v=gs.ti_vec3,
        )
        self.simplex_support = struct_support.field(
            shape=self._solver._batch_shape((self._solver.n_geoms_, self._solver.n_geoms_, 4)),
            layout=ti.Layout.SOA,
        )
        self.simplex_size = ti.field(
            gs.ti_int, shape=self._solver._batch_shape((self._solver.n_geoms_, self._solver.n_geoms_))
        )

    def reset(self):
        pass

    @ti.kernel
    def clear(self):
        self.simplex_size.fill(0)

    @ti.func
    def func_point_in_geom_aabb(self, point, i_g, i_b):
        return (point < self._solver.geoms_state[i_g, i_b].aabb_max).all() and (
            point > self._solver.geoms_state[i_g, i_b].aabb_min
        ).all()

    @ti.func
    def func_is_geom_aabbs_overlap(self, i_ga, i_gb, i_b):
        return not (
            (self._solver.geoms_state[i_ga, i_b].aabb_max <= self._solver.geoms_state[i_gb, i_b].aabb_min).any()
            or (self._solver.geoms_state[i_ga, i_b].aabb_min >= self._solver.geoms_state[i_gb, i_b].aabb_max).any()
        )

    @ti.func
    def func_find_intersect_midpoint(self, i_ga, i_gb):
        # return the center of the intersecting AABB of AABBs of two geoms
        intersect_lower = ti.max(self._solver.geoms_state[i_ga].aabb_min, self._solver.geoms_state[i_gb].aabb_min)
        intersect_upper = ti.min(self._solver.geoms_state[i_ga].aabb_max, self._solver.geoms_state[i_gb].aabb_max)
        return 0.5 * (intersect_lower + intersect_upper)

    @ti.func
    def mpr_swap(self, i, j, i_ga, i_gb, i_b):
        self.simplex_support[i_ga, i_gb, i, i_b].v1, self.simplex_support[i_ga, i_gb, j, i_b].v1 = (
            self.simplex_support[i_ga, i_gb, j, i_b].v1,
            self.simplex_support[i_ga, i_gb, i, i_b].v1,
        )
        self.simplex_support[i_ga, i_gb, i, i_b].v2, self.simplex_support[i_ga, i_gb, j, i_b].v2 = (
            self.simplex_support[i_ga, i_gb, j, i_b].v2,
            self.simplex_support[i_ga, i_gb, i, i_b].v2,
        )
        self.simplex_support[i_ga, i_gb, i, i_b].v, self.simplex_support[i_ga, i_gb, j, i_b].v = (
            self.simplex_support[i_ga, i_gb, j, i_b].v,
            self.simplex_support[i_ga, i_gb, i, i_b].v,
        )

    @ti.func
    def mpr_point_segment_dist2(self, P, A, B):
        AB = B - A
        AP = P - A
        AB_AB = AB.dot(AB)
        AP_AB = AP.dot(AB)
        t = AP_AB / AB_AB
        if t < 0.0:
            t = gs.ti_float(0.0)
        elif t > 1.0:
            t = gs.ti_float(1.0)
        Q = A + AB * t
        pdir = Q

        return (P - Q).norm() ** 2, pdir

    @ti.func
    def mpr_point_tri_depth(self, P, x0, B, C):
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
        if ti.abs(d) < gs.EPS:
            s = t = -1.0
        else:
            s = (q * r - w * p) / d
            t = (-s * r - q) / w

        if (
            (ti.abs(s) < gs.EPS or s > 0.0)
            and (ti.abs(s - 1.0) < gs.EPS or s < 1.0)
            and (ti.abs(t) < gs.EPS or t > 0.0)
            and (ti.abs(t - 1.0) < gs.EPS or t < 1.0)
            and (ti.abs(t + s - 1.0) < gs.EPS or t + s < 1.0)
        ):
            pdir = x0 + d1 * s + d2 * t
            dist = (P - pdir).norm() ** 2
        else:

            dist, pdir = self.mpr_point_segment_dist2(P, x0, B)
            dist2, pdir2 = self.mpr_point_segment_dist2(P, x0, C)
            if dist2 < dist:
                dist = dist2
                pdir = pdir2

            dist2, pdir2 = self.mpr_point_segment_dist2(P, B, C)
            if dist2 < dist:
                dist = dist2
                pdir = pdir2

        return ti.sqrt(dist), pdir

    @ti.func
    def mpr_portal_dir(self, i_ga, i_gb, i_b):
        ans = gs.ti_vec3([0.0, 0.0, 0.0])
        v2v1 = self.simplex_support[i_ga, i_gb, 2, i_b].v - self.simplex_support[i_ga, i_gb, 1, i_b].v
        v3v1 = self.simplex_support[i_ga, i_gb, 3, i_b].v - self.simplex_support[i_ga, i_gb, 1, i_b].v
        direction = v2v1.cross(v3v1)
        ans = direction.normalized()
        return ans

    @ti.func
    def mpr_portal_encapsules_origin(self, direction, i_ga, i_gb, i_b):
        dot = direction.dot(self.simplex_support[i_ga, i_gb, 1, i_b].v)

        return dot > 0.0
        # return dot > 0 or ti.abs(dot) < CCD_EPS

    @ti.func
    def mpr_portal_can_encapsule_origin(self, v, direction):
        dot = v.dot(direction)
        return dot >= 0

    @ti.func
    def mpr_portal_reach_tolerance(self, v, direction, i_ga, i_gb, i_b):
        dv1 = self.simplex_support[i_ga, i_gb, 1, i_b].v.dot(direction)
        dv2 = self.simplex_support[i_ga, i_gb, 2, i_b].v.dot(direction)
        dv3 = self.simplex_support[i_ga, i_gb, 3, i_b].v.dot(direction)
        dv4 = v.dot(direction)
        dot1 = ti.min(dv4 - dv1, dv4 - dv2, dv4 - dv3)

        mpr_tolerance = 1e-5
        return dot1 <= mpr_tolerance

    @ti.func
    def support_sphere(self, direction, i_g, i_b):
        sphere_center = self._solver.geoms_state[i_g, i_b].pos
        sphere_radius = self._solver.geoms_info[i_g].data[0]
        return sphere_center + direction * sphere_radius

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
        vid = 0
        geom_type = self._solver.geoms_info[i_g].type
        if geom_type == gs.GEOM_TYPE.SPHERE:
            v = self.support_sphere(direction, i_g, i_b)
        elif geom_type == gs.GEOM_TYPE.BOX:
            v, vid = self.support_box(direction, i_g, i_b)
        elif geom_type == gs.GEOM_TYPE.TERRAIN:
            if ti.static(self._solver.collider._has_terrain):
                v, vid = self.support_prism(direction, i_g, i_b)
        else:
            v, vid = self.support_field._func_support_world(direction, i_g, i_b)

        return v, vid

    @ti.func
    def compute_support(self, direction, i_ga, i_gb, i_b):
        v1, vid = self.support_driver(direction, i_ga, i_b)
        v2, vid = self.support_driver(-direction, i_gb, i_b)

        v = v1 - v2
        return v, v1, v2

    @ti.func
    def func_geom_support(self, direction, i_g, i_b):
        g_state = self._solver.geoms_state[i_g, i_b]
        direction_in_init_frame = gu.ti_transform_by_quat(direction, gu.ti_inv_quat(g_state.quat))

        dot_max = gs.ti_float(-1e20)
        v = ti.Vector.zero(gs.ti_float, 3)
        vid = 0

        g_info = self._solver.geoms_info[i_g]

        for i_v in range(g_info.vert_start, g_info.vert_end):
            pos = self._solver.verts_info[i_v].init_pos
            dot = pos.dot(direction_in_init_frame)
            if dot > dot_max:
                v = pos
                dot_max = dot
                vid = i_v
        v_ = gu.ti_transform_by_trans_quat(v, g_state.pos, g_state.quat)

        return v_, vid

    @ti.func
    def mpr_refine_portal(self, i_ga, i_gb, i_b):
        ret = 1
        while True:
            direction = self.mpr_portal_dir(i_ga, i_gb, i_b)

            if self.mpr_portal_encapsules_origin(direction, i_ga, i_gb, i_b):
                ret = 0
                break

            v, v1, v2 = self.compute_support(direction, i_ga, i_gb, i_b)

            if not self.mpr_portal_can_encapsule_origin(v, direction) or self.mpr_portal_reach_tolerance(
                v, direction, i_ga, i_gb, i_b
            ):
                ret = -1
                break

            self.mpr_expand_portal(v, v1, v2, i_ga, i_gb, i_b)
        return ret

    @ti.func
    def mpr_find_pos(self, i_ga, i_gb, i_b):
        b = ti.Vector([0.0, 0.0, 0.0, 0.0], dt=gs.ti_float)

        sum_ = gs.ti_float(0.0)
        inv_ = gs.ti_float(0.0)
        direction = self.mpr_portal_dir(i_ga, i_gb, i_b)

        for i in range(4):
            i1, i2, i3 = (i + 1) % 4, (i + 2) % 4, (i + 3) % 4
            vec = self.simplex_support[i_ga, i_gb, i1, i_b].v.cross(self.simplex_support[i_ga, i_gb, i2, i_b].v)
            b[i] = vec.dot(self.simplex_support[i_ga, i_gb, i3, i_b].v) * (1 - i % 2 * 2)

            sum_ = sum_ + b[i]

        if sum_ == 0 or sum_ < 0:
            sum_ = 0
            for i in range(4):
                if i == 0:
                    b[i] = 0
                else:
                    i1, i2 = i % 3 + 1, (i + 1) % 3 + 1
                    vec = self.simplex_support[i_ga, i_gb, i1, i_b].v.cross(self.simplex_support[i_ga, i_gb, i2, i_b].v)
                    b[i] = vec.dot(direction)
                    sum_ = sum_ + b[i]

        inv_ = 1 / sum_

        p1 = gs.ti_vec3([0.0, 0.0, 0.0])
        p2 = gs.ti_vec3([0.0, 0.0, 0.0])
        for i in range(4):
            vec = self.simplex_support[i_ga, i_gb, i, i_b].v1 * b[i]
            p1 += vec
            vec = self.simplex_support[i_ga, i_gb, i, i_b].v2 * b[i]
            p2 += vec

        p1 = p1 * inv_
        p2 = p2 * inv_
        pos = (p1 + p2) * 0.5

        return pos

    @ti.func
    def mpr_find_penetr_touch(self, i_ga, i_gb, i_b):
        is_col = False
        penetration = gs.ti_float(0.0)
        normal = gs.ti_vec3([1.0, 0.0, 0.0])
        pos = (self.simplex_support[i_ga, i_gb, 1, i_b].v1 + self.simplex_support[i_ga, i_gb, 1, i_b].v2) * 0.5
        return is_col, normal, penetration, pos

    @ti.func
    def mpr_find_penetr_segment(self, i_ga, i_gb, i_b):
        is_col = True
        penetration = self.simplex_support[i_ga, i_gb, 1, i_b].v.norm()
        normal = self.simplex_support[i_ga, i_gb, 1, i_b].v * -1.0
        pos = (self.simplex_support[i_ga, i_gb, 1, i_b].v1 + self.simplex_support[i_ga, i_gb, 1, i_b].v2) * 0.5

        return is_col, normal, penetration, pos

    @ti.func
    def mpr_find_penetration(self, i_ga, i_gb, i_b):
        iterations = 0
        max_iterations = 100

        is_col = False
        pos = gs.ti_vec3([1.0, 0.0, 0.0])
        normal = gs.ti_vec3([1.0, 0.0, 0.0])
        penetration = gs.ti_float(0.0)

        while True:
            direction = self.mpr_portal_dir(i_ga, i_gb, i_b)
            v, v1, v2 = self.compute_support(direction, i_ga, i_gb, i_b)
            if self.mpr_portal_reach_tolerance(v, direction, i_ga, i_gb, i_b) or iterations > max_iterations:
                depth, pdir = self.mpr_point_tri_depth(
                    gs.ti_vec3([0.0, 0.0, 0.0]),
                    self.simplex_support[i_ga, i_gb, 1, i_b].v,
                    self.simplex_support[i_ga, i_gb, 2, i_b].v,
                    self.simplex_support[i_ga, i_gb, 3, i_b].v,
                )
                pdir = pdir.normalized()
                # if ti.abs(depth) < 1e-10:
                #     pdir = gs.ti_vec3([1., 0., 0.])
                # else:
                #     pdir = pdir.normalized()

                is_col = True
                pos = self.mpr_find_pos(i_ga, i_gb, i_b)
                normal = pdir * -1.0
                penetration = depth
                break

            self.mpr_expand_portal(v, v1, v2, i_ga, i_gb, i_b)
            iterations += 1

        return is_col, normal, penetration, pos

    @ti.func
    def mpr_expand_portal(self, v, v1, v2, i_ga, i_gb, i_b):
        v4v0 = v.cross(self.simplex_support[i_ga, i_gb, 0, i_b].v)
        dot = self.simplex_support[i_ga, i_gb, 1, i_b].v.dot(v4v0)

        if dot > 0:
            dot = self.simplex_support[i_ga, i_gb, 2, i_b].v.dot(v4v0)
            if dot > 0:

                self.simplex_support[i_ga, i_gb, 1, i_b].v1 = v1
                self.simplex_support[i_ga, i_gb, 1, i_b].v2 = v2
                self.simplex_support[i_ga, i_gb, 1, i_b].v = v

            else:
                self.simplex_support[i_ga, i_gb, 3, i_b].v1 = v1
                self.simplex_support[i_ga, i_gb, 3, i_b].v2 = v2
                self.simplex_support[i_ga, i_gb, 3, i_b].v = v

        else:
            dot = self.simplex_support[i_ga, i_gb, 3, i_b].v.dot(v4v0)
            if dot > 0:
                self.simplex_support[i_ga, i_gb, 2, i_b].v1 = v1
                self.simplex_support[i_ga, i_gb, 2, i_b].v2 = v2
                self.simplex_support[i_ga, i_gb, 2, i_b].v = v

            else:
                self.simplex_support[i_ga, i_gb, 1, i_b].v1 = v1
                self.simplex_support[i_ga, i_gb, 1, i_b].v2 = v2
                self.simplex_support[i_ga, i_gb, 1, i_b].v = v

    @ti.func
    def mpr_discover_portal(self, i_ga, i_gb, i_b):
        ret = 0
        self.simplex_size[i_ga, i_gb, i_b] = 0

        g_info = self._solver.geoms_info[i_ga]
        g_state = self._solver.geoms_state[i_ga, i_b]

        center_a = gu.ti_transform_by_trans_quat(g_info.center, g_state.pos, g_state.quat)

        g_info = self._solver.geoms_info[i_gb]
        g_state = self._solver.geoms_state[i_gb, i_b]
        center_b = gu.ti_transform_by_trans_quat(g_info.center, g_state.pos, g_state.quat)

        self.simplex_support[i_ga, i_gb, 0, i_b].v1 = center_a
        self.simplex_support[i_ga, i_gb, 0, i_b].v2 = center_b
        self.simplex_support[i_ga, i_gb, 0, i_b].v = center_a - center_b
        self.simplex_size[i_ga, i_gb, i_b] = 1

        if self.simplex_support[i_ga, i_gb, 0, i_b].v.norm() < self.CCD_EPS:
            self.simplex_support[i_ga, i_gb, 0, i_b].v += gs.ti_vec3([10.0 * self.CCD_EPS, 0.0, 0.0])

        direction = (self.simplex_support[i_ga, i_gb, 0, i_b].v * -1).normalized()

        v, v1, v2 = self.compute_support(direction, i_ga, i_gb, i_b)

        self.simplex_support[i_ga, i_gb, 1, i_b].v1 = v1
        self.simplex_support[i_ga, i_gb, 1, i_b].v2 = v2
        self.simplex_support[i_ga, i_gb, 1, i_b].v = v
        self.simplex_size[i_ga, i_gb, i_b] = 2

        dot = v.dot(direction)

        # if dot < 0 or ti.abs(dot) < CCD_EPS:
        if dot < self.CCD_EPS:
            ret = -1
        else:
            direction = self.simplex_support[i_ga, i_gb, 0, i_b].v.cross(self.simplex_support[i_ga, i_gb, 1, i_b].v)
            if direction.norm() < self.CCD_EPS:
                # if portal.points[1].v == ccd_vec3_origin:
                if self.simplex_support[i_ga, i_gb, 1, i_b].v.norm() < self.CCD_EPS:
                    ret = 1
                else:
                    ret = 2
            else:
                direction = direction.normalized()
                v, v1, v2 = self.compute_support(direction, i_ga, i_gb, i_b)
                dot = v.dot(direction)
                # if dot < 0 or ti.abs(dot) < CCD_EPS:
                if dot < self.CCD_EPS:
                    ret = -1
                else:
                    self.simplex_support[i_ga, i_gb, 2, i_b].v1 = v1
                    self.simplex_support[i_ga, i_gb, 2, i_b].v2 = v2
                    self.simplex_support[i_ga, i_gb, 2, i_b].v = v
                    self.simplex_size[i_ga, i_gb, i_b] = 3

                    va = self.simplex_support[i_ga, i_gb, 1, i_b].v - self.simplex_support[i_ga, i_gb, 0, i_b].v
                    vb = self.simplex_support[i_ga, i_gb, 2, i_b].v - self.simplex_support[i_ga, i_gb, 0, i_b].v
                    direction = va.cross(vb)
                    direction = direction.normalized()

                    dot = direction.dot(self.simplex_support[i_ga, i_gb, 0, i_b].v)
                    if dot > 0:
                        self.mpr_swap(1, 2, i_ga, i_gb, i_b)
                        direction = direction * -1.0

                    while self.simplex_size[i_ga, i_gb, i_b] < 4:
                        v, v1, v2 = self.compute_support(direction, i_ga, i_gb, i_b)
                        dot = v.dot(direction)
                        # if dot < 0 or ti.abs(dot) < CCD_EPS:
                        if dot < self.CCD_EPS:
                            ret = -1
                            break

                        cont = False

                        va = self.simplex_support[i_ga, i_gb, 1, i_b].v.cross(v)
                        dot = va.dot(self.simplex_support[i_ga, i_gb, 0, i_b].v)
                        # if dot < 0 or ti.abs(dot) < CCD_EPS:
                        if dot < -self.CCD_EPS:
                            self.simplex_support[i_ga, i_gb, 2, i_b].v1 = v1
                            self.simplex_support[i_ga, i_gb, 2, i_b].v2 = v2
                            self.simplex_support[i_ga, i_gb, 2, i_b].v = v
                            cont = True

                        if not cont:
                            va = v.cross(self.simplex_support[i_ga, i_gb, 2, i_b].v)
                            dot = va.dot(self.simplex_support[i_ga, i_gb, 0, i_b].v)
                            # if dot < 0 or ti.abs(dot) < CCD_EPS:
                            if dot < -self.CCD_EPS:
                                self.simplex_support[i_ga, i_gb, 1, i_b].v1 = v1
                                self.simplex_support[i_ga, i_gb, 1, i_b].v2 = v2
                                self.simplex_support[i_ga, i_gb, 1, i_b].v = v
                                cont = True

                        if cont:
                            va = self.simplex_support[i_ga, i_gb, 1, i_b].v - self.simplex_support[i_ga, i_gb, 0, i_b].v
                            vb = self.simplex_support[i_ga, i_gb, 2, i_b].v - self.simplex_support[i_ga, i_gb, 0, i_b].v
                            direction = va.cross(vb)
                            direction = direction.normalized()
                        else:

                            self.simplex_support[i_ga, i_gb, 3, i_b].v1 = v1
                            self.simplex_support[i_ga, i_gb, 3, i_b].v2 = v2
                            self.simplex_support[i_ga, i_gb, 3, i_b].v = v
                            self.simplex_size[i_ga, i_gb, i_b] = 4

        return ret

    @ti.func
    def func_mpr_contact(self, i_ga, i_gb, i_b):
        res = self.mpr_discover_portal(i_ga, i_gb, i_b)

        is_col = False
        pos = gs.ti_vec3([1.0, 0.0, 0.0])
        normal = gs.ti_vec3([1.0, 0.0, 0.0])
        penetration = gs.ti_float(0.0)

        if res == 1:
            is_col, normal, penetration, pos = self.mpr_find_penetr_touch(i_ga, i_gb, i_b)
        elif res == 2:
            is_col, normal, penetration, pos = self.mpr_find_penetr_segment(i_ga, i_gb, i_b)
        elif res == 0:
            res = self.mpr_refine_portal(i_ga, i_gb, i_b)
            if res >= 0:
                is_col, normal, penetration, pos = self.mpr_find_penetration(i_ga, i_gb, i_b)
        else:
            pass

        return is_col, normal, penetration, pos
