from typing import TYPE_CHECKING
from math import pi

import numpy as np
import taichi as ti

import genesis as gs
import genesis.utils.geom as gu

if TYPE_CHECKING:
    from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
from .support_field_decomp import SupportField


@ti.data_oriented
class Support:
    def __init__(self, rigid_solver: "RigidSolver") -> None:
        self._solver = rigid_solver
        self.support_field = SupportField(rigid_solver)

    @ti.func
    def support_sphere(self, direction, i_g, i_b, shrink):
        sphere_center = self._solver.geoms_state[i_g, i_b].pos
        sphere_radius = self._solver.geoms_info[i_g].data[0]

        # Shrink the sphere to a point
        res = sphere_center
        if not shrink:
            res += direction * sphere_radius
        return res

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
    def support_capsule(self, direction, i_g, i_b, shrink):
        res = gs.ti_vec3(0, 0, 0)
        g_state = self._solver.geoms_state[i_g, i_b]
        capsule_center = g_state.pos
        capsule_radius = self._solver.geoms_info[i_g].data[0]
        capsule_halflength = 0.5 * self._solver.geoms_info[i_g].data[1]

        if shrink:
            local_dir = gu.ti_transform_by_quat(direction, gu.ti_inv_quat(g_state.quat))
            res[2] = capsule_halflength if local_dir[2] >= 0.0 else -capsule_halflength
            res = gu.ti_transform_by_trans_quat(res, capsule_center, g_state.quat)
        else:
            capsule_axis = gu.ti_transform_by_quat(ti.Vector([0.0, 0.0, 1.0], dt=gs.ti_float), g_state.quat)
            capsule_endpoint_side = ti.math.sign(direction.dot(capsule_axis))
            if capsule_endpoint_side == 0.0:
                capsule_endpoint_side = 1.0
            capsule_endpoint = capsule_center + capsule_halflength * capsule_endpoint_side * capsule_axis
            res = capsule_endpoint + direction * capsule_radius
        return res

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

        v_ = ti.Vector(
            [
                d_box_sign[0] * self._solver.geoms_info[i_g].data[0] * 0.5,
                d_box_sign[1] * self._solver.geoms_info[i_g].data[1] * 0.5,
                d_box_sign[2] * self._solver.geoms_info[i_g].data[2] * 0.5,
            ],
            dt=gs.ti_float,
        )
        vid = (v_[0] > 0) * 1 + (v_[1] > 0) * 2 + (v_[2] > 0) * 4
        vid += self._solver.geoms_info[i_g].vert_start
        v = gu.ti_transform_by_trans_quat(v_, g_state.pos, g_state.quat)
        return v, vid
