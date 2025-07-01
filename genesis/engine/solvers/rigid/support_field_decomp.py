from typing import TYPE_CHECKING
from math import pi

import numpy as np
import taichi as ti

import genesis as gs
import genesis.utils.geom as gu

if TYPE_CHECKING:
    from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver


@ti.data_oriented
class SupportField:
    def __init__(self, rigid_solver: "RigidSolver") -> None:
        self.solver = rigid_solver
        self.support_res = 180
        if self.solver._enable_collision:
            self._compute_support()

    def _get_direction_grid(self):
        theta = np.arange(self.support_res) / self.support_res * 2 * pi - pi
        phi = np.arange(self.support_res) / self.support_res * pi

        spherical_coords = np.zeros([self.support_res, self.support_res, 2])
        spherical_coords[:, :, 0] = theta[:, None]
        spherical_coords[:, :, 1] = phi[None]

        x = np.sin(spherical_coords[:, :, 1]) * np.cos(spherical_coords[:, :, 0])
        y = np.sin(spherical_coords[:, :, 1]) * np.sin(spherical_coords[:, :, 0])
        z = np.cos(spherical_coords[:, :, 1])
        v = np.stack((x, y, z), axis=-1)
        return v

    def _compute_support(self) -> None:
        v = self._get_direction_grid()
        v1 = v.reshape([-1, 3])
        support_v = []
        support_vid = []
        support_cell_start = []
        start = 0
        if self.solver.n_geoms > 0:
            init_pos = self.solver.verts_info.init_pos.to_numpy()
            for i_g in range(self.solver.n_geoms):
                vert_start = self.solver.geoms_info.vert_start[i_g]
                vert_end = self.solver.geoms_info.vert_end[i_g]
                this_pos = init_pos[vert_start:vert_end]

                num_v = v1.shape[0]
                window_size = int(5e8 // this_pos.shape[0])
                max_indices = np.empty(num_v, dtype=np.intp)

                for i in range(0, num_v, window_size):
                    end = min(i + window_size, num_v)
                    dot_chunk = v1[i:end] @ this_pos.T
                    max_indices[i:end] = np.argmax(dot_chunk, axis=1)

                support = this_pos[max_indices]

                support_cell_start.append(start)
                support_v.append(support)
                support_vid.append(max_indices)
                start += support.shape[0]

            support_v = np.concatenate(support_v)
            support_vid = np.concatenate(support_vid, dtype=gs.np_int)
            support_cell_start = np.array(support_cell_start, dtype=gs.np_int)
        else:
            support_v = np.zeros([1, 3], dtype=gs.np_float)
            support_vid = np.zeros([1], dtype=gs.np_int)
            support_cell_start = np.zeros([1], dtype=gs.np_int)

        self.n_support_cells = start
        self.support_cell_start = ti.field(dtype=gs.ti_int, shape=self.solver.n_geoms_)
        self.support_v = ti.Vector.field(3, dtype=gs.ti_float, shape=max(1, self.n_support_cells))
        self.support_vid = ti.field(dtype=gs.ti_int, shape=max(1, self.n_support_cells))
        self._kernel_init_support(support_cell_start, support_v, support_vid)

    @ti.kernel
    def _kernel_init_support(
        self,
        support_cell_start: ti.types.ndarray(),
        support_v: ti.types.ndarray(),
        support_vid: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self.solver._para_level < gs.PARA_LEVEL.PARTIAL)
        for i in range(self.solver.n_geoms):
            self.support_cell_start[i] = support_cell_start[i]

        for i in range(self.n_support_cells):
            self.support_vid[i] = support_vid[i]
            for j in ti.static(range(3)):
                self.support_v[i][j] = support_v[i, j]

    @ti.func
    def _func_support_world(self, d, i_g, i_b):
        """
        support position for a world direction
        """

        g_state = self.solver.geoms_state[i_g, i_b]
        d_mesh = gu.ti_transform_by_quat(d, gu.ti_inv_quat(g_state.quat))
        v, vid = self._func_support_mesh(d_mesh, i_g)
        v_ = gu.ti_transform_by_trans_quat(v, g_state.pos, g_state.quat)
        return v_, vid

    @ti.func
    def _func_support_mesh(self, d_mesh, i_g):
        """
        support point at mesh frame coordinate.
        """
        theta = ti.atan2(d_mesh[1], d_mesh[0])  # [-pi, pi]
        phi = ti.acos(d_mesh[2])  # [0, pi]

        support_res = gs.ti_int(self.support_res)
        dot_max = gs.ti_float(-1e20)
        v = ti.Vector([0.0, 0.0, 0.0], dt=gs.ti_float)
        vid = 0

        ii = (theta + pi) / pi / 2 * support_res
        jj = phi / pi * support_res

        for i4 in range(4):
            i, j = gs.ti_int(0), gs.ti_int(0)
            if i4 % 2:
                i = gs.ti_int(ti.math.ceil(ii) % support_res)
            else:
                i = gs.ti_int(ti.math.floor(ii) % support_res)

            if i4 // 2 > 0:
                j = gs.ti_int(ti.math.clamp(ti.math.ceil(jj), 0, support_res - 1))
                if j == support_res - 1:
                    j = support_res - 2
            else:
                j = gs.ti_int(ti.math.clamp(ti.math.floor(jj), 0, support_res - 1))
                if j == 0:
                    j = 1

            support_idx = gs.ti_int(self.support_cell_start[i_g] + i * support_res + j)
            _vid = self.support_vid[support_idx]
            pos = self.support_v[support_idx]
            dot = pos.dot(d_mesh)

            if dot > dot_max:
                v = pos
                dot_max = dot
                vid = _vid

        return v, vid

    @ti.func
    def _func_support_sphere(self, d, i_g, i_b, shrink):
        sphere_center = self.solver.geoms_state[i_g, i_b].pos
        sphere_radius = self.solver.geoms_info[i_g].data[0]

        # Shrink the sphere to a point
        res = sphere_center
        if not shrink:
            res += d * sphere_radius
        return res

    @ti.func
    def _func_support_ellipsoid(self, d, i_g, i_b):
        g_state = self.solver.geoms_state[i_g, i_b]
        ellipsoid_center = g_state.pos
        ellipsoid_scaled_axis = ti.Vector(
            [
                self.solver.geoms_info[i_g].data[0] ** 2,
                self.solver.geoms_info[i_g].data[1] ** 2,
                self.solver.geoms_info[i_g].data[2] ** 2,
            ],
            dt=gs.ti_float,
        )
        ellipsoid_scaled_axis = gu.ti_transform_by_quat(ellipsoid_scaled_axis, g_state.quat)
        dist = ellipsoid_scaled_axis / ti.sqrt(d.dot(1.0 / ellipsoid_scaled_axis))
        return ellipsoid_center + d * dist

    @ti.func
    def _func_support_capsule(self, d, i_g, i_b, shrink):
        res = gs.ti_vec3(0, 0, 0)
        g_state = self.solver.geoms_state[i_g, i_b]
        capsule_center = g_state.pos
        capsule_radius = self.solver.geoms_info[i_g].data[0]
        capsule_halflength = 0.5 * self.solver.geoms_info[i_g].data[1]

        if shrink:
            local_dir = gu.ti_transform_by_quat(d, gu.ti_inv_quat(g_state.quat))
            res[2] = capsule_halflength if local_dir[2] >= 0.0 else -capsule_halflength
            res = gu.ti_transform_by_trans_quat(res, capsule_center, g_state.quat)
        else:
            capsule_axis = gu.ti_transform_by_quat(ti.Vector([0.0, 0.0, 1.0], dt=gs.ti_float), g_state.quat)
            capsule_endpoint_side = -1.0 if d.dot(capsule_axis) < 0.0 else 1.0
            capsule_endpoint = capsule_center + capsule_halflength * capsule_endpoint_side * capsule_axis
            res = capsule_endpoint + d * capsule_radius
        return res

    @ti.func
    def _func_support_prism(self, d, i_g, i_b):
        istart = 3
        if d[2] < 0:
            istart = 0

        ibest = istart
        best = self.solver.collider.prism[istart, i_b].dot(d)
        for i in range(istart + 1, istart + 3):
            dot = self.solver.collider.prism[i, i_b].dot(d)
            if dot > best:
                ibest = i
                best = dot

        return self.solver.collider.prism[ibest, i_b], ibest

    @ti.func
    def _func_support_box(self, d, i_g, i_b):
        g_state = self.solver.geoms_state[i_g, i_b]
        d_box = gu.ti_inv_transform_by_quat(d, g_state.quat)

        v_ = ti.Vector(
            [
                (-1.0 if d_box[0] < 0.0 else 1.0) * self.solver.geoms_info[i_g].data[0] * 0.5,
                (-1.0 if d_box[1] < 0.0 else 1.0) * self.solver.geoms_info[i_g].data[1] * 0.5,
                (-1.0 if d_box[2] < 0.0 else 1.0) * self.solver.geoms_info[i_g].data[2] * 0.5,
            ],
            dt=gs.ti_float,
        )
        vid = (v_[0] > 0.0) * 1 + (v_[1] > 0.0) * 2 + (v_[2] > 0.0) * 4
        vid += self.solver.geoms_info[i_g].vert_start
        v = gu.ti_transform_by_trans_quat(v_, g_state.pos, g_state.quat)
        return v, vid
