import numpy as np
import taichi as ti

import genesis as gs
import genesis.utils.geom as gu


@ti.data_oriented
class SDF:
    def __init__(self, rigid_solver):
        self.solver = rigid_solver

        struct_geom_info = ti.types.struct(
            T_mesh_to_sdf=gs.ti_mat4,
            sdf_res=gs.ti_ivec3,
            sdf_max=gs.ti_float,
            sdf_cell_size=gs.ti_float,
            sdf_cell_start=gs.ti_int,
        )

        self.geoms_info = struct_geom_info.field(shape=self.solver.n_geoms_, layout=ti.Layout.SOA)
        self.geoms_sdf_start = ti.field(dtype=gs.ti_int, shape=self.solver.n_geoms_)
        self.geoms_sdf_val = ti.field(dtype=gs.ti_float, shape=self.solver.n_cells_)
        self.geoms_sdf_grad = ti.Vector.field(3, dtype=gs.ti_float, shape=self.solver.n_cells_)
        self.geoms_sdf_closest_vert = ti.field(dtype=gs.ti_int, shape=self.solver.n_cells_)

        if self.solver.n_geoms > 0:
            geoms = self.solver.geoms
            self._kernel_init_geom_fields(
                geoms_T_mesh_to_sdf=np.array([geom.T_mesh_to_sdf for geom in geoms], dtype=gs.np_float),
                geoms_sdf_res=np.array([geom.sdf_res for geom in geoms], dtype=gs.np_int),
                geoms_sdf_cell_start=np.array([geom.cell_start for geom in geoms], dtype=gs.np_int),
                geoms_sdf_val=np.concatenate([geom.sdf_val_flattened for geom in geoms], dtype=gs.np_float),
                geoms_sdf_grad=np.concatenate([geom.sdf_grad_flattened for geom in geoms], dtype=gs.np_float),
                geoms_sdf_max=np.array([geom.sdf_max for geom in geoms], dtype=gs.np_float),
                geoms_sdf_cell_size=np.array([geom.sdf_cell_size for geom in geoms], dtype=gs.np_float),
                geoms_sdf_closest_vert=np.concatenate(
                    [geom.sdf_closest_vert_flattened for geom in geoms], dtype=gs.np_int
                ),
            )

    @ti.kernel
    def _kernel_init_geom_fields(
        self,
        geoms_T_mesh_to_sdf: ti.types.ndarray(),
        geoms_sdf_res: ti.types.ndarray(),
        geoms_sdf_cell_start: ti.types.ndarray(),
        geoms_sdf_val: ti.types.ndarray(),
        geoms_sdf_grad: ti.types.ndarray(),
        geoms_sdf_max: ti.types.ndarray(),
        geoms_sdf_cell_size: ti.types.ndarray(),
        geoms_sdf_closest_vert: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self.solver._para_level < gs.PARA_LEVEL.PARTIAL)
        for i in range(self.solver.n_geoms):
            for j, k in ti.ndrange(4, 4):
                self.geoms_info[i].T_mesh_to_sdf[j, k] = geoms_T_mesh_to_sdf[i, j, k]

            for j in range(3):
                self.geoms_info[i].sdf_res[j] = geoms_sdf_res[i, j]

            self.geoms_info[i].sdf_cell_start = geoms_sdf_cell_start[i]
            self.geoms_info[i].sdf_max = geoms_sdf_max[i]
            self.geoms_info[i].sdf_cell_size = geoms_sdf_cell_size[i]

        for i in range(self.solver.n_cells_):
            self.geoms_sdf_val[i] = geoms_sdf_val[i]
            self.geoms_sdf_closest_vert[i] = geoms_sdf_closest_vert[i]
            for j in ti.static(range(3)):
                self.geoms_sdf_grad[i][j] = geoms_sdf_grad[i, j]

    @ti.func
    def ravel_cell_idx(self, cell_idx, sdf_res, geom_idx):
        return (
            self.geoms_info[geom_idx].sdf_cell_start
            + cell_idx[0] * sdf_res[1] * sdf_res[2]
            + cell_idx[1] * sdf_res[2]
            + cell_idx[2]
        )

    @ti.func
    def proxy_sdf_grad_sdf(self, pos_sdf, geom_idx):
        """
        Use direction to sdf center to approximate gradient direction.
        Only considers region outside of cube.
        """
        center = (self.geoms_info[geom_idx].sdf_res - 1) / 2.0
        proxy_sdf_grad = gu.ti_normalize(pos_sdf - center)
        return proxy_sdf_grad

    @ti.func
    def proxy_sdf_sdf(self, pos_sdf, geom_idx):
        """
        Use distance to center as a proxy sdf, strictly greater than any point inside the cube to ensure value comparison is valid. Only considers region outside of cube.
        """
        center = (self.geoms_info[geom_idx].sdf_res - 1) / 2.0
        sd = (pos_sdf - center).norm() / self.geoms_info[geom_idx].sdf_cell_size
        return sd + self.geoms_info[geom_idx].sdf_max

    @ti.func
    def true_sdf_grad_sdf(self, pos_sdf, geom_idx):
        """
        True sdf grad interpolated using stored sdf grad grid.
        """
        sdf_grad_sdf = ti.Vector.zero(gs.ti_float, 3)
        if self.solver.geoms_info[geom_idx].type == gs.GEOM_TYPE.TERRAIN:  # Terrain uses finite difference
            if ti.static(self.solver.collider._has_terrain):  # for speed up compilation
                # since we are in sdf frame, delta can be a relatively big value
                delta = gs.ti_float(1e-2)

                for i in ti.static(range(3)):
                    inc = pos_sdf
                    dec = pos_sdf
                    inc[i] += delta
                    dec[i] -= delta
                    sdf_grad_sdf[i] = (self.true_sdf_sdf(inc, geom_idx) - self.true_sdf_sdf(dec, geom_idx)) / (
                        2 * delta
                    )

        else:
            geom_sdf_res = self.geoms_info[geom_idx].sdf_res
            base = ti.min(ti.floor(pos_sdf, gs.ti_int), geom_sdf_res - 2)
            for offset in ti.grouped(ti.ndrange(2, 2, 2)):
                pos_cell = base + offset
                w_xyz = 1 - ti.abs(pos_sdf - pos_cell)
                w = w_xyz[0] * w_xyz[1] * w_xyz[2]
                sdf_grad_sdf = (
                    sdf_grad_sdf + w * self.geoms_sdf_grad[self.ravel_cell_idx(pos_cell, geom_sdf_res, geom_idx)]
                )

        return sdf_grad_sdf

    @ti.func
    def true_sdf_sdf(self, pos_sdf, geom_idx):
        """
        True sdf interpolated using stored sdf grid.
        """
        geom_sdf_res = self.geoms_info[geom_idx].sdf_res
        base = ti.min(ti.floor(pos_sdf, gs.ti_int), geom_sdf_res - 2)
        signed_dist = gs.ti_float(0.0)
        for offset in ti.grouped(ti.ndrange(2, 2, 2)):
            pos_cell = base + offset
            w_xyz = 1 - ti.abs(pos_sdf - pos_cell)
            w = w_xyz[0] * w_xyz[1] * w_xyz[2]
            signed_dist = signed_dist + w * self.geoms_sdf_val[self.ravel_cell_idx(pos_cell, geom_sdf_res, geom_idx)]

        return signed_dist

    @ti.func
    def sdf_world(self, pos_world, geom_idx, batch_idx):
        """
        sdf value from world coordinate
        """

        g_state = self.solver.geoms_state[geom_idx, batch_idx]

        sd = gs.ti_float(0.0)
        if self.solver.geoms_info[geom_idx].type == gs.GEOM_TYPE.SPHERE:
            sd = (pos_world - g_state.pos).norm() - self.solver.geoms_info[geom_idx].data[0]

        elif self.solver.geoms_info[geom_idx].type == gs.GEOM_TYPE.PLANE:
            pos_to_plane_center = pos_world - g_state.pos
            plane_normal = gs.ti_vec3(
                [
                    self.solver.geoms_info[geom_idx].data[0],
                    self.solver.geoms_info[geom_idx].data[1],
                    self.solver.geoms_info[geom_idx].data[2],
                ]
            )
            sd = pos_to_plane_center.dot(plane_normal)

        else:
            pos_mesh = gu.ti_inv_transform_by_trans_quat(pos_world, g_state.pos, g_state.quat)
            pos_sdf = gu.ti_transform_by_T(pos_mesh, self.geoms_info[geom_idx].T_mesh_to_sdf)
            sd = self.sdf_sdf(pos_sdf, geom_idx)

        return sd

    @ti.func
    def sdf_sdf(self, pos_sdf, geom_idx):
        """
        sdf value at sdf frame coordinate.
        Note that the stored sdf magnitude is already w.r.t world/mesh frame.
        """
        signed_dist = gs.ti_float(0.0)
        if self.is_outside_sdf_grid(pos_sdf, geom_idx):
            signed_dist = self.proxy_sdf_sdf(pos_sdf, geom_idx)
        else:
            signed_dist = self.true_sdf_sdf(pos_sdf, geom_idx)
        return signed_dist

    @ti.func
    def sdf_grad_sdf(self, pos_sdf, geom_idx):
        """
        sdf grad at sdf frame coordinate.
        Note that the stored sdf magnitude is already w.r.t world/mesh frame.
        """
        grad_sdf = ti.Vector.zero(gs.ti_float, 3)
        if self.is_outside_sdf_grid(pos_sdf, geom_idx):
            grad_sdf = self.proxy_sdf_grad_sdf(pos_sdf, geom_idx)
        else:
            grad_sdf = self.true_sdf_grad_sdf(pos_sdf, geom_idx)
        return grad_sdf

    @ti.func
    def sdf_normal_world(self, pos_world, geom_idx, batch_idx):
        return gu.ti_normalize(self.sdf_grad_world(pos_world, geom_idx, batch_idx))

    @ti.func
    def sdf_grad_world(self, pos_world, geom_idx, batch_idx):
        g_state = self.solver.geoms_state[geom_idx, batch_idx]

        grad_world = ti.Vector.zero(gs.ti_float, 3)
        if self.solver.geoms_info[geom_idx].type == gs.GEOM_TYPE.SPHERE:
            grad_world = gu.ti_normalize(pos_world - g_state.pos)

        else:
            pos_mesh = gu.ti_inv_transform_by_trans_quat(pos_world, g_state.pos, g_state.quat)
            pos_sdf = gu.ti_transform_by_T(pos_mesh, self.geoms_info[geom_idx].T_mesh_to_sdf)
            grad_sdf = self.sdf_grad_sdf(pos_sdf, geom_idx)

            grad_mesh = grad_sdf  # no rotation between mesh and sdf frame
            grad_world = gu.ti_transform_by_quat(grad_mesh, g_state.quat)
        return grad_world

    @ti.func
    def is_outside_sdf_grid(self, pos_sdf, geom_idx):
        res = self.geoms_info[geom_idx].sdf_res
        return (pos_sdf >= res - 1).any() or (pos_sdf <= 0).any()

    @ti.func
    def _func_find_closest_vert(self, pos_world, geom_idx, i_b):
        """
        Returns vert of geom that's cloest to pos_world
        """
        g_state = self.solver.geoms_state[geom_idx, i_b]
        geom_sdf_res = self.geoms_info[geom_idx].sdf_res
        pos_mesh = gu.ti_inv_transform_by_trans_quat(pos_world, g_state.pos, g_state.quat)
        pos_sdf = gu.ti_transform_by_T(pos_mesh, self.geoms_info[geom_idx].T_mesh_to_sdf)
        nearest_cell = ti.cast(ti.min(ti.max(pos_sdf, 0), geom_sdf_res - 1), gs.ti_int)
        return (
            self.geoms_sdf_closest_vert[self.ravel_cell_idx(nearest_cell, geom_sdf_res, geom_idx)]
            + self.solver.geoms_info[geom_idx].vert_start
        )
