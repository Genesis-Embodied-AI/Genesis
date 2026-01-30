import math
from typing import TYPE_CHECKING

import gstaichi as ti
import numpy as np

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class
from genesis.engine.solvers.rigid.collider import support_field_local

if TYPE_CHECKING:
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver


class SupportField:
    def __init__(self, rigid_solver: "RigidSolver", is_active: bool = True) -> None:
        self.solver = rigid_solver
        self._is_active = is_active

        self._support_res = 180
        if self.solver._enable_collision:
            self._init_support()

    def _get_direction_grid(self):
        support_res = self._support_res
        theta = np.arange(support_res) / support_res * 2 * math.pi - math.pi
        phi = np.arange(support_res) / support_res * math.pi

        spherical_coords = np.zeros([support_res, support_res, 2])
        spherical_coords[:, :, 0] = theta[:, None]
        spherical_coords[:, :, 1] = phi[None]

        x = np.sin(spherical_coords[:, :, 1]) * np.cos(spherical_coords[:, :, 0])
        y = np.sin(spherical_coords[:, :, 1]) * np.sin(spherical_coords[:, :, 0])
        z = np.cos(spherical_coords[:, :, 1])
        v = np.stack((x, y, z), axis=-1)
        return v

    def _init_support(self) -> None:
        if not self.is_active:
            self._support_field_info = array_class.get_support_field_info(
                self.solver.n_geoms, 0, support_res=self._support_res
            )
            return

        v = self._get_direction_grid()
        v1 = v.reshape([-1, 3])
        num_v = len(v1)

        support_v = []
        support_vid = []
        support_cell_start = []
        start = 0
        if self.solver.n_geoms > 0:
            init_pos = self.solver.verts_info.init_pos.to_numpy()
            geoms_vert_start = self.solver.geoms_info.vert_start.to_numpy()
            geoms_vert_end = self.solver.geoms_info.vert_end.to_numpy()
            for i_g in range(self.solver.n_geoms):
                this_pos = init_pos[geoms_vert_start[i_g] : geoms_vert_end[i_g]]

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

        self._support_field_info = array_class.get_support_field_info(
            self.solver.n_geoms, n_support_cells=start, support_res=self._support_res
        )

        _kernel_init_support(
            self.solver._static_rigid_sim_config,
            self._support_field_info,
            support_cell_start,
            support_v,
            support_vid,
        )

    @property
    def is_active(self):
        return self._is_active


@ti.kernel
def _kernel_init_support(
    static_rigid_sim_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_cell_start: ti.types.ndarray(),
    support_v: ti.types.ndarray(),
    support_vid: ti.types.ndarray(),
):
    n_geoms = support_field_info.support_cell_start.shape[0]
    n_support_cells = support_field_info.support_v.shape[0]

    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i in range(n_geoms):
        support_field_info.support_cell_start[i] = support_cell_start[i]

    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i in range(n_support_cells):
        support_field_info.support_vid[i] = support_vid[i]
        for j in ti.static(range(3)):
            support_field_info.support_v[i][j] = support_v[i, j]


@ti.func
def _func_support_world(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    support_field_info: array_class.SupportFieldInfo,
    d,
    i_g,
    i_b,
):
    """
    support position for a world direction
    """
    pos = geoms_state.pos[i_g, i_b]
    quat = geoms_state.quat[i_g, i_b]
    return support_field_local._func_support_world_local(geoms_info, support_field_info, d, i_g, pos, quat)


@ti.func
def _func_support_mesh(support_field_info: array_class.SupportFieldInfo, d_mesh, i_g):
    """
    support point at mesh frame coordinate.
    """
    theta = ti.atan2(d_mesh[1], d_mesh[0])  # [-pi, pi]
    phi = ti.acos(d_mesh[2])  # [0, pi]

    support_res = support_field_info.support_res[None]
    dot_max = gs.ti_float(-1e20)
    v = ti.Vector([0.0, 0.0, 0.0], dt=gs.ti_float)
    vid = 0

    ii = (theta + math.pi) / math.pi / 2 * support_res
    jj = phi / math.pi * support_res

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

        support_idx = gs.ti_int(support_field_info.support_cell_start[i_g] + i * support_res + j)
        _vid = support_field_info.support_vid[support_idx]
        pos = support_field_info.support_v[support_idx]
        dot = pos.dot(d_mesh)

        if dot > dot_max:
            v = pos
            dot_max = dot
            vid = _vid

    return v, vid


@ti.func
def _func_support_sphere(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    d,
    i_g,
    i_b,
    shrink,
):
    pos = geoms_state.pos[i_g, i_b]
    quat = geoms_state.quat[i_g, i_b]
    return support_field_local._func_support_sphere_local(geoms_info, d, i_g, pos, quat, shrink)


@ti.func
def _func_support_ellipsoid(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    d,
    i_g,
    i_b,
):
    pos = geoms_state.pos[i_g, i_b]
    quat = geoms_state.quat[i_g, i_b]
    return support_field_local._func_support_ellipsoid_local(geoms_info, d, i_g, pos, quat)


@ti.func
def _func_support_capsule(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    d,
    i_g,
    i_b,
    shrink,
):
    pos = geoms_state.pos[i_g, i_b]
    quat = geoms_state.quat[i_g, i_b]
    return support_field_local._func_support_capsule_local(geoms_info, d, i_g, pos, quat, shrink)


@ti.func
def _func_support_prism(
    collider_state: array_class.ColliderState,
    d,
    i_g,
    i_b,
):
    istart = 3
    if d[2] < 0:
        istart = 0

    ibest = istart
    best = collider_state.prism[istart, i_b].dot(d)
    for i in range(istart + 1, istart + 3):
        dot = collider_state.prism[i, i_b].dot(d)
        if dot > best:
            ibest = i
            best = dot

    return collider_state.prism[ibest, i_b], ibest


@ti.func
def _func_support_box(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    d,
    i_g,
    i_b,
):
    pos = geoms_state.pos[i_g, i_b]
    quat = geoms_state.quat[i_g, i_b]
    return support_field_local._func_support_box_local(geoms_info, d, i_g, pos, quat)


@ti.func
def _func_count_supports_world(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    support_field_info: array_class.SupportFieldInfo,
    d,
    i_g,
    i_b,
):
    """
    Count the number of valid support points for the given world direction.
    """
    quat = geoms_state.quat[i_g, i_b]
    return support_field_local._func_count_supports_world_local(geoms_info, support_field_info, d, i_g, quat)


@ti.func
def _func_count_supports_mesh(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    support_field_info: array_class.SupportFieldInfo,
    d_mesh,
    i_g,
):
    """
    Count the number of valid support points for a mesh in the given direction.
    """
    theta = ti.atan2(d_mesh[1], d_mesh[0])  # [-pi, pi]
    phi = ti.acos(d_mesh[2])  # [0, pi]

    support_res = support_field_info.support_res[None]
    dot_max = gs.ti_float(-1e20)

    ii = (theta + math.pi) / math.pi / 2 * support_res
    jj = phi / math.pi * support_res

    count = gs.ti_int(0)
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

        support_idx = gs.ti_int(support_field_info.support_cell_start[i_g] + i * support_res + j)
        _vid = support_field_info.support_vid[support_idx]
        pos = support_field_info.support_v[support_idx]
        dot = pos.dot(d_mesh)

        if dot > dot_max:
            count = 1
        elif dot == dot_max:
            count += 1

    return count


@ti.func
def _func_count_supports_box(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    d,
    i_g,
    i_b,
):
    """
    Count the number of valid support points for a box in the given direction.

    If the direction has 1 zero component, there are 2 possible support points. If the direction has 2 zero
    components, there are 4 possible support points.
    """
    quat = geoms_state.quat[i_g, i_b]
    return support_field_local._func_count_supports_box_local(d, quat)


from genesis.utils.deprecated_module_wrapper import create_virtual_deprecated_module

create_virtual_deprecated_module(__name__, "genesis.engine.solvers.rigid.support_field_decomp")
