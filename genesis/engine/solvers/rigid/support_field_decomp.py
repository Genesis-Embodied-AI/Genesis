from typing import TYPE_CHECKING
from math import pi
from dataclasses import dataclass

import numpy as np
import gstaichi as ti

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class

if TYPE_CHECKING:
    from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver


@ti.data_oriented
class SupportField:
    # @dataclass(frozen=True)
    # class SupportFieldStaticConfig:
    #     # store static arguments here
    #     support_res: int = 180  # resolution of the support field

    @ti.data_oriented
    class SupportFieldStaticConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def __init__(self, rigid_solver: "RigidSolver") -> None:
        self.solver = rigid_solver
        self._support_field_static_config = SupportField.SupportFieldStaticConfig(
            support_res=180,
        )
        if self.solver._enable_collision:
            self._compute_support()

    def _get_direction_grid(self):
        support_res = self._support_field_static_config.support_res
        theta = np.arange(support_res) / support_res * 2 * pi - pi
        phi = np.arange(support_res) / support_res * pi

        spherical_coords = np.zeros([support_res, support_res, 2])
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

        n_support_cells = start
        self._support_field_info = array_class.get_support_field_info(self.solver.n_geoms, n_support_cells)

        _kernel_init_support(
            self.solver._static_rigid_sim_config,
            self.solver._static_rigid_sim_cache_key,
            self._support_field_info,
            support_cell_start,
            support_v,
            support_vid,
        )


@ti.kernel
def _kernel_init_support(
    static_rigid_sim_config: ti.template(),
    static_rigid_sim_cache_key: array_class.StaticRigidSimCacheKey,
    support_field_info: array_class.SupportFieldInfo,
    support_cell_start: ti.types.ndarray(),
    support_v: ti.types.ndarray(),
    support_vid: ti.types.ndarray(),
):
    n_geoms = support_field_info.support_cell_start.shape[0]
    n_support_cells = support_field_info.support_v.shape[0]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i in range(n_geoms):
        support_field_info.support_cell_start[i] = support_cell_start[i]

    for i in range(n_support_cells):
        support_field_info.support_vid[i] = support_vid[i]
        for j in ti.static(range(3)):
            support_field_info.support_v[i][j] = support_v[i, j]


@ti.func
def _func_support_world(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    d,
    i_f,
    i_g,
    i_b,
):
    """
    support position for a world direction
    """

    g_pos = geoms_state.pos[i_f, i_g, i_b]
    g_quat = geoms_state.quat[i_f, i_g, i_b]
    d_mesh = gu.ti_transform_by_quat(d, gu.ti_inv_quat(g_quat))
    v, vid = _func_support_mesh(support_field_info, support_field_static_config, d_mesh, i_g)
    v_ = gu.ti_transform_by_trans_quat(v, g_pos, g_quat)
    return v_, vid


@ti.func
def _func_support_mesh(
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    d_mesh,
    i_g,
):
    """
    support point at mesh frame coordinate.
    """
    theta = ti.atan2(d_mesh[1], d_mesh[0])  # [-pi, pi]
    phi = ti.acos(d_mesh[2])  # [0, pi]

    support_res = gs.ti_int(support_field_static_config.support_res)
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
    i_f,
    i_g,
    i_b,
    shrink,
):
    sphere_center = geoms_state.pos[i_f, i_g, i_b]
    sphere_radius = geoms_info.data[i_g][0]

    # Shrink the sphere to a point
    res = sphere_center
    if not shrink:
        res += d * sphere_radius
    return res


@ti.func
def _func_support_ellipsoid(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    d,
    i_f,
    i_g,
    i_b,
):
    ellipsoid_center = geoms_state.pos[i_f, i_g, i_b]
    ellipsoid_scaled_axis = ti.Vector(
        [
            geoms_info.data[i_g][0] ** 2,
            geoms_info.data[i_g][1] ** 2,
            geoms_info.data[i_g][2] ** 2,
        ],
        dt=gs.ti_float,
    )
    ellipsoid_scaled_axis = gu.ti_transform_by_quat(ellipsoid_scaled_axis, geoms_state.quat[i_f, i_g, i_b])
    dist = ellipsoid_scaled_axis / ti.sqrt(d.dot(1.0 / ellipsoid_scaled_axis))
    return ellipsoid_center + d * dist


@ti.func
def _func_support_capsule(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    d,
    i_f,
    i_g,
    i_b,
    shrink,
):
    res = gs.ti_vec3(0, 0, 0)
    g_pos = geoms_state.pos[i_f, i_g, i_b]
    g_quat = geoms_state.quat[i_f, i_g, i_b]
    capsule_center = g_pos
    capsule_radius = geoms_info.data[i_g][0]
    capsule_halflength = 0.5 * geoms_info.data[i_g][1]

    if shrink:
        local_dir = gu.ti_transform_by_quat(d, gu.ti_inv_quat(g_quat))
        res[2] = capsule_halflength if local_dir[2] >= 0.0 else -capsule_halflength
        res = gu.ti_transform_by_trans_quat(res, capsule_center, g_quat)
    else:
        capsule_axis = gu.ti_transform_by_quat(ti.Vector([0.0, 0.0, 1.0], dt=gs.ti_float), g_quat)
        capsule_endpoint_side = -1.0 if d.dot(capsule_axis) < 0.0 else 1.0
        capsule_endpoint = capsule_center + capsule_halflength * capsule_endpoint_side * capsule_axis
        res = capsule_endpoint + d * capsule_radius
    return res


@ti.func
def _func_support_prism(
    collider_state: array_class.ColliderState,
    d,
    i_f,
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
    i_f,
    i_g,
    i_b,
):
    g_pos = geoms_state.pos[i_f, i_g, i_b]
    g_quat = geoms_state.quat[i_f, i_g, i_b]
    d_box = gu.ti_inv_transform_by_quat(d, g_quat)

    v_ = ti.Vector(
        [
            (-1.0 if d_box[0] < 0.0 else 1.0) * geoms_info.data[i_g][0] * 0.5,
            (-1.0 if d_box[1] < 0.0 else 1.0) * geoms_info.data[i_g][1] * 0.5,
            (-1.0 if d_box[2] < 0.0 else 1.0) * geoms_info.data[i_g][2] * 0.5,
        ],
        dt=gs.ti_float,
    )
    vid = (v_[0] > 0.0) * 1 + (v_[1] > 0.0) * 2 + (v_[2] > 0.0) * 4
    vid += geoms_info.vert_start[i_g]
    v = gu.ti_transform_by_trans_quat(v_, g_pos, g_quat)
    return v, vid


@ti.func
def _func_count_supports_world(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    d,
    i_f,
    i_g,
    i_b,
):
    """
    Count the number of valid support points for the given world direction.
    """
    d_mesh = gu.ti_transform_by_quat(d, gu.ti_inv_quat(geoms_state.quat[i_f, i_g, i_b]))
    return _func_count_supports_mesh(
        geoms_state, geoms_info, support_field_info, support_field_static_config, d_mesh, i_g
    )


@ti.func
def _func_count_supports_mesh(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    d_mesh,
    i_g,
):
    """
    Count the number of valid support points for a mesh in the given direction.
    """
    theta = ti.atan2(d_mesh[1], d_mesh[0])  # [-pi, pi]
    phi = ti.acos(d_mesh[2])  # [0, pi]

    support_res = gs.ti_int(support_field_static_config.support_res)
    dot_max = gs.ti_float(-1e20)

    ii = (theta + pi) / pi / 2 * support_res
    jj = phi / pi * support_res

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
    i_f,
    i_g,
    i_b,
):
    """
    Count the number of valid support points for a box in the given direction.

    If the direction has 1 zero component, there are 2 possible support points. If the direction has 2 zero
    components, there are 4 possible support points.
    """
    g_quat = geoms_state.quat[i_f, i_g, i_b]
    d_box = gu.ti_inv_transform_by_quat(d, g_quat)

    return 2 ** (d_box == 0.0).cast(gs.ti_int).sum()
