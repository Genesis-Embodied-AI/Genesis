import numpy as np
import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class


class SDF:
    def __init__(self, rigid_solver):
        self.solver = rigid_solver
        self._sdf_info = array_class.get_sdf_info(self.solver.n_geoms, self.solver.n_cells)
        self._is_active = False

    def activate(self):
        if self._is_active:
            return

        if self.solver.n_geoms > 0:
            geoms = self.solver.geoms
            sdf_kernel_init_geom_fields(
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
                static_rigid_sim_config=self.solver._static_rigid_sim_config,
                sdf_info=self._sdf_info,
            )

        self._is_active = True

    @property
    def is_active(self):
        return self._is_active


@qd.kernel
def sdf_kernel_init_geom_fields(
    geoms_T_mesh_to_sdf: qd.types.ndarray(),
    geoms_sdf_res: qd.types.ndarray(),
    geoms_sdf_cell_start: qd.types.ndarray(),
    geoms_sdf_val: qd.types.ndarray(),
    geoms_sdf_grad: qd.types.ndarray(),
    geoms_sdf_max: qd.types.ndarray(),
    geoms_sdf_cell_size: qd.types.ndarray(),
    geoms_sdf_closest_vert: qd.types.ndarray(),
    static_rigid_sim_config: qd.template(),
    sdf_info: array_class.SDFInfo,
):
    n_geoms = sdf_info.geoms_sdf_start.shape[0]
    n_cells = sdf_info.geoms_sdf_val.shape[0]

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i in range(n_geoms):
        for j, k in qd.static(qd.ndrange(4, 4)):
            sdf_info.geoms_info.T_mesh_to_sdf[i][j, k] = geoms_T_mesh_to_sdf[i, j, k]

        for j in qd.static(range(3)):
            sdf_info.geoms_info.sdf_res[i][j] = geoms_sdf_res[i, j]

        sdf_info.geoms_info.sdf_cell_start[i] = geoms_sdf_cell_start[i]
        sdf_info.geoms_info.sdf_max[i] = geoms_sdf_max[i]
        sdf_info.geoms_info.sdf_cell_size[i] = geoms_sdf_cell_size[i]

    for i in range(n_cells):
        sdf_info.geoms_sdf_val[i] = geoms_sdf_val[i]
        sdf_info.geoms_sdf_closest_vert[i] = geoms_sdf_closest_vert[i]
        for j in qd.static(range(3)):
            sdf_info.geoms_sdf_grad[i][j] = geoms_sdf_grad[i, j]


@qd.func
def sdf_func_world(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    sdf_info: array_class.SDFInfo,
    pos_world,
    geom_idx,
    batch_idx,
):
    """
    sdf value from world coordinate
    """

    g_pos = geoms_state.pos[geom_idx, batch_idx]
    g_quat = geoms_state.quat[geom_idx, batch_idx]

    return sdf_func_world_local(
        geoms_info=geoms_info,
        sdf_info=sdf_info,
        pos_world=pos_world,
        geom_idx=geom_idx,
        geom_pos=g_pos,
        geom_quat=g_quat,
    )


@qd.func
def sdf_func_world_local(
    geoms_info: array_class.GeomsInfo,
    sdf_info: array_class.SDFInfo,
    pos_world: qd.types.vector(3, dtype=gs.qd_float),
    geom_idx,
    geom_pos: qd.types.vector(3, dtype=gs.qd_float),
    geom_quat: qd.types.vector(4, dtype=gs.qd_float),
):
    """
    Computes SDF value from world coordinate, using provided geometry pose
    instead of reading from geoms_state.
    """
    sd = gs.qd_float(0.0)

    if geoms_info.type[geom_idx] == gs.GEOM_TYPE.SPHERE:
        sd = (pos_world - geom_pos).norm() - geoms_info.data[geom_idx][0]

    elif geoms_info.type[geom_idx] == gs.GEOM_TYPE.PLANE:
        pos_mesh = gu.qd_inv_transform_by_trans_quat(pos_world, geom_pos, geom_quat)
        geom_data = geoms_info.data[geom_idx]
        plane_normal = gs.qd_vec3([geom_data[0], geom_data[1], geom_data[2]])
        sd = pos_mesh.dot(plane_normal)

    else:
        pos_mesh = gu.qd_inv_transform_by_trans_quat(pos_world, geom_pos, geom_quat)
        pos_sdf = gu.qd_transform_by_T(pos_mesh, sdf_info.geoms_info.T_mesh_to_sdf[geom_idx])
        sd = sdf_func_sdf(sdf_info, pos_sdf, geom_idx)

    return sd


@qd.func
def sdf_func_sdf(sdf_info: array_class.SDFInfo, pos_sdf, geom_idx):
    """
    sdf value at sdf frame coordinate.
    Note that the stored sdf magnitude is already w.r.t world/mesh frame.
    """
    signed_dist = gs.qd_float(0.0)
    if sdf_func_is_outside_sdf_grid(sdf_info, pos_sdf, geom_idx):
        signed_dist = sdf_func_proxy_sdf(sdf_info, pos_sdf, geom_idx)
    else:
        signed_dist = sdf_func_true_sdf(sdf_info, pos_sdf, geom_idx)
    return signed_dist


@qd.func
def sdf_func_is_outside_sdf_grid(sdf_info: array_class.SDFInfo, pos_sdf, geom_idx):
    res = sdf_info.geoms_info.sdf_res[geom_idx]
    return (pos_sdf >= res - 1).any() or (pos_sdf <= 0).any()


@qd.func
def sdf_func_proxy_sdf(sdf_info: array_class.SDFInfo, pos_sdf, geom_idx):
    """
    Use distance to center as a proxy sdf, strictly greater than any point inside the cube to ensure value comparison is valid. Only considers region outside of cube.
    """
    center = (sdf_info.geoms_info.sdf_res[geom_idx] - 1) / 2.0
    sd = (pos_sdf - center).norm() / sdf_info.geoms_info.sdf_cell_size[geom_idx]
    return sd + sdf_info.geoms_info.sdf_max[geom_idx]


@qd.func
def sdf_func_true_sdf(sdf_info: array_class.SDFInfo, pos_sdf, geom_idx):
    """
    True sdf interpolated using stored sdf grid.
    """
    geom_sdf_res = sdf_info.geoms_info.sdf_res[geom_idx]
    base = qd.min(qd.floor(pos_sdf, gs.qd_int), geom_sdf_res - 2)
    signed_dist = gs.qd_float(0.0)
    for offset in qd.grouped(qd.ndrange(2, 2, 2)):
        pos_cell = base + offset
        w_xyz = 1 - qd.abs(pos_sdf - pos_cell)
        w = w_xyz[0] * w_xyz[1] * w_xyz[2]
        signed_dist = (
            signed_dist
            + w * sdf_info.geoms_sdf_val[sdf_func_ravel_cell_idx(sdf_info, pos_cell, geom_sdf_res, geom_idx)]
        )

    return signed_dist


@qd.func
def sdf_func_ravel_cell_idx(sdf_info: array_class.SDFInfo, cell_idx, sdf_res, geom_idx):
    return (
        sdf_info.geoms_info.sdf_cell_start[geom_idx]
        + cell_idx[0] * sdf_res[1] * sdf_res[2]
        + cell_idx[1] * sdf_res[2]
        + cell_idx[2]
    )


@qd.func
def sdf_func_grad_world(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    collider_static_config: qd.template(),
    sdf_info: array_class.SDFInfo,
    pos_world,
    geom_idx,
    batch_idx,
):
    g_pos = geoms_state.pos[geom_idx, batch_idx]
    g_quat = geoms_state.quat[geom_idx, batch_idx]

    return sdf_func_grad_world_local(
        geoms_info=geoms_info,
        rigid_global_info=rigid_global_info,
        collider_static_config=collider_static_config,
        sdf_info=sdf_info,
        pos_world=pos_world,
        geom_idx=geom_idx,
        geom_pos=g_pos,
        geom_quat=g_quat,
    )


@qd.func
def sdf_func_grad(
    geoms_info: array_class.GeomsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    collider_static_config: qd.template(),
    sdf_info: array_class.SDFInfo,
    pos_sdf,
    geom_idx,
):
    """
    sdf grad at sdf frame coordinate.
    Note that the stored sdf magnitude is already w.r.t world/mesh frame.
    """
    grad_sdf = qd.Vector.zero(gs.qd_float, 3)
    if sdf_func_is_outside_sdf_grid(sdf_info, pos_sdf, geom_idx):
        grad_sdf = sdf_func_proxy_grad(rigid_global_info, sdf_info, pos_sdf, geom_idx)
    else:
        grad_sdf = sdf_func_true_grad(geoms_info, collider_static_config, sdf_info, pos_sdf, geom_idx)
    return grad_sdf


@qd.func
def sdf_func_proxy_grad(
    rigid_global_info: array_class.RigidGlobalInfo, sdf_info: array_class.SDFInfo, pos_sdf, geom_idx
):
    """
    Use direction to sdf center to approximate gradient direction.
    Only considers region outside of cube.
    """
    center = (sdf_info.geoms_info.sdf_res[geom_idx] - 1) / 2.0
    proxy_sdf_grad = gu.qd_normalize(pos_sdf - center, rigid_global_info.EPS[None])
    return proxy_sdf_grad


@qd.func
def sdf_func_true_grad(
    geoms_info: array_class.GeomsInfo,
    collider_static_config: qd.template(),
    sdf_info: array_class.SDFInfo,
    pos_sdf,
    geom_idx,
):
    """
    True sdf grad interpolated using stored sdf grad grid.
    """
    sdf_grad_sdf = qd.Vector.zero(gs.qd_float, 3)
    if geoms_info.type[geom_idx] == gs.GEOM_TYPE.TERRAIN:  # Terrain uses finite difference
        if qd.static(collider_static_config.has_terrain):  # for speed up compilation
            # since we are in sdf frame, delta can be a relatively big value
            delta = gs.qd_float(1e-2)

            for i in qd.static(range(3)):
                inc = pos_sdf
                dec = pos_sdf
                inc[i] += delta
                dec[i] -= delta
                sdf_grad_sdf[i] = (
                    sdf_func_true_sdf(sdf_info, inc, geom_idx) - sdf_func_true_sdf(sdf_info, dec, geom_idx)
                ) / (2 * delta)

    else:
        geom_sdf_res = sdf_info.geoms_info.sdf_res[geom_idx]
        base = qd.min(qd.floor(pos_sdf, gs.qd_int), geom_sdf_res - 2)
        for offset in qd.grouped(qd.ndrange(2, 2, 2)):
            pos_cell = base + offset
            w_xyz = 1 - qd.abs(pos_sdf - pos_cell)
            w = w_xyz[0] * w_xyz[1] * w_xyz[2]
            sdf_grad_sdf = (
                sdf_grad_sdf
                + w * sdf_info.geoms_sdf_grad[sdf_func_ravel_cell_idx(sdf_info, pos_cell, geom_sdf_res, geom_idx)]
            )

    return sdf_grad_sdf


@qd.func
def sdf_func_normal_world(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    collider_static_config: qd.template(),
    sdf_info: array_class.SDFInfo,
    pos_world,
    geom_idx,
    batch_idx,
):
    g_pos = geoms_state.pos[geom_idx, batch_idx]
    g_quat = geoms_state.quat[geom_idx, batch_idx]

    return sdf_func_normal_world_local(
        geoms_info=geoms_info,
        rigid_global_info=rigid_global_info,
        collider_static_config=collider_static_config,
        sdf_info=sdf_info,
        pos_world=pos_world,
        geom_idx=geom_idx,
        geom_pos=g_pos,
        geom_quat=g_quat,
    )


@qd.func
def sdf_func_normal_world_local(
    geoms_info: array_class.GeomsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    collider_static_config: qd.template(),
    sdf_info: array_class.SDFInfo,
    pos_world: qd.types.vector(3, dtype=gs.qd_float),
    geom_idx,
    geom_pos: qd.types.vector(3, dtype=gs.qd_float),
    geom_quat: qd.types.vector(4, dtype=gs.qd_float),
):
    """
    Computes normalized SDF gradient (surface normal) in world coordinates,
    using provided geometry pose instead of reading from geoms_state.
    """
    return gu.qd_normalize(
        sdf_func_grad_world_local(
            geoms_info,
            rigid_global_info,
            collider_static_config,
            sdf_info,
            pos_world,
            geom_idx,
            geom_pos,
            geom_quat,
        ),
        rigid_global_info.EPS[None],
    )


@qd.func
def sdf_func_grad_world_local(
    geoms_info: array_class.GeomsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    collider_static_config: qd.template(),
    sdf_info: array_class.SDFInfo,
    pos_world: qd.types.vector(3, dtype=gs.qd_float),
    geom_idx,
    geom_pos: qd.types.vector(3, dtype=gs.qd_float),
    geom_quat: qd.types.vector(4, dtype=gs.qd_float),
):
    """
    Computes SDF gradient in world coordinates, using provided geometry pose
    instead of reading from geoms_state.
    """
    EPS = rigid_global_info.EPS[None]

    grad_world = qd.Vector.zero(gs.qd_float, 3)

    if geoms_info.type[geom_idx] == gs.GEOM_TYPE.SPHERE:
        grad_world = gu.qd_normalize(pos_world - geom_pos, EPS)

    elif geoms_info.type[geom_idx] == gs.GEOM_TYPE.PLANE:
        geom_data = geoms_info.data[geom_idx]
        plane_normal = gs.qd_vec3([geom_data[0], geom_data[1], geom_data[2]])
        grad_world = gu.qd_transform_by_quat(plane_normal, geom_quat)

    else:
        pos_mesh = gu.qd_inv_transform_by_trans_quat(pos_world, geom_pos, geom_quat)
        pos_sdf = gu.qd_transform_by_T(pos_mesh, sdf_info.geoms_info.T_mesh_to_sdf[geom_idx])
        grad_sdf = sdf_func_grad(geoms_info, rigid_global_info, collider_static_config, sdf_info, pos_sdf, geom_idx)

        grad_mesh = grad_sdf  # no rotation between mesh and sdf frame
        grad_world = gu.qd_transform_by_quat(grad_mesh, geom_quat)

    return grad_world


@qd.func
def sdf_func_find_closest_vert(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    sdf_info: array_class.SDFInfo,
    pos_world,
    geom_idx,
    i_b,
):
    """
    Returns vert of geom that's closest to pos_world
    """
    g_pos = geoms_state.pos[geom_idx, i_b]
    g_quat = geoms_state.quat[geom_idx, i_b]
    geom_sdf_res = sdf_info.geoms_info.sdf_res[geom_idx]
    pos_mesh = gu.qd_inv_transform_by_trans_quat(pos_world, g_pos, g_quat)
    pos_sdf = gu.qd_transform_by_T(pos_mesh, sdf_info.geoms_info.T_mesh_to_sdf[geom_idx])
    nearest_cell = qd.cast(qd.min(qd.max(pos_sdf, 0), geom_sdf_res - 1), gs.qd_int)
    return (
        sdf_info.geoms_sdf_closest_vert[sdf_func_ravel_cell_idx(sdf_info, nearest_cell, geom_sdf_res, geom_idx)]
        + geoms_info.vert_start[geom_idx]
    )
