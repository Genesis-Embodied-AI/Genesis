import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils import array_class
from genesis.utils import sdf as sdf_utils


@qd.func
def func_planner_snapshot_world(
    envs_idx: qd.types.ndarray(),
    obstacle_geoms_idx: qd.types.ndarray(),
    planner_world: array_class.PlannerWorldState,
    dyn_state: array_class.DynState,
    rigid_info: array_class.RigidInfo,
    planner_config: qd.template(),
):
    """Freeze the obstacle geoms' world poses and axis-aligned bounding boxes (AABBs) for the whole plan."""
    planner_world.n_geoms[None] = obstacle_geoms_idx.shape[0]

    qd.loop_config(serialize=qd.static(planner_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_gw, i_b_ in qd.ndrange(obstacle_geoms_idx.shape[0], envs_idx.shape[0]):
        i_g = obstacle_geoms_idx[i_gw]
        i_b = envs_idx[i_b_]
        planner_world.geoms_idx[i_gw] = i_g
        g_pos = dyn_state.geoms.pos[i_g, i_b]
        g_quat = dyn_state.geoms.quat[i_g, i_b]
        planner_world.geoms_pos[i_gw, i_b] = g_pos
        planner_world.geoms_quat[i_gw, i_b] = g_quat

        aabb_min = gu.qd_transform_by_trans_quat(rigid_info.geoms_init_AABB[i_g, 0], g_pos, g_quat)
        aabb_max = aabb_min
        for i_corner in range(1, 8):
            corner = gu.qd_transform_by_trans_quat(rigid_info.geoms_init_AABB[i_g, i_corner], g_pos, g_quat)
            aabb_min = qd.min(aabb_min, corner)
            aabb_max = qd.max(aabb_max, corner)
        planner_world.geoms_aabb_min[i_gw, i_b] = aabb_min
        planner_world.geoms_aabb_max[i_gw, i_b] = aabb_max
        planner_world.geoms_is_active[i_gw, i_b] = True


@qd.kernel
def kernel_planner_snapshot_world(
    envs_idx: qd.types.ndarray(),
    obstacle_geoms_idx: qd.types.ndarray(),
    planner_world: array_class.PlannerWorldState,
    dyn_state: array_class.DynState,
    rigid_info: array_class.RigidInfo,
    planner_config: qd.template(),
):
    func_planner_snapshot_world(envs_idx, obstacle_geoms_idx, planner_world, dyn_state, rigid_info, planner_config)


@qd.func
def func_planner_world_sd(
    i_gw,
    i_b,
    x,
    planner_world: array_class.PlannerWorldState,
    dyn_info: array_class.DynInfo,
    sdf_info: array_class.SDFInfo,
):
    """
    Signed distance from a world point to one snapshot obstacle geom.

    Analytic for box / capsule / cylinder (and sphere / plane inside sdf_func_world_local); grid signed distance
    field (SDF) for meshes and terrains. Grid answers are metric only within the geom's grid band - consumers gate
    their activation distance with planner_world.geoms_max_band.
    """
    i_g = planner_world.geoms_idx[i_gw]
    g_pos = planner_world.geoms_pos[i_gw, i_b]
    g_quat = planner_world.geoms_quat[i_gw, i_b]
    geom_type = dyn_info.geoms.type[i_g]

    sd = gs.qd_float(0.0)
    if geom_type == gs.GEOM_TYPE.BOX:
        x_local = gu.qd_inv_transform_by_trans_quat(x, g_pos, g_quat)
        half = gs.qd_vec3(
            0.5 * dyn_info.geoms.data[i_g][0], 0.5 * dyn_info.geoms.data[i_g][1], 0.5 * dyn_info.geoms.data[i_g][2]
        )
        q = qd.abs(x_local) - half
        q_out = qd.max(q, 0.0)
        sd = q_out.norm() + qd.min(qd.max(q[0], qd.max(q[1], q[2])), 0.0)
    elif geom_type == gs.GEOM_TYPE.CAPSULE:
        x_local = gu.qd_inv_transform_by_trans_quat(x, g_pos, g_quat)
        halflength = 0.5 * dyn_info.geoms.data[i_g][1]
        x_local[2] = x_local[2] - qd.math.clamp(x_local[2], -halflength, halflength)
        sd = x_local.norm() - dyn_info.geoms.data[i_g][0]
    elif geom_type == gs.GEOM_TYPE.CYLINDER:
        x_local = gu.qd_inv_transform_by_trans_quat(x, g_pos, g_quat)
        radius = dyn_info.geoms.data[i_g][0]
        halflength = 0.5 * dyn_info.geoms.data[i_g][1]
        d_radial = qd.sqrt(x_local[0] ** 2 + x_local[1] ** 2) - radius
        d_axial = qd.abs(x_local[2]) - halflength
        sd = qd.min(qd.max(d_radial, d_axial), 0.0) + qd.sqrt(qd.max(d_radial, 0.0) ** 2 + qd.max(d_axial, 0.0) ** 2)
    else:
        sd = sdf_utils.sdf_func_world_local(i_g, x, g_pos, g_quat, dyn_info.geoms, sdf_info)
    return sd


@qd.func
def func_planner_world_sd_grad(
    i_gw,
    i_b,
    x,
    planner_world: array_class.PlannerWorldState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    collider_static_config: qd.template(),
):
    """World-frame gradient of func_planner_world_sd, unit-norm away from the geom."""
    i_g = planner_world.geoms_idx[i_gw]
    g_pos = planner_world.geoms_pos[i_gw, i_b]
    g_quat = planner_world.geoms_quat[i_gw, i_b]
    geom_type = dyn_info.geoms.type[i_g]

    grad = gs.qd_vec3(0.0, 0.0, 1.0)
    if geom_type == gs.GEOM_TYPE.SPHERE:
        delta = x - g_pos
        if delta.norm() > rigid_info.EPS[None]:
            grad = delta / delta.norm()
    elif geom_type == gs.GEOM_TYPE.BOX:
        x_local = gu.qd_inv_transform_by_trans_quat(x, g_pos, g_quat)
        half = gs.qd_vec3(
            0.5 * dyn_info.geoms.data[i_g][0], 0.5 * dyn_info.geoms.data[i_g][1], 0.5 * dyn_info.geoms.data[i_g][2]
        )
        q = qd.abs(x_local) - half
        grad_local = gs.qd_vec3(0.0, 0.0, 0.0)
        if qd.max(q[0], qd.max(q[1], q[2])) > 0.0:
            # Outside: gradient follows the positive-part vector.
            q_out = qd.max(q, 0.0)
            grad_local = q_out / qd.max(q_out.norm(), rigid_info.EPS[None])
        else:
            # Inside: gradient points across the closest face.
            i_face = 0
            if q[1] > q[i_face]:
                i_face = 1
            if q[2] > q[i_face]:
                i_face = 2
            grad_local[i_face] = 1.0
        for j in qd.static(range(3)):
            if x_local[j] < 0.0:
                grad_local[j] = -grad_local[j]
        grad = gu.qd_transform_by_quat(grad_local, g_quat)
    elif geom_type == gs.GEOM_TYPE.CAPSULE:
        x_local = gu.qd_inv_transform_by_trans_quat(x, g_pos, g_quat)
        halflength = 0.5 * dyn_info.geoms.data[i_g][1]
        x_local[2] = x_local[2] - qd.math.clamp(x_local[2], -halflength, halflength)
        if x_local.norm() > rigid_info.EPS[None]:
            grad = gu.qd_transform_by_quat(x_local / x_local.norm(), g_quat)
    elif geom_type == gs.GEOM_TYPE.CYLINDER:
        x_local = gu.qd_inv_transform_by_trans_quat(x, g_pos, g_quat)
        radius = dyn_info.geoms.data[i_g][0]
        halflength = 0.5 * dyn_info.geoms.data[i_g][1]
        r_norm = qd.sqrt(x_local[0] ** 2 + x_local[1] ** 2)
        d_radial = r_norm - radius
        d_axial = qd.abs(x_local[2]) - halflength
        grad_local = gs.qd_vec3(0.0, 0.0, 1.0 if x_local[2] > 0.0 else -1.0)
        if d_radial > qd.max(d_axial, 0.0) and r_norm > rigid_info.EPS[None]:
            grad_local = gs.qd_vec3(x_local[0] / r_norm, x_local[1] / r_norm, 0.0)
        grad = gu.qd_transform_by_quat(grad_local, g_quat)
    else:
        grad = sdf_utils.sdf_func_grad_world_local(
            i_g, x, g_pos, g_quat, dyn_info.geoms, rigid_info, sdf_info, collider_static_config
        )
        if grad.norm() > rigid_info.EPS[None]:
            grad = grad / grad.norm()
    return grad


@qd.func
def func_planner_world_aabb_skip(i_gw, i_b, x, band, planner_world: array_class.PlannerWorldState):
    """True when x is farther than band from the geom's snapshot AABB - its signed distance surely exceeds band."""
    aabb_min = planner_world.geoms_aabb_min[i_gw, i_b]
    aabb_max = planner_world.geoms_aabb_max[i_gw, i_b]
    delta = qd.max(qd.max(aabb_min - x, x - aabb_max), 0.0)
    return delta.norm_sqr() > band * band
