"""
Thread-local SDF functions that accept pos/quat as parameters.

These functions are designed to work with local variable perturbations,
avoiding race conditions by not reading from global geoms_state.
"""

import gstaichi as ti
import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class

# Import the shared SDF helper functions that don't depend on geoms_state
from genesis.utils.sdf import (
    sdf_func_sdf,
    sdf_func_grad,
)


@ti.func
def sdf_func_world_local(
    geoms_info: array_class.GeomsInfo,
    sdf_info: array_class.SDFInfo,
    pos_world: ti.types.vector(3, dtype=gs.ti_float),
    geom_idx,
    geom_pos: ti.types.vector(3, dtype=gs.ti_float),
    geom_quat: ti.types.vector(4, dtype=gs.ti_float),
):
    """
    Thread-local version of sdf_func_world.
    
    Computes SDF value from world coordinate, using provided geometry pose
    instead of reading from geoms_state.
    
    Args:
        geoms_info: Geometry information
        sdf_info: SDF information
        pos_world: World position to evaluate SDF at
        geom_idx: Geometry index
        geom_pos: Thread-local geometry position
        geom_quat: Thread-local geometry quaternion
    
    Returns:
        Signed distance value
    """
    sd = gs.ti_float(0.0)
    
    if geoms_info.type[geom_idx] == gs.GEOM_TYPE.SPHERE:
        sd = (pos_world - geom_pos).norm() - geoms_info.data[geom_idx][0]
    
    elif geoms_info.type[geom_idx] == gs.GEOM_TYPE.PLANE:
        pos_mesh = gu.ti_inv_transform_by_trans_quat(pos_world, geom_pos, geom_quat)
        geom_data = geoms_info.data[geom_idx]
        plane_normal = gs.ti_vec3([geom_data[0], geom_data[1], geom_data[2]])
        sd = pos_mesh.dot(plane_normal)
    
    else:
        pos_mesh = gu.ti_inv_transform_by_trans_quat(pos_world, geom_pos, geom_quat)
        pos_sdf = gu.ti_transform_by_T(pos_mesh, sdf_info.geoms_info.T_mesh_to_sdf[geom_idx])
        sd = sdf_func_sdf(sdf_info, pos_sdf, geom_idx)
    
    return sd


@ti.func
def sdf_func_grad_world_local(
    geoms_info: array_class.GeomsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    collider_static_config: ti.template(),
    sdf_info: array_class.SDFInfo,
    pos_world: ti.types.vector(3, dtype=gs.ti_float),
    geom_idx,
    geom_pos: ti.types.vector(3, dtype=gs.ti_float),
    geom_quat: ti.types.vector(4, dtype=gs.ti_float),
):
    """
    Thread-local version of sdf_func_grad_world.
    
    Computes SDF gradient in world coordinates, using provided geometry pose
    instead of reading from geoms_state.
    
    Args:
        geoms_info: Geometry information
        rigid_global_info: Global rigid body information
        collider_static_config: Collider static configuration
        sdf_info: SDF information
        pos_world: World position to evaluate gradient at
        geom_idx: Geometry index
        geom_pos: Thread-local geometry position
        geom_quat: Thread-local geometry quaternion
    
    Returns:
        Gradient vector in world coordinates
    """
    EPS = rigid_global_info.EPS[None]
    
    grad_world = ti.Vector.zero(gs.ti_float, 3)
    
    if geoms_info.type[geom_idx] == gs.GEOM_TYPE.SPHERE:
        grad_world = gu.ti_normalize(pos_world - geom_pos, EPS)
    
    elif geoms_info.type[geom_idx] == gs.GEOM_TYPE.PLANE:
        geom_data = geoms_info.data[geom_idx]
        plane_normal = gs.ti_vec3([geom_data[0], geom_data[1], geom_data[2]])
        grad_world = gu.ti_transform_by_quat(plane_normal, geom_quat)
    
    else:
        pos_mesh = gu.ti_inv_transform_by_trans_quat(pos_world, geom_pos, geom_quat)
        pos_sdf = gu.ti_transform_by_T(pos_mesh, sdf_info.geoms_info.T_mesh_to_sdf[geom_idx])
        grad_sdf = sdf_func_grad(geoms_info, rigid_global_info, collider_static_config, sdf_info, pos_sdf, geom_idx)
        
        grad_mesh = grad_sdf  # no rotation between mesh and sdf frame
        grad_world = gu.ti_transform_by_quat(grad_mesh, geom_quat)
    
    return grad_world


@ti.func
def sdf_func_normal_world_local(
    geoms_info: array_class.GeomsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    collider_static_config: ti.template(),
    sdf_info: array_class.SDFInfo,
    pos_world: ti.types.vector(3, dtype=gs.ti_float),
    geom_idx,
    geom_pos: ti.types.vector(3, dtype=gs.ti_float),
    geom_quat: ti.types.vector(4, dtype=gs.ti_float),
):
    """
    Thread-local version of sdf_func_normal_world.
    
    Computes normalized SDF gradient (surface normal) in world coordinates,
    using provided geometry pose instead of reading from geoms_state.
    
    Args:
        geoms_info: Geometry information
        rigid_global_info: Global rigid body information
        collider_static_config: Collider static configuration
        sdf_info: SDF information
        pos_world: World position to evaluate normal at
        geom_idx: Geometry index
        geom_pos: Thread-local geometry position
        geom_quat: Thread-local geometry quaternion
    
    Returns:
        Normalized surface normal in world coordinates
    """
    return gu.ti_normalize(
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
