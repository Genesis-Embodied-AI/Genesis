from typing import TYPE_CHECKING

import gstaichi as ti
import numpy as np
from genesis.engine.bvh import AABB, LBVH, STACK_SIZE
from genesis.utils.raycast import kernel_update_aabbs, ray_aabb_intersection, ray_triangle_intersection

import genesis as gs

from .ray import RayHit
from .vec3 import Vec3

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


# Constant to indicate no hit occurred
NO_HIT_DISTANCE = -1.0



@ti.kernel
def kernel_cast_single_ray_for_viewer(
    fixed_verts_state: ti.template(),
    free_verts_state: ti.template(),
    verts_info: ti.template(),
    faces_info: ti.template(),
    bvh_nodes: ti.template(),
    bvh_morton_codes: ti.template(),
    ray_start: ti.types.ndarray(ndim=1),  # [3]
    ray_direction: ti.types.ndarray(ndim=1),  # [3]
    max_range: ti.f32,
    envs_idx: ti.types.ndarray(ndim=1),  # [n_envs]
    result: ti.types.ndarray(ndim=1),  # [9]: [distance, geom_idx, hit_x, hit_y, hit_z, normal_x, normal_y, normal_z, env_idx]
):
    """
    Taichi kernel for casting a single ray for viewer interaction.
    
    This loops over all environments in envs_idx and returns the closest hit.
    
    Returns:
        result[0]: distance to hit point (NO_HIT_DISTANCE if no hit)
        result[1]: geom_idx of hit geometry
        result[2]: hit_point x coordinate
        result[3]: hit_point y coordinate
        result[4]: hit_point z coordinate
        result[5]: normal x coordinate
        result[6]: normal y coordinate
        result[7]: normal z coordinate
        result[8]: env_idx of hit environment
    """
    n_triangles = faces_info.verts_idx.shape[0]
    
    # Setup ray
    ray_start_world = ti.math.vec3(ray_start[0], ray_start[1], ray_start[2])
    ray_direction_world = ti.math.vec3(ray_direction[0], ray_direction[1], ray_direction[2])
    
    # Initialize result with no hit
    result[0] = -1.0  # NO_HIT_DISTANCE
    result[1] = -1.0  # no geom
    result[2] = 0.0  # hit_point x
    result[3] = 0.0  # hit_point y
    result[4] = 0.0  # hit_point z
    result[5] = 0.0  # normal x
    result[6] = 0.0  # normal y
    result[7] = 0.0  # normal z
    result[8] = -1.0  # no env
    
    global_closest_distance = max_range
    global_hit_face = -1
    global_hit_env_idx = -1
    global_hit_normal = ti.math.vec3(0.0, 0.0, 0.0)
    
    # Loop over all environments in envs_idx
    for i_b in range(envs_idx.shape[0]):
        rendered_env_idx = ti.cast(envs_idx[i_b], ti.i32)
        
        hit_face = -1
        closest_distance = global_closest_distance
        hit_normal = ti.math.vec3(0.0, 0.0, 0.0)
        
        # Stack for non-recursive BVH traversal
        node_stack = ti.Vector.zero(ti.i32, STACK_SIZE)
        node_stack[0] = 0  # Start at root node
        stack_idx = 1
        
        while stack_idx > 0:
            stack_idx -= 1
            node_idx = node_stack[stack_idx]
            
            node = bvh_nodes[i_b, node_idx]
            
            # Check if ray hits the node's bounding box
            aabb_t = ray_aabb_intersection(ray_start_world, ray_direction_world, node.bound.min, node.bound.max)
            
            if aabb_t >= 0.0 and aabb_t < closest_distance:
                if node.left == -1:  # Leaf node
                    # Get original triangle/face index
                    sorted_leaf_idx = node_idx - (n_triangles - 1)
                    i_f = ti.cast(bvh_morton_codes[0, sorted_leaf_idx][1], ti.i32)
                    
                    # Get triangle vertices
                    tri_vertices = ti.Matrix.zero(gs.ti_float, 3, 3)
                    for i in ti.static(range(3)):
                        i_v = faces_info.verts_idx[i_f][i]
                        i_fv = verts_info.verts_state_idx[i_v]
                        if verts_info.is_fixed[i_v]:
                            tri_vertices[:, i] = fixed_verts_state.pos[i_fv]
                        else:
                            tri_vertices[:, i] = free_verts_state.pos[i_fv, rendered_env_idx]
                    v0, v1, v2 = tri_vertices[:, 0], tri_vertices[:, 1], tri_vertices[:, 2]
                    
                    # Perform ray-triangle intersection
                    hit_result = ray_triangle_intersection(ray_start_world, ray_direction_world, v0, v1, v2)
                    
                    if hit_result.w > 0.0 and hit_result.x < closest_distance and hit_result.x >= 0.0:
                        closest_distance = hit_result.x
                        hit_face = i_f
                        # Compute triangle normal
                        edge1 = v1 - v0
                        edge2 = v2 - v0
                        hit_normal = edge1.cross(edge2).normalized()
                else:  # Internal node
                    # Push children onto stack
                    if stack_idx < ti.static(STACK_SIZE - 2):
                        node_stack[stack_idx] = node.left
                        node_stack[stack_idx + 1] = node.right
                        stack_idx += 2
        
        # Update global closest if this environment had a closer hit
        if hit_face >= 0 and closest_distance < global_closest_distance:
            global_closest_distance = closest_distance
            global_hit_face = hit_face
            global_hit_env_idx = rendered_env_idx
            global_hit_normal = hit_normal
    
    # Store result
    if global_hit_face >= 0:
        result[0] = global_closest_distance  # distance (positive value indicates hit)
        # Find which geom this face belongs to
        i_g = faces_info.geom_idx[global_hit_face]
        result[1] = gs.ti_float(i_g)
        # Compute hit point
        hit_point = ray_start_world + global_closest_distance * ray_direction_world
        result[2] = hit_point.x
        result[3] = hit_point.y
        result[4] = hit_point.z
        # Store normal
        result[5] = global_hit_normal.x
        result[6] = global_hit_normal.y
        result[7] = global_hit_normal.z
        result[8] = gs.ti_float(global_hit_env_idx)



class ViewerRaycaster:
    """
    BVH-accelerated raycaster for viewer interaction plugins.
    
    This class manages a BVH structure built from the scene's rigid geometry
    and provides efficient single-ray casting for interactive applications.
    Only considers environments specified in rendered_envs_idx.
    """

    def __init__(self, scene: "Scene"):
        """
        Initialize the ViewerRaycaster.
        
        Parameters
        ----------
        scene : Scene
            The scene to build the raycaster for.
        """
        self.scene = scene
        self.solver = scene.sim.rigid_solver

        # Store rendered_envs_idx as numpy array for Taichi kernel

        # self.rendered_envs_idx = np.asarray(scene.vis_options.rendered_envs_idx or [0], dtype=gs.np_int)
        self.rendered_envs_idx = np.asarray([0], dtype=gs.np_int)

        # Build the BVH structure for rendered environments.
        n_faces = self.solver.faces_info.geom_idx.shape[0]
        
        if n_faces == 0:
            gs.logger.warning("No faces found in scene, viewer raycasting will not work.")
            self.aabb = None
            self.bvh = None
            return
        
        self.aabb = AABB(n_batches=len(self.rendered_envs_idx), n_aabbs=n_faces)
        self.bvh = LBVH(
            self.aabb,
            max_n_query_result_per_aabb=0,  # Not used for ray queries
            n_radix_sort_groups=min(64, n_faces),
        )
        
        self.update_bvh()
    
    def update_bvh(self):
        """Update the BVH structure with current geometry state."""
        if self.bvh is None:
            return
        
        # Update vertex positions
        from genesis.engine.solvers.rigid.rigid_solver_decomp import kernel_update_all_verts
        
        kernel_update_all_verts(
            geoms_info=self.solver.geoms_info,
            geoms_state=self.solver.geoms_state,
            verts_info=self.solver.verts_info,
            free_verts_state=self.solver.free_verts_state,
            fixed_verts_state=self.solver.fixed_verts_state,
        )
        
        # Update AABBs for each rendered environment
        kernel_update_aabbs(
            free_verts_state=self.solver.free_verts_state,
            fixed_verts_state=self.solver.fixed_verts_state,
            verts_info=self.solver.verts_info,
            faces_info=self.solver.faces_info,
            aabb_state=self.aabb,
        )
        
        # Rebuild BVH
        self.bvh.build()
    
    def cast_ray(
        self,
        ray_origin: np.ndarray,
        ray_direction: np.ndarray,
        max_range: float = 1000.0,
    ) -> RayHit:
        """
        Cast a single ray against all rendered environments and return the closest hit.
        
        Parameters
        ----------
        ray_origin : np.ndarray, shape (3,)
            The origin point of the ray in world coordinates.
        ray_direction : np.ndarray, shape (3,)
            The direction vector of the ray (will be normalized).
        max_range : float, optional
            Maximum distance to check for intersections. Default is 1000.0.
        
        Returns
        -------
        RayHit
            A RayHit object containing distance, position, normal, and geom.
            If no hit, returns RayHit.no_hit().
        """
        ray_direction = ray_direction / (np.linalg.norm(ray_direction) + gs.EPS)
        
        ray_start_np = np.asarray(ray_origin, dtype=gs.np_float)
        ray_dir_np = np.asarray(ray_direction, dtype=gs.np_float)
        result_np = np.zeros(9, dtype=gs.np_float)
        
        kernel_cast_single_ray_for_viewer(
            fixed_verts_state=self.solver.fixed_verts_state,
            free_verts_state=self.solver.free_verts_state,
            verts_info=self.solver.verts_info,
            faces_info=self.solver.faces_info,
            bvh_nodes=self.bvh.nodes,
            bvh_morton_codes=self.bvh.morton_codes,
            ray_start=ray_start_np,
            ray_direction=ray_dir_np,
            max_range=max_range,
            envs_idx=self.rendered_envs_idx,
            result=result_np,
        )
        
        distance = float(result_np[0])
        if distance < NO_HIT_DISTANCE + gs.EPS:  # NO_HIT_DISTANCE
            return RayHit.no_hit()
        
        geom_idx = int(result_np[1])
        position = Vec3(result_np[2:5])
        normal = Vec3(result_np[5:8])
        geom = self.solver.geoms[geom_idx]
        
        return RayHit(distance=distance, position=position, normal=normal, geom=geom)
