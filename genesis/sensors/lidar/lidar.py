from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import taichi as ti
import torch
import trimesh

import genesis as gs
from genesis.engine.entities import RigidEntity
from genesis.engine.bvh import AABB, LBVH
from genesis.utils.geom import (
    transform_by_trans_quat,
    ti_transform_by_quat,
    ti_quat_mul,
    ti_normalize,
    ti_identity_quat,
    trans_quat_to_T,
    quat_to_R,
    R_to_quat,
)
from genesis.utils.misc import tensor_to_array

from ..base_sensor import Sensor


def get_link_mesh(link, use_visual_mesh=False):
    """
    Extract mesh from a given link.

    Args:
        link: Genesis link object
        use_visual_mesh: Whether to use visual mesh (True) or collision mesh (False)

    Returns:
        trimesh.Trimesh: Combined mesh for the link
    """
    meshes = []
    if use_visual_mesh:
        geoms = link.vgeoms
    else:
        geoms = link.geoms

    for i, geom in enumerate(geoms):
        T = trans_quat_to_T(
            (geom.get_pos() - link.get_pos()).cpu(),
            R_to_quat((quat_to_R(geom.get_quat()) @ torch.linalg.inv(quat_to_R(link.get_quat())))).cpu(),
        )
        if T.ndim == 3:
            T = T[0]  # NOTE: we use the canonical space so batch can be ignored
        mesh = geom.get_trimesh().copy()  # NOTE: avoid in-place write
        mesh.apply_transform(T)
        meshes.append(mesh)

    if len(meshes) == 0:
        # Return empty mesh if no geometry
        return trimesh.Trimesh()

    combined_mesh = trimesh.util.concatenate(meshes)
    return combined_mesh


# Global constants for LiDAR ray casting - will be initialized when needed
NO_HIT_RAY_VAL = None
NO_HIT_SEGMENTATION_VAL = None


def _ensure_lidar_constants_initialized():
    """Ensure LiDAR constants are initialized."""
    global NO_HIT_RAY_VAL, NO_HIT_SEGMENTATION_VAL

    if NO_HIT_RAY_VAL is None:
        NO_HIT_RAY_VAL = ti.field(dtype=ti.f32, shape=())
        NO_HIT_SEGMENTATION_VAL = ti.field(dtype=ti.i32, shape=())

        # Initialize constants
        NO_HIT_RAY_VAL[None] = 1000.0
        NO_HIT_SEGMENTATION_VAL[None] = -2


@ti.data_oriented
class LidarMeshData:
    """
    Mesh data structure with BVH acceleration for LiDAR ray casting.
    """

    def __init__(self, vertices: np.ndarray, triangles: np.ndarray):
        """
        Initialize mesh data with BVH acceleration.

        Args:
            vertices: Nx3 array of vertex positions
            triangles: Mx3 array of triangle indices
        """
        self.n_vertices = vertices.shape[0]
        self.n_triangles = triangles.shape[0]

        # Store mesh geometry
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=self.n_vertices)
        self.triangles = ti.Vector.field(3, dtype=ti.i32, shape=self.n_triangles)

        # Copy data to Taichi fields
        self.vertices.from_numpy(vertices.astype(np.float32))
        self.triangles.from_numpy(triangles.astype(np.int32))

        # Pre-store numpy arrays for kernel usage (static mesh optimization)
        self.vertices_np = vertices.astype(np.float32)
        self.triangles_np = triangles.astype(np.int32)

        # Build BVH acceleration structure
        self._build_bvh()

    def _build_bvh(self):
        """Build BVH for triangle acceleration."""
        # Ensure we have at least one triangle
        if self.n_triangles == 0:
            raise ValueError("Cannot create BVH for mesh with 0 triangles")

        # Create AABB for each triangle
        self.triangle_aabbs = AABB(n_batches=1, n_aabbs=self.n_triangles)
        self._compute_triangle_aabbs()

        # Pre-compute AABB arrays for efficient kernel usage (static mesh optimization)
        self.triangle_aabbs_min = np.zeros((self.n_triangles, 3), dtype=np.float32)
        self.triangle_aabbs_max = np.zeros((self.n_triangles, 3), dtype=np.float32)

        # Copy AABB data once during initialization
        for i in range(self.n_triangles):
            aabb = self.triangle_aabbs.aabbs[0, i]
            self.triangle_aabbs_min[i, 0] = aabb.min[0]
            self.triangle_aabbs_min[i, 1] = aabb.min[1]
            self.triangle_aabbs_min[i, 2] = aabb.min[2]
            self.triangle_aabbs_max[i, 0] = aabb.max[0]
            self.triangle_aabbs_max[i, 1] = aabb.max[1]
            self.triangle_aabbs_max[i, 2] = aabb.max[2]

        # Build the BVH tree
        self.bvh = LBVH(self.triangle_aabbs)
        self.bvh.build()

    @ti.kernel
    def _compute_triangle_aabbs(self):
        """Compute AABB for each triangle."""
        for i in range(self.n_triangles):
            # Get triangle vertices
            v0_idx = self.triangles[i][0]
            v1_idx = self.triangles[i][1]
            v2_idx = self.triangles[i][2]

            v0 = self.vertices[v0_idx]
            v1 = self.vertices[v1_idx]
            v2 = self.vertices[v2_idx]

            # Compute AABB with small epsilon for robustness
            eps = 1e-6
            min_pos = ti.min(ti.min(v0, v1), v2) - eps
            max_pos = ti.max(ti.max(v0, v1), v2) + eps

            # Store in AABB structure
            self.triangle_aabbs.aabbs[0, i].min = min_pos
            self.triangle_aabbs.aabbs[0, i].max = max_pos


# Global Taichi kernels - defined outside class to avoid initialization issues
_lidar_kernels = None


@ti.func
def ray_triangle_intersection(ray_start, ray_dir, v0, v1, v2):
    """
    Möller-Trumbore ray-triangle intersection.

    Returns: vec4(t, u, v, hit) where hit=1.0 if intersection found, 0.0 otherwise
    """
    result = ti.math.vec4(0.0, 0.0, 0.0, 0.0)

    # Compute edge vectors
    edge1 = v1 - v0
    edge2 = v2 - v0

    # Begin calculating determinant - also used to calculate u parameter
    h = ray_dir.cross(edge2)
    a = edge1.dot(h)

    # Check all conditions in sequence without early returns
    valid = True

    # Declare all variables at the top to avoid scope issues
    t = 0.0
    u = 0.0
    v = 0.0
    f = 0.0
    s = ti.math.vec3(0.0, 0.0, 0.0)
    q = ti.math.vec3(0.0, 0.0, 0.0)

    # If determinant is near zero, ray lies in plane of triangle
    if ti.abs(a) < 1e-8:
        valid = False

    if valid:
        f = 1.0 / a
        s = ray_start - v0
        u = f * s.dot(h)

        # Check u parameter bounds
        if u < 0.0 or u > 1.0:
            valid = False

    if valid:
        q = s.cross(edge1)
        v = f * ray_dir.dot(q)

        # Check v parameter bounds
        if v < 0.0 or u + v > 1.0:
            valid = False

    if valid:
        # At this stage we can compute t to find out where the intersection point is on the line
        t = f * edge2.dot(q)

        # Ray intersection
        if t <= 1e-8:  # Invalid intersection
            valid = False

    # Set result only if valid
    if valid:
        result = ti.math.vec4(t, u, v, 1.0)

    return result


@ti.func
def ray_aabb_intersection(ray_start, ray_dir, aabb_min, aabb_max):
    """
    Fast ray-AABB intersection test.
    Returns the t value of intersection, or -1.0 if no intersection.
    """
    result = -1.0

    # Use the slab method for ray-AABB intersection
    inv_dir = 1.0 / ray_dir

    # Handle potential division by zero with large values
    if ti.abs(ray_dir.x) < 1e-10:
        inv_dir.x = 1e10 if ray_dir.x >= 0.0 else -1e10
    if ti.abs(ray_dir.y) < 1e-10:
        inv_dir.y = 1e10 if ray_dir.y >= 0.0 else -1e10
    if ti.abs(ray_dir.z) < 1e-10:
        inv_dir.z = 1e10 if ray_dir.z >= 0.0 else -1e10

    t1 = (aabb_min - ray_start) * inv_dir
    t2 = (aabb_max - ray_start) * inv_dir

    tmin = ti.min(t1, t2)
    tmax = ti.max(t1, t2)

    t_near = ti.max(ti.max(tmin.x, tmin.y), tmin.z)
    t_far = ti.min(ti.min(tmax.x, tmax.y), tmax.z)

    # Check if ray intersects AABB
    if t_near <= t_far and t_far >= 0.0:
        result = ti.max(t_near, 0.0)

    return result


@ti.kernel
def lidar_cast_rays_kernel_bvh(
    # Mesh data
    mesh_vertices: ti.types.ndarray(ndim=2),  # [n_vertices, 3]
    mesh_triangles: ti.types.ndarray(ndim=2),  # [n_triangles, 3]
    # BVH data structures
    bvh_nodes: ti.template(),  # The BVH node tree
    bvh_morton_codes: ti.template(),  # Maps sorted leaves to original triangle indices
    # Per-ray data
    lidar_positions: ti.types.ndarray(ndim=3),  # [n_env, n_cam, 3]
    lidar_quaternions: ti.types.ndarray(ndim=3),  # [n_env, n_cam, 4] (wxyz format)
    ray_vectors: ti.types.ndarray(ndim=3),  # [n_scan_lines, n_points, 3]
    far_plane: ti.f32,
    # Output arrays
    hit_points: ti.types.ndarray(ndim=5),  # [n_env, n_cam, n_scan_lines, n_points, 3]
    hit_distances: ti.types.ndarray(ndim=4),  # [n_env, n_cam, n_scan_lines, n_points]
    world_frame: ti.i32,
):
    """
    Taichi kernel for LiDAR ray casting, accelerated by a Bounding Volume Hierarchy (BVH).
    """
    n_triangles = mesh_triangles.shape[0]

    # Parallel execution over all rays
    for env_id, cam_id, scan_line, point_index in ti.ndrange(
        hit_points.shape[0], hit_points.shape[1], hit_points.shape[2], hit_points.shape[3]
    ):
        # --- 1. Setup Ray ---
        lidar_position = ti.math.vec3(
            lidar_positions[env_id, cam_id, 0], lidar_positions[env_id, cam_id, 1], lidar_positions[env_id, cam_id, 2]
        )

        lidar_quat = ti.math.vec4(
            lidar_quaternions[env_id, cam_id, 1],  # x
            lidar_quaternions[env_id, cam_id, 2],  # y
            lidar_quaternions[env_id, cam_id, 3],  # z
            lidar_quaternions[env_id, cam_id, 0],  # w
        )

        ray_dir_local = ti.math.vec3(
            ray_vectors[scan_line, point_index, 0],
            ray_vectors[scan_line, point_index, 1],
            ray_vectors[scan_line, point_index, 2],
        )
        ray_dir_local = ti_normalize(ray_dir_local)

        # Transform ray direction to world coordinates.
        ray_direction_world = ti_transform_by_quat(ray_dir_local, lidar_quat)

        # --- 2. BVH Traversal ---
        min_t = far_plane
        hit_face = -1

        # Stack for non-recursive traversal, size 64 is typical for BVH
        stack = ti.Vector.zero(ti.i32, 64)
        stack[0] = 0  # Start traversal at the root node (index 0)
        stack_ptr = 1

        while stack_ptr > 0:
            stack_ptr -= 1
            node_idx = stack[stack_ptr]

            # Since n_batches=1, we index the BVH with [0, node_idx]
            node = bvh_nodes[0, node_idx]

            # Check if ray hits the node's bounding box
            aabb_t = ray_aabb_intersection(lidar_position, ray_direction_world, node.bound.min, node.bound.max)

            if aabb_t >= 0.0 and aabb_t < min_t:
                if node.left == -1:  # It's a LEAF node
                    # A leaf node corresponds to one of the sorted triangles.
                    # We need to find the original triangle index.
                    sorted_leaf_idx = node_idx - (n_triangles - 1)
                    original_tri_idx = bvh_morton_codes[0, sorted_leaf_idx][1]

                    # Get triangle vertices
                    v0_idx = mesh_triangles[original_tri_idx, 0]
                    v1_idx = mesh_triangles[original_tri_idx, 1]
                    v2_idx = mesh_triangles[original_tri_idx, 2]

                    v0 = ti.math.vec3(mesh_vertices[v0_idx, 0], mesh_vertices[v0_idx, 1], mesh_vertices[v0_idx, 2])
                    v1 = ti.math.vec3(mesh_vertices[v1_idx, 0], mesh_vertices[v1_idx, 1], mesh_vertices[v1_idx, 2])
                    v2 = ti.math.vec3(mesh_vertices[v2_idx, 0], mesh_vertices[v2_idx, 1], mesh_vertices[v2_idx, 2])

                    # Perform the expensive ray-triangle intersection test
                    hit_result = ray_triangle_intersection(lidar_position, ray_direction_world, v0, v1, v2)

                    if hit_result.w > 0.0 and hit_result.x < min_t and hit_result.x >= 0.0:
                        min_t = hit_result.x
                        hit_face = original_tri_idx
                        # hit_u, hit_v could be stored here if needed

                else:  # It's an INTERNAL node
                    # Push children onto the stack for further traversal
                    # Make sure stack doesn't overflow
                    if stack_ptr < 62:
                        stack[stack_ptr] = node.left
                        stack[stack_ptr + 1] = node.right
                        stack_ptr += 2

        # --- 3. Process Hit Result ---
        if hit_face >= 0:
            dist = min_t
            hit_distances[env_id, cam_id, scan_line, point_index] = dist

            if world_frame:
                hit_point = lidar_position + dist * ray_direction_world
                hit_points[env_id, cam_id, scan_line, point_index, 0] = hit_point.x
                hit_points[env_id, cam_id, scan_line, point_index, 1] = hit_point.y
                hit_points[env_id, cam_id, scan_line, point_index, 2] = hit_point.z
            else:
                hit_point = dist * ray_dir_local
                hit_points[env_id, cam_id, scan_line, point_index, 0] = hit_point.x
                hit_points[env_id, cam_id, scan_line, point_index, 1] = hit_point.y
                hit_points[env_id, cam_id, scan_line, point_index, 2] = hit_point.z

        else:
            hit_distances[env_id, cam_id, scan_line, point_index] = 1000.0
            hit_points[env_id, cam_id, scan_line, point_index, 0] = 0.0
            hit_points[env_id, cam_id, scan_line, point_index, 1] = 0.0
            hit_points[env_id, cam_id, scan_line, point_index, 2] = 0.0


@ti.data_oriented
class LidarKernels:
    """
    Simplified LiDAR kernels without complex BVH integration.
    """

    def __init__(self):
        self.mesh_data = None

    def register_mesh(self, vertices: np.ndarray, triangles: np.ndarray):
        """
        Register a mesh for ray casting.

        Args:
            vertices: Nx3 array of vertex positions
            triangles: Mx3 array of triangle indices
        """
        self.mesh_data = LidarMeshData(vertices, triangles)

    def cast_rays(
        self, lidar_positions, lidar_quaternions, ray_vectors, far_plane, hit_points, hit_distances, world_frame
    ):
        """Call the Taichi kernel for ray casting."""
        if self.mesh_data is None:
            raise RuntimeError("No mesh registered")

        # Call the BVH-accelerated kernel
        lidar_cast_rays_kernel_bvh(
            self.mesh_data.vertices_np,
            self.mesh_data.triangles_np,
            self.mesh_data.bvh.nodes,
            self.mesh_data.bvh.morton_codes,
            lidar_positions,
            lidar_quaternions,
            ray_vectors,
            far_plane,
            hit_points,
            hit_distances,
            world_frame,
        )


@ti.data_oriented
class LidarSensor(Sensor):
    """
    LiDAR sensor that performs ray casting to get distance measurements and point clouds.

    Parameters
    ----------
    entity : RigidEntity
        The entity to which this sensor is attached.
    link_idx : int, optional
        The index of the link to which this sensor is attached. If None, defaults to the base link.
    use_local_frame : bool
        Whether to return points in the local frame of the sensor. Defaults to False (world frame).
    config : dict, optional
        LiDAR configuration with parameters like:
        - n_scan_lines: Number of vertical scan lines
        - n_points_per_line: Number of horizontal points per scan line
        - fov_vertical: Vertical field of view in degrees
        - fov_horizontal: Horizontal field of view in degrees
        - max_range: Maximum sensing range
        - min_range: Minimum sensing range
    """

    _mesh_registered = False
    _kernels = None
    _scene_geometry_cache = None
    _scene_mesh_info = None  # Store detailed mesh information
    _scene_mesh_data = None  # Store original mesh data for static extraction

    # TODO: Future API for dynamic mesh support
    # _dynamic_entities = []  # List of entities that can move
    # _update_dynamic_mesh = False  # Flag to update dynamic entities

    def __init__(
        self,
        entity: RigidEntity,
        link_idx: Optional[int] = None,
        use_local_frame: bool = False,
        n_scan_lines: int = 32,
        n_points_per_line: int = 64,
        fov_vertical: float = 30.0,
        fov_horizontal: float = 360.0,
        max_range: float = 20.0,
        min_range: float = 0.1,
    ):  # Yiling: let's use direct argument for better typing

        self._entity = entity
        self._sim = entity._sim
        self.link_idx = link_idx if link_idx is not None else entity.base_link_idx
        self._use_local_frame = use_local_frame

        self.config = {
            "n_scan_lines": n_scan_lines,
            "n_points_per_line": n_points_per_line,
            "fov_vertical": fov_vertical,
            "fov_horizontal": fov_horizontal,
            "max_range": max_range,
            "min_range": min_range,
        }

        # Initialize kernels globally if not done
        if LidarSensor._kernels is None:
            # Ensure Taichi constants are initialized
            _ensure_lidar_constants_initialized()
            LidarSensor._kernels = LidarKernels()

        # Generate ray pattern
        self.ray_vectors = self._create_ray_pattern()

        # Note: For static mesh approach, no per-instance caching needed
        # TODO: Add dynamic entity tracking for future dynamic mesh support

    def _create_ray_pattern(self) -> np.ndarray:
        """Create LiDAR ray pattern based on configuration."""
        n_scan_lines = self.config["n_scan_lines"]
        n_points_per_line = self.config["n_points_per_line"]
        fov_v = np.radians(self.config["fov_vertical"])
        fov_h = np.radians(self.config["fov_horizontal"])

        # Create angular grids
        vertical_angles = np.linspace(-fov_v / 2, fov_v / 2, n_scan_lines)
        horizontal_angles = np.linspace(-fov_h / 2, fov_h / 2, n_points_per_line)

        # Generate ray vectors in spherical coordinates
        ray_vectors = np.zeros((n_scan_lines, n_points_per_line, 3), dtype=np.float32)

        for i, v_angle in enumerate(vertical_angles):
            for j, h_angle in enumerate(horizontal_angles):
                # Convert spherical to cartesian (x=forward, y=left, z=up)
                ray_vectors[i, j, 0] = np.cos(v_angle) * np.cos(h_angle)  # x (forward)
                ray_vectors[i, j, 1] = np.cos(v_angle) * np.sin(h_angle)  # y (left)
                ray_vectors[i, j, 2] = np.sin(v_angle)  # z (up)

        return ray_vectors

    def _extract_static_scene_data(self):
        """Extract static mesh data from scene entities once during initialization."""
        if LidarSensor._scene_mesh_data is not None:
            return  # Already extracted

        import time

        start_time = time.time()

        mesh_data = {"entities": [], "total_geometry_count": 0}

        # Process rigid entities
        if self._sim.rigid_solver.is_active():
            for rigid_entity in self._sim.rigid_solver.entities:
                # Skip the LiDAR sensor's own entity to avoid self-detection
                if rigid_entity == self._entity:
                    continue

                entity_data = {"type": "rigid", "entity": rigid_entity, "geometries": []}

                # Choose visual or collision geometry based on surface vis_mode
                if rigid_entity.surface.vis_mode == "visual":
                    geoms = rigid_entity.vgeoms
                else:
                    geoms = rigid_entity.geoms

                for geom_idx, geom in enumerate(geoms):
                    try:
                        # Get the original trimesh from the geometry (in local coordinates)
                        if "sdf" in rigid_entity.surface.vis_mode:
                            mesh = geom.get_sdf_trimesh()
                        else:
                            mesh = geom.get_trimesh()

                        if len(mesh.vertices) > 0 and len(mesh.faces) > 0:
                            # Store original mesh data (no transformation applied)
                            geom_data = {
                                "geom": geom,
                                "vertices": mesh.vertices.astype(np.float32),
                                "faces": mesh.faces.astype(np.int32),
                                "vertex_count": len(mesh.vertices),
                                "triangle_count": len(mesh.faces),
                            }
                            entity_data["geometries"].append(geom_data)
                            mesh_data["total_geometry_count"] += 1

                    except Exception as e:
                        print(f"Warning: Could not extract mesh from geom {geom}: {e}")
                        continue

                if entity_data["geometries"]:  # Only add if we found geometries
                    mesh_data["entities"].append(entity_data)

        # Process avatar entities if present
        if hasattr(self._sim, "avatar_solver") and self._sim.avatar_solver.is_active():
            for avatar_entity in self._sim.avatar_solver.entities:
                if avatar_entity == self._entity:
                    continue

                entity_data = {"type": "avatar", "entity": avatar_entity, "geometries": []}

                # Choose visual or collision geometry
                if avatar_entity.surface.vis_mode == "visual":
                    geoms = avatar_entity.vgeoms
                else:
                    geoms = avatar_entity.geoms

                for geom in geoms:
                    try:
                        if "sdf" in avatar_entity.surface.vis_mode:
                            mesh = geom.get_sdf_trimesh()
                        else:
                            mesh = geom.get_trimesh()

                        if len(mesh.vertices) > 0 and len(mesh.faces) > 0:
                            geom_data = {
                                "geom": geom,
                                "vertices": mesh.vertices.astype(np.float32),
                                "faces": mesh.faces.astype(np.int32),
                                "vertex_count": len(mesh.vertices),
                                "triangle_count": len(mesh.faces),
                            }
                            entity_data["geometries"].append(geom_data)
                            mesh_data["total_geometry_count"] += 1

                    except Exception as e:
                        print(f"Warning: Could not extract mesh from avatar geom {geom}: {e}")
                        continue

                if entity_data["geometries"]:
                    mesh_data["entities"].append(entity_data)

        # Process tool entities if present
        if hasattr(self._sim, "tool_solver") and self._sim.tool_solver.is_active():
            for tool_entity in self._sim.tool_solver.entities:
                if tool_entity == self._entity:
                    continue

                entity_data = {"type": "tool", "entity": tool_entity, "geometries": []}

                try:
                    vertices = tool_entity.mesh.raw_vertices
                    faces = tool_entity.mesh.faces_np

                    if len(vertices) > 0 and len(faces) > 0:
                        geom_data = {
                            "tool_mesh": tool_entity.mesh,
                            "vertices": vertices.astype(np.float32),
                            "faces": faces.astype(np.int32),
                            "vertex_count": len(vertices),
                            "triangle_count": len(faces),
                        }
                        entity_data["geometries"].append(geom_data)
                        mesh_data["total_geometry_count"] += 1

                except Exception as e:
                    print(f"Warning: Could not extract mesh from tool entity {tool_entity}: {e}")
                    continue

                if entity_data["geometries"]:
                    mesh_data["entities"].append(entity_data)

        end_time = time.time()
        mesh_data["extraction_time"] = (end_time - start_time) * 1000  # in milliseconds

        LidarSensor._scene_mesh_data = mesh_data
        print(f"LiDAR: Extracted static scene data with {mesh_data['total_geometry_count']} geometries")
        print(f"       from {len(mesh_data['entities'])} entities in {mesh_data['extraction_time']:.1f}ms")

        # Check if we're dealing with multi-environment scenario
        n_envs = getattr(self._sim, "n_envs", 1)
        if n_envs > 1:
            print(f"Note: Genesis replicates entities across {n_envs} environments.")
            print(f"      LiDAR uses mesh data from all replicated entities with proper world transforms.")

    def _extract_scene_geometry(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract static mesh data from Genesis scene once during initialization."""
        # Only extract once - static mesh approach
        if LidarSensor._scene_geometry_cache is not None:
            return LidarSensor._scene_geometry_cache

        # Ensure static mesh data is extracted
        if LidarSensor._scene_mesh_data is None:
            self._extract_static_scene_data()

        import time

        start_time = time.time()

        all_vertices = []
        all_triangles = []
        vertex_offset = 0

        # Store detailed mesh information for debugging
        mesh_info = {
            "entities": [],
            "total_vertices": 0,
            "total_triangles": 0,
            "geometry_count": 0,
            "extraction_time": None,
        }

        # Process each entity using pre-extracted static data
        # For static mesh, we transform to world coordinates at initialization time
        for entity_data in LidarSensor._scene_mesh_data["entities"]:
            entity = entity_data["entity"]
            entity_info = {
                "type": entity_data["type"],
                "entity": entity,
                "geometries": [],
                "vertex_count": 0,
                "triangle_count": 0,
            }

            # Get entity world transformation at initialization time
            try:
                entity_pos = entity.get_pos()
                entity_quat = entity.get_quat()

                # Convert to numpy arrays for transformation
                if hasattr(entity_pos, "cpu"):
                    entity_pos_np = entity_pos.cpu().numpy()
                else:
                    entity_pos_np = np.array(entity_pos)

                if hasattr(entity_quat, "cpu"):
                    entity_quat_np = entity_quat.cpu().numpy()
                else:
                    entity_quat_np = np.array(entity_quat)

                # Compute world transformation matrix
                world_transform = trans_quat_to_T(entity_pos_np, entity_quat_np)
                if world_transform.ndim == 3:
                    world_transform = world_transform[0]

                # Process each geometry for this entity
                for geom_data in entity_data["geometries"]:
                    # Apply world transformation to pre-extracted vertices
                    original_vertices = geom_data["vertices"]
                    original_faces = geom_data["faces"]

                    # Transform vertices to world coordinates (static - done once)
                    vertices_homogeneous = np.hstack(
                        [original_vertices, np.ones((len(original_vertices), 1), dtype=np.float32)]
                    )
                    transformed_vertices = (world_transform @ vertices_homogeneous.T).T[:, :3]

                    # Store geometry info
                    geom_info = {
                        "geom": geom_data.get("geom"),
                        "vertices": len(transformed_vertices),
                        "triangles": len(original_faces),
                        "vertex_offset": vertex_offset,
                        "world_pos": entity_pos_np,
                        "world_quat": entity_quat_np,
                    }
                    entity_info["geometries"].append(geom_info)
                    entity_info["vertex_count"] += len(transformed_vertices)
                    entity_info["triangle_count"] += len(original_faces)
                    mesh_info["geometry_count"] += 1

                    # Add transformed vertices and faces
                    all_vertices.append(transformed_vertices.astype(np.float32))
                    triangles = original_faces.astype(np.int32) + vertex_offset
                    all_triangles.append(triangles)

                    vertex_offset += len(transformed_vertices)

            except Exception as e:
                print(f"Warning: Could not get transformation for entity {entity}: {e}")
                continue

            if entity_info["geometries"]:  # Only add if we found geometries
                mesh_info["entities"].append(entity_info)

        # Combine all meshes
        if len(all_vertices) > 0:
            combined_vertices = np.vstack(all_vertices)
            combined_triangles = np.vstack(all_triangles)
        else:
            # If no geometry found, create a minimal ground plane
            print("Warning: No scene geometry found, creating minimal ground plane")
            ground_size = 50.0
            combined_vertices = np.array(
                [
                    [-ground_size, -ground_size, 0],
                    [ground_size, -ground_size, 0],
                    [ground_size, ground_size, 0],
                    [-ground_size, ground_size, 0],
                ],
                dtype=np.float32,
            )
            combined_triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

            # Add ground plane to mesh info
            mesh_info["entities"].append(
                {
                    "type": "ground_plane",
                    "entity": None,
                    "geometries": [{"vertices": 4, "triangles": 2, "vertex_offset": 0}],
                    "vertex_count": 4,
                    "triangle_count": 2,
                }
            )
            mesh_info["geometry_count"] += 1

        # Finalize mesh info
        end_time = time.time()
        mesh_info["total_vertices"] = len(combined_vertices)
        mesh_info["total_triangles"] = len(combined_triangles)
        mesh_info["extraction_time"] = (end_time - start_time) * 1000  # in milliseconds

        # Cache the static mesh data permanently
        LidarSensor._scene_geometry_cache = (combined_vertices, combined_triangles)
        LidarSensor._scene_mesh_info = mesh_info

        print(
            f"LiDAR: Extracted static scene geometry with {len(combined_vertices)} vertices and {len(combined_triangles)} triangles"
        )
        print(f"       from {mesh_info['geometry_count']} geometries across {len(mesh_info['entities'])} entities")
        print(f"       static mesh extraction took {mesh_info['extraction_time']:.1f}ms")

        return LidarSensor._scene_geometry_cache

    def _ensure_mesh_registered(self):
        """Ensure the scene mesh is registered with the kernels (static mesh - done once)."""
        # Only register once for static mesh
        if not LidarSensor._mesh_registered:
            vertices, triangles = self._extract_scene_geometry()
            LidarSensor._kernels.register_mesh(vertices, triangles)
            LidarSensor._mesh_registered = True

    def read(self, envs_idx: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read the LiDAR sensor data.

        Args:
            envs_idx: Optional list of environment indices to read from.

        Returns:
            hit_points: Hit points array [n_env, n_scan_lines, n_points, 3]
            hit_distances: Hit distances array [n_env, n_scan_lines, n_points]
        """
        # Ensure static mesh is registered (done once)
        self._ensure_mesh_registered()

        # Get sensor pose
        link_pos = self._sim.rigid_solver.get_links_pos(links_idx=self.link_idx).squeeze(axis=1)
        link_quat = self._sim.rigid_solver.get_links_quat(links_idx=self.link_idx).squeeze(axis=1)

        n_envs = link_pos.shape[0]
        n_scan_lines, n_points = self.ray_vectors.shape[:2]

        # Prepare output arrays
        hit_points = np.zeros((n_envs, 1, n_scan_lines, n_points, 3), dtype=np.float32)
        hit_distances = np.zeros((n_envs, 1, n_scan_lines, n_points), dtype=np.float32)

        # Convert to numpy arrays and reshape for kernel
        lidar_positions = tensor_to_array(link_pos).reshape(n_envs, 1, 3)
        lidar_quaternions = tensor_to_array(link_quat).reshape(n_envs, 1, 4)  # wxyz format

        # Call Taichi kernel for ray casting against static mesh
        LidarSensor._kernels.cast_rays(
            lidar_positions,
            lidar_quaternions,
            self.ray_vectors,
            self.config["max_range"],
            hit_points,
            hit_distances,
            0 if self._use_local_frame else 1,
        )

        # Remove the camera dimension (we only have 1 camera per sensor)
        hit_points = hit_points.squeeze(1)  # [n_env, n_scan_lines, n_points, 3]
        hit_distances = hit_distances.squeeze(1)  # [n_env, n_scan_lines, n_points]

        # Return requested subset
        if envs_idx is not None:
            return hit_points[envs_idx], hit_distances[envs_idx]
        else:
            return hit_points, hit_distances

    def get_point_cloud(self, envs_idx: Optional[List[int]] = None) -> np.ndarray:
        """
        Get the point cloud from the LiDAR sensor.

        Args:
            envs_idx: Optional list of environment indices to read from.

        Returns:
            Point cloud array [n_env, n_points, 3] where n_points = n_scan_lines * n_points_per_line
        """
        hit_points, hit_distances = self.read(envs_idx)

        if hit_points is None:
            return None

        # Filter out invalid points (beyond max range)
        valid_mask = hit_distances < self.config["max_range"]

        # Reshape to flat point cloud
        n_envs = hit_points.shape[0]
        n_total_points = hit_points.shape[1] * hit_points.shape[2]

        point_cloud = hit_points.reshape(n_envs, n_total_points, 3)
        valid_mask = valid_mask.reshape(n_envs, n_total_points)

        # Zero out invalid points
        point_cloud[~valid_mask] = 0.0

        return point_cloud

    def get_distances(self, envs_idx: Optional[List[int]] = None) -> np.ndarray:
        """
        Get the distance measurements from the LiDAR sensor.

        Args:
            envs_idx: Optional list of environment indices to read from.

        Returns:
            Distance array [n_env, n_scan_lines, n_points]
        """
        _, hit_distances = self.read(envs_idx)
        return hit_distances

    @property
    def n_scan_lines(self) -> int:
        """Number of vertical scan lines."""
        return self.config["n_scan_lines"]

    @property
    def n_points_per_line(self) -> int:
        """Number of horizontal points per scan line."""
        return self.config["n_points_per_line"]

    @property
    def max_range(self) -> float:
        """Maximum sensing range."""
        return self.config["max_range"]

    @property
    def min_range(self) -> float:
        """Minimum sensing range."""
        return self.config["min_range"]

    def get_mesh_info(self) -> Optional[Dict]:
        """
        Get detailed information about the extracted scene mesh.

        Returns:
            Dictionary containing mesh information:
            - entities: List of entity information with geometry details
            - total_vertices: Total number of vertices in the scene
            - total_triangles: Total number of triangles in the scene
            - geometry_count: Number of geometries processed
            - extraction_time: Time taken to extract meshes (in milliseconds)
        """
        # Ensure mesh is extracted
        self._extract_scene_geometry()
        return LidarSensor._scene_mesh_info

    def get_scene_mesh(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get the combined scene mesh data used for ray casting.

        Returns:
            vertices: Nx3 array of vertex positions
            triangles: Mx3 array of triangle indices
        """
        try:
            return self._extract_scene_geometry()
        except Exception as e:
            print(f"Error extracting scene mesh: {e}")
            return None, None

    def save_scene_mesh(self, filepath: str) -> bool:
        """
        Save the extracted scene mesh to a file.

        Args:
            filepath: Path where to save the mesh (supports .obj, .ply, .stl formats)

        Returns:
            True if successful, False otherwise
        """
        try:
            vertices, triangles = self._extract_scene_geometry()
            if vertices is not None and triangles is not None:
                # Create trimesh object
                mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
                mesh.export(filepath)
                print(f"LiDAR: Scene mesh saved to {filepath}")
                return True
            else:
                print("LiDAR: No scene mesh available to save")
                return False
        except Exception as e:
            print(f"LiDAR: Error saving scene mesh to {filepath}: {e}")
            return False

    def print_mesh_summary(self):
        """Print a summary of the extracted scene mesh information."""
        mesh_info = self.get_mesh_info()
        if mesh_info is None:
            print("LiDAR: No mesh information available")
            return

        print("=== LiDAR Scene Mesh Summary ===")
        print(f"Total vertices: {mesh_info['total_vertices']}")
        print(f"Total triangles: {mesh_info['total_triangles']}")
        print(f"Geometry count: {mesh_info['geometry_count']}")
        print(f"Entity count: {len(mesh_info['entities'])}")
        print(f"Extraction time: {mesh_info['extraction_time']:.1f}ms")
        print()

        for i, entity_info in enumerate(mesh_info["entities"]):
            print(f"Entity {i+1} ({entity_info['type']}):")
            print(f"  Vertices: {entity_info['vertex_count']}")
            print(f"  Triangles: {entity_info['triangle_count']}")
            print(f"  Geometries: {len(entity_info['geometries'])}")

            for j, geom_info in enumerate(entity_info["geometries"]):
                print(f"    Geometry {j+1}: {geom_info['vertices']} vertices, {geom_info['triangles']} triangles")
        print("================================")

    # TODO: Future API for dynamic mesh support
    def add_dynamic_entity(self, entity):
        """
        Add an entity to the dynamic mesh tracking list (future feature).

        Args:
            entity: Entity that can move and needs mesh updates
        """
        raise NotImplementedError("Dynamic mesh support not yet implemented")

    def update_dynamic_meshes(self):
        """
        Update meshes for dynamic entities (future feature).
        This would re-transform only the meshes of entities that have moved.
        """
        raise NotImplementedError("Dynamic mesh support not yet implemented")

    def set_static_mode(self, static: bool = True):
        """
        Toggle between static and dynamic mesh modes (future feature).

        Args:
            static: If True, use static mesh (current behavior).
                   If False, enable dynamic mesh updates.
        """
        if not static:
            raise NotImplementedError("Dynamic mesh support not yet implemented")
        # Static mode is always enabled for now
