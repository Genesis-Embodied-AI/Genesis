from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import taichi as ti
import torch
import trimesh

import genesis as gs
import genesis.utils.array_class as array_class
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
import genesis.engine.solvers.rigid.rigid_solver_decomp as rigid_solver_decomp


from ..base_sensor import Sensor

MapLidarFaces = ti.template()


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
def kernel_update_aabbs(
    map_lidar_faces: MapLidarFaces,
    free_verts_state: array_class.VertsState,
    fixed_verts_state: array_class.VertsState,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    aabb_state: array_class.AABBState,
):
    _B = free_verts_state.pos.shape[1]
    # n_faces = faces_info.geom_idx.shape[0]
    n_faces = map_lidar_faces.shape[0]
    # step 1: update free verts
    for i_b, i_f_ in ti.ndrange(_B, n_faces):
        i_f = map_lidar_faces[i_f_]
        aabb_state.aabbs[i_b, i_f].min.fill(np.inf)
        aabb_state.aabbs[i_b, i_f].max.fill(-np.inf)

        is_free = verts_info.is_free[faces_info.verts_idx[i_f][0]]
        if is_free:
            for i in ti.static(range(3)):
                i_v = verts_info.verts_state_idx[faces_info.verts_idx[i_f][i]]
                pos_v = free_verts_state.pos[i_v, i_b]
                aabb_state.aabbs[i_b, i_f].min = ti.min(aabb_state.aabbs[i_b, i_f].min, pos_v)
                aabb_state.aabbs[i_b, i_f].max = ti.max(aabb_state.aabbs[i_b, i_f].max, pos_v)

        elif i_b == 0:  #
            for i in ti.static(range(3)):
                i_v = verts_info.verts_state_idx[faces_info.verts_idx[i_f][i]]
                pos_v = fixed_verts_state.pos[i_v]
                aabb_state.aabbs[i_b, i_f].min = ti.min(aabb_state.aabbs[i_b, i_f].min, pos_v)
                aabb_state.aabbs[i_b, i_f].max = ti.max(aabb_state.aabbs[i_b, i_f].max, pos_v)


@ti.kernel
def kernel_cast_rays_bvh(
    map_lidar_faces: MapLidarFaces,
    fixed_verts_state: array_class.VertsState,
    free_verts_state: array_class.VertsState,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
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
    n_triangles = map_lidar_faces.shape[0]
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

                    i_f = map_lidar_faces[original_tri_idx]
                    is_free = verts_info.is_free[faces_info.verts_idx[i_f][0]]

                    v0 = ti.Vector.zero(gs.ti_float, 3)
                    v1 = ti.Vector.zero(gs.ti_float, 3)
                    v2 = ti.Vector.zero(gs.ti_float, 3)

                    if is_free:
                        v0 = free_verts_state.pos[verts_info.verts_state_idx[faces_info.verts_idx[i_f][0]], env_id]
                        v1 = free_verts_state.pos[verts_info.verts_state_idx[faces_info.verts_idx[i_f][1]], env_id]
                        v2 = free_verts_state.pos[verts_info.verts_state_idx[faces_info.verts_idx[i_f][2]], env_id]

                    else:
                        v0 = fixed_verts_state.pos[verts_info.verts_state_idx[faces_info.verts_idx[i_f][0]]]
                        v1 = fixed_verts_state.pos[verts_info.verts_state_idx[faces_info.verts_idx[i_f][1]]]
                        v2 = fixed_verts_state.pos[verts_info.verts_state_idx[faces_info.verts_idx[i_f][2]]]

                    # Perform the expensive ray-triangle intersection test
                    hit_result = ray_triangle_intersection(lidar_position, ray_direction_world, v0, v1, v2)

                    if hit_result.w > 0.0 and hit_result.x < min_t and hit_result.x >= 0.0:
                        min_t = hit_result.x
                        hit_face = i_f
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
        only_cast_fixed: bool = True,
    ):

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

        self.only_cast_fixed = only_cast_fixed

        # Generate ray pattern
        self.ray_vectors = self._create_ray_pattern()

        # build bvh
        self.solver = self._sim.rigid_solver
        self.is_built = False

    def filter_lidar_faces(self):
        n_lidar_faces = self.solver.faces_info.geom_idx.shape[0]
        np_map_lidar_faces = np.arange(n_lidar_faces)
        if self.only_cast_fixed:
            # count the number of faces in a fixed geoms
            geom_is_fixed = np.logical_not(self.solver.geoms_info.is_free.to_numpy())
            faces_geom = self.solver.faces_info.geom_idx.to_numpy()
            n_lidar_faces = np.sum(geom_is_fixed[faces_geom])
            np_map_lidar_faces = np.where(geom_is_fixed[faces_geom])[0]
        # from IPython import embed; embed()
        return n_lidar_faces, np_map_lidar_faces

    def build(self):
        n_lidar_faces, np_map_lidar_faces = self.filter_lidar_faces()

        self.n_lidar_faces = n_lidar_faces
        self.map_lidar_faces = ti.field(ti.i32, (n_lidar_faces))
        self.map_lidar_faces.from_numpy(np_map_lidar_faces)

        self.aabbs = AABB(n_batches=self.solver.free_verts_state.pos.shape[1], n_aabbs=self.n_lidar_faces)

        rigid_solver_decomp.kernel_update_all_verts(
            geoms_state=self.solver.geoms_state,
            verts_info=self.solver.verts_info,
            free_verts_state=self.solver.free_verts_state,
            fixed_verts_state=self.solver.fixed_verts_state,
        )

        kernel_update_aabbs(
            map_lidar_faces=self.map_lidar_faces,
            free_verts_state=self.solver.free_verts_state,
            fixed_verts_state=self.solver.fixed_verts_state,
            verts_info=self.solver.verts_info,
            faces_info=self.solver.faces_info,
            aabb_state=self.aabbs,
        )

        self.bvh = LBVH(self.aabbs)
        self.bvh.build()

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

    def read(self, envs_idx: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read the LiDAR sensor data.

        Args:
            envs_idx: Optional list of environment indices to read from.

        Returns:
            hit_points: Hit points array [n_env, n_scan_lines, n_points, 3]
            hit_distances: Hit distances array [n_env, n_scan_lines, n_points]
        """
        if not self.is_built:
            self.build()
            self.is_built = True

        if not self.only_cast_fixed:
            rigid_solver_decomp.kernel_update_all_verts(
                geoms_state=self.solver.geoms_state,
                verts_info=self.solver.verts_info,
                free_verts_state=self.solver.free_verts_state,
                fixed_verts_state=self.solver.fixed_verts_state,
            )

            kernel_update_aabbs(
                map_lidar_faces=self.map_lidar_faces,
                free_verts_state=self.solver.free_verts_state,
                fixed_verts_state=self.solver.fixed_verts_state,
                verts_info=self.solver.verts_info,
                faces_info=self.solver.faces_info,
                aabb_state=self.aabbs,
            )

        n_envs = self.solver.free_verts_state.pos.shape[1]
        n_scan_lines, n_points = self.ray_vectors.shape[:2]

        # Prepare output arrays
        hit_points = torch.zeros(size=(n_envs, 1, n_scan_lines, n_points, 3), dtype=gs.tc_float)
        hit_distances = torch.zeros(size=(n_envs, 1, n_scan_lines, n_points), dtype=gs.tc_float)

        # Convert to numpy arrays and reshape for kernel
        lidar_positions = self.solver.get_links_pos(links_idx=self.link_idx).squeeze(axis=1).reshape(n_envs, 1, 3)
        lidar_quaternions = (
            self.solver.get_links_quat(links_idx=self.link_idx).squeeze(axis=1).reshape(n_envs, 1, 4)
        )  # wxyz format

        kernel_cast_rays_bvh(
            map_lidar_faces=self.map_lidar_faces,
            fixed_verts_state=self.solver.fixed_verts_state,
            free_verts_state=self.solver.free_verts_state,
            verts_info=self.solver.verts_info,
            faces_info=self.solver.faces_info,
            bvh_nodes=self.bvh.nodes,
            bvh_morton_codes=self.bvh.morton_codes,
            lidar_positions=lidar_positions,
            lidar_quaternions=lidar_quaternions,
            ray_vectors=self.ray_vectors,
            far_plane=self.config["max_range"],
            hit_points=hit_points,
            hit_distances=hit_distances,
            world_frame=0 if self._use_local_frame else 1,
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
        print("TODO: get_mesh_info not implemented")
        return
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
        print("TODO: get_scene_mesh not implemented")
        return
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
        print("TODO: save_scene_mesh not implemented")
        return
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
        print("TODO: print_mesh_summary not implemented")
        return
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
