import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple, Type

import gstaichi as ti
import numpy as np
import torch
from pydantic import Field

import genesis as gs
import genesis.engine.solvers.rigid.rigid_solver_decomp as rigid_solver_decomp
import genesis.utils.array_class as array_class
from genesis.engine.bvh import AABB, LBVH
from genesis.sensors.sensor_manager import register_sensor
from genesis.utils.geom import (
    ti_normalize,
    ti_transform_by_quat,
    ti_transform_by_trans_quat,
    trans_to_T,
    transform_by_quat,
    transform_by_trans_quat,
)
from genesis.utils.misc import concat_with_tensor, make_tensor_field, ti_to_torch
from genesis.vis.rasterizer_context import RasterizerContext

from ..base_sensor import (
    RigidSensorMetadataMixin,
    RigidSensorMixin,
    RigidSensorOptionsMixin,
    Sensor,
    SensorOptions,
    SharedSensorMetadata,
)
from .patterns import RaycastPattern

if TYPE_CHECKING:
    from genesis.ext.pyrender.mesh import Mesh
    from genesis.utils.ring_buffer import TensorRingBuffer


DEBUG_COLORS = (
    (1.0, 0.2, 0.2, 1.0),
    (0.2, 1.0, 0.2, 1.0),
    (0.2, 0.6, 1.0, 1.0),
    (1.0, 1.0, 0.2, 1.0),
)
# A constant stack size should be sufficient for BVH traversal.
# https://madmann91.github.io/2021/01/06/bvhs-part-2.html
# https://forums.developer.nvidia.com/t/thinking-parallel-part-ii-tree-traversal-on-the-gpu/148342
STACK_SIZE = ti.static(64)


@ti.func
def ray_triangle_intersection(ray_start, ray_dir, v0, v1, v2):
    """
    Möller-Trumbore ray-triangle intersection.

    Returns: vec4(t, u, v, hit) where hit=1.0 if intersection found, 0.0 otherwise
    """
    result = ti.Vector.zero(gs.ti_float, 4)

    edge1 = v1 - v0
    edge2 = v2 - v0

    # Begin calculating determinant - also used to calculate u parameter
    h = ray_dir.cross(edge2)
    a = edge1.dot(h)

    # Check all conditions in sequence without early returns
    valid = True

    t = gs.ti_float(0.0)
    u = gs.ti_float(0.0)
    v = gs.ti_float(0.0)
    f = gs.ti_float(0.0)
    s = ti.Vector.zero(gs.ti_float, 3)
    q = ti.Vector.zero(gs.ti_float, 3)

    # If determinant is near zero, ray lies in plane of triangle
    if ti.abs(a) < gs.EPS:
        valid = False

    if valid:
        f = 1.0 / a
        s = ray_start - v0
        u = f * s.dot(h)

        if u < 0.0 or u > 1.0:
            valid = False

    if valid:
        q = s.cross(edge1)
        v = f * ray_dir.dot(q)

        if v < 0.0 or u + v > 1.0:
            valid = False

    if valid:
        # At this stage we can compute t to find out where the intersection point is on the line
        t = f * edge2.dot(q)

        # Ray intersection
        if t <= gs.EPS:
            valid = False

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
    sign = ti.select(ray_dir >= 0.0, 1.0, -1.0)
    ray_dir = sign * ti.max(ti.abs(ray_dir), gs.EPS)
    inv_dir = 1.0 / ray_dir

    t1 = (aabb_min - ray_start) * inv_dir
    t2 = (aabb_max - ray_start) * inv_dir

    tmin = ti.min(t1, t2)
    tmax = ti.max(t1, t2)

    t_near = ti.max(tmin.x, tmin.y, tmin.z, 0.0)
    t_far = ti.min(tmax.x, tmax.y, tmax.z)

    # Check if ray intersects AABB
    if t_near <= t_far:
        result = t_near

    return result


@ti.kernel
def kernel_update_aabbs(
    map_faces: ti.template(),
    free_verts_state: array_class.VertsState,
    fixed_verts_state: array_class.VertsState,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    aabb_state: array_class.AABBState,
):
    for i_b, i_f_ in ti.ndrange(free_verts_state.pos.shape[1], map_faces.shape[0]):
        i_f = map_faces[i_f_]
        aabb_state.aabbs[i_b, i_f].min.fill(ti.math.inf)
        aabb_state.aabbs[i_b, i_f].max.fill(-ti.math.inf)

        for i in ti.static(range(3)):
            i_v = verts_info.verts_state_idx[faces_info.verts_idx[i_f][i]]
            if verts_info.is_fixed[faces_info.verts_idx[i_f][i]]:
                pos_v = fixed_verts_state.pos[i_v]
                aabb_state.aabbs[i_b, i_f].min = ti.min(aabb_state.aabbs[i_b, i_f].min, pos_v)
                aabb_state.aabbs[i_b, i_f].max = ti.max(aabb_state.aabbs[i_b, i_f].max, pos_v)
            else:
                pos_v = free_verts_state.pos[i_v, i_b]
                aabb_state.aabbs[i_b, i_f].min = ti.min(aabb_state.aabbs[i_b, i_f].min, pos_v)
                aabb_state.aabbs[i_b, i_f].max = ti.max(aabb_state.aabbs[i_b, i_f].max, pos_v)


@ti.kernel
def kernel_cast_rays(
    map_faces: ti.template(),
    fixed_verts_state: array_class.VertsState,
    free_verts_state: array_class.VertsState,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    bvh_nodes: ti.template(),
    bvh_morton_codes: ti.template(),  # maps sorted leaves to original triangle indices
    links_pos: ti.types.ndarray(ndim=3),  # [n_env, n_sensors, 3]
    links_quat: ti.types.ndarray(ndim=3),  # [n_env, n_sensors, 4]
    ray_starts: ti.types.ndarray(ndim=2),  # [n_points, 3]
    ray_directions: ti.types.ndarray(ndim=2),  # [n_points, 3]
    max_ranges: ti.types.ndarray(ndim=1),  # [n_sensors]
    no_hit_values: ti.types.ndarray(ndim=1),  # [n_sensors]
    is_world_frame: ti.types.ndarray(ndim=1),  # [n_sensors]
    points_to_sensor_idx: ti.types.ndarray(ndim=1),  # [n_points]
    sensor_cache_offsets: ti.types.ndarray(ndim=1),  # [n_sensors] - cache start index for each sensor
    sensor_point_offsets: ti.types.ndarray(ndim=1),  # [n_sensors] - point start index for each sensor
    sensor_point_counts: ti.types.ndarray(ndim=1),  # [n_sensors] - number of points for each sensor
    output_hits: ti.types.ndarray(ndim=2),  # [n_env, total_cache_size]
):
    """
    Taichi kernel for ray casting, accelerated by a Bounding Volume Hierarchy (BVH).

    The result `output_hits` will be a 2D array of shape (n_env, total_cache_size) where in the second dimension,
    each sensor's data is stored as [sensor_points (n_points * 3), sensor_ranges (n_points)].
    """

    n_triangles = map_faces.shape[0]
    n_points = ray_starts.shape[0]
    # batch, point
    for i_b, i_p in ti.ndrange(output_hits.shape[0], n_points):
        i_s = points_to_sensor_idx[i_p]

        # --- 1. Setup Ray ---
        link_pos = ti.math.vec3(links_pos[i_b, i_s, 0], links_pos[i_b, i_s, 1], links_pos[i_b, i_s, 2])
        link_quat = ti.math.vec4(
            links_quat[i_b, i_s, 0], links_quat[i_b, i_s, 1], links_quat[i_b, i_s, 2], links_quat[i_b, i_s, 3]
        )

        ray_start_local = ti.math.vec3(ray_starts[i_p, 0], ray_starts[i_p, 1], ray_starts[i_p, 2])
        ray_start_world = ti_transform_by_trans_quat(ray_start_local, link_pos, link_quat)

        ray_dir_local = ti.math.vec3(ray_directions[i_p, 0], ray_directions[i_p, 1], ray_directions[i_p, 2])
        ray_direction_world = ti_normalize(ti_transform_by_quat(ray_dir_local, link_quat))

        # --- 2. BVH Traversal ---
        max_range = max_ranges[i_s]
        hit_face = -1

        # Stack for non-recursive traversal
        node_stack = ti.Vector.zero(ti.i32, STACK_SIZE)
        node_stack[0] = 0  # Start traversal at the root node (index 0)
        stack_idx = 1

        while stack_idx > 0:
            stack_idx -= 1
            node_idx = node_stack[stack_idx]

            node = bvh_nodes[i_b, node_idx]

            # Check if ray hits the node's bounding box
            aabb_t = ray_aabb_intersection(ray_start_world, ray_direction_world, node.bound.min, node.bound.max)

            if aabb_t >= 0.0 and aabb_t < max_range:
                if node.left == -1:  # is leaf node
                    # A leaf node corresponds to one of the sorted triangles. Find the original triangle index.
                    sorted_leaf_idx = node_idx - (n_triangles - 1)
                    original_tri_idx = ti.cast(bvh_morton_codes[0, sorted_leaf_idx][1], ti.i32)

                    i_f = map_faces[original_tri_idx]
                    is_fixed = verts_info.is_fixed[faces_info.verts_idx[i_f][0]]

                    v0 = ti.Vector.zero(gs.ti_float, 3)
                    v1 = ti.Vector.zero(gs.ti_float, 3)
                    v2 = ti.Vector.zero(gs.ti_float, 3)

                    if is_fixed:
                        v0 = fixed_verts_state.pos[verts_info.verts_state_idx[faces_info.verts_idx[i_f][0]]]
                        v1 = fixed_verts_state.pos[verts_info.verts_state_idx[faces_info.verts_idx[i_f][1]]]
                        v2 = fixed_verts_state.pos[verts_info.verts_state_idx[faces_info.verts_idx[i_f][2]]]

                    else:
                        v0 = free_verts_state.pos[verts_info.verts_state_idx[faces_info.verts_idx[i_f][0]], i_b]
                        v1 = free_verts_state.pos[verts_info.verts_state_idx[faces_info.verts_idx[i_f][1]], i_b]
                        v2 = free_verts_state.pos[verts_info.verts_state_idx[faces_info.verts_idx[i_f][2]], i_b]

                    # Perform the expensive ray-triangle intersection test
                    hit_result = ray_triangle_intersection(ray_start_world, ray_direction_world, v0, v1, v2)

                    if hit_result.w > 0.0 and hit_result.x < max_range and hit_result.x >= 0.0:
                        max_range = hit_result.x
                        hit_face = i_f
                        # hit_u, hit_v could be stored here if needed

                else:  # It's an INTERNAL node
                    # Push children onto the stack for further traversal
                    # Make sure stack doesn't overflow
                    if stack_idx < ti.static(STACK_SIZE - 2):
                        node_stack[stack_idx] = node.left
                        node_stack[stack_idx + 1] = node.right
                        stack_idx += 2

        # --- 3. Process Hit Result ---
        # The format of output_hits is: [sensor1 points][sensor1 ranges][sensor2 points][sensor2 ranges]...
        i_p_sensor = i_p - sensor_point_offsets[i_s]
        i_p_offset = sensor_cache_offsets[i_s]  # cumulative cache offset for this sensor
        n_points_in_sensor = sensor_point_counts[i_s]  # number of points in this sensor

        i_p_dist = i_p_offset + n_points_in_sensor * 3 + i_p_sensor  # index for distance output

        if hit_face >= 0:
            dist = max_range
            # Store distance at: cache_offset + (num_points_in_sensor * 3) + point_idx_in_sensor
            output_hits[i_b, i_p_dist] = dist

            if is_world_frame[i_s]:
                hit_point = ray_start_world + dist * ray_direction_world

                # Store points at: cache_offset + point_idx_in_sensor * 3
                output_hits[i_b, i_p_offset + i_p_sensor * 3 + 0] = hit_point.x
                output_hits[i_b, i_p_offset + i_p_sensor * 3 + 1] = hit_point.y
                output_hits[i_b, i_p_offset + i_p_sensor * 3 + 2] = hit_point.z
            else:
                # Local frame output along provided local ray direction
                hit_point = dist * ti_normalize(
                    ti.math.vec3(ray_directions[i_p, 0], ray_directions[i_p, 1], ray_directions[i_p, 2])
                )
                output_hits[i_b, i_p_offset + i_p_sensor * 3 + 0] = hit_point.x
                output_hits[i_b, i_p_offset + i_p_sensor * 3 + 1] = hit_point.y
                output_hits[i_b, i_p_offset + i_p_sensor * 3 + 2] = hit_point.z

        else:
            # No hit
            output_hits[i_b, i_p_offset + i_p_sensor * 3 + 0] = 0.0
            output_hits[i_b, i_p_offset + i_p_sensor * 3 + 1] = 0.0
            output_hits[i_b, i_p_offset + i_p_sensor * 3 + 2] = 0.0
            output_hits[i_b, i_p_dist] = no_hit_values[i_s]


class RaycasterOptions(RigidSensorOptionsMixin, SensorOptions):
    """
    Raycaster sensor that performs ray casting to get distance measurements and point clouds.

    Parameters
    ----------
    pattern: RaycastPatternOptions
        The raycasting pattern for the sensor.
    min_range : float, optional
        The minimum sensing range in meters. Defaults to 0.0.
    max_range : float, optional
        The maximum sensing range in meters. Defaults to 20.0.
    no_hit_value : float, optional
        The value to return for no hit. Defaults to max_range if not specified.
    return_world_frame : bool, optional
        Whether to return points in the world frame. Defaults to False (local frame).
    debug_sphere_radius: float, optional
        The radius of each debug sphere drawn in the scene. Defaults to 0.02.
    debug_ray_start_color: float, optional
        The color of each debug ray start sphere drawn in the scene. Defaults to (0.5, 0.5, 1.0, 1.0).
    debug_ray_hit_color: float, optional
        The color of each debug ray hit point sphere drawn in the scene. Defaults to (1.0, 0.5, 0.5, 1.0).
    """

    pattern: RaycastPattern
    min_range: float = 0.0
    max_range: float = 20.0
    no_hit_value: float = Field(default_factory=lambda data: data["max_range"])
    return_world_frame: bool = False

    debug_sphere_radius: float = 0.02
    debug_ray_start_color: tuple[float, float, float, float] = (0.5, 0.5, 1.0, 1.0)
    debug_ray_hit_color: tuple[float, float, float, float] = (1.0, 0.5, 0.5, 1.0)

    def model_post_init(self, _):
        if self.min_range < 0.0:
            gs.raise_exception(f"[{type(self).__name__}] min_range should be non-negative. Got: {self.min_range}.")
        if self.max_range <= self.min_range:
            gs.raise_exception(
                f"[{type(self).__name__}] max_range {self.max_range} should be greater than min_range {self.min_range}."
            )


@dataclass
class RaycasterSharedMetadata(RigidSensorMetadataMixin, SharedSensorMetadata):
    bvh: LBVH | None = None
    aabb: AABB | None = None
    needs_aabb_update: bool = False
    map_faces: Any | None = None
    n_faces: int = 0

    sensors_ray_start_idx: list[int] = field(default_factory=list)
    total_n_rays: int = 0

    min_ranges: torch.Tensor = make_tensor_field((0,))
    max_ranges: torch.Tensor = make_tensor_field((0,))
    no_hit_values: torch.Tensor = make_tensor_field((0,))
    return_world_frame: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_bool)

    patterns: list[RaycastPattern] = field(default_factory=list)
    ray_dirs: torch.Tensor = make_tensor_field((0, 3))
    ray_starts: torch.Tensor = make_tensor_field((0, 3))
    ray_starts_world: torch.Tensor = make_tensor_field((0, 3))
    ray_dirs_world: torch.Tensor = make_tensor_field((0, 3))

    points_to_sensor_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    sensor_cache_offsets: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    sensor_point_offsets: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    sensor_point_counts: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    output_hits: torch.Tensor = make_tensor_field((0, 0))  # FIXME: remove once we have contiguous cache slices


class RaycasterData(NamedTuple):
    points: torch.Tensor
    distances: torch.Tensor


@register_sensor(RaycasterOptions, RaycasterSharedMetadata, RaycasterData)
@ti.data_oriented
class RaycasterSensor(RigidSensorMixin, Sensor):

    def __init__(
        self,
        options: RaycasterOptions,
        shared_metadata: RaycasterSharedMetadata,
        data_cls: Type[RaycasterData],
        manager: "gs.SensorManager",
    ):
        super().__init__(options, shared_metadata, data_cls, manager)
        self.debug_objects: list["Mesh | None"] = []
        self.ray_starts: torch.Tensor = torch.empty((0, 3), device=gs.device, dtype=gs.tc_float)

    @classmethod
    def _build_bvh(cls, shared_metadata: RaycasterSharedMetadata):
        n_faces = shared_metadata.solver.faces_info.geom_idx.shape[0]
        torch_map_faces = torch.arange(n_faces, dtype=torch.int32, device=gs.device)

        shared_metadata.map_faces = ti.field(ti.i32, (n_faces))
        shared_metadata.map_faces.from_torch(torch_map_faces)
        shared_metadata.n_faces = n_faces

        shared_metadata.aabb = AABB(n_batches=shared_metadata.solver.free_verts_state.pos.shape[1], n_aabbs=n_faces)

        rigid_solver_decomp.kernel_update_all_verts(
            geoms_state=shared_metadata.solver.geoms_state,
            verts_info=shared_metadata.solver.verts_info,
            free_verts_state=shared_metadata.solver.free_verts_state,
            fixed_verts_state=shared_metadata.solver.fixed_verts_state,
        )

        kernel_update_aabbs(
            map_faces=shared_metadata.map_faces,
            free_verts_state=shared_metadata.solver.free_verts_state,
            fixed_verts_state=shared_metadata.solver.fixed_verts_state,
            verts_info=shared_metadata.solver.verts_info,
            faces_info=shared_metadata.solver.faces_info,
            aabb_state=shared_metadata.aabb,
        )
        shared_metadata.bvh = LBVH(shared_metadata.aabb)
        shared_metadata.bvh.build()

    def build(self):
        super().build()  # set shared metadata from RigidSensorMixin

        # first lidar sensor initialization: build aabb and bvh
        if self._shared_metadata.bvh is None:
            geom_is_fixed = ti_to_torch(self._shared_metadata.solver.geoms_info.is_fixed)
            self._shared_metadata.needs_aabb_update = bool((~geom_is_fixed).any().item())

            self._shared_metadata.output_hits = torch.empty(
                (self._manager._sim._B, 0), device=gs.device, dtype=gs.tc_float
            )
            self._shared_metadata.sensor_cache_offsets = concat_with_tensor(
                self._shared_metadata.sensor_cache_offsets, 0
            )
            self._build_bvh(self._shared_metadata)

        self._shared_metadata.patterns.append(self._options.pattern)
        pos_offset = self._shared_metadata.offsets_pos[0, -1, :]  # all envs have same offset on build
        quat_offset = self._shared_metadata.offsets_quat[0, -1, :]

        ray_starts = self._options.pattern.ray_starts.reshape(-1, 3)
        self.ray_starts = transform_by_trans_quat(ray_starts, pos_offset, quat_offset)
        self._shared_metadata.ray_starts = torch.cat([self._shared_metadata.ray_starts, self.ray_starts])

        ray_dirs = self._options.pattern.ray_dirs.reshape(-1, 3)
        ray_dirs = transform_by_quat(ray_dirs, quat_offset)
        self._shared_metadata.ray_dirs = torch.cat([self._shared_metadata.ray_dirs, ray_dirs])

        num_rays = math.prod(self._options.pattern.return_shape)
        self._shared_metadata.sensors_ray_start_idx.append(self._shared_metadata.total_n_rays)

        # These fields are used to properly index into the big cache tensor in kernel_cast_rays
        self._shared_metadata.output_hits = concat_with_tensor(
            self._shared_metadata.output_hits,
            torch.empty((self._manager._sim._B, self._cache_size), device=gs.device, dtype=gs.tc_float),
            dim=-1,
        )
        self._shared_metadata.sensor_cache_offsets = concat_with_tensor(
            self._shared_metadata.sensor_cache_offsets, self._cache_size
        )
        self._shared_metadata.sensor_point_offsets = concat_with_tensor(
            self._shared_metadata.sensor_point_offsets, self._shared_metadata.total_n_rays
        )
        self._shared_metadata.sensor_point_counts = concat_with_tensor(
            self._shared_metadata.sensor_point_counts, num_rays
        )
        self._shared_metadata.total_n_rays += num_rays

        self._shared_metadata.points_to_sensor_idx = concat_with_tensor(
            self._shared_metadata.points_to_sensor_idx,
            [self._idx] * num_rays,
            flatten=True,
        )
        self._shared_metadata.return_world_frame = concat_with_tensor(
            self._shared_metadata.return_world_frame, self._options.return_world_frame
        )
        self._shared_metadata.min_ranges = concat_with_tensor(self._shared_metadata.min_ranges, self._options.min_range)
        self._shared_metadata.max_ranges = concat_with_tensor(self._shared_metadata.max_ranges, self._options.max_range)
        no_hit_value = self._options.no_hit_value if self._options.no_hit_value is not None else self._options.max_range
        self._shared_metadata.no_hit_values = concat_with_tensor(self._shared_metadata.no_hit_values, no_hit_value)

    @classmethod
    def reset(cls, shared_metadata: RaycasterSharedMetadata, envs_idx):
        super().reset(shared_metadata, envs_idx)
        cls._build_bvh(shared_metadata)

    def _get_return_format(self) -> tuple[tuple[int, ...], ...]:
        shape = self._options.pattern.return_shape
        return (*shape, 3), shape

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def _update_shared_ground_truth_cache(
        cls, shared_metadata: RaycasterSharedMetadata, shared_ground_truth_cache: torch.Tensor
    ):
        if not shared_metadata.needs_aabb_update:
            rigid_solver_decomp.kernel_update_all_verts(
                geoms_state=shared_metadata.solver.geoms_state,
                verts_info=shared_metadata.solver.verts_info,
                free_verts_state=shared_metadata.solver.free_verts_state,
                fixed_verts_state=shared_metadata.solver.fixed_verts_state,
            )

            kernel_update_aabbs(
                map_faces=shared_metadata.map_faces,
                free_verts_state=shared_metadata.solver.free_verts_state,
                fixed_verts_state=shared_metadata.solver.fixed_verts_state,
                verts_info=shared_metadata.solver.verts_info,
                faces_info=shared_metadata.solver.faces_info,
                aabb_state=shared_metadata.aabb,
            )

        links_pos = shared_metadata.solver.get_links_pos(links_idx=shared_metadata.links_idx)
        links_quat = shared_metadata.solver.get_links_quat(links_idx=shared_metadata.links_idx)
        if shared_metadata.solver.n_envs == 0:
            links_pos = links_pos.unsqueeze(0)
            links_quat = links_quat.unsqueeze(0)

        kernel_cast_rays(
            map_faces=shared_metadata.map_faces,
            fixed_verts_state=shared_metadata.solver.fixed_verts_state,
            free_verts_state=shared_metadata.solver.free_verts_state,
            verts_info=shared_metadata.solver.verts_info,
            faces_info=shared_metadata.solver.faces_info,
            bvh_nodes=shared_metadata.bvh.nodes,
            bvh_morton_codes=shared_metadata.bvh.morton_codes,
            links_pos=links_pos.contiguous(),
            links_quat=links_quat.contiguous(),
            ray_starts=shared_metadata.ray_starts,
            ray_directions=shared_metadata.ray_dirs,
            max_ranges=shared_metadata.max_ranges,
            no_hit_values=shared_metadata.no_hit_values,
            is_world_frame=shared_metadata.return_world_frame,
            points_to_sensor_idx=shared_metadata.points_to_sensor_idx,
            sensor_cache_offsets=shared_metadata.sensor_cache_offsets,
            sensor_point_offsets=shared_metadata.sensor_point_offsets,
            sensor_point_counts=shared_metadata.sensor_point_counts,
            output_hits=(
                shared_ground_truth_cache if shared_ground_truth_cache.is_contiguous() else shared_metadata.output_hits
            ),
        )
        if not shared_ground_truth_cache.is_contiguous():
            shared_ground_truth_cache[:] = shared_metadata.output_hits

    @classmethod
    def _update_shared_cache(
        cls,
        shared_metadata: RaycasterSharedMetadata,
        shared_ground_truth_cache: torch.Tensor,
        shared_cache: torch.Tensor,
        buffered_data: "TensorRingBuffer",
    ):
        buffered_data.append(shared_ground_truth_cache)
        cls._apply_delay_to_shared_cache(shared_metadata, shared_cache, buffered_data)

    def _draw_debug(self, context: "RasterizerContext", buffer_updates: dict[str, np.ndarray]):
        """
        Draw hit points as spheres in the scene.

        Only draws for first rendered environment.
        """
        env_idx = context.rendered_envs_idx[0]

        points = self.read(envs_idx=env_idx if self._manager._sim.n_envs > 0 else None).points.reshape(-1, 3)

        pos = self._link.get_pos(envs_idx=env_idx)
        quat = self._link.get_quat(envs_idx=env_idx)

        ray_starts = transform_by_trans_quat(self.ray_starts, pos, quat)

        if not self._options.return_world_frame:
            points = transform_by_trans_quat(points + self.ray_starts, pos, quat)

        if not self.debug_objects:
            for ray_start in ray_starts:
                self.debug_objects.append(
                    context.draw_debug_sphere(
                        ray_start,
                        radius=self._options.debug_sphere_radius,
                        color=self._options.debug_ray_start_color,
                    )
                )
            for point in points:
                self.debug_objects.append(
                    context.draw_debug_sphere(
                        point,
                        radius=self._options.debug_sphere_radius,
                        color=self._options.debug_ray_hit_color,
                    )
                )
        else:
            buffer_updates.update(
                context.get_buffer_debug_objects(self.debug_objects, trans_to_T(torch.cat([ray_starts, points], dim=0)))
            )
