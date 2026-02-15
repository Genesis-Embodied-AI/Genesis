import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple, Type

import quadrants as ti
import numpy as np
import torch

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.bvh import AABB, LBVH
from genesis.options.sensors import (
    Raycaster as RaycasterOptions,
)
from genesis.options.sensors import (
    RaycastPattern,
)
from genesis.utils.geom import (
    ti_normalize,
    ti_transform_by_quat,
    ti_transform_by_trans_quat,
    transform_by_quat,
    transform_by_trans_quat,
)
from genesis.utils.misc import concat_with_tensor, make_tensor_field
from genesis.utils.raycast_ti import bvh_ray_cast, kernel_update_verts_and_aabbs
from genesis.vis.rasterizer_context import RasterizerContext

from .base_sensor import (
    RigidSensorMetadataMixin,
    RigidSensorMixin,
    Sensor,
    SharedSensorMetadata,
)
from .sensor_manager import register_sensor

if TYPE_CHECKING:
    from genesis.ext.pyrender.mesh import Mesh
    from genesis.utils.ring_buffer import TensorRingBuffer


@ti.kernel
def kernel_cast_rays(
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
        ray_direction_world = ti_normalize(ti_transform_by_quat(ray_dir_local, link_quat), gs.EPS)

        # --- 2. BVH Traversal for ray intersection ---
        max_range = max_ranges[i_s]
        hit_face, hit_distance, hit_normal = bvh_ray_cast(
            ray_start=ray_start_world,
            ray_dir=ray_direction_world,
            max_range=max_range,
            i_b=i_b,
            bvh_nodes=bvh_nodes,
            bvh_morton_codes=bvh_morton_codes,
            faces_info=faces_info,
            verts_info=verts_info,
            fixed_verts_state=fixed_verts_state,
            free_verts_state=free_verts_state,
        )

        # --- 3. Process Hit Result ---
        # The format of output_hits is: [sensor1 points][sensor1 ranges][sensor2 points][sensor2 ranges]...
        i_p_sensor = i_p - sensor_point_offsets[i_s]
        i_p_offset = sensor_cache_offsets[i_s]  # cumulative cache offset for this sensor
        n_points_in_sensor = sensor_point_counts[i_s]  # number of points in this sensor

        i_p_dist = i_p_offset + n_points_in_sensor * 3 + i_p_sensor  # index for distance output

        if hit_face >= 0:
            dist = hit_distance
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
                    ti.math.vec3(ray_directions[i_p, 0], ray_directions[i_p, 1], ray_directions[i_p, 2]), gs.EPS
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


@dataclass
class RaycasterSharedMetadata(RigidSensorMetadataMixin, SharedSensorMetadata):
    bvh: LBVH | None = None
    aabb: AABB | None = None

    sensors_ray_start_idx: list[int] = field(default_factory=list)
    total_n_rays: int = 0

    min_ranges: torch.Tensor = make_tensor_field((0,))
    max_ranges: torch.Tensor = make_tensor_field((0,))
    no_hit_values: torch.Tensor = make_tensor_field((0,))
    return_world_frame: torch.Tensor = make_tensor_field((0,), dtype=gs.tc_bool)

    patterns: list[RaycastPattern] = field(default_factory=list)
    ray_dirs: torch.Tensor = make_tensor_field((0, 3))
    ray_starts: torch.Tensor = make_tensor_field((0, 3))
    ray_starts_world: torch.Tensor = make_tensor_field((0, 3))
    ray_dirs_world: torch.Tensor = make_tensor_field((0, 3))

    points_to_sensor_idx: torch.Tensor = make_tensor_field((0,), dtype=gs.tc_int)
    sensor_cache_offsets: torch.Tensor = make_tensor_field((0,), dtype=gs.tc_int)
    sensor_point_offsets: torch.Tensor = make_tensor_field((0,), dtype=gs.tc_int)
    sensor_point_counts: torch.Tensor = make_tensor_field((0,), dtype=gs.tc_int)


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
        self.debug_objects: list["Mesh"] = []
        self.ray_starts: torch.Tensor = torch.empty((0, 3), device=gs.device, dtype=gs.tc_float)

    @classmethod
    def _update_bvh(cls, shared_metadata: RaycasterSharedMetadata):
        """Rebuild BVH from current geometry in the scene."""
        kernel_update_verts_and_aabbs(
            geoms_info=shared_metadata.solver.geoms_info,
            geoms_state=shared_metadata.solver.geoms_state,
            verts_info=shared_metadata.solver.verts_info,
            faces_info=shared_metadata.solver.faces_info,
            free_verts_state=shared_metadata.solver.free_verts_state,
            fixed_verts_state=shared_metadata.solver.fixed_verts_state,
            static_rigid_sim_config=shared_metadata.solver._static_rigid_sim_config,
            aabb_state=shared_metadata.aabb,
        )

        shared_metadata.bvh.build()

    def build(self):
        super().build()  # set shared metadata from RigidSensorMixin

        # first lidar sensor initialization: build aabb and bvh
        if self._shared_metadata.bvh is None:
            self._shared_metadata.sensor_cache_offsets = concat_with_tensor(
                self._shared_metadata.sensor_cache_offsets, 0
            )
            n_faces = self._shared_metadata.solver.faces_info.geom_idx.shape[0]
            n_envs = self._shared_metadata.solver.free_verts_state.pos.shape[1]
            self._shared_metadata.aabb = AABB(n_batches=n_envs, n_aabbs=n_faces)

            # FIXME: Empirically, the values 0 and 64 seem to be sufficient and decrease memory usage.
            # Should these parameters be exposed to the user?
            self._shared_metadata.bvh = LBVH(
                self._shared_metadata.aabb, max_n_query_result_per_aabb=0, n_radix_sort_groups=64
            )
            self._update_bvh(self._shared_metadata)

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
        self._shared_metadata.sensor_cache_offsets = concat_with_tensor(
            self._shared_metadata.sensor_cache_offsets, self._cache_size * (self._idx + 1)
        )
        self._shared_metadata.sensor_point_offsets = concat_with_tensor(
            self._shared_metadata.sensor_point_offsets, self._shared_metadata.total_n_rays
        )
        self._shared_metadata.sensor_point_counts = concat_with_tensor(
            self._shared_metadata.sensor_point_counts, num_rays
        )
        self._shared_metadata.total_n_rays += num_rays

        self._shared_metadata.points_to_sensor_idx = concat_with_tensor(
            self._shared_metadata.points_to_sensor_idx, [self._idx] * num_rays, flatten=True
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
        cls._update_bvh(shared_metadata)

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
        cls._update_bvh(shared_metadata)

        links_pos = shared_metadata.solver.get_links_pos(links_idx=shared_metadata.links_idx)
        links_quat = shared_metadata.solver.get_links_quat(links_idx=shared_metadata.links_idx)
        if shared_metadata.solver.n_envs == 0:
            links_pos = links_pos[None]
            links_quat = links_quat[None]

        output_hits = shared_ground_truth_cache.contiguous()
        kernel_cast_rays(
            fixed_verts_state=shared_metadata.solver.fixed_verts_state,
            free_verts_state=shared_metadata.solver.free_verts_state,
            verts_info=shared_metadata.solver.verts_info,
            faces_info=shared_metadata.solver.faces_info,
            bvh_nodes=shared_metadata.bvh.nodes,
            bvh_morton_codes=shared_metadata.bvh.morton_codes,
            links_pos=links_pos,
            links_quat=links_quat,
            ray_starts=shared_metadata.ray_starts,
            ray_directions=shared_metadata.ray_dirs,
            max_ranges=shared_metadata.max_ranges,
            no_hit_values=shared_metadata.no_hit_values,
            is_world_frame=shared_metadata.return_world_frame,
            points_to_sensor_idx=shared_metadata.points_to_sensor_idx,
            sensor_cache_offsets=shared_metadata.sensor_cache_offsets,
            sensor_point_offsets=shared_metadata.sensor_point_offsets,
            sensor_point_counts=shared_metadata.sensor_point_counts,
            output_hits=output_hits,
        )
        if not shared_ground_truth_cache.is_contiguous():
            shared_ground_truth_cache.copy_(output_hits)

    @classmethod
    def _update_shared_cache(
        cls,
        shared_metadata: RaycasterSharedMetadata,
        shared_ground_truth_cache: torch.Tensor,
        shared_cache: torch.Tensor,
        buffered_data: "TensorRingBuffer",
    ):
        buffered_data.set(shared_ground_truth_cache)
        cls._apply_delay_to_shared_cache(shared_metadata, shared_cache, buffered_data)

    def _draw_debug(self, context: "RasterizerContext", buffer_updates: dict[str, np.ndarray]):
        """
        Draw hit points as spheres in the scene.

        Only draws for first rendered environment.
        """
        env_idx = context.rendered_envs_idx[0] if self._manager._sim.n_envs > 0 else None

        data = self.read(env_idx)
        points = data.points.reshape((-1, 3))

        pos = self._link.get_pos(env_idx).reshape((3,))
        quat = self._link.get_quat(env_idx).reshape((4,))

        ray_starts = transform_by_trans_quat(self.ray_starts, pos, quat)

        if not self._options.return_world_frame:
            points = transform_by_trans_quat(points + self.ray_starts, pos, quat)

        for debug_object in self.debug_objects:
            context.clear_debug_object(debug_object)
        self.debug_objects.clear()

        self.debug_objects += [
            context.draw_debug_spheres(
                ray_starts,
                radius=self._options.debug_sphere_radius,
                color=self._options.debug_ray_start_color,
            ),
            context.draw_debug_spheres(
                points,
                radius=self._options.debug_sphere_radius,
                color=self._options.debug_ray_hit_color,
            ),
        ]
