import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple, Type

import numpy as np
import quadrants as qd
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
    euler_to_quat,
    qd_normalize,
    qd_transform_by_quat,
    qd_transform_by_trans_quat,
    transform_by_quat,
    transform_by_trans_quat,
)
from genesis.utils.misc import concat_with_tensor, make_tensor_field, qd_to_numpy
from genesis.utils.raycast_qd import (
    bvh_ray_cast,
    bvh_ray_cast_visual,
    kernel_merge_ray_hits,
    kernel_update_visual_aabbs,
    kernel_update_verts_and_aabbs,
)
from genesis.engine.solvers.rigid.abd.forward_kinematics import kernel_forward_kinematics
from genesis.vis.rasterizer_context import RasterizerContext

from .base_sensor import (
    RigidSensorMetadataMixin,
    RigidSensorMixin,
    Sensor,
    SharedSensorMetadata,
)

if TYPE_CHECKING:
    from genesis.ext.pyrender.mesh import Mesh
    from genesis.utils.ring_buffer import TensorRingBuffer

    from .sensor_manager import SensorManager


@qd.kernel
def kernel_cast_rays(
    fixed_verts_state: array_class.VertsState,
    free_verts_state: array_class.VertsState,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    bvh_nodes: qd.template(),
    bvh_morton_codes: qd.template(),  # maps sorted leaves to original triangle indices
    links_pos: qd.types.ndarray(ndim=3),  # [n_env, n_sensors, 3]
    links_quat: qd.types.ndarray(ndim=3),  # [n_env, n_sensors, 4]
    ray_starts: qd.types.ndarray(ndim=2),  # [n_points, 3]
    ray_directions: qd.types.ndarray(ndim=2),  # [n_points, 3]
    max_ranges: qd.types.ndarray(ndim=1),  # [n_sensors]
    no_hit_values: qd.types.ndarray(ndim=1),  # [n_sensors]
    is_world_frame: qd.types.ndarray(ndim=1),  # [n_sensors]
    points_to_sensor_idx: qd.types.ndarray(ndim=1),  # [n_points]
    sensor_cache_offsets: qd.types.ndarray(ndim=1),  # [n_sensors] - cache start index for each sensor
    sensor_point_offsets: qd.types.ndarray(ndim=1),  # [n_sensors] - point start index for each sensor
    sensor_point_counts: qd.types.ndarray(ndim=1),  # [n_sensors] - number of points for each sensor
    output_hits: qd.types.ndarray(ndim=2),  # [total_cache_size, n_env]
    eps: float,
):
    """
    Quadrants kernel for ray casting, accelerated by a Bounding Volume Hierarchy (BVH).

    The result `output_hits` will be a 2D array of shape (total_cache_size, n_env) where in the first dimension,
    each sensor's data is stored as [sensor_points (n_points * 3), sensor_ranges (n_points)].
    """

    n_points = ray_starts.shape[0]
    for i_p, i_b in qd.ndrange(n_points, output_hits.shape[-1]):
        i_s = points_to_sensor_idx[i_p]

        # --- 1. Setup Ray ---
        link_pos = qd.math.vec3(links_pos[i_b, i_s, 0], links_pos[i_b, i_s, 1], links_pos[i_b, i_s, 2])
        link_quat = qd.math.vec4(
            links_quat[i_b, i_s, 0], links_quat[i_b, i_s, 1], links_quat[i_b, i_s, 2], links_quat[i_b, i_s, 3]
        )

        ray_start_local = qd.math.vec3(ray_starts[i_p, 0], ray_starts[i_p, 1], ray_starts[i_p, 2])
        ray_start_world = qd_transform_by_trans_quat(ray_start_local, link_pos, link_quat)

        ray_dir_local = qd.math.vec3(ray_directions[i_p, 0], ray_directions[i_p, 1], ray_directions[i_p, 2])
        ray_direction_world = qd_normalize(qd_transform_by_quat(ray_dir_local, link_quat), eps)

        # --- 2. BVH Traversal for ray intersection ---
        max_range = max_ranges[i_s]
        hit_face, hit_distance, _hit_normal = bvh_ray_cast(
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
            eps=eps,
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
            output_hits[i_p_dist, i_b] = dist

            if is_world_frame[i_s]:
                hit_point = ray_start_world + dist * ray_direction_world

                # Store points at: cache_offset + point_idx_in_sensor * 3
                output_hits[i_p_offset + i_p_sensor * 3 + 0, i_b] = hit_point.x
                output_hits[i_p_offset + i_p_sensor * 3 + 1, i_b] = hit_point.y
                output_hits[i_p_offset + i_p_sensor * 3 + 2, i_b] = hit_point.z
            else:
                # Local frame output along provided local ray direction
                hit_point = dist * qd_normalize(
                    qd.math.vec3(ray_directions[i_p, 0], ray_directions[i_p, 1], ray_directions[i_p, 2]), eps
                )
                output_hits[i_p_offset + i_p_sensor * 3 + 0, i_b] = hit_point.x
                output_hits[i_p_offset + i_p_sensor * 3 + 1, i_b] = hit_point.y
                output_hits[i_p_offset + i_p_sensor * 3 + 2, i_b] = hit_point.z
        else:
            # No hit
            output_hits[i_p_offset + i_p_sensor * 3 + 0, i_b] = 0.0
            output_hits[i_p_offset + i_p_sensor * 3 + 1, i_b] = 0.0
            output_hits[i_p_offset + i_p_sensor * 3 + 2, i_b] = 0.0
            output_hits[i_p_dist, i_b] = no_hit_values[i_s]


@qd.kernel
def kernel_cast_rays_visual(
    vverts_info: array_class.VVertsInfo,
    vverts_state: array_class.VVertsState,
    vfaces_info: array_class.VFacesInfo,
    vgeoms_state: array_class.VGeomsState,
    bvh_nodes: qd.template(),
    bvh_morton_codes: qd.template(),
    links_pos: qd.types.ndarray(ndim=3),
    links_quat: qd.types.ndarray(ndim=3),
    ray_starts: qd.types.ndarray(ndim=2),
    ray_directions: qd.types.ndarray(ndim=2),
    max_ranges: qd.types.ndarray(ndim=1),
    no_hit_values: qd.types.ndarray(ndim=1),
    is_world_frame: qd.types.ndarray(ndim=1),
    points_to_sensor_idx: qd.types.ndarray(ndim=1),
    sensor_cache_offsets: qd.types.ndarray(ndim=1),
    sensor_point_offsets: qd.types.ndarray(ndim=1),
    sensor_point_counts: qd.types.ndarray(ndim=1),
    output_hits: qd.types.ndarray(ndim=2),
    eps: float,
):
    """Visual-mesh variant of kernel_cast_rays. Uses vfaces/vverts instead of collision geometry."""
    n_points = ray_starts.shape[0]
    for i_p, i_b in qd.ndrange(n_points, output_hits.shape[-1]):
        i_s = points_to_sensor_idx[i_p]

        link_pos = qd.math.vec3(links_pos[i_b, i_s, 0], links_pos[i_b, i_s, 1], links_pos[i_b, i_s, 2])
        link_quat = qd.math.vec4(
            links_quat[i_b, i_s, 0], links_quat[i_b, i_s, 1], links_quat[i_b, i_s, 2], links_quat[i_b, i_s, 3]
        )

        ray_start_local = qd.math.vec3(ray_starts[i_p, 0], ray_starts[i_p, 1], ray_starts[i_p, 2])
        ray_start_world = qd_transform_by_trans_quat(ray_start_local, link_pos, link_quat)

        ray_dir_local = qd.math.vec3(ray_directions[i_p, 0], ray_directions[i_p, 1], ray_directions[i_p, 2])
        ray_direction_world = qd_normalize(qd_transform_by_quat(ray_dir_local, link_quat), eps)

        max_range = max_ranges[i_s]
        hit_face, hit_distance, _hit_normal = bvh_ray_cast_visual(
            ray_start=ray_start_world,
            ray_dir=ray_direction_world,
            max_range=max_range,
            i_b=i_b,
            bvh_nodes=bvh_nodes,
            bvh_morton_codes=bvh_morton_codes,
            vverts_info=vverts_info,
            vverts_state=vverts_state,
            vfaces_info=vfaces_info,
            vgeoms_state=vgeoms_state,
            eps=eps,
        )

        i_p_sensor = i_p - sensor_point_offsets[i_s]
        i_p_offset = sensor_cache_offsets[i_s]
        n_points_in_sensor = sensor_point_counts[i_s]
        i_p_dist = i_p_offset + n_points_in_sensor * 3 + i_p_sensor

        if hit_face >= 0:
            dist = hit_distance
            output_hits[i_p_dist, i_b] = dist

            if is_world_frame[i_s]:
                hit_point = ray_start_world + dist * ray_direction_world
                output_hits[i_p_offset + i_p_sensor * 3 + 0, i_b] = hit_point.x
                output_hits[i_p_offset + i_p_sensor * 3 + 1, i_b] = hit_point.y
                output_hits[i_p_offset + i_p_sensor * 3 + 2, i_b] = hit_point.z
            else:
                hit_point = dist * qd_normalize(
                    qd.math.vec3(ray_directions[i_p, 0], ray_directions[i_p, 1], ray_directions[i_p, 2]), eps
                )
                output_hits[i_p_offset + i_p_sensor * 3 + 0, i_b] = hit_point.x
                output_hits[i_p_offset + i_p_sensor * 3 + 1, i_b] = hit_point.y
                output_hits[i_p_offset + i_p_sensor * 3 + 2, i_b] = hit_point.z
        else:
            output_hits[i_p_offset + i_p_sensor * 3 + 0, i_b] = 0.0
            output_hits[i_p_offset + i_p_sensor * 3 + 1, i_b] = 0.0
            output_hits[i_p_offset + i_p_sensor * 3 + 2, i_b] = 0.0
            output_hits[i_p_dist, i_b] = no_hit_values[i_s]


@dataclass
class _SolverBVH:
    """BVH state for one solver's visual geometry."""

    solver: object  # KinematicSolver or RigidSolver
    bvh: LBVH
    aabb: AABB
    raycast_mask: np.ndarray  # (n_vfaces,) int8; 1 for vfaces opted into visual raycasting, 0 otherwise


@dataclass
class RaycasterSharedMetadata(RigidSensorMetadataMixin, SharedSensorMetadata):
    bvh: LBVH | None = None
    aabb: AABB | None = None
    use_visual_bvh: bool = False
    visual_raycast_mask: np.ndarray | None = None  # per-vface int8 mask for the primary visual BVH

    # Additional solvers whose visual geometry is also cast against (multi-solver support)
    extra_visual_bvhs: list[_SolverBVH] = field(default_factory=list)
    _secondary_cache: torch.Tensor | None = None

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


class RaycasterData(NamedTuple):
    points: torch.Tensor
    distances: torch.Tensor


class RaycasterSensor(RigidSensorMixin, Sensor[RaycasterOptions, RaycasterSharedMetadata, RaycasterData]):
    def __init__(self, options: RaycasterOptions, shared_metadata: RaycasterSharedMetadata, manager: "SensorManager"):
        super().__init__(options, shared_metadata, manager)
        self.debug_objects: list["Mesh"] = []
        self.ray_starts: torch.Tensor = torch.empty((0, 3), device=gs.device, dtype=gs.tc_float)

    @staticmethod
    def _compute_visual_raycast_mask(solver):
        """Build a per-vface mask (int8, shape (n_vfaces,)) selecting vfaces opted into visual raycasting.

        A vface is opted in iff its owning vgeom belongs to an entity whose material has
        use_visual_raycasting=True. Kinematic materials enforce True by default; Rigid materials default to False
        and let the user opt their visual mesh into the raycast BVH explicitly.
        """
        n_vfaces = solver.vfaces_info.vgeom_idx.shape[0]
        if n_vfaces == 0:
            return np.zeros(0, dtype=np.int8)
        vgeom_enabled = np.zeros(solver.n_vgeoms_, dtype=np.bool_)
        for entity in solver.entities:
            if not entity.material.use_visual_raycasting:
                continue
            for vgeom in entity.vgeoms:
                vgeom_enabled[vgeom.idx] = True
        vface_vgeom_idx = qd_to_numpy(solver.vfaces_info.vgeom_idx)
        return vgeom_enabled[vface_vgeom_idx].astype(np.int8)

    @classmethod
    def _update_visual_bvh_for_solver(cls, solver, aabb, bvh, raycast_mask):
        """Update a visual-mesh BVH for a single solver.

        Reads vverts_state.pos as the source of vvert positions. The buffer is seeded by FK at scene.build() and
        refreshed for each user-driven entity via set_vverts; entries set via set_vverts survive across calls until
        set_vverts(None) re-runs FK over the entity's vgeoms. raycast_mask gates which vfaces contribute to the BVH;
        masked-out vfaces keep an inverted AABB and are skipped by ray queries.
        """
        if not solver._is_forward_pos_updated:
            kernel_forward_kinematics(
                solver.scene._envs_idx,
                links_state=solver.links_state,
                links_info=solver.links_info,
                joints_state=solver.joints_state,
                joints_info=solver.joints_info,
                dofs_state=solver.dofs_state,
                dofs_info=solver.dofs_info,
                entities_info=solver.entities_info,
                rigid_global_info=solver._rigid_global_info,
                static_rigid_sim_config=solver._static_rigid_sim_config,
            )
            solver._is_forward_pos_updated = True
        solver.update_vgeoms()
        kernel_update_visual_aabbs(
            vverts_info=solver.vverts_info,
            vverts_state=solver.vverts_state,
            vfaces_info=solver.vfaces_info,
            vgeoms_state=solver.vgeoms_state,
            raycast_mask=raycast_mask,
            aabb_state=aabb,
        )
        bvh.build()

    @classmethod
    def _update_bvh(cls, shared_metadata: RaycasterSharedMetadata):
        """Rebuild BVH from current geometry in the scene."""
        solver = shared_metadata.solver

        if shared_metadata.use_visual_bvh:
            cls._update_visual_bvh_for_solver(
                solver, shared_metadata.aabb, shared_metadata.bvh, shared_metadata.visual_raycast_mask
            )
        else:
            kernel_update_verts_and_aabbs(
                geoms_info=solver.geoms_info,
                geoms_state=solver.geoms_state,
                verts_info=solver.verts_info,
                faces_info=solver.faces_info,
                free_verts_state=solver.free_verts_state,
                fixed_verts_state=solver.fixed_verts_state,
                static_rigid_sim_config=solver._static_rigid_sim_config,
                aabb_state=shared_metadata.aabb,
            )
            shared_metadata.bvh.build()

        # Update extra solver BVHs
        for entry in shared_metadata.extra_visual_bvhs:
            cls._update_visual_bvh_for_solver(entry.solver, entry.aabb, entry.bvh, entry.raycast_mask)

    def build(self):
        super().build()

        # Static sensor (entity_idx<0): the base RigidSensorMixin.build only sets self._link=None and returns, so the
        # per-sensor solver/index lists and the offsets arrays still need entries here for the raycaster's cross-solver
        # gather and the [..., -1, :] read below to stay aligned. links_idx is left alone - it is consumed only by
        # non-raycaster sensors.
        if self._options.entity_idx is None or self._options.entity_idx < 0:
            sim = self._manager._sim
            batch_size = sim._B
            self._shared_metadata._sensor_link_solvers.append(None)
            self._shared_metadata._sensor_link_indices.append(-1)
            self._shared_metadata.offsets_pos = concat_with_tensor(
                self._shared_metadata.offsets_pos,
                self._options.pos_offset,
                expand=(batch_size, 1, 3),
                dim=1,
            )
            self._shared_metadata.offsets_quat = concat_with_tensor(
                self._shared_metadata.offsets_quat,
                euler_to_quat([self._options.euler_offset]),
                expand=(batch_size, 1, 4),
                dim=1,
            )

        # first lidar sensor initialization: build aabb and bvh
        if self._shared_metadata.bvh is None:
            self._shared_metadata.sensor_cache_offsets = concat_with_tensor(
                self._shared_metadata.sensor_cache_offsets, 0
            )

            from genesis.engine.solvers.rigid.rigid_solver import RigidSolver

            sim = self._manager._sim
            solver = self._shared_metadata.solver
            n_envs = solver._B

            # Primary solver: RigidSolver always uses its collision BVH; KinematicSolver has no collision geometry so
            # it falls back to the visual BVH built from vverts_state.
            use_visual = not isinstance(solver, RigidSolver)
            self._shared_metadata.use_visual_bvh = use_visual

            if use_visual:
                n_faces = solver.vfaces_info.vgeom_idx.shape[0]
                self._shared_metadata.visual_raycast_mask = self._compute_visual_raycast_mask(solver)
            else:
                n_faces = solver.faces_info.geom_idx.shape[0]
            self._shared_metadata.aabb = AABB(n_batches=n_envs, n_aabbs=n_faces)
            self._shared_metadata.bvh = LBVH(
                self._shared_metadata.aabb, max_n_query_result_per_aabb=0, n_radix_sort_groups=64
            )

            # Add a secondary visual BVH for every other active solver that has at least one entity opted into visual
            # raycasting. The merge kernel keeps the closer hit, so collision and visual meshes coexist transparently;
            # this is how a Rigid entity with use_visual_raycasting=True gets its visual mesh raycast alongside the
            # default collision BVH.
            for other_solver in [sim.rigid_solver, sim.kinematic_solver]:
                if not other_solver.is_active:
                    continue
                if other_solver is solver and use_visual:
                    # Already covered by the primary visual BVH.
                    continue
                if other_solver.vfaces_info.vgeom_idx.shape[0] == 0:
                    continue
                extra_mask = self._compute_visual_raycast_mask(other_solver)
                if not extra_mask.any():
                    continue
                extra_n_faces = other_solver.vfaces_info.vgeom_idx.shape[0]
                extra_aabb = AABB(n_batches=n_envs, n_aabbs=extra_n_faces)
                extra_bvh = LBVH(extra_aabb, max_n_query_result_per_aabb=0, n_radix_sort_groups=64)
                self._shared_metadata.extra_visual_bvhs.append(
                    _SolverBVH(
                        solver=other_solver,
                        bvh=extra_bvh,
                        aabb=extra_aabb,
                        raycast_mask=extra_mask,
                    )
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

        # When multi-solver merge is active, the merge kernel uses distance comparison to pick the closer hit. This
        # only works if no_hit_value >= max_range; otherwise a "no hit" from one BVH could shadow a real hit from the
        # other. The negated form also rejects NaN (every IEEE 754 comparison with NaN is False).
        if self._shared_metadata.extra_visual_bvhs and not (no_hit_value >= self._options.max_range):
            gs.raise_exception(
                f"no_hit_value ({no_hit_value}) must be >= max_range ({self._options.max_range}) "
                f"when multi-solver visual raycasting is active (the merge kernel compares raw distances)."
            )

    @classmethod
    def reset(cls, shared_metadata: RaycasterSharedMetadata, shared_ground_truth_cache: torch.Tensor, envs_idx):
        super().reset(shared_metadata, shared_ground_truth_cache, envs_idx)
        cls._update_bvh(shared_metadata)

    def _get_return_format(self) -> tuple[tuple[int, ...], ...]:
        shape = self._options.pattern.return_shape
        return (*shape, 3), shape

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def _cast_visual_rays(cls, solver, bvh, shared_metadata, links_pos, links_quat, output_cache):
        """Cast rays against a single solver's visual BVH."""
        kernel_cast_rays_visual(
            solver.vverts_info,
            solver.vverts_state,
            solver.vfaces_info,
            solver.vgeoms_state,
            bvh.nodes,
            bvh.morton_codes,
            links_pos,
            links_quat,
            shared_metadata.ray_starts,
            shared_metadata.ray_dirs,
            shared_metadata.max_ranges,
            shared_metadata.no_hit_values,
            shared_metadata.return_world_frame,
            shared_metadata.points_to_sensor_idx,
            shared_metadata.sensor_cache_offsets,
            shared_metadata.sensor_point_offsets,
            shared_metadata.sensor_point_counts,
            output_cache,
            gs.EPS,
        )

    @classmethod
    def _update_shared_ground_truth_cache(
        cls, shared_metadata: RaycasterSharedMetadata, shared_ground_truth_cache: torch.Tensor
    ):
        cls._update_bvh(shared_metadata)

        # Gather per-sensor link poses. Sensors can live on different solvers (e.g. a depth camera mounted on a rigid
        # robot pointing at a kinematic mesh), so each sensor column is fetched from its own solver via bulk lookups;
        # static sensors (solver=None) keep identity transforms.
        solvers_list = shared_metadata._sensor_link_solvers
        indices_list = shared_metadata._sensor_link_indices
        n_sensors = len(solvers_list)
        B = shared_metadata.solver._B
        links_pos = torch.zeros(B, n_sensors, 3, device=gs.device, dtype=gs.tc_float)
        links_quat = torch.zeros(B, n_sensors, 4, device=gs.device, dtype=gs.tc_float)
        links_quat[:, :, 0] = 1.0
        groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
        solver_by_id: dict[int, object] = {}
        for i, (solver, link_idx) in enumerate(zip(solvers_list, indices_list)):
            if solver is not None:
                solver_by_id[id(solver)] = solver
                groups[id(solver)].append((i, link_idx))
        for sid, members in groups.items():
            solver = solver_by_id[sid]
            link_indices = torch.tensor([m[1] for m in members], device=gs.device, dtype=gs.tc_int)
            pos = solver.get_links_pos(links_idx=link_indices)
            quat = solver.get_links_quat(links_idx=link_indices)
            if solver.n_envs == 0:
                pos = pos[None]
                quat = quat[None]
            for j, (sensor_col, _) in enumerate(members):
                links_pos[:, sensor_col, :] = pos[:, j, :]
                links_quat[:, sensor_col, :] = quat[:, j, :]

        # Cast against primary BVH
        if shared_metadata.use_visual_bvh:
            cls._cast_visual_rays(
                shared_metadata.solver,
                shared_metadata.bvh,
                shared_metadata,
                links_pos,
                links_quat,
                shared_ground_truth_cache,
            )
        else:
            kernel_cast_rays(
                shared_metadata.solver.fixed_verts_state,
                shared_metadata.solver.free_verts_state,
                shared_metadata.solver.verts_info,
                shared_metadata.solver.faces_info,
                shared_metadata.bvh.nodes,
                shared_metadata.bvh.morton_codes,
                links_pos,
                links_quat,
                shared_metadata.ray_starts,
                shared_metadata.ray_dirs,
                shared_metadata.max_ranges,
                shared_metadata.no_hit_values,
                shared_metadata.return_world_frame,
                shared_metadata.points_to_sensor_idx,
                shared_metadata.sensor_cache_offsets,
                shared_metadata.sensor_point_offsets,
                shared_metadata.sensor_point_counts,
                shared_ground_truth_cache,
                gs.EPS,
            )

        # Cast against extra solver BVHs and merge closest hits
        for entry in shared_metadata.extra_visual_bvhs:
            if shared_metadata._secondary_cache is None:
                shared_metadata._secondary_cache = torch.full_like(shared_ground_truth_cache, float("inf"))
            shared_metadata._secondary_cache.fill_(float("inf"))
            cls._cast_visual_rays(
                entry.solver,
                entry.bvh,
                shared_metadata,
                links_pos,
                links_quat,
                shared_metadata._secondary_cache,
            )
            kernel_merge_ray_hits(
                shared_ground_truth_cache,
                shared_metadata._secondary_cache,
                shared_metadata.points_to_sensor_idx,
                shared_metadata.sensor_cache_offsets,
                shared_metadata.sensor_point_offsets,
                shared_metadata.sensor_point_counts,
            )

    @classmethod
    def _update_shared_cache(
        cls,
        shared_metadata: RaycasterSharedMetadata,
        shared_ground_truth_cache: torch.Tensor,
        shared_cache: torch.Tensor,
        buffered_data: "TensorRingBuffer",
    ):
        cls._apply_delay_to_shared_cache(shared_metadata, shared_cache, buffered_data)

    def _draw_debug(self, context: "RasterizerContext"):
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
