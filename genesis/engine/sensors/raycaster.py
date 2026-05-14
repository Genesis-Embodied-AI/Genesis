import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

import genesis as gs
from genesis.engine.bvh import AABB, LBVH
from genesis.engine.solvers.rigid.rigid_solver import RigidSolver
from genesis.options.sensors import (
    Raycaster as RaycasterOptions,
)
from genesis.options.sensors import (
    RaycastPattern,
)
from genesis.utils.geom import transform_by_quat, transform_by_trans_quat
from genesis.utils.misc import concat_with_tensor, make_tensor_field, qd_to_numpy
from genesis.utils.raycast_qd import (
    kernel_cast_rays,
    kernel_cast_rays_visual,
    kernel_update_visual_aabbs,
    kernel_update_verts_and_aabbs,
)
from genesis.vis.rasterizer_context import RasterizerContext

from .base_sensor import (
    KinematicSensorMetadataMixin,
    KinematicSensorMixin,
    Sensor,
    SharedSensorMetadata,
)

if TYPE_CHECKING:
    from genesis.engine.solvers.kinematic_solver import KinematicSolver
    from genesis.ext.pyrender.mesh import Mesh
    from genesis.utils.ring_buffer import TensorRingBuffer

    from .sensor_manager import SensorManager


class _SolverBVH(NamedTuple):
    """One BVH built against a solver's mesh. raycast_mask is None for a collision BVH (faces_info / verts_info,
    no per-face mask), otherwise an int8 array of shape (n_vfaces,) selecting which visual faces contribute."""

    solver: "KinematicSolver"
    bvh: LBVH
    aabb: AABB
    raycast_mask: np.ndarray | None


@dataclass
class RaycasterSharedMetadata(KinematicSensorMetadataMixin, SharedSensorMetadata):
    # All BVHs (one per active solver per mesh type) cast against each frame. The first is written into the output
    # cache with is_merge=False (initializes hits or no_hit_value), the rest merge in closer hits. Per-sensor link
    # poses are gathered via KinematicSensorMetadataMixin.solver_groups, independent of which BVH is being cast.
    solver_bvhs: list[_SolverBVH] = field(default_factory=list)

    # Per-step scratch tensors for sensor link poses, lazily allocated on the first cast (B and n_sensors known).
    links_pos: torch.Tensor | None = None
    links_quat: torch.Tensor | None = None

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


class RaycasterSensor(KinematicSensorMixin, Sensor[RaycasterOptions, RaycasterSharedMetadata, RaycasterData]):
    def __init__(self, options: RaycasterOptions, sensor_idx: int, manager: "SensorManager"):
        super().__init__(options, sensor_idx, manager)
        self.debug_objects: list["Mesh"] = []
        self.ray_starts: torch.Tensor = torch.empty((0, 3), device=gs.device, dtype=gs.tc_float)

    @staticmethod
    def _compute_visual_raycast_mask(solver: "KinematicSolver") -> np.ndarray:
        """Build a per-vface mask (int8, shape (n_vfaces,)) selecting vfaces opted into visual raycasting.

        A vface is opted in iff its owning vgeom belongs to an entity whose material has
        use_visual_raycasting=True.
        """
        n_vfaces = solver.vfaces_info.vgeom_idx.shape[0]
        if n_vfaces == 0:
            return np.zeros(0, dtype=np.int8)
        vgeom_enabled = np.zeros(solver.n_vgeoms, dtype=np.bool_)
        for entity in solver.entities:
            if not entity.material.use_visual_raycasting:
                continue
            for vgeom in entity.vgeoms:
                vgeom_enabled[vgeom.idx] = True
        vface_vgeom_idx = qd_to_numpy(solver.vfaces_info.vgeom_idx)
        return vgeom_enabled[vface_vgeom_idx].astype(np.int8)

    @classmethod
    def _update_bvh(cls, shared_metadata: RaycasterSharedMetadata):
        """Rebuild every BVH from current geometry in the scene."""
        for entry in shared_metadata.solver_bvhs:
            if entry.raycast_mask is None:
                kernel_update_verts_and_aabbs(
                    geoms_info=entry.solver.geoms_info,
                    geoms_state=entry.solver.geoms_state,
                    verts_info=entry.solver.verts_info,
                    faces_info=entry.solver.faces_info,
                    free_verts_state=entry.solver.free_verts_state,
                    fixed_verts_state=entry.solver.fixed_verts_state,
                    static_rigid_sim_config=entry.solver._static_rigid_sim_config,
                    aabb_state=entry.aabb,
                )
                entry.bvh.build()
            else:
                # Reads vverts_state.pos as the source of vvert positions. The buffer is seeded by FK at
                # scene.build() and refreshed for each user-driven entity via set_vverts; entries set via
                # set_vverts survive across calls until set_vverts(None) re-runs FK over the entity's vgeoms.
                # raycast_mask gates which vfaces contribute to the BVH; masked-out vfaces keep an inverted
                # AABB and are skipped by ray queries.
                entry.solver.update_forward_pos()
                entry.solver.update_vgeoms()
                kernel_update_visual_aabbs(
                    vverts_info=entry.solver.vverts_info,
                    vverts_state=entry.solver.vverts_state,
                    vfaces_info=entry.solver.vfaces_info,
                    vgeoms_state=entry.solver.vgeoms_state,
                    face_mask=entry.raycast_mask,
                    aabb_state=entry.aabb,
                )
                entry.bvh.build()

    def build(self):
        super().build()

        # First raycast sensor: build the per-(solver, mesh-type) BVHs once. Rigid solvers get a collision BVH
        # covering all collision faces; any solver with entities opting in via material.use_visual_raycasting gets
        # a visual BVH masked to those entities' vfaces. Collision and visual entries coexist transparently because
        # the cast kernels merge in place via is_merge.
        if not self._shared_metadata.solver_bvhs:
            self._shared_metadata.sensor_cache_offsets = concat_with_tensor(
                self._shared_metadata.sensor_cache_offsets, 0
            )

            sim = self._manager._sim
            for solver in (sim.rigid_solver, sim.kinematic_solver):
                if not solver.is_active:
                    continue
                n_envs = solver._B
                if isinstance(solver, RigidSolver):
                    n_faces = solver.faces_info.geom_idx.shape[0]
                    aabb = AABB(n_batches=n_envs, n_aabbs=n_faces)
                    bvh = LBVH(aabb, max_n_query_result_per_aabb=0, n_radix_sort_groups=64)
                    self._shared_metadata.solver_bvhs.append(_SolverBVH(solver, bvh, aabb, None))
                n_vfaces = solver.vfaces_info.vgeom_idx.shape[0]
                if n_vfaces > 0:
                    mask = self._compute_visual_raycast_mask(solver)
                    if mask.any():
                        aabb = AABB(n_batches=n_envs, n_aabbs=n_vfaces)
                        bvh = LBVH(aabb, max_n_query_result_per_aabb=0, n_radix_sort_groups=64)
                        self._shared_metadata.solver_bvhs.append(_SolverBVH(solver, bvh, aabb, mask))

            if not self._shared_metadata.solver_bvhs:
                gs.raise_exception(
                    "Raycaster sensor has no geometry to raycast against: rigid_solver is inactive and no entity "
                    "has material.use_visual_raycasting=True."
                )

            self._update_bvh(self._shared_metadata)

        self._shared_metadata.patterns.append(self._options.pattern)

        ray_starts = self._options.pattern.ray_starts.reshape(-1, 3)
        self.ray_starts = transform_by_trans_quat(
            ray_starts, self._shared_metadata.offsets_pos[0, -1, :], self._shared_metadata.offsets_quat[0, -1, :]
        )
        self._shared_metadata.ray_starts = torch.cat([self._shared_metadata.ray_starts, self.ray_starts])

        ray_dirs = self._options.pattern.ray_dirs.reshape(-1, 3)
        ray_dirs = transform_by_quat(ray_dirs, self._shared_metadata.offsets_quat[0, -1, :])
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
        self._shared_metadata.no_hit_values = concat_with_tensor(
            self._shared_metadata.no_hit_values, self._options.no_hit_value
        )

        # Multi-BVH merge passes use raw distance comparison to pick the closer hit; this only works if
        # no_hit_value >= max_range. The negated form also rejects NaN (every IEEE 754 comparison with NaN is False).
        if len(self._shared_metadata.solver_bvhs) > 1 and not (self._options.no_hit_value >= self._options.max_range):
            gs.raise_exception(
                f"no_hit_value ({self._options.no_hit_value}) must be >= max_range ({self._options.max_range}) "
                f"when multiple BVHs are active (the merge step compares raw distances)."
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
    def _update_shared_ground_truth_cache(
        cls, shared_metadata: RaycasterSharedMetadata, shared_ground_truth_cache: torch.Tensor
    ):
        cls._update_bvh(shared_metadata)

        # Allocate the link-pose scratch buffers on first cast (B and n_sensors are known here). Identity quat is
        # baked into the initial allocation so static sensors (entity_idx<0) leave their rows at identity, letting
        # the cast kernel apply pos_offset / euler_offset in world frame.
        if shared_metadata.links_pos is None:
            B = shared_metadata.solver_bvhs[0].solver._B
            shared_metadata.links_pos = torch.zeros(
                B, shared_metadata.n_sensors, 3, device=gs.device, dtype=gs.tc_float
            )
            shared_metadata.links_quat = torch.zeros(
                B, shared_metadata.n_sensors, 4, device=gs.device, dtype=gs.tc_float
            )
            shared_metadata.links_quat[:, :, 0] = 1.0

        # Gather link poses per sensor. Sensors are pre-bucketed into shared_metadata.solver_groups at build time so
        # this loop issues one bulk get_links_pos / get_links_quat per solver with already-tensor-typed indices.
        links_pos = shared_metadata.links_pos
        links_quat = shared_metadata.links_quat
        for group in shared_metadata.solver_groups:
            pos = group.solver.get_links_pos(links_idx=group.links_idx)
            quat = group.solver.get_links_quat(links_idx=group.links_idx)
            if group.solver.n_envs == 0:
                pos = pos[None]
                quat = quat[None]
            links_pos[:, group.sensor_cols, :] = pos
            links_quat[:, group.sensor_cols, :] = quat

        # First entry initializes the cache (is_merge=False, writes a hit or no_hit_value into every slot). Each
        # subsequent entry merges in place (is_merge=True, writes only where it found a closer hit).
        for i, entry in enumerate(shared_metadata.solver_bvhs):
            solver = entry.solver
            args_common = (
                entry.bvh.nodes,
                entry.bvh.morton_codes,
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
                i > 0,
            )
            if entry.raycast_mask is None:
                kernel_cast_rays(
                    solver.fixed_verts_state,
                    solver.free_verts_state,
                    solver.verts_info,
                    solver.faces_info,
                    *args_common,
                )
            else:
                kernel_cast_rays_visual(
                    solver.vverts_info,
                    solver.vverts_state,
                    solver.vfaces_info,
                    solver.vgeoms_state,
                    *args_common,
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

        pos = self._link.get_pos(env_idx)
        quat = self._link.get_quat(env_idx)
        if pos.ndim == 2:
            pos, quat = pos[0], quat[0]

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
