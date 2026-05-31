import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

import genesis as gs
from genesis.engine.bvh import AABB, LBVH
from genesis.engine.solvers.rigid.rigid_solver import RigidSolver
from genesis.options.sensors import Raycaster as RaycasterOptions
from genesis.options.sensors import RaycastPattern
from genesis.utils.geom import transform_by_quat, transform_by_trans_quat
from genesis.utils.misc import concat_with_tensor, make_tensor_field, qd_to_numpy
from genesis.utils.raycast_qd import (
    kernel_cast_rays,
    kernel_cast_rays_visual,
    kernel_update_visual_aabbs,
    kernel_update_verts_and_aabbs,
)
from genesis.vis.rasterizer_context import RasterizerContext

from .base_sensor import KinematicSensorMetadataMixin, KinematicSensorMixin, Sensor, SimpleSensorMetadata, SimpleSensor

if TYPE_CHECKING:
    from genesis.engine.solvers.kinematic_solver import KinematicSolver
    from genesis.ext.pyrender.mesh import Mesh
    from genesis.utils.ring_buffer import TensorRingBuffer

    from .sensor_manager import SensorManager


@dataclass
class _SolverBVH:
    """
    One BVH built against a solver's mesh.

    ``raycast_mask`` is ``None`` for a collision BVH (``faces_info`` / ``verts_info``, no per-face mask), otherwise an
    int8 array of shape ``(n_vfaces,)`` selecting which visual faces contribute.
    """

    solver: "KinematicSolver"
    bvh: LBVH
    aabb: AABB
    raycast_mask: np.ndarray | None

    # True when no link in the solver has a DOF the physics can move, so its geometry only ever changes through an
    # explicit user set_pos/set_quat. Full BVH rebuilds (the dominant per-step cost for static-terrain raycasting) are
    # then skipped while the link poses are unchanged since the last build; a set_pos is detected via the pose cache
    # below and triggers a rebuild. Collision-only: visual BVHs stay always-rebuilt because set_vverts can change their
    # vverts without moving any link.
    maybe_static: bool = False
    # Whether build() has already run for this entry (reset() clears it to force one rebuild).
    built: bool = False
    # Snapshot of all link poses at the last build, used to detect a set_pos on an otherwise-static solver. Only
    # populated for maybe_static entries.
    cached_link_pos: torch.Tensor | None = None
    cached_link_quat: torch.Tensor | None = None
    # True when every env's link poses match, so the BVH is identical across envs and the cast reads one shared copy
    # (batch 0) with coalesced node loads instead of scattering over n_env identical trees. Recomputed each build.
    shared_across_envs: bool = False
    # Compacted face subset this BVH covers: ``face_ids[k]`` is the global face index at BVH leaf slot ``k``. Static
    # (fixed-link) and dynamic (movable-link) collision faces get separate subsets so the static tree is built once
    # while only the small dynamic subset rebuilds. A 1-D int32 device tensor; ``None`` for visual BVH entries.
    face_ids: torch.Tensor | None = None
    # Global link indices whose faces are in this subset. The rebuild-skip pose check + shared-across-envs test read
    # only these links (via get_links_pos(links_idx=link_ids)), so a static subset stays static and shared even while
    # the robot's links move. ``None`` for visual BVH entries (which never skip).
    link_ids: torch.Tensor | None = None


@dataclass
class RaycasterSharedMetadata(KinematicSensorMetadataMixin, SimpleSensorMetadata):
    # All BVHs (one per active solver per mesh type) cast against each frame. The first is written into the output cache
    # with is_merge=False (initializes hits or no_hit_value), the rest merge in closer hits. Per-sensor link poses are
    # gathered via KinematicSensorMetadataMixin.solver_groups, independent of which BVH is being cast.
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


class RaycasterSensor(KinematicSensorMixin, SimpleSensor[RaycasterOptions, RaycasterSharedMetadata, RaycasterData]):
    def __init__(self, options: RaycasterOptions, sensor_idx: int, manager: "SensorManager"):
        super().__init__(options, sensor_idx, manager)
        self.debug_objects: list["Mesh"] = []
        self.ray_starts: torch.Tensor = torch.empty((0, 3), device=gs.device, dtype=gs.tc_float)

    @staticmethod
    def _partition_collision_faces(solver: "RigidSolver") -> list[tuple[torch.Tensor, torch.Tensor, bool]]:
        """Partition the solver's collision faces into static (fixed-link) and dynamic (movable-link) subsets.

        Returns a list of ``(face_ids, link_ids, maybe_static)`` — one entry per non-empty subset, where
        ``face_ids`` are global face indices and ``link_ids`` the global indices of the links owning them.
        A pure-static or pure-dynamic solver yields a single entry (equivalent to one full-mesh BVH); a mixed
        scene (robot on terrain) yields two, so the static terrain tree can be built once and shared while only
        the robot subset rebuilds per step.
        """
        face_geom = qd_to_numpy(solver.faces_info.geom_idx).reshape(-1)  # (n_faces,) global geom per face
        geom_link = qd_to_numpy(solver.geoms_info.link_idx).reshape(-1)  # (n_geoms,) global link per geom
        link_fixed = np.array([bool(link.is_fixed) for link in solver.links], dtype=bool)  # (n_links,)
        face_link = geom_link[face_geom]  # (n_faces,) global link per face
        face_static = link_fixed[face_link]  # (n_faces,) is this face on a fixed link?

        out: list[tuple[torch.Tensor, torch.Tensor, bool]] = []
        for is_static in (True, False):
            sel = np.nonzero(face_static == is_static)[0]
            if sel.size == 0:
                continue
            face_ids = torch.as_tensor(sel, dtype=gs.tc_int, device=gs.device)
            link_ids = torch.as_tensor(np.unique(face_link[sel]), dtype=gs.tc_int, device=gs.device)
            out.append((face_ids, link_ids, bool(is_static)))
        return out

    @staticmethod
    def _compute_visual_raycast_mask(solver: "KinematicSolver") -> np.ndarray:
        """Build a per-vface mask (int8, shape (n_vfaces,)) selecting vfaces opted into visual raycasting.
        A vface is opted in iff its owning vgeom belongs to an entity whose material has use_visual_raycasting=True.
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
        """Rebuild every BVH from current geometry in the scene.

        For a maybe_static entry (no link the physics can move) the rebuild is skipped while the link poses are
        byte-identical to the snapshot from the last build, since the tree would come out unchanged. A user set_pos on
        such a (fixed) link shows up as a pose difference and forces a rebuild, so correctness is preserved while the
        common static-terrain case avoids the per-step rebuild entirely — the dominant cost on that path.
        """
        for entry in shared_metadata.solver_bvhs:
            # The pose check reads only this subset's links (link_ids), so a static-terrain subset stays
            # skippable even while the robot's links move in the same solver; a set_pos on a fixed terrain
            # link still shows up here and forces a rebuild. link_ids is None only for visual entries, which
            # never set maybe_static, so the short-circuit never reaches the get_links_pos call for them.
            if (
                entry.maybe_static
                and entry.built
                and entry.cached_link_pos is not None
                and torch.equal(entry.solver.get_links_pos(links_idx=entry.link_ids), entry.cached_link_pos)
                and torch.equal(entry.solver.get_links_quat(links_idx=entry.link_ids), entry.cached_link_quat)
            ):
                continue
            if entry.raycast_mask is None:
                kernel_update_verts_and_aabbs(
                    geoms_info=entry.solver.geoms_info,
                    geoms_state=entry.solver.geoms_state,
                    verts_info=entry.solver.verts_info,
                    faces_info=entry.solver.faces_info,
                    free_verts_state=entry.solver.free_verts_state,
                    fixed_verts_state=entry.solver.fixed_verts_state,
                    static_rigid_sim_config=entry.solver._static_rigid_sim_config,
                    face_ids=entry.face_ids,
                    aabb_state=entry.aabb,
                )
                entry.bvh.build()
            else:
                # Reads vverts_state.pos as the source of vvert positions. The buffer is seeded by FK at scene.build()
                # and refreshed for each user-driven entity via set_vverts; entries set via set_vverts survive across
                # calls until set_vverts(None) re-runs FK over the entity's vgeoms. raycast_mask gates which vfaces
                # contribute to the BVH; masked-out vfaces keep an inverted AABB and are skipped by ray queries.
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
            entry.built = True
            # Snapshot the post-build link poses so the next call can detect a set_pos and rebuild only if needed, and
            # decide whether one shared BVH copy can serve all envs (see shared_across_envs).
            if entry.maybe_static:
                pos = entry.solver.get_links_pos(links_idx=entry.link_ids)
                quat = entry.solver.get_links_quat(links_idx=entry.link_ids)
                entry.cached_link_pos = pos
                entry.cached_link_quat = quat
                # Identical link poses across envs <=> identical world geometry in every env, so the per-env trees are
                # bit-identical and the cast can read a single copy (batch 0) with coalesced loads. get_links_pos
                # returns (n_envs, n_links, 3) for a batched solver; a single-env solver gains nothing from sharing.
                entry.shared_across_envs = bool(
                    pos.ndim == 3
                    and pos.shape[0] > 1
                    and torch.equal(pos, pos[:1].expand_as(pos))
                    and torch.equal(quat, quat[:1].expand_as(quat))
                )

    def build(self):
        super().build()

        # First raycast sensor: build the per-(solver, mesh-type) BVHs once. Rigid solvers get a collision BVH covering
        # all collision faces; any solver with entities opting in via material.use_visual_raycasting gets a visual BVH
        # masked to those entities' vfaces. Collision and visual entries coexist transparently because the cast kernels
        # merge in place via is_merge.
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
                    # Split the solver's collision faces into a static subset (faces whose owning link is
                    # fixed — terrain, walls) and a dynamic subset (faces on links the physics can move —
                    # the robot). Each gets its own compacted BVH so (a) the static subset's tree is built
                    # once then skipped + shared across envs (the dominant per-step cost for one robot on a
                    # big static terrain), and (b) the dynamic rebuild scales with the robot's face count,
                    # not the whole scene. The cast kernels merge the two via is_merge, so the result is
                    # identical to one combined BVH. This is the RPL "multi-depth" decomposition
                    # (arXiv:2602.03002): cast dynamic robot + static terrain meshes separately, then merge.
                    for face_ids, link_ids, maybe_static in self._partition_collision_faces(solver):
                        n_sub = int(face_ids.shape[0])
                        aabb = AABB(n_batches=n_envs, n_aabbs=n_sub)
                        bvh = LBVH(aabb, max_n_query_result_per_aabb=0, n_radix_sort_groups=64)
                        self._shared_metadata.solver_bvhs.append(
                            _SolverBVH(
                                solver,
                                bvh,
                                aabb,
                                None,
                                maybe_static=maybe_static,
                                face_ids=face_ids,
                                link_ids=link_ids,
                            )
                        )
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

        # Multi-BVH merge passes use raw distance comparison to pick the closer hit; this only works if no_hit_value >=
        # max_range. The negated form also rejects NaN (every IEEE 754 comparison with NaN is False).
        if len(self._shared_metadata.solver_bvhs) > 1 and not (self._options.no_hit_value >= self._options.max_range):
            gs.raise_exception(
                f"no_hit_value ({self._options.no_hit_value}) must be >= max_range ({self._options.max_range}) "
                f"when multiple BVHs are active (the merge step compares raw distances)."
            )

    @classmethod
    def reset(cls, shared_metadata: RaycasterSharedMetadata, current_ground_truth_data_T: torch.Tensor, envs_idx):
        super().reset(shared_metadata, current_ground_truth_data_T, envs_idx)
        # A reset may change otherwise-static geometry (e.g. re-randomized terrain), so force every entry to rebuild
        # once here; static entries then resume being skipped on subsequent steps.
        for entry in shared_metadata.solver_bvhs:
            entry.built = False
        cls._update_bvh(shared_metadata)

    def _get_return_format(self) -> tuple[tuple[int, ...], ...]:
        shape = self._options.pattern.return_shape
        return ((*shape, 3), shape)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def _update_raw_data(cls, shared_metadata: RaycasterSharedMetadata, raw_data_T: torch.Tensor):
        cls._update_bvh(shared_metadata)

        # Allocate the link-pose scratch buffers on first cast (B and n_sensors are known here). Identity quat is baked
        # into the initial allocation so static sensors (entity_idx<0) leave their rows at identity, letting the cast
        # kernel apply pos_offset / euler_offset in world frame.
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
                raw_data_T,
                gs.EPS,
                i > 0,
                entry.shared_across_envs,
            )
            if entry.raycast_mask is None:
                kernel_cast_rays(
                    solver.fixed_verts_state,
                    solver.free_verts_state,
                    solver.verts_info,
                    solver.faces_info,
                    entry.face_ids,
                    *args_common,
                )
            else:
                kernel_cast_rays_visual(
                    solver.vverts_info, solver.vverts_state, solver.vfaces_info, solver.vgeoms_state, *args_common
                )

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
                ray_starts, radius=self._options.debug_sphere_radius, color=self._options.debug_ray_start_color
            ),
            context.draw_debug_spheres(
                points, radius=self._options.debug_sphere_radius, color=self._options.debug_ray_hit_color
            ),
        ]
