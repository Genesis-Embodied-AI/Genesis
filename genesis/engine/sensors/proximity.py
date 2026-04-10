from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.engine.solvers.rigid.abd.forward_kinematics import func_update_all_verts
from genesis.options.sensors import Proximity as ProximityOptions
from genesis.utils.misc import concat_with_tensor, make_tensor_field, tensor_to_array
from genesis.utils.raycast_qd import get_triangle_vertices

from .base_sensor import (
    NoisySensorMetadataMixin,
    NoisySensorMixin,
    RigidSensorMetadataMixin,
    RigidSensorMixin,
    Sensor,
    SharedSensorMetadata,
)
from .kinematic_tactile import _func_closest_point_on_triangle

if TYPE_CHECKING:
    from genesis.utils.ring_buffer import TensorRingBuffer
    from genesis.vis.rasterizer_context import RasterizerContext

    from .sensor_manager import SensorManager


@qd.kernel(fastcache=gs.use_fastcache)
def _kernel_proximity(
    probe_positions_local: qd.types.ndarray(),
    probe_sensor_idx: qd.types.ndarray(),
    links_idx: qd.types.ndarray(),
    sensor_cache_start: qd.types.ndarray(),
    sensor_probe_start: qd.types.ndarray(),
    n_probes_per_sensor: qd.types.ndarray(),
    track_link_start: qd.types.ndarray(),
    track_link_end: qd.types.ndarray(),
    track_link_flat: qd.types.ndarray(),
    max_range: qd.types.ndarray(),
    static_rigid_sim_config: qd.template(),
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    faces_info: array_class.FacesInfo,
    verts_info: array_class.VertsInfo,
    fixed_verts_state: array_class.VertsState,
    free_verts_state: array_class.VertsState,
    positions_output: qd.types.ndarray(),
    output: qd.types.ndarray(),
):
    total_n_probes = probe_positions_local.shape[0]
    n_batches = output.shape[-1]

    func_update_all_verts(
        geoms_info, geoms_state, verts_info, free_verts_state, fixed_verts_state, static_rigid_sim_config
    )

    for i_p, i_b in qd.ndrange(total_n_probes, n_batches):
        i_s = probe_sensor_idx[i_p]
        sensor_link_idx = links_idx[i_s]
        link_pos = links_state.pos[sensor_link_idx, i_b]
        link_quat = links_state.quat[sensor_link_idx, i_b]

        probe_pos_local = qd.Vector(
            [probe_positions_local[i_p, 0], probe_positions_local[i_p, 1], probe_positions_local[i_p, 2]]
        )
        probe_pos = link_pos + gu.qd_transform_by_quat(probe_pos_local, link_quat)

        best_dist_sq = max_range[i_s] * max_range[i_s]
        best_point = probe_pos
        start = track_link_start[i_s]
        end = track_link_end[i_s]

        for k in range(start, end):
            i_l = track_link_flat[k]
            I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
            geom_start = links_info.geom_start[I_l]
            geom_end = links_info.geom_end[I_l]

            for i_g in range(geom_start, geom_end):
                face_start = geoms_info.face_start[i_g]
                face_end = geoms_info.face_end[i_g]

                for i_f in range(face_start, face_end):
                    tri_verts = get_triangle_vertices(
                        i_f, i_b, faces_info, verts_info, fixed_verts_state, free_verts_state
                    )
                    v0 = tri_verts[:, 0]
                    v1 = tri_verts[:, 1]
                    v2 = tri_verts[:, 2]
                    closest = _func_closest_point_on_triangle(probe_pos, v0, v1, v2)
                    diff = closest - probe_pos
                    dist_sq = diff.dot(diff)
                    if dist_sq < best_dist_sq:
                        best_dist_sq = dist_sq
                        best_point = closest

        best_dist = qd.sqrt(best_dist_sq)

        probe_idx_in_sensor = i_p - sensor_probe_start[i_s]
        cache_start = sensor_cache_start[i_s]
        n_probes = n_probes_per_sensor[i_s]
        probe_global_idx = sensor_probe_start[i_s] + probe_idx_in_sensor

        output[cache_start + probe_idx_in_sensor, i_b] = best_dist
        for j in qd.static(range(3)):
            positions_output[i_b, probe_global_idx, j] = best_point[j]  # not part of cache, stays (B, ...)


@dataclass
class ProximitySensorMetadataMixin:
    """Shared metadata for proximity sensors: probe layout and tracked links."""

    total_n_probes: int = 0
    probe_positions: torch.Tensor = make_tensor_field((0, 3))
    n_probes_per_sensor: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    probe_sensor_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    sensor_cache_start: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    sensor_probe_start: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    track_link_start: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    track_link_end: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    track_link_flat: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    nearest_positions: torch.Tensor = make_tensor_field((0, 0, 3))
    max_range: torch.Tensor = make_tensor_field((0,))


@dataclass
class ProximityMetadata(
    ProximitySensorMetadataMixin, RigidSensorMetadataMixin, NoisySensorMetadataMixin, SharedSensorMetadata
):
    """Shared metadata for the Proximity sensor class."""


class ProximitySensor(
    RigidSensorMixin[ProximityMetadata],
    NoisySensorMixin[ProximityMetadata],
    Sensor[ProximityOptions, ProximityMetadata, tuple],
):
    """Proximity sensor: distance and nearest point from probe positions to tracked mesh surfaces."""

    def __init__(self, sensor_options: ProximityOptions, sensor_idx: int, sensor_manager: "SensorManager"):
        self._probe_local_pos = torch.tensor(sensor_options.probe_local_pos, dtype=gs.tc_float, device=gs.device)
        self._n_probes = int(np.prod(self._probe_local_pos.shape[:-1]))
        super().__init__(sensor_options, sensor_idx, sensor_manager)
        self._debug_objects: list = []
        self._nearest_points_slice: slice = slice(None)

    def _get_return_format(self) -> tuple[tuple[int, ...], ...]:
        return (self._n_probes,)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    def build(self):
        super().build()

        self._shared_metadata.probe_positions = concat_with_tensor(
            self._shared_metadata.probe_positions, self._probe_local_pos, expand=(self._n_probes, 3)
        )
        self._shared_metadata.n_probes_per_sensor = concat_with_tensor(
            self._shared_metadata.n_probes_per_sensor, self._n_probes, expand=(1,)
        )
        self._shared_metadata.sensor_cache_start = concat_with_tensor(
            self._shared_metadata.sensor_cache_start,
            sum(self._shared_metadata.cache_sizes[:-1]) if self._shared_metadata.cache_sizes else 0,
            expand=(1,),
        )
        self._shared_metadata.sensor_probe_start = concat_with_tensor(
            self._shared_metadata.sensor_probe_start,
            self._shared_metadata.total_n_probes,
            expand=(1,),
        )
        self._shared_metadata.probe_sensor_idx = concat_with_tensor(
            self._shared_metadata.probe_sensor_idx,
            torch.full((self._n_probes,), self._idx, dtype=gs.tc_int, device=gs.device),
            expand=(self._n_probes,),
        )

        track_link_idx = np.asarray(self._options.track_link_idx, dtype=gs.np_int)
        n_tracked = len(track_link_idx)
        start = (
            int(self._shared_metadata.track_link_flat.shape[0])
            if self._shared_metadata.track_link_flat.numel() > 0
            else 0
        )
        self._shared_metadata.track_link_start = concat_with_tensor(
            self._shared_metadata.track_link_start, start, expand=(1,)
        )
        self._shared_metadata.track_link_end = concat_with_tensor(
            self._shared_metadata.track_link_end, start + n_tracked, expand=(1,)
        )
        track_flat = torch.tensor(track_link_idx, dtype=gs.tc_int, device=gs.device)
        self._shared_metadata.track_link_flat = concat_with_tensor(
            self._shared_metadata.track_link_flat, track_flat, expand=(n_tracked,)
        )
        self._shared_metadata.max_range = concat_with_tensor(
            self._shared_metadata.max_range, float(self._options.max_range), expand=(1,)
        )

        self._shared_metadata.total_n_probes += self._n_probes
        self._shared_metadata.nearest_positions = torch.zeros(
            (self._manager._sim._B, self._shared_metadata.total_n_probes, 3), dtype=gs.tc_float, device=gs.device
        )
        slice_start = self._shared_metadata.sensor_probe_start[self._idx]
        self._nearest_points_slice = slice(slice_start, slice_start + self._n_probes)

    @classmethod
    def reset(cls, shared_metadata: ProximityMetadata, shared_ground_truth_cache: torch.Tensor, envs_idx):
        super().reset(shared_metadata, shared_ground_truth_cache, envs_idx)
        shared_metadata.nearest_positions[envs_idx] = shared_metadata.probe_positions

    @classmethod
    def _update_shared_ground_truth_cache(
        cls, shared_metadata: ProximityMetadata, shared_ground_truth_cache: torch.Tensor
    ):
        solver = shared_metadata.solver
        shared_ground_truth_cache.zero_()
        _kernel_proximity(
            shared_metadata.probe_positions,
            shared_metadata.probe_sensor_idx,
            shared_metadata.links_idx,
            shared_metadata.sensor_cache_start,
            shared_metadata.sensor_probe_start,
            shared_metadata.n_probes_per_sensor,
            shared_metadata.track_link_start,
            shared_metadata.track_link_end,
            shared_metadata.track_link_flat,
            shared_metadata.max_range,
            solver._static_rigid_sim_config,
            solver.links_state,
            solver.links_info,
            solver.geoms_info,
            solver.geoms_state,
            solver.faces_info,
            solver.verts_info,
            solver.fixed_verts_state,
            solver.free_verts_state,
            shared_metadata.nearest_positions,
            shared_ground_truth_cache,
        )

    @classmethod
    def _update_shared_cache(
        cls,
        shared_metadata: ProximityMetadata,
        shared_ground_truth_cache: torch.Tensor,
        shared_cache: torch.Tensor,
        buffered_data: "TensorRingBuffer",
    ):
        buffered_data.set(shared_ground_truth_cache)
        torch.normal(0.0, shared_metadata.jitter_ts, out=shared_metadata.cur_jitter_ts)
        cls._apply_delay_to_shared_cache(
            shared_metadata,
            shared_cache,
            buffered_data,
            shared_metadata.cur_jitter_ts,
            shared_metadata.interpolate,
        )
        cls._add_noise_drift_bias(shared_metadata, shared_cache)
        cls._quantize_to_resolution(shared_metadata.resolution, shared_cache)

    def _draw_debug(self, context: "RasterizerContext"):
        env_idx = context.rendered_envs_idx[0] if self._manager._sim.n_envs > 0 else None
        for obj in self._debug_objects:
            context.clear_debug_object(obj)
        self._debug_objects.clear()

        link_pos = self._link.get_pos(env_idx).squeeze()
        link_quat = self._link.get_quat(env_idx).squeeze()
        probe_world = tensor_to_array(gu.transform_by_trans_quat(self._probe_local_pos, link_pos, link_quat))
        points = self.nearest_points[env_idx]

        self._debug_objects.append(
            context.draw_debug_spheres(
                poss=np.concatenate([probe_world, points]),
                radius=self._options.debug_sphere_radius,
                color=self._options.debug_color,
            )
        )
        for i in range(len(probe_world)):
            line_obj = context.draw_debug_line(
                probe_world[i],
                points[i],
                radius=self._options.debug_sphere_radius / 4.0,
                color=self._options.debug_color,
            )
            self._debug_objects.append(line_obj)

    @property
    def nearest_points(self) -> torch.Tensor:
        return self._shared_metadata.nearest_positions[:, self._nearest_points_slice, :]
