from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.engine.solvers.rigid.abd.forward_kinematics import func_update_all_verts
from genesis.options.sensors import SurfaceDistanceProbe as SurfaceDistanceProbeOptions
from genesis.utils.misc import concat_with_tensor, make_tensor_field, tensor_to_array
from genesis.utils.raycast_qd import get_triangle_vertices

from .base_sensor import RigidSensorMetadataMixin, RigidSensorMixin, SimpleSensor, SimpleSensorMetadata
from .probe import ProbeSensorMetadataMixin, ProbeSensorMixin, func_noised_probe_radius

if TYPE_CHECKING:
    from genesis.utils.ring_buffer import TensorRingBuffer
    from genesis.vis.rasterizer_context import RasterizerContext

    from .sensor_manager import SensorManager


@qd.func
def _func_closest_point_on_triangle(point: gs.qd_vec3, v0: gs.qd_vec3, v1: gs.qd_vec3, v2: gs.qd_vec3) -> gs.qd_vec3:
    """
    Find the point on the surface of a triangle closest to a given point.

    Reference: Christer Ericson, *Real-Time Collision Detection*, §5.1.5.
    """
    ab = v1 - v0
    ac = v2 - v0
    ap = point - v0

    d1 = ab.dot(ap)
    d2 = ac.dot(ap)

    # Region A (vertex v0)
    closest = v0
    if not (d1 <= 0.0 and d2 <= 0.0):
        bp = point - v1
        d3 = ab.dot(bp)
        d4 = ac.dot(bp)

        # Region B (vertex v1)
        if d3 >= 0.0 and d4 <= d3:
            closest = v1
        else:
            cp = point - v2
            d5 = ab.dot(cp)
            d6 = ac.dot(cp)

            # Region C (vertex v2)
            if d6 >= 0.0 and d5 <= d6:
                closest = v2
            else:
                vc = d1 * d4 - d3 * d2
                # Region AB (edge v0-v1)
                if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                    w = d1 / (d1 - d3)
                    closest = v0 + w * ab
                else:
                    vb = d5 * d2 - d1 * d6
                    # Region AC (edge v0-v2)
                    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                        w = d2 / (d2 - d6)
                        closest = v0 + w * ac
                    else:
                        va = d3 * d6 - d5 * d4
                        # Region BC (edge v1-v2)
                        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
                            w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
                            closest = v1 + w * (v2 - v1)
                        else:
                            # Inside the triangle face
                            denom = 1.0 / (va + vb + vc)
                            v = vb * denom
                            w = vc * denom
                            closest = v0 + v * ab + w * ac

    return closest


@qd.kernel
def _kernel_surface_distance_probe(
    probe_positions_local: qd.types.ndarray(),
    probe_radii: qd.types.ndarray(),
    probe_radii_noise: qd.types.ndarray(),
    probe_sensor_idx: qd.types.ndarray(),
    links_idx: qd.types.ndarray(),
    sensor_cache_start: qd.types.ndarray(),
    sensor_probe_start: qd.types.ndarray(),
    track_link_start: qd.types.ndarray(),
    track_link_end: qd.types.ndarray(),
    track_link_flat: qd.types.ndarray(),
    static_rigid_sim_config: qd.template(),
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    faces_info: array_class.FacesInfo,
    verts_info: array_class.VertsInfo,
    fixed_verts_state: array_class.VertsState,
    free_verts_state: array_class.VertsState,
    positions_gt: qd.types.ndarray(),
    positions_measured: qd.types.ndarray(),
    output_gt: qd.types.ndarray(),
    output_measured: qd.types.ndarray(),
):
    total_n_probes = probe_positions_local.shape[0]
    n_batches = output_gt.shape[-1]

    func_update_all_verts(
        geoms_state, geoms_info, verts_info, free_verts_state, fixed_verts_state, static_rigid_sim_config
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

        max_r_gt = probe_radii[i_p]
        best_dist_sq_gt = max_r_gt * max_r_gt
        best_point_gt = probe_pos

        probe_radius_noise = probe_radii_noise[i_p]
        use_noised_radius = probe_radius_noise > gs.EPS
        max_r_m = max_r_gt
        if use_noised_radius:
            max_r_m = func_noised_probe_radius(max_r_gt, probe_radius_noise)
        best_dist_sq_m = max_r_m * max_r_m
        best_point_m = probe_pos

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
                    if dist_sq < best_dist_sq_gt:
                        best_dist_sq_gt = dist_sq
                        best_point_gt = closest
                    if use_noised_radius and dist_sq < best_dist_sq_m:
                        best_dist_sq_m = dist_sq
                        best_point_m = closest

        best_dist_gt = qd.sqrt(best_dist_sq_gt)
        best_dist_m = best_dist_gt
        if use_noised_radius:
            best_dist_m = qd.sqrt(best_dist_sq_m)
        else:
            for j in qd.static(range(3)):
                best_point_m[j] = best_point_gt[j]

        probe_idx_in_sensor = i_p - sensor_probe_start[i_s]
        cache_start = sensor_cache_start[i_s]
        probe_global_idx = sensor_probe_start[i_s] + probe_idx_in_sensor

        output_gt[cache_start + probe_idx_in_sensor, i_b] = best_dist_gt
        output_measured[cache_start + probe_idx_in_sensor, i_b] = best_dist_m
        for j in qd.static(range(3)):
            positions_gt[i_b, probe_global_idx, j] = best_point_gt[j]
            positions_measured[i_b, probe_global_idx, j] = best_point_m[j]


@dataclass
class SurfaceDistanceProbeSensorMetadataMixin(ProbeSensorMetadataMixin):
    """Shared metadata for surface distance probe sensors: tracked links and nearest-point buffer."""

    track_link_start: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    track_link_end: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    track_link_flat: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    nearest_positions: torch.Tensor = make_tensor_field((0, 0, 3))
    nearest_positions_measured: torch.Tensor = make_tensor_field((0, 0, 3))


@dataclass
class SurfaceDistanceProbeMetadata(
    SurfaceDistanceProbeSensorMetadataMixin, RigidSensorMetadataMixin, SimpleSensorMetadata
):
    """Shared metadata for the SurfaceDistanceProbe sensor class."""


class SurfaceDistanceProbeSensor(
    ProbeSensorMixin[SurfaceDistanceProbeMetadata],
    RigidSensorMixin[SurfaceDistanceProbeMetadata],
    SimpleSensor[SurfaceDistanceProbeOptions, SurfaceDistanceProbeMetadata, tuple],
):
    """Surface distance probe: distance and nearest point from probe positions to tracked mesh surfaces."""

    def __init__(self, sensor_options: SurfaceDistanceProbeOptions, sensor_idx: int, sensor_manager: "SensorManager"):
        super().__init__(sensor_options, sensor_idx, sensor_manager)
        self._debug_objects: list = []
        self._nearest_points_slice: slice | None = None

    def _get_return_format(self) -> tuple[tuple[int, ...], ...]:
        return (self._n_probes,)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    def build(self):
        super().build()

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

        self._shared_metadata.nearest_positions = torch.zeros(
            (self._manager._sim._B, self._shared_metadata.total_n_probes, 3), dtype=gs.tc_float, device=gs.device
        )
        self._shared_metadata.nearest_positions_measured = torch.zeros(
            (self._manager._sim._B, self._shared_metadata.total_n_probes, 3), dtype=gs.tc_float, device=gs.device
        )
        slice_start = self._shared_metadata.sensor_probe_start[self._idx]
        self._nearest_points_slice = slice(slice_start, slice_start + self._n_probes)

    @classmethod
    def reset(cls, shared_metadata: SurfaceDistanceProbeMetadata, shared_ground_truth_cache: torch.Tensor, envs_idx):
        super().reset(shared_metadata, shared_ground_truth_cache, envs_idx)
        # Pre-first-step placeholder. The kernel writes world-frame nearest points on each step; before that, an
        # uninitialized read returns zeros rather than misleading link-local positions.
        shared_metadata.nearest_positions[envs_idx] = 0.0
        shared_metadata.nearest_positions_measured[envs_idx] = 0.0

    @classmethod
    def _update_current_timestep_data(
        cls,
        shared_metadata: SurfaceDistanceProbeMetadata,
        current_ground_truth_data_T: torch.Tensor,
        ground_truth_data_timeline: "TensorRingBuffer | None",
        measured_data_timeline: "TensorRingBuffer",
    ):
        solver = shared_metadata.solver
        current_ground_truth_data_T.zero_()
        measured = measured_data_timeline.at(0, copy=False)
        measured.zero_()
        if shared_metadata.measured_scratch_T.shape != current_ground_truth_data_T.shape:
            shared_metadata.measured_scratch_T = torch.empty_like(current_ground_truth_data_T)
        measured_cols_b = shared_metadata.measured_scratch_T

        _kernel_surface_distance_probe(
            shared_metadata.probe_positions,
            shared_metadata.probe_radii,
            shared_metadata.probe_radii_noise,
            shared_metadata.probe_sensor_idx,
            shared_metadata.links_idx,
            shared_metadata.sensor_cache_start,
            shared_metadata.sensor_probe_start,
            shared_metadata.track_link_start,
            shared_metadata.track_link_end,
            shared_metadata.track_link_flat,
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
            shared_metadata.nearest_positions_measured,
            current_ground_truth_data_T,
            measured_cols_b,
        )
        if ground_truth_data_timeline is not None:
            ground_truth_data_timeline.at(0, copy=False).copy_(current_ground_truth_data_T.T)
        measured.copy_(measured_cols_b.T)

    def _draw_debug(self, context: "RasterizerContext"):
        env_idx = context.rendered_envs_idx[0] if self._manager._sim.n_envs > 0 else None
        for obj in self._debug_objects:
            context.clear_debug_object(obj)
        self._debug_objects.clear()

        link_pos = self._link.get_pos(env_idx).squeeze()
        link_quat = self._link.get_quat(env_idx).squeeze()
        probe_world = tensor_to_array(gu.transform_by_trans_quat(self._probe_local_pos, link_pos, link_quat))
        points = tensor_to_array(self.nearest_points[env_idx]).reshape(-1, 3)

        self._debug_objects.append(
            context.draw_debug_spheres(
                poss=np.concatenate([probe_world, points]),
                radius=self._options.debug_sphere_radius,
                color=self._options.debug_probe_color,
            )
        )
        for i in range(len(probe_world)):
            line_obj = context.draw_debug_line(
                probe_world[i],
                points[i],
                radius=self._options.debug_sphere_radius / 4.0,
                color=self._options.debug_probe_color,
            )
            self._debug_objects.append(line_obj)

    @property
    def nearest_points(self) -> torch.Tensor:
        """Nearest mesh points for the measured (noisy-radius) query, aligned with ``read()``."""
        return self._shared_metadata.nearest_positions_measured[..., self._nearest_points_slice, :]

    @property
    def nearest_points_ground_truth(self) -> torch.Tensor:
        """Nearest mesh points for the nominal-radius ground-truth query."""
        return self._shared_metadata.nearest_positions[..., self._nearest_points_slice, :]
