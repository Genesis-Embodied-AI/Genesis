from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, NamedTuple, TypeVar

import numpy as np
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
import genesis.utils.sdf as sdf
from genesis.options.sensors import ElastomerTaxel as ElastomerTaxelSensorOptions
from genesis.options.sensors import ProximityTaxel as ProximityTaxelOptions
from genesis.utils.misc import concat_with_tensor, make_tensor_field, tensor_to_array
from genesis.utils.point_cloud import sample_mesh_point_cloud

from .base_sensor import RigidSensorMetadataMixin, RigidSensorMixin, SimpleSensor, SimpleSensorMetadata
from .probe import (
    ProbeSensorMetadataMixin,
    ProbeSensorMixin,
    ProbesWithNormalSensorMetadataMixin,
    ProbesWithNormalSensorMixin,
    func_noised_probe_radius,
)

if TYPE_CHECKING:
    from genesis.options.sensors import SensorOptions
    from genesis.utils.ring_buffer import TensorRingBuffer
    from genesis.vis.rasterizer_context import RasterizerContext

    from .sensor_manager import SensorManager


def _get_mesh_geom_chunks(link, prefer_visual: bool) -> list[tuple[object, np.ndarray, np.ndarray]]:
    """Return per-geom mesh chunks in link-local frame."""
    if prefer_visual:
        geoms = list(link.vgeoms) if link.vgeoms else list(link.geoms)
        use_vverts = bool(link.vgeoms)
    else:
        geoms = list(link.geoms) if link.geoms else list(link.vgeoms)
        use_vverts = not bool(link.geoms) and bool(link.vgeoms)

    chunks: list[tuple[object, np.ndarray, np.ndarray]] = []
    for geom in geoms:
        if use_vverts:
            verts = np.asarray(geom.init_vverts, dtype=np.float32)
            faces = np.asarray(geom.init_vfaces, dtype=np.int32)
        else:
            verts = np.asarray(geom.init_verts, dtype=np.float32)
            faces = np.asarray(geom.init_faces, dtype=np.int32)
        if verts.size == 0 or faces.size == 0:
            continue
        verts_link = gu.transform_by_trans_quat(verts, geom.init_pos, geom.init_quat)
        chunks.append((geom, verts_link.astype(np.float32, copy=False), np.asarray(faces, dtype=np.int32)))
    return chunks


def _n_sample_points_per_link(n_sample_points: int | list | tuple, n_links: int) -> list[int]:
    if n_links <= 0:
        return []
    if isinstance(n_sample_points, (list, tuple)):
        counts = [int(x) for x in n_sample_points]
        if len(counts) != n_links:
            gs.raise_exception(
                f"Point cloud tactile n_sample_points length must match track_link_idx ({n_links}), got {len(counts)}."
            )
        if any(c < 0 for c in counts):
            gs.raise_exception("n_sample_points entries must be non-negative.")
        return counts
    n_total = int(n_sample_points)
    if n_total < 0:
        gs.raise_exception("n_sample_points must be non-negative.")
    base, rem = divmod(n_total, n_links)
    return [base + (1 if i < rem else 0) for i in range(n_links)]


def _mesh_area(verts: np.ndarray, faces: np.ndarray) -> float:
    tris = verts[faces]
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    return float(0.5 * np.linalg.norm(cross, axis=1).sum())


def _split_count_by_area(n_total: int, geom_chunks: list[tuple[object, np.ndarray, np.ndarray]]) -> list[int]:
    n_chunks = len(geom_chunks)
    if n_chunks <= 0:
        return []
    if n_total <= 0:
        return [0] * n_chunks

    areas = np.asarray([_mesh_area(verts, faces) for _, verts, faces in geom_chunks], dtype=np.float64)
    if float(areas.sum()) <= gs.EPS:
        areas.fill(1.0)

    if n_total < n_chunks:
        counts = np.zeros(n_chunks, dtype=np.int64)
        counts[np.argsort(-areas)[:n_total]] = 1
        return counts.tolist()

    raw_extra = (n_total - n_chunks) * areas / float(areas.sum())
    extra = np.floor(raw_extra).astype(np.int64)
    remainder = n_total - n_chunks - int(extra.sum())
    if remainder > 0:
        extra[np.argsort(-(raw_extra - extra))[:remainder]] += 1
    return (extra + 1).tolist()


def _active_envs_mask_tensor(geom, batch_size: int) -> torch.Tensor:
    if geom.active_envs_mask is None:
        return torch.ones((batch_size,), dtype=gs.tc_bool, device=gs.device)
    return geom.active_envs_mask.to(device=gs.device, dtype=gs.tc_bool)


def _sample_track_links_point_cloud_tensors(
    solver, track_link_idx: np.ndarray, n_sample_points: int | list | tuple, prefer_visual: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    FPS-sample meshes on ``track_link_idx`` into concatenated link-local positions and normals.

    Returns
    -------
    idx_cat, pos_cat, nrm_cat, active_cat
        Global link index per row, positions (N, 3), normals (N, 3), and active env mask (N, B), all on ``gs.device``.
    """
    n_per_link = _n_sample_points_per_link(n_sample_points, int(track_link_idx.shape[0]))
    if sum(n_per_link) == 0:
        gs.raise_exception("n_sample_points must allocate at least one sample in total.")

    link_idx_chunks: list[torch.Tensor] = []
    pos_chunks: list[torch.Tensor] = []
    nrm_chunks: list[torch.Tensor] = []
    active_chunks: list[torch.Tensor] = []

    for i_l in range(int(track_link_idx.shape[0])):
        n_pts = n_per_link[i_l]
        link_idx = int(track_link_idx[i_l])
        link = solver.links[link_idx]
        geom_chunks = _get_mesh_geom_chunks(link, prefer_visual)
        if not geom_chunks:
            gs.raise_exception(f"No mesh geometry on tracked link index {link_idx}.")
        for n_geom_pts, (geom, verts, faces) in zip(_split_count_by_area(n_pts, geom_chunks), geom_chunks):
            if n_geom_pts <= 0:
                continue
            # Fixed seed: the cache key already discriminates between meshes (vertices+faces hashed), so the same mesh
            # always resolves to the same sample, which keeps tactile readings reproducible across build/reset cycles.
            pts_np, nrm_np = sample_mesh_point_cloud(
                verts, faces, n_geom_pts, seed=0, use_cache=True, return_normals=True
            )

            li = torch.full((pts_np.shape[0],), link_idx, dtype=gs.tc_int, device=gs.device)
            link_idx_chunks.append(li)
            pos_chunks.append(torch.tensor(pts_np, dtype=gs.tc_float, device=gs.device))
            nrm_chunks.append(torch.tensor(nrm_np, dtype=gs.tc_float, device=gs.device))
            active_chunks.append(_active_envs_mask_tensor(geom, solver._B).expand(pts_np.shape[0], solver._B))

    if not pos_chunks:
        gs.raise_exception("PointCloudTactile sensor produced an empty object point cloud.")

    return (
        torch.cat(link_idx_chunks, dim=0),
        torch.cat(pos_chunks, dim=0),
        torch.cat(nrm_chunks, dim=0),
        torch.cat(active_chunks, dim=0),
    )


@qd.func
def _func_vec3_at(values: qd.types.ndarray(), i: int):  # -> gs.qd_vec3:
    return qd.Vector([values[i, 0], values[i, 1], values[i, 2]], dt=float)


@qd.kernel
def _kernel_point_cloud_proximity_taxel(
    probe_positions_local: qd.types.ndarray(),
    probe_local_normal: qd.types.ndarray(),
    links_idx: qd.types.ndarray(),
    sensor_cache_start: qd.types.ndarray(),
    sensor_probe_start: qd.types.ndarray(),
    n_probes_per_sensor: qd.types.ndarray(),
    sensor_pc_start: qd.types.ndarray(),
    sensor_pc_n: qd.types.ndarray(),
    pc_link_idx: qd.types.ndarray(),
    pc_pos_link: qd.types.ndarray(),
    pc_active_envs_mask: qd.types.ndarray(),
    probe_radii: qd.types.ndarray(),
    probe_radii_noise: qd.types.ndarray(),
    stiffness: qd.types.ndarray(),
    shear_coupling: qd.types.ndarray(),
    proximity_density_scale: qd.types.ndarray(),
    links_state: array_class.LinksState,
    eps: float,
    output_gt: qd.types.ndarray(),
    output_measured: qd.types.ndarray(),
    taxel_signal_buf: qd.types.ndarray(),
):
    for i_b, i_s in qd.ndrange(output_gt.shape[-1], sensor_pc_start.shape[0]):
        sensor_link_idx = links_idx[i_s]
        s_pos = links_state.pos[sensor_link_idx, i_b]
        s_quat = links_state.quat[sensor_link_idx, i_b]

        k_stiff = stiffness[i_s]
        k_shear = shear_coupling[i_s]
        dens = proximity_density_scale[i_s, i_b]
        n_probes = n_probes_per_sensor[i_s]
        cache_start = sensor_cache_start[i_s]

        pc_start = sensor_pc_start[i_s]
        pc_end = pc_start + sensor_pc_n[i_s]

        s_vel = links_state.cd_vel[sensor_link_idx, i_b]
        s_ang = links_state.cd_ang[sensor_link_idx, i_b]
        s_com = links_state.root_COM[sensor_link_idx, i_b]

        for _i_p in range(n_probes_per_sensor[i_s]):
            i_p = sensor_probe_start[i_s] + _i_p
            probe_local = _func_vec3_at(probe_positions_local, i_p)
            probe_world = s_pos + gu.qd_transform_by_quat(probe_local, s_quat)

            a_loc = _func_vec3_at(probe_local_normal, i_p)
            a_w = gu.qd_transform_by_quat(a_loc, s_quat)
            a_norm = qd.sqrt(a_w.dot(a_w)) + eps
            for j in qd.static(range(3)):
                a_w[j] = a_w[j] / a_norm

            R_gt = probe_radii[i_p]
            R_gt_sq = R_gt * R_gt
            probe_radius_noise = probe_radii_noise[i_p]
            use_noised_radius = probe_radius_noise > eps
            R_m = R_gt
            if use_noised_radius:
                R_m = func_noised_probe_radius(R_gt, probe_radius_noise)
            R_m_sq = R_m * R_m

            v_tax = s_vel + s_ang.cross(probe_world - s_com)

            sum_p_gt = gs.qd_float(0.0)
            fv_gt = qd.Vector.zero(gs.qd_float, 3)
            tau_w_gt = qd.Vector.zero(gs.qd_float, 3)
            sum_p_m = gs.qd_float(0.0)
            fv_m = qd.Vector.zero(gs.qd_float, 3)
            tau_w_m = qd.Vector.zero(gs.qd_float, 3)

            for i_o in range(pc_start, pc_end):
                if not pc_active_envs_mask[i_o, i_b]:
                    continue
                i_l = pc_link_idx[i_o]
                lp = links_state.pos[i_l, i_b]
                lq = links_state.quat[i_l, i_b]
                pos_l = _func_vec3_at(pc_pos_link, i_o)
                pw = lp + gu.qd_transform_by_quat(pos_l, lq)

                dvec = pw - probe_world
                dsq = dvec.dot(dvec)
                dist = qd.sqrt(dsq)

                if dsq <= R_gt_sq and dist > eps:
                    P_i_gt = R_gt - dist
                    if P_i_gt > 0.0:
                        rcom_o = links_state.root_COM[i_l, i_b]
                        cdv_o = links_state.cd_vel[i_l, i_b]
                        cda_o = links_state.cd_ang[i_l, i_b]
                        v_pc = cdv_o + cda_o.cross(pw - rcom_o)
                        v_rel = v_pc - v_tax
                        vdota = v_rel.dot(a_w)
                        v_t = qd.Vector.zero(gs.qd_float, 3)
                        for j in qd.static(range(3)):
                            v_t[j] = v_rel[j] - a_w[j] * vdota

                        sum_p_gt = sum_p_gt + P_i_gt
                        for j in qd.static(range(3)):
                            fv_gt[j] = fv_gt[j] + P_i_gt * v_t[j]

                        ctmp = dvec.cross(a_w)
                        for j in qd.static(range(3)):
                            tau_w_gt[j] = tau_w_gt[j] + P_i_gt * ctmp[j]

                if use_noised_radius and dsq <= R_m_sq and dist > eps:
                    P_i_m = R_m - dist
                    if P_i_m > 0.0:
                        rcom_o = links_state.root_COM[i_l, i_b]
                        cdv_o = links_state.cd_vel[i_l, i_b]
                        cda_o = links_state.cd_ang[i_l, i_b]
                        v_pc = cdv_o + cda_o.cross(pw - rcom_o)
                        v_rel = v_pc - v_tax
                        vdota = v_rel.dot(a_w)
                        v_t = qd.Vector.zero(gs.qd_float, 3)
                        for j in qd.static(range(3)):
                            v_t[j] = v_rel[j] - a_w[j] * vdota

                        sum_p_m = sum_p_m + P_i_m
                        for j in qd.static(range(3)):
                            fv_m[j] = fv_m[j] + P_i_m * v_t[j]

                        ctmp = dvec.cross(a_w)
                        for j in qd.static(range(3)):
                            tau_w_m[j] = tau_w_m[j] + P_i_m * ctmp[j]

            if not use_noised_radius:
                sum_p_m = sum_p_gt
                for j in qd.static(range(3)):
                    fv_m[j] = fv_gt[j]
                    tau_w_m[j] = tau_w_gt[j]

            taxel_signal_buf[i_p, i_b] = sum_p_m

            f_w_gt = qd.Vector.zero(gs.qd_float, 3)
            for j in qd.static(range(3)):
                f_w_gt[j] = k_stiff * dens * sum_p_gt * a_w[j]
            if k_shear > eps:
                for j in qd.static(range(3)):
                    f_w_gt[j] = f_w_gt[j] + k_shear * dens * fv_gt[j]

            tau_scaled_gt = qd.Vector.zero(gs.qd_float, 3)
            for j in qd.static(range(3)):
                tau_scaled_gt[j] = k_stiff * dens * tau_w_gt[j]

            f_l_gt = gu.qd_inv_transform_by_quat(f_w_gt, s_quat)
            t_l_gt = gu.qd_inv_transform_by_quat(tau_scaled_gt, s_quat)

            f_w_m = qd.Vector.zero(gs.qd_float, 3)
            for j in qd.static(range(3)):
                f_w_m[j] = k_stiff * dens * sum_p_m * a_w[j]
            if k_shear > eps:
                for j in qd.static(range(3)):
                    f_w_m[j] = f_w_m[j] + k_shear * dens * fv_m[j]

            tau_scaled_m = qd.Vector.zero(gs.qd_float, 3)
            for j in qd.static(range(3)):
                tau_scaled_m[j] = k_stiff * dens * tau_w_m[j]

            f_l_m = gu.qd_inv_transform_by_quat(f_w_m, s_quat)
            t_l_m = gu.qd_inv_transform_by_quat(tau_scaled_m, s_quat)

            force_start = cache_start + _i_p * 3
            torque_start = cache_start + n_probes * 3 + _i_p * 3
            for j in qd.static(range(3)):
                output_gt[force_start + j, i_b] = f_l_gt[j]
            for j in qd.static(range(3)):
                output_gt[torque_start + j, i_b] = t_l_gt[j]
            for j in qd.static(range(3)):
                output_measured[force_start + j, i_b] = f_l_m[j]
            for j in qd.static(range(3)):
                output_measured[torque_start + j, i_b] = t_l_m[j]


@dataclass
class PointCloudTactileSharedMetadata(ProbeSensorMetadataMixin, RigidSensorMetadataMixin, SimpleSensorMetadata):
    """Shared sensor-manager state for point-cloud–tracked tactile sensors (probes + merged track PC)."""

    pc_link_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    pc_pos_link: torch.Tensor = make_tensor_field((0, 3))
    pc_normal_link: torch.Tensor = make_tensor_field((0, 3))
    pc_active_envs_mask: torch.Tensor = make_tensor_field((0, 0), dtype_factory=lambda: gs.tc_bool)
    sensor_pc_start: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    sensor_pc_n: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)


PointCloudTactileSensorMetadataMixinT = TypeVar(
    "PointCloudTactileSensorMetadataMixinT", bound=PointCloudTactileSharedMetadata
)


class PointCloudTactileSensorMixin(ProbeSensorMixin[PointCloudTactileSensorMetadataMixinT]):
    def __init__(self, sensor_options: "SensorOptions", sensor_idx: int, sensor_manager: "SensorManager"):
        super().__init__(sensor_options, sensor_idx, sensor_manager)
        self._debug_objects: list = []
        self._probe_start_idx = -1
        self._debug_pc_chunks: list[tuple[int, torch.Tensor, torch.Tensor]] | None = None

    def build(self):
        super().build()
        self._probe_start_idx = int(self._shared_metadata.sensor_probe_start[self._idx].item())

        pc_start_row = self._shared_metadata.pc_pos_link.shape[0]
        idx_cat, pos_cat, nrm_cat, active_cat = _sample_track_links_point_cloud_tensors(
            self._shared_metadata.solver,
            np.asarray(self._options.track_link_idx, dtype=gs.np_int),
            self._options.n_sample_points,
            self._options.use_visual_mesh,
        )
        if self._options.draw_debug:
            self._debug_pc_chunks = []
            for lid in torch.unique(idx_cat):
                mask = idx_cat == lid
                self._debug_pc_chunks.append((int(lid.item()), pos_cat[mask].clone(), active_cat[mask].clone()))
        else:
            self._debug_pc_chunks = None

        self._shared_metadata.pc_link_idx = concat_with_tensor(
            self._shared_metadata.pc_link_idx, idx_cat, expand=(idx_cat.shape[0],)
        )
        self._shared_metadata.pc_pos_link = concat_with_tensor(
            self._shared_metadata.pc_pos_link, pos_cat, expand=(pos_cat.shape[0], 3)
        )
        self._shared_metadata.pc_normal_link = concat_with_tensor(
            self._shared_metadata.pc_normal_link, nrm_cat, expand=(nrm_cat.shape[0], 3)
        )
        self._shared_metadata.pc_active_envs_mask = concat_with_tensor(
            self._shared_metadata.pc_active_envs_mask, active_cat
        )
        self._shared_metadata.sensor_pc_start = concat_with_tensor(
            self._shared_metadata.sensor_pc_start, pc_start_row, expand=(1,)
        )
        self._shared_metadata.sensor_pc_n = concat_with_tensor(
            self._shared_metadata.sensor_pc_n, self._shared_metadata.pc_pos_link.shape[0] - pc_start_row, expand=(1,)
        )

    def _draw_debug_probes(
        self, context: "RasterizerContext", get_magnitude_1d: Callable[[list[int] | None], np.ndarray]
    ) -> None:
        for obj in self._debug_objects:
            context.clear_debug_object(obj)
        self._debug_objects.clear()

        envs_idx, n_debug_envs, env_offsets, probe_world = self._compute_probes_world_pos(context)

        magnitude = get_magnitude_1d(envs_idx).reshape(-1)
        for is_contact in (False, True):
            (probes_idx,) = np.nonzero(magnitude >= gs.EPS if is_contact else magnitude < gs.EPS)
            if probes_idx.size == 0:
                continue
            spheres_obj = context.draw_debug_spheres(
                poss=probe_world[probes_idx],
                radius=self._shared_metadata.probe_radii[self._probe_start_idx].item(),
                color=self._options.debug_contact_color if is_contact else self._options.debug_probe_color,
            )
            self._debug_objects.append(spheres_obj)

        if self._debug_pc_chunks is None:
            return
        world_chunks: list[np.ndarray] = []
        for link_idx, pos_local, active_envs_mask in self._debug_pc_chunks:
            trk_link = self._shared_metadata.solver.links[link_idx]
            if envs_idx is not None:
                active_mask = tensor_to_array(active_envs_mask[:, envs_idx].T).astype(bool)
                if not active_mask.any():
                    continue
                trk_pos = trk_link.get_pos(envs_idx)[:, None, :]
                trk_quat = trk_link.get_quat(envs_idx)[:, None, :]
                pc_world = gu.transform_by_trans_quat(pos_local[None, :, :], trk_pos, trk_quat)
                pc_world = tensor_to_array(pc_world) + env_offsets[:, None, :]
                world_chunks.append(pc_world[active_mask])
            else:
                active_mask = active_envs_mask[:, 0]
                pos_active = pos_local[active_mask]
                if pos_active.numel() == 0:
                    continue
                trk_pos = trk_link.get_pos(envs_idx).reshape(3)
                trk_quat = trk_link.get_quat(envs_idx).reshape(4)
                world_chunks.append(tensor_to_array(gu.transform_by_trans_quat(pos_active, trk_pos, trk_quat)))
        if not world_chunks:
            return
        pc_world = np.concatenate(world_chunks, axis=0)
        pc_obj = context.draw_debug_spheres(
            poss=pc_world,
            radius=float(self._options.debug_point_cloud_radius),
            color=self._options.debug_point_cloud_color,
        )
        self._debug_objects.append(pc_obj)

    def _debug_probe_buffer_magnitudes(self, buffer: torch.Tensor, envs_idx: list[int] | None) -> np.ndarray:
        values = buffer[self._probe_start_idx : self._probe_start_idx + self._n_probes]
        if envs_idx is None:
            return tensor_to_array(values[:, 0])
        return tensor_to_array(values[:, envs_idx].T)


class ProximityTaxelData(NamedTuple):
    """Per-taxel estimates in link-local frame."""

    force: torch.Tensor
    torque: torch.Tensor


@dataclass
class ProximityTaxelMetadata(PointCloudTactileSharedMetadata, ProbesWithNormalSensorMetadataMixin):
    stiffness: torch.Tensor = make_tensor_field((0,))
    shear_coupling: torch.Tensor = make_tensor_field((0,))
    proximity_density_scale: torch.Tensor = make_tensor_field((0, 0))
    taxel_signal_buf: torch.Tensor = make_tensor_field((0, 0))


class ProximityTaxelSensor(
    PointCloudTactileSensorMixin[ProximityTaxelMetadata],
    ProbesWithNormalSensorMixin[ProximityTaxelMetadata],
    RigidSensorMixin[ProximityTaxelMetadata],
    SimpleSensor[ProximityTaxelOptions, ProximityTaxelMetadata, ProximityTaxelData],
):
    """Spherical point-cloud taxels: per-taxel force and torque in link-local frame vs tracked meshes."""

    def build(self):
        super().build()
        self._shared_metadata.stiffness = concat_with_tensor(
            self._shared_metadata.stiffness, float(self._options.stiffness), expand=(1,)
        )
        self._shared_metadata.shear_coupling = concat_with_tensor(
            self._shared_metadata.shear_coupling, float(self._options.shear_coupling), expand=(1,)
        )
        pc_start = self._shared_metadata.sensor_pc_start[-1].item()
        pc_end = pc_start + self._shared_metadata.sensor_pc_n[-1].item()
        active_count = (
            self._shared_metadata.pc_active_envs_mask[pc_start:pc_end].sum(dim=0).clamp_min(1).to(dtype=gs.tc_float)
        )
        self._shared_metadata.proximity_density_scale = concat_with_tensor(
            self._shared_metadata.proximity_density_scale,
            self._options.density_scalar / active_count,
            expand=(1, self._manager._sim._B),
        )
        self._shared_metadata.taxel_signal_buf = torch.zeros(
            (self._shared_metadata.total_n_probes, self._manager._sim._B), dtype=gs.tc_float, device=gs.device
        )

    def _get_return_format(self) -> tuple[tuple[int, ...], ...]:
        return ((self._n_probes, 3), (self._n_probes, 3))

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def reset(cls, shared_metadata: ProximityTaxelMetadata, shared_ground_truth_cache: torch.Tensor, envs_idx):
        super().reset(shared_metadata, shared_ground_truth_cache, envs_idx)
        shared_metadata.taxel_signal_buf[:, envs_idx] = 0.0

    @classmethod
    def _update_current_timestep_data(
        cls,
        shared_metadata: ProximityTaxelMetadata,
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

        _kernel_point_cloud_proximity_taxel(
            shared_metadata.probe_positions,
            shared_metadata.probe_local_normal,
            shared_metadata.links_idx,
            shared_metadata.sensor_cache_start,
            shared_metadata.sensor_probe_start,
            shared_metadata.n_probes_per_sensor,
            shared_metadata.sensor_pc_start,
            shared_metadata.sensor_pc_n,
            shared_metadata.pc_link_idx,
            shared_metadata.pc_pos_link,
            shared_metadata.pc_active_envs_mask,
            shared_metadata.probe_radii,
            shared_metadata.probe_radii_noise,
            shared_metadata.stiffness,
            shared_metadata.shear_coupling,
            shared_metadata.proximity_density_scale,
            solver.links_state,
            gs.EPS,
            current_ground_truth_data_T,
            measured_cols_b,
            shared_metadata.taxel_signal_buf,
        )
        if ground_truth_data_timeline is not None:
            ground_truth_data_timeline.at(0, copy=False).copy_(current_ground_truth_data_T.T)
        measured.copy_(measured_cols_b.T)

    def _draw_debug(self, context: "RasterizerContext"):
        self._draw_debug_probes(
            context,
            lambda envs_idx: self._debug_probe_buffer_magnitudes(self._shared_metadata.taxel_signal_buf, envs_idx),
        )


_GRID_TOL = 1.0e-5


def _next_pow2(n: int) -> int:
    """Smallest power of 2 >= n (1 if n==0)."""
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p *= 2
    return p


def _expand_probe_normals(normals: np.ndarray, n_probes: int, probe_shape: tuple[int, ...]) -> np.ndarray:
    normals = np.asarray(normals, dtype=gs.np_float)
    if normals.ndim == 1:
        return np.broadcast_to(normals, (n_probes, 3)).copy()
    if normals.shape == (*probe_shape, 3):
        return normals.reshape(n_probes, 3).copy()
    if normals.shape == (n_probes, 3):
        return normals.copy()
    gs.raise_exception(
        "ElastomerTaxel probe_local_normal must be one normal or match probe_local_pos shape. "
        f"Got normal shape {normals.shape} for probe shape {probe_shape}."
    )


def _normalize_elastomer_probe_layout(
    probe_pos: np.ndarray, probe_normals: np.ndarray, is_grid: bool
) -> tuple[np.ndarray, np.ndarray, bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probe_shape = probe_pos.shape[:-1]
    flat = probe_pos.reshape(-1, 3)
    normals = _expand_probe_normals(probe_normals, flat.shape[0], probe_shape)

    normal_norms = np.linalg.norm(normals, axis=1)
    if np.any(normal_norms < gs.EPS):
        gs.raise_exception("ElastomerTaxel probe_local_normal entries must be non-zero.")
    normals = normals / normal_norms[:, None]

    use_grid_fft = False
    grid_normal = np.zeros(3, dtype=gs.np_float)
    tangent_u = np.zeros(3, dtype=gs.np_float)
    tangent_v = np.zeros(3, dtype=gs.np_float)
    grid_spacing = np.zeros(2, dtype=gs.np_float)

    if is_grid:
        if len(probe_shape) != 2:
            gs.raise_exception("ElastomerTaxel grid probe_local_pos must have shape (ny, nx, 3).")
        ny, nx = int(probe_shape[0]), int(probe_shape[1])
        if nx >= 2 and ny >= 2:
            grid = probe_pos.reshape(ny, nx, 3)
            step_u = grid[0, 1] - grid[0, 0]
            step_v = grid[1, 0] - grid[0, 0]
            spacing_u = float(np.linalg.norm(step_u))
            spacing_v = float(np.linalg.norm(step_v))
            if spacing_u >= gs.EPS and spacing_v >= gs.EPS:
                tangent_u_candidate = (step_u / spacing_u).astype(gs.np_float)
                tangent_v_candidate = (step_v / spacing_v).astype(gs.np_float)
                normal_candidate = normals[0].astype(gs.np_float, copy=False)
                normals_are_uniform = bool(np.all(normals @ normal_candidate >= 1.0 - _GRID_TOL))
                axes_are_orthogonal = abs(float(tangent_u_candidate @ tangent_v_candidate)) <= _GRID_TOL
                axes_in_plane = (
                    abs(float(tangent_u_candidate @ normal_candidate)) <= _GRID_TOL
                    and abs(float(tangent_v_candidate @ normal_candidate)) <= _GRID_TOL
                )
                expected = (
                    grid[0, 0]
                    + np.arange(nx, dtype=gs.np_float)[None, :, None] * step_u[None, None, :]
                    + np.arange(ny, dtype=gs.np_float)[:, None, None] * step_v[None, None, :]
                )
                is_regular = bool(np.max(np.linalg.norm(grid - expected, axis=-1)) <= _GRID_TOL)
                use_grid_fft = normals_are_uniform and axes_are_orthogonal and axes_in_plane and is_regular
                if use_grid_fft:
                    grid_normal = normal_candidate
                    tangent_u = tangent_u_candidate
                    tangent_v = tangent_v_candidate
                    grid_spacing = np.array((spacing_u, spacing_v), dtype=gs.np_float)

    return (
        flat.astype(gs.np_float, copy=False),
        normals.astype(gs.np_float, copy=False),
        use_grid_fft,
        grid_normal.astype(gs.np_float, copy=False),
        tangent_u.astype(gs.np_float, copy=False),
        tangent_v.astype(gs.np_float, copy=False),
        grid_spacing.astype(gs.np_float, copy=False),
    )


@qd.func
def _func_elastomer_min_sdf_over_active_geoms(
    i_b: int,
    point_world,  # : gs.qd_vec3,
    geom_start: int,
    geom_n: int,
    geom_idx: qd.types.ndarray(),
    geom_active_envs_mask: qd.types.ndarray(),
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    sdf_info: array_class.SDFInfo,
) -> float:
    min_sdf = float(1.0e6)
    geom_end = geom_start + geom_n
    for i_gm in range(geom_start, geom_end):
        if not geom_active_envs_mask[i_gm, i_b]:
            continue
        i_g = geom_idx[i_gm]
        sd = sdf.sdf_func_world(geoms_state, geoms_info, sdf_info, point_world, i_g, i_b)
        if sd < min_sdf:
            min_sdf = sd
    return min_sdf


@qd.func
def _func_elastomer_tangent(
    vec,  # : gs.qd_vec3,
    normal,  # : gs.qd_vec3
):  # -> gs.qd_vec3:
    return vec - normal * vec.dot(normal)


@qd.func
def _func_elastomer_clear_surface_state(
    i_b: int,
    i_o: int,
    surface_pos_sensor_buf: qd.types.ndarray(),
    surface_entry_pos_sensor_buf: qd.types.ndarray(),
    surface_depth_buf: qd.types.ndarray(),
    surface_initialized_buf: qd.types.ndarray(),
):
    surface_depth_buf[i_b, i_o] = 0.0
    surface_initialized_buf[i_b, i_o] = False
    for k in qd.static(range(3)):
        surface_pos_sensor_buf[i_b, i_o, k] = 0.0
        surface_entry_pos_sensor_buf[i_b, i_o, k] = 0.0


@qd.func
def _func_elastomer_update_surface_anchor(
    i_b: int,
    i_o: int,
    sdf_value: float,
    point_sensor,  # : gs.qd_vec3,
    sdf_enter: float,
    sdf_exit: float,
    surface_entry_pos_sensor_buf: qd.types.ndarray(),
    surface_initialized_buf: qd.types.ndarray(),
):
    if sdf_value > sdf_exit:
        surface_initialized_buf[i_b, i_o] = False
        for k in qd.static(range(3)):
            surface_entry_pos_sensor_buf[i_b, i_o, k] = 0.0
    elif (not surface_initialized_buf[i_b, i_o]) and sdf_value < -sdf_enter:
        surface_initialized_buf[i_b, i_o] = True
        for k in qd.static(range(3)):
            surface_entry_pos_sensor_buf[i_b, i_o, k] = point_sensor[k]


@qd.func
def _func_elastomer_direct_dilate_contribution(
    source_pos,  # : gs.qd_vec3,
    source_normal,  # : gs.qd_vec3,
    target_pos,  # : gs.qd_vec3,
    target_normal,  # : gs.qd_vec3,
    depth: float,
    lam: float,
    scale: float,
):  # -> gs.qd_vec3:
    source_contact_pos = source_pos - source_normal * depth
    diff = target_pos - source_contact_pos
    planar_diff = _func_elastomer_tangent(diff, target_normal)
    return diff * depth * qd.exp(-lam * planar_diff.dot(planar_diff)) * scale


@qd.func
def _func_elastomer_direct_shear_contribution(
    point_sensor,  # : gs.qd_vec3,
    entry_sensor,  # : gs.qd_vec3,
    probe_pos,  # : gs.qd_vec3,
    probe_normal,  # : gs.qd_vec3,
    depth: float,
    lam: float,
    scale: float,
    eps: float,
):  # -> gs.qd_vec3:
    shear_disp = point_sensor - entry_sensor
    shear_tangent = _func_elastomer_tangent(shear_disp, probe_normal)
    contribution = qd.Vector.zero(gs.qd_float, 3)
    if shear_tangent.dot(shear_tangent) > eps * eps:
        diff = probe_pos - point_sensor
        planar_diff = _func_elastomer_tangent(diff, probe_normal)
        contribution = shear_tangent * (depth * qd.exp(-lam * planar_diff.dot(planar_diff)) * scale)
    return contribution


def _collect_collision_geom_idx(solver, track_link_idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    geom_idx: list[int] = []
    active_masks: list[torch.Tensor] = []
    for link_idx in track_link_idx:
        link_i = int(link_idx)
        if link_i < 0 or link_i >= len(solver.links):
            gs.raise_exception(f"ElastomerTaxel track_link_idx contains invalid global link index {link_i}.")
        link = solver.links[link_i]
        for geom in link.geoms:
            geom_idx.append(int(geom.idx))
            active_masks.append(_active_envs_mask_tensor(geom, solver._B))
    if not geom_idx:
        gs.raise_exception("ElastomerTaxel tracked links must have collision geometry for SDF queries.")
    return torch.tensor(geom_idx, dtype=gs.tc_int, device=gs.device), torch.stack(active_masks, dim=0)


@torch.jit.script
def _precompute_hydroshear_dilate_kernel_fft(
    lambda_d: float, grid_spacing: tuple[float, float], fft_n: tuple[int, int], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    i = torch.arange(fft_n[0], dtype=dtype, device=device)
    j = torch.arange(fft_n[1], dtype=dtype, device=device)
    xx, yy = torch.meshgrid((i - fft_n[0] // 2) * grid_spacing[0], (j - fft_n[1] // 2) * grid_spacing[1], indexing="ij")
    g = torch.exp(torch.tensor(-lambda_d, dtype=dtype, device=device) * (xx * xx + yy * yy))
    k = torch.stack((xx * g, yy * g, g), dim=0)
    k = torch.fft.ifftshift(k, dim=(-2, -1))
    return torch.fft.fft2(k)


@qd.kernel(fastcache=True)
def _kernel_elastomer_probe_depth_and_direct_dilate(
    use_grid_fft: qd.types.ndarray(),
    probe_positions_local: qd.types.ndarray(),
    probe_local_normal: qd.types.ndarray(),
    probe_sensor_idx: qd.types.ndarray(),
    links_idx: qd.types.ndarray(),
    sensor_cache_start: qd.types.ndarray(),
    sensor_probe_start: qd.types.ndarray(),
    n_probes_per_sensor: qd.types.ndarray(),
    sensor_track_geom_start: qd.types.ndarray(),
    sensor_track_geom_n: qd.types.ndarray(),
    track_geom_idx: qd.types.ndarray(),
    track_geom_active_envs_mask: qd.types.ndarray(),
    lambda_d: qd.types.ndarray(),
    dilate_scale: qd.types.ndarray(),
    links_state: array_class.LinksState,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    sdf_info: array_class.SDFInfo,
    probe_depth_buf: qd.types.ndarray(),
    output: qd.types.ndarray(),
):
    total_n_probes = probe_positions_local.shape[0]
    n_batches = probe_depth_buf.shape[0]

    for i_b, i_p in qd.ndrange(n_batches, total_n_probes):
        i_s = probe_sensor_idx[i_p]
        sensor_link_idx = links_idx[i_s]
        link_pos = links_state.pos[sensor_link_idx, i_b]
        link_quat = links_state.quat[sensor_link_idx, i_b]
        probe_local = _func_vec3_at(probe_positions_local, i_p)
        probe_world = link_pos + gu.qd_transform_by_quat(probe_local, link_quat)

        min_sdf = _func_elastomer_min_sdf_over_active_geoms(
            i_b,
            probe_world,
            sensor_track_geom_start[i_s],
            sensor_track_geom_n[i_s],
            track_geom_idx,
            track_geom_active_envs_mask,
            geoms_state,
            geoms_info,
            sdf_info,
        )

        depth = qd.max(gs.qd_float(0.0), -min_sdf)
        probe_depth_buf[i_b, i_p] = depth

        if use_grid_fft[i_s] or depth <= gs.qd_float(0.0):
            continue

        probe_start = sensor_probe_start[i_s]
        n_probes = n_probes_per_sensor[i_s]
        lam = lambda_d[i_s]
        scale = dilate_scale[i_s]
        cache_start = sensor_cache_start[i_s]
        source_normal = _func_vec3_at(probe_local_normal, i_p)
        for j in range(n_probes):
            j_p = probe_start + j
            contribution = _func_elastomer_direct_dilate_contribution(
                probe_local,
                source_normal,
                _func_vec3_at(probe_positions_local, j_p),
                _func_vec3_at(probe_local_normal, j_p),
                depth,
                lam,
                scale,
            )
            for k in qd.static(range(3)):
                qd.atomic_add(output[cache_start + j * 3 + k, i_b], contribution[k])


@qd.kernel(fastcache=True)
def _kernel_elastomer_surface_state_and_direct_shear(
    probe_positions_local: qd.types.ndarray(),
    probe_local_normal: qd.types.ndarray(),
    sensor_cache_start: qd.types.ndarray(),
    sensor_probe_start: qd.types.ndarray(),
    n_probes_per_sensor: qd.types.ndarray(),
    links_idx: qd.types.ndarray(),
    sensor_elastomer_geom_start: qd.types.ndarray(),
    sensor_elastomer_geom_n: qd.types.ndarray(),
    elastomer_geom_idx: qd.types.ndarray(),
    elastomer_geom_active_envs_mask: qd.types.ndarray(),
    sensor_pc_start: qd.types.ndarray(),
    sensor_pc_n: qd.types.ndarray(),
    pc_link_idx: qd.types.ndarray(),
    pc_pos_link: qd.types.ndarray(),
    pc_active_envs_mask: qd.types.ndarray(),
    lambda_s: qd.types.ndarray(),
    shear_scale: qd.types.ndarray(),
    links_state: array_class.LinksState,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    sdf_info: array_class.SDFInfo,
    eps: float,
    sdf_enter: qd.types.ndarray(),
    sdf_exit: qd.types.ndarray(),
    surface_pos_sensor_buf: qd.types.ndarray(),
    surface_entry_pos_sensor_buf: qd.types.ndarray(),
    surface_depth_buf: qd.types.ndarray(),
    surface_initialized_buf: qd.types.ndarray(),
    output: qd.types.ndarray(),
):
    """
    HydroShear-style shear: while an indenter surface sample is inside the elastomer SDF (with hysteresis), anchor its
    sensor-frame position at first penetration and spread tangential (current - anchor) to probes.
    """
    n_batches = surface_pos_sensor_buf.shape[0]
    n_sensors = sensor_pc_start.shape[0]

    for i_b, i_s in qd.ndrange(n_batches, n_sensors):
        pc_start = sensor_pc_start[i_s]
        pc_end = pc_start + sensor_pc_n[i_s]
        for i_o in range(pc_start, pc_end):
            if not pc_active_envs_mask[i_o, i_b]:
                _func_elastomer_clear_surface_state(
                    i_b,
                    i_o,
                    surface_pos_sensor_buf,
                    surface_entry_pos_sensor_buf,
                    surface_depth_buf,
                    surface_initialized_buf,
                )
                continue

            sensor_link_idx = links_idx[i_s]
            track_link_idx = pc_link_idx[i_o]

            track_pos = links_state.pos[track_link_idx, i_b]
            track_quat = links_state.quat[track_link_idx, i_b]
            point_link = _func_vec3_at(pc_pos_link, i_o)
            point_world = track_pos + gu.qd_transform_by_quat(point_link, track_quat)

            sensor_pos = links_state.pos[sensor_link_idx, i_b]
            sensor_quat = links_state.quat[sensor_link_idx, i_b]
            point_sensor = gu.qd_inv_transform_by_trans_quat(point_world, sensor_pos, sensor_quat)
            for k in qd.static(range(3)):
                surface_pos_sensor_buf[i_b, i_o, k] = point_sensor[k]

            min_sdf = _func_elastomer_min_sdf_over_active_geoms(
                i_b,
                point_world,
                sensor_elastomer_geom_start[i_s],
                sensor_elastomer_geom_n[i_s],
                elastomer_geom_idx,
                elastomer_geom_active_envs_mask,
                geoms_state,
                geoms_info,
                sdf_info,
            )

            sdf_value = min_sdf
            depth = qd.max(gs.qd_float(0.0), -sdf_value)
            surface_depth_buf[i_b, i_o] = depth

            _func_elastomer_update_surface_anchor(
                i_b,
                i_o,
                sdf_value,
                point_sensor,
                sdf_enter[i_s],
                sdf_exit[i_s],
                surface_entry_pos_sensor_buf,
                surface_initialized_buf,
            )

            if not surface_initialized_buf[i_b, i_o] or shear_scale[i_s] <= gs.qd_float(0.0) or depth <= eps:
                continue

            entry = qd.Vector(
                [
                    surface_entry_pos_sensor_buf[i_b, i_o, 0],
                    surface_entry_pos_sensor_buf[i_b, i_o, 1],
                    surface_entry_pos_sensor_buf[i_b, i_o, 2],
                ],
                dt=gs.qd_float,
            )

            probe_start = sensor_probe_start[i_s]
            n_probes = n_probes_per_sensor[i_s]
            cache_start = sensor_cache_start[i_s]
            lam = lambda_s[i_s]
            scale = shear_scale[i_s]
            for j in range(n_probes):
                probe_idx = probe_start + j
                contribution = _func_elastomer_direct_shear_contribution(
                    point_sensor,
                    entry,
                    _func_vec3_at(probe_positions_local, probe_idx),
                    _func_vec3_at(probe_local_normal, probe_idx),
                    depth,
                    lam,
                    scale,
                    eps,
                )
                for k in qd.static(range(3)):
                    qd.atomic_add(output[cache_start + j * 3 + k, i_b], contribution[k])


def _elastomer_taxel_grid_fft_dilate(
    use_grid_fft: torch.Tensor,
    probe_depth_buf: torch.Tensor,
    sensor_probe_start: torch.Tensor,
    sensor_cache_start: torch.Tensor,
    grid_n: torch.Tensor,
    fft_depth_buffer: torch.Tensor,
    fft_kernel_list: list[torch.Tensor],
    lambda_d: torch.Tensor,
    dilate_scale: torch.Tensor,
    grid_normal: torch.Tensor,
    grid_tangent_u: torch.Tensor,
    grid_tangent_v: torch.Tensor,
    grid_dilate_out_buffer: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """
    HydroShear-style marker dilation via 2D FFT in the validated probe tangent basis.
    """
    n_batches = probe_depth_buf.shape[0]
    n_sensors = grid_n.shape[0]

    for i_s in range(n_sensors):
        if not use_grid_fft[i_s]:
            continue

        g_nx = grid_n[i_s, 0].item()
        g_ny = grid_n[i_s, 1].item()
        probe_start = sensor_probe_start[i_s].item()
        cache_start = sensor_cache_start[i_s].item()

        Kx = fft_kernel_list[i_s][0]
        Ky = fft_kernel_list[i_s][1]
        Kn = fft_kernel_list[i_s][2]
        # Linear-convolution padding (fft size >= 2*g - 1) avoids circular wrap across sensor borders.
        fft_nx, fft_ny = Kx.shape[0], Kx.shape[1]

        # H = contact depth grid (ix, iy); zero-pad to fft size for linear convolution. Probes from
        # generate_grid_points_on_plane are flat as iy*nx+ix; view (g_ny, g_nx) then transpose -> (g_nx, g_ny).
        depth_slice = probe_depth_buf[:, probe_start : probe_start + g_nx * g_ny]
        H_buf = fft_depth_buffer[:, i_s, :fft_nx, :fft_ny]
        H_buf.zero_()
        H_buf[:, :g_nx, :g_ny].copy_(depth_slice.view(n_batches, g_ny, g_nx).transpose(1, 2))

        # D_x = real(ifft(fft(H) * fft(K_x))), D_y = real(ifft(fft(H) * fft(K_y))).
        H_fft = torch.fft.fft2(H_buf)
        disp_u = torch.fft.ifft2(H_fft * Kx).real[:, :g_nx, :g_ny]
        disp_v = torch.fft.ifft2(H_fft * Ky).real[:, :g_nx, :g_ny]
        disp_n = torch.fft.ifft2(torch.fft.fft2(H_buf * H_buf) * Kn).real[:, :g_nx, :g_ny]

        disp_u = disp_u * dilate_scale[i_s]
        disp_v = disp_v * dilate_scale[i_s]
        disp_n = disp_n * dilate_scale[i_s]

        # Cache order is probe flat index iy*nx+ix; (g_nx, g_ny) transpose(1, 2).reshape gives (g_ny, g_nx) -> iy*nx+ix.
        disp_u_flat = disp_u.transpose(1, 2).reshape(n_batches, -1)
        disp_v_flat = disp_v.transpose(1, 2).reshape(n_batches, -1)
        disp_n_flat = disp_n.transpose(1, 2).reshape(n_batches, -1)

        grid_size = g_nx * g_ny * 3
        out_block = grid_dilate_out_buffer[:, :grid_size]
        tangent_u = grid_tangent_u[i_s]
        tangent_v = grid_tangent_v[i_s]
        normal = grid_normal[i_s]
        for k in range(3):
            out_block[:, k:grid_size:3] = (
                disp_u_flat * tangent_u[k] + disp_v_flat * tangent_v[k] + disp_n_flat * normal[k]
            )
        output[cache_start : cache_start + grid_size].copy_(out_block.T)


@dataclass
class ElastomerTaxelSensorMetadata(PointCloudTactileSharedMetadata, ProbesWithNormalSensorMetadataMixin):
    track_geom_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    track_geom_active_envs_mask: torch.Tensor = make_tensor_field((0, 0), dtype_factory=lambda: gs.tc_bool)
    sensor_track_geom_start: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    sensor_track_geom_n: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)

    elastomer_geom_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    elastomer_geom_active_envs_mask: torch.Tensor = make_tensor_field((0, 0), dtype_factory=lambda: gs.tc_bool)
    sensor_elastomer_geom_start: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    sensor_elastomer_geom_n: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)

    lambda_d: torch.Tensor = make_tensor_field((0,))
    lambda_s: torch.Tensor = make_tensor_field((0,))
    dilate_scale: torch.Tensor = make_tensor_field((0,))
    shear_scale: torch.Tensor = make_tensor_field((0,))
    elastomer_contact_sdf_enter: torch.Tensor = make_tensor_field((0,))
    elastomer_contact_sdf_exit: torch.Tensor = make_tensor_field((0,))

    probe_depth_buf: torch.Tensor = make_tensor_field((0, 0))
    surface_pos_sensor_buf: torch.Tensor = make_tensor_field((0, 0, 3))
    surface_entry_pos_sensor_buf: torch.Tensor = make_tensor_field((0, 0, 3))
    surface_depth_buf: torch.Tensor = make_tensor_field((0, 0))
    surface_initialized_buf: torch.Tensor = make_tensor_field((0, 0), dtype_factory=lambda: gs.tc_bool)

    is_grid: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_bool)
    use_grid_fft: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_bool)
    grid_n: torch.Tensor = make_tensor_field((0, 2), dtype_factory=lambda: gs.tc_int)
    grid_spacing: torch.Tensor = make_tensor_field((0, 2))
    grid_normal: torch.Tensor = make_tensor_field((0, 3))
    grid_tangent_u: torch.Tensor = make_tensor_field((0, 3))
    grid_tangent_v: torch.Tensor = make_tensor_field((0, 3))
    fft_kernel_list: list[torch.Tensor] = field(default_factory=list)
    fft_depth_buffer: torch.Tensor = make_tensor_field((0, 0, 0, 0))
    grid_dilate_out_buffer: torch.Tensor = make_tensor_field((0, 0))


class ElastomerTaxelSensor(
    PointCloudTactileSensorMixin[ElastomerTaxelSensorMetadata],
    ProbesWithNormalSensorMixin[ElastomerTaxelSensorMetadata],
    RigidSensorMixin[ElastomerTaxelSensorMetadata],
    SimpleSensor[ElastomerTaxelSensorOptions, ElastomerTaxelSensorMetadata],
):
    def __init__(self, sensor_options: ElastomerTaxelSensorOptions, sensor_idx: int, sensor_manager: "SensorManager"):
        super().__init__(sensor_options, sensor_idx, sensor_manager)

        self._is_grid = self._probe_local_pos.ndim > 2
        self._shape = self._probe_local_pos.shape[:-1]

        (probe_pos, probe_normals, use_grid_fft, grid_normal, grid_tangent_u, grid_tangent_v, grid_spacing) = (
            _normalize_elastomer_probe_layout(
                np.asarray(sensor_options.probe_local_pos, dtype=gs.np_float),
                np.asarray(sensor_options.probe_local_normal, dtype=gs.np_float),
                self._is_grid,
            )
        )
        self._probe_local_pos = torch.tensor(probe_pos, dtype=gs.tc_float, device=gs.device)
        self._probe_local_normal = torch.tensor(probe_normals, dtype=gs.tc_float, device=gs.device)
        self._use_grid_fft = use_grid_fft
        self._grid_normal = torch.tensor(grid_normal, dtype=gs.tc_float, device=gs.device)
        self._grid_tangent_u = torch.tensor(grid_tangent_u, dtype=gs.tc_float, device=gs.device)
        self._grid_tangent_v = torch.tensor(grid_tangent_v, dtype=gs.tc_float, device=gs.device)
        self._grid_spacing = torch.tensor(grid_spacing, dtype=gs.tc_float, device=gs.device)

    def build(self):
        super().build()

        solver = self._shared_metadata.solver
        solver.collider.activate_sdf()
        B = self._manager._sim._B
        if self._link is None:
            gs.raise_exception("ElastomerTaxel must be attached to a rigid link with collision geometry.")

        elastomer_geom_start_row = self._shared_metadata.elastomer_geom_idx.shape[0]
        elastomer_geom_idx, elastomer_geom_active_envs_mask = _collect_collision_geom_idx(
            solver, np.asarray((self._link.idx,), dtype=gs.np_int)
        )
        self._shared_metadata.elastomer_geom_idx = concat_with_tensor(
            self._shared_metadata.elastomer_geom_idx, elastomer_geom_idx, expand=(elastomer_geom_idx.shape[0],)
        )
        self._shared_metadata.elastomer_geom_active_envs_mask = concat_with_tensor(
            self._shared_metadata.elastomer_geom_active_envs_mask, elastomer_geom_active_envs_mask
        )
        self._shared_metadata.sensor_elastomer_geom_start = concat_with_tensor(
            self._shared_metadata.sensor_elastomer_geom_start, elastomer_geom_start_row, expand=(1,)
        )
        self._shared_metadata.sensor_elastomer_geom_n = concat_with_tensor(
            self._shared_metadata.sensor_elastomer_geom_n,
            self._shared_metadata.elastomer_geom_idx.shape[0] - elastomer_geom_start_row,
            expand=(1,),
        )

        track_link_idx = np.asarray(self._options.track_link_idx, dtype=gs.np_int)
        geom_start_row = self._shared_metadata.track_geom_idx.shape[0]
        geom_idx, geom_active_envs_mask = _collect_collision_geom_idx(solver, track_link_idx)
        self._shared_metadata.track_geom_idx = concat_with_tensor(
            self._shared_metadata.track_geom_idx, geom_idx, expand=(geom_idx.shape[0],)
        )
        self._shared_metadata.track_geom_active_envs_mask = concat_with_tensor(
            self._shared_metadata.track_geom_active_envs_mask, geom_active_envs_mask
        )
        self._shared_metadata.sensor_track_geom_start = concat_with_tensor(
            self._shared_metadata.sensor_track_geom_start, geom_start_row, expand=(1,)
        )
        self._shared_metadata.sensor_track_geom_n = concat_with_tensor(
            self._shared_metadata.sensor_track_geom_n,
            self._shared_metadata.track_geom_idx.shape[0] - geom_start_row,
            expand=(1,),
        )

        self._shared_metadata.lambda_d = concat_with_tensor(
            self._shared_metadata.lambda_d, float(self._options.lambda_d), expand=(1,)
        )
        self._shared_metadata.lambda_s = concat_with_tensor(
            self._shared_metadata.lambda_s, float(self._options.lambda_s), expand=(1,)
        )
        self._shared_metadata.dilate_scale = concat_with_tensor(
            self._shared_metadata.dilate_scale, float(self._options.dilate_scale), expand=(1,)
        )
        self._shared_metadata.shear_scale = concat_with_tensor(
            self._shared_metadata.shear_scale, float(self._options.shear_scale), expand=(1,)
        )
        self._shared_metadata.elastomer_contact_sdf_enter = concat_with_tensor(
            self._shared_metadata.elastomer_contact_sdf_enter,
            float(self._options.elastomer_contact_sdf_enter),
            expand=(1,),
        )
        self._shared_metadata.elastomer_contact_sdf_exit = concat_with_tensor(
            self._shared_metadata.elastomer_contact_sdf_exit,
            float(self._options.elastomer_contact_sdf_exit),
            expand=(1,),
        )

        self._shared_metadata.probe_depth_buf = torch.zeros(
            (B, self._shared_metadata.total_n_probes), dtype=gs.tc_float, device=gs.device
        )
        total_n_surface = self._shared_metadata.pc_pos_link.shape[0]
        self._shared_metadata.surface_pos_sensor_buf = torch.zeros(
            (B, total_n_surface, 3), dtype=gs.tc_float, device=gs.device
        )
        self._shared_metadata.surface_entry_pos_sensor_buf = torch.zeros(
            (B, total_n_surface, 3), dtype=gs.tc_float, device=gs.device
        )
        self._shared_metadata.surface_depth_buf = torch.zeros((B, total_n_surface), dtype=gs.tc_float, device=gs.device)
        self._shared_metadata.surface_initialized_buf = torch.zeros(
            (B, total_n_surface), dtype=gs.tc_bool, device=gs.device
        )

        self._shared_metadata.is_grid = concat_with_tensor(self._shared_metadata.is_grid, self._is_grid, expand=(1,))
        self._shared_metadata.use_grid_fft = concat_with_tensor(
            self._shared_metadata.use_grid_fft, self._use_grid_fft, expand=(1,)
        )

        grid_n = torch.tensor((0, 0), dtype=gs.tc_int, device=gs.device)
        grid_spacing = torch.tensor((0.0, 0.0), dtype=gs.tc_float, device=gs.device)
        grid_normal = torch.zeros(3, dtype=gs.tc_float, device=gs.device)
        grid_tangent_u = torch.zeros(3, dtype=gs.tc_float, device=gs.device)
        grid_tangent_v = torch.zeros(3, dtype=gs.tc_float, device=gs.device)
        kernel_fft = torch.tensor(0.0, dtype=gs.tc_float, device=gs.device)
        if self._use_grid_fft:
            nx, ny = int(self._shape[1]), int(self._shape[0])
            grid_n = torch.tensor((nx, ny), dtype=gs.tc_int, device=gs.device)
            grid_spacing = self._grid_spacing
            grid_normal = self._grid_normal
            grid_tangent_u = self._grid_tangent_u
            grid_tangent_v = self._grid_tangent_v
            fft_n = tuple(_next_pow2(2 * n - 1) for n in (nx, ny))
            kernel_fft = _precompute_hydroshear_dilate_kernel_fft(
                self._options.lambda_d, grid_spacing.tolist(), fft_n, gs.device, gs.tc_float
            )
            n_sensors = len(self._shared_metadata.lambda_d)
            prev = self._shared_metadata.fft_depth_buffer.shape
            max_fft_n = (max(fft_n[0], prev[2]), max(fft_n[1], prev[3]))
            self._shared_metadata.fft_depth_buffer = torch.zeros(
                (B, n_sensors, max_fft_n[0], max_fft_n[1]), dtype=gs.tc_float, device=gs.device
            )
            grid_size = nx * ny * 3
            out_buf = self._shared_metadata.grid_dilate_out_buffer
            if out_buf.numel() == 0 or out_buf.shape[1] < grid_size:
                self._shared_metadata.grid_dilate_out_buffer = torch.empty(
                    (B, max(out_buf.shape[1] if out_buf.numel() > 0 else 0, grid_size)),
                    dtype=gs.tc_float,
                    device=gs.device,
                )

        self._shared_metadata.grid_n = concat_with_tensor(self._shared_metadata.grid_n, grid_n, expand=(1, 2))
        self._shared_metadata.grid_spacing = concat_with_tensor(
            self._shared_metadata.grid_spacing, grid_spacing, expand=(1, 2)
        )
        self._shared_metadata.grid_normal = concat_with_tensor(
            self._shared_metadata.grid_normal, grid_normal, expand=(1, 3)
        )
        self._shared_metadata.grid_tangent_u = concat_with_tensor(
            self._shared_metadata.grid_tangent_u, grid_tangent_u, expand=(1, 3)
        )
        self._shared_metadata.grid_tangent_v = concat_with_tensor(
            self._shared_metadata.grid_tangent_v, grid_tangent_v, expand=(1, 3)
        )
        self._shared_metadata.fft_kernel_list.append(kernel_fft)

    def _get_return_format(self) -> tuple[int, ...]:
        return (self._n_probes, 3)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def reset(cls, shared_metadata: ElastomerTaxelSensorMetadata, shared_ground_truth_cache: torch.Tensor, envs_idx):
        super().reset(shared_metadata, shared_ground_truth_cache, envs_idx)
        shared_metadata.probe_depth_buf[envs_idx, :] = 0.0
        shared_metadata.surface_pos_sensor_buf[envs_idx, :, :] = 0.0
        shared_metadata.surface_entry_pos_sensor_buf[envs_idx, :, :] = 0.0
        shared_metadata.surface_depth_buf[envs_idx, :] = 0.0
        shared_metadata.surface_initialized_buf[envs_idx, :] = False

    @classmethod
    def _update_current_timestep_data(
        cls,
        shared_metadata: ElastomerTaxelSensorMetadata,
        current_ground_truth_data_T: torch.Tensor,
        ground_truth_data_timeline: "TensorRingBuffer | None",
        measured_data_timeline: "TensorRingBuffer",
    ):
        solver = shared_metadata.solver
        current_ground_truth_data_T.zero_()
        shared_metadata.probe_depth_buf.zero_()
        shared_metadata.surface_depth_buf.zero_()

        measured = measured_data_timeline.at(0, copy=False)
        measured.zero_()

        _kernel_elastomer_probe_depth_and_direct_dilate(
            shared_metadata.use_grid_fft,
            shared_metadata.probe_positions,
            shared_metadata.probe_local_normal,
            shared_metadata.probe_sensor_idx,
            shared_metadata.links_idx,
            shared_metadata.sensor_cache_start,
            shared_metadata.sensor_probe_start,
            shared_metadata.n_probes_per_sensor,
            shared_metadata.sensor_track_geom_start,
            shared_metadata.sensor_track_geom_n,
            shared_metadata.track_geom_idx,
            shared_metadata.track_geom_active_envs_mask,
            shared_metadata.lambda_d,
            shared_metadata.dilate_scale,
            solver.links_state,
            solver.geoms_state,
            solver.geoms_info,
            solver.collider._sdf._sdf_info,
            shared_metadata.probe_depth_buf,
            current_ground_truth_data_T,
        )
        _elastomer_taxel_grid_fft_dilate(
            shared_metadata.use_grid_fft,
            shared_metadata.probe_depth_buf,
            shared_metadata.sensor_probe_start,
            shared_metadata.sensor_cache_start,
            shared_metadata.grid_n,
            shared_metadata.fft_depth_buffer,
            shared_metadata.fft_kernel_list,
            shared_metadata.lambda_d,
            shared_metadata.dilate_scale,
            shared_metadata.grid_normal,
            shared_metadata.grid_tangent_u,
            shared_metadata.grid_tangent_v,
            shared_metadata.grid_dilate_out_buffer,
            current_ground_truth_data_T,
        )
        _kernel_elastomer_surface_state_and_direct_shear(
            shared_metadata.probe_positions,
            shared_metadata.probe_local_normal,
            shared_metadata.sensor_cache_start,
            shared_metadata.sensor_probe_start,
            shared_metadata.n_probes_per_sensor,
            shared_metadata.links_idx,
            shared_metadata.sensor_elastomer_geom_start,
            shared_metadata.sensor_elastomer_geom_n,
            shared_metadata.elastomer_geom_idx,
            shared_metadata.elastomer_geom_active_envs_mask,
            shared_metadata.sensor_pc_start,
            shared_metadata.sensor_pc_n,
            shared_metadata.pc_link_idx,
            shared_metadata.pc_pos_link,
            shared_metadata.pc_active_envs_mask,
            shared_metadata.lambda_s,
            shared_metadata.shear_scale,
            solver.links_state,
            solver.geoms_state,
            solver.geoms_info,
            solver.collider._sdf._sdf_info,
            gs.EPS,
            shared_metadata.elastomer_contact_sdf_enter,
            shared_metadata.elastomer_contact_sdf_exit,
            shared_metadata.surface_pos_sensor_buf,
            shared_metadata.surface_entry_pos_sensor_buf,
            shared_metadata.surface_depth_buf,
            shared_metadata.surface_initialized_buf,
            current_ground_truth_data_T,
        )

        if ground_truth_data_timeline is not None:
            ground_truth_data_timeline.at(0, copy=False).copy_(current_ground_truth_data_T.T)
        measured.copy_(current_ground_truth_data_T.T)

    def _draw_debug(self, context: "RasterizerContext"):
        self._draw_debug_probes(
            context, lambda envs_idx: tensor_to_array(torch.linalg.norm(self.read_ground_truth(envs_idx), dim=-1))
        )
