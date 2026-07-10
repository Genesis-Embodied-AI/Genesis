from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, NamedTuple, TypeVar

import numpy as np
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
import genesis.utils.sdf as sdf
from genesis.engine.bvh import STACK_SIZE as _BVH_STACK_SIZE
from genesis.options.sensors import ElastomerTaxel as ElastomerTaxelSensorOptions
from genesis.options.sensors import ProximityTaxel as ProximityTaxelOptions
from genesis.utils.misc import concat_with_tensor, make_tensor_field, tensor_to_array
from genesis.utils.point_cloud import sample_mesh_point_cloud
from genesis.utils.raycast_qd import (
    closest_point_on_triangle,
    get_triangle_vertices,
    triangle_face_normal,
)

from .base_sensor import RigidSensorMetadataMixin, RigidSensorMixin, SimpleSensor, SimpleSensorMetadata
from .probe import (
    ProbeSensorMetadataMixin,
    ProbeSensorMixin,
    ProbesWithNormalSensorMetadataMixin,
    ProbesWithNormalSensorMixin,
    func_noised_probe_radius,
    get_measured_bufs,
)
from .raycaster import RaycastContext
from .tactile_shared import (
    BVH_LEAF_SIZE,
    BVH_STACK_SIZE,
    BVHMetadata,
    ChunkedBVHData,
    ContactDepthQueryMetadataMixin,
    ContactDepthQuerySensorMixin,
    GridFFTConvMetadataMixin,
    build_static_chunk_bvh,
    func_aabb_intersects_aabb,
    func_sphere_intersects_aabb,
    func_vec3_at,
    get_mesh_geom_chunks,
    next_pow2,
    normalize_grid_probe_layout,
    register_grid_fft_sensor,
)

# Conservative cap for global-BVH closest-point walks in raycast mode. Points farther than this from every candidate
# triangle map to depth = 0 (so the elastomer "out of contact" branch fires). Sized to cover realistic elastomer
# penetrations -- bumping it widens BVH traversal cost but doesn't change correctness for in-contact probes.
_ELASTOMER_RAYCAST_QUERY_DIST = 0.1

if TYPE_CHECKING:
    from genesis.options.sensors import SensorOptions
    from genesis.utils.ring_buffer import TensorRingBuffer
    from genesis.vis.rasterizer_context import RasterizerContext

    from .sensor_manager import SensorManager


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


class GridFFTMeta(NamedTuple):
    """
    Per-grid-FFT-sensor record for HydroShear dilation.

    ``sensor_idx``/``g_ny``/``g_nx``/``probe_start``/``cache_start`` are the leading fields every grid-FFT sensor
    shares (the contract ``register_grid_fft_sensor`` relies on); ``lambda_d``/``spacing_u``/``spacing_v`` are the
    HydroShear kernel params consumed by ``_dilate_kernel_builder``.
    """

    sensor_idx: int
    g_ny: int
    g_nx: int
    probe_start: int
    cache_start: int
    lambda_d: float
    spacing_u: float
    spacing_v: float


def _build_candidate_geom_mask(
    B: int,
    n_sensors: int,
    n_geoms: int,
    geom_starts: torch.Tensor,
    geom_ns: torch.Tensor,
    geom_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Build a ``(B, n_sensors, n_geoms)`` bool mask marking, per sensor, which scene geoms are candidates.

    ``geom_idx`` is the flat per-sensor concatenation of candidate geom indices; ``geom_starts``/``geom_ns`` give
    each sensor's slice into it. The mask is broadcast identically across all ``B`` environments.
    """
    mask = torch.zeros((B, n_sensors, n_geoms), dtype=gs.tc_bool, device=gs.device)
    starts = tensor_to_array(geom_starts)
    ns = tensor_to_array(geom_ns)
    idx = tensor_to_array(geom_idx)
    for i_s in range(n_sensors):
        lo = int(starts[i_s])
        hi = lo + int(ns[i_s])
        if hi > lo:
            mask[:, i_s, idx[lo:hi]] = True
    return mask


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

    areas = np.asarray([_mesh_area(verts, faces) for _, verts, faces in geom_chunks], dtype=gs.np_float)
    if float(areas.sum()) <= gs.EPS:
        areas.fill(1.0)

    if n_total < n_chunks:
        counts = np.zeros(n_chunks, dtype=gs.np_int)
        counts[np.argsort(-areas)[:n_total]] = 1
        return counts.tolist()

    raw_extra = (n_total - n_chunks) * areas / float(areas.sum())
    extra = np.floor(raw_extra).astype(gs.np_int)
    remainder = n_total - n_chunks - int(extra.sum())
    if remainder > 0:
        extra[np.argsort(-(raw_extra - extra))[:remainder]] += 1
    return (extra + 1).tolist()


def _active_envs_mask_tensor(geom, batch_size: int) -> torch.Tensor:
    if geom.active_envs_mask is None:
        return torch.ones((batch_size,), dtype=gs.tc_bool, device=gs.device)
    return geom.active_envs_mask.to(device=gs.device, dtype=gs.tc_bool)


def _group_geoms_by_variant(
    geom_chunks: list[tuple[object, np.ndarray, np.ndarray]], batch_size: int
) -> list[tuple[torch.Tensor, list[tuple[object, np.ndarray, np.ndarray]]]]:
    """
    Partition a link's geoms into heterogeneous-variant groups by ``active_envs_mask``.

    Geoms sharing a mask are one variant; ``None`` masks (homogeneous) collapse into a single all-True group.
    Returns ``[(mask, geom_chunks_for_variant), ...]`` preserving the original geom order within each group.
    """
    groups: dict[bytes, tuple[torch.Tensor, list[tuple[object, np.ndarray, np.ndarray]]]] = {}
    for chunk in geom_chunks:
        geom = chunk[0]
        mask = _active_envs_mask_tensor(geom, batch_size)
        key = tensor_to_array(mask).astype(np.bool_).tobytes()
        if key not in groups:
            groups[key] = (mask, [])
        groups[key][1].append(chunk)
    return list(groups.values())


def _sample_track_links_point_cloud_tensors(
    solver, track_link_idx: np.ndarray, n_sample_points: int | list | tuple, prefer_visual: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    FPS-sample meshes on ``track_link_idx`` into concatenated link-local positions and normals.

    The per-link budget from ``n_sample_points`` is allocated to every heterogeneous variant on a link
    (geoms grouped by ``active_envs_mask``), so each parallel environment sees the full requested point
    count regardless of which variant is active. Within a variant, the budget is split across geoms by
    surface area.

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
        geom_chunks = get_mesh_geom_chunks(link, prefer_visual)
        if not geom_chunks:
            gs.raise_exception(f"No mesh geometry on tracked link index {link_idx}.")
        for variant_mask, variant_chunks in _group_geoms_by_variant(geom_chunks, solver._B):
            for n_geom_pts, (geom, verts, faces) in zip(_split_count_by_area(n_pts, variant_chunks), variant_chunks):
                if n_geom_pts <= 0:
                    continue
                # Fixed seed: the cache key already discriminates between meshes (vertices+faces hashed), so the same
                # mesh always resolves to the same sample, which keeps tactile readings reproducible across
                # build/reset cycles.
                pts_np, nrm_np = sample_mesh_point_cloud(
                    verts, faces, n_geom_pts, seed=0, use_cache=True, return_normals=True
                )

                li = torch.full((pts_np.shape[0],), link_idx, dtype=gs.tc_int, device=gs.device)
                link_idx_chunks.append(li)
                pos_chunks.append(torch.tensor(pts_np, dtype=gs.tc_float, device=gs.device))
                nrm_chunks.append(torch.tensor(nrm_np, dtype=gs.tc_float, device=gs.device))
                active_chunks.append(variant_mask.expand(pts_np.shape[0], solver._B))

    if not pos_chunks:
        gs.raise_exception("PointCloudTactile sensor produced an empty object point cloud.")

    return (
        torch.cat(link_idx_chunks, dim=0),
        torch.cat(pos_chunks, dim=0),
        torch.cat(nrm_chunks, dim=0),
        torch.cat(active_chunks, dim=0),
    )


_ELASTOMER_QUERY_AABB_MARGIN = 1e-3


@dataclass
class PointCloudBVH(BVHMetadata):
    """
    BVH over the tracked point clouds of one sensor class.

    ``leaf_elem_idx`` entries are absolute rows into ``pc_pos_link`` / ``pc_active_envs_mask`` / ``pc_normal_link``
    so a leaf hit resolves to per-point data with one indirection. See ``BVHMetadata`` for the shared scaffolding
    semantics.
    """

    # Inverse of sensor_chunk_start/count: chunk_sensor_idx[i_c] is the owning sensor's index. Enables
    # (env, chunk)-parallel kernels (e.g. ElastomerTaxel surface state) without rescanning sensor_chunk_start
    # in every thread; ProximityTaxel parallelizes per-probe and does not consume this field.
    chunk_sensor_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)

    def append_sensor(self, *, pc_start_row: int, idx_cat: torch.Tensor, pos_cat: torch.Tensor) -> None:
        """
        Build per-tracked-link chunks for one sensor and append into the flat tensors.

        Must be called immediately after extending ``pc_pos_link`` by ``pos_cat`` so each leaf's element index
        (``pc_start_row + local_row``) addresses the freshly-grown rows.
        """
        n_local = int(pos_cat.shape[0])
        if n_local == 0:
            gs.raise_exception("PointCloudBVH.append_sensor called with empty point cloud.")

        idx_np = tensor_to_array(idx_cat).astype(gs.np_int)
        pos_np = tensor_to_array(pos_cat).astype(gs.np_float, copy=False)
        unique_links = np.unique(idx_np)

        chunk_start_for_sensor = int(self.chunk_link_idx.shape[0])
        node_offset = int(self.node_min.shape[0])
        point_offset = int(self.leaf_elem_idx.shape[0])

        new_chunk_link_idx: list[int] = []
        new_chunk_node_start: list[int] = []
        new_chunk_node_count: list[int] = []
        all_node_min: list[np.ndarray] = []
        all_node_max: list[np.ndarray] = []
        all_node_left: list[np.ndarray] = []
        all_node_right: list[np.ndarray] = []
        all_node_leaf_start: list[np.ndarray] = []
        all_node_leaf_count: list[np.ndarray] = []
        all_leaf_elem_idx: list[np.ndarray] = []

        for link_idx in unique_links:
            local_rows = np.nonzero(idx_np == int(link_idx))[0].astype(gs.np_int)
            global_rows = (int(pc_start_row) + local_rows).astype(gs.np_int)
            pts_link = pos_np[local_rows]

            # Point cloud: AABB per element is degenerate (the point itself), so pass the points as both
            # centroids and the per-element min/max bounds.
            nmin, nmax, nleft, nright, npstart, npn, pidx = build_static_chunk_bvh(
                pts_link, pts_link, pts_link, global_rows, BVH_LEAF_SIZE
            )

            new_chunk_link_idx.append(int(link_idx))
            new_chunk_node_start.append(node_offset)
            new_chunk_node_count.append(int(nmin.shape[0]))

            all_node_min.append(nmin)
            all_node_max.append(nmax)
            # Rebase intra-chunk child / leaf-start indices into the flat tensors' absolute space.
            all_node_left.append(np.where(nleft >= 0, nleft + node_offset, nleft).astype(gs.np_int))
            all_node_right.append(np.where(nright >= 0, nright + node_offset, nright).astype(gs.np_int))
            all_node_leaf_start.append(np.where(npn > 0, npstart + point_offset, npstart).astype(gs.np_int))
            all_node_leaf_count.append(npn)
            all_leaf_elem_idx.append(pidx)

            node_offset += int(nmin.shape[0])
            point_offset += int(pidx.shape[0])

        nm = torch.tensor(np.concatenate(all_node_min, axis=0), dtype=gs.tc_float, device=gs.device)
        nx = torch.tensor(np.concatenate(all_node_max, axis=0), dtype=gs.tc_float, device=gs.device)
        nl = torch.tensor(np.concatenate(all_node_left, axis=0), dtype=gs.tc_int, device=gs.device)
        nr = torch.tensor(np.concatenate(all_node_right, axis=0), dtype=gs.tc_int, device=gs.device)
        nps = torch.tensor(np.concatenate(all_node_leaf_start, axis=0), dtype=gs.tc_int, device=gs.device)
        npn_t = torch.tensor(np.concatenate(all_node_leaf_count, axis=0), dtype=gs.tc_int, device=gs.device)
        pidx_t = torch.tensor(np.concatenate(all_leaf_elem_idx, axis=0), dtype=gs.tc_int, device=gs.device)
        cli = torch.tensor(new_chunk_link_idx, dtype=gs.tc_int, device=gs.device)
        cns = torch.tensor(new_chunk_node_start, dtype=gs.tc_int, device=gs.device)
        cnn = torch.tensor(new_chunk_node_count, dtype=gs.tc_int, device=gs.device)
        # Sensor index for this batch of chunks = current sensor count (the entry we're about to add).
        sensor_idx_for_chunks = int(self.sensor_chunk_start.shape[0])
        csi = torch.full((len(unique_links),), sensor_idx_for_chunks, dtype=gs.tc_int, device=gs.device)

        self.node_min = concat_with_tensor(self.node_min, nm, expand=(nm.shape[0], 3))
        self.node_max = concat_with_tensor(self.node_max, nx, expand=(nx.shape[0], 3))
        self.node_left = concat_with_tensor(self.node_left, nl, expand=(nl.shape[0],))
        self.node_right = concat_with_tensor(self.node_right, nr, expand=(nr.shape[0],))
        self.node_leaf_start = concat_with_tensor(self.node_leaf_start, nps, expand=(nps.shape[0],))
        self.node_leaf_count = concat_with_tensor(self.node_leaf_count, npn_t, expand=(npn_t.shape[0],))
        self.leaf_elem_idx = concat_with_tensor(self.leaf_elem_idx, pidx_t, expand=(pidx_t.shape[0],))
        self.chunk_link_idx = concat_with_tensor(self.chunk_link_idx, cli, expand=(cli.shape[0],))
        self.chunk_sensor_idx = concat_with_tensor(self.chunk_sensor_idx, csi, expand=(csi.shape[0],))
        self.chunk_node_start = concat_with_tensor(self.chunk_node_start, cns, expand=(cns.shape[0],))
        self.chunk_node_count = concat_with_tensor(self.chunk_node_count, cnn, expand=(cnn.shape[0],))
        self.sensor_chunk_start = concat_with_tensor(self.sensor_chunk_start, chunk_start_for_sensor, expand=(1,))
        self.sensor_chunk_count = concat_with_tensor(self.sensor_chunk_count, len(unique_links), expand=(1,))


@qd.kernel
def _kernel_point_cloud_proximity_taxel_bvh(
    probe_positions_local: qd.types.ndarray(),
    probe_local_normal: qd.types.ndarray(),
    probe_sensor_idx: qd.types.ndarray(),
    links_idx: qd.types.ndarray(),
    sensor_cache_start: qd.types.ndarray(),
    sensor_probe_start: qd.types.ndarray(),
    n_probes_per_sensor: qd.types.ndarray(),
    bvh: ChunkedBVHData,
    pc_pos_link: qd.types.ndarray(),
    pc_active_envs_mask: qd.types.ndarray(),
    probe_radii: qd.types.ndarray(),
    probe_radii_noise: qd.types.ndarray(),
    probe_gains: qd.types.ndarray(),
    stiffness: qd.types.ndarray(),
    shear_coupling: qd.types.ndarray(),
    proximity_density_scale: qd.types.ndarray(),
    links_state: array_class.LinksState,
    eps: float,
    output_gt: qd.types.ndarray(),
    output_measured: qd.types.ndarray(),
    taxel_signal_buf: qd.types.ndarray(),
):
    total_n_probes = probe_positions_local.shape[0]
    n_batches = output_gt.shape[-1]

    for i_p, i_b in qd.ndrange(total_n_probes, n_batches):
        i_s = probe_sensor_idx[i_p]
        sensor_link_idx = links_idx[i_s]
        s_pos = links_state.pos[sensor_link_idx, i_b]
        s_quat = links_state.quat[sensor_link_idx, i_b]

        k_stiff = stiffness[i_s]
        k_shear = shear_coupling[i_s]
        dens = proximity_density_scale[i_s, i_b]
        n_probes = n_probes_per_sensor[i_s]
        cache_start = sensor_cache_start[i_s]
        _i_p = i_p - sensor_probe_start[i_s]

        s_vel = links_state.cd_vel[sensor_link_idx, i_b]
        s_ang = links_state.cd_ang[sensor_link_idx, i_b]
        s_com = links_state.root_COM[sensor_link_idx, i_b]

        probe_local = func_vec3_at(probe_positions_local, i_p)
        probe_world = s_pos + gu.qd_transform_by_quat(probe_local, s_quat)

        a_loc = func_vec3_at(probe_local_normal, i_p)
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
        # Conservative traversal radius covers both branches; exact tests run per leaf candidate.
        R_query = qd.max(R_gt, R_m)
        R_query_sq = R_query * R_query

        v_tax = s_vel + s_ang.cross(probe_world - s_com)

        sum_p_gt = gs.qd_float(0.0)
        fv_gt = qd.Vector.zero(gs.qd_float, 3)
        tau_w_gt = qd.Vector.zero(gs.qd_float, 3)
        sum_p_m = gs.qd_float(0.0)
        fv_m = qd.Vector.zero(gs.qd_float, 3)
        tau_w_m = qd.Vector.zero(gs.qd_float, 3)

        chunk_start = bvh.sensor_chunk_start[i_s]
        n_chunks = bvh.sensor_chunk_count[i_s]
        for c_off in range(n_chunks):
            i_c = chunk_start + c_off
            track_link_idx = bvh.chunk_link_idx[i_c]
            track_pos = links_state.pos[track_link_idx, i_b]
            track_quat = links_state.quat[track_link_idx, i_b]
            rcom_o = links_state.root_COM[track_link_idx, i_b]
            cdv_o = links_state.cd_vel[track_link_idx, i_b]
            cda_o = links_state.cd_ang[track_link_idx, i_b]
            # BVH nodes live in tracked-link local frame: bring the probe sphere center over.
            probe_link = gu.qd_inv_transform_by_trans_quat(probe_world, track_pos, track_quat)

            stack = qd.Vector.zero(gs.qd_int, qd.static(BVH_STACK_SIZE))
            stack[0] = bvh.chunk_node_start[i_c]
            stack_idx = 1

            while stack_idx > 0:
                stack_idx -= 1
                n = stack[stack_idx]
                bmin = func_vec3_at(bvh.node_min, n)
                bmax = func_vec3_at(bvh.node_max, n)
                if not func_sphere_intersects_aabb(probe_link, R_query_sq, bmin, bmax):
                    continue
                left = bvh.node_left[n]
                if left == -1:
                    pstart = bvh.node_leaf_start[n]
                    pn = bvh.node_leaf_count[n]
                    for j in range(pn):
                        i_o = bvh.leaf_elem_idx[pstart + j]
                        if not pc_active_envs_mask[i_o, i_b]:
                            continue
                        pos_l = func_vec3_at(pc_pos_link, i_o)
                        d_link = pos_l - probe_link
                        dsq = d_link.dot(d_link)
                        dist = qd.sqrt(dsq)

                        hit_gt = dsq <= R_gt_sq and dist > eps
                        hit_m = use_noised_radius and dsq <= R_m_sq and dist > eps
                        if hit_gt or hit_m:
                            # Same-frame conversion: dvec_world = R_track * d_link, and the world
                            # point pw is reachable via probe_world + dvec_world (equivalent to
                            # track_pos + R_track * pos_l, up to float order).
                            d_world = gu.qd_transform_by_quat(d_link, track_quat)
                            pw = probe_world + d_world
                            v_pc = cdv_o + cda_o.cross(pw - rcom_o)
                            v_rel = v_pc - v_tax
                            vdota = v_rel.dot(a_w)
                            v_t = qd.Vector.zero(gs.qd_float, 3)
                            for k2 in qd.static(range(3)):
                                v_t[k2] = v_rel[k2] - a_w[k2] * vdota
                            ctmp = d_world.cross(a_w)

                            if hit_gt:
                                P_i_gt = R_gt - dist
                                if P_i_gt > 0.0:
                                    sum_p_gt = sum_p_gt + P_i_gt
                                    for k2 in qd.static(range(3)):
                                        fv_gt[k2] = fv_gt[k2] + P_i_gt * v_t[k2]
                                        tau_w_gt[k2] = tau_w_gt[k2] + P_i_gt * ctmp[k2]
                            if hit_m:
                                P_i_m = R_m - dist
                                if P_i_m > 0.0:
                                    sum_p_m = sum_p_m + P_i_m
                                    for k2 in qd.static(range(3)):
                                        fv_m[k2] = fv_m[k2] + P_i_m * v_t[k2]
                                        tau_w_m[k2] = tau_w_m[k2] + P_i_m * ctmp[k2]
                else:
                    right = bvh.node_right[n]
                    # Median split bounds depth at log2(N / leaf_size) << BVH_STACK_SIZE; the guard mirrors the
                    # global rigid-BVH kernel so a future build strategy can't silently overflow the stack.
                    if stack_idx < qd.static(BVH_STACK_SIZE - 2):
                        stack[stack_idx] = left
                        stack[stack_idx + 1] = right
                        stack_idx += 2

        if not use_noised_radius:
            sum_p_m = sum_p_gt
            for j in qd.static(range(3)):
                fv_m[j] = fv_gt[j]
                tau_w_m[j] = tau_w_gt[j]

        # Per-(env, probe) gain on the measured-branch accumulated penetration. Force and torque computed from
        # these accumulators downstream scale linearly with gain because they're proportional to ``sum_p``.
        gain_m = probe_gains[i_b, i_p]
        sum_p_m = sum_p_m * gain_m
        for j in qd.static(range(3)):
            fv_m[j] = fv_m[j] * gain_m
            tau_w_m[j] = tau_w_m[j] * gain_m

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
    """Shared sensor-manager state for point-cloud-tracked tactile sensors (probes + merged track PC)."""

    pc_link_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    pc_pos_link: torch.Tensor = make_tensor_field((0, 3))
    pc_normal_link: torch.Tensor = make_tensor_field((0, 3))
    pc_active_envs_mask: torch.Tensor = make_tensor_field((0, 0), dtype_factory=lambda: gs.tc_bool)
    sensor_pc_start: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    sensor_pc_n: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    pc_bvh: PointCloudBVH = field(default_factory=PointCloudBVH)


PointCloudTactileSensorMetadataMixinT = TypeVar(
    "PointCloudTactileSensorMetadataMixinT", bound=PointCloudTactileSharedMetadata
)


class PointCloudTactileSensorMixin(ProbeSensorMixin[PointCloudTactileSensorMetadataMixinT]):
    def __init__(
        self,
        options: "SensorOptions",
        idx: int,
        shared_context,
        shared_metadata,
        manager: "SensorManager",
    ):
        super().__init__(options, idx, shared_context, shared_metadata, manager)
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

        # BVH growth follows pc_pos_link growth in lockstep: each leaf's leaf_elem_idx is an absolute
        # row into the just-grown pc_pos_link.
        self._shared_metadata.pc_bvh.append_sensor(
            pc_start_row=pc_start_row,
            idx_cat=idx_cat,
            pos_cat=pos_cat,
        )

    def _draw_debug_probes(
        self,
        context: "RasterizerContext",
        color_groups_fn: Callable[[list[int] | None], list[tuple]] | None = None,
    ) -> tuple[list[int] | None, int, np.ndarray | None]:
        envs_idx, n_debug_envs, env_offsets = super()._draw_debug_probes(context, color_groups_fn)

        if self._debug_pc_chunks is None:
            return envs_idx, n_debug_envs, env_offsets
        world_chunks: list[np.ndarray] = []
        for link_idx, pos_local, active_envs_mask in self._debug_pc_chunks:
            track_link = self._shared_metadata.solver.links[link_idx]
            if envs_idx is not None:
                active_mask = tensor_to_array(active_envs_mask[:, envs_idx].T).astype(bool)
                if not active_mask.any():
                    continue
                track_pos = track_link.get_pos(envs_idx, relative=False)[:, None, :]
                track_quat = track_link.get_quat(envs_idx, relative=False)[:, None, :]
                pc_world = gu.transform_by_trans_quat(pos_local[None, :, :], track_pos, track_quat)
                pc_world = tensor_to_array(pc_world) + env_offsets[:, None, :]
                world_chunks.append(pc_world[active_mask])
            else:
                active_mask = active_envs_mask[:, 0]
                pos_active = pos_local[active_mask]
                if pos_active.numel() == 0:
                    continue
                track_pos = track_link.get_pos(envs_idx, relative=False).reshape(3)
                track_quat = track_link.get_quat(envs_idx, relative=False).reshape(4)
                world_chunks.append(tensor_to_array(gu.transform_by_trans_quat(pos_active, track_pos, track_quat)))
        if world_chunks:
            self._debug_objects.append(
                context.draw_debug_spheres(
                    poss=np.concatenate(world_chunks, axis=0),
                    radius=float(self._options.debug_point_cloud_radius),
                    color=self._options.debug_point_cloud_color,
                )
            )
        return envs_idx, n_debug_envs, env_offsets

    def _debug_probe_buffer_magnitudes(self, buffer: torch.Tensor, envs_idx: list[int] | None) -> np.ndarray:
        values = buffer[self._probe_start_idx : self._probe_start_idx + self._n_probes]
        if envs_idx is None:
            return tensor_to_array(values[:, 0])
        return tensor_to_array(values[:, envs_idx].T)


class ProximityTaxelReturnType(NamedTuple):
    """Per-taxel estimates in link-local frame."""

    force: torch.Tensor
    torque: torch.Tensor


@dataclass
class ProximityTaxelMetadata(
    PointCloudTactileSharedMetadata,
    ProbesWithNormalSensorMetadataMixin,
):
    stiffness: torch.Tensor = make_tensor_field((0,))
    shear_coupling: torch.Tensor = make_tensor_field((0,))
    proximity_density_scale: torch.Tensor = make_tensor_field((0, 0))
    taxel_signal_buf: torch.Tensor = make_tensor_field((0, 0))


class ProximityTaxelSensor(
    PointCloudTactileSensorMixin[ProximityTaxelMetadata],
    ProbesWithNormalSensorMixin[ProximityTaxelMetadata],
    RigidSensorMixin[ProximityTaxelMetadata],
    SimpleSensor[ProximityTaxelOptions, None, ProximityTaxelMetadata, ProximityTaxelReturnType],
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
        shape = (*self._probe_layout_shape, 3)
        return shape, shape

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
        shared_context: None,
        shared_metadata: ProximityTaxelMetadata,
        current_ground_truth_data_T: torch.Tensor,
        ground_truth_data_timeline: "TensorRingBuffer | None",
        measured_data_timeline: "TensorRingBuffer",
    ):
        solver = shared_metadata.solver
        measured, measured_cols_b = get_measured_bufs(
            shared_metadata, current_ground_truth_data_T, measured_data_timeline
        )
        bvh = shared_metadata.pc_bvh
        _kernel_point_cloud_proximity_taxel_bvh(
            shared_metadata.probe_positions,
            shared_metadata.probe_local_normal,
            shared_metadata.probe_sensor_idx,
            shared_metadata.links_idx,
            shared_metadata.sensor_cache_start,
            shared_metadata.sensor_probe_start,
            shared_metadata.n_probes_per_sensor,
            bvh.kernel_bvh,
            shared_metadata.pc_pos_link,
            shared_metadata.pc_active_envs_mask,
            shared_metadata.probe_radii,
            shared_metadata.probe_radii_noise,
            shared_metadata.probe_gains,
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
            self._tactile_color_groups_fn(
                lambda envs_idx: (
                    self._debug_probe_buffer_magnitudes(self._shared_metadata.taxel_signal_buf, envs_idx) >= gs.EPS
                ),
            ),
        )


@qd.func
def _func_elastomer_min_sdf_over_active_geoms(
    i_b: int,
    point_world: qd.types.vector(3),
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
        # AABB pre-cull: the geom is fully contained in its world AABB, so a point strictly outside
        # the AABB has sdf > 0 and can't be the min when any other geom contains the point. If no
        # geom contains the point, min_sdf stays at 1.0e6 -- callers map that to depth=0 and the
        # surface-state "exit" branch (sdf > sdf_exit), both correct.
        amin = geoms_state.aabb_min[i_g, i_b]
        amax = geoms_state.aabb_max[i_g, i_b]
        if (
            point_world[0] < amin[0]
            or point_world[0] > amax[0]
            or point_world[1] < amin[1]
            or point_world[1] > amax[1]
            or point_world[2] < amin[2]
            or point_world[2] > amax[2]
        ):
            continue
        sd = sdf.sdf_func_world(geoms_state, geoms_info, sdf_info, point_world, i_g, i_b)
        if sd < min_sdf:
            min_sdf = sd
    return min_sdf


@qd.func
def _func_elastomer_tangent(
    vec: qd.types.vector(3),
    normal: qd.types.vector(3),
) -> qd.types.vector(3):
    return vec - normal * vec.dot(normal)


@qd.func
def _func_elastomer_update_surface_anchor(
    i_b: int,
    i_o: int,
    sdf_value: float,
    point_sensor: qd.types.vector(3),
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
    source_pos: qd.types.vector(3),
    target_pos: qd.types.vector(3),
    target_normal: qd.types.vector(3),
    depth: float,
    lam: float,
    scale: float,
    normal_exponent: float,
) -> qd.types.vector(3):
    # Tangential marker spreading is linear in penetration depth; the out-of-plane bulge follows a
    # ``depth ** normal_exponent`` power law (mirrors the FFT path's H / H**normal_exponent channel split).
    planar_diff = _func_elastomer_tangent(target_pos - source_pos, target_normal)
    falloff = qd.exp(-lam * planar_diff.dot(planar_diff)) * scale
    return (planar_diff * depth + target_normal * qd.pow(depth, normal_exponent)) * falloff


@qd.func
def _func_elastomer_direct_shear_contribution(
    point_sensor: qd.types.vector(3),
    entry_sensor: qd.types.vector(3),
    probe_pos: qd.types.vector(3),
    probe_normal: qd.types.vector(3),
    depth: float,
    lam: float,
    scale: float,
    eps: float,
) -> qd.types.vector(3):
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
    """Real FFT of the 3-plane HydroShear dilation kernel ``(Ku, Kv, Kn)``.

    ``fft_n`` is ``(fft_ny, fft_nx)`` row-major: axis 0 spans the tangent_v direction, axis 1 the tangent_u
    direction. ``grid_spacing`` is ``(spacing_u, spacing_v)``. The output is a complex
    ``(3, fft_ny, fft_nx // 2 + 1)`` half-spectrum ready to multiply against ``rfft2(field)``.
    """
    iv = torch.arange(fft_n[0], dtype=dtype, device=device)
    iu = torch.arange(fft_n[1], dtype=dtype, device=device)
    vv, uu = torch.meshgrid(
        (iv - fft_n[0] // 2) * grid_spacing[1], (iu - fft_n[1] // 2) * grid_spacing[0], indexing="ij"
    )
    g = torch.exp(torch.tensor(-lambda_d, dtype=dtype, device=device) * (uu * uu + vv * vv))
    k = torch.stack((uu * g, vv * g, g), dim=0)
    k = torch.fft.ifftshift(k, dim=(-2, -1))
    return torch.fft.rfft2(k)


def _dilate_kernel_builder(meta_entry: GridFFTMeta, fft_n: tuple[int, int]) -> torch.Tensor:
    """``register_grid_fft_sensor`` kernel builder for HydroShear dilation: 3 planes ``(Ku, Kv, Kn)``."""
    return _precompute_hydroshear_dilate_kernel_fft(
        meta_entry.lambda_d, (meta_entry.spacing_u, meta_entry.spacing_v), fft_n, gs.device, gs.tc_float
    )


@qd.func
def _func_elastomer_min_signed_dist_bvh(
    i_b: int,
    i_s: int,
    probe_world: qd.types.vector(3),
    max_query_dist: float,
    bvh_nodes: qd.template(),
    bvh_morton_codes: qd.template(),
    faces_info: array_class.FacesInfo,
    verts_info: array_class.VertsInfo,
    fixed_verts_state: array_class.VertsState,
    free_verts_state: array_class.VertsState,
    track_geom_mask: qd.types.ndarray(),
) -> float:
    """
    BVH-based signed distance from ``probe_world`` to the nearest triangle of any geom flagged for this sensor in
    ``track_geom_mask`` (shape ``(B, n_sensors, n_geoms)``).

    Sign is positive when the probe is outside the surface (closest-triangle face-normal points away from probe),
    negative when inside. Mirrors the return contract of ``_func_elastomer_min_sdf_over_active_geoms`` so callers
    consume ``max(0, -signed)`` identically.

    Uses ``max_query_dist`` as the BVH cull radius: probes farther than that from every candidate triangle are
    treated as fully outside (returns ``+max_query_dist``), which downstream maps to depth = 0.
    """
    n_triangles = faces_info.verts_idx.shape[0]
    best_dist = max_query_dist
    best_dist_sq = best_dist * best_dist
    best_signed = max_query_dist

    node_stack = qd.Vector.zero(gs.qd_int, qd.static(_BVH_STACK_SIZE))
    node_stack[0] = 0
    stack_idx = 1

    while stack_idx > 0:
        stack_idx -= 1
        node_idx = node_stack[stack_idx]
        node = bvh_nodes[i_b, node_idx]

        if not func_sphere_intersects_aabb(probe_world, best_dist_sq, node.bound.min, node.bound.max):
            continue

        if node.left == -1:
            sorted_leaf_idx = node_idx - (n_triangles - 1)
            i_f = qd.cast(bvh_morton_codes[i_b, sorted_leaf_idx][1], gs.qd_int)
            i_g = faces_info.geom_idx[i_f]
            if not track_geom_mask[i_b, i_s, i_g]:
                continue

            tri = get_triangle_vertices(i_f, i_b, faces_info, verts_info, fixed_verts_state, free_verts_state)
            v0 = tri[:, 0]
            v1 = tri[:, 1]
            v2 = tri[:, 2]
            closest = closest_point_on_triangle(probe_world, v0, v1, v2)
            diff = probe_world - closest
            d_sq = diff.dot(diff)
            if d_sq < best_dist_sq:
                d = qd.sqrt(d_sq)
                fn = triangle_face_normal(v0, v1, v2)
                # Sign: probe outside if (probe - closest) aligns with outward face normal.
                sign_v = qd.select(diff.dot(fn) >= gs.qd_float(0.0), gs.qd_float(1.0), gs.qd_float(-1.0))
                best_signed = d * sign_v
                best_dist = d
                best_dist_sq = d_sq
        else:
            if stack_idx < qd.static(_BVH_STACK_SIZE - 2):
                node_stack[stack_idx] = node.left
                node_stack[stack_idx + 1] = node.right
                stack_idx += 2

    return best_signed


@qd.kernel(fastcache=False)
def _kernel_elastomer_probe_depth_bvh(
    probe_positions_local: qd.types.ndarray(),
    probe_sensor_idx: qd.types.ndarray(),
    probe_radii: qd.types.ndarray(),
    links_idx: qd.types.ndarray(),
    track_geom_mask: qd.types.ndarray(),
    max_query_dist: float,
    bvh_nodes: qd.template(),
    bvh_morton_codes: qd.template(),
    links_state: array_class.LinksState,
    faces_info: array_class.FacesInfo,
    verts_info: array_class.VertsInfo,
    fixed_verts_state: array_class.VertsState,
    free_verts_state: array_class.VertsState,
    probe_depth_buf: qd.types.ndarray(),
):
    """
    Per-probe contact depth from the rigid solver's global collision BVH, gated by ``track_geom_mask``.

    Mirrors ``_kernel_elastomer_probe_depth``'s output contract (write into ``probe_depth_buf``); the dilate
    accumulator consumes the same buffer downstream.
    """
    total_n_probes = probe_positions_local.shape[0]
    n_batches = probe_depth_buf.shape[0]

    for i_b, i_p in qd.ndrange(n_batches, total_n_probes):
        if probe_radii[i_p] <= gs.qd_float(0.0):
            probe_depth_buf[i_b, i_p] = gs.qd_float(0.0)
            continue
        i_s = probe_sensor_idx[i_p]
        sensor_link_idx = links_idx[i_s]
        link_pos = links_state.pos[sensor_link_idx, i_b]
        link_quat = links_state.quat[sensor_link_idx, i_b]
        probe_local = func_vec3_at(probe_positions_local, i_p)
        probe_world = link_pos + gu.qd_transform_by_quat(probe_local, link_quat)

        signed = _func_elastomer_min_signed_dist_bvh(
            i_b,
            i_s,
            probe_world,
            max_query_dist,
            bvh_nodes,
            bvh_morton_codes,
            faces_info,
            verts_info,
            fixed_verts_state,
            free_verts_state,
            track_geom_mask,
        )
        probe_depth_buf[i_b, i_p] = qd.max(gs.qd_float(0.0), -signed)


@qd.kernel(fastcache=True)
def _kernel_elastomer_probe_depth(
    probe_positions_local: qd.types.ndarray(),
    probe_sensor_idx: qd.types.ndarray(),
    probe_radii: qd.types.ndarray(),
    links_idx: qd.types.ndarray(),
    sensor_track_geom_start: qd.types.ndarray(),
    sensor_track_geom_n: qd.types.ndarray(),
    track_geom_idx: qd.types.ndarray(),
    track_geom_active_envs_mask: qd.types.ndarray(),
    links_state: array_class.LinksState,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    sdf_info: array_class.SDFInfo,
    probe_depth_buf: qd.types.ndarray(),
):
    """Per-probe contact depth from track-geom SDF, parallel over (env, probe).

    Writes only ``probe_depth_buf``; dilate accumulation is split into a separate target-major kernel that runs
    without atomics.
    """
    total_n_probes = probe_positions_local.shape[0]
    n_batches = probe_depth_buf.shape[0]

    for i_b, i_p in qd.ndrange(n_batches, total_n_probes):
        # Inactive filler probe: no SDF query, contributes no dilation.
        if probe_radii[i_p] <= gs.qd_float(0.0):
            probe_depth_buf[i_b, i_p] = gs.qd_float(0.0)
            continue
        i_s = probe_sensor_idx[i_p]
        sensor_link_idx = links_idx[i_s]
        link_pos = links_state.pos[sensor_link_idx, i_b]
        link_quat = links_state.quat[sensor_link_idx, i_b]
        probe_local = func_vec3_at(probe_positions_local, i_p)
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

        probe_depth_buf[i_b, i_p] = qd.max(gs.qd_float(0.0), -min_sdf)


@qd.kernel(fastcache=True)
def _kernel_elastomer_dilate_accumulate(
    use_grid_fft: qd.types.ndarray(),
    probe_positions_local: qd.types.ndarray(),
    probe_local_normal: qd.types.ndarray(),
    probe_sensor_idx: qd.types.ndarray(),
    probe_radii: qd.types.ndarray(),
    sensor_cache_start: qd.types.ndarray(),
    sensor_probe_start: qd.types.ndarray(),
    n_probes_per_sensor: qd.types.ndarray(),
    lambda_d: qd.types.ndarray(),
    dilate_scale: qd.types.ndarray(),
    normal_exponent: qd.types.ndarray(),
    probe_depth_buf: qd.types.ndarray(),
    output: qd.types.ndarray(),
):
    """Target-major dilate accumulator for non-grid sensors.

    Each (env, target_probe) thread sums Gaussian contributions from every in-contact source probe of its sensor
    into a register and writes once -- no atomic_add. Grid sensors are skipped (FFT path handles them). Output write
    is an OVERWRITE because output was pre-zeroed at step start and no other writer touches a non-grid sensor's range
    before shear-accumulate.
    """
    total_n_probes = probe_positions_local.shape[0]
    n_batches = probe_depth_buf.shape[0]

    for i_b, i_p in qd.ndrange(n_batches, total_n_probes):
        i_s = probe_sensor_idx[i_p]
        if use_grid_fft[i_s]:
            continue
        n_probes = n_probes_per_sensor[i_s]
        probe_start = sensor_probe_start[i_s]
        cache_start = sensor_cache_start[i_s]
        lam = lambda_d[i_s]
        scale = dilate_scale[i_s]
        n_exp = normal_exponent[i_s]
        _i_p = i_p - probe_start

        # Inactive filler probe (probe_radius == 0): reads zero, no dilation accumulated.
        if probe_radii[i_p] <= gs.qd_float(0.0):
            for k in qd.static(range(3)):
                output[cache_start + _i_p * 3 + k, i_b] = gs.qd_float(0.0)
            continue

        target_local = func_vec3_at(probe_positions_local, i_p)
        target_normal = func_vec3_at(probe_local_normal, i_p)

        acc = qd.Vector.zero(gs.qd_float, 3)
        for j in range(n_probes):
            j_p = probe_start + j
            src_depth = probe_depth_buf[i_b, j_p]
            if src_depth <= gs.qd_float(0.0):
                continue
            contribution = _func_elastomer_direct_dilate_contribution(
                func_vec3_at(probe_positions_local, j_p),
                target_local,
                target_normal,
                src_depth,
                lam,
                scale,
                n_exp,
            )
            for k in qd.static(range(3)):
                acc[k] = acc[k] + contribution[k]

        for k in qd.static(range(3)):
            output[cache_start + _i_p * 3 + k, i_b] = acc[k]


@qd.kernel(fastcache=True)
def _kernel_elastomer_surface_state_bvh(
    links_idx: qd.types.ndarray(),
    sensor_elastomer_geom_start: qd.types.ndarray(),
    sensor_elastomer_geom_n: qd.types.ndarray(),
    elastomer_geom_idx: qd.types.ndarray(),
    elastomer_geom_active_envs_mask: qd.types.ndarray(),
    bvh_chunk_sensor_idx: qd.types.ndarray(),
    bvh: ChunkedBVHData,
    pc_pos_link: qd.types.ndarray(),
    pc_active_envs_mask: qd.types.ndarray(),
    sdf_enter: qd.types.ndarray(),
    sdf_exit: qd.types.ndarray(),
    aabb_margin: float,
    links_state: array_class.LinksState,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    sdf_info: array_class.SDFInfo,
    surface_pos_sensor_buf: qd.types.ndarray(),
    surface_entry_pos_sensor_buf: qd.types.ndarray(),
    surface_depth_buf: qd.types.ndarray(),
    surface_initialized_buf: qd.types.ndarray(),
    surface_candidate_buf: qd.types.ndarray(),
):
    """Per-(env, chunk): compute the chunk-local query AABB in registers, BVH-traverse, and write
    per-candidate surface state.

    The AABB fill and BVH traversal share one kernel so the AABB stays in thread-local state instead of
    round-tripping through a (B, n_chunks, 3) buffer. No probe work happens here -- the shear contribution is
    accumulated in a separate target-major kernel that reads surface_pos_sensor_buf / surface_depth_buf /
    surface_entry_pos_sensor_buf.
    """
    n_batches = surface_pos_sensor_buf.shape[0]
    n_chunks = bvh_chunk_sensor_idx.shape[0]

    for i_b, i_c in qd.ndrange(n_batches, n_chunks):
        i_s = bvh_chunk_sensor_idx[i_c]

        # 1) Build the world-space elastomer-geom union AABB for sensor i_s, env i_b.
        wmin = qd.Vector([gs.qd_float(1e30), gs.qd_float(1e30), gs.qd_float(1e30)], dt=gs.qd_float)
        wmax = qd.Vector([gs.qd_float(-1e30), gs.qd_float(-1e30), gs.qd_float(-1e30)], dt=gs.qd_float)
        any_active = False
        gm_start = sensor_elastomer_geom_start[i_s]
        gm_n = sensor_elastomer_geom_n[i_s]
        for i_gm in range(gm_start, gm_start + gm_n):
            if not elastomer_geom_active_envs_mask[i_gm, i_b]:
                continue
            i_g = elastomer_geom_idx[i_gm]
            gmin = geoms_state.aabb_min[i_g, i_b]
            gmax = geoms_state.aabb_max[i_g, i_b]
            for k in qd.static(range(3)):
                if gmin[k] < wmin[k]:
                    wmin[k] = gmin[k]
                if gmax[k] > wmax[k]:
                    wmax[k] = gmax[k]
            any_active = True

        if not any_active:
            continue

        # 2) Expand by sdf_exit + margin so any point with sdf <= sdf_exit (the surface-state
        # exit threshold) is inside the AABB.
        expand = sdf_exit[i_s] + gs.qd_float(aabb_margin)
        for k in qd.static(range(3)):
            wmin[k] = wmin[k] - expand
            wmax[k] = wmax[k] + expand

        # 3) Transform 8 corners into the chunk's tracked-link local frame to get qmin/qmax.
        track_link_idx = bvh.chunk_link_idx[i_c]
        track_pos = links_state.pos[track_link_idx, i_b]
        track_quat = links_state.quat[track_link_idx, i_b]
        qmin = qd.Vector([gs.qd_float(1e30), gs.qd_float(1e30), gs.qd_float(1e30)], dt=gs.qd_float)
        qmax = qd.Vector([gs.qd_float(-1e30), gs.qd_float(-1e30), gs.qd_float(-1e30)], dt=gs.qd_float)
        for cx in qd.static(range(2)):
            for cy in qd.static(range(2)):
                for cz in qd.static(range(2)):
                    cw_x = wmax[0] if cx == 1 else wmin[0]
                    cw_y = wmax[1] if cy == 1 else wmin[1]
                    cw_z = wmax[2] if cz == 1 else wmin[2]
                    corner_world = qd.Vector([cw_x, cw_y, cw_z], dt=gs.qd_float)
                    corner_link = gu.qd_inv_transform_by_trans_quat(corner_world, track_pos, track_quat)
                    for k in qd.static(range(3)):
                        if corner_link[k] < qmin[k]:
                            qmin[k] = corner_link[k]
                        if corner_link[k] > qmax[k]:
                            qmax[k] = corner_link[k]

        # 4) BVH-traverse the chunk with the chunk-local query AABB. For each visited active point:
        # mark candidate, write point_sensor / depth, run anchor (enter/exit hysteresis).
        sensor_link_idx = links_idx[i_s]
        sensor_pos = links_state.pos[sensor_link_idx, i_b]
        sensor_quat = links_state.quat[sensor_link_idx, i_b]

        stack = qd.Vector.zero(gs.qd_int, qd.static(BVH_STACK_SIZE))
        stack[0] = bvh.chunk_node_start[i_c]
        stack_idx = 1

        while stack_idx > 0:
            stack_idx -= 1
            n = stack[stack_idx]
            bmin = func_vec3_at(bvh.node_min, n)
            bmax = func_vec3_at(bvh.node_max, n)
            if not func_aabb_intersects_aabb(bmin, bmax, qmin, qmax):
                continue
            left = bvh.node_left[n]
            if left == -1:
                pstart = bvh.node_leaf_start[n]
                pn = bvh.node_leaf_count[n]
                for j in range(pn):
                    i_o = bvh.leaf_elem_idx[pstart + j]
                    if not pc_active_envs_mask[i_o, i_b]:
                        continue
                    surface_candidate_buf[i_b, i_o] = True

                    point_link = func_vec3_at(pc_pos_link, i_o)
                    point_world = track_pos + gu.qd_transform_by_quat(point_link, track_quat)
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

                    surface_depth_buf[i_b, i_o] = qd.max(gs.qd_float(0.0), -min_sdf)

                    _func_elastomer_update_surface_anchor(
                        i_b,
                        i_o,
                        min_sdf,
                        point_sensor,
                        sdf_enter[i_s],
                        sdf_exit[i_s],
                        surface_entry_pos_sensor_buf,
                        surface_initialized_buf,
                    )
            else:
                right = bvh.node_right[n]
                # Median split bounds depth at log2(N / leaf_size) << BVH_STACK_SIZE; the guard mirrors the
                # global rigid-BVH kernel so a future build strategy can't silently overflow the stack.
                if stack_idx < qd.static(BVH_STACK_SIZE - 2):
                    stack[stack_idx] = left
                    stack[stack_idx + 1] = right
                    stack_idx += 2


@qd.kernel(fastcache=False)
def _kernel_elastomer_surface_state_via_global_bvh(
    links_idx: qd.types.ndarray(),
    sensor_elastomer_geom_start: qd.types.ndarray(),
    sensor_elastomer_geom_n: qd.types.ndarray(),
    elastomer_geom_idx: qd.types.ndarray(),
    elastomer_geom_active_envs_mask: qd.types.ndarray(),
    elastomer_candidate_geom_mask: qd.types.ndarray(),
    bvh_chunk_sensor_idx: qd.types.ndarray(),
    bvh: ChunkedBVHData,
    pc_pos_link: qd.types.ndarray(),
    pc_active_envs_mask: qd.types.ndarray(),
    sdf_enter: qd.types.ndarray(),
    sdf_exit: qd.types.ndarray(),
    aabb_margin: float,
    max_query_dist: float,
    global_bvh_nodes: qd.template(),
    global_bvh_morton_codes: qd.template(),
    links_state: array_class.LinksState,
    geoms_state: array_class.GeomsState,
    faces_info: array_class.FacesInfo,
    verts_info: array_class.VertsInfo,
    fixed_verts_state: array_class.VertsState,
    free_verts_state: array_class.VertsState,
    surface_pos_sensor_buf: qd.types.ndarray(),
    surface_entry_pos_sensor_buf: qd.types.ndarray(),
    surface_depth_buf: qd.types.ndarray(),
    surface_initialized_buf: qd.types.ndarray(),
    surface_candidate_buf: qd.types.ndarray(),
):
    """
    Raycast variant of ``_kernel_elastomer_surface_state_bvh``.

    Same outer (env, chunk) traversal over the point-cloud BVH per tracked link, but the inner signed-distance query
    at each PC point uses ``_func_elastomer_min_signed_dist_bvh`` over the rigid solver's global collision BVH (gated
    by ``elastomer_candidate_geom_mask``) instead of the analytic SDF. Output contract matches the SDF variant so the
    dilate / shear pipeline downstream is unchanged.
    """
    n_batches = surface_pos_sensor_buf.shape[0]
    n_chunks = bvh_chunk_sensor_idx.shape[0]

    for i_b, i_c in qd.ndrange(n_batches, n_chunks):
        i_s = bvh_chunk_sensor_idx[i_c]

        wmin = qd.Vector([gs.qd_float(1e30), gs.qd_float(1e30), gs.qd_float(1e30)], dt=gs.qd_float)
        wmax = qd.Vector([gs.qd_float(-1e30), gs.qd_float(-1e30), gs.qd_float(-1e30)], dt=gs.qd_float)
        any_active = False
        gm_start = sensor_elastomer_geom_start[i_s]
        gm_n = sensor_elastomer_geom_n[i_s]
        for i_gm in range(gm_start, gm_start + gm_n):
            if not elastomer_geom_active_envs_mask[i_gm, i_b]:
                continue
            i_g = elastomer_geom_idx[i_gm]
            gmin = geoms_state.aabb_min[i_g, i_b]
            gmax = geoms_state.aabb_max[i_g, i_b]
            for k in qd.static(range(3)):
                if gmin[k] < wmin[k]:
                    wmin[k] = gmin[k]
                if gmax[k] > wmax[k]:
                    wmax[k] = gmax[k]
            any_active = True

        if not any_active:
            continue

        expand = sdf_exit[i_s] + gs.qd_float(aabb_margin)
        for k in qd.static(range(3)):
            wmin[k] = wmin[k] - expand
            wmax[k] = wmax[k] + expand

        track_link_idx = bvh.chunk_link_idx[i_c]
        track_pos = links_state.pos[track_link_idx, i_b]
        track_quat = links_state.quat[track_link_idx, i_b]
        qmin = qd.Vector([gs.qd_float(1e30), gs.qd_float(1e30), gs.qd_float(1e30)], dt=gs.qd_float)
        qmax = qd.Vector([gs.qd_float(-1e30), gs.qd_float(-1e30), gs.qd_float(-1e30)], dt=gs.qd_float)
        for cx in qd.static(range(2)):
            for cy in qd.static(range(2)):
                for cz in qd.static(range(2)):
                    cw_x = wmax[0] if cx == 1 else wmin[0]
                    cw_y = wmax[1] if cy == 1 else wmin[1]
                    cw_z = wmax[2] if cz == 1 else wmin[2]
                    corner_world = qd.Vector([cw_x, cw_y, cw_z], dt=gs.qd_float)
                    corner_link = gu.qd_inv_transform_by_trans_quat(corner_world, track_pos, track_quat)
                    for k in qd.static(range(3)):
                        if corner_link[k] < qmin[k]:
                            qmin[k] = corner_link[k]
                        if corner_link[k] > qmax[k]:
                            qmax[k] = corner_link[k]

        sensor_link_idx = links_idx[i_s]
        sensor_pos = links_state.pos[sensor_link_idx, i_b]
        sensor_quat = links_state.quat[sensor_link_idx, i_b]

        stack = qd.Vector.zero(gs.qd_int, qd.static(BVH_STACK_SIZE))
        stack[0] = bvh.chunk_node_start[i_c]
        stack_idx = 1

        while stack_idx > 0:
            stack_idx -= 1
            n = stack[stack_idx]
            bmin = func_vec3_at(bvh.node_min, n)
            bmax = func_vec3_at(bvh.node_max, n)
            if not func_aabb_intersects_aabb(bmin, bmax, qmin, qmax):
                continue
            left = bvh.node_left[n]
            if left == -1:
                pstart = bvh.node_leaf_start[n]
                pn = bvh.node_leaf_count[n]
                for j in range(pn):
                    i_o = bvh.leaf_elem_idx[pstart + j]
                    if not pc_active_envs_mask[i_o, i_b]:
                        continue
                    surface_candidate_buf[i_b, i_o] = True

                    point_link = func_vec3_at(pc_pos_link, i_o)
                    point_world = track_pos + gu.qd_transform_by_quat(point_link, track_quat)
                    point_sensor = gu.qd_inv_transform_by_trans_quat(point_world, sensor_pos, sensor_quat)
                    for k in qd.static(range(3)):
                        surface_pos_sensor_buf[i_b, i_o, k] = point_sensor[k]

                    min_sdf = _func_elastomer_min_signed_dist_bvh(
                        i_b,
                        i_s,
                        point_world,
                        max_query_dist,
                        global_bvh_nodes,
                        global_bvh_morton_codes,
                        faces_info,
                        verts_info,
                        fixed_verts_state,
                        free_verts_state,
                        elastomer_candidate_geom_mask,
                    )

                    surface_depth_buf[i_b, i_o] = qd.max(gs.qd_float(0.0), -min_sdf)

                    _func_elastomer_update_surface_anchor(
                        i_b,
                        i_o,
                        min_sdf,
                        point_sensor,
                        sdf_enter[i_s],
                        sdf_exit[i_s],
                        surface_entry_pos_sensor_buf,
                        surface_initialized_buf,
                    )
            else:
                right = bvh.node_right[n]
                # Median split bounds depth at log2(N / leaf_size) << BVH_STACK_SIZE; the guard mirrors the
                # global rigid-BVH kernel so a future build strategy can't silently overflow the stack.
                if stack_idx < qd.static(BVH_STACK_SIZE - 2):
                    stack[stack_idx] = left
                    stack[stack_idx + 1] = right
                    stack_idx += 2


@qd.kernel(fastcache=True)
def _kernel_elastomer_shear_accumulate(
    probe_positions_local: qd.types.ndarray(),
    probe_local_normal: qd.types.ndarray(),
    probe_sensor_idx: qd.types.ndarray(),
    probe_radii: qd.types.ndarray(),
    sensor_cache_start: qd.types.ndarray(),
    sensor_probe_start: qd.types.ndarray(),
    sensor_pc_start: qd.types.ndarray(),
    lambda_s: qd.types.ndarray(),
    shear_scale: qd.types.ndarray(),
    eps: float,
    surface_pos_sensor_buf: qd.types.ndarray(),
    surface_entry_pos_sensor_buf: qd.types.ndarray(),
    surface_depth_buf: qd.types.ndarray(),
    shear_active_pc_idx: qd.types.ndarray(),
    shear_active_pc_count: qd.types.ndarray(),
    output: qd.types.ndarray(),
):
    """Target-major shear accumulator: per (env, target_probe), iterate over the sensor's compact active surface-point
    index and sum Gaussian contributions into a register, then += the result into ``output``.

    No atomic_add (each (i_b, i_p) thread owns its output slot). Consumes the compact index produced by
    ``_build_shear_active_pc_index`` (must run after the surface-state kernel AND after the post-kernel
    ``surface_initialized_buf &= candidate`` cleanup). Inner-loop cost is O(active_count[i_b, i_s]) rather than
    O(sensor_pc_n[i_s]), so the kernel scales with contact density rather than total point-cloud size.
    """
    total_n_probes = probe_positions_local.shape[0]
    n_batches = surface_pos_sensor_buf.shape[0]

    for i_b, i_p in qd.ndrange(n_batches, total_n_probes):
        i_s = probe_sensor_idx[i_p]
        scale = shear_scale[i_s]
        if scale <= gs.qd_float(0.0):
            continue
        # Inactive filler probe: dilate already wrote 0 to this output slot.
        if probe_radii[i_p] <= gs.qd_float(0.0):
            continue
        lam = lambda_s[i_s]
        cache_start = sensor_cache_start[i_s]
        _i_p = i_p - sensor_probe_start[i_s]
        pc_start = sensor_pc_start[i_s]
        n_active = shear_active_pc_count[i_b, i_s]

        probe_local = func_vec3_at(probe_positions_local, i_p)
        probe_normal = func_vec3_at(probe_local_normal, i_p)

        acc = qd.Vector.zero(gs.qd_float, 3)
        for j in range(n_active):
            i_o = shear_active_pc_idx[i_b, pc_start + j]
            depth = surface_depth_buf[i_b, i_o]
            if depth <= eps:
                continue
            point_sensor = qd.Vector(
                [
                    surface_pos_sensor_buf[i_b, i_o, 0],
                    surface_pos_sensor_buf[i_b, i_o, 1],
                    surface_pos_sensor_buf[i_b, i_o, 2],
                ],
                dt=gs.qd_float,
            )
            entry = qd.Vector(
                [
                    surface_entry_pos_sensor_buf[i_b, i_o, 0],
                    surface_entry_pos_sensor_buf[i_b, i_o, 1],
                    surface_entry_pos_sensor_buf[i_b, i_o, 2],
                ],
                dt=gs.qd_float,
            )
            contribution = _func_elastomer_direct_shear_contribution(
                point_sensor,
                entry,
                probe_local,
                probe_normal,
                depth,
                lam,
                scale,
                eps,
            )
            for k in qd.static(range(3)):
                acc[k] = acc[k] + contribution[k]

        for k in qd.static(range(3)):
            output[cache_start + _i_p * 3 + k, i_b] = output[cache_start + _i_p * 3 + k, i_b] + acc[k]


def _build_shear_active_pc_index(
    surface_initialized_buf: torch.Tensor,
    sensor_pc_start: torch.Tensor,
    sensor_pc_n: torch.Tensor,
    shear_scale: torch.Tensor,
    active_pc_idx: torch.Tensor,
    active_pc_count: torch.Tensor,
) -> None:
    """Build the compact per-(env, sensor) active surface-point index consumed by
    ``_kernel_elastomer_shear_accumulate``.

    Mutates ``active_pc_idx`` and ``active_pc_count`` in place.

    For each sensor ``s`` with ``shear_scale[s] > 0``, gathers the indices of True entries in
    ``surface_initialized_buf[:, pc_start[s] : pc_start[s] + pc_n[s]]`` into the per-sensor compact slice
    ``active_pc_idx[:, pc_start[s] : pc_start[s] + active_count[:, s]]``; the per-(env, sensor) active
    count is written to ``active_pc_count[:, s]``. Sensors with ``shear_scale == 0`` are skipped and
    their count is left at zero so the kernel's outer early-exit handles them with no extra work.

    Uses exclusive cumsum + ``torch.nonzero`` for the per-sensor scatter so per-env Python loops are
    avoided; cost is ~O(B * total_n_surface) torch ops over the whole pass.
    """
    active_pc_count.zero_()
    n_sensors = sensor_pc_start.shape[0]
    if n_sensors == 0:
        return
    # Single host sync up front so the per-sensor loop is metadata-only on the Python side.
    pc_starts = sensor_pc_start.tolist()
    pc_ns = sensor_pc_n.tolist()
    scales = shear_scale.tolist()
    idx_dtype = active_pc_idx.dtype
    for i_s in range(n_sensors):
        if scales[i_s] <= 0.0:
            continue
        pc_start = int(pc_starts[i_s])
        pc_n = int(pc_ns[i_s])
        if pc_n == 0:
            continue
        mask = surface_initialized_buf[:, pc_start : pc_start + pc_n]  # (B, pc_n) bool
        int_mask = mask.to(idx_dtype)
        write_pos = torch.cumsum(int_mask, dim=1) - int_mask  # exclusive cumsum
        active_pc_count[:, i_s] = int_mask.sum(dim=1)
        bs, js = torch.nonzero(mask, as_tuple=True)
        if bs.numel() > 0:
            active_pc_idx[bs, pc_start + write_pos[bs, js]] = (pc_start + js).to(idx_dtype)


def _elastomer_taxel_grid_fft_dilate(
    grid_fft_meta: list[GridFFTMeta],
    grid_fft_kernels_stacked: torch.Tensor,
    probe_depth_buf: torch.Tensor,
    probe_radii: torch.Tensor,
    grid_fft_buffer: torch.Tensor,
    dilate_scale: torch.Tensor,
    normal_exponent: torch.Tensor,
    grid_normal: torch.Tensor,
    grid_tangent_u: torch.Tensor,
    grid_tangent_v: torch.Tensor,
    grid_dilate_out_buffer: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """
    Elastomer marker dilation via 2D FFT in the validated probe tangent basis.

    All grid sensors share the global ``grid_fft_max_n`` (= last two dims of ``grid_fft_buffer``); their
    kernels are stacked into ``grid_fft_kernels_stacked`` of shape (n_grid, 3, fft_ny, fft_nx). The four heavy
    FFTs (fft of H, fft of H**normal_exponent, ifft for Ku/Kv/Kn) thus run as batched ops over the grid-sensor
    axis, dropping
    launches from 4*n_grid to 4. The H-fill and write-back stages remain per-sensor (small Python loops over
    view/copy and per-sensor tangent decomposition). Grid axes are ``(ny, nx)`` row-major throughout (matching
    the probe flat index ``iy * nx + ix``), so no transpose is needed on either the fill or write-back side.
    """
    if not grid_fft_meta:
        return
    n_batches = probe_depth_buf.shape[0]
    fft_ny, fft_nx = grid_fft_buffer.shape[-2], grid_fft_buffer.shape[-1]

    # 1) Fill the active region of the (B, n_grid, fft_ny, fft_nx) depth buffer. The zero-padding region is never
    # written here and stays zero from allocation, so no per-step ``zero_()`` is needed.
    for grid_pos, meta in enumerate(grid_fft_meta):
        depth_slice = probe_depth_buf[:, meta.probe_start : meta.probe_start + meta.g_ny * meta.g_nx]
        grid_fft_buffer[:, grid_pos, : meta.g_ny, : meta.g_nx].copy_(depth_slice.view(n_batches, meta.g_ny, meta.g_nx))

    # 2) Batched real FFTs across (B, n_grid). Inputs are real so ``rfft2`` (half spectrum) is ~2x cheaper than the
    # full complex ``fft2``. Kernels broadcast over B when multiplying.
    H_fft = torch.fft.rfft2(grid_fft_buffer)
    # The normal channel follows depth ** normal_exponent, so it convolves the per-grid powered depth field;
    # the tangential (u, v) channels stay linear in depth and convolve the raw field H.
    exps = normal_exponent[[meta.sensor_idx for meta in grid_fft_meta]].reshape(1, -1, 1, 1)
    Hp_fft = torch.fft.rfft2(grid_fft_buffer.pow(exps))
    Ku_all = grid_fft_kernels_stacked[:, 0]  # (n_grid, fft_ny, fft_nx // 2 + 1) complex
    Kv_all = grid_fft_kernels_stacked[:, 1]
    Kn_all = grid_fft_kernels_stacked[:, 2]
    disp_u_all = torch.fft.irfft2(H_fft * Ku_all, s=(fft_ny, fft_nx))  # (B, n_grid, fft_ny, fft_nx)
    disp_v_all = torch.fft.irfft2(H_fft * Kv_all, s=(fft_ny, fft_nx))
    disp_n_all = torch.fft.irfft2(Hp_fft * Kn_all, s=(fft_ny, fft_nx))

    # 3) Per-sensor write-back: slice to (g_ny, g_nx), apply scale + tangent decomposition, copy
    # into the sensor's output range. Tangent vectors are per-sensor so can't trivially batch here.
    for grid_pos, meta in enumerate(grid_fft_meta):
        sensor_idx, g_ny, g_nx = meta.sensor_idx, meta.g_ny, meta.g_nx
        probe_start, cache_start = meta.probe_start, meta.cache_start
        scale_s = dilate_scale[sensor_idx]
        disp_u = disp_u_all[:, grid_pos, :g_ny, :g_nx] * scale_s
        disp_v = disp_v_all[:, grid_pos, :g_ny, :g_nx] * scale_s
        disp_n = disp_n_all[:, grid_pos, :g_ny, :g_nx] * scale_s
        # (B, g_ny, g_nx) reshapes directly to the probe flat index iy*nx+ix -- no transpose.
        disp_u_flat = disp_u.reshape(n_batches, -1)
        disp_v_flat = disp_v.reshape(n_batches, -1)
        disp_n_flat = disp_n.reshape(n_batches, -1)
        grid_size = g_ny * g_nx * 3
        out_block = grid_dilate_out_buffer[:, :grid_size]
        tangent_u = grid_tangent_u[sensor_idx]
        tangent_v = grid_tangent_v[sensor_idx]
        normal = grid_normal[sensor_idx]
        # Zero inactive filler probes (probe_radius == 0): they are non-sources, but the FFT still smears
        # neighbour dilation into their cells, so mask the per-probe write-back.
        active = (probe_radii[probe_start : probe_start + g_ny * g_nx] > 0.0).to(disp_u_flat.dtype)
        for k in range(3):
            out_block[:, k:grid_size:3] = (
                disp_u_flat * tangent_u[k] + disp_v_flat * tangent_v[k] + disp_n_flat * normal[k]
            ) * active
        output[cache_start : cache_start + grid_size].copy_(out_block.T)


@dataclass
class ElastomerTaxelSensorMetadata(
    GridFFTConvMetadataMixin,
    ContactDepthQueryMetadataMixin,
    PointCloudTactileSharedMetadata,
    ProbesWithNormalSensorMetadataMixin,
):
    track_geom_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    track_geom_active_envs_mask: torch.Tensor = make_tensor_field((0, 0), dtype_factory=lambda: gs.tc_bool)
    sensor_track_geom_start: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    sensor_track_geom_n: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)

    elastomer_geom_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    elastomer_geom_active_envs_mask: torch.Tensor = make_tensor_field((0, 0), dtype_factory=lambda: gs.tc_bool)
    sensor_elastomer_geom_start: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    sensor_elastomer_geom_n: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)

    # Per-(B, sensor, geom) bitmask of elastomer (sensor-own) geoms, used by the global-BVH surface-state kernel
    # to gate triangles back to the sensor's elastomer surface. Separate from ``sensor_candidate_geom_mask`` which
    # gates by tracked-object geoms for the probe-depth kernel.
    elastomer_candidate_geom_mask: torch.Tensor = make_tensor_field((0, 0, 0), dtype_factory=lambda: gs.tc_bool)

    lambda_d: torch.Tensor = make_tensor_field((0,))
    lambda_s: torch.Tensor = make_tensor_field((0,))
    dilate_scale: torch.Tensor = make_tensor_field((0,))
    shear_scale: torch.Tensor = make_tensor_field((0,))
    normal_exponent: torch.Tensor = make_tensor_field((0,))
    elastomer_contact_sdf_enter: torch.Tensor = make_tensor_field((0,))
    elastomer_contact_sdf_exit: torch.Tensor = make_tensor_field((0,))

    probe_depth_buf: torch.Tensor = make_tensor_field((0, 0))
    surface_pos_sensor_buf: torch.Tensor = make_tensor_field((0, 0, 3))
    surface_entry_pos_sensor_buf: torch.Tensor = make_tensor_field((0, 0, 3))
    surface_depth_buf: torch.Tensor = make_tensor_field((0, 0))
    surface_initialized_buf: torch.Tensor = make_tensor_field((0, 0), dtype_factory=lambda: gs.tc_bool)

    # Per-(env, pc-row) BVH-candidate flag, zeroed each step and written True by the surface-state
    # kernel for every visited active point. Post-kernel torch ops use ``!candidate`` to invalidate
    # stale surface_initialized / surface_entry_pos for points the BVH skipped this step.
    surface_candidate_buf: torch.Tensor = make_tensor_field((0, 0), dtype_factory=lambda: gs.tc_bool)

    # Compact per-(env, sensor) active surface-point index, rebuilt every step right after the
    # ``surface_initialized_buf &= candidate`` cleanup and consumed by ``_kernel_elastomer_shear_accumulate``.
    # For sensor ``s`` in env ``i_b``, the first ``shear_active_pc_count[i_b, s]`` entries of
    # ``shear_active_pc_idx[i_b, sensor_pc_start[s]:]`` hold the global pc-row indices whose
    # ``surface_initialized_buf`` is True. Sensors with ``shear_scale == 0`` have count = 0.
    shear_active_pc_idx: torch.Tensor = make_tensor_field((0, 0), dtype_factory=lambda: gs.tc_int)
    shear_active_pc_count: torch.Tensor = make_tensor_field((0, 0), dtype_factory=lambda: gs.tc_int)

    # Per-sensor flag selecting the FFT dilation path vs the direct (non-grid) dilation kernel.
    use_grid_fft: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_bool)
    # Per-grid-FFT-sensor tangent basis, consumed by the dilation write-back. See ``GridFFTMeta`` for the per-sensor
    # ``grid_fft_meta`` record layout.
    grid_normal: torch.Tensor = make_tensor_field((0, 3))
    grid_tangent_u: torch.Tensor = make_tensor_field((0, 3))
    grid_tangent_v: torch.Tensor = make_tensor_field((0, 3))
    # Scratch for the per-sensor tangent-decomposition write-back, lazily grown to the largest grid.
    grid_dilate_out_buffer: torch.Tensor = make_tensor_field((0, 0))

    # True iff at least one configured ElastomerTaxel has shear_scale > 0. Set during build by OR-ing
    # each sensor's value, so per-step gating avoids an O(n_sensors) reduction + device sync.
    any_shear: bool = False


class ElastomerTaxelSensor(
    ContactDepthQuerySensorMixin,
    PointCloudTactileSensorMixin[ElastomerTaxelSensorMetadata],
    ProbesWithNormalSensorMixin[ElastomerTaxelSensorMetadata],
    RigidSensorMixin[ElastomerTaxelSensorMetadata],
    SimpleSensor[ElastomerTaxelSensorOptions, RaycastContext, ElastomerTaxelSensorMetadata],
):
    def __init__(
        self,
        options: ElastomerTaxelSensorOptions,
        idx: int,
        shared_context,
        shared_metadata,
        manager: "SensorManager",
    ):
        super().__init__(options, idx, shared_context, shared_metadata, manager)
        # FFT-grid eligibility check (flat pos/normals are already populated by the base mixins). 2D layouts with
        # non-degenerate spacing use the FFT dilation path; strictly irregular grids still take that path with
        # averaged metadata and only emit a warning.
        self._is_grid = len(self._probe_layout_shape) == 2
        _, _, self._use_grid_fft, is_grid_regular, grid_normal, grid_tangent_u, grid_tangent_v, grid_spacing = (
            normalize_grid_probe_layout(
                np.asarray(options.probe_local_pos, dtype=gs.np_float),
                np.asarray(options.probe_local_normal, dtype=gs.np_float),
                self._is_grid,
            )
        )
        self._grid_normal = torch.tensor(grid_normal, dtype=gs.tc_float, device=gs.device)
        self._grid_tangent_u = torch.tensor(grid_tangent_u, dtype=gs.tc_float, device=gs.device)
        self._grid_tangent_v = torch.tensor(grid_tangent_v, dtype=gs.tc_float, device=gs.device)
        self._grid_spacing = torch.tensor(grid_spacing, dtype=gs.tc_float, device=gs.device)

        if self._use_grid_fft and not is_grid_regular:
            gs.logger.warning(
                "ElastomerTaxel grid is not strictly regular (uniform spacing, uniform normals, orthogonal "
                "tangents); FFT dilation will use averaged spacing and normal as a best-fit approximation."
            )

    def build(self):
        super().build()

        solver = self._shared_metadata.solver
        B = self._manager._sim._B
        if self._link is None:
            gs.raise_exception("ElastomerTaxel must be attached to a rigid link with collision geometry.")
        # The class-wide contact_depth_query backend is resolved + activated by ContactDepthQuerySensorMixin.build.

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
        self._shared_metadata.normal_exponent = concat_with_tensor(
            self._shared_metadata.normal_exponent, float(self._options.normal_exponent), expand=(1,)
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
        if float(self._options.shear_scale) > 0.0:
            self._shared_metadata.any_shear = True

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

        self._shared_metadata.surface_candidate_buf = torch.zeros(
            (B, total_n_surface), dtype=gs.tc_bool, device=gs.device
        )

        # Compact active-point index for the shear accumulator. Re-allocated on each ElastomerTaxel build so the
        # ``(B, total_n_surface)`` idx buffer and ``(B, n_sensors)`` count buffer absorb the newly registered sensor.
        # Both are allocated unconditionally (zero-init); the per-step build at ``_build_shear_active_pc_index``
        # leaves entries for non-shear sensors at count == 0, so unread regions remain harmless zeros.
        n_sensors_built = self._shared_metadata.n_probes_per_sensor.shape[0]
        self._shared_metadata.shear_active_pc_idx = torch.zeros((B, total_n_surface), dtype=gs.tc_int, device=gs.device)
        self._shared_metadata.shear_active_pc_count = torch.zeros(
            (B, n_sensors_built), dtype=gs.tc_int, device=gs.device
        )

        # Build the (B, n_sensors, n_geoms) candidate-geom masks scattered from track_geom_idx (probe-depth) and
        # elastomer_geom_idx (surface-anchor). Only needed in raycast mode but allocated cheaply (bool, total
        # scene-geom count) so we tolerate the small idle cost in sdf mode.
        if self._shared_metadata.contact_depth_query == "raycast":
            n_geoms = solver.n_geoms
            self._shared_metadata.sensor_candidate_geom_mask = _build_candidate_geom_mask(
                B,
                n_sensors_built,
                n_geoms,
                self._shared_metadata.sensor_track_geom_start,
                self._shared_metadata.sensor_track_geom_n,
                self._shared_metadata.track_geom_idx,
            )
            self._shared_metadata.elastomer_candidate_geom_mask = _build_candidate_geom_mask(
                B,
                n_sensors_built,
                n_geoms,
                self._shared_metadata.sensor_elastomer_geom_start,
                self._shared_metadata.sensor_elastomer_geom_n,
                self._shared_metadata.elastomer_geom_idx,
            )

        self._shared_metadata.use_grid_fft = concat_with_tensor(
            self._shared_metadata.use_grid_fft, self._use_grid_fft, expand=(1,)
        )

        grid_normal = torch.zeros(3, dtype=gs.tc_float, device=gs.device)
        grid_tangent_u = torch.zeros(3, dtype=gs.tc_float, device=gs.device)
        grid_tangent_v = torch.zeros(3, dtype=gs.tc_float, device=gs.device)
        if self._use_grid_fft:
            nx, ny = int(self._probe_layout_shape[1]), int(self._probe_layout_shape[0])
            grid_normal = self._grid_normal
            grid_tangent_u = self._grid_tangent_u
            grid_tangent_v = self._grid_tangent_v
            spacing_u, spacing_v = float(self._grid_spacing[0].item()), float(self._grid_spacing[1].item())
            # FFT size is (ny, nx) row-major. Sizing each axis to ``2n - 1`` (the full linear-convolution support)
            # rounded up to a power of 2 guarantees zero circular wraparound regardless of the dilation kernel's
            # decay -- the ``x*g`` / ``y*g`` first-moment kernels decay slower than the Gaussian itself.
            this_fft_n = (next_pow2(2 * ny - 1), next_pow2(2 * nx - 1))
            cache_start_py = int(self._shared_metadata.sensor_cache_start[self._idx].item())
            register_grid_fft_sensor(
                self._shared_metadata,
                meta_entry=GridFFTMeta(
                    sensor_idx=self._idx,
                    g_ny=ny,
                    g_nx=nx,
                    probe_start=self._probe_start_idx,
                    cache_start=cache_start_py,
                    lambda_d=float(self._options.lambda_d),
                    spacing_u=spacing_u,
                    spacing_v=spacing_v,
                ),
                this_fft_n=this_fft_n,
                kernel_builder=_dilate_kernel_builder,
                n_buffer_channels=0,
                batch_size=B,
            )
            grid_size = nx * ny * 3
            out_buf = self._shared_metadata.grid_dilate_out_buffer
            if out_buf.numel() == 0 or out_buf.shape[1] < grid_size:
                self._shared_metadata.grid_dilate_out_buffer = torch.empty(
                    (B, max(out_buf.shape[1] if out_buf.numel() > 0 else 0, grid_size)),
                    dtype=gs.tc_float,
                    device=gs.device,
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

    def _get_return_format(self) -> tuple[int, ...]:
        return (*self._probe_layout_shape, 3)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def reset(cls, shared_metadata: ElastomerTaxelSensorMetadata, shared_ground_truth_cache: torch.Tensor, envs_idx):
        super().reset(shared_metadata, shared_ground_truth_cache, envs_idx)
        # probe_depth_buf is overwritten every step; surface_pos/entry/depth are only consumed where
        # surface_initialized=True so they're implicitly invalidated by clearing it; surface_candidate_buf
        # is .zero_()'d at step start.
        shared_metadata.surface_initialized_buf[envs_idx, :] = False

    @classmethod
    def _update_current_timestep_data(
        cls,
        shared_context: RaycastContext,
        shared_metadata: ElastomerTaxelSensorMetadata,
        current_ground_truth_data_T: torch.Tensor,
        ground_truth_data_timeline: "TensorRingBuffer | None",
        measured_data_timeline: "TensorRingBuffer",
    ):
        solver = shared_metadata.solver
        # No pre-zeros: probe_depth is fully overwritten by _kernel_elastomer_probe_depth;
        # current_ground_truth_data_T is fully overwritten by FFT-dilate union dilate-accumulate (then
        # shear-accumulate += on top); surface_depth_buf is only read where surface_initialized=True,
        # which is set in lockstep with that same depth write; measured is .copy_'d at the end.
        measured = measured_data_timeline.at(0, copy=False)

        if (shared_metadata.contact_depth_query or "sdf") == "sdf":
            _kernel_elastomer_probe_depth(
                shared_metadata.probe_positions,
                shared_metadata.probe_sensor_idx,
                shared_metadata.probe_radii,
                shared_metadata.links_idx,
                shared_metadata.sensor_track_geom_start,
                shared_metadata.sensor_track_geom_n,
                shared_metadata.track_geom_idx,
                shared_metadata.track_geom_active_envs_mask,
                solver.links_state,
                solver.geoms_state,
                solver.geoms_info,
                solver.collider._sdf._sdf_info,
                shared_metadata.probe_depth_buf,
            )
        else:
            _kernel_elastomer_probe_depth_bvh(
                shared_metadata.probe_positions,
                shared_metadata.probe_sensor_idx,
                shared_metadata.probe_radii,
                shared_metadata.links_idx,
                shared_metadata.sensor_candidate_geom_mask,
                _ELASTOMER_RAYCAST_QUERY_DIST,
                shared_context.collision_bvh_context.bvh.nodes,
                shared_context.collision_bvh_context.bvh.morton_codes,
                solver.links_state,
                solver.faces_info,
                solver.verts_info,
                solver.fixed_verts_state,
                solver.free_verts_state,
                shared_metadata.probe_depth_buf,
            )
        _kernel_elastomer_dilate_accumulate(
            shared_metadata.use_grid_fft,
            shared_metadata.probe_positions,
            shared_metadata.probe_local_normal,
            shared_metadata.probe_sensor_idx,
            shared_metadata.probe_radii,
            shared_metadata.sensor_cache_start,
            shared_metadata.sensor_probe_start,
            shared_metadata.n_probes_per_sensor,
            shared_metadata.lambda_d,
            shared_metadata.dilate_scale,
            shared_metadata.normal_exponent,
            shared_metadata.probe_depth_buf,
            current_ground_truth_data_T,
        )
        # FFT runs after the qd dilate kernel: on Metal, write-only kernel outputs zero unwritten slots on copy-back,
        # which would erase the grid range the FFT just wrote.
        _elastomer_taxel_grid_fft_dilate(
            shared_metadata.grid_fft_meta,
            shared_metadata.grid_fft_kernels_stacked,
            shared_metadata.probe_depth_buf,
            shared_metadata.probe_radii,
            shared_metadata.grid_fft_buffer,
            shared_metadata.dilate_scale,
            shared_metadata.normal_exponent,
            shared_metadata.grid_normal,
            shared_metadata.grid_tangent_u,
            shared_metadata.grid_tangent_v,
            shared_metadata.grid_dilate_out_buffer,
            current_ground_truth_data_T,
        )
        if shared_metadata.any_shear:
            bvh = shared_metadata.pc_bvh
            shared_metadata.surface_candidate_buf.zero_()
            if (shared_metadata.contact_depth_query or "sdf") == "sdf":
                _kernel_elastomer_surface_state_bvh(
                    shared_metadata.links_idx,
                    shared_metadata.sensor_elastomer_geom_start,
                    shared_metadata.sensor_elastomer_geom_n,
                    shared_metadata.elastomer_geom_idx,
                    shared_metadata.elastomer_geom_active_envs_mask,
                    bvh.chunk_sensor_idx,
                    bvh.kernel_bvh,
                    shared_metadata.pc_pos_link,
                    shared_metadata.pc_active_envs_mask,
                    shared_metadata.elastomer_contact_sdf_enter,
                    shared_metadata.elastomer_contact_sdf_exit,
                    _ELASTOMER_QUERY_AABB_MARGIN,
                    solver.links_state,
                    solver.geoms_state,
                    solver.geoms_info,
                    solver.collider._sdf._sdf_info,
                    shared_metadata.surface_pos_sensor_buf,
                    shared_metadata.surface_entry_pos_sensor_buf,
                    shared_metadata.surface_depth_buf,
                    shared_metadata.surface_initialized_buf,
                    shared_metadata.surface_candidate_buf,
                )
            else:
                _kernel_elastomer_surface_state_via_global_bvh(
                    shared_metadata.links_idx,
                    shared_metadata.sensor_elastomer_geom_start,
                    shared_metadata.sensor_elastomer_geom_n,
                    shared_metadata.elastomer_geom_idx,
                    shared_metadata.elastomer_geom_active_envs_mask,
                    shared_metadata.elastomer_candidate_geom_mask,
                    bvh.chunk_sensor_idx,
                    bvh.kernel_bvh,
                    shared_metadata.pc_pos_link,
                    shared_metadata.pc_active_envs_mask,
                    shared_metadata.elastomer_contact_sdf_enter,
                    shared_metadata.elastomer_contact_sdf_exit,
                    _ELASTOMER_QUERY_AABB_MARGIN,
                    _ELASTOMER_RAYCAST_QUERY_DIST,
                    shared_context.collision_bvh_context.bvh.nodes,
                    shared_context.collision_bvh_context.bvh.morton_codes,
                    solver.links_state,
                    solver.geoms_state,
                    solver.faces_info,
                    solver.verts_info,
                    solver.fixed_verts_state,
                    solver.free_verts_state,
                    shared_metadata.surface_pos_sensor_buf,
                    shared_metadata.surface_entry_pos_sensor_buf,
                    shared_metadata.surface_depth_buf,
                    shared_metadata.surface_initialized_buf,
                    shared_metadata.surface_candidate_buf,
                )
            # Invalidate stale surface state for points the BVH did not visit. surface_initialized
            # and entry-pos survive across steps; depth/pos are gated by initialized downstream so
            # they don't need clearing. The shear accumulator below reads from a compact index
            # rebuilt from surface_initialized -- without this step, stale True from a prior step
            # would inject phantom contributions.
            cand = shared_metadata.surface_candidate_buf
            shared_metadata.surface_initialized_buf &= cand
            # Implicit bool->float broadcast zeros entries where cand=False, no `~` allocation.
            shared_metadata.surface_entry_pos_sensor_buf.mul_(cand.unsqueeze(-1))
            _build_shear_active_pc_index(
                shared_metadata.surface_initialized_buf,
                shared_metadata.sensor_pc_start,
                shared_metadata.sensor_pc_n,
                shared_metadata.shear_scale,
                shared_metadata.shear_active_pc_idx,
                shared_metadata.shear_active_pc_count,
            )
            _kernel_elastomer_shear_accumulate(
                shared_metadata.probe_positions,
                shared_metadata.probe_local_normal,
                shared_metadata.probe_sensor_idx,
                shared_metadata.probe_radii,
                shared_metadata.sensor_cache_start,
                shared_metadata.sensor_probe_start,
                shared_metadata.sensor_pc_start,
                shared_metadata.lambda_s,
                shared_metadata.shear_scale,
                gs.EPS,
                shared_metadata.surface_pos_sensor_buf,
                shared_metadata.surface_entry_pos_sensor_buf,
                shared_metadata.surface_depth_buf,
                shared_metadata.shear_active_pc_idx,
                shared_metadata.shear_active_pc_count,
                current_ground_truth_data_T,
            )

        if ground_truth_data_timeline is not None:
            ground_truth_data_timeline.at(0, copy=False).copy_(current_ground_truth_data_T.T)
        measured.copy_(current_ground_truth_data_T.T)

    def _draw_debug(self, context: "RasterizerContext"):
        def mask(envs_idx):
            disp = self.read_ground_truth(envs_idx)
            if self._options.history_length > 0:
                disp = disp.select(1 if self._manager._sim.n_envs > 0 else 0, -1)
            return torch.linalg.norm(disp, dim=-1) >= gs.EPS

        self._draw_debug_probes(context, self._tactile_color_groups_fn(mask))
