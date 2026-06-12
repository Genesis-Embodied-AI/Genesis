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


_POINT_CLOUD_BVH_LEAF_SIZE = 8
_POINT_CLOUD_BVH_STACK_SIZE = 32
_ELASTOMER_QUERY_AABB_MARGIN = 1e-3


def _build_static_chunk_bvh(points: np.ndarray, global_rows: np.ndarray, leaf_size: int):
    """Median-split AABB BVH over ``points`` (a single tracked link's local-frame point cloud).

    Leaves carry the caller-provided ``global_rows`` (absolute rows into ``pc_pos_link``); the
    kernel can therefore index directly into the per-class point-cloud tensors with no extra
    indirection. Internal nodes use -1 for ``node_left`` / ``node_right``.
    """
    node_min: list[np.ndarray] = []
    node_max: list[np.ndarray] = []
    node_left: list[int] = []
    node_right: list[int] = []
    node_point_start: list[int] = []
    node_point_n: list[int] = []
    point_idx: list[int] = []

    def _alloc() -> int:
        i = len(node_min)
        node_min.append(np.zeros(3, dtype=np.float32))
        node_max.append(np.zeros(3, dtype=np.float32))
        node_left.append(-1)
        node_right.append(-1)
        node_point_start.append(-1)
        node_point_n.append(0)
        return i

    def _build(rows: np.ndarray, pts: np.ndarray) -> int:
        nid = _alloc()
        bmin = pts.min(axis=0).astype(np.float32)
        bmax = pts.max(axis=0).astype(np.float32)
        node_min[nid] = bmin
        node_max[nid] = bmax
        if rows.shape[0] <= leaf_size:
            start = len(point_idx)
            point_idx.extend(int(r) for r in rows)
            node_point_start[nid] = start
            node_point_n[nid] = int(rows.shape[0])
            return nid
        axis = int(np.argmax(bmax - bmin))
        order = np.argsort(pts[:, axis], kind="stable")
        mid = order.shape[0] // 2
        node_left[nid] = _build(rows[order[:mid]], pts[order[:mid]])
        node_right[nid] = _build(rows[order[mid:]], pts[order[mid:]])
        return nid

    if points.shape[0] == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
        )

    root = _build(global_rows.astype(np.int32, copy=False), points.astype(np.float32, copy=False))
    assert root == 0
    return (
        np.stack(node_min, axis=0),
        np.stack(node_max, axis=0),
        np.asarray(node_left, dtype=np.int32),
        np.asarray(node_right, dtype=np.int32),
        np.asarray(node_point_start, dtype=np.int32),
        np.asarray(node_point_n, dtype=np.int32),
        np.asarray(point_idx, dtype=np.int32),
    )


@dataclass
class PointCloudBVH:
    """Static link-local BVH over the tracked point clouds of one sensor class.

    One chunk per (sensor, tracked_link); each chunk is built once in that link's LOCAL frame at
    scene-build time and never rebuilt. Queries must transform into a chunk's link-local frame at
    query time. Chunk nodes are flat-packed across all chunks: chunk_node_start/n delimits each
    chunk's contiguous run; node_left/right are ABSOLUTE indices into the flat node tensors (-1
    for leaves). Leaves' point_idx values are absolute rows into pc_pos_link / pc_active_envs_mask
    / pc_normal_link, so a leaf hit resolves to per-point data with one indirection.
    """

    sensor_chunk_start: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    sensor_chunk_n: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)

    chunk_link_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    # Inverse of sensor_chunk_start/n: chunk_sensor_idx[i_c] is the owning sensor's index. Enables
    # (env, chunk)-parallel kernels without redundant per-thread scans of sensor_chunk_start.
    chunk_sensor_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    chunk_node_start: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    chunk_node_n: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)

    node_min: torch.Tensor = make_tensor_field((0, 3))
    node_max: torch.Tensor = make_tensor_field((0, 3))
    node_left: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    node_right: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    node_point_start: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    node_point_n: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)

    point_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)

    def append_sensor(self, *, pc_start_row: int, idx_cat: torch.Tensor, pos_cat: torch.Tensor) -> None:
        """Build per-tracked-link chunks for one sensor and append into the flat tensors.

        Must be called immediately after extending ``pc_pos_link`` by ``pos_cat`` — each leaf's
        ``point_idx`` entry is ``pc_start_row + local_row`` and must address the freshly-grown
        rows.
        """
        n_local = int(pos_cat.shape[0])
        if n_local == 0:
            gs.raise_exception("PointCloudBVH.append_sensor called with empty point cloud.")

        idx_np = tensor_to_array(idx_cat).astype(np.int64)
        pos_np = tensor_to_array(pos_cat).astype(np.float32, copy=False)
        unique_links = np.unique(idx_np)

        chunk_start_for_sensor = int(self.chunk_link_idx.shape[0])
        node_offset = int(self.node_min.shape[0])
        point_offset = int(self.point_idx.shape[0])

        new_chunk_link_idx: list[int] = []
        new_chunk_node_start: list[int] = []
        new_chunk_node_n: list[int] = []
        all_node_min: list[np.ndarray] = []
        all_node_max: list[np.ndarray] = []
        all_node_left: list[np.ndarray] = []
        all_node_right: list[np.ndarray] = []
        all_node_point_start: list[np.ndarray] = []
        all_node_point_n: list[np.ndarray] = []
        all_point_idx: list[np.ndarray] = []

        for link_idx in unique_links:
            local_rows = np.nonzero(idx_np == int(link_idx))[0].astype(np.int32)
            global_rows = (int(pc_start_row) + local_rows).astype(np.int32)
            pts_link = pos_np[local_rows]

            nmin, nmax, nleft, nright, npstart, npn, pidx = _build_static_chunk_bvh(
                pts_link, global_rows, _POINT_CLOUD_BVH_LEAF_SIZE
            )

            new_chunk_link_idx.append(int(link_idx))
            new_chunk_node_start.append(node_offset)
            new_chunk_node_n.append(int(nmin.shape[0]))

            all_node_min.append(nmin)
            all_node_max.append(nmax)
            # Rebase intra-chunk child / leaf-start indices into the flat tensors' absolute space.
            all_node_left.append(np.where(nleft >= 0, nleft + node_offset, nleft).astype(np.int32))
            all_node_right.append(np.where(nright >= 0, nright + node_offset, nright).astype(np.int32))
            all_node_point_start.append(np.where(npn > 0, npstart + point_offset, npstart).astype(np.int32))
            all_node_point_n.append(npn)
            all_point_idx.append(pidx)

            node_offset += int(nmin.shape[0])
            point_offset += int(pidx.shape[0])

        nm = torch.tensor(np.concatenate(all_node_min, axis=0), dtype=gs.tc_float, device=gs.device)
        nx = torch.tensor(np.concatenate(all_node_max, axis=0), dtype=gs.tc_float, device=gs.device)
        nl = torch.tensor(np.concatenate(all_node_left, axis=0), dtype=gs.tc_int, device=gs.device)
        nr = torch.tensor(np.concatenate(all_node_right, axis=0), dtype=gs.tc_int, device=gs.device)
        nps = torch.tensor(np.concatenate(all_node_point_start, axis=0), dtype=gs.tc_int, device=gs.device)
        npn_t = torch.tensor(np.concatenate(all_node_point_n, axis=0), dtype=gs.tc_int, device=gs.device)
        pidx_t = torch.tensor(np.concatenate(all_point_idx, axis=0), dtype=gs.tc_int, device=gs.device)
        cli = torch.tensor(new_chunk_link_idx, dtype=gs.tc_int, device=gs.device)
        cns = torch.tensor(new_chunk_node_start, dtype=gs.tc_int, device=gs.device)
        cnn = torch.tensor(new_chunk_node_n, dtype=gs.tc_int, device=gs.device)
        # Sensor index for this batch of chunks = current sensor count (the entry we're about to add).
        sensor_idx_for_chunks = int(self.sensor_chunk_start.shape[0])
        csi = torch.full((len(unique_links),), sensor_idx_for_chunks, dtype=gs.tc_int, device=gs.device)

        self.node_min = concat_with_tensor(self.node_min, nm, expand=(nm.shape[0], 3))
        self.node_max = concat_with_tensor(self.node_max, nx, expand=(nx.shape[0], 3))
        self.node_left = concat_with_tensor(self.node_left, nl, expand=(nl.shape[0],))
        self.node_right = concat_with_tensor(self.node_right, nr, expand=(nr.shape[0],))
        self.node_point_start = concat_with_tensor(self.node_point_start, nps, expand=(nps.shape[0],))
        self.node_point_n = concat_with_tensor(self.node_point_n, npn_t, expand=(npn_t.shape[0],))
        self.point_idx = concat_with_tensor(self.point_idx, pidx_t, expand=(pidx_t.shape[0],))
        self.chunk_link_idx = concat_with_tensor(self.chunk_link_idx, cli, expand=(cli.shape[0],))
        self.chunk_sensor_idx = concat_with_tensor(self.chunk_sensor_idx, csi, expand=(csi.shape[0],))
        self.chunk_node_start = concat_with_tensor(self.chunk_node_start, cns, expand=(cns.shape[0],))
        self.chunk_node_n = concat_with_tensor(self.chunk_node_n, cnn, expand=(cnn.shape[0],))
        self.sensor_chunk_start = concat_with_tensor(self.sensor_chunk_start, chunk_start_for_sensor, expand=(1,))
        self.sensor_chunk_n = concat_with_tensor(self.sensor_chunk_n, len(unique_links), expand=(1,))


@qd.func
def _func_vec3_at(values: qd.types.ndarray(), i: int) -> qd.types.vector(3):
    return qd.Vector([values[i, 0], values[i, 1], values[i, 2]], dt=float)


@qd.func
def _func_sphere_intersects_aabb(center, radius_sq, bmin, bmax):  # -> bool
    """Squared-distance sphere-vs-AABB test: returns True iff the closest point of the AABB to
    ``center`` is within ``radius_sq``. Used by ProximityTaxel BVH traversal."""
    d_sq = gs.qd_float(0.0)
    for k in qd.static(range(3)):
        v = center[k]
        lo = bmin[k]
        hi = bmax[k]
        if v < lo:
            d = lo - v
            d_sq = d_sq + d * d
        elif v > hi:
            d = v - hi
            d_sq = d_sq + d * d
    return d_sq <= radius_sq


@qd.func
def _func_aabb_intersects_aabb(amin, amax, bmin, bmax):  # -> bool
    """Standard 6-axis AABB-vs-AABB overlap test. Used by ElastomerTaxel BVH traversal."""
    return (
        amin[0] <= bmax[0]
        and amax[0] >= bmin[0]
        and amin[1] <= bmax[1]
        and amax[1] >= bmin[1]
        and amin[2] <= bmax[2]
        and amax[2] >= bmin[2]
    )


@qd.kernel
def _kernel_point_cloud_proximity_taxel_bvh(
    probe_positions_local: qd.types.ndarray(),
    probe_local_normal: qd.types.ndarray(),
    probe_sensor_idx: qd.types.ndarray(),
    links_idx: qd.types.ndarray(),
    sensor_cache_start: qd.types.ndarray(),
    sensor_probe_start: qd.types.ndarray(),
    n_probes_per_sensor: qd.types.ndarray(),
    bvh_sensor_chunk_start: qd.types.ndarray(),
    bvh_sensor_chunk_n: qd.types.ndarray(),
    bvh_chunk_link_idx: qd.types.ndarray(),
    bvh_chunk_node_start: qd.types.ndarray(),
    bvh_node_min: qd.types.ndarray(),
    bvh_node_max: qd.types.ndarray(),
    bvh_node_left: qd.types.ndarray(),
    bvh_node_right: qd.types.ndarray(),
    bvh_node_point_start: qd.types.ndarray(),
    bvh_node_point_n: qd.types.ndarray(),
    bvh_point_idx: qd.types.ndarray(),
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

        chunk_start = bvh_sensor_chunk_start[i_s]
        n_chunks = bvh_sensor_chunk_n[i_s]
        for c_off in range(n_chunks):
            i_c = chunk_start + c_off
            track_link_idx = bvh_chunk_link_idx[i_c]
            track_pos = links_state.pos[track_link_idx, i_b]
            track_quat = links_state.quat[track_link_idx, i_b]
            rcom_o = links_state.root_COM[track_link_idx, i_b]
            cdv_o = links_state.cd_vel[track_link_idx, i_b]
            cda_o = links_state.cd_ang[track_link_idx, i_b]
            # BVH nodes live in tracked-link local frame: bring the probe sphere center over.
            probe_link = gu.qd_inv_transform_by_trans_quat(probe_world, track_pos, track_quat)

            stack = qd.Vector.zero(gs.qd_int, qd.static(_POINT_CLOUD_BVH_STACK_SIZE))
            stack[0] = bvh_chunk_node_start[i_c]
            stack_idx = 1

            while stack_idx > 0:
                stack_idx -= 1
                n = stack[stack_idx]
                bmin = _func_vec3_at(bvh_node_min, n)
                bmax = _func_vec3_at(bvh_node_max, n)
                if not _func_sphere_intersects_aabb(probe_link, R_query_sq, bmin, bmax):
                    continue
                left = bvh_node_left[n]
                if left == -1:
                    pstart = bvh_node_point_start[n]
                    pn = bvh_node_point_n[n]
                    for j in range(pn):
                        i_o = bvh_point_idx[pstart + j]
                        if not pc_active_envs_mask[i_o, i_b]:
                            continue
                        pos_l = _func_vec3_at(pc_pos_link, i_o)
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
                    right = bvh_node_right[n]
                    stack[stack_idx] = left
                    stack_idx += 1
                    stack[stack_idx] = right
                    stack_idx += 1

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

        # BVH growth follows pc_pos_link growth in lockstep: each leaf's point_idx is an absolute
        # row into the just-grown pc_pos_link.
        self._shared_metadata.pc_bvh.append_sensor(
            pc_start_row=pc_start_row,
            idx_cat=idx_cat,
            pos_cat=pos_cat,
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
                trk_pos = trk_link.get_pos(envs_idx, relative=False)[:, None, :]
                trk_quat = trk_link.get_quat(envs_idx, relative=False)[:, None, :]
                pc_world = gu.transform_by_trans_quat(pos_local[None, :, :], trk_pos, trk_quat)
                pc_world = tensor_to_array(pc_world) + env_offsets[:, None, :]
                world_chunks.append(pc_world[active_mask])
            else:
                active_mask = active_envs_mask[:, 0]
                pos_active = pos_local[active_mask]
                if pos_active.numel() == 0:
                    continue
                trk_pos = trk_link.get_pos(envs_idx, relative=False).reshape(3)
                trk_quat = trk_link.get_quat(envs_idx, relative=False).reshape(4)
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


class ProximityTaxelReturnType(NamedTuple):
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
        shared_context: None,
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

        bvh = shared_metadata.pc_bvh
        _kernel_point_cloud_proximity_taxel_bvh(
            shared_metadata.probe_positions,
            shared_metadata.probe_local_normal,
            shared_metadata.probe_sensor_idx,
            shared_metadata.links_idx,
            shared_metadata.sensor_cache_start,
            shared_metadata.sensor_probe_start,
            shared_metadata.n_probes_per_sensor,
            bvh.sensor_chunk_start,
            bvh.sensor_chunk_n,
            bvh.chunk_link_idx,
            bvh.chunk_node_start,
            bvh.node_min,
            bvh.node_max,
            bvh.node_left,
            bvh.node_right,
            bvh.node_point_start,
            bvh.node_point_n,
            bvh.point_idx,
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
    source_normal: qd.types.vector(3),
    target_pos: qd.types.vector(3),
    target_normal: qd.types.vector(3),
    depth: float,
    lam: float,
    scale: float,
) -> qd.types.vector(3):
    source_contact_pos = source_pos - source_normal * depth
    diff = target_pos - source_contact_pos
    planar_diff = _func_elastomer_tangent(diff, target_normal)
    return diff * depth * qd.exp(-lam * planar_diff.dot(planar_diff)) * scale


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
    i = torch.arange(fft_n[0], dtype=dtype, device=device)
    j = torch.arange(fft_n[1], dtype=dtype, device=device)
    xx, yy = torch.meshgrid((i - fft_n[0] // 2) * grid_spacing[0], (j - fft_n[1] // 2) * grid_spacing[1], indexing="ij")
    g = torch.exp(torch.tensor(-lambda_d, dtype=dtype, device=device) * (xx * xx + yy * yy))
    k = torch.stack((xx * g, yy * g, g), dim=0)
    k = torch.fft.ifftshift(k, dim=(-2, -1))
    return torch.fft.fft2(k)


@qd.kernel(fastcache=True)
def _kernel_elastomer_probe_depth(
    probe_positions_local: qd.types.ndarray(),
    probe_sensor_idx: qd.types.ndarray(),
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
    """Per-probe contact depth from track-geom SDF, parallel over (env, probe). Writes only
    ``probe_depth_buf``; dilate accumulation is split into a separate target-major kernel that
    runs without atomics."""
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

        probe_depth_buf[i_b, i_p] = qd.max(gs.qd_float(0.0), -min_sdf)


@qd.kernel(fastcache=True)
def _kernel_elastomer_dilate_accumulate(
    use_grid_fft: qd.types.ndarray(),
    probe_positions_local: qd.types.ndarray(),
    probe_local_normal: qd.types.ndarray(),
    probe_sensor_idx: qd.types.ndarray(),
    sensor_cache_start: qd.types.ndarray(),
    sensor_probe_start: qd.types.ndarray(),
    n_probes_per_sensor: qd.types.ndarray(),
    lambda_d: qd.types.ndarray(),
    dilate_scale: qd.types.ndarray(),
    probe_depth_buf: qd.types.ndarray(),
    output: qd.types.ndarray(),
):
    """Target-major dilate accumulator for non-grid sensors. Each (env, target_probe) thread sums
    Gaussian contributions from every in-contact source probe of its sensor into a register and
    writes once -- no atomic_add. Grid sensors are skipped (FFT path handles them).
    Output write is an OVERWRITE because output was pre-zeroed at step start and no other writer
    touches a non-grid sensor's range before shear-accumulate."""
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
        _i_p = i_p - probe_start

        target_local = _func_vec3_at(probe_positions_local, i_p)
        target_normal = _func_vec3_at(probe_local_normal, i_p)

        acc = qd.Vector.zero(gs.qd_float, 3)
        for j in range(n_probes):
            j_p = probe_start + j
            src_depth = probe_depth_buf[i_b, j_p]
            if src_depth <= gs.qd_float(0.0):
                continue
            contribution = _func_elastomer_direct_dilate_contribution(
                _func_vec3_at(probe_positions_local, j_p),
                _func_vec3_at(probe_local_normal, j_p),
                target_local,
                target_normal,
                src_depth,
                lam,
                scale,
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
    bvh_chunk_link_idx: qd.types.ndarray(),
    bvh_chunk_node_start: qd.types.ndarray(),
    bvh_node_min: qd.types.ndarray(),
    bvh_node_max: qd.types.ndarray(),
    bvh_node_left: qd.types.ndarray(),
    bvh_node_right: qd.types.ndarray(),
    bvh_node_point_start: qd.types.ndarray(),
    bvh_node_point_n: qd.types.ndarray(),
    bvh_point_idx: qd.types.ndarray(),
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

    Merges what was previously two kernels (per-chunk AABB fill + BVH traversal) so the AABB stays
    in thread-local state instead of round-tripping through a (B, n_chunks, 3) buffer. No probe
    work happens here -- the shear contribution is accumulated in a separate target-major kernel
    that reads surface_pos_sensor_buf / surface_depth_buf / surface_entry_pos_sensor_buf.
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
        track_link_idx = bvh_chunk_link_idx[i_c]
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

        stack = qd.Vector.zero(gs.qd_int, qd.static(_POINT_CLOUD_BVH_STACK_SIZE))
        stack[0] = bvh_chunk_node_start[i_c]
        stack_idx = 1

        while stack_idx > 0:
            stack_idx -= 1
            n = stack[stack_idx]
            bmin = _func_vec3_at(bvh_node_min, n)
            bmax = _func_vec3_at(bvh_node_max, n)
            if not _func_aabb_intersects_aabb(bmin, bmax, qmin, qmax):
                continue
            left = bvh_node_left[n]
            if left == -1:
                pstart = bvh_node_point_start[n]
                pn = bvh_node_point_n[n]
                for j in range(pn):
                    i_o = bvh_point_idx[pstart + j]
                    if not pc_active_envs_mask[i_o, i_b]:
                        continue
                    surface_candidate_buf[i_b, i_o] = True

                    point_link = _func_vec3_at(pc_pos_link, i_o)
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
                right = bvh_node_right[n]
                stack[stack_idx] = left
                stack_idx += 1
                stack[stack_idx] = right
                stack_idx += 1


@qd.kernel(fastcache=True)
def _kernel_elastomer_shear_accumulate(
    probe_positions_local: qd.types.ndarray(),
    probe_local_normal: qd.types.ndarray(),
    probe_sensor_idx: qd.types.ndarray(),
    sensor_cache_start: qd.types.ndarray(),
    sensor_probe_start: qd.types.ndarray(),
    sensor_pc_start: qd.types.ndarray(),
    sensor_pc_n: qd.types.ndarray(),
    lambda_s: qd.types.ndarray(),
    shear_scale: qd.types.ndarray(),
    eps: float,
    surface_pos_sensor_buf: qd.types.ndarray(),
    surface_entry_pos_sensor_buf: qd.types.ndarray(),
    surface_depth_buf: qd.types.ndarray(),
    surface_initialized_buf: qd.types.ndarray(),
    output: qd.types.ndarray(),
):
    """Target-major shear accumulator: per (env, target_probe), iterate over the sensor's pc rows
    that are flagged ``surface_initialized`` and sum Gaussian contributions into a register, then
    += the result into ``output``. No atomic_add (each (i_b, i_p) thread owns its output slot).

    Must run after the surface-state kernel AND after the Patch-3 torch cleanup that invalidates
    ``surface_initialized_buf`` for BVH-pruned points -- otherwise stale True flags from prior
    steps would corrupt this step's accumulation.
    """
    total_n_probes = probe_positions_local.shape[0]
    n_batches = surface_pos_sensor_buf.shape[0]

    for i_b, i_p in qd.ndrange(n_batches, total_n_probes):
        i_s = probe_sensor_idx[i_p]
        scale = shear_scale[i_s]
        if scale <= gs.qd_float(0.0):
            continue
        lam = lambda_s[i_s]
        cache_start = sensor_cache_start[i_s]
        _i_p = i_p - sensor_probe_start[i_s]
        pc_start = sensor_pc_start[i_s]
        pc_end = pc_start + sensor_pc_n[i_s]

        probe_local = _func_vec3_at(probe_positions_local, i_p)
        probe_normal = _func_vec3_at(probe_local_normal, i_p)

        acc = qd.Vector.zero(gs.qd_float, 3)
        for i_o in range(pc_start, pc_end):
            if not surface_initialized_buf[i_b, i_o]:
                continue
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


def _elastomer_taxel_grid_fft_dilate(
    fft_grid_meta: list[tuple[int, int, int, int, int, float, float, float]],
    fft_grid_kernels_stacked: torch.Tensor,
    probe_depth_buf: torch.Tensor,
    fft_depth_buffer: torch.Tensor,
    dilate_scale: torch.Tensor,
    grid_normal: torch.Tensor,
    grid_tangent_u: torch.Tensor,
    grid_tangent_v: torch.Tensor,
    grid_dilate_out_buffer: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """
    Elastomer marker dilation via 2D FFT in the validated probe tangent basis.

    All grid sensors share the global ``fft_max_n`` (= last two dims of ``fft_depth_buffer``); their
    kernels are stacked into ``fft_grid_kernels_stacked`` of shape (n_grid, 3, fft_max_n[0],
    fft_max_n[1]). The four heavy FFTs (fft of H, fft of H*H, ifft for Kx/Ky/Kn) thus run as
    batched ops over the grid-sensor axis, dropping launches from 4·n_grid to 4. The H-fill and
    write-back stages remain per-sensor (small Python loops over view/copy and per-sensor tangent
    decomposition).
    """
    if not fft_grid_meta:
        return
    n_batches = probe_depth_buf.shape[0]

    # 1) Fill the (B, n_grid, fft_max_nx, fft_max_ny) depth buffer. Per-sensor view+copy only.
    fft_depth_buffer.zero_()
    for grid_pos, (_, g_nx, g_ny, probe_start, _, _, _, _) in enumerate(fft_grid_meta):
        depth_slice = probe_depth_buf[:, probe_start : probe_start + g_nx * g_ny]
        fft_depth_buffer[:, grid_pos, :g_nx, :g_ny].copy_(depth_slice.view(n_batches, g_ny, g_nx).transpose(1, 2))

    # 2) Batched FFTs across (B, n_grid). Broadcast over B when multiplying by per-sensor kernels.
    H_fft = torch.fft.fft2(fft_depth_buffer)
    H2_fft = torch.fft.fft2(fft_depth_buffer * fft_depth_buffer)
    Kx_all = fft_grid_kernels_stacked[:, 0]  # (n_grid, fft_max_nx, fft_max_ny) complex
    Ky_all = fft_grid_kernels_stacked[:, 1]
    Kn_all = fft_grid_kernels_stacked[:, 2]
    disp_u_all = torch.fft.ifft2(H_fft * Kx_all).real  # (B, n_grid, fft_max_nx, fft_max_ny)
    disp_v_all = torch.fft.ifft2(H_fft * Ky_all).real
    disp_n_all = torch.fft.ifft2(H2_fft * Kn_all).real

    # 3) Per-sensor write-back: slice to (g_nx, g_ny), apply scale + tangent decomposition, copy
    # into the sensor's output range. Tangent vectors are per-sensor so can't trivially batch here.
    for grid_pos, meta in enumerate(fft_grid_meta):
        sensor_idx, g_nx, g_ny, _, cache_start, _, _, _ = meta
        scale_s = dilate_scale[sensor_idx]
        disp_u = disp_u_all[:, grid_pos, :g_nx, :g_ny] * scale_s
        disp_v = disp_v_all[:, grid_pos, :g_nx, :g_ny] * scale_s
        disp_n = disp_n_all[:, grid_pos, :g_nx, :g_ny] * scale_s
        # Cache order is probe flat index iy*nx+ix; (g_nx, g_ny) transpose(1, 2).reshape gives (g_ny, g_nx) -> iy*nx+ix.
        disp_u_flat = disp_u.transpose(1, 2).reshape(n_batches, -1)
        disp_v_flat = disp_v.transpose(1, 2).reshape(n_batches, -1)
        disp_n_flat = disp_n.transpose(1, 2).reshape(n_batches, -1)
        grid_size = g_nx * g_ny * 3
        out_block = grid_dilate_out_buffer[:, :grid_size]
        tangent_u = grid_tangent_u[sensor_idx]
        tangent_v = grid_tangent_v[sensor_idx]
        normal = grid_normal[sensor_idx]
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

    # Per-(env, pc-row) BVH-candidate flag, zeroed each step and written True by the surface-state
    # kernel for every visited active point. Post-kernel torch ops use ``!candidate`` to invalidate
    # stale surface_initialized / surface_entry_pos for points the BVH skipped this step.
    surface_candidate_buf: torch.Tensor = make_tensor_field((0, 0), dtype_factory=lambda: gs.tc_bool)

    is_grid: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_bool)
    use_grid_fft: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_bool)
    grid_n: torch.Tensor = make_tensor_field((0, 2), dtype_factory=lambda: gs.tc_int)
    grid_spacing: torch.Tensor = make_tensor_field((0, 2))
    grid_normal: torch.Tensor = make_tensor_field((0, 3))
    grid_tangent_u: torch.Tensor = make_tensor_field((0, 3))
    grid_tangent_v: torch.Tensor = make_tensor_field((0, 3))
    # Stacked complex FFT kernels for all grid-FFT sensors, shape (n_grid, 3, fft_max_n[0],
    # fft_max_n[1]). All sensors share fft_max_n so torch.fft.fft2 batches across the grid axis
    # in a single launch. When a new grid sensor expands fft_max_n at build time, prior sensors'
    # kernels are recomputed at the new size so the stack stays uniform.
    fft_grid_kernels_stacked: torch.Tensor = make_tensor_field((0, 0, 0, 0), dtype_factory=lambda: torch.complex64)
    fft_depth_buffer: torch.Tensor = make_tensor_field((0, 0, 0, 0))
    grid_dilate_out_buffer: torch.Tensor = make_tensor_field((0, 0))
    # Per-grid-FFT-sensor metadata captured at build time. Tuple fields:
    # (sensor_idx, g_nx, g_ny, probe_start, cache_start, lambda_d, spacing_u, spacing_v).
    # Indexed positionally to match rows of fft_grid_kernels_stacked / sensor axis of
    # fft_depth_buffer. Iterating it directly avoids per-step .item() device syncs.
    fft_grid_meta: list[tuple[int, int, int, int, int, float, float, float]] = field(default_factory=list)
    # Global max FFT size across all grid sensors. Mutated only at build time.
    fft_max_n: tuple[int, int] = (0, 0)

    # True iff at least one configured ElastomerTaxel has shear_scale > 0. Set during build by OR-ing
    # each sensor's value, so per-step gating avoids an O(n_sensors) reduction + device sync.
    any_shear: bool = False


class ElastomerTaxelSensor(
    PointCloudTactileSensorMixin[ElastomerTaxelSensorMetadata],
    ProbesWithNormalSensorMixin[ElastomerTaxelSensorMetadata],
    RigidSensorMixin[ElastomerTaxelSensorMetadata],
    SimpleSensor[ElastomerTaxelSensorOptions, None, ElastomerTaxelSensorMetadata],
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

        self._is_grid = self._probe_local_pos.ndim > 2
        self._shape = self._probe_local_pos.shape[:-1]

        (probe_pos, probe_normals, use_grid_fft, grid_normal, grid_tangent_u, grid_tangent_v, grid_spacing) = (
            _normalize_elastomer_probe_layout(
                np.asarray(options.probe_local_pos, dtype=gs.np_float),
                np.asarray(options.probe_local_normal, dtype=gs.np_float),
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
        if float(self._options.shear_scale) > 0.0:
            self._shared_metadata.any_shear = True
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

        self._shared_metadata.surface_candidate_buf = torch.zeros(
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
        if self._use_grid_fft:
            nx, ny = int(self._shape[1]), int(self._shape[0])
            grid_n = torch.tensor((nx, ny), dtype=gs.tc_int, device=gs.device)
            grid_spacing = self._grid_spacing
            grid_normal = self._grid_normal
            grid_tangent_u = self._grid_tangent_u
            grid_tangent_v = self._grid_tangent_v
            spacing_u, spacing_v = float(grid_spacing[0].item()), float(grid_spacing[1].item())
            this_fft_n = tuple(_next_pow2(2 * n - 1) for n in (nx, ny))
            cache_start_py = int(self._shared_metadata.sensor_cache_start[self._idx].item())
            self._shared_metadata.fft_grid_meta.append(
                (
                    self._idx,
                    nx,
                    ny,
                    self._probe_start_idx,
                    cache_start_py,
                    float(self._options.lambda_d),
                    spacing_u,
                    spacing_v,
                )
            )

            # Expand the global FFT size if this sensor needs more padding. When that happens, all
            # prior grid sensors' kernels are rebuilt at the new size (their FFTs depend on the
            # transform length, so frequency-domain padding wouldn't be equivalent).
            prev_max = self._shared_metadata.fft_max_n
            new_max = (max(prev_max[0], this_fft_n[0]), max(prev_max[1], this_fft_n[1]))
            self._shared_metadata.fft_max_n = new_max
            n_grid = len(self._shared_metadata.fft_grid_meta)
            stacked = torch.empty(
                (n_grid, 3, new_max[0], new_max[1]),
                dtype=torch.complex64,
                device=gs.device,
            )
            for grid_pos, (_, _, _, _, _, lam_d, sp_u, sp_v) in enumerate(self._shared_metadata.fft_grid_meta):
                stacked[grid_pos] = _precompute_hydroshear_dilate_kernel_fft(
                    lam_d, (sp_u, sp_v), new_max, gs.device, gs.tc_float
                )
            self._shared_metadata.fft_grid_kernels_stacked = stacked

            # fft_depth_buffer is keyed by grid-sensor position (not raw sensor i_s), sized at the
            # current global max FFT size. Reallocate every time we add a grid sensor.
            self._shared_metadata.fft_depth_buffer = torch.zeros(
                (B, n_grid, new_max[0], new_max[1]), dtype=gs.tc_float, device=gs.device
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

    def _get_return_format(self) -> tuple[int, ...]:
        return (self._n_probes, 3)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def reset(cls, shared_metadata: ElastomerTaxelSensorMetadata, shared_ground_truth_cache: torch.Tensor, envs_idx):
        super().reset(shared_metadata, shared_ground_truth_cache, envs_idx)
        # Only the hysteresis flag needs clearing on env reset. probe_depth_buf is overwritten every
        # step; surface_pos/entry/depth are only consumed where surface_initialized=True so they're
        # implicitly invalidated by clearing it; surface_candidate_buf is .zero_()'d at step start.
        shared_metadata.surface_initialized_buf[envs_idx, :] = False

    @classmethod
    def _update_current_timestep_data(
        cls,
        shared_context: None,
        shared_metadata: ElastomerTaxelSensorMetadata,
        current_ground_truth_data_T: torch.Tensor,
        ground_truth_data_timeline: "TensorRingBuffer | None",
        measured_data_timeline: "TensorRingBuffer",
    ):
        solver = shared_metadata.solver
        # No pre-zeros: probe_depth is fully overwritten by _kernel_elastomer_probe_depth;
        # current_ground_truth_data_T is fully overwritten by FFT-dilate ∪ dilate-accumulate (then
        # shear-accumulate += on top); surface_depth_buf is only read where surface_initialized=True,
        # which is set in lockstep with that same depth write; measured is .copy_'d at the end.
        measured = measured_data_timeline.at(0, copy=False)

        _kernel_elastomer_probe_depth(
            shared_metadata.probe_positions,
            shared_metadata.probe_sensor_idx,
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
        _kernel_elastomer_dilate_accumulate(
            shared_metadata.use_grid_fft,
            shared_metadata.probe_positions,
            shared_metadata.probe_local_normal,
            shared_metadata.probe_sensor_idx,
            shared_metadata.sensor_cache_start,
            shared_metadata.sensor_probe_start,
            shared_metadata.n_probes_per_sensor,
            shared_metadata.lambda_d,
            shared_metadata.dilate_scale,
            shared_metadata.probe_depth_buf,
            current_ground_truth_data_T,
        )
        # FFT runs after the qd dilate kernel: on Metal, write-only kernel outputs zero unwritten slots on copy-back,
        # which would erase the grid range the FFT just wrote.
        _elastomer_taxel_grid_fft_dilate(
            shared_metadata.fft_grid_meta,
            shared_metadata.fft_grid_kernels_stacked,
            shared_metadata.probe_depth_buf,
            shared_metadata.fft_depth_buffer,
            shared_metadata.dilate_scale,
            shared_metadata.grid_normal,
            shared_metadata.grid_tangent_u,
            shared_metadata.grid_tangent_v,
            shared_metadata.grid_dilate_out_buffer,
            current_ground_truth_data_T,
        )
        if shared_metadata.any_shear:
            bvh = shared_metadata.pc_bvh
            shared_metadata.surface_candidate_buf.zero_()
            _kernel_elastomer_surface_state_bvh(
                shared_metadata.links_idx,
                shared_metadata.sensor_elastomer_geom_start,
                shared_metadata.sensor_elastomer_geom_n,
                shared_metadata.elastomer_geom_idx,
                shared_metadata.elastomer_geom_active_envs_mask,
                bvh.chunk_sensor_idx,
                bvh.chunk_link_idx,
                bvh.chunk_node_start,
                bvh.node_min,
                bvh.node_max,
                bvh.node_left,
                bvh.node_right,
                bvh.node_point_start,
                bvh.node_point_n,
                bvh.point_idx,
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
            # Invalidate stale surface state for points the BVH did not visit. surface_initialized
            # and entry-pos survive across steps; depth/pos are gated by initialized downstream so
            # they don't need clearing. The shear accumulator below gates on surface_initialized_buf
            # -- without this step, stale True from a prior step would corrupt accumulation.
            cand = shared_metadata.surface_candidate_buf
            shared_metadata.surface_initialized_buf &= cand
            # Implicit bool→float broadcast zeros entries where cand=False, no `~` allocation.
            shared_metadata.surface_entry_pos_sensor_buf.mul_(cand.unsqueeze(-1))
            _kernel_elastomer_shear_accumulate(
                shared_metadata.probe_positions,
                shared_metadata.probe_local_normal,
                shared_metadata.probe_sensor_idx,
                shared_metadata.sensor_cache_start,
                shared_metadata.sensor_probe_start,
                shared_metadata.sensor_pc_start,
                shared_metadata.sensor_pc_n,
                shared_metadata.lambda_s,
                shared_metadata.shear_scale,
                gs.EPS,
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
