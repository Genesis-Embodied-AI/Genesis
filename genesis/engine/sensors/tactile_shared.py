from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, NamedTuple

import numpy as np
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import concat_with_tensor, make_tensor_field

if TYPE_CHECKING:
    from genesis.utils.ring_buffer import TensorRingBuffer


_GRID_TOL = 1.0e-5  # Tolerance for grid-regularity / orthogonality / normal-uniformity checks.


def next_pow2(n: int) -> int:
    """
    Smallest power of 2 >= ``n`` (1 if ``n == 0``).
    """
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p *= 2
    return p


# ==================== BVH helpers (shared by point-cloud and triangle-mesh sensors) ====================


BVH_LEAF_SIZE = 8
BVH_STACK_SIZE = 32


def get_mesh_geom_chunks(link, prefer_visual: bool) -> list[tuple[object, np.ndarray, np.ndarray]]:
    """
    Return per-geom mesh chunks ``(geom, verts_link, faces)`` in link-local frame.

    ``prefer_visual`` picks vgeoms over geoms when both exist; falls back to the other type when the
    preferred one is absent. Empty meshes are dropped from the list.
    """
    if prefer_visual:
        geoms = list(link.vgeoms) if link.vgeoms else list(link.geoms)
        use_vverts = bool(link.vgeoms)
    else:
        geoms = list(link.geoms) if link.geoms else list(link.vgeoms)
        use_vverts = not bool(link.geoms) and bool(link.vgeoms)

    chunks: list[tuple[object, np.ndarray, np.ndarray]] = []
    for geom in geoms:
        # init_*verts / init_*faces come from loaded mesh data whose dtype is not under our control, so coerce here.
        if use_vverts:
            verts = np.asarray(geom.init_vverts, dtype=gs.np_float)
            faces = np.asarray(geom.init_vfaces, dtype=gs.np_int)
        else:
            verts = np.asarray(geom.init_verts, dtype=gs.np_float)
            faces = np.asarray(geom.init_faces, dtype=gs.np_int)
        if verts.size == 0 or faces.size == 0:
            continue
        verts_link = gu.transform_by_trans_quat(verts, geom.init_pos, geom.init_quat)
        chunks.append((geom, verts_link.astype(gs.np_float, copy=False), faces))
    return chunks


def build_static_chunk_bvh(
    centroids: np.ndarray,
    aabb_mins: np.ndarray,
    aabb_maxs: np.ndarray,
    global_rows: np.ndarray,
    leaf_size: int,
) -> tuple[np.ndarray, ...]:
    """
    Median-split AABB BVH over a static set of elements (points, triangles, etc.) in link-local frame.

    Split decisions use ``centroids`` along the longest-spread axis; node AABBs union the per-element
    ``aabb_mins``/``aabb_maxs``. For point-cloud BVHs, callers pass ``centroids == aabb_mins == aabb_maxs``
    (the points themselves); for triangle BVHs, callers pass per-triangle centroid + min/max bounds.

    Leaves carry the caller-provided ``global_rows`` (absolute rows into the sensor-class element table);
    the kernel indexes directly into that table with no extra indirection. Internal nodes use -1 for
    ``node_left`` / ``node_right``. Returns ``(node_min, node_max, node_left, node_right, node_elem_start,
    node_elem_n, elem_idx)``.
    """
    node_min: list[np.ndarray] = []
    node_max: list[np.ndarray] = []
    node_left: list[int] = []
    node_right: list[int] = []
    node_elem_start: list[int] = []
    node_elem_n: list[int] = []
    elem_idx: list[int] = []

    def _alloc() -> int:
        i = len(node_min)
        node_min.append(np.zeros(3, dtype=gs.np_float))
        node_max.append(np.zeros(3, dtype=gs.np_float))
        node_left.append(-1)
        node_right.append(-1)
        node_elem_start.append(-1)
        node_elem_n.append(0)
        return i

    def _build(rows: np.ndarray, cents: np.ndarray, a_mins: np.ndarray, a_maxs: np.ndarray) -> int:
        nid = _alloc()
        bmin = a_mins.min(axis=0).astype(gs.np_float)
        bmax = a_maxs.max(axis=0).astype(gs.np_float)
        node_min[nid] = bmin
        node_max[nid] = bmax
        if rows.shape[0] <= leaf_size:
            start = len(elem_idx)
            elem_idx.extend(int(r) for r in rows)
            node_elem_start[nid] = start
            node_elem_n[nid] = int(rows.shape[0])
            return nid
        axis = int(np.argmax(bmax - bmin))
        order = np.argsort(cents[:, axis], kind="stable")
        mid = order.shape[0] // 2
        node_left[nid] = _build(rows[order[:mid]], cents[order[:mid]], a_mins[order[:mid]], a_maxs[order[:mid]])
        node_right[nid] = _build(rows[order[mid:]], cents[order[mid:]], a_mins[order[mid:]], a_maxs[order[mid:]])
        return nid

    if centroids.shape[0] == 0:
        return (
            np.zeros((0, 3), dtype=gs.np_float),
            np.zeros((0, 3), dtype=gs.np_float),
            np.zeros((0,), dtype=gs.np_int),
            np.zeros((0,), dtype=gs.np_int),
            np.zeros((0,), dtype=gs.np_int),
            np.zeros((0,), dtype=gs.np_int),
            np.zeros((0,), dtype=gs.np_int),
        )

    root = _build(
        global_rows.astype(gs.np_int, copy=False),
        centroids.astype(gs.np_float, copy=False),
        aabb_mins.astype(gs.np_float, copy=False),
        aabb_maxs.astype(gs.np_float, copy=False),
    )
    assert root == 0
    return (
        np.stack(node_min, axis=0),
        np.stack(node_max, axis=0),
        np.asarray(node_left, dtype=gs.np_int),
        np.asarray(node_right, dtype=gs.np_int),
        np.asarray(node_elem_start, dtype=gs.np_int),
        np.asarray(node_elem_n, dtype=gs.np_int),
        np.asarray(elem_idx, dtype=gs.np_int),
    )


@qd.func
def func_vec3_at(values: qd.types.ndarray(), i: int) -> qd.types.vector(3):
    return qd.Vector([values[i, 0], values[i, 1], values[i, 2]], dt=float)


@qd.func
def func_sphere_intersects_aabb(center, radius_sq, bmin, bmax):  # -> bool
    """
    Squared-distance sphere-vs-AABB test: True iff the closest AABB point to ``center`` is within ``radius_sq``.

    Reused as a closest-point cull by passing ``radius_sq = current_best_dist_sq``.
    """
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
def func_aabb_intersects_aabb(amin, amax, bmin, bmax):  # -> bool
    """
    Standard 6-axis AABB-vs-AABB overlap test.
    """
    return (
        amin[0] <= bmax[0]
        and amax[0] >= bmin[0]
        and amin[1] <= bmax[1]
        and amax[1] >= bmin[1]
        and amin[2] <= bmax[2]
        and amax[2] >= bmin[2]
    )


@dataclass(eq=True, kw_only=False, frozen=True)
class ChunkedBVHData:
    """
    Bundle of the flat ``BVHMetadata`` scaffolding tensors passed to a traversal kernel as one argument.

    See ``BVHMetadata`` for the field semantics. Element payload tables (``tri_verts``, point-cloud
    positions, ...) stay separate kernel arguments because they differ per sensor class.
    """

    sensor_chunk_start: qd.types.ndarray()
    sensor_chunk_count: qd.types.ndarray()
    chunk_link_idx: qd.types.ndarray()
    chunk_node_start: qd.types.ndarray()
    node_min: qd.types.ndarray()
    node_max: qd.types.ndarray()
    node_left: qd.types.ndarray()
    node_right: qd.types.ndarray()
    node_leaf_start: qd.types.ndarray()
    node_leaf_count: qd.types.ndarray()
    leaf_elem_idx: qd.types.ndarray()


@dataclass
class BVHMetadata:
    """
    Element-agnostic scaffolding for a static, link-local, chunked AABB BVH shared across one sensor class.

    One *chunk* per (sensor, tracked_link): each chunk is a small subtree built once at scene init in the
    tracked link's local frame and never rebuilt. Subclasses (PointCloudBVH, TriangleMeshBVH) layer on
    element-specific payload tables; ``leaf_elem_idx`` entries are absolute rows into those tables.

    Per-sensor slice into the chunk arrays:
        ``chunks[sensor_chunk_start[s] : sensor_chunk_start[s] + sensor_chunk_count[s]]``
    Per-chunk slice into the flat node arrays:
        ``nodes[chunk_node_start[c] : chunk_node_start[c] + chunk_node_count[c]]``
    Per-leaf slice into ``leaf_elem_idx``:
        ``leaf_elem_idx[node_leaf_start[n] : node_leaf_start[n] + node_leaf_count[n]]``
    ``node_left == -1`` marks a leaf; otherwise ``node_left``/``node_right`` are absolute child indices.
    """

    sensor_chunk_start: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    sensor_chunk_count: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)

    chunk_link_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    chunk_node_start: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    chunk_node_count: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)

    node_min: torch.Tensor = make_tensor_field((0, 3))
    node_max: torch.Tensor = make_tensor_field((0, 3))
    node_left: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    node_right: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)

    node_leaf_start: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    node_leaf_count: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    leaf_elem_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)

    # Cached scaffolding bundle for kernel calls, built once on first use (the BVH is static after scene init).
    _kernel_bvh: "ChunkedBVHData | None" = field(default=None, init=False, compare=False, repr=False)

    @property
    def kernel_bvh(self) -> ChunkedBVHData:
        """
        The scaffolding fields bundled into a single ``ChunkedBVHData`` traversal-kernel argument.
        """
        if self._kernel_bvh is None:
            self._kernel_bvh = ChunkedBVHData(
                sensor_chunk_start=self.sensor_chunk_start,
                sensor_chunk_count=self.sensor_chunk_count,
                chunk_link_idx=self.chunk_link_idx,
                chunk_node_start=self.chunk_node_start,
                node_min=self.node_min,
                node_max=self.node_max,
                node_left=self.node_left,
                node_right=self.node_right,
                node_leaf_start=self.node_leaf_start,
                node_leaf_count=self.node_leaf_count,
                leaf_elem_idx=self.leaf_elem_idx,
            )
        return self._kernel_bvh


# ============================ FFT helpers ============================


@dataclass
class GridFFTConvMetadataMixin:
    """
    Shared per-sensor-class state for the per-grid 2D-FFT convolution passes.

    Attributes
    ----------
    grid_fft_meta : list of NamedTuple
        Per-grid-FFT-sensor metadata records. The leading 5 fields are always
        ``(sensor_idx, g_ny, g_nx, probe_start, cache_start)``; sensors append their kernel params after that
        (e.g. ``GridFFTMeta`` for HydroShear dilation).
    grid_fft_max_n : (int, int)
        Global FFT size ``(fft_ny, fft_nx)``, the elementwise max over all registered grid sensors. Build-time only.
    grid_fft_kernels_stacked : torch.Tensor
        Stacked complex ``rfft2`` kernels (half spectrum), shape ``(n_grid, n_planes, fft_ny, fft_nx // 2 + 1)``.
        Recomputed when the FFT size grows.
    grid_fft_buffer : torch.Tensor
        Reused per-step real buffer: ``(B, n_grid, n_channels, fft_ny, fft_nx)`` when registered with
        ``n_buffer_channels > 0``, else ``(B, n_grid, fft_ny, fft_nx)``. Reallocated on each registration.
    any_grid_fft : bool
        Python fast-path flag; True iff at least one grid-FFT sensor is registered.
    """

    grid_fft_meta: list[NamedTuple] = field(default_factory=list)
    grid_fft_max_n: tuple[int, int] = (0, 0)
    grid_fft_kernels_stacked: torch.Tensor = make_tensor_field((0, 0, 0, 0), dtype_factory=lambda: torch.complex64)
    grid_fft_buffer: torch.Tensor = make_tensor_field((0, 0, 0, 0))
    any_grid_fft: bool = False


def register_grid_fft_sensor(
    metadata: GridFFTConvMetadataMixin,
    meta_entry: NamedTuple,
    this_fft_n: tuple[int, int],
    kernel_builder: Callable[[NamedTuple, tuple[int, int]], torch.Tensor],
    n_buffer_channels: int,
    batch_size: int,
) -> None:
    """
    Register one grid-shaped sensor for FFT convolution; (re)build the stacked kernels and the per-step buffer.

    Parameters
    ----------
    meta_entry : NamedTuple
        Metadata record appended to ``grid_fft_meta``; its leading 5 fields must be
        ``(sensor_idx, g_ny, g_nx, probe_start, cache_start)``, followed by any sensor-specific kernel params.
    this_fft_n : (int, int)
        The ``(ny, nx)`` FFT size this sensor needs. The shared ``grid_fft_max_n`` grows to the elementwise max;
        when it grows, every prior sensor's kernel is recomputed at the new size (frequency-domain padding is not
        equivalent to spatial zero-padding).
    kernel_builder : callable
        ``kernel_builder(meta_entry, fft_n) -> (n_planes, fft_ny, fft_nx // 2 + 1)`` complex tensor (an ``rfft2``
        half spectrum). Must be deterministic from the meta record, since it is re-invoked whenever the FFT size grows.
    n_buffer_channels : int
        When ``> 0``, allocate a 5D ``(B, n_grid, n_buffer_channels, ny, nx)`` per-step buffer; else a 4D
        ``(B, n_grid, ny, nx)`` one.
    """
    metadata.grid_fft_meta.append(meta_entry)
    cur = metadata.grid_fft_max_n
    new_n = (max(cur[0], this_fft_n[0]), max(cur[1], this_fft_n[1]))
    metadata.grid_fft_max_n = new_n
    n_grid = len(metadata.grid_fft_meta)
    metadata.grid_fft_kernels_stacked = torch.stack([kernel_builder(m, new_n) for m in metadata.grid_fft_meta], dim=0)
    buffer_shape = (
        (batch_size, n_grid, n_buffer_channels, new_n[0], new_n[1])
        if n_buffer_channels > 0
        else (batch_size, n_grid, new_n[0], new_n[1])
    )
    metadata.grid_fft_buffer = torch.zeros(buffer_shape, dtype=gs.tc_float, device=gs.device)
    metadata.any_grid_fft = True


def expand_probe_normals(normals: np.ndarray, n_probes: int, probe_shape: tuple[int, ...]) -> np.ndarray:
    """Broadcast ``normals`` to a flat ``(n_probes, 3)`` array.

    Accepts a single shared normal of shape ``(3,)``, a grid-shaped ``(*probe_shape, 3)`` array, or an already-flat
    ``(n_probes, 3)``. Any other shape raises.
    """
    normals = np.asarray(normals, dtype=gs.np_float)
    if normals.ndim == 1:
        return np.broadcast_to(normals, (n_probes, 3)).copy()
    if normals.shape == (*probe_shape, 3):
        return normals.reshape(n_probes, 3).copy()
    if normals.shape == (n_probes, 3):
        return normals.copy()
    gs.raise_exception(
        "probe_local_normal must be one normal or match probe_local_pos shape. "
        f"Got normal shape {normals.shape} for probe shape {probe_shape}."
    )


def normalize_grid_probe_layout(
    probe_pos: np.ndarray, probe_normals: np.ndarray, is_grid: bool
) -> tuple[np.ndarray, np.ndarray, bool, bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate a probe layout and extract grid-FFT metadata when the layout qualifies.

    Returns ``(flat_positions, flat_normals, use_grid_fft, is_grid_regular, grid_normal, tangent_u, tangent_v,
    grid_spacing)``.

    ``use_grid_fft`` is True when the layout has shape ``(ny, nx, 3)`` with ``ny, nx >= 2`` and non-degenerate
    spacing along both axes -- the FFT path is usable and the grid metadata is populated as a best-fit
    approximation (average step vectors over all adjacent pairs, average unit normal over all probes).

    ``is_grid_regular`` is True when, in addition, the layout is strictly regular: normals uniform within
    tolerance, tangents orthogonal, both tangents in the plane perpendicular to the normal, and all probes lie
    on the regular ``(spacing_u, spacing_v)`` rectangle implied by the averaged steps. Callers that proceed
    with FFT on an irregular layout (``use_grid_fft`` and not ``is_grid_regular``) should warn the user.

    When ``use_grid_fft`` is False, the tangent / spacing / normal entries are zero.
    """
    probe_shape = probe_pos.shape[:-1]
    flat = probe_pos.reshape(-1, 3)
    normals = expand_probe_normals(probe_normals, flat.shape[0], probe_shape)

    normal_norms = np.linalg.norm(normals, axis=1)
    if np.any(normal_norms < gs.EPS):
        gs.raise_exception("probe_local_normal entries must be non-zero.")
    normals = normals / normal_norms[:, None]

    use_grid_fft = False
    is_grid_regular = False
    grid_normal = np.zeros(3, dtype=gs.np_float)
    tangent_u = np.zeros(3, dtype=gs.np_float)
    tangent_v = np.zeros(3, dtype=gs.np_float)
    grid_spacing = np.zeros(2, dtype=gs.np_float)

    if is_grid:
        if len(probe_shape) != 2:
            gs.raise_exception("Grid probe_local_pos must have shape (ny, nx, 3).")
        ny, nx = int(probe_shape[0]), int(probe_shape[1])
        if nx >= 2 and ny >= 2:
            grid = probe_pos.reshape(ny, nx, 3)
            # Averaged step vectors across all adjacent pairs along each axis -- robust to local jitter.
            avg_step_u = (grid[:, 1:, :] - grid[:, :-1, :]).reshape(-1, 3).mean(axis=0)
            avg_step_v = (grid[1:, :, :] - grid[:-1, :, :]).reshape(-1, 3).mean(axis=0)
            spacing_u = float(np.linalg.norm(avg_step_u))
            spacing_v = float(np.linalg.norm(avg_step_v))
            if spacing_u >= gs.EPS and spacing_v >= gs.EPS:
                tangent_u_candidate = (avg_step_u / spacing_u).astype(gs.np_float)
                tangent_v_candidate = (avg_step_v / spacing_v).astype(gs.np_float)
                # Average unit normal across all probes. If they cancel out (e.g. opposing normals), fall back
                # to the first probe's normal so downstream FFT still has a defined orientation.
                avg_normal = normals.mean(axis=0)
                normal_norm = float(np.linalg.norm(avg_normal))
                if normal_norm < gs.EPS:
                    normal_candidate = normals[0].astype(gs.np_float, copy=False)
                else:
                    normal_candidate = (avg_normal / normal_norm).astype(gs.np_float)

                normals_are_uniform = bool(np.all(normals @ normal_candidate >= 1.0 - _GRID_TOL))
                axes_are_orthogonal = abs(float(tangent_u_candidate @ tangent_v_candidate)) <= _GRID_TOL
                axes_in_plane = (
                    abs(float(tangent_u_candidate @ normal_candidate)) <= _GRID_TOL
                    and abs(float(tangent_v_candidate @ normal_candidate)) <= _GRID_TOL
                )
                expected = (
                    grid[0, 0]
                    + np.arange(nx, dtype=gs.np_float)[None, :, None] * avg_step_u[None, None, :]
                    + np.arange(ny, dtype=gs.np_float)[:, None, None] * avg_step_v[None, None, :]
                )
                is_regular = bool(np.max(np.linalg.norm(grid - expected, axis=-1)) <= _GRID_TOL)

                use_grid_fft = True
                is_grid_regular = normals_are_uniform and axes_are_orthogonal and axes_in_plane and is_regular
                grid_normal = normal_candidate
                tangent_u = tangent_u_candidate
                tangent_v = tangent_v_candidate
                grid_spacing = np.array((spacing_u, spacing_v), dtype=gs.np_float)

    return (
        flat.astype(gs.np_float, copy=False),
        normals.astype(gs.np_float, copy=False),
        use_grid_fft,
        is_grid_regular,
        grid_normal.astype(gs.np_float, copy=False),
        tangent_u.astype(gs.np_float, copy=False),
        tangent_v.astype(gs.np_float, copy=False),
        grid_spacing.astype(gs.np_float, copy=False),
    )


# ============================ Contact prefilter ============================


@dataclass
class ContactPrefilterMetadataMixin:
    """
    Per-(env, sensor) prefilter buffers shared by tactile sensors whose kernels query the collider's contacts per
    probe (KinematicTaxel, ContactDepthProbe), populated each step in ``kinematic_tactile.py``:

    - ``sensor_contacts_idx`` / ``sensor_n_contacts``: compact list of contact indices whose ``link_a`` or
      ``link_b`` matches the sensor's tracked link (``_kernel_build_sensor_contact_idx``). Feeds the raycast
      path's BVH-mask builder. Shape ``(B, n_sensors, max_contacts)`` / ``(B, n_sensors)``; the per-sensor cap
      (``_MAX_CONTACTS_PER_SENSOR``) is read off ``sensor_contacts_idx.shape[2]``.
    - ``sensor_geoms_idx`` / ``sensor_n_geoms``: compact *deduplicated* list of the opposing contacting geoms
      for the same link (``_kernel_build_sensor_geom_idx``). Feeds the SDF path so each probe queries one SDF
      per distinct geom instead of one per contact point. Shape ``(B, n_sensors, max_geoms)`` /
      ``(B, n_sensors)``; the cap (``_MAX_GEOMS_PER_SENSOR``) is read off ``sensor_geoms_idx.shape[2]``.
    """

    sensor_contacts_idx: torch.Tensor = make_tensor_field((0, 0, 0), dtype_factory=lambda: gs.tc_int)
    sensor_n_contacts: torch.Tensor = make_tensor_field((0, 0), dtype_factory=lambda: gs.tc_int)
    sensor_geoms_idx: torch.Tensor = make_tensor_field((0, 0, 0), dtype_factory=lambda: gs.tc_int)
    sensor_n_geoms: torch.Tensor = make_tensor_field((0, 0), dtype_factory=lambda: gs.tc_int)


# ============================ Contact depth query mode (SDF vs raycast) ============================


@dataclass
class ContactDepthQueryMetadataMixin:
    """
    Shared per-sensor-class state for the contact-depth query backend.

    ``contact_depth_query`` is the resolved mode for every sensor of this class - ``"sdf"`` or ``"raycast"``. The
    backend is dispatched once per sensor class (a single kernel covers all of the class's sensors), so every sensor
    must agree: each sensor's ``build()`` calls ``resolve_contact_depth_query``, which records the first non-``None``
    mode and raises if a later sensor requests a different one. ``None`` defers to whatever another sensor set;
    ``None`` at update time falls back to ``"sdf"``.

    When mode is ``"raycast"``, the collision/visual BVHs come from the shared ``RaycastContext`` (the sensor's
    ``shared_context``), built lazily on raycast opt-in and refreshed once per step by ``SensorManager``.
    ``sensor_candidate_geom_mask`` is a per-(env, sensor, geom) bool gate - scattered per step from the contact
    prefilter (KinematicTactile family) or once at build from ``sensor_track_geom_idx`` (ElastomerTaxel) - so BVH
    leaves whose ``faces_info.geom_idx`` falls outside the mask are skipped.
    """

    contact_depth_query: str | None = None
    sensor_candidate_geom_mask: torch.Tensor = make_tensor_field((0, 0, 0), dtype_factory=lambda: gs.tc_bool)


def resolve_contact_depth_query(metadata: ContactDepthQueryMetadataMixin, mode: str | None, sensor_name: str) -> None:
    """
    Record the class-wide contact-depth backend for one sensor, rejecting a conflict with an earlier sensor.

    The backend is dispatched per sensor class, so every sensor of a class must agree on ``"sdf"`` vs ``"raycast"``.
    ``mode is None`` defers to whatever another sensor set (default ``"sdf"``); a different non-``None`` ``mode``
    raises rather than silently switching the whole class by build order.
    """
    if mode is None:
        return
    existing = metadata.contact_depth_query
    if existing is not None and existing != mode:
        gs.raise_exception(
            f"{sensor_name} sensors disagree on contact_depth_query ({existing!r} vs {mode!r}). All sensors of a "
            "tactile class share one contact-depth backend; use the same mode for every sensor of this class (build "
            "separate scenes to compare backends)."
        )
    metadata.contact_depth_query = mode


class ContactDepthQuerySensorMixin:
    """
    Sensor-side counterpart to ``ContactDepthQueryMetadataMixin``.

    Its ``build()`` resolves the class-wide contact-depth backend (raising if this class's sensors disagree) and
    activates only the chosen backend: the shared ``RaycastContext`` BVH in raycast mode, else the collider SDF.
    Mixing this in (alongside the metadata mixin) means subclasses get the resolution from the super-``build()`` chain
    and need not call ``resolve_contact_depth_query`` themselves. Requires ``_shared_metadata``,
    ``_options.contact_depth_query``, and a ``_shared_context`` ``RaycastContext``.
    """

    def build(self):
        super().build()
        resolve_contact_depth_query(self._shared_metadata, self._options.contact_depth_query, type(self).__name__)
        if self._shared_metadata.contact_depth_query == "raycast":
            self._shared_context.activate()  # builds the BVH lazily, only on raycast opt-in
        else:
            self._shared_metadata.solver.collider.activate_sdf()
