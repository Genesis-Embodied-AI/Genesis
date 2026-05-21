"""Watertight wrap of a triangle soup via SDF dual contouring + feature-preserving QEM decimation.

The main entrypoint is `watertighten_mesh`; see its docstring for the algorithm steps, parameter meanings, and the
adaptive defaults.
"""

from typing import Tuple

import igl
import numba as nb
import numpy as np
import trimesh

import genesis as gs


# SDF grid `pitch` is picked as `max(feature_size * PITCH_FEATURE_FRACTION, bbox_diag / MAX_CELLS_AXIS)`, clamped to the
# `[MIN_PITCH_ABS, MAX_ALPHA / PITCH_RATIO]` interval. When the bbox-ratio cap fires we warn that sub-pitch features are
# lost. `alpha = pitch * PITCH_RATIO` keeps the iso-surface always between adjacent grid samples.
MIN_PITCH_ABS = 2e-4
MAX_ALPHA = 0.05
MAX_CELLS_AXIS = 450
PITCH_FEATURE_FRACTION = 0.4
PITCH_RATIO = 1.5
# Low percentile so a small rim region inside a bulkier asset still drives the pitch; the strict minimum is avoided so
# degenerate / grazing / open-shell rays cannot pull the estimate to zero.
FEATURE_SIZE_PERCENTILE = 10.0


# ===============================================================================================================
# Numpy / numba vector helpers
# ===============================================================================================================

# Cube corner offsets: 8 corners at the unit cube corners. Edge endpoints: 12 (c0, c1) pairs into the corners array.
# Corners 0..7 are at (0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,0,1), (1,0,1), (1,1,1), (0,1,1).
_CORNERS = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    dtype=np.int32,
)
_EDGE_ENDPOINTS = np.array(
    [(0, 1), (1, 2), (3, 2), (0, 3), (4, 5), (5, 6), (7, 6), (4, 7), (0, 4), (1, 5), (2, 6), (3, 7)],
    dtype=np.int32,
)


@nb.jit(nopython=True, cache=True)
def _solve3x3(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
    """Solve a symmetric 3x3 system `A x = b` via cofactor expansion; returns `(x, det)` with caller fallback near zero.

    The cofactor form skips the LU factorisation `np.linalg.solve` would do, which pays off on the millions of single
    solves the QEM inner loop runs through `_q_optimal`.
    """
    cof = np.empty(3, dtype=A.dtype)
    cof[0] = A[1, 1] * A[2, 2] - A[1, 2] * A[1, 2]
    cof[1] = A[0, 2] * A[1, 2] - A[0, 1] * A[2, 2]
    cof[2] = A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]
    det = A[0, 0] * cof[0] + A[0, 1] * cof[1] + A[0, 2] * cof[2]
    if abs(det) < 1e-12:
        return np.zeros(3, dtype=A.dtype), det
    inv_det = 1.0 / det
    diag_aa = A[0, 0] * A[2, 2] - A[0, 2] * A[0, 2]
    diag_bb = A[0, 0] * A[1, 1] - A[0, 1] * A[0, 1]
    cross = A[0, 1] * A[0, 2] - A[0, 0] * A[1, 2]
    x = np.empty(3, dtype=A.dtype)
    x[0] = (cof[0] * b[0] + cof[1] * b[1] + cof[2] * b[2]) * inv_det
    x[1] = (cof[1] * b[0] + diag_aa * b[1] + cross * b[2]) * inv_det
    x[2] = (cof[2] * b[0] + cross * b[1] + diag_bb * b[2]) * inv_det
    return x, det


def _batched_solve3x3(A: np.ndarray, b: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Solve a batch of symmetric 3x3 systems with per-row fallback on singular blocks.

    Do not replace `np.linalg.solve` here with a numba cofactor loop: the per-row float rounding order differs from
    numpy's LU decomp, and the `test_nonconvex_nonwatertight_collision` sim assertion is sensitive to that. The
    `b[..., None]` / `[..., 0]` dance is needed because `np.linalg.solve` treats a 2D RHS as a matrix `(M, K)`, not
    a stack of column vectors.
    """
    det = np.linalg.det(A)
    ok = np.abs(det) > 1e-12
    out = fallback.copy()
    if ok.any():
        out[ok] = np.linalg.solve(A[ok], b[ok][..., None])[..., 0]
    return out


# ===============================================================================================================
# Separable 3D Gaussian blur
# ===============================================================================================================


@nb.jit(nopython=True, cache=True)
def _gaussian_axis(field: np.ndarray, kernel: np.ndarray, axis: int, out: np.ndarray) -> None:
    """Separable 1D Gaussian convolution along `axis` with `nearest` boundary.

    Three branches because numba does not lift the axis index out of the inner loop.
    """
    shape = field.shape
    half = kernel.shape[0] // 2
    if axis == 0:
        for j in range(shape[1]):
            for k in range(shape[2]):
                for i in range(shape[0]):
                    acc = 0.0
                    for t in range(kernel.shape[0]):
                        idx = min(max(i + t - half, 0), shape[0] - 1)
                        acc += field[idx, j, k] * kernel[t]
                    out[i, j, k] = acc
    elif axis == 1:
        for i in range(shape[0]):
            for k in range(shape[2]):
                for j in range(shape[1]):
                    acc = 0.0
                    for t in range(kernel.shape[0]):
                        idx = min(max(j + t - half, 0), shape[1] - 1)
                        acc += field[i, idx, k] * kernel[t]
                    out[i, j, k] = acc
    else:
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    acc = 0.0
                    for t in range(kernel.shape[0]):
                        idx = min(max(k + t - half, 0), shape[2] - 1)
                        acc += field[i, j, idx] * kernel[t]
                    out[i, j, k] = acc


def gaussian_blur_3d(field: np.ndarray, sigma: float) -> np.ndarray:
    """Separable 3D Gaussian blur. Returns a freshly-allocated array."""
    if sigma <= 0.0:
        return field.copy()
    radius = max(1, int(4.0 * sigma + 0.5))
    x = np.arange(-radius, radius + 1, dtype=field.dtype)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    a = np.empty_like(field)
    b = np.empty_like(field)
    _gaussian_axis(field, kernel, 0, a)
    _gaussian_axis(a, kernel, 1, b)
    _gaussian_axis(b, kernel, 2, a)
    return a


# ===============================================================================================================
# SDF gradient (central differences)
# ===============================================================================================================


def _sdf_gradient(field: np.ndarray, pitch: float) -> np.ndarray:
    """Central-difference gradient of `field` packed as `(3, *field.shape)`, with forward/backward at the boundary."""
    grad = np.empty((3,) + field.shape, dtype=field.dtype)
    inv = 1.0 / pitch
    half = 0.5 * inv
    for axis in range(3):
        moved = np.moveaxis(field, axis, 0)
        grad_axis = np.empty_like(moved)
        grad_axis[1:-1] = (moved[2:] - moved[:-2]) * half
        grad_axis[0] = (moved[1] - moved[0]) * inv
        grad_axis[-1] = (moved[-1] - moved[-2]) * inv
        grad[axis] = np.moveaxis(grad_axis, 0, axis)
    return grad


# ===============================================================================================================
# Dual contouring iso-surface extraction (vectorised over active cells)
# ===============================================================================================================


def _dc_extract(
    field: np.ndarray, grad: np.ndarray, level: float, pitch: float, origin: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Dual contouring: one QEF-solved vertex per active cell + quads per sign-changing axis-aligned grid edge.

    Vectorised over active cells: each of the 12 cube edges contributes one batched gather + outer-product update; the
    per-cell 3x3 QEF solve is one batched `np.linalg.solve` (with cell-centre regularisation so coplanar-crossing cells
    stay well-defined); face emission is three slicing passes (one per axis). Flat features land flat regardless of
    grid orientation (in contrast to naive surface nets, which jitters on diagonal walls).
    """
    shape = field.shape
    sign = field < level
    cube_idx = sign[:-1, :-1, :-1].astype(np.uint8)
    cube_idx |= sign[1:, :-1, :-1].astype(np.uint8) << 1
    cube_idx |= sign[1:, 1:, :-1].astype(np.uint8) << 2
    cube_idx |= sign[:-1, 1:, :-1].astype(np.uint8) << 3
    cube_idx |= sign[:-1, :-1, 1:].astype(np.uint8) << 4
    cube_idx |= sign[1:, :-1, 1:].astype(np.uint8) << 5
    cube_idx |= sign[1:, 1:, 1:].astype(np.uint8) << 6
    cube_idx |= sign[:-1, 1:, 1:].astype(np.uint8) << 7
    active = (cube_idx != 0) & (cube_idx != 255)
    cell = np.stack(np.where(active), axis=1).astype(np.int32)
    n_active = cell.shape[0]
    if n_active == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)
    # (i, j, k) -> active-cell index, used both by face emission and by edge gathers.
    cell_vid = np.full((shape[0] - 1, shape[1] - 1, shape[2] - 1), -1, dtype=np.int32)
    cell_vid[cell[:, 0], cell[:, 1], cell[:, 2]] = np.arange(n_active, dtype=np.int32)

    # Per-cell QEF accumulators: A is the 3x3 outer-product sum, b is the (A . crossing) sum projected on the tangent
    # normal, c_sum is the unweighted centroid of crossings (used as singular-system fallback), ncross is the
    # active-edge count per cell.
    A = np.zeros((n_active, 3, 3), dtype=np.float64)
    b = np.zeros((n_active, 3), dtype=np.float64)
    c_sum = np.zeros((n_active, 3), dtype=np.float64)
    ncross = np.zeros(n_active, dtype=np.int32)

    # Iterate the 12 cube edges. Each iteration is one set of vectorised gathers + outer-product updates; we never
    # construct a (n_cells * 12) intermediate array.
    for edge in range(12):
        off0 = _CORNERS[_EDGE_ENDPOINTS[edge, 0]]
        off1 = _CORNERS[_EDGE_ENDPOINTS[edge, 1]]
        idx0 = cell + off0  # (n_active, 3) corner-0 grid indices for this edge
        idx1 = cell + off1
        f0 = field[idx0[:, 0], idx0[:, 1], idx0[:, 2]]
        f1 = field[idx1[:, 0], idx1[:, 1], idx1[:, 2]]
        mask = (f0 < level) != (f1 < level)
        if not mask.any():
            continue
        denom = np.where(f1 == f0, 1.0, f1 - f0)
        t = np.where(mask, (level - f0) / denom, 0.0)
        # Iso-crossing position in world units: origin + (grid_index + offset0 + t * (offset1 - offset0)) * pitch.
        position = origin + (idx0 + t[:, None] * (off1 - off0)) * pitch
        # Gradient linearly interpolated between the two corners.
        g0 = grad[:, idx0[:, 0], idx0[:, 1], idx0[:, 2]].T
        g1 = grad[:, idx1[:, 0], idx1[:, 1], idx1[:, 2]].T
        normal = g0 + t[:, None] * (g1 - g0)
        gnorm = np.linalg.norm(normal, axis=1)
        valid = mask & (gnorm > 1e-12)
        if not valid.any():
            continue
        normal[valid] /= gnorm[valid, None]
        normal[~valid] = 0.0
        # Outer-product contribution per active cell + active edge: A += n n^T, b += n (n . p), c_sum += p.
        weight = valid.astype(np.float64)
        A += normal[:, :, None] * normal[:, None, :] * weight[:, None, None]
        b += normal * (np.einsum("ij,ij->i", normal, position) * weight)[:, None]
        c_sum += position * weight[:, None]
        ncross += valid.astype(np.int32)

    # Regularise toward the cell center so the solve stays well-defined where all crossings lie on a single plane.
    cell_center = origin + (cell + 0.5) * pitch
    reg = 1e-3
    A[:, 0, 0] += reg
    A[:, 1, 1] += reg
    A[:, 2, 2] += reg
    b += reg * cell_center
    fallback_center = c_sum / np.maximum(ncross[:, None], 1)
    pos = _batched_solve3x3(A, b, fallback_center)
    # Clamp each vertex to its owning cell's AABB so the surface-nets-style quad emission below stays manifold.
    cell_min = origin + cell * pitch
    pos = np.clip(pos, cell_min, cell_min + pitch)
    verts = np.ascontiguousarray(pos, dtype=np.float64)

    # Face emission. For each axis-aligned grid edge with a sign change at the right indexing window, write two
    # triangles connecting the four cell-vertices around it. The winding flips with the field sign so the outward
    # normal points from `inside` to `outside`.
    faces_axes = []
    # Axis 0: edges from (i, j, k) to (i+1, j, k), 0 <= i < nx-1, 1 <= j <= ny-2, 1 <= k <= nz-2.
    v0 = field[:-1, 1:-1, 1:-1]
    v1 = field[1:, 1:-1, 1:-1]
    sc = (v0 < level) != (v1 < level)
    if sc.any():
        ii, jj, kk = np.where(sc)
        below = v0[ii, jj, kk] < level
        c0 = cell_vid[ii, jj, kk]
        c1 = cell_vid[ii, jj + 1, kk]
        c2 = cell_vid[ii, jj + 1, kk + 1]
        c3 = cell_vid[ii, jj, kk + 1]
        faces_axes.append(_emit_quads(c0, c1, c2, c3, below))
    # Axis 1: edges from (i, j, k) to (i, j+1, k).
    v0 = field[1:-1, :-1, 1:-1]
    v1 = field[1:-1, 1:, 1:-1]
    sc = (v0 < level) != (v1 < level)
    if sc.any():
        ii, jj, kk = np.where(sc)
        below = v0[ii, jj, kk] < level
        c0 = cell_vid[ii, jj, kk]
        c1 = cell_vid[ii, jj, kk + 1]
        c2 = cell_vid[ii + 1, jj, kk + 1]
        c3 = cell_vid[ii + 1, jj, kk]
        faces_axes.append(_emit_quads(c0, c1, c2, c3, below))
    # Axis 2: edges from (i, j, k) to (i, j, k+1).
    v0 = field[1:-1, 1:-1, :-1]
    v1 = field[1:-1, 1:-1, 1:]
    sc = (v0 < level) != (v1 < level)
    if sc.any():
        ii, jj, kk = np.where(sc)
        below = v0[ii, jj, kk] < level
        c0 = cell_vid[ii, jj, kk]
        c1 = cell_vid[ii + 1, jj, kk]
        c2 = cell_vid[ii + 1, jj + 1, kk]
        c3 = cell_vid[ii, jj + 1, kk]
        faces_axes.append(_emit_quads(c0, c1, c2, c3, below))
    if not faces_axes:
        return verts, np.zeros((0, 3), dtype=np.int32)
    faces = np.concatenate(faces_axes, axis=0)
    return verts, faces


def _emit_quads(c0: np.ndarray, c1: np.ndarray, c2: np.ndarray, c3: np.ndarray, below: np.ndarray) -> np.ndarray:
    """For each sign-changing edge, emit two triangles forming a quad over the 4 surrounding cell-vertices.
    `below` selects the winding (True = inside-to-outside cross, outward normal points along +axis)."""
    n = c0.shape[0]
    faces = np.empty((2 * n, 3), dtype=np.int32)
    # CCW triangulation for outward normal on the +axis side.
    faces[0::2, 0] = c0
    faces[0::2, 1] = np.where(below, c1, c3)
    faces[0::2, 2] = c2
    faces[1::2, 0] = c0
    faces[1::2, 1] = c2
    faces[1::2, 2] = np.where(below, c3, c1)
    return faces


# ===============================================================================================================
# Quadric error mesh decimation (Garland-Heckbert with manifold preservation + feature-preserving cost cutoff)
# ===============================================================================================================


@nb.jit(nopython=True, cache=True)
def _q_cost(q: np.ndarray, vertex: np.ndarray) -> float:
    """Quadric cost `[vertex; 1]^T Q [vertex; 1]` for the packed 10-element upper-triangular `q`."""
    return (
        q[0] * vertex[0] * vertex[0]
        + 2.0 * q[1] * vertex[0] * vertex[1]
        + 2.0 * q[2] * vertex[0] * vertex[2]
        + 2.0 * q[3] * vertex[0]
        + q[4] * vertex[1] * vertex[1]
        + 2.0 * q[5] * vertex[1] * vertex[2]
        + 2.0 * q[6] * vertex[1]
        + q[7] * vertex[2] * vertex[2]
        + 2.0 * q[8] * vertex[2]
        + q[9]
    )


@nb.jit(nopython=True, cache=True)
def _q_optimal(q: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Vertex that minimises `v^T Q v` over the packed-10 quadric; falls back to `fallback` on a singular 3x3 block."""
    A = np.empty((3, 3), dtype=q.dtype)
    A[0, 0] = q[0]
    A[0, 1] = q[1]
    A[0, 2] = q[2]
    A[1, 0] = q[1]
    A[1, 1] = q[4]
    A[1, 2] = q[5]
    A[2, 0] = q[2]
    A[2, 1] = q[5]
    A[2, 2] = q[7]
    rhs = np.empty(3, dtype=q.dtype)
    rhs[0] = -q[3]
    rhs[1] = -q[6]
    rhs[2] = -q[8]
    x, det = _solve3x3(A, rhs)
    if abs(det) < 1e-12:
        return fallback
    return x


# Log-spaced bucket heap for QEM candidate edges. Push and pop are O(1) (pop scans `bh_heads` forward from the last
# known min, typically just a handful of empty buckets). Within a bucket the order is LIFO; costs in the same bucket are
# within ~0.1% of each other (`QEM_BUCKETS = 1024` over ~15 decades of cost) so the QEM collapse sequence is essentially
# unchanged. Storage layout (single unified table):
#   bh_entries[i]: (cost, u, v, ver_u, ver_v, next_idx) where `next_idx` chains within a bucket or the free list.
#   bh_heads[b]:   first entry index in bucket `b` (-1 if empty).
#   bh_state:      [free_head, min_bucket_hint, populated_count].
#   bh_params:     [log_min, log_to_bucket_scale, n_buckets].
QEM_BUCKETS = 1024
QEM_BUCKET_COST_FLOOR = 1e-30


@nb.jit(nopython=True, cache=True)
def _bh_bucket(cost: float, log_min: float, scale: float, n_buckets: int) -> int:
    """Map cost to a bucket index in `[0, n_buckets)`, clamped at both ends and at the `QEM_BUCKET_COST_FLOOR`."""
    if cost < QEM_BUCKET_COST_FLOOR:
        return 0
    b = int((np.log(cost) - log_min) * scale)
    if b < 0:
        return 0
    if b >= n_buckets:
        return n_buckets - 1
    return b


@nb.jit(nopython=True, cache=True)
def _bh_push(
    bh_entries: np.ndarray,
    bh_heads: np.ndarray,
    bh_state: np.ndarray,
    bh_params: np.ndarray,
    cost: float,
    u: int,
    v: int,
    vu: int,
    vv: int,
) -> bool:
    """Push `(cost, u, v, vu, vv)` into its log-range bucket; returns False on a full free list (entry dropped)."""
    free_head = int(bh_state[0])
    if free_head < 0:
        return False
    idx = free_head
    bh_state[0] = bh_entries[idx, 5]
    n_buckets = int(bh_params[2])
    b = _bh_bucket(cost, bh_params[0], bh_params[1], n_buckets)
    bh_entries[idx, 0] = cost
    bh_entries[idx, 1] = u
    bh_entries[idx, 2] = v
    bh_entries[idx, 3] = vu
    bh_entries[idx, 4] = vv
    bh_entries[idx, 5] = bh_heads[b]
    bh_heads[b] = idx
    if b < int(bh_state[1]):
        bh_state[1] = b
    bh_state[2] += 1
    return True


@nb.jit(nopython=True, cache=True)
def _bh_pop_min(
    bh_entries: np.ndarray, bh_heads: np.ndarray, bh_state: np.ndarray, bh_params: np.ndarray, popped: np.ndarray
) -> bool:
    """Pop the head of the lowest non-empty bucket, writing it into `popped` (cost, u, v, vu, vv); False on empty."""
    n_buckets = int(bh_params[2])
    hint = int(bh_state[1])
    while hint < n_buckets and bh_heads[hint] < 0:
        hint += 1
    bh_state[1] = hint
    if hint >= n_buckets:
        return False
    idx = bh_heads[hint]
    popped[0] = bh_entries[idx, 0]
    popped[1] = bh_entries[idx, 1]
    popped[2] = bh_entries[idx, 2]
    popped[3] = bh_entries[idx, 3]
    popped[4] = bh_entries[idx, 4]
    bh_heads[hint] = bh_entries[idx, 5]
    bh_entries[idx, 5] = bh_state[0]
    bh_state[0] = idx
    bh_state[2] -= 1
    return True


@nb.jit(nopython=True, cache=True)
def _vf_add(v: int, f: int, vf: np.ndarray, n_per: np.ndarray) -> bool:
    """Append face `f` to v's incidence list; False on overflow so the caller can resize and retry."""
    c = n_per[v]
    if c >= vf.shape[1]:
        return False
    vf[v, c] = f
    n_per[v] = c + 1
    return True


@nb.jit(nopython=True, cache=True)
def _vf_remove(v: int, f: int, vf: np.ndarray, n_per: np.ndarray) -> None:
    """Swap-remove face `f` from v's incidence list; no-op if not present."""
    c = n_per[v]
    for k in range(c):
        if vf[v, k] == f:
            vf[v, k] = vf[v, c - 1]
            n_per[v] = c - 1
            return


@nb.jit(nopython=True, cache=True)
def _neighbors(v: int, faces: np.ndarray, vf: np.ndarray, n_per: np.ndarray, out: np.ndarray) -> int:
    """Write distinct vertex neighbours of `v` into `out` and return the count, or -1 on `out` overflow."""
    n = 0
    cap = out.shape[0]
    for k in range(n_per[v]):
        f = vf[v, k]
        for s in range(3):
            u = faces[f, s]
            if u == v:
                continue
            dup = False
            for q in range(n):
                if out[q] == u:
                    dup = True
                    break
            if not dup:
                if n >= cap:
                    return -1
                out[n] = u
                n += 1
    return n


@nb.jit(nopython=True, cache=True)
def _collapse_safe(
    u: int,
    v: int,
    target: np.ndarray,
    verts: np.ndarray,
    faces: np.ndarray,
    vf: np.ndarray,
    n_per: np.ndarray,
    ring_u: np.ndarray,
    ring_v: np.ndarray,
) -> bool:
    """Return True iff collapsing edge (u, v) to `target` preserves manifold topology and triangle orientation.

    The link condition (Edelsbrunner) requires exactly two common neighbours of u and v (the wings of the two triangles
    incident on the edge); anything else creates non-manifold topology. Each surviving incident face is then checked
    for a normal flip with the moved vertex plugged in.
    """
    nu = _neighbors(u, faces, vf, n_per, ring_u)
    nv = _neighbors(v, faces, vf, n_per, ring_v)
    if nu < 0 or nv < 0:
        return False
    shared = 0
    for i in range(nu):
        w = ring_u[i]
        for j in range(nv):
            if ring_v[j] == w:
                shared += 1
                break
    if shared != 2:
        return False
    for endpoint in (u, v):
        replaced = u if endpoint == u else v
        for k in range(n_per[endpoint]):
            f = vf[endpoint, k]
            a = faces[f, 0]
            b_ = faces[f, 1]
            c = faces[f, 2]
            if (a == u or b_ == u or c == u) and (a == v or b_ == v or c == v):
                continue  # vanishes on collapse
            # Triangle vertices stacked as a (3, 3) array, then patched with the target position.
            tri = np.empty((3, 3), dtype=verts.dtype)
            tri[0] = verts[a]
            tri[1] = verts[b_]
            tri[2] = verts[c]
            normal_before = np.empty(3, dtype=verts.dtype)
            edge1 = tri[1] - tri[0]
            edge2 = tri[2] - tri[0]
            normal_before[0] = edge1[1] * edge2[2] - edge1[2] * edge2[1]
            normal_before[1] = edge1[2] * edge2[0] - edge1[0] * edge2[2]
            normal_before[2] = edge1[0] * edge2[1] - edge1[1] * edge2[0]
            if a == replaced:
                tri[0] = target
            if b_ == replaced:
                tri[1] = target
            if c == replaced:
                tri[2] = target
            edge1 = tri[1] - tri[0]
            edge2 = tri[2] - tri[0]
            normal_after = np.empty(3, dtype=verts.dtype)
            normal_after[0] = edge1[1] * edge2[2] - edge1[2] * edge2[1]
            normal_after[1] = edge1[2] * edge2[0] - edge1[0] * edge2[2]
            normal_after[2] = edge1[0] * edge2[1] - edge1[1] * edge2[0]
            if normal_before @ normal_after <= 0.0:
                return False
    return True


def _seed_quadrics_and_heap(
    verts: np.ndarray,
    faces: np.ndarray,
    v_Q: np.ndarray,
    bh_entries: np.ndarray,
    bh_heads: np.ndarray,
    bh_state: np.ndarray,
    bh_params: np.ndarray,
    max_cost: float,
) -> int:
    """Vectorised QEM warm-start: build per-vertex quadrics + bulk-load the initial bucket heap. Returns entry count.

    Filters out edges already past `max_cost` so they don't enter the heap at all. The per-edge optimal-collapse
    position uses the symmetric 3x3 upper-left of the quadric (with edge-midpoint fallback on singular blocks),
    matching the `_q_optimal` / `_q_cost` semantics the main loop uses.
    """
    # Plane quadric per face: `plane` packs `(normal, d)` as `(n_faces, 4)`, `face_q` is the packed-10 outer product.
    tri = verts[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(normal, axis=1, keepdims=True)
    valid = norm.squeeze(-1) > 1e-30
    normal = np.where(valid[:, None], normal / np.where(norm > 1e-30, norm, 1.0), 0.0)
    plane_offset = -np.einsum("fk,fk->f", normal, tri[:, 0]) * valid
    plane = np.concatenate((normal, plane_offset[:, None]), axis=1)
    Q_I = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3], dtype=np.int64)
    Q_J = np.array([0, 1, 2, 3, 1, 2, 3, 2, 3, 3], dtype=np.int64)
    face_q = plane[:, Q_I] * plane[:, Q_J]
    np.add.at(v_Q, faces[:, 0], face_q)
    np.add.at(v_Q, faces[:, 1], face_q)
    np.add.at(v_Q, faces[:, 2], face_q)
    # One entry per unique undirected edge; sort each face corner pair into (min, max) and drop duplicates.
    edges = np.unique(np.sort(np.stack((faces, np.roll(faces, -1, axis=1)), axis=-1).reshape(-1, 2), axis=1), axis=0)
    u, v = edges[:, 0], edges[:, 1]
    q_sum = v_Q[u] + v_Q[v]
    # Per-edge optimal collapse position via the symmetric 3x3 upper-left quadric block, with edge-midpoint fallback on
    # singular rows (matches `_q_optimal`). `Q_BLOCK` indexes the symmetric 3x3 out of the packed-10 layout.
    Q_BLOCK = np.array([[0, 1, 2], [1, 4, 5], [2, 5, 7]], dtype=np.int64)
    midpoint = 0.5 * (verts[u] + verts[v])
    opt = _batched_solve3x3(q_sum[:, Q_BLOCK], -q_sum[:, [3, 6, 8]], midpoint)
    # Cost = `[opt; 1]^T Q [opt; 1]` with `Q` rebuilt as a 4x4 symmetric. Same layout as `_q_cost`.
    Q_4X4 = np.array([[0, 1, 2, 3], [1, 4, 5, 6], [2, 5, 7, 8], [3, 6, 8, 9]], dtype=np.int64)
    opt_h = np.concatenate((opt, np.ones((opt.shape[0], 1), dtype=opt.dtype)), axis=1)
    cost = np.einsum("ni,nij,nj->n", opt_h, q_sum[:, Q_4X4], opt_h)
    # Drop entries already past `max_cost`; they would never be collapsed.
    keep = cost <= max_cost
    cost_kept = cost[keep]
    u_kept = u[keep]
    v_kept = v[keep]
    capacity = bh_entries.shape[0]
    n_entries = min(cost_kept.shape[0], capacity)
    cost_kept = cost_kept[:n_entries]
    u_kept = u_kept[:n_entries]
    v_kept = v_kept[:n_entries]
    log_min, scale, n_buckets = bh_params[0], bh_params[1], int(bh_params[2])
    bh_heads[:] = -1
    if n_entries > 0:
        log_cost = np.log(np.maximum(cost_kept, QEM_BUCKET_COST_FLOOR))
        bucket_idx = np.clip(((log_cost - log_min) * scale).astype(np.int64), 0, n_buckets - 1)
        # Sort by (bucket, cost): entries land grouped by bucket and, within each bucket, sorted by cost ascending. The
        # linked list then pops the bucket's cheapest entry first, matching a strict-min priority queue for the initial
        # heap. Re-seeds during the main loop still go LIFO into bh_heads (their cost is bounded by `max_cost` and they
        # share a bucket with entries within ~0.1% of their own cost, so the order error is small in practice).
        order = np.lexsort((cost_kept, bucket_idx))
        sorted_buckets = bucket_idx[order]
        bh_entries[:n_entries, 0] = cost_kept[order]
        bh_entries[:n_entries, 1] = u_kept[order]
        bh_entries[:n_entries, 2] = v_kept[order]
        bh_entries[:n_entries, 3:5] = 0
        # `next_idx[k] = k + 1` when entry k+1 is in the same bucket as k; otherwise -1 (end-of-bucket marker).
        next_idx = np.where(sorted_buckets[:-1] == sorted_buckets[1:], np.arange(1, n_entries, dtype=np.int64), -1)
        bh_entries[: n_entries - 1, 5] = next_idx
        bh_entries[n_entries - 1, 5] = -1
        # `bh_heads[b]` is the first entry of bucket `b`, found at the first occurrence of `b` in `sorted_buckets`.
        starts = np.concatenate(([True], sorted_buckets[1:] != sorted_buckets[:-1]))
        bh_heads[sorted_buckets[starts]] = np.flatnonzero(starts)
    # Chain unused slots `n_entries..capacity-1` into the free list; `bh_state[0]` is the free-list head (-1 = full).
    if n_entries < capacity:
        bh_entries[n_entries:-1, 5] = np.arange(n_entries + 1, capacity, dtype=np.int64)
        bh_entries[-1, 5] = -1
        bh_state[0] = n_entries
    else:
        bh_state[0] = -1
    bh_state[1] = sorted_buckets[0] if n_entries > 0 else n_buckets
    bh_state[2] = n_entries
    return n_entries


@nb.jit(nopython=True, cache=True)
def _qem_main(
    verts: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
    max_cost: float,
    v_alive: np.ndarray,
    v_Q: np.ndarray,
    v_version: np.ndarray,
    f_alive: np.ndarray,
    vf: np.ndarray,
    n_per: np.ndarray,
    bh_entries: np.ndarray,
    bh_heads: np.ndarray,
    bh_state: np.ndarray,
    bh_params: np.ndarray,
) -> int:
    """Garland-Heckbert decimation with manifold guards and a feature-preserving cost cutoff.

    Returns the alive face count on exit (target reached, heap empty, or cost cutoff). Returns -1 on incidence
    buffer overflow; the caller is expected to grow the buffer and retry. The seed quadrics + heap warm-start
    are produced vectorised in `_seed_quadrics_and_heap`; this entry point only runs the inherently serial
    collapse loop, driven by the O(1) bucket-heap `_bh_push` / `_bh_pop_min` operations.
    """
    q_sum = np.empty(10, dtype=np.float64)
    ring_u = np.empty(64, dtype=np.int32)
    ring_v = np.empty(64, dtype=np.int32)
    popped = np.empty(5, dtype=np.float64)

    alive_faces = int(f_alive.sum())

    while alive_faces > target_faces:
        if not _bh_pop_min(bh_entries, bh_heads, bh_state, bh_params, popped):
            break
        cost = popped[0]
        # Bucket heap pops in approximate cost order; once the head bucket's cost exceeds the cutoff every
        # remaining entry does too, so we stop without touching any feature region.
        if cost > max_cost:
            break
        u = int(popped[1])
        v = int(popped[2])
        vu = int(popped[3])
        vv = int(popped[4])
        if not v_alive[u] or not v_alive[v]:
            continue
        if vu != v_version[u] or vv != v_version[v]:
            continue
        q_sum[:] = v_Q[u] + v_Q[v]
        mid = 0.5 * (verts[u] + verts[v])
        opt = _q_optimal(q_sum, mid)
        new_cost = _q_cost(q_sum, opt)
        if new_cost > max_cost:
            continue
        if not _collapse_safe(u, v, opt, verts, faces, vf, n_per, ring_u, ring_v):
            continue

        verts[u] = opt
        v_Q[u] = q_sum
        v_alive[v] = False
        v_version[u] += 1
        v_version[v] += 1

        # Reassign or kill v's incident faces. Snapshot the list first because we mutate vf[v] below.
        n_v = n_per[v]
        snap = vf[v, :n_v].copy()
        n_per[v] = 0
        for k in range(n_v):
            f = snap[k]
            if not f_alive[f]:
                continue
            a = faces[f, 0]
            b_ = faces[f, 1]
            c = faces[f, 2]
            if a == u or b_ == u or c == u:
                f_alive[f] = False
                alive_faces -= 1
                if a != v:
                    _vf_remove(a, f, vf, n_per)
                if b_ != v and b_ != a:
                    _vf_remove(b_, f, vf, n_per)
                if c != v and c != a and c != b_:
                    _vf_remove(c, f, vf, n_per)
                continue
            if a == v:
                faces[f, 0] = u
            if b_ == v:
                faces[f, 1] = u
            if c == v:
                faces[f, 2] = u
            if not _vf_add(u, f, vf, n_per):
                return -1

        # Re-seed candidate edges around the survivor.
        nu = _neighbors(u, faces, vf, n_per, ring_u)
        if nu < 0:
            continue
        for k in range(nu):
            w = ring_u[k]
            if not v_alive[w]:
                continue
            q_sum[:] = v_Q[u] + v_Q[w]
            mid = 0.5 * (verts[u] + verts[w])
            opt = _q_optimal(q_sum, mid)
            new_cost = _q_cost(q_sum, opt)
            # Skip re-seeds already past the cost cutoff - they would only be popped and immediately discarded.
            if new_cost <= max_cost:
                _bh_push(bh_entries, bh_heads, bh_state, bh_params, new_cost, u, w, v_version[u], v_version[w])

    return alive_faces


def _decimate_qem(verts: np.ndarray, faces: np.ndarray, target_faces: int, max_cost: float):
    """Garland-Heckbert decimation with manifold preservation + feature-preserving cost cutoff.

    Seed quadrics + bucket heap are computed once (they only depend on the input geometry) and restored via
    `numpy.copyto` on each retry of the `max_deg` doubling loop.
    """
    verts = np.ascontiguousarray(verts, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    n_verts = verts.shape[0]
    n_faces = faces.shape[0]
    if n_faces <= target_faces:
        return verts.copy(), faces.copy()
    # `6 * n_faces` slots covers initial unique edges (~3 * n_faces / 2) plus a per-collapse re-seed budget
    # (~6 neighbours * n_faces / 2) on every input we've tried; the 1M floor keeps small assets usable.
    capacity = max(6 * n_faces, 1_000_000)
    bh_entries = np.zeros((capacity, 6), dtype=np.float64)
    bh_heads_pristine = np.full(QEM_BUCKETS, -1, dtype=np.int64)
    bh_state_pristine = np.zeros(3, dtype=np.int64)
    # ~15 decades of log span below `max_cost` matches the QEM dynamic range on the reference assets.
    log_max = np.log(max(max_cost, QEM_BUCKET_COST_FLOOR)) + 1e-3
    log_min = log_max - 15.0 * np.log(10.0)
    bh_params = np.array([log_min, QEM_BUCKETS / (log_max - log_min), QEM_BUCKETS], dtype=np.float64)
    v_Q_pristine = np.zeros((n_verts, 10), dtype=np.float64)
    entries_pristine = np.empty_like(bh_entries)
    _seed_quadrics_and_heap(
        verts,
        faces,
        v_Q_pristine,
        bh_entries,
        bh_heads_pristine,
        bh_state_pristine,
        bh_params,
        max_cost,
    )
    np.copyto(entries_pristine, bh_entries)
    bh_heads = np.empty_like(bh_heads_pristine)
    bh_state = np.empty_like(bh_state_pristine)
    v_Q = np.empty_like(v_Q_pristine)
    max_deg = 32
    while True:
        verts_w = verts.copy()
        faces_w = faces.copy()
        vf = np.full((n_verts, max_deg), -1, dtype=np.int32)
        n_per = np.zeros(n_verts, dtype=np.int32)
        v_alive = np.ones(n_verts, dtype=np.bool_)
        v_version = np.zeros(n_verts, dtype=np.int32)
        f_alive = np.ones(n_faces, dtype=np.bool_)
        if n_per_overflows(faces_w, n_verts, max_deg):
            max_deg *= 2
            continue
        _init_vf(faces_w, vf, n_per)
        np.copyto(bh_entries, entries_pristine)
        np.copyto(bh_heads, bh_heads_pristine)
        np.copyto(bh_state, bh_state_pristine)
        np.copyto(v_Q, v_Q_pristine)
        result = _qem_main(
            verts_w,
            faces_w,
            target_faces,
            max_cost,
            v_alive,
            v_Q,
            v_version,
            f_alive,
            vf,
            n_per,
            bh_entries,
            bh_heads,
            bh_state,
            bh_params,
        )
        if result == -1:
            max_deg *= 2
            continue
        break
    used = v_alive
    new_vid = np.full(n_verts, -1, dtype=np.int32)
    new_vid[used] = np.arange(used.sum(), dtype=np.int32)
    new_verts = verts_w[used]
    new_faces = new_vid[faces_w[f_alive]]
    return new_verts, new_faces


@nb.jit(nopython=True, cache=True)
def n_per_overflows(faces: np.ndarray, n_verts: int, max_deg: int) -> bool:
    """Return True iff any vertex's incident-face count exceeds `max_deg`."""
    counts = np.zeros(n_verts, dtype=np.int32)
    for f in range(faces.shape[0]):
        for s in range(3):
            counts[faces[f, s]] += 1
            if counts[faces[f, s]] > max_deg:
                return True
    return False


@nb.jit(nopython=True, cache=True)
def _init_vf(faces: np.ndarray, vf: np.ndarray, n_per: np.ndarray) -> None:
    for f in range(faces.shape[0]):
        for s in range(3):
            v = faces[f, s]
            vf[v, n_per[v]] = f
            n_per[v] += 1


# ===============================================================================================================
# Adaptive parameter selection + top-level orchestrator
# ===============================================================================================================


SDF_REFINE_FACTOR = 4


def _sdf_field(verts: np.ndarray, faces: np.ndarray, alpha: float, pitch: float, blur_radius: int):
    """Two-pass unsigned distance field: coarse SDF identifies the iso band, fine SDF is computed only inside it.

    The 1-Lipschitz property of the unsigned distance proves any coarse block whose `[block_min, block_max]` does not
    overlap `[alpha - margin, alpha + margin]` cannot contain an iso-crossing fine cell. The fine band is dilated by
    `blur_radius` voxels so the downstream Gaussian-blur stencil only ever reads cells with real fine SDF (otherwise
    the trilinear-upsampled coarse values bleed into the blurred iso band and the downstream sim assertion is
    sensitive enough to fail). Far cells (outside the dilated band) get trilinear-upsampled coarse values; their sign
    relative to `alpha` is correct by Lipschitz so DC does not emit a vertex there, and their exact value is irrelevant.

    Returns the field (shape `(n_x, n_y, n_z)`) and the world-space origin of the `(0, 0, 0)` corner.
    """
    refine = SDF_REFINE_FACTOR
    coarse_pitch = refine * pitch
    verts = verts.astype(np.float64, copy=False)
    faces = faces.astype(np.int32, copy=False)
    aabb_min = verts.min(axis=0) - (alpha + 2.0 * pitch)
    aabb_max = verts.max(axis=0) + (alpha + 2.0 * pitch)
    n_fine = np.ceil((aabb_max - aabb_min) / pitch).astype(np.int64) + 1
    # Round up so fine[::refine] always lands inside the coarse grid even after the upsampling crop below.
    n_coarse = (n_fine + refine - 1) // refine + 1
    coarse_axes = [aabb_min[d] + np.arange(n_coarse[d]) * coarse_pitch for d in range(3)]
    coarse_grid = np.stack(np.meshgrid(*coarse_axes, indexing="ij"), axis=-1)
    sqd_coarse, _, _ = igl.point_mesh_squared_distance(coarse_grid.reshape(-1, 3), verts, faces)
    coarse_field = np.sqrt(sqd_coarse).reshape(n_coarse)

    # Per-coarse-block min/max across the eight cube corners via a 2x2x2 sliding window.
    blocks = np.lib.stride_tricks.sliding_window_view(coarse_field, (2, 2, 2))
    margin = coarse_pitch * (3.0**0.5) * 0.5
    near_block = (blocks.min(axis=(-3, -2, -1)) < alpha + margin) & (blocks.max(axis=(-3, -2, -1)) > alpha - margin)
    # Fine mask: repeat each near-block flag by `refine` along every axis, then trim/pad to `n_fine`. Tail cells outside
    # the coarse-block grid land on the SDF padding region and are safely far from any feature; default to True.
    near_fine = np.repeat(np.repeat(np.repeat(near_block, refine, 0), refine, 1), refine, 2)
    pad_widths = [(0, max(0, n_fine[d] - near_fine.shape[d])) for d in range(3)]
    near_fine = np.pad(near_fine, pad_widths, constant_values=True)[: n_fine[0], : n_fine[1], : n_fine[2]]
    # Dilate by `blur_radius` (cubic structuring element, separable per axis as shifted ORs over `2 * radius + 1`).
    if blur_radius > 0 and near_fine.any():
        for axis in range(3):
            moved = np.moveaxis(near_fine, axis, 0).copy()
            dilated = moved.copy()
            for shift in range(1, blur_radius + 1):
                dilated[shift:] |= moved[:-shift]
                dilated[:-shift] |= moved[shift:]
            near_fine = np.moveaxis(dilated, 0, axis)
    # Trilinear upsample of `coarse_field` for the far region (samples at fine[::refine] match coarse exactly; the last
    # `refine - 1` cells along each axis nearest-extrapolate).
    field = coarse_field
    for axis in range(3):
        moved = np.moveaxis(field, axis, 0)
        n_axis = moved.shape[0] * refine
        positions = np.arange(n_axis) * (1.0 / refine)
        k_lo = np.minimum(np.floor(positions).astype(np.int64), moved.shape[0] - 1)
        k_hi = np.minimum(k_lo + 1, moved.shape[0] - 1)
        weight_hi = (positions - k_lo).reshape((-1,) + (1,) * (moved.ndim - 1))
        field = np.moveaxis((1.0 - weight_hi) * moved[k_lo] + weight_hi * moved[k_hi], 0, axis)
    field = field[: n_fine[0], : n_fine[1], : n_fine[2]].astype(np.float64, copy=False)
    if near_fine.any():
        fine_axes = [aabb_min[d] + np.arange(n_fine[d]) * pitch for d in range(3)]
        fine_grid = np.stack(np.meshgrid(*fine_axes, indexing="ij"), axis=-1)
        near_pts = fine_grid[near_fine]
        sqd_fine, _, _ = igl.point_mesh_squared_distance(near_pts, verts, faces)
        field[near_fine] = np.sqrt(sqd_fine)
    return field, aabb_min


def _estimate_feature_size(verts: np.ndarray, faces: np.ndarray) -> float:
    """Return the `FEATURE_SIZE_PERCENTILE`-th percentile local solid thickness across the surface.

    For each face we cast a ray from its centroid along the inward normal and measure the distance to the first
    other-side hit; that distance is the local solid thickness (= the largest inscribed sphere touching the face). The
    asset-wide percentile is the wrap's pitch driver. Falls back to `bbox_diag` if no rays land.
    """
    src = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    bbox_diag = np.linalg.norm(src.extents)
    if bbox_diag <= 0.0 or len(src.faces) == 0:
        return bbox_diag
    centroids = src.triangles_center
    normals = src.face_normals
    # Subsample - ray casting on the BVH is linear in ray count and a few thousand rays already saturates coverage.
    if len(centroids) > 4000:
        idx = np.random.default_rng(0).choice(len(centroids), 4000, replace=False)
        centroids = centroids[idx]
        normals = normals[idx]
    origins = centroids - (bbox_diag * 1e-5) * normals
    locs, ray_idx, _ = src.ray.intersects_location(origins, -normals, multiple_hits=False)
    if len(locs) == 0:
        return bbox_diag
    distances = np.linalg.norm(locs - origins[ray_idx], axis=1)
    return np.percentile(distances, FEATURE_SIZE_PERCENTILE)


def _adaptive_params(verts: np.ndarray, faces: np.ndarray, aggressiveness: int):
    """Pick alpha / pitch / max_cost from the input's estimated feature size, with a bbox-derived compute cap.

    `max_cost = (aggressiveness * alpha / 6)^2` is the empirical cost-to-aggressiveness mapping that produces visually
    comparable output across asset sizes (aggressiveness=4 -> `max_cost ~ 1e-3` on a 22 m asset).
    """
    bbox_diag = np.linalg.norm(verts.max(axis=0) - verts.min(axis=0))
    feature_size = _estimate_feature_size(verts, faces)
    pitch_feature = max(MIN_PITCH_ABS, feature_size * PITCH_FEATURE_FRACTION)
    pitch_compute = bbox_diag / MAX_CELLS_AXIS
    pitch = min(max(pitch_feature, pitch_compute), MAX_ALPHA / PITCH_RATIO)
    alpha = pitch * PITCH_RATIO
    if pitch_compute > pitch_feature:
        gs.logger.warning(
            f"Watertighten: compute cap reached (feature_size={feature_size * 1000.0:.2f} mm, "
            f"bbox_diag={bbox_diag * 1000.0:.0f} mm). Sampling at pitch={pitch * 1000.0:.2f} mm to keep the SDF grid "
            f"<= {MAX_CELLS_AXIS}^3 cells; features below ~{2.0 * pitch * 1000.0:.1f} mm will be lost in the wrap."
        )
    if aggressiveness >= 8:
        max_cost = float("inf")
    else:
        max_cost = (aggressiveness * alpha / 6.0) ** 2
    return alpha, pitch, max_cost


def watertighten_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    aggressiveness: int = 7,
    target_face_num: int = 500,
    sigma: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a watertight wrap of the input triangle soup, strictly outside it and within `alpha` of it.

    Pipeline: padded unsigned distance field on a regular grid (`_sdf_field`); separable Gaussian blur to kill
    sub-voxel aliasing; dual-contouring iso-surface extraction at level `alpha` (`_dc_extract`); analytical snap onto
    the `alpha`-isosurface to remove SDF-grid noise; quadric-error decimation with manifold + normal-flip guards
    (`_decimate_qem`); inward ray-cast snap of each wrap vertex onto the source surface where the source is locally
    closed (vertices that bridge open holes stay at the `alpha`-iso to preserve the wrap closure). All adaptive
    parameters (`alpha`, `pitch`, cost cutoff) come from `_adaptive_params`.

    Parameters
    ----------
    aggressiveness
        Integer 0..8. Pass 0 to bypass the wrap (returns the input unchanged). 1 is near-lossless, 5 collapses flat
        regions while preserving features, 8 ignores the cost cutoff entirely.
    target_face_num
        Soft floor: decimation stops at this many faces or when the cheapest remaining collapse exceeds the
        aggressiveness-derived cost threshold, whichever happens first.
    sigma
        Gaussian blur width on the SDF in voxel units. The default 0.8 kills sub-voxel aliasing without softening
        real features.
    """
    if aggressiveness <= 0:
        return verts.copy(), faces.copy()
    verts = np.ascontiguousarray(verts, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    alpha, pitch, max_cost = _adaptive_params(verts, faces, aggressiveness)
    blur_radius = max(1, int(4.0 * sigma + 0.5)) if sigma > 0.0 else 0
    field, origin = _sdf_field(verts, faces, alpha, pitch, blur_radius=blur_radius)
    field = gaussian_blur_3d(field, sigma)
    grad = _sdf_gradient(field, pitch)
    v, f = _dc_extract(field, grad, alpha, pitch, origin)
    # One Newton step on the unsigned distance: snap each wrap vertex to the analytical `alpha`-isosurface of the source
    # mesh. Erases SDF-grid discretisation noise without changing topology - flat source regions stay flat in the wrap.
    _, _, closest = igl.point_mesh_squared_distance(v, verts, faces)
    outward = v - closest
    out_norm = np.linalg.norm(outward, axis=1, keepdims=True)
    moved = out_norm[:, 0] > 1e-9
    v[moved] = (closest + alpha * (outward / np.maximum(out_norm, 1e-12)))[moved]
    v, f = _decimate_qem(v, f, target_faces=max(2 * target_face_num, 1), max_cost=max_cost)
    # Snap each wrap vertex onto its nearest source point - removes the `alpha`-offset inflation entirely on solid
    # bulk regions where the source has well-defined two-sided geometry. Vertices that bridge an open hole in the
    # source must NOT be snapped: their nearest source point sits on the hole's rim, and pulling them onto it would
    # collapse the wrap onto a sliver of geometry, destroying the closure that the wrap is supposed to provide.
    # We distinguish the two cases with a single inward ray cast: bulk vertices hit the source on their way inward,
    # hole-bridging vertices do not.
    _, _, closest = igl.point_mesh_squared_distance(v, verts, faces)
    direction = closest - v
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    inward = direction / np.maximum(norm, 1e-12)
    src = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    _, ray_idx, _ = src.ray.intersects_location(v, inward, multiple_hits=False)
    has_hit = np.zeros(v.shape[0], dtype=np.bool_)
    has_hit[ray_idx] = True
    v[has_hit] = closest[has_hit]
    return v, f
