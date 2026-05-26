# pyright: reportInvalidTypeForm=false
"""Numerical-equivalence test for the 32x32 register-tile Cholesky primitive.

This test exercises `genesis.utils._tile32.Tile32x32Cholesky` directly via a small standalone kernel and checks the
factorization against numpy on a known SPD matrix. It does *not* go through the rigid solver — it validates just the
tile primitive.

Mirrors the spirit of the upstream `test_subgroup_*` tests in `quadrants/tests/python/test_simt.py`: build a one-warp
kernel, run it on a small input, compare to a CPU reference.

Run on GPU only; the tile primitive requires GPU backend.
"""
from __future__ import annotations

import numpy as np
import pytest
import quadrants as qd

import genesis as gs
from genesis.utils._tile32 import Tile32x32Cholesky


def _make_spd_matrix(n: int, seed: int = 0) -> np.ndarray:
    """Build an SPD float32 matrix of size n x n with a well-conditioned spectrum."""
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((n, n)).astype(np.float32)
    A = R.T @ R + (n * 1e-2) * np.eye(n, dtype=np.float32)
    # Symmetrize numerically (R^T R already symmetric in exact arithmetic).
    return ((A + A.T) * 0.5).astype(np.float32)


@pytest.fixture(autouse=True)
def _gs_init(request):
    # Skip on CPU backends — tile32 is GPU only.
    backend = getattr(gs, "cuda", None)
    if backend is None:
        pytest.skip("genesis missing gs.cuda")
    if not hasattr(gs, "device") or getattr(gs.device, "type", "") != "cuda":
        try:
            gs.init(backend=gs.cuda, precision="32")
        except Exception as exc:
            pytest.skip(f"GPU init failed: {exc}")
    yield


def _run_tile32_cholesky(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Run the tile32 Cholesky on a single 32x32 SPD matrix.

    Returns the lower-triangular L (numpy fp32) such that A ≈ L @ L^T.
    """
    n = A.shape[0]
    assert n == 32, f"this helper expects 32x32 input, got {n}x{n}"
    H = qd.field(dtype=qd.f32, shape=(1, n, n))
    L = qd.field(dtype=qd.f32, shape=(1, n, n))
    # Fill H from numpy.
    H_np = A.astype(np.float32)
    H.from_numpy(H_np.reshape(1, n, n))

    @qd.kernel
    def factor():
        qd.loop_config(block_dim=32)
        for i_flat in range(1 * 32):
            i_b = i_flat // 32
            t = Tile32x32Cholesky.zeros(dtype=qd.f32)
            t._load3d(H, i_b, 0, 32, 0, 32)
            t.cholesky_(qd.f32(eps))
            t._store3d(L, i_b, 0, 32, 0, 32)

    factor()
    L_np = L.to_numpy()[0]
    return L_np


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tile32_cholesky_matches_numpy(seed):
    """tile32.cholesky_ on a random 32x32 SPD matrix matches numpy's np.linalg.cholesky within fp32 tolerance."""
    A = _make_spd_matrix(32, seed=seed)
    L_ours = _run_tile32_cholesky(A, eps=1e-10)
    L_ref = np.linalg.cholesky(A).astype(np.float32)

    # Only the lower triangle is meaningful in our output (upper is whatever the load leaves it, since cholesky_ writes
    # only diag + below-diag registers). Compare lower triangle including diagonal.
    tri = np.tril_indices(32)
    diff = np.abs(L_ours[tri] - L_ref[tri])
    rel = diff / (np.abs(L_ref[tri]) + 1e-6)
    max_abs = float(diff.max())
    max_rel = float(rel.max())
    # Reconstruction check: L L^T ≈ A.
    A_rec = L_ours @ L_ours.T
    # mask to lower triangle since our upper is undefined.
    A_lower = np.tril(A)
    A_rec_lower = np.tril(A_rec)
    recon_err = float(np.linalg.norm(A_lower - A_rec_lower) / np.linalg.norm(A_lower))

    assert max_abs < 1e-3, f"tile32 vs numpy diff too large: max_abs={max_abs}, max_rel={max_rel}"
    assert recon_err < 1e-4, f"L L^T does not reconstruct A: relerr={recon_err}"
