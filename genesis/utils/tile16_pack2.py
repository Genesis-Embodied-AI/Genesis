# pyright: reportInvalidTypeForm=false

"""Pack-2 register-resident 16x16 tile operations.

Local copy of ``quadrants/python/quadrants/lang/simt/_tile16.py``, modified so that a single 32-lane CUDA warp
processes two independent 16x16 tiles in parallel (lanes 0-15 = tile A, lanes 16-31 = tile B). When invoked from
a kernel launched with ``block_dim=32`` and a per-half-warp ``i_b`` (e.g. ``i_b = i // 16``), every method below
correctly handles two envs per warp with no cross-tile contamination.

When invoked from a kernel launched with ``block_dim=16`` (the legacy genesis layout), every method below is
*identical in behavior* to the upstream ``qd.simt.Tile16x16``: the per-warp ``tile_base`` is always 0 and
``local == tid``, so the pack-2 codepath degenerates to the original.

The changes from upstream are confined to:

  - Compute ``tile_base = (tid >> 4) << 4`` and ``local = tid - tile_base`` once per method.
  - Use ``local`` (not ``tid``) for any row arithmetic and for ``tid == k`` / ``tid > k`` predicates.
  - Use ``tile_base + qd.i32(k)`` (not ``qd.u32(k)``) as the source-lane index in ``subgroup.shuffle``.

Two quadrants-side miscompilation traps avoided in this file (both produce silently-wrong codegen as of commit
``b0906ebd`` on the cluster image):

  - ``tid & qd.i32(~15)`` does *not* mask correctly. Use ``(tid >> qd.i32(4)) << qd.i32(4)`` instead.
  - ``qd.i32(env_pair_base * 2)`` does *not* multiply correctly when ``env_pair_base`` is a kernel-loop i32.
    Use ``env_pair_base << qd.i32(1)`` instead. (Not used in this file directly, but worth noting for
    call-site code that addresses two envs per warp.)

See ``perso_hugh/doc/cholesky_mjw_vs_gs_2026may21.md`` (sections T17 + T18) for the design rationale and
end-to-end benchmark.
"""

from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any, NoReturn

import quadrants as qd

# Import the upstream OuterProduct / VecSliceProxy classes so that `qd.outer(...)` and 2D/3D vector
# slices coming from kernel code (which use the upstream classes) are correctly recognized by our
# local _augassign / _resolve_vec_proxy below. (Mixing the local copies would silently break
# `tile -= qd.outer(v, v)` because the isinstance check would fail.)
from quadrants.lang.simt._tile16 import _OuterProduct as _UpstreamOuterProduct
from quadrants.lang.simt._tile16 import _VecSliceProxy as _UpstreamVecSliceProxy
from quadrants.lang.simt._tile16 import _tile16_cache as _upstream_tile16_cache

if _TYPE_CHECKING:

    class _Tile16x16Proto:  # noqa: E303
        SIZE: int

        def __init__(self, *args: Any, **kwargs: Any) -> None: ...  # noqa: E704
        @classmethod
        def zeros(cls) -> "_Tile16x16Proto": ...  # noqa: E704
        @classmethod
        def eye(cls) -> "_Tile16x16Proto": ...  # noqa: E704
        def eye_(self) -> None: ...  # noqa: E704
        def cholesky_(self, eps: Any) -> None: ...  # noqa: E704
        def solve_triangular_(self, B: "_Tile16x16Proto", lower: bool = True) -> None: ...  # noqa: E704
        def _load(self, arr: Any, row_start: Any, row_end: Any, col_start: Any, col_end: Any) -> None: ...  # noqa: E704
        def _store(
            self, arr: Any, row_start: Any, row_end: Any, col_start: Any, col_end: Any
        ) -> None: ...  # noqa: E704
        def _load3d(
            self, arr: Any, batch: Any, row_start: Any, row_end: Any, col_start: Any, col_end: Any
        ) -> None: ...  # noqa: E704
        def _store3d(
            self, arr: Any, batch: Any, row_start: Any, row_end: Any, col_start: Any, col_end: Any
        ) -> None: ...  # noqa: E704
        def _get_col(self, k: Any) -> Any: ...  # noqa: E704
        def _set_col(self, k: Any, val: Any) -> None: ...  # noqa: E704
        def _ger_sub(self, a: Any, b: Any) -> None: ...  # noqa: E704
        def _trsm(self, L: "_Tile16x16Proto") -> None: ...  # noqa: E704
        def __isub__(self, other: Any) -> "_Tile16x16Proto": ...  # noqa: E704
        def __getitem__(self, key: Any) -> Any: ...  # noqa: E704
        def __setitem__(self, key: Any, value: Any) -> None: ...  # noqa: E704


_TILE = 16


class _OuterProduct:
    """Deferred outer product proxy. See upstream for details."""

    _qd_is_deferred = True

    def __init__(self, a: Any, b: Any) -> None:
        self.a = a
        self.b = b

    def __add__(self, other: Any) -> NoReturn:
        raise TypeError("OuterProduct does not support composition; apply each update separately")

    def __radd__(self, other: Any) -> NoReturn:
        raise TypeError("OuterProduct does not support composition; apply each update separately")


def outer(a: Any, b: Any) -> _OuterProduct:
    """Create a deferred outer product. Same semantics as ``qd.outer`` upstream.

    Usage::

        t -= outer(a, b)   # equivalent to t._ger_sub(a, b)
    """
    return _OuterProduct(a, b)


class _DeferredProxyMixin:
    _proxy_description = "Tile proxy"

    def _misuse(self, op: str = "used") -> NoReturn:
        raise TypeError(
            f"{self._proxy_description} was {op}, but it is only valid in tile operations"
        )

    def __add__(self, other: Any) -> NoReturn:
        self._misuse("added")

    def __radd__(self, other: Any) -> NoReturn:
        self._misuse("added")

    def __sub__(self, other: Any) -> NoReturn:
        self._misuse("subtracted")

    def __mul__(self, other: Any) -> NoReturn:
        self._misuse("multiplied")

    def __getitem__(self, key: Any) -> NoReturn:
        self._misuse("subscripted")

    def __repr__(self) -> str:
        return f"<{self._proxy_description} — not a value>"


class _TileSliceProxy(_DeferredProxyMixin):
    """Deferred 2D/3D array slice for tile load/store."""

    _qd_is_deferred = True
    _proxy_description = "Array slice proxy (arr[r0:r1, c0:c1])"

    def __init__(
        self, arr: Any, row_start: Any, row_stop: Any, col_start: Any, col_stop: Any, batch_idx: Any = None
    ) -> None:
        self.arr = arr
        self.row_start = row_start
        self.row_stop = row_stop
        self.col_start = col_start
        self.col_stop = col_stop
        self.batch_idx = batch_idx

    def _assign(self, tile: Any) -> None:
        if self.batch_idx is not None:
            tile._store3d(self.arr, self.batch_idx, self.row_start, self.row_stop, self.col_start, self.col_stop)
        else:
            tile._store(self.arr, self.row_start, self.row_stop, self.col_start, self.col_stop)


class _VecSliceProxy(_DeferredProxyMixin):
    """Deferred column-vector load. Per-lane scalar."""

    _qd_is_deferred = True
    _proxy_description = "Vec slice proxy (arr[r0:r1, col])"

    def __init__(self, arr: Any, row_start: Any, row_stop: Any, col: Any, batch_idx: Any = None) -> None:
        self.arr = arr
        self.row_start = row_start
        self.row_stop = row_stop
        self.col = col
        self.batch_idx = batch_idx


class _TileRefProxy:
    """Proxy returned by tile[:] for the LHS of a load assignment."""

    _qd_is_deferred = True

    def __init__(self, tile: Any) -> None:
        self.tile = tile

    def _assign(self, value: Any) -> None:
        if isinstance(value, _TileSliceProxy):
            if value.batch_idx is not None:
                self.tile._load3d(
                    value.arr, value.batch_idx, value.row_start, value.row_stop, value.col_start, value.col_stop
                )
            else:
                self.tile._load(value.arr, value.row_start, value.row_stop, value.col_start, value.col_stop)
        else:
            raise TypeError(f"Tile16x16Pack2[:] can only be assigned from an array slice, got {type(value)}")


_tile16_cache: dict[Any, Any] = {}


def _make_tile16x16(dtype=None) -> "type[_Tile16x16Proto]":
    if dtype is None:
        dtype = qd.f32
    if dtype in _tile16_cache:
        return _tile16_cache[dtype]  # pyright: ignore[reportReturnType]
    cls = _make_tile16x16_class(dtype)
    _tile16_cache[dtype] = cls
    return cls  # pyright: ignore[reportReturnType]


def _make_tile16x16_class(dtype):
    class _Tile16x16:
        """Pack-2-aware register-resident 16x16 tile.

        All methods compute ``tile_base = (tid >> 4) << 4`` so that under ``block_dim=32`` lanes 0-15 form tile A
        and lanes 16-31 form tile B, and under ``block_dim=16`` (or fewer active lanes) ``tile_base`` is always 0
        and the implementation degenerates to the upstream single-tile-per-warp behavior.
        """

        r0: dtype
        r1: dtype
        r2: dtype
        r3: dtype
        r4: dtype
        r5: dtype
        r6: dtype
        r7: dtype
        r8: dtype
        r9: dtype
        r10: dtype
        r11: dtype
        r12: dtype
        r13: dtype
        r14: dtype
        r15: dtype

        @qd.func
        def _load(self, arr: qd.template(), row_start, row_stop, col_start, col_stop):
            """Load from a 2D array. Each lane loads arr[row_start + local, col_start:col_stop]."""
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            tile_base = (tid >> qd.i32(4)) << qd.i32(4)
            local = tid - tile_base
            arr_row_stop = arr.shape[0]
            if arr_row_stop < row_stop:
                row_stop = arr_row_stop
            row = row_start + local
            if row < row_stop:
                arr_col_stop = arr.shape[1]
                if arr_col_stop < col_stop:
                    col_stop = arr_col_stop
                for j in qd.static(range(_TILE)):
                    if col_start + j < col_stop:
                        self._set_col(j, arr[row, col_start + j])

        @qd.func
        def _load3d(self, arr: qd.template(), batch, row_start, row_stop, col_start, col_stop):
            """Load from a 3D array. Each lane loads arr[batch, row_start+local, col_start:col_stop].

            For pack-2: ``batch`` should be per-half-warp (e.g. ``i_b = i // 16`` with ``block_dim=32`` gives
            ``batch = warp*2`` for lanes 0-15 and ``warp*2 + 1`` for lanes 16-31). Each lane reads its env.
            """
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            tile_base = (tid >> qd.i32(4)) << qd.i32(4)
            local = tid - tile_base
            arr_row_stop = arr.shape[1]
            if arr_row_stop < row_stop:
                row_stop = arr_row_stop
            row = row_start + local
            if row < row_stop:
                arr_col_stop = arr.shape[2]
                if arr_col_stop < col_stop:
                    col_stop = arr_col_stop
                for j in qd.static(range(_TILE)):
                    if col_start + j < col_stop:
                        self._set_col(j, arr[batch, row, col_start + j])

        @qd.func
        def _store(self, arr: qd.template(), row_start, row_stop, col_start, col_stop):
            """Store to a 2D array. Each lane stores to arr[row_start + local, col_start:col_stop]."""
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            tile_base = (tid >> qd.i32(4)) << qd.i32(4)
            local = tid - tile_base
            arr_row_stop = arr.shape[0]
            if arr_row_stop < row_stop:
                row_stop = arr_row_stop
            row = row_start + local
            if row < row_stop:
                arr_col_stop = arr.shape[1]
                if arr_col_stop < col_stop:
                    col_stop = arr_col_stop
                for j in qd.static(range(_TILE)):
                    if col_start + j < col_stop:
                        arr[row, col_start + j] = self._get_col(j)

        @qd.func
        def _store3d(self, arr: qd.template(), batch, row_start, row_stop, col_start, col_stop):
            """Store to a 3D array. Each lane stores to arr[batch, row_start+local, col_start:col_stop]."""
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            tile_base = (tid >> qd.i32(4)) << qd.i32(4)
            local = tid - tile_base
            arr_row_stop = arr.shape[1]
            if arr_row_stop < row_stop:
                row_stop = arr_row_stop
            row = row_start + local
            if row < row_stop:
                arr_col_stop = arr.shape[2]
                if arr_col_stop < col_stop:
                    col_stop = arr_col_stop
                for j in qd.static(range(_TILE)):
                    if col_start + j < col_stop:
                        arr[batch, row, col_start + j] = self._get_col(j)

        @qd.func
        def eye_(self):
            """Set this tile to the 16x16 identity matrix (pack-2: each half-warp sets its own identity)."""
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            tile_base = (tid >> qd.i32(4)) << qd.i32(4)
            local = tid - tile_base
            for j in qd.static(range(_TILE)):
                self._set_col(j, qd.cast(1.0, dtype) if local == j else qd.cast(0.0, dtype))

        @qd.func
        def _get_col(self, k):
            val = qd.cast(0.0, dtype)
            if k == 0:
                val = self.r0
            if k == 1:
                val = self.r1
            if k == 2:
                val = self.r2
            if k == 3:
                val = self.r3
            if k == 4:
                val = self.r4
            if k == 5:
                val = self.r5
            if k == 6:
                val = self.r6
            if k == 7:
                val = self.r7
            if k == 8:
                val = self.r8
            if k == 9:
                val = self.r9
            if k == 10:
                val = self.r10
            if k == 11:
                val = self.r11
            if k == 12:
                val = self.r12
            if k == 13:
                val = self.r13
            if k == 14:
                val = self.r14
            if k == 15:
                val = self.r15
            return val

        @qd.func
        def _set_col(self, k, val):
            if k == 0:
                self.r0 = val
            if k == 1:
                self.r1 = val
            if k == 2:
                self.r2 = val
            if k == 3:
                self.r3 = val
            if k == 4:
                self.r4 = val
            if k == 5:
                self.r5 = val
            if k == 6:
                self.r6 = val
            if k == 7:
                self.r7 = val
            if k == 8:
                self.r8 = val
            if k == 9:
                self.r9 = val
            if k == 10:
                self.r10 = val
            if k == 11:
                self.r11 = val
            if k == 12:
                self.r12 = val
            if k == 13:
                self.r13 = val
            if k == 14:
                self.r14 = val
            if k == 15:
                self.r15 = val

        @qd.func
        def _ger_sub(self, a, b):
            """General rank-1 subtract in-place: self -= a @ b^T (per half-warp)."""
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            tile_base = (tid >> qd.i32(4)) << qd.i32(4)
            for j in qd.static(range(_TILE)):
                bc = qd.simt.subgroup.shuffle(b, qd.u32(tile_base + qd.i32(j)))
                self._set_col(j, self._get_col(j) - a * bc)

        @qd.func
        def cholesky_(self, eps):
            """In-place 16x16 Cholesky factorization (per half-warp)."""
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            tile_base = (tid >> qd.i32(4)) << qd.i32(4)
            local = tid - tile_base
            # Literal 16 (not _TILE) to satisfy genesis's pure-kernel policy, which forbids module-level
            # name lookups inside @qd.func bodies. Upstream Tile16x16 uses the same pattern; both must
            # match the static tile size.
            for k in range(16):
                diag_val = qd.cast(0.0, dtype)
                if local == k:
                    s = qd.cast(0.0, dtype)
                    for j in range(16):
                        if k > j:
                            c = self._get_col(j)
                            s += c * c
                    diag_val = qd.sqrt(qd.max(self._get_col(k) - s, eps))
                    self._set_col(k, diag_val)

                diag_k = qd.simt.subgroup.shuffle(diag_val, qd.u32(tile_base + qd.i32(k)))

                dot = qd.cast(0.0, dtype)
                for j in range(16):
                    if k > j:
                        my_col = self._get_col(j)
                        Lkj = qd.simt.subgroup.shuffle(my_col, qd.u32(tile_base + qd.i32(k)))
                        dot += Lkj * my_col  # type: ignore[reportOperatorIssue]

                if local > k:  # type: ignore[reportOperatorIssue]
                    new_val = (self._get_col(k) - dot) / diag_k  # type: ignore[reportOperatorIssue]
                    self._set_col(k, new_val)

        @qd.func
        def _trsm(self, L):
            """In-place triangular solve: self solves L @ X^T = B (per half-warp)."""
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            tile_base = (tid >> qd.i32(4)) << qd.i32(4)
            for c in range(16):  # literal 16 — see cholesky_ for rationale
                dot = qd.cast(0.0, dtype)
                for j in range(16):
                    if c > j:
                        Lkj = qd.simt.subgroup.shuffle(L._get_col(j), qd.u32(tile_base + qd.i32(c)))
                        dot += self._get_col(j) * Lkj  # type: ignore[reportOperatorIssue]

                diag_c = qd.simt.subgroup.shuffle(L._get_col(c), qd.u32(tile_base + qd.i32(c)))
                new_val = (self._get_col(c) - dot) / diag_c  # type: ignore[reportOperatorIssue]
                self._set_col(c, new_val)

        def solve_triangular_(self, B: Any, lower: bool = True) -> None:
            """Triangular solve: X @ self^T = B, storing result X in B in-place."""
            if not lower:
                raise TypeError("Tile16x16Pack2.solve_triangular_: only lower=True is supported")
            B._trsm(self)

        @qd.func
        def _resolve_vec2d(self, arr: qd.template(), row_start, row_stop, col):
            """Load one scalar per lane from a 2D array column (per half-warp local row)."""
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            tile_base = (tid >> qd.i32(4)) << qd.i32(4)
            local = tid - tile_base
            arr_row_stop = arr.shape[0]
            if arr_row_stop < row_stop:
                row_stop = arr_row_stop
            v = dtype(0.0)
            if row_start + local < row_stop:
                v = arr[row_start + local, col]
            return v

        @qd.func
        def _resolve_vec3d(self, arr: qd.template(), batch, row_start, row_stop, col):
            """Load one scalar per lane from a 3D array column (per half-warp local row + per-half-warp batch)."""
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            tile_base = (tid >> qd.i32(4)) << qd.i32(4)
            local = tid - tile_base
            arr_row_stop = arr.shape[1]
            if arr_row_stop < row_stop:
                row_stop = arr_row_stop
            v = dtype(0.0)
            if row_start + local < row_stop:
                v = arr[batch, row_start + local, col]
            return v

        def _resolve_vec_proxy(self, proxy: Any) -> Any:
            if proxy.batch_idx is not None:
                return self._resolve_vec3d(proxy.arr, proxy.batch_idx, proxy.row_start, proxy.row_stop, proxy.col)
            return self._resolve_vec2d(proxy.arr, proxy.row_start, proxy.row_stop, proxy.col)

        def _augassign(self, other: Any, op: str) -> None:
            # Accept both the local and upstream OuterProduct / VecSliceProxy classes so that this tile
            # works transparently with `qd.outer(...)` (which constructs the upstream OuterProduct).
            if isinstance(other, (_OuterProduct, _UpstreamOuterProduct)):
                if op == "Sub":
                    a_orig = other.a
                    b_orig = other.b
                    vec_proxy_types = (_VecSliceProxy, _UpstreamVecSliceProxy)
                    a = self._resolve_vec_proxy(a_orig) if isinstance(a_orig, vec_proxy_types) else a_orig
                    b = (
                        a
                        if (b_orig is a_orig)
                        else (self._resolve_vec_proxy(b_orig) if isinstance(b_orig, vec_proxy_types) else b_orig)
                    )
                    self._ger_sub(a, b)
                else:
                    raise TypeError(f"Tile16x16Pack2: unsupported augmented assignment op '{op}' with outer product")
            else:
                raise TypeError(f"Tile16x16Pack2: unsupported augmented assignment with {type(other)}")

    # StructType.__call__ already defaults missing args to 0
    result = qd.dataclass(_Tile16x16)
    result.SIZE = _TILE  # type: ignore[reportAttributeAccessIssue]
    result.zeros = result  # type: ignore[reportAttributeAccessIssue]

    @qd.func
    def _eye():
        t = result()
        t.eye_()  # type: ignore[reportAttributeAccessIssue]
        return t

    result.eye = _eye  # type: ignore[reportAttributeAccessIssue]

    # Register the pack-2 class into quadrants' upstream `_tile16_cache` so that the slice-dispatch
    # in `quadrants/lang/simt/tile_slicing.py` recognizes `tile[:]` as a tile-ref and
    # `arr[batch, r:r2, c:c2]` as a tile-slice. The cache is iterated as `.values()` for isinstance
    # checks, so we just need any unique key that doesn't collide with upstream's dtype-keyed
    # entries. Using `("pack2", dtype)` keeps both this and the upstream `Tile16x16[dtype]` alive
    # in the same kernel (the upstream tile uses `dtype` as key).
    _upstream_tile16_cache[("pack2", dtype)] = result
    return result


class _Tile16x16Proxy:
    """Proxy for dtype-at-point-of-use tile creation.

    Use as ``Tile16x16Pack2.zeros(dtype=gs.qd_float)`` inside a kernel.
    """

    SIZE = _TILE

    @staticmethod
    def _resolve(dtype):
        from quadrants.lang import impl  # pylint: disable=import-outside-toplevel

        if dtype is None:
            dtype = impl.get_runtime().default_fp
        if dtype in _tile16_cache:
            return _tile16_cache[dtype]
        return _make_tile16x16(dtype)

    def zeros(self, *, dtype=None):
        return self._resolve(dtype)()

    def eye(self, *, dtype=None):
        return self._resolve(dtype).eye()


Tile16x16Pack2 = _Tile16x16Proxy()


# Eagerly register pack-2 tile classes for the standard float dtypes into the upstream tile cache so
# that quadrants' slice-dispatch (try_tile_slice in quadrants/lang/simt/tile_slicing.py) recognizes
# `tile[:]` and `arr[batch, r:r2, c:c2]` as tile operations even on the *very first* statement of the
# first tiled kernel to compile. Without this, the dispatch can briefly see an empty upstream cache
# (the cache is only populated on the first `Tile16x16Pack2.zeros/.eye(dtype=...)` call, which races
# the subscript build inside the same statement / kernel).
for _dt in (qd.f32, qd.f64):
    try:
        _make_tile16x16(_dt)
    except Exception:  # pylint: disable=broad-except
        # Some quadrants builds may restrict dataclass construction outside an active runtime; skip
        # silently and rely on the lazy path. Any failure here just reverts to the lazy registration.
        pass
