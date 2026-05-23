# pyright: reportInvalidTypeForm=false

"""
Register-resident 32x32 tile with an optimised Cholesky factorization.

Vector-storage variant of the 32x32 register-tile primitive. Holds the per-thread tile row as a single
``qd.types.vector(32, dtype)`` field ``r`` rather than 32 named scalar fields (``r0..r31``).  The two storage layouts
are functionally interchangeable -- with python-int (qd.static) indices, ``self.r[k]`` lowers to the same
direct-register access as ``self.r0`` / ``self.r1`` / ... etc; with runtime indices, both lower to a 32-way switch
table (the named-field variant materializes the table by hand via a 32-way ``if k == N`` if-cascade; the
vector-storage variant lets the AST transformer materialize the switch).

Compile time wins from the vector storage are substantial: every if-cascade in the named-field variant is 64 lines
(32 if-stmts + 32 assigns) of AST that the transformer walks even though all but one branch is dead. With vector
storage the same cascade collapses to a single ``self.r[<k>] = val`` line. With ~7 cascade sites per tile method and
``cholesky_``'s nested ``qd.static`` loops emitting two cascades per outer-k iteration, the named-field variant
produced O(2k) AST nodes per tile method post-unroll; the vector-storage variant produces O(few hundred).

Algorithm is bit-identical to the named-field version. Used by
`genesis.engine.solvers.rigid.constraint.solver` for blocked Cholesky on the constraint Hessian. Exports
``Tile32x32Cholesky``, a proxy with the same surface as ``quadrants.simt.Tile16x16`` (``.eye(dtype=...)``,
``.zeros(dtype=...)``). Callers use explicit ``_load3d`` / ``_store3d`` / ``_resolve_vec3d`` / ``_ger_sub`` methods
rather than slice syntax.

FIXME: move the changes in this file back into Quadrants.
"""

from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any, NoReturn

import quadrants as qd

if _TYPE_CHECKING:

    class _Tile32x32Proto:  # noqa: E303
        """Static type stub so pyright sees Tile32x32 methods correctly."""

        SIZE: int

        def __init__(self, *args: Any, **kwargs: Any) -> None: ...  # noqa: E704
        @classmethod
        def zeros(cls) -> "_Tile32x32Proto": ...  # noqa: E704
        @classmethod
        def eye(cls) -> "_Tile32x32Proto": ...  # noqa: E704
        def eye_(self) -> None: ...  # noqa: E704
        def cholesky_(self, eps: Any) -> None: ...  # noqa: E704
        def solve_triangular_(self, B: "_Tile32x32Proto", lower: bool = True) -> None: ...  # noqa: E704
        def _load(self, arr: Any, row_start: Any, row_end: Any, col_start: Any, col_end: Any) -> None: ...  # noqa: E704
        def _store(self, arr: Any, row_start: Any, row_end: Any, col_start: Any, col_end: Any) -> None: ...  # noqa: E704
        def _load3d(self, arr: Any, batch: Any, row_start: Any, row_end: Any, col_start: Any, col_end: Any) -> None: ...  # noqa: E704
        def _store3d(
            self, arr: Any, batch: Any, row_start: Any, row_end: Any, col_start: Any, col_end: Any
        ) -> None: ...  # noqa: E704
        def _get_col(self, k: Any) -> Any: ...  # noqa: E704
        def _set_col(self, k: Any, val: Any) -> None: ...  # noqa: E704
        def _ger_sub(self, a: Any, b: Any) -> None: ...  # noqa: E704
        def _trsm(self, L: "_Tile32x32Proto") -> None: ...  # noqa: E704
        def __isub__(self, other: Any) -> "_Tile32x32Proto": ...  # noqa: E704
        def __getitem__(self, key: Any) -> Any: ...  # noqa: E704
        def __setitem__(self, key: Any, value: Any) -> None: ...  # noqa: E704


_TILE = 32


class _OuterProduct:
    """Deferred outer product proxy for use with augmented assignment on Tile32x32.

    Created by qd.outer(a, b). Not a quadrants expression -- only valid as the RHS of ``tile -= qd.outer(a, b)``.
    """

    _qd_is_deferred = True

    def __init__(self, a: Any, b: Any) -> None:
        self.a = a
        self.b = b

    def __add__(self, other: Any) -> NoReturn:
        raise TypeError("OuterProduct does not support composition; apply each update separately")

    def __radd__(self, other: Any) -> NoReturn:
        raise TypeError("OuterProduct does not support composition; apply each update separately")


def outer(a: Any, b: Any) -> _OuterProduct:
    """Create a deferred outer product for use with Tile32x32 augmented assignment.

    Usage::

        t -= qd.outer(a, b)   # equivalent to t._ger_sub(a, b)
        t -= qd.outer(v, v)   # symmetric case (a == b)
    """
    return _OuterProduct(a, b)


class _DeferredProxyMixin:
    """Raises clear errors if a deferred tile proxy is accidentally used as a value."""

    _proxy_description = "Tile proxy"

    def _misuse(self, op: str = "used") -> NoReturn:
        raise TypeError(
            f"{self._proxy_description} was {op}, but it is only valid in tile operations (tile[:] = ..., ... = tile, qd.outer(...))"
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
        return f"<{self._proxy_description} — not a value; use with tile[:] = ... or qd.outer(...)>"


class _TileSliceProxy(_DeferredProxyMixin):
    """Deferred 2D/3D array slice for tile load/store.

    Created by subscripting a Field or ndarray with 2D slices, e.g. ``arr[row_start:row_stop, col_start:col_stop]``.
    Not a quadrants expression -- only valid as the RHS of a tile assignment (load) or as the LHS target (store).
    """

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
        """Store path: arr[r:r+n_rows, c:c+n_cols] = tile."""
        if self.batch_idx is not None:
            tile._store3d(self.arr, self.batch_idx, self.row_start, self.row_stop, self.col_start, self.col_stop)
        else:
            tile._store(self.arr, self.row_start, self.row_stop, self.col_start, self.col_stop)


class _VecSliceProxy(_DeferredProxyMixin):
    """Deferred column-vector load from a 2D/3D array.

    Created by ``arr[row_start:row_stop, col]`` or ``arr[batch_idx, row_start:row_stop, col]``.
    Each subgroup thread loads one element; out-of-range threads get 0.
    Only valid as an argument to ``qd.outer()`` in tile augmented assignment.
    """

    _qd_is_deferred = True
    _proxy_description = "Vec slice proxy (arr[r0:r1, col])"

    def __init__(self, arr: Any, row_start: Any, row_stop: Any, col: Any, batch_idx: Any = None) -> None:
        self.arr = arr
        self.row_start = row_start
        self.row_stop = row_stop
        self.col = col
        self.batch_idx = batch_idx


class _TileRefProxy:
    """Proxy returned by tile[:] for the LHS of a load assignment.

    Enables ``tile[:] = arr[r:r+32, c:n]``.  The ``[:]`` is required to distinguish in-place tile loads from
    variable rebinding.
    """

    _qd_is_deferred = True

    def __init__(self, tile: Any) -> None:
        self.tile = tile

    def _assign(self, value: Any) -> None:
        """Load path: tile[:] = arr[r:r+n, c:c+n]. Dispatches to _load or _load3d."""
        if isinstance(value, _TileSliceProxy):
            if value.batch_idx is not None:
                self.tile._load3d(
                    value.arr, value.batch_idx, value.row_start, value.row_stop, value.col_start, value.col_stop
                )
            else:
                self.tile._load(value.arr, value.row_start, value.row_stop, value.col_start, value.col_stop)
        else:
            raise TypeError(f"Tile32x32[:] can only be assigned from an array slice, got {type(value)}")


# Per-dtype class cache. Independent of quadrants' own Tile16x16 cache so this
# module never mutates upstream state.
_tile32_cache: dict[Any, type] = {}


def _make_tile32x32(dtype=None) -> "type[_Tile32x32Proto]":
    """Build (and memoize) a Tile32x32 dataclass with the optimised cholesky_."""
    if dtype is None:
        dtype = qd.f32
    cached = _tile32_cache.get(dtype)
    if cached is not None:
        return cached  # pyright: ignore[reportReturnType]
    cls = _make_tile32x32_class(dtype)
    _tile32_cache[dtype] = cls
    return cls  # pyright: ignore[reportReturnType]


def _make_tile32x32_class(dtype):
    # 32 elements broken into 4 sub-vectors of 8.  vec32 fails to register-promote on cuda 7.x
    # (places the per-thread tile row in local memory, costing -19% FPS on dex_hand), but vec8
    # reliably register-promotes via SROA (matches the 12x12 / 144-element matrices that
    # quadrants's per-thread linalg ops rely on).  All hot indexing in this module is static
    # (via qd.static unrolls in cholesky_, _ger_sub, _load3d, _store3d, eye_), so the 4-way
    # bank selection (j // 8) folds at trace time to direct sub-vector + intra-vector indexing.
    vec8_dtype = qd.types.vector(8, dtype)

    # Helper: emit a python-int 4-way cascade for static k accesses.  Folded at trace time.
    def _static_read(self_, k_py_int):
        bank = k_py_int // 8
        off = k_py_int % 8
        if bank == 0:
            return self_.b0[off]
        if bank == 1:
            return self_.b1[off]
        if bank == 2:
            return self_.b2[off]
        return self_.b3[off]

    _static_read.__module__ = "quadrants.gen.tile32_cholesky"

    class _Tile32x32Cholesky:
        """A 32x32 tile distributed one row per subgroup thread, held in 4 vec8 sub-banks ``b0..b3``."""

        b0: vec8_dtype
        b1: vec8_dtype
        b2: vec8_dtype
        b3: vec8_dtype

        @qd.func
        def _load(self, arr: qd.template(), row_start, row_stop, col_start, col_stop):
            """Load from a 2D array within [row_start, row_stop) x [col_start, col_stop)."""
            arr_row_stop = arr.shape[0]
            if arr_row_stop < row_stop:
                row_stop = arr_row_stop
            row = row_start + qd.simt.subgroup.invocation_id()
            if row < row_stop:
                arr_col_stop = arr.shape[1]
                if arr_col_stop < col_stop:
                    col_stop = arr_col_stop
                for j in qd.static(range(8)):
                    if col_start + j < col_stop:
                        self.b0[j] = arr[row, col_start + j]
                for j in qd.static(range(8)):
                    if col_start + 8 + j < col_stop:
                        self.b1[j] = arr[row, col_start + 8 + j]
                for j in qd.static(range(8)):
                    if col_start + 16 + j < col_stop:
                        self.b2[j] = arr[row, col_start + 16 + j]
                for j in qd.static(range(8)):
                    if col_start + 24 + j < col_stop:
                        self.b3[j] = arr[row, col_start + 24 + j]

        @qd.func
        def _load3d(self, arr: qd.template(), batch, row_start, row_stop, col_start, col_stop):
            """Load from a 3D array within [row_start, row_stop) x [col_start, col_stop)."""
            arr_row_stop = arr.shape[1]
            if arr_row_stop < row_stop:
                row_stop = arr_row_stop
            row = row_start + qd.simt.subgroup.invocation_id()
            if row < row_stop:
                arr_col_stop = arr.shape[2]
                if arr_col_stop < col_stop:
                    col_stop = arr_col_stop
                for j in qd.static(range(8)):
                    if col_start + j < col_stop:
                        self.b0[j] = arr[batch, row, col_start + j]
                for j in qd.static(range(8)):
                    if col_start + 8 + j < col_stop:
                        self.b1[j] = arr[batch, row, col_start + 8 + j]
                for j in qd.static(range(8)):
                    if col_start + 16 + j < col_stop:
                        self.b2[j] = arr[batch, row, col_start + 16 + j]
                for j in qd.static(range(8)):
                    if col_start + 24 + j < col_stop:
                        self.b3[j] = arr[batch, row, col_start + 24 + j]

        @qd.func
        def _store(self, arr: qd.template(), row_start, row_stop, col_start, col_stop):
            """Store to a 2D array within [row_start, row_stop) x [col_start, col_stop)."""
            arr_row_stop = arr.shape[0]
            if arr_row_stop < row_stop:
                row_stop = arr_row_stop
            row = row_start + qd.simt.subgroup.invocation_id()
            if row < row_stop:
                arr_col_stop = arr.shape[1]
                if arr_col_stop < col_stop:
                    col_stop = arr_col_stop
                for j in qd.static(range(8)):
                    if col_start + j < col_stop:
                        arr[row, col_start + j] = self.b0[j]
                for j in qd.static(range(8)):
                    if col_start + 8 + j < col_stop:
                        arr[row, col_start + 8 + j] = self.b1[j]
                for j in qd.static(range(8)):
                    if col_start + 16 + j < col_stop:
                        arr[row, col_start + 16 + j] = self.b2[j]
                for j in qd.static(range(8)):
                    if col_start + 24 + j < col_stop:
                        arr[row, col_start + 24 + j] = self.b3[j]

        @qd.func
        def _store3d(self, arr: qd.template(), batch, row_start, row_stop, col_start, col_stop):
            """Store to a 3D array within [row_start, row_stop) x [col_start, col_stop)."""
            arr_row_stop = arr.shape[1]
            if arr_row_stop < row_stop:
                row_stop = arr_row_stop
            row = row_start + qd.simt.subgroup.invocation_id()
            if row < row_stop:
                arr_col_stop = arr.shape[2]
                if arr_col_stop < col_stop:
                    col_stop = arr_col_stop
                for j in qd.static(range(8)):
                    if col_start + j < col_stop:
                        arr[batch, row, col_start + j] = self.b0[j]
                for j in qd.static(range(8)):
                    if col_start + 8 + j < col_stop:
                        arr[batch, row, col_start + 8 + j] = self.b1[j]
                for j in qd.static(range(8)):
                    if col_start + 16 + j < col_stop:
                        arr[batch, row, col_start + 16 + j] = self.b2[j]
                for j in qd.static(range(8)):
                    if col_start + 24 + j < col_stop:
                        arr[batch, row, col_start + 24 + j] = self.b3[j]

        @qd.func
        def eye_(self):
            """Set this tile to the 32x32 identity matrix.  Each thread sets its diagonal element to 1.0 and all
            others to 0.0."""
            tid = qd.simt.subgroup.invocation_id()
            for j in qd.static(range(8)):
                self.b0[j] = qd.cast(1.0, dtype) if tid == j else qd.cast(0.0, dtype)
                self.b1[j] = qd.cast(1.0, dtype) if tid == (8 + j) else qd.cast(0.0, dtype)
                self.b2[j] = qd.cast(1.0, dtype) if tid == (16 + j) else qd.cast(0.0, dtype)
                self.b3[j] = qd.cast(1.0, dtype) if tid == (24 + j) else qd.cast(0.0, dtype)

        @qd.func
        def _get_col(self, k):
            """Return the value of register (column) k.  4-way cascade selecting the sub-bank, with intra-bank
            indexing.  With static k the cascade folds to a direct sub-bank scalar access; with runtime k
            quadrants emits a 4-way switch over the sub-banks (each of which is small enough to register-promote)."""
            val = qd.cast(0.0, dtype)
            if k < 8:
                val = self.b0[k]
            elif k < 16:
                val = self.b1[k - 8]
            elif k < 24:
                val = self.b2[k - 16]
            else:
                val = self.b3[k - 24]
            return val

        @qd.func
        def _set_col(self, k, val):
            """Set register (column) k to val.  Same 4-way bank-selection lowering as _get_col."""
            if k < 8:
                self.b0[k] = val
            elif k < 16:
                self.b1[k - 8] = val
            elif k < 24:
                self.b2[k - 16] = val
            else:
                self.b3[k - 24] = val

        # Direct field read by python-int index, bypassing the 4-way runtime cascade.  Used at qd.static-unrolled
        # call sites inside this module (cholesky_, _ger_sub) for compile-time register access.
        def _r(self, k):
            return _static_read(self, k)

        _r.__module__ = "quadrants.gen.tile32_cholesky"

        @qd.func
        def _ger_sub(self, a, b):
            """General rank-1 subtract in-place: self -= a @ b^T."""
            for j in qd.static(range(8)):
                bc = qd.simt.subgroup.shuffle(b, qd.u32(j))
                self.b0[j] = self.b0[j] - a * bc
            for j in qd.static(range(8)):
                bc = qd.simt.subgroup.shuffle(b, qd.u32(8 + j))
                self.b1[j] = self.b1[j] - a * bc
            for j in qd.static(range(8)):
                bc = qd.simt.subgroup.shuffle(b, qd.u32(16 + j))
                self.b2[j] = self.b2[j] - a * bc
            for j in qd.static(range(8)):
                bc = qd.simt.subgroup.shuffle(b, qd.u32(24 + j))
                self.b3[j] = self.b3[j] - a * bc

        @qd.func
        def cholesky_(self, eps):
            """In-place 32x32 Cholesky factorization via subgroup shuffles.

            On return, the lower triangle holds L such that A = L @ L^T.  Diagonal clamped to
            sqrt(max(value, eps)) for numerical stability.

            All register access in this body uses python-int indices via the qd.static-unrolled outer/inner loops,
            so ``self._r(k)`` (a thin getattr+vec8-subscript wrapper, see ``_static_read`` above) and the write-side
            ``if k < 8: self.b0[k - 0] = val ...`` cascade are folded at trace time to direct sub-vector accesses.
            The 4 sub-vectors (``b0..b3``) are small enough to reliably register-promote (vs. the vec32 single-field
            layout which fell back to local memory and cost -19% FPS).
            """
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            my_norm_sq = qd.cast(0.0, dtype)
            for k in qd.static(range(32)):
                # Python ints — used for static sub-bank dispatch at compile time.
                kb = k // 8
                ko = k % 8
                diag_val = qd.cast(0.0, dtype)
                if tid == k:
                    diag_val = qd.sqrt(qd.max(self._r(k) - my_norm_sq, eps))
                    if kb == 0:
                        self.b0[ko] = diag_val
                    elif kb == 1:
                        self.b1[ko] = diag_val
                    elif kb == 2:
                        self.b2[ko] = diag_val
                    else:
                        self.b3[ko] = diag_val

                diag_k = qd.simt.subgroup.shuffle(diag_val, qd.u32(k))

                dot0 = qd.cast(0.0, dtype)
                dot1 = qd.cast(0.0, dtype)
                for j in qd.static(range(32)):
                    if k > j:
                        my_col = self._r(j)
                        Lkj = qd.simt.subgroup.shuffle(my_col, qd.u32(k))
                        if j % 2 == 0:
                            dot0 += Lkj * my_col  # type: ignore[reportOperatorIssue]
                        else:
                            dot1 += Lkj * my_col  # type: ignore[reportOperatorIssue]
                dot = dot0 + dot1

                new_val = qd.cast(0.0, dtype)
                if tid > k:  # type: ignore[reportOperatorIssue]
                    new_val = (self._r(k) - dot) / diag_k  # type: ignore[reportOperatorIssue]
                    if kb == 0:
                        self.b0[ko] = new_val
                    elif kb == 1:
                        self.b1[ko] = new_val
                    elif kb == 2:
                        self.b2[ko] = new_val
                    else:
                        self.b3[ko] = new_val
                if tid > k:  # type: ignore[reportOperatorIssue]
                    my_norm_sq += new_val * new_val

        @qd.func
        def _trsm(self, L):
            """In-place triangular solve: solve self @ L^T = B (original self).

            L is a Tile32x32 holding the lower-triangular Cholesky factor (from cholesky_).  On return, self holds
            the solution X.
            """
            for c in range(32):
                dot = qd.cast(0.0, dtype)
                for j in range(32):
                    if c > j:
                        Lkj = qd.simt.subgroup.shuffle(L._get_col(j), qd.u32(c))
                        dot += self._get_col(j) * Lkj  # type: ignore[reportOperatorIssue]

                diag_c = qd.simt.subgroup.shuffle(L._get_col(c), qd.u32(c))
                new_val = (self._get_col(c) - dot) / diag_c  # type: ignore[reportOperatorIssue]
                self._set_col(c, new_val)

        def solve_triangular_(self, B: Any, lower: bool = True) -> None:
            """Triangular solve: X @ self^T = B, storing result X in B in-place.

            self must be lower-triangular and non-singular (all diagonal elements non-zero).  Passing a singular
            matrix causes division by zero, producing inf/NaN without warning.  Only lower=True is supported.
            """
            if not lower:
                raise TypeError("Tile32x32Cholesky.solve_triangular_: only lower=True is supported")
            B._trsm(self)

        solve_triangular_.__module__ = "quadrants.gen.tile32_cholesky"

        @qd.func
        def _resolve_vec2d(self, arr: qd.template(), row_start, row_stop, col):
            """Load one scalar per thread from a 2D array column, clamped to array bounds."""
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            arr_row_stop = arr.shape[0]
            if arr_row_stop < row_stop:
                row_stop = arr_row_stop
            v = dtype(0.0)
            if row_start + tid < row_stop:
                v = arr[row_start + tid, col]
            return v

        @qd.func
        def _resolve_vec3d(self, arr: qd.template(), batch, row_start, row_stop, col):
            """Load one scalar per thread from a 3D array column, clamped to array bounds."""
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            arr_row_stop = arr.shape[1]
            if arr_row_stop < row_stop:
                row_stop = arr_row_stop
            v = dtype(0.0)
            if row_start + tid < row_stop:
                v = arr[batch, row_start + tid, col]
            return v

        def _resolve_vec_proxy(self, proxy: _VecSliceProxy) -> Any:
            """Materialize a _VecSliceProxy into a scalar by dispatching to _resolve_vec2d or _resolve_vec3d."""
            if proxy.batch_idx is not None:
                return self._resolve_vec3d(proxy.arr, proxy.batch_idx, proxy.row_start, proxy.row_stop, proxy.col)
            return self._resolve_vec2d(proxy.arr, proxy.row_start, proxy.row_stop, proxy.col)

        def _augassign(self, other: Any, op: str) -> None:
            """Handle augmented assignment (e.g. tile -= qd.outer(a, b)).

            Resolves _VecSliceProxy arguments and dispatches to _ger_sub.  Only 'Sub' is supported.
            """
            if isinstance(other, _OuterProduct):
                if op == "Sub":
                    a_orig = other.a
                    b_orig = other.b
                    a = self._resolve_vec_proxy(a_orig) if isinstance(a_orig, _VecSliceProxy) else a_orig
                    b = (
                        a
                        if (b_orig is a_orig)
                        else (self._resolve_vec_proxy(b_orig) if isinstance(b_orig, _VecSliceProxy) else b_orig)
                    )
                    self._ger_sub(a, b)
                else:
                    raise TypeError(f"Tile32x32Cholesky: unsupported augmented assignment op '{op}' with outer product")
            else:
                raise TypeError(f"Tile32x32Cholesky: unsupported augmented assignment with {type(other)}")

    # StructType.__call__ already defaults missing args to 0, so Tile() produces a zero-initialized tile
    # without needing default values in the class definition (which @qd.dataclass doesn't support).
    result = qd.dataclass(_Tile32x32Cholesky)
    result.SIZE = _TILE  # type: ignore[reportAttributeAccessIssue]
    result.zeros = result  # type: ignore[reportAttributeAccessIssue]

    @qd.func
    def _eye():
        t = result()
        t.eye_()  # type: ignore[reportAttributeAccessIssue]
        return t

    result.eye = _eye  # type: ignore[reportAttributeAccessIssue]
    return result


class _Tile32x32CholeskyProxy:
    """Proxy for dtype-at-point-of-use tile creation.

    Use as ``Tile32x32Cholesky.zeros(dtype=qd.f32)`` inside a kernel. The dtype is resolved at kernel compilation
    time, defaulting to the compile config's ``default_fp`` if omitted.
    """

    SIZE = _TILE

    @staticmethod
    def _resolve(dtype):
        from quadrants.lang import impl  # pylint: disable=import-outside-toplevel
        from quadrants.lang.exception import (  # pylint: disable=import-outside-toplevel
            QuadrantsSyntaxError,
        )

        arch = impl.current_cfg().arch
        if arch in (qd.cpu, qd.x64, getattr(qd, "arm64", None)):
            raise QuadrantsSyntaxError(
                f"Tile32x32Cholesky requires a GPU backend (cuda, metal, vulkan, amdgpu). Current arch is {arch}."
            )
        if dtype is None:
            dtype = impl.get_runtime().default_fp
        return _make_tile32x32(dtype)

    def zeros(self, *, dtype=None):
        """Zero-initialized tile."""
        return self._resolve(dtype)()

    def eye(self, *, dtype=None):
        """Identity tile (diagonal = 1, rest = 0)."""
        return self._resolve(dtype).eye()


# Re-declare the proxy constructors as belonging to a quadrants.* module so
# the AST transformer's external-function check (which exempts callees whose
# `__module__` starts with `"quadrants."`) does not warn that they are not
# @qd.func when invoked from inside a kernel.
_Tile32x32CholeskyProxy.zeros.__module__ = "quadrants.gen.tile32_cholesky"
_Tile32x32CholeskyProxy.eye.__module__ = "quadrants.gen.tile32_cholesky"


Tile32x32Cholesky = _Tile32x32CholeskyProxy()
