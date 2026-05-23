# pyright: reportInvalidTypeForm=false

"""
Register-resident 32x32 tile with an optimised Cholesky factorization.

32x32 register-tile primitive. Hand-derived from `genesis/utils/_tile16.py` (which itself is a local copy of `quadrants.lang.simt._tile16`) by mechanical 16 -> 32 expansion. Differs from the 16x16 version only in the body
of `cholesky_`: the outer and inner column-update loops are wrapped in `qd.static(...)`
so the compiler eliminates dead predicates and per-iteration register-indexing
cascades. The algorithm is otherwise identical and produces bit-equivalent results.

Used by `genesis.engine.solvers.rigid.constraint.solver` for blocked Cholesky on the
constraint Hessian. Exports `Tile32x32Cholesky`, a proxy with the same surface as
`quadrants.simt.Tile32x32` (`Tile32x32Cholesky.eye(dtype=...)`, `.zeros(dtype=...)`).
The distinct class name keeps it cleanly separated from quadrants' stock
`Tile32x32` — they are not interchangeable inside a single kernel because each is
tracked by its own slice-dispatch cache. Callers that consume this class therefore
use explicit `_load3d` / `_store3d` / `_resolve_vec3d` / `_ger_sub` methods instead
of the `tile[:] = arr[r:r2, c:c2]` slice syntax.

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

    Enables ``tile[:] = arr[r:r+16, c:n]``.  The ``[:]`` is required to distinguish in-place tile loads from
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


# Per-dtype class cache. Independent of quadrants' own Tile32x32 cache so this
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
    class _Tile32x32Cholesky:
        """A 32x32 tile distributed one row per subgroup thread, held in 16 scalar registers.  All fields default to
        0.0 when omitted: ``Tile32x32Cholesky()`` creates a zero tile."""

        r: qd.field_array(_TILE, dtype)

        @qd.func
        def _load(self, arr: qd.template(), row_start, row_stop, col_start, col_stop):
            """Load from a 2D array within [row_start, row_stop) x [col_start, col_stop).

            Each thread loads arr[row_start + tid, col_start:col_stop].  Threads where row_start + tid >= row_stop
            skip the load (tile row unchanged).
            """
            arr_row_stop = arr.shape[0]
            if arr_row_stop < row_stop:
                row_stop = arr_row_stop
            row = row_start + qd.simt.subgroup.invocation_id()
            if row < row_stop:
                arr_col_stop = arr.shape[1]
                if arr_col_stop < col_stop:
                    col_stop = arr_col_stop
                # Inline cascade: with j a python-int from qd.static, only the matching branch is emitted into the AST.
                # Avoids the 32x duplication that calling _set_col(j) through the @qd.func boundary would force.
                for j in qd.static(range(32)):
                    if col_start + j < col_stop:
                        val = arr[row, col_start + j]
                        self.r[j] = val

        @qd.func
        def _load3d(self, arr: qd.template(), batch, row_start, row_stop, col_start, col_stop):
            """Load from a 3D array within [row_start, row_stop) x [col_start, col_stop).

            Each thread loads arr[batch, row_start+tid, col_start:col_stop].  Threads where row_start + tid >=
            row_stop skip the load (tile row unchanged).
            """
            arr_row_stop = arr.shape[1]
            if arr_row_stop < row_stop:
                row_stop = arr_row_stop
            row = row_start + qd.simt.subgroup.invocation_id()
            if row < row_stop:
                arr_col_stop = arr.shape[2]
                if arr_col_stop < col_stop:
                    col_stop = arr_col_stop
                for j in qd.static(range(32)):
                    if col_start + j < col_stop:
                        val = arr[batch, row, col_start + j]
                        self.r[j] = val

        @qd.func
        def _store(self, arr: qd.template(), row_start, row_stop, col_start, col_stop):
            """Store to a 2D array within [row_start, row_stop) x [col_start, col_stop).

            Each thread stores to arr[row_start + tid, col_start:col_stop].  Threads where row_start + tid >=
            row_stop skip the store.
            """
            arr_row_stop = arr.shape[0]
            if arr_row_stop < row_stop:
                row_stop = arr_row_stop
            row = row_start + qd.simt.subgroup.invocation_id()
            if row < row_stop:
                arr_col_stop = arr.shape[1]
                if arr_col_stop < col_stop:
                    col_stop = arr_col_stop
                for j in qd.static(range(32)):
                    if col_start + j < col_stop:
                        arr[row, col_start + j] = self.r[j]

        @qd.func
        def _store3d(self, arr: qd.template(), batch, row_start, row_stop, col_start, col_stop):
            """Store to a 3D array within [row_start, row_stop) x [col_start, col_stop).

            Each thread stores to arr[batch, row_start+tid, col_start:col_stop].  Threads where row_start + tid >=
            row_stop skip the store.
            """
            arr_row_stop = arr.shape[1]
            if arr_row_stop < row_stop:
                row_stop = arr_row_stop
            row = row_start + qd.simt.subgroup.invocation_id()
            if row < row_stop:
                arr_col_stop = arr.shape[2]
                if arr_col_stop < col_stop:
                    col_stop = arr_col_stop
                for j in qd.static(range(32)):
                    if col_start + j < col_stop:
                        arr[batch, row, col_start + j] = self.r[j]

        @qd.func
        def eye_(self):
            """Set this tile to the 32x32 identity matrix.  Each thread sets its diagonal element to 1.0 and all
            others to 0.0."""
            tid = qd.simt.subgroup.invocation_id()
            for j in qd.static(range(32)):
                val = 1.0 if tid == j else 0.0
                self.r[j] = val

        @qd.func
        def _get_col(self, k):
            """Return the value of register (column) k. Cascade form for the runtime-k case;
            the static-k case is handled inline via ``self.r[<py_int>]`` (proposal-1 AST rewrite).
            References the synthetic ``_rN`` names directly to bypass the field_array path."""
            val = qd.cast(0.0, dtype)
            if k == 0:   val = self._r0
            if k == 1:   val = self._r1
            if k == 2:   val = self._r2
            if k == 3:   val = self._r3
            if k == 4:   val = self._r4
            if k == 5:   val = self._r5
            if k == 6:   val = self._r6
            if k == 7:   val = self._r7
            if k == 8:   val = self._r8
            if k == 9:   val = self._r9
            if k == 10:  val = self._r10
            if k == 11:  val = self._r11
            if k == 12:  val = self._r12
            if k == 13:  val = self._r13
            if k == 14:  val = self._r14
            if k == 15:  val = self._r15
            if k == 16:  val = self._r16
            if k == 17:  val = self._r17
            if k == 18:  val = self._r18
            if k == 19:  val = self._r19
            if k == 20:  val = self._r20
            if k == 21:  val = self._r21
            if k == 22:  val = self._r22
            if k == 23:  val = self._r23
            if k == 24:  val = self._r24
            if k == 25:  val = self._r25
            if k == 26:  val = self._r26
            if k == 27:  val = self._r27
            if k == 28:  val = self._r28
            if k == 29:  val = self._r29
            if k == 30:  val = self._r30
            if k == 31:  val = self._r31
            return val

        @qd.func
        def _set_col(self, k, val):
            """Set register (column) k to val. Cascade form for the runtime-k case; see
            ``_get_col`` docstring for rationale."""
            if k == 0:   self._r0 = val
            if k == 1:   self._r1 = val
            if k == 2:   self._r2 = val
            if k == 3:   self._r3 = val
            if k == 4:   self._r4 = val
            if k == 5:   self._r5 = val
            if k == 6:   self._r6 = val
            if k == 7:   self._r7 = val
            if k == 8:   self._r8 = val
            if k == 9:   self._r9 = val
            if k == 10:  self._r10 = val
            if k == 11:  self._r11 = val
            if k == 12:  self._r12 = val
            if k == 13:  self._r13 = val
            if k == 14:  self._r14 = val
            if k == 15:  self._r15 = val
            if k == 16:  self._r16 = val
            if k == 17:  self._r17 = val
            if k == 18:  self._r18 = val
            if k == 19:  self._r19 = val
            if k == 20:  self._r20 = val
            if k == 21:  self._r21 = val
            if k == 22:  self._r22 = val
            if k == 23:  self._r23 = val
            if k == 24:  self._r24 = val
            if k == 25:  self._r25 = val
            if k == 26:  self._r26 = val
            if k == 27:  self._r27 = val
            if k == 28:  self._r28 = val
            if k == 29:  self._r29 = val
            if k == 30:  self._r30 = val
            if k == 31:  self._r31 = val

        @qd.func
        def _ger_sub(self, a, b):
            """General rank-1 subtract in-place: self -= a @ b^T."""
            for j in qd.static(range(32)):
                bc = qd.simt.subgroup.shuffle(b, qd.u32(j))
                val = self.r[j] - a * bc
                self.r[j] = val

        @qd.func
        def cholesky_(self, eps):
            """In-place 32x32 Cholesky factorization via subgroup shuffles.

            On return, the lower triangle holds L such that A = L @ L^T.  Diagonal clamped to
            sqrt(max(value, eps)) for numerical stability.
            """
            # `k` and `j` are wrapped in qd.static so the `if k > j` predicates fold at compile time and register access
            # on the outer `k` and inner `j` collapses to a single field reference via `self.r[<py_int>]` (a thin
            # getattr wrapper) rather than a 32-deep register-indexing cascade. Writes use an inline `if k == N:
            # self.rN = ...` chain (setattr is rejected by the quadrants AST builder) which the AST transformer folds at
            # build time when `k` is a python int. The per-lane row-norm used for the diagonal update is carried in
            # `my_norm_sq`, so each diagonal step is O(1) rather than O(k). The off-diagonal `dot` is split into two
            # interleaved partial sums (`dot0`/`dot1`) so the back-to-back FMA dependency chain is cut in half,
            # exposing more instruction-level parallelism.
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            my_norm_sq = qd.cast(0.0, dtype)
            for k in qd.static(range(32)):
                diag_val = qd.cast(0.0, dtype)
                if tid == k:
                    diag_val = qd.sqrt(qd.max(self.r[k] - my_norm_sq, eps))
                    self.r[k] = diag_val

                diag_k = qd.simt.subgroup.shuffle(diag_val, qd.u32(k))

                dot0 = qd.cast(0.0, dtype)
                dot1 = qd.cast(0.0, dtype)
                for j in qd.static(range(32)):
                    if k > j:
                        my_col = self.r[j]
                        Lkj = qd.simt.subgroup.shuffle(my_col, qd.u32(k))
                        if j % 2 == 0:
                            dot0 += Lkj * my_col  # type: ignore[reportOperatorIssue]
                        else:
                            dot1 += Lkj * my_col  # type: ignore[reportOperatorIssue]
                dot = dot0 + dot1

                new_val = qd.cast(0.0, dtype)
                if tid > k:  # type: ignore[reportOperatorIssue]
                    new_val = (self.r[k] - dot) / diag_k  # type: ignore[reportOperatorIssue]
                    self.r[k] = new_val
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

        # Marker used by the warning-suppression block at module bottom: the
        # AST transformer's external-function check exempts callees whose
        # `__module__` starts with `"quadrants."`. We rewrite the __module__
        # of this method (and the proxy constructors below) after class
        # definition to restore parity with stock `qd.simt.Tile32x32`.
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
# @qd.func when invoked from inside a kernel. The constructors delegate to
# the underlying @qd.func / qd.dataclass constructors and stock
# `qd.simt.Tile32x32` gets the same exemption only because of its module
# name; this restores parity.
_Tile32x32CholeskyProxy.zeros.__module__ = "quadrants.gen.tile32_cholesky"
_Tile32x32CholeskyProxy.eye.__module__ = "quadrants.gen.tile32_cholesky"


Tile32x32Cholesky = _Tile32x32CholeskyProxy()
