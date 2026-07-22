import ctypes
import datetime
import functools
import io
import logging
import math
import numbers
import os
import random
import sys
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import field
from importlib import import_module
from itertools import combinations
from typing import Any, NoReturn, Optional, Sequence

import cpuinfo
import quadrants as qd
import numpy as np
import psutil
import pyglet
import torch


import genesis as gs


LOGGER = logging.getLogger(__name__)


class DeprecationError(Exception):
    pass


def raise_exception(msg="Something went wrong.") -> NoReturn:
    raise gs.GenesisException(msg)


def raise_exception_from(msg="Something went wrong.", cause=None) -> NoReturn:
    raise gs.GenesisException(msg) from cause


class redirect_libc_stderr:
    """
    Context-manager that temporarily redirects C / C++ std::cerr (i.e. the C `stderr` file descriptor 2) to a given
    Python file-like object's fd.

    Works on macOS, Linux (glibc / musl), and Windows (MSVCRT / Universal CRT ≥ VS2015).
    """

    def __init__(self, fd):
        self.fd = fd
        self.stderr_fileno = None
        self.original_stderr_fileno = None

    def __enter__(self):
        try:
            self.stderr_fileno = sys.stderr.fileno()
        except (io.UnsupportedOperation, AttributeError):
            # Do nothing is not a real OS-level file descriptor but rather some IO buffer
            return self

        self.original_stderr_fileno = os.dup(self.stderr_fileno)
        sys.stderr.flush()

        if os.name == "posix":  # macOS, Linux, *BSD, ...
            libc = ctypes.CDLL(None)
            libc.fflush(None)
            libc.dup2(self.fd.fileno(), self.stderr_fileno)
        elif os.name == "nt":  # Windows
            # FIXME: Do not redirect stderr on Windows OS when running pytest, otherwise it will raise this exception:
            # "OSError: [WinError 6] The handle is invalid"
            if "PYTEST_VERSION" not in os.environ:
                msvcrt = ctypes.CDLL("msvcrt")
                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

                msvcrt.fflush(None)
                msvcrt._dup2(self.fd.fileno(), self.stderr_fileno)

                STDERR_HANDLE = -12
                new_os_handle = msvcrt._get_osfhandle(self.fd.fileno())
                kernel32.SetStdHandle(STDERR_HANDLE, new_os_handle)
        else:
            gs.logger.warning(f"Unsupported platform for redirecting libc stderr: {sys.platform}")

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.stderr_fileno is None:
            return

        if os.name == "posix":
            libc = ctypes.CDLL(None)
            sys.stderr.flush()
            libc.fflush(None)
            libc.dup2(self.original_stderr_fileno, self.stderr_fileno)
        elif os.name == "nt":
            if "PYTEST_VERSION" not in os.environ:
                msvcrt = ctypes.CDLL("msvcrt")
                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

                sys.stderr.flush()
                msvcrt.fflush(None)
                msvcrt._dup2(self.original_stderr_fileno, self.stderr_fileno)

                STDERR_HANDLE = -12
                orig_os_handle = msvcrt._get_osfhandle(self.original_stderr_fileno)
                kernel32.SetStdHandle(STDERR_HANDLE, orig_os_handle)

        os.close(self.original_stderr_fileno)
        self.stderr_fileno = None
        self.original_stderr_fileno = None


def assert_initialized(cls):
    original_init = cls.__init__

    @functools.wraps(original_init)
    def new_init(self, *args, **kwargs):
        if not gs._initialized:
            gs.raise_exception("Genesis hasn't been initialized. Did you call `gs.init()`?")
        original_init(self, *args, **kwargs)

    cls.__init__ = new_init
    return cls


def assert_unbuilt(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if self.is_built:
            gs.raise_exception("Scene is already built.")
        return method(self, *args, **kwargs)

    return wrapper


def assert_built(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self.is_built:
            gs.raise_exception(f"{type(self).__name__} is not built yet.")
        return method(self, *args, **kwargs)

    return wrapper


def with_lock(method):
    """Acquire ``self._lock`` before running the wrapped method."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


def set_random_seed(seed):
    # Note: we don't set seed for quadrants, since Quadrants doesn't support stochastic operations in gradient computation.
    # Therefore, we only allow deterministic Quadrants operations.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(backend: gs.constants.backend, device_idx: Optional[int] = None):
    if backend == gs.gpu:
        if torch.cuda.is_available():
            if torch.version.hip:
                backend = gs.amdgpu
            else:  # torch.version.cuda:
                backend = gs.cuda
        elif sys.platform == "darwin":
            backend = gs.metal
        else:
            gs.raise_exception("No Torch GPU device available.")

    if backend in (gs.cuda, gs.amdgpu):
        if (
            not torch.cuda.is_available()
            or (backend == gs.cuda and not torch.version.cuda)
            or (backend == gs.amdgpu and not torch.version.hip)
        ):
            gs.raise_exception(f"Torch device 'cuda' not available for backend '{backend}'.")
        if device_idx is None:
            device_idx = torch.cuda.current_device()
        device = torch.device("cuda", device_idx)
        device_property = torch.cuda.get_device_properties(device)
        device_name = device_property.name
        total_mem = device_property.total_memory / 1024**3
    elif backend == gs.metal:
        if not torch.backends.mps.is_available():
            gs.raise_exception("Torch device 'mps' not available.")
        # on mac, cpu and gpu are in the same physical hardware and sharing memory
        _, device_name, total_mem, _ = get_device(gs.cpu)
        assert not device_idx, "Specifying device index other than 0 is not support for Torch Metal device."
        device = torch.device("mps")
    else:
        cpu_info = cpuinfo.get_cpu_info()
        device_name = next(filter(None, map(cpu_info.get, ("brand_raw", "hardware_raw", "vendor_id_raw"))))
        total_mem = psutil.virtual_memory().total / 1024**3
        assert not device_idx, "Specifying device index other than 0 is not support for Torch CPU device."
        device = torch.device("cpu")
    return device, device_name, total_mem, backend


def get_gpu_core_count() -> int:
    """Return the number of GPU compute cores for the active device.

    This is the env count above which one-thread-per-env already saturates the GPU, so cooperative or tiled kernels
    stop being worthwhile. NVIDIA reports 128 CUDA cores per SM and AMD/ROCm 64 stream processors per CU; for backends
    where the driver cannot be queried (Metal, or a GPU without a torch.cuda device) an upper-bound estimate is used.
    """
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(torch.cuda.current_device())
        cores_per_unit = 64 if torch.version.hip else 128
        return gpu_props.multi_processor_count * cores_per_unit
    if gs.backend == gs.metal:
        # Upper-bound estimate for Apple Silicon: 40 GPU cores * 128 ALUs.
        return 5120
    # Fallback for other GPU backends (e.g. Vulkan), using AMD MI350X (256 CUs * 64 cores) as a baseline.
    return 16384


def fits_in_gpu_shared_memory(*dims: int) -> bool:
    """Whether a dense ``gs.qd_float`` array of shape ``dims`` fits in one block's GPU shared memory."""
    if gs.backend == gs.cpu:
        gs.raise_exception("CPU backend not supported by this method.")
    itemsize = 4 if gs.qd_float == qd.f32 else 8
    return math.prod(dims) * itemsize <= qd.lang.impl.get_max_shared_memory_bytes(is_lowerbound_ok=True)


def get_src_dir():
    return os.path.dirname(gs.__file__)


def get_gen_log_dir():
    current_time = datetime.datetime.now()
    unique_id = current_time.strftime("%Y%m%d_%H%M%S_%f")
    return os.path.join(os.path.dirname(gs.__file__), "gen", "logs", unique_id)


def get_assets_dir():
    return os.path.join(get_src_dir(), "assets")


def get_cache_dir():
    cache_dir = os.environ.get("GS_CACHE_FILE_PATH")
    if cache_dir is not None:
        return cache_dir
    root_cache_dir = None
    if sys.platform == "linux":
        root_cache_dir = os.environ.get("XDG_CACHE_HOME")
    if root_cache_dir is None:
        root_cache_dir = os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(root_cache_dir, "genesis")


def get_gsd_cache_dir():
    return os.path.join(get_cache_dir(), "gsd")


def get_gnd_cache_dir():
    return os.path.join(get_cache_dir(), "terrain")


def get_cvx_cache_dir():
    return os.path.join(get_cache_dir(), "cvx")


def get_ptc_cache_dir():
    return os.path.join(get_cache_dir(), "ptc")


def get_fps_pc_cache_dir():
    return os.path.join(get_cache_dir(), "fps_pc")


def get_tet_cache_dir():
    return os.path.join(get_cache_dir(), "tet")


def get_gel_cache_dir():
    return os.path.join(get_cache_dir(), "gel")


def get_remesh_cache_dir():
    return os.path.join(get_cache_dir(), "rm")


def get_wt_cache_dir():
    return os.path.join(get_cache_dir(), "wt")


def get_wth_cache_dir():
    return os.path.join(get_cache_dir(), "wth")


def get_exr_cache_dir():
    return os.path.join(get_cache_dir(), "exr")


def get_usd_cache_dir():
    return os.path.join(get_cache_dir(), "usd")


_CLEARABLE_CACHES: list[Callable[[], None]] = []


def register_cache_clear(cache_clear: Callable[[], None]) -> None:
    """Register a callback that drops a module-level cache, invoked by clear_caches on genesis teardown.

    Pass the cache's own clearing method, e.g. the cache_clear of a functools.lru_cache or the clear of a manual dict
    cache. This lets module-level asset caches (parsed meshes, baked textures, ...) release the large arrays they hold
    for destroyed scenes without the teardown path having to know about each one.
    """
    _CLEARABLE_CACHES.append(cache_clear)


def clear_caches() -> None:
    """Drop every cache registered through register_cache_clear."""
    for cache_clear in _CLEARABLE_CACHES:
        cache_clear()


class SizeCappedCache:
    """An LRU cache bounded by the total byte footprint of its values rather than by their count.

    Each value is stored together with an explicit size in bytes; once the running total exceeds max_bytes the
    least-recently-used entries are evicted until it fits again (the most-recent entry is always kept). This suits
    caching a handful of large, scene-independent arrays - such as processed collision geometry - where the number of
    distinct entries is a poor proxy for the memory actually held. An optional max_entries also caps the entry count,
    so values whose reported size is small or zero (e.g. flat-color textures) cannot accumulate without bound. It
    registers itself with register_cache_clear so it is dropped together with the other asset caches on genesis
    teardown.
    """

    def __init__(self, max_bytes: int, max_entries: "int | None" = None) -> None:
        self._max_bytes = max_bytes
        self._max_entries = max_entries
        self._store: "OrderedDict[Any, tuple[Any, int]]" = OrderedDict()
        self._total_bytes = 0
        register_cache_clear(self.clear)

    def get(self, key: Any) -> Any:
        entry = self._store.get(key)
        if entry is None:
            return None
        self._store.move_to_end(key)
        return entry[0]

    def put(self, key: Any, value: Any, n_bytes: int) -> None:
        previous = self._store.pop(key, None)
        if previous is not None:
            self._total_bytes -= previous[1]
        self._store[key] = (value, n_bytes)
        self._total_bytes += n_bytes
        while len(self._store) > 1 and (
            self._total_bytes > self._max_bytes
            or (self._max_entries is not None and len(self._store) > self._max_entries)
        ):
            _, (_, evicted_bytes) = self._store.popitem(last=False)
            self._total_bytes -= evicted_bytes

    def clear(self) -> None:
        self._store.clear()
        self._total_bytes = 0


def geometric_mean(a, b):
    """Geometric mean of two non-negative values: sqrt(a * b)."""
    if a < 0 or b < 0:
        gs.raise_exception(f"geometric_mean requires non-negative values, got {a} and {b}.")
    return math.sqrt(a * b)


def harmonic_mean(a, b):
    """Harmonic mean of two non-negative values: 2 * (a * b) / (a + b)."""
    if a < 0 or b < 0:
        gs.raise_exception(f"harmonic_mean requires non-negative values, got {a} and {b}.")
    if a == 0 or b == 0:
        return 0.0
    return 2 * (a * b) / (a + b)


def assert_gs_tensor(x):
    if not isinstance(x, gs.Tensor):
        gs.raise_exception("Only accepts genesis.Tensor.")


def to_gs_tensor(x, dtype: torch.dtype | None = None):
    if isinstance(x, gs.Tensor):
        tensor = x
    elif isinstance(x, torch.Tensor):
        tensor = gs.Tensor(x)
    else:
        tensor = gs.from_numpy(np.asarray(x))
    return tensor.to(dtype=dtype, device=gs.device)


def tensor_to_cpu(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
    return x


def tensor_to_array(x: torch.Tensor, dtype: type[np.generic] | None = None) -> np.ndarray:
    return np.asarray(tensor_to_cpu(x), dtype=dtype)


def is_approx_multiple(a, b, tol=1e-7):
    return abs(a % b) < tol or abs(b - (a % b)) < tol


def gaussian_crosstalk_kernel(n_rows: int, n_cols: int, sigma: float, spacing: float | tuple[float, float] = 1.0):
    """
    Build an L1-normalized 2D Gaussian convolution kernel for spatial tactile crosstalk.

    The kernel is a discrete isotropic Gaussian ``exp(-(d / sigma)**2 / 2)`` sampled on an ``n_rows x n_cols`` grid
    centered on the self taxel, then normalized to sum 1 (so a uniform field passes through unchanged). Pass the
    result as a sensor's ``crosstalk_kernel`` to spread each taxel's signal onto its neighbors.

    ``n_rows`` and ``n_cols`` must be odd so the kernel has a center tap (the self weight). ``spacing`` is the taxel
    pitch in the same units as ``sigma`` (a scalar, or ``(row_spacing, col_spacing)`` for an anisotropic grid);
    default ``1.0`` measures ``sigma`` in taxel cells.
    """
    if n_rows % 2 == 0 or n_cols % 2 == 0:
        raise_exception(
            f"gaussian_crosstalk_kernel requires odd n_rows, n_cols (center tap); got ({n_rows}, {n_cols})."
        )
    if sigma <= 0.0:
        raise_exception(f"gaussian_crosstalk_kernel requires sigma > 0; got {sigma}.")
    s_row, s_col = (spacing, spacing) if isinstance(spacing, numbers.Number) else spacing
    rows = (np.arange(n_rows, dtype=float) - n_rows // 2) * s_row
    cols = (np.arange(n_cols, dtype=float) - n_cols // 2) * s_col
    g_row = np.exp(-(rows**2) / (2.0 * sigma * sigma))
    g_col = np.exp(-(cols**2) / (2.0 * sigma * sigma))
    kernel = np.outer(g_row, g_col)
    return kernel / kernel.sum()


def concat_with_tensor(
    tensor: torch.Tensor, value, expand: tuple[int, ...] | None = None, dim: int = 0, flatten: bool = False
):
    """Helper method to concatenate a value (not necessarily a tensor) with a tensor."""
    if not isinstance(value, torch.Tensor):
        if isinstance(value, (numbers.Real, np.floating, numbers.Integral, np.integer)):
            value = [value]
        value = torch.tensor(value, dtype=tensor.dtype, device=tensor.device)
    if expand is not None:
        value = value.expand(*expand)
    if dim < 0:
        dim = tensor.ndim + dim
    if flatten:
        value = value.flatten()
    assert (
        0 <= dim < tensor.ndim
        and tensor.ndim == value.ndim
        and all(e_1 == e_2 for i, (e_1, e_2) in enumerate(zip(tensor.shape, value.shape)) if e_1 > 0 and i != dim)
    )
    if tensor.numel() == 0:
        return value
    return torch.cat([tensor, value], dim=dim)


def make_tensor_field(shape: tuple[int, ...] = (), dtype_factory: Callable[[], torch.dtype] | None = None):
    """
    Helper method to create a tensor field for dataclasses.

    Parameters
    ----------
    shape : tuple
        The shape of the tensor field. It must have zero elements, otherwise it will trigger an exception.
    dtype_factory : Callable[[], torch.dtype], optional
        The factory function to create the dtype of the tensor field. Default is gs.tc_float.
        A factory is used because gs types may not be available at the time of field creation.
    """
    assert not shape or math.prod(shape) == 0

    def _default_factory():
        nonlocal shape, dtype_factory
        dtype = dtype_factory() if dtype_factory is not None else gs.tc_float
        return torch.empty(shape, dtype=dtype, device=gs.device)

    return field(default_factory=_default_factory)


def try_get_display_size() -> tuple[int | None, int | None, float | None]:
    """
    Try to connect to display if it exists and get the screen size.

    If there is no display, this function will throw an exception.

    Returns
    -------
    screen_height : int | None
        The height of the screen in pixels.
    screen_width : int | None
        The width of the screen in pixels.
    screen_scale : float | None
        The scale of the screen.
    """
    # Resolve pyglet's native display backend directly (under 'pyglet.canvas' before 2.0, 'pyglet.display' from 2.0 on),
    # never the placeholder headless one whose finalizer calls eglTerminate on the EGL display the offscreen renderers
    # share. A headless process then raises here, reported as no display - the fallback to a default size is the
    # viewer's concern. Reuse a display pyglet already has open if any (the isinstance check skips a headless one).
    native = {
        "darwin": ("cocoa", "CocoaDisplay"),
        "win32": ("win32", "Win32Display"),
        "cygwin": ("win32", "Win32Display"),
        "linux": ("xlib", "XlibDisplay"),
    }.get(pyglet.compat_platform)
    if native is None:
        raise NotImplementedError(f"No display interface available for platform '{pyglet.compat_platform}'.")
    # The backend submodule depends on the platform and pyglet version, and a foreign-platform one fails to import
    # (e.g. 'win32' off Windows needs Windows-only ctypes), so it cannot be a top-level import; resolving it by
    # computed name avoids a platform-by-version tree of local imports.
    if pyglet.version < "2.0":
        Display = getattr(import_module(f"pyglet.canvas.{native[0]}"), native[1])
        display = next((d for d in pyglet.canvas._displays if isinstance(d, Display)), None)
    else:
        Display = getattr(import_module(f"pyglet.display.{native[0]}"), native[1])
        display = next((d for d in pyglet.display._displays if isinstance(d, Display)), None)
    if display is None:
        display = Display()

    screen = display.get_default_screen()
    if pyglet.version < "2.0":
        screen_scale = 1.0
    else:
        try:
            screen_scale = screen.get_scale()
        except NotImplementedError:
            screen_scale = 1.0
    return screen.height, screen.width, screen_scale


def has_display() -> bool:
    """
    Check if a display is connected.
    """
    try:
        try_get_display_size()
        return True
    except Exception:
        return False


def indices_to_mask(
    *indices: Any, keepdim: bool = True, to_torch: bool = True, boolean_mask: bool = False, raise_if_fancy: bool = False
) -> tuple[slice | int | torch.Tensor, ...]:
    """Converts a sequence of slice-like objects into a multi-dimensional mask corresponding to their cross-product.

    Args:
        keepdim (bool): Whether to keep all dimensions even if masks are integers. Defaults to True.
        to_torch (bool): Whether to force casting collections to torch.Tensor.
        boolean_mask (bool): Whether boolean mask are supported more must be converted to indices via `torch.nonzero`.
        raise_if_fancy (bool): Whether fancy indexing is supported for should raise an exception.
        copy (bool, optional): Wether to raise an exception if the resulting mask requires advanced indexing (aka. fancy
        indexing), which would trigger a copy when extracting slice.
    """
    mask: list[slice | int | torch.Tensor] = []

    is_all_none = True
    num_tensors = 0
    is_tensor: list[bool] = [False] * len(indices)
    for i in range(len(indices) - 1, -1, -1):
        arg = indices[i]
        if arg is None:
            if is_all_none:
                continue
            arg = slice(None)
        else:
            is_all_none = False
            if (arg_type := type(arg)) is slice:
                pass
            elif arg_type is range:
                arg = slice(arg.start, arg.stop, arg.step)
            elif arg_type is int:
                if keepdim:
                    arg = slice(arg, arg + 1)
            else:  # np.ndarray, torch.tensor, list, tuple, np.int32...
                try:
                    is_torch_, is_numpy_ = False, False
                    if isinstance(arg, torch.Tensor):
                        if not boolean_mask and arg.dtype == torch.bool:
                            arg = arg.nonzero()[:, 0]
                        is_scalar_ = arg.dtype != torch.bool and arg.numel() == 1
                        is_torch_ = True
                    elif isinstance(arg, np.ndarray):
                        is_scalar_ = arg.size == 1
                        is_numpy_ = True
                    else:
                        is_scalar_ = len(arg) == 1
                    if is_scalar_:
                        arg = slice(idx := arg.item() if is_torch_ or is_numpy_ else arg[0], idx + 1)
                    else:
                        if raise_if_fancy:
                            gs.raise_exception("This mask requires advanced indexing but 'raise_if_fancy=True'.")
                        if not is_torch_ and to_torch:
                            # Must convert masks to torch if not slice or int since torch will do it anyway.
                            # Note that being contiguous is not required and does not affect performance.
                            arg = torch.tensor(arg, dtype=gs.tc_int, device=gs.device)
                        is_tensor[i] = True
                        num_tensors += 1
                except TypeError:
                    # Try casting to int if 'len' is undefined.
                    # Dealing with this fairly unusual use-case in try-except to avoid slowing down the hot path.
                    arg = int(arg)
                    if keepdim:
                        arg = slice(arg, arg + 1)
        mask.insert(0, arg)

    if num_tensors > 1:
        tensor_idx = 0
        for i in range(len(mask)):
            if is_tensor[i]:
                # assert isinstance(arg, torch.Tensor)
                shape = [1] * num_tensors
                shape[tensor_idx] = -1
                try:
                    mask[i] = mask[i].reshape(shape)
                except AttributeError as e:
                    gs.raise_exception_from("Multi-dimensional masking only supported for 'to_torch=True'.", e)
                tensor_idx += 1

    return tuple(mask)


def _maybe_transpose(tc, value, transpose):
    if not transpose or len(value.shape) <= 1:
        return tc
    return tc.movedim(len(value.shape) - 1, 0)


def _maybe_transpose_np(arr, value, transpose):
    if not transpose or len(value.shape) <= 1:
        return arr
    return np.moveaxis(arr, len(value.shape) - 1, 0)


def _apply_masks(out, value, row_mask, col_mask, keepdim, copy, *, to_torch):
    if row_mask is None and col_mask is None:
        return out
    raise_if_fancy = copy is False
    if len(value.shape) < 2:
        if row_mask is not None and col_mask is not None:
            gs.raise_exception("Cannot specify both row and column masks for tensor with 1D batch.")
        mask = indices_to_mask(
            row_mask if col_mask is None else col_mask,
            to_torch=to_torch,
            keepdim=keepdim,
            raise_if_fancy=raise_if_fancy,
        )
    else:
        mask = indices_to_mask(row_mask, col_mask, to_torch=to_torch, keepdim=keepdim, raise_if_fancy=raise_if_fancy)
    return out[mask]


def qd_to_torch(
    value: qd.Tensor | qd.Field | qd.Ndarray,
    row_mask: int | range | slice | tuple[int, ...] | list[int] | torch.Tensor | np.ndarray | None = None,
    col_mask: int | range | slice | tuple[int, ...] | list[int] | torch.Tensor | np.ndarray | None = None,
    keepdim: bool = True,
    transpose: bool = False,
    *,
    copy: bool | None = None,
) -> torch.Tensor:
    """Converts a Quadrants field / ndarray instance to a PyTorch tensor.

    Args:
        value (qd.Field | qd.Ndarray): Field or Ndarray to be converted.
        row_mask (optional): Rows to extract from batch dimension after transpose if requested.
        col_mask (optional): Columns to extract from batch dimension after transpose if requested.
        keepdim (bool): Whether to keep all dimensions even if masks are integers.
        transpose (bool): Whether move to front the first non-batch dimension.
        copy (bool, optional): Wether to enforce returning a copy no matter what. None to avoid copy if possible
        without raising an exception if not.
    """
    if isinstance(value, qd.Tensor):
        value = value._unwrap()

    # Try efficient shortcut first and only fallback to standard branching if necessary.
    # FIXME: Ideally one should detect if slicing would require a copy to avoid enforcing copy here.
    is_copy = False
    if not gs.use_zerocopy:
        # Transpose if necessary and requested.
        # Note that it is worth transposing here before slicing, as it preserve row-major memory alignment in case of
        # advanced masking, which would spare computation later on if expected from the user.
        if copy is False:
            gs.raise_exception("Specifying 'copy=False' is not supported by this method if 'gs.use_zerocopy=False'.")
        tensor = _maybe_transpose(value.to_torch(), value, transpose)
        is_copy = True
    else:
        try:
            tensor = value._T_tc if transpose else value._tc
            is_copy = False
        except AttributeError:
            try:
                tc = value.to_torch(copy=False)
            except (ValueError, RuntimeError, TypeError):
                if copy is False:
                    raise
                tensor = _maybe_transpose(value.to_torch(), value, transpose)
                is_copy = True
            else:
                value._tc = tc
                value._T_tc = _maybe_transpose(tc, value, True)
                tensor = value._T_tc if transpose else value._tc
                is_copy = False

    if not is_copy:
        # FIXME: DLPack may return old values on Apple Metal if sync is not systematically called manually
        if gs.backend == gs.metal:
            qd.sync()
        if copy:
            tensor = tensor.clone()
            if gs.backend == gs.metal:
                torch.mps.synchronize()

    return _apply_masks(tensor, value, row_mask, col_mask, keepdim, copy, to_torch=True)


def qd_to_numpy(
    value: qd.Tensor | qd.Field | qd.Ndarray,
    row_mask: int | range | slice | tuple[int, ...] | list[int] | torch.Tensor | np.ndarray | None = None,
    col_mask: int | range | slice | tuple[int, ...] | list[int] | torch.Tensor | np.ndarray | None = None,
    keepdim: bool = True,
    transpose: bool = False,
    *,
    copy: bool | None = None,
) -> np.ndarray:
    """Converts a Quadrants field / ndarray instance to a Numpy array.

    Args:
        value (qd.Field | qd.Ndarray): Field or Ndarray to be converted.
        row_mask (optional): Rows to extract from batch dimension after transpose if requested.
        col_mask (optional): Columns to extract from batch dimension after transpose if requested.
        keepdim (bool, optional): Whether to keep all dimensions even if masks are integers.
        transpose (bool, optional): Whether move to front the first non-batch dimension.
        copy (bool, optional): Wether to enforce returning a copy no matter what. None to avoid copy if possible
        without raising an exception if not.
    """
    if isinstance(value, qd.Tensor):
        value = value._unwrap()

    # Try efficient shortcut first and only fallback to standard branching if necessary.
    # FIXME: Ideally one should detect if slicing would require a copy to avoid enforcing copy here.
    if not gs.use_zerocopy:
        # Transpose if necessary and requested.
        # Note that it is worth transposing here before slicing, as it preserve row-major memory alignment in case of
        # advanced masking, which would spare computation later on if expected from the user.
        if copy is False:
            gs.raise_exception("Specifying 'copy=False' is not supported if 'gs.use_zerocopy=False'.")
        array = _maybe_transpose_np(value.to_numpy(), value, transpose)
        is_copy = True
    elif gs.backend != gs.cpu:
        if copy is False:
            gs.raise_exception("Specifying 'copy=False' is not supported by this method if 'gs.backend != gs.cpu'.")
        array = tensor_to_array(qd_to_torch(value, transpose=transpose))
        is_copy = True
    else:
        try:
            array = value._T_np if transpose else value._np
            is_copy = False
        except AttributeError:
            try:
                tc = value.to_torch(copy=False)
            except (RuntimeError, TypeError, ValueError):
                if copy is False:
                    raise
                array = _maybe_transpose_np(value.to_numpy(), value, transpose)
                is_copy = True
            else:
                value._np = tc.numpy()
                value._T_np = _maybe_transpose(tc, value, True).numpy()
                array = value._T_np if transpose else value._np
                is_copy = False

    if copy and not is_copy:
        array = array.copy()

    return _apply_masks(array, value, row_mask, col_mask, keepdim, copy, to_torch=False)


def qd_zero_grad(value) -> None:
    """Zero the `.grad` buffers of a Quadrants field/ndarray, or every grad-bearing slot of a `dataclass` /
    `@qd.data_oriented` struct-of-arrays.

    Reverse-mode accumulation in Genesis writes through `qd.atomic_add`, so adjoint buffers must start at zero between
    consecutive `loss.backward()` calls. Solvers call this from `reset_grad` to clear all owned adjoint storage without
    enumerating fields by name. Zeroing goes through an in-place `zero_()` on the zero-copy torch view of each grad
    buffer, a contiguous memset on the underlying device memory. The writes are left unsynchronized so a caller can
    batch many calls under a single flush: on Metal, call `torch.mps.synchronize()` after the batch and before the
    next quadrants kernel reads the buffers (see set_base_links_quat).
    """
    if value is None:
        return

    if isinstance(value, (qd.Tensor, qd.Field, qd.Ndarray)):
        if value.has_grad():
            grad = value.grad
            if gs.use_zerocopy:
                try:
                    grad_view = qd_to_torch(grad, copy=False)
                    grad_view.zero_()
                except ValueError:
                    # No zero-copy view for this buffer (e.g. an interleaved AOS struct member, or a field whose
                    # in-tree byte offset the installed torch cannot carry through DLPack); fill it in place through
                    # quadrants instead.
                    grad.fill(0.0)
            else:
                grad.fill(0.0)
        return

    cls = type(value)
    try:
        annotations = cls.__dict__["__annotations__"]
    except KeyError as err:
        raise_exception_from(
            f"qd_zero_grad: expected `qd.Field`, `qd.Ndarray`, or a `dataclass` / `@qd.data_oriented` "
            f"struct-of-arrays; got `{cls.__name__}`.",
            cause=err,
        )
    for attr_name in annotations:
        qd_zero_grad(getattr(value, attr_name, None))


def sanitize_index(
    index: int | range | slice | tuple[int, ...] | list[int] | torch.Tensor | np.ndarray | None,
    expected_size: int,
    max_size: int,
    dim: int,
    name: str,
) -> torch.Tensor:
    if index is None:
        index = range(max_size)
    elif isinstance(index, slice):
        index = range(
            index.start or 0,
            index.stop if index.stop is not None else max_size,
            index.step or 1,
        )
    elif isinstance(index, (int, np.integer)):
        index = [index]
    elif isinstance(index, torch.Tensor) and index.dtype == torch.bool:
        index, *_ = torch.where(index)

    index = torch.as_tensor(index, dtype=gs.tc_int, device=gs.device)

    ndim = index.ndim
    if ndim == 0:
        index = index[None]
    elif ndim > 1:
        dim_info = f" `{name}`" if name else ""
        gs.raise_exception(f"Invalid shape: {index.shape}. Expecting 0D or 1D tensor for {dim}-th index{dim_info}.")

    if expected_size != -1 and expected_size != len(index):
        dim_info = f" `{name}`" if name else ""
        gs.raise_exception(
            f"Invalid shape: {index.shape}. Expecting 1D tensor of length {expected_size} for {dim}-th index{dim_info}."
        )

    # FIXME: This check is too expensive
    # if not (0 <= dim_idx & dim_idx < size).all():
    #     dim_info = f" `{name}`" if name else ""
    #     gs.raise_exception(f"Indices out-of-range for {i}-th index{dim_info}.")

    return index.contiguous()


def sanitize_indices(
    indices: Sequence[int | range | slice | tuple[int, ...] | list[int] | torch.Tensor | np.ndarray | None],
    expected_shape: Sequence[int],
    max_shape: Sequence[int],
    dim_names: tuple[str, ...] | list[str],
) -> tuple[torch.Tensor, ...]:
    indices_: list[torch.Tensor] = []
    expected_shape = list(expected_shape)
    for i, dim_idx in enumerate(indices):
        dim_idx = sanitize_index(dim_idx, expected_shape[i], max_shape[i], i, dim_names[i])
        expected_shape[i] = len(dim_idx)
        indices_.append(dim_idx)
    return tuple(indices_)


def broadcast_tensor(
    tensor: "np.typing.ArrayLike | None",
    dtype: torch.dtype,
    expected_shape: tuple[int, ...] | list[int],
    dim_names: tuple[str, ...] | list[str] | None = None,
) -> torch.Tensor:
    if dim_names is None:
        dim_names = ("",) * len(expected_shape)

    if tensor is None:
        if any(size == -1 for size in expected_shape):
            gs.raise_exception(
                "Tensor not pre-allocated and expected shape not fully specified but allocation is not skipped."
            )
        return torch.empty(expected_shape, dtype=dtype, device=gs.device)

    tensor_ = torch.as_tensor(tensor, dtype=dtype, device=gs.device)

    tensor_shape = tensor_.shape
    tensor_ndim = len(tensor_shape)
    expected_ndim = len(expected_shape)

    # Expand current tensor shape with extra dims of size 1 if necessary before expanding to expected shape
    if tensor_ndim == 0:
        tensor_ = tensor_[None]
    elif tensor_ndim < expected_ndim and not all(
        [d1 == d2 or d2 == -1 for d1, d2 in zip(tensor_shape, expected_shape[-tensor_ndim:])]
    ):
        # Try expanding first dimensions if priority
        for dims_valid in tuple(combinations(range(expected_ndim), tensor_ndim))[::-1]:
            curr_idx = 0
            expanded_shape = []
            for i in range(expected_ndim):
                if i in dims_valid:
                    dim, size = tensor_.shape[curr_idx], expected_shape[i]
                    if dim == size or dim == 1 or size == -1:
                        expanded_shape.append(dim)
                        curr_idx += 1
                    else:
                        break
                else:
                    expanded_shape.append(1)
            else:
                if curr_idx == tensor_ndim:
                    tensor_ = tensor_.reshape(expanded_shape)
                    break
    elif tensor_ndim > expected_ndim:
        gs.raise_exception(f"Invalid input shape: {tensor_shape}. Expecting at most {expected_ndim}D tensor.")

    try:
        tensor_ = tensor_.expand(expected_shape)
    except RuntimeError as e:
        msg_err = f"Invalid input shape: {tuple(tensor_.shape)}."
        msg_infos: list[str] = []
        for i, name in enumerate(dim_names):
            size = expected_shape[i]
            if size > 0 and i < tensor_.ndim and (dim := tensor_.shape[i]) != 1 and dim != size:
                if name:
                    msg_infos.append(f"Dimension {i} consistent with len({name})={size}")
                else:
                    msg_infos.append(f"Dimension {i} consistent with required size {size}")
        if msg_infos:
            msg_err += f" {' & '.join(msg_infos)}."
        else:
            msg_err += f" Expected shape: {tuple(expected_shape)}."
        gs.raise_exception_from(msg_err, e)

    return tensor_


def sanitize_indexed_tensor(
    tensor: "np.typing.ArrayLike | None",
    dtype: torch.dtype,
    indices: Sequence[int | range | slice | tuple[int, ...] | list[int] | torch.Tensor | np.ndarray | None],
    expected_shape: tuple[int, ...] | list[int],
    max_shape: tuple[int, ...] | list[int],
    dim_names: tuple[str, ...] | list[str],
    skip_allocation: bool = False,
) -> tuple[torch.Tensor | None, tuple[torch.Tensor, ...]]:
    indices_ = sanitize_indices(indices, expected_shape, max_shape, dim_names)

    is_preallocated = tensor is not None
    if is_preallocated or not skip_allocation:
        expected_shape = [*map(len, indices_), *expected_shape[len(indices_) :]]
        tensor = broadcast_tensor(tensor, dtype, expected_shape, dim_names).contiguous()

    return tensor, tuple(indices_)


def get_indexed_shape(tensor_shape, indices):
    """Compute the resulting shape after advanced indexing without performing the operation."""
    ndim = len(tensor_shape)

    # Expand ellipsis if present
    ellipsis_count = sum(1 for idx in indices if idx is Ellipsis)
    if ellipsis_count == 1:
        idx = indices.index(Ellipsis)
        indices = (*indices[:idx], *(slice(None),) * (ndim - len(indices) + 1), *indices[idx + 1 :])
    elif ellipsis_count > 1:
        raise IndexError("Only one ellipsis (...) is allowed")

    # Compute the broadcasted shape of all tensor indices
    broadcast_shape = torch.broadcast_shapes(*[idx.shape for idx in indices if isinstance(idx, torch.Tensor)])

    # Build output shape
    output_shape = []
    curr_idx = 0
    inserted_broadcast = False
    for idx in indices:
        if isinstance(idx, int):
            curr_idx += 1
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(tensor_shape[curr_idx])
            if step > 0:
                size = max(0, (stop - start + step - 1) // step)
            else:
                size = max(0, (stop - start + step + 1) // step)
            output_shape.append(size)
            curr_idx += 1
        else:  # isinstance(idx, torch.Tensor):
            if not inserted_broadcast:
                output_shape.extend(broadcast_shape)
                inserted_broadcast = True
            curr_idx += 1
    output_shape += tensor_shape[curr_idx:]

    return tuple(output_shape)


def assign_indexed_tensor(
    tensor: torch.Tensor,
    indices: tuple[int | slice | torch.Tensor, ...],
    value: "np.typing.ArrayLike",
    dim_names: tuple[str, ...] | list[str] | None = None,
) -> None:
    if isinstance(tensor, np.ndarray):
        value = torch.as_tensor(value)
    try:
        tensor[indices] = value
    except (TypeError, RuntimeError):
        # Try extended broadcasting as a fallback to avoid slowing down the hot path
        indexed_shape = get_indexed_shape(tensor.shape, indices) if indices else tensor.shape
        tensor[indices] = broadcast_tensor(value, tensor.dtype, indexed_shape, dim_names)
