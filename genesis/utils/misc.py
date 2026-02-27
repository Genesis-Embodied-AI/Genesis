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
from dataclasses import field
from itertools import combinations
from typing import Any, NoReturn, Optional, Type, Sequence

import cpuinfo
import quadrants as qd
import numpy as np
import psutil
import pyglet
import torch

from quadrants.lang.util import to_pytorch_type, to_numpy_type
from quadrants._kernels import tensor_to_ext_arr, matrix_to_ext_arr, ndarray_to_ext_arr, ndarray_matrix_to_ext_arr

import genesis as gs


LOGGER = logging.getLogger(__name__)


# FIXME: qd.Field does not support zero-copy on Metal for 'torch<=2.9.1'.
# See: https://github.com/pytorch/pytorch/pull/168193
TORCH_MPS_SUPPORT_DLPACK_FIELD = tuple(map(int, torch.__version__.replace("+", ".").split(".")[:3])) > (2, 9, 1)


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
        except io.UnsupportedOperation:
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


def get_tet_cache_dir():
    return os.path.join(get_cache_dir(), "tet")


def get_gel_cache_dir():
    return os.path.join(get_cache_dir(), "gel")


def get_remesh_cache_dir():
    return os.path.join(get_cache_dir(), "rm")


def get_exr_cache_dir():
    return os.path.join(get_cache_dir(), "exr")


def get_usd_cache_dir():
    return os.path.join(get_cache_dir(), "usd")


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


def tensor_to_array(x: torch.Tensor, dtype: Type[np.generic] | None = None) -> np.ndarray:
    return np.asarray(tensor_to_cpu(x), dtype=dtype)


def is_approx_multiple(a, b, tol=1e-7):
    return abs(a % b) < tol or abs(b - (a % b)) < tol


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


def make_tensor_field(shape: tuple[int, ...] = (), dtype: torch.dtype | None = None):
    """
    Helper method to create a tensor field for dataclasses.

    Parameters
    ----------
    shape : tuple
        The shape of the tensor field. It must have zero elements, otherwise it will trigger an exception.
    dtype : torch.dtype, optional
        Data type of the tensor field. Default is gs.tc_float.
    """
    assert not shape or math.prod(shape) == 0

    def _default_factory():
        nonlocal shape, dtype
        return torch.empty(shape, dtype=dtype or gs.tc_float, device=gs.device)

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
    if pyglet.version < "2.0":
        display = pyglet.canvas.Display()
        screen = display.get_default_screen()
        screen_scale = 1.0
    else:
        display = pyglet.display.get_display()
        screen = display.get_default_screen()
        try:
            screen_scale = screen.get_scale()
        except NotImplementedError:
            # Probably some headless screen
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


# -------------------------------------- QUADRANTS SPECIALIZATION --------------------------------------

_to_torch_type_fast = functools.lru_cache(maxsize=None)(to_pytorch_type)
_to_numpy_type_fast = functools.lru_cache(maxsize=None)(to_numpy_type)

TO_EXT_ARR_FAST_MAP = dict(
    (
        (qd.ScalarField, tensor_to_ext_arr),
        (qd.MatrixField, matrix_to_ext_arr),
        (qd.ScalarNdarray, ndarray_to_ext_arr),
        (qd.MatrixNdarray, ndarray_matrix_to_ext_arr),
    )
)


def qd_to_python(
    value: qd.Field | qd.Ndarray,
    transpose: bool = False,
    copy: bool | None = None,
    to_torch: bool = True,
) -> torch.Tensor | np.ndarray:
    """Converts a Quadrants field / ndarray instance to a PyTorch tensor / Numpy array.

    Args:
        value (qd.Field | qd.Ndarray): Field or Ndarray to be converted.
        transpose (bool, optional): Whether to move the last batch dimension in front. Defaults to False.
        copy (bool, optional): Wether to enforce returning a copy no matter what. None to avoid copy if possible
        without raising an exception if not.
        to_torch (bool): Whether to convert to Torch tensor or Numpy array. Defaults to True.
    """
    # Get batch size if possible
    try:
        batch_shape = value.shape
    except AttributeError:
        if isinstance(value, qd.Matrix):
            raise ValueError("Tensor of type 'qd.Vector', 'qd.Matrix' not supported.")
        raise

    # Check if copy mode is supported while setting default mode if not specified.
    # FIXME: Torch>2.9.1 still does not support bytes_offset for 0-dim dlpack.
    data_type = type(value)
    is_field = issubclass(data_type, qd.Field)
    use_zerocopy = gs.use_zerocopy and (
        (TORCH_MPS_SUPPORT_DLPACK_FIELD or gs.backend != gs.metal or not is_field)
        and (batch_shape or not issubclass(data_type, qd.ScalarField))
    )
    if not use_zerocopy or (not to_torch and gs.backend != gs.cpu):
        if copy is False:
            gs.raise_exception(
                "Specifying 'copy=False' is not supported if 'gs.use_zerocopy=False' or ('to_torch=False' and "
                "'gs.backend != gs.cpu')."
            )
        copy = True
    elif copy is None:
        copy = False

    # Leverage zero-copy if enabled
    if use_zerocopy:
        while True:
            try:
                if to_torch or gs.backend != gs.cpu:
                    out = value._T_tc if transpose else value._tc
                else:
                    out = value._T_np if transpose else value._np
                break
            except AttributeError:
                # "Cache" no-owning python-side views of the original Quadrants memory buffer as a hidden attribute
                value_tc = torch.utils.dlpack.from_dlpack(value.to_dlpack())
                if issubclass(data_type, qd.MatrixField) and value.m == 1:
                    value_tc = value_tc.reshape((*batch_shape, value.n))
                value._tc = value_tc
                value._T_tc = value_tc.movedim(batch_ndim - 1, 0) if (batch_ndim := len(batch_shape)) > 1 else value_tc
                if gs.backend == gs.cpu:
                    value._np = value_tc.numpy()
                    value._T_np = value._T_tc.numpy()

        # FIXME: DLPack may return old values on Apple Metal for field if sync is not systematically called manually
        if is_field and gs.backend == gs.metal:
            qd.sync()

        if copy:
            if to_torch:
                out = out.clone()
            elif gs.backend != gs.cpu:
                out = tensor_to_array(out)
            else:
                out = out.copy()
        return out

    # Extract value as a whole.
    # Note that this is usually much faster than using a custom kernel to extract a slice.
    # The implementation is based on `quadrants.lang.(ScalarField | MatrixField).to_torch`.
    is_metal = gs.device.type == "mps"
    out_dtype = _to_torch_type_fast(value.dtype) if to_torch else _to_numpy_type_fast(value.dtype)
    if issubclass(data_type, (qd.ScalarField, qd.ScalarNdarray)):
        if to_torch:
            out = torch.zeros(batch_shape, dtype=out_dtype, device="cpu" if is_metal else gs.device)
        else:
            out = np.zeros(batch_shape, dtype=out_dtype)
        TO_EXT_ARR_FAST_MAP[data_type](value, out)
    elif issubclass(data_type, qd.MatrixField):
        as_vector = value.m == 1
        shape_ext = (value.n,) if as_vector else (value.n, value.m)
        if to_torch:
            out = torch.empty(batch_shape + shape_ext, dtype=out_dtype, device="cpu" if is_metal else gs.device)
        else:
            out = np.zeros(batch_shape + shape_ext, dtype=out_dtype)
        TO_EXT_ARR_FAST_MAP[data_type](value, out, as_vector)
    elif issubclass(data_type, (qd.VectorNdarray, qd.MatrixNdarray)):
        layout_is_aos = 1
        as_vector = issubclass(data_type, qd.VectorNdarray)
        shape_ext = (value.n,) if as_vector else (value.n, value.m)
        if to_torch:
            out = torch.empty(batch_shape + shape_ext, dtype=out_dtype, device="cpu" if is_metal else gs.device)
        else:
            out = np.zeros(batch_shape + shape_ext, dtype=out_dtype)
        TO_EXT_ARR_FAST_MAP[qd.MatrixNdarray](value, out, layout_is_aos, as_vector)
    else:
        gs.raise_exception(f"Unsupported type '{type(value)}'.")
    if to_torch and is_metal:
        out = out.to(gs.device)

    # Transpose if necessary and requested.
    # Note that it is worth transposing here before slicing, as it preserve row-major memory alignment in case of
    # advanced masking, which would spare computation later on if expected from the user.
    if transpose and (batch_ndim := len(batch_shape)) > 1:
        if to_torch:
            out = out.movedim(batch_ndim - 1, 0)
        else:
            out = np.moveaxis(out, batch_ndim - 1, 0)

    return out


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


def qd_to_torch(
    value: qd.Field | qd.Ndarray,
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
    # Try efficient shortcut first and only fallback to standard branching if necessary.
    # FIXME: Ideally one should detect if slicing would require a copy to avoid enforcing copy here.
    if gs.use_zerocopy:
        try:
            tensor = value._T_tc if transpose else value._tc
            # FIXME: DLPack may return old values on Apple Metal for field if sync is not systematically called manually
            if isinstance(value, qd.Field) and gs.backend == gs.metal:
                qd.sync()
            if copy:
                tensor = tensor.clone()
        except AttributeError:
            tensor = qd_to_python(value, transpose, copy=copy, to_torch=True)
    else:
        tensor = qd_to_python(value, transpose, copy=copy, to_torch=True)

    if row_mask is None and col_mask is None:
        return tensor

    raise_if_fancy = copy is False
    if len(value.shape) < 2:
        if row_mask is not None and col_mask is not None:
            gs.raise_exception("Cannot specify both row and column masks for tensor with 1D batch.")
        mask = indices_to_mask(
            row_mask if col_mask is None else col_mask, to_torch=True, keepdim=keepdim, raise_if_fancy=raise_if_fancy
        )
    else:
        mask = indices_to_mask(row_mask, col_mask, to_torch=True, keepdim=keepdim, raise_if_fancy=raise_if_fancy)
    return tensor[mask]


def qd_to_numpy(
    value: qd.Field | qd.Ndarray,
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
    tensor = qd_to_python(value, transpose, copy=copy, to_torch=False)
    if row_mask is None and col_mask is None:
        return tensor

    raise_if_fancy = copy is False
    if len(value.shape) < 2:
        if row_mask is not None and col_mask is not None:
            gs.raise_exception("Cannot specify both row and column masks for tensor with 1D batch.")
        mask = indices_to_mask(
            row_mask if col_mask is None else col_mask, to_torch=False, keepdim=keepdim, raise_if_fancy=raise_if_fancy
        )
    else:
        mask = indices_to_mask(row_mask, col_mask, to_torch=False, keepdim=keepdim, raise_if_fancy=raise_if_fancy)
    return tensor[mask]


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
