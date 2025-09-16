import ctypes
import datetime
import functools
import logging
import math
import platform
import random
import types
import shutil
import sys
import os
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, NoReturn, Optional, Type

import numpy as np
import cpuinfo
import psutil
import torch

import gstaichi as ti
from gstaichi.lang.util import is_ti_template, to_pytorch_type
from gstaichi._kernels import tensor_to_ext_arr, matrix_to_ext_arr, ndarray_to_ext_arr, ndarray_matrix_to_ext_arr
from gstaichi.lang import impl
from gstaichi.types import primitive_types
from gstaichi.lang.exception import handle_exception_from_cpp

import genesis as gs
from genesis.constants import backend as gs_backend


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

    # --------------------------------------------------
    # Enter: duplicate stderr → tmp, dup2(target) → stderr
    # --------------------------------------------------
    def __enter__(self):
        self.stderr_fileno = sys.stderr.fileno()
        self.original_stderr_fileno = os.dup(self.stderr_fileno)
        sys.stderr.flush()

        if os.name == "posix":  # macOS, Linux, *BSD, …
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

    # --------------------------------------------------
    # Exit: restore previous stderr, close the temp copy
    # --------------------------------------------------
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
            raise RuntimeError("Genesis hasn't been initialized. Did you call `gs.init()`?")
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
    # Note: we don't set seed for taichi, since taichi doesn't support stochastic operations in gradient computation.
    # Therefore, we only allow deterministic taichi operations.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_platform():
    name = platform.platform()
    # in python 3.8, platform.platform() uses mac_ver() on macOS
    # it will return 'macOS-XXXX' instead of 'Darwin-XXXX'
    if name.lower().startswith("darwin") or name.lower().startswith("macos"):
        return "macOS"

    if name.lower().startswith("windows"):
        return "Windows"

    if name.lower().startswith("linux"):
        return "Linux"

    if "bsd" in name.lower():
        return "Unix"

    assert False, f"Unknown platform name {name}"


def get_device(backend: gs_backend, device_idx: Optional[int] = None):
    if backend == gs_backend.cpu:
        device_name = cpuinfo.get_cpu_info()["brand_raw"]
        total_mem = psutil.virtual_memory().total / 1024**3
        device = torch.device("cpu", device_idx)
    elif backend == gs_backend.cuda:
        if not torch.cuda.is_available():
            gs.raise_exception("torch cuda not available")
        device = torch.device("cuda", device_idx)
        device_property = torch.cuda.get_device_properties(device)
        device_name = device_property.name
        total_mem = device_property.total_memory / 1024**3
    elif backend == gs_backend.metal:
        if not torch.backends.mps.is_available():
            gs.raise_exception("metal device not available")
        # on mac, cpu and gpu are in the same physical hardware and sharing memory
        _, device_name, total_mem, _ = get_device(gs_backend.cpu)
        device = torch.device("mps", device_idx)
    elif backend == gs_backend.vulkan:
        if torch.cuda.is_available():
            device, device_name, total_mem, _ = get_device(gs_backend.cuda)
        elif torch.xpu.is_available():  # pytorch 2.5+ supports Intel XPU device
            if device_idx is None:
                device_idx = torch.xpu.current_device()
            device = torch.device("xpu", device_idx)
            device_property = torch.xpu.get_device_properties(device_idx)
            device_name = device_property.name
            total_mem = device_property.total_memory / 1024**3
        else:  # pytorch tensors on cpu
            # logger may not be configured at this point
            logger = getattr(gs, "logger", None) or LOGGER
            logger.warning("No Intel XPU device available. Falling back to CPU for torch device.")
            device, device_name, total_mem, _ = get_device(gs_backend.cpu)
    else:  # backend == gs_backend.gpu:
        if torch.cuda.is_available():
            return get_device(gs_backend.cuda)
        elif get_platform() == "macOS":
            return get_device(gs_backend.metal)
        else:
            return get_device(gs_backend.vulkan)

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
    return os.path.join(os.path.expanduser("~"), ".cache", "genesis")


def get_gsd_cache_dir():
    return os.path.join(get_cache_dir(), "gsd")


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


def clean_cache_files():
    folder = gs.utils.misc.get_cache_dir()
    try:
        shutil.rmtree(folder)
    except:
        pass
    os.makedirs(folder)


def assert_gs_tensor(x):
    if not isinstance(x, gs.Tensor):
        gs.raise_exception("Only accepts genesis.Tensor.")


def to_gs_tensor(x):
    if isinstance(x, gs.Tensor):
        return x

    elif isinstance(x, list):
        return gs.from_numpy(np.array(x))

    elif isinstance(x, np.ndarray):
        return gs.from_numpy(x)

    elif isinstance(x, torch.Tensor):
        return gs.Tensor(x)

    else:
        gs.raise_exception("Only accepts genesis.Tensor, torch.Tensor, np.ndarray or List.")


def tensor_to_cpu(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
    return x


def tensor_to_array(x: torch.Tensor, dtype: Type[np.generic] | None = None) -> np.ndarray:
    return np.asarray(tensor_to_cpu(x), dtype=dtype)


def is_approx_multiple(a, b, tol=1e-7):
    return abs(a % b) < tol or abs(b - (a % b)) < tol


def concat_with_tensor(tensor: torch.Tensor, value, expand: tuple[int, ...] | None = None, dim: int = 0):
    """Helper method to concatenate a value (not necessarily a tensor) with a tensor."""
    if not isinstance(value, torch.Tensor):
        value = torch.tensor([value], dtype=tensor.dtype, device=tensor.device)
    if expand is not None:
        value = value.expand(*expand)
    if dim < 0:
        dim = tensor.ndim + dim
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


# -------------------------------------- TAICHI SPECIALIZATION --------------------------------------

ALLOCATE_TENSOR_WARNING = (
    "Tensor had to be re-allocated because of incorrect dtype/device or non-contiguous memory. This may "
    "impede performance if it occurs in the critical path of your application."
)

TI_PROG_WEAKREF: weakref.ReferenceType | None = None
TI_DATA_CACHE: OrderedDict[int, "FieldMetadata"] = OrderedDict()
MAX_CACHE_SIZE = 1000


@dataclass
class FieldMetadata:
    ndim: int
    shape: tuple[int, ...]
    dtype: ti._lib.core.DataTypeCxx
    mapping_key: Any


def _ensure_compiled(self, *args):
    # Note that the field is enough to determine the key because all the other arguments depends on it.
    # This may not be the case anymore if the output is no longer dynamically allocated at some point.
    ti_data_meta = TI_DATA_CACHE[id(args[0])]
    key = ti_data_meta.mapping_key
    if key is None:
        extracted = []
        for arg, kernel_arg in zip(args, self.mapper.arguments):
            anno = kernel_arg.annotation
            if is_ti_template(anno):
                subkey = arg
            else:  # isinstance(annotation, (ti.types.ndarray_type.NdarrayType, torch.Tensor, np.ndarray))
                needs_grad = getattr(arg, "requires_grad", False) if anno.needs_grad is None else anno.needs_grad
                subkey = (arg.dtype, len(arg.shape), needs_grad, anno.boundary)
            extracted.append(subkey)
        key = tuple(extracted)
        ti_data_meta.mapping_key = key
    instance_id = self.mapper.mapping.get(key)
    if instance_id is None:
        key = ti.lang.kernel_impl.Kernel.ensure_compiled(self, *args)
    else:
        key = (self.func, instance_id, self.autodiff_mode)
    return key


def _launch_kernel(self, t_kernel, compiled_kernel_data, *args):
    launch_ctx = t_kernel.make_launch_context()

    template_num = 0
    for i, v in enumerate(args):
        needed = self.arg_metas[i].annotation

        # template
        if is_ti_template(needed):
            template_num += 1
            continue

        # ti.ndarray
        if isinstance(v, ti.Ndarray):
            v_primal = v.arr
            v_grad = v.grad.arr if v.grad else None
            if v_grad is None:
                launch_ctx.set_arg_ndarray((i - template_num,), v_primal)
            else:
                launch_ctx.set_arg_ndarray_with_grad((i - template_num,), v_primal, v_grad)
            continue

        # ti.field
        array_shape = v.shape
        if needed.dtype is None or id(needed.dtype) in primitive_types.type_ids:
            element_dim = 0
        else:
            is_soa = needed.layout == ti.Layout.SOA
            element_dim = needed.dtype.ndim
            array_shape = v.shape[element_dim:] if is_soa else v.shape[:-element_dim]

        if v.requires_grad and v.grad is None:
            v.grad = torch.zeros_like(v)
        if v.requires_grad:
            if not isinstance(v.grad, torch.Tensor):
                raise ValueError(
                    f"Expecting torch.Tensor for gradient tensor, but getting {v.grad.__class__.__name__} instead"
                )
            if not v.grad.is_contiguous():
                raise ValueError(
                    "Non contiguous gradient tensors are not supported, please call tensor.grad.contiguous() "
                    "before passing it into taichi kernel."
                )

        launch_ctx.set_arg_external_array_with_shape(
            (i - template_num,),
            int(v.data_ptr()),
            v.element_size() * v.nelement(),
            array_shape,
            int(v.grad.data_ptr()) if v.grad is not None else 0,
        )

    try:
        prog = impl.get_runtime().prog
        if compiled_kernel_data is None:
            compile_result = prog.compile_kernel(prog.config(), prog.get_device_caps(), t_kernel)
            compiled_kernel_data = compile_result.compiled_kernel_data
        prog.launch_kernel(compiled_kernel_data, launch_ctx)
    except Exception as e:
        e = handle_exception_from_cpp(e)
        if impl.get_runtime().print_full_traceback:
            raise e
        raise e from None


def _destroy_callback(ref: weakref.ReferenceType):
    global TI_PROG_WEAKREF
    TI_DATA_CACHE.clear()
    for kernel in TO_EXT_ARR_FAST_MAP.values():
        kernel._primal.mapper.mapping.clear()
    TI_PROG_WEAKREF = None


_to_pytorch_type_fast = functools.lru_cache(maxsize=None)(to_pytorch_type)
TO_EXT_ARR_FAST_MAP = {}
for data_type, func in (
    (ti.ScalarField, tensor_to_ext_arr),
    (ti.MatrixField, matrix_to_ext_arr),
    (ti.ScalarNdarray, ndarray_to_ext_arr),
    (ti.MatrixNdarray, ndarray_matrix_to_ext_arr),
):
    func = ti.kernel(func._primal.func)
    func._primal.launch_kernel = types.MethodType(_launch_kernel, func._primal)
    func._primal.ensure_compiled = types.MethodType(_ensure_compiled, func._primal)
    TO_EXT_ARR_FAST_MAP[data_type] = func


def ti_to_torch(
    value,
    row_mask: slice | int | range | list | torch.Tensor | np.ndarray | None = None,
    col_mask: slice | int | range | list | torch.Tensor | np.ndarray | None = None,
    keepdim=True,
    transpose=False,
    *,
    unsafe=False,
) -> torch.Tensor:
    """Converts a GsTaichi field or ndarray instance to a PyTorch tensor.

    Args:
        value (ti.Field | ti.Ndarray): Field or Ndarray to convert to Pytorch tensor.
        row_mask (optional): Rows to extract from batch dimension after transpose if requested.
        col_mask (optional): Columns to extract from batch dimension after transpose if requested.
        keepdim (bool, optional): Whether to keep all dimensions even if masks are integers.
        transpose (bool, optional): Whether move to front the first non-batch dimension.
        unsafe (bool, optional): Whether to skip validity check of the masks.

    Returns:
        torch.tensor: The result torch tensor.
    """
    global TI_PROG_WEAKREF

    # Keep track of taichi runtime to automatically clear cache if destroyed
    if TI_PROG_WEAKREF is None:
        TI_PROG_WEAKREF = weakref.ref(impl.get_runtime().prog, _destroy_callback)

    # Get metadata
    ti_data_id = id(value)
    ti_data_meta = TI_DATA_CACHE.get(ti_data_id)
    if ti_data_meta is None:
        if isinstance(value, ti.MatrixField):
            ndim = value.ndim
        elif isinstance(value, ti.Ndarray):
            ndim = len(value.element_shape)
        else:
            ndim = 0
        ti_data_meta = FieldMetadata(ndim, value.shape, value.dtype, None)
        if len(TI_DATA_CACHE) == MAX_CACHE_SIZE:
            TI_DATA_CACHE.popitem(last=False)
        TI_DATA_CACHE[ti_data_id] = ti_data_meta

    # Make sure that the user-arguments are valid if requested
    ti_data_shape = ti_data_meta.shape
    is_1D_batch = len(ti_data_shape) == 1
    if not unsafe:
        _ti_data_shape = ti_data_shape[::-1] if transpose else ti_data_shape
        if is_1D_batch:
            if transpose and row_mask is not None:
                gs.raise_exception("Cannot specify row mask for fields with 1D batch and `transpose=True`.")
            elif not transpose and col_mask is not None:
                gs.raise_exception("Cannot specify column mask for fields with 1D batch and `transpose=False`.")
        for i, mask in enumerate((col_mask if transpose else row_mask,) if is_1D_batch else (row_mask, col_mask)):
            if mask is None or isinstance(mask, slice):
                # Slices are always valid by default. Nothing to check.
                is_out_of_bounds = False
            elif isinstance(mask, (int, np.integer)):
                # Do not allow negative indexing for consistency with Taichi
                is_out_of_bounds = not (0 <= mask < _ti_data_shape[i])
            elif isinstance(mask, torch.Tensor):
                if not mask.ndim <= 1:
                    gs.raise_exception(f"Expecting 1D tensor for masks.")
                # Resort on post-mortem analysis for bounds check because runtime would be to costly
                is_out_of_bounds = None
            else:  # np.ndarray, list, tuple, range
                try:
                    mask_start, mask_end = min(mask), max(mask)
                except ValueError:
                    gs.raise_exception(f"Expecting 1D tensor for masks.")
                is_out_of_bounds = not (0 <= mask_start <= mask_end < _ti_data_shape[i])
            if is_out_of_bounds:
                gs.raise_exception("Masks are out-of-range.")

    # Must convert masks to torch if not slice or int since torch will do it anyway.
    # Note that being contiguous is not required and does not affect performance.
    must_allocate = False
    is_row_mask_tensor = not (row_mask is None or isinstance(row_mask, (slice, int, np.integer)))
    if is_row_mask_tensor:
        _row_mask = torch.as_tensor(row_mask, dtype=gs.tc_int, device=gs.device)
        must_allocate = _row_mask is not row_mask
        row_mask = _row_mask
    is_col_mask_tensor = not (col_mask is None or isinstance(col_mask, (slice, int, np.integer)))
    if is_col_mask_tensor:
        _col_mask = torch.as_tensor(col_mask, dtype=gs.tc_int, device=gs.device)
        must_allocate = _col_mask is not col_mask
        col_mask = _col_mask
    if must_allocate:
        gs.logger.debug(ALLOCATE_TENSOR_WARNING)

    # Extract value as a whole.
    # Note that this is usually much faster than using a custom kernel to extract a slice.
    # The implementation is based on `taichi.lang.(ScalarField | MatrixField).to_torch`.
    is_metal = gs.device.type == "mps"
    tc_dtype = _to_pytorch_type_fast(ti_data_meta.dtype)
    data_type = type(value)
    if issubclass(data_type, (ti.ScalarField, ti.ScalarNdarray)):
        out = torch.zeros(size=ti_data_shape, dtype=tc_dtype, device="cpu" if is_metal else gs.device)
        TO_EXT_ARR_FAST_MAP[data_type](value, out)
    elif issubclass(data_type, ti.MatrixField):
        as_vector = value.m == 1
        shape_ext = (value.n,) if as_vector else (value.n, value.m)
        out = torch.empty(ti_data_shape + shape_ext, dtype=tc_dtype, device="cpu" if is_metal else gs.device)
        TO_EXT_ARR_FAST_MAP[data_type](value, out, as_vector)
    elif issubclass(data_type, (ti.VectorNdarray, ti.MatrixNdarray)):
        layout_is_aos = 1
        as_vector = issubclass(data_type, ti.VectorNdarray)
        shape_ext = (value.n,) if as_vector else (value.n, value.m)
        out = torch.empty(ti_data_shape + shape_ext, dtype=tc_dtype, device="cpu" if is_metal else gs.device)
        TO_EXT_ARR_FAST_MAP[ti.MatrixNdarray](value, out, layout_is_aos, as_vector)
    else:
        gs.raise_exception(f"Unsupported type '{type(value)}'.")
    if is_metal:
        out = out.to(gs.device)
    ti.sync()

    # Transpose if necessary and requested.
    # Note that it is worth transposing here rather than outside this function, as it preserve row-major memory
    # alignment in case of advanced masking, which would spare computation later on if expected from the user.
    if transpose and not is_1D_batch:
        out = out.movedim(out.ndim - ti_data_meta.ndim - 1, 0)

    # Extract slice if necessary.
    # Note that unsqueeze is MUCH faster than indexing with `[row_mask]` to keep batch dimensions,
    # because this required allocating GPU data.
    is_single_col = (is_col_mask_tensor and col_mask.ndim == 0) or isinstance(col_mask, (int, np.integer))
    is_single_row = (is_row_mask_tensor and row_mask.ndim == 0) or isinstance(row_mask, (int, np.integer))
    try:
        if is_col_mask_tensor and is_row_mask_tensor:
            if not is_single_col and not is_single_row:
                out = out[row_mask.unsqueeze(1), col_mask]
            else:
                out = out[row_mask, col_mask]
        else:
            if col_mask is not None:
                out = out[col_mask] if is_1D_batch else out[:, col_mask]
            if row_mask is not None:
                out = out[row_mask]
    except IndexError as e:
        if not unsafe and is_out_of_bounds is None:
            for i, mask in enumerate((col_mask if transpose else row_mask,) if is_1D_batch else (row_mask, col_mask)):
                # Do bounds analysis at this point because it skipped
                if not (0 <= mask[0] <= mask[-1] < ti_data_shape[i]):
                    gs.raise_exception_from("Masks are out-of-range.", e)

    # Make sure that masks are 1D if all dimensions must be kept
    if keepdim:
        if is_single_row:
            out = out.unsqueeze(0)
        if is_single_col:
            out = out.unsqueeze(0 if is_1D_batch else 1)

    return out
