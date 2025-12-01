import ctypes
import datetime
import functools
import io
import logging
import math
import numbers
import os
import platform
import random
import sys
import types
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, NoReturn, Optional, Type, Sequence

import cpuinfo
import gstaichi as ti
import numpy as np
import psutil
import pyglet
import torch

from gstaichi.lang.util import is_ti_template, to_pytorch_type, to_numpy_type
from gstaichi._kernels import tensor_to_ext_arr, matrix_to_ext_arr, ndarray_to_ext_arr, ndarray_matrix_to_ext_arr
from gstaichi.lang import impl
from gstaichi.lang.exception import handle_exception_from_cpp
from gstaichi.types import primitive_types

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
        cpu_info = cpuinfo.get_cpu_info()
        device_name = next(filter(None, map(cpu_info.get, ("brand_raw", "hardware_raw", "vendor_id_raw"))))
        total_mem = psutil.virtual_memory().total / 1024**3
        device = torch.device("cpu", device_idx)
    elif backend == gs_backend.cuda:
        if not torch.cuda.is_available():
            gs.raise_exception("torch cuda not available")
        if device_idx is None:
            device_idx = torch.cuda.current_device()
        device = torch.device("cuda", device_idx)
        device_property = torch.cuda.get_device_properties(device)
        device_name = device_property.name
        total_mem = device_property.total_memory / 1024**3
    elif backend == gs_backend.metal:
        if not torch.backends.mps.is_available():
            gs.raise_exception("Torch metal backend not available.")
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
            logger.warning("Torch GPU backend not available. Falling back to CPU device.")
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
    root_cache_dir = None
    if get_platform() == "Linux":
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


# -------------------------------------- TAICHI SPECIALIZATION --------------------------------------

TI_PROG_WEAKREF: weakref.ReferenceType | None = None
TI_MAPPING_KEY_CACHE: OrderedDict[int, Any] = OrderedDict()
MAX_CACHE_SIZE = 1000


def _ensure_compiled(self, *args):
    # Note that the field is enough to determine the key because all the other arguments depends on it.
    # This may not be the case anymore if the output is no longer dynamically allocated at some point.
    cache_id = id(args[0])
    key = TI_MAPPING_KEY_CACHE.get(cache_id)
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
        TI_MAPPING_KEY_CACHE[cache_id] = key
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
                launch_ctx.set_arg_ndarray(i - template_num, v_primal)
            else:
                launch_ctx.set_arg_ndarray_with_grad(i - template_num, v_primal, v_grad)
            continue

        # ti.field
        array_shape = v.shape
        if needed.dtype is None or id(needed.dtype) in primitive_types.type_ids:
            element_dim = 0
        else:
            is_soa = needed.layout == ti.Layout.SOA
            element_dim = needed.dtype.ndim
            array_shape = v.shape[element_dim:] if is_soa else v.shape[:-element_dim]

        if isinstance(v, np.ndarray):  # numpy
            arr_ptr = int(v.ctypes.data)
            nbytes = v.nbytes
            grad_ptr = 0  # nullptr
        else:  # torch
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

            arr_ptr = int(v.data_ptr())
            nbytes = v.element_size() * v.nelement()
            grad_ptr = int(v.grad.data_ptr()) if v.grad is not None else 0

        launch_ctx.set_arg_external_array_with_shape(i - template_num, arr_ptr, nbytes, array_shape, grad_ptr)

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
    TI_MAPPING_KEY_CACHE.clear()
    for kernel in TO_EXT_ARR_FAST_MAP.values():
        kernel._primal.mapper.mapping.clear()
    TI_PROG_WEAKREF = None


_to_torch_type_fast = functools.lru_cache(maxsize=None)(to_pytorch_type)
_to_numpy_type_fast = functools.lru_cache(maxsize=None)(to_numpy_type)
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


def ti_to_python(
    value: ti.Field | ti.Ndarray,
    transpose: bool = False,
    copy: bool | None = None,
    to_torch: bool = True,
) -> torch.Tensor | np.ndarray:
    """Converts a GsTaichi field / ndarray instance to a PyTorch tensor / Numpy array.

    Args:
        value (ti.Field | ti.Ndarray): Field or Ndarray to be converted.
        transpose (bool, optional): Whether to move the last batch dimension in front. Defaults to False.
        copy (bool, optional): Wether to enforce returning a copy no matter what. None to avoid copy if possible
        without raising an exception if not.
        to_torch (bool): Whether to convert to Torch tensor or Numpy array. Defaults to True.
    """
    # Check if copy mode is supported while setting default mode if not specified.
    # FIXME: ti.Field does not support zero-copy on Metal for now because of a bug in Torch itself.
    # See: https://github.com/pytorch/pytorch/pull/168193
    # FIXME: Zero-copy is currently broken for ti.Field for some reason...
    data_type = type(value)
    use_zerocopy = (
        gs.use_zerocopy
        and (to_torch or gs.backend == gs.cpu)
        and not issubclass(data_type, ti.Field)
        # and (gs.backend != gs.metal or not issubclass(data_type, ti.Field))
    )
    if not use_zerocopy:
        if copy is False:
            gs.raise_exception(
                "Specifying 'copy=False' is not supported if 'gs.use_zerocopy=False' or ('to_torch=False' and "
                "'gs.backend != gs.cpu')."
            )
        copy = True
    elif copy is None:
        copy = False

    # Leverage zero-copy if enabled
    batch_shape = value.shape
    if use_zerocopy:
        try:
            if to_torch or gs.backend != gs.cpu:
                out = value._T_tc if transpose else value._tc
            else:
                out = value._T_np if transpose else value._np
        except AttributeError:
            value._tc = torch.utils.dlpack.from_dlpack(value.to_dlpack())
            value._T_tc = value._tc.movedim(batch_ndim - 1, 0) if (batch_ndim := len(batch_shape)) > 1 else value._tc
            if to_torch:
                out = value._T_tc if transpose else value._tc
            if gs.backend == gs.cpu:
                value._np = value._tc.numpy()
                value._T_np = value._T_tc.numpy()
                if not to_torch:
                    out = value._T_np if transpose else value._np
        if copy:
            if to_torch:
                out = out.clone()
            else:
                out = tensor_to_array(out)
        return out

    # Keep track of taichi runtime to automatically clear cache if destroyed
    global TI_PROG_WEAKREF
    if TI_PROG_WEAKREF is None:
        TI_PROG_WEAKREF = weakref.ref(impl.get_runtime().prog, _destroy_callback)

    # Extract value as a whole.
    # Note that this is usually much faster than using a custom kernel to extract a slice.
    # The implementation is based on `taichi.lang.(ScalarField | MatrixField).to_torch`.
    is_metal = gs.device.type == "mps"
    out_dtype = _to_torch_type_fast(value.dtype) if to_torch else _to_numpy_type_fast(value.dtype)
    if issubclass(data_type, (ti.ScalarField, ti.ScalarNdarray)):
        if to_torch:
            out = torch.zeros(batch_shape, dtype=out_dtype, device="cpu" if is_metal else gs.device)
        else:
            out = np.zeros(batch_shape, dtype=out_dtype)
        TO_EXT_ARR_FAST_MAP[data_type](value, out)
    elif issubclass(data_type, ti.MatrixField):
        as_vector = value.m == 1
        shape_ext = (value.n,) if as_vector else (value.n, value.m)
        if to_torch:
            out = torch.empty(batch_shape + shape_ext, dtype=out_dtype, device="cpu" if is_metal else gs.device)
        else:
            out = np.zeros(batch_shape + shape_ext, dtype=out_dtype)
        TO_EXT_ARR_FAST_MAP[data_type](value, out, as_vector)
    elif issubclass(data_type, (ti.VectorNdarray, ti.MatrixNdarray)):
        layout_is_aos = 1
        as_vector = issubclass(data_type, ti.VectorNdarray)
        shape_ext = (value.n,) if as_vector else (value.n, value.m)
        if to_torch:
            out = torch.empty(batch_shape + shape_ext, dtype=out_dtype, device="cpu" if is_metal else gs.device)
        else:
            out = np.zeros(batch_shape + shape_ext, dtype=out_dtype)
        TO_EXT_ARR_FAST_MAP[ti.MatrixNdarray](value, out, layout_is_aos, as_vector)
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
    *indices: Any, keepdim: bool = True, to_torch: bool = True, raise_if_fancy: bool = False
) -> tuple[slice | int | torch.Tensor, ...]:
    """Converts a sequence of slice-like objects into a multi-dimensional mask corresponding to their cross-product.

    Args:
        keepdim (bool): Whether to keep all dimensions even if masks are integers. Defaults to True.
        to_torch (bool): Whether to force casting collections to torch.Tensor.
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
                        is_scalar_ = arg.numel() == 1
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


def ti_to_torch(
    value: ti.Field | ti.Ndarray,
    row_mask: int | range | slice | tuple[int, ...] | list[int] | torch.Tensor | np.ndarray | None = None,
    col_mask: int | range | slice | tuple[int, ...] | list[int] | torch.Tensor | np.ndarray | None = None,
    keepdim: bool = True,
    transpose: bool = False,
    *,
    copy: bool | None = None,
) -> torch.Tensor:
    """Converts a GsTaichi field / ndarray instance to a PyTorch tensor.

    Args:
        value (ti.Field | ti.Ndarray): Field or Ndarray to be converted.
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
            if copy:
                tensor = tensor.clone()
        except AttributeError:
            tensor = ti_to_python(value, transpose, copy=copy, to_torch=True)
    else:
        tensor = ti_to_python(value, transpose, copy=copy, to_torch=True)

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


def ti_to_numpy(
    value: ti.Field | ti.Ndarray,
    row_mask: int | range | slice | tuple[int, ...] | list[int] | torch.Tensor | np.ndarray | None = None,
    col_mask: int | range | slice | tuple[int, ...] | list[int] | torch.Tensor | np.ndarray | None = None,
    keepdim: bool = True,
    transpose: bool = False,
    *,
    copy: bool | None = None,
) -> np.ndarray:
    """Converts a GsTaichi field / ndarray instance to a Numpy array.

    Args:
        value (ti.Field | ti.Ndarray): Field or Ndarray to be converted.
        row_mask (optional): Rows to extract from batch dimension after transpose if requested.
        col_mask (optional): Columns to extract from batch dimension after transpose if requested.
        keepdim (bool, optional): Whether to keep all dimensions even if masks are integers.
        transpose (bool, optional): Whether move to front the first non-batch dimension.
        copy (bool, optional): Wether to enforce returning a copy no matter what. None to avoid copy if possible
        without raising an exception if not.
    """
    tensor = ti_to_python(value, transpose, copy=copy, to_torch=False)
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
    tensor: np.typing.ArrayLike | None,
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
    elif 2 <= tensor_ndim < expected_ndim:
        # Try expanding first dimensions if priority
        for dims_valid in tuple(combinations(range(expected_ndim), tensor_ndim))[::-1]:
            curr_idx = 0
            expanded_shape = []
            for i in range(expected_ndim):
                if i in dims_valid:
                    dim, size = tensor_.shape[curr_idx], expected_shape[i]
                    if dim == size or dim == 1:
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
            dim, size = tensor_.shape[i], expected_shape[i]
            if size > 0 and i < tensor_.ndim and dim != 1 and dim != size:
                if name:
                    msg_infos.append(f"Dimension {i} consistent with len({name})(={size})")
                else:
                    msg_infos.append(f"Dimension {i} consistent with required size {size}")
        if msg_infos:
            msg_err += f" {' & '.join(msg_infos)}."
        gs.raise_exception_from(msg_err, e)

    return tensor_


def sanitize_indexed_tensor(
    tensor: np.typing.ArrayLike | None,
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
    value: np.typing.ArrayLike,
    dim_names: tuple[str, ...] | list[str] | None = None,
) -> None:
    try:
        tensor[indices] = value
    except (TypeError, RuntimeError):
        # Try extended broadcasting as a fallback to avoid slowing down the hot path
        indexed_shape = get_indexed_shape(tensor.shape, indices) if indices else tensor.shape
        tensor[indices] = broadcast_tensor(value, tensor.dtype, indexed_shape, dim_names)
