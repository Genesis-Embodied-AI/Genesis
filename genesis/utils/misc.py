import datetime
import functools
import os
import platform
import random
import shutil
import subprocess

import numpy as np
import psutil
import torch
from taichi.lang import runtime_ops
from taichi._kernels import matrix_to_ext_arr

import genesis as gs
from genesis.constants import backend as gs_backend


class DeprecationError(Exception):
    pass


def raise_exception(msg="Something went wrong."):
    gs.logger._error_msg = msg
    raise gs.GenesisException(msg)


def raise_exception_from(msg="Something went wrong.", cause=None):
    gs.logger._error_msg = msg
    raise gs.GenesisException(msg) from cause


def assert_initialized(cls):
    original_init = cls.__init__

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
            gs.raise_exception("Scene is not built yet.")
        return method(self, *args, **kwargs)

    return wrapper


def set_random_seed(seed):
    # Note: we don't set seed for taichi, since taichi doesn't support stochastic operations in gradient computation. Therefore, we only allow deterministic taichi operations.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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


def get_cpu_name():
    if get_platform() == "macOS":
        os.environ["PATH"] = os.environ["PATH"] + os.pathsep + "/usr/sbin"
        command = "sysctl -n machdep.cpu.brand_string"
        process = subprocess.run(command, shell=True, capture_output=True, text=True)
        return process.stdout.strip()

    elif get_platform() == "Linux":
        command = "cat /proc/cpuinfo"
        process = subprocess.run(command, shell=True, capture_output=True, text=True)
        all_info = process.stdout.strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return line.replace("\t", "").replace("model name: ", "")

    else:
        return platform.processor()


def get_device(backend: gs_backend):
    if backend == gs_backend.cuda:
        if not torch.cuda.is_available():
            gs.raise_exception("cuda device not available")

        device = torch.device("cuda")
        device_property = torch.cuda.get_device_properties(0)
        device_name = device_property.name
        total_mem = device_property.total_memory / 1024**3

    elif backend == gs_backend.metal:
        if not torch.backends.mps.is_available():
            gs.raise_exception("metal device not available")

        # on mac, cpu and gpu are in the same device
        _, device_name, total_mem, _ = get_device(gs_backend.cpu)
        device = torch.device("mps")

    elif backend == gs_backend.vulkan:
        if torch.xpu.is_available():  # pytorch 2.5+ Intel XPU device
            device = torch.device("xpu")
            device_property = torch.xpu.get_device_properties(0)
            device_name = device_property.name
            total_mem = device_property.total_memory / 1024**3
        else:  # pytorch tensors on cpu
            gs.logger.warning("Vulkan support only available on Intel XPU device. Falling back to CPU.")
            device, device_name, total_mem, _ = get_device(gs_backend.cpu)

    elif backend == gs_backend.gpu:
        if torch.cuda.is_available():
            return get_device(gs_backend.cuda)
        elif get_platform() == "macOS":
            return get_device(gs_backend.metal)
        else:
            return get_device(gs_backend.vulkan)

    else:
        device_name = get_cpu_name()
        total_mem = psutil.virtual_memory().total / 1024**3
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
        x = x.cpu()
    return x


def tensor_to_array(x):
    return np.array(tensor_to_cpu(x))


def is_approx_multiple(a, b, tol=1e-7):
    return abs(a % b) < tol or abs(b - (a % b)) < tol


def ti_mat_field_to_torch(
    field,
    row_mask: slice | int | range | list | torch.Tensor | np.ndarray | None = None,
    col_mask: slice | int | range | list | torch.Tensor | np.ndarray | None = None,
    transpose=False,
    *,
    unsafe=False,
) -> torch.Tensor:
    """Converts a Taichi matrix field instance to a PyTorch tensor.

    Args:
        field (ti.Matrix): Matrix field to convert to Pytorch tensor.
        row_mask (optional): Rows to extract from batch dimension after transpose if requested.
        col_mask (optional): Columns to extract from batch dimension field after transpose if requested.
        transpose (bool, optional): Whether to transpose the first two batch dimensions.
        unsafe (bool, optional): Whether to skip validity check of the masks.

    Returns:
        torch.tensor: The result torch tensor.
    """

    # Make sure that the user-arguments are valid if requested
    # FIXME: Add clear error messages instead of raise/assert.
    if not unsafe:
        field_shape = field.shape
        if transpose:
            field_shape = field_shape[::-1]
        for i, mask in enumerate((row_mask, col_mask)):
            if mask is None or isinstance(mask, slice):
                # Slices are always valid by default. Nothing to check.
                pass
            elif isinstance(mask, int):
                # Do not allow negative indexing for consistency with Taichi
                assert 0 <= mask < field_shape[i]
            else:
                mask_start, mask_end = mask[0], mask[-1]
                # ValueError if not convertible to int, as a validation check
                mask_start, mask_end = int(mask_start), int(mask_end)
                assert 0 <= mask_start <= mask_end < field_shape[i]

    # Must convert masks to torch if not slice or int since torch will do it anyway.
    # Note that being contiguous is not required and does not affect performance.
    row_is_tensor = not (row_mask is None or isinstance(row_mask, (slice, int)))
    if row_is_tensor:
        row_mask = torch.as_tensor(row_mask, dtype=gs.tc_int, device=gs.device)
    col_is_tensor = not (col_mask is None or isinstance(col_mask, (slice, int)))
    if col_is_tensor:
        col_mask = torch.as_tensor(col_mask, dtype=gs.tc_int, device=gs.device)

    # Extract field as a whole.
    # Note that this is usually much faster than using a custom kernel to extract a slice.
    tensor = field.to_torch(device=gs.device)

    # Transpose if requested.
    # Note that it is worth transposing here rather than outside this function, as it preserve row-major memory
    # alignment in case of advanced masking, which would spare computation later on if expected from the user.
    if transpose:
        tensor = tensor.transpose(1, 0)

    # Extract slice if necessary
    if col_is_tensor and row_is_tensor:
        tensor = tensor[row_mask.unsqueeze(1), col_mask]
    else:
        if col_mask is not None:
            tensor = tensor[:, col_mask]
        if row_mask is not None:
            tensor = tensor[row_mask]

    return tensor
