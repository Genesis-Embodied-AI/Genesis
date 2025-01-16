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

import genesis as gs
from genesis.constants import backend as gs_backend


def raise_exception(msg="Something went wrong."):
    gs.logger._error_msg = msg
    raise gs.GenesisException(msg)


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
