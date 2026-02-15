import base64
import ctypes
import gc
import logging
import os
import re
import subprocess
from argparse import SUPPRESS
import sys
from enum import Enum
from io import BytesIO
from pathlib import Path

import numpy as np
import setproctitle
import psutil
import pyglet
import pytest
from _pytest.mark import Expression, MarkMatcher
from PIL import Image
from syrupy.extensions.image import PNGImageSnapshotExtension

# Mock tkinter module for backward compatibility because it is a hard dependency for old Genesis versions
has_tkinter = False
try:
    import tkinter

    has_tkinter = True
except ImportError:
    tkinter = type(sys)("tkinter")
    tkinter.Tk = type(sys)("Tk")
    tkinter.filedialog = type(sys)("filedialog")
    sys.modules["tkinter"] = tkinter
    sys.modules["tkinter.Tk"] = tkinter.Tk
    sys.modules["tkinter.filedialog"] = tkinter.filedialog

# Determine whether a screen is available
if has_tkinter:
    has_display = True
    try:
        root = tkinter.Tk()
        root.withdraw()
        root.destroy()
    except tkinter.TclError:
        has_display = False
else:
    # Assuming headless server if tkinter is not installed unless DISPLAY env var is available on Linux
    if sys.platform.startswith("linux"):
        has_display = bool(os.environ.get("DISPLAY"))
    else:
        has_display = False

# Determine whether EGL driver is available
has_egl = True
try:
    pyglet.lib.load_library("EGL")
except ImportError:
    has_egl = False

# Forcibly disable Mujoco OpenGL to avoid conflicts with Genesis
os.environ["MUJOCO_GL"] = "0"

# Forcibly disable tqdm to avoid random crashes on the MacOS CI
os.environ["TQDM_DISABLE"] = "1"

# pyglet must be configured in headless mode before importing Genesis if necessary.
# Note that environment variables are used instead of global options to ease option propagation to subprocesses.
if not has_display and has_egl:
    pyglet.options["headless"] = True
    os.environ["PYGLET_HEADLESS"] = "1"

IS_INTERACTIVE_VIEWER_AVAILABLE = has_display or has_egl

TOL_SINGLE = 5e-5
TOL_DOUBLE = 1e-9
IMG_STD_ERR_THR = 1.0
IMG_NUM_ERR_THR = 0.001
IMG_BLUR_KERNEL_SIZE = 1  # Size of the blur kernel (must be odd)


def is_mem_monitoring_supported():
    try:
        assert sys.platform.startswith("linux")
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, timeout=10)
        return True, None
    except Exception as exc:  # platform or nvidia-smi unavailable
        return False, exc


def pytest_make_parametrize_id(config, val, argname):
    if isinstance(val, Enum):
        return val.name
    if isinstance(val, type):
        return ".".join((val.__module__, val.__name__))
    return f"{val}"


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config: pytest.Config) -> None:
    # Make sure that no unsupported markers have been specified in CLI
    declared_markers = set(name for spec in config.getini("markers") if (name := spec.split(":")[0]) != "forked")
    try:
        eval(config.option.markexpr, {"__builtins__": {}}, {key: None for key in declared_markers})
    except NameError as e:
        raise pytest.UsageError(f"Unknown marker in CLI expression: '{e.name}'")

    # Only launch memory monitor from the main process, not from xdist workers
    mem_filepath = config.getoption("--mem-monitoring-filepath")
    if mem_filepath and not os.environ.get("PYTEST_XDIST_WORKER"):
        supported, reason = is_mem_monitoring_supported()
        if not supported:
            raise pytest.UsageError(f"--mem-monitoring-filepath is not supported on this platform: {reason}")
        subprocess.Popen(
            [
                sys.executable,
                "tests/monitor_test_mem.py",
                "--die-with-parent",
                "--out-file",
                mem_filepath,
            ]
        )

    # Make sure that benchmarks are running on GPU and the number of workers if valid
    expr = Expression.compile(config.option.markexpr)
    is_benchmarks = expr.evaluate(MarkMatcher.from_markers((pytest.mark.benchmarks,)))
    if is_benchmarks:
        # Make sure that GPU backend is enforced
        backend = config.getoption("--backend")
        if backend == "cpu":
            raise ValueError("Running benchmarks on CPU is not supported.")
        config.option.backend = "gpu"

    # Force disabling forked for non-linux systems
    if not sys.platform.startswith("linux"):
        config.option.forked = False

    # Force disabling distributed framework if interactive viewer is enabled
    show_viewer = config.getoption("--vis", IS_INTERACTIVE_VIEWER_AVAILABLE)
    if show_viewer:
        config.option.numprocesses = 0

    # Force disabling reruns if debugger is enabled
    is_pdb_enabled = config.getoption("--pdb")
    if is_pdb_enabled:
        config.option.reruns = 0

    # Force headless rendering if available and the interactive viewer is disabled.
    # FIXME: It breaks rendering on some platform...
    # if not show_viewer and has_egl:
    #     pyglet.options["headless"] = True

    # Make sure that the number of workers is not too large if specified
    if isinstance(config.option.numprocesses, int):
        max_workers = max(pytest_xdist_auto_num_workers(config), 1)
        if config.option.numprocesses > max_workers:
            raise ValueError(f"The number of workers cannot exceed '{max_workers}' on this machine.")

    # Properly configure Taichi std out stream right away to avoid significant performance penalty (~10%)
    # Note that this variable must be set in the main thread BEFORE spawning the distributed workers, otherwise
    # the variable will be set incorrectly. Although, Genesis is already setting this env variable properly at import,
    # relying on this mechanism is fragile.
    os.environ.setdefault("TI_ENABLE_PYBUF", "0" if sys.stdout is sys.__stdout__ else "1")

    # Disable Quadrants dynamic array mode by default on MacOS because it is not supported by Metal
    if sys.platform == "darwin":
        os.environ.setdefault("GS_ENABLE_NDARRAY", "0")

    # Enforce special environment variable before importing test modules if distributed framework is enabled
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id and worker_id.startswith("gw"):
        # Enforce GPU affinity
        gpu_indices = _get_gpu_indices()
        if gpu_indices:
            worker_num = int(worker_id[2:])
            gpu_index = gpu_indices[worker_num % len(gpu_indices)]
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            os.environ["TI_VISIBLE_DEVICE"] = str(gpu_index)

        # Limit CPU threading
        if is_benchmarks:
            # FIXME: Enabling multi-threading in benchmark is making compile time estimation unreliable
            num_cpu_per_worker = "1"
        else:
            physical_core_count = psutil.cpu_count(logical=config.option.logical)
            num_workers = int(os.environ["PYTEST_XDIST_WORKER_COUNT"])
            num_cpu_per_worker = str(max(int(physical_core_count / num_workers), 1))
        os.environ["TI_NUM_THREADS"] = num_cpu_per_worker
        os.environ["OMP_NUM_THREADS"] = num_cpu_per_worker
        os.environ["OPENBLAS_NUM_THREADS"] = num_cpu_per_worker
        os.environ["MKL_NUM_THREADS"] = num_cpu_per_worker
        os.environ["VECLIB_MAXIMUM_THREADS"] = num_cpu_per_worker
        os.environ["NUMEXPR_NUM_THREADS"] = num_cpu_per_worker
        os.environ["NUMBA_NUM_THREADS"] = num_cpu_per_worker


def _get_gpu_indices():
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None:
        return tuple(map(int, cuda_visible_devices.split(",")))

    if sys.platform == "linux":
        nvidia_gpu_indices = []
        nvidia_gpu_interface_path = "/proc/driver/nvidia/gpus/"
        if os.path.exists(nvidia_gpu_interface_path):
            return tuple(range(len(os.listdir(nvidia_gpu_interface_path))))

    return (0,)


def _torch_get_gpu_idx(device):
    if sys.platform == "darwin":
        return 0

    if sys.platform == "linux":
        import torch

        device_property = torch.cuda.get_device_properties(device)
        device_uuid = str(device_property.uuid)

        nvidia_gpu_interface_path = "/proc/driver/nvidia/gpus/"
        for device_idx, device_path in enumerate(os.listdir(nvidia_gpu_interface_path)):
            with open(os.path.join(nvidia_gpu_interface_path, device_path, "information"), "r") as f:
                device_info = f.read()
            if re.search(rf"GPU UUID:\s+GPU-{device_uuid}", device_info):
                return device_idx

    return -1


def _get_egl_index(gpu_index):
    from OpenGL import EGL
    from OpenGL.EGL.NV.device_cuda import EGL_CUDA_DEVICE_NV

    # Get the list of Nvidia GPU that are visible
    nvidia_gpu_indices = _get_gpu_indices()

    # Define some ctypes for convenience
    EGLDeviceEXT = ctypes.c_void_p
    EGLAttrib = ctypes.c_ssize_t
    EGLint = ctypes.c_int
    EGLuint = ctypes.c_uint

    # Load EGL extension functions dynamically
    EGLuint = ctypes.c_uint
    eglQueryDevicesEXT_addr = EGL.eglGetProcAddress(b"eglQueryDevicesEXT")
    if not eglQueryDevicesEXT_addr:
        raise RuntimeError("eglQueryDevicesEXT not available")
    eglQueryDevicesEXT = ctypes.CFUNCTYPE(EGLuint, EGLint, ctypes.POINTER(EGLDeviceEXT), ctypes.POINTER(EGLint))(
        eglQueryDevicesEXT_addr
    )
    eglQueryDeviceAttribEXT_addr = EGL.eglGetProcAddress(b"eglQueryDeviceAttribEXT")
    if not eglQueryDeviceAttribEXT_addr:
        raise RuntimeError("eglQueryDeviceAttribEXT not available")
    eglQueryDeviceAttribEXT = ctypes.CFUNCTYPE(EGLuint, EGLDeviceEXT, EGLint, ctypes.POINTER(EGLAttrib))(
        eglQueryDeviceAttribEXT_addr
    )

    # Query EGL devices
    num_devices = EGLint()
    eglQueryDevicesEXT(0, None, ctypes.byref(num_devices))
    devices = (EGLDeviceEXT * num_devices.value)()
    eglQueryDevicesEXT(num_devices, devices, ctypes.byref(num_devices))
    egl_map = {}
    for i in range(num_devices.value):
        dev = devices[i]
        cuda_id = EGLAttrib()
        if eglQueryDeviceAttribEXT(dev, EGL_CUDA_DEVICE_NV, ctypes.byref(cuda_id)):
            egl_map[nvidia_gpu_indices[cuda_id.value]] = i

    return egl_map[gpu_index]


def pytest_xdist_auto_num_workers(config):
    # Get available memory (RAM & VRAM) and number of cores
    physical_core_count = psutil.cpu_count(logical=config.option.logical)
    ram_memory = psutil.virtual_memory().total / 1024**3
    if sys.platform == "darwin":
        # On Apple ARM, cpu and gpu are part of the same physical device with unified memory
        num_gpus = 1
        vram_memory = ram_memory
    else:
        # Cannot rely on 'torch' because this would force loading devices before configuring CUDA device visibility
        devices_vram_memory = None
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
            )
            devices_vram_memory = tuple(int(e.strip()) for e in result.stdout.splitlines())
        except (FileNotFoundError, subprocess.CalledProcessError):
            try:
                result = subprocess.run(
                    ["rocm-smi", "--showmeminfo", "vram", "-d", "0-255"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                    text=True,
                )
                devices_vram_memory = tuple(
                    int(m.group(1)) for m in re.finditer(r"VRAM Total:\s+(\d+)\s*MiB", result.stdout)
                )
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass
        if devices_vram_memory is not None:
            assert len(set(devices_vram_memory)) == 1, "Heterogeonous Nvidia GPU devices not supported."
            num_gpus = len(devices_vram_memory)
            vram_memory = sum(devices_vram_memory) / 1024
        else:
            # FIXME: There is easy way for Intel ARC device. Ignore device visibilty issue for now...
            import torch

            if torch.xpu.is_available():
                num_gpus = torch.xpu.device_count()
                vram_memory = 0.0
                for device_idx in range(num_gpus):
                    device = torch.device("xpu", device_idx)
                    device_property = torch.cuda.get_device_properties(device)
                    vram_memory += device_property.total_memory / 1024**3
            else:
                # Ignore VRAM if no GPU is available
                num_gpus = 0
                vram_memory = float("inf")

    # Compute the default number of workers based on available RAM, VRAM, and number of physical cores.
    # Note that if `forked` is not enabled, up to 7.5Gb per worker is necessary on Linux because Taichi
    # does not completely release memory between each test.
    if sys.platform == "darwin":
        ram_memory_per_worker = vram_memory_per_worker = 3.0
    elif config.option.forked:
        ram_memory_per_worker = 5.5
        vram_memory_per_worker = 1.8
    else:
        ram_memory_per_worker = 7.5
        vram_memory_per_worker = 2.5
    num_workers = min(
        physical_core_count,
        max(ram_memory / ram_memory_per_worker, 1),
        max(vram_memory / vram_memory_per_worker, 1),
    )

    # Special treatment for benchmarks
    expr = Expression.compile(config.option.markexpr)
    is_benchmarks = expr.evaluate(MarkMatcher.from_markers((pytest.mark.benchmarks,)))
    if is_benchmarks:
        num_cpu_per_gpu = 8
        num_workers = min(
            num_workers,
            num_gpus,
            max(physical_core_count / num_cpu_per_gpu, 1),
        )

    return int(num_workers)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    # Run slow tests first

    slow = [item for item in items if "slow" in item.keywords]
    fast = [item for item in items if "slow" not in item.keywords]

    max_workers = config.option.numprocesses
    if max_workers is None:
        max_workers = int(os.environ["PYTEST_XDIST_WORKER_COUNT"])
    max_workers = max(max_workers, 1)

    buckets = [[] for _ in range(max_workers)]
    for idx, item in enumerate(slow + fast):
        bucket_idx = idx % max_workers
        buckets[bucket_idx].append(item)
    items[:] = [item for bucket in sorted(buckets, key=len) for item in bucket]


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    # Include test name in process title
    test_name = item.nodeid.replace(" ", "")
    dtype = "ndarray" if os.environ.get("GS_ENABLE_NDARRAY") == "1" else "field"
    test_name = test_name[:-1] + f"-{dtype}]"

    setproctitle.setproctitle(f"pytest: {test_name}")

    # Match CUDA device with EGL device.
    # Note that this must be done here instead of 'pytest_cmdline_main', otherwise it will segfault when using
    # 'pytest-forked', because EGL instances are not allowed to cross thread boundaries.
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id and worker_id.startswith("gw"):
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is not None:
            gpu_index = int(cuda_visible_devices)
            if has_egl:
                try:
                    os.environ["EGL_DEVICE_ID"] = str(_get_egl_index(gpu_index))
                except (AttributeError, KeyError):
                    # AttributeError: CUDA is not supported on this machine
                    # KeyError: The selected GPU does not support CUDA
                    pass


def pytest_addoption(parser):
    parser.addoption("--backend", action="store", default=None, help="Default simulation backend.")
    parser.addoption(
        "--logical", action="store_true", default=False, help="Consider logical cores in default number of workers."
    )
    if IS_INTERACTIVE_VIEWER_AVAILABLE:
        parser.addoption("--vis", action="store_true", default=False, help="Enable interactive viewer.")
    parser.addoption("--dev", action="store_true", default=False, help="Enable genesis debug mode.")
    supported, _reason = is_mem_monitoring_supported()
    help_text = (
        "Run memory monitoring, and store results to mem_monitoring_filepath. CUDA on linux ONLY."
        if supported
        else SUPPRESS
    )
    parser.addoption("--mem-monitoring-filepath", type=str, help=help_text)


@pytest.fixture(scope="session")
def show_viewer(pytestconfig):
    return pytestconfig.getoption("--vis", IS_INTERACTIVE_VIEWER_AVAILABLE)


@pytest.fixture(scope="session")
def backend(pytestconfig):
    return pytestconfig.getoption("--backend") or "cpu"


@pytest.fixture(scope="session")
def asset_tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("assets")


@pytest.fixture
def tol():
    import numpy as np
    import genesis as gs

    return TOL_DOUBLE if gs.np_float == np.float64 else TOL_SINGLE


@pytest.fixture
def precision(request, backend):
    import genesis as gs

    precision = None
    for mark in request.node.iter_markers("precision"):
        if mark.args:
            if precision is not None:
                pytest.fail("'precision' can only be specified once.")
            (precision,) = mark.args
    if precision is None:
        precision = "64" if backend == gs.cpu else "32"
    return precision


@pytest.fixture
def mujoco_compatibility(request):
    mujoco_compatibility = None
    for mark in request.node.iter_markers("mujoco_compatibility"):
        if mark.args:
            if mujoco_compatibility is not None:
                pytest.fail("'mujoco_compatibility' can only be specified once.")
            (mujoco_compatibility,) = mark.args
    if mujoco_compatibility is None:
        mujoco_compatibility = True
    return mujoco_compatibility


@pytest.fixture
def adjacent_collision(request):
    adjacent_collision = None
    for mark in request.node.iter_markers("adjacent_collision"):
        if mark.args:
            if adjacent_collision is not None:
                pytest.fail("'adjacent_collision' can only be specified once.")
            (adjacent_collision,) = mark.args
    if adjacent_collision is None:
        adjacent_collision = False
    return adjacent_collision


@pytest.fixture
def gjk_collision(request):
    gjk_collision = None
    for mark in request.node.iter_markers("gjk_collision"):
        if mark.args:
            if gjk_collision is not None:
                pytest.fail("'gjk_collision' can only be specified once.")
            (gjk_collision,) = mark.args
    if gjk_collision is None:
        gjk_collision = False
    return gjk_collision


@pytest.fixture
def merge_fixed_links(request):
    merge_fixed_links = None
    for mark in request.node.iter_markers("merge_fixed_links"):
        if mark.args:
            if merge_fixed_links is not None:
                pytest.fail("'merge_fixed_links' can only be specified once.")
            (merge_fixed_links,) = mark.args
    if merge_fixed_links is None:
        merge_fixed_links = True
    return merge_fixed_links


@pytest.fixture
def multi_contact(request):
    multi_contact = None
    for mark in request.node.iter_markers("multi_contact"):
        if mark.args:
            if multi_contact is not None:
                pytest.fail("'multi_contact' can only be specified once.")
            (multi_contact,) = mark.args
    if multi_contact is None:
        multi_contact = True
    return multi_contact


@pytest.fixture
def dof_damping(request):
    dof_damping = None
    for mark in request.node.iter_markers("dof_damping"):
        if mark.args:
            if dof_damping is not None:
                pytest.fail("'dof_damping' can only be specified once.")
            (dof_damping,) = mark.args
    if dof_damping is None:
        dof_damping = False
    return dof_damping


@pytest.fixture
def taichi_offline_cache(request):
    taichi_offline_cache = None
    for mark in request.node.iter_markers("taichi_offline_cache"):
        if mark.args:
            if taichi_offline_cache is not None:
                pytest.fail("'taichi_offline_cache' can only be specified once.")
            (taichi_offline_cache,) = mark.args
    if taichi_offline_cache is None:
        taichi_offline_cache = True
    return taichi_offline_cache


@pytest.fixture
def performance_mode(request):
    performance_mode = None
    for mark in request.node.iter_markers("performance_mode"):
        if mark.args:
            if performance_mode is not None:
                pytest.fail("'performance_mode' can only be specified once.")
            (performance_mode,) = mark.args
    if performance_mode is None:
        performance_mode = False
    return performance_mode


@pytest.fixture
def debug(request):
    debug = None
    for mark in request.node.iter_markers("debug"):
        if mark.args:
            if debug is not None:
                pytest.fail("'debug' can only be specified once.")
            (debug,) = mark.args
    return debug


@pytest.fixture(scope="function", autouse=True)
def initialize_genesis(
    request, monkeypatch, tmp_path, backend, precision, performance_mode, debug, taichi_offline_cache
):
    import genesis as gs

    # Early return if backend is None
    if backend is None:
        yield
        return

    # Convert backend from string to enum if necessary
    if isinstance(backend, str):
        backend = getattr(gs.constants.backend, backend)

    logging_level = request.config.getoption("--log-cli-level", logging.INFO)
    if debug is None:
        debug = request.config.getoption("--dev")

    if not taichi_offline_cache:
        monkeypatch.setenv("TI_OFFLINE_CACHE", "0")
        # FIXME: Must set temporary cache even if caching is forcibly disabled because this flag is not always honored
        monkeypatch.setenv("TI_OFFLINE_CACHE_FILE_PATH", str(tmp_path / ".cache" / "taichi"))
        monkeypatch.setenv("GS_CACHE_FILE_PATH", str(tmp_path / ".cache" / "genesis"))
        monkeypatch.setenv("GS_ENABLE_FASTCACHE", "0")

    # Redirect name terrain cache directory to some test-local temporary location to avoid conflict and persistence
    monkeypatch.setattr("genesis.utils.misc.get_gnd_cache_dir", lambda: str(tmp_path / ".cache" / "terrain"))

    try:
        # Skip if requested backend is not available
        try:
            gs.utils.get_device(backend)
        except gs.GenesisException:
            pytest.skip(f"Backend '{backend}' not available on this machine")

        # Skip test if not supported by this machine
        if sys.platform == "darwin" and backend != gs.cpu:
            if os.environ.get("TI_ENABLE_METAL", "1") != "0" and precision == "64":
                pytest.skip("Apple Metal GPU does not support 64bits precision.")
            if os.environ.get("GS_ENABLE_NDARRAY") == "1":
                pytest.skip(
                    "Using Quadrants dynamic array type is not supported on Apple Metal GPU because this backend only "
                    "supports up to 31 kernel parameters, which is not enough for most solvers."
                )

        gs.init(
            backend=backend,
            precision=precision,
            debug=debug,
            seed=0,
            logging_level=logging_level,
            performance_mode=performance_mode,
        )
        gc.collect()

        if gs.backend != gs.cpu and gs.device.index is not None:
            if _torch_get_gpu_idx(gs.device.index) not in _get_gpu_indices():
                raise RuntimeError(f"Invalid CUDA GPU device, got {gs.device.index}, expected {_get_gpu_indices()}.")

        if backend != gs.cpu and gs.backend == gs.cpu:
            pytest.skip("No GPU available on this machine")

        yield
    finally:
        gs.destroy()
        # Double garbage collection is over-zealous since gstaichi 2.2.1 but let's do it anyway
        gc.collect()
        gc.collect()


@pytest.fixture
def mj_sim(
    xml_path, gs_solver, gs_integrator, merge_fixed_links, multi_contact, adjacent_collision, dof_damping, gjk_collision
):
    from .utils import build_mujoco_sim

    return build_mujoco_sim(
        xml_path,
        gs_solver,
        gs_integrator,
        merge_fixed_links,
        multi_contact,
        adjacent_collision,
        dof_damping,
        gjk_collision,
    )


@pytest.fixture
def gs_sim(
    xml_path,
    gs_solver,
    gs_integrator,
    merge_fixed_links,
    multi_contact,
    mujoco_compatibility,
    adjacent_collision,
    gjk_collision,
    show_viewer,
    mj_sim,
):
    from .utils import build_genesis_sim

    return build_genesis_sim(
        xml_path,
        gs_solver,
        gs_integrator,
        merge_fixed_links,
        multi_contact,
        mujoco_compatibility,
        adjacent_collision,
        gjk_collision,
        show_viewer,
        mj_sim,
    )


@pytest.fixture(scope="session")
def cube_verts_and_faces():
    cx, cy, cz = (0.0, 0.0, 0.0)
    edge_length = 1.0

    h = edge_length / 2.0

    verts = [
        (cx - h, cy - h, cz - h),  # v0
        (cx + h, cy - h, cz - h),  # v1
        (cx + h, cy + h, cz - h),  # v2
        (cx - h, cy + h, cz - h),  # v3
        (cx - h, cy - h, cz + h),  # v4
        (cx + h, cy - h, cz + h),  # v5
        (cx + h, cy + h, cz + h),  # v6
        (cx - h, cy + h, cz + h),  # v7
    ]

    faces = [
        (1, 2, 3, 4),
        (5, 6, 7, 8),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 4, 8, 7),
        (4, 1, 5, 8),
    ]
    return verts, faces


@pytest.fixture(scope="session")
def box_obj_path(asset_tmp_path, cube_verts_and_faces):
    """Fixture that generates a temporary cube .obj file"""
    verts, faces = cube_verts_and_faces

    filename = str(asset_tmp_path / "fixture_box_obj_path.obj")
    with open(filename, "w", encoding="utf-8") as f:
        for x, y, z in verts:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        f.write("\n")
        for a, b, c, d in faces:
            f.write(f"f {a} {b} {c} {d}\n")

    return filename


def _apply_blur(img_arr: np.ndarray, kernel_size: int) -> np.ndarray:
    # Early return if nothing to do:
    if kernel_size == 1:
        return img_arr

    # Create normalized box kernel
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size**2)

    pad_size = kernel_size // 2
    h, w = img_arr.shape[:2]

    # Pad the image
    if img_arr.ndim == 2:
        padded = np.pad(img_arr, pad_size, mode="edge")
    else:
        padded = np.pad(img_arr, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode="edge")

    # Apply convolution
    blurred_arr = np.zeros_like(img_arr, dtype=np.float32)
    if img_arr.ndim == 2:
        for i in range(h):
            for j in range(w):
                blurred_arr[i, j] = np.sum(padded[i : i + kernel_size, j : j + kernel_size] * kernel)
    else:
        for c in range(img_arr.shape[-1]):
            for i in range(h):
                for j in range(w):
                    blurred_arr[i, j, c] = np.sum(padded[i : i + kernel_size, j : j + kernel_size, c] * kernel)

    return blurred_arr


class PixelMatchSnapshotExtension(PNGImageSnapshotExtension):
    _std_err_threshold: float = IMG_STD_ERR_THR
    _ratio_err_threshold: float = IMG_NUM_ERR_THR
    _blurred_kernel_size: int = IMG_BLUR_KERNEL_SIZE

    def matches(self, *, serialized_data, snapshot_data) -> bool:
        img_arrays, blurred_arrays = [], []
        for data in (serialized_data, snapshot_data):
            buffer = BytesIO()
            buffer.write(data)
            buffer.seek(0)
            img_array = np.atleast_3d(np.asarray(Image.open(buffer))).astype(np.float32)
            blurred_array = _apply_blur(img_array, self._blurred_kernel_size)
            img_arrays.append(img_array)
            blurred_arrays.append(blurred_array)

        if img_arrays[0].shape != img_arrays[1].shape:
            return False

        # Compute difference on blurred images
        img_err = np.minimum(np.abs(blurred_arrays[1] - blurred_arrays[0]), 255).astype(np.uint8)

        if (
            np.max(np.std(img_err.reshape((-1, img_err.shape[-1])), axis=0)) > self._std_err_threshold
            and (np.abs(img_err) > np.finfo(np.float32).eps).sum() > self._ratio_err_threshold * img_err.size
        ):
            raw_bytes = BytesIO()
            img_delta = np.minimum(np.abs(img_arrays[1] - img_arrays[0]), 255).astype(np.uint8)
            img_obj = Image.fromarray(img_delta.squeeze(-1) if img_delta.shape[-1] == 1 else img_delta)
            img_obj.save(raw_bytes, "PNG")
            raw_bytes.seek(0)
            print(base64.b64encode(raw_bytes.read()))
            return False
        return True


@pytest.fixture
def png_snapshot(request, snapshot):
    snapshot_obj = snapshot.use_extension(PixelMatchSnapshotExtension)
    snapshot_dir = Path(PixelMatchSnapshotExtension.dirname(test_location=snapshot_obj.test_location))
    snapshot_name = PixelMatchSnapshotExtension.get_snapshot_name(test_location=snapshot_obj.test_location)

    must_update_snapshot = request.config.getoption("--snapshot-update")
    if must_update_snapshot:
        for path in (Path(snapshot_dir.parent) / snapshot_dir.name).glob(f"{snapshot_name}*"):
            assert path.is_file()
            path.unlink()
    else:
        from .utils import get_hf_dataset

        snapshot_name_ = "".join(f"[{char}]" if char in ("[", "]") else char for char in snapshot_name)
        get_hf_dataset(
            pattern=f"{snapshot_dir.name}/{snapshot_name_}*", repo_name="snapshots", local_dir=snapshot_dir.parent
        )

    return snapshot_obj
