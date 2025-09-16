import base64
import ctypes
import gc
import os
import re
import sys
from enum import Enum
from io import BytesIO
from pathlib import Path

import psutil
import pyglet
import pytest
from _pytest.mark import Expression, MarkMatcher
from PIL import Image
from syrupy.extensions.image import PNGImageSnapshotExtension
from syrupy.location import PyTestLocation

has_display = True
try:
    from tkinter import Tk

    root = Tk()
    root.destroy()
except Exception:  # ImportError, TclError
    # Mock tkinter module for backward compatibility because it is a hard dependency for old Genesis versions
    tkinter = type(sys)("tkinter")
    tkinter.Tk = type(sys)("Tk")
    tkinter.filedialog = type(sys)("filedialog")
    sys.modules["tkinter"] = tkinter
    sys.modules["tkinter.Tk"] = tkinter.Tk
    sys.modules["tkinter.filedialog"] = tkinter.filedialog

    # Assuming headless server if tkinder is not installed
    has_display = False

has_egl = True
try:
    pyglet.lib.load_library("EGL")
except ImportError:
    has_egl = False

if not has_display and has_egl:
    # It is necessary to configure pyglet in headless mode if necessary before importing Genesis
    pyglet.options["headless"] = True
    os.environ["GS_VIEWER_ALLOW_OFFSCREEN"] = "1"

IS_INTERACTIVE_VIEWER_AVAILABLE = has_display or has_egl

TOL_SINGLE = 5e-5
TOL_DOUBLE = 1e-9
IMG_STD_ERR_THR = 1.0
IMG_NUM_ERR_THR = 0.001


def pytest_make_parametrize_id(config, val, argname):
    if isinstance(val, Enum):
        return val.name
    if isinstance(val, type):
        return ".".join((val.__module__, val.__name__))
    return f"{val}"


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config: pytest.Config) -> None:
    # Force disabling forked for non-linux systems
    if not sys.platform.startswith("linux"):
        config.option.forked = False

    # Make sure that benchmarks are running on GPU and the number of workers if valid
    expr = Expression.compile(config.option.markexpr)
    is_benchmarks = expr.evaluate(MarkMatcher.from_markers((pytest.mark.benchmarks,)))
    if is_benchmarks:
        # Make sure that GPU backend is enforced
        backend = config.getoption("--backend")
        if backend == "cpu":
            raise ValueError("Running benchmarks on CPU is not supported.")
        config.option.backend = "gpu"

        # Make sure that the number of workers is not too large
        if isinstance(config.option.numprocesses, int):
            max_workers = max(pytest_xdist_auto_num_workers(config), 1)
            if config.option.numprocesses > max_workers:
                raise ValueError(
                    f"The number of workers for running benchmarks cannot exceed '{max_workers}' on this machine."
                )

    # Force disabling distributed framework if interactive viewer is enabled
    show_viewer = config.getoption("--vis")
    if show_viewer:
        config.option.numprocesses = 0

    # Disable low-level parallelization if distributed framework is enabled.
    # FIXME: It should be set to `max(int(physical_core_count / num_workers), 1)`, but 'num_workers' may be unknown.
    if not is_benchmarks and config.option.numprocesses != 0:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ["NUMBA_NUM_THREADS"] = "1"


def _get_gpu_indices():
    nvidia_gpu_indices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if nvidia_gpu_indices is not None:
        return tuple(sorted(map(int, nvidia_gpu_indices.split(","))))

    if sys.platform == "linux":
        nvidia_gpu_indices = []
        nvidia_gpu_interface_path = "/proc/driver/nvidia/gpus/"
        if os.path.exists(nvidia_gpu_interface_path):
            for device_path in os.listdir(nvidia_gpu_interface_path):
                with open(os.path.join(nvidia_gpu_interface_path, device_path, "information"), "r") as f:
                    gpu_id = int(re.search(r"Device Minor:\s+(\d+)", f.read()).group(1))
                nvidia_gpu_indices.append(gpu_id)
            return tuple(sorted(nvidia_gpu_indices))

    return (0,)


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
    import genesis as gs

    # Get available memory (RAM & VRAM) and number of cores
    physical_core_count = psutil.cpu_count(logical=config.option.logical)
    _, _, ram_memory, _ = gs.utils.get_device(gs.cpu)
    _, _, vram_memory, backend = gs.utils.get_device(gs.gpu)
    num_gpus = len(_get_gpu_indices())
    vram_memory *= num_gpus
    if backend == gs.cpu:
        # Ignore VRAM if no GPU is available
        vram_memory = float("inf")

    # Compute the default number of workers based on available RAM, VRAM, and number of physical cores.
    # Note that if `forked` is not enabled, up to 7.5Gb per worker is necessary on Linux because Taichi
    # does not completely release memory between each test.
    if sys.platform in ("darwin", "win32"):
        ram_memory_per_worker = 3.0
        vram_memory_per_worker = 1.0  # Does not really makes sense on Apple Silicon
    elif config.option.forked:
        ram_memory_per_worker = 5.5
        vram_memory_per_worker = 1.2
    else:
        ram_memory_per_worker = 7.5
        vram_memory_per_worker = 1.6
    num_workers = min(
        physical_core_count,
        max(int(ram_memory / ram_memory_per_worker), 1),
        max(int(vram_memory / vram_memory_per_worker), 1),
    )

    # Special treatment for benchmarks
    expr = Expression.compile(config.option.markexpr)
    is_benchmarks = expr.evaluate(MarkMatcher.from_markers((pytest.mark.benchmarks,)))
    if is_benchmarks:
        num_cpu_per_gpu = 4
        num_workers = min(
            num_workers,
            num_gpus,
            max(int(physical_core_count / num_cpu_per_gpu), 1),
        )

    return num_workers


def pytest_runtest_setup(item):
    # Enforce GPU affinity if distributed framework is enabled
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id and worker_id.startswith("gw"):
        worker_num = int(worker_id[2:])
        gpu_indices = _get_gpu_indices()
        gpu_index = gpu_indices[worker_num % len(gpu_indices)]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        os.environ["TI_VISIBLE_DEVICE"] = str(gpu_index)
        try:
            os.environ["EGL_DEVICE_ID"] = str(_get_egl_index(gpu_index))
        except Exception:
            pass


def pytest_addoption(parser):
    parser.addoption("--backend", action="store", default=None, help="Default simulation backend.")
    parser.addoption(
        "--logical", action="store_true", default=False, help="Consider logical cores in default number of workers."
    )
    parser.addoption("--vis", action="store_true", default=False, help="Enable interactive viewer.")
    parser.addoption("--dev", action="store_true", default=False, help="Enable genesis debug mode.")


@pytest.fixture(scope="session")
def show_viewer(pytestconfig):
    return pytestconfig.getoption("--vis") and IS_INTERACTIVE_VIEWER_AVAILABLE


@pytest.fixture(scope="session")
def backend(pytestconfig):
    import genesis as gs

    backend = pytestconfig.getoption("--backend") or gs.cpu
    if isinstance(backend, str):
        return getattr(gs.constants.backend, backend)
    return backend


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


@pytest.fixture(scope="function", autouse=True)
def initialize_genesis(request, monkeypatch, backend, precision, taichi_offline_cache):
    import genesis as gs

    # Early return if backend is None
    if backend is None:
        yield
        return

    logging_level = request.config.getoption("--log-cli-level")
    debug = request.config.getoption("--dev")

    try:
        if not taichi_offline_cache:
            monkeypatch.setenv("TI_OFFLINE_CACHE", "0")

        # Skip test if gstaichi ndarray mode is enabled but not supported by this specific test
        if os.environ.get("GS_USE_NDARRAY") == "1":
            for mark in request.node.iter_markers("field_only"):
                if not mark.args or mark.args[0]:
                    pytest.skip(f"This test does not support GsTaichi ndarray mode. Skipping...")
            if os.environ.get("GS_BETA_PURE") == "1" and backend != gs.cpu and sys.platform == "darwin":
                pytest.skip("fast cache not supported on mac gpus when using ndarray.")

        try:
            gs.utils.get_device(backend)
        except gs.GenesisException:
            pytest.skip(f"Backend '{backend}' not available on this machine")
        gs.init(backend=backend, precision=precision, debug=debug, seed=0, logging_level=logging_level)
        gc.collect()

        if gs.backend != gs.cpu:
            device_index = gs.device.index
            if device_index is not None and device_index not in _get_gpu_indices():
                assert RuntimeError("Wrong CUDA GPU device.")

        import gstaichi as ti

        ti_runtime = ti.lang.impl.get_runtime()
        ti_config = ti.lang.impl.current_cfg()
        if ti_config.arch == ti.metal and precision == "64":
            pytest.skip("Apple Metal GPU does not support 64bits precision.")

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

    filename = str(asset_tmp_path / f"fixture_box_obj_path.obj")
    with open(filename, "w", encoding="utf-8") as f:
        for x, y, z in verts:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        f.write("\n")
        for a, b, c, d in faces:
            f.write(f"f {a} {b} {c} {d}\n")

    return filename


class PixelMatchSnapshotExtension(PNGImageSnapshotExtension):
    _std_err_threshold: float = IMG_STD_ERR_THR
    _ratio_err_threshold: float = IMG_NUM_ERR_THR

    def matches(self, *, serialized_data, snapshot_data) -> bool:
        import numpy as np

        img_arrays = []
        for data in (serialized_data, snapshot_data):
            buffer = BytesIO()
            buffer.write(data)
            buffer.seek(0)
            img_arrays.append(np.atleast_3d(np.asarray(Image.open(buffer))).astype(np.int32))
        img_delta = np.minimum(np.abs(np.diff(img_arrays, axis=0)), 255).astype(np.uint8)
        if (
            np.max(np.std(img_delta.reshape((-1, img_delta.shape[-1])), axis=0)) > self._std_err_threshold
            and (np.abs(img_delta) > np.finfo(np.float32).eps).sum() > self._ratio_err_threshold * img_delta.size
        ):
            raw_bytes = BytesIO()
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

    must_update_snapshop = request.config.getoption("--snapshot-update")
    if must_update_snapshop:
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
