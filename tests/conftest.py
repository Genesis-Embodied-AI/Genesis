import gc
import os
import re
import sys
from enum import Enum

import pytest
from _pytest.mark import Expression, MarkMatcher

# Mock tkinter module for backward compatibility because old Genesis versions require it
try:
    import tkinter
except ImportError:
    tkinter = type(sys)("tkinter")
    tkinter.Tk = type(sys)("Tk")
    tkinter.filedialog = type(sys)("filedialog")
    sys.modules["tkinter"] = tkinter
    sys.modules["tkinter.Tk"] = tkinter.Tk
    sys.modules["tkinter.filedialog"] = tkinter.filedialog


TOL_SINGLE = 5e-5
TOL_DOUBLE = 1e-9


def pytest_make_parametrize_id(config, val, argname):
    if isinstance(val, Enum):
        return val.name
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


def pytest_xdist_auto_num_workers(config):
    import psutil
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
    # Enforce GPU affinity that distributed framework is enabled
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id and worker_id.startswith("gw"):
        worker_num = int(worker_id[2:])
        gpu_indices = _get_gpu_indices()
        gpu_num = worker_num % len(gpu_indices)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_indices[gpu_num])
        os.environ["TI_VISIBLE_DEVICE"] = str(gpu_indices[gpu_num])


def pytest_addoption(parser):
    parser.addoption("--backend", action="store", default=None, help="Default simulation backend.")
    parser.addoption(
        "--logical", action="store_true", default=False, help="Consider logical cores in default number of workers."
    )
    parser.addoption("--vis", action="store_true", default=False, help="Enable interactive viewer.")
    parser.addoption("--dev", action="store_true", default=False, help="Enable genesis debug mode.")


@pytest.fixture(scope="session")
def show_viewer(pytestconfig):
    return pytestconfig.getoption("--vis")


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
def initialize_genesis(request, backend, precision, taichi_offline_cache):
    import pyglet
    import gstaichi as ti
    import genesis as gs
    from genesis.utils.misc import ALLOCATE_TENSOR_WARNING

    logging_level = request.config.getoption("--log-cli-level")
    debug = request.config.getoption("--dev")

    try:
        if not taichi_offline_cache:
            os.environ["TI_OFFLINE_CACHE"] = "0"

        try:
            gs.utils.get_device(backend)
        except gs.GenesisException:
            pytest.skip(f"Backend '{backend}' not available on this machine")
        gs.init(backend=backend, precision=precision, debug=debug, seed=0, logging_level=logging_level)

        ti_runtime = ti.lang.impl.get_runtime()
        ti_arch = ti_runtime.prog.config().arch
        if ti_arch == ti.metal and precision == "64":
            gs.destroy()
            pytest.skip("Apple Metal GPU does not support 64bits precision.")

        if backend != gs.cpu and gs.backend == gs.cpu:
            gs.destroy()
            pytest.skip("No GPU available on this machine")
        yield
    finally:
        pyglet.app.exit()
        gs.destroy()
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
