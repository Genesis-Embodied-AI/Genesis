import gc
import os
import sys
from enum import Enum

import psutil
import pyglet
import numpy as np
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

import genesis as gs
from genesis.utils.misc import ALLOCATE_TENSOR_WARNING

from .utils import MjSim, build_mujoco_sim, build_genesis_sim


TOL_SINGLE = 5e-5
TOL_DOUBLE = 1e-9


def pytest_make_parametrize_id(config, val, argname):
    if isinstance(val, Enum):
        return val.name
    return f"{val}"


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config: pytest.Config) -> None:
    # Force disabling distributed framework if benchmarks are selected
    expr = Expression.compile(config.option.markexpr)
    is_benchmarks = expr.evaluate(MarkMatcher.from_markers((pytest.mark.benchmarks,)))
    if is_benchmarks:
        config.option.numprocesses = 0

    # Force disabling forked for non-linux systems
    if not sys.platform.startswith("linux"):
        config.option.forked = False

    # Force disabling distributed framework if interactive viewer is enabled
    show_viewer = config.getoption("--vis")
    if show_viewer:
        config.option.numprocesses = 0


def pytest_xdist_auto_num_workers(config):
    # Get available memory (RAM & VRAM) and number of physical cores.
    physical_core_count = psutil.cpu_count(logical=False)
    _, _, ram_memory, _ = gs.utils.get_device(gs.cpu)
    _, _, vram_memory, _ = gs.utils.get_device(gs.gpu)

    # Compute the default number of workers based on available RAM, VRAM, and number of physical cores.
    # Note that if `forked` is not enabled, up to 7.5Gb per worker is necessary on Linux because Taichi
    # does not completely release memory between each test.
    if sys.platform == "darwin":
        ram_memory_per_worker = 3.0
        vram_memory_per_worker = 1.0  # Does not really makes sense on Apple Silicon
    elif config.option.forked:
        ram_memory_per_worker = 5.5
        vram_memory_per_worker = 1.2
    else:
        ram_memory_per_worker = 7.5
        vram_memory_per_worker = 1.6
    return min(
        int(ram_memory / ram_memory_per_worker),
        int(vram_memory / vram_memory_per_worker),
        physical_core_count,
    )


def pytest_addoption(parser):
    parser.addoption("--backend", action="store", default=gs.cpu, help="Default simulation backend.")
    parser.addoption("--vis", action="store_true", default=False, help="Enable interactive viewer.")


@pytest.fixture(scope="session")
def show_viewer(pytestconfig):
    return pytestconfig.getoption("--vis")


@pytest.fixture(scope="session")
def backend(pytestconfig):
    backend = pytestconfig.getoption("--backend", "cpu")
    if isinstance(backend, str):
        return getattr(gs.constants.backend, backend)
    return backend


@pytest.fixture(scope="session")
def asset_tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("assets")


@pytest.fixture
def tol():
    return TOL_DOUBLE if gs.np_float == np.float64 else TOL_SINGLE


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
def initialize_genesis(request, backend, taichi_offline_cache):
    logging_level = request.config.getoption("--log-cli-level")
    if backend == gs.cpu:
        precision = "64"
        debug = True
    else:
        precision = "32"
        debug = False
    try:
        if not taichi_offline_cache:
            os.environ["TI_OFFLINE_CACHE"] = "0"
        gs.init(backend=backend, precision=precision, debug=debug, seed=0, logging_level=logging_level)
        gs.logger.addFilter(lambda record: ALLOCATE_TENSOR_WARNING not in record.getMessage())
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
