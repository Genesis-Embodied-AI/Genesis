import gc
import os
import sys
from enum import Enum

import psutil
import pyglet
import numpy as np
import pytest
from _pytest.mark import Expression, MarkMatcher

import mujoco
import genesis as gs
from genesis.utils.mesh import get_assets_dir

from .utils import MjSim


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
        ram_memory_per_worker = 4.5
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
def atol():
    return TOL_DOUBLE if gs.np_float == np.float64 else TOL_SINGLE


@pytest.fixture(scope="function", autouse=True)
def initialize_genesis(request, backend):
    logging_level = request.config.getoption("--log-cli-level")
    if backend == gs.cpu:
        precision = "64"
        debug = True
    else:
        precision = "32"
        debug = False
    try:
        gs.init(backend=backend, precision=precision, debug=debug, seed=0, logging_level=logging_level)
        if backend != gs.cpu and gs.backend == gs.cpu:
            gs.destroy()
            pytest.skip("No GPU available on this machine")
        yield
    finally:
        pyglet.app.exit()
        gs.destroy()
        gc.collect()


@pytest.fixture
def mpr_vanilla(request):
    mpr_vanilla = None
    for mark in request.node.iter_markers("mpr_vanilla"):
        if mark.args:
            if mpr_vanilla is not None:
                pytest.fail("'mpr_vanilla' can only be specified once.")
            (mpr_vanilla,) = mark.args
    if mpr_vanilla is None:
        mpr_vanilla = True
    return mpr_vanilla


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
def mj_sim(xml_path, gs_solver, gs_integrator, multi_contact, adjacent_collision, dof_damping):
    if gs_solver == gs.constraint_solver.CG:
        mj_solver = mujoco.mjtSolver.mjSOL_CG
    elif gs_solver == gs.constraint_solver.Newton:
        mj_solver = mujoco.mjtSolver.mjSOL_NEWTON
    else:
        raise ValueError(f"Solver '{gs_solver}' not supported")
    if gs_integrator == gs.integrator.Euler:
        mj_integrator = mujoco.mjtIntegrator.mjINT_EULER
    elif gs_integrator == gs.integrator.implicitfast:
        mj_integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    else:
        raise ValueError(f"Integrator '{gs_integrator}' not supported")

    if not os.path.isabs(xml_path):
        xml_path = os.path.join(get_assets_dir(), xml_path)

    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.solver = mj_solver
    model.opt.integrator = mj_integrator
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    model.opt.disableflags &= ~np.uint32(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
    model.opt.disableflags &= ~np.uint32(mujoco.mjtDisableBit.mjDSBL_REFSAFE)
    model.opt.disableflags &= ~np.uint32(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_NATIVECCD
    if multi_contact:
        model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD
    else:
        model.opt.enableflags &= ~np.uint32(mujoco.mjtEnableBit.mjENBL_MULTICCD)
    if adjacent_collision:
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_FILTERPARENT
    else:
        model.opt.disableflags &= ~np.uint32(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
    data = mujoco.MjData(model)

    # Joint damping is not properly supported in Genesis for now
    if not dof_damping:
        model.dof_damping[:] = 0.0

    return MjSim(model, data)


@pytest.fixture
def gs_sim(
    xml_path, gs_solver, gs_integrator, multi_contact, mpr_vanilla, adjacent_collision, dof_damping, show_viewer, mj_sim
):
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            res=(960, 640),
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(
            dt=mj_sim.model.opt.timestep,
            substeps=1,
            gravity=mj_sim.model.opt.gravity.tolist(),
        ),
        rigid_options=gs.options.RigidOptions(
            integrator=gs_integrator,
            constraint_solver=gs_solver,
            enable_mpr_vanilla=mpr_vanilla,
            box_box_detection=True,
            enable_self_collision=True,
            enable_adjacent_collision=adjacent_collision,
            enable_multi_contact=multi_contact,
            iterations=mj_sim.model.opt.iterations,
            tolerance=mj_sim.model.opt.tolerance,
            ls_iterations=mj_sim.model.opt.ls_iterations,
            ls_tolerance=mj_sim.model.opt.ls_tolerance,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    gs_robot = scene.add_entity(
        gs.morphs.MJCF(file=xml_path),
        visualize_contact=True,
    )
    gs_sim = scene.sim

    # Force matching Mujoco safety factor for constraint time constant.
    # Note that this time constant affects the penetration depth at rest.
    gs_sim.rigid_solver._sol_constraint_min_resolve_time = 2.0 * gs_sim._substep_dt

    # Joint damping is not properly supported in Genesis for now
    if not dof_damping:
        for joint in gs_robot.joints:
            joint.dofs_damping[:] = 0.0

    scene.build()

    return gs_sim


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
