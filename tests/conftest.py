import xml.etree.ElementTree as ET
from enum import Enum

import pytest
import mujoco
import genesis as gs

from .utils import MjSim


def pytest_make_parametrize_id(config, val, argname):
    if isinstance(val, Enum):
        return val.name
    return f"{val}"


def pytest_addoption(parser):
    parser.addoption("--vis", action="store_true", default=False, help="Enable interactive viewer.")


@pytest.fixture(scope="session")
def show_viewer(pytestconfig):
    return pytestconfig.getoption("--vis")


@pytest.fixture
def backend(request):
    if hasattr(request, "param"):
        return getattr(gs.constants.backend, "cpu")
    return gs.cpu


@pytest.fixture(scope="function", autouse=True)
def initialize_genesis(request, backend):
    logging_level = request.config.getoption("--log-cli-level")
    gs.init(backend=backend, precision="64", logging_level=logging_level, seed=0, debug=True)
    yield
    gs.destroy()


@pytest.fixture
def mj_sim(xml_path, gs_solver, gs_integrator):
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

    mj_sim.model = mujoco.MjModel.from_xml_path(xml_path)
    mj_sim.model.opt.solver = mj_solver
    mj_sim.model.opt.integrator = mj_integrator
    mj_sim.model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    mj_sim.data = mujoco.MjData(mj_sim.model)

    return MjSim(mj_sim.model, mj_sim.data)


@pytest.fixture
def gs_sim(xml_path, gs_solver, gs_integrator, show_viewer, mj_sim):
    gs_sim.scene = gs.Scene(
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
            box_box_detection=True,
            enable_self_collision=True,
            enable_adjacent_collision=True,
            contact_resolve_time=0.02,
            iterations=mj_sim.model.opt.iterations,
            tolerance=mj_sim.model.opt.tolerance,
            ls_iterations=mj_sim.model.opt.ls_iterations,
            ls_tolerance=mj_sim.model.opt.ls_tolerance,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    gs_robot = gs_sim.scene.add_entity(
        gs.morphs.MJCF(file=xml_path),
        visualize_contact=False,
    )
    gs_sim.scene.build()

    return gs_sim.scene.sim


@pytest.fixture
def xml_path(request, tmp_path, model_name):
    mjcf = request.getfixturevalue(model_name)
    xml_tree = ET.ElementTree(mjcf)
    file_path = tmp_path / f"{model_name}.xml"
    xml_tree.write(file_path, encoding="utf-8", xml_declaration=True)
    return str(file_path)


@pytest.fixture(scope="session")
def box_plan():
    """Generate an XML model for a box on a plane."""
    mjcf = ET.Element("mujoco", model="one_box")
    ET.SubElement(mjcf, "option", timestep="0.01")
    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "geom", contype="1", conaffinity="1", condim="3", friction="1. 0.5 0.5")
    worldbody = ET.SubElement(mjcf, "worldbody")
    ET.SubElement(worldbody, "geom", type="plane", name="floor", pos="0. 0. 0.", size="40. 40. 40.")
    box_body = ET.SubElement(worldbody, "body", pos="0. 0. 0.3")
    ET.SubElement(box_body, "geom", type="box", size="0.2 0.2 0.2", pos="0. 0. 0.")
    ET.SubElement(box_body, "joint", type="free")
    return mjcf
