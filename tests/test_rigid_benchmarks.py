import os
import pytest

import numpy as np
import torch

import genesis as gs


N_FRAME_FPS = 10
REPORT_FILE = "speed_test.txt"


pytestmark = [pytest.mark.benchmarks]


@pytest.fixture(scope="session", autouse=True)
def setup_txt_logging():
    if os.path.exists(REPORT_FILE):
        os.remove(REPORT_FILE)


@pytest.fixture
def anymal_c(solver, n_envs):
    scene = gs.Scene(
        show_viewer=False,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            constraint_solver=solver,
        ),
    )

    ########################## entities ##########################
    scene.add_entity(
        gs.morphs.Plane(),
    )
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/anymal_c/urdf/anymal_c.urdf",
            pos=(0, 0, 0.8),
        ),
    )
    ########################## build ##########################
    scene.build(n_envs=n_envs)

    ######################## simulate #########################
    joint_names = [
        "RH_HAA",
        "LH_HAA",
        "RF_HAA",
        "LF_HAA",
        "RH_HFE",
        "LH_HFE",
        "RF_HFE",
        "LF_HFE",
        "RH_KFE",
        "LH_KFE",
        "RF_KFE",
        "LF_KFE",
    ]
    motor_dofs = [robot.get_joint(name).dof_idx_local for name in joint_names]

    robot.set_dofs_kp(np.full(12, 1000), motor_dofs)
    if n_envs > 0:
        robot.control_dofs_position(np.zeros((n_envs, 12)), motor_dofs)
    else:
        robot.control_dofs_position(np.zeros(12), motor_dofs)

    vec_fps = []
    for i in range(1000):
        scene.step()
        vec_fps.append(scene.FPS_tracker.total_fps)
    total_fps = 1.0 / (1.0 / np.array(vec_fps[-N_FRAME_FPS:])).mean()
    return total_fps


@pytest.fixture
def batched_franka(solver, n_envs):
    scene = gs.Scene(
        show_viewer=False,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            constraint_solver=solver,
        ),
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        visualize_contact=True,
    )

    ########################## build ##########################
    scene.build(n_envs=n_envs, env_spacing=(1.0, 1.0))

    ######################## simulate #########################
    vec_fps = []
    for i in range(1000):
        scene.step()
        vec_fps.append(scene.FPS_tracker.total_fps)
    total_fps = 1.0 / (1.0 / np.array(vec_fps[-N_FRAME_FPS:])).mean()
    return total_fps


@pytest.fixture
def random(solver, n_envs):
    scene = gs.Scene(
        show_viewer=False,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            constraint_solver=solver,
        ),
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )

    robot = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/anymal_c/urdf/anymal_c.urdf",
            pos=(0, 0, 0.8),
        ),
        visualize_contact=True,
    )

    ########################## build ##########################
    scene.build(n_envs=n_envs, env_spacing=(1.0, 1.0))

    ######################## simulate #########################
    vec_fps = []
    robot.set_dofs_kp(np.full(12, 1000), np.arange(6, 18))
    dofs = torch.arange(6, 18, device=gs.device)
    robot.control_dofs_position(torch.zeros((n_envs, 12), device=gs.device), dofs)
    for i in range(1000):
        robot.control_dofs_position(torch.rand((n_envs, 12), device=gs.device) * 0.1 - 0.05, dofs)
        scene.step()
        vec_fps.append(scene.FPS_tracker.total_fps)
    total_fps = 1.0 / (1.0 / np.array(vec_fps[-N_FRAME_FPS:])).mean()
    return total_fps


@pytest.fixture
def cubes(solver, n_envs, n_cubes, is_island):
    scene = gs.Scene(
        show_viewer=False,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            constraint_solver=solver,
            use_contact_island=is_island,
        ),
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    # cube = scene.add_entity(
    #     gs.morphs.MJCF(file='xml/one_box.xml'),
    #     visualize_contact=True,
    # )

    for i in range(n_cubes):
        cube = scene.add_entity(
            gs.morphs.Box(
                size=(0.1, 0.1, 0.1),
                pos=(0.0, 0.2 * i, 0.045),
            ),
        )

    ########################## build ##########################
    scene.build(n_envs=n_envs)

    ######################## simulate #########################
    vec_fps = []
    for i in range(1000):
        scene.step()
        vec_fps.append(scene.FPS_tracker.total_fps)
    total_fps = 1.0 / (1.0 / np.array(vec_fps[-N_FRAME_FPS:])).mean()
    return total_fps


@pytest.mark.parametrize(
    "runnable",
    ["random", "anymal_c", "batched_franka"],
)
@pytest.mark.parametrize(
    "solver",
    [gs.constraint_solver.CG, gs.constraint_solver.Newton],
)
@pytest.mark.parametrize("n_envs", [30000])
@pytest.mark.parametrize("backend", [gs.gpu])
def test_speed(capsys, request, pytestconfig, runnable, solver, n_envs):
    total_fps = request.getfixturevalue(runnable)
    msg = f"{runnable} \t| {solver} \t| {total_fps:,.2f} fps \t| {n_envs} envs\n"
    if pytestconfig.getoption("-v"):
        with capsys.disabled():
            print(f"\n{msg}")
    with open(REPORT_FILE, "a") as file:
        file.write(msg)


@pytest.mark.parametrize(
    "solver",
    [gs.constraint_solver.CG, gs.constraint_solver.Newton],
)
@pytest.mark.parametrize("n_cubes", [1, 10])
@pytest.mark.parametrize("is_island", [False, True])
@pytest.mark.parametrize("n_envs", [8192])
@pytest.mark.parametrize("backend", [gs.gpu])
def test_cubes(capsys, request, pytestconfig, solver, n_cubes, is_island, n_envs):
    total_fps = request.getfixturevalue("cubes")
    msg = f"{is_island} island \t| {n_cubes} cubes \t| {solver} \t| {total_fps:,.2f} fps \t| {n_envs} envs\n"
    if pytestconfig.getoption("-v"):
        with capsys.disabled():
            print(f"\n{msg}")
    with open(REPORT_FILE, "a") as file:
        file.write(msg)
