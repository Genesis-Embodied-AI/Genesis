import hashlib
import numbers
import os
import pytest
import time
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb

import genesis as gs

from .utils import (
    get_hardware_fingerprint,
    get_platform_fingerprint,
    get_git_commit_timestamp,
    get_git_commit_info,
)


BENCHMARK_NAME = "rigid_body"
REPORT_FILE = "speed_test.txt"

STEP_DT = 0.01
DURATION_WARMUP = 45.0
DURATION_RECORD = 15.0

pytestmark = [
    pytest.mark.benchmarks,
    pytest.mark.taichi_offline_cache(False),
]


def pprint_oneline(data, delimiter, digits=None):
    msg_items = []
    for key, value in data.items():
        if isinstance(value, Enum):
            value = value.name
        if digits is not None and isinstance(value, (numbers.Real, np.floating)):
            value = f"{value:.{digits}f}"
        msg_item = "=".join((key, str(value)))
        msg_items.append(msg_item)
    return delimiter.join(msg_items)


def get_rigid_solver_options(**kwargs):
    timestamp = get_git_commit_timestamp()

    # Beyond this point, track performance for default options, finally !
    if not (get_git_commit_timestamp("bbab229d74e5f30e2f641ccf6b009a65f3cbec0f") > timestamp):
        options = dict()

    # Try to be comparable to previous official release (ie 0.2.1) as much as possible.
    elif timestamp >= get_git_commit_timestamp("e46a1ffd33f681155422896c2e343e576e0a72b1"):
        # * Rename 'constraint_resolve_time' in 'constraint_timeconst'
        options = dict(
            enable_mujoco_compatibility=True,
            enable_self_collision=False,
            constraint_timeconst=2 * STEP_DT,
            constraint_solver=gs.constraint_solver.CG,
            max_collision_pairs=100,
            iterations=50,
            tolerance=1e-5,
        )
    elif timestamp >= get_git_commit_timestamp("b1ae77d5c838967dff6f85ee83796a4a82811061"):
        # * 'constraint_solver' now default to Newton (instead of CG)
        # * 'iterations' now default to 100 (instead of 50)
        # * 'tolerance' now default to 1e-8 (instead of 1e-5)
        options = dict(
            enable_mujoco_compatibility=True,
            enable_self_collision=False,
            constraint_resolve_time=2 * STEP_DT,
            constraint_solver=gs.constraint_solver.CG,
            max_collision_pairs=100,
            iterations=50,
            tolerance=1e-5,
        )
    elif timestamp >= get_git_commit_timestamp("6638c6389978594637da216b72be8d7a8f2272c4"):
        # * 'enable_mpr_vanilla' has been renamed in 'enable_mujoco_compatibility'
        options = dict(
            enable_mujoco_compatibility=True,
            constraint_resolve_time=2 * STEP_DT,
            max_collision_pairs=100,
        )
    elif timestamp >= get_git_commit_timestamp("5d04ec4c3ecba4a1e295d6a4a677c041e69092a7"):
        # * Expose option 'enable_multi_contact' (default to True)
        # * Expose 'enable_mpr_vanilla' (default to False)
        options = dict(
            enable_mpr_vanilla=True,
            enable_self_collision=False,
            constraint_resolve_time=2 * STEP_DT,
            max_collision_pairs=100,
        )
    elif timestamp >= get_git_commit_timestamp("6638c6389978594637da216b72be8d7a8f2272c4"):
        # * 'enable_self_collision' now default to True (instead of False)
        options = dict(
            enable_self_collision=False,
            constraint_resolve_time=2 * STEP_DT,
            max_collision_pairs=100,
        )
    elif timestamp >= get_git_commit_timestamp("361d9500cd321b25a63a28ace7a3d94fb9e45f65"):
        # * 'max_collision_pairs' now default to 300 (instead of 100)
        # * 'constraint_resolve_time' now default to None (instead of 2 * DT)
        options = dict(
            constraint_resolve_time=2 * STEP_DT,
            max_collision_pairs=100,
        )

    # Official release 0.2.1. Using the default options is fine
    else:
        options = {}

    return {**options, **kwargs}


def get_file_morph_options(**kwargs):
    timestamp = get_git_commit_timestamp()

    # Beyond this point, track performance for default options, finally !
    if not (get_git_commit_timestamp("bbab229d74e5f30e2f641ccf6b009a65f3cbec0f") > timestamp):
        options = {}

    # Try to be comparable to previous official release (ie 0.2.1) as much as possible.
    elif timestamp >= get_git_commit_timestamp("bbab229d74e5f30e2f641ccf6b009a65f3cbec0f"):
        # * 'decimation' has been enabled back by default
        # * 'decimate_aggressiveness' now defaults to 5
        options = dict(
            decimate=False,
            decimate_aggressiveness=7,
            decompose_robot_error_threshold=float("inf"),
            coacd_options=gs.options.CoacdOptions(
                resolution=2000,
                mcts_iterations=150,
                extrude_margin=0.01,
            ),
        )
    elif timestamp >= get_git_commit_timestamp("d7ea71d5490d0eba6c70a2dfe5943de62227fe68"):
        # * 'decompose_error_threshold' has been split in 'decompose_object_error_threshold' (default to 0.15) and
        #   'decompose_robot_error_threshold' (default to inf)
        options = dict(
            decimate=False,
            decimate_aggressiveness=7,
            decompose_robot_error_threshold=float("inf"),
            coacd_options=gs.options.CoacdOptions(
                resolution=2000,
                mcts_iterations=150,
                extrude_margin=0.01,
            ),
        )
    elif timestamp >= get_git_commit_timestamp("0e7b4be511d261d6ad25a382e5aa335468f5718b"):
        # * 'decimate_aggressiveness' has been exposed and default to 2
        options = dict(
            decimate=False,
            decimate_aggressiveness=7,
            decompose_error_threshold=float("inf"),
            coacd_options=gs.options.CoacdOptions(
                resolution=2000,
                mcts_iterations=150,
                extrude_margin=0.01,
            ),
        )
    elif timestamp >= get_git_commit_timestamp("361d9500cd321b25a63a28ace7a3d94fb9e45f65"):
        # * 'decimate' now defaults to 'convexify'
        # * 'decimate' aggressiveness has been updated from 0 to 2 (but not exposed), it was 7 originally
        options = dict(
            decimate=False,
            decompose_error_threshold=float("inf"),
            coacd_options=gs.options.CoacdOptions(
                resolution=2000,
                mcts_iterations=150,
                extrude_margin=0.01,
            ),
        )
    elif timestamp >= get_git_commit_timestamp("ec6e16949a65dbc62d318a734eeb7f17b0011e03"):
        # * 'decompose_error_threshold' default value updated to 0.15
        options = dict(
            decompose_error_threshold=float("inf"),
            coacd_options=gs.options.CoacdOptions(
                resolution=2000,
                mcts_iterations=150,
                extrude_margin=0.01,
            ),
        )
    elif timestamp >= get_git_commit_timestamp("3bc64493a537b7f52fca6b5fd2dd81f764c34433"):
        # * Move 'decimate', 'decompose_nonconvex' options from Mesh to FileMorph morphs (parent class)
        #   Before that, decimation and convex decomposition could not be enabled at all.
        # * 'convexify' has been enabled back by default
        # * 'decompose_nonconvex' has be deprecated in favor of 'decompose_error_threshold' (default to 0.2)
        # * 'CoacdOptions' options has been updated
        options = dict(
            decompose_error_threshold=float("inf"),
            coacd_options=gs.options.CoacdOptions(
                resolution=2000,
                mcts_iterations=150,
                extrude_margin=0.01,
            ),
        )

    elif timestamp >= get_git_commit_timestamp("8ea732b1a3b340ba7dff295fbd3527cb34b5b676"):
        # * 'convexify' has been disabled by default
        # * 'decimate' has been disabled by default (it only affects Mesh morphs at that time)
        # * 'decompose_nonconvex' has been disabled by default (it only affects Mesh morphs at that time)
        options = dict(
            convexify=True,
        )

    # Official release 0.2.1. Using the default options is fine
    else:
        options = {}

    return {**options, **kwargs}


@pytest.fixture(scope="session")
def stream_writers(backend, printer_session):
    log_path = Path(REPORT_FILE)
    if os.path.exists(log_path):
        os.remove(log_path)
    fd = open(log_path, "w")

    yield (lambda msg: print(msg, file=fd), printer_session)

    fd.close()


@pytest.fixture(scope="function")
def factory_logger(stream_writers):
    class Logger:
        def __init__(self, hparams: dict[str, Any]):
            self.hparams = hparams
            self.benchmark_id = "-".join((BENCHMARK_NAME, pprint_oneline(hparams, delimiter="-")))

            self.logger = None
            self.wandb_run = None

        def __enter__(self):
            nonlocal stream_writers

            if "WANDB_API_KEY" in os.environ:
                assert gs.backend is not None
                revision, timestamp = get_git_commit_info()

                hardware_fringerprint = get_hardware_fingerprint(include_gpu=(gs.backend != gs.cpu))
                platform_fringerprint = get_platform_fingerprint()
                machine_uuid = hashlib.md5(
                    "-".join((hardware_fringerprint, platform_fringerprint)).encode("UTF-8")
                ).hexdigest()

                benchmark_uuid = hashlib.md5(self.benchmark_id.encode("UTF-8")).hexdigest()

                run_uuid = hashlib.md5(
                    "-".join((hardware_fringerprint, platform_fringerprint, self.benchmark_id, revision)).encode(
                        "UTF-8"
                    )
                ).hexdigest()

                self.wandb_run = wandb.init(
                    project="genesis-benchmarks",
                    name="-".join((self.benchmark_id, revision)),
                    id=run_uuid,
                    tags=[BENCHMARK_NAME, benchmark_uuid],
                    config={
                        "revision": revision,
                        "timestamp": timestamp,
                        "machine_uuid": machine_uuid,
                        "hardware": hardware_fringerprint,
                        "platform": platform_fringerprint,
                        "backend": str(gs.backend.name),
                        "benchmark_id": self.benchmark_id,
                        **self.hparams,
                    },
                    settings=wandb.Settings(
                        x_disable_stats=True,
                        console="off",
                    ),
                )
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            if self.wandb_run is not None:
                self.wandb_run.finish()

        def write(self, items):
            nonlocal stream_writers

            if self.wandb_run is not None:
                self.wandb_run.log(
                    {
                        "timestamp": self.wandb_run.config["timestamp"],
                        **items,
                    }
                )

            if stream_writers:
                msg = (
                    pprint_oneline(self.hparams, delimiter=" \t| ")
                    + " \t| "
                    + pprint_oneline(items, delimiter=" \t| ", digits=0)
                )
                for writer in stream_writers:
                    writer(msg)

    return Logger


@pytest.fixture
def anymal_c(solver, n_envs, gjk):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            **get_rigid_solver_options(
                dt=STEP_DT,
                constraint_solver=solver,
                enable_self_collision=False,
                use_gjk_collision=gjk,
            )
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=False,
        show_FPS=False,
    )

    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(
            **get_file_morph_options(
                file="urdf/anymal_c/urdf/anymal_c.urdf",
                pos=(0, 0, 0.8),
            )
        ),
    )
    time_start = time.time()
    scene.build(n_envs=n_envs)
    compile_time = time.time() - time_start

    joints_name = (
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
    )
    motors_dof_idx = [robot.get_joint(name).dof_start for name in joints_name]
    robot.set_dofs_kp(np.full(12, 1000), motors_dof_idx)
    if n_envs > 0:
        robot.control_dofs_position(np.zeros((n_envs, 12)), motors_dof_idx)
    else:
        robot.control_dofs_position(np.zeros(12), motors_dof_idx)

    num_steps = 0
    is_recording = False
    time_start = time.time()
    while True:
        scene.step()
        time_elapsed = time.time() - time_start
        if is_recording:
            num_steps += 1
            if time_elapsed > DURATION_RECORD:
                break
        elif time_elapsed > DURATION_WARMUP:
            time_start = time.time()
            is_recording = True
    runtime_fps = num_steps * n_envs / time_elapsed
    realtime_factor = runtime_fps * STEP_DT

    return {"compile_time": compile_time, "runtime_fps": runtime_fps, "realtime_factor": realtime_factor}


@pytest.fixture
def batched_franka(solver, n_envs, gjk):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            **get_rigid_solver_options(
                dt=STEP_DT,
                constraint_solver=solver,
                enable_self_collision=False,
                use_gjk_collision=gjk,
            )
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=False,
        show_FPS=False,
    )

    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        gs.morphs.MJCF(
            **get_file_morph_options(
                file="xml/franka_emika_panda/panda.xml",
            )
        ),
        visualize_contact=True,
    )
    time_start = time.time()
    scene.build(n_envs=n_envs, env_spacing=(1.0, 1.0))
    compile_time = time.time() - time_start

    num_steps = 0
    is_recording = False
    time_start = time.time()
    while True:
        scene.step()
        time_elapsed = time.time() - time_start
        if is_recording:
            num_steps += 1
            if time_elapsed > DURATION_RECORD:
                break
        elif time_elapsed > DURATION_WARMUP:
            time_start = time.time()
            is_recording = True
    runtime_fps = num_steps * n_envs / time_elapsed
    realtime_factor = runtime_fps * STEP_DT

    return {"compile_time": compile_time, "runtime_fps": runtime_fps, "realtime_factor": realtime_factor}


@pytest.fixture
def random(solver, n_envs, gjk):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            **get_rigid_solver_options(
                dt=STEP_DT,
                constraint_solver=solver,
                enable_self_collision=False,
                use_gjk_collision=gjk,
            )
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=False,
        show_FPS=False,
    )

    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(
            **get_file_morph_options(
                file="urdf/anymal_c/urdf/anymal_c.urdf",
                pos=(0, 0, 0.8),
            )
        ),
        visualize_contact=True,
    )
    time_start = time.time()
    scene.build(n_envs=n_envs, env_spacing=(1.0, 1.0))
    compile_time = time.time() - time_start

    robot.set_dofs_kp(np.full((12,), fill_value=1000.0), np.arange(6, 18))
    dofs = torch.arange(6, 18, device=gs.device)
    robot.control_dofs_position(torch.zeros((n_envs, 12), device=gs.device), dofs)

    num_steps = 0
    is_recording = False
    time_start = time.time()
    while True:
        robot.control_dofs_position(torch.rand((n_envs, 12), device=gs.device) * 0.1 - 0.05, dofs)
        scene.step()
        time_elapsed = time.time() - time_start
        if is_recording:
            num_steps += 1
            if time_elapsed > DURATION_RECORD:
                break
        elif time_elapsed > DURATION_WARMUP:
            time_start = time.time()
            is_recording = True
    runtime_fps = num_steps * n_envs / time_elapsed
    realtime_factor = runtime_fps * STEP_DT

    return {"compile_time": compile_time, "runtime_fps": runtime_fps, "realtime_factor": realtime_factor}


@pytest.fixture
def cubes(solver, n_envs, n_cubes, enable_island, gjk):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            **get_rigid_solver_options(
                dt=STEP_DT,
                constraint_solver=solver,
                use_contact_island=enable_island,
                use_gjk_collision=gjk,
            )
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=False,
        show_FPS=False,
    )

    scene.add_entity(gs.morphs.Plane())
    for i in range(n_cubes):
        scene.add_entity(
            gs.morphs.Box(
                size=(0.1, 0.1, 0.1),
                pos=(0.0, 0.2 * i, 0.045),
            ),
        )
    time_start = time.time()
    scene.build(n_envs=n_envs)
    compile_time = time.time() - time_start

    num_steps = 0
    is_recording = False
    time_start = time.time()
    while True:
        scene.step()
        time_elapsed = time.time() - time_start
        if is_recording:
            num_steps += 1
            if time_elapsed > DURATION_RECORD:
                break
        elif time_elapsed > DURATION_WARMUP:
            time_start = time.time()
            is_recording = True
    runtime_fps = num_steps * n_envs / time_elapsed
    realtime_factor = runtime_fps * STEP_DT

    return {"compile_time": compile_time, "runtime_fps": runtime_fps, "realtime_factor": realtime_factor}


@pytest.fixture
def box_pyramid(solver, n_envs, n_cubes, enable_island, gjk, enable_mujoco_compatibility):
    x_pos = 0.0

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            **get_rigid_solver_options(
                dt=STEP_DT,
                constraint_solver=solver,
                use_contact_island=enable_island,
                use_gjk_collision=gjk,
                box_box_detection=False,
                enable_mujoco_compatibility=enable_mujoco_compatibility,
            )
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(x_pos, -3.5, 2.5),
            camera_lookat=(x_pos, 0.0, 0.5),
            camera_fov=30,
            max_FPS=60,
        ),
        show_viewer=False,
        show_FPS=False,
    )

    scene.add_entity(gs.morphs.Plane(pos=(x_pos, 0, 0)))
    # create pyramid of boxes
    box_size = 0.25
    box_spacing = box_size
    vec_one = np.array([1.0, 1.0, 1.0])
    box_pos_offset = (x_pos - 0.5, 1, 0.0) + 0.5 * box_size * vec_one
    for i in range(n_cubes):
        for j in range(n_cubes - i):
            scene.add_entity(
                gs.morphs.Box(
                    size=box_size * vec_one,
                    pos=box_pos_offset + box_spacing * np.array([i + 0.5 * j, 0.0, j]),
                ),
            )

    time_start = time.time()
    scene.build(n_envs=n_envs)
    compile_time = time.time() - time_start

    num_steps = 0
    is_recording = False
    time_start = time.time()
    while True:
        scene.step()
        time_elapsed = time.time() - time_start
        if is_recording:
            num_steps += 1
            if time_elapsed > DURATION_RECORD:
                break
        elif time_elapsed > DURATION_WARMUP:
            time_start = time.time()
            is_recording = True
    runtime_fps = num_steps * n_envs / time_elapsed
    realtime_factor = runtime_fps * STEP_DT

    return {"compile_time": compile_time, "runtime_fps": runtime_fps, "realtime_factor": realtime_factor}


@pytest.mark.parametrize("runnable", ["anymal_c", "batched_franka"])
@pytest.mark.parametrize("solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("n_envs", [30000])
@pytest.mark.parametrize("gjk", [False, True])
def test_speed(factory_logger, request, runnable, solver, n_envs, gjk):
    with factory_logger(
        {
            "env": runnable,
            "batch_size": n_envs,
            "constraint_solver": solver,
            "use_contact_island": False,
            "gjk_collision": gjk,
        }
    ) as logger:
        logger.write(request.getfixturevalue(runnable))


@pytest.mark.parametrize("solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("n_cubes", [10])
@pytest.mark.parametrize("enable_island", [False, True])
@pytest.mark.parametrize("n_envs", [8192])
@pytest.mark.parametrize("gjk", [False, True])
def test_cubes(factory_logger, request, n_cubes, solver, enable_island, n_envs, gjk):
    with factory_logger(
        {
            "env": f"cube#{n_cubes}",
            "batch_size": n_envs,
            "constraint_solver": solver,
            "use_contact_island": enable_island,
            "gjk_collision": gjk,
        }
    ) as logger:
        logger.write(request.getfixturevalue("cubes"))


# FIXME:Increasing the batch size triggers CUDA out-of-memory error (Nvidia H100)
# FIXME:Increasing # cubes triggers CUDA illegal memory access error for all collision methods (Nvidia RTX 5900)
@pytest.mark.parametrize("solver", [gs.constraint_solver.Newton])
@pytest.mark.parametrize("n_cubes", [5])
@pytest.mark.parametrize("enable_island", [False])
@pytest.mark.parametrize("n_envs", [2048])
@pytest.mark.parametrize(
    "gjk, enable_mujoco_compatibility",
    [
        (False, True),  # MPR
        (False, False),  # MPR+SDF
        (True, False),  # GJK
    ],
)
def test_box_pyramid(factory_logger, request, n_cubes, solver, enable_island, n_envs, gjk, enable_mujoco_compatibility):
    with factory_logger(
        {
            "env": f"box_pyramid#{n_cubes}",
            "batch_size": n_envs,
            "constraint_solver": solver,
            "use_contact_island": enable_island,
            "gjk_collision": gjk,
            "enable_mujoco_compatibility": enable_mujoco_compatibility,
        }
    ) as logger:
        logger.write(request.getfixturevalue("box_pyramid"))
