import hashlib
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import wandb

import genesis as gs

from .utils import (
    get_hardware_fingerprint,
    get_hf_dataset,
    get_platform_fingerprint,
    get_git_commit_timestamp,
    get_git_commit_info,
    pprint_oneline,
)


BENCHMARK_NAME = "rigid_body"
REPORT_FILE = "speed_test.txt"

STEP_DT = 0.01
DURATION_WARMUP = 45.0
DURATION_RECORD = 15.0

pytestmark = [
    pytest.mark.benchmarks,
    pytest.mark.disable_cache(False),
]


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
def stream_writers(printer_session):
    report_path = Path(REPORT_FILE)

    # Delete old unrelated worker-specific reports
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id == "gw0":
        worker_count = int(os.environ["PYTEST_XDIST_WORKER_COUNT"])

        for path in report_path.parent.glob("-".join((report_path.stem, "*.txt"))):
            _, worker_id_ = path.stem.rsplit("-", 1)
            worker_num = int(worker_id_[2:])
            if worker_num >= worker_count:
                path.unlink()

    # Create new empty worker-specific report
    report_name = "-".join(filter(None, (report_path.stem, worker_id)))
    report_path = report_path.with_name(f"{report_name}.txt")
    if report_path.exists():
        report_path.unlink()
    fd = open(report_path, "w")

    yield (lambda msg: print(msg, file=fd, flush=True), printer_session)

    fd.close()


@pytest.fixture(scope="function")
def factory_logger(stream_writers):
    class Logger:
        def __init__(self, hparams: dict[str, Any]):
            self.hparams = {
                **hparams,
                "dtype": "ndarray" if gs.use_ndarray else "field",
                "backend": str(gs.backend.name),
            }
            self.benchmark_id = "-".join((BENCHMARK_NAME, pprint_oneline(self.hparams, delimiter="-")))

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
                    + pprint_oneline(items, delimiter=" \t| ", digits=1)
                )
                for writer in stream_writers:
                    writer(msg)

    return Logger


@pytest.fixture
def go2(solver, n_envs, gjk, pytorch_profiler_step):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            **get_rigid_solver_options(
                dt=STEP_DT,
                **(dict(constraint_solver=solver) if solver is not None else {}),
                **(dict(use_gjk_collision=gjk) if gjk is not None else {}),
            )
        ),
        show_viewer=False,
        show_FPS=False,
    )

    scene.add_entity(
        gs.morphs.Plane(),
    )
    robot = scene.add_entity(
        gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf"),
        vis_mode="collision",
    )
    time_start = time.time()
    scene.build(n_envs=n_envs)
    compile_time = time.time() - time_start

    ctrl_pos = torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5],
        dtype=gs.tc_float,
        device=gs.device,
    )
    robot.control_dofs_position(ctrl_pos, dofs_idx_local=slice(6, None))

    init_qpos = torch.tensor(
        [[0.0, 0.0, 0.42, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5]],
        dtype=gs.tc_float,
        device=gs.device,
    ).repeat((scene.n_envs, 1))
    dofs_lower_bound, dofs_upper_bound = robot.get_dofs_limit()
    init_qpos[:, 7:] = dofs_lower_bound[6:] + (dofs_upper_bound[6:] - dofs_lower_bound[6:]) * torch.rand(
        (scene.n_envs, robot.n_dofs - 6), dtype=gs.tc_float, device=gs.device
    )
    robot.set_qpos(init_qpos)

    num_steps = 0
    is_recording = False
    time_start = time.time()
    while True:
        scene.step()
        pytorch_profiler_step()
        time_elapsed = time.time() - time_start
        if is_recording:
            num_steps += 1
            if time_elapsed > DURATION_RECORD:
                break
        elif time_elapsed > DURATION_WARMUP:
            time_start = time.time()
            is_recording = True
    runtime_fps = int(num_steps * max(n_envs, 1) / time_elapsed)
    realtime_factor = runtime_fps * STEP_DT

    return {"compile_time": compile_time, "runtime_fps": runtime_fps, "realtime_factor": realtime_factor}


def _anymal(solver, n_envs, gjk, control, profiler_step):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            **get_rigid_solver_options(
                dt=STEP_DT,
                **(dict(constraint_solver=solver) if solver is not None else {}),
                **(dict(use_gjk_collision=gjk) if gjk is not None else {}),
            )
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

    motors_dof_idx = slice(6, None)
    robot.set_dofs_kp(1000.0, motors_dof_idx)
    robot.control_dofs_position(0.0, motors_dof_idx)

    if control == "uniform":
        rand_shape = (12,)
    elif control == "per_env":
        rand_shape = (n_envs, 12)
    else:
        rand_shape = None

    num_steps = 0
    is_recording = False
    time_start = time.time()
    while True:
        if rand_shape is not None:
            robot.control_dofs_position(
                torch.rand(rand_shape, dtype=gs.tc_float, device=gs.device) * 0.1 - 0.05, motors_dof_idx
            )
        scene.step()
        profiler_step()
        time_elapsed = time.time() - time_start
        if is_recording:
            num_steps += 1
            if time_elapsed > DURATION_RECORD:
                break
        elif time_elapsed > DURATION_WARMUP:
            time_start = time.time()
            is_recording = True
    runtime_fps = int(num_steps * max(n_envs, 1) / time_elapsed)
    realtime_factor = runtime_fps * STEP_DT

    return {"compile_time": compile_time, "runtime_fps": runtime_fps, "realtime_factor": realtime_factor}


@pytest.fixture
def anymal_zero(solver, n_envs, gjk, pytorch_profiler_step):
    return _anymal(solver, n_envs, gjk, control=None, profiler_step=pytorch_profiler_step)


@pytest.fixture
def anymal_uniform(solver, n_envs, gjk, pytorch_profiler_step):
    return _anymal(solver, n_envs, gjk, control="uniform", profiler_step=pytorch_profiler_step)


@pytest.fixture
def anymal_random(solver, n_envs, gjk, pytorch_profiler_step):
    return _anymal(solver, n_envs, gjk, control="per_env", profiler_step=pytorch_profiler_step)


def _franka(solver, n_envs, gjk, is_collision_free, is_randomized, accessors, profiler_step):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            **get_rigid_solver_options(
                dt=STEP_DT,
                enable_neutral_collision=True,
                **(dict(constraint_solver=solver) if solver is not None else {}),
                **(dict(use_gjk_collision=gjk) if gjk is not None else {}),
            )
        ),
        show_viewer=False,
        show_FPS=False,
    )

    scene.add_entity(gs.morphs.Plane())
    franka = scene.add_entity(
        gs.morphs.MJCF(
            **get_file_morph_options(
                file="xml/franka_emika_panda/panda.xml",
            )
        ),
    )
    time_start = time.time()
    scene.build(n_envs=n_envs)
    compile_time = time.time() - time_start

    qpos0 = torch.tensor([0, 0, 0, -1.0, 0, 1.0, 0, 0.02, 0.02], dtype=gs.tc_float, device=gs.device)
    if n_envs > 0:
        qpos0 = torch.tile(qpos0, (n_envs, 1))
    if is_collision_free:
        franka.set_qpos(qpos0)
        franka.control_dofs_position(qpos0)

    if n_envs > 0 and is_randomized:
        vel0 = 0.2 * torch.clip(torch.randn((n_envs, franka.n_dofs), dtype=gs.tc_float, device=gs.device), -1.0, 1.0)
        vel0[:, [link.dof_start for link in franka.links if not link.name.startswith("link") and link.n_dofs]] = 0.0
    else:
        vel0 = torch.zeros((*((n_envs,) if n_envs > 0 else ()), franka.n_dofs), dtype=gs.tc_float, device=gs.device)
    franka.set_dofs_velocity(vel0)

    if n_envs > 0:
        n_reset_envs = max(int(0.02 * n_envs), 1)
        reset_envs_idx = torch.randperm(n_envs, dtype=gs.tc_int, device=gs.device)[:n_reset_envs]
        reset_envs_mask = torch.isin(scene._envs_idx, reset_envs_idx)
    else:
        reset_envs_mask = None

    dofs_stiffness = franka.get_dofs_stiffness()
    dofs_damping = franka.get_dofs_damping()

    num_steps = 0
    is_recording = False
    time_start = time.time()
    while True:
        scene.step()
        profiler_step()
        if accessors:
            franka.get_ang()
            franka.get_vel()
            franka.get_dofs_position()
            franka.get_dofs_velocity()
            franka.get_links_pos()
            franka.get_links_quat()
            franka.get_links_vel()
            franka.get_contacts()
            franka.control_dofs_position(qpos0)
            franka.set_dofs_stiffness(dofs_stiffness)
            franka.set_dofs_damping(dofs_damping)
            franka.set_dofs_velocity(vel0, envs_idx=reset_envs_mask, skip_forward=True)
            franka.set_qpos(qpos0, envs_idx=reset_envs_mask, zero_velocity=False, skip_forward=True)

        time_elapsed = time.time() - time_start
        if is_recording:
            num_steps += 1
            if time_elapsed > DURATION_RECORD:
                break
        elif time_elapsed > DURATION_WARMUP:
            time_start = time.time()
            is_recording = True
    runtime_fps = int(num_steps * max(n_envs, 1) / time_elapsed)
    realtime_factor = runtime_fps * STEP_DT

    return {"compile_time": compile_time, "runtime_fps": runtime_fps, "realtime_factor": realtime_factor}


@pytest.fixture
def franka(solver, n_envs, gjk, pytorch_profiler_step):
    return _franka(
        solver,
        n_envs,
        gjk,
        is_collision_free=False,
        is_randomized=False,
        accessors=False,
        profiler_step=pytorch_profiler_step,
    )


@pytest.fixture
def franka_random(solver, n_envs, gjk, pytorch_profiler_step):
    return _franka(
        solver,
        n_envs,
        gjk,
        is_collision_free=False,
        is_randomized=True,
        accessors=False,
        profiler_step=pytorch_profiler_step,
    )


@pytest.fixture
def franka_free(solver, n_envs, gjk, pytorch_profiler_step):
    return _franka(
        solver,
        n_envs,
        gjk,
        is_collision_free=True,
        is_randomized=False,
        accessors=False,
        profiler_step=pytorch_profiler_step,
    )


@pytest.fixture
def franka_accessors(solver, n_envs, gjk, pytorch_profiler_step):
    return _franka(
        solver,
        n_envs,
        gjk,
        is_collision_free=True,
        is_randomized=False,
        accessors=True,
        profiler_step=pytorch_profiler_step,
    )


def _duck_in_box(solver, n_envs, gjk, hard, profiler_step):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            **(dict(constraint_solver=solver) if solver is not None else {}),
            **(dict(use_gjk_collision=gjk) if gjk is not None else {}),
        ),
        show_viewer=False,
        show_FPS=False,
    )
    scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/tank.obj",
            scale=5.0,
            pos=(0.0, 0.0, 0.0),
            euler=(90, 0, 90),
            fixed=True,
        ),
        vis_mode="collision",
    )
    if hard:
        mesh_kwargs = dict(
            pos=(0.0, 0.0, 0.035),
        )
    else:
        mesh_kwargs = dict(
            pos=(0.1, 0.1, 0.035),
            decompose_object_error_threshold=float("inf"),
        )
    duck = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/duck.obj",
            scale=0.04,
            euler=(90, 0, 90),
            **mesh_kwargs,
        ),
    )

    time_start = time.time()
    scene.build(n_envs=n_envs)
    compile_time = time.time() - time_start

    if n_envs > 0:
        duck.set_dofs_velocity(0.5 * torch.rand((n_envs, 6), dtype=gs.tc_float, device=gs.device))

    num_steps = 0
    is_recording = False
    time_start = time.time()
    while True:
        scene.step()
        profiler_step()
        time_elapsed = time.time() - time_start
        if is_recording:
            num_steps += 1
            if time_elapsed > DURATION_RECORD:
                break
        elif time_elapsed > DURATION_WARMUP:
            time_start = time.time()
            is_recording = True
    runtime_fps = int(num_steps * max(n_envs, 1) / time_elapsed)
    realtime_factor = runtime_fps * STEP_DT

    return {"compile_time": compile_time, "runtime_fps": runtime_fps, "realtime_factor": realtime_factor}


@pytest.fixture
def duck_in_box_easy(solver, n_envs, gjk, pytorch_profiler_step):
    return _duck_in_box(solver, n_envs, gjk, hard=False, profiler_step=pytorch_profiler_step)


@pytest.fixture
def duck_in_box_hard(solver, n_envs, gjk, pytorch_profiler_step):
    return _duck_in_box(solver, n_envs, gjk, hard=True, profiler_step=pytorch_profiler_step)


def _box_pyramid(solver, n_envs, gjk, n_cubes, profiler_step):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            **get_rigid_solver_options(
                dt=STEP_DT,
                **(dict(constraint_solver=solver) if solver is not None else {}),
                **(dict(use_gjk_collision=gjk) if gjk is not None else {}),
            )
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -3.5, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            max_FPS=60,
        ),
        show_viewer=False,
        show_FPS=False,
    )

    scene.add_entity(gs.morphs.Plane())
    box_size = 0.25
    box_spacing = (1.0 - 1e-3) * box_size
    box_pos_offset = (-0.5, 1.0, 0.0) + 0.5 * np.array([box_size, box_size, box_size])
    for i in range(n_cubes):
        for j in range(n_cubes - i):
            scene.add_entity(
                gs.morphs.Box(
                    size=[box_size, box_size, box_size],
                    pos=box_pos_offset + box_spacing * np.array([i + 0.5 * j, 0.0, j]),
                ),
            )

    time_start = time.time()
    scene.build(n_envs=n_envs)
    compile_time = time.time() - time_start

    if n_envs > 0:
        for box in scene.entities[1:]:
            box.set_dofs_velocity(0.04 * torch.rand((n_envs, 6), dtype=gs.tc_float, device=gs.device))

    num_steps = 0
    is_recording = False
    time_start = time.time()
    while True:
        scene.step()
        profiler_step()
        time_elapsed = time.time() - time_start
        if is_recording:
            num_steps += 1
            if time_elapsed > DURATION_RECORD:
                break
        elif time_elapsed > DURATION_WARMUP:
            time_start = time.time()
            is_recording = True
    runtime_fps = int(num_steps * max(n_envs, 1) / time_elapsed)
    realtime_factor = runtime_fps * STEP_DT

    return {"compile_time": compile_time, "runtime_fps": runtime_fps, "realtime_factor": realtime_factor}


@pytest.fixture
def box_pyramid_3(solver, n_envs, gjk, pytorch_profiler_step):
    return _box_pyramid(solver, n_envs, gjk, n_cubes=3, profiler_step=pytorch_profiler_step)


@pytest.fixture
def box_pyramid_4(solver, n_envs, gjk, pytorch_profiler_step):
    return _box_pyramid(solver, n_envs, gjk, n_cubes=4, profiler_step=pytorch_profiler_step)


@pytest.fixture
def box_pyramid_5(solver, n_envs, gjk, pytorch_profiler_step):
    return _box_pyramid(solver, n_envs, gjk, n_cubes=5, profiler_step=pytorch_profiler_step)


@pytest.fixture
def box_pyramid_6(solver, n_envs, gjk, pytorch_profiler_step):
    return _box_pyramid(solver, n_envs, gjk, n_cubes=6, profiler_step=pytorch_profiler_step)


@pytest.fixture
def g1_fall(solver, n_envs, gjk, pytorch_profiler_step):
    """G1 humanoid robot falling from above a plane."""
    import quadrants as qd

    # This is sufficient, as long as we use sync
    duration_warmup = 20.0
    duration_record = 5.0

    # Needed in order to reproduce the benefits for parallelizing Hessian that
    # we see on Eden Beyond Mimic
    step_dt = 0.005

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            dt=step_dt,
            iterations=10,
            tolerance=1e-5,
            ls_iterations=20,
            **(dict(constraint_solver=solver) if solver is not None else {}),
            **(dict(use_gjk_collision=gjk) if gjk is not None else {}),
        ),
        show_viewer=False,
        show_FPS=False,
    )

    scene.add_entity(
        gs.morphs.Plane(),
    )
    asset_path = get_hf_dataset(pattern="unitree_g1/*")
    robot = scene.add_entity(
        gs.morphs.MJCF(
            **get_file_morph_options(
                file=f"{asset_path}/unitree_g1/g1_29dof_rev_1_0.xml",
                pos=(0, 0, 1.0),
            )
        ),
        vis_mode="collision",
    )
    time_start = time.time()
    scene.build(n_envs=n_envs)
    compile_time = time.time() - time_start

    # Set initial position with robot elevated above ground
    init_qpos = torch.zeros((robot.n_qs,), dtype=gs.tc_float, device=gs.device)
    init_qpos[2] = 1.0  # z position
    init_qpos[3] = 1.0  # quaternion w component
    robot.set_qpos(init_qpos)

    random_forces = torch.zeros((n_envs, robot.n_dofs), dtype=gs.tc_float, device=gs.device)
    max_force = 50.0

    num_steps = 0
    is_recording = False
    qd.sync()
    time_start = time.time()
    while True:
        random_forces.uniform_(-max_force, max_force)
        robot.control_dofs_force(random_forces)
        scene.step()
        pytorch_profiler_step()
        time_elapsed = time.time() - time_start
        if is_recording:
            num_steps += 1
            if time_elapsed > duration_record:
                qd.sync()
                time_elapsed = time.time() - time_start
                break
        elif time_elapsed > duration_warmup:
            qd.sync()
            time_start = time.time()
            is_recording = True
    runtime_fps = int(num_steps * max(n_envs, 1) / time_elapsed)
    realtime_factor = runtime_fps * step_dt

    return {"compile_time": compile_time, "runtime_fps": runtime_fps, "realtime_factor": realtime_factor}


@pytest.mark.parametrize(
    "runnable, solver, gjk, n_envs, backend",
    [
        ("duck_in_box_easy", None, True, 30000, gs.gpu),
        ("duck_in_box_easy", None, False, 30000, gs.gpu),
        ("duck_in_box_hard", None, True, 30000, gs.gpu),
        ("duck_in_box_hard", None, False, 30000, gs.gpu),
        ("duck_in_box_hard", None, None, 0, gs.cpu),
        ("anymal_random", None, None, 30000, gs.gpu),
        ("anymal_uniform", None, None, 30000, gs.gpu),
        ("anymal_zero", None, None, 30000, gs.gpu),
        ("anymal_zero", None, None, 0, gs.cpu),
        ("go2", None, True, 4096, gs.gpu),
        ("go2", gs.constraint_solver.CG, False, 4096, gs.gpu),
        ("go2", gs.constraint_solver.Newton, False, 4096, gs.gpu),
        ("franka_accessors", None, None, 0, gs.cpu),
        ("franka_accessors", None, None, 30000, gs.gpu),
        ("franka_free", None, None, 30000, gs.gpu),
        ("franka", None, None, 30000, gs.gpu),
        ("franka_random", None, False, 30000, gs.gpu),
        ("franka_random", None, True, 30000, gs.gpu),
        ("franka_random", gs.constraint_solver.CG, None, 30000, gs.gpu),
        ("franka_random", gs.constraint_solver.Newton, None, 30000, gs.gpu),
        ("franka_random", None, None, 0, gs.cpu),
        ("box_pyramid_3", None, None, 4096, gs.gpu),
        ("box_pyramid_4", None, None, 4096, gs.gpu),
        ("box_pyramid_5", None, None, 4096, gs.gpu),
        ("box_pyramid_6", None, True, 4096, gs.gpu),
        ("box_pyramid_6", None, False, 4096, gs.gpu),
        ("g1_fall", gs.constraint_solver.Newton, None, 4096, gs.gpu),
    ],
)
def test_speed(factory_logger, request, runnable, solver, gjk, n_envs):
    with factory_logger(
        {
            "env": runnable,
            "batch_size": n_envs,
            **({"constraint_solver": solver} if solver is not None else {}),
            "use_contact_island": False,
            **({"gjk_collision": gjk} if gjk is not None else {}),
        }
    ) as logger:
        logger.write(request.getfixturevalue(runnable))
