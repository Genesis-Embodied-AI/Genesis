import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import genesis as gs

from .test_rigid_benchmarks import STEP_DT, SceneMeta, get_rigid_solver_options, run_benchmark
from .utils import pprint_oneline

pytestmark = [
    pytest.mark.benchmarks,
    pytest.mark.disable_cache(False),
]

# Match the heaviest rigid pyramid benchmark (n_cubes=6 → 21 boxes); entity_idx cycles 1..21 (plane is 0).
PYRAMID_N_CUBES = 6


def make_box_pyramid_with_sensors(
    n_envs: int,
    n_cubes: int,
    n_sensors_each: int,
    solver=None,
    gjk=None,
    **scene_kwargs,
):
    """Box pyramid from rigid benchmarks plus ``n_sensors_each`` instances of every non-camera sensor type."""
    from genesis.engine.sensors.sensor_manager import SensorManager

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            **get_rigid_solver_options(
                dt=STEP_DT,
                **(dict(constraint_solver=solver) if solver is not None else {}),
                **(dict(use_gjk_collision=gjk) if gjk is not None else {}),
            )
        ),
        **{
            "viewer_options": gs.options.ViewerOptions(
                camera_pos=(0.0, -3.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                max_FPS=60,
            ),
            "show_viewer": False,
            "show_FPS": False,
            "profiling_options": gs.options.ProfilingOptions(show_FPS=False),
            **scene_kwargs,
        },
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

    track_sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.05,
            pos=(2.0, 0.0, 0.25),
            fixed=True,
        ),
    )
    track_link_idx = (track_sphere.base_link_idx,)

    n_box_entities = n_cubes * (n_cubes + 1) // 2
    sensors = []
    for option_cls in SensorManager.SENSOR_TYPES_MAP.keys():
        if issubclass(option_cls, gs.sensors.BaseCameraOptions):
            continue
        for k in range(n_sensors_each):
            entity_idx = 1 + (k % n_box_entities)
            sensor_kwargs: dict[str, Any] = {}
            if issubclass(option_cls, gs.sensors.RigidSensorOptionsMixin):
                sensor_kwargs["entity_idx"] = entity_idx
            if issubclass(option_cls, gs.sensors.Raycaster):
                sensor_kwargs["pattern"] = gs.sensors.raycaster.DepthCameraPattern()
            if issubclass(option_cls, gs.sensors.Proximity):
                sensor_kwargs["track_link_idx"] = track_link_idx
            if issubclass(option_cls, gs.sensors.TemperatureGrid):
                sensor_kwargs["properties_dict"] = {-1: gs.sensors.TemperatureProperties()}
            sensors.append(scene.add_sensor(option_cls(**sensor_kwargs)))

    time_start = time.time()
    scene.build(n_envs=n_envs)
    compile_time = time.time() - time_start

    if n_envs > 0:
        for entity in scene.entities[1:-1]:
            entity.set_dofs_velocity(0.04 * torch.rand((n_envs, 6), dtype=gs.tc_float, device=gs.device))

    def step():
        scene.step()
        for sensor in sensors:
            sensor.read()

    return scene, step, SceneMeta(compile_time=compile_time), sensors


@pytest.fixture(scope="session")
def stream_writers(printer_session, request):
    report_path = Path(request.config.getoption("--speed-test-filepath"))

    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id == "gw0":
        worker_count = int(os.environ["PYTEST_XDIST_WORKER_COUNT"])

        for path in report_path.parent.glob("-".join((report_path.stem, "*.txt"))):
            _, worker_id_ = path.stem.rsplit("-", 1)
            worker_num = int(worker_id_[2:])
            if worker_num >= worker_count:
                path.unlink()

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

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

        def write(self, items):
            nonlocal stream_writers

            if stream_writers:
                msg = (
                    pprint_oneline(self.hparams, delimiter=" \t| ")
                    + " \t| "
                    + pprint_oneline(items, delimiter=" \t| ", digits=1)
                )
                for writer in stream_writers:
                    writer(msg)

    return Logger


@pytest.mark.parametrize("n_sensors_each", [2, 8])
@pytest.mark.parametrize(
    "solver, gjk, n_envs",
    [
        (None, None, 2),
        (None, None, 512),
    ],
)
def test_sensor_pyramid_speed(factory_logger, n_sensors_each, solver, gjk, n_envs):
    with factory_logger(
        {
            "env": "sensor_pyramid",
            "batch_size": n_envs,
            "n_sensors_each": n_sensors_each,
            "n_cubes": PYRAMID_N_CUBES,
            **({"constraint_solver": solver} if solver is not None else {}),
            "use_contact_island": False,
            **({"gjk_collision": gjk} if gjk is not None else {}),
        }
    ) as logger:
        _, step_fn, meta, _ = make_box_pyramid_with_sensors(
            n_envs, PYRAMID_N_CUBES, n_sensors_each, solver=solver, gjk=gjk
        )
        logger.write(run_benchmark(step_fn, n_envs=n_envs, meta=meta))
