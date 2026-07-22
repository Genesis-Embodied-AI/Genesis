import time

import numpy as np
import pytest

import genesis as gs

# Plans alternate between two configurations on opposite sides of the central pillar, so every call does real
# obstacle-avoidance work instead of replaying a trivial straight line.
QPOS_A = [-0.9, 0.6, 0.0, -1.4, 0.0, 2.0, 0.8, 0.02, 0.02]
QPOS_B = [0.9, 0.6, 0.0, -1.4, 0.0, 2.0, 0.8, 0.02, 0.02]
N_PLANS_WARMUP = 2
N_PLANS_RECORD = 8

pytestmark = [
    pytest.mark.benchmarks,
    pytest.mark.cache(False),
    pytest.mark.debug(False),
]


# ---------------------------------------------------------------------------
# Scene factories
# ---------------------------------------------------------------------------


def make_franka_pillars(n_envs, is_cluttered=False, **scene_kwargs):
    scene = gs.Scene(
        **{"show_viewer": False, "show_FPS": False, **scene_kwargs},
    )
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 1.2),
            pos=(0.45, 0.0, 0.6),
            fixed=True,
        ),
    )
    if is_cluttered:
        scene.add_entity(
            gs.morphs.Box(
                size=(0.1, 0.1, 0.9),
                pos=(0.0, 0.55, 0.45),
                fixed=True,
            ),
        )
        scene.add_entity(
            gs.morphs.Box(
                size=(0.1, 0.1, 0.9),
                pos=(0.0, -0.55, 0.45),
                fixed=True,
            ),
        )
        scene.add_entity(
            gs.morphs.Box(
                size=(0.5, 0.5, 0.05),
                pos=(-0.6, 0.0, 0.2),
                fixed=True,
            ),
        )
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
    )
    time_start = time.time()
    scene.build(n_envs=n_envs)
    compile_time = time.time() - time_start

    return scene, franka, compile_time


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_planning_benchmark(franka, compile_time, n_envs):
    qpos_a = np.array(QPOS_A)
    qpos_b = np.array(QPOS_B)

    # The first plan triggers planner buffer allocation and kernel compilation; fold it into compile_time so the
    # recorded latency reflects steady-state planning only.
    time_start = time.time()
    path = franka.plan_path(qpos_b, qpos_start=qpos_a, seed=0)
    compile_time += time.time() - time_start
    assert bool(path.is_valid.all())

    for i_plan in range(N_PLANS_WARMUP):
        franka.plan_path(qpos_a if i_plan % 2 else qpos_b, qpos_start=qpos_b if i_plan % 2 else qpos_a, seed=i_plan)

    n_valid = 0
    time_start = time.time()
    for i_plan in range(N_PLANS_RECORD):
        path = franka.plan_path(
            qpos_a if i_plan % 2 else qpos_b, qpos_start=qpos_b if i_plan % 2 else qpos_a, seed=i_plan
        )
        n_valid += int(path.is_valid.all())
    time_elapsed = time.time() - time_start
    assert n_valid == N_PLANS_RECORD

    plan_latency = time_elapsed / N_PLANS_RECORD
    plans_per_sec = N_PLANS_RECORD * max(n_envs, 1) / time_elapsed

    return dict(compile_time=compile_time, plan_latency=plan_latency, plans_per_sec=plans_per_sec)


# ---------------------------------------------------------------------------
# Pytest fixture wrappers (thin: just wire factory -> runner)
# ---------------------------------------------------------------------------


@pytest.fixture
def franka_pillar_plan(n_envs):
    _, franka, compile_time = make_franka_pillars(n_envs)
    return run_planning_benchmark(franka, compile_time, n_envs)


@pytest.fixture
def franka_clutter_plan(n_envs):
    _, franka, compile_time = make_franka_pillars(n_envs, is_cluttered=True)
    return run_planning_benchmark(franka, compile_time, n_envs)


# ---------------------------------------------------------------------------
# Parametrized benchmark test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "runnable, n_envs, backend",
    [
        ("franka_pillar_plan", 0, gs.cpu),
        ("franka_pillar_plan", 1024, gs.gpu),
        ("franka_clutter_plan", 0, gs.cpu),
        ("franka_clutter_plan", 1024, gs.gpu),
    ],
)
def test_planning_speed(factory_logger, request, runnable, n_envs):
    with factory_logger(
        {
            "env": runnable,
            "batch_size": n_envs,
        }
    ) as logger:
        logger.write(request.getfixturevalue(runnable))
