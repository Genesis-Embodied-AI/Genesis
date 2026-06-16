"""Phase 0 proof-gate harness for the island + hibernation reboot.

Measurement-only, runs on current main. Each invocation builds ONE config (one scene, one backend,
one n_envs, one feature setting) and prints a single parseable RESULT line, mirroring table.py's
one-scene-per-process convention. A shell loop drives the full matrix and the arms are diffed
afterward.

Scenes deliberately isolate one mechanism by toggling the other off (see the plan
~/.claude/plans/unified_island_hibernation_solver.md):
  island_best       K independent 3-box stacks, spaced apart  -> K islands  (run hibernation OFF)
  island_worst      one tower of N boxes                       -> 1 island   (run hibernation OFF)
  hibernation_best  one tower of N boxes that settles to rest  -> 1 sleeping island (hib ON vs OFF)
  hibernation_worst N frictionless boxes sliding forever       -> never rest (hib ON vs OFF, overhead)

Island structure is known by construction, so no internal island-count readout is needed.

Usage:
    python tests/bench_island_hibernation.py --scene island_worst --backend cpu \
        --n_envs 1 --size 16 --island on --hibernation off --solver newton
"""

import argparse
import time

import numpy as np

import genesis as gs
from genesis.vis.rasterizer import Rasterizer

# Headless physics-only benchmark: skip building the offscreen GL renderer. This environment has no
# display and no osmesa, so the pyglet offscreen context cannot be created. Physics timing is fully
# independent of rendering and no camera is ever used here, so the renderer is dead weight.
Rasterizer.build = lambda self: None

# Boxes are 0.1 m cubes; stacks/towers are built with a small vertical gap so they drop a hair and
# settle into resting contact during warmup rather than spawning interpenetrating.
BOX_SIZE = 0.1
STACK_GAP = 0.005
GRID_SPACING = 0.5  # horizontal spacing between independent islands (>> box size, so no interaction)

BACKENDS = {"cpu": gs.cpu, "gpu": gs.gpu}


def _add_box(scene, pos, friction=None):
    return scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=pos,
        ),
        material=gs.materials.Rigid(friction=friction) if friction is not None else None,
    )


def build_island_best(scene, size):
    # `size` independent stacks of 3 boxes on a grid; spacing >> box so the stacks never interact.
    n_cols = int(np.ceil(np.sqrt(size)))
    for i_stack in range(size):
        col = i_stack % n_cols
        row = i_stack // n_cols
        x = col * GRID_SPACING
        y = row * GRID_SPACING
        for i_box in range(3):
            z = BOX_SIZE / 2 + i_box * (BOX_SIZE + STACK_GAP)
            _add_box(scene, (x, y, z))
    return None


def build_island_worst(scene, size):
    # One tower of `size` boxes: every box contacts its neighbours -> a single coupled island.
    for i_box in range(size):
        z = BOX_SIZE / 2 + i_box * (BOX_SIZE + STACK_GAP)
        _add_box(scene, (0.0, 0.0, z))
    return None


def build_hibernation_best(scene, size):
    # Same family as island_worst (one tower) - it settles into one resting island that should sleep.
    return build_island_worst(scene, size)


def build_hibernation_worst(scene, size):
    # `size` frictionless boxes on a grid, each kicked horizontally after build -> they slide forever
    # at constant speed (no friction on a flat plane), velocity always above the hibernation
    # threshold, so hibernation never fires. Returns the boxes so the caller can set the velocity.
    n_cols = int(np.ceil(np.sqrt(size)))
    boxes = []
    for i_box in range(size):
        col = i_box % n_cols
        row = i_box // n_cols
        x = col * GRID_SPACING
        y = row * GRID_SPACING
        # Minimum allowed friction (0.01); the kick velocity is large enough that the slight
        # deceleration never brings velocity below the hibernation threshold within the run.
        boxes.append(_add_box(scene, (x, y, BOX_SIZE / 2 + 0.01), friction=0.01))
    return boxes


SCENE_BUILDERS = {
    "island_best": build_island_best,
    "island_worst": build_island_worst,
    "hibernation_best": build_hibernation_best,
    "hibernation_worst": build_hibernation_worst,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, choices=list(SCENE_BUILDERS))
    parser.add_argument("--backend", default="cpu", choices=list(BACKENDS))
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--island", default="off", choices=["on", "off"])
    parser.add_argument("--hibernation", default="off", choices=["on", "off"])
    parser.add_argument("--solver", default="newton", choices=["newton", "cg"])
    parser.add_argument("--settle_steps", type=int, default=200)
    parser.add_argument("--timed_steps", type=int, default=100)
    args = parser.parse_args()

    use_island = args.island == "on"
    use_hibernation = args.hibernation == "on"
    # Hibernation requires the contact-island path; enforce it so the config is not silently dropped.
    if use_hibernation and not use_island:
        use_island = True

    gs.init(backend=BACKENDS[args.backend])

    constraint_solver = gs.constraint_solver.CG if args.solver == "cg" else gs.constraint_solver.Newton

    t0 = time.perf_counter()
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            gravity=(0, 0, -9.8),
        ),
        rigid_options=gs.options.RigidOptions(
            constraint_solver=constraint_solver,
            use_contact_island=use_island,
            use_hibernation=use_hibernation,
            # Disable pruning on BOTH arms so island-vs-monolith differs only by the island machinery
            # (the island path force-disables pruning anyway; matching it keeps the comparison fair).
            contact_pruning_tolerance=None,
            enable_self_collision=False,
        ),
        show_viewer=False,
        show_FPS=False,
    )
    plane_friction = 0.01 if args.scene == "hibernation_worst" else None
    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(friction=plane_friction) if plane_friction is not None else None,
    )
    boxes = SCENE_BUILDERS[args.scene](scene, args.size)
    scene.build(n_envs=args.n_envs)
    build_s = time.perf_counter() - t0

    # hibernation_worst: kick every box along +x once; the frictionless plane keeps it sliding.
    if args.scene == "hibernation_worst" and boxes is not None:
        vel = np.zeros((6,), dtype=np.float64)
        vel[0] = 2.0  # 2 m/s, stays well above hibernation_thresh_vel (1e-3) despite min friction
        vel_batch = np.tile(vel, (args.n_envs, 1))
        for box in boxes:
            box.set_dofs_velocity(vel_batch)

    for _ in range(args.settle_steps):
        scene.step()

    t0 = time.perf_counter()
    for _ in range(args.timed_steps):
        scene.step()
    elapsed = time.perf_counter() - t0

    ms_per_step = elapsed / args.timed_steps * 1e3
    per_env_ms = ms_per_step / max(args.n_envs, 1)
    print(
        f"RESULT scene={args.scene} backend={args.backend} n_envs={args.n_envs} size={args.size} "
        f"island={args.island} hib={args.hibernation} solver={args.solver} "
        f"build_s={build_s:.1f} ms_per_step={ms_per_step:.2f} per_env_ms={per_env_ms:.3f}"
    )


if __name__ == "__main__":
    main()
