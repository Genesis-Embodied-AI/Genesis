"""Islands + hibernation showcase: a grid of ducks is dropped onto the ground and left to settle.

While the ducks are falling and colliding every body is awake and the full constraint solve runs each step. Once a
duck comes to rest it hibernates: its island is skipped by forward kinematics, forward dynamics, integration, and the
constraint solve. As the ducks settle, the simulation step rate climbs sharply - the payoff of the islands +
hibernation pipeline.

The step rate is streamed to a live plot through the recorder facilities and the scene is recorded to a video.
"""

import argparse
import os
import time

import genesis as gs
from genesis.utils.misc import qd_to_numpy
from genesis.utils.tools import FPSTracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--grid", type=int, default=12, help="Ducks per horizontal side; 1 to 20.")
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer.")
    parser.add_argument("-r", "--record", action="store_true", help="Record the scene to hibernation_ducks.mp4.")
    args = parser.parse_args()
    if not 1 <= args.grid <= 20:
        raise ValueError(f"--grid must be between 1 and 20 (got {args.grid}).")

    # Hibernation runs on field storage (performance_mode) and shows its benefit on CPU, where skipping sleeping
    # islands turns directly into a higher serial step rate.
    gs.init(backend=gs.cpu, performance_mode=True)

    # A 6 ms step is the largest at which the convex-hull ducks still settle to true rest; a coarser step leaves them
    # buzzing on their curved bellies and they never drop below the hibernation velocity threshold.
    dt = 6e-3
    n_side = args.grid
    n_layers = 3
    n_ducks = n_side * n_side * n_layers

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            gravity=(0.0, 0.0, -9.81),
        ),
        rigid_options=gs.options.RigidOptions(
            use_contact_island=True,
            use_hibernation=True,
            max_collision_pairs=80 * n_ducks,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    # A wide footprint that is only a few ducks deep: dropped from a modest height the ducks scatter and settle
    # mostly separated, so each forms its own island and hibernates independently. A deep pile would couple them
    # into large islands that stay awake, and a high drop would interpenetrate the hulls.
    spacing = 0.3
    layer_gap = 0.15
    z_floor = 0.35
    offset = 0.5 * (n_side - 1) * spacing
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_layers):
                scene.add_entity(
                    gs.morphs.Mesh(
                        file="meshes/duck.obj",
                        scale=0.02,
                        pos=(i * spacing - offset, j * spacing - offset, z_floor + k * layer_gap),
                        euler=(90.0, 0.0, (i * 37 + j * 53 + k * 71) % 360),
                    ),
                    vis_mode="collision",
                )

    camera = None
    if args.record:
        span = n_side * spacing
        camera = scene.add_camera(
            res=(1280, 720),
            pos=(0.8 * span, 0.8 * span, 0.5 * span),
            lookat=(0.0, 0.0, -0.1 * span),
            fov=42,
            GUI=False,
        )

    # The step rate climbs as ducks fall asleep; stream it to a live line plot.
    fps_tracker = FPSTracker(n_envs=0, alpha=0.0)
    step_rate = [0.0]

    def step_rate_data():
        return {"step_rate": [step_rate[0]]}

    scene.start_recording(
        step_rate_data,
        gs.recorders.MPLLinePlot(title="Simulation step rate", labels={"step_rate": ["steps/s"]}, history_length=10000),
    )

    scene.build(n_envs=1)

    # camera.start_recording stores each rendered frame and encodes them all at stop_recording, so rendering never
    # enters the timed region and the reported step rate stays the physics-only rate.
    if args.record:
        camera.start_recording()
    sim_clock = 0.0
    n_steps = int(7.0 / dt) if "PYTEST_VERSION" not in os.environ else 5
    render_every = max(1, round((1.0 / dt) / 30.0))
    for i_step in range(n_steps):
        tic = time.perf_counter()
        scene.step()
        sim_clock += time.perf_counter() - tic
        measured = fps_tracker.step(sim_clock)
        if measured is not None:
            step_rate[0] = measured
        if args.record and i_step % render_every == 0:
            camera.render()
    if args.record:
        camera.stop_recording(save_to_filename="hibernation_ducks.mp4", fps=30)

    n_asleep = int(qd_to_numpy(scene.rigid_solver.entities_state.is_hibernated, transpose=True).sum())
    gs.logger.info(f"{n_asleep}/{n_ducks} ducks hibernated; final step rate {step_rate[0]:,.0f} steps/s.")


if __name__ == "__main__":
    main()
