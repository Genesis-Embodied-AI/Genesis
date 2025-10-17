import os
import argparse

import numpy as np
import torch
from tqdm import tqdm

import genesis as gs

SCENE_POS = (0.5, 0.5, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver", choices=["explicit", "implicit"], default="implicit", help="FEM solver type (default: implicit)"
    )
    parser.add_argument("--dt", type=float)
    parser.add_argument("--substeps", type=int)
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--vis", "-v", action="store_true", default=False)

    args = parser.parse_args()
    args.seconds = 0.01 if "PYTEST_VERSION" in os.environ else args.seconds

    if args.solver == "explicit":
        dt = args.dt if args.dt is not None else 1e-4
        substeps = args.substeps if args.substeps is not None else 5
    else:  # implicit
        dt = args.dt if args.dt is not None else 1e-3
        substeps = args.substeps if args.substeps is not None else 1

    gs.init(backend=gs.gpu, performance_mode=True)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=substeps,
            gravity=(0, 0, -9.81),
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=args.solver == "implicit",
            enable_vertex_constraints=True,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    blob = scene.add_entity(
        morph=gs.morphs.Sphere(pos=tuple(map(sum, zip(SCENE_POS, (-0.3, -0.3, 0)))), radius=0.1),
        material=gs.materials.FEM.Elastic(E=1.0e4, nu=0.45, rho=1000.0, model="linear_corotated"),
    )

    cube = scene.add_entity(
        morph=gs.morphs.Box(pos=tuple(map(sum, zip(SCENE_POS, (0.3, 0.3, 0)))), size=(0.2, 0.2, 0.2)),
        material=gs.materials.FEM.Elastic(E=1.0e6, nu=0.45, rho=1000.0, model="linear_corotated"),
    )

    video_fps = 1 / dt
    max_fps = 100
    frame_interval = max(1, int(video_fps / max_fps)) if max_fps > 0 else 1
    print(f"video_fps: {video_fps}, frame_interval: {frame_interval}")

    cam = scene.add_camera(
        res=(640, 480),
        pos=(-2.0, 3.0, 2.0),
        lookat=tuple(map(sum, zip(SCENE_POS, (0.0, 0.0, -0.8)))),
        fov=30,
    )

    scene.build()
    cam.start_recording()

    pinned_idx = [0]
    circle_radius = 0.3

    circle_period = 10.0
    angle_step = 2 * np.pi * dt / circle_period
    current_angle = 0.0

    initial_vertex_pos = cube.init_positions[pinned_idx]
    circle_center = initial_vertex_pos - torch.tensor(
        [-circle_radius * np.cos(current_angle), -circle_radius * np.sin(current_angle), 0.0],
        device=cube.init_positions.device,
        dtype=cube.init_positions.dtype,
    )

    def get_next_circle_position():
        """Get next position on circular path with incremental step."""
        nonlocal current_angle
        offset = torch.tensor(
            [-circle_radius * np.cos(current_angle), -circle_radius * np.sin(current_angle), 0.0],
            device=cube.init_positions.device,
            dtype=cube.init_positions.dtype,
        )
        current_angle += angle_step
        return circle_center + offset

    debug_circle = None
    total_steps = int(args.seconds / dt)

    try:
        target_positions = blob.init_positions[pinned_idx]
        scene.draw_debug_spheres(poss=target_positions, radius=0.02, color=(1, 0, 1, 0.8))
        blob.set_vertex_constraints(
            verts_idx=pinned_idx,
            target_poss=target_positions,
            is_soft_constraint=True,
            stiffness=1e4,
        )

        target_positions = get_next_circle_position()
        debug_circle = scene.draw_debug_spheres(poss=target_positions, radius=0.02, color=(0, 1, 0, 0.8))
        cube.set_vertex_constraints(
            verts_idx=pinned_idx,
            target_poss=target_positions,
        )

        for step in tqdm(range(total_steps), total=total_steps):
            if debug_circle is not None:
                scene.clear_debug_object(debug_circle)

            new_pos = get_next_circle_position()
            debug_circle = scene.draw_debug_spheres(poss=new_pos, radius=0.02, color=(0, 1, 0, 0.8))
            cube.update_constraint_targets(
                verts_idx=pinned_idx,
                target_poss=new_pos,
            )

            scene.step()

            if step % frame_interval == 0:
                cam.render()

    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")

        actual_fps = video_fps / frame_interval
        video_filename = f"fem_hard_soft_{args.solver}_dt={dt}_substeps={substeps}.mp4"
        cam.stop_recording(save_to_filename=video_filename, fps=actual_fps)
        gs.logger.info(f"Saved video to {video_filename}")


if __name__ == "__main__":
    main()
