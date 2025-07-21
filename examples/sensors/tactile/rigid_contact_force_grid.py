import argparse

import genesis as gs
import numpy as np
import trimesh
from genesis.sensors import NPZFileWriter, RigidContactForceGridSensor, SensorDataRecorder, VideoFileWriter
from genesis.utils.misc import tensor_to_array
from tqdm import tqdm


def visualize_grid_sensor(scene: gs.Scene, sensor: RigidContactForceGridSensor, min_force=0.0, max_force=1.0):
    """
    Draws debug objects on scene to visualize the contact grid sensor data.

    Note: This method is very inefficient and purely for demo/debugging purposes.
    This processes the grid data from the sensor, which means the transformation from global-> local frame is undone to
    revert back to into global frame to draw the debug objects.
    """
    grid_data = sensor.read()

    link_pos = tensor_to_array(scene._sim.rigid_solver.get_links_pos(links_idx=sensor.link_idx).squeeze(axis=1))
    link_quat = tensor_to_array(scene._sim.rigid_solver.get_links_quat(links_idx=sensor.link_idx).squeeze(axis=1))

    sensor_dims = sensor.max_bounds - sensor.min_bounds
    grid_cell_size = sensor_dims / np.array(sensor.grid_size)

    debug_objs = []

    for x in range(grid_data.shape[1]):
        for y in range(grid_data.shape[2]):
            for z in range(grid_data.shape[3]):
                force = grid_data[0, x, y, z]
                force_magnitude = np.linalg.norm(force)

                color_intensity = np.clip(force_magnitude, min_force, max_force) / max_force
                color = np.array([color_intensity, 0.0, 1.0 - color_intensity, 0.4])

                mesh = trimesh.creation.box(extents=grid_cell_size)
                mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=np.tile(color, [len(mesh.vertices), 1]))

                local_pos = sensor.min_bounds + (np.array([x, y, z]) + 0.5) * grid_cell_size

                T = trimesh.transformations.quaternion_matrix(link_quat)
                T[:3, 3] = link_pos

                local_T = np.eye(4)
                local_T[:3, 3] = local_pos
                final_T = T @ local_T

                debug_objs.append(scene.draw_debug_mesh(mesh, T=final_T))

    return debug_objs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", "-t", type=float, default=1, help="Number of seconds to simulate")
    parser.add_argument("--dt", type=float, default=1e-2, help="Simulation time step")
    parser.add_argument(
        "--substeps",
        type=int,
        default=1,
        help="Number of substeps",
    )
    parser.add_argument("--n_envs", type=int, default=0, help="Number of environments (default: 0)")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument("--vis", "-v", action="store_true", help="Show visualization GUI")
    parser.add_argument(
        "--debug", "-d", action="store_true", default=True, help="Draw the contact grid sensor debug objects"
    )

    args = parser.parse_args()

    gs.init(backend=gs.gpu, logging_level=None)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=args.dt,
            substeps=args.substeps,
            gravity=(0, 0, -9.81),
        ),
        rigid_options=gs.options.RigidOptions(
            use_gjk_collision=True,
            constraint_timeconst=max(0.01, 2 * args.dt / args.substeps),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    block = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.2),
            euler=(0, 10, 0),
            size=(2.0, 2.0, 0.05),
            visualization=not args.debug,
        ),
        material=gs.materials.Rigid(
            gravity_compensation=1.0,
        ),
    )

    sphere1 = scene.add_entity(
        gs.morphs.Sphere(pos=(0.7, 0.4, 0.4), radius=0.1),
        surface=gs.surfaces.Default(
            color=(1.0, 0.0, 0.0, 1.0),
        ),
    )
    sphere2 = scene.add_entity(
        gs.morphs.Sphere(pos=(-0.5, -0.3, 0.5), radius=0.1),
        surface=gs.surfaces.Default(
            color=(0.0, 1.0, 0.0, 1.0),
        ),
    )
    sphere3 = scene.add_entity(
        gs.morphs.Sphere(pos=(-0.2, 0.7, 0.6), radius=0.1),
        surface=gs.surfaces.Default(
            color=(0.0, 0.0, 1.0, 1.0),
        ),
    )

    steps = int(args.seconds / args.dt)
    cam = scene.add_camera(res=(640, 480), pos=(-2, 3, 1.5), lookat=(0.0, 0.0, 0.1), fov=30, GUI=args.vis)
    grid_sensor = RigidContactForceGridSensor(block, grid_size=(4, 4, 2))

    scene.build(n_envs=args.n_envs)

    data_recorder = SensorDataRecorder()
    data_recorder.add_sensor(cam, VideoFileWriter(filename="grid_test.mp4"))
    data_recorder.add_sensor(grid_sensor, NPZFileWriter(filename="grid_test.npz"))
    data_recorder.start_recording()

    try:
        for _ in tqdm(range(steps), total=steps):
            scene.step()

            if args.debug:
                scene.clear_debug_objects()
                visualize_grid_sensor(scene, grid_sensor)

            data_recorder.step()

    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")

        data_recorder.stop_recording()


if __name__ == "__main__":
    main()
