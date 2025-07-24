import argparse

import genesis as gs
import numpy as np
import trimesh
from genesis.sensors import NPZFileWriter, RigidContactForceGridSensor, SensorDataRecorder, VideoFileWriter
from genesis.utils.misc import tensor_to_array
from huggingface_hub import snapshot_download
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
    parser.add_argument("--seconds", "-t", type=float, default=0.5, help="Number of seconds to simulate")
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

    ########################## scene setup ##########################
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=args.dt,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        rigid_options=gs.options.RigidOptions(
            use_gjk_collision=True,
            constraint_timeconst=max(0.01, 2 * args.dt / args.substeps),
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    # define which fingertips we want to add sensors to
    sensorized_link_names = [
        "index_3_tip",
        "middle_3_tip",
        "ring_3_tip",
        "thumb_3_tip",
    ]

    # load the hand .urdf
    asset_path = snapshot_download(
        repo_id="Genesis-Intelligence/assets", allow_patterns="allegro_hand/*", repo_type="dataset"
    )
    hand = scene.add_entity(
        morph=gs.morphs.URDF(
            file=f"{asset_path}/allegro_hand/allegro_hand_right_glb.urdf",
            pos=(0.0, 0.0, 0.1),
            euler=(0.0, -90.0, 180.0),
            fixed=True,  # Fix the base so the whole hand doesn't flop on the ground
            links_to_keep=sensorized_link_names,  # Make sure the links we want to sensorize aren't merged
        ),
        material=gs.materials.Rigid(),
    )

    # Some arbitrary objects to interact with the hand: spheres arranged in a circle
    pos_radius = 0.06
    for i in range(10):
        scene.add_entity(
            gs.morphs.Sphere(
                pos=(pos_radius * np.cos(i * np.pi / 5) + 0.02, pos_radius * np.sin(i * np.pi / 5), 0.3 + 0.04 * i),
                radius=0.02,
            ),
            surface=gs.surfaces.Default(
                color=(0.0, 1.0, 1.0, 0.5),
            ),
        )

    ########################## add sensors ##########################
    data_recorder = SensorDataRecorder(step_dt=args.dt)
    sensors = []
    for link in hand.links:
        if link.name in sensorized_link_names:
            sensor = RigidContactForceGridSensor(entity=hand, link_idx=link.idx, grid_size=(2, 2, 2))
            data_recorder.add_sensor(
                sensor,
                NPZFileWriter(filename=f"{link.name}_contact_forces.npz"),
            )
            sensors.append(sensor)

    # Add camera for visualization
    cam = scene.add_camera(
        res=(1280, 960),
        pos=(0.5, 0.7, 0.7),
        lookat=(0.0, 0.0, 0.1),
        fov=20,
        GUI=args.vis,
    )
    # we can also record the camera video using data_recorder
    data_recorder.add_sensor(cam, VideoFileWriter(filename="hand_test.mp4"))

    ########################## build ##########################
    scene.build(n_envs=args.n_envs)

    dofs_position = [0.1, 0, -0.1, 0.7, 0.6, 0.6, 0.6, 1.0, 0.65, 0.65, 0.65, 1.0, 0.6, 0.6, 0.6, 0.7]
    if args.n_envs > 0:
        dofs_position = [dofs_position] * args.n_envs
    hand.set_dofs_position(np.array(dofs_position))

    data_recorder.start_recording()
    max_observed_force_magnitude = 0.0
    try:
        steps = int(args.seconds / args.dt)
        for _ in tqdm(range(steps), total=steps):
            scene.step()

            scene.clear_debug_objects()
            for sensor in sensors:
                grid_forces = sensor.read()
                grid_force_magnitudes = np.linalg.norm(grid_forces, axis=-1)
                max_observed_force_magnitude = max(max_observed_force_magnitude, np.max(grid_force_magnitudes))
                visualize_grid_sensor(scene, sensor, max_force=0.3)

            data_recorder.step()

    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")

        data_recorder.stop_recording()

        print("Max force recorded:", max_observed_force_magnitude)


if __name__ == "__main__":
    main()
