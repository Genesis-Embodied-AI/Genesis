import argparse
import genesis as gs
from genesis.sensors import SensorDataRecorder, RecordingOptions, RigidContactForceGridSensor
from genesis.sensors.data_handlers import VideoFileStreamer, NPZFileWriter
from tqdm import tqdm

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", "-t", type=float, default=1, help="Number of seconds to simulate")
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Time step",
    )
    parser.add_argument("--n_envs", type=int, default=0, help="Number of environments (default: 0)")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument("--vis", "-v", action="store_true", help="Show visualization GUI")

    args = parser.parse_args()

    gs.init(backend=gs.gpu, logging_level=None)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=args.dt,
            gravity=(0, 0, -9.81),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        vis_options=gs.options.VisOptions(
            background_color=(0.6, 0.6, 0.6),
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    hand = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/allegro_hand/allegro_hand_right_glb.urdf",
            pos=(0.0, 0.0, 0.1),
            euler=(0.0, -90.0, 0.0),
            fixed=True,
            merge_fixed_links=False,
        ),
        material=gs.materials.Rigid(gravity_compensation=1.0),
    )

    box = scene.add_entity(
        gs.morphs.Box(pos=(-0.02, 0.0, 0.22), size=(0.09, 0.12, 0.03)),
        # gs.morphs.Box(pos=(0, 0, 0.15), size=(0.06, 0.09, 0.03)),
        surface=gs.surfaces.Default(
            color=(1.0, 0.4, 0.0, 0.5),
        ),
    )

    steps = int(args.seconds / args.dt)
    cam = scene.add_camera(res=(1080, 720), pos=(0.5, 0.7, 0.7), lookat=(0.0, 0.0, 0.1), fov=20, GUI=args.vis)

    # ---- post-build setup ----
    scene.build(n_envs=args.n_envs)

    dofs_position = [0.1, 0, -0.1, 0.7, 0.6, 0.6, 0.6, 1.0, 0.65, 0.65, 0.65, 1.0, 0.6, 0.6, 0.6, 0.7]
    if args.n_envs > 0:
        dofs_position = [dofs_position] * args.n_envs
    hand.set_dofs_position(np.array(dofs_position))

    data_recorder = SensorDataRecorder(step_dt=args.dt)
    data_recorder.add_sensor(
        cam, RecordingOptions(handler=VideoFileStreamer(filename="hand_test.mp4", fps=1 / args.dt))
    )
    sensors = []
    for link in hand.links:
        if "tip" in link.name:
            print("Adding sensor for link:", link.name)
            sensor = RigidContactForceGridSensor(entity=hand, link_idx=link.idx, grid_size=(4, 4, 2))
            data_recorder.add_sensor(
                sensor, RecordingOptions(handler=NPZFileWriter(filename=f"{link.name}_contact_forces.npz"))
            )
            sensors.append(sensor)
    print("Joints:", [joint.name for joint in hand.joints])

    data_recorder.start_recording()

    # ---- run simulation ----
    try:
        for _ in tqdm(range(steps), total=steps):
            scene.step()
            data_recorder.step()

    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")

        data_recorder.stop_recording()


if __name__ == "__main__":
    main()
