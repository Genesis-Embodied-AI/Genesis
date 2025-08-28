import argparse

from custom_data_handlers import IS_MATPLOTLIB_AVAILABLE, IS_PYQTGRAPH_AVAILABLE, MPLPlotter, PyQtGraphPlotter
from huggingface_hub import snapshot_download
from tqdm import tqdm

import genesis as gs
from genesis.sensors.data_handlers import NPZFileWriter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=1e-2, help="Simulation time step")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument("--vis", "-v", action="store_true", help="Show visualization GUI")
    parser.add_argument("--seconds", "-t", type=float, default=3, help="Number of seconds to simulate")

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
            constraint_timeconst=max(0.01, 2 * args.dt),
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(0.0, 0.0, 0.2),
        ),
        material=gs.materials.Rigid(rho=1.0),  # mass = 1 kg
    )
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
        ),
        material=gs.materials.Rigid(),
    )
    palm = hand.get_link("base_link")

    force_sensor = scene.add_sensor(gs.sensors.ContactForce(entity_idx=palm.idx))

    plotter_cls = PyQtGraphPlotter if IS_PYQTGRAPH_AVAILABLE else (MPLPlotter if IS_MATPLOTLIB_AVAILABLE else None)
    if plotter_cls is not None:
        force_sensor.add_recording(
            gs.options.RecordingOptions(
                handler=plotter_cls(title="Force Sensor Measured Data", labels=["force_x", "force_y", "force_z"])
            )
        )
    else:
        print("pyqtgraph not found, skipping real-time plotting.")

    force_sensor.add_recording(gs.options.RecordingOptions(handler=NPZFileWriter(filename="force_data.npz")))

    scene.build()

    try:
        steps = int(args.seconds / args.dt)
        for _ in tqdm(range(steps), total=steps):
            scene.step()

    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")

        scene.stop_recording()


if __name__ == "__main__":
    main()
