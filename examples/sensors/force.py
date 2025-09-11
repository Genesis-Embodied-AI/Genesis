import argparse

from tqdm import tqdm

import genesis as gs
from genesis.recorders.plotters import IS_MATPLOTLIB_AVAILABLE, IS_PYQTGRAPH_AVAILABLE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dt", "--timestep", type=float, default=1e-2, help="Simulation time step")
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI", default=True)
    parser.add_argument("-nv", "--no-vis", action="store_false", dest="vis", help="Disable visualization GUI")
    parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument("-t", "--seconds", type=float, default=2, help="Number of seconds to simulate")

    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level=None)

    ########################## scene setup ##########################
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=args.timestep),
        rigid_options=gs.options.RigidOptions(
            use_gjk_collision=True,
            constraint_timeconst=max(0.01, 2 * args.timestep),
        ),
        vis_options=gs.options.VisOptions(show_world_frame=True),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(0.0, 0.0, 0.2),
        ),
        material=gs.materials.Rigid(rho=1.0),
    )
    # load the hand .urdf
    hand = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/shadow_hand/shadow_hand.urdf",
            pos=(0.0, -0.3, 0.1),
            euler=(-90.0, 0.0, 0.0),
            fixed=True,  # Fix the base so the whole hand doesn't flop on the ground
        ),
        material=gs.materials.Rigid(),
    )
    palm = hand.get_link("palm")

    force_sensor = scene.add_sensor(gs.sensors.ContactForce(entity_idx=hand.idx, link_idx_local=palm.idx_local))

    labels = ["force_x", "force_y", "force_z"]
    if IS_PYQTGRAPH_AVAILABLE:
        force_sensor.start_recording(gs.recorders.PyQtPlot(title="Force Sensor Measured Data", labels=labels))
    elif IS_MATPLOTLIB_AVAILABLE:
        print("pyqtgraph not found, falling back to matplotlib.")
        force_sensor.start_recording(gs.recorders.MPLPlot(title="Force Sensor Measured Data", labels=labels))
    else:
        print("matplotlib or pyqtgraph not found, skipping real-time plotting.")

    force_sensor.start_recording(gs.recorders.NPZFile(filename="force_data.npz", save_on_reset=True))

    scene.build()

    try:
        steps = int(args.seconds / args.timestep)
        for _ in tqdm(range(steps)):
            scene.step()

    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")

        scene.stop_recording()


if __name__ == "__main__":
    main()
