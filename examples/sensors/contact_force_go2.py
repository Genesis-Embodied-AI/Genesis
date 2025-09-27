import argparse

from tqdm import tqdm

import genesis as gs
from genesis.recorders.plotters import IS_MATPLOTLIB_AVAILABLE, IS_PYQTGRAPH_AVAILABLE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dt", "--timestep", type=float, default=1e-2, help="Simulation time step")
    parser.add_argument("-v", "--vis", action="store_true", default=True, help="Show visualization GUI")
    parser.add_argument("-nv", "--no-vis", action="store_false", dest="vis", help="Disable visualization GUI")
    parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument("-t", "--seconds", type=float, default=2, help="Number of seconds to simulate")
    parser.add_argument("-f", "--force", action="store_true", default=True, help="Use ContactForceSensor (xyz float)")
    parser.add_argument("-nf", "--no-force", action="store_false", dest="force", help="Use ContactSensor (boolean)")

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

    foot_link_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
    go2 = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=(0.0, 0.0, 0.2),
            links_to_keep=foot_link_names,
        )
    )

    for link_name in foot_link_names:
        if args.force:
            force_sensor = scene.add_sensor(
                gs.sensors.ContactForce(
                    entity_idx=go2.idx,
                    link_idx_local=go2.get_link(link_name).idx_local,
                    draw_debug=True,
                )
            )
            if IS_PYQTGRAPH_AVAILABLE:
                force_sensor.start_recording(
                    gs.recorders.PyQtLinePlot(
                        title="Force Sensor Data",
                        labels=["force_x", "force_y", "force_z"],
                    )
                )
            elif IS_MATPLOTLIB_AVAILABLE:
                print("pyqtgraph not found, falling back to matplotlib.")
                force_sensor.start_recording(
                    gs.recorders.MPLLinePlot(
                        title="Force Sensor Data",
                        labels=["force_x", "force_y", "force_z"],
                    )
                )
            else:
                print("matplotlib or pyqtgraph not found, skipping real-time plotting.")
        else:
            contact_sensor = scene.add_sensor(
                gs.sensors.Contact(
                    entity_idx=go2.idx,
                    link_idx_local=go2.get_link(link_name).idx_local,
                    draw_debug=True,
                )
            )

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
