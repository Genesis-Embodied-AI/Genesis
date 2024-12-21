import argparse
import os
import pickle as pkl

import numpy as np

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu)

    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(2.5, 0.0, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        max_FPS=60,
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        viewer_options=viewer_options,
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    drone = scene.add_entity(
        morph=gs.morphs.Drone(
            file="urdf/drones/cf2x.urdf",
            pos=(0.0, 0, 0.02),
        ),
    )

    ########################## build ##########################
    scene.build()
    
    gs.tools.run_in_another_thread(fn=run_sim, args=(scene, drone, args.vis))
    if args.vis:
        scene.viewer.start()


def run_sim(scene, drone, enable_vis):
    traj = pkl.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fly_traj.pkl"), "rb"))
    for i in range(len(traj)):
        # 14468 is hover rpm
        drone.set_propellels_rpm((1 + 0.05 * traj[i]) * 14468.429183500699)
        scene.step()
    
    if enable_vis:
        scene.viewer.stop()


if __name__ == "__main__":
    main()
