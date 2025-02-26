import argparse

import numpy as np

import genesis as gs


def run_sim(scene, enable_vis):
    from time import time

    t_prev = time()
    i = 0
    while True:
        i += 1

        scene.step()

        t_now = time()
        print(1 / (t_now - t_prev), "FPS")
        t_prev = t_now
        if i > 1000:
            break

    if enable_vis:

        scene.viewer.stop()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu, seed=0, precision="32", logging_level="debug")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
        ),
        viewer_options=gs.options.ViewerOptions(
            res = (2560,1664),
            camera_pos=(3.5, 5.0, 2.5),  # Move the camera back further
            camera_lookat=(0.0, 2.0, 1.5),
            camera_fov=40,
        ),
        show_viewer=args.vis,
        pbd_options=gs.options.PBDOptions(
            lower_bound=(0.0, 0.0, 0.0),
            upper_bound=(1.0, 1.0, 1.0),
            max_density_solver_iterations=10,
            max_viscosity_solver_iterations=1,
        ),
    )

    ########################## entities ##########################

    liquid = scene.add_entity(
        material=gs.materials.PBD.Liquid(rho=1.0, density_relaxation=1.0, viscosity_relaxation=0.0, sampler="regular"),
        morph=gs.morphs.Box(lower=(0.2, 0.1, 0.1), upper=(0.4, 0.3, 0.5)),
    )
    scene.build()

    gs.tools.run_in_another_thread(fn=run_sim, args=(scene, args.vis))
    if args.vis:
        scene.viewer.start()


if __name__ == "__main__":
    main()
