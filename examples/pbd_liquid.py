import argparse

import numpy as np

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(seed=0, precision="32", logging_level="debug")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 1.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
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

    for i in range(10000):
        scene.step()


if __name__ == "__main__":
    main()
