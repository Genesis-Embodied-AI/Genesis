import argparse

import numpy as np

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(seed=0, precision="32", logging_level="debug")

    ########################## create a scene ##########################

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=4e-3,
            substeps=10,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-1.0, -1.0, -0.01),
            upper_bound=(1.0, 1.0, 2.0),
            grid_density=64,
            enable_CPIC=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.2, 0.9, 3.5),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=35,
            max_FPS=120,
        ),
        show_viewer=True,
        vis_options=gs.options.VisOptions(
            visualize_mpm_boundary=True,
        ),
    )

    plane = scene.add_entity(
        material=gs.materials.Rigid(),
        morph=gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
    )
    cutter = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/cross_cutter.obj",
            euler=(90, 0, 0),
            scale=0.8,
            pos=(0.0, 0.0, 0.3),
            fixed=True,
            convexify=False,
        ),
        surface=gs.surfaces.Iron(),
    )
    dragon = scene.add_entity(
        material=gs.materials.MPM.Elastic(sampler="pbs-64"),
        morph=gs.morphs.Mesh(
            file="meshes/dragon/dragon.obj",
            scale=0.007,
            euler=(0, 0, 90),
            pos=(0.3, -0.0, 1.3),
        ),
        surface=gs.surfaces.Rough(
            color=(0.6, 1.0, 0.8, 1.0),
            vis_mode="particle",
        ),
    )
    scene.build()

    horizon = 400
    for i in range(horizon):
        scene.step()


if __name__ == "__main__":
    main()
