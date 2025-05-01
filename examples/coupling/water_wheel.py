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
            lower_bound=(0.0, 0.0, 0.0),
            upper_bound=(1.0, 1.0, 1.5),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(5.5, 6.5, 3.2),
            camera_lookat=(0.5, 1.5, 1.5),
            camera_fov=35,
            max_FPS=120,
        ),
        show_viewer=args.vis,
        sph_options=gs.options.SPHOptions(
            particle_size=0.02,
        ),
    )

    plane = scene.add_entity(gs.morphs.Plane())
    wheel_0 = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/wheel/fancy_wheel.urdf",
            pos=(0.5, 0.25, 1.6),
            euler=(0, 0, 0),
            fixed=True,
            convexify=False,
        ),
    )

    emitter = scene.add_emitter(
        material=gs.materials.SPH.Liquid(sampler="regular"),
        max_particles=100000,
        surface=gs.surfaces.Glass(
            color=(0.7, 0.85, 1.0, 0.7),
        ),
    )
    scene.build()

    horizon = 500
    for i in range(horizon):
        print(i)
        emitter.emit(
            pos=np.array([0.5, 1.0, 3.5]),
            direction=np.array([0.0, 0, -1.0]),
            speed=5.0,
            droplet_shape="circle",
            droplet_size=0.22,
        )
        scene.step()


if __name__ == "__main__":
    main()
