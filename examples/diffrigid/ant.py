import argparse

import numpy as np

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-n", "--n_envs", type=int, default=49)
    args = parser.parse_args()

    args.vis = True
    args.n_envs = 1

    ########################## init ##########################
    gs.init(backend=gs.cpu)

    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=30,
        max_FPS=60,
    )

    scene = gs.Scene(
        viewer_options=viewer_options,
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
    )
    ant = scene.add_entity(
        gs.morphs.MJCF(file="xml/humanoid.xml"),
    )

    ########################## build ##########################
    scene.build(n_envs=args.n_envs, env_spacing=(1, 1))

    for i in range(10000):
        scene.step()


if __name__ == "__main__":
    main()
