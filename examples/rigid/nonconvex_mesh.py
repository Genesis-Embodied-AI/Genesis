import argparse

import genesis as gs


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu if args.cpu else gs.gpu)

    ########################## create a scene ##########################
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=args.vis,
        rigid_options=gs.options.RigidOptions(),
    )

    ########################## entities ##########################
    tank = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/tank.obj",
            convexify=False,
            scale=5.0,
            fixed=True,
            euler=(90, 0, 0),
        ),
    )
    ball = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.1,
            pos=(0.0, 0.0, 1.0),
        ),
    )

    ########################## build ##########################
    scene.build()
    for i in range(1000):
        scene.step()


if __name__ == "__main__":
    main()
