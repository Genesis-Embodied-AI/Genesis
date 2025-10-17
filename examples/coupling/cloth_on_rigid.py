import argparse
import os

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="info")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
            substeps=10,
        ),
        pbd_options=gs.options.PBDOptions(
            particle_size=1e-2,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0.0, 1.0),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=40,
        ),
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=[1],
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    frictionless_rigid = gs.materials.Rigid(
        needs_coup=True,
        coup_friction=0.0,
    )

    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
        material=frictionless_rigid,
    )

    obj = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=0.2,
            pos=(0.0, 0.0, 0.0),
            fixed=True,
        ),
        material=frictionless_rigid,
    )

    cloth = scene.add_entity(
        material=gs.materials.PBD.Cloth(),
        morph=gs.morphs.Mesh(
            file="meshes/cloth.obj",
            scale=1.0,
            pos=(0.0, 0.0, 0.3),
            euler=(180.0, 0.0, 0.0),
        ),
        surface=gs.surfaces.Default(
            color=(0.2, 0.4, 0.8, 1.0),
        ),
    )

    ########################## build ##########################
    scene.build(n_envs=2)

    horizon = 500 if "PYTEST_VERSION" not in os.environ else 5

    for i in range(horizon):
        scene.step()


if __name__ == "__main__":
    main()
