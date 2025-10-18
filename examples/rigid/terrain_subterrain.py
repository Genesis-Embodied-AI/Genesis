import argparse
import time

import numpy as np
import torch

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
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            constraint_solver=gs.constraint_solver.Newton,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(-5.0, -5.0, 10.0),
            camera_lookat=(5.0, 5.0, 0.0),
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )

    horizontal_scale = 0.25
    vertical_scale = 0.005
    ########################## entities ##########################
    terrain = scene.add_entity(
        morph=gs.morphs.Terrain(
            n_subterrains=(2, 2),
            subterrain_size=(6.0, 6.0),
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            subterrain_types=[
                ["flat_terrain", "random_uniform_terrain"],
                ["pyramid_sloped_terrain", "discrete_obstacles_terrain"],
            ],
        ),
    )
    ball = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(1.0, 1.0, 1.0),
            radius=0.1,
        ),
    )
    ########################## build ##########################
    scene.build(n_envs=100)

    ball.set_pos(torch.cartesian_prod(*(torch.arange(1, 11),) * 2, torch.tensor((1,))))

    (terrain_geom,) = terrain.geoms
    height_field = terrain_geom.metadata["height_field"]
    rows = (horizontal_scale * torch.arange(height_field.shape[0])).reshape((-1, 1)).expand(height_field.shape)
    cols = (horizontal_scale * torch.arange(height_field.shape[1])).reshape((1, -1)).expand(height_field.shape)
    heights = vertical_scale * torch.as_tensor(height_field)
    poss = torch.stack((rows, cols, heights), dim=-1).reshape((-1, 3))

    scene.draw_debug_spheres(poss=poss, radius=0.05, color=(0.0, 0.0, 1.0, 0.7))
    for _ in range(1000):
        scene.step()


if __name__ == "__main__":
    main()
