import argparse
import time

import numpy as np
import torch

import genesis as gs


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(seed=0, backend=gs.cpu if args.cpu else gs.gpu)

    ########################## create a scene ##########################

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(-5.0, -5.0, 10.0),
            camera_lookat=(5.0, 5.0, 0.0),
            camera_fov=40,
        ),
        show_viewer=args.vis,
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            constraint_solver=gs.constraint_solver.Newton,
        ),
        vis_options=gs.options.VisOptions(
            # geom_type='sdf',
        ),
    )

    horizontal_scale = 0.25
    vertical_scale = 0.005
    height_field = np.zeros([40, 40])
    heights_range = np.arange(-10, 20, 10)
    height_field[5:35, 5:35] = 200 + np.random.choice(heights_range, (30, 30))
    ########################## entities ##########################
    terrain = scene.add_entity(
        morph=gs.morphs.Terrain(
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            height_field=height_field,
            name="example",
            # from_stored=True,
        ),
    )
    ########################## build ##########################
    scene.build(n_envs=1)

    height_field = terrain.geoms[0].metadata["height_field"]
    rows = horizontal_scale * torch.arange(0, height_field.shape[0], 1, device="cuda").unsqueeze(1).repeat(
        1, height_field.shape[1]
    ).unsqueeze(-1)
    cols = horizontal_scale * torch.arange(0, height_field.shape[1], 1, device="cuda").unsqueeze(0).repeat(
        height_field.shape[0], 1
    ).unsqueeze(-1)
    heights = vertical_scale * torch.tensor(height_field, device="cuda").unsqueeze(-1)

    poss = torch.cat([rows, cols, heights], dim=-1).reshape(-1, 3)
    scene.draw_debug_spheres(poss=poss, radius=0.05, color=(0, 0, 1, 0.7))
    for _ in range(1000):
        time.sleep(0.5)
        scene.step()


if __name__ == "__main__":
    main()
