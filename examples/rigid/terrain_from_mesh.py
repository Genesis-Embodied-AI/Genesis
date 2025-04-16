import genesis as gs
import numpy as np
import argparse
from genesis.utils.terrain import mesh_to_heightfield


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu if args.cpu else gs.gpu)

    ########################## create a scene ##########################
    scene = gs.Scene(
        show_viewer=True,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -50, 0),
            camera_lookat=(0, 0, 0),
        ),
    )

    horizontal_scale = 2.0
    path_terrain = "./genesis/assets/meshes/terrain_45.obj"
    hf_terrain, xs, ys = mesh_to_heightfield(path_terrain, spacing=horizontal_scale, oversample=3)
    print("hf_terrain", hf_terrain.shape, np.max(hf_terrain))

    # default heightfield starts at 0, 0, 0
    # translate to the center of the mesh
    translation = np.array([np.nanmin(xs), np.nanmin(ys), 0])

    terrain_heightfield = scene.add_entity(
        morph=gs.morphs.Terrain(
            horizontal_scale=horizontal_scale,
            vertical_scale=1.0,
            height_field=hf_terrain,
            pos=translation,
        ),
        vis_mode="collision",
    )

    terrain_ori = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=path_terrain,
            fixed=True,
            convexify=False,
        ),
    )

    ball = scene.add_entity(
        gs.morphs.Sphere(
            pos=(10, 15, 18),
            radius=2,
        ),
        vis_mode="collision",
    )

    scene.build()

    for i in range(2000):
        scene.step()


if __name__ == "__main__":
    main()
