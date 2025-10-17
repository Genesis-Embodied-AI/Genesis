import argparse
import math
import os

import numpy as np

import genesis as gs
import genesis.utils.geom as gu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    parser.add_argument("--horizon", type=int, default=100 if "PYTEST_VERSION" not in os.environ else 25)
    parser.add_argument("--num_teleports", type=int, default=5)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="info", performance_mode=True)

    dt: float = 2e-2
    particle_size: float = 1e-2

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=10,
        ),
        pbd_options=gs.options.PBDOptions(
            particle_size=particle_size,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
            max_FPS=50,
        ),
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=[0],
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    rigid_material = gs.materials.Rigid(needs_coup=True, coup_friction=0.0)

    # create ground plane
    scene.add_entity(gs.morphs.Plane(), rigid_material)

    # create box
    box_morph = gs.morphs.Box(pos=[0.25, 0.25, 0.25], size=[0.25, 0.25, 0.25])
    box = scene.add_entity(box_morph, rigid_material)

    # create cloth
    cloth_pos = (0.25, 0.25, 0.25 + 0.125 + particle_size)
    cloth_scale = 0.5
    cloth_morph = gs.morphs.Mesh(pos=cloth_pos, scale=cloth_scale, file="meshes/cloth.obj")
    cloth_material = gs.materials.PBD.Cloth()
    cloth_surface = gs.surfaces.Default(color=(0.2, 0.4, 0.8, 1.0))  # , vis_mode="particle")
    cloth = scene.add_entity(cloth_morph, cloth_material, cloth_surface)

    ########################## build ##########################
    scene.build(n_envs=0)

    particles_idx = [0, 1, 2, 3, 4, 5, 6, 7]
    box_link_idx = box.link_start

    cloth.fix_particles_to_link(box_link_idx, particles_idx_local=particles_idx)

    box.set_dofs_velocity([-0.0, 1.0, 0.0], dofs_idx_local=[0, 1, 2])

    for i in range(args.horizon):
        scene.step()

    for j in range(args.num_teleports):
        if j == 2:
            cloth.release_particle(particles_idx)

        new_pos = (0.2 * math.sin(j), 0.2 * math.cos(j), 0.5)
        new_rot = gu.euler_to_quat((-30 * j, -40 * j, 70 * j))
        box.set_pos(new_pos)
        box.set_quat(new_rot)

        for _ in range(args.horizon):
            scene.step()


if __name__ == "__main__":
    main()
