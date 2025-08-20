import gstaichi as ti

import argparse

import numpy as np
from numpy.typing import ArrayLike


import genesis as gs

dt: float = 2e-3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(seed=0, precision="32", logging_level="debug", backend=gs.cpu if args.cpu else gs.gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=10,
        ),
        pbd_options=gs.options.PBDOptions(
            particle_size=1e-2,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=[0],
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    frictionless_rigid = gs.materials.Rigid(needs_coup=True, coup_friction=0.0)

    plane = scene.add_entity(
        material=frictionless_rigid,
        morph=gs.morphs.Plane(),
    )

    cube = scene.add_entity(
        material=frictionless_rigid,
        morph=gs.morphs.Box(
            pos=(0.25, 0.75, 0.2),
            size=(0.2, 0.2, 0.2),
            euler=(30, 40, 0),
            fixed=False,
        ),
    )

    # sphere = scene.add_entity(gs.morphs.Sphere.create((1, 1, 1), 0.5))

    cloth = scene.add_entity(
        material=gs.materials.PBD.Cloth(),
        morph=gs.morphs.Mesh(
            file="meshes/cloth.obj",
            scale=1.0,
            pos=(0.5, 0.5, 0.5),
            euler=(180.0, 0.0, 0.0),
        ),
        surface=gs.surfaces.Default(
            color=(0.2, 0.4, 0.8, 1.0),
            vis_mode="particle",
        ),
    )

    ########################## build ##########################
    scene.build(n_envs=0)

    # scene.pbd_solver.fix_particle(0, 0)
    # scene.pbd_solver.fix_particle(1, 0)
    # scene.pbd_solver.fix_particle(2, 0)
    # scene.pbd_solver.fix_particle(3, 0)
    # scene.pbd_solver.fix_particle(4, 0)
    # scene.pbd_solver.fix_particle(5, 0)
    # scene.pbd_solver.fix_particle(6, 0)
    # scene.pbd_solver.fix_particle(7, 0)

    # scene.pbd_solver.set_animate_particles_by_link(np.array([0, 1, 2, 3, 4, 5, 6, 7]), cube.links[0].idx, scene.rigid_solver.links_state)
    particles_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    particles_idx = np.array([0])
    link_idx = cube.links[0].idx
    scene.pbd_solver.set_animate_particles_by_link(particles_idx, link_idx, scene.rigid_solver.links_state)

    horizon = 500

    for i in range(horizon):
        scene.step()

        # for i in range(8):
        #     pos: gs.ti_vec3 =scene.pbd_solver.get_particle_position_ref(i, 0)
        #     pos_copy = ti.math.vec3(pos)
        #     pos_copy[2] += 0.5 * dt  # this is a reference !?
        #     scene.pbd_solver.set_particle_position_vec(i, pos_copy, 0)


if __name__ == "__main__":
    main()
