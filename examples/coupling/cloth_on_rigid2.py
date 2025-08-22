from typing import TYPE_CHECKING

import gstaichi as ti

import argparse

import numpy as np
from numpy.typing import ArrayLike

import genesis as gs

import genesis.utils.geom as gu

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_entity import RigidEntity

dt: float = 2e-2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(seed=0, precision="32", logging_level="debug", backend=gs.cpu if args.cpu else gs.gpu)

    particle_size = 1e-2

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
    frictionless_rigid = gs.materials.Rigid(needs_coup=True, coup_friction=0.0)

    plane = scene.add_entity(
        material=frictionless_rigid,
        morph=gs.morphs.Plane(),
    )

    cube: "RigidEntity" = scene.add_entity(
        material=frictionless_rigid,
        morph=gs.morphs.Box(
            pos=(0.25, 0.25, 0.25),
            size=(0.25, 0.25, 0.25),
        ),
    )

    # sphere = scene.add_entity(gs.morphs.Sphere.create((1, 1, 1), 0.5))

    cloth = scene.add_entity(
        material=gs.materials.PBD.Cloth(),
        morph=gs.morphs.Mesh(
            file="meshes/cloth.obj",
            scale=0.5,
            pos=(0.25, 0.25, 0.25 + 0.125 + particle_size),  # offset by particle size
        ),
        surface=gs.surfaces.Default(
            color=(0.2, 0.4, 0.8, 1.0),
            vis_mode="particle",
        ),
    )

    ########################## build ##########################
    scene.build(n_envs=0)

    particles_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    link_idx = cube.links[0].idx
    scene.pbd_solver.set_animate_particles_by_link(particles_idx, link_idx, scene.rigid_solver.links_state)

    horizon = 50
    for i in range(horizon):
        scene.step()

    num_teleports = 5
    for j in range(num_teleports):
        if j == num_teleports - 3:
            scene.pbd_solver.clear_animate_particles_by_link(particles_idx)
        pos = np.zeros(3)
        pos[0] = np.sin(j) * 0.2
        pos[1] = np.cos(j) * 0.2
        pos[2] = 0.5
        cube.set_pos(pos)
        cube.set_quat(gu.euler_to_quat((-30 * j, -40 * j, 70 * j)))
        for i in range(horizon):
            scene.step()


if __name__ == "__main__":
    main()
