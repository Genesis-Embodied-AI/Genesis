from typing import TYPE_CHECKING

import argparse

import numpy as np
import torch

import genesis as gs

import genesis.utils.geom as gu

from genesis.utils.misc import to_gs_tensor

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_entity import RigidEntity
    from genesis.engine.materials.base import Material
    from genesis.options.surfaces import Surface
    from genesis.options.morphs import Morph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(seed=0, precision="32", logging_level="debug", backend=gs.cpu if args.cpu else gs.gpu)

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
    rigid_material: "Material" = gs.materials.Rigid(needs_coup=True, coup_friction=0.0)

    # create ground plane
    scene.add_entity(gs.morphs.Plane(), rigid_material)

    # create box
    box_pos = [0.25, 0.25, 0.25]
    box_size = [0.25, 0.25, 0.25]
    box_morph: "Morph" = gs.morphs.Box(pos=box_pos, size=box_size)
    box: "RigidEntity" = scene.add_entity(box_morph, rigid_material)

    # create cloth
    cloth_pos = (0.25, 0.25, 0.25 + 0.125 + particle_size)
    cloth_scale = 0.5
    cloth_morph: "Morph" = gs.morphs.Mesh(pos=cloth_pos, scale=cloth_scale, file="meshes/cloth.obj")
    cloth_material: "Material" = gs.materials.PBD.Cloth()
    cloth_surface: "Surface" = gs.surfaces.Default(color=(0.2, 0.4, 0.8, 1.0))  # , vis_mode="particle")
    scene.add_entity(cloth_morph, cloth_material, cloth_surface)

    ########################## build ##########################
    scene.build(n_envs=0)

    particles_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    box_link_idx = box.links[0].idx
    scene.pbd_solver.set_animate_particles_by_link(particles_idx, box_link_idx, scene.rigid_solver.links_state)

    vel = np.array([-0.0, 1.0, 0.0], dtype=np.float32)
    box.set_dofs_velocity(vel, dofs_idx_local=np.arange(3))

    horizon = 50
    for i in range(horizon):
        scene.step()

    num_teleports = 5
    for j in range(num_teleports):
        if j == 2:
            if False:
                # Clearing attachment causes the cloth to vanish. Haven't debugged.
                # All is okay when clearing attachment immeidately, or after 1st teleport.
                # Not okay when clearng attachment after 2+ teleports:
                # scene.pbd_solver.clear_animate_particles_by_link(particles_idx)
                pass
        new_pos = (0.2 * np.sin(j), 0.2 * np.cos(j), 0.5)
        new_rot = gu.euler_to_quat((-30 * j, -40 * j, 70 * j))
        box.set_pos(new_pos)
        box.set_quat(new_rot)
        for i in range(horizon):
            scene.step()

        # validation of particle positions: todo: move to unit test:
        unused_f = 0
        pbd_state: "PBDSolverState" = scene.pbd_solver.get_state(unused_f)

        link_pos = scene.rigid_solver.links[box_link_idx].get_pos()
        link_quat = scene.rigid_solver.links[box_link_idx].get_quat()

        for i_p_ in range(particles_idx.shape[0]):
            i_p = particles_idx[i_p_]
            i_env = 0
            local_pos = scene.pbd_solver.particle_animation_info.local_pos[i_p, i_env].to_numpy()


if __name__ == "__main__":
    main()
