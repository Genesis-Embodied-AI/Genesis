import argparse
import numpy as np
import genesis as gs

from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.gpu)

    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(0, -3.5, 2.5),
        camera_lookat=(0.0, 0.0, 1.0),
        camera_fov=40,
        max_FPS=60,
    )

    scene = gs.Scene(
        viewer_options=viewer_options,
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            pos=(0, 0, 1.0),
            size=(0.2, 0.2, 0.2),
        ),
    )
    ########################## build ##########################
    scene.build()

    for solver in scene.sim.solvers:
        if not isinstance(solver, RigidSolver):
            continue
        rigid_solver = solver

    link_idx = [
        1,
    ]
    rotation_direction = 1
    for i in range(1000):
        cube_pos = rigid_solver.get_links_pos(link_idx)
        cube_pos[:, 2] -= 1
        force = -100 * cube_pos
        rigid_solver.apply_links_external_force(
            force=force,
            links_idx=link_idx,
        )

        torque = [
            [0, 0, rotation_direction * 5],
        ]
        rigid_solver.apply_links_external_torque(
            torque=torque,
            links_idx=link_idx,
        )

        scene.step()

        if (i + 50) % 100 == 0:
            rotation_direction *= -1


if __name__ == "__main__":
    main()
