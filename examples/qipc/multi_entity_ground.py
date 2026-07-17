"""QIPC coupler demo: Franka Panda + free-falling cube + ground with IPC contact.

Demonstrates multi-entity support: a fixed-base Panda arm holds its home pose
while a free-falling cube lands on the ground without penetration.
"""
import argparse

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    gs.init(precision="64", logging_level="info")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0.0, 0.0, -9.81),
        ),
        coupler_options=gs.options.QIPCCouplerOptions(
            contact_enable=True,
            contact_d_hat=0.01,
            init_collision_pair_capacity=5000,
            debug_viewer=args.vis,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    franka = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=(0, 0, 0.5),
        ),
        material=gs.materials.Rigid(
            qipc_abd_kappa=1e8,
            qipc_kappa_pivot=1e7,
            qipc_kappa_axis=1e7,
            qipc_home_qpos=[0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04],
        ),
    )

    cube = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/simple/two_cube_revolute.urdf",
            pos=(0.5, 0.0, 0.5),
            fixed=False,
        ),
        material=gs.materials.Rigid(
            qipc_abd_kappa=1e8,
        ),
    )

    scene.build()

    N_STEPS = 300
    for i in range(N_STEPS):
        franka.control_dofs_position([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04])
        scene.step()

        if i % 50 == 0:
            qpos = franka.get_dofs_position()
            cube_pos = cube.get_pos()
            cube_z = float(cube_pos[2]) if cube_pos.dim() == 1 else float(cube_pos[0, 2])
            print(f"step {i:4d} | franka_q0={float(qpos[0]):.4f} | cube_z={cube_z:.4f}")

    cube_pos = cube.get_pos()
    z = float(cube_pos[2]) if cube_pos.dim() == 1 else float(cube_pos[0, 2])
    print(f"\nFinal cube z = {z:.4f} (should be > 0, IPC no-penetration)")


if __name__ == "__main__":
    main()
