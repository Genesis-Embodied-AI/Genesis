"""QIPC coupler demo: Franka Panda 7-DOF holding its home pose via PD control."""
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
            contact_enable=False,
            debug_viewer=args.vis,
        ),
        show_viewer=args.vis,
    )

    panda = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=(0, 0, 0),
        ),
        material=gs.materials.Rigid(
            qipc_abd_kappa=1e8,
            qipc_kappa_pivot=1e7,
            qipc_kappa_axis=1e7,
            qipc_home_qpos=[0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04],
        ),
    )

    scene.build()

    N_STEPS = 500
    for i in range(N_STEPS):
        panda.control_dofs_position([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04])
        scene.step()

        if i % 50 == 0:
            qpos = panda.get_dofs_position()
            print(f"step {i:4d} | t={scene.sim.cur_t:.2f}s | qpos[0]={float(qpos[0]):.4f}")


if __name__ == "__main__":
    main()
