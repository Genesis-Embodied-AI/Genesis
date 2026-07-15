"""QIPC coupler demo: two-cube revolute joint with sinusoidal PD tracking."""
import argparse
import math

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

    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/simple/two_cube_revolute.urdf",
            pos=(0, 0, 0.3),
            fixed=True,
        ),
        material=gs.materials.Rigid(
            qipc_abd_kappa=1e8,
            qipc_kappa_pivot=1e5,
            qipc_kappa_axis=1e5,
            qipc_default_kp=500.0,
            qipc_default_kv=50.0,
        ),
    )

    scene.build()

    FREQ = 0.5
    AMPLITUDE = 0.8
    N_STEPS = 500

    for i in range(N_STEPS):
        t = scene.sim.cur_t
        target = AMPLITUDE * math.sin(2 * math.pi * FREQ * t)
        robot.control_dofs_position(target)
        scene.step()

        if i % 50 == 0:
            qpos = robot.get_dofs_position()
            print(f"step {i:4d} | t={t:.3f}s | target={target:.4f} | qpos={float(qpos[0]):.4f}")


if __name__ == "__main__":
    main()
