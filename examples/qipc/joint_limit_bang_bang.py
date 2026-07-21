"""QIPC demo: bang-bang velocity control with joint limits.

A two-cube robot with a single revolute joint is driven by alternating
positive/negative velocity commands. The joint should bounce between its limits.

Usage:
    uv run python examples/qipc/joint_limit_bang_bang.py -v
"""
import argparse
import math

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("--joint-type", choices=["revolute", "prismatic"], default="revolute")
    args = parser.parse_args()

    import xml.etree.ElementTree as ET

    joint_type = args.joint_type
    if joint_type == "revolute":
        limits = (-0.5, 0.5)
    else:
        limits = (-0.3, 0.3)

    mjcf = ET.Element("mujoco", model=f"two_cube_{joint_type}")
    ET.SubElement(mjcf, "compiler", angle="radian")
    worldbody = ET.SubElement(mjcf, "worldbody")
    base = ET.SubElement(worldbody, "body", name="base")
    ET.SubElement(base, "geom", type="box", size="0.05 0.05 0.05")
    ET.SubElement(base, "inertial", mass="1.0", pos="0 0 0", diaginertia="0.00667 0.00667 0.00667")
    child = ET.SubElement(base, "body", name="moving", pos="0.1 0 0")
    ET.SubElement(child, "geom", type="box", size="0.05 0.05 0.05", pos="0.1 0 0")
    ET.SubElement(child, "inertial", mass="1.0", pos="0 0 0", diaginertia="0.00667 0.00667 0.00667")
    mj_type = "hinge" if joint_type == "revolute" else "slide"
    axis = "0 1 0" if joint_type == "revolute" else "1 0 0"
    lo, hi = limits
    ET.SubElement(child, "joint", name="joint1", type=mj_type, axis=axis, range=f"{lo} {hi}")
    mjcf_content = ET.tostring(mjcf, encoding="unicode")

    gs.init(precision="64", logging_level="info")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, 0.0)),
        coupler_options=gs.options.QIPCCouplerOptions(
            contact_enable=False,
            debug_viewer=args.vis,
        ),
        show_viewer=args.vis,
    )

    robot = scene.add_entity(
        morph=gs.morphs.MJCF(file=mjcf_content, pos=(0, 0, 0.5)),
        material=gs.materials.Rigid(
            qipc_abd_kappa=1e8,
            qipc_kappa_pivot=1e5,
            qipc_kappa_axis=1e5,
            qipc_default_kp=0.0,
            qipc_default_kv=100.0,
        ),
    )

    scene.build()

    V_MAX = 2.0
    HALF_PERIOD = 60
    NUM_OSCILLATIONS = 3
    total_steps = 2 * HALF_PERIOD * NUM_OSCILLATIONS

    for step in range(total_steps):
        phase = (step // HALF_PERIOD) % 2
        vel_target = V_MAX if phase == 0 else -V_MAX
        robot.control_dofs_velocity(vel_target, dofs_idx_local=-1)
        scene.step()

        if step % 20 == 0:
            pos = float(robot.get_dofs_position(dofs_idx_local=-1)[0])
            print(f"step {step:4d} | vel_cmd={vel_target:+.1f} | theta={pos:.4f} | limits=[{lo}, {hi}]")

    print(f"\nDone. Joint limits: [{lo}, {hi}]")


if __name__ == "__main__":
    main()
