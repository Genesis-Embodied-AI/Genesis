"""A rolling ball coasting to rest under rolling friction, the everyday behavior a point contact cannot produce.

Two balls are launched rolling side by side at the same speed. Sliding friction does no work on a rolling contact,
so the zero-coefficient ball coasts forever; the other carries a rolling friction coefficient and slows to a stop
within a couple of meters, like a real ball on a floor. The elliptic cone brakes it at the exact Coulomb bound, a
constant deceleration of (5/7) * friction_rolling * g / r. Marker bars show the rotation in the viewer.
"""

import argparse
import xml.etree.ElementTree as ET

import genesis as gs


def marked_ball_mjcf():
    """MJCF model of a free ball with a marker bar (visual only) so its rotation shows in the viewer."""
    mjcf = ET.Element("mujoco", model="marked_ball")
    worldbody = ET.SubElement(mjcf, "worldbody")
    ball_body = ET.SubElement(worldbody, "body", name="ball", pos="0. 0. 0.1")
    ET.SubElement(ball_body, "joint", name="root", type="free")
    ET.SubElement(ball_body, "geom", type="sphere", size="0.1")
    bar_kwargs = {"contype": "0", "conaffinity": "0", "rgba": "1. 1. 0. 1."}
    ET.SubElement(ball_body, "geom", type="box", size="0.13 0.012 0.012", pos="0. 0. 0.06", **bar_kwargs)
    return ET.tostring(mjcf, encoding="unicode")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.gpu)

    ########################## create a scene ##########################
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, -4.5, 2.0),
            camera_lookat=(2.0, 0.3, 0.1),
        ),
        rigid_options=gs.options.RigidOptions(
            friction_cone=gs.friction_cone.elliptic,
            enable_torsional_friction=True,
            enable_rolling_friction=True,
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    # The contact coefficient is the maximum over the pair, so the zero-coefficient plane leaves each ball coasting
    # at its own rate.
    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            friction_torsional=0.0,
            friction_rolling=0.0,
        ),
    )
    balls = [
        scene.add_entity(
            gs.morphs.MJCF(
                file=marked_ball_mjcf(),
                pos=(0.0, 0.6 * i_ball, 0.0),
            ),
            material=gs.materials.Rigid(
                friction_rolling=friction_rolling,
            ),
        )
        for i_ball, friction_rolling in enumerate((0.0, 0.005))
    ]

    ########################## build ##########################
    scene.build()

    # Launch both balls rolling without slipping: v = w * r.
    for ball in balls:
        ball.set_dofs_velocity([1.0, 0.0, 0.0, 0.0, 10.0, 0.0])
    for i_step in range(500):
        scene.step()
        if i_step % 100 == 99:
            speeds = [float(ball.get_dofs_velocity()[..., 0]) for ball in balls]
            distances = [float(ball.get_pos()[..., 0]) for ball in balls]
            print(
                f"step {i_step + 1}: free ball {speeds[0]:.2f} m/s at {distances[0]:.2f} m, "
                f"braked ball {speeds[1]:.2f} m/s at {distances[1]:.2f} m"
            )


if __name__ == "__main__":
    main()
