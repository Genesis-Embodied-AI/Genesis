"""In-hand pivoting under the elliptic cone with torsional friction, the maximum-realism friction configuration.

A sphere is pinched between two fixed plates and spun about the pinch axis, the canonical grasp scenario a point
contact cannot resist with sliding friction alone: without torsional friction the grasped sphere pivots forever.
The elliptic cone with its auto-resolved high 'impratio' holds the sphere against gravity without the tangential
creep of the pyramidal cone, and its exact Coulomb bound brakes the spin at the rate set by 'friction_torsional'
times the grip normal force. A marker bar shows the rotation in the viewer.
"""

import argparse
import xml.etree.ElementTree as ET

import genesis as gs


def marked_sphere_mjcf():
    """MJCF model of a free sphere with a marker bar (visual only) so its spin shows in the viewer."""
    mjcf = ET.Element("mujoco", model="marked_sphere")
    worldbody = ET.SubElement(mjcf, "worldbody")
    sphere_body = ET.SubElement(worldbody, "body", name="sphere", pos="0. 0. 0.")
    ET.SubElement(sphere_body, "joint", name="root", type="free")
    ET.SubElement(sphere_body, "geom", type="sphere", size="0.1")
    bar_kwargs = {"contype": "0", "conaffinity": "0", "rgba": "1. 1. 0. 1."}
    ET.SubElement(sphere_body, "geom", type="box", size="0.13 0.012 0.012", pos="0. 0. 0.06", **bar_kwargs)
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
            camera_pos=(0.9, -0.9, 0.6),
            camera_lookat=(0.0, 0.0, 0.3),
        ),
        rigid_options=gs.options.RigidOptions(
            friction_cone=gs.friction_cone.elliptic,
            enable_torsional_friction=True,
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    scene.add_entity(
        gs.morphs.Plane(),
    )
    # Two fixed plates pinch the sphere with a slight interpenetration that supplies the grip normal force.
    for i_side in range(2):
        scene.add_entity(
            gs.morphs.Box(
                size=(0.3, 0.02, 0.3),
                pos=(0.0, (2 * i_side - 1) * 0.1095, 0.3),
                fixed=True,
            ),
            material=gs.materials.Rigid(
                friction_torsional=0.002,
            ),
        )
    sphere = scene.add_entity(
        gs.morphs.MJCF(
            file=marked_sphere_mjcf(),
            pos=(0.0, 0.0, 0.3),
        ),
        material=gs.materials.Rigid(
            friction_torsional=0.002,
        ),
    )

    ########################## build ##########################
    scene.build()

    # Let the pinch settle, then spin the sphere about the pinch axis.
    for _ in range(50):
        scene.step()
    z_settled = float(sphere.get_dofs_position()[..., 2])
    sphere.set_dofs_velocity([0.0, 0.0, 0.0, 0.0, 8.0, 0.0])
    for i_step in range(300):
        scene.step()
        if (i_step < 100 and i_step % 20 == 19) or i_step % 100 == 99:
            spin = float(sphere.get_dofs_velocity()[..., 4])
            sag = z_settled - float(sphere.get_dofs_position()[..., 2])
            print(f"step {i_step + 1}: spin {spin:.2f} rad/s, sag since settling {1e3 * sag:.2f} mm")


if __name__ == "__main__":
    main()
