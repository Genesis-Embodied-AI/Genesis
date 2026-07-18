"""
Interactive per-environment geometry scaling.

A sphere, a box and a duck mesh drop onto a plane. Pressing the scale key randomizes the size of every object
(via entity.set_scale) and re-drops it, so it settles at the new size. Requires the scene built with
RigidOptions(enable_geom_scaling=True, batch_links_info=True).

Run with the viewer to interact:

    python examples/rigid/geom_scale.py --vis

Keyboard controls:
    g      randomize the scale of every object
    esc    quit
"""

import argparse
import os
import random

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    gs.init()

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, 0.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.3),
            camera_fov=40,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_geom_scaling=True,
            batch_links_info=True,
            gravity=(0.0, 0.0, -9.81),
            dt=0.01,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(
        gs.morphs.Plane(),
    )
    sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.1,
            pos=(0.0, -0.7, 0.5),
        ),
    )
    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(0.0, 0.0, 0.5),
        ),
    )
    duck = scene.add_entity(
        material=gs.materials.Rigid(rho=200),
        morph=gs.morphs.Mesh(
            file="meshes/duck/duck.obj",
            scale=0.001,
            euler=(90.0, 0.0, 90.0),
            pos=(0.0, 0.7, 0.5),
            convexify=True,
        ),
    )

    if scene.viewer is not None:
        scene.viewer.add_plugin(
            gs.vis.viewer_plugins.MouseInteractionPlugin(
                use_force=True,
                spring_const=500.0,
            )
        )

    scene.build()

    objects = ((sphere, -0.7), (box, 0.0), (duck, 0.7))
    DROP_Z = 0.8

    pending_randomize = [True]
    is_running = [True]

    def request_randomize():
        pending_randomize[0] = True

    def stop():
        is_running[0] = False

    if scene.viewer is not None:
        scene.viewer.register_keybinds(
            Keybind("randomize_scale", Key.G, KeyAction.RELEASE, callback=request_randomize),
            Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        )

    while is_running[0]:
        if pending_randomize[0]:
            pending_randomize[0] = False
            for entity, y in objects:
                entity.set_scale(random.uniform(0.5, 2.0))
                entity.set_pos((0.0, y, DROP_Z))
                entity.zero_all_dofs_velocity()

        scene.step()

        if "PYTEST_VERSION" in os.environ:
            break


if __name__ == "__main__":
    main()
