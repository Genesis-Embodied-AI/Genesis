import argparse
import math
import os

import genesis as gs
import genesis.vis.keybindings as kb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mouse interaction viewer plugin example.")
    parser.add_argument("--use_force", action="store_true", help="Apply spring forces instead of setting position")
    args = parser.parse_args()

    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=True,
    )

    scene.add_entity(gs.morphs.Plane())

    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(-0.3, -0.3, 0),
            radius=0.1,
        ),
    )
    for i in range(6):
        angle = i * (2 * math.pi / 6)
        radius = 0.5 + i * 0.1
        cube = scene.add_entity(
            morph=gs.morphs.Box(
                pos=(radius * math.cos(angle), radius * math.sin(angle), 0.1 + i * 0.1),
                size=(0.2, 0.2, 0.2),
            ),
        )

    scene.viewer.add_plugin(
        gs.vis.viewer_plugins.MouseInteractionPlugin(
            use_force=args.use_force,
            color=(0.1, 0.6, 0.8, 0.6),
        )
    )

    scene.build()

    is_running = True

    def stop():
        global is_running
        is_running = False

    scene.viewer.register_keybinds(
        kb.Keybind("quit", kb.Key.ESCAPE, kb.KeyAction.PRESS, callback=stop),
    )

    try:
        while is_running:
            scene.step()

            if "PYTEST_VERSION" in os.environ:
                break
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")
