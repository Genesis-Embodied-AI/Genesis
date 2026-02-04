import math

import genesis as gs
import genesis.vis.keybindings as kb

if __name__ == "__main__":
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
            use_force=True,
            spring_const=1.0,
            spring_damping=1.0,
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
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")
