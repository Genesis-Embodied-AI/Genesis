"""Interactive joint control example using ImGui overlay.

Demonstrates:
- Simulation controls (play/pause/step/reset)
- Entity browser with joint sliders
- Visualization toggles
- Camera controls
- Custom user panels via register_panel()
- Scene rebuild (add entities, change scale)
"""

import os
import time

import genesis as gs
from genesis.engine.interactive_scene import InteractiveScene
from genesis.ext.pyrender.overlay import ImGuiOverlayPlugin

gs.init()

interactive = InteractiveScene()
interactive.rebuild(
    scene_kwargs=dict(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, 2.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=True,
    ),
    entities_kwargs={
        "Plane": dict(
            morph=gs.morphs.Plane(),
        ),
        "Panda": dict(
            morph=gs.morphs.MJCF(
                file="xml/franka_emika_panda/panda.xml",
            ),
        ),
        "Box": dict(
            morph=gs.morphs.Box(
                pos=(0, 0, 1.0),
                size=(0.2, 0.2, 0.2),
            ),
        ),
    },
)

plugin = ImGuiOverlayPlugin()
interactive.viewer.add_plugin(plugin)


def custom_panel(imgui):
    imgui.text("Custom Demo Panel")
    imgui.text("This panel was registered via register_panel()")


plugin.register_panel(custom_panel)

is_test = "PYTEST_VERSION" in os.environ
horizon = 5 if is_test else None

frame = 0
while interactive.viewer.is_alive():
    if plugin.rebuild_requested:
        interactive.rebuild(entities_kwargs=plugin.pending_entities_kwargs)
    if plugin.should_step():
        interactive.scene.step()
    frame += 1
    if horizon is not None and frame >= horizon:
        break
    time.sleep(0.01)
