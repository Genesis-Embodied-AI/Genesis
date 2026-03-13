"""Interactive joint control example using ImGui overlay.

Demonstrates:
- Simulation controls (play/pause/step/reset)
- Entity browser with joint sliders
- Visualization toggles
- Camera controls
- Custom user panels via register_panel()
- Scene rebuild (add entities, change scale)
"""

import time

import numpy as np

import genesis as gs
from genesis.ext.pyrender.imgui_overlay import ImGuiOverlayPlugin

gs.init()

# Store scene options for rebuild
scene_kwargs = dict(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(2.0, 2.0, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
    ),
    show_viewer=True,
)

scene = gs.Scene(**scene_kwargs)
scene.add_entity(gs.morphs.Plane())
scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
scene.add_entity(gs.morphs.Box(pos=(0, 0, 1.0), size=(0.2, 0.2, 0.2)))
scene.build()


def rebuild_scene(plugin):
    """Rebuild scene from plugin's entity_specs."""
    global scene
    specs = plugin.entity_specs

    # Save camera state
    cam_pos = scene.viewer.camera_pos.copy()
    cam_lookat = np.array(scene.viewer._pyrender_viewer._trackball._n_target).copy()

    scene.destroy()

    scene = gs.Scene(**scene_kwargs)
    for spec in specs:
        scene.add_entity(
            spec["morph"],
            material=spec["material"],
            surface=spec["surface"],
            visualize_contact=spec["visualize_contact"],
        )
    scene.build()

    # Re-register plugin on new viewer
    scene.viewer._pyrender_viewer.register_plugin(plugin)

    # Restore camera
    scene.viewer.set_camera_pose(pos=cam_pos, lookat=cam_lookat)


plugin = ImGuiOverlayPlugin(rebuild_fn=rebuild_scene)
scene.viewer._pyrender_viewer.register_plugin(plugin)


def custom_panel(imgui):
    imgui.text("Custom Demo Panel")
    imgui.text("This panel was registered via register_panel()")


plugin.register_panel(custom_panel)

while scene.viewer.is_alive():
    if plugin.rebuild_requested:
        rebuild_scene(plugin)
    if plugin.should_step():
        scene.step()
    time.sleep(0.01)
