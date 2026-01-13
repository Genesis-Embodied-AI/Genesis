import sys

import pyglet
import pytest

import genesis as gs
from genesis.ext.pyrender.interaction.keybindings import KeyAction, Keybind

from .conftest import IS_INTERACTIVE_VIEWER_AVAILABLE
from .utils import rgb_array_to_png_bytes

CAM_RES = (640, 480)


@pytest.mark.required
@pytest.mark.skipif(not IS_INTERACTIVE_VIEWER_AVAILABLE, reason="Interactive viewer not supported on this platform.")
def test_default_viewer_plugin(monkeypatch, png_snapshot):
    # Disable text rendering as it is messing up with pixel matching when using old CPU-based Mesa driver
    monkeypatch.setattr("genesis.ext.pyrender.renderer.Renderer.render_texts", lambda *args, **kwargs: None)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=1,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(4.0, 0.0, 3.0),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=40,
            res=CAM_RES,
            run_in_thread=(sys.platform == "linux"),
        ),
        show_viewer=True,
        show_FPS=False,
    )

    scene.add_entity(morph=gs.morphs.Plane())
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.2),
            size=(0.2, 0.2, 0.2),
            euler=(30, 40, 0),
        )
    )
    scene.build()

    pyrender_viewer = scene.visualizer.viewer._pyrender_viewer
    assert pyrender_viewer.is_active

    # Add a custom keybind
    flag = False

    def toggle_flag():
        nonlocal flag
        flag = True

    scene.viewer.register_keybinds(
        Keybind(key_code=pyglet.window.key._0, key_action=KeyAction.PRESS, callback=toggle_flag)
    )

    # Press key to toggle flag on
    pyrender_viewer.dispatch_event("on_key_press", pyglet.window.key._0, 0)
    # Press key to turn off shadows
    pyrender_viewer.dispatch_event("on_key_press", pyglet.window.key.H, 0)

    scene.step()

    assert flag, "Custom keybind callback was not called."

    rgb_arr, *_ = pyrender_viewer.render_offscreen(
        pyrender_viewer._camera_node, pyrender_viewer._renderer, rgb=True, depth=False, seg=False, normal=False
    )
    assert rgb_array_to_png_bytes(rgb_arr) == png_snapshot


@pytest.mark.required
@pytest.mark.skipif(not IS_INTERACTIVE_VIEWER_AVAILABLE, reason="Interactive viewer not supported on this platform.")
def test_mouse_spring_plugin(png_snapshot):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=1,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(4.0, 0.0, 3.0),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=40,
            res=CAM_RES,
            run_in_thread=(sys.platform == "linux"),
            viewer_plugin=gs.options.viewer_plugins.MouseSpringPlugin(),
        ),
        show_viewer=True,
        show_FPS=False,
    )

    scene.add_entity(morph=gs.morphs.Plane())
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.4, 0.0, 0.2),
            size=(0.2, 0.2, 0.2),
            euler=(30, 40, 0),
        )
    )
    scene.build()

    pyrender_viewer = scene.visualizer.viewer._pyrender_viewer
    assert pyrender_viewer.is_active

    scene.step()

    viewport_size = pyrender_viewer._viewport_size
    x, y = viewport_size[0] // 2, viewport_size[1] // 2
    dx, dy = 4, 8
    pyrender_viewer.dispatch_event("on_mouse_press", x, y, 1, 0)

    for _ in range(30):
        x += dx
        y += dy
        pyrender_viewer.dispatch_event("on_mouse_drag", x, y, dx, dy, 1, 0)
        scene.step()

    rgb_arr, *_ = pyrender_viewer.render_offscreen(
        pyrender_viewer._camera_node, pyrender_viewer._renderer, rgb=True, depth=False, seg=False, normal=False
    )
    assert rgb_array_to_png_bytes(rgb_arr) == png_snapshot
