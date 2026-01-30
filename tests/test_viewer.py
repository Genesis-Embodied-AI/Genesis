import sys

import OpenGL.error
import pyglet
import pytest

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind

from .conftest import IS_INTERACTIVE_VIEWER_AVAILABLE
from .utils import rgb_array_to_png_bytes

CAM_RES = (480, 320)


@pytest.mark.required
@pytest.mark.skipif(not IS_INTERACTIVE_VIEWER_AVAILABLE, reason="Interactive viewer not supported on this platform.")
@pytest.mark.xfail(sys.platform == "win32", raises=OpenGL.error.Error, reason="Invalid OpenGL context.")
def test_interactive_viewer_disable_viewer_defaults():
    """Test that keyboard shortcuts can be disabled in the interactive viewer."""

    # Test with keyboard shortcuts DISABLED
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            disable_help_text=True,
            disable_default_keybinds=True,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=True,
    )
    scene.build()
    pyrender_viewer = scene.visualizer.viewer._pyrender_viewer
    assert pyrender_viewer.is_active

    # Verify the flag is set correctly
    assert pyrender_viewer._disable_help_text is True
    # Verify that no keybindings are registered
    assert len(pyrender_viewer._keybindings) == 0


@pytest.mark.required
@pytest.mark.skipif(not IS_INTERACTIVE_VIEWER_AVAILABLE, reason="Interactive viewer not supported on this platform.")
def test_default_viewer_plugin(png_snapshot):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=1,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, 0.0, 1.0),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=30,
            res=CAM_RES,
            run_in_thread=(sys.platform == "linux"),
            disable_help_text=False,
            disable_default_keybinds=False,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=True,
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

    scene.viewer.register_keybinds(Keybind(key_code=Key._0, key_action=KeyAction.PRESS, callback=toggle_flag))

    # Press key to toggle flag on
    pyrender_viewer.dispatch_event("on_key_press", Key._0, 0)
    # Press key to turn off shadows
    pyrender_viewer.dispatch_event("on_key_press", Key.H, 0)

    scene.step()

    assert flag, "Custom keybind callback was not called."

    # Snapshot should not have shadows
    rgb_arr, *_ = pyrender_viewer.render_offscreen(
        pyrender_viewer._camera_node, pyrender_viewer._renderer, rgb=True, depth=False, seg=False, normal=False
    )
    assert rgb_array_to_png_bytes(rgb_arr) == png_snapshot
