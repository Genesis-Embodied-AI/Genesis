import sys
import time

import OpenGL.error
import pyglet
import pytest

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind, KeyMod

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
def test_default_viewer_plugin():
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

    assert len(pyrender_viewer._keybindings) > 0, "Expected default keybindings to be registered."

    # Press key toggle world frame
    pyrender_viewer.dispatch_event("on_key_press", Key.W, 0)

    # Add a custom keybind
    flags = [False, False, False]

    def toggle_flag(idx):
        flags[idx] = not flags[idx]

    scene.viewer.register_keybinds(
        Keybind(
            name="toggle_flag_0",
            key=Key._0,
            key_action=KeyAction.PRESS,
            callback=lambda: toggle_flag(0),
        ),
        Keybind(
            name="toggle_flag_1",
            key=Key._1,
            key_action=KeyAction.PRESS,
            key_mods=(KeyMod.SHIFT, KeyMod.CTRL),
            callback=toggle_flag,
            args=(1,),
        ),
    )

    # Press key to toggle flag on
    pyrender_viewer.dispatch_event("on_key_press", Key._0, 0)
    # Press key with modifiers to toggle flag off
    pyrender_viewer.dispatch_event("on_key_press", Key._1, KeyMod.SHIFT | KeyMod.CTRL)

    if pyrender_viewer.run_in_thread:
        for i in range(100):
            if flags[0] and flags[1]:
                break
            time.sleep(0.1)
    else:
        pyrender_viewer.dispatch_pending_events()
        pyrender_viewer.dispatch_events()

    assert pyrender_viewer.gs_context.world_frame_shown, "Expected world frame to be shown after pressing 'W' key."

    assert flags[0], "Expected custom keybind callback to toggle flag on."
    assert flags[1], "Expected custom keybind with key modifiers to toggle flag on."

    # Remove the keybind and press key to verify it no longer works
    scene.viewer.remove_keybind("toggle_flag_0")
    pyrender_viewer.dispatch_event("on_key_press", Key._0, 0)
    # Remap the keybind and check it works
    scene.viewer.remap_keybind("toggle_flag_1", new_key=Key._2, new_key_mods=None)
    pyrender_viewer.dispatch_event("on_key_press", Key._2, 0)

    if pyrender_viewer.run_in_thread:
        for i in range(1000):
            if not flags[1]:
                break
            time.sleep(0.1)
    else:
        pyrender_viewer.dispatch_pending_events()
        pyrender_viewer.dispatch_events()

    assert flags[0], "Keybind was not removed properly."
    assert not flags[1], "Expected rebinded keybind to toggle flag off."

    # Error when remapping non-existent keybind
    with pytest.raises(ValueError):
        scene.viewer.remap_keybind("non_existent_keybind", new_key=Key._3)

    # Error when adding a keybind with same key
    with pytest.raises(ValueError):
        scene.viewer.register_keybinds(
            Keybind(name="conflicting_keybind", key=Key._2, key_action=KeyAction.PRESS, callback=lambda: None),
        )
