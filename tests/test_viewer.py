import sys
import time

import numpy as np
import OpenGL.error
import pytest

import genesis as gs
from genesis.utils.misc import tensor_to_array
from genesis.vis.keybindings import Key, KeyAction, Keybind, KeyMod, MouseButton

from .conftest import IS_INTERACTIVE_VIEWER_AVAILABLE
from .utils import assert_allclose

CAM_RES = (480, 320)


# Note that software emulation is so slow that it may takes minutes to render a single frame...
def wait_for_viewer_events(viewer, condition_fn, timeout=300.0, sleep_interval=0.1):
    """Utility function to wait for viewer events to be processed in a threaded viewer."""
    if not viewer.run_in_thread:
        viewer.dispatch_pending_events()
        viewer.dispatch_events()

    for _ in range(int(timeout / sleep_interval)):
        if condition_fn():
            return
        time.sleep(sleep_interval)
    else:
        raise AssertionError("Keyboard event not processed before timeout")


@pytest.mark.required
@pytest.mark.skipif(not IS_INTERACTIVE_VIEWER_AVAILABLE, reason="Interactive viewer not supported on this platform.")
@pytest.mark.xfail(sys.platform == "win32", raises=OpenGL.error.Error, reason="Invalid OpenGL context.")
def test_interactive_viewer_disable_viewer_defaults():
    """Test that keyboard shortcuts can be disabled in the interactive viewer."""

    # Test with keyboard shortcuts DISABLED
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            run_in_thread=(sys.platform == "linux"),
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
    # Press key toggle world frame
    pyrender_viewer.dispatch_event("on_key_press", Key.W, 0)

    wait_for_viewer_events(pyrender_viewer, lambda: flags[0] and flags[1])

    assert flags[0], "Expected custom keybind callback to toggle flag on."
    assert flags[1], "Expected custom keybind with key modifiers to toggle flag on."
    assert pyrender_viewer.gs_context.world_frame_shown, "Expected world frame to be shown after pressing 'W' key."

    # Remove the keybind and press key to verify it no longer works
    scene.viewer.remove_keybind("toggle_flag_0")
    pyrender_viewer.dispatch_event("on_key_press", Key._0, 0)
    # Remap the keybind and check it works
    scene.viewer.remap_keybind("toggle_flag_1", new_key=Key._2, new_key_mods=None)
    pyrender_viewer.dispatch_event("on_key_press", Key._2, 0)

    wait_for_viewer_events(pyrender_viewer, lambda: not flags[1])

    assert flags[0], "Keybind was not removed properly."
    assert not flags[1], "Expected rebinded keybind to toggle flag off."

    # Error when remapping non-existent keybind
    with pytest.raises(ValueError):
        scene.viewer.remap_keybind("non_existent_keybind", new_key=Key._3, new_key_mods=None)

    # Error when adding a keybind with same key
    with pytest.raises(ValueError):
        scene.viewer.register_keybinds(
            Keybind(name="conflicting_keybind", key=Key._2, key_action=KeyAction.PRESS, callback=lambda: None),
        )


@pytest.mark.required
@pytest.mark.skipif(not IS_INTERACTIVE_VIEWER_AVAILABLE, reason="Interactive viewer not supported on this platform.")
def test_mouse_interaction_plugin():
    DT = 0.01
    MASS = 100.0
    BOX_LENGTH = 0.2
    STEPS = 20
    DRAG_DY = 8
    SPRING_CONST = 1000.0
    CAM_FOV = 30
    CAM_POS = (0.0, 0.6, 1.2)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, 0.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            # Forces odd resolution so that mouse clicks are centered on pixels
            res=(2 * (CAM_RES[0] // 2) + 1, 2 * (CAM_RES[0] // 2) + 1),
            camera_pos=CAM_POS,
            # looking to the top of the box
            camera_lookat=(0.0, 0.0, BOX_LENGTH),
            camera_fov=CAM_FOV,
            run_in_thread=(sys.platform == "linux"),
        ),
        show_viewer=True,
        show_FPS=False,
    )

    scene.add_entity(morph=gs.morphs.Plane())
    box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, BOX_LENGTH / 2),
            size=(BOX_LENGTH, BOX_LENGTH, BOX_LENGTH),
        ),
        material=gs.materials.Rigid(
            rho=MASS / (BOX_LENGTH**3),
        ),
    )
    _mouse_plugin = scene.viewer.add_plugin(
        gs.vis.viewer_plugins.MouseInteractionPlugin(
            use_force=True,
            spring_const=SPRING_CONST,
        )
    )
    scene.build()

    pyrender_viewer = scene.visualizer.viewer._pyrender_viewer
    assert pyrender_viewer.is_active

    class EventCounterHandler:
        def __init__(self):
            self.count = 0

        def on_mouse_press(self, x: int, y: int, button: int, modifiers: int):
            self.count += 1

        def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int):
            self.count += 1

        def on_mouse_release(self, x: int, y: int, buttons: int, modifiers: int):
            self.count += 1

    event_counter = EventCounterHandler()
    expected_count = 0

    def check_event_count():
        nonlocal expected_count
        expected_count += 1
        return lambda: event_counter.count == expected_count

    pyrender_viewer.push_handlers(event_counter)

    scene.step()

    assert_allclose(box.get_vel(), 0, tol=gs.EPS)

    initial_pos = box.get_pos().clone()

    viewport_size = pyrender_viewer._viewport_size
    x, y = viewport_size[0] // 2, viewport_size[1] // 2

    # Press mouse to grab the box
    pyrender_viewer.dispatch_event("on_mouse_press", x, y, MouseButton.LEFT, 0)
    # Ensure event is processed
    wait_for_viewer_events(pyrender_viewer, check_event_count())

    rgb_arrs = []
    for i in range(STEPS):
        y += DRAG_DY
        pyrender_viewer.dispatch_event("on_mouse_drag", x, y, 0, DRAG_DY, MouseButton.LEFT, 0)
        wait_for_viewer_events(pyrender_viewer, check_event_count())
        scene.step()
        if (i + 1) % (STEPS // 2) == 0:
            rgb_arr, *_ = pyrender_viewer.render_offscreen(
                pyrender_viewer._camera_node, pyrender_viewer._renderer, rgb=True, depth=False, seg=False, normal=False
            )
            rgb_arrs.append(rgb_arr)

    assert not np.array_equal(rgb_arrs[0], rgb_arrs[1]), "Expected images to be different after dragging the object."

    final_pos = box.get_pos()
    final_vel = box.get_vel()

    assert_allclose(
        final_vel[:2],
        0.0,
        tol=0.002,
        err_msg="Final x and y velocities should be near zero since dragging only in z direction.",
    )

    distance_to_box = np.linalg.norm(tensor_to_array(initial_pos) - CAM_POS)
    pixels_to_world = 2.0 * distance_to_box * np.tan(np.radians(CAM_FOV) / 2.0) / viewport_size[1]
    total_world_displacement = STEPS * DRAG_DY * pixels_to_world

    displacement_z = final_pos[2] - initial_pos[2]
    assert displacement_z > gs.EPS, "Box should have moved upward"
    assert displacement_z < total_world_displacement, (
        "Box displacement should be less than mouse displacement from spring lag"
    )

    pyrender_viewer.dispatch_event("on_mouse_release", x, y, MouseButton.LEFT, 0)
    scene.step()
    wait_for_viewer_events(pyrender_viewer, check_event_count())
    rgb_arr, *_ = pyrender_viewer.render_offscreen(
        pyrender_viewer._camera_node, pyrender_viewer._renderer, rgb=True, depth=False, seg=False, normal=False
    )
    assert not np.array_equal(rgb_arrs[-1], rgb_arr), "Expected visualization to change after releasing the object."

    # The forces from mouse spring are approximate, so use a large tolerance.
    # FIXME: Use a more accurate model to predict final velocity.
    total_sim_time = STEPS * DT
    avg_mouse_velocity = total_world_displacement / total_sim_time
    num_tau = total_sim_time * np.sqrt(SPRING_CONST / MASS)
    velocity_fraction = 1.0 - (1.0 + num_tau) * np.exp(-num_tau)
    expected_vel_z = avg_mouse_velocity * velocity_fraction

    assert_allclose(
        final_vel[2],
        expected_vel_z,
        rtol=0.5,
        err_msg="Final z velocity does not match expected value based on spring dynamics.",
    )
