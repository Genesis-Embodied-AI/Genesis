"""Screenshot integration test for ImGuiOverlayPlugin."""

import pytest

import genesis as gs
from genesis.ext.pyrender.overlay import ImGuiOverlayPlugin

from .conftest import IS_INTERACTIVE_VIEWER_AVAILABLE
from .utils import rgb_array_to_png_bytes

try:
    import imgui_bundle  # noqa: F401

    _IMGUI_BUNDLE_AVAILABLE = True
except ImportError:
    _IMGUI_BUNDLE_AVAILABLE = False


@pytest.mark.required
@pytest.mark.skipif(not IS_INTERACTIVE_VIEWER_AVAILABLE, reason="Interactive viewer not supported on this platform.")
@pytest.mark.skipif(not _IMGUI_BUNDLE_AVAILABLE, reason="imgui-bundle not installed (no Python 3.10 wheels).")
def test_imgui_overlay_screenshot(png_snapshot, monkeypatch):
    # ImGui font rasterization differs across platforms (freetype vs CoreText) and renderers. Loosen the tolerance
    # so the same baseline matches on Linux and macOS.
    png_snapshot.extension._std_err_threshold = 5.0

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            # Keep ``res`` small enough to fit the virtual display area of GitHub-hosted Apple M1 macos-15 runners:
            # the on-screen capture below reads from the window framebuffer, whose size the OS clamps to the display.
            res=(640, 480),
            camera_pos=(2.0, 2.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            # The capture path at the end of this test calls ``pyrender_viewer.on_draw`` and reads the window
            # framebuffer directly. That can only run on the thread that owns the GL context, so run the viewer
            # in the test thread instead of its own background thread.
            run_in_thread=False,
        ),
        vis_options=gs.options.VisOptions(
            shadow=False,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=True,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
        name="plane",
    )
    scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        name="panda",
    )

    # Wrap ``on_draw`` at the class level (before instantiation) so the pyglet event registration in ``add_plugin``
    # picks up the wrapper. Reset ``_last_time`` after each call so the next frame's delta_time falls back to the
    # deterministic 1/60 default instead of the wall clock.
    original_on_draw = ImGuiOverlayPlugin.on_draw

    def _on_draw_pinned_fps(self):
        original_on_draw(self)
        self._last_time = None

    monkeypatch.setattr(ImGuiOverlayPlugin, "on_draw", _on_draw_pinned_fps)

    # Pin the panel to a fixed width so changes in entity names / labels do not shift the layout.
    imgui_plugin = ImGuiOverlayPlugin(panel_width=420)
    scene.viewer.add_plugin(imgui_plugin)

    scene.build()
    scene.step()

    try:
        # ``render_offscreen`` only renders the 3D scene (it is the path also used for in-scene camera captures while
        # the interactive viewer is alive), so it deliberately skips the viewer's plugin loop and the ImGui overlay
        # never appears in its output. Drive ``Viewer.on_draw`` synchronously from the test thread instead, which is
        # only legal because ``run_in_thread=False`` keeps the viewer (and the GL context it owns) on this thread.
        pyrender_viewer = scene.viewer._pyrender_viewer
        pyrender_viewer.switch_to()
        pyrender_viewer.on_draw()
        rgb = pyrender_viewer._renderer.jit.read_color_buf(*pyrender_viewer._viewport_size, rgba=False)
        assert rgb_array_to_png_bytes(rgb) == png_snapshot
    finally:
        scene.viewer.stop()
