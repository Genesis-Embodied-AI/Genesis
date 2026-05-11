"""Screenshot integration test for ImGuiOverlayPlugin."""

import io
import sys
import threading

import pytest
from PIL import Image

import genesis as gs
from genesis.ext.pyrender.overlay import ImGuiOverlayPlugin
from genesis.vis.viewer_plugins.viewer_plugin import ViewerPlugin

from .conftest import IS_INTERACTIVE_VIEWER_AVAILABLE


class _FrameCapturePlugin(ViewerPlugin):
    """Read the default framebuffer after all prior plugins (including ImGui) have drawn."""

    capture_next_frame = False
    result = None

    def __init__(self):
        super().__init__()
        self.ready = threading.Event()

    def on_draw(self):
        if self.capture_next_frame:
            self.result = self.viewer._renderer.jit.read_color_buf(*self.viewer._viewport_size, rgba=False)
            self.capture_next_frame = False
            self.ready.set()


@pytest.mark.required
@pytest.mark.skipif(not IS_INTERACTIVE_VIEWER_AVAILABLE, reason="Interactive viewer not supported on this platform.")
def test_imgui_overlay_screenshot(png_snapshot, monkeypatch):
    # ImGui font rasterization differs across platforms (freetype vs CoreText) and renderers.
    # Loosen the tolerance so the same baseline matches on Linux and macOS.
    png_snapshot.extension._std_err_threshold = 5.0

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            res=(960, 720),
            camera_pos=(2.0, 2.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            run_in_thread=(sys.platform == "linux"),
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

    # Wrap ``on_draw`` at the class level (before instantiation) so the pyglet event registration
    # in ``add_plugin`` picks up the wrapper. Reset ``_last_time`` after each call so the next
    # frame's delta_time falls back to the deterministic 1/60 default instead of the wall clock.
    original_on_draw = ImGuiOverlayPlugin.on_draw

    def _on_draw_pinned_fps(self):
        original_on_draw(self)
        self._last_time = None

    monkeypatch.setattr(ImGuiOverlayPlugin, "on_draw", _on_draw_pinned_fps)

    # Pin the panel to a fixed width so changes in entity names / labels do not shift the layout.
    imgui_plugin = ImGuiOverlayPlugin(panel_width=420)
    scene.viewer.add_plugin(imgui_plugin)
    capture_plugin = _FrameCapturePlugin()
    scene.viewer.add_plugin(capture_plugin)

    scene.build()
    scene.step()

    try:
        # ImGui builds its font atlas lazily on first draw and tab bars settle to their final width
        # only after a couple frames. Warm up a few frames before capturing so the recorded image is
        # stable across runs.
        for _ in range(3):
            scene.viewer.update()

        capture_plugin.capture_next_frame = True
        # Drive one synchronous draw on the main-thread viewer (macOS/Windows). In threaded mode
        # (Linux) this is a no-op and the viewer thread will draw at its own refresh cadence.
        scene.viewer.update()
        capture_plugin.ready.wait(timeout=30.0)
        assert capture_plugin.result is not None
        buf = io.BytesIO()
        Image.fromarray(capture_plugin.result).save(buf, format="PNG")
        assert buf.getvalue() == png_snapshot
    finally:
        scene.viewer.stop()
