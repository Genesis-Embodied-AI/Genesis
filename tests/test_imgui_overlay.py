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
            res=(960, 720),
            camera_pos=(2.0, 2.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
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
        pyrender_viewer = scene.viewer._pyrender_viewer
        rgb, *_ = pyrender_viewer.render_offscreen(
            pyrender_viewer._camera_node, pyrender_viewer._renderer, rgb=True, depth=False, seg=False, normal=False
        )
        assert rgb_array_to_png_bytes(rgb) == png_snapshot
    finally:
        scene.viewer.stop()
