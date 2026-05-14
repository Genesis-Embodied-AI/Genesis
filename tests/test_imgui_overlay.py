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


def _apply_deterministic_imgui_overrides(monkeypatch):
    """Make ImGui rendering and timing pixel-identical across renderers for snapshot tests."""
    from imgui_bundle import imgui

    # Pin ``on_draw`` so it resets ``_last_time`` after every call (forcing the FPS history to use the
    # deterministic 1/60 fallback instead of the wall clock) and parks the ImGui mouse cursor off-panel at
    # the start of every frame. The mouse-park must go through ``add_mouse_pos_event`` rather than a direct
    # ``io.mouse_pos`` assignment because ImGui rebuilds ``MousePos`` from the queued event stream inside
    # ``new_frame``: pyglet posts a cursor-position event whenever it processes Win32 messages, so a direct
    # write is overwritten before any widget reads it. The headless Windows runner still keeps a desktop
    # cursor position internally, and the window-local coordinates pyglet derives from it shift across runs
    # with window placement - landing on a widget on some runs and not others, which made the snapshot flaky.
    # Appending an off-panel event as the LAST entry in the queue at the start of every frame guarantees
    # that ``new_frame`` resolves ``MousePos`` to ``(-1, -1)`` regardless of what pyglet queued earlier.
    # ``_init_imgui`` is pre-called so ``self._io`` is available; the real ``on_draw`` short-circuits its
    # own init via the ``_init_attempted`` guard.
    original_on_draw = ImGuiOverlayPlugin.on_draw

    def _on_draw_deterministic(self):
        if not self._init_attempted:
            self._init_imgui()
        if self._available:
            self._io.add_mouse_pos_event(-1.0, -1.0)
        original_on_draw(self)
        self._last_time = None

    monkeypatch.setattr(ImGuiOverlayPlugin, "on_draw", _on_draw_deterministic)

    # Discard the plugin's 18 px ``ImFontConfig`` so ProggyClean loads at its native 13 px. ProggyClean is a bitmap
    # font, so glyph rasterization is a memcpy on every renderer (stb_truetype is not byte-identical across software
    # vs hardware OpenGL). The patch must run before the plugin's lazy ``_init_imgui`` so the renderer uploads the
    # font texture at the correct size from the start.
    original_add_font_default = imgui.ImFontAtlas.add_font_default
    monkeypatch.setattr(imgui.ImFontAtlas, "add_font_default", lambda atlas, _=None: original_add_font_default(atlas))

    # Disable shape anti-aliasing (lines, fills, textured-line shortcut) and baked thick-line atlas entries so window
    # borders and button rounding do not drift between renderers either.
    original_init_imgui = ImGuiOverlayPlugin._init_imgui

    def _init_imgui_deterministic(self):
        original_init_imgui(self)
        if not self._available:
            return
        style = self._imgui.get_style()
        style.anti_aliased_lines = False
        style.anti_aliased_fill = False
        style.anti_aliased_lines_use_tex = False
        # Pyglet's backend sets ``display_framebuffer_scale`` from the window's pixel ratio (2.0 on Retina macOS,
        # 1.0 on most Linux runners); ImGui scales vertex positions by that factor, so the same layout produces
        # different pixel grids across platforms. Pin to 1.0 so vertex positions are byte-identical everywhere.
        self._io.display_framebuffer_scale = (1.0, 1.0)
        self._io.fonts.flags |= self._imgui.ImFontAtlasFlags_.no_baked_lines.value

    monkeypatch.setattr(ImGuiOverlayPlugin, "_init_imgui", _init_imgui_deterministic)


@pytest.mark.required
@pytest.mark.skipif(not IS_INTERACTIVE_VIEWER_AVAILABLE, reason="Interactive viewer not supported on this platform.")
@pytest.mark.skipif(not _IMGUI_BUNDLE_AVAILABLE, reason="imgui-bundle not installed (no Python 3.10 wheels).")
def test_imgui_overlay_screenshot(png_snapshot, monkeypatch):
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            # Keep ``res`` small enough to fit the virtual display area of GitHub-hosted Apple M1 macos-15 runners:
            # the on-screen capture below reads from the window framebuffer, whose size the OS clamps to the display.
            res=(640, 480),
            camera_pos=(4.5, -1.2, 2.5),
            camera_lookat=(0.0, -1.2, 0.5),
            # The capture path at the end of this test calls ``pyrender_viewer.on_draw`` and reads the window
            # framebuffer directly. That can only run on the thread that owns the GL context, so run the viewer
            # in the test thread instead of its own background thread.
            run_in_thread=False,
            # ``_render_help_text`` rasterizes "[i]: show keyboard instructions" via Genesis's own font path,
            # which is not byte-identical across software / hardware renderers; disable it so the captured
            # frame contains only the deterministic ImGui overlay.
            enable_help_text=False,
        ),
        vis_options=gs.options.VisOptions(
            shadow=False,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=True,
    )
    # The ground plane is a thin fixed box rather than ``gs.morphs.Plane`` because the latter's reflection /
    # shading is not byte-identical between Apple Software Renderer and Mesa llvmpipe. Apple Software Renderer
    # also misrasterizes the plane when any of its vertices fall outside the camera frustum, so the camera is
    # pulled back below to keep all four corners visible.
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(2.0, 2.0, 0.02),
            pos=(0.0, 0.0, -0.01),
            fixed=True,
        ),
        surface=gs.surfaces.Default(
            color=(0.60, 0.85, 0.55, 1.0),
        ),
        name="ground",
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.15, 0.15, 0.15),
            pos=(0.0, 0.4, 0.075),
        ),
        surface=gs.surfaces.Default(
            color=(0.85, 0.45, 0.20, 1.0),
        ),
        name="cube",
    )
    # Shift the robot in the camera's right direction so the ImGui panel on the left hides less of it.
    scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=(0.0, 0.0, 0.0),
        ),
        name="panda",
    )

    _apply_deterministic_imgui_overrides(monkeypatch)

    # Pin the panel to a fixed width so changes in entity names / labels do not shift the layout.
    imgui_plugin = ImGuiOverlayPlugin(panel_width=420)
    scene.viewer.add_plugin(imgui_plugin)

    scene.build()

    # ``render_offscreen`` only renders the 3D scene (it is the path also used for in-scene camera captures while the
    # interactive viewer is alive), so it deliberately skips the viewer's plugin loop and the ImGui overlay never
    # appears in its output. Drive ``Viewer.on_draw`` synchronously from the test thread instead, which is only legal
    # because ``run_in_thread=False`` keeps the viewer (and the GL context it owns) on this thread.
    pyrender_viewer = scene.viewer._pyrender_viewer
    pyrender_viewer.switch_to()
    pyrender_viewer.on_draw()
    rgb = pyrender_viewer._renderer.jit.read_color_buf(*pyrender_viewer._viewport_size, rgba=False)
    assert rgb_array_to_png_bytes(rgb) == png_snapshot
