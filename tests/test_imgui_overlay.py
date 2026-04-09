"""Screenshot integration test for ImGuiOverlayPlugin."""

import io
import os
import sys
import threading

import numpy as np
import OpenGL.error
import pytest
from PIL import Image

import genesis as gs
from genesis.ext.pyrender.imgui_overlay import ImGuiOverlayPlugin
from genesis.vis.viewer_plugins.viewer_plugin import ViewerPlugin

from .conftest import IS_INTERACTIVE_VIEWER_AVAILABLE

SNAPSHOT_FAILURE_DIR = os.path.join(os.path.dirname(__file__), "__snapshot_failures__")


class _FrameCapturePlugin(ViewerPlugin):
    """Test helper that reads the default framebuffer after all prior plugins (including ImGui) have drawn."""

    def __init__(self):
        super().__init__()
        self._armed = False
        self._result = None
        self._ready = threading.Event()

    def on_draw(self):
        if self._armed:
            renderer = self.viewer._renderer
            viewport = self.viewer._viewport_size
            self._result = renderer.jit.read_color_buf(*viewport, rgba=False)
            self._armed = False
            self._ready.set()

    def capture(self, timeout=30.0):
        self._ready.clear()
        self._armed = True
        self._ready.wait(timeout=timeout)
        return self._result


def _save_failure_images(received_arr, snapshot_data):
    """Save received image and diff on snapshot mismatch for CI artifact upload."""
    os.makedirs(SNAPSHOT_FAILURE_DIR, exist_ok=True)

    received_img = Image.fromarray(received_arr)
    received_img.save(os.path.join(SNAPSHOT_FAILURE_DIR, "received.png"))

    try:
        snapshot_img = Image.open(io.BytesIO(snapshot_data))
        snapshot_arr = np.asarray(snapshot_img).astype(np.int16)
        received_i16 = received_arr.astype(np.int16)
        if snapshot_arr.shape == received_i16.shape:
            diff = np.minimum(np.abs(snapshot_arr - received_i16), 255).astype(np.uint8)
            Image.fromarray(diff).save(os.path.join(SNAPSHOT_FAILURE_DIR, "diff.png"))
        snapshot_img.save(os.path.join(SNAPSHOT_FAILURE_DIR, "expected.png"))
    except Exception:
        pass


@pytest.mark.required
@pytest.mark.skipif(not IS_INTERACTIVE_VIEWER_AVAILABLE, reason="Interactive viewer not supported on this platform.")
@pytest.mark.xfail(sys.platform == "win32", raises=OpenGL.error.Error, reason="Invalid OpenGL context.")
def test_imgui_overlay_screenshot(png_snapshot):
    """Verify that the ImGui overlay renders visibly on top of the scene."""
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            res=(960, 720),
            camera_pos=(2.0, 2.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            run_in_thread=(sys.platform == "linux"),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=True,
    )
    scene.add_entity(gs.morphs.Plane(), name="plane")
    scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"), name="panda")

    imgui_plugin = ImGuiOverlayPlugin()
    scene.viewer.add_plugin(imgui_plugin)

    capture_plugin = _FrameCapturePlugin()
    scene.viewer.add_plugin(capture_plugin)

    scene.build()
    scene.step()

    rgb_arr = capture_plugin.capture()
    assert rgb_arr is not None
    assert rgb_arr.ndim == 3 and rgb_arr.shape[2] == 3

    try:
        img = Image.fromarray(rgb_arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        received_bytes = buf.getvalue()
        assert received_bytes == png_snapshot
    except AssertionError:
        _save_failure_images(rgb_arr, png_snapshot if isinstance(png_snapshot, bytes) else b"")
        if sys.platform == "darwin" and scene.visualizer._rasterizer._renderer._is_software:
            pytest.xfail("Flaky on MacOS with Apple Software Renderer. Pixel-matching failure.")
        raise
    finally:
        scene.viewer.stop()
