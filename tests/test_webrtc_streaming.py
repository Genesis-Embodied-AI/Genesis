import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def initialize_genesis():
    # Override tests/conftest.py autouse fixture: this unit test only validates the streaming app wiring.
    yield


def _load_webrtc_module():
    pytest.importorskip("aiortc")
    pytest.importorskip("aiohttp")
    pytest.importorskip("av")

    module_path = Path(__file__).resolve().parents[1] / "genesis" / "vis" / "streaming" / "webrtc_aiortc.py"
    spec = importlib.util.spec_from_file_location("genesis.vis.streaming.webrtc_aiortc", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_streaming_common_module():
    module_path = Path(__file__).resolve().parents[1] / "genesis" / "vis" / "streaming" / "common.py"
    spec = importlib.util.spec_from_file_location("genesis.vis.streaming.common", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyCamera:
    res = (64, 48)

    def render(self, rgb=True, depth=False, segmentation=False, normal=False, force_render=False):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        return frame, None, None, None


def _collect_routes(app) -> set[tuple[str, str]]:
    routes = set()
    for route in app.router.routes():
        info = route.get_info()
        if "path" in info:
            routes.add((route.method, info["path"]))
        elif "formatter" in info:
            routes.add((route.method, info["formatter"]))
    return routes


def test_webrtc_streamer_routes_created():
    module = _load_webrtc_module()
    WebRTCStreamer = module.WebRTCStreamer

    streamer = WebRTCStreamer(camera=DummyCamera(), host="127.0.0.1", port=0, fps=30)
    app = streamer.create_app()
    routes = _collect_routes(app)

    assert ("GET", "/") in routes
    assert ("POST", "/offer") in routes
    assert ("POST", "/shutdown") not in routes


def test_webrtc_streamer_shutdown_route_enabled():
    module = _load_webrtc_module()
    WebRTCStreamer = module.WebRTCStreamer

    streamer = WebRTCStreamer(
        camera=DummyCamera(),
        host="127.0.0.1",
        port=0,
        fps=30,
        allow_browser_shutdown=True,
    )
    app = streamer.create_app()
    routes = _collect_routes(app)

    assert ("GET", "/") in routes
    assert ("POST", "/offer") in routes
    assert ("POST", "/shutdown") in routes


def test_webrtc_index_html_has_chrome_playback_fallback():
    module = _load_webrtc_module()
    html = module.WebRTCStreamer._index_html(ice_servers=[], allow_shutdown=False)

    assert "video.muted = true;" in html
    assert "fallbackStream" in html
    assert "await video.play();" in html


def test_build_stream_url_wildcard_host_uses_detected_ip():
    module = _load_streaming_common_module()
    module._detect_non_loopback_ipv4 = lambda: "10.20.30.40"

    url = module.build_stream_url(host="0.0.0.0", port=8000)

    assert url == "http://10.20.30.40:8000"


def test_build_stream_url_wildcard_host_falls_back_to_loopback():
    module = _load_streaming_common_module()
    module._detect_non_loopback_ipv4 = lambda: None

    url = module.build_stream_url(host="0.0.0.0", port=8000)

    assert url == "http://127.0.0.1:8000"
