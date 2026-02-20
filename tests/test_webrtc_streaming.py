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
