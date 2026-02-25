import asyncio
import json
import logging
import time
from fractions import Fraction
from typing import Any, Sequence

import numpy as np

try:
    from aiohttp import web
    from aiortc import (
        RTCConfiguration,
        RTCIceServer,
        RTCPeerConnection,
        RTCRtpSender,
        RTCSessionDescription,
        VideoStreamTrack,
    )
    from av import VideoFrame
except ImportError as exc:
    raise ImportError(
        "WebRTC streaming dependencies are missing. Install them with `pip install genesis-world[webrtc]` "
        'or `pip install -e ".[webrtc]"`.'
    ) from exc

__all__ = ["GenesisCameraVideoTrack", "WebRTCStreamer"]

LOGGER = logging.getLogger(__name__)


def _to_numpy_uint8_rgb(frame_data: Any) -> np.ndarray:
    """Convert a render output tensor / array into contiguous uint8 RGB (H, W, 3)."""
    if hasattr(frame_data, "detach") and hasattr(frame_data, "cpu"):
        frame_data = frame_data.detach().cpu().numpy()
    else:
        frame_data = np.asarray(frame_data)

    if frame_data.ndim >= 4:
        frame_data = frame_data[0]

    if frame_data.ndim != 3:
        raise ValueError(f"Expected 3D frame array, got shape {frame_data.shape}.")

    if frame_data.shape[-1] != 3:
        if frame_data.shape[0] == 3:
            frame_data = np.moveaxis(frame_data, 0, -1)
        else:
            raise ValueError(f"Expected RGB channels in last dimension, got shape {frame_data.shape}.")

    if np.issubdtype(frame_data.dtype, np.floating):
        frame_data = np.nan_to_num(frame_data, nan=0.0, posinf=255.0, neginf=0.0)
        max_value = float(frame_data.max()) if frame_data.size else 0.0
        if max_value <= 1.0:
            frame_data = frame_data * 255.0
        frame_data = np.clip(frame_data, 0.0, 255.0)
    elif np.issubdtype(frame_data.dtype, np.bool_):
        frame_data = frame_data.astype(np.uint8) * 255
    else:
        frame_data = np.clip(frame_data, 0, 255)

    return np.ascontiguousarray(frame_data.astype(np.uint8))


class GenesisCameraVideoTrack(VideoStreamTrack):
    """aiortc track that streams RGB frames from a Genesis camera."""

    kind = "video"

    def __init__(
        self,
        camera: Any,
        fps: int = 30,
        max_render_retries: int = 3,
        retry_backoff_seconds: float = 0.02,
    ):
        super().__init__()
        if fps <= 0:
            raise ValueError("fps must be > 0.")
        self._camera = camera
        self._fps = int(fps)
        self._frame_interval = 1.0 / self._fps
        self._next_frame_time: float | None = None
        self._pts = 0
        self._time_base = Fraction(1, self._fps)
        self._max_render_retries = max(1, int(max_render_retries))
        self._retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self._last_good_frame: np.ndarray | None = None
        self._consecutive_failures = 0

    async def recv(self) -> VideoFrame:
        await self._sleep_until_next_frame()
        frame_array = await self._render_frame_with_retry()

        frame = VideoFrame.from_ndarray(frame_array, format="rgb24")
        frame.pts = self._pts
        frame.time_base = self._time_base
        self._pts += 1
        return frame

    async def _sleep_until_next_frame(self) -> None:
        now = time.monotonic()
        if self._next_frame_time is None:
            self._next_frame_time = now
            return

        self._next_frame_time += self._frame_interval
        sleep_time = self._next_frame_time - now
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
        else:
            # Reset the schedule if the producer is lagging too far behind.
            self._next_frame_time = now

    async def _render_frame_with_retry(self) -> np.ndarray:
        last_error: Exception | None = None

        for attempt in range(1, self._max_render_retries + 1):
            try:
                rgb_arr, _, _, _ = self._camera.render(
                    rgb=True,
                    depth=False,
                    segmentation=False,
                    normal=False,
                    force_render=False,
                )
                frame = _to_numpy_uint8_rgb(rgb_arr)
                self._last_good_frame = frame
                self._consecutive_failures = 0
                return frame
            except Exception as exc:
                last_error = exc
                if attempt < self._max_render_retries:
                    await asyncio.sleep(self._retry_backoff_seconds * attempt)

        self._consecutive_failures += 1
        if self._consecutive_failures in (1, 10) or self._consecutive_failures % 50 == 0:
            LOGGER.warning(
                "Camera render failed %s time(s) in a row; reusing the previous frame. Last error: %s",
                self._consecutive_failures,
                last_error,
            )

        if self._last_good_frame is not None:
            return self._last_good_frame

        width, height = self._camera.res
        return np.zeros((height, width, 3), dtype=np.uint8)


class WebRTCStreamer:
    """Small aiohttp + aiortc server that streams a Genesis camera to a browser."""

    def __init__(
        self,
        camera: Any,
        host: str = "0.0.0.0",
        port: int = 8000,
        fps: int = 30,
        ice_servers: Sequence[str | dict[str, Any]] | None = None,
        token: str | None = None,
        video_bitrate_bps: int | None = 8_000_000,
        allow_browser_shutdown: bool = False,
    ):
        self._camera = camera
        self._host = host
        self._port = int(port)
        self._fps = int(fps)
        self._token = token
        self._video_bitrate_bps = int(video_bitrate_bps) if video_bitrate_bps is not None else None
        self._allow_browser_shutdown = bool(allow_browser_shutdown)
        self._shutdown_requested = asyncio.Event()

        self._ice_servers_json = self._normalize_ice_servers_for_browser(ice_servers)
        rtc_ice_servers = self._parse_ice_servers(ice_servers)
        self._rtc_configuration = RTCConfiguration(iceServers=rtc_ice_servers)

        self._peer_connections: set[RTCPeerConnection] = set()
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.BaseSite | None = None

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    def create_app(self) -> web.Application:
        if self._app is None:
            app = web.Application()
            app.router.add_get("/", self._handle_index)
            app.router.add_post("/offer", self._handle_offer)
            if self._allow_browser_shutdown:
                app.router.add_post("/shutdown", self._handle_shutdown)
            app.on_shutdown.append(self._on_shutdown)
            self._app = app
        return self._app

    async def start(self) -> None:
        if self._runner is not None:
            return
        if self._shutdown_requested.is_set():
            self._shutdown_requested = asyncio.Event()
        app = self.create_app()
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()

    async def stop(self) -> None:
        self._shutdown_requested.set()
        if self._runner is None:
            return
        await self._runner.cleanup()
        self._runner = None
        self._site = None

    def run(self) -> None:
        asyncio.run(self._run_forever())

    async def wait_for_shutdown_request(self) -> None:
        await self._shutdown_requested.wait()

    async def _run_forever(self) -> None:
        await self.start()
        try:
            await self.wait_for_shutdown_request()
        finally:
            await self.stop()

    async def _handle_index(self, request: web.Request) -> web.Response:
        if not self._is_authorized(request):
            return web.Response(status=401, text="Unauthorized")
        return web.Response(
            text=self._index_html(self._ice_servers_json, allow_shutdown=self._allow_browser_shutdown),
            content_type="text/html",
        )

    async def _handle_offer(self, request: web.Request) -> web.Response:
        if not self._is_authorized(request):
            return web.json_response({"error": "Unauthorized"}, status=401)

        LOGGER.info("Received WebRTC offer from %s", request.remote)
        try:
            params = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON payload."}, status=400)

        if not isinstance(params, dict):
            return web.json_response({"error": "Offer payload must be a JSON object."}, status=400)

        sdp = params.get("sdp")
        offer_type = params.get("type")
        if offer_type != "offer":
            return web.json_response({"error": "SDP type must be 'offer'."}, status=400)
        if not isinstance(sdp, str) or not isinstance(offer_type, str):
            return web.json_response({"error": "Invalid SDP offer payload."}, status=400)

        offer = RTCSessionDescription(sdp=sdp, type=offer_type)
        peer_connection = RTCPeerConnection(configuration=self._rtc_configuration)
        self._peer_connections.add(peer_connection)

        @peer_connection.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            LOGGER.info("Peer connection state changed to %s", peer_connection.connectionState)
            if peer_connection.connectionState in {"failed", "closed"}:
                await peer_connection.close()
                self._peer_connections.discard(peer_connection)

        @peer_connection.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange() -> None:
            LOGGER.info("ICE connection state changed to %s", peer_connection.iceConnectionState)

        try:
            video_track = GenesisCameraVideoTrack(self._camera, fps=self._fps)
            sender = peer_connection.addTrack(video_track)
            self._prefer_h264_codec(peer_connection, sender)
            await self._configure_sender(sender)

            await peer_connection.setRemoteDescription(offer)
            answer = await peer_connection.createAnswer()
            await peer_connection.setLocalDescription(answer)
            await self._wait_for_ice_gathering(peer_connection)
        except Exception as exc:
            await peer_connection.close()
            self._peer_connections.discard(peer_connection)
            LOGGER.exception("Failed to process WebRTC offer: %s", exc)
            return web.json_response({"error": "Failed to process offer."}, status=500)

        local_description = peer_connection.localDescription
        if local_description is None:
            return web.json_response({"error": "Missing local description."}, status=500)
        return web.json_response({"sdp": local_description.sdp, "type": local_description.type})

    async def _handle_shutdown(self, request: web.Request) -> web.Response:
        if not self._is_authorized(request):
            return web.json_response({"error": "Unauthorized"}, status=401)
        LOGGER.info("Shutdown requested from %s", request.remote)
        self._shutdown_requested.set()
        return web.json_response({"status": "shutdown_requested"})

    async def _on_shutdown(self, _app: web.Application) -> None:
        peer_connections = list(self._peer_connections)
        self._peer_connections.clear()
        if peer_connections:
            await asyncio.gather(*(pc.close() for pc in peer_connections), return_exceptions=True)

    def _is_authorized(self, request: web.Request) -> bool:
        if self._token is None:
            return True
        query_token = request.query.get("token")
        header = request.headers.get("Authorization", "")
        return query_token == self._token or header == f"Bearer {self._token}"

    async def _configure_sender(self, sender: Any) -> None:
        if self._video_bitrate_bps is None:
            return

        try:
            params = sender.getParameters()
            encodings = getattr(params, "encodings", None)
            if encodings:
                for encoding in encodings:
                    if hasattr(encoding, "maxBitrate"):
                        encoding.maxBitrate = self._video_bitrate_bps
                    elif isinstance(encoding, dict):
                        encoding["maxBitrate"] = self._video_bitrate_bps
                result = sender.setParameters(params)
                if asyncio.iscoroutine(result):
                    await result
                LOGGER.info("Configured video bitrate cap to %.2f Mbps", self._video_bitrate_bps / 1_000_000.0)
        except Exception as exc:
            LOGGER.warning("Failed to apply sender bitrate settings: %s", exc)

    @staticmethod
    def _prefer_h264_codec(peer_connection: RTCPeerConnection, sender: Any) -> None:
        """Prefer H264 for broad browser compatibility while keeping codec fallbacks."""
        try:
            transceiver = next((t for t in peer_connection.getTransceivers() if t.sender == sender), None)
            if transceiver is None:
                return

            capabilities = RTCRtpSender.getCapabilities("video")
            if capabilities is None:
                return

            codecs = [codec for codec in capabilities.codecs if codec.mimeType.lower() != "video/rtx"]
            h264_codecs = [codec for codec in codecs if codec.mimeType.lower() == "video/h264"]
            if not h264_codecs:
                return

            preferred = h264_codecs + [codec for codec in codecs if codec not in h264_codecs]
            transceiver.setCodecPreferences(preferred)
        except Exception as exc:
            LOGGER.debug("Failed to prioritize H264 codec: %s", exc)

    @staticmethod
    async def _wait_for_ice_gathering(peer_connection: RTCPeerConnection, timeout: float = 5.0) -> None:
        if peer_connection.iceGatheringState == "complete":
            return

        complete = asyncio.Event()

        @peer_connection.on("icegatheringstatechange")
        def on_icegatheringstatechange() -> None:
            if peer_connection.iceGatheringState == "complete":
                complete.set()

        try:
            await asyncio.wait_for(complete.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            LOGGER.debug("Timed out waiting for ICE gathering completion.")

    @staticmethod
    def _parse_ice_servers(ice_servers: Sequence[str | dict[str, Any]] | None) -> list[RTCIceServer]:
        if not ice_servers:
            return []

        parsed_servers: list[RTCIceServer] = []
        for server in ice_servers:
            if isinstance(server, str):
                parsed_servers.append(RTCIceServer(urls=[server]))
                continue

            if isinstance(server, dict):
                urls = server.get("urls")
                if urls is None:
                    raise ValueError("ICE server dictionaries must contain a 'urls' key.")
                parsed_servers.append(
                    RTCIceServer(
                        urls=urls,
                        username=server.get("username"),
                        credential=server.get("credential"),
                    )
                )
                continue

            raise TypeError(f"Unsupported ICE server type: {type(server)}")

        return parsed_servers

    @staticmethod
    def _normalize_ice_servers_for_browser(ice_servers: Sequence[str | dict[str, Any]] | None) -> list[dict[str, Any]]:
        if not ice_servers:
            return []

        normalized_servers: list[dict[str, Any]] = []
        for server in ice_servers:
            if isinstance(server, str):
                normalized_servers.append({"urls": [server]})
            elif isinstance(server, dict):
                urls = server.get("urls")
                if urls is None:
                    raise ValueError("ICE server dictionaries must contain a 'urls' key.")
                if isinstance(urls, str):
                    urls = [urls]
                normalized_server = {"urls": list(urls)}
                if server.get("username") is not None:
                    normalized_server["username"] = server["username"]
                if server.get("credential") is not None:
                    normalized_server["credential"] = server["credential"]
                normalized_servers.append(normalized_server)
            else:
                raise TypeError(f"Unsupported ICE server type: {type(server)}")

        return normalized_servers

    @staticmethod
    def _index_html(ice_servers: list[dict[str, Any]], allow_shutdown: bool) -> str:
        html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Genesis WebRTC Stream</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      padding: 16px;
      font-family: sans-serif;
      background: #101417;
      color: #d9e2ec;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .app {
      width: min(95vw, 1024px);
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 10px;
      text-align: center;
    }
    h1 { margin: 0; font-size: 1.15rem; }
    #status {
      margin: 0;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid #486581;
      background: #243b53;
      color: #d9e2ec;
      font-weight: 600;
      min-height: 38px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .state-row {
      width: 100%;
      display: flex;
      justify-content: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .controls {
      display: flex;
      justify-content: center;
      gap: 10px;
    }
    .chip {
      border-radius: 999px;
      padding: 6px 10px;
      border: 1px solid #486581;
      font-size: 0.9rem;
      background: #243b53;
      color: #d9e2ec;
    }
    .btn {
      border-radius: 8px;
      border: 1px solid #486581;
      background: #1f2f41;
      color: #d9e2ec;
      padding: 8px 12px;
      cursor: pointer;
      font-weight: 600;
    }
    .btn:hover { filter: brightness(1.1); }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .btn-danger { border-color: #c53030; background: #521b1b; color: #feb2b2; }
    .tone-info { border-color: #486581; background: #243b53; color: #d9e2ec; }
    .tone-ok { border-color: #2f855a; background: #1c4532; color: #9ae6b4; }
    .tone-warn { border-color: #b7791f; background: #4a2e0f; color: #fbd38d; }
    .tone-err { border-color: #c53030; background: #521b1b; color: #feb2b2; }
    video {
      width: 100%;
      border: 1px solid #243b53;
      background: #000;
      display: block;
    }
  </style>
</head>
<body>
  <main class="app">
    <h1>Genesis Headless WebRTC Stream</h1>
    <p id="status" class="tone-info">Connecting...</p>
    <div class="state-row">
      <span id="peer-state" class="chip tone-info">Peer: new</span>
      <span id="ice-state" class="chip tone-info">ICE: new</span>
    </div>
    <div class="controls">
      <button id="close-btn" class="btn btn-danger" disabled>Close Stream</button>
    </div>
    <video id="video" autoplay playsinline></video>
  </main>
  <script>
    const statusEl = document.getElementById("status");
    const peerStateEl = document.getElementById("peer-state");
    const iceStateEl = document.getElementById("ice-state");
    const closeBtn = document.getElementById("close-btn");
    const video = document.getElementById("video");
    video.muted = true;
    video.setAttribute("muted", "muted");
    const iceServers = __ICE_SERVERS__;
    const allowShutdown = __ALLOW_SHUTDOWN__;
    let currentPc = null;
    let fallbackStream = null;
    const params = new URLSearchParams(window.location.search);
    const token = params.get("token");
    const withToken = (path) => token ? `${path}?token=${encodeURIComponent(token)}` : path;
    const offerPath = withToken("/offer");
    const shutdownPath = withToken("/shutdown");

    function toneForState(state) {
      if (state === "connected" || state === "completed") return "tone-ok";
      if (state === "disconnected" || state === "checking" || state === "connecting") return "tone-warn";
      if (state === "failed" || state === "closed") return "tone-err";
      return "tone-info";
    }

    function setStatus(message, toneClass = "tone-info") {
      statusEl.textContent = message;
      statusEl.className = toneClass;
    }

    function setChipState(chipEl, label, value) {
      const toneClass = toneForState(value);
      chipEl.textContent = `${label}: ${value}`;
      chipEl.className = `chip ${toneClass}`;
    }

    function closeCurrentStream() {
      if (currentPc) {
        currentPc.close();
        currentPc = null;
      }
      if (video.srcObject) {
        for (const track of video.srcObject.getTracks()) {
          track.stop();
        }
      }
      if (fallbackStream) {
        for (const track of fallbackStream.getTracks()) {
          track.stop();
        }
      }
      fallbackStream = null;
      video.srcObject = null;
      setChipState(peerStateEl, "Peer", "closed");
      setChipState(iceStateEl, "ICE", "closed");
      setStatus("Stream closed by user.", "tone-info");
      closeBtn.disabled = !allowShutdown;
    }

    function waitForIceGatheringComplete(pc) {
      if (pc.iceGatheringState === "complete") {
        return Promise.resolve();
      }
      return new Promise((resolve) => {
        function checkState() {
          if (pc.iceGatheringState === "complete") {
            pc.removeEventListener("icegatheringstatechange", checkState);
            resolve();
          }
        }
        pc.addEventListener("icegatheringstatechange", checkState);
      });
    }

    async function startStream() {
      const pc = new RTCPeerConnection({ iceServers });
      currentPc = pc;
      closeBtn.disabled = false;
      pc.addTransceiver("video", { direction: "recvonly" });

      pc.ontrack = async (event) => {
        const stream = event.streams && event.streams.length > 0 ? event.streams[0] : null;
        if (stream) {
          video.srcObject = stream;
        } else {
          if (!fallbackStream) {
            fallbackStream = new MediaStream();
          }
          fallbackStream.addTrack(event.track);
          video.srcObject = fallbackStream;
        }
        try {
          await video.play();
        } catch (error) {
          setStatus(`Connected, but autoplay was blocked: ${error}`, "tone-warn");
          return;
        }
        setStatus("Receiving video", "tone-ok");
      };

      pc.onconnectionstatechange = () => {
        setChipState(peerStateEl, "Peer", pc.connectionState);
        setStatus(`Peer connection state: ${pc.connectionState}`, toneForState(pc.connectionState));
      };

      pc.oniceconnectionstatechange = () => {
        setChipState(iceStateEl, "ICE", pc.iceConnectionState);
        if (pc.iceConnectionState === "failed") {
          setStatus("ICE failed. Check STUN/TURN and firewall settings.", "tone-err");
        }
      };

      try {
        setStatus("Creating offer...", "tone-info");
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        await waitForIceGatheringComplete(pc);

        setStatus("Sending offer...", "tone-info");
        const response = await fetch(offerPath, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(pc.localDescription),
        });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const answer = await response.json();
        await pc.setRemoteDescription(answer);
        setStatus("WebRTC negotiation complete", "tone-ok");
      } catch (error) {
        setStatus(`Connection failed: ${error}`, "tone-err");
      }
    }

    if (!allowShutdown) {
      closeBtn.title = "Server-side shutdown is disabled.";
      closeBtn.disabled = true;
    }

    closeBtn.addEventListener("click", async () => {
      closeCurrentStream();
      if (!allowShutdown) {
        return;
      }
      try {
        setStatus("Requesting server shutdown...", "tone-info");
        const response = await fetch(shutdownPath, { method: "POST" });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        setStatus("Server shutdown requested. CLI process should exit shortly.", "tone-ok");
        closeBtn.disabled = true;
      } catch (error) {
        setStatus(`Failed to request shutdown: ${error}`, "tone-err");
      }
    });
    startStream();
  </script>
</body>
</html>
"""
        return html.replace("__ICE_SERVERS__", json.dumps(ice_servers)).replace(
            "__ALLOW_SHUTDOWN__", "true" if allow_shutdown else "false"
        )
