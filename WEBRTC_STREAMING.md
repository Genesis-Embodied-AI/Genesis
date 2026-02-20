# Headless WebRTC Camera Streaming (MVP)

Use this to view a Genesis camera remotely in a browser on headless / SSH machines.

## Install

```bash
pip install -e ".[webrtc]"
```

## Run

Basic scene example:

```bash
python examples/streaming/webrtc_camera.py --host 0.0.0.0 --port 8000 --width 1280 --height 720 --video-bitrate-mbps 8
```

Manipulation environment example:

```bash
python examples/streaming/webrtc_manipulation_env.py --host 0.0.0.0 --port 8000 --width 640 --height 480 --video-bitrate-mbps 10
```

Then open:

```text
http://<server-ip>:8000
```

The example creates `gs.Scene(show_viewer=False)` and a camera with `GUI=False`, so it works without an interactive
display.

## Optional ICE / Token

- CLI: `--ice-server stun:stun.l.google.com:19302` (repeatable)
- Env: `GS_WEBRTC_ICE_SERVERS=stun:stun.l.google.com:19302,turn:turn.example.com:3478`
- Optional token:
  - CLI: `--token my-secret`
  - Env: `GS_WEBRTC_TOKEN=my-secret`

## Networking Notes

- Signaling HTTP uses the TCP port you set (`--port`, default `8000`).
- WebRTC media typically uses dynamic UDP ports for peer-to-peer transport.
- Open firewall / cloud security group rules accordingly.

## Browser Controls

- The stream page includes a `Close Stream` button that:
  - closes the current WebRTC connection in the browser
  - requests server shutdown so the example CLI process exits

## Quality Tips

- Increase camera resolution (`--width`, `--height`) for sharper output.
- Increase WebRTC encoder target (`--video-bitrate-mbps`) to reduce compression artifacts.
- Reduce `--fps` if bandwidth is limited and you need better per-frame quality.
