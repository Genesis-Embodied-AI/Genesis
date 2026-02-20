# Genesis Streaming Examples

Simple headless WebRTC examples for viewing Genesis camera output in a browser.

## Prerequisites

```bash
pip install -e ".[webrtc]"
```

## Run Examples

Basic scene camera:

```bash
python3 examples/streaming/webrtc_camera.py --host 0.0.0.0 --port 8000
```

Manipulation environment camera:

```bash
python3 examples/streaming/webrtc_manipulation_env.py --host 0.0.0.0 --port 8000
```

Open:

```text
http://<server-ip>:8000
```

The page provides a `Close Stream` button that closes the current peer connection and requests server shutdown so the CLI exits.

## Quality Tuning

- Increase resolution: `--width 1280 --height 720` (or higher)
- Increase bitrate: `--video-bitrate-mbps 8` (or higher)
- Reduce FPS if network is limited: `--fps 20`

## Common CLI Options

- `--host`, `--port`, `--public-host`
- `--fps`, `--video-bitrate-mbps`
- `--ice-server` (repeatable), `--no-default-stun`
- `--token` (or `GS_WEBRTC_TOKEN`)

## Tests

Run the streaming module test:

```bash
uv run pytest tests/test_webrtc_streaming.py
```

If WebRTC deps are not installed in the environment, the test is expected to skip.

## GitHub Workflow (Clone, Branch, Push, PR)

1. Clone and enter repo:

```bash
git clone https://github.com/Genesis-Embodied-AI/Genesis.git
cd Genesis
```

2. Create a feature branch:

```bash
git checkout -b feature/webrtc-streaming
```

3. Install deps and run tests:

```bash
pip install -e ".[webrtc]"
uv run pytest tests/test_webrtc_streaming.py
```

4. Commit cleanly:

```bash
git add genesis/vis/streaming examples/streaming tests/test_webrtc_streaming.py WEBRTC_STREAMING.md README.md pyproject.toml
git commit -m "[FEATURE] Add headless WebRTC streaming MVP with examples and docs"
```

5. Push branch:

```bash
git push -u origin feature/webrtc-streaming-mvp
```

