"""
Headless WebRTC streaming example using a plain Genesis Scene camera.

Usage:
    python examples/streaming/webrtc_camera.py --host 0.0.0.0 --port 8000

Then open the printed terminal URL:
    WebRTC viewer URL: http://<server-ip>:8000

Use --public-host to override the host shown in the printed URL.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure local repository package is imported when running this script directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import genesis as gs

from genesis.vis.streaming import (
    WebRTCStreamer,
    add_streamer_cli_args,
    build_stream_url,
    collect_ice_servers_from_args,
    resolve_stream_token,
    video_bitrate_bps_from_args,
)


async def _step_scene(scene: gs.Scene, step_fps: float) -> None:
    step_dt = 1.0 / step_fps
    while True:
        scene.step()
        await asyncio.sleep(step_dt)


async def _run(args: argparse.Namespace) -> None:
    gs.init()

    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        gs.morphs.Box(
            size=(0.4, 0.4, 0.4),
            pos=(0.0, 0.0, 1.0),
        )
    )

    camera = scene.add_camera(
        res=(args.width, args.height),
        pos=(2.5, -2.0, 1.8),
        lookat=(0.0, 0.0, 0.6),
        fov=45,
        GUI=False,
    )
    scene.build()

    ice_servers = collect_ice_servers_from_args(args)
    token = resolve_stream_token(args)

    streamer = WebRTCStreamer(
        camera=camera,
        host=args.host,
        port=args.port,
        fps=args.fps,
        ice_servers=ice_servers,
        token=token,
        video_bitrate_bps=video_bitrate_bps_from_args(args),
        allow_browser_shutdown=True,
    )
    await streamer.start()

    url = build_stream_url(args.host, args.port, token=token, public_host=args.public_host)
    print(f"WebRTC viewer URL: {url}")

    step_task = asyncio.create_task(_step_scene(scene, args.step_fps))
    try:
        await streamer.wait_for_shutdown_request()
    finally:
        step_task.cancel()
        await asyncio.gather(step_task, return_exceptions=True)
        await streamer.stop()
        scene.destroy()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genesis headless WebRTC camera streaming example.")
    add_streamer_cli_args(parser, default_fps=30, default_video_bitrate_mbps=8.0)
    parser.add_argument("--step-fps", type=float, default=60.0, help="Scene stepping frequency.")
    parser.add_argument("--width", type=int, default=1280, help="Camera width.")
    parser.add_argument("--height", type=int, default=720, help="Camera height.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
