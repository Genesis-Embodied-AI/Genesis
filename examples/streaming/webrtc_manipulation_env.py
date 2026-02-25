"""
Headless WebRTC streaming example using the manipulation `GraspEnv`.

Usage:
    python examples/streaming/webrtc_manipulation_env.py --host 0.0.0.0 --port 8000

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

MANIP_DIR = REPO_ROOT / "examples" / "manipulation"
if str(MANIP_DIR) not in sys.path:
    sys.path.insert(0, str(MANIP_DIR))

import torch

import genesis as gs
from grasp_env import GraspEnv

from genesis.vis.streaming import (
    WebRTCStreamer,
    add_streamer_cli_args,
    build_stream_url,
    collect_ice_servers_from_args,
    resolve_stream_token,
    video_bitrate_bps_from_args,
)


def _get_grasp_cfgs(num_envs: int, image_resolution: tuple[int, int]) -> tuple[dict, dict, dict]:
    env_cfg = {
        "num_envs": num_envs,
        "num_obs": 14,
        "num_actions": 6,
        "action_scales": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        "episode_length_s": 3.0,
        "ctrl_dt": 0.01,
        "box_size": [0.08, 0.03, 0.06],
        "box_collision": False,
        "box_fixed": True,
        "image_resolution": image_resolution,
        "use_rasterizer": True,
        "visualize_camera": False,
    }
    reward_cfg = {"keypoints": 1.0}
    robot_cfg = {
        "ee_link_name": "hand",
        "gripper_link_names": ["left_finger", "right_finger"],
        "default_arm_dof": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
        "default_gripper_dof": [0.04, 0.04],
        "ik_method": "dls_ik",
    }
    return env_cfg, reward_cfg, robot_cfg


async def _step_env(env: GraspEnv, step_fps: float) -> None:
    step_dt = 1.0 / step_fps
    zero_actions = torch.zeros((env.num_envs, env.num_actions), device=gs.device, dtype=gs.tc_float)
    while True:
        env.step(zero_actions)
        await asyncio.sleep(step_dt)


async def _run(args: argparse.Namespace) -> None:
    gs.init(backend=gs.gpu, precision="32", performance_mode=True, logging_level="warning")

    if args.num_envs < 10:
        raise ValueError("`--num-envs` must be >= 10 for the current `GraspEnv` rendered_envs configuration.")

    env_cfg, reward_cfg, robot_cfg = _get_grasp_cfgs(
        num_envs=args.num_envs,
        image_resolution=(args.width, args.height),
    )
    env = GraspEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=False,
    )
    camera = env.left_cam

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

    step_task = asyncio.create_task(_step_env(env, args.step_fps))
    try:
        await streamer.wait_for_shutdown_request()
    finally:
        step_task.cancel()
        await asyncio.gather(step_task, return_exceptions=True)
        await streamer.stop()
        env.scene.destroy()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genesis headless WebRTC manipulation streaming example.")
    add_streamer_cli_args(parser, default_fps=30, default_video_bitrate_mbps=10.0)
    parser.add_argument("--step-fps", type=float, default=60.0, help="Manipulation env stepping frequency.")
    parser.add_argument("--num-envs", type=int, default=10, help="Number of manipulation environments (must be >= 10).")
    parser.add_argument("--width", type=int, default=640, help="Camera width.")
    parser.add_argument("--height", type=int, default=480, help="Camera height.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
