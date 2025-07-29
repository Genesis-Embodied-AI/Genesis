import argparse
import os
import re
import pickle
from importlib import metadata

import torch
import cv2
import numpy as np

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from grasp_env import GraspEnv
from behavior_cloning import BehaviorCloning


def load_rl_policy(env, train_cfg, log_dir):
    """Load reinforcement learning policy."""
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    # Find the latest checkpoint
    checkpoint_files = [f for f in os.listdir(log_dir) if re.match(r"model_\d+\.pt", f)]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")

    last_ckpt = sorted(checkpoint_files)[-1]
    runner.load(os.path.join(log_dir, last_ckpt))
    print(f"Loaded RL checkpoint from {last_ckpt}")

    return runner.get_inference_policy(device=gs.device)


def load_bc_policy(env, bc_cfg, log_dir):
    """Load behavior cloning policy."""
    # Create behavior cloning instance
    bc_runner = BehaviorCloning(env, bc_cfg, None, device=gs.device)

    # Find the latest checkpoint
    checkpoint_files = [f for f in os.listdir(log_dir) if re.match(r"checkpoint_\d+\.pt", f)]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")

    last_ckpt = sorted(checkpoint_files)[-1]
    last_ckpt_file = os.path.join(log_dir, last_ckpt)
    print(f"Loaded BC checkpoint from {last_ckpt_file}")
    bc_runner.load(last_ckpt_file)

    return bc_runner._policy


def get_stereo_frame(env, step_count):
    """Get stereo frame as numpy array."""
    # Get individual camera images
    rgb_left, _, _, _ = env.left_cam.render(rgb=True, depth=False)
    rgb_right, _, _, _ = env.right_cam.render(rgb=True, depth=False)

    # Convert to numpy arrays and format for OpenCV
    left_img = rgb_left[0].cpu().numpy()  # Take first environment
    right_img = rgb_right[0].cpu().numpy()  # Take first environment

    # Convert from RGB to BGR for OpenCV
    left_img = cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR)
    right_img = cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR)

    # Resize images for better display
    scale = 2.0
    left_img = cv2.resize(left_img, (int(left_img.shape[1] * scale), int(left_img.shape[0] * scale)))
    right_img = cv2.resize(right_img, (int(right_img.shape[1] * scale), int(right_img.shape[0] * scale)))
    # Concatenate images horizontally (side by side)
    stereo_img = np.hstack([left_img, right_img])
    # Add a vertical line separator between the two images
    separator_x = left_img.shape[1]
    cv2.line(
        stereo_img,
        (separator_x, 0),
        (separator_x, stereo_img.shape[0]),
        (255, 255, 255),
        2,
    )
    return stereo_img


def display_stereo_images(env, step_count):
    """Display left and right RGB images side by side in one window."""
    stereo_img = get_stereo_frame(env, step_count)

    # Display combined image
    cv2.imshow("Stereo Cameras", stereo_img)

    # Wait for key press (1ms delay)
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        return False
    return True


def save_frames_as_video(frames, video_path, fps=60):
    """Save a list of frames as a video."""
    if not frames:
        print("No frames to save!")
        return

    # Get frame dimensions from the first frame
    height, width = frames[0].shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for {video_path}")
        return

    # Write frames to video
    print(f"Saving {len(frames)} frames to {video_path}...")
    for i, frame in enumerate(frames):
        video_writer.write(frame)
        if i % 30 == 0:  # Progress indicator every 30 frames
            print(f"Progress: {i + 1}/{len(frames)} frames")

    # Release video writer
    video_writer.release()
    print(f"Video saved successfully: {video_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="grasp")
    parser.add_argument(
        "--stage",
        type=str,
        default="rl",
        choices=["rl", "bc"],
        help="Model type: 'rl' for reinforcement learning, 'bc' for behavior cloning",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record stereo images as video during evaluation",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path to save the video file (default: auto-generated)",
    )
    args = parser.parse_args()

    # Set PyTorch default dtype to float32 for better performance
    torch.set_default_dtype(torch.float32)

    gs.init()

    log_dir = f"logs/{args.exp_name + '_' + args.stage}"

    # Load configurations
    if args.stage == "rl":
        # For RL, load the standard configs
        env_cfg, reward_cfg, robot_cfg, rl_train_cfg, bc_train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    else:
        # For BC, we need to load the configs and create BC config
        env_cfg, reward_cfg, robot_cfg, rl_train_cfg, bc_train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))

    # set the max FPS for visualization
    env_cfg["max_visualize_FPS"] = 60
    # set the box collision
    env_cfg["box_collision"] = True
    # set the box fixed
    env_cfg["box_fixed"] = False
    # set the number of envs for evaluation
    env_cfg["num_envs"] = 10

    env = GraspEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=True,
    )

    # Load the appropriate policy based on model type
    if args.stage == "rl":
        policy = load_rl_policy(env, rl_train_cfg, log_dir)
    else:
        policy = load_bc_policy(env, bc_train_cfg, log_dir)
        policy.eval()
        # Verify policy is float32
        print(f"Policy dtype: {next(policy.parameters()).dtype}")

    obs, _ = env.reset()

    # Initialize frame list for recording
    frames = []
    max_sim_step = int(env_cfg["episode_length_s"] * env_cfg["max_visualize_FPS"])

    with torch.no_grad():
        for step in range(max_sim_step):
            if args.stage == "rl":
                actions = policy(obs)
            else:
                # Get stereo grayscale images and ensure float32
                rgb_obs = env.get_stereo_rgb_images(normalize=True).float()
                ee_pose = env.robot.ee_pose.float()

                actions = policy(rgb_obs, ee_pose)

                # Display stereo images for BC evaluation
                if not display_stereo_images(env, step):
                    print("Evaluation stopped by user (ESC key pressed)")
                    break

                # Collect frame for video recording
                if args.record:
                    frame = get_stereo_frame(env, step)
                    frames.append(frame)

            obs, rews, dones, infos = env.step(actions)
        env.grasp_and_lift_demo()

        # Save video if recording was enabled
        if args.record and frames:
            video_path = f"{log_dir}/stereo_evaluation.mp4"
            # Save frames as video
            fps = env_cfg["max_visualize_FPS"]
            save_frames_as_video(frames, video_path, fps)

        # Close OpenCV windows if BC evaluation
        if args.stage == "bc":
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""
# evaluation
# For reinforcement learning model:
python examples/manipulation/grasp_eval.py --stage rl

# For behavior cloning model:
python examples/manipulation/grasp_eval.py --stage bc

# With video recording:
python examples/manipulation/grasp_eval.py --stage bc --record
"""
