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


def display_stereo_images(env, step_count):
    """Display left and right RGB images side by side in one window."""
    # Get individual camera images
    rgb_left, _, _, _ = env.left_cam.render(rgb=True, depth=False)
    rgb_right, _, _, _ = env.right_cam.render(rgb=True, depth=False)

    # Convert to numpy arrays and format for OpenCV
    left_img = rgb_left[0].cpu().numpy()  # Take first environment
    right_img = rgb_right[0].cpu().numpy()  # Take first environment

    # Convert from RGB to BGR for OpenCV
    left_img = cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR)
    right_img = cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR)

    # Resize images for better display (optional)
    scale = 2.0
    left_img = cv2.resize(left_img, (int(left_img.shape[1] * scale), int(left_img.shape[0] * scale)))
    right_img = cv2.resize(right_img, (int(right_img.shape[1] * scale), int(right_img.shape[0] * scale)))

    # Add text labels
    cv2.putText(
        left_img,
        f"Left Camera - Step {step_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        right_img,
        f"Right Camera - Step {step_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

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

    # Display combined image
    cv2.imshow("Stereo Cameras", stereo_img)

    # Wait for key press (1ms delay)
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        return False
    return True


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
    parser.add_argument("--record", action="store_true", default=False)
    args = parser.parse_args()

    # Set PyTorch default dtype to float32 for better performance
    torch.set_default_dtype(torch.float32)

    gs.init()

    log_dir = f"logs/{args.exp_name + '_' + args.stage}"

    # Load configurations
    if args.stage == "rl":
        # For RL, load the standard configs
        env_cfg, reward_cfg, robot_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    else:
        # For BC, we need to load the configs and create BC config
        env_cfg, reward_cfg, robot_cfg, rl_train_cfg, bc_train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
        train_cfg = bc_train_cfg

    # set the max FPS for visualization
    env_cfg["max_visualize_FPS"] = 60
    # set the box collision
    env_cfg["box_collision"] = True
    # set the box fixed
    env_cfg["box_fixed"] = False

    env = GraspEnv(
        num_envs=10,
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=True,
    )

    # Load the appropriate policy based on model type
    if args.stage == "rl":
        policy = load_rl_policy(env, train_cfg, log_dir)
    else:
        policy = load_bc_policy(env, train_cfg, log_dir)
        # Verify policy is float32
        print(f"Policy dtype: {next(policy.parameters()).dtype}")

    obs, _ = env.reset()

    max_sim_step = int(env_cfg["episode_length_s"] * env_cfg["max_visualize_FPS"])
    with torch.no_grad():
        if args.record:
            for step in range(max_sim_step):
                if args.stage == "rl":
                    actions = policy(obs)
                else:
                    # Get stereo grayscale images and ensure float32
                    rgb_obs = env.get_stereo_rgb_images(normalize=True).float()
                    ee_pose = env.robot.ee_pose.float()

                    # Diagnostic prints for first few steps
                    if step < 3:
                        print(f"Step {step}: RGB obs shape: {rgb_obs.shape}, dtype: {rgb_obs.dtype}")
                        print(f"Step {step}: EE pose shape: {ee_pose.shape}, dtype: {ee_pose.dtype}")

                    actions = policy(rgb_obs, ee_pose)
                obs, rews, dones, infos = env.step(actions)
            env.grasp_and_lift_demo(render=True)
        else:
            for step in range(max_sim_step):
                if args.stage == "rl":
                    actions = policy(obs)
                else:
                    # Get stereo grayscale images and ensure float32
                    rgb_obs = env.get_stereo_rgb_images(normalize=True).float()
                    ee_pose = env.robot.ee_pose.float()

                    # Diagnostic prints for first few steps
                    if step < 3:
                        print(f"Step {step}: RGB obs shape: {rgb_obs.shape}, dtype: {rgb_obs.dtype}")
                        print(f"Step {step}: EE pose shape: {ee_pose.shape}, dtype: {ee_pose.dtype}")

                    actions = policy(rgb_obs, ee_pose)

                    # Display stereo images for BC evaluation
                    if not display_stereo_images(env, step):
                        print("Evaluation stopped by user (ESC key pressed)")
                        break

                obs, rews, dones, infos = env.step(actions)
            env.grasp_and_lift_demo(render=False)

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

# Note
If you experience slow performance or encounter other issues
during evaluation, try removing the --record option.
"""
