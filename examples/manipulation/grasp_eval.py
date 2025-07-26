import argparse
import os
import re
import pickle
from importlib import metadata

import torch

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="grasp")
    parser.add_argument("--stage", type=str, default="rl")
    parser.add_argument("--record", action="store_true", default=False)
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name + '_' + args.stage}"
    env_cfg, reward_cfg, robot_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))

    # for video recording
    env_cfg["visualize_camera"] = args.record
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

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    #
    checkpoint_files = [f for f in os.listdir(log_dir) if re.match(r"model_\d+\.pt", f)]
    last_ckpt = sorted(checkpoint_files)[-1]
    runner.load(os.path.join(log_dir, last_ckpt))
    print(f"Loaded checkpoint from {last_ckpt}")
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()

    max_sim_step = int(env_cfg["episode_length_s"] * env_cfg["max_visualize_FPS"])
    with torch.no_grad():
        if args.record:
            env.cam.start_recording()
            for _ in range(max_sim_step):
                actions = policy(obs)
                obs, rews, dones, infos = env.step(actions)
                env.cam.render()
            env.grasp_and_lift_demo(render=True)
            env.cam.stop_recording(save_to_filename="video.mp4", fps=env_cfg["max_visualize_FPS"])
        else:
            for _ in range(max_sim_step):
                actions = policy(obs)
                obs, rews, dones, infos = env.step(actions)
            env.grasp_and_lift_demo(render=False)


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/manipulation/grasp_eval.py

# Note
If you experience slow performance or encounter other issues
during evaluation, try removing the --record option.
"""
