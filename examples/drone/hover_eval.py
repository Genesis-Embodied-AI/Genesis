import argparse
import os
import pickle

import torch
from hover_env import HoverEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="drone-hovering")
    parser.add_argument("--ckpt", type=int, default=500)
    parser.add_argument("--record", action="store_true", default=False)
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    env = HoverEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = env.reset()
    if args.record:
        env.cam.start_recording()

    max_sim_step = int(env_cfg["episode_length_s"]/0.01)    # 0.01 is the simulation time step
    with torch.no_grad():
        for i in range(max_sim_step):
            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)
            if args.record:
                env.cam.render()
    if args.record:
        env.cam.stop_recording(save_to_filename="video.mp4", fps=60)

if __name__ == "__main__":
    main()

"""
# evaluation
python examples/drone/hover_eval.py
"""
