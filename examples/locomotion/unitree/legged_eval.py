import argparse
import os
import pickle

import torch
from legged_env import LeggedEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
import copy
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go1_walking")
    parser.add_argument("--ckpt", type=int, default=100)
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    # Get all subdirectories in the base log directory
    subdirs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]

    # Sort subdirectories by their names (assuming they are timestamped in lexicographical order)
    most_recent_subdir = sorted(subdirs)[-1] if subdirs else None
    log_dir = os.path.join(log_dir, most_recent_subdir)
    env_cfg, obs_cfg, noise_cfg, reward_cfg, command_cfg, train_cfg, terrain_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    env = LeggedEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        noise_cfg=noise_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        terrain_cfg=terrain_cfg,
        show_viewer=True,
    )

    

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    # List all files in the most recent subdirectory
    files = os.listdir(log_dir)

    # Regex to match filenames like 'model_100.pt' and extract the number
    model_files = [(f, int(re.search(r'model_(\d+)\.pt', f).group(1)))
                for f in files if re.search(r'model_(\d+)\.pt', f)]
    model_file = max(model_files, key=lambda x: x[1])[0]

    resume_path = os.path.join(log_dir,  model_file)
    runner.load(resume_path)
    # resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    # runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")
    # export policy as a jit module (used to run it from C++)
    EXPORT_POLICY = True
    if EXPORT_POLICY:
        path = os.path.join(log_dir, 'exported', 'policies')
        # export_policy_as_jit(runner.alg.actor_critic, path)
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(runner.alg.actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)
        print('Exported policy as jit script to: ', path)
        # Convert the policy to a version-less format
        versionless_path = os.path.join(log_dir, 'exported', 'policies', "policy_safe.pt")
        loaded_model = torch.jit.load(path)
        loaded_model.eval()
        loaded_model.save(versionless_path)
        print("Model successfully converted to version-less format: ", versionless_path)
    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go1_eval.py -e go1-walking -v --ckpt 100
"""
