import argparse
import os
import pickle
import shutil

from g1_env import G1Env
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 24,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 500,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
           'left_hip_yaw_joint': 0.0,
           'left_hip_roll_joint': 0.0,
           'left_hip_pitch_joint': -0.1,
           'left_knee_joint': 0.3,
           'left_ankle_pitch_joint': -0.2,
           'left_ankle_roll_joint': 0.0,
           'right_hip_yaw_joint': 0.0,
           'right_hip_roll_joint': 0.0,
           'right_hip_pitch_joint': -0.1,
           'right_knee_joint': 0.3,
           'right_ankle_pitch_joint': -0.2,
           'right_ankle_roll_joint': 0.0,
        },
        "dof_names": [
            'left_hip_yaw_joint',
            'left_hip_roll_joint',
            'left_hip_pitch_joint',
            'left_knee_joint',
            'left_ankle_pitch_joint',
            'left_ankle_roll_joint',
            'right_hip_yaw_joint',
            'right_hip_roll_joint',
            'right_hip_pitch_joint',
            'right_knee_joint',
            'right_ankle_pitch_joint',
            'right_ankle_roll_joint',
        ],
        # PD
        "kp": 120.0,
        "kd": 3.0,
        # termination
        "terminate_after_contacts_on": ["pelvis"],
        "termination_if_pelvis_z_less_than": 0.2,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.8],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 10.0,
        "clip_observations": 10.0,
    }
    obs_cfg = {
        "num_obs": 47,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.78,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.5,
            "alive": 0.15,
            "gait_contact": 0.18,
            "gait_swing": -0.18,
            "lin_vel_z": -2.0,
            "ang_vel_xy": -0.05,
            "base_height": -10.0,
            "action_rate": -0.01,
            "contact_no_vel": -0.2,
            "feet_swing_height": -20.0,
            "orientation": -1.0,
            "hip_pos": -1.0,
            "dof_vel": -0.001,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 0.5],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }
    domain_rand_cfg = {
        'randomize_friction': True,
        'friction_range': [0.001, 1.25],
        'randomize_base_mass': True,
        'added_mass_range': [-1.0, 1.0],
        'push_robots': True,
        'push_interval_s': 3.5, # seconds
        'max_push_vel_xy': 1.5, # meters/seconds
        'max_push_vel_rp': 800.0, # degrees/seconds
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg, domain_rand_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="g1-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=500)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, domain_rand_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = G1Env(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg,
        reward_cfg=reward_cfg, command_cfg=command_cfg,
        domain_rand_cfg=domain_rand_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, domain_rand_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/g1_train.py
"""
