import argparse
import os
import pickle
import shutil
from importlib import metadata

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

from allegro_env import AllegroEnv


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
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
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 16,  # Allegro hand has 16 DOF
        # joint/link names - Allegro hand joint names
        "default_joint_angles": {  # [rad] - initial grasping pose
            "index_joint_0": 0.1,
            "index_joint_1": 0.0,
            "index_joint_2": -0.1,
            "index_joint_3": 0.7,
            "middle_joint_0": 0.6,
            "middle_joint_1": 0.6,
            "middle_joint_2": 0.6,
            "middle_joint_3": 1.0,
            "ring_joint_0": 0.65,
            "ring_joint_1": 0.65,
            "ring_joint_2": 0.65,
            "ring_joint_3": 1.0,
            "thumb_joint_0": 0.6,
            "thumb_joint_1": 0.6,
            "thumb_joint_2": 0.6,
            "thumb_joint_3": 0.7,
        },
        "joint_names": [
            "index_joint_0",
            "index_joint_1",
            "index_joint_2",
            "index_joint_3",
            "middle_joint_0",
            "middle_joint_1",
            "middle_joint_2",
            "middle_joint_3",
            "ring_joint_0",
            "ring_joint_1",
            "ring_joint_2",
            "ring_joint_3",
            "thumb_joint_0",
            "thumb_joint_1",
            "thumb_joint_2",
            "thumb_joint_3",
        ],
        # PD control parameters
        "kp": 40.0,
        "kd": 1.0,
        # termination conditions
        "termination_height": 0.1,  # terminate if object falls below this height
        "max_obj_distance": 0.3,  # terminate if object moves too far from hand
        # initial poses
        "hand_init_pos": [0.0, 0.0, 0.1],
        "hand_init_euler": [0.0, -90.0, 0.0],
        "obj_init_pos": [-0.02, 0.0, 0.22],
        "obj_init_euler": [0.0, 0.0, 0.0],  # roll, pitch, yaw in degrees
        "obj_size": [0.09, 0.12, 0.03],
        # episode settings
        "episode_length_s": 10.0,
        "resampling_time_s": 2.0,
        "action_scale": 0.5,
        "clip_actions": 10.0,
    }

    # Calculate observation dimensions
    # obj_pos (3) + obj_euler (3) + obj_ang_vel (3) + commands (3) + dof_pos (16) + dof_vel (16) + actions (16) + tactile (4*32=128)
    num_tactile_sensors = 4  # index, middle, ring, thumb tips
    tactile_dims = num_tactile_sensors * 32  # 4x4x2 grid per sensor

    obs_cfg = {
        "num_obs": 3 + 3 + 3 + 3 + 16 + 16 + 16 + tactile_dims,  # 188 total
        "obs_scales": {
            "obj_pos": 10.0,
            "obj_euler": 0.01,  # degrees to radians scale
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "tactile": 0.01,
        },
    }

    reward_cfg = {
        "tracking_sigma": 0.25,
        "target_height": 0.22,
        "target_contact": 50.0,  # target tactile sensor activation
        "reward_scales": {
            "tracking_ang_vel": 2.0,
            "object_stable": 1.0,
            "action_rate": -0.01,
            "similar_to_default": -0.05,
            "object_height": -10.0,
            "tactile_contact": 0.5,
            "rotation_progress": 5.0,
        },
    }

    command_cfg = {
        "num_commands": 3,  # angular velocities around x, y, z axes
        "ang_vel_x_range": [-0.5, 0.5],  # rad/s
        "ang_vel_y_range": [-0.5, 0.5],  # rad/s
        "ang_vel_z_range": [-1.0, 1.0],  # rad/s
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="allegro-rotation")
    parser.add_argument("-B", "--num_envs", type=int, default=2048)
    parser.add_argument("--max_iterations", type=int, default=500)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = AllegroEnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python examples/sensors/in_hand_rotate/allegro_train.py
"""
