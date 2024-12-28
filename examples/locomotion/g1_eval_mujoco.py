import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
import os
from rsl_rl.runners import OnPolicyRunner
import pickle


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

class dummy_env:
    def __init__(self, num_envs, env_cfg, obs_cfg, device='cuda'):
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.num_privileged_obs = None
        self.num_obs = obs_cfg["num_obs"]
        self.num_actions = env_cfg["num_actions"]

    def reset(self):
        return None, None

if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="g1-walking")
    parser.add_argument("--ckpt", type=int, default=300)
    args = parser.parse_args()
    log_dir = f"logs/{args.exp_name}"

    config = {
        'policy_path': os.path.join(log_dir, f"model_{args.ckpt}.pt"),
        'xml_path': 'genesis/assets/urdf/g1/scene.xml',
        'simulation_duration': 60.0,
        'simulation_dt': 0.002,
        'control_decimation': 10,
        'kps': [100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40],
        'kds': [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2],
        'default_angles': [-0.1,  0.0,  0.0,  0.3, -0.2, 0.0, 
                           -0.1,  0.0,  0.0,  0.3, -0.2, 0.0],
        'ang_vel_scale': 0.25,
        'dof_pos_scale': 1.0,
        'dof_vel_scale': 0.05,
        'action_scale': 0.25,
        'cmd_scale': [2.0, 2.0, 0.25],
        'num_actions': 12,
        'num_obs': 47,
        'cmd_init': [0.5, 0, 0],
    }
    policy_path = config["policy_path"]
    xml_path = config["xml_path"]

    simulation_duration = config["simulation_duration"]
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]

    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)

    default_angles = np.array(config["default_angles"], dtype=np.float32)

    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale = config["action_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

    num_actions = config["num_actions"]
    num_obs = config["num_obs"]
    
    cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    env_cfg, obs_cfg, _, _, train_cfg, _ = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    env = dummy_env(1, env_cfg, obs_cfg)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    runner.load(policy_path)
    policy = runner.get_inference_policy(device="cuda:0")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9 : 9 + num_actions] = qj
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
                # obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                obs_tensor = torch.tensor(obs, device='cuda')
                # policy inference
                action = policy(obs_tensor).detach().cpu().numpy()
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
