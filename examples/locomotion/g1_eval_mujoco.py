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
    def __init__(self, num_envs, env_cfg, obs_cfg, mjc_cfg, device='cuda'):
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.num_privileged_obs = None
        self.num_obs = obs_cfg["num_obs"]
        self.num_actions = env_cfg["num_actions"]
        self.mjc_cfg = mjc_cfg

        self.xml_path = self.mjc_cfg["xml_path"]

        self.simulation_duration = self.mjc_cfg["simulation_duration"]
        self.simulation_dt = self.mjc_cfg["simulation_dt"]
        self.control_decimation = self.mjc_cfg["control_decimation"]

        self.kps = np.array(self.mjc_cfg["kps"], dtype=np.float32)
        self.kds = np.array(self.mjc_cfg["kds"], dtype=np.float32)

        self.default_angles = np.array(self.mjc_cfg["default_angles"], dtype=np.float32)

        self.ang_vel_scale = self.mjc_cfg["ang_vel_scale"]
        self.dof_pos_scale = self.mjc_cfg["dof_pos_scale"]
        self.dof_vel_scale = self.mjc_cfg["dof_vel_scale"]
        self.action_scale = self.mjc_cfg["action_scale"]
        self.cmd_scale = np.array(self.mjc_cfg["cmd_scale"], dtype=np.float32)

        self.num_actions = self.mjc_cfg["num_actions"]
        self.num_obs = self.mjc_cfg["num_obs"]
        
        self.cmd = np.array(self.mjc_cfg["cmd_init"], dtype=np.float32)

        # define context variables
        self.actions = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.obs = np.zeros(self.num_obs, dtype=np.float32)

        self.counter = 0

        # Load robot model
        self.m = mujoco.MjModel.from_xml_path(self.xml_path)
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = self.simulation_dt
    
    def step(self, actions):
        self.actions = actions
        self.target_dof_pos = self.actions * self.action_scale + self.default_angles
        tau = pd_control(self.target_dof_pos, self.d.qpos[7:], self.kps, np.zeros_like(self.kds), self.d.qvel[6:], self.kds)
        self.d.ctrl[:] = tau
        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mujoco.mj_step(self.m, self.d)

        self.counter += 1
        if self.counter % self.control_decimation == 0:
            # Apply control signal here.

            # create observation
            qj = self.d.qpos[7:]
            dqj = self.d.qvel[6:]
            quat = self.d.qpos[3:7]
            omega = self.d.qvel[3:6]

            qj = (qj - self.default_angles) * self.dof_pos_scale
            dqj = dqj * self.dof_vel_scale
            gravity_orientation = get_gravity_orientation(quat)
            omega = omega * self.ang_vel_scale

            period = 0.8
            count = self.counter * self.simulation_dt
            phase = count % period / period
            sin_phase = np.sin(2 * np.pi * phase)
            cos_phase = np.cos(2 * np.pi * phase)

            self.obs[:3] = omega
            self.obs[3:6] = gravity_orientation
            self.obs[6:9] = self.cmd * self.cmd_scale
            self.obs[9 : 9 + self.num_actions] = qj
            self.obs[9 + self.num_actions : 9 + 2 * self.num_actions] = dqj
            self.obs[9 + 2 * self.num_actions : 9 + 3 * self.num_actions] = self.actions
            self.obs[9 + 3 * self.num_actions : 9 + 3 * self.num_actions + 2] = np.array([sin_phase, cos_phase])
        return self.obs, None, None, None, None


    def reset(self):
        return self.obs, None

if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="g1-walking")
    parser.add_argument("--ckpt", type=int, default=500)
    args = parser.parse_args()
    log_dir = f"logs/{args.exp_name}"

    mjc_cfg = {
        'policy_path': os.path.join(log_dir, f"model_{args.ckpt}.pt"),
        'xml_path': 'genesis/assets/urdf/g1/scene.xml',
        'simulation_duration': 20.0,
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

    # load policy
    env_cfg, obs_cfg, _, _, train_cfg, _ = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    env = dummy_env(1, env_cfg, obs_cfg, mjc_cfg)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = mjc_cfg["policy_path"]
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = env.reset()
    obs = torch.from_numpy(obs).to(env.device)
    with mujoco.viewer.launch_passive(env.m, env.d) as viewer:
        with torch.no_grad():
            # Close the viewer automatically after simulation_duration wall-seconds.
            start = time.time()
            while viewer.is_running() and time.time() - start < env.simulation_duration:
                step_start = time.time()
                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                actions = policy(obs).cpu().numpy()
                obs, _, _, _, _ = env.step(actions)
                obs = torch.from_numpy(obs).to(env.device)

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = env.m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
