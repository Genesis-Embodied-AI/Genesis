import argparse

import numpy as np

import genesis as gs

import torch

from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

class Controller(torch.nn.Module):
    def __init__(self, obs_dim, n_dofs, hidden_dim=64):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_dofs = n_dofs
        
        # Batch normalization layer
        self.bn = torch.nn.BatchNorm1d(obs_dim)
        
        # MLP layers (2-3 layers)
        self.fc1 = torch.nn.Linear(obs_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        
        # Output layers for mean and log_std
        self.mean_layer = torch.nn.Linear(hidden_dim, n_dofs)
        self.log_std_layer = torch.nn.Linear(hidden_dim, n_dofs)
        
        # Initialize log_std to small values
        self.log_std_layer.weight.data.fill_(0.0)
        self.log_std_layer.bias.data.fill_(-0.5)
    
    def forward(self, obs):
        """
        Args:
            obs: observation tensor of shape (batch_size, obs_dim)
        Returns:
            mean: mean of action distribution, shape (batch_size, n_dofs)
            std: standard deviation of action distribution, shape (batch_size, n_dofs)
        """
        # Batch normalization
        if obs.shape[0] > 1:
            x = self.bn(obs)
        else:
            x = obs
        
        # MLP layers with ReLU activation
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        
        # Output mean and log_std
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        
        # Clamp log_std to prevent extreme values
        log_std = torch.clamp(log_std, min=-10, max=2)
        std = torch.exp(log_std)
        
        return mean, std
    
    def sample_action(self, obs):
        """
        Sample action from the policy distribution.
        
        Args:
            obs: observation tensor of shape (batch_size, obs_dim)
        Returns:
            action: sampled action, shape (batch_size, n_dofs)
            mean: mean of action distribution, shape (batch_size, n_dofs)
            std: standard deviation of action distribution, shape (batch_size, n_dofs)
        """
        mean, std = self.forward(obs)
        noise = torch.randn_like(mean)
        action = mean + std * noise
        return action, mean, std

def observe_fn(state):
    qpos = state.qpos
    dofs_vel = state.dofs_vel
    dofs_acc = state.dofs_acc

    return torch.cat([qpos, dofs_vel, dofs_acc], dim=1).detach()

def reward_fn(state, dt, prev_state=None):
    pos = state.pos
    
    height_clip = torch.clamp(pos[:, 2] - 0.8, -float('inf'), 1.0)
    height_reward = torch.where(height_clip <= 0.0, -200 * (height_clip ** 2), height_clip)
    forward_reward = (pos[:, 0] - prev_state.pos[:, 0].detach()) / dt if prev_state is not None else 0.0
    
    reward = height_reward * 0.01 + forward_reward
    
    return reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-n", "--n_envs", type=int, default=49)
    args = parser.parse_args()

    args.vis = False
    args.n_envs = 64

    dt = 0.01
    substeps = 1
    horizon_steps = 128
    window_substeps = 32
    window_steps = int(window_substeps / substeps)
    iteration = 10000
    lr = 1e-4
    render_every = 100

    ########################## init ##########################
    gs.init(backend=gs.gpu, logging_level="warn")

    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=30,
        max_FPS=60,
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=substeps, requires_grad=True, grad_window_steps=window_steps),
        viewer_options=viewer_options,
        rigid_options=gs.options.RigidOptions(
            use_gjk_collision=True,
            enable_joint_limit=False,
        ),
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=(0,),
            show_world_frame=True,
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    # plane = scene.add_entity(
    #     gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True, pos=(0, 0, -0.5)),
    # )
    plane = scene.add_entity(
        gs.morphs.Box(size=(10, 10, 0.6), pos=(0, 0, -0.3), fixed=True),
    )
    ant = scene.add_entity(
        gs.morphs.MJCF(file="xml/walker_no_ground.xml"),
        # vis_mode="collision"
    )
    cam = scene.add_camera(
        pos=(3.5, 0.5, 2.5),
        lookat=(0.0, 0.0, 0.5),
        fov=40,
        GUI=False,
        env_idx=0,
    )
    cam.follow_entity(ant)

    ########################## build ##########################
    scene.build(n_envs=args.n_envs, env_spacing=(1, 1))

    n_dofs = ant.n_dofs

    # rand_force = torch.randn((n_dofs,), dtype=torch.float32)
    # for i in range(10000):
    #     ant.control_dofs_force(rand_force)
    #     scene.step()

    #     if i % 1000 == 0 and i > 0:
    #         print("----------------------------------------------------------")
    #         rand_force = torch.randn((n_dofs,), dtype=torch.float32)

    #         # Reset env
    #         rigid_solver = scene.sim.rigid_solver
    #         qpos = rigid_solver._rigid_global_info.qpos.to_numpy()
    #         dofs_vel = rigid_solver.dofs_state.vel.to_numpy()
    #         dofs_acc = rigid_solver.dofs_state.acc.to_numpy()
    #         dofs_acc_smooth = rigid_solver.dofs_state.acc_smooth.to_numpy()
    #         solver_qacc_ws = rigid_solver.constraint_solver.constraint_state.qacc_ws.to_numpy()

    #         scene.reset()
    #         scene.sim.rigid_solver._rigid_global_info.qpos.from_numpy(qpos)
    #         rigid_solver.dofs_state.vel.from_numpy(dofs_vel)
    #         rigid_solver.dofs_state.acc.from_numpy(dofs_acc)
    #         rigid_solver.dofs_state.acc_smooth.from_numpy(dofs_acc_smooth)
    #         rigid_solver.constraint_solver.constraint_state.qacc_ws.from_numpy(solver_qacc_ws)
    #         rigid_solver.load_test()

    
    # Initialize controller
    # Get obs_dim by computing observation once
    scene.reset()
    state = ant.get_state()
    obs = observe_fn(state)
    obs_dim = obs.shape[1]
    
    controller = Controller(obs_dim=obs_dim, n_dofs=n_dofs, hidden_dim=64)
    optimizer = torch.optim.Adam(controller.parameters(), lr=lr)

    rewards = []
    for iter in range(iteration):
        scene.reset()
        acc_reward = None
        prev_state = None
        
        record = (iter % render_every == 0) or (iter == iteration - 1)
        if record: 
            cam.start_recording()
        print("running forward pass...")
        for step in range(horizon_steps):
            scene.step()
            if record: 
                cam.render()

            # Determine observation and reward
            state = ant.get_state()
            obs = observe_fn(state)
            reward = reward_fn(state, dt, prev_state)
            if acc_reward is None:
                acc_reward = reward
            else:
                acc_reward += reward
            
            prev_state = state

            # Determine action
            action, mean, std = controller.sample_action(obs)
            # Apply action (assuming action is force/torque for dofs)
            ant.control_dofs_force(action)

            truncate = step == horizon_steps - 1 # step % window_steps == 0 or step == horizon_steps - 1
            if truncate and step > 0:
                print("running backward pass...")
                acc_reward = acc_reward / (step + 1)
                mean_reward = acc_reward.mean()
                loss = -mean_reward

                optimizer.zero_grad()
                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
                optimizer.step()

                print(f"[ITER {iter}] Mean Reward: {mean_reward.item():.4g} | Grad Norm: {grad_norm:.4g}")

                rewards.append(mean_reward.detach().item())

                plt.plot(rewards)
                plt.savefig("rewards.png")
                plt.close()

                # Reset env
                # scene.sim.reset_grad()
                # scene._forward_ready = True
                # scene._backward_ready = True
                # scene._t = t
                # scene.sim._cur_substep_global = substep_global
                
                # rigid_solver._rigid_global_info.qpos.from_numpy(_qpos)
                # rigid_solver.dofs_state.vel.from_numpy(_dofs_vel)
                # rigid_solver.dofs_state.acc.from_numpy(_dofs_acc)
                # rigid_solver.dofs_state.acc_smooth.from_numpy(_dofs_acc_smooth)
                # rigid_solver.constraint_solver.constraint_state.qacc_ws.from_numpy(_solver_qacc_ws)
                # rigid_solver.load_test()

                # rigid_solver = scene.sim.rigid_solver
                # qpos = rigid_solver._rigid_global_info.qpos.to_numpy()
                # dofs_vel = rigid_solver.dofs_state.vel.to_numpy()
                # dofs_acc = rigid_solver.dofs_state.acc.to_numpy()
                # dofs_acc_smooth = rigid_solver.dofs_state.acc_smooth.to_numpy()
                # solver_qacc_ws = rigid_solver.constraint_solver.constraint_state.qacc_ws.to_numpy()
                # scene_t = scene._t
                # scene.reset()
                # scene._t = scene_t
                # rigid_solver._rigid_global_info.qpos.from_numpy(qpos)
                # rigid_solver.dofs_state.vel.from_numpy(dofs_vel)
                # rigid_solver.dofs_state.acc.from_numpy(dofs_acc)
                # rigid_solver.dofs_state.acc_smooth.from_numpy(dofs_acc_smooth)
                # rigid_solver.constraint_solver.constraint_state.qacc_ws.from_numpy(solver_qacc_ws)
                # rigid_solver.load_test()

                acc_reward = None

                # if step // window_steps > 1:
                break
        
        if record: 
            cam.stop_recording(save_to_filename=f"ant_video_{iter:06d}.mp4", fps=30)
        

if __name__ == "__main__":
    main()
