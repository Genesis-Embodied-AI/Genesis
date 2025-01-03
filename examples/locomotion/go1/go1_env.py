import torch
import math
import genesis as gs
from genesis.utils.terrain import parse_terrain

from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
import numpy as np
import random
def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class Go1Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="cuda"):
        self.device = torch.device(device)

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = obs_cfg["num_privileged_obs"]
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequence on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        # self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        horizontal_scale = 0.25
        vertical_scale = 0.005
        ########################## entities ##########################
        self.cols = 5
        self.rows = 5
        self.margin = 4
        n_subterrains=(self.cols+self.margin, self.rows+self.margin)
        supported_subterrain_types = [
            "flat_terrain",
            "random_uniform_terrain",
            "pyramid_sloped_terrain",
            "discrete_obstacles_terrain",
            "wave_terrain",
            "pyramid_stairs_terrain",
            # "sloped_terrain",
            # "stepping_stones_terrain",
        ]
        # probs = [
        #     0.3,
        #     0.5,
        #     0.8,
        #     0.5,
        #     0.5,
        #     0.8,
        # ]
        probs = [
            0.4,
            0.5,
            0.001,
            0.2,
            0.1,
            0.001,
        ]
        total = sum(probs)
        normalized_probs = [p / total for p in probs]
        subterrain_grid = self.generate_subterrain_grid(self.rows+self.margin, self.cols+self.margin, supported_subterrain_types, normalized_probs)



        self.terrain  = gs.morphs.Terrain(
                    n_subterrains=n_subterrains,
                    horizontal_scale=horizontal_scale,
                    vertical_scale=vertical_scale,
                    subterrain_types=subterrain_grid
                )        
        _, _, self.terrain.height_field = parse_terrain(morph=self.terrain, surface=gs.surfaces.Default())
        self.scene.add_entity(self.terrain)
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go1/urdf/go1.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
                links_to_keep = self.env_cfg["feet_names"]
            ),
        )
        self.envs_origins = torch.zeros((self.num_envs, 7), device=self.device)

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]
        self.hip_dofs = [self.robot.get_joint(str(name +"_joint")).dof_idx_local for name in self.env_cfg["hip_names"]]

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        # Store link indices that trigger termination or penalty

        # self.termination_contact_indices = env_cfg.get("termination_contact_indices", [])
        # self.penalised_contact_indices = env_cfg.get("penalised_contact_indices", [])
        # Convert link names to indices
        self.termination_contact_indices = [self.robot.get_link(name).idx_local  for name in self.env_cfg["termination_contact_names"]]
        self.penalised_contact_indices = [self.robot.get_link(name).idx_local  for name in self.env_cfg["penalised_contact_names"]]
        self.feet_indices = [self.robot.get_link(name).idx_local  for name in self.env_cfg["feet_names"]]
        for link in self.robot._links:
            print(link.name)
        self.init_foot()
        print(f"termination_contact_indicies {self.termination_contact_indices}")
        print(f"penalised_contact_indices {self.penalised_contact_indices}")
        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.privileged_obs_buf = torch.zeros((self.num_envs, self.num_privileged_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.hip_actions = torch.zeros((self.num_envs, len(self.hip_dofs)), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.hip_pos = torch.zeros_like(self.hip_actions)
        self.hip_vel = torch.zeros_like(self.hip_actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.default_hip_pos = torch.tensor(
            [
                self.env_cfg["default_joint_angles"][name]
                for name in self.env_cfg["dof_names"]
                if "hip" in name
            ],
            device=self.device,
            dtype=gs.tc_float,
        )
        print(f"Default dof pos {self.default_dof_pos}")
        print(f"Default hip pos {self.default_hip_pos}")
        self.extras = dict()  # extra information for logging

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)

    def generate_subterrain_grid(self, rows, cols, terrain_types, weights):
        """
        Generate a 2D grid (rows x cols) of terrain strings chosen randomly
        based on 'weights', but do NOT place 'pyramid_sloped_terrain' adjacent 
        to another 'pyramid_sloped_terrain'.
        """
        grid = [[None for _ in range(cols)] for _ in range(rows)]

        for i in range(rows):
            for j in range(cols):
                if i == 0 or i == 1 or i == (rows-1) or i == (rows-2) or j == 0 or j == 1 or j == (cols-1) or j == (cols-2):
                    grid[i][j] = "flat_terrain"
                    continue
                # Randomly pick a terrain type based on the given weights
                terrain_choice = random.choices(terrain_types, weights=weights, k=1)[0]
                grid[i][j] = terrain_choice
        return grid

    def init_foot(self):
        self.feet_num = len(self.feet_indices)
       
        self.step_period = self.reward_cfg["step_period"]
        self.step_offset = self.reward_cfg["step_offset"]
        self.step_height = self.reward_cfg["feet_height_target"]
        #todo get he first feet_pos here
        # Get positions for all links and slice using indices
        all_links_pos = self.robot.get_links_pos()
        all_links_vel = self.robot.get_links_vel()

        self.feet_pos = all_links_pos[:, self.feet_indices, :]
        self.feet_vel = all_links_vel[:, self.feet_indices, :]

    def update_feet_state(self):
        # Get positions for all links and slice using indices
        all_links_pos = self.robot.get_links_pos()
        all_links_vel = self.robot.get_links_vel()

        self.feet_pos = all_links_pos[:, self.feet_indices, :]
        self.feet_vel = all_links_vel[:, self.feet_indices, :]

    def post_physics_step_callback(self):
        self.update_feet_state()
        self.phase = (self.episode_length_buf * self.dt) % self.step_period / self.step_period
        # Assign phases for quadruped legs
        """
        small_offset = 0.05  # tweak as needed, 0 < small_offset < step_offset typically
        self.phase_FL_RR = self.phase
        self.phase_FR_RL = (self.phase + self.step_offset) % 1

        # Now offset one leg in each diagonal pair slightly
        phase_FL = self.phase_FL_RR
        phase_RR = (self.phase_FL_RR + small_offset) % 1     # shifted by small_offset

        phase_FR = self.phase_FR_RL
        phase_RL = (self.phase_FR_RL + small_offset) % 1     # shifted by small_offset

        # Concatenate in the order (FL, FR, RL, RR)
        self.leg_phase = torch.cat([
            phase_FL.unsqueeze(1),
            phase_FR.unsqueeze(1),
            phase_RL.unsqueeze(1),
            phase_RR.unsqueeze(1)
        ], dim=-1)
        """

        # Assign phases for quadruped legs
        self.phase_FL_RR = self.phase  # Front-left (FL) and Rear-right (RR) in sync
        self.phase_FR_RL = (self.phase + self.step_offset) % 1  # Front-right (FR) and Rear-left (RL) offset

        # Assign phases to legs based on their indices (FL, FR, RL, RR) order matters
        self.leg_phase = torch.cat([
            self.phase_FL_RR.unsqueeze(1),  # FL
            self.phase_FR_RL.unsqueeze(1),  # FR
            self.phase_FR_RL.unsqueeze(1),  # RL
            self.phase_FL_RR.unsqueeze(1)   # RR
        ], dim=-1)


    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self.hip_pos[:] = self.robot.get_dofs_position(self.hip_dofs)
        self.hip_vel[:] = self.robot.get_dofs_position(self.hip_dofs)
        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)

        self.post_physics_step_callback()
        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        # self.check_termination()
        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        sin_phase = torch.sin(2 * np.pi * self.leg_phase)  # Shape: (batch_size, 4)
        cos_phase = torch.cos(2 * np.pi * self.leg_phase)  # Shape: (batch_size, 4)

        # compute observations
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
                sin_phase, #4
                cos_phase #4
            ],
            axis=-1,
        )
        # compute observations
        self.privileged_obs_buf = torch.cat(
            [
                self.base_lin_vel * self.obs_scales["lin_vel"],  # 3
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
                sin_phase, #4
                cos_phase #4
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf


    def check_termination(self):
        """Check if environments need to be reset."""
        # (n_envs, n_links, 3) tensor of net contact forces
        contact_forces = self.robot.get_links_net_contact_force()

        # 1) Terminate if base link(s) contact force is too high
        # contact_forces[:, self.termination_contact_indices, :] => shape (n_envs, len(termination_ids), 3)
        base_contact = torch.norm(contact_forces[:, self.termination_contact_indices, :], dim=-1)
        # if any link in `termination_contact_indices` exceeds threshold, reset that env
        self.reset_buf = torch.any(base_contact > 1.0, dim=1)

        # 2) Terminate if pitch or roll exceed thresholds
        self.reset_buf |= torch.logical_or(
            torch.abs(self.base_euler[:, 1]) > 1.0,  # pitch
            torch.abs(self.base_euler[:, 0]) > 0.8   # roll
        )

        # 3) Time-out termination
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.hip_pos[envs_idx] = self.default_hip_pos
        self.hip_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        for index in envs_idx:
            x, y, z, q = self._random_robot_position()
            self.envs_origins[index, 0] = x
            self.envs_origins[index, 1] = y
            self.envs_origins[index, 2] = z 
            self.base_quat[index] = transform_quat_by_quat(q, self.base_quat[index])
        # print(envs_idx)
        # if 0 in envs_idx.tolist():
        # self.scene.viewer_options.camera_pos=(self.envs_origins[0, 0]+2.0, self.envs_origins[0, 1], self.envs_origins[0, 2] )
        # self.scene.viewer_options.camera_lookat=(self.envs_origins[0, 0], self.envs_origins[0, 1], self.envs_origins[0, 2] +0.5)
 
        self.robot.set_pos(self.base_pos[envs_idx]+self.envs_origins[envs_idx, :3], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_pos(self.envs_origins[envs_idx, :3], zero_velocity=False, envs_idx=envs_idx)
        # self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)


    def _random_robot_position(self):
        # 1. Sample random row, col(a subterrain)
        # 0.775 ~ l2_norm(0.7, 0.31)
        go2_size_xy = 0.775
        valid_row = self.rows -2
        valid_col = self.cols -2
        row = np.random.randint(int((valid_row*self.terrain.subterrain_size[0]-go2_size_xy)/self.terrain.horizontal_scale))
        col = np.random.randint(int((valid_col*self.terrain.subterrain_size[1]-go2_size_xy)/self.terrain.horizontal_scale))
        # 2. Convert (row, col) -> (x, y) in world coords
        # Each cell is horizontal_scale in size
        x = row*self.terrain.horizontal_scale + go2_size_xy/2
        y = col*self.terrain.horizontal_scale + go2_size_xy/2
        # 3. Get terrain height in meters
        z = self.terrain.height_field[row, col]*self.terrain.vertical_scale
        # z = 0.5terrain_choice

        # 4. Add a small offset so the robot spawns above the ground
        # z += 0.1  # for example

        # 5. rotation quaternion
        angle = np.random.uniform(2*np.pi)
        q = torch.tensor([np.cos(angle), 0, 0, np.sin(angle)], device=self.device)
        
        return x, y, z, q

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, self.privileged_obs_buf

    # ------------ reward functions----------------


    def _reward_collision(self):
        """
        Penalize collisions on selected bodies.
        Returns the per-env penalty value as a 1D tensor of shape (n_envs,).
        """
        # (n_envs, n_links, 3) net contact forces
        contact_forces = self.robot.get_links_net_contact_force()

        # Extract forces for the undesired-contact links
        # => shape (n_envs, len(self.penalised_contact_indices), 3)
        undesired_forces = torch.norm(contact_forces[:, self.penalised_contact_indices, :], dim=-1)
        
        # Boolean of whether each penalized link has force > threshold (e.g. > 0.1)
        collisions = (undesired_forces > 0.1).float()  # shape (n_envs, len(...))
        
        # Sum over those links to get # of collisions per environment
        return collisions.sum(dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    def _reward_collision(self):
        """
        Penalize collisions on selected bodies.
        Returns the per-env penalty value as a 1D tensor of shape (n_envs,).
        """
        # (n_envs, n_links, 3) net contact forces
        contact_forces = self.robot.get_links_net_contact_force()
        # print(contact_forces)
        # Extract forces for the undesired-contact links
        # => shape (n_envs, len(self.penalised_contact_indices), 3)
        undesired_forces = torch.norm(contact_forces[:, self.penalised_contact_indices, :], dim=-1)
        # print(undesired_forces)
        # Boolean of whether each penalized link has force > threshold (e.g. > 0.1)
        collisions = (undesired_forces > 0.1).float()  # shape (n_envs, len(...))
        
        # Sum over those links to get # of collisions per environment
        return collisions.sum(dim=1)

    def _reward_contact_no_vel(self):
        contact_forces = self.robot.get_links_net_contact_force()
        # Penalize contact with no velocity
        contact = torch.norm(contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        # print(contact)
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        # print(contact_feet_vel)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))

    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        contact_forces = self.robot.get_links_net_contact_force()
        # Iterate over legs (order: FL, FR, RL, RR)
        for i in range(self.feet_num):
            # Determine if the current phase indicates a stance phase (< 0.55)
            is_stance = self.leg_phase[:, i] < 0.55

            # Check if the foot is in contact with the ground
            contact = contact_forces[:, self.feet_indices[i], 2] > 1

            # Reward correct contact behavior (stance matches contact)
            res += ~(contact ^ is_stance)  # XOR for mismatch, negate for correct match

        return res

    def _reward_hip_vel(self):
        return torch.sum(torch.square(self.hip_vel), dim=(1))

    def _reward_hip_pos(self):
        return torch.sum(torch.abs(self.hip_pos- self.default_hip_pos), dim=(1))


    def _reward_feet_swing_height(self):
        contact_forces = self.robot.get_links_net_contact_force()
        contact = torch.norm(contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - self.step_height) * ~contact
        return torch.sum(pos_error, dim=(1))

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)