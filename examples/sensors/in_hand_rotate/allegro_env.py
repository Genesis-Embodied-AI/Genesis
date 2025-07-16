import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, xyz_to_quat
from genesis.sensors import RigidContactForceGridSensor


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class AllegroEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.dt = 0.02  # control frequency
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
                camera_pos=(0.3, 0.3, 0.4),
                camera_lookat=(0.0, 0.0, 0.2),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        self.scene.add_entity(gs.morphs.Plane())

        # add hand
        self.hand_init_pos = torch.tensor(self.env_cfg["hand_init_pos"], device=gs.device)
        self.hand_init_euler = torch.tensor(self.env_cfg["hand_init_euler"], device=gs.device)
        self.hand = self.scene.add_entity(
            morph=gs.morphs.URDF(
                file="urdf/allegro_hand/allegro_hand_right_glb.urdf",
                pos=self.hand_init_pos.cpu().numpy(),
                euler=self.hand_init_euler.cpu().numpy(),
                fixed=True,
                merge_fixed_links=False,
            ),
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )

        # add sensors
        self.sensors = []
        for link in self.hand.links:
            if "tip" in link.name:
                sensor = self.hand.add_sensor(
                    RigidContactForceGridSensor,
                    name=f"tactile_{link.name}",
                    grid_size=(4, 4, 2),
                    link_idx=link.idx,
                )
                self.sensors.append(sensor)

        # add object to rotate
        self.obj_init_pos = torch.tensor(self.env_cfg["obj_init_pos"], device=gs.device)
        self.obj_init_euler = torch.tensor(self.env_cfg["obj_init_euler"], device=gs.device)
        self.obj_init_quat = xyz_to_quat(self.obj_init_euler, degrees=True)
        self.obj = self.scene.add_entity(
            gs.morphs.Box(
                pos=self.obj_init_pos.cpu().numpy(),
                euler=self.obj_init_euler.cpu().numpy(),
                size=self.env_cfg["obj_size"],
            ),
            surface=gs.surfaces.Default(
                color=(1.0, 0.4, 0.0, 0.5),
            ),
        )

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motors_dof_idx = [self.hand.get_joint(name).dof_start for name in self.env_cfg["joint_names"]]

        # PD control parameters
        self.hand.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.hand.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # initialize buffers
        self.obj_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.obj_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.obj_euler = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.obj_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.obj_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.initial_obj_euler = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)

        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["ang_vel"]] * self.num_commands,
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=gs.device,
            dtype=gs.tc_float,
        )

        # tactile sensor data
        self.tactile_data = torch.zeros(
            (self.num_envs, len(self.sensors) * 32), device=gs.device, dtype=gs.tc_float
        )  # 4*4*2 = 32 per sensor

        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

    def _resample_commands(self, envs_idx):
        """Resample rotation commands (desired angular velocities around x, y, z axes)"""
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["ang_vel_x_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["ang_vel_y_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_z_range"], (len(envs_idx),), gs.device)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        target_dof_pos = self.actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.hand.control_dofs_position(target_dof_pos, self.motors_dof_idx)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.obj_pos[:] = self.obj.get_pos()
        self.obj_quat[:] = self.obj.get_quat()
        self.obj_euler[:] = quat_to_xyz(self.obj_quat, rpy=True, degrees=True)
        self.obj_lin_vel[:] = self.obj.get_vel()
        self.obj_ang_vel[:] = self.obj.get_ang()
        self.dof_pos[:] = self.hand.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.hand.get_dofs_velocity(self.motors_dof_idx)

        # get tactile sensor data
        tactile_idx = 0
        for sensor in self.sensors:
            sensor_data = sensor.get_data().flatten(start_dim=1)  # flatten grid data
            sensor_size = sensor_data.shape[1]
            self.tactile_data[:, tactile_idx : tactile_idx + sensor_size] = sensor_data
            tactile_idx += sensor_size

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        # terminate if object falls below threshold
        self.reset_buf |= self.obj_pos[:, 2] < self.env_cfg["termination_height"]
        # terminate if object moves too far from hand
        obj_hand_dist = torch.norm(self.obj_pos - self.hand_init_pos, dim=1)
        self.reset_buf |= obj_hand_dist > self.env_cfg["max_obj_distance"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self.obs_buf = torch.cat(
            [
                self.obj_pos * self.obs_scales["obj_pos"],  # 3
                self.obj_euler * self.obs_scales["obj_euler"],  # 3
                self.obj_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 16
                self.dof_vel * self.obs_scales["dof_vel"],  # 16
                self.actions,  # 16
                self.tactile_data * self.obs_scales["tactile"],  # tactile sensor data
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.hand.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset object
        self.obj_pos[envs_idx] = self.obj_init_pos
        self.obj_quat[envs_idx] = self.obj_init_quat.reshape(1, -1)
        self.obj.set_pos(self.obj_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.obj.set_quat(self.obj_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.initial_obj_euler[envs_idx] = self.obj_init_euler.reshape(1, -1)

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

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_tracking_ang_vel(self):
        """Tracking of angular velocity commands"""
        ang_vel_error = torch.sum(torch.square(self.commands - self.obj_ang_vel), dim=1)
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_object_stable(self):
        """Reward for keeping object stable (low linear velocity)"""
        return torch.exp(-torch.norm(self.obj_lin_vel, dim=1) / 0.1)

    def _reward_action_rate(self):
        """Penalize changes in actions"""
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        """Penalize joint poses far away from default pose"""
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_object_height(self):
        """Penalize object falling"""
        return torch.square(self.obj_pos[:, 2] - self.reward_cfg["target_height"])

    def _reward_tactile_contact(self):
        """Reward for maintaining contact with object"""
        total_tactile = torch.sum(torch.abs(self.tactile_data), dim=1)
        return torch.exp(-torch.square(total_tactile - self.reward_cfg["target_contact"]) / 10.0)

    def _reward_rotation_progress(self):
        """Reward for achieving rotation around desired axes"""
        euler_diff = self.obj_euler - self.initial_obj_euler
        # Normalize angles to [-180, 180]
        euler_diff = torch.remainder(euler_diff + 180, 360) - 180

        # Reward rotation around commanded axes
        commanded_rotation = torch.sum(torch.abs(euler_diff * self.commands), dim=1) / 180.0  # normalize to [0,1]
        return commanded_rotation
