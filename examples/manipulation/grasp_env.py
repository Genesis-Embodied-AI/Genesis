import torch
import numpy as np
import math
from typing import Literal
import genesis as gs
from genesis.utils.geom import xyz_to_quat, transform_quat_by_quat, transform_by_quat

import time


class GraspEnv:
    def __init__(
        self,
        num_envs,
        env_cfg,
        reward_cfg,
        robot_cfg,
        show_viewer=False,
    ):
        self.num_envs = num_envs
        self.num_obs = env_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.device = gs.device

        self.ctrl_dt = 0.01  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.ctrl_dt)

        # configs
        self.env_cfg = env_cfg
        self.reward_cfg = reward_cfg

        self.action_scales = torch.tensor(env_cfg["action_scales"], device=self.device)
        self.reward_scales = reward_cfg["reward_scales"]

        # == setup scene ==
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.ctrl_dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.ctrl_dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.ctrl_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            show_viewer=show_viewer,
        )

        # == add ground ==
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # == add robot ==
        self.robot = Manipulator(
            num_envs=self.num_envs,
            scene=self.scene,
            args=robot_cfg,
            device=gs.device,
        )

        # == add object ==
        self.object = self.scene.add_entity(
            gs.morphs.Box(size=(0.05, 0.05, 0.05), collision=False),
            material=gs.materials.Rigid(gravity_compensation=1),
        )

        # build
        self.scene.build(n_envs=num_envs, env_spacing=(1.0, 1.0))

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.ctrl_dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        self.keypoints_offset = self.get_keypoint_offsets(batch_size=self.num_envs, device=self.device, unit_length=0.5)
        # == init buffers ==
        self._init_buffers()

    def _init_buffers(self):
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=gs.device)
        #
        self.goal_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.action_buf = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        #
        self.hand_z_offset = 0.0

        #
        self.extras = dict()
        self.extras["observations"] = dict()

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        self.episode_length_buf[envs_idx] = 0

        # reset robot
        self.robot.reset(envs_idx)
        # reset object
        num_reset = len(envs_idx)
        random_x = torch.rand(num_reset, device=self.device) * 0.4 + 0.2
        random_y = (torch.rand(num_reset, device=self.device) - 0.5) * 0.0  # 0.0 ~ 0.0
        random_z = torch.ones(num_reset, device=self.device) * 0.15
        self.goal_pos[envs_idx] = torch.stack([random_x, random_y, random_z], dim=-1)

        #
        self.object.set_pos(torch.stack([random_x, random_y, random_z], dim=-1), envs_idx=envs_idx)
        self.object.set_quat(
            torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).repeat(num_reset, 1),
            envs_idx=envs_idx,
        )

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))

        obs, self.extras = self.get_observations()
        return obs, None

    def step(self, actions):
        # update time
        self.episode_length_buf += 1

        # apply action
        actions = self.rescale_action(actions)
        self.action_buf = actions
        self.robot.apply_action(actions)
        self.scene.step()

        # check termination
        env_reset_idx = self.is_episode_complete()
        if len(env_reset_idx) > 0:
            self.reset_idx(env_reset_idx)

        # compute reward
        reward = torch.zeros((self.num_envs,), device=gs.device, dtype=torch.float32)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            reward += rew
            self.episode_sums[name] += rew

        # get observations and fill extras
        obs, self.extras = self.get_observations()

        return obs, reward, self.reset_buf, self.extras

    def get_privileged_observations(self):
        return None

    def is_episode_complete(self):
        time_out_buf = self.episode_length_buf > self.max_episode_length

        # check if the ee is in the valid position
        self.reset_buf = time_out_buf

        # fill time out buffer for reward/value bootstrapping
        time_out_idx = (time_out_buf).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0
        return self.reset_buf.nonzero(as_tuple=True)[0]

    def get_observations(self):
        # Current end-effector pose
        ee_pos, ee_quat = self.robot.ee_pose[:, :3], self.robot.ee_pose[:, 3:7]
        # obj_pos, obj_quat = self.object.get_pos(), self.object.get_quat()
        obj_pos = self.goal_pos
        #
        pos_diff = ee_pos - obj_pos
        obs_components = [
            pos_diff,  # 3D position difference
            ee_quat,  # current orientation (4D quaternion)
            obj_pos,  # goal pose (7D: pos + quat)
        ]
        obs_tensor = torch.cat(obs_components, dim=-1)
        self.extras["observations"]["critic"] = obs_tensor
        return obs_tensor, self.extras

    def rescale_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.action_scales

    # ------------ reward functions----------------
    def _reward_keypoints(self):
        keypoints_offset = self.keypoints_offset

        #
        hand_pos_keypoints = self._to_world_frame(
            self.robot.ee_pose[:, :3], self.robot.ee_pose[:, 3:7], keypoints_offset
        )

        #
        facing_down_quaternion = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        goal_pos = self.goal_pos
        goal_pos[:, 2] += self.hand_z_offset
        object_pos_keypoints = self._to_world_frame(goal_pos, facing_down_quaternion, keypoints_offset)
        #
        dist = torch.norm(hand_pos_keypoints - object_pos_keypoints, p=2, dim=-1).sum(-1)
        return torch.exp(-dist)

    def _reward_action(self):
        return -torch.norm(self.action_buf, dim=-1).sum(-1)

    def _reward_downward_facing(self):
        """
        Reward for ensuring the hand is close to the object AND facing downward.
        Ignores yaw - only cares about position and downward orientation.
        """
        # Get current hand pose and object position
        hand_pos = self.robot.ee_pose[:, :3]  # [num_envs, 3]
        hand_quat = self.robot.ee_pose[:, 3:7]  # [num_envs, 4]
        obj_pos = self.goal_pos

        # Position reward: hand should be close to the object with z-offset
        target_hand_pos = obj_pos.clone()
        target_hand_pos[:, 2] += self.hand_z_offset  # Add z-offset to keep hand above object
        pos_error = torch.norm(hand_pos - target_hand_pos, dim=-1)
        pos_reward = torch.exp(-pos_error)  # Exponential decay with distance

        # Orientation reward: hand should be facing downward
        # Extract z-axis from hand quaternion
        # For quaternion [w, x, y, z], the z-axis of the rotated frame is:
        # [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        w, x, y, z = hand_quat[:, 0], hand_quat[:, 1], hand_quat[:, 2], hand_quat[:, 3]
        hand_z_axis = torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1)

        # Desired z-axis: pointing downward [0, 0, -1]
        target_z_axis = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)

        # Calculate how well the z-axis aligns with downward direction
        # Dot product gives us alignment (1 = perfectly aligned, -1 = opposite)
        alignment = torch.sum(hand_z_axis * target_z_axis, dim=-1)

        # Convert to orientation reward (1 = perfect downward facing, 0 = horizontal, -1 = facing up)
        orientation_reward = (alignment + 1.0) / 2.0  # Maps [-1, 1] to [0, 1]

        # Combine position and orientation rewards
        # You can adjust these weights based on your preference
        pos_weight = 1.0
        orientation_weight = 0.5

        total_reward = pos_weight * pos_reward + orientation_weight * orientation_reward

        return total_reward

    def _to_world_frame(
        self,
        position: torch.Tensor,  # [B, 3]
        quaternion: torch.Tensor,  # [B, 4]
        keypoints_offset: torch.Tensor,  # [B, 7, 3]
    ) -> torch.Tensor:
        world = torch.zeros_like(keypoints_offset)
        for k in range(keypoints_offset.shape[1]):
            world[:, k] = position + transform_by_quat(keypoints_offset[:, k], quaternion)
        return world

    @staticmethod
    def get_keypoint_offsets(batch_size, device, unit_length=0.5):
        """
        Get uniformly-spaced keypoints along a line of unit length, centered at body center.
        """
        keypoint_offsets = (
            torch.tensor(
                [
                    [0, 0, 0],  # origin
                    [-1.0, 0, 0],  # x-negative
                    [1.0, 0, 0],  # x-positive
                    [0, -1.0, 0],  # y-negative
                    [0, 1.0, 0],  # y-positive
                    [0, 0, -1.0],  # z-negative
                    [0, 0, 1.0],  # z-positive
                ],
                device=device,
                dtype=torch.float32,
            )
            * unit_length
        )
        return keypoint_offsets.unsqueeze(0).repeat(batch_size, 1, 1)


## ------------ robot ----------------
class Manipulator:
    def __init__(self, num_envs: int, scene: gs.Scene, args: dict, device: str = "cpu"):
        # == set members ==
        self._device = device
        self._scene = scene
        self._num_envs = num_envs
        self._args = args

        # == Genesis configurations ==
        material: gs.materials.Rigid = gs.materials.Rigid()
        morph: gs.morphs.URDF = gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=(0.0, 0.0, 0.0),
            quat=(1.0, 0.0, 0.0, 0.0),
        )
        self._robot_entity: gs.Entity = scene.add_entity(material=material, morph=morph)

        self._ctrl_mode: Literal["pose", "rel_pose", "dls"] = "dls"

        # == some buffer initialization ==
        self._init()

    def _init(self):
        self._arm_dof_dim = self._robot_entity.n_dofs - 1  # total number of arm: joints
        self._gripper_dim = 1  # number of gripper joints

        #
        self._arm_dof_idx = torch.arange(self._arm_dof_dim, device=self._device)
        self._fingers_dof = torch.arange(
            self._arm_dof_dim,
            self._arm_dof_dim + self._gripper_dim,
            device=self._device,
        )
        #
        self._ee_link = self._robot_entity.get_link(self._args["ee_link_name"])
        self._left_finger_link = self._robot_entity.get_link(self._args["gripper_link_names"][0])
        self._right_finger_link = self._robot_entity.get_link(self._args["gripper_link_names"][1])
        #
        self._default_joint_angles = self._args["default_arm_dof"]
        if self._args["default_gripper_dof"] is not None:
            self._default_joint_angles += self._args["default_gripper_dof"]

    def reset(self, envs_idx: torch.IntTensor):
        if len(envs_idx) == 0:
            return
        self.reset_home(envs_idx)

    def reset_home(self, envs_idx: torch.IntTensor | None = None):
        if envs_idx is None:
            envs_idx = torch.arange(self._num_envs, device=self._device)
        default_joint_angles = torch.tensor(
            self._default_joint_angles, dtype=torch.float32, device=self._device
        ).repeat(len(envs_idx), 1)
        self._robot_entity.set_qpos(default_joint_angles, envs_idx=envs_idx)

    def apply_action(self, action: torch.Tensor) -> None:
        """
        Apply the action to the robot.
        """
        if self._ctrl_mode == "pose":
            assert action.shape == (self._num_envs, 7)
            self._apply_pose_action(action)
        elif self._ctrl_mode == "rel_pose":
            assert action.shape == (self._num_envs, 6)
            self._apply_rel_pose_action(action)
        elif self._ctrl_mode == "dls":
            assert action.shape == (self._num_envs, 6)
            self._apply_dls_ctrl(action)
        else:
            raise ValueError(f"Invalid control mode: {self._ctrl_mode}")

    def _apply_pose_action(self, action: torch.Tensor) -> None:
        """
        action: (num_envs, 7): [x, y, z, qw, qx, qy, qz]
        """
        target_position = action[:, :3]
        target_orientation = action[:, 3:7]
        #
        q_pos = self._robot_entity.inverse_kinematics(
            link=self._ee_link,
            pos=target_position,
            quat=target_orientation,
            dofs_idx_local=self._arm_dof_idx,
            max_samples=10,
            max_solver_iters=20,
        )

        # set gripper to open
        q_pos[:, self._fingers_dof] = torch.tensor([0.04], device=self._device)
        self._robot_entity.control_dofs_position(position=q_pos)

    def _apply_rel_pose_action(self, action: torch.Tensor) -> None:
        """
        action: (num_envs, 6): [dx, dy, dz, delta_roll, delta_pitch, delta_yaw]
        """
        delta_position = action[:, :3]
        delta_orientation = action[:, 3:6]

        # compute target pose
        target_position = delta_position + self._ee_link.get_pos()
        quat_rel = xyz_to_quat(delta_orientation, rpy=True, degrees=False)
        target_orientation = transform_quat_by_quat(quat_rel, self._ee_link.get_quat())

        #
        self._apply_pose_action(torch.cat([target_position, target_orientation], dim=-1))

    def _apply_dls_ctrl(self, delta_pose: torch.Tensor) -> None:
        #
        lambda_val = 0.01
        jacobian = self._robot_entity.get_jacobian(link=self._ee_link)
        jacobian_T = jacobian.transpose(1, 2)
        lambda_matrix = (lambda_val**2) * torch.eye(n=jacobian.shape[1], device=self._device)
        delta_joint_pos = (
            jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)
        ).squeeze(-1)

        q_pos = self._robot_entity.get_qpos() + delta_joint_pos
        q_pos[:, self._fingers_dof] = torch.tensor([0.04], device=self._device)
        self._robot_entity.control_dofs_position(position=q_pos)

    @property
    def base_pos(self):
        return self._robot_entity.get_pos()

    @property
    def ee_pose(self) -> torch.Tensor:
        pos, quat = self._ee_link.get_pos(), self._ee_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

    @property
    def left_finger_pose(self) -> torch.Tensor:
        pos, quat = self._left_finger_link.get_pos(), self._left_finger_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

    @property
    def right_finger_pose(self) -> torch.Tensor:
        pos, quat = (
            self._right_finger_link.get_pos(),
            self._right_finger_link.get_quat(),
        )
        return torch.cat([pos, quat], dim=-1)
