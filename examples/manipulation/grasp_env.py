import torch
import math
from typing import Literal

import genesis as gs
from genesis.utils.geom import (
    xyz_to_quat,
    transform_quat_by_quat,
    transform_by_quat,
)


class GraspEnv:
    def __init__(
        self,
        env_cfg: dict,
        reward_cfg: dict,
        robot_cfg: dict,
        show_viewer: bool = False,
    ) -> None:
        self.num_envs = env_cfg["num_envs"]
        self.num_obs = env_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.image_width = env_cfg["image_resolution"][0]
        self.image_height = env_cfg["image_resolution"][1]
        self.rgb_image_shape = (3, self.image_height, self.image_width)
        self.device = gs.device

        self.ctrl_dt = env_cfg["ctrl_dt"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.ctrl_dt)

        # configs
        self.env_cfg = env_cfg
        self.reward_scales = reward_cfg
        self.action_scales = torch.tensor(env_cfg["action_scales"], device=self.device)

        # == setup scene ==
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.ctrl_dt, substeps=2),
            rigid_options=gs.options.RigidOptions(
                dt=self.ctrl_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(10))),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.ctrl_dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            renderer=gs.options.renderers.BatchRenderer(
                use_rasterizer=env_cfg["use_rasterizer"],
            ),
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
            gs.morphs.Box(
                size=env_cfg["box_size"],
                fixed=env_cfg["box_fixed"],
                collision=env_cfg["box_collision"],
            ),
            # material=gs.materials.Rigid(gravity_compensation=1),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(1.0, 0.0, 0.0),
                ),
            ),
        )
        if self.env_cfg["visualize_camera"]:
            self.vis_cam = self.scene.add_camera(
                res=(1280, 720),
                pos=(1.5, 0.0, 0.2),
                lookat=(0.0, 0.0, 0.2),
                fov=60,
                GUI=self.env_cfg["visualize_camera"],
                debug=True,
            )

        # == add stero camera ==
        self.left_cam = self.scene.add_camera(
            res=(self.image_width, self.image_height),
            pos=(1.25, 0.3, 0.3),
            lookat=(0.0, 0.0, 0.0),
            fov=60,
            GUI=self.env_cfg["visualize_camera"],
        )
        self.right_cam = self.scene.add_camera(
            res=(self.image_width, self.image_height),
            pos=(1.25, -0.3, 0.3),
            lookat=(0.0, 0.0, 0.0),
            fov=60,
            GUI=self.env_cfg["visualize_camera"],
        )

        # build
        self.scene.build(n_envs=env_cfg["num_envs"])
        # set pd gains (must be called after scene.build)
        self.robot.set_pd_gains()

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.ctrl_dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        self.keypoints_offset = self.get_keypoint_offsets(batch_size=self.num_envs, device=self.device, unit_length=0.5)
        # == init buffers ==
        self._init_buffers()
        self.reset()

    def _init_buffers(self) -> None:
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=gs.device)
        self.goal_pose = torch.zeros(self.num_envs, 7, device=gs.device)
        self.extras = dict()
        self.extras["observations"] = dict()

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        if len(envs_idx) == 0:
            return
        self.episode_length_buf[envs_idx] = 0

        # reset robot
        self.robot.reset(envs_idx)

        # reset object
        num_reset = len(envs_idx)
        random_x = torch.rand(num_reset, device=self.device) * 0.4 + 0.2  # 0.2 ~ 0.6
        random_y = (torch.rand(num_reset, device=self.device) - 0.5) * 0.5  # -0.25 ~ 0.25
        random_z = torch.ones(num_reset, device=self.device) * 0.025  # 0.15 ~ 0.15
        random_pos = torch.stack([random_x, random_y, random_z], dim=-1)

        # downward facing quaternion to align with the hand
        q_downward = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).repeat(num_reset, 1)
        # randomly yaw the object
        random_yaw = (torch.rand(num_reset, device=self.device) * 2 * math.pi - math.pi) * 0.25
        q_yaw = torch.stack(
            [
                torch.cos(random_yaw / 2),
                torch.zeros(num_reset, device=self.device),
                torch.zeros(num_reset, device=self.device),
                torch.sin(random_yaw / 2),
            ],
            dim=-1,
        )
        goal_yaw = transform_quat_by_quat(q_yaw, q_downward)

        self.goal_pose[envs_idx] = torch.cat([random_pos, goal_yaw], dim=-1)
        self.object.set_pos(random_pos, envs_idx=envs_idx)
        self.object.set_quat(goal_yaw, envs_idx=envs_idx)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def reset(self) -> tuple[torch.Tensor, dict]:
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))

        obs, self.extras = self.get_observations()
        return obs, self.extras

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # update time
        self.episode_length_buf += 1

        # apply action based on task
        actions = self.rescale_action(actions)

        self.robot.apply_action(actions, open_gripper=True)
        self.scene.step()

        # check termination
        env_reset_idx = self.is_episode_complete()
        if len(env_reset_idx) > 0:
            self.reset_idx(env_reset_idx)

        # compute reward based on task
        reward = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            reward += rew
            self.episode_sums[name] += rew

        # get observations and fill extras
        obs, self.extras = self.get_observations()

        return obs, reward, self.reset_buf, self.extras

    def get_privileged_observations(self) -> None:
        return None

    def is_episode_complete(self) -> torch.Tensor:
        time_out_buf = self.episode_length_buf > self.max_episode_length

        # check if the ee is in the valid position
        self.reset_buf = time_out_buf

        # fill time out buffer for reward/value bootstrapping
        time_out_idx = (time_out_buf).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0
        return self.reset_buf.nonzero(as_tuple=True)[0]

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        # Current end-effector pose
        finger_pos, finger_quat = (
            self.robot.center_finger_pose[:, :3],
            self.robot.center_finger_pose[:, 3:7],
        )
        obj_pos, obj_quat = self.object.get_pos(), self.object.get_quat()
        #
        obs_components = [
            finger_pos - obj_pos,  # 3D position difference
            finger_quat,  # current orientation (w, x, y, z)
            obj_pos,  # goal position
            obj_quat,  # goal orientation (w, x, y, z)
        ]
        obs_tensor = torch.cat(obs_components, dim=-1)
        self.extras["observations"]["critic"] = obs_tensor
        return obs_tensor, self.extras

    def rescale_action(self, action: torch.Tensor) -> torch.Tensor:
        rescaled_action = action * self.action_scales
        return rescaled_action

    def get_stereo_rgb_images(self, normalize: bool = True) -> torch.Tensor:
        rgb_left, _, _, _ = self.left_cam.render(rgb=True, depth=False, segmentation=False, normal=False)
        rgb_right, _, _, _ = self.right_cam.render(rgb=True, depth=False, segmentation=False, normal=False)

        # Convert to proper format
        rgb_left = rgb_left.permute(0, 3, 1, 2)[:, :3]  # shape (B, 3, H, W)
        rgb_right = rgb_right.permute(0, 3, 1, 2)[:, :3]  # shape (B, 3, H, W)

        # Normalize if requested
        if normalize:
            rgb_left = torch.clamp(rgb_left, min=0.0, max=255.0) / 255.0
            rgb_right = torch.clamp(rgb_right, min=0.0, max=255.0) / 255.0

        # Concatenate left and right rgb images along channel dimension
        # Result: [B, 6, H, W] where channel 0 is left rgb, channel 1 is right rgb
        stereo_rgb = torch.cat([rgb_left, rgb_right], dim=1)
        return stereo_rgb

    # ------------ begin reward functions----------------
    def _reward_keypoints(self) -> torch.Tensor:
        keypoints_offset = self.keypoints_offset
        # there is a offset between the finger tip and the finger base frame
        finger_tip_z_offset = torch.tensor(
            [0.0, 0.0, -0.06],
            device=self.device,
            dtype=gs.tc_float,
        ).repeat(self.num_envs, 1)
        finger_pos_keypoints = self._to_world_frame(
            self.robot.center_finger_pose[:, :3] + finger_tip_z_offset,
            self.robot.center_finger_pose[:, 3:7],
            keypoints_offset,
        )
        object_pos_keypoints = self._to_world_frame(self.object.get_pos(), self.object.get_quat(), keypoints_offset)
        dist = torch.norm(finger_pos_keypoints - object_pos_keypoints, p=2, dim=-1).sum(-1)
        return torch.exp(-dist)

    # ------------ end reward functions----------------

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
    def get_keypoint_offsets(batch_size: int, device: str, unit_length: float = 0.5) -> torch.Tensor:
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

    def grasp_and_lift_demo(self) -> None:
        total_steps = 500
        goal_pose = self.robot.ee_pose.clone()
        # lift pose (above the object)
        lift_height = 0.3
        lift_pose = goal_pose.clone()
        lift_pose[:, 2] += lift_height
        # final pose (above the table)
        final_pose = goal_pose.clone()
        final_pose[:, 0] = 0.3
        final_pose[:, 1] = 0.0
        final_pose[:, 2] = 0.4
        # reset pose (home pose)
        reset_pose = torch.tensor([0.2, 0.0, 0.4, 0.0, 1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        for i in range(total_steps):
            if i < total_steps / 4:  # grasping
                self.robot.go_to_goal(goal_pose, open_gripper=False)
            elif i < total_steps / 2:  # lifting
                self.robot.go_to_goal(lift_pose, open_gripper=False)
            elif i < total_steps * 3 / 4:  # final
                self.robot.go_to_goal(final_pose, open_gripper=False)
            else:  # reset
                self.robot.go_to_goal(reset_pose, open_gripper=True)
            self.scene.step()


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

        self._gripper_open_dof = 0.04
        self._gripper_close_dof = 0.00

        self._ik_method: Literal["rel_pose", "dls"] = args["ik_method"]

        # == some buffer initialization ==
        self._init()

    def set_pd_gains(self):
        # set control gains
        # Note: the following values are tuned for achieving best behavior with Franka
        # Typically, each new robot would have a different set of parameters.
        # Sometimes high-quality URDF or XML file would also provide this and will be parsed.
        self._robot_entity.set_dofs_kp(
            torch.tensor([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
        )
        self._robot_entity.set_dofs_kv(
            torch.tensor([450, 450, 350, 350, 200, 200, 200, 10, 10]),
        )
        self._robot_entity.set_dofs_force_range(
            torch.tensor([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            torch.tensor([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        )

    def _init(self):
        self._arm_dof_dim = self._robot_entity.n_dofs - 2  # total number of arm: joints
        self._gripper_dim = 2  # number of gripper joints

        self._arm_dof_idx = torch.arange(self._arm_dof_dim, device=self._device)
        self._fingers_dof = torch.arange(
            self._arm_dof_dim,
            self._arm_dof_dim + self._gripper_dim,
            device=self._device,
        )
        self._left_finger_dof = self._fingers_dof[0]
        self._right_finger_dof = self._fingers_dof[1]
        self._ee_link = self._robot_entity.get_link(self._args["ee_link_name"])
        self._left_finger_link = self._robot_entity.get_link(self._args["gripper_link_names"][0])
        self._right_finger_link = self._robot_entity.get_link(self._args["gripper_link_names"][1])
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

    def apply_action(self, action: torch.Tensor, open_gripper: bool) -> None:
        """
        Apply the action to the robot.
        """
        q_pos = self._robot_entity.get_qpos()
        if self._ik_method == "gs_ik":
            q_pos = self._gs_ik(action)
        elif self._ik_method == "dls_ik":
            q_pos = self._dls_ik(action)
        else:
            raise ValueError(f"Invalid control mode: {self._ik_method}")
        # set gripper to open
        if open_gripper:
            q_pos[:, self._fingers_dof] = self._gripper_open_dof
        else:
            q_pos[:, self._fingers_dof] = self._gripper_close_dof
        self._robot_entity.control_dofs_position(position=q_pos)

    def _gs_ik(self, action: torch.Tensor) -> torch.Tensor:
        """
        Genesis inverse kinematics
        """
        delta_position = action[:, :3]
        delta_orientation = action[:, 3:6]

        # compute target pose
        target_position = delta_position + self._ee_link.get_pos()
        quat_rel = xyz_to_quat(delta_orientation, rpy=True, degrees=False)
        target_orientation = transform_quat_by_quat(quat_rel, self._ee_link.get_quat())
        q_pos = self._robot_entity.inverse_kinematics(
            link=self._ee_link,
            pos=target_position,
            quat=target_orientation,
            dofs_idx_local=self._arm_dof_idx,
        )
        return q_pos

    def _dls_ik(self, action: torch.Tensor) -> torch.Tensor:
        """
        Damped least squares inverse kinematics
        """
        delta_pose = action[:, :6]
        lambda_val = 0.01
        jacobian = self._robot_entity.get_jacobian(link=self._ee_link)
        jacobian_T = jacobian.transpose(1, 2)
        lambda_matrix = (lambda_val**2) * torch.eye(n=jacobian.shape[1], device=self._device)
        delta_joint_pos = (
            jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)
        ).squeeze(-1)
        return self._robot_entity.get_qpos() + delta_joint_pos

    def go_to_goal(self, goal_pose: torch.Tensor, open_gripper: bool = True):
        q_pos = self._robot_entity.inverse_kinematics(
            link=self._ee_link,
            pos=goal_pose[:, :3],
            quat=goal_pose[:, 3:7],
            dofs_idx_local=self._arm_dof_idx,
        )
        if open_gripper:
            q_pos[:, self._fingers_dof] = self._gripper_open_dof
        else:
            q_pos[:, self._fingers_dof] = self._gripper_close_dof
        self._robot_entity.control_dofs_position(position=q_pos)

    @property
    def base_pos(self):
        return self._robot_entity.get_pos()

    @property
    def ee_pose(self) -> torch.Tensor:
        """
        The end-effector pose (the hand pose)
        """
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

    @property
    def center_finger_pose(self) -> torch.Tensor:
        """
        The center finger pose is the average of the left and right finger poses.
        """
        left_finger_pose = self.left_finger_pose
        right_finger_pose = self.right_finger_pose
        center_finger_pos = (left_finger_pose[:, :3] + right_finger_pose[:, :3]) / 2
        center_finger_quat = left_finger_pose[:, 3:7]
        return torch.cat([center_finger_pos, center_finger_quat], dim=-1)
