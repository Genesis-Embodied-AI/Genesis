import os
import torch
import time
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import deque

from torch.utils.tensorboard import SummaryWriter


class BehaviorCloning:
    """Multi-task behavior cloning with action prediction and object pose estimation"""

    def __init__(self, env, cfg: dict, teacher: nn.Module, device: str = "cpu"):
        self._env = env
        self._cfg = cfg
        self._device = device
        self._teacher = teacher
        self._num_steps_per_env = cfg["num_steps_per_env"]

        # Stereo rgb: 6 channels (3 left + 3 right)
        rgb_shape = (6, env.image_height, env.image_width)
        action_dim = env.num_actions

        # Multi-task policy with action and pose heads
        self._policy = Policy(cfg["policy"], action_dim).to(device)

        # Initialize optimizer
        self._optimizer = torch.optim.Adam(self._policy.parameters(), lr=cfg["learning_rate"])

        # Experience buffer with pose data
        self._buffer = ExperienceBuffer(
            num_envs=env.num_envs,
            max_size=self._cfg["buffer_size"],
            img_shape=rgb_shape,
            state_dim=self._cfg["policy"]["action_head"]["state_obs_dim"],
            action_dim=action_dim,
            device=self._device,
        )

        # Training state
        self._current_iter = 0

    def learn(self, num_learning_iterations: int, log_dir: str) -> None:
        self._rewbuffer = deque(maxlen=100)
        self._cur_reward_sum = torch.zeros(self._env.num_envs, dtype=torch.float, device=self._device)
        self._buffer.clear()

        tf_writer = SummaryWriter(log_dir)

        for it in range(num_learning_iterations):
            # Collect experience
            start_time = time.time()
            self._collect_with_rl_teacher()
            end_time = time.time()
            forward_time = end_time - start_time

            # Training steps for both action and pose prediction
            total_action_loss = 0.0
            total_pose_loss = 0.0
            num_batches = 0

            start_time = time.time()
            generator = self._buffer.get_batches(self._cfg.get("mini_batches_size", 32), self._cfg["num_epochs"])
            for batch in generator:
                # Forward pass for both action and pose prediction
                pred_action = self._policy(batch["rgb_obs"].float(), batch["robot_pose"].float())
                pred_left_pose, pred_right_pose = self._policy.predict_pose(batch["rgb_obs"].float())

                # Compute action prediction loss
                action_loss = F.mse_loss(pred_action, batch["actions"].float())

                # Compute pose estimation loss (position + orientation)
                pose_left_loss = self._compute_pose_loss(pred_left_pose, batch["object_poses"].float())
                pose_right_loss = self._compute_pose_loss(pred_right_pose, batch["object_poses"].float())
                pose_loss = pose_left_loss + pose_right_loss

                # Combined loss with weights
                total_loss = action_loss + pose_loss

                # Backward pass
                self._optimizer.zero_grad()
                total_loss.backward()
                self._optimizer.step()
                torch.nn.utils.clip_grad_norm_(self._policy.parameters(), self._cfg["max_grad_norm"])

                total_action_loss += action_loss
                total_pose_loss += pose_loss
                num_batches += 1

            end_time = time.time()
            backward_time = end_time - start_time

            # Compute average losses
            avg_action_loss = total_action_loss / num_batches if num_batches > 0 else 0.0
            avg_pose_loss = total_pose_loss / num_batches if num_batches > 0 else 0.0

            fps = (self._num_steps_per_env * self._env.num_envs) / (forward_time)
            # Logging
            if (it + 1) % 10 == 0:
                current_lr = self._optimizer.param_groups[0]["lr"]

                tf_writer.add_scalar("loss/action_loss", avg_action_loss, it)
                tf_writer.add_scalar("loss/pose_loss", avg_pose_loss, it)
                tf_writer.add_scalar("loss/total_loss", avg_action_loss + avg_pose_loss, it)
                tf_writer.add_scalar("lr", current_lr, it)
                tf_writer.add_scalar("buffer_size", self._buffer.size, it)
                tf_writer.add_scalar("speed/forward", forward_time, it)
                tf_writer.add_scalar("speed/backward", backward_time, it)
                tf_writer.add_scalar("speed/fps", int(fps), it)

                #
                print("--------------------------------")
                info_str = f" | Iteration:     {it + 1:04d}\n"
                info_str += f" | Action Loss:   {avg_action_loss:.6f}\n"
                info_str += f" | Pose Loss:     {avg_pose_loss:.6f}\n"
                info_str += f" | Total Loss:    {avg_action_loss + avg_pose_loss:.6f}\n"
                info_str += f" | Learning Rate: {current_lr:.6f}\n"
                info_str += f" | Forward Time:  {forward_time:.2f}s\n"
                info_str += f" | Backward Time: {backward_time:.2f}s\n"
                info_str += f" | FPS:           {int(fps)}"
                print(info_str)

                if len(self._rewbuffer) > 0:
                    tf_writer.add_scalar("reward/mean", np.mean(self._rewbuffer), it)

            # Save checkpoints periodically
            if (it + 1) % 50 == 0:
                self.save(os.path.join(log_dir, f"checkpoint_{it + 1:04d}.pt"))

        tf_writer.close()

    def _compute_pose_loss(self, pred_poses: torch.Tensor, target_poses: torch.Tensor) -> torch.Tensor:
        """Compute pose loss with separate position and orientation components."""
        # Split into position and orientation
        pred_pos = pred_poses[:, :3]
        pred_quat = pred_poses[:, 3:7]
        target_pos = target_poses[:, :3]
        target_quat = target_poses[:, 3:7]

        # Position loss (MSE)
        pos_loss = F.mse_loss(pred_pos, target_pos)

        # Orientation loss (quaternion distance)
        # Normalize quaternions
        pred_quat = F.normalize(pred_quat, p=2, dim=1)
        target_quat = F.normalize(target_quat, p=2, dim=1)

        # Quaternion distance: 1 - |dot(q1, q2)|
        quat_dot = torch.sum(pred_quat * target_quat, dim=1)
        quat_loss = torch.mean(1.0 - torch.abs(quat_dot))

        return pos_loss + quat_loss

    def _collect_with_rl_teacher(self) -> None:
        """Collect experience from environment using stereo rgb images and object poses."""
        # Get state observation
        obs, _ = self._env.get_observations()
        with torch.inference_mode():
            for _ in range(self._num_steps_per_env):
                # Get stereo rgb images
                rgb_obs = self._env.get_stereo_rgb_images(normalize=True)

                # Get teacher action
                teacher_action = self._teacher(obs).detach()

                # Get end-effector position
                ee_pose = self._env.robot.ee_pose

                # Get object pose in camera frame
                # object_pose_camera = self._get_object_pose_in_camera_frame()
                object_pose = torch.cat(
                    [
                        self._env.object.get_pos(),
                        self._env.object.get_quat(),
                    ],
                    dim=-1,
                )

                # Store in buffer
                self._buffer.add(rgb_obs, ee_pose, object_pose, teacher_action)

                # Step environment with student action
                student_action = self._policy(rgb_obs.float(), ee_pose.float())

                # Simple Dagger: use student action if its difference with teacher action is less than 0.5
                action_diff = torch.norm(student_action - teacher_action, dim=-1)
                condition = (action_diff < 1.0).unsqueeze(-1).expand_as(student_action)
                action = torch.where(condition, student_action, teacher_action)

                next_obs, reward, done, _ = self._env.step(action)
                self._cur_reward_sum += reward

                obs = next_obs
                new_ids = (done > 0).nonzero(as_tuple=False)
                self._rewbuffer.extend(self._cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                self._cur_reward_sum[new_ids] = 0

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self._policy.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "current_iter": self._current_iter,
            "config": self._cfg,
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self._device, weights_only=False)
        self._policy.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_iter = checkpoint["current_iter"]
        print(f"Model loaded from {path}")

    def load_finetuned_model(self, path: str) -> None:
        """Load a fine-tuned model checkpoint."""
        checkpoint = torch.load(path, map_location=self._device, weights_only=False)
        self._policy.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._current_iter = checkpoint["current_iter"]
        print(f"Fine-tuned model loaded from {path}")


class ExperienceBuffer:
    """Experience buffer."""

    def __init__(
        self,
        num_envs: int,
        max_size: int,
        img_shape: tuple[int, int, int],
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
    ):
        self._img_shape = img_shape
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._num_envs = num_envs
        self._max_size = max_size
        self._device = device
        self._size = 0
        self._ptr = 0  # pointer to the next free slot in the buffer

        # Initialize buffers
        self._rgb_obs = torch.zeros(max_size, num_envs, *img_shape, device=device)
        self._robot_pose = torch.zeros(max_size, num_envs, state_dim, device=device)
        self._object_poses = torch.zeros(max_size, num_envs, 7, device=device)
        self._actions = torch.zeros(max_size, num_envs, action_dim, device=device)

    def add(
        self,
        rgb_obs: torch.Tensor,
        robot_pose: torch.Tensor,
        object_poses: torch.Tensor,
        actions: torch.Tensor,
    ) -> None:
        """Add experience to buffer."""
        ptr = self._ptr % self._max_size
        self._rgb_obs[ptr].copy_(rgb_obs)
        self._robot_pose[ptr].copy_(robot_pose)
        self._object_poses[ptr].copy_(object_poses)
        self._actions[ptr].copy_(actions)
        self._ptr = self._ptr + 1
        self._size = min(self._size + 1, self._max_size)

    def get_batches(self, mini_batches_size: int, num_epochs: int):
        """Generate batches for training."""
        buffer_size = self._size * self._num_envs
        indices = torch.randperm(buffer_size, device=self._device)
        # calculate the size of each mini-batch
        num_batches = min(buffer_size // mini_batches_size, 10)
        for _ in range(num_epochs):
            for batch_idx in range(num_batches):
                start = batch_idx * mini_batches_size
                end = start + mini_batches_size
                mb_indices = indices[start:end]

                # Yield a mini-batch of data
                batch = {
                    "rgb_obs": self._rgb_obs.view(-1, *self._img_shape)[mb_indices],
                    "robot_pose": self._robot_pose.view(-1, self._state_dim)[mb_indices],
                    "object_poses": self._object_poses.view(-1, 7)[mb_indices],
                    "actions": self._actions.view(-1, self._action_dim)[mb_indices],
                }
                yield batch

    def clear(self) -> None:
        """Clear the buffer."""
        self._rgb_obs.zero_()
        self._robot_pose.zero_()
        self._actions.zero_()
        self._size = 0
        self._ptr = 0

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self._size == self._max_size

    @property
    def size(self) -> int:
        """Get buffer size."""
        return self._size


class Policy(nn.Module):
    """Multi-task behavior cloning policy with shared stereo encoder/decoder."""

    def __init__(self, config: dict, action_dim: int):
        super().__init__()

        # Shared encoder for both left and right cameras
        self.shared_encoder = self._build_cnn(config["vision_encoder"])

        # Feature fusion layer to combine stereo features
        vision_encoder_conv_out_channels = config["vision_encoder"]["conv_layers"][-1]["out_channels"]
        vision_encoder_output_dim = vision_encoder_conv_out_channels * 4 * 4

        self.feature_fusion = nn.Sequential(
            nn.Linear(vision_encoder_output_dim * 2, vision_encoder_output_dim),  # 2 cameras
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # MLP for action prediction
        mlp_cfg = config["action_head"]
        self.state_obs_dim = config["action_head"]["state_obs_dim"]
        if self.state_obs_dim is not None:
            mlp_cfg["input_dim"] = vision_encoder_output_dim + self.state_obs_dim
        else:
            mlp_cfg["input_dim"] = vision_encoder_output_dim
        mlp_cfg["output_dim"] = action_dim
        self.mlp = self._build_mlp(mlp_cfg)

        # MLP for pose prediction
        pose_mlp_cfg = config["pose_head"]
        pose_mlp_cfg["input_dim"] = vision_encoder_output_dim
        pose_mlp_cfg["output_dim"] = 7
        self.pose_mlp = self._build_mlp(pose_mlp_cfg)

        # Force float32 for better performance
        self.float()

    def _build_cnn(self, config: dict) -> nn.Sequential:
        """Build CNN encoder for grayscale images."""
        layers = []

        # Build layers from configuration
        for conv_config in config["conv_layers"]:
            layers.extend(
                [
                    nn.Conv2d(
                        conv_config["in_channels"],
                        conv_config["out_channels"],
                        kernel_size=conv_config["kernel_size"],
                        stride=conv_config["stride"],
                        padding=conv_config["padding"],
                    ),
                    nn.BatchNorm2d(conv_config["out_channels"]),
                    nn.ReLU(),
                ]
            )

        # Add adaptive pooling if specified
        if config.get("pooling") == "adaptive_avg":
            layers.append(nn.AdaptiveAvgPool2d((4, 4)))

        return nn.Sequential(*layers)

    def _build_mlp(self, config: dict) -> nn.Sequential:
        mlp_input_dim = config["input_dim"]
        layers = []
        for hidden_dim in config["hidden_dims"]:
            layers.extend([nn.Linear(mlp_input_dim, hidden_dim), nn.ReLU()])
            mlp_input_dim = hidden_dim
        layers.append(nn.Linear(mlp_input_dim, config["output_dim"]))
        return nn.Sequential(*layers)

    def get_features(self, rgb_obs: torch.Tensor) -> torch.Tensor:
        # Split stereo rgb images
        left_rgb = rgb_obs[:, 0:3]  # First 3 channels (RGB)
        right_rgb = rgb_obs[:, 3:6]  # Last 3 channels (RGB)

        # Use shared encoder for both images
        left_features = self.shared_encoder(left_rgb).flatten(start_dim=1)
        right_features = self.shared_encoder(right_rgb).flatten(start_dim=1)
        return left_features, right_features

    def forward(self, rgb_obs: torch.Tensor, state_obs: torch.Tensor | None = None) -> dict:
        """Forward pass with shared stereo encoder for rgb images."""
        # Ensure float32 for better performance
        rgb_obs = rgb_obs.float()
        if state_obs is not None:
            state_obs = state_obs.float()

        # Get features
        left_features, right_features = self.get_features(rgb_obs)

        # Concatenate features (much more efficient than concatenating raw images)
        combined_features = torch.cat([left_features, right_features], dim=-1)
        # Feature fusion
        fused_features = self.feature_fusion(combined_features)

        # Add state information if available
        if state_obs is not None and self.state_obs_dim is not None:
            final_features = torch.cat([fused_features, state_obs], dim=-1)
        else:
            final_features = fused_features

        # Predict actions
        return self.mlp(final_features)

    def predict_pose(self, rgb_obs: torch.Tensor) -> torch.Tensor:
        """Predict pose from rgb images and state observations."""
        # Ensure float32 for better performance
        rgb_obs = rgb_obs.float()
        left_features, right_features = self.get_features(rgb_obs)
        left_pose = self.pose_mlp(left_features)
        right_pose = self.pose_mlp(right_features)
        return left_pose, right_pose
