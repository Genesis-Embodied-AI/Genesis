import os
import torch
import time
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import deque

from torch.utils.tensorboard import SummaryWriter


class BehaviorCloning:
    """Behavior cloning implementation."""

    def __init__(self, env, cfg: dict, teacher: nn.Module, device: str = "cpu"):
        self._env = env
        self._cfg = cfg
        self._device = device
        self._teacher = teacher
        self._num_steps_per_env = cfg["num_steps_per_env"]

        # Stereo rgb: 6 channels (3 left + 3 right)
        rgb_shape = (6, env.image_height, env.image_width)
        action_dim = env.num_actions

        # Vision-based policy
        self._policy = Policy(cfg["policy"], action_dim).to(device)

        # Initialize optimizer
        self._optimizer = torch.optim.Adam(self._policy.parameters(), lr=cfg["learning_rate"])

        # Learning rate scheduler
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._optimizer,
            T_max=self._cfg["lr_schedule"]["T_max"],
            eta_min=self._cfg["lr_schedule"]["min_lr"],
        )

        # Experience buffer
        self._buffer = ExperienceBuffer(
            num_envs=env.num_envs,
            max_size=self._cfg["buffer_size"],
            img_shape=rgb_shape,
            state_dim=self._cfg["policy"]["mlp_head"]["state_obs_dim"],
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
            self._collect_with_rl_teacher(it)
            end_time = time.time()
            forward_time = end_time - start_time

            # Training steps for action prediction only
            total_loss = 0.0
            num_batches = 0

            start_time = time.time()
            generator = self._buffer.get_batches(self._cfg.get("mini_batches_size", 32), self._cfg["num_epochs"])
            for batch in generator:
                # Forward pass for multi-task prediction
                student_outputs = self._policy(batch["rgb_obs"].float(), batch["robot_pose"].float())

                # Compute action prediction loss
                action_loss = F.mse_loss(student_outputs["actions"], batch["actions"].float())

                # Compute pose prediction loss (using state_obs as ground truth pose)
                # This is pseudo-loss for pose prediction
                pose_loss = F.mse_loss(student_outputs["pose"], batch["obj_pose"].float())

                # Combined multi-task loss
                total_batch_loss = (
                    self._cfg["action_loss_weight"] * action_loss + self._cfg["pose_loss_weight"] * pose_loss
                )

                # Backward pass
                self._optimizer.zero_grad()
                total_batch_loss.backward()
                self._optimizer.step()
                torch.nn.utils.clip_grad_norm_(self._policy.parameters(), self._cfg["max_grad_norm"])
                total_loss += total_batch_loss
                num_batches += 1

            end_time = time.time()
            backward_time = end_time - start_time

            # Compute average loss
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

            # Update learning rate schedule (cosine annealing updates every step)
            if self._scheduler is not None:
                self._scheduler.step()

            fps = (self._num_steps_per_env * self._env.num_envs) / (forward_time)
            # Logging
            if (it + 1) % 10 == 0:
                current_lr = self._optimizer.param_groups[0]["lr"]

                tf_writer.add_scalar("loss/total_loss", avg_loss, it)
                tf_writer.add_scalar("loss/action_loss", action_loss, it)
                tf_writer.add_scalar("loss/pose_loss", pose_loss, it)
                tf_writer.add_scalar("lr", current_lr, it)
                tf_writer.add_scalar("buffer_size", self._buffer.size, it)
                tf_writer.add_scalar("speed/forward", forward_time, it)
                tf_writer.add_scalar("speed/backward", backward_time, it)
                tf_writer.add_scalar("speed/fps", int(fps), it)

                #
                print("--------------------------------\n")
                info_str = f" | Iteration: {it + 1}\n"
                info_str += f" | Total Loss: {avg_loss:.6f}\n"
                info_str += f" | Action Loss: {action_loss:.6f}\n"
                info_str += f" | Pose Loss: {pose_loss:.6f}\n"
                info_str += f" | LR: {current_lr:.6f}\n"
                info_str += f" | Forward Time: {forward_time:.2f}s\n"
                info_str += f" | Backward Time: {backward_time:.2f}s\n"
                info_str += f" | FPS: {int(fps)}\n"
                print(info_str)

                if len(self._rewbuffer) > 0:
                    tf_writer.add_scalar("reward/mean", np.mean(self._rewbuffer), it)

            # Save checkpoints periodically
            if (it + 1) % 50 == 0:
                self.save(os.path.join(log_dir, f"checkpoint_{it + 1:04d}.pt"))

        tf_writer.close()

    def _collect_with_rl_teacher(self, it: int) -> None:
        """Collect experience from environment using stereo rgb images."""
        # Get state observation
        obs, _ = self._env.get_observations()
        with torch.inference_mode():
            for i in range(self._num_steps_per_env):
                # Get stereo rgb images
                rgb_obs = self._env.get_stereo_rgb_images(normalize=True)

                # Get teacher action
                teacher_action = self._teacher(obs).detach()

                # Get end-effector position
                robot_pose = self._env.robot.ee_pose
                obj_pose = torch.cat([self._env.object.get_pos(), self._env.object.get_quat()], dim=-1)

                # Store in buffer
                self._buffer.add(rgb_obs, robot_pose, obj_pose, teacher_action)

                # Step environment with student action
                student_outputs = self._policy(rgb_obs.float(), robot_pose.float())
                student_action = student_outputs["actions"].detach()

                # Simple Dagger: use student action if its difference with teacher action is less than 0.5
                action_diff = torch.norm(student_action - teacher_action, dim=-1)
                condition = (action_diff < 0.5).unsqueeze(-1).expand_as(student_action)
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
        self._obj_pose = torch.zeros(max_size, num_envs, state_dim, device=device)
        self._actions = torch.zeros(max_size, num_envs, action_dim, device=device)

    def add(
        self,
        rgb_obs: torch.Tensor,
        robot_pose: torch.Tensor,
        obj_pose: torch.Tensor,
        actions: torch.Tensor,
    ) -> None:
        """Add experience to buffer."""
        ptr = self._ptr % self._max_size
        self._rgb_obs[ptr].copy_(rgb_obs)
        self._robot_pose[ptr].copy_(robot_pose)
        self._obj_pose[ptr].copy_(obj_pose)
        self._actions[ptr].copy_(actions)
        self._ptr = self._ptr + 1
        self._size = min(self._size + 1, self._max_size)

    def get_batches(self, mini_batches_size: int, num_epochs: int):
        """Generate batches for training."""
        buffer_size = self._size * self._num_envs
        # calculate the size of each mini-batch
        num_batches = 10
        for epoch in range(num_epochs):
            indices = torch.randperm(buffer_size, device=self._device)
            for batch_idx in range(num_batches):
                start = batch_idx * mini_batches_size
                end = start + mini_batches_size
                mb_indices = indices[start:end]

                # Yield a mini-batch of data
                batch = {
                    "rgb_obs": self._rgb_obs.view(-1, *self._img_shape)[mb_indices],
                    "robot_pose": self._robot_pose.view(-1, self._state_dim)[mb_indices],
                    "obj_pose": self._obj_pose.view(-1, self._state_dim)[mb_indices],
                    "actions": self._actions.view(-1, self._action_dim)[mb_indices],
                }
                yield batch

    def clear(self) -> None:
        """Clear the buffer."""
        self._rgb_obs.zero_()
        self._robot_pose.zero_()
        self._obj_pose.zero_()
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
        mlp_cfg = config["mlp_head"]
        self.state_obs_dim = config["mlp_head"]["state_obs_dim"]
        if self.state_obs_dim is not None:
            mlp_cfg["input_dim"] = vision_encoder_output_dim + self.state_obs_dim
        else:
            mlp_cfg["input_dim"] = vision_encoder_output_dim
        mlp_cfg["output_dim"] = action_dim
        self.mlp = self._build_mlp(mlp_cfg)

        # MLP for pose prediction (7D: 3D position + 4D quaternion)
        pose_mlp_cfg = mlp_cfg.copy()
        pose_mlp_cfg["output_dim"] = 7  # 3D position + 4D quaternion
        self.pose_mlp = self._build_mlp(pose_mlp_cfg)

        # Force float32 for better performance
        self.float()

    def _build_cnn(self, config: dict):
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

    def _build_mlp(self, config: dict):
        mlp_input_dim = config["input_dim"]
        layers = []
        for hidden_dim in config["hidden_dims"]:
            layers.extend([nn.Linear(mlp_input_dim, hidden_dim), nn.ReLU()])
            mlp_input_dim = hidden_dim
        layers.append(nn.Linear(mlp_input_dim, config["output_dim"]))
        return nn.Sequential(*layers)

    def forward(self, rgb_obs: torch.Tensor, state_obs: torch.Tensor | None = None) -> dict:
        """Forward pass with shared stereo encoder for rgb images."""
        # Ensure float32 for better performance
        rgb_obs = rgb_obs.float()
        if state_obs is not None:
            state_obs = state_obs.float()

        # Split stereo rgb images
        left_rgb = rgb_obs[:, 0:3]  # First 3 channels (RGB)
        right_rgb = rgb_obs[:, 3:6]  # Last 3 channels (RGB)

        # Use shared encoder for both images
        left_features = self.shared_encoder(left_rgb).flatten(start_dim=1)
        right_features = self.shared_encoder(right_rgb).flatten(start_dim=1)

        # Concatenate features (much more efficient than concatenating raw images)
        combined_features = torch.cat([left_features, right_features], dim=-1)

        # Feature fusion
        fused_features = self.feature_fusion(combined_features)

        # Add state information if available
        if state_obs is not None and self.state_obs_dim is not None:
            final_features = torch.cat([fused_features, state_obs], dim=-1)
        else:
            final_features = fused_features

        # Predict actions and pose
        actions = self.mlp(final_features)
        pose = self.pose_mlp(final_features)

        return {"actions": actions, "pose": pose}
