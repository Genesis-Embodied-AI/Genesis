import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import cv2
import numpy as np
from collections import deque


class BehaviorCloning:
    """Behavior cloning implementation."""

    def __init__(self, env, cfg: dict, teacher: nn.Module, device: str = "cpu"):
        self._env = env
        self._cfg = cfg
        self._device = device
        self._teacher = teacher

        # Initialize policy
        rgb_shape = (
            6,
            env.image_height,
            env.image_width,
        )  # Stereo rgb: 6 channels (3 left + 3 right)
        action_dim = env.num_actions

        #
        self._policy = Policy(cfg["policy"], action_dim).to(device)

        # Initialize optimizer
        self._optimizer = torch.optim.Adam(self._policy.parameters(), lr=cfg["learning_rate"])

        # Initialize learning rate scheduler
        self._scheduler = self._create_lr_scheduler()

        # Initialize buffer
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

    def _create_lr_scheduler(self):
        """Create cosine annealing learning rate scheduler."""
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self._optimizer,
            T_max=self._cfg["lr_schedule"]["T_max"],
            eta_min=self._cfg["lr_schedule"]["min_lr"],
        )

    def learn(self, num_learning_iterations: int, log_dir: str) -> None:
        """Main training loop with automatic pretraining and fine-tuning."""
        # Phase 1: Pretrain autoencoder (encoder + decoder)
        # pretrain_iterations = 10
        # print(
        #     f"=== Phase 1: Autoencoder Pretraining ({pretrain_iterations} iterations) ==="
        # )
        #
        # self._pretrain_autoencoder(pretrain_iterations, log_dir)
        #
        # Phase 2: Freeze encoder and train action MLP head
        finetune_iterations = num_learning_iterations
        print(f"=== Phase 2: Action Fine-tuning ({finetune_iterations} iterations) ===")
        print("Freezing vision encoder and training only action MLP head...")

        self._policy_only_training(finetune_iterations, log_dir)

        print("Training completed! Encoder was pretrained and then frozen for action fine-tuning.")

    def _pretrain_autoencoder(self, num_pretrain_iterations: int, log_dir: str) -> None:
        """Pretrain the autoencoder (encoder + decoder) on stereo grayscale image reconstruction."""
        for it in range(num_pretrain_iterations):
            # Collect experience
            # self._buffer.clear()
            self._collect_with_rl_teacher(it)

            # Training steps for reconstruction only
            total_loss = 0.0
            num_batches = 0

            generator = self._buffer.get_batches(self._cfg.get("mini_batches_size", 32), self._cfg["num_epochs"])
            for batch in generator:
                # Forward pass for stereo grayscale image reconstruction only
                reconstructed_images = self._policy.reconstruct(batch["grayscale_obs"].float())

                # Compute reconstruction loss with better quality metrics
                mse_loss = F.mse_loss(reconstructed_images, batch["grayscale_obs"].float())

                # Add L1 loss for better edge preservation
                l1_loss = F.l1_loss(reconstructed_images, batch["grayscale_obs"].float())

                # Combined loss (MSE + L1)
                reconstruction_loss = mse_loss + 0.1 * l1_loss

                # Backward pass
                self._optimizer.zero_grad()
                reconstruction_loss.backward()
                self._optimizer.step()
                torch.nn.utils.clip_grad_norm_(self._policy.parameters(), self._cfg["max_grad_norm"])
                total_loss += reconstruction_loss
                num_batches += 1

            # Compute average loss
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

            # Logging
            print(f"Pretrain Iteration {it}, Reconstruction Loss: {avg_loss:.6f}")
            print(f"Buffer size: {self._buffer.size}")

            if (it + 1) % 2 == 0:
                # Save reconstructed images periodically
                sample_batch = next(self._buffer.get_batches(1, 1))
                self.save_reconstructed_images(sample_batch, it, save_dir="./pretrain_reconstructions")

        # Save pretraining checkpoints
        self.save(os.path.join(log_dir, f"pretrain_checkpoint_{it + 1:04d}.pt"))

    def _policy_only_training(self, num_finetune_iterations: int, log_dir: str) -> None:
        """Fine-tune the policy for action prediction after pretraining."""
        # Freeze the shared vision encoder and decoder
        # for param in self._policy.shared_encoder.parameters():
        #     param.requires_grad = False
        # for param in self._policy.shared_decoder.parameters():
        #     param.requires_grad = False
        # for param in self._policy.feature_fusion.parameters():
        #     param.requires_grad = False

        # Only train the MLP head
        for param in self._policy.mlp.parameters():
            param.requires_grad = True

        self._rewbuffer = deque(maxlen=100)
        self._cur_reward_sum = torch.zeros(self._env.num_envs, dtype=torch.float, device=self._device)
        self._buffer.clear()
        for it in range(num_finetune_iterations):
            # Collect experience
            self._collect_with_rl_teacher(it)

            # Training steps for action prediction only
            total_loss = 0.0
            num_batches = 0

            generator = self._buffer.get_batches(self._cfg.get("mini_batches_size", 32), self._cfg["num_epochs"])
            for batch in generator:
                # Forward pass for action prediction only
                student_actions = self._policy(batch["rgb_obs"].float(), batch["state_obs"].float())

                # Compute action prediction loss only
                action_loss = F.mse_loss(student_actions, batch["actions"].float())

                # Backward pass
                self._optimizer.zero_grad()
                action_loss.backward()
                self._optimizer.step()
                torch.nn.utils.clip_grad_norm_(self._policy.parameters(), self._cfg["max_grad_norm"])
                total_loss += action_loss
                num_batches += 1

            # Compute average loss
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

            # Update learning rate schedule (cosine annealing updates every step)
            if self._scheduler is not None:
                self._scheduler.step()

            # Logging
            if (it + 1) % 10 == 0:
                current_lr = self._optimizer.param_groups[0]["lr"]
                print(f"Policy Training Iteration {it}, Action Loss: {avg_loss:.6f}, LR: {current_lr:.6f}")
                print(f"Buffer size: {self._buffer.size}")

                if len(self._rewbuffer) > 0:
                    print(f"Reward mean: {np.mean(self._rewbuffer)}")

            # Save checkpoints periodically
            if (it + 1) % 50 == 0:
                self.save(os.path.join(log_dir, f"checkpoint_{it + 1:04d}.pt"))

    def _collect_with_rl_teacher(self, it: int) -> None:
        """Collect experience from environment using stereo rgb images."""
        # Get state observation
        obs, _ = self._env.get_observations()
        with torch.inference_mode():
            for i in range(self._cfg.get("num_steps", 32)):
                # Get stereo rgb images
                rgb_obs = self._env.get_stereo_rgb_images(normalize=True)

                # Get teacher action
                teacher_action = self._teacher(obs).detach()

                # Get end-effector position
                ee_pose = self._env.robot.ee_pose

                # Store in buffer
                self._buffer.add(rgb_obs, ee_pose, teacher_action)

                # Step environment with student action
                student_action = self._policy(rgb_obs.float(), ee_pose.float()).detach()

                # Simple Dagger: use student action if its difference with teacher action is less than 0.1
                action_diff = torch.norm(student_action - teacher_action, dim=-1)
                condition = (action_diff < 0.1).unsqueeze(-1).expand_as(student_action)
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

    def save_reconstructed_images(self, batch, iteration: int, save_dir: str = "./reconstructions"):
        """Save original and reconstructed stereo grayscale images for visualization."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            # Get reconstructions
            reconstructed = self._policy.reconstruct(batch["grayscale_obs"])

            # Get first image from batch
            original_stereo = batch["grayscale_obs"][0].cpu().numpy()  # [2, H, W]
            recon_stereo = reconstructed[0].cpu().numpy()  # [2, H, W]

            # Split stereo grayscale images
            original_images = [
                original_stereo[0:1].transpose(1, 2, 0),  # left
                original_stereo[1:2].transpose(1, 2, 0),  # right
            ]
            recon_images = [
                recon_stereo[0:1].transpose(1, 2, 0),  # left
                recon_stereo[1:2].transpose(1, 2, 0),  # right
            ]
            camera_names = ["left", "right"]

            # Normalize to 0-255 range
            def normalize_image(img):
                img = (img - img.min()) / (img.max() - img.min()) * 255
                return img.astype(np.uint8)

            # Process and save individual images
            for i, (original, recon, camera) in enumerate(zip(original_images, recon_images, camera_names)):
                # Normalize
                original_norm = normalize_image(original)
                recon_norm = normalize_image(recon)

                # Convert grayscale to BGR for OpenCV (repeat single channel 3 times)
                original_bgr = cv2.cvtColor(original_norm, cv2.COLOR_GRAY2BGR)
                recon_bgr = cv2.cvtColor(recon_norm, cv2.COLOR_GRAY2BGR)

                # Save comparison (original vs reconstructed)
                comparison = np.hstack([original_bgr, recon_bgr])
                cv2.imwrite(str(save_path / f"{camera}_comparison_{iteration}.png"), comparison)

            print(f"Saved stereo grayscale reconstruction images to {save_dir}")

    def load_pretrained_autoencoder(self, path: str) -> None:
        """Load a pretrained autoencoder checkpoint."""
        checkpoint = torch.load(path, map_location=self._device, weights_only=False)
        self._policy.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._current_iter = checkpoint["current_iter"]
        print(f"Pretrained autoencoder loaded from {path}")

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
        self._state_obs = torch.zeros(max_size, num_envs, state_dim, device=device)
        self._actions = torch.zeros(max_size, num_envs, action_dim, device=device)

    def add(
        self,
        rgb_obs: torch.Tensor,
        state_obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> None:
        """Add experience to buffer."""
        ptr = self._ptr % self._max_size
        self._rgb_obs[ptr].copy_(rgb_obs)
        self._state_obs[ptr].copy_(state_obs)
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
                    "state_obs": self._state_obs.view(-1, self._state_dim)[mb_indices],
                    "actions": self._actions.view(-1, self._action_dim)[mb_indices],
                }
                yield batch

    def clear(self) -> None:
        """Clear the buffer."""
        print("Clearing buffer...")
        self._rgb_obs.zero_()
        self._state_obs.zero_()
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

        # Shared decoder for reconstruction (if needed)
        self.shared_decoder = self._build_decoder(config["vision_encoder"])

        # MLP for action prediction
        mlp_cfg = config["mlp_head"]
        self.state_obs_dim = config["mlp_head"]["state_obs_dim"]
        if self.state_obs_dim is not None:
            mlp_cfg["input_dim"] = vision_encoder_output_dim + self.state_obs_dim
        else:
            mlp_cfg["input_dim"] = vision_encoder_output_dim
        mlp_cfg["output_dim"] = action_dim
        self.mlp = self._build_mlp(mlp_cfg)

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

    def _build_decoder(self, config: dict):
        """Build decoder that properly restores original image dimensions for grayscale."""
        layers = []

        # The encoder produces 4x4x128 features
        # We need to reconstruct back to 64x64x1 (grayscale)

        # First, upsample from 4x4 to 16x16 (4x scale factor)
        layers.append(nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False))
        layers.append(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU())

        # Then upsample from 16x16 to 32x32 (2x scale factor)
        layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
        layers.append(nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())

        # Finally upsample from 32x32 to 64x64 (2x scale factor)
        layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
        layers.append(nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(16))
        layers.append(nn.ReLU())

        # Final layer to get back to 1 channel (grayscale)
        layers.append(
            nn.Conv2d(
                in_channels=16,
                out_channels=1,  # Grayscale output
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )

        return nn.Sequential(*layers)

    def _build_mlp(self, config: dict):
        mlp_input_dim = config["input_dim"]
        layers = []
        for hidden_dim in config["hidden_dims"]:
            layers.extend([nn.Linear(mlp_input_dim, hidden_dim), nn.ReLU()])
            mlp_input_dim = hidden_dim
        layers.append(nn.Linear(mlp_input_dim, config["output_dim"]))
        return nn.Sequential(*layers)

    def forward(self, rgb_obs: torch.Tensor, state_obs: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass with shared stereo encoder for rgb images."""
        # Ensure float32 for better performance
        rgb_obs = rgb_obs.float()
        if state_obs is not None:
            state_obs = state_obs.float()

        # Split stereo grayscale images (assuming format: [B, 2, H, W] where first channel is left, second is right)
        left_rgb = rgb_obs[:, 0:3]  # First channel (grayscale)
        right_rgb = rgb_obs[:, 3:6]  # Second channel (grayscale)

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

        # Predict actions
        actions = self.mlp(final_features)

        return actions

    def reconstruct(self, rgb_obs: torch.Tensor) -> torch.Tensor:
        """Reconstruct stereo rgb images using shared decoder."""
        rgb_obs = rgb_obs.float()

        # Split stereo grayscale images
        left_rgb = rgb_obs[:, 0:3]
        right_rgb = rgb_obs[:, 3:6]

        # Use shared encoder for both images
        left_encoded = self.shared_encoder(left_rgb)
        right_encoded = self.shared_encoder(right_rgb)

        # Use shared decoder for both reconstructions
        left_reconstructed = self.shared_decoder(left_encoded)
        right_reconstructed = self.shared_decoder(right_encoded)

        # Verify dimensions match
        expected_shape = rgb_obs.shape
        reconstructed_shape = torch.cat([left_reconstructed, right_reconstructed], dim=1).shape

        if expected_shape != reconstructed_shape:
            print(f"Warning: Expected shape {expected_shape}, got {reconstructed_shape}")
            print(f"Left reconstructed shape: {left_reconstructed.shape}")
            print(f"Right reconstructed shape: {right_reconstructed.shape}")

        # Concatenate reconstructions
        reconstructed = torch.cat([left_reconstructed, right_reconstructed], dim=1)

        return reconstructed
