import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class BehaviorCloning:
    """Behavior cloning implementation."""

    def __init__(self, env, cfg: dict, teacher: nn.Module, device: str = "cpu"):
        self._env = env
        self._cfg = cfg
        self._device = device
        self._teacher = teacher

        # Initialize policy
        rgb_shape = env.rgb_image_shape
        action_dim = env.num_actions

        #
        self._policy = Policy(cfg["policy"], action_dim).to(device)

        # Initialize optimizer
        self._optimizer = torch.optim.Adam(self._policy.parameters(), lr=cfg["learning_rate"])

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

    def learn(self, num_learning_iterations: int, log_dir: str) -> None:
        """Main training loop."""
        for it in range(self._current_iter, self._current_iter + num_learning_iterations):
            # Collect experience
            self._collect_experience(it)

            # Training steps
            total_loss = 0.0
            num_batches = 0

            generator = self._buffer.get_batches(self._cfg.get("num_mini_batches", 32), self._cfg["num_epochs"])
            for batch in generator:
                # Forward pass
                student_actions = self._policy(batch["depth_obs"], batch["state_obs"])

                # Compute loss
                loss = F.mse_loss(student_actions, batch["actions"])

                # Backward pass
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                torch.nn.utils.clip_grad_norm_(self._policy.parameters(), self._cfg["max_grad_norm"])
                total_loss += loss
                num_batches += 1

            # Compute average loss
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

            # Logging
            if it % 10 == 0:
                print(f"Iteration {it}, Average Loss: {avg_loss:.6f}")
                print(f"Buffer size: {self._buffer.size}")

            # Save checkpoints
            if it % self._cfg["save_freq"] == 0:
                self.save(os.path.join(log_dir, f"checkpoint_{it}.pt"))

            self.current_iter = it

    def _collect_experience(self, it: int) -> None:
        """Collect experience from environment."""
        log_dir = "./depth_images"
        folder = Path(log_dir) / "experience" / f"iter_{it}"
        folder.mkdir(parents=True, exist_ok=True)

        # Get state observation
        obs, _ = self._env.get_observations()
        with torch.inference_mode():
            for i in range(self._cfg.get("num_steps", 32)):
                # save depth image
                rgb_obs = self._env.get_rgb_image(normalize=True)

                # Get teacher action
                teacher_action = self._teacher(obs).detach()

                # Get end-effector position
                ee_pose = self._env.robot.ee_pose

                # Store in buffer
                self._buffer.add(rgb_obs, ee_pose, teacher_action)

                # Step environment with student action
                student_action = self._policy(rgb_obs, ee_pose).detach()

                # Simple Dagger: use student action if its difference with teacher action is less than 0.1
                action_diff = torch.norm(student_action - teacher_action, dim=-1)
                condition = (action_diff < 0.1).unsqueeze(-1).expand_as(student_action)
                action = torch.where(condition, student_action, teacher_action)

                next_obs, reward, done, _ = self._env.step(action)

                obs = next_obs

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
        checkpoint = torch.load(path, map_location=self._device)
        self._policy.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_iter = checkpoint["current_iter"]
        print(f"Model loaded from {path}")


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
        self._depth_obs = torch.zeros(max_size, num_envs, *img_shape, device=device)
        self._state_obs = torch.zeros(max_size, num_envs, state_dim, device=device)
        self._actions = torch.zeros(max_size, num_envs, action_dim, device=device)

    def add(self, depth_obs: torch.Tensor, state_obs: torch.Tensor, actions: torch.Tensor) -> None:
        """Add experience to buffer."""
        ptr = self._ptr % self._max_size
        self._depth_obs[ptr].copy_(depth_obs)
        self._state_obs[ptr].copy_(state_obs)
        self._actions[ptr].copy_(actions)
        self._ptr = self._ptr + 1
        self._size = min(self._size + 1, self._max_size)

    def get_batches(self, num_mini_batches: int, num_epochs: int):
        """Generate batches for training."""
        buffer_size = self._size * self._num_envs
        indices = torch.randperm(buffer_size, device=self._device)
        # calculate the size of each mini-batch
        batch_size = buffer_size // num_mini_batches

        for _ in range(num_epochs):
            for start in range(0, len(indices), batch_size):
                end = start + batch_size
                mb_indices = indices[start:end]
                # Yield a mini-batch of data
                batch = {
                    "depth_obs": self._depth_obs.view(-1, *self._img_shape)[mb_indices],
                    "state_obs": self._state_obs.view(-1, self._state_dim)[mb_indices],
                    "actions": self._actions.view(-1, self._action_dim)[mb_indices],
                }
                yield batch

    def is_full(self) -> bool:
        return self._size >= self._max_size

    @property
    def size(self) -> int:
        return self._size


class Policy(nn.Module):
    """Simple behavior cloning policy."""

    def __init__(self, config: dict, action_dim: int):
        super().__init__()

        # Vision encoder
        self.vision_encoder = self._build_cnn(config["vision_encoder"])

        # MLP
        mlp_cfg = config["mlp_head"]
        vision_encoder_conv_out_channels = config["vision_encoder"]["conv_layers"][-1]["out_channels"]
        vision_encoder_output_dim = vision_encoder_conv_out_channels * 4 * 4

        self.state_obs_dim = config["mlp_head"]["state_obs_dim"]
        if self.state_obs_dim is not None:
            mlp_cfg["input_dim"] = vision_encoder_output_dim + self.state_obs_dim
        else:
            mlp_cfg["input_dim"] = vision_encoder_output_dim
        mlp_cfg["output_dim"] = action_dim
        self.mlp = self._build_mlp(mlp_cfg)

    def _build_cnn(self, config: dict):
        layers = []
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
                    nn.ReLU(),
                ]
            )
        # add adaptive avg pooling
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

    def forward(self, depth_obs: torch.Tensor, state_obs: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass through the policy."""
        # Encode vision
        vision_features = self.vision_encoder(depth_obs).flatten(start_dim=1)

        if state_obs is not None and self.state_obs_dim is not None:
            # Concatenate vision and state features
            combined_features = torch.cat([vision_features, state_obs], dim=-1)
        else:
            combined_features = vision_features

        # Predict actions
        actions = self.mlp(combined_features)

        return actions
