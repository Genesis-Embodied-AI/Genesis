import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class BehaviorCloning:
    """Behavior cloning implementation."""

    def __init__(self, env, cfg: dict, teacher: nn.Module, device: str = "cpu"):
        self._env = env
        self._cfg = cfg
        self._device = device
        self._teacher = teacher

        # Initialize policy
        depth_shape = env.depth_image_shape
        state_dim = env.num_obs
        action_dim = env.num_actions

        #
        self._policy = Policy(cfg["policy"], action_dim).to(device)

        # Initialize optimizer
        self._optimizer = torch.optim.Adam(self._policy.parameters(), lr=cfg["learning_rate"])

        # Initialize buffer
        self._buffer = ExperienceBuffer(
            max_size=self._cfg["buffer_size"],
            img_shape=depth_shape,
            state_dim=state_dim,
            action_dim=action_dim,
            device=self._device,
        )

        # Training state
        self.current_iter = 0

    def train(self, num_iters: int, log_dir: str) -> None:
        """Main training loop."""
        # Get initial observations
        obs, _ = self._env.get_observations()

        for it in range(self.current_iter, self.current_iter + num_iters):
            # Collect experience
            self._collect_experience(obs)

            # Training steps
            total_loss = 0.0
            num_batches = 0

            for _ in range(self._cfg["num_epochs"]):
                for batch in self._buffer.get_batches(self._cfg.get("batch_size", 32)):
                    # Forward pass
                    student_actions = self._policy(batch["depth_obs"], batch["state_obs"])

                    # Compute loss
                    loss = F.mse_loss(student_actions, batch["actions"])

                    # Backward pass
                    self._optimizer.zero_grad()
                    loss.backward()
                    self._optimizer.step()
                    total_loss += loss
                    num_batches += 1

            # Compute average loss
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

            # Logging
            if it % 10 == 0:
                print(f"Iteration {it}, Average Loss: {avg_loss:.6f}")

            # Save checkpoints
            if it % self._cfg["save_freq"] == 0:
                self.save(os.path.join(log_dir, f"checkpoint_{it}.pt"))

            self.current_iter = it

    def _collect_experience(self, obs):
        """Collect experience from environment."""
        with torch.inference_mode():
            for step in range(self._cfg.get("batch_size", 32)):
                # Get depth observation
                depth_obs = self._env.get_depth_image(normalize=True)

                # Get teacher action
                teacher_action = self._teacher.act(obs).detach()

                # Get state observation
                state_obs = self._env.get_privileged_observations()

                # Store in buffer
                self._buffer.add(depth_obs, state_obs, teacher_action)

                # Step environment with student action
                student_action = self._policy(depth_obs, state_obs).detach()
                obs, reward, done, _ = self._env.step(student_action)

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self._policy.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "current_iter": self.current_iter,
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
        max_size: int,
        img_shape: tuple[int, int, int],
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
    ):
        self.max_size = max_size
        self.device = device
        self.size = 0
        self.ptr = 0

        # Initialize buffers
        self.depth_obs = torch.zeros(max_size, *img_shape, device=device)
        self.state_obs = torch.zeros(max_size, state_dim, device=device)
        self.actions = torch.zeros(max_size, action_dim, device=device)

    def add(self, depth_obs: torch.Tensor, state_obs: torch.Tensor, actions: torch.Tensor):
        """Add experience to buffer."""
        batch_size = depth_obs.shape[0]
        #
        self.depth_obs[self.ptr : self.ptr + batch_size] = depth_obs
        self.state_obs[self.ptr : self.ptr + batch_size] = state_obs
        self.actions[self.ptr : self.ptr + batch_size] = actions

        self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)

    def get_batches(self, batch_size: int):
        """Generate batches for training."""
        if self.size < 2:
            return

        indices = torch.randperm(self.size, device=self.device)

        for i in range(0, self.size, batch_size):
            batch_indices = indices[i : i + batch_size]

            yield {
                "depth_obs": self.depth_obs[batch_indices],
                "state_obs": self.state_obs[batch_indices],
                "actions": self.actions[batch_indices],
            }


class Policy(nn.Module):
    """Simple behavior cloning policy."""

    def __init__(self, config: dict, action_dim: int):
        super().__init__()

        # Vision encoder
        self.vision_encoder = self._build_cnn(config["vision_encoder"])

        # MLP
        mlp_cfg = config["mlp_head"]
        mlp_cfg["input_dim"] = config["vision_encoder"]["conv_layers"][-1]["out_channels"]
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
        return nn.Sequential(*layers)

    def _build_mlp(self, config: dict):
        mlp_input_dim = config["input_dim"]
        layers = []
        for hidden_dim in config["hidden_dims"]:
            layers.extend([nn.Linear(mlp_input_dim, hidden_dim), nn.ReLU()])
            mlp_input_dim = hidden_dim
        layers.append(nn.Linear(mlp_input_dim, config["output_dim"]))
        return nn.Sequential(*layers)

    def forward(self, depth_obs: torch.Tensor, state_obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the policy."""
        # Encode vision
        vision_features = self.vision_encoder(depth_obs)

        # Concatenate vision and state features
        combined_features = torch.cat([vision_features, state_obs], dim=-1)

        # Predict actions
        actions = self.mlp(combined_features)

        return actions
