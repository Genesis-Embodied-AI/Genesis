import argparse
import os
import numpy as np
from huggingface_hub import snapshot_download

import genesis as gs


class AutoJointAnimator:
    def __init__(self, scene: gs.Scene):
        rigid = scene.sim.rigid_solver

        # Get joint limits and handle invalid values (inf/nan -> reasonable defaults)
        joint_limits = rigid.dofs_info.limit.to_numpy()
        # Clip invalid limits to reasonable range
        joint_limits = np.clip(joint_limits, -np.pi, np.pi)

        # Get initial joint positions
        init_positions = rigid.get_dofs_position().numpy()
        if init_positions.ndim > 1:
            init_positions = init_positions[0]  # Take first environment if batched

        # Store joint lower and upper limits
        self.joint_lower = joint_limits[:, 0]
        self.joint_upper = joint_limits[:, 1]
        joint_ranges_original = self.joint_upper - self.joint_lower

        # Handle zero or invalid ranges
        valid_range_mask = joint_ranges_original > 1e-6

        # Normalize initial positions to [-1, 1] range
        # This maps: lower → -1, upper → 1
        normalized_init = np.zeros_like(init_positions)
        for i in range(len(init_positions)):
            if valid_range_mask[i]:
                normalized_init[i] = 2.0 * (init_positions[i] - self.joint_lower[i]) / joint_ranges_original[i] - 1.0
            else:
                normalized_init[i] = 0.0  # Center if no valid range

        # Clamp to valid range for arcsin
        normalized_init = np.clip(normalized_init, -1.0, 1.0)

        # Map normalized initial position to phase offset using arcsin
        # This ensures sin(phase_offset) = normalized_init, so the animation starts at the initial position
        self.phase_offsets = np.arcsin(normalized_init)

        self.rigid = rigid

    def animate(self, scene: gs.Scene):
        # Calculate target positions using sin function to interpolate between lower and upper limits
        # sin ranges from -1 to 1, we map it to [lower, upper]
        # Formula: target = lower + (upper - lower) * (sin(...) + 1) / 2
        t = scene.t * scene.dt
        theta = np.pi * t + self.phase_offsets
        theta = theta % (2 * np.pi)
        sin_values = np.sin(theta)

        # Map sin from [-1, 1] to [0, 1]
        normalized = (sin_values + 1.0) / 2.0

        # Interpolate between lower and upper limits
        target = self.joint_lower + (self.joint_upper - self.joint_lower) * normalized

        # Apply the target positions
        self.rigid.control_dofs_position(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_steps", type=int, default=1000)
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    args.num_steps = 1 if "PYTEST_VERSION" in os.environ else args.num_steps
    args.vis = False if "PYTEST_VERSION" in os.environ else args.vis

    gs.init(backend=gs.cpu)

    dt = 0.002
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
            # enable_interaction=True,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=dt,
            # constraint_solver=gs.constraint_solver.Newton,
            gravity=(0, 0, -9.8),
            enable_collision=True,
            enable_joint_limit=True,
            max_collision_pairs=1000,
        ),
        show_viewer=args.vis,
    )

    # Download asset from HuggingFace
    asset_path = snapshot_download(
        repo_type="dataset",
        repo_id="Genesis-Intelligence/assets",
        revision="main",
        allow_patterns="usd/Lightwheel_Kitchen001/Kitchen001/*",
        max_workers=1,
    )

    entities = scene.add_stage(
        morph=gs.morphs.USDMorph(
            file=f"{asset_path}/usd/Lightwheel_Kitchen001/Kitchen001/Kitchen001.usd",
            # convexify=False,  # turn it off to use the original mesh and sdf collision detection
        ),
        # vis_mode="collision",
        # visualize_contact=True,
    )

    scene.build()

    joint_animator = AutoJointAnimator(scene)

    for _ in range(args.num_steps):
        joint_animator.animate(scene)
        scene.step()


if __name__ == "__main__":
    main()
