import argparse
import os

import numpy as np
from huggingface_hub import snapshot_download

import genesis as gs
from genesis.utils.misc import ti_to_numpy


class JointAnimator:
    """
    A simple JointAnimator to animate the joints' positions of the scene.

    It uses the sin function to interpolate between the lower and upper limits of the joints.
    """

    def __init__(self, scene: gs.Scene):
        self.rigid = scene.sim.rigid_solver
        n_dofs = self.rigid.n_dofs
        joint_limits = ti_to_numpy(self.rigid.dofs_info.limit)
        joint_limits = np.clip(joint_limits, -np.pi, np.pi)

        init_positions = self.rigid.get_dofs_position().numpy()

        self.joint_lower = joint_limits[:, 0]
        self.joint_upper = joint_limits[:, 1]

        valid_range_mask = (self.joint_upper - self.joint_lower) > gs.EPS

        normalized_init_pos = np.where(
            valid_range_mask,
            2.0 * (init_positions - self.joint_lower) / (self.joint_upper - self.joint_lower) - 1.0,
            0.0,
        )
        self.init_phase = np.arcsin(normalized_init_pos)

        # hard code some joint dynamics parameters to make the control more sensitive
        self.rigid.set_dofs_frictionloss(np.zeros(n_dofs))
        self.rigid.set_dofs_kp(np.full(n_dofs, 100.0))
        self.rigid.set_dofs_armature(np.zeros(n_dofs))

    def animate(self, scene: gs.Scene):
        """Calculate target positions using sin function to interpolate between lower and upper limits"""
        t = scene.t * scene.dt
        theta = np.pi * t + self.init_phase
        theta = theta % (2 * np.pi)
        sin_values = np.sin(theta)
        normalized = (sin_values + 1.0) / 2.0
        target = self.joint_lower + (self.joint_upper - self.joint_lower) * normalized
        self.rigid.control_dofs_position(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_steps", type=int, default=1)
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    dt = 0.002
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=dt,
            gravity=(0, 0, -9.8),
            enable_collision=True,
            enable_joint_limit=True,
            max_collision_pairs=1000,
        ),
        show_viewer=args.vis,
    )

    asset_path = snapshot_download(
        repo_type="dataset",
        repo_id="Genesis-Intelligence/assets",
        revision="main",
        allow_patterns="usd/Refrigerator055/*",
        max_workers=1,
    )

    entities = scene.add_stage(
        morph=gs.morphs.USD(
            file=f"{asset_path}/usd/Refrigerator055/Refrigerator055.usd",
        ),
        # vis_mode="collision",
        # visualize_contact=True,
    )

    scene.build()

    joint_animator = JointAnimator(scene)

    for _ in range(args.num_steps):
        joint_animator.animate(scene)
        scene.step()


if __name__ == "__main__":
    main()
