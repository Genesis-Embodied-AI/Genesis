import argparse

import numpy as np
import torch

import genesis as gs
from genesis.utils import geom as gu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    viewer_options = gs.options.ViewerOptions(
        camera_pos=(0, -3.5, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        max_FPS=60,
    )

    scene = gs.Scene(
        viewer_options=viewer_options,
        show_viewer=args.vis,
    )

    scene.add_entity(morph=gs.morphs.Plane())

    robot = scene.add_entity(
        morph=gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        material=gs.materials.Rigid(),
    )

    scene.build()

    link4 = robot.get_link("link4")
    right_finger = robot.get_link("right_finger")
    hand_link = robot.get_link("hand")

    local_point_offsets = torch.tensor(
        [[0.05, 0.0, 0.1], [0.02, 0.0, 0.01], [0.08, 0.0, 0.05], [0.0, 0.03, 0.08]], dtype=gs.tc_float, device=gs.device
    )

    point_links_idx_local = torch.tensor(
        [link4.idx_local, right_finger.idx_local, hand_link.idx_local, hand_link.idx_local],
        dtype=gs.tc_int,
        device=gs.device,
    )

    all_points_info = [
        {"link_name": "link4", "point_idx": 0},
        {"link_name": "right_finger", "point_idx": 1},
        {"link_name": "hand", "point_idx": 2},
        {"link_name": "hand", "point_idx": 3},
    ]

    center = torch.tensor([0.5, 0.0, 0.4])
    radius = 0.15
    frequency = 0.3

    point_colors = [
        (0, 0, 1, 0.8),
        (1, 0, 0, 0.6),
        (0, 1, 0, 0.6),
        (1, 0, 1, 0.6),
    ]

    n_steps = 1000
    for step in range(n_steps):
        t = step * scene.dt

        target_pos = center + torch.tensor(
            [radius * np.cos(2 * np.pi * frequency * t), radius * np.sin(2 * np.pi * frequency * t), 0.0]
        )

        qpos_target = robot.inverse_kinematics(
            link=hand_link,
            pos=target_pos,
            max_solver_iters=20,
            damping=0.1,
            pos_tol=1e-3,
        )
        robot.control_dofs_position(qpos_target)

        scene.step()

        world_positions, world_velocities = robot.get_link_point_position_velocity(
            local_offsets=local_point_offsets, links_idx_local=point_links_idx_local
        )

        scene.clear_debug_objects()

        for i, point_info in enumerate(all_points_info):
            point_idx = point_info["point_idx"]

            point_pos = world_positions[point_idx].numpy()

            scene.draw_debug_sphere(pos=point_pos, radius=0.012, color=point_colors[i % len(point_colors)])

        for i, point_info in enumerate(all_points_info):
            point_idx = point_info["point_idx"]

            point_vel = world_velocities[point_idx]
            point_pos = world_positions[point_idx].numpy()

            scene.draw_debug_arrow(
                pos=point_pos, vec=point_vel.numpy() * 0.3, radius=0.005, color=point_colors[i % len(point_colors)]
            )


if __name__ == "__main__":
    main()
