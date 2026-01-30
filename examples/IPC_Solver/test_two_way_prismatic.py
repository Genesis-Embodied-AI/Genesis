"""
Test two-way coupling with prismatic-like motion.

This example demonstrates two-way coupling between IPC and Genesis for
rigid bodies with oscillating motion (simulating prismatic joint behavior).
"""

import argparse
import logging

import numpy as np

import genesis as gs


def main():
    gs.init(backend=gs.gpu, logging_level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("--vis_ipc", action="store_true", default=False)
    args = parser.parse_args()

    dt = 1e-2
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=(0.0, 0.0, -9.8)),
        coupler_options=gs.options.IPCCouplerOptions(
            dt=dt,
            gravity=(0.0, 0.0, -9.8),
            contact_d_hat=0.01,
            contact_friction_mu=0.3,
            IPC_self_contact=False,
            two_way_coupling=True,
            disable_genesis_contact=True,
            enable_ipc_gui=args.vis_ipc,
        ),
        show_viewer=args.vis,
    )

    # Add ground plane
    scene.add_entity(gs.morphs.Plane())

    # Add simple two-cube robot with prismatic joint
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/simple/two_cube_prismatic.urdf",
            pos=(0, 0, 0.2),
            fixed=True,
        ),
    )

    # Set two-way coupling for the robot
    scene.sim.coupler.set_entity_coupling_type(
        entity=robot,
        coupling_type="two_way_soft_constraint",
    )

    gs.logger.info("Building scene...")
    scene.build(n_envs=0)
    gs.logger.info("Scene built successfully!")

    # Run simulation with oscillating vertical motion on slider
    max_steps = 100
    omega = 2.0 * np.pi  # 1 Hz oscillation

    print("\nRunning simulation...")
    for i in range(max_steps):
        t = i * dt
        # Apply sinusoidal target position to prismatic joint
        target_qpos = 0.15 + 0.1 * np.sin(omega * t)  # Oscillate between 0.05 and 0.25
        robot.set_dofs_position([target_qpos], zero_velocity=False)

        scene.step()

        # After some warmup steps, check transform consistency
        if i > 50 and i % 10 == 0:
            # Get transforms from abd_data_by_link for slider
            link_idx = 1  # slider link (moving part)
            env_idx = 0
            if (
                hasattr(scene.sim.coupler, "abd_data_by_link")
                and link_idx in scene.sim.coupler.abd_data_by_link
                and env_idx in scene.sim.coupler.abd_data_by_link[link_idx]
            ):
                abd_data = scene.sim.coupler.abd_data_by_link[link_idx][env_idx]
                genesis_transform = abd_data["aim_transform"]
                ipc_transform = abd_data["transform"]

                if genesis_transform is not None and ipc_transform is not None:
                    genesis_pos = genesis_transform[:3, 3]
                    ipc_pos = ipc_transform[:3, 3]

                    # Compare positions (mainly check z-axis for prismatic)
                    pos_diff = np.linalg.norm(genesis_pos - ipc_pos)
                    print(f"Step {i}: Position difference = {pos_diff:.6f}")
                    if pos_diff >= 0.001:
                        gs.logger.warning(f"Position difference too large: {pos_diff}")

    print("\nSimulation completed!")


if __name__ == "__main__":
    main()
