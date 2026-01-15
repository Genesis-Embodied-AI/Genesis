"""
Genesis IPC Cloth Band Twisting Simulation

This example demonstrates:
- Four rigid cube grippers (two on each end of the band)
- Band using Cloth material with IPC
- Counter-rotation animation to twist the band
"""

import numpy as np
import genesis as gs
import logging
import argparse

from huggingface_hub import snapshot_download


def main():
    gs.init(backend=gs.gpu, logging_level=logging.INFO, performance_mode=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("--vis_ipc", action="store_true", default=False)
    args = parser.parse_args()

    # Simulation parameters
    dt = 0.01
    d_hat = 0.001

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=(0.0, 0.0, 0)),
        coupler_options=gs.options.IPCCouplerOptions(
            dt=dt,
            gravity=(0.0, 0.0, 0),
            contact_d_hat=d_hat,  # Contact barrier distance
            ipc_constraint_strength=(1, 1),  # (translation, rotation) strength ratios,
            contact_friction_mu=0.5,  # Higher friction to grip the band
            IPC_self_contact=False,
            two_way_coupling=True,
            enable_ipc_gui=args.vis_ipc,
        ),
        show_viewer=args.vis or args.vis_ipc,
    )

    # Band dimensions
    band_length = 1.2
    band_width = 0.13
    band_height = 0.43

    # Cube gripper parameters
    cube_size = 0.03  # 3cm cubes
    grip_offset = 0.02  # Distance from band edge to cube center

    asset_path = snapshot_download(
        repo_type="dataset",
        repo_id="Genesis-Intelligence/assets",
        revision="72b04f7125e21df1bebd54a7f7b39d1cd832331c",
        allow_patterns=["band.obj", "band_gripper.urdf"],
        max_workers=1,
    )
    # Add band (cloth with IPC) - positioned horizontally along Y axis
    band = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/band.obj",
        ),
        material=gs.materials.FEM.Cloth(
            E=1e5,  # 5.0 MPa Young's modulus
            nu=0.45,  # Poisson's ratio
            rho=100,  # 0.1e3 kg/m^3 density
            thickness=0.002,  # 2mm thickness
            bending_stiffness=50.0,
        ),
        surface=gs.surfaces.Plastic(color=(0.8, 0.3, 0.3, 1.0), double_sided=True),
    )

    # Add gripper system from URDF (4 cubes with floating joints)
    gripper = scene.add_entity(
        morph=gs.morphs.URDF(
            file=f"{asset_path}/band_gripper.urdf",
            fixed=True,  # Fix the base link
        ),
    )

    gs.logger.info("Building scene...")
    scene.build(n_envs=1)
    gs.logger.info("Scene built successfully!")

    left_top_dofs = np.arange(0, 6)
    left_bottom_dofs = np.arange(6, 12)
    right_top_dofs = np.arange(12, 18)
    right_bottom_dofs = np.arange(18, 24)

    # PD gains for position and orientation control
    pos_kp = 50.0  # Position stiffness
    pos_kv = 1.0  # Position damping

    # Set control gains for all gripper DOFs
    n_dofs = gripper.n_dofs
    kp_gains = np.ones(n_dofs) * pos_kp
    kv_gains = np.ones(n_dofs) * pos_kv

    gripper.set_dofs_kp(kp_gains)
    gripper.set_dofs_kv(kv_gains)

    print(f"Gripper has {n_dofs} DOFs")

    # Calculate gripper positions (band extends along Y axis)
    left_y = -0.6  # Left end at -0.6 (matching URDF initial position)
    right_y = 0.6  # Right end at +0.6 (matching URDF initial position)

    # Initial gripper positions from URDF
    initial_z_top = 0.2  # Top grippers start at z=0.2
    initial_z_bottom = -0.2  # Bottom grippers start at z=-0.2

    # Target positions for tight grip
    final_z_top = -0.05  # Move down to -0.05 (slight overlap)
    final_z_bottom = 0.05  # Move up to 0.05 (slight overlap)

    # Simulation parameters for animation
    total_time = 10.0  # total simulation time
    total_frames = int(total_time / dt)

    # Phase 1: Gripping (move cubes to grip the band)
    grip_duration = 2.0  # 2 seconds to slowly grip
    grip_frames = int(grip_duration / dt)

    # Phase 2: Twisting (rotate the grippers)
    twist_start_time = grip_duration
    twist_duration = 6.0
    twist_frames = int(twist_duration / dt)
    angular_vel = np.deg2rad(180)  # 180 degrees per second

    print(f"\nStarting simulation for {total_frames} frames...")
    print(f"Phase 1 - Gripping: 0-{grip_duration}s")
    print(f"Phase 2 - Twisting: {twist_start_time}-{twist_start_time + twist_duration}s")

    for frame in range(total_frames):
        total_t = frame * dt

        # Phase 1: Gripping - move cubes closer to grip the band
        if total_t < grip_duration:
            progress = total_t / grip_duration
            # Smooth easing function
            ease = 0.5 - 0.5 * np.cos(progress * np.pi)

            # Slowly move grippers from initial position to final gripping position
            current_z_top = initial_z_top + (final_z_top - initial_z_top) * ease
            current_z_bottom = initial_z_bottom + (final_z_bottom - initial_z_bottom) * ease

            # No rotation in phase 1
            zero_rotation = np.array([0.0, 0.0, 0.0])

            # Prepare control targets for all 4 grippers: [x, y, z, rx, ry, rz]
            # Left end grippers (at -Y)
            left_top_target = np.array(
                [0.0, left_y, current_z_top, zero_rotation[0], zero_rotation[1], zero_rotation[2]]
            )

            left_bottom_target = np.array(
                [0.0, left_y, current_z_bottom, zero_rotation[0], zero_rotation[1], zero_rotation[2]]
            )

            # Right end grippers (at +Y)
            right_top_target = np.array(
                [0.0, right_y, current_z_top, zero_rotation[0], zero_rotation[1], zero_rotation[2]]
            )

            right_bottom_target = np.array(
                [0.0, right_y, current_z_bottom, zero_rotation[0], zero_rotation[1], zero_rotation[2]]
            )

            # Control each gripper using DOF indices
            gripper.control_dofs_position(left_top_target, left_top_dofs)
            gripper.control_dofs_position(left_bottom_target, left_bottom_dofs)
            gripper.control_dofs_position(right_top_target, right_top_dofs)
            gripper.control_dofs_position(right_bottom_target, right_bottom_dofs)

        # Phase 2: Twisting - rotate grippers in opposite directions around Y axis
        elif total_t < twist_start_time + twist_duration:
            twist_time = total_t - twist_start_time
            angle = angular_vel * twist_time

            # Maintain tight grip position (final positions from phase 1)
            current_z_top = final_z_top
            current_z_bottom = final_z_bottom

            # Calculate rotated positions for gripper pairs
            # Each pair rotates as a rigid body around Y axis at their Y position

            # Left pair rotates clockwise (positive angle around Y)
            cos_left = np.cos(angle)
            sin_left = np.sin(angle)
            # Grippers rotate around the center (0, left_y, 0)
            left_top_x = current_z_top * sin_left
            left_top_z = current_z_top * cos_left

            left_bottom_x = current_z_bottom * sin_left
            left_bottom_z = current_z_bottom * cos_left

            left_top_target = np.array(
                [left_top_x, left_y, left_top_z, 0.0, angle, 0.0]  # rx, ry, rz - rotate around Y
            )

            left_bottom_target = np.array([left_bottom_x, left_y, left_bottom_z, 0.0, angle, 0.0])

            # Right pair rotates counter-clockwise (negative angle around Y)
            cos_right = np.cos(-angle)
            sin_right = np.sin(-angle)

            right_top_x = current_z_top * sin_right
            right_top_z = current_z_top * cos_right

            right_bottom_x = current_z_bottom * sin_right
            right_bottom_z = current_z_bottom * cos_right

            right_top_target = np.array([right_top_x, right_y, right_top_z, 0.0, -angle, 0.0])

            right_bottom_target = np.array([right_bottom_x, right_y, right_bottom_z, 0.0, -angle, 0.0])

            # Control each gripper using DOF indices
            gripper.control_dofs_position(left_top_target, left_top_dofs)
            gripper.control_dofs_position(left_bottom_target, left_bottom_dofs)
            gripper.control_dofs_position(right_top_target, right_top_dofs)
            gripper.control_dofs_position(right_bottom_target, right_bottom_dofs)

        # Step simulation
        scene.step()

        # Progress reporting
        if frame % 100 == 0:
            if total_t < grip_duration:
                phase = "Gripping"
            elif total_t < twist_start_time + twist_duration:
                phase = "Twisting"
            else:
                phase = "Settling"
            print(f"  Frame {frame}/{total_frames} (t={total_t:.2f}s) - {phase}")


if __name__ == "__main__":
    main()
