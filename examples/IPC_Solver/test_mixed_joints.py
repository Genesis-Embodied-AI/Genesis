"""
Test for external_articulation coupling with mixed joints (revolute + prismatic).
Tests both rotational and translational joint control.
"""

import genesis as gs
import numpy as np


def main():
    # Initialize Genesis
    gs.init(backend=gs.gpu, logging_level="info", performance_mode=True)

    # Create scene
    dt = 0.01

    # Configure IPC coupler with external_articulation strategy
    coupler_options = gs.options.IPCCouplerOptions(
        dt=dt,
        gravity=(0.0, 0.0, -9.8),
        coupling_strategy="external_articulation",
        contact_friction_mu=0.5,
        IPC_self_contact=True,
        enable_ipc_gui=True,
        disable_ipc_logging=False,
        newton_velocity_tol=5e-4,
    )

    # Disable rigid collision when using IPC
    rigid_options = gs.options.RigidOptions(
        enable_collision=False,
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            gravity=(0.0, 0.0, -9.8),
        ),
        rigid_options=rigid_options,
        coupler_options=coupler_options,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0.0, 1.0),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
            max_FPS=60,
        ),
        show_viewer=True,
    )

    # Add ground plane
    scene.add_entity(gs.morphs.Plane())

    # Create robot with mixed joints (revolute + prismatic)
    robot = scene.add_entity(
        gs.morphs.URDF(file="urdf/simple/mixed_joints_robot.urdf", pos=(0.0, 0.0, 0.5), fixed=True)
    )

    # Build scene
    print("Building scene...")
    scene.build()
    print("Scene built!")

    # Get initial qpos
    qpos_init = robot.get_qpos().cpu().numpy()
    print(f"\nInitial qpos: {qpos_init}")
    print(f"Number of DOFs: {robot.n_dofs}")
    print(f"Number of links: {robot.n_links}")
    print(f"Number of joints: {len([j for link_joints in robot._joints for j in link_joints])}")

    # Phase 1: Hold at zero position (settle)
    print("\n=== Phase 1: Settling at zero (50 steps) ===")
    target_zero = np.zeros(robot.n_dofs)
    for i in range(50):
        robot.control_dofs_position(target_zero)
        scene.step()

        if i % 10 == 0:
            current_qpos = robot.get_qpos().cpu().numpy()
            print(f"Step {i:3d}: qpos = {current_qpos}")

    # Phase 2: Oscillating motion (both joints move)
    print("\n=== Phase 2: Oscillating motion (both joints) ===")

    # Parameters for oscillation
    revolute_amplitude = 1  # radians for revolute joint (rotation)
    prismatic_amplitude = 0.2  # meters for prismatic joint (translation)
    period = 250  # steps per cycle

    step = 0
    print(f"\nStarting oscillation:")
    print(f"  Revolute joint: ±{revolute_amplitude} rad (±{np.degrees(revolute_amplitude):.1f}°)")
    print(f"  Prismatic joint: ±{prismatic_amplitude} m")
    print(f"  Period: {period} steps\n")

    while scene.viewer.is_alive():
        phase = 2.0 * np.pi * step / period

        target = np.zeros(robot.n_dofs)
        if robot.n_dofs >= 1:
            # Revolute joint (DOF 0): sinusoidal rotation
            target[0] = revolute_amplitude * np.sin(phase)
        if robot.n_dofs >= 2:
            # Prismatic joint (DOF 1): sinusoidal translation
            target[1] = prismatic_amplitude * np.sin(phase + np.pi / 2)  # 90° phase shift

        robot.control_dofs_position(target)
        scene.step()

        # Print status every 100 steps
        if step % 100 == 0 and step > 0:
            current_qpos = robot.get_qpos().cpu().numpy()
            print(
                f"Step {step:4d}: revolute={current_qpos[0]:+.3f} rad ({np.degrees(current_qpos[0]):+.1f}°), "
                f"prismatic={current_qpos[1]:+.4f} m"
            )

        step += 1


if __name__ == "__main__":
    main()
