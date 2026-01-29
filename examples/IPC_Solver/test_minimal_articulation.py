"""
Minimal test for external_articulation coupling - single joint.
Uses the simplest robot to debug the coupling mechanism.
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
        contact_friction_mu=0.5,
        IPC_self_contact=False,
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
        ),
        show_viewer=True,
    )

    # Add ground plane
    scene.add_entity(gs.morphs.Plane())

    # Create simple two-cube robot with one joint
    robot = scene.add_entity(gs.morphs.URDF(file="urdf/simple/two_cube_joint.urdf", pos=(0.0, 0.0, 0.5), fixed=True))

    # Build scene
    print("Building scene...")
    scene.build()
    print("Scene built!")

    # Get initial qpos
    print(f"\n=== Robot Info ===")
    print(f"robot.n_dofs = {robot.n_dofs}")
    print(f"robot.n_links = {robot.n_links}")
    qpos_init = robot.get_qpos().cpu().numpy()
    print(f"qpos shape: {qpos_init.shape}")
    print(f"qpos values: {qpos_init}")
    print(f"Number of joints: {len([j for link_joints in robot._joints for j in link_joints])}")

    # Phase 1: Hold at zero position (settle)
    print("\n=== Phase 1: Settling at zero (10 steps) ===")
    target_zero = np.zeros(robot.n_dofs)
    for i in range(10):
        robot.control_dofs_position(target_zero)
        scene.step()

        if i == 0:
            print(f"\n=== After first step ===")
            qpos_after = robot.get_qpos().cpu().numpy()
            print(f"qpos after step: {qpos_after}")
            print(f"Has NaN: {np.any(np.isnan(qpos_after))}")

        if i % 10 == 0:
            current_qpos = robot.get_qpos().cpu().numpy()
            print(f"Step {i:3d}: qpos = {current_qpos}")

    # Phase 2: Oscillating motion (back and forth)
    print("\n=== Phase 2: Oscillating motion ===")

    # Parameters for oscillation
    amplitude = 1.5  # radians for the joint
    period = 200  # steps per cycle
    total_steps = 5000

    for i in range(total_steps):
        # Calculate sinusoidal target for the joint
        phase = 2.0 * np.pi * i / period

        target = np.zeros(robot.n_dofs)
        if robot.n_dofs >= 1:
            target[0] = amplitude * np.sin(phase)

        robot.control_dofs_position(target)
        scene.step()

        if i % 50 == 0:
            current_qpos = robot.get_qpos().cpu().numpy()
            error = np.linalg.norm(current_qpos - target)
            print(f"Step {i:3d}: target = {target[0]:.3f}, qpos = {current_qpos[0]:.3f}, error = {error:.4f}")

    # Final result
    final_qpos = robot.get_qpos().cpu().numpy()
    print(f"\n=== Final Results ===")
    print(f"Target qpos:  {target_zero}")
    print(f"Final qpos:   {final_qpos}")
    print(f"Final error:  {np.linalg.norm(final_qpos - target_zero):.6f}")

    # Keep viewer open with continued oscillation
    print("\nTest complete! Continuing oscillation... Close viewer to exit.")
    step = 0
    while scene.viewer.is_alive():
        phase = 2.0 * np.pi * step / period

        target = np.zeros(robot.n_dofs)
        if robot.n_dofs >= 1:
            target[0] = amplitude * np.sin(phase)

        robot.control_dofs_position(target)
        scene.step()
        step += 1


if __name__ == "__main__":
    main()
