"""
Test for external_articulation coupling with simple prismatic joint.
Two cubes connected by a single prismatic joint (sliding motion).
"""

import genesis as gs
import numpy as np


def main():
    # Initialize Genesis
    gs.init(backend=gs.gpu, logging_level="info", performance_mode=True)

    # Create scene
    dt = 0.1

    # Configure IPC coupler with external_articulation strategy
    coupler_options = gs.options.IPCCouplerOptions(
        dt=dt,
        gravity=(0.0, 0.0, -9.8),
        coupling_strategy="external_articulation",
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
            max_FPS=60,
        ),
        show_viewer=True,
    )

    # Add ground plane
    scene.add_entity(gs.morphs.Plane())

    # Create simple two-cube robot with prismatic joint
    robot = scene.add_entity(
        gs.morphs.URDF(file="urdf/simple/two_cube_prismatic.urdf", pos=(0.0, 0.0, 0.5), fixed=True)
    )

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

    # Phase 1: Hold at zero position (settle)
    print("\n=== Phase 1: Settling at zero (50 steps) ===")
    target_zero = np.zeros(robot.n_dofs)
    for i in range(50):
        robot.control_dofs_position(target_zero)
        scene.step()

        if i % 10 == 0:
            current_qpos = robot.get_qpos().cpu().numpy()
            print(f"Step {i:3d}: qpos = {current_qpos}")

    # Phase 2: Oscillating motion (back and forth sliding)
    print("\n=== Phase 2: Oscillating motion (prismatic joint) ===")

    # Parameters for oscillation
    amplitude = 0.25  # meters for prismatic joint (within ±0.3 limit)
    period_time = 2.0  # seconds per cycle
    period = int(period_time / dt)  # steps per cycle

    step = 0
    print(f"\nStarting oscillation:")
    print(f"  Prismatic joint: ±{amplitude} m")
    print(f"  Period: {period} steps\n")

    while scene.viewer.is_alive():
        phase = 2.0 * np.pi * step / period

        target = np.zeros(robot.n_dofs)
        if robot.n_dofs >= 1:
            target[0] = amplitude * np.sin(phase)

        robot.control_dofs_position(target)
        scene.step()

        # Print status every 50 steps
        if step % 50 == 0 and step > 0:
            current_qpos = robot.get_qpos().cpu().numpy()
            print(
                f"Step {step:4d}: target = {target[0]:+.4f} m, actual = {current_qpos[0]:+.6f} m, error = {abs(target[0] - current_qpos[0]):.6f} m"
            )

        step += 1


if __name__ == "__main__":
    main()
