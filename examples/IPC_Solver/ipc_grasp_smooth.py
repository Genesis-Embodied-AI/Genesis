"""
Smooth grasping test with gradual movement.
Based on ipc_grasp.py but uses smooth interpolation to target positions.
"""

import genesis as gs
import logging
import argparse
import numpy as np


def main():
    gs.init(backend=gs.gpu, logging_level=logging.INFO, performance_mode=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ipc", action="store_true", default=False)
    parser.add_argument("--vis_ipc", action="store_true", default=False)
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    dt = 1e-2

    coupler_options = (
        gs.options.IPCCouplerOptions(
            dt=dt,
            gravity=(0.0, 0.0, -9.8),
            coupling_strategy="external_articulation",
            contact_friction_mu=0.8,
            IPC_self_contact=False,
            disable_ipc_logging=False,
            newton_semi_implicit_enable=False,
            enable_ipc_gui=args.vis_ipc,
        )
        if args.ipc
        else None
    )
    args.vis = args.vis or args.vis_ipc
    rigid_options = gs.options.RigidOptions(
        enable_collision=False,  # Disable rigid collision when using IPC
    )
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=(0.0, 0.0, -9.8)),
        rigid_options=rigid_options,
        coupler_options=coupler_options,
        show_viewer=args.vis,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, -1.0, 1.5),
            camera_lookat=(0.5, 0.0, 0.2),
            camera_fov=40,
        ),
    )

    # scene.add_entity(gs.morphs.Plane())

    # Add Franka robot
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda_non_overlap.xml"),
    )

    # Add object to grasp (FEM if using IPC, Rigid otherwise)
    material = (
        gs.materials.FEM.Elastic(E=5.0e4, nu=0.45, rho=1000.0, model="stable_neohookean")
        if args.ipc
        else gs.materials.Rigid()
    )
    cube = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.65, 0.0, 0.03), size=(0.05, 0.05, 0.05)),
        material=material,
        surface=gs.surfaces.Plastic(color=(0.2, 0.8, 0.2, 0.5)),
    )

    scene.build()
    print("Scene built successfully!")

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)

    # Phase 0: Smoothly move from home position to initial configuration
    print("\n=== Phase 0: Moving to initial configuration ===")

    # Get current qpos (home position)
    qpos_home = franka.get_qpos().cpu().numpy()
    print(f"Home qpos: {qpos_home}")

    # Target initial configuration
    qpos_init = np.array([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04])
    print(f"Target init qpos: {qpos_init}")

    # Smoothly interpolate to initial configuration
    n_steps_init = int(2.0 / dt)  # 2 seconds to reach initial config
    for i in range(n_steps_init):
        alpha = i / n_steps_init
        qpos_current = qpos_home * (1 - alpha) + qpos_init * alpha

        franka.control_dofs_position(qpos_current[:-2], motors_dof)
        franka.control_dofs_position(qpos_current[-2:], fingers_dof)
        scene.step()

        if i % 50 == 0:
            print(f"Step {i}/{n_steps_init}: Moving to initial config")

    # Hold initial position briefly
    for i in range(int(0.5 / dt)):
        franka.control_dofs_position(qpos_init[:-2], motors_dof)
        franka.control_dofs_position(qpos_init[-2:], fingers_dof)
        scene.step()

    # Phase 1: Smoothly move to pre-grasp position
    print("\n=== Phase 1: Moving to pre-grasp position ===")
    end_effector = franka.get_link("hand")

    # Get initial end-effector position
    ee_pos_init = end_effector.get_pos().cpu().numpy()
    ee_quat_init = end_effector.get_quat().cpu().numpy()
    print(f"Initial EE position: {ee_pos_init}")

    # Target pre-grasp position (above the cube)
    target_pos = np.array([0.65, 0.0, 0.135])
    target_quat = np.array([0, 1, 0, 0])

    n_steps_approach = int(1.0 / dt)  # 1 second to approach
    for i in range(n_steps_approach):
        alpha = i / n_steps_approach  # Interpolation factor (0 to 1)

        # Linear interpolation for position
        current_target_pos = ee_pos_init * (1 - alpha) + target_pos * alpha

        # Compute IK for current target
        qpos_target = franka.inverse_kinematics(
            link=end_effector,
            pos=current_target_pos,
            quat=target_quat,
        )

        # Control motors (excluding fingers)
        franka.control_dofs_position(qpos_target[:-2], motors_dof)
        scene.step()

        if i % 50 == 0:
            current_pos = end_effector.get_pos().cpu().numpy()
            print(f"Step {i}/{n_steps_approach}: EE at {current_pos}, target: {current_target_pos}")

    # Hold position
    print("\n=== Phase 2: Holding pre-grasp position ===")
    qpos_pregrasp = franka.get_qpos().cpu().numpy()
    for i in range(int(0.2 / dt)):
        franka.control_dofs_position(qpos_pregrasp[:-2], motors_dof)
        scene.step()

    # Phase 3: Smoothly close fingers to grasp
    print("\n=== Phase 3: Closing fingers to grasp ===")

    # Increase finger stiffness for grasping
    current_kp = franka.get_dofs_kp()
    current_kp[fingers_dof] = current_kp[fingers_dof] * 10
    franka.set_dofs_kp(current_kp)

    finger_pos_open = 0.04
    finger_pos_closed = 0.0
    n_steps_grasp = int(0.5 / dt)  # 0.5 seconds to close fingers

    for i in range(n_steps_grasp):
        alpha = i / n_steps_grasp
        finger_pos = finger_pos_open * (1 - alpha) + finger_pos_closed * alpha

        franka.control_dofs_position(qpos_pregrasp[:-2], motors_dof)
        franka.control_dofs_position(np.array([finger_pos, finger_pos]), fingers_dof)
        scene.step()

        if i % 25 == 0:
            current_finger_pos = franka.get_qpos().cpu().numpy()[7:9]
            print(f"Step {i}/{n_steps_grasp}: Fingers at {current_finger_pos}, target: {finger_pos}")

    # Hold grasp
    print("\n=== Phase 4: Holding grasp ===")
    for i in range(int(0.2 / dt)):
        franka.control_dofs_position(qpos_pregrasp[:-2], motors_dof)
        franka.control_dofs_position(np.array([finger_pos_closed, finger_pos_closed]), fingers_dof)
        scene.step()

    # Phase 5: Smoothly lift the object
    print("\n=== Phase 5: Lifting object ===")

    lift_start_pos = target_pos
    lift_end_pos = np.array([0.65, 0.0, 0.4])
    n_steps_lift = int(1.5 / dt)  # 1.5 seconds to lift

    for i in range(n_steps_lift):
        alpha = i / n_steps_lift
        current_target_pos = lift_start_pos * (1 - alpha) + lift_end_pos * alpha

        qpos_target = franka.inverse_kinematics(
            link=end_effector,
            pos=current_target_pos,
            quat=target_quat,
        )

        franka.control_dofs_position(qpos_target[:-2], motors_dof)
        franka.control_dofs_position(np.array([finger_pos_closed, finger_pos_closed]), fingers_dof)
        scene.step()

        if i % 50 == 0:
            current_pos = end_effector.get_pos().cpu().numpy()
            print(f"Step {i}/{n_steps_lift}: EE at {current_pos}, target: {current_target_pos}")

    # Hold lifted position
    print("\n=== Phase 6: Holding lifted position ===")
    print("Grasp complete! Object should be lifted.")

    while scene.viewer.is_alive():
        qpos_final = franka.get_qpos().cpu().numpy()
        franka.control_dofs_position(qpos_final[:-2], motors_dof)
        franka.control_dofs_position(np.array([finger_pos_closed, finger_pos_closed]), fingers_dof)
        scene.step()


if __name__ == "__main__":
    main()
