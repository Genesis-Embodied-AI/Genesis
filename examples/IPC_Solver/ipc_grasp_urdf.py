"""
Smooth grasping test with gradual movement.
Based on ipc_grasp.py but uses smooth interpolation to target positions.
"""

import genesis as gs
import logging
import argparse
import numpy as np


def main():
    gs.init(backend=gs.gpu, logging_level=logging.DEBUG, performance_mode=True)

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
            disable_ipc_ground_contact=True,
            disable_ipc_logging=True,
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

    scene.add_entity(gs.morphs.Plane())

    # Add Franka robot
    # franka = scene.add_entity(
    #     gs.morphs.MJCF(file="xml/franka_emika_panda/panda_non_overlap.xml"
    #                    ),

    # )
    franka = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/panda_bullet/panda.urdf",
            fixed=True,
        ),
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

    # DOF indices
    # With fixed=True, qpos structure: [0:7] 7 arm revolutes, [7:9] 2 finger prismatic
    motors_dof = np.arange(7)  # Arm joint DOFs
    fingers_dof = np.arange(7, 9)  # Finger DOFs

    # Phase 0: Move to initial configuration
    print("\n=== Phase 0: Moving to initial configuration ===")

    # Increase control gains
    current_kp = franka.get_dofs_kp()
    current_kp[fingers_dof] = current_kp[fingers_dof] * 10
    current_kp[motors_dof] = current_kp[motors_dof] * 10
    franka.set_dofs_kp(current_kp)

    current_kv = franka.get_dofs_kv()
    current_kv[fingers_dof] = current_kv[fingers_dof] * 10
    current_kv[motors_dof] = current_kv[motors_dof] * 10
    franka.set_dofs_kv(current_kv)

    # Get current qpos (home position)
    qpos_home = franka.get_qpos().cpu().numpy()
    print(f"Home qpos: {qpos_home}")

    # Target initial configuration (7 arm joints + 2 fingers)
    qpos_init = np.array([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04])
    print(f"Target init qpos: {qpos_init}")

    # Directly control to initial configuration
    n_steps_init = int(10.0 / dt)  # 5 seconds
    for i in range(n_steps_init):
        franka.control_dofs_position(qpos_init[:-2], motors_dof)
        franka.control_dofs_position(qpos_init[-2:], fingers_dof)
        scene.step()

        if i % 50 == 0:
            current_qpos = franka.get_qpos().cpu().numpy()
            print(f"Step {i}/{n_steps_init}: qpos = {current_qpos}")

    # # Phase 1: Move to pre-grasp position
    # print("\n=== Phase 1: Moving to pre-grasp position ===")
    end_effector = franka.get_link("panda_link7")  # Last link before fingers

    # # Get initial end-effector position
    # ee_pos_init = end_effector.get_pos().cpu().numpy()
    # ee_quat_init = end_effector.get_quat().cpu().numpy()
    # print(f"Initial EE position: {ee_pos_init}")
    # print(f"Initial EE quaternion: {ee_quat_init}")

    # # Target pre-grasp position
    # target_pos = np.array([0.6054622, 0.01968566, 0.07157667])
    # target_quat = np.array([0.01685536,  0.91546345, -0.37735498, -0.1387286])

    # # Compute IK for target
    # qpos_target = franka.inverse_kinematics(
    #     link=end_effector,
    #     pos=target_pos,
    #     quat=target_quat,
    # )

    # # Directly control to target position
    # n_steps_approach = int(3.0 / dt)  # 3 seconds
    # for i in range(n_steps_approach):
    #     franka.control_dofs_position(qpos_target[:-2], motors_dof)
    #     scene.step()

    #     if i % 50 == 0:
    #         current_pos = end_effector.get_pos().cpu().numpy()
    #         print(f"Step {i}/{n_steps_approach}: EE at {current_pos}")

    # Save pre-grasp joint configuration
    qpos_pregrasp = qpos_init

    # Phase 3: Close fingers to grasp
    print("\n=== Phase 3: Closing fingers to grasp ===")

    finger_pos_closed = 0.0
    n_steps_grasp = int(0.5 / dt)  # 0.5 seconds to close fingers

    # Directly control to closed position
    for i in range(n_steps_grasp):
        franka.control_dofs_position(qpos_pregrasp[:-2], motors_dof)
        franka.control_dofs_position(np.array([finger_pos_closed, finger_pos_closed]), fingers_dof)
        scene.step()

        if i % 25 == 0:
            current_finger_pos = franka.get_qpos().cpu().numpy()[7:9]
            print(f"Step {i}/{n_steps_grasp}: Fingers at {current_finger_pos}")

    # Phase 4: Lift the object
    print("\n=== Phase 4: Lifting object ===")

    lift_end_pos = np.array([0.6054622, 0.01968566, 0.07157667 + 0.3])  # Lift 30 cm up
    target_quat = np.array([0.01685536, 0.91546345, -0.37735498, -0.1387286])
    n_steps_lift = int(1.5 / dt)  # 1.5 seconds to lift

    # Compute IK for lifted position
    qpos_lift = franka.inverse_kinematics(
        link=end_effector,
        pos=lift_end_pos,
        quat=target_quat,
    )

    # Directly control to lifted position
    for i in range(n_steps_lift):
        franka.control_dofs_position(qpos_lift[:-2], motors_dof)
        franka.control_dofs_position(np.array([finger_pos_closed, finger_pos_closed]), fingers_dof)
        scene.step()

        if i % 50 == 0:
            current_pos = end_effector.get_pos().cpu().numpy()
            print(f"Step {i}/{n_steps_lift}: EE at {current_pos}")

    # Hold lifted position
    print("\n=== Phase 5: Holding lifted position ===")
    print("Grasp complete! Object should be lifted.")

    while scene.viewer.is_alive():
        franka.control_dofs_position(qpos_lift[:-2], motors_dof)
        franka.control_dofs_position(np.array([finger_pos_closed, finger_pos_closed]), fingers_dof)
        scene.step()


if __name__ == "__main__":
    main()
