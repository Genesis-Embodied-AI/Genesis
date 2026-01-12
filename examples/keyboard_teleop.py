"""
Keyboard Controls:
↑	- Move Forward (North)
↓	- Move Backward (South)
←	- Move Left (West)
→	- Move Right (East)
n	- Move Up
m	- Move Down
j	- Rotate Counterclockwise
k	- Rotate Clockwise
u	- Reset Scene
space	- Press to close gripper, release to open gripper
esc	- Quit

Plus all default viewer controls (press 'i' to see them)
"""

import random

import numpy as np
from scipy.spatial.transform import Rotation as R

import genesis as gs
from genesis.ext.pyrender.interaction.keybindings import KeyAction, Keybind

if __name__ == "__main__":
    ########################## init ##########################
    gs.init(precision="32", logging_level="info", backend=gs.cpu)
    np.set_printoptions(precision=7, suppress=True)

    ########################## create a scene ##########################
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            substeps=4,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=True,
            enable_collision=True,
            gravity=(0, 0, -9.8),
            box_box_detection=True,
            constraint_timeconst=0.01,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0.0, 0.7),
            camera_lookat=(0.2, 0.0, 0.1),
            camera_fov=50,
            max_FPS=60,
        ),
        show_viewer=True,
        show_FPS=False,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )

    robot = scene.add_entity(
        material=gs.materials.Rigid(gravity_compensation=1),
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            euler=(0, 0, 0),
        ),
    )

    cube = scene.add_entity(
        material=gs.materials.Rigid(rho=300),
        morph=gs.morphs.Box(
            pos=(0.5, 0.0, 0.07),
            size=(0.04, 0.04, 0.04),
        ),
        surface=gs.surfaces.Default(color=(0.5, 1, 0.5)),
    )

    target = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            scale=0.15,
            collision=False,
        ),
        surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
    )

    ########################## build ##########################
    scene.build()

    # Initialize robot control state
    robot_init_pos = np.array([0.5, 0, 0.55])
    robot_init_R = R.from_euler("y", np.pi)

    # Get DOF indices
    n_dofs = robot.n_dofs
    motors_dof = np.arange(n_dofs - 2)
    fingers_dof = np.arange(n_dofs - 2, n_dofs)
    ee_link = robot.get_link("hand")

    # Initialize target pose
    target_pos = robot_init_pos.copy()
    target_R = [robot_init_R]  # Use list to make it mutable in closures

    # Control parameters
    dpos = 0.002
    drot = 0.01

    # Helper function to reset robot
    def reset_robot():
        """Reset robot and cube to initial positions."""
        target_pos[:] = robot_init_pos.copy()
        target_R[0] = robot_init_R
        target_quat = target_R[0].as_quat(scalar_first=True)
        target.set_qpos(np.concatenate([target_pos, target_quat]))
        q = robot.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat)
        robot.set_qpos(q[:-2], motors_dof)

        # Randomize cube position
        cube.set_pos((random.uniform(0.2, 0.4), random.uniform(-0.2, 0.2), 0.05))
        cube.set_quat(R.from_euler("z", random.uniform(0, np.pi * 2)).as_quat(scalar_first=True))

    # Initialize robot pose
    reset_robot()

    # Robot teleoperation callback functions
    def move_forward():
        target_pos[0] -= dpos

    def move_backward():
        target_pos[0] += dpos

    def move_left():
        target_pos[1] -= dpos

    def move_right():
        target_pos[1] += dpos

    def move_up():
        target_pos[2] += dpos

    def move_down():
        target_pos[2] -= dpos

    def rotate_ccw():
        target_R[0] = R.from_euler("z", drot) * target_R[0]

    def rotate_cw():
        target_R[0] = R.from_euler("z", -drot) * target_R[0]

    def close_gripper():
        robot.control_dofs_force(np.array([-1.0, -1.0]), fingers_dof)

    def open_gripper():
        robot.control_dofs_force(np.array([1.0, 1.0]), fingers_dof)

    # Register robot teleoperation keybindings
    from pyglet.window import key

    scene.viewer.register_keybinds(
        Keybind(key_code=key.UP, key_action=KeyAction.HOLD, name="move_forward", callback=move_forward),
        Keybind(key_code=key.DOWN, key_action=KeyAction.HOLD, name="move_backward", callback=move_backward),
        Keybind(key_code=key.LEFT, key_action=KeyAction.HOLD, name="move_left", callback=move_left),
        Keybind(key_code=key.RIGHT, key_action=KeyAction.HOLD, name="move_right", callback=move_right),
        Keybind(key_code=key.N, key_action=KeyAction.HOLD, name="move_up", callback=move_up),
        Keybind(key_code=key.M, key_action=KeyAction.HOLD, name="move_down", callback=move_down),
        Keybind(key_code=key.J, key_action=KeyAction.HOLD, name="rotate_ccw", callback=rotate_ccw),
        Keybind(key_code=key.K, key_action=KeyAction.HOLD, name="rotate_cw", callback=rotate_cw),
        Keybind(key_code=key.U, key_action=KeyAction.HOLD, name="reset_scene", callback=reset_robot),
        Keybind(key_code=key.SPACE, key_action=KeyAction.PRESS, name="close_gripper", callback=close_gripper),
        Keybind(key_code=key.SPACE, key_action=KeyAction.RELEASE, name="open_gripper", callback=open_gripper),
    )

    ########################## run simulation ##########################
    try:
        while True:
            # Update target entity visualization
            target_quat = target_R[0].as_quat(scalar_first=True)
            target.set_qpos(np.concatenate([target_pos, target_quat]))

            # Control arm with inverse kinematics
            q, err = robot.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat, return_error=True)
            robot.control_dofs_position(q[:-2], motors_dof)

            scene.step()
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")
