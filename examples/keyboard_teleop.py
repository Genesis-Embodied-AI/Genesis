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
import pyglet
from scipy.spatial.transform import Rotation as R
from typing_extensions import override

import genesis as gs
from genesis.ext.pyrender.interaction import register_viewer_plugin
from genesis.ext.pyrender.interaction.plugins.viewer_controls import ViewerDefaultControls
from genesis.options.viewer_interactions import ViewerDefaultControls as ViewerDefaultControlsOptions


class FrankaTeleopOptions(ViewerDefaultControlsOptions):
    """Options for Franka teleoperation plugin."""

    pass


@register_viewer_plugin(FrankaTeleopOptions)
class FrankaTeleopPlugin(ViewerDefaultControls):
    """
    Viewer plugin for teleoperating Franka robot with keyboard.
    Extends ViewerDefaultControls to add robot-specific controls.
    """

    def __init__(self, viewer, options=None, camera=None, scene=None, viewport_size=None):
        super().__init__(viewer, options, camera, scene, viewport_size)

        # Robot control state
        self.robot = None
        self.target_entity = None
        self.cube_entity = None
        self.target_pos = None
        self.target_R = None
        self.robot_init_pos = np.array([0.5, 0, 0.55])
        self.robot_init_R = R.from_euler("y", np.pi)

        # Control parameters
        self.dpos = 0.002
        self.drot = 0.01
        self.is_close_gripper = False

        # keybindings
        self.keybindings.extend(
            dict(
                move_forward=pyglet.window.key.UP,
                move_backward=pyglet.window.key.DOWN,
                move_left=pyglet.window.key.LEFT,
                move_right=pyglet.window.key.RIGHT,
                move_up=pyglet.window.key.N,
                move_down=pyglet.window.key.M,
                rotate_ccw=pyglet.window.key.J,
                rotate_cw=pyglet.window.key.K,
                reset_scene=pyglet.window.key.U,
                close_gripper=pyglet.window.key.SPACE,
            )
        )
        self._instr_texts = (
            ["> [i]: show keyboard instructions"],
            ["< [i]: hide keyboard instructions"]
            + self.keybindings.as_instruction_texts(padding=3, exclude=("toggle_keyboard_instructions")),
        )

    def set_entities(self, robot, target_entity, cube_entity):
        """Set references to scene entities."""
        self.robot = robot
        self.target_entity = target_entity
        self.cube_entity = cube_entity

        # Initialize target pose
        self.target_pos = self.robot_init_pos.copy()
        self.target_R = self.robot_init_R

        # Get DOF indices
        n_dofs = robot.n_dofs
        self.motors_dof = np.arange(n_dofs - 2)
        self.fingers_dof = np.arange(n_dofs - 2, n_dofs)
        self.ee_link = robot.get_link("hand")

        # Reset to initial pose
        self._reset_robot()

    def _reset_robot(self):
        """Reset robot and cube to initial positions."""
        if self.robot is None:
            return

        self.target_pos = self.robot_init_pos.copy()
        self.target_R = self.robot_init_R
        target_quat = self.target_R.as_quat(scalar_first=True)
        self.target_entity.set_qpos(np.concatenate([self.target_pos, target_quat]))
        q = self.robot.inverse_kinematics(link=self.ee_link, pos=self.target_pos, quat=target_quat)
        self.robot.set_qpos(q[:-2], self.motors_dof)

        # Randomize cube position
        self.cube_entity.set_pos((random.uniform(0.2, 0.4), random.uniform(-0.2, 0.2), 0.05))
        self.cube_entity.set_quat(R.from_euler("z", random.uniform(0, np.pi * 2)).as_quat(scalar_first=True))

    @override
    def on_key_press(self, symbol: int, modifiers: int):
        # First handle default viewer controls
        result = super().on_key_press(symbol, modifiers)

        if self.robot is None:
            return result

        # Handle teleoperation controls
        if symbol == pyglet.window.key.UP:
            self.target_pos[0] -= self.dpos
        elif symbol == pyglet.window.key.DOWN:
            self.target_pos[0] += self.dpos
        elif symbol == pyglet.window.key.RIGHT:
            self.target_pos[1] += self.dpos
        elif symbol == pyglet.window.key.LEFT:
            self.target_pos[1] -= self.dpos
        elif symbol == pyglet.window.key.N:
            self.target_pos[2] += self.dpos
        elif symbol == pyglet.window.key.M:
            self.target_pos[2] -= self.dpos
        elif symbol == pyglet.window.key.J:
            self.target_R = R.from_euler("z", self.drot) * self.target_R
        elif symbol == pyglet.window.key.K:
            self.target_R = R.from_euler("z", -self.drot) * self.target_R
        elif symbol == pyglet.window.key.U:
            self._reset_robot()
        elif symbol == pyglet.window.key.SPACE:
            self.is_close_gripper = True

        return result

    @override
    def on_key_release(self, symbol: int, modifiers: int):
        result = super().on_key_release(symbol, modifiers)

        if symbol == pyglet.window.key.SPACE:
            self.is_close_gripper = False

        return result

    @override
    def update_on_sim_step(self):
        """Update robot control every simulation step."""
        super().update_on_sim_step()

        if self.robot is None:
            return

        # Update target entity visualization
        target_quat = self.target_R.as_quat(scalar_first=True)
        self.target_entity.set_qpos(np.concatenate([self.target_pos, target_quat]))

        # Control arm with inverse kinematics
        q, err = self.robot.inverse_kinematics(
            link=self.ee_link, pos=self.target_pos, quat=target_quat, return_error=True
        )
        self.robot.control_dofs_position(q[:-2], self.motors_dof)

        # Control gripper
        if self.is_close_gripper:
            self.robot.control_dofs_force(np.array([-1.0, -1.0]), self.fingers_dof)
        else:
            self.robot.control_dofs_force(np.array([1.0, 1.0]), self.fingers_dof)


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
            viewer_plugin=FrankaTeleopOptions(),
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

    # Set up the teleoperation plugin with entity references
    teleop_plugin = scene.viewer.viewer_interaction
    teleop_plugin.set_entities(robot, target, cube)

    print("\nKeyboard Controls:")
    print("↑\t- Move Forward (North)")
    print("↓\t- Move Backward (South)")
    print("←\t- Move Left (West)")
    print("→\t- Move Right (East)")
    print("n\t- Move Up")
    print("m\t- Move Down")
    print("j\t- Rotate Counterclockwise")
    print("k\t- Rotate Clockwise")
    print("u\t- Reset Scene")
    print("space\t- Press to close gripper, release to open gripper")
    print("\nPress 'i' in the viewer to see all keyboard controls")

    ########################## run simulation ##########################
    while True:
        scene.step()
