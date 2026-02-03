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

Uses ViewerOptions.reserved_keys so these keys are handled by the application
(when the viewer has focus) and do not trigger the viewer's shortcuts; the
viewer's shortcuts (e.g. [a], [z], [r]) remain available for other keys.
"""

import os
import random
import threading

import genesis as gs
import numpy as np
import pyglet.window.key as pyglet_key
from pynput import keyboard
from scipy.spatial.transform import Rotation as R

# Keys reserved for this example: viewer will not use them for its shortcuts
RESERVED_KEYS = (
    pyglet_key.UP,
    pyglet_key.DOWN,
    pyglet_key.LEFT,
    pyglet_key.RIGHT,
    pyglet_key.N,
    pyglet_key.M,
    pyglet_key.J,
    pyglet_key.K,
    pyglet_key.U,
    pyglet_key.SPACE,
    pyglet_key.ESCAPE,
)


class KeyboardDevice:
    def __init__(self):
        self.pressed_keys = set()
        self.lock = threading.Lock()
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)

    def start(self):
        self.listener.start()

    def stop(self):
        try:
            self.listener.stop()
        except NotImplementedError:
            # Dummy backend does not implement stop
            pass
        self.listener.join()

    def on_press(self, key: keyboard.Key):
        with self.lock:
            self.pressed_keys.add(key)

    def on_release(self, key: keyboard.Key):
        with self.lock:
            self.pressed_keys.discard(key)

    def get_cmd(self):
        return self.pressed_keys


def build_scene():
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
            reserved_keys=RESERVED_KEYS,
        ),
        show_viewer=True,
        show_FPS=False,
    )

    ########################## entities ##########################
    entities = dict()
    entities["plane"] = scene.add_entity(
        gs.morphs.Plane(),
    )

    entities["robot"] = scene.add_entity(
        material=gs.materials.Rigid(gravity_compensation=1),
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            euler=(0, 0, 0),
        ),
    )
    entities["cube"] = scene.add_entity(
        material=gs.materials.Rigid(rho=300),
        morph=gs.morphs.Box(
            pos=(0.5, 0.0, 0.07),
            size=(0.04, 0.04, 0.04),
        ),
        surface=gs.surfaces.Default(color=(0.5, 1, 0.5)),
    )

    entities["target"] = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            scale=0.15,
            collision=False,
        ),
        surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
    )

    ########################## build ##########################
    scene.build()

    return scene, entities


def run_sim(scene, entities, clients):
    robot = entities["robot"]
    target_entity = entities["target"]

    robot_init_pos = np.array([0.5, 0, 0.55])
    robot_init_R = R.from_euler("y", np.pi)
    target_pos = robot_init_pos.copy()
    target_R = robot_init_R

    n_dofs = robot.n_dofs
    motors_dof = np.arange(n_dofs - 2)
    fingers_dof = np.arange(n_dofs - 2, n_dofs)
    ee_link = robot.get_link("hand")

    def reset_scene():
        nonlocal target_pos, target_R
        target_pos = robot_init_pos.copy()
        target_R = robot_init_R
        target_quat = target_R.as_quat(scalar_first=True)
        target_entity.set_qpos(np.concatenate([target_pos, target_quat]))
        q = robot.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat)
        robot.set_qpos(q[:-2], motors_dof)

        entities["cube"].set_pos((random.uniform(0.2, 0.4), random.uniform(-0.2, 0.2), 0.05))
        entities["cube"].set_quat(R.from_euler("z", random.uniform(0, np.pi * 2)).as_quat(scalar_first=True))

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
    print("esc\t- Quit")

    # reset scen before starting teleoperation
    reset_scene()

    # start teleoperation (reserved keys from viewer when it has focus, else pynput)
    stop = False
    while not stop:
        pynput_pressed = clients["keyboard"].pressed_keys.copy()
        viewer_pressed = scene.viewer.get_pressed_keys()

        def is_pressed(pynput_key, pyglet_sym):
            return pynput_key in pynput_pressed or pyglet_sym in viewer_pressed

        # reset scene
        if is_pressed(keyboard.KeyCode.from_char("u"), pyglet_key.U):
            reset_scene()

        # stop teleoperation
        stop = is_pressed(keyboard.Key.esc, pyglet_key.ESCAPE)

        # get ee target pose
        is_close_gripper = False
        dpos = 0.002
        drot = 0.01
        if is_pressed(keyboard.Key.up, pyglet_key.UP):
            target_pos[0] -= dpos
        if is_pressed(keyboard.Key.down, pyglet_key.DOWN):
            target_pos[0] += dpos
        if is_pressed(keyboard.Key.right, pyglet_key.RIGHT):
            target_pos[1] += dpos
        if is_pressed(keyboard.Key.left, pyglet_key.LEFT):
            target_pos[1] -= dpos
        if is_pressed(keyboard.KeyCode.from_char("n"), pyglet_key.N):
            target_pos[2] += dpos
        if is_pressed(keyboard.KeyCode.from_char("m"), pyglet_key.M):
            target_pos[2] -= dpos
        if is_pressed(keyboard.KeyCode.from_char("j"), pyglet_key.J):
            target_R = R.from_euler("z", drot) * target_R
        if is_pressed(keyboard.KeyCode.from_char("k"), pyglet_key.K):
            target_R = R.from_euler("z", -drot) * target_R
        if is_pressed(keyboard.Key.space, pyglet_key.SPACE):
            is_close_gripper = True

        # control arm
        target_quat = target_R.as_quat(scalar_first=True)
        target_entity.set_qpos(np.concatenate([target_pos, target_quat]))
        q, _err = robot.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat, return_error=True)
        robot.control_dofs_position(q[:-2], motors_dof)

        # control gripper
        if is_close_gripper:
            robot.control_dofs_force(np.array([-1.0, -1.0]), fingers_dof)
        else:
            robot.control_dofs_force(np.array([1.0, 1.0]), fingers_dof)

        scene.step()

        if "PYTEST_VERSION" in os.environ:
            break


def main():
    clients = dict()
    clients["keyboard"] = KeyboardDevice()
    clients["keyboard"].start()

    scene, entities = build_scene()
    run_sim(scene, entities, clients)


if __name__ == "__main__":
    main()
