"""
Keyboard Controls:
↑	- Move Forward (North)
↓	- Move Backward (South)
←	- Move Left (West)
→	- Move Right (East)
n	- Move Up
m	- Move Down
j	- Yaw Left (Rotate Counterclockwise around Z)
k	- Yaw Right (Rotate Clockwise around Z)
i	- Pitch Up (Rotate around Y)
o	- Pitch Down (Rotate around Y)
l	- Roll Left (Rotate around X)
;	- Roll Right (Rotate around X)
u	- Reset Scene
space	- Press to close gripper, release to open gripper
esc	- Quit
"""

import random
import threading
import argparse
import genesis as gs
import numpy as np
import pickle
import os
from datetime import datetime
from pynput import keyboard
from scipy.spatial.transform import Rotation as R


class KeyboardDevice:
    def __init__(self):
        self.pressed_keys = set()
        self.lock = threading.Lock()
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)

    def start(self):
        self.listener.start()

    def stop(self):
        self.listener.stop()
        self.listener.join()

    def on_press(self, key: keyboard.Key):
        with self.lock:
            self.pressed_keys.add(key)

    def on_release(self, key: keyboard.Key):
        with self.lock:
            self.pressed_keys.discard(key)

    def get_cmd(self):
        return self.pressed_keys


def build_scene(use_ipc=False):
    ########################## init ##########################
    gs.init(seed=0, precision="32", logging_level="info", backend=gs.cpu)
    np.set_printoptions(precision=7, suppress=True)

    ########################## create a scene ##########################
    coupler_options = gs.options.IPCCouplerOptions(
        dt=1e-3,
        gravity=(0.0, 0.0, -9.8),
        ipc_constraint_strength=(100, 100),  # (translation, rotation) strength ratios,
        contact_friction_mu=0.8,
        IPC_self_contact=False,  # Disable rigid-rigid contact in IPC
        two_way_coupling=False,  # Enable two-way coupling (forces from IPC to Genesis rigid bodies)
    ) if use_ipc else None


    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1e-3),
        coupler_options=coupler_options,
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
    SCENE_POS = (0.0, 0.0, 0.0)
    entities["robot"].set_ipc_link_filter(link_names=["left_finger", "right_finger"])

    material = gs.materials.FEM.Elastic(E=1.0e4, nu=0.45, rho=1000.0, model="stable_neohookean") if use_ipc else gs.materials.Rigid()

    # entities["cube"] = scene.add_entity(
    #     material=material,
    #     morph=gs.morphs.Box(
    #         pos=(0.5, 0.0, 0.07),
    #         size=(0.04, 0.04, 0.04),
    #     ),
    #     surface=gs.surfaces.Default(color=(0.5, 1, 0.5)),
    # )
    if use_ipc:
        cloth = scene.add_entity(
            morph=gs.morphs.Mesh(
                file=r"meshes\grid20x20.obj",
                scale=0.5, 
                pos=tuple(map(sum, zip(SCENE_POS, (1.0, 0.0, 0.3)))),
                euler=(90, 0, 0),
            ),
            material=gs.materials.Cloth(
                E=1e5,  # Young's modulus (Pa) - soft cloth (10 kPa)
                nu=0.499,  # Poisson's ratio - nearly incompressible
                rho=200,  # Density (kg/m³) - typical fabric
                thickness=0.001,  # Shell thickness (m) - 1mm
                bending_stiffness=100.0,  # Bending resistance
            ),
            surface=gs.surfaces.Plastic(color=(0.3, 0.5, 0.8, 1.0), double_sided=True),
        )

    # Add 16 rigid cubes uniformly distributed under the cloth (4x4 grid)
    cube_size = 0.05
    cube_height = 0.1  # Height below cloth
    grid_spacing = 0.15  # Spacing between cubes

    for i in range(4):
        for j in range(4):
            x = (i + 1.7) * grid_spacing  # Center the grid
            y = (j - 1.5) * grid_spacing
            scene.add_entity(
                morph=gs.morphs.Box(
                    pos=tuple(map(sum, zip(SCENE_POS, (x, y, cube_height)))),
                    size=(cube_size, cube_size, cube_size),
                ),
                material=gs.materials.Rigid(rho=500, friction=0.3),
                surface=gs.surfaces.Plastic(color=(0.8, 0.3, 0.2, 0.8)),
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


def run_sim(scene, entities, clients, mode="interactive", trajectory_file=None):
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

    # Trajectory recording
    trajectory = []
    recording = (mode == "record")

    def reset_scene():
        nonlocal target_pos, target_R
        target_pos = robot_init_pos.copy()
        target_R = robot_init_R
        target_quat = target_R.as_quat(scalar_first=True)
        target_entity.set_qpos(np.concatenate([target_pos, target_quat]))
        q = robot.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat)
        robot.set_qpos(q[:-2], motors_dof)

        # entities["cube"].set_pos((random.uniform(0.2, 0.4), random.uniform(-0.2, 0.2), 0.05))
        # entities["cube"].set_quat(R.from_euler("z", random.uniform(0, np.pi * 2)).as_quat(scalar_first=True))

    # Load trajectory if in playback mode
    if mode == "playback":
        if trajectory_file and os.path.exists(trajectory_file):
            with open(trajectory_file, 'rb') as f:
                trajectory = pickle.load(f)
            print(f"\nLoaded trajectory from {trajectory_file}")
            print(f"Trajectory length: {len(trajectory)} steps")
        else:
            print(f"Error: Trajectory file {trajectory_file} not found!")
            return

    print(f"\nMode: {mode.upper()}")
    if mode == "record":
        print("Recording trajectory... Press ESC to stop and save.")
    elif mode == "playback":
        print("Playing back trajectory...")

    print("\nKeyboard Controls:")
    print("↑\t- Move Forward (North)")
    print("↓\t- Move Backward (South)")
    print("←\t- Move Left (West)")
    print("→\t- Move Right (East)")
    print("n\t- Move Up")
    print("m\t- Move Down")
    print("j/k\t- Yaw Left/Right (Rotate around Z axis)")
    print("i/o\t- Pitch Up/Down (Rotate around Y axis)")
    print("l/;\t- Roll Left/Right (Rotate around X axis)")
    print("u\t- Reset Scene")
    print("space\t- Press to close gripper, release to open gripper")
    print("esc\t- Quit")

    # reset scene before starting teleoperation
    reset_scene()

    # start teleoperation or playback
    stop = False
    step_count = 0

    while not stop:
        if mode == "playback":
            # Playback mode: replay recorded trajectory
            if step_count < len(trajectory):
                step_data = trajectory[step_count]
                target_pos = step_data['target_pos']
                target_R = R.from_quat(step_data['target_quat'])
                is_close_gripper = step_data['gripper_closed']
                step_count += 1
                print(f"\rPlayback step: {step_count}/{len(trajectory)}", end="")
                # Check if user wants to stop playback
                pressed_keys = clients["keyboard"].pressed_keys.copy()
                stop = keyboard.Key.esc in pressed_keys
            else:
                print("\nPlayback finished!")
                break
        else:
            # Interactive or recording mode
            pressed_keys = clients["keyboard"].pressed_keys.copy()

            # reset scene:
            reset_flag = False
            reset_flag |= keyboard.KeyCode.from_char("u") in pressed_keys
            if reset_flag:
                reset_scene()

            # stop teleoperation
            stop = keyboard.Key.esc in pressed_keys

            # get ee target pose
            is_close_gripper = False
            dpos = 0.002
            drot = 0.01
            for key in pressed_keys:
                if key == keyboard.Key.up:
                    target_pos[0] -= dpos
                elif key == keyboard.Key.down:
                    target_pos[0] += dpos
                elif key == keyboard.Key.right:
                    target_pos[1] += dpos
                elif key == keyboard.Key.left:
                    target_pos[1] -= dpos
                elif key == keyboard.KeyCode.from_char("n"):
                    target_pos[2] += dpos
                elif key == keyboard.KeyCode.from_char("m"):
                    target_pos[2] -= dpos
                elif key == keyboard.KeyCode.from_char("j"):
                    target_R = R.from_euler("z", drot) * target_R
                elif key == keyboard.KeyCode.from_char("k"):
                    target_R = R.from_euler("z", -drot) * target_R
                elif key == keyboard.KeyCode.from_char("i"):
                    target_R = R.from_euler("y", drot) * target_R
                elif key == keyboard.KeyCode.from_char("o"):
                    target_R = R.from_euler("y", -drot) * target_R
                elif key == keyboard.KeyCode.from_char("l"):
                    target_R = R.from_euler("x", drot) * target_R
                elif key == keyboard.KeyCode.from_char(";"):
                    target_R = R.from_euler("x", -drot) * target_R
                elif key == keyboard.Key.space:
                    is_close_gripper = True

            # Record current state if recording
            if recording:
                step_data = {
                    'target_pos': target_pos.copy(),
                    'target_quat': target_R.as_quat(),  # x,y,z,w format
                    'gripper_closed': is_close_gripper,
                    'step': step_count
                }
                trajectory.append(step_data)

        # control arm
        target_quat = target_R.as_quat(scalar_first=True)
        target_entity.set_qpos(np.concatenate([target_pos, target_quat]))
        q, err = robot.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat, return_error=True)
        robot.control_dofs_position(q[:-2], motors_dof)
        # control gripper
        if is_close_gripper:
            robot.control_dofs_force(np.array([-1.0, -1.0]), fingers_dof)
        else:
            robot.control_dofs_force(np.array([1.0, 1.0]), fingers_dof)

        scene.step()
        step_count += 1

    # Save trajectory if recording
    if recording and len(trajectory) > 0:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        traj_dir = os.path.join(script_dir, "trajectories")
        os.makedirs(traj_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(traj_dir, f"trajectory_{timestamp}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(trajectory, f)
        print(f"\nTrajectory saved to {filename}")
        print(f"Total steps: {len(trajectory)}")


def list_trajectories():
    """List all saved trajectories"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    traj_dir = os.path.join(script_dir, "trajectories")

    if not os.path.exists(traj_dir):
        print("No trajectories folder found.")
        return []

    files = [f for f in os.listdir(traj_dir) if f.endswith('.pkl')]
    if not files:
        print("No trajectory files found.")
        return []

    print("\nAvailable trajectories:")
    for i, f in enumerate(files):
        print(f"  {i}: {f}")
    return files


def main():
    parser = argparse.ArgumentParser(description="Interactive IPC Arm Control with Trajectory Recording/Playback")
    parser.add_argument("--ipc", action="store_true", default=True, help="Enable IPC coupling")
    parser.add_argument("--mode", type=str, default="playback", choices=["interactive", "record", "playback"],
                        help="Mode: interactive, record (save trajectory), or playback (replay trajectory, default)")
    parser.add_argument("--trajectory", type=str, default="grap_cloth1.pkl",
                        help="Trajectory file to load (for playback mode, default: grap_cloth1.pkl)")
    parser.add_argument("--list", action="store_true", help="List available trajectories and exit")
    args = parser.parse_args()

    if args.list:
        list_trajectories()
        return

    # Handle trajectory selection for playback
    trajectory_file = args.trajectory
    if args.mode == "playback":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        traj_dir = os.path.join(script_dir, "trajectories")

        if trajectory_file is None:
            files = list_trajectories()
            if not files:
                return
            try:
                idx = int(input("\nSelect trajectory index: "))
                trajectory_file = os.path.join(traj_dir, files[idx])
            except (ValueError, IndexError):
                print("Invalid selection.")
                return
        elif not os.path.isabs(trajectory_file):
            trajectory_file = os.path.join(traj_dir, os.path.basename(trajectory_file))

    clients = dict()
    clients["keyboard"] = KeyboardDevice()
    clients["keyboard"].start()

    scene, entities = build_scene(use_ipc=args.ipc)
    run_sim(scene, entities, clients, mode=args.mode, trajectory_file=trajectory_file)


if __name__ == "__main__":
    main()
