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
"""

import argparse
import csv
import os
from datetime import datetime

import numpy as np
from huggingface_hub import snapshot_download

import genesis as gs
import genesis.utils.geom as gu
from genesis.vis.keybindings import Key, KeyAction, Keybind


def build_scene(use_ipc=False, show_viewer=False, enable_ipc_gui=False):
    ########################## init ##########################
    gs.init(seed=0, precision="32", logging_level="info", backend=gs.gpu, performance_mode=True)
    np.set_printoptions(precision=7, suppress=True)

    dt = 1e-3

    ########################## create a scene ##########################
    coupler_options = (
        gs.options.IPCCouplerOptions(
            dt=dt,
            gravity=(0.0, 0.0, -9.8),
            ipc_constraint_strength=(1, 1),  # (translation, rotation) strength ratios,
            contact_friction_mu=0.8,
            IPC_self_contact=False,  # Disable rigid-rigid contact in IPC
            two_way_coupling=True,  # Enable two-way coupling (forces from IPC to Genesis rigid bodies)
            enable_ipc_gui=enable_ipc_gui,
        )
        if use_ipc
        else None
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=(0, 0, -9.8)),
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
        show_viewer=show_viewer,
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

    if use_ipc:
        scene.sim.coupler.set_ipc_link_filter(
            entity=entities["robot"],
            link_names=["left_finger", "right_finger"],
        )
        cloth = scene.add_entity(
            morph=gs.morphs.Mesh(
                file="meshes/grid20x20.obj",
                scale=0.5,
                pos=(0.5, 0.0, 0.1),
                euler=(90, 0, 0),
            ),
            material=gs.materials.FEM.Cloth(
                E=1e5,  # Young's modulus (Pa) - soft cloth (10 kPa)
                nu=0.499,  # Poisson's ratio - nearly incompressible
                rho=200,  # Density (kg/m³) - typical fabric
                thickness=0.002,  # Shell thickness (m) - 2mm
                bending_stiffness=100.0,  # Bending resistance
            ),
            surface=gs.surfaces.Plastic(color=(0.3, 0.5, 0.8, 1.0), double_sided=True),
        )

    # Add 16 rigid cubes uniformly distributed under the cloth (4x4 grid)
    cube_size = 0.05
    cube_height = 0.02501  # Height
    grid_spacing = 0.15  # Spacing between cubes

    for i in range(4):
        for j in range(4):
            x = (i + 1.7) * grid_spacing  # Center the grid
            y = (j - 1.5) * grid_spacing
            scene.add_entity(
                morph=gs.morphs.Box(
                    pos=(x, y, cube_height),
                    size=(cube_size, cube_size, cube_size),
                    fixed=True,
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


def run_sim(scene, entities, add_keybinds, mode="interactive", trajectory_file=None):
    robot = entities["robot"]
    target_entity = entities["target"]
    is_running = True

    robot_init_pos = np.array([0.5, 0, 0.55])
    robot_init_quat = gu.xyz_to_quat(np.array([0, np.pi, 0]))  # Rotation around Y axis
    target_pos = robot_init_pos.copy()
    target_quat = robot_init_quat.copy()

    n_dofs = robot.n_dofs
    motors_dof = np.arange(n_dofs - 2)
    fingers_dof = np.arange(n_dofs - 2, n_dofs)
    ee_link = robot.get_link("hand")

    # Trajectory recording
    trajectory = []
    recording = mode == "record"

    # Gripper state (use list for mutability in closures)
    gripper_closed = [False]

    # Control parameters
    dpos = 0.002
    drot = 0.01

    def reset_scene():
        target_pos[:] = robot_init_pos
        target_quat[:] = robot_init_quat
        target_entity.set_qpos(np.concatenate([target_pos, target_quat]))
        q = robot.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat)
        robot.set_qpos(q[:-2], motors_dof)

        # entities["cube"].set_pos((random.uniform(0.2, 0.4), random.uniform(-0.2, 0.2), 0.05))
        # entities["cube"].set_quat(R.from_euler("z", random.uniform(0, np.pi * 2)).as_quat(scalar_first=True))

    # Register keybindings
    if add_keybinds:

        def move(dpos_delta: tuple[float, float, float]):
            target_pos[:] += np.array(dpos_delta, dtype=gs.np_float)

        def rotate(axis: str, angle: float):
            # Create rotation quaternion for the specified axis
            euler = np.zeros(3)
            axis_map = {"x": 0, "y": 1, "z": 2}
            euler[axis_map[axis]] = angle
            drot_quat = gu.xyz_to_quat(euler)
            target_quat[:] = gu.transform_quat_by_quat(target_quat, drot_quat)

        def toggle_gripper(close: bool = True):
            gripper_closed[0] = close

        def stop():
            nonlocal is_running
            is_running = False

        scene.viewer.register_keybinds(
            Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=move, args=((-dpos, 0, 0),)),
            Keybind("move_backward", Key.DOWN, KeyAction.HOLD, callback=move, args=((dpos, 0, 0),)),
            Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=move, args=((0, -dpos, 0),)),
            Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=move, args=((0, dpos, 0),)),
            Keybind("move_up", Key.N, KeyAction.HOLD, callback=move, args=((0, 0, dpos),)),
            Keybind("move_down", Key.M, KeyAction.HOLD, callback=move, args=((0, 0, -dpos),)),
            Keybind("yaw_left", Key.J, KeyAction.HOLD, callback=rotate, args=("z", drot)),
            Keybind("yaw_right", Key.K, KeyAction.HOLD, callback=rotate, args=("z", -drot)),
            Keybind("pitch_up", Key.I, KeyAction.HOLD, callback=rotate, args=("y", drot)),
            Keybind("pitch_down", Key.O, KeyAction.HOLD, callback=rotate, args=("y", -drot)),
            Keybind("roll_left", Key.L, KeyAction.HOLD, callback=rotate, args=("x", drot)),
            Keybind("roll_right", Key.SEMICOLON, KeyAction.HOLD, callback=rotate, args=("x", -drot)),
            Keybind("reset_scene", Key.U, KeyAction.HOLD, callback=reset_scene),
            Keybind("close_gripper", Key.SPACE, KeyAction.PRESS, callback=toggle_gripper, args=(True,)),
            Keybind("open_gripper", Key.SPACE, KeyAction.RELEASE, callback=toggle_gripper, args=(False,)),
            Keybind("quit", Key.ESCAPE, KeyAction.PRESS, callback=stop),
        )

    # Load trajectory if in playback mode
    if mode == "playback":
        if not trajectory_file or not os.path.exists(trajectory_file):
            file_name = "grasp_cloth1.csv"
            trajectory_file = snapshot_download(
                repo_type="dataset",
                repo_id="Genesis-Intelligence/assets",
                revision="72b04f7125e21df1bebd54a7f7b39d1cd832331c",
                allow_patterns=f"{file_name}",
                max_workers=1,
            )
            trajectory_file = os.path.join(trajectory_file, file_name)

        with open(trajectory_file, "r", newline="") as f:
            reader = csv.DictReader(f)
            trajectory = []
            for row in reader:
                step_data = {
                    "target_pos": np.array([float(row["pos_x"]), float(row["pos_y"]), float(row["pos_z"])]),
                    "target_quat": np.array(
                        [float(row["quat_x"]), float(row["quat_y"]), float(row["quat_z"]), float(row["quat_w"])]
                    ),
                    "gripper_closed": row["gripper_closed"] == "True",
                    "step": int(row["step"]),
                }
                trajectory.append(step_data)
        print(f"\nLoaded trajectory from {trajectory_file}")
        print(f"Trajectory length: {len(trajectory)} steps")

    print(f"\nMode: {mode.upper()}")
    if mode == "record":
        print("Recording trajectory...")
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
    if mode in ["interactive", "record"]:
        print("\nPlus all default viewer controls (press 'i' to see them)")

    # reset scene before starting teleoperation
    reset_scene()

    # start teleoperation or playback
    step_count = 0

    try:
        while is_running:
            if mode == "playback":
                # Playback mode: replay recorded trajectory
                if step_count < len(trajectory):
                    step_data = trajectory[step_count]
                    target_pos[:] = step_data["target_pos"]
                    target_quat[:] = step_data["target_quat"]
                    gripper_closed[0] = step_data["gripper_closed"]
                    step_count += 1
                    print(f"\rPlayback step: {step_count}/{len(trajectory)}", end="")
                else:
                    print("\nPlayback finished!")
                    break
            else:
                # Interactive or recording mode
                # Movement is handled by keybinding callbacks
                # Record current state if recording
                if recording:
                    step_data = {
                        "target_pos": target_pos.copy(),
                        "target_quat": target_quat.copy(),
                        "gripper_closed": gripper_closed[0],
                        "step": step_count,
                    }
                    trajectory.append(step_data)

            # control arm
            target_entity.set_qpos(np.concatenate([target_pos, target_quat]))
            q, err = robot.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat, return_error=True)
            robot.control_dofs_position(q[:-2], motors_dof)
            # control gripper
            if gripper_closed[0]:
                robot.control_dofs_force(np.array([-1.0, -1.0]), fingers_dof)
            else:
                robot.control_dofs_force(np.array([1.0, 1.0]), fingers_dof)

            scene.step()
            step_count += 1

            if "PYTEST_VERSION" in os.environ:
                break
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")

    # Save trajectory if recording
    if recording and len(trajectory) > 0:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        traj_dir = os.path.join(script_dir, "trajectories")
        os.makedirs(traj_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(traj_dir, f"trajectory_{timestamp}.csv")
        with open(filename, "w", newline="") as f:
            fieldnames = ["step", "pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w", "gripper_closed"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for step_data in trajectory:
                writer.writerow(
                    {
                        "step": step_data["step"],
                        "pos_x": step_data["target_pos"][0],
                        "pos_y": step_data["target_pos"][1],
                        "pos_z": step_data["target_pos"][2],
                        "quat_x": step_data["target_quat"][0],
                        "quat_y": step_data["target_quat"][1],
                        "quat_z": step_data["target_quat"][2],
                        "quat_w": step_data["target_quat"][3],
                        "gripper_closed": step_data["gripper_closed"],
                    }
                )
        print(f"\nTrajectory saved to {filename}")
        print(f"Total steps: {len(trajectory)}")


def list_trajectories():
    """List all saved trajectories"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    traj_dir = os.path.join(script_dir, "trajectories")

    if not os.path.exists(traj_dir):
        print("No trajectories folder found.")
        return []

    files = [f for f in os.listdir(traj_dir) if f.endswith(".csv")]
    if not files:
        print("No trajectory files found.")
        return []

    print("\nAvailable trajectories:")
    for i, f in enumerate(files):
        print(f"  {i}: {f}")
    return files


def main():
    parser = argparse.ArgumentParser(description="Interactive IPC Arm Control with Trajectory Recording/Playback")
    parser.add_argument("--ipc", action="store_true", default=False, help="Enable IPC coupling")
    parser.add_argument(
        "--mode",
        type=str,
        default="playback",
        choices=["interactive", "record", "playback"],
        help="Mode: interactive, record (save trajectory), or playback (replay trajectory, default)",
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        default="grasp_cloth1.csv",
        help="Trajectory file to load (for playback mode, default: grasp_cloth1.csv)",
    )
    parser.add_argument("--list", action="store_true", help="List available trajectories and exit")
    parser.add_argument("-v", "--vis", action="store_true", default=False, help="Show Genesis viewer")
    parser.add_argument("--vis_ipc", action="store_true", default=False, help="Show IPC GUI")
    args = parser.parse_args()
    args.vis = args.vis or args.vis_ipc

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

    scene, entities = build_scene(use_ipc=args.ipc, show_viewer=args.vis, enable_ipc_gui=False)
    run_sim(
        scene,
        entities,
        add_keybinds=args.vis or args.mode in ["interactive", "record"],
        mode=args.mode,
        trajectory_file=trajectory_file,
    )


if __name__ == "__main__":
    main()
