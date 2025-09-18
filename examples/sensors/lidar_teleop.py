import argparse
import threading

import numpy as np
from custom_recorders import MPLImageViewerOptions, PointCloudDrawerOptions
from pynput import keyboard

import genesis as gs
from genesis.sensors.raycaster.camera_pattern import DepthCameraPattern
from genesis.sensors.raycaster.lidar_pattern import (
    SphericalPattern,
    SpinningLidarPattern,
)
from genesis.utils.geom import euler_to_quat

KEY_DPOS = 0.05
KEY_DANGLE = 0.1


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


def build_scene(show_viewer: bool = True) -> gs.Scene:
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.02, substeps=2, gravity=(0.0, 0.0, -9.81)),
        rigid_options=gs.options.RigidOptions(
            dt=0.02,
            gravity=(0.0, 0.0, -9.81),
            enable_collision=True,
            constraint_solver=gs.constraint_solver.Newton,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(6.0, 6.0, 4.0),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=60,
            max_FPS=60,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    scene.add_entity(gs.morphs.Plane())

    # create ring of obstacles to visualize raycaster sensor hits
    inner_radius = 3.0
    for i in range(8):
        angle = 2 * np.pi * i / 8
        x = inner_radius * np.cos(angle)
        y = inner_radius * np.sin(angle)
        scene.add_entity(gs.morphs.Cylinder(height=1.5, radius=0.3, pos=(x, y, 0.75), fixed=True))

    outer_radius = 5.0
    for i in range(6):
        angle = 2 * np.pi * i / 6 + np.pi / 6
        x = outer_radius * np.cos(angle)
        y = outer_radius * np.sin(angle)
        scene.add_entity(gs.morphs.Box(size=(0.5, 0.5, 2.0), pos=(x, y, 1.0), fixed=True))

    return scene


def create_robot_with_lidar(scene, args):
    """
    Create fixed-base robot with a LiDAR or Depth Camera sensor attached.

    Parameters
    ----------
    scene : gs.Scene
        The scene to create the robot in.
    args : argparse.Namespace
        The arguments to create the robot with.

    Returns
    -------
    robot : gs.engine.entities.RigidEntity
        The robot entity.
    sensor : gs.sensors.Raycaster
        The LiDAR or Depth Camera sensor.
    """

    robot_kwargs = dict(
        pos=(0.0, 0.0, 0.35),
        quat=(1.0, 0.0, 0.0, 0.0),
        fixed=True,
    )

    if args.use_box:
        robot = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), **robot_kwargs))
    else:
        robot = scene.add_entity(gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", **robot_kwargs))

    sensor_kwargs = dict(
        entity_idx=robot.idx,
        pos_offset=(0.3, 0.0, 0.0),
        euler_offset=(0.0, 0.0, 0.0),
        return_world_frame=True,
    )

    if args.pattern == "depth":
        sensor = scene.add_sensor(gs.sensors.DepthCamera(pattern=DepthCameraPattern(), **sensor_kwargs))
        return robot, sensor

    if args.pattern == "livox":
        pattern_cfg = "avia"
    elif args.pattern == "spinning":
        pattern_cfg = SpinningLidarPattern()
    else:
        pattern_cfg = SphericalPattern()

    sensor = scene.add_sensor(gs.sensors.Lidar(pattern=pattern_cfg, **sensor_kwargs))
    return robot, sensor


def run(scene: gs.Scene, robot, sensor: gs.sensors.Lidar, n_envs: int, kb: KeyboardDevice, is_depth: bool = False):
    if is_depth:
        scene.start_recording(
            data_func=(lambda: sensor.read()[0]) if n_envs > 0 else sensor.read,
            rec_options=MPLImageViewerOptions(hz=30),
        )
    else:
        scene.start_recording(
            data_func=(lambda: sensor.read()["hit_points"][0]) if n_envs > 0 else (lambda: sensor.read()["hit_points"]),
            rec_options=PointCloudDrawerOptions(
                hz=30,
                sphere_radius=0.02,
                draw_debug_spheres=scene.draw_debug_spheres,
                clear_debug_object=scene.clear_debug_object,
            ),
        )

    scene.build(n_envs=n_envs)

    print("\nKeyboard Controls:")
    print("↑/↓/←/→: Move XY, n/m: Up/Down, u/o: Roll CCW/CW, i/k: Pitch Up/Down, j/l: Yaw CCW/CW, r: Reset, esc: Quit")

    init_pos = np.array([0.0, 0.0, 0.35], dtype=np.float32)
    init_euler = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    target_pos = init_pos.copy()
    target_euler = init_euler.copy()

    def apply_pose_to_all_envs(pos_np: np.ndarray, quat_np: np.ndarray):
        robot.set_pos(pos_np, zero_velocity=False)
        robot.set_quat(quat_np, zero_velocity=False)

    apply_pose_to_all_envs(target_pos, euler_to_quat(target_euler))

    try:
        while True:
            pressed = kb.pressed_keys.copy()
            if keyboard.Key.esc in pressed:
                break
            if keyboard.KeyCode.from_char("r") in pressed:
                target_pos[:] = init_pos
                target_euler[:] = init_euler

            if keyboard.Key.up in pressed:
                target_pos[0] += KEY_DPOS
            if keyboard.Key.down in pressed:
                target_pos[0] -= KEY_DPOS
            if keyboard.Key.right in pressed:
                target_pos[1] -= KEY_DPOS
            if keyboard.Key.left in pressed:
                target_pos[1] += KEY_DPOS
            if keyboard.KeyCode.from_char("n") in pressed:
                target_pos[2] += KEY_DPOS
            if keyboard.KeyCode.from_char("m") in pressed:
                target_pos[2] -= KEY_DPOS

            if keyboard.KeyCode.from_char("u") in pressed:
                target_euler[0] += KEY_DANGLE  # roll CCW around +X
            if keyboard.KeyCode.from_char("o") in pressed:
                target_euler[0] -= KEY_DANGLE  # roll CW around +X
            if keyboard.KeyCode.from_char("i") in pressed:
                target_euler[1] += KEY_DANGLE  # pitch up around +Y
            if keyboard.KeyCode.from_char("k") in pressed:
                target_euler[1] -= KEY_DANGLE  # pitch down around +Y
            if keyboard.KeyCode.from_char("j") in pressed:
                target_euler[2] += KEY_DANGLE  # yaw CCW around +Z
            if keyboard.KeyCode.from_char("l") in pressed:
                target_euler[2] -= KEY_DANGLE  # yaw CW around +Z

            apply_pose_to_all_envs(target_pos, euler_to_quat(target_euler))

            scene.step()

    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")

        scene.stop_recording()


def main():
    parser = argparse.ArgumentParser(description="Genesis LiDAR/Depth Camera Visualization with Keyboard Teleop")
    parser.add_argument("--n-envs", type=int, default=0, help="Number of environments to replicate")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of GPU")
    parser.add_argument("--use-box", action="store_true", help="Use Box as robot instead of Go2")
    parser.add_argument(
        "--pattern",
        type=str,
        default="depth",
        choices=["spherical", "spinning", "depth"],
        help="Sensor pattern type",
    )

    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="info")

    kb = KeyboardDevice()
    kb.start()

    scene = build_scene(show_viewer=True)
    robot, lidar = create_robot_with_lidar(scene, args)

    run(scene, robot, lidar, n_envs=args.n_envs, kb=kb, is_depth=args.pattern == "depth")


if __name__ == "__main__":
    main()
