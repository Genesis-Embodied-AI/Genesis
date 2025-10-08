import argparse
import os
import threading

import numpy as np

import genesis as gs
from genesis.sensors.raycaster.patterns import DepthCameraPattern, GridPattern, SphericalPattern
from genesis.utils.geom import euler_to_quat

IS_PYNPUT_AVAILABLE = False
try:
    from pynput import keyboard

    IS_PYNPUT_AVAILABLE = True
except ImportError:
    pass

# Position and angle increments for keyboard teleop control
KEY_DPOS = 0.1
KEY_DANGLE = 0.1

# Movement when no keyboard control is available
MOVE_RADIUS = 1.0
MOVE_RATE = 1.0 / 100.0

# Number of obstacles to create in a ring around the robot
NUM_CYLINDERS = 8
NUM_BOXES = 6
CYLINDER_RING_RADIUS = 3.0
BOX_RING_RADIUS = 5.0


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

    def on_press(self, key: "keyboard.Key"):
        with self.lock:
            self.pressed_keys.add(key)

    def on_release(self, key: "keyboard.Key"):
        with self.lock:
            self.pressed_keys.discard(key)

    def get_cmd(self):
        return self.pressed_keys


def main():
    parser = argparse.ArgumentParser(description="Genesis LiDAR/Depth Camera Visualization with Keyboard Teleop")
    parser.add_argument("-B", "--n_envs", type=int, default=0, help="Number of environments to replicate")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of GPU")
    parser.add_argument("--use-box", action="store_true", help="Use Box as robot instead of Go2")
    parser.add_argument("-f", "--fixed", action="store_true", help="Load obstacles as fixed.")
    parser.add_argument(
        "--pattern",
        type=str,
        default="spherical",
        choices=["spherical", "depth", "grid"],
        help="Sensor pattern type",
    )

    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="info")

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(6.0, 6.0, 4.0),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=60,
            max_FPS=60,
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=True,
    )

    scene.add_entity(gs.morphs.Plane())

    # create ring of obstacles to visualize raycaster sensor hits
    for i in range(NUM_CYLINDERS):
        angle = 2 * np.pi * i / NUM_CYLINDERS
        x = CYLINDER_RING_RADIUS * np.cos(angle)
        y = CYLINDER_RING_RADIUS * np.sin(angle)
        scene.add_entity(
            gs.morphs.Cylinder(
                height=1.5,
                radius=0.3,
                pos=(x, y, 0.75),
                fixed=args.fixed,
            )
        )

    for i in range(NUM_BOXES):
        angle = 2 * np.pi * i / NUM_BOXES + np.pi / 6
        x = BOX_RING_RADIUS * np.cos(angle)
        y = BOX_RING_RADIUS * np.sin(angle)
        scene.add_entity(
            gs.morphs.Box(
                size=(0.5, 0.5, 2.0),
                pos=(x, y, 1.0),
                fixed=args.fixed,
            )
        )

    robot_kwargs = dict(
        pos=(0.0, 0.0, 0.35),
        quat=(1.0, 0.0, 0.0, 0.0),
        fixed=True,
    )

    if args.use_box:
        robot = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), **robot_kwargs))
        pos_offset = (0.0, 0.0, 0.2)
    else:
        robot = scene.add_entity(gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", **robot_kwargs))
        pos_offset = (0.3, 0.0, 0.1)

    sensor_kwargs = dict(
        entity_idx=robot.idx,
        pos_offset=pos_offset,
        euler_offset=(0.0, 0.0, 0.0),
        return_world_frame=True,
        draw_debug=True,
    )

    if args.pattern == "depth":
        sensor = scene.add_sensor(gs.sensors.DepthCamera(pattern=DepthCameraPattern(), **sensor_kwargs))
        scene.start_recording(
            data_func=(lambda: sensor.read_image()[0]) if args.n_envs > 0 else sensor.read_image,
            rec_options=gs.recorders.MPLImagePlot(),
        )
    else:
        if args.pattern == "grid":
            pattern_cfg = GridPattern()
        else:
            if args.pattern != "spherical":
                gs.logger.warning(f"Unrecognized raycaster pattern: {args.pattern}. Using 'spherical' instead.")
            pattern_cfg = SphericalPattern()

        sensor = scene.add_sensor(gs.sensors.Lidar(pattern=pattern_cfg, **sensor_kwargs))

    scene.build(n_envs=args.n_envs)

    if IS_PYNPUT_AVAILABLE:
        kb = KeyboardDevice()
        kb.start()

        print("Keyboard Controls:")
        # Avoid using same keys as interactive viewer keyboard controls
        print("[↑/↓/←/→]: Move XY")
        print("[j/k]: Down/Up")
        print("[n/m]: Roll CCW/CW")
        print("[,/.]: Pitch Up/Down")
        print("[o/p]: Yaw CCW/CW")
        print("[\\]: Reset")
        print("[esc]: Quit")
    else:
        print("Keyboard teleop is disabled since pynput is not installed. To install, run `pip install pynput`.")

    init_pos = np.array([0.0, 0.0, 0.35], dtype=np.float32)
    init_euler = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    target_pos = init_pos.copy()
    target_euler = init_euler.copy()

    def apply_pose_to_all_envs(pos_np: np.ndarray, quat_np: np.ndarray):
        if args.n_envs > 0:
            pos_np = np.expand_dims(pos_np, axis=0).repeat(args.n_envs, axis=0)
            quat_np = np.expand_dims(quat_np, axis=0).repeat(args.n_envs, axis=0)
        robot.set_pos(pos_np)
        robot.set_quat(quat_np)

    apply_pose_to_all_envs(target_pos, euler_to_quat(target_euler))

    try:
        while True:
            if IS_PYNPUT_AVAILABLE:
                pressed = kb.pressed_keys.copy()
                if keyboard.Key.esc in pressed:
                    break
                if keyboard.KeyCode.from_char("\\") in pressed:
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
                if keyboard.KeyCode.from_char("j") in pressed:
                    target_pos[2] -= KEY_DPOS
                if keyboard.KeyCode.from_char("k") in pressed:
                    target_pos[2] += KEY_DPOS

                if keyboard.KeyCode.from_char("n") in pressed:
                    target_euler[0] += KEY_DANGLE  # roll CCW around +X
                if keyboard.KeyCode.from_char("m") in pressed:
                    target_euler[0] -= KEY_DANGLE  # roll CW around +X
                if keyboard.KeyCode.from_char(",") in pressed:
                    target_euler[1] += KEY_DANGLE  # pitch up around +Y
                if keyboard.KeyCode.from_char(".") in pressed:
                    target_euler[1] -= KEY_DANGLE  # pitch down around +Y
                if keyboard.KeyCode.from_char("o") in pressed:
                    target_euler[2] += KEY_DANGLE  # yaw CCW around +Z
                if keyboard.KeyCode.from_char("p") in pressed:
                    target_euler[2] -= KEY_DANGLE  # yaw CW around +Z
            else:
                # move in a circle if no keyboard control
                target_pos[0] = MOVE_RADIUS * np.cos(scene.t * MOVE_RATE)
                target_pos[1] = MOVE_RADIUS * np.sin(scene.t * MOVE_RATE)

            apply_pose_to_all_envs(target_pos, euler_to_quat(target_euler))
            scene.step()

            if "PYTEST_VERSION" in os.environ:
                break
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")


if __name__ == "__main__":
    main()
