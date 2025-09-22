import argparse

import numpy as np
from pynput import keyboard

import genesis as gs
from genesis.sensors.raycaster.camera_pattern import DepthCameraPattern
from genesis.sensors.raycaster.lidar_pattern import GridPattern, SphericalPattern, SpinningLidarPattern
from genesis.utils.geom import euler_to_quat
from genesis.utils.keyboard import KeyboardDevice

KEY_DPOS = 0.05
KEY_DANGLE = 0.1


def build_scene(show_viewer: bool, is_free: bool) -> gs.Scene:
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(6.0, 6.0, 4.0),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=60,
            max_FPS=60,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    scene.add_entity(gs.morphs.Plane(is_free=is_free))

    # create ring of obstacles to visualize raycaster sensor hits
    inner_radius = 3.0
    for i in range(8):
        angle = 2 * np.pi * i / 8
        x = inner_radius * np.cos(angle)
        y = inner_radius * np.sin(angle)
        scene.add_entity(gs.morphs.Cylinder(height=1.5, radius=0.3, pos=(x, y, 0.75), is_free=is_free))

    outer_radius = 5.0
    for i in range(6):
        angle = 2 * np.pi * i / 6 + np.pi / 6
        x = outer_radius * np.cos(angle)
        y = outer_radius * np.sin(angle)
        scene.add_entity(gs.morphs.Box(size=(0.5, 0.5, 2.0), pos=(x, y, 1.0), is_free=is_free))

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
        pos_offset = (0.0, 0.0, 0.2)
    else:
        robot = scene.add_entity(gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", **robot_kwargs))
        pos_offset = (0.3, 0.0, 0.1)

    sensor_kwargs = dict(
        entity_idx=robot.idx,
        pos_offset=pos_offset,
        euler_offset=(0.0, 0.0, 0.0),
        return_world_frame=True,
        only_cast_fixed=args.fixed,
        draw_debug=True,
    )

    if args.pattern == "depth":
        sensor = scene.add_sensor(gs.sensors.DepthCamera(pattern=DepthCameraPattern(), **sensor_kwargs))
        return robot, sensor
    elif args.pattern == "spherical":
        pattern_cfg = SphericalPattern()
    elif args.pattern == "grid":
        pattern_cfg = GridPattern()
    else:
        if args.pattern != "spinning":
            gs.logger.warning(f"Unrecognized raycaster pattern: {args.pattern}. Using 'spinning' instead.")
        pattern_cfg = SpinningLidarPattern()

    sensor = scene.add_sensor(gs.sensors.Lidar(pattern=pattern_cfg, **sensor_kwargs))
    return robot, sensor


def run(scene: gs.Scene, robot, sensor: gs.sensors.Lidar, n_envs: int, kb: KeyboardDevice, is_depth: bool = False):
    if is_depth:
        scene.start_recording(
            data_func=(lambda: sensor.read_image()[0]) if n_envs > 0 else sensor.read_image,
            rec_options=gs.recorders.MPLImagePlot(),
        )

    scene.build(n_envs=n_envs)

    print("Keyboard Controls:")
    # Avoid using same keys as interactive viewer keyboard controls
    print("[↑/↓/←/→]: Move XY")
    print("[j/k]: Down/Up")
    print("[n/m]: Roll CCW/CW")
    print("[,/.]: Pitch Up/Down")
    print("[o/p]: Yaw CCW/CW")
    print("[\\]: Reset")
    print("[esc]: Quit")

    init_pos = np.array([0.0, 0.0, 0.35], dtype=np.float32)
    init_euler = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    target_pos = init_pos.copy()
    target_euler = init_euler.copy()

    def apply_pose_to_all_envs(pos_np: np.ndarray, quat_np: np.ndarray):
        if n_envs > 0:
            pos_np = np.expand_dims(pos_np, axis=0).repeat(n_envs, axis=0)
            quat_np = np.expand_dims(quat_np, axis=0).repeat(n_envs, axis=0)
        robot.set_pos(pos_np, zero_velocity=False)
        robot.set_quat(quat_np, zero_velocity=False)

    apply_pose_to_all_envs(target_pos, euler_to_quat(target_euler))

    try:
        while True:
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

            apply_pose_to_all_envs(target_pos, euler_to_quat(target_euler))

            scene.step()

    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")


def main():
    parser = argparse.ArgumentParser(description="Genesis LiDAR/Depth Camera Visualization with Keyboard Teleop")
    parser.add_argument("-B", "--n_envs", type=int, default=0, help="Number of environments to replicate")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of GPU")
    parser.add_argument("--use-box", action="store_true", help="Use Box as robot instead of Go2")
    parser.add_argument(
        "-f",
        "--fixed",
        action="store_true",
        help="Load obstacles as fixed and cast only against fixed objects (is_free=False)",
        default=True,
    )
    parser.add_argument(
        "-nf",
        "--no-fixed",
        action="store_false",
        dest="fixed",
        help="Load obstacles as dynamic (is_free=True), raycaster will update BVH every step",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="spinning",
        choices=["spherical", "spinning", "depth", "grid"],
        help="Sensor pattern type",
    )

    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="info")

    kb = KeyboardDevice()
    kb.start()

    scene = build_scene(show_viewer=True, is_free=not args.fixed)
    robot, lidar = create_robot_with_lidar(scene, args)

    run(scene, robot, lidar, n_envs=args.n_envs, kb=kb, is_depth=args.pattern == "depth")


if __name__ == "__main__":
    main()
