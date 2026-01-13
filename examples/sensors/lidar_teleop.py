import argparse
import os

import numpy as np

import genesis as gs
from genesis.ext.pyrender.interaction.keybindings import KeyAction, Keybind
from genesis.utils.geom import euler_to_quat

# Position and angle increments for keyboard teleop control
KEY_DPOS = 0.1
KEY_DANGLE = 0.1

# Number of obstacles to create in a ring around the robot
NUM_CYLINDERS = 8
NUM_BOXES = 6
CYLINDER_RING_RADIUS = 3.0
BOX_RING_RADIUS = 5.0


def main():
    parser = argparse.ArgumentParser(description="Genesis LiDAR/Depth Camera Visualization with Keyboard Teleop")
    parser.add_argument("-B", "--n_envs", type=int, default=0, help="Number of environments to replicate")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of GPU")
    parser.add_argument("--use-box", action="store_true", help="Use Box as robot instead of Go2")
    parser.add_argument(
        "--pattern", type=str, default="spherical", choices=("spherical", "depth", "grid"), help="Sensor pattern type"
    )
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="info")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, -1.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(-6.0, 0.0, 4.0),
            camera_lookat=(0.0, 0.0, 0.5),
            max_FPS=60,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
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
                fixed=True,
            )
        )

    for i in range(NUM_BOXES):
        angle = 2 * np.pi * i / NUM_BOXES + np.pi / 6
        x = BOX_RING_RADIUS * np.cos(angle)
        y = BOX_RING_RADIUS * np.sin(angle)
        scene.add_entity(
            gs.morphs.Box(
                size=(0.5, 0.5, 2.0 * (i + 1) / NUM_BOXES),
                pos=(x, y, 1.0),
                fixed=False,
            )
        )

    entity_kwargs = dict(
        pos=(0.0, 0.0, 0.35),
        quat=(1.0, 0.0, 0.0, 0.0),
        fixed=True,
    )

    if args.use_box:
        robot = scene.add_entity(
            gs.morphs.Box(
                size=(0.1, 0.1, 0.1),
                **entity_kwargs,
            )
        )
        pos_offset = (0.0, 0.0, 0.2)
    else:
        robot = scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                **entity_kwargs,
            )
        )
        pos_offset = (0.3, 0.0, 0.1)

    sensor_kwargs = dict(
        entity_idx=robot.idx,
        pos_offset=pos_offset,
        euler_offset=(0.0, 0.0, 0.0),
        return_world_frame=True,
        draw_debug=True,
    )

    if args.pattern == "depth":
        sensor = scene.add_sensor(gs.sensors.DepthCamera(pattern=gs.sensors.DepthCameraPattern(), **sensor_kwargs))
        scene.start_recording(
            data_func=(lambda: sensor.read_image()[0]) if args.n_envs > 0 else sensor.read_image,
            rec_options=gs.recorders.MPLImagePlot(),
        )
    else:
        if args.pattern == "grid":
            pattern_cfg = gs.sensors.GridPattern()
        else:
            if args.pattern != "spherical":
                gs.logger.warning(f"Unrecognized raycaster pattern: {args.pattern}. Using 'spherical' instead.")
            pattern_cfg = gs.sensors.SphericalPattern()

        sensor = scene.add_sensor(gs.sensors.Lidar(pattern=pattern_cfg, **sensor_kwargs))

    scene.build(n_envs=args.n_envs)

    # Initialize pose state
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

    # Define control callbacks
    def reset_pose():
        target_pos[:] = init_pos
        target_euler[:] = init_euler

    def move_forward():
        target_pos[0] += KEY_DPOS

    def move_backward():
        target_pos[0] -= KEY_DPOS

    def move_right():
        target_pos[1] -= KEY_DPOS

    def move_left():
        target_pos[1] += KEY_DPOS

    def move_down():
        target_pos[2] -= KEY_DPOS

    def move_up():
        target_pos[2] += KEY_DPOS

    def roll_ccw():
        target_euler[0] += KEY_DANGLE

    def roll_cw():
        target_euler[0] -= KEY_DANGLE

    def pitch_up():
        target_euler[1] += KEY_DANGLE

    def pitch_down():
        target_euler[1] -= KEY_DANGLE

    def yaw_ccw():
        target_euler[2] += KEY_DANGLE

    def yaw_cw():
        target_euler[2] -= KEY_DANGLE

    # Register keybindings
    from pyglet.window import key

    scene.viewer.register_keybinds(
        Keybind(key_code=key.UP, key_action=KeyAction.HOLD, name="move_forward", callback=move_forward),
        Keybind(key_code=key.DOWN, key_action=KeyAction.HOLD, name="move_backward", callback=move_backward),
        Keybind(key_code=key.RIGHT, key_action=KeyAction.HOLD, name="move_right", callback=move_right),
        Keybind(key_code=key.LEFT, key_action=KeyAction.HOLD, name="move_left", callback=move_left),
        Keybind(key_code=key.J, key_action=KeyAction.HOLD, name="move_down", callback=move_down),
        Keybind(key_code=key.K, key_action=KeyAction.HOLD, name="move_up", callback=move_up),
        Keybind(key_code=key.N, key_action=KeyAction.HOLD, name="roll_ccw", callback=roll_ccw),
        Keybind(key_code=key.M, key_action=KeyAction.HOLD, name="roll_cw", callback=roll_cw),
        Keybind(key_code=key.COMMA, key_action=KeyAction.HOLD, name="pitch_up", callback=pitch_up),
        Keybind(key_code=key.PERIOD, key_action=KeyAction.HOLD, name="pitch_down", callback=pitch_down),
        Keybind(key_code=key.O, key_action=KeyAction.HOLD, name="yaw_ccw", callback=yaw_ccw),
        Keybind(key_code=key.P, key_action=KeyAction.HOLD, name="yaw_cw", callback=yaw_cw),
        Keybind(key_code=key.BACKSLASH, key_action=KeyAction.HOLD, name="reset", callback=reset_pose),
    )

    # Print controls
    print("Keyboard Controls:")
    print("[↑/↓/←/→]: Move XY")
    print("[j/k]: Down/Up")
    print("[n/m]: Roll CCW/CW")
    print("[,/.]: Pitch Up/Down")
    print("[o/p]: Yaw CCW/CW")
    print("[\\]: Reset")

    apply_pose_to_all_envs(target_pos, euler_to_quat(target_euler))

    try:
        while True:
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
