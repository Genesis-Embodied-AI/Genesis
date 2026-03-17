"""
Interactive ElastomerDisplacementSensor visualization with keyboard teleop.
"""

import argparse
import os

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.recorders.plotters import IS_MATPLOTLIB_AVAILABLE
from genesis.utils.misc import tensor_to_array
from genesis.vis.keybindings import Key, KeyAction, Keybind

# Teleop
KEY_DPOS = 0.08
FORCE_SCALE = 100.0

# Pusher (sphere with tactile on bottom hemisphere, or box with grid sensor on bottom face)
PUSHER_SIZE = 0.1
PROBE_RADIUS = 0.01
DILATE_COEFFICIENT = 1e1
SHEAR_COEFFICIENT = 1e-2
TWIST_COEFFICIENT = 1e-2
HEMISPHERE_N_THETA = 4
HEMISPHERE_N_PHI = 12
GRID_SIZE = (6, 8)  # (nx, ny) for --grid

# Sandbox
SANDBOX_SIZE = 1.2
WALL_THICKNESS = 0.08
WALL_HEIGHT = 0.25

# Objects inside sandbox
OBJ_SIZE = 0.08


def main():
    parser = argparse.ArgumentParser(description="Interactive ElastomerDisplacementSensor Visualization")
    parser.add_argument("--vis", "-v", action="store_true", default=False, help="Show visualization GUI")
    parser.add_argument("--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("--seconds", "-t", type=float, default=3.0, help="Seconds to simulate (headless mode)")
    parser.add_argument("--grid", action="store_true", help="Use grid of probes instead of hemisphere probes")
    args = parser.parse_args()

    gs.init(
        backend=gs.gpu if args.gpu else gs.cpu,
        precision="32",
        logging_level="info",
    )

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(-SANDBOX_SIZE * 1.8, 0.0, WALL_HEIGHT + 1.5),
            camera_lookat=(0.0, 0.0, WALL_HEIGHT / 2),
            max_FPS=60,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=True,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    # Sandbox: four fixed walls
    for i in range(4):
        scene.add_entity(
            gs.morphs.Box(
                size=(SANDBOX_SIZE + 2 * WALL_THICKNESS, WALL_THICKNESS, WALL_HEIGHT),
                pos=(
                    *((1 - 2 * (i % 2)) * (SANDBOX_SIZE / 2 + WALL_THICKNESS / 2), 0.0)[:: (2 * (i >= 2) - 1)],
                    WALL_HEIGHT / 2,
                ),
                euler=(0, 0, 90 * (i >= 2)),
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=(0.5, 0.45, 0.4, 1.0)),
        )

    # Controllable pusher: sphere with hemisphere tactile, or box with grid tactile
    sensor_kwargs = dict(
        link_idx_local=0,
        probe_local_normal=(0.0, 0.0, -1.0),
        probe_radius=PROBE_RADIUS,
        draw_debug=args.vis,
        dilate_coefficient=DILATE_COEFFICIENT,
        shear_coefficient=SHEAR_COEFFICIENT,
        twist_coefficient=TWIST_COEFFICIENT,
    )
    if args.grid:
        pusher_pos_init = np.array([0.2, -0.3, PUSHER_SIZE / 2 + OBJ_SIZE], dtype=np.float32)
        pusher = scene.add_entity(
            gs.morphs.Box(
                size=(PUSHER_SIZE, PUSHER_SIZE, PUSHER_SIZE / 2),
                pos=pusher_pos_init,
            ),
            surface=gs.surfaces.Default(
                color=(0.15, 0.55, 0.95, 1.0),
            ),
        )
        tactile = scene.add_sensor(
            gs.sensors.ElastomerDisplacement(
                entity_idx=pusher.idx,
                probe_local_pos=gu.generate_grid_points_on_plane(
                    lo=[-PUSHER_SIZE / 2, -PUSHER_SIZE / 2, -PUSHER_SIZE / 4],
                    hi=[PUSHER_SIZE / 2, PUSHER_SIZE / 2, -PUSHER_SIZE / 4],
                    normal=(0.0, 0.0, -1.0),
                    nx=GRID_SIZE[0],
                    ny=GRID_SIZE[1],
                ),
                **sensor_kwargs,
            )
        )
    else:
        pusher_pos_init = np.array([0.2, -0.3, PUSHER_SIZE + 0.01], dtype=np.float32)
        pusher = scene.add_entity(
            gs.morphs.Sphere(
                radius=PUSHER_SIZE,
                pos=pusher_pos_init,
            ),
            surface=gs.surfaces.Default(
                color=(0.15, 0.55, 0.95, 1.0),
            ),
        )

        theta = np.linspace(np.pi / 2, np.pi, HEMISPHERE_N_THETA, endpoint=False)
        phi = np.linspace(0, 2 * np.pi, HEMISPHERE_N_PHI, endpoint=False)
        theta, phi = map(np.ravel, np.meshgrid(theta, phi, indexing="ij"))
        x, y, z = np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)
        probe_positions = PUSHER_SIZE * np.stack([x, y, z], axis=-1, dtype=gs.np_float)

        tactile = scene.add_sensor(
            gs.sensors.ElastomerDisplacement(
                entity_idx=pusher.idx,
                probe_local_pos=probe_positions,
                **sensor_kwargs,
            )
        )

    if args.vis:
        if IS_MATPLOTLIB_AVAILABLE:
            plot_normal = (0.0, 0.0, -1.0) if args.grid else (0.0, 0.0, 1.0)
            tactile.start_recording(
                rec_options=gs.recorders.MPLVectorFieldPlot(
                    title="Tactile Displacement",
                    positions=tactile.probe_local_pos,
                    normal=plot_normal,
                    scale_factor=4.0,
                    max_magnitude=1.0e-1,
                ),
            )

    # Colored objects inside the sandbox (sitting on floor)
    margin = SANDBOX_SIZE / 2 - OBJ_SIZE
    rect = scene.add_entity(
        gs.morphs.Box(
            size=(OBJ_SIZE, OBJ_SIZE * 1.5, OBJ_SIZE),
            pos=(margin * 0.5, 0.0, OBJ_SIZE / 2),
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.3, 0.3, 1.0),
        ),
    )
    cylinder = scene.add_entity(
        gs.morphs.Cylinder(
            radius=OBJ_SIZE / 2,
            height=OBJ_SIZE * 1.2,
            pos=(0.0, margin * 0.5, OBJ_SIZE / 2),
        ),
        surface=gs.surfaces.Default(
            color=(0.3, 1.0, 0.3, 1.0),
        ),
    )
    small_sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=OBJ_SIZE / 2,
            pos=(0.0, 0.0, OBJ_SIZE / 2),
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.9, 0.0, 1.0),
        ),
    )
    big_sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=OBJ_SIZE,
            pos=(-margin * 0.4, -margin * 0.4, OBJ_SIZE),
        ),
        surface=gs.surfaces.Default(
            color=(0.3, 0.3, 1.0, 1.0),
        ),
    )
    objects = (rect, cylinder, small_sphere, big_sphere)

    scene.build()

    if args.grid:
        pusher.set_dofs_kp(FORCE_SCALE / KEY_DPOS)
        pusher.set_dofs_kv(0.1 * FORCE_SCALE / KEY_DPOS)
    else:
        pusher.set_dofs_kp(FORCE_SCALE / KEY_DPOS, dofs_idx_local=slice(0, 3))
        pusher.set_dofs_kv(0.1 * FORCE_SCALE / KEY_DPOS, dofs_idx_local=slice(0, 3))
        pusher.set_dofs_kv(1.0, dofs_idx_local=slice(3, 6))
    pusher.control_dofs_position(pusher.get_dofs_position())

    # Register keybindings
    is_running = True
    if args.vis:
        target_pos = pusher_pos_init.copy()
        obj_idx = 0

        def stop():
            nonlocal is_running
            is_running = False

        def reset_pose():
            target_pos[:] = pusher_pos_init
            pusher.set_dofs_position(pusher_pos_init, dofs_idx_local=slice(0, 3))

        def translate(index: int, is_negative: bool):
            target_pos_i = target_pos[index] + (-1 if is_negative else 1) * KEY_DPOS
            if index == 2:
                target_pos_i = np.clip(target_pos_i, 0.0, WALL_HEIGHT)
            target_pos[index] = target_pos_i

        def drop_object():
            nonlocal obj_idx
            drop_pos = pusher.get_pos()
            drop_pos[2] += 0.2 * (obj_idx + 1)
            objects[obj_idx].set_pos(drop_pos)
            objects[obj_idx].set_quat(np.array([1, 0, 0, 0], dtype=np.float32))
            obj_idx = (obj_idx + 1) % len(objects)

        scene.viewer.register_keybinds(
            Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=translate, args=(0, False)),
            Keybind("move_backward", Key.DOWN, KeyAction.HOLD, callback=translate, args=(0, True)),
            Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=translate, args=(1, True)),
            Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=translate, args=(1, False)),
            Keybind("move_down", Key.J, KeyAction.HOLD, callback=translate, args=(2, True)),
            Keybind("move_up", Key.K, KeyAction.HOLD, callback=translate, args=(2, False)),
            Keybind("drop_object", Key.SPACE, KeyAction.RELEASE, callback=drop_object),
            Keybind("reset", Key.BACKSPACE, KeyAction.RELEASE, callback=reset_pose),
            Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        )

    print("\n=== Interactive ElastomerDisplacementSensor ===")
    print(f"Sandbox {SANDBOX_SIZE}m × {SANDBOX_SIZE}m; pusher box with {tactile.n_probes} probes")

    if args.vis:
        if IS_MATPLOTLIB_AVAILABLE:
            print("Live vector field plot: tactile displacement (2D projection, color = magnitude)")
        print()
        print("Keyboard Controls:")
        print("  [↑/↓/←/→]  Move pusher in XY")
        print("  [j / k]     Lower / raise pusher")
        print("  [SPACE]     Drop an object at pusher location")
        print("  [\\]         Reset pusher position")
    else:
        print(f"Running headless for {args.seconds}s ...")
    print()

    # Simulation loop
    try:
        while is_running:
            if args.vis:
                pusher.control_dofs_position(target_pos, dofs_idx_local=slice(0, 3))
            else:
                print(tactile.read())

            scene.step()

            if args.vis:
                cur_pos = tensor_to_array(pusher.get_pos())
                target_pos[:] = np.clip(target_pos - cur_pos, -KEY_DPOS, KEY_DPOS) + cur_pos

            if "PYTEST_VERSION" in os.environ:
                break
            if not args.vis and scene.t > args.seconds:
                break
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted.")
    finally:
        gs.logger.info("Simulation finished.")


if __name__ == "__main__":
    main()
