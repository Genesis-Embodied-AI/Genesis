"""
Interactive TemperatureGrid sensor visualization with keyboard teleop.

A platform has a temperature grid sensor on its top surface. Move a "hot" pusher
and drop objects onto the platform; the grid shows temperature (blue=cool, red=hot)
from contact-based blending of each body's base_temperature and conductivity.
"""

import argparse
import os

import numpy as np

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind

# Teleop
KEY_DPOS = 0.05
PUSHER_SIZE = 0.1

# Temperature grid
GRID_SIZE = (10, 10, 1)

# Objects
PLATFORM_SIZE = 1.5
PLATFORM_HEIGHT = 0.3
OBJ_Z = PLATFORM_HEIGHT * 1.4
OBJ_SIZE = 0.1


def main():
    parser = argparse.ArgumentParser(description="Interactive TemperatureGrid sensor visualization")
    parser.add_argument("--vis", "-v", action="store_true", default=False, help="Show visualization GUI")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of GPU")
    parser.add_argument("--seconds", "-t", type=float, default=3.0, help="Seconds to simulate (headless mode)")
    parser.add_argument("--simulate-all-links", "-l", action="store_true", help="Simulate all link temperatures")
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="info")

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(-PLATFORM_SIZE * 2, 0.0, PLATFORM_HEIGHT + 1.5),
            camera_lookat=(0.0, 0.0, PLATFORM_HEIGHT),
            max_FPS=60,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    platform = scene.add_entity(
        gs.morphs.Box(
            size=(PLATFORM_SIZE, PLATFORM_SIZE, PLATFORM_HEIGHT),
            pos=(0.0, 0.0, PLATFORM_HEIGHT / 2),
            fixed=True,
            visualization=False,  # sensor debug_draw will be shown
        ),
    )

    pusher_start = np.array([0.0, 0.0, PLATFORM_HEIGHT + PUSHER_SIZE / 2 - 0.02], dtype=np.float32)
    pusher = scene.add_entity(
        gs.morphs.Cylinder(
            radius=PUSHER_SIZE,
            height=PUSHER_SIZE,
            pos=pusher_start,
        ),
        surface=gs.surfaces.Default(color=(1.0, 0.2, 0.2, 1.0)),
    )

    rect = scene.add_entity(
        gs.morphs.Box(
            size=(OBJ_SIZE, OBJ_SIZE * 2, OBJ_SIZE),
            pos=(PLATFORM_SIZE / 4, 0, OBJ_Z),
        ),
    )
    cylinder = scene.add_entity(
        gs.morphs.Cylinder(
            radius=OBJ_SIZE / 2,
            height=OBJ_SIZE * 1.2,
            pos=(0, PLATFORM_SIZE / 4, OBJ_Z),
        ),
    )
    sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=OBJ_SIZE / 2,
            pos=(-PLATFORM_SIZE / 4, -PLATFORM_SIZE / 4, OBJ_Z),
        ),
    )
    objects = [rect, cylinder, sphere]

    # Build properties_dict: one TemperatureProperties per link (required for contact blending).
    # Platform is room temp; pusher is hot; objects are warm; ground plane is cool.
    properties_dict = {
        -1: gs.sensors.TemperatureProperties(
            base_temperature=-40.0,
            conductivity=200.0,
            density=2000.0,
            specific_heat=1.0,
            emissivity=0.85,
        ),
        platform.base_link_idx: gs.sensors.TemperatureProperties(
            base_temperature=22.0,
            conductivity=100.0,
            density=1000.0,
            specific_heat=1.0,
            emissivity=0.9,
        ),
        pusher.base_link_idx: gs.sensors.TemperatureProperties(
            base_temperature=80.0,
            conductivity=600.0,
            density=2000.0,
            specific_heat=1.0,
            emissivity=0.8,
        ),
    }

    temperature_sensor = scene.add_sensor(
        gs.sensors.TemperatureGrid(
            entity_idx=platform.idx,
            link_idx_local=0,
            grid_size=GRID_SIZE,
            properties_dict=properties_dict,
            draw_debug=args.vis,
            debug_temperature_range=(0.0, 100.0),
            simulate_all_link_temperatures=args.simulate_all_links,
            ambient_temperature=22.0,
            convection_coefficient=0.0,
        )
    )

    scene.build()

    is_running = True
    if args.vis:
        target_pos = pusher_start.copy()
        next_obj_idx = 0

        def stop():
            nonlocal is_running
            is_running = False

        def reset_pose():
            target_pos[:] = pusher_start

        def translate(index: int, is_negative: bool):
            target_pos[index] += (-1 if is_negative else 1) * KEY_DPOS

        def drop_object():
            nonlocal next_obj_idx
            idx = next_obj_idx % len(objects)
            drop_pos = target_pos.copy()
            drop_pos[2] = PLATFORM_HEIGHT * 2
            objects[idx].set_pos(drop_pos)
            objects[idx].set_quat(np.array([1, 0, 0, 0], dtype=np.float32))
            next_obj_idx += 1

        scene.viewer.register_keybinds(
            Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=translate, args=(0, False)),
            Keybind("move_backward", Key.DOWN, KeyAction.HOLD, callback=translate, args=(0, True)),
            Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=translate, args=(1, True)),
            Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=translate, args=(1, False)),
            Keybind("move_down", Key.J, KeyAction.HOLD, callback=translate, args=(2, True)),
            Keybind("move_up", Key.K, KeyAction.HOLD, callback=translate, args=(2, False)),
            Keybind("drop_object", Key.SPACE, KeyAction.PRESS, callback=drop_object),
            Keybind("reset", Key.BACKSLASH, KeyAction.PRESS, callback=reset_pose),
            Keybind("quit", Key.ESCAPE, KeyAction.PRESS, callback=stop),
        )

    print("\n=== Interactive TemperatureGrid ===")
    print(f"Platform {PLATFORM_SIZE}m × {PLATFORM_SIZE}m with grid {GRID_SIZE}")
    if args.vis:
        print()
        print("Keyboard Controls:")
        print("  [↑/↓/←/→]  Move pusher (hot) in XY")
        print("  [j / k]     Lower / raise pusher")
        print("  [SPACE]     Drop an object at pusher location")
        print("  [\\]         Reset pusher position")
    else:
        print(f"Running headless for {args.seconds}s ...")
    print()

    steps = int(args.seconds / scene.sim_options.dt) if not args.vis else None
    step = 0

    try:
        while is_running:
            if args.vis:
                pusher.set_pos(target_pos)
                pusher.set_quat(np.array([1, 0, 0, 0], dtype=np.float32))

            scene.step()

            data = temperature_sensor.read()
            t_min, t_max = float(data.min()), float(data.max())
            if step % 100 == 0:
                print(
                    f"step={step}, time={step * scene.sim_options.dt:.2f}s: "
                    f"Grid temperature range [{t_min:.1f}, {t_max:.1f}] °C"
                )
                if args.simulate_all_links:
                    print(f"Link temperatures: {temperature_sensor.link_temperatures}")

            step += 1
            if "PYTEST_VERSION" in os.environ:
                break
            if not args.vis and step >= steps:
                break
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted.")
    finally:
        gs.logger.info("Simulation finished.")


if __name__ == "__main__":
    main()
