"""
Interactive KinematicContactProbe visualization with keyboard teleop.

A platform with a grid of contact probes sits in the scene.
Use keyboard controls to move the "pusher" cylinder across the probe surface and push around objects.
"""

import argparse
import os

import numpy as np

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind

# Teleop
KEY_DPOS = 0.05
PUSHER_SIZE = 0.1

# Probe sensors
GRID_SIZE = 5
PROBE_RADIUS = 0.05

# Objects
PLATFORM_SIZE = 1.5
PLATFORM_HEIGHT = 0.3
OBJ_Z = PLATFORM_HEIGHT * 1.4
OBJ_SIZE = PLATFORM_SIZE / 8.0


def _build_probe_grid(grid_n: int, platform_size: float, platform_height: float):
    spacing = platform_size / (grid_n + 1)
    centre = (grid_n - 1) / 2.0

    i = np.repeat(np.arange(grid_n), grid_n)
    j = np.tile(np.arange(grid_n), grid_n)
    x = (i - centre) * spacing
    y = (j - centre) * spacing
    z = np.full_like(x, platform_height / 2)  # top surface in link-local frame
    positions = np.stack([x, y, z], axis=-1)
    normals = np.tile([0.0, 0.0, 1.0], (grid_n * grid_n, 1))
    radii = PROBE_RADIUS + i * (PROBE_RADIUS / 10.0)

    return positions, normals, radii


def main():
    parser = argparse.ArgumentParser(description="Interactive KinematicContactProbe Visualization")
    parser.add_argument("--vis", "-v", action="store_true", default=False, help="Show visualization GUI")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of GPU")
    parser.add_argument("--seconds", "-t", type=float, default=3.0, help="Seconds to simulate (headless mode)")
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="info")

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(-PLATFORM_SIZE * 2, 0.0, PLATFORM_HEIGHT + 1.5),
            camera_lookat=(0.0, 0.0, PLATFORM_HEIGHT),
            max_FPS=60,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    platform = scene.add_entity(
        gs.morphs.Box(
            size=(PLATFORM_SIZE, PLATFORM_SIZE, PLATFORM_HEIGHT),
            pos=(0.0, 0.0, PLATFORM_HEIGHT / 2),
            fixed=True,
        ),
    )

    probe_positions, probe_normals, probe_radii = _build_probe_grid(GRID_SIZE, PLATFORM_SIZE, PLATFORM_HEIGHT)
    n_probes = len(probe_positions)

    probe = scene.add_sensor(
        gs.sensors.KinematicContactProbe(
            entity_idx=platform.idx,
            link_idx_local=0,
            probe_local_pos=probe_positions,
            probe_local_normal=probe_normals,
            radius=probe_radii,
            stiffness=5000.0,
            draw_debug=args.vis,
        )
    )

    pusher_start = np.array([0.0, 0.0, PLATFORM_HEIGHT + PUSHER_SIZE / 2 - 0.02], dtype=np.float32)

    pusher = scene.add_entity(
        gs.morphs.Cylinder(
            radius=PUSHER_SIZE,
            height=PUSHER_SIZE,
            pos=pusher_start,
        ),
        surface=gs.surfaces.Default(
            color=(0.15, 0.55, 0.95, 1.0),
        ),
    )

    # Add objects
    rect = scene.add_entity(
        gs.morphs.Box(
            size=(OBJ_SIZE, OBJ_SIZE * 2, OBJ_SIZE),
            pos=(PLATFORM_SIZE / 4, 0, OBJ_Z),
        ),
        surface=gs.surfaces.Default(color=(1.0, 0.3, 0.3, 1.0)),
    )
    cylinder = scene.add_entity(
        gs.morphs.Cylinder(
            radius=OBJ_SIZE / 2,
            height=OBJ_SIZE * 1.2,
            pos=(0, PLATFORM_SIZE / 4, OBJ_Z),
        ),
        surface=gs.surfaces.Default(color=(0.3, 1.0, 0.3, 1.0)),
    )
    sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=OBJ_SIZE / 2,
            pos=(-PLATFORM_SIZE / 4, -PLATFORM_SIZE / 4, OBJ_Z),
        ),
        surface=gs.surfaces.Default(color=(0.3, 0.3, 1.0, 1.0)),
    )
    objects = [rect, cylinder, sphere]

    scene.build()

    is_running = True
    # Register keybindings
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

    # ── Print info ─────────────────────────────────────────────────────
    print("\n=== Interactive KinematicContactProbe ===")
    print(f"Platform {PLATFORM_SIZE}m × {PLATFORM_SIZE}m with {GRID_SIZE}×{GRID_SIZE} probes ({n_probes} total)")
    print(f"Probe radii range: {min(probe_radii):.4f} – {max(probe_radii):.4f} m")
    if args.vis:
        print()
        print("Keyboard Controls:")
        print("  [↑/↓/←/→]  Move pusher box in XY")
        print("  [j / k]     Lower / raise pusher box")
        print("  [SPACE]     Drop an object at pusher location")
        print("  [\\]         Reset pusher position")
    else:
        print(f"Running headless for {args.seconds}s ...")
    print()

    # Simulation loop
    steps = int(args.seconds / scene.sim_options.dt) if not args.vis else None
    step = 0

    try:
        while is_running:
            if args.vis:
                pusher.set_pos(target_pos)
                pusher.set_quat(np.array([1, 0, 0, 0], dtype=np.float32))

            scene.step()

            # Read probe data and print any active contacts
            data = probe.read()
            active = (data.penetration > 0).nonzero(as_tuple=False)
            if active.numel() > 0:
                idxs = active.squeeze(-1).tolist()
                if isinstance(idxs, int):
                    idxs = [idxs]
                depths = data.penetration[active.squeeze(-1)].tolist()
                if isinstance(depths, float):
                    depths = [depths]
                parts = [f"probe{i}={d:.4f}" for i, d in zip(idxs, depths)]
                print(f"Step {step}: Contact: {', '.join(parts)}")

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
