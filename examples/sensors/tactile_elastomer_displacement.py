"""
Interactive ElastomerDisplacementSensor visualization with keyboard teleop.
"""

import argparse
import os

import numpy as np

import genesis as gs
from genesis.recorders.plotters import IS_MATPLOTLIB_AVAILABLE
from genesis.vis.keybindings import Key, KeyAction, Keybind

# Teleop
KEY_DPOS = 0.01

# Pusher (sphere with tactile on bottom hemisphere, or box with grid sensor on bottom face)
PUSHER_SIZE = 0.1
PROBE_RADIUS = 0.01
HEMISPHERE_N_THETA = 4
HEMISPHERE_N_PHI = 12
GRID_SIZE = (6, 8)  # (nx, ny) for --grid

# Sandbox
SANDBOX_SIZE = 1.2
WALL_THICKNESS = 0.08
WALL_HEIGHT = 0.25

# Objects inside sandbox
OBJ_SIZE = 0.08


def _build_hemisphere_probes(radius: float, n_theta: int, n_phi: int):
    """Probe positions and outward normals on the bottom hemisphere (z <= 0 in link frame)."""
    theta = np.linspace(np.pi / 2, np.pi, n_theta, endpoint=False)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    theta, phi = np.meshgrid(theta, phi, indexing="ij")
    theta = theta.ravel()
    phi = phi.ravel()
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    positions = np.stack([x, y, z], axis=-1)
    normals = positions / radius
    return positions.astype(np.float32), normals.astype(np.float32)


def _build_grid_probe_positions(bounds: tuple, grid_size: tuple[int, int]) -> np.ndarray:
    """Probe positions on a 2D grid (row-major iy, ix) for plotting. Same layout as ElastomerDisplacementGridSensor."""
    lo, hi = np.array(bounds[0]), np.array(bounds[1])
    nx, ny = grid_size[0], grid_size[1]
    dx = (hi[0] - lo[0]) / (nx - 1) if nx > 1 else 0.0
    dy = (hi[1] - lo[1]) / (ny - 1) if ny > 1 else 0.0
    positions = []
    for iy in range(ny):
        for ix in range(nx):
            positions.append((lo[0] + ix * dx, lo[1] + iy * dy, lo[2]))
    return np.array(positions, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Interactive ElastomerDisplacementSensor Visualization")
    parser.add_argument("--vis", "-v", action="store_true", default=False, help="Show visualization GUI")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of GPU")
    parser.add_argument("--seconds", "-t", type=float, default=3.0, help="Seconds to simulate (headless mode)")
    parser.add_argument("--grid", action="store_true", help="Use box pusher with ElastomerDisplacementGridSensor")
    args = parser.parse_args()

    gs.init(
        backend=gs.cpu if args.cpu else gs.gpu,
        precision="32",
        logging_level="info",
        debug=True,
    )

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(-SANDBOX_SIZE * 1.8, 0.0, WALL_HEIGHT + 0.8),
            camera_lookat=(0.0, 0.0, WALL_HEIGHT / 2),
            max_FPS=60,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=True,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    # Sandbox: four fixed walls (no top or bottom)
    half = SANDBOX_SIZE / 2
    t = WALL_THICKNESS
    wall_surface = gs.surfaces.Default(color=(0.5, 0.45, 0.4, 1.0))
    scene.add_entity(
        gs.morphs.Box(
            size=(SANDBOX_SIZE + 2 * t, t, WALL_HEIGHT),
            pos=(0.0, half + t / 2, WALL_HEIGHT / 2),
            fixed=True,
        ),
        surface=wall_surface,
    )
    scene.add_entity(
        gs.morphs.Box(
            size=(SANDBOX_SIZE + 2 * t, t, WALL_HEIGHT),
            pos=(0.0, -half - t / 2, WALL_HEIGHT / 2),
            fixed=True,
        ),
        surface=wall_surface,
    )
    scene.add_entity(
        gs.morphs.Box(
            size=(t, SANDBOX_SIZE, WALL_HEIGHT),
            pos=(half + t / 2, 0.0, WALL_HEIGHT / 2),
            fixed=True,
        ),
        surface=wall_surface,
    )
    scene.add_entity(
        gs.morphs.Box(
            size=(t, SANDBOX_SIZE, WALL_HEIGHT),
            pos=(-half - t / 2, 0.0, WALL_HEIGHT / 2),
            fixed=True,
        ),
        surface=wall_surface,
    )

    # Controllable pusher: sphere with hemisphere tactile, or box with grid tactile
    if args.grid:
        box_size = np.array((PUSHER_SIZE, PUSHER_SIZE, PUSHER_SIZE / 2), dtype=np.float32)
        pusher_start = np.array([0.0, 0.0, box_size[2] / 2 + 0.01], dtype=np.float32)
        pusher = scene.add_entity(
            gs.morphs.Box(size=box_size, pos=pusher_start),
            surface=gs.surfaces.Default(color=(0.15, 0.55, 0.95, 1.0)),
        )
        half_xy = box_size[:2] / 2
        grid_bounds = [
            [-float(half_xy[0]), -float(half_xy[1]), -float(box_size[2]) / 2],
            [float(half_xy[0]), float(half_xy[1]), -float(box_size[2]) / 2],
        ]
        tactile = scene.add_sensor(
            gs.sensors.ElastomerDisplacementGridSensor(
                entity_idx=pusher.idx,
                link_idx_local=0,
                probe_local_pos_grid_bounds=grid_bounds,
                probe_grid_size=GRID_SIZE,
                probe_local_normal=(0.0, 0.0, -1.0),
                radius=PROBE_RADIUS,
                draw_debug=args.vis,
            )
        )
        probe_positions = _build_grid_probe_positions(grid_bounds, GRID_SIZE)
        n_probes = len(probe_positions)
    else:
        pusher_start = np.array([0.0, 0.0, PUSHER_SIZE + 0.01], dtype=np.float32)
        pusher = scene.add_entity(
            gs.morphs.Sphere(radius=PUSHER_SIZE, pos=pusher_start),
            surface=gs.surfaces.Default(color=(0.15, 0.55, 0.95, 1.0)),
        )
        probe_positions, probe_normals = _build_hemisphere_probes(PUSHER_SIZE, HEMISPHERE_N_THETA, HEMISPHERE_N_PHI)
        n_probes = len(probe_positions)
        tactile = scene.add_sensor(
            gs.sensors.ElastomerDisplacementSensor(
                entity_idx=pusher.idx,
                link_idx_local=0,
                probe_local_pos=probe_positions,
                probe_local_normal=probe_normals,
                radius=PROBE_RADIUS,
                draw_debug=args.vis,
            )
        )

    if args.vis:
        scene.viewer.add_plugin(
            gs.vis.viewer_plugins.MouseInteractionPlugin(
                use_force=True,
            )
        )
        if IS_MATPLOTLIB_AVAILABLE:
            plot_normal = (0.0, 0.0, -1.0) if args.grid else (0.0, 0.0, 1.0)
            scene.start_recording(
                data_func=lambda: tactile.read(),
                rec_options=gs.recorders.MPLVectorFieldPlot(
                    title="Tactile Displacement",
                    positions=probe_positions,
                    normal=plot_normal,
                    scale_factor=10.0,
                    max_magnitude=1.0e-2,
                ),
            )

    # Colored objects inside the sandbox (sitting on floor)
    floor_z = OBJ_SIZE / 2
    margin = half - OBJ_SIZE
    rect = scene.add_entity(
        gs.morphs.Box(
            size=(OBJ_SIZE, OBJ_SIZE * 1.5, OBJ_SIZE),
            pos=(margin * 0.5, 0.0, floor_z),
        ),
        surface=gs.surfaces.Default(color=(1.0, 0.3, 0.3, 1.0)),
    )
    cylinder = scene.add_entity(
        gs.morphs.Cylinder(
            radius=OBJ_SIZE / 2,
            height=OBJ_SIZE * 1.2,
            pos=(0.0, margin * 0.5, floor_z),
        ),
        surface=gs.surfaces.Default(color=(0.3, 1.0, 0.3, 1.0)),
    )
    small_sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=OBJ_SIZE / 2,
            pos=(-margin * 0.4, -margin * 0.4, floor_z),
        ),
        surface=gs.surfaces.Default(color=(0.3, 0.3, 1.0, 1.0)),
    )
    objects = [rect, cylinder, small_sphere]

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
            drop_pos[2] = WALL_HEIGHT + 0.5
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
    print("\n=== Interactive ElastomerDisplacementSensor ===")
    print(f"Sandbox {SANDBOX_SIZE}m × {SANDBOX_SIZE}m; pusher box with {n_probes} probes")

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
    steps = int(args.seconds / scene.sim_options.dt) if not args.vis else None
    step = 0

    try:
        while is_running:
            if args.vis:
                pusher.set_pos(target_pos)
                pusher.set_quat(np.array([1, 0, 0, 0], dtype=np.float32))

            scene.step()

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
