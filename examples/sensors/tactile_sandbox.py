"""
Interactive demo of tactile sensors on a fixed taxel pad (box or dome) with controllable objects.
Sensor types: ContactDepthProbe, ElastomerTaxel, KinematicTaxel, ProximityTaxel.

Note that the sensor readings here have not been calibrated to any units, and is purely for visualization purposes.
"""

import argparse
import os
import tempfile
from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh

import genesis as gs
import genesis.utils.geom as gu
from genesis.recorders.plotters import IS_MATPLOTLIB_AVAILABLE
from genesis.utils.misc import tensor_to_array
from genesis.vis.keybindings import Key, KeyAction, Keybind

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidEntity
    from genesis.engine.sensors.base_sensor import Sensor

KEY_DPOS = 0.005
FORCE_SCALE = 100.0
ROT_FORCE_SCALE = 200.0

GRID_SIZE = 20  # 20x20 taxels for square
PROBE_RADIUS = 0.004
OBJECT_INITIAL_CLEARANCE = 0.01
OBJECT_MAX_PENETRATION = 1e-3

SENSOR_OBJ_SIZE = 0.15
SENSOR_OBJ_Z = 0.05
OBJECT_SIZE = 0.08

OBJ_PER_ENV_LABELS = ("torus", "sphere", "duck", "dragon")


def _add_tactile_sensor(
    scene: gs.Scene,
    entity: "RigidEntity",
    link_idx_local: int,
    sensor_type: str,
    probe_local_pos: np.ndarray,
    probe_normal: tuple[float, float, float] | np.ndarray,
    track_link_idx: tuple[int, ...],
) -> "Sensor":
    common = dict(
        entity_idx=entity.idx,
        link_idx_local=link_idx_local,
        draw_debug=True,
        probe_radius=PROBE_RADIUS,
    )
    if sensor_type == "elastomer":
        return scene.add_sensor(
            gs.sensors.ElastomerTaxel(
                probe_local_pos=probe_local_pos,
                probe_local_normal=probe_normal,
                track_link_idx=track_link_idx,
                n_sample_points=2000,
                dilate_scale=1.0,
                shear_scale=100.0,
                **common,
            )
        )

    probe_local_pos = probe_local_pos.reshape(-1, 3)
    if sensor_type == "depth":
        return scene.add_sensor(
            gs.sensors.ContactDepthProbe(
                probe_local_pos=probe_local_pos,
                **common,
            )
        )
    if sensor_type == "kinematic":
        return scene.add_sensor(
            gs.sensors.KinematicTaxel(
                probe_local_pos=probe_local_pos,
                probe_local_normal=probe_normal,
                normal_stiffness=500.0,
                normal_damping=1.0,
                shear_scalar=5.0,
                twist_scalar=5.0,
                normal_exponent=1.5,
                **common,
            )
        )

    common["probe_radius"] = PROBE_RADIUS * 2
    common["debug_point_cloud_radius"] = 0.001
    if sensor_type == "proximity":
        return scene.add_sensor(
            gs.sensors.ProximityTaxel(
                probe_local_pos=probe_local_pos,
                track_link_idx=track_link_idx,
                n_sample_points=4000,
                stiffness=200.0,
                shear_coupling=100.0,
                probe_local_normal=probe_normal,
                probe_radius_noise=0.0001,
                **common,
            )
        )
    raise ValueError(sensor_type)


def _plot_tactile_sensor(
    scene: gs.Scene,
    sensor_type: str,
    labels: tuple[str, ...],
    sensors: "tuple[Sensor, ...]",
    n_envs: int = 1,
    plot_normal: tuple[float, float, float] = (0.0, 0.0, -1.0),
) -> None:
    if not IS_MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available; skipping plot setup.")
        return

    if sensor_type == "elastomer":
        for env_idx in range(n_envs):
            for label, sensor in zip(labels, sensors):
                scene.start_recording(
                    lambda s=sensor, i=env_idx: s.read()[i],
                    gs.recorders.MPLVectorFieldPlot(
                        title=f"({label} {OBJ_PER_ENV_LABELS[env_idx]}) ElastomerTaxel marker displacements",
                        positions=sensor.probe_local_pos.reshape(-1, 3),
                        normal=plot_normal,
                        scale_factor=1.0,
                        max_magnitude=0.01,
                    ),
                )
    elif sensor_type == "kinematic":
        for env_idx in range(n_envs):
            for label, sensor in zip(labels, sensors):
                scene.start_recording(
                    lambda s=sensor, i=env_idx: s.read().force[i],
                    gs.recorders.MPLVectorFieldPlot(
                        title=f"({label} {OBJ_PER_ENV_LABELS[env_idx]}) KinematicTaxel force",
                        positions=sensor.probe_local_pos.reshape(-1, 3),
                        normal=plot_normal,
                        scale_factor=0.01,
                        max_magnitude=1.0,
                    ),
                )
    elif sensor_type == "proximity":
        for env_idx in range(n_envs):
            for label, sensor in zip(labels, sensors):
                scene.start_recording(
                    lambda s=sensor, i=env_idx: s.read().force[i],
                    gs.recorders.MPLVectorFieldPlot(
                        title=f"({label} {OBJ_PER_ENV_LABELS[env_idx]}) ProximityTaxel force",
                        positions=sensor.probe_local_pos.reshape(-1, 3),
                        normal=plot_normal,
                        scale_factor=0.5,
                        max_magnitude=1.0,
                    ),
                )
    elif sensor_type == "depth":
        for env_idx in range(n_envs):
            scene.start_recording(
                lambda i=env_idx: tuple(sensor.read()[i].max() for sensor in sensors),
                gs.recorders.MPLLinePlot(
                    title=f"ContactDepthProbe max depth ({OBJ_PER_ENV_LABELS[env_idx]})",
                    labels=labels,
                    x_label="step",
                    y_label="depth",
                    history_length=200,
                ),
            )


def _print_sensor_reading(sensor_type: str, sensor: "Sensor", t: float) -> None:
    data = sensor.read()
    if sensor_type == "elastomer":
        magnitude = torch.linalg.norm(data, dim=-1)
        if magnitude.max() > gs.EPS:
            print(f"t={t:.2f}s  max|displacement|={magnitude.max():.5f}")
    elif sensor_type == "depth":
        max_depth = data.max()
        if max_depth > gs.EPS:
            print(f"t={t:.2f}s  max depth={max_depth:.4f}")
    elif sensor_type == "kinematic":
        magnitude = torch.linalg.norm(data.force, axis=-1).max()
        if magnitude > gs.EPS:
            print(f"t={t:.2f}s  max|F|={magnitude:.4f}")
    elif sensor_type == "proximity":
        magnitude = torch.linalg.norm(data.force, dim=-1)
        if magnitude.max() > gs.EPS:
            print(f"t={t:.2f}s  mean|F|={magnitude.mean():.5f}  max|F|={magnitude.max():.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive tactile sandbox with selectable sensor type")
    parser.add_argument("--vis", "-v", action="store_true", default=False, help="Show visualization GUI")
    parser.add_argument("--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument(
        "--set-pos", action="store_true", help="Set object position directly instead of using control force."
    )
    parser.add_argument("--seconds", "-t", type=float, default=3.0, help="Seconds to simulate (headless mode)")
    parser.add_argument("--dome", action="store_true", help="Change the sensor object to a dome instead of a box")
    parser.add_argument(
        "--sensor",
        choices=("elastomer", "depth", "kinematic", "proximity"),
        default="elastomer",
        help="Type of tactile sensor to use.",
    )
    args = parser.parse_args()

    gs.init(
        backend=gs.gpu if args.gpu else gs.cpu,
        precision="32",
        logging_level="info",
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
            substeps=4,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(-0.7, 0.2, 1.0),
            camera_lookat=(0.0, 0.0, SENSOR_OBJ_Z),
            max_FPS=60,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=True,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    if args.dome:
        sensor_morph = gs.morphs.Sphere(
            radius=SENSOR_OBJ_SIZE / 2,
            pos=(0.0, 0.0, -SENSOR_OBJ_SIZE / 2 + SENSOR_OBJ_Z),
            fixed=True,
        )
    else:
        sensor_morph = gs.morphs.Box(
            size=(SENSOR_OBJ_SIZE, SENSOR_OBJ_SIZE, SENSOR_OBJ_Z),
            pos=(0.0, 0.0, SENSOR_OBJ_Z / 2),
            fixed=True,
        )
    sensor_obj = scene.add_entity(
        morph=sensor_morph,
        surface=gs.surfaces.Default(
            color=(0.8, 0.8, 0.8, 1.0),
        ),
        material=gs.materials.Rigid(
            friction=0.6,
        ),
    )
    probe_normal_axis = (0.0, 0.0, 1.0)
    if args.dome:
        sphere_radius = SENSOR_OBJ_SIZE / 2
        probe_local_pos, probe_normal = gu.generate_ring_points_on_sphere(
            radius=sphere_radius,
            cap_axis=probe_normal_axis,
            n_rings=GRID_SIZE,
            arc_spacing=2.0 * PROBE_RADIUS,
            return_normals=True,
        )
    else:
        probe_normal = probe_normal_axis
        probe_z = SENSOR_OBJ_Z / 2
        probe_local_pos = gu.generate_grid_points_on_plane(
            lo=(-SENSOR_OBJ_SIZE / 2, -SENSOR_OBJ_SIZE / 2, probe_z),
            hi=(SENSOR_OBJ_SIZE / 2, SENSOR_OBJ_SIZE / 2, probe_z),
            normal=probe_normal_axis,
            nx=GRID_SIZE,
            ny=GRID_SIZE,
        )

    # Procedural torus written to a temp .obj (avoids checking a 2k-line mesh into the repo).
    torus_path = os.path.join(tempfile.gettempdir(), "tactile_sandbox_torus.obj")
    if not os.path.exists(torus_path):
        trimesh.creation.torus(major_radius=0.3, minor_radius=0.1).export(torus_path)

    obj = scene.add_entity(
        morph=[
            gs.morphs.Mesh(
                file=torus_path,
                euler=(90.0, 0.0, 0.0),
                scale=OBJECT_SIZE,
            ),
            gs.morphs.Sphere(
                radius=OBJECT_SIZE / 2,
            ),
            gs.morphs.Mesh(
                file="meshes/duck.obj",
                euler=(90.0, 0.0, 0.0),
                scale=0.03,
            ),
            gs.morphs.Mesh(
                file="meshes/dragon/dragon.obj",
                euler=(90.0, 0.0, 90.0),
                scale=0.001,
            ),
        ],
        surface=gs.surfaces.Default(color=(1.0, 1.0, 1.0, 1.0)),
        material=gs.materials.Rigid(friction=0.5),
    )

    sensor = _add_tactile_sensor(
        scene,
        sensor_obj,
        0,
        args.sensor,
        probe_local_pos,
        probe_normal,
        track_link_idx=(obj.base_link_idx,),
    )
    if args.vis and "PYTEST_VERSION" not in os.environ:
        _plot_tactile_sensor(scene, args.sensor, ("",), (sensor,), n_envs=4, plot_normal=probe_normal_axis)
    scene.build(n_envs=4, env_spacing=(SENSOR_OBJ_SIZE * 1.2, SENSOR_OBJ_SIZE * 1.2))

    obj_init_pos = tensor_to_array(obj.get_pos())
    obj_init_quat = tensor_to_array(obj.get_quat())
    obj_aabb = tensor_to_array(obj.get_vAABB())
    obj_init_pos[..., 2] += SENSOR_OBJ_Z + OBJECT_INITIAL_CLEARANCE - obj_aabb[..., 0, 2]
    obj.set_pos(obj_init_pos)

    obj_target_pos = obj_init_pos.copy()
    obj_target_euler = gu.quat_to_xyz(obj_init_quat, rpy=True, degrees=True)
    obj_target_quat = obj_init_quat.copy()
    obj_contact_pos = obj_init_pos.copy()
    obj_contact_pos[..., 2] -= OBJECT_INITIAL_CLEARANCE + OBJECT_MAX_PENETRATION

    is_running = True

    if args.vis:
        obj.set_dofs_kp(FORCE_SCALE / KEY_DPOS, dofs_idx_local=slice(0, 3))
        obj.set_dofs_kp(ROT_FORCE_SCALE / KEY_DPOS, dofs_idx_local=slice(3, 6))
        obj.set_dofs_kv(0.1 * FORCE_SCALE / KEY_DPOS, dofs_idx_local=slice(0, 6))

        def stop():
            nonlocal is_running
            is_running = False

        def reset_pose():
            nonlocal obj_target_pos, obj_target_euler, obj_target_quat
            obj_target_pos = obj_init_pos.copy()
            obj_target_euler = gu.quat_to_xyz(obj_init_quat, rpy=True, degrees=True)
            obj_target_quat = obj_init_quat.copy()
            obj.set_pos(obj_init_pos)
            obj.set_quat(obj_init_quat)

        def translate(index: int, is_negative: bool):
            nonlocal obj_target_pos
            delta = (-1 if is_negative else 1) * KEY_DPOS
            obj_target_pos[..., index] += delta
            # obj_target_pos[..., 2] = np.maximum(
            #     obj_target_pos[..., 2], obj_init_pos[..., 2] - OBJECT_INITIAL_CLEARANCE - OBJECT_MAX_PENETRATION
            # )

        def rotate(axis_idx: int, is_negative: bool):
            nonlocal obj_target_euler, obj_target_quat
            delta = -5.0 if is_negative else 5.0
            obj_target_euler[..., axis_idx] += delta
            obj_target_quat = gu.euler_to_quat(obj_target_euler)

        scene.viewer.register_keybinds(
            Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=translate, args=(0, False)),
            Keybind("move_backward", Key.DOWN, KeyAction.HOLD, callback=translate, args=(0, True)),
            Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=translate, args=(1, True)),
            Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=translate, args=(1, False)),
            Keybind("move_down", Key.J, KeyAction.HOLD, callback=translate, args=(2, True)),
            Keybind("move_up", Key.K, KeyAction.HOLD, callback=translate, args=(2, False)),
            Keybind("rotate_cw", Key.M, KeyAction.HOLD, callback=rotate, args=(2, True)),
            Keybind("rotate_ccw", Key.N, KeyAction.HOLD, callback=rotate, args=(2, False)),
            Keybind("rotate_roll_left", Key.COMMA, KeyAction.HOLD, callback=rotate, args=(0, True)),
            Keybind("rotate_roll_right", Key.PERIOD, KeyAction.HOLD, callback=rotate, args=(0, False)),
            Keybind("reset", Key.BACKSLASH, KeyAction.RELEASE, callback=reset_pose),
            Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        )

    print("\n=== Tactile Sensor Sandbox ===")
    n_taxels = probe_local_pos.reshape(-1, 3).shape[0]
    layout = f"dome ({GRID_SIZE} latitude rings)" if args.dome else f"plane grid {probe_local_pos.shape[:-1]}"
    print(f"sensor={args.sensor}; taxels={n_taxels}; {layout}")
    if args.vis and IS_MATPLOTLIB_AVAILABLE:
        print("Matplotlib live plot enabled when supported.")
    if args.vis:
        print()
        print("Keyboard Controls:")
        print("  [UP/DOWN/LEFT/RIGHT] Move selected object in XY")
        print("  [j / k]              Lower / raise selected object")
        print("  [n / m]              Rotate object around Z axis")
        print("  [SPACE]              Select next object")
        print("  [\\]                  Reset objects")
    else:
        obj.set_pos(obj_contact_pos)
        print(f"Running headless for {args.seconds}s ...")
    print()

    try:
        while is_running:
            t = scene.t * scene.dt
            if args.vis:
                if args.set_pos:
                    obj.set_pos(obj_target_pos)
                    obj.set_quat(obj_target_quat)
                else:
                    obj.control_dofs_position(gu.quat_to_xyz(obj_target_quat), dofs_idx_local=slice(3, 6))
                    obj.control_dofs_position(obj_target_pos, dofs_idx_local=slice(0, 3))
            _print_sensor_reading(args.sensor, sensor, t)

            scene.step()

            if "PYTEST_VERSION" in os.environ:
                break
            if not args.vis and t >= args.seconds:
                break
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted.")
    finally:
        gs.logger.info("Simulation finished.")


if __name__ == "__main__":
    main()
