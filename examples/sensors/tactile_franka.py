"""
Interactive demo of tactile sensors attached to Franka Panda grippers and matplotlib visualization.
Sensor types: ContactDepthProbe, ElastomerTaxel, KinematicTaxel, ProximityTaxel.
"""

import argparse
import os
from typing import TYPE_CHECKING

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.recorders.plotters import IS_MATPLOTLIB_AVAILABLE
from genesis.vis.keybindings import Key, KeyAction, Keybind

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidEntity
    from genesis.engine.sensors.base_sensor import Sensor

OBJ_DENSITY = 300
OBJ_SIZE = 0.04
CUBE_INIT_XY = (0.3, 0.1)
SPHERE_INIT_XY = (0.3, -0.1)
ROBOT_INIT_HEIGHT = 0.18

DPOS = 0.005
DROT = 0.04


def _add_tactile_sensor(
    scene: gs.Scene,
    entity: "RigidEntity",
    link_idx_local: int,
    sensor_type: str,
    probe_local_pos: np.ndarray,
    probe_normal: tuple[float, float, float],
    track_link_idx: tuple[int, ...],
) -> "Sensor":
    common = dict(
        entity_idx=entity.idx,
        link_idx_local=link_idx_local,
        draw_debug=True,
    )
    if sensor_type == "elastomer":
        return scene.add_sensor(
            gs.sensors.ElastomerTaxel(
                probe_local_pos=probe_local_pos,
                probe_local_normal=probe_normal,
                probe_radius=0.002,
                track_link_idx=track_link_idx,
                dilate_scale=10.0,
                shear_scale=100.0,
                **common,
            )
        )
    probe_local_pos = probe_local_pos.reshape(-1, 3)  # flatten
    if sensor_type == "depth":
        return scene.add_sensor(
            gs.sensors.ContactDepthProbe(
                probe_local_pos=probe_local_pos,
                probe_radius=0.002,
                **common,
            )
        )
    if sensor_type == "kinematic":
        return scene.add_sensor(
            gs.sensors.KinematicTaxel(
                probe_local_pos=probe_local_pos,
                probe_local_normal=probe_normal,
                probe_radius=0.002,
                normal_stiffness=5000.0,
                normal_damping=1.0,
                normal_exponent=1.5,
                shear_scalar=1.0,
                twist_scalar=1.0,
                **common,
            )
        )
    if sensor_type == "proximity":
        return scene.add_sensor(
            gs.sensors.ProximityTaxel(
                probe_local_pos=probe_local_pos,
                track_link_idx=track_link_idx,
                probe_radius=0.02,
                n_sample_points=500,
                stiffness=10.0,
                shear_coupling=1.0,
                probe_local_normal=probe_normal,
                probe_radius_noise=0.005,
                **common,
            )
        )

    raise ValueError(sensor_type)


def _plot_tactile_sensor(
    scene: gs.Scene,
    sensor_type: str,
    labels: tuple[str, ...],
    sensors: "tuple[Sensor, ...]",
    plot_normal: tuple[float, float, float] = (0.0, 0.0, -1.0),
) -> None:
    if not IS_MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available; skipping plot setup.")
        return

    if sensor_type == "elastomer":
        for label, sensor in zip(labels, sensors):
            sensor.start_recording(
                gs.recorders.MPLVectorFieldPlot(
                    title=f"({label}) ElastomerTaxel marker displacements",
                    positions=sensor.probe_local_pos.reshape(-1, 3),
                    normal=plot_normal,
                    scale_factor=1.0,
                    max_magnitude=0.005,
                ),
            )
    elif sensor_type == "kinematic":
        for label, sensor in zip(labels, sensors):
            scene.start_recording(
                lambda: sensor.read().force,
                gs.recorders.MPLVectorFieldPlot(
                    title=f"({label}) KinematicTaxel force",
                    positions=sensor.probe_local_pos.reshape(-1, 3),
                    normal=plot_normal,
                    scale_factor=0.01,
                    max_magnitude=1.0,
                ),
            )
        scene.start_recording(
            lambda: tuple(sensor.read().force.norm(dim=-1).max() for sensor in sensors),
            gs.recorders.MPLLinePlot(
                title="KinematicTaxel max force magnitude",
                labels=labels,
                x_label="step",
                y_label="|F|",
                history_length=200,
            ),
        )
    elif sensor_type == "proximity":
        for label, sensor in zip(labels, sensors):
            scene.start_recording(
                lambda: sensor.read().force,
                gs.recorders.MPLVectorFieldPlot(
                    title=f"({label}) ProximityTaxel force",
                    positions=sensor.probe_local_pos.reshape(-1, 3),
                    normal=plot_normal,
                    scale_factor=0.2,
                    max_magnitude=1.0,
                ),
            )
    elif sensor_type == "depth":
        scene.start_recording(
            lambda: tuple(sensor.read().max() for sensor in sensors),
            gs.recorders.MPLLinePlot(
                title="ContactDepthProbe max depth",
                labels=labels,
                x_label="step",
                y_label="depth",
                history_length=200,
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Franka fingertip tactile with selectable sensor type")
    parser.add_argument(
        "--sensor",
        choices=("elastomer", "depth", "kinematic", "proximity"),
        default="elastomer",
        help="Tactile sensor implementation",
    )
    parser.add_argument("--gpu", action="store_true", help="Run on GPU instead of CPU")
    args = parser.parse_args()

    gs.init(
        backend=gs.gpu if args.gpu else gs.cpu,
        precision="32",
        logging_level="info",
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            substeps=4,
        ),
        rigid_options=gs.options.RigidOptions(
            box_box_detection=True,
            constraint_timeconst=0.01,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0.0, 0.7),
            camera_lookat=(0.2, 0.0, 0.1),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=True,
        ),
        show_viewer=True,
    )

    scene.add_entity(gs.morphs.Plane())

    franka = scene.add_entity(
        morph=gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )

    cube = scene.add_entity(
        morph=gs.morphs.Box(
            size=(OBJ_SIZE, OBJ_SIZE, OBJ_SIZE),
            pos=(*CUBE_INIT_XY, OBJ_SIZE / 2),
        ),
        material=gs.materials.Rigid(rho=OBJ_DENSITY),
        surface=gs.surfaces.Default(color=(0.5, 1.0, 0.5)),
    )

    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=OBJ_SIZE / 2,
            pos=(*SPHERE_INIT_XY, OBJ_SIZE / 2),
        ),
        material=gs.materials.Rigid(rho=OBJ_DENSITY),
        surface=gs.surfaces.Default(color=(1.0, 0.5, 0.5)),
    )

    probe_normal = (0.0, -1.0, 0.0)
    probe_local_pos = gu.generate_grid_points_on_plane(
        lo=(-0.006, 0.0, 0.04),
        hi=(0.008, 0.0, 0.05),
        normal=probe_normal,
        nx=8,
        ny=8,
    )
    track_idx = (int(cube.base_link_idx), int(sphere.base_link_idx))

    left = _add_tactile_sensor(
        scene, franka, franka.get_link("left_finger").idx_local, args.sensor, probe_local_pos, probe_normal, track_idx
    )
    right = _add_tactile_sensor(
        scene, franka, franka.get_link("right_finger").idx_local, args.sensor, probe_local_pos, probe_normal, track_idx
    )
    _plot_tactile_sensor(scene, args.sensor, ("left", "right"), (left, right), plot_normal=probe_normal)

    scene.build()

    n_dofs = franka.n_dofs
    motor_dofs_idx = np.arange(n_dofs - 2)
    fingers_dof = np.arange(n_dofs - 2, n_dofs)
    franka.set_dofs_kp([100.0, 100.0], fingers_dof)
    franka.set_dofs_kv([10.0, 10.0], fingers_dof)
    ee_link = franka.get_link("hand")

    target_init_pos = np.array((*SPHERE_INIT_XY, ROBOT_INIT_HEIGHT), dtype=gs.np_float)
    target_init_quat = gu.euler_to_quat((0.0, 180.0, 0.0))
    target_pos, target_quat = target_init_pos.copy(), target_init_quat.copy()

    target_ik = scene.draw_debug_frame(
        T=gu.trans_quat_to_T(target_pos, target_quat),
        axis_length=0.15,
        origin_size=0.01,
        axis_radius=0.007,
    )
    scene.viewer.update(force=True)

    def move(dpos_xyz: tuple[float, float, float]):
        target_pos[:] += dpos_xyz

    def rotate(drot: float):
        drot_quat = gu.xyz_to_quat(np.array([0, 0, drot]))
        target_quat[:] = gu.transform_quat_by_quat(target_quat, drot_quat)

    def toggle_gripper(close: bool):
        pos = -0.03 if close else 0.04
        franka.control_dofs_position(pos, dofs_idx_local=fingers_dof)

    def reset_robot():
        target_pos[:], target_quat[:] = target_init_pos, target_init_quat
        pose = gu.trans_quat_to_T(target_pos, target_quat)
        scene.update_debug_objects((target_ik,), (pose,))

        qpos = franka.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat, dofs_idx_local=motor_dofs_idx)
        franka.set_qpos(qpos[motor_dofs_idx], motor_dofs_idx)
        toggle_gripper(False)

        cube.set_pos(cube.base_link.pos)
        sphere.set_pos(sphere.base_link.pos)

    reset_robot()

    is_running = True

    def stop():
        nonlocal is_running
        is_running = False

    scene.viewer.register_keybinds(
        Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=move, args=((-DPOS, 0, 0),)),
        Keybind("move_back", Key.DOWN, KeyAction.HOLD, callback=move, args=((DPOS, 0, 0),)),
        Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=move, args=((0, -DPOS, 0),)),
        Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=move, args=((0, DPOS, 0),)),
        Keybind("move_up", Key.K, KeyAction.HOLD, callback=move, args=((0, 0, DPOS),)),
        Keybind("move_down", Key.J, KeyAction.HOLD, callback=move, args=((0, 0, -DPOS),)),
        Keybind("rotate_ccw", Key.N, KeyAction.HOLD, callback=rotate, args=(DROT,)),
        Keybind("rotate_cw", Key.M, KeyAction.HOLD, callback=rotate, args=(-DROT,)),
        Keybind("reset_scene", Key.BACKSLASH, KeyAction.RELEASE, callback=reset_robot),
        Keybind("close_gripper", Key.SPACE, KeyAction.PRESS, callback=toggle_gripper, args=(True,)),
        Keybind("open_gripper", Key.SPACE, KeyAction.RELEASE, callback=toggle_gripper, args=(False,)),
        Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
    )

    try:
        while is_running:
            pose = gu.trans_quat_to_T(target_pos, target_quat)
            scene.update_debug_objects((target_ik,), (pose,))

            qpos = franka.inverse_kinematics(
                link=ee_link, pos=target_pos, quat=target_quat, dofs_idx_local=motor_dofs_idx
            )
            franka.control_dofs_position(qpos[motor_dofs_idx], motor_dofs_idx)

            scene.step()

            if "PYTEST_VERSION" in os.environ:
                break
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")


if __name__ == "__main__":
    main()
