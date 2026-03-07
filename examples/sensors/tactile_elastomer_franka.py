import argparse
import os

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.recorders.plotters import IS_MATPLOTLIB_AVAILABLE
from genesis.vis.keybindings import Key, KeyAction, Keybind

OBJ_DENSITY = 300
OBJ_SIZE = 0.04
CUBE_INIT_POS = (0.6, 0.2, OBJ_SIZE / 2)
SPHERE_INIT_POS = (0.4, -0.2, OBJ_SIZE / 2)

ROBOT_INIT_POS = (SPHERE_INIT_POS[0], SPHERE_INIT_POS[1], 0.15)
ROBOT_INIT_QUAT = gu.xyz_to_quat(np.array([0, np.pi, 0]))

# Control parameters
DPOS = 0.005
DROT = 0.04


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", default=False)
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0.0, 0.7),
            camera_lookat=(0.2, 0.0, 0.1),
            enable_default_keybinds=False,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=True,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )

    robot = scene.add_entity(
        material=gs.materials.Rigid(gravity_compensation=1),
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            euler=(0, 0, 0),
        ),
    )

    cube = scene.add_entity(
        morph=gs.morphs.Box(
            pos=CUBE_INIT_POS,
            size=(OBJ_SIZE, OBJ_SIZE, OBJ_SIZE),
        ),
        material=gs.materials.Rigid(rho=OBJ_DENSITY),
        surface=gs.surfaces.Default(color=(0.5, 1.0, 0.5)),
    )

    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=SPHERE_INIT_POS,
            radius=OBJ_SIZE / 2,
        ),
        material=gs.materials.Rigid(rho=OBJ_DENSITY),
        surface=gs.surfaces.Default(color=(1.0, 0.5, 0.5)),
    )

    target = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            scale=0.15,
            collision=False,
        ),
        surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
    )

    probe_normal = (0.0, -1.0, 0.0)
    probe_local_pos = gu.generate_grid_points_on_plane(
        lo=(-0.006, 0.0, 0.04),
        hi=(0.008, 0.0, 0.05),
        normal=probe_normal,
        nx=8,
        ny=8,
    )
    tactile_sensor_kwargs = dict(
        entity_idx=robot.idx,
        probe_local_pos=probe_local_pos,
        probe_local_normal=probe_normal,
        probe_radius=0.002,
        draw_debug=True,
        dilate_coefficient=1e1,
        shear_coefficient=1e-2,
        twist_coefficient=1e-2,
    )

    left_finger_tactile_sensor = scene.add_sensor(
        gs.sensors.ElastomerDisplacement(
            link_idx_local=robot.get_link("left_finger").idx_local,
            **tactile_sensor_kwargs,
        )
    )
    right_finger_tactile_sensor = scene.add_sensor(
        gs.sensors.ElastomerDisplacement(
            link_idx_local=robot.get_link("right_finger").idx_local,
            probe_local_pos=probe_local_pos,
            **tactile_sensor_kwargs,
        )
    )

    if IS_MATPLOTLIB_AVAILABLE:
        rec_kwargs = dict(
            normal=probe_normal,
            scale_factor=1.0,
            max_magnitude=1.0e-2,
        )
        left_finger_tactile_sensor.start_recording(
            rec_options=gs.recorders.MPLVectorFieldPlot(
                title="Tactile Displacement (Left)",
                positions=left_finger_tactile_sensor.probe_local_pos,
                **rec_kwargs,
            ),
        )
        right_finger_tactile_sensor.start_recording(
            rec_options=gs.recorders.MPLVectorFieldPlot(
                title="Tactile Displacement (Right)",
                positions=right_finger_tactile_sensor.probe_local_pos,
                **rec_kwargs,
            ),
        )

    scene.build()

    # Get DOF indices
    n_dofs = robot.n_dofs
    motors_dof = np.arange(n_dofs - 2)
    fingers_dof = np.arange(n_dofs - 2, n_dofs)
    ee_link = robot.get_link("hand")

    # Initialize target pose
    target_pos = np.array(ROBOT_INIT_POS)
    target_quat = np.array(ROBOT_INIT_QUAT)

    # Robot teleoperation callback functions
    def move(dpos: tuple[float, float, float]):
        target_pos[:] += np.array(dpos, dtype=gs.np_float)

    def rotate(drot: float):
        drot_quat = gu.xyz_to_quat(np.array([0, 0, drot]))
        target_quat[:] = gu.transform_quat_by_quat(target_quat, drot_quat)

    def toggle_gripper(close: bool):
        pos = -1.0 if close else 1.0
        robot.control_dofs_force(np.array([pos, pos]), fingers_dof)

    def reset_robot():
        target_pos[:] = ROBOT_INIT_POS
        target_quat[:] = ROBOT_INIT_QUAT
        target.set_qpos(np.concatenate([target_pos, target_quat]))
        q = robot.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat)
        robot.set_qpos(q[:-2], motors_dof)
        toggle_gripper(False)

        cube.set_pos(CUBE_INIT_POS)
        sphere.set_pos(SPHERE_INIT_POS)

    reset_robot()

    is_running = True

    def stop():
        global is_running
        is_running = False

    scene.viewer.register_keybinds(
        Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=move, args=((-DPOS, 0, 0),)),
        Keybind("move_back", Key.DOWN, KeyAction.HOLD, callback=move, args=((DPOS, 0, 0),)),
        Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=move, args=((0, -DPOS, 0),)),
        Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=move, args=((0, DPOS, 0),)),
        Keybind("move_up", Key.N, KeyAction.HOLD, callback=move, args=((0, 0, DPOS),)),
        Keybind("move_down", Key.M, KeyAction.HOLD, callback=move, args=((0, 0, -DPOS),)),
        Keybind("rotate_ccw", Key.J, KeyAction.HOLD, callback=rotate, args=(DROT,)),
        Keybind("rotate_cw", Key.K, KeyAction.HOLD, callback=rotate, args=(-DROT,)),
        Keybind("reset_scene", Key.U, KeyAction.HOLD, callback=reset_robot),
        Keybind("close_gripper", Key.SPACE, KeyAction.PRESS, callback=toggle_gripper, args=(True,)),
        Keybind("open_gripper", Key.SPACE, KeyAction.RELEASE, callback=toggle_gripper, args=(False,)),
        Keybind("quit", Key.ESCAPE, KeyAction.PRESS, callback=stop),
    )

    ########################## run simulation ##########################
    try:
        while is_running:
            # Update target entity visualization
            target.set_qpos(np.concatenate([target_pos, target_quat]))

            # Control arm with inverse kinematics
            q, err = robot.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat, return_error=True)
            robot.control_dofs_position(q[:-2], motors_dof)

            scene.step()

            if "PYTEST_VERSION" in os.environ:
                break
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")
