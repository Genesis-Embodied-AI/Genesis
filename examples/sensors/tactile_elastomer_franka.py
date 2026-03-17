import os

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.recorders.plotters import IS_MATPLOTLIB_AVAILABLE
from genesis.vis.keybindings import Key, KeyAction, Keybind

OBJ_DENSITY = 300
OBJ_SIZE = 0.04
CUBE_INIT_XY = (0.6, 0.2)
SPHERE_INIT_XY = (0.4, -0.2)
ROBOT_INIT_HEIGHT = 0.18

# Control parameters
DPOS = 0.005
DROT = 0.04


if __name__ == "__main__":
    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0.0, 0.7),
            camera_lookat=(0.2, 0.0, 0.1),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=True,
        ),
        show_viewer=True,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )

    franka = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
    )

    cube = scene.add_entity(
        morph=gs.morphs.Box(
            size=(OBJ_SIZE, OBJ_SIZE, OBJ_SIZE),
            pos=(*CUBE_INIT_XY, OBJ_SIZE / 2),
        ),
        material=gs.materials.Rigid(
            rho=OBJ_DENSITY,
        ),
        surface=gs.surfaces.Default(
            color=(0.5, 1.0, 0.5),
        ),
    )

    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=OBJ_SIZE / 2,
            pos=(*SPHERE_INIT_XY, OBJ_SIZE / 2),
        ),
        material=gs.materials.Rigid(
            rho=OBJ_DENSITY,
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.5, 0.5),
        ),
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
        entity_idx=franka.idx,
        probe_local_pos=probe_local_pos,
        probe_local_normal=probe_normal,
        probe_radius=0.002,
        dilate_coefficient=1e1,
        shear_coefficient=1e-2,
        twist_coefficient=1e-2,
        draw_debug=True,
    )

    left_finger_tactile_sensor = scene.add_sensor(
        gs.sensors.ElastomerDisplacement(
            link_idx_local=franka.get_link("left_finger").idx_local,
            **tactile_sensor_kwargs,
        )
    )
    right_finger_tactile_sensor = scene.add_sensor(
        gs.sensors.ElastomerDisplacement(
            link_idx_local=franka.get_link("right_finger").idx_local,
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
    n_dofs = franka.n_dofs
    motor_dofs_idx = np.arange(n_dofs - 2)
    fingers_dof = np.arange(n_dofs - 2, n_dofs)
    ee_link = franka.get_link("hand")

    # Initialize target pose
    target_init_pos = np.array((*SPHERE_INIT_XY, ROBOT_INIT_HEIGHT), dtype=gs.np_float)
    target_init_quat = gu.xyz_to_quat(np.array([0.0, 180.0, 0.0], dtype=gs.np_float), degrees=True)
    target_pos, target_quat = target_init_pos.copy(), target_init_quat.copy()

    target_ik = scene.draw_debug_frame(
        T=gu.trans_quat_to_T(target_pos, target_quat),
        axis_length=0.15,
        origin_size=0.01,
        axis_radius=0.007,
    )
    scene.viewer.update(force=True)

    # Robot teleoperation callback functions
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
        scene.visualizer.context.update_debug_objects((target_ik,), (pose,))

        qpos = franka.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat, dofs_idx_local=motor_dofs_idx)
        franka.set_qpos(qpos[motor_dofs_idx], motor_dofs_idx)
        toggle_gripper(False)

        cube.set_pos(cube.base_link.pos)
        sphere.set_pos(sphere.base_link.pos)

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
        Keybind("move_up", Key.K, KeyAction.HOLD, callback=move, args=((0, 0, DPOS),)),
        Keybind("move_down", Key.J, KeyAction.HOLD, callback=move, args=((0, 0, -DPOS),)),
        Keybind("rotate_ccw", Key.N, KeyAction.HOLD, callback=rotate, args=(DROT,)),
        Keybind("rotate_cw", Key.M, KeyAction.HOLD, callback=rotate, args=(-DROT,)),
        Keybind("reset_scene", Key.BACKSLASH, KeyAction.RELEASE, callback=reset_robot),
        Keybind("close_gripper", Key.SPACE, KeyAction.PRESS, callback=toggle_gripper, args=(True,)),
        Keybind("open_gripper", Key.SPACE, KeyAction.RELEASE, callback=toggle_gripper, args=(False,)),
        Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
    )

    ########################## run simulation ##########################
    try:
        while is_running:
            # Update target entity visualization
            pose = gu.trans_quat_to_T(target_pos, target_quat)
            scene.visualizer.context.update_debug_objects((target_ik,), (pose,))

            # Control arm with inverse kinematics
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
