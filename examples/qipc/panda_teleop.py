"""
QIPC coupler demo: keyboard teleoperation of Franka Panda end-effector.

Keyboard Controls:
    Arrow keys  - Move in XY plane
    J/K         - Move down/up
    N/M         - Yaw left/right
    U/O         - Pitch up/down
    L/;         - Roll left/right
    Space       - Toggle gripper
    Backslash   - Reset to home pose
    Esc         - Quit
"""
import argparse

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.vis.keybindings import Key, KeyAction, Keybind

DELTA_POS = 0.003
DELTA_ROT = 0.02


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    args = parser.parse_args()

    gs.init(precision="64", logging_level="info")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0.0, 0.0, -9.81),
        ),
        coupler_options=gs.options.QIPCCouplerOptions(
            contact_enable=False,
            debug_viewer=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, -1.0, 1.2),
            camera_lookat=(0.4, 0.0, 0.4),
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )

    franka = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=(0, 0, 0),
        ),
        material=gs.materials.Rigid(
            qipc_abd_kappa=1e8,
            qipc_kappa_pivot=1e7,
            qipc_kappa_axis=1e7,
            # qipc_home_qpos=[0, 0, 0, 0, 0, 0, 0, 0.0, 0.0],
            qipc_home_qpos=[0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04],
        ),
    )

    motor_dofs_idx = slice(0, 7)
    finger_dofs_idx = slice(7, 9)
    ee_link = franka.get_link("hand")

    target_init_pos = np.array([0.5, 0.0, 0.5], dtype=gs.np_float)
    target_init_quat = gu.xyz_to_quat(np.array([0.0, 180.0, 0.0], dtype=gs.np_float), degrees=True)
    target_pos = target_init_pos.copy()
    target_quat = target_init_quat.copy()

    scene.build()

    coupler_jc = scene.sim._coupler._jc
    coupler_jc[7:9].set_dofs_kp(500.0)
    coupler_jc[7:9].set_dofs_kv(50.0)

    qpos = franka.inverse_kinematics(
        link=ee_link,
        pos=target_pos,
        quat=target_quat,
        dofs_idx_local=motor_dofs_idx,
    )
    franka.control_dofs_position(qpos[motor_dofs_idx], dofs_idx_local=motor_dofs_idx)
    franka.control_dofs_position(0.04, dofs_idx_local=finger_dofs_idx)

    target_ik = scene.draw_debug_frame(
        T=gu.trans_quat_to_T(target_pos, target_quat),
        axis_length=0.15,
        origin_size=0.01,
        axis_radius=0.007,
    )

    if scene.viewer is None:
        gs.logger.warning("Viewer is not active. Keyboard input requires the Genesis viewer.")
        return

    scene.viewer.update(force=True)

    is_gripper_closed = np.array(False, dtype=gs.np_bool)
    is_running = True

    def move(dpos_xyz):
        target_pos[:] += dpos_xyz

    def rotate(axis_idx, delta):
        delta_xyz = np.zeros(3, dtype=gs.np_float)
        delta_xyz[axis_idx] = delta
        target_quat[:] = gu.transform_quat_by_quat(target_quat, gu.xyz_to_quat(delta_xyz))

    def reset_pose():
        target_pos[:] = target_init_pos
        target_quat[:] = target_init_quat

    def set_gripper(is_closed):
        is_gripper_closed[()] = is_closed

    def stop():
        nonlocal is_running
        is_running = False

    scene.viewer.register_keybinds(
        Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=move, args=((-DELTA_POS, 0, 0),)),
        Keybind("move_back", Key.DOWN, KeyAction.HOLD, callback=move, args=((DELTA_POS, 0, 0),)),
        Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=move, args=((0, -DELTA_POS, 0),)),
        Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=move, args=((0, DELTA_POS, 0),)),
        Keybind("move_up", Key.K, KeyAction.HOLD, callback=move, args=((0, 0, DELTA_POS),)),
        Keybind("move_down", Key.J, KeyAction.HOLD, callback=move, args=((0, 0, -DELTA_POS),)),
        Keybind("yaw_left", Key.N, KeyAction.HOLD, callback=rotate, args=(2, DELTA_ROT)),
        Keybind("yaw_right", Key.M, KeyAction.HOLD, callback=rotate, args=(2, -DELTA_ROT)),
        Keybind("pitch_up", Key.U, KeyAction.HOLD, callback=rotate, args=(1, DELTA_ROT)),
        Keybind("pitch_down", Key.O, KeyAction.HOLD, callback=rotate, args=(1, -DELTA_ROT)),
        Keybind("roll_left", Key.L, KeyAction.HOLD, callback=rotate, args=(0, DELTA_ROT)),
        Keybind("roll_right", Key.SEMICOLON, KeyAction.HOLD, callback=rotate, args=(0, -DELTA_ROT)),
        Keybind("reset_pose", Key.BACKSLASH, KeyAction.RELEASE, callback=reset_pose),
        Keybind("close_gripper", Key.SPACE, KeyAction.PRESS, callback=set_gripper, args=(True,)),
        Keybind("open_gripper", Key.SPACE, KeyAction.RELEASE, callback=set_gripper, args=(False,)),
        Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        overwrite=True,
    )

    while is_running and scene.viewer.is_alive():
        scene.update_debug_objects((target_ik,), (gu.trans_quat_to_T(target_pos, target_quat),))

        qpos = franka.inverse_kinematics(
            link=ee_link,
            pos=target_pos,
            quat=target_quat,
            init_qpos=qpos,
            dofs_idx_local=motor_dofs_idx,
        )
        franka.control_dofs_position(qpos[motor_dofs_idx], dofs_idx_local=motor_dofs_idx)

        if is_gripper_closed[()]:
            franka.control_dofs_position(0.0, dofs_idx_local=finger_dofs_idx)
        else:
            franka.control_dofs_position(0.04, dofs_idx_local=finger_dofs_idx)
            

        scene.step()


if __name__ == "__main__":
    main()
