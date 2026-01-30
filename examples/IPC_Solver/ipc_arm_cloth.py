"""
Keyboard Controls:
↑	- Move Forward (North)
↓	- Move Backward (South)
←	- Move Left (West)
→	- Move Right (East)
n	- Move Up
m	- Move Down
j/k	- Yaw Left/Right (Rotate around Z axis)
i/o	- Pitch Up/Down (Rotate around Y axis)
l/;	- Roll Left/Right (Rotate around X axis)
u	- Reset Scene
space	- Press to close gripper, release to open gripper
esc	- Quit
"""

import argparse
import logging

import genesis as gs
import numpy as np
from scipy.spatial.transform import Rotation as R

from genesis.vis.keybindings import Key, KeyAction, Keybind
from huggingface_hub import snapshot_download


def main():
    gs.init(backend=gs.gpu, logging_level=logging.INFO, performance_mode=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--vis_ipc", action="store_true", default=False)
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    dt = 2e-2

    coupler_options = gs.options.IPCCouplerOptions(
        dt=dt,
        gravity=(0.0, 0.0, -9.8),
        ipc_constraint_strength=(100.0, 100.0),
        contact_friction_mu=0.5,
        fem_fem_friction_mu=0.00,
        contact_d_hat=0.001,
        IPC_self_contact=False,
        contact_enable=True,
        disable_genesis_contact=True,
        disable_ipc_logging=True,
        newton_semi_implicit_enable=False,  # True: you will see time stealing artifact
        enable_ipc_gui=args.vis_ipc,
        line_search_max_iter=8,
        line_search_report_energy=False,
        newton_velocity_tol=1e-1,
        newton_transrate_tol=1,
        linear_system_tol_rate=1e-3,
        contact_resistance=1e7,
    )

    rigid_options = gs.options.RigidOptions(
        enable_collision=False,  # Disable rigid collision when using IPC
    )
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=(0.0, 0.0, -9.8)),
        rigid_options=rigid_options,
        coupler_options=coupler_options,
        show_viewer=args.vis,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, -1.0, 1.5),
            camera_lookat=(0.5, 0.0, 0.2),
            camera_fov=40,
        ),
    )

    # Add Franka robot
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=(0.0, 0.0, 0.005),
        ),
        vis_mode="collision",
    )

    scene.sim.coupler.set_entity_coupling_type(
        entity=franka,
        coupling_type="two_way_soft_constraint",
    )
    scene.sim.coupler.set_ipc_coupling_link_filter(
        entity=franka,
        link_names=["left_finger", "right_finger"],
    )
    cloth_asset_path = snapshot_download(
        repo_type="dataset",
        repo_id="Genesis-Intelligence/assets",
        revision="main",
        allow_patterns="grid*.obj",
        max_workers=1,
    )

    scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{cloth_asset_path}/grid40x40.obj",
            scale=0.5,
            pos=(0.5, 0.0, 0.1),
            euler=(90, 0, 0),
        ),
        material=gs.materials.FEM.Cloth(
            E=6e4,
            nu=0.49,
            rho=200,
            thickness=0.001,
            bending_stiffness=10.0,
        ),
        surface=gs.surfaces.Plastic(color=(0.3, 0.1, 0.8, 1.0), double_sided=True),
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{cloth_asset_path}/grid20x20.obj",
            scale=0.3,
            pos=(0.5, 0.0, 0.14),
            euler=(90, 0, 0),
        ),
        material=gs.materials.FEM.Cloth(
            E=6e4,
            nu=0.49,
            rho=200,
            thickness=0.001,
            bending_stiffness=40.0,
        ),
        surface=gs.surfaces.Plastic(color=(0.3, 0.5, 0.8, 1.0), double_sided=True),
    )

    # Add 16 rigid cubes uniformly distributed under the cloth (4x4 grid)
    cube_size = 0.05
    cube_height = 0.02501
    grid_spacing = 0.15

    for i in range(4):
        for j in range(4):
            x = (i + 1.7) * grid_spacing
            y = (j - 1.5) * grid_spacing
            box = scene.add_entity(
                morph=gs.morphs.Box(
                    pos=(x, y, cube_height),
                    size=(cube_size, cube_size, cube_size),
                    fixed=True,
                ),
                material=gs.materials.Rigid(rho=500, friction=0.3),
                surface=gs.surfaces.Plastic(color=(0.8, 0.3, 0.2, 0.8)),
            )
            scene.sim.coupler.set_entity_coupling_type(
                entity=box,
                coupling_type="ipc_only",
            )

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)

    ee_link = franka.get_link("hand")
    target_entity = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            scale=0.15,
            collision=False,
        ),
        surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
    )

    target_init_pos = np.array([0.5, 0.0, 0.6], dtype=np.float32)
    target_init_R = R.from_euler("y", np.pi)
    target_pos = target_init_pos.copy()
    target_R = target_init_R

    dpos = 0.003
    drot = 0.02

    plane = scene.add_entity(
        gs.morphs.Plane(),
    )

    scene.build()

    print("Scene built successfully!")

    if args.vis:
        if scene.viewer is None:
            gs.logger.warning("Viewer is not active. Keyboard input requires the Genesis viewer.")
            return

        # Mutable state for keybind callbacks (list refs so closures can update)
        target_R_ref = [target_R]
        gripper_close_ref = [False]
        is_running = True

        def move(dpos_xyz: tuple[float, float, float]):
            target_pos[:] += np.array(dpos_xyz, dtype=np.float32)

        def rotate_yaw(delta: float):
            target_R_ref[0] = R.from_euler("z", delta) * target_R_ref[0]

        def rotate_pitch(delta: float):
            target_R_ref[0] = R.from_euler("y", delta) * target_R_ref[0]

        def rotate_roll(delta: float):
            target_R_ref[0] = R.from_euler("x", delta) * target_R_ref[0]

        def reset_scene():
            target_pos[:] = target_init_pos
            target_R_ref[0] = target_init_R
            target_quat = target_R_ref[0].as_quat(scalar_first=True)
            target_entity.set_qpos(np.concatenate([target_pos, target_quat]))
            q, _ = franka.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat, return_error=True)
            franka.control_dofs_position(q[:-2], motors_dof)

        def set_gripper(close: bool):
            gripper_close_ref[0] = close

        def stop():
            nonlocal is_running
            is_running = False

        scene.viewer.register_keybinds(
            Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=move, args=((-dpos, 0, 0),)),
            Keybind("move_back", Key.DOWN, KeyAction.HOLD, callback=move, args=((dpos, 0, 0),)),
            Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=move, args=((0, -dpos, 0),)),
            Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=move, args=((0, dpos, 0),)),
            Keybind("move_up", Key.N, KeyAction.HOLD, callback=move, args=((0, 0, dpos),)),
            Keybind("move_down", Key.M, KeyAction.HOLD, callback=move, args=((0, 0, -dpos),)),
            Keybind("yaw_left", Key.J, KeyAction.HOLD, callback=rotate_yaw, args=(drot,)),
            Keybind("yaw_right", Key.K, KeyAction.HOLD, callback=rotate_yaw, args=(-drot,)),
            Keybind("pitch_up", Key.I, KeyAction.HOLD, callback=rotate_pitch, args=(drot,)),
            Keybind("pitch_down", Key.O, KeyAction.HOLD, callback=rotate_pitch, args=(-drot,)),
            Keybind("roll_left", Key.L, KeyAction.HOLD, callback=rotate_roll, args=(drot,)),
            Keybind("roll_right", Key.SEMICOLON, KeyAction.HOLD, callback=rotate_roll, args=(-drot,)),
            Keybind("reset_scene", Key.U, KeyAction.PRESS, callback=reset_scene),
            Keybind("close_gripper", Key.SPACE, KeyAction.PRESS, callback=set_gripper, args=(True,)),
            Keybind("open_gripper", Key.SPACE, KeyAction.RELEASE, callback=set_gripper, args=(False,)),
            Keybind("quit", Key.ESCAPE, KeyAction.PRESS, callback=stop),
        )

        try:
            while is_running and scene.viewer.is_alive():
                target_quat = target_R_ref[0].as_quat(scalar_first=True)
                target_entity.set_qpos(np.concatenate([target_pos, target_quat]))
                q, _ = franka.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat, return_error=True)
                franka.control_dofs_position(q[:-2], motors_dof)

                if gripper_close_ref[0]:
                    franka.control_dofs_velocity(np.array([-0.1, -0.1]), fingers_dof)
                else:
                    franka.control_dofs_velocity(np.array([0.1, 0.1]), fingers_dof)

                scene.step()
        except KeyboardInterrupt:
            gs.logger.info("Simulation interrupted, exiting.")
        finally:
            gs.logger.info("Simulation finished.")
    else:
        scene.step()


if __name__ == "__main__":
    main()
