import argparse

import numpy as np
import genesis as gs
import pandas as pd

def make_step(scene, cam):
    """フランカを目標位置に移動させるステップ関数"""
    scene.step()
    cam.render()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", default="grasp_can.mp4")
    args = parser.parse_args()

    ########################## init ##########################
    # gs.init(backend=gs.cpu, debug=True, logging_level="error")
    gs.init(backend=gs.gpu)

    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=30,
        max_FPS=60,
    )

    scene = gs.Scene(
        viewer_options=viewer_options,
        rigid_options=gs.options.RigidOptions(dt=0.005),
        show_viewer=False,          # ★ GUI を開かない
    )

    # ---- 追加: オフスクリーンカメラ ------------------------
    cam = scene.add_camera(
        res=(1280, 720),
        # X 軸方向からのサイドビュー、Z を 0.1（缶の中心高さ程度）にして水平に
        pos=(3.0, 0.0, 0.1),
        lookat=(0.0, 0.0, 0.1),
        fov=30,
        GUI=False,
    )
    # --------------------------------------------------------

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
    )

    chips_can = scene.add_entity(
        material=gs.materials.Rigid(rho=300),
        morph=gs.morphs.Mesh(
            file="data/objects/001_chips_can/poisson/textured.obj",
            scale=1.0,
            pos=(0.65, 0.0, 0.036),
            euler=(0, 0, 0),
        ),
        # visualize_contact=True,
    )
    
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )

    ########################## build ##########################
    scene.build()

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)

    # Optional: set control gains
    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )

    end_effector = franka.get_link("hand")
    # move to pre-grasp pose
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.6]),
        quat=np.array([0, 1, 0, 0]),
    )
    qpos[-2:] = 0.105
    franka.set_dofs_position(qpos[:-2], motors_dof)
    franka.set_dofs_position(qpos[-2:], fingers_dof)
    cam.start_recording()
    #=================この中を調整========================
    # reach

    for i in range(300):
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=np.array([0.65, 0.0, 0.6-0.01*i]),
            quat=np.array([0, 1, 0, 0]),
        )
        franka.control_dofs_position(qpos[:-2], motors_dof)
        make_step(scene, cam)

    # grasp
    franka.control_dofs_position(qpos[:-2], motors_dof)
    franka.control_dofs_position(np.array([0, 0]), fingers_dof)
    for _ in range(100):
        make_step(scene, cam)

    # # lift
    # qpos = franka.inverse_kinematics(
    #     link=end_effector,
    #     pos=np.array([0.65, 0.0, 0.3]),
    #     quat=np.array([0, 1, 0, 0]),
    # )
    # franka.control_dofs_position(qpos[:-2], motors_dof)
    # franka.control_dofs_force(np.array([-5, -5]), fingers_dof)
    # for _ in range(100):
    #     make_step(scene, cam)

    # ---- 追加: 録画終了・保存 -------------------------------
    cam.stop_recording(save_to_filename=args.video, fps=60)
    print(f"saved -> {args.video}")
    # --------------------------------------------------------
if __name__ == "__main__":
    main()
