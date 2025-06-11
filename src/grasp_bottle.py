import argparse

import numpy as np
import genesis as gs
import pandas as pd

def make_step(scene, cam, franka, df):
    """フランカを目標位置に移動させるステップ関数"""
    scene.step()
    cam.render()
    # scene.clear_debug_objects()
    links_force_torque = franka.get_links_force_torque([9, 10]) # 手先のlocal_indexは9, 10
    links_force_torque = [x.item() for x in links_force_torque[0]] + [x.item() for x in links_force_torque[1]]
    print(links_force_torque[1])
    df.loc[len(df)] = [
        scene.t,
        links_force_torque[0], links_force_torque[1], links_force_torque[2],
        links_force_torque[3], links_force_torque[4], links_force_torque[5],
        links_force_torque[6], links_force_torque[7], links_force_torque[8],
        links_force_torque[9], links_force_torque[10], links_force_torque[11],
    ]
    # #force
    # scale = 0.1
    # scene.draw_debug_arrow(
    #     pos=franka.get_link("left_finger").get_pos().tolist(),
    #     vec=(links_force_torque[0][:3]*scale).tolist(),
    #     color=(1, 0, 0),
    # )
    # scene.draw_debug_arrow(
    #     pos=franka.get_link("right_finger").get_pos().tolist(),
    #     vec=(links_force_torque[1][:3]*scale).tolist(),
    #     color=(1, 0, 0),
    # )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", default="grasp_bottle_joint.mp4")
    parser.add_argument("-o", "--outfile", default="grasp_bottle_joint.csv")
    args = parser.parse_args()
    df = pd.DataFrame(columns=["step", "left_fx", "left_fy", "left_fz", "left_tx", "left_ty", "left_tz",
                           "right_fx", "right_fy", "right_fz", "right_tx", "right_ty", "right_tz"])
    ########################## init ##########################
    gs.init(backend=gs.cpu, precision="32", debug=True, logging_level="error")

    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=30,
        max_FPS=60,
    )

    scene = gs.Scene(
        viewer_options=viewer_options,
        rigid_options=gs.options.RigidOptions(dt=0.01),
        show_viewer=False,          # ★ GUI を開かない
    )

    # ---- 追加: オフスクリーンカメラ ------------------------
    cam = scene.add_camera(
        res=(1280, 720),
        pos=(3, -1, 1.5),
        lookat=(0.0, 0.0, 0.0),
        fov=30,
        GUI=False,                 # ★ 描画ウィンドウも開かない
    )
    # --------------------------------------------------------

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
    )
    bottle = scene.add_entity(
        material=gs.materials.Rigid(rho=300),
        morph=gs.morphs.URDF(
            file="urdf/3763/mobility_vhacd.urdf",
            scale=0.09,
            pos=(0.65, 0.0, 0.036),
            euler=(0, 90, 0),
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
    print("bottle mass:", bottle.get_mass())
    # print("end_effector:", end_effector)
    # print("right_finger:", franka.get_link("right_finger"))
    # move to pre-grasp pose
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.25]),
        quat=np.array([0, 1, 0, 0]),
    )
    qpos[-2:] = 0.04
    cam.start_recording()
    path = franka.plan_path(qpos)
    for waypoint in path:
        franka.control_dofs_position(waypoint)
        make_step(scene, cam, franka, df)             # ← 変更
    for _ in range(30):
        make_step(scene, cam, franka, df)             # ← 変更

    # reach
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.142]),
        quat=np.array([0, 1, 0, 0]),
    )
    franka.control_dofs_position(qpos[:-2], motors_dof)
    for _ in range(100):
        make_step(scene, cam, franka, df)             # ← 変更

    # grasp
    franka.control_dofs_position(qpos[:-2], motors_dof)
    franka.control_dofs_position(np.array([0, 0]), fingers_dof)
    for _ in range(100):
        make_step(scene, cam, franka, df)             # ← 変更

    # lift
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.3]),
        quat=np.array([0, 1, 0, 0]),
    )
    franka.control_dofs_position(qpos[:-2], motors_dof)
    franka.control_dofs_force(np.array([-5, -5]), fingers_dof)
    for _ in range(250):
        make_step(scene, cam, franka, df)             # ← 変更

    franka.control_dofs_position(qpos[:-2], motors_dof)
    franka.control_dofs_force(np.array([-10, -10]), fingers_dof)
    for _ in range(100):
        make_step(scene, cam, franka, df)             # ← 変更

    # ── 5. ハンドを 90度 回転 ──────────────────────────
    n_rot = 90      # 2π を 180 ステップ ⇒ 1 ステップ 2°
    j7_init = franka.get_qpos()[6]
    # link7 = franka.get_link("link7")
    for i in range(n_rot + 1):
        j7 = j7_init + np.deg2rad(i)
        franka.control_dofs_position(
            np.array([j7]),
            np.array([6]),
        )
        franka.control_dofs_force(np.array([-10, -10]), fingers_dof)
        make_step(scene, cam, franka, df)

    # ---- 追加: 録画終了・保存 -------------------------------
    cam.stop_recording(save_to_filename=args.video, fps=60)
    print(f"saved -> {args.video}")
    df.to_csv(args.outfile, index=False)
    print(f"saved -> {args.outfile}")
    # --------------------------------------------------------
if __name__ == "__main__":
    main()
