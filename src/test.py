import argparse

import numpy as np
import genesis as gs
import pandas as pd
import os




def make_step(scene, cam):
    """フランカを目標位置に移動させるステップ関数"""
    scene.step()
    cam.render()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", default="grasp_can.mp4")
    parser.add_argument("-o", "--output_csv", default="simulation_results.csv", help="Output CSV file name") # CSV 파일명 인자 추가
    args = parser.parse_args()

    
    ########################## init ##########################
    # gs.init(backend=gs.cpu, debug=True, logging_level="error")
    gs.init(backend=gs.cpu, logging_level="error") #change gs.gpu to gs.cpu!!
    
    
    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=30,
        max_FPS=60,
    )


    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=5e-3,
            substeps=15,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=30,
            max_FPS=60,
        ),
        #show_viewer=False,
        show_viewer=True,
        vis_options=gs.options.VisOptions(
            visualize_mpm_boundary=True,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(0.55, -0.1, -0.05),
            upper_bound=(0.75, 0.1, 0.3),
            grid_density=128,
        ),
    )


    # ---- 追加: オフスクリーンカメラ ------------------------
    cam = scene.add_camera(
        res=(1280, 720),
        # X 軸方向からのサイドビュー、Z を 0.1（缶の中心高さ程度）にして水平に
        pos=(2.0, 2.0, 0.1),
        lookat=(0.0, 0.0, 0.1),
        fov=30,
        GUI=False,
    )
    # --------------------------------------------------------


    #以下を修正する
    
    # move to pre-grasp pose
    x = 0.45
    y = 0.45
    z = 0.6
    first_z = z

    # sacle of object
    scale_obj = 1.0

    # position of object
    pos_x = 0.45
    pos_y = 0.45
    pos_z = 0.0

    # euler of object
    euler_x = 0
    euler_y = 0
    euler_z = 0


    #ここのファイル名を変えるだけでパスを変えます。
    current_object_name = "010_potted_meat_can"
    


    print("current_object_name : ", current_object_name)

    print(f"first_franka (x, y, z): , {x}, {y}, {z}")


    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
    )


    chips_can = scene.add_entity(
        material=gs.materials.Rigid(rho=300),
        morph=gs.morphs.Mesh(
            file = f"data/objects/{current_object_name}/poisson/textured.obj",
            scale = scale_obj, #record
            pos=(pos_x, pos_y, pos_z),
            euler=(euler_x, euler_y, euler_z)
            #record
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

    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([x, y, z]),
        quat=np.array([0, 1, 0, 0]),
    )


    qpos[-2:] = 0.07


    franka.set_dofs_position(qpos[:-2], motors_dof)
    franka.set_dofs_position(qpos[-2:], fingers_dof)
    cam.start_recording()


    #=================この中を調整========================
    # reach
    for i in range (100):
        franka.set_dofs_position(qpos[:-2], motors_dof)
        franka.set_dofs_position(qpos[-2:], fingers_dof)
        make_step(scene, cam)

    for i in range(95):
        #record optimized moments
        z -= 0.005

        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=np.array([x, y, z]),
            quat=np.array([0, 1, 0, 0]),
        )

        qpos[-2:] = 0.07
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(qpos[-2:], fingers_dof)
        make_step(scene, cam)


    print(f"\nfinal_franka (x, y, z): , {x}, {y}, {z}")

    print("\nscale of object : ", scale_obj)
    print(f"pos(x, y, z) : , {pos_x}, {pos_y}, {pos_z}")
    print(f"euler(x, y, z) : , {euler_x}, {euler_y}, {euler_z}")
    

    # ===== CSV ファイル保存のためのデータ準備と保存 ======
    # 'able to hold' はシミュレーション結果によって異なるため、ここでは一時的に 'yes' に設定しています。
    # 'frankaの最初のz軸のデータ (zを変えた場合)' はFカラムのデータであり、シミュレーションの目的に応じて値を設定してください。
    # ここでは 'z' 変数が変更された最終的なZ値を表すと仮定し、`first_z` を使用します。

    # データ行を辞書形式で作成します。
    data_row = {
        'object': current_object_name,
        'able to hold': 'yes', # この値はシミュレーションの成否に応じて動的に変更されるべきです。
        'franka pos': f"{x}, {y}, {z}",
        'object euler': f"{euler_x}, {euler_y}, {euler_z}",
        'object scale': scale_obj,
        'frankaの最初のz軸のデータ (zを変えた場合)': first_z # 例として first_z の値を使用
    }

    # 現在のデータ行をDataFrameに変換
    new_df_row = pd.DataFrame([data_row])

    # 既存のCSVファイルを読み込む、または空のDataFrameを作成
    if os.path.exists(args.output_csv):
        existing_df = pd.read_csv(args.output_csv)
    else:
        # ファイルが存在しない場合、空のDataFrameを作成し、カラムを定義します。
        existing_df = pd.DataFrame(columns=new_df_row.columns)

    # 'object' カラムを基準に現在のオブジェクトがすでに存在するか確認
    if current_object_name in existing_df['object'].values:
        # すでに存在する場合、その行を更新
        # locを使用して特定の条件に一致する行のデータを更新します。
        existing_df.loc[existing_df['object'] == current_object_name, list(data_row.keys())] = new_df_row.values[0]
        print(f"\n既存の '{current_object_name}' のエントリが {args.output_csv} で更新されました")
    else:
        # 存在しない場合、新しい行を追加
        # pandas 2.0以降では concat が推奨されます。
        existing_df = pd.concat([existing_df, new_df_row], ignore_index=True)
        print(f"\n新しい '{current_object_name}' のエントリが {args.output_csv} に追加されました")

    # header=True に設定して、常にヘッダーを含めます。
    existing_df.to_csv(args.output_csv, index=False, header=True)

    # ===================================================




    for _ in range(10):
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(qpos[-2:], fingers_dof)
        make_step(scene, cam)

    # grasp
    for _ in range(100):
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(np.array([0, 0]), fingers_dof)
        make_step(scene, cam)

    for i in range(100):
        z += 0.005
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=np.array([x, y, z]),
            quat = np.array([0, 1, 0, 0]),
        )
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(np.array([0, 0]), fingers_dof)
        make_step(scene, cam)


    # # lift
    # qpos = franka.inverse_kinematics(
    #     link=end_effector,
    #     pos=np.array([0.45, 0.0, 0.3]),
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