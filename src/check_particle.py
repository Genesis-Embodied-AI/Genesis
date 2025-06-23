import argparse
import numpy as np
import pandas as pd

from load_object_data import load_object_sheet 

import genesis as gs

# CSVファイルを読み込む
file_path = 'object_sheet.csv'
object_data_list = load_object_sheet(file_path)

"""
# 読み込んだデータの一部を表示して確認
for i, item in enumerate(object_data_list[:5]): # 最初の5つのエントリを表示
    print(f"--- Entry {i+1} ---")
    for key, value in item.items():
        print(f"{key}: {value}, type: {type(value)}")
    print("-" * 20)
"""

gs.init(backend=gs.cpu,logging_level="error")
coup_friction_list = [0.1, 1.0, 5.0]

def pet_poisson(object_name, object_euler, object_scale, grasp_pos, coup_friction=0.1):
    df = pd.DataFrame(columns=["step", "left_fx", "left_fy", "left_fz", "left_tx", "left_ty", "left_tz",
                           "right_fx", "right_fy", "right_fz", "right_tx", "right_ty", "right_tz"])
    ########################## init ##########################
    # gs.init(backend=gs.cpu, debug=True, logging_level="error")
    #gs.init(backend=gs.cpu)
    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=30,
        max_FPS=60,
    )
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=15,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=30,
        ),
        show_viewer=False,
        vis_options=gs.options.VisOptions(
            visualize_mpm_boundary=True,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(0.0, -0.1, -0.05),
            upper_bound=(0.75, 1.0, 1.0),
            grid_density=128,
        ),
    )
    # ---- 追加: オフスクリーンカメラ ------------------------
    # --------------------------------------------------------
    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
    )
    chips_can = scene.add_entity(
        material=gs.materials.MPM.ElastoPlastic( #PET
            E=2.45e6,
            nu=0.4,
            rho=1400,
            use_von_mises=True,
            von_mises_yield_stress=18000,
        ),
        morph=gs.morphs.Mesh(
            file=object_path,
            scale=object_scale, #record
            pos=(0.45, 0.45, 0.0),
            euler=object_euler, #record
        ),
    )

    ########################## build ##########################
    #scene.build()
    
def pet_tsdf(object_name, object_euler, object_scale, grasp_pos, coup_friction=0.1):
    object_path = f"data/objects/{object_name}/tsdf/textured.obj"
    print(f"object_path: {object_path}")
    df = pd.DataFrame(columns=["step", "left_fx", "left_fy", "left_fz", "left_tx", "left_ty", "left_tz",
                           "right_fx", "right_fy", "right_fz", "right_tx", "right_ty", "right_tz"])
    ########################## init ##########################
    # gs.init(backend=gs.cpu, debug=True, logging_level="error")
    #gs.init(backend=gs.cpu)
    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=30,
        max_FPS=60,
    )
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=15,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=30,
        ),
        show_viewer=False,
        vis_options=gs.options.VisOptions(
            visualize_mpm_boundary=True,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(0.0, -0.1, -0.05),
            upper_bound=(10.75, 10.0, 10.0),
            grid_density=128,
        ),
    )
    # ---- 追加: オフスクリーンカメラ ------------------------
    # --------------------------------------------------------
    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
    )
    chips_can = scene.add_entity(
        material=gs.materials.MPM.ElastoPlastic( #PET
            E=2.45e6,
            nu=0.4,
            rho=1400,
            use_von_mises=True,
            von_mises_yield_stress=18000,
        ),
        morph=gs.morphs.Mesh(
            file=object_path,
            scale=object_scale, #record
            pos=(0.45, 0.45, 0.0),
            euler=object_euler, #record
        ),
    )


    ########################## build ##########################
    #scene.build()
    
def pet_google(object_name, object_euler, object_scale, grasp_pos, coup_friction=0.1):
    object_path = f"data/objects/{object_name}/google_16k/textured.obj"
    print(f"object_path: {object_path}")
    df = pd.DataFrame(columns=["step", "left_fx", "left_fy", "left_fz", "left_tx", "left_ty", "left_tz",
                           "right_fx", "right_fy", "right_fz", "right_tx", "right_ty", "right_tz"])
    ########################## init ##########################
    # gs.init(backend=gs.cpu, debug=True, logging_level="error")
    #gs.init(backend=gs.cpu)
    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=30,
        max_FPS=60,
    )
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=15,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=30,
        ),
        show_viewer=False,
        vis_options=gs.options.VisOptions(
            visualize_mpm_boundary=True,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(0.0, -0.1, -0.05),
            upper_bound=(0.75, 1.0, 1.0),
            grid_density=128,
        ),
    )
    # ---- 追加: オフスクリーンカメラ ------------------------
    # --------------------------------------------------------
    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
    )
    chips_can = scene.add_entity(
        material=gs.materials.MPM.ElastoPlastic( #PET
            E=2.45e6,
            nu=0.4,
            rho=1400,
            use_von_mises=True,
            von_mises_yield_stress=18000,
        ),
        morph=gs.morphs.Mesh(
            file=object_path,
            scale=object_scale, #record
            pos=(0.45, 0.45, 0.0),
            euler=object_euler, #record
        ),
    )
    
import argparse
import numpy as np
import genesis as gs
import pandas as pd
def make_step(scene, cam, franka, df):
    """フランカを目標位置に移動させるステップ関数"""
    scene.step()
    if scene.t % 20 == 0:
        cam.render()
    links_force_torque = franka.get_links_force_torque([9, 10]) # 手先のlocal_indexは9, 10
    links_force_torque = [x.item() for x in links_force_torque[0]] + [x.item() for x in links_force_torque[1]]
    df.loc[len(df)] = [
        scene.t,
        links_force_torque[0], links_force_torque[1], links_force_torque[2],
        links_force_torque[3], links_force_torque[4], links_force_torque[5],
        links_force_torque[6], links_force_torque[7], links_force_torque[8],
        links_force_torque[9], links_force_torque[10], links_force_torque[11],
    ]
    # #force

def pet(object_name, object_euler, object_scale, grasp_pos, object_path, coup_friction=0.1):
    default_video_path = f"data/videos/pet/grasp_{object_name}_pet_{coup_friction}.mp4"
    default_outfile_path = f"data/csv/pet/grasp_{object_name}_pet_{coup_friction}.csv"
    print(f"object_path: {object_path}")
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", default=default_video_path)
    parser.add_argument("-o", "--outfile", default=default_outfile_path)
    args = parser.parse_args()
    df = pd.DataFrame(columns=["step", "left_fx", "left_fy", "left_fz", "left_tx", "left_ty", "left_tz",
                           "right_fx", "right_fy", "right_fz", "right_tx", "right_ty", "right_tz"])
    ########################## init ##########################
    # gs.init(backend=gs.cpu, debug=True, logging_level="error")
    #gs.init(backend=gs.cpu)
    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=30,
        max_FPS=60,
    )
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=15,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=30,
        ),
        show_viewer=False,
        vis_options=gs.options.VisOptions(
            visualize_mpm_boundary=True,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-10.0, -10.1, -10.05),
            upper_bound=(10.75, 10.0, 10.0),
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
    )
    # --------------------------------------------------------
    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
    )
    chips_can = scene.add_entity(
        material=gs.materials.MPM.ElastoPlastic( #PET
            E=2.45e6,
            nu=0.4,
            rho=1400,
            use_von_mises=True,
            von_mises_yield_stress=18000,
        ),
        morph=gs.morphs.Mesh(
            file=object_path,
            scale=object_scale, #record
            pos=(0.45, 0.45, 0.0),
            euler=object_euler, #record
        ),
    )



parser = argparse.ArgumentParser(description="Run specific functions based on command line arguments.")

# オプション引数を定義
# -t または --tsdf が指定された場合
parser.add_argument(
    '-t', '--tsdf',
    action='store_true',  # フラグとして機能させる (指定されたらTrueになる)
    help='Execute the TSDF function.'
)

parser.add_argument(
    '-g', '--google16k',
    action='store_true',  # フラグとして機能させる (指定されたらTrueになる)
    help='Execute the Google function.'
)

args = parser.parse_args()

tsdf_list = ["001_chips_can", "023_wine_glass", "029_plate", "033_spatula"]
google16k_list = ["028_skillet_lid"]

for object_data in object_data_list[0:]:
    
    if object_data['skip']:
        print(f"Skipping object ID: {object_data['id']}")
        continue
    #for i in range(len(coup_friction_list)):
    for i in range(1):
        coup_friction = coup_friction_list[i]
        print(f"Processing object ID: {object_data['id']} with coup_friction: {coup_friction}")
    if object_data['id'] in tsdf_list:
        object_path = f"data/objects/{object_data['id']}/tsdf/textured.obj"
        print(f"Using TSDF path for object ID: {object_data['id']}")
    elif object_data['id'] in google16k_list:
        object_path = f"data/objects/{object_data['id']}/google_16k/textured.obj"
        print(f"Using Google 16k path for object ID: {object_data['id']}")
    else:
        object_path = f"data/objects/{object_data['id']}/poisson/textured.obj"
    
        pet(
            object_name=object_data['id'],
            object_euler=object_data['object_euler'],
            object_scale=object_data['object_scale'],
            grasp_pos=object_data['franka_pos'],
            object_path=object_path,
            coup_friction=coup_friction,
        )
        """
        if args.tsdf:
            pet_tsdf(
                object_name=object_data['id'],
                object_euler=object_data['object_euler'],
                object_scale=object_data['object_scale'],
                grasp_pos=object_data['franka_pos'],
                coup_friction=coup_friction
            )
        elif args.google16k:
            pet_google(
                object_name=object_data['id'],
                object_euler=object_data['object_euler'],
                object_scale=object_data['object_scale'],
                grasp_pos=object_data['franka_pos'],
                coup_friction=coup_friction
            )
        else:
            pet_poisson(
                object_name=object_data['id'],
                object_euler=object_data['object_euler'],
                object_scale=object_data['object_scale'],
                grasp_pos=object_data['franka_pos'],
                coup_friction=coup_friction
            )
        """


