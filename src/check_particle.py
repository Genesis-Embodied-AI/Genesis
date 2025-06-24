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
google16k_list = ["003_cracker_box", "022_windex_bottle", "028_skillet_lid", "029_plate", "030_fork", "031_spoon", "032_knife", "035_power_drill", "036_wood_block", "037_scissors", "038_padlock", "040_large_marker", "042_adjustable_wrench", "043_phillips_screwdriver", "044_flat_screwdriver", "048_hammers", "049_small_clamp", "050_miduim_clamp", "051_large_clamp", "052_exstra_large_clamp", "053_mini_soccer_ball", "054_softtball", "055_baseball", "056_tennis_ball", "057_racquet_ball", "058_golf_ball", "059_chain"]


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




path = """ ,skip,able to hold,franka pos,object euler,object scale,frankaの最初のz軸のデータ（zを変えた場合）,備考,
001_chips_can,no,yes,"0.45, 0.45, 0.225","0, 0, 0",0.8,z=0.7,,
002_master_chef_can,no,yes,"0.45, 0.45, 0.125","0, 0, 0",0.6,,,
003_cracker_box,no,yes,"0.45, 0.45, 0.275","0, 0, 80",1.2,z = 0.79,,
004_sugar_box,no,yes,"0.45, 0.47, 0.225","0, 0, 120",1,z = 0.7,,
005_tomato_soup_can,no,yes,"0.44, 0.44, 0.125","0, 0, 120",1,,,
006_mustard_bottle,no,yes,"0.45, 0.45, 0.255","0, 0, 60",1,z = 0.73,持ち上げた後、落としてしまう,
007_tuna_fish_can,no,yes,"0.45, 0.45, 0.125","0, 0, 60",1,,,
008_pudding_box,no,yes,"0.45, 0.45, 0.125","0, 0, 0",1,,,
009_gelatin_box,no,yes,"0.45, 0.45, 0.12499999999999956","0, 0, 0",1,,,
010_potted_meat_can,no,yes,"0.45, 0.45, 0.12499999999999956","0, 0, 0",1,,,
011_banana,no,yes,"0.45, 0.45, 0.12499999999999956","0, 0, 0",1,,,
012_strawberry,no,yes,"0.45, 0.45, 0.12499999999999956","0, 0, 0",1,,,
013_apple,no,yes,"0.45, 0.45, 0.12499999999999956","0, 0, 0",1,,,
014_lemon,no,yes,"0.45, 0.45, 0.12499999999999956","0, 0, 0",1,,,
015_peach,no,yes,"0.45, 0.45, 0.12499999999999956","0, 0, 0",1,,,
016_pear,no,yes,"0.44, 0.44, 0.12499999999999956","0, 0, 0",1,,,
017_orange,no,yes,"0.45, 0.45, 0.12499999999999956","0, 0, 0",1,,,
018_plum,no,yes,"0.45, 0.45, 0.12499999999999956","0, 0, 0",1,,,
019_pitcher_base,no,no,"0.45, 0.45, 0.3249999999999996","0, 0, 0",1,0.8,,
021_bleach_cleanser,no,no,"0.44, 0.44, 0.2949999999999996","0, 0, 50",1,0.77,,
022_windex_bottle,yes,yes,"0.4, 0.4, 0.12499999999999956","0, 0, 30",1,,持ち上げた後、落としてしまう,
023_wine_glass,no,no,"0.45, 0.45, 0.12499999999999956","0, 0, 0",1,,,
024_bowl,no,yes,"0.45, 0.45, 0.12499999999999956","0, 0, 0",0.7,,,
025_mug,no,yes,"0.45, 0.45, 0.12499999999999956","0, 0, 0",0.7,,,
026_sponge,no,yes,"0.45, 0.45, 0.12499999999999956","0, 0, 20",1,,,
028_skillet_lid,no,yes,"0.45, 0.45, 0.12499999999999956","0, 0, 0",1,,,
029_plate,no,yes,"0.45, 0.45, 0.08499999999999963","0, 0, 0",0.3,0.56,,
030_fork,no,yes,"0.45, 0.45, 0.12499999999999956","0, 0, 0",1,,,
031_spoon,no,yes,"0.45, 0.45, 0.12499999999999956","0, 0, 0",1,,,
032_knife,no,yes,"0.45, 0.43, 0.12499999999999956","0, 0, 0",1.2,,,
033_spatula,no,yes,"0.45, 0.4, 0.12499999999999956","0, 0, 0",1,,,
035_power_drill,no,yes,"0.45, 0.45, 0.12499999999999956","0, 0, 0",1,,,
036_wood_block,no,yes,"0.45, 0.44, 0.22499999999999953","0, 0, 0",0.8,0.7,,
037_scissors,no,yes,"0.44, 0.45, 0.10499999999999954","0, 0, 90",1,0.58,,
038_padlock,no,yes,"0.45, 0.45, 0.12499999999999956","0, 0, 0",1,,,
039_key,yes,no,,,,,オブジェクトが存在しない,
040_large_marker,no,yes,"0.45, 0.43, 0.12499999999999956","0, 0, 90",1,,,
041_small_marker,no,yes,"0.45, 0.43, 0.12499999999999956","0, 0, 90",1,,,
042_adjustable_wrench,no,yes,"0.45, 0.44, 0.10499999999999954","0, 0, 50",1.2,0.58,,
043_phillips_screwdriver,no,yes,"0.45, 0.45, 0.12499999999999956","0, 0, 50",1,,,
044_flat_screwdriver,no,yes,"0.45, 0.45, 0.12499999999999956","0, 0, 50",1,,,
046_plastic_bolt,yes,no,,,,,オブジェクトが存在しない,
047_plastic_nut,yes,no,,,,,オブジェクトが存在しない,
048_hammer,no,yes,"0.35, 0.42, 0.12499999999999956","0, 0, 65",1.1,,,
049_small_clamp,yes,no,,,,,オブジェクトが存在しない,
050_medium_clamp,no,yes,"0.45, 0.45, 0.12499999999999956","0, 0, 0",1,,,
051_large_clamp,no,yes,"0.45, 0.45, 0.12499999999999956","0, 0, 0",1,,,
052_extra_large_clamp,no,yes,"0.45, 0.4, 0.12499999999999956","0, 0, 0",1,,,
053_mini_soccer_ball,no,yes,"0.43, 0.45, 0.12499999999999956","0, 0, 0",0.8,,,
054_softball,no,yes,"0.45, 0.45, 0.12499999999999956","0, 0, 0",1,,,
055_baseball,no,yes,"0.45, 0.45, 0.12499999999999956","0, 0, 0",1,,,
056_tennis_ball,no,yes,"0.45, 0.45, 0.12499999999999956","0, 0, 0",1,,,
057_racquetball,no,yes,"0.45, 0.3, 0.12499999999999956","0, 0, 0",1.5,,,
058_golf_ball,no,yes,"0.45, 0.45, 0.12499999999999956","0, 0, 0",1.5,,,
059_chain,no,no,"0.38, 0.38, 0.09499999999999953","0, 0, 0",1.5,0.57,チェーン難しいな,
061_foam_brick,no,yes,0.45 0.45 0.125,0 0 110,1,, grasp_061_foam_brick.mp4,
062_dice,no,yes,0.45 0.45 0.125,0 0 0,1.5,,grasp_062.mp4,
063-a_marbles,no,yes,0.45 0.45 0.125,0 0 0,1,, grasp_063_a.mp4,
063-b_marbles,yes,no,,,,,,オブジェクトが存在しない
063-c_marbles,yes,no,,,,,,オブジェクトが存在しない
063-d_marbles,yes,yes,0.45 0.45 0.125,0 0 0,2.5,, grasp_063_d.mp4,
063-f_marbles,yes,no,,,,,,オブジェクトが存在しない
065-a_cups,no,yes,0.45 0.45 0.115,0 0 0,1,z=0.59,grasp_065_a.mp4,
065-b_cups,no,no,0.45 0.45 0.125 ,0 0 0,1,z=0.6, grasp_065_b.mp4,
065-c_cups,no,yes,0.45 0.45 0.125,0 0 0,1,,grasp_065_c.mp4,持ち上げることには成功したのですが、カップのモデルがグリッパーに張り付いてしまう不具合が発生しています。
065-d_cups,no,yes,0.45 0.45 0.125,0 0 0,0.7,,grasp_065_d.mp4,持ち上げることには成功したのですが、カップのモデルがグリッパーに張り付いてしまう不具合が発生しています。
065-e_cups,no,no,0.45 0.45 0.125,0 0 0,0.5/0/7/0/8/1,,,カップのモデルがグリッパーに張り付いてしまう不具合が発生しています。
065-f_cups,no,yes,0.44 0.44 0.125,0 0 0,0.8,,,持ち上げることには成功したのですが、カップのモデルがグリッパーに張り付いてしまう不具合が発生しています。
065-g_cups,yes,,,,,,,
065-h_cups,yes,no,,,,,,
065-i_cups,no,yes,0.45 0.45 0.115,0 0 0,0.4,z=0.59,grasp_065_i.mp4,グリッパーでカップの壁を、内側と外側から挟んで掴む方が良いですか？
065-j_cups,no,yes,0.45 0.45 0.115,0 0 40,0.35,z=0.59, grasp_065_j.mp4,
070-a_colored_wood_blocks,no,yes,0.45 0.45 0.125,0 0 0,0.3,,,オブジェクトモデルが正常に生成されていないのかもしれません。
070-b_colored_wood_blocks,yes,no,,,,, grasp_070_a.mp4,オブジェクトが存在しない
071_nine_hole_peg_test,no,yes,0.45 0.45 0.115,0 0 45,0.5,z=0.59, grasp_071.mp4,
072-a_toy_airplane,no,yes,0.45 0.45 0.125,0 0 45,0.8,, grasp_072_a.mp4,
072-b_toy_airplane,yes,no,0.45 0.45 ,0 0 0,1,,grasp_072_b.mp4,オブジェクトを生成した時点で転倒してしまい、修正ができません。
072-c_toy_airplane,no,yes,0.42 0.42 0.135,0 0 40(0),0.8/1,z=0.61,grasp_072_c.mp4,オブジェクトの把持には成功しましたが、角度と位置が良くありません。
072-d_toy_airplane,no,yes,0.45 0.45 0.125,0 0 0,0.9,,grasp_072_d.mp4,
072-e_toy_airplane,yes,no,0.45 0.45,0/n 0/n 0/n,0.7/1/1.2/1.5,,grasp_072_e.mp4,オブジェクトを生成した時点で転倒してしまい、修正ができません。
072-f_toy_airplane,no,yes,0.45 0.45 0.125,0 0 0,1.5,,grasp_072_f.mp4,
072-g_toy_airplane,yes,no,,,,,,オブジェクトが存在しない
072-h_toy_airplane,no,yes,0.45 0.45 0.125,0 0 0,0.9,,grasp_072_h.mp4,持ち上げるには成功したものの、オブジェクトが初期状態から不安定なため、X軸とY軸を固定して平面上に設置することができません。
072-i_toy_airplane,no,yes,0.45 0.45 0.125,0 0 0,1,, grasp_072_i.mp4,
072-j_toy_airplane,no,yes,0.45 0.45 0.125,-30 0 0,1,,grasp_072_j.mp4,
072-k_toy_airplane,no,yes,0.45 0.45 0.115,0 0 0,1,z=0.59, grasp_072_k.mp4,この物体は特殊な形状をしており、X軸パラメータの変更時に反転する問題があります。直接把持は可能ですが、実世界状況には即していません。
073-a_lego_duplo,no,yes,0.45 0.45 0.125,0 0 40 ,1,,grasp_073_a.mp4,
073-b_lego_duplo,no,yes,0.45 0.45 0.125,0 0 40,1,,grasp_073_b.mp4,
073-c_lego_duplo,no,yes,0.45 0.45 0.115,0 0 40,1.5,z=0.59,grasp_073_c.mp4,この物体はアーチ型で、中央から持ち上げることができまが、ちょっと不安定です。
073-d_lego_duplo,no,yes,0.45 0.45 0.125,0 0 0,0.8,,grasp_073_d.mp4,
073-e_lego_duplo,no,yes,0.45 0.45 0.135,0 0 40,1.5,z=0.61, grasp_073_e.mp4,この物体はアーチ型で、何度も試しましたが、中央から持ち上げることができません。
073-f_lego_duplo,no,yes,0.45 0.45 0.128,0 0 45,1,Z=0.603,grasp_073_f.mp4,
073-g_lego_duplo,no,yes,0.45 0.45 0.13,0 0 -45,1,z=0.605,grasp_073_g.mp4,
073-h_lego_duplo,no,yes,0.45 0.45 0.125,0 0 -55,1,,grasp_073_h.mp4,
073-i_lego_duplo,no,yes,0.45 0.45 0.135,0 0 -60,1,z=0.61,grasp_073_i.mp4,
073-j_lego_duplo,no,yes,0.45 0.45 0.125,0 0 -20,0.9,,grasp_073_j.mp4,
073-k_lego_duplo,no,yes,0.45 0.45 0.125,0 0 0,1/1.2/1.5,, grasp_073_k.mp4,
073-l_lego_duplo,no,yes,0.45 0.45 0.125,0 0 0,1,, grasp_073_l.mp4,
073-m_lego_duplo,no,yes,0.45 0.45 0.125,-30 0 -60,0.48,,grasp_073_m.mp4,
076_timer,no,yes,0.45 0.45 0.121,0 0 -50,0.66,z=0.596,grasp_076_timer.mp4,
077_rubiks_cube,no,yes,0.45 0.45 0.135,0 0 -55,0.6/0.7/0.8/0.9,z=0.61,grasp_077_rubiks_cube.mp4," """