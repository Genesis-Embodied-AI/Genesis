from load_object_data import load_object_sheet 
from pet import pet
from pp import pp
from rubber import rubber
from aluminium import aluminium

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

gs.init(backend=gs.cpu)
coup_friction_list = [0.1, 1.0, 5.0]
tsdf_list = ["001_chips_can", "023_wine_glass", "029_plate", "033_spatula"]
google16k_list = ["028_skillet_lid"]

for object_data in object_data_list:
    
    if object_data['skip']:
        print(f"Skipping object ID: {object_data['id']}")
        continue
    if object_data['id'] in tsdf_list:
        object_path = f"data/objects/{object_data['id']}/tsdf/textured.obj"
        print(f"Using TSDF path for object ID: {object_data['id']}")
    elif object_data['id'] in google16k_list:
        object_path = f"data/objects/{object_data['id']}/google_16k/textured.obj"
        print(f"Using Google 16k path for object ID: {object_data['id']}")
    else:
        object_path = f"data/objects/{object_data['id']}/poisson/textured.obj"
    #for i in range(len(coup_friction_list)):
    for i in range(1):
        coup_friction = coup_friction_list[i]
        print(f"Processing object ID: {object_data['id']} with coup_friction: {coup_friction}")
        """
        aluminium(
            object_name=object_path,
            object_euler=object_data['object_euler'],
            object_scale=object_data['object_scale'],
            grasp_pos=object_data['franka_pos'],
            object_path=object_path,
            coup_friction=coup_friction,
        )
        
        # 各関数を呼び出す
        steel(
            object_name=object_path,
            object_euler=object_data['object_euler'],
            object_scale=object_data['object_scale'],
            grasp_pos=object_data['franka_pos'],
            object_path=object_path,
            coup_friction=coup_friction,
        )
        """
        pet(
            object_name=object_path,
            object_euler=object_data['object_euler'],
            object_scale=object_data['object_scale'],
            grasp_pos=object_data['franka_pos'],
            object_path=object_path,
            coup_friction=coup_friction,
        )
        """
        pp(
            object_name=object_path,
            object_euler=object_data['object_euler'],
            object_scale=object_data['object_scale'],
            grasp_pos=object_data['franka_pos'],
            object_path=object_path,
            coup_friction=coup_friction,
        )
        
        rubber(
            object_name=object_path,
            object_euler=object_data['object_euler'],
            object_scale=object_data['object_scale'],
            grasp_pos=object_data['franka_pos'],
            object_path=object_path,
            coup_friction=coup_friction,
        )
        """