from load_object_data import load_object_sheet 
from pet import pet
from pp import pp
from rubber import rubber

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

for object_data in object_data_list[1:10]:
    
    if object_data['skip']:
        print(f"Skipping object ID: {object_data['id']}")
        continue
    pet(
        object_name=object_data['id'],
        object_euler=object_data['object_euler'],
        object_scale=object_data['object_scale'],
        grasp_pos=object_data['franka_pos']
    )
    """
    pp(
        object_name=object_data['id'],
        object_euler=object_data['object_euler'],
        object_scale=object_data['object_scale'],
        grasp_pos=object_data['franka_pos']
    )
    rubber(
        object_name=object_data['id'],
        object_euler=object_data['object_euler'],
        object_scale=object_data['object_scale'],
        grasp_pos=object_data['franka_pos']
    )
    """
