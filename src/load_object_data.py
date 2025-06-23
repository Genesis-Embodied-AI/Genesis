import csv
import numpy as np

def load_object_sheet(filepath):
    """
    指定されたCSVファイルを読み込み、各列を適切なデータ型に変換して返します。
    列の順番が 'ID', 'skip', 'able to hold' となっているCSVに対応しています。

    Args:
        filepath (str): 読み込むCSVファイルのパス。

    Returns:
        list: 変換されたデータを含むリスト。各要素は以下の形式の辞書です。
              {
                  'id': str,
                  'skip': bool,
                  'able_to_hold': bool,
                  'franka_pos': numpy.ndarray,
                  'object_euler': numpy.ndarray,
                  'object_scale': float or list of float,
                  'franka_initial_z': float or None,
                  'remarks': str
              }
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)

        # ヘッダー行をスキップ
        next(reader)

        for row in reader:
            # 空行または列数が足りない行をスキップ
            # 最低でも ID, skip, able to hold, franka pos, object euler, object scale が必要 (6列目まで)
            if not row or len(row) < 6:
                continue

            # 1列目: 文字列 (ID)
            obj_id = row[0].strip()

            # 2列目: True/False (skip)
            skip = row[1].strip().lower() == 'yes'

            # 3列目: True/False (able to hold)
            able_to_hold = row[2].strip().lower() == 'yes'

            # 4列目: numpy配列 (franka pos)
            franka_pos_str = row[3].strip()
            if ',' in franka_pos_str:
                franka_pos = np.array([float(x.strip()) for x in franka_pos_str.split(',')])
            else:
                # スペース区切りの場合、複数の連続したスペースを考慮してsplit()
                franka_pos = np.array([float(x.strip()) for x in franka_pos_str.split()])

            # 5列目: numpy配列 (object euler)
            object_euler_str = row[4].strip()
            object_euler_str = object_euler_str.replace('(0)', '').strip()
            if '/' in object_euler_str:
                euler_components = []
                for val_str in object_euler_str.split():
                    try:
                        euler_components.append(float(val_str.split('/')[0]))
                    except ValueError:
                        euler_components.append(np.nan) # 変換できない場合はNaN
                object_euler = np.array(euler_components)
            else:
                if ',' in object_euler_str:
                    object_euler = np.array([float(x.strip()) for x in object_euler_str.split(',')])
                else:
                    # スペース区切りの場合、複数の連続したスペースを考慮してsplit()
                    object_euler = np.array([float(x.strip()) for x in object_euler_str.split()])


            # 6列目: 数字 (object scale) - スラッシュ区切りの場合があるため対応
            object_scale_str = row[5].strip()
            if '/' in object_scale_str:
                object_scale = [float(x.strip()) for x in object_scale_str.split('/') if x.strip()]
                if len(object_scale) == 1:
                    object_scale = object_scale[0] # 要素が1つの場合は単一の数値として保持
            else:
                try:
                    object_scale = float(object_scale_str)
                except ValueError:
                    object_scale = None # 数値に変換できない場合はNone

            # 7列目: frankaの最初のz軸のデータ（zを変えた場合）
            franka_initial_z = None
            if len(row) > 6 and row[6].strip():
                z_str = row[6].strip().lower().replace('z=', '').replace('z =', '').strip()
                try:
                    franka_initial_z = float(z_str)
                except ValueError:
                    if '/' in z_str:
                        try:
                            franka_initial_z = float(z_str.split('/')[0])
                        except ValueError:
                            pass
                    pass

            # 8列目: 備考 (存在しない場合もある)
            remarks = row[7] if len(row) > 7 else ""

            data.append({
                'id': obj_id,
                'skip': skip,
                'able_to_hold': able_to_hold,
                'franka_pos': franka_pos,
                'object_euler': object_euler,
                'object_scale': object_scale,
                'franka_initial_z': franka_initial_z,
                'remarks': remarks
            })
    return data