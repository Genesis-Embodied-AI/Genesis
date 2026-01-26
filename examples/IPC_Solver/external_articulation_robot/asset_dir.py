import os 
import pathlib

class AssetDir:
    this_file = pathlib.Path(os.path.dirname(__file__)).resolve()
    _assets_path = pathlib.Path(this_file / '../../../genesis/assets/').resolve()

    @staticmethod
    def asset_path():
        return str(AssetDir._assets_path)
    
    @staticmethod
    def output_path(file):
        file_dir = (pathlib.Path(file).parent / 'output').absolute()
        return str(file_dir)

