import genesis as gs
from pxr import Usd
from genesis.utils import usda
from genesis.utils import usd_parser_utils as usd_utils

def import_from_usd_stage(path:str):
    stage = Usd.Stage.Open(path)
    context = usd_utils.UsdParserContext(stage)
    # ... [TODO]
    usda.parse_usd_materials(stage, context)
    
    
    # 1. Parse materials
    
    # 2. classify the prims:
    # - ArticulationRootAPI -> articulation
    # - Top RigidBodyAPI -> rigid body
    
    # - Others
    pass