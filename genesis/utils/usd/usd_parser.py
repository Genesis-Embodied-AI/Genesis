"""
USD Parser

Main parser entrance for importing USD stages into Genesis scenes.
Provides the parse pipeline: materials -> articulations -> rigid bodies.
"""


from typing import List, Dict, Literal
import genesis as gs
from genesis.engine.entities.base_entity import Entity as GSEntity

try:
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf
except ImportError:
    gs.raise_exception("pxr module not found. Please install it with `pip install genesis[usd]`.")

from .usd_parser_context import UsdParserContext
from .usd_rendering_material_parser import UsdRenderingMaterialParser
from .usd_articulation_parser import UsdArticulationParser
from .usd_rigid_body_parser import UsdRigidBodyParser

class UsdParser:
    """
    Main USD parser for importing stages into Genesis scenes.
    """
    
    @staticmethod
    def import_from_stage(scene: gs.Scene, stage: Usd.Stage, vis_mode: Literal["visual", "collision"]) -> Dict[str, GSEntity]:
        """
        Import all entities from a USD stage into the scene.
        
        Parse Pipeline:
        1. Find all rendering materials, record them in UsdParserContext, return created gs Entity
        2. Find all articulations, record in UsdParserContext, return created gs Entity
        3. Find all rigid bodies (not in articulation), record in UsdParserContext, return created gs Entity
        4. Collect all gs Entity, return them to Users
        
        Parameters
        ----------
        scene : gs.Scene
            The scene to add entities to.
        stage : Usd.Stage
            The USD stage to import from.
            
        Returns
        -------
        List
            List of created entities (both articulations and rigid bodies).
        """
        stage_file = stage.GetRootLayer().identifier if stage.GetRootLayer() else ""
        
        # Create parser context
        context = UsdParserContext(stage)
        context._vis_mode = vis_mode
        
        
        # Return Values
        entities: Dict[str, GSEntity] = {}
        
        # Step 1: Parse all rendering materials
        material_parser = UsdRenderingMaterialParser(context)
        materials = material_parser.parse_all_materials()
        gs.logger.info(f"Parsed {len(materials)} materials from USD stage.")
        
        # Step 2: Find all articulations
        articulation_roots = UsdArticulationParser.find_all_articulation_roots(stage, context)
        gs.logger.info(f"Found {len(articulation_roots)} articulation(s) in USD stage.")
        
        for articulation_root in articulation_roots:
            morph = gs.morphs.USDArticulation(
                parser_ctx=context,
                file=stage_file,
                prim_path=str(articulation_root.GetPath()),
            )
            entity = scene.add_entity(morph)
            entities[str(articulation_root.GetPath())] = entity
            gs.logger.info(f"Imported articulation from prim: {articulation_root.GetPath()}")
        
        # Step 3: Find all rigid bodies (not in articulation)
        rigid_bodies = UsdRigidBodyParser.find_all_rigid_bodies(stage, context)
        gs.logger.info(f"Found {len(rigid_bodies)} rigid body(ies) in USD stage.")
        
        for rigid_body_prim in rigid_bodies:
            morph = gs.morphs.USDRigidBody(
                parser_ctx=context,
                file=stage_file,
                prim_path=str(rigid_body_prim.GetPath()),
            )
            entity = scene.add_entity(morph)
            entities[str(rigid_body_prim.GetPath())] = entity
            gs.logger.info(f"Imported rigid body from prim: {rigid_body_prim.GetPath()}")
        
        if not entities:
            gs.logger.warning("No articulations or rigid bodies found in USD stage.")
        return entities

# A simple entrance function for users
def import_from_usd(scene:gs.Scene, path:str, vis_mode: Literal["visual", "collision"]):
    stage = Usd.Stage.Open(path)
    return UsdParser.import_from_stage(scene, stage, vis_mode)