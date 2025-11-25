import genesis as gs
from pxr import Usd
from genesis.utils import usda
from genesis.utils import usd_parser_utils as usd_utils
from genesis.utils import usd_physics

def import_from_usd_stage(scene: gs.Scene, path: str, surface: gs.surfaces.Surface = None):
    """
    Import all articulations and rigid bodies from a USD stage into the scene.
    
    Parameters
    ----------
    scene : gs.Scene
        The scene to add entities to.
    path : str
        Path to the USD file.
    surface : gs.surfaces.Surface, optional
        Surface to use for entities. If None, uses gs.surfaces.Default().
    
    Returns
    -------
    entities : list
        List of created entities (both articulations and rigid bodies).
    """
    if surface is None:
        surface = gs.surfaces.Default()
    
    stage = Usd.Stage.Open(path)
    context = usd_utils.UsdParserContext(stage)
    # 0. Parse materials to context for later use
    usda.parse_materials_to_context(context, surface)
    
    entities = []
    
    # WITH Physics API
    
    # 1. Find all articulation roots
    articulation_roots = usd_physics.UsdArticulationParser.find_all_articulation_roots(stage)
    
    # 2. Find all top-level rigid bodies (not part of articulations)
    rigid_bodies = usd_physics.UsdRigidBodyParser.find_all_rigid_bodies(stage)
    
    # 3. Import articulations
    for articulation_root in articulation_roots:
        morph = gs.morphs.USDArticulation(
            parser_ctx=context,
            file=path,
            prim_path=str(articulation_root.GetPath()),
        )
        entity = scene.add_entity(morph, surface=surface)
        entities.append(entity)
        gs.logger.info(f"Imported articulation from prim: {articulation_root.GetPath()}")
    
    # 4. Import rigid bodies
    for rigid_body_prim in rigid_bodies:
        morph = gs.morphs.USDRigidBody(
            parser_ctx=context,
            file=path,
            prim_path=str(rigid_body_prim.GetPath()),
        )
        entity = scene.add_entity(morph, surface=surface)
        entities.append(entity)
        gs.logger.info(f"Imported rigid body from prim: {rigid_body_prim.GetPath()}")
    
    if not entities:
        gs.logger.warning(f"No articulations or rigid bodies found in USD file: {path}")
    
    return entities