"""
USD Stage

Main parser entrance for importing USD stages into Genesis scenes.
Provides the parse pipeline: materials -> articulations -> rigid bodies.
"""

from typing import List

import genesis as gs

from .usd_context import UsdContext


def parse_usd_stage(stage: gs.morphs.USD) -> List[gs.morphs.FileMorph]:
    context: UsdContext = stage.usd_ctx

    # # parse all rendering materials
    # context.find_all_materials()
    # gs.logger.debug(f"Parsed {len(context.materials)} materials from USD stage.")

    # find all articulations and rigid bodies
    entity_prims = context.find_all_rigid_entities()
    gs.logger.debug(f"Found {len(entity_prims)} rigid entity(ies) from USD stage.")

    morphs: List[gs.morphs.FileMorph] = []
    for entity_prim in entity_prims:
        entity_morph = stage.copy()
        entity_morph.prim_path = str(entity_prim.GetPath())
        morphs.append(entity_morph)

    if not morphs:
        gs.logger.warning("No entities found in USD stage.")

    return morphs
