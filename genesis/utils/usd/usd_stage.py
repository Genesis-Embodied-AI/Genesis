from typing import List

import genesis as gs

from .usd_context import UsdContext


def parse_usd_stage(stage: gs.morphs.USD) -> List[gs.morphs.USD]:
    """
    Parse a USD stage and extract all rigid entities (articulations and rigid bodies) as separate USD morphs.

    This function identifies all rigid entities in the USD stage using the stage's USD context,
    then creates a separate USD morph for each entity. Each morph is configured to point to
    the specific entity's prim path, allowing individual entities to be loaded and processed
    independently.

    Parameters
    ----------
    stage : gs.morphs.USD
        The USD morph containing the stage to parse. The morph must have a valid `usd_ctx`
        attribute that provides access to the USD context.

    Returns
    -------
    morphs: List[gs.morphs.USD]
        A list of USD morphs, one for each rigid entity found in the stage. Each morph is
        a copy of the input stage with its `prim_path` set to the corresponding entity's
        prim path. The list is guaranteed to be non-empty (raises an exception if no
        entities are found).
    """
    context: UsdContext = stage.usd_ctx

    # find all articulations and rigid bodies
    entity_prims = context.find_all_rigid_entities()
    gs.logger.debug(f"Found {len(entity_prims)} rigid entity(ies) from USD stage.")

    morphs: List[gs.morphs.USD] = []
    for entity_prim in entity_prims:
        entity_morph = stage.copy()
        entity_morph.prim_path = str(entity_prim.GetPath())
        morphs.append(entity_morph)

    if not morphs:
        gs.raise_exception("No entities found in USD stage.")

    return morphs
