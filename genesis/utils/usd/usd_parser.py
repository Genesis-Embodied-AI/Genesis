"""
USD Parser

Main parser entrance for importing USD stages into Genesis scenes.
Provides the parse pipeline: materials -> articulations -> rigid bodies.
"""

from typing import Dict, Literal

from pxr import Usd

import genesis as gs
from genesis.options.morphs import USD

from .usd_parser_context import UsdParserContext
from .usd_rendering_material_parser import parse_all_materials
from .usd_rigid_entity_parser import parse_all_rigid_entities


def import_from_stage(
    scene: gs.Scene,
    stage: Usd.Stage | str,
    vis_mode: Literal["visual", "collision"],
    usd_morph: USD,
    visualize_contact: bool = False,
):
    """
    Import all entities from a USD stage or file into the scene.

    Parse Pipeline:
    1. Parse all rendering materials and record them in UsdParserContext
    2. Parse all rigid entities (articulations and rigid bodies) and return created gs Entities

    Parameters
    ----------
    scene : gs.Scene
        The scene to add entities to.
    stage : Usd.Stage | str
        The USD stage to import from, or a file path string to open.
    vis_mode : Literal["visual", "collision"]
        Visualization mode.
    usd_morph : USD
        USD morph configuration.
    visualize_contact : bool, optional
        Whether to visualize contact, by default False.

    Returns
    -------
    Dict[str, Entity]
        Dictionary of created entities (both articulations and rigid bodies) keyed by prim path.
    """
    from genesis.engine.entities.base_entity import Entity as GSEntity

    # Open stage if a file path is provided
    if isinstance(stage, str):
        stage = Usd.Stage.Open(stage)

    # Create parser context
    context = UsdParserContext(stage)
    context._vis_mode = vis_mode
    usd_morph.parser_ctx = context

    # Return Values
    entities: Dict[str, GSEntity] = {}

    # Step 1: Parse all rendering materials
    materials = parse_all_materials(context)
    gs.logger.debug(f"Parsed {len(materials)} materials from USD stage.")

    # Step 2: Parse all rigid entities (articulations and rigid bodies)
    entities = parse_all_rigid_entities(scene, stage, context, usd_morph, vis_mode, visualize_contact)
    gs.logger.debug(f"Parsed {len(entities)} rigid entities from USD stage.")

    if not entities:
        gs.logger.warning(f"No articulations or rigid bodies found in USD: {usd_morph.file}")
    return entities
