"""
USD Parser

Main parser entrance for importing USD stages into Genesis scenes.
Provides the parse pipeline: materials -> articulations -> rigid bodies.
"""

import copy
from typing import Dict, List, Literal

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

import genesis as gs
from genesis.options.morphs import USD

from .usd_articulation_parser import UsdArticulationParser
from .usd_parser_context import UsdParserContext
from .usd_rendering_material_parser import parse_all_materials
from .usd_rigid_body_parser import UsdRigidBodyParser


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
    1. Find all rendering materials, record them in UsdParserContext, return created gs Entity
    2. Find all articulations, record in UsdParserContext, return created gs Entity
    3. Find all rigid bodies (not in articulation), record in UsdParserContext, return created gs Entity
    4. Collect all gs Entity, return them to Users

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

    # Step 2: Find all articulations
    articulation_roots = UsdArticulationParser.find_all_articulation_roots(stage, context)
    gs.logger.debug(f"Found {len(articulation_roots)} articulation(s) in USD stage.")

    for articulation_root in articulation_roots:
        morph = copy.copy(usd_morph)
        morph.prim_path = str(articulation_root.GetPath())
        morph.parsing_type = "articulation"
        entity = scene.add_entity(morph, vis_mode=vis_mode, visualize_contact=visualize_contact)
        entities[str(articulation_root.GetPath())] = entity
        gs.logger.debug(f"Imported articulation from prim: {articulation_root.GetPath()}")

    # Step 3: Find all rigid bodies (not in articulation)
    rigid_bodies = UsdRigidBodyParser.find_all_rigid_bodies(stage, context)
    gs.logger.debug(f"Found {len(rigid_bodies)} rigid body(ies) in USD stage.")

    for rigid_body_prim in rigid_bodies:
        morph = copy.copy(usd_morph)
        morph.prim_path = str(rigid_body_prim.GetPath())
        morph.parsing_type = "rigid_body"
        entity = scene.add_entity(morph, vis_mode=vis_mode, visualize_contact=visualize_contact)
        entities[str(rigid_body_prim.GetPath())] = entity
        gs.logger.debug(f"Imported rigid body from prim: {rigid_body_prim.GetPath()}")

    if not entities:
        gs.logger.warning(f"No articulations or rigid bodies found in USD: {usd_morph.file}")
    return entities
