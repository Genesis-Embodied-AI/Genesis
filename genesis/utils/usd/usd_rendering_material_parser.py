"""
USD Rendering Material Parser

Parser for extracting and parsing rendering materials from USD stages.
"""

from pxr import Usd, UsdShade

import genesis as gs

from . import usda
from .usd_parser_context import UsdParserContext


def parse_all_materials(context: UsdParserContext) -> dict:
    """
    Find all materials in the USD stage and parse them.

    Parameters
    ----------
    context : UsdParserContext
        The parser context to store materials in.

    Returns
    -------
    dict
        The materials dictionary (same as context.materials).
        Key: material_id (str) - unique identifier for the material
        Value: tuple of (material_surface, uv_name) - parsed material surface and UV name
    """
    stage = context.stage
    materials = context.materials
    default_surface = gs.surfaces.Default()

    # Parse materials from the stage
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Material):
            material_usd = UsdShade.Material(prim)
            material_spec = prim.GetPrimStack()[-1]
            material_id = material_spec.layer.identifier + material_spec.path.pathString

            if material_id not in materials:
                material, uv_name, require_bake = usda.parse_usd_material(material_usd, default_surface)
                materials[material_id] = (material, uv_name)
                if require_bake:
                    gs.logger.debug(
                        f"Material {material_id} requires baking (not yet implemented in context-based parsing)"
                    )

    return materials
