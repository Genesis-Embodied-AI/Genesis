import genesis as gs

"""
USD Parser System for Genesis

This package provides an extendable USD parser system with:
- UsdParserUtils: Utility functions for transforms, mesh conversion, etc.
- UsdParserContext: Context for tracking materials, articulations, and rigid bodies
- parse_all_materials: Function for parsing rendering materials
- parse_usd_rigid_entity: Unified parser for both articulations and rigid bodies
- import_from_stage: Main parser function for importing from a USD stage
- import_from_usd: Main parser function for importing from a USD file
"""

# Check if USD support is available before importing modules that depend on it
try:
    from pxr import Usd
except ImportError as e:
    gs.raise_exception_from("pxr module not found. Please install it with `pip install genesis-world[usd]`.", e)
else:
    # USD support is available - import the parser modules
    from .usd_stage import parse_usd_stage
    from .usd_context import UsdContext
    from .usd_rigid_entity import parse_usd
    from .usd_rendering_material_parser import parse_all_materials
