import genesis as gs

"""
USD Parser System for Genesis

This package provides an extendable USD parser system with:
- UsdParserUtils: Utility functions for transforms, mesh conversion, etc.
- UsdParserContext: Context for tracking materials, articulations, and rigid bodies
- parse_all_materials: Function for parsing rendering materials
- UsdArticulationParser: Parser for articulations
- UsdRigidBodyParser: Parser for rigid bodies
- import_from_stage: Main parser function for importing from a USD stage
- import_from_usd: Main parser function for importing from a USD file
"""

# Check if USD support is available before importing modules that depend on it
try:
    from pxr import Usd
except ImportError:
    gs.raise_exception_from("pxr module not found. Please install it with `pip install genesis-world[usd]`.")
else:
    # USD support is available - import the parser modules
    from .usd_parser import import_from_stage, import_from_stage
    from .usd_articulation_parser import parse_usd_articulation
    from .usd_rigid_body_parser import parse_usd_rigid_body
    from .usd_rendering_material_parser import parse_all_materials
