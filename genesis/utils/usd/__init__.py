"""
USD Parser System for Genesis

This package provides an extendable USD parser system with:
- UsdParserUtils: Utility functions for transforms, mesh conversion, etc.
- UsdParserContext: Context for tracking materials, articulations, and rigid bodies
- UsdRenderingMaterialParser: Parser for rendering materials
- UsdArticulationParser: Parser for articulations
- UsdRigidBodyParser: Parser for rigid bodies
- UsdParser: Main parser entrance with import_from_stage function
"""

# Check if USD support is available before importing modules that depend on it
try:
    from pxr import Usd
except ImportError:
    # USD support not available - don't import dependent modules
    pass
else:
    # USD support is available - import the parser modules
    from .usd_parser import UsdParser
    from .usd_parser import import_from_usd
    from .usd_articulation_parser import parse_usd_articulation
    from .usd_rigid_body_parser import parse_usd_rigid_body
