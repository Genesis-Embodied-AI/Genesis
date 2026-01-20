import genesis as gs

# Check if USD support is available before importing modules that depend on it
try:
    from pxr import Usd
except ImportError as e:
    gs.raise_exception_from("pxr module not found. Please install it with `pip install genesis-world[usd]`.", e)
else:
    # USD support is available - import the parser modules
    from .usd_stage import parse_usd_stage
    from .usd_rigid_entity import parse_usd
    from .usd_context import UsdContext