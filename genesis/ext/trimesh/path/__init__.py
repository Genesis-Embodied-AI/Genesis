"""
trimesh.path
-------------

Handle 2D and 3D vector paths such as those contained in an
SVG or DXF file.
"""

try:
    from .path import Path2D, Path3D
except BaseException as E:
    from .. import exceptions

    Path2D = exceptions.ExceptionWrapper(E)
    Path3D = exceptions.ExceptionWrapper(E)

# explicitly add objects to all as per pep8
__all__ = ["Path2D", "Path3D"]
