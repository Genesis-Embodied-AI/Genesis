"""
https://github.com/mikedh/trimesh
------------------------------------

Trimesh is a pure Python (2.7- 3.3+) library for loading and using triangular
meshes with an emphasis on watertight meshes. The goal of the library is to
provide a fully featured Trimesh object which allows for easy manipulation
and analysis, in the style of the Polygon object in the Shapely library.
"""

# current version
from .version import __version__

# geometry objects
from .base import Trimesh
from .points import PointCloud
from .scene.scene import Scene

# utility functions
from .util import unitize
from .transformations import transform_points

# general numeric tolerances
from .constants import tol

# loader functions
from .exchange.load import load, load_mesh, load_path, load_remote, available_formats

# avoid a circular import in trimesh.base
from . import voxel
from . import bounds
from . import nsphere
from . import collision
from . import smoothing
from . import primitives

try:
    # handle vector paths
    from . import path
except BaseException as E:
    # raise a useful error if path hasn't loaded
    from .exceptions import ExceptionWrapper

    path = ExceptionWrapper(E)

# explicitly list imports in __all__
# as otherwise flake8 gets mad
__all__ = [
    __version__,
    "Trimesh",
    "PointCloud",
    "Scene",
    "voxel",
    "unitize",
    "bounds",
    "nsphere",
    "collision",
    "smoothing",
    "tol",
    "path",
    "load",
    "load_mesh",
    "load_path",
    "load_remote",
    "primitives",
    "transform_points",
    "available_formats",
]
