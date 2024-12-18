# flake8: NOQA
"""
visual
-------------

Handle visual properties for meshes, including color and texture
"""

from .color import (
    ColorVisuals,
    random_color,
    to_rgba,
    DEFAULT_COLOR,
    interpolate,
    uv_to_color,
    uv_to_interpolated_color,
    linear_color_map,
)

from .texture import TextureVisuals
from .objects import create_visual, concatenate

from . import color
from . import texture
from . import objects
from . import material
from .. import resolvers


# explicitly list imports in __all__
# as otherwise flake8 gets mad
__all__ = [
    "color",
    "texture",
    "resolvers",
    "TextureVisuals",
    "ColorVisuals",
    "random_color",
    "to_rgba",
    "create_visual",
    "DEFAULT_COLOR",
    "interpolate",
    "linear_color_map",
    "uv_to_color",
    "uv_to_interpolated_color",
]
