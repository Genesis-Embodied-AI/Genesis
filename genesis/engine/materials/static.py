import taichi as ti
import genesis as gs
from .base import Material


@ti.data_oriented
class Static(Material):
    """
    Static material class for static entities.
    This class is intentionally empty.
    """
