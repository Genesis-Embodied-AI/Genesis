import numpy as np
import gstaichi as ti

from ..base import Material


@ti.data_oriented
class Base(Material):
    """
    The base class of PBD materials.

    Note
    ----
    This class should *not* be instantiated directly.
    """

    def __init__(self):
        super().__init__()
