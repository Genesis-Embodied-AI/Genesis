import numpy as np
import gstaichi as ti

from ..base import Material


@ti.data_oriented
class Base(Material):
    def __init__(self):
        super().__init__()
