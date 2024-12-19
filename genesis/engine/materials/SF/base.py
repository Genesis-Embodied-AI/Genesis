import numpy as np
import taichi as ti

from ..base import Material


@ti.data_oriented
class Base(Material):
    def __init__(self):
        super().__init__()
