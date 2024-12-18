import taichi as ti

from .rigid import Rigid


@ti.data_oriented
class Avatar(Rigid):
    pass
