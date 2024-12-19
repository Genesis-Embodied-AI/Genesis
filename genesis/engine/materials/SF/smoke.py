import taichi as ti

from .base import Base


@ti.data_oriented
class Smoke(Base):
    def __init__(self):
        super().__init__()

    @property
    def sampler(self):
        return "regular"
