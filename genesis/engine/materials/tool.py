import taichi as ti

from .base import Material


@ti.data_oriented
class Tool(Material):
    def __init__(
        self,
        friction=0.0,
        coup_softness=0.01,
        collision=True,
        sdf_res=128,
    ):
        super().__init__()

        self.friction = friction
        self.coup_softness = coup_softness
        self.collision = collision
        self.sdf_res = sdf_res
