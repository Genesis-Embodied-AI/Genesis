import taichi as ti

from ..base import Material


@ti.data_oriented
class Base(Material):
    def __init__(
        self,
        sampler="pbs",  # particle sampler
    ):
        super().__init__()

        self._sampler = sampler

    @classmethod
    def _repr_type(cls):
        return f"<gs.materials.SPH.{cls.__name__}>"

    @property
    def sampler(self):
        return self._sampler
