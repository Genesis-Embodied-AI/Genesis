import taichi as ti

from .base import Base


@ti.data_oriented
class Liquid(Base):
    def __init__(
        self,
        rho=1000.0,
        sampler="pbs",
        density_relaxation=0.2,
        viscosity_relaxation=0.01,
    ):
        super().__init__()

        self._rho = rho
        self._sampler = sampler
        self._density_relaxation = density_relaxation
        self._viscosity_relaxation = viscosity_relaxation

    @property
    def rho(self):
        return self._rho

    @property
    def sampler(self):
        return self._sampler

    @property
    def density_relaxation(self):
        return self._density_relaxation

    @property
    def viscosity_relaxation(self):
        return self._viscosity_relaxation
