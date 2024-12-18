import taichi as ti

from .base import Base


@ti.data_oriented
class Particle(Base):
    """
    This creates particle-based entities that has no inter-particle interaction at all; i.e. it is only affected by external forces.
    This is useful for creating particle-based animations.
    This material will be handled by the PBD solver, but there's actually nothing to solve really. It's just hosted under the PBD system.
    """

    def __init__(
        self,
        rho=1000.0,
        sampler="pbs",
    ):
        super().__init__()

        self._rho = rho
        self._sampler = sampler

    @property
    def rho(self):
        return self._rho

    @property
    def sampler(self):
        return self._sampler
