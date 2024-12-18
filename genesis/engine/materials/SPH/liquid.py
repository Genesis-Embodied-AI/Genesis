import taichi as ti

from .base import Base


@ti.data_oriented
class Liquid(Base):
    def __init__(
        self,
        rho=1000.0,
        stiffness=50000.0,
        exponent=7.0,
        mu=0.005,
        gamma=0.01,
        sampler="pbs",
    ):
        super().__init__(sampler)

        self._rho = rho
        self._stiffness = stiffness
        self._exponent = exponent
        self._mu = mu  # viscosity
        self._gamma = gamma  # surface tension

    @property
    def rho(self):
        return self._rho

    @property
    def stiffness(self):
        return self._stiffness

    @property
    def exponent(self):
        return self._exponent

    @property
    def mu(self):
        return self._mu

    @property
    def gamma(self):
        return self._gamma
