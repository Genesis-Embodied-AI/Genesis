import gstaichi as ti

import genesis as gs

from .base import Base


@ti.data_oriented
class Liquid(Base):
    """
    The liquid material class for SPH.

    Parameters
    ----------
    rho: float, optional
        The density (kg/m^3) the material tends to maintain in equilibrium (i.e., the “rest” or undeformed state). Default is 1000.
    stiffness: float, optional
        State stiffness (N/m^2). A material constant controlling how pressure increases with compression. Default is 50000.0.
    exponent: float, optional
        State exponent. Controls how nonlinearly pressure scales with density. Larger values mean stiffer response to compression. Default is 7.0.
    mu: float, optional
        The viscosity of the liquid. A measure of the internal friction of the fluid or material. Default is 0.005
    gamma: float, optional
        The surface tension of the liquid. Controls how strongly the material “clumps” together at boundaries. Default is 0.01
    sampler: str, optional
        Particle sampler ('pbs', 'regular', 'random'). Note that 'pbs' is only supported on Linux for now. Defaults to
        'pbs' on supported platforms, 'random' otherwise.
    """

    def __init__(
        self,
        rho=1000.0,
        stiffness=50000.0,
        exponent=7.0,
        mu=0.005,
        gamma=0.01,
        sampler=None,
    ):
        if sampler is None:
            sampler = "pbs" if gs.platform == "Linux" else "random"

        super().__init__(sampler)

        self._rho = rho
        self._stiffness = stiffness
        self._exponent = exponent
        self._mu = mu  # viscosity
        self._gamma = gamma  # surface tension

    @property
    def rho(self):
        """The density (kg/m^3) the material tends to maintain in equilibrium (i.e., the “rest” or undeformed state)."""
        return self._rho

    @property
    def stiffness(self):
        """State stiffness (N/m^2). A material constant controlling how pressure increases with compression."""
        return self._stiffness

    @property
    def exponent(self):
        """State exponent. Controls how nonlinearly pressure scales with density. Larger values mean stiffer response to compression."""
        return self._exponent

    @property
    def mu(self):
        """The viscosity of the liquid. A measure of the internal friction of the fluid or material."""
        return self._mu

    @property
    def gamma(self):
        """The surface tension of the liquid. Controls how strongly the material “clumps” together at boundaries."""
        return self._gamma
