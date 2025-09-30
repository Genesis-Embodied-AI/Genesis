import gstaichi as ti

import genesis as gs

from .base import Base


@ti.data_oriented
class Liquid(Base):
    """
    The liquid material class for PBD.

    Parameters
    ----------
    rho: float, optional
        The rest density of the fluid in kg/m³. Default is 1000.0.
    sampler: str, optional
        Particle sampler ('pbs', 'regular', 'random'). Note that 'pbs' is only supported on Linux for now. Defaults to
        'pbs' on supported platforms, 'random' otherwise.
    density_relaxation: float, optional
        Relaxation factor for solving the density constraint. Controls the strength of positional correction to enforce incompressibility.
        Higher values lead to faster convergence but can cause instability. Default is 0.2.
    viscosity_relaxation: float, optional
        Relaxation factor used in the viscosity solver. Influences the smoothing of relative velocities between neighboring particles.
        Higher values lead to more viscous (syrupy) fluid behavior. Default is 0.01.
    """

    def __init__(
        self,
        rho=1000.0,
        sampler=None,
        density_relaxation=0.2,
        viscosity_relaxation=0.01,
    ):
        if sampler is None:
            sampler = "pbs" if gs.platform == "Linux" else "random"

        super().__init__()

        self._rho = rho
        self._sampler = sampler
        self._density_relaxation = density_relaxation
        self._viscosity_relaxation = viscosity_relaxation

    @property
    def rho(self):
        """The rest density of the fluid (kg/m³)."""
        return self._rho

    @property
    def sampler(self):
        """Particle sampler ('pbs', 'regular', 'random')."""
        return self._sampler

    @property
    def density_relaxation(self):
        """Relaxation coefficient for density constraint solving."""
        return self._density_relaxation

    @property
    def viscosity_relaxation(self):
        """Relaxation coefficient for viscosity-based smoothing."""
        return self._viscosity_relaxation
