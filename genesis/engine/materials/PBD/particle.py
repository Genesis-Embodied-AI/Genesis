import gstaichi as ti

import genesis as gs

from .base import Base


@ti.data_oriented
class Particle(Base):
    """
    The liquid material class for PBD.

    Note
    ----
    This creates particle-based entities that has no inter-particle interaction at all; i.e. it is only affected by external forces.
    This is useful for creating particle-based animations.
    This material will be handled by the PBD solver, but there's actually nothing to solve really. It's just hosted under the PBD system.

    Parameters
    ----------
    rho: float, optional
        The rest density. Default is 1000.0.
    sampler: str, optional
        Particle sampler ('pbs', 'regular', 'random'). Note that 'pbs' is only supported on Linux for now. Defaults to
        'pbs' on supported platforms, 'random' otherwise.
    """

    def __init__(
        self,
        rho=1000.0,
        sampler=None,
    ):
        if sampler is None:
            sampler = "pbs" if gs.platform == "Linux" else "random"

        super().__init__()

        self._rho = rho
        self._sampler = sampler

    @property
    def rho(self):
        """The rest density."""
        return self._rho

    @property
    def sampler(self):
        """Particle sampler ('pbs', 'regular', 'random')."""
        return self._sampler
