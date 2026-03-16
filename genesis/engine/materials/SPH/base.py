import sys
import platform
from typing import TYPE_CHECKING

from ..base import Material

if TYPE_CHECKING:
    from genesis.engine.entities.sph_entity import SPHEntity


class Base(Material["SPHEntity"]):
    """
    The base class of SPH materials.

    Note
    ----
    This class should *not* be instantiated directly.

    Parameters
    ----------
    sampler: str, optional
        Particle sampler ('pbs', 'regular', 'random'). Note that 'pbs' is only supported on Linux x86 for now. Defaults
        to 'pbs' on supported platforms, 'random' otherwise.
    """

    def __init__(
        self,
        sampler=None,
    ):
        if sampler is None:
            sampler = "pbs" if (sys.platform == "linux" and platform.machine() == "x86_64") else "random"

        super().__init__()

        self._sampler = sampler

    @property
    def sampler(self):
        """Particle sampler ('pbs', 'regular', 'random')."""
        return self._sampler
