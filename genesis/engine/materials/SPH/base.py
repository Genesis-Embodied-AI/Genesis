import gstaichi as ti

import genesis as gs

from ..base import Material


@ti.data_oriented
class Base(Material):
    """
    The base class of SPH materials.

    Note
    ----
    This class should *not* be instantiated directly.

    Parameters
    ----------
    sampler: str, optional
        Particle sampler ('pbs', 'regular', 'random'). Note that 'pbs' is only supported on Linux for now. Defaults to
        'pbs' on supported platforms, 'random' otherwise.
    """

    def __init__(
        self,
        sampler=None,
    ):
        if sampler is None:
            sampler = "pbs" if gs.platform == "Linux" else "random"

        super().__init__()

        self._sampler = sampler

    @classmethod
    def _repr_type(cls):
        return f"<gs.materials.SPH.{cls.__name__}>"

    @property
    def sampler(self):
        """Particle sampler ('pbs', 'regular', 'random')."""
        return self._sampler
