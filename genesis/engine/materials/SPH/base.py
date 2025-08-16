import gstaichi as ti

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
        Particle sampler ('pbs', 'regular', 'random'). Default is 'pbs'.
    """

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
        """Particle sampler ('pbs', 'regular', 'random')."""
        return self._sampler
