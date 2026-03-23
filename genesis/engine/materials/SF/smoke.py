from typing import Literal

from .base import Base

SamplerType = Literal["pbs", "random", "regular"]


class Smoke(Base):
    """
    Smoke material for the stable fluids solver.
    """

    sampler: SamplerType = "regular"
