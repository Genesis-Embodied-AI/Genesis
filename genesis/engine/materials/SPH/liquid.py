from genesis.typing import NonNegativeFloat, PositiveFloat

from .base import Base, SamplerType


class Liquid(Base):
    """
    The liquid material class for SPH.

    Parameters
    ----------
    rho : float, optional
        The rest density (kg/m³). Default is 1000.0.
    stiffness : float, optional
        State stiffness (N/m²). Controls how pressure increases with compression. Default is 50000.0.
    exponent : float, optional
        State exponent. Controls how nonlinearly pressure scales with density. Default is 7.0.
    mu : float, optional
        The viscosity of the liquid. Default is 0.005.
    gamma : float, optional
        The surface tension of the liquid. Default is 0.01.
    sampler : str, optional
        Particle sampler. Defaults to 'regular' for numerical stability with SPH.
    """

    rho: PositiveFloat = 1000.0
    stiffness: PositiveFloat = 50000.0
    exponent: PositiveFloat = 7.0
    mu: NonNegativeFloat = 0.005
    gamma: NonNegativeFloat = 0.01
    sampler: SamplerType = "regular"
