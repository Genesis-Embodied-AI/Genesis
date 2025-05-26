import taichi as ti

from .base import Base


@ti.data_oriented
class Cloth(Base):
    """
    The cloth material class for PBD.

    Parameters
    ----------
    rho: float, optional
        The density of the cloth. Default is 4.0.
        Note that this is kg/m^2, not kg/m^3, as cloth is a 2D material, so the entity mass will be calculated as rho * surface_area.
    static_friction: float, optional
        Static friction coefficient. Represents the resistance to the start of sliding motion between two contacting particles.
        In collision resolution, it determines how much tangential force can be applied before sliding begins. Default is 0.15.
    kinetic_friction: float, optional
        Kinetic (Dynamic) Friction Coefficient. Represents the resistance during sliding motion between two contacting particles.
        Applied when particles are already sliding; limits the tangential force to simulate energy loss due to friction. Default is 0.0.
    stretch_compliance: float, optional
        The stretch compliance (m/N). Controls the softness of the stretch constraint between particles.
        Low values correspond to very stiff; enforces near-constant distance. High values correspond to softer response; more stretch allowed. Default is 0.0.
    bending_compliance: float, optional
        The bending compliance (rad/N). Controls how easily the material bends (e.g., at the fold of a cloth or edge of a soft body).
        Appears in inner edge constraints, determining how strongly the shape resists changes in angle. Default is 0.0.
    stretch_relaxation: float, optional
        The stretch relaxation of the cloth. Smaller value weakens the stretch constraint. Default is 0.3.
    bending_relaxation: float, optional
        The bending relaxation of the cloth. Smaller value weakens the bending constraint. Default is 0.1.
    air_resistance: float, optional
        The air resistance of the cloth. Damping force due to air drag. Default is 1e-3.
    """

    def __init__(
        self,
        rho=4.0,
        static_friction=0.15,
        kinetic_friction=0.15,
        stretch_compliance=1e-7,
        bending_compliance=1e-5,
        stretch_relaxation=0.3,
        bending_relaxation=0.1,
        air_resistance=1e-3,
    ):
        super().__init__()

        self._rho = rho
        self._static_friction = static_friction
        self._kinetic_friction = kinetic_friction
        self._stretch_compliance = stretch_compliance
        self._bending_compliance = bending_compliance
        self._stretch_relaxation = stretch_relaxation
        self._bending_relaxation = bending_relaxation
        self._air_resistance = air_resistance

    @property
    def rho(self):
        """The density of the cloth."""
        return self._rho

    @property
    def static_friction(self):
        """Static friction coefficient."""
        return self._static_friction

    @property
    def kinetic_friction(self):
        """Kinetic friction coefficient."""
        return self._kinetic_friction

    @property
    def stretch_compliance(self):
        """The stretch compliance of the cloth."""
        return self._stretch_compliance

    @property
    def bending_compliance(self):
        """The bending compliance of the cloth."""
        return self._bending_compliance

    @property
    def stretch_relaxation(self):
        """The stretch relaxation of the cloth."""
        return self._stretch_relaxation

    @property
    def bending_relaxation(self):
        """The bending relaxation of the cloth."""
        return self._bending_relaxation

    @property
    def air_resistance(self):
        """The air resistance of the cloth."""
        return self._air_resistance
