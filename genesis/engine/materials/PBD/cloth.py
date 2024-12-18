import taichi as ti

from .base import Base


@ti.data_oriented
class Cloth(Base):
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
        """
        Cloth Material of PBD solver.

        Args:
            rho (float): The density of the cloth. Defaults to 4.0. Note that this is kg/m^2, not kg/m^3, as cloth is a 2D material, so the entity mass will be calculated as rho * surface_area.
            static_friction (float): Static friction coefficient. Defaults to 0.15.
            kinetic_friction (float): Kinetic friction coefficient. Defaults to 0.15.
            stretch_compliance (float): The stretch compliance of the cloth. Defaults to 1e-7.
            bending_compliance (float): The bending compliance of the cloth. Defaults to 1e-5.
            stretch_relaxation (float): The stretch relaxation of the cloth. Defaults to 0.3. Smaller value weakens the stretch constraint.
            bending_relaxation (float): The bending relaxation of the cloth. Defaults to 0.1. Smaller value weakens the bending constraint.
            air_resistance (float): The air resistance of the cloth. Defaults to 1e-3.
        """
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
        return self._rho

    @property
    def static_friction(self):
        return self._static_friction

    @property
    def kinetic_friction(self):
        return self._kinetic_friction

    @property
    def stretch_compliance(self):
        return self._stretch_compliance

    @property
    def bending_compliance(self):
        return self._bending_compliance

    @property
    def stretch_relaxation(self):
        return self._stretch_relaxation

    @property
    def bending_relaxation(self):
        return self._bending_relaxation

    @property
    def air_resistance(self):
        return self._air_resistance
