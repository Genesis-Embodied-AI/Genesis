import taichi as ti

from .base import Base


@ti.data_oriented
class Elastic(Base):
    def __init__(
        self,
        rho=1000.0,
        static_friction=0.15,
        kinetic_friction=0.15,
        stretch_compliance=0.0,
        bending_compliance=0.0,
        volume_compliance=0.0,
        stretch_relaxation=0.1,
        bending_relaxation=0.1,
        volume_relaxation=0.1,
        thickness=1e-4,
    ):
        """
        3D Elastic Volumentric Material of PBD solver.

        Args:
            rho (float): The density of the elastic material (kg/m^3). Defaults to 1000.0.
            static_friction (float): Static friction coefficient. Defaults to 0.15.
            kinetic_friction (float): Kinetic friction coefficient. Defaults to 0.15.
            stretch_compliance (float): The stretch compliance of the cloth. Defaults to 0.0.
            bending_compliance (float): The bending compliance of the cloth. Defaults to 0.0.
            volume_compliance (float): The volume compliance of the cloth. Defaults to 0.0.
            stretch_relaxation (float): The stretch relaxation of the cloth. Defaults to 0.1. Smaller value weakens the stretch constraint.
            bending_relaxation (float): The bending relaxation of the cloth. Defaults to 0.1. Smaller value weakens the bending constraint.
            volume_relaxation (float): The volume relaxation of the cloth. Defaults to 0.1. Smaller value weakens the volume constraint.
            thickness (float): The thickness of the elastic material. Defaults to 1e-4. TODO: what is this?
        """
        super().__init__()
        self._rho = rho
        self._static_friction = static_friction
        self._kinetic_friction = kinetic_friction
        self._stretch_compliance = stretch_compliance
        self._bending_compliance = bending_compliance
        self._volume_compliance = volume_compliance
        self._stretch_relaxation = stretch_relaxation
        self._bending_relaxation = bending_relaxation
        self._volume_relaxation = volume_relaxation
        self._thickness = thickness

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
    def volume_compliance(self):
        return self._volume_compliance

    @property
    def stretch_relaxation(self):
        return self._stretch_relaxation

    @property
    def bending_relaxation(self):
        return self._bending_relaxation

    @property
    def volume_relaxation(self):
        return self._volume_relaxation

    @property
    def thickness(self):
        return self._thickness
