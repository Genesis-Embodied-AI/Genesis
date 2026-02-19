from .base import Base


class Cloth(Base):
    """
    Thin shell material for cloth simulation with IPC coupler.

    Parameters
    ----------
    E : float, optional
        Young's modulus in Pa. Default is 1e5.
    nu : float, optional
        Poisson's ratio. Default is 0.45.
    rho : float, optional
        Density in kg/m³. Default is 200.0.
    thickness : float, optional
        Shell thickness in meters. Default is 0.001.
    bending_stiffness : float, optional
        Bending resistance coefficient. Default is 10.0.
    friction_mu : float, optional
        Friction coefficient for IPC contact. Default is 0.5.
    """

    def __init__(self, E=1e5, nu=0.45, rho=200.0, thickness=0.001, bending_stiffness=10.0, friction_mu=0.5):
        super().__init__(E=E, nu=nu, rho=rho, friction_mu=friction_mu)
        self._thickness = thickness
        self._bending_stiffness = bending_stiffness

    @property
    def thickness(self):
        """Shell thickness in meters."""
        return self._thickness

    @property
    def bending_stiffness(self):
        """Bending resistance coefficient."""
        return self._bending_stiffness
