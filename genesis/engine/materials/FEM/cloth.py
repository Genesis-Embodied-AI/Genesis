from .base import Base


class Cloth(Base):
    """
    Thin shell material for cloth simulation with IPC coupler.

    Parameters
    ----------
    E : float, optional
        Young's modulus in Pa. Default is 1e4.
    nu : float, optional
        Poisson's ratio. Default is 0.49.
    rho : float, optional
        Density in kg/m³. Default is 200.0.
    thickness : float, optional
        Shell thickness in meters. Default is 0.001.
    bending_stiffness : float, optional
        Bending resistance coefficient. If None, no bending resistance.
        Default is None.
    friction_mu : float, optional
        Friction coefficient for IPC contact. Default is 0.5.
    model : str, optional
        FEM material model. Default is "stable_neohookean".
    """

    def __init__(
        self,
        E=1e4,                    # Young's modulus (Pa)
        nu=0.49,                  # Poisson's ratio
        rho=200.0,                # Density (kg/m³)
        thickness=0.001,          # Shell thickness (m)
        bending_stiffness=None,   # Optional bending stiffness
        friction_mu=0.5,          # Friction coefficient for IPC contact
        model="stable_neohookean",
    ):
        super().__init__(E=E, nu=nu, rho=rho, friction_mu=friction_mu)
        self._thickness = thickness
        self._bending_stiffness = bending_stiffness
        self._model = model

    @property
    def thickness(self):
        """Shell thickness in meters."""
        return self._thickness

    @property
    def bending_stiffness(self):
        """Bending resistance coefficient."""
        return self._bending_stiffness

    @property
    def model(self):
        """FEM material model name."""
        return self._model

    def __repr__(self):
        return f"<gs.materials.FEM.Cloth(E={self.E}, nu={self.nu}, rho={self.rho}, thickness={self.thickness})>"
