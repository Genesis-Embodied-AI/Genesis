import gstaichi as ti
import genesis as gs
from .elastic import Elastic


@ti.data_oriented
class Muscle(Elastic):
    """
    The muscle material class for FEM.

    Parameters
    ----------
    E: float, optional
        Young's modulus, which controls stiffness. Default is 1e6.
    nu: float, optional
        Poisson ratio, describing the material's volume change under stress. Default is 0.2.
    rho: float, optional
        Material density (kg/m^3). Default is 1000.
    model: str, optional
        Constitutive model to use for stress computation. Options are:
        - 'linear': Linear elasticity model
        - 'stable_neohookean': A numerically stable Neo-Hookean model
        Default is 'linear'.
    n_groups: int, optional
        Number of muscle groups. Default is 1.
    """

    def __init__(
        self,
        E=1e6,  # Young's modulus
        nu=0.2,  # Poisson's ratio
        rho=1000.0,  # density (kg/m^3)
        model="linear",
        n_groups=1,  # number of muscle group
    ):
        super().__init__(E, nu, rho, model)

        # inherit from Elastic
        self._update_stress_without_actuation = self.update_stress
        self.update_stress = self._update_stress_with_actuation

        self._stiffness = E  # NOTE: use Young's modulus as muscle stiffness
        self._n_groups = n_groups

    @ti.func
    def _update_stress_with_actuation(self, mu, lam, J, F, actu, m_dir):
        stress = self._update_stress_without_actuation(mu, lam, J, F, actu, m_dir)

        l = (F @ m_dir).norm(1e-12)
        mmT = m_dir.outer_product(m_dir)
        stress += self._stiffness * (actu / l) * F @ mmT

        return stress

    @property
    def stiffness(self):
        """Muscle stiffness. Equivalent to Young's modulus."""
        return self._stiffness

    @property
    def n_groups(self):
        """Number of muscle groups."""
        return self._n_groups
