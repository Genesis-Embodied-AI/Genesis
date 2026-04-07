from typing import Any

import quadrants as qd
from pydantic import Field, PrivateAttr

from genesis.typing import PositiveInt, ValidFloat

from .elastic import Elastic


@qd.data_oriented
class Muscle(Elastic):
    """
    The muscle material class for FEM.

    Parameters
    ----------
    E : float, optional
        Young's modulus, which controls stiffness. Default is 1e6.
    nu : float, optional
        Poisson ratio. Default is 0.2.
    rho : float, optional
        Material density (kg/m³). Default is 1000.
    model : str, optional
        Constitutive model ('linear' or 'stable_neohookean'). Default is 'linear'.
    n_groups : int, optional
        Number of muscle groups. Default is 1.
    friction_mu : float, optional
        Contact friction coefficient. Default is 0.1.
    contact_resistance : float | None, optional
        IPC contact resistance/stiffness override. Default is None.
    """

    n_groups: PositiveInt = 1

    # Auto-generated — equals E.
    stiffness: ValidFloat = Field(default=0.0, exclude=True)

    _update_stress_without_actuation: Any = PrivateAttr(default=None)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)

        self.stiffness = self.E
        self._update_stress_without_actuation = self.update_stress
        self.update_stress = self._update_stress_with_actuation

    @qd.func
    def _update_stress_with_actuation(self, mu, lam, J, F, actu, m_dir):
        stress = self._update_stress_without_actuation(mu, lam, J, F, actu, m_dir)

        l = (F @ m_dir).norm(1e-12)
        mmT = m_dir.outer_product(m_dir)
        stress += self.stiffness * (actu / l) * F @ mmT

        return stress
