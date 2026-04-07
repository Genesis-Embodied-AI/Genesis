from typing import Any, Literal

import quadrants as qd
from pydantic import Field, PrivateAttr

from genesis.typing import PositiveFloat, PositiveInt, ValidFloat

from .elastic import Elastic


@qd.data_oriented
class Muscle(Elastic):
    """
    The muscle material class for MPM.

    Parameters
    ----------
    E : float, optional
        Young's modulus. Default is 1e6.
    nu : float, optional
        Poisson ratio. Default is 0.2.
    rho : float, optional
        Density (kg/m³). Default is 1000.
    model : str, optional
        Stress model ('corotation', 'neohooken'). Default is 'neohooken'.
    n_groups : int, optional
        Number of muscle groups. Default is 1.
    """

    E: PositiveFloat = 1e6
    model: Literal["corotation", "neohooken"] = "neohooken"
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
    def _update_stress_with_actuation(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        stress = self._update_stress_without_actuation(U, S, V, F_tmp, F_new, J, Jp, actu, m_dir)

        AAt = m_dir.outer_product(m_dir)
        stress += self.stiffness * actu * F_tmp @ AAt @ F_tmp.transpose()

        return stress
