import taichi as ti

import genesis as gs

from .base import Base


@ti.data_oriented
class Elastic(Base):
    """
    Reference for the default values of `E` and `nu`: https://github.com/taichi-dev/taichi_elements/blob/d19678869a28b09a32ef415b162e35dc929b792d/engine/mpm_solver.py#L201
    Elastic objects is softened by multiplying the default E by 0.3.
    """

    def __init__(
        self,
        E=3e5,
        nu=0.2,
        rho=1000.0,
        lam=None,
        mu=None,
        sampler="pbs",
        model="corotation",
    ):
        super().__init__(E, nu, rho, lam, mu, sampler)

        if model == "corotation":
            self.update_stress = self.update_stress_corotation
        elif model == "neohooken":
            self.update_stress = self.update_stress_neohooken
        else:
            gs.raise_exception(f"Unrecognized constitutive model: {model}")

        self._model = model

    @ti.func
    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp):
        F_new = F_tmp
        S_new = S
        Jp_new = Jp
        return F_new, S_new, Jp_new

    @ti.func
    def update_stress_corotation(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        stress = 2 * self._mu * (F_new - U @ V.transpose()) @ F_new.transpose() + ti.Matrix.identity(
            gs.ti_float, 3
        ) * self._lam * J * (J - 1)

        return stress

    @ti.func
    def update_stress_neohooken(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        stress = self._mu * (F_tmp @ F_tmp.transpose()) + ti.Matrix.identity(gs.ti_float, 3) * (
            self._lam * ti.log(J) - self._mu
        )

        return stress

    @property
    def model(self):
        return self._model
