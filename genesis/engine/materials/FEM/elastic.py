import taichi as ti

import genesis as gs

from .base import Base


@ti.data_oriented
class Elastic(Base):
    def __init__(
        self,
        E=1e6,  # Young's modulus
        nu=0.2,  # Poisson's ratio
        rho=1000.0,  # density (kg/m^3)
        model="linear",
    ):
        super().__init__(E, nu, rho)

        if model == "linear":
            self.update_stress = self.update_stress_linear
        elif model == "stable_neohooken":
            self.update_stress = self.update_stress_stable_neohooken
        else:
            gs.raise_exception(f"Unrecognized constitutive model: {model}")

        self._model = model

    @ti.func
    def update_stress_linear(self, mu, lam, J, F, actu, m_dir):
        I = ti.Matrix.identity(dt=gs.ti_float, n=3)
        stress = mu * (F + F.transpose() - 2 * I) + lam * (F - I).trace() * I

        return stress

    @ti.func
    def update_stress_stable_neohooken(self, mu, lam, J, F, actu, m_dir):
        IC = (F.transpose() @ F).trace()
        dJdF0 = F[:, 1].cross(F[:, 2])
        dJdF1 = F[:, 2].cross(F[:, 0])
        dJdF2 = F[:, 0].cross(F[:, 1])
        dJdF = ti.Matrix.cols([dJdF0, dJdF1, dJdF2])
        alpha = 1 + 0.75 * mu / lam
        stress = mu * (1 - 1 / (IC + 1)) * F + lam * (J - alpha) * dJdF

        return stress

    @property
    def model(self):
        return self._model
