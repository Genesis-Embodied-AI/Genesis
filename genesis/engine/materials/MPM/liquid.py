import taichi as ti

import genesis as gs

from .base import Base


@ti.data_oriented
class Liquid(Base):
    def __init__(
        self,
        E=1e6,
        nu=0.2,
        rho=1000.0,
        lam=None,
        mu=None,
        viscous=False,
        sampler="pbs",
    ):
        super().__init__(E, nu, rho, lam, mu, sampler)

        if not viscous:
            self._mu = 0.0

    @ti.func
    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp):
        F_new = ti.Matrix.identity(gs.ti_float, 3) * ti.pow(J, 1.0 / 3.0)
        S_new = S
        Jp_new = Jp
        return F_new, S_new, Jp_new

    @ti.func
    def update_stress(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        # NOTE: class member function inheritance will still introduce redundant computation graph in taichi
        stress = 2 * self._mu * (F_tmp - U @ V.transpose()) @ F_tmp.transpose() + ti.Matrix.identity(
            gs.ti_float, 3
        ) * self._lam * J * (J - 1)

        return stress
