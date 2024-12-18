import numpy as np
import taichi as ti

import genesis as gs

from .base import Base


@ti.data_oriented
class Sand(Base):
    def __init__(
        self,
        E=1e6,
        nu=0.2,
        rho=1000.0,
        lam=None,
        mu=None,
        sampler="random",
        friction_angle=45,
    ):
        super().__init__(E, nu, rho, lam, mu, sampler)

        self._default_Jp = 0.0

        self.friction_angle = np.deg2rad(friction_angle)
        sin_phi = np.sin(self.friction_angle)
        self.alpha = np.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi)

    @ti.func
    def sand_projection(self, S, Jp):
        S_out = ti.Matrix.zero(gs.ti_float, 3, 3)
        epsilon = ti.Vector.zero(gs.ti_float, 3)
        for i in ti.static(range(3)):
            epsilon[i] = ti.log(max(abs(S[i, i]), 1e-4))
            S_out[i, i] = 1
        tr = epsilon.sum() + Jp
        epsilon_hat = epsilon - tr / 3
        epsilon_hat_norm = epsilon_hat.norm(gs.EPS)

        Jp_new = gs.ti_float(0.0)
        if tr >= 0.0:
            Jp_new = tr
        else:
            Jp_new = 0.0
            delta_gamma = epsilon_hat_norm + (3 * self._lam + 2 * self._mu) / (2 * self._mu) * tr * self.alpha
            for i in ti.static(range(3)):
                S_out[i, i] = ti.exp(epsilon[i] - max(0, delta_gamma) / epsilon_hat_norm * epsilon_hat[i])

        return S_out, Jp_new

    @ti.func
    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp):
        S_new, Jp_new = self.sand_projection(S, Jp)
        F_new = U @ S_new @ V.transpose()
        return F_new, S_new, Jp_new

    @ti.func
    def update_stress(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        log_S_sum = gs.ti_float(0.0)
        center = ti.Matrix.zero(gs.ti_float, 3, 3)
        for i in ti.static(range(3)):
            log_S_sum += ti.log(S[i, i])
            center[i, i] = 2.0 * self._mu * ti.log(S[i, i]) * (1 / S[i, i])
        for i in ti.static(range(3)):
            center[i, i] += self._lam * log_S_sum * (1 / S[i, i])
        stress = U @ center @ V.transpose() @ F_new.transpose()
        return stress
