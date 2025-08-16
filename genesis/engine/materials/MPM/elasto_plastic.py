import gstaichi as ti

import genesis as gs

from .base import Base


@ti.data_oriented
class ElastoPlastic(Base):
    """
    The elasto-plastic material class for MPM.

    Note
    ----
    Default yield ratio comes from the SNOW material in taichi's MPM implementation:
    https://github.com/taichi-dev/taichi_elements/blob/d19678869a28b09a32ef415b162e35dc929b792d/engine/mpm_solver.py#L434

    Parameters
    ----------
    E: float, optional
        Young's modulus. Default is 1e6.
    nu: float, optional
        Poisson ratio. Default is 0.2.
    rho: float, optional
        Density (kg/m^3). Default is 1000.
    lam: float, optional
        The first Lame's parameter. Default is None, computed by E and nu.
    mu: float, optional
        The second Lame's parameter. Default is None, computed by E and nu.
    sampler: str, optional
        Particle sampler ('pbs', 'regular', 'random'). Default is 'pbs'.
    yield_lower: float, optional
        Lower bound for the yield clamp (ignored if using von Mises). Default is 2.5e-2.
    yield_higher: float, optional
        Upper bound for the yield clamp (ignored if using von Mises). Default is 4.5e-2.
    use_von_mises: bool, optional
        Whether to use von Mises yield criterion. Default is True.
    von_mises_yield_stress: float, optional
        Yield stress for von Mises criterion. Default is 10000.
    """

    def __init__(
        self,
        E=1e6,  # Young's modulus
        nu=0.2,  # Poisson's ratio
        rho=1000.0,  # density (kg/m^3)
        lam=None,
        mu=None,
        sampler="pbs",
        yield_lower=2.5e-2,
        yield_higher=4.5e-3,
        use_von_mises=True,  # von Mises yield criterion
        von_mises_yield_stress=10000.0,
    ):
        super().__init__(E, nu, rho, lam, mu, sampler)

        self._yield_lower = yield_lower
        self._yield_higher = yield_higher
        self._use_von_mises = use_von_mises
        self._von_mises_yield_stress = von_mises_yield_stress

    @ti.func
    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp):
        F_new = ti.Matrix.zero(gs.ti_float, 3, 3)
        S_new = ti.Matrix.zero(gs.ti_float, 3, 3)
        if ti.static(self.use_von_mises):
            S_new = ti.max(S, 0.05)  # to prevent NaN
            epsilon = ti.Vector([ti.log(S_new[0, 0]), ti.log(S_new[1, 1]), ti.log(S_new[2, 2])])
            epsilon_hat = epsilon - (epsilon.sum() / 3)
            epsilon_hat_norm = epsilon_hat.norm(gs.EPS)
            delta_gamma = epsilon_hat_norm - self._von_mises_yield_stress / (2 * self._mu)

            if delta_gamma > 0:  # Yields
                epsilon -= (delta_gamma / epsilon_hat_norm) * epsilon_hat
                S_new = ti.Matrix.zero(gs.ti_float, 3, 3)
                for d in ti.static(range(3)):
                    S_new[d, d] = ti.exp(epsilon[d])
                F_new = U @ S_new @ V.transpose()
            else:
                F_new = F_tmp

        else:
            S_new = ti.Matrix.zero(gs.ti_float, 3, 3)
            for d in ti.static(range(3)):
                S_new[d, d] = min(max(S[d, d], 1 - self._yield_lower), 1 + self._yield_higher)
            F_new = U @ S_new @ V.transpose()

        Jp_new = Jp
        return F_new, S_new, Jp_new

    @property
    def yield_lower(self):
        """Lower bound for the yield clamp (ignored if using von Mises)."""
        return self._yield_lower

    @property
    def yield_higher(self):
        """Upper bound for the yield clamp (ignored if using von Mises)."""
        return self._yield_higher

    @property
    def use_von_mises(self):
        """Whether to use von Mises yield criterion."""
        return self._use_von_mises

    @property
    def von_mises_yield_stress(self):
        """Yield stress for von Mises criterion."""
        return self._von_mises_yield_stress
