import taichi as ti

import genesis as gs

from .elasto_plastic import ElastoPlastic


@ti.data_oriented
class Snow(ElastoPlastic):
    """
    The snow material class for MPM.

    Note
    ----
    Snow is a special type of ElastoPlastic that get's harder when compressed.
    It doesn't support von Mises yield criterion.

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
        Particle sampler ('pbs', 'regular', 'random'). Default is 'random'.
    yield_lower: float, optional
        Lower bound of yield condition. Default is 2.5e-2.
    yield_higher: float, optional
        Upper bound of yield condition. Default is 4.5e-3.
    """

    def __init__(
        self,
        E=1e6,  # Young's modulus
        nu=0.2,  # Poisson's ratio
        rho=1000.0,  # density (kg/m^3)
        lam=None,
        mu=None,
        sampler="random",
        yield_lower=2.5e-2,
        yield_higher=4.5e-3,
    ):
        super().__init__(
            E=E,
            nu=nu,
            rho=rho,
            lam=lam,
            mu=mu,
            sampler=sampler,
            yield_lower=yield_lower,
            yield_higher=yield_higher,
            use_von_mises=False,
        )

    @ti.func
    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp):
        S_new = ti.Matrix.zero(gs.ti_float, 3, 3)
        Jp_new = Jp
        for d in ti.static(range(3)):
            S_new[d, d] = min(max(S[d, d], 1 - self._yield_lower), 1 + self._yield_higher)
            Jp_new *= S[d, d] / S_new[d, d]
        F_new = U @ S_new @ V.transpose()
        return F_new, S_new, Jp_new

    @ti.func
    def update_stress(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        # Hardening coefficient: material harder when compressed
        h = ti.exp(10 * (1.0 - Jp))
        mu, lam = self._mu * h, self._lam * h

        r = U @ V.transpose()
        stress = 2 * mu * (F_new - r) @ F_new.transpose() + ti.Matrix.identity(gs.ti_float, 3) * lam * J * (J - 1)

        return stress
