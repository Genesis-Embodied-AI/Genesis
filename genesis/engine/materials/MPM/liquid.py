import gstaichi as ti

import genesis as gs

from .base import Base


@ti.data_oriented
class Liquid(Base):
    """
    The liquid material class for MPM.

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
        Particle sampler ('pbs', 'regular', 'random'). Note that 'pbs' is only supported on Linux for now. Defaults to
        'pbs' on supported platforms, 'random' otherwise.
    viscous: str, bool
        Whether the liquid is viscous. Simply set mu to zero when non-viscuous. Default is False.
    """

    def __init__(
        self,
        E=1e6,
        nu=0.2,
        rho=1000.0,
        lam=None,
        mu=None,
        viscous=False,
        sampler=None,
    ):
        if sampler is None:
            sampler = "pbs" if gs.platform == "Linux" else "random"

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
