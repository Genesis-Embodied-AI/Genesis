import gstaichi as ti

import genesis as gs

from .base import Base


@ti.data_oriented
class Elastic(Base):
    """
    The elastic material class for MPM.

    Note
    ----
    Elastic objects is softened by multiplying the default E by 0.3.
    Reference for the default values of `E` and `nu`:
    https://github.com/taichi-dev/taichi_elements/blob/d19678869a28b09a32ef415b162e35dc929b792d/engine/mpm_solver.py#L201

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
    model: str, optional
        Stress model ('corotation', 'neohooken'). Default is 'corotation'.
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
        """Stress model ('corotation', 'neohooken')"""
        return self._model
