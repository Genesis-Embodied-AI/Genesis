import gstaichi as ti

import genesis as gs

from .elastic import Elastic


@ti.data_oriented
class Muscle(Elastic):
    """
    The muscle material class for MPM.

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
    n_groups: int, optional
        Number of muscle group. Default is 1.
    """

    def __init__(
        self,
        E=1e6,
        nu=0.2,
        rho=1000.0,
        lam=None,
        mu=None,
        sampler="pbs",
        model="neohooken",
        n_groups=1,  # number of muscle group
    ):
        super().__init__(E, nu, rho, lam, mu, sampler, model)

        # inherit from Elastic
        self._update_stress_without_actuation = self.update_stress
        self.update_stress = self._update_stress_with_actuation

        self._stiffness = E  # NOTE: use Young's modulus as muscle stiffness
        self._n_groups = n_groups

    @ti.func
    def _update_stress_with_actuation(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        stress = self._update_stress_without_actuation(U, S, V, F_tmp, F_new, J, Jp, actu, m_dir)

        # TODO: need to consider rotation in deformation gradient
        AAt = m_dir.outer_product(m_dir)
        stress += self._stiffness * actu * F_tmp @ AAt @ F_tmp.transpose()

        return stress

    @property
    def stiffness(self):
        """Muscle stiffness."""
        return self._stiffness

    @property
    def n_groups(self):
        """Number of muscle groups."""
        return self._n_groups
