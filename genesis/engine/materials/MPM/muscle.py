import taichi as ti

import genesis as gs

from .elastic import Elastic


@ti.data_oriented
class Muscle(Elastic):
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
        return self._stiffness

    @property
    def n_groups(self):
        return self._n_groups
