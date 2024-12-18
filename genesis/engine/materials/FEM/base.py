import numpy as np
import taichi as ti

import genesis as gs

from ..base import Material


@ti.data_oriented
class Base(Material):
    def __init__(
        self,
        E=1e6,  # Young's modulus
        nu=0.2,  # Poisson's ratio
        rho=1000.0,  # density (kg/m^3)
    ):
        super().__init__()

        self._E = E
        self._nu = nu
        self._rho = rho

        # lame parameters: https://github.com/taichi-dev/taichi_elements/blob/d19678869a28b09a32ef415b162e35dc929b792d/engine/mpm_solver.py#L203
        self._mu = E / (2.0 * (1.0 + nu))
        self._lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # will be set when added to solver
        self._idx = None

    @ti.func
    def update_stress(self, mu, lam, J, F, actu, m_dir):
        raise NotImplementedError

    @property
    def idx(self):
        return self._idx

    @property
    def E(self):
        return self._E

    @property
    def nu(self):
        return self._nu

    @property
    def mu(self):
        return self._mu

    @property
    def lam(self):
        return self._lam

    @property
    def rho(self):
        return self._rho
