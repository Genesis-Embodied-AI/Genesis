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
        lam=None,  # Lame's first parameter
        mu=None,  # Lame's second parameter
        sampler="pbs",  # particle sampler
    ):
        """
        lam and mu will be computed based on E and nu if not provided.
        """
        super().__init__()

        self._E = E
        self._nu = nu
        self._rho = rho
        self._sampler = sampler
        self._default_Jp = 1.0

        # lame parameters: https://github.com/taichi-dev/taichi_elements/blob/d19678869a28b09a32ef415b162e35dc929b792d/engine/mpm_solver.py#L203
        if mu is None:
            self._mu = E / (2.0 * (1.0 + nu))
        else:
            self._mu = mu

        if lam is None:
            self._lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        else:
            self._lam = lam

        # will be set when added to solver
        self._idx = None

    @classmethod
    def _repr_type(cls):
        return f"<gs.materials.MPM.{cls.__name__}>"

    @ti.func
    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp):
        raise NotImplementedError

    @ti.func
    def update_stress(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        # NOTE: class member function inheritance will still introduce redundant computation graph in taichi
        stress = 2 * self._mu * (F_new - U @ V.transpose()) @ F_new.transpose() + ti.Matrix.identity(
            gs.ti_float, 3
        ) * self._lam * J * (J - 1)

        return stress

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            keys = self.__dict__.keys()
            for key in keys:
                # these keys are not relevant to material models
                if key not in ["_id", "_idx", "_sampler"] and key.startswith("_"):
                    if self.__dict__[key] != other.__dict__[key]:
                        return False
            return True
        return False

    @property
    def uid(self):
        return self._uid

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

    @property
    def sampler(self):
        return self._sampler
