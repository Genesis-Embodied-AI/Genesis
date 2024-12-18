import taichi as ti

from .base import Material


@ti.data_oriented
class Hybrid(Material):
    def __init__(
        self,
        mat_rigid,
        mat_soft,
        fixed=True,
        use_default_coupling=False,
        damping=0.0,
        thickness=0.05,
        soft_dv_coef=0.01,
        func_instantiate_rigid_from_soft=None,
        func_instantiate_soft_from_rigid=None,
        func_instantiate_rigid_soft_association=None,
    ):
        super().__init__()

        self._mat_rigid = mat_rigid
        self._mat_soft = mat_soft
        self._thickness = thickness
        self._fixed = fixed
        self._use_default_coupling = use_default_coupling
        self._damping = damping
        self._soft_dv_coef = soft_dv_coef
        self._func_instantiate_rigid_from_soft = func_instantiate_rigid_from_soft
        self._func_instantiate_soft_from_rigid = func_instantiate_soft_from_rigid
        self._func_instantiate_rigid_soft_association = func_instantiate_rigid_soft_association

    @property
    def mat_rigid(self):
        return self._mat_rigid

    @property
    def mat_soft(self):
        return self._mat_soft

    @property
    def thickness(self):
        return self._thickness

    @property
    def fixed(self):
        return self._fixed

    @property
    def use_default_coupling(self):
        return self._use_default_coupling

    @property
    def damping(self):
        return self._damping

    @property
    def soft_dv_coef(self):
        return self._soft_dv_coef
