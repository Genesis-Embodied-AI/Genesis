import gstaichi as ti

from .base import Material


@ti.data_oriented
class Hybrid(Material):
    """
    The class for hybrid body material (soft skin actuated by inner rigid skeleton).

    Parameters
    ----------
    mat_rigid: gs.materials.base.Material
        The material of the rigid body.
    mat_soft: gs.materials.base.Material
        The material of the soft body.
    fixed: bool, optional
        Whether the rigid entity is with a fixed base link. Default is False.
    use_default_coupling: bool, optional
        Whether to use default solver coupling. Default is False
    damping: float, optional
        Damping coefficient between soft and rigid. Default is 0.0.
    thickness: float, optional
        The thickness to instantiate soft skin. Default is 0.05.
    soft_dv_coef: float, optional
        The coefficient to apply delta velocity from rigid to soft. Default is 0.01.
    func_instantiate_rigid_from_soft: callable, optional
        The function to instantiate rigid body from the geometry of soft body. Default is None.
    func_instantiate_soft_from_rigid: callable, optional
        The function to instantiate soft body from the geometry of rigid body. Default is None.
    func_instantiate_rigid_soft_association: callable, optional
        The function that determines the association of the rigid and the soft body. Default is None.
    """

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
        """The material of the rigid body."""
        return self._mat_rigid

    @property
    def mat_soft(self):
        """The material of the soft body."""
        return self._mat_soft

    @property
    def thickness(self):
        """The thickness to instantiate soft skin."""
        return self._thickness

    @property
    def fixed(self):
        """Whether the rigid entity is with a fixed base link."""
        return self._fixed

    @property
    def use_default_coupling(self):
        """Whether to use default solver coupling."""
        return self._use_default_coupling

    @property
    def damping(self):
        """Damping coefficient between soft and rigid."""
        return self._damping

    @property
    def soft_dv_coef(self):
        """The coefficient to apply delta velocity from rigid to soft."""
        return self._soft_dv_coef
