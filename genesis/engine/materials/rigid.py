import taichi as ti

import genesis as gs

from .base import Material


@ti.data_oriented
class Rigid(Material):
    """
    The Rigid class represents a material used in rigid body simulation.

    Note
    ----
    This class is intended for use with the rigid solver and provides parameters
    relevant to physical interactions such as friction, density, and signed distance fields (SDFs).

    Parameters
    ----------
        rho : float, optional
            The density of the material used to compute mass. Default is 200.0.
        friction : float, optional
            Friction coefficient within the rigid solver. If None, a default of 1.0 may be used or parsed from file.
        needs_coup : bool, optional
            Whether the material participates in coupling with other solvers. Default is True.
        coup_friction : float, optional
            Friction used during coupling. Must be non-negative. Default is 0.1.
        coup_softness : float, optional
            Softness of coupling interaction. Must be non-negative. Default is 0.002.
        coup_restitution : float, optional
            Restitution coefficient in collision coupling. Should be between 0 and 1. Default is 0.0.
        sdf_cell_size : float, optional
            Cell size in SDF grid in meters. Defines grid resolution. Default is 0.005.
        sdf_min_res : int, optional
            Minimum resolution of the SDF grid. Must be at least 16. Default is 32.
        sdf_max_res : int, optional
            Maximum resolution of the SDF grid. Must be >= sdf_min_res. Default is 128.
        gravity_compensation : float, optional
            Compensation factor for gravity. 1.0 cancels gravity. Default is 0.
    """

    def __init__(
        self,
        rho=200.0,
        friction=None,
        needs_coup=True,
        coup_friction=0.1,
        coup_softness=0.002,
        coup_restitution=0.0,
        sdf_cell_size=0.005,
        sdf_min_res=32,
        sdf_max_res=128,
        gravity_compensation=0,
    ):
        super().__init__()

        if friction is not None:
            if friction < 1e-2 or friction > 5.0:
                gs.raise_exception("`friction` must be in the range [1e-2, 5.0] for simulation stability.")

        if coup_friction < 0:
            gs.raise_exception("`coup_friction` must be non-negative.")

        if coup_softness < 0:
            gs.raise_exception("`coup_softness` must be non-negative.")

        if coup_restitution < 0 or coup_restitution > 1:
            gs.raise_exception("`coup_restitution` must be in the range [0, 1].")

        if coup_restitution != 0:
            gs.logger.warning("Non-zero `coup_restitution` could lead to instability. Use with caution.")

        if sdf_min_res < 16:
            gs.raise_exception("`sdf_min_res` must be at least 16.")

        if sdf_min_res > sdf_max_res:
            gs.raise_exception("`sdf_min_res` must be smaller than or equal to `sdf_max_res`.")

        self._friction = float(friction) if friction is not None else None
        self._needs_coup = bool(needs_coup)
        self._coup_friction = float(coup_friction)
        self._coup_softness = float(coup_softness)
        self._coup_restitution = float(coup_restitution)
        self._sdf_cell_size = float(sdf_cell_size)
        self._sdf_min_res = int(sdf_min_res)
        self._sdf_max_res = int(sdf_max_res)
        self._rho = float(rho)
        self._gravity_compensation = float(gravity_compensation)

    @property
    def gravity_compensation(self):
        """Gravity compensation factor. 1.0 cancels gravity."""
        return self._gravity_compensation

    @property
    def friction(self):
        """Friction coefficient used within the rigid solver."""
        return self._friction

    @property
    def needs_coup(self):
        """Whether this material requires solver coupling."""
        return self._needs_coup

    @property
    def coup_friction(self):
        """Friction coefficient used in coupling interactions."""
        return self._coup_friction

    @property
    def coup_softness(self):
        """Softness parameter controlling the influence range of coupling."""
        return self._coup_softness

    @property
    def coup_restitution(self):
        """Restitution coefficient used during contact in coupling."""
        return self._coup_restitution

    @property
    def sdf_cell_size(self):
        """Size of each SDF grid cell in meters."""
        return self._sdf_cell_size

    @property
    def sdf_min_res(self):
        """Minimum allowed resolution for the SDF grid."""
        return self._sdf_min_res

    @property
    def sdf_max_res(self):
        """Maximum allowed resolution for the SDF grid."""
        return self._sdf_max_res

    @property
    def rho(self):
        """Density of the rigid material."""
        return self._rho
