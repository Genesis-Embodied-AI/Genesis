import taichi as ti

import genesis as gs

from .base import Material


@ti.data_oriented
class Rigid(Material):
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
        """
        Initialize a Rigid material object.

        Args:
            friction (float): The friction coefficient for the material within the rigid solver. If None, entities will either attempt to parse friction from input file (e.g. MJCF) or use default value of 1.0. Defaults to None.

            rho (float): The default density of the material used to compute mass for each link. Defaults to 1000.0. Note that the mass will be overridden if the entity comes from a urdf/mjcf and with mass specified.

            needs_coup (bool): Whether the material needs coupling with other solvers. Defaults to True.

            coup_friction (float): The friction coefficient for the material during coupling. Defaults to 0.1.

            coup_softness (float): The coupling softness of the material. Defaults to 0.01. When coup_softness is 0, the coupling influence at any point outside the object is 0. (i.e. it's a step function). When coup_softness is > 0, the step function becomes a smooth function. The bigger the value, the smoother the function, i.e. the coupling influence extends further away from the object. For any contact that's distance=d from the object surface, \texttt{influence} = e^(-d/softness).

            coup_restitution (float): The restitution coefficient for other materials colliding with rigid. Defaults to 0.0.

            sdf_cell_size (float): The physical size (in meter) of each cell in the generated SDF grid. This can be interpreted as the size of the detailed features that the SDF grid can capture. This value detemines the final resolution of the SDF grid. Defaults to 0.01 (1cm). For example, for an object with size 1m x 0.5m x 0.5m, the SDF grid will have a physical size of 1.2m x 0.6m x 0.6m (as there's a safety padding of 20%), and the resolution of the grid will be 121 x 61 x 61. (Note that there's additional 1 because the sdf size is measured from the center of the lower cell to the center of the upper cell.)

            sdf_min_res and sdf_max_res (int): The minimum and maximum resolution of the generated SDF grid. Defaults to 32 & 128. The actual resolution of the SDF grid will be clamped between these two values.

            gravity_compensation (float): Apply a force to compensate gravity. A value of 1 will make a zero-gravity behavior. Default to 0.
        """
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
        return self._gravity_compensation

    @property
    def friction(self):
        return self._friction

    @property
    def needs_coup(self):
        return self._needs_coup

    @property
    def coup_friction(self):
        return self._coup_friction

    @property
    def coup_softness(self):
        return self._coup_softness

    @property
    def coup_restitution(self):
        return self._coup_restitution

    @property
    def sdf_cell_size(self):
        return self._sdf_cell_size

    @property
    def sdf_min_res(self):
        return self._sdf_min_res

    @property
    def sdf_max_res(self):
        return self._sdf_max_res

    @property
    def rho(self):
        return self._rho
