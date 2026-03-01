from .base import Material


class Kinematic(Material):
    """
    Visualization-only material for ghost/reference entities.

    Kinematic entities are rendered but do not participate in physics simulation,
    collision detection, or constraint solving.

    Parameters
    ----------
    rho : float, optional
        The density of the material used for inertial parsing. Default is 200.0.
    """

    def __init__(self, rho=200.0):
        super().__init__()
        self._rho = float(rho)

    @property
    def rho(self):
        return self._rho
