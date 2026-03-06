from .base import Material


class Kinematic(Material):
    """
    Visualization-only material for ghost/reference entities.

    Kinematic entities are rendered but do not participate in physics simulation,
    collision detection, or constraint solving.
    """

    ...
