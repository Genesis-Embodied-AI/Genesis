from typing import Literal

from .base import EntityT, Material


class Kinematic(Material[EntityT]):
    """
    Visualization-only material for ghost/reference entities.

    Kinematic entities are rendered but do not participate in physics simulation, collision detection, or constraint
    solving. Their visual mesh is always raycastable (use_visual_raycasting=True); the Rigid subclass relaxes the
    field type to StrictBool with default False.
    """

    use_visual_raycasting: Literal[True] = True
