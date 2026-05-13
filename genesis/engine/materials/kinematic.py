from pydantic import StrictBool

from .base import EntityT, Material


class Kinematic(Material[EntityT]):
    """
    Visualization-only material for ghost/reference entities.

    Kinematic entities are rendered but do not participate in physics simulation,
    collision detection, or constraint solving.

    Parameters
    ----------
    use_visual_raycasting : bool, optional
        When True, the entity's visual mesh is included in the raycaster BVH so that
        depth cameras and lidars can see it. Must be set before ``scene.build()``.
        Defaults to False.
    """

    use_visual_raycasting: StrictBool = False
