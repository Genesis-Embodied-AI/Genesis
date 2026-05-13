from typing import Any

from pydantic import StrictBool

import genesis as gs

from .base import EntityT, Material


class Kinematic(Material[EntityT]):
    """
    Visualization-only material for ghost/reference entities.

    Kinematic entities are rendered but do not participate in physics simulation, collision detection, or constraint
    solving.

    Parameters
    ----------
    use_visual_raycasting : bool, optional
        When True, the entity's visual mesh is included in the raycaster BVH so that depth cameras and lidars can see
        it. Must be set before scene.build(). Defaults to True for Kinematic materials (kinematic entities exist
        primarily to be visualized and sensed) and to False for Rigid materials (rigid entities are raycast against
        their collision geometry by default).
    """

    use_visual_raycasting: StrictBool = True

    def model_post_init(self, context: Any) -> None:
        # Only plain Kinematic enforces the True default; the Rigid subclass overrides it to False.
        if type(self) is Kinematic and not self.use_visual_raycasting:
            gs.raise_exception("'use_visual_raycasting' must be True for Kinematic materials.")
