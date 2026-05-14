from .base import EntityT, Material


class Kinematic(Material[EntityT]):
    """
    Visualization-only material for ghost/reference entities.

    Kinematic entities are rendered but do not participate in physics simulation, collision detection, or constraint
    solving. They are ignored by raycaster sensors by default; set use_visual_raycasting=True to include their visual
    mesh in the raycaster BVH.
    """
