from .base import EntityT, Material


class Kinematic(Material[EntityT]):
    """
    Visualization-only material for ghost/reference entities.

    Kinematic entities are rendered but do not participate in physics simulation, collision detection, or constraint
    solving.

    Parameters
    ----------
    use_visual_raycasting : bool, optional
        Expose this entity's visual mesh to every consumer that casts against visual geometry: raycaster sensors, and
        viewer plugins built on RaycasterViewerPlugin with use_visual_geom set. Each such consumer then spans the
        entity's visual triangles in its BVH, rebuilt as the entity moves, so reserve it for the entities that are
        meant to be sensed or picked. Default is False.
    """
