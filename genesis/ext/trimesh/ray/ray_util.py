import numpy as np

from .. import util
from .. import bounds
from .. import constants


@constants.log_time
def contains_points(intersector, points, check_direction=None):
    """
    Check if a mesh contains a set of points, using ray tests.

    If the point is on the surface of the mesh, behavior is
    undefined.

    Parameters
    ---------
    mesh: Trimesh object
    points: (n,3) points in space

    Returns
    ---------
    contains : (n) bool
                  Whether point is inside mesh or not
    """
    # convert points to float and make sure they are 3D
    points = np.asanyarray(points, dtype=np.float64)
    if not util.is_shape(points, (-1, 3)):
        raise ValueError("points must be (n,3)")

    # placeholder result with no hits we'll fill in later
    contains = np.zeros(len(points), dtype=bool)

    # cull points outside of the axis aligned bounding box
    # this avoids running ray tests unless points are close
    inside_aabb = bounds.contains(intersector.mesh.bounds, points)

    # if everything is outside the AABB, exit early
    if not inside_aabb.any():
        return contains

    # default ray direction is random, but we are not generating
    # uniquely each time so the behavior of this function is easier to debug
    default_direction = np.array([0.4395064455, 0.617598629942, 0.652231566745])
    if check_direction is None:
        # if no check direction is specified use the default
        # stack it only for points inside the AABB
        ray_directions = np.tile(default_direction, (inside_aabb.sum(), 1))
    else:
        # if a direction is passed use it
        ray_directions = np.tile(np.array(check_direction).reshape(3), (inside_aabb.sum(), 1))

    # cast a ray both forwards and backwards
    location, index_ray, c = intersector.intersects_location(
        np.vstack((points[inside_aabb], points[inside_aabb])), np.vstack((ray_directions, -ray_directions))
    )

    # if we hit nothing in either direction just return with no hits
    if len(index_ray) == 0:
        return contains

    # reshape so bi_hits[0] is the result in the forward direction and
    #            bi_hits[1] is the result in the backwards directions
    bi_hits = np.bincount(index_ray, minlength=len(ray_directions) * 2).reshape((2, -1))
    # a point is probably inside if it hits a surface an odd number of times
    bi_contains = np.mod(bi_hits, 2) == 1

    # if the mod of the hit count is the same in both
    # directions, we can save that result and move on
    agree = np.equal(*bi_contains)

    # in order to do an assignment we can only have one
    # level of boolean indexes, for example this doesn't work:
    # contains[inside_aabb][agree] = bi_contains[0][agree]
    # no error is thrown, but nothing gets assigned
    # to get around that, we create a single mask for assignment
    mask = inside_aabb.copy()
    mask[mask] = agree

    # set contains flags for things inside the AABB and who have
    # ray tests that agree in both directions
    contains[mask] = bi_contains[0][agree]

    # if one of the rays in either direction hit nothing
    # it is a very solid indicator we are in free space
    # as the edge cases we are working around tend to
    # add hits rather than miss hits
    one_freespace = (bi_hits == 0).any(axis=0)

    # rays where they don't agree and one isn't in free space
    # are deemed to be broken
    broken = np.logical_and(np.logical_not(agree), np.logical_not(one_freespace))

    # if all rays agree return
    if not broken.any():
        return contains

    # try to run again with a new random vector
    # only do it if check_direction isn't specified
    # to avoid infinite recursion
    if check_direction is None:
        # we're going to run the check again in a random direction
        new_direction = util.unitize(np.random.random(3) - 0.5)
        # do the mask trick again to be able to assign results
        mask = inside_aabb.copy()
        mask[mask] = broken

        contains[mask] = contains_points(intersector, points[inside_aabb][broken], check_direction=new_direction)

        constants.log.debug("detected %d broken contains test, attempted to fix", broken.sum())

    return contains
