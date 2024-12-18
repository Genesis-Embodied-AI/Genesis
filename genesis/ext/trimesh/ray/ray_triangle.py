"""
A basic slow implementation of ray- triangle queries.
"""

import numpy as np


from .ray_util import contains_points

from ..constants import tol

from .. import util
from .. import caching
from .. import grouping
from .. import intersections
from .. import triangles as triangles_mod


class RayMeshIntersector(object):
    """
    An object to query a mesh for ray intersections.
    Precomputes an r-tree for each triangle on the mesh.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        self._cache = caching.Cache(self.mesh.__hash__)

    def intersects_id(self, ray_origins, ray_directions, return_locations=False, multiple_hits=True, **kwargs):
        """
        Find the intersections between the current mesh and an
        array of rays.

        Parameters
        ------------
        ray_origins :  (m, 3) float
          Ray origin points
        ray_directions : (m, 3) float
          Ray direction vectors
        multiple_hits :  bool
          Consider multiple hits of each ray or not
        return_locations : bool
          Return hit locations or not

        Returns
        -----------
        index_triangle : (h,) int
          Index of triangles hit
        index_ray : (h,) int
          Index of ray that hit triangle
        locations : (h, 3) float
          [optional] Position of intersection in space
        """
        (index_tri, index_ray, locations) = ray_triangle_id(
            triangles=self.mesh.triangles,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            tree=self.mesh.triangles_tree,
            multiple_hits=multiple_hits,
            triangles_normal=self.mesh.face_normals,
        )
        if return_locations:
            if len(index_tri) == 0:
                return index_tri, index_ray, locations
            unique = grouping.unique_rows(np.column_stack((locations, index_ray)))[0]
            return index_tri[unique], index_ray[unique], locations[unique]
        return index_tri, index_ray

    def intersects_location(self, ray_origins, ray_directions, **kwargs):
        """
        Return unique cartesian locations where rays hit the mesh.
        If you are counting the number of hits a ray had, this method
        should be used as if only the triangle index is used on- edge hits
        will be counted twice.

        Parameters
        ------------
        ray_origins : (m, 3) float
          Ray origin points
        ray_directions : (m, 3) float
          Ray direction vectors

        Returns
        ---------
        locations : (n) sequence of (m,3) float
          Intersection points
        index_ray : (n,) int
          Array of ray indexes
        index_tri: (n,) int
          Array of triangle (face) indexes
        """
        (index_tri, index_ray, locations) = self.intersects_id(
            ray_origins=ray_origins, ray_directions=ray_directions, return_locations=True, **kwargs
        )
        return locations, index_ray, index_tri

    def intersects_first(self, ray_origins, ray_directions, **kwargs):
        """
        Find the index of the first triangle a ray hits.


        Parameters
        ----------
        ray_origins : (n, 3) float
          Origins of rays
        ray_directions : (n, 3) float
          Direction (vector) of rays

        Returns
        ----------
        triangle_index : (n,) int
          Index of triangle ray hit, or -1 if not hit
        """

        (index_tri, index_ray) = self.intersects_id(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            return_locations=False,
            multiple_hits=False,
            **kwargs,
        )

        # put the result into the form of "one triangle index per ray"
        result = np.ones(len(ray_origins), dtype=np.int64) * -1
        result[index_ray] = index_tri

        return result

    def intersects_any(self, ray_origins, ray_directions, **kwargs):
        """
        Find out if each ray hit any triangle on the mesh.

        Parameters
        ------------
        ray_origins : (m, 3) float
          Ray origin points
        ray_directions : (m, 3) float
          Ray direction vectors

        Returns
        ---------
        hit : (m,) bool
          Whether any ray hit any triangle on the mesh
        """
        index_tri, index_ray = self.intersects_id(ray_origins, ray_directions)
        hit_any = np.zeros(len(ray_origins), dtype=bool)
        hit_idx = np.unique(index_ray)
        if len(hit_idx) > 0:
            hit_any[hit_idx] = True
        return hit_any

    def contains_points(self, points):
        """
        Check if a mesh contains a list of points, using ray tests.

        If the point is on the surface of the mesh the behavior
        is undefined.

        Parameters
        ------------
        points : (n, 3) float
          Points in space

        Returns
        ---------
        contains : (n,) bool
          Whether point is inside mesh or not
        """

        return contains_points(self, points)


def ray_triangle_id(triangles, ray_origins, ray_directions, triangles_normal=None, tree=None, multiple_hits=True):
    """
    Find the intersections between a group of triangles and rays

    Parameters
    -------------
    triangles : (n, 3, 3) float
      Triangles in space
    ray_origins : (m, 3) float
      Ray origin points
    ray_directions : (m, 3) float
      Ray direction vectors
    triangles_normal : (n, 3) float
      Normal vector of triangles, optional
    tree : rtree.Index
      Rtree object holding triangle bounds

    Returns
    -----------
    index_triangle : (h,) int
      Index of triangles hit
    index_ray : (h,) int
      Index of ray that hit triangle
    locations : (h, 3) float
      Position of intersection in space
    """
    triangles = np.asanyarray(triangles, dtype=np.float64)
    ray_origins = np.asanyarray(ray_origins, dtype=np.float64)
    ray_directions = np.asanyarray(ray_directions, dtype=np.float64)

    # if we didn't get passed an r-tree for the bounds of each
    # triangle create one here
    if tree is None:
        tree = triangles_mod.bounds_tree(triangles)

    # find the list of likely triangles and which ray they
    # correspond with, via rtree queries
    ray_candidates, ray_id = ray_triangle_candidates(ray_origins=ray_origins, ray_directions=ray_directions, tree=tree)

    # get subsets which are corresponding rays and triangles
    # (c,3,3) triangle candidates
    triangle_candidates = triangles[ray_candidates]
    # (c,3) origins and vectors for the rays
    line_origins = ray_origins[ray_id]
    line_directions = ray_directions[ray_id]

    # get the plane origins and normals from the triangle candidates
    plane_origins = triangle_candidates[:, 0, :]
    if triangles_normal is None:
        plane_normals, triangle_ok = triangles_mod.normals(triangle_candidates)
        if not triangle_ok.all():
            raise ValueError("Invalid triangles!")
    else:
        plane_normals = triangles_normal[ray_candidates]

    # find the intersection location of the rays with the planes
    location, valid = intersections.planes_lines(
        plane_origins=plane_origins,
        plane_normals=plane_normals,
        line_origins=line_origins,
        line_directions=line_directions,
    )

    if len(triangle_candidates) == 0 or not valid.any():
        # we got no hits so return early with empty array
        return (np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.float64))

    # find the barycentric coordinates of each plane intersection on the
    # triangle candidates
    barycentric = triangles_mod.points_to_barycentric(triangle_candidates[valid], location)

    # the plane intersection is inside the triangle if all barycentric
    # coordinates are between 0.0 and 1.0
    hit = np.logical_and((barycentric > -tol.zero).all(axis=1), (barycentric < (1 + tol.zero)).all(axis=1))

    # the result index of the triangle is a candidate with a valid
    # plane intersection and a triangle which contains the plane
    # intersection point
    index_tri = ray_candidates[valid][hit]
    # the ray index is a subset with a valid plane intersection and
    # contained by a triangle
    index_ray = ray_id[valid][hit]
    # locations are already valid plane intersections, just mask by hits
    location = location[hit]

    # only return points that are forward from the origin
    vector = location - ray_origins[index_ray]
    distance = util.diagonal_dot(vector, ray_directions[index_ray])
    forward = distance > -1e-6

    index_tri = index_tri[forward]
    index_ray = index_ray[forward]
    location = location[forward]
    distance = distance[forward]

    if multiple_hits:
        return index_tri, index_ray, location

    # since we are not returning multiple hits, we need to
    # figure out which hit is first
    if len(index_ray) == 0:
        return index_tri, index_ray, location

    # find the first hit
    first = np.array([g[distance[g].argmin()] for g in grouping.group(index_ray)])

    return index_tri[first], index_ray[first], location[first]


def ray_triangle_candidates(ray_origins, ray_directions, tree):
    """
    Do broad- phase search for triangles that the rays
    may intersect.

    Does this by creating a bounding box for the ray as it
    passes through the volume occupied by the tree

    Parameters
    ------------
    ray_origins:      (m,3) float, ray origin points
    ray_directions:   (m,3) float, ray direction vectors
    tree:             rtree object, contains AABB of each triangle

    Returns
    ----------
    ray_candidates: (n,) int, triangle indexes
    ray_id:         (n,) int, corresponding ray index for a triangle candidate
    """
    ray_bounding = ray_bounds(ray_origins=ray_origins, ray_directions=ray_directions, bounds=tree.bounds)
    ray_candidates = [[]] * len(ray_origins)
    ray_id = [[]] * len(ray_origins)

    for i, bounds in enumerate(ray_bounding):
        ray_candidates[i] = np.array(list(tree.intersection(bounds)), dtype=np.int64)
        ray_id[i] = np.ones(len(ray_candidates[i]), dtype=np.int64) * i

    ray_id = np.hstack(ray_id)
    ray_candidates = np.hstack(ray_candidates)

    return ray_candidates, ray_id


def ray_bounds(ray_origins, ray_directions, bounds, buffer_dist=1e-5):
    """
    Given a set of rays and a bounding box for the volume of interest
    where the rays will be passing through, find the bounding boxes
    of the rays as they pass through the volume.

    Parameters
    ------------
    ray_origins:      (m,3) float, ray origin points
    ray_directions:   (m,3) float, ray direction vectors
    bounds:           (2,3) bounding box (min, max)
    buffer_dist:      float, distance to pad zero width bounding boxes

    Returns
    ---------
    ray_bounding: (n) set of AABB of rays passing through volume
    """

    ray_origins = np.asanyarray(ray_origins, dtype=np.float64)
    ray_directions = np.asanyarray(ray_directions, dtype=np.float64)

    # bounding box we are testing against
    bounds = np.asanyarray(bounds)

    # find the primary axis of the vector
    axis = np.abs(ray_directions).argmax(axis=1)
    axis_bound = bounds.reshape((2, -1)).T[axis]
    axis_ori = np.array([ray_origins[i][a] for i, a in enumerate(axis)]).reshape((-1, 1))
    axis_dir = np.array([ray_directions[i][a] for i, a in enumerate(axis)]).reshape((-1, 1))

    # parametric equation of a line
    # point = direction*t + origin
    # p = dt + o
    # t = (p-o)/d
    t = (axis_bound - axis_ori) / axis_dir

    # prevent the bounding box from including triangles
    # behind the ray origin
    t[t < buffer_dist] = buffer_dist

    # the value of t for both the upper and lower bounds
    t_a = t[:, 0].reshape((-1, 1))
    t_b = t[:, 1].reshape((-1, 1))

    # the cartesion point for where the line hits the plane defined by
    # axis
    on_a = (ray_directions * t_a) + ray_origins
    on_b = (ray_directions * t_b) + ray_origins

    on_plane = np.column_stack((on_a, on_b)).reshape((-1, 2, ray_directions.shape[1]))

    ray_bounding = np.hstack((on_plane.min(axis=1), on_plane.max(axis=1)))
    # pad the bounding box by TOL_BUFFER
    # not sure if this is necessary, but if the ray is  axis aligned
    # this function will otherwise return zero volume bounding boxes
    # which may or may not screw up the r-tree intersection queries
    ray_bounding += np.array([-1, -1, -1, 1, 1, 1]) * buffer_dist

    return ray_bounding
