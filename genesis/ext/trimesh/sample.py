"""
sample.py
------------

Randomly sample surface and volume of meshes.
"""

import numpy as np

from . import util
from . import transformations
from .visual import uv_to_interpolated_color

if hasattr(np.random, "default_rng"):
    # newer versions of Numpy
    default_rng = np.random.default_rng
else:
    # Python 2 Numpy
    class default_rng(np.random.RandomState):
        def random(self, *args, **kwargs):
            return self.random_sample(*args, **kwargs)


def sample_surface(mesh, count, face_weight=None, sample_color=False, seed=None):
    """
    Sample the surface of a mesh, returning the specified
    number of points

    For individual triangle sampling uses this method:
    http://mathworld.wolfram.com/TrianglePointPicking.html

    Parameters
    -----------
    mesh : trimesh.Trimesh
      Geometry to sample the surface of
    count : int
      Number of points to return
    face_weight : None or len(mesh.faces) float
      Weight faces by a factor other than face area.
      If None will be the same as face_weight=mesh.area
    sample_color : bool
      Option to calculate the color of the sampled points.
      Default is False.
    seed : None or int
      If passed as an integer will provide deterministic results
      otherwise pulls the seed from operating system entropy.

    Returns
    ---------
    samples : (count, 3) float
      Points in space on the surface of mesh
    face_index : (count,) int
      Indices of faces for each sampled point
    colors : (count, 4) float
      Colors of each sampled point
      Returns only when the sample_color is True
    """

    if face_weight is None:
        # len(mesh.faces) float, array of the areas
        # of each face of the mesh
        face_weight = mesh.area_faces

    # cumulative sum of weights (len(mesh.faces))
    weight_cum = np.cumsum(face_weight)

    # seed the random number generator as requested
    random = default_rng(seed).random

    # last value of cumulative sum is total summed weight/area
    face_pick = random(count) * weight_cum[-1]
    # get the index of the selected faces
    face_index = np.searchsorted(weight_cum, face_pick)

    # pull triangles into the form of an origin + 2 vectors
    tri_origins = mesh.vertices[mesh.faces[:, 0]]
    tri_vectors = mesh.vertices[mesh.faces[:, 1:]].copy()
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

    # pull the vectors for the faces we are going to sample from
    tri_origins = tri_origins[face_index]
    tri_vectors = tri_vectors[face_index]

    if sample_color and hasattr(mesh.visual, "uv"):
        uv_origins = mesh.visual.uv[mesh.faces[:, 0]]
        uv_vectors = mesh.visual.uv[mesh.faces[:, 1:]].copy()
        uv_origins_tile = np.tile(uv_origins, (1, 2)).reshape((-1, 2, 2))
        uv_vectors -= uv_origins_tile
        uv_origins = uv_origins[face_index]
        uv_vectors = uv_vectors[face_index]

    # randomly generate two 0-1 scalar components to multiply edge vectors b
    random_lengths = random((len(tri_vectors), 2, 1))

    # points will be distributed on a quadrilateral if we use 2 0-1 samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    if sample_color:
        if hasattr(mesh.visual, "uv"):
            sample_uv_vector = (uv_vectors * random_lengths).sum(axis=1)
            uv_samples = sample_uv_vector + uv_origins
            texture = mesh.visual.material.image
            colors = uv_to_interpolated_color(uv_samples, texture)
        else:
            colors = mesh.visual.face_colors[face_index]

        return samples, face_index, colors

    return samples, face_index


def volume_mesh(mesh, count):
    """
    Use rejection sampling to produce points randomly
    distributed in the volume of a mesh.


    Parameters
    -----------
    mesh : trimesh.Trimesh
      Geometry to sample
    count : int
      Number of points to return

    Returns
    ---------
    samples : (n, 3) float
      Points in the volume of the mesh where n <= count
    """
    points = (np.random.random((count, 3)) * mesh.extents) + mesh.bounds[0]
    contained = mesh.contains(points)
    samples = points[contained][:count]
    return samples


def volume_rectangular(extents, count, transform=None):
    """
    Return random samples inside a rectangular volume,
    useful for sampling inside oriented bounding boxes.

    Parameters
    -----------
    extents :   (3,) float
      Side lengths of rectangular solid
    count : int
      Number of points to return
    transform : (4, 4) float
      Homogeneous transformation matrix

    Returns
    ---------
    samples : (count, 3) float
      Points in requested volume
    """
    samples = np.random.random((count, 3)) - 0.5
    samples *= extents
    if transform is not None:
        samples = transformations.transform_points(samples, transform)
    return samples


def sample_surface_even(mesh, count, radius=None, seed=None):
    """
    Sample the surface of a mesh, returning samples which are
    VERY approximately evenly spaced. This is accomplished by
    sampling and then rejecting pairs that are too close together.

    Note that since it is using rejection sampling it may return
    fewer points than requested (i.e. n < count). If this is the
    case a log.warning will be emitted.

    Parameters
    -----------
    mesh : trimesh.Trimesh
      Geometry to sample the surface of
    count : int
      Number of points to return
    radius : None or float
      Removes samples below this radius
    seed : None or int
      Provides deterministic values

    Returns
    ---------
    samples : (n, 3) float
      Points in space on the surface of mesh
    face_index : (n,) int
      Indices of faces for each sampled point
    """
    from .points import remove_close

    # guess radius from area
    if radius is None:
        radius = np.sqrt(mesh.area / (3 * count))

    # get points on the surface
    points, index = sample_surface(mesh, count * 3, seed=seed)

    # remove the points closer than radius
    points, mask = remove_close(points, radius)

    # we got all the samples we expect
    if len(points) >= count:
        return points[:count], index[mask][:count]

    # warn if we didn't get all the samples we expect
    util.log.warning("only got {}/{} samples!".format(len(points), count))

    return points, index[mask]


def sample_surface_sphere(count):
    """
    Correctly pick random points on the surface of a unit sphere

    Uses this method:
    http://mathworld.wolfram.com/SpherePointPicking.html

    Parameters
    -----------
    count : int
      Number of points to return

    Returns
    ----------
    points : (count, 3) float
      Random points on the surface of a unit sphere
    """
    # get random values 0.0-1.0
    u, v = np.random.random((2, count))
    # convert to two angles
    theta = np.pi * 2 * u
    phi = np.arccos((2 * v) - 1)
    # convert spherical coordinates to cartesian
    points = util.spherical_to_vector(np.column_stack((theta, phi)))
    return points
