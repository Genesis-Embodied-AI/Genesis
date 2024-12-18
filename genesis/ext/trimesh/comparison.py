"""
comparison.py
----------------

Provide methods for quickly hashing and comparing meshes.
"""

import numpy as np

from . import util

from hashlib import sha256
from .constants import tol

# how many significant figures to use for each
# field of the identifier based on hand-tuning
id_sigfig = np.array(
    [
        5,  # area
        10,  # euler number
        5,  # area/volume ratio
        2,  # convex/mesh area ratio
        2,  # convex area/volume ratio
        3,  # max radius squared / area
        1,
    ]
)  # sign of triangle count for mirrored


def identifier_simple(mesh):
    """
    Return a basic identifier for a mesh consisting of
    properties that have been hand tuned to be somewhat
    robust to rigid transformations and different
    tesselations.

    Parameters
    ------------
    mesh : trimesh.Trimesh
      Source geometry

    Returns
    ----------
    identifier : (7,) float
      Identifying values of the mesh
    """
    # verify the cache once
    mesh._cache.verify()

    # don't check hashes during identifier as we aren't
    # changing any data values of the mesh inside block
    # if we did change values in cache block things would break
    with mesh._cache:
        # pre-allocate identifier so indexes of values can't move around
        # like they might if we used hstack or something else
        identifier = np.zeros(7, dtype=np.float64)
        # avoid thrashing the cache unnecessarily
        mesh_area = mesh.area
        # start with properties that are valid regardless of watertightness
        # note that we're going to try to make all parameters relative
        # to area so other values don't get blown up at weird scales
        identifier[0] = mesh_area
        # avoid divide-by-zero later
        if mesh_area < tol.merge:
            mesh_area = 1.0
        # topological constant and the only thing we can really
        # trust in this fallen world
        identifier[1] = mesh.euler_number

        # if we have a watertight mesh include volume and inertia
        if mesh.is_volume:
            # side length of a cube ratio
            # 1.0 for cubes, different values for other things
            identifier[2] = ((mesh_area / 6.0) ** (1.0 / 2.0)) / (mesh.volume ** (1.0 / 3.0))
        else:
            # if we don't have a watertight mesh add information about the
            # convex hull which is slow to compute and unreliable
            try:
                # get the hull area and volume
                hull = mesh.convex_hull
                hull_area = hull.area
                hull_volume = hull.volume
            except BaseException:
                # in-plane or single point geometry has no hull
                hull_area = 6.0
                hull_volume = 1.0
            # just what we're looking for in a hash but hey
            identifier[3] = mesh_area / hull_area
            # cube side length ratio for the hull
            if hull_volume > 1e-12:
                identifier[4] = ((hull_area / 6.0) ** (1.0 / 2.0)) / (hull_volume ** (1.0 / 3.0))
            # calculate maximum mesh radius
            vertices = mesh.vertices - mesh.centroid
            # add in max radius^2 to area ratio
            R2 = np.dot((vertices**2), [1, 1, 1]).max()
            identifier[5] = R2 / mesh_area

    # mirrored meshes will look identical in terms of
    # area, volume, etc: use a count of relative edge
    # lengths to differentiate identical but mirrored meshes
    # this doesn't work well on meshes with a small number of faces
    if len(mesh.faces) > 50:
        # does this mesh have edges that differ substantially in length
        # if not this method for detecting reflection will not work
        # and the result will definitely be garbage
        edges_length = mesh.edges_unique_length
        variance = edges_length.std() / edges_length.mean()
        if variance > 0.25:
            # the length of each edge in faces
            norms = edges_length[mesh.edges_unique_inverse].reshape((-1, 3))
            # stack edge length and get the relative difference
            stack = np.diff(np.column_stack((norms, norms[:, 0])), axis=1)
            pick_idx = np.abs(stack).argmin(axis=1)
            # get the edge length diff
            pick = stack.reshape(-1)[pick_idx + (np.arange(len(pick_idx)) * 3)]
            # reduce to the bare minimum that tests stable
            identifier[6] = np.sign(pick.sum())
    return identifier


def identifier_hash(identifier):
    """
    Hash an identifier array in a way that is hand-tuned to be
    somewhat robust to likely changes.

    Parameters
    ------------
    identifier : (n,) float
      Vector of properties

    Returns
    ----------
    hash : (64,) str
      A SHA256 of the identifier vector at hand-tuned precision.
    """

    # convert identifier to integers and order of magnitude
    as_int, multiplier = util.sigfig_int(identifier, id_sigfig)

    # make all scales positive
    if (multiplier < 0).any():
        multiplier += np.abs(multiplier.min())
    data = (as_int * (10**multiplier)).astype(np.int64)
    return sha256(data.tobytes()).hexdigest()
