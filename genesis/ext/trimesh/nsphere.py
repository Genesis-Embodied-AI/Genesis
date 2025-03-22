"""
nsphere.py
--------------

Functions for fitting and minimizing nspheres:
circles, spheres, hyperspheres, etc.
"""

import numpy as np

from . import util
from . import convex

from .constants import log, tol

try:
    # scipy is a soft dependency
    from scipy import spatial
    from scipy.optimize import leastsq
except BaseException as E:
    # raise the exception when someone tries to use it
    from . import exceptions

    leastsq = exceptions.ExceptionWrapper(E)
    spatial = exceptions.ExceptionWrapper(E)

try:
    import psutil

    def _MAX_MEMORY():
        # if we have psutil check actual free memory when called
        return psutil.virtual_memory().free / 2.0

except BaseException:

    def _MAX_MEMORY():
        # use a hardcoded best guess estimate
        return 1e9


def minimum_nsphere(obj):
    """
    Compute the minimum n- sphere for a mesh or a set of points.

    Uses the fact that the minimum n- sphere will be centered at one of
    the vertices of the furthest site voronoi diagram, which is n*log(n)
    but should be pretty fast due to using the scipy/qhull implementations
    of convex hulls and voronoi diagrams.

    Parameters
    ----------
    obj : (n, d) float or trimesh.Trimesh
      Points or mesh to find minimum bounding nsphere

    Returns
    ----------
    center : (d,) float
      Center of fitted n- sphere
    radius : float
      Radius of fitted n-sphere
    """

    # reduce the input points or mesh to the vertices of the convex hull
    # since we are computing the furthest site voronoi diagram this reduces
    # the input complexity substantially and returns the same value
    points = convex.hull_points(obj)

    # we are scaling the mesh to a unit cube
    # this used to pass qhull_options 'QbB' to Voronoi however this had a bug somewhere
    # to avoid this we scale to a unit cube ourselves inside this function
    points_origin = points.min(axis=0)
    points_scale = np.ptp(points, axis=0).min()
    points = (points - points_origin) / points_scale

    # if all of the points are on an n-sphere already the voronoi
    # method will fail so we check a least squares fit before
    # bothering to compute the voronoi diagram
    fit_C, fit_R, fit_E = fit_nsphere(points)
    # return fit radius and center to global scale
    fit_R = (((points - fit_C) ** 2).sum(axis=1).max() ** 0.5) * points_scale
    fit_C = (fit_C * points_scale) + points_origin

    if fit_E < 1e-6:
        # points were on an n-sphere so just return fit
        return fit_C, fit_R

    # calculate a furthest site voronoi diagram
    # this will fail if the points are ALL on the surface of
    # the n-sphere but hopefully the least squares check caught those cases
    # , qhull_options='QbB Pp')
    voronoi = spatial.Voronoi(points, furthest_site=True)

    # find the maximum radius^2 point for each of the voronoi vertices
    # this is worst case quite expensive but we have taken
    # convex hull to reduce n for this operation
    # we are doing comparisons on the radius squared then rooting once
    try:
        # cdist is massivly faster than looping or tiling methods
        # although it does create a very large intermediate array
        # first, get an order of magnitude memory size estimate
        # a float64 would be 8 bytes per entry plus overhead
        memory_estimate = len(voronoi.vertices) * len(points) * 9
        if memory_estimate > _MAX_MEMORY():
            raise MemoryError
        radii_2 = spatial.distance.cdist(voronoi.vertices, points, metric="sqeuclidean").max(axis=1)
    except MemoryError:
        # log the MemoryError
        log.warning("MemoryError: falling back to slower check!")
        # fall back to a potentially very slow list comprehension
        radii_2 = np.array([((points - v) ** 2).sum(axis=1).max() for v in voronoi.vertices])

    # we want the smallest sphere so take the min of the radii
    radii_idx = radii_2.argmin()

    # return voronoi radius and center to global scale
    radius_v = np.sqrt(radii_2[radii_idx]) * points_scale
    center_v = (voronoi.vertices[radii_idx] * points_scale) + points_origin

    if radius_v > fit_R:
        return fit_C, fit_R

    return center_v, radius_v


def fit_nsphere(points, prior=None):
    """
    Fit an n-sphere to a set of points using least squares.

    Parameters
    ------------
    points : (n, d) float
      Points in space
    prior : (d,) float
      Best guess for center of nsphere

    Returns
    ---------
    center : (d,) float
      Location of center
    radius : float
      Mean radius across circle
    error : float
      Peak to peak value of deviation from mean radius
    """
    # make sure points are numpy array
    points = np.asanyarray(points, dtype=np.float64)
    # create ones so we can dot instead of using slower sum
    ones = np.ones(points.shape[1])

    def residuals(center):
        # do the axis sum with a dot
        # this gets called a LOT so worth optimizing
        radii_sq = np.dot((points - center) ** 2, ones)
        # residuals are difference between mean
        # use our sum mean vs .mean() as it is slightly faster
        return radii_sq - (radii_sq.sum() / len(radii_sq))

    if prior is None:
        guess = points.mean(axis=0)
    else:
        guess = np.asanyarray(prior)

    center_result, return_code = leastsq(residuals, guess, xtol=1e-8)

    if return_code not in [1, 2, 3, 4]:
        raise ValueError("Least square fit failed!")

    radii = util.row_norm(points - center_result)
    radius = radii.mean()
    error = np.ptp(radii)
    return center_result, radius, error


def is_nsphere(points):
    """
    Check if a list of points is an nsphere.

    Parameters
    -----------
    points : (n, dimension) float
      Points in space

    Returns
    -----------
    check : bool
      True if input points are on an nsphere
    """
    center, radius, error = fit_nsphere(points)
    check = error < tol.merge
    return check
