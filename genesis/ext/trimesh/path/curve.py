import numpy as np

from ..constants import res_path as res
from ..constants import tol_path as tol


def discretize_bezier(points, count=None, scale=1.0):
    """
    Parameters
    ----------
    points : (order, dimension) float
      Control points of the bezier curve
      For a 2D cubic bezier, order=3, dimension=2
    count : int, or None
      Number of segments
    scale : float
      Scale of curve
    Returns
    ----------
    discrete: (n, dimension) float
     Points forming a a polyline representation
    """
    # make sure we have a numpy array
    points = np.asanyarray(points, dtype=np.float64)

    if count is None:
        # how much distance does a small percentage of the curve take
        # this is so we can figure out how finely we have to sample t
        norm = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
        count = np.ceil(norm / (res.seg_frac * scale))
        count = int(np.clip(count, res.min_sections * len(points), res.max_sections * len(points)))
    count = int(count)

    # parameterize incrementing 0.0 - 1.0
    t = np.linspace(0.0, 1.0, count)
    # decrementing 1.0-0.0
    t_d = 1.0 - t
    n = len(points) - 1
    # binomial coefficients, i, and each point
    iterable = zip(binomial(n), np.arange(len(points)), points)
    # run the actual interpolation
    stacked = [((t**i) * (t_d ** (n - i))).reshape((-1, 1)) * p * c for c, i, p in iterable]
    result = np.sum(stacked, axis=0)

    # test to make sure end points are correct
    test = np.sum((result[[0, -1]] - points[[0, -1]]) ** 2, axis=1)
    assert (test < tol.merge).all()
    assert len(result) >= 2

    return result


def discretize_bspline(control, knots, count=None, scale=1.0):
    """
    Given a B-Splines control points and knot vector, return
    a sampled version of the curve.

    Parameters
    ----------
    control : (o, d) float
      Control points of the b- spline
    knots : (j,) float
      B-spline knots
    count : int
      Number of line segments to discretize the spline
      If not specified will be calculated as something reasonable

    Returns
    ----------
    discrete : (count, dimension) float
       Points on a polyline version of the B-spline
    """

    # evaluate the b-spline using scipy/fitpack
    from scipy.interpolate import splev

    # (n, d) control points where d is the dimension of vertices
    control = np.asanyarray(control, dtype=np.float64)
    degree = len(knots) - len(control) - 1
    if count is None:
        norm = np.linalg.norm(np.diff(control, axis=0), axis=1).sum()
        count = int(
            np.clip(norm / (res.seg_frac * scale), res.min_sections * len(control), res.max_sections * len(control))
        )

    ipl = np.linspace(knots[0], knots[-1], count)
    discrete = splev(ipl, [knots, control.T, degree])
    discrete = np.column_stack(discrete)

    return discrete


def binomial(n):
    """
    Return all binomial coefficients for a given order.

    For n > 5, scipy.special.binom is used, below we hardcode.

    Parameters
    --------------
    n : int
      Order of binomial

    Returns
    ---------------
    binom : (n + 1,) int
      Binomial coefficients of a given order
    """
    if n == 1:
        return [1, 1]
    elif n == 2:
        return [1, 2, 1]
    elif n == 3:
        return [1, 3, 3, 1]
    elif n == 4:
        return [1, 4, 6, 4, 1]
    elif n == 5:
        return [1, 5, 10, 10, 5, 1]
    else:
        from scipy.special import binom

        return binom(n, np.arange(n + 1))
