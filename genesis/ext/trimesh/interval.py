"""
interval.py
--------------

Deal with 1D intervals which are defined by:
  [start position, end position]
"""

import numpy as np


def check(a, b, digits):
    """
    Check input ranges, convert them to vector form,
    and get a fixed precision integer version of them.

    Parameters
    --------------
    a : (2, ) or (2, n) float
      Start and end of a 1D interval
    b : (2, ) or (2, n) float
      Start and end of a 1D interval
    digits : int
      How many digits to consider

    Returns
    --------------
    a : (2, n) float
      Ranges as vector
    b : (2, n) float
      Ranges as vector
    a_int : (2, n) int64
      Ranges rounded to digits, as vector
    b_int : (2, n) int64
      Ranges rounded to digits, as vector
    is_1D : bool
      If True, input was single pair of ranges
    """
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)

    if a.shape != b.shape or a.shape[-1] != 2:
        raise ValueError("ranges must be identical and (2,)!")

    # if input was single interval reshape it here
    is_1D = False
    if len(a.shape) == 1:
        a = a.reshape((-1, 2))
        b = b.reshape((-1, 2))
        is_1D = True

    # make sure ranges are sorted
    a.sort(axis=1)
    b.sort(axis=1)

    # compare in fixed point as integers
    a_int = (a * 10**digits).round().astype(np.int64)
    b_int = (b * 10**digits).round().astype(np.int64)

    return a, b, a_int, b_int, is_1D


def intersection(a, b, digits=8):
    """
    Given a pair of ranges, merge them in to
    one range if they overlap at all

    Parameters
    --------------
    a : (2, ) float
      Start and end of a 1D interval
    b : (2, ) float
      Start and end of a 1D interval
    digits : int
      How many digits to consider

    Returns
    --------------
    intersects : bool or (n,) bool
      Indicates if the ranges overlap at all
    new_range : (2, ) or (2, 2) float
      The unioned range from the two inputs,
      or both of the original ranges if not overlapping
    """
    # check shape and convert
    a, b, a_int, b_int, is_1D = check(a, b, digits)

    # what are the starting and ending points of the overlap
    overlap = np.zeros(a.shape, dtype=np.float64)

    # A fully overlaps B
    current = np.logical_and(a_int[:, 0] <= b_int[:, 0], a_int[:, 1] >= b_int[:, 1])
    overlap[current] = b[current]

    # B fully overlaps A
    current = np.logical_and(a_int[:, 0] >= b_int[:, 0], a_int[:, 1] <= b_int[:, 1])
    overlap[current] = a[current]

    # A starts B ends
    # A:, 0   B:, 0     A:, 1        B:, 1
    current = np.logical_and(
        np.logical_and(a_int[:, 0] <= b_int[:, 0], b_int[:, 0] < a_int[:, 1]), a_int[:, 1] < b_int[:, 1]
    )
    overlap[current] = np.column_stack([b[current][:, 0], a[current][:, 1]])

    # B starts A ends
    # B:, 0  A:, 0    B:, 1  A:, 1
    current = np.logical_and(
        np.logical_and(b_int[:, 0] <= a_int[:, 0], a_int[:, 0] < b_int[:, 1]), b_int[:, 1] < a_int[:, 1]
    )
    overlap[current] = np.column_stack([a[current][:, 0], b[current][:, 1]])

    # is range overlapping at all
    intersects = np.ptp(overlap, axis=1) > 10**-digits

    if is_1D:
        return intersects[0], overlap[0]

    return intersects, overlap
