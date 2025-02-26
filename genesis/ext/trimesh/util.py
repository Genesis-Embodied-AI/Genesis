"""
util.py
-----------

Standalone functions which require only imports from numpy and the
standard library.

Other libraries may be imported must be wrapped in try/except blocks
or imported inside of a function
"""

import abc
import sys
import copy
import json
import uuid
import base64
import random
import shutil
import logging
import zipfile
import tempfile
import collections

import numpy as np

if sys.version_info >= (3, 4):
    # for newer version of python
    ABC = abc.ABC
else:
    # an abstract base class that works on older versions
    ABC = abc.ABCMeta("ABC", (), {})

# a flag we can check elsewhere for Python 3
PY3 = sys.version_info.major >= 3

if PY3:
    # for type checking
    basestring = str
    # Python 3
    from io import BytesIO, StringIO
    from shutil import which  # noqa
    from time import perf_counter as now  # noqa
else:
    # Python 2
    from StringIO import StringIO
    from distutils.spawn import find_executable as which  # noqa

    # monkey patch StringIO so `with` statements work
    StringIO.__enter__ = lambda a: a
    StringIO.__exit__ = lambda a, b, c, d: a.close()
    BytesIO = StringIO
    from time import time as now  # noqa


try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

# create a default logger
log = logging.getLogger("trimesh")

# include constants here so we don't have to import
# a floating point threshold for 0.0
# we are setting it to 100x the resolution of a float64
# which works out to be 1e-13
TOL_ZERO = np.finfo(np.float64).resolution * 100
# how close to merge vertices
TOL_MERGE = 1e-8
# enable additional potentially slow checks
_STRICT = False

_IDENTITY = np.eye(4, dtype=np.float64)
_IDENTITY.flags["WRITEABLE"] = False


def has_module(name):
    """
    Check to see if a module is installed by name without
    actually importing the module.

    Parameters
    ------------
    name : str
      The name of the module to check

    Returns
    ------------
    installed : bool
      True if module is installed
    """
    # this should work on Python 2.7 and 3.4+
    import pkgutil

    return pkgutil.find_loader(name) is not None


def unitize(vectors, check_valid=False, threshold=None):
    """
    Unitize a vector or an array or row-vectors.

    Parameters
    ------------
    vectors : (n,m) or (j) float
       Vector or vectors to be unitized
    check_valid :  bool
       If set, will return mask of nonzero vectors
    threshold : float
       Cutoff for a value to be considered zero.

    Returns
    ---------
    unit :  (n,m) or (j) float
       Input vectors but unitized
    valid : (n,) bool or bool
        Mask of nonzero vectors returned if `check_valid`
    """
    # make sure we have a numpy array
    vectors = np.asanyarray(vectors)

    # allow user to set zero threshold
    if threshold is None:
        threshold = TOL_ZERO

    if len(vectors.shape) == 2:
        # for (m, d) arrays take the per-row unit vector
        # using sqrt and avoiding exponents is slightly faster
        # also dot with ones is faser than .sum(axis=1)
        norm = np.sqrt(np.dot(vectors * vectors, [1.0] * vectors.shape[1]))
        # non-zero norms
        valid = norm > threshold
        # in-place reciprocal of nonzero norms
        norm[valid] **= -1
        # multiply by reciprocal of norm
        unit = vectors * norm.reshape((-1, 1))

    elif len(vectors.shape) == 1:
        # treat 1D arrays as a single vector
        norm = np.sqrt(np.dot(vectors, vectors))
        valid = norm > threshold
        if valid:
            unit = vectors / norm
        else:
            unit = vectors.copy()
    else:
        raise ValueError("vectors must be (n, ) or (n, d)!")

    if check_valid:
        return unit[valid], valid
    return unit


def euclidean(a, b):
    """
    Euclidean distance between vectors a and b.

    Parameters
    ------------
    a : (n,) float
       First vector
    b : (n,) float
       Second vector

    Returns
    ------------
    distance : float
        Euclidean distance between A and B
    """
    a = np.asanyarray(a, dtype=np.float64)
    b = np.asanyarray(b, dtype=np.float64)
    return np.sqrt(((a - b) ** 2).sum())


def is_file(obj):
    """
    Check if an object is file-like

    Parameters
    ------------
    obj : object
       Any object type to be checked

    Returns
    -----------
    is_file : bool
        True if object is a file
    """
    return hasattr(obj, "read") or hasattr(obj, "write")


def is_pathlib(obj):
    """
    Check if the object is a `pathlib.Path` or subclass.

    Parameters
    ------------
    obj : object
      Object to be checked

    Returns
    ------------
    is_pathlib : bool
      Is the input object a pathlib path
    """
    # check class name rather than a pathlib import
    name = obj.__class__.__name__
    return hasattr(obj, "absolute") and name.endswith("Path")


def is_string(obj):
    """
    Check if an object is a string.

    Parameters
    ------------
    obj : object
       Any object type to be checked

    Returns
    ------------
    is_string : bool
        True if obj is a string
    """
    return isinstance(obj, basestring)


def is_none(obj):
    """
    Check to see if an object is None or not.

    Handles the case of np.array(None) as well.

    Parameters
    -------------
    obj : object
      Any object type to be checked

    Returns
    -------------
    is_none : bool
        True if obj is None or numpy None-like
    """
    if obj is None:
        return True
    if is_sequence(obj) and len(obj) == 1 and obj[0] is None:
        return True
    return False


def is_sequence(obj):
    """
    Check if an object is a sequence or not.

    Parameters
    -------------
    obj : object
      Any object type to be checked

    Returns
    -------------
    is_sequence : bool
        True if object is sequence
    """
    seq = not hasattr(obj, "strip") and hasattr(obj, "__getitem__") or hasattr(obj, "__iter__")

    # check to make sure it is not a set, string, or dictionary
    seq = seq and all(not isinstance(obj, i) for i in (dict, set, basestring))

    # PointCloud objects can look like an array but are not
    seq = seq and type(obj).__name__ not in ["PointCloud"]

    # numpy sometimes returns objects that are single float64 values
    # but sure look like sequences, so we check the shape
    if hasattr(obj, "shape"):
        seq = seq and obj.shape != ()

    return seq


def is_shape(obj, shape, allow_zeros=False):
    """
    Compare the shape of a numpy.ndarray to a target shape,
    with any value less than zero being considered a wildcard

    Note that if a list-like object is passed that is not a numpy
    array, this function will not convert it and will return False.

    Parameters
    ------------
    obj :   np.ndarray
      Array to check the shape on
    shape : list or tuple
      Any negative term will be considered a wildcard
      Any tuple term will be evaluated as an OR
    allow_zeros: bool
      if False, zeros do not match negatives in shape

    Returns
    ---------
    shape_ok : bool
      True if shape of obj matches query shape

    Examples
    ------------------------
    In [1]: a = np.random.random((100, 3))

    In [2]: a.shape
    Out[2]: (100, 3)

    In [3]: trimesh.util.is_shape(a, (-1, 3))
    Out[3]: True

    In [4]: trimesh.util.is_shape(a, (-1, 3, 5))
    Out[4]: False

    In [5]: trimesh.util.is_shape(a, (100, -1))
    Out[5]: True

    In [6]: trimesh.util.is_shape(a, (-1, (3, 4)))
    Out[6]: True

    In [7]: trimesh.util.is_shape(a, (-1, (4, 5)))
    Out[7]: False
    """

    # if the obj.shape is different length than
    # the goal shape it means they have different number
    # of dimensions and thus the obj is not the query shape
    if not hasattr(obj, "shape") or len(obj.shape) != len(shape):
        return False

    # empty lists with any flexible dimensions match
    if len(obj) == 0 and -1 in shape:
        return True

    # loop through each integer of the two shapes
    # multiple values are sequences
    # wildcards are less than zero (i.e. -1)
    for i, target in zip(obj.shape, shape):
        # check if current field has multiple acceptable values
        if is_sequence(target):
            if i in target:
                # obj shape is in the accepted values
                continue
            else:
                return False

        # check if current field is a wildcard
        if target < 0:
            if i == 0 and not allow_zeros:
                # if a dimension is 0, we don't allow
                # that to match to a wildcard
                # it would have to be explicitly called out as 0
                return False
            else:
                continue
        # since we have a single target and a single value,
        # if they are not equal we have an answer
        if target != i:
            return False

    # since none of the checks failed the obj.shape
    # matches the pattern
    return True


def make_sequence(obj):
    """
    Given an object, if it is a sequence return, otherwise
    add it to a length 1 sequence and return.

    Useful for wrapping functions which sometimes return single
    objects and other times return lists of objects.

    Parameters
    -------------
    obj : object
      An object to be made a sequence

    Returns
    --------------
    as_sequence : (n,) sequence
       Contains input value
    """
    if is_sequence(obj):
        return np.array(list(obj))
    else:
        return np.array([obj])


def vector_hemisphere(vectors, return_sign=False):
    """
    For a set of 3D vectors alter the sign so they are all in the
    upper hemisphere.

    If the vector lies on the plane all vectors with negative Y
    will be reversed.

    If the vector has a zero Z and Y value vectors with a
    negative X value will be reversed.

    Parameters
    ------------
    vectors : (n, 3) float
      Input vectors
    return_sign : bool
      Return the sign mask or not

    Returns
    ----------
    oriented: (n, 3) float
      Vectors with same magnitude as source
      but possibly reversed to ensure all vectors
      are in the same hemisphere.
    sign : (n,) float
      [OPTIONAL] sign of original vectors
    """
    # vectors as numpy array
    vectors = np.asanyarray(vectors, dtype=np.float64)

    if is_shape(vectors, (-1, 2)):
        # 2D vector case
        # check the Y value and reverse vector
        # direction if negative.
        negative = vectors < -TOL_ZERO
        zero = np.logical_not(np.logical_or(negative, vectors > TOL_ZERO))

        signs = np.ones(len(vectors), dtype=np.float64)
        # negative Y values are reversed
        signs[negative[:, 1]] = -1.0

        # zero Y and negative X are reversed
        signs[np.logical_and(zero[:, 1], negative[:, 0])] = -1.0

    elif is_shape(vectors, (-1, 3)):
        # 3D vector case
        negative = vectors < -TOL_ZERO
        zero = np.logical_not(np.logical_or(negative, vectors > TOL_ZERO))
        # move all                          negative Z to positive
        # then for zero Z vectors, move all negative Y to positive
        # then for zero Y vectors, move all negative X to positive
        signs = np.ones(len(vectors), dtype=np.float64)
        # all vectors with negative Z values
        signs[negative[:, 2]] = -1.0
        # all on-plane vectors with negative Y values
        signs[np.logical_and(zero[:, 2], negative[:, 1])] = -1.0
        # all on-plane vectors with zero Y values
        # and negative X values
        signs[np.logical_and(np.logical_and(zero[:, 2], zero[:, 1]), negative[:, 0])] = -1.0

    else:
        raise ValueError("vectors must be (n, 3)!")

    # apply the signs to the vectors
    oriented = vectors * signs.reshape((-1, 1))

    if return_sign:
        return oriented, signs

    return oriented


def vector_to_spherical(cartesian):
    """
    Convert a set of cartesian points to (n, 2) spherical unit
    vectors.

    Parameters
    ------------
    cartesian : (n, 3) float
       Points in space

    Returns
    ------------
    spherical : (n, 2) float
       Angles, in radians
    """
    cartesian = np.asanyarray(cartesian, dtype=np.float64)
    if not is_shape(cartesian, (-1, 3)):
        raise ValueError("Cartesian points must be (n, 3)!")

    unit, valid = unitize(cartesian, check_valid=True)
    unit[np.abs(unit) < TOL_MERGE] = 0.0

    x, y, z = unit.T
    spherical = np.zeros((len(cartesian), 2), dtype=np.float64)
    spherical[valid] = np.column_stack((np.arctan2(y, x), np.arccos(z)))
    return spherical


def spherical_to_vector(spherical):
    """
    Convert a set of (n, 2) spherical vectors to (n, 3) vectors

    Parameters
    ------------
    spherical : (n , 2) float
       Angles, in radians

    Returns
    -----------
    vectors : (n, 3) float
      Unit vectors
    """
    spherical = np.asanyarray(spherical, dtype=np.float64)
    if not is_shape(spherical, (-1, 2)):
        raise ValueError("spherical coordinates must be (n, 2)!")

    theta, phi = spherical.T
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    vectors = np.column_stack((ct * sp, st * sp, cp))
    return vectors


def pairwise(iterable):
    """
    For an iterable, group values into pairs.

    Parameters
    ------------
    iterable : (m, ) list
       A sequence of values

    Returns
    -----------
    pairs: (n, 2)
      Pairs of sequential values

    Example
    -----------
    In [1]: data
    Out[1]: [0, 1, 2, 3, 4, 5, 6]

    In [2]: list(trimesh.util.pairwise(data))
    Out[2]: [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]

    """
    # looping through a giant numpy array would be dumb
    # so special case ndarrays and use numpy operations
    if isinstance(iterable, np.ndarray):
        iterable = iterable.reshape(-1)
        stacked = np.column_stack((iterable, iterable))
        pairs = stacked.reshape(-1)[1:-1].reshape((-1, 2))
        return pairs

    # if we have a normal iterable use itertools
    import itertools

    a, b = itertools.tee(iterable)
    # pop the first element of the second item
    next(b)

    return zip(a, b)


try:
    # prefer the faster numpy version of multi_dot
    # only included in recent-ish version of numpy
    multi_dot = np.linalg.multi_dot
except AttributeError:
    log.debug("np.linalg.multi_dot not available, using fallback")

    def multi_dot(arrays):
        """
        Compute the dot product of two or more arrays in a single function call.
        In most versions of numpy this is included, this slower function is
        provided for backwards compatibility with ancient versions of numpy.
        """
        arrays = np.asanyarray(arrays)
        result = arrays[0].copy()
        for i in arrays[1:]:
            result = np.dot(result, i)
        return result


def diagonal_dot(a, b):
    """
    Dot product by row of a and b.

    There are a lot of ways to do this though
    performance varies very widely. This method
    uses a dot product to sum the row and avoids
    function calls if at all possible.

    Comparing performance of some equivalent versions:
    ```
    In [1]: import numpy as np; import trimesh

    In [2]: a = np.random.random((10000, 3))

    In [3]: b = np.random.random((10000, 3))

    In [4]: %timeit (a * b).sum(axis=1)
    1000 loops, best of 3: 181 us per loop

    In [5]: %timeit np.einsum('ij,ij->i', a, b)
    10000 loops, best of 3: 62.7 us per loop

    In [6]: %timeit np.diag(np.dot(a, b.T))
    1 loop, best of 3: 429 ms per loop

    In [7]: %timeit np.dot(a * b, np.ones(a.shape[1]))
    10000 loops, best of 3: 61.3 us per loop

    In [8]: %timeit trimesh.util.diagonal_dot(a, b)
    10000 loops, best of 3: 55.2 us per loop
    ```

    Parameters
    ------------
    a : (m, d) float
      First array
    b : (m, d) float
      Second array

    Returns
    -------------
    result : (m,) float
      Dot product of each row
    """
    # make sure `a` is numpy array
    # doing it for `a` will force the multiplication to
    # convert `b` if necessary and avoid function call otherwise
    a = np.asanyarray(a)
    # 3x faster than (a * b).sum(axis=1)
    # avoiding np.ones saves 5-10% sometimes
    return np.dot(a * b, [1.0] * a.shape[1])


def row_norm(data):
    """
    Compute the norm per-row of a numpy array.

    This is identical to np.linalg.norm(data, axis=1) but roughly
    three times faster due to being less general.

    In [3]: %timeit trimesh.util.row_norm(a)
    76.3 us +/- 651 ns per loop

    In [4]: %timeit np.linalg.norm(a, axis=1)
    220 us +/- 5.41 us per loop

    Parameters
    -------------
    data : (n, d) float
      Input 2D data to calculate per-row norm of

    Returns
    -------------
    norm : (n,) float
      Norm of each row of input array
    """
    return np.sqrt(np.dot(data**2, [1] * data.shape[1]))


def stack_3D(points, return_2D=False):
    """
    For a list of (n, 2) or (n, 3) points return them
    as (n, 3) 3D points, 2D points on the XY plane.

    Parameters
    ------------
    points :  (n, 2) or (n, 3) float
      Points in either 2D or 3D space
    return_2D : bool
      Were the original points 2D?

    Returns
    ----------
    points : (n, 3) float
      Points in space
    is_2D : bool
      [OPTIONAL] if source points were (n, 2)
    """
    points = np.asanyarray(points, dtype=np.float64)
    shape = points.shape

    if shape == (0,):
        is_2D = False
    elif len(shape) != 2:
        raise ValueError("Points must be 2D array!")
    elif shape[1] == 2:
        points = np.column_stack((points, np.zeros(len(points))))
        is_2D = True
    elif shape[1] == 3:
        is_2D = False
    else:
        raise ValueError("Points must be (n, 2) or (n, 3)!")

    if return_2D:
        return points, is_2D

    return points


def grid_arange(bounds, step):
    """
    Return a grid from an (2,dimension) bounds with samples step distance apart.

    Parameters
    ------------
    bounds: (2,dimension) list of [[min x, min y, etc], [max x, max y, etc]]
    step:   float, or (dimension) floats, separation between points

    Returns
    ---------
    grid: (n, dimension), points inside the specified bounds
    """
    bounds = np.asanyarray(bounds, dtype=np.float64)
    if len(bounds) != 2:
        raise ValueError("bounds must be (2, dimension!")

    # allow single float or per-dimension spacing
    step = np.asanyarray(step, dtype=np.float64)
    if step.shape == ():
        step = np.tile(step, bounds.shape[1])

    grid_elements = [np.arange(*b, step=s) for b, s in zip(bounds.T, step)]
    grid = np.vstack(np.meshgrid(*grid_elements, indexing="ij")).reshape(bounds.shape[1], -1).T
    return grid


def grid_linspace(bounds, count):
    """
    Return a grid spaced inside a bounding box with edges spaced using np.linspace.

    Parameters
    ------------
    bounds: (2,dimension) list of [[min x, min y, etc], [max x, max y, etc]]
    count:  int, or (dimension,) int, number of samples per side

    Returns
    ---------
    grid: (n, dimension) float, points in the specified bounds
    """
    bounds = np.asanyarray(bounds, dtype=np.float64)
    if len(bounds) != 2:
        raise ValueError("bounds must be (2, dimension!")

    count = np.asanyarray(count, dtype=np.int64)
    if count.shape == ():
        count = np.tile(count, bounds.shape[1])

    grid_elements = [np.linspace(*b, num=c) for b, c in zip(bounds.T, count)]
    grid = np.vstack(np.meshgrid(*grid_elements, indexing="ij")).reshape(bounds.shape[1], -1).T
    return grid


def multi_dict(pairs):
    """
    Given a set of key value pairs, create a dictionary.
    If a key occurs multiple times, stack the values into an array.

    Can be called like the regular dict(pairs) constructor

    Parameters
    ------------
    pairs: (n, 2) array of key, value pairs

    Returns
    ----------
    result: dict, with all values stored (rather than last with regular dict)

    """
    result = collections.defaultdict(list)
    for k, v in pairs:
        result[k].append(v)
    return result


def tolist(data):
    """
    Ensure that any arrays or dicts passed containing
    numpy arrays are properly converted to lists

    Parameters
    -------------
    data : any
      Usually a dict with some numpy arrays as values

    Returns
    ----------
    result : any
      JSON-serializable version of data
    """
    result = json.loads(jsonify(data))
    return result


def is_binary_file(file_obj):
    """
    Returns True if file has non-ASCII characters (> 0x7F, or 127)
    Should work in both Python 2 and 3
    """
    start = file_obj.tell()
    fbytes = file_obj.read(1024)
    file_obj.seek(start)
    is_str = isinstance(fbytes, str)
    for fbyte in fbytes:
        if is_str:
            code = ord(fbyte)
        else:
            code = fbyte
        if code > 127:
            return True
    return False


def distance_to_end(file_obj):
    """
    For an open file object how far is it to the end

    Parameters
    ------------
    file_obj: open file-like object

    Returns
    ----------
    distance: int, bytes to end of file
    """
    position_current = file_obj.tell()
    file_obj.seek(0, 2)
    position_end = file_obj.tell()
    file_obj.seek(position_current)
    distance = position_end - position_current
    return distance


def decimal_to_digits(decimal, min_digits=None):
    """
    Return the number of digits to the first nonzero decimal.

    Parameters
    -----------
    decimal:    float
    min_digits: int, minimum number of digits to return

    Returns
    -----------

    digits: int, number of digits to the first nonzero decimal
    """
    digits = abs(int(np.log10(decimal)))
    if min_digits is not None:
        digits = np.clip(digits, min_digits, 20)
    return digits


def attach_to_log(level=logging.DEBUG, handler=None, loggers=None, colors=True, capture_warnings=True, blacklist=None):
    """
    Attach a stream handler to all loggers.

    Parameters
    ------------
    level : enum
      Logging level, like logging.INFO
    handler : None or logging.Handler
      Handler to attach
    loggers : None or (n,) logging.Logger
      If None, will try to attach to all available
    colors : bool
      If True try to use colorlog formatter
    blacklist : (n,) str
      Names of loggers NOT to attach to
    """

    # default blacklist includes ipython debugging stuff
    if blacklist is None:
        blacklist = ["TerminalIPythonApp", "PYREADLINE", "pyembree", "shapely", "matplotlib", "parso"]

    # make sure we log warnings from the warnings module
    logging.captureWarnings(capture_warnings)

    # create a basic formatter
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s (%(filename)s:%(lineno)3s) %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    if colors:
        try:
            from colorlog import ColoredFormatter

            formatter = ColoredFormatter(
                ("%(log_color)s%(levelname)-8s%(reset)s " + "%(filename)17s:%(lineno)-4s  %(blue)4s%(message)s"),
                datefmt=None,
                reset=True,
                log_colors={"DEBUG": "cyan", "INFO": "green", "WARNING": "yellow", "ERROR": "red", "CRITICAL": "red"},
            )
        except ImportError:
            pass

    # if no handler was passed use a StreamHandler
    if handler is None:
        handler = logging.StreamHandler()

    # add the formatters and set the level
    handler.setFormatter(formatter)
    handler.setLevel(level)

    # if nothing passed use all available loggers
    if loggers is None:
        # de-duplicate loggers using a set
        loggers = set(logging.Logger.manager.loggerDict.values())
    # add the warnings logging
    loggers.add(logging.getLogger("py.warnings"))

    # disable pyembree warnings
    logging.getLogger("pyembree").disabled = True

    # loop through all available loggers
    for logger in loggers:
        # skip loggers on the blacklist
        if logger.__class__.__name__ != "Logger" or any(logger.name.startswith(b) for b in blacklist):
            continue
        logger.addHandler(handler)
        logger.setLevel(level)

    # set nicer numpy print options
    np.set_printoptions(precision=5, suppress=True)


def stack_lines(indices):
    """
    Stack a list of values that represent a polyline into
    individual line segments with duplicated consecutive values.

    Parameters
    ------------
    indices : (m,) any
      List of items to be stacked

    Returns
    ---------
    stacked : (n, 2) any
      Stacked items

    Examples
    ----------
    In [1]: trimesh.util.stack_lines([0, 1, 2])
    Out[1]:
    array([[0, 1],
           [1, 2]])

    In [2]: trimesh.util.stack_lines([0, 1, 2, 4, 5])
    Out[2]:
    array([[0, 1],
           [1, 2],
           [2, 4],
           [4, 5]])

    In [3]: trimesh.util.stack_lines([[0, 0], [1, 1], [2, 2], [3, 3]])
    Out[3]:
    array([[0, 0],
           [1, 1],
           [1, 1],
           [2, 2],
           [2, 2],
           [3, 3]])

    """
    indices = np.asanyarray(indices)
    if len(indices) == 0:
        return np.array([])
    elif is_sequence(indices[0]):
        shape = (-1, len(indices[0]))
    else:
        shape = (-1, 2)
    return np.column_stack((indices[:-1], indices[1:])).reshape(shape)


def append_faces(vertices_seq, faces_seq):
    """
    Given a sequence of zero-indexed faces and vertices
    combine them into a single array of faces and
    a single array of vertices.

    Parameters
    -----------
    vertices_seq : (n, ) sequence of (m, d) float
      Multiple arrays of verticesvertex arrays
    faces_seq : (n, ) sequence of (p, j) int
      Zero indexed faces for matching vertices

    Returns
    ----------
    vertices : (i, d) float
      Points in space
    faces : (j, 3) int
      Reference vertex indices
    """
    # the length of each vertex array
    vertices_len = np.array([len(i) for i in vertices_seq])
    # how much each group of faces needs to be offset
    face_offset = np.append(0, np.cumsum(vertices_len)[:-1])

    new_faces = []
    for offset, faces in zip(face_offset, faces_seq):
        if len(faces) == 0:
            continue
        # apply the index offset
        new_faces.append(faces + offset)
    # stack to clean (n, 3) float
    vertices = vstack_empty(vertices_seq)
    # stack to clean (n, 3) int
    faces = vstack_empty(new_faces)

    return vertices, faces


def array_to_string(array, col_delim=" ", row_delim="\n", digits=8, value_format="{}"):
    """
    Convert a 1 or 2D array into a string with a specified number
    of digits and delimiter. The reason this exists is that the
    basic numpy array to string conversions are surprisingly bad.

    Parameters
    ------------
    array : (n,) or (n, d) float or int
       Data to be converted
       If shape is (n,) only column delimiter will be used
    col_delim : str
      What string should separate values in a column
    row_delim : str
      What string should separate values in a row
    digits : int
      How many digits should floating point numbers include
    value_format : str
       Format string for each value or sequence of values
       If multiple values per value_format it must divide
       into array evenly.

    Returns
    ----------
    formatted : str
       String representation of original array
    """
    # convert inputs to correct types
    array = np.asanyarray(array)
    digits = int(digits)
    row_delim = str(row_delim)
    col_delim = str(col_delim)
    value_format = str(value_format)

    # abort for non-flat arrays
    if len(array.shape) > 2:
        raise ValueError("conversion only works on 1D/2D arrays not %s!", str(array.shape))

    # abort for structured arrays
    if array.dtype.names is not None:
        raise ValueError("array is  structured, use structured_array_to_string instead")

    # allow a value to be repeated in a value format
    repeats = value_format.count("{")

    if array.dtype.kind in ["i", "u"]:
        # integer types don't need a specified precision
        format_str = value_format + col_delim
    elif array.dtype.kind == "f":
        # add the digits formatting to floats
        format_str = value_format.replace("{}", "{:." + str(digits) + "f}") + col_delim
    else:
        raise ValueError("dtype %s not convertible!", array.dtype.name)

    # length of extra delimiters at the end
    end_junk = len(col_delim)
    # if we have a 2D array add a row delimiter
    if len(array.shape) == 2:
        format_str *= array.shape[1]
        # cut off the last column delimiter and add a row delimiter
        format_str = format_str[: -len(col_delim)] + row_delim
        end_junk = len(row_delim)

    # expand format string to whole array
    format_str *= len(array)

    # if an array is repeated in the value format
    # do the shaping here so we don't need to specify indexes
    shaped = np.tile(array.reshape((-1, 1)), (1, repeats)).reshape(-1)

    # run the format operation and remove the extra delimiters
    formatted = format_str.format(*shaped)[:-end_junk]

    return formatted


def structured_array_to_string(array, col_delim=" ", row_delim="\n", digits=8, value_format="{}"):
    """
    Convert an unstructured array into a string with a specified
    number of digits and delimiter. The reason thisexists is
    that the basic numpy array to string conversionsare
    surprisingly bad.

    Parameters
    ------------
    array : (n,) or (n, d) float or int
       Data to be converted
       If shape is (n,) only column delimiter will be used
    col_delim : str
      What string should separate values in a column
    row_delim : str
      What string should separate values in a row
    digits : int
      How many digits should floating point numbers include
    value_format : str
       Format string for each value or sequence of values
       If multiple values per value_format it must divide
       into array evenly.

    Returns
    ----------
    formatted : str
       String representation of original array
    """
    # convert inputs to correct types
    array = np.asanyarray(array)
    digits = int(digits)
    row_delim = str(row_delim)
    col_delim = str(col_delim)
    value_format = str(value_format)

    # abort for non-flat arrays
    if len(array.shape) > 1:
        raise ValueError("conversion only works on 1D/2D arrays not %s!", str(array.shape))

    # abort for unstructured arrays
    if array.dtype.names is None:
        raise ValueError("array is not structured, use array_to_string instead")

    # do not allow a value to be repeated in a value format
    if value_format.count("{") > 1:
        raise ValueError(
            "value_format %s is invalid, repeating unstructured array " + "values is unsupported", value_format
        )

    format_str = ""
    for name in array.dtype.names:
        kind = array[name].dtype.kind
        element_row_length = array[name].shape[1] if len(array[name].shape) == 2 else 1
        if kind in ["i", "u"]:
            # integer types need a no-decimal formatting
            element_format_str = value_format.replace("{}", "{:0.0f}") + col_delim
        elif kind == "f":
            # add the digits formatting to floats
            element_format_str = value_format.replace("{}", "{:." + str(digits) + "f}") + col_delim
        else:
            raise ValueError("dtype %s not convertible!", array.dtype)
        format_str += element_row_length * element_format_str

    # length of extra delimiters at the end
    format_str = format_str[: -len(col_delim)] + row_delim
    # expand format string to whole array
    format_str *= len(array)

    # loop through flat fields and flatten to single array
    count = len(array)
    # will upgrade everything to a float
    flattened = np.hstack([array[k].reshape((count, -1)) for k in array.dtype.names]).reshape(-1)

    # run the format operation and remove the extra delimiters
    formatted = format_str.format(*flattened)[: -len(row_delim)]

    return formatted


def array_to_encoded(array, dtype=None, encoding="base64"):
    """
    Export a numpy array to a compact serializable dictionary.

    Parameters
    ------------
    array : array
      Any numpy array
    dtype : str or None
      Optional dtype to encode array
    encoding : str
      'base64' or 'binary'

    Returns
    ---------
    encoded : dict
      Has keys:
      'dtype':  str, of dtype
      'shape':  tuple of shape
      'base64': str, base64 encoded string
    """
    array = np.asanyarray(array)
    shape = array.shape
    # ravel also forces contiguous
    flat = np.ravel(array)
    if dtype is None:
        dtype = array.dtype

    encoded = {"dtype": np.dtype(dtype).str, "shape": shape}
    if encoding in ["base64", "dict64"]:
        packed = base64.b64encode(flat.astype(dtype).tobytes())
        if hasattr(packed, "decode"):
            packed = packed.decode("utf-8")
        encoded["base64"] = packed
    elif encoding == "binary":
        encoded["binary"] = array.tobytes(order="C")
    else:
        raise ValueError("encoding {} is not available!".format(encoding))
    return encoded


def decode_keys(store, encoding="utf-8"):
    """
    If a dictionary has keys that are bytes decode them to a str.

    Parameters
    ------------
    store : dict
      Dictionary with data

    Returns
    ---------
    result : dict
      Values are untouched but keys that were bytes
      are converted to ASCII strings.

    Example
    -----------
    In [1]: d
    Out[1]: {1020: 'nah', b'hi': 'stuff'}

    In [2]: trimesh.util.decode_keys(d)
    Out[2]: {1020: 'nah', 'hi': 'stuff'}
    """
    keys = store.keys()
    for key in keys:
        if hasattr(key, "decode"):
            decoded = key.decode(encoding)
            if key != decoded:
                store[key.decode(encoding)] = store[key]
                store.pop(key)
    return store


def comment_strip(text, starts_with="#", new_line="\n"):
    """
    Strip comments from a text block.

    Parameters
    -----------
    text : str
      Text to remove comments from
    starts_with : str
      Character or substring that starts a comment
    new_line : str
      Character or substring that ends a comment

    Returns
    -----------
    stripped : str
      Text with comments stripped
    """
    # if not contained exit immediately
    if starts_with not in text:
        return text

    # start by splitting into chunks by the comment indicator
    split = (text + new_line).split(starts_with)

    # special case files that start with a comment
    if text.startswith(starts_with):
        lead = ""
    else:
        lead = split[0]

    # take each comment up until the newline
    removed = [i.split(new_line, 1) for i in split]
    # add the leading string back on
    result = lead + new_line + new_line.join(i[1] for i in removed if len(i) > 1 and len(i[1]) > 0)
    # strip leading and trailing whitespace
    result = result.strip()

    return result


def encoded_to_array(encoded):
    """
    Turn a dictionary with base64 encoded strings back into a numpy array.

    Parameters
    ------------
    encoded : dict
      Has keys:
        dtype: string of dtype
        shape: int tuple of shape
        base64: base64 encoded string of flat array
        binary:  decode result coming from numpy.tobytes

    Returns
    ----------
    array: numpy array
    """

    if not isinstance(encoded, dict):
        if is_sequence(encoded):
            as_array = np.asanyarray(encoded)
            return as_array
        else:
            raise ValueError("Unable to extract numpy array from input")

    encoded = decode_keys(encoded)

    dtype = np.dtype(encoded["dtype"])
    if "base64" in encoded:
        array = np.frombuffer(base64.b64decode(encoded["base64"]), dtype)
    elif "binary" in encoded:
        array = np.frombuffer(encoded["binary"], dtype=dtype)
    if "shape" in encoded:
        array = array.reshape(encoded["shape"])
    return array


def is_instance_named(obj, name):
    """
    Given an object, if it is a member of the class 'name',
    or a subclass of 'name', return True.

    Parameters
    ------------
    obj : instance
      Some object of some class
    name: str
      The name of the class we want to check for

    Returns
    ---------
    is_instance : bool
      Whether the object is a member of the named class
    """
    try:
        if isinstance(name, list):
            return any(is_instance_named(obj, i) for i in name)
        else:
            type_named(obj, name)
        return True
    except ValueError:
        return False


def type_bases(obj, depth=4):
    """
    Return the bases of the object passed.
    """
    bases = collections.deque([list(obj.__class__.__bases__)])
    for i in range(depth):
        bases.append([i.__base__ for i in bases[-1] if i is not None])
    try:
        bases = np.hstack(bases)
    except IndexError:
        bases = []
    return [i for i in bases if hasattr(i, "__name__")]


def type_named(obj, name):
    """
    Similar to the type() builtin, but looks in class bases
    for named instance.

    Parameters
    ------------
    obj : any
      Object to look for class of
    name : str
      Nnme of class

    Returns
    ----------
    class : Optional[Callable]
      Camed class, or None
    """
    # if obj is a member of the named class, return True
    name = str(name)
    if obj.__class__.__name__ == name:
        return obj.__class__
    for base in type_bases(obj):
        if base.__name__ == name:
            return base
    raise ValueError("Unable to extract class of name " + name)


def concatenate(a, b=None):
    """
    Concatenate two or more meshes.

    Parameters
    ------------
    a : trimesh.Trimesh
      Mesh or list of meshes to be concatenated
      object, or list of such
    b : trimesh.Trimesh
      Mesh or list of meshes to be concatenated

    Returns
    ----------
    result : trimesh.Trimesh
      Concatenated mesh
    """

    # get a flat list of meshes
    flat = []
    if a is not None:
        if is_sequence(a):
            flat.extend(a)
        else:
            flat.append(a)
    if b is not None:
        if is_sequence(b):
            flat.extend(b)
        else:
            flat.append(b)

    if len(flat) == 1:
        # if there is only one mesh just return the first
        return flat[0].copy()
    elif len(flat) == 0:
        # if there are no meshes return an empty list
        return []

    is_mesh = [f for f in flat if is_instance_named(f, "Trimesh")]
    is_path = [f for f in flat if is_instance_named(f, "Path")]

    if len(is_path) > len(is_mesh):
        from .path.util import concatenate as concatenate_path

        return concatenate_path(is_path)

    if len(is_mesh) == 0:
        return []

    # extract the trimesh type to avoid a circular import
    # and assert that all inputs are Trimesh objects
    trimesh_type = type_named(is_mesh[0], "Trimesh")

    # append faces and vertices of meshes
    vertices, faces = append_faces([m.vertices.copy() for m in is_mesh], [m.faces.copy() for m in is_mesh])

    # only save face normals if already calculated
    face_normals = None
    if all("face_normals" in m._cache for m in is_mesh):
        face_normals = np.vstack([m.face_normals for m in is_mesh])

    try:
        # concatenate visuals
        visual = is_mesh[0].visual.concatenate([m.visual for m in is_mesh[1:]])
    except BaseException:
        log.debug("failed to combine visuals", exc_info=True)
        visual = None

    # create the mesh object
    return trimesh_type(vertices=vertices, faces=faces, face_normals=face_normals, visual=visual, process=False)


def submesh(mesh, faces_sequence, repair=True, only_watertight=False, min_faces=None, append=False):
    """
    Return a subset of a mesh.

    Parameters
    ------------
    mesh : Trimesh
        Source mesh to take geometry from
    faces_sequence : sequence (p,) int
        Indexes of mesh.faces
    repair : bool
        Try to make submeshes watertight
    only_watertight : bool
        Only return submeshes which are watertight
    append : bool
        Return a single mesh which has the faces appended,
        if this flag is set, only_watertight is ignored

    Returns
    ---------
    if append : Trimesh object
    else        list of Trimesh objects
    """
    # evaluate generators so we can escape early
    faces_sequence = list(faces_sequence)

    if len(faces_sequence) == 0:
        return []

    # avoid nuking the cache on the original mesh
    original_faces = mesh.faces.view(np.ndarray)
    original_vertices = mesh.vertices.view(np.ndarray)

    faces = []
    vertices = []
    normals = []
    visuals = []

    # for reindexing faces
    mask = np.arange(len(original_vertices))

    for index in faces_sequence:
        # sanitize indices in case they are coming in as a set or tuple
        index = np.asanyarray(index)
        if len(index) == 0:
            # regardless of type empty arrays are useless
            continue
        if index.dtype.kind == "b":
            # if passed a bool with no true continue
            if not index.any():
                continue
            # if fewer faces than minimum
            if min_faces is not None and index.sum() < min_faces:
                continue
        elif min_faces is not None and len(index) < min_faces:
            continue

        current = original_faces[index]
        unique = np.unique(current.reshape(-1))

        # redefine face indices from zero
        mask[unique] = np.arange(len(unique))
        normals.append(mesh.face_normals[index])
        faces.append(mask[current])
        vertices.append(original_vertices[unique])

        try:
            visuals.append(mesh.visual.face_subset(index))
        except BaseException:
            visuals = None

    if len(vertices) == 0:
        return np.array([])

    # we use type(mesh) rather than importing Trimesh from base
    # to avoid a circular import
    trimesh_type = type_named(mesh, "Trimesh")
    if append:
        visual = None
        try:
            visuals = np.array(visuals)
            visual = visuals[0].concatenate(visuals[1:])
        except BaseException:
            pass
        # re-index faces and stack
        vertices, faces = append_faces(vertices, faces)
        appended = trimesh_type(
            vertices=vertices, faces=faces, face_normals=np.vstack(normals), visual=visual, process=False
        )
        return appended

    if visuals is None:
        visuals = [None] * len(vertices)

    # generate a list of Trimesh objects
    result = [
        trimesh_type(
            vertices=v, faces=f, face_normals=n, visual=c, metadata=copy.deepcopy(mesh.metadata), process=False
        )
        for v, f, n, c in zip(vertices, faces, normals, visuals)
    ]
    result = np.array(result)
    if only_watertight or repair:
        # fill_holes will attempt a repair and returns the
        # watertight status at the end of the repair attempt
        watertight = np.array([i.fill_holes() and len(i.faces) >= 4 for i in result])
    if only_watertight:
        # remove unrepairable meshes
        result = result[watertight]

    return result


def zero_pad(data, count, right=True):
    """
    Parameters
    ------------
    data : (n,)
      1D array
    count : int
      Minimum length of result array

    Returns
    ---------
    padded : (m,)
      1D array where m >= count
    """
    if len(data) == 0:
        return np.zeros(count)
    elif len(data) < count:
        padded = np.zeros(count)
        if right:
            padded[-len(data) :] = data
        else:
            padded[: len(data)] = data
        return padded
    else:
        return np.asanyarray(data)


def jsonify(obj, **kwargs):
    """
    A version of json.dumps that can handle numpy arrays
    by creating a custom encoder for numpy dtypes.

    Parameters
    --------------
    obj : list, dict
      A JSON-serializable blob
    kwargs : dict
      Passed to json.dumps

    Returns
    --------------
    dumped : str
      JSON dump of obj
    """

    class EdgeEncoder(json.JSONEncoder):
        def default(self, obj):
            # will work for numpy.ndarrays
            # as well as their int64/etc objects
            if hasattr(obj, "tolist"):
                return obj.tolist()
            elif hasattr(obj, "timestamp"):
                return obj.timestamp()
            return json.JSONEncoder.default(self, obj)

    # run the dumps using our encoder
    return json.dumps(obj, cls=EdgeEncoder, **kwargs)


def convert_like(item, like):
    """
    Convert an item to have the dtype of another item

    Parameters
    ------------
    item : any
      Item to be converted
    like : any
      Object with target dtype
      If None, item is returned unmodified

    Returns
    ----------
    result: item, but in dtype of like
    """
    # if it's a numpy array
    if isinstance(like, np.ndarray):
        return np.asanyarray(item, dtype=like.dtype)

    # if it's already the desired type just return it
    if isinstance(item, like.__class__) or is_none(like):
        return item

    # if it's an array with one item return it
    if is_sequence(item) and len(item) == 1 and isinstance(item[0], like.__class__):
        return item[0]

    if isinstance(item, str) and like.__class__.__name__ == "Polygon" and item.startswith("POLYGON"):
        # break our rule on imports but only a little bit
        # the import was a WKT serialized polygon
        from shapely import wkt

        return wkt.loads(item)

    # otherwise just run the conversion
    item = like.__class__(item)

    return item


def bounds_tree(bounds):
    """
    Given a set of axis aligned bounds create an r-tree for
    broad-phase collision detection.

    Parameters
    ------------
    bounds : (n, 2D) or (n, 2, D) float
      Non-interleaved bounds where D=dimension
      E.G a 2D bounds tree:
      [(minx, miny, maxx, maxy), ...]

    Returns
    ---------
    tree : Rtree
      Tree containing bounds by index
    """
    # rtree is a soft dependency
    import rtree

    # make sure we've copied bounds
    bounds = np.array(bounds, dtype=np.float64, copy=True)
    if len(bounds.shape) == 3:
        # should be min-max per bound
        if bounds.shape[1] != 2:
            raise ValueError("bounds not (n, 2, dimension)!")
        # reshape to one-row-per-hyperrectangle
        bounds = bounds.reshape((len(bounds), -1))
    elif len(bounds.shape) != 2 or bounds.size == 0:
        raise ValueError("Bounds must be (n, dimension * 2)!")

    # check to make sure we have correct shape
    dimension = bounds.shape[1]
    if (dimension % 2) != 0:
        raise ValueError("Bounds must be (n,dimension*2)!")
    dimension = int(dimension / 2)

    # some versions of rtree screw up indexes on stream loading
    # do a test here so we know if we are free to use stream loading
    # or if we have to do a loop to insert things which is 5x slower
    rtree_test = rtree.index.Index([(1564, [0, 0, 0, 10, 10, 10], None)], properties=rtree.index.Property(dimension=3))
    rtree_stream_ok = next(rtree_test.intersection([1, 1, 1, 2, 2, 2])) == 1564

    properties = rtree.index.Property(dimension=dimension)
    if rtree_stream_ok:
        # stream load was verified working on import above
        tree = rtree.index.Index(zip(np.arange(len(bounds)), bounds, [None] * len(bounds)), properties=properties)
    else:
        # in some rtree versions stream loading goofs the index
        log.warning("rtree stream loading broken! Try upgrading rtree!")
        tree = rtree.index.Index(properties=properties)
        for i, b in enumerate(bounds):
            tree.insert(i, b)
    return tree


def wrap_as_stream(item):
    """
    Wrap a string or bytes object as a file object.

    Parameters
    ------------
    item: str or bytes
      Item to be wrapped

    Returns
    ---------
    wrapped : file-like object
      Contains data from item
    """
    if not PY3:
        # in python 2 StringIO handles bytes and str
        return StringIO(item)
    if isinstance(item, str):
        return StringIO(item)
    elif isinstance(item, bytes):
        return BytesIO(item)
    raise ValueError("{} is not wrappable!".format(type(item).__name__))


def sigfig_round(values, sigfig=1):
    """
    Round a single value to a specified number of significant figures.

    Parameters
    ------------
    values : float
      Value to be rounded
    sigfig : int
      Number of significant figures to reduce to

    Returns
    ----------
    rounded : float
      Value rounded to the specified number of significant figures


    Examples
    ----------
    In [1]: trimesh.util.round_sigfig(-232453.00014045456, 1)
    Out[1]: -200000.0

    In [2]: trimesh.util.round_sigfig(.00014045456, 1)
    Out[2]: 0.0001

    In [3]: trimesh.util.round_sigfig(.00014045456, 4)
    Out[3]: 0.0001405
    """
    as_int, multiplier = sigfig_int(values, sigfig)
    rounded = as_int * (10**multiplier)

    return rounded


def sigfig_int(values, sigfig):
    """
    Convert a set of floating point values into integers
    with a specified number of significant figures and an
    exponent.

    Parameters
    ------------
    values : (n,) float or int
      Array of values
    sigfig : (n,) int
      Number of significant figures to keep

    Returns
    ------------
    as_int : (n,) int
      Every value[i] has sigfig[i] digits
    multiplier : (n, int)
      Exponent, so as_int * 10 ** multiplier is
      the same order of magnitude as the input
    """
    values = np.asanyarray(values).reshape(-1)
    sigfig = np.asanyarray(sigfig, dtype=np.int64).reshape(-1)

    if sigfig.shape != values.shape:
        raise ValueError("sigfig must match identifier")

    exponent = np.zeros(len(values))
    nonzero = np.abs(values) > TOL_ZERO
    exponent[nonzero] = np.floor(np.log10(np.abs(values[nonzero])))

    multiplier = exponent - sigfig + 1
    as_int = (values / (10**multiplier)).round().astype(np.int64)

    return as_int, multiplier


def decompress(file_obj, file_type):
    """
    Given an open file object and a file type, return all components
    of the archive as open file objects in a dict.

    Parameters
    ------------
    file_obj : file-like
      Containing compressed data
    file_type : str
      File extension, 'zip', 'tar.gz', etc

    Returns
    ---------
    decompressed : dict
      Data from archive in format {file name : file-like}
    """

    file_type = str(file_type).lower()
    if isinstance(file_obj, bytes):
        file_obj = wrap_as_stream(file_obj)

    if file_type.endswith("zip"):
        archive = zipfile.ZipFile(file_obj)
        return {name: wrap_as_stream(archive.read(name)) for name in archive.namelist()}
    if "tar" in file_type[-6:]:
        import tarfile

        archive = tarfile.open(fileobj=file_obj, mode="r")
        return {name: archive.extractfile(name) for name in archive.getnames()}
    raise ValueError("Unsupported type passed!")


def compress(info, **kwargs):
    """
    Compress data stored in a dict.

    Parameters
    -----------
    info : dict
      Data to compress in form:
      {file name in archive: bytes or file-like object}
    kwargs : dict
      Passed to zipfile.ZipFile
    Returns
    -----------
    compressed : bytes
      Compressed file data
    """
    if PY3:
        file_obj = BytesIO()
    else:
        file_obj = StringIO()

    with zipfile.ZipFile(file_obj, mode="w", compression=zipfile.ZIP_DEFLATED, **kwargs) as zipper:
        for name, data in info.items():
            if hasattr(data, "read"):
                # if we were passed a file object, read it
                data = data.read()
            zipper.writestr(name, data)
    file_obj.seek(0)
    compressed = file_obj.read()
    return compressed


def split_extension(file_name, special=None):
    """
    Find the file extension of a file name, including support for
    special case multipart file extensions (like .tar.gz)

    Parameters
    ------------
    file_name : str
      File name
    special : list of str
      Multipart extensions
      eg: ['tar.bz2', 'tar.gz']

    Returns
    ----------
    extension : str
      Last characters after a period, or
      a value from 'special'
    """
    file_name = str(file_name)

    if special is None:
        special = ["tar.bz2", "tar.gz"]
    if file_name.endswith(tuple(special)):
        for end in special:
            if file_name.endswith(end):
                return end
    return file_name.split(".")[-1]


def triangle_strips_to_faces(strips):
    """
    Convert a sequence of triangle strips to (n, 3) faces.

    Processes all strips at once using np.concatenate and is significantly
    faster than loop-based methods.

    From the OpenGL programming guide describing a single triangle
    strip [v0, v1, v2, v3, v4]:

    Draws a series of triangles (three-sided polygons) using vertices
    v0, v1, v2, then v2, v1, v3  (note the order), then v2, v3, v4,
    and so on. The ordering is to ensure that the triangles are all
    drawn with the same orientation so that the strip can correctly form
    part of a surface.

    Parameters
    ------------
    strips: (n,) list of (m,) int
      Vertex indices

    Returns
    ------------
    faces : (m, 3) int
      Vertex indices representing triangles
    """

    # save the length of each list in the list of lists
    lengths = np.array([len(i) for i in strips])
    # looping through a list of lists is extremely slow
    # combine all the sequences into a blob we can manipulate
    blob = np.concatenate(strips)

    # preallocate and slice the blob into rough triangles
    tri = np.zeros((len(blob) - 2, 3), dtype=np.int64)
    for i in range(3):
        tri[: len(blob) - 3, i] = blob[i : -3 + i]
    # the last triangle is left off from the slicing, add it back
    tri[-1] = blob[-3:]

    # remove the triangles which were implicit but not actually there
    # because we combined everything into one big array for speed
    length_index = np.cumsum(lengths)[:-1]
    keep = np.ones(len(tri), dtype=bool)
    keep[length_index - 2] = False
    keep[length_index - 1] = False
    tri = tri[keep]

    # flip every other triangle so they generate correct normals/winding
    length_index = np.append(0, np.cumsum(lengths - 2))
    flip = np.zeros(length_index[-1], dtype=bool)
    for i in range(len(length_index) - 1):
        flip[length_index[i] + 1 : length_index[i + 1]][::2] = True
    tri[flip] = np.fliplr(tri[flip])

    return tri


def triangle_fans_to_faces(fans):
    """
    Convert fans of m + 2 vertex indices in fan format to m triangles

    Parameters
    ----------
    fans: (n,) list of (m + 2,) int
      Vertex indices

    Returns
    -------
    faces: (m, 3) int
      Vertex indices representing triangles
    """

    faces = [np.transpose([fan[0] * np.ones(len(fan) - 2, dtype=int), fan[1:-1], fan[2:]]) for fan in fans]
    return np.concatenate(faces, axis=1)


def vstack_empty(tup):
    """
    A thin wrapper for numpy.vstack that ignores empty lists.

    Parameters
    ------------
    tup : tuple or list of arrays
      With the same number of columns

    Returns
    ------------
    stacked : (n, d) array
      With same number of columns as
      constituent arrays.
    """
    # filter out empty arrays
    stackable = [i for i in tup if len(i) > 0]
    # if we only have one array just return it
    if len(stackable) == 1:
        return np.asanyarray(stackable[0])
    # if we have nothing return an empty numpy array
    elif len(stackable) == 0:
        return np.array([])
    # otherwise just use vstack as normal
    return np.vstack(stackable)


def write_encoded(file_obj, stuff, encoding="utf-8"):
    """
    If a file is open in binary mode and a
    string is passed, encode and write.

    If a file is open in text mode and bytes are
    passed decode bytes to str and write.

    Assumes binary mode if file_obj does not have
    a 'mode' attribute (e.g. io.BufferedRandom).

    Parameters
    -----------
    file_obj : file object
      With 'write' and 'mode'
    stuff :  str or bytes
      Stuff to be written
    encoding : str
      Encoding of text
    """
    binary_file = "b" in getattr(file_obj, "mode", "b")
    string_stuff = isinstance(stuff, basestring)
    binary_stuff = isinstance(stuff, bytes)

    if not PY3:
        file_obj.write(stuff)
    elif binary_file and string_stuff:
        file_obj.write(stuff.encode(encoding))
    elif not binary_file and binary_stuff:
        file_obj.write(stuff.decode(encoding))
    else:
        file_obj.write(stuff)
    file_obj.flush()
    return stuff


def unique_id(length=12):
    """
    Generate a random alphanumeric unique identifier
    using UUID logic.

    Parameters
    ------------
    length : int
      Length of desired identifier

    Returns
    ------------
    unique : str
      Unique alphanumeric identifier
    """
    return uuid.UUID(int=random.getrandbits(128), version=4).hex[:length]


def generate_basis(z, epsilon=1e-12):
    """
    Generate an arbitrary basis (also known as a coordinate frame)
    from a given z-axis vector.

    Parameters
    ------------
    z : (3,) float
      A vector along the positive z-axis.
    epsilon : float
      Numbers smaller than this considered zero.

    Returns
    ---------
    x : (3,) float
      Vector along x axis.
    y : (3,) float
      Vector along y axis.
    z : (3,) float
      Vector along z axis.
    """
    # get a copy of input vector
    z = np.array(z, dtype=np.float64, copy=True)
    # must be a 3D vector
    if z.shape != (3,):
        raise ValueError("z must be (3,) float!")

    z_norm = np.linalg.norm(z)
    if z_norm < epsilon:
        return np.eye(3)

    # normalize vector in-place
    z /= z_norm
    # X as arbitrary perpendicular vector
    x = np.array([-z[1], z[0], 0.0])
    # avoid degenerate case
    x_norm = np.linalg.norm(x)
    if x_norm < epsilon:
        # this means that
        # so a perpendicular X is just X
        x = np.array([-z[2], z[1], 0.0])
        x /= np.linalg.norm(x)
    else:
        # otherwise normalize X in-place
        x /= x_norm
    # get perpendicular Y with cross product
    y = np.cross(z, x)
    # append result values into (3, 3) vector
    result = np.array([x, y, z], dtype=np.float64)

    if _STRICT:
        # run checks to make sure axis are perpendicular
        assert np.abs(np.dot(x, z)) < 1e-8
        assert np.abs(np.dot(y, z)) < 1e-8
        assert np.abs(np.dot(x, y)) < 1e-8
        # all vectors should be unit vector
        assert np.allclose(np.linalg.norm(result, axis=1), 1.0)

    return result


def isclose(a, b, atol):
    """
    A replacement for np.isclose that does fewer checks
    and validation and as a result is roughly 4x faster.

    Note that this is used in tight loops, and as such
    a and b MUST be np.ndarray, not list or "array-like"

    Parameters
    ------------
    a : np.ndarray
      To be compared
    b : np.ndarray
      To be compared
    atol : float
      Acceptable distance between `a` and `b` to be "close"

    Returns
    -----------
    close : np.ndarray, bool
      Per-element closeness
    """
    diff = a - b
    close = np.logical_and(diff > -atol, diff < atol)

    return close


def allclose(a, b, atol=1e-8):
    """
    A replacement for np.allclose that does few checks
    and validation and as a result is faster.

    Parameters
    ------------
    a : np.ndarray
      To be compared
    b : np.ndarray
      To be compared
    atol : float
      Acceptable distance between `a` and `b` to be "close"

    Returns
    -----------
    bool indicating if all elements are within `atol`.
    """
    return float(np.ptp((a - b))) < atol


class FunctionRegistry(Mapping):
    """
    Non-overwritable mapping of string keys to functions.

    This allows external packages to register additional implementations
    of common functionality without risk of breaking implementations provided
    by trimesh.

    See trimesh.voxel.morphology for example usage.
    """

    def __init__(self, **kwargs):
        self._dict = {}
        for k, v in kwargs.items():
            self[k] = v

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise ValueError("key must be a string, got %s" % str(key))
        if key in self:
            raise KeyError("Cannot set new value to existing key %s" % key)
        if not callable(value):
            raise ValueError("Cannot set value which is not callable.")
        self._dict[key] = value

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __contains__(self, key):
        return key in self._dict

    def __call__(self, key, *args, **kwargs):
        return self[key](*args, **kwargs)


class TemporaryDirectory(object):
    """
    Same basic usage as tempfile.TemporaryDirectory
    but functional in Python 2.7+.

    Example
    ---------
    ```
    with trimesh.util.TemporaryDirectory() as path:
       writable = os.path.join(path, 'hi.txt')
    ```
    """

    def __enter__(self):
        self.path = tempfile.mkdtemp()
        return self.path

    def __exit__(self, *args, **kwargs):
        shutil.rmtree(self.path)


def decode_text(text, initial="utf-8"):
    """
    Try to decode byte input as a string.

    Tries initial guess (UTF-8) then if that fails it
    uses chardet to try another guess before failing.

    Parameters
    ------------
    text : bytes
      Data that might be a string
    initial : str
      Initial guess for text encoding.

    Returns
    ------------
    decoded : str
      Data as a string
    """
    # if not bytes just return input
    if not hasattr(text, "decode"):
        return text
    try:
        # initially guess file is UTF-8 or specified encoding
        text = text.decode(initial)
    except UnicodeDecodeError:
        # detect different file encodings
        import chardet

        # try to detect the encoding of the file
        # only look at the first 1000 characters otherwise
        # for big files chardet looks at everything and is slow
        detect = chardet.detect(text[:1000])
        # warn on files that aren't UTF-8
        log.debug("Data not {}! Trying {} (confidence {})".format(initial, detect["encoding"], detect["confidence"]))
        # try to decode again, unwrap in try
        text = text.decode(detect["encoding"], errors="ignore")
    return text


def to_ascii(text):
    """
    Force a string or other to ASCII text ignoring errors.

    Parameters
    -----------
    text : any
      Input to be converted to ASCII string

    Returns
    -----------
    ascii : str
      Input as an ASCII string
    """
    if hasattr(text, "encode"):
        # case for existing strings
        return text.encode("ascii", errors="ignore").decode("ascii")
    elif hasattr(text, "decode"):
        # case for bytes
        return text.decode("ascii", errors="ignore")
    # otherwise just wrap as a string
    return str(text)


def is_ccw(points, return_all=False):
    """
    Check if connected 2D points are counterclockwise.

    Parameters
    -----------
    points : (n, 2) float
      Connected points on a plane
    return_all : bool
      Return polygon area and centroid or just counter-clockwise.

    Returns
    ----------
    ccw : bool
      True if points are counter-clockwise
    area : float
      Only returned if `return_centroid`
    centroid : (2,) float
      Centroid of the polygon.
    """
    points = np.array(points, dtype=np.float64)

    if len(points.shape) != 2 or points.shape[1] != 2:
        raise ValueError("only defined for `(n, 2)` points")

    # the "shoelace formula"
    product = np.subtract(*(points[:-1, [1, 0]] * points[1:]).T)
    # the area of the polygon
    area = product.sum() / 2.0
    # check the sign of the area
    ccw = area < 0.0

    if not return_all:
        return ccw

    # the centroid of the polygon uses the same formula
    centroid = ((points[:-1] + points[1:]) * product.reshape((-1, 1))).sum(axis=0) / (6.0 * area)

    return ccw, area, centroid


def unique_name(start, contains, counts=None):
    """
    Deterministically generate a unique name not
    contained in a dict, set or other grouping with
    `__includes__` defined. Will create names of the
    form "start_10" and increment accordingly.

    Parameters
    -----------
    start : str
      Initial guess for name.
    contains : dict, set, or list
      Bundle of existing names we can *not* use.
    counts : None or dict
      Maps name starts encountered before to increments in
      order to speed up finding a unique name as otherwise
      it potentially has to iterate through all of contains.
      Should map to "how many times has this `start`
      been attempted, i.e. `counts[start]: int`.
      Note that this *will be mutated* in-place by this function!

    Returns
    ---------
    unique : str
      A name that is not contained in `contains`
    """
    # exit early if name is not in bundle
    if start is not None and len(start) > 0 and start not in contains:
        return start

    # start checking with zero index unless found
    if counts is None:
        increment = 0
    else:
        increment = counts.get(start, 0)
    if start is not None and len(start) > 0:
        formatter = start + "_{}"
        # split by our delimiter once
        split = start.rsplit("_", 1)
        if len(split) == 2 and increment == 0:
            try:
                # start incrementing from the existing
                # trailing value
                # if it is not an integer this will fail
                increment = int(split[1])
                # include the first split value
                formatter = split[0] + "_{}"
            except BaseException:
                pass
    else:
        formatter = "geometry_{}"

    # if contains is empty we will only need to check once
    for i in range(increment + 1, 2 + increment + len(contains)):
        check = formatter.format(i)
        if check not in contains:
            if counts is not None:
                counts[start] = i
            return check

    # this should really never happen since we looped
    # through the full length of contains
    raise ValueError("Unable to establish unique name!")
