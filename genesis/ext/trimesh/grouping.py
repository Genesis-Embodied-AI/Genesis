"""
grouping.py
-------------

Functions for grouping values and rows.
"""

import numpy as np

from . import util

from .constants import log, tol

try:
    from scipy.spatial import cKDTree
except BaseException as E:
    # wrapping just ImportError fails in some cases
    # will raise the error when someone tries to use KDtree
    from . import exceptions

    cKDTree = exceptions.ExceptionWrapper(E)


def merge_vertices(mesh, merge_tex=None, merge_norm=None, digits_vertex=None, digits_norm=None, digits_uv=None):
    """
    Removes duplicate vertices, grouped by position and
    optionally texture coordinate and normal.

    Parameters
    -------------
    mesh : Trimesh object
      Mesh to merge vertices on
    merge_tex : bool
      If True textured meshes with UV coordinates will
      have vertices merged regardless of UV coordinates
    merge_norm : bool
      If True, meshes with vertex normals will have
      vertices merged ignoring different normals
    digits_vertex : None or int
      Number of digits to consider for vertex position
    digits_norm : int
      Number of digits to consider for unit normals
    digits_uv : int
      Number of digits to consider for UV coordinates
    """
    # no vertices so exit early
    if len(mesh.vertices) == 0:
        return
    if merge_tex is None:
        merge_tex = False
    if merge_norm is None:
        merge_norm = False
    if digits_norm is None:
        digits_norm = 2
    if digits_uv is None:
        digits_uv = 4
    if digits_vertex is None:
        # use tol.merge if digit precision not passed
        digits_vertex = util.decimal_to_digits(tol.merge)

    # if we have a ton of unreferenced vertices it will
    # make the unique_rows call super slow so cull first
    if hasattr(mesh, "faces") and len(mesh.faces) > 0:
        referenced = np.zeros(len(mesh.vertices), dtype=bool)
        referenced[mesh.faces] = True
    else:
        # this is used for geometry without faces
        referenced = np.ones(len(mesh.vertices), dtype=bool)

    # collect vertex attributes into sequence we can stack
    stacked = [mesh.vertices * (10**digits_vertex)]

    # UV texture visuals require us to update the
    # vertices and normals differently
    if (
        not merge_tex
        and mesh.visual.defined
        and mesh.visual.kind == "texture"
        and mesh.visual.uv is not None
        and len(mesh.visual.uv) == len(mesh.vertices)
    ):
        # get an array with vertices and UV coordinates
        # converted to integers at requested precision
        stacked.append(mesh.visual.uv * (10**digits_uv))

    # check to see if we have vertex normals
    normals = mesh._cache["vertex_normals"]
    if not merge_norm and np.shape(normals) == mesh.vertices.shape:
        stacked.append(normals * (10**digits_norm))

    # stack collected vertex properties and round to integer
    stacked = np.column_stack(stacked).round().astype(np.int64)

    # check unique rows of referenced vertices
    u, i = unique_rows(stacked[referenced], keep_order=True)

    # construct an inverse using the subset
    inverse = np.zeros(len(mesh.vertices), dtype=np.int64)
    inverse[referenced] = i
    # get the vertex mask
    mask = np.nonzero(referenced)[0][u]
    # run the update including normals and UV coordinates
    mesh.update_vertices(mask=mask, inverse=inverse)


def group(values, min_len=0, max_len=np.inf):
    """
    Return the indices of values that are identical

    Parameters
    ----------
    values : (n,) int
      Values to group
    min_len : int
      The shortest group allowed
      All groups will have len >= min_length
    max_len : int
      The longest group allowed
      All groups will have len <= max_length

    Returns
    ----------
    groups : sequence
      Contains indices to form groups
      IE [0,1,0,1] returns [[0,2], [1,3]]
    """
    original = np.asanyarray(values)

    # save the sorted order and then apply it
    order = original.argsort()
    values = original[order]

    # find the indexes which are duplicates
    if values.dtype.kind == "f":
        # for floats in a sorted array, neighbors are not duplicates
        # if the difference between them is greater than approximate zero
        nondupe = np.greater(np.abs(np.diff(values)), tol.zero)
    else:
        # for ints and strings we can check exact non- equality
        # for all other types this will only work if they defined
        # an __eq__
        nondupe = values[1:] != values[:-1]

    dupe_idx = np.append(0, np.nonzero(nondupe)[0] + 1)
    dupe_len = np.diff(np.concatenate((dupe_idx, [len(values)])))
    dupe_ok = np.logical_and(np.greater_equal(dupe_len, min_len), np.less_equal(dupe_len, max_len))
    groups = [order[i : (i + j)] for i, j in zip(dupe_idx[dupe_ok], dupe_len[dupe_ok])]
    return groups


def hashable_rows(data, digits=None):
    """
    We turn our array into integers based on the precision
    given by digits and then put them in a hashable format.

    Parameters
    ---------
    data : (n, m) array
      Input data
    digits : int or None
      How many digits to add to hash if data is floating point
      If None, tol.merge will be used

    Returns
    ---------
    hashable : (n,) array
      Custom data type which can be sorted
      or used as hash keys
    """
    # if there is no data return immediately
    if len(data) == 0:
        return np.array([])

    # get array as integer to precision we care about
    as_int = float_to_int(data, digits=digits)

    # if it is flat integers already, return
    if len(as_int.shape) == 1:
        return as_int

    # if array is 2D and smallish, we can try bitbanging
    # this is significantly faster than the custom dtype
    if len(as_int.shape) == 2 and as_int.shape[1] <= 4:
        # time for some righteous bitbanging
        # can we pack the whole row into a single 64 bit integer
        precision = int(np.floor(64 / as_int.shape[1]))
        # if the max value is less than precision we can do this
        if np.abs(as_int).max() < 2 ** (precision - 1):
            # the resulting package
            hashable = np.zeros(len(as_int), dtype=np.int64)
            # loop through each column and bitwise xor to combine
            # make sure as_int is int64 otherwise bit offset won't work
            for offset, column in enumerate(as_int.astype(np.int64).T):
                # will modify hashable in place
                np.bitwise_xor(hashable, column << (offset * precision), out=hashable)
            return hashable

    # reshape array into magical data type that is weird but hashable
    dtype = np.dtype((np.void, as_int.dtype.itemsize * as_int.shape[1]))
    # make sure result is contiguous and flat
    hashable = np.ascontiguousarray(as_int).view(dtype).reshape((-1,))
    return hashable


def float_to_int(data, digits=None, dtype=np.int32):
    """
    Given a numpy array of float/bool/int, return as integers.

    Parameters
    -------------
    data :  (n, d) float, int, or bool
      Input data
    digits : float or int
      Precision for float conversion
    dtype : numpy.dtype
      What datatype should result be returned as

    Returns
    -------------
    as_int : (n, d) int
      Data as integers
    """
    # convert to any numpy array
    data = np.asanyarray(data)

    # if data is already an integer or boolean we're done
    # if the data is empty we are also done
    if data.dtype.kind in "ib" or data.size == 0:
        return data.astype(dtype)
    elif data.dtype.kind != "f":
        data = data.astype(np.float64)

    # populate digits from kwargs
    if digits is None:
        digits = util.decimal_to_digits(tol.merge)
    elif isinstance(digits, float) or isinstance(digits, np.float64):
        digits = util.decimal_to_digits(digits)
    elif not (isinstance(digits, int) or isinstance(digits, np.integer)):
        log.warning("Digits were passed as %s!", digits.__class__.__name__)
        raise ValueError("Digits must be None, int, or float!")

    # data is float so convert to large integers
    data_max = np.abs(data).max() * 10**digits
    # ignore passed dtype if we have something large
    dtype = [np.int32, np.int64][int(data_max > 2**31)]
    # multiply by requested power of ten
    # then subtract small epsilon to avoid "go either way" rounding
    # then do the rounding and convert to integer
    as_int = np.round((data * 10**digits) - 1e-6).astype(dtype)

    return as_int


def unique_ordered(data, return_index=False, return_inverse=False):
    """
    Returns the same as np.unique, but ordered as per the
    first occurrence of the unique value in data.

    Examples
    ---------
    In [1]: a = [0, 3, 3, 4, 1, 3, 0, 3, 2, 1]

    In [2]: np.unique(a)
    Out[2]: array([0, 1, 2, 3, 4])

    In [3]: trimesh.grouping.unique_ordered(a)
    Out[3]: array([0, 3, 4, 1, 2])
    """
    # uniques are the values, sorted
    # index is the value in the original `data`
    # i.e. `data[index] == unique`
    # inverse is how to re-construct `data` from `unique`
    # i.e. `unique[inverse] == data`
    unique, index, inverse = np.unique(data, return_index=True, return_inverse=True)

    # we want to maintain the original index order
    order = index.argsort()

    if not return_index and not return_inverse:
        return unique[order]

    # collect return values
    # start with the unique values in original order
    result = [unique[order]]
    # the new index values
    if return_index:
        # re-order the index in the original array
        result.append(index[order])
    if return_inverse:
        # create the new inverse from the order of the order
        result.append(order.argsort()[inverse])

    return result


def unique_bincount(values, minlength=0, return_inverse=False, return_counts=False):
    """
    For arrays of integers find unique values using bin counting.
    Roughly 10x faster for correct input than np.unique

    Parameters
    --------------
    values : (n,) int
      Values to find unique members of
    minlength : int
      Maximum value that will occur in values (values.max())
    return_inverse : bool
      If True, return an inverse such that unique[inverse] == values
    return_counts : bool
      If True, also return the number of times each
      unique item appears in values

    Returns
    ------------
    unique : (m,) int
      Unique values in original array
    inverse : (n,) int, optional
      An array such that unique[inverse] == values
      Only returned if return_inverse is True
    counts : (m,) int, optional
      An array holding the counts of each unique item in values
      Only returned if return_counts is True
    """
    values = np.asanyarray(values)
    if len(values.shape) != 1 or values.dtype.kind != "i":
        raise ValueError("input must be 1D integers!")

    try:
        # count the number of occurrences of each value
        counts = np.bincount(values, minlength=minlength)
    except TypeError:
        # casting failed on 32 bit windows
        log.warning("casting failed, falling back!")
        # fall back to numpy unique
        return np.unique(values, return_inverse=return_inverse, return_counts=return_counts)

    # which bins are occupied at all
    # counts are integers so this works
    unique_bin = counts.astype(bool)

    # which values are unique
    # indexes correspond to original values
    unique = np.where(unique_bin)[0]
    ret = (unique,)

    if return_inverse:
        # find the inverse to reconstruct original
        inverse = (np.cumsum(unique_bin) - 1)[values]
        ret += (inverse,)

    if return_counts:
        unique_counts = counts[unique]
        ret += (unique_counts,)

    if len(ret) == 1:
        return ret[0]
    return ret


def merge_runs(data, digits=None):
    """
    Merge duplicate sequential values. This differs from unique_ordered
    in that values can occur in multiple places in the sequence, but
    only consecutive repeats are removed

    Parameters
    -----------
    data: (n,) float or int

    Returns
    --------
    merged: (m,) float or int

    Examples
    ---------
    In [1]: a
    Out[1]:
    array([-1, -1, -1,  0,  0,  1,  1,  2,  0,
            3,  3,  4,  4,  5,  5,  6,  6,  7,
            7,  8,  8,  9,  9,  9])

    In [2]: trimesh.grouping.merge_runs(a)
    Out[2]: array([-1,  0,  1,  2,  0,  3,  4,  5,  6,  7,  8,  9])
    """
    data = np.asanyarray(data)
    mask = np.empty(len(data), dtype=bool)
    mask[0] = True
    mask[1:] = np.abs(data[1:] - data[:-1]) > tol.merge

    return data[mask]


def unique_float(data, return_index=False, return_inverse=False, digits=None):
    """
    Identical to the numpy.unique command, except evaluates floating point
    numbers, using a specified number of digits.

    If digits isn't specified, the library default TOL_MERGE will be used.
    """
    data = np.asanyarray(data)
    as_int = float_to_int(data, digits)
    _junk, unique, inverse = np.unique(as_int, return_index=True, return_inverse=True)

    if (not return_index) and (not return_inverse):
        return data[unique]

    result = [data[unique]]

    if return_index:
        result.append(unique)
    if return_inverse:
        result.append(inverse)
    return tuple(result)


def unique_rows(data, digits=None, keep_order=False):
    """
    Returns indices of unique rows. It will return the
    first occurrence of a row that is duplicated:
    [[1,2], [3,4], [1,2]] will return [0,1]

    Parameters
    ---------
    data : (n, m) array
      Floating point data
    digits : int or None
      How many digits to consider

    Returns
    --------
    unique :  (j,) int
      Index in data which is a unique row
    inverse : (n,) int
      Array to reconstruct original
      Example: data[unique][inverse] == data
    """
    # get rows hashable so we can run unique function on it
    rows = hashable_rows(data, digits=digits)

    # we are throwing away the first value which is the
    # garbage row-hash and only returning index and inverse
    if keep_order:
        # keeps order of original occurrence
        return unique_ordered(rows, return_index=True, return_inverse=True)[1:]
    # returns values sorted by row-hash but since our row-hash
    # were pretty much garbage the sort order isn't meaningful
    return np.unique(rows, return_index=True, return_inverse=True)[1:]


def unique_value_in_row(data, unique=None):
    """
    For a 2D array of integers find the position of a
    value in each row which only occurs once.

    If there are more than one value per row which
    occur once, the last one is returned.

    Parameters
    ----------
    data :   (n, d) int
      Data to check values
    unique : (m,) int
      List of unique values contained in data.
      Generated from np.unique if not passed

    Returns
    ---------
    result : (n, d) bool
      With one or zero True values per row.


    Examples
    -------------------------------------
    In [0]: r = np.array([[-1,  1,  1],
                          [-1,  1, -1],
                          [-1,  1,  1],
                          [-1,  1, -1],
                          [-1,  1, -1]], dtype=np.int8)

    In [1]: unique_value_in_row(r)
    Out[1]:
           array([[ True, False, False],
                  [False,  True, False],
                  [ True, False, False],
                  [False,  True, False],
                  [False,  True, False]], dtype=bool)

    In [2]: unique_value_in_row(r).sum(axis=1)
    Out[2]: array([1, 1, 1, 1, 1])

    In [3]: r[unique_value_in_row(r)]
    Out[3]: array([-1,  1, -1,  1,  1], dtype=int8)
    """
    if unique is None:
        unique = np.unique(data)
    data = np.asanyarray(data)
    result = np.zeros_like(data, dtype=bool, subok=False)
    for value in unique:
        test = np.equal(data, value)
        test_ok = test.sum(axis=1) == 1
        result[test_ok] = test[test_ok]
    return result


def group_rows(data, require_count=None, digits=None):
    """
    Returns index groups of duplicate rows, for example:
    [[1,2], [3,4], [1,2]] will return [[0,2], [1]]


    Note that using require_count allows numpy advanced
    indexing to be used in place of looping and
    checking hashes and is ~10x faster.


    Parameters
    ----------
    data : (n, m) array
      Data to group
    require_count : None or int
      Only return groups of a specified length, eg:
      require_count =  2
      [[1,2], [3,4], [1,2]] will return [[0,2]]
    digits : None or int
    If data is floating point how many decimals
    to consider, or calculated from tol.merge

    Returns
    ----------
    groups : sequence (*,) int
      Indices from in indicating identical rows.
    """

    def group_dict():
        """
        Simple hash table based grouping.
        The loop and appends make this rather slow on
        large arrays but it works on irregular groups.
        """
        observed = dict()
        hashable = hashable_rows(data, digits=digits)
        for index, key in enumerate(hashable):
            key_string = key.tobytes()
            if key_string in observed:
                observed[key_string].append(index)
            else:
                observed[key_string] = [index]
        return list(observed.values())

    def group_slice():
        # create a representation of the rows that can be sorted
        hashable = hashable_rows(data, digits=digits)
        # record the order of the rows so we can get the original indices back
        # later
        order = np.argsort(hashable)
        # but for now, we want our hashes sorted
        hashable = hashable[order]
        # this is checking each neighbour for equality, example:
        # example: hashable = [1, 1, 1]; dupe = [0, 0]
        dupe = hashable[1:] != hashable[:-1]
        # we want the first index of a group, so we can slice from that location
        # example: hashable = [0 1 1]; dupe = [1,0]; dupe_idx = [0,1]
        dupe_idx = np.append(0, np.nonzero(dupe)[0] + 1)
        # if you wanted to use this one function to deal with non- regular groups
        # you could use: np.array_split(dupe_idx)
        # this is roughly 3x slower than using the group_dict method above.
        start_ok = np.diff(np.concatenate((dupe_idx, [len(hashable)]))) == require_count
        groups = np.tile(dupe_idx[start_ok].reshape((-1, 1)), require_count) + np.arange(require_count)
        groups_idx = order[groups]
        if require_count == 1:
            return groups_idx.reshape(-1)
        return groups_idx

    if require_count is None:
        return group_dict()
    else:
        return group_slice()


def boolean_rows(a, b, operation=np.intersect1d):
    """
    Find the rows in two arrays which occur in both rows.

    Parameters
    ---------
    a: (n, d) int
        Array with row vectors
    b: (m, d) int
        Array with row vectors
    operation : function
        Numpy boolean set operation function:
          -np.intersect1d
          -np.setdiff1d

    Returns
    --------
    shared: (p, d) array containing rows in both a and b
    """
    a = np.asanyarray(a, dtype=np.int64)
    b = np.asanyarray(b, dtype=np.int64)

    av = a.view([("", a.dtype)] * a.shape[1]).ravel()
    bv = b.view([("", b.dtype)] * b.shape[1]).ravel()
    shared = operation(av, bv).view(a.dtype).reshape(-1, a.shape[1])

    return shared


def group_vectors(vectors, angle=1e-4, include_negative=False):
    """
    Group vectors based on an angle tolerance, with the option to
    include negative vectors.

    Parameters
    -----------
    vectors : (n,3) float
        Direction vector
    angle : float
        Group vectors closer than this angle in radians
    include_negative : bool
        If True consider the same:
        [0,0,1] and [0,0,-1]

    Returns
    ------------
    new_vectors : (m,3) float
        Direction vector
    groups : (m,) sequence of int
        Indices of source vectors
    """

    vectors = np.asanyarray(vectors, dtype=np.float64)
    angle = float(angle)

    if include_negative:
        vectors = util.vector_hemisphere(vectors)

    spherical = util.vector_to_spherical(vectors)
    angles, groups = group_distance(spherical, angle)
    new_vectors = util.spherical_to_vector(angles)
    return new_vectors, groups


def group_distance(values, distance):
    """
    Find groups of points which have neighbours closer than radius,
    where no two points in a group are farther than distance apart.

    Parameters
    ---------
    points :   (n, d) float
        Points of dimension d
    distance : float
        Max distance between points in a cluster

    Returns
    ----------
    unique : (m, d) float
        Median value of each group
    groups : (m) sequence of int
        Indexes of points that make up a group

    """
    values = np.asanyarray(values, dtype=np.float64)

    consumed = np.zeros(len(values), dtype=bool)
    tree = cKDTree(values)

    # (n, d) set of values that are unique
    unique = []
    # (n) sequence of indices in values
    groups = []

    for index, value in enumerate(values):
        if consumed[index]:
            continue
        group = np.array(tree.query_ball_point(value, distance), dtype=np.int64)
        consumed[group] = True
        unique.append(np.median(values[group], axis=0))
        groups.append(group)
    return np.array(unique), groups


def clusters(points, radius):
    """
    Find clusters of points which have neighbours closer than radius

    Parameters
    ---------
    points : (n, d) float
        Points of dimension d
    radius : float
        Max distance between points in a cluster

    Returns
    ----------
    groups : (m,) sequence of int
        Indices of points in a cluster

    """
    from . import graph

    tree = cKDTree(points)

    # some versions return pairs as a set of tuples
    pairs = tree.query_pairs(r=radius, output_type="ndarray")
    # group connected components
    groups = graph.connected_components(pairs)

    return groups


def blocks(data, min_len=2, max_len=np.inf, wrap=False, digits=None, only_nonzero=False):
    """
    Find the indices in an array of contiguous blocks
    of equal values.

    Parameters
    ------------
    data : (n,) array
      Data to find blocks on
    min_len : int
      The minimum length group to be returned
    max_len : int
      The maximum length group to be retuurned
    wrap : bool
      Combine blocks on both ends of 1D array
    digits : None or int
      If dealing with floats how many digits to consider
    only_nonzero : bool
      Only return blocks of non- zero values

    Returns
    ---------
    blocks : (m) sequence of (*,) int
      Indices referencing data
    """
    data = float_to_int(data, digits=digits)

    # keep an integer range around so we can slice
    arange = np.arange(len(data))
    arange.flags["WRITEABLE"] = False

    nonzero = arange[1:][data[1:] != data[:-1]]
    infl = np.zeros(len(nonzero) + 2, dtype=int)
    infl[-1] = len(data)
    infl[1:-1] = nonzero

    # the length of each chunk
    infl_len = infl[1:] - infl[:-1]

    # check the length of each group
    infl_ok = np.logical_and(infl_len >= min_len, infl_len <= max_len)

    if only_nonzero:
        # check to make sure the values of each contiguous block
        # are True by checking the first value of each block
        infl_ok = np.logical_and(infl_ok, data[infl[:-1]])

    # inflate start/end indexes into full ranges of values
    blocks = [arange[infl[i] : infl[i + 1]] for i, ok in enumerate(infl_ok) if ok]

    if wrap:
        # wrap only matters if first and last points are the same
        if data[0] != data[-1]:
            return blocks
        # if we are only grouping nonzero things and
        # the first and last point are zero we can exit
        if only_nonzero and not bool(data[0]):
            return blocks

        # so now first point equals last point, so the cases are:
        # - first and last point are in a block: combine two blocks
        # - first OR last point are in block: add other point to block
        # - neither are in a block: check if combined is eligible block

        # first point is in a block
        first = len(blocks) > 0 and blocks[0][0] == 0
        # last point is in a block
        last = len(blocks) > 0 and blocks[-1][-1] == (len(data) - 1)

        # CASE: first and last point are BOTH in block: combine blocks
        if first and last:
            blocks[0] = np.append(blocks[-1], blocks[0])
            blocks.pop()
        else:
            # combined length
            combined = infl_len[0] + infl_len[-1]
            # exit if lengths aren't OK
            if combined < min_len or combined > max_len:
                return blocks
            # new block combines both ends
            new_block = np.append(np.arange(infl[-2], infl[-1]), np.arange(infl[0], infl[1]))
            # we are in a first OR last situation now
            if first:
                # first was already in a block so replace it with combined
                blocks[0] = new_block
            elif last:
                # last was already in a block so replace with superset
                blocks[-1] = new_block
            else:
                # both are false
                # combined length generated new block
                blocks.append(new_block)

    return blocks


def group_min(groups, data):
    """
    Given a list of groups find the minimum element of data
    within each group

    Parameters
    -----------
    groups : (n,) sequence of (q,) int
        Indexes of each group corresponding to each element in data
    data : (m,)
        The data that groups indexes reference

    Returns
    -----------
    minimums : (n,)
        Minimum value of data per group

    """
    # sort with major key groups, minor key data
    order = np.lexsort((data, groups))
    groups = groups[order]  # this is only needed if groups is unsorted
    data = data[order]
    # construct an index which marks borders between groups
    index = np.empty(len(groups), "bool")
    index[0] = True
    index[1:] = groups[1:] != groups[:-1]
    return data[index]
