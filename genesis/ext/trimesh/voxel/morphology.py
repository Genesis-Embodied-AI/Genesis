"""Basic morphology operations that create new encodings."""

import numpy as np

from . import encoding as enc
from . import ops

from ..constants import log_time
from .. import util


try:
    from scipy import ndimage
except BaseException as E:
    # scipy is a soft dependency
    from ..exceptions import ExceptionWrapper

    ndimage = ExceptionWrapper(E)


def _dense(encoding, rank=None):
    if isinstance(encoding, np.ndarray):
        dense = encoding
    elif isinstance(encoding, enc.Encoding):
        dense = encoding.dense
    else:
        raise ValueError("encoding must be np.ndarray or Encoding, got %s" % str(encoding))
    if rank:
        _assert_rank(dense, rank)
    return dense


def _sparse_indices(encoding, rank=None):
    if isinstance(encoding, np.ndarray):
        sparse_indices = encoding
    elif isinstance(encoding, enc.Encoding):
        sparse_indices = encoding.sparse_indices
    else:
        raise ValueError("encoding must be np.ndarray or Encoding, got %s" % str(encoding))

    _assert_sparse_rank(sparse_indices, 3)
    return sparse_indices


def _assert_rank(value, rank):
    if len(value.shape) != rank:
        raise ValueError("Expected rank %d, got shape %s" % (rank, str(value.shape)))


def _assert_sparse_rank(value, rank=None):
    if len(value.shape) != 2:
        raise ValueError("sparse_indices must be rank 2, got shape %s" % str(value.shape))
    if rank is not None:
        if value.shape[-1] != rank:
            raise ValueError("sparse_indices.shape[1] must be %d, got %d" % (rank, value.shape[-1]))


@log_time
def fill_base(encoding):
    """
    Given a sparse surface voxelization, fill in between columns.

    Parameters
    --------------
    encoding: Encoding object or sparse array with shape (?, 3)

    Returns
    --------------
    A new filled encoding object.
    """
    return enc.SparseBinaryEncoding(ops.fill_base(_sparse_indices(encoding, rank=3)))


@log_time
def fill_orthographic(encoding):
    """
    Fill the given encoding by orthographic projection method.

    Any voxel in the dense representation with no free ray along the x, y, z
    axes in each direction is assigned filled. This is likely faster than fill
    holes, and is more stable with regards to small holes.

    Parameters
    --------------
    encoding: Encoding object or dense rank-3 array.

    Returns
    --------------
    A new filled encoding object.
    """
    return enc.DenseEncoding(ops.fill_orthographic(_dense(encoding, rank=3)))


@log_time
def fill_holes(encoding, **kwargs):
    """
    Encoding wrapper around scipy.ndimage.morphology.binary_fill_holes.

    https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.ndimage.morphology.binary_fill_holes.html#scipy.ndimage.morphology.binary_fill_holes

    Parameters
    --------------
    encoding: Encoding object or dense rank-3 array.
    **kwargs: see scipy.ndimage.morphology.binary_fill_holes.

    Returns
    --------------
    A new filled in encoding object.
    """
    return enc.DenseEncoding(ndimage.binary_fill_holes(_dense(encoding, rank=3), **kwargs))


fillers = util.FunctionRegistry(
    base=fill_base,
    orthographic=fill_orthographic,
    holes=fill_holes,
)


def fill(encoding, method="base", **kwargs):
    """
    Fill the given encoding using the specified implementation.

    See `fillers` for available implementations or to add your own, e.g. via
    `fillers['custom_key'] = custom_fn`.

    `custom_fn` should have signature `(encoding, **kwargs) -> filled_encoding`
    and should not modify encoding.

    Parameters
    --------------
    encoding: Encoding object (left unchanged).
    method: method present in `fillers`.
    **kwargs: additional kwargs passed to the specified implementation.

    Returns
    --------------
    A new filled Encoding object.
    """
    return fillers(method, encoding=encoding, **kwargs)


def binary_dilation(encoding, **kwargs):
    """
    Encoding wrapper around scipy.ndimage.morphology.binary_dilation.

    https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.ndimage.morphology.binary_dilation.html#scipy.ndimage.morphology.binary_dilation
    """
    return enc.DenseEncoding(ndimage.binary_dilation(_dense(encoding, rank=3), **kwargs))


def binary_closing(encoding, **kwargs):
    """
    Encoding wrapper around scipy.ndimage.morphology.binary_closing.

    https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.ndimage.morphology.binary_closing.html#scipy.ndimage.morphology.binary_closing
    """
    return enc.DenseEncoding(ndimage.binary_closing(_dense(encoding, rank=3), **kwargs))


def surface(encoding, structure=None):
    """
    Get elements on the surface of encoding.

    A surface element is any one in encoding that is adjacent to an empty
    voxel.

    Parameters
    --------------
    encoding: Encoding or dense rank-3 array
    structure: adjacency structure. If None, square connectivity is used.

    Returns
    --------------
    new surface Encoding.
    """
    dense = _dense(encoding, rank=3)
    # padding/unpadding resolves issues with occupied voxels on the boundary
    dense = np.pad(dense, np.ones((3, 2), dtype=int), mode="constant")
    empty = np.logical_not(dense)
    dilated = ndimage.binary_dilation(empty, structure=structure)
    surface = np.logical_and(dense, dilated)[1:-1, 1:-1, 1:-1]
    return enc.DenseEncoding(surface)
