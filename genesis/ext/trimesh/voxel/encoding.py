"""OO interfaces to encodings for ND arrays which caching."""

import numpy as np

import abc
from ..util import ABC, log

from . import runlength
from .. import caching

try:
    from scipy import sparse as sp
except BaseException as E:
    from ..exceptions import ExceptionWrapper

    sp = ExceptionWrapper(E)


def _empty_stripped(shape):
    num_dims = len(shape)
    encoding = DenseEncoding(np.empty(shape=(0,) * num_dims, dtype=bool))
    padding = np.zeros(shape=(num_dims, 2), dtype=int)
    padding[:, 1] = shape
    return encoding, padding


class Encoding(ABC):
    """
    Base class for objects that implement a specific subset of of ndarray ops.

    This presents a unified interface for various different ways of encoding
    conceptually dense arrays and to interoperate between them.

    Example implementations are ND sparse arrays, run length encoded arrays
    and dense encodings (wrappers around np.ndarrays).
    """

    def __init__(self, data):
        self._data = data
        self._cache = caching.Cache(id_function=self._data.__hash__)

    @abc.abstractproperty
    def dtype(self):
        pass

    @abc.abstractproperty
    def shape(self):
        pass

    @abc.abstractproperty
    def sum(self):
        pass

    @abc.abstractproperty
    def size(self):
        pass

    @abc.abstractproperty
    def sparse_indices(self):
        pass

    @abc.abstractproperty
    def sparse_values(self):
        pass

    @abc.abstractproperty
    def dense(self):
        pass

    @abc.abstractmethod
    def gather_nd(self, indices):
        pass

    @abc.abstractmethod
    def mask(self, mask):
        pass

    @abc.abstractmethod
    def get_value(self, index):
        pass

    @abc.abstractmethod
    def copy(self):
        pass

    @property
    def is_empty(self):
        return self.sparse_indices[self.sparse_values != 0].size == 0

    @caching.cache_decorator
    def stripped(self):
        """
        Get encoding with all zeros stripped from the start and end
        of each axis.

        Returns
        ------------
        encoding: ?
        padding : (n, 2) int
          Padding at the start and end that was stripped
        """
        if self.is_empty:
            return _empty_stripped(self.shape)
        dense = self.dense
        shape = dense.shape
        ndims = len(shape)
        padding = []
        slices = []
        for dim, size in enumerate(shape):
            axis = tuple(range(dim)) + tuple(range(dim + 1, ndims))
            filled = np.any(dense, axis=axis)
            (indices,) = np.nonzero(filled)
            lower = indices.min()
            upper = indices.max() + 1
            padding.append([lower, size - upper])
            slices.append(slice(lower, upper))
        return DenseEncoding(dense[tuple(slices)]), np.array(padding, int)

    def _flip(self, axes):
        return FlippedEncoding(self, axes)

    def crc(self):
        log.warning(
            "`geometry.crc()` is deprecated and will "
            + "be removed in October 2023: replace "
            + "with `geometry.__hash__()` or `hash(geometry)`"
        )
        return self.__hash__()

    def hash(self):
        log.warning(
            "`geometry.hash()` is deprecated and will "
            + "be removed in October 2023: replace "
            + "with `geometry.__hash__()` or `hash(geometry)`"
        )
        return self.__hash__()

    def __hash__(self):
        """
        Get the hash of the current transformation matrix.

        Returns
        ------------
        hash : str
          Hash of transformation matrix
        """
        return self._data.__hash__()

    @property
    def ndims(self):
        return len(self.shape)

    def reshape(self, shape):
        return self.flat if len(shape) == 1 else ShapedEncoding(self, shape)

    @property
    def flat(self):
        return FlattenedEncoding(self)

    def flip(self, axis=0):
        return _flipped(self, axis)

    @property
    def sparse_components(self):
        return self.sparse_indices, self.sparse_values

    @property
    def data(self):
        return self._data

    def run_length_data(self, dtype=np.int64):
        if self.ndims != 1:
            raise ValueError("`run_length_data` only valid for flat encodings")
        return runlength.dense_to_rle(self.dense, dtype=dtype)

    def binary_run_length_data(self, dtype=np.int64):
        if self.ndims != 1:
            raise ValueError("`run_length_data` only valid for flat encodings")
        return runlength.dense_to_brle(self.dense, dtype=dtype)

    def transpose(self, perm):
        return _transposed(self, perm)

    def _transpose(self, perm):
        return TransposedEncoding(self, perm)

    @property
    def mutable(self):
        return self._data.mutable

    @mutable.setter
    def mutable(self, value):
        self._data.mutable = value


class DenseEncoding(Encoding):
    """Simple `Encoding` implementation based on a numpy ndarray."""

    def __init__(self, data):
        if not isinstance(data, caching.TrackedArray):
            if not isinstance(data, np.ndarray):
                raise ValueError("DenseEncoding data must be a numpy array")
            data = caching.tracked_array(data)
        super(DenseEncoding, self).__init__(data=data)

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def shape(self):
        return self._data.shape

    @caching.cache_decorator
    def sum(self):
        return self._data.sum()

    @caching.cache_decorator
    def is_empty(self):
        return not np.any(self._data)

    @property
    def size(self):
        return self._data.size

    @property
    def sparse_components(self):
        indices = self.sparse_indices
        values = self.gather(indices)
        return indices, values

    @caching.cache_decorator
    def sparse_indices(self):
        return np.column_stack(np.where(self._data))

    @caching.cache_decorator
    def sparse_values(self):
        return self.sparse_components[1]

    def _flip(self, axes):
        dense = self.dense
        for a in axes:
            dense = np.flip(dense, a)
        return DenseEncoding(dense)

    @property
    def dense(self):
        return self._data

    def gather(self, indices):
        return self._data[indices]

    def gather_nd(self, indices):
        return self._data[tuple(indices.T)]

    def mask(self, mask):
        return self._data[mask if isinstance(mask, np.ndarray) else mask.dense]

    def get_value(self, index):
        return self._data[tuple(index)]

    def reshape(self, shape):
        return DenseEncoding(self._data.reshape(shape))

    def _transpose(self, perm):
        return DenseEncoding(self._data.transpose(perm))

    @property
    def flat(self):
        return DenseEncoding(self._data.reshape((-1,)))

    def copy(self):
        return DenseEncoding(self._data.copy())


class SparseEncoding(Encoding):
    """
    `Encoding` implementation based on an ND sparse implementation.

    Since the scipy.sparse implementations are for 2D arrays only, this
    implementation uses a single-column CSC matrix with index
    raveling/unraveling.
    """

    def __init__(self, indices, values, shape=None):
        """
        Parameters
        ------------
        indices: (m, n)-sized int array of indices
        values: (m, n)-sized dtype array of values at the specified indices
        shape: (n,) iterable of integers. If None, the maximum value of indices
            + 1 is used.
        """
        data = caching.DataStore()
        super(SparseEncoding, self).__init__(data)
        data["indices"] = indices
        data["values"] = values
        indices = data["indices"]
        if len(indices.shape) != 2:
            raise ValueError("indices must be 2D, got shaped %s" % str(indices.shape))
        if data["values"].shape != (indices.shape[0],):
            raise ValueError("values and indices shapes inconsistent: %s and %s" % (data["values"], data["indices"]))
        if shape is None:
            self._shape = tuple(data["indices"].max(axis=0) + 1)
        else:
            self._shape = tuple(shape)
            if not np.all(indices < self._shape):
                raise ValueError("all indices must be less than shape")
        if not np.all(indices >= 0):
            raise ValueError("all indices must be non-negative")

    @staticmethod
    def from_dense(dense_data):
        sparse_indices = np.where(dense_data)
        values = dense_data[sparse_indices]
        return SparseEncoding(np.stack(sparse_indices, axis=-1), values, shape=dense_data.shape)

    def copy(self):
        return SparseEncoding(indices=self.sparse_indices.copy(), values=self.sparse_values.copy(), shape=self.shape)

    @property
    def sparse_indices(self):
        return self._data["indices"]

    @property
    def sparse_values(self):
        return self._data["values"]

    @property
    def dtype(self):
        return self.sparse_values.dtype

    @caching.cache_decorator
    def sum(self):
        return self.sparse_values.sum()

    @property
    def ndims(self):
        return self.sparse_indices.shape[-1]

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def sparse_components(self):
        return self.sparse_indices, self.sparse_values

    @caching.cache_decorator
    def dense(self):
        sparse = self._csc
        # sparse.todense gives an `np.matrix` which cannot be reshaped
        dense = np.empty(shape=sparse.shape, dtype=sparse.dtype)
        sparse.todense(out=dense)
        return np.reshape(dense, self.shape)

    @caching.cache_decorator
    def _csc(self):
        values = self.sparse_values
        indices = self._flat_indices(self.sparse_indices)
        indptr = [0, len(indices)]
        return sp.csc_matrix((values, indices, indptr), shape=(self.size, 1))

    def _flat_indices(self, indices):
        assert indices.shape[1] == 3 and len(indices.shape) == 2
        return np.ravel_multi_index(indices.T, self.shape)

    def _shaped_indices(self, flat_indices):
        return np.column_stack(np.unravel_index(flat_indices, self.shape))

    def gather_nd(self, indices):
        mat = self._csc[self._flat_indices(indices)].todense()
        # mat is a np matrix, which stays rank 2 after squeeze
        # np.asarray changes this to a standard rank 2 array.
        return np.asarray(mat).squeeze(axis=-1)

    def mask(self, mask):
        i, _ = np.where(self._csc[mask.reshape((-1,))])
        return self._shaped_indices(i)

    def get_value(self, index):
        return self._gather_nd(np.expand_dims(index, axis=0))[0]

    @caching.cache_decorator
    def stripped(self):
        """
        Get encoding with all zeros stripped from the start/end of each axis.

        Returns:
            encoding: SparseEncoding with same values but indices shifted down
                by padding[:, 0]
            padding: (n, 2) array of ints denoting padding at the start/end
                that was stripped
        """
        if self.is_empty:
            return _empty_stripped(self.shape)
        indices = self.sparse_indices
        pad_left = np.min(indices, axis=0)
        pad_right = np.max(indices, axis=0)
        pad_right *= -1
        pad_right += self.shape
        padding = np.column_stack((pad_left, pad_right))
        return SparseEncoding(indices - pad_left, self.sparse_values), padding


def SparseBinaryEncoding(indices, shape=None):
    """
    Convenient factory constructor for SparseEncodings with values all ones.

    Parameters
    ------------
    indices: (m, n) sparse indices into conceptual rank-n array
    shape: length n iterable or None. If None, maximum of indices along first
        axis + 1 is used

    Returns
    ------------
    rank n bool `SparseEncoding` with True values at each index.
    """
    return SparseEncoding(indices, np.ones(shape=(indices.shape[0],), dtype=bool), shape)


class RunLengthEncoding(Encoding):
    """1D run length encoding.

    See `trimesh.voxel.runlength` documentation for implementation details.
    """

    def __init__(self, data, dtype=None):
        """
        Parameters
        ------------
        data: run length encoded data.
        dtype: dtype of encoded data. Each second value of data is cast will be
            cast to this dtype if provided.
        """
        super(RunLengthEncoding, self).__init__(data=caching.tracked_array(data))
        if dtype is None:
            dtype = self._data.dtype
        if len(self._data.shape) != 1:
            raise ValueError("data must be 1D numpy array")
        self._dtype = dtype

    @caching.cache_decorator
    def is_empty(self):
        return not np.any(np.logical_and(self._data[::2], self._data[1::2]))

    @property
    def ndims(self):
        return 1

    @property
    def shape(self):
        return (self.size,)

    @property
    def dtype(self):
        return self._dtype

    def crc(self):
        log.warning(
            "`geometry.crc()` is deprecated and will "
            + "be removed in October 2023: replace "
            + "with `geometry.__hash__()` or `hash(geometry)`"
        )
        return self.__hash__()

    def hash(self):
        log.warning(
            "`geometry.hash()` is deprecated and will "
            + "be removed in October 2023: replace "
            + "with `geometry.__hash__()` or `hash(geometry)`"
        )
        return self.__hash__()

    def __hash__(self):
        """
        Get the hash of the current transformation matrix.

        Returns
        ------------
        hash : str
          Hash of transformation matrix
        """
        return self._data.__hash__()

    @staticmethod
    def from_dense(dense_data, dtype=np.int64, encoding_dtype=np.int64):
        return RunLengthEncoding(runlength.dense_to_rle(dense_data, dtype=encoding_dtype), dtype=dtype)

    @staticmethod
    def from_rle(rle_data, dtype=None):
        if dtype != rle_data.dtype:
            rle_data = runlength.rle_to_rle(rle_data, dtype=dtype)
        return RunLengthEncoding(rle_data)

    @staticmethod
    def from_brle(brle_data, dtype=None):
        return RunLengthEncoding(runlength.brle_to_rle(brle_data, dtype=dtype))

    @caching.cache_decorator
    def stripped(self):
        if self.is_empty:
            return _empty_stripped(self.shape)
        data, padding = runlength.rle_strip(self._data)
        if padding == (0, 0):
            encoding = self
        else:
            encoding = RunLengthEncoding(data, dtype=self._dtype)
        padding = np.expand_dims(padding, axis=0)
        return encoding, padding

    @caching.cache_decorator
    def sum(self):
        return (self._data[::2] * self._data[1::2]).sum()

    @caching.cache_decorator
    def size(self):
        return runlength.rle_length(self._data)

    def _flip(self, axes):
        if axes != (0,):
            raise ValueError("encoding is 1D - cannot flip on axis %s" % str(axes))
        return RunLengthEncoding(runlength.rle_reverse(self._data))

    @caching.cache_decorator
    def sparse_components(self):
        return runlength.rle_to_sparse(self._data)

    @caching.cache_decorator
    def sparse_indices(self):
        return self.sparse_components[0]

    @caching.cache_decorator
    def sparse_values(self):
        return self.sparse_components[1]

    @caching.cache_decorator
    def dense(self):
        return runlength.rle_to_dense(self._data, dtype=self._dtype)

    def gather(self, indices):
        return runlength.rle_gather_1d(self._data, indices, dtype=self._dtype)

    def gather_nd(self, indices):
        indices = np.squeeze(indices, axis=-1)
        return self.gather(indices)

    def sorted_gather(self, ordered_indices):
        return np.array(tuple(runlength.sorted_rle_gather_1d(self._data, ordered_indices)), dtype=self._dtype)

    def mask(self, mask):
        return np.array(tuple(runlength.rle_mask(self._data, mask)), dtype=self._dtype)

    def get_value(self, index):
        for value in self.sorted_gather((index,)):
            return np.asanyarray(value, dtype=self._dtype)

    def copy(self):
        return RunLengthEncoding(self._data.copy(), dtype=self.dtype)

    def run_length_data(self, dtype=np.int64):
        return runlength.rle_to_rle(self._data, dtype=dtype)

    def binary_run_length_data(self, dtype=np.int64):
        return runlength.rle_to_brle(self._data, dtype=dtype)


class BinaryRunLengthEncoding(RunLengthEncoding):
    """1D binary run length encoding.

    See `trimesh.voxel.runlength` documentation for implementation details.
    """

    def __init__(self, data):
        """
        Parameters
        ------------
        data: binary run length encoded data.
        """
        super(BinaryRunLengthEncoding, self).__init__(data=data, dtype=bool)

    @caching.cache_decorator
    def is_empty(self):
        return not np.any(self._data[1::2])

    @staticmethod
    def from_dense(dense_data, encoding_dtype=np.int64):
        return BinaryRunLengthEncoding(runlength.dense_to_brle(dense_data, dtype=encoding_dtype))

    @staticmethod
    def from_rle(rle_data, dtype=None):
        return BinaryRunLengthEncoding(runlength.rle_to_brle(rle_data, dtype=dtype))

    @staticmethod
    def from_brle(brle_data, dtype=None):
        if dtype != brle_data.dtype:
            brle_data = runlength.brle_to_brle(brle_data, dtype=dtype)
        return BinaryRunLengthEncoding(brle_data)

    @caching.cache_decorator
    def stripped(self):
        if self.is_empty:
            return _empty_stripped(self.shape)
        data, padding = runlength.rle_strip(self._data)
        if padding == (0, 0):
            encoding = self
        else:
            encoding = BinaryRunLengthEncoding(data)
        padding = np.expand_dims(padding, axis=0)
        return encoding, padding

    @caching.cache_decorator
    def sum(self):
        return self._data[1::2].sum()

    @caching.cache_decorator
    def size(self):
        return runlength.brle_length(self._data)

    def _flip(self, axes):
        if axes != (0,):
            raise ValueError("encoding is 1D - cannot flip on axis %s" % str(axes))
        return BinaryRunLengthEncoding(runlength.brle_reverse(self._data))

    @property
    def sparse_components(self):
        return self.sparse_indices, self.sparse_values

    @caching.cache_decorator
    def sparse_values(self):
        return np.ones(shape=(self.sum,), dtype=bool)

    @caching.cache_decorator
    def sparse_indices(self):
        return runlength.brle_to_sparse(self._data)

    @caching.cache_decorator
    def dense(self):
        return runlength.brle_to_dense(self._data)

    def gather(self, indices):
        return runlength.brle_gather_1d(self._data, indices)

    def gather_nd(self, indices):
        indices = np.squeeze(indices)
        return self.gather(indices)

    def sorted_gather(self, ordered_indices):
        gen = runlength.sorted_brle_gather_1d(self._data, ordered_indices)
        return np.array(tuple(gen), dtype=bool)

    def mask(self, mask):
        gen = runlength.brle_mask(self._data, mask)
        return np.array(tuple(gen), dtype=bool)

    def copy(self):
        return BinaryRunLengthEncoding(self._data.copy())

    def run_length_data(self, dtype=np.int64):
        return runlength.brle_to_rle(self._data, dtype=dtype)

    def binary_run_length_data(self, dtype=np.int64):
        return runlength.brle_to_brle(self._data, dtype=dtype)


class LazyIndexMap(Encoding):
    """
    Abstract class for implementing lazy index mapping operations.

    Implementations include transpose, flatten/reshaping and flipping

    Derived classes must implement:
        * _to_base_indices(indices)
        * _from_base_indices(base_indices)
        * shape
        * dense
        * mask(mask)
    """

    @abc.abstractmethod
    def _to_base_indices(self, indices):
        pass

    @abc.abstractmethod
    def _from_base_indices(self, base_indices):
        pass

    @property
    def is_empty(self):
        return self._data.is_empty

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def sum(self):
        return self._data.sum

    @property
    def size(self):
        return self._data.size

    @property
    def sparse_indices(self):
        return self._from_base_indices(self._data.sparse_indices)

    @property
    def sparse_values(self):
        return self._data.sparse_values

    def gather_nd(self, indices):
        return self._data.gather_nd(self._to_base_indices(indices))

    def get_value(self, index):
        return self._data[tuple(self._to_base_indices(index))]


class FlattenedEncoding(LazyIndexMap):
    """
    Lazily flattened encoding.

    Dense equivalent is np.reshape(data, (-1,)) (np.flatten creates a copy).
    """

    def _to_base_indices(self, indices):
        return np.column_stack(np.unravel_index(indices, self._data.shape))

    def _from_base_indices(self, base_indices):
        return np.expand_dims(np.ravel_multi_index(base_indices.T, self._data.shape), axis=-1)

    @property
    def shape(self):
        return (self.size,)

    @property
    def dense(self):
        return self._data.dense.reshape((-1,))

    def mask(self, mask):
        return self._data.mask(mask.reshape(self._data.shape))

    @property
    def flat(self):
        return self

    def copy(self):
        return FlattenedEncoding(self._data.copy())


class ShapedEncoding(LazyIndexMap):
    """
    Lazily reshaped encoding.

    Numpy equivalent is `np.reshape`
    """

    def __init__(self, encoding, shape):
        if isinstance(encoding, Encoding):
            if encoding.ndims != 1:
                encoding = encoding.flat
        else:
            raise ValueError("encoding must be an Encoding")
        super(ShapedEncoding, self).__init__(data=encoding)
        self._shape = tuple(shape)
        nn = self._shape.count(-1)
        size = np.prod(self._shape)
        if nn == 1:
            size = np.abs(size)
            if self._data.size % size != 0:
                raise ValueError(
                    "cannot reshape encoding of size %d into shape %s" % (self._data.size, str(self._shape))
                )
            rem = self._data.size // size
            self._shape = tuple(rem if s == -1 else s for s in self._shape)
        elif nn > 2:
            raise ValueError("shape cannot have more than one -1 value")
        elif np.prod(self._shape) != self._data.size:
            raise ValueError("cannot reshape encoding of size %d into shape %s" % (self._data.size, str(self._shape)))

    def _from_base_indices(self, base_indices):
        return np.column_stack(np.unravel_index(base_indices, self.shape))

    def _to_base_indices(self, indices):
        return np.expand_dims(np.ravel_multi_index(indices.T, self.shape), axis=-1)

    @property
    def flat(self):
        return self._data

    @property
    def shape(self):
        return self._shape

    @property
    def dense(self):
        return self._data.dense.reshape(self.shape)

    def mask(self, mask):
        return self._data.mask(mask.flat)

    def copy(self):
        return ShapedEncoding(encoding=self._data.copy(), shape=self.shape)


class TransposedEncoding(LazyIndexMap):
    """
    Lazily transposed encoding

    Dense equivalent is `np.transpose`
    """

    def __init__(self, base_encoding, perm):
        if not isinstance(base_encoding, Encoding):
            raise ValueError("base_encoding must be an Encoding, got %s" % str(base_encoding))
        if len(base_encoding.shape) != len(perm):
            raise ValueError(
                "base_encoding has %d ndims - cannot transpose with perm %s" % (base_encoding.ndims, str(perm))
            )
        super(TransposedEncoding, self).__init__(base_encoding)
        perm = np.array(perm, dtype=np.int64)
        if not all(i in perm for i in range(base_encoding.ndims)):
            raise ValueError("perm %s is not a valid permutation" % str(perm))
        inv_perm = np.empty_like(perm)
        inv_perm[perm] = np.arange(base_encoding.ndims)
        self._perm = perm
        self._inv_perm = inv_perm

    def transpose(self, perm):
        return _transposed(self._data, [self._perm[p] for p in perm])

    def _transpose(self, perm):
        raise RuntimeError("Should not be here")

    @property
    def perm(self):
        return self._perm

    @property
    def shape(self):
        shape = self._data.shape
        return tuple(shape[p] for p in self._perm)

    def _to_base_indices(self, indices):
        return np.take(indices, self._perm, axis=-1)

    def _from_base_indices(self, base_indices):
        try:
            return np.take(base_indices, self._inv_perm, axis=-1)
        except TypeError:
            # windows sometimes tries to use wrong dtypes
            return np.take(base_indices.astype(np.int64), self._inv_perm.astype(np.int64), axis=-1)

    @property
    def dense(self):
        return self._data.dense.transpose(self._perm)

    def gather(self, indices):
        return self._data.gather(self._base_indices(indices))

    def mask(self, mask):
        return self._data.mask(mask.transpose(self._inv_perm)).transpose(self._perm)

    def get_value(self, index):
        return self._data[tuple(self._base_indices(index))]

    @property
    def data(self):
        return self._data

    def copy(self):
        return TransposedEncoding(base_encoding=self._data.copy(), perm=self._perm)


class FlippedEncoding(LazyIndexMap):
    """
    Encoding with entries flipped along one or more axes.

    Dense equivalent is `np.flip`
    """

    def __init__(self, encoding, axes):
        ndims = encoding.ndims
        if isinstance(axes, np.ndarray) and axes.size == 1:
            axes = (axes.item(),)
        elif isinstance(axes, int):
            axes = (axes,)
        axes = tuple(a + ndims if a < 0 else a for a in axes)
        self._axes = tuple(sorted(axes))
        if len(set(self._axes)) != len(self._axes):
            raise ValueError("Axes cannot contain duplicates, got %s" % str(self._axes))
        super(FlippedEncoding, self).__init__(encoding)
        if not all(0 <= a < self._data.ndims for a in axes):
            raise ValueError("Invalid axes %s for %d-d encoding" % (str(axes), self._data.ndims))

    def _to_base_indices(self, indices):
        indices = indices.copy()
        shape = self.shape
        for a in self._axes:
            indices[:, a] *= -1
            indices[:, a] += shape
        return indices

    def _from_base_indices(self, base_indices):
        return self._to_base_indices(base_indices)

    @property
    def shape(self):
        return self._data.shape

    @property
    def dense(self):
        dense = self._data.dense
        for a in self._axes:
            dense = np.flip(dense, a)
        return dense

    def mask(self, mask):
        if not isinstance(mask, Encoding):
            mask = DenseEncoding(mask)
        mask = mask.flip(self._axes)
        return self._data.mask(mask).flip(self._axes)

    def copy(self):
        return FlippedEncoding(self._data.copy(), self._axes)

    def flip(self, axis=0):
        if isinstance(axis, np.ndarray):
            if axis.size == 1:
                axis = (axis.item(),)
            else:
                axis = tuple(axis)
        elif isinstance(axis, int):
            axes = (axis,)
        else:
            axes = tuple(axis)
        return _flipped(self, self._axes + axes)

    def _flip(self, axes):
        raise RuntimeError("Should not be here")


def _flipped(encoding, axes):
    if not hasattr(axes, "__iter__"):
        axes = (axes,)
    unique_ax = set()
    ndims = encoding.ndims
    axes = tuple(a + ndims if a < 0 else a for a in axes)
    for a in axes:
        if a in unique_ax:
            unique_ax.remove(a)
        else:
            unique_ax.add(a)
    if len(unique_ax) == 0:
        return encoding
    else:
        return encoding._flip(tuple(sorted(unique_ax)))


def _transposed(encoding, perm):
    ndims = encoding.ndims
    perm = tuple(p + ndims if p < 0 else p for p in perm)
    if np.all(np.arange(ndims) == perm):
        return encoding
    else:
        return encoding._transpose(perm)
