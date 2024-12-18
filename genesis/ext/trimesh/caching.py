"""
caching.py
-----------

Functions and classes that help with tracking changes
in `numpy.ndarray` and clearing cached values based
on those changes.

You should really `pip install xxhash`:

```
In [23]: %timeit int(blake2b(d).hexdigest(), 16)
102 us +/- 684 ns per loop

In [24]: %timeit int(sha256(d).hexdigest(), 16)
142 us +/- 3.73 us

In [25]: %timeit xxh3_64_intdigest(d)
3.37 us +/- 116 ns per loop
```
"""

import os
import time
import warnings
import numpy as np

from functools import wraps
from .constants import log
from .util import is_sequence

try:
    from collections.abc import Mapping
except BaseException:
    from collections import Mapping


# sha256 is always available
from hashlib import sha256 as _sha256


def sha256(item):
    return int(_sha256(item).hexdigest(), 16)


try:
    # blake2b is available on Python 3 and
    from hashlib import blake2b as _blake2b

    def hash_fallback(item):
        return int(_blake2b(item).hexdigest(), 16)

except BaseException:
    # fallback to sha256
    hash_fallback = sha256

# xxhash is up to 30x faster than sha256:
# `pip install xxhash`
try:
    # newest version of algorithm
    from xxhash import xxh3_64_intdigest as hash_fast
except BaseException:
    try:
        # older version of the algorithm
        from xxhash import xxh64_intdigest as hash_fast
    except BaseException:
        # use hashlib as a fallback hashing library
        log.debug("falling back to hashlib " + "hashing: `pip install xxhash`" + "for 50x faster cache checks")
        hash_fast = hash_fallback


def tracked_array(array, dtype=None):
    """
    Properly subclass a numpy ndarray to track changes.

    Avoids some pitfalls of subclassing by forcing contiguous
    arrays and does a view into a TrackedArray.

    Parameters
    ------------
    array : array- like object
      To be turned into a TrackedArray
    dtype : np.dtype
      Which dtype to use for the array

    Returns
    ------------
    tracked : TrackedArray
      Contains input array data.
    """
    # if someone passed us None, just create an empty array
    if array is None:
        array = []
    # make sure it is contiguous then view it as our subclass
    tracked = np.ascontiguousarray(array, dtype=dtype).view(TrackedArray)
    # should always be contiguous here
    assert tracked.flags["C_CONTIGUOUS"]

    return tracked


def cache_decorator(function):
    """
    A decorator for class methods, replaces @property
    but will store and retrieve function return values
    in object cache.

    Parameters
    ------------
    function : method
      This is used as a decorator:
      ```
      @cache_decorator
      def foo(self, things):
        return 'happy days'
      ```
    """

    # use wraps to preserve docstring
    @wraps(function)
    def get_cached(*args, **kwargs):
        """
        Only execute the function if its value isn't stored
        in cache already.
        """
        self = args[0]
        # use function name as key in cache
        name = function.__name__
        # do the dump logic ourselves to avoid
        # verifying cache twice per call
        self._cache.verify()
        # access cache dict to avoid automatic validation
        # since we already called cache.verify manually
        if name in self._cache.cache:
            # already stored so return value
            return self._cache.cache[name]
        # value not in cache so execute the function
        value = function(*args, **kwargs)
        # store the value
        if self._cache.force_immutable and hasattr(value, "flags") and len(value.shape) > 0:
            value.flags.writeable = False

        self._cache.cache[name] = value

        return value

    # all cached values are also properties
    # so they can be accessed like value attributes
    # rather than functions
    return property(get_cached)


class TrackedArray(np.ndarray):
    """
    Subclass of numpy.ndarray that provides hash methods
    to track changes.

    General method is to aggressively set 'modified' flags
    on operations which might (but don't necessarily) alter
    the array, ideally we sometimes compute hashes when we
    don't need to, but we don't return wrong hashes ever.

    We store boolean modified flag for each hash type to
    make checks fast even for queries of different hashes.

    Methods
    ----------
    __hash__ : int
      Runs the fastest available hash in this order:
        `xxh3_64, xxh_64, blake2b, sha256`
    """

    def __array_finalize__(self, obj):
        """
        Sets a modified flag on every TrackedArray
        This flag will be set on every change as well as
        during copies and certain types of slicing.
        """

        self._dirty_hash = True
        if isinstance(obj, type(self)):
            obj._dirty_hash = True

    def __array_wrap__(self, out_arr, context=None):
        """
        Return a numpy scalar if array is 0d.
        See https://github.com/numpy/numpy/issues/5819
        """
        if out_arr.ndim:
            return np.ndarray.__array_wrap__(self, out_arr, context)
        # Match numpy's behavior and return a numpy dtype scalar
        return out_arr[()]

    @property
    def mutable(self):
        return self.flags["WRITEABLE"]

    @mutable.setter
    def mutable(self, value):
        self.flags.writeable = value

    def hash(self):
        warnings.warn(
            "`array.hash()` is deprecated and will "
            + "be removed in October 2023: replace "
            + "with `array.__hash__()` or `hash(array)`",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.__hash__()

    def crc(self):
        warnings.warn(
            "`array.crc()` is deprecated and will "
            + "be removed in October 2023: replace "
            + "with `array.__hash__()` or `hash(array)`",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.__hash__()

    def md5(self):
        warnings.warn(
            "`array.md5()` is deprecated and will "
            + "be removed in October 2023: replace "
            + "with `array.__hash__()` or `hash(array)`",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.__hash__()

    def __hash__(self):
        """
        Return a fast hash of the contents of the array.

        Returns
        -------------
        hash : long int
          A hash of the array contents.
        """
        # repeat the bookkeeping to get a contiguous array
        if not self._dirty_hash and hasattr(self, "_hashed"):
            # we have a valid hash without recomputing.
            return self._hashed

        # run a hashing function on the C-order bytes copy
        hashed = hash_fast(self.tobytes(order="C"))

        # assign the value and set the flag
        self._hashed = hashed
        self._dirty_hash = False

        return hashed

    def __iadd__(self, *args, **kwargs):
        """
        In-place addition.

        The i* operations are in- place and modify the array,
        so we better catch all of them.
        """

        self._dirty_hash = True
        return super(self.__class__, self).__iadd__(*args, **kwargs)

    def __isub__(self, *args, **kwargs):
        self._dirty_hash = True
        return super(self.__class__, self).__isub__(*args, **kwargs)

    def fill(self, *args, **kwargs):
        self._dirty_hash = True
        return super(self.__class__, self).fill(*args, **kwargs)

    def partition(self, *args, **kwargs):
        self._dirty_hash = True
        return super(self.__class__, self).partition(*args, **kwargs)

    def put(self, *args, **kwargs):
        self._dirty_hash = True
        return super(self.__class__, self).put(*args, **kwargs)

    def byteswap(self, *args, **kwargs):
        self._dirty_hash = True
        return super(self.__class__, self).byteswap(*args, **kwargs)

    def itemset(self, *args, **kwargs):
        self._dirty_hash = True
        return super(self.__class__, self).itemset(*args, **kwargs)

    def sort(self, *args, **kwargs):
        self._dirty_hash = True
        return super(self.__class__, self).sort(*args, **kwargs)

    def setflags(self, *args, **kwargs):
        self._dirty_hash = True
        return super(self.__class__, self).setflags(*args, **kwargs)

    def __imul__(self, *args, **kwargs):
        self._dirty_hash = True
        return super(self.__class__, self).__imul__(*args, **kwargs)

    def __idiv__(self, *args, **kwargs):
        self._dirty_hash = True
        return super(self.__class__, self).__idiv__(*args, **kwargs)

    def __itruediv__(self, *args, **kwargs):
        self._dirty_hash = True
        return super(self.__class__, self).__itruediv__(*args, **kwargs)

    def __imatmul__(self, *args, **kwargs):
        self._dirty_hash = True
        return super(self.__class__, self).__imatmul__(*args, **kwargs)

    def __ipow__(self, *args, **kwargs):
        self._dirty_hash = True
        return super(self.__class__, self).__ipow__(*args, **kwargs)

    def __imod__(self, *args, **kwargs):
        self._dirty_hash = True
        return super(self.__class__, self).__imod__(*args, **kwargs)

    def __ifloordiv__(self, *args, **kwargs):
        self._dirty_hash = True
        return super(self.__class__, self).__ifloordiv__(*args, **kwargs)

    def __ilshift__(self, *args, **kwargs):
        self._dirty_hash = True
        return super(self.__class__, self).__ilshift__(*args, **kwargs)

    def __irshift__(self, *args, **kwargs):
        self._dirty_hash = True
        return super(self.__class__, self).__irshift__(*args, **kwargs)

    def __iand__(self, *args, **kwargs):
        self._dirty_hash = True
        return super(self.__class__, self).__iand__(*args, **kwargs)

    def __ixor__(self, *args, **kwargs):
        self._dirty_hash = True
        return super(self.__class__, self).__ixor__(*args, **kwargs)

    def __ior__(self, *args, **kwargs):
        self._dirty_hash = True
        return super(self.__class__, self).__ior__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        self._dirty_hash = True
        return super(self.__class__, self).__setitem__(*args, **kwargs)

    def __setslice__(self, *args, **kwargs):
        self._dirty_hash = True
        return super(self.__class__, self).__setslice__(*args, **kwargs)


class Cache(object):
    """
    Class to cache values which will be stored until the
    result of an ID function changes.
    """

    def __init__(self, id_function, force_immutable=False):
        """
        Create a cache object.

        Parameters
        ------------
        id_function : function
          Returns hashable value
        force_immutable : bool
          If set will make all numpy arrays read-only
        """
        self._id_function = id_function
        # for stored numpy arrays set `flags.writable = False`
        self.force_immutable = bool(force_immutable)
        # call the id function for initial value
        self.id_current = self._id_function()
        # a counter for locks
        self._lock = 0
        # actual store for data
        self.cache = {}

    def delete(self, key):
        """
        Remove a key from the cache.
        """
        if key in self.cache:
            self.cache.pop(key, None)

    def verify(self):
        """
        Verify that the cached values are still for the same
        value of id_function and delete all stored items if
        the value of id_function has changed.
        """
        # if we are in a lock don't check anything
        if self._lock != 0:
            return

        # check the hash of our data
        id_new = self._id_function()

        # things changed
        if id_new != self.id_current:
            if len(self.cache) > 0:
                log.debug("%d items cleared from cache: %s", len(self.cache), str(list(self.cache.keys())))
            # hash changed, so dump the cache
            # do it manually rather than calling clear()
            # as we are internal logic and can avoid function calls
            self.cache = {}
            # set the id to the new data hash
            self.id_current = id_new

    def clear(self, exclude=None):
        """
        Remove elements in the cache.

        Parameters
        -----------
        exclude : list
          List of keys in cache to not clear.
        """
        if exclude is None:
            self.cache = {}
        else:
            self.cache = {k: v for k, v in self.cache.items() if k in exclude}

    def update(self, items):
        """
        Update the cache with a set of key, value pairs without
        checking id_function.
        """
        self.cache.update(items)

        if self.force_immutable:
            for v in self.cache.values():
                if hasattr(v, "flags") and len(v.shape) > 0:
                    v.flags.writeable = False
        self.id_set()

    def id_set(self):
        """
        Set the current ID to the value of the ID function.
        """
        self.id_current = self._id_function()

    def __getitem__(self, key):
        """
        Get an item from the cache. If the item
        is not in the cache, it will return None

        Parameters
        -------------
        key : hashable
               Key in dict

        Returns
        -------------
        cached : object, or None
          Object that was stored
        """
        self.verify()
        if key in self.cache:
            return self.cache[key]
        return None

    def __setitem__(self, key, value):
        """
        Add an item to the cache.

        Parameters
        ------------
        key : hashable
          Key to reference value
        value : any
          Value to store in cache
        """
        # dumpy cache if ID function has changed
        self.verify()
        # make numpy arrays read-only if asked to
        if self.force_immutable and hasattr(value, "flags") and len(value.shape) > 0:
            value.flags.writeable = False
        # assign data to dict
        self.cache[key] = value

        return value

    def __contains__(self, key):
        self.verify()
        return key in self.cache

    def __len__(self):
        self.verify()
        return len(self.cache)

    def __enter__(self):
        self._lock += 1

    def __exit__(self, *args):
        self._lock -= 1
        self.id_current = self._id_function()


class DiskCache(object):
    """
    Store results of expensive operations on disk
    with an option to expire the results. This is used
    to cache the multi-gigabyte test corpuses in
    `tests/corpus.py`
    """

    def __init__(self, path, expire_days=30):
        """
        Create a cache on disk for storing expensive results.

        Parameters
        --------------
        path : str
          A writeable location on the current file path.
        expire_days : int or float
          How old should results be considered expired.

        """
        # store how old we allow results to be
        self.expire_days = expire_days
        # store the location for saving results
        self.path = os.path.abspath(os.path.expanduser(path))
        # make sure the specified path exists
        os.makedirs(self.path, exist_ok=True)

    def get(self, key, fetch):
        """
        Get a key from the cache or run a calculation.

        Parameters
        -----------
        key : str
          Key to reference item with
        fetch : function
          If key isn't stored and recent run this
          function and store its result on disk.
        """
        # hash the key so we have a fixed length string
        key_hash = _sha256(key.encode("utf-8")).hexdigest()
        # full path of result on local disk
        path = os.path.join(self.path, key_hash)

        # check to see if we can use the cache
        if os.path.isfile(path):
            # compute the age of the existing file in days
            age_days = (time.time() - os.stat(path).st_mtime) / 86400.0
            if age_days < self.expire_days:
                # this nested condition means that
                # the file both exists and is recent
                # enough, so just return its contents
                with open(path, "rb") as f:
                    return f.read()

        log.debug("not in cache fetching: `{}`".format(key))
        # since we made it here our data isn't cached
        # run the expensive function to fetch the file
        raw = fetch()
        # write the data so we can save it
        with open(path, "wb") as f:
            f.write(raw)

        # return the data
        return raw


class DataStore(Mapping):
    """
    A class to store multiple numpy arrays and track them all
    for changes.

    Operates like a dict that only stores numpy.ndarray
    """

    def __init__(self):
        self.data = {}

    def __iter__(self):
        return iter(self.data)

    def pop(self, key):
        return self.data.pop(key, None)

    def __delitem__(self, key):
        self.data.pop(key, None)

    @property
    def mutable(self):
        """
        Is data allowed to be altered or not.

        Returns
        -----------
        is_mutable : bool
          Can data be altered in the DataStore
        """
        return getattr(self, "_mutable", True)

    @mutable.setter
    def mutable(self, value):
        """
        Is data allowed to be altered or not.

        Parameters
        ------------
        is_mutable : bool
          Should data be allowed to be altered
        """
        # make sure passed value is a bool
        is_mutable = bool(value)
        # apply the flag to any data stored
        for v in self.data.values():
            v.mutable = value
        # save the mutable setting
        self._mutable = is_mutable

    def is_empty(self):
        """
        Is the current DataStore empty or not.

        Returns
        ----------
        empty : bool
          False if there are items in the DataStore
        """
        if len(self.data) == 0:
            return True
        for v in self.data.values():
            if is_sequence(v):
                if len(v) == 0:
                    return True
                else:
                    return False
            elif bool(np.isreal(v)):
                return False
        return True

    def clear(self):
        """
        Remove all data from the DataStore.
        """
        self.data = {}

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, data):
        """
        Store an item in the DataStore
        """
        # we shouldn't allow setting on immutable datastores
        if not self.mutable:
            raise ValueError("DataStore is configured immutable!")

        if hasattr(data, "hash"):
            # don't bother to re-track TrackedArray
            tracked = data
        else:
            # otherwise wrap data
            tracked = tracked_array(data)
        # apply our mutability setting

        if hasattr(self, "_mutable"):
            # apply our mutability setting only if it was explicitly set
            tracked.mutable = self.mutable
        # store data
        self.data[key] = tracked

    def __contains__(self, key):
        return key in self.data

    def __len__(self):
        return len(self.data)

    def update(self, values):
        if not isinstance(values, dict):
            raise ValueError("Update only implemented for dicts")
        for key, value in values.items():
            self[key] = value

    def __hash__(self):
        """
        Get a hash reflecting everything in the DataStore.

        Returns
        ----------
        hash : str
          hash of data in hexadecimal
        """
        # only hash values that aren't None
        # or if they are arrays require length greater than zero
        return hash_fast(
            np.array(
                [hash(v) for v in self.data.values() if v is not None and (not hasattr(v, "__len__") or len(v) > 0)],
                dtype=np.int64,
            ).tobytes()
        )

    def crc(self):
        """
        Get a CRC reflecting everything in the DataStore.

        Returns
        ----------
        crc : int
          CRC of data
        """
        warnings.warn(
            "`array.crc()` is deprecated and will "
            + "be removed in October 2023: replace "
            + "with `array.__hash__()` or `hash(array)`",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.__hash__()

    def fast_hash(self):
        """
        Get a CRC32 or xxhash.xxh64 reflecting the DataStore.

        Returns
        ------------
        hashed : int
          Checksum of data
        """
        warnings.warn(
            "`array.fast_hash()` is deprecated and will "
            + "be removed in October 2023: replace "
            + "with `array.__hash__()` or `hash(array)`",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.__hash__()

    def hash(self):
        warnings.warn(
            "`array.hash()` is deprecated and will "
            + "be removed in October 2023: replace "
            + "with `array.__hash__()` or `hash(array)`",
            category=DeprecationWarning,
            stacklevel=2,
        )

        return self.__hash__()
