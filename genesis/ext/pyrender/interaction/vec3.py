import numpy as np
from numpy.typing import NDArray


class Vec3:
    """
    Use this wrapper around np.array if you want to ensure adherence to float32 arithmethic
    with runtime checks, and avoid hidden and costly conversions between float32 and float64.

    This also makes vector dimensionality explicit for linting and static analysis.
    """
    v: NDArray[np.float32]

    def __init__(self, v: NDArray[np.float32]):
        assert v.shape == (3,), f"Vec3 must be initialized with a 3-element array, got {v.shape}"
        assert v.dtype == np.float32, f"Vec3 must be initialized with a float32 array, got {v.dtype}"
        self.v = v

    def __add__(self, other: 'Vec3') -> 'Vec3':
        return Vec3(self.v + other.v)

    def __sub__(self, other: 'Vec3') -> 'Vec3':
        return Vec3(self.v - other.v)

    def __mul__(self, other: float) -> 'Vec3':
        return Vec3(self.v * np.float32(other))

    def __rmul__(self, other: float) -> 'Vec3':
        return Vec3(self.v * np.float32(other))

    def dot(self, other: 'Vec3') -> float:
        return np.dot(self.v, other.v).item()

    def cross(self, other: 'Vec3') -> 'Vec3':
        return Vec3(np.cross(self.v, other.v))

    def normalized(self) -> 'Vec3':
        return Vec3(self.v / (np.linalg.norm(self.v) + 1e-24))

    def copy(self) -> 'Vec3':
        return Vec3(self.v.copy())

    def __repr__(self) -> str:
        return f"Vec3({self.v[0]}, {self.v[1]}, {self.v[2]})"


    @classmethod
    def from_xyz(cls, x: float, y: float, z: float) -> 'Vec3':
        return cls(np.array([x, y, z], dtype=np.float32))

    @classmethod
    def from_int32(cls, v: NDArray[np.int32]) -> 'Vec3':
        assert v.shape == (3,), f"Vec3 must be initialized with a 3-element array, got {v.shape}"
        assert v.dtype == np.int32, f"from_int32 must be initialized with a int32 array, got {v.dtype}"
        return cls.from_xyz(*v)

    @classmethod
    def from_int64(cls, v: NDArray[np.int64]) -> 'Vec3':
        assert v.shape == (3,), f"Vec3 must be initialized with a 3-element array, got {v.shape}"
        assert v.dtype == np.int64, f"from_int64 must be initialized with a int64 array, got {v.dtype}"
        return cls.from_xyz(*v)

    @classmethod
    def from_float64(cls, v: NDArray[np.float64]) -> 'Vec3':
        assert v.shape == (3,), f"Vec3 must be initialized with a 3-element array, got {v.shape}"
        assert v.dtype == np.float64, f"from_float64 must be initialized with a float64 array, got {v.dtype}"
        return cls.from_xyz(*v)


    @classmethod
    def zero(cls):
        return cls(np.array([0, 0, 0], dtype=np.float32))

    @classmethod
    def one(cls):
        return cls(np.array([1, 1, 1], dtype=np.float32))
