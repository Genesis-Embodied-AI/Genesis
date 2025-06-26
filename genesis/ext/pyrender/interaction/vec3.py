import numpy as np
from numpy.typing import NDArray

class Vec3:
    v: NDArray[np.float32]

    def __init__(self, v: NDArray[np.float32]):
        assert v.shape == (3,), f"Vec3 must be initialized with a 3-element array, got {v.shape}"
        self.v = v

    def __add__(self, other: 'Vec3') -> 'Vec3': return Vec3(self.v + other.v)
    def __sub__(self, other: 'Vec3') -> 'Vec3': return Vec3(self.v - other.v)
    def __mul__(self, other: float) -> 'Vec3': return Vec3(self.v * other)
    def __rmul__(self, other: float) -> 'Vec3': return Vec3(self.v * other)
    def dot(self, other: 'Vec3') -> float: return np.dot(self.v, other.v)
    def cross(self, other: 'Vec3') -> 'Vec3': return Vec3(np.cross(self.v, other.v))

    def normalized(self) -> 'Vec3': return Vec3(self.v / (np.linalg.norm(self.v) + 1e-24))

    def copy(self) -> 'Vec3': return Vec3(self.v.copy())

    def __repr__(self) -> str: return f"Vec3({self.v[0]}, {self.v[1]}, {self.v[2]})"

    @classmethod
    def from_xyz(cls, x: float, y: float, z: float) -> 'Vec3': return cls(np.array([x, y, z], dtype=np.float32))

    @classmethod
    def zero(cls): return cls(np.array([0, 0, 0]))
    @classmethod
    def one(cls): return cls(np.array([1, 1, 1]))
    
