from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
from numpy.typing import NDArray, ArrayLike

from genesis.utils.misc import tensor_to_array

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_geom import RigidGeom
    from genesis.engine.entities.rigid_entity.rigid_link import RigidLink

# If not needing runtime checks, we can just use annotated types:
# Vec3 = Annotated[npt.NDArray[np.float32], (3,)]
# Aabb = Annotated[npt.NDArray[np.float32], (2, 3)]


def _ensure_torch_imported() -> None:
    global gs
    import genesis as gs
    global torch
    import torch


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

    def __neg__(self) -> 'Vec3':
        return Vec3(-self.v)

    def dot(self, other: 'Vec3') -> float:
        return np.dot(self.v, other.v).item()

    def cross(self, other: 'Vec3') -> 'Vec3':
        return Vec3(np.cross(self.v, other.v))

    def normalized(self) -> 'Vec3':
        return Vec3(self.v / (np.linalg.norm(self.v) + 1e-24))

    def magnitude(self) -> float:
        return np.linalg.norm(self.v)

    def sqr_magnitude(self) -> float:
        return np.dot(self.v, self.v)

    def copy(self) -> 'Vec3':
        return Vec3(self.v.copy())

    def __repr__(self) -> str:
        return f"Vec3({self.v[0]}, {self.v[1]}, {self.v[2]})"

    def as_tensor(self) -> 'torch.Tensor':
        _ensure_torch_imported()
        return torch.tensor(self.v, dtype=gs.tc_float) 

    @property
    def x(self) -> float:
        return self.v[0]

    @property
    def y(self) -> float:
        return self.v[1]

    @property
    def z(self) -> float:
        return self.v[2]

    @classmethod
    def from_xyz(cls, x: float, y: float, z: float) -> 'Vec3':
        return cls(np.array([x, y, z], dtype=np.float32))

    @classmethod
    def from_array(cls, v: np.ndarray) -> 'Vec3':
        assert v.shape == (3,), f"Vec3 must be initialized with a 3-element array, got {v.shape}"
        assert v.dtype == np.int32 or v.dtype == np.int64 or v.dtype == np.float32 or v.dtype == np.float64, \
            f"from_array must be initialized with a array of ints/floats 32/64-bit, got {v.dtype}"
        return cls.from_xyz(*v)

    @classmethod
    def from_tensor(cls, v: 'torch.Tensor') -> 'Vec3':
        _ensure_torch_imported()
        array: np.ndarray = tensor_to_array(v)
        return cls.from_array(array)

    @classmethod
    def from_arraylike(cls, v: ArrayLike) -> 'Vec3':
        if isinstance(v, np.ndarray):
            return cls.from_array(v)
        elif isinstance(v, torch.Tensor):
            return cls.from_tensor(v)
        elif isinstance(v, ArrayLike):
            assert len(v) == 3, f"Vec3 must be initialized with a 3-element ArrayLike, got {len(v)}"
            return cls.from_xyz(*v)
        assert False


    @classmethod
    def zero(cls) -> 'Vec3':
        return cls(np.array([0, 0, 0], dtype=np.float32))

    @classmethod
    def one(cls) -> 'Vec3':
        return cls(np.array([1, 1, 1], dtype=np.float32))

    @classmethod
    def full(cls, fill_value: float) -> 'Vec3':
        return cls(np.full((3,), fill_value, dtype=np.float32))


class Quat:
    v: NDArray[np.float32]
    def __init__(self, v: NDArray[np.float32]):
        assert v.shape == (4,), f"Quat must be initialized with a 4-element array, got {v.shape}"
        assert v.dtype == np.float32, f"Quat must be initialized with a float32 array, got {v.dtype}"
        self.v = v

    def get_inverse(self) -> 'Quat':
        quat_inv = self.v.copy()
        quat_inv[1:] *= -1
        return Quat(quat_inv)

    def __mul__(self, other: Union['Quat', Vec3]) -> Union['Quat', Vec3]:
        if isinstance(other, Quat):
            # Quaternion * Quaternion
            w1, x1, y1, z1 = self.w, self.x, self.y, self.z
            w2, x2, y2, z2 = other.w, other.x, other.y, other.z
            return Quat.from_wxyz(
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            )
        elif isinstance(other, Vec3):  # (other, np.ndarray) and other.shape == (3,):
            # Quaternion * Vector3 -> rotate vector
            v_quat = Quat.from_wxyz(0, *other.v)
            result = self * v_quat * self.get_inverse()
            return Vec3(result.v[1:])
        else:
            return NotImplemented

    def copy(self) -> 'Quat':
        return Quat(self.v.copy())

    def __repr__(self) -> str:
        return f"Quat({self.v[0]}, {self.v[1]}, {self.v[2]}, {self.v[3]})"

    def as_tensor(self) -> 'torch.Tensor':
        _ensure_torch_imported()
        return torch.tensor(self.v, dtype=gs.tc_float) 

    @property
    def w(self) -> float:
        return self.v[0]

    @property
    def x(self) -> float:
        return self.v[1]

    @property
    def y(self) -> float:
        return self.v[2]

    @property
    def z(self) -> float:
        return self.v[3]

    @classmethod
    def from_wxyz(cls, w: float, x: float, y: float, z: float) -> 'Quat':
        return cls(np.array([w, x, y, z], dtype=np.float32))

    @classmethod
    def from_array(cls, v: np.ndarray) -> 'Quat':
        assert v.shape == (4,), f"Quat must be initialized with a 4-element array, got {v.shape}"
        return cls.from_wxyz(*v)

    @classmethod
    def from_tensor(cls, v: 'torch.Tensor') -> 'Quat':
        _ensure_torch_imported()
        array: np.ndarray = tensor_to_array(v)
        return cls.from_array(array)

    @classmethod
    def from_arraylike(cls, v: ArrayLike) -> 'Quat':
        if isinstance(v, np.ndarray):
            return cls.from_array(v)
        elif isinstance(v, torch.Tensor):
            return cls.from_tensor(v)
        elif isinstance(v, ArrayLike):
            assert len(v) == 4, f"Quat must be initialized with a 4-element ArrayLike, got {len(v)}"
            return cls.from_wxyz(*v)
        assert False


@dataclass
class Pose:
    pos: Vec3
    rot: Quat

    # todo: consider using a single np.array with views

    def transform_point(self, point: Vec3) -> Vec3:
        return self.pos + self.rot * point

    def inverse_transform_point(self, point: Vec3) -> Vec3:
        return self.rot.get_inverse() * (point - self.pos)

    def transform_direction(self, direction: Vec3) -> Vec3:
        return self.rot * direction

    def inverse_transform_direction(self, direction: Vec3) -> Vec3:
        return self.rot.get_inverse() * direction

    def get_inverse(self) -> 'Pose':
        inv_rot = self.rot.get_inverse()
        # inv_pos = -1.0 * (inv_rot * self.pos)
        # faster -- avoid repeated quat inversion:
        pos_quat = Quat.from_wxyz(0, *self.pos.v)
        inv_pos = inv_rot * pos_quat * self.rot
        inv_pos = Vec3(-inv_pos.v[1:])
        return Pose(inv_pos, inv_rot)

    def __mul__(self, other: Union['Pose', Vec3]) -> Union['Pose', Vec3]:
        if isinstance(other, Pose):
            return Pose(self.pos + self.rot * other.pos, self.rot * other.rot)
        elif isinstance(other, Vec3):
            return self.pos + self.rot * other
        else:
            return NotImplemented        

    def __repr__(self) -> str:
        return f"Pose(pos={self.pos}, rot={self.rot})"

    @classmethod
    def from_geom(cls, geom: 'RigidGeom') -> 'Pose':
        assert geom._solver.n_envs == 0, "ViewerInteraction only supports single-env for now"
        # geom.get_pos() and .get_quat() are squeezed if n_envs == 0
        pos = Vec3.from_tensor(geom.get_pos())
        quat = Quat.from_tensor(geom.get_quat())
        return Pose(pos, quat)

    @classmethod
    def from_link(cls, link: 'RigidLink') -> 'Pose':
        assert link._solver.n_envs == 0, "ViewerInteraction only supports single-env for now"
        # geom.get_pos() and .get_quat() are squeezed if n_envs == 0
        pos = Vec3.from_tensor(link.get_pos())
        quat = Quat.from_tensor(link.get_quat())
        return Pose(pos, quat)


@dataclass
class Color:
    r: float
    g: float
    b: float
    a: float

    def tuple(self) -> tuple[float, float, float, float]:
        return (self.r, self.g, self.b, self.a)

    def with_alpha(self, alpha: float) -> 'Color':
        return Color(self.r, self.g, self.b, alpha)

    @classmethod
    def red(cls) -> 'Color':
        return cls(1.0, 0.0, 0.0, 1.0)

    @classmethod
    def green(cls) -> 'Color':
        return cls(0.0, 1.0, 0.0, 1.0)

    @classmethod
    def blue(cls) -> 'Color':
        return cls(0.0, 0.0, 1.0, 1.0)

    @classmethod
    def yellow(cls) -> 'Color':
        return cls(1.0, 1.0, 0.0, 1.0)
