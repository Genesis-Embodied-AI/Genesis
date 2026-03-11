import enum
from pathlib import PurePath
from typing import TYPE_CHECKING, Annotated, Sequence, Iterable

import numpy as np

from pydantic import Field, BeforeValidator, StrictFloat, StrictInt, GetPydanticSchema
from pydantic_core import core_schema


# dynamic loading
ACTIVE = 1
INACTIVE = 0


# type aliases
if TYPE_CHECKING:
    NumericType = int | float | bool | np.number
    NumArrayType = Sequence[NumericType]
    IArrayType = Sequence[int | np.integer]
    FArrayType = Sequence[NumericType]
    Vec2IType = IArrayType
    Vec3FType = FArrayType
    ColorFloat = float
    ColorArrayType = FArrayType
    MaybeColorArrayType = FArrayType
    Color3Type = FArrayType
    Vec4FType = FArrayType
    Vec3FArrayType = Sequence[Sequence[NumericType]]
    Matrix3x3Type = Vec3FArrayType
    NDArrayType = np.ndarray
    PathType = str | PurePath
else:
    NumericType = int | float | bool
    NumArrayType = Annotated[tuple[NumericType, ...], Field(min_length=1, strict=False)]
    IArrayType = Annotated[tuple[StrictInt, ...], Field(min_length=1, strict=False)]
    FArrayType = Annotated[tuple[float, ...], Field(min_length=1, strict=False)]
    Vec2IType = Annotated[tuple[StrictInt, StrictInt], Field(strict=False)]
    Vec3FType = Annotated[tuple[float, float, float], Field(strict=False)]
    ColorFloat = Annotated[StrictFloat, Field(ge=0.0, le=1.0, strict=False)]
    ColorArrayType = Annotated[tuple[ColorFloat, ...], Field(min_length=1, strict=False)]
    MaybeColorArrayType = Annotated[
        tuple[ColorFloat, ...],
        BeforeValidator(lambda v: v if isinstance(v, Iterable) else (v,)),
        Field(min_length=1, strict=False),
    ]
    Color3Type = Annotated[tuple[ColorFloat, ColorFloat, ColorFloat], Field(strict=False)]
    Vec4FType = Annotated[tuple[float, float, float, float], Field(strict=False)]
    Vec3FArrayType = Annotated[tuple[Vec3FType, ...], Field(min_length=1, strict=False)]
    Matrix3x3Type = Annotated[tuple[Vec3FType, Vec3FType, Vec3FType], Field(strict=False)]
    NDArrayType = Annotated[
        np.ndarray, GetPydanticSchema(lambda tp, handler: core_schema.no_info_plain_validator_function(lambda v: v))
    ]
    PathType = Annotated[str, BeforeValidator(lambda v: str(v) if isinstance(v, PurePath) else v)]

MaybeNumArrayType = NumArrayType | NumericType
MaybeVec3FType = Vec3FType | float
MaybeVec3FArrayType = Vec3FArrayType | Vec3FType
MaybeMatrix3x3Type = Matrix3x3Type | MaybeVec3FType


class IntEnum(enum.IntEnum):
    def __repr__(self):
        return f"<gs.{self.__class__.__name__}.{self.name}: {self.value}>"

    def __format__(self, format_spec):
        return f"<{self.name}: {self.value}>"


# geom type in rigid solver
class GEOM_TYPE(IntEnum):
    # Beware PLANE must be the first geometry type as this is assumed by MPR collision detection.
    PLANE = 0
    SPHERE = 1
    ELLIPSOID = 2
    CYLINDER = 3
    CAPSULE = 4
    BOX = 5
    MESH = 6
    TERRAIN = 7


# joint type in rigid solver, ranked by number of dofs
class JOINT_TYPE(IntEnum):
    FIXED = 0
    REVOLUTE = 1
    PRISMATIC = 2
    SPHERICAL = 3
    FREE = 4


class EQUALITY_TYPE(IntEnum):
    CONNECT = 0
    WELD = 1
    JOINT = 2


class CTRL_MODE(IntEnum):
    FORCE = 0
    VELOCITY = 1
    POSITION = 2


######### User accessible constants do not capitalize #########
# rigid solver intergrator
class integrator(IntEnum):
    Euler = 0
    implicitfast = 1
    approximate_implicitfast = 2


# rigid solver constraint solver
class constraint_solver(IntEnum):
    CG = 0
    Newton = 1


# backend
class backend(IntEnum):
    cpu = 0
    gpu = 1
    cuda = 2
    amdgpu = 3
    metal = 4

    def __format__(self, format_spec):
        return f"gs.{self.name}"


# image types for visualization
class IMAGE_TYPE(IntEnum):
    RGB = 0
    DEPTH = 1
    SEGMENTATION = 2
    NORMAL = 3

    def __format__(self, format_spec):
        return self.name


# parallelize
class PARA_LEVEL(IntEnum):
    NEVER = 0  # when using cpu
    PARTIAL = 1  # when using gpu for non-batched scene
    ALL = 2  # when using gpu for batched scene
