import math
from pathlib import PurePath
from typing import TYPE_CHECKING, Annotated, Mapping, Sequence

import numpy as np

from pydantic import Field, BeforeValidator, StrictFloat, StrictInt, GetPydanticSchema
from pydantic_core import core_schema


def _normalize_vector(vec):
    vec = tuple(map(float, vec))
    norm = math.sqrt(sum(e**2 for e in vec))
    if norm > 0:
        vec = tuple(e / norm for e in vec)
    return vec


def _is_sequence(v):
    return hasattr(v, "__len__") and hasattr(v, "__getitem__") and not isinstance(v, (str, bytes, Mapping))


def _scalar_or_sequence_to_tuple(v):
    return tuple(v) if _is_sequence(v) else (v,)


# type aliases
if TYPE_CHECKING:
    ValidFloat = float
    NonNegativeFloat = float
    PositiveFloat = float
    NonNegativeInt = int
    PositiveInt = int
    NumericType = int | float | bool | np.number
    NumArrayType = Sequence[NumericType]
    IArrayType = Sequence[int | np.integer]
    FArrayType = Sequence[NumericType]
    Vec2IType = IArrayType
    PositiveVec2IType = IArrayType
    Vec2FType = FArrayType
    PositiveVec2FType = FArrayType
    Vec3FType = FArrayType
    UnitVec3FType = FArrayType
    UnitVec4FType = FArrayType
    ColorFloat = float
    ColorArrayType = FArrayType
    MaybeColorArrayType = FArrayType | float
    MaybeFArrayType = FArrayType | float
    Color3Type = FArrayType
    Vec4FType = FArrayType
    Vec3FArrayType = Sequence[Sequence[NumericType]]
    Matrix3x3Type = Vec3FArrayType
    Matrix4x4Type = Sequence[Sequence[NumericType]]
    StrArrayType = Sequence[str]
    NDArrayType = np.ndarray
    PathType = str | PurePath
else:
    ValidFloat = Annotated[float, Field(strict=False, allow_inf_nan=False)]
    NonNegativeFloat = Annotated[float, Field(strict=False, ge=0, allow_inf_nan=False)]
    PositiveFloat = Annotated[float, Field(strict=False, gt=0, allow_inf_nan=False)]
    NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
    PositiveInt = Annotated[int, Field(strict=True, gt=0)]
    NumericType = int | float | bool
    NumArrayType = Annotated[tuple[NumericType, ...], Field(min_length=1, strict=False)]
    IArrayType = Annotated[tuple[StrictInt, ...], Field(min_length=1, strict=False)]
    FArrayType = Annotated[tuple[ValidFloat, ...], Field(min_length=1, strict=False)]
    Vec2IType = Annotated[tuple[StrictInt, StrictInt], Field(strict=False)]
    PositiveVec2IType = Annotated[tuple[PositiveInt, PositiveInt], Field(strict=False)]
    Vec2FType = Annotated[tuple[ValidFloat, ValidFloat], Field(strict=False)]
    PositiveVec2FType = Annotated[tuple[PositiveFloat, PositiveFloat], Field(strict=False)]
    Vec3FType = Annotated[tuple[ValidFloat, ValidFloat, ValidFloat], Field(strict=False)]
    UnitVec3FType = Annotated[
        tuple[ValidFloat, ValidFloat, ValidFloat], BeforeValidator(_normalize_vector), Field(strict=False)
    ]
    UnitVec4FType = Annotated[
        tuple[ValidFloat, ValidFloat, ValidFloat, ValidFloat], BeforeValidator(_normalize_vector), Field(strict=False)
    ]
    ColorFloat = Annotated[StrictFloat, Field(ge=0.0, le=1.0, strict=False, allow_inf_nan=False)]
    ColorArrayType = Annotated[tuple[ColorFloat, ...], Field(min_length=1, strict=False)]
    MaybeColorArrayType = Annotated[
        tuple[ColorFloat, ...],
        BeforeValidator(_scalar_or_sequence_to_tuple),
        Field(min_length=1, strict=False),
    ]
    MaybeFArrayType = Annotated[
        tuple[ValidFloat, ...],
        BeforeValidator(_scalar_or_sequence_to_tuple),
        Field(min_length=1, strict=False),
    ]
    Color3Type = Annotated[tuple[ColorFloat, ColorFloat, ColorFloat], Field(strict=False)]
    Vec4FType = Annotated[tuple[ValidFloat, ValidFloat, ValidFloat, ValidFloat], Field(strict=False)]
    StrArrayType = Annotated[tuple[str, ...], Field(strict=False)]
    Vec3FArrayType = Annotated[tuple[Vec3FType, ...], Field(min_length=1, strict=False)]
    Matrix3x3Type = Annotated[tuple[Vec3FType, Vec3FType, Vec3FType], Field(strict=False)]
    Matrix4x4Type = Annotated[
        tuple[Vec4FType, Vec4FType, Vec4FType, Vec4FType],
        BeforeValidator(lambda v: tuple(tuple(row) for row in v) if isinstance(v, np.ndarray) else v),
        Field(strict=False),
    ]
    NDArrayType = Annotated[
        np.ndarray, GetPydanticSchema(lambda tp, handler: core_schema.no_info_plain_validator_function(lambda v: v))
    ]
    PathType = Annotated[str, BeforeValidator(lambda v: str(v) if isinstance(v, PurePath) else v)]

MaybeNumArrayType = NumArrayType | NumericType
MaybeVec3FType = Vec3FType | float
MaybeVec3FArrayType = Vec3FArrayType | Vec3FType
MaybeMatrix3x3Type = Matrix3x3Type | MaybeVec3FType
