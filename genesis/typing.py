import math
from pathlib import PurePath
from typing import TYPE_CHECKING, Annotated, Mapping, Sequence

import numpy as np

from pydantic import Field, BeforeValidator, GetPydanticSchema
from pydantic_core import core_schema, PydanticCustomError


def _coerce_int(v):
    """Accept numpy integers, reject booleans and floats."""
    if isinstance(v, (bool, np.bool_)):
        raise PydanticCustomError("invalid_type", "Input should be a valid integer, not boolean", {"value": v})
    if isinstance(v, np.integer):
        return int(v)
    return v


def _normalize(vec):
    if not _is_sequence(vec):
        raise PydanticCustomError("invalid_type", "Input should be a valid sequence of scalars", {"value": vec})
    sq_norm = 0.0
    for e in vec:
        if _is_sequence(e):
            raise PydanticCustomError("invalid_type", "Input should be a valid sequence of scalars", {"value": vec})
        sq_norm += e**2
    if sq_norm > 0:
        inv_norm = 1.0 / math.sqrt(sq_norm)
        vec = tuple(e * inv_norm for e in vec)
        return vec
    raise PydanticCustomError("zero_division", "Cannot be normalized", {"value": vec})


def _is_sequence(v):
    if isinstance(v, (str, bytes, Mapping)):
        return False
    if not (hasattr(v, "__len__") and hasattr(v, "__getitem__")):
        return False
    try:
        tuple(v)
    except TypeError:
        return False
    return True


# type aliases
if TYPE_CHECKING:
    ValidFloat = float | np.floating
    NonNegativeFloat = ValidFloat
    PositiveFloat = ValidFloat
    StrictInt = int | np.integer
    NonNegativeInt = StrictInt
    PositiveInt = StrictInt
    NumericType = int | float | bool | np.number
    NumArrayType = Sequence[NumericType] | np.ndarray
    IArrayType = Sequence[StrictInt] | np.ndarray
    FArrayType = Sequence[ValidFloat] | np.ndarray
    PositiveFArrayType = FArrayType
    Vec2IType = IArrayType
    PositiveVec2IType = IArrayType
    Vec2FType = FArrayType
    PositiveVec2FType = FArrayType
    Vec3FType = FArrayType
    LaxVec3FType = FArrayType | ValidFloat
    UnitVec3FType = FArrayType
    UnitVec4FType = FArrayType
    UnitInterval = ValidFloat
    UnitIntervalArrayType = FArrayType
    LaxUnitIntervalArrayType = FArrayType | ValidFloat
    LaxFArrayType = FArrayType | ValidFloat
    LaxPositiveFArrayType = LaxFArrayType
    UnitIntervalVec3Type = FArrayType
    UnitIntervalVec4Type = FArrayType
    Vec4FType = FArrayType
    Vec3FArrayType = Sequence[Sequence[NumericType]] | np.ndarray
    UnitVec3FArrayType = Vec3FArrayType
    Vec3FLaxArrayType = Vec3FArrayType | Vec3FType
    UnitVec3FLaxArrayType = Vec3FLaxArrayType
    RotationMatrixType = Vec3FArrayType
    Matrix4x4Type = Sequence[Sequence[NumericType]] | np.ndarray
    StrArrayType = Sequence[str]
    NDArrayType = np.ndarray
    PathType = str | PurePath
else:
    ValidFloat = Annotated[float, Field(allow_inf_nan=False, strict=False)]
    NonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False, strict=False)]
    PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False, strict=False)]
    StrictInt = Annotated[int, BeforeValidator(_coerce_int), Field(strict=True)]
    NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
    PositiveInt = Annotated[StrictInt, Field(gt=0)]
    NumericType = int | float | bool
    NumArrayType = Annotated[tuple[NumericType, ...], Field(min_length=1, strict=False)]
    IArrayType = Annotated[tuple[StrictInt, ...], Field(min_length=1, strict=False)]
    FArrayType = Annotated[tuple[ValidFloat, ...], Field(min_length=1, strict=False)]
    PositiveFArrayType = Annotated[tuple[PositiveFloat, ...], Field(min_length=1, strict=False)]
    Vec2IType = Annotated[tuple[StrictInt, StrictInt], Field(strict=False)]
    PositiveVec2IType = Annotated[tuple[PositiveInt, PositiveInt], Field(strict=False)]
    Vec2FType = Annotated[tuple[ValidFloat, ValidFloat], Field(strict=False)]
    PositiveVec2FType = Annotated[tuple[PositiveFloat, PositiveFloat], Field(strict=False)]
    Vec3FType = Annotated[tuple[ValidFloat, ValidFloat, ValidFloat], Field(strict=False)]
    LaxVec3FType = Annotated[
        tuple[ValidFloat, ValidFloat, ValidFloat],
        BeforeValidator(lambda v: v if _is_sequence(v) else (v,) * 3),
        Field(strict=False),
    ]
    UnitVec3FType = Annotated[
        tuple[ValidFloat, ValidFloat, ValidFloat], BeforeValidator(_normalize), Field(strict=False)
    ]
    UnitVec4FType = Annotated[
        tuple[ValidFloat, ValidFloat, ValidFloat, ValidFloat], BeforeValidator(_normalize), Field(strict=False)
    ]
    UnitInterval = Annotated[ValidFloat, Field(ge=0.0, le=1.0, strict=False, allow_inf_nan=False)]
    UnitIntervalArrayType = Annotated[tuple[UnitInterval, ...], Field(min_length=1, strict=False)]
    LaxUnitIntervalArrayType = Annotated[
        tuple[UnitInterval, ...],
        BeforeValidator(lambda v: v if _is_sequence(v) else (v,)),
        Field(min_length=1, strict=False),
    ]
    LaxFArrayType = Annotated[
        tuple[ValidFloat, ...],
        BeforeValidator(lambda v: v if _is_sequence(v) else (v,)),
        Field(min_length=1, strict=False),
    ]
    LaxPositiveFArrayType = Annotated[
        tuple[PositiveFloat, ...],
        BeforeValidator(lambda v: v if _is_sequence(v) else (v,)),
        Field(min_length=1, strict=False),
    ]
    UnitIntervalVec3Type = Annotated[tuple[UnitInterval, UnitInterval, UnitInterval], Field(strict=False)]
    UnitIntervalVec4Type = Annotated[tuple[UnitInterval, UnitInterval, UnitInterval, UnitInterval], Field(strict=False)]
    Vec4FType = Annotated[tuple[ValidFloat, ValidFloat, ValidFloat, ValidFloat], Field(strict=False)]
    StrArrayType = Annotated[tuple[str, ...], Field(strict=False)]
    Vec3FArrayType = Annotated[tuple[Vec3FType, ...], Field(min_length=1, strict=False)]
    UnitVec3FArrayType = Annotated[tuple[UnitVec3FType, ...], Field(min_length=1, strict=False)]
    Vec3FLaxArrayType = Annotated[
        tuple[Vec3FType, ...],
        BeforeValidator(lambda v: v if _is_sequence(v) and len(v) > 0 and _is_sequence(v[0]) else (v,)),
        Field(min_length=1, strict=False),
    ]
    UnitVec3FLaxArrayType = Annotated[
        tuple[UnitVec3FType, ...],
        BeforeValidator(lambda v: v if _is_sequence(v) and len(v) > 0 and _is_sequence(v[0]) else (v,)),
        Field(min_length=1, strict=False),
    ]
    RotationMatrixType = Annotated[
        tuple[UnitIntervalVec3Type, UnitIntervalVec3Type, UnitIntervalVec3Type], Field(strict=False)
    ]
    Matrix4x4Type = Annotated[
        tuple[Vec4FType, Vec4FType, Vec4FType, Vec4FType],
        BeforeValidator(lambda v: tuple(tuple(row) for row in v) if isinstance(v, np.ndarray) else v),
        Field(strict=False),
    ]
    NDArrayType = Annotated[
        np.ndarray, GetPydanticSchema(lambda tp, handler: core_schema.no_info_plain_validator_function(lambda v: v))
    ]
    PathType = Annotated[str, BeforeValidator(lambda v: str(v) if isinstance(v, PurePath) else v)]
