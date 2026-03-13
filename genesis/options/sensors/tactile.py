from typing import TYPE_CHECKING, Annotated, Any, Sequence

from pydantic import Field

import genesis as gs
from genesis.typing import (
    NumericType,
    PositiveFloat,
    PositiveFArrayType,
    Vec3FArrayType,
    UnitIntervalVec4Type,
    UnitVec3FType,
    UnitVec3FArrayType,
    NonNegativeFloat,
)

from .options import NoisySensorOptionsMixin, RigidSensorOptionsMixin, SensorOptions


if TYPE_CHECKING:
    Vec3FGridType = Sequence[Sequence[Sequence[NumericType]]]
else:
    Vec3FGridType = Annotated[tuple[Vec3FArrayType, ...], Field(min_length=1, strict=False)]


class KinematicTactileSensorMixin(SensorOptions):
    """
    Parameters
    ----------
    probe_local_normal : array-like[array-like[float, float, float]]
        Probe sensing directions in link-local frame. Penetration is measured along this axis.
    probe_radius : float | array-like[float]
        Probe sensing radius in meters. Objects within this distance are detected. Default: 0.005 (5mm)
    debug_sphere_color: array-like[float, float, float, float]
        The color of the debug sphere. Defaults to (1.0, 0.5, 0.0, 0.4).
    debug_contact_color: array-like[float, float, float, float]
        The color of the debug contact. Defaults to (1.0, 0.2, 0.0, 0.8).
    """

    probe_local_normal: UnitVec3FArrayType | UnitVec3FType = (0.0, 0.0, 1.0)
    probe_radius: PositiveFArrayType | PositiveFloat = 0.005

    debug_sphere_color: UnitIntervalVec4Type = (1.0, 0.5, 0.0, 0.4)
    debug_contact_color: UnitIntervalVec4Type = (1.0, 0.2, 0.0, 0.8)


class KinematicContactProbe(RigidSensorOptionsMixin, NoisySensorOptionsMixin, KinematicTactileSensorMixin):
    """
    A tactile sensor which queries contact depth relative to given probe normals and within the radius of the probe
    positions along a rigid entity link.

    The returned force is an spring-like (kinematic) estimate based on contact depth, computed as
    F = stiffness * penetration * probe_normal, as opposed to the actual impulse force on the link from the contact
    obtained from the physics solver.

    Note
    ----
    If this sensor is attached to a fixed entity, it will not detect contacts with other fixed entities.

    Parameters
    ----------
    probe_local_pos : array-like[array-like[float, float, float]]
        Probe positions in link-local frame. One (x, y, z) per probe.
    stiffness : float
        User-defined coefficient for force estimation. Default: 1000.0.
    """

    probe_local_pos: Vec3FArrayType = [(0.0, 0.0, 0.0)]
    stiffness: NonNegativeFloat = 1000.0

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)

        if isinstance(self.probe_local_normal[0], Sequence) and len(self.probe_local_pos) != len(
            self.probe_local_normal
        ):
            gs.raise_exception(
                "probe_local_pos and probe_local_normal must have the same length. "
                f"Got {len(self.probe_local_pos)} positions and {len(self.probe_local_normal)} normals."
            )
        if isinstance(self.probe_radius, Sequence) and len(self.probe_radius) != len(self.probe_local_pos):
            gs.raise_exception(
                "If radius is array-like, it must have the same length as probe_local_pos. "
                f"Got {len(self.probe_radius)} radii and {len(self.probe_local_pos)} probe positions."
            )


class ElastomerDisplacement(
    RigidSensorOptionsMixin,
    NoisySensorOptionsMixin,
    KinematicTactileSensorMixin,
    SensorOptions,
):
    """
    A tactile sensor which estimates the displacement of the elastomer based on the contact force and depth of
    penetration along the probe normal on collisions.

    Note
    ----
    When probe_local_pos is a 2D array, the displacement is computed by FFT-based convolution over the grid.
    This provides a significant speedup when the number of probes is large.

    The equations for the displacement are as follows (from FOTS paper https://arxiv.org/pdf/2404.19217):
    dilate_displacement = Σ_i min(dilate_max_delta, penetration_depth) * (M - C_i) * exp(-λd ||M - C_i||²)
    shear_displacement = min(shear_max_delta, shear_velocity * dt) * exp(-λs ||M - G||²)
    twist_displacement = min(twist_max_delta, twist_angle) * (M - G) * exp(-λt ||M - G||²)

    Parameters
    ----------
    probe_local_pos: array-like[array-like[float, float, float]], shape (N, 3) or (M, N, 3)
        Probe positions in link-local frame.
    dilate_coefficient: float
        The coefficient of the exponential function that can affect the displacement caused by dilate motion.
    shear_coefficient: float
        The coefficient of the exponential function that can affect the displacement caused by shear motion.
    twist_coefficient: float
        The coefficient of the exponential function that can affect the displacement caused by twist motion.
    dilate_max_delta: float
        The maximum dilate depth in meters.
    shear_max_delta: float
        The maximum shear magnitude in meters.
    twist_max_delta: float
        The maximum twist angle in degrees.
    """

    probe_local_pos: Vec3FArrayType | Vec3FGridType = ((0.0, 0.0, 0.0),)
    dilate_coefficient: NonNegativeFloat = 1e-2
    shear_coefficient: NonNegativeFloat = 1e-2
    twist_coefficient: NonNegativeFloat = 1e-2
    dilate_max_delta: NonNegativeFloat = 0.1
    shear_max_delta: NonNegativeFloat = 0.1
    twist_max_delta: NonNegativeFloat = 50.0

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)

        num_prob = len(self.probe_local_pos)
        if isinstance(self.probe_local_pos[0][0], Sequence):
            num_prob *= len(self.probe_local_pos[0])

        if isinstance(self.probe_local_normal[0], Sequence) and len(self.probe_local_normal) != num_prob:
            gs.raise_exception(
                "probe_local_pos and probe_local_normal must have the same length. "
                f"Got {num_prob} positions and {len(self.probe_local_normal)} normals."
            )
        if isinstance(self.probe_radius, Sequence) and len(self.probe_radius) != num_prob:
            gs.raise_exception(
                "If radius is array-like, it must have the same length as probe_local_pos. "
                f"Got {len(self.probe_radius)} radii and {num_prob} probe positions."
            )
