from typing import TYPE_CHECKING, Annotated, Any, Sequence

from pydantic import Field, StrictBool

import genesis as gs
from genesis.typing import (
    IArrayType,
    NonNegativeFloat,
    NonNegativeInt,
    NumericType,
    PositiveFloat,
    PositiveInt,
    UnitIntervalVec4Type,
    UnitVec3FArrayType,
    UnitVec3FType,
    Vec3FArrayType,
)

from .options import (
    ProbeSensorOptionsMixin,
    ProbesWithNormalSensorOptionsMixin,
    RigidSensorOptionsMixin,
    SensorT,
    SimpleSensorOptions,
)

if TYPE_CHECKING:
    from genesis.engine.sensors.kinematic_tactile import (
        ContactDepthProbeSensor,
        ContactProbeSensor,
        ElastomerTaxelSensor,
        KinematicTaxelSensor,
        ProximityTaxelSensor,
    )

    Vec3FGridType = Sequence[Sequence[Sequence[NumericType]]]
    UnitVec3FGridType = Sequence[Sequence[Sequence[NumericType]]]
else:
    Vec3FGridType = Annotated[tuple[Vec3FArrayType, ...], Field(min_length=1, strict=False)]
    UnitVec3FGridType = Annotated[tuple[UnitVec3FArrayType, ...], Field(min_length=1, strict=False)]


class TactileProbeSensorOptionsMixin(ProbeSensorOptionsMixin[SensorT]):
    """
    Tactile probe sensors use SDF contact-depth queries around each probe position instead of physics solver
    contact impulses. This allows fast contact sensing at arbitrary probe locations without affecting simulation.

    Note
    ----
    If this sensor is attached to a fixed entity, it will not detect contacts with other fixed entities.

    Parameters
    ----------
    debug_contact_color: array-like[float, float, float, float]
        The color of the debug contact. Defaults to (1.0, 0.2, 0.0, 0.8).
    """

    debug_contact_color: UnitIntervalVec4Type = (1.0, 0.2, 0.0, 0.8)


class PointCloudTactileSensorMixin(TactileProbeSensorOptionsMixin[SensorT]):
    """
    Parameters
    ----------
    track_link_idx : array-like[int]
        Global link indices whose mesh geometry is used to sample a point cloud from.
    n_sample_points: int | array-like[int]
        Total FPS samples split across ``track_link_idx``, or one count per tracked link.
    use_visual_mesh : bool
        Whether to use the visual mesh when sampling the point cloud.
    debug_point_cloud_color : array-like[float, float, float, float]
        The rgba color of the debug tracked object point cloud spheres.
    debug_point_cloud_radius : float
        The radius of the debug tracked object point cloud spheres.
    """

    track_link_idx: IArrayType = Field(default_factory=tuple)
    n_sample_points: IArrayType | NonNegativeInt = 500
    use_visual_mesh: StrictBool = True

    debug_point_cloud_color: UnitIntervalVec4Type = (1.0, 0.8, 0.0, 1.0)
    debug_point_cloud_radius: PositiveFloat = 0.002


class ContactProbe(
    RigidSensorOptionsMixin["ContactProbeSensor"],
    SimpleSensorOptions["ContactProbeSensor"],
    TactileProbeSensorOptionsMixin["ContactProbeSensor"],
):
    """
    Returns boolean contact per probe based on the contact depth threshold.

    Parameters
    ----------
    contact_threshold: float
        A probe is considered in contact if the penetration depth is greater than or equal to this threshold (meters).
    """

    contact_threshold: NonNegativeFloat = 0.0001


class ContactDepthProbe(
    RigidSensorOptionsMixin["ContactDepthProbeSensor"],
    SimpleSensorOptions["ContactDepthProbeSensor"],
    TactileProbeSensorOptionsMixin["ContactDepthProbeSensor"],
):
    """
    Returns contact depth in meters per probe.
    """


class KinematicTaxel(
    RigidSensorOptionsMixin["KinematicTaxelSensor"],
    SimpleSensorOptions["KinematicTaxelSensor"],
    TactileProbeSensorOptionsMixin["KinematicTaxelSensor"],
    ProbesWithNormalSensorOptionsMixin["KinematicTaxelSensor"],
):
    """
    A tactile sensor which estimates force and torque per taxel by querying contact depth relative to given probe
    normals and within the radius of the probe positions along a rigid entity link and the relative velocity of the
    probe and the entity in contact.

    The returned force is a spring-damper estimate based on contact depth and relative motion:
        v_n = dot(relative_velocity, probe_normal) * probe_normal
        v_t = relative_velocity - v_n
        s = penetration ** normal_exponent
        F = (-normal_stiffness * s * probe_normal) - (normal_damping * s * v_n) - (shear_scalar * v_t)
        T = cross(probe_local_pos, F) - twist_scalar * dot(relative_angular_velocity, probe_normal) * probe_normal
    as opposed to the actual impulse force on the link from the contact obtained from the physics solver.

    Note
    ----
    If this sensor is attached to a fixed entity, it will not detect contacts with other fixed entities.

    Parameters
    ----------
    normal_stiffness : float
        Stiffness for normal force estimation based on contact penetration depth and spring-damper model.
    normal_damping : float
        Damping for normal force estimation based on contact penetration depth and spring-damper model.
    normal_exponent : float
        Exponent for contact force estimation based on contact penetration depth and nonlinear spring-damper model.
        Default is 1.0, which means linear spring-damper model. Use 1.5 for Hertzian (spherical) contact.
    shear_scalar : float, optional
        Coefficient for shear force estimation based on relative linear velocity of the probe and entity in contact.
    twist_scalar : float, optional
        Coefficient for twist torque estimation based on relative angular velocity of the probe and entity in contact.
    """

    normal_stiffness: NonNegativeFloat = 1000.0
    normal_damping: NonNegativeFloat = 1.0
    normal_exponent: NonNegativeFloat = 1.0
    shear_scalar: NonNegativeFloat = 1.0
    twist_scalar: NonNegativeFloat = 1.0

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)

        if self.normal_exponent < 1.0:
            gs.raise_exception(f"normal_exponent must be greater than or equal to 1.0. Got {self.normal_exponent}.")


class ElastomerTaxel(
    RigidSensorOptionsMixin["ElastomerTaxelSensor"],
    SimpleSensorOptions["ElastomerTaxelSensor"],
    PointCloudTactileSensorMixin["ElastomerTaxelSensor"],
    ProbesWithNormalSensorOptionsMixin["ElastomerTaxelSensor"],
):
    """
    An elastomer tactile sensor that implements HydroShear-style marker displacement from Genesis SDF queries.
    The tracked rigid links are sampled into indenter on-surface points for shear history, while marker dilation is
    queried directly from the tracked geometry SDF.

    Note
    ----
    ``probe_local_pos`` may be either an arbitrary set of probes with shape ``(N, 3)`` or a grid-shaped set with shape
    ``(M, N, 3)``. Regular planar grids with one shared normal use FFT acceleration for dilation; other layouts use the
    direct dilation path. Shear is computed directly.

    Parameters
    ----------
    probe_local_pos: array-like[array-like[float, float, float]], shape (N, 3) or (M, N, 3)
        Probe positions in link-local frame.
    probe_local_normal : array-like[float, float, float] or array-like[array-like[float, float, float]]
        Unit direction(s) in link-local frame: one normal for all probes, or one normal per probe matching
        ``probe_local_pos``.
    track_link_idx : array-like[int]
        Global rigid link indices whose collision geometry is queried by SDF and whose mesh is sampled for shear.
    n_sample_points: int | array-like[int]
        Total surface samples split across ``track_link_idx``, or one count per tracked link.
    lambda_d: float
        Exponential coefficient for dilation spread.
    lambda_s: float
        Exponential coefficient for shear spread from tracked surface points.
    dilate_scale: float
        Scalar gain applied to dilation displacement.
    shear_scale: float
        Scalar gain applied to shear displacement.
    elastomer_contact_sdf_enter: float
        Positive margin on signed distance: a tracked surface point starts anchoring shear when its elastomer SDF
        value is below ``-elastomer_contact_sdf_enter``.
    elastomer_contact_sdf_exit: float
        Positive margin: the anchor clears when the elastomer SDF value rises above ``+elastomer_contact_sdf_exit``
        (hysteresis band between enter and exit reduces chatter).

    Note
    ----
    Genesis reuses rigid-body SDFs for HydroShear queries. For non-analytic tracked meshes, the collision geometry
    should be watertight enough for signed-distance preprocessing, and the attached elastomer link's collision geometry
    should represent the compliant contact surface.
    """

    probe_local_pos: Vec3FArrayType | Vec3FGridType = ((0.0, 0.0, 0.0),)
    probe_local_normal: UnitVec3FArrayType | UnitVec3FGridType | UnitVec3FType = (0.0, 0.0, 1.0)

    lambda_d: NonNegativeFloat = 700.0
    lambda_s: NonNegativeFloat = 300.0
    dilate_scale: NonNegativeFloat = 1.0
    shear_scale: NonNegativeFloat = 1.0

    elastomer_contact_sdf_enter: NonNegativeFloat = 1e-5
    elastomer_contact_sdf_exit: NonNegativeFloat = 1e-4

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if len(self.track_link_idx) == 0:
            gs.raise_exception("ElastomerTaxel requires at least one tracked link in track_link_idx.")


class ProximityTaxel(
    RigidSensorOptionsMixin["ProximityTaxelSensor"],
    SimpleSensorOptions["ProximityTaxelSensor"],
    PointCloudTactileSensorMixin["ProximityTaxelSensor"],
    ProbesWithNormalSensorOptionsMixin["ProximityTaxelSensor"],
):
    """
    A tactile sensor which estimates force and torque per taxel from proximity to point clouds sampled on tracked
    meshes within a **spherical** sensing volume of nominal ``probe_radius`` around each taxel.

    For each taxel, every tracked point inside that sphere contributes a penetration depth ``P_i = R_eff - ||p_i - o||``
    where ``R_eff`` is drawn each simulation step when ``probe_radius_noise`` is non-zero (additive uniform noise
    in meters around the sensing radius, clipped nonnegative). Normal force is aligned with ``probe_local_normal``;
    shear uses tangential relative velocity. Generic SimpleSensor imperfections (bias, resolution, etc.) still apply.
    Outputs are in link-local frame.

    Parameters
    ----------
    probe_local_normal : array-like[array-like[float, float, float]]
        Unit direction(s) for the normal force channel in link-local frame: one ``(3,)`` for all taxels, or one row per
        taxel matching ``probe_local_pos``. Default ``(0, 0, 1)``.
    stiffness : float
        Linear spring stiffness (N/m) scaling summed penetration depths into total reported force.
    shear_coupling : float
        Scales penetration-weighted tangential slip ``sum_i P_i * v_{t,i}`` into a shear force contribution (see
        sensor documentation). Set to ``0.0`` to disable shear and use only the normal channel.
    density_scalar : int
        Reference point count for normalizing summed penetrations against tracked cloud size
        (scale is ``density_scalar / max(N_pc, 1)`` for this sensor's tracked samples).
    """

    stiffness: NonNegativeFloat = 100.0
    shear_coupling: NonNegativeFloat = 0.0
    density_scalar: PositiveInt = 100
