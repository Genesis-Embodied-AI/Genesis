from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import Field, StrictBool

import genesis as gs
from genesis.typing import (
    FArrayType,
    FGridType,
    IArrayType,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    UnitIntervalVec3Type,
    UnitIntervalVec4Type,
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


def _validate_filler_probe_radius(probe_radius, sensor_name: str) -> None:
    """
    Validate a ``probe_radius`` that permits 0-valued (inactive padding for grid) entries.
    """
    radii = np.atleast_1d(np.asarray(probe_radius, dtype=float))
    if np.any(radii < 0.0):
        gs.raise_exception(f"{sensor_name} probe_radius entries must be non-negative. Got {probe_radius}.")
    if not np.any(radii > 0.0):
        gs.raise_exception(f"{sensor_name} requires at least one positive probe_radius. Got {probe_radius}.")


class TactileProbeSensorOptionsMixin(ProbeSensorOptionsMixin[SensorT]):
    """
    Tactile probe sensors estimate contact from geometric depth queries (SDF or raycast) around each probe position
    rather than reading the physics solver's contact impulses, so they sense at arbitrary probe locations without
    affecting simulation.

    Parameters
    ----------
    debug_contact_color: array-like[float, float, float]
        RGB color of the debug probe spheres while in contact.
    contact_depth_query : {"sdf", "raycast"} or None, optional
        Per-probe contact-depth backend. ``"sdf"`` queries the per-geom analytic SDF grid (fast, exact for primitives,
        requires SDF activation). ``"raycast"`` walks the rigid solver's per-frame collision-mesh BVH and takes the
        signed distance to the nearest candidate triangle (sign from the triangle's face normal, negative inside),
        so ``pen = R - signed_distance`` matches the SDF backend while handling arbitrary meshes uniformly (shares
        the BVH with ``RaycasterSensor``). ``None`` (default) defers the choice: all sensors of the same class must
        agree, and the resolved mode is ``"sdf"`` if no sensor of that class sets it.
    """

    debug_contact_color: UnitIntervalVec3Type = (1.0, 0.2, 0.0)

    contact_depth_query: Literal["sdf", "raycast"] | None = None


class PointCloudTactileSensorMixin(TactileProbeSensorOptionsMixin[SensorT]):
    """
    Options mixin for tactile sensors that sample a point cloud from tracked link meshes.

    Parameters
    ----------
    track_link_idx : array-like[int]
        Global link indices whose mesh geometry is used to sample a point cloud from.
    n_sample_points: int | array-like[int]
        Total FPS samples split across ``track_link_idx``, or one count per tracked link. Per-variant
        counts are not supported: when a tracked link belongs to a heterogeneous entity, the per-link
        count is allocated to every variant on that link (so each parallel environment sees the full
        count regardless of which variant is active).
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

    Note
    ----
    The depth query only runs against geometry the rigid solver reports as in contact with the sensor's link (the
    depth itself comes from an SDF/raycast query, not the contact impulse, but contact *existence* is gated by the
    solver). Since the solver skips collision between two fixed entities, a sensor on a fixed entity will not detect
    contacts with other fixed entities.

    Parameters
    ----------
    probe_radius : float | array-like[float] or shape ``(M, N)`` grid
        Probe sensing radius in meters. A scalar is shared by every probe; an array (or grid) must match the
        layout of ``probe_local_pos``. Array entries of ``0`` mark inactive filler probes -- they always read
        ``False`` and skip the SDF query -- so an irregular taxel set can be padded into a regular grid.
    contact_threshold: float
        Penetration depth (meters) at or above which a probe latches into contact.
    release_threshold: float, optional
        Penetration depth (meters) at or below which a latched probe releases (Schmitt-trigger hysteresis). Must be
        ``<= contact_threshold``. Defaults to ``contact_threshold`` (no hysteresis).
    """

    # Permits 0-valued (inactive filler) entries; see _validate_filler_probe_radius.
    probe_radius: PositiveFloat | FArrayType | FGridType = 0.01

    contact_threshold: NonNegativeFloat = 0.0001
    release_threshold: NonNegativeFloat | None = None

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        _validate_filler_probe_radius(self.probe_radius, "ContactProbe")
        if self.release_threshold is not None and self.release_threshold > self.contact_threshold:
            gs.raise_exception(
                f"release_threshold ({self.release_threshold}) must be <= contact_threshold ({self.contact_threshold})."
            )


class ContactDepthProbe(
    RigidSensorOptionsMixin["ContactDepthProbeSensor"],
    SimpleSensorOptions["ContactDepthProbeSensor"],
    TactileProbeSensorOptionsMixin["ContactDepthProbeSensor"],
):
    """
    Returns contact depth in meters per probe.

    Note
    ----
    The depth query only runs against geometry the rigid solver reports as in contact with the sensor's link (the
    depth itself comes from an SDF/raycast query, not the contact impulse, but contact *existence* is gated by the
    solver). Since the solver skips collision between two fixed entities, a sensor on a fixed entity will not detect
    contacts with other fixed entities.
    """


class KinematicTaxel(
    RigidSensorOptionsMixin["KinematicTaxelSensor"],
    SimpleSensorOptions["KinematicTaxelSensor"],
    TactileProbeSensorOptionsMixin["KinematicTaxelSensor"],
):
    """
    A tactile sensor which estimates force and torque per taxel by querying contact depth within the radius of the
    probe positions along a rigid entity link and the relative velocity of the probe and the entity in contact.

    The force and torque are aligned with the contact surface normal ``n`` at each probe -- the SDF gradient in
    ``"sdf"`` mode, or the nearest-triangle face normal in ``"raycast"`` mode. The returned force is a spring-damper
    estimate based on contact depth and relative motion:
        v_n = dot(relative_velocity, n) * n
        v_t = relative_velocity - v_n
        s = penetration ** normal_exponent
        F = (normal_stiffness * s * n) + (normal_damping * s * dot(relative_velocity, n) * n) - (shear_scalar * v_t)
        T = cross(probe_local_pos, F) - twist_scalar * dot(relative_angular_velocity, n) * n
    as opposed to the actual impulse force on the link from the contact obtained from the physics solver.

    Note
    ----
    The depth query only runs against geometry the rigid solver reports as in contact with the sensor's link (the
    force/torque come from the spring-damper estimate above, not the contact impulse, but contact *existence* is
    gated by the solver). Since the solver skips collision between two fixed entities, a sensor on a fixed entity
    will not detect contacts with other fixed entities.

    ``probe_local_pos`` may be either an arbitrary set of probes with shape ``(N, 3)`` or a grid-shaped set with shape
    ``(M, N, 3)``. A probe whose ``probe_radius`` is 0 is treated as an inactive filler -- it reads 0 force/torque and
    is skipped -- so an irregular taxel set can be padded into a regular grid.

    Parameters
    ----------
    probe_radius : float | array-like[float]
        Probe sensing radius in meters. A scalar is shared by every probe; an array must match the probe count.
        Array entries of 0 mark inactive filler probes (see the grid note above); at least one must be positive.
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

    # Permits 0-valued (inactive filler) entries; see _validate_filler_probe_radius.
    probe_radius: PositiveFloat | FArrayType | FGridType = 0.01

    normal_stiffness: NonNegativeFloat = 1000.0
    normal_damping: NonNegativeFloat = 1.0
    normal_exponent: NonNegativeFloat = 1.0
    shear_scalar: NonNegativeFloat = 1.0
    twist_scalar: NonNegativeFloat = 1.0

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)

        _validate_filler_probe_radius(self.probe_radius, "KinematicTaxel")
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
    direct dilation path. Shear is computed directly. A probe whose ``probe_radius`` is 0 is treated as an inactive
    filler -- it reads 0 and is excluded from dilation/shear -- so an irregular taxel set can be padded into a
    regular grid for FFT acceleration.

    Parameters
    ----------
    probe_local_pos: array-like[array-like[float, float, float]], shape (N, 3) or (M, N, 3)
        Probe positions in link-local frame.
    probe_local_normal : array-like[float, float, float] or array-like[array-like[float, float, float]]
        Unit direction(s) in link-local frame: one normal for all probes, or one normal per probe matching
        ``probe_local_pos``.
    probe_radius : float | array-like[float]
        Probe sensing radius in meters. A scalar is shared by every probe; an array must match the probe count.
        Array entries of 0 mark inactive filler probes (see the grid note above); at least one must be positive.
    track_link_idx : array-like[int]
        Global rigid link indices whose collision geometry is queried by SDF and whose mesh is sampled for shear.
    n_sample_points: int | array-like[int]
        Total surface samples split across ``track_link_idx``, or one count per tracked link.
    lambda_d: float
        Gaussian falloff coefficient (in 1/m^2) for the dilation kernel ``exp(-lambda_d * r^2)`` that smears each
        in-contact probe's normal/tangential bulge across its neighbors. Larger values give sharper, more localized
        markers; smaller values smear the bulge across more probes.
    lambda_s: float
        Gaussian falloff coefficient (in 1/m^2) for the shear kernel ``exp(-lambda_s * r^2)`` that spreads each
        anchored tracked-surface point's tangential displacement to nearby probes. Larger values keep shear tightly
        local to the contact patch; smaller values produce a softer, more diffuse shear response.
    dilate_scale: float
        Scalar gain applied to dilation displacement.
    shear_scale: float
        Scalar gain applied to shear displacement.
    normal_exponent: float
        Exponent of the penetration-depth power law for the normal (out-of-plane) marker dilation: the normal
        bulge scales as ``depth ** normal_exponent``. Must be >= 1.0. Default ``2.0`` (the HydroShear quadratic
        normal response). Tangential dilation and shear stay linear in depth regardless of this value.
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

    # Permits 0-valued (inactive filler) entries; see _validate_filler_probe_radius.
    probe_radius: PositiveFloat | FArrayType | FGridType = 0.01

    lambda_d: NonNegativeFloat = 700.0
    lambda_s: NonNegativeFloat = 300.0
    dilate_scale: NonNegativeFloat = 1.0
    shear_scale: NonNegativeFloat = 1.0
    normal_exponent: NonNegativeFloat = 2.0

    elastomer_contact_sdf_enter: NonNegativeFloat = 1e-5
    elastomer_contact_sdf_exit: NonNegativeFloat = 1e-4

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        _validate_filler_probe_radius(self.probe_radius, "ElastomerTaxel")
        if len(self.track_link_idx) == 0:
            gs.raise_exception("ElastomerTaxel requires at least one tracked link in track_link_idx.")
        if self.normal_exponent < 1.0:
            gs.raise_exception(f"normal_exponent must be greater than or equal to 1.0. Got {self.normal_exponent}.")


class ProximityTaxel(
    RigidSensorOptionsMixin["ProximityTaxelSensor"],
    SimpleSensorOptions["ProximityTaxelSensor"],
    PointCloudTactileSensorMixin["ProximityTaxelSensor"],
    ProbesWithNormalSensorOptionsMixin["ProximityTaxelSensor"],
):
    """
    A tactile sensor which estimates force and torque per taxel from proximity to point clouds sampled on tracked
    meshes within a **spherical** sensing volume of nominal ``probe_radius`` around each taxel.

    For each taxel, every tracked point inside that sphere contributes a penetration depth ``P_i = R - ||p_i - o||``
    where ``R`` is the sensing radius. Normal force is aligned with ``probe_local_normal``; shear uses tangential
    relative velocity. Generic SimpleSensor imperfections (bias, resolution, etc.) still apply. Outputs are in
    link-local frame.

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
