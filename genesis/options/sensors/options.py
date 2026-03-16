from typing import TYPE_CHECKING, Annotated, Any, Generic, TypeVar

import numpy as np
from pydantic import BeforeValidator, Field, StrictBool, StrictInt, model_validator

import genesis as gs
from genesis.typing import (
    FArrayType,
    IArrayType,
    LaxVec3FType,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    RotationMatrixType,
    UnitIntervalVec3Type,
    UnitIntervalVec4Type,
    Vec3FArrayType,
    Vec3FType,
    Vec4FType,
    _is_sequence,
)

from ..options import Options
from .raycaster import DepthCameraPattern, RaycastPattern

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.sensors.base_sensor import Sensor
    from genesis.engine.sensors.contact_force import ContactForceSensor, ContactSensor
    from genesis.engine.sensors.imu import IMUSensor
    from genesis.engine.sensors.proximity import ProximitySensor
    from genesis.engine.sensors.raycaster import RaycasterSensor


SensorT = TypeVar("SensorT", bound="Sensor")


if TYPE_CHECKING:
    NonNegativeUnboundedFloat = float
    LaxNonNegativeUnboundedVec3FType = Vec3FType | float
else:
    NonNegativeUnboundedFloat = Annotated[float, Field(ge=0, strict=False)]
    LaxNonNegativeUnboundedVec3FType = Annotated[
        tuple[NonNegativeUnboundedFloat, NonNegativeUnboundedFloat, NonNegativeUnboundedFloat],
        BeforeValidator(lambda v: v if _is_sequence(v) else (v,) * 3),
        Field(strict=False),
    ]
CrossCouplingAxisType = RotationMatrixType | UnitIntervalVec3Type | float


class SensorOptions(Options, Generic[SensorT]):
    """
    Base class for all sensor options.

    Each sensor should have their own options class that inherits from this class.
    The associated sensor class registers itself via ``Sensor.__init_subclass__`` when parameterized
    with this options class, e.g. ``class MySensor(Sensor[MyOptions, MyMetadata, MyData]): ...``

    Parameters
    ----------
    delay : float
        The read delay time in seconds. Data read will be outdated by this amount. Defaults to 0.0 (no delay).
    update_ground_truth_only : bool
        If True, the sensor will only update the ground truth data, and not the measured data. Defaults to False.
    draw_debug : bool
        If True and visualizer is active, the sensor will draw debug shapes in the scene. Defaults to False.
    """

    delay: NonNegativeFloat = 0.0
    update_ground_truth_only: StrictBool = False
    draw_debug: StrictBool = False

    def validate_scene(self, scene: "Scene"):
        """
        Validate the sensor options values before the sensor is added to the scene.

        Use pydantic's model_post_init() for validation that does not require scene context.
        """
        assert scene.sim is not None
        delay_hz = self.delay / scene.sim.dt
        if not np.isclose(delay_hz, round(delay_hz), atol=gs.EPS):
            gs.logger.warning(
                f"{type(self).__name__}: Read delay should be a multiple of the simulation time step. Got {self.delay}"
                f" and {scene.sim.dt}. Actual read delay will be {1 / round(delay_hz)}."
            )


class RigidSensorOptionsMixin(SensorOptions[SensorT]):
    """
    Base options class for sensors that are attached to a RigidEntity.

    Parameters
    ----------
    entity_idx : int
        The global entity index of the RigidEntity to which this sensor is attached. -1 or None for static sensors.
    link_idx_local : int, optional
        The local index of the RigidLink of the RigidEntity to which this sensor is attached.
    pos_offset : array-like[float, float, float], optional
        The positional offset of the sensor from the RigidLink.
    euler_offset : array-like[float, float, float], optional
        The rotational offset of the sensor from the RigidLink in degrees.
    """

    entity_idx: StrictInt | None = Field(default=-1, ge=-1)
    link_idx_local: NonNegativeInt = 0
    pos_offset: Vec3FType = (0.0, 0.0, 0.0)
    euler_offset: Vec3FType = (0.0, 0.0, 0.0)

    def validate_scene(self, scene: "Scene"):
        from genesis.engine.entities import RigidEntity

        super().validate_scene(scene)
        if self.entity_idx is not None and self.entity_idx >= 0:
            if self.entity_idx >= len(scene.entities):
                gs.raise_exception(f"Invalid RigidEntity index {self.entity_idx}.")
            entity = scene.entities[self.entity_idx]
            if not isinstance(entity, RigidEntity):
                gs.raise_exception(f"Entity at index {self.entity_idx} is not a RigidEntity.")
            if self.link_idx_local >= entity.n_links:
                gs.raise_exception(f"Invalid RigidLink index {self.link_idx_local} for entity {self.entity_idx}.")


class NoisySensorOptionsMixin(SensorOptions[SensorT]):
    """
    Base options class for analog sensors that are attached to a RigidEntity.

    Parameters
    ----------
    resolution : float | array-like[float, ...], optional
        The measurement resolution of the sensor (smallest increment of change in the sensor reading).
        Default is 0.0, which means no quantization is applied.
    bias : float | array-like[float, ...], optional
        The constant additive bias of the sensor.
    noise : float | array-like[float, ...], optional
        The standard deviation of the additive white noise.
    random_walk : float | array-like[float, ...], optional
        The standard deviation of the random walk, which acts as accumulated bias drift.
    jitter : float, optional
        The jitter in seconds modeled as a a random additive delay sampled from a normal distribution.
        Jitter cannot be greater than delay. `interpolate` should be True when `jitter` is greater than 0.
    interpolate : bool, optional
        If True, the sensor data is interpolated between data points for delay + jitter.
        Otherwise, the sensor data at the closest time step will be used. Default is False.
    """

    resolution: FArrayType | float = 0.0
    bias: FArrayType | float = 0.0
    noise: FArrayType | float = 0.0
    random_walk: FArrayType | float = 0.0
    jitter: NonNegativeFloat = 0.0
    interpolate: StrictBool = False

    def model_post_init(self, context: Any) -> None:
        if self.jitter > 0 and not self.interpolate:
            gs.raise_exception(f"{type(self).__name__}: `interpolate` should be True when `jitter` is greater than 0.")
        if self.jitter > self.delay:
            gs.raise_exception(f"{type(self).__name__}: Jitter must be less than or equal to read delay.")


class Contact(RigidSensorOptionsMixin["ContactSensor"]):
    """
    Sensor that returns bool based on whether associated RigidLink is in contact.

    Parameters
    ----------
    debug_sphere_radius : float, optional
        The radius of the debug sphere. Defaults to 0.05.
    debug_color : array-like[float, float, float, float], optional
        The rgba color of the debug sphere. Defaults to (1.0, 0.0, 1.0, 0.5).
    """

    debug_sphere_radius: PositiveFloat = 0.05
    debug_color: UnitIntervalVec4Type = (1.0, 0.0, 1.0, 0.5)


class ContactForce(RigidSensorOptionsMixin["ContactForceSensor"], NoisySensorOptionsMixin["ContactForceSensor"]):
    """
    Sensor that returns the total contact force being applied to the associated RigidLink in its local frame.

    Parameters
    ----------
    min_force : float | array-like[float, float, float], optional
        The minimum detectable absolute force per each axis. Values below this will be treated as 0. Default is 0.
    max_force : float | array-like[float, float, float], optional
        The maximum output absolute force per each axis. Values above this will be clipped. Default is infinity.
    debug_color : array-like[float, float, float, float], optional
        The rgba color of the debug arrow. Defaults to (1.0, 0.0, 1.0, 0.5).
    debug_scale : float, optional
        The scale factor for the debug force arrow. Defaults to 0.01.
    """

    resolution: LaxVec3FType = 0.0

    min_force: LaxNonNegativeUnboundedVec3FType = 0.0
    max_force: LaxNonNegativeUnboundedVec3FType = np.inf

    debug_color: UnitIntervalVec4Type = (1.0, 0.0, 1.0, 0.5)
    debug_scale: PositiveFloat = 0.01

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if np.any(np.array(self.max_force) <= np.array(self.min_force)):
            gs.raise_exception(f"min_force should be less than max_force, got: {self.min_force} and {self.max_force}")


class IMU(RigidSensorOptionsMixin["IMUSensor"], NoisySensorOptionsMixin["IMUSensor"]):
    """
    IMU sensor returns the linear acceleration (accelerometer) and angular velocity (gyroscope)
    of the associated entity link.

    Parameters
    ----------
    acc_resolution : float, optional
        The measurement resolution of the accelerometer (smallest increment of change in the sensor reading).
        Default is 0.0, which means no quantization is applied.
    acc_cross_axis_coupling : float | array-like[float, float, float] | array-like with shape (3,3)
        Accelerometer axes alignment as a 3x3 rotation matrix, where diagonal elements represent alignment (0.0 to 1.0)
        for each axis, and off-diagonal elements account for cross-axis misalignment effects.
        - If a scalar is provided (float), all off-diagonal elements are set to the scalar value.
        - If a 3-element vector is provided (array-like[float, float, float]), off-diagonal elements are set.
        - If a full 3x3 matrix is provided, it is used directly.
    acc_bias : array-like[float, float, float]
        The constant additive bias for each axis of the accelerometer.
    acc_noise : array-like[float, float, float]
        The standard deviation of the white noise for each axis of the accelerometer.
    acc_random_walk : array-like[float, float, float]
        The standard deviation of the random walk, which acts as accumulated bias drift.
    gyro_resolution : float, optional
        The measurement resolution of the gyroscope (smallest increment of change in the sensor reading).
        Default is 0.0, which means no quantization is applied.
    gyro_cross_axis_coupling : float | array-like[float, float, float] | array-like with shape (3,3)
        Gyroscope axes alignment as a 3x3 rotation matrix, similar to `acc_cross_axis_coupling`.
    gyro_bias : array-like[float, float, float]
        The constant additive bias for each axis of the gyroscope.
    gyro_noise : array-like[float, float, float]
        The standard deviation of the white noise for each axis of the gyroscope.
    gyro_random_walk : array-like[float, float, float]
        The standard deviation of the bias drift for each axis of the gyroscope.
    mag_resolution : float, optional
        The measurement resolution of the magnetometer (smallest increment of change in the sensor reading).
        Default is 0.0, which means no quantization is applied.
    mag_cross_axis_coupling : float | array-like[float, float, float] | array-like with shape (3,3)
        Magnetometer axes alignment as a 3x3 rotation matrix, similar to `acc_cross_axis_coupling`.
    mag_bias : array-like[float, float, float]
        The constant additive bias for each axis of the magnetometer.
    mag_noise : array-like[float, float, float]
        The standard deviation of the white noise for each axis of the gyroscope.
    mag_random_walk : array-like[float, float, float]
        The standard deviation of the bias drift for each axis of the magnetometer.
    debug_acc_color : array-like[float, float, float, float], optional
        The rgba color of the debug acceleration arrow. Defaults to (1.0, 0.0, 0.0, 0.6).
    debug_acc_scale: float, optional
        The scale factor for the debug acceleration arrow. Defaults to 0.01.
    debug_gyro_color : array-like[float, float, float, float], optional
        The rgba color of the debug gyroscope arrow. Defaults to (0.0, 1.0, 0.0, 0.6).
    debug_gyro_scale: float, optional
        The scale factor for the debug gyroscope arrow. Defaults to 0.01.
    debug_mag_color : array-like[float, float, float, float], optional
        The rgba color of the debug magnetometer arrow. Defaults to (0.0, 0.0, 1.0, 0.6).
    debug_mag_scale: float, optional
        The scale factor for the debug magnetometer arrow. Defaults to 0.01.
    """

    # Accelerometer
    acc_resolution: LaxVec3FType = 0.0
    acc_cross_axis_coupling: CrossCouplingAxisType = 0.0
    acc_noise: LaxVec3FType = 0.0
    acc_bias: LaxVec3FType = 0.0
    acc_random_walk: LaxVec3FType = 0.0

    # Gyroscope
    gyro_resolution: LaxVec3FType = 0.0
    gyro_cross_axis_coupling: CrossCouplingAxisType = 0.0
    gyro_noise: LaxVec3FType = 0.0
    gyro_bias: LaxVec3FType = 0.0
    gyro_random_walk: LaxVec3FType = 0.0

    # Magnetometer (New)
    mag_resolution: LaxVec3FType = 0.0
    mag_cross_axis_coupling: CrossCouplingAxisType = 0.0
    mag_noise: LaxVec3FType = 0.0
    mag_bias: LaxVec3FType = 0.0
    mag_random_walk: LaxVec3FType = 0.0
    magnetic_field: LaxVec3FType = (0.0, 0.0, 0.5)

    debug_acc_color: UnitIntervalVec4Type = (1.0, 0.0, 0.0, 0.6)
    debug_acc_scale: PositiveFloat = 0.01
    debug_gyro_color: UnitIntervalVec4Type = (0.0, 1.0, 0.0, 0.6)
    debug_gyro_scale: PositiveFloat = 0.01
    debug_mag_color: UnitIntervalVec4Type = (0.0, 0.0, 1.0, 0.6)
    debug_mag_scale: PositiveFloat = 0.5

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)

        # FIXME: Resolution should be made private or converted to properties in mixin to prevent setting them directly
        self.resolution = self.acc_resolution + self.gyro_resolution + self.mag_resolution
        self.bias = self.acc_bias + self.gyro_bias + self.mag_bias
        self.random_walk = self.acc_random_walk + self.gyro_random_walk + self.mag_random_walk
        self.noise = self.acc_noise + self.gyro_noise + self.mag_noise


class ProximityOptions(RigidSensorOptionsMixin["ProximitySensor"], NoisySensorOptionsMixin["ProximitySensor"]):
    """
    Proximity sensor that reports distance and nearest point from probe positions to tracked mesh surfaces.

    Attached to a rigid entity link. Takes a list of local probe positions and a list of global link indices
    to track; for each probe, outputs the distance and nearest point (world frame) to the closest mesh
    surface among the tracked links. If no mesh is within max_range, reports max_range and the probe
    position as nearest point.

    Parameters
    ----------
    probe_local_pos : array-like[array-like[float, float, float]]
        Probe positions in link-local frame. One (x, y, z) per probe.
    track_link_idx : array-like[int]
        Global link indices (solver link space) whose mesh geoms are used for distance queries.
    max_range : float
        Maximum reporting range in meters. When no mesh is within this distance, distance is
        clamped to max_range and nearest points is the probe position. Default: 10.0.
    debug_sphere_radius: float, optional
        The radius of each debug sphere drawn in the scene. Defaults to 0.008.
    debug_color: array-like[float, float, float, float], optional
        The rgba color of the debug sphere. Defaults to (0.2, 0.6, 1.0, 0.6).
    """

    probe_local_pos: Vec3FArrayType = [(0.0, 0.0, 0.0)]
    track_link_idx: IArrayType = Field(default_factory=tuple)
    max_range: PositiveFloat = 10.0

    debug_sphere_radius: PositiveFloat = 0.008
    debug_color: UnitIntervalVec4Type = (0.2, 0.6, 1.0, 0.6)

    def validate_scene(self, scene: "Scene"):
        super().validate_scene(scene)
        n_links = scene.sim.rigid_solver.n_links
        for i, link_idx in enumerate(self.track_link_idx):
            if not (0 <= link_idx < n_links):
                gs.raise_exception(f"ProximityOptions.track_link_idx[{i}]={link_idx} is out of range [0, {n_links}).")


class Raycaster(RigidSensorOptionsMixin["RaycasterSensor"]):
    """
    Raycaster sensor that performs ray casting to get distance measurements and point clouds.

    Parameters
    ----------
    pattern: RaycastPatternOptions
        The raycasting pattern for the sensor.
    min_range : float, optional
        The minimum sensing range in meters. Defaults to 0.0.
    max_range : float, optional
        The maximum sensing range in meters. Defaults to 20.0.
    no_hit_value : float, optional
        The value to return for no hit. Defaults to max_range if not specified.
    return_world_frame : bool, optional
        Whether to return points in the world frame. Defaults to False (local frame).
    debug_sphere_radius: float, optional
        The radius of each debug sphere drawn in the scene. Defaults to 0.02.
    debug_ray_start_color: array-like[float, float, float, float], optional
        The color of each debug ray start sphere drawn in the scene. Defaults to (0.5, 0.5, 1.0, 1.0).
    debug_ray_hit_color: array-like[float, float, float, float], optional
        The color of each debug ray hit point sphere drawn in the scene. Defaults to (1.0, 0.5, 0.5, 1.0).
    """

    pattern: RaycastPattern
    min_range: NonNegativeFloat = 0.0
    max_range: PositiveFloat = 20.0
    no_hit_value: float
    return_world_frame: StrictBool = False

    debug_sphere_radius: PositiveFloat = 0.02
    debug_ray_start_color: Vec4FType = (0.5, 0.5, 1.0, 1.0)
    debug_ray_hit_color: Vec4FType = (1.0, 0.5, 0.5, 1.0)

    @model_validator(mode="before")
    @classmethod
    def default_no_hit_value(cls, data: dict) -> dict:
        if "no_hit_value" not in data:
            data["no_hit_value"] = data.get("max_range", cls.model_fields["max_range"].default)
        return data

    def model_post_init(self, context: Any) -> None:
        if self.max_range <= self.min_range:
            gs.raise_exception(
                f"[{type(self).__name__}] max_range {self.max_range} should be greater than min_range {self.min_range}."
            )


class DepthCamera(Raycaster):
    """
    Depth camera that uses ray casting to obtain depth images.

    Parameters
    ----------
    pattern: DepthCameraPattern
        The raycasting pattern configuration for the sensor.
    """

    pattern: DepthCameraPattern
