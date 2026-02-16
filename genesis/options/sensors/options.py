from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field, conlist

import genesis as gs

from ..options import Options
from .raycaster import DepthCameraPattern, RaycastPattern

Vec3FType = conlist(float, min_length=3, max_length=3)
Vec4FType = conlist(float, min_length=4, max_length=4)
Vec3FArrayType = conlist(Vec3FType, min_length=1)
FArrayType = conlist(float, min_length=1)
MaybeVec3FType = float | Vec3FType
Matrix3x3Type = conlist(conlist(float, min_length=3, max_length=3), min_length=3, max_length=3)
MaybeMatrix3x3Type = Matrix3x3Type | MaybeVec3FType

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


class SensorOptions(Options):
    """
    Base class for all sensor options.

    Each sensor should have their own options class that inherits from this class.
    The options class should be registered with the SensorManager using the @register_sensor decorator.

    Parameters
    ----------
    delay : float
        The read delay time in seconds. Data read will be outdated by this amount. Defaults to 0.0 (no delay).
    update_ground_truth_only : bool
        If True, the sensor will only update the ground truth data, and not the measured data. Defaults to False.
    draw_debug : bool
        If True and visualizer is active, the sensor will draw debug shapes in the scene. Defaults to False.
    """

    delay: float = 0.0
    update_ground_truth_only: bool = False
    draw_debug: bool = False

    def validate(self, scene: "Scene"):
        """
        Validate the sensor options values before the sensor is added to the scene.

        Use pydantic's model_post_init() for validation that does not require scene context.
        """
        delay_hz = self.delay / scene._sim.dt
        if not np.isclose(delay_hz, round(delay_hz), atol=gs.EPS):
            gs.logger.warning(
                f"{type(self).__name__}: Read delay should be a multiple of the simulation time step. Got {self.delay}"
                f" and {scene._sim.dt}. Actual read delay will be {1 / round(delay_hz)}."
            )


class RigidSensorOptionsMixin:
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

    entity_idx: int | None = -1
    link_idx_local: int = 0
    pos_offset: Vec3FType = (0.0, 0.0, 0.0)
    euler_offset: Vec3FType = (0.0, 0.0, 0.0)

    def validate(self, scene: "Scene"):
        from genesis.engine.entities import RigidEntity

        super().validate(scene)
        if self.entity_idx is not None and self.entity_idx >= len(scene.entities):
            gs.raise_exception(f"Invalid RigidEntity index {self.entity_idx}.")
        if self.entity_idx is not None and self.entity_idx >= 0:
            entity = scene.entities[self.entity_idx]
            if not isinstance(entity, RigidEntity):
                gs.raise_exception(f"Entity at index {self.entity_idx} is not a RigidEntity.")
            if self.link_idx_local < 0 or self.link_idx_local >= entity.n_links:
                gs.raise_exception(f"Invalid RigidLink index {self.link_idx_local} for entity {self.entity_idx}.")


class NoisySensorOptionsMixin:
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

    resolution: float | FArrayType = 0.0
    bias: float | FArrayType = 0.0
    noise: float | FArrayType = 0.0
    random_walk: float | FArrayType = 0.0
    jitter: float = 0.0
    interpolate: bool = False

    def model_post_init(self, _):
        if self.jitter > 0 and not self.interpolate:
            gs.raise_exception(f"{type(self).__name__}: `interpolate` should be True when `jitter` is greater than 0.")
        if self.jitter > self.delay:
            gs.raise_exception(f"{type(self).__name__}: Jitter must be less than or equal to read delay.")


class Contact(RigidSensorOptionsMixin, SensorOptions):
    """
    Sensor that returns bool based on whether associated RigidLink is in contact.

    Parameters
    ----------
    debug_sphere_radius : float, optional
        The radius of the debug sphere. Defaults to 0.05.
    debug_color : array-like[float, float, float, float], optional
        The rgba color of the debug sphere. Defaults to (1.0, 0.0, 1.0, 0.5).
    """

    debug_sphere_radius: float = 0.05
    debug_color: Vec4FType = (1.0, 0.0, 1.0, 0.5)


class ContactForce(RigidSensorOptionsMixin, NoisySensorOptionsMixin, SensorOptions):
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

    min_force: MaybeVec3FType = 0.0
    max_force: MaybeVec3FType = np.inf

    debug_color: Vec4FType = (1.0, 0.0, 1.0, 0.5)
    debug_scale: float = 0.01

    def model_post_init(self, _):
        if not (isinstance(self.min_force, float) or len(self.min_force) == 3):
            gs.raise_exception(f"min_force must be a float or array-like of 3 floats, got: {self.min_force}")
        if not (isinstance(self.max_force, float) or len(self.max_force) == 3):
            gs.raise_exception(f"max_force must be a float or array-like of 3 floats, got: {self.max_force}")
        if np.any(np.array(self.min_force) < 0):
            gs.raise_exception(f"min_force must be non-negative, got: {self.min_force}")
        if np.any(np.array(self.max_force) <= np.array(self.min_force)):
            gs.raise_exception(f"min_force should be less than max_force, got: {self.min_force} and {self.max_force}")
        if self.resolution is not None and not (isinstance(self.resolution, float) or len(self.resolution) == 3):
            gs.raise_exception(f"resolution must be a float or array-like of 3 floats, got: {self.resolution}")


class KinematicContactProbe(RigidSensorOptionsMixin, NoisySensorOptionsMixin, SensorOptions):
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
    probe_local_normal : array-like[array-like[float, float, float]]
        Probe sensing directions in link-local frame. Penetration is measured along this axis.
    radius : float | array-like[float]
        Probe sensing radius in meters. Objects within this distance are detected. Default: 0.005 (5mm)
    stiffness : float
        User-defined coefficient for force estimation. Default: 1000.0.
    """

    probe_local_pos: Vec3FArrayType = [(0.0, 0.0, 0.0)]
    probe_local_normal: Vec3FArrayType = [(0.0, 0.0, 1.0)]
    radius: float | FArrayType = 0.005
    stiffness: float = 1000.0

    debug_sphere_color: Vec4FType = (1.0, 0.5, 0.0, 0.4)
    debug_contact_color: Vec4FType = (1.0, 0.2, 0.0, 0.8)

    def model_post_init(self, _):
        if np.any(np.array(self.radius) < 0):
            gs.raise_exception(f"radius must be non-negative, got: {self.radius}")
        if self.stiffness < 0:
            gs.raise_exception(f"stiffness must be non-negative, got: {self.stiffness}")

        probe_local_pos = self._validate_probe_arrays(self.probe_local_pos)
        probe_local_normal = self._validate_probe_arrays(self.probe_local_normal)
        norms = np.linalg.norm(probe_local_normal, axis=1)
        if np.any(norms < gs.EPS):
            gs.raise_exception(f"probe_local_normal must be non-zero vectors, got: {probe_local_normal}")

        if len(probe_local_pos) != len(probe_local_normal):
            gs.raise_exception(
                "probe_local_pos and probe_local_normal must have the same length. "
                f"Got {len(probe_local_pos)} positions and {len(probe_local_normal)} normals."
            )
        if not isinstance(self.radius, float) and len(self.radius) != len(probe_local_pos):
            gs.raise_exception(
                "If radius is array-like, it must have the same length as probe_local_pos. "
                f"Got {len(self.radius)} radii and {len(probe_local_pos)} probe positions."
            )

    def _validate_probe_arrays(self, values: Vec3FArrayType) -> np.ndarray:
        array = np.array(values, dtype=float)
        if array.ndim != 2 or array.shape[1] != 3:
            gs.raise_exception(f"Probe locals array must have shape (N, 3), got: {array.shape}")
        if array.shape[0] == 0:
            gs.raise_exception("Probe locals array must have at least one entry")
        return array


class IMU(RigidSensorOptionsMixin, NoisySensorOptionsMixin, SensorOptions):
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
    acc_resolution: MaybeVec3FType = 0.0
    acc_cross_axis_coupling: MaybeMatrix3x3Type = 0.0
    acc_noise: MaybeVec3FType = 0.0
    acc_bias: MaybeVec3FType = 0.0
    acc_random_walk: MaybeVec3FType = 0.0

    # Gyroscope
    gyro_resolution: MaybeVec3FType = 0.0
    gyro_cross_axis_coupling: MaybeMatrix3x3Type = 0.0
    gyro_noise: MaybeVec3FType = 0.0
    gyro_bias: MaybeVec3FType = 0.0
    gyro_random_walk: MaybeVec3FType = 0.0

    # Magnetometer (New)
    mag_resolution: MaybeVec3FType = 0.0
    mag_cross_axis_coupling: MaybeMatrix3x3Type = 0.0
    mag_noise: MaybeVec3FType = 0.0
    mag_bias: MaybeVec3FType = 0.0
    mag_random_walk: MaybeVec3FType = 0.0
    magnetic_field: MaybeVec3FType = (0.0, 0.0, 0.5)

    debug_acc_color: Vec4FType = (1.0, 0.0, 0.0, 0.6)
    debug_acc_scale: float = 0.01
    debug_gyro_color: Vec4FType = (0.0, 1.0, 0.0, 0.6)
    debug_gyro_scale: float = 0.01
    debug_mag_color: Vec4FType = (0.0, 0.0, 1.0, 0.6)
    debug_mag_scale: float = 0.5

    def model_post_init(self, _):
        self._validate_cross_axis_coupling(self.acc_cross_axis_coupling)
        self._validate_cross_axis_coupling(self.gyro_cross_axis_coupling)
        self._validate_cross_axis_coupling(self.mag_cross_axis_coupling)

    def _validate_cross_axis_coupling(self, cross_axis_coupling):
        cross_axis_coupling_np = np.array(cross_axis_coupling)
        if cross_axis_coupling_np.shape not in ((), (3,), (3, 3)):
            gs.raise_exception(
                f"cross_axis_coupling shape should be (), (3,), or (3, 3), got: {cross_axis_coupling_np.shape}"
            )
        if np.any(cross_axis_coupling_np < 0.0) or np.any(cross_axis_coupling_np > 1.0):
            gs.raise_exception(f"cross_axis_coupling values should be between 0.0 and 1.0, got: {cross_axis_coupling}")


class Raycaster(RigidSensorOptionsMixin, SensorOptions):
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
    min_range: float = 0.0
    max_range: float = 20.0
    no_hit_value: float = Field(default_factory=lambda data: data["max_range"])
    return_world_frame: bool = False

    debug_sphere_radius: float = 0.02
    debug_ray_start_color: Vec4FType = (0.5, 0.5, 1.0, 1.0)
    debug_ray_hit_color: Vec4FType = (1.0, 0.5, 0.5, 1.0)

    def model_post_init(self, _):
        if self.min_range < 0.0:
            gs.raise_exception(f"[{type(self).__name__}] min_range should be non-negative. Got: {self.min_range}.")
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
