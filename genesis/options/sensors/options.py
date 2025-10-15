from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from pydantic import Field

import genesis as gs

from ..options import Options
from .raycaster import RaycastPattern, DepthCameraPattern


Tuple3FType = tuple[float, float, float]
MaybeTuple3FType = float | Tuple3FType
Matrix3x3Type = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
MaybeMatrix3x3Type = Matrix3x3Type | MaybeTuple3FType


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
        The global entity index of the RigidEntity to which this sensor is attached.
    link_idx_local : int, optional
        The local index of the RigidLink of the RigidEntity to which this sensor is attached.
    pos_offset : tuple[float, float, float]
        The positional offset of the sensor from the RigidLink.
    euler_offset : tuple[float, float, float]
        The rotational offset of the sensor from the RigidLink in degrees.
    """

    entity_idx: int
    link_idx_local: int = 0
    pos_offset: Tuple3FType = (0.0, 0.0, 0.0)
    euler_offset: Tuple3FType = (0.0, 0.0, 0.0)

    def validate(self, scene: "Scene"):
        from genesis.engine.entities import RigidEntity

        super().validate(scene)
        if self.entity_idx < 0 or self.entity_idx >= len(scene.entities):
            gs.raise_exception(f"Invalid RigidEntity index {self.entity_idx}.")
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
    resolution : float | tuple[float, ...], optional
        The measurement resolution of the sensor (smallest increment of change in the sensor reading).
        Default is 0.0, which means no quantization is applied.
    bias : float | tuple[float, ...], optional
        The constant additive bias of the sensor.
    noise : float | tuple[float, ...], optional
        The standard deviation of the additive white noise.
    random_walk : float | tuple[float, ...], optional
        The standard deviation of the random walk, which acts as accumulated bias drift.
    jitter : float, optional
        The jitter in seconds modeled as a a random additive delay sampled from a normal distribution.
        Jitter cannot be greater than delay. `interpolate` should be True when `jitter` is greater than 0.
    interpolate : bool, optional
        If True, the sensor data is interpolated between data points for delay + jitter.
        Otherwise, the sensor data at the closest time step will be used. Default is False.
    """

    resolution: float | tuple[float, ...] = 0.0
    bias: float | tuple[float, ...] = 0.0
    noise: float | tuple[float, ...] = 0.0
    random_walk: float | tuple[float, ...] = 0.0
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
    debug_color : float, optional
        The rgba color of the debug sphere. Defaults to (1.0, 0.0, 1.0, 0.5).
    """

    debug_sphere_radius: float = 0.05
    debug_color: tuple[float, float, float, float] = (1.0, 0.0, 1.0, 0.5)


class ContactForce(RigidSensorOptionsMixin, NoisySensorOptionsMixin, SensorOptions):
    """
    Sensor that returns the total contact force being applied to the associated RigidLink in its local frame.

    Parameters
    ----------
    min_force : float | tuple[float, float, float], optional
        The minimum detectable absolute force per each axis. Values below this will be treated as 0. Default is 0.
    max_force : float | tuple[float, float, float], optional
        The maximum output absolute force per each axis. Values above this will be clipped. Default is infinity.
    debug_color : float, optional
        The rgba color of the debug arrow. Defaults to (1.0, 0.0, 1.0, 0.5).
    debug_scale : float, optional
        The scale factor for the debug force arrow. Defaults to 0.01.
    """

    min_force: MaybeTuple3FType = 0.0
    max_force: MaybeTuple3FType = np.inf

    debug_color: tuple[float, float, float, float] = (1.0, 0.0, 1.0, 0.5)
    debug_scale: float = 0.01

    def model_post_init(self, _):
        if not (
            isinstance(self.min_force, float) or (isinstance(self.min_force, Sequence) and len(self.min_force) == 3)
        ):
            gs.raise_exception(f"min_force must be a float or tuple of 3 floats, got: {self.min_force}")
        if not (
            isinstance(self.max_force, float) or (isinstance(self.max_force, Sequence) and len(self.max_force) == 3)
        ):
            gs.raise_exception(f"max_force must be a float or tuple of 3 floats, got: {self.max_force}")
        if np.any(np.array(self.min_force) < 0):
            gs.raise_exception(f"min_force must be non-negative, got: {self.min_force}")
        if np.any(np.array(self.max_force) <= np.array(self.min_force)):
            gs.raise_exception(f"min_force should be less than max_force, got: {self.min_force} and {self.max_force}")
        if self.resolution is not None and not (
            isinstance(self.resolution, float) or (isinstance(self.resolution, Sequence) and len(self.resolution) == 3)
        ):
            gs.raise_exception(f"resolution must be a float or tuple of 3 floats, got: {self.resolution}")


class IMU(RigidSensorOptionsMixin, NoisySensorOptionsMixin, SensorOptions):
    """
    IMU sensor returns the linear acceleration (accelerometer) and angular velocity (gyroscope)
    of the associated entity link.

    Parameters
    ----------
    acc_resolution : float, optional
        The measurement resolution of the accelerometer (smallest increment of change in the sensor reading).
        Default is 0.0, which means no quantization is applied.
    acc_axes_skew : float | tuple[float, float, float] | Sequence[float]
        Accelerometer axes alignment as a 3x3 rotation matrix, where diagonal elements represent alignment (0.0 to 1.0)
        for each axis, and off-diagonal elements account for cross-axis misalignment effects.
        - If a scalar is provided (float), all off-diagonal elements are set to the scalar value.
        - If a 3-element vector is provided (tuple[float, float, float]), off-diagonal elements are set.
        - If a full 3x3 matrix is provided, it is used directly.
    acc_bias : tuple[float, float, float]
        The constant additive bias for each axis of the accelerometer.
    acc_noise : tuple[float, float, float]
        The standard deviation of the white noise for each axis of the accelerometer.
    acc_random_walk : tuple[float, float, float]
        The standard deviation of the random walk, which acts as accumulated bias drift.
    gyro_resolution : float, optional
        The measurement resolution of the gyroscope (smallest increment of change in the sensor reading).
        Default is 0.0, which means no quantization is applied.
    gyro_axes_skew : float | tuple[float, float, float] | Sequence[float]
        Gyroscope axes alignment as a 3x3 rotation matrix, similar to `acc_axes_skew`.
    gyro_bias : tuple[float, float, float]
        The constant additive bias for each axis of the gyroscope.
    gyro_noise : tuple[float, float, float]
        The standard deviation of the white noise for each axis of the gyroscope.
    gyro_random_walk : tuple[float, float, float]
        The standard deviation of the bias drift for each axis of the gyroscope.
    debug_acc_color : float, optional
        The rgba color of the debug acceleration arrow. Defaults to (0.0, 1.0, 1.0, 0.5).
    debug_acc_scale: float, optional
        The scale factor for the debug acceleration arrow. Defaults to 0.01.
    debug_gyro_color : float, optional
        The rgba color of the debug gyroscope arrow. Defaults to (1.0, 1.0, 0.0, 0.5).
    debug_gyro_scale: float, optional
        The scale factor for the debug gyroscope arrow. Defaults to 0.01.
    """

    acc_resolution: MaybeTuple3FType = 0.0
    gyro_resolution: MaybeTuple3FType = 0.0
    acc_axes_skew: MaybeMatrix3x3Type = 0.0
    gyro_axes_skew: MaybeMatrix3x3Type = 0.0
    acc_noise: MaybeTuple3FType = 0.0
    gyro_noise: MaybeTuple3FType = 0.0
    acc_bias: MaybeTuple3FType = 0.0
    gyro_bias: MaybeTuple3FType = 0.0
    acc_random_walk: MaybeTuple3FType = 0.0
    gyro_random_walk: MaybeTuple3FType = 0.0

    debug_acc_color: tuple[float, float, float, float] = (0.0, 1.0, 1.0, 0.5)
    debug_acc_scale: float = 0.01
    debug_gyro_color: tuple[float, float, float, float] = (1.0, 1.0, 0.0, 0.5)
    debug_gyro_scale: float = 0.01

    def model_post_init(self, _):
        self._validate_axes_skew(self.acc_axes_skew)
        self._validate_axes_skew(self.gyro_axes_skew)

    def _validate_axes_skew(self, axes_skew):
        axes_skew_np = np.array(axes_skew)
        if axes_skew_np.shape not in ((), (3,), (3, 3)):
            gs.raise_exception(f"axes_skew shape should be (), (3,), or (3, 3), got: {axes_skew_np.shape}")
        if np.any(axes_skew_np < 0.0) or np.any(axes_skew_np > 1.0):
            gs.raise_exception(f"axes_skew values should be between 0.0 and 1.0, got: {axes_skew}")


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
    debug_ray_start_color: float, optional
        The color of each debug ray start sphere drawn in the scene. Defaults to (0.5, 0.5, 1.0, 1.0).
    debug_ray_hit_color: float, optional
        The color of each debug ray hit point sphere drawn in the scene. Defaults to (1.0, 0.5, 0.5, 1.0).
    """

    pattern: RaycastPattern
    min_range: float = 0.0
    max_range: float = 20.0
    no_hit_value: float = Field(default_factory=lambda data: data["max_range"])
    return_world_frame: bool = False

    debug_sphere_radius: float = 0.02
    debug_ray_start_color: tuple[float, float, float, float] = (0.5, 0.5, 1.0, 1.0)
    debug_ray_hit_color: tuple[float, float, float, float] = (1.0, 0.5, 0.5, 1.0)

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
