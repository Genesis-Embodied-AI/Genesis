from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import gstaichi as ti
import numpy as np
import torch

import genesis as gs
from genesis.utils.geom import (
    inv_transform_by_trans_quat,
    transform_quat_by_quat,
)
from genesis.utils.misc import concat_with_tensor, make_tensor_field

from .base_sensor import (
    MaybeTuple3FType,
    NoisySensorMetadataMixin,
    NoisySensorMixin,
    NoisySensorOptionsMixin,
    NumericType,
    RigidSensorMetadataMixin,
    RigidSensorMixin,
    RigidSensorOptionsMixin,
    Sensor,
    SensorOptions,
    SharedSensorMetadata,
    _to_tuple,
)
from .sensor_manager import register_sensor

if TYPE_CHECKING:
    from genesis.utils.ring_buffer import TensorRingBuffer

Matrix3x3Type = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
MaybeMatrix3x3Type = Matrix3x3Type | MaybeTuple3FType


def _view_metadata_as_acc_gyro(metadata_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get views of the metadata tensor (B, n_imus * 6) as a tuple of acc and gyro metadata tensors (B, n_imus * 3).
    """
    batch_shape, n_data = metadata_tensor.shape[:-1], metadata_tensor.shape[-1]
    n_imus = n_data // 6
    metadata_tensor_per_sensor = metadata_tensor.reshape((*batch_shape, n_imus, 2, 3))

    return (
        metadata_tensor_per_sensor[..., 0, :].reshape(*batch_shape, n_imus * 3),
        metadata_tensor_per_sensor[..., 1, :].reshape(*batch_shape, n_imus * 3),
    )


def _get_skew_to_alignment_matrix(input: MaybeMatrix3x3Type, out: torch.Tensor | None = None) -> torch.Tensor:
    """
    Convert the alignment input to a matrix. Modifies in place if provided, else allocate a new matrix.
    """
    if out is None:
        out = torch.eye(3, dtype=gs.tc_float, device=gs.device)

    if isinstance(input, float):
        # set off-diagonal elements to the scalar value
        torch.diagonal(out)[:] = input
        out.fill_diagonal_(1.0)
    elif isinstance(input, torch.Tensor):
        out.copy_(input)
    else:
        np_input = np.array(input)
        if np_input.shape == (3,):
            # set off-diagonal elements to the vector values
            out[1, 0] = np_input[0]
            out[2, 0] = np_input[0]
            out[0, 1] = np_input[1]
            out[2, 1] = np_input[1]
            out[0, 2] = np_input[2]
            out[1, 2] = np_input[2]
        elif np_input.shape == (3, 3):
            out.copy_(torch.tensor(np_input, dtype=gs.tc_float, device=gs.device))
    return out


class IMUOptions(RigidSensorOptionsMixin, NoisySensorOptionsMixin, SensorOptions):
    """
    IMU sensor returns the linear acceleration (accelerometer) and angular velocity (gyroscope)
    of the associated entity link.

    Parameters
    ----------
    entity_idx : int
        The global entity index of the RigidEntity to which this IMU sensor is attached.
    link_idx_local : int, optional
        The local index of the RigidLink of the RigidEntity to which this IMU sensor is attached.
    pos_offset : tuple[float, float, float], optional
        The positional offset of the IMU sensor from the RigidLink.
    euler_offset : tuple[float, float, float], optional
        The rotational offset of the IMU sensor from the RigidLink in degrees.
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
    delay : float, optional
        The delay in seconds, affecting how outdated the sensor data is when it is read.
    jitter : float, optional
        The jitter in seconds modeled as a a random additive delay sampled from a normal distribution.
        Jitter cannot be greater than delay. `interpolate` should be True when `jitter` is greater than 0.
    interpolate : bool, optional
        If True, the sensor data is interpolated between data points for delay + jitter.
        Otherwise, the sensor data at the closest time step will be used. Default is False.
    update_ground_truth_only : bool, optional
        If True, the sensor will only update the ground truth data, and not the measured data.
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

    def validate(self, scene):
        super().validate(scene)
        self._validate_axes_skew(self.acc_axes_skew)
        self._validate_axes_skew(self.gyro_axes_skew)

    def _validate_axes_skew(self, axes_skew):
        axes_skew_np = np.array(axes_skew)
        if axes_skew_np.shape not in ((), (3,), (3, 3)):
            gs.raise_exception(f"axes_skew shape should be (), (3,), or (3, 3), got: {axes_skew_np.shape}")
        if np.any(axes_skew_np < 0.0) or np.any(axes_skew_np > 1.0):
            gs.raise_exception(f"axes_skew values should be between 0.0 and 1.0, got: {axes_skew}")


@dataclass
class IMUSharedMetadata(RigidSensorMetadataMixin, NoisySensorMetadataMixin, SharedSensorMetadata):
    """
    Shared metadata between all IMU sensors.
    """

    alignment_rot_matrix: torch.Tensor = make_tensor_field((0, 0, 3, 3))
    acc_indices: torch.Tensor = make_tensor_field((0, 0), dtype_factory=lambda: gs.tc_int)
    gyro_indices: torch.Tensor = make_tensor_field((0, 0), dtype_factory=lambda: gs.tc_int)


@register_sensor(IMUOptions, IMUSharedMetadata)
@ti.data_oriented
class IMUSensor(
    RigidSensorMixin[IMUSharedMetadata],
    NoisySensorMixin[IMUSharedMetadata],
    Sensor[IMUSharedMetadata],
):
    @gs.assert_built
    def set_acc_axes_skew(self, axes_skew: MaybeMatrix3x3Type, envs_idx=None):
        envs_idx = self._sanitize_envs_idx(envs_idx)
        rot_matrix = _get_skew_to_alignment_matrix(axes_skew)
        self._shared_metadata.alignment_rot_matrix[envs_idx, self._idx * 2, :, :] = rot_matrix

    @gs.assert_built
    def set_gyro_axes_skew(self, axes_skew: MaybeMatrix3x3Type, envs_idx=None):
        envs_idx = self._sanitize_envs_idx(envs_idx)
        rot_matrix = _get_skew_to_alignment_matrix(axes_skew)
        self._shared_metadata.alignment_rot_matrix[envs_idx, self._idx * 2 + 1, :, :] = rot_matrix

    @gs.assert_built
    def set_acc_bias(self, bias, envs_idx=None):
        self._set_metadata_field(bias, self._shared_metadata.acc_bias, field_size=3, envs_idx=envs_idx)

    @gs.assert_built
    def set_gyro_bias(self, bias, envs_idx=None):
        self._set_metadata_field(bias, self._shared_metadata.gyro_bias, field_size=3, envs_idx=envs_idx)

    @gs.assert_built
    def set_acc_random_walk(self, random_walk, envs_idx=None):
        self._set_metadata_field(random_walk, self._shared_metadata.acc_random_walk, field_size=3, envs_idx=envs_idx)

    @gs.assert_built
    def set_gyro_random_walk(self, random_walk, envs_idx=None):
        self._set_metadata_field(random_walk, self._shared_metadata.gyro_random_walk, field_size=3, envs_idx=envs_idx)

    @gs.assert_built
    def set_acc_noise(self, noise, envs_idx=None):
        self._set_metadata_field(noise, self._shared_metadata.acc_noise, field_size=3, envs_idx=envs_idx)

    @gs.assert_built
    def set_gyro_noise(self, noise, envs_idx=None):
        self._set_metadata_field(noise, self._shared_metadata.gyro_noise, field_size=3, envs_idx=envs_idx)

    # ================================ internal methods ================================

    def build(self):
        """
        Initialize all shared metadata needed to update all IMU sensors.
        """
        self._options.resolution = _to_tuple(
            self._options.acc_resolution, self._options.gyro_resolution, length_per_value=3
        )
        self._options.bias = _to_tuple(self._options.acc_bias, self._options.gyro_bias, length_per_value=3)
        self._options.random_walk = _to_tuple(
            self._options.acc_random_walk, self._options.gyro_random_walk, length_per_value=3
        )
        self._options.noise = _to_tuple(self._options.acc_noise, self._options.gyro_noise, length_per_value=3)
        super().build()  # set all shared metadata from RigidSensorBase and NoisySensorBase

        self._shared_metadata.acc_bias, self._shared_metadata.gyro_bias = _view_metadata_as_acc_gyro(
            self._shared_metadata.bias
        )
        self._shared_metadata.acc_random_walk, self._shared_metadata.gyro_random_walk = _view_metadata_as_acc_gyro(
            self._shared_metadata.random_walk
        )
        self._shared_metadata.acc_noise, self._shared_metadata.gyro_noise = _view_metadata_as_acc_gyro(
            self._shared_metadata.noise
        )
        self._shared_metadata.alignment_rot_matrix = concat_with_tensor(
            self._shared_metadata.alignment_rot_matrix,
            torch.stack(
                [
                    _get_skew_to_alignment_matrix(self._options.acc_axes_skew),
                    _get_skew_to_alignment_matrix(self._options.gyro_axes_skew),
                ],
            ),
            expand=(self._manager._sim._B, 2, 3, 3),
            dim=1,
        )

    def _get_return_format(self) -> dict[str, tuple[int, ...]]:
        return {
            "lin_acc": (3,),
            "ang_vel": (3,),
        }

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def _update_shared_ground_truth_cache(
        cls, shared_metadata: IMUSharedMetadata, shared_ground_truth_cache: torch.Tensor
    ):
        """
        Update the current ground truth values for all IMU sensors.
        """
        assert shared_metadata.solver is not None
        gravity = shared_metadata.solver.get_gravity()
        quats = shared_metadata.solver.get_links_quat(links_idx=shared_metadata.links_idx)
        acc = shared_metadata.solver.get_links_acc(links_idx=shared_metadata.links_idx)
        ang = shared_metadata.solver.get_links_ang(links_idx=shared_metadata.links_idx)

        offset_quats = transform_quat_by_quat(quats, shared_metadata.offsets_quat)

        # acc/ang shape: (B, n_imus, 3)
        local_acc = inv_transform_by_trans_quat(acc, shared_metadata.offsets_pos, offset_quats)
        local_ang = inv_transform_by_trans_quat(ang, shared_metadata.offsets_pos, offset_quats)

        *batch_size, n_imus, _ = local_acc.shape
        local_acc = local_acc - gravity.unsqueeze(-2).expand((*batch_size, n_imus, -1))

        # cache shape: (B, n_imus * 6)
        strided_ground_truth_cache = shared_ground_truth_cache.reshape((*batch_size, n_imus, 2, 3))
        strided_ground_truth_cache[..., 0, :].copy_(local_acc)
        strided_ground_truth_cache[..., 1, :].copy_(local_ang)

    @classmethod
    def _update_shared_cache(
        cls,
        shared_metadata: IMUSharedMetadata,
        shared_ground_truth_cache: torch.Tensor,
        shared_cache: torch.Tensor,
        buffered_data: "TensorRingBuffer",
    ):
        """
        Update the current measured sensor data for all IMU sensors.
        """
        buffered_data.append(shared_ground_truth_cache)
        torch.normal(0.0, shared_metadata.jitter_ts, out=shared_metadata.cur_jitter_ts)
        cls._apply_delay_to_shared_cache(
            shared_metadata,
            shared_cache,
            buffered_data,
            shared_metadata.cur_jitter_ts,
            shared_metadata.interpolate,
        )
        # apply rotation matrix to the shared cache
        shared_cache_xyz_view = shared_cache.view(shared_cache.shape[0], -1, 3)
        shared_cache_xyz_view.copy_(
            torch.matmul(shared_metadata.alignment_rot_matrix, shared_cache_xyz_view.unsqueeze(-1)).squeeze(-1)
        )
        # apply additive noise and bias to the shared cache
        cls._add_noise_drift_bias(shared_metadata, shared_cache)
        cls._quantize_to_resolution(shared_metadata.resolution, shared_cache)
