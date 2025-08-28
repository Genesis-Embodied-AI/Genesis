from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

import gstaichi as ti
import numpy as np
import torch

import genesis as gs
from genesis.utils.geom import (
    inv_transform_by_trans_quat,
    transform_quat_by_quat,
)

from .base_sensor import (
    AnalogSensorBase,
    AnalogSensorMetadataBase,
    AnalogSensorOptionsBase,
)
from .sensor_manager import register_sensor

if TYPE_CHECKING:
    from genesis.utils.ring_buffer import TensorRingBuffer


class IMUOptions(AnalogSensorOptionsBase):
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
        The measurement resolution of the accelerometer. Default is 1e-6.
    acc_axes_skew : float | tuple[float, float, float] | Iterable[float]
        Accelerometer axes alignment as a 3x3 rotation matrix, where diagonal elements represent alignment (0.0 to 1.0)
        for each axis, and off-diagonal elements account for cross-axis misalignment effects.
        - If a scalar is provided (float), all off-diagonal elements are set to the scalar value.
        - If a 3-element vector is provided (tuple[float, float, float]), off-diagonal elements are set.
    acc_noise_std : tuple[float, float, float]
        The standard deviation of the white noise for each axis of the accelerometer.
    acc_bias : tuple[float, float, float]
        The additive bias for each axis of the accelerometer.
    acc_bias_drift_std : tuple[float, float, float]
        The standard deviation of the bias drift for each axis of the accelerometer.
    gyro_resolution : float, optional
        The measurement resolution of the gyroscope. Default is 1e-5.
    gyro_axes_skew : float | tuple[float, float, float] | Iterable[float]
        Gyroscope axes alignment as a 3x3 rotation matrix, similar to `acc_axes_skew`.
    gyro_noise_std : tuple[float, float, float]
        The standard deviation of the white noise for each axis of the gyroscope.
    gyro_bias : tuple[float, float, float]
        The additive bias for each axis of the gyroscope.
    gyro_bias_drift_std : tuple[float, float, float]
        The standard deviation of the bias drift for each axis of the gyroscope.
    delay : float, optional
        The delay in seconds before the sensor data is read.
    jitter : float, optional
        The time jitter standard deviation in seconds before the sensor data is read.
    interpolate_for_delay : bool, optional
        If True, the sensor data is interpolated between data points for delay + jitter.
        Otherwise, the sensor data at the closest time step will be used. Default is False.
    update_ground_truth_only : bool, optional
        If True, the sensor will only update the ground truth cache, and not the measured cache.
    """

    acc_resolution: float = 1e-6
    gyro_resolution: float = 1e-5
    acc_axes_skew: float | tuple[float, float, float] | Iterable[float] = 0.0
    gyro_axes_skew: float | tuple[float, float, float] | Iterable[float] = 0.0
    acc_noise_std: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gyro_noise_std: tuple[float, float, float] = (0.0, 0.0, 0.0)
    acc_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gyro_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)
    acc_bias_drift_std: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gyro_bias_drift_std: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def validate(self, scene):
        super().validate(scene)
        self._validate_axes_skew(self.acc_axes_skew)
        self._validate_axes_skew(self.gyro_axes_skew)

    def _validate_axes_skew(self, axes_skew):
        np_axes_skew = np.array(axes_skew)
        assert np_axes_skew.shape in [(), (3,), (3, 3)], "Invalid input shape for axes alignment."
        assert np.all(np_axes_skew >= 0.0) and np.all(
            np_axes_skew <= 1.0
        ), "Values for axes alignment matrix should be between 0.0 and 1.0."


@dataclass
class IMUSharedMetadata(AnalogSensorMetadataBase):
    """
    Shared metadata between all IMU sensors.
    """

    alignment_rot_matrix: torch.Tensor = field(
        default_factory=lambda: torch.tensor([], dtype=gs.tc_float, device=gs.device)
    )
    acc_indices: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=gs.tc_float, device=gs.device))
    gyro_indices: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=gs.tc_float, device=gs.device))


@register_sensor(IMUOptions, IMUSharedMetadata)
@ti.data_oriented
class IMUSensor(AnalogSensorBase):
    @gs.assert_built
    def set_acc_axes_skew(self, axes_skew, envs_idx=None):
        envs_idx = self._sanitize_envs_idx(envs_idx)
        rot_matrix = self._get_skew_to_alignment_matrix(axes_skew)
        self._shared_metadata.alignment_rot_matrix[envs_idx, self._sensor_idx * 2, :, :] = rot_matrix

    @gs.assert_built
    def set_gyro_axes_skew(self, axes_skew, envs_idx=None):
        envs_idx = self._sanitize_envs_idx(envs_idx)
        rot_matrix = self._get_skew_to_alignment_matrix(axes_skew)
        self._shared_metadata.alignment_rot_matrix[envs_idx, self._sensor_idx * 2 + 1, :, :] = rot_matrix

    @gs.assert_built
    def set_acc_bias(self, bias, envs_idx=None):
        self._set_metadata_tensor(bias, self._shared_metadata.acc_bias, envs_idx, 3)

    @gs.assert_built
    def set_gyro_bias(self, bias, envs_idx=None):
        self._set_metadata_tensor(bias, self._shared_metadata.gyro_bias, envs_idx, 3)

    @gs.assert_built
    def set_acc_bias_drift_std(self, bias_drift_std, envs_idx=None):
        self._set_metadata_tensor(bias_drift_std, self._shared_metadata.acc_bias_drift_std, envs_idx, 3)

    @gs.assert_built
    def set_gyro_bias_drift_std(self, bias_drift_std, envs_idx=None):
        self._set_metadata_tensor(bias_drift_std, self._shared_metadata.gyro_bias_drift_std, envs_idx, 3)

    @gs.assert_built
    def set_acc_noise_std(self, noise_std, envs_idx=None):
        self._set_metadata_tensor(noise_std, self._shared_metadata.acc_noise_std, envs_idx, 3)

    @gs.assert_built
    def set_gyro_noise_std(self, noise_std, envs_idx=None):
        self._set_metadata_tensor(noise_std, self._shared_metadata.gyro_noise_std, envs_idx, 3)

    # ================================ internal methods ================================

    def build(self):
        """
        Initialize all shared metadata needed to update all IMU sensors.
        """
        self._options.resolution = (self._options.acc_resolution, self._options.gyro_resolution)
        self._options.bias = tuple(self._options.acc_bias) + tuple(self._options.gyro_bias)
        self._options.bias_drift_std = tuple(self._options.acc_bias_drift_std) + tuple(
            self._options.gyro_bias_drift_std
        )
        self._options.noise_std = tuple(self._options.acc_noise_std) + tuple(self._options.gyro_noise_std)
        super().build()  # set all shared metadata from RigidSensorBase and AnalogSensorBase

        self._shared_metadata.acc_bias, self._shared_metadata.gyro_bias = self._view_metadata_as_acc_gyro(
            self._shared_metadata.bias
        )
        self._shared_metadata.acc_bias_drift_std, self._shared_metadata.gyro_bias_drift_std = (
            self._view_metadata_as_acc_gyro(self._shared_metadata.bias_drift_std)
        )
        self._shared_metadata.acc_noise_std, self._shared_metadata.gyro_noise_std = self._view_metadata_as_acc_gyro(
            self._shared_metadata.noise_std
        )
        self._shared_metadata.alignment_rot_matrix = torch.cat(
            [
                self._shared_metadata.alignment_rot_matrix,
                torch.stack(
                    [
                        self._get_skew_to_alignment_matrix(self._options.acc_axes_skew),
                        self._get_skew_to_alignment_matrix(self._options.gyro_axes_skew),
                    ],
                ).expand(self._manager._sim._B, -1, -1, -1),
            ],
            dim=1,
        )

    def get_return_format(self) -> dict[str, tuple[int, ...]]:
        return {
            "lin_acc": (3,),
            "ang_vel": (3,),
        }

    def get_cache_length(self) -> int:
        return 1

    @classmethod
    def update_shared_ground_truth_cache(
        cls, shared_metadata: IMUSharedMetadata, shared_ground_truth_cache: torch.Tensor
    ):
        """
        Update the current ground truth values for all IMU sensors.
        """
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
    def update_shared_cache(
        cls,
        shared_metadata: dict[str, Any],
        shared_ground_truth_cache: torch.Tensor,
        shared_cache: torch.Tensor,
        buffered_data: "TensorRingBuffer",
    ):
        """
        Update the current measured sensor data for all IMU sensors.
        """
        buffered_data.append(shared_ground_truth_cache)
        torch.normal(0, shared_metadata.jitter_std_in_steps, out=shared_metadata.jitter_in_steps)
        cls._apply_delay_to_shared_cache(
            shared_metadata,
            shared_cache,
            buffered_data,
            shared_metadata.jitter_in_steps,
            shared_metadata.interpolate_for_delay,
        )
        # apply rotation matrix to the shared cache
        shared_cache_xyz_view = shared_cache.view(shared_cache.shape[0], -1, 3)
        shared_cache_xyz_view.copy_(
            torch.matmul(shared_metadata.alignment_rot_matrix, shared_cache_xyz_view.unsqueeze(-1)).squeeze(-1)
        )
        # apply additive noise and bias to the shared cache
        cls._add_noise_drift_bias(shared_metadata, shared_cache)
        cls._quantize_to_resolution(shared_metadata, shared_cache_xyz_view.permute(0, 2, 1))

    @classmethod
    def get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    # ================================ helper methods ================================

    def _view_metadata_as_acc_gyro(self, metadata_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get views of the metadata tensor (B, n_imus * 6) as a tuple of acc and gyro metadata tensors (B, n_imus * 3).
        """
        batch_size, n_data = metadata_tensor.shape if metadata_tensor.ndim == 2 else (1, metadata_tensor.shape[-1])
        n_imus = n_data // 6
        reshaped_tensor = metadata_tensor.reshape(batch_size, n_imus, 2, 3)
        return (
            reshaped_tensor[..., 0, :].reshape(batch_size, n_imus * 3),
            reshaped_tensor[..., 1, :].reshape(batch_size, n_imus * 3),
        )

    def _get_skew_to_alignment_matrix(
        self, input: float | tuple[float, float, float] | Iterable[float], matrix: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Convert the alignment input to a matrix. Modifies in place if provided, else allocate a new matrix.
        """
        if matrix is None:
            matrix = torch.eye(3, dtype=gs.tc_float, device=gs.device)

        if isinstance(input, float):
            # set off-diagonal elements to the scalar value
            matrix[~torch.eye(3, dtype=gs.tc_bool, device=gs.device)] = input
        elif isinstance(input, torch.Tensor):
            matrix.copy_(input)
        else:
            np_input = np.array(input)
            if np_input.shape == (3,):
                # set off-diagonal elements to the vector values
                matrix[1, 0] = np_input[0]
                matrix[2, 0] = np_input[0]
                matrix[0, 1] = np_input[1]
                matrix[2, 1] = np_input[1]
                matrix[0, 2] = np_input[2]
                matrix[1, 2] = np_input[2]
            elif np_input.shape == (3, 3):
                matrix.copy_(torch.tensor(np_input, dtype=gs.tc_float, device=gs.device))
        return matrix
