from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Type

import gstaichi as ti
import numpy as np
import torch

import genesis as gs
from genesis.options.sensors import IMU as IMUOptions
from genesis.options.sensors import MaybeMatrix3x3Type
from genesis.utils.geom import (
    inv_transform_by_quat,
    transform_by_quat,
    transform_quat_by_quat,
)
from genesis.utils.misc import concat_with_tensor, make_tensor_field, tensor_to_array

from .base_sensor import (
    NoisySensorMetadataMixin,
    NoisySensorMixin,
    RigidSensorMetadataMixin,
    RigidSensorMixin,
    Sensor,
    SharedSensorMetadata,
    _to_tuple,
)
from .sensor_manager import register_sensor

if TYPE_CHECKING:
    from genesis.ext.pyrender.mesh import Mesh
    from genesis.utils.ring_buffer import TensorRingBuffer
    from genesis.vis.rasterizer_context import RasterizerContext


def _get_cross_axis_coupling_to_alignment_matrix(
    input: MaybeMatrix3x3Type, out: torch.Tensor | None = None
) -> torch.Tensor:
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


@dataclass
class IMUSharedMetadata(RigidSensorMetadataMixin, NoisySensorMetadataMixin, SharedSensorMetadata):
    """
    Shared metadata between all IMU sensors.
    """

    alignment_rot_matrix: torch.Tensor = make_tensor_field((0, 0, 3, 3))
    acc_indices: torch.Tensor = make_tensor_field((0, 0), dtype_factory=lambda: gs.tc_int)
    gyro_indices: torch.Tensor = make_tensor_field((0, 0), dtype_factory=lambda: gs.tc_int)


class IMUData(NamedTuple):
    lin_acc: torch.Tensor
    ang_vel: torch.Tensor


@register_sensor(IMUOptions, IMUSharedMetadata, IMUData)
@ti.data_oriented
class IMUSensor(
    RigidSensorMixin[IMUSharedMetadata],
    NoisySensorMixin[IMUSharedMetadata],
    Sensor[IMUSharedMetadata],
):
    def __init__(
        self,
        options: IMUOptions,
        shared_metadata: IMUSharedMetadata,
        data_cls: Type[IMUData],
        manager: "gs.SensorManager",
    ):
        super().__init__(options, shared_metadata, data_cls, manager)

        self.debug_objects: list["Mesh"] = []
        self.quat_offset: torch.Tensor
        self.pos_offset: torch.Tensor

    @gs.assert_built
    def set_acc_cross_axis_coupling(self, cross_axis_coupling: MaybeMatrix3x3Type, envs_idx=None):
        envs_idx = self._sanitize_envs_idx(envs_idx)
        rot_matrix = _get_cross_axis_coupling_to_alignment_matrix(cross_axis_coupling)
        self._shared_metadata.alignment_rot_matrix[envs_idx, self._idx * 2, :, :] = rot_matrix

    @gs.assert_built
    def set_gyro_cross_axis_coupling(self, cross_axis_coupling: MaybeMatrix3x3Type, envs_idx=None):
        envs_idx = self._sanitize_envs_idx(envs_idx)
        rot_matrix = _get_cross_axis_coupling_to_alignment_matrix(cross_axis_coupling)
        self._shared_metadata.alignment_rot_matrix[envs_idx, self._idx * 2 + 1, :, :] = rot_matrix

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

        self._shared_metadata.alignment_rot_matrix = concat_with_tensor(
            self._shared_metadata.alignment_rot_matrix,
            torch.stack(
                [
                    _get_cross_axis_coupling_to_alignment_matrix(self._options.acc_cross_axis_coupling),
                    _get_cross_axis_coupling_to_alignment_matrix(self._options.gyro_cross_axis_coupling),
                ],
            ),
            expand=(self._manager._sim._B, 2, 3, 3),
            dim=1,
        )
        if self._options.draw_debug:
            self.quat_offset = self._shared_metadata.offsets_quat[0, self._idx]
            self.pos_offset = self._shared_metadata.offsets_pos[0, self._idx]

    def _get_return_format(self) -> tuple[tuple[int, ...], ...]:
        return (3,), (3,)

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
        # Extract acceleration and gravity in world frame
        assert shared_metadata.solver is not None
        gravity = shared_metadata.solver.get_gravity()
        quats = shared_metadata.solver.get_links_quat(links_idx=shared_metadata.links_idx)
        acc = shared_metadata.solver.get_links_acc(links_idx=shared_metadata.links_idx)
        ang = shared_metadata.solver.get_links_ang(links_idx=shared_metadata.links_idx)
        if acc.ndim == 2:  # n_envs = 0
            acc = acc[None]
            ang = ang[None]

        offset_quats = transform_quat_by_quat(quats, shared_metadata.offsets_quat)

        # Additional acceleration if offset: a_imu = a_link + α × r + ω × (ω × r)
        if torch.any(torch.abs(shared_metadata.offsets_pos) > gs.EPS):
            ang_acc = shared_metadata.solver.get_links_acc_ang(links_idx=shared_metadata.links_idx)
            if ang_acc.ndim == 2:  # n_envs = 0
                ang_acc = ang_acc[None]
            offset_pos_world = transform_by_quat(shared_metadata.offsets_pos, quats)
            tangential_acc = torch.cross(ang_acc, offset_pos_world, dim=-1)
            centripetal_acc = torch.cross(ang, torch.cross(ang, offset_pos_world, dim=-1), dim=-1)
            acc += tangential_acc + centripetal_acc

        # Subtract gravity then move to local frame
        # acc/ang shape: (B, n_imus, 3)
        local_acc = inv_transform_by_quat(acc - gravity[..., None, :], offset_quats)
        local_ang = inv_transform_by_quat(ang, offset_quats)

        # cache shape: (B, n_imus * 6)
        *batch_size, n_imus, _ = local_acc.shape
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

    def _draw_debug(self, context: "RasterizerContext", buffer_updates: dict[str, np.ndarray]):
        """
        Draw debug arrow for the IMU acceleration.

        Only draws for first rendered environment.
        """
        env_idx = context.rendered_envs_idx[0] if self._manager._sim.n_envs > 0 else None

        quat = self._link.get_quat(env_idx).reshape((4,))
        pos = self._link.get_pos(env_idx).reshape((3,)) + transform_by_quat(self.pos_offset, quat)

        # cannot specify envs_idx for read() when n_envs=0
        data = self.read(env_idx)
        acc_vec = data.lin_acc.reshape((3,)) * self._options.debug_acc_scale
        gyro_vec = data.ang_vel.reshape((3,)) * self._options.debug_gyro_scale

        # transform from local frame to world frame
        offset_quat = transform_quat_by_quat(self.quat_offset, quat)
        acc_vec = tensor_to_array(transform_by_quat(acc_vec, offset_quat))
        gyro_vec = tensor_to_array(transform_by_quat(gyro_vec, offset_quat))

        for debug_object in self.debug_objects:
            context.clear_debug_object(debug_object)
        self.debug_objects.clear()

        self.debug_objects += filter(
            None,
            (
                context.draw_debug_arrow(pos=pos, vec=acc_vec, color=self._options.debug_acc_color),
                context.draw_debug_arrow(pos=pos, vec=gyro_vec, color=self._options.debug_gyro_color),
            ),
        )
